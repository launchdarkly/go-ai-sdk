//go:build unix

package aiserver

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strconv"
	"time"
	"unicode/utf8"

	"golang.org/x/sys/unix"
)

// maxManifestBytes caps the manifest read; a larger file is corrupt.
const maxManifestBytes = 8 * 1024 * 1024

// maxPathComponentBytes is NAME_MAX on Linux and macOS: a longer key cannot
// become a directory name, so it is reported before any filesystem call.
const maxPathComponentBytes = 255

// windowsReservedNames cannot be directory names on Windows; rejected
// everywhere so a managed root written on one OS is usable on another.
var windowsReservedNames = func() map[string]bool {
	m := map[string]bool{"con": true, "prn": true, "aux": true, "nul": true}
	for d := 1; d <= 9; d++ {
		m["com"+strconv.Itoa(d)] = true
		m["lpt"+strconv.Itoa(d)] = true
	}
	return m
}()

func writeSkills(sel selection, root string, opts *WriteSkillsOptions) (ReconcileReport, error) {
	cfg, err := parseWriteOptions(opts)
	if err != nil {
		return ReconcileReport{}, err
	}
	rootPath, err := resolveRoot(root)
	if err != nil {
		return ReconcileReport{}, err
	}
	// Pinned once and held for the whole reconcile; everything below,
	// including the reads that decide what to do, is relative to it.
	rootFd, err := fsOps.openRoot(rootPath)
	if err != nil {
		return newReconcileReport([]ReconcileAction{runError(fmt.Sprintf(
			"the skills root %s could not be pinned for the reconcile: %v; no action was taken", rootPath, err))}), nil
	}
	defer closeFd(rootFd)

	manifest, manifestErr := loadManifest(rootFd)
	entries, _ := manifest["entries"].(map[string]any)
	var actions []ReconcileAction
	if manifestErr != "" {
		actions = append(actions, runError(manifestErr))
	}

	requests, incomplete, err := resolveRequests(sel, cfg)
	if err != nil {
		return ReconcileReport{}, err
	}
	written, timedOut := writeAll(rootPath, rootFd, requests, entries, cfg)
	actions = append(actions, written...)

	// Prune only when the run knows what it owns (manifest intact) and what is
	// current (retrieval and writes completed).
	if cfg.prune && manifestErr == "" && !incomplete && !timedOut {
		requested := map[string]bool{}
		for _, r := range requests {
			requested[r.key] = true
		}
		actions = append(actions, prune(rootPath, rootFd, entries, requested, cfg)...)
	}
	if manifestErr == "" {
		manifest["manifestVersion"] = ManifestVersion
		manifest["entries"] = entries
		if err := writeManifest(rootFd, manifest); err != nil {
			actions = append(actions, runError(fmt.Sprintf("the skills manifest %s could not be written: %v", ManifestFilename, err)))
		}
	}
	return newReconcileReport(actions), nil
}

// resolveRoot validates the root up front (caller errors), creating only the
// leaf directory. It is not the security boundary: the pin taken right after
// it is.
func resolveRoot(root string) (string, error) {
	if root == "" {
		return "", invalidArg("the skills root is empty")
	}
	st, err := os.Lstat(root)
	switch {
	case err == nil:
		if st.Mode()&os.ModeSymlink != 0 {
			return "", invalidArg("the skills root must be a real directory, not a symlink: %s", root)
		}
		if !st.IsDir() {
			return "", invalidArg("the skills root is not a directory: %s", root)
		}
	case errors.Is(err, os.ErrNotExist):
		parent := filepath.Dir(filepath.Clean(root))
		pst, perr := os.Stat(parent)
		if perr != nil || !pst.IsDir() {
			return "", invalidArg("the parent of the skills root does not exist: %s. WriteSkills creates only the leaf directory", parent)
		}
		if err := os.Mkdir(root, dirMode); err != nil {
			return "", invalidArg("the skills root could not be created: %v", err)
		}
	default:
		return "", invalidArg("the skills root could not be inspected: %v", err)
	}
	real, err := filepath.EvalSymlinks(root)
	if err == nil {
		real, err = filepath.Abs(real)
	}
	if err != nil {
		return "", invalidArg("the skills root could not be resolved: %v", err)
	}
	return real, nil
}

func freshManifest() map[string]any {
	return map[string]any{"manifestVersion": ManifestVersion, "entries": map[string]any{}}
}

// loadManifest returns the manifest, or a fresh one and an error message when
// it cannot be trusted. Any such message suppresses every destructive action
// and leaves the file in place.
func loadManifest(rootFd int) (map[string]any, string) {
	refuse := "; refusing every destructive action"
	data, exists, err := readRegularAt(rootFd, ManifestFilename, maxManifestBytes)
	if !exists {
		return freshManifest(), ""
	}
	if err != nil {
		return freshManifest(), fmt.Sprintf("the skills manifest %s could not be read: %v%s", ManifestFilename, err, refuse)
	}
	if len(data) > maxManifestBytes {
		return freshManifest(), fmt.Sprintf("the skills manifest %s is larger than the %d byte cap%s", ManifestFilename, maxManifestBytes, refuse)
	}
	if !utf8.Valid(data) {
		return freshManifest(), fmt.Sprintf("the skills manifest %s is not valid UTF-8%s", ManifestFilename, refuse)
	}
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber() // keep unknown numbers verbatim through the rewrite
	var parsed any
	err = dec.Decode(&parsed)
	if err == nil {
		if _, extra := dec.Token(); !errors.Is(extra, io.EOF) {
			err = errors.New("trailing data after the JSON value")
		}
	}
	if err != nil {
		return freshManifest(), fmt.Sprintf("the skills manifest %s is not valid JSON (%v)%s", ManifestFilename, err, refuse)
	}
	obj, ok := parsed.(map[string]any)
	if !ok {
		return freshManifest(), fmt.Sprintf("the skills manifest %s is not a JSON object%s", ManifestFilename, refuse)
	}
	// Bounded at both ends: a future schema cannot be read, and versions
	// below 1 were never written.
	num, isNum := obj["manifestVersion"].(json.Number)
	v, perr := strconv.ParseInt(string(num), 10, 64)
	if !isNum || perr != nil || v < 1 || v > ManifestVersion {
		return freshManifest(), fmt.Sprintf("the skills manifest %s declares manifestVersion %v, which this SDK cannot read%s",
			ManifestFilename, obj["manifestVersion"], refuse)
	}
	if _, ok := obj["entries"].(map[string]any); !ok {
		return freshManifest(), fmt.Sprintf("the skills manifest %s has a malformed 'entries' map%s", ManifestFilename, refuse)
	}
	return obj, ""
}

func writeManifest(rootFd int, manifest map[string]any) error {
	data, err := marshalSortedASCII(manifest, true)
	if err != nil {
		return err
	}
	return atomicWrite(rootFd, ManifestFilename, data)
}

func writeAll(rootPath string, rootFd int, requests []pendingWrite, entries map[string]any, cfg writeConfig) ([]ReconcileAction, bool) {
	var (
		actions  []ReconcileAction
		timedOut bool
	)
	for _, r := range requests {
		switch {
		case r.skill == nil:
			msg := r.err
			if msg == "" {
				msg = fmt.Sprintf("skill '%s' could not be resolved", safeKey(r.key))
			}
			actions = append(actions, ReconcileAction{key: r.key, action: ActionError, err: msg})
		case cfg.expired():
			timedOut = true
			actions = append(actions, ReconcileAction{key: r.key, action: ActionError, version: r.skill.version,
				err: fmt.Sprintf("the timeout was exhausted before skill '%s' could be written", safeKey(r.key))})
		default:
			actions = append(actions, writeOne(rootPath, rootFd, *r.skill, entries))
		}
	}
	return actions, timedOut
}

// keyRejection says why key must not become a directory name, or "". Always
// re-checked locally, whatever validated it upstream.
func keyRejection(key string) string {
	if r := skillKeyRejection(key); r != "" {
		return fmt.Sprintf("%q is not a valid skill key: it %s", key, r)
	}
	if len(key) > maxPathComponentBytes {
		return fmt.Sprintf("skill key '%s...' is %d bytes, over the %d-byte limit for a single directory name", key[:32], len(key), maxPathComponentBytes)
	}
	if windowsReservedNames[key] {
		return fmt.Sprintf("skill key '%s' is a name Windows reserves for a device; it is rejected on every platform so a managed root is portable", key)
	}
	return ""
}

// unsafeReason is the materialization layer's own check, shared by the write
// and prune paths. The primitives refuse the same things independently.
func unsafeReason(rootFd int, key string, requireDir bool) string {
	st, err := lstatAt(rootFd, key)
	if err != nil {
		return ""
	}
	if isSymlink(st) {
		return fmt.Sprintf("%s is a symlink", key)
	}
	if !isDir(st) {
		if requireDir {
			return fmt.Sprintf("%s exists and is not a directory", key)
		}
		return ""
	}
	if tst, err := lstatAt(rootFd, key+"/"+SkillFilename); err == nil && isSymlink(tst) {
		return "the target file is a symlink"
	}
	return ""
}

func writeOne(rootPath string, rootFd int, skill Skill, entries map[string]any) ReconcileAction {
	key := skill.key
	failed := func(format string, args ...any) ReconcileAction {
		return ReconcileAction{key: key, action: ActionError, version: max(skill.version, 0), err: fmt.Sprintf(format, args...)}
	}
	if r := keyRejection(key); r != "" {
		return failed("%s; nothing was written", r)
	}
	if skill.version < 1 {
		return failed("skill '%s' has version %d, which is not an integer >= 1; nothing was written", key, skill.version)
	}
	relative := key + "/" + SkillFilename
	target := filepath.Join(rootPath, key, SkillFilename)
	if r := unsafeReason(rootFd, key, true); r != "" {
		return failed("'%s' was refused: %s; nothing was written", relative, r)
	}
	// Re-verify immediately before writing: a Skill can be built directly.
	v, reason := verifyBytes(key, skill.content, false, skill.contentHash, skill.version)
	if reason != "" {
		return failed("skill '%s' failed verification immediately before writing: %s; nothing was written", key, reason)
	}
	entry, _ := entries[relative].(map[string]any)
	managed := entry != nil && entry["key"] == key

	// Pinned before deciding anything and held through the write, so the
	// sweep, the comparison read and the rename all see one directory.
	skillFd, err := openOrCreateDir(rootFd, key)
	if err != nil {
		if errors.Is(err, errSymlinkRefused) || errors.Is(err, errNotDirectory) {
			return failed("'%s' was refused: %v", relative, err)
		}
		return failed("the directory for skill '%s' could not be created: %v", key, err)
	}
	defer closeFd(skillFd)
	sweepOrphans(skillFd, key)

	onDisk, exists, err := readRegularAt(skillFd, SkillFilename, len(v.bytes))
	if err != nil {
		if !managed {
			// "We could not look" is fail-closed, and worded apart from the
			// collision refusal.
			return failed("'%s' exists, the manifest does not record it as managed under key '%s', and it could not "+
				"be read to compare against the resolved content: %v; refusing to overwrite a file this SDK may not have written", relative, key, err)
		}
		return failed("'%s' could not be read: %v", relative, err)
	}
	var action ReconcileActionKind
	switch {
	case !exists:
		action = ActionWritten
	case sha256Hex(onDisk) == v.hash:
		// Adoption (crash recovery): the bytes already are the content, so the
		// file is claimed, or re-keyed, without being rewritten.
		updateEntry(entries, relative, skill, v.hash, false)
		recordMaterialized(key, len(v.bytes), v.hash, ActionSkippedCurrent)
		return ReconcileAction{key: key, action: ActionSkippedCurrent, version: skill.version, path: target}
	case !managed:
		return failed("'%s' exists but the manifest does not record it as managed under key '%s'; refusing to overwrite a file this SDK did not write", relative, key)
	default:
		action = ActionUpdated // a new version or local tampering: LaunchDarkly wins
	}
	if err := atomicWrite(skillFd, SkillFilename, v.bytes); err != nil {
		return failed("'%s' could not be written: %v", relative, err)
	}
	updateEntry(entries, relative, skill, v.hash, true)
	recordMaterialized(key, len(v.bytes), v.hash, action)
	return ReconcileAction{key: key, action: action, version: skill.version, path: target}
}

func sha256Hex(b []byte) string {
	sum := sha256.Sum256(b)
	return hex.EncodeToString(sum[:])
}

// sweepOrphans removes temp files a killed reconcile left in the pinned skill
// directory, which would otherwise keep prune's rmdir from succeeding. Only
// strictly matching names, only regular files. Never fails the run.
func sweepOrphans(skillFd int, key string) {
	names, err := listDirAt(skillFd)
	if err != nil {
		currentLogger().Warn("orphaned temp files could not be swept", "skill_key", key, "error", err)
		return
	}
	for _, name := range names {
		if !isTempName(name) {
			continue
		}
		st, err := lstatAt(skillFd, name)
		if err != nil || st.Mode&unix.S_IFMT != unix.S_IFREG {
			continue
		}
		if err := unlinkAt(skillFd, name); err != nil {
			currentLogger().Warn("an orphaned temp file could not be removed", "skill_key", key, "error", err)
		}
	}
}

// updateEntry records a managed path, merging into the existing entry so a
// newer SDK's fields survive. touch is false when nothing was written: then
// writtenAt is only filled in if missing.
func updateEntry(entries map[string]any, relative string, skill Skill, hash string, touch bool) {
	entry := map[string]any{}
	if existing, ok := entries[relative].(map[string]any); ok {
		entry = maps.Clone(existing)
	}
	entry["key"] = skill.key
	entry["version"] = skill.version
	entry["sha256"] = hash
	if _, has := entry["writtenAt"]; touch || !has {
		entry["writtenAt"] = time.Now().UTC().Format("2006-01-02T15:04:05Z")
	}
	entries[relative] = entry
}

// prune removes managed skills no longer requested: how revocation takes
// effect. The budget is checked once per entry; an entry not reached stays in
// the manifest for the next run.
func prune(rootPath string, rootFd int, entries map[string]any, requested map[string]bool, cfg writeConfig) []ReconcileAction {
	var actions []ReconcileAction
	for _, relative := range slices.Sorted(maps.Keys(entries)) {
		entry, ok := entries[relative].(map[string]any)
		if !ok {
			continue
		}
		key, ok := entry["key"].(string)
		if !ok || requested[key] {
			continue
		}
		version, _ := versionOf(entry["version"])
		refuse := func(format string, args ...any) {
			actions = append(actions, ReconcileAction{key: key, action: ActionError, version: version, err: fmt.Sprintf(format, args...)})
		}
		if cfg.expired() {
			refuse("the timeout was exhausted before '%s' could be pruned; it was left in place", relative)
			continue
		}
		if keyRejection(key) != "" || relative != key+"/"+SkillFilename {
			refuse("manifest entry '%s' does not name a path this SDK could own under key '%s'; it was left in place", relative, safeKey(key))
			continue
		}
		actions = append(actions, pruneOne(rootPath, rootFd, relative, key, version, entries))
	}
	return actions
}

func pruneOne(rootPath string, rootFd int, relative, key string, version int, entries map[string]any) ReconcileAction {
	refuse := func(format string, args ...any) ReconcileAction {
		return ReconcileAction{key: key, action: ActionError, version: version, err: fmt.Sprintf(format, args...)}
	}
	if r := unsafeReason(rootFd, key, false); r != "" {
		return refuse("'%s' was not removed: %s", relative, r)
	}
	removed := false
	skillFd, err := openDirNoFollow(rootFd, key)
	switch {
	case errors.Is(err, errDirMissing):
		// Already gone.
	case err != nil:
		return refuse("'%s' was not removed: %v", relative, err)
	default:
		// Pinned before the existence probe and held through the unlink, so a
		// swapped directory cannot redirect the delete.
		sweepOrphans(skillFd, key)
		_, statErr := lstatAt(skillFd, SkillFilename)
		if statErr == nil {
			if err := unlinkAt(skillFd, SkillFilename); err != nil {
				closeFd(skillFd)
				if errors.Is(err, errSymlinkRefused) {
					return refuse("'%s' was not removed: the target file is a symlink", relative)
				}
				return refuse("'%s' could not be removed: %v", relative, err)
			}
			removed = true
		}
		closeFd(skillFd)
	}
	if removed {
		// rmdir refuses a symlink and a non-empty directory; either way the
		// directory is left alone.
		_ = rmdirAt(rootFd, key)
	}
	delete(entries, relative)
	if removed {
		recordRevoked(key, version)
	}
	return ReconcileAction{key: key, action: ActionRemoved, version: version, path: filepath.Join(rootPath, key, SkillFilename)}
}
