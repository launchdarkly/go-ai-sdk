//go:build unix

package skills

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"

	"golang.org/x/sys/unix"
)

// Descriptor-pinned filesystem primitives.
//
// The invariant: every operation runs relative to a descriptor pinned to a
// directory that was already verified, never against a re-resolved path, so a
// directory swapped for a symlink mid-run cannot redirect anything. Every
// POSIX platform Go supports has the *at() family, so the swap window is
// closed rather than narrowed and there is no lstat floor.

// supportsDirFD is the capability probe. Constant here: the build
// tag is the probe, since x/sys/unix exposes the *at() family on every unix
// GOOS.
const supportsDirFD = true

const (
	fileMode = 0o644
	dirMode  = 0o755
)

var (
	errDirMissing     = errors.New("the directory does not exist or is not a directory")
	errSymlinkRefused = errors.New("refusing to follow a symlink")
	errNotDirectory   = errors.New("the path exists and is not a directory")
	errNotRegular     = errors.New("the target file is not a regular file")
)

// fsOps is the single interceptable call site for the destructive operations
// and for the root pin. Tests replace fields and delegate to
// the real implementation.
var fsOps = struct {
	openRoot   func(path string) (int, error)
	createTemp func(dirFd int, name string, flags int, mode uint32) (int, error)
	rename     func(oldDirFd int, oldName string, newDirFd int, newName string) error
	unlink     func(dirFd int, name string) error
}{
	openRoot:   realOpenRoot,
	createTemp: unix.Openat,
	rename:     unix.Renameat,
	unlink:     func(dirFd int, name string) error { return unix.Unlinkat(dirFd, name, 0) },
}

func realOpenRoot(path string) (int, error) {
	fd, err := unix.Open(path, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return -1, err
	}
	if err := requireDirFd(fd); err != nil {
		_ = unix.Close(fd)
		return -1, err
	}
	return fd, nil
}

func requireDirFd(fd int) error {
	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		return err
	}
	if st.Mode&unix.S_IFMT != unix.S_IFDIR {
		return errNotDirectory
	}
	return nil
}

// lstatAt stats name inside dirFd without following a final symlink.
func lstatAt(dirFd int, name string) (unix.Stat_t, error) {
	var st unix.Stat_t
	err := unix.Fstatat(dirFd, name, &st, unix.AT_SYMLINK_NOFOLLOW)
	return st, err
}

func isSymlink(st unix.Stat_t) bool { return st.Mode&unix.S_IFMT == unix.S_IFLNK }
func isDir(st unix.Stat_t) bool     { return st.Mode&unix.S_IFMT == unix.S_IFDIR }

// openDirNoFollow opens <parentFd>/<name> as a directory without following a
// symlink and pins it. errDirMissing means nothing (or a non-directory) is
// there; a symlink is errSymlinkRefused, never "missing".
func openDirNoFollow(parentFd int, name string) (int, error) {
	fd, err := unix.Openat(parentFd, name, unix.O_RDONLY|unix.O_DIRECTORY|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err == nil {
		if err := requireDirFd(fd); err != nil {
			_ = unix.Close(fd)
			return -1, err
		}
		return fd, nil
	}
	switch {
	case errors.Is(err, unix.ENOENT):
		return -1, errDirMissing
	case errors.Is(err, unix.ELOOP), errors.Is(err, unix.ENOTDIR):
		// O_NOFOLLOW|O_DIRECTORY on a symlink is ELOOP or ENOTDIR depending on
		// the kernel; ask the entry itself so a link never reads as missing.
		if st, serr := lstatAt(parentFd, name); serr == nil && isSymlink(st) {
			return -1, errSymlinkRefused
		}
		if errors.Is(err, unix.ENOTDIR) {
			return -1, errDirMissing
		}
	}
	return -1, err
}

// openOrCreateDir creates <parentFd>/<name> if absent and pins it. A
// directory this call created gets 0755 through its descriptor, whatever the
// umask, keeping a setgid bit it inherited. An existing directory
// keeps its mode.
func openOrCreateDir(parentFd int, name string) (int, error) {
	created := false
	switch err := unix.Mkdirat(parentFd, name, dirMode); {
	case err == nil:
		created = true
	case errors.Is(err, unix.EEXIST):
		st, serr := lstatAt(parentFd, name)
		if serr != nil {
			return -1, serr
		}
		if isSymlink(st) {
			return -1, errSymlinkRefused
		}
		if !isDir(st) {
			return -1, errNotDirectory
		}
	default:
		return -1, err
	}
	fd, err := openDirNoFollow(parentFd, name)
	if err != nil {
		return -1, err
	}
	if created {
		var st unix.Stat_t
		err := unix.Fstat(fd, &st)
		if err == nil {
			err = unix.Fchmod(fd, uint32(st.Mode&unix.S_ISGID)|dirMode)
		}
		if err != nil {
			_ = unix.Close(fd)
			return -1, err
		}
	}
	return fd, nil
}

// readRegularAt is the one bounded read primitive shared by the manifest read
// and the adoption comparison read: no-follow, non-blocking
// (a FIFO with no writer never hangs it), the regular-file check from fstat on
// the descriptor, and at most maxBytes+1 bytes so an over-long file is
// detected without being read whole. exists is false only for ENOENT.
func readRegularAt(dirFd int, name string, maxBytes int) (data []byte, exists bool, err error) {
	fd, err := unix.Openat(dirFd, name, unix.O_RDONLY|unix.O_NOFOLLOW|unix.O_NONBLOCK|unix.O_CLOEXEC, 0)
	if err != nil {
		if errors.Is(err, unix.ENOENT) {
			return nil, false, nil
		}
		if errors.Is(err, unix.ELOOP) {
			return nil, true, errSymlinkRefused
		}
		return nil, true, err
	}
	defer unix.Close(fd)
	var st unix.Stat_t
	if err := unix.Fstat(fd, &st); err != nil {
		return nil, true, err
	}
	if st.Mode&unix.S_IFMT != unix.S_IFREG {
		return nil, true, errNotRegular
	}
	buf := make([]byte, 0, min(maxBytes+1, 64*1024))
	chunk := make([]byte, 64*1024)
	for remaining := maxBytes + 1; remaining > 0; {
		n, err := unix.Read(fd, chunk[:min(remaining, len(chunk))])
		if err != nil {
			if errors.Is(err, unix.EINTR) {
				continue
			}
			return nil, true, err
		}
		if n == 0 {
			break
		}
		buf = append(buf, chunk[:n]...)
		remaining -= n
	}
	return buf, true, nil
}

var tempNameRE = regexp.MustCompile(`^\.SKILL\.md\.[0-9a-f]{16}\.tmp$`)

// isTempName is deliberately strict: the orphan sweep deletes what it matches.
func isTempName(name string) bool { return tempNameRE.MatchString(name) }

func tempName(target string) (string, error) {
	var b [8]byte
	if _, err := io.ReadFull(rand.Reader, b[:]); err != nil {
		return "", err
	}
	return "." + target + "." + hex.EncodeToString(b[:]) + ".tmp", nil
}

// atomicWrite writes data to <dirFd>/<name> so no partial file is ever
// observable: a temp file created exclusively (O_EXCL, O_NOFOLLOW) in the
// target's own directory under an unpredictable name, chmod 0644 through its
// descriptor, written, fsynced, renamed over the target relative to the same
// descriptor, then the directory fsynced. A failed run leaves no temp file.
func atomicWrite(dirFd int, name string, data []byte) error {
	var (
		fd   = -1
		temp string
		err  error
	)
	for range 128 {
		if temp, err = tempName(name); err != nil {
			return err
		}
		fd, err = fsOps.createTemp(dirFd, temp, unix.O_WRONLY|unix.O_CREAT|unix.O_EXCL|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0o600)
		if !errors.Is(err, unix.EEXIST) {
			break
		}
	}
	if err != nil {
		return fmt.Errorf("could not create a temporary file: %w", err)
	}
	cleanup := func() { _ = unix.Unlinkat(dirFd, temp, 0) }
	werr := unix.Fchmod(fd, fileMode)
	for rest := data; werr == nil && len(rest) > 0; {
		var n int
		n, werr = unix.Write(fd, rest)
		if errors.Is(werr, unix.EINTR) {
			werr = nil
		}
		rest = rest[n:]
	}
	if werr == nil {
		werr = unix.Fsync(fd)
	}
	if cerr := unix.Close(fd); werr == nil {
		werr = cerr
	}
	if werr != nil {
		cleanup()
		return werr
	}
	if err := fsOps.rename(dirFd, temp, dirFd, name); err != nil {
		cleanup()
		return err
	}
	_ = unix.Fsync(dirFd) // best effort: not every platform syncs directories
	return nil
}

// unlinkAt removes <dirFd>/<name>, refusing a symlink there: a link where the
// SDK expects its own file means the disk does not match the manifest.
func unlinkAt(dirFd int, name string) error {
	st, err := lstatAt(dirFd, name)
	if err != nil {
		return err
	}
	if isSymlink(st) {
		return errSymlinkRefused
	}
	return fsOps.unlink(dirFd, name)
}

// listDirAt lists the pinned directory through a fresh descriptor on "." so
// the listing cannot be redirected either.
func listDirAt(dirFd int) ([]string, error) {
	fd, err := unix.Openat(dirFd, ".", unix.O_RDONLY|unix.O_DIRECTORY|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	f := os.NewFile(uintptr(fd), ".")
	defer f.Close()
	return f.Readdirnames(-1)
}

func rmdirAt(parentFd int, name string) error {
	return unix.Unlinkat(parentFd, name, unix.AT_REMOVEDIR)
}

func closeFd(fd int) {
	if fd >= 0 {
		_ = unix.Close(fd)
	}
}
