//go:build unix

package skills

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

// The primitives are tested on their own: the materialization layer's
// checks refuse the same things, so its tests prove only the pair.

func openDir(t *testing.T, path string) int {
	t.Helper()
	fd, err := realOpenRoot(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { closeFd(fd) })
	return fd
}

func TestPrimitiveRefusesSymlinkedDirectory(t *testing.T) {
	parent := t.TempDir()
	_ = os.Symlink(t.TempDir(), filepath.Join(parent, "link"))
	pfd := openDir(t, parent)
	if _, err := openDirNoFollow(pfd, "link"); !errors.Is(err, errSymlinkRefused) {
		t.Fatalf("openDirNoFollow: %v", err)
	}
	if _, err := openOrCreateDir(pfd, "link"); !errors.Is(err, errSymlinkRefused) {
		t.Fatalf("openOrCreateDir: %v", err)
	}
	if _, err := openDirNoFollow(pfd, "missing"); !errors.Is(err, errDirMissing) {
		t.Fatalf("missing: %v", err)
	}
	_ = os.WriteFile(filepath.Join(parent, "file"), nil, 0o644)
	if _, err := openOrCreateDir(pfd, "file"); !errors.Is(err, errNotDirectory) {
		t.Fatalf("file: %v", err)
	}
}

func TestPrimitiveRefusesSymlinkedUnlinkTarget(t *testing.T) {
	dir := t.TempDir()
	victim := filepath.Join(t.TempDir(), "victim")
	_ = os.WriteFile(victim, []byte("v"), 0o644)
	_ = os.Symlink(victim, filepath.Join(dir, "SKILL.md"))
	if err := unlinkAt(openDir(t, dir), "SKILL.md"); !errors.Is(err, errSymlinkRefused) {
		t.Fatalf("unlinkAt: %v", err)
	}
	if _, err := os.Lstat(filepath.Join(dir, "SKILL.md")); err != nil {
		t.Fatal("the link itself was removed")
	}
}

func TestPrimitiveExclusiveTempCreation(t *testing.T) {
	dir := t.TempDir()
	fd := openDir(t, dir)
	real := fsOps.createTemp
	t.Cleanup(func() { fsOps.createTemp = real })
	var names []string
	var flagsSeen []int
	var dirFds []int
	planted := ""
	fsOps.createTemp = func(dirFd int, name string, flags int, mode uint32) (int, error) {
		names = append(names, name)
		flagsSeen = append(flagsSeen, flags)
		dirFds = append(dirFds, dirFd)
		if planted == "" {
			// Take the first name with a real file: exclusivity must refuse it.
			planted = name
			_ = os.WriteFile(filepath.Join(dir, name), []byte("planted"), 0o644)
		}
		return real(dirFd, name, flags, mode)
	}
	if err := atomicWrite(fd, "SKILL.md", []byte("data")); err != nil {
		t.Fatal(err)
	}
	if len(names) != 2 || names[0] == names[1] {
		t.Fatalf("a taken name must produce a fresh one: %v", names)
	}
	for i, f := range flagsSeen {
		if f&unix.O_EXCL == 0 || f&unix.O_CREAT == 0 || f&unix.O_NOFOLLOW == 0 || dirFds[i] != fd {
			t.Fatalf("temp open flags %x in fd %d (want fd %d)", f, dirFds[i], fd)
		}
	}
	if readFile(t, filepath.Join(dir, planted)) != "planted" || readFile(t, filepath.Join(dir, "SKILL.md")) != "data" {
		t.Fatal("wrote through a taken temp name")
	}
	if !strings.HasPrefix(names[1], ".SKILL.md.") || !isTempName(names[1]) {
		t.Fatalf("temp name %q", names[1])
	}
}

func TestPrimitiveBoundedNonBlockingRead(t *testing.T) {
	dir := t.TempDir()
	_ = os.WriteFile(filepath.Join(dir, "big"), make([]byte, 1<<20), 0o644)
	fd := openDir(t, dir)
	data, exists, err := readRegularAt(fd, "big", 10)
	if err != nil || !exists || len(data) != 11 {
		t.Fatalf("read %d bytes (want maxBytes+1 = 11): %v", len(data), err)
	}
	if err := unix.Mkfifo(filepath.Join(dir, "fifo"), 0o644); err != nil {
		t.Skip("no FIFOs:", err)
	}
	withDeadline(t, 5*time.Second, func() {
		if _, _, err := readRegularAt(fd, "fifo", 10); !errors.Is(err, errNotRegular) {
			t.Errorf("FIFO: %v", err)
		}
	})
	_ = os.Symlink(filepath.Join(dir, "big"), filepath.Join(dir, "link"))
	if _, _, err := readRegularAt(fd, "link", 10); !errors.Is(err, errSymlinkRefused) {
		t.Fatalf("symlink: %v", err)
	}
	if _, exists, err := readRegularAt(fd, "absent", 10); exists || err != nil {
		t.Fatalf("absent: exists=%v err=%v", exists, err)
	}
}

// Positive control for descriptor-relative addressing: pin, swap, write.
func TestPrimitiveWriteLandsInThePinnedDirectory(t *testing.T) {
	parent, outside := t.TempDir(), t.TempDir()
	dir := filepath.Join(parent, "d")
	_ = os.Mkdir(dir, 0o755)
	fd := openDir(t, dir)
	_ = os.Rename(dir, filepath.Join(parent, "aside"))
	_ = os.Symlink(outside, dir)
	if err := atomicWrite(fd, "SKILL.md", []byte("pinned")); err != nil {
		t.Fatal(err)
	}
	if readFile(t, filepath.Join(parent, "aside", "SKILL.md")) != "pinned" || len(listDir(t, outside)) != 0 {
		t.Fatal("the write followed the name, not the pinned inode")
	}
	if err := unlinkAt(fd, "SKILL.md"); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(parent, "aside", "SKILL.md")); !os.IsNotExist(err) {
		t.Fatal("the unlink followed the name")
	}
}
