// Package testenv gives tests the isolated environment of
// desktop/mygo/DESIGN.md 15 and the core library to call.
package testenv

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// root is the repository root, found before any test moves the working
// directory, as prepare_environment does.
var root = func() string {
	dir, _ := os.Getwd()
	for ; dir != filepath.Dir(dir); dir = filepath.Dir(dir) {
		if _, err := os.Stat(filepath.Join(dir, "Cargo.toml")); err == nil {
			return dir
		}
	}
	return ""
}()

// Repo is the repository root.
func Repo(t testing.TB) string {
	t.Helper()
	if root == "" {
		t.Fatal("no repository root above the test")
	}
	return root
}

// Fixture is a repository file named by its slash-separated path.
func Fixture(t testing.TB, rel string) string {
	return filepath.Join(Repo(t), filepath.FromSlash(rel))
}

// Isolate points HOME, MARKITAI_HOME and every user folder a platform
// reads at a new temporary folder for the test, and returns it.
func Isolate(t testing.TB) string {
	t.Helper()
	home := t.TempDir()
	// prepare_environment moves the process into the app's data folder, and
	// Windows removes no folder a process works in. Cleanups run last first,
	// so the test goes back before its temporary folder is removed.
	if wd, err := os.Getwd(); err == nil {
		t.Cleanup(func() { os.Chdir(wd) })
	}
	for k, v := range map[string]string{
		"HOME":            home,
		"MARKITAI_HOME":   filepath.Join(home, ".markitai"),
		"XDG_DATA_HOME":   filepath.Join(home, ".local", "share"),
		"XDG_CONFIG_HOME": filepath.Join(home, ".config"),
		"XDG_CACHE_HOME":  filepath.Join(home, ".cache"),
		"XDG_STATE_HOME":  filepath.Join(home, ".local", "state"),
		"USERPROFILE":     home,
		"APPDATA":         filepath.Join(home, "AppData", "Roaming"),
		"LOCALAPPDATA":    filepath.Join(home, "AppData", "Local"),
		"TEMP":            filepath.Join(home, "t"),
		"TMP":             filepath.Join(home, "t"),
		"TERM":            "dumb",
	} {
		t.Setenv(k, v)
	}
	os.Mkdir(filepath.Join(home, "t"), 0o755)
	return home
}

// Library is the core library to test against: MARKITAI_APP_LIB when
// set, else where build.py has cargo leave it: the release build in the
// target folder (CARGO_TARGET_DIR when set), and on Windows that
// folder's mygo-windows build, which links the C runtime statically.
func Library(t testing.TB) string {
	if p := os.Getenv("MARKITAI_APP_LIB"); p != "" {
		return p
	}
	target := os.Getenv("CARGO_TARGET_DIR")
	if target == "" {
		target = filepath.Join(Repo(t), "target")
	}
	switch runtime.GOOS {
	case "windows":
		triple := map[string]string{"amd64": "x86_64-pc-windows-msvc", "arm64": "aarch64-pc-windows-msvc"}[runtime.GOARCH]
		return filepath.Join(target, "mygo-windows", triple, "release", "markitai_app.dll")
	case "darwin":
		return filepath.Join(target, "release", "libmarkitai_app.dylib")
	default:
		return filepath.Join(target, "release", "libmarkitai_app.so")
	}
}
