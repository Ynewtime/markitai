package core

import (
	"debug/elf"
	"debug/macho"
	"debug/pe"
	"regexp"
	"runtime"
	"strings"
	"testing"

	"markitai.local/mygo/internal/testenv"
)

// TestLibraryShipsAlone checks what the core library asks of the machine
// that runs the app: on macOS the bundle's rpath names it and it links
// only system libraries, on Windows it needs no Visual C++
// redistributable, on Linux only the C runtime.
func TestLibraryShipsAlone(t *testing.T) {
	path := testenv.Library(t)
	switch runtime.GOOS {
	case "darwin":
		f, err := macho.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		if id := installName(f); id != "@rpath/libmarkitai_app.dylib" {
			t.Errorf("install name %q, want @rpath/libmarkitai_app.dylib (build.py passes it to the linker)", id)
		}
		libs, _ := f.ImportedLibraries()
		for _, l := range libs {
			if !strings.HasPrefix(l, "/usr/lib/") && !strings.HasPrefix(l, "/System/Library/") {
				t.Errorf("links %s, outside the system", l)
			}
		}
	case "windows":
		f, err := pe.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		libs, _ := f.ImportedLibraries()
		for _, l := range libs {
			if l = strings.ToLower(l); strings.HasPrefix(l, "vcruntime") || strings.HasPrefix(l, "msvcp") || strings.HasPrefix(l, "api-ms-win-crt-") {
				t.Errorf("imports %s: the C runtime is not linked statically (build.py's +crt-static)", l)
			}
		}
		t.Logf("imports %v", libs)
	default:
		f, err := elf.Open(path)
		if err != nil {
			t.Fatal(err)
		}
		defer f.Close()
		libs, _ := f.ImportedLibraries()
		for _, l := range libs {
			if !cRuntime.MatchString(l) {
				t.Errorf("needs %s, beyond the C runtime", l)
			}
		}
		t.Logf("needs %v", libs)
	}
}

// cRuntime matches the libraries of glibc and the GCC runtime, which every
// Linux system that runs the app has.
var cRuntime = regexp.MustCompile(`^(libc|libm|libgcc_s|libpthread|libdl|librt)\.so\.\d+$|^ld-linux-[a-z0-9_-]+\.so\.\d+$`)

// installName is the library's LC_ID_DYLIB, which debug/macho leaves as
// raw bytes.
func installName(f *macho.File) string {
	const lcIDDylib = 0xd
	for _, l := range f.Loads {
		raw := l.Raw()
		if len(raw) < 12 || f.ByteOrder.Uint32(raw) != lcIDDylib {
			continue
		}
		off := f.ByteOrder.Uint32(raw[8:])
		if int(off) >= len(raw) {
			return ""
		}
		name := raw[off:]
		if i := strings.IndexByte(string(name), 0); i >= 0 {
			name = name[:i]
		}
		return string(name)
	}
	return ""
}
