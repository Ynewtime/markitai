package main

import (
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo"
)

var (
	procSHGetKnownFolderPath = syscall.NewLazyDLL("shell32.dll").NewProc("SHGetKnownFolderPath")
	procCoTaskMemFree        = syscall.NewLazyDLL("ole32.dll").NewProc("CoTaskMemFree")

	folderIDDownloads = syscall.GUID{Data1: 0x374DE290, Data2: 0x123F, Data3: 0x4565, Data4: [8]byte{0x91, 0x64, 0x39, 0xC4, 0x92, 0x5E, 0x46, 0x7B}}
	folderIDProfile   = syscall.GUID{Data1: 0x5E6C858F, Data2: 0x0E22, Data3: 0x4760, Data4: [8]byte{0x9A, 0xFE, 0xEA, 0x33, 0x17, 0xB6, 0x71, 0x73}}
)

// useDownloadsFolder makes the known folder FOLDERID_Downloads the app's
// downloads folder, where a download the workbench starts goes: it follows
// the folder's moves (Properties > Location) and redirection, which mygo's
// %USERPROFILE%\Downloads does not. A process whose USERPROFILE names
// another folder than the account's profile, as the tests' isolated home,
// keeps its downloads in that folder, as the rest of its files; so does a
// build that chose the folder itself (downloadsChosen).
func useDownloadsFolder() {
	if downloadsChosen {
		return
	}
	profile, ok := knownFolder(&folderIDProfile)
	if !ok || !strings.EqualFold(filepath.Clean(profile), filepath.Clean(os.Getenv("USERPROFILE"))) {
		return
	}
	if dir, ok := knownFolder(&folderIDDownloads); ok {
		mygo.App.SetPath(mygo.PathDownloads, dir)
	}
}

// knownFolder is the path of a known folder of the user.
func knownFolder(id *syscall.GUID) (string, bool) {
	var p *uint16
	r, _, _ := procSHGetKnownFolderPath.Call(uintptr(unsafe.Pointer(id)), 0, 0, uintptr(unsafe.Pointer(&p)))
	if p != nil {
		// Freed whether the call succeeded or not.
		defer procCoTaskMemFree.Call(uintptr(unsafe.Pointer(p)))
	}
	if r != 0 || p == nil {
		return "", false
	}
	n := 0
	for *(*uint16)(unsafe.Add(unsafe.Pointer(p), 2*n)) != 0 {
		n++
	}
	return syscall.UTF16ToString(unsafe.Slice(p, n)), true
}
