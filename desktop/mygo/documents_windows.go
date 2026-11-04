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

	folderIDDocuments = syscall.GUID{Data1: 0xFDD39AD0, Data2: 0x238F, Data3: 0x46AF, Data4: [8]byte{0xAD, 0xB4, 0x6C, 0x85, 0x48, 0x03, 0x69, 0xC7}}
	folderIDProfile   = syscall.GUID{Data1: 0x5E6C858F, Data2: 0x0E22, Data3: 0x4760, Data4: [8]byte{0x9A, 0xFE, 0xEA, 0x33, 0x17, 0xB6, 0x71, 0x73}}
)

// documentsFolder is the known folder FOLDERID_Documents, which follows
// OneDrive and other redirection (DESIGN 4.3). A process whose USERPROFILE
// names another folder than the account's profile, as the tests' isolated
// home, keeps its Documents in that folder, as the rest of its files.
func documentsFolder() string {
	fallback, _ := mygo.App.Path(mygo.PathDocuments)
	profile, ok := knownFolder(&folderIDProfile)
	if !ok || !strings.EqualFold(filepath.Clean(profile), filepath.Clean(os.Getenv("USERPROFILE"))) {
		return fallback
	}
	if dir, ok := knownFolder(&folderIDDocuments); ok {
		return filepath.Clean(dir)
	}
	return fallback
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
