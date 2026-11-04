//go:build windows

package main

import (
	"os"
	"syscall"
)

// procAttr hides the child's console and gives it a process group of its own.
func procAttr() *syscall.SysProcAttr {
	const createNoWindow, createNewProcessGroup = 0x08000000, 0x00000200
	return &syscall.SysProcAttr{CreationFlags: createNoWindow | createNewProcessGroup, HideWindow: true}
}

// openAppend opens path to append to, made if missing, and shared for
// deletion, as files are on Unix: os.OpenFile's are not, and a serve still
// writing to its log would keep the next launch from moving it aside.
func openAppend(path string) (*os.File, error) {
	name, err := syscall.UTF16PtrFromString(path)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	// os.OpenFile's for O_APPEND (STANDARD_RIGHTS_WRITE, FILE_WRITE_EA among them), and
	// FILE_READ_ATTRIBUTES for Stat.
	const access = syscall.FILE_APPEND_DATA | syscall.FILE_WRITE_ATTRIBUTES | syscall.SYNCHRONIZE |
		0x00020000 | 0x00000010 | 0x00000080
	h, err := syscall.CreateFile(name, access,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, nil,
		syscall.OPEN_ALWAYS, syscall.FILE_ATTRIBUTE_NORMAL, 0)
	if err != nil {
		return nil, &os.PathError{Op: "open", Path: path, Err: err}
	}
	return os.NewFile(uintptr(h), path), nil
}

// signalName is "" on Windows, which ends processes with exit codes.
func signalName(*os.ProcessState) string { return "" }
