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

func stopNow(p *os.Process) error { return p.Kill() }
