//go:build unix

package main

import (
	"os"
	"syscall"
)

// procAttr gives the child a process group of its own, so a Ctrl-C in the
// terminal that started the app reaches the app alone.
func procAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{Setpgid: true} }

// openAppend opens path to append to, made if missing.
func openAppend(path string) (*os.File, error) {
	return os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o600)
}
