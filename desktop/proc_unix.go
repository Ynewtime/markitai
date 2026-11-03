//go:build unix

package main

import (
	"os"
	"syscall"
)

// procAttr gives the child a process group of its own, so a Ctrl-C in the
// terminal that started the app reaches the app alone.
func procAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{Setpgid: true} }

// stopNow sends SIGHUP: serve ends the process groups of the runtimes it
// started (Chromium, LibreOffice), then exits without draining.
func stopNow(p *os.Process) error { return p.Signal(syscall.SIGHUP) }
