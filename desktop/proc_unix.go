//go:build unix

package main

import "syscall"

// procAttr gives the child a process group of its own, so a Ctrl-C in the
// terminal that started the app reaches the app alone.
func procAttr() *syscall.SysProcAttr { return &syscall.SysProcAttr{Setpgid: true} }
