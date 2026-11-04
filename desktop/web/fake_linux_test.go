package main

import (
	"os"
	"syscall"
)

// dieWithParentNow has the kernel kill a fake serve as its parent ends;
// one whose parent ended before it asked ends now.
func dieWithParentNow(parent int) {
	_, _, _ = syscall.RawSyscall(syscall.SYS_PRCTL, syscall.PR_SET_PDEATHSIG, uintptr(syscall.SIGKILL), 0)
	if os.Getppid() != parent {
		os.Exit(3)
	}
}
