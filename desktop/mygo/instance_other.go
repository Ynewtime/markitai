//go:build !windows

package main

import "github.com/egoist/mygo"

// singleInstance takes the single-instance lock (section 7.6), or hands
// the command line to the instance that has it and reports false.
func singleInstance(_, _ string) bool { return mygo.App.RequestSingleInstanceLock() }
