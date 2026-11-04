package main

import (
	"os"

	"github.com/egoist/mygo"
)

// singleInstance takes the single-instance lock (section 7.6), or hands
// the command line to the instance that has it and reports false.
//
// mygo binds the lock's socket under XDG_RUNTIME_DIR, else TEMP, and only
// in a path of fewer than 100 bytes: under a long TEMP every launch got a
// window of its own. A relative path has no such limit, so the lock is
// taken in the data folder, which every launch of this user shares, by the
// name alone, when a file can be made there; else under TEMP, as mygo
// does. The files named on the command line are made absolute first, as
// the instance that gets them reads them from its own folder.
// prepare_environment keeps the data folder the working one, where mygo
// removes the socket as the app quits.
func singleInstance(data, wd string) bool {
	os.Args = append(os.Args[:1:1], fileArgs(os.Args[1:], wd)...)
	if err := os.MkdirAll(data, 0o755); err != nil || os.Chdir(data) != nil || !writable(".") {
		return mygo.App.RequestSingleInstanceLock()
	}
	old, had := os.LookupEnv("XDG_RUNTIME_DIR")
	os.Setenv("XDG_RUNTIME_DIR", ".")
	defer func() {
		if had {
			os.Setenv("XDG_RUNTIME_DIR", old)
		} else {
			os.Unsetenv("XDG_RUNTIME_DIR")
		}
	}()
	return mygo.App.RequestSingleInstanceLock()
}
