//go:build !linux

package main

import "github.com/egoist/mygo"

// nameApp keeps the name of mygo.json outside Linux, where the bundle and
// the installer name the app.
func nameApp() {}

// prepareToolkit has nothing to ready outside Linux.
func prepareToolkit() {}

// toolkitReady has nothing to ready outside Linux.
func toolkitReady() {}

// openURL opens url with its default program.
func openURL(url string) error { return mygo.Shell.OpenExternal(url) }

// showInFolder asks the file manager to show path.
func showInFolder(path string) { mygo.Shell.ShowItemInFolder(path) }

// closeQuestions has nothing to close outside Linux, where nothing ends the
// app without its quit (terminate).
func closeQuestions(*mygo.Window) {}
