//go:build !linux

package main

// prepareToolkit has nothing to ready outside Linux.
func prepareToolkit() {}

// toolkitReady has nothing to ready outside Linux: the preview's script
// takes the mouse's back button (actions.go).
func toolkitReady(func()) {}
