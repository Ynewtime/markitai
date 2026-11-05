//go:build !linux

package main

// dragged has no drag to count outside Linux, where the page gets the files
// of a drop itself.
func (w *window) dragged() int { return 0 }

// dropFiles has nothing to hand over outside Linux.
func (w *window) dropFiles() {}
