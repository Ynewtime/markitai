//go:build !linux

package main

// dieWithParentNow leaves it to dieWithParent outside Linux.
func dieWithParentNow(int) {}
