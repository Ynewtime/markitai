//go:build !linux

package main

// holdTermination leaves SIGTERM to mygo outside Linux: macOS ends a
// session through the app's quit, which the app may cancel, and Windows
// through the window's end.
func holdTermination() {}

// takeTermination takes nothing outside Linux.
func takeTermination(func()) {}
