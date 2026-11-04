//go:build !windows

package main

// useDownloadsFolder keeps mygo's downloads folder outside Windows, the
// user's Downloads: $HOME/Downloads on macOS, XDG_DOWNLOAD_DIR on Linux.
func useDownloadsFolder() {}
