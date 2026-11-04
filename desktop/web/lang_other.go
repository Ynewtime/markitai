//go:build !windows

package main

// localizeSystemDialogs has nothing to add outside Windows: macOS's panels
// take the app's language from its localizations (mygo.json), and on Linux
// the portal's file chooser speaks the desktop's language, GTK's own the
// app's (prepareToolkit).
func localizeSystemDialogs() {}
