//go:build !darwin

package main

import "github.com/egoist/mygo"

// showAbout shows the About panel where mygo has one; the About item is in
// macOS's app menu only.
func showAbout(name, credits, copyright string) {
	mygo.App.ShowAboutPanel(mygo.AboutPanelOptions{ApplicationName: name, Copyright: copyright, Credits: credits})
}
