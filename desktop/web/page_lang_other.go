//go:build !darwin

package main

import "github.com/egoist/mygo"

// pageLanguage is the language the window's pages start in, as
// navigator.language: mygo hands the web view its own (App.Locale) first.
func pageLanguage() string { return mygo.App.Locale() }
