//go:build !windows

package main

import "github.com/egoist/mygo"

// documentsFolder is the user's Documents folder (DESIGN 4.3).
func documentsFolder() string {
	dir, _ := mygo.App.Path(mygo.PathDocuments)
	return dir
}
