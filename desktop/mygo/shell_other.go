//go:build !linux

package main

import "github.com/egoist/mygo"

func (systemDesk) showItemInFolder(path string)  { mygo.Shell.ShowItemInFolder(path) }
func (systemDesk) openPath(path string) error    { return mygo.Shell.OpenPath(path) }
func (systemDesk) openExternal(url string) error { return mygo.Shell.OpenExternal(url) }
