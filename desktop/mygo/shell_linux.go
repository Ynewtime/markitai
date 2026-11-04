package main

import (
	"path/filepath"
	"unsafe"

	"github.com/egoist/mygo"
)

// The programs the app opens start in the user's environment, without the
// variables set for the app's toolkit (appEnv, section 10.5): a launch
// context gives them the user's values, and the app's environment never
// changes as it runs. mygo's Shell launches with the app's.

func (systemDesk) openPath(path string) error {
	var err error
	mygo.RunOnMain(func() { err = launchAsUser(fileURI(path)) })
	return err
}

func (systemDesk) openExternal(url string) error {
	var err error
	mygo.RunOnMain(func() { err = launchAsUser(url) })
	return err
}

// showItemInFolder asks the file manager to show path, as mygo does, and
// else opens its folder.
func (systemDesk) showItemInFolder(path string) {
	mygo.RunOnMain(func() {
		if loadGLib() != nil {
			mygo.Shell.ShowItemInFolder(path)
			return
		}
		s := glib.variantTypeNew("s")
		items := []unsafe.Pointer{glib.variantNewString(fileURI(path))}
		uris := glib.variantNewArray(s, &items[0], 1)
		glib.variantTypeFree(s)
		err := sessionCall("org.freedesktop.FileManager1", "/org/freedesktop/FileManager1", "org.freedesktop.FileManager1", "ShowItems",
			tuple(uris, glib.variantNewString("")), -1)
		if err != nil {
			launchAsUser(fileURI(filepath.Dir(path)))
		}
	})
}

func fileURI(path string) string {
	if loadGLib() != nil {
		return "file://" + path
	}
	f := glib.fileNewForPath(path)
	defer glib.objectUnref(f)
	return takeString(glib.fileGetURI(f))
}

// launchAsUser opens uri with its default program, in the user's
// environment.
func launchAsUser(uri string) error {
	if err := loadGLib(); err != nil {
		return err
	}
	ctx := glib.launchContextNew()
	defer glib.objectUnref(ctx)
	for key, user := range appEnv {
		if user == nil {
			glib.launchContextUnset(ctx, key)
		} else {
			glib.launchContextSetenv(ctx, key, *user)
		}
	}
	var gerr *gError
	if glib.launchDefaultForURI(uri, ctx, &gerr) == 0 {
		return takeError(gerr)
	}
	return nil
}
