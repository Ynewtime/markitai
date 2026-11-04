//go:build darwin || linux

package core

import "github.com/ebitengine/purego"

func open(path string) (uintptr, error) {
	return purego.Dlopen(path, purego.RTLD_NOW|purego.RTLD_LOCAL)
}

func sym(h uintptr, name string) (uintptr, error) { return purego.Dlsym(h, name) }
