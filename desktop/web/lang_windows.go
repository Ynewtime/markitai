package main

import (
	"strings"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo"
)

var setProcessPreferredUILanguages = syscall.NewLazyDLL("kernel32.dll").NewProc("SetProcessPreferredUILanguages")

// localizeSystemDialogs gives the process mygo's languages again with
// zh-CN before any Chinese: Windows finds the system's Simplified Chinese
// texts, as those of the file dialogs, under zh-CN, not under the zh-Hans
// mygo names, as Mygo found, and the app shows any Chinese as Simplified
// Chinese (languagesOn). Once the app runs, after mygo set the list, and
// again whenever the app's language changes.
func localizeSystemDialogs() {
	var list []uint16
	for _, tag := range dialogLanguages(append([]string{mygo.App.Locale()}, mygo.App.PreferredLanguages()...)) {
		u, err := syscall.UTF16FromString(tag)
		if err != nil {
			continue
		}
		list = append(list, u...)
	}
	list = append(list, 0)
	const muiLanguageName = 0x8
	var n uint32
	setProcessPreferredUILanguages.Call(muiLanguageName, uintptr(unsafe.Pointer(&list[0])), uintptr(unsafe.Pointer(&n)))
}

// dialogLanguages is tags, each once, with zh-CN put before the first
// Chinese.
func dialogLanguages(tags []string) []string {
	var out []string
	add := func(t string) {
		for _, o := range out {
			if strings.EqualFold(o, t) {
				return
			}
		}
		out = append(out, t)
	}
	for _, t := range tags {
		if workbenchLang(t) == "zh" {
			add("zh-CN")
		}
		if t != "" {
			add(t)
		}
	}
	return out
}
