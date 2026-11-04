package main

import (
	"log"
	"strings"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo"

	"markitai.local/mygo/internal/l10n"
)

// localizeSystemText has nothing to do before the app starts on Windows:
// mygo gives the process its languages as it starts.
func localizeSystemText(l10n.Lang, []string) {}

var setProcessPreferredUILanguages = syscall.NewLazyDLL("kernel32.dll").NewProc("SetProcessPreferredUILanguages")

// localizeSystemDialogs gives the process mygo's languages again with
// zh-CN before zh-Hans: Windows finds the system's Simplified Chinese
// texts, as those of the file dialogs, under zh-CN, not under the zh-Hans
// mygo names (section 10.5). Call it once the app runs, after mygo set the
// list.
func localizeSystemDialogs() {
	tags := dialogLanguages(append([]string{mygo.App.Locale()}, mygo.App.PreferredLanguages()...))
	if len(tags) == 0 {
		return
	}
	// MUI_LANGUAGE_NAME: names, each ending with a NUL, and one more NUL.
	var list []uint16
	for _, tag := range tags {
		u, err := syscall.UTF16FromString(tag)
		if err != nil {
			continue
		}
		list = append(list, u...)
	}
	list = append(list, 0)
	const muiLanguageName = 0x8
	var n uint32
	if ok, _, err := setProcessPreferredUILanguages.Call(muiLanguageName, uintptr(unsafe.Pointer(&list[0])), uintptr(unsafe.Pointer(&n))); ok == 0 {
		log.Printf("SetProcessPreferredUILanguages %q: %v", tags, err)
	}
}

// dialogLanguages is tags, each once, with zh-CN put before zh-Hans.
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
		if strings.EqualFold(t, "zh-Hans") {
			add("zh-CN")
		}
		if t != "" {
			add(t)
		}
	}
	return out
}
