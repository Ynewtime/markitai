package main

import (
	"bytes"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo"
)

// pageLanguage is the language the window's pages start in, as
// navigator.language: WebKit takes the first of the app's languages as
// macOS orders them at launch (an -AppleLanguages argument, the app's own
// language, as App.SetLocale keeps it, else the user's), whether the app
// is written in it or not. mygo's Locale is the best of the app's own
// languages instead, which a workbench that follows the system does not
// pick: with Traditional Chinese first the workbench shows Simplified
// Chinese, and with French then Japanese it shows English.
var pageLanguage = sync.OnceValue(func() string {
	if tag := firstPreferredLanguage(); tag != "" {
		return tag
	}
	return mygo.App.Locale()
})

// firstPreferredLanguage is the first of CFLocaleCopyPreferredLanguages,
// "" when it cannot be read.
func firstPreferredLanguage() string {
	cf, err := purego.Dlopen("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation",
		purego.RTLD_LAZY|purego.RTLD_GLOBAL)
	if err != nil {
		return ""
	}
	var (
		copyPreferred func() uintptr
		count         func(array uintptr) int
		at            func(array uintptr, i int) uintptr
		cString       func(s uintptr, buf *byte, size int, encoding uint32) bool
		release       func(ref uintptr)
	)
	purego.RegisterLibFunc(&copyPreferred, cf, "CFLocaleCopyPreferredLanguages")
	purego.RegisterLibFunc(&count, cf, "CFArrayGetCount")
	purego.RegisterLibFunc(&at, cf, "CFArrayGetValueAtIndex")
	purego.RegisterLibFunc(&cString, cf, "CFStringGetCString")
	purego.RegisterLibFunc(&release, cf, "CFRelease")
	langs := copyPreferred()
	if langs == 0 {
		return ""
	}
	defer release(langs)
	if count(langs) == 0 {
		return ""
	}
	const utf8 = 0x08000100 // kCFStringEncodingUTF8
	buf := make([]byte, 128)
	if !cString(at(langs, 0), &buf[0], len(buf), utf8) {
		return ""
	}
	return string(buf[:bytes.IndexByte(buf, 0)])
}
