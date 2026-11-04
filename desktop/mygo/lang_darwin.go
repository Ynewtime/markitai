package main

import (
	"log"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"

	"markitai.local/mygo/internal/l10n"
)

// localizeSystemText puts the app's language first in AppleLanguages in
// the process's argument domain, before AppKit starts, so its panels and
// the items it adds to menus follow the app (section 10.5). mygo's
// SetLanguages does nothing on macOS. The argument domain lasts for the
// process and writes nothing. Foundation keeps the languages it reads
// first, so main calls it before anything else asks for a language.
func localizeSystemText(lang l10n.Lang, _ []string) {
	if _, err := purego.Dlopen("/System/Library/Frameworks/Foundation.framework/Foundation", purego.RTLD_GLOBAL|purego.RTLD_NOW); err != nil {
		log.Printf("AppleLanguages: %v", err)
		return
	}
	sel := objc.RegisterName
	str := func(s string) objc.ID {
		return objc.ID(objc.GetClass("NSString")).Send(sel("stringWithUTF8String:"), s)
	}
	defaults := objc.ID(objc.GetClass("NSUserDefaults")).Send(sel("standardUserDefaults"))
	domain := str("NSArgumentDomain")
	args := defaults.Send(sel("volatileDomainForName:"), domain).Send(sel("mutableCopy"))
	args.Send(sel("setObject:forKey:"), objc.ID(objc.GetClass("NSArray")).Send(sel("arrayWithObject:"), str(string(lang))), str("AppleLanguages"))
	defaults.Send(sel("setVolatileDomain:forName:"), args, domain)
	log.Printf("AppleLanguages in the argument domain: %v", appleLanguages())
}

// appleLanguages reads AppleLanguages as the process sees it.
func appleLanguages() []string {
	sel := objc.RegisterName
	defaults := objc.ID(objc.GetClass("NSUserDefaults")).Send(sel("standardUserDefaults"))
	list := defaults.Send(sel("stringArrayForKey:"), objc.ID(objc.GetClass("NSString")).Send(sel("stringWithUTF8String:"), "AppleLanguages"))
	if list == 0 {
		return nil
	}
	n := objc.Send[uint](list, sel("count"))
	out := make([]string, 0, n)
	for i := range n {
		out = append(out, goString(objc.Send[*byte](list.Send(sel("objectAtIndex:"), i), sel("UTF8String"))))
	}
	return out
}

// localizeSystemDialogs has nothing to add on macOS: the dialogs are
// AppKit's.
func localizeSystemDialogs() {}

func goString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return unsafe.String(p, n)
}
