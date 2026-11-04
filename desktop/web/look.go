package main

import (
	"strings"
	"sync/atomic"

	"github.com/egoist/mygo"
)

// Look is the workbench's own choice of language and theme, as it keeps
// them in localStorage (markitai.lang, markitai.theme): "" where it follows
// the system, as the theme does until the user picks one in its Settings.
// The preload reports it (pages.go), and the app shows its own words,
// window and dialogs in the same:
//
//   - The language becomes mygo's (App.SetLocale), which mygo keeps for the
//     next launches and hands to the web view as navigator.language, the
//     language a workbench that follows the system takes, and on Windows
//     to the system's dialogs.
//   - The theme becomes the window's (Theme.SetSource), kept in web.json, so
//     the next launch's status pages come in it before the workbench does.
type Look struct {
	Lang  string // "en", "zh" or "ja"
	Theme string // "light" or "dark"
}

// lookOf keeps only the values the workbench stores.
func lookOf(lang, theme string) Look {
	var l Look
	if _, ok := texts[lang]; ok {
		l.Lang = lang
	}
	if theme == "light" || theme == "dark" {
		l.Theme = theme
	}
	return l
}

// reports orders the workbench's reports of its look (MarkitaiWeb.Look),
// which can come out of order: a report of a document is newer than the
// ones it numbered before, and a document's first report retires the
// documents before it, whose later reports come late.
type reports struct {
	doc     string
	n       int
	retired map[string]bool
}

// newer says whether the n-th report of doc is the latest so far, and
// takes it as such.
func (r *reports) newer(doc string, n int) bool {
	switch {
	case r.retired[doc]:
		return false
	case doc == r.doc:
		if n <= r.n {
			return false
		}
	default:
		if r.doc != "" {
			if r.retired == nil {
				r.retired = map[string]bool{}
			}
			r.retired[r.doc] = true
		}
		r.doc = doc
	}
	r.n = n
	return true
}

// webLanguages are the workbench's languages as mygo names them, its own
// first: mygo resolves the system's languages among them, so a workbench
// that follows the system and the app's own words agree.
var webLanguages = []string{"en", "zh-Hans", "ja"}

// locale is the mygo language of a workbench language.
func locale(lang string) string {
	if lang == "zh" {
		return "zh-Hans"
	}
	return lang
}

// workbenchLang is the workbench's language for a page whose
// navigator.language is tag, as its boot.js picks it.
func workbenchLang(tag string) string {
	switch tag = strings.ToLower(tag); {
	case strings.HasPrefix(tag, "zh"):
		return "zh"
	case strings.HasPrefix(tag, "ja"):
		return "ja"
	}
	return "en"
}

// shownTexts are the app's words in the language it shows now. Safe from
// any goroutine.
func shownTexts() Texts { return texts[shownLang()] }

// chose says the app took the workbench's language this run (window.look).
var chose atomic.Bool

// shownLang is the workbench's language the app shows: the one the
// workbench chose this run, else the one a workbench that follows the
// system shows, by the language the window's pages start in.
func shownLang() string {
	if chose.Load() {
		return workbenchLang(mygo.App.Locale())
	}
	return workbenchLang(pageLanguage())
}

// themeSource is the appearance the app's window and dialogs take for a
// workbench theme: the system's where the workbench follows it.
func themeSource(theme string) mygo.ThemeSource {
	switch theme {
	case "light":
		return mygo.ThemeLight
	case "dark":
		return mygo.ThemeDark
	}
	return mygo.ThemeSystem
}
