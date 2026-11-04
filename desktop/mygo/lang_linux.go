package main

import (
	"markitai.local/mygo/internal/l10n"
)

// localizeSystemText gives GTK's own words, as those of its context menus
// and file chooser, the app's language through gettext's LANGUAGE, as
// Native does (section 10.5): GTK reads it as it starts, after main. A
// first choice the system already gives is set as well, which changes
// nothing. The programs the app opens keep the user's (asUser).
func localizeSystemText(lang l10n.Lang, _ []string) {
	setAppEnv("LANGUAGE", toolkitLanguage(lang))
}

// toolkitLanguage is the gettext language of lang.
func toolkitLanguage(lang l10n.Lang) string {
	if lang == l10n.ZhHans {
		return "zh_CN"
	}
	return "en"
}

// localizeSystemDialogs has nothing to add on Linux: the portal's file
// chooser speaks the desktop's language, as Native's does.
func localizeSystemDialogs() {}
