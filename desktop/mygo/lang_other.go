//go:build !darwin

package main

import "markitai.local/mygo/internal/l10n"

// localizeSystemText has nothing to do here: mygo labels Windows's and
// Linux's system text in the app's language (section 10.5).
func localizeSystemText(l10n.Lang) {}
