package main

import (
	"slices"
	"testing"

	"markitai.local/mygo/internal/l10n"
)

// TestAppleLanguages sets the app's language in the test process's
// argument domain, which lasts for the process and writes nothing.
func TestAppleLanguages(t *testing.T) {
	localizeSystemText(l10n.ZhHans)
	if got := appleLanguages(); !slices.Equal(got, []string{"zh-Hans"}) {
		t.Errorf("AppleLanguages %q", got)
	}
	localizeSystemText(l10n.En)
	if got := appleLanguages(); !slices.Equal(got, []string{"en"}) {
		t.Errorf("AppleLanguages %q", got)
	}
}
