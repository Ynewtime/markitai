package main

import (
	"strings"
	"testing"
)

// WebKit's language of the window's pages, read as macOS orders the
// languages, is a language tag.
func TestFirstPreferredLanguageIsATag(t *testing.T) {
	tag := firstPreferredLanguage()
	if tag == "" || strings.ContainsAny(tag, " \x00_") {
		t.Errorf("firstPreferredLanguage() = %q", tag)
	}
	if workbenchLang(tag) == "" {
		t.Errorf("no workbench language for %q", tag)
	}
}
