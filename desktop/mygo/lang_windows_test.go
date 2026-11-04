package main

import (
	"slices"
	"testing"
)

// TestDialogLanguages puts zh-CN before zh-Hans, where Windows keeps its
// Simplified Chinese texts, and names each language once (section 10.5).
func TestDialogLanguages(t *testing.T) {
	for _, c := range []struct{ in, want []string }{
		{[]string{"zh-Hans", "zh-Hans", "en-US", "en"}, []string{"zh-CN", "zh-Hans", "en-US", "en"}},
		{[]string{"en", "zh-hans", "ZH-CN"}, []string{"en", "zh-CN", "zh-hans"}},
		{[]string{"", "fr-FR"}, []string{"fr-FR"}},
		{nil, nil},
	} {
		if got := dialogLanguages(c.in); !slices.Equal(got, c.want) {
			t.Errorf("dialogLanguages(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
