package main

import (
	"slices"
	"testing"
)

func TestDialogLanguagesPutZhCNBeforeAnyChinese(t *testing.T) {
	cases := []struct{ tags, want []string }{
		{[]string{"zh-Hans", "zh-Hans-CN", "en-US"}, []string{"zh-CN", "zh-Hans", "zh-Hans-CN", "en-US"}},
		{[]string{"zh-Hant", "zh-Hant-TW"}, []string{"zh-CN", "zh-Hant", "zh-Hant-TW"}},
		{[]string{"en", "zh-Hant-TW"}, []string{"en", "zh-CN", "zh-Hant-TW"}},
		{[]string{"ja", "", "ja"}, []string{"ja"}},
	}
	for _, c := range cases {
		if got := dialogLanguages(c.tags); !slices.Equal(got, c.want) {
			t.Errorf("dialogLanguages(%v) = %v; want %v", c.tags, got, c.want)
		}
	}
}
