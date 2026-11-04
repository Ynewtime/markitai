package l10n

import (
	"bytes"
	"os"
	"testing"
	"testing/fstest"
	"time"

	"github.com/egoist/mygo/i18n"
)

func TestCopyMatchesDesktopStrings(t *testing.T) {
	spec, err := os.ReadFile("../../../strings.json")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(spec, catalogJSON) {
		t.Error("internal/l10n/strings.json differs from desktop/strings.json: run desktop/mygo/build.py, which copies it")
	}
}

func TestEveryKeyHasBothLanguages(t *testing.T) {
	for _, k := range Keys() {
		if En.Raw(k) == "" || ZhHans.Raw(k) == "" {
			t.Errorf("%s lacks a language", k)
		}
	}
	for _, k := range []string{"app.name.mygo", "menu.about.mygo", "detail.show.linux", "preview.unavailable", "common.copy"} {
		if En.Raw(k) == "" {
			t.Errorf("no %s", k)
		}
	}
}

func TestChoose(t *testing.T) {
	for in, want := range map[string]Lang{"zh-Hant-TW": ZhHans, "zh-Hans-CN": ZhHans, "ZH-cn": ZhHans, "en-US": En, "ja-JP": En, "": En} {
		if got := Choose([]string{in, "zh-Hans"}); got != want {
			t.Errorf("Choose(%q) = %s, want %s", in, got, want)
		}
	}
	if Choose(nil) != En {
		t.Error("no languages is not English")
	}
}

func TestFormats(t *testing.T) {
	for _, c := range []struct{ got, want string }{
		{En.Duration(0), "0.1 s"},
		{En.Duration(0.04), "0.1 s"},
		{ZhHans.Duration(0.04), "0.1 秒"},
		{En.Duration(0.26), "0.3 s"},
		{ZhHans.Duration(18.6), "18.6 秒"},
		{En.Duration(125), "2 min 5 s"},
		{ZhHans.Duration(125), "2 分 5 秒"},
		{En.P("state.warnings", 1), "1 warning"},
		{ZhHans.P("state.warnings", 3), "3 条警告"},
		{En.Grouped(24310), "24,310"},
		{ZhHans.Grouped(24310), "24,310"},
		{USD(0.0412), "$0.0412"},
		{USD(12), "$12.00"},
		{Elapsed(75), "1:15"},
		{En.T("action.open_settings", "settings", En.T("nav.settings")), "Open Settings"},
		{ZhHans.T("action.open_settings", "settings", ZhHans.T("nav.settings")), "打开设置"},
	} {
		if c.got != c.want {
			t.Errorf("got %q, want %q", c.got, c.want)
		}
	}
}

func TestFinished(t *testing.T) {
	now := time.Date(2026, 11, 4, 15, 0, 0, 0, time.Local)
	today := time.Date(2026, 11, 4, 9, 5, 0, 0, time.Local)
	yesterday := time.Date(2026, 11, 3, 23, 59, 0, 0, time.Local)
	older := time.Date(2026, 10, 2, 14, 42, 0, 0, time.Local)
	for _, c := range []struct{ got, want string }{
		{En.Finished(today, now), "Today, 9:05 AM"},
		{ZhHans.Finished(today, now), "今天 09:05"},
		{En.Finished(yesterday, now), "Yesterday, 11:59 PM"},
		{En.Finished(older, now), "Oct 2, 2026, 2:42 PM"},
		{ZhHans.Finished(older, now), "2026年10月2日 14:42"},
	} {
		if c.got != c.want {
			t.Errorf("got %q, want %q", c.got, c.want)
		}
	}
}

func TestErrorKeys(t *testing.T) {
	for code, want := range map[string]struct {
		sentence string
		second   Action
	}{
		"not_found":          {"error.not_found", RemoveItem},
		"image_only":         {"error.image_only", OpenSettings},
		"config_error":       {"error.config_error", OpenConfig},
		"fetch_error":        {"error.fetch_error", NoAction},
		"internal_error":     {"error.internal", NoAction},
		"something_new":      {"error.unknown", NoAction},
		"output_unavailable": {"error.output_unavailable", OpenSettings},
	} {
		s, a := ErrorKey(code)
		if s != want.sentence || a != want.second {
			t.Errorf("%s: %s %s, want %+v", code, s, a, want)
		}
		_ = En.Raw(s)
	}
	for code, e := range errors {
		_ = En.Raw(e.sentence)
		if e.second != NoAction {
			_ = En.Raw(string(e.second))
		}
		_ = code
	}
}

// TestMygoTextChanged: a library's catalog of namespace mygo, written as
// mygo's own menu strings are, says Hide and Quit with the space once the
// app changed them, and the rest as mygo wrote them.
func TestMygoTextChanged(t *testing.T) {
	ChangeMygoText()
	lib, err := i18n.NewLibraryCatalog(fstest.MapFS{
		"en.json":      {Data: []byte(`{"menuHide": "Hide {app}", "menuQuitApp": "Quit {app}", "menuAbout": "About {app}"}`)},
		"zh-Hans.json": {Data: []byte(`{"menuHide": "隐藏{app}", "menuQuitApp": "退出{app}", "menuAbout": "关于{app}"}`)},
	}, "en", "mygo")
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct{ locale, key, want string }{
		{"zh-Hans", "menuHide", "隐藏 Markitai Mygo"},
		{"zh-Hans", "menuQuitApp", "退出 Markitai Mygo"},
		{"zh-Hans", "menuAbout", "关于Markitai Mygo"},
		{"en", "menuHide", "Hide Markitai Mygo"},
		{"en", "menuQuitApp", "Quit Markitai Mygo"},
	} {
		if got := lib.Format(c.locale, c.key, map[string]any{"app": "Markitai Mygo"}); got != c.want {
			t.Errorf("%s %s: %q, want %q", c.locale, c.key, got, c.want)
		}
	}
}
