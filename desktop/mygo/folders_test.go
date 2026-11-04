package main

import (
	"os"
	"path/filepath"
	"testing"
	"unicode/utf8"

	"markitai.local/mygo/internal/testenv"
)

func TestFolders(t *testing.T) {
	home := testenv.Isolate(t)
	env := map[string]string{"LOCALAPPDATA": filepath.Join(home, "Local"), "XDG_DATA_HOME": filepath.Join(home, "xd"),
		"XDG_CACHE_HOME": filepath.Join(home, "xc"), "XDG_CONFIG_HOME": filepath.Join(home, "xf")}
	getenv := func(k string) string { return env[k] }
	for _, c := range []struct{ goos, data, cache, settings string }{
		{"darwin", filepath.Join(home, "Library", "Application Support", appID), filepath.Join(home, "Library", "Caches", appID),
			filepath.Join(home, "Library", "Application Support", appID, "settings.json")},
		{"windows", filepath.Join(home, "Local", "Markitai Mygo"), filepath.Join(home, "Local", "Markitai Mygo", "Cache"),
			filepath.Join(home, "Local", "Markitai Mygo", "settings.json")},
		{"linux", filepath.Join(home, "xd", appID), filepath.Join(home, "xc", appID), filepath.Join(home, "xf", appID, "settings.json")},
	} {
		data, cache, settings := mygoFolders(c.goos, home, getenv)
		if data != c.data || cache != c.cache || settings != c.settings {
			t.Errorf("%s: %s %s %s", c.goos, data, cache, settings)
		}
	}
	data, _, settings := mygoFolders("linux", home, func(string) string { return "relative" })
	if data != filepath.Join(home, ".local", "share", appID) || settings != filepath.Join(home, ".config", appID, "settings.json") {
		t.Errorf("Linux without XDG folders: %s %s", data, settings)
	}
}

func TestFileArgs(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "a.pdf")
	os.WriteFile(file, nil, 0o644)
	got := fileArgs([]string{"-psn_0_123", "-AppleLanguages", "(zh-Hans)", "a.pdf", file, filepath.Join(dir, "gone.pdf")}, dir)
	if len(got) != 2 || got[0] != file || got[1] != file {
		t.Errorf("%q", got)
	}
}

func TestMiddleCut(t *testing.T) {
	width := func(s string) float32 { return float32(utf8.RuneCountInString(s)) }
	if got := middleCut("Quarterly Report Q3 2026.pdf", 12, width); got != "Quarty…26.pdf" && utf8.RuneCountInString(got) > 12 {
		t.Errorf("%q", got)
	}
	got := middleCut("Quarterly Report Q3 2026.pdf", 12, width)
	if utf8.RuneCountInString(got) != 12 || got[:3] != "Qua" || got[len(got)-4:] != ".pdf" {
		t.Errorf("%q does not keep both ends", got)
	}
	if got := middleCut("short.pdf", 12, width); got != "short.pdf" {
		t.Errorf("%q", got)
	}
}

func TestPreviewNavigation(t *testing.T) {
	base := folderURL("/Users/example/Documents/Markitai/Report Q3")
	if base != "file:///Users/example/Documents/Markitai/Report%20Q3/" {
		t.Errorf("base %q", base)
	}
	if got := folderURL(`C:\Users\example\Report`); filepath.Separator == '\\' && got != "file:///C:/Users/example/Report/" {
		t.Errorf("a Windows base %q", got)
	}
	for _, c := range []struct {
		url             string
		allow, external bool
	}{
		{base, true, false},
		{base + "#fn-1", true, false},
		{"https://example.com/", false, true},
		{"http://example.com/", false, true},
		{"mailto:someone@example.com", false, true},
		{"file:///etc/passwd", false, false},
		{base + "other.html", false, false},
		{"javascript:alert(1)", false, false},
	} {
		allow, external := navigation(base, c.url)
		if allow != c.allow || external != c.external {
			t.Errorf("%s: allow %v, external %v", c.url, allow, external)
		}
	}
}
