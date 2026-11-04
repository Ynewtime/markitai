package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo"
)

func TestKeysAreTheMenuBarsShortcuts(t *testing.T) {
	if got := keys("darwin"); got != nil {
		t.Errorf("macOS takes %v; its menu bar has them", got)
	}
	for goos, quits := range map[string]bool{"linux": true, "windows": false} {
		k := keys(goos)
		for key, want := range map[string]key{"=": keyZoomIn, "+": keyZoomIn, "-": keyZoomOut, "0": keyZoomReset, "F11": keyFullScreen} {
			if k[key] != want {
				t.Errorf("%s: %q is %q; want %q", goos, key, k[key], want)
			}
		}
		if (k["q"] == keyQuit) != quits {
			t.Errorf("%s: Ctrl+Q quits: %v; want %v", goos, k["q"] == keyQuit, quits)
		}
	}
	if got := jsKeys(keys("windows")); got != `{"=": "zoom-in", "+": "zoom-in", "-": "zoom-out", "0": "zoom-reset", "F11": "full-screen"}` {
		t.Errorf("jsKeys = %s", got)
	}
	if got := jsKeys(nil); got != "{}" {
		t.Errorf("jsKeys(nil) = %s", got)
	}
}

func TestPreloadTakesKeysOnlyWhereTheyAreListed(t *testing.T) {
	for _, goos := range []string{"darwin", "linux", "windows"} {
		s := preloadScript(goos)
		for _, want := range []string{`"MarkitaiWeb." + method`, openPrint, "markitai.service-token", "pageshow", "e.button === 3"} {
			if !strings.Contains(s, want) {
				t.Errorf("%s: the preload lacks %q", goos, want)
			}
		}
		if strings.Contains(s, `"q": "quit"`) != (goos == "linux") {
			t.Errorf("%s: the preload's Ctrl+Q is wrong:\n%s", goos, s)
		}
	}
}

func TestZoomStepsAsMacOSMenu(t *testing.T) {
	cases := []struct {
		from float64
		up   bool
		want float64
	}{
		{1, true, 1.1}, {1, false, 0.9}, {1.1, true, 1.25}, {0.25, false, 0.25}, {5, true, 5},
		{1.05, true, 1.1}, {1.05, false, 1}, {1.0000001, true, 1.1},
	}
	for _, c := range cases {
		if got := nextZoom(c.from, c.up); got != c.want {
			t.Errorf("nextZoom(%v, %v) = %v; want %v", c.from, c.up, got, c.want)
		}
	}
}

func TestContextMenuKeepsEditingAndCopy(t *testing.T) {
	for _, id := range []string{mygo.ContextMenuCut, mygo.ContextMenuPaste, mygo.ContextMenuSelectAll, mygo.ContextMenuSpellingGuess} {
		if !menuKeeps(true, false, id) {
			t.Errorf("a text field's menu drops %s", id)
		}
	}
	if !menuKeeps(false, true, mygo.ContextMenuCopy) || menuKeeps(false, true, mygo.ContextMenuSelectAll) {
		t.Error("a selection's menu is not Copy alone")
	}
	for _, id := range []string{"back", "forward", "reload", "print", "save-page", "share", "inspect", mygo.ContextMenuCopy} {
		if menuKeeps(false, false, id) {
			t.Errorf("the page's menu keeps %s", id)
		}
		if id != mygo.ContextMenuCopy && (menuKeeps(true, true, id) || menuKeeps(false, true, id)) {
			t.Errorf("a menu keeps %s", id)
		}
	}
}

func TestWindowTitleNamesMarkitaiWeb(t *testing.T) {
	for title, want := range map[string]string{
		"":                        "Markitai Web",
		"1 · Tasks · Markitai":    "1 · Tasks · Markitai Web",
		"Markitai Web":            "Markitai Web",
		"Starting · Markitai Web": "Starting · Markitai Web",
		"report.pdf":              "report.pdf · Markitai Web",
		"Markitai Native":         "Markitai Native · Markitai Web",
	} {
		if got := windowTitle(title); got != want {
			t.Errorf("windowTitle(%q) = %q; want %q", title, got, want)
		}
	}
}

func TestSaveFiltersAndExtension(t *testing.T) {
	if got := saveFilters("Report.pdf"); !slices.EqualFunc(got, []mygo.FileFilter{{Name: "PDF", Extensions: []string{"pdf"}}}, filterEqual) {
		t.Errorf("saveFilters = %v", got)
	}
	if got := saveFilters("notes.MD"); len(got) != 1 || got[0].Name != "Markdown" || got[0].Extensions[0] != "md" {
		t.Errorf("saveFilters = %v", got)
	}
	if got := saveFilters("archive.tar.xz"); len(got) != 1 || got[0].Name != "XZ" {
		t.Errorf("saveFilters = %v", got)
	}
	if got := saveFilters("README"); got != nil {
		t.Errorf("saveFilters = %v", got)
	}

	dir := t.TempDir()
	taken := filepath.Join(dir, "taken.pdf")
	if err := os.WriteFile(taken, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	answers := func(paths ...string) (func(string) (string, error), *[]string) {
		var asked []string
		return func(at string) (string, error) {
			asked = append(asked, at)
			p := paths[0]
			paths = paths[1:]
			return p, nil
		}, &asked
	}
	never := func(string) bool { t.Error("asked to replace"); return false }
	ask, _ := answers(filepath.Join(dir, "new"))
	if got := askSavePath(filepath.Join(dir, "Report.pdf"), ask, never); got != filepath.Join(dir, "new.pdf") {
		t.Errorf("a name without an extension saves as %q", got)
	}
	ask, _ = answers(filepath.Join(dir, "new.txt"))
	if got := askSavePath(filepath.Join(dir, "Report.pdf"), ask, never); got != filepath.Join(dir, "new.txt") {
		t.Errorf("a name with an extension saves as %q", got)
	}
	ask, _ = answers(filepath.Join(dir, "taken"))
	if got := askSavePath(filepath.Join(dir, "Report.pdf"), ask, func(p string) bool { return p == taken }); got != taken {
		t.Errorf("a replaced file saves as %q", got)
	}
	ask, asked := answers(filepath.Join(dir, "taken"), filepath.Join(dir, "other.pdf"))
	if got := askSavePath(filepath.Join(dir, "Report.pdf"), ask, func(string) bool { return false }); got != filepath.Join(dir, "other.pdf") {
		t.Errorf("after keeping the file it saves as %q", got)
	}
	if want := []string{filepath.Join(dir, "Report.pdf"), taken}; !slices.Equal(*asked, want) {
		t.Errorf("the panel opened at %v; want %v", *asked, want)
	}
	ask, _ = answers("")
	if got := askSavePath(filepath.Join(dir, "Report.pdf"), ask, never); got != "" {
		t.Errorf("a canceled panel saves as %q", got)
	}
}

func filterEqual(a, b mygo.FileFilter) bool {
	return a.Name == b.Name && slices.Equal(a.Extensions, b.Extensions)
}

// mygo runs each call of the preload in a goroutine of its own: a report
// of the look that comes after a newer one is dropped, the newer kept.
func TestReportsTakeTheLatestLook(t *testing.T) {
	var r reports
	for _, c := range []struct {
		doc  string
		n    int
		want bool
	}{
		{"a", 2, true}, {"a", 1, false}, {"a", 2, false}, {"a", 3, true},
		{"b", 1, true}, {"a", 4, false}, {"b", 3, true}, {"b", 2, false},
		{"c", 2, true}, {"b", 4, false}, {"a", 9, false}, {"c", 3, true},
	} {
		if got := r.newer(c.doc, c.n); got != c.want {
			t.Errorf("report %d of %s: newer = %v; want %v", c.n, c.doc, got, c.want)
		}
	}
	if !strings.Contains(preload, `call("Look", lang, theme, doc, ++looks)`) {
		t.Error("the preload numbers no report of the look")
	}
}

func TestLookKeepsTheWorkbenchsValues(t *testing.T) {
	for _, c := range []struct {
		lang, theme string
		want        Look
	}{
		{"zh", "dark", Look{"zh", "dark"}}, {"ja", "light", Look{"ja", "light"}}, {"en", "", Look{Lang: "en"}},
		{"fr", "auto", Look{}}, {"", "", Look{}}, {"zh-Hans", "Dark", Look{}},
	} {
		if got := lookOf(c.lang, c.theme); got != c.want {
			t.Errorf("lookOf(%q, %q) = %v; want %v", c.lang, c.theme, got, c.want)
		}
	}
	for tag, want := range map[string]string{"zh-Hans": "zh", "zh-CN": "zh", "ja": "ja", "ja-JP": "ja", "en": "en", "de": "en", "": "en"} {
		if got := workbenchLang(tag); got != want {
			t.Errorf("workbenchLang(%q) = %q; want %q", tag, got, want)
		}
	}
	for lang := range texts {
		if workbenchLang(locale(lang)) != lang || !slices.Contains(webLanguages, locale(lang)) {
			t.Errorf("%s does not go through mygo's languages", lang)
		}
	}
	if themeSource("dark") != mygo.ThemeDark || themeSource("light") != mygo.ThemeLight || themeSource("") != mygo.ThemeSystem {
		t.Error("themeSource")
	}
}

func TestEveryLanguageHasEveryText(t *testing.T) {
	for lang, tx := range texts {
		if tx.Lang != lang || tx.HTMLTag == "" || tx.Exists == nil || tx.ExistsDetail == "" || tx.Replace == "" || tx.ExitedBody == nil {
			t.Errorf("%s lacks texts: %+v", lang, tx)
		}
		for _, id := range editItems[:8] {
			if tx.MenuItems[id] == "" {
				t.Errorf("%s has no label for %s", lang, id)
			}
		}
	}
}

func TestQuitQuestionMarksNoDestructiveButtonOnLinux(t *testing.T) {
	if destructiveQuit("linux") != nil || !slices.Equal(destructiveQuit("windows"), []int{1}) || !slices.Equal(destructiveQuit("darwin"), []int{1}) {
		t.Error("destructiveQuit")
	}
}

func TestCodeTextWritesWindowsStatusesInHex(t *testing.T) {
	for _, c := range []struct {
		goos string
		code int
		want string
	}{
		{"windows", 1, "1"}, {"windows", 0xC0000005, "0xC0000005"}, {"windows", 4294967295, "0xFFFFFFFF"},
		{"linux", 255, "255"}, {"darwin", 1, "1"},
	} {
		if got := codeText(c.goos, c.code); got != c.want {
			t.Errorf("codeText(%s, %d) = %q; want %q", c.goos, c.code, got, c.want)
		}
	}
}

func TestUserEnvGivesTheUsersValuesBack(t *testing.T) {
	saved := appEnv
	t.Cleanup(func() { appEnv = saved })
	user := "fr_FR"
	appEnv = map[string]*string{"LANGUAGE": &user, "GTK_USE_PORTAL": nil}
	got := userEnv([]string{"PATH=/bin", "LANGUAGE=zh_CN", "GTK_USE_PORTAL=1", "HOME=/h"})
	slices.Sort(got)
	if want := []string{"HOME=/h", "LANGUAGE=fr_FR", "PATH=/bin"}; !slices.Equal(got, want) {
		t.Errorf("userEnv = %v; want %v", got, want)
	}
	env := childEnv([]string{"PATH=/bin", "LANG=en_US.UTF-8", "LANGUAGE=zh_CN", "GTK_USE_PORTAL=1"}, "t", "en", "/h")
	if value(env, "LANGUAGE") != "fr_FR" || slices.ContainsFunc(env, func(kv string) bool { return strings.HasPrefix(kv, "GTK_USE_PORTAL=") }) {
		t.Errorf("serve's environment has the app's: %v", env)
	}
}
