package core

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"markitai.local/mygo/internal/testenv"
)

// load opens the library with HOME, MARKITAI_HOME and the user folders
// in a temporary folder.
func load(t *testing.T) (*Lib, string) {
	t.Helper()
	home := testenv.Isolate(t)
	l, err := Load(testenv.Library(t))
	if err != nil {
		t.Fatal(err)
	}
	return l, home
}

func TestCallsThroughTheLibrary(t *testing.T) {
	l, home := load(t)
	if v := l.Version(); !strings.Contains(v, ".") {
		t.Errorf("version %q", v)
	}
	data := filepath.Join(home, "data")
	env, err := l.PrepareEnvironment(data)
	if err != nil || env.LoginEnvironment != "inherited" {
		t.Fatalf("prepare_environment: %+v, %v", env, err)
	}
	if wd, _ := os.Getwd(); !sameFile(wd, data) {
		t.Errorf("working directory %s, want %s", wd, data)
	}
	cfg, err := l.ConfigFile()
	if err != nil || !strings.HasPrefix(cfg.Path, home) || cfg.Exists {
		t.Errorf("config_file: %+v, %v", cfg, err)
	}

	src := filepath.Join(home, "Notes.md")
	os.WriteFile(src, []byte("# Notes\n\nHello.\n"), 0o644)
	out := filepath.Join(home, "out")
	a, err := l.ClaimOutputFolder(out, src)
	b, err2 := l.ClaimOutputFolder(out, src)
	if err != nil || err2 != nil || a.Name != "Notes" || b.Name != "Notes (2)" {
		t.Fatalf("claims %+v %+v, %v %v", a, b, err, err2)
	}
	res, err := l.ConvertJSON(map[string]any{"source": src, "options": map[string]any{"output_dir": a.Path, "llm": false, "config": map[string]any{}}})
	if err != nil || !strings.Contains(string(res), "Hello.") {
		t.Fatalf("convert_json: %s, %v", res, err)
	}
	_, err = l.ConvertJSON(map[string]any{"source": filepath.Join(home, "missing.pdf"), "options": map[string]any{"config": map[string]any{}}})
	var ce *Error
	if !errors.As(err, &ce) || ce.Code != "not_found" {
		t.Errorf("a missing file: %v", err)
	}
	html, err := l.PreviewHTML("# Hi", "dark", nil, nil)
	if err != nil || !strings.Contains(html, `data-theme="dark"`) {
		t.Errorf("preview_html: %v", err)
	}
}

func sameFile(a, b string) bool {
	fa, err1 := os.Stat(a)
	fb, err2 := os.Stat(b)
	return err1 == nil && err2 == nil && os.SameFile(fa, fb)
}
