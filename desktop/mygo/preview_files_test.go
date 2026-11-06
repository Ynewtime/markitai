package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

func TestPreviewFoldersServeOnlyTheirFolder(t *testing.T) {
	dir := t.TempDir()
	folder := filepath.Join(dir, "Report Q3")
	assets := filepath.Join(folder, ".markitai", "assets")
	os.MkdirAll(assets, 0o755)
	os.WriteFile(filepath.Join(assets, "a b.png"), []byte("picture"), 0o644)
	os.WriteFile(filepath.Join(dir, "secret.txt"), []byte("secret"), 0o644)
	os.Symlink(filepath.Join(dir, "secret.txt"), filepath.Join(folder, "link.txt"))

	var f previewFolders
	base := f.url(folder)
	if !strings.HasPrefix(base, previewScheme+"://") || !strings.HasSuffix(base, "/") {
		t.Fatalf("base %q", base)
	}
	if again := f.url(folder); again != base {
		t.Errorf("the same folder at %q and %q", base, again)
	}
	if other := f.url(dir); other == base {
		t.Errorf("two folders at %q", base)
	}

	get := func(url string) *httptest.ResponseRecorder {
		r, err := http.NewRequest(http.MethodGet, url, nil)
		if err != nil {
			t.Fatal(err)
		}
		w := httptest.NewRecorder()
		f.ServeHTTP(w, r)
		return w
	}
	got := get(base + ".markitai/assets/a%20b.png")
	if got.Code != http.StatusOK || got.Body.String() != "picture" {
		t.Errorf("the image answered %d %q", got.Code, got.Body)
	}
	if kind := got.Header().Get("Content-Type"); kind != "image/png" {
		t.Errorf("the image's type %q", kind)
	}
	for _, url := range []string{
		base,
		base + ".markitai/assets/",
		base + "../secret.txt",
		base + "%2e%2e/secret.txt",
		base + "link.txt",
		base + "missing.png",
		previewScheme + "://0000000000000000/secret.txt",
	} {
		if w := get(url); w.Code != http.StatusNotFound {
			t.Errorf("%s answered %d %q", url, w.Code, w.Body)
		}
	}
}

func TestPreviewReadsAFolderTheAppServes(t *testing.T) {
	h := start(t, l10n.En)
	h.a.previews.folders = &previewFolders{}
	h.convert(h.write("note.md", "# Note\n\n![local](picture.png)\n"))
	done := h.item("note.md").State.(model.Done)
	h.until("the preview", func() bool { return len(h.pane.html) > 0 })
	base := h.a.previews.folders.url(done.Folder)
	if got := h.pane.base[0]; got != base {
		t.Errorf("base %q, want %q", got, base)
	}
	html := h.pane.html[0]
	if !strings.Contains(html, "img-src "+previewScheme+": data:") {
		t.Errorf("the policy leaves out the app's files: %.400s", html)
	}
	if strings.Contains(html, "file:") {
		t.Errorf("the page still names a file URL: %.400s", html)
	}
}
