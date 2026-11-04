package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// convert drops files on Convert and waits until they are Done.
func (h *harness) convert(paths ...string) {
	h.t.Helper()
	h.a.addPaths(paths)
	h.settle()
	h.until("conversions", func() bool {
		return model.Count[model.Queued](&h.a.queue)+model.Count[model.Converting](&h.a.queue) == 0
	})
	h.rest()
}

// entryLabel is a History row's name.
func (h *harness) entryLabel(title string) string {
	e := h.entry(title)
	if e == nil {
		h.t.Fatalf("no entry %q", title)
	}
	name := e.Title + ", " + h.a.lang.Finished(e.FinishedAt, h.tt.Now())
	if n := len(e.Warnings); n > 0 {
		name += " · " + h.a.lang.P("state.warnings", n)
	}
	return name
}

func (h *harness) entry(title string) *model.Entry {
	for i := range h.a.history.Entries {
		if h.a.history.Entries[i].Title == title {
			return &h.a.history.Entries[i]
		}
	}
	return nil
}

func TestHistory(t *testing.T) {
	h := start(t, l10n.En)
	alpha, beta := h.write("alpha.md", "# Alpha\n"), h.write("beta.md", "# Beta\n")
	h.convert(alpha)
	h.convert(beta)
	h.click(h.T("nav.history"))
	h.rest()
	if a, b := h.find(h.entryLabel("alpha.md")), h.find(h.entryLabel("beta.md")); b.Y >= a.Y {
		t.Errorf("beta at %v is not above alpha at %v", b.Y, a.Y)
	}
	h.wantText(h.T("detail.empty"))

	h.click(h.entryLabel("alpha.md"))
	h.until("alpha's Markdown", func() bool { return h.has(h.T("detail.preview")) })

	h.click(h.T("history.search"))
	h.tt.Type("BET")
	h.settle()
	if _, ok := h.tt.Find(h.entryLabel("alpha.md")); ok {
		t.Error("the search kept alpha")
	}
	h.find(h.entryLabel("beta.md"))
	h.wantText(h.T("detail.empty"))
	h.tt.Key(ui.Cmd, ui.KeyA)
	h.tt.Type("zzz")
	h.settle()
	h.wantText(h.T("history.no_results", "query", "zzz"))
	h.tt.Key(ui.Cmd, ui.KeyA)
	h.tt.Key(0, ui.KeyBackspace)
	h.settle()

	md := h.entry("alpha.md").MarkdownPath
	if err := h.tt.RightClick(h.entryLabel("alpha.md")); err != nil {
		t.Fatal(err)
	}
	if err := h.tt.ChooseMenuItem(h.T("history.remove")); err != nil {
		t.Fatal(err)
	}
	h.settle()
	if h.entry("alpha.md") != nil {
		t.Fatal("Remove from History kept the entry")
	}
	if _, err := os.Stat(md); err != nil {
		t.Errorf("Remove from History removed the file: %v", err)
	}

	h.click(h.T("history.clear"))
	h.wantText(h.a.lang.P("history.clear.title", 1))
	h.wantText(h.T("history.clear.body"))
	h.click(h.T("common.cancel"))
	if len(h.a.history.Entries) != 1 {
		t.Fatal("Cancel cleared History")
	}
	h.click(h.T("history.clear"))
	h.click(h.T("history.clear.confirm"))
	if len(h.a.history.Entries) != 0 {
		t.Fatal("Clear kept entries")
	}
	h.wantText(h.T("history.empty.title"))
	h.wantText(h.T("history.empty.body"))
	again, _, _ := model.LoadHistory(h.a.files.history, time.Now())
	if len(again.Entries) != 0 {
		t.Error("the file kept entries")
	}
	if _, err := os.Stat(md); err != nil {
		t.Errorf("Clear History removed the files: %v", err)
	}
}

func TestHistoryKeepsTheNewest500(t *testing.T) {
	h := startWith(t, l10n.En, func(home string, f files) {
		var entries []model.Entry
		for i := 500; i >= 1; i-- {
			entries = append(entries, model.Entry{ID: model.ItemID(fmt.Sprintf("00000000-0000-4000-8000-%012d", i)),
				Source: filepath.Join(home, fmt.Sprintf("%d.md", i)), Kind: model.File, Title: fmt.Sprintf("%d.md", i),
				Folder: home, MarkdownPath: filepath.Join(home, fmt.Sprintf("%d.md", i)), FinishedAt: time.Now().UTC().Truncate(time.Second),
				Warnings: []string{}, MarkitaiVersion: "2.3.2"})
		}
		data, _ := json.Marshal(map[string]any{"version": 1, "entries": entries})
		os.MkdirAll(filepath.Dir(f.history), 0o755)
		os.WriteFile(f.history, data, 0o644)
	})
	h.convert(h.write("new.md", "# New\n"))
	again, _, err := model.LoadHistory(h.a.files.history, time.Now())
	if err != nil || len(again.Entries) != model.HistoryLimit || again.Entries[0].Title != "new.md" || again.Entries[499].Title != "2.md" {
		t.Fatalf("%d entries, first %q, last %q, %v", len(again.Entries), again.Entries[0].Title, again.Entries[len(again.Entries)-1].Title, err)
	}
}

func TestUnreadableHistory(t *testing.T) {
	var f files
	h := startWith(t, l10n.En, func(_ string, files files) {
		f = files
		os.MkdirAll(filepath.Dir(files.history), 0o755)
		os.WriteFile(files.history, []byte("{not json"), 0o644)
	})
	matches, _ := filepath.Glob(filepath.Join(f.data, "history-unreadable-*.json"))
	if len(matches) != 1 {
		t.Fatalf("renamed files %v", matches)
	}
	h.wantText(h.T("history.reset", "name", filepath.Base(matches[0])))
	if !h.announced(h.T("history.reset", "name", filepath.Base(matches[0]))) {
		t.Error("the reset was not announced")
	}
	again := newApp(l10n.En, offline{h.lib}, h.desk, h.pane, h.a.post, f)
	if again.notice.text != "" {
		t.Errorf("the next launch reported %q", again.notice.text)
	}
}

func TestMissingMarkdown(t *testing.T) {
	h := start(t, l10n.En)
	h.convert(h.write("gone.md", "# Gone\n"))
	md := h.entry("gone.md").MarkdownPath
	if err := os.Remove(md); err != nil {
		t.Fatal(err)
	}
	h.click(h.T("nav.history"))
	h.rest()
	h.click(h.entryLabel("gone.md"))
	h.until("the missing file", func() bool { return h.has(h.T("detail.missing.title")) })
	h.wantText(h.T("detail.missing.body", "path", h.a.abbreviated(md)))
	for _, act := range []action{actCopy, actSaveAs, actShow, actToggleView} {
		if h.a.enabled(act) {
			t.Errorf("%s is enabled", act)
		}
	}
	if !h.a.enabled(actRemove) {
		t.Fatal("Remove from History is disabled")
	}
	h.tt.Key(ui.Cmd|ui.Shift, ui.KeyC)
	h.settle()
	if h.tt.Clipboard() != "" {
		t.Error("Copy Markdown copied a missing file")
	}
	h.click(h.entryLabel("gone.md"))
	mods, key := removeKey(h.a.goos)
	h.tt.Key(mods, key)
	h.settle()
	if len(h.a.history.Entries) != 0 {
		t.Error("the entry stayed")
	}
}
