package main

import (
	"os"
	"testing"
	"time"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// startConverting adds paths and leaves them Converting: their outcomes
// wait in posts until the test runs them.
func (h *harness) startConverting(paths ...string) {
	h.t.Helper()
	h.a.addPaths(paths)
	h.tt.Frame()
}

func TestQuitAsksWhileConverting(t *testing.T) {
	for _, lang := range []l10n.Lang{l10n.En, l10n.ZhHans} {
		t.Run(string(lang), func(t *testing.T) {
			h := start(t, lang)
			h.startConverting(h.write("a.md", "# A\n"))
			if h.a.mayQuit() {
				t.Fatal("the close path quit while converting")
			}
			h.tt.Frame()
			if h.desk.shows != 1 {
				t.Error("the window was not shown")
			}
			h.wantText(h.T("quit.title"))
			h.wantText(h.a.lang.P("quit.body", 1))
			if err := h.tt.Click(h.T("common.cancel")); err != nil {
				t.Fatal(err)
			}
			if h.a.asking != askingNone || h.desk.quits != 0 || h.has(h.T("quit.title")) {
				t.Fatal("Cancel quit")
			}

			if h.a.mayQuit() {
				t.Fatal("the close path quit while converting")
			}
			h.startConverting(h.write("b.md", "# B\n"), h.write("c.md", "# C\n"))
			sep := " "
			if lang == l10n.ZhHans {
				sep = ""
			}
			h.wantText(h.a.lang.P("quit.body", 2) + sep + h.T("quit.body.queued"))
			h.until("the alert to close by itself", func() bool { return h.desk.quits == 1 })
			if model.Count[model.Converting](&h.a.queue) != 0 || h.has(h.T("quit.title")) || !h.a.quitting {
				t.Error("the alert stayed after the conversions ended")
			}
		})
	}
}

func TestQuitDeletesTheClaimedFolder(t *testing.T) {
	h := start(t, l10n.En)
	h.startConverting(h.write("a.md", "# A\n"))
	folder := h.item("a.md").State.(model.Converting).Folder
	deadline := time.Now().Add(30 * time.Second)
	for len(h.posts) == 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	h.a.mayQuit()
	h.tt.Frame()
	if err := h.tt.Click(h.T("quit.confirm")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(folder); !os.IsNotExist(err) {
		t.Errorf("the claimed folder stayed: %v", err)
	}
	if h.desk.quits != 1 || !h.a.quitting || !h.a.mayQuit() {
		t.Error("Quit did not quit")
	}
}

func TestQuitWithNothingConverting(t *testing.T) {
	h := start(t, l10n.En)
	if !h.a.mayQuit() || h.a.asking != askingNone {
		t.Error("an idle app asked")
	}
}
