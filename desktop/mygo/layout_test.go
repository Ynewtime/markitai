package main

import (
	"testing"
	"time"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

func (h *harness) rowLabel(title string) string { return title + ", " + h.a.subtitle(h.item(title)) }

func TestStackedLayout(t *testing.T) {
	h := start(t, l10n.En)
	h.convert(h.write("a.md", "# A\n"))
	h.convert(h.write("b.md", "# B\n"))
	h.tt.SetSize(800, 780)
	h.rest()
	if h.has(h.T("detail.copy")) || h.has(h.T("detail.empty")) {
		t.Fatal("the detail shows beside the list at 800 wide")
	}
	list := h.find(h.T("a11y.conversions"))
	if list.W < 500 {
		t.Errorf("the list is %v wide", list.W)
	}

	h.click(h.rowLabel("b.md"))
	h.rest()
	if h.a.router.Path() != pathConvertItem || !h.has(h.T("common.back")) || !h.has(h.T("detail.copy")) {
		t.Fatalf("a click did not push the detail: %s", h.a.router.Path())
	}
	h.tt.Key(0, ui.KeyEscape)
	h.rest()
	if h.a.router.Path() != pathConvert || h.item("b.md").ID != h.a.queue.Selected {
		t.Fatalf("Escape: %s", h.a.router.Path())
	}

	h.tt.Key(0, ui.KeyDown)
	h.rest()
	if h.a.router.Path() != pathConvert || h.item("a.md").ID != h.a.queue.Selected {
		t.Fatalf("Down: %s, selection moved %v", h.a.router.Path(), h.item("a.md").ID == h.a.queue.Selected)
	}
	h.tt.Key(0, ui.KeyEnter)
	h.rest()
	if h.a.router.Path() != pathConvertItem {
		t.Fatalf("Enter: %s", h.a.router.Path())
	}
	h.click(h.T("common.back"))
	h.rest()
	if h.a.router.Path() != pathConvert || h.item("a.md").ID != h.a.queue.Selected {
		t.Fatalf("Back: %s", h.a.router.Path())
	}

	h.tt.Key(0, ui.KeyEnter)
	h.rest()
	h.tt.SetSize(1200, 780)
	h.rest()
	if h.a.router.Path() != pathConvert || h.item("a.md").ID != h.a.queue.Selected || !h.has(h.T("detail.copy")) {
		t.Fatalf("widening: %s", h.a.router.Path())
	}

	h.tt.SetSize(800, 780)
	h.click(h.T("nav.history"))
	h.rest()
	h.click(h.entryLabel("a.md"))
	h.rest()
	if h.a.router.Path() != pathHistoryItem {
		t.Fatalf("a History click: %s", h.a.router.Path())
	}
	h.click(h.T("common.back"))
	h.rest()
	h.tt.Key(0, ui.KeyUp)
	h.rest()
	if h.a.router.Path() != pathHistory || h.a.historyID != h.entry("b.md").ID {
		t.Fatalf("Up in History: %s", h.a.router.Path())
	}
	h.tt.Key(0, ui.KeyEnter)
	h.rest()
	if h.a.router.Path() != pathHistoryItem {
		t.Fatalf("Enter in History: %s", h.a.router.Path())
	}
}

func TestNarrowWindow(t *testing.T) {
	h := start(t, l10n.En)
	h.tt.SetSize(640, 480)
	h.rest()
	if nav, ok := h.tt.Find(h.T("nav.convert")); !ok || nav.X+nav.W > sidebarWidth {
		t.Errorf("the sidebar gave way: %v", nav)
	}
	h.find(h.T("convert.choose"))
	h.convert(h.write("a.md", "# A\n"))
	h.find(h.T("a11y.conversions"))
}

// TestMotion follows the spike's TestRowMotionFollowsReduceMotion: a row
// removed fades while those below slide up over 150 ms, unless the
// desktop asks for less motion.
func TestMotion(t *testing.T) {
	for _, reduce := range []bool{false, true} {
		h := start(t, l10n.En)
		h.tt.SetPreferences(ui.Preferences{ReduceMotion: reduce, TextScale: 1})
		h.convert(h.write("a.md", "# A\n"), h.write("b.md", "# B\n"))
		h.rest()
		// Items added together keep their order: a.md is above b.md.
		below := h.rowLabel("b.md")
		before := h.find(below)
		if err := h.tt.RightClick(h.rowLabel("a.md")); err != nil {
			t.Fatal(err)
		}
		if err := h.tt.ChooseMenuItem(h.T("convert.remove")); err != nil {
			t.Fatal(err)
		}
		h.tt.Frame()
		if h.item("a.md") != nil {
			t.Fatal("the row stayed")
		}
		if pending := h.tt.FramePending(); pending == reduce {
			t.Errorf("reduce %v: a frame pending %v after the removal", reduce, pending)
		}
		h.tt.Advance(50 * time.Millisecond)
		h.tt.Frame()
		mid := ui.Rect{}
		if p, ok := h.tt.Find(below); ok {
			mid = p
		}
		h.tt.Advance(400 * time.Millisecond)
		h.tt.Frame()
		after := h.find(below)
		if after.Y >= before.Y {
			t.Fatalf("the row below did not move up: %v to %v", before.Y, after.Y)
		}
		moving := mid.Y > after.Y && mid.Y < before.Y
		if moving == reduce {
			t.Errorf("reduce %v: at 50 ms the row was at %v, between %v and %v: %v", reduce, mid.Y, before.Y, after.Y, moving)
		}
	}
}

// TestReduceMotionSettles checks that with Reduce Motion no change leaves
// a frame pending, but the spinner's.
func TestReduceMotionSettles(t *testing.T) {
	h := start(t, l10n.En)
	h.tt.SetPreferences(ui.Preferences{ReduceMotion: true, TextScale: 1})
	h.tt.Frame()
	h.convert(h.inHome(pdfWarnings))
	steps := []struct {
		name string
		do   func()
	}{
		{"markdown tab", func() { h.click(h.T("detail.markdown")) }},
		{"preview tab", func() { h.click(h.T("detail.preview")) }},
		{"warnings", func() {
			h.click(h.a.lang.P("state.warnings", len(h.item("policy-assets.pdf").State.(model.Done).Result.Warnings)))
		}},
		{"copy", func() { h.click(h.T("detail.copy")) }},
		{"history", func() { h.click(h.T("nav.history")) }},
		{"settings", func() { h.click(h.T("nav.settings")) }},
		{"convert", func() { h.click(h.T("nav.convert")) }},
		{"notice", func() { h.a.addPaths([]string{h.home}); h.settle() }},
		{"clear finished", func() { h.click(h.T("convert.clear_finished")) }},
	}
	for _, s := range steps {
		s.do()
		if h.tt.FramePending() {
			t.Errorf("%s: a frame is pending", s.name)
		}
	}
	h.startConverting(h.write("a.md", "# A\n"))
	h.find(h.T("state.converting"))
}
