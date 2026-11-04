package main

import (
	"image/png"
	"os"
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/ui"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// TestDesignShots draws the app's real view at scale 2, light and dark,
// in both languages, into MARKITAI_MYGO_DESIGN_SHOTS, for the design
// review. It is skipped without that folder.
func TestDesignShots(t *testing.T) {
	dir := os.Getenv("MARKITAI_MYGO_DESIGN_SHOTS")
	if dir == "" {
		t.Skip("MARKITAI_MYGO_DESIGN_SHOTS is not set")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for _, lang := range []l10n.Lang{l10n.En, l10n.ZhHans} {
		for _, dark := range []bool{false, true} {
			theme := "light"
			if dark {
				theme = "dark"
			}
			shot := func(tt *ui.Tester, name string) {
				t.Helper()
				tt.Advance(2e9)
				tt.Frame()
				f, err := os.Create(filepath.Join(dir, string(lang)+"-"+theme+"-"+name+".png"))
				if err != nil {
					t.Fatal(err)
				}
				defer f.Close()
				if err := png.Encode(f, tt.Image()); err != nil {
					t.Fatal(err)
				}
			}
			h := start(t, lang)
			h.tt.SetScale(2)
			h.tt.SetDark(dark)
			shot(h.tt, "01-convert-empty")
			h.click(h.T("nav.history"))
			shot(h.tt, "02-history-empty")

			h.click(h.T("nav.convert"))
			h.convert(h.inHome(pdfWarnings))
			h.convert(h.inHome(pngNoText))
			h.convert(h.inHome(pdfTable))
			failed := h.item("english.png").ID
			h.startConverting(h.write("Quarterly Report Q3 2026 with a long name.md", "# Q3\n"), h.write("notes.md", "# Notes\n"),
				h.write("minutes.md", "# Minutes\n"))
			h.a.queue.Selected = failed
			shot(h.tt, "03-convert-items-failed")
			h.until("conversions", func() bool { return model.Count[model.Converting](&h.a.queue) == 0 })
			h.rest()

			h.click(h.rowLabel("policy-assets.pdf"))
			h.click(h.T("detail.markdown"))
			shot(h.tt, "04-detail-markdown")
			h.click(h.a.lang.P("state.warnings", len(h.item("policy-assets.pdf").State.(model.Done).Result.Warnings)))
			shot(h.tt, "05-warnings-open")

			h.click(h.T("nav.history"))
			h.click(h.entryLabel("policy-assets.pdf"))
			h.until("the entry", func() bool { return h.has(h.T("detail.preview")) })
			h.click(h.T("detail.markdown"))
			h.settle()
			shot(h.tt, "06-history-detail")
			h.click(h.T("history.search"))
			h.tt.Type("zzz")
			shot(h.tt, "07-history-no-match")
			h.tt.Key(ui.Cmd, ui.KeyA)
			h.tt.Key(0, ui.KeyBackspace)
			h.click(h.T("history.clear"))
			shot(h.tt, "08-clear-history-alert")
			h.tt.Key(0, ui.KeyEscape)

			h.click(h.T("nav.settings"))
			shot(h.tt, "09-settings")

			h.click(h.T("nav.convert"))
			h.tt.SetSize(800, 780)
			shot(h.tt, "10-stacked-list")
			h.click(h.rowLabel("policy-assets.pdf"))
			shot(h.tt, "11-stacked-detail")
			h.tt.SetSize(640, 480)
			h.tt.Key(0, ui.KeyEscape)
			shot(h.tt, "12-narrow-640")
			h.tt.SetSize(1200, 780)

			h.startConverting(h.write("draft.md", "# Draft\n"), h.write("plan.md", "# Plan\n"), h.write("todo.md", "# Todo\n"))
			h.a.mayQuit()
			shot(h.tt, "13-quit-alert")
			h.tt.Key(0, ui.KeyEscape)
			h.until("conversions", func() bool { return model.Count[model.Converting](&h.a.queue) == 0 })

			fatal := ui.NewTester((&app{lang: lang, desk: &fakeDesk{}, goos: "darwin", fatal: true}).view, 1200, 780)
			fatal.SetLocale(string(lang))
			fatal.SetScale(2)
			fatal.SetDark(dark)
			shot(fatal, "14-fatal-abi")
		}
	}
}
