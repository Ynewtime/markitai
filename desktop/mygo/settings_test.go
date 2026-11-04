package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"testing"

	"markitai.local/mygo/internal/l10n"
	"markitai.local/mygo/internal/model"
)

// toggle clicks the switch of the Settings row titled key. Its title and
// the switch share the name, and Tester.Find finds the title, so the
// switch is found at the row's middle and the group's end, where Show
// ends too.
func (h *harness) toggle(key string) {
	h.t.Helper()
	title, note := h.find(h.T(key)), h.find(h.T(key+".note"))
	show := h.find(h.T("settings.output.show"))
	h.tt.ClickAt(show.X+show.W-10, (title.Y+note.Y+note.H)/2)
	h.settle()
}

func (h *harness) savedSettings() map[string]any {
	h.t.Helper()
	data, err := os.ReadFile(h.a.files.settings)
	if err != nil {
		h.t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		h.t.Fatal(err)
	}
	return m
}

func TestSettings(t *testing.T) {
	h := start(t, l10n.En)
	h.click(h.T("nav.settings"))
	h.rest()

	h.toggle("settings.desc")
	if h.a.settings.Desc {
		t.Fatal("Image descriptions turned on without AI")
	}
	h.toggle("settings.llm")
	h.toggle("settings.desc")
	h.toggle("settings.alt")
	h.toggle("settings.ocr")
	h.toggle("settings.screenshot")
	saved := h.savedSettings()
	for _, k := range []string{"conversion.llm", "conversion.desc", "conversion.alt", "conversion.ocr", "conversion.screenshot"} {
		if saved[k] != true {
			t.Errorf("%s is %v in settings.json", k, saved[k])
		}
	}
	if o := h.a.settings.Options(); !o.LLM || !o.Desc || !o.Alt || !o.OCR || !o.Screenshot {
		t.Errorf("options %+v", o)
	}

	// Below the switches the page runs past a short window, the more so
	// with a Windows path in full.
	h.tt.SetSize(1200, 560)
	h.scrollToEnd(h.T("settings.output.folder"))
	locked := filepath.Join(h.home, "Locked")
	lockFolder(t, locked)
	h.desk.folder = locked
	os.MkdirAll(h.a.files.output, 0o755)
	h.click(h.T("settings.output.change"))
	h.wantText(h.T("settings.output.unwritable"))
	if h.a.settings.OutputFolder != nil {
		t.Fatal("an unwritable folder was taken")
	}
	out := filepath.Join(h.home, "Converted")
	os.MkdirAll(out, 0o755)
	h.desk.folder = out
	h.click(h.T("settings.output.change"))
	if h.a.output() != out || h.savedSettings()["output.folder"] != out {
		t.Fatalf("output %q", h.a.output())
	}
	if starts := h.desk.starts; len(starts) != 2 || starts[0] != h.a.files.output || starts[1] != h.a.files.output {
		t.Errorf("Change… started at %q, want the output folder %q", starts, h.a.files.output)
	}
	h.wantNoText(h.T("settings.output.unwritable"))
	h.wantText(h.a.abbreviated(out))
	h.scrollToEnd(h.T("settings.output.folder"))
	h.click(h.T("settings.output.show"))
	if len(h.desk.shown) != 1 || h.desk.shown[0] != out {
		t.Errorf("shown %q", h.desk.shown)
	}
	entries, _ := os.ReadDir(out)
	if len(entries) != 0 {
		t.Errorf("the write test left %v", entries)
	}

	cfg, err := h.lib.ConfigFile()
	if err != nil || cfg.Exists {
		t.Fatalf("configuration %+v %v", cfg, err)
	}
	h.wantText(h.T("settings.config.user.missing"))
	h.click(h.T("settings.config.create"))
	if data, err := os.ReadFile(cfg.Path); err != nil || string(data) != "{}\n" {
		t.Fatalf("created %q, %v", data, err)
	}
	if len(h.desk.opened) != 1 || h.desk.opened[0] != cfg.Path {
		t.Errorf("opened %q", h.desk.opened)
	}
	h.wantText(h.a.abbreviated(cfg.Path))
	h.desk.openErr = errors.New("no app")
	h.click(h.T("settings.config.open"))
	h.wantText(h.T("error.open_failed", "name", filepath.Base(cfg.Path)))
	if len(h.desk.opened) != 2 {
		t.Errorf("opened %q", h.desk.opened)
	}

	h.click(h.T("settings.config.builtin"))
	if h.a.settings.Config != model.ConfigBuiltin || h.savedSettings()["configuration.source"] != "builtin" {
		t.Errorf("configuration source %v", h.a.settings.Config)
	}
	h.wantText(h.T("settings.about.copyright"))
	h.wantText(h.lib.Version())
}
