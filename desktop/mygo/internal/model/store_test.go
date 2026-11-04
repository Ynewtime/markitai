package model

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func entry(i int) Entry {
	return Entry{ID: ItemID(fmt.Sprint(i)), Source: fmt.Sprintf("/in/%d.pdf", i), Kind: File, Title: fmt.Sprintf("%d.pdf", i), Warnings: []string{}}
}

func TestHistoryKeepsTheNewest500(t *testing.T) {
	path := filepath.Join(t.TempDir(), "data", "history.json")
	h, reset, err := LoadHistory(path, time.Now())
	if err != nil || reset != "" || len(h.Entries) != 0 {
		t.Fatalf("a missing file: %v %q %d", err, reset, len(h.Entries))
	}
	for i := 1; i <= 501; i++ {
		if err := h.Add(entry(i)); err != nil {
			t.Fatal(err)
		}
	}
	again, _, err := LoadHistory(path, time.Now())
	if err != nil || len(again.Entries) != HistoryLimit || again.Entries[0].ID != "501" || again.Entries[499].ID != "2" {
		t.Fatalf("reloaded %d entries, first %v, err %v", len(again.Entries), again.Entries[0].ID, err)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Error("the temporary file stayed")
	}
	if again.Remove("501") != nil || len(again.Entries) != 499 || again.Entry("501") != nil {
		t.Error("remove")
	}
	if again.Clear() != nil || len(again.Entries) != 0 {
		t.Error("clear")
	}
	data, _ := os.ReadFile(path)
	if string(data) != "{\n  \"version\": 1,\n  \"entries\": []\n}\n" {
		t.Errorf("an empty History file reads %q", data)
	}
}

func TestHistoryFileShape(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.json")
	h, _, _ := LoadHistory(path, time.Now())
	it := &Item{ID: "6f1c2a9e-3b4d-4c55-9a8e-2f0b7d1e4c3a", Source: Source{File, "/in/Q3.pdf"}, Title: "Q3.pdf", Options: Options{LLM: true}}
	d := Done{Folder: "/out/Q3", Result: Result{MarkdownPath: "/out/Q3/Q3.pdf.llm.md", Duration: 18.6,
		Usage: &Usage{Requests: 6, InputTokens: 19870, OutputTokens: 4440, CostUSD: 0.0412, Cost: CostComplete}}}
	at := time.Date(2026, 11, 4, 10, 42, 18, 500, time.FixedZone("x", 8*3600))
	if err := h.Add(NewEntry(it, d, "2.3.2", at)); err != nil {
		t.Fatal(err)
	}
	var got map[string]any
	data, _ := os.ReadFile(path)
	json.Unmarshal(data, &got)
	e := got["entries"].([]any)[0].(map[string]any)
	want := map[string]any{
		"id": "6f1c2a9e-3b4d-4c55-9a8e-2f0b7d1e4c3a", "source": "/in/Q3.pdf", "kind": "file", "title": "Q3.pdf",
		"folder": "/out/Q3", "markdown_path": "/out/Q3/Q3.pdf.llm.md", "finished_at": "2026-11-04T02:42:18Z",
		"duration": 18.6, "markitai_version": "2.3.2",
	}
	for k, v := range want {
		if e[k] != v {
			t.Errorf("%s = %v, want %v", k, e[k], v)
		}
	}
	if w, ok := e["warnings"].([]any); !ok || len(w) != 0 {
		t.Errorf("warnings %v, want []", e["warnings"])
	}
	if u := e["usage"].(map[string]any); u["cost"] != "complete" || u["input_tokens"] != 19870.0 {
		t.Errorf("usage %v", u)
	}
	if o := e["options"].(map[string]any); o["config"] != "user" || o["llm"] != true {
		t.Errorf("options %v", o)
	}
	if len(e) != 12 {
		t.Errorf("entry has %d fields: %v", len(e), e)
	}
}

func TestUnreadableHistoryIsRenamedOnce(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "history.json")
	now := time.Date(2026, 11, 4, 2, 42, 18, 0, time.UTC)
	for _, bad := range []string{"{not json", `{"version": 2, "entries": []}`} {
		os.WriteFile(path, []byte(bad), 0o644)
		h, reset, err := LoadHistory(path, now)
		if err != nil || reset != "history-unreadable-20261104T024218Z.json" || len(h.Entries) != 0 {
			t.Fatalf("%q: reset %q, err %v", bad, reset, err)
		}
		if kept, _ := os.ReadFile(filepath.Join(dir, reset)); string(kept) != bad {
			t.Errorf("the old file holds %q", kept)
		}
		if _, reset, _ := LoadHistory(path, now); reset != "" {
			t.Error("reported twice")
		}
		os.Remove(filepath.Join(dir, reset))
	}
}

func TestSettings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config", "settings.json")
	if s := LoadSettings(path); s != DefaultSettings() || s.Zoom != 1 {
		t.Fatalf("a missing file: %+v", s)
	}
	folder := "/out"
	s := Settings{LLM: true, Alt: true, OutputFolder: &folder, Config: ConfigBuiltin, Zoom: 1.7}
	if err := s.Save(path); err != nil {
		t.Fatal(err)
	}
	data, _ := os.ReadFile(path)
	for _, key := range []string{`"conversion.llm": true`, `"conversion.ocr": false`, `"conversion.screenshot": false`, `"conversion.desc": false`,
		`"conversion.alt": true`, `"output.folder": "/out"`, `"configuration.source": "builtin"`, `"view.zoom": 1.7`} {
		if !strings.Contains(string(data), key) {
			t.Errorf("settings.json lacks %s:\n%s", key, data)
		}
	}
	got := LoadSettings(path)
	if !got.LLM || !got.Alt || *got.OutputFolder != "/out" || got.Config != ConfigBuiltin || got.Zoom != 1.7 {
		t.Errorf("reloaded %+v", got)
	}
	if o := got.Options(); !o.LLM || !o.Alt || o.Config != ConfigBuiltin {
		t.Errorf("options %+v", o)
	}
	os.WriteFile(path, []byte(`{"conversion.llm": "yes"}`), 0o644)
	if LoadSettings(path) != DefaultSettings() {
		t.Error("an unreadable file is not the defaults")
	}
	os.WriteFile(path, []byte(`{"output.folder": null, "view.zoom": 9}`), 0o644)
	if s := LoadSettings(path); s.OutputFolder != nil || s.Zoom != ZoomMax {
		t.Errorf("null folder and zoom 9: %+v", s)
	}
	for in, want := range map[float64]float64{0.3: 0.5, 1.04: 1, 1.25: 1.3, 3.2: 3} {
		if got := ClampZoom(in); got != want {
			t.Errorf("ClampZoom(%v) = %v, want %v", in, got, want)
		}
	}
}
