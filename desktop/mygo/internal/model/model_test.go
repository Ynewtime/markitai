package model

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

func files(names ...string) []Source {
	var s []Source
	for _, n := range names {
		s = append(s, Source{Kind: File, Value: "/in/" + n})
	}
	return s
}

func titles(q *Queue) string {
	var s []string
	for _, it := range q.Items {
		s = append(s, fmt.Sprintf("%s:%T", it.Title, it.State))
	}
	return fmt.Sprint(s)
}

func TestQueueRunsTwoOldestFirst(t *testing.T) {
	var q Queue
	now := time.Now()
	q.Add(files("a.pdf", "b.pdf"), Options{}, now)
	q.Add(files("c.pdf"), Options{}, now)
	if got := titles(&q); got != "[c.pdf:model.Queued a.pdf:model.Queued b.pdf:model.Queued]" {
		t.Fatalf("newest first, given order kept: %s", got)
	}
	if q.Item(q.Selected).Title != "a.pdf" {
		t.Errorf("selected %v, want the first item added to the empty list", q.Item(q.Selected).Title)
	}
	claimed := 0
	jobs, _ := q.Start(func(string) (string, string, string) { claimed++; return fmt.Sprint("/out/", claimed), "", "" }, "/out", now)
	if len(jobs) != 2 || q.Item(jobs[0].ID).Title != "a.pdf" || q.Item(jobs[1].ID).Title != "b.pdf" {
		t.Fatalf("jobs %+v, state %s", jobs, titles(&q))
	}
	if _, dup := q.Add(files("a.pdf"), Options{}, now); len(dup) != 1 {
		t.Error("a source converting was added again")
	}
	if q.Remove(jobs[0].ID) {
		t.Error("a converting item was removed")
	}
	q.Finish(jobs[0].ID, Failed{Code: "unsupported"})
	if next, _ := q.Start(func(string) (string, string, string) { return "/out/3", "", "" }, "/out", now); len(next) != 1 || q.Item(next[0].ID).Title != "c.pdf" {
		t.Errorf("the freed slot started %+v", next)
	}
	q.Remove(jobs[0].ID)
	if q.Item(q.Selected).Title != "b.pdf" {
		t.Errorf("selection after removing the selected row: %s", q.Item(q.Selected).Title)
	}
}

func TestClaimFailureFailsTheItem(t *testing.T) {
	var q Queue
	q.Add(files("a.pdf"), Options{}, time.Now())
	jobs, failed := q.Start(func(string) (string, string, string) { return "", "output_unavailable", "Permission denied" }, "/ro", time.Now())
	f, ok := q.Items[0].State.(Failed)
	if len(jobs) != 0 || len(failed) != 1 || !ok || f.Code != "output_unavailable" || f.Folder != "/ro" {
		t.Errorf("jobs %v, state %+v", jobs, q.Items[0].State)
	}
}

func TestRequestSendsAltAndDescOnlyWithLLM(t *testing.T) {
	b, _ := json.Marshal(Request(Source{File, "/in/a.pdf"}, "/out/a", Options{Alt: true, Desc: true, Config: ConfigBuiltin}))
	want := `{"source":"/in/a.pdf","options":{"output_dir":"/out/a","llm":false,"ocr":false,"screenshot":false,"alt":false,"desc":false,"config":{}}}`
	if string(b) != want {
		t.Errorf("request\n %s\nwant\n %s", b, want)
	}
}

func TestOutcome(t *testing.T) {
	done := Outcome("/out/a", json.RawMessage(`{"markdown":"# A","llm_markdown":"# B","output_path":"/out/a/a.md","llm_output_path":"/out/a/a.llm.md","duration":2.5,
		"warnings":["w"],"frontmatter":{"title":"  Page  "},"skip_reason":null,
		"usage":{"requests":2,"input_tokens":10,"output_tokens":5,"cost_usd":0.5,"by_model":{"m":{"cost_status":"partial"}}}}`), "", "")
	d, ok := done.(Done)
	if !ok || d.Result.Markdown != "# B" || d.Result.MarkdownPath != "/out/a/a.llm.md" || d.Result.Title != "Page" || d.Result.Usage.Cost != CostPartial {
		t.Errorf("done %+v", done)
	}
	if f, ok := Outcome("", json.RawMessage(`{"skip_reason":"exists","output_path":"x"}`), "", "").(Failed); !ok || f.Code != "skipped" || f.Message != "exists" {
		t.Errorf("a skip is Failed skipped: %+v", f)
	}
	if f := Outcome("", nil, "image_only", "m").(Failed); f.Code != "image_only" {
		t.Errorf("an error keeps its code: %+v", f)
	}
}

func TestParseURL(t *testing.T) {
	for in, want := range map[string]string{
		" example.com/a ":       "https://example.com/a",
		"http://example.com":    "http://example.com",
		"ftp://example.com":     "",
		"not a url":             "",
		"https://":              "",
		"javascript:alert(1).x": "",
	} {
		s, ok := ParseURL(in)
		if (want == "") == ok || s.Value != want {
			t.Errorf("ParseURL(%q) = %q, %v; want %q", in, s.Value, ok, want)
		}
	}
}

func TestIDsAreUUIDv4(t *testing.T) {
	seen := map[ItemID]bool{}
	for range 100 {
		id := newID()
		if len(id) != 36 || id[14] != '4' || !strings.ContainsRune("89ab", rune(id[19])) || seen[id] {
			t.Fatalf("id %s", id)
		}
		seen[id] = true
	}
}

func TestRetryQueuesBehindWaitingItemsWithNewOptions(t *testing.T) {
	var q Queue
	now := time.Now()
	q.Add(files("a.pdf"), Options{}, now)
	jobs, _ := q.Start(func(string) (string, string, string) { return "/out/a", "", "" }, "/out", now)
	q.Finish(jobs[0].ID, Failed{Code: "unsupported"})
	if q.Retry(jobs[0].ID, Options{OCR: true}) == false || !q.Item(jobs[0].ID).Options.OCR {
		t.Fatal("retry did not capture the new options")
	}
	if _, ok := q.Item(jobs[0].ID).State.(Queued); !ok {
		t.Error("retry did not queue")
	}
	if q.Retry(jobs[0].ID, Options{}) {
		t.Error("a Queued item was retried")
	}
}

func TestClearFinishedKeepsWaitingItems(t *testing.T) {
	var q Queue
	now := time.Now()
	q.Add(files("a.pdf", "b.pdf", "c.pdf"), Options{}, now)
	jobs, _ := q.Start(func(s string) (string, string, string) { return s + ".out", "", "" }, "/out", now)
	q.Finish(jobs[0].ID, Done{})
	if !q.AnyFinished() {
		t.Fatal("a Done item is not finished")
	}
	if removed := q.ClearFinished(); len(removed) != 1 || removed[0] != jobs[0].ID {
		t.Errorf("removed %v", removed)
	}
	if got := titles(&q); got != "[b.pdf:model.Converting c.pdf:model.Queued]" {
		t.Errorf("left %s", got)
	}
	if f := q.ConvertingFolders(); len(f) != 1 || f[0] != "/in/b.pdf.out" {
		t.Errorf("converting folders %v", f)
	}
	if Count[Converting](&q) != 1 || Count[Queued](&q) != 1 {
		t.Error("counts")
	}
}
