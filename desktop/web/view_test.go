package main

import (
	"encoding/base64"
	"encoding/json"
	"regexp"
	"runtime"
	"slices"
	"strings"
	"testing"
)

func TestRender(t *testing.T) {
	c := child{PID: 7, Port: 51234, Token: "secret"}
	cases := []struct {
		name    string
		phase   Phase
		texts   Texts
		has     []string
		hasNot  []string
		workURL string
	}{
		{name: "ready shows the workbench", phase: Ready{Child: c}, texts: en,
			workURL: "http://127.0.0.1:51234/#token=secret"},
		{name: "preparing", phase: Preparing{}, texts: en,
			has: []string{`data-phase="starting"`, "Starting Markitai Web"}, hasNot: []string{"markitai-web:"}},
		{name: "starting", phase: Starting{Child: c}, texts: en,
			has: []string{`data-phase="starting"`, "Starting Markitai Web"}, hasNot: []string{"markitai-web:"}},
		{name: "an exit shows the tail, retry and the log", phase: Failed{Cause: Exited{Code: 1}, Tail: "home is busy <b>"},
			texts: en, has: []string{`data-phase="failed"`, `data-cause="exited"`, "exited with code 1",
				"home is busy &lt;b&gt;", `href="markitai-web:retry"`, `href="markitai-web:show-log"`,
				"finishing conversions"}},
		{name: "not ready", phase: Failed{Cause: NotReady{}}, texts: en,
			has: []string{`data-cause="not-ready"`, "60 seconds", `href="markitai-web:retry"`}, hasNot: []string{"<pre>"}},
		{name: "no program", phase: Failed{Cause: NoProgram{Path: "/a/bin/markitai"}}, texts: en,
			has: []string{`data-cause="no-program"`, "/a/bin/markitai"}},
		{name: "spawn failed", phase: Failed{Cause: SpawnFailed{Err: "permission denied"}}, texts: en,
			has: []string{`data-cause="spawn-failed"`, "permission denied", `href="markitai-web:retry"`}},
		{name: "a drain offers to stop now", phase: Stopping{Mode: WhenDone, PID: 7}, texts: en,
			has: []string{`data-phase="stopping"`, `href="markitai-web:stop-now"`}},
		{name: "stopping now offers nothing", phase: Stopping{Mode: Now, PID: 7}, texts: en,
			has: []string{`data-phase="stopping"`}, hasNot: []string{"markitai-web:"}},
		{name: "Chinese", phase: Failed{Cause: Exited{Code: 2}}, texts: zh,
			has: []string{`lang="zh-CN"`, "重试", "代码 2"}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pg := render(tc.phase, tc.texts)
			if pg.URL != tc.workURL {
				t.Errorf("URL = %q; want %q", pg.URL, tc.workURL)
			}
			if tc.workURL != "" {
				if pg.HTML != "" {
					t.Error("the workbench comes with a status page")
				}
				return
			}
			for _, s := range tc.has {
				if !strings.Contains(pg.HTML, s) {
					t.Errorf("page lacks %q:\n%s", s, pg.HTML)
				}
			}
			for _, s := range tc.hasNot {
				if strings.Contains(pg.HTML, s) {
					t.Errorf("page has %q:\n%s", s, pg.HTML)
				}
			}
			if strings.Contains(pg.HTML, "secret") {
				t.Error("a status page shows the token")
			}
		})
	}
	if pg := render(Stopped{}, en); pg != (page{}) {
		t.Errorf("Stopped renders %v", pg)
	}
}

// A request is honored only where offers lists it; the status pages offer
// exactly their buttons, so the table is the pages.
func TestOffersAreThePagesButtons(t *testing.T) {
	c := child{PID: 7, Port: 51234, Token: "t"}
	phases := []Phase{Preparing{}, Preparing{Dying: 6}, Starting{Child: c}, Stopping{Mode: WhenDone, PID: 7},
		Stopping{Mode: Now, PID: 7}, Failed{Cause: Exited{Code: 1}}, Failed{Cause: NotReady{}, Dying: 7},
		Failed{Cause: NoProgram{Path: "/x"}}, Failed{Cause: SpawnFailed{Err: "e"}}}
	hrefs := regexp.MustCompile(`href="(markitai-web:[a-z-]+)"`)
	for _, p := range phases {
		var shown []request
		for _, m := range hrefs.FindAllStringSubmatch(render(p, en).HTML, -1) {
			shown = append(shown, route(m[1]))
		}
		if !slices.Equal(shown, offers(p)) {
			t.Errorf("%#v shows %v and offers %v", p, shown, offers(p))
		}
	}
	if got := offers(Ready{Child: c}); !slices.Equal(got, []request{reqPrint}) {
		t.Errorf("the workbench offers %v; want Export PDF alone", got)
	}
	for _, p := range []Phase{Ready{Child: c}, Starting{Child: c}, Stopping{Mode: Now, PID: 7}, Preparing{}} {
		for _, r := range []request{reqRetry, reqStopNow, reqShowLog} {
			if slices.Contains(offers(p), r) {
				t.Errorf("%#v honors %v, which its page does not show", p, r)
			}
		}
	}
}

func TestStays(t *testing.T) {
	ready := Ready{Child: child{PID: 7, Port: 51234, Token: "t"}}
	cases := []struct {
		phase Phase
		url   string
		want  bool
	}{
		{ready, "http://127.0.0.1:51234/", true},
		{ready, "http://127.0.0.1:51234/api/download?ticket=x", true},
		{ready, "blob:http://127.0.0.1:51234/0b5c", true},
		{ready, "http://127.0.0.1:51235/", false},
		{ready, "http://localhost:51234/", false},
		{ready, "https://example.com/", false},
		{ready, "file:///etc/passwd", false},
		{ready, "about:blank", false},
		{Failed{Cause: Exited{Code: 1}}, "http://127.0.0.1:51234/", false},
		{Starting{Child: ready.Child}, "http://127.0.0.1:51234/", false},
		{Failed{Cause: Exited{Code: 1}}, "https://example.com/", false},
		{Stopping{Mode: WhenDone, PID: 7}, "file:///etc/passwd", false},
	}
	for _, tc := range cases {
		if got := stays(tc.phase, tc.url); got != tc.want {
			t.Errorf("stays(%T, %q) = %v; want %v", tc.phase, tc.url, got, tc.want)
		}
	}
}

// The window never refuses the page render gives it. mygo asks about the
// app's own loads too when two overlap, as when serve fails at once and the
// Failed page follows the Starting page: it marks only the first load as
// the app's.
func TestStaysOnThePageItShows(t *testing.T) {
	c := child{PID: 7, Port: 51234, Token: "t"}
	phases := []Phase{Preparing{}, Preparing{Dying: 6}, Starting{Child: c}, Ready{Child: c},
		Stopping{Mode: WhenDone, PID: 7}, Stopping{Mode: Now, PID: 7}, Failed{Cause: Exited{Code: 1}},
		Failed{Cause: NotReady{}, Dying: 7}, Failed{Cause: NoProgram{Path: "/x"}}, Failed{Cause: SpawnFailed{Err: "e"}}}
	for _, p := range phases {
		pg := render(p, en)
		address := pg.URL
		switch {
		case pg.HTML != "" && runtime.GOOS == "windows":
			// WebView2 asks about HTML it is given as a data: URL.
			address = "data:text/html;charset=utf-8;base64," + base64.StdEncoding.EncodeToString([]byte(pg.HTML))
		case pg.HTML != "":
			// paint loads a status page with no base URL.
			address = "about:blank"
		}
		if !stays(p, address) {
			t.Errorf("%#v refuses its own page at %q", p, address)
		}
	}
}

// A status page loads at about:blank, which WebView2 asks about as a data:
// URL of the page's HTML; elsewhere a data: URL is no status page.
func TestAtStatusPage(t *testing.T) {
	cases := []struct {
		goos, url string
		want      bool
	}{
		{"darwin", "about:blank", true},
		{"linux", "about:blank", true},
		{"windows", "about:blank", true},
		{"windows", "data:text/html;charset=utf-8;base64,PCFkb2N0eXBlIGh0bWw+", true},
		{"darwin", "data:text/html;charset=utf-8;base64,PCFkb2N0eXBlIGh0bWw+", false},
		{"linux", "data:text/html,<p>x</p>", false},
		{"windows", "data:image/svg+xml,<svg/>", false},
		{"windows", "https://example.com/", false},
		{"windows", "file:///C:/Windows/win.ini", false},
	}
	for _, tc := range cases {
		if got := atStatusPage(tc.goos, tc.url); got != tc.want {
			t.Errorf("atStatusPage(%q, %q) = %v; want %v", tc.goos, tc.url, got, tc.want)
		}
	}
}

func TestRoute(t *testing.T) {
	cases := map[string]request{
		"markitai-web:print":         reqDenied,
		"markitai-web:retry":         reqRetry,
		"markitai-web:stop-now":      reqStopNow,
		"markitai-web:show-log":      reqShowLog,
		"markitai-web:other":         reqDenied,
		"https://markitai.dev/":      reqExternal,
		"http://example.com/a?b=c":   reqExternal,
		"mailto:someone@example.com": reqExternal,
		"mailto:":                    reqDenied,
		"javascript:alert(1)":        reqDenied,
		"file:///etc/passwd":         reqDenied,
		"mygo://localhost/":          reqDenied,
		"about:blank":                reqDenied,
		"https:":                     reqDenied,
	}
	for raw, want := range cases {
		if got := route(raw); got != want {
			t.Errorf("route(%q) = %v; want %v", raw, got, want)
		}
	}
}

// Export PDF is the workbench's alone: what the preload's window.print opens
// is honored, a link in converted content asking for it is not.
func TestOnlyThePreloadAsksForExportPDF(t *testing.T) {
	opened := regexp.MustCompile(`window\.open\("([^"]+)"\)`).FindStringSubmatch(preload)
	if opened == nil || route(opened[1]) != reqPrint {
		t.Fatalf("the preload's window.print opens %q, which is not Export PDF", opened)
	}
	key := strings.TrimPrefix(opened[1], "markitai-web:print?")
	for _, forged := range []string{"markitai-web:print", "markitai-web:print?", opened[1] + "x",
		"markitai-web:print?" + string(newToken()), "markitai-web:print?" + strings.ToUpper(key)} {
		if got := route(forged); got != reqDenied {
			t.Errorf("route(%q) = %v; want denied", forged, got)
		}
	}
}

func TestFileName(t *testing.T) {
	cases := map[string]string{
		"Report Q3":          "Report Q3",
		"a/b\\c:d*e?f\"g<h>": "a-b-c-d-e-f-g-h-",
		" .hidden. ":         "hidden",
		"":                   "Markitai Web",
		"line\nbreak":        "line-break",
		"报告":                 "报告",
	}
	for title, want := range cases {
		if got := fileName(title); got != want {
			t.Errorf("fileName(%q) = %q; want %q", title, got, want)
		}
	}
	long := fileName(strings.Repeat("长", 160))
	if len(long) > 200 || !strings.HasPrefix(long, "长") || strings.ToValidUTF8(long, "") != long {
		t.Errorf("a long Chinese title gives %d bytes", len(long))
	}
}

// On Windows the window's content rules let documents load from the
// workbench alone: the first rule blocks every document, the second undoes
// that for the workbench's origin. Go's regexp reads the WebKit subset they
// use alike.
func TestDocumentRules(t *testing.T) {
	var rules []struct {
		Trigger struct {
			URLFilter    string   `json:"url-filter"`
			ResourceType []string `json:"resource-type"`
		} `json:"trigger"`
		Action struct {
			Type string `json:"type"`
		} `json:"action"`
	}
	if err := json.Unmarshal([]byte(documentRules("http://127.0.0.1:23600")), &rules); err != nil {
		t.Fatal(err)
	}
	if len(rules) != 2 || rules[0].Action.Type != "block" || rules[1].Action.Type != "ignore-previous-rules" {
		t.Fatalf("rules %+v; want a block, then ignore-previous-rules", rules)
	}
	for _, r := range rules {
		if !slices.Equal(r.Trigger.ResourceType, []string{"document"}) {
			t.Errorf("rule %+v is not for documents alone", r)
		}
	}
	matches := func(filter, url string) bool {
		return regexp.MustCompile("(?i)" + filter).MatchString(url)
	}
	for _, url := range []string{"http://127.0.0.1:23600/", "http://127.0.0.1:57499/x", "https://example.com/"} {
		if !matches(rules[0].Trigger.URLFilter, url) {
			t.Errorf("the block misses %q", url)
		}
	}
	cases := []struct {
		url  string
		kept bool
	}{
		{"http://127.0.0.1:23600/", true},
		{"http://127.0.0.1:23600/#token=t", true},
		{"http://127.0.0.1:23600/api/download?ticket=x", true},
		{"http://127.0.0.1:57499/x", false},
		{"http://127.0.0.1:236001/", false},
		{"http://127.0.0.1:23600.example.com/", false},
		{"http://127.0.0.1:23600@example.com/", false},
		{"http://127.0.0.1:2360/", false},
		{"http://127a0a0a1:23600/", false},
		{"https://127.0.0.1:23600/", false},
		{"https://example.com/?u=http://127.0.0.1:23600/", false},
	}
	for _, tc := range cases {
		if got := matches(rules[1].Trigger.URLFilter, tc.url); got != tc.kept {
			t.Errorf("%q kept: %v; want %v", tc.url, got, tc.kept)
		}
	}
}
