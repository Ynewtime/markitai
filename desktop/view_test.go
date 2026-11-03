package main

import (
	"regexp"
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
			has: []string{`data-phase="starting"`, "Starting Markitai"}, hasNot: []string{"markitai-desktop:"}},
		{name: "starting", phase: Starting{Child: c}, texts: en,
			has: []string{`data-phase="starting"`, "Starting Markitai"}, hasNot: []string{"markitai-desktop:"}},
		{name: "an exit shows the tail, retry and the log", phase: Failed{Cause: Exited{Code: 1}, Tail: "home is busy <b>"},
			texts: en, has: []string{`data-phase="failed"`, `data-cause="exited"`, "exited with code 1",
				"home is busy &lt;b&gt;", `href="markitai-desktop:retry"`, `href="markitai-desktop:show-log"`,
				"finishing conversions"}},
		{name: "not ready", phase: Failed{Cause: NotReady{}}, texts: en,
			has: []string{`data-cause="not-ready"`, "60 seconds", `href="markitai-desktop:retry"`}, hasNot: []string{"<pre>"}},
		{name: "no program", phase: Failed{Cause: NoProgram{Path: "/a/bin/markitai"}}, texts: en,
			has: []string{`data-cause="no-program"`, "/a/bin/markitai"}},
		{name: "spawn failed", phase: Failed{Cause: SpawnFailed{Err: "permission denied"}}, texts: en,
			has: []string{`data-cause="spawn-failed"`, "permission denied", `href="markitai-desktop:retry"`}},
		{name: "a drain offers to stop now", phase: Stopping{Mode: WhenDone, PID: 7}, texts: en,
			has: []string{`data-phase="stopping"`, `href="markitai-desktop:stop-now"`}},
		{name: "stopping now offers nothing", phase: Stopping{Mode: Now, PID: 7}, texts: en,
			has: []string{`data-phase="stopping"`}, hasNot: []string{"markitai-desktop:"}},
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
	hrefs := regexp.MustCompile(`href="(markitai-desktop:[a-z-]+)"`)
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
		if pg.HTML != "" {
			// paint loads a status page with no base URL.
			address = "about:blank"
		}
		if !stays(p, address) {
			t.Errorf("%#v refuses its own page at %q", p, address)
		}
	}
}

func TestRoute(t *testing.T) {
	cases := map[string]request{
		"markitai-desktop:print":     reqPrint,
		"markitai-desktop:retry":     reqRetry,
		"markitai-desktop:stop-now":  reqStopNow,
		"markitai-desktop:show-log":  reqShowLog,
		"markitai-desktop:other":     reqDenied,
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

func TestFileName(t *testing.T) {
	cases := map[string]string{
		"Report Q3":          "Report Q3",
		"a/b\\c:d*e?f\"g<h>": "a-b-c-d-e-f-g-h-",
		" .hidden. ":         "hidden",
		"":                   "Markitai",
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
