package main

import (
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
		{name: "a drain offers to stop now", phase: Stopping{Mode: WhenDone, Child: c}, texts: en,
			has: []string{`data-phase="stopping"`, `href="markitai-desktop:stop-now"`}},
		{name: "stopping now offers nothing", phase: Stopping{Mode: Now, Child: c}, texts: en,
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

func TestRoute(t *testing.T) {
	cases := map[string]request{
		"markitai-desktop:print":    reqPrint,
		"markitai-desktop:retry":    reqRetry,
		"markitai-desktop:stop-now": reqStopNow,
		"markitai-desktop:show-log": reqShowLog,
		"markitai-desktop:other":    reqDenied,
		"https://markitai.dev/":     reqExternal,
		"http://example.com/a?b=c":  reqExternal,
		"javascript:alert(1)":       reqDenied,
		"file:///etc/passwd":        reqDenied,
		"mygo://localhost/":         reqDenied,
		"about:blank":               reqDenied,
		"https:":                    reqDenied,
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
