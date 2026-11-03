package main

import (
	_ "embed"
	"html/template"
	"strings"
)

//go:embed status.html
var statusSource string

var statusPage = template.Must(template.New("status").Parse(statusSource))

// page is what the window shows for a phase: the workbench at its own origin
// (URL), or a status document (HTML). Both empty: leave the window as it is.
type page struct {
	URL  string
	HTML string
}

// status is the template's data. Phase and Cause become data-phase and
// data-cause on <body>, which scripts/desktop_e2e.py reads.
type status struct {
	Lang    string
	Phase   string
	Cause   string
	Title   string
	Body    []string
	Tail    string
	Buttons []button
}

type button struct {
	Label string
	// Href is a markitai-desktop: request (route in window.go); template.URL,
	// as html/template rewrites other schemes than http(s) and mailto.
	Href    template.URL
	Primary bool
}

// render maps a phase to what the window shows.
func render(p Phase, t Texts) page {
	s := status{Lang: "en"}
	if t.Lang == "zh" {
		s.Lang = "zh-CN"
	}
	retry := button{Label: t.Retry, Href: openRetry, Primary: true}
	showLog := button{Label: t.ShowLog, Href: openShowLog}
	switch p := p.(type) {
	case Ready:
		return page{URL: p.URL()}
	case Stopped:
		return page{}
	case Starting:
		s.Phase, s.Title = "starting", t.Starting
	case Stopping:
		s.Phase, s.Title = "stopping", t.StoppingNow
		if p.Mode == WhenDone {
			s.Title, s.Body = t.StoppingTitle, []string{t.StoppingBody}
			s.Buttons = []button{{Label: t.StopNow, Href: openStopNow}}
		}
	case Failed:
		s.Phase, s.Tail = "failed", p.Tail
		switch c := p.Cause.(type) {
		case Exited:
			s.Cause, s.Title = "exited", t.ExitedTitle
			s.Body = []string{t.ExitedBody(c.Code), t.ExitedHint}
			s.Buttons = []button{retry, showLog}
		case NotReady:
			s.Cause, s.Title = "not-ready", t.NotReadyTitle
			s.Body = []string{t.NotReadyBody(int(readyTimeout.Seconds()))}
			s.Buttons = []button{retry, showLog}
		case NoProgram:
			s.Cause, s.Title, s.Body = "no-program", t.NoProgramTitle, []string{t.NoProgramBody(c.Path)}
			s.Buttons = []button{retry}
		case SpawnFailed:
			s.Cause, s.Title, s.Body = "spawn-failed", t.SpawnTitle, []string{t.SpawnBody(c.Err)}
			s.Buttons = []button{retry}
		}
	}
	var b strings.Builder
	if err := statusPage.Execute(&b, s); err != nil {
		panic(err)
	}
	return page{HTML: b.String()}
}
