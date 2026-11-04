// Package model holds what Markitai Mygo keeps and decides without a
// window: the Convert list and its queue (desktop/DESIGN.md 4.1), History
// (4.4) and Settings (4.5), with their files. It knows nothing of the UI
// or of the core's library: the app feeds it outcomes.
package model

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// SourceKind tells a local file from a web page. Its JSON is History's
// "kind": "file" or "url".
type SourceKind uint8

const (
	File SourceKind = iota + 1
	URL
)

func (k SourceKind) MarshalText() ([]byte, error) {
	switch k {
	case File:
		return []byte("file"), nil
	case URL:
		return []byte("url"), nil
	}
	return nil, fmt.Errorf("model: source kind %d", k)
}

func (k *SourceKind) UnmarshalText(b []byte) error {
	switch string(b) {
	case "file":
		*k = File
	case "url":
		*k = URL
	default:
		return fmt.Errorf("model: source kind %q", b)
	}
	return nil
}

// Source is what an item converts: an absolute path or an http(s) URL.
type Source struct {
	Kind  SourceKind
	Value string
}

// ConfigSource is configuration.source: the user's markitai configuration,
// or the core's built-in defaults. Its JSON is "user" or "builtin".
type ConfigSource uint8

const (
	ConfigUser ConfigSource = iota
	ConfigBuiltin
)

func (c ConfigSource) MarshalText() ([]byte, error) {
	if c == ConfigBuiltin {
		return []byte("builtin"), nil
	}
	return []byte("user"), nil
}

func (c *ConfigSource) UnmarshalText(b []byte) error {
	switch string(b) {
	case "user":
		*c = ConfigUser
	case "builtin":
		*c = ConfigBuiltin
	default:
		return fmt.Errorf("model: configuration source %q", b)
	}
	return nil
}

// Options are the settings an item captured when it was added or retried.
// Their JSON is a History entry's "options".
type Options struct {
	LLM        bool         `json:"llm"`
	OCR        bool         `json:"ocr"`
	Screenshot bool         `json:"screenshot"`
	Alt        bool         `json:"alt"`
	Desc       bool         `json:"desc"`
	Config     ConfigSource `json:"config"`
}

// ItemID is an item's UUID v4. A History entry keeps it as its id.
type ItemID string

func newID() ItemID {
	var b [16]byte
	rand.Read(b[:])
	b[6] = b[6]&0x0f | 0x40
	b[8] = b[8]&0x3f | 0x80
	return ItemID(fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]))
}

// State is one of Queued, Converting, Done and Failed.
type State interface{ state() }

type Queued struct{}

type Converting struct {
	StartedAt time.Time
	Folder    string
}

type Done struct {
	Folder string
	Result Result
}

type Failed struct {
	Code, Message string
	// Folder is the output folder no folder could be claimed in, for
	// output_unavailable's sentence.
	Folder string
}

func (Queued) state()     {}
func (Converting) state() {}
func (Done) state()       {}
func (Failed) state()     {}

// Item is one source in the Convert list.
type Item struct {
	ID      ItemID
	Source  Source
	Title   string
	AddedAt time.Time
	Options Options
	State   State
	// seq orders items by when they were added, and given: the queue
	// starts the lowest first.
	seq uint64
}

// Finished reports Done or Failed.
func (it *Item) Finished() bool {
	switch it.State.(type) {
	case Done, Failed:
		return true
	}
	return false
}

// CostStatus is how much of the AI cost the price catalog covers (4.2).
type CostStatus string

const (
	CostComplete CostStatus = "complete"
	CostPartial  CostStatus = "partial"
	CostUnknown  CostStatus = "unknown"
)

// Usage is the AI usage of a conversion that asked a model at least once.
// Its JSON is a History entry's "usage".
type Usage struct {
	Requests     int64      `json:"requests"`
	InputTokens  int64      `json:"input_tokens"`
	OutputTokens int64      `json:"output_tokens"`
	CostUSD      float64    `json:"cost_usd"`
	Cost         CostStatus `json:"cost"`
}

// Result holds the fields of a conversion's result the app uses (4.2).
type Result struct {
	Markdown     string
	MarkdownPath string
	Duration     float64
	Warnings     []string
	Usage        *Usage
	Title        string
}

// Job is a conversion to run off the UI thread: the item went Converting
// with its folder claimed.
type Job struct {
	ID      ItemID
	Folder  string
	Request ConvertRequest
}

// ConvertRequest is convert_json's request (4.1 step 2).
type ConvertRequest struct {
	Source  string         `json:"source"`
	Options RequestOptions `json:"options"`
}

type RequestOptions struct {
	OutputDir  string    `json:"output_dir"`
	LLM        bool      `json:"llm"`
	OCR        bool      `json:"ocr"`
	Screenshot bool      `json:"screenshot"`
	Alt        bool      `json:"alt"`
	Desc       bool      `json:"desc"`
	Config     *struct{} `json:"config,omitempty"`
}

// Request builds the request: alt and desc only with llm, config omitted
// for the user's configuration and {} for built-in defaults.
func Request(src Source, folder string, o Options) ConvertRequest {
	r := ConvertRequest{Source: src.Value, Options: RequestOptions{
		OutputDir: folder, LLM: o.LLM, OCR: o.OCR, Screenshot: o.Screenshot,
		Alt: o.LLM && o.Alt, Desc: o.LLM && o.Desc,
	}}
	if o.Config == ConfigBuiltin {
		r.Options.Config = &struct{}{}
	}
	return r
}

// Outcome maps a convert_json result, or its error code and message, to
// Done or Failed (4.1 step 4).
func Outcome(folder string, result json.RawMessage, code, message string) State {
	if code != "" {
		return Failed{Code: code, Message: message}
	}
	var r struct {
		Markdown      string          `json:"markdown"`
		LLMMarkdown   *string         `json:"llm_markdown"`
		OutputPath    string          `json:"output_path"`
		LLMOutputPath string          `json:"llm_output_path"`
		SkipReason    *string         `json:"skip_reason"`
		Duration      float64         `json:"duration"`
		Warnings      []string        `json:"warnings"`
		Frontmatter   json.RawMessage `json:"frontmatter"`
		Usage         struct {
			Requests     int64           `json:"requests"`
			InputTokens  int64           `json:"input_tokens"`
			OutputTokens int64           `json:"output_tokens"`
			CostUSD      float64         `json:"cost_usd"`
			ByModel      json.RawMessage `json:"by_model"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(result, &r); err != nil {
		return Failed{Code: "invalid_json", Message: err.Error()}
	}
	if r.SkipReason != nil {
		return Failed{Code: "skipped", Message: *r.SkipReason}
	}
	path := r.LLMOutputPath
	if path == "" {
		path = r.OutputPath
	}
	if path == "" {
		return Failed{Code: "internal_error", Message: "The conversion finished without an output file."}
	}
	md := r.Markdown
	if r.LLMMarkdown != nil {
		md = *r.LLMMarkdown
	}
	if r.Warnings == nil {
		r.Warnings = []string{}
	}
	res := Result{Markdown: md, MarkdownPath: path, Duration: r.Duration, Warnings: r.Warnings, Title: frontmatterTitle(r.Frontmatter)}
	if r.Usage.Requests > 0 {
		res.Usage = &Usage{Requests: r.Usage.Requests, InputTokens: r.Usage.InputTokens, OutputTokens: r.Usage.OutputTokens,
			CostUSD: r.Usage.CostUSD, Cost: costStatus(r.Usage.ByModel, r.Usage.CostUSD)}
	}
	return Done{Folder: folder, Result: res}
}

func costStatus(byModel json.RawMessage, cost float64) CostStatus {
	type row struct {
		CostStatus string `json:"cost_status"`
	}
	var rows []row
	if json.Unmarshal(byModel, &rows) != nil {
		var asMap map[string]row
		if json.Unmarshal(byModel, &asMap) == nil {
			for _, r := range asMap {
				rows = append(rows, r)
			}
		}
	}
	for _, r := range rows {
		if r.CostStatus != "complete" {
			if cost > 0 {
				return CostPartial
			}
			return CostUnknown
		}
	}
	return CostComplete
}

func frontmatterTitle(raw json.RawMessage) string {
	var fm struct {
		Title any `json:"title"`
	}
	if json.Unmarshal(raw, &fm) != nil {
		return ""
	}
	s, _ := fm.Title.(string)
	s = strings.TrimSpace(s)
	if r := []rune(s); len(r) > 200 {
		s = string(r[:200])
	}
	return s
}

// ParseURL applies the URL field's rule (7.1): trim, prefix https:// to a
// text with no scheme and a dot, and accept only http and https with a
// host.
func ParseURL(text string) (Source, bool) {
	text = strings.TrimSpace(text)
	if !strings.Contains(text, "://") && strings.Contains(text, ".") {
		text = "https://" + text
	}
	u, err := url.Parse(text)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return Source{}, false
	}
	return Source{Kind: URL, Value: text}, true
}

// FileSources turns dropped, chosen or opened paths into sources, and
// names the folders it refuses: every folder but a Numbers document
// (7.1).
func FileSources(paths []string) (sources []Source, folders []string) {
	for _, p := range paths {
		abs, err := filepath.Abs(p)
		if err != nil {
			abs = p
		}
		if fi, err := os.Stat(abs); err == nil && fi.IsDir() && !strings.EqualFold(filepath.Ext(abs), ".numbers") {
			folders = append(folders, filepath.Base(abs))
			continue
		}
		sources = append(sources, Source{Kind: File, Value: abs})
	}
	return sources, folders
}

// Title is a source's title before it is Done: a file's name, a URL
// without its scheme.
func (s Source) Title() string {
	if s.Kind == File {
		return filepath.Base(s.Value)
	}
	if _, rest, ok := strings.Cut(s.Value, "://"); ok {
		return rest
	}
	return s.Value
}
