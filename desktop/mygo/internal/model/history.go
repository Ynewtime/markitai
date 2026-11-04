package model

import (
	"encoding/json"
	"errors"
	"io/fs"
	"os"
	"path/filepath"
	"slices"
	"time"
)

// HistoryLimit is how many entries History keeps.
const HistoryLimit = 500

// Entry is one finished conversion in History (desktop/DESIGN.md 4.4).
type Entry struct {
	ID              ItemID     `json:"id"`
	Source          string     `json:"source"`
	Kind            SourceKind `json:"kind"`
	Title           string     `json:"title"`
	Folder          string     `json:"folder"`
	MarkdownPath    string     `json:"markdown_path"`
	FinishedAt      time.Time  `json:"finished_at"`
	Duration        float64    `json:"duration"`
	Warnings        []string   `json:"warnings"`
	Usage           *Usage     `json:"usage"`
	Options         Options    `json:"options"`
	MarkitaiVersion string     `json:"markitai_version"`
}

// NewEntry records a Done item.
func NewEntry(it *Item, d Done, markitaiVersion string, finishedAt time.Time) Entry {
	warnings := d.Result.Warnings
	if warnings == nil {
		warnings = []string{}
	}
	return Entry{
		ID: it.ID, Source: it.Source.Value, Kind: it.Source.Kind, Title: it.Title,
		Folder: d.Folder, MarkdownPath: d.Result.MarkdownPath,
		FinishedAt: finishedAt.UTC().Truncate(time.Second), Duration: d.Result.Duration,
		Warnings: warnings, Usage: d.Result.Usage, Options: it.Options, MarkitaiVersion: markitaiVersion,
	}
}

// History is the list of finished conversions, newest first, and the file
// that keeps it. Each change writes the whole file.
type History struct {
	path    string
	Entries []Entry
}

type historyFile struct {
	Version int     `json:"version"`
	Entries []Entry `json:"entries"`
}

// LoadHistory reads path. A missing file is an empty History. A file that
// is not valid JSON, or not version 1, is renamed to
// history-unreadable-<UTC time>.json beside it and History starts empty:
// reset is the new name, which the app reports once.
func LoadHistory(path string, now time.Time) (h *History, reset string, err error) {
	h = &History{path: path, Entries: []Entry{}}
	data, err := os.ReadFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return h, "", nil
	}
	if err != nil {
		return h, "", err
	}
	var f historyFile
	if json.Unmarshal(data, &f) == nil && f.Version == 1 {
		if f.Entries != nil {
			h.Entries = f.Entries
		}
		return h, "", nil
	}
	reset = "history-unreadable-" + now.UTC().Format("20060102T150405Z") + ".json"
	if err := os.Rename(path, filepath.Join(filepath.Dir(path), reset)); err != nil {
		return h, "", err
	}
	return h, reset, nil
}

// Add puts e first, keeping at most HistoryLimit entries.
func (h *History) Add(e Entry) error {
	h.Entries = append([]Entry{e}, h.Entries...)
	if len(h.Entries) > HistoryLimit {
		h.Entries = h.Entries[:HistoryLimit]
	}
	return h.save()
}

// Remove deletes the entry with id. The files stay.
func (h *History) Remove(id ItemID) error {
	h.Entries = slices.DeleteFunc(h.Entries, func(e Entry) bool { return e.ID == id })
	return h.save()
}

// Clear empties History. The files stay.
func (h *History) Clear() error {
	h.Entries = []Entry{}
	return h.save()
}

// Entry returns the entry with id, or nil.
func (h *History) Entry(id ItemID) *Entry {
	if i := slices.IndexFunc(h.Entries, func(e Entry) bool { return e.ID == id }); i >= 0 {
		return &h.Entries[i]
	}
	return nil
}

func (h *History) save() error {
	data, err := json.MarshalIndent(historyFile{Version: 1, Entries: h.Entries}, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(h.path, append(data, '\n'))
}

// writeAtomic writes path.tmp in the same folder, flushes it and renames it
// over path, so a crash leaves the old file or the new one.
func writeAtomic(path string, data []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.OpenFile(tmp, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0o644)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err == nil {
		err = os.Rename(tmp, path)
	}
	if err != nil {
		os.Remove(tmp)
	}
	return err
}
