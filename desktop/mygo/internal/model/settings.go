package model

import (
	"encoding/json"
	"math"
	"os"
)

// Zoom bounds and step for view.zoom.
const (
	ZoomMin  = 0.5
	ZoomMax  = 3.0
	ZoomStep = 0.1
)

// Settings are the keys of desktop/DESIGN.md 4.5 other than window.*:
// Markitai Mygo leaves the window frame to mygo's StateKey. The file is one
// flat JSON object with those keys.
type Settings struct {
	LLM        bool `json:"conversion.llm"`
	OCR        bool `json:"conversion.ocr"`
	Screenshot bool `json:"conversion.screenshot"`
	Desc       bool `json:"conversion.desc"`
	Alt        bool `json:"conversion.alt"`
	// OutputFolder is an absolute path, nil for the platform's default.
	OutputFolder *string      `json:"output.folder"`
	Config       ConfigSource `json:"configuration.source"`
	Zoom         float64      `json:"view.zoom"`
}

// DefaultSettings are the settings of a missing or unreadable file.
func DefaultSettings() Settings { return Settings{Zoom: 1} }

// LoadSettings reads path. A missing or unreadable file, or a key that
// does not parse, gives the defaults.
func LoadSettings(path string) Settings {
	s := DefaultSettings()
	data, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(data, &s) != nil {
		return DefaultSettings()
	}
	s.Zoom = ClampZoom(s.Zoom)
	if s.OutputFolder != nil && *s.OutputFolder == "" {
		s.OutputFolder = nil
	}
	return s
}

// Save writes the settings to path with the temporary-file-and-rename
// rule.
func (s Settings) Save(path string) error {
	data, err := json.MarshalIndent(s, "", "  ")
	if err != nil {
		return err
	}
	return writeAtomic(path, append(data, '\n'))
}

// Options are what an item added now captures.
func (s Settings) Options() Options {
	return Options{LLM: s.LLM, OCR: s.OCR, Screenshot: s.Screenshot, Alt: s.Alt, Desc: s.Desc, Config: s.Config}
}

// ClampZoom keeps z in 0.5 to 3.0, on the 0.1 grid; 1 when z is not a
// number.
func ClampZoom(z float64) float64 {
	if math.IsNaN(z) || z == 0 {
		return 1
	}
	return math.Round(min(max(z, ZoomMin), ZoomMax)*10) / 10
}
