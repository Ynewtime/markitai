// Package l10n says the app's copy, desktop/strings.json, in English or
// Simplified Chinese, with the formats of desktop/DESIGN.md 4.2 and 6.5.
package l10n

import (
	"embed"
	"encoding/json"
	"fmt"
	"io/fs"
	"math"
	"slices"
	"strconv"
	"strings"
	"time"

	"github.com/egoist/mygo/i18n"
)

// strings.json is a copy of desktop/strings.json: go:embed cannot reach
// outside the module. TestCopyMatchesDesktopStrings and build.py --check
// keep the two equal.
//
//go:embed strings.json
var catalogJSON []byte

var catalog = func() map[string]map[string]string {
	var m map[string]map[string]string
	if err := json.Unmarshal(catalogJSON, &m); err != nil {
		panic(err)
	}
	return m
}()

// mygo/ changes mygo's own strings, by keys "mygo.<key>": in Simplified
// Chinese, the app menu's Hide and Quit items with a space before the
// app's name, as its About item (menu.about.mygo) and macOS's own menus
// have; mygo's zh-Hans.json has none. en.json, the source language a
// catalog needs, changes nothing.
//
//go:embed mygo/*.json
var mygoText embed.FS

// ChangeMygoText makes mygo say its strings as mygo/ changes them. main
// calls it before it makes the menus.
func ChangeMygoText() {
	files, err := fs.Sub(mygoText, "mygo")
	if err != nil {
		panic(err)
	}
	c, err := i18n.NewCatalog(files, string(En))
	if err != nil {
		panic(err)
	}
	i18n.Register(c)
}

// Keys lists every copy key, sorted.
func Keys() []string {
	keys := make([]string, 0, len(catalog))
	for k := range catalog {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// Lang is "en" or "zh-Hans".
type Lang string

const (
	En     Lang = "en"
	ZhHans Lang = "zh-Hans"
)

// Choose picks the app's language from the user's preferred languages,
// first first: zh-Hans when the first starts with zh, else English.
func Choose(preferred []string) Lang {
	if len(preferred) > 0 && strings.HasPrefix(strings.ToLower(preferred[0]), "zh") {
		return ZhHans
	}
	return En
}

// Raw is key's text with its placeholders unfilled.
func (l Lang) Raw(key string) string {
	s, ok := catalog[key][string(l)]
	if !ok {
		panic("l10n: no string " + key)
	}
	return s
}

// T says key, with {name} placeholders filled from args, given as name,
// value pairs.
func (l Lang) T(key string, args ...any) string {
	s := l.Raw(key)
	for i := 0; i+1 < len(args); i += 2 {
		s = strings.ReplaceAll(s, "{"+fmt.Sprint(args[i])+"}", fmt.Sprint(args[i+1]))
	}
	return s
}

// P says the plural pair key.one or key.other for count, with {count}.
func (l Lang) P(key string, count int, args ...any) string {
	k := key + ".other"
	if count == 1 {
		k = key + ".one"
	}
	return l.T(k, append([]any{"count", count}, args...)...)
}

// Duration writes seconds as 4.2 says: one decimal below a minute, rounded
// up to at least 0.1 s, minutes and seconds from there.
func (l Lang) Duration(seconds float64) string {
	seconds = math.Max(seconds, 0.1)
	if tenths := math.Round(seconds * 10); tenths < 600 {
		return l.T("unit.seconds", "value", strconv.FormatFloat(math.Max(tenths, 1)/10, 'f', 1, 64))
	}
	total := int(math.Round(seconds))
	return l.T("unit.minutes", "minutes", total/60, "seconds", total%60)
}

// Elapsed is m:ss.
func Elapsed(seconds int) string { return fmt.Sprintf("%d:%02d", seconds/60, seconds%60) }

// Grouped writes a count with the locale's digit grouping.
func (l Lang) Grouped(n int64) string {
	return i18n.FormatNumber(string(l), n, i18n.NumberOptions{Integer: true})
}

// USD writes four decimals below $1, two from $1.
func USD(cost float64) string {
	if cost < 1 {
		return fmt.Sprintf("$%.4f", cost)
	}
	return fmt.Sprintf("$%.2f", cost)
}

// Finished writes when a History entry finished, in local time:
// date.today or date.yesterday with the short time, else the locale's
// medium date and short time (6.5).
func (l Lang) Finished(at, now time.Time) string {
	at, now = at.Local(), now.Local()
	clock := i18n.FormatDate(string(l), at, i18n.DateOptions{Time: i18n.ShortStyle})
	day := func(t time.Time) time.Time { return time.Date(t.Year(), t.Month(), t.Day(), 0, 0, 0, 0, time.Local) }
	switch day(now).Sub(day(at)).Round(time.Hour) {
	case 0:
		return l.T("date.today", "time", clock)
	case 24 * time.Hour:
		return l.T("date.yesterday", "time", clock)
	}
	return i18n.FormatDate(string(l), at, i18n.DateOptions{Date: i18n.MediumStyle, Time: i18n.ShortStyle})
}

// Action is the second button of an error block (desktop/DESIGN.md 9),
// after Retry.
type Action string

const (
	NoAction     Action = ""
	RemoveItem   Action = "convert.remove"
	OpenSettings Action = "action.open_settings"
	OpenConfig   Action = "settings.config.open"
)

// errors maps an error code to its sentence key and second button.
var errors = map[string]struct {
	sentence string
	second   Action
}{
	"not_found":           {"error.not_found", RemoveItem},
	"is_directory":        {"error.is_directory", RemoveItem},
	"no_model_configured": {"error.no_model_configured", OpenSettings},
	"invalid_input":       {"error.invalid_input", NoAction},
	"unsupported":         {"error.unsupported", RemoveItem},
	"conversion_error":    {"error.conversion_error", NoAction},
	"fetch_error":         {"error.fetch_error", NoAction},
	"config_error":        {"error.config_error", OpenConfig},
	"io_error":            {"error.io_error", OpenSettings},
	"output_unavailable":  {"error.output_unavailable", OpenSettings},
	"image_only":          {"error.image_only", OpenSettings},
	"skipped":             {"error.skipped", NoAction},
	"invalid_json":        {"error.internal", NoAction},
	"internal_error":      {"error.internal", NoAction},
}

// ErrorKey maps an error code to its sentence key and second button. Any
// other code is error.unknown with none.
func ErrorKey(code string) (sentence string, second Action) {
	if e, ok := errors[code]; ok {
		return e.sentence, e.second
	}
	return "error.unknown", NoAction
}
