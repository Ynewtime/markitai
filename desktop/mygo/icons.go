package main

import "github.com/egoist/mygo/ui"

// glyph is a 24-unit stroke icon in the text's color (section 5.5).
func glyph(shapes string) *ui.SVG {
	return ui.MustParseSVG([]byte(`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">` + shapes + `</svg>`))
}

const page = `<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5"/>`

var (
	iconConvert  = glyph(page + `<path d="M9 13h6M9 17h6"/>`)
	iconHistory  = glyph(`<path d="M3 12a9 9 0 1 0 2.6-6.4L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>`)
	iconSettings = glyph(`<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M2 12h3M19 12h3M4.9 19.1L7 17M17 7l2.1-2.1"/>`)
	iconAddFiles = glyph(page + `<path d="M12 11v6M9 14l3 3 3-3"/>`)
	iconLink     = glyph(`<path d="M10 13a5 5 0 0 0 7.5.5l3-3a5 5 0 0 0-7-7l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.5-.5l-3 3a5 5 0 0 0 7 7l1.7-1.7"/>`)
	iconFile     = glyph(page)
	iconGlobe    = glyph(`<circle cx="12" cy="12" r="9"/><path d="M3 12h18"/><path d="M12 3a14 14 0 0 1 0 18a14 14 0 0 1 0-18"/>`)
	iconQueued   = glyph(`<path d="M6 3h12M6 21h12"/><path d="M7 3v2.5a5 5 0 0 0 2.2 4.1L12 12l2.8-2.4A5 5 0 0 0 17 5.5V3"/><path d="M7 21v-2.5a5 5 0 0 1 2.2-4.1L12 12l2.8 2.4a5 5 0 0 1 2.2 4.1V21"/>`)
	iconDone     = glyph(`<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.5 2.5L16 9.5"/>`)
	iconWarning  = glyph(`<path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>`)
	iconFailed   = glyph(`<path d="M7.9 2h8.2L22 7.9v8.2L16.1 22H7.9L2 16.1V7.9z"/><path d="M15 9l-6 6M9 9l6 6"/>`)
	iconCopy     = glyph(`<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>`)
	iconCheck    = glyph(`<path d="M5 12.5l4.5 4.5L19 7.5"/>`)
	iconSaveAs   = glyph(`<path d="M12 3v12M7 10l5 5 5-5"/><path d="M5 21h14"/>`)
	iconFolder   = glyph(`<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>`)
	iconClear    = glyph(`<circle cx="12" cy="12" r="9"/><path d="M15 9l-6 6M9 9l6 6"/>`)
	iconChevron  = glyph(`<path d="M9 6l6 6-6 6"/>`)
	iconTrash    = glyph(`<path d="M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3"/>`)
)
