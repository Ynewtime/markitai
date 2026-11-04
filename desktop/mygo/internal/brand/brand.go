// Package brand holds the app icon that Settings > About draws, so no
// image is read from disk at run time.
package brand

import _ "embed"

// Icon is a copy of desktop/icon.png: go:embed cannot reach outside the
// module. TestIconMatchesDesktopIcon and build.py --check keep the two
// equal.
//
//go:embed icon.png
var Icon []byte
