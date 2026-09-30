# Markitai's pinned document reader

This directory contains the published `anydoc` 0.2.4 package's include set
(`src`, `examples/convert.rs`, `README.md`, `LICENSE`) and its package manifest.
`UPSTREAM.json` records the original archive checksum, upstream commit and every
copied file's original checksum. The original MIT license remains in place. No
Cargo registry source was modified. The workspace coordinator owns the path
patch and lockfile.

The local changes, each marked `markitai` in its comment, are limited to these
upstream files:

- `src/shared/html.rs`: a `pre` element's code block keeps the language its
  markup names, a `language-…` or `lang-…` class on the `pre` or its `code`
  child (upstream always left it empty), so EPUB code fences carry their info
  string. Other class spellings and characters outside `[A-Za-z0-9+#._-]` give
  no language.
- `src/formats/sheet/numfmt.rs`, `src/formats/sheet/mod.rs`,
  `src/formats/sheet/xlsx.rs`: a date/time format records whether it shows
  seconds, and a time of day, alone or after a date, is written `hh:mm` when
  it does not (`h:mm`, `yyyy-mm-dd hh:mm`); seconds are dropped as a
  spreadsheet displays them, not rounded into the minute. Elapsed durations
  are unchanged.

Tests covering both changes were added beside the upstream ones; the upstream
suite passes in an isolated copy (287 tests).
