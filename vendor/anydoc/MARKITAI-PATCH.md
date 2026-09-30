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
- `src/model/mod.rs`, `src/formats/odf/mod.rs`, `src/formats/ppt/mod.rs`, and
  the other places that build a `Document` (`docx`, `doc`, `rtf`, `pptx`, the
  Markdown renderer's tests): `Document` gains `slide_starts`, the index in
  `blocks` where each slide of an ODP or a legacy PPT begins, so a consumer can
  mark slide boundaries (upstream joined the slides into one run of blocks).
  A blank slide keeps its entry: ODP records every `draw:page`, and PPT now
  keeps an empty slide's segment, so its speaker notes also stay with it
  instead of falling to the end. A PPT read in raw stream order (unusable
  persist directory) has no slide boundaries and leaves the field empty, as do
  all other formats, including PPTX. Notes without a resolvable owner still
  come last, inside the last slide's range.

Tests covering these changes were added beside the upstream ones; the upstream
suite passes in an isolated copy (291 tests).
