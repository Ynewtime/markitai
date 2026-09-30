# Markitai's pinned PDF reader policy

This directory contains the published `pdf-inspector` 1.25.1 package's required
source and BCMap data. `UPSTREAM.json` records the original archive checksum,
upstream commit and every copied file's original checksum. The original MIT
license and `external/bcmaps/LICENSE` remain in place. No Cargo registry source
was modified. The workspace coordinator owns the path patch and lockfile.

The local changes are limited to thirteen upstream files:

- `src/text_utils.rs`: page-number recognition requires a complete folio
  expression, preserving paragraphs such as `Page 42 explains the result` and
  `Page 3 of 10 contains the method`. A shared text-mode predicate distinguishes
  explicitly requested mode-3 OCR layers from mode-7 clipping-only glyphs.
- `src/markdown/postprocess.rs`: require complete folios after reader-generated
  heading/emphasis wrappers, while preserving code and HTML literal blocks.
  Update the old tests that intentionally deleted substantive `Page N` prose.
  URL linking and space/punctuation tidying skip fenced code, which is kept
  verbatim (a URL in a string literal, alignment spaces).
- `src/extractor/content_stream.rs`: keep the graphics-state rendering mode
  across text objects; reject invalid `Tr` values; suppress nonpainting runs
  without changing their cursor advances. All-nonpainting `ActualText` glyphs
  cannot reintroduce their replacement text.
- `src/extractor/xobjects.rs`: apply the same mode and cursor rules inside Form
  XObjects, preserving inherited state and nested graphics-state restoration.
- `src/extractor/layout.rs`: an isolated edge number that shares its baseline
  with other text is kept as content (a printed table's last row) unless a
  folio-like edge number in the same band corroborates it: another page with the
  same value or value per page step of one or two, or a same-page neighbour one
  apart on its baseline (spreads). Explicit folios and isolated numbers are the
  only evidence; baseline lookups use a per-page index. Page-selected extraction
  fetches document context for such numbers. The added unit test fails on the
  unmodified file and on a variant without the spread rules. Run in an isolated copy with
  optional and dev dependencies removed, the crate's unit tests gave 1,596
  passed and 21 failed; the same 21 fail without this change (17 need fixture
  files outside the copied scope, 4 assert upstream behavior).
- `src/extractor/fonts.rs`: a Type3 font's mirrored `FontMatrix` flips its
  runs' glyph side only when its glyphs are drawn above the glyph-space origin
  (`FontBBox`). Chrome/Skia mirrors the matrix over glyphs drawn y-down, so its
  runs stood one font size below their baseline and interleaved with the
  embedded fonts of the same line; dvips/PK fonts keep the old handling.
- `src/extractor/mod.rs`, `src/extractor/scripts.rs`: a run 5% or more smaller
  or larger and at least 0.2 em off the baseline is not merged into its
  neighbour, and a run clearly off the baseline (0.2 em) is a script up to 0.86
  of the anchor size: browsers print `<sup>`/`<sub>` at 0.83 em.
- `src/markdown/convert.rs`, `src/markdown/classify.rs`,
  `src/markdown/analysis.rs`: a fixed-pitch line set off from the text above
  by more than 1.7 em opens a code block even when the page's median-based
  paragraph threshold kept the paragraph open; inside a block a URL line is
  code, blank lines are restored from the block's line pitch, the first run's
  leading spaces are kept and fixed-pitch runs are indented by their offset in
  glyph advances. A run is a mono-set link only when it is a bare URL.
  Fixed-pitch lines do not set the body size unless nothing else does.
- `src/types.rs`, `src/markdown/mod.rs` (the `classify` module is visible to
  the crate), with `src/extractor/mod.rs` and `src/markdown/postprocess.rs`:
  runs known to be fixed-pitch are not merged with proportional ones, and
  `TextLine::text_with_markup` renders them in prose as inline code spans
  (exclusive like a decoration; mono links, bare URLs and runs holding a
  backtick stay text). `text_with_formatting` is unchanged. A URL inside an
  inline code span is not turned into a link.

  With these changes, the isolated copy's unit tests give 1,606 passed and the
  same 21 failed. Each added test fails on the unmodified code it covers.

The page-level OCR, font decoding, repair, limits and reliability routing remain
the upstream paths. Markitai's own visibility warnings and layout agreement
checks remain enabled. The only new public API is `TextLine::text_with_markup`; no optional runtime
dependency is added.
Opacity, masks, occlusion, full text clipping and mixed-visibility marked content
are not claimed to be solved by this patch.

Markitai authors its own complete PDF regressions in
`crates/markitai-core/src/formats/pdf/policy_tests.rs`: two/40 pages containing
72/1,440 paragraphs; actual folios; graphics-state nesting; all four text-show
operators and preserved glyph positions; inherited nested Forms; hidden
`ActualText`; and failure of the plain-text fallback for invisible-only pages.
The old archived reproduction is retained independently in project validation
evidence. Build and test results are recorded only after the coordinator runs
the integrated gates.
