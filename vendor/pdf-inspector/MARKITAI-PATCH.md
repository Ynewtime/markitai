# Markitai's pinned PDF reader policy

This directory contains the published `pdf-inspector` 1.25.1 package's required
source and BCMap data. `UPSTREAM.json` records the original archive checksum,
upstream commit and every copied file's original checksum. The original MIT
license and `external/bcmaps/LICENSE` remain in place. No Cargo registry source
was modified. The workspace coordinator owns the path patch and lockfile.

The local changes are limited to five upstream files:

- `src/text_utils.rs`: page-number recognition requires a complete folio
  expression, preserving paragraphs such as `Page 42 explains the result` and
  `Page 3 of 10 contains the method`. A shared text-mode predicate distinguishes
  explicitly requested mode-3 OCR layers from mode-7 clipping-only glyphs.
- `src/markdown/postprocess.rs`: require complete folios after reader-generated
  heading/emphasis wrappers, while preserving code and HTML literal blocks.
  Update the old tests that intentionally deleted substantive `Page N` prose.
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
  optional and dev dependencies removed, the crate's unit tests give 1,596
  passed and 21 failed; the same 21 fail without this change (17 need fixture
  files outside the copied scope, 4 assert upstream behavior).

The page-level OCR, font decoding, repair, limits and reliability routing remain
the upstream paths. Markitai's own visibility warnings and layout agreement
checks remain enabled. No new public API or optional runtime dependency is added.
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
