# Markitai's pinned PDF reader policy

This directory contains the published `pdf-inspector` 1.25.1 package's required
source and BCMap data. `UPSTREAM.json` records the original archive checksum,
upstream commit and every copied file's original checksum. The original MIT
license and `external/bcmaps/LICENSE` remain in place. No Cargo registry source
was modified. The workspace coordinator owns the path patch and lockfile.

The local changes are limited to fifteen upstream files:

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
- `src/lib.rs`: per-page Markdown receives a tagged PDF's structure-tree tables,
  as the whole-document conversion does. Without them a table drawn without
  rules is found only from the alignment of its text, which splits wrapped,
  vertically centred cells into broken rows. Structure roles are not passed;
  on the Chrome-printed corpus they changed nothing further. The tables' cell
  MCIDs are known to text extraction while it runs (`with_table_cells`).
- `src/extractor/mod.rs` (also listed below): the runs of two different
  structure-tree table cells are not merged into one item, which would keep
  only the first cell's MCID and leave the next cell empty (a table in the
  browser's default style, cells a few pixels apart). Other runs merge as
  before; refusing every MCID boundary turned a sidenoted essay into a false
  table.
- `src/markdown/mod.rs` (also listed below): a structure-tree table is used
  when it holds at least 80% of the text items inside its own bounds, besides
  the upstream rule of half its band, so a fully tagged table among long
  paragraphs is kept; partial tagging still falls through to geometry.
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
  glyph advances; two or more consecutive fixed-pitch lines form a block even
  at ordinary leading (AppKit prints `<pre>` without margins). A run is a
  mono-set link only when it is a bare URL, one word starting with a scheme or
  `www.`. Text after a list is separated by a blank line so it cannot read as
  the last item's lazy continuation.
  Fixed-pitch lines do not set the body size unless nothing else does.
- `src/types.rs`, `src/markdown/mod.rs` (the `classify` module is visible to
  the crate), with `src/extractor/mod.rs` and `src/markdown/postprocess.rs`:
  runs known to be fixed-pitch are not merged with proportional ones, and
  `TextLine::text_with_markup` renders them in prose as inline code spans
  (exclusive like a decoration; mono links, bare URLs and runs holding a
  backtick stay text). `text_with_formatting` is unchanged. A URL inside an
  inline code span is not turned into a link.
- `src/markdown/convert.rs` (again): a line set off only by the paragraph
  spacing around it (`find_isolated_lines`), at body size and not bold, is no
  heading candidate when it reads as running text: it ends a sentence (`.`,
  `!`, `:` and the full-width forms, after any footnote marker or closing
  quote) or starts in lowercase. Upstream scored such a line at least 0.5
  (standalone 0.2 plus isolated 0.3, plus a share for its size's rarity), so
  any one-line body paragraph of two to six words became a heading one level
  below the page's last size tier ("Memory use varies widely between
  services." under a table became `###`). A larger or bold line keeps its
  isolation, and a question mark does not count since a question is a common
  title. On the 215 Chrome- and Quartz-printed corpus PDFs this removes 35
  page-reader headings, none of which the reference or the defuddle
  expectation has.
- `src/lib.rs`, `src/extractor/mod.rs`, `src/extractor/content_stream.rs`
  (again): one load for several readings. `LoadedPdf::load_mem` loads a file
  as the one-shot readings do; its `pages_markdown` returns what
  `extract_pages_markdown_mem` returns and its
  `text_with_positions_and_rotations` what
  `extract_text_with_positions_and_rotations_mem_with_options` returns, for
  the same bytes, without loading them again. A page's text extraction is
  split into the walk of its content stream (`read_page_runs`) and the
  switches that only shape the result (`PageRuns::finish`: bold from the
  weight class, CMap coverage, and the merge of runs into items, which reads
  the table cells of `with_table_cells`). The walk depends on nothing else
  the readings vary, so a `LoadedPdf` keeps each page's runs
  (`PageRunCache`, at most 500,000 runs, only for readings that skip
  invisible text and report no coverage) and the position reading after the
  per-page Markdown finishes the kept runs under its own switches instead of
  walking the pages again; `forget_page_runs` frees them. The font CMaps are
  read once per load, with the first reading of the text. `extract_pages_markdown_mem_impl` keeps its load and
  render repairs and reads through the shared
  `extract_pages_markdown_from_doc`. `as_loaded_by_lopdf` gives the
  document to a caller that would otherwise load the same bytes with
  `lopdf::Document::load_mem`, only when the two are the same document: the
  loader read the caller's bytes unchanged (`LoadRepairs::rewrote_bytes`:
  leading bytes dropped, bare structure names fixed, container repaired),
  repaired no object after the load, and every object the cross-reference
  table lists as in use was loaded. The last test is how an object stream
  left out past the loader's 8 MB decompression bound shows, which lopdf
  alone keeps; an entry whose offset leads to no object header, or to the
  header of another loaded object, holds nothing either load could add
  (Quartz lists freed objects as in use at offset 0). Decryption is the same
  in both loads (the empty password, without a password); lopdf drops a
  decrypted file's `/Encrypt` object, so such a file is not shared either.
  Two added tests compare every reading with its one-shot function, check
  that the position reading takes kept runs and finishes them under its own
  bold-from-weight switch, and check the sharing test on a plain file, a
  Quartz-style offset-0 entry, leading bytes, an encrypted file and an
  unused 9 MB object stream; each fails on a mutation of the code it covers.

  With these changes, the isolated copy's unit tests give 1,613 passed and the
  same 21 failed (1,611 before the two tests of the last item, 1,609 before
  the two of the item before it). Each added test fails on the unmodified
  code it covers.

The page-level OCR, font decoding, repair, limits and reliability routing remain
the upstream paths. Markitai's own visibility warnings and layout agreement
checks remain enabled. The only new public API is `TextLine::text_with_markup`
and `LoadedPdf` (`load_mem`, `document`, `as_loaded_by_lopdf`,
`pages_markdown`, `text_with_positions_and_rotations`, `forget_page_runs`); no
optional runtime dependency is added.
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
