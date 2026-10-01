# Markitai's pinned PDF reader policy

This directory contains the published `pdf-inspector` 1.25.1 package's required
source and BCMap data. `UPSTREAM.json` records the original archive checksum,
upstream commit and every copied file's original checksum. The original MIT
license and `external/bcmaps/LICENSE` remain in place. No Cargo registry source
was modified. The workspace coordinator owns the path patch and lockfile.

The local changes, each marked `markitai` (or, for sorts, made through
`crate::sort`), are limited to the upstream files listed here:

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
- `src/sort.rs` (new, `mod sort` in `src/lib.rs`) and 132 sort calls in
  `src/bidi.rs`, `src/text_utils.rs`,
  `src/extractor/{layout,mod,reading_order,scripts,underline,word_gaps}.rs`,
  `src/markdown/{analysis,furniture,heading,mod}.rs` and
  `src/tables/{detect_heuristic,detect_lines,detect_rects,detect_struct,grid,mod}.rs`
  (twelve of these files were otherwise unmodified): one compiled sort per
  element type. A closure is a type of its own, so every `sort_by` with a
  closure compiled the standard library's stable sort again, about 4 KB for a
  list of floats and up to 25 KB for a list of items, and once more for each
  caller of a generic function such as `sort_rtl_cell_items`.
  `crate::sort::stable(v, &mut compare)` takes the comparator as a trait
  object. A call keeps its element type and its comparator, and the stable
  sort's steps depend only on those and on the comparisons' results, so the
  order is the same, also where a comparator is not a total order
  (`partial_cmp` read as `Equal`). `f32_ascending` and `f32_descending`
  replace `sort_by(|a, b| a.total_cmp(b))` and its reverse: two floats
  `total_cmp` finds equal have the same bits, so the descending order is the
  ascending one reversed. Calls are changed only where their element type had
  more than one sorting closure in markitai's CLI (two `sort_by_key` calls
  become `stable` with their key's `cmp`); every call of `crate::sort` is a
  markitai change. The float sorts of `src/lib.rs` (vector grids, TSR) are
  left as they were, since the CLI does not compile them. No sort became
  unstable and no unstable sort changed.

  Measured on `3129ec2` with `cargo build --release -p markitai-cli` (rustc
  1.98.1, macOS 27.0.1 on an 18-core Apple M5 Max): the CLI is 22,773,504
  bytes before and 21,947,328 after (−826,176, −3.6%, including four
  markitai-core sorts changed alongside). In an unstripped build of the same
  profile with v0 symbol names, the code of `core::slice::sort`
  instantiations naming `pdf_inspector` (symbol-address differences) fell
  from 849,180 to 155,728 bytes. The 216 Chrome- and Quartz-printed PDFs of
  the R41 quality corpus give byte-identical output apart from the
  `markitai_processed` line, and paired, alternating runs over them show no
  slowdown (sums −0.4% and −0.2%, within the noise of a loaded machine). Two
  added tests compare the float orders bit for bit (signed zeros, NaNs) with
  the closures they replace and check the stable order of 64 tied pairs. The
  isolated copy's unit tests give 1,619 passed and the same 21 failed with
  every change listed here (1,617 before this item). Reversing `stable`'s comparator fails 183 more tests, reversing
  `f32_ascending` 64, dropping `f32_descending`'s reversal 49, and an
  unstable sort inside `stable` fails the added stability test.

  A second step shares the sorts of element types that had a compiled sort
  for a single comparator. `crate::sort::total(v, compare)` is for a
  comparator that is a total order (integers, `total_cmp`, `Ord` keys, or the
  column-valley scores of `validate_and_build_columns`, products of two
  counts that cannot be NaN): it sorts the positions `0..len` through the
  `usize` instance of `stable` and then moves the elements into that order.
  A total order has exactly one stable order, so the result is the one
  `sort_by` gives whichever compiled sort finds it; comparators that are not
  shown to be total (`partial_cmp(..).unwrap_or(Equal)` on coordinates,
  `compare_positioned_blocks`) keep `stable` or their own sort.
  `integers(v)` replaces `v.sort()` on `u16`, `u32` and `usize` by
  `sort_unstable`, the same order because equal integers are identical, and
  shares the unstable instance the table detectors compile anyway;
  `ToUnicodeCMap::mapped_runs` packs its `(u16, u16)` pairs as
  `start << 16 | end`, whose integer order is the pairs' order. `map` and
  `set` build a `BTreeMap`/`BTreeSet` entry by entry where `collect` sorted
  the collected entries with a sort compiled for each iterator type (a
  repeated key keeps its last value, as there). Calls changed in
  `src/detector.rs`, `src/lib.rs`, `src/overlong_numerals.rs`,
  `src/tounicode.rs`, `src/extractor/{content_stream,layout,mod,scripts,underline}.rs`
  and `src/markdown/{convert,furniture,mod}.rs`; `src/tables/` was left as
  it was, so element types the table detectors also sort keep their
  instance.

  Measured on `7fcdd34` with the same build: in the unstripped v0 build, the
  sort instantiations whose v0 symbol names `pdf_inspector` as the
  instantiating crate fell from 192,776 to 122,712 bytes (the remainder:
  sorts of `src/tables/`, `compare_positioned_blocks`, unstable sorts with
  ties, the shared `usize`, `u16`, `u32` and `f32` instances, and generic
  sorts of `unicode-bidi` and `unicode-normalization` compiled here). With
  the CLI's own sorts shared alongside, the macOS CLI is 21,731,920 bytes
  before and 21,500,560 after, of which this step is −82,592. The 216
  Chrome- and Quartz-printed PDFs of the R41 quality corpus give
  byte-identical output apart from the `markitai_processed` line, and
  paired, alternating runs over them show no slowdown (sums +0.37% and
  +0.17%, against +0.58% and +0.32% for a byte copy of the base binary). The
  isolated copy's unit tests give 1,621 passed and the same 21 failed (two
  tests added). Reversing `total`'s comparator fails 16 more tests, an
  unstable sort inside `total` 1, a permutation that stops after one swap
  per cycle 7, a descending `integers` 17, a `map` that keeps the first
  value 1 and a `set` that drops an element 16.
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
- `src/glyph_names.rs`, with three added files, `src/glyph_names/lookup.rs`,
  `src/glyph_names/sorted_names.rs` (generated) and
  `scripts/sorted_glyph_names.py`: a glyph name's character is found by
  binary search in a table packed at compile time instead of in a `HashMap`
  built at first use, which `build_glyph_to_unicode_map` filled with 4,528
  inserts and `GLYPH_TO_UNICODE` with four overrides. That function was the
  second largest in markitai's CLI: 108,960 bytes of code (symbol-address
  difference in an unstripped release-profile build). The list and the
  overrides are unchanged and compiled only for tests, as the table's source
  and the tests' reference. `lookup.rs` checks at compile time that the names
  are non-empty and strictly ascending (so none appears twice), that their
  bytes fit 16-bit offsets and that every character is in the Basic
  Multilingual Plane, and packs one byte string with 16-bit bounds, 16-bit
  code points and the range of names each first byte starts (69,831 bytes,
  no pointers for a position-independent executable to relocate). A search
  covers one first byte's names, about 7 steps on average instead of 12.
  The new public `glyph_to_unicode(name) -> Option<char>` is the exact
  lookup the map gave; `glyph_to_char` and `glyph_name_to_string` use it and
  keep their signatures and results. `build_glyph_to_unicode_map` is no
  longer public outside tests; nothing in the workspace called it.

  After changing the list, run
  `python3 vendor/pdf-inspector/scripts/sorted_glyph_names.py`; `--check`
  exits 1 when the table is stale. It reads every `m.insert` line of
  `glyph_names.rs`, a later insert of a name replacing an earlier one as in
  the map, and stops at an insert in another form or an unknown escape. A
  table out of order, with a name twice or with a character outside the
  plane fails to compile; one with a name missing, added or recoded fails
  the tests.

  lopdf's glyph list (`Glyph::from_name`) cannot replace the table: lopdf
  does not export it, and the lists differ. Of the 4,532 names here, 230
  are not among lopdf's 4,495 (the four overrides, 201 ZapfDingbats names
  `a1`… and 25 others such as `angbracketleftBig` and `controlNULL`) and 28
  have another code there (`angbracketleft` is U+3008 here, U+27E8 in
  lopdf); 193 of lopdf's names are not here.

  Tests (four, in `lookup.rs`): every listed name gives its character and
  the table holds exactly those names; 248,737 changed names (every prefix,
  a byte before or after, one byte changed by one, letters' case flipped;
  9,055 of them are listed names, a count also computed independently from
  the list) agree with the map; `glyph_to_char` reads every listed name, also
  with a suffix, and `glyph_name_to_string` every one but the five joined by
  underscores; a few named cases. A separate harness compiling the old and
  new modules side by side agreed on the same names and on 360,575 inputs to
  `glyph_to_char` and `glyph_name_to_string` (suffixes, names joined by
  underscores, `uni` and `u` forms). Of 22 mutants of the search, packing,
  first-byte ranges, compile-time checks, table data and the two readers, 21
  fail these tests (three at compile time); the other starts the search
  earlier, which is still correct. With the other
  changes, the isolated copy's unit tests give 1,617 passed and the same 21
  failed.

  Measured at `cb8b881` with `cargo build --release -p markitai-cli` (rustc
  1.98.1, macOS 27.0.1 on an 18-core Apple M5 Max): the CLI is 23,087,024
  bytes before and 23,004,448 after (−82,576, −0.36%; code −111,464,
  read-only data +20,480). The 216 Chrome- and Quartz-printed PDFs of the R41
  PDF quality corpus give byte-identical output, as does a generated PDF that
  names all 4,532 glyphs through `/Differences` (the characters of 3,921 of
  them appear in its Markdown). In a release-mode harness an exact lookup
  takes about 29 ns instead of about 10 ns, but nothing is built: the map
  took about 100–200 µs and 200 KB of heap at first use. Page-Markdown
  extraction of the 216 files looks names up 31,829 times (16 found), at most
  2,874 times in one file, well below the roughly 5,000 lookups per process
  at which the search would cost more than building the map.

- `src/painted_bullets.rs` (new, `pub mod painted_bullets` in `src/lib.rs`)
  and `src/lib.rs` (`LoadedPdf::pages_markdown_with_marks`, one parameter of
  `extract_pages_markdown_from_doc`): list bullets painted as shapes. A
  browser paints a list item's marker as a small filled disc or square or a
  stroked ring, so the text layer holds no marker and the page reader ran the
  items into one paragraph (a bulleted sidebar became one line of words). A
  caller that reads the page's vector graphics passes a function giving each
  1-indexed page's such marks (`PaintedMark`: a box in the page's user space
  and the colour it is painted in; the caller keeps out stroked shapes with
  straight sides, which are checkboxes or frames). `targets` decides which marks start a text
  line: both extents 0.15 to 0.5 of the item's font size, ending no more than
  two em before the item and at most half a point inside it, centred between
  its baseline and 0.8 em above it, painted in a dark neutral, in the item's
  colour or in the colour of most of the page's text, no other text on that
  baseline ending within three em before the mark, one mark per item. Each
  such mark becomes a `•` item just before the first item of its line, on
  that line's baseline and otherwise a copy of that item (font, size, marked
  content), which the reader's list detection takes like a bullet character.
  Font statistics and folio decisions are made before the bullets are added,
  and a bullet is never a folio. `pages_markdown` is
  `pages_markdown_with_marks` without marks, so its output is unchanged, as is
  every other reading. Markitai's layout reader applies `targets` to its own
  lines. Three added tests cover the rule's colour, size, position and
  neighbour conditions and the page reader's output with and without marks;
  of 13 mutations of the rule and of the injection, 12 fail them and the
  other (a centre below the baseline) is already excluded by the window of
  candidate lines. The isolated copy's unit tests give 1,624 passed and the
  same 21 failed.

The page-level OCR, font decoding, repair, limits and reliability routing remain
the upstream paths. Markitai's own visibility warnings and layout agreement
checks remain enabled. The only new public APIs are `TextLine::text_with_markup`,
`LoadedPdf` (`load_mem`, `document`, `as_loaded_by_lopdf`, `pages_markdown`,
`pages_markdown_with_marks`, `text_with_positions_and_rotations`,
`forget_page_runs`), `painted_bullets` (`PaintedMark`, `targets`) and
`glyph_names::glyph_to_unicode`; no optional runtime dependency is added.
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
