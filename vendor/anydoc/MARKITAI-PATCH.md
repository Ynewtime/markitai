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
- `src/formats/ppt/mod.rs`: a group shape whose own shape's tertiary options
  set bit 0 of `tableProperties` (0x039F) is read as one table instead of one
  paragraph per cell. Each cell shape's text goes through the ordinary text
  shape path; the cells' child anchors draw the grid (edges within 8 master
  units are one line), a cell spanning lines is a merged cell, a shape stored
  over a cell adds its text to it, zero-size shapes (border lines) are
  skipped, and the first row is the header. A table group inside a cell is
  read as plain shapes, and a group with an unanchored cell, or one drawing
  more positions than `MAX_GRID_SLOTS`, keeps upstream's paragraphs.
- `src/formats/docx/content.rs`, `src/formats/docx/styles.rs`,
  `src/formats/docx/numbering.rs`, `src/formats/docx/mod.rs` (module
  declarations and tests only), and three files added beside them
  (`symbols.rs`, `numerals.rs`, `scripts.rs`); Word content the reader lost or
  misread, found with a corpus of one document per complex feature:
  - the base text of a `w:ruby` phonetic guide is read (upstream skipped the
    element, so furigana and pinyin documents lost the words themselves); the
    guide text stays out;
  - `w:noBreakHyphen` is a `-` (upstream dropped it and ran "e-mail" together)
    and `w:sym` is read through a Symbol/Wingdings-to-Unicode table
    (`symbols.rs`; unmapped private-use codes are still dropped);
  - hidden text (`w:vanish`, from the run, its character style or the
    paragraph style, an explicit off value winning) is left out, as Word does
    not display or print it; the field marks of a hidden run still balance;
  - a table row whose deletion is tracked (`w:trPr/w:del`) is dropped, as
    its text already was, instead of leaving an empty row;
  - `Table::header_rows` is the number of leading `w:tblHeader` rows and no
    longer the guess `resolve_header_rows` made from column types;
  - `w:numFmt` values for CJK counting (一、二、三), legal and heavenly-stem
    numerals, circled/parenthesised/full-width digits, zero-padded digits and
    English ordinals write their own characters in list and heading labels
    (`numerals.rs`); before, every one of them was Arabic digits;
  - the words of VML WordArt (`v:textpath`'s `string` attribute) are read where
    no text-box content exists (upstream left a WordArt title out);
  - a run raised or lowered with `w:vertAlign` is written in Unicode
    superscript or subscript forms when every character has one ("10⁻³",
    "H₂O"; `scripts.rs`), and left at the baseline otherwise ("1st").
- `src/formats/docx/content.rs`, `src/formats/docx/styles.rs`,
  `src/formats/docx/mod.rs` and the added `src/formats/docx/code.rs`: a run in
  a monospaced font (its `w:rFonts`, else its character style's, else its
  paragraph style's, the `w:default` paragraph style's or `docDefaults`) is
  code (`Style::code`, which upstream never set for Word); a paragraph whose
  text runs are all monospaced, or a blank one whose runs or paragraph mark are,
  is a `BlockStyle::Code` paragraph outside table cells, and a heading in such a
  font drops the code style. Code is not marked when monospace sets more than
  75% of the body's visible characters. An empty code-style paragraph is a
  blank line of its block (upstream dropped it, also for `HTML Preformatted`),
  and a code block whose lines carry their numbers (as a leading column or
  alternating) loses them.

- `src/formats/rtf/tables.rs`, `src/formats/rtf/mod.rs`, `src/formats/rtf/table.rs`;
  RTF content lost or misread in documents TextEdit and Word save:
  - `\fcharset0` is Windows-1252 under any `\ansicpg` (only charset 1 follows
    the code page) and charset 77 is Mac Roman; TextEdit declares
    `\ansicpg936` on a Chinese system and writes its charset-0 fonts' bytes
    in 1252, which upstream decoded as GBK (`Apple\'92s` became `Apple抯`,
    `\'bd` a replacement character);
  - a `\nestcell` straight after a `\nestrow`, with no `\itap` or `\pard`
    between them, closes the cell holding the finished nested table (TextEdit
    ends a table nested two deep that way; upstream read it one level too
    deep and lost every row of the inner table);
  - a completed nested table is placed in its cell before the cell's next
    paragraph or list item, not after the cell's last paragraph;
  - a `HYPERLINK` field whose result spans paragraphs links each
    paragraph's part (upstream kept the text and dropped the link);
  - list paragraphs inside a table cell form a list, numbered from the list
    tables as in the body (upstream wrote them as plain paragraphs);
  - `\super` and `\sub` text (until `\nosupersub` or `\plain`) is written in
    Unicode super/subscript forms where every character has one, through
    the DOCX reader's `scripts.rs` (made `pub(crate)` in `src/formats/docx/mod.rs`).
- `src/formats/odf/text.rs`, `src/formats/odf/styles.rs`, `src/formats/odf/mod.rs`,
  `src/package/xml.rs` (the ODF chart namespace):
  - `style:text-position` (`super`, `sub`, or a raise percentage, through
    `parent-style-name`) is read as for Word's `w:vertAlign`;
  - text a style hides (`text:display="none"`) is left out, as Word's hidden
    text is;
  - `office:annotation` (a comment: author, date and text) is skipped; it
    is not in the text namespace, so upstream read its children into the
    annotated sentence;
  - the base text of `text:ruby` is read and the guide text is not;
  - a `draw:object` holding a chart yields the chart's title and its own
    data table (`chart:chart`'s `table:table`, first row as the header) in
    place of the replacement picture, in text documents and on slides;
  - a slide table with `table:use-first-row-styles="true"` (Impress's
    header row) takes its first row as the header.
- `src/lib.rs`, `src/formats/mod.rs`, `src/formats/sheet/mod.rs`: a public
  `format_number(code, value, date1904)` renders a number through the sheet
  readers' number-format engine, so a PPTX chart's cached values read as the
  chart shows them (dates, percentages) instead of bare serial numbers.

`Cargo.toml` asks `zip` for `deflate-flate2-zlib-rs` instead of `deflate`, as
the workspace crates do: the same deflate backend without the zopfli encoder,
which zip uses only above level 9 and Markitai never requests.

`rustfmt.toml` (`use_small_heuristics = "Max"`) is not in the published
package; it reproduces upstream's formatting, so `cargo fmt` in this directory
changes nothing upstream wrote.

Tests covering these changes were added beside the upstream ones; the upstream
suite passes in an isolated copy (333 tests).
