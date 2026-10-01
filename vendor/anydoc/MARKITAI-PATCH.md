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
  spreadsheet displays them, not rounded into the minute. An elapsed duration
  likewise shows seconds only when its format names them (`[h]:mm` → `27:05`),
  and runs from its bracketed unit, which carries the whole span, to the
  smallest unit its format names: `[mm]:ss` → `1625:00`, `[s]` → `97500`,
  `[h]` → `27` (upstream wrote every span as hours, minutes and seconds).
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
- `src/formats/ppt/mod.rs` and the added `src/formats/ppt/ole.rs`: a shape
  whose client data holds an `ExObjRefAtom` shows an embedded OLE object,
  which is read as its data where the shape is (upstream read nothing for
  it). The `ExObjList`'s `ExEmbed` entries map object ids through the
  persist directory to `ExOleObjStg` records, decoded on first use and
  zlib-inflated (instance 1) under `MAX_ENTRY_BYTES` per object and
  `MAX_TOTAL_BYTES` per deck. The storage is a compound file: a BIFF
  `Workbook`/`Book` stream goes to `sheet::embedded_workbook`; LibreOffice's
  `package_stream` is an ODF package whose chart reads as its title and
  local data table (first row the header; repeated cells and rows expand only
  up to content, charged against `MAX_GRID_SLOTS`) and whose spreadsheet goes
  to `odf::parse`; an OOXML `Package` stream goes to `sheet::parse`. A lone
  sheet's name heading is dropped and several become paragraphs; the
  object's own images keep only their alt text (its asset ids are not the
  deck's) and its note references are dropped. Anything else adds nothing
  and logs at debug level. Each time a shape shows an object, its table
  positions are charged against `MAX_GRID_SLOTS`.
- `src/formats/sheet/xls.rs`, `src/formats/sheet/mod.rs`: the XLS reader
  gains `chart_data`, behind `sheet::embedded_workbook` (no change to `parse`).
  The globals keep WINDOW1's shown sheet; when it is a chart sheet, or the
  stream has no sheet directory and holds a top-level chart substream (MS
  Graph), the chart's cached data ([MS-XLS] SERIESDATA: SIIndex 1 values and
  2 category labels by point and series, through their XFs), series names
  (the SeriesText after BRAI 0 in a Series block) and title (the attached
  label whose ObjectLink names the chart) become a table under a title
  paragraph. Series marked SerParent (trendlines, error bars) without values
  get no column, unnamed series are `Series N`, points without a label are
  numbered from 1; Begin nesting past `MAX_RECORD_DEPTH` and a cache past
  `MAX_GRID_SLOTS` are resource limits. Otherwise the worksheets are read by
  `parse`.
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
  `src/formats/docx/mod.rs` and the added `src/formats/docx/code.rs` (the
  body's monospace share; the face list and the shared decisions are in
  `src/shared/code.rs`, below): a run in
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
- `src/shared/code.rs` (added, declared in `src/shared/mod.rs`; the face list,
  share, run counting and line-number removal moved here from
  `src/formats/docx/code.rs`), `src/formats/odf/text.rs`, `styles.rs`,
  `table.rs`, `mod.rs`, `src/formats/rtf/mod.rs`, `tables.rs`: code set in a
  monospaced font in OpenDocument text and RTF, as for Word:
  - an ODF run's font is the face its `style:font-name` names
    (`office:font-face-decls`, by `svg:font-family`), else `fo:font-family`
    (a list ending in the generic `monospace` counts), through
    `parent-style-name` over the paragraph default style; RTF reads the font
    table's names (TextEdit's fonts one after another and Word's group per
    font, `\*\panose` and `\*\falt` groups aside) and `\deff`;
  - a face matches by name, also in PostScript form (`Menlo-Regular`,
    `CourierNewPSMT`, `SFMono-Regular`); a face only declared fixed-pitch
    (ODF `style:font-pitch="fixed"` or the `modern` generic family, RTF
    `\fmodern` or `\fprq1`) counts unless its name or RTF charset is CJK;
  - a paragraph all in such a font is a line of code outside table cells
    and lists (ODF) or list items (RTF), unless it is a heading (which drops
    the code style) or holds an image or a displayed formula; a blank one in
    such a font is a blank line of the listing;
  - the readers count the share while reading; when monospace sets more than
    three quarters of the body (notes aside) the document is read a second
    time without code;
  - in all three readers (`listing_tables`, also called from
    `src/formats/docx/mod.rs`), a table that only lays out a code listing (one
    row of code after an optional cell of line numbers, one numbered line per
    row, or one cell of code) becomes that code block, and ODF and RTF code
    blocks lose their line numbers as Word's do.
- `src/shared/tabs.rs` (added, declared in `src/shared/mod.rs`),
  `src/shared/blockstyle.rs`, `src/shared/visual.rs`, and
  `src/formats/docx/content.rs`, `styles.rs`, `mod.rs`,
  `src/formats/odf/text.rs`, `styles.rs`, `mod.rs`, `src/formats/rtf/mod.rs`:
  columns set with tab stops are a table (rules in the module):
  - in the body, a reader keeps each tab (`w:tab`, `text:tab`, `\tab`) as a
    text inline of exactly `\t` (upstream wrote a space at once) and records
    each plain top-level paragraph holding one with its tab stops: Word's
    `w:tabs` over the paragraph style's `basedOn` chain (`clear` removes a
    stop, bar tabs draw only a line; table-of-contents and index styles
    are skipped), the nearest ODF `style:tab-stops`, RTF `\tx` with the
    alignment and leader words before it, reset by `\pard`;
  - `tabs::finish` turns each qualifying run of such paragraphs into a
    `Block::Table` and writes every other tab back as a space, so nothing
    else renders differently: `StyledRun::push` reads a code line's tabs as
    spaces, an ODF heading's tabs become spaces before its anchor is taken,
    and `Looks::skip` keeps a table's rows out of the heading guess;
    `w:ptab`, notes in Word and presentations keep upstream's space.
- `src/shared/typed_lists.rs` (added, declared in `src/shared/mod.rs`),
  `src/shared/tabs.rs`, and the readers that feed it:
  `src/formats/docx/content.rs`, `styles.rs`, `mod.rs`;
  `src/formats/odf/text.rs`, `styles.rs`, `mod.rs`; `src/formats/rtf/mod.rs`,
  `tables.rs`; `src/formats/doc/mod.rs`, `sprm.rs`. A run of plain body
  paragraphs that open with a typed bullet or number (`Tab • Tab text`, as
  `textutil` saves HTML lists in Word documents) is a list (rules in the
  module):
  - each reader records every plain top-level paragraph with its indent:
    Word's `w:ind` over the paragraph style's through `basedOn` (`w:left` or
    `w:start`, `w:hanging`, `w:firstLine`, their `…Chars` forms, and
    `textutil`'s `w:first-line`; table-of-contents and index styles are
    skipped), ODF `fo:margin-left` and `fo:text-indent` through
    `parent-style-name`, RTF `\li` and `\fi` until `\pard`, and Word 97
    `sprmPDxaLeft`, `sprmPDxaLeft1` and their Word 97 forms (`PapDelta`
    gains `left` and `first_line`) over the style chain;
  - the Word 97 reader keeps the body's tabs as `tabs::TAB` inlines and ends
    with `tabs::finish` (no tab stops are read there, so no table is found)
    instead of `Looks::apply`; a tab elsewhere is the space it was;
  - `tabs::finish` takes the paragraphs: after finding tables it applies the
    heading guess, which no longer takes a paragraph opening with a bullet,
    then places the lists among the paragraphs left (not a table's rows)
    and writes the remaining tabs back as spaces;
  - in the body of DOCX, ODT and RTF, a paragraph set all in a monospaced
    font that opens with a bullet is an item, not a line of code;
  - RTF text in a font named `Symbol` or `Wingdings` maps each byte through
    the DOCX reader's symbol table (`symbols.rs`, made `pub(crate)` in
    `src/formats/docx/mod.rs`) where it has a character, so Word's Wingdings
    square and arrowhead bullets (`\'a7`, `\'d8`) are no longer `§` and
    `Ø`; other bytes decode as before.

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
- `src/shared/visual.rs` (added, declared in `src/shared/mod.rs`) and the
  readers that feed it: `src/formats/docx/content.rs`, `styles.rs`, `mod.rs`;
  `src/formats/odf/text.rs`, `styles.rs`, `mod.rs`; `src/formats/rtf/mod.rs`,
  `tables.rs`; `src/formats/doc/mod.rs`, `sprm.rs`, `stsh.rs`. A document with
  no heading style or outline level anywhere may still show its headings as
  short bold paragraphs set above the body size (TextEdit and `textutil` save
  a web page that way); those become headings, ranked by size:
  - each reader now resolves text size, which upstream never read: Word's
    `w:sz` (run, character style, paragraph or default style, `docDefaults`,
    else 10 points), ODF `fo:font-size` (absolute units or a percentage,
    through `parent-style-name`, over the paragraph default style or 12
    points), RTF `\fs` (a style's too; `\plain` resets to 12 points) and
    Word 97 `sprmCHps` (style chain, CHPX, piece `Prm`, else 10 points);
  - while reading, a reader reports the size of every piece of visible body
    text (not notes, field instructions or hidden text) and which top-level
    paragraphs are plain: not a heading, list item, table cell, text box or
    styled container; the Word 97 reader's `emit_paragraph` now says whether
    it wrote one;
  - an ODF `text:section` is read into the body's own blocks (upstream read
    it as a separate container and appended the result, which gives the same
    blocks) so its paragraphs count as the body's;
  - `Looks::apply` (rules and their evidence in the module) leaves the
    document alone if any heading exists, and otherwise turns into a heading,
    without its bold or the line breaks around it, each such paragraph whose
    text is all one size at least a point above the body size and bold (or a
    third above without bold), at most 15 words, with a letter, no image, and
    no prose ending; the largest such size is level 1, the next level 2, up
    to 6.
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
suite passes in an isolated copy (401 tests).
