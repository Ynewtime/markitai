# Markitai's pinned document reader

This directory contains the published `anydoc` 0.2.4 package's include set
(`src`, `examples/convert.rs`, `README.md`, `LICENSE`) and its package manifest.
`UPSTREAM.json` records the original archive checksum, upstream commit and every
copied file's original checksum. The original MIT license remains in place. No
Cargo registry source was modified. The workspace coordinator owns the path
patch and lockfile.

The local changes, each marked `markitai` in its comment, are limited to these
upstream files:

- `src/formats/doc/mod.rs`, `src/formats/doc/objects.rs` and the added
  `pictures.rs`/`pictures_tests.rs`: main-story floating pictures follow their
  UTF-16 text anchor through shape IDs and picture-store slots, including
  delayed JPEG/PNG/EMF/WMF data in `WordDocument`. Hidden fields and unrelated
  pictures are excluded; repeated references reuse the asset. Record bounds,
  identities and complete metafile decompression are checked under resource
  limits. Complete picture-store records survive a stale entry count with a
  warning. Header pictures and complex group geometry are not inferred.
  Synthetic tests cover ordering, visibility, malformed records and limits.
- `src/model/style.rs`, `src/formats/doc/sprm.rs`, `src/formats/doc/stsh.rs`,
  `src/render/markdown/inline.rs`: legacy Word's `sprmCKul` (0x2A3E) preserves
  underline through character-style inheritance and direct formatting.
  Defined nonzero Kul patterns normalize to an underlined run; zero explicitly
  clears the inherited value, and unknown operands do not act as toggles.
  Markdown writes generated `<u>` tags around escaped content, including code
  spans, without bridging a gap of unstyled whitespace. Link labels omit them:
  Word's Hyperlink style would otherwise wrap every link.
  Decorative line patterns are normalized, not reproduced. `shared/delta.rs`
  and `docx/styles.rs` retain an existing underline while applying other
  properties; this change does not add underline parsing to other formats.
  Synthetic parser, inheritance and renderer tests cover the behavior.
- `src/shared/html.rs`: a `pre` element's code block keeps the language its
  markup names, a `language-…` or `lang-…` class on the `pre` or its `code`
  child (upstream always left it empty), so EPUB code fences carry their info
  string. Other class spellings and characters outside `[A-Za-z0-9+#._-]` give
  no language. A ruby group is its base text followed by its reading in
  parentheses, once per group (`漢字(kanji)`; each `rtc` a group of its own,
  `rp` left out), where upstream ran them together (`漢kan字ji`). A definition
  list's `dt` is a bold paragraph and each `dd` a block of its own (upstream
  joined them as `TermText`). A Markdown footnote mark written in text
  (`[^5]`, a label of letters, digits, `-` and `_` up to 32 bytes), outside
  code, is a note reference, so a book's `text.[^5]` and `[^5]: …` definition
  are written as marks instead of escaped text.
- `src/formats/sheet/numfmt.rs`, `src/formats/sheet/mod.rs`,
  `src/formats/sheet/xlsx.rs`: a date/time format records whether it shows
  seconds, and a time of day, alone or after a date, is written `hh:mm` when
  it does not (`h:mm`, `yyyy-mm-dd hh:mm`); seconds are dropped as a
  spreadsheet displays them, not rounded into the minute. An elapsed duration
  likewise shows seconds only when its format names them (`[h]:mm` → `27:05`),
  and runs from its bracketed unit, which carries the whole span, to the
  smallest unit its format names: `[mm]:ss` → `1625:00`, `[s]` → `97500`,
  `[h]` → `27` (upstream wrote every span as hours, minutes and seconds).
  Elapsed formats with fixed fractional-second digits (`[h]:mm:ss.00`,
  `[m]:ss.000`, `[s].00`) retain those digits, using the existing bounded
  decimal rounding before splitting units so carries cross minute/hour
  boundaries correctly. Quoted/escaped dots do not enable fraction parsing;
  ordinary date/time and nonfractional elapsed rendering stay unchanged.
  Regression tests cover actual seconds-token classification, nonzero/fixed-zero
  fractions, decimal half rounding, carries across units, negative/zero spans,
  escaped/quoted dots, unchanged date rendering and actual XLSX style/cell
  parsing (`elapsed_fractional_seconds_require_an_actual_seconds_decimal_token`,
  `fractional_elapsed_formats_preserve_precision_and_carry_across_units`,
  `fractional_duration_survives_the_actual_xlsx_styles_and_cell_reader`).
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
- `src/lib.rs`, `src/formats/mod.rs`, `src/formats/ppt/mod.rs`,
  `src/formats/ppt/ole.rs`: a public `embedded_object(bytes)` reads an
  embedded object's own file, as a PPTX keeps it in `ppt/embeddings`: a
  compound file goes through the same storage reader as a PPT object, and a
  zipped package is read as an ODF chart or spreadsheet, or else through
  `sheet::parse` as an OOXML workbook, with the same detachment and sheet-name
  handling. Anything else gives no blocks; only resource limits are errors.
- `src/shared/list.rs` and the readers that build list entries
  (`src/formats/docx/content.rs`, `numbering.rs`; `src/formats/rtf/mod.rs`;
  `src/formats/doc/mod.rs`; `src/shared/typed_lists.rs`; `src/formats/pptx/mod.rs`
  and `src/formats/ppt/mod.rs` only fill the new fields): a paragraph that
  continues a list item goes into it (upstream closed the list there):
  - `ListEntry` gains `indent` (where the item's text lines start, in
    twips) and `continues`; `ListEntry::continuation` makes an entry that
    `build_lists` appends to the item open at its level, after that item's
    nested lists, taking no number and splitting no list;
    `continuation_level` picks the deepest open item whose text starts no
    further right than the paragraph, when the paragraph reaches the
    outermost item's text and that text sits at least `2 * ALIGNED` (10
    points) right of the body text before the list; `unmarked_level` places
    a paragraph numbered at a level that shows no marker;
  - DOCX: a numbering level's `w:pPr/w:ind` is read (`LevelDef::left`,
    `first_line`) and applies between the style's indent and the
    paragraph's own; a bullet level whose `lvlText` is only spaces (and no
    picture) shows no marker, as `none` does (pandoc's "no marker" list,
    which upstream read as items); `resolve_numbering` also returns the
    level a paragraph is numbered at; a plain paragraph after an item
    continues it by indent or by such an unmarked level; empty paragraphs
    after an item are held and close the list only when another item
    follows, as before; `emit_paragraph` says when a paragraph went into the
    list so it is not recorded as a plain one;
  - RTF and Word 97: an item's `\li` / `sprmPDxaLeft` is its text indent, a
    plain paragraph after it continues it by indent (Word 97 also by a
    level with no marker), empty paragraphs are held as in DOCX, and the
    body text's indent is tracked; RTF table cells are unchanged;
  - ODF (`src/formats/odf/text.rs`): a `text:list` that continues the one
    before it (`text:continue-numbering`, `text:continue-list`) and opens with
    a `text:list-header` (LibreOffice's unnumbered entry) puts the header's
    blocks into the previous list's last item, as deep as the header is
    nested, and that list takes the following items when they carry on its
    count (`push_list`, `header_depth`, `last_item`; `parse_list` returns how
    many leading blocks the header gave);
  - lists typed by hand: a paragraph right after an item with a hanging
    indent, set in to the item's text, continues it (`Run`, `More`,
    `Member`); one after a rejected item stays text, and one that lines up
    with no open item ends the list there.
- `src/shared/typed_lists.rs`: hyphen and en-dash lines are dialogue (no
  list) when a line's clause after a comma is a speech incise
  (`speech_incise`: at most three words led by one of `SPEECH_VERBS`, about a
  hundred speech verbs in fourteen languages, or by a French inversion such
  as `dit-il`, `dis-je`, or an English pronoun and such a verb). On 21,825
  dash and `<li>` list items of local Markdown and HTML it fires on none.
- `src/shared/tabs.rs`: the lines of one paragraph split by line breaks are
  rows of a table set with tab stops when every line holds a tab and they
  split into the same number of cells (`rows`); the thresholds are those
  for paragraphs.
- `src/formats/docx/altchunk.rs` (added), `src/formats/docx/mod.rs`,
  `src/formats/docx/content.rs`, `src/package/archive.rs`: a `w:altChunk`
  is read where it stands (upstream ignored it), by the part's content type
  (`[Content_Types].xml`, else extension, else first bytes): HTML/XHTML
  through `shared::html` after `html_as_xml` makes well-formed XML of it
  (at most `MAX_HTML_DEPTH` open elements, deeper tags dropped with their
  text kept, so the XML reader's depth limit cannot fail the document; with
  `data:` images); MHT web archives (`message/rfc822`,
  `multipart/related`, as html-docx-js writes them) by their HTML part with
  quoted-printable/base64 decoding and pictures by `Content-Location` or
  `cid:`; RTF through `formats::rtf::parse`; plain text as a paragraph per
  line; Word documents through `parse_at` at most `MAX_DEPTH` (3) deep, whose
  archive counts its reads on from the outer one's (`Package::open_spent`,
  `total_read`, `charge`). An embedded document's assets are re-added with
  scoped origins, its notes appended after the document's, its anchors and
  note ids prefixed `chunkN-`. Other formats, deeper nesting, missing and
  unreadable parts add nothing and a warning; resource limits stay errors.
  `Ctx` gains `chunk_depth` and `embedded`.
- `src/shared/fields.rs`, `src/formats/docx/content.rs`, `src/formats/rtf/mod.rs`
  (and `FieldFrame`'s new field, left empty by `src/formats/doc/mod.rs`): a
  legacy form field shows its state, which Word keeps in the field data and
  not in the empty result: `FormField` (kind, state, default, entries) and
  `FormField::shown` give `☒`/`☐` for a check box (state 0/1, else the
  default; RTF's 25 is unset) and the chosen entry of a drop-down list (state,
  else default, else the first), only when the instruction names the same
  form field; `form_field_result` uses it for a field whose result shows
  nothing and `field_result` otherwise. DOCX reads `w:ffData` on the begin
  `w:fldChar` (`w:checkBox` `w:checked`/`w:default`, `w:ddList`
  `w:result`/`w:default`/`w:listEntry`; not in a hidden run); RTF reads
  `\*\formfield` inside the instruction (`\fftype`, `\ffres`, `\ffdefres`,
  each `\*\ffl` captured as an entry; `\ffname`, `\ffdeftext` and the other
  text destinations no longer join the instruction) and shows nothing for a
  field that started in hidden text.
- `src/formats/docx/symbols.rs` (`is_symbol_font`, `symbol_face`,
  `symbol_text`), `src/formats/docx/content.rs`, `src/formats/odf/styles.rs`,
  `src/formats/odf/text.rs`, `src/formats/rtf/mod.rs`: text set in the Symbol
  or Wingdings font shows the font's glyphs: each character `U+0020`-`U+00FF`
  or `F020`-`F0FF` maps through the font's table, others stay. DOCX finds the
  run's font as the monospace check does (now resolved for every run, not
  only when code fonts are in play); ODF by `style:font-name` through the
  face declarations (`svg:font-family`) or `fo:font-family` over
  `parent-style-name` (memoised like the monospace answer; OpenSymbol is not a
  symbol font here); RTF already mapped a symbol font's bytes and now maps a
  `\uN` in one the same way (Word's `\u-3913` is the Symbol bullet).
- `src/formats/docx/content.rs`: a VML picture's alt text is its shape's
  `alt` attribute when no `wp:docPr` describes the drawing.
- `src/formats/rtf/mod.rs`, `src/formats/rtf/table.rs`; RTF text Word does not
  show, and text upstream lost or misplaced:
  - hidden (`\v`) and deleted (`\deleted`) text is left out (`CharState`
    gains `hidden` and `deleted`, reset by `\plain`): no text, tab, line
    break, picture or math is pushed, and a hidden or deleted `\par` joins the
    paragraph to the next instead of ending it;
  - `\xe`, `\tc` and `\tcn` (index and contents entries) are suppressed
    destinations;
  - `\upr` reads only its `\*\ud` Unicode part, with the `suppress` and
    `capture` the `\upr` group opened with (`CharState::upr`), so a `\upr` in
    a suppressed destination stays suppressed;
  - a destination suppressed as a whole (every `SUPPRESSED_DESTINATIONS`
    entry but `shpinst` and `object`) sets `CharState::excluded`, inherited by
    its groups: `accepts_text` refuses text there, and the destinations that
    un-suppress (`\shptxt`, `\shppict`, `\result`, `\footnote`, `\bkmkstart`,
    `\nesttableprops`) set `suppress` to `excluded` instead of `false`; before,
    a header's or footer's text box, picture, object result, footnote or
    bookmark (and a page background's shape text) ran into the body;
  - shapes (`ShapeState`, opened by `\shp`/`\shpgrp` outside excluded
    destinations) and shape properties (`PropState`, `\sp` with `\sn`/`\sv`
    captured as `Capture::PropName`/`PropValue`; `sp`, `sn`, `sv` left
    `SUPPRESSED_DESTINATIONS`): `wzDescription` is the alt text of the
    `\pict` whose `\*\picprop` holds it (`PictState::alt`) or of the shape's
    picture; the `pib` property's `\pict` is read (its `\sv` un-suppressed),
    which makes the shape's `\shprslt` a suppressed copy; `gtextUNICODE` is
    WordArt, pushed as text when the shape closes (not for a hidden one or
    one with a text box), its `\shprslt` copy suppressed too;
  - `\NeXTGraphic` (TextEdit's picture attachment in an RTFD's text) is an
    excluded destination, and the attachment character after it (`¬` or
    `U+FFFC`) is dropped (`Parser::attachment`);
  - an empty paragraph set in a monospaced font inside a table cell is kept
    as an empty paragraph (`TableState::push_cell_blank_line`) when code
    fonts are in play, so a listing laid out in a cell keeps its blank lines
    when `listing_tables` makes it code; a cell renders it as nothing.
- `src/formats/odf/text.rs`; OpenDocument text upstream lost or ran
  together:
  - a drawing element that is a child of the body or of any block container
    (a page-anchored frame, picture, group or shape) is read where it stands
    (`walk_drawing`), its picture as a paragraph and its text as blocks;
  - in a paragraph and in groups, a shape holding text (`is_drawing`: custom
    shape, rectangle, ellipse, circle, polygon, polyline, path, line,
    connector, caption, regular polygon, measure) is read as a text box is,
    as blocks after the paragraph; `draw:g` and `draw:a` read their drawing
    children in order; other drawing elements give nothing;
  - `text:numbered-paragraph` is a list item at its `text:level`, labelled
    with its `text:number` (a decimal `N.` of a decimal list is the list's
    own number), joining the list before it when it numbers on from it
    (`push_numbered_paragraph`); an empty one adds nothing;
  - `text:user-index`, `text:table-index` and `text:object-index` are read as
    the other indexes are;
  - an inline text element with `text:display="none"` (an invisible
    variable or user field) and `text:script` are skipped, and a
    `text:section` with `text:display="none"` is left out;
  - a heading's `text:number` is its label when the outline style gives
    none, and `text:number` is no longer read inline.
- `src/model/mod.rs` and every place that builds a `Document` (the format
  readers and the Markdown renderer's tests): `Document` gains `warnings`,
  sentences about content a reader left out on purpose (so far only the
  altChunk warnings); empty for every other reader.
- `src/sort.rs` (added, declared in `src/lib.rs`), `src/formats/doc/mod.rs`,
  `src/render/markdown/mod.rs`, `src/shared/html.rs` and the added
  `src/shared/tabs.rs` and `src/shared/visual.rs`: a `sort_by_key` call
  compiles the standard library's stable sort again for its element type and
  key closure. The four such calls, all with integer keys, go through one
  compiled sort instead (`sort::by_key` sorts the positions with the caller's
  key, then moves the elements into that order); an integer key is a total
  order, which has exactly one stable order, so the order is the one
  `sort_by_key` gave. Heading sizes are sorted with the `u32` `sort_unstable`
  the crate already compiles and then reversed (equal sizes are identical).
  Sort code in an unstripped release build of markitai's CLI: 16,528 → 6,524
  bytes (the one shared sort). The CLI's output for 601 DOCX, ODT, RTF, DOC,
  PPTX, XLSX, EPUB and other office inputs is byte-identical.

- `src/formats/doc/mod.rs`: the object replacement character U+FFFC is read as
  an inline picture mark, as the special character `\u{1}` is. macOS's Word 97
  exporter (TextEdit, `textutil -convert doc`) writes it where a picture was
  and stores no picture data (the file has no Data stream), so upstream wrote
  a stray `￼` paragraph where the picture was; it is now nothing, and a file
  whose character does carry a `sprmCPicLocation` gives its picture. The test
  is `the_mark_a_word_97_file_keeps_for_a_picture_it_did_not_store_is_not_text`
  in `crates/markitai-core/src/formats/native/docx_tests.rs`, on the added
  fixture `textedit-word97-picture.doc`.

- `src/formats/sheet/xlsx.rs`, `xls.rs`, `xlsb.rs` and the added
  `src/formats/sheet/notes.rs`: every container appends a sheet through
  `push_sheet`, and `build_table` returns the grid with what it left out. A
  formula cell whose `<v>` is absent, or empty for a non-string result (openpyxl
  writes `<v></v>`), shows `=formula` as code and is counted in one workbook
  warning; hidden rows and columns that hold content are counted in a warning
  per sheet; an `http`, `https` or `mailto` cell hyperlink (`hyperlinks`,
  through the worksheet's relationships) becomes a link; legacy notes
  (`comments`) and threaded comments (`threadedComment`, authors from the
  workbook's `person` part, replacing their legacy placeholder) follow the
  table as a list under a `Notes` heading, notes of hidden cells left out; and
  a cell's line breaks stay `LineBreak`s (`shared::text::clean_cell_text`).
  Upstream dropped all four silently and turned line breaks into spaces.
- `src/formats/sheet/numfmt.rs`, `xlsx.rs`: the built-in currency (5-8) and
  accounting (41-44) ids, whose currency is the system locale's, resolve to
  their grouping, decimals and negative parentheses without a currency symbol
  (`(1,234.50)`; upstream: General), and a date section that names its month or weekday
  with an English or system locale, or labels a numeric month `月`/`월`, and
  shows a four-digit year is `Rendered::Spelled`, written as the format shows
  it (`Wednesday, March 4, 2026`, `2026年3月4日`) instead of an ISO date.
- `src/formats/doc/sprm.rs`, `stsh.rs`, `mod.rs`: `sprmCIss` (0x2A48) from the
  style chain, the CHPX and the piece Prm writes raised and lowered runs in
  Unicode forms through `docx::scripts::Script`, as the DOCX and RTF readers
  do; a field's instructions are not converted.
- `src/lib.rs`: `Format::from_extension` maps the templates (`dot`, `dotx`,
  `dotm`, `ott`, `potx`, `potm`, `xlt`, `xltx`, `xltm`, `ots`, `otp`) to their
  documents' formats (content detection already did), and `diagram_data`
  exposes `shared::drawingml::diagram_blocks` for a PPTX's SmartArt.
- `src/shared/officeart.rs`, `src/formats/doc/pictures.rs`,
  `src/formats/ppt/pictures.rs`: the DOC reader's complete picture decode moves
  to `officeart::complete_blip`, and the PPT picture bank uses it too, so a
  slide's EMF or WMF picture is kept when it inflates (zlib) or is stored to
  exactly its declared size, as in DOC, instead of being omitted as an
  unsupported format; a partial metafile is still omitted with that warning.
  The unused raw-deflate `fbse_blip` helper is removed. `pictures_tests.rs`
  covers an FBSE-embedded and a direct EMF and a short one.

`Cargo.toml` asks `zip` for `deflate-flate2-zlib-rs` instead of `deflate`, as
the workspace crates do: the same deflate backend without the zopfli encoder,
which zip uses only above level 9 and Markitai never requests.

`rustfmt.toml` (`use_small_heuristics = "Max"`) is not in the published
package; it reproduces upstream's formatting, so `cargo fmt` in this directory
changes nothing upstream wrote.

Tests covering these changes were added beside the upstream ones; the upstream
suite passes in an isolated copy (451 tests; 519 with the added ones), and so does its own
`cargo clippy --all-targets -- -D warnings` after one upstream line in
`src/formats/docx/numbering.rs` passes `level_value` by value instead of by
reference (`needless_borrows_for_generic_args`).
