# Document formats and conversion contracts

The Rust engine parses documents locally and returns a `Document`: Markdown,
metadata, embedded asset bytes and explicit warnings. It never writes assets or
fetches external content inside a format adapter. Output naming, profiles,
network policy and optional model enhancement belong to the orchestration layer.

## Implemented readers

| Inputs | Reader | Current behavior |
| --- | --- | --- |
| TXT, MD, MARKDOWN | Rust text decoder | Preserves text and existing frontmatter; accepts UTF-8, BOM-marked UTF-16 and Windows-1252 |
| HTML, HTM, XHTML | scraper + htmd | Selects an article/main candidate, extracts metadata, removes navigation/scripts/hidden content, resolves relative HTTP links and images |
| CSV, TSV | csv | CSV preserves the reference's header width and raw cells; TSV retains the widest row and escapes table delimiters |
| IPYNB | serde_json | Markdown cells, fenced code and raw cells; metadata title and code language; code fences sized to protect embedded backticks |
| JSON | serde_json | Validated, pretty-printed fenced JSON; an additive Rust format |
| XML | quick-xml | Structured headings, attributes and mixed text, plus a source fence for small inputs; document types are rejected |
| EML | mail-parser | Decoded subject, body and MIME attachments; attachment bytes returned separately |
| MSG | cfb + native properties | Outlook headers, Unicode/ANSI body, HTML fallback and bounded by-value attachments |
| RST, Org, TeX | native markup readers | Structured sections, lists, code, math, links and tables; unsupported constructs retained with warnings |
| JPEG, PNG, GIF, BMP, TIFF, WebP | image + native LLM transport + macOS Vision | Standalone vision inputs, shared raster assets and complete TIFF page OCR/vision with bounded decoding |
| DOC, DOCX, DOCM | anydoc document model | Headings, styled text, lists, tables, links, formulas, notes and embedded assets |
| PPT, PPS, POT | anydoc document model | Legacy presentation content through the shared Markdown renderer, behind a numbered slide marker per slide; embedded charts and worksheets read as their data tables |
| PPTX, PPTM, PPSX, PPSM | bounded ZIP + PresentationML reader | Ordered slide markers, title placeholders, plain text frames, grouped shapes, tables, referenced images, cached chart data and speaker notes |
| XLS, XLSX, XLSM, XLSB | anydoc document model | Native sheet content; XLS/XLSX/XLSM single-sheet names are recovered from package metadata; exact cell-format compatibility has not been established |
| ODT, ODS, ODP, RTF | anydoc document model | Native structured documents through the same Markdown renderer; ODP slides carry numbered slide markers |
| NUMBERS | bounded ZIP/directory IWA preflight + iwork | Ordered sheets/tables, rectangular saved values and explicit formatting/unsupported-content warnings; see [Numbers](numbers.md) |
| EPUB | anydoc + OPF metadata | Spine content and the original title/authors/language/publisher/date/description/identifier preamble |
| PDF | pdf-inspector + lopdf; optional macOS CoreGraphics/Vision | Per-page text/layout, partial recovery and embedded images; explicit local-file page OCR and screenshots through the shared media pipeline |

The native Office renderer reads the document once and preserves referenced
embedded bytes. Shared image preparation then applies configured filtering and
compression. Unreferenced archive images are omitted. References use `.markitai/assets/{name}` until the output layer assigns
final paths. A merged table's origin contains its content; covered cells are
empty, and a warning records this Markdown representation.

Word documents (DOCX/DOCM) are read as the document looks with its tracked
changes accepted: inserted and moved-in text is kept; deleted text and rows are
not, and neither is hidden text (`w:vanish`). Comments, headers and footers are
not part of the Markdown; a document with comments gets a warning naming their
number. A table's first row is a Markdown header only when Word marks it as a
repeated header row; otherwise the header line is blank and every row is data.
The base text of a phonetic guide (ruby) is kept and the guide is not, and the
words of VML WordArt are read from the shape.
Superscript and subscript runs are written as Unicode super/subscript
characters when every character has one (`10⁻³`, `H₂O`) and stay at the
baseline otherwise. `w:sym` characters from the Symbol and Wingdings fonts map
to their Unicode marks; a non-breaking hyphen is `-`. Chinese, Japanese and
enclosed-digit numbering (`一、`, `（二）`, `①`) keeps the document's own
characters in list and heading labels. A list label Markdown does not read as
a marker (`a)`, `(1)`, `一、`) is kept as text, one item to a line. Emphasis
beside a letter has its edge punctuation moved outside the markers
(`**注意**：请`), since CommonMark would not open or close it otherwise.
Indentation typed with spaces is kept, because code pasted into Word depends
on it; four or more leading spaces therefore render as a code block. Fields keep
their last computed result, and floating shapes appear where their anchor
paragraph is.
Text set in a monospaced font (Courier, Consolas, Menlo, or a family named
`… Mono`, also by its PostScript name such as `Menlo-Regular`; by the run's own
font, else its character style's, else its paragraph style's or the document
default) is code: a paragraph that is all such text is a line of a fenced code
block, blank lines of the listing included, and such text inside prose is
inline code. A heading set in it stays a plain heading, a list item stays a list
item, and a table cell keeps it inline. When monospace carries more than three
quarters of the document's visible characters it is the document's typeface (a
typewriter manuscript) and marks nothing. A code block whose lines carry their
own numbers, as a column before them or alternating with them (a web page's
numbered listing saved as a document), loses the numbers, and a table that only
lays out a listing (one row holding the code after an optional cell of line
numbers, as syntax highlighters build it; one numbered line per row; or a single
cell of code) is that code block, without the numbers.

Columns set with tab stops become a table. A run of at least three consecutive
plain body paragraphs (not headings, list items, table cells, text boxes, notes
or code) at the same tab stops, each split by its tabs into the same number of
cells and no longer than one printed line (100 characters, no line break), is a
table's rows. At the default stops a run of tabs separates one pair of columns,
since authors press Tab until the text lines up; at stops the author set each
tab moves to the next column, so a cell may be empty. A column no row fills,
such as a tab that indents every row, is dropped. Three columns are enough; two
need stops the author set and a first column that is not a list label (`1.`,
`a)`) or a field label (`Date:`). A first column of bullets is a list typed by
hand (below), not a table; a tab stop with a leader or a
table-of-contents or index style, or a last column of page numbers counting up
with no header above them (a contents page) keeps the paragraphs as text. The
first row is the header only when it alone is bold; otherwise every row is data
and the header line is blank, as for Word tables. Every other tab, including
those in headings, lists, cells, notes and code, reads as a single space as
before. On 27 cases written as DOCX, ODT and RTF (stops set on the paragraph or
its style, default stops pressed several times, decimal stops, empty cells,
links in cells, an indented table, a typewriter document; contents pages with
leaders, by hand and in a contents style, verse, a lone tab, two rows, lists
typed by hand, memo labels, two default-stop columns, prose, and rows inside
lists, headings, code, cells or one paragraph's lines), each format finds all
9 tables and makes no table of the 18 text cases. The 108 `textutil` pages,
whose only tabs belong to lists typed by hand, give no table.

Lists typed by hand are lists. Many documents type their lists: each item is
a paragraph that starts with a bullet or a number and a tab or a space, as
macOS `textutil` saves every HTML list in Word documents (`Tab • Tab text`
under a hanging indent, `Tab ◦ Tab text` one level down, `Tab 1 Tab text` in an
ordered list). In DOCX, ODT, RTF and Word 97 files, a run of consecutive plain
body paragraphs (not headings, real list items, table cells, text boxes,
notes, code, a table-of-contents style or a table set with tab stops) that
open that way is a list:

- a bullet that is only ever one (`•`, `◦`, `▪`, `●`, `➢`, `・`, Word's Symbol
  and Wingdings bullets, a ballot box, a tick) makes an item by itself; a
  ballot box becomes a task-list box (`[ ]`, `[x]`) and a tick stays in the
  item's text;
- a character that also starts sentences (`-`, `–`, `*`, `+`, `o`, an arrow)
  needs a tab after it, an item above it that it is indented under, or a
  sibling at its level with the same mark and no other such mark beside
  them (`+ fast`, `- loud` stay text). An em dash needs a tab or a parent,
  and hyphens or en dashes whose lines end as speech does (`?`, `!`, `…`, a
  quotation mark) or set narration apart (`– Oui, – dit-elle.`) are dialogue
  and stay text, as do a lone `- and then…`, an attribution (`— Name`) and
  `* * *`;
- numbers make items when at least two at one level count up by one in one
  form (`1.` `2.`, `a)` `b)`, `(i)` `(ii)`, `一、` `二、`), or when a decimal
  number stands alone before a tab (`Tab 2 Tab`, an `<ol start="2">`),
  unless numbers that do not count up stand beside it (an inventory) or its
  text is all bold (a numbered heading typed by hand). A number before a
  space (`3 apples`), an initial (`A. Smith`), a year, an outline number
  (`1.2`) or a quantity (`1.5 kg`) is text;
- the level comes from where the marker sits (the paragraph's left and
  hanging indent, then the tabs and spaces before the marker): an item set
  further right than the one before it is nested under it;
- decimal numbers keep their count (`2.` for `<ol start="2">`), and other
  labels are kept as written, one item to a line, as for real lists.

A bullet line set in a monospaced font is a list item with inline code, not a
code block, and a paragraph opening with a bullet is never taken for a heading
set by hand. RTF text in a `Symbol` or `Wingdings` font reads as that font's
characters where they are known (Word's Wingdings square bullet `\'a7` was
`§`). On the 108 `textutil` pages, DOCX now gives 261 list items (11 before),
exactly as many as the ODT and RTF versions of each page, whose lists are real
ones; 256 of them have the same level, number and text (two keep the
`<ol start>` numbers the ODT loses, two sit under section headings numbered in
their text that the page's CSS sets as body text, which therefore read as a
numbered list, and one differs by a thin space). The Word 97 files give 260 (12
before; the other two sit in a one-cell layout table). The strict word
comparison with the reference is unchanged in every format, and DOCX now has
the same 64 code blocks and 35 inline code spans as ODT and RTF. On 80 cases
written as DOCX, ODT and RTF (42 lists with 105 items: bullets by hand, by tab,
by Symbol and Wingdings codes, nested by tabs, spaces or indents, numbers in
twelve forms, checklists, a monospaced bullet; 38 look-alikes: dialogue in four
languages, attributions, quotations, scene breaks, numbered headings, numbers
that skip, inventories, years, initials, outline numbers, quantities, pros and
cons, section signs, bullets in a table cell, a heading or code), each format
finds every item and nothing else. Of these, 25 were written after the rules
were first set; they found three look-alikes read as lists (French dialogue
ending in full stops, Spanish dialogue with a spaced dash, pros and cons),
which the em-dash and mixed-mark rules now keep as text. Dialogue set with
hyphens or en dashes whose lines all end in full stops, with no narration set
apart by a dash, still reads as a list.

OpenDocument text and RTF follow the same conventions. An ODT run's font is
the face `style:font-name` names (its `svg:font-family`), else `fo:font-family`
(a list ending in the generic `monospace` counts), through `parent-style-name`
over the paragraph default style; RTF reads the font table's names (TextEdit's
fonts one after another or Word's group per font) and `\deff`. A face the
document only declares fixed-pitch (ODF `style:font-pitch="fixed"` or the
`modern` family, RTF `\fmodern` or `\fprq1`) is a code face too, unless its
name or RTF charset is CJK (MS Gothic, SimSun), whose Latin letters are
fixed-width under body text. Tab stops are a paragraph style's
`style:tab-stops` (a `style:leader-style` other than `none` is a leader) and RTF
`\tx` after `\tqr`, `\tqc`, `\tqdec` or a leader such as `\tldot`, until
`\pard`. On the 108 `textutil` pages, where ODT and RTF had no code at all, each
now gives 64 code blocks and 35 inline code spans, as DOCX does on the same
pages since a hand-typed list item set in Menlo became a list item with inline
code there too (it was DOCX's 65th block); the only words that leave ODT and
RTF output are 55 listing line numbers.

Raised and lowered text (`style:text-position`; RTF `\super` and `\sub`) is
written in Unicode super/subscript forms where every character has one. ODF
text a style hides
(`text:display="none"`) is left out, the base text of a phonetic guide is kept
without the guide, and comments (`office:annotation`) are left out with a
warning naming their number, as for Word. A chart embedded in an ODT or ODP
reads as its title and the data table the chart document stores, first row as
the header, instead of its replacement picture. In RTF a charset-0 font's bytes
are Windows-1252 whatever `\ansicpg` declares (TextEdit on a Chinese system
writes `\ansicpg936` over 1252 text), a hyperlink spanning paragraphs links
each paragraph's part, lists inside table cells stay lists, and a nested
table TextEdit closes with `\nestcell` directly after `\nestrow` (no `\itap`)
is read at the right depth; text after a nested table in a cell stays after
it. Text that goes through the shared document renderer (Word, OpenDocument,
RTF, EPUB, legacy PowerPoint and spreadsheets) escapes `<` before a letter,
`/`, `!` or `?` and an `&` that starts an entity, so a Markdown renderer shows
`std::vector<int>` or a literal `&copy;` instead of reading them as HTML; PPTX
text frames keep their plain-text contract.

A DOCX, ODT, RTF or Word 97 document with no heading style or outline level
anywhere often still shows its headings as short bold paragraphs set above the
body size (TextEdit and `textutil` save a web page's headings that way). In
such a document a plain top-level paragraph (not a list item, table cell, text
box or quote/code container) becomes a heading when all its visible text has
one size, at least one point above the body size in bold or a third above it
without bold, with at most 15 words (two CJK characters count as one), a letter,
no image, and no closing `.`, `,`, `;`, `:` or their full-width forms. The body
size is the size of most of the body's visible characters (notes excluded).
The largest such size is level 1, the next level 2, and so on to 6; the heading
drops its bold and the line breaks around it. Bold text at the body size stays
a paragraph, as does anything smaller: in the measurement such lines were
table headers, labels and lead-ins at least as often as headings. On the 108
R42 pages saved by `textutil`, against their HTML `h1`–`h6`, 281–282 of 361
headings are found per format (none before) with no paragraph wrongly
promoted and no other output change; the reference's RTF guess (bold, 4 points
above the body, under 12 words) finds 188. On 110 other pages held out from
choosing these rules, 275–282 of 305 are found (the reference's RTF guess
finds 198); one promoted paragraph is not an HTML heading, a bold 13-point
sidebar label over 12-point text in the DOCX, where `textutil` flattens the
layout table. Headings set bold at the body size (an `h4`) or that a page's CSS
sets at body size are not found.

OOXML presentations have a separate reader because the generic document model
flattens slide boundaries. The package's presentation relationships and
`sldIdLst` determine slide order, including empty slides; filename sorting and
heading counts do not determine boundaries. Every slide receives
`<!-- Slide number: N -->`. Shapes are stably ordered by their effective top/left
coordinates, with layout/master placeholder coordinates used when missing.
Group children are ordered within their group. As in the reference reader, the
first top-level placeholder with index zero supplies the first-level title;
its type need not literally be `title`. Grouped or later placeholders do not
turn the remainder of the slide into headings. Body text retains paragraph boundaries and does not
acquire extra list markers from master styles, matching the reference's plain
text-frame contract. Shared normal-mode cleanup remains responsible for repeated
footers; pure output keeps the extracted text.

Presentation tables use their first row as the Markdown header. Images remain
at their shape position, keep normalized description text, and return their
original embedded bytes; shared image processing owns encoding. Repeated
references to the same package image share one asset. Cached chart category and
series values become a table; linked workbooks are never opened or recalculated.
Cached numbers read through their cache's (or point's) format code with the
spreadsheet readers' engine, so date categories are ISO dates rather than
serial numbers (`c:date1904` honoured) and `0%` values are percentages; text
points and General numbers are unchanged.
Speaker notes follow their slide under `### Notes:`. Missing or malformed slides
retain their numbered marker with a warning, while readable slides survive; a
package with no readable slide fails. Unknown shapes retain available DrawingML
text with a warning, and unsupported charts are explicitly identified. These
fallbacks do not imply complete drawing, chart-type or SmartArt support.

The vendored anydoc records where each slide of an ODP or a legacy PPT begins
(`slide_starts`, see its `MARKITAI-PATCH.md`), and the renderer writes the same
`<!-- Slide number: N -->` line before each slide, blank slides included, with a
blank line between slides. This differs from the reference, whose legacy PPT
output has no slide markers and which does not read ODP. In these formats a
slide's speaker notes keep their place after it as a quote. An ODP table styled
with a header row (`table:use-first-row-styles`, Impress's default) takes its
first row as the header, as PPTX and PPT tables do. A legacy PPT whose persist directory is
unusable is read in raw stream order, where slides cannot be told apart, and
carries no markers.

A legacy PPT table is a group shape whose `tableProperties` mark it as one; each
cell is a shape of the group with its own text box. The vendored anydoc places
the cells on the grid their anchors draw (edges within 8 master units, 1/72
inch, are one grid line), keeps a cell spanning several lines as a merged cell,
adds the text of a shape stored over a cell to that cell, and takes the first
row as the header, as PowerPoint styles it. Border lines are ignored, trailing
empty rows are dropped, and a group with a cell that has no anchor keeps its
cells' text as paragraphs, as before.

A legacy PPT shape that shows an embedded OLE object (an `ExObjRefAtom` in its
client data) reads as the data the object holds, at the shape's place in the
slide's drawing order; before, such a slide kept only its text. The deck's
`ExObjList` names the object's storage (`ExOleObjStg`, zlib-compressed or not),
a compound file read by what it contains:

- a BIFF workbook (`Workbook` or `Book` stream). When the workbook window shows
  a chart sheet (an Excel chart object), or the stream has no sheet directory
  and holds a chart substream (MS Graph), the chart's cached series become a
  table under the chart's title: categories down the first column, one column
  per series named by its cached name (`Series N` when it has none), points
  without a category numbered from 1, and trendlines or error bars that cache
  no values of their own left out. Otherwise (an Excel worksheet object) the
  workbook reads as the XLS reader reads it; the name of a lone sheet with data
  is dropped and several sheets keep their names as paragraphs, because a
  heading would read as another slide title. Values show their number formats
  in both cases.
- LibreOffice's `package_stream`, an OpenDocument package: a chart reads as its
  title and the data table it keeps, first row as the header, as in an ODP,
  and a spreadsheet as its tables.
- an Excel 2007 object's `Package` stream, an OOXML workbook, read as one.

Equations, documents, pictures, linked objects and controls still add no
text; a debug log names each object left out and why. The object's preview
picture stays a document asset as before, while an image inside an embedded
spreadsheet keeps only its alt text. Decompression stops at the 128 MiB
entry cap and at 512 MiB for all of a deck's objects, chart caches nested
more than 64 `Begin` levels deep are rejected, and an object's tables count
against the 4,000,000-position grid budget each time a shape shows it.
An embedded worksheet is read whole, not only the cell range the object
displays on the slide.

The chart of the reference `sample.ppt` is a LibreOffice object; its table
matches both the chart's own data and the bar heights of the preview picture
PowerPoint displays. Excel and MS Graph objects are read by the [MS-XLS] chart
cache layout and tested with objects generated from the specifications
(`crates/markitai-core/tests/fixtures/legacy-ppt`); no object saved by Excel,
MS Graph or PowerPoint was available, so their real-world layout is not yet
verified. PPTX `p:oleObj` frames are not read this way.

The presentation reader limits packages to 16,384 entries and 10,000 slides,
each XML part to 16 MiB, each asset to 64 MiB and total decompressed parts to
256 MiB. XML has per-part limits of 200,000 nodes and 127 nested elements.
The layout/master cache retains at most 8 MiB of source XML and 50,000 parsed
nodes; larger individual parts can be read within the part limits without being
cached. This avoids accumulating expanded layout trees across large decks.
Document types, escaping package paths, duplicate relationship IDs and malformed
XML are rejected. Internal targets are resolved inside the archive; external
image HTTP(S) URLs may remain references but are not fetched. Failed optional
notes, layouts, charts or images are named in warnings without discarding the
remaining slide content. These limits cover this reader's explicit passes, not
every allocation inside ZIP/XML libraries.

PDF pages with unreliable or missing native text retain their numbered marker
and emit a warning naming the page and the parser's OCR reason. Other pages are
retained. A document with neither readable text nor recoverable images fails
explicitly. An unnamed layout/GID rejection, or a scan classification caused by
declared but unused image resources, can fall back to bounded font-decoded plain
text only when execution inspection finds no used image, inline/pattern-image
warning, visibility signal or stream error, no Type3 font is present, and decoded
text passes conservative Unicode/content checks. Such
pages explicitly warn that reading order, paragraph boundaries and styling may
differ. A scan with executed raster content, garbled text, vector-text and
invisible-text reasons are never overridden by this fallback. Used image XObjects, including those inside nested Form XObjects,
are collected in page order and deduplicated by PDF object identity. Original
JPEG streams are preserved; supported 8-bit DeviceRGB/DeviceGray samples become
PNG. Images are appended to their page rather than positioned within its text.
Unsupported filters, color spaces, remapping, transparency masks and inline
images produce explicit warnings. Metadata/stream/image/asset limits bound the
additional package and image passes; they are not a claim that every upstream
parser allocation is bounded.

Eligible upright text pages also have a conservative positioned-text refinement
for document-wide heading levels, paragraph gaps, continuous styling and complete
ruled tables. Empty cells/columns are preserved; ambiguous geometry retains the
existing page reader. See [PDF layout](pdf.md) for acceptance checks, resource
limits and the cost of the additional parsing pass.

PDF inspection reports invisible rendering modes, transparent text, white text
and very small text when they accompany text operators, including inside forms.
This is a diagnostic contract: complete hidden-text removal and
image-region text grouping remain
unimplemented. The pinned reader patch suppresses nonpainting Tr3/Tr7 text while
preserving graphics state and cursor movement; other visibility and layout
heuristics retain their documented limits in [PDF layout](pdf.md).
The browser runtime is outside this module. No Python interpreter,
Node runtime, Office installation, LibreOffice or hosted extraction service is
used by the readers above.

## Explicit remaining compatibility work

Numbers table decoding accepts modern single-file ZIP and directory packages with scoped limits in [Numbers](numbers.md). A `.numbers` directory is one document, including when its contents are invalid; ordinary directories keep their existing batch/API behavior.
Office presentations and word-processing files can opt into complete page capture
and local OCR supplements through [isolated LibreOffice export](office-rendering.md);
this optional installed program is separate from the CLI binary. Native text
extraction does not launch it. Spreadsheet screenshots are explicitly unsupported.
macOS HEIF/AVIF primary images use native
ImageIO decoding, with the scoped limits in [images](images.md); other platforms
still return an explicit unsupported error. Local and static/automatic URL PDFs support explicit
page rendering, screenshots and OCR through the [PDF media pipeline](pdf-ocr.md),
including its documented accuracy gap. URL media preserves original request
identity while processing downloaded bytes without a second download.
Local image OCR is
available on macOS through [Vision](ocr.md). Standalone SVG
rasterization is implemented with bounded native rendering; referenced embedded
images can use [caption/description analysis](image-enrichment.md).
Multi-page TIFF preserves its original download and every page preview, applying
orientation before local OCR or complete-page model requests. Pixel, page and
encoded-byte budgets reject oversized documents without silently omitting pages. `supports_extension` reports local text readers; standalone image
classification and vision extraction are separate orchestration paths. See
[images](images.md), [MSG](msg.md), [markup](markup.md) and [HTML](html.md) for
their scoped contracts and limits.

The HTML reader implements article selection, structural chrome removal,
technical code blocks, scoped footnote/math recovery and structured announcement
content. It does not reproduce the original engine's complete set of site
resolvers, conversation-thread models, adaptive recovery, schema.org body
fallback or browser CSS visibility. The precise implemented boundaries are in
[HTML](html.md), [code blocks](html-code.md) and
[article selection](html-article.md). Basic success on an HTML fixture is not
evidence that its full extraction contract matches.

The [EML reader](eml.md) resolves Content-ID images within the selected MIME body
scope and retains missing or ambiguous references with warnings. The email readers
preserve body and attachments, but complete header, attachment
and layout parity is pending. XML now has structured prose and the sample fixture
is exact; arbitrary dialect parity remains open.
Office conversion can differ in whitespace, table header selection, numbering,
anchors, font-driven headings and metadata. Such differences must remain visible
in differential reports rather than being normalized away.

As in the reference, legacy DOC/PPT, RTF, ODT and ODS output ends with a newline
while DOCX, XLS/XLSX and EPUB output does not; headings carry no trailing spaces.
Presentation output has no trailing whitespace on a line and collapses the blank
runs that text-free shapes leave, as the reference's final pass does.

Word 97 files saved by the macOS exporter (TextEdit, `textutil`) declare a mini
stream no stream uses and write it inconsistently: its MiniFAT chains unused
mini sectors to sector 0, or the FAT is one sector short so the exporter's own
directory, MiniFAT and mini-stream sectors lie beyond it. Strict OLE readers,
the reference's included, reject every such file. When the original fails, a
bounded repair reads a copy that appends the missing FAT entries (extending
the directory while its entries name siblings not yet read) and detaches the
unused mini stream; a file whose mini stream holds a stream is never changed,
the original error stands if the copy fails too, and a warning reports the
repair ([record](validation/office-quality-round42.md)).

Adjacent text runs of one style render as one run, so a word the source split
into runs (RTF writes each `\u` character as its own; Word splits at revision
marks) keeps one emphasis span. A source list label that is only digits becomes
`1.`, a bullet glyph becomes the list's bullet, and other labels stay as the
source wrote them. In document formats (not spreadsheets) a table whose rows
never hold two non-empty cells, and whose cells hold a table or several blocks
with content, lays out a saved web page: its cells are written as the document's
blocks. A table nested in a data table's cell becomes one line per row. Tables
of single paragraphs keep their structure even with empty columns.

The renderer retains referenced anchors and omits unused ones. EPUB links within
the assembled book keep working through these anchors; they intentionally differ
from the reference's links to source XHTML files that are not exported. DOCX
tables keep their first row as data with a blank Markdown header. ODS uses its
first row as the header and removes wholly empty trailing columns beyond both
content and merged spans. A five-column merged title remains five columns even
when its body data occupies only three; this differs from the reference's
three-column rendering of the ODS fixture. RTF heading bold markers are omitted while other emphasis is
retained. Hidden XLS/XLSX worksheets are currently omitted by the upstream parser
and reported explicitly. XLSB sheet metadata, older XLS code pages other than
Windows-1252, exact presentation image encoding, PDF table
layout and PDF image placement require further compatibility work.

## Error and output principles

- A malformed or unsupported input returns an error. An error description is
  never persisted as a successful Markdown document.
- Native document parsers do not call remote OCR automatically.
- HTML links are restricted to ordinary HTTP(S), mail and telephone references.
  Relative references become absolute when the caller supplies a source URL; a
  saved web page's links resolve against its `<base href>` or canonical link,
  while its image paths and other local files' references stay relative. Script links lose the destination while
  keeping their visible label.
- Format adapters return embedded bytes rather than downloading remote images.
- Text readers retain original Markdown frontmatter for the output layer to
  merge according to its public contract.

## Reference audit and acceptance corpus

The [frozen r9 format audit](validation/formats-recovery-r9.md) of `0022e09`
records 17 strict passes out of 24 fixtures, with four output drifts and three
expected image-only rejections. All 24 records and input hashes are unchanged
from r8. Metadata matches for all 21 successful conversions and assets for
19/21; PDF and PPTX assets still differ. Complete format parity remains open.

The reference implementation was inspected read-only at
`/Users/example-user/work/markitai`. At the audit it contained 209 Python source modules
and 278 test modules. Converter, web extraction, LLM and provider code accounted
for approximately 8.8k, 15k, 9.8k and 5.5k lines respectively. These figures
describe migration scope and are not performance measurements.

The existing project supplies these useful compatibility inputs:

| Reference path (relative to `packages/markitai`) | Purpose |
| --- | --- |
| `tests/fixtures/` | Office/PDF/email/markup/structured text and legacy format examples |
| `benchmarks/docs_snapshots/expected/` | Four normalized end-to-end Markdown snapshots: PDF, DOCX, PPTX, XLSX |
| `tests/defuddle_fixtures/` | 209 paired HTML and expected Markdown cases |
| `tests/fixtures/web/*.expected.json` | Semantic thread and required/forbidden text contracts |
| `benchmarks/local_fixtures/` | Two additional website fixtures |
| `tests/unit/test_structured_text_parity.py` | Encoding, CSV quoting/ragged rows and notebook failure cases |

The recorded reference web benchmark dated 2026-09-22 had a mean score of 96.2
for 209 cases. Its guardrail floor of 86.58 is a tolerated regression boundary,
not the reference quality. The separate two-case corpus scored 99.47. These are
reference records only; this Rust engine has not passed those full gates.

The reference repository's `scripts/benchmarks/` also contains a 49-case cold/warm
performance harness and a 40-case quality audit of Markdown, metadata, image
hashes, dimensions, frontmatter and output assets. Use a frozen source identity
and isolated output directories. Run timing sequentially without concurrent
builds. Record failures and unsupported cases in the denominator. Compare speed
only after a case's content and asset checks pass.

## Dependency choices

The native Office parser dependency is [anydoc](https://github.com/firecrawl/anydoc),
whose Rust API exposes both Markdown and a structured document with embedded
assets. It already backed legacy DOC/PPT in the reference implementation. OOXML
presentations now use their own bounded package reader to retain slide identity
and the reference text-frame contract; other Office formats keep the shared
renderer. Its
native PDF reader cannot do OCR. This project adds its own Office renderer to
avoid discarding embedded images and to control Markitai's output contract. The
PDF adapter calls pdf-inspector's per-page API directly so that one OCR-required
page does not discard readable pages. A separate lopdf pass retrieves embedded
image streams and reports visibility signals without an external rendering engine.

HTML uses [scraper](https://docs.rs/scraper/) for an HTML5 DOM and selectors and
[htmd](https://docs.rs/htmd/) for Markdown serialization after explicit cleaning.
Email uses [mail-parser](https://docs.rs/mail-parser/) to handle MIME and transfer
encodings. Cargo.lock fixes the resolved versions. Changes to dependency versions
must run the corresponding output contracts.

The reference project previously evaluated Calamine and a PDF layout alternative
without adopting them. Their availability alone is insufficient justification
for replacing the committed behavior; compare cell formatting, layout and assets
before selecting a different backend.
