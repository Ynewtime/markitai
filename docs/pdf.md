# Native PDF text and layout

The PDF reader uses `pdf-inspector` for decoding and page-level reliability
decisions, with `lopdf` for bounded content inspection and embedded images.
There is no Python runtime, external converter or implicit OCR fallback.
Pages are delimited internally by `<!-- Page number: N -->`; final Markdown omits
these comments unless `--page-markers` (`output.page_markers=true`) keeps them.

## Hidden text policy

`security.pdf_sanitize` applies to local and downloaded PDFs, including the
native text supplied to page OCR, screenshots and LLM enhancement:

- `off` suppresses hidden-text security notices and keeps the native reader's
  existing visibility rules. It does not make invisible `Tr 3`/`Tr 7` text
  visible, and does not disable decoding or reliability warnings.
- `warn` (the default) keeps that same extracted body and reports suspicious
  white, transparent, very small or invisible text and inspection limits.
- `remove` filters suspicious text show operators before page layout, keeping
  their advances and clipping state so later visible text stays in place.
  It changes the extracted body in memory; the input PDF, embedded pictures
  and screenshot pixels keep their original bytes. A matching visible copy
  of the same words is retained. A shared Form is filtered per invocation,
  with the nearest complete resources dictionary taking precedence.

Removal uses effective text size (at most one point), separate fill/stroke
opacity (at most 0.01 on every active paint channel) and white device-color
text. White text is only filtered when no path, image, shading or other
background paint was found on that page; white text on a black panel stays
visible. Unsupported color spaces, soft masks, blending, malformed resources
and inspection or rewrite budgets are reported honestly. A page whose initial
inspection was incomplete keeps its original body. Other uncertain paint may
be retained with a notice that removal was incomplete.

A searchable scan's already accepted OCR layer is retained in all three modes
under the rules below. Filtering never promotes newly masked text into an
accepted OCR layer. Normal documents need no filtering or second layout pass;
a valid direct resources dictionary inherited from a Pages ancestor is
materialized only in memory to make it readable by the pinned native reader.

This is bounded filtering of the extracted text, not a general PDF security
sanitizer. It does not certify content, cleanse attachments, annotations,
metadata, links or image pixels, or prevent prompt injection in visible text.
Use OCR over page pixels when those pixels must be the source of the text.

## Downloaded PDFs

Static and automatic fetches can hand one bounded HTTP response directly to the
same PDF media pipeline as local files. PDF MIME types identify the representation;
generic downloads and PDF paths additionally use a bounded header check. Explicit
HTML/text responses keep their declared interpretation. Redirects and extensionless
downloads need no temporary input file or second download when media is enabled.

The original URL remains the public source and naming input. A redirected final
URL is recorded with sensitive components redacted. PDF responses are never stored
as extracted HTML/text cache entries, including when subsequent PDF parsing fails.
Explicit browser PDF responses stream through the same authenticated CDP
session into the native reader, retaining `playwright` strategy metadata.
Remote extraction services retain their selected backend.

Downloaded PDFs support requested native OCR and page screenshots. Screenshot-only
PDF output retains its Markdown and page references, unlike webpage capture-only
output. Configuration with only `screenshot_only=true` does not implicitly enable
PDF screenshots; the CLI flag still enables them. URL `pure` takes precedence over
screenshot-only for model requests: it sends extracted text without image blocks or
appended page comments. Ordinary visual requests include every captured page.

This is an intentional extension: the reference URL converter did not pass local
PDF OCR/screenshot options through to the PDF converter. No public request/result
schema changes are required. Native media still has the platform and accuracy
limitations described in [PDF rendering](pdf-rendering.md) and [OCR](ocr.md).

Layout refinement runs only after the existing page checks. A page requiring
OCR, containing suspicious hidden text, read from an embedded OCR text layer
(see [Searchable scans](#searchable-scans)), or having incomplete content
inspection does not enter refinement. Text size for the hidden-text check is the effective
size after the text matrix, transformation and Form matrices: Quartz writes
`1 Tf` and scales with the text matrix, which is ordinary 12pt text. The existing narrowly guarded font-decoded recovery
for a false scan verdict remains in place. A missing or unreadable page keeps
its place, with an explicit warning (and its page marker under `--page-markers`).

The pinned dependency has a small, tracked policy patch under
`vendor/pdf-inspector`. Its original license, bundled character-map license and
per-file upstream hashes are retained. This patch changes extraction policy;
it does not weaken the layout character-agreement or page reliability gates.

Chrome (Skia) prints web fonts it cannot embed as Type3 fonts whose mirrored
`FontMatrix` is paired with glyphs drawn y-down. Their glyph side is now read
from the `FontBBox` as well as the matrix, so such runs stand on their baseline
instead of one font size below it, where they interleaved with the embedded
fonts of the same line.

Page-number cleanup recognizes complete folio expressions such as `Page 3 of 10`
and retains substantive paragraphs beginning `Page N`, including prose following
`Page 3 of 10`. Reader-generated headings and emphasis around complete folios
do not prevent cleanup; code and HTML literal blocks are preserved. Positional
running-furniture detection remains separate.

A lone one-to-four-digit number in the top or bottom page band is still removed
as a folio, except when it shares its baseline with other text and no folio-like
edge number in the same band corroborates it. Corroboration is another page with
the same value or the same value per page step (one or two printed pages per PDF
page), or a same-page neighbour one apart on its baseline (a two-up spread). Only
explicit folios and isolated edge numbers are evidence. That keeps the numeric
cell of a table's first or last row on each page of a printed table (for example
`361` beside `Row 19`), which was previously deleted. Running and spread folios
are removed as before. A number that only one page offers is still decided by the
upstream rule, as is a table value that happens to repeat across pages. Folios of
non-contiguous excerpts that share their baseline with footer text can now remain
as text; this errs toward keeping content. Page-selected extraction in the
dependency reads other pages to decide; Markitai's reader analyses the whole
document. Authored regression tests require every paragraph in source order.

Text rendering mode persists across `BT`/`ET` text objects and nested `q`/`Q`
graphics states. Normal extraction excludes `Tr 3` invisible text and `Tr 7`
clipping-only text in pages and Form XObjects. Hidden text still advances the
text cursor, so following visible glyphs keep their positions. An `ActualText`
replacement whose glyphs are all nonpainting cannot restore hidden text. The
dependency's explicit OCR-layer path may still request mode 3; clipping-only
mode 7 is never treated as an OCR layer. The only mode-3 text the core reads is
a searchable scan's OCR text layer that passes the checks below; it retains the
existing guard against plain-text recovery of suspicious pages.

Text drawn outside the visible page area (CropBox intersected with MediaBox) is
left out of the page body, as a viewer never shows it, and a warning gives the
number of items left out. A word run counts when its whole extent lies more than
6 points beyond the area; short glyph fragments and runs that continue an
on-page line stay, because rotated display text can leave them there.

These rules do not establish complete rendered visibility: transparency,
blending, soft masks, occlusion, arbitrary clipping and mixed-visibility
`ActualText` spans still need broader interpretation. Pages with visibility
signals retain an explicit warning and do not enter layout refinement.

## Searchable scans

A searchable scan (OCRmyPDF, Tesseract, Acrobat "searchable image", a scanner's
own OCR) is a page image with the recognizer's words laid over it in render mode
3: invisible, but placed where the printed words are. Markitai reads such a page
from that layer instead of reporting it as a scan needing OCR, but only when all
of the following hold, and never silently.

The page reader judges the layer by the geometry its content scan already
follows (no further pass over the content): every text-showing operator the page
and the forms it invokes execute runs in mode 3 (one visible or mode-7 operator
refuses the page: mixed pages keep their visible text only); the images drawn
cover at least half of the page; every operator has a font size and a position,
and lies on the page; the left half of each estimated glyph box (half an em per
byte, so a two-byte code's box is twice its glyph) lies within one cell of the
images on a 64 × 64 grid, and the images cover 90% of the cells of the layer's
bounding box, so no line runs off the scan; no more than a tenth of the
operators are under 2 points or over a quarter of the page height (sizes include
the text matrix and transformation, so `1 Tf` scaled by `Tm` is ordinary text);
the text is no denser than 400 bytes per square inch of image and its glyph
boxes cover the image at most three times over (no stuffing). The layer's text,
read as the page's text, must then carry at least 40 letters and digits, pass
the garbage and decoding checks every page passes, and come from fonts whose
glyphs have identities. The producer named in the file plays no part.

This module then requires its own inspection of the page to agree: all of the
page read (no budget, nesting or decoding failure) and no visibility signal but
the invisible rendering mode — white, transparent or one-point text elsewhere
keeps the page a scan. When the page draws exactly one image XObject whose
samples it can read (JPEG; Flate, LZW, run-length and ASCII filters; 1- to
16-bit gray, RGB, CMYK, indexed or stencil samples), the layer is compared with
the image's ink on the same grid: the paper is the 90th percentile of the
image's luminance, a pixel darker than 60% of it is ink, and a cell is print
when 1–50% of its pixels are ink (more is a photograph's shadow, a fill or a
border). At least 90% of the cells the layer's text surely covers must lie
within a cell of print, and at least 60% of the print cells within a cell of
where the text may reach. A layer that fails is not used, with a warning that
says how much of it lies on print; the page then reads as before. JBIG2, CCITT
fax, JPEG 2000 and inline images, an image over 32 Mi pixels (a JPEG's own
header is checked before its pixels are decoded), a page drawing several
images, and an image too dark to tell paper from ink leave the layer unchecked:
it is read, and the warning says why it could not be checked.

The thresholds were measured on 59 Tesseract 5.5 layers over 200–300 dpi gray,
RGB JPEG, bitonal and OCRmyPDF-style (layer in a Form XObject) renderings of
Chrome-printed corpus pages: at least 97.8% of each layer's cells lay on print
and all of the print under the layers' reach. Another page's layer laid over a
page of the same layout reached 86.7%, injected words longer than the lines they
covered 89.0%, and words over blank paper 7%; all three are refused.

A page read from its layer has no visibility warning; one warning names every
such page, checked alike, its producer and that recognition errors in the layer
are kept (`PDF pages 1-3: the text was read from the invisible OCR text layer
laid over each page image (Tesseract 5.5.3); it lines up with the text in the
image, …`). Metadata `ocr_layer_pages` lists the pages. Such a page is read
without headings, code, bold, italic or underline (an OCR layer's font is a
stand-in and its sizes are line heights), does not set the body size other pages'
headings are judged against, does not enter layout refinement and takes no part
in running-header detection, since its text has no visible positions. Its page
image is extracted as an asset as before.

The comparison checks where the layer's words lie, not what they say. A layer
whose words differ from the print but sit on its lines — another page's layer
over a page of the same layout, or deliberately injected text of the same line
lengths — is read; the warning and metadata always say where the text came from,
and such text has the trust of an OCR engine reading the same pixels. Only
recognizing the page again can tell them apart (see [PDF OCR](pdf-ocr.md)).
Whole-document classification (`detect_pdf_type`) still reports such pages as
`invisible_text_layer`: they show a raster and nothing else.

## Typed pages and final assembly

The reader separates extraction from document assembly. `extract_pages` returns
one `PdfPage` per source page in source order, with a one-based page number, the
native Markdown body, the reader's OCR verdict and reason, and the names of
successfully extracted assets. The body has neither generated page markers nor
appended image references. `visibility_suspect` comes directly from graphics-state
inspection; orchestration does not recover that decision by parsing warnings.
`ocr_layer` marks a page whose body is its searchable scan's OCR text layer and
says whether the layer was checked against the image; the warning naming those
pages, and the `ocr_layer_pages` metadata, are written at assembly for the pages
still read from their layer, so a page recognized again by OCR drops out.
Embedded objects shared between pages still produce a single asset, with each
page retaining its own reference.

The ordinary `extract` entrypoint calls `finish`, which preserves page markers,
image-reference order, metadata, inspection diagnostics and the final empty-content
error. An unreadable or blank page is available to a media/OCR caller before that
final check. Missing-text warnings are deferred until assembly, so replacing a
page with successfully recognized text also removes its stale OCR-required
warning. `ocr_completed` distinguishes a successfully recognized blank page from
one still awaiting OCR. The caller records the blank result explicitly; it does
not manufacture text.

`finish_with_media` permits an empty native body only for the orchestration path
that has actually rendered pages or completed OCR. It is not the default reader
entrypoint. Per-asset OCR text is appended after its matching page image reference;
an unrelated asset name cannot inject text into another page. Optional screenshots
use the final published basename in a trailing `Page N` image comment. Basenames
are encoded as raw UTF-8 URI path data, preserving literal percent signs, query
and fragment characters while preventing a filename from closing the comment.

Inspection and recovery warnings remain independent of OCR success. Callers may
append warnings and metadata before assembly; they replace any reader-only
limitation message after assembly when a separate backend supplies that feature.
The typed extraction itself adds no renderer, OCR runtime or new public JSON
fields. Validation of the media backend is recorded separately from these reader
and assembly tests.

## Layout reconstruction

Reliable pages receive one additional positioned-text pass. Font sizes are
weighted by character count across eligible pages; larger sizes determine a
consistent heading hierarchy across the document. Physical line gaps distinguish
wrapped paragraphs from separate paragraphs. Adjacent runs retain bold, italic,
underline and strikeout styling, including a continuous style across a wrapped
line.

Ruled tables are reconstructed from complete horizontal and vertical borders.
Stroked lines and thin filled polygons are supported, including transformed
coordinates and a nonzero page origin. The complete grid determines the column
count: empty cells and empty trailing columns remain. Multiple physical text
lines in a cell use `<br>`. A row spanning a border, an incomplete grid or
overlapping table regions prevents speculative reconstruction. The first row with
text is represented as a Markdown table header (a row with nothing in it above
it heads nothing); this representation does not establish that an untagged PDF
declared it a semantic header.

A tagged PDF's tables come from its structure tree on each page, as in the
page reader's whole-document conversion (`8b9747d`): a table drawn without rules
whose cells wrap and are vertically centred is otherwise read from its text's
alignment alone, which splits it into broken rows. Text extraction keeps the
runs of two structure-tree cells apart (`03deb93`): cells a few pixels apart,
as in the browser's default table style, would otherwise merge into one item
that keeps only the first cell's marked content. A fully tagged table among
long paragraphs is used when it holds 80% of the text inside its own bounds.

Where the structure tree holds no table cells on a page (an untagged PDF, or a
table Chrome tags as layout because it has neither header cells nor borders),
the layout pass reconstructs a table drawn without rules from its geometry when
the evidence is clear. Columns are the left edges at which two or more rows with
three cells or more, one gap of at least two em among them, start a cell; nine
in ten of those rows' cells must start at such an edge, and a label column
further left may come from the wrapped labels between them. Rows are told apart
by their spacing rather than by a shared baseline: one cell's lines follow at
the table's smallest line step, which must match the running text's line pitch
or have another column's line centred between two of a cell's lines; a step 12%
(at least two CSS pixels) larger starts a row, and a step between the two
declines the table. Cells whose text overlaps vertically form one row, so
vertically centred cells of different heights stay together. A run the
extractor merged across two neighbouring cells is split at the space nearest
the column edge, and a cell's wrapped lines are rejoined with spaces (after a
line-end hyphen that follows a letter, without one). The table needs three rows,
each of two cells or more, two rows of three cells, a cell that wraps, no cell
overlapping two rows of another column, and no more than a third of its cells
over twelve words. A line of one cell just above or below it stays with the
text around. Rows at the top of the next page that keep to the columns and line
pitch of a table ending the previous page continue it (see below); the first row
of any other table is its header, as for ruled tables. Tables of single-line
rows and of cells a few pixels apart remain with the page reader's
alignment-based detection, and a page with any other side-by-side text that
region segmentation (below) does not divide keeps the page reader's output.
Single-line rows were not given a geometric detector
of their own: column alignment is all the evidence they offer, and on the
adversarial layouts the page reader's alignment-based detection, which
already reads such tables, also makes tables of a footer link grid, a row of
statistic cards and newspaper columns. The page reader no longer makes a
table of the words of justified columns: a hypothesis of five columns or
more, seven in ten of its filled cells one word, whose runs stand closer in
a row than one and a half body ems (spaced words, not a table's columns),
is text.

Columns set side by side read one after the other. Before lines are
assembled, the page is cut recursively, as an XY-cut does (three levels at
most): into bands at whitespace across its width at least half again the
running text's line pitch, and each band down its gutters. A gutter is
whitespace at least an em (and 8pt) wide between the runs that do not span
the band (60% of its width), half again as wide as the gaps between the runs
of its lines mostly are (their 95th percentile: a justified column spreads
its words), with two lines and three runs or more on each side (a ruled
table counts with its rows) standing beside each other for two line
pitches or more. Lines spanning a candidate gutter (a title over the
columns, a paragraph across the page between them) cut the band above and
below them when nothing stands beside them, and otherwise rule the gutter
out. Where half the lines of the shorter side or more share a baseline with
the other's (a borderless table's rows, a form's labels and values), the
gutter holds only when both sides read as running text (three lines or
more, half of them filling seven tenths of the side's width, three and a
half words a line, less than half of them code) and their blocks do not
start at the same heights, as table rows and a grid's cards do; a borderless
table found across the whole page is an obstacle no gutter crosses. Bands
read top to bottom and a band's columns left to right (right-to-left pages
never get here); a band whose gutters match those of the band above
continues its columns, unless its columns start at one height, as the next
row of a card grid does. Each region is then read as a page is, with the
document's heading sizes: its lines, borderless tables, code and lists.
Only the first region may continue the previous page's table, and only the
last one's closing table continues on the next page. A page without a
gutter is one region and reads exactly as before. A résumé's contact column,
a bulleted sidebar beside an article and a newspaper's justified columns
read column by column; short lines on an article's baselines (a list of
links beside it) still leave the page to the page reader.

A table that a page break cuts is read page by page, and each part would be a
table of its own. A table ending in the lowest fifth of its page continues when
the next page opens with a table of the same columns: for a ruled table, column
borders within 1.5pt and a first row that repeats the header or is not set apart
as one (bold throughout, above a row that is not); for a borderless table, rows
that keep to its columns and line pitch. When the document is assembled, the
continuation's rows join the table they continue, so they stand under the marker
of the page the table starts on, and the next page keeps only what follows them
(or only its marker, without a missing-text warning). A part that repeats the
header row of the table ending the previous page joins it too, also when the page
reader read both and when running headers or footers stand between (one-line
blocks that open, or close, two pages or more and half of them, digits aside);
layout geometry alone does not reach across such lines. The repeated header row
is dropped; any other first row becomes a body row. Other text between the
parts, another column count, a header of empty cells, a single column or a page
recognized by OCR keeps the tables apart. A part on its own (per-page
extraction, or a page recognized by OCR next to it) is headed by its first row,
so no table has an empty header row. Page images stay after their own page's
text.

Before replacing page Markdown, decoded alphanumeric character counts must agree
with the existing reader. This is a conservative agreement check, not proof that
two decoders are independently correct. Link annotations carry targets, not page
text, so they do not block refinement; the layout pass renders them as links
(see [Links, escaping and block structure](#links-escaping-and-block-structure)). The body
size that headings must exceed ignores text inside detected table grids and
fixed-pitch text, so a table- or code-heavy page keeps its prose as paragraphs.
Unknown form-field semantics,
rotated pages or text, invalid geometry, right-to-left text (whose runs need the
page reader's bidirectional ordering) and ambiguous side-by-side prose retain
the existing page reader's output. A page can therefore remain unchanged even
when another page gains layout fidelity.

Fixed-pitch text becomes code. A fixed-pitch line opens a fenced block at the
start, after a heading, when it is set off from the text above by more than
1.7 em, or when the next line is fixed-pitch too (AppKit prints `<pre>` without
margins); a single one at ordinary leading continues its paragraph, as a wrapped
inline literal does. Inside a block a URL line stays code, a gap of half again the
block's line pitch (two em before a pitch is known) restores a blank line, and
each run is placed at its column in glyph advances from the block's left edge,
which restores indentation and alignment. A fence is longer than any backtick
run inside the code. Fixed-pitch runs in prose become inline code spans; mono
links, bare URLs and runs holding a backtick stay text.

Browsers paint list bullets as shapes rather than characters. The geometry pass
also collects compact painted marks of 1–8pt with the colour they are painted
in: filled shapes, and stroked rings drawn with curves only (a stroked shape
with straight sides is a checkbox or a frame). A mark of 0.15 to 0.5 times the
size of a line's first run, ending no more than two em before it and at most
half a point inside it, centred between its baseline and 0.8 em above it,
painted in a dark neutral, in the run's colour or in the colour of most of the
page's text (a legend swatch or a status dot has a colour of its own), with no
other text on that baseline ending within three em before it, makes the line a
list item (`pdf_inspector::painted_bullets::targets`). The page reader is given
the same marks of each page, outside ruled tables and in the page's own
coordinates, and reads such a mark as a bullet character, so a page left to it
keeps its painted lists. Within a region, a list item beside other text on its
line, or lines of two columns whose baselines interleave (less than three
quarters of an em apart without overlapping horizontally; markers, runs of fewer
than four characters and code aside), leave the page to the page reader: a
bulleted sidebar beside an article that region segmentation leaves whole does not
run its items into the article's lines. Bullet characters
remain list markers. A number (`3.`/`3)`, up to three digits) opens an item
where a new line could start: first in the flow, inside a list, after a gap or
heading, or offset from the line above. A number on a line of its own marks the
next line. Items nest by their marker's offset (a browser indents 40px),
consecutive items form one tight list, and a heading keeps its level. Super/subscript runs keep their
anchor's line and render as `<sup>`/`<sub>`; digits the page reader fuses into
their word stay Unicode superscripts. Headings need more than 1.15 times the body
size, so a browser `<h3>` (1.17 em) is one.

The refinement limits are 250,000 positioned items and 16 MiB decoded text across
the selected document, 20,000 items per page, 200,000 operations and 8,192 rule
edges per geometry pass, 64 graphics-state levels, 32 tables, 32 columns and
4,096 cells per table. Existing 64 MiB expanded page/Form inspection and image
budgets remain. Positioned-item limits are checked after the dependency returns;
they are not a claim of a hard allocation limit inside that dependency.

Each page's content streams are expanded and parsed once. Inspection expands
them within the 64 MiB page budget, keeping where each stream lies, and parses
their operations; it first applies the existing visibility, Form and warning
checks, and only an eligible page's same parsed operations are then read for
table borders and painted bullets. When the page reader's document is the one
inspected, inspection hands it the same operations, from which the reader walks
the page's text runs, and the same expanded streams, from which it takes the
page's OCR signals; the reader then neither expands nor parses that page again.
The reader still reads a page itself, with the same result, when its content
holds a comment (the reader strips comments before it parses) or more than a
million operators, or when the reader repaired the file on load and this module
inspects its own load. Raw expanded page streams and parsed operations are
released at the end of that page's preparation; the reader keeps the page's runs
until the positioned-text pass completes, as it did when it walked them itself.
Only the page's frame and bounded grid coordinates survive until then. Form
inspection retains the existing shared 64 MiB page/Form byte budget, 256 content
inspections and 32 nested Form levels, and, like operator filtering, leaves a
content stream of more than a million operators uninspected with a warning,
which makes the page's inspection incomplete; Form operations are not used as
speculative table borders. Pages with incomplete inspection retain the original
warning and fallback behavior.

## Links, escaping and block structure

A `/Link` annotation whose action is a `/URI` (also when the URI string is an
indirect object, as Quartz writes it) becomes a Markdown link over the runs it
covers in the layout pass: `[link text](https://…)`, or `<https://…>` when the
text is the address. Only `http`, `https` and `mailto` targets are carried;
spaces and angle brackets are percent-encoded and parentheses escaped. A link
with another target (`javascript:`, `file:`, a relative or in-document
reference) keeps its text without a link. A run belongs to a link when the
middle of its height is inside the link's box and four fifths of its width
overlap it; punctuation ending the run outside the box stays outside the
link. A run the extractor merged with linked words (`See this report for…`)
is split at word boundaries: a word belongs to the link holding its middle
and half its width, sentence punctuation and unpaired quotation marks or
brackets at either end of the linked words stay outside, and the split is
made only when the words placed in each link span most of its box (half
their union), since word positions inside a run are estimated from typical
(Helvetica) glyph widths. The underline a browser draws under links is link
styling: `<u>` remains only for underlines that four fifths of the run's
width are not under links. A heading keeps a link's text without its target:
a link there is navigation (a site's name over its home page, a post's
permalink), and the heading may name the document. The page reader does not
render links.

Prose is escaped as the other readers escape it: `<` only where it would
open an HTML tag, comment or entity (`Vec\<String>`, `\<div>`), so `a < b`,
`x -> y` and `(x) => x` read as written instead of as `&lt;`/`&gt;`. A
paragraph whose text starts like a block of its own (`# 1`, `+ note`,
`2026. The year`, `> quote`) has that mark escaped.

Headings set at the body size (a browser's `<h4>`–`<h6>`) are found by their
face: a block of one or two lines whose runs are all in a face other than the
body text's that either sets text at a heading size elsewhere in the document
or is bold, neither italic nor fixed-pitch, set off from the text before and
after it (paragraph spacing, a heading by size, the start or end of the flow),
of up to 14 words and 120 characters, not ending in `.`, `,` or `;` and,
past three words, not in `:` (a lead-in). It ranks below every heading size.
The extractor names a run merged from several faces after its first, so
such a line can be a bold label followed by body text; the lead-in and
set-off rules keep the common cases text.

A line ending short of its column starts a new paragraph even without extra
spacing: wrapped text fills every line but a paragraph's last, so when the
next line's first word and an em and a quarter would have fitted after it
(the column's right edge is the farthest any line starting at the same left
edge reaches on the page), the next line begins a new block. Rows of a link
list, byline and metadata lines and paragraphs printed without spacing are
separated this way. A line continues its paragraph when it ends in a hyphen,
when it is a label of its own (at most four characters without a letter: a
footnote's number), when the next line starts in lower case, and when it has
five words or more and ends within an em of the line below or of the line
above at the text's pitch (text beside a floated figure keeps a measure of
its own).

Lines starting with the same symbol followed by a space (check marks,
crosses, arrows, stars, geometric shapes, pictographs), two lines of the flow
or more, are list items that keep the symbol: `- ✅ True HEPA filter`. In the
page reader, a check box or check mark (`✅ ✔ ✓ ☑ ☐ ☒ ✗ ✘ ❌ ❎`) followed by
a space starts a list item the same way.

A short run raised off a line that the extractor did not read as a
superscript (a reference list's `^ a b` back-links, a footnote's number in
another face) joins the line it is raised from, rendered `<sup>`: at most
twelve characters, all smaller than that line's text, raised at most 0.6 of
its size, within its extent or two em before or after it, and overlapping
none of its runs. On a baseline of its own it would otherwise form a line
that joins the paragraph above. In the page reader, a footnote's mark
opening its line (up to three digits, one or two of `*†‡§¶#`, or one
lower-case letter), smaller than the note's text and raised off it, is set
apart from the text when 0.12 em or more separates them:
`1 Corresponding author`, not `1Corresponding author`. Set against the text,
or inside a line, a raised run attaches to its neighbours as before.

The page reader keeps a paragraph set in a heading size together: three or
more consecutive lines at the same heading level, each at wrap spacing below
the one above, with more than 30 words, are one paragraph (an abstract set
larger than the body text), not a heading per wrapped line. A line set at
least 1.1 times the body size wholly in an upright face other than the body
text's is a strong signal for the reader's standalone-heading rule (a
browser's `<h3>`, 1.17 em, in a Type 3 face whose name says nothing of its
weight), and a line larger than the one above it is no wrapped line of a list
item. The page reader ranks headings on one ladder for the document: the
sizes 1.2 times the body size or more that start lines of text on any of its
pages (largest first, sizes within 0.5pt as one, four at most; bold lines 1.05
times the body size when no size qualifies). A page whose own heading sizes
are on that ladder, read at the document's body size, ranks by it, so a
section heading the size of the second tier stays a second-level heading on
a page without the title above it; a page whose body size the reader corrects
from its own text, or with a heading size the ladder lacks, keeps its own
tiers. The layout pass ranks by its own document-wide sizes (above), so a
reader page and a layout page of one document can still rank a size
differently.

A sparse page, 12 to 19 runs, that the page reader's column detection
(20 runs or more) keeps whole is read in two columns when its runs show one
clear gutter: the widest gap between their extents, 24pt or more and clear
of the outer tenths, three runs on three baselines or more each side, the
sides beside each other over two fifths of the text's height, and, where
half the shorter side's lines share the other's baselines (a form's labels
and values), both sides running text. Its columns then read one after the
other.

A running page header is removed like the page numbers the reader removes:
the first block of the page Markdown, the same text (emphasis, white space
and digits or separators at either end, a running page number, aside) on
more than half of the pages with text and three at least, which on each
page where it is removed is the page's topmost text, on one baseline in the
top 15% of the page and smaller than that page's body text. Positioned text
is read again only for a document with such a candidate. A paragraph that
opens every page at the body size is kept; footers other than page numbers
are not removed.

A document information `/Title` that names no document is not taken as the
document title, so the first heading names it: the name of the source file (a
name with a document or image extension, `multi.html`, `report.docx`; an
Office print driver's `Microsoft Word - …`; a single lower-case slug, `code`,
`layout-table`, as a browser writes for a page without `<title>`), or the name
an application gives a document nobody named, in any case and with a number
or separators after it: `Untitled`, `Untitled Document`, `Untitled-2`,
`PowerPoint Presentation`, `Title`, `Sans titre`, `无标题`, and, numbered,
`Document1`, `Presentation1`, `Book1`, `文档1`. `Untitled Love Song`,
`Document Management Policy` and a bare `Book` are kept.

Clearly bounded vertical bar charts are preserved as PNG crops in their body
position, including axes and legends, without `--screenshot`, OCR or an LLM.
Detection requires a local rectangular clip, multiple aligned bars of different
heights, grid lines, an axis and short labels, with no crossing or side-column
text. Only a successfully rendered and placed crop replaces the scattered labels;
otherwise the text remains. The crop does not provide an inferred data table.

This uses the existing CoreGraphics/hayro renderer at 144 DPI, opened only when
needed, once per document and once per candidate page. Limits are four charts per
page, 8 million rendered pixels per page and 64 million per document, within the
shared asset-byte budget. Unsupported drawings, rotated pages and ambiguous
layouts stay on the ordinary text path. `security.pdf_sanitize=remove` disables
automatic chart crops so image generation cannot bypass hidden-text removal.

## Forms, annotations and protected files

A PDF encrypted with an empty user password (an owner password that only
restricts printing, copying or editing) opens as other readers open it,
whether it is encrypted with RC4 (40 or 128 bits), AES-128 or AES-256, and
also when its encryption dictionary is written in the trailer itself, as
MuPDF and PyMuPDF write it: lopdf read the dictionary only through a
reference, so such a file loaded with its strings and streams still
encrypted and was reported as needing a password. A PDF that asks for a
password fails with `the PDF is encrypted and needs a password to open`.

A form field's value is read where its widget stands, as the text beside
it labels it: after the nearest run on the widget's row to its left (with no
other widget between), else after the nearest run just above it across its
width, following a colon when the label ends in a letter, a digit or a
bracket (`1. First name: Maria`, `Email: m@x.org`); a multi-line value runs
on, and a line holding a value is no heading in the page reader. A check box or radio
button shows `☒` or `☐` (its appearance state, `/AS`, else the field's
value), as Word's and RTF's legacy check boxes do, and is labelled first by
the run just to its right (`☒ I agree to the terms`). A field's kind, flags,
value and options pass down to its widgets, so each widget of a radio group
or of a field shown twice reads its own state or the shared value. A choice
shows the display text `/Opt` pairs with its export value. A widget no text
labels keeps its own rectangle and is labelled `label: value` with the
field's tooltip (`/TU`), else the last part of its name (`f1_01` for
`topmostSubform[0].Page1[0].f1_01[0]`), never the full name; such values
follow the page's text. Password fields, push buttons, signatures, empty text
fields and hidden widgets show nothing. A page of more than 2,000 widgets or
20,000 runs keeps every value under its field's own label.

A FreeText annotation draws its text from its own appearance, outside the
page content: each line of its `/Contents` (else of its rich text, `/RC`,
without the markup) is read as page text at the annotation's `/DA` size,
inside its box (`/Rect` less `/RD`) and its alignment (`/Q`), before the
first run below it. A Text annotation (a sticky note) and the other markup
annotations reviewers comment with (highlights, underlines, strike-outs,
squiggles, carets, shapes, lines, ink, stamps and file attachments) carry a
comment that is no page content: each page's comments follow it, after its
images, as a `**Comments**` section of one quotation each, in the order of
the page's `/Annots` with each reply after the comment it answers:
`> Comment (Reviewer) on "thirty days": Should this be 45 days?`,
`> Reply (Author): …`, `> Stamp: Approved`. A text markup annotation quotes
the words its `/QuadPoints` boxes cover (half of a word's characters or
more, their places estimated from Helvetica's widths across each run; at
most 200 characters), except on a turned page; an annotation without
`/Contents` or rich text, a hidden one, a pop-up (which shows another
annotation's comment) and media annotations give no comment. At most 1,000
annotations of a page are read. Link annotations and form widgets are read
as described above.

## Right-to-left text

Pages holding Hebrew, Arabic or another right-to-left script are read by the
page reader, which puts each line's runs into reading order with the Unicode
Bidirectional Algorithm. Glyphs are painted in display order, and a line can
display the same glyphs for a left-to-right and a right-to-left paragraph:
a Latin word at the left end of a Persian line is the first word of the one
and the last of the other, and a full stop at the left end closes a
right-to-left sentence. Rules P2 and P3 take a paragraph's direction from its
first strong character in reading order, which is what is unknown here, and a
producer may set the direction regardless of the letters (a browser lays a
page out left to right unless it says otherwise, whatever its script). So a
paragraph's direction is read from its alignment: lines of one paragraph
share the edge they start from. Lines are linked into paragraphs (the nearest
overlapping line below, a size within a fifth, a step within 2.5 em, a shared
left or right edge, and no step a quarter wider than the one above or below);
a pair of lines sharing its right edges with left edges at least 1.5 em apart
votes right to left, the mirror case left to right, and a paragraph reads the
way its votes go. A paragraph whose own lines say nothing (one line, lines of
one width) compares itself with up to three overlapping lines above and below
within 8 em; a line holding runs a column apart (3 em) is no evidence. An
indented first line votes against a justified paragraph's last line, so such a
paragraph ties and, like every line without evidence, keeps the previous
letter-based rule. The same decision is taken where the extractor merges a
line's fragments and where lines are assembled.

The fragments of a line are merged only between neighbours on the page: where
the reading turns round at the end of an embedded run, the runs either side
stay items of their own, which line assembly orders and spaces again. A
punctuation item at either end of a line, and one painted in display order,
goes where the algorithm puts its characters. A full stop or comma shown
between the space after a Latin word and the left end of a right-to-left word
(a left-to-right line ending inside a right-to-left phrase) is read as the end
of that phrase, set against the word it follows. A number shown between a
Latin word and a right-to-left phrase still reads with the Latin word: the
display does not tell `Rust 2025 …` from `… 2025`.

Brackets at an odd level are shown by their mirror images, and producers map
those glyphs differently. Glyphs painted in display order are read as the
mirror images and turned back, except on a page that marks its reversed
strings `/ReversedChars` (Chrome), whose glyphs decode to the brackets
written. Text stored in reading order keeps its characters, unless its
brackets stand against right-to-left letters the wrong way round (`)الأمر`,
`جداً(`) more often than the right way, as CoreText stores them; those at odd
levels are then turned back. CoreText also walks a right-to-left run with
negative character spacing or with an offset back after each glyph: such a
run's box is taken from its glyphs, not the pen's travel, an offset after a
glyph walked back is no word or column gap, and the string votes for storage in
reading order. Character spacing of an em or more, taken back by the offsets,
is read the same way. The word-gap floor of a line shown one glyph per item
leaves out gaps of an em or more, so the column gaps of a table row no longer
hide the word spaces inside its cells.

On authored pages printed by Chrome and by AppKit/Quartz (Arabic, Persian and
Hebrew, right to left, in the default direction, `dir=auto`, justified and
centred, with a table), 51 and 50 of 52 blocks keep their text in reading
order, against 39 and 13 before; on 406 corpus PDFs only the six holding
right-to-left text changed. Remaining differences are paragraph boundaries on
short pages (the page reader's own threshold), kashida inserted by
justification, which is kept as text, the number ambiguity above, and zero
width non-joiners, which the PDFs do not carry.

## CJK fonts and text that cannot be decoded

A Type0 font without a ToUnicode CMap whose `/Encoding` is a CMap other than
Identity-H/V is read as PDF 32000-1:2008, 9.10.2 prescribes: each code selects a
CID through that CMap, and the CID reads as the `Adobe-<Ordering>-UCS2` CMap of
the descendant's character collection has it. This covers the predefined CMaps
of non-embedded Japanese, Chinese and Korean fonts written by Distiller,
Ghostscript, iText, PyMuPDF and older Office exports (`90ms-RKSJ-H`,
`GBK-EUC-H`, `GB-EUC-H`, `B5pc-H`, `ETen-B5-H`, `KSCms-UHC-H`, `Uni*-UCS2-H`,
`Uni*-UTF16-H` and their `-V` forms; the 169 pdf.js binary CMaps are compiled
in) and CMaps embedded as streams, with their `usecmap` base. Strings split by
the CMap's codespace, so the one-byte ASCII of Shift-JIS, GBK, Big5 and UHC
between two-byte ideographs and the four-byte surrogate pairs of UTF-16 each
stay one code; widths are read by the CID each code selects, and word spacing
applies to the one-byte code 32. A code the CMap or the collection does not
map shows as U+FFFD, a one-byte control code as nothing. Before this, no
built-in CMap parsed (the binary reader did not follow the format past each
record's first entry), and such a page lost all of its text, its Latin lines
included, to `suspected_garbled_text`.

The parsed CMaps are kept for the life of the process: the first document
that needs one pays about a millisecond to read it (measured in a release build
of the reader on Apple silicon: 0.8 ms for `90ms-RKSJ-H` with Adobe-Japan1,
1.2 ms for `UniCNS-UTF16-H` with Adobe-CNS1), later ones nothing. Identity-H/V
fonts, with or without a ToUnicode CMap, are read as before: the collections'
UCS2 CMaps are not applied to an Identity-encoded font's CIDs, which may be its
program's glyph indices (only Korea1 keeps its table reading). A font with a
ToUnicode CMap whose encoding is a predefined CMap takes the encoding's reading
as its fallback, and as its reading when the ToUnicode CMap has fewer than ten
entries; otherwise its codes are still split by the ToUnicode CMap's width.
Vertical CMaps read like their horizontal forms; text laid out vertically is
not reconstructed, and comes out in the order of its lines across the columns.

A text run that still cannot be decoded is left out of its page rather than
costing the page all of its text. A run is left out when it shows a strong sign
of failed decoding (private-use or C1-control code points, `$` between letters,
symbol soup), when U+FFFD makes up at least half of it, or when it shows two
U+FFFD in a row and no letter of any script. The page then needs OCR only if
the runs kept are themselves unreliable (the page-level replacement-character
and cipher checks), if what was left out is more than half of the page's
characters, or if fewer than 20 letters and digits remain. A page listing a
font that names its glyphs by index only (`gidNNNN` without a CMap) no longer
needs OCR for that alone when 20 readable characters remain. U+FFFD in a run
that is kept stays in place; the Markdown no longer sends a page to OCR for a
single replacement character.

Such a page carries a warning, deferred like the missing-text warnings so that
OCR removes it:

```text
PDF page 1: 1 text run (20 characters) could not be decoded and was omitted; run again with --ocr to recover it.
```

or that characters `show as U+FFFD`, or that a font on the page names its
glyphs by index only. With `--ocr` such a page is recognized from its image as
a whole, as a page that needs OCR is. The reader reports these pages in
`PagesExtractionResult::omitted_text_by_page` (`PageOmittedText`: page, runs,
characters, replacement characters, glyphs without identity), and
`LoadedPdf::text_with_positions_and_rotations` leaves out the same runs, so the
layout, running-header and comment readers see the page as its Markdown does.
Probes of the CMaps listed above and of PyMuPDF's built-in CJK fonts convert to
the text PyMuPDF reads, and a page with a run of undecodable Identity-H text
keeps its Latin lines; the other 228 PDFs of the local corpora convert
byte-identically.

## Images and remaining work

Executed embedded raster images retain their existing extraction path and are
appended to the corresponding page. Samples are encoded without colour
conversion: DeviceGray/DeviceRGB, ICC-based profiles with one or three components
and CalGray/CalRGB use their component count (the profile is not embedded), and
Indexed images with a gray or RGB base become palette PNGs at 1, 2, 4 or 8 bits,
clamping indices above the maximum. CMYK, Lab, nested or malformed palettes,
decode arrays, masks and other depths remain explicit warnings. JPEG streams are
passed through unchanged. Exact placement and reference JPEG encoding
remain different. Detected bar-chart crops use the full platform renderer;
geometry detection itself does not reproduce PDF painting. Other vector figures
are not automatically preserved. Use page screenshots when a full visual copy
is needed.

Table geometry is deliberately disabled when resource-dependent colours,
transparency, Form invocations, shading or inline images make its verdict
uncertain. A path containing curves contributes no edges, image XObject
invocations are passed over, and rules inside a curved or polygonal clip are
withheld until its graphics state is restored. Horizontal rules of the same width
form one grid only while vertical borders join them, so a bordered block below a
table of the same width stays separate. A named graphics state is accepted only when it changes nothing
the verdict depends on: full opacity, a Normal or Compatible blend mode, no soft
mask, and otherwise only line and rendering parameters. Chrome's print output sets
such a state around its per-cell border rectangles, so its continued tables are
reconstructed with every row, as are pages with links, images or curved marks.
Text extraction can still proceed. Merged-cell tables, borderless
tables outside the evidence described above, complex columns, mathematical
layout and structured vector charts remain open. Explicit
[page media processing](pdf-ocr.md) provides rendering and local OCR
separately from this text reader, with its own accuracy limits.
Extractable chart labels alone do not establish visual recovery; check the
referenced chart image against the source.

The page reader loads the file once for the page Markdown, the positioned pass and,
when it read the bytes unchanged, this module's inspection; each page's content is
expanded and parsed once, as described above. A font that is an indirect object is
decoded once per document, its encoding and width table taken by every page and
Form that lists it (up to 2^18 kept codes in all; a font written in place is decoded
where it is listed), and the reader's text pass decodes each Form XObject once per
document (64 KiB of Form content kept at most) however often it is invoked. The
ToUnicode CMaps of fonts that only a Form XObject lists are read with the
page's fonts also when the Form's `/Resources` is an indirect object (MuPDF
writes them so) and in the fast mode of the region readers; such a font's
two-byte codes were otherwise read as UTF-16 (`Papers` shown as `1BQFST`). The
reader's OCR signals still interpret the expanded bytes with a byte-level scan of
their own: a scan over parsed operations would measure literal strings by their
decoded bytes, not by the bytes it measures now. This module's inspection and the
OCR signals still expand Form XObjects themselves. A file under 256 KiB is parsed
on one thread: lopdf parses on a pool of one thread per core, whose start and idle
spinning cost more processor time than a small file's parse, and larger files keep
the pool, which shortens their load.
