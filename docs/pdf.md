# Native PDF text and layout

The PDF reader uses `pdf-inspector` for decoding and page-level reliability
decisions, with `lopdf` for bounded content inspection and embedded images.
There is no Python runtime, external converter or implicit OCR fallback.

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
OCR, containing suspicious hidden text, or having incomplete content inspection
does not enter refinement. Text size for the hidden-text check is the effective
size after the text matrix, transformation and Form matrices: Quartz writes
`1 Tf` and scales with the text matrix, which is ordinary 12pt text. The existing narrowly guarded font-decoded recovery
for a false scan verdict remains in place. A missing or unreadable page keeps
its page marker and an explicit warning.

The pinned dependency has a small, tracked policy patch under
`vendor/pdf-inspector`. Its original license, bundled character-map license and
per-file upstream hashes are retained. This patch changes extraction policy;
it does not weaken the layout character-agreement or page reliability gates.

Chrome (Skia) prints web fonts it cannot embed as Type3 fonts whose mirrored
`FontMatrix` is paired with glyphs drawn y-down. Their glyph side is now read
from the `FontBBox` as well as the matrix, so such runs stand on their baseline
instead of one font size below it, where they interleaved with the embedded
fonts of the same line. On 108 Chrome-printed reference fixtures, readable words
missing against the reference fell from 309 to 95 of 25,684 with this and the
code, list and heading changes below ([record](validation/pdf-quality-round41.md)).

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
document. The
[historical two/40-page reproduction](validation/pdf-page-prefix-round15.json)
retains the original inputs and old outputs with 72/1,440 missing paragraphs;
new authored regressions require every paragraph in source order.

Text rendering mode persists across `BT`/`ET` text objects and nested `q`/`Q`
graphics states. Normal extraction excludes `Tr 3` invisible text and `Tr 7`
clipping-only text in pages and Form XObjects. Hidden text still advances the
text cursor, so following visible glyphs keep their positions. An `ActualText`
replacement whose glyphs are all nonpainting cannot restore hidden text. The
dependency's explicit OCR-layer path may still request mode 3; clipping-only
mode 7 is never treated as an OCR layer. The core does not use that recovery
path and retains the existing guard against plain-text recovery of suspicious
pages.

These rules do not establish complete rendered visibility: transparency,
blending, soft masks, occlusion, arbitrary clipping and mixed-visibility
`ActualText` spans still need broader interpretation. Pages with visibility
signals retain an explicit warning and do not enter layout refinement.

## Typed pages and final assembly

The reader separates extraction from document assembly. `extract_pages` returns
one `PdfPage` per source page in source order, with a one-based page number, the
native Markdown body, the reader's OCR verdict and reason, and the names of
successfully extracted assets. The body has neither generated page markers nor
appended image references. `visibility_suspect` comes directly from graphics-state
inspection; orchestration does not recover that decision by parsing warnings.
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
overlapping table regions prevents speculative reconstruction. The first row is
represented as a Markdown table header; this representation does not establish
that an untagged PDF declared it a semantic header.

A tagged PDF's tables come from its structure tree on each page, as in the
page reader's whole-document conversion (`8b9747d`): a table drawn without rules
whose cells wrap and are vertically centred is otherwise read from its text's
alignment alone, which splits it into broken rows. Untagged pages still rely on
that alignment and on ruled grids. Text extraction keeps the runs of two
structure-tree cells apart (`03deb93`): cells a few pixels apart, as in the
browser's default table style, would otherwise merge into one item that keeps
only the first cell's marked content. A fully tagged table among long
paragraphs is used when it holds 80% of the text inside its own bounds.

Before replacing page Markdown, decoded alphanumeric character counts must agree
with the existing reader. This is a conservative agreement check, not proof that
two decoders are independently correct. Link annotations carry targets, not page
text, and neither reader renders them, so they do not block refinement. The body
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
also collects compact painted marks of 1–8pt; a mark of at most half the line's
size, ending no more than two em before its first run and centred on the lower
part of its text, makes the line a list item. Bullet characters remain list
markers. A number (`3.`/`3)`, up to three digits) opens an item where a new line
could start: first in the flow, inside a list, after a gap or heading, or offset
from the line above. A number on a line of its own marks the next line. Items
nest by their marker's offset (a browser indents 40px), consecutive items form
one tight list, and a heading keeps its level. Super/subscript runs keep their
anchor's line and render as `<sup>`/`<sub>`; digits the page reader fuses into
their word stay Unicode superscripts. Headings need more than 1.15 times the body
size, so a browser `<h3>` (1.17 em) is one.

The refinement limits are 250,000 positioned items and 16 MiB decoded text across
the selected document, 20,000 items per page, 200,000 operations and 8,192 rule
edges per geometry pass, 64 graphics-state levels, 32 tables, 32 columns and
4,096 cells per table. Existing 64 MiB expanded page/Form inspection and image
budgets remain. Positioned-item limits are checked after the dependency returns;
they are not a claim of a hard allocation limit inside that dependency.

Content inspection and table geometry share one bounded content decode per page.
Inspection first applies the existing visibility, Form and warning checks; only
an eligible page's same parsed operations are then read for table borders. Raw
expanded page streams and parsed operations are released at the end of that
page's preparation. Only its frame and bounded grid coordinates survive until
the positioned-text pass completes. Form inspection retains the existing shared
64 MiB page/Form byte budget, 256 content inspections and 32 nested Form levels;
Form operations are not used as speculative table borders. Pages with incomplete
inspection retain the original warning and fallback behavior.

## Images and remaining work

Executed embedded raster images retain their existing extraction path and are
appended to the corresponding page. Samples are encoded without colour
conversion: DeviceGray/DeviceRGB, ICC-based profiles with one or three components
and CalGray/CalRGB use their component count (the profile is not embedded), and
Indexed images with a gray or RGB base become palette PNGs at 1, 2, 4 or 8 bits,
clamping indices above the maximum. CMYK, Lab, nested or malformed palettes,
decode arrays, masks and other depths remain explicit warnings. JPEG streams are
passed through unchanged. Exact placement and reference JPEG encoding
remain different. There is no screenshot or vector-figure renderer in this
reader. `resvg` renders SVG but does not interpret PDF graphics state: a faithful
PDF figure also needs font programs, Form matrices, clipping, shadings, blend
modes and masks. Drawing table borders from a bounded subset of PDF operators
does not establish those rendering semantics.

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
Text extraction can still proceed. Borderless or merged-cell
tables, complex columns, mathematical layout and structured vector charts remain
open. Explicit [page media processing](pdf-ocr.md) provides rendering and local
OCR separately from this text reader, with its own accuracy limits.
The historical five-page sample's chart must not be presented as recovered
merely because its textual labels are extractable.

The additional positioned pass still reparses the PDF. Local page-content reuse
does not remove the dependency's separate document and font decoding. Sharing one
decoded document with that dependency needs a future API change; no speedup or
corpus-parity claim follows from local reuse alone. Focused tests author
their own PDF streams for heading consistency, paragraphs, continuous emphasis,
complete tables, hidden text, rotated/invalid geometry, compressed multi-stream
pages, inspection budget boundaries and unreadable or deeply nested Forms. Validation results
are recorded by the coordinator after the source is frozen.
