# PDF page OCR and screenshots

The PDF media layer combines typed native page results with a native renderer
and local OCR. It operates in memory; the output layer owns safe file publication.
The Rust core does not start Python, Node, PyMuPDF or a rendering command. The
initial renderer and OCR backends use macOS system frameworks; other platforms
return an explicit error when the required backend is unavailable.

The [round-seventeen validation](validation/pdf-native-round17.md) records
coordinated tests and real CLI/binding acceptance. Routing checks pass; exact
transcription has the unresolved error below. Existing
[image OCR evidence](validation/native-backends-round16.md) does not establish
this PDF integration's general fidelity.

## Page and model routing

Local OCR is selected when OCR is enabled and LLM is disabled, or when
`MARKITAI_NO_VLM_OCR` opts out of VLM OCR. With `ocr.per_page_routing=true`, only
unreliable, empty or visibility-suspect native pages require recognition.
Disabling routing recognizes every page. Other pages retain their original
Markdown, including headings, links and tables. This uses typed reliability
evidence rather than the reference reader's raw character-count threshold.

A searchable scan's page read from its embedded OCR text layer (see
[Searchable scans](pdf.md#searchable-scans)) is a reliable native page: routing
keeps the layer rather than recognizing the page again, since the layer is the
recognizer the producer chose and may read scripts local OCR does not. Its scan
is not recognized again as a picture of the page either. To have local OCR
replace such layers, disable routing (`ocr.per_page_routing: false`): every page
is then recognized, the layer warning and `ocr_layer_pages` drop those pages,
and the recognized text replaces the layer's. A page whose layer was refused
(off the print, past the inspection's reach, or with other hidden text) is a
scan and is recognized as before.

Extracted pictures of at least 40,000 pixels on retained native pages are also
recognized; their text follows the corresponding image reference. Shared images
are recognized once. Unreadable pictures or a missing backend keep the native
page and reference with a warning; invalid configuration or an unsupported
language on an available backend fails explicitly. Page OCR failure fails the document. Successful blank OCR keeps the
page marker and emits a notice without fabricated body text.

OCR plus LLM normally renders every page, including reliable native pages,
independently of `screenshot.enabled`. The local routing setting does not select
which pages the VLM receives. Explicit screenshots remain a separate request:
with VLM OCR disabled, local OCR plus screenshot enhancement still sends page
images, and the warning explains how to keep those images local.

For PDF files, pure mode without screenshot-only selects text enhancement even
if OCR+LLM produced page images. When page images exist, screenshot-only takes
precedence over pure and selects an image-only prompt. Without LLM, PDF
screenshot-only retains ordinary base Markdown, unlike the URL path. This module
reads `screenshot.enabled` directly; configuration-only `screenshot_only=true`
does not imply capture. The explicit CLI flag applies that implication earlier.

The public in-memory API may use rendered pages for enhancement without returning
persistent screenshot paths. The media object supplies `has_reliable_text` so
fallback decisions use recovered content rather than page markers or comments.
The caller owns the LLM request and public result shaping.

## Hidden text and media

The [hidden text policy](pdf.md#hidden-text-policy) is applied before native page
routing and before that body is supplied to an LLM. It applies equally when
OCR or screenshots are enabled. Successful bounded removal can leave a native
page reliable; incomplete or uncertain filtering retains its visibility
verdict, allowing the existing OCR routing to replace its body. Local OCR and
VLM OCR read the original rendered page pixels. Embedded image assets and
screenshots are not generated from the temporary filtered text document.

An accepted embedded OCR layer is retained as described above. To recognize it
again from pixels, use `ocr.per_page_routing=false`. `off` only suppresses
hidden-text security notices: the native reader still omits nonpainting text
unless its established searchable-scan checks accept it as an OCR layer.

## Pixels, encoding and publication

Pages render at 150 DPI as upright RGB composited over white. Each selected page
is rendered once: screenshot encoding borrows the pixels, then local OCR consumes
the original full-resolution image. Screenshot compression cannot degrade that
OCR input. Reliable native pages without screenshots do not need rendering.

PDF screenshots use `image.format`, `image.quality`, `image.max_width` and
`image.max_height`; browser viewport and tiling settings do not apply. JPEG, PNG
and WebP extensions match actual encoded bytes. WebP is lossless, with a warning
that quality does not affect this encoder. Oversized output falls back to lower
JPEG qualities and then a maximum 1024×1024 image, changing the extension to
`.jpg`. Remaining oversize output is an error rather than a missing page.

Names start as `<output-stem>.page0001.<extension>`, including CLI-reserved
document rename suffixes. Direct core calls without a reserved stem may reuse
identical earlier captures while renaming their Markdown output. The output
layer chooses a safe final name before assembly
adds a reference after that page's content:

```markdown
<!-- Page number: 1 -->

Page body.

<!-- ![Page 1](.markitai/screenshots/report.pdf.page0001.jpg) -->
```

The typed page assembler escapes names as URI path components. Screenshots are
distinct from embedded assets and image counts. `PreparedPdf::finish` returns the
same owned screenshot buffers with the assembled document instead of duplicating
their full payloads.

Successful PDF vision enhancement appends an ordered page-image reference list
even when the model omits it. Pure text enhancement keeps its original body
policy. Before publishing Markdown, the output layer verifies that previously
published screenshots still match their bytes; a concurrent modification fails
instead of silently renaming an already-referenced image.

## Recognition language

Page, embedded-picture, TIFF-page and Office-page recognition share the image
recognizer's language policy. Under the default `ocr.lang` (`en`) each page or
picture is read as English first and, only when that reading failed, as Chinese,
Korean and Japanese ([details](ocr.md#the-default-language)), so a scanned
Chinese, Japanese or Korean page comes out as text without configuration, and a
page that reads as sound English costs nothing more. Each page is judged on its
own: a document that mixes languages reads every page in its own. A page
without text costs two more readings, which are cheap when no text is found.
A page that no reading can read adds a warning, `Local OCR could not read PDF
page N: ...` (`Office page N`, `TIFF page N` or `this image` for the other
inputs), and keeps only the lines its English reading is sure of; an embedded
picture never warns, as a picture without text is ordinary. A written language, `en-US` included, reads
that language alone, exactly as before.

Measured with the binaries of [local OCR](ocr.md#the-default-language) (macOS
27.0.1, whole `--ocr --no-llm` conversions, before and after alternating, median
of three; scanned pages are the R45, held-out Chinese, Japanese and Korean
images at 150 DPI, one per page): six English pages 483 ms before and 486 ms
after, with identical Markdown; six Chinese pages 388 and 914 ms, from symbols
to the text of every page; twelve pages, three each of English,
Chinese, Japanese and Korean, 619 and 1,852 ms, with the nine pages of the other
languages read (four had no text before); six blank 1600×1200 pages 457 and
1,173 ms, about 120 ms for each page without text; and the six-page scan of the
formats comparison 430 and 612 ms, its sixth page, Chinese, read instead of
"completed with no recognized text".

## Bounds and remaining gaps

- At most 1,000 pages, checked before native text extraction, and 32 million
  pixels per rendered page.
- At most two billion cumulative media pixels. All selected page dimensions are
  checked before rendering; native-picture recognition also consumes this budget.
- At most 5 MiB per screenshot and 100 MiB across retained screenshots. Encoder
  writes stop at the per-image bound to permit a bounded fallback.
- A positive `llm.max_vision_pages_per_document` is checked before rendering when
  the selected LLM branch will send page images. Pure text is not blocked by an
  irrelevant image limit.

These limits fail the document instead of truncating its page sequence. They are
not a measured peak-memory ceiling: readers, system renderers and image encoders
also allocate working storage.

Internal document metadata distinguishes native pages (pages read from an OCR
text layer count among them; `ocr_layer_pages` lists those) and attempted,
nonempty and blank local OCR pages/pictures. Successful blank recognition counts as OCR
use; failed picture attempts do not. The selected `ocr_path` is recorded;
pixel buffers, page plans and internal image arrays are not exposed as metadata.
These internal diagnostics do not add fields to the existing local-file public
frontmatter contract.

Embedded images remain appended to their source page. Exact placement and
editable vector reconstruction are not implemented. Screenshots preserve visual
composition but do not turn vector drawings into structured Markdown. Recognition
shares the language, reading-order and cancellation limitations in [local OCR](ocr.md).
No handwriting, complex-table or multi-column accuracy guarantee is claimed.
The authored scan's six-word transcript ends in `2026`, which the 150-DPI page
recognition returns as `2ø26`: the fixture's bitmap font draws its zero with a
stroke across the whole glyph, like `ø`. A run of letters drawn like a zero
between two digits of a number is now read as zeros ([local OCR](ocr.md#turned-pages-code-numbers-and-table-cells)),
so the page reads exactly and its exact-match test runs in the ordinary gate.
Menlo's slashed zeros in the rendered number set were read correctly before
([R45](validation/ocr-quality-round45.md)).
Static/automatic URL fetches and initial browser PDF responses hand requested
PDF media directly to this pipeline,
including redirects and extensionless downloads. The original URL remains the
source and naming input. URL pure mode sends text even with screenshot-only;
PDF capture-only output retains Markdown and page references. This deliberately
extends the reference URL converter, which did not pass local media settings
through. Browser PDFs reuse the authenticated CDP response and preserve their
`playwright` strategy; remote extraction services retain their own path. See
[downloaded PDFs](pdf.md#downloaded-pdfs) for the content/identity contract and
[round-eighteen validation](validation/url-pdf-round18.md) for execution evidence.
