# PDF page rendering

Page images for screenshots, scanned-page OCR and Office page capture
(LibreOffice → PDF → pixels) are drawn in process. No Python, browser or
command-line renderer is involved, and the renderer writes no files.

| Platform | Backend | Notes |
|---|---|---|
| macOS | CoreGraphics | the system PDF renderer |
| Windows, Linux | [hayro](https://github.com/LaurenzV/hayro) 0.7.1, pinned exactly | pure Rust, compiled into every non-macOS build |
| macOS with `--features markitai-core/portable-media` | both | CoreGraphics by default; `MARKITAI_PDF_RENDERER=portable` selects hayro |

`MARKITAI_PDF_RENDERER` exists for development and comparison. It accepts
`coregraphics` or `portable`; an unknown name, or a backend the build does not
contain, is an error rather than a silent use of the default. The default
macOS build does not contain hayro, so its size is unchanged.

## Shared contract

`PdfRasterSession::dimensions(page, dpi)` validates a one-based page number and
the complete drawing frame before allocating pixels. `render(page, dpi)` uses
the same checks and returns upright RGB8 pixels composited onto white. The frame
is CropBox intersected with MediaBox, with the page's quarter-turn rotation;
dimensions are the frame at the requested DPI, rounded up. The media
orchestration layer chooses 150 DPI and manages document-wide page, pixel and
screenshot budgets.

Each raster is limited to 32 million pixels. Coordinates and DPI must be finite,
dimensions must be positive, and all row/buffer products are checked. Input is
limited to 500 MiB and must carry a PDF header in its first KiB; both checks run
before either backend parses a byte. Invalid or password-protected documents
and invalid frames fail explicitly; no password is tried beyond the empty user
password that owner-password-only files carry. `dimensions` allows the caller to
reject a cumulative budget excess before creating that page's bitmap. A session
is confined to its caller's thread.

Rasterization is separate from native text reliability, OCR routing and
screenshot publication, which must report their own page outcomes rather than
infer success from a nonempty pixel buffer. Neither backend has a time limit:
the application limits above are not a renderer sandbox or a hard timeout.

## CoreGraphics (macOS)

The session holds one `CGPDFDocument` and an owned immutable copy of its input.
Page drawing includes native text, vector paths, image masks and Form XObjects;
it does not approximate a page by collecting embedded images. CoreGraphics
supplies the point-space crop/rotation transform; the adapter then applies DPI
scaling explicitly, since Quartz's fit transform does not upscale to a larger
bitmap, and centres the page in the rounded-up bitmap. At the maximum size the
RGBX drawing buffer occupies 128 MB and the RGB result 96 MB; CoreGraphics may
allocate more internally. Annotations are not drawn. The backend follows
Apple's documented
[PDF page drawing and transform APIs](https://developer.apple.com/library/archive/documentation/GraphicsImaging/Conceptual/drawingwithquartz2d/dq_pdf/dq_pdf.html).

## hayro (Windows, Linux)

The session parses the document once and draws one page at a time into an
opaque white pixmap (premultiplied RGBA, composited over white, then RGB). The
content is scaled uniformly from the top-left corner, so rounding up leaves at
most one white pixel column or row at the right or bottom edge where
CoreGraphics centres the page. Annotation appearance streams (form field
values, stamps) are drawn, as the reference's PyMuPDF does; CoreGraphics does
not draw them.

- **Encryption.** A file containing `/Encrypt` is first opened by the vendored
  lopdf, the library text extraction uses, so text and pixels come from one
  decryption: an owner-password-only document is decrypted with the empty user
  password and the plaintext handed to hayro; one that needs a password fails
  as locked. If lopdf cannot read the file, hayro gets the original bytes and
  applies its own empty-password decryption (RC4, AES-128, AES-256).
- **Page boxes.** hayro reads boxes as `f32`; the frame is computed from the
  shortest decimal that reads back as the same value (the number the file
  wrote), so `593.76` points give 1,237 pixels at 150 DPI on both backends
  instead of 1,238.
- **Limits.** hayro sizes pixmaps with 16-bit sides, so a page side over 65,535
  pixels fails in `dimensions` (CoreGraphics accepts it within the pixel
  limit). The pixmap is 4 bytes per pixel (128 MB at the limit) plus the 96 MB
  RGB result; hayro and its rasterizer allocate tile and layer buffers besides.
- **Failures.** A panic inside hayro while reading the document or drawing a
  page becomes this conversion's error; the session's document stays usable.
  The default panic message still reaches standard error.
- **Caching.** Font and image decoding is cached per page draw, not across
  pages of a session.

### Fonts and character maps

Embedded fonts (TrueType, CFF, Type 1, Type 3, OpenType) are drawn from the
file. For text that relies on fonts the PDF does not embed:

- The 14 standard fonts, and every other non-embedded simple font (which hayro
  maps onto them by name and flags), use hayro's embedded substitutes: PDFium's
  Foxit faces (about 264 KB). They are the same on every host, including
  containers without fonts, and stretched to the PDF's widths, but cover Latin
  only: Cyrillic or Greek glyph names in a non-embedded simple font draw blank.
  Host faces were measured and rejected for these: reading the host font list
  costs about 75–115 ms per process (833 faces on macOS), common PDFs would pay
  it for Helvetica alone, and pixels would vary by host.
- A non-embedded composite (CID) font, the usual case in Chinese, Japanese and
  Korean PDFs, is looked up among the host's fonts: by the PDF's font name, then
  by the font's character collection (Adobe-GB1, CNS1, Japan1, Korea1; serif or
  sans-serif list first from the descriptor flag or name), then other
  collections, which share most Han characters. hayro maps each character
  through the font's ToUnicode or the collection's UCS-2 map into the face
  found. Families listed include the Windows defaults (SimSun, Microsoft YaHei,
  MingLiU, MS Mincho, Yu Gothic, Batang, Malgun Gothic), macOS and the common
  Linux packages (Noto CJK, Source Han, WenQuanYi, IPA, Nanum). The host list is
  read once per process, only when such a font first appears; face data is read
  once and kept. Without a suitable host face, the Foxit face draws the Latin it
  can and the other glyphs stay blank.
- The 61 predefined CMaps a PDF may name without embedding (`UniGB-UCS2-H`,
  `90ms-RKSJ-H`, …) are embedded (Adobe's cmap-resources, about 251 KB
  compressed).

The Foxit faces (© PDFium Authors, BSD-3-Clause, from Foxit Software), the
CMaps (© Adobe, BSD-3-Clause) and a CC0 CMYK ICC profile come inside the
`hayro-interpret` and `hayro-cmap` crates' `assets` directories, not their
top-level licence files; a binary distribution must carry those notices.

### Not supported

- Knockout transparency groups are drawn as ordinary groups.
- Non-embedded simple fonts outside the Latin set, and non-embedded CID fonts
  without a host face, as above.
- A PDF hayro cannot parse fails as unreadable; hayro reconstructs a damaged
  cross-reference table and, failing that, finds pages by scanning objects, so
  its page order can then differ from the text extractor's.

## Renderer differences and validation

CoreGraphics and hayro can differ in font substitution, glyph smoothing and
annotation appearances. Missing non-Latin fonts can omit text in hayro; install
a suitable host face or use an embedded-font source PDF. These pixel differences
can also change OCR, especially code listings and line numbers. Renderer parity
does not imply a handwriting, table or multi-column recognition guarantee.

The native raster tests exercise each compiled backend with authored fixtures;
font-dependent cases require a suitable installed font. Comparison tools remain
in `docs/validation/drivers/portable-raster-r1/`: `compare.py` records pixels,
failures and timings; `make-scanned.py` prepares scanned PDFs; `ocr-e2e.py` scores
full OCR conversions. Store measurements under ignored `.local/`, recording the
backend, source, fonts and platform used.
