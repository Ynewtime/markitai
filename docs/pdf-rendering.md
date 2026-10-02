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

## Validation

`pdf_raster/tests.rs` runs every assertion against each backend the build
contains: authored native, scanned and blank pages; all four rotations with
independent corner colors; nonzero and intersected crop origins; nested Forms,
vector fills and zero alpha; a real image soft mask; owned input lifetime;
locked, owner-password-only, malformed and oversized documents; fractional page
boxes; a non-embedded standard font. Portable-only tests cover a non-embedded
Adobe-GB1 font drawn with a host face (skipped where none is installed), the
16-bit side limit and an `/Encrypt` marker inside plain content. The fixture
manifest records input hashes and independent expectations. On 2026-10-02 the
`pdf_raster` and `office_render` tests (30) passed natively on Ubuntu 24.04
amd64 (OrbStack, Rust 1.98.1), where hayro is the only backend; the CJK host
face test skipped there for lack of a Chinese font. The workspace also passes
`clippy -D warnings` for `x86_64-pc-windows-msvc` (type check, not linked).

`docs/validation/drivers/portable-raster-r1/` holds the comparison drivers:
`compare.py` renders a corpus with both backends on macOS and records per-page
similarity, sizes, failures by cause and render times; `make-scanned.py` binds
the R45 OCR images into scanned PDFs with ground truth; `ocr-e2e.py` converts
scanned PDFs with `--ocr` through both backends and Vision and scores the
character error rate.

### Comparison r1 (2026-10-02, macOS 27.0 arm64, Rust 1.99.0, release profile)

Inputs: the 108 Quartz-laid-out and 108 Chrome-printed reference fixtures
(`.local/pdf-corpus-r2`, `-r1`), the six `pdf_raster` fixtures, 14 scanned PDFs
bound from the R45 OCR images (100 pages with ground truth), four existing
scanned PDFs and every other distinct local PDF (275): 516 documents, 64.5 MB,
SHA-256 list in the run's `inputs.json`. 150 DPI, at most 40 pages each, 1,663
pages compared. Evidence: `.local/w2a/compare-r2` (all inputs), `-r3`/`-r4`/`-r5`
(timing on the first five groups, 471 pages).

| Group | Pages | SSIM (median) | Ink IoU median / p10 | hayro ink / CoreGraphics ink |
|---|---|---|---|---|
| Chrome-printed | 184 | 0.9995 | 0.986 / 0.961 | 0.995 |
| Quartz-laid-out | 162 | 0.989 | 0.881 / 0.862 | 0.850 |
| scanned, with ground truth | 100 | 0.9986 | 0.992 / 0.890 | 1.008 |
| other local PDFs | 1,192 | 0.990 | 0.882 / 0.818 | 0.890 |

Page counts and pixel sizes agree on every page; both backends fail the same 13
documents (3 without a PDF header, 7 unreadable, 2 needing a password, 1 over
500 MiB; on one truncated file hayro reports no pages where CoreGraphics
reports an unreadable document). No page failed to render and hayro reported no
skipped font or image. Of the 681 pages under 0.85 ink SSIM or ink IoU, 648
differ in glyph rasterization, not content: CoreGraphics smooths (emboldens)
glyphs, hayro draws their exact coverage, so Quartz-made text has about 15% less
ink (476 pages use a non-embedded standard font, where the Foxit faces also
differ from macOS Helvetica/Times; 58 a non-embedded CID font drawn with another
host face; 48 Quartz text; 23 other embedded-font text; 43 scans or vector
pages at threshold edges). Content differs on 33 pages: 30 in three synthetic
glyph-name test files (non-Latin glyph names in non-embedded standard fonts,
blank in hayro) and 3 form/annotation probes whose appearance streams only
hayro draws.

Scanned-PDF OCR end to end (`--ocr`, Vision, the same `portable-media` binary
with each renderer; 18 PDFs, 112 pages): 15 outputs are identical, the largest
difference is 0.69% CER (Chinese, 300 DPI), 0.024% pooled. Against the ground
truth: CoreGraphics 1.289%, hayro 1.296%. The existing scans are
`quality-r1/cli/batch-ocr/scan1.pdf`, `lazy-fw-r1/inputs/scanned.pdf`,
`release-qa-r1/work/img/scan.pdf` and
`quality-r1/p-impl/pdf-inputs/scanpdf--scan6.pdf`. All local scans are JPEG;
JBIG2, CCITT and JPEG 2000 pages were not available to compare.

Rendered text read by OCR (`--all-pages`: `ocr.per_page_routing: false`, the
path Office page capture takes; the first 15 Quartz and 15 Chrome fixtures plus
a five-page LibreOffice 4.2 Writer export, `release-qa-r1/work/docs/scanned.pdf`,
which despite its name holds text; 40 pages): five outputs identical,
median 0.77% CER and 1.01% pooled over the 29 documents where OCR kept every
page. In the other two, the OCR layer's unread-script judgement dropped whole
pages of Lorem ipsum (pages 3–4 after CoreGraphics, 4–5 after hayro), so their
CER (60%, 45%) measures that all-or-nothing page decision, not the pixels.
Code listings with line numbers vary most (up to 20%) under both renderers.

Render time per page (471 pages, one thread; parallel workers were building on
the same host, so CoreGraphics, measured alternately, is the control):

| hayro build | median | p99 | total | CLI size (`portable-media`) |
|---|---|---|---|---|
| all `"z"` | 8.2 ms | 64.3 ms | 6.17 s | 24,552,608 B |
| `vello_cpu`, `vello_common`, `fearless_simd` at 3 (chosen) | 4.6 ms | 54.8 ms | 4.06 s | 24,684,160 B |
| every hayro crate at 3 | 3.5 ms | 29.9 ms | 2.65 s | 25,784,496 B |
| CoreGraphics | 2.9 ms | 35.9 ms | 2.62–2.75 s | — |

Pixels are identical across the three builds. The default macOS CLI is
22,095,792 B both before and after this change (HEAD `4b2e2ef`); hayro adds
2,588,368 B to a macOS `portable-media` CLI, a first estimate for the
Windows/Linux increase.
