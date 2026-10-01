# Local image OCR

On macOS 11 or later, local OCR uses the operating system's Vision framework
through Rust bindings. It runs in the current process without Python, Node,
Tesseract, a browser, downloaded OCR weights, provider credentials or a remote
recognition service. Other platforms currently return an explicit unsupported
error for this local path. An x86-64 build running under Rosetta 2 on Apple
silicon fails recognition with an error that names Rosetta (Vision reports
failure there without an error of its own); use the arm64 build ([details](validation/macos-x86_64-rosetta.md)).
This page covers image inputs, including every page
of a multi-page TIFF and HEIF/AVIF images; scanned PDF pages use the same
recognizer as described in [PDF page OCR](pdf-ocr.md), and Office page images in
[Office rendering](office-rendering.md).

```sh
markitai scan.png --ocr                                        # English (default)
markitai scan.png --ocr --config-json '{"ocr":{"lang":"zh"}}'  # Simplified Chinese
markitai scan.png --ocr -o out/                                 # keep the image asset
```

## Selection and output

`ocr.enabled=true` (`--ocr` in the CLI) selects local OCR when LLM enhancement is
disabled. The image reference remains in the base document and recognized text
is appended below it. Metadata records `ocr_used: true` and `ocr_path: vision`.
A successfully processed image with no recognized text retains its image
reference and emits a warning. An invalid image, unsupported language or engine
failure fails the conversion instead of inserting error text into Markdown.

With LLM enabled, the existing image-to-model route remains the default.
Setting `MARKITAI_NO_VLM_OCR` to a nonempty value other than `0`, `false` or `no`
(case insensitive, ignoring surrounding whitespace), together with OCR enabled,
chooses local OCR followed by text enhancement. Image bytes
are not included in that model request. Enabling OCR does not enable LLM or
require a configured model. Disabling both still produces the existing image-only
error or CLI skip behavior.

OCR consumes the original raster bytes, independently of preview compression,
image width settings and embedded-image filters. Static SVG uses the existing
bounded in-process SVG rasterizer. JPEG, PNG, GIF, BMP, TIFF and WebP use the
enabled Rust decoders; animation uses its first image. HEIF and AVIF are decoded
by macOS ImageIO ([images](images.md#heif-and-avif)), and the primary image is
recognized. A multi-page TIFF is recognized page by page: each page keeps its
`<!-- Page number: N -->` marker and preview, followed by its text, and a page
without text adds a warning instead of ending the conversion.

## Language selection

`ocr.lang` defaults to `en`. Each call asks the configured accurate Vision
recognizer which languages its current OS and request revision support, then
requires a match. It never substitutes English for an unsupported language.
System revisions can change supported languages and recognition results.

| Configuration spelling | Requested Vision language |
| --- | --- |
| `en` | `en-US` |
| `zh`, `zh_cn`, `zh-cn`, `cn`, `ch` | `zh-Hans` |
| `zh_tw`, `zh-tw`, `cht`, `chinese_cht` | `zh-Hant` |
| `ja`, `jp`, `japan` | `ja-JP` |
| `ko`, `korean` | `ko-KR` |
| `ar`, `arabic` | `ar-SA`, if installed Vision supports it |
| `fr`, `french`; `de`, `german` | `fr-FR`; `de-DE` |
| `es`; `it`; `pt` | `es-ES`; `it-IT`; `pt-BR` |

Case, surrounding whitespace and underscore/hyphen aliases are normalized.
Other short ISO codes may match the system's regional spelling. An explicit
region or script must match exactly, ignoring case. The reference's RapidOCR
model-family names such as `latin` and `cyrillic` are not Vision languages and
are rejected. Structural configuration acceptance does not imply runtime
language availability. `ocr.per_page_routing` has no effect on this image path.

## Bounds and reading order

Encoded input is limited to 64 MiB. Decoded images must fit 32 million pixels,
with a 256 MiB decoder allocation limit. EXIF orientation is applied before OCR;
alpha is composited over white. A bounded PNG buffer of at most 128 MiB transfers
the normalized pixels to Vision in memory. Source paths, user configuration
files and provider keys are not passed to the engine.

An internal renderer entry point also accepts upright RGB pixels already
composited on white. It rechecks nonzero dimensions, the same 32-million-pixel
limit and an exact three-byte-per-pixel buffer before bounded PNG encoding.
It does not rotate, resize or recolor those pixels. Language selection, Vision
recognition and reading-order assembly are shared with encoded-image OCR;
the encoded-image path still applies its original format, orientation and alpha
rules. This entry point alone does not establish PDF routing or rendering support.

The request uses accurate recognition with language correction. At most 10,000
observations and 8 MiB of recognized text are accepted. Text and line rectangles
are checked for finite, valid confidence and geometry. Internal rectangles use
top-left pixel coordinates after orientation correction, and mean confidence
uses nonempty recognized lines. These diagnostics are not new fields in the
public conversion JSON.

Lines are ordered top to bottom; observations sharing at least 60% of the shorter
vertical band form a left-to-right row. A vertical gap greater than 80% of the
taller adjacent row separates paragraphs. Text columns are read column by column:
the widest gutter, at least one line height wide, between the lines narrower
than three fifths of the text splits the page when both sides hold at least
three lines of prose (a median of 12 characters) side by side. Lines crossing
the gutter, such as a title, separate sections read left column first, and
nested columns are found in each side. Short cells side by side, such as a
receipt's items and prices, stay rows. The reference's table, vertical-writing
and marginal-note reconstruction is not reproduced. On a rendered corpus
([R45](validation/ocr-quality-round45.md)) English and number text is read more
accurately than by the reference's RapidOCR, two-column pages in order, and
Chinese, with the [aids below](#chinese-recognition-aids), at 1.7 to 2.6 times
its character error rate. Vision inference itself has no wall-clock cancellation deadline
in this slice, and recognition quality is not guaranteed for handwriting, small
text or every supported language.

## Chinese recognition aids

Two steps run only when the requested language is `zh-Hans` or `zh-Hant`.
English and every other language take exactly the path described above.

- **Small text is read again, enlarged.** When the median height of the
  recognized lines that hold Han characters is below 24 pixels (12-point text at
  96 DPI or smaller), the image is enlarged with Lanczos filtering toward
  32-pixel lines, by 1.25 to 4 times and within the 32-million-pixel limit, and
  that second reading replaces the first. Rectangles stay in the original
  image's pixels. The first reading is discarded, so such an image takes about
  twice as long.
- **Dropped characters are recovered.** Vision sometimes drops a character,
  most often 的, and stretches its neighbour's box over the gap. In a line with
  at least four Han characters, a Han character box wider than 1.6 times the
  line's median Han box marks a suspect. The region of the suspect and up to four
  recognized neighbours on each side is read again. One Han character is
  inserted only when that reading places it directly before or after the
  suspect and repeats up to two neighbours on each side exactly; any other
  reading, including a failed one, leaves the line unchanged. At most 32
  regions are read again per image.

Measured on macOS 27.0.1 (26A434), Apple silicon, `cargo build -p markitai-cli
--release --locked`, before (r18, 21,731,920 bytes) and after (21,748,496
bytes). Character error rate is edit distance over ground-truth length with
whitespace removed; the reference is RapidOCR (PP-OCRv6) in the reference's
virtual environment.

| Corpus / variant | Reference | Before | After |
|---|---:|---:|---:|
| R45 Chinese 300 DPI / 150 DPI / scan-like | 0.34% / 0.52% / 0.52% | 0.86% / 1.03% / 1.55% | 0.69% / 0.86% / 1.37% |
| Held-out Chinese 300 / 150 / scan-like | 1.60% / 2.17% / 2.57% | 1.44% / 1.28% / 1.56% | 1.28% / 1.16% / 1.52% |
| Held-out Chinese 96 DPI / 72 DPI | 0.96% / 1.68% | 2.21% / 8.82% | 1.93% / 3.21% |
| Full Chinese pages 150 / 96 DPI | 1.92% / 2.89% | 1.19% / 2.97% | 1.16% / 2.06% |
| Traditional check 150 / 96 / 72 DPI | not run | 0.30% / 0.60% / 5.44% | 0.00% / 0.30% / 3.93% |

The R45 English, number and two-column text (126 images) is identical before
and after. Of 150 held-out images, 61 changed: 167 fewer and 12 more edits; the
added edits are single characters read differently from the enlarged copies.
The held-out corpus renders 30 paragraphs of the reference's Chinese guides
and changelog (disjoint from R45's text) in Heiti TC Light and Medium, Hiragino
Sans GB, Songti SC and Arial Unicode at 12 points: 300 and 150 DPI, a
scan-like copy (as in R45) and 96 and 72 DPI screenshots. The full pages are
two A4-proportioned pages in Hiragino Sans GB and Songti SC at 150 and 96 DPI;
the Traditional check is eight short paragraphs written for it, in Heiti TC
and Songti TC. All are synthetic renders, not scans or photographs.

Whole conversions (`--ocr --no-llm`, one at a time, after a warm-up, median of
five runs of the mean per image; the quieter of two alternating rounds) are
unchanged for English (153 ms) and within 5 ms for ordinary Chinese images
(207 to 236 ms) and a full 150 DPI page (656 to 661 ms). Images enlarged for a
second reading take longer: held-out 96 and 72 DPI paragraphs 202 to 268 ms
and 191 to 269 ms, a full 96 DPI page 584 to 1,284 ms. The other round, on a
busier machine, showed the same pattern with more spread.

Tried with a Swift probe of the same Vision request and not adopted: other
settings (language correction off, `zh-Hans` with `en-US`, request revision 3:
identical output; `en-US` first: unusable); enlarging text that is not small
(no gain at 150 and 300 DPI); reading every line again in its own region (the
held-out error rate rose from 3.1% to 8.7%); voting across three readings at
different scales (2 to 3 fewer edits of 1,746 for three times the time);
restoring the case of Latin words inside Chinese lines from an English reading
of their line (6 and 5 fewer edits on R45 and held-out text, 4 more on the full
pages, and up to 0.8 seconds more per page); normalizing full-width symbols
such as ＃ to ASCII (the held-out text itself uses the full-width ／); and
lexical replacements such as 己 to 已 outside 自己 (two cases in R45, none held
out). Vision never put a space between two Han characters, so no spacing
cleanup is applied. Remaining Chinese errors are mostly lookalike characters
(界/果, 器/嚣, 已/己), 的 read as another character, and Latin letters inside
Chinese lines (`l`/`I`, the case of `o`, `s` and `c`, `--` read as `-`).

## Validation fixtures

The authored [English PNG](../crates/markitai-core/src/ocr/fixtures/english.png)
contains three lines drawn from simple bitmap glyphs, without textual PNG
metadata. Its independent [expected text](../crates/markitai-core/src/ocr/fixtures/english.txt)
is `MARKITAI OCR`, `LOCAL TEXT ONLY` and `2026`. The macOS test requires every
token, valid confidence and at least three line rectangles. The same test checks
a white image, undecodable input and an unavailable language. Pure tests cover
language aliases, bounds, alpha handling, row/paragraph assembly and malformed
geometry. These are focused checks, not an OCR accuracy corpus or a claim that
results match RapidOCR on arbitrary documents.

Two macOS tests draw sentences adapted from this repository's Chinese changelog with
the in-process SVG renderer in Hiragino Sans GB: at 13 pixels per em the text
must be read exactly from an enlarged copy with rectangles in the original
pixels, while 25-pixel Chinese and 13-pixel English are read once; at 25 pixels
a 的 that this system's recognizer drops must be recovered. They depend on the
installed recognizer, like the English fixture. Pure tests cover the
enlargement threshold, factor and pixel limit, the Lanczos copy's size and
rounding, the wide-box suspects and their regions, and the anchored insertion,
including ambiguous, non-Han and mismatched readings.

Renderer-entry tests compare the same fixture's normalized PNG bytes and Vision
observations with the encoded-image path. Additional checks reject zero-sized,
oversized and excess-storage RGB layouts without allocating a maximum-sized
image, and verify that ordinary RGB rows and colors remain unchanged.

The [frozen release check](validation/native-backends-round16.md) observed a
25.897-second first image OCR call and much shorter subsequent calls. The cause
is Vision compiling its recognition models for the device on first use: it
caches them under `~/Library/Caches/<executable name>/com.apple.e5rt.e5bundlecache/`
per system build, for the executable that ran. The first OCR after installing
or updating Markitai, or after switching between two builds, takes about 25–45
seconds; later calls take about 130 ms per image. A process that may not write
that cache directory (a sandbox limited to its output folder) fails with
Vision's `missingError`. A separate small native CPU-only probe does not
justify changing the default compute policy. These observations are not an
OCR speed guarantee.

The command-line executables link Foundation, CoreFoundation, CoreGraphics,
ImageIO and Vision delay-initialized: from macOS 15, dyld initializes them on
first use instead of at launch, which saves about 1 ms in every process that
needs none of them (HTML, Office and text PDF conversions). OCR, HEIF/AVIF
decoding and PDF rasterization open their framework explicitly before first
use, about 1 ms more once per process; earlier systems initialize them at
launch as before, and the language bindings are unaffected.

Implementation references: Apple's [text-recognition guide](https://developer.apple.com/documentation/vision/recognizing-text-in-images),
[recognition request](https://developer.apple.com/documentation/vision/vnrecognizetextrequest)
and [in-memory request handler](https://developer.apple.com/documentation/vision/vnimagerequesthandler/init(data:options:)),
plus the maintained [objc2 Vision bindings](https://docs.rs/objc2-vision/0.3.2/objc2_vision/).
