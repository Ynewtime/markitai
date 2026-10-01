# Local image OCR

On macOS 11 or later, local OCR uses the operating system's Vision framework
through Rust bindings. It runs in the current process without Python, Node,
Tesseract, a browser, downloaded OCR weights, provider credentials or a remote
recognition service. Other platforms currently return an explicit unsupported
error for this local path. This page covers image inputs, including every page
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
Chinese with two to three times its character error rate. Vision inference itself has no wall-clock cancellation deadline
in this slice, and recognition quality is not guaranteed for handwriting, small
text or every supported language.

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
