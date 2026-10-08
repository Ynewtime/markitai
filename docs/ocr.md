# Local image OCR

On macOS 11 or later, local OCR uses the operating system's Vision framework
through Rust bindings. It runs in the current process without Python, Node,
Tesseract, a browser, downloaded OCR weights, provider credentials or a remote
recognition service. On Windows and Linux it uses the [portable
engine](#the-portable-engine-windows-and-linux): PaddleOCR's models, downloaded
once into the private Markitai home, run in process by a pure-Rust inference
engine. The default macOS x86-64 build cannot use Vision under Rosetta 2 on
Apple silicon; use the arm64 build. A macOS build with `portable-media` can
instead fall back to the portable engine, as described below.
Vision and the portable engine have different language defaults; choose an
explicit language when the automatic reading is incomplete. Both share bounded
image preparation and reading-order handling.
This page covers image inputs, including every page
of a multi-page TIFF and, on macOS, HEIF/AVIF images; scanned PDF pages use the same
recognizer as described in [PDF page OCR](pdf-ocr.md), and Office page images in
[Office rendering](office-rendering.md).

```sh
markitai scan.png --ocr --no-llm
markitai scan.png --ocr --no-llm --config-json '{"ocr":{"lang":"zh"}}'
markitai scan.png --ocr --no-llm -o out/
```

## Selection and output

`ocr.enabled=true` (`--ocr` in the CLI) selects local OCR when LLM enhancement is
disabled. The image reference remains in the base document and recognized text
is appended below it. Metadata records `ocr_used: true` and `ocr_path: vision`
(`paddle` for the portable engine).
A successfully processed image with no recognized text retains its image
reference and emits a warning. An invalid image, unsupported language or engine
failure fails the conversion instead of inserting error text into Markdown. (With Vision
and its [default language](#the-default-language), only the English reading can fail
it: a Chinese, Japanese or Korean reading that this system lacks or that fails is
skipped.)

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
`<!-- Page number: N -->` marker (kept with `--page-markers`) and preview,
followed by its text, and a page
without text adds a warning instead of ending the conversion.

## Code and machine-readable text

OCR can preserve readable prose while changing the punctuation inside a code
example. In Chinese text, Vision can confuse ASCII quotes and braces with
quotation marks or full-width characters, and `#` with `＃`. Explicitly choosing
`ocr.lang=zh` does not guarantee exact code transcription. A successful conversion
means recognition completed; it does not certify that extracted JSON, commands
or configuration can be parsed or executed unchanged.

Prefer the original text or digital document when it is available. Otherwise,
compare code symbols and important values with the retained image before using
them, including in an AI Agent workflow. Markitai keeps the image reference beside
the OCR text and does not guess replacement punctuation that could alter meaning.

## Language selection

The following table describes Vision. For Windows/Linux and macOS builds using
Paddle, see [portable languages](#languages).

`ocr.lang` defaults to `en`, which is a policy of its own: [the default
language](#the-default-language) reads English and, when that fails, Chinese,
Japanese or Korean. Every other value names one Vision language, and each call
asks the configured accurate Vision recognizer which languages its current OS
and request revision support, then requires a match. It never substitutes
English for an unsupported language, and never reads another language than the
one written. System revisions can change supported languages and recognition
results.

| Configuration spelling | Requested Vision language |
| --- | --- |
| `en` | `en-US`, then `zh-Hans`, `ko-KR` and `ja-JP` when the reading failed ([default language](#the-default-language)) |
| `en-US`, `en_US` | `en-US` only |
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

A language the system lacks fails with an error that names it and lists the
languages the system reads. On macOS 27.0.1 (26A434) the accurate recognizer
(request revision 3) reads `en-US`, `fr-FR`, `it-IT`, `de-DE`, `es-ES`,
`pt-BR`, `zh-Hans`, `zh-Hant`, `yue-Hans`, `yue-Hant`, `ko-KR`, `ja-JP`,
`ru-RU`, `uk-UA`, `th-TH`, `vi-VT`, `ar-SA`, `ars-SA`, `tr-TR`, `id-ID`,
`cs-CZ`, `da-DK`, `nl-NL`, `no-NO`, `nn-NO`, `nb-NO`, `ms-MY`, `pl-PL`, `ro-RO`,
`sv-SE`, `fi-FI`, `hi-IN` and `mr-IN`. Greek and Hebrew are not among them:
`ocr.lang` `el` and `he` are rejected because this system's Vision cannot read
those scripts, not because of their spelling. This is a limit of that Vision
runtime; the portable engine has a Greek recognizer (`el`). Vision reads Thai
(`th`), Arabic (`ar`), Hindi (`hi`), Vietnamese (`vi`) and its other supported
languages when selected explicitly. (Vision returns Arabic lines with their letters
in reverse order, `لوسكلا بلكلا` for `الكلب الكسول`; they are passed on as
returned.)

## The default language

With Vision, `ocr.lang=en` is a multilingual fallback policy, not English-only
recognition.
Vision first reads English, then tries supported Chinese, Japanese and Korean
readings when the initial result is inadequate. Explicit `en-US` selects English
alone. A forced language cannot make an unsupported Vision language available.
OS versions and installed language support can change the result.

## Bounds and reading order

Encoded image input is limited to 64 MiB, decoded images to 32 million pixels,
and decoder allocations to 256 MiB. The normalized PNG passed to Vision is
limited to 128 MiB. Recognition output is limited to 10,000 lines and an 8 MiB
text budget, including line separators. EXIF orientation is applied and transparency
is composited on white. OCR uses the original image, independently of output
preview compression or size settings. PDF, TIFF and Office workflows also have
their own page and document budgets.

The reader groups lines and columns and retains page order; it is not a full
layout reconstruction engine. Small text, handwriting, vertical writing,
complex tables and marginal notes can be incomplete or misordered. Vision
recognition has no hard wall-clock cancellation deadline. A resource limit is
not a peak-RSS or accuracy guarantee.

## Turned pages, code, numbers and table cells

Recognition uses orientation and bounded alternate readings to improve turned
pages, code and number-heavy lines. These are conservative recognition aids,
not permission to infer missing data. A numeric or punctuation substitution can
still occur; compare important values and executable text with the source image.

## Chinese, Japanese and Korean recognition aids

The reader can choose alternate language readings for uncertain lines and
preserve columns, but supported scripts do not guarantee exact transcription.
Use an explicit `ocr.lang` when the default misses a script; neither repeated
recognition nor a successful exit certifies every character.

## The portable engine (Windows and Linux)

Windows and Linux read images with PaddleOCR's PP-OCR models, run in the
conversion process by the pure-Rust [`tract`](https://github.com/sonos/tract)
inference engine (`tract-onnx` 0.23.8): no Python, ONNX Runtime, Tesseract or
system service. A macOS build has it only with the `portable-media` feature
(`cargo build -p markitai-cli --release --features portable-media`); Vision
stays that build's default, the portable engine is chosen when Vision cannot
run (macOS before 11, or an x86_64 build under Rosetta), and
`MARKITAI_OCR_BACKEND=paddle` or `vision` chooses one explicitly (any other
value is a configuration error; `vision` is an unsupported-capability error
where Vision is absent). Metadata records `ocr_path: paddle`, and `markitai
doctor` names the engine in its `rapidocr` check.

### Models

The model files are not part of the executable. A manifest compiled into it
([`models.json`](../crates/markitai-core/src/ocr/paddle/models.json)) names
each one with its official URL, size, SHA-256 and license (Apache-2.0;
provenance in [`licenses/paddleocr`](../licenses/paddleocr/README.md)): the
ONNX files RapidOCR 3.9.2 publishes on its ModelScope mirror, the same the
reference downloads. A model lives at
`MARKITAI_HOME/models/ocr/<name>-<first 8 digits of its SHA-256>/<file>`.

| Role | Model | Bytes |
| --- | --- | ---: |
| Text detection | PP-OCRv6 small (`ppocrv6-det-small`) | 9,929,594 |
| Line direction (0° or 180°) | PP-OCR mobile v2.0 classifier (`ppocr-cls-mobile-v2`) | 585,532 |
| Recognition, default and Latin/Chinese/Japanese languages | PP-OCRv6 small multilingual (`ppocrv6-rec-small`) | 21,234,383 |
| Recognition, Korean | PP-OCRv5 mobile (`korean-ppocrv5-rec-mobile`) | 13,488,748 |
| Recognition, other scripts | PP-OCRv5 mobile: Arabic, Thai, Greek, East Slavic, Cyrillic, Latin, Devanagari, Tamil, Telugu | 7.8–8.1 MB each |

`markitai doctor --fix` installs the files the selected portable engine and
configured `ocr.lang` need: the default set above is 45.2 MB; an explicit
language needs the detector, classifier and its recognizer. It names each file
installed. A normal OCR downloads missing models as needed, with a notice once
per process on standard error (`Local OCR: downloading …`). It does not
necessarily download all four default files before reading its first image.
The default macOS Vision engine does not inspect or install portable models.

`markitai doctor` checks paths, size and SHA-256 with bounded, no-follow reads;
it creates no model state, downloads nothing and does not parse or run ONNX.
A verified file is not a recognition-accuracy test. The four model states are:

| State | Ordinary OCR | `doctor --fix` |
| --- | --- | --- |
| Missing | Downloads the model when needed; an offline error gives its URL, size, digest and manual path. | Installs it without overwriting a file that appeared meanwhile. |
| Ready | Verifies bytes when loading the model; graphs are reused within the process. | Keeps it. |
| Corrupt | Does not overwrite it; required-model failures name `doctor --fix`. | Can replace a safe damaged managed file after verifying the new size and digest. |
| Unsafe | Rejects an unsafe path in the selected model set before loading or downloading. | Refuses it and names the path reason before downloading. |

Unsafe entries include symbolic links, Windows junctions/reparse points,
special files, multiply linked files, foreign ownership and non-private managed
model directories/files. The Markitai home container may have ordinary readable
permissions such as 0755 on Unix, but must belong to the current user and not
be writable by others; its managed model descendants must be private. These
checks do not require every system ancestor to be private.

Privacy follows the existing platform contract: Unix checks owner and permission
bits; Windows checks the process user's SID (or its token's default owner) and
inherits the parent ACL. It does not tighten or audit every Windows ACL entry.
Use a private user directory for the managed home; model hashing and identity
checks are separate from filesystem access permissions.

Downloads use the page-fetching proxy settings and HTTPS-only redirects, and
verify each file against the bundled size and SHA-256 before publication. Repair
does not overwrite an unsafe path. A failed download leaves the existing model;
if an error occurs after publication, run `doctor` again to inspect its state.
The optional Korean reading may retain the first reading if its model cannot
load or infer. A Korean model that cannot load is tried once per process, so a
long-running `markitai serve` or MCP process uses a model installed later only
after a restart. Ordinary OCR never repairs a corrupt model implicitly.

### Languages

| `ocr.lang` | Recognizer |
| --- | --- |
| `en` (default) | multilingual, then Korean for the lines it cannot read ([below](#the-portable-default-language)) |
| `en-US`, `zh`, `zh_cn`, `cn`, `ch`, `zh-Hans`, `zh_tw`, `cht`, `chinese_cht`, `zh-Hant`, `ja`, `jp`, `japan`, `ja-JP`, `fr`, `de`, `es`, `it`, `pt`, `vi`, `tr`, … (the reference's PP-OCRv6 list) | multilingual only |
| `ko`, `korean`, `ko-KR` | Korean |
| `ar`, `arabic`, `fa`, `ur`, `ar-SA` | Arabic |
| `th`; `el`; `ta`; `te` | Thai; Greek; Tamil; Telugu |
| `ru`, `uk`, `be`, `eslav`, `ru-RU` | East Slavic |
| `cyrillic`, `bg`, `mk`, `kk`, `ky`, `mn`, `tg` | Cyrillic |
| `latin` | Latin (PP-OCRv5) |
| `hi`, `mr`, `ne`, `devanagari` | Devanagari |

Case, surrounding whitespace and underscores are normalized as for Vision; a
region or script after a known language is accepted (`pt-BR` reads as `pt`).
Any other value fails with an error that lists these. Vision's spellings are
accepted so that one configuration works on every system.

### The portable default language

The default multilingual recognizer covers Latin, Simplified/Traditional Chinese
and Japanese. Uncertain lines may be read again with the Korean recognizer;
a suitable stronger Korean reading replaces the first one. An unavailable
optional Korean model can leave the first reading in place.

When many text-like lines remain unread, the conversion preserves confident
text and warns about a possible unsupported script, low image quality or missing
optional model. The warning does not identify the cause with certainty.
Choose `ocr.lang` and check `markitai doctor --fix` as appropriate. A blank image
still receives the ordinary no-text warning.

### Limits

- Korean Hanja are not in the Korean model's character set.
- The default is not reliable for Cyrillic, Greek, Arabic, Thai or Devanagari;
  choose the relevant explicit language.
- Spaces between numbers and words, code symbols and small glyphs can change.
- First recognition and first model loading can be much slower than subsequent
  calls. Download time, cold caches and parallel work are separate costs.
- Model files are additional installed disk space, separate from the CLI binary.

Measured quality and performance have specific source, platform and corpus
boundaries. They do not establish every language,
full-document fidelity or equal performance across platforms.
