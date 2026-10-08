# Images and shared assets

Raster preparation lives in the Rust core, so CLI and language adapters use the
same decoding, resize and asset rules. Native decoding uses no Python process or
external image editor. Optional [image enrichment](image-enrichment.md) can fetch
actual remote image references before passing their bytes to these decoders.
JPEG, PNG, GIF, BMP, TIFF and WebP use Rust codecs. Static SVG inputs are rendered
natively with resvg. On macOS, HEIF/HEIC and AVIF use ImageIO's native codecs.

## Standalone inputs

A standalone raster file with LLM enabled is decoded within resource limits and
sent as actual image content to the configured vision-capable model pool.
Orientation is applied when encoding a new payload. The
base document retains a local image reference, while the enhanced document is
the model's Markdown. Standalone inputs bypass embedded-image size filtering.
When re-encoding GIF or animated WebP, the first image supplies the pixels.
With compression disabled their original streams are retained; provider-specific
animation handling has not been validated. Multi-page TIFF follows the document
page workflow below; animated GIF/WebP frames are not treated as document pages.

With both LLM and OCR disabled, CLI image items are skipped with
`skip_reason="image_only"`; no Markdown file is written. The shared library API
returns a conversion error explaining `llm=True` and `ocr=True`, matching the
reference Python API. Missing files still report a missing-file error.

`ocr=True` without LLM selects [local image OCR](ocr.md): Vision on macOS by
default, or the portable engine with installed model weights on Windows/Linux. With LLM
enabled, it selects image vision unless `MARKITAI_NO_VLM_OCR` disables that
upload; then local OCR supplies text for enhancement without image blocks.
[PDF page OCR and screenshots](pdf-rendering.md) use CoreGraphics on macOS and
the in-process hayro renderer on Windows/Linux.
[Browser screenshots](browser.md) are available with an installed Chromium executable.

## HEIF and AVIF

The macOS decoder accepts bytes in memory, validates the ISO BMFF container's
declared box boundaries, then asks ImageIO for the actual format and primary
image. Content takes precedence over a misleading filename. The primary image
is selected explicitly, including when its index is not zero. Other images in
a collection or sequence are not additional document pages: standalone and
embedded conversion warn when the container has more than one image. This
matches the reference's primary-image behavior; it does not claim complete
HEIF sequence extraction.

Image dimensions are checked before pixel decoding against the 32-million-pixel
limit, then checked again against the returned bitmap. ImageIO must report a
complete image; a partially available image is an error. Pixels are drawn into
an explicit sRGB RGBA8 buffer, with transparency preserved and EXIF orientation
applied once. Local OCR receives upright pixels composited on white. Vision and
embedded output use PNG when compression is disabled, or the configured encoder
when enabled; HEIF/AVIF bytes are never labeled as PNG and sent unchanged.
Standalone inputs retain the normal one-preview document shape. Original files
are not modified.

There is no Python, libheif or external decoder process. Other platforms report
`unsupported`, as does a macOS runtime unable to decode a particular codec. OS
codec availability depends on the installed macOS version; current verification
does not establish support on every version allowed by the binary deployment
target. HDR/high-bit-depth images are converted to an 8-bit sRGB representation;
HDR tone fidelity and every HEIF coding variant are not claimed. The owned input,
pixel and encoded buffers are bounded, but ImageIO's internal allocations and
decode time are not an OS sandbox or a peak-RSS guarantee.

Regression fixtures include locally authored HEIC quadrant images with EXIF
orientations 1 and 6, a two-image collection whose second image is primary,
and local OCR text. Real AVIF white-pixel and transparent-circle files come from
libavif v1.3.0 under its stated BSD license. Source URLs, exact hashes and local
generation commands are stored beside the fixtures in
`crates/markitai-core/src/images/fixtures/heif/provenance.json`; the license and
authored HEIC generator are included. Tests inspect decoded pixels, alpha,
primary-image selection, truncated data, actual model PNG payloads and local
OCR without model image uploads. A system capability list alone is not decode
evidence.

## Multi-page TIFF

TIFF and BigTIFF content is detected from bytes, including under a misleading
supported image suffix. All linked page directories are inspected before OCR,
preview encoding or a model request. Pages retain their file order and each
page's EXIF orientation. Grayscale, RGB, RGBA and CMYK use the existing native
TIFF/image codecs; unsupported sample layouts or compression methods fail
explicitly. Additional TIFF SubIFD thumbnails are not extra document pages.

A multi-page document contains an original TIFF download link, then a preview
for every page, each after a page marker when `--page-markers` keeps them. The
original bytes are retained even when compression is enabled. Page previews use
the configured encoder, or lossless PNG at original upright dimensions when
compression is disabled. MIME types describe the encoded bytes. Single-page TIFF
retains its existing one-preview output.

Local OCR recognizes each upright page before advancing to the next page. It
receives original-resolution RGB pixels composited over white, independently of
preview resizing. Blank pages retain their marker and preview with a page-specific
warning. With vision enabled, all page previews are sent in order in one request;
`MARKITAI_NO_VLM_OCR` selects local OCR and sends only recognized text for LLM
enhancement. This extends the reference vision route, which previously prepared
only one TIFF preview. The reference local OCR route already reads all TIFF pages.

Limits are 1,000 pages, 32 million pixels per page, 2 billion cumulative pixels,
100 MiB of multi-page input, and 256 MiB of retained original/encoded asset and
vision bytes combined. Decoding retains one page's pixels at a time, with the
existing 256 MiB decoder allocation ceiling. Encoders fail at their remaining
byte budget rather than returning a truncated image. OCR Markdown is bounded to
64 MiB. These are resource budgets, not a measured peak-RSS guarantee: codec
workspaces, orientation/white-background buffers and request serialization need
additional memory. A positive `llm.max_vision_pages_per_document` is checked before
any page pixels are decoded for vision; it does not restrict local OCR. Malformed
later pages or exhausted limits fail the whole operation, rather than reporting
earlier pages as a complete document.

Embedded multi-page TIFF assets are preserved unchanged during ordinary asset
compression, so that pass cannot flatten them to the first frame. Image analysis
uses the same all-page vision preparation. No external decoder or Python runtime
is used. Local OCR follows the same platform and model requirements as
[single-image OCR](ocr.md).

## SVG inputs

A standalone SVG retains its original bytes as the document's SVG asset. The
vision request receives a separate PNG rendered in process; no Python, browser,
external executable or network fetch is involved. Actual raster magic takes
precedence over an incorrect `.svg` suffix, and SVG XML under a supported image
suffix is recognized as SVG. UTF-8 XML, viewBox sizing, static shapes, gradients,
clipping, masks, text and embedded base64 PNG/JPEG/GIF/WebP images use resvg's
static renderer. The PNG preserves transparency and renders at 2048 pixels wide with proportional
height, matching the reference's default vision width even for small vector
viewBoxes. Raster compression and maximum-width settings do not lower this
vector preview resolution. An output exceeding 32 million pixels is rejected
before allocating the canvas. Renderer and font differences remain possible.
When the host lacks fontdb's default serif, sans-serif or monospace family, the
first installed face covering basic Latin letters stands in; icon-only fonts
are skipped, so SVG text on a host without a text font fails as unavailable.

System fonts are loaded lazily once when text is present. Their files are the
only ambient rendering resources read. SVG-specified file paths and network
resources are never loaded. Font selection therefore depends on installed fonts;
missing required fonts or glyphs are explicit errors. Empty or fully transparent
renderings fail instead of uploading a blank image. Script/event handlers,
animation, foreignObject, XML processing instructions/DTDs, external image/use
references and nested SVG data images are rejected. Hyperlinks are preserved
but never followed. Unsupported SVG features outside resvg's static support are
not claimed to have browser fidelity. Embedded SVG assets inside other document
formats retain the existing original-byte/warning path. Standalone SVG vision, local OCR and [image enrichment](image-enrichment.md)
use the separate bounded PNG.

## Descriptions in history archives

[Image enrichment](image-enrichment.md) can publish an `images.json` index beside
the output assets. CLI history merges indexes from all copied source roots. It
keeps the first index's header and all matching source records, including distinct
descriptions of an asset whose identical bytes share one archived file. Each
record's `path` is rewritten using the actual collision-renaming map to the final
history directory; the remaining record fields are preserved. Indexes no longer
point back to the original output directories after those directories are removed.

Shared indexes may describe assets outside the current archive. Rows without an
asset in the copied set, missing/invalid file paths, and URI paths are omitted;
stderr reports the number omitted. Invalid JSON or a non-array `images` field
fails the optional archive, preserving the source and converted outputs. Reading
all source indexes has a combined 16 MiB limit, each merged index has a 16 MiB
serialization limit, and index bytes also count toward the archive budget.
Internal `.images.lock` files are validated as regular files and excluded from
the archive. Symlinked index/lock leaves are rejected.

## Embedded assets

With an output directory, or in CLI stdout mode while
[`image.stdout_persist`](#images-on-stdout) is on, complete inline
`data:image/...` references in any converted Markdown become owned assets
first, as the reference workflow saves embedded base64 images; they then follow
the rules below. Undecodable data keeps its original reference with a warning.
A converter's elided placeholder such as `data:image/png;base64...` (the HTML
reader keeps no payload) is left as it is without a warning. In-memory library
calls keep the data URI inline instead of naming an unwritten file. Local and
remote image references are localized only by image enrichment.

After document extraction and before output profiles, the core processes raster
assets already held in memory. It applies EXIF orientation, configured minimum
width/height/area filters and raw-byte deduplication. All references to a duplicate
are rewritten to the retained asset. Filtering removes rendered references;
examples inside code spans and fenced blocks remain literal.

Each preparation stage collects original paths and applies one mapping to the
document. File publication does the same for content-addressed destinations in
both base and enhanced Markdown. A generated filename matching another original
asset name therefore cannot redirect or remove an already relocated reference.
When input assets use the same source path more than once, the first entry
determines that path's reference mapping, including retained or filtered assets.
The original asset order is preserved.

Rewriting handles Markdown links, reference definitions, wikilinks and exact
frontmatter path scalars, plus HTML `a[href]`, `img[src/srcset]`,
`source[src/srcset]`, `video[src/poster]`, `audio[src]` and `track[src]`.
All relevant attributes are visited once; the first duplicate attribute wins.
Candidate lists preserve surviving descriptors and URI suffixes. Filtering a
candidate retains alternatives; filtering media destinations removes the affected
attribute rather than its container. Audio/video sources alone do not count as
image references; image candidate lists and video posters do. Static CSS resource
positions in actual style attributes/blocks are also relocated, as specified in
[CSS resources](css-resources.md). Ordinary CSS strings, custom lazy-load
attributes, code and comments remain literal.

With compression enabled, maximum width/height use proportional Lanczos resize.
JPEG uses the configured quality and composites alpha onto white. PNG keeps
transparency. WebP currently uses a lossless native encoder, with a warning that
`image.quality` has no effect for WebP. Native encoded bytes are not claimed to
match Pillow's encoders; image-output parity needs pixel/quality measurements in
addition to the existing strict asset-hash audit.

With compression disabled, supported embedded image bytes remain unchanged.
The actual decoded format determines the output suffix. Standalone JPEG/PNG/
WebP/GIF assets and vision payloads also preserve original bytes in this mode;
BMP/TIFF/HEIF/AVIF previews become PNG without resizing. Metadata such as EXIF orientation
is retained in the unchanged payload; orientation is baked in only when encoding
new pixels. Provider size limits are not yet reproduced for every service.

Unsupported or malformed embedded images retain their original bytes with a
warning. This is an explicit fidelity choice: conversion does not quietly erase
an asset simply because its codec is unavailable. Non-image attachments bypass
raster processing entirely. Input files are never modified.

## Images on stdout

Without `-o`, a single file or URL is printed to stdout and no output directory
exists. The CLI then saves the images and page captures the document refers
to in one store and links them with absolute `file://` URIs, so the printed
Markdown opens with its pictures after the process exits. The following example
uses the fictional account `example-user`:

```text
![Chart](file:///Users/example-user/.markitai/assets/blobs/7416822bd6078af29cc72e66.jpg)
<!-- ![Page 1](file:///Users/example-user/.markitai/assets/blobs/336f4a11bd5d2a0c50ade34e.jpg) -->
```

| Key | Default | Effect |
|---|---|---|
| `image.stdout_persist` | `true` | Save stdout images and link them; `false` keeps `.markitai/...` references that point nowhere and prints one warning |
| `image.stdout_persist_dir` | `~/.markitai/assets` | Store directory; the default follows `MARKITAI_HOME`, another path keeps its meaning, a relative one resolves against the current directory |
| `image.stdout_fetch_external` | `false` | Accepted; no effect in this build (see below) |

Files are written below `blobs/` and named by the first 24 hex digits of the
SHA-256 of their bytes, the same names `-o` uses for assets, with the source
extension reduced to ASCII letters and digits. Identical images, within a
document or across runs, share one file; an existing file is verified byte for
byte and never rewritten, so links printed earlier keep showing the same image.
A file under that name with other bytes (damaged, or edited by hand) is left
alone and the full 64-digit digest names the image instead. New files are
written with the same no-clobber staging and ordering barrier as output assets.
Links name the canonical store path. Base and enhanced Markdown, Markdown
links, HTML image and media attributes, CSS resource positions and the
generated `<!-- ![Page N](…) -->` page references are rewritten; code and other
comments stay literal. Only referenced images are saved: a document without
image references does not create the store. The store is never cleaned up; it
is not part of `markitai cache clear`, and deleting it only breaks links in
Markdown printed earlier.

Persisting runs before output profiles, so `rag` and `obsidian` keep the
`file://` image links (an Obsidian wikilink cannot carry one). `--pure`, LLM
enhancement, image analysis (`--alt`/`--desc`, whose records then name the
stored file; no `images.json` is written), OCR and screenshots all use the same
store. `-o`, `--json` (which requires `-o`), directory and URL-list runs, `serve`,
MCP and the language bindings never use it: their results keep relative
references.

The stdout asset store uses the core path policy: with `output.allow_symlinks`
off, a symbolic link anywhere in the store path (other than a root-owned system
link such as macOS `/var`) is refused and nothing is written through it. This
internal store does not use the CLI's user-selected directory-alias resolution. When
images cannot be saved, the conversion still succeeds; those references stay
relative and stderr says how many failed, where and why.

Differences from the reference 1.2.0, which uses `blobs/<16 hex digits>` names
(compared on macOS with a debug build, generated DOCX/PDF/HTML fixtures and the
reference `sample.pptx`, running the reference with a private `HOME`):

- The reference expands `stdout_persist_dir` against the real home even when
  `MARKITAI_HOME` is set; here the default follows `MARKITAI_HOME`.
- It also keeps a `refs/<source>/<image>` symlink index. That index is
  mutable, keyed by source and image names, and only a browsing aid, so it is
  not created here.
- With persistence off or a failed save it replaces each reference with an
  `![image: name]()` placeholder; here the original reference and its alt text
  remain and the warning explains them.
- On a terminal with an inline image protocol it prints images inline, and
  `stdout_fetch_external` downloads remote images for that display. This build
  has no terminal image output, so remote images stay as links.

## Resource boundaries

Each raster decode allows at most 32 million pixels. Rust codecs use a 256 MiB
decoder allocation budget; ImageIO uses the separate boundaries described above.
Dimensions are checked before decoded pixel allocation;
codec limits can reject earlier. The total temporary working set may be larger
while resizing, compositing or encoding because those buffers coexist. This is
not a peak-RSS guarantee. General local inputs have the reference's 500 MiB
preflight limit; individual parsers retain their tighter format-specific limits.

SVG XML is limited to 8 MiB, 50,000 nodes and 64 nesting levels before the recursive XML parser or renderer runs.
The natural SVG canvas and the total decoded embedded raster pixels each have
a 32-million-pixel limit. SVG render surfaces, font data, geometry, filter
temporaries and encoding buffers can coexist; these bounds are not a total
process-memory or execution-time guarantee. PNG/JPEG/GIF/WebP embedded payloads
also pass the existing raster decoder limits. No SVGZ decompressor is enabled.

The implementation uses the bounded decoder interfaces in
[image](https://docs.rs/image/0.25.10/image/struct.ImageReader.html) and its
[resource limits](https://docs.rs/image/0.25.10/image/struct.Limits.html).
No codec downloads or external executable installation occur during conversion.
SVG uses [resvg 0.48.1](https://docs.rs/resvg/0.48.1/resvg/) with explicit text,
system-font and raster-image features.
