# Architecture

The dependency graph points from adapters into one Rust core:

```text
markitai CLI ─────────────┐
Node N-API adapter ───────┤
Python extension ────────┼── markitai-core ── native parsers / HTTP
Go cgo ── stable C ABI ──┘
```

`markitai-core` owns configuration, conversion, metadata, output shaping,
network policy, and results. Adapters translate language-level values and
errors; they do not implement another conversion engine. The initial shared
boundary is a versioned JSON request/result, retaining the Python result
field names. Typed language wrappers provide ergonomic interfaces.

Each conversion owns its mutable state and warnings. File handles and foreign
buffers have explicit ownership. Expensive clients may be shared only through
thread-safe handles; credentials never enter output metadata or logs.

The CLI has no Python/Node/Go runtime dependency. Pure conversion and extraction
use Rust libraries. Local image and PDF-page OCR use macOS Vision by default;
Windows and Linux use the in-process portable OCR engine with separately installed
model weights. PDF pages render through CoreGraphics on macOS and hayro on
Windows/Linux, sharing a document session and bounded page pixels. See
[OCR](ocr.md) and [PDF rendering](pdf-rendering.md) for selection and limits.
Typed page state keeps native text, OCR outcomes and final capture names distinct
until output assembly. Optional
browser conversion controls an installed Chromium executable through Rust CDP.
Chromium and LibreOffice must be installed separately. Portable OCR downloads
missing model weights when first needed, with a stderr notice; `doctor --fix`
can prepare them in advance. These runtime resources have bounded input
contracts and are not included in the CLI binary size.

Packages are built with the `release` profile: workspace crates and the
conversion hot path (PDF, Office, HTML, image and regex crates) are optimized
for speed and every other dependency for size. `dist` optimizes everything
for size. Both use fat LTO and stripped symbols; measurements select a profile
explicitly. Optimizations must preserve test results before they are accepted.
