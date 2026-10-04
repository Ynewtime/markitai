# Changelog

## [1.3.0]

### Added

- Native Rust conversion core, standalone CLI, and in-process Python, Node.js and Go bindings. CLI packages need no Python, Node.js or Go runtime.
- Windows x64/ARM64 ZIP and macOS/Linux CLI archives with command aliases, dependency licenses and offline Markdown guides, including `llms.txt` and `llms-full.txt`. Optional static Go packages support macOS arm64 and Linux x86-64; Windows Go/cgo distribution remains unvalidated.
- Embedded `markitai serve` workbench with Chinese/English, light/dark themes, responsive layouts, file/folder/URL submission, upload progress and cancellation, result previews, base/enhanced differences, history, retries and downloads.
- REST conversion jobs, event streams and OpenAPI documentation; native stdio MCP conversion tools and bounded batch jobs through `markitai mcp` or `markitai-mcp`.
- Native RST, Org, TeX and Outlook MSG readers, notebook outputs, Numbers tables and directory packages, MIME-aware email attachments, and Office metadata extraction.
- Local image/PDF OCR with macOS Vision or portable PaddleOCR, with explicit language selection and English/Chinese/Japanese/Korean default routing. Portable models install separately with integrity checks; `doctor` inspects them and `doctor --fix` repairs them.
- PDF screenshots and scanned-page OCR on Windows/Linux through the in-process hayro renderer; macOS uses CoreGraphics by default. Optional LibreOffice export adds Office screenshots and OCR. Multi-page TIFF, SVG vision previews and macOS HEIF/AVIF decoding are also supported.
- PDF hidden-text policy `security.pdf_sanitize` with `off`, `warn` and `remove` modes. Supported suspicious text can be excluded from the extracted body; original assets remain intact, and incomplete inspection is reported.
- Optional isolated Chromium fetching, screenshots, origin-scoped Basic/Digest authentication and authenticated PDF downloads; explicit browser installation through `doctor --fix`, reusable isolated sessions and operating-system manual proxy discovery.
- Cloudflare URL/file conversion and Jina/Defuddle fetch fallbacks, subject to remote-processing consent and local-only policies. The workbench requests consent per operation; privacy notices identify off-device image and URL processing.
- Weighted model routing, fallback groups, retries, concurrency and request budgets, custom prompts, structured document metadata and image descriptions. Persistent fetch/LLM caches and identical-request coalescing avoid repeated work.
- Additional OpenAI-compatible providers, model discovery and connection probes, plus optional Copilot, Claude and ChatGPT subscription conversion through supported official runtimes. Subscription authentication status and delegated login are available.
- OpenAI text Batch submission, collection and restart continuation with private request records, reconciliation and output ownership checks. Observed usage, unknown pricing and failed-attempt costs remain distinguishable.
- CLI reports, optional history, resumable directory/URL-list batches, controlled interruption, publication locks and recovery receipts. Shared assets and stdout image assets persist under isolated Markitai state.
- Chinese/English CLI help and guided conversion, configuration initialization/editing/validation, capability diagnostics and private rotating logs. Windows supports native console handling, subscription command shims, recovery and process-tree cleanup.

### Changed

- Omit PDF page and slide-number comments from final Markdown by default; `output.page_markers` and `--page-markers` keep them. Internal page alignment, RAG page provenance and literal code examples are preserved.
- Workbench composer: Options now includes the CLI command line, Upload offers file or folder selection from one split button, the URL card has a clearer focus state and a filled Convert button, and the appearance menu is more compact with a new icon.
- Remove personal machine paths from repository documentation and provenance; add checks to prevent committing private paths and common secret formats.

- Minimum Rust version is now 1.92. Updated compatible dependencies while retaining the existing Node API level.
- Reduce repeated PDF/HTML parsing, cache font and layout work, share compiled helpers, and reduce binary/static-library size. macOS defers optional framework loading; Python loads optional modules on demand.
- Reduce redundant output synchronization while retaining durable publication and recovery ordering. Output files follow the process umask; private state remains private. Trusted directory aliases are accepted while document and metadata links remain guarded.
- Improve narrow-terminal help, batch summaries, dry-run previews and actionable errors. Keep exit-code reference material in the documentation rather than the help footer. Configuration validation identifies explicitly set options that have no runtime effect; machine-readable results retain stable fields.
- Cache CI dependencies, avoid duplicate Windows CLI builds, and show package-stage progress and timeout diagnostics. Simplify user and Agent guides; retain development history in Git instead of duplicate status documents.
- Document binding API/version differences, platform capability limits and OCR transcription limits. Readable OCR output does not guarantee exact punctuation in code or configuration; Rosetta Vision OCR requires the native arm64 build.

### Fixed

- Preserve clearly bounded PDF bar charts as images at their text position instead of flattening axes and legends into prose; retain text when rendering or placement cannot be verified.
- Preserve legacy DOC underlining from direct formatting and inherited character styles (link labels stay plain links) without applying it to unrelated repeated text.
- Recover supported legacy DOC floating pictures at their body-text anchors, including separately stored image data, without importing unrelated images.
- Require serve tokens for local and remote API access, enforce mutation Origin checks, and bind stored LLM credentials to their configured endpoints. Reject client-supplied environment references; use authenticated resource loading and single-use download tickets in the workbench.
- Bound Office/EPUB metadata memory use, handle binding serialization failures without panics, reject non-UTF-8 physical output directories before publication, and convert Python configuration panics to ordinary exceptions.
- Restore interrupted serve jobs as visible failures, retain completed outputs, prevent concurrent history writers and clean expired private upload stages. Retries retain prior successful results and their options, while reporting new failures and costs separately.
- Model connection tests and model discovery read keys from `.env` and `~/.markitai/.env` like conversion does, instead of failing with HTTP 401 when the key is only in a dotenv file.
- Workbench errors use reusable notifications instead of expanding rows. Correct stopped-item status, saved retry options, stale live updates, unavailable-model guidance, keyboard focus, mobile layout and project links; printing excludes authenticated link targets.
- Improve PDF column and right-to-left reading order, document-wide heading levels, lists, code, links, annotations, forms, tagged/ruled/borderless tables and cross-page continuation. Preserve valid page-edge data and warn when only part of a document is readable.
- Read supported searchable-scan OCR layers, non-embedded CJK fonts, indirect form resources and owner-password-only PDFs correctly. Password-required PDFs still fail explicitly; password form values are omitted.
- Preserve Office/OpenDocument/RTF text, headings, lists, tab-aligned tables, monospaced code, symbols, fields and supported embedded content. Keep presentation slide boundaries, blank slides, tables and supported chart data; exclude hidden/deleted content without dropping ordinary text.
- Correct spreadsheet headers, spans, formulas without cached values, hyperlinks, notes, number/date/duration formats and supported picture placement. Improve XLSX/XLSM screenshot overflow and default font colours without changing original documents; tolerate transient LibreOffice output files.
- Improve HTML article/site extraction, tables, footnotes, callouts, math, code, lazy images and saved-page links; preserve streamed and declarative Shadow DOM content. Remove navigation and unrelated rails without losing short article content.
- Extract embedded X Articles with their actual title and document structure, excluding duplicate layouts, author controls and replies from the surrounding timeline.
- Keep Markdown literal text, fenced code, hard line breaks, footnote boundaries and safely quoted YAML metadata. Preserve EPUB code languages, notebook outputs, email body/attachment associations and original binary downloads.
- Correct non-UTF-8 HTML, text, CSV and email decoding; sniff CSV/TSV delimiters without losing quoted fields, decimal commas or columns wider than the header.
- Improve local OCR for small CJK text, rotation, columns, table numbers and code indentation; explain unsupported languages and recognition failures without substituting garbage text.
- Treat login, verification and JSON refusal pages as fetch failures instead of successful documents, preserving service warnings and cache controls. Refresh cached extraction after upgrades; improve network/encoding errors and authenticated PDF handling without bypassing site refusals.
- Keep model fallback working after a deployment rejects authentication, classify billing/region errors, respect declared input windows and retain observed failed-attempt usage. Correct eligible Anthropic pricing, trim repetitive model tails and avoid caching those answers.
- Fix recovery option matching, output conflicts, asset relocation and concurrent cache/asset publication. Preserve prior bytes on skips, validate content before asset reuse and stop queued MCP work when its client disconnects.
- Correct Windows path aliases, locks, atomic renames, LibreOffice discovery and external process cleanup. CLI ZIPs use a static CRT and check runtime DLL imports; Linux static Go consumers use a non-executable stack.
- Handle closed stdout pipes quietly, honour the last paired flag, report interrupted and skipped work accurately, and return failure when a noninteractive cache-clear confirmation cannot be read.
