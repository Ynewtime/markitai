# Changelog

## [1.3.0]

### Added

- Give the `markitai serve` workspace a Chinese and English interface that follows the browser language, light, dark and automatic themes (printing stays light), page-wide drag and drop, upload cancellation and a `POST /api/jobs/{id}/cancel` route that stops queued items, a bottom notice region that stays in view, offline detection with reconnection, oversized files named before upload, plain explanations for skipped and failed items, a working 375-pixel layout, kept focus and expanded details across updates, and AA contrast.
- Manual operating-system proxy discovery for static fetch and the native browser (macOS, Windows, KDE and GNOME), with the reference's single environment proxy order, exception semantics and loopback bypass.
- Optional pinned official Codex CLI adapter for ChatGPT subscription conversion, authentication and Unix login, with bounded process cleanup and aggregate usage accounting.
- Base/enhanced Markdown difference preview and printing of the sanitized selected document in the embedded Web UI.
- Complete Unix CLI archive with one executable, relative command aliases and bundled project, dependency, pricing and model-catalog notices.
- Compatible `markitai-mcp` executable alias for the existing stdio service, including bare client launches and global configuration options.

- Optional official Claude CLI conversion, discovery, authentication and Unix login, preserving aggregate-only token usage without inventing request counts.

- Native frozen Batch continuation and bounded remote reconciliation, including strict manual binding of uncertain submissions and output ownership checks before paid creation.
- Optional official Copilot CLI conversion, model discovery/probe, authentication status and Unix login delegation, with observed unpriced usage and account-safe cache bypass.

- Reviewed offline model pricing, explicit unknown-price coverage, fixed-point per-document continuation budgets and retained pricing attribution in CLI and binding packages.
- OpenAI text Batch submission and collection with private durable request/result evidence, preserved observed usage and receipt-checked publication after restarts.
- Origin-scoped browser Digest authentication, including session algorithms, stale nonces and authenticated PDF downloads.
- Exact-version license authority links and an offline selectors source archive in every certified package; three recovered historical MIT notices retain separate review markers.

- Caller-owned bounded browser reuse with isolated contexts or origin/identity-scoped domain sessions, shared within CLI runs, active REST jobs and MCP connections.
- Shared-runtime usage and latency model routing based on provider observations, separate from paid usage accounting.
- Exact-byte private backups before valid legacy recovery-state adoption, plus current CLI help and removed-option migration hints.
- Offline exact-version upstream license evidence in static Go packages, preserving unresolved notice-only entries.

- Shared-runtime least-busy model selection with atomic active-request reservation and bounded private deployment identities.
- Bounded persistent learning of anonymous JavaScript-required domains, with compatible inspection and clearing commands.
- Optional macOS arm64 Go static delivery with packaged native archive/header, isolated installed consumers and dependency notice inventory.

- Modern Numbers directory packages as atomic documents in the core, CLI discovery, wizard, history and recovery, using the existing bounded Rust table decoder.

- Optional last-attempt diagnostics across CLI JSON, reports, recovery/history, REST events/retries and MCP, preserving observed failure usage without changing existing success totals.

- Shared-runtime coalescing of identical active typed document and first visual-batch requests, with separate owner accounting.
- Additive detailed Rust failures and optional terminal usage in C JSON and Python/Node/Go errors.

- Capability-aware structured model tools/schema/text transport, bounded repair and shared paid usage across validation attempts.
- Same-session authenticated browser PDF streaming into the native document and page-media pipeline.
- Complete spreadsheet sheet canvases through isolated LibreOffice, including hidden and empty sheets.
- Explicit native installation of the official Chrome headless shell with verified startup and atomic activation through `doctor --fix`.

- Typed visual document metadata, complete ordered ten-page batches and persistent caches keyed by actual page image content.
- Origin-scoped native browser Basic authentication and configuration-aware runtime diagnostics.
- Guided terminal conversion through the ordinary CLI pipeline.

- Bounded native Numbers table extraction with saved formula values and preserved sheet/table order.
- Optional isolated LibreOffice page export for presentations and word-processing documents, complete page screenshots and local OCR supplements.
- Typed document descriptions/tags, protected-literal Unicode chunking, shared request accounting and persistent per-chunk caching for local documents and text URLs.

- Rust rewrite project, architecture, compatibility audit, and development control center.
- Initial native conversion core, CLI, and Node.js/Python/Go adapters.
- Isolated contract tests, format differential audit, and reproducible CLI measurements.
- Native RST, Org and TeX readers; bounded Office metadata and PDF image extraction.
- Full reference configuration defaults and nested validation with isolated state resolution.
- Native Outlook MSG extraction, raster asset processing and standalone model vision.
- Weighted LLM routing, fallback groups, retries, request budgets and custom prompts.
- Complete 209-fixture HTML audit harness with separate compatibility and quality diagnostics.
- Bundled SQLite document LLM cache with cross-process reuse, bypass controls and CLI statistics/clearing.
- Persistent static HTML/text fetch cache with conditional HTTP requests, TTL and separate cache-hit reporting.
- Native HTML math-source and structured BBCode recovery, plus presentation slide boundaries and shape-aware extraction.
- Structured HTML footnotes with repeated-reference handling, safe definitions and preserved multi-block note content.
- Persisted CLI reports for files, URLs, directories and URL lists, with mode-specific schemas, stable task hashes and atomic conflict handling.
- Recovery-state codec and durable store with legacy ordering, process locks, bounded journals and crash replay fences.
- Unix directory/URL-list resume with durable dispatch, separate file/URL concurrency limits, restored reports and controlled interruption.
- Per-member CLI publication locks and prepared ownership receipts for safe failed-item retries across process restarts.
- Optional CLI history archives with compatible job metadata, bounded private copies, Unicode conflict handling and relocated asset references.
- Native bounded SVG rendering for model vision while preserving original vector assets.
- Shared per-run LLM request limits across file and URL workers, with retry waits and cache hits outside the limit.
- Native REST job submission, snapshots and event streams, result/asset/ZIP downloads, and persistent history access.
- Native stdio MCP with four compatible conversion tools, structured results and bounded in-memory batch jobs.
- Structural HTML article selection and technical code normalization with preserved languages, literal lines and editor boundaries.
- Native PDF layout refinement with document-wide heading levels, paragraph boundaries, continuous styling and complete ruled tables.
- Static React streamed-content recovery and a bounded compatibility heuristic for script-stripped article snapshots.

- Native macOS Vision image OCR and optional isolated Chromium fetching/screenshots through Rust CDP.
- URL screenshot-only output, persistent screenshot tiles and multi-image model requests with complete request-budget validation.
- Native macOS PDF page rendering, local per-page OCR and bounded JPEG/PNG/WebP screenshots with typed page references and complete-page vision requests.
- Static/automatic URL PDF media processing reuses bounded downloaded bytes, supports redirects and extensionless downloads, and preserves URL identity and pure-mode precedence.

- Complete bounded TIFF page decoding, orientation, original-file retention and per-page OCR/vision previews.
- Image caption/description analysis with real-reference updates, shared document request budgets and atomic cross-process images.json merging.
- Persistent REST item retry/enhance and deletion with per-item options, shared-asset ownership and recoverable output transactions.
- macOS HEIF/AVIF primary-image decoding through ImageIO, with orientation, alpha, bounded pixel checks and native OCR/vision routing.
- MIME-scoped EML body selection and Content-ID image binding, preserving unresolved references and ordinary attachment downloads.
- Private CLI file logs with level, rotation and retention controls, plus terminal configuration editing and initialization.
- Portable native validation and installable-artifact CI for Linux, Windows and both macOS architectures; remote execution remains pending.
- Embedded browser workspace with file/URL jobs, safe Markdown preview, history and model settings, served directly by the native CLI.
- Revision-checked provider/deployment settings, private atomic configuration saves and per-job configuration snapshots.
- Bounded native model discovery with credential-scoped caching, plus single-model connection probes without document conversion.

### Changed

- Vendor lopdf 0.45.0 with its glyph-name lookup as a binary search over a packed table instead of a 4,495-arm `match` (531,880 bytes of code): release binary −462,352 bytes (−2.0%), PDF output byte-identical on 216 corpus PDFs and every glyph name, speed unchanged within noise.
- Build `zip` with the same flate2/zlib-rs deflate backend but without the zopfli encoder, which it uses only above level 9 and Markitai never requests (release binary −82,800 bytes; 982 inputs convert byte-identically).
- Make a single conversion into a fresh output directory issue 3 full-cache flushes instead of 10: its claim directories are created first and share one media fence per volume, with the same per-level checks and host synchronization, before any lock or work. Single-file conversions on corpus-r4 run 23–36% faster; existing output directories are unchanged.
- Report a web page's `word_count` in URL frontmatter (each CJK character one word, as the reference counts), and keep the internal reader identity (`converter`) out of user-facing URL frontmatter.
- Match the reference's document tails and spacing: legacy DOC/PPT, RTF, ODT and ODS Markdown ends with a newline, headings drop trailing spaces, and presentation output collapses blank runs from text-free shapes and trailing whitespace. On the reference's 20 non-HTML fixtures, 16 now match byte for byte; the rest are the accepted asset names and the recorded EPUB-anchor, ODS-span and PDF-layout differences.
- Render HTML callouts and alerts (Obsidian, GitHub, Bootstrap, callout asides, Hugo/Docsy admonitions) as `> [!type] Title` blockquotes keeping collapsed bodies, TeX image-service images as `$…$`/`$$…$$` math, and lists, rules and empty quote lines in the reference's spelling (`* item`, `1. item`, `---`, `>`).
- Render EML attachments in the reference's `## Attachments` section: image attachments as images, other attachments with their size, and attached messages quoted one level deep with nested attachments listed by name; an empty body leaves a bare `## Content`. Every attachment keeps its download link and body Content-ID images stay bound.
- Use `sha2` 0.11, whose SHA-256 selects the CPU's SHA instructions at run time on aarch64 and x86, for content proofs, cache keys and asset names; digests and their spelling are unchanged, and the duplicate `sha2` 0.10 is gone.
- Reuse one filesystem observation per recovery-state validation operation instead of re-walking the same path ancestors for every saved path, halving the directory run's `lstat` calls without changing any check; recovery-state directory creation also shares one full-cache flush.
- Durably create a run's missing output directories with one shared macOS full-cache flush per volume before name reservation, instead of two flushes per directory, keeping every host synchronization and the no-work-before-commit rule.

- Verify and report three historical upstream MIT texts separately from legal review, preserving the original license bytes and exact-version notices.

- Group controlled output metadata initialization in bounded admission windows while retaining directory identities, durability fences and the original skip path.

- Bounded directory and URL-list publication groups retain durable receipts, recovery ordering and paid failure usage while sharing macOS full-cache flushes; dispatch admission is persisted in bounded windows.

### Fixed

- Read RTF and OpenDocument content the readers lost or misread: TextEdit's nested tables (`\nestcell` right after `\nestrow`) keep the comment threads they hold, `\fcharset0` text decodes as Windows-1252 whatever `\ansicpg` declares (files saved by TextEdit on a Chinese system no longer turn `’` into `抯`), hyperlinks spanning paragraphs keep their target and lists in table cells keep their markers; OpenDocument comments no longer interleave with the text (a warning names their number), hidden text and ruby guides stay out, raised and lowered text uses Unicode script forms and chart objects write their title and data table; an ODP table styled with a header row heads its Markdown table; PPTX chart values follow their number format (dates, percentages, 1904 dates); and text such as `std::vector<int>` or `<iostream>` in document formats is escaped instead of disappearing as an HTML tag. On the textutil corpus, RTF words missing against the same pages' DOCX output fell from 446 to 4 and ODT from 72 to 6.
- Stand Chrome-printed PDF text set in Type3 fonts on its baseline (it sat one font size low and interleaved with the embedded fonts of the same line), fence printed code blocks with their indentation, blank lines and URL lines, render fixed-pitch words in prose as inline code, rebuild lists whose bullets are painted shapes (nested, numbered, and numbers on a line of their own), and recognize browser superscripts (0.83 em) and `<h3>` headings (1.17 em). On 108 Chrome-printed reference fixtures, readable words missing against the reference fell from 309 to 95 of 25,684. AppKit/Quartz-printed PDFs (`1 Tf` scaled by the text matrix, `<pre>` without margins) are no longer mistaken for hidden one-point text and keep their code blocks.
- Take a workbook sheet's first row as its header (XLSX, XLSM, XLS, XLSB, as for ODS and in the reference) instead of writing an empty header row above it.
- Read X posts (both the `data-testid` and the 2026 `data-tweet-id` markup) as the post, its media and the quoted post, without avatars, player controls, counters or timelines; read Substack notes without the feed and app promotion; choose a named article over its page when the rest is recommendation, newsletter or disclaimer rails; drop spacer, pixel and icon images (emoji images keep their character); treat GitHub's page sidebar as chrome. Against defuddle's 209 expected outputs, words extra to the expected body fell from 3,555 to 2,658 with none lost, and all seven semantic contracts of the reference's captured web pages hold.
- Keep row headers, header-row data cells and spanned cells of web tables in their columns (htmd dropped them and shifted the remaining values), write data tables without header cells as tables and layout tables as their content, and leave out spacer columns; leave out in-page tables of contents, MediaWiki edit links and links that show nothing; choose one article over teaser cards linking elsewhere, and plain `article`/`article-text` regions over pages whose rest is marked `nocontent`; read Hacker News discussions as comments nested by reply level and story lists as numbered stories. Against defuddle's 209 expected outputs, words extra to the expected body fell from 2,658 to 2,192, lost words from 228 to 210 and table-row differences from 71 to 24; all seven semantic contracts still hold.
- Write a link wrapped around blocks (a card's cover, heading and summary) with the link on its heading instead of as one broken Markdown link holding blocks, and honor framework hide classes (`hidden`, `invisible`, variants such as `md:hidden`, unless a responsive class shows the element again) on full pages; against defuddle's expected outputs, extra words fell further to 2,143.
- Leave out what sits beside a web page's article on full pages: short blocks after the body (subscription boxes, calls to action, related posts, author boxes, previous/next links), link-only blocks before it (banners, breadcrumbs), eyebrow labels above the title, trailing tag and link lines, tooltips, empty sections, self-links in headings and empty bullets, while keeping code, math, notes, reference sections and data tables. Against defuddle's expected outputs, extra words fell from 2,143 to 1,655 with lost words unchanged (210), table-row differences 24 → 17, list items 63 → 34; all seven semantic contracts hold.
- Read local OCR text columns column by column instead of joining side-by-side lines into rows (two-column rendered pages: 68% character error rate → 0.2–0.4%), keeping side-by-side short cells such as receipt items and prices as rows, and report the Vision framework's error description when recognition fails.
- Resolve a saved web page's relative links against its `<base href>` or canonical link (root-relative ones only when that is the home page) and give scheme-relative addresses `https:`, keeping image paths that point at the page's saved files.
- Name every supported document and image extension when a file cannot be converted, as the reference does, and close a batch with what was converted, how long it took and what it cost, skipped items by reason with the next step, and failed and unfinished counts.
- Keep EPUB code-block languages from `language-`/`lang-` classes (rendered EPUB corpus: lost words 260 → 54, the reference 260) and write spreadsheet times without seconds when their format shows none (`h:mm` → `14:05`), through a vendored anydoc 0.2.4 whose two changes are listed in `vendor/anydoc/MARKITAI-PATCH.md`.
- Write ODP and legacy PPT slides behind the same `<!-- Slide number: N -->` markers as PPTX, blank slides included, from slide boundaries the vendored anydoc now records; the reference writes none for PPT and does not read ODP.
- Write a legacy PPT table as a Markdown table, laid out from its cells' positions with merged cells and the first row as header, instead of one paragraph per cell.
- Read Word content that was lost or misread and write it as the document shows it: the base text of phonetic guides (ruby), Symbol and Wingdings characters, non-breaking hyphens and WordArt are kept; hidden text and table rows deleted in tracked changes are left out; raised and lowered runs use Unicode superscript and subscript forms when every character has one (`10⁻³`, `H₂O`); Chinese, Japanese and enclosed-digit numbering keeps its own characters (`一、`, `（二）`, `①`); a row Word repeats as the table header heads the Markdown table; a heading with a soft return stays one heading and a heading in a table cell is bold text; list labels Markdown does not read stay one item per line; emphasis beside a letter moves its edge punctuation outside the markers so it renders (`**注意**：请`); a document with review comments gets a warning naming their number.
- Write Word text set in a monospaced font as code: whole paragraphs as fenced code blocks (blank lines kept, line numbers of pasted listings dropped) and runs inside prose as inline code, unless monospace is the document's own typeface. On the textutil Office corpus 25 DOCX outputs gain code fences or inline code; the only words that disappear are 49 listing line numbers.
- Keep adjacent cells of a tagged PDF table apart when they sit a few pixels apart (the browser's default table style), which had merged them and shifted values into the wrong columns, and use a fully tagged table that sits among long paragraphs.
- Read a tagged PDF's tables drawn without rules from its structure tree on each page, as whole-document conversion does, instead of splitting wrapped, vertically centred cells into broken rows (Chrome-printed corpus: table-row difference from the expected outputs 24 → 18, the reference 29).
- Stop turning a one-line PDF paragraph at body size into a heading when it reads as running text (it ends a sentence or a lead-in colon, or starts in lowercase), which the page reader did for any isolated line of two to six words; across 215 corpus PDFs this removed 35 page-reader headings, none of which the reference or the expected outputs have.
- Read Word 97 files saved by the macOS exporter (TextEdit, `textutil`), which neither the reference nor the native reader could open: their unused mini stream and short FAT are repaired in a copy, with a warning. Join a word the source split into several style runs into one emphasis span (a bold Persian word was one bold span per letter), write RTF list labels (`1`, `•`) as Markdown markers, and write layout tables of web pages saved as documents as their content, with nested tables as cell lines. On 108 reference fixtures saved by `textutil`, all DOC files now convert, and DOCX words missing against the reference fell from 4 to 0 and RTF from 184 to 139 of 25,321 (the remainder is the reference's duplicated comments and numbers).
- Keep the reading order of right-to-left PDF pages: layout reconstruction no longer reorders a Persian or Hebrew line's runs left to right, and such pages keep the page reader's bidirectional ordering.
- Rejoin PDF bold spans that the page reader closed and reopened at each line wrap (`**wrapped** **text**`), leaving bold-italic markers and code fences unchanged.
- Keep declarative Shadow DOM content (`<template shadowrootmode>`, open or closed, nested) in HTML conversion, as the reference does, instead of dropping it with inert templates.
- Make the whole workspace type-check and pass `clippy -D warnings` for `x86_64-pc-windows-msvc`, as the configured Windows CI job runs it: the registry proxy decoder satisfies Windows-only lints, and Unix-only batch publication, recovery-state and resumed-report code no longer warns on platforms where the batch command reports it as unimplemented.
- Reconstruct PDF pages with link annotations, image XObjects, curved marks or rounded clips, keep a same-width bordered block from merging into a table's grid, and stop table cells and fixed-pitch code from turning ordinary prose into headings.
- Keep the numeric cell of a table row near a page edge that PDF page-number cleanup deleted as a folio when no other page corroborates it, extract ICC-based, calibrated and indexed PDF images, and reconstruct ruled tables drawn under a neutral graphics state such as Chrome's print output.
- Write HTML tables in the reference's compact row spelling (`| a | b |` with one `---` per column) instead of padding cells to column width, and place an XLSX/XLS sheet table directly under its heading as the reference does.
- Include the required `ttlMs`/`cacheScope` cache directives in 2026-07-28 MCP tool listings, which the official Python SDK client otherwise rejects, and accept integral numbers such as `2.0` for `batch_convert` concurrency.
- Quote frontmatter strings that YAML 1.1 readers would type as timestamps, booleans, numbers or null, matching the reference output bytes for the processing time.
- Create output documents, assets, image sidecars and reports with the process umask instead of private 0600, keeping ownership records and recovery state private.

- Hide links carrying the service token when the Web workspace itself is printed, so browser-generated PDFs never embed authenticated URLs.
- Render only the document body in Web previews and prints, leaving YAML frontmatter to the Source view as in the reference workspace.

- Save complete inline base64 images in converted Markdown as owned assets when an output directory is used, and keep an alt-text placeholder for HTML images whose source is inline data instead of dropping them, matching the reference.

- Terminate official subscription runtimes, Chromium and LibreOffice process groups started by the Unix CLI before it exits on an interrupt, termination or hangup, so they no longer outlive a conversion.

- Keep the current supplemental-license unresolved list consistent with verified package records, preserving the original collection list separately as history.

- Synchronize newly created output ancestors before batch filename probing so later ownership checks cannot bypass their directory-entry durability.

- Keep available URL and file worker capacity supplied when the other class has an earlier-sorted backlog.

- Explicitly release owned file locks before inherited child descriptors close, including partial acquisition and validation failures.

- Correct Copilot protocol connect fields and its independent cache directory, verified against the pinned official runtime offline.
- Release checkpoint locks explicitly across fork/exec inheritance; retry interrupted fixture/runtime reads without extending deadlines.
- Remove redundant checkpoint directory barriers while preserving per-append file durability, and avoid unused worker classes.

- Detect response-body timeouts when recording routing latency penalties.
- Resolve the selected macOS SDK automatically for isolated Go static package consumers.

- In-memory compatibility for legacy enhanced histories, consistent result/retry/delete families and preserved native filename ownership.

- Bounded long-text splitting for literal marker-like input, and stop visual dispatch when the first metadata batch cannot succeed.
- Private Unix permissions for Office diagnostic and rendering workspaces.

- Concurrent first-time SQLite cache initialization retries bounded WAL/schema lock contention without replaying stored-row transactions.

- PDF page-number cleanup preserves substantive Page N paragraphs; nonpainting text modes persist across text blocks and nested Forms without changing glyph positions.
- VLM OCR opt-out selects local recognition; explicit PDF screenshot enhancement remains an independent request and reports when it sends page images.
- Screenshot-only history retains every capture tile without treating JPEG bytes as Markdown.

- HTML code targets share membership and language indexes; documents without note references skip footnote indexing, and PDF inspection and table geometry reuse each page's decoded operations.
- Markdown cleanup preserves fenced code bytes, including blank lines, trailing spaces and literal image/link examples.
- Ordinary and pure output assembly, structured-data title fallbacks, XML prose, and email dates.
- PDF page failures retain readable pages and report incomplete extraction explicitly.
- HTML metadata precedence, quotes, soft line breaks and safe URL spelling.
- CSS custom properties, quoted strings and comments no longer hide visible HTML content.
- Source line breaks preserve separating spaces after inline links, code and footnote references.
- Footnote detection preserves hidden-content, mathematical, navigation and continuation boundaries.
- CLI paired flags use their last occurrence; image-only skips and compression controls preserve their existing behavior.
- Report skip preserves existing bytes, and enhanced-output reports point to the finalized document.
- Named URL identities retain their original spelling; HTTP URLs ending in `.urls` remain URLs, and empty-directory JSON stays machine-readable.
- Pure URL conversions retain the actual fetch strategy for CLI reporting without adding a binding JSON field.
- Concurrent writers reuse identical content-addressed assets after exact byte verification; conflicting stored bytes fail without replacement.
- HTML image candidate lists and media attributes retain their asset destinations after filtering, publication and history relocation.
- Asset preparation and publication rewrite original paths in one pass, preventing renamed paths from being redirected or removed by a later asset.
- CSS resource destinations in HTML styles follow asset renaming and filtering while preserving ordinary strings, comments and code examples.
