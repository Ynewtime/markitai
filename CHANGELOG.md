# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Live progress checklist (StageList)**: multi-stage live progress for single-URL and single-file conversions — completed stages persist as `✓ Fetched via fxtwitter (2.1s)` lines, and the active stage shows a spinner with an elapsed-time suffix. Stdout-mode conversions (no `-o`) finally show progress; they were previously fully silent through fetch + LLM enhancement

### Fixed

- **`--resume` was a no-op**: the CLI batch entry point accepted the flag but always reprocessed every file from scratch. It now correctly loads saved state — completed files are skipped, failed/interrupted files are retried, newly-discovered files are picked up — and reports `Resuming batch: N completed, M remaining`
- **Output naming reverted to the append scheme**: `sample.pdf` → `sample.pdf.md` (not `sample.md`), undoing the 0.15.0 extension-replacement change, which hid the source format, mangled multi-suffix names, and made single-file and batch conversions of the same file disagree

### Removed

- **Legacy progress facilities retired**: internal `ConversionStatus`, `ProgressReporter`, and `OutputManager` are replaced by StageList; `markitai.utils`/`markitai.cli` no longer export `ProgressReporter`, and `attempt_login()` lost its unused `output_manager` parameter

## [0.18.0] - 2026-07-09

### Changed

- **Web extraction parity with Defuddle**: main-content selection and noise-pattern removal now closely port Defuddle's algorithms (scoring, content patterns, content-boundary detection). Benchmark corpus mean vs. the Defuddle ground truth: 91.04 → 92.72

## [0.17.0] - 2026-07-08

### Removed

- **Gemini CLI provider** (`gemini-cli/`) — Google retired the underlying OAuth onboarding. Use a direct `GEMINI_API_KEY` or route through OpenRouter instead

### Added

- `COPILOT_GITHUB_TOKEN` auth support, checked ahead of `GH_TOKEN`/`GITHUB_TOKEN`
- Elapsed-time indicator on slow conversion stages, so a long-running LLM call doesn't look hung

### Fixed

- **Cloudflare fetch now respects site-specific extractors**: switched from the `/markdown` to the `/content` endpoint so Cloudflare-routed pages get the same extraction quality as every other strategy
- Reduced console log noise from duplicate retry/validation messages

### Changed

- Local providers no longer waste tokens on unused extended-thinking/reasoning output
- Provider detection and `init` now suggest cheaper default models

## [0.16.0] - 2026-07-07

### Added

- **Bilibili opus extractor** for `bilibili.com/opus/<id>` posts
- **Anti-bot/CAPTCHA detection**: challenge pages (Geetest, Cloudflare, reCAPTCHA, hCaptcha) are now recognized instead of silently treated as real content

### Changed

- **X/Twitter extraction is DOM-first again**, falling back to the FxTwitter/oEmbed enricher only when native extraction comes up short

### Fixed

- X Article URL matching, fetch performance, and frontmatter word-count bugs

## [0.15.0] - 2026-07-04

Maintenance overhaul: dependency refresh, Python 3.14 support, and a multi-round audit fixing 30+ verified bugs across batch processing, fetch/cache, LLM providers, image handling, and configuration.

### Added

- Python 3.14 support; MIT License added to package metadata
- Grouped, faster `--help` via rich-click
- Garbled/scanned-text detection for PDFs, with an advisory suggesting `--ocr`
- Repeated header/footer suppression across PDF pages
- VLM degeneration guard (truncates repetition-loop vision/OCR output)
- HTML extraction and footnote-handling parity with Defuddle (MathJax/MathML, code blocks, footnotes across many site types)
- Unified `-s/--strategy` fetch flag (old per-backend flags kept as deprecated aliases)
- **Remote-fetch consent**: URLs are no longer sent to third-party services without consent (`fetch.remote_consent`, `MARKITAI_NO_REMOTE_FETCH`)
- PDF hidden-text sanitization (prompt-injection guard): `security.pdf_sanitize`
- Per-page OCR routing for mixed digital/scanned documents
- Conversion-quality benchmark harness against a Defuddle ground-truth corpus
- Release automation via release-please

### Fixed

Selected highlights from a multi-round quality and bug-hunt pass:

- X/Twitter DOM extractor rebuilt for X's 2026 markup redesign; FxTwitter fallback now actually reachable from the default fetch chain
- `.eml` email support (native), HEIC/HEIF/AVIF image input (`markitai[heif]`), quality guardrails gate for CI
- `markitai init` merges into an existing config instead of overwriting it; clearer login-failure guidance across providers
- Fetch/cache correctness: stale AUTO-strategy cache revalidation, Playwright context leaks, proxy auto-detection false positives, null URLs crashing batches
- LLM/provider correctness: failures no longer silently reported as success, vision cache poisoning, Copilot concurrent temp-file races, event-loop stalls from blocking calls, retry backoff holding concurrency slots
- Image/conversion correctness: EXIF orientation, LA-mode transparency, uncompressed-image naming, EMF/WMF mislabeling, OCR engine config drift, temp-directory leaks
- Config/CLI correctness: config editor validates before save, symlink safety check fixed, `llm.concurrency` lower bound enforced, JSON log formatting, `config set` type coercion and bracket-notation support

### Changed

- Mixing an input path with a subcommand is now an error instead of silently dropping the input
- `-o out.md` on a single file/URL writes exactly that file
- Diagnostics moved to stderr so piped stdout output stays clean
- Output naming switched to an extension-replacement scheme (`sample.pdf` → `sample.md`) — **reverted in [Unreleased]**
- `image.stdout_persist` now defaults on
- Reports (`.markitai/reports/`) are batch-only by default
- Dependency refresh (litellm, opencv-python, playwright, instructor, and others)

### Security

- litellm supply-chain pin lifted (`>=1.83.0`) now that upstream has audited and signed releases

## [0.14.0] - 2026-03-25

### Added

- Steam News extractor; structural MathML-to-LaTeX conversion; LibreOffice functional (not just presence) check

### Fixed

- PDF math-content extraction fallback; BBCode XSS prevention in Steam content; flaky integration tests hardened

### Security

- litellm pinned to `<1.82.7` (supply-chain incident)

## [0.13.1] - 2026-03-23

### Added

- Config editor redesign: fuzzy search, scrollable list, in-place UI refresh
- Field descriptions added to 66 Pydantic config settings

### Fixed

- Esc-key support, bool-editor consistency, and Literal-type value preservation in the config editor

## [0.12.1] - 2026-03-22

### Added

- Inline terminal image display (Kitty/iTerm2) for stdout mode, backed by a content-addressed asset store
- Chinese user-journey documentation

### Fixed

- LLM errors now visible in quiet/stdout mode; Kitty protocol image format fix; `init` no longer generates duplicate provider entries

## [0.12.0] - 2026-03-20

### Added

- Native HTML extraction pipeline: resolver-based extraction, frontmatter builder, quality profiles, and structured extractors for GitHub Discussions, X threads, and YouTube
- `--static` and `--kreuzberg` CLI flags

### Changed

- HTML files now route through the native webextract pipeline by default

### Fixed

- URL stdout fallback, thread-safety for shared caches/semaphores, atomic config writes

## [0.11.2] - 2026-03-14

### Fixed

- Windows RAM detection for task sizing; lazy `~/.markitai/` directory creation (no side effects on read-only use); output/log dirs now default to `None` instead of a hardcoded path

## [0.11.1] - 2026-03-14

### Added

- Pure-mode option in the interactive wizard

### Fixed

- `--pure` no longer wrongly triggered vision/screenshot paths; lowered an over-aggressive "content too short" threshold

## [0.11.0] - 2026-03-13

### Added

- **`--pure` mode**: transparent LLM pass-through (text cleaning only, no frontmatter/post-processing), decoupled from `--llm`
- `--keep-base` to force writing the base `.md` alongside `.llm.md`

### Fixed

- URL processors now respect `--pure`/`--llm`/`--keep-base` consistently with file processing

## [0.10.0] - 2026-03-12

### Added

- Auto-detect LLM providers from environment variables and authenticated CLIs when no config exists

### Changed

- `-v` is now `--verbose` (was `--version`); `-V` is `--version`
- Cold-startup time cut via lazy imports (~5s → ~0.3s)

### Fixed

- Alt-text language now matches the document's language instead of defaulting to English

## [0.9.2] - 2026-03-11

### Fixed

- Copilot/Claude login now always uses inherited stdio (fixes credential storage failures); clearer error messages instead of opaque wrapped exceptions

## [0.9.1] - 2026-03-09

### Added

- `markitai doctor --suggest-extras` as the single source of truth for install-script extras

### Fixed

- Login-guard and extras-parsing bugs in the install scripts; Rich-markup escaping fix for provider names

## [0.9.0] - 2026-03-09

### Added

- Configurable global/per-domain fetch **strategy priority**, and `local_only_patterns`/`inherit_no_proxy` to restrict sensitive domains to local-only strategies

### Fixed

- LLM output no longer translates mixed-language page content into the wrong language

## [0.8.1] - 2026-03-06

### Added

- **Defuddle fetch strategy** (free, no auth) as a new top-priority option; `--defuddle` CLI flag

### Changed

- Default strategy order updated to lead with Defuddle/Jina

## [0.8.0] - 2026-03-06

### Added

- 20+ new file formats via markitdown/kreuzberg (HTML, CSV, EPUB, MSG, IPYNB, Numbers, TSV, XML, ODS, ODT, SVG, RTF, RST, ORG, TEX, EML); GIF/BMP/TIFF image support

### Fixed

- Claude Agent SDK compatibility bump; i18n test isolation; import-time log noise from kreuzberg registration

## [0.7.0] - 2026-03-05

### Added

- **ChatGPT provider** (`chatgpt/`) via OAuth device-code flow
- **Gemini CLI provider** (`gemini-cli/`) — later removed in 0.17.0
- `weight: 0` to explicitly disable a model in routing

### Fixed

- Router division-by-zero when all models were weight-0

## [0.6.1] - 2026-03-05

### Fixed

- Claude Agent SDK compliance fixes; auth pre-checks now recognize more environment-variable-based credentials

## [0.6.0] - 2026-03-04

### Added

- **Cloudflare integration**: Browser Rendering for URLs plus Workers AI `toMarkdown` for files
- Fetch Policy Engine with domain profiles and Playwright session persistence
- Pluggable static HTTP backend (`httpx`/`curl-cffi`)

### Fixed

- Router division-by-zero when all vision models were disabled; removed 21 dead functions across the codebase

## [0.5.2] - 2026-02-07

### Fixed

- SQLite connection leaks; Windows path-handling bugs; stale OAuth-expiry false positives; Pyright warnings cleared

## [0.5.1] - 2026-02-07

### Added

- Playwright auto-scroll for lazy-loaded content; DOM noise cleanup (nav/ads/cookie banners) before extraction; `python -m markitai` support

### Changed

- Default models modernized across `init`/interactive/doctor; cache fingerprint switched to a full-content hash (was a short prefix, prone to collisions)

## [0.5.0] - 2026-02-06

### Added

- **`markitai init`** setup wizard and **interactive mode** (`-I`); `doctor --fix` auto-install

### Changed

- CLI startup ~3x faster via lazy module loading; batch UI simplified to a compact progress display

### Fixed

- Windows LibreOffice/FFmpeg detection; Playwright default wait condition (was causing hangs)

## [0.4.2] - 2026-02-03

### Changed

- Playwright wait defaults tuned for better SPA support

### Fixed

- X/Twitter pages now wait for full JS rendering before capture; caches respect the configured directory instead of a hardcoded path

## [0.4.1] - 2026-02-02

### Added

- **`markitai doctor`** diagnostic command; adaptive timeout for local providers; prompt caching for long Claude Agent system prompts

## [0.4.0] - 2026-01-28

### Added

- **Claude Agent SDK** and **GitHub Copilot SDK** local providers; HTTP conditional caching (ETag/Last-Modified) for URLs; `--quiet`/`-q` flag

### Changed

- Major module reorganization (`cli/`, `llm/`, `providers/`)

## [0.3.2] - 2026-01-27

### Added

- Chinese README and setup scripts

## [0.3.1] - 2026-01-27

### Added

- **SPA domain learning**: auto-detect and cache JS-heavy sites to skip wasted static-fetch attempts
- Windows performance tuning (thread pool sizing, OCR engine singleton, faster image compression)

### Fixed

- Prompt-leakage prevention (system/user prompt split); auto-proxy detection for fetching

## [0.3.0] - 2026-01-26

### Added

- **Direct URL conversion** and `.urls` batch file support
- Multi-strategy fetching (`static`/`agent-browser`/`jina`/`auto`) with a SQLite fetch cache and screenshot capture
- `--no-cache-for <pattern>` selective cache bypass; `cache stats -v`
- Official VitePress documentation website (bilingual)
- MIT License; CI/CD workflows

## [0.2.4] - 2026-01-21

### Fixed

- Office/PPTX compatibility patches; symlink-safety hardening; LLM empty-response retry; frontmatter field ordering

## [0.2.3] - 2026-01-20

### Added

- **Persistent SQLite LLM cache** with LRU eviction; `cache stats`/`cache clear` commands
- Vision-aware model routing; parallel PDF/image processing

## [0.2.2] - 2026-01-20

### Added

- `constants.py` module consolidating hardcoded values; broader unit test coverage

## [0.2.1] - 2026-01-20

### Added

- Per-file LLM usage/cost tracking; typed usage/asset models; cross-platform Office/LibreOffice detection

### Changed

- File-conflict renaming switched to `.v2.md`-style natural sort order

## [0.2.0] - 2026-01-19

### Added

- **Monorepo rewrite**: uv workspace, LiteLLM-based provider access, new converter/workflow architecture, JSON-schema-validated config

### Breaking Changes

- New config format and CLI syntax; dropped support below Python 3.13; legacy `src/markitai/` architecture removed

## [0.1.6] - 2026-01-14

### Fixed

- Model routing bugs; documentation accuracy pass

## [0.1.5] - 2026-01-13

### Changed

- Prompt management and cleaner-module refactor

## [0.1.4] - 2026-01-13

### Fixed

- LLM JSON-parsing edge cases; log formatting

## [0.1.3] - 2026-01-12

### Changed

- Adopted `src` layout; added CI workflow

## [0.1.2] - 2026-01-12

### Added

- Network resilience (retry/timeout handling); AI-assistant docs (`CLAUDE.md`, `AGENTS.md`)

## [0.1.1] - 2026-01-11

### Changed

- Major architecture refactor to a service-layer pattern

## [0.1.0] - 2026-01-10

### Added

- Capability-based model routing, lazy provider initialization, concurrent fallback on timeout, `--fast` execution mode, per-model batch statistics

## [0.0.1] - 2026-01-08

### Added

- **Initial release**: CLI (`convert`/`batch`/`config`/`provider`), Office/PDF/HTML conversion, 5 LLM providers with fallback, image processing, batch processing with resume

[0.12.1]: https://github.com/Ynewtime/markitai/compare/v0.12.0...v0.12.1
[0.12.0]: https://github.com/Ynewtime/markitai/compare/v0.11.2...v0.12.0
[0.11.2]: https://github.com/Ynewtime/markitai/compare/v0.11.1...v0.11.2
[0.11.1]: https://github.com/Ynewtime/markitai/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/Ynewtime/markitai/compare/v0.10.0...v0.11.0
[0.10.0]: https://github.com/Ynewtime/markitai/compare/v0.9.2...v0.10.0
[0.9.2]: https://github.com/Ynewtime/markitai/compare/v0.9.1...v0.9.2
[0.9.1]: https://github.com/Ynewtime/markitai/compare/v0.9.0...v0.9.1
[0.9.0]: https://github.com/Ynewtime/markitai/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/Ynewtime/markitai/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/Ynewtime/markitai/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/Ynewtime/markitai/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/Ynewtime/markitai/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/Ynewtime/markitai/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/Ynewtime/markitai/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/Ynewtime/markitai/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/Ynewtime/markitai/compare/v0.4.2...v0.5.0
[0.4.2]: https://github.com/Ynewtime/markitai/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/Ynewtime/markitai/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/Ynewtime/markitai/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/Ynewtime/markitai/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/Ynewtime/markitai/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/Ynewtime/markitai/compare/v0.2.4...v0.3.0
[0.2.4]: https://github.com/Ynewtime/markitai/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/Ynewtime/markitai/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/Ynewtime/markitai/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/Ynewtime/markitai/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/Ynewtime/markitai/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/Ynewtime/markitai/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/Ynewtime/markitai/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/Ynewtime/markitai/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/Ynewtime/markitai/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/Ynewtime/markitai/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/Ynewtime/markitai/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/Ynewtime/markitai/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/Ynewtime/markitai/releases/tag/v0.0.1
