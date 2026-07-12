# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.21.0] - 2026-07-12

### Changed

- **Legacy `.xls` files now convert in pure Python** (xlrd via MarkItDown, same engine as `.xlsx`): no Office application or LibreOffice is launched or required, on any platform. Cell content is identical to the previous Excel-automation output (verified live); conversion is also faster and no longer serialized behind an app lock
- **The persistent LLM cache is now scoped to your model pool**: switching model configuration stops serving results produced by the old models, while rotating an API key keeps every hit; document cleanups also keep hitting after a file is renamed (identical content, different name). Existing cache entries miss once after upgrading — the old entries were unreachable anyway (see Fixed)
- **Structured LLM calls now go through the full transport retry loop** (exponential backoff, quota/billing short-circuit, empty-response retry) that plain-text calls always had; they previously bypassed it entirely. With a persistently degenerate model the worst-case attempt count is accordingly higher — that is the cost of actually retrying
- **Standalone `.urls` batches gain vision enhancement** in the `--screenshot` multi-source corner, matching what a single-URL conversion of the same page produces
- **Internal: the fetch, LLM, and CLI subsystems were restructured** behind unchanged behavior — machine-enforced import-layering contracts now run in CI, fetch.py shrank from ~3000 to ~1100 lines with per-strategy modules, the LLM pipeline is a single engine instead of five hand-rolled copies, and report/exit-code/output handling is single-sourced across all input paths. A 43-item regression matrix over every 0.15–0.20 fix verified nothing regressed

### Removed

- **Python 3.14 support**: `requires-python` narrows to `>=3.11,<3.14` across the package and installers, and the CI/publish matrices no longer test 3.14. The platform-conditional ONNX Runtime constraint that existed only to keep 3.14 resolvable is gone with it
- Excel COM automation (Windows) and Excel AppleScript automation (macOS), together with the `.xls` entries in the batch pre-conversion and heavy-task paths — `.xls` no longer needs them

### Fixed

- **The cross-session LLM cache never actually hit**: the persistent cache's reads and writes disagreed on the cache key's model column, so no run ever saw a previous run's results — every rerun of an already-processed document paid the full LLM cost again. Reads and writes now agree (locked by a regression test), and reprocessing an unchanged document is near-instant and free
- **macOS `.doc`/`.ppt` conversion via Microsoft Office was broken end to end**: Word's and PowerPoint's AppleScript `open` returns nothing, so the script's document binding was never established — the first reference died, the error handler's own reference to the same variable masked the real error behind an inscrutable "variable openedItem is not defined", the document was never closed, and the stranded zombie documents degraded the app further with every retry. The script now binds the document by its unique staged name, closes it under both its pre- and post-save names, and always propagates the original error. Excel's "Parameter error -50" had the same root and is fixed the same way (now moot for `.xls` — see Removed — but the fix also covers the PPTX-to-PDF export path)
- **macOS: `.doc`/`.ppt` conversion failing on an Office app's first scripted launch after an Office update** ("the document never registered" / PowerPoint error -9074). In that state the app silently drops Word's parametered `open` request while answering everything else; Word's script now retries with a plain `open` mid-poll (verified to penetrate and heal the state) and logs the recovery. Error messages for the stall and for -9074 now carry the verified remedy — open the app manually once so it finishes first-run setup — instead of the misleading "app stuck or overloaded, quit and retry"
- **Truncated or degenerate LLM output no longer poisons the cache**: screenshot extraction and URL enhancement now reject length-truncated responses like every other call site, and URL enhancement no longer caches output whose degenerate tail had to be cut
- **`markitai -I` no longer crashes with a raw traceback when a config file is invalid**: the interactive wizard now reports the same actionable message as every other command
- **Directory-batch reports no longer drop screenshot counts for URL entries**
- **The defuddle HTTP client now rebuilds when proxy or timeout settings change** (it previously kept the first client for the whole process, like jina always did correctly)
- **Disabled models (`weight: 0`) no longer affect cache scoping**: disabling a model stops serving its old cached results, and adding or removing a disabled entry no longer invalidates the pool's cache
- **Copilot authentication status reads correctly with current Copilot CLIs**: newer CLI versions write JSONC-style comments and renamed the login key, which made `doctor` and `auth copilot status` report a parse error as "not logged in". Both formats are now accepted, an unparseable config reports as indeterminate rather than unauthenticated, and a login the CLI itself reports as successful is never vetoed by the follow-up config read

## [0.20.0] - 2026-07-10

### Added

- **macOS can use installed Microsoft Office when LibreOffice is absent**: legacy `.doc`, `.ppt`, and `.xls` files are converted through Word, PowerPoint, or Excel via AppleScript, and PowerPoint can export PPTX slides to PDF for rendering. The fallback is enabled by `office.macos_fallback`, can be disabled for headless sessions, and reports one-time Automation permission requirements through `doctor` and the documentation

### Changed

- **Remote fetching keeps public URLs frictionless while tightening privacy**: public URLs may use remote fallbacks without confirmation, with one process-wide stderr notice covering defuddle.md, Jina, Cloudflare, FxTwitter, and Twitter oEmbed; private, local, DNS-resolved non-global, and credential-bearing URLs (including sensitive path tokens) remain local, while `MARKITAI_NO_REMOTE_FETCH=1` blocks every remote path, including an explicit remote `-s`
- **`doctor` reports capability health instead of package presence**: normal checks fail only for RapidOCR and configured workflows; Playwright is verified by launching Chromium, active model environment references are checked, and a requested `--fix` only installs and rechecks Chromium without changing project dependencies
- **`markitai init` now preserves existing configuration by default**: pressing Enter selects Keep, while Update and Overwrite remain explicit choices
- **Onboarding starts with the portable installer**: the homepage detects Windows versus macOS or Linux before first paint and recommends `setup.ps1` or `setup.sh`, keeps `uv tool install markitai` as a manual option, includes a 60-second no-LLM example, and improves Chinese navigation plus screen-reader and high-contrast support
- **Copilot pricing metadata recognizes `gpt-5.6-luna` when explicitly configured**: generally available models remain the automatic OpenAI and ChatGPT onboarding defaults while the limited-preview model stays opt-in

### Fixed

- **Quiet mode is consistent across single and batch work**: Markdown requested on stdout is preserved, errors stay on stderr, quiet dry runs omit previews, partial URL batches keep successful outputs and exit 10, and informational progress or success paths remain hidden
- **Image conversions with no enabled extraction path no longer report success**: a standalone image without `--ocr` or `--llm` now exits 1 with an actionable message instead of producing no output with a successful status
- **Installer reruns preserve intent**: the shell and PowerShell setup scripts skip `markitai init --yes` when `~/.markitai/config.json` already exists, preserve existing extras, and honor an explicit `MARKITAI_VERSION` even when Markitai is already installed
- **Homepage quick-start commands remain readable in light mode**: the dark command panel now consistently uses light text and transparent code backgrounds, stacks before commands become cramped, and serves a real `/favicon.ico` instead of returning 404
- **Python 3.14 dependency resolution avoids an incompatible ONNX Runtime pin**: platform-aware constraints keep Magika's Windows cap where required while allowing supported ONNX Runtime releases elsewhere

### Security

- **Configuration output hides secrets by default**: `markitai config list` recursively redacts secrets and custom header values, reduces `api_base` values to their origin, and only reveals original values when `--show-secrets` is explicitly passed
- **URL credentials stay local and out of diagnostics**: userinfo, sensitive path tokens, query strings, and fragments are removed from terminal errors, progress labels, dry-run previews, console and file logs, and generated output names; hostnames resolving to any non-global address cannot cross a remote dispatch boundary
- **macOS Office automation isolates untrusted documents**: the fallback disables macros and external-link updates while opening read-only staged copies, binds and closes only the exact document it opened, serializes app access across processes, and keeps recoverable staging files private
- **Headless setup never implies consent to optional software**: without a usable terminal, the portable installer installs only uv, Python, and Markitai unless `MARKITAI_INSTALL_OPTIONAL=1` explicitly enables optional packages, browser binaries, system dependencies, and third-party CLIs

## [0.19.0] - 2026-07-10

### Changed

- **Remote extraction no longer prompts by default** (`fetch.remote_consent` default `ask` → `always`): public URLs fall back to remote extraction services (defuddle.md, Jina, Cloudflare — tried one at a time, only after local strategies fail) without an interactive confirmation; the first use is disclosed via an INFO log. Private/local URLs never use remote services regardless of this setting, and URLs carrying credentials in the netloc (`user:pass@host`) are now treated as private too. Set `fetch.remote_consent=ask`/`never` or `MARKITAI_NO_REMOTE_FETCH=1` to restore prompting or disable remote services
- **Consent prompt rewording** (for `remote_consent=ask`): the prompt now explains why it appears (local extraction didn't succeed), that services are tried one at a time (first success wins), and dynamically lists only the services actually in the chain — Cloudflare (which runs against your own account credentials) only appears when configured. Interactive prompts also pause the live progress display instead of tearing it

### Added

- **Live progress checklist (StageList)**: multi-stage live progress for single-URL and single-file conversions — completed stages persist as `✓ Fetched via fxtwitter (2.1s)` lines, and the active stage shows a spinner with an elapsed-time suffix. Stdout-mode conversions (no `-o`) finally show progress; they were previously fully silent through fetch + LLM enhancement

### Fixed

- **`--resume` was a no-op**: the CLI batch entry point accepted the flag but always reprocessed every file from scratch. It now correctly loads saved state — completed files are skipped, failed/interrupted files are retried, newly-discovered files are picked up — and reports `Resuming batch: N completed, M remaining`
- **Output naming reverted to the append scheme**: `sample.pdf` → `sample.pdf.md` (not `sample.md`), undoing the 0.15.0 extension-replacement change, which hid the source format, mangled multi-suffix names, and made single-file and batch conversions of the same file disagree
- **Windows install one-liner 404**: the website now serves `setup.ps1` (docs pointed to https://markitai.dev/setup.ps1 but only setup.sh was deployed); Chinese changelog edits now trigger site redeploys
- **Prompt REMINDER leaked into cleaned output**: with smaller models (observed with `gpt-5.4-mini`), the vision-cleaning prompt's trailing `REMINDER: ...` instruction — and its `---` delimiter — could be echoed verbatim at the end of `.llm.md` output. The prompt now delimits the document with `<document>` tags and puts all instructions before the content, and a new output guard strips echoed prompt fragments, including from previously cached results
- **Image alt text was silently skipped for some URL conversions**: URLs with a screenshot but no multi-source content (e.g. X posts via site extractors) fell through to text-only LLM processing without image analysis, and the URL-batch path never analyzed images at all — `--alt`/`--desc` had no effect there. Both paths now analyze downloaded images (alt text + `images.json`). The stdout asset rewrite also no longer overwrites LLM-generated alt text with the bare filename
- **Batch image analysis no longer trips over bare-payload JSON**: small models sometimes answer a single-image batch with the bare item instead of the `{"images": [...]}` wrapper; this burned Instructor retries and fell back to per-image analysis with an ERROR log. The JSON repair layer now coerces such shapes in place, so the batch succeeds directly
- **Weak models could mangle social-post bodies during LLM cleanup** (flattened quoted-post blockquotes, respaced CJK text — observed with `claude-agent/haiku`): content profiled as `social_post` now passes its body through verbatim and the LLM only generates metadata. For all other document types, the document-processing prompt gained explicit blockquote-preservation and CJK-spacing rules
- **ChatGPT connection errors were non-retryable and unreadable**: httpx transport failures (connection reset/refused, timeouts) were mapped to a non-retryable `ProviderError` with an empty message, bypassing every retry layer. They are now marked retryable and carry the underlying error text
- **Console log lines no longer tear the live progress display**: log output now routes through the shared rich stderr console, so lines print above the StageList spinner instead of leaving stale frames behind; quiet/stdout mode also applies the same third-party retry-noise filter as normal mode (raw instructor retry errors previously leaked through)
- **Failed LLM enhancement is now visible in the output**: when every LLM path fails, the fallback `.llm.md` frontmatter carries `llm_enhanced: false` and an ERROR-level log is emitted — previously the only hint of degraded output was an empty description

### Removed

- **Legacy progress facilities retired**: internal `ConversionStatus`, `ProgressReporter`, and `OutputManager` are replaced by StageList; `markitai.utils`/`markitai.cli` no longer export `ProgressReporter`, and `attempt_login()` lost its unused `output_manager` parameter
- **Dead code sweep**: unreachable async enricher registry, unused exception hierarchy, deprecated no-caller helpers, test-only utility functions, and ~5MB of unreferenced test fixtures; `markitdown` dependency narrowed from `[all]` to the office extras actually used (drops azure/audio/pdfminer/youtube transitive deps); `httpx` and `lxml` are now declared directly

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
- Output naming switched to an extension-replacement scheme (`sample.pdf` → `sample.md`) — **reverted in 0.19.0**
- `image.stdout_persist` now defaults on
- Reports (`.markitai/reports/`) are batch-only by default
- Dependency refresh (litellm, opencv-python, playwright, instructor, and others)

### Security

- litellm supply-chain pin lifted (`>=1.83.0`) now that upstream has audited and signed releases

## [0.14.0] - 2026-03-25

- Added: Steam News extractor; structural MathML-to-LaTeX conversion; LibreOffice functional (not just presence) check
- Fixed: PDF math-content extraction fallback; BBCode XSS prevention in Steam content; flaky integration tests hardened
- Security: litellm pinned to `<1.82.7` (supply-chain incident)

## [0.13.1] - 2026-03-23

- Added: Config editor redesign: fuzzy search, scrollable list, in-place UI refresh
- Added: Field descriptions added to 66 Pydantic config settings
- Fixed: Esc-key support, bool-editor consistency, and Literal-type value preservation in the config editor

## [0.12.1] - 2026-03-22

- Added: Inline terminal image display (Kitty/iTerm2) for stdout mode, backed by a content-addressed asset store
- Added: Chinese user-journey documentation
- Fixed: LLM errors now visible in quiet/stdout mode; Kitty protocol image format fix; `init` no longer generates duplicate provider entries

## [0.12.0] - 2026-03-20

- Added: Native HTML extraction pipeline: resolver-based extraction, frontmatter builder, quality profiles, and structured extractors for GitHub Discussions, X threads, and YouTube
- Added: `--static` and `--kreuzberg` CLI flags
- Changed: HTML files now route through the native webextract pipeline by default
- Fixed: URL stdout fallback, thread-safety for shared caches/semaphores, atomic config writes

## [0.11.2] - 2026-03-14

- Fixed: Windows RAM detection for task sizing; lazy `~/.markitai/` directory creation (no side effects on read-only use); output/log dirs now default to `None` instead of a hardcoded path

## [0.11.1] - 2026-03-14

- Added: Pure-mode option in the interactive wizard
- Fixed: `--pure` no longer wrongly triggered vision/screenshot paths; lowered an over-aggressive "content too short" threshold

## [0.11.0] - 2026-03-13

- Added: **`--pure` mode**: transparent LLM pass-through (text cleaning only, no frontmatter/post-processing), decoupled from `--llm`
- Added: `--keep-base` to force writing the base `.md` alongside `.llm.md`
- Fixed: URL processors now respect `--pure`/`--llm`/`--keep-base` consistently with file processing

## [0.10.0] - 2026-03-12

- Added: Auto-detect LLM providers from environment variables and authenticated CLIs when no config exists
- Changed: `-v` is now `--verbose` (was `--version`); `-V` is `--version`
- Changed: Cold-startup time cut via lazy imports (~5s → ~0.3s)
- Fixed: Alt-text language now matches the document's language instead of defaulting to English

## [0.9.2] - 2026-03-11

- Fixed: Copilot/Claude login now always uses inherited stdio (fixes credential storage failures); clearer error messages instead of opaque wrapped exceptions

## [0.9.1] - 2026-03-09

- Added: `markitai doctor --suggest-extras` as the single source of truth for install-script extras
- Fixed: Login-guard and extras-parsing bugs in the install scripts; Rich-markup escaping fix for provider names

## [0.9.0] - 2026-03-09

- Added: Configurable global/per-domain fetch **strategy priority**, and `local_only_patterns`/`inherit_no_proxy` to restrict sensitive domains to local-only strategies
- Fixed: LLM output no longer translates mixed-language page content into the wrong language

## [0.8.1] - 2026-03-06

- Added: **Defuddle fetch strategy** (free, no auth) as a new top-priority option; `--defuddle` CLI flag
- Changed: Default strategy order updated to lead with Defuddle/Jina

## [0.8.0] - 2026-03-06

- Added: 20+ new file formats via markitdown/kreuzberg (HTML, CSV, EPUB, MSG, IPYNB, Numbers, TSV, XML, ODS, ODT, SVG, RTF, RST, ORG, TEX, EML); GIF/BMP/TIFF image support
- Fixed: Claude Agent SDK compatibility bump; i18n test isolation; import-time log noise from kreuzberg registration

## [0.7.0] - 2026-03-05

- Added: **ChatGPT provider** (`chatgpt/`) via OAuth device-code flow
- Added: **Gemini CLI provider** (`gemini-cli/`) — later removed in 0.17.0
- Added: `weight: 0` to explicitly disable a model in routing
- Fixed: Router division-by-zero when all models were weight-0

## [0.6.1] - 2026-03-05

- Fixed: Claude Agent SDK compliance fixes; auth pre-checks now recognize more environment-variable-based credentials

## [0.6.0] - 2026-03-04

- Added: **Cloudflare integration**: Browser Rendering for URLs plus Workers AI `toMarkdown` for files
- Added: Fetch Policy Engine with domain profiles and Playwright session persistence
- Added: Pluggable static HTTP backend (`httpx`/`curl-cffi`)
- Fixed: Router division-by-zero when all vision models were disabled; removed 21 dead functions across the codebase

## [0.5.2] - 2026-02-07

- Fixed: SQLite connection leaks; Windows path-handling bugs; stale OAuth-expiry false positives; Pyright warnings cleared

## [0.5.1] - 2026-02-07

- Added: Playwright auto-scroll for lazy-loaded content; DOM noise cleanup (nav/ads/cookie banners) before extraction; `python -m markitai` support
- Changed: Default models modernized across `init`/interactive/doctor; cache fingerprint switched to a full-content hash (was a short prefix, prone to collisions)

## [0.5.0] - 2026-02-06

- Added: **`markitai init`** setup wizard and **interactive mode** (`-I`); `doctor --fix` auto-install
- Changed: CLI startup ~3x faster via lazy module loading; batch UI simplified to a compact progress display
- Fixed: Windows LibreOffice/FFmpeg detection; Playwright default wait condition (was causing hangs)

## [0.4.2] - 2026-02-03

- Changed: Playwright wait defaults tuned for better SPA support
- Fixed: X/Twitter pages now wait for full JS rendering before capture; caches respect the configured directory instead of a hardcoded path

## [0.4.1] - 2026-02-02

- Added: **`markitai doctor`** diagnostic command; adaptive timeout for local providers; prompt caching for long Claude Agent system prompts

## [0.4.0] - 2026-01-28

- Added: **Claude Agent SDK** and **GitHub Copilot SDK** local providers; HTTP conditional caching (ETag/Last-Modified) for URLs; `--quiet`/`-q` flag
- Changed: Major module reorganization (`cli/`, `llm/`, `providers/`)

## [0.3.2] - 2026-01-27

- Added: Chinese README and setup scripts

## [0.3.1] - 2026-01-27

- Added: **SPA domain learning**: auto-detect and cache JS-heavy sites to skip wasted static-fetch attempts
- Added: Windows performance tuning (thread pool sizing, OCR engine singleton, faster image compression)
- Fixed: Prompt-leakage prevention (system/user prompt split); auto-proxy detection for fetching

## [0.3.0] - 2026-01-26

- Added: **Direct URL conversion** and `.urls` batch file support
- Added: Multi-strategy fetching (`static`/`agent-browser`/`jina`/`auto`) with a SQLite fetch cache and screenshot capture
- Added: `--no-cache-for <pattern>` selective cache bypass; `cache stats -v`
- Added: Official VitePress documentation website (bilingual)
- Added: MIT License; CI/CD workflows

## [0.2.4] - 2026-01-21

- Fixed: Office/PPTX compatibility patches; symlink-safety hardening; LLM empty-response retry; frontmatter field ordering

## [0.2.3] - 2026-01-20

- Added: **Persistent SQLite LLM cache** with LRU eviction; `cache stats`/`cache clear` commands
- Added: Vision-aware model routing; parallel PDF/image processing

## [0.2.2] - 2026-01-20

- Added: `constants.py` module consolidating hardcoded values; broader unit test coverage

## [0.2.1] - 2026-01-20

- Added: Per-file LLM usage/cost tracking; typed usage/asset models; cross-platform Office/LibreOffice detection
- Changed: File-conflict renaming switched to `.v2.md`-style natural sort order

## [0.2.0] - 2026-01-19

- Added: **Monorepo rewrite**: uv workspace, LiteLLM-based provider access, new converter/workflow architecture, JSON-schema-validated config
- Breaking: New config format and CLI syntax; dropped support below Python 3.13; legacy `src/markitai/` architecture removed

## [0.1.6] - 2026-01-14

- Fixed: Model routing bugs; documentation accuracy pass

## [0.1.5] - 2026-01-13

- Changed: Prompt management and cleaner-module refactor

## [0.1.4] - 2026-01-13

- Fixed: LLM JSON-parsing edge cases; log formatting

## [0.1.3] - 2026-01-12

- Changed: Adopted `src` layout; added CI workflow

## [0.1.2] - 2026-01-12

- Added: Network resilience (retry/timeout handling); AI-assistant docs (`CLAUDE.md`, `AGENTS.md`)

## [0.1.1] - 2026-01-11

- Changed: Major architecture refactor to a service-layer pattern

## [0.1.0] - 2026-01-10

- Added: Capability-based model routing, lazy provider initialization, concurrent fallback on timeout, `--fast` execution mode, per-model batch statistics

## [0.0.1] - 2026-01-08

- Added: **Initial release**: CLI (`convert`/`batch`/`config`/`provider`), Office/PDF/HTML conversion, 5 LLM providers with fallback, image processing, batch processing with resume
