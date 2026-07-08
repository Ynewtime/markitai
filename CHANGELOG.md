# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Removed

- **Gemini CLI provider** (`gemini-cli/`): removed entirely, along with `markitai auth gemini`, the `gemini-cli` install extra, and its `google-auth`/`google-auth-oauthlib` dependencies. Google retired individual-tier Gemini Code Assist onboarding for new projects ("This client is no longer supported... migrate to the Antigravity suite of products"), so fresh logins could no longer complete. Use a direct Google Gemini API key (`gemini/<model>`) or route through OpenRouter (`openrouter/google/<model>`) instead

### Added

- **`COPILOT_GITHUB_TOKEN` auth support**: Copilot auth detection and login hints now check `COPILOT_GITHUB_TOKEN` first (matching `copilot login --help`'s own documented precedence), falling back to `GH_TOKEN` then `GITHUB_TOKEN`

### Fixed

- **Console log noise**: `instructor`'s internal per-attempt retry logs (pure duplicates of markitai's own `[LLM:...] Failed:` summaries) are no longer echoed to the console (file logs keep full detail); third-party log origin resolution in the loguru interceptor was fixed so third-party INFO filtering works as intended; pydantic `ValidationError`s now collapse to a one-line summary naming the model and fields instead of dumping the raw multi-line block. Per-call LLM timing summaries are now visible with `-v`

## [0.16.0] - 2026-07-07

### Added

- **Bilibili opus extractor**: site-specific extractor for `bilibili.com/opus/<id>` (专栏/动态) posts, scoped to the `.bili-opus-view` card and keeping only the title/author/content modules — generic extraction previously pulled the login prompt, stat/share sidebar, and an internal content-ID tag in as if they were article text
- **Anti-bot/CAPTCHA challenge detection**: `_is_invalid_content()` recognizes Geetest, Cloudflare, reCAPTCHA, and hCaptcha challenge pages; an explicitly-chosen fetch strategy (e.g. `-s playwright`) now raises a clear error instead of silently returning the challenge page as if it were real content (the `auto` chain already fell back per-strategy on this)

### Changed

- **X/Twitter extraction is DOM-first again**: tweets and articles render through Playwright's DOM extractor first (rebuilt for X's 2026 redesign in 0.15.0, so it's the higher-fidelity path) and only fall back to the FxTwitter→oEmbed enricher when native extraction comes up short — replacing the previous design that always tried the FxTwitter API before launching a browser. `fetch_fxtwitter.py` is retired; its logic lives in `webextract/enrichers/x_oembed.py`'s `XOEmbedEnricher`, reached from one shared fallback path instead of two separately-maintained intercepts

### Fixed

- **X Article URL matching**: `XArticleExtractor` only matched the legacy `x.com/i/articles/<id>` system path; the common `x.com/<user>/article/<id>` form (singular — confirmed against defuddle's reference extractor) fell through to generic extraction and returned the login-wall page instead of content
- **X Article fetch performance**: article pages are 100% login-walled for anonymous visitors, but fetching one still paid for a full browser launch, a fixed post-load wait, and auto-scroll before falling back to the enricher; article URLs now skip straight to the enricher when no screenshot is requested — `-s playwright` on an article drops from 5-10s+ to ~1.7-1.9s
- **X Article frontmatter**: `word_count`/`content_profile` were computed from the discarded login-wall page (`word_count: 2`) rather than the enriched article content

## [0.15.0] - 2026-07-04

Maintenance overhaul: full dependency refresh, Python 3.14 support, and a
multi-dimension audit that fixed 30+ verified bugs across batch processing,
fetch/cache, LLM providers, image handling, and configuration.

### Added

- **Python 3.14 Support**: `requires-python` relaxed to `<3.15`; full test suite passes on 3.14 (the previous onnxruntime blocker is resolved). CI matrix and classifiers updated
- **MIT License**: LICENSE file added and declared in package metadata (`License-Expression: MIT`)
- **Grouped `--help`**: options are organized into panels (Output & Configuration / LLM Enhancement / OCR / Fetch & Conversion Backends / Batch Processing / Cache & Images / Logging & Info) via rich-click; lazy subcommand loading preserved so `--help` stays ~100ms
- **Garbled-text detection**: PDFs whose extracted text is unreadable (broken cmap/substitution ciphers) are detected via a CJK-safe vowel-ratio heuristic
- **Scan/garbled advisory**: converting a PDF with scanned-looking or garbled pages without `--ocr` now emits one consolidated warning naming the affected pages and suggesting `--ocr`
- **Repeated header/footer suppression**: running headers/footers (incl. "Page N of M" patterns) repeated across ≥60% of pages are stripped from PDF output — cleaner Markdown, fewer wasted LLM tokens; headings and tables are never touched, <4-page documents exempt
- **VLM degeneration guard**: vision/screenshot extraction results are checked for repetition loops (a known VLM-OCR failure mode); degenerate tails are truncated with a warning and never persisted to cache, so retries aren't poisoned
- **HTML extraction quality (ported from Defuddle upstream)**: MathJax `script[type="math/tex"]` equations preserved as LaTeX; Wikipedia/MediaWiki MathML survives hidden-element removal; partial-selector clutter removal no longer deletes code blocks (`<pre>`-protection); anchor-wrapped headings unwrap cleanly; code-block language tags validated against an allowlist (no more ```codeblock); whitespace inside `<pre>` preserved
- **Footnote engine (full Defuddle port)**: footnotes/citations across Wikipedia, arXiv, Substack, WordPress, Word/Google Docs exports, Tufte sidenotes and more are standardized and emitted as real Markdown footnotes (`[^1]` / `[^1]: ...`) with renumbering, duplicate-reference handling, multi-paragraph definitions, and back-link stripping — 15 ground-truth fixtures now match Defuddle's expected output
- **Unified fetch strategy flag**: new `-s/--strategy auto|static|playwright|defuddle|jina|cloudflare`; the five per-backend flags (`--playwright`, `--defuddle`, `--static`, `--jina`, `--cloudflare`) remain as deprecated aliases that print a migration notice
- **Remote-fetch consent**: URLs are no longer sent to third-party extraction services (defuddle.md, Jina, Cloudflare) without consent — `fetch.remote_consent: ask|always|never` (default `ask`: interactive runs prompt once per process; non-interactive/quiet runs skip remote and crawl locally); `MARKITAI_NO_REMOTE_FETCH=1` forces `never`; explicit `-s defuddle`/`-s jina` counts as consent
- **PDF hidden-text sanitization**: invisible text (white-on-white, <2pt, zero-opacity, off-page) — a prompt-injection vector for LLM pipelines — is detected; `security.pdf_sanitize: off|warn|remove` (default `warn` logs a consolidated advisory naming pages)
- **Per-page OCR routing**: `--ocr` on mixed digital/scanned documents keeps the native text layer for digital pages and only OCRs scanned/garbled ones (`ocr.per_page_routing`, default on)
- **Conversion-quality benchmark harness**: dev-only `packages/markitai/benchmarks/` scores the HTML pipeline against the Defuddle ground-truth corpus (rapidfuzz block alignment + order score, marker-style); committed baseline: mean 91.04 over 83 fixtures
- **Release automation**: release-please drives versioning/changelog from conventional commits (release = merge the release PR; publishing chains via workflow dispatch); PR coverage comments via py-cov-action, no external service. Operational note: tag the 0.15.0 release manually first (or set `bootstrap-sha`) so release-please anchors correctly

#### Hunt round 5 (tweet pipeline root fix & flow polish)

- **FxTwitter now actually serves default tweet fetches**: the intercept only existed in the top-level PLAYWRIGHT dispatch branch — the default auto chain reached playwright through its own loop and skipped FxTwitter entirely, dropping users onto noisy DOM extraction. Fixed in the chain (with telemetry/return contract); regression tests pin the intercept
- **X DOM extractor rebuilt for X's 2026 redesign**: X removed every `data-testid` and moved to React+Tailwind markup, so the semantic tweet extractor never matched and the generic pipeline leaked avatars, cookie banners, stats bars, and blob: video links. The extractor now handles both the new (`data-tweet-id`, hover-card slots, permalink-text timestamps) and legacy markup, verified against a committed real-DOM fixture and live; the x.com domain profile's wait selector no longer burns a 10s timeout per tweet
- **"All fetch strategies failed" is never empty**: five silent skip paths (JS-detected static, consent-gated remote, missing playwright/browser, missing CF credentials) now record their reason; an all-skipped chain explains itself
- **`markitai init` merges instead of dead-ending**: with an existing config the wizard offers Update (default) / Overwrite / Keep — Update non-destructively appends newly detected providers; `init -y` applies it automatically and reports what changed. Auth login hints are config-aware ("adds it to your existing config" / "Already enabled in `<path>`"). Dependency + provider detection now run concurrently under a spinner (Gemini's userinfo call, up to 5s, no longer serializes the flow)
- **Login failure cards**: provider login failures render the status-card style with context-aware hints (never suggesting the command that just failed; install command first); gemini login no longer dumps a raw traceback
- **`mkai` PyPI alias stub dropped**: PyPI's name-similarity guard rejects `mkai` (existing `mk-ai` project) — the same guard blocks would-be squatters, which was the stub's purpose; the `mkai` command itself still ships with markitai

#### Hunt round 4 (UX & quality polish)

- **Tweet conversion at defuddle parity**: the FxTwitter path (which serves default x.com fetches, with playwright as fallback) and the DOM extractor were both reworked — bold `**Name @handle** · date` author line, paragraphs preserved, t.co links expanded, video rendered as poster + link (was a broken mp4 embed), quoted tweets as blockquotes with author/date/media/permalink, author threads joined into the post body, card previews. Corpus mean 91.04 → 92.26
- **Live progress feedback**: long single-input conversions show a pure-ASCII stage-aware spinner on stderr (`Fetching (static)…` → `Rendering (playwright)…` → `Enhancing with LLM…`, bridged from fetch-stage logs); suppressed for pipes/--quiet/-v; stdout stays pure. Root cause of the "looks stuck" complaint: the old spinner machinery was constructed disabled in file-output mode
- **`markitai auth` status cards**: all four providers render a unified glyph card (✓/✗ login state, CLI/SDK presence, usage + next-step); bare `markitai auth` shows an all-provider overview; ChatGPT guidance now points at `markitai auth chatgpt login` (device-code flow verified live) instead of "pip install litellm"
- **Fetch errors are never blank**: exceptions with empty messages (e.g. httpx.ConnectError) now render their type via format_error_message across all fetch strategies

#### Hunt round 3 (release prep)

- **`.eml` email support (native, zero deps)**: headers/body/attachments via stdlib `email`; HTML bodies go through the standard HTML pipeline, image attachments flow into the assets/vision pipeline, nested messages render quoted (depth 1); header values sanitized against injection. EML no longer delegates to kreuzberg
- **HEIC/HEIF/AVIF input**: new `markitai[heif]` extra (pillow-heif); 12-byte ftyp sniff, decode-to-PNG at the boundary with EXIF orientation applied, then the normal OCR/vision/compression pipeline — iPhone photos just work
- **Quality guardrails gate**: `benchmarks/guardrails.json` pins a per-fixture minimum score (0.9 × current) plus corpus/local mean floors; `--check` fails CI (new ~1min job) when extraction quality regresses; `--update-guardrails` regenerates deliberately
- **`--config-json '<json>'`**: inline config overrides for agents/CI — merged over the config file, under explicit CLI flags
- **Subcommand help polish**: all 26 subcommand helps now render rich panels with Examples; empty states got helpful hints (`cache stats` with no cache, silent `init -y`, `config get` unknown key → list hint); action commands print the natural next step
- **`mkai` short command**: ships with markitai as a second console script. (A separate PyPI alias package was evaluated and dropped: PyPI's name-similarity guard rejects `mkai` because `mk-ai` exists — the same guard equally prevents anyone else from squatting the name)
- **kreuzberg floor raised to >=4.9.6**: picks up the no-OCR-backend PDF fix (4.7.3), image-heavy-PDF hang fixes (4.9.x), PPTX slide-order fix (4.8.0). Note: kreuzberg >=4.8.0 is Elastic License 2.0 (optional extra; accepted)
- **Cache-hit visibility**: the `Fetched via <strategy>` line notes `(cached)` — a cached defuddle result had masqueraded as the live default strategy (fresh fetches win with `static`)
- Dependency patches: litellm 1.90.3, rapidocr 3.9.1

#### Hunt round 2 (follow-up fixes)

- **Playwright browser detection fixed**: newer Playwright ships `Google Chrome for Testing.app` instead of `Chromium.app`, so the path-only check reported "browser not found" even after a successful install; detection now uses Playwright's own `INSTALLATION_COMPLETE` marker (bundle-name/version agnostic) with executable paths as fallback. The `uv tool run --from 'markitai[all]'` install hint was dropped (triggers a uv warning and resolves an ephemeral env whose Playwright version can mismatch)
- **Remote-fetch consent is now lazy**: the consent prompt fired before the chain ran, so even URLs satisfied by the local-first chain asked "Allow sending URLs to remote services?"; consent is now resolved only when a remote strategy is actually about to run — local successes never prompt
- **Config filename unified in all messages**: hints/errors that said "in markitai.json" (doctor's LLM hint, the Cloudflare workflow error) now show the actually-loaded config path, matching the doctor header
- **`-h` paragraph spacing normalized**: rich-click renders `\b` blocks with two trailing blank lines but plain paragraphs with none; docstring now renders with exactly one blank line between all sections (`TEXT_PARAGRAPH_LINEBREAKS`)
- **Dependency patch bumps**: litellm 1.90.3, rapidocr 3.9.1

#### CLI & fetch polish (post-release hunt round)

- **`mkai` short alias**: installed alongside `markitai` (verified conflict-free on PyPI/homebrew/system)
- **`-b/--backend native|kreuzberg|cloudflare`**: file conversion backend is now its own orthogonal flag; `--kreuzberg` remains as a deprecated alias. `-s` is purely the URL fetch strategy
- **Local-first auto chain**: default order is now `static → playwright → defuddle → jina → cloudflare` (static's native extraction matches remote defuddle on the ground-truth corpus and beats it on CJK spacing); SPA/JS-heavy domains go straight to the browser
- **`markitai doctor` first run 36s → ~5s, warm 1.0s → 0.33s**: two root causes — (a) the RapidOCR check imported the real module, pulling in opencv's 119MB dylib whose one-time macOS dyld signature validation cost ~25s on a fresh install (now probed via package metadata only, no import); (b) an unconditional litellm import cost 0.55s even with no models configured (now deferred). Output normalized (consistent inline item format, single-blank-line sections, failure summary uses ✗ not ✓) and now shows the loaded config file path
- **Actionable configuration errors**: Cloudflare credentials, Playwright missing-browser, and Jina auth errors now include the concrete config file path, copy-pasteable `markitai config set`/env commands, and credential acquisition steps (token URL + required permissions)
- **Jina refusal fallback**: service refusals (e.g. github.com 451 anonymous block) no longer dump raw JSON — interactive runs are asked once per run whether to fall back to the auto chain (default yes); non-interactive runs fall back automatically with a warning
- **Claude subscription detection fixed on macOS**: Claude Code stores OAuth tokens in the Keychain, not `~/.claude/.credentials.json` — markitai reported "not authenticated" on logged-in machines and `init` silently skipped claude-agent. `claude-agent/` models use the Claude subscription quota via the local CLI (no API key needed); `markitai auth claude status` now shows identity, plan, CLI/SDK state and a config snippet
- **stdout no longer hard-wraps content**: Rich's console.print wrapped output at terminal width, breaking long URLs mid-token in piped output; content is now written raw
- **Help panels aligned**: metavar column removed (appended to help text instead); deprecated aliases use terse uniform descriptions
- **HTML extraction**: GitHub repo pages now extract just the README (was: file-tree tables, About sidebar, star counts — ~950 junk words); frontmatter gains `published` date, full untruncated titles without site suffixes, and no longer emits homepage canonical_url on article pages; bilibili/Twitter-widget iframes survive as links (root cause: embed canonicalization ran after sanitize stripped iframes). Benchmark mean 91.86 → 92.24, embedded-videos fixture +31 points, zero regressions; local self-baseline fixtures (GitHub repo + CJK blog) added to the benchmark
- **Xberg (kreuzberg successor)**: evaluated — PyPI `xberg` is currently a placeholder aliasing kreuzberg and real 1.0 wheels aren't published; kreuzberg v4 stays (LTS, maintained), with a documented migration checklist in the converter for when Xberg 1.0 ships

### Changed (behavior)

- **Mixing INPUT with a subcommand is now an error**: `markitai note.txt config list` previously dropped note.txt silently; it now fails with guidance. A file named like a subcommand (`config`, `doctor`, ...) gets a stderr hint to use `./config`
- **`-o out.md` with a single file/URL writes that file**: previously it silently created a directory named `out.md`; batch/directory input with a file-looking `-o` now errors clearly
- **Diagnostics go to stderr**: warnings/notices no longer pollute piped stdout output (`markitai x --alt | pandoc` receives pure markdown)
- **Output naming replaces the extension**: `sample.pdf` → `sample.md` (was `sample.pdf.md`). Colliding batch inputs (`a.pdf` + `a.docx`) and outputs that would overwrite the source keep the legacy `<name>.<ext>.md` scheme, per file. Re-running an old batch re-converts once under the new names
- **stdout image links now survive the process**: `image.stdout_persist` defaults to on (assets persisted under `~/.markitai/assets`); absolute temp-dir links from the PDF pipeline are normalized; opting out prints an ephemeral-links warning to stderr
- **Reports are batch-only by default**: single-file/single-URL conversions no longer write `.markitai/reports/` unless `output.report = true` (tri-state; `false` disables even for batches)
- **Repo hygiene**: AI-session artifacts (`.claude/` memory, `docs/superpowers/` working plans containing local dev paths) removed from the repository and gitignored

### Fixed

#### Batch & Workflow (correctness of success/failure reporting)

- **LLM failures no longer masquerade as success**: LLM API failures previously returned success-shaped results — batch marked files COMPLETED with `cache_hit=true` pointing at `.llm.md` files that were never written, and `--resume` skipped them. Failures now propagate, the file is marked FAILED, and the base-markdown fallback path actually runs
- **Resume re-processes interrupted files**: files that were IN_PROGRESS at crash time were silently dropped on `--resume` (never re-queued, counted in no summary bucket); they are now converted to FAILED on state load and re-processed
- **Per-file LLM cost attribution**: usage contexts are cleared after each file, fixing double-counted costs for same-basename files in different directories
- **Base64 image index desync**: an undecodable data URI shifted every subsequent image reference one position, attaching wrong images to wrong locations; extraction and replacement now apply the same skip rule
- **Path traversal via custom output names**: a `.urls` entry with a crafted output name (`../../x` or absolute path) could write converted output outside the output directory; custom names are now sanitized like auto-derived ones

#### Fetch & Cache

- **AUTO-strategy cache revalidation**: HTTP validators (ETag/Last-Modified) were discarded on the default fetch path, so cached pages were served stale forever; validators are now stored and conditional revalidation works as designed
- **Playwright context leaks**: cookie-validation errors leaked browser contexts; a `new_page()` failure in persistent mode raised `UnboundLocalError` masking the real error; concurrent same-domain fetches could double-create or double-close cached contexts (now lock-protected)
- **HTTP client cleanup**: old clients are no longer closed via unreferenced fire-and-forget tasks that asyncio could garbage-collect before running
- **URL list robustness**: a `null` or non-string `url` in a JSON URL list crashed the whole batch; it is now skipped with a warning
- **Proxy auto-detection**: SOCKS-only ports (Tor 9050, SOCKS5 1080, V2Ray 10808) were mislabeled as HTTP proxies and are removed from detection; detected proxies now log at WARNING

#### LLM & Providers

- **Batch vision deadlock**: language rewrites re-acquired the already-held concurrency semaphore — with `llm.concurrency=1` a single rewrite hung the whole pipeline; rewrites now run after the semaphore is released
- **Vision cache poisoning**: batch results were zipped positionally with requested images; a skipped/reordered model response persisted the wrong analysis under the wrong image's content hash across sessions. Results are now aligned by the echoed `image_index`, and ambiguous batches skip cache persistence
- **Copilot concurrent temp-file races**: the singleton provider's shared temp-file list let one request's cleanup delete another in-flight request's image attachments; tracking is now per-request
- **gemini-cli rate-limit failover**: a 429 raised a non-retryable error that aborted the request instead of retrying on another pool model; it now raises litellm's retryable `RateLimitError` (cooldown still recorded)
- **Event-loop stalls**: blocking token refreshes (ChatGPT/gemini-cli auth) and CPU-heavy native HTML extraction now run in worker threads instead of freezing all concurrent tasks
- **Retry backoff released**: exponential-backoff sleeps no longer hold a concurrency-semaphore slot, so a rate-limit burst can't collapse throughput of healthy models
- **Dynamic max_tokens**: the retry path now takes the minimum output limit across the model pool (matching instructor call sites) instead of the top-weight model only
- **Screenshot extraction cache**: keyed by content fingerprint instead of filename, so re-fetches of changed pages aren't served stale extractions

#### Images & Conversion

- **EXIF orientation**: rotation is baked in before re-encoding on all compression paths (OpenCV and Pillow); phone photos no longer come out sideways
- **LA-mode transparency**: grayscale+alpha images composite onto white when converting to JPEG instead of dropping the alpha channel
- **Uncompressed image naming**: with `image.compress = false`, original bytes are now saved under their actual format's extension/MIME instead of the configured output format's
- **EMF/WMF conversion failures**: unconverted EMF bytes are no longer mislabeled as PNG; failures now log a visible warning
- **OCR engine consistency**: a failed engine rebuild no longer leaves the old engine permanently served under the new config's fingerprint
- **Temp directory leaks**: converter paths that render page/slide images without an output directory now clean up their temp directories at process exit

#### Configuration & CLI

- **Config editor validation**: `markitai config edit` validated nothing before saving — an out-of-range value bricked every subsequent CLI invocation (including the editor itself). The editor now validates before save, and config loading reports a clear actionable error instead of a raw traceback
- **Symlink safety check**: the nested-symlink branch inspected the resolved path (which by definition has no symlinks) and never fired; it now walks the original path's ancestors
- **Config bounds**: `llm.concurrency` requires `>=1` (a persisted `0` hung every LLM task forever); router retry/timeout fields gained sane lower bounds
- **`config set` type coercion**: values are coerced by the target field's declared type — string fields keep leading zeros (API keys), bool fields accept `1`/`0`
- **`config set` bracket notation**: `llm.model_list[0].litellm_params.weight` now works for set (previously only get)
- **JSON log format**: log lines are built with proper JSON serialization; messages containing quotes/newlines no longer produce invalid JSON
- **`cache clear` prompt**: shows the actual configured cache directory instead of a hardcoded path
- **`config get` null handling**: existing-but-null fields print `null` and exit 0 instead of "Key not found" exit 1
- **Missing config path visibility**: a nonexistent `MARKITAI_CONFIG`/explicit config path now warns (and `~` is expanded) instead of silently running with defaults
- **Loguru misuse**: printf-style logging calls that silently dropped the URL and traceback now use loguru idioms

#### Post-review hardening

- **Parallel LLM task isolation**: a document-processing failure no longer leaves the sibling image-analysis task running detached (and vice versa); image-analysis failure now degrades gracefully (`.llm.md` kept without alt text) instead of failing the whole file
- **Usage cleanup on vision fallback paths**: partial LLM usage is cleared when vision enhancement fails, so it isn't attributed to the next file
- **`markitai doctor` exit code**: exits 1 when required dependencies are missing (previously always 0, and the summary claimed success); failure summary now reports the missing count
- **Symlink check refinement**: root-owned symlinks on POSIX (e.g. `/var/run -> /run`) are treated as OS artifacts instead of raising false positives

### Changed

- **Dependency refresh**: all dependencies upgraded ~4 months forward, including litellm 1.82.6 → 1.90.x, opencv-python 4.x → 5.x, starlette → 1.x, claude-agent-sdk 0.1 → 0.2, github-copilot-sdk 0.2 → 1.x, instructor 1.15, playwright 1.61, pymupdf 1.28. Test suite fully green on the new set. (markitdown stays at 0.1.5 — 0.1.6 requires a pre-release azure dependency; rich stays at 14.x — capped by instructor `<15`)
- **Version single-sourcing**: the package version is now read from `src/markitai/__init__.py` at build time (`dynamic = ["version"]`); no more triple-bump
- **Release guard**: `publish.yml` verifies the release tag matches the built version before publishing
- **Dependabot on uv ecosystem**: lockfile-aware dependency PRs (previously the pip ecosystem produced PRs that always failed `uv sync --frozen`)
- **README**: rewritten with install instructions (uv tool/pipx), extras table, and quick start — this is also the PyPI long description
- **CONTRIBUTING.md**: new contributor guide (dev setup, commands, conventions, release steps)
- **pre-commit**: pyright moved from per-commit to pre-push (full-project check was tens of seconds per commit)
- **`.env.example`**: bilingual (EN/zh) comments
- **bs4 4.15 compatibility**: attrs-only `find`/`find_all` calls pass an explicit tag matcher; `NavigableString` imported from `bs4.element` (upstream `__all__` regression)
- **Ruff target aligned to floor**: `target-version = "py311"` (was py313, which could suggest syntax breaking 3.11 support)
- **Modernized asyncio idioms**: `asyncio.get_event_loop()` → `asyncio.get_running_loop()` in async code

### Security

- **litellm supply-chain pin lifted**: `litellm>=1.83.0` replaces the `<1.82.7` emergency pin — the March 2026 compromise affected only 1.82.7/1.82.8, upstream audited 1.78.0–1.82.6 clean, and releases are signed since 1.83.0

## [0.14.0] - 2026-03-25

### Added

- **Steam News Extractor**: Site-specific extractor for `store.steampowered.com/news/` pages that parses BBCode announcements from JSON data attributes
- **MathML-to-LaTeX Converter**: Structural MathML conversion for pages without LaTeX annotations (KaTeX/MathJax), handling `msup`, `msub`, `mfrac`, `msqrt`, `mover`, `munder`, `mtable`, and 70+ Unicode math symbol replacements
- **LibreOffice Functional Check**: `is_libreoffice_functional()` verifies LibreOffice can actually convert files, not just that the binary exists
- **CSS Modules Hidden Detection**: Detect hashed hidden class names like `isHidden-vzcyV0` from CSS-in-JS frameworks

### Fixed

- **Math Content Extraction**: Body fallback now triggers when all retry levels fail to reach the sparse threshold, fixing KaTeX pages where scoring selected a single math div instead of the full article
- **Integration Test Reliability**: Batch test fixture filters to files with registered converters; LibreOffice tests skip properly when installation is non-functional
- **CLI Preset Validation**: Unknown presets now show available options and exit with error instead of silently continuing
- **BBCode XSS Prevention**: Raw HTML in Steam BBCode content is escaped before conversion to prevent injection

### Security

- **litellm Supply-Chain Pin**: Pin litellm to `<1.82.7` to exclude compromised versions

### Changed

- **CI Resilience**: Windows LibreOffice install retries up to 3 times with backoff to handle transient Chocolatey failures

## [0.13.1] - 2026-03-23

### Added

- **Config Editor Redesign**: Replace questionary select with a custom prompt_toolkit UI featuring a visible search box with frame, fuzzy filtering, scrollable list with cursor, and "↑ N more above / ↓ N more below" scroll indicators
- **Fuzzy Match Search**: Case-insensitive fuzzy matching for config settings (characters in order, not necessarily consecutive) with scoring that rewards consecutive and early matches
- **Config Field Descriptions**: Add `Field(description=...)` to 66 Pydantic config fields, displayed inline in the config editor
- **In-Place UI Refresh**: Use ANSI cursor position queries to erase only the lines occupied by each UI component, preserving terminal history

### Fixed

- **Esc Key Support**: Inject Esc key bindings into all questionary prompts (text, select) via prompt_toolkit `merge_key_bindings`; questionary 2.1.1 `select()` only binds Ctrl+C/Ctrl+Q natively
- **Bool Editor**: Replace `questionary.confirm()` with `questionary.select()` using `Choice(value=True/False)` for consistent Esc support
- **Search + j/k Conflict**: Disable `use_jk_keys` when `use_search_filter` is enabled (questionary 2.1.1 raises `ValueError` otherwise)
- **Literal Type Preservation**: Use `Choice(value=original)` to preserve original typed values (int, str) when editing Literal fields, instead of converting to string

## [0.12.1] - 2026-03-22

### Added

- **Stdout Terminal Image Display**: Inline image rendering for Kitty/iTerm2 terminals in stdout mode, with three-tier resolution cascade (terminal protocol → persistent asset store → markdown placeholder)
- **Content-Addressed Asset Store**: Persistent image storage with symlink refs at `~/.markitai/assets/`, enabling stdout image persistence across sessions
- **Terminal Image Protocol Detection**: Auto-detect Kitty and iTerm2 graphics protocols for native inline image display
- **`stdout_persist` Config Fields**: New `image.stdout_persist` and `image.stdout_persist_dir` settings for controlling stdout image persistence
- **External Image Inline Display**: Download and inline-display external images in single URL stdout mode (`image.stdout_fetch_external`)
- **User Journey Documentation**: Comprehensive Chinese user journey document covering all features and workflows

### Fixed

- **Stdout Mode LLM Errors**: Make LLM errors visible in quiet/stdout mode via ERROR-level log handler
- **LLM Warning Implementation**: Address third-party review findings on LLM warning display
- **Kitty Graphics Protocol**: Convert images to PNG for Kitty protocol compatibility
- **Stdout Image Handling**: Resolve three bugs in stdout image asset resolution and display
- **Cross-Platform Tests**: Fix Windows test failures and missing Playwright browser handling
- **`markitai init` Duplicate Routes**: Deduplicate overlapping default provider entries in generated configs, preferring Claude CLI over Anthropic API and direct Gemini API over OpenRouter Gemini

### Changed

- **Stdout Asset Resolution**: Rename `strip_asset_references` to `resolve_asset_references` with three-tier cascade logic
- **Terminal Image Rendering**: Harden rendering pipeline and improve test coverage
- **`markitai init` Default Config**: Stop writing redundant default `image.compress` and `image.quality` settings into newly generated configs

## [0.12.0] - 2026-03-20

### Added

- **Native HTML Extraction Parity**: Introduce resolver-based extraction pipeline with typed extraction results, frontmatter builder, quality profiles, and semantic models for threaded pages
- **Structured Site Extractors**: Rebuild threaded extraction on shared abstractions and add native resolver coverage for GitHub Discussions, X threads, and YouTube pages
- **Webextract Quality Enhancements**: Add noise removal, enhanced scoring, standardization, multi-level retry, content patterns, heading anchors, callouts, srcset optimization, and code language detection
- **CLI Force Flags**: Add `--static` to force static HTTP with native webextract and `--kreuzberg` to force kreuzberg conversion for all formats
- **Async Enrichment Pipeline**: Add policy-aware enrichers and thread inclusion rules for structured extraction
- **Language-Aware Vision Retry**: Retry and rewrite image analysis outputs in the document language

### Fixed

- **URL Stdout Fallback**: URL mode without `-o` now writes to stdout instead of erroring
- **Concurrency Safety**: Make `ContentCache`, `_image_cache`, model cooldown tracking, and `io_semaphore` thread-safe and reuse the cached semaphore instance
- **Atomic Writes**: Use atomic write patterns for `ConfigManager.save()` and async byte writes
- **Resource Cleanup**: Reset semaphores and proxy-bypass state in shared-client cleanup
- **Observability**: Add debug logging for previously silent exception handlers
- **Webextract Regressions**: Fix `None` `tag.attrs`, selector conflicts, math protection, callout/task-list/table formatting, X.com Playwright crash, tweet noise, and resolver acceptance parity
- **Tooling Hygiene**: Resolve remaining Ruff, Pyright, Pytest, and Bandit issues and close low-priority parity coverage gaps

### Changed

- **HTML Conversion Path**: Route HTML files through the native webextract pipeline by default
- **Fetch Internals**: Split `fetch.py` into smaller modules and decompose `fetch_url()` into composable sub-functions
- **CLI Logging UX**: Improve batch progress reporting and quiet/verbose URL logs
- **Release Cleanup**: Update dependencies, CI and website docs, model metadata, and clean up project structure for the `0.12.0` release

### Removed

- **Obsolete Project Docs**: Remove outdated root docs, archived plans, and historical reference material during project cleanup

## [0.11.2] - 2026-03-14

### Fixed

- **Windows Compatibility**: Add Windows `GlobalMemoryStatusEx` RAM detection for proper heavy task semaphore sizing
- **Lazy Directory Creation**: Defer `~/.markitai/` directory creation from import-time to first write — prevents side effects when the tool is only imported or used read-only
  - `SPADomainCache`: mkdir moved from `__init__` to `_save()`
  - `SQLiteCache`: mkdir moved from `__init__` to `_get_connection()` with `_dir_ensured` flag to avoid repeated syscalls
- **Default Output/Log Dir**: `DEFAULT_OUTPUT_DIR` and `DEFAULT_LOG_DIR` now default to `None` instead of hardcoded paths — output directory must be explicitly specified via CLI `-o` or config file
- **Pyright Warnings**: Eliminate all 27 pyright warnings — suppress `reportUnsupportedDunderAll` for PEP 562 lazy-loading modules, fix `curl_cffi` `ProxySpec` TypedDict type mismatch
- **Schema Sync**: Update `config.schema.json` to match new `OutputConfig.dir` and `LogConfig.dir` nullable types

## [0.11.1] - 2026-03-14

### Added

- **Interactive Pure Mode**: Add pure mode option to interactive CLI wizard

### Fixed

- **Pure Mode Vision Bypass**: `--pure` now correctly skips screenshot-only and vision enhancement paths, falling through to text-only LLM processing
- **Pure Mode Warning False Positive**: `--pure --screenshot-only` no longer warns about `--screenshot` being ignored
- **URL Content Validation**: Lower `too_short` threshold from 100 to 30 characters — minimal landing pages were incorrectly rejected after stripping markdown syntax
- **Type Safety**: Fix `merge_llm_usage` parameter type to accept `LLMUsageByModel` (pyright warning)
- **Dead Code**: Remove unused `_format_standalone_image_markdown` alias

### Changed

- **CI**: Upgrade GitHub Actions to Node.js 24 compatible versions

## [0.11.0] - 2026-03-13

### Added

- **Pure Mode (`--pure`)**: Full implementation of transparent LLM pass-through mode — text cleaning only, no frontmatter generation or post-processing
- **Pure Mode Decoupled from LLM**: `--pure` no longer implies `--llm`; `--pure` alone writes raw markdown without frontmatter, `--pure --llm` sends content through LLM cleaning only
- **Image Vision in Pure Mode**: `--llm --pure` with image inputs routes to Vision analysis path (`process_image_with_vision_pure`)
- **`--keep-base` CLI Option**: Explicitly write base `.md` even in LLM mode (default: skip base `.md` when LLM is enabled)
- **Image-Only Format Handling**: Skip image-only formats (PNG, JPG, etc.) in non-LLM/non-OCR mode with clear warning
- **LLM Fallback**: Write `.md` as fallback when LLM processing fails
- **Batch Skip Summary**: Group skipped files by reason with example filenames in batch summary
- **Pure Mode Warning**: Warn when `--pure` silently overrides `--alt`/`--desc`/`--screenshot`
- **Mode-Specific Cleaner Prompt**: `{mode_rules}` template variable in cleaner prompt — standard mode gets image placeholder rules, pure mode gets YAML frontmatter preservation rules

### Fixed

- **URL Processors**: Respect `--pure`/`--llm`/`--keep-base` flags for base `.md` output in both single and batch URL processing
- **Pure Mode Frontmatter**: `process_with_llm` uses `clean_document_pure()` instead of `process_document()` in pure mode, preventing LLM-generated frontmatter (description, tags, etc.)
- **Source Frontmatter Reconstruction**: Reconstruct original YAML frontmatter from defuddle metadata before sending to LLM in pure mode
- **Vision Prompt Drift**: Add placeholder REMINDER to vision prompts to reduce LLM drift on `__MARKITAI_IMG_N__` placeholders
- **Stabilization Dedup**: Deduplicate stabilization calls and add `paged_stabilized` guard
- **Vision JSON Mode**: Fix wrong message index in vision `json_mode` and race condition in parallel gather
- **Misc Fixes**: Frontmatter regex, env variable quoting, Ctrl+C handling, hardcoded weight, docstring corrections
- **SVG as Image-Only**: Treat SVG as image-only format in batch mode

### Changed

- **Output Strategy**: LLM mode skips writing base `.md` by default (use `--keep-base` to override)
- **Test Performance**: Optimize test suite speed (~70s → ~30s)

## [0.10.0] - 2026-03-12

### Added

- **Auto-detect LLM Providers**: When no `markitai.json` config exists, automatically detect available providers from environment variables and authenticated CLI tools (Claude CLI, Copilot CLI, Gemini CLI, ChatGPT OAuth)
- **Shared Provider Detection**: Extract provider detection into `cli/providers_detect.py` shared module for reuse across interactive and non-interactive modes

### Changed

- **Interactive Mode UX**: Separate OCR and screenshots from LLM features into independent "Additional options" prompt, since they are local processing capabilities (RapidOCR, Playwright) that don't require LLM
- **Feature Display**: Unified `build_feature_str()` in `ui.py` separates LLM features from local features with `|` delimiter (e.g., `LLM alt desc | OCR screenshot`)
- **Interactive Mode Flow**: Show configured models after user confirms LLM enablement, not before; warn when no provider detected
- **Dependencies**: Raise minimum constraints to match tested versions (pymupdf4llm >=1.27.2, litellm >=1.82.0, pydantic >=2.12.0, pytest >=9.0.0, ruff >=0.15.0)
- **CLI Flags**: `-v` is now `--verbose` (was `--version`), `-V` is now `--version`

### Fixed

- **Image Alt Text Language**: Strip YAML frontmatter before extracting document context for image analysis, so alt text matches the document's actual language instead of defaulting to English
- **Interactive Provider Display**: Show actual configured models from config file instead of auto-detected provider name
- **URL Processor Feature Display**: Add missing OCR to URL processor dry-run features list
- **Cold Startup Performance**: Lazy imports in `cli/`, `processors/`, and `workflow/` `__init__.py` reduce cold startup from ~5s to ~0.3s

### Removed

- **Language Field**: Remove LLM-generated `language` field from Frontmatter model — LLM should only generate `description` and `tags`, not infer extra metadata

## [0.9.2] - 2026-03-11

### Fixed

- **Copilot/Claude Login**: Revert subprocess output interception for copilot/claude-agent login — always use inherited stdio so the CLI sees a real TTY, fixing credential storage failures
- **Login Output Display**: Detect URL and device code on the same line (copilot outputs both together); track externally-printed lines for clean erasure after login
- **Error Message Clarity**: Fix `format_error_message` following `__context__` (implicit exception chain) to wrapper exceptions like tenacity `RetryError`, replacing informative provider errors with opaque `<Future at 0x...>` messages in logs; now only follows `__cause__` (explicit `raise X from Y`)
- **Error Message Consistency**: Use `format_error_message` in CLI catch-all handlers (`file.py`, `workflow/core.py`) to prevent opaque chained exception messages reaching users

### Added

- `SubprocessInterceptor` URL+code same-line formatting for copilot device code flow
- `OutputManager.track_external_lines()` for tracking terminal output from inherited-stdio subprocesses

## [0.9.1] - 2026-03-09

### Fixed

- **Provider Auth Preflight**: Add `can_attempt_login()` guard to skip login prompt when provider SDK is missing; fix Rich markup swallowing `[gemini-cli]` via `escape()`; fix "Login failed: Login failed:" duplication
- **Install Scripts Extras Parsing**: Fix greedy regex (`\[.*\]` → `\[[^]]*\]`) that captured TOML outer brackets, corrupting extras names like `gemini-cli}]`
- **Install Scripts Resilience**: Progressive fallback when full extras install fails (retry without SDK-dependent extras); fix `set -e` silent exit on `uv tool install` failure; fix PowerShell 5.x `Join-Path` 3-arg incompatibility
- **Install Scripts Extras Strategy**: Merge-based finalize (no longer replaces manually tracked extras); generic receipt parsing (future-proof for new extras)

### Added

- `markitai doctor --suggest-extras` as single source of truth for install scripts to query recommended extras
- `can_attempt_login()` provider guard with `get_auth_resolution_hint()` fallback messages
- i18n key `not_found` for zh-CN and en in both setup scripts

## [0.9.0] - 2026-03-09

### Added

- **Fetch Strategy Priority**: Configurable global and per-domain strategy ordering via `strategy_priority` in `policy` and `domain_profiles`
- **Domain/IP Exemption**: `local_only_patterns` config field restricts specified domains/IPs to local-only strategies (static, playwright) — supports exact domain, suffix (`.internal.com`), wildcard (`*.internal.com`), IP, and CIDR notation (`10.0.0.0/8`, `fd00::/8`)
- **NO_PROXY Integration**: `inherit_no_proxy` (default: true) automatically merges `NO_PROXY` environment variable patterns into local-only exemptions
- **Fetch Security Feature**: README documentation for the new information security compliance capabilities

### Fixed

- **LLM Language Consistency**: Strengthened 5 prompt templates to prevent language translation when fetching mixed-language content (e.g., English UI + Chinese body) — LLM now determines output language from body text, not UI elements

## [0.8.1] - 2026-03-06

### Added

- **Defuddle Fetch Strategy**: New `defuddle` strategy (`GET https://defuddle.md/<url>`) as top-priority URL fetch method — free, no auth, returns clean Markdown with YAML frontmatter (title, author, published, description, word_count, domain)
- **Aggressive Strategy Ordering**: Default ordering changed to `defuddle → jina → static → playwright → cloudflare` (both default and SPA scenarios)
- **CLI `--defuddle` Flag**: Force defuddle-only URL fetching (mutually exclusive with `--playwright`, `--jina`, `--cloudflare`)
- **DefuddleConfig**: Configurable timeout and RPM rate limiting (conservative defaults for undocumented API limits)

### Changed

- **FetchPolicyEngine**: Simplified ordering logic — removed `has_jina_key` branching; defuddle+jina always first
- **max_strategy_hops**: Default increased from 4 to 5 to accommodate the new strategy

## [0.8.0] - 2026-03-06

### Added

- **Extended Format Support**: 20+ new file formats via markitdown and kreuzberg converters
  - **Markitdown-based**: HTML/HTM/XHTML, CSV, EPUB, MSG, IPYNB (Jupyter Notebook), Apple Numbers
  - **Kreuzberg-based** (optional dependency): TSV, XML, ODS, ODT, SVG, RTF, RST, ORG, TEX, EML
  - Kreuzberg is a pure Rust wheel — install with `uv pip install markitai[kreuzberg]`
- **Extended Image Support**: GIF, BMP, TIFF now supported by ImageConverter; BMP/TIFF auto-converted to PNG for LLM vision APIs
- **LLM Vision Format Helpers**: `is_llm_supported_image()`, `get_llm_effective_mime()` in `utils/mime.py` for transparent BMP/TIFF → PNG handling

### Fixed

- **Claude Agent SDK v0.1.46 compatibility**: Removed deprecated `allow_dangerously_skip_permissions` parameter (`permission_mode="bypassPermissions"` is sufficient)
- **i18n test isolation**: Fixed global state leak in `test_i18n.py` causing 3 integration tests to fail when run in full suite
- **Import-time log leakage**: Kreuzberg registration logs changed from `logger.debug` to `logger.trace` to prevent terminal noise before CLI log setup

### Changed

- **Converter registry**: New `FileFormat` enum members for all added formats; kreuzberg registers as gap-filler (only for formats without native converters)
- **Test fixtures**: Renamed to consistent `sample.*` naming convention; added fixtures for all new formats; removed orphaned `sample.mobi`
- **Markitdown lazy init**: `MarkItDown()` in `markitdown_ext.py` now initialized on first use instead of import time

## [0.7.0] - 2026-03-05

### Added

- **ChatGPT Provider** (`chatgpt/`): Subscription-based provider using ChatGPT OAuth Device Code Flow and Responses API. No extra SDK required — uses LiteLLM's built-in authenticator. Models: `chatgpt/gpt-5.2`, `chatgpt/codex-mini`, etc.
- **Gemini CLI Provider** (`gemini-cli/`): Uses Google's Gemini CLI OAuth credentials (`~/.gemini/oauth_creds.json`) with automatic token refresh. Optional SDK: `uv add markitai[gemini-cli]`. Models: `gemini-cli/gemini-2.5-pro`, `gemini-cli/gemini-2.5-flash`, etc.
- **Weight=0 Model Disabling**: Setting `weight: 0` in model config now explicitly disables the model (excluded from routing). Useful for temporarily disabling models without removing config.
- **Interactive Mode Enhancements**: Updated onboarding wizard with ChatGPT and Gemini CLI provider options

### Fixed

- **ZeroDivisionError in Router**: Models with `weight=0` are now filtered before LiteLLM Router creation, preventing `division by zero` in `simple-shuffle` routing strategy when all selected models have zero weight
- **Router Weight Selection**: `_select_model` fallback uses `random.choice()` instead of `random.uniform(0, 0)` when all models have zero weight

### Changed

- **Weight Field Semantics**: `weight` field description updated to clarify that 0 = disabled. Minimum value enforced at 0 (negative weights rejected by validation)

## [0.6.1] - 2026-03-05

### Fixed

- **Claude Agent SDK compliance**: Add `allow_dangerously_skip_permissions=True` when using `bypassPermissions`, pass system messages via SDK's `system_prompt` parameter instead of XML tags, set `additionalProperties: false` in JSON object schema
- **Auth pre-check gaps**: Detect `GH_TOKEN`/`GITHUB_TOKEN` env vars as valid Copilot authentication, detect `CLAUDE_CODE_USE_BEDROCK`/`VERTEX`/`FOUNDRY` env vars as valid Claude authentication
- **Resolution hints**: Include env var alternatives in authentication error messages

### Changed

- **Docs**: Update configuration guide and ai-tools-setup with env var auth methods

## [0.6.0] - 2026-03-04

### Added

- **Cloudflare Integration**: Unified cloud backend with two capabilities:
  - **Browser Rendering**: `--cloudflare` flag for cloud-based URL rendering via CF `/markdown` API, with rate limiting, cache TTL, and advanced params (`user_agent`, `cookies`, `wait_for_selector`, `http_credentials`)
  - **Workers AI toMarkdown**: Cloud-based document conversion for PDF/XLSX/DOCX/PPTX (converter backend)
- **Fetch Policy Engine** (`fetch_policy.py`): Policy-driven strategy ordering with domain-specific profiles, session persistence, and adaptive targeting
- **Domain Profiles**: Per-domain fetch config (`wait_for_selector`, `wait_for`, `extra_wait_ms`, `prefer_strategy`) in `markitai.json`
- **Playwright Session Persistence**: `session_mode` (`isolated`/`domain_persistent`) and `session_ttl_seconds` for reusing browser contexts across requests
- **Static HTTP Abstraction** (`fetch_http.py`): Pluggable HTTP backend with `httpx` (default) and `curl-cffi` (TLS fingerprint impersonation) via `MARKITAI_STATIC_HTTP` env var
- **Content Validation Gate**: All fetch strategies now validate content quality before accepting results
- **`api_base` env: syntax**: `"api_base": "env:MY_BASE_URL"` in model config for environment variable expansion
- **CF Markdown for Agents**: Content negotiation via `Accept: text/markdown` header for Cloudflare-enabled sites

### Changed

- **Vision Router Fallback**: When all vision models are disabled (`weight=0`), falls back to main router with warning instead of crashing
- **Playwright UTF-8 Encoding**: Force UTF-8 for HTML-to-Markdown conversion to prevent encoding errors
- **Integration Test Resilience**: Cloudflare integration tests now skip on rate limit (429) instead of failing

### Fixed

- **ZeroDivisionError in Vision Router**: Models with `weight=0` (disabled) are now filtered out before litellm Router creation, preventing `division by zero` in `simple-shuffle` routing strategy
- **Dead Code Cleanup**: Removed 21 dead functions/classes across 15+ files (backward compat aliases, deprecated functions, unused constants)

### Removed

- `_html_to_text`, `_normalize_bypass_list`, `_get_proxy_bypass`, `get_proxy_for_url`, `_url_to_session_id` from `fetch.py`
- `sanitize_error_message` from `security.py`
- `_deep_update`, `get_config` from `config.py`
- `order_dict_keys_sorted`, `_order_image_entry` from `json_order.py`
- `reset_consoles` from `console.py`
- `get_llm_not_configured_hint` from `hints.py`
- `remove_uncommented_screenshots`, `_UNCOMMENTED_SCREENSHOT_RE` from `llm/content.py`
- `get_pending_urls`, `finish_url_processing` from `batch.py`
- `LLMUsageAccumulator` from `workflow/helpers.py`
- `DEFAULT_LOG_PANEL_MAX_LINES` from `constants.py`
- Multiple backward-compatibility aliases from `cli/processors/`

## [0.5.2] - 2026-02-07

### Fixed

- **SQLite ResourceWarning**: Close SQLite connections explicitly via `_connect()` context manager, preventing `ResourceWarning: unclosed database` on Python 3.13
- **Windows path handling**: `context_display_name()` now handles `C:/` forward-slash Windows paths (was only handling `C:\`)
- **Windows install hints**: `markitai doctor` shows platform-specific install commands (PowerShell/winget on Windows, curl on Unix)
- **OAuth token expiry**: `markitai doctor` no longer reports "Token expired" when a valid refresh token exists
- **Config get output**: `markitai config get` renders Pydantic models as formatted JSON with syntax highlighting instead of raw Python repr
- **Copilot ProviderError**: Added missing `provider` kwarg when raising `ProviderError` for unsupported models
- **Pyright warnings**: Resolved all Pyright warnings (lazy `__all__`, type narrowing, optional imports)

### Changed

- **26 documentation fixes**: Comprehensive audit fixing docstring-to-code mismatches across all modules (llm, providers, converter, utils, config)

## [0.5.1] - 2026-02-07

### Added

- **Playwright auto-scroll**: Auto-scroll pages to trigger lazy-loaded content before extraction (up to 8 steps, inspired by baoyu-skills url-to-markdown)
- **DOM noise cleanup**: Remove navigation, ads, cookie banners, popups, and inline event handlers before content extraction
- **`python -m markitai`**: Add `__main__.py` for `-m` invocation support (fixes Windows execution)
- **Multi-provider detection**: Interactive mode (`-I`) now detects and displays all available LLM providers (DeepSeek, OpenRouter included)
- **Copilot GPT-5 series support**: GPT-5, GPT-5.1, GPT-5.2, GPT-5.1-Codex-Mini/Max, GPT-5.2-Codex now fully supported via Copilot provider
- **22 new unit tests**: Vision fallback strategies, smart_truncate edge cases, content protection roundtrip, cache fingerprint collision resistance, batch thread safety

### Changed

- **Default models modernized**: Updated outdated defaults across init/interactive/doctor (haiku→sonnet, gpt-4o→gpt-5.2, gemini-2.0→2.5, claude-sonnet-4→4.5)
- **Init wizard**: Multi-provider default selection, API keys stored in `.env` instead of plaintext config, next-steps hints after completion
- **LLM code deduplication**: `document.py` now delegates `_protect_image_positions` / `_restore_image_positions` to `content.py` shared functions
- **Cache fingerprint**: SHA256 over full content + page structure replaces `text[:1000]` prefix-based cache keys, preventing collisions for documents with identical prefixes
- **Batch thread safety**: Double-checked locking with timeout-based lock acquisition (5s) replaces non-blocking `acquire(blocking=force)`
- **LiteLLM model database**: Refreshed with 35 new models including Claude Opus 4.6

### Fixed

- **DOM cleanup JS syntax error**: Selectors with double quotes (e.g., `[role="banner"]`) now properly escaped via `json.dumps()` instead of f-string interpolation
- **Copilot model blocklist**: Removed outdated GPT-5 series from `UNSUPPORTED_MODELS` (only o1/o3 reasoning models remain blocked)
- **CLI provider display**: Truncate provider list with `(+N more)` when >3 detected to prevent line overflow

## [0.5.0] - 2026-02-06

### Added

- **Unified UI system**: New `ui.py` components and `i18n.py` module with Chinese/English support across all CLI commands
- **`markitai init`**: One-stop setup wizard — checks dependencies, detects LLM providers, generates config
- **Interactive mode** (`-I`): Guided setup with questionary prompts for new users
- **`doctor --fix`**: Auto-install missing components (e.g., Playwright)
- **Cross-platform install hints**: Platform-specific installation commands in doctor output
- **`MARKITAI_LOG_FORMAT`**: Environment variable override for log format
- **JSON repair**: Fallback parser for malformed LLM JSON responses using `json_repair`

### Changed

#### Performance

- **CLI startup**: Lazy-load processor and command modules (~3x faster `--help`)
- **Dependency checks**: Parallelized doctor and init with `ThreadPoolExecutor`
- **LLM processing**: Pre-compiled regex patterns and batched replacements
- **PDF rendering**: Parallel page rendering for standard and LLM modes
- **URL fetching**: Async-safe cache locking for concurrent requests
- **Executor**: Auto-detect heavy task limit based on system RAM
- **Image processing**: Offloaded CPU-intensive work to thread pool
- **Cache stats**: Merged stats and model breakdown into single SQLite query

#### Refactoring

- **Batch UI**: Replaced Rich table/LogPanel with compact unified UI (progress bar with current file, completion summary)
- **Log format**: Default changed to human-readable text (was JSON)
- **LLM cache**: Deduplicated `SQLiteCache`/`PersistentCache` into `llm/cache.py`
- **Single file output**: Layered output with `--verbose` for detailed logs
- **Setup scripts**: Consolidated 10 scripts into 2 unified files (`setup.sh` + `setup.ps1`) with built-in i18n

### Fixed

- **Windows**: LibreOffice detection with fallback to `Program Files` paths (not just PATH)
- **Windows**: FFmpeg/CLI path display — show "installed" instead of long winget package paths
- **Windows**: `config path` alignment with dynamic padding and continuous `│` column
- **Playwright**: Default `wait_for` changed to `domcontentloaded` (was `networkidle`, caused hangs)
- **Config**: Schema and function defaults synced with constants
- **Exceptions**: Preserved exception chains (`raise from`) across codebase
- **Cache**: Prevented stale `markitai_processed` timestamp on cache hit
- **CLI**: Version flag reverted to `-v/--version`, `--verbose` kept without short flag

### CI

- Added Windows LibreOffice install step (`choco`) to CI matrix
- Changed to `--all-extras` for comprehensive dependency testing
- Publish workflow: split unit/integration tests with `SKIP_LLM_TESTS`

## [0.4.2] - 2026-02-03

### Changed

- **Playwright defaults**: `wait_for` changed to `networkidle`, `extra_wait_ms` to 5000ms for better SPA support
- **Frontmatter validation**: Pydantic validators reject empty description/tags, triggering Instructor auto-retry
- **VitePress**: Upgraded to 2.0.0-alpha.16

### Fixed

- **X/Twitter content**: Pages now wait for full JS rendering before capture
- **Cache directories**: All caches now respect `cache.global_dir` config instead of hardcoded paths
- **Setup scripts**: Improved piped execution (`curl | sh`), proper Playwright installation paths
- **Config init**: Added `--yes/-y` flag for non-interactive use

## [0.4.1] - 2026-02-02

### Added

- **`markitai doctor`**: New diagnostic command for system health and auth status checking
- **Adaptive timeout**: Local providers auto-adjust timeout based on request complexity
- **Prompt caching**: Claude Agent caches long system prompts (>4KB) for cost reduction

### Changed

- `check-deps` renamed to `doctor` (old name kept as alias)
- Improved error messages with resolution hints for local providers

### Fixed

- Request timeouts on large documents with Claude Agent / Copilot
- JSON extraction issues with control characters and markdown code blocks

## [0.4.0] - 2026-01-28

### Added

- **Claude Agent SDK**: `claude-agent/sonnet|opus|haiku` via Claude Code CLI
- **GitHub Copilot SDK**: `copilot/claude-sonnet-4.5|gpt-4o|o1` models
- **URL HTTP caching**: ETag/Last-Modified conditional requests
- **Quiet mode**: `--quiet` / `-q` flag (auto-enabled for single file)
- **Module refactoring**: `cli.py` → `cli/`, `llm.py` → `llm/`, new `providers/`
- **Setup scripts hardening**: default N for high-impact ops, version pinning
- **Docs**: CONTRIBUTING.md, architecture.md, ai-tools-setup.md, dependabot.yml

### Changed

- Python 3.13, docs reorganized to `docs/archive/`
- agent-browser locked to 0.7.6 (Windows bug in 0.8.x)
- Default `extra_wait_ms`: 1000 → 3000, Instructor mode: `JSON` → `MD_JSON`

### Fixed

- **Windows**: UTF-8 console, Copilot CLI path discovery, script argument quoting
- **LLM**: Frontmatter regex fallback, `source` field fix, vision/frontmatter error handling
- **Prompts**: Enhanced prompt leakage prevention, placeholder protection rules
- **Content**: Social media cleanup rules (X/Twitter, Facebook, Instagram)
- **Setup**: WSL detection, Python pymanager support, PATH refresh order

## [0.3.2] - 2026-01-27

### Added

- Chinese README (`README_ZH.md`) with language toggle
- Chinese setup scripts: `setup-zh.sh`, `setup-zh.ps1`, `setup-dev-zh.sh`, `setup-dev-zh.ps1`

### Changed

- Improved setup scripts with better error handling and user feedback
- Updated Python version note: 3.11-3.13 (3.14 not yet supported)
- Updated documentation language toggle links

## [0.3.1] - 2026-01-27

### Fixed

#### Prompt Leakage Prevention
- Split all prompts into `*_system.md` (role definition) and `*_user.md` (content template)
- Added `_validate_no_prompt_leakage()` to detect and handle prompt leakage in LLM output
- Updated LLM calls to use proper `[{"role": "system"}, {"role": "user"}]` message structure

#### LLM Compatibility
- Fixed `max_tokens` exceeding deepseek limit by using minimum across all router models
- Fixed terminal window popup on Windows when running agent-browser verification

#### URL Fetching
- Improved error messages for browser fetch timeout (no longer suggests installing when already attempted)
- Added auto-proxy detection for Jina API and browser fetching
  - Checks environment variables: `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY`
  - Auto-detects local proxy ports: 7890 (Clash), 10808 (V2Ray), 1080 (SOCKS5), etc.

### Added

#### SPA Domain Learning
- New `SPADomainCache` for automatic detection and caching of JavaScript-heavy sites
- `markitai cache spa-domains` command to view/manage learned domains
- `markitai cache clear --include-spa-domains` option

#### Windows Performance Optimizations
- Thread pool optimization: Windows defaults to 4 workers (vs 8 on Linux/macOS)
- ONNX Runtime global singleton with preheat for OCR engine
- OpenCV-based image compression (releases GIL, 20-40% faster)
- Batch subprocess execution for agent-browser commands

### Changed

- Default image quality: 85 → 75
- Default image max_height: 1080 → 99999 (effectively unlimited)
- Default image min_area filter: 2500 → 5000
- Default URL concurrency: 3 → 5
- Default scan_max_depth: 10 → 5
- Extended fallback_patterns with more social media domains

## [0.3.0] - 2026-01-26

### Added

#### URL Conversion Support
- **Direct URL conversion**: `markitai <url>` converts web pages to Markdown
- **URL batch processing**: Support `.urls` file format (text or JSON), auto-detected from input
- **URL image downloading**: `download_url_images()` with concurrent downloads (5 parallel)
- Automatic relative URL resolution for images
- Cross-platform filename sanitization (Windows illegal characters handling)

#### Multi-Source URL Fetching (fetch.py)
- **Three fetch strategies**: `--static` / `--agent-browser` / `--jina`
  - `static`: MarkItDown direct HTTP fetch (default, fastest)
  - `browser`: agent-browser headless rendering (for JS-heavy pages)
  - `jina`: Jina Reader API (cloud-based, no local deps)
  - `auto`: Smart fallback (static → browser/jina if JS detected)
- **FetchCache**: SQLite-based URL cache with LRU eviction (100MB default)
- **Screenshot capture**: `--screenshot` for full-page screenshots via browser
- **Multi-source content**: Parallel static + browser fetch with quality validation
- Domain pattern matching for auto-browser fallback (x.com, twitter.com, etc.)
- `FetchResult` with `static_content`, `browser_content`, `screenshot_path`

#### agent-browser Integration
- Headless browser automation via `agent-browser` CLI
- Configurable wait states: `load`, `domcontentloaded`, `networkidle`
- Extra wait time for SPA rendering (`extra_wait_ms`)
- Session isolation for concurrent fetches
- `verify_agent_browser_ready()` with cached readiness check
- Screenshot compression with Pillow (JPEG quality + max height)

#### URL LLM Enhancement
- New `prompts/url_enhance.md` for URL-specific content cleaning
- Multi-source LLM processing: combine static + browser + screenshot
- Smart content selection based on validity detection

#### Cache Enhancements
- **`--no-cache-for <pattern>`**: Selective cache bypass with glob patterns
  - Single file: `--no-cache-for file1.pdf`
  - Glob pattern: `--no-cache-for "*.pdf"`
  - Mixed: `--no-cache-for "*.pdf,reports/**"`
- **`markitai cache stats -v`**: Verbose mode with detailed cache entries
- **`--limit N`**: Control number of entries in verbose output (default: 20)
- **`--scope project|global|all`**: Filter cache statistics by scope
- `SQLiteCache.list_entries()`: List cache entries with metadata
- `SQLiteCache.stats_by_model()`: Per-model cache statistics
- Improved cache hash: head + tail + length algorithm for better invalidation

#### Workflow Core Refactor (workflow/core.py)
- **`ConversionContext`**: Unified single-file conversion context
- **`convert_document_core()`**: Main conversion pipeline
  - `validate_and_detect_format()` → `convert_document()` → `process_embedded_images()`
  - `write_base_markdown()` → `process_with_vision_llm()` / `process_with_standard_llm()`
- Parallel document + image processing with proper dependency handling
- Alt text injection after LLM processing completes (race condition fix)

#### Official Website
- VitePress 2.x documentation site with bilingual support (English/Chinese)
- Custom theme with brand colors matching logo
- Local search integration
- GitHub Actions auto-deployment to GitHub Pages

#### Project
- **MIT License**: Added LICENSE file

#### CI/CD
- **`.github/workflows/ci.yml`**: Automated testing on push/PR
- **`.github/workflows/deploy-website.yml`**: Website deployment to GitHub Pages

#### Code Architecture
- New `utils/paths.py`: `ensure_dir()`, `ensure_subdir()`, `ensure_assets_dir()`
- New `utils/mime.py`: `get_mime_type()`, `get_extension_from_mime()`
- New `utils/text.py`: `normalize_markdown_whitespace()`, text utilities
- New `utils/executor.py`: `run_in_executor()` with shared ThreadPoolExecutor
- New `utils/output.py`: Output formatting helpers
- New `json_order.py`: Ordered JSON serialization for reports/state files
- New `urls.py`: `.urls` file parser (JSON and plain text formats)
- `LLMUsageAccumulator` class for centralized cost tracking
- `create_llm_processor()` factory function
- Unified `detect_language()` with `get_language_name()` helper
- Centralized `IMAGE_EXTENSIONS`, `JS_REQUIRED_PATTERNS` constants

#### Configuration
- **`supports_vision` now optional**: Auto-detected from litellm when not explicitly set
  - No need to manually configure for most models (GPT-4o, Gemini, Claude, etc.)
  - Explicit `supports_vision: true/false` overrides auto-detection if needed

### Changed

#### Package Rename
- **`markit` → `markitai`**: Package renamed for clarity
- CLI command remains `markitai`

#### Python Version
- **Python 3.11+ support**: Lowered minimum Python version from 3.13 to 3.11

#### CLI Behavior
- **Single file mode**: Direct stdout output (no logging by default)
- **`--verbose`**: Show logs before output in single file mode
- Batch processing behavior unchanged

#### Code Quality
- Refactored PowerShell COM conversion scripts (~18% code reduction)
- Unified MIME type mapping across codebase
- Extracted common fixtures to `conftest.py`
- Improved error messages for network failures (SSL/connection/proxy)
- Architecture diagram updated in `docs/spec.md`

### Fixed
- URL filename cross-platform compatibility
- Cache invalidation for large documents (tail changes now detected)
- Image analysis race condition with `.llm.md` file writing

## [0.2.4] - 2026-01-21

### Changed
- Restructured `assets.json` format with flat asset array
- Extract Live display management for early log capture
- Improved MS Office detection with file path fallback

### Fixed
- Add openpyxl FileVersion compatibility patch
- Add pptx XMLSyntaxError compatibility patch
- Enhanced `check_symlink_safety` with nested symlink detection
- LLM empty response retry logic
- `normalize_frontmatter` for consistent YAML field order

## [0.2.3] - 2026-01-20

### Added

#### Persistent LLM Cache
- SQLite-based cache with LRU eviction and size limits (default 1GB)
- Dual-layer lookup: project cache + global cache
- `CacheConfig` in `MarkitaiConfig` with enabled/no_cache/max_size options
- **`--no-cache` CLI flag**: Skip reading but still write (Bun semantics)
- **`markitai cache stats [--json]`**: View cache statistics
- **`markitai cache clear [--scope]`**: Clear cache by scope

#### Vision Router Optimization
- Smart router selection: auto-detect image content in messages
- `vision_router` property filtering only `supports_vision=true` models
- Replace hardcoded "vision" model name with "default" + smart routing

#### Legacy Office Conversion
- MS Office COM batch conversion: one app launch per file type
- `check_ms_word/excel/powerpoint_available()` registry-based detection
- Pre-convert legacy files before batch processing to reduce overhead

#### Performance (Phase 3)
- **Parallel PDF processing**: Concurrent page OCR & rendering
- **Parallel image processing**: `ProcessPoolExecutor` for CPU-bound compression
- Adaptive worker count based on file size
- LRU eviction and byte-size limits for image cache
- Batch semaphore for memory pressure control

### Changed
- OCR optimization: `recognize_numpy()` and `recognize_pixmap()` for direct array processing
- Reuse already-rendered pixmap in PDF OCR (avoid re-rendering)

### Fixed
- EMF/WMF format detection and PNG conversion support
- `DATA_URI_PATTERN` regex for hyphenated MIME types (x-emf, x-wmf)
- Base64 stripping: remove hallucinated images instead of replacing
- Batch timing: record `start_at` before pre-conversion for accurate duration
- Pyright venv detection: add venvPath/venv to pyproject.toml

## [0.2.2] - 2026-01-20

### Added
- `constants.py` module to consolidate hardcoded values
- Unit tests for image and llm modules
- `convert_to_markdown.py` reference script

### Changed
- Centralized constants usage across config.py, llm.py, batch.py, image.py
- Improved LLM content restoration with garbage detection logic
- Enable parallel batch processing for image analysis
- Move state saving outside semaphore to reduce blocking

### Fixed
- Rich Panel markup parsing issue (escape file paths)

## [0.2.1] - 2026-01-20

### Added

#### LLM Usage Tracking
- Context-based usage tracking (per-file instead of global)
- `get_context_cost()` and `get_context_usage()` for per-file stats
- Thread-safe lock for concurrent access to usage dictionaries

#### Type System
- `types.py` with TypedDict definitions (ModelUsageStats, LLMUsageByModel, AssetDescription)
- `ImageAnalysis.llm_usage` for multi-model tracking (renamed from `model`)

#### Model Configuration
- `get_model_max_output_tokens()` using litellm.get_model_info()
- Auto-inject max_tokens with fallback to conservative default (8192)

#### Office Detection
- `utils/office.py` module with cross-platform detection
- `has_ms_office()`: Windows COM-based MS Office detection
- `find_libreoffice()`: PATH + common paths search with `@lru_cache`

#### Image Processing
- `strip_base64_images()` method
- `remove_nonexistent_images()` to clean LLM-hallucinated references
- Normalize whitespace for standalone image `.llm.md` output

### Changed
- File conflict rename strategy: `.2.md` → `.v2.md` for natural sort order
- Batch state: add `screenshots` field (separate from embedded images)
- Batch state: add `log_file` field for run traceability
- Store file paths as relative to input_dir in batch state

## [0.2.0] - 2026-01-19

### Added
- **Monorepo architecture** with uv workspace (`packages/markitai/`)
- **LiteLLM integration** for unified LLM provider access
- New converter modules: `pdf`, `office`, `image`, `text`, `legacy`
- Workflow system for single file processing (`workflow/single.py`)
- Markdown-based prompt management system (`prompts/*.md`)
- Unified config with JSON schema validation (`config.schema.json`)
- Security module for path validation (`security.py`)
- Comprehensive test suite with fixtures

### Changed
- CLI rewritten with Click (replaced Typer)
- Requires Python 3.13+

### Removed
- Old `src/markitai/` structure and all legacy code
- Complex pipeline/router/state machine architecture
- Individual LLM provider implementations (OpenAI, Anthropic, etc.)
- Docker and CI scripts (to be re-added later)

### Breaking Changes
- Configuration format changed (see migration guide)
- CLI command syntax updated
- Python 3.12 and below no longer supported

## [0.1.6] - 2026-01-14

### Fixed
- Model routing strategy bugs
- Documentation accuracy improvements

## [0.1.5] - 2026-01-13

### Changed
- Refactored prompt management system for better maintainability
- Simplified cleaner module logic

## [0.1.4] - 2026-01-13

### Fixed
- JSON parsing edge cases in LLM responses
- Log formatting improvements for readability

## [0.1.3] - 2026-01-12

### Added
- Test coverage improved to 81%

### Changed
- Adopted `src` layout for project structure
- Reorganized documentation to `docs/reference/`
- Added GitHub Actions CI workflow

### Fixed
- Provider-specific bugs in fallback handling

## [0.1.2] - 2026-01-12

### Added
- Resilience features for network failures (retry logic, timeout handling)
- `CLAUDE.md` and `AGENTS.md` documentation for AI assistants

### Changed
- Log optimization for cleaner, more informative output

## [0.1.1] - 2026-01-11

### Changed
- Major architecture refactoring with service layer pattern
- Enhanced LLM support with better error handling and retries

## [0.1.0] - 2026-01-10

### Added

#### Capability-Based Model Routing
- `required_capability` and `prefer_capability` parameters for LLM calls
- Text tasks prioritize text-only models for cost efficiency
- Vision tasks automatically use vision-capable models
- Backward compatible: parameters default to None (round-robin behavior)

#### Lazy Model Initialization
- Providers loaded on-demand instead of all at startup
- Significantly reduced initialization time for single-file conversions
- `warmup()` method for batch mode to validate providers upfront
- `required_capabilities` parameter in `initialize()`

#### Concurrent Fallback Mechanism
- Primary model timeout triggers parallel backup model execution
- Neither model is interrupted - first response wins
- Configurable via `llm.concurrent_fallback_timeout` (default: 180s)
- Handles Gemini 504 timeout scenarios gracefully

#### Execution Mode Support
- `--fast` flag for speed-optimized batch processing
- Fast mode: skips validation, limits fallback attempts, reduces logging
- Default mode: full validation, detailed logging, comprehensive retries
- Configurable via `execution.mode` in config file

#### Enhanced Statistics
- `BatchStats` class for comprehensive processing metrics
- Per-model tracking: calls, tokens, duration, estimated cost
- `ModelCostConfig` for optional cost estimation
- Summary format: "Complete: X success, Y failed | Total: Xs | Tokens: N"

### Changed
- CLI architecture refactored for better modularity
- Config format migrated from JSON to YAML

## [0.0.1] - 2026-01-08

### Added
- **Initial release**
- CLI commands: `convert`, `batch`, `config`, `provider`
- Multi-format support: Word (.doc, .docx), PowerPoint (.ppt, .pptx), Excel (.xls, .xlsx), PDF, HTML
- LLM enhancement: markdown formatting, frontmatter generation, image alt text
- 5 LLM providers with fallback: OpenAI, Anthropic, Gemini, Ollama, OpenRouter
- 3 PDF engines: pymupdf4llm (default), pymupdf, pdfplumber
- Image processing: extraction, compression (oxipng/mozjpeg), LLM analysis
- Batch processing with resume capability and concurrency control
- Unit and integration tests
- Docker multi-stage build
- Chinese and English documentation

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
