# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.24.0] - 2026-08-26

### Added

- **markitai is now a library as well as a CLI**: `markitai.convert("report.pdf")` and its async twin `aconvert` return a typed `ConversionOutput` — base and LLM-enhanced markdown, parsed frontmatter, asset and screenshot paths, per-image analysis and usage totals — reusing the CLI's configuration layering and flag semantics instead of growing a parallel config. Parser-noise suppression moved below the CLI so library calls keep stdout clean (locked by a subprocess regression test), and `aconvert` keeps the event loop free by running the CPU-bound conversion in a worker thread. Provisional while markitai is 0.x
- **MCP agents get markitai through the `mcp` extra**: `markitai.mcp` ships inside the main wheel and exposes `convert_document`, `convert_url`, `batch_convert` and `job_status` over stdio — `claude mcp add markitai -- uvx --from "markitai[mcp]" markitai-mcp` is the whole setup, with no second package to release or version. Results are written to disk with a truncated inline preview so a large document never floods an agent's context, and LLM enhancement stays off unless the caller asks for it
- **`--profile rag|obsidian|okf` shapes the output for its consumer**: `rag` moves images out of the hidden `.markitai/` directory into a visible `assets/` (hidden paths are skipped by common ingestors such as LlamaIndex's `SimpleDirectoryReader`), rewrites PDF page markers to `<!-- page: N -->`, and warns on pipe tables whose rows disagree with their header; `obsidian` adds optional `![[wikilink]]` image references; `okf` maps the frontmatter onto Open Knowledge Format v0.2 field names. Profiles are orthogonal to presets — presets decide what runs, a profile decides what the files look like — and with no profile the output stays byte-identical to before. The `images.json` sidecar schema is now frozen, documented, and guarded by a test
- **Conversion quality is now measured, three ways**: a snapshot guardrail freezes the default conversion output for a PDF/DOCX/PPTX/XLSX fixture set (normalized against timestamps, versions and absolute paths; refreshed only by an explicit `--update`); an opt-in olmOCR-bench runner scores the `--ocr` path against the public benchmark — its first runs put honest numbers on two real gaps (15.6% on handwritten scans, 23.1% where header/footer suppression is missing); and a blind A/B harness judges base vs LLM-enhanced output with position-swap debiasing, per-format aggregation, resumable checkpoints and half-price Batches request builders. Nothing calls a paid judge without explicit credentials; judging 20 documents costs roughly $0.08–$0.42 with mainstream judge models
- **A per-document LLM request circuit breaker** (`llm.max_requests_per_document`, default 50, `0` disables): transport retries × validation retries × fallbacks can no longer multiply without bound. When a document trips the cap, its remaining enhancement is skipped, the unenhanced output is kept, and the usage report carries a zero-cost marker entry saying so
- **The website serves `/llms.txt` and the README carries an honest comparison table**: the index lists the bilingual docs for LLM consumption, and the table sets markitai against markitdown, docling and anydoc — stating what each of them does better rather than only what we do
- **The serve API contract is now machine-checked**: every JSON route declares a pydantic response model, `scripts/export_openapi.py` exports the OpenAPI schema with the SSE event payloads injected (they never appear in route signatures, which is exactly where the frontend mirror had drifted), and a contract test compares that schema field by field against the webapp's hand-written type mirror. Writing the test first surfaced four real drifts — the mirror lacked `ItemPayload.output_name`, kept its own copy of the job-item cap, marked `api_base_placeholder` optional though the server always sends it, and cited a contract document that does not exist — all fixed, with the cap now served through `capabilities.limits` and read at runtime. A new CI job lints and type-checks the webapp so the mirror cannot rot silently again
- **The defuddle port has a manifest and an upstream watch**: `PORT_MANIFEST.md` records which upstream sources each `webextract` module tracks and the commit the parity corpus is pinned to — a unit test keeps the two pins equal and the sync script rewrites them together — and a weekly workflow opens an issue when upstream cuts a release ahead of the pin. The port had no attribution map and no upstream signal at all until now
- **`--llm-batch` runs a directory's LLM stage at half price**: a directory batch can submit its enhancement to the provider's Batch API instead of converting in real time, on a single-model OpenAI or Anthropic pool. It waits up to `--llm-batch-timeout` (1h by default) and then hands off — the batch keeps running server-side and `--llm-batch-collect <id> -o <dir>` finishes it later from state written next to the output, so the original input directory is not needed. Cache hits are served without being submitted at all, and any document the batch could not satisfy is re-run live rather than lost. Not yet combinable with `--alt`/`--desc`/`--screenshot`/`--ocr`, which say so instead of failing quietly
- **Legacy `.doc`/`.ppt` conversion through `markitai[legacy]`**: the Office 97-2003 formats now go through the bundled-Rust anydoc backend — millisecond-scale, and with no Microsoft Office or LibreOffice installation on any platform. `doctor` reports it as an optional capability. Two honest limits versus the automation it replaces, documented rather than glossed: embedded images are not extracted, and PPT tables arrive as plain text
- **`--ocr --llm` is now named, measured and gated**: that combination has always sent page images to a vision model instead of running RapidOCR, but nothing said so. `--ocr` help, `doctor` and both CLI guides now call it VLM-OCR; converter metadata carries `ocr_path` (`vlm`/`rapidocr`/`none`) so a report can tell the two apart; the olmOCR-bench runner gained a `--vlm` mode for a head-to-head against local OCR; and a one-time disclosure prints before any page image leaves the machine, with `MARKITAI_NO_VLM_OCR=1` forcing the local path outright
- **Long page screenshots are tiled instead of shrunk**: a tall page used to be downscaled into one very long JPEG — detail lost irreversibly, and the result too tall for a vision model to read. Pages taller than `screenshot.tile_height` (2000px) are now split into full-width vertical tiles, and the screenshot-only path reads them tile by tile
- **Mathematics comes back as LaTeX**: PDF extraction has no notion of a formula — pymupdf hands a display equation over as an image and mangles inline math into markdown noise that swallows the prose beside it. The prompts now ask for `$...$` / `$$...$$` at every level that sees the page: `--ocr --llm` repairs inline math against the page image, and `--alt`/`--desc` recovers a display equation's LaTeX into `images.json`, which is the only place its content can become text again. Without an LLM a display equation is still kept as an image reference — lossless, but not text. Web pages were already covered: MathJax and MathML convert with no model involved
- **CLI conversions can opt into the web UI history**: `--record-history` (or `MARKITAI_RECORD_HISTORY`, or `history.record` in the config; flag > env > config, off by default) records a completed CLI run as a job under `~/.markitai/serve/jobs/`, copying outputs and referenced assets so it appears in the `markitai serve` history page live, without restarting the server. CLI-recorded entries carry `origin: "cli"` and show a "CLI" badge in the web UI, and share the same seven-day TTL cleanup, deletion, and archive download as web-created jobs. Recording is skipped in stdout/pipe mode, never fails a conversion, and prints a one-line dim confirmation to stderr (suppressed by `--quiet`)
- **`NOTICE` records the third-party obligations markitai's own MIT licence cannot cover on its own**: the AGPL-3.0 PyMuPDF stack and what it means for redistribution and network use, the defuddle port (MIT © kepano) that `webextract/` is built from, and the marker port (Apache-2.0) behind the benchmark scorer. The first two had no attribution anywhere in the repository until now
- **CI fails on a non-commercial or unexpected copyleft dependency**: `scripts/check_licenses.py` reads licence metadata from a no-extras install and rejects anything non-commercial or proprietary, plus any AGPL/GPL package outside an explicit allowlist (currently exactly the three PyMuPDF packages)

### Changed

- **The three LLM routing surfaces merged into one `MarkitaiRouter`**: local providers dispatch through their handler table, standard models go back to litellm's own group balancing, per-deployment cooldowns and fallbacks instead of a hand-rolled reimplementation — a net deletion of several hundred lines, with the weighted-selection and error-classification logic now existing exactly once. `router_settings` keys now mean what they say: `num_retries` drives markitai's single transport-retry loop (it was previously zeroed silently), and a group-wide cooldown retries with backoff instead of failing outright
- **Structured LLM calls use the strongest mode a model actually supports**: native tool calling first, then `response_format` JSON schema — which also switches on the claude-agent provider's until-now-unused native path — with "ask for JSON inside markdown" demoted to the last rung only. The ladder degrades automatically per call, so a capability table that overpromises costs at most an extra request instead of a failure. Four of the five JSON-repair layers are retired: repair now exists in one place and runs only on the last rung, and a clean call on a capable model goes from three potential retries to one native request
- **Self-explanatory errors no longer leak exception class names**: a conversion failing because the OCR backend is missing, a file exceeds the size limit, or a format is unsupported prints just the actionable message instead of prefixing internal class names to it. Unexpected errors keep the class name — there it is a diagnostic, not noise — and the full traceback now reaches the debug log, where it was previously discarded entirely
- **The parity corpus is pinned to the same upstream as the algorithm**: fixtures resynced from the March 2026 snapshot to defuddle 0.19.3 (83 → 208), after porting the six upstream behaviours the new corpus exposed as missing. On the 83 shared fixtures the quality benchmark rose 92.73 → 94.07 with no fixture regressing; the full 208-fixture baseline stands at 95.59 with per-fixture guardrail floors re-established
- **One URL pipeline behind all four entry points**: `serve`, the Python API, `markitai <url>` and URL batches carried near-identical copies of fetch → localize images → enhance → write. They now wrap a single cascade in `workflow/url.py`, which also means `serve` and the library gained vision enhancement, screenshot-only reading and per-image alt/description analysis without a line of their own
- **Prompt caching actually engages**: five system templates had their per-request variables near the top, so the stable prefix was a few lines long and no provider's prefix cache could match. The variable segments moved to the tail (instruction text unchanged), usage now reports `cached_input_tokens`, and the router attaches Anthropic `cache_control` at its exits
- **Converting a `.txt` no longer loads a PDF and Office toolchain**: converter modules were all imported so their registration decorators would run, which cost ~430ms per process and pulled in markitdown, Magika and onnxruntime for every conversion regardless of format. A format now imports only the module that handles it — `import markitai.converter` drops to ~9ms, and text conversions touch none of that stack
- **A provider is described in one table, not five**: the model each provider defaults to, and where its API key is read from, were hand-copied across credential detection, the setup wizard, `serve`'s startup candidates and `init` — and had drifted (three tables on a previous model generation, one wizard missing OpenRouter). Both facts now live in `constants` with guard tests that fail on a second copy. The defaults are also documented, so what markitai picks for you is checkable
- **Retirement warnings and cost estimates are derived, not transcribed**: the warning for a retired model recommended a literal replacement that had itself gone stale, and stamped every model with one hardcoded retirement date. It now reads the replacement from the provider-default table (following the provider you actually configured) and the date from litellm, printing none where litellm has none
- **webextract parses and copies less**: fragment and document parsing converged on one lxml path, five copies of the block-tag table converged on the constants module, the retry pipeline stopped deep-copying the whole tree on every attempt, and `srcset` picking exists once. Parity and quality floors unchanged
- **A default install is 155 MB smaller** — 633 MB down to 477 MB, measured on a clean environment. `opencv-python` left the core entirely, and RapidOCR moved behind a new `ocr` extra. **Scanned-document and image OCR now needs `uv tool install "markitai[ocr]" --force`**: `doctor` reports OCR as an optional capability rather than a missing requirement, `--ocr` without it fails with that exact command, and the "this PDF looks scanned" advisory prints it too — so you are never sent around the loop of advice, flag, error, reinstall
- **Images come out better without OpenCV**: compressing through Pillow alone beat OpenCV on every sample measured — higher PSNR and SSIM, and 7.7% smaller files — because OpenCV's `INTER_LANCZOS4` downscale skips anti-alias prefiltering and bakes the aliasing into the JPEG. The default install had been producing *worse* images than its own fallback path. On the real conversion path the throughput cost is at most 14%, because each document's worker pool re-imports anyway and `import cv2` costs 63.7 ms against Pillow's 23.2 ms
- **Every dependency moved to its current release**, `litellm` 1.91.1 to 1.97.0 and `markitdown` 0.1.6 to 0.1.7 among 56 updates, with the licence audit re-run against the upgraded tree so the PyMuPDF floor and the allowlist still hold
- **The PDF engine floor moved to `pymupdf4llm>=1.28.2`**: 1.28.0 pulled in `pymupdf-layout` under a **Polyform Noncommercial** licence, which forbids commercial use outright rather than conditioning it. 1.28.2 restored AGPL-3.0 dual licensing. Pinning backwards is not a way out — 1.27.2.x pins a `pymupdf-layout` carrying the same non-commercial terms

### Removed

- **The three-platform Office automation is gone** — about 1.3k lines of Windows COM, macOS AppleScript and LibreOffice CLI driving, along with the batch pre-conversion machinery built around it. `markitai[legacy]` covers the formats it existed for; the LibreOffice and Microsoft Office hints now speak only about PPTX slide rendering, which still uses them
- **Two redundant defence layers inside the LLM path**: image analysis had a hand-rolled JSON-mode strategy sitting below a structured ladder whose own bottom rung already does the same repair, and a hand-maintained Copilot price table that live traffic had stopped reaching. Both deleted after the paths that replaced them were verified against real calls
- **FFmpeg is no longer checked, advertised, or installed**: markitai has never supported audio or video — its format table contains no such entry, and the dependency comment has said the audio extras are unused for as long as they have been declared — yet `doctor` presented FFmpeg as "audio/video file processing", `init` checked for it, and the installers offered to fetch it
- **The six deprecated fetch and backend flag aliases are gone** (`--playwright`, `--defuddle`, `--static`, `--jina`, `--cloudflare`, `--kreuzberg`) in favour of `-s <strategy>` and `-b <backend>`, the documented spelling since 0.15.0. Passing a removed name is a usage error that names its replacement, so an old command line tells you exactly what to change instead of failing with "no such option". `markitai --help` goes from 36 options to 30, and its fetch/backend panel from 8 to 2

### Fixed

- **Six extraction gaps against defuddle 0.19.3**: article text inside dismissible `aria-hidden` overlays is kept (upstream issue 232); CodeMirror-rendered code blocks keep their content and language; mid-article image rows are no longer deleted as related-post cards; Substack Notes get a dedicated extractor instead of carrying page chrome; SVG figures keep their content with external-CSS fallback styling resolved; inline related-story blocks no longer take the surrounding prose with them. The resync also fixed what it newly exposed: Hugo admonitions, lightbox image duplication, LaTeX image services, `<noscript>` image fallbacks and line-numbered code layouts
- **Embedded PDF images survive symlinked and out-of-tree output directories**: pymupdf4llm may write image paths in symlink-resolved form (macOS `/tmp` → `/private/tmp`) or relative to the process working directory, which the single-spelling reference rewrite missed — leaving absolute paths in the markdown that alt text, reference validation and asset discovery all failed to match. All three spellings are rewritten now
- **`router_settings.fallbacks` actually works**: configured fallback groups now take over when a primary group's deployments fail. The setting was documented but bypassed entirely — deployments were addressed directly by id, so litellm's fallback machinery never saw a group to fall back from. A configuration that names no `default` group now fails at startup with a clear error instead of silently routing around the setting
- **Batched requests to reasoning models stopped failing outright**: OpenAI's batch deployments reject function tools while reasoning is on, so every document in a batch came back as a 400 and fell through to a live re-run — the batch bought nothing. Requests now carry `reasoning_effort="none"` when the model is reasoning-capable and the request uses tools, which is what the provider's own error message asks for
- **Claude models no longer cost $0 in the usage report**: for a name its table does not carry, litellm infers a provider and answers with zero prices — which read as "free" and hid the real entry a fuzzy match would have found. Every Copilot and Claude-CLI run through five Claude models was priced at nothing. Relatedly, `gpt-5.4-codex` was priced as plain `gpt-5.4`, because the fuzzy matcher accepted a candidate missing the suffix that selects the tier
- **A finished conversion can no longer report failure**: markitdown imports Magika, Magika imports onnxruntime, and onnxruntime's teardown at interpreter shutdown occasionally aborts the process — turning a conversion that had already written its output into exit code 134, and skipping the cleanup of its temporary directory. The CLI now finishes on its own terms once the work is done
- **`--llm-batch-collect` honours `--config-json`**: it built its configuration without the inline overrides, so a collect run's live re-run fallback used whatever pool the config file happened to name
- **The test suite is hermetic to the developer's own configuration**: a real `~/.markitai/config.json` or `.env` used to make parts of the suite fail on the machine of anyone who actually uses markitai
- **Install hints named the wrong package**: every message the CLI prints was interpolated into a markup-enabled render, so rich read the `[ocr]`, `[serve]` and `[browser]` in an extras name as a style tag and dropped it. The remedy printed to the terminal was `uv tool install "markitai" --force` — reinstalling exactly what you already had, with none of the missing capability. Caller text is now escaped in the shared message helpers, so titles, warnings and error details keep their brackets too
- **`--ocr` without the OCR extra no longer reports success**: converting an image asked for text recognition, silently fell back to an image placeholder when the backend was missing, and exited 0 — the file "converted" with nothing read from it. A missing backend is now a hard failure carrying the install command, while a genuine OCR failure (engine present, but it errored or found no text) still degrades to the placeholder as before
- **The wheel now carries `LICENSE` and `NOTICE`**: `license-files` named a path that does not exist inside the package directory, so hatchling matched nothing and every published wheel shipped without a licence at all. (Escaping upwards is not the fix — it builds, but writes a `../../` path-traversal entry into the wheel; the package keeps its own copies instead, guarded against drift by a test.)
- **CMYK JPEGs no longer fail image compression**: images from print-sourced PDFs reached a code path that only handled RGBA, P and LA, and were dropped
- **Improving a prompt now actually changes the output**: the persistent LLM cache was keyed on a call's *category* (`document_process`, `cleaner`, …) and never on the prompt itself, so a document converted before a prompt change kept serving the old result forever — including for prompts you customise yourself through `prompts.*`. The key now carries a digest of the resolved templates and of the instructions embedded in code, so editing either invalidates exactly the entries it should. Cached entries from earlier versions miss once after upgrading
- **A failed LLM call can no longer leave a document permanently blank**: when every retry came back empty, that empty string was written to the persistent cache, which has no expiry — the document then "converted" to nothing on every future run, with no remedy but clearing the cache by hand. Empty and whitespace-only results are now rejected before any cache layer sees them, and the conversion falls back to the unenhanced text with an error in the log
- **Truncated LLM responses are counted in the cost report**: a response cut off at the token limit was discarded *before* its usage was recorded, so the calls most likely to be expensive were exactly the ones missing from the totals
- **Reddit, Hacker News and YouTube pages are quality-checked again**: their extraction gate was registered under a profile name the extractors never emit, so all three silently fell through to the generic-article check and their own noise and length heuristics never ran. The registry is now keyed by the profile enum itself, so the name can no longer drift
- **`NO_PROXY` is honoured when fetching pages**: the unified proxy resolver had no callers at all, so static HTTP fetches went through the proxy regardless of the exception list — including the system exception list on macOS and Windows
- **The installer asks about mirrors only when the default index is actually slow**: it used to warn every user who had no proxy configured — so someone in Frankfurt installing markitai met an irrelevant warning and a question about China mirrors as their first interaction. It now makes one short, timed request to PyPI and validates both the status code and that a body came back, and stays silent unless that fails. `MARKITAI_USE_MIRROR=1` (or `0`) skips the probe either way
- **A "no" to OCR is now respected by the installer**: `doctor --suggest-extras` lists `ocr` unconditionally and the installer merges suggestions back into the selection, so declining it still installed it
- **`NO_PROXY` is honoured on the remaining fetch paths too** — the Playwright browser launch, both Cloudflare paths and the batch renderer resolved a proxy without consulting the exception list. One blind spot is documented rather than papered over: a batch shares a single browser whose proxy is fixed at launch, so a batch mixing exempt and non-exempt hosts keeps the proxy and logs a warning naming the URLs that will still use it
- **The screenshot viewport settings finally do something**: `screenshot.viewport_width` and `viewport_height` were declared, described and published in the JSON schema, but no code read them — every capture rendered at Playwright's own 1280x720 default. URL screenshots now open the browser at the configured size, which by default means the 1920x1080 the setting has always claimed. A guard test walks the config models and fails on any other field nothing reads
- **Two settings that looked configurable never were**: `auto_proxy` and the screenshot `full_page` flag were read with `getattr(config, …, True)` although no config model declares either, so the fallback always won and neither appeared in `config list` or the JSON schema. Both are gone — `NO_PROXY` is the supported way to bypass a proxy, and `screenshot.max_height` remains the bound on a runaway page. A test now fails on any config attribute read that no model declares
- **Proxy auto-detection no longer invents a proxy that isn't there**: it probed a handful of common local ports for a TCP connection, which succeeds for *every* port under a TUN-mode VPN and otherwise often just finds an unrelated dev server. Detection now trusts `HTTP_PROXY`/`HTTPS_PROXY`/`ALL_PROXY` and the operating system's proxy settings only

### Security

- **`markitai serve` now issues an access token**: the startup banner prints sign-in URLs, and requests from other machines must present the token — without it they get a 401, replacing the old unauthenticated partial access. An authenticated peer gets the same rights as the local machine, intranet URLs and history included, which is what makes a LAN bind actually usable. Loopback requests stay trusted without a token, `MARKITAI_SERVE_TOKEN` fixes the token across restarts, and `--no-auth` restores the previous behaviour
- **`markitai serve` can no longer be steered into the network it runs on**: with `--host 0.0.0.0`, any machine on the LAN could submit a URL and have the server fetch it — private addresses and cloud metadata endpoints included. The 0.23.0 Host/Origin guard stops a malicious web page, not a direct API call. Requests arriving from another machine are now limited to public URLs, while the machine running the server keeps full access, so converting your own intranet pages still works from there
- **Binding beyond loopback now says what that means**: `markitai serve --host 0.0.0.0` warns that the server has no authentication and that anyone who can reach it may convert files and read, download or delete the entire conversion history. The default loopback bind stays silent
- **`fetch.remote_consent=ask` now asks before X/Twitter enrichment too**: that one path went remote without prompting and honoured only a decision made earlier, so a link could reach a third-party service under a setting that promises to ask first. It now shares the single process-wide consent decision with every other remote service — one answer still covers them all

## [0.23.0] - 2026-07-18

### Added

- **Export any web-workspace preview to PDF**: the rendered Markdown prints to a clean A4 document — real four-side margins, whole code blocks and figures that never clip mid-page, and repeating table headers. A "PDF settings" menu holds an optional custom header/footer (on by default); PDFs always print light so the result is identical in Chrome, Safari and macOS Preview

### Changed

- **The web workspace is now usable on phones and narrow windows**: a single ≤780px layout gives the composer, options and task list one coherent column — the header collapses to identity plus history/settings, language and theme move into the settings dialog, external links and notifications dock to the bottom, and the batch download sits under the task list
- **Batch download moved next to the conversion options** on desktop, so it no longer competes with the Clear action in the workspace header

### Fixed

- **The local server now refuses cross-origin and rebinding requests**: a Host/Origin allow-list (loopback, IP-literal hosts, and any `--allowed-host` you pass) blocks a malicious web page from reaching the API — which could otherwise read saved provider credentials or drive server-side URL fetches — while same-origin use and LAN binds keep working
- **In-place retry and LLM enhancement are reliable**: retrying an item while its siblings are still converting no longer strands it as "queued"; a failed enhancement keeps the working base result instead of downgrading the row to an output-less error; an interrupted rerun no longer erases the whole job from history on restart; and re-runs overwrite their own prior outputs instead of accumulating versioned duplicates or serving a stale enhanced variant
- **Scanned-document OCR stays accurate across a batch**: the sparse-page tiled fallback no longer lowers detection thresholds process-wide, so later pages are OCR'd at full confidence
- **LLM alt text applies to images whose names contain spaces or CJK**: references written percent-encoded were not being matched, so captions were silently dropped for those assets
- **Quoted posts from X keep their exact text**: an interior ellipsis or "show more" run inside the quote no longer fabricates a truncation marker on complete text
- **Editing only a provider's API key no longer pins its default API base**: the editor field is empty (with the default shown as a placeholder) when the connection uses the provider default, so saving keeps tracking that default
- **Whole-history downloads clean up after themselves**: the generated archive is removed once the download finishes and can no longer survive as an orphan, and a file deleted mid-build is skipped instead of failing the download
- **The completion notification reads "N done" for a clean job** instead of "N done · 0 failed", and the local server's job finalization no longer blocks other jobs' live progress while sizing large output directories

## [0.22.0] - 2026-07-14

### Added

- **`markitai serve` adds a local web workspace for conversion**: install the `serve` extra to upload files or folders, submit URLs, watch live progress, preview and download results, and revisit seven days of on-disk history from a bilingual, accessible interface
- **LLM setup moves into the web workspace**: discover available local and API-backed providers, browse live model lists, configure weighted deployments, test connections without exposing stored credentials, and use the same settings for conversion jobs
- **Web jobs gain recovery and comparison tools**: retry failed file or URL items, filter large result sets, receive background completion notifications, copy an equivalent CLI command, and compare the base Markdown with its LLM-enhanced variant

### Fixed

- **PDFs with spaces or parentheses in the filename now get embedded-image analysis**: pymupdf4llm sanitizes the source name when writing extracted images (`My Paper.pdf` → `My_Paper.pdf-0001-00.jpg`), but embedded images were collected by globbing on the raw input name, so `--alt`/`--desc`/`rich` silently skipped every embedded image while page screenshots kept working. Collection now resolves the image refs written into the Markdown, keeps the previous prefix match as a fallback, and warns when a referenced asset is missing from disk
- **Web result downloads work with nested artifacts on Windows**: API paths use URL separators instead of filesystem backslashes, so images and other generated assets open through the same links on every supported OS

## [0.21.1] - 2026-07-12

### Fixed

- **Fresh installs on macOS and Windows no longer try to compile litellm from source**: litellm 1.92.0 shipped Linux-only wheels, so `uv tool install markitai` on other platforms fell back to building its new Rust extension from the sdist — failing outright without a Rust toolchain, or with a wall of pyo3 errors when the build environment picked an unsupported Python (e.g. a system-default 3.14). That release is excluded; dependency resolution lands on a version with wheels for every supported platform

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
