# Troubleshooting

Start with `markitai doctor`. It checks the browser, LibreOffice, local OCR,
the built-in readers and service, and the configured models, without sending a
model request or opening a remote page. `markitai doctor --json` prints the same
checks for scripts. Add `-c FILE` to check a particular configuration.

## Exit status

| Status | Meaning |
|---|---|
| 0 | Success, including dry runs and skipped items |
| 1 | A single input failed, or a runtime error such as a missing input file or model |
| 2 | Usage error: unknown or removed option, missing `-c` file, `--json` without `-o`, interactive mode without a terminal |
| 10 | A directory or `.urls` batch finished with at least one failed item |
| 130 / 143 | Interrupted by Ctrl-C / SIGTERM |

Usage errors are printed on stderr only; `--json` prints its JSON document only
after arguments are accepted. Failed batch items are listed in the JSON output
and in the batch report under `out/.markitai/reports/`.

## Common problems

**An image produces no output.** A standalone image has no text unless you ask
for recognition; Markitai says on stderr that it skipped the image. Add `--ocr`
(on-device OCR, macOS) or `--llm` (vision model). With `--json` the item shows
`"skip_reason": "image_only"`.

**The first OCR takes half a minute.** Vision compiles its recognition models
for this executable the first time it runs and caches them, so the first OCR
after installing or updating Markitai takes about 25–45 seconds and later ones
well under a second per page. A sandbox that forbids writing
`~/Library/Caches/markitai` makes OCR fail with `missingError`.

**Images are missing when I print to stdout.** Without `-o`, Markdown goes to
stdout and image references such as `.markitai/assets/…` are not written
anywhere. Use `-o out/` to keep images; the `image.stdout_persist*` settings are
accepted but not implemented in this build.

**The output is named `….v2.md`.** The target already existed and the default
conflict policy is `rename`. Choose `overwrite` or `skip` with
`markitai config set output.on_conflict overwrite`, or for one run with
`--config-json '{"output":{"on_conflict":"overwrite"}}'`.

**`Error: No model configured; set MODEL and a provider API key, or llm.model_list`.**
`--llm` (or `llm.enabled`) needs a model. Set an API key such as
`OPENAI_API_KEY` and run `markitai init --yes`, or configure `llm.model_list`;
see [quick start](quickstart.md#7-enhance-with-a-language-model) and [LLM](llm.md).

**`Warning: LLM enhancement failed; base Markdown retained: Environment variable not found: …`.**
A configured model refers to `env:NAME` and that variable is not set. The base
Markdown is still written because `llm.on_failure` defaults to `fallback`; set
it to `fail` to make this an error. Put the key in the environment, in `.env`
in the current directory, or in `MARKITAI_HOME/.env` (`~/.markitai/.env`).

**`Warning: Some observed LLM requests could not be priced`.** The model or
endpoint is not in the bundled [price catalog](pricing.md), so `cost_usd` only
counts the requests whose price is known. It does not mean the call was free.

**`Error: --playwright has been removed, use '-s playwright' instead.`** The old
`--playwright`, `--static`, `--jina`, `--defuddle` and `--cloudflare` switches
are now values of `-s/--strategy`. `--kreuzberg` is gone because RTF is read
natively.

**`Fetch strategy 'cloudflare' is not implemented` / `Cloudflare file conversion is not implemented`.**
The Cloudflare strategy and `-b cloudflare` backend are not available in this
build. Use `-s auto`, `static`, `playwright`, `jina` or `defuddle`, and the
default native file backend.

**`Remote fetching is disabled by policy`.** `-s jina` and `-s defuddle` send the
URL to a third-party service. They are refused when `--no-remote-fetch` or
`MARKITAI_NO_REMOTE_FETCH=1` is set or `fetch.remote_consent` is not `always`.
Local, private and credential-bearing URLs are never sent to these services.

**`Chromium is not installed; install Chrome/Chromium or set MARKITAI_BROWSER_EXECUTABLE …`.**
JavaScript pages, `-s playwright` and web screenshots need Chrome or Chromium.
Install one, set `MARKITAI_BROWSER_EXECUTABLE` to its executable, or run
`markitai doctor --fix` to download the official headless shell into
`MARKITAI_HOME/browsers/native`. An explicit `MARKITAI_BROWSER_EXECUTABLE` that
points to a missing file is not replaced automatically; fix or unset it first.
See [browser installation](browser-installation.md).

**`Office screenshots require an installed LibreOffice (soffice on PATH) …`.**
Page images of Word, PowerPoint and spreadsheet files are exported through an
installed LibreOffice and the macOS PDF renderer. Text conversion does not need
it. See [Office page rendering](office-rendering.md).

**OCR or PDF page images fail on Linux or Windows.** On-device OCR, PDF page
rendering and HEIF/AVIF decoding use macOS system frameworks and return an
explicit unsupported error elsewhere. A standalone PNG, JPEG, WebP, GIF, BMP,
TIFF or SVG image can still be read by a vision model with `--llm`; PDF page
images need macOS.

**`Error: ocr.lang is not supported by the installed macOS Vision text recognizer`.**
`ocr.lang` must be a language the macOS Vision recognizer supports, such as
`en`, `zh`, `ja` or `de`; see [local OCR](ocr.md#language-selection).

**`Error: Interactive configuration requires a terminal`.** `-I`, `config edit`
and `init` without `--yes` need an interactive terminal. In scripts use
`config set` or `init --yes`.

**`Error: Configuration file does not exist: …`.** `-c` must name an existing
file (except for `config set` and `config edit`, which create it). Run
`markitai config path` to see which file is in use; `config validate FILE`
checks a file. See [configuration](configuration.md).

**A batch was interrupted.** On macOS and Linux, run the same command again with
`--resume`; completed items are kept and unfinished items are retried. Resume
is not available on Windows.

**`Error: Unsupported file format: '.xyz'`.** The message lists every recognized
extension. Rename files that have the wrong extension.

## Caches, logs and state

- `markitai cache stats` shows the model-answer and web-page caches;
  `markitai cache clear -y` empties them. `--no-cache` skips cached answers for
  one run but still saves fresh ones. See [cache](cache.md).
- File logs are off by default. Enable them with
  `--config-json '{"log":{"dir":"./logs"}}'` (or `MARKITAI_LOG_DIR`) and choose
  the level with `--log-level DEBUG`; logs never go to stdout.
- Everything Markitai stores lives under `MARKITAI_HOME` (default `~/.markitai`):
  `config.json`, `.env`, `cache.db`, `fetch_cache.db`, `learned_spa_domains.db`,
  `browsers/` and `serve/jobs/` history; the three databases follow
  `cache.global_dir` when you change it. Deleting a trial `MARKITAI_HOME`
  resets everything; batch state and reports stay in each output directory
  under `.markitai/`.
- Proxies come from `HTTPS_PROXY`, `HTTP_PROXY`, `ALL_PROXY` (and lowercase
  forms) and `NO_PROXY`, otherwise from the system's manual proxy setting. See
  [URL fetching](fetch.md#proxies).

## Reporting a problem

Include `markitai --version`, the platform, `markitai doctor --json`, the exact
command and the error text. `config list` hides keys and tokens by default; do
not add `--show-secrets` to a report.
