# Compatibility and migration

Markitai 2.0 is the Rust rewrite of Markitai 1.2.0. It is released as
executables with setup scripts on GitHub Releases and as wheels on PyPI for
six platforms (see [bindings](bindings.md#installation)).
Output quality takes precedence over byte-for-byte equality with the Python
implementation. A supported extension or passing build is not a promise of
complete format fidelity. What later releases may change is in
[versioning](versioning.md).

## Commands and configuration

The entry points remain `markitai`, `mkai` and `markitai-mcp` (`markitai mcp`),
from the executables and from the Python wheel, whose extras (`ocr`, `serve`,
`mcp`, `claude-agent` and the rest) remain as names that add nothing (see
[bindings](bindings.md#python)).
Existing public option spellings and configuration keys are retained where
implemented; unavailable requests return an explicit error. See the current
[CLI guide](cli.md), [configuration](configuration.md), and `markitai --help`
for actual accepted arguments and defaults. Exit codes are documented in
[troubleshooting](troubleshooting.md#exit-status).

Only one configuration file is selected; explicit CLI overrides and presets
apply over that file. `MARKITAI_HOME` isolates Markitai state. It does not redirect
an official subscription runtime's independent login store. Keep credentials in
configuration or the process environment, not in task content or published logs.

## Deliberate differences

- **Local conversion:** parsers run in Rust without a Python installation.
  Optional dynamic browsing and Office page capture still require Chromium and
  LibreOffice respectively; portable OCR uses separately downloaded model weights.
- **Document output:** Markdown structure and metadata aim to retain useful
  information, while fixing reference defects. Dynamic times and host paths
  vary. See [formats](formats.md), [output](output.md) and [reports](reports.md).
- **Images on stdout:** referenced assets persist under the selected Markitai
  home by default, with content-addressed `file://` links. No reference-style
  source symlink index is created; failed persistence keeps references and warns.
  See [images](images.md#images-on-stdout).
- **Office capture:** complete-sheet screenshots include hidden and empty sheets
  for supported workbooks. They are a full sheet canvas, not printed pagination.
  Numbers remains a table reader rather than a page renderer.
- **PDF and OCR:** text layout, hidden-text checks and OCR routing use native
  decisions. OCR success does not certify exact code, punctuation or numbers.
  See [PDF](pdf.md), [OCR](ocr.md) and [Office rendering](office-rendering.md).
- **Service authentication:** the startup token is required for all API clients,
  including loopback, unless the operator explicitly chooses `--no-auth`.
  Python markitai let a client on the same computer call the API without it; a
  local script now sends `Authorization: Bearer <token>` (or `?token=`), with the
  token printed at startup or fixed through `MARKITAI_SERVE_TOKEN` (see
  [serve](serve.md#network-and-file-boundaries)). Workbench connection IDs can reuse server
  credentials only under the
  [settings endpoint restrictions](service-settings.md#credentials-and-partial-updates).
- **Cloud processing:** `serve` requires fresh request authorization for selected
  Cloudflare work; a saved selection is not consent. Known model cost excludes
  Cloudflare charges. CLI/core remote-fetch policy is described in [fetch](fetch.md).
- **Remote fetch fallback:** Python markitai's built-in `fetch.remote_consent:
  always` let `auto` send a URL the local strategies could not read to
  defuddle.md, Jina Reader or Cloudflare. Here remote fetching stays opt-in: the
  same default keeps `auto` local, and the default `fetch.fallback_patterns`
  list has no effect. To keep Python's behaviour, write
  `fetch.remote_consent: always` in your configuration (`markitai config set
  fetch.remote_consent always`); a failure a remote reader could have repaired
  says so. A `serve` job a user token owns never uses a remote service, as in
  Python 1.3.0. See [fetch](fetch.md#consent).
- **Output recovery:** the CLI verifies native ownership evidence before replacing
  earlier batch results. A matching filename or digest alone is insufficient.
  Current-user directory aliases can be accepted; linked document leaves are
  refused by default. A folder batch therefore skips the symbolic links in the
  folder that Python markitai followed, and names them in its summary and
  `--json` (`skip_reason: "symlink"`); `output.allow_symlinks: true` follows
  them. See [ownership](output-ownership.md).
- **Caching:** native entries preserve their own content/model/prompt identity;
  older reference rows can remain inspectable without being reusable as new typed
  answers. Cache hits do not imply a fresh provider request or current model output.
- **ChatGPT subscription:** Python markitai signed in to ChatGPT itself
  (LiteLLM's device-code login, stored in `~/.config/litellm/chatgpt/`); this
  build uses the official Codex CLI instead and implements no private OAuth.
  Install Codex (`npm install -g @openai/codex`) and run `markitai auth chatgpt
  login` once; the Python login is not reused. `chatgpt/<model>` takes any model
  the installed Codex offers (`gpt-6-luna`, Python's default, included);
  [ChatGPT](subscription-chatgpt.md) has the details.
- **Bindings:** Python, Node and Go share the conversion envelope but their
  convenience functions and error types differ. Windows Go/cgo packages are not
  delivered. See [bindings](bindings.md) and [MCP](mcp.md).

## Coming from Python markitai 1.2

Besides the differences above, scripts written for Python markitai notice these:

- **Page and slide markers:** Python always wrote `<!-- Page number: N -->` (PDF)
  and `<!-- Slide number: N -->` (presentations), with no switch. They are left
  out by default here; `--page-markers` or `"output": {"page_markers": true}`
  keeps them, for scripts that split a document by page. The `rag` profile
  writes its own `<!-- page: N -->` either way. See [formats](formats.md).
- **Exit status and streams:** a usage error (unknown option, missing value)
  exits 2, where Python exited 0 or 1. An image converted without `--ocr` or
  `--llm` is skipped with a warning and exit 0 (Python: 1). `--llm` with no
  usable model stops with exit 1 before converting (Python wrote the base
  Markdown and exited 0); `--dry-run` warns about it. The `✓ Wrote out/a.md`
  line goes to stderr, so stdout holds only document output (Python printed
  the written path on stdout). See [troubleshooting](troubleshooting.md#exit-status).
- **Batch-only options on other inputs:** Python ignored them; here they stop
  the run before anything is converted, so a wrapper that always passes them
  breaks on single files. `--resume` with a single file or URL exits 1 (only
  directory and `.urls` batches save progress). `-g`/`--glob` and
  `--max-depth` need a directory input and exit 2 otherwise, a `.urls` list
  included. `--llm-batch-timeout` without `--llm-batch` exits 2, for a
  directory too. `--llm-batch` on anything but a directory exits 2 (Python: 1),
  and `--llm-batch-collect` with an input argument exits 2. Pass these options
  only to the runs they apply to. See [CLI](cli.md).
- **Asset names:** extracted pictures are named by their content
  (`.markitai/assets/510f5d9476ab159c03ccda77.jpg`, so a picture used twice is
  stored once) instead of numbered `<document>.0001.<ext>` names, and with
  `image.compress` on (the default) PNG and HEIC pictures are written as `.jpg`. Presentation
  screenshots are `<deck>.pptx.page0001.jpg`, not `slide0001`. Read the names
  from the Markdown, `images.json` or `-f json` rather than guessing them.
- **Slide and page images of Office files:** only an installed LibreOffice
  draws them, on every system; Python could also drive an installed Microsoft
  Office (PowerPoint through COM on Windows, through AppleScript on macOS with
  `office.macos_fallback`, which is accepted here with a warning that it has no
  effect). Without LibreOffice the text is converted and a warning says to
  install it. See [Office rendering](office-rendering.md).
- **Knobs that are gone:** `markitai doctor --timings` and `doctor
  --suggest-extras` (there are no extras to install; `doctor` names what is
  missing), `MARKITAI_PDF_WORKERS` (PDFs are read in process, without worker
  processes), `MARKITAI_STATIC_HTTP` (one static HTTP client), inline terminal
  images (`image.stdout_fetch_external` has no effect; pictures on stdout are
  kept as files, see [images](images.md#images-on-stdout)) and
  `batch.heavy_task_limit` (warned as having no effect; use `-j`/
  `batch.concurrency`). See [configuration](configuration.md).

## Limits and validation

Unsupported formats or options fail explicitly; a nonempty result is not proof
that every element was recovered. Warnings identify omitted or normalized content.
Use retained source documents and screenshots to check important output, especially
scans, formulas, legacy Office files and complex PDF layouts.

Real provider login, subscription inference and billing require their own account
and entitlement checks; fixture success does not establish them. Tests and recorded
measurements apply only to their named source, package, platform and corpus.
An older reference audit or a guide's example command does not establish
verification of a different version or platform.
