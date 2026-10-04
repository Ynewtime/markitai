# Compatibility and migration

This is the Rust rewrite of Markitai 1.2.0, currently **1.3.0-dev**. The current
handoff is for local validation; a stable release is not being published.
Output quality takes precedence over byte-for-byte equality with the Python
implementation. A supported extension or passing build is not a promise of
complete format fidelity.

## Commands and configuration

The entry points remain `markitai`, `mkai` and `markitai-mcp` (`markitai mcp`).
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
  Workbench connection IDs can reuse server credentials only under the
  [settings endpoint restrictions](service-settings.md#credentials-and-partial-updates).
- **Cloud processing:** `serve` requires fresh request authorization for selected
  Cloudflare work; a saved selection is not consent. Known model cost excludes
  Cloudflare charges. CLI/core remote-fetch policy is described in [fetch](fetch.md).
- **Output recovery:** the CLI verifies native ownership evidence before replacing
  earlier batch results. A matching filename or digest alone is insufficient.
  Current-user directory aliases can be accepted; linked document leaves are
  refused by default. See [ownership](output-ownership.md).
- **Caching:** native entries preserve their own content/model/prompt identity;
  older reference rows can remain inspectable without being reusable as new typed
  answers. Cache hits do not imply a fresh provider request or current model output.
- **Bindings:** Python, Node and Go share the conversion envelope but their
  convenience functions and error types differ. Windows Go/cgo packages are not
  delivered. See [bindings](bindings.md) and [MCP](mcp.md).

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
