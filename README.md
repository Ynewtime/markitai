# Markitai

Markitai converts documents, web pages and images to Markdown. A single Rust
core powers the `markitai` command (alias `mkai`), a local REST service with a
browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. The CLI executable needs no Python, Node.js or Go runtime; language
bindings need their respective language runtime.

Markitai 2.0 is the Rust rewrite of Python markitai 1.2. See the
[changelog](CHANGELOG.md) and
[compatibility contracts](docs/compatibility.md).

## Install

macOS and Linux:

```sh
curl -fsSL https://markitai.dev/setup.sh | sh
```

Windows (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```

The setup scripts download the release archive for your machine from
[GitHub Releases](https://github.com/Ynewtime/markitai/releases), check it
against the release's `SHA256SUMS`, and install `markitai`, `mkai` and
`markitai-mcp` for the current user, without administrator rights. Releases
cover macOS 11+ (Apple silicon, Intel), Linux with glibc 2.28+ (x86-64,
ARM64) and Windows (x64, ARM64). Python users can instead run
`pip install markitai` or `uv tool install markitai` (Python 3.10+). PyPI has
wheels for these six platforms only and no 1.x any more: elsewhere pip has
nothing to install and leaves an installed 1.x as it is. Manual downloads,
updates, uninstalling and going back to 1.x are in the
[installation steps](docs/quickstart.md#1-install).

The archives include license notices and selected Markdown guides for offline
use. OCR models, Chromium and LibreOffice are optional separate components.
See [platform support](docs/quickstart.md#platform-support) for requirements,
[unsigned macOS builds](docs/quickstart.md#unsigned-builds-on-macos) for Gatekeeper,
and [bindings](docs/bindings.md#installation) for language packages.

## Quick start

```sh
markitai report.docx -o out/                  # writes out/report.docx.md
markitai ./documents -o out/                  # whole folder, with a report
markitai scan.png --ocr -o out/               # local OCR; models may need preparation
markitai report.docx -o out/ --json           # structured result for automation
markitai doctor                              # inspect optional components
markitai --help                              # commands, options and examples
markitai serve                               # browser workspace on 127.0.0.1:3600
```

For a separate trial state, set `MARKITAI_HOME` before using the CLI:
`export MARKITAI_HOME="$PWD/.local/try-home"`. Plain document conversion does
not need a model connection. Existing configuration still applies; use
`--no-llm -b native` for local document processing without model enhancement.
URL inputs still need the network, even with `--no-remote-fetch`. Local OCR uses the portable Paddle engine on every
platform (macOS Vision with `MARKITAI_OCR_BACKEND=vision`); `doctor --fix` can prepare missing Paddle
models before offline use. JavaScript pages need Chrome/Chromium; Office page
images need LibreOffice. Model enhancement (`--llm`) uses configured provider
credentials or a subscription runtime.

## Documentation

- [Quick start](docs/quickstart.md): installation and the first ten tasks.
- [CLI](docs/cli.md) and [MCP](docs/mcp.md): command and Agent contracts.
- [Agent index](llms.txt) and [standalone Agent guide](llms-full.txt): plain
  Markdown entry points, also included in CLI archives.
- [Documentation index](docs/index.md): the full source-checkout documentation.
  The offline archive contains selected guides; other topic links require this
  repository checkout.

## License

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
