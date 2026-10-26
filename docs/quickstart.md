# Quick start

Markitai converts documents, web pages and images to Markdown. One Rust core
powers the `markitai` command (also installed as `mkai`), a local REST service
with a browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. This guide covers markitai 2.0, the Rust rewrite of Python markitai
1.2.

## 1. Install

### Setup scripts

macOS and Linux:

```sh
curl -fsSL https://markitai.dev/setup.sh | sh
```

Windows (Windows PowerShell 5.1 or PowerShell 7):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://markitai.dev/setup.ps1 | iex"
```

The scripts download the latest release's archive for this machine from
[GitHub Releases](https://github.com/Ynewtime/markitai/releases) together with
its `SHA256SUMS`, refuse an archive whose SHA-256 differs, and install for the
current user, without administrator rights:

- macOS and Linux: `markitai` with its `mkai` and `markitai-mcp` links in
  `~/.local/bin`, the notices and offline guides in
  `~/.local/share/markitai/current`. No shell profile is edited; if
  `~/.local/bin` is not on `PATH`, the script says what to add.
- Windows: `markitai.exe`, `mkai.exe` and `markitai-mcp.exe` with the notices
  and guides in `%LOCALAPPDATA%\Programs\markitai`, which is added to the user
  Path (open a new terminal afterwards).

`MARKITAI_VERSION=X.Y.Z` installs a given release, and `MARKITAI_INSTALL_DIR`
another folder, which later runs reuse. On Windows use an empty or dedicated
folder: setup.ps1 refuses one that already holds files of the names it
installs. Setup records what it installed; an update or uninstall removes
only those files. A markitai 1.x that `uv tool` installed is removed once the
new one is in place; `~/.markitai` (configuration, caches, history) is kept. With a terminal, setup offers
`markitai doctor --fix` for the optional components (local OCR models, a
headless browser) whenever one is missing; it keeps no answer, so
`MARKITAI_INSTALL_OPTIONAL=1` or `0` answers without asking. `markitai update` updates it in place (running the same
command again does too). To uninstall, which also keeps `~/.markitai`:

```sh
curl -fsSL https://markitai.dev/setup.sh | sh -s -- --uninstall
```

```powershell
& ([scriptblock]::Create((irm https://markitai.dev/setup.ps1))) -Uninstall
```

With `pip` or `uv` you can instead install the Python package, which carries
the same program and commands: `pip install markitai` or
`uv tool install markitai` (Python 3.10 or later; see
[bindings](bindings.md#installation) for the platforms).

### Release archives

Each [release](https://github.com/Ynewtime/markitai/releases) has one archive
per platform, `SHA256SUMS` and build provenance:

| Platform | Archive |
|---|---|
| macOS on Apple silicon | `markitai-v<version>-aarch64-apple-darwin.tar.gz` |
| macOS on Intel | `markitai-v<version>-x86_64-apple-darwin.tar.gz` |
| Linux x86-64 | `markitai-v<version>-x86_64-unknown-linux-gnu.tar.gz` |
| Linux ARM64 | `markitai-v<version>-aarch64-unknown-linux-gnu.tar.gz` |
| Windows x64 | `markitai-v<version>-x86_64-pc-windows-msvc.zip` |
| Windows ARM64 | `markitai-v<version>-aarch64-pc-windows-msvc.zip` |

Check a download with `sha256sum -c SHA256SUMS --ignore-missing` (on macOS,
`grep <archive> SHA256SUMS | shasum -a 256 -c`) and, with the GitHub CLI, its
origin with `gh attestation verify <archive> --repo Ynewtime/markitai`. To
install one by hand on macOS or Linux, extract it into a new folder and keep
the files together, which preserves both aliases and the offline guides:

```sh
Archive="$HOME/Downloads/markitai-v2.3.0-aarch64-apple-darwin.tar.gz"
Install="$HOME/.local/share/markitai/2.3.0"
mkdir -p "$Install" && tar -xzf "$Archive" -C "$Install"
export PATH="$Install:$PATH"
markitai --version                  # markitai 2.3.0
```

The `export` applies to this terminal; for new terminals add the same line to
`~/.zshrc` for zsh or `~/.bashrc` for bash (`~/.bash_profile` for login bash
terminals). An MCP client can always use the absolute executable path.

On Windows, choose the archive for the system architecture (Settings → System
→ About → System type: x64 for Intel/AMD PCs, ARM64 for ARM-based PCs) and
extract it in PowerShell:

```powershell
$Archive = "$HOME\Downloads\markitai-v2.3.0-aarch64-pc-windows-msvc.zip"
$Install = Join-Path $env:LOCALAPPDATA "Programs\markitai-2.3.0"
Expand-Archive -LiteralPath $Archive -DestinationPath $Install
$env:Path = "$Install;$env:Path"
markitai --version
```

For future terminals, add the folder to your **user** Path in Windows
Environment Variables. The three EXEs need no `.cmd` wrapper.

The CLI needs no Python, Node.js or Go runtime. Unix archives contain one
executable and relative `mkai` / `markitai-mcp` links; Windows ZIPs contain
three EXE entries. License notices and selected offline Markdown guides are
included; OCR models are not. Node.js, Python and Go bindings are separate
packages with their own runtime requirements; see [bindings](bindings.md#installation).

### Rolling back to 1.x

To go back to Python markitai 1.2, which is no longer on PyPI, uninstall as
above, then install the 1.2.0 tag with the PDF library version it was released
with:

```sh
uv tool install "git+https://github.com/Ynewtime/markitai@v1.2.0#subdirectory=packages/markitai" --with "pymupdf4llm<2"
```

### Unsigned builds on macOS

The release executables and wheels are not code-signed with an Apple
Developer ID or notarized. macOS Gatekeeper therefore blocks a copy that a
web browser downloaded (it carries the download "quarantine" mark) and reports
that Apple cannot verify it; an archive fetched with `curl` or `wget` has no
such mark, so the setup script is not affected. Once you are sure the archive
came from the project and its checksum matches `SHA256SUMS`, remove the mark
from that file before extracting it:

```sh
xattr -d com.apple.quarantine markitai-v2.3.0-aarch64-apple-darwin.tar.gz
```

If you have already extracted it, run the same command on
`$Install/markitai` instead. This exempts only that file; it does
not change Gatekeeper for any other program, and there is no need to turn
Gatekeeper off. The same applies to a downloaded `.tgz` or wheel before you
install it.

### Build from source

You need Rust 1.92 or later and a native C/C++ toolchain/linker; development
and native CI use Rust 1.99.1. On macOS install Xcode Command Line Tools;
on Windows use Visual Studio Build Tools with the C++ workload, Windows SDK
and the tools for your MSVC x64 or ARM64 target. Linux needs its C/C++ build
tools. Build from this repository checkout with locked dependency versions:

```sh
cargo build --release --locked -p markitai-cli
```

This produces `target/release/markitai` and `target/release/mkai`
(with `.exe` on Windows). Both use the same core, but their bytes need not match;
the first release build takes several minutes. On Apple silicon, build for an
Intel Mac with `rustup target add x86_64-apple-darwin` and
`cargo build --release --locked --target x86_64-apple-darwin -p markitai-cli`; the
executable lands in `target/x86_64-apple-darwin/release/`. Add the native build
directory to the current shell with
`export PATH="$PWD/target/release:$PATH"` (use the target-specific directory
when cross-building). `markitai mcp` starts MCP directly; a source build does
not create the archive's `markitai-mcp` alias. Maintainers produce archives
with the package drivers described in [native CI](ci.md).

### Platform support

The table describes available backends, rather than promising every format
has equal fidelity on every platform. Windows x64 CI and Windows ARM64/Linux
VMs have run native Rust tests and installed-package checks. Results apply to
the commit and package recorded in each run; physical Intel Macs remain
untested. Consult [native CI](ci.md) for validation scope and
[formats](formats.md) for reader gaps.

| Capability | macOS | Linux x86-64 / ARM64 | Windows x64 / ARM64 |
|---|---|---|---|
| Native text readers, `serve`, CLI and MCP | Available; x86-64 checked under Rosetta 2 | Available | Available |
| Local OCR | Paddle by default; Vision with `MARKITAI_OCR_BACKEND=vision` | Paddle; models must be prepared | Paddle; models must be prepared |
| PDF page rendering | System CoreGraphics; optional portable hayro build | Built-in hayro | Built-in hayro |
| HEIF / HEIC decoding | System ImageIO | Built-in decoder | Built-in decoder |
| AVIF decoding | System ImageIO | Built-in decoder | Built-in decoder |
| JavaScript pages / web screenshots | Needs Chrome/Chromium | Needs Chrome/Chromium | Needs Chrome/Chromium |
| Office page images / vision OCR | Needs LibreOffice | Needs LibreOffice | Needs LibreOffice |
| Batch `--resume` | Available | Available | Available |

macOS binaries declare a minimum of macOS 11.0; testing used newer macOS.
The Linux archives and wheels are built in the manylinux_2_28 image and need
glibc 2.28 or later (RHEL 8, Debian 10, Ubuntu 18.10 and newer); musl-based
Linux (Alpine) is not supported. A Linux build from source can have a
different requirement. Windows CLI archives target MSVC x64
or ARM64 and statically link the MSVC runtime; they carry all three executable entries. This does not make optional
LibreOffice, browser or OCR model installations part of the archive. Neither a
CLI archive nor a passing version probe proves all Node/Python/Go bindings work; Windows Go/cgo
package acceptance remains separate and unverified.

On Apple silicon, use the arm64 archive. Under Rosetta, Vision OCR does not work;
the default Paddle engine does. Run `markitai doctor`
to inspect the backends selected by your build and configuration. It does not
download models. `markitai doctor --fix` explicitly prepares missing Paddle
models and can repair safely managed corrupt models; unsafe model paths are
rejected. Once models are ready, local OCR needs no provider request. See
[local OCR](ocr.md#the-portable-engine-windows-and-linux) for languages and
model preparation, including offline installations.

## 2. Use separate trial state (optional)

Markitai keeps its configuration, caches, browser installation and history in
`~/.markitai`. Point `MARKITAI_HOME` at another directory for separate trial
configuration, caches, browser installation and history. Current-directory
`.env` files and process environment variables can still affect configuration;
changing this directory does not block network access or ignore an existing
`./markitai.json`:

```sh
export MARKITAI_HOME="$PWD/.local/try-home"
```

## 3. Convert your first document

```sh
markitai report.docx              # Markdown with frontmatter on stdout
markitai report.docx --pure       # source body only, no generated frontmatter
markitai report.docx -o out/      # writes out/report.docx.md
markitai report.docx -o notes.md  # writes exactly notes.md
```

For a text file containing a heading, a sentence and a list, standard output
looks like this:

```markdown
---
title: Notes
source: note.txt
markitai_processed: '2026-10-01T09:43:34.187+08:00'
---

# Notes

Hello world. Ünïcode 世界.

- one
- two
```

On stdout, images the document refers to are saved once under
`MARKITAI_HOME/assets/blobs/` and linked with absolute `file://` URIs; see
[images on stdout](images.md#images-on-stdout).

With `-o`, output files keep the source extension (`report.docx.md`), extracted
images go to `out/.markitai/assets/` under content-hash names, and page images
go to `out/.markitai/screenshots/`. Converting again does not overwrite: the
default `output.on_conflict` is `rename`, giving `report.docx.v2.md`. Use
`skip` or `overwrite` instead with
`markitai config set output.on_conflict overwrite`, or for one run with
`--config-json '{"output":{"on_conflict":"overwrite"}}'`.

The Rust CLI also creates `.markitai/ownership` beside output documents for
writer coordination and batch recovery. It is internal metadata, not extracted
content or a cache; `--no-cache` does not disable it. See
[output ownership](output-ownership.md). Keep referenced assets and screenshots
when sharing Markdown; they are separate from this recovery metadata.

Recognized extensions (61, including document and spreadsheet templates):

```text
.avif .bmp .csv .doc .docm .docx .dot .dotm .dotx .eml .epub .gif .heic .heif
.htm .html .ipynb .jpeg .jpg .json .latex .markdown .md .msg .numbers .odp
.ods .odt .org .ots .ott .otp .pdf .png .pot .potm .potx .pps .ppsm .ppsx
.ppt .pptm .pptx .rst .rtf .svg .tex .tif .tiff .tsv .txt .webp .xhtml .xls
.xlsb .xlsm .xlsx .xlt .xltm .xltx .xml
```

Recognition does not guarantee every reader/backend can process every file.
An unsupported extension prints the supported list; platform and fidelity
limits are described in [formats](formats.md).

## 4. Convert a folder or a list of URLs

```sh
markitai ./documents -o out/                  # recursive; keeps relative paths
markitai ./documents -o out/ --dry-run        # list inputs and targets only
markitai ./documents -o out/ -g '**/*.pdf' -j 4
markitai ./documents -o out/ -g '!drafts/**'  # "!" excludes
markitai links.urls -o out/                   # one URL per line
markitai ./documents -o out/ --resume         # continue an interrupted batch
```

Resume keeps compatible completed work; it does not detect every model,
prompt or output-profile change. Use a new output directory without `--resume`
when you want to reprocess everything after changing those settings.

A `.urls` file holds one `URL [output-name]` per line; blank lines and `#`
comments are ignored. Batches print a summary and write a JSON report to
`out/.markitai/reports/`; the exit status is 10 when any item failed. Quote
glob patterns so the shell does not expand them. See [CLI](cli.md) and
[reports](reports.md).

## 5. Web pages

```sh
markitai https://example.com/article -o out/
markitai https://example.com/app -s playwright -o out/   # render with local Chrome
markitai https://example.com/article --screenshot -o out/
```

The default `auto` strategy downloads the page directly and switches to a
locally installed Chrome or Chromium when the page needs JavaScript. A
downloaded PDF goes through the PDF reader. If no browser is found,
`markitai doctor --fix` downloads Google's Chrome for Testing headless shell
into `MARKITAI_HOME/browsers/native` (or `~/.markitai/browsers/native`).
`-s jina` and `-s defuddle` send the URL to those third-party services;
`--no-remote-fetch` or `MARKITAI_NO_REMOTE_FETCH=1` forbids those extraction
services, but source URL requests, browser visits and enabled model requests
still use the network. For offline conversion use local files with `-b native --no-llm`; prepare any OCR models, browser or LibreOffice installation in advance.
There is no global offline/network-sandbox switch. See
[URL fetching](fetch.md) and [browser fetching](browser.md).

## 6. Images, scans and page images

```sh
markitai scan.png --ocr                  # local recognition; prepare Paddle models first when needed
markitai scanned.pdf --ocr -o out/       # recognizes pages without usable text
markitai slides.pdf --screenshot -o out/ # one image per page
```

A standalone image needs `--ocr` or `--llm`: without either there is no text to
extract, so the image is skipped (`skip_reason: "image_only"` with `--json`)
and nothing is printed. OCR languages are set with `ocr.lang`; see
[local OCR](ocr.md) and [PDF page OCR](pdf-ocr.md).

## 7. Enhance with a language model

Model processing is off by default and needs credentials. The quickest setup
uses an API key from the environment:

```sh
export OPENAI_API_KEY=...   # or ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY, OPENROUTER_API_KEY
markitai init --yes         # saves the detected model to the user config; LLM stays off
markitai doctor             # "✓ LLM API: …"; no model request is sent
markitai report.pdf --llm -o out/   # writes out/report.pdf.llm.md
```

Keys can also live in a `.env` file in the current directory or in
`MARKITAI_HOME/.env` (`~/.markitai/.env`); values already in the environment
win. A folder's `.env` may set keys and options, but not programs, paths,
endpoints, proxies, home folders or credential files (see
[configuration](configuration.md)); and a `./markitai.json` found in the
folder never sends an environment key to an address of its own. `init` stores only the model name, never the key, and also sets
`output.dir` to `./output` for folders converted without `-o`. Change the model with
`markitai config set 'llm.model_list[0].litellm_params.model' openai/gpt-4.1-mini`
and turn processing on permanently with `markitai config set llm.enabled true`.
Enhanced output replaces the base file unless you add `--keep-base`. `--alt`
and `--desc` add image captions and descriptions.

`cost_usd` is computed only for the models in the [price catalog](pricing.md);
other models report an unknown cost rather than zero. Routing, retries,
fallbacks and request limits are in [LLM](llm.md); subscription runtimes
(GitHub Copilot, Claude, ChatGPT) are in [subscriptions](subscriptions.md).

## 8. Browser workspace and REST API

```sh
markitai serve              # http://127.0.0.1:3600, opens your browser
markitai serve --port 3700 --no-open
```

The page converts uploads and URLs, previews results, keeps history and edits
model connections. Open the startup URL including its `#token=` fragment; API
requests require the token even from localhost unless you explicitly selected
`--no-auth`. Treat that URL like a password. The same jobs are available over REST:

```sh
# Set MARKITAI_SERVE_TOKEN to the token printed by the running server.
curl -H "Authorization: Bearer $MARKITAI_SERVE_TOKEN" \
  -F files=@note.txt -F 'options={}' http://127.0.0.1:3600/api/jobs
```

See [REST service](serve.md) and [workspace](web-ui.md).

## 9. Use it from an AI assistant (MCP)

`markitai mcp` (or Unix `markitai-mcp` / Windows `markitai-mcp.exe`) serves the tools
`convert_document`, `convert_url`, `batch_convert` and `job_status` over
stdin/stdout. A typical MCP client entry:

```json
{
  "mcpServers": {
    "markitai": {
      "command": "/absolute/path/to/markitai",
      "args": ["mcp"],
      "env": {"MARKITAI_HOME": "/absolute/path/to/separate-state"}
    }
  }
}
```

Use the actual executable path (`markitai.exe` on Windows) and native absolute
paths in the configuration. Plain conversion does not need a provider key.
Paths passed to document tools must be absolute; `~` is expanded. For structured
CLI output use `markitai report.docx -o out/ --json`. See [MCP](mcp.md) and the
[standalone Agent guide](../llms-full.txt).

## 10. Next steps

- [Configuration](configuration.md): file locations, commands and environment variables.
- [Troubleshooting](troubleshooting.md): common errors and exit codes.
- [CLI](cli.md): commands, options and output behavior.
- [Documentation index](index.md): all topics.

The CLI archive includes this guide, [CLI](cli.md), [MCP](mcp.md), an
[index](index.md) and the root `llms.txt` / `llms-full.txt` for offline reading.
Other topic links refer to the full repository documentation; they are not
all copied into the archive.
