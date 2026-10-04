# Quick start

Markitai converts documents, web pages and images to Markdown. One Rust core
powers the `markitai` command (also installed as `mkai`), a local REST service
with a browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. This guide covers development version **1.3.0-dev**, the Rust rewrite of
reference release 1.2.0. Stable 1.3.0 has not been published.

## 1. Install

Choose the archive supplied for your machine. `1.3.0-dev` is a development
version; stable 1.3.0 has not been published. The names below describe packages
produced by the project, not links to an available stable release.

| Platform | Development archive |
|---|---|
| macOS on Apple silicon | `markitai-1.3.0-dev-aarch64-apple-darwin-single-binary.tar.gz` |
| macOS on Intel | `markitai-1.3.0-dev-x86_64-apple-darwin-single-binary.tar.gz` |
| Linux x86-64 | `markitai-1.3.0-dev-x86_64-unknown-linux-gnu-single-binary.tar.gz` |
| Windows x64 | `markitai-1.3.0-dev-x86_64-pc-windows-msvc.zip` |
| Windows ARM64 | `markitai-1.3.0-dev-aarch64-pc-windows-msvc.zip` |

On macOS or Linux, replace `Archive` with the path to your downloaded archive.
Use a new installation directory; if `mkdir` reports that it exists, choose a
new directory before continuing. Keeping the files together preserves both
aliases and the offline documentation:

```sh
Archive="/path/to/markitai-1.3.0-dev-aarch64-apple-darwin-single-binary.tar.gz"
Install="$HOME/.local/share/markitai/1.3.0-dev"
mkdir -p "$HOME/.local/share/markitai"
mkdir "$Install" && tar -xzf "$Archive" -C "$Install"
export PATH="$Install:$PATH"
markitai --version                  # markitai 1.3.0-dev
mkai --help
markitai-mcp --help
```

The `export` makes the commands available in this terminal immediately; it
does not rely on `~/.local/bin` already being on `PATH`. To keep them available
in new terminals, add `export PATH="$HOME/.local/share/markitai/1.3.0-dev:$PATH"`
once to `~/.zshrc` for zsh or `~/.bashrc` for non-login bash terminals
(`~/.bash_profile` for login bash terminals). Use the actual installation
directory if you changed it. An MCP client can always use the
absolute executable path, independent of shell profiles.

On Windows, select x64 or ARM64 and use PowerShell. Replace `Archive` with your
ZIP path; choose a new `Install` directory if it already exists:

```powershell
$Archive = "C:\Downloads\markitai-1.3.0-dev-aarch64-pc-windows-msvc.zip"
$Install = Join-Path $env:LOCALAPPDATA "Programs\Markitai-1.3.0-dev"
if (Test-Path -LiteralPath $Install) { throw "Choose a new installation directory" }
Expand-Archive -LiteralPath $Archive -DestinationPath $Install
& "$Install\markitai.exe" --version
$env:Path = "$Install;$env:Path"
mkai.exe --help
markitai-mcp.exe --help
```

These are three direct EXE entries, with no `.cmd` wrapper or administrator
installation. The PATH assignment applies to this terminal. For future
terminals, add the installation directory to your **user** Path in Windows
Environment Variables, then open a new terminal.

The CLI needs no Python, Node.js or Go runtime. Unix archives contain one
executable and relative `mkai` / `markitai-mcp` links; Windows ZIPs contain
three EXE entries. License notices and selected offline Markdown guides are
included; OCR models are not. Node.js, Python and Go bindings are separate
packages with their own runtime requirements; see [bindings](bindings.md#installation).

### Unsigned builds on macOS

The development executables and packages are not code-signed with an Apple
Developer ID or notarized yet. macOS Gatekeeper therefore blocks a copy that a
web browser downloaded (it carries the download "quarantine" mark) and reports
that Apple cannot verify it; an archive fetched with `curl` or `wget` has no
such mark. Once you are sure the archive came from the project and its checksum matches
the supplied package record, remove the mark from that file before extracting it:

```sh
xattr -d com.apple.quarantine markitai-1.3.0-dev-aarch64-apple-darwin-single-binary.tar.gz
```

If you have already extracted it, run the same command on
`$Install/markitai` instead. This exempts only that file; it does
not change Gatekeeper for any other program, and there is no need to turn
Gatekeeper off. The same applies to a downloaded `.tgz` or wheel before you
install it.

### Build from source

You need Rust 1.92 or later (development uses current stable):

```sh
cargo build --release -p markitai-cli
```

This produces `target/release/markitai` and the identical `target/release/mkai`;
the first release build takes several minutes. On Apple silicon, build for an
Intel Mac with `rustup target add x86_64-apple-darwin` and
`cargo build --release --target x86_64-apple-darwin -p markitai-cli`; the
executable lands in `target/x86_64-apple-darwin/release/`. Add the native build
directory to the current shell with
`export PATH="$PWD/target/release:$PATH"` (use the target-specific directory
when cross-building). `markitai mcp` starts MCP directly; a source build does
not create the archive's `markitai-mcp` alias. Maintainers produce archives
with the package drivers described in [native CI](ci.md).

### Platform support

The table describes available backends, rather than promising every format
has equal fidelity on every platform. Windows x64 CI and Windows ARM64/Linux
VMs have run native Rust tests and installed-package checks; physical Intel
Macs remain untested. Consult [native CI](ci.md) for each recorded scope and
[formats](formats.md) for reader gaps.

| Capability | macOS | Linux x86-64 | Windows x64 / ARM64 |
|---|---|---|---|
| Native text readers, `serve`, CLI and MCP | Available; x86-64 checked under Rosetta 2 | Available | Available |
| Local OCR | Vision by default; optional portable Paddle build | Paddle; models must be prepared | Paddle; models must be prepared |
| PDF page rendering | System CoreGraphics; optional portable hayro build | Built-in hayro | Built-in hayro |
| HEIF / AVIF decoding | System ImageIO | Explicit unsupported error | Explicit unsupported error |
| JavaScript pages / web screenshots | Needs Chrome/Chromium | Needs Chrome/Chromium | Needs Chrome/Chromium |
| Office page images / OCR | Needs LibreOffice | Needs LibreOffice | Needs LibreOffice |
| Batch `--resume` | Available | Available | Available |

macOS binaries declare a minimum of macOS 11.0; testing used newer macOS.
The Ubuntu 24.04 Linux archives require glibc 2.39 or later; another Linux
build can have a different requirement. The Windows archives target MSVC64
and carry their required executable entries. Neither a CLI archive nor a
passing version probe proves all Node/Python/Go bindings work; Windows Go/cgo
package acceptance remains separate and unverified.

On Apple silicon, use the arm64 archive. The default x86-64 build under Rosetta
cannot use Vision OCR; portable builds can use Paddle. Run `markitai doctor`
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
changing this directory does not block network access:

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
`--no-remote-fetch` or `MARKITAI_NO_REMOTE_FETCH=1` forbids that. See
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
markitai doctor             # "LLM API: ok"; no model request is sent
markitai report.pdf --llm -o out/   # writes out/report.pdf.llm.md
```

Keys can also live in a `.env` file in the current directory or in
`MARKITAI_HOME/.env` (`~/.markitai/.env`); values already in the environment
win. `init` stores only the model name, never the key, and also sets
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
model connections. The same jobs are available over REST:

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
- [CLI](cli.md): every option and subcommand.
- [Documentation index](index.md): all topics.

The CLI archive includes this guide, [CLI](cli.md), [MCP](mcp.md), an
[index](index.md) and the root `llms.txt` / `llms-full.txt` for offline reading.
Other topic links refer to the full repository documentation; they are not
all copied into the archive.
