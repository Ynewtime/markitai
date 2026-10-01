# Quick start

Markitai converts documents, web pages and images to Markdown. One Rust core
powers the `markitai` command (also installed as `mkai`), a local REST service
with a browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. This guide covers version 1.3.0, the Rust rewrite of reference
release 1.2.0.

## 1. Install

Markitai 1.3.0 is distributed as one archive per platform. Each holds the
`markitai` executable, relative `mkai` and `markitai-mcp` links and all license
notices; the executable needs no Python, Node.js or Go runtime.

| Platform | Archive | Executable |
|---|---|---|
| macOS on Apple silicon | `markitai-1.3.0-aarch64-apple-darwin-single-binary.tar.gz` | about 22 MB |
| macOS on Intel (tested under Rosetta 2 only) | `markitai-1.3.0-x86_64-apple-darwin-single-binary.tar.gz` | about 25 MB |
| Linux x86-64 with glibc 2.39 or later | `markitai-1.3.0-x86_64-unknown-linux-gnu-single-binary.tar.gz` | about 26 MB |

The archives are about 12 MB each. The macOS builds declare macOS 11.0 as their
minimum but have been run only on macOS 27; the Linux build is linked against
glibc 2.39 (Ubuntu 24.04, also Debian 13 and later) and has been tested on
Ubuntu 24.04. The files sit at the top of the archive, so extract it into a
directory of its own and put the commands on your `PATH`:

```sh
mkdir -p ~/.local/share/markitai ~/.local/bin
tar -xzf markitai-1.3.0-aarch64-apple-darwin-single-binary.tar.gz -C ~/.local/share/markitai
ln -sf ~/.local/share/markitai/markitai ~/.local/bin/markitai
ln -sf ~/.local/share/markitai/mkai ~/.local/bin/mkai                  # optional short name
ln -sf ~/.local/share/markitai/markitai-mcp ~/.local/bin/markitai-mcp  # optional MCP launcher name
markitai --version   # markitai 1.3.0
```

If `markitai` is not found, add `~/.local/bin` to `PATH` in your shell profile.
On Apple silicon use the arm64 archive: the x86-64 build also runs there under
Rosetta 2, but cannot use local OCR. The Node.js package, the Python wheel and
the Go packages are covered in [bindings](bindings.md#installation).

### Unsigned builds on macOS

The 1.3.0 executables and packages are not code-signed with an Apple
Developer ID or notarized yet. macOS Gatekeeper therefore blocks a copy that a
web browser downloaded (it carries the download "quarantine" mark) and reports
that Apple cannot verify it; an archive fetched with `curl` or `wget` has no
such mark. Once you are sure the archive is the one published for the
release, remove the mark from that file before extracting it:

```sh
xattr -d com.apple.quarantine markitai-1.3.0-aarch64-apple-darwin-single-binary.tar.gz
```

If you have already extracted it, run the same command on
`~/.local/share/markitai/markitai` instead. This exempts only that file; it does
not change Gatekeeper for any other program, and there is no need to turn
Gatekeeper off. The same applies to a downloaded `.tgz` or wheel before you
install it.

### Build from source

You need Rust 1.89 or later (development uses current stable):

```sh
cargo build --release -p markitai-cli
```

This produces `target/release/markitai` and the identical `target/release/mkai`;
the first release build takes several minutes. On Apple silicon, build for an
Intel Mac with `rustup target add x86_64-apple-darwin` and
`cargo build --release --target x86_64-apple-darwin -p markitai-cli`; the
executable lands in `target/x86_64-apple-darwin/release/`. Link the executable
into `~/.local/bin` as above. Maintainers produce the release archives with the
package drivers described in [native CI](ci.md).

### Platform support

| Capability | macOS arm64 | macOS x86-64 (Intel) | Linux x86-64 | Windows |
|---|---|---|---|---|
| Document, web-page, e-mail and data conversion; `serve`; `mcp`; bindings | Tested | CLI archive and Rust tests checked under Rosetta 2 only; no binding packages | Tested (Ubuntu 24.04 under OrbStack emulation), including the static Go package | No release build; type-checks only, never linked or run |
| Local OCR, PDF page images, HEIF/AVIF images | Built in (system frameworks) | Built in, not tested on Intel hardware; under Rosetta 2 OCR fails with an explicit error | Explicit "unsupported" error | Unsupported |
| JavaScript pages and web screenshots (`-s playwright`, `--screenshot`) | Needs Chrome/Chromium | Not tested | Needs Chrome/Chromium (not exercised in the Linux rounds) | Not tested |
| Office page screenshots | Needs LibreOffice | Not tested | Not available | Not available |
| Batch `--resume`, subscription models | Supported | Supported (tests under Rosetta 2) | Supported | Not available |

Physical Intel Macs and physical Intel/AMD Linux machines have not been tested.
On Apple silicon, an x86-64 build running under Rosetta 2 cannot use local OCR,
and `markitai doctor` says so ([details](validation/macos-x86_64-rosetta.md)).
Run `markitai doctor` to see which optional pieces are present on your machine.

## 2. Try it without touching your real settings (optional)

Markitai keeps its configuration, caches, browser installation and history in
`~/.markitai`. Point `MARKITAI_HOME` at another directory to keep a trial
completely separate:

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

Recognized extensions are `.avif .bmp .csv .doc .docm .docx .eml .epub .gif
.heic .heif .htm .html .ipynb .jpeg .jpg .json .latex .markdown .md .msg
.numbers .odp .ods .odt .org .pdf .png .pot .pps .ppsm .ppsx .ppt .pptm .pptx
.rst .rtf .svg .tex .tif .tiff .tsv .txt .webp .xhtml .xls .xlsb .xlsm .xlsx
.xml`. An unsupported file prints this list. Reader quality and known gaps are
described in [formats](formats.md).

## 4. Convert a folder or a list of URLs

```sh
markitai ./documents -o out/                  # recursive; keeps relative paths
markitai ./documents -o out/ --dry-run        # list inputs and targets only
markitai ./documents -o out/ -g '**/*.pdf' -j 4
markitai ./documents -o out/ -g '!drafts/**'  # "!" excludes
markitai links.urls -o out/                   # one URL per line
markitai ./documents -o out/ --resume         # continue an interrupted run (Unix)
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
markitai scan.png --ocr                  # macOS: on-device text recognition
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
curl -F files=@note.txt -F 'options={}' http://127.0.0.1:3600/api/jobs
```

See [REST service](serve.md) and [workspace](web-ui.md).

## 9. Use it from an AI assistant (MCP)

`markitai mcp` (or the `markitai-mcp` link) serves the tools
`convert_document`, `convert_url`, `batch_convert` and `job_status` over
stdin/stdout. A typical MCP client entry:

```json
{
  "mcpServers": {
    "markitai": {
      "command": "/absolute/path/to/markitai",
      "args": ["mcp"],
      "env": {"OPENAI_API_KEY": "..."}
    }
  }
}
```

Paths passed to the tools must be absolute. See [MCP](mcp.md).

## 10. Next steps

- [Configuration](configuration.md): file locations, commands and environment variables.
- [Troubleshooting](troubleshooting.md): common errors and exit codes.
- [CLI](cli.md): every option and subcommand.
- [Documentation index](index.md): all topics.
