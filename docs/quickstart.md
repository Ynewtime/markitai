# Quick start

Markitai converts documents, web pages and images to Markdown. One Rust core
powers the `markitai` command (also installed as `mkai`), a local REST service
with a browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. This guide covers version 1.3.0-dev, the development build of the
rewrite of reference release 1.2.0.

## 1. Get the command

No release packages are published yet; build from a checkout. You need Rust
1.89 or later (development uses current stable).

```sh
cargo build --release -p markitai-cli
```

This produces `target/release/markitai` and the identical `target/release/mkai`
(about 23 MB each on macOS arm64; the first release build takes several
minutes). Put the executable on your `PATH`, for example:

```sh
mkdir -p ~/.local/bin
ln -sf "$PWD/target/release/markitai" ~/.local/bin/markitai
ln -sf "$PWD/target/release/markitai" ~/.local/bin/mkai          # optional short name
ln -sf "$PWD/target/release/markitai" ~/.local/bin/markitai-mcp  # optional MCP launcher name
markitai --version   # markitai 1.3.0-dev
```

The executable needs no Python, Node.js or Go runtime. Maintainers can produce
the self-contained archive `markitai-<version>-<host>-single-binary.tar.gz`
(one executable, relative `mkai`/`markitai-mcp` links and all license notices)
with the package driver described in [native CI](ci.md). For the language
bindings see [bindings](bindings.md#installation).

### Platform support

| Capability | macOS arm64 | Linux x86-64 | Windows |
|---|---|---|---|
| Document, web-page, e-mail and data conversion; `serve`; `mcp`; bindings | Tested | Tested (Ubuntu under OrbStack emulation) | Type-checks only; never linked or run |
| Local OCR, PDF page images, HEIF/AVIF images | Built in (system frameworks) | Explicit "unsupported" error | Unsupported |
| JavaScript pages and web screenshots (`-s playwright`, `--screenshot`) | Needs Chrome/Chromium | Needs Chrome/Chromium (not exercised in the Linux rounds) | Not tested |
| Office page screenshots | Needs LibreOffice | Not available | Not available |
| Batch `--resume`, subscription models | Supported | Supported | Not available |

Intel Macs and physical Intel/AMD Linux machines have not been tested. Run
`markitai doctor` to see which optional pieces are present on your machine.

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
