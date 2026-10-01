# Markitai

Markitai converts documents, web pages and images to Markdown. A single Rust
core powers the `markitai` command (alias `mkai`), a local REST service with a
browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. The command is one executable with no Python, Node.js or Go runtime.

This repository is the Rust rewrite of Markitai 1.2.0. Version 1.3.0-dev is a
development build: it is not release-ready and no packages are published yet.
[Project status](docs/STATUS.md) lists what is delivered and verified, and
[compatibility](docs/compatibility.md) the reference contracts.

## Quick start

```sh
cargo build --release -p markitai-cli         # Rust 1.89 or later
export PATH="$PWD/target/release:$PATH"

markitai report.docx                          # Markdown on stdout
markitai report.docx -o out/                  # writes out/report.docx.md
markitai ./documents -o out/                  # whole folder, with a report
markitai https://example.com/article -o out/  # web page
markitai scan.png --ocr                       # on-device OCR (macOS)
markitai doctor                               # check optional components
markitai serve                                # browser workspace on 127.0.0.1:3600
```

Set `MARKITAI_HOME` to keep configuration, caches and history in a directory
other than `~/.markitai`, for example while trying it out:
`export MARKITAI_HOME="$PWD/.local/try-home"`.

Model enhancement (`--llm`) needs an API key or a configured model; see the
[quick start](docs/quickstart.md#7-enhance-with-a-language-model).

## Documentation

- [Quick start](docs/quickstart.md): installation, first conversions, models,
  workspace and MCP.
- [Troubleshooting](docs/troubleshooting.md).
- [Documentation index](docs/index.md): CLI, configuration, formats, services
  and bindings.
- [Development](docs/development.md): building, testing and contributing.

## License

MIT; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
