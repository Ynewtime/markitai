# Markitai

Markitai converts documents, web pages and images to Markdown. A single Rust
core powers the `markitai` command (alias `mkai`), a local REST service with a
browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. The command is one executable with no Python, Node.js or Go runtime.

Markitai 1.3.0 is the Rust rewrite of Markitai 1.2.0. The
[changelog](CHANGELOG.md) lists what changed, [project status](docs/STATUS.md)
what is delivered and verified on which platforms, and
[compatibility](docs/compatibility.md) the reference contracts.

## Install

Each release archive holds one executable, relative `mkai` and `markitai-mcp`
links and the license notices:

| Platform | Archive |
|---|---|
| macOS on Apple silicon | `markitai-1.3.0-aarch64-apple-darwin-single-binary.tar.gz` |
| macOS on Intel (tested under Rosetta 2 only) | `markitai-1.3.0-x86_64-apple-darwin-single-binary.tar.gz` |
| Linux x86-64 with glibc 2.39 or later | `markitai-1.3.0-x86_64-unknown-linux-gnu-single-binary.tar.gz` |

```sh
mkdir -p ~/.local/share/markitai ~/.local/bin
tar -xzf markitai-1.3.0-aarch64-apple-darwin-single-binary.tar.gz -C ~/.local/share/markitai
ln -sf ~/.local/share/markitai/markitai ~/.local/bin/markitai
markitai --version                            # markitai 1.3.0
```

The builds are not code-signed or notarized yet. If macOS refuses to run a
copy downloaded with a browser, see
[unsigned builds on macOS](docs/quickstart.md#unsigned-builds-on-macos).
The Node.js package (`markitai-1.3.0.tgz`), the Python wheel
(`markitai-1.3.0-cp310-abi3-<platform>.whl`) and the static Go packages for
macOS arm64 and Linux x86-64 are installed as described in
[bindings](docs/bindings.md#installation). To build from source instead, see
the [quick start](docs/quickstart.md#build-from-source).

## Quick start

```sh
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
