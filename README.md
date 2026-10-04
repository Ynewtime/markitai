# Markitai

Markitai converts documents, web pages and images to Markdown. A single Rust
core powers the `markitai` command (alias `mkai`), a local REST service with a
browser workspace, an MCP server, and in-process Node.js, Python and Go
bindings. The CLI executable needs no Python, Node.js or Go runtime; language
bindings need their respective language runtime.

This is **1.3.0-dev**, the Rust rewrite of reference release 1.2.0, available
for local verification. Stable 1.3.0 is not published. See the
[changelog](CHANGELOG.md), [current status](docs/STATUS.md) and
[compatibility contracts](docs/compatibility.md).

## Install

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

On Windows, choose x64 for Intel/AMD or ARM64 for an ARM-based PC (check
Settings → System → About → System type), then use PowerShell. Replace `Archive` with your
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

The archives include license notices, `llms.txt`, `llms-full.txt` and selected
Markdown guides in `docs/` for offline reading. They do not include OCR models.
For builds from source, platform requirements, optional components and unsigned
macOS downloads, see the [quick start](docs/quickstart.md#1-install). Language
packages have separate installation steps in [bindings](docs/bindings.md#installation).

## Quick start

```sh
markitai report.docx -o out/                  # writes out/report.docx.md
markitai ./documents -o out/                  # whole folder, with a report
markitai scan.png --ocr -o out/               # local OCR; models may need preparation
markitai report.docx -o out/ --json           # structured result for automation
markitai doctor                              # inspect optional components
markitai --help                              # commands, options and exit codes
markitai serve                               # browser workspace on 127.0.0.1:3600
```

For a separate trial state, set `MARKITAI_HOME` before using the CLI:
`export MARKITAI_HOME="$PWD/.local/try-home"`. Plain document conversion does
not need a model connection. Existing configuration still applies; use
`--no-llm -b native` for local document processing without model enhancement.
URL inputs still need the network, even with `--no-remote-fetch`. Local OCR uses macOS Vision by default on macOS,
and Paddle on Windows and Linux; `doctor --fix` can prepare missing Paddle
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
