# markitai

Markitai converts documents, web pages and images to Markdown. This package
carries its Rust engine as a native extension: the `markitai` command (alias
`mkai`), the MCP server `markitai-mcp` and a Python API, with no other Python
dependency. Python 3.10 or later on macOS 11+ (Apple silicon and Intel),
Linux with glibc 2.28+ (x86-64 and ARM64) and Windows (x64 and ARM64).
PyPI has wheels for these six platforms only. Elsewhere (musl/Alpine, an older
glibc or macOS, 32-bit Windows, other CPUs, free-threaded Python) pip finds no
2.x release it can install, so it installs or keeps markitai 1.x (on Python
3.11-3.13). To use 2.x there, build it from the source archive on the
[GitHub Release](https://github.com/Ynewtime/markitai/releases):
`pip install markitai-<version>.tar.gz` needs Rust 1.92 or later and a long,
memory-hungry build.

```sh
pip install markitai          # or: uv tool install markitai
markitai report.docx -o out/  # writes out/report.docx.md
markitai ./documents -o out/  # a whole folder, with a report
markitai serve                # browser workspace on 127.0.0.1:3600
markitai doctor               # optional components (OCR models, a browser)
```

```python
import markitai

result = markitai.convert("report.docx")
print(result.markdown)
```

`await markitai.aconvert(...)` runs the same conversion without blocking an
event loop.

## MCP

`markitai-mcp` (or `markitai mcp`) serves four tools over stdio:
`convert_document`, `convert_url`, `batch_convert` and `job_status`. An MCP
host can start it with `uvx --from markitai markitai mcp`.

<!-- mcp-name: io.github.Ynewtime/markitai -->

## Without Python

The same program ships as a standalone executable for each platform, with
setup scripts, on the [GitHub Releases](https://github.com/Ynewtime/markitai/releases)
page. Documentation: [markitai.dev](https://markitai.dev) and the
[repository](https://github.com/Ynewtime/markitai).
