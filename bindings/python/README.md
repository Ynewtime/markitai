# markitai

Markitai converts documents, web pages and images to Markdown. This package
carries its Rust engine as a native extension: the `markitai` command (alias
`mkai`), the MCP server `markitai-mcp` and a Python API, with no other Python
dependency. Python 3.10 or later on macOS 11+ (Apple silicon and Intel),
Linux with glibc 2.28+ (x86-64 and ARM64) and Windows (x64 and ARM64).

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
