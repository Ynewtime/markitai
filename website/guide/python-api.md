# Python API

Markitai can be used as a library, without the CLI. `markitai.convert()` and its async twin `markitai.aconvert()` run the same conversion pipeline the CLI uses — including LLM enhancement — and return a typed result.

::: warning Provisional
The Python API is **provisional** while markitai is 0.x: signatures and result fields may change in minor releases.
:::

```bash
uv add markitai   # or: pip install markitai
```

## Basic Conversion

```python
import markitai

# In-memory: convert and get the Markdown string back
out = markitai.convert("report.pdf")
print(out.markdown)

# Write outputs (and extracted image assets) to a directory
out = markitai.convert("report.pdf", output_dir="out/")
print(out.output_path)  # out/report.pdf.md
print(out.assets)  # images under out/.markitai/assets/

# URLs work the same way
out = markitai.convert("https://example.com/article")
```

Without `output_dir`, the conversion runs in a private temp directory that is deleted afterwards: you get the Markdown in memory, path fields are `None`, and image links keep their relative `.markitai/assets/...` form. Pass `output_dir` whenever you need the image files.

Library calls keep `stdout` clean — parser noise from PyMuPDF/ONNX is suppressed at the `convert()` entry point, no CLI initialization required. Diagnostics go through [loguru](https://github.com/Delgan/loguru) on stderr; silence them with `logger.disable("markitai")`.

## LLM-Enhanced Conversion

`llm=True` runs the full enhancement pipeline (cleanup + frontmatter) and returns both variants:

```python
import os
import markitai

os.environ["MODEL"] = "openai/gpt-4o-mini"  # or configure llm.model_list

out = markitai.convert("report.pdf", output_dir="out/", llm=True)
print(out.llm_markdown)  # enhanced body
print(out.frontmatter["title"])  # parsed YAML frontmatter
print(out.usage.cost_usd)  # LLM spend for this conversion
```

Models resolve exactly like in the CLI: `llm.model_list` from your [configuration](/guide/configuration) first, then the `MODEL` environment variable. With `config=None` (the default) the same config file hierarchy as the CLI is loaded; pass a `markitai.MarkitaiConfig` instance for full programmatic control:

```python
from markitai import MarkitaiConfig

cfg = MarkitaiConfig()  # pure defaults, ignores config files
cfg.llm.pure = True  # raw LLM cleanup, no frontmatter
out = markitai.convert("notes.docx", config=cfg, llm=True)
```

Keyword toggles (`llm`, `ocr`, `screenshot`, `alt`, `desc`) override the config, mirroring the CLI flags; `None` keeps the configured value.

## Async Usage

Inside an event loop, use `aconvert` — CPU-bound converter work (PyMuPDF, ONNX, Office extraction) runs in a shared thread pool, so the loop stays responsive:

```python
import asyncio
import markitai


async def main() -> None:
    results = await asyncio.gather(
        markitai.aconvert("a.pdf", output_dir="out/"),
        markitai.aconvert("https://example.com/b", output_dir="out/"),
    )
    for out in results:
        print(out.source, "->", out.output_path or len(out.markdown))


asyncio.run(main())
```

Calling the sync `convert()` from a running loop raises `RuntimeError`. In long-lived apps, call `await markitai.fetch.close_shared_clients()` on shutdown to release shared HTTP clients (the sync `convert()` does this automatically).

## ConversionOutput

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | Input path or URL as given |
| `markdown` | `str` | Base converted Markdown body (frontmatter stripped) |
| `llm_markdown` | `str \| None` | LLM-enhanced body, `None` without LLM |
| `frontmatter` | `dict` | Parsed YAML frontmatter of the richest output |
| `output_path` | `Path \| None` | Written base `.md` file |
| `llm_output_path` | `Path \| None` | Written `.llm.md` file |
| `assets` | `list[Path]` | Extracted image files |
| `screenshots` | `list[Path]` | Rendered page/screenshot files |
| `images` | `list[dict]` | Per-image LLM analysis entries (alt/description) |
| `usage` | `ConversionUsage` | `cost_usd`, token totals, per-model breakdown |
| `skip_reason` | `str \| None` | `"exists"` when skipped by conflict policy |
| `duration` | `float` | Wall-clock seconds |

Failures raise instead of returning partial results: `ConversionError` for pipeline failures, `FetchError` for unreachable URLs, `ValueError` when LLM is enabled without a resolvable model. One call converts one file or URL; directory batches remain CLI territory.
