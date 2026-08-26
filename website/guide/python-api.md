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

os.environ["MODEL"] = "openai/gpt-5.4-nano"  # or configure llm.model_list

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

Keyword toggles (`llm`, `ocr`, `screenshot`, `alt`, `desc`) override the config, mirroring the CLI flags; `None` keeps the configured value. URL conversions with a screenshot and multi-source content automatically get vision enhancement (the screenshot guides the LLM), and `screenshot.screenshot_only` reads the page image(s) directly — same as the CLI. `profile="rag" | "obsidian" | "okf"` mirrors `--profile` and shapes the written output for a downstream consumer — see [Output Profiles](./output-profiles.md) for the LlamaIndex ingestion recipe.

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

::: tip Short-lived host processes
markitdown pulls Magika, and Magika pulls onnxruntime, into every markitai process — even one that only converts a `.txt`. Under load onnxruntime's teardown can abort at interpreter shutdown, turning a conversion that already wrote its output into exit code 134. A one-shot script can end with `from markitai.utils.shutdown import finalize_process; finalize_process(0)`, which cleans up markitai's temp directories, flushes both streams and leaves without that teardown. The `markitai` CLI already exits this way. A long-lived host should not use it — it takes the whole process down.
:::

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
