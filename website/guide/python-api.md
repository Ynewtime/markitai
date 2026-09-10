# Python API

Use markitai as a library. `markitai.convert()` and its async twin `markitai.aconvert()` run the same pipeline as the CLI, LLM enhancement included, and return a typed result.

::: warning API stability
The Python API is provisional: signatures and result fields may still change in minor releases. The CLI is the stable interface. Pin an exact version if you depend on a library detail.
:::

```bash
uv add markitai   # or: pip install markitai
```

## Basic Conversion

```python
import markitai

# In memory: get the Markdown string back
out = markitai.convert("report.pdf")
print(out.markdown)

# Write outputs and extracted images to a directory
out = markitai.convert("report.pdf", output_dir="out/")
print(out.output_path)  # out/report.pdf.md
print(out.assets)       # images under out/.markitai/assets/

# URLs work the same way
out = markitai.convert("https://example.com/article")
```

Without `output_dir` the conversion runs in a temporary directory that is deleted afterwards. You get the Markdown in memory, path fields are `None`, and image links keep their relative `.markitai/assets/...` form. Pass `output_dir` whenever you need the image files.

Library calls keep `stdout` clean. Diagnostics go through [loguru](https://github.com/Delgan/loguru) on stderr; silence them with `logger.disable("markitai")`.

## LLM-Enhanced Conversion

`llm=True` runs cleanup and frontmatter generation and returns both variants:

```python
import os
import markitai

os.environ["MODEL"] = "openai/gpt-5.6-luna"  # or configure llm.model_list

out = markitai.convert("report.pdf", output_dir="out/", llm=True)
print(out.llm_markdown)          # enhanced body
print(out.frontmatter["title"])  # parsed YAML frontmatter
print(out.usage.cost_usd)        # LLM spend for this conversion
```

Models resolve as in the CLI: `llm.model_list` from your [configuration](/guide/configuration) first, then the `MODEL` environment variable. By default the same config files as the CLI are loaded. Pass a `markitai.MarkitaiConfig` for full control:

```python
from markitai import MarkitaiConfig

cfg = MarkitaiConfig()   # pure defaults, ignores config files
cfg.llm.pure = True      # raw LLM cleanup, no frontmatter
out = markitai.convert("notes.docx", config=cfg, llm=True)
```

The keyword toggles `llm`, `ocr`, `screenshot`, `alt` and `desc` mirror the CLI flags and override the config; `None` keeps the configured value. `profile="rag" | "obsidian" | "okf"` mirrors `--profile`, see [Output Profiles](./output-profiles.md).

## Async Usage

Inside an event loop, use `aconvert`. CPU-bound work runs in a thread pool, so the loop stays responsive:

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

Calling the sync `convert()` from a running loop raises `RuntimeError`. Long-lived apps should call `await markitai.fetch.close_shared_clients()` on shutdown to release shared HTTP clients.

A short-lived script can occasionally end with exit code 134 after a successful conversion, an onnxruntime teardown quirk. End such scripts with `from markitai.utils.shutdown import finalize_process; finalize_process(0)` to avoid it. Do not use it in a long-lived host, since it takes the whole process down.

## ConversionOutput

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | Input path or URL as given |
| `markdown` | `str` | Base Markdown body (frontmatter stripped) |
| `llm_markdown` | `str \| None` | LLM-enhanced body, `None` without LLM |
| `frontmatter` | `dict` | Parsed YAML frontmatter of the richest output |
| `output_path` | `Path \| None` | Written base `.md` file |
| `llm_output_path` | `Path \| None` | Written `.llm.md` file |
| `assets` | `list[Path]` | Extracted image files |
| `screenshots` | `list[Path]` | Rendered page or screenshot files |
| `images` | `list[dict]` | Per-image LLM analysis entries (alt, description) |
| `usage` | `ConversionUsage` | `cost_usd`, token totals, per-model breakdown |
| `skip_reason` | `str \| None` | `"exists"` when skipped by the conflict policy |
| `duration` | `float` | Wall-clock seconds |

Failures raise instead of returning partial results: `ConversionError` for pipeline failures, `FetchError` for unreachable URLs, `ValueError` when LLM is enabled without a model. One call converts one file or URL; directory batches stay with the CLI.
