# Python API

`markitai.convert()` and its async twin `markitai.aconvert()` run the CLI's pipeline, LLM enhancement included, and return a typed result.

::: warning API stability
The Python API is provisional: signatures and fields can change in minor releases. Pin an exact version; the CLI is the stable interface.
:::

Add it to your project (the [setup script](/guide/getting-started) installs a command, not an importable library). Extras work as in the CLI, e.g. `uv add "markitai[ocr]"`:

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
print(out.assets)  # images under out/.markitai/assets/

# URLs work the same way
out = markitai.convert("https://example.com/article")
```

- Without `output_dir` the work happens in a deleted temporary directory: Markdown in memory, path fields `None`, image links left as `.markitai/assets/...`.
- One call converts one file or URL; directory batches stay with the CLI.
- `stdout` stays clean; [loguru](https://github.com/Delgan/loguru) logs to stderr (`logger.disable("markitai")` silences it).

## LLM-Enhanced Conversion

`llm=True` adds cleanup and frontmatter and returns both variants:

```python
import os
import markitai

os.environ["MODEL"] = "openai/gpt-6-luna"  # or configure llm.model_list

out = markitai.convert("report.pdf", output_dir="out/", llm=True)
print(out.llm_markdown)  # enhanced body
print(out.frontmatter["title"])  # parsed YAML frontmatter
print(out.usage.cost_usd)  # LLM spend for this conversion
```

```mermaid
flowchart LR
  L[llm.model_list] -->|empty| M[MODEL]
  M -->|unset| D[Detected keys]
  D -->|none| E[NoModelConfiguredError]
```

The model is chosen as in the CLI ([configuration](/guide/configuration), then [detected](/guide/configuration#defaults-markitai-picks-for-you) API keys and signed-in subscriptions); several detected providers share one pool, logged once per process. Unlike the CLI, no `.env` file is read: export keys first. The CLI's config files are loaded unless you pass a `MarkitaiConfig`:

```python
from markitai import MarkitaiConfig

cfg = MarkitaiConfig()  # pure defaults, ignores config files
cfg.llm.pure = True  # raw LLM cleanup, no frontmatter
out = markitai.convert("notes.docx", config=cfg, llm=True)
```

The keywords `llm`, `ocr`, `screenshot`, `alt`, `desc` and `profile` ([Output Profiles](./output-profiles.md)) mirror the CLI flags; `None` keeps the configured value.

## Async Usage

Inside an event loop use `aconvert`; CPU-bound work runs in a thread pool:

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

- The sync `convert()` raises `RuntimeError` inside a running loop.
- Long-lived apps call `await markitai.fetch.close_shared_clients()` on shutdown.

::: details A script that exits with code 134
An onnxruntime teardown quirk can end a short script with code 134 after a successful conversion. End it with `from markitai.utils.shutdown import finalize_process; finalize_process(0)`; never in a long-lived host, since it ends the process.
:::

## Parallel PDF Extraction

The CLI, `markitai serve` and the MCP server extract long or concurrent PDFs page by page in worker processes (OCR included): same output, several times faster. A script opts in:

```python
import markitai


def main() -> None:
    markitai.enable_worker_processes()
    for name in ("a.pdf", "b.pdf"):
        markitai.convert(name, output_dir="out/")


if __name__ == "__main__":
    main()
```

Workers re-import `__main__`, so keep the work under `if __name__ == "__main__":`. `MARKITAI_PDF_WORKERS` sets their number.

## ConversionOutput

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | Input as given |
| `markdown` | `str` | Base body, no frontmatter |
| `llm_markdown` | `str \| None` | Enhanced body, or `None` |
| `frontmatter` | `dict` | Parsed frontmatter |
| `output_path` | `Path \| None` | Written base `.md` file |
| `llm_output_path` | `Path \| None` | Written `.llm.md` file |
| `assets` | `list[Path]` | Extracted images |
| `screenshots` | `list[Path]` | Screenshot files |
| `images` | `list[dict]` | Per-image LLM analysis |
| `usage` | `ConversionUsage` | `cost_usd`, tokens, per model |
| `skip_reason` | `str \| None` | `"exists"` when skipped |
| `duration` | `float` | Wall-clock seconds |
| `warnings` | `list[str]` | Notices (below) |
| `page_sources` | `list[dict]` | PDF page text origin (below) |

`warnings` (per call, also logged): scanned-looking pages (retry with `ocr=True`), hidden PDF text (possible prompt injection), OCR without text, unrendered slides, a missed URL screenshot, a failed image analysis, or a failed LLM enhancement (`llm_markdown` is `None`; `llm.on_failure = "fail"` raises `ConversionError`).

`page_sources`: one `{"page", "source", "reason"}` per page; `source` is `"text"`, `"ocr"` or `"mixed"`, `reason` a short note such as `"no text layer"`.

### Errors

Failures raise; the markitai ones import from `markitai.api`:

| Exception | When |
|-----------|------|
| `ConversionError` | Pipeline failed or empty |
| `FetchError` | URL unreachable |
| `NoModelConfiguredError` | LLM on, no usable model (a `ValueError`; nothing sent) |
| `ConfigFileError` | Invalid config file |
| `FileNotFoundError`, `IsADirectoryError` | Bad path |
