# Pure Mode Decoupling & Image Vision Path Design

## Problem

Two design coupling issues in the current output strategy:

1. **`--pure` is coupled to `--llm`**: `--pure` implicitly sets `cfg.llm.enabled = True`, preventing independent use. Users cannot get raw conversion output (no frontmatter) without triggering LLM processing.

2. **Image inputs produce useless output with `--pure`**: `process_with_pure_llm()` sends the markdown text `# sample\n\n![sample](path)` to the LLM for text cleaning. The LLM cannot extract content from an image reference — only Vision analysis or OCR can.

## Design

### Change 1: Decouple `--pure` from `--llm`

**`--pure` becomes an independent flag** controlling the processing pipeline (skip frontmatter and post-processing), orthogonal to `--llm` (enable LLM).

#### CLI changes (`cli/main.py`)

- Remove `cfg.llm.enabled = True` from the `--pure` handler (line 510)
- Remove the same from `MARKITAI_PURE` env var handler (line 518)
- Update help text: `"Pure mode: skip frontmatter and post-processing. With --llm: raw MD → LLM → output."`

#### `write_base_markdown()` changes (`workflow/core.py`)

When `cfg.llm.pure` is set and `cfg.llm.enabled` is not:

- Skip `add_basic_frontmatter()` — write `ctx.conversion_result.markdown` directly
- Output file is `.md` (no LLM, so no `.llm.md`)

The existing LLM-mode skip (`cfg.llm.enabled and not cfg.llm.keep_base`) remains unchanged.

Decision tree for `write_base_markdown()`:

```
if llm.enabled and not llm.keep_base:
    skip writing (in-memory only for LLM consumption)
elif llm.pure and not llm.enabled:
    write raw markdown without frontmatter
else:
    write with frontmatter (current default behavior)
```

### Change 2: Image Vision path for `--llm --pure`

When `cfg.llm.pure` AND `cfg.llm.enabled` AND input is an image-only format:

- Do NOT call `process_with_pure_llm()` (text cleaning path — useless for images)
- Instead call new function `process_image_with_vision_pure(ctx)`:
  1. Get the image file path from `.markitai/assets/` (already copied by ImageConverter)
  2. Format adaptation if needed (BMP/TIFF → PNG, reusing existing ImageConverter transcoding)
  3. Call `processor.analyze_image(image_path)` — sends image to Vision model
  4. Write Vision model's raw output directly to `.llm.md` — no frontmatter wrapping, no post-processing
  5. If Vision model returns structured data (caption, description, extracted_text), format as simple markdown

#### `process_image_with_vision_pure()` output format

```markdown
# {filename}

{description}

{extracted_text if available}
```

markitai does not add or modify the LLM's output. If the LLM returns frontmatter, it is preserved as-is.

#### Failure handling

If Vision analysis fails (no model configured, API error):
- `_write_base_md_fallback(ctx)` writes `.md` as fallback (existing mechanism)
- The fallback contains the raw image reference markdown

### Rule A — image-only skip condition

The existing skip condition in `convert_document_core()` Step 1.5:

```python
if (
    ctx.detected_format in IMAGE_ONLY_FORMATS
    and not ctx.config.llm.enabled
    and not ctx.config.ocr.enabled
):
    return ConversionStepResult(success=True, skip_reason="image_only")
```

**No change needed.** After decoupling `--pure` from `--llm`:

- `--pure` alone → `llm.enabled=False`, `ocr.enabled=False` → Rule A triggers, skip
- `--llm --pure` → `llm.enabled=True` → Rule A does not trigger, proceeds to Vision path
- `--llm` → `llm.enabled=True` → Rule A does not trigger, proceeds to standard Vision path

## Behavior Matrix

### Image inputs (JPEG, JPG, PNG, WEBP, GIF, BMP, TIFF)

| Flags | Behavior | Output |
|-------|----------|--------|
| (none) | Rule A skip, terminal warning | No file |
| `--pure` | Rule A skip, terminal warning | No file |
| `--ocr` | RapidOCR text extraction | `.md` (with frontmatter) |
| `--ocr --pure` | RapidOCR text extraction | `.md` (without frontmatter) |
| `--llm` | Standard Vision analysis + post-processing | `.llm.md` |
| `--llm --pure` | Vision analysis, raw LLM output | `.llm.md` |
| `--ocr --llm` | OCR + LLM cleaning | `.llm.md` |
| `--ocr --llm --pure` | OCR + LLM pure cleaning | `.llm.md` |

### Non-image inputs (CSV, PDF, DOCX, etc.)

| Flags | Behavior | Output |
|-------|----------|--------|
| (none) | Convert + frontmatter | `.md` |
| `--pure` | Convert, no frontmatter | `.md` |
| `--llm` | Convert + LLM cleaning + frontmatter | `.llm.md` |
| `--llm --pure` | Convert + LLM pure cleaning, raw output | `.llm.md` |

## Edge Cases

1. **`MARKITAI_PURE=1` env var**: Same decoupling — only sets `cfg.llm.pure = True`, does not enable LLM
2. **`--llm --pure` + image + Vision model not configured**: LLM fails → `_write_base_md_fallback()` writes `.md` with raw image reference
3. **`--keep-base` + `--pure --llm`**: `--keep-base` still controls base `.md` retention, behavior unchanged
4. **Image format adaptation**: When Vision model doesn't support BMP/TIFF, ImageConverter already transcodes to PNG in `.markitai/assets/` — `process_image_with_vision_pure()` uses the transcoded file

## Files to Modify

1. **`cli/main.py`**: Remove `--pure` → `llm.enabled = True` coupling; update help text
2. **`workflow/core.py`**:
   - `write_base_markdown()`: Add pure-without-LLM branch (skip frontmatter)
   - `convert_document_core()` Step 7: Route `pure + image` to new Vision path
   - New function: `process_image_with_vision_pure(ctx)`
3. **Test files**: Update existing `--pure` tests, add new tests for decoupled behavior and image Vision path

## Files NOT Modified

- `config.py`: `LLMConfig.pure` field semantics unchanged
- `converter/image.py`: ImageConverter behavior unchanged
- `llm/vision.py`: Reuse existing `analyze_image()` method
- `cli/processors/file.py` / `batch.py`: Rule A condition unchanged, works correctly after decoupling
