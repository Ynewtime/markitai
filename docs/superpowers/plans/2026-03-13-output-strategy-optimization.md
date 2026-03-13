# Output Strategy Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Skip image-only formats in non-LLM/non-OCR mode, and only output `.llm.md` (not `.md`) in LLM mode, with `--keep-base` opt-in.

**Architecture:** Two rules layered into the existing pipeline: Rule A adds an early-return check in `convert_document_core()` for image-only formats without LLM/OCR; Rule B conditionally skips the `write_base_markdown()` step and adds `--keep-base` CLI flag. LLM failure falls back to writing `.md`.

**Tech Stack:** Python, Click (CLI), Pydantic v2 (config), pytest (testing)

**Spec:** `docs/superpowers/specs/2026-03-13-output-strategy-optimization-design.md`

---

## Chunk 1: Define IMAGE_ONLY_FORMATS, add `keep_base` config, and `--keep-base` CLI option

### Task 1: Add `IMAGE_ONLY_FORMATS` to `converter/base.py` and export it

**Files:**
- Modify: `packages/markitai/src/markitai/converter/base.py` (after `FileFormat` enum, around line 79)
- Modify: `packages/markitai/src/markitai/converter/__init__.py` (add export)
- Test: `packages/markitai/tests/unit/test_converter.py` (existing file)

- [ ] **Step 1: Write the failing test**

In `packages/markitai/tests/unit/test_converter.py`, add:

```python
from markitai.converter.base import IMAGE_ONLY_FORMATS


class TestImageOnlyFormats:
    """Tests for IMAGE_ONLY_FORMATS set."""

    def test_contains_raster_image_formats(self) -> None:
        """IMAGE_ONLY_FORMATS contains all raster image formats."""
        expected = {
            FileFormat.JPEG,
            FileFormat.JPG,
            FileFormat.PNG,
            FileFormat.WEBP,
            FileFormat.GIF,
            FileFormat.BMP,
            FileFormat.TIFF,
        }
        assert IMAGE_ONLY_FORMATS == expected

    def test_excludes_svg(self) -> None:
        """SVG is not in IMAGE_ONLY_FORMATS (it has text extraction capability)."""
        assert FileFormat.SVG not in IMAGE_ONLY_FORMATS

    def test_excludes_document_formats(self) -> None:
        """Document formats are not in IMAGE_ONLY_FORMATS."""
        for fmt in (FileFormat.PDF, FileFormat.DOCX, FileFormat.CSV, FileFormat.HTML):
            assert fmt not in IMAGE_ONLY_FORMATS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_converter.py::TestImageOnlyFormats -v`
Expected: FAIL with `ImportError: cannot import name 'IMAGE_ONLY_FORMATS'`

- [ ] **Step 3: Write minimal implementation**

In `packages/markitai/src/markitai/converter/base.py`, after the `FileFormat` enum (after line 79), add:

```python
IMAGE_ONLY_FORMATS: frozenset[FileFormat] = frozenset({
    FileFormat.JPEG,
    FileFormat.JPG,
    FileFormat.PNG,
    FileFormat.WEBP,
    FileFormat.GIF,
    FileFormat.BMP,
    FileFormat.TIFF,
})
"""Raster image formats that produce no textual content without LLM/OCR.

SVG is intentionally excluded — it is XML-based and can contain extractable
text elements via KreuzbergConverter.
"""
```

Then in `packages/markitai/src/markitai/converter/__init__.py`, add `IMAGE_ONLY_FORMATS` to the existing imports from `base.py`:

```python
from markitai.converter.base import IMAGE_ONLY_FORMATS  # add to existing imports
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_converter.py::TestImageOnlyFormats -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/converter/base.py packages/markitai/src/markitai/converter/__init__.py packages/markitai/tests/unit/test_converter.py
git commit -m "feat: define IMAGE_ONLY_FORMATS constant for raster image formats"
```

### Task 2: Add `keep_base` field to `LLMConfig`

**Files:**
- Modify: `packages/markitai/src/markitai/config.py` (`LLMConfig` class)
- Test: `packages/markitai/tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

In `packages/markitai/tests/unit/test_config.py`, add:

```python
class TestLLMConfigKeepBase:
    """Tests for LLMConfig.keep_base field."""

    def test_keep_base_defaults_to_false(self) -> None:
        """keep_base should default to False."""
        from markitai.config import LLMConfig

        cfg = LLMConfig()
        assert cfg.keep_base is False

    def test_keep_base_can_be_enabled(self) -> None:
        """keep_base can be set to True."""
        from markitai.config import LLMConfig

        cfg = LLMConfig(keep_base=True)
        assert cfg.keep_base is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_config.py::TestLLMConfigKeepBase -v`
Expected: FAIL with `ValidationError` (unexpected field `keep_base`)

- [ ] **Step 3: Write minimal implementation**

In `packages/markitai/src/markitai/config.py`, add `keep_base` to `LLMConfig` class after `pure`:

```python
    keep_base: bool = Field(
        default=False,
        description="Keep base .md file alongside .llm.md in LLM mode",
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_config.py::TestLLMConfigKeepBase -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/config.py packages/markitai/tests/unit/test_config.py
git commit -m "feat: add keep_base field to LLMConfig"
```

### Task 3: Add `--keep-base` CLI option

**Files:**
- Modify: `packages/markitai/src/markitai/cli/main.py`
- Test: `packages/markitai/tests/unit/cli/test_main.py`

- [ ] **Step 1: Write the failing test**

Check existing test patterns in `tests/unit/cli/test_main.py` first. Add:

```python
def test_keep_base_option_exists(cli_runner: CliRunner) -> None:
    """--keep-base option should be recognized by the CLI."""
    result = cli_runner.invoke(cli, ["--help"])
    assert "--keep-base" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_main.py::test_keep_base_option_exists -v`
Expected: FAIL (option not in help output)

- [ ] **Step 3: Write minimal implementation**

In `packages/markitai/src/markitai/cli/main.py`:

1. Add the Click option decorator near `--pure` (around line 251):

```python
@click.option(
    "--keep-base",
    is_flag=True,
    help="Keep base .md file alongside .llm.md in LLM mode (for debugging/comparison).",
)
```

2. Add `keep_base` to the function signature.

3. Wire it in the config application section (where `--pure` is handled, around line 502):

```python
if keep_base:
    cfg.llm.keep_base = True
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_main.py::test_keep_base_option_exists -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/main.py packages/markitai/tests/unit/cli/test_main.py
git commit -m "feat: add --keep-base CLI option"
```

## Chunk 2: Rule A — Skip image-only formats in non-LLM/non-OCR mode

### Task 4: Add `detected_format` field to `ConversionContext` and store format in `validate_and_detect_format()`

**Context:** Currently `validate_and_detect_format()` detects the format as a local variable `fmt` but never stores it on `ctx`. We need `ctx.detected_format` for the image-only skip check.

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py` (add field to `ConversionContext` dataclass, set it in `validate_and_detect_format()`)
- Test: `packages/markitai/tests/unit/test_workflow_core.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
from markitai.config import MarkitaiConfig
from markitai.converter.base import FileFormat
from markitai.workflow.core import ConversionContext, validate_and_detect_format
from markitai.constants import MAX_DOCUMENT_SIZE


class TestDetectedFormat:
    """Tests for detected_format being stored on ConversionContext."""

    def test_detected_format_set_after_validation(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """validate_and_detect_format should set ctx.detected_format."""
        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        assert result.success
        assert ctx.detected_format == FileFormat.CSV

    def test_detected_format_for_image(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """validate_and_detect_format should detect BMP as image format."""
        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        assert result.success
        assert ctx.detected_format == FileFormat.BMP
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestDetectedFormat -v`
Expected: FAIL with `AttributeError: 'ConversionContext' object has no attribute 'detected_format'`

- [ ] **Step 3: Write minimal implementation**

1. In `ConversionContext` dataclass (around line 64, "Intermediate state" section), add:

```python
    detected_format: FileFormat | None = None
```

Add the import at the top of the file (in the TYPE_CHECKING block or directly):

```python
from markitai.converter.base import FileFormat
```

2. In `validate_and_detect_format()` (around line 150), after `fmt = detect_format(ctx.effective_input)`, store it:

```python
    fmt = detect_format(ctx.effective_input)
    ctx.detected_format = fmt
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestDetectedFormat -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/core.py packages/markitai/tests/unit/test_workflow_core.py
git commit -m "feat: store detected_format on ConversionContext"
```

### Task 5: Add image-only skip logic in `convert_document_core()`

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py` (in `convert_document_core()`, after format detection step)
- Test: `packages/markitai/tests/unit/test_workflow_core.py`

- [ ] **Step 1: Write the failing test**

Tests call `convert_document_core()` (the public API) and assert on outputs — not internal steps.

```python
from markitai.workflow.core import convert_document_core


class TestImageOnlySkip:
    """Tests for skipping image-only formats in non-LLM/non-OCR mode."""

    async def test_image_skipped_without_llm_or_ocr(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Image format should be skipped when neither LLM nor OCR is enabled."""
        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.success is True
        assert result.skip_reason == "image_only"
        # No output files should be created
        output_files = list(output_dir.glob("*.md"))
        assert len(output_files) == 0

    async def test_image_not_skipped_with_llm(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Image format should NOT be skipped when LLM is enabled."""
        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.skip_reason != "image_only"

    async def test_image_not_skipped_with_ocr(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Image format should NOT be skipped when OCR is enabled."""
        input_path = fixtures_dir / "sample.bmp"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.ocr.enabled = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.skip_reason != "image_only"

    async def test_document_not_skipped_without_llm(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Document formats should never be skipped by this rule."""
        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.skip_reason != "image_only"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestImageOnlySkip -v`
Expected: FAIL (`test_image_skipped_without_llm_or_ocr` fails — no skip logic yet)

- [ ] **Step 3: Write minimal implementation**

In `packages/markitai/src/markitai/workflow/core.py`, in `convert_document_core()`, after the `validate_and_detect_format` step and before `prepare_output_directory`, insert:

```python
    # Step 1.5: Skip image-only formats when neither LLM nor OCR is enabled
    if (
        ctx.detected_format is not None
        and ctx.detected_format in IMAGE_ONLY_FORMATS
        and not ctx.config.llm.enabled
        and not ctx.config.ocr.enabled
    ):
        logger.info(
            f"[Core] Skipped {ctx.input_path.name} "
            f"(image file, no text to extract without LLM or OCR)"
        )
        return ConversionStepResult(
            success=True,
            skip_reason="image_only",
        )
```

Add the import at the top of the function (or file level):

```python
from markitai.converter.base import IMAGE_ONLY_FORMATS
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestImageOnlySkip -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/core.py packages/markitai/tests/unit/test_workflow_core.py
git commit -m "feat: skip image-only formats in non-LLM/non-OCR mode (Rule A)"
```

### Task 6: Add skip message in CLI file processor

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/file.py` (skip handling section)
- Test: `packages/markitai/tests/unit/cli/test_file_processor.py`

- [ ] **Step 1: Write the failing test**

Check existing test patterns in `test_file_processor.py` first. Add a test that verifies when an image file is processed without LLM/OCR, a skip warning is shown. Adapt test to use the patterns found in the file (e.g., `CliRunner`, `capsys`, or mock console).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_file_processor.py::<test_name> -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

In `packages/markitai/src/markitai/cli/processors/file.py`, in the skip handling block (where `result.skip_reason == "exists"` is checked), add before or alongside it:

```python
if result.skip_reason == "image_only":
    progress.stop_spinner()
    if not quiet:
        ui.warning(
            f"Skipped {input_path.name} (image file, no text to extract). "
            f"Use --llm or --ocr for content extraction."
        )
    return
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_file_processor.py::<test_name> -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/processors/file.py packages/markitai/tests/unit/cli/test_file_processor.py
git commit -m "feat: show skip warning for image files in non-LLM mode"
```

### Task 7: Handle image skip in batch processor

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/batch.py`
- Test: `packages/markitai/tests/unit/test_batch_processor.py`

- [ ] **Step 1: Write the failing test**

Check existing patterns in `test_batch_processor.py`. Add a test that in batch mode, image files without LLM/OCR are skipped and counted in the summary.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_batch_processor.py::<test_name> -v`
Expected: FAIL

- [ ] **Step 3: Write minimal implementation**

In `packages/markitai/src/markitai/cli/processors/batch.py`, in the `process_file()` closure, handle the `image_only` skip reason returned by `convert_document_core()`. The `ProcessResult` should indicate the file was skipped so the batch summary can report skipped count.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_batch_processor.py::<test_name> -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/processors/batch.py packages/markitai/tests/unit/test_batch_processor.py
git commit -m "feat: handle image-only skip in batch processor"
```

## Chunk 3: Rule B — LLM mode outputs only `.llm.md`

### Task 8: Conditionally skip `write_base_markdown()` in LLM mode

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py` (`write_base_markdown()` function)
- Test: `packages/markitai/tests/unit/test_workflow_core.py`

- [ ] **Step 1: Write the failing test**

Tests call `convert_document_core()` (public API) and assert on file output — not internal step functions.

```python
class TestLLMOnlyOutput:
    """Tests for LLM mode outputting only .llm.md."""

    async def test_llm_mode_skips_base_md_file(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """In LLM mode, base .md file should not exist on disk after conversion."""
        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        # Run full pipeline — LLM will fail (no model), but base .md
        # should not have been written before LLM step.
        # LLM failure triggers fallback, so we need to check that
        # write_base_markdown was skipped. We do this by testing
        # write_base_markdown directly (it's a public function).
        from markitai.workflow.core import (
            convert_document,
            prepare_output_directory,
            process_embedded_images,
            resolve_output_file,
            validate_and_detect_format,
            write_base_markdown,
        )

        validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        prepare_output_directory(ctx)
        resolve_output_file(ctx)
        await convert_document(ctx)
        await process_embedded_images(ctx)
        result = write_base_markdown(ctx)
        assert result.success is True
        # Base .md should NOT exist on disk (LLM mode, no --keep-base)
        assert not ctx.output_file.exists()
        # But in-memory markdown is still available for LLM processing
        assert ctx.conversion_result is not None
        assert len(ctx.conversion_result.markdown) > 0

    async def test_keep_base_writes_md_file(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """With --keep-base, base .md SHOULD be written even in LLM mode."""
        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.keep_base = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        from markitai.workflow.core import (
            convert_document,
            prepare_output_directory,
            process_embedded_images,
            resolve_output_file,
            validate_and_detect_format,
            write_base_markdown,
        )

        validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        prepare_output_directory(ctx)
        resolve_output_file(ctx)
        await convert_document(ctx)
        await process_embedded_images(ctx)
        result = write_base_markdown(ctx)
        assert result.success is True
        assert ctx.output_file.exists()

    async def test_non_llm_always_writes_md_file(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """Without LLM, base .md should always be written (unchanged behavior)."""
        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        assert result.success is True
        assert ctx.output_file is not None
        assert ctx.output_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestLLMOnlyOutput -v`
Expected: FAIL (`test_llm_mode_skips_base_md_file` will fail because `.md` is always written)

- [ ] **Step 3: Write minimal implementation**

In `packages/markitai/src/markitai/workflow/core.py`, modify `write_base_markdown()`:

```python
def write_base_markdown(ctx: ConversionContext) -> ConversionStepResult:
    """Write base markdown file with basic frontmatter.

    In LLM mode without --keep-base, the base .md is NOT written to disk.
    The conversion result remains available in memory (ctx.conversion_result.markdown)
    for downstream consumers like stabilize_written_llm_output().
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(
            success=False, error="Missing conversion result or output file"
        )

    # In LLM mode, skip writing .md unless --keep-base is set
    if ctx.config.llm.enabled and not ctx.config.llm.keep_base:
        logger.debug(
            f"[Core] Skipped writing base .md (LLM mode, no --keep-base): "
            f"{ctx.output_file}"
        )
        return ConversionStepResult(success=True)

    title = ctx.conversion_result.metadata.get("title")
    base_md_content = add_basic_frontmatter(
        ctx.conversion_result.markdown,
        ctx.input_path.name,
        title=title if isinstance(title, str) else None,
    )
    atomic_write_text(ctx.output_file, base_md_content)
    logger.debug(f"Written output: {ctx.output_file}")

    return ConversionStepResult(success=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestLLMOnlyOutput -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/core.py packages/markitai/tests/unit/test_workflow_core.py
git commit -m "feat: skip writing base .md in LLM mode unless --keep-base (Rule B)"
```

### Task 9: Handle LLM failure fallback — write `.md` when LLM fails

**Context:** The LLM processing section in `convert_document_core()` (step 7) has three failure paths: pure mode (line ~963), vision mode (line ~982), and standard mode (line ~1015). Each currently returns early on failure. We need to write `.md` as fallback before returning.

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/core.py` (`convert_document_core()` step 7)
- Test: `packages/markitai/tests/unit/test_workflow_core.py`

- [ ] **Step 1: Write the failing test**

```python
class TestLLMFailureFallback:
    """Tests for writing .md as fallback when LLM processing fails."""

    async def test_md_written_on_llm_failure(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """When LLM fails, .md should be written as fallback."""
        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        # No model configured — LLM will fail
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        result = await convert_document_core(ctx, MAX_DOCUMENT_SIZE)
        # LLM should fail (no models configured)
        assert not result.success
        # But .md should be written as fallback
        assert ctx.output_file is not None
        assert ctx.output_file.exists()
        # Verify it has content (not empty)
        content = ctx.output_file.read_text(encoding="utf-8")
        assert len(content) > 0

    async def test_md_not_written_on_llm_success_without_keep_base(
        self, tmp_path: Path, fixtures_dir: Path
    ) -> None:
        """When LLM succeeds, .md should NOT be on disk (without --keep-base)."""
        # This is covered by TestLLMOnlyOutput, but here for completeness
        input_path = fixtures_dir / "sample.csv"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        ctx = ConversionContext(
            input_path=input_path, output_dir=output_dir, config=cfg
        )
        from markitai.workflow.core import (
            convert_document,
            prepare_output_directory,
            process_embedded_images,
            resolve_output_file,
            validate_and_detect_format,
            write_base_markdown,
        )

        validate_and_detect_format(ctx, MAX_DOCUMENT_SIZE)
        prepare_output_directory(ctx)
        resolve_output_file(ctx)
        await convert_document(ctx)
        await process_embedded_images(ctx)
        write_base_markdown(ctx)
        # .md should not exist (LLM mode, not yet failed)
        assert not ctx.output_file.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestLLMFailureFallback -v`
Expected: FAIL (`test_md_written_on_llm_failure` fails — `.md` not written on LLM failure)

- [ ] **Step 3: Write minimal implementation**

Add a helper function and modify the LLM failure paths in `convert_document_core()`:

```python
def _write_base_md_fallback(ctx: ConversionContext) -> None:
    """Write base .md as fallback when LLM processing fails.

    Called when LLM mode is active but LLM processing fails. Ensures the user
    gets at least the base conversion result instead of nothing.
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return
    if ctx.output_file.exists():
        return  # Already written (e.g., --keep-base was set)
    title = ctx.conversion_result.metadata.get("title")
    base_md_content = add_basic_frontmatter(
        ctx.conversion_result.markdown,
        ctx.input_path.name,
        title=title if isinstance(title, str) else None,
    )
    atomic_write_text(ctx.output_file, base_md_content)
    logger.warning(
        f"[Core] LLM processing failed, wrote base .md as fallback: "
        f"{ctx.output_file}"
    )
```

Then in `convert_document_core()`, at each LLM failure return point, call `_write_base_md_fallback(ctx)` before returning. There are three paths:

1. Pure mode failure (around line 963):
```python
            result = await process_with_pure_llm(ctx)
            if not result.success:
                _write_base_md_fallback(ctx)
                return result
```

2. Vision mode failure (around line 982):
```python
                if not vision_result.success:
                    _write_base_md_fallback(ctx)
                    return vision_result
```

3. Standard mode failure (around line 1015):
```python
                result = await process_with_standard_llm(ctx)
                if not result.success:
                    _write_base_md_fallback(ctx)
                    return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestLLMFailureFallback -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/core.py packages/markitai/tests/unit/test_workflow_core.py
git commit -m "feat: write .md as fallback when LLM processing fails"
```

### Task 10: Verify `stabilize_written_llm_output` works without `.md` on disk

**Context:** `stabilize_written_llm_output()` calls `_read_markdown_body(ctx.output_file, ctx.conversion_result.markdown)`. When `.md` is not on disk, `_read_markdown_body` should fall back to the in-memory `ctx.conversion_result.markdown`. This task verifies that the existing fallback works correctly.

**Files:**
- Test: `packages/markitai/tests/unit/test_workflow_core.py`
- Verify: `packages/markitai/src/markitai/workflow/single.py` (`_read_markdown_body`)

- [ ] **Step 1: Write the verification test**

```python
from markitai.workflow.single import _read_markdown_body


class TestReadMarkdownBodyFallback:
    """Tests that _read_markdown_body falls back to in-memory content."""

    def test_fallback_when_file_missing(self, tmp_path: Path) -> None:
        """When output .md file doesn't exist, should return fallback string."""
        nonexistent = tmp_path / "nonexistent.md"
        fallback = "# Hello\n\nSome content"
        result = _read_markdown_body(nonexistent, fallback)
        assert result == fallback

    def test_reads_file_when_exists(self, tmp_path: Path) -> None:
        """When output .md file exists, should read from it."""
        md_file = tmp_path / "test.md"
        md_file.write_text("---\ntitle: test\n---\n\n# From File\n\nContent")
        result = _read_markdown_body(md_file, "fallback")
        assert "From File" in result
        assert "fallback" not in result
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cd packages/markitai && uv run pytest tests/unit/test_workflow_core.py::TestReadMarkdownBodyFallback -v`
Expected: PASS (this should already work — the fallback logic exists in `_read_markdown_body`)

- [ ] **Step 3: Commit**

```bash
git add packages/markitai/tests/unit/test_workflow_core.py
git commit -m "test: verify _read_markdown_body fallback works without .md on disk"
```

## Chunk 4: CLI integration and final verification

### Task 11: Update CLI file processor for LLM-only output

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/file.py` (if needed)
- Test: `packages/markitai/tests/unit/cli/test_file_processor.py`

- [ ] **Step 1: Review existing fallback logic**

Read `file.py` lines 256-266. The existing logic already prefers `.llm.md` in LLM mode and falls back to `.md` when `.llm.md` doesn't exist. This fallback now correctly serves the LLM failure case. Verify this is sufficient — it may not need code changes, just a test.

- [ ] **Step 2: Write a test verifying the fallback**

Test that when `.llm.md` exists, it's used; when only `.md` exists (LLM failure), `.md` is used.

- [ ] **Step 3: Run test**

Run: `cd packages/markitai && uv run pytest tests/unit/cli/test_file_processor.py::<test_name> -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add packages/markitai/src/markitai/cli/processors/file.py packages/markitai/tests/unit/cli/test_file_processor.py
git commit -m "test: verify file processor handles LLM-only output correctly"
```

### Task 12: Run full test suite and fix regressions

- [ ] **Step 1: Run the full test suite**

Run: `cd packages/markitai && uv run pytest -x -v`
Expected: All tests PASS. If any fail, investigate and fix.

- [ ] **Step 2: Run linting and type checking**

Run: `cd packages/markitai && uv run ruff check --fix && uv run ruff format && uv run pyright`
Expected: Clean

- [ ] **Step 3: Manual smoke tests**

```bash
# Test 1: Image file without LLM — should show skip warning, no output
markitai packages/markitai/tests/fixtures/sample.bmp --no-cache -o output --verbose

# Test 2: Document file without LLM — should produce .md only
markitai packages/markitai/tests/fixtures/sample.csv --no-cache -o output --verbose

# Test 3: Batch with --pure — image files get .llm.md, documents get .llm.md, no .md files
markitai packages/markitai/tests/fixtures/ --no-cache -o output --verbose --pure

# Test 4: --keep-base — both .md and .llm.md should exist
markitai packages/markitai/tests/fixtures/sample.csv --no-cache -o output --verbose --pure --keep-base
```

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: address test regressions from output strategy changes"
```
