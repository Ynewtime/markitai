# Pure Mode Decoupling & Image Vision Path Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decouple `--pure` from `--llm` so they work independently, and add a Vision analysis path for image inputs under `--llm --pure`.

**Architecture:** Two changes: (1) Remove `cfg.llm.enabled = True` from `--pure` handler, add pure-without-LLM branch in `write_base_markdown()` to write raw markdown without frontmatter; (2) Add routing in Step 7 to detect image-only formats under `--pure` and call a new `process_image_with_vision_pure()` that uses the existing `analyze_image()` Vision API, formatting the result into `.llm.md`.

**Tech Stack:** Python 3.13, Click CLI, Pydantic v2, pytest, asyncio

**Spec:** `docs/superpowers/specs/2026-03-13-pure-decoupling-and-image-vision-design.md`

---

## Chunk 1: Decouple `--pure` from `--llm`

### Task 1: Update `--pure` CLI flag tests to expect decoupled behavior

**Files:**
- Modify: `packages/markitai/tests/unit/cli/test_main.py:255-291` (TestPureCLIFlag)

- [ ] **Step 1: Write failing tests for decoupled --pure**

Add tests that verify `--pure` does NOT set `llm.enabled = True`:

```python
# In TestPureCLIFlag class, add:

def test_pure_flag_does_not_enable_llm(self, tmp_path):
    """--pure alone should NOT set llm.enabled = True."""
    from unittest.mock import patch

    from click.testing import CliRunner

    from markitai.cli.main import app

    runner = CliRunner()
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("# Hello", encoding="utf-8")

    captured_cfg = {}

    original_convert = None

    async def capture_cfg(input_path, output_dir, cfg, **kwargs):
        captured_cfg["llm_enabled"] = cfg.llm.enabled
        captured_cfg["llm_pure"] = cfg.llm.pure

    with patch(
        "markitai.cli.processors.file.process_single_file",
        side_effect=capture_cfg,
    ):
        runner.invoke(
            app,
            [str(txt_file), "--pure", "-o", str(tmp_path / "out")],
        )

    assert captured_cfg.get("llm_pure") is True
    assert captured_cfg.get("llm_enabled") is False

def test_pure_env_var_does_not_enable_llm(self, tmp_path):
    """MARKITAI_PURE=1 should NOT set llm.enabled = True."""
    from unittest.mock import patch

    from click.testing import CliRunner

    from markitai.cli.main import app

    runner = CliRunner()
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("# Hello", encoding="utf-8")

    captured_cfg = {}

    async def capture_cfg(input_path, output_dir, cfg, **kwargs):
        captured_cfg["llm_enabled"] = cfg.llm.enabled
        captured_cfg["llm_pure"] = cfg.llm.pure

    with patch(
        "markitai.cli.processors.file.process_single_file",
        side_effect=capture_cfg,
    ):
        runner.invoke(
            app,
            [str(txt_file), "-o", str(tmp_path / "out")],
            env={"MARKITAI_PURE": "1"},
        )

    assert captured_cfg.get("llm_pure") is True
    assert captured_cfg.get("llm_enabled") is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_main.py::TestPureCLIFlag -v`
Expected: FAIL — `llm_enabled` is `True` (current coupling)

- [ ] **Step 3: Remove `--pure` → `llm.enabled` coupling in cli/main.py**

In `packages/markitai/src/markitai/cli/main.py`:

Line 510 — remove `cfg.llm.enabled = True`:
```python
# Before:
    if pure:
        cfg.llm.pure = True
        cfg.llm.enabled = True  # --pure implies --llm

# After:
    if pure:
        cfg.llm.pure = True
```

Line 518 — remove `cfg.llm.enabled = True`:
```python
# Before:
    if not pure and os.environ.get("MARKITAI_PURE", "").strip() in ("1", "true", "yes"):
        cfg.llm.pure = True
        cfg.llm.enabled = True

# After:
    if not pure and os.environ.get("MARKITAI_PURE", "").strip() in ("1", "true", "yes"):
        cfg.llm.pure = True
```

Update `--pure` help text to clarify independence:
```python
# Find the --pure option definition and update help:
help="Pure mode: skip frontmatter and post-processing. With --llm: raw MD → LLM → output."
```

- [ ] **Step 4: Update existing test that asserts --pure implies --llm**

The existing `test_pure_flag_recognized` test description says "implies --llm". Update the docstring:
```python
def test_pure_flag_recognized(self, tmp_path):
    """--pure should be a recognized CLI flag."""
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/cli/test_main.py::TestPureCLIFlag -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/tests/unit/cli/test_main.py packages/markitai/src/markitai/cli/main.py
git commit -m "decouple --pure from --llm: remove implicit llm.enabled=True"
```

---

### Task 2: Add pure-without-LLM branch in `write_base_markdown()`

**Files:**
- Modify: `packages/markitai/tests/unit/test_workflow_core.py` (add tests)
- Modify: `packages/markitai/src/markitai/workflow/core.py:388-423` (write_base_markdown)

- [ ] **Step 1: Write failing test for pure-without-LLM writing raw markdown**

Add a new test class in `packages/markitai/tests/unit/test_workflow_core.py`:

```python
class TestPureWithoutLLMOutput:
    """Tests for --pure without --llm: write raw markdown without frontmatter."""

    def test_pure_without_llm_writes_raw_markdown(self, tmp_path, fixtures_dir):
        """--pure without --llm should write .md without frontmatter."""
        from markitai.workflow.core import ConversionContext, write_base_markdown

        cfg = MarkitaiConfig()
        cfg.llm.pure = True
        cfg.llm.enabled = False

        input_file = tmp_path / "sample.txt"
        input_file.write_text("# Hello\n\nSome content here.")

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.txt.md"
        ctx.conversion_result = ConvertResult(
            markdown="# Hello\n\nSome content here.",
            images=[],
            metadata={},
        )

        result = write_base_markdown(ctx)

        assert result.success
        assert ctx.output_file.exists()
        content = ctx.output_file.read_text()
        # Should NOT have frontmatter
        assert not content.startswith("---")
        # Should have the raw markdown
        assert "# Hello" in content
        assert "Some content here." in content

    def test_pure_without_llm_no_frontmatter(self, tmp_path):
        """Raw markdown should not contain any YAML frontmatter markers."""
        from markitai.workflow.core import ConversionContext, write_base_markdown

        cfg = MarkitaiConfig()
        cfg.llm.pure = True
        cfg.llm.enabled = False

        input_file = tmp_path / "sample.txt"
        input_file.write_text("Just plain text.")

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.txt.md"
        ctx.conversion_result = ConvertResult(
            markdown="Just plain text.",
            images=[],
            metadata={},
        )

        result = write_base_markdown(ctx)

        assert result.success
        content = ctx.output_file.read_text()
        assert content == "Just plain text."

    def test_default_mode_still_adds_frontmatter(self, tmp_path):
        """Default mode (no --pure, no --llm) should still add frontmatter."""
        from markitai.workflow.core import ConversionContext, write_base_markdown

        cfg = MarkitaiConfig()
        # Neither pure nor llm

        input_file = tmp_path / "sample.txt"
        input_file.write_text("# Hello")

        ctx = ConversionContext(
            input_path=input_file,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.txt.md"
        ctx.conversion_result = ConvertResult(
            markdown="# Hello",
            images=[],
            metadata={},
        )

        result = write_base_markdown(ctx)

        assert result.success
        content = ctx.output_file.read_text()
        assert content.startswith("---")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestPureWithoutLLMOutput -v`
Expected: FAIL — pure-without-LLM currently follows the default frontmatter path

- [ ] **Step 3: Implement pure-without-LLM branch**

In `packages/markitai/src/markitai/workflow/core.py`, modify `write_base_markdown()`:

```python
def write_base_markdown(ctx: ConversionContext) -> ConversionStepResult:
    """Write base markdown file with basic frontmatter.

    Decision tree:
    1. LLM enabled without --keep-base: skip writing (in-memory only)
    2. Pure mode without LLM: write raw markdown without frontmatter
    3. Default: write with frontmatter
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

    # Pure mode without LLM: write raw markdown without frontmatter
    if ctx.config.llm.pure and not ctx.config.llm.enabled:
        atomic_write_text(ctx.output_file, ctx.conversion_result.markdown)
        logger.debug(f"[Core] Written raw output (pure mode): {ctx.output_file}")
        return ConversionStepResult(success=True)

    # Default: write with frontmatter
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

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestPureWithoutLLMOutput -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `uv run pytest packages/markitai/tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/tests/unit/test_workflow_core.py packages/markitai/src/markitai/workflow/core.py
git commit -m "feat: --pure without --llm writes raw markdown without frontmatter"
```

---

## Chunk 2: Image Vision Path for `--llm --pure`

### Task 3: Add `process_image_with_vision_pure()` function

**Files:**
- Modify: `packages/markitai/tests/unit/test_workflow_core.py` (add tests)
- Modify: `packages/markitai/src/markitai/workflow/core.py` (add function)

- [ ] **Step 1: Write failing test for `process_image_with_vision_pure()`**

Add tests in `packages/markitai/tests/unit/test_workflow_core.py`:

```python
class TestProcessImageWithVisionPure:
    """Tests for process_image_with_vision_pure() — Vision analysis for --llm --pure + image."""

    @pytest.mark.asyncio
    async def test_calls_analyze_image_and_writes_llm_md(self, tmp_path, fixtures_dir):
        """Should call analyze_image() and write formatted result to .llm.md."""
        from unittest.mock import AsyncMock, MagicMock

        from markitai.converter.base import ConvertResult, FileFormat
        from markitai.llm.types import ImageAnalysis
        from markitai.workflow.core import ConversionContext, process_image_with_vision_pure

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True

        input_path = fixtures_dir / "sample.jpg"
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.jpg.md"
        ctx.detected_format = FileFormat.JPEG
        ctx.conversion_result = ConvertResult(
            markdown="# sample\n\n![sample](.markitai/assets/sample.jpg)",
            images=[],
            metadata={"asset_path": ".markitai/assets/sample.jpg"},
        )

        # Create the asset file so get_saved_images() finds it
        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        asset_file = assets_dir / "sample.jpg"
        asset_file.write_bytes(b"\xff\xd8\xff\xe0")  # JPEG header stub

        # Mock the processor
        mock_analysis = ImageAnalysis(
            caption="A sunset over mountains",
            description="A beautiful sunset with orange and purple hues over a mountain range.",
            extracted_text=None,
        )
        mock_processor = MagicMock()
        mock_processor.analyze_image = AsyncMock(return_value=mock_analysis)
        ctx.shared_processor = mock_processor

        result = await process_image_with_vision_pure(ctx)

        assert result.success
        mock_processor.analyze_image.assert_called_once()

        # Check .llm.md was written
        llm_file = tmp_path / "sample.jpg.llm.md"
        assert llm_file.exists()
        content = llm_file.read_text()
        assert "sample" in content
        assert "sunset" in content.lower() or "description" in content.lower()

    @pytest.mark.asyncio
    async def test_extracted_text_included_when_present(self, tmp_path, fixtures_dir):
        """When extracted_text is available, it should appear in the output."""
        from unittest.mock import AsyncMock, MagicMock

        from markitai.converter.base import ConvertResult, FileFormat
        from markitai.llm.types import ImageAnalysis
        from markitai.workflow.core import ConversionContext, process_image_with_vision_pure

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True

        input_path = fixtures_dir / "sample.jpg"
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.jpg.md"
        ctx.detected_format = FileFormat.JPEG
        ctx.conversion_result = ConvertResult(
            markdown="# sample\n\n![sample](.markitai/assets/sample.jpg)",
            images=[],
            metadata={"asset_path": ".markitai/assets/sample.jpg"},
        )

        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        asset_file = assets_dir / "sample.jpg"
        asset_file.write_bytes(b"\xff\xd8\xff\xe0")

        mock_analysis = ImageAnalysis(
            caption="Receipt",
            description="A scanned receipt from a grocery store.",
            extracted_text="Apples  $3.99\nBread   $2.49\nTotal   $6.48",
        )
        mock_processor = MagicMock()
        mock_processor.analyze_image = AsyncMock(return_value=mock_analysis)
        ctx.shared_processor = mock_processor

        result = await process_image_with_vision_pure(ctx)

        assert result.success
        llm_file = tmp_path / "sample.jpg.llm.md"
        content = llm_file.read_text()
        assert "Apples" in content
        assert "$6.48" in content

    @pytest.mark.asyncio
    async def test_failure_returns_error(self, tmp_path, fixtures_dir):
        """When Vision analysis fails, should return error result."""
        from unittest.mock import AsyncMock, MagicMock

        from markitai.converter.base import ConvertResult, FileFormat
        from markitai.workflow.core import ConversionContext, process_image_with_vision_pure

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True

        input_path = fixtures_dir / "sample.jpg"
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.jpg.md"
        ctx.detected_format = FileFormat.JPEG
        ctx.conversion_result = ConvertResult(
            markdown="# sample\n\n![sample](.markitai/assets/sample.jpg)",
            images=[],
            metadata={"asset_path": ".markitai/assets/sample.jpg"},
        )

        assets_dir = tmp_path / ".markitai" / "assets"
        assets_dir.mkdir(parents=True)
        asset_file = assets_dir / "sample.jpg"
        asset_file.write_bytes(b"\xff\xd8\xff\xe0")

        mock_processor = MagicMock()
        mock_processor.analyze_image = AsyncMock(
            side_effect=Exception("Vision model not configured")
        )
        ctx.shared_processor = mock_processor

        result = await process_image_with_vision_pure(ctx)

        assert not result.success
        assert "Vision" in result.error or "failed" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_saved_images_returns_error(self, tmp_path, fixtures_dir):
        """When no saved images found in assets, should return error."""
        from markitai.converter.base import ConvertResult, FileFormat
        from markitai.workflow.core import ConversionContext, process_image_with_vision_pure

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True

        input_path = fixtures_dir / "sample.jpg"
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.jpg.md"
        ctx.detected_format = FileFormat.JPEG
        ctx.conversion_result = ConvertResult(
            markdown="# sample\n\n![sample](.markitai/assets/sample.jpg)",
            images=[],
            metadata={},
        )
        # No assets directory — no saved images

        result = await process_image_with_vision_pure(ctx)

        assert not result.success
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestProcessImageWithVisionPure -v`
Expected: FAIL — `process_image_with_vision_pure` does not exist yet

- [ ] **Step 3: Implement `process_image_with_vision_pure()`**

Add in `packages/markitai/src/markitai/workflow/core.py`, after `process_with_pure_llm()` (after line 627):

```python
async def process_image_with_vision_pure(
    ctx: ConversionContext,
) -> ConversionStepResult:
    """Pure Vision mode: analyze standalone image with Vision model, write raw result.

    For --llm --pure with image-only inputs. Uses analyze_image() to get
    structured ImageAnalysis, then formats as markdown and writes to .llm.md.

    No frontmatter or post-processing is applied — the output is constructed
    directly from the ImageAnalysis structured fields.

    Args:
        ctx: Conversion context

    Returns:
        ConversionStepResult indicating success or failure
    """
    if ctx.conversion_result is None or ctx.output_file is None:
        return ConversionStepResult(success=False, error="Missing conversion result")

    # Get saved image from assets (handles transcoded formats via get_saved_images)
    saved_images = get_saved_images(ctx)
    if not saved_images:
        return ConversionStepResult(
            success=False,
            error=f"No saved image found for {ctx.input_path.name}",
        )

    image_path = saved_images[0]

    # Use shared processor or create new one
    from markitai.workflow.helpers import create_llm_processor

    processor = ctx.shared_processor
    if processor is None:
        processor = create_llm_processor(ctx.config)

    try:
        analysis = await processor.analyze_image(
            image_path, context=ctx.input_path.name
        )
    except Exception as e:
        return ConversionStepResult(
            success=False,
            error=f"Vision analysis failed: {format_error_message(e)}",
        )

    # Format output: # {filename}\n\n{description}\n\n{extracted_text}
    sections = [f"# {ctx.input_path.stem}\n"]

    if analysis.description:
        sections.append(f"{analysis.description.strip()}\n")

    if analysis.extracted_text and analysis.extracted_text.strip():
        sections.append(f"{analysis.extracted_text.strip()}\n")

    content = "\n".join(sections)

    # Write to .llm.md
    llm_output = ctx.output_file.with_suffix(".llm.md")
    atomic_write_text(llm_output, content)
    logger.info(f"[Core] Written pure Vision output: {llm_output}")

    # Track cost if available
    if analysis.llm_usage:
        merge_llm_usage(ctx.llm_usage, analysis.llm_usage)

    return ConversionStepResult(success=True)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestProcessImageWithVisionPure -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/tests/unit/test_workflow_core.py packages/markitai/src/markitai/workflow/core.py
git commit -m "feat: add process_image_with_vision_pure() for --llm --pure image analysis"
```

---

### Task 4: Route `--pure + image` to Vision path in Step 7

**Files:**
- Modify: `packages/markitai/tests/unit/test_workflow_core.py` (add routing test)
- Modify: `packages/markitai/src/markitai/workflow/core.py:1016-1022` (Step 7 routing)

- [ ] **Step 1: Write failing test for routing logic**

Add test in `packages/markitai/tests/unit/test_workflow_core.py`:

```python
class TestPureImageRouting:
    """Tests for --pure Step 7 routing: image-only → Vision path, others → text path."""

    @pytest.mark.asyncio
    async def test_pure_image_routes_to_vision(self, tmp_path, fixtures_dir):
        """--llm --pure with image input should route to process_image_with_vision_pure."""
        from unittest.mock import AsyncMock, patch

        from markitai.converter.base import ConvertResult, FileFormat
        from markitai.workflow.core import ConversionContext, convert_document_core

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True

        input_path = fixtures_dir / "sample.jpg"
        ctx = ConversionContext(
            input_path=input_path,
            output_dir=tmp_path,
            config=cfg,
        )

        mock_vision = AsyncMock(return_value=ConversionStepResult(success=True))
        mock_pure_llm = AsyncMock(return_value=ConversionStepResult(success=True))

        with (
            patch("markitai.workflow.core.validate_and_detect_format") as mock_validate,
            patch("markitai.workflow.core.prepare_output_directory") as mock_prep,
            patch("markitai.workflow.core.resolve_output_file") as mock_resolve,
            patch("markitai.workflow.core.convert_document") as mock_convert,
            patch("markitai.workflow.core.process_embedded_images") as mock_embed,
            patch("markitai.workflow.core.write_base_markdown") as mock_write,
            patch("markitai.workflow.core.process_image_with_vision_pure", mock_vision),
            patch("markitai.workflow.core.process_with_pure_llm", mock_pure_llm),
        ):
            mock_validate.return_value = ConversionStepResult(success=True)
            mock_prep.return_value = ConversionStepResult(success=True)
            mock_resolve.return_value = ConversionStepResult(success=True)
            mock_convert.return_value = ConversionStepResult(success=True)
            mock_embed.return_value = ConversionStepResult(success=True)
            mock_write.return_value = ConversionStepResult(success=True)

            # Simulate format detection setting detected_format
            def set_format(c, max_size=None):
                c.detected_format = FileFormat.JPEG
                c.output_file = tmp_path / "sample.jpg.md"
                c.conversion_result = ConvertResult(
                    markdown="# sample", images=[], metadata={}
                )
                return ConversionStepResult(success=True)

            mock_validate.side_effect = set_format

            await convert_document_core(ctx, max_document_size=500_000_000)

            mock_vision.assert_called_once()
            mock_pure_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_pure_non_image_routes_to_text_llm(self, tmp_path, fixtures_dir):
        """--llm --pure with non-image input should route to process_with_pure_llm."""
        from unittest.mock import AsyncMock, patch

        from markitai.converter.base import ConvertResult, FileFormat
        from markitai.workflow.core import ConversionContext, convert_document_core

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True

        input_file = tmp_path / "sample.txt"
        input_file.write_text("# sample")
        ctx = ConversionContext(
            input_path=input_file,
            output_dir=tmp_path,
            config=cfg,
        )

        mock_vision = AsyncMock(return_value=ConversionStepResult(success=True))
        mock_pure_llm = AsyncMock(return_value=ConversionStepResult(success=True))

        with (
            patch("markitai.workflow.core.validate_and_detect_format") as mock_validate,
            patch("markitai.workflow.core.prepare_output_directory") as mock_prep,
            patch("markitai.workflow.core.resolve_output_file") as mock_resolve,
            patch("markitai.workflow.core.convert_document") as mock_convert,
            patch("markitai.workflow.core.process_embedded_images") as mock_embed,
            patch("markitai.workflow.core.write_base_markdown") as mock_write,
            patch("markitai.workflow.core.process_image_with_vision_pure", mock_vision),
            patch("markitai.workflow.core.process_with_pure_llm", mock_pure_llm),
        ):
            mock_validate.return_value = ConversionStepResult(success=True)
            mock_prep.return_value = ConversionStepResult(success=True)
            mock_resolve.return_value = ConversionStepResult(success=True)
            mock_convert.return_value = ConversionStepResult(success=True)
            mock_embed.return_value = ConversionStepResult(success=True)
            mock_write.return_value = ConversionStepResult(success=True)

            def set_format(c, max_size=None):
                c.detected_format = FileFormat.TXT
                c.output_file = tmp_path / "sample.txt.md"
                c.conversion_result = ConvertResult(
                    markdown="# sample", images=[], metadata={}
                )
                return ConversionStepResult(success=True)

            mock_validate.side_effect = set_format

            await convert_document_core(ctx, max_document_size=500_000_000)

            mock_pure_llm.assert_called_once()
            mock_vision.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestPureImageRouting -v`
Expected: FAIL — current code always calls `process_with_pure_llm` regardless of format

- [ ] **Step 3: Update Step 7 routing logic**

In `packages/markitai/src/markitai/workflow/core.py`, modify lines 1016-1022:

```python
# Before:
        if ctx.config.llm.pure and not ctx.config.screenshot.screenshot_only:
            # Pure mode: raw MD → LLM → .llm.md, nothing else
            # --screenshot-only takes precedence over --pure (mutually exclusive)
            result = await process_with_pure_llm(ctx)
            if not result.success:
                _write_base_md_fallback(ctx)
                return result

# After:
        if ctx.config.llm.pure and not ctx.config.screenshot.screenshot_only:
            # Pure mode: --screenshot-only takes precedence (mutually exclusive)
            from markitai.converter.base import IMAGE_ONLY_FORMATS

            if ctx.detected_format in IMAGE_ONLY_FORMATS:
                # Image input: use Vision model to analyze the actual image
                result = await process_image_with_vision_pure(ctx)
            else:
                # Non-image: raw MD → LLM text cleaning → .llm.md
                result = await process_with_pure_llm(ctx)
            if not result.success:
                _write_base_md_fallback(ctx)
                return result
```

Note: `IMAGE_ONLY_FORMATS` is already imported at line 962 (Step 1.5), but that import is inside the `convert_document_core()` function scope at a different block. We need to import it again in this block, or move the import to the top of the function. Since the existing pattern uses local imports, we follow the same pattern.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestPureImageRouting -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest packages/markitai/tests/ -x -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/tests/unit/test_workflow_core.py packages/markitai/src/markitai/workflow/core.py
git commit -m "feat: route --llm --pure image inputs to Vision analysis path"
```

---

### Task 5: Verify behavior matrix with integration-level tests

**Files:**
- Modify: `packages/markitai/tests/unit/test_workflow_core.py` (add behavior matrix tests)

- [ ] **Step 1: Write tests for key behavior matrix scenarios**

Add tests verifying the complete behavior matrix from the spec:

```python
class TestPureDecouplingBehaviorMatrix:
    """Verify the behavior matrix from the design spec."""

    def test_pure_alone_image_triggers_rule_a_skip(self, tmp_path, fixtures_dir):
        """--pure alone + image → Rule A skip (llm.enabled=False, ocr.enabled=False)."""
        from markitai.converter.base import IMAGE_ONLY_FORMATS, FileFormat
        from markitai.workflow.core import ConversionContext, ConversionStepResult

        cfg = MarkitaiConfig()
        cfg.llm.pure = True
        # llm.enabled and ocr.enabled are both False

        ctx = ConversionContext(
            input_path=fixtures_dir / "sample.jpg",
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.detected_format = FileFormat.JPEG

        # Rule A check
        assert ctx.detected_format in IMAGE_ONLY_FORMATS
        assert not ctx.config.llm.enabled
        assert not ctx.config.ocr.enabled
        # This combination should trigger Rule A skip in convert_document_core

    def test_pure_alone_non_image_writes_raw_md(self, tmp_path):
        """--pure alone + non-image → .md without frontmatter."""
        from markitai.converter.base import ConvertResult
        from markitai.workflow.core import ConversionContext, write_base_markdown

        cfg = MarkitaiConfig()
        cfg.llm.pure = True

        input_file = tmp_path / "sample.txt"
        input_file.write_text("# Test content")
        ctx = ConversionContext(
            input_path=input_file,
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.output_file = tmp_path / "sample.txt.md"
        ctx.conversion_result = ConvertResult(
            markdown="# Test content",
            images=[],
            metadata={},
        )

        result = write_base_markdown(ctx)
        assert result.success
        content = ctx.output_file.read_text()
        assert not content.startswith("---")
        assert "# Test content" in content

    def test_llm_pure_image_does_not_hit_rule_a(self, tmp_path, fixtures_dir):
        """--llm --pure + image → llm.enabled=True, Rule A does NOT trigger."""
        from markitai.converter.base import IMAGE_ONLY_FORMATS, FileFormat

        cfg = MarkitaiConfig()
        cfg.llm.enabled = True
        cfg.llm.pure = True

        from markitai.workflow.core import ConversionContext

        ctx = ConversionContext(
            input_path=fixtures_dir / "sample.jpg",
            output_dir=tmp_path,
            config=cfg,
        )
        ctx.detected_format = FileFormat.JPEG

        # Rule A should NOT trigger because llm.enabled=True
        rule_a_triggers = (
            ctx.detected_format in IMAGE_ONLY_FORMATS
            and not ctx.config.llm.enabled
            and not ctx.config.ocr.enabled
        )
        assert not rule_a_triggers
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_core.py::TestPureDecouplingBehaviorMatrix -v`
Expected: PASS

- [ ] **Step 3: Run lint and type checks**

Run: `uv run ruff check packages/markitai/src packages/markitai/tests && uv run pyright packages/markitai/src/markitai/workflow/core.py`
Expected: Clean

- [ ] **Step 4: Commit**

```bash
git add packages/markitai/tests/unit/test_workflow_core.py
git commit -m "test: add behavior matrix verification tests for --pure decoupling"
```

---

### Task 6: Final verification

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest packages/markitai/tests/ -q`
Expected: All tests pass, no regressions

- [ ] **Step 2: Run pre-commit hooks**

Run: `uv run pre-commit run --all-files`
Expected: All hooks pass

- [ ] **Step 3: Verify with the original failing command (manual)**

The user's original test command that exposed the issue:
```bash
rimraf logs output; markitai packages/markitai/tests/fixtures/ --no-cache -o output --verbose --pure
```

Expected behavior after fix:
- Image files (sample.jpg, sample.bmp, sample.gif): Rule A skip, terminal warning
- Non-image files: `.md` output without frontmatter, no `.llm.md` files
- No LLM calls made (since `--llm` was not passed)
