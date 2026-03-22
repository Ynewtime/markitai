# Code Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 6 confirmed issues from the 2026-03-22 code review (P0-P2), each with a failing test first.

**Architecture:** Each fix is independent and modifiable in isolation. TDD: write the failing test, implement the minimal fix, verify. No refactors beyond what's needed.

**Tech Stack:** Python 3.11+, pytest, unittest.mock

**Reference:** `docs/code-review-2026-03-22.md`

---

## File Map

| Fix | Source file | Test file |
|-----|-----------|-----------|
| 1.1 PDF temp_dir | `packages/markitai/src/markitai/converter/pdf.py` | `packages/markitai/tests/unit/test_converter_pdf.py` |
| 1.2 HTML sanitize | `packages/markitai/src/markitai/webextract/sanitize.py` | `packages/markitai/tests/unit/webextract/test_sanitize.py` |
| 2.1 Render depth | `packages/markitai/src/markitai/webextract/render.py` | `packages/markitai/tests/unit/webextract/test_render_thread.py` |
| 2.2 LLM stderr | `packages/markitai/src/markitai/workflow/single.py` | `packages/markitai/tests/unit/test_workflow_single.py` |
| 2.3 COM split | `packages/markitai/src/markitai/converter/office.py` | (manual Windows verification only) |
| 3.1 CJK tokens | `packages/markitai/src/markitai/providers/__init__.py` | `packages/markitai/tests/unit/test_providers.py` |

---

## Task 1: PDF temp_dir cleanup (P0-1)

**Files:**
- Modify: `packages/markitai/src/markitai/converter/pdf.py:183-345`
- Test: `packages/markitai/tests/unit/test_converter_pdf.py`

- [ ] **Step 1: Write failing test — temp_dir leaked on exception**

Add to `packages/markitai/tests/unit/test_converter_pdf.py`:

```python
class TestPdfTempDirCleanup:
    """Tests for temp_dir cleanup on exception paths."""

    @patch("markitai.converter.pdf.pymupdf4llm")
    @patch("markitai.converter.pdf.tempfile.mkdtemp")
    def test_temp_dir_cleaned_on_exception(
        self, mock_mkdtemp: Mock, mock_pymupdf4llm: Mock, tmp_path: Path
    ) -> None:
        """Temp directory is removed even when conversion throws."""
        leaked_dir = tmp_path / "leaked_temp"
        leaked_dir.mkdir()
        mock_mkdtemp.return_value = str(leaked_dir)
        mock_pymupdf4llm.to_markdown.side_effect = RuntimeError("corrupt PDF")

        converter = PdfConverter()
        with pytest.raises(RuntimeError, match="corrupt PDF"):
            converter.convert(Path("fake.pdf"), output_dir=None)

        assert not leaked_dir.exists(), "temp_dir should be cleaned up on exception"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && python -m pytest tests/unit/test_converter_pdf.py::TestPdfTempDirCleanup::test_temp_dir_cleaned_on_exception -v -x -p no:xdist`
Expected: FAIL — `leaked_dir` still exists because cleanup is not in finally block.

- [ ] **Step 3: Implement fix — wrap convert() body in try-finally**

In `packages/markitai/src/markitai/converter/pdf.py`, wrap lines 183–345 of the `convert()` method. The change:

```python
        # Determine image output path
        temp_dir: Path | None = None
        try:  # <-- ADD THIS
            if output_dir:
                image_path = ensure_assets_dir(output_dir)
                write_images = True
            else:
                # Use temp directory if no output dir specified
                temp_dir = Path(tempfile.mkdtemp())
                image_path = temp_dir
                write_images = True

            # ... entire existing body unchanged through line 339 ...

            return ConvertResult(
                markdown=markdown,
                images=images,
                metadata=metadata,
            )
        finally:  # <-- ADD THIS
            # Guarantee temp dir cleanup even on exception
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
```

Remove the old cleanup block (lines 333-339) and move its logic into the normal return path (before `return ConvertResult`):

```python
            # Clean up temporary directory if used
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
                images = [img for img in images if img.path and img.path.exists()]
                metadata.pop("reference_images", None)

            return ConvertResult(
                markdown=markdown,
                images=images,
                metadata=metadata,
            )
        finally:
            # Guarantee cleanup on exception path too
            if temp_dir and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/markitai && python -m pytest tests/unit/test_converter_pdf.py::TestPdfTempDirCleanup -v -x -p no:xdist`
Expected: PASS

- [ ] **Step 5: Run full PDF converter test suite**

Run: `cd packages/markitai && python -m pytest tests/unit/test_converter_pdf.py -v -p no:xdist`
Expected: All existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/converter/pdf.py packages/markitai/tests/unit/test_converter_pdf.py
git commit -m "fix: ensure PDF temp_dir cleanup in finally block

Wraps the convert() body in try-finally so temp directories are cleaned
up even when pymupdf4llm or image processing throws an exception.
Prevents disk space leaks during batch PDF processing."
```

---

## Task 2: HTML sanitize URL-decode enhancement (P0-2)

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/sanitize.py`
- Test: `packages/markitai/tests/unit/webextract/test_sanitize.py`

- [ ] **Step 1: Write failing test — URL-encoded javascript bypass**

Add to `packages/markitai/tests/unit/webextract/test_sanitize.py`:

```python
def test_sanitize_removes_url_encoded_javascript() -> None:
    """URL-encoded javascript: should be caught after decoding."""
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment(
        '<a href="javascript%3Aalert(1)">click</a>'
    )
    assert "javascript" not in sanitized.lower() or 'href' not in sanitized


def test_sanitize_removes_vbscript_links() -> None:
    """vbscript: scheme should be removed."""
    from markitai.webextract.sanitize import sanitize_html_fragment

    sanitized = sanitize_html_fragment(
        '<a href="vbscript:evil()">click</a>'
    )
    assert "vbscript" not in sanitized.lower() or 'href' not in sanitized


def test_sanitize_checks_formaction_attribute() -> None:
    """formaction with javascript: should be removed."""
    from markitai.webextract.sanitize import sanitize_html_fragment

    # Note: input/button/form tags are in REMOVE_TAGS and get decomposed entirely,
    # but we test that even if a non-removed tag has formaction, it's cleaned
    sanitized = sanitize_html_fragment(
        '<div formaction="javascript:evil()"><p>safe</p></div>'
    )
    assert "formaction" not in sanitized
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/markitai && python -m pytest tests/unit/webextract/test_sanitize.py -v -x -p no:xdist`
Expected: FAIL — URL-encoded `javascript%3A` is not caught; `vbscript:` not in prefix list; `formaction` not checked.

- [ ] **Step 3: Implement fix**

Replace content of `packages/markitai/src/markitai/webextract/sanitize.py`:

```python
from __future__ import annotations

from urllib.parse import unquote

from bs4 import BeautifulSoup, Tag

UNSAFE_URL_PREFIXES = (
    "javascript:",
    "data:text/html",
    "data:image/svg+xml",
    "data:text/javascript",
    "vbscript:",
)
REMOVE_TAGS = {
    "script",
    "style",
    "object",
    "embed",
    "iframe",
    "noscript",
    "form",
    "button",
    "input",
    "textarea",
    "select",
}
_URL_ATTRS = ("href", "src", "action", "formaction")


def sanitize_html_fragment(html: str) -> str:
    """Remove unsafe attributes, links, and obvious noise tags.

    Args:
        html: HTML fragment to sanitize.

    Returns:
        Sanitized HTML string.
    """

    soup = BeautifulSoup(html, "html.parser")
    for tag in list(soup.find_all(True)):
        _sanitize_tag(tag)
    return str(soup)


def sanitize_tag_tree(root: Tag) -> None:
    """Sanitize a parsed tag tree in place.

    Args:
        root: Root tag to sanitize.
    """

    for tag in list(root.find_all(True)):
        _sanitize_tag(tag)


def _sanitize_tag(tag: Tag) -> None:
    if tag.name in REMOVE_TAGS:
        # Preserve checkbox inputs for task list support
        if tag.name == "input" and tag.get("type") == "checkbox":
            return
        tag.decompose()
        return

    if not tag.attrs:
        return

    for attr in list(tag.attrs):
        if attr.startswith("on"):
            del tag.attrs[attr]

    for attr in _URL_ATTRS:
        value = tag.get(attr)
        if isinstance(value, str):
            # Decode URL encoding before checking to prevent bypass via %3A etc.
            decoded = unquote(value).strip().lower()
            if decoded.startswith(UNSAFE_URL_PREFIXES):
                del tag.attrs[attr]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/markitai && python -m pytest tests/unit/webextract/test_sanitize.py -v -p no:xdist`
Expected: All PASS (old tests + new tests).

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/sanitize.py packages/markitai/tests/unit/webextract/test_sanitize.py
git commit -m "fix: harden HTML sanitization with URL-decode and broader attribute coverage

Adds urllib.parse.unquote() before prefix matching to catch encoded
bypasses like javascript%3A. Extends checks to formaction/action attrs.
Adds vbscript: and data:text/javascript to unsafe prefix list."
```

---

## Task 3: Render depth limit (P1-1)

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/render.py:136-148`
- Test: `packages/markitai/tests/unit/webextract/test_render_thread.py`

- [ ] **Step 1: Write failing test — deep nesting causes RecursionError**

Add to `packages/markitai/tests/unit/webextract/test_render_thread.py`:

```python
class TestRenderDepthLimit:
    """Tests for reply nesting depth limit."""

    def test_deeply_nested_thread_does_not_crash(self) -> None:
        """A thread with 200+ levels of nesting should not hit RecursionError."""
        # Build a chain: item_0 <- item_1 <- item_2 <- ... <- item_199
        items = []
        for i in range(200):
            parent = "root" if i == 0 else f"item_{i - 1}"
            items.append(
                ConversationItem(
                    id=f"item_{i}",
                    author_name=f"user_{i}",
                    text=f"Reply level {i}",
                    parent_id=parent,
                )
            )

        thread = _make_thread(replies=items)
        # Should not raise RecursionError
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "item_0" in html  # top-level reply rendered
        assert "Reply level 0" in html

    def test_items_beyond_depth_limit_are_truncated(self) -> None:
        """Items beyond MAX_REPLY_DEPTH should not produce nested blockquotes."""
        from markitai.webextract.render import _MAX_REPLY_DEPTH

        depth = _MAX_REPLY_DEPTH + 10
        items = []
        for i in range(depth):
            parent = "root" if i == 0 else f"item_{i - 1}"
            items.append(
                ConversationItem(
                    id=f"item_{i}",
                    author_name=f"user_{i}",
                    text=f"Reply level {i}",
                    parent_id=parent,
                )
            )

        thread = _make_thread(replies=items)
        html = render_semantic_content(SemanticExtraction(thread=thread))

        # Count nesting depth by counting 'reply-thread' blockquotes
        nesting_count = html.count('class="reply-thread"')
        assert nesting_count <= _MAX_REPLY_DEPTH
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && python -m pytest tests/unit/webextract/test_render_thread.py::TestRenderDepthLimit::test_items_beyond_depth_limit_are_truncated -v -x -p no:xdist`
Expected: FAIL — `_MAX_REPLY_DEPTH` doesn't exist yet (ImportError), or if constant is added first, nesting_count exceeds limit because all items are rendered without depth cap.

Note: `test_deeply_nested_thread_does_not_crash` is a regression guard — 200 levels may not hit Python's 1000-frame recursion limit. The reliable failing-first test is `test_items_beyond_depth_limit_are_truncated`.

- [ ] **Step 3: Implement fix — add depth parameter**

In `packages/markitai/src/markitai/webextract/render.py`:

Add after line 21 (after the imports):

```python
_MAX_REPLY_DEPTH = 50
```

Replace `_render_item_tree` (lines 136-148):

```python
def _render_item_tree(
    item: ConversationItem,
    items: list[ConversationItem],
    depth: int = 0,
) -> str:
    """Render a conversation item along with any nested replies.

    Stops recursing beyond ``_MAX_REPLY_DEPTH`` to prevent stack overflow
    on deeply nested threads (e.g. Reddit, HackerNews).
    """
    parts = [_render_item(item)]
    if depth < _MAX_REPLY_DEPTH:
        children = _iter_child_items(item.id, items)
        if children:
            parts.append('<blockquote class="reply-thread">')
            for child in children:
                parts.append(_render_item_tree(child, items, depth + 1))
            parts.append("</blockquote>")
    return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/markitai && python -m pytest tests/unit/webextract/test_render_thread.py -v -p no:xdist`
Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/render.py packages/markitai/tests/unit/webextract/test_render_thread.py
git commit -m "fix: add depth limit to thread renderer to prevent RecursionError

Caps nested reply rendering at 50 levels. Threads deeper than this
(e.g. Reddit, HackerNews) are truncated rather than crashing with
a stack overflow."
```

---

## Task 4: LLM stderr warnings for silent exception paths (P1-2)

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/single.py:221-223, 385-387, 527-531`
- Test: `packages/markitai/tests/unit/test_workflow_single.py`

- [ ] **Step 1: Write failing test — process_document_pure silent failure**

Add to `packages/markitai/tests/unit/test_workflow_single.py`.

**Important:** The `mock_config` and `mock_processor` fixtures are defined inside `TestSingleFileWorkflow` (lines 45-64) and are not accessible from a new class. Duplicate them in the new class, or move them to module-level/conftest. The simplest approach is to duplicate them:

```python
class TestLLMFailureStderrWarnings:
    """Tests that LLM failures produce stderr warnings, not silent fallbacks."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.llm.concurrency = 5
        config.image.alt_enabled = True
        config.image.desc_enabled = True
        config.cache.no_cache = False
        config.cache.no_cache_patterns = []
        return config

    @pytest.fixture
    def mock_processor(self):
        processor = MagicMock()
        processor.get_context_cost = MagicMock(return_value=0.05)
        processor.get_context_usage = MagicMock(
            return_value={"gpt-4": {"requests": 1, "input_tokens": 100}}
        )
        return processor

    @pytest.mark.asyncio
    async def test_process_document_pure_warns_on_failure(
        self, mock_config, mock_processor, capsys
    ) -> None:
        """process_document_pure should print warning to stderr on exception."""
        mock_processor.clean_document_pure = AsyncMock(
            side_effect=RuntimeError("API error")
        )
        workflow = SingleFileWorkflow(mock_config, mock_processor)
        result = await workflow.process_document_pure(
            "# test", "test.md", Path("/tmp/test.md")
        )
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower() or "failed" in captured.err.lower()

    @pytest.mark.asyncio
    async def test_analyze_images_warns_on_failure(
        self, mock_config, mock_processor, capsys
    ) -> None:
        """analyze_images should print warning to stderr on exception."""
        mock_processor.analyze_images_batch = AsyncMock(
            side_effect=RuntimeError("Vision API error")
        )
        workflow = SingleFileWorkflow(mock_config, mock_processor)
        result = await workflow.analyze_images(
            [Path("/tmp/img.png")], "# test", Path("/tmp/test.md")
        )
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower() or "failed" in captured.err.lower()

    @pytest.mark.asyncio
    async def test_extract_from_screenshots_warns_on_failure(
        self, mock_config, mock_processor, capsys
    ) -> None:
        """extract_from_screenshots should print warning to stderr on exception."""
        mock_processor.extract_from_screenshots = AsyncMock(
            side_effect=RuntimeError("Screenshot extraction error")
        )
        workflow = SingleFileWorkflow(mock_config, mock_processor)
        result = await workflow.extract_from_screenshots(
            page_images=[{"path": "/tmp/page1.png", "page": 1}],
            source="test.pdf",
        )
        captured = capsys.readouterr()
        assert "warning" in captured.err.lower() or "failed" in captured.err.lower()
```

Note: The test fixtures `mock_config` and `mock_processor` already exist in the file. `AsyncMock` may need to be imported from `unittest.mock`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd packages/markitai && python -m pytest tests/unit/test_workflow_single.py::TestLLMFailureStderrWarnings -v -x -p no:xdist`
Expected: FAIL — captured stderr is empty because these three paths only call `logger.error()`.

- [ ] **Step 3: Implement fix — add stderr warnings to 3 exception handlers**

In `packages/markitai/src/markitai/workflow/single.py`:

**At line 221-223** (`process_document_pure` exception handler):

```python
        except Exception as e:
            logger.error(f"Pure LLM processing failed: {format_error_message(e)}")
            from rich.console import Console

            Console(stderr=True).print(
                f"[yellow]Warning: Pure LLM processing failed: {format_error_message(e)}[/yellow]"
            )
            return markdown, 0.0, {}
```

**At line 385-387** (`analyze_images` exception handler):

```python
        except Exception as e:
            logger.error(f"Image analysis failed: {format_error_message(e)}")
            from rich.console import Console

            Console(stderr=True).print(
                f"[yellow]Warning: Image analysis failed: {format_error_message(e)}[/yellow]"
            )
            return markdown, 0.0, {}, None
```

**At line 527-531** (`extract_from_screenshots` exception handler):

```python
        except Exception as e:
            logger.error(
                f"Screenshot-only extraction failed: {format_error_message(e)}"
            )
            from rich.console import Console

            Console(stderr=True).print(
                f"[yellow]Warning: Screenshot-only extraction failed: {format_error_message(e)}[/yellow]"
            )
            return "", _fallback_frontmatter(source, original_title), 0.0, {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/markitai && python -m pytest tests/unit/test_workflow_single.py::TestLLMFailureStderrWarnings -v -p no:xdist`
Expected: All PASS.

- [ ] **Step 5: Run full workflow single test suite**

Run: `cd packages/markitai && python -m pytest tests/unit/test_workflow_single.py -v -p no:xdist`
Expected: All existing tests still PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/workflow/single.py packages/markitai/tests/unit/test_workflow_single.py
git commit -m "fix: add stderr warnings for silent LLM failure paths

process_document_pure, analyze_images, and extract_from_screenshots
now print a visible warning to stderr when they catch exceptions,
matching the pattern already used by process_document and
enhance_with_vision."
```

---

## Task 5: COM export/compress separation (P1-3, optimization)

**Files:**
- Modify: `packages/markitai/src/markitai/converter/office.py:246-346`
- Test: Manual Windows verification only (COM is Windows-only)

- [ ] **Step 1: Refactor _render_slides_with_com — separate phases**

In `packages/markitai/src/markitai/converter/office.py`, replace `_render_slides_with_com` (lines 246-346):

```python
    def _render_slides_with_com(
        self, input_path: Path, screenshots_dir: Path, image_format: str
    ) -> tuple[list[ExtractedImage], list[dict]]:
        """Render slides using PowerPoint COM automation.

        Two-phase design:
        1. Export slides to image files inside COM context (fast, minimal work)
        2. Compress images outside COM context (CPU-intensive, safe to fail partially)
        """
        import pythoncom  # type: ignore[import-not-found]
        import win32com.client  # type: ignore[import-not-found]

        logger.debug(f"Rendering slides with PowerPoint COM: {input_path.name}")

        export_format = "JPG" if image_format == "jpg" else image_format.upper()
        exported_slides: list[tuple[int, Path, str]] = []

        # Phase 1: Export all slides (inside COM context — keep it minimal)
        pythoncom.CoInitialize()
        ppt = None
        presentation = None
        try:
            ppt = win32com.client.Dispatch("PowerPoint.Application")
            presentation = ppt.Presentations.Open(
                str(input_path.resolve()),
                ReadOnly=True,
                Untitled=False,
                WithWindow=False,
            )

            for i, slide in enumerate(presentation.Slides, 1):
                image_name = f"{input_path.name}.slide{i:04d}.{image_format}"
                image_path = screenshots_dir / image_name
                slide.Export(str(image_path.resolve()), export_format)
                exported_slides.append((i, image_path, image_name))
                logger.debug(f"Exported slide {i}/{len(presentation.Slides)}")

            presentation.Close()
            presentation = None

        finally:
            if presentation:
                try:
                    presentation.Close()
                except Exception as e:
                    logger.debug("[PPTX] COM presentation.Close() failed: {}", e)
            if ppt:
                try:
                    ppt.Quit()
                except Exception as e:
                    logger.debug("[PPTX] COM ppt.Quit() failed: {}", e)
            pythoncom.CoUninitialize()

        # Phase 2: Compress images and collect results (outside COM context)
        img_processor = ImageProcessor(self.config.image if self.config else None)
        images: list[ExtractedImage] = []
        slide_images: list[dict] = []

        for i, image_path, image_name in exported_slides:
            from PIL import Image

            width, height = 0, 0
            try:
                with Image.open(image_path) as img:
                    original_width, original_height = img.size

                    if self.config and self.config.image.compress:
                        format_map = {
                            "jpg": "JPEG",
                            "jpeg": "JPEG",
                            "png": "PNG",
                            "webp": "WEBP",
                        }
                        output_format = format_map.get(image_format, "JPEG")
                        compressed_img, compressed_data = img_processor.compress(
                            img.copy(),
                            quality=self.config.image.quality,
                            max_size=(
                                self.config.image.max_width,
                                self.config.image.max_height,
                            ),
                            output_format=output_format,
                        )
                        image_path.write_bytes(compressed_data)
                        width, height = compressed_img.size
                    else:
                        width, height = original_width, original_height
            except Exception as e:
                logger.debug(
                    "[PPTX] Image compression failed for slide {}: {}", i, e
                )
                # Fall back to original dimensions if compression fails
                try:
                    with Image.open(image_path) as img:
                        width, height = img.size
                except Exception:
                    pass

            images.append(
                ExtractedImage(
                    path=image_path,
                    index=i,
                    original_name=image_name,
                    mime_type=f"image/{image_format}",
                    width=width,
                    height=height,
                )
            )
            slide_images.append(
                {
                    "page": i,
                    "path": str(image_path),
                    "name": image_name,
                }
            )

        return images, slide_images
```

- [ ] **Step 2: Verify no other code calls _render_slides_with_com with different signature**

Run: `grep -rn "_render_slides_with_com" packages/markitai/src/`
Expected: Only the definition and one call site in the same file.

- [ ] **Step 3: Commit**

```bash
git add packages/markitai/src/markitai/converter/office.py
git commit -m "refactor: separate COM export from image compression in PPTX renderer

Splits _render_slides_with_com into two phases: COM slide export first,
then image compression after COM cleanup. This ensures a compression
failure on one slide doesn't prevent subsequent slides from exporting,
and keeps the COM context minimal."
```

---

## Task 6: CJK-aware token estimation (P2-1)

**Files:**
- Modify: `packages/markitai/src/markitai/providers/__init__.py:135-138`
- Test: `packages/markitai/tests/unit/test_providers.py`

- [ ] **Step 1: Write failing test — Chinese text token undercount**

Add to `packages/markitai/tests/unit/test_providers.py`:

```python
class TestCountTokensFallback:
    """Tests for the character-based token estimation fallback."""

    def test_english_text_estimation(self) -> None:
        """English text: ~1 token per 4 chars."""
        from markitai.providers import count_tokens

        # Use a non-OpenAI model to force fallback estimation
        result = count_tokens("Hello world, this is a test.", "claude-sonnet-4.5")
        # 28 chars / 4 ≈ 7 tokens — should be in reasonable range
        assert 5 <= result <= 15

    def test_chinese_text_estimation_not_undercounted(self) -> None:
        """Chinese text should not be estimated at 1/4 char per token."""
        from markitai.providers import count_tokens

        chinese_text = "这是一段中文文本用于测试令牌计数的准确性"  # 19 CJK chars
        result = count_tokens(chinese_text, "claude-sonnet-4.5")
        # Old behavior: 19*3 bytes // 4 ≈ 14 (for UTF-8 len) or 19 // 4 = 4
        # CJK should estimate ~1.5 tokens per char = ~28
        assert result >= 15, f"Chinese token count {result} is too low (undercounted)"

    def test_mixed_cjk_and_english(self) -> None:
        """Mixed content should count CJK and English differently."""
        from markitai.providers import count_tokens

        mixed = "Hello 你好世界 World"  # 12 non-CJK + 4 CJK chars
        result = count_tokens(mixed, "deepseek-chat")
        # CJK portion: 4 * 1.5 = 6, English portion: 12 // 4 = 3 → ~9
        assert result >= 6, f"Mixed token count {result} is too low"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/markitai && python -m pytest tests/unit/test_providers.py::TestCountTokensFallback::test_chinese_text_estimation_not_undercounted -v -x -p no:xdist`
Expected: FAIL — `assert result >= 15` fails because `len("这是...") // 4` gives a very small number.

- [ ] **Step 3: Implement fix — CJK-aware estimation**

In `packages/markitai/src/markitai/providers/__init__.py`, replace lines 135-138:

```python
    # Fallback: character-based estimation with CJK awareness
    # English: ~1 token per 4 characters
    # CJK (Chinese/Japanese/Korean): ~1.5 tokens per character
    return _estimate_tokens_by_chars(text)
```

Add the helper function before `count_tokens` (e.g., after line 94):

```python
def _estimate_tokens_by_chars(text: str) -> int:
    """Estimate token count with CJK-aware character classification.

    English/Latin text averages ~1 token per 4 characters.
    CJK characters average ~1.5 tokens per character.
    """
    cjk_count = 0
    for ch in text:
        cp = ord(ch)
        if (
            0x4E00 <= cp <= 0x9FFF      # CJK Unified Ideographs
            or 0x3400 <= cp <= 0x4DBF    # CJK Extension A
            or 0x3000 <= cp <= 0x303F    # CJK Symbols and Punctuation
            or 0xFF00 <= cp <= 0xFFEF    # Fullwidth Forms
            or 0xAC00 <= cp <= 0xD7AF    # Hangul Syllables
            or 0x3040 <= cp <= 0x30FF    # Hiragana + Katakana
        ):
            cjk_count += 1
    non_cjk_count = len(text) - cjk_count
    return int(cjk_count * 1.5) + non_cjk_count // 4


```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd packages/markitai && python -m pytest tests/unit/test_providers.py::TestCountTokensFallback -v -p no:xdist`
Expected: All PASS.

- [ ] **Step 5: Run full providers test suite**

Run: `cd packages/markitai && python -m pytest tests/unit/test_providers.py -v -p no:xdist`
Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/providers/__init__.py packages/markitai/tests/unit/test_providers.py
git commit -m "fix: CJK-aware token count estimation for non-OpenAI models

The character-based fallback now counts CJK characters at ~1.5
tokens/char instead of the English ratio of ~0.25 tokens/char.
Fixes 4-6x underestimation of token counts for Chinese content."
```

---

## Dropped: Task 7 (Fetch strategy display)

**Reason:** `url.py:237` already contains `progress.log(f"Fetched via {used_strategy}: {url}")`. The original review's observation was incorrect — strategy is already displayed after fetch completes. No change needed.

---

## Summary

| Phase | Tasks | Risk | Est. time |
|-------|-------|------|-----------|
| Phase 1 (P0) | Task 1, Task 2 | Low | ~1 day |
| Phase 2 (P1) | Task 3, Task 4, Task 5 | Low-Medium | ~1-2 days |
| Phase 3 (P2) | Task 6 | Low | ~0.5 day |

All tasks are independent and can be implemented in any order. Task 5 (COM split) requires Windows for verification.
