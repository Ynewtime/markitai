# 批处理性能优化 Implementation Plan

**Goal:** 修复代码中 8 处性能瓶颈，主要是串行 await 改并行、减少不必要的等待和重复计算

**Architecture:** 每个优化项独立且互不影响，可逐项实现和验证。核心改动集中在 4 个文件：batch.py (URL vision 并行化)、single.py (截图提取并行化)、vision.py (回退并行化)、processor.py (vision router 预初始化 + 模型过滤缓存)、fetch.py (截图条件压缩)、constants.py (常量调整)

**Tech Stack:** Python 3.13, asyncio, pytest, pytest-asyncio

---

### Task 1: Vision 路径下 URL 图片分析并行化

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/batch.py:351-396`
- Test: `packages/markitai/tests/unit/test_batch_processor.py`

**Step 1: Write the failing test**

在 `test_batch_processor.py` 中添加测试，验证 vision 路径下 `process_url_with_vision` 和 `analyze_images_with_llm` 被并行调用（通过 `asyncio.gather`），而不是串行 await。

```python
class TestVisionPathParallelization:
    """Test that vision enhancement and image analysis run in parallel."""

    @pytest.mark.asyncio
    async def test_vision_path_runs_in_parallel(self, monkeypatch):
        """Vision path should use asyncio.gather for vision + image analysis."""
        import asyncio
        from unittest.mock import AsyncMock, patch, MagicMock

        call_order = []

        async def mock_vision(*args, **kwargs):
            call_order.append(("vision_start", asyncio.get_event_loop().time()))
            await asyncio.sleep(0.05)
            call_order.append(("vision_end", asyncio.get_event_loop().time()))
            return ("content", 0.0, {})

        async def mock_analyze(*args, **kwargs):
            call_order.append(("analyze_start", asyncio.get_event_loop().time()))
            await asyncio.sleep(0.05)
            call_order.append(("analyze_end", asyncio.get_event_loop().time()))
            return ("content", 0.0, {}, [])

        # Patch at module level
        with patch(
            "markitai.cli.processors.batch.process_url_with_vision", mock_vision
        ), patch(
            "markitai.cli.processors.batch.analyze_images_with_llm", mock_analyze
        ):
            # Import after patching
            from markitai.cli.processors.batch import _process_single_url_inner

            # If running in parallel, both starts should happen before either end
            # This verifies asyncio.gather is being used
            starts = [t for name, t in call_order if name.endswith("_start")]
            ends = [t for name, t in call_order if name.endswith("_end")]

            if len(starts) >= 2 and len(ends) >= 2:
                # Both tasks started before the first one ended = parallel
                assert starts[1] < ends[0], (
                    "Tasks should run in parallel: second task should start before first task ends"
                )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_batch_processor.py::TestVisionPathParallelization -v`
Expected: FAIL (current code runs sequentially)

**Step 3: Implement the fix**

修改 `batch.py:351-396`，将 vision 路径改为使用 `asyncio.gather`：

```python
                if use_vision_enhancement:
                    multi_source_content = build_multi_source_content(
                        fetch_result.static_content,
                        fetch_result.browser_content,
                        markdown_for_llm,
                    )

                    logger.debug(
                        f"[URL] Using vision enhancement for multi-source URL: {url}"
                    )

                    assert screenshot_path is not None

                    if should_analyze_images:
                        # Run vision enhancement and image analysis in parallel
                        vision_task = process_url_with_vision(
                            multi_source_content,
                            screenshot_path,
                            url,
                            cfg,
                            output_file,
                            processor=shared_processor,
                            original_title=fetch_result.title if fetch_result else None,
                        )
                        img_task = analyze_images_with_llm(
                            downloaded_images,
                            multi_source_content,
                            output_file,
                            cfg,
                            Path(url),
                            concurrency_limit=cfg.llm.concurrency,
                            processor=shared_processor,
                        )

                        vision_result, img_result = await asyncio.gather(
                            vision_task, img_task
                        )

                        _, cost, url_llm_usage = vision_result
                        _, image_cost, image_usage, img_analysis = img_result

                        _merge_llm_usage(url_llm_usage, image_usage)
                        llm_cost = cost + image_cost
                    else:
                        _, cost, url_llm_usage = await process_url_with_vision(
                            multi_source_content,
                            screenshot_path,
                            url,
                            cfg,
                            output_file,
                            processor=shared_processor,
                            original_title=fetch_result.title if fetch_result else None,
                        )
                        llm_cost = cost
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_batch_processor.py -v -k "vision"`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/processors/batch.py packages/markitai/tests/unit/test_batch_processor.py
git commit -m "perf: parallelize vision enhancement and image analysis in URL processing"
```

---

### Task 2: `extract_from_screenshots` 逐页串行改并行

**Files:**
- Modify: `packages/markitai/src/markitai/workflow/single.py:380-392`
- Test: `packages/markitai/tests/unit/test_workflow_single.py`

**Step 1: Write the failing test**

```python
class TestExtractFromScreenshotsParallel:
    """Test that screenshot extraction processes pages in parallel."""

    @pytest.mark.asyncio
    async def test_pages_processed_concurrently(self):
        """Multiple pages should be processed via asyncio.gather, not sequentially."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, patch
        from pathlib import Path

        call_times = []

        async def mock_extract(image_path, context=""):
            call_times.append(("start", asyncio.get_event_loop().time()))
            await asyncio.sleep(0.05)
            call_times.append(("end", asyncio.get_event_loop().time()))
            return (f"Content for {context}", "")

        mock_processor = MagicMock()
        mock_processor.extract_from_screenshot = AsyncMock(side_effect=mock_extract)
        mock_processor.get_context_cost = MagicMock(return_value=0.0)
        mock_processor.get_context_usage = MagicMock(return_value={})

        from markitai.workflow.single import LLMEnhancer

        enhancer = LLMEnhancer.__new__(LLMEnhancer)
        enhancer.processor = mock_processor

        page_images = [
            {"path": f"/tmp/page{i}.png", "page": i} for i in range(1, 4)
        ]

        result = await enhancer.extract_from_screenshots(page_images, source="test")

        # With 3 pages at 0.05s each:
        # Sequential: ~0.15s, Parallel: ~0.05s
        starts = [t for name, t in call_times if name == "start"]
        ends = [t for name, t in call_times if name == "end"]

        assert len(starts) == 3, "All 3 pages should be processed"
        # All starts should happen within a short window (parallel)
        assert starts[-1] - starts[0] < 0.03, (
            "All pages should start nearly simultaneously (parallel)"
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py::TestExtractFromScreenshotsParallel -v`
Expected: FAIL (current code is sequential)

**Step 3: Implement the fix**

修改 `single.py:380-392`：

```python
            # Process all pages in parallel using asyncio.gather
            async def _extract_page(i: int, image_path: Path) -> str:
                page_source = f"{source}:page{i}"
                cleaned, _ = await self.processor.extract_from_screenshot(
                    image_path, context=page_source
                )
                if cleaned.strip():
                    return f"<!-- Page {i} -->\n\n{cleaned}"
                return ""

            page_results = await asyncio.gather(
                *[
                    _extract_page(i, image_path)
                    for i, image_path in enumerate(image_paths, 1)
                ]
            )
            all_content = [r for r in page_results if r]

            # Merge all page content
            merged_content = "\n\n".join(all_content)
```

需要在文件顶部确认有 `import asyncio`（已有）。

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_workflow_single.py -v -k "parallel"`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/workflow/single.py packages/markitai/tests/unit/test_workflow_single.py
git commit -m "perf: parallelize screenshot extraction across pages"
```

---

### Task 3: Vision 批分析回退时并行化

**Files:**
- Modify: `packages/markitai/src/markitai/llm/vision.py:514-538`
- Test: `packages/markitai/tests/unit/test_llm.py`

**Step 1: Write the failing test**

```python
class TestVisionFallbackParallel:
    """Test that fallback image analysis runs in parallel."""

    @pytest.mark.asyncio
    async def test_fallback_analyzes_images_concurrently(self):
        """When batch analysis fails, individual images should be analyzed in parallel."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock, PropertyMock
        from pathlib import Path

        call_times = []

        async def mock_analyze_image(image_path, language="en", context=""):
            call_times.append(("start", asyncio.get_event_loop().time()))
            await asyncio.sleep(0.05)
            call_times.append(("end", asyncio.get_event_loop().time()))
            return ImageAnalysis(
                caption="test", description="test", extracted_text=None
            )

        # Setup: create a VisionMixin instance that will trigger fallback
        # and verify the fallback uses parallel processing
        # The key assertion: all starts happen before any end
        starts = [t for name, t in call_times if name == "start"]
        if len(starts) >= 2:
            assert starts[-1] - starts[0] < 0.03, (
                "Fallback should process images in parallel"
            )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_llm.py::TestVisionFallbackParallel -v`

**Step 3: Implement the fix**

修改 `vision.py:514-538`，将串行 for 循环改为 `asyncio.gather`：

```python
            # Fallback: analyze each image concurrently (uses persistent cache)
            async def _analyze_one(i: int, image_path: Path) -> ImageAnalysis:
                if i in unsupported_results:
                    return unsupported_results[i]
                if i in cached_results:
                    return cached_results[i]
                try:
                    return await self.analyze_image(image_path, language, context)
                except Exception:
                    return ImageAnalysis(
                        caption="Image",
                        description="Image analysis failed",
                        extracted_text=None,
                    )

            fallback_results = list(
                await asyncio.gather(
                    *[
                        _analyze_one(i, image_path)
                        for i, image_path in enumerate(image_paths)
                    ]
                )
            )
            return fallback_results
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_llm.py -v -k "fallback"`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/llm/vision.py packages/markitai/tests/unit/test_llm.py
git commit -m "perf: parallelize fallback image analysis in vision module"
```

---

### Task 4: Vision Router 预初始化 + 模型过滤缓存

**Files:**
- Modify: `packages/markitai/src/markitai/llm/processor.py:466-467` (init) 和 `326-373` (select_model)
- Test: `packages/markitai/tests/unit/test_llm_processor.py`

**Step 1: Write the failing test**

```python
class TestVisionRouterEagerInit:
    """Test that vision router is created eagerly during init."""

    def test_vision_router_created_on_init(self, llm_config):
        """Vision router should be pre-created, not lazily initialized."""
        from markitai.llm.processor import LLMProcessor

        processor = LLMProcessor(config=llm_config)

        # Access internal state - vision router should already exist
        assert processor._vision_router is not None, (
            "Vision router should be eagerly created in __init__"
        )
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py::TestVisionRouterEagerInit -v`
Expected: FAIL (`_vision_router` is `None` initially)

**Step 3: Implement the fix**

在 `processor.py` 的 `__init__` 末尾（`_setup_callbacks()` 之后），添加：

```python
        # Eagerly initialize vision router to avoid first-call latency
        _ = self.vision_router
```

对于 HybridRouter 的 `_select_model` 中模型过滤缓存，在 `HybridRouter.__init__` 中预计算：

```python
    def __init__(self, ...):
        ...
        # Pre-compute image-capable models per group
        self._image_capable_cache: dict[str, list[tuple[str, float, bool]]] = {}
        for group_name, models in self._model_groups.items():
            self._image_capable_cache[group_name] = [
                (m, w, is_local)
                for m, w, is_local in models
                if self._is_image_capable(m)
            ]
```

然后在 `_select_model` 中使用缓存：

```python
        if has_images and len(models) > 1:
            image_capable = self._image_capable_cache.get(model_name, [])
            if image_capable:
                ...
                models = image_capable
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py -v -k "vision_router"`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/llm/processor.py packages/markitai/tests/unit/test_llm_processor.py
git commit -m "perf: eagerly init vision router and cache image-capable model list"
```

---

### Task 5: 截图条件压缩

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py:1328-1366`
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: Write the failing test**

```python
class TestConditionalScreenshotCompression:
    """Test that screenshot compression is skipped when unnecessary."""

    def test_skip_compression_when_height_within_limit(self, tmp_path):
        """PIL compression should be skipped if image height < max_height and format is already JPEG."""
        from unittest.mock import patch, MagicMock
        from pathlib import Path
        from PIL import Image

        # Create a small JPEG screenshot (height < 10000)
        img = Image.new("RGB", (800, 600), "white")
        screenshot_path = tmp_path / "test.jpg"
        img.save(screenshot_path, "JPEG", quality=85)
        original_size = screenshot_path.stat().st_size

        from markitai.fetch import _compress_screenshot

        with patch("markitai.fetch.Image") as mock_pil:
            _compress_screenshot(screenshot_path, quality=65, max_height=10000)

            # PIL.Image.open should NOT be called since height < max_height
            # and format is already JPEG
            # (This test will fail initially because current code always opens with PIL)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch.py::TestConditionalScreenshotCompression -v`

**Step 3: Implement the fix**

修改 `fetch.py` 的 `_compress_screenshot`，先检查文件是否需要处理：

```python
def _compress_screenshot(
    screenshot_path: Path,
    quality: int = 85,
    max_height: int = 10000,
) -> None:
    try:
        from PIL import Image

        # Quick check: get image dimensions without full decode
        with Image.open(screenshot_path) as img:
            width, height = img.size
            needs_resize = height > max_height
            needs_convert = img.mode in ("RGBA", "P")

        # Skip re-compression if image doesn't need resize or conversion
        # Playwright already saves JPEG with specified quality
        if not needs_resize and not needs_convert:
            logger.debug(
                f"Screenshot within limits ({width}x{height}), skipping re-compression"
            )
            return

        # Only re-process if needed
        with Image.open(screenshot_path) as img:
            if needs_convert:
                img = img.convert("RGB")

            if needs_resize:
                ratio = max_height / height
                new_width = int(width * ratio)
                img = img.resize((new_width, max_height), Image.Resampling.LANCZOS)
                logger.debug(
                    f"Resized screenshot from {width}x{height} to {new_width}x{max_height}"
                )

            img.save(screenshot_path, "JPEG", quality=quality, optimize=True)
            logger.debug(
                f"Compressed screenshot to quality={quality}: {screenshot_path}"
            )
    except ImportError:
        logger.warning("Pillow not installed, skipping screenshot compression")
    except Exception as e:
        logger.warning(f"Failed to compress screenshot: {e}")
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_fetch.py -v -k "compression"`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch.py
git commit -m "perf: skip screenshot re-compression when resize/conversion not needed"
```

---

### Task 6: 增大 `max_pages_per_batch` 减少 API 调用

**Files:**
- Modify: `packages/markitai/src/markitai/constants.py:52`
- Test: 无需新测试（纯常量变更，已有测试覆盖批处理逻辑）

**Step 1: 修改常量**

```python
DEFAULT_MAX_PAGES_PER_BATCH = 10  # Pages per LLM call for document processing
```

注意：从 5 改到 10。注释说"reduced from 10 to avoid max_tokens"，但当前 `DynamicTokens` 已能自动调整 max_tokens 到模型限制，且大多数 vision 模型支持 64K+ output tokens，所以可以安全恢复到 10。

**Step 2: 运行现有测试确认不会破坏**

Run: `uv run pytest packages/markitai/tests/unit/ -v -k "batch" --no-header -q`
Expected: All PASS

**Step 3: Commit**

```bash
git add packages/markitai/src/markitai/constants.py
git commit -m "perf: increase max_pages_per_batch from 5 to 10 to reduce API calls"
```

---

### Task 7: 降低 Instructor 默认重试次数

**Files:**
- Modify: `packages/markitai/src/markitai/constants.py:35`
- Test: 无需新测试（纯常量变更）

**Step 1: 修改常量**

```python
DEFAULT_INSTRUCTOR_MAX_RETRIES = 2
```

从 3 降到 2。注释已说"Increased to 2"，但当前值实际是 3（即初始请求 + 3 次重试 = 4 次调用）。改为 2（初始 + 2 次重试 = 3 次调用），在保持容错的同时减少 25% 最坏情况延迟。

**Step 2: 运行现有测试确认不会破坏**

Run: `uv run pytest packages/markitai/tests/unit/ -v --no-header -q`
Expected: All PASS

**Step 3: Commit**

```bash
git add packages/markitai/src/markitai/constants.py
git commit -m "perf: reduce instructor max retries from 3 to 2"
```

---

### Task 8: 降低 Playwright extra_wait_ms 默认值

**Files:**
- Modify: `packages/markitai/src/markitai/constants.py` (找到 `DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS`)
- Test: 无需新测试（纯常量变更，且有 domain profile 可按需覆盖）

**Step 1: 查找并修改常量**

找到 `DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS = 5000` 改为 `3000`：

```python
DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS = 3000  # Extra wait after page load for JS rendering
```

从 5s 降到 3s。对需要更长等待的域名，用户可通过 domain_profiles 配置覆盖。

**Step 2: 运行现有测试确认不会破坏**

Run: `uv run pytest packages/markitai/tests/unit/ -v -k "playwright or fetch" --no-header -q`
Expected: All PASS

**Step 3: Commit**

```bash
git add packages/markitai/src/markitai/constants.py
git commit -m "perf: reduce default Playwright extra_wait_ms from 5000 to 3000"
```

---

## 验证

全部修改完成后，运行完整测试套件：

```bash
uv run pytest packages/markitai/tests/unit/ -v --tb=short
```

所有测试应通过。如有失败，根据报错调整。
