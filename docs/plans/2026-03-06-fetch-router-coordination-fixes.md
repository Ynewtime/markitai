# Fetch / Router / Coordination 修复实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复 3 个真实批处理测试中发现的问题 — defuddle 在 screenshot 模式下缺失、gemini-cli 429 限速风暴、alt text 文件轮询超时竞态。

**Architecture:** P0 在 `_fetch_multi_source` 中添加 defuddle 为第 4 个并行抓取任务并调整优先级；P1 在 HybridRouter 和 LocalProviderWrapper 中添加 per-model cooldown 机制，在 `_select_model` 时跳过正在 cooldown 的模型；P2 在 `analyze_images_with_llm` 中用 `asyncio.Event` 替代文件轮询，由调用者注入 event 信号。

**Tech Stack:** Python asyncio, pytest, unittest.mock (AsyncMock)

---

## Task 1: P0 — `_fetch_multi_source` 添加 defuddle（测试）

**Files:**
- Test: `packages/markitai/tests/unit/test_fetch.py`

**Step 1: 写失败测试 — defuddle 优先于 static**

在 `TestFetchMultiSourceAdditional` 类中（约 line 2416）添加：

```python
@pytest.mark.asyncio
async def test_fetch_multi_source_defuddle_preferred_over_static(self):
    """Defuddle content should be preferred over static when both are valid."""
    from markitai.fetch import _fetch_multi_source

    mock_config = type(
        "MockConfig",
        (),
        {
            "jina": type("J", (), {"get_resolved_api_key": lambda self: None})(),
            "defuddle": type("D", (), {"timeout": 30, "rpm": 20})(),
            "strategy": "auto",
            "fallback_patterns": [],
            "playwright": type("PW", (), {"timeout": 30000, "extra_wait_ms": 5000})(),
        },
    )()

    with (
        patch(
            "markitai.fetch.fetch_with_static",
            new_callable=AsyncMock,
            return_value=type(
                "R", (), {"content": "# Static content\n\nSome valid static text here."}
            )(),
        ),
        patch(
            "markitai.fetch.fetch_with_defuddle",
            new_callable=AsyncMock,
            return_value=type(
                "R",
                (),
                {
                    "content": "# Defuddle content\n\nClean article text from defuddle.",
                    "strategy_used": "defuddle",
                    "title": "Test",
                    "url": "https://example.com",
                    "final_url": "https://example.com",
                    "metadata": {"api": "defuddle"},
                    "screenshot_path": None,
                },
            )(),
        ),
        patch(
            "markitai.fetch.is_playwright_available",
            return_value=False,
        ),
    ):
        result = await _fetch_multi_source("https://example.com", mock_config)
        assert "defuddle" in result.strategy_used.lower() or "Defuddle" in result.content
```

**Step 2: 写失败测试 — defuddle 失败时回退到 static**

```python
@pytest.mark.asyncio
async def test_fetch_multi_source_falls_back_when_defuddle_fails(self):
    """Should fall back to static when defuddle fetch fails."""
    from markitai.fetch import _fetch_multi_source

    mock_config = type(
        "MockConfig",
        (),
        {
            "jina": type("J", (), {"get_resolved_api_key": lambda self: None})(),
            "defuddle": type("D", (), {"timeout": 30, "rpm": 20})(),
            "strategy": "auto",
            "fallback_patterns": [],
            "playwright": type("PW", (), {"timeout": 30000, "extra_wait_ms": 5000})(),
        },
    )()

    with (
        patch(
            "markitai.fetch.fetch_with_static",
            new_callable=AsyncMock,
            return_value=type(
                "R", (), {"content": "# Valid static content\n\nEnough text here for validation."}
            )(),
        ),
        patch(
            "markitai.fetch.fetch_with_defuddle",
            new_callable=AsyncMock,
            side_effect=Exception("Defuddle API error"),
        ),
        patch(
            "markitai.fetch.is_playwright_available",
            return_value=False,
        ),
    ):
        result = await _fetch_multi_source("https://example.com", mock_config)
        assert result.strategy_used == "static"
```

**Step 3: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_fetch.py::TestFetchMultiSourceAdditional::test_fetch_multi_source_defuddle_preferred_over_static tests/unit/test_fetch.py::TestFetchMultiSourceAdditional::test_fetch_multi_source_falls_back_when_defuddle_fails -v`
Expected: FAIL — `_fetch_multi_source` 没有调用 `fetch_with_defuddle`

---

## Task 2: P0 — `_fetch_multi_source` 添加 defuddle（实现）

**Files:**
- Modify: `packages/markitai/src/markitai/fetch.py` (`_fetch_multi_source` 函数, lines 2405-2669)

**Step 1: 添加 `fetch_defuddle` 内部函数**

在 `fetch_jina()` 定义之后（约 line 2524），添加：

```python
    # Task 4: Defuddle parallel fetch (free, no auth required)
    async def fetch_defuddle_content() -> str | None:
        """Fetch with Defuddle content extraction API."""
        try:
            result = await fetch_with_defuddle(
                url,
                config.defuddle.timeout,
                config.defuddle.rpm,
            )
            logger.debug(f"[URL] Defuddle fetch success: {len(result.content)} chars")
            return result.content
        except Exception as e:
            logger.debug(f"[URL] Defuddle fetch failed: {e}")
            return None
```

**Step 2: 修改 `asyncio.gather` 为 4 路并行**

将 line 2527-2529 从：
```python
static_content, browser_fetch_result, jina_content = await asyncio.gather(
    fetch_static(), fetch_browser(), fetch_jina()
)
```
改为：
```python
static_content, browser_fetch_result, jina_content, defuddle_content = (
    await asyncio.gather(
        fetch_static(), fetch_browser(), fetch_jina(), fetch_defuddle_content()
    )
)
```

**Step 3: 添加 defuddle 内容校验**

在 jina 校验之后（约 line 2551），添加：

```python
defuddle_invalid, defuddle_reason = (
    _is_invalid_content(defuddle_content)
    if defuddle_content
    else (True, "fetch_failed")
)

if defuddle_content and defuddle_invalid:
    logger.debug(f"[URL] Defuddle content invalid: {defuddle_reason}")
```

**Step 4: 修改优先级逻辑**

将 lines 2567-2583 的 `if/elif` 链从 `static > browser > jina` 改为 `defuddle > static > browser > jina`：

```python
if not defuddle_invalid:
    # Defuddle is valid → use defuddle (best content cleaning)
    assert defuddle_content is not None
    primary_content = defuddle_content
    strategy_used = "defuddle"
elif not static_invalid:
    # Static is valid → use static
    assert static_content is not None
    primary_content = static_content
    final_static_content = static_content
    strategy_used = "static"
elif not browser_invalid:
    # Browser is valid → use browser
    assert browser_content is not None
    primary_content = browser_content
    final_browser_content = browser_content
    strategy_used = "browser"
elif not jina_invalid:
    # Jina is valid → use Jina
    assert jina_content is not None
    primary_content = jina_content
    strategy_used = "jina"
```

保留后续的错误处理分支（lines 2584-2638）不变。

**Step 5: 更新函数 docstring**

将 docstring（lines 2414-2422）的 strategy 列表更新为包含 defuddle，优先级说明改为 `defuddle > static > browser > jina`。

**Step 6: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_fetch.py::TestFetchMultiSourceAdditional -v`
Expected: ALL PASS

**Step 7: 运行完整 fetch 测试确认无回归**

Run: `uv run pytest tests/unit/test_fetch.py -x -q`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add packages/markitai/src/markitai/fetch.py packages/markitai/tests/unit/test_fetch.py
git commit -m "feat(fetch): add defuddle to _fetch_multi_source parallel strategy

Defuddle was missing from the screenshot-mode (_fetch_multi_source)
fetch path. When --preset rich or --screenshot was used, fetch_url()
bypassed the policy engine and only tried static/browser/jina.

Now defuddle runs as a 4th parallel task with highest content priority
(defuddle > static > browser > jina), matching fetch_policy.py ordering."
```

---

## Task 3: P1 — HybridRouter cooldown 机制（测试）

**Files:**
- Test: `packages/markitai/tests/unit/test_llm_processor.py`

**Step 1: 写失败测试 — cooldown 中的模型被跳过**

在 `TestHybridRouter` 类中（约 line 494）添加：

```python
def test_select_model_skips_cooldown_model(self):
    """Models in cooldown should be skipped during selection."""
    import time

    standard_router = MagicMock(spec=Router)
    standard_router.model_list = []
    local_wrapper = MagicMock(spec=LocalProviderWrapper)
    local_wrapper.model_list = [
        {
            "model_name": "default",
            "litellm_params": {"model": "claude-agent/sonnet", "weight": 10},
        },
        {
            "model_name": "default",
            "litellm_params": {"model": "gemini-cli/gemini-3-flash", "weight": 10},
        },
    ]
    hybrid = HybridRouter(standard_router, local_wrapper)

    # Put gemini-cli in cooldown for 60 seconds
    hybrid.record_cooldown("gemini-cli/gemini-3-flash", 60.0)

    # All selections should avoid the cooldown model
    selections = {hybrid._select_model("default") for _ in range(50)}
    assert selections == {"claude-agent/sonnet"}
```

**Step 2: 写失败测试 — cooldown 过期后恢复路由**

```python
def test_select_model_routes_after_cooldown_expires(self):
    """Models should be routable again after cooldown expires."""
    standard_router = MagicMock(spec=Router)
    standard_router.model_list = []
    local_wrapper = MagicMock(spec=LocalProviderWrapper)
    local_wrapper.model_list = [
        {
            "model_name": "default",
            "litellm_params": {"model": "claude-agent/sonnet", "weight": 10},
        },
        {
            "model_name": "default",
            "litellm_params": {"model": "gemini-cli/gemini-3-flash", "weight": 10},
        },
    ]
    hybrid = HybridRouter(standard_router, local_wrapper)

    # Set cooldown that already expired
    hybrid._model_cooldowns["gemini-cli/gemini-3-flash"] = time.monotonic() - 1.0

    # Both models should now be selectable
    selections = {hybrid._select_model("default") for _ in range(100)}
    assert len(selections) == 2
```

**Step 3: 写失败测试 — 所有模型 cooldown 时选最快解除的**

```python
def test_select_model_picks_soonest_expiring_when_all_in_cooldown(self):
    """When all models are in cooldown, pick the one expiring soonest."""
    standard_router = MagicMock(spec=Router)
    standard_router.model_list = []
    local_wrapper = MagicMock(spec=LocalProviderWrapper)
    local_wrapper.model_list = [
        {
            "model_name": "default",
            "litellm_params": {"model": "model-a", "weight": 10},
        },
        {
            "model_name": "default",
            "litellm_params": {"model": "model-b", "weight": 10},
        },
    ]
    hybrid = HybridRouter(standard_router, local_wrapper)

    now = time.monotonic()
    hybrid._model_cooldowns["model-a"] = now + 120.0  # expires later
    hybrid._model_cooldowns["model-b"] = now + 10.0   # expires sooner

    selected = hybrid._select_model("default")
    assert selected == "model-b"
```

**Step 4: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_llm_processor.py::TestHybridRouter::test_select_model_skips_cooldown_model tests/unit/test_llm_processor.py::TestHybridRouter::test_select_model_routes_after_cooldown_expires tests/unit/test_llm_processor.py::TestHybridRouter::test_select_model_picks_soonest_expiring_when_all_in_cooldown -v`
Expected: FAIL — `record_cooldown` 和 `_model_cooldowns` 不存在

---

## Task 4: P1 — HybridRouter cooldown 机制（实现）

**Files:**
- Modify: `packages/markitai/src/markitai/llm/processor.py`

**Step 1: 在 HybridRouter.__init__ 中添加 cooldown 状态**

在 `self._image_capable_cache` 初始化之后（约 line 335），添加：

```python
# Per-model cooldown tracking (model_id → monotonic expiry time)
self._model_cooldowns: dict[str, float] = {}
```

**Step 2: 添加 `record_cooldown` 方法**

在 `_is_image_capable` 方法附近添加：

```python
def record_cooldown(self, model_id: str, seconds: float) -> None:
    """Record that a model should be avoided for the given duration.

    Args:
        model_id: The model that hit rate limit (e.g., "gemini-cli/gemini-3-flash")
        seconds: How long to avoid routing to this model
    """
    self._model_cooldowns[model_id] = time.monotonic() + seconds
    logger.info(
        f"[HybridRouter] Model {model_id} in cooldown for {seconds:.0f}s"
    )
```

**Step 3: 修改 `_select_model` 添加 cooldown 过滤**

在 `_select_model` 方法中，在 `active = [(m, w, loc) for m, w, loc in models if w > 0]` 之后（约 line 401），添加 cooldown 过滤：

```python
# Filter out weight<=0 models (disabled by user)
active = [(m, w, loc) for m, w, loc in models if w > 0]
if not active:
    return random.choice(models)[0]

# Filter out models in cooldown
now = time.monotonic()
available = [
    (m, w, loc)
    for m, w, loc in active
    if self._model_cooldowns.get(m, 0) <= now
]

if not available:
    # All active models in cooldown — pick soonest to expire
    soonest = min(active, key=lambda x: self._model_cooldowns.get(x[0], 0))
    logger.debug(
        f"[HybridRouter] All models in cooldown, using soonest-expiring: {soonest[0]}"
    )
    return soonest[0]

active = available
```

**Step 4: 在 `acompletion` 中捕获 429 并记录 cooldown**

修改 `acompletion` 方法（line 416-442），在调用 local_wrapper 或 standard_router 时添加 try/except：

```python
async def acompletion(
    self,
    model: str,
    messages: list[Any],
    **kwargs: Any,
) -> Any:
    has_images = self._has_images(messages)
    selected_model = self._select_model(model, has_images)

    try:
        if self._is_local_model(selected_model):
            logger.debug(f"[HybridRouter] Routing to local provider: {selected_model}")
            return await self.local_wrapper.acompletion(
                selected_model, messages, **kwargs
            )
        else:
            logger.debug(f"[HybridRouter] Routing to standard router: {selected_model}")
            return await self.standard_router.acompletion(model, messages, **kwargs)
    except Exception as e:
        # Check for rate limit / quota errors and record cooldown
        error_msg = str(e).lower()
        is_rate_limit = any(
            p in error_msg for p in ("429", "rate limit", "quota", "too many requests")
        )
        if is_rate_limit:
            # Parse retry-after if available, default to 60s
            cooldown_seconds = 60.0
            import re as _re

            match = _re.search(r"(\d+)\s*s", error_msg)
            if match:
                cooldown_seconds = float(match.group(1))
            self.record_cooldown(selected_model, cooldown_seconds)
        raise
```

**Step 5: 同步添加到 LocalProviderWrapper**

在 `LocalProviderWrapper` 类中添加相同的 `_model_cooldowns`、`record_cooldown`、cooldown 过滤逻辑（结构完全相同）。

**Step 6: 确保 `import time` 在文件顶部**

检查 `processor.py` 是否已 `import time`。

**Step 7: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_llm_processor.py::TestHybridRouter -v`
Expected: ALL PASS

**Step 8: 运行完整 processor 测试确认无回归**

Run: `uv run pytest tests/unit/test_llm_processor.py -x -q`
Expected: ALL PASS

**Step 9: Commit**

```bash
git add packages/markitai/src/markitai/llm/processor.py packages/markitai/tests/unit/test_llm_processor.py
git commit -m "feat(router): add per-model cooldown to avoid 429 rate limit storms

When a model returns 429/quota errors, HybridRouter now records a
cooldown period and skips that model in subsequent _select_model calls.
If all models are in cooldown, the soonest-expiring one is selected.

This prevents the cascading retry storm seen with gemini-cli (18 retries,
>2 minutes wasted) when it has low RPM limits."
```

---

## Task 5: P2 — asyncio.Event 替代文件轮询（测试）

**Files:**
- Test: `packages/markitai/tests/unit/test_llm_processor_cli.py`

**Step 1: 写失败测试 — event 信号正常传递时跳过轮询**

在现有 `test_analyze_images_waits_for_llm_file` 测试附近添加：

```python
@pytest.mark.asyncio
async def test_analyze_images_uses_event_instead_of_polling(
    self, tmp_path, mock_llm_response
):
    """When llm_ready_event is provided, should await event instead of polling."""
    import asyncio

    from markitai.cli.processors.llm import analyze_images_with_llm

    # Create test image and output file
    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    output = tmp_path / "output.md"
    output.write_text("# Test\n\n![](assets/test.jpg)\n")
    llm_output = tmp_path / "output.llm.md"

    cfg = self._make_config(alt=True, desc=False)
    event = asyncio.Event()

    async def write_llm_file_after_delay():
        await asyncio.sleep(0.1)
        llm_output.write_text("# Test LLM\n\n![](assets/test.jpg)\n")
        event.set()

    writer_task = asyncio.create_task(write_llm_file_after_delay())

    result = await analyze_images_with_llm(
        [img], "# Test", output, cfg, processor=mock_llm_response,
        llm_ready_event=event,
    )
    await writer_task

    # File should have been found via event, not polling timeout
    assert llm_output.exists()
```

**Step 2: 写失败测试 — event 超时后兜底检查文件**

```python
@pytest.mark.asyncio
async def test_analyze_images_event_timeout_falls_back_to_file_check(
    self, tmp_path, mock_llm_response
):
    """When event times out, should fall back to checking if file exists."""
    import asyncio

    from markitai.cli.processors.llm import analyze_images_with_llm

    img = tmp_path / "test.jpg"
    img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
    output = tmp_path / "output.md"
    output.write_text("# Test\n\n![](assets/test.jpg)\n")
    llm_output = tmp_path / "output.llm.md"
    # Pre-create the file so fallback check finds it
    llm_output.write_text("# Test LLM\n\n![](assets/test.jpg)\n")

    cfg = self._make_config(alt=True, desc=False)
    event = asyncio.Event()  # Never set — will timeout

    result = await analyze_images_with_llm(
        [img], "# Test", output, cfg, processor=mock_llm_response,
        llm_ready_event=event, llm_ready_timeout=0.1,
    )

    # Should have found file via fallback check
    assert llm_output.exists()
```

**Step 3: 运行测试确认失败**

Run: `uv run pytest tests/unit/test_llm_processor_cli.py::TestAnalyzeImagesWithLLM::test_analyze_images_uses_event_instead_of_polling tests/unit/test_llm_processor_cli.py::TestAnalyzeImagesWithLLM::test_analyze_images_event_timeout_falls_back_to_file_check -v`
Expected: FAIL — `llm_ready_event` 参数不存在

---

## Task 6: P2 — asyncio.Event 替代文件轮询（实现）

**Files:**
- Modify: `packages/markitai/src/markitai/cli/processors/llm.py` (lines 154-292)
- Modify: `packages/markitai/src/markitai/cli/processors/batch.py` (lines 369-439)

**Step 1: 修改 `analyze_images_with_llm` 签名添加 event 参数**

在函数签名（line 154-162）中添加：

```python
async def analyze_images_with_llm(
    image_paths: list[Path],
    markdown: str,
    output_file: Path,
    cfg: MarkitaiConfig,
    input_path: Path | None = None,
    concurrency_limit: int | None = None,
    processor: LLMProcessor | None = None,
    llm_ready_event: asyncio.Event | None = None,
    llm_ready_timeout: float = 300.0,
) -> tuple[str, float, dict[str, dict[str, Any]], ImageAnalysisResult | None]:
```

**Step 2: 替换轮询逻辑**

将 lines 270-292 的轮询代码替换为：

```python
elif alt_enabled:
    llm_output = output_file.with_suffix(".llm.md")

    # Wait for .llm.md to be ready
    if llm_ready_event is not None:
        # Use event-based signaling (preferred)
        try:
            await asyncio.wait_for(
                llm_ready_event.wait(), timeout=llm_ready_timeout
            )
        except asyncio.TimeoutError:
            logger.debug(
                f"[ImageAnalysis] Event timeout after {llm_ready_timeout}s, "
                f"checking file existence as fallback"
            )
    else:
        # Legacy polling fallback (for callers that don't provide event)
        max_wait_seconds = 300
        poll_interval = 0.5
        waited = 0.0
        while not llm_output.exists() and waited < max_wait_seconds:
            await asyncio.sleep(poll_interval)
            waited += poll_interval

    if llm_output.exists():
        llm_content = llm_output.read_text(encoding="utf-8")
        for image_path, analysis, _ in results:
            if analysis is None:
                continue
            old_pattern = rf"!\[[^\]]*\]\([^)]*{re.escape(image_path.name)}\)"
            new_ref = f"![{analysis.caption}](assets/{image_path.name})"
            llm_content = re.sub(old_pattern, new_ref, llm_content)
        atomic_write_text(llm_output, llm_content)
    else:
        logger.warning(
            f"Skipped alt text update: {llm_output} not created within timeout"
        )
```

**Step 3: 确保 `import asyncio` 在文件顶部**

检查 `llm.py` 是否已 `import asyncio`。

**Step 4: 在 batch.py 中传递 event**

在 `batch.py` 的 URL 处理路径中（约 lines 369-400），创建 event 并传递：

Path 1（vision + images, 约 line 369）：
```python
llm_ready_event = asyncio.Event()

async def vision_with_signal():
    result = await process_url_with_vision(...)
    llm_ready_event.set()
    return result

vision_task = vision_with_signal()
img_task = analyze_images_with_llm(
    ...,
    llm_ready_event=llm_ready_event,
)
vision_result, img_result = await asyncio.gather(vision_task, img_task)
```

Path 2（doc + images, 约 line 412）：
```python
llm_ready_event = asyncio.Event()

async def doc_with_signal():
    result = await process_with_llm(...)
    llm_ready_event.set()
    return result

doc_task = doc_with_signal()
img_task = analyze_images_with_llm(
    ...,
    llm_ready_event=llm_ready_event,
)
doc_result, img_result = await asyncio.gather(doc_task, img_task)
```

**Step 5: 在 url.py 的调用点也传递 event**

检查 `url.py` 中的 `analyze_images_with_llm` 调用，同样添加 event 传递。

**Step 6: 运行测试确认通过**

Run: `uv run pytest tests/unit/test_llm_processor_cli.py::TestAnalyzeImagesWithLLM -v`
Expected: ALL PASS

**Step 7: 运行完整测试确认无回归**

Run: `uv run pytest tests/unit/ -x -q`
Expected: ALL PASS

**Step 8: Commit**

```bash
git add packages/markitai/src/markitai/cli/processors/llm.py packages/markitai/src/markitai/cli/processors/batch.py packages/markitai/tests/unit/test_llm_processor_cli.py
git commit -m "feat(llm): replace file polling with asyncio.Event for alt text coordination

The alt text update in analyze_images_with_llm used a 120s file-polling
loop to wait for .llm.md creation. When vision enhancement was delayed
by provider rate limits (190s for gemini-cli 429), the timeout was
exceeded and alt text updates were silently skipped.

Now uses asyncio.Event injected by the caller. Vision/doc tasks set the
event after writing .llm.md. Timeout raised to 300s with file-existence
fallback. Legacy polling preserved for callers that don't provide event."
```

---

## Task 7: 最终验证

**Step 1: Lint + Format**

Run: `uv run ruff check src/ tests/ && uv run ruff format --check src/ tests/`
Expected: All checks passed

**Step 2: 完整单元测试**

Run: `uv run pytest tests/unit/ -x -q`
Expected: ALL PASS, 0 failures

**Step 3: Pyright 类型检查**

Run: `uv run pyright`
Expected: 0 errors

**Step 4: 修复任何 lint/type 问题后最终 commit**

```bash
git add -A
git commit -m "chore: fix lint and type issues from P0/P1/P2 fixes"
```
