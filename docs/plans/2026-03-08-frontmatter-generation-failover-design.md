# Frontmatter Generation Failover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix cascading LLM failure when a model returns 400 (region restriction) so that description/tags frontmatter are always populated.

**Architecture:** Three targeted fixes in the LLM pipeline: (1) HybridRouter applies cooldown on model-level 400 errors so retries pick a different model, (2) `process_document` logs the exception instead of silently swallowing it, (3) `_build_fallback_frontmatter` extracts description/tags from `extra_meta` when LLM fails.

**Tech Stack:** Python, LiteLLM Router, Instructor, Pydantic, pytest (asyncio)

---

## Problem

When processing URL `https://ynewtime.com/jekyll-ynewtime/人是什么单位`, the `.llm.md` output has `description: ''` and no `tags`. Root cause analysis revealed three cascading failures:

1. **HybridRouter** routed the request to `gemini/gemini-3.1-flash-lite-preview`, which returned 400 "User location is not supported for the API use."
2. Both `_process_document_combined()` (Instructor) and fallback `clean_markdown()` were routed to the same failing model — no failover.
3. The exception from step 1 was silently swallowed (`except Exception: pass`).
4. `_build_fallback_frontmatter()` hardcodes `description=""`, `tags=[]`, ignoring defuddle's metadata that already contained a valid description.

## Fix 1: HybridRouter Model-Level Error Cooldown

**File**: `packages/markitai/src/markitai/llm/processor.py`

### Design

Expand `HybridRouter.acompletion()` error handling to detect "model-level" 400 errors (region restrictions, model unavailable, etc.) and apply a long cooldown (3600s). This ensures subsequent retries (from Instructor's `max_retries` or `_call_llm_with_retry`) automatically select a different model.

Additionally, in `_call_llm_with_retry()`, treat model-level errors as retryable (move from generic `except Exception` raise to retry loop with backoff).

### Error Patterns

Extract a class-level constant `MODEL_LEVEL_ERROR_PATTERNS` on `HybridRouter`:

```python
MODEL_LEVEL_ERROR_PATTERNS = (
    "user location is not supported",
    "failed_precondition",
    "model is not available",
    "model not found",
    "model_not_available",
    "region is not supported",
    "not available in your region",
)
```

### Changes

**A. `HybridRouter.acompletion()`** — after existing rate limit cooldown:

- Detect model-level error patterns in exception message
- Apply `record_cooldown(selected_model, 3600.0)` (long cooldown — these errors don't self-heal)
- Log warning with model name and error
- Re-raise (caller's retry mechanism picks a different model)

**B. `_call_llm_with_retry()`** — in generic `except Exception` branch:

- Check for model-level error patterns
- If matched and retries remain: log warning, backoff delay, `continue` (retry loop)
- Otherwise: fall through to existing auth detection + raise logic

### Coverage

- **Instructor path** (`_process_document_combined`): Instructor calls `router.acompletion()` → HybridRouter applies cooldown → Instructor's `max_retries` retries → router selects different model
- **_call_llm path** (`clean_markdown`, etc.): `_call_llm_with_retry()` detects pattern → retries → router selects different model via cooldown

## Fix 2: Log Structured Processing Failure

**File**: `packages/markitai/src/markitai/llm/document.py`

### Design

Replace silent `except Exception: pass` in `process_document()` with a `logger.warning` that records the failure reason. Preserves the graceful degradation behavior.

### Change

```python
# Before
except Exception:
    pass

# After
except Exception as e:
    logger.warning(
        f"[LLM:{source}] Structured document processing failed, "
        f"falling back to cleaner: {format_error_message(e)}"
    )
```

## Fix 3: Fallback Frontmatter Uses Source Metadata

**File**: `packages/markitai/src/markitai/llm/document.py`

### Design

In `_build_fallback_frontmatter()`, extract `description` and `tags` from `extra_meta` (populated by fetch strategies like defuddle) when available. Only affects the fallback path — normal LLM-generated frontmatter is unaffected.

### Change

Before calling `build_frontmatter_dict(description="", tags=[], ...)`, check `extra_meta`:

```python
fallback_desc = ""
fallback_tags: list[str] = []
if extra_meta:
    if isinstance(extra_meta.get("description"), str) and extra_meta["description"].strip():
        fallback_desc = extra_meta["description"]
        logger.info(f"[{source}] Using source metadata description as fallback")
    if isinstance(extra_meta.get("tags"), list) and extra_meta["tags"]:
        fallback_tags = extra_meta["tags"]
        logger.info(f"[{source}] Using source metadata tags as fallback")
```

Then pass `fallback_desc` and `fallback_tags` to `build_frontmatter_dict()`.

### No Change to `build_frontmatter_dict`

The `canonical_keys` blocking logic in `build_frontmatter_dict()` remains unchanged. The fix operates upstream in `_build_fallback_frontmatter()`.

## Testing

| Fix | Test |
|-----|------|
| Fix 1 | Mock `HybridRouter.acompletion()` to raise BadRequestError with "user location is not supported". Verify: (a) cooldown recorded, (b) retry selects different model. Also test `_call_llm_with_retry` retries on model-level errors. |
| Fix 2 | Mock `_process_document_combined` to raise. Verify warning log contains error message. |
| Fix 3 | Call `_build_fallback_frontmatter(extra_meta={"description": "test desc", "tags": ["a", "b"]})`. Verify output YAML contains description and tags. Also test with empty/missing extra_meta. |

## Out of Scope

- `RETRYABLE_ERRORS` tuple unchanged (BadRequestError not added globally)
- `build_frontmatter_dict` canonical_keys logic unchanged
- Instructor call logic in `_process_document_combined` unchanged
- `word_count` calculation in defuddle (separate concern)

---

## Implementation Tasks

### Task 1: HybridRouter model-level error cooldown

**Files:**
- Modify: `packages/markitai/src/markitai/llm/processor.py:314-542` (HybridRouter class)
- Test: `packages/markitai/tests/unit/test_llm_processor.py`

**Step 1: Write the failing tests**

Add to `TestHybridRouter` class in `test_llm_processor.py`:

```python
@pytest.mark.asyncio
async def test_acompletion_applies_cooldown_on_model_level_error(self):
    """Model-level 400 errors (region restriction) should trigger cooldown."""
    from litellm.exceptions import BadRequestError

    standard_router = MagicMock()
    standard_router.model_list = [
        {
            "model_name": "default",
            "litellm_params": {"model": "gemini/gemini-flash", "weight": 1},
        },
    ]
    standard_router.acompletion = AsyncMock(
        side_effect=BadRequestError(
            message="GeminiException BadRequestError - User location is not supported for the API use.",
            model="gemini/gemini-flash",
            llm_provider="gemini",
        )
    )

    local_wrapper = LocalProviderWrapper(
        [
            {
                "model_name": "default",
                "litellm_params": {"model": "claude-agent/sonnet", "weight": 1},
            }
        ]
    )

    hybrid = HybridRouter(standard_router, local_wrapper)

    with pytest.raises(BadRequestError):
        await hybrid.acompletion(
            "default", [{"role": "user", "content": "Hello"}]
        )

    # Model should be in cooldown
    assert "gemini/gemini-flash" in hybrid._model_cooldowns
    # Cooldown should be long (3600s)
    remaining = hybrid._model_cooldowns["gemini/gemini-flash"] - time.monotonic()
    assert remaining > 3500

@pytest.mark.asyncio
async def test_acompletion_no_cooldown_on_regular_bad_request(self):
    """Regular 400 errors (bad content) should NOT trigger model cooldown."""
    from litellm.exceptions import BadRequestError

    standard_router = MagicMock()
    standard_router.model_list = [
        {
            "model_name": "default",
            "litellm_params": {"model": "gemini/gemini-flash", "weight": 1},
        },
    ]
    standard_router.acompletion = AsyncMock(
        side_effect=BadRequestError(
            message="Invalid request: content too long",
            model="gemini/gemini-flash",
            llm_provider="gemini",
        )
    )

    local_wrapper = LocalProviderWrapper([])
    hybrid = HybridRouter(standard_router, local_wrapper)

    with pytest.raises(BadRequestError):
        await hybrid.acompletion(
            "default", [{"role": "user", "content": "Hello"}]
        )

    # No cooldown for regular errors
    assert "gemini/gemini-flash" not in hybrid._model_cooldowns
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py::TestHybridRouter::test_acompletion_applies_cooldown_on_model_level_error packages/markitai/tests/unit/test_llm_processor.py::TestHybridRouter::test_acompletion_no_cooldown_on_regular_bad_request -v -m ""`
Expected: FAIL — no cooldown logic exists for model-level errors yet

**Step 3: Implement HybridRouter model-level cooldown**

In `processor.py`, add class constant to `HybridRouter` (after line 383):

```python
# Model-level error patterns that indicate the model itself is unavailable
# (not a content/request issue). These warrant long cooldown.
MODEL_LEVEL_ERROR_PATTERNS = (
    "user location is not supported",
    "failed_precondition",
    "model is not available",
    "model not found",
    "model_not_available",
    "region is not supported",
    "not available in your region",
)
```

Then expand `acompletion()` error handler (line 528-542), after the existing rate limit block:

```python
except Exception as e:
    error_msg = str(e).lower()
    is_rate_limit = any(
        p in error_msg
        for p in ("429", "rate limit", "quota", "too many requests")
    )
    if is_rate_limit:
        cooldown_seconds = 60.0
        import re as _re

        match = _re.search(r"(\d+)\s*s", error_msg)
        if match:
            cooldown_seconds = float(match.group(1))
        self.record_cooldown(selected_model, cooldown_seconds)

    # Model-level errors: region restriction, model not available, etc.
    # Apply long cooldown so retries pick a different model.
    is_model_level = any(
        p in error_msg for p in self.MODEL_LEVEL_ERROR_PATTERNS
    )
    if is_model_level:
        self.record_cooldown(selected_model, 3600.0)
        logger.warning(
            f"[HybridRouter] Model {selected_model} unavailable "
            f"(model-level error), cooldown 3600s: "
            f"{format_error_message(e)}"
        )

    raise
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py::TestHybridRouter::test_acompletion_applies_cooldown_on_model_level_error packages/markitai/tests/unit/test_llm_processor.py::TestHybridRouter::test_acompletion_no_cooldown_on_regular_bad_request -v -m ""`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/llm/processor.py packages/markitai/tests/unit/test_llm_processor.py
git commit -m "fix: HybridRouter applies cooldown on model-level 400 errors"
```

---

### Task 2: _call_llm_with_retry retries on model-level errors

**Files:**
- Modify: `packages/markitai/src/markitai/llm/processor.py:1485-1516` (_call_llm_with_retry generic except)
- Test: `packages/markitai/tests/unit/test_llm_processor.py`

**Step 1: Write the failing test**

Add new test class or add to existing tests:

```python
class TestCallLlmModelLevelRetry:
    """Tests for _call_llm_with_retry retrying on model-level errors."""

    @pytest.mark.asyncio
    async def test_retries_on_model_level_error(
        self, llm_config, prompts_config
    ):
        """Model-level 400 should be retried, not immediately raised."""
        from litellm.exceptions import BadRequestError
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        # First call: model-level error. Second call: success.
        success_response = MagicMock()
        success_response.choices = [MagicMock()]
        success_response.choices[0].message.content = "OK"
        success_response.model = "claude-agent/sonnet"
        success_response.usage = MagicMock(
            prompt_tokens=10, completion_tokens=5
        )
        success_response._hidden_params = {}

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=[
                BadRequestError(
                    message="User location is not supported for the API use.",
                    model="gemini/gemini-flash",
                    llm_provider="gemini",
                ),
                success_response,
            ]
        )
        processor._router = mock_router

        result = await processor._call_llm_with_retry(
            "default",
            [{"role": "user", "content": "test"}],
            call_id="test:1",
            context="test",
            max_retries=2,
        )

        assert result.content == "OK"
        assert mock_router.acompletion.await_count == 2

    @pytest.mark.asyncio
    async def test_does_not_retry_regular_bad_request(
        self, llm_config, prompts_config
    ):
        """Regular 400 should raise immediately, not retry."""
        from litellm.exceptions import BadRequestError
        from markitai.llm import LLMProcessor

        processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

        mock_router = MagicMock()
        mock_router.acompletion = AsyncMock(
            side_effect=BadRequestError(
                message="Invalid request body",
                model="gemini/gemini-flash",
                llm_provider="gemini",
            )
        )
        processor._router = mock_router

        with pytest.raises(BadRequestError):
            await processor._call_llm_with_retry(
                "default",
                [{"role": "user", "content": "test"}],
                call_id="test:1",
                context="test",
                max_retries=2,
            )

        # Should NOT retry
        assert mock_router.acompletion.await_count == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py::TestCallLlmModelLevelRetry -v -m ""`
Expected: FAIL — `test_retries_on_model_level_error` raises instead of retrying

**Step 3: Implement model-level retry in _call_llm_with_retry**

In `_call_llm_with_retry()`, modify the generic `except Exception` block (line 1485+). Insert model-level check **before** existing auth detection:

```python
            except Exception as e:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                error_msg_lower = str(e).lower()

                # Model-level errors are retryable (HybridRouter cooldown
                # ensures the next attempt picks a different model)
                if any(
                    p in error_msg_lower
                    for p in HybridRouter.MODEL_LEVEL_ERROR_PATTERNS
                ):
                    last_exception = e
                    status_code = getattr(e, "status_code", "N/A")
                    if attempt < max_retries:
                        logger.warning(
                            f"[LLM:{call_id}] Model-level error "
                            f"(status={status_code}), retrying: "
                            f"{format_error_message(e)} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        delay = min(
                            DEFAULT_RETRY_BASE_DELAY * (2**attempt),
                            DEFAULT_RETRY_MAX_DELAY,
                        )
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(
                            f"[LLM:{call_id}] Model-level error after "
                            f"{max_retries + 1} attempts: "
                            f"{format_error_message(e)} "
                            f"time={elapsed_ms:.0f}ms"
                        )
                        raise

                # existing auth_patterns check and generic raise below...
                status_code = getattr(e, "status_code", "N/A")
                auth_patterns = (...)
                ...
                raise
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py::TestCallLlmModelLevelRetry -v -m ""`
Expected: PASS

**Step 5: Run full HybridRouter + related tests**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py::TestHybridRouter packages/markitai/tests/unit/test_llm_processor.py::TestCallLlmModelLevelRetry -v -m ""`
Expected: All PASS

**Step 6: Commit**

```bash
git add packages/markitai/src/markitai/llm/processor.py packages/markitai/tests/unit/test_llm_processor.py
git commit -m "fix: _call_llm_with_retry retries on model-level 400 errors"
```

---

### Task 3: Log structured processing failure in process_document

**Files:**
- Modify: `packages/markitai/src/markitai/llm/document.py:1292-1293`
- Test: `packages/markitai/tests/unit/test_document_utils.py`

**Step 1: Write the failing test**

Add to `TestProcessDocument` class in `test_document_utils.py`:

```python
@pytest.mark.asyncio
async def test_process_document_logs_warning_on_instructor_failure(
    self,
    llm_config: LLMConfig,
    prompts_config: PromptsConfig,
    mock_llm_response_factory,
) -> None:
    """process_document should log warning when structured processing fails."""
    from markitai.llm import LLMProcessor

    processor = LLMProcessor(llm_config, prompts_config, no_cache=True)

    clean_response = mock_llm_response_factory(
        content="# Cleaned\n\nContent."
    )
    mock_router = MagicMock()
    mock_router.acompletion = AsyncMock(return_value=clean_response)
    processor._router = mock_router

    with (
        patch("markitai.llm.document.instructor.from_litellm") as mock_instr,
        patch("markitai.llm.document.logger") as mock_logger,
    ):
        mock_client = MagicMock()
        mock_client.chat.completions.create_with_completion = AsyncMock(
            side_effect=ValueError("Instructor parse failed")
        )
        mock_instr.return_value = mock_client

        await processor.process_document("# Raw", "test.md")

    # Verify warning was logged (not silently swallowed)
    warning_calls = [
        call
        for call in mock_logger.warning.call_args_list
        if "Structured document processing failed" in str(call)
    ]
    assert len(warning_calls) >= 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_document_utils.py::TestProcessDocument::test_process_document_logs_warning_on_instructor_failure -v -m ""`
Expected: FAIL — current code has `except Exception: pass`, no warning logged

**Step 3: Implement the logging**

In `document.py`, replace line 1292-1293:

```python
# Before:
        except Exception:
            pass  # Fall through to fallback

# After:
        except Exception as e:
            logger.warning(
                f"[LLM:{source}] Structured document processing failed, "
                f"falling back to cleaner: {format_error_message(e)}"
            )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_document_utils.py::TestProcessDocument::test_process_document_logs_warning_on_instructor_failure -v -m ""`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/llm/document.py packages/markitai/tests/unit/test_document_utils.py
git commit -m "fix: log warning when structured document processing fails"
```

---

### Task 4: Fallback frontmatter uses extra_meta description/tags

**Files:**
- Modify: `packages/markitai/src/markitai/llm/document.py:1108-1142` (_build_fallback_frontmatter)
- Test: `packages/markitai/tests/unit/test_document_utils.py`

**Step 1: Write the failing tests**

Add to `TestBuildFallbackFrontmatter` class in `test_document_utils.py`:

```python
def test_uses_extra_meta_description(self) -> None:
    """Fallback should use description from extra_meta when available."""
    mixin = DocumentMixin()
    result = mixin._build_fallback_frontmatter(
        "test.pdf",
        "Content",
        extra_meta={"description": "A great article about testing"},
    )
    assert "A great article about testing" in result

def test_uses_extra_meta_tags(self) -> None:
    """Fallback should use tags from extra_meta when available."""
    mixin = DocumentMixin()
    result = mixin._build_fallback_frontmatter(
        "test.pdf",
        "Content",
        extra_meta={"tags": ["python", "testing"]},
    )
    assert "python" in result
    assert "testing" in result

def test_ignores_empty_extra_meta_description(self) -> None:
    """Empty description in extra_meta should not be used."""
    mixin = DocumentMixin()
    result = mixin._build_fallback_frontmatter(
        "test.pdf",
        "Content",
        extra_meta={"description": "  "},
    )
    assert "description: ''" in result

def test_ignores_non_string_extra_meta_description(self) -> None:
    """Non-string description in extra_meta should not be used."""
    mixin = DocumentMixin()
    result = mixin._build_fallback_frontmatter(
        "test.pdf",
        "Content",
        extra_meta={"description": 42},
    )
    assert "description: ''" in result

def test_no_extra_meta_still_works(self) -> None:
    """No extra_meta should produce empty description as before."""
    mixin = DocumentMixin()
    result = mixin._build_fallback_frontmatter("test.pdf", "Content")
    assert "description:" in result
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest packages/markitai/tests/unit/test_document_utils.py::TestBuildFallbackFrontmatter::test_uses_extra_meta_description packages/markitai/tests/unit/test_document_utils.py::TestBuildFallbackFrontmatter::test_uses_extra_meta_tags -v -m ""`
Expected: FAIL — description/tags from extra_meta are not used

**Step 3: Implement extra_meta extraction in _build_fallback_frontmatter**

In `document.py`, modify `_build_fallback_frontmatter()` (around line 1128-1141):

```python
    def _build_fallback_frontmatter(
        self,
        source: str,
        content: str,
        title: str | None = None,
        fetch_strategy: str | None = None,
        extra_meta: dict[str, Any] | None = None,
    ) -> str:
        """Build fallback frontmatter when LLM fails."""
        from markitai.utils.frontmatter import (
            build_frontmatter_dict,
            frontmatter_to_yaml,
        )

        # Extract description/tags from extra_meta as fallback
        fallback_desc = ""
        fallback_tags: list[str] = []
        if extra_meta:
            meta_desc = extra_meta.get("description")
            if isinstance(meta_desc, str) and meta_desc.strip():
                fallback_desc = meta_desc
                logger.info(
                    f"[{source}] Using source metadata description as fallback"
                )
            meta_tags = extra_meta.get("tags")
            if isinstance(meta_tags, list) and meta_tags:
                fallback_tags = meta_tags
                logger.info(
                    f"[{source}] Using source metadata tags as fallback"
                )

        frontmatter_dict = build_frontmatter_dict(
            source=source,
            description=fallback_desc,
            tags=fallback_tags,
            title=title,
            content=content,
            fetch_strategy=fetch_strategy,
            extra_meta=extra_meta,
        )
        return frontmatter_to_yaml(frontmatter_dict).strip()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest packages/markitai/tests/unit/test_document_utils.py::TestBuildFallbackFrontmatter -v -m ""`
Expected: All PASS (new + existing tests)

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/llm/document.py packages/markitai/tests/unit/test_document_utils.py
git commit -m "fix: fallback frontmatter uses description/tags from source metadata"
```

---

### Task 5: Integration verification

**Files:** None (verification only)

**Step 1: Run all affected test files**

Run: `uv run pytest packages/markitai/tests/unit/test_llm_processor.py packages/markitai/tests/unit/test_document_utils.py packages/markitai/tests/unit/test_llm.py packages/markitai/tests/unit/test_frontmatter.py -v -m ""`
Expected: All PASS

**Step 2: Run full unit test suite**

Run: `uv run pytest packages/markitai/tests/unit/ -x`
Expected: All PASS

**Step 3: Run linting**

Run: `uv run ruff check packages/markitai/src/markitai/llm/processor.py packages/markitai/src/markitai/llm/document.py`
Expected: No errors
