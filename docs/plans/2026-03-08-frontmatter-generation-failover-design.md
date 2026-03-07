# Frontmatter Generation Failover Design

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
