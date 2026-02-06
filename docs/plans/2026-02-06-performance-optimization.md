# Performance Optimization Plan

Date: 2026-02-06

## Summary

7 performance bottlenecks identified through systematic investigation. Fixes ordered by impact.

---

## Fix 1: Heavy Task Semaphore — Configurable + Auto-Detect

**Problem:** `executor.py:59` hardcodes `_HEAVY_TASK_LIMIT = 2`. Batch mode with 10 concurrency only converts 2 Office/PDF files simultaneously.

**Solution:**
1. Add `heavy_task_limit` field to `BatchConfig` in `config.py` (default: `0` = auto)
2. Auto-detect based on system memory in `executor.py`:
   - RAM >= 32GB → 8
   - RAM >= 16GB → 4
   - RAM >= 8GB → 3
   - else → 2
3. `get_heavy_task_semaphore()` reads from config, falls back to auto-detect
4. Pass config value from batch processor to executor

**Files:**
- `config.py` — add `heavy_task_limit` field
- `constants.py` — add `DEFAULT_HEAVY_TASK_LIMIT = 0` (auto)
- `utils/executor.py` — auto-detect logic + accept config override
- `workflow/core.py` — pass config to semaphore init

---

## Fix 2: PDF Standard Mode — Parallel Page Rendering

**Problem:** `pdf.py:202-226` renders page screenshots sequentially in a `for` loop. OCR mode already uses `ThreadPoolExecutor`.

**Solution:**
1. Extract common parallel page rendering function: `_render_pages_parallel(doc, pages, output_dir, dpi, max_workers)`
2. Both OCR and standard mode call this shared function
3. Adaptive worker count (same logic as OCR mode):
   - Small (<10MB): `min(cpu_count // 2, total_pages, 6)`
   - Medium (10-50MB): up to 4
   - Large (>50MB): up to 2

**Files:**
- `converter/pdf.py` — extract `_render_pages_parallel()`, refactor both paths

---

## Fix 3: CLI Lazy Import — Processors + Commands

**Problem:** `main.py:44-50` eagerly imports all processor modules; `main.py:606-616` eagerly imports all command modules. Even `markitai --help` loads litellm, converters, etc.

**Solution:**
1. **Processors:** Move imports into the main() function body, import conditionally:
   - `process_single_file` — only when processing a file
   - `process_batch` — only when processing a directory
   - `process_url` / `process_url_batch` — only when processing URLs
2. **Commands:** Use Click's lazy loading pattern:
   - Replace direct imports with `@app.group()` lazy registration
   - Or simply move `from ... import` + `app.add_command()` into a `_register_commands()` function called after argument parsing determines the subcommand

**Files:**
- `cli/main.py` — restructure imports

---

## Fix 4: Doctor/Init — Parallel Checks

**Problem:** `doctor.py:195-675` runs all checks sequentially (Playwright, LibreOffice, FFmpeg, RapidOCR, auth checks). `init.py:82-137` also sequential.

**Solution:**
1. **doctor:** Group independent checks into a `concurrent.futures.ThreadPoolExecutor`:
   - Group 1 (system tools): Playwright, LibreOffice, FFmpeg, RapidOCR — all independent
   - Group 2 (auth): Claude auth, Copilot auth — independent of each other
   - Group 3 (models): Vision model check — depends on config only
   - Run Group 1 + Group 2 in parallel, then Group 3
2. **init:** Same pattern for `_check_deps()` — run all 4 checks in parallel
3. Output order remains deterministic (collect results, print in fixed order)

**Files:**
- `cli/commands/doctor.py` — parallelize `_doctor_impl()`
- `cli/commands/init.py` — parallelize `_check_deps()`

---

## Fix 5: Async Blocking Operations — asyncio.to_thread()

**Problem:** CPU-intensive operations block the event loop:
- `workflow/core.py:252-254` — base64 image extraction (regex)
- `workflow/core.py:289-303` — base64 replacement with paths (regex)
- `llm/processor.py:2349` — base64 encoding (CPU)

**Solution:**
Wrap blocking calls with `await asyncio.to_thread()`:
1. `process_embedded_images()` in core.py:
   - `extract_base64_images()` → `await asyncio.to_thread()`
   - `replace_base64_with_paths()` → `await asyncio.to_thread()`
2. `_get_cached_image()` in processor.py:
   - base64 encoding on cache miss → `await asyncio.to_thread()`
   - Note: already has LRU cache, so this only affects first read per image

**Files:**
- `workflow/core.py` — wrap image extraction/replacement
- `llm/processor.py` — wrap base64 encoding (if method is async-compatible)

---

## Fix 6: Cache Stats — Merge Queries

**Problem:** `cache stats --verbose` executes 3 separate full table scans:
1. `stats()` — COUNT + SUM
2. `stats_by_model()` — GROUP BY model
3. `list_entries()` — ORDER BY + SUBSTR

**Solution:**
Add `stats_verbose()` method to `SQLiteCache` that combines queries:
1. Single query for count, total size, and by-model breakdown
2. Separate query for recent entries (unavoidable, different shape)
Result: 3 queries → 2 queries, and the main stats query is a single pass.

**Files:**
- `llm/cache.py` — add `stats_verbose()` combining stats + by_model
- `cli/commands/cache.py` — call `stats_verbose()` in verbose mode

---

## Fix 7: Fetch Cache Lock — Async-Safe Access

**Problem:** `fetch.py:172` uses `threading.Lock()` for all SQLite access. Under concurrent URL fetching (5 URLs), cache reads/writes serialize.

**Solution:**
Replace `threading.Lock()` with `asyncio.Lock()` for async contexts, keeping thread lock for sync contexts:
1. Add `async_lock: asyncio.Lock` to FetchCache
2. Async methods use `async with self._async_lock:`
3. Sync methods keep existing `threading.Lock()`
4. SQLite connection per-operation (already the case) ensures thread safety

**Files:**
- `fetch.py` — add async lock, update async cache methods

---

## Implementation Order

1. Fix 3 (CLI lazy import) — highest user-visible impact, lowest risk
2. Fix 4 (doctor/init parallel) — quick win
3. Fix 1 (heavy task semaphore) — config change + auto-detect
4. Fix 2 (PDF parallel rendering) — refactoring needed
5. Fix 5 (async blocking) — mechanical changes
6. Fix 6 (cache stats merge) — small optimization
7. Fix 7 (fetch cache lock) — small optimization

## Testing Strategy

- Existing test suite (2005 tests) must pass after each fix
- Performance verification: time `markitai doctor` and `markitai --help` before/after Fix 3+4
- Batch mode test: convert 10+ Office files to verify Fix 1+2 improvement
