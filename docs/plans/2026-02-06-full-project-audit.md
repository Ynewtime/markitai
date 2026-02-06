# Full Project Audit Report & Fix Plan

Date: 2026-02-06

## Summary

8 parallel audit agents scanned the entire codebase across 8 dimensions: performance, CLI UX, logging/error handling, security, cross-platform compatibility, CI/CD, code quality, and configuration. ~120 issues identified, 16 HIGH, ~40 MEDIUM, ~60 LOW.

---

## Phase 1: HIGH Priority (Must Fix)

### 1.1 Performance — SQLite & Regex

**P-1. Deduplicate SQLiteCache/PersistentCache**
- `llm/processor.py:642-1059` duplicates `llm/cache.py:28-567` verbatim (~1000 lines)
- Fix: Remove from processor.py, import from cache.py

**P-2. Persistent SQLite connections**
- `llm/processor.py:671` and `llm/cache.py:58` create new `sqlite3.connect()` on every `_get_connection()` call
- Fix: Use persistent connection per instance (like FetchCache in `fetch.py:148`)

**P-3. Pre-compile regex patterns in content.py**
- `llm/content.py` — `protect_content()` (line 134), `unprotect_content()` (line 192), `extract_protected_content()` (line 81), `fix_malformed_image_refs()` all compile regex on every call
- Fix: Move to module-level `_PATTERN = re.compile(...)` constants

**P-4. Batch regex in apply_alt_text_updates()**
- `workflow/core.py:388` — compiles regex per image asset inside loop, O(images * markdown_length)
- Fix: Collect all replacements, single pass with alternation pattern

**P-5. Merge sequential regex in _remove_uncommented_screenshots()**
- `llm/document.py:1414-1562` — 10+ separate `re.sub()` calls on full document
- Fix: Combine compatible patterns into single alternation

### 1.2 Configuration Drift — Playwright Defaults

**C-1. Schema wait_for default mismatch**
- `config.schema.json:12` says `"domcontentloaded"`, actual default is `"networkidle"` (`constants.py:156`)
- Fix: Update schema to `"networkidle"`

**C-2. Schema extra_wait_ms default mismatch**
- `config.schema.json:23` says `3000`, actual default is `5000` (`constants.py:157`)
- Fix: Update schema to `5000`

**C-3. fetch_playwright.py hardcoded stale defaults**
- `fetch_playwright.py:205-206, 276-277` — function signatures hardcode `wait_for="domcontentloaded"` and `extra_wait_ms=3000`
- Fix: Use `DEFAULT_PLAYWRIGHT_WAIT_FOR` and `DEFAULT_PLAYWRIGHT_EXTRA_WAIT_MS` from constants

### 1.3 Logging — Missing Exception Chains

**L-1. Missing `from e` in re-raised exceptions**
- `converter/legacy.py:530` — `raise RuntimeError("...timed out")` loses timeout details
- `providers/claude_agent.py:564` — `raise RuntimeError(f"...{e}")` loses Claude exception chain
- `providers/copilot.py:717` — `raise RuntimeError(f"...{e}")` loses Copilot exception chain
- Fix: Add `from e` to all three

**L-2. Silent worker failures in image.py**
- `image.py:126, 187` — `_compress_image_cv2/pillow` return None on any exception, caller can't distinguish no-op from crash
- Fix: Log at debug level with exception details

### 1.4 Code Quality — Provider Duplication

**Q-1. Provider SDK ~200 lines duplicated**
- `providers/claude_agent.py` and `providers/copilot.py` share `_messages_to_prompt`, `_has_images`, `completion()`, `_UNSUPPORTED_PARAMS`
- Fix: Extract `BaseLocalProvider(CustomLLM)` base class in `providers/base_local.py`

### 1.5 CI/CD — Publish Safety

**CI-1. Publish workflow has no CI gate**
- `.github/workflows/publish.yml` — release can publish to PyPI even if tests fail
- Fix: Add `needs: [lint, test]` or require CI status checks

### 1.6 CLI UX — Misleading Output

**UX-1. `-v` flag maps to `--version` instead of `--verbose`**
- `cli/main.py:226-227`
- Fix: Use `-V` for version (Python convention)

**UX-2. `cache stats --json` outputs ANSI via console.print()**
- `cli/commands/cache.py:136-138` — breaks piping to jq
- Fix: Use `click.echo()` for JSON output (like doctor.py:621)

**UX-3. Fabricated parse time metric**
- `cli/processors/file.py:263` — `parse_time = duration * 0.8` is not a real measurement
- Fix: Remove fake breakdown or add actual timing

**UX-4. Stale deprecation date**
- `providers/__init__.py:344` — "will be retired on February 13, 2025" (over a year ago)
- Fix: Remove deprecated models or update to past tense

---

## Phase 2: MEDIUM Priority (Should Fix)

### 2.1 Performance

| # | Problem | File |
|---|---------|------|
| P-6 | `img.copy()` unnecessary before compress | `image.py:963` |
| P-7 | BatchState properties iterate all files on every call | `batch.py:136-181` |
| P-8 | `download_url_images()` loop string replacement O(N*len) | `image.py:1507` |
| P-9 | `_compute_llm_usage/_compute_summary` double iteration | `batch.py:817-911` |
| P-10 | `discover_files()` collects list before dedup | `batch.py:721` |
| P-11 | New ImageProcessor per PDF page in OCR mode | `converter/pdf.py:473` |

### 2.2 Configuration

| # | Problem | File |
|---|---------|------|
| C-4 | Schema missing `heavy_task_limit` field | `config.schema.json` |
| C-5 | Config int fields missing `ge=` constraints (12+ fields) | `config.py` multiple |
| C-6 | No JSON parse error handling in config loading | `config.py:489-492` |
| C-7 | `config set` doesn't support array index notation | `config.py:622-649` |

### 2.3 Logging & Error Handling

| # | Problem | File |
|---|---------|------|
| L-3 | `image.py` has 10+ swallowed exceptions (except: pass/continue) | `image.py` throughout |
| L-4 | `converter/text.py` has no logging at all | `converter/text.py` |
| L-5 | `converter/base.py` — `detect_format()` returns UNKNOWN silently | `converter/base.py` |
| L-6 | `workflow/core.py` uses overly broad `except Exception` | `core.py:174, 212` |
| L-7 | `workflow/single.py` catches Exception instead of ProviderError | `single.py:137, 293, 343` |
| L-8 | `FetchCache` has `close()` but no async context manager | `fetch.py` |
| L-9 | `workflow/helpers.py:393` swallows JSON decode errors silently | `helpers.py:393-400` |

### 2.4 CLI UX

| # | Problem | File |
|---|---------|------|
| UX-5 | Raw `console.print("[red]Error...")` instead of `ui.error()` | `main.py`, `interactive.py`, `doctor.py` |
| UX-6 | Most user-facing strings bypass i18n system | `main.py`, `interactive.py`, `init.py` |
| UX-7 | Interactive mode URL validation only checks len > 0 | `interactive.py:158` |
| UX-8 | Exit codes inconsistent (SystemExit vs sys.exit vs ctx.exit) | multiple files |
| UX-9 | Interactive mode subprocess exit code not propagated | `main.py:79` |
| UX-10 | init wizard shows "not found" for every unavailable provider | `init.py:202` |
| UX-11 | `main.py:358` logs warning for missing config (should be debug) | `main.py:358` |
| UX-12 | No progress indicator for provider detection in init/interactive | `init.py:194` |

### 2.5 Code Quality

| # | Problem | File |
|---|---------|------|
| Q-2 | `is_vision_model()` triplicated | `validators.py:60`, `doctor.py:575`, `processor.py:1702` |
| Q-3 | `clean_control_characters` — two different implementations! | `utils/text.py:9` vs `providers/json_mode.py:37` |
| Q-4 | `process_with_standard_llm` 110 lines with deep nesting | `workflow/core.py:509` |
| Q-5 | `analyze_images` 155 lines, does too much | `workflow/single.py:141` |
| Q-6 | Inconsistent error handling: some propagate, others swallow | workflow functions |
| Q-7 | Stale deprecation date in providers | `providers/__init__.py:344` |
| Q-8 | Fabricated parse_time metric | `cli/processors/file.py:263` |

### 2.6 CI/CD

| # | Problem | File |
|---|---------|------|
| CI-2 | Version duplicated in two pyproject.toml files | root + packages/ |
| CI-3 | No `uv lock --check` in CI | `ci.yml` |
| CI-4 | No coverage threshold enforcement in CI | `ci.yml` |
| CI-5 | Coverage only uploaded as artifact, no Codecov integration | `ci.yml:100-108` |
| CI-6 | No version tag validation in publish workflow | `publish.yml` |
| CI-7 | LibreOffice not installed on Windows CI runners | `ci.yml:80-88` |
| CI-8 | No `uv lock --check` to catch lockfile drift | `ci.yml` |

### 2.7 Security

| # | Problem | File |
|---|---------|------|
| S-1 | PowerShell file path injection (mitigated by single-quote escape) | `converter/legacy.py:132` |
| S-2 | URL fetching has no SSRF protection (no private IP blocking) | `fetch.py`, `fetch_playwright.py` |

---

## Phase 3: LOW Priority (Nice to Have)

### Performance (5)
- `re` redundantly imported at function scope (`core.py:644`)
- `fnmatch` imported inside method on every call (`processor.py:993`, `cache.py:383`)
- `hashlib` imported inside method (`cache.py:526`)
- Proxy env vars checked every batch (`image.py:1324`)
- Text utility imports inside method body (`document.py:1399`)

### Configuration (8)
- `batch.py:788` hardcoded fallback `5` instead of constant `10`
- `DEFAULT_CACHE_CONTENT_TRUNCATE` unused constant
- `DEFAULT_LOG_PANEL_MAX_LINES` unused constant
- `MARKITAI_LOG_DIR` env var not reflected in config system
- `config set` doesn't handle JSON arrays/objects
- `config get` returns None for legitimate null values
- `OCRConfig.lang` has no format validation
- Minimal config template hardcodes model name

### Logging (10+)
- `cli/main.py:596` LiteLLM cleanup exception swallowed
- `converter/pdf.py:155,164` compression/dimension failures silent
- `converter/office.py:332,337` COM cleanup exceptions silent
- `fetch_playwright.py:241,427` content fallback exceptions silent
- `ocr.py:89` info-level for internal preheat (should be debug)
- `converter/office.py:396` info-level for timing (should be debug)

### CLI UX (12)
- Subcommands lack usage examples
- `config set` doesn't document valid key paths
- `--no-compress` not paired with `--compress`
- `--no-cache` vs `--no-cache-for` naming confusion
- `config set` validation error loses attempted value
- Interactive mode doesn't handle Ctrl+C (None returns)
- `doctor --fix` only supports Playwright but not documented
- `config validate` has no progress indicator

### Security (5)
- Credential path exposure in error messages (`auth.py:147,216`)
- SQLite `check_same_thread=False` without full write locking (`fetch.py:159`)
- `ensure_subdir` doesn't validate name parameter (`paths.py:29`)
- Setup scripts pipe remote code to shell (`setup.sh:552`, `setup.ps1:528`)
- Image download without size limits (`image.py:1416`)

### Cross-Platform (2)
- No Windows reserved filename checking (CON, PRN, etc.) in sanitize functions
- Windows RAM detection falls back to conservative default of 2

### Code Quality (14+)
- SDK availability checks duplicated (`auth.py` vs provider files)
- `_context_display_name` duplicated in LLM mixins
- `get_response_cost` duplicated
- Backward compatibility aliases that may be unused
- 3 empty `TYPE_CHECKING` blocks
- Deprecated `concurrency_limit` parameter still in API
- `images` vs `assets` naming transition incomplete
- `ConversionContext.converter` typed as `Any`
- Dict-based page_images without TypedDict
- `apply_alt_text_updates` uses `Any` for image_analysis
- `utils/progress.py` creates second Console to avoid circular import
- 7+ deferred imports in `workflow/core.py`
- No tests for `utils/office.py`, `utils/output.py`, `utils/paths.py`

---

## Positive Findings

- **Cross-platform**: Excellent 3-platform support with proper guards, fallbacks, retry logic
- **Security fundamentals**: Parameterized SQL, no shell=True, atomic writes, symlink checks, path validation
- **Test coverage**: 58 test files, 2005 tests, 3 OS x 3 Python CI matrix
- **Logging infrastructure**: loguru + InterceptHandler + level-based routing
- **File safety**: Windows retry logic for atomic rename/unlink is production-quality
- **Setup scripts**: POSIX + PowerShell with matching features and i18n

---

## Implementation Order

1. **Phase 1** (HIGH) — 16 items, highest impact, many are quick fixes
   - Start with config drift fixes (C-1, C-2, C-3) — 3 files, 5 min
   - Then exception chain fixes (L-1) — 3 files, 5 min
   - Then CI publish gate (CI-1) — 1 file, 5 min
   - Then CLI fixes (UX-1 through UX-4) — 4 files, 15 min
   - Then SQLite dedup (P-1, P-2) — 2 files, 30 min
   - Then regex precompile (P-3, P-4, P-5) — 3 files, 30 min
   - Then provider base class (Q-1) — 3 files, 60 min

2. **Phase 2** (MEDIUM) — ~40 items, grouped by file proximity
   - Config validation + schema sync
   - image.py logging improvements
   - CLI output standardization
   - Code dedup (is_vision_model, clean_control_characters)
   - CI improvements

3. **Phase 3** (LOW) — ~60 items, opportunistic fixes during related work
