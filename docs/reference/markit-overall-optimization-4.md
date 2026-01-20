# Markit Overall Optimization Report 4

> Deep-dive analysis of stability, performance, and consistency risks

---

## Scope

- Static analysis only (no runtime execution)
- Focus areas: spec alignment, reliability, performance/resource usage, security/privacy, maintainability

---

## High Priority Findings

### 1) Spec vs Implementation Drift

- Python version: spec says >=3.13, implementation enforces ==3.13.*. This reduces deploy flexibility and contradicts the spec.
  - Evidence: `packages/markit/pyproject.toml`, `pyproject.toml`, `docs/spec.md`
- Hash length mismatch: spec says 12 hex digits, implementation uses 6. Short hashes increase collision risk for state/report files.
  - Evidence: `docs/spec.md` vs `packages/markit/src/markit/cli.py` `compute_task_hash()`
- Report naming mismatch: spec describes reports under `reports/{dir_name}.report.json`, code generates `reports/markit.<hash>.report.json`.
  - Evidence: `docs/spec.md` vs `packages/markit/src/markit/cli.py` and `packages/markit/src/markit/batch.py`

Impact: user expectations around resume/outputs diverge from actual behavior, causing confusion and potential data loss when relying on file names.

---

### 2) Reliability Risks From Exception Handling

- Many error paths swallow exceptions (`except Exception`) and continue, which makes failures silent and hard to debug, especially in batch runs.
  - Evidence: `packages/markit/src/markit/llm.py`, `packages/markit/src/markit/image.py`, `packages/markit/src/markit/cli.py`, `packages/markit/src/markit/converter/office.py`, `packages/markit/src/markit/converter/legacy.py`
- OCR temp file handling uses `NamedTemporaryFile(delete=False)` and deletes in `finally`, but exceptions can still leak temporary files.
  - Evidence: `packages/markit/src/markit/ocr.py` `recognize_bytes()`

Impact: hidden failures reduce reliability; temp file leakage grows over time on long-running systems.

---

### 3) Resource Pressure (Concurrency + Caches + Large Files)

- Default concurrency is high and uncoordinated: batch (10) + LLM (10) + OCR page thread pool can stack unexpectedly.
  - Evidence: `packages/markit/src/markit/constants.py`, `packages/markit/src/markit/batch.py`, `packages/markit/src/markit/converter/pdf.py`
- LLM image cache is capped at 500MB in-memory; persistent cache is 1GB per layer, enabled by default.
  - Evidence: `packages/markit/src/markit/llm.py`, `packages/markit/src/markit/constants.py`
- PDF pipeline can render screenshots, extract embedded images, and run OCR in parallel, causing repeated heavy I/O and CPU.
  - Evidence: `packages/markit/src/markit/converter/pdf.py`

Impact: OOM risk on large documents, rate-limit pressure on LLM providers, disk pressure due to cache growth.

---

### 4) Security/Privacy Exposure via Logs

- Default log level is DEBUG; LLM calls log model, token counts, and detailed metadata to file logs.
  - Evidence: `packages/markit/src/markit/constants.py`, `packages/markit/src/markit/cli.py`, `packages/markit/src/markit/llm.py`
- Output dir symlink safety only checks the target path once; nested symlinks inside output are not validated.
  - Evidence: `packages/markit/src/markit/security.py`, `packages/markit/src/markit/cli.py`

Impact: sensitive workflow data can leak via logs; symlink path traversal may write outside intended directories.

---

### 5) Correctness and Data Integrity

- Short task hash (6 hex) can collide and cause reports/states to overwrite each other.
  - Evidence: `packages/markit/src/markit/cli.py`
- File conflict resolver increments versions up to 9999, but lacks a fallback if conflicts exceed this limit.
  - Evidence: `packages/markit/src/markit/cli.py` `resolve_output_path()`
- LLM image reference cleanup can remove valid references if names don’t match assets on disk, with no fallback.
  - Evidence: `packages/markit/src/markit/image.py` `remove_nonexistent_images()`

Impact: silent data loss or wrong outputs for large batch runs or edge cases.

---

### 6) Architecture and Duplication

- LLM/vision/image handling logic is duplicated between `cli.py` and `SingleFileWorkflow`, increasing divergence risk.
  - Evidence: `packages/markit/src/markit/cli.py`, `packages/markit/src/markit/workflow/single.py`
- Multiple TODOs in `llm.py` indicate incomplete typing/contract consistency.
  - Evidence: `packages/markit/src/markit/llm.py` (TODO markers)

Impact: maintenance cost grows and behavior can diverge across paths.

---

## Root Cause Themes

- Spec alignment gaps: docs describe one behavior; implementation uses another.
- Overlapping pipelines: OCR + screenshots + embedded extraction produce repeated work.
- Error handling strategy is overly tolerant, masking failures.
- Defaults are tuned for development (DEBUG, high concurrency) rather than stable production use.

---

## Recommended Deep Fix Directions

1) Align spec and implementation:
   - Standardize report/state naming and hash length.
   - Clarify Python version policy across docs and configs.

2) Establish resource governance:
   - Introduce a unified concurrency controller (batch + LLM + OCR).
   - Make cache limits configurable and reduce default sizes.

3) Harden error handling:
   - Replace broad `except Exception` with typed exceptions where possible.
   - Escalate failures for critical paths; only degrade when documented.

4) Logging and privacy controls:
   - Change default log level to INFO; sanitize LLM request metadata.
   - Validate nested symlinks in output paths.

5) Consolidate workflow:
   - Route all single-file processing through `SingleFileWorkflow`.
   - De-duplicate LLM/image logic in `cli.py`.

---

## Evidence Index

- `packages/markit/pyproject.toml`
- `pyproject.toml`
- `docs/spec.md`
- `packages/markit/src/markit/cli.py`
- `packages/markit/src/markit/batch.py`
- `packages/markit/src/markit/llm.py`
- `packages/markit/src/markit/ocr.py`
- `packages/markit/src/markit/image.py`
- `packages/markit/src/markit/security.py`
- `packages/markit/src/markit/converter/pdf.py`
- `packages/markit/src/markit/converter/office.py`
- `packages/markit/src/markit/converter/legacy.py`
