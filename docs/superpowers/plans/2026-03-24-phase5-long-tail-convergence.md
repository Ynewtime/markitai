# Phase 5: Long-Tail Convergence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix remaining achievable parity failures through retry strategy enhancement, footnote protection, and quality gate fix.

**Architecture:** Implement defuddle's hidden-content retry (spec Module 5.3) to handle card-style index pages and hidden content. Protect footnote elements from hidden removal. Fix the _QUOTE_CARD_PATTERN false positive (spec Module 3.2).

**Tech Stack:** Python, BeautifulSoup, pytest

**Spec:** `docs/superpowers/specs/2026-03-23-webextract-quality-speed-optimization-design.md` (Modules 3.2, 5.3)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `packages/markitai/src/markitai/webextract/pipeline.py` | Add hidden-content retry + index-page retry |
| Modify | `packages/markitai/src/markitai/webextract/removals/hidden.py` | Protect footnote elements from hidden removal |
| Modify | `packages/markitai/src/markitai/webextract/quality.py` | Fix _QUOTE_CARD_PATTERN false positive |

---

### Task 1: Hidden-content retry and index-page mode

When content is very sparse after extraction, try using the largest hidden content subtree, then try with all removals disabled (index page mode).

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py`
- Modify: `packages/markitai/tests/unit/webextract/test_pipeline.py`

### Task 2: Protect footnote elements from hidden removal

Footnote definitions often use `display:none` or `visibility:hidden` for styling. They should be protected like math elements.

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/removals/hidden.py`
- Test: existing hidden removal tests

### Task 3: Fix _QUOTE_CARD_PATTERN false positive

The `^Quote$` pattern in quality.py false-positives on tweets that legitimately quote other tweets.

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/quality.py`
- Test: existing quality tests

### Task 4: Verify parity improvement

Run parity + full suite. Target: ≤ 7 failures.
