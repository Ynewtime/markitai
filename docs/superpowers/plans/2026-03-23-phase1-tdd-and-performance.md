# Phase 1: TDD Foundation & Performance Optimization

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish defuddle parity test infrastructure and optimize the webextract pipeline's performance by eliminating redundant parsing, joining selectors, and caching across retries.

**Architecture:** Add defuddle's 83 HTML fixtures as quality benchmarks via a copy script. Introduce `_ExtractionContext` to cache parsed DOM and metadata across retry levels, replace per-selector querying with joined CSS selectors, and merge markdown preprocessing into standardization to eliminate an extra BeautifulSoup parse.

**Scope:** This plan covers Module 7 **P0 optimizations only** (redundant parsing, selector joining, preprocessing merge). Module 7 P1 items (single-pass batch removal, candidate stats memoization) will be addressed in a follow-up plan after P0 improvements are measured.

**Tech Stack:** Python, pytest, BeautifulSoup (lxml), markitai webextract pipeline

**Spec:** `docs/superpowers/specs/2026-03-23-webextract-quality-speed-optimization-design.md` (Modules 6 & 7)

**Defuddle source:** `/home/y/dev/defuddle` (test fixtures at `tests/fixtures/` and `tests/expected/`)

---

## File Structure

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `scripts/sync_defuddle_fixtures.sh` | Copy fixtures from local defuddle repo at pinned commit |
| Create | `packages/markitai/tests/defuddle_fixtures/VERSION` | Record defuddle commit hash |
| Create | `packages/markitai/tests/defuddle_fixtures/fixtures/` | HTML input files (copied) |
| Create | `packages/markitai/tests/defuddle_fixtures/expected/` | Expected markdown files (copied) |
| Create | `packages/markitai/tests/integration/test_defuddle_parity_quality.py` | Parameterized parity tests against defuddle fixtures |
| Modify | `packages/markitai/src/markitai/webextract/pipeline.py` | Introduce `_ExtractionContext`, eliminate redundant `parse_html()` in retries |
| Modify | `packages/markitai/src/markitai/webextract/removals/selectors.py` | Join exact selectors into single CSS query |
| Modify | `packages/markitai/src/markitai/webextract/pipeline.py` | Also: bypass `render_markdown()` in `_extract_once()` to avoid redundant re-parse |
| Modify | `packages/markitai/tests/integration/test_webextract_benchmarks.py` | Add defuddle fixture benchmarks |

---

### Task 1: Sync defuddle test fixtures

**Files:**
- Create: `scripts/sync_defuddle_fixtures.sh`
- Create: `packages/markitai/tests/defuddle_fixtures/VERSION`
- Create: `packages/markitai/tests/defuddle_fixtures/fixtures/*.html` (copied)
- Create: `packages/markitai/tests/defuddle_fixtures/expected/*.md` (copied)

- [ ] **Step 1: Create the sync script**

```bash
#!/usr/bin/env bash
# scripts/sync_defuddle_fixtures.sh
# Copy test fixtures from local defuddle repo at a pinned commit.
# Usage: ./scripts/sync_defuddle_fixtures.sh /path/to/defuddle

set -euo pipefail

DEFUDDLE_DIR="${1:?Usage: $0 /path/to/defuddle}"
DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/packages/markitai/tests/defuddle_fixtures"

if [[ ! -d "$DEFUDDLE_DIR/tests/fixtures" ]]; then
    echo "Error: $DEFUDDLE_DIR/tests/fixtures not found" >&2
    exit 1
fi

# Ensure destination exists before writing anything
mkdir -p "$DEST_DIR/fixtures" "$DEST_DIR/expected"

# Record version
COMMIT=$(git -C "$DEFUDDLE_DIR" rev-parse HEAD)
echo "defuddle commit: $COMMIT" > "$DEST_DIR/VERSION"
echo "synced at: $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$DEST_DIR/VERSION"

# Sync fixtures (clean then copy)
rm -rf "$DEST_DIR/fixtures/"* "$DEST_DIR/expected/"*
cp "$DEFUDDLE_DIR"/tests/fixtures/*.html "$DEST_DIR/fixtures/"
cp "$DEFUDDLE_DIR"/tests/expected/*.md "$DEST_DIR/expected/"

echo "Synced $(ls "$DEST_DIR/fixtures/" | wc -l) fixtures, $(ls "$DEST_DIR/expected/" | wc -l) expected files"
echo "From defuddle commit: $COMMIT"
```

- [ ] **Step 2: Run the sync script**

```bash
chmod +x scripts/sync_defuddle_fixtures.sh
./scripts/sync_defuddle_fixtures.sh /home/y/dev/defuddle
```

Expected: `Synced 83 fixtures, 85 expected files`

- [ ] **Step 3: Add .gitignore for large fixture files if needed, verify files exist**

```bash
ls packages/markitai/tests/defuddle_fixtures/fixtures/ | wc -l
ls packages/markitai/tests/defuddle_fixtures/expected/ | wc -l
cat packages/markitai/tests/defuddle_fixtures/VERSION
```

- [ ] **Step 4: Commit**

```bash
git add scripts/sync_defuddle_fixtures.sh packages/markitai/tests/defuddle_fixtures/
git commit -m "test: add defuddle fixture sync script and initial fixtures"
```

---

### Task 2: Defuddle parity test infrastructure

**Files:**
- Create: `packages/markitai/tests/integration/test_defuddle_parity_quality.py`

The existing `test_defuddle_parity.py` uses markitai's own fixtures with `expected.json` contracts. We create a **separate** test file for defuddle's Markdown-based expected output.

- [ ] **Step 1: Write the parity test module**

```python
"""Parity tests: markitai extraction vs defuddle expected output.

Runs extract_web_content on defuddle's HTML fixtures and compares
against defuddle's expected Markdown output. This establishes a
quality baseline and tracks improvement as we port defuddle features.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from markitai.webextract import extract_web_content

_DEFUDDLE_DIR = Path(__file__).parents[1] / "defuddle_fixtures"
_FIXTURES_DIR = _DEFUDDLE_DIR / "fixtures"
_EXPECTED_DIR = _DEFUDDLE_DIR / "expected"


@dataclass
class DefuddleFixture:
    """A defuddle test fixture with HTML input and expected Markdown."""

    name: str
    html_path: Path
    expected_path: Path
    url: str


def _extract_og_url(html: str) -> str | None:
    """Extract og:url from raw HTML for proper extractor routing."""
    match = re.search(
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:url["\']',
        html, re.IGNORECASE,
    )
    return match.group(1) if match else None


def _extract_canonical_url(html: str) -> str | None:
    """Extract <link rel="canonical"> URL."""
    match = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
        html, re.IGNORECASE,
    )
    return match.group(1) if match else None


def _fixture_url(html_path: Path) -> str:
    """Determine the best URL for a fixture.

    Priority: og:url from HTML → canonical URL → fallback from filename.
    This matches the existing test harness strategy.
    """
    html = html_path.read_text(encoding="utf-8", errors="replace")
    url = _extract_og_url(html) or _extract_canonical_url(html)
    if url:
        return url
    # Fallback: construct from filename (last resort)
    name = html_path.stem
    parts = name.split("--", 1)
    domain_part = parts[1] if len(parts) > 1 else parts[0]
    return f"https://{domain_part}"


def _collect_fixtures() -> list[DefuddleFixture]:
    """Discover all paired fixture/expected files."""
    if not _FIXTURES_DIR.exists():
        return []
    fixtures = []
    for html_path in sorted(_FIXTURES_DIR.glob("*.html")):
        name = html_path.stem
        expected_path = _EXPECTED_DIR / f"{name}.md"
        if expected_path.exists():
            url = _fixture_url(html_path)
            fixtures.append(
                DefuddleFixture(
                    name=name,
                    html_path=html_path,
                    expected_path=expected_path,
                    url=url,
                )
            )
    return fixtures


def _parse_expected_markdown(path: Path) -> tuple[dict[str, str], str]:
    """Parse defuddle expected output: JSON metadata preamble + markdown body.

    Defuddle expected files have the format:
        ```json
        {"title": "...", "author": "...", ...}
        ```

        [markdown content]

    Returns:
        Tuple of (metadata dict, markdown body string).
    """
    text = path.read_text(encoding="utf-8")
    # Find JSON block at start
    json_match = re.search(
        r"```json\s*\n(.*?)\n\s*```", text, re.DOTALL
    )
    metadata: dict[str, str] = {}
    body = text
    if json_match:
        try:
            metadata = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
        body = text[json_match.end() :].strip()
    return metadata, body


def _word_count(text: str) -> int:
    """Count words in text, handling CJK characters."""
    # CJK characters count as individual words
    cjk = len(re.findall(
        r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]",
        text,
    ))
    words = len(re.sub(
        r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]",
        " ",
        text,
    ).split())
    return words + cjk


ALL_FIXTURES = _collect_fixtures()
FIXTURE_IDS = [f.name for f in ALL_FIXTURES]


# Known site-chrome noise patterns that should NOT appear in clean extraction
_NOISE_PATTERNS: list[str] = [
    "Sign up",
    "Log in",
    "Cookie",
    "Accept all cookies",
    "Privacy Policy",
    "Terms of Service",
    "Don't miss what's happening",
    "Something went wrong",
    "Retry",
    "New to X?",
]


@pytest.mark.skipif(not ALL_FIXTURES, reason="No defuddle fixtures found")
class TestDefuddleParityQuality:
    """Level 1 parity: metadata + content coverage + no severe noise.

    NOTE: This is a baseline-establishment task, not a TDD cycle.
    Some tests are expected to fail initially — they track quality
    improvement as we port defuddle features.
    """

    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=FIXTURE_IDS)
    def test_content_is_not_empty(self, fixture: DefuddleFixture) -> None:
        """Extraction must produce non-empty markdown."""
        html = fixture.html_path.read_text(encoding="utf-8")
        result = extract_web_content(html, fixture.url)
        assert result.markdown.strip(), f"{fixture.name}: empty markdown"

    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=FIXTURE_IDS)
    def test_metadata_title_extracted(
        self, fixture: DefuddleFixture
    ) -> None:
        """Extraction must produce a non-empty title when defuddle does."""
        expected_meta, _ = _parse_expected_markdown(fixture.expected_path)
        expected_title = expected_meta.get("title", "")
        if not expected_title:
            pytest.skip("Defuddle expected has no title")

        html = fixture.html_path.read_text(encoding="utf-8")
        result = extract_web_content(html, fixture.url)
        actual_title = getattr(result.metadata, "title", "") or ""
        assert actual_title.strip(), (
            f"{fixture.name}: no title extracted "
            f"(defuddle expects: {expected_title!r})"
        )

    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=FIXTURE_IDS)
    def test_no_site_chrome_noise(self, fixture: DefuddleFixture) -> None:
        """Extracted markdown must not contain known site-chrome noise."""
        html = fixture.html_path.read_text(encoding="utf-8")
        result = extract_web_content(html, fixture.url)
        md = result.markdown

        found_noise = [p for p in _NOISE_PATTERNS if p in md]
        assert not found_noise, (
            f"{fixture.name}: site chrome noise found: {found_noise}"
        )

    @pytest.mark.parametrize("fixture", ALL_FIXTURES, ids=FIXTURE_IDS)
    def test_word_count_within_tolerance(
        self, fixture: DefuddleFixture
    ) -> None:
        """Word count should be within 50% of defuddle's output.

        This is intentionally generous — we tighten as we port features.
        """
        html = fixture.html_path.read_text(encoding="utf-8")
        _, expected_body = _parse_expected_markdown(fixture.expected_path)
        result = extract_web_content(html, fixture.url)

        expected_wc = _word_count(expected_body)
        actual_wc = _word_count(result.markdown)

        if expected_wc == 0:
            pytest.skip("Expected output has zero words")

        ratio = actual_wc / expected_wc
        assert 0.5 < ratio < 2.0, (
            f"{fixture.name}: word count ratio {ratio:.2f} "
            f"(actual={actual_wc}, expected={expected_wc})"
        )
```

- [ ] **Step 2: Run the tests to establish baseline**

```bash
cd packages/markitai && uv run pytest tests/integration/test_defuddle_parity_quality.py -v --tb=short 2>&1 | tail -30
```

Expected: Some tests pass, some fail. This is the baseline. Record pass/fail counts.

- [ ] **Step 3: Commit**

```bash
git add packages/markitai/tests/integration/test_defuddle_parity_quality.py
git commit -m "test: add defuddle parity quality tests (baseline)"
```

---

### Task 3: Profile extraction performance baseline

**Files:**
- Modify: `packages/markitai/tests/integration/test_webextract_benchmarks.py`

- [ ] **Step 1: Add profiling benchmark for defuddle fixtures**

Add to the end of `test_webextract_benchmarks.py`:

```python
@pytest.mark.slow
def test_defuddle_fixture_extraction_performance() -> None:
    """Profile extraction time on defuddle fixtures to establish baseline."""
    defuddle_dir = Path(__file__).parents[1] / "defuddle_fixtures" / "fixtures"
    if not defuddle_dir.exists():
        pytest.skip("No defuddle fixtures")

    results: list[tuple[str, float]] = []
    for html_path in sorted(defuddle_dir.glob("*.html"))[:10]:  # Sample 10
        html = html_path.read_text(encoding="utf-8")
        url = _extract_og_url(html) or f"https://{html_path.stem.split('--', 1)[-1]}"

        start = time.perf_counter()
        extract_web_content(html, url)
        elapsed_ms = (time.perf_counter() - start) * 1000
        results.append((html_path.stem, elapsed_ms))

    # Report
    for name, ms in sorted(results, key=lambda x: -x[1]):
        print(f"  {ms:7.1f}ms  {name}")

    avg_ms = sum(ms for _, ms in results) / len(results)
    print(f"\n  Average: {avg_ms:.1f}ms over {len(results)} fixtures")

    # Each fixture should complete within budget
    for name, ms in results:
        assert ms < _SINGLE_ITERATION_BUDGET_MS, (
            f"{name}: {ms:.1f}ms exceeds {_SINGLE_ITERATION_BUDGET_MS}ms budget"
        )
```

Also add the missing import at the top if not present:

```python
from pathlib import Path
```

- [ ] **Step 2: Run baseline profiling**

```bash
cd packages/markitai && uv run pytest tests/integration/test_webextract_benchmarks.py::test_defuddle_fixture_extraction_performance -v -s
```

Expected: Performance numbers printed. Record baseline for comparison after optimization.

- [ ] **Step 3: Commit**

```bash
git add packages/markitai/tests/integration/test_webextract_benchmarks.py
git commit -m "test: add defuddle fixture performance baseline benchmark"
```

---

### Task 3.5: Validate deepcopy vs re-parse performance

Before committing to `copy.deepcopy` in Task 5, verify it is faster than re-parsing. The spec warns that deepcopy on BeautifulSoup trees may be slow due to recursive back-references.

**Files:** No files created — run in-session microbenchmark.

- [ ] **Step 1: Run microbenchmark**

```bash
cd packages/markitai && python -c "
import copy, time
from markitai.webextract.dom import parse_html
from pathlib import Path

# Use a large fixture
fixtures_dir = Path('tests/defuddle_fixtures/fixtures')
html_files = sorted(fixtures_dir.glob('*.html'))
html = max(html_files, key=lambda f: f.stat().st_size).read_text()
print(f'HTML size: {len(html):,} bytes')

# Benchmark parse_html
times = []
for _ in range(5):
    start = time.perf_counter()
    soup = parse_html(html)
    times.append((time.perf_counter() - start) * 1000)
print(f'parse_html:     avg {sum(times)/len(times):.1f}ms')

# Benchmark deepcopy
soup = parse_html(html)
times = []
for _ in range(5):
    start = time.perf_counter()
    copy.deepcopy(soup)
    times.append((time.perf_counter() - start) * 1000)
print(f'copy.deepcopy:  avg {sum(times)/len(times):.1f}ms')
"
```

Expected: deepcopy should be faster than parse_html. If deepcopy is slower, Task 5 should instead cache `raw_html` and re-parse from string (still faster than the current approach which also re-runs `_pick_root` and `_maybe_apply_schema_fallback` per retry).

---

### Task 4: Join exact selectors into single CSS query

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/removals/selectors.py:33-46`
- Modify: `packages/markitai/src/markitai/webextract/constants.py:14`
- Test: `packages/markitai/tests/unit/webextract/test_removals.py` (existing)

- [ ] **Step 1: Write failing test for joined selector behavior**

Add to existing selector tests or create if needed:

```python
def test_joined_selector_removes_same_elements_as_individual() -> None:
    """Joining selectors into one query must produce identical results."""
    html = """
    <div>
        <nav class="navigation">nav</nav>
        <div class="ad">ad</div>
        <footer>footer</footer>
        <article>
            <p>Main content here with enough words to be meaningful.</p>
        </article>
        <aside class="sidebar">sidebar</aside>
    </div>
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    root = soup.find("div")
    assert root is not None

    removed = remove_by_selectors(root, main_content=None, use_partial=False)
    # nav, ad, footer, aside should all be removed
    remaining_text = root.get_text(strip=True)
    assert "Main content" in remaining_text
    assert "nav" not in remaining_text.lower().split()
    assert "sidebar" not in remaining_text
```

- [ ] **Step 2: Run test to verify it passes (validates current behavior)**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_removals.py -v -k "test_joined"
```

- [ ] **Step 3: Add EXACT_SELECTORS_JOINED constant**

In `packages/markitai/src/markitai/webextract/constants.py`, after the `EXACT_SELECTORS` list (around line 146), add:

```python
# Pre-joined CSS selector for single-query removal (performance optimization).
# Some selectors may not be valid for joining (e.g., pseudo-selectors) — those
# are kept in EXACT_SELECTORS_UNJOINABLE and queried individually.
EXACT_SELECTORS_JOINED: str = ", ".join(EXACT_SELECTORS)
```

- [ ] **Step 4: Update `remove_by_selectors` to use joined query**

Replace Phase 1 in `packages/markitai/src/markitai/webextract/removals/selectors.py:33-46`:

```python
    # Phase 1: Exact CSS selectors (single joined query for performance)
    try:
        for el in root.select(EXACT_SELECTORS_JOINED):
            eid = id(el)
            if eid in seen_ids:
                continue
            if _should_protect(el, main_content):
                continue
            to_remove.append(el)
            seen_ids.add(eid)
    except Exception:  # noqa: BLE001
        # Fallback: query individually if joined selector fails
        for selector in EXACT_SELECTORS:
            try:
                for el in root.select(selector):
                    eid = id(el)
                    if eid in seen_ids:
                        continue
                    if _should_protect(el, main_content):
                        continue
                    to_remove.append(el)
                    seen_ids.add(eid)
            except Exception:  # noqa: BLE001
                continue
```

Update import:

```python
from markitai.webextract.constants import (
    EXACT_SELECTORS,
    EXACT_SELECTORS_JOINED,
    PARTIAL_SELECTOR_REGEX,
    TEST_ATTRIBUTES,
)
```

- [ ] **Step 5: Run all existing selector tests**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_removals.py -v
```

Expected: All tests pass.

- [ ] **Step 6: Run full webextract test suite for regression**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/ -x -q
```

Expected: No regressions.

- [ ] **Step 7: Commit**

```bash
git add packages/markitai/src/markitai/webextract/constants.py packages/markitai/src/markitai/webextract/removals/selectors.py
git commit -m "perf: join exact selectors into single CSS query"
```

---

### Task 5: Introduce ExtractionContext to eliminate redundant parsing

This is the biggest performance win — replacing up to 5 `parse_html()` calls with 1 parse + `copy.deepcopy()`.

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py:155-373`
- Test: existing `packages/markitai/tests/unit/webextract/test_pipeline.py`

- [ ] **Step 1: Snapshot current extraction output BEFORE refactoring**

Run the following to capture a golden snapshot of the current output. This will be compared AFTER the refactor to verify equivalence:

```bash
cd packages/markitai && uv run python -c "
from pathlib import Path
from markitai.webextract import extract_web_content

fixture_dir = Path('tests/fixtures/web')
html = (fixture_dir / 'x_status_2030105637204676808.playwright.html').read_text()
url = 'https://x.com/ixiaowenz/status/2030105637204676808'
result = extract_web_content(html, url)

snapshot = Path('tests/unit/webextract/_extraction_context_snapshot.txt')
snapshot.write_text(f'word_count={result.word_count}\n---\n{result.markdown}')
print(f'Snapshot saved: {result.word_count} words, {len(result.markdown)} chars')
"
```

- [ ] **Step 2: Write regression test that compares against the snapshot**

```python
def test_extraction_context_output_matches_pre_refactor_snapshot() -> None:
    """Verify ExtractionContext refactor produces identical output."""
    from pathlib import Path

    snapshot_path = Path(__file__).parent / "_extraction_context_snapshot.txt"
    if not snapshot_path.exists():
        pytest.skip("Pre-refactor snapshot not found — run Step 1 first")

    fixture_dir = Path(__file__).parents[2] / "fixtures" / "web"
    html_path = fixture_dir / "x_status_2030105637204676808.playwright.html"
    if not html_path.exists():
        pytest.skip("Fixture not found")

    html = html_path.read_text(encoding="utf-8")
    url = "https://x.com/ixiaowenz/status/2030105637204676808"
    result = extract_web_content(html, url)

    snapshot = snapshot_path.read_text(encoding="utf-8")
    header, expected_md = snapshot.split("\n---\n", 1)
    expected_wc = int(header.split("=")[1])

    assert result.word_count == expected_wc, (
        f"Word count changed: {result.word_count} vs {expected_wc}"
    )
    assert result.markdown == expected_md, "Markdown output changed after refactor"
```

- [ ] **Step 2b: Run test — should pass (snapshot matches current output)**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_pipeline.py -v -k "test_extraction_context_output"
```

- [ ] **Step 3: Apply `_ExtractionContext` + `_extract_generic` + `_extract_with_retry` refactor atomically**

**IMPORTANT**: Steps 3a-3c below MUST be applied together in a single edit session before running tests. They change the function signatures and call patterns — applying them separately would break the code.

**Step 3a**: Add `_ExtractionContext` class before `_extract_generic` in pipeline.py:

```python
import copy


class _ExtractionContext:
    """Cache expensive computations across retry levels.

    Avoids re-parsing HTML on each retry — parses once, then uses
    ``copy.deepcopy`` for fresh roots on each retry attempt.
    """

    def __init__(self, html: str, url: str) -> None:
        self.raw_html = html
        self.url = url
        self.original_soup = parse_html(html)
        self.metadata = extract_metadata(self.original_soup, url)
        self.md_instance = _create_markitdown()

    def fresh_soup_and_root(
        self, extractor: object | None, diagnostics: dict[str, object]
    ) -> tuple[BeautifulSoup, Tag | BeautifulSoup]:
        """Return a fresh deep-copy of the parsed soup with root selected."""
        soup = copy.deepcopy(self.original_soup)
        root = _pick_root(soup, extractor)
        root = _maybe_apply_schema_fallback(soup, root, diagnostics)
        return soup, root
```

**Step 3b**: Refactor `_extract_generic` to use context

Replace the body of `_extract_generic` (lines 155-216):

```python
def _extract_generic(html: str, url: str) -> ExtractedWebContent:
    """Run the generic extraction pipeline (no resolver match)."""
    ctx = _ExtractionContext(html, url)
    extractor = find_extractor(url)
    root = _pick_root(ctx.original_soup, extractor)
    diagnostics: dict[str, object] = {
        "extractor": extractor.name if extractor is not None else "generic",
        "schema_fallback_used": False,
        "adaptive_retry_used": False,
        "removed_partial_selectors": False,
    }

    root = _maybe_apply_schema_fallback(ctx.original_soup, root, diagnostics)

    # Multi-level extraction with adaptive retry
    result = _extract_with_retry(
        ctx,
        root,
        diagnostics,
        extractor=extractor,
    )

    clean_html = result[0]
    markdown = result[1]
    word_count = count_words(markdown)

    extractor_name = extractor.name if extractor is not None else "generic"
    content_profile = _EXTRACTOR_CONTENT_PROFILES.get(
        extractor_name, ContentProfile.GENERIC_ARTICLE
    )

    info = ExtractionInfo(
        content_profile=content_profile,
        extractor_name=extractor_name,
        word_count=word_count,
    )

    quality = assess_native_markdown(markdown, profile=content_profile.value)

    return ExtractedWebContent(
        clean_html=clean_html,
        markdown=markdown,
        metadata=ctx.metadata,
        word_count=word_count,
        info=info,
        quality=quality,
        semantic=None,
        diagnostics={**diagnostics, "metadata": asdict(ctx.metadata)},
    )
```

**Step 3c**: Refactor `_extract_with_retry` to use context instead of re-parsing.

Replace `_extract_with_retry` signature and body to use `_ExtractionContext`:

```python
def _extract_with_retry(
    ctx: _ExtractionContext,
    root: Tag | BeautifulSoup,
    diagnostics: dict[str, object],
    *,
    extractor: object | None,
) -> tuple[str, str]:
    """Multi-level adaptive retry extraction using cached context."""
    url = getattr(ctx.metadata, "canonical_url", "") or ""
    use_scoring = extractor is None

    # Level 1: Full pipeline
    clean_html, markdown, removal_stats = _extract_once(
        root, ctx.metadata, ctx.md_instance, url, use_scoring=use_scoring,
    )
    diagnostics["removal_stats"] = removal_stats
    word_count = count_words(markdown)

    schema_used = diagnostics.get("schema_fallback_used", False)
    if word_count >= _RETRY_SPARSE_THRESHOLD or schema_used:
        return clean_html, markdown

    # Level 2: Retry without partial selectors (deepcopy instead of re-parse)
    _soup2, root2 = ctx.fresh_soup_and_root(extractor, diagnostics)
    clean2, md2, _ = _extract_once(
        root2, ctx.metadata, ctx.md_instance, url,
        use_partial_selectors=False, use_scoring=use_scoring,
    )
    wc2 = count_words(md2)
    if wc2 > word_count * 2:
        clean_html, markdown, word_count = clean2, md2, wc2
        diagnostics["adaptive_retry_used"] = True
        diagnostics["retry_level"] = 2
    if word_count >= _RETRY_SPARSE_THRESHOLD:
        return clean_html, markdown

    # Level 3: Retry without hidden element removal
    _soup3, root3 = ctx.fresh_soup_and_root(extractor, diagnostics)
    clean3, md3, _ = _extract_once(
        root3, ctx.metadata, ctx.md_instance, url,
        use_hidden_removal=False, use_scoring=use_scoring,
    )
    wc3 = count_words(md3)
    if wc3 > word_count:
        clean_html, markdown, word_count = clean3, md3, wc3
        diagnostics["adaptive_retry_used"] = True
        diagnostics["retry_level"] = 3
    if word_count >= _RETRY_SPARSE_THRESHOLD:
        return clean_html, markdown

    # Level 4: Retry with all removals disabled
    _soup4, root4 = ctx.fresh_soup_and_root(extractor, diagnostics)
    clean4, md4, _ = _extract_once(
        root4, ctx.metadata, ctx.md_instance, url,
        use_partial_selectors=False, use_hidden_removal=False,
        use_scoring=False,
    )
    wc4 = count_words(md4)
    if wc4 > word_count:
        clean_html, markdown, word_count = clean4, md4, wc4
        diagnostics["adaptive_retry_used"] = True
        diagnostics["retry_level"] = 4

    # Fallback: broaden to <body> (deepcopy instead of re-parse)
    if word_count <= _RETRY_VERY_SPARSE_THRESHOLD:
        body_soup = copy.deepcopy(ctx.original_soup)
        body = body_soup.body
        if body is not None:
            body_html, body_md, _ = _extract_once(
                body, ctx.metadata, ctx.md_instance, url,
                use_partial_selectors=False, use_hidden_removal=False,
                use_scoring=False,
            )
            if count_words(body_md) > word_count:
                clean_html = body_html
                markdown = body_md
                diagnostics["adaptive_retry_used"] = True
                diagnostics["retry_level"] = "body_fallback"

    return clean_html, markdown
```

- [ ] **Step 4: Remove now-unused `_rebuild_retry_root` function**

Delete the `_rebuild_retry_root` function (lines 364-373 in old code) — it is fully replaced by `_ExtractionContext.fresh_soup_and_root()`.

- [ ] **Step 5: Run all webextract tests**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/ -x -q
cd packages/markitai && uv run pytest tests/integration/test_defuddle_parity.py -x -q
cd packages/markitai && uv run pytest tests/integration/test_webextract_benchmarks.py -x -q
```

Expected: All tests pass. If `copy.deepcopy` causes issues, the determinism test from Step 1 will catch it.

- [ ] **Step 6: Commit**

```bash
git add packages/markitai/src/markitai/webextract/pipeline.py
git commit -m "perf: introduce ExtractionContext to eliminate redundant HTML parsing"
```

---

### Task 6: Eliminate extra parse in pipeline's markdown path

**Goal**: Eliminate the redundant `BeautifulSoup` parse that `_preprocess_for_markdown()` creates inside `render_markdown()`. **DO NOT modify `render_markdown()`'s public contract** — external callers (including `test_markdown_fidelity.py`) depend on it performing preprocessing on its own.

**Strategy**: In the pipeline's `_extract_once()`, call the three preprocessing functions directly on the already-parsed `root` Tag, then call `_html_to_markdown()` directly (bypassing `render_markdown()` and its redundant re-parse). `render_markdown()` remains unchanged for all external callers.

**Files:**
- Modify: `packages/markitai/src/markitai/webextract/pipeline.py:223-248` (`_extract_once`)
- Test: existing tests in `packages/markitai/tests/unit/webextract/`

- [ ] **Step 1: Write regression test to snapshot current pipeline output**

```python
def test_pipeline_output_stable_after_preprocess_bypass() -> None:
    """Bypassing render_markdown in pipeline must not change extraction output."""
    html = """
    <article>
        <p>Main content paragraph with enough words to be meaningful for extraction.</p>
        <img srcset="small.jpg 400w, large.jpg 800w" src="small.jpg" alt="test">
        <iframe src="https://www.youtube.com/embed/dQw4w9WgXcQ"></iframe>
        <figure><img src="photo.jpg" alt="photo"><figcaption>A caption</figcaption></figure>
    </article>
    """
    result = extract_web_content(html, "https://example.com/article")
    assert "large.jpg" in result.markdown  # srcset picks largest
    assert "youtube.com/watch" in result.markdown  # embed canonicalized
    assert "caption" in result.markdown.lower()  # figcaption preserved
```

- [ ] **Step 2: Run test to verify it passes (current behavior)**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/test_pipeline.py -v -k "test_pipeline_output_stable"
```

- [ ] **Step 3: Modify `_extract_once` to bypass `render_markdown`**

In `pipeline.py`, change `_extract_once` (around line 223-248). Instead of calling `render_markdown(clean_html)` which re-parses HTML for preprocessing, call the preprocessing functions on the `root` Tag directly, then call `_html_to_markdown()`:

```python
from markitai.webextract.markdown import (
    _html_to_markdown,
    _postprocess_markdown,
    _resolve_srcset,
    _canonicalize_embeds,
    _preserve_figure_captions,
)

def _extract_once(
    root: Tag | BeautifulSoup,
    metadata: object,
    md_instance: object,
    url: str,
    *,
    use_partial_selectors: bool = True,
    use_hidden_removal: bool = True,
    use_scoring: bool = True,
) -> tuple[str, str, dict[str, int]]:
    """Run extraction pipeline once and return (clean_html, markdown, removal_stats)."""
    title = getattr(metadata, "title", None)
    removal_stats: dict[str, int] = {}
    if isinstance(root, Tag):
        removal_stats = apply_removals(
            root,
            use_partial_selectors=use_partial_selectors,
            use_hidden_removal=use_hidden_removal,
            use_scoring=use_scoring,
        )
    if isinstance(root, Tag):
        standardize_content(root, title=title, base_url=url)
    sanitize_tag_tree(root)

    # Apply markdown preprocessing on the already-parsed Tag (avoids re-parse)
    _resolve_srcset(root)
    _canonicalize_embeds(root)
    _preserve_figure_captions(root)

    clean_html = str(root)
    markdown = _html_to_markdown(clean_html, md_instance)
    markdown = _postprocess_markdown(markdown)
    return clean_html, markdown, removal_stats
```

- [ ] **Step 4: Run all tests — including markdown fidelity tests**

```bash
cd packages/markitai && uv run pytest tests/unit/webextract/ -x -q
cd packages/markitai && uv run pytest tests/unit/webextract/test_markdown_fidelity.py -v
cd packages/markitai && uv run pytest tests/integration/ -x -q
```

Expected: All tests pass. `render_markdown()` is unchanged, so `test_markdown_fidelity.py` tests remain green. The pipeline regression test from Step 1 confirms identical output.

- [ ] **Step 5: Commit**

```bash
git add packages/markitai/src/markitai/webextract/pipeline.py
git commit -m "perf: bypass render_markdown in pipeline to eliminate extra BeautifulSoup parse"
```

---

### Task 7: Measure performance improvement

**Files:**
- No new files — run existing benchmarks

- [ ] **Step 1: Run performance benchmarks**

```bash
cd packages/markitai && uv run pytest tests/integration/test_webextract_benchmarks.py -v -s
```

- [ ] **Step 2: Run defuddle fixture benchmarks**

```bash
cd packages/markitai && uv run pytest tests/integration/test_webextract_benchmarks.py::test_defuddle_fixture_extraction_performance -v -s
```

- [ ] **Step 3: Compare with baseline from Task 3**

Compare the average ms/fixture before and after optimizations. Document the improvement.

- [ ] **Step 4: Run full test suite to verify everything is green**

```bash
cd packages/markitai && uv run pytest tests/ -x -q
```

Expected: All tests pass. Phase 1 complete.

---

## What's Next

After Phase 1 lands:
- **Phase 2 plan**: Markdown engine enhancement (Module 1 P0) + Playwright speed optimization (Module 4)
- **Phase 3 plan**: Noise removal (Module 2) + FxTwitter API (Module 3.1)
- **Phase 4 plan**: Markdown engine P1-P2 + Pipeline enhancement (Module 5) + remaining Module 3
