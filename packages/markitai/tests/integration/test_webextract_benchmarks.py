from __future__ import annotations

"""Benchmark guardrails for native HTML extraction.

These tests verify that extraction performance does not regress beyond
defined time budgets.  Marked ``@pytest.mark.slow`` because they run
multiple iterations.
"""

import re
import time
from dataclasses import dataclass
from pathlib import Path

import pytest

from markitai.webextract import extract_web_content

_FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "web"

# Time budget in milliseconds for a single-iteration extraction.
# This is intentionally generous to avoid flakiness on slow CI runners.
_SINGLE_ITERATION_BUDGET_MS = 1000


@dataclass
class BenchmarkStats:
    """Performance statistics for a fixture extraction benchmark.

    Attributes:
        fixture_name: Name of the fixture that was benchmarked.
        total_ms: Wall-clock time in milliseconds for all iterations.
        iterations: Number of iterations performed.
        avg_ms: Average time per iteration in milliseconds.
        min_ms: Fastest iteration in milliseconds.
        max_ms: Slowest iteration in milliseconds.
    """

    fixture_name: str
    total_ms: float
    iterations: int
    avg_ms: float
    min_ms: float
    max_ms: float


def _extract_og_url(html: str) -> str | None:
    """Extract og:url from raw HTML for proper extractor routing.

    Args:
        html: Raw HTML content.

    Returns:
        The og:url value, or ``None`` if not found.
    """
    match = re.search(
        r'<meta[^>]+property=["\']og:url["\'][^>]+content=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:url["\']',
        html,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def run_fixture_benchmark(
    fixture_name: str,
    iterations: int = 10,
) -> BenchmarkStats:
    """Benchmark extraction performance on a fixture.

    Reads the fixture HTML once and then runs ``extract_web_content``
    ``iterations`` times, measuring wall-clock time for each run.

    Args:
        fixture_name: Base name of the fixture (e.g.
            ``"x_status_2030105637204676808"``).
        iterations: Number of times to run extraction.  Defaults to 10.

    Returns:
        A :class:`BenchmarkStats` with timing statistics.
    """
    html_path = _FIXTURE_DIR / f"{fixture_name}.playwright.html"
    html = html_path.read_text(encoding="utf-8")
    url = _extract_og_url(html) or f"https://example.com/{fixture_name}"

    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        extract_web_content(html, url)
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)

    total_ms = sum(times_ms)
    return BenchmarkStats(
        fixture_name=fixture_name,
        total_ms=total_ms,
        iterations=iterations,
        avg_ms=total_ms / iterations,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_native_extraction_benchmark_does_not_regress_beyond_budget() -> None:
    """Single-iteration extraction must complete within the time budget.

    Budget: 1000 ms per iteration (generous for slow CI machines).
    We measure the total across 10 iterations and check average.
    """
    stats = run_fixture_benchmark("x_status_2030105637204676808", iterations=10)
    # Total for 10 iterations should be well under 10 seconds.
    # The budget per iteration is 1000 ms.
    assert stats.total_ms < _SINGLE_ITERATION_BUDGET_MS * stats.iterations, (
        f"Extraction too slow: avg {stats.avg_ms:.1f} ms/iter "
        f"(budget: {_SINGLE_ITERATION_BUDGET_MS} ms/iter)"
    )


@pytest.mark.slow
def test_generic_article_benchmark_does_not_regress() -> None:
    """Generic article extraction must complete within the time budget."""
    stats = run_fixture_benchmark("generic_article", iterations=10)
    assert stats.total_ms < _SINGLE_ITERATION_BUDGET_MS * stats.iterations, (
        f"Extraction too slow: avg {stats.avg_ms:.1f} ms/iter "
        f"(budget: {_SINGLE_ITERATION_BUDGET_MS} ms/iter)"
    )


def test_single_extraction_completes_quickly() -> None:
    """A single extraction call must complete within the budget (non-slow marker)."""
    html_path = _FIXTURE_DIR / "x_status_2030105637204676808.playwright.html"
    html = html_path.read_text(encoding="utf-8")
    url = _extract_og_url(html) or "https://x.com/ixiaowenz/status/2030105637204676808"

    start = time.perf_counter()
    extract_web_content(html, url)
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert elapsed_ms < _SINGLE_ITERATION_BUDGET_MS, (
        f"Single extraction took {elapsed_ms:.1f} ms "
        f"(budget: {_SINGLE_ITERATION_BUDGET_MS} ms)"
    )
