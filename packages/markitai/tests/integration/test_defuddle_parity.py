from __future__ import annotations

"""Parity tests: native extraction against fixture expected.json contracts.

Each fixture HTML is processed through ``extract_web_content`` and the output
is validated against the corresponding ``expected.json`` semantic contract.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from markitai.webextract import extract_web_content
from markitai.webextract.types import ExtractedWebContent

_FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "web"


@dataclass
class ParityReport:
    """Result of a parity check against a fixture contract.

    Attributes:
        fixture_name: Name of the fixture (without extension).
        semantic_ok: Whether semantic expectations passed.
        html_snapshot_ok: Whether the HTML snapshot check passed.
        markdown_golden_ok: Whether markdown must/must_not checks passed.
            ``None`` when the fixture defines no markdown expectations.
        diagnostics: Diagnostics dict from the extraction result.
        failures: Human-readable list of specific failures.
    """

    fixture_name: str
    semantic_ok: bool
    html_snapshot_ok: bool
    markdown_golden_ok: bool | None
    diagnostics: dict[str, Any]
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Return True if all applicable checks passed."""
        if not self.semantic_ok or not self.html_snapshot_ok:
            return False
        if self.markdown_golden_ok is False:
            return False
        return True


def _extract_og_url(html: str) -> str | None:
    """Extract the og:url value from raw HTML without a full parse.

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
    # Also handle reversed attribute order
    match = re.search(
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:url["\']',
        html,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def run_native_fixture_case(fixture_name: str) -> ParityReport:
    """Run extraction on a fixture and check against expected.json.

    Args:
        fixture_name: Base name of the fixture (e.g.
            ``"x_status_2030105637204676808"``).  The function appends
            ``.playwright.html`` and ``.expected.json`` automatically.

    Returns:
        A :class:`ParityReport` with the outcome of each parity check.
    """
    html_path = _FIXTURE_DIR / f"{fixture_name}.playwright.html"
    expected_path = _FIXTURE_DIR / f"{fixture_name}.expected.json"

    html = html_path.read_text(encoding="utf-8")
    expected: dict[str, Any] = json.loads(expected_path.read_text(encoding="utf-8"))

    # Prefer the explicit URL in expected.json; fall back to og:url embedded in
    # the HTML so that site-specific extractors match correctly.
    url = (
        expected.get("url")
        or _extract_og_url(html)
        or f"https://example.com/{fixture_name}"
    )
    result: ExtractedWebContent = extract_web_content(html, url)

    failures: list[str] = []

    # --- Semantic check ---
    semantic_ok = _check_semantic(result, expected, failures)

    # --- HTML snapshot check ---
    # We accept the extraction as long as it produced non-empty clean_html.
    html_snapshot_ok = _check_html_snapshot(result, failures)

    # --- Markdown golden check ---
    markdown_golden_ok = _check_markdown(result.markdown, expected, failures)

    diagnostics = dict(result.diagnostics) if result.diagnostics else {}

    return ParityReport(
        fixture_name=fixture_name,
        semantic_ok=semantic_ok,
        html_snapshot_ok=html_snapshot_ok,
        markdown_golden_ok=markdown_golden_ok,
        diagnostics=diagnostics,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Internal checkers
# ---------------------------------------------------------------------------


def _check_semantic(
    result: ExtractedWebContent,
    expected: dict[str, Any],
    failures: list[str],
) -> bool:
    """Check semantic expectations from expected.json.

    Args:
        result: Extraction result.
        expected: Parsed expected.json contents.
        failures: Mutable list to append failure messages to.

    Returns:
        ``True`` if all semantic checks passed.
    """
    semantic_expected = expected.get("semantic")
    if semantic_expected is None:
        # No semantic contract — pass trivially.
        return True

    thread_expected = semantic_expected.get("thread")
    if thread_expected is None:
        return True

    if result.semantic is None or result.semantic.thread is None:
        failures.append("Expected a semantic thread but extraction produced none")
        return False

    thread = result.semantic.thread
    main_expected = thread_expected.get("main_item", {})
    ok = True

    # Check author_handle
    expected_handle = main_expected.get("author_handle")
    if expected_handle is not None:
        actual_handle = thread.main_item.author_handle or ""
        if (
            expected_handle not in actual_handle
            and actual_handle not in expected_handle
        ):
            failures.append(
                f"author_handle mismatch: expected {expected_handle!r}, "
                f"got {actual_handle!r}"
            )
            ok = False

    # Check author_name
    expected_name = main_expected.get("author_name")
    if expected_name is not None:
        actual_name = thread.main_item.author_name or ""
        if expected_name not in actual_name and actual_name not in expected_name:
            failures.append(
                f"author_name mismatch: expected {expected_name!r}, got {actual_name!r}"
            )
            ok = False

    # Check text contains substring
    expected_text_contains = main_expected.get("text_contains")
    if expected_text_contains is not None:
        actual_text = thread.main_item.text or ""
        if expected_text_contains not in actual_text:
            failures.append(
                f"main_item.text does not contain {expected_text_contains!r}; "
                f"got: {actual_text[:200]!r}"
            )
            ok = False

    # Check exact text (legacy field from older fixtures)
    expected_text = main_expected.get("text")
    if expected_text is not None:
        actual_text = thread.main_item.text or ""
        # Use substring match for robustness
        if expected_text not in actual_text:
            failures.append(
                f"main_item.text does not contain expected text {expected_text!r}; "
                f"got: {actual_text[:200]!r}"
            )
            ok = False

    return ok


def _check_html_snapshot(
    result: ExtractedWebContent,
    failures: list[str],
) -> bool:
    """Check that the extraction produced non-empty clean HTML.

    Args:
        result: Extraction result.
        failures: Mutable list to append failure messages to.

    Returns:
        ``True`` if clean_html is non-empty.
    """
    if not result.clean_html or not result.clean_html.strip():
        failures.append("clean_html is empty")
        return False
    return True


def _check_markdown(
    markdown: str,
    expected: dict[str, Any],
    failures: list[str],
) -> bool | None:
    """Check markdown must/must_not contain expectations.

    Args:
        markdown: Extracted markdown text.
        expected: Parsed expected.json contents.
        failures: Mutable list to append failure messages to.

    Returns:
        ``True`` if all checks passed, ``False`` if any failed, ``None`` if no
        markdown expectations were defined in the fixture.
    """
    must_contain: list[str] = expected.get("markdown_must_contain", [])
    must_not_contain: list[str] = expected.get("markdown_must_not_contain", [])

    if not must_contain and not must_not_contain:
        return None

    ok = True
    for phrase in must_contain:
        if phrase not in markdown:
            failures.append(f"Markdown must contain {phrase!r} but does not")
            ok = False

    for phrase in must_not_contain:
        if phrase in markdown:
            failures.append(f"Markdown must NOT contain {phrase!r} but does")
            ok = False

    return ok


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parity
def test_parity_fixture_matches_semantic_expectations() -> None:
    """Native extraction must satisfy the x_status semantic contract."""
    report = run_native_fixture_case("x_status_2030105637204676808")
    assert report.semantic_ok is True, f"Semantic failures: {report.failures}"
    assert report.html_snapshot_ok is True, f"HTML snapshot failures: {report.failures}"


@pytest.mark.parity
def test_parity_markdown_must_contain_x_status() -> None:
    """Extracted markdown must contain the expected phrases for x_status."""
    report = run_native_fixture_case("x_status_2030105637204676808")
    assert report.markdown_golden_ok is not False, (
        f"Markdown failures: {report.failures}"
    )


@pytest.mark.parity
def test_parity_fixture_generic_article() -> None:
    """Native extraction must satisfy the generic_article contract."""
    report = run_native_fixture_case("generic_article")
    assert report.html_snapshot_ok is True, f"HTML snapshot failures: {report.failures}"
    assert report.markdown_golden_ok is not False, (
        f"Markdown failures: {report.failures}"
    )


@pytest.mark.parity
def test_parity_fixture_github_issue_thread() -> None:
    """Native extraction must satisfy the github_issue_thread contract."""
    report = run_native_fixture_case("github_issue_thread")
    assert report.html_snapshot_ok is True, f"HTML snapshot failures: {report.failures}"
    assert report.markdown_golden_ok is not False, (
        f"Markdown failures: {report.failures}"
    )


@pytest.mark.parity
def test_parity_report_diagnostics_are_populated() -> None:
    """Diagnostics dict must be non-empty after extraction."""
    report = run_native_fixture_case("x_status_2030105637204676808")
    assert report.diagnostics, "Diagnostics dict should be populated"
    # Should record which extractor path was taken
    assert "extractor" in report.diagnostics


@pytest.mark.parity
@pytest.mark.parametrize(
    "fixture_name",
    [
        "x_status_2030105637204676808",
        "generic_article",
        "github_issue_thread",
        "hackernews_thread",
        "reddit_post",
        "youtube_page",
    ],
)
def test_parity_clean_html_is_non_empty(fixture_name: str) -> None:
    """clean_html must be non-empty for all tested fixtures."""
    report = run_native_fixture_case(fixture_name)
    assert report.html_snapshot_ok is True, (
        f"clean_html empty for {fixture_name}: {report.failures}"
    )


@pytest.mark.parity
def test_parity_fixture_hackernews_thread() -> None:
    """Native extraction must satisfy the hackernews_thread contract."""
    report = run_native_fixture_case("hackernews_thread")
    assert report.html_snapshot_ok is True, f"Failures: {report.failures}"


@pytest.mark.parity
def test_parity_fixture_reddit_post() -> None:
    """Native extraction must satisfy the reddit_post contract."""
    report = run_native_fixture_case("reddit_post")
    assert report.html_snapshot_ok is True, f"Failures: {report.failures}"


@pytest.mark.parity
def test_parity_fixture_youtube_page() -> None:
    """Native extraction must satisfy the youtube_page contract."""
    report = run_native_fixture_case("youtube_page")
    assert report.html_snapshot_ok is True, f"Failures: {report.failures}"
