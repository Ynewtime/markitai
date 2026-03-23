from __future__ import annotations

"""Quality parity tests: markitai extraction against defuddle expected outputs.

NOTE: This is a baseline-establishment task, not a TDD cycle. Some tests are
expected to fail initially as we measure how markitai's extraction compares to
defuddle's reference outputs. The pass/fail counts form our starting baseline
that subsequent improvements will be measured against.
"""

import json
import re
from pathlib import Path

import pytest

from markitai.webextract import extract_web_content

_FIXTURE_DIR = Path(__file__).parents[1] / "defuddle_fixtures"
_HTML_DIR = _FIXTURE_DIR / "fixtures"
_EXPECTED_DIR = _FIXTURE_DIR / "expected"


def _discover_fixtures() -> list[str]:
    """Return fixture stems that have both an HTML and expected markdown file."""
    if not _HTML_DIR.is_dir() or not _EXPECTED_DIR.is_dir():
        return []
    html_stems = {p.stem for p in _HTML_DIR.glob("*.html")}
    expected_stems = {p.stem for p in _EXPECTED_DIR.glob("*.md")}
    return sorted(html_stems & expected_stems)


ALL_FIXTURES = _discover_fixtures()


def _extract_og_url(html: str) -> str | None:
    """Extract og:url from raw HTML."""
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


def _extract_canonical_url(html: str) -> str | None:
    """Extract canonical URL from <link rel="canonical"> tag."""
    match = re.search(
        r'<link[^>]+rel=["\']canonical["\'][^>]+href=["\']([^"\']+)["\']',
        html,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)
    match = re.search(
        r'<link[^>]+href=["\']([^"\']+)["\'][^>]+rel=["\']canonical["\']',
        html,
        re.IGNORECASE,
    )
    return match.group(1) if match else None


def _url_from_filename(stem: str) -> str:
    """Construct a plausible URL from a fixture filename.

    Fixture names use ``--`` to separate category from slug, and the slug
    often encodes a domain path using ``-`` separators.
    """
    parts = stem.split("--", 1)
    slug = parts[1] if len(parts) > 1 else parts[0]
    return f"https://example.com/{slug}"


def _infer_url(html: str, stem: str) -> str:
    """Infer the real URL: og:url > canonical > filename fallback."""
    return (
        _extract_og_url(html)
        or _extract_canonical_url(html)
        or _url_from_filename(stem)
    )


def _parse_expected(text: str) -> tuple[dict[str, str], str]:
    """Parse a defuddle expected markdown file.

    Format:
        ```json
        {"title": "...", "author": "...", "site": "...", "published": "..."}
        ```

        [markdown body]

    Returns:
        (metadata_dict, markdown_body)
    """
    metadata: dict[str, str] = {}
    body = text

    match = re.match(r"```json\s*\n(.*?)\n```\s*\n?(.*)", text, re.DOTALL)
    if match:
        try:
            metadata = json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
        body = match.group(2).strip()

    return metadata, body


def _count_words(text: str) -> int:
    """Count words in text, treating each CJK character as one word."""
    count = 0
    non_cjk_buf: list[str] = []

    for ch in text:
        if _is_cjk(ch):
            # Flush buffered non-CJK text
            if non_cjk_buf:
                count += len("".join(non_cjk_buf).split())
                non_cjk_buf.clear()
            count += 1
        else:
            non_cjk_buf.append(ch)

    if non_cjk_buf:
        count += len("".join(non_cjk_buf).split())

    return count


def _is_cjk(ch: str) -> bool:
    """Check if a character is a CJK ideograph."""
    cp = ord(ch)
    return any(
        start <= cp <= end
        for start, end in [
            (0x4E00, 0x9FFF),  # CJK Unified Ideographs
            (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
            (0x20000, 0x2A6DF),  # Extension B
            (0x2A700, 0x2B73F),  # Extension C
            (0x2B740, 0x2B81F),  # Extension D
            (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
            (0x3000, 0x303F),  # CJK Symbols and Punctuation
            (0x3040, 0x309F),  # Hiragana
            (0x30A0, 0x30FF),  # Katakana
            (0xAC00, 0xD7AF),  # Hangul Syllables
        ]
    )


# Noise patterns that indicate site chrome leaked into extraction
_NOISE_PATTERNS = [
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


_extraction_cache: dict[str, tuple[object, dict[str, str], str]] = {}


def _load_and_extract(stem: str) -> tuple[object, dict[str, str], str]:
    """Load fixture, run extraction, parse expected.

    Results are cached so the 4 parametrized tests per fixture share a
    single extraction call instead of repeating it 4 times.
    """
    if stem in _extraction_cache:
        return _extraction_cache[stem]

    html = (_HTML_DIR / f"{stem}.html").read_text(encoding="utf-8")
    expected_text = (_EXPECTED_DIR / f"{stem}.md").read_text(encoding="utf-8")

    url = _infer_url(html, stem)
    result = extract_web_content(html, url)
    expected_meta, expected_body = _parse_expected(expected_text)

    entry = (result, expected_meta, expected_body)
    _extraction_cache[stem] = entry
    return entry


@pytest.mark.skipif(not ALL_FIXTURES, reason="No defuddle fixtures found")
class TestDefuddleParityQuality:
    """Quality parity tests comparing markitai output to defuddle expected outputs.

    NOTE: This is a baseline-establishment task. Some tests are expected to fail
    initially. The pass/fail ratio establishes the starting point for iterative
    improvement.
    """

    @pytest.mark.parametrize("fixture", ALL_FIXTURES)
    def test_content_is_not_empty(self, fixture: str) -> None:
        result, _, _ = _load_and_extract(fixture)
        assert result.markdown and result.markdown.strip(), (
            f"Extraction produced empty markdown for {fixture}"
        )

    @pytest.mark.parametrize("fixture", ALL_FIXTURES)
    def test_metadata_title_extracted(self, fixture: str) -> None:
        result, expected_meta, _ = _load_and_extract(fixture)
        expected_title = expected_meta.get("title", "")
        if not expected_title:
            pytest.skip("Defuddle expected has no title for this fixture")
        assert result.metadata and result.metadata.title, (
            f"Expected title '{expected_title}' but got no title for {fixture}"
        )

    @pytest.mark.parametrize("fixture", ALL_FIXTURES)
    def test_no_site_chrome_noise(self, fixture: str) -> None:
        result, _, _ = _load_and_extract(fixture)
        md = result.markdown or ""
        found = [p for p in _NOISE_PATTERNS if p in md]
        assert not found, f"Site chrome noise found in {fixture}: {found}"

    @pytest.mark.parametrize("fixture", ALL_FIXTURES)
    def test_word_count_within_tolerance(self, fixture: str) -> None:
        result, _, expected_body = _load_and_extract(fixture)
        expected_wc = _count_words(expected_body)
        if expected_wc == 0:
            pytest.skip("Defuddle expected body is empty")

        actual_wc = _count_words(result.markdown or "")
        ratio = actual_wc / expected_wc
        assert 0.5 <= ratio <= 2.0, (
            f"Word count ratio {ratio:.2f} out of tolerance for {fixture}: "
            f"actual={actual_wc}, expected={expected_wc}"
        )
