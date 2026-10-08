"""Running header/footer removal across page and slide boundaries.

Used by the PDF converter on its per-page text and by the shared markdown
cleanup on every output that carries page or slide markers.
"""

from __future__ import annotations

import math
import re

# Documents with fewer pages are exempt: too few samples to call a line
# "running" chrome with confidence.
_HEADER_FOOTER_MIN_DOC_PAGES = 4
# A normalized boundary line must appear on >= this fraction of pages...
_HEADER_FOOTER_MIN_FRACTION = 0.6
# ...and on >= this many pages in absolute terms.
_HEADER_FOOTER_MIN_PAGES = 3
# Number of leading/trailing non-empty lines per page examined as candidates.
_BOUNDARY_LINE_COUNT = 2

_DIGIT_RUN_RE = re.compile(r"\d+")
_WS_RUN_RE = re.compile(r"\s+")


def normalize_boundary_line(line: str) -> str:
    """Normalize a line for cross-page header/footer matching.

    Lowercases, collapses whitespace, and replaces every digit run with
    ``#`` so ``Page 1 of 6`` and ``Page 2 of 6`` collapse to the same key.

    Args:
        line: Raw line text from a page boundary

    Returns:
        Normalized key used to match repeated lines across pages
    """
    collapsed = _WS_RUN_RE.sub(" ", line).strip().lower()
    return _DIGIT_RUN_RE.sub("#", collapsed)


def _boundary_line_indices(lines: list[str]) -> list[int]:
    """Indices of the first/last ``_BOUNDARY_LINE_COUNT`` non-empty lines."""
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    head = non_empty[:_BOUNDARY_LINE_COUNT]
    tail = non_empty[-_BOUNDARY_LINE_COUNT:]
    return sorted(set(head + tail))


def _is_protected_line(line: str) -> bool:
    """Markdown headings and table rows are never stripped as chrome."""
    stripped = line.lstrip()
    return stripped.startswith(("#", "|"))


def strip_repeated_page_lines(
    page_texts: list[str],
) -> tuple[list[str], set[str]]:
    """Strip running headers/footers repeated across page boundaries.

    For each page the first/last two non-empty lines are candidates. A
    candidate whose normalized form (see :func:`normalize_boundary_line`)
    appears on >=60% of pages and on at least 3 pages is a running
    header/footer and is removed from every page's boundary. Markdown
    headings (``#``) and table rows (``|``) are never removed, and
    documents with fewer than 4 pages are exempt entirely.

    Args:
        page_texts: Per-page markdown text, in page order

    Returns:
        Tuple of (page texts with repeated lines removed, set of
        normalized lines that were stripped)
    """
    if len(page_texts) < _HEADER_FOOTER_MIN_DOC_PAGES:
        return page_texts, set()

    pages_lines = [text.splitlines() for text in page_texts]

    counts: dict[str, int] = {}
    for lines in pages_lines:
        seen: set[str] = set()
        for idx in _boundary_line_indices(lines):
            if _is_protected_line(lines[idx]):
                continue
            norm = normalize_boundary_line(lines[idx])
            if norm:
                seen.add(norm)
        for norm in seen:
            counts[norm] = counts.get(norm, 0) + 1

    threshold = max(
        _HEADER_FOOTER_MIN_PAGES,
        math.ceil(len(page_texts) * _HEADER_FOOTER_MIN_FRACTION),
    )
    repeated = {norm for norm, count in counts.items() if count >= threshold}
    if not repeated:
        return page_texts, set()

    result: list[str] = []
    stripped: set[str] = set()
    for page_index, lines in enumerate(pages_lines):
        drop: set[int] = set()
        for idx in _boundary_line_indices(lines):
            if _is_protected_line(lines[idx]):
                continue
            norm = normalize_boundary_line(lines[idx])
            if norm in repeated:
                drop.add(idx)
                stripped.add(norm)
        if drop:
            result.append(
                "\n".join(line for i, line in enumerate(lines) if i not in drop)
            )
        else:
            result.append(page_texts[page_index])
    return result, stripped


_PAGE_MARKER_RE = re.compile(r"(<!-- (?:Page|Slide) (?:number: ?)?\d+ -->)")
# Rounds of edge stripping for marker-delimited pages. A slide footer can be
# three lines (date, footer text, slide number) while one round looks at two
# lines per edge, so a second round takes what the first one uncovered.
_MARKER_STRIP_ROUNDS = 2


def strip_repeated_marker_lines(content: str) -> str:
    """Apply :func:`strip_repeated_page_lines` to marker-delimited pages.

    The blocks after each ``<!-- Page number: N -->`` / ``<!-- Slide number:
    N -->`` marker are the pages; text before the first marker is left alone.
    """
    parts = _PAGE_MARKER_RE.split(content)
    original = parts[2::2]
    pages = original
    for _ in range(_MARKER_STRIP_ROUNDS):
        pages, stripped = strip_repeated_page_lines(pages)
        if not stripped:
            break
    for i, (old, new) in enumerate(zip(original, pages, strict=True)):
        if new is not old:
            # Keep the blank lines that separate a page from the next marker.
            parts[2 + 2 * i] = new.rstrip("\n") + old[len(old.rstrip("\n")) :]
    return "".join(parts)
