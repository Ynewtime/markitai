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
# Longer lines are body text, never running headers/footers.
_BOUNDARY_LINE_MAX_CHARS = 100

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


def _boundary_line_indices(
    lines: list[str], count: int = _BOUNDARY_LINE_COUNT
) -> list[int]:
    """Indices of the first/last *count* non-empty lines."""
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    head = non_empty[:count]
    tail = non_empty[-count:]
    return sorted(set(head + tail))


def _is_protected_line(line: str) -> bool:
    """Markdown headings and table rows are never stripped as chrome."""
    stripped = line.lstrip()
    return stripped.startswith(("#", "|"))


def _candidate_key(line: str) -> str:
    """The key a boundary line is matched by, or "" when it is no candidate:
    a heading, a table row, or a line too long to be a header/footer."""
    if _is_protected_line(line) or len(line.strip()) > _BOUNDARY_LINE_MAX_CHARS:
        return ""
    return normalize_boundary_line(line)


def _edge_runs(lines: list[str], count: int) -> tuple[list[int], list[int]]:
    """The first and the last *count* non-empty lines, each run listed from
    the page's edge inwards."""
    non_empty = [i for i, line in enumerate(lines) if line.strip()]
    return non_empty[:count], non_empty[::-1][:count]


def strip_repeated_page_lines(
    page_texts: list[str],
    *,
    third_line: bool = False,
) -> tuple[list[str], set[str]]:
    """Strip running headers/footers repeated across page boundaries.

    For each page the first/last two non-empty lines are candidates, when
    they are short (at most 100 characters). A candidate whose normalized
    form (see :func:`normalize_boundary_line`) appears on >=60% of pages
    and on at least 3 pages is a running header/footer and is removed from
    every page's boundary, unless it is repeated within its own pages: a
    form found more than once on most of the pages that show it is the
    shape of the body's lines (``Section 3.1 …``, ``Section 3.2 …`` in a
    page of numbered paragraphs), not a header or footer. Markdown headings
    (``#``) and table rows (``|``) are never removed, and documents with
    fewer than 4 pages are exempt entirely.

    With *third_line*, the third line from an edge is a candidate too when
    the two outside it are removed (a slide footer of date, text and number),
    decided in the same pass: nothing is looked at again after a removal,
    so the rule never eats into the body round after round.

    Args:
        page_texts: Per-page markdown text, in page order
        third_line: Also examine the third line inside two removed ones

    Returns:
        Tuple of (page texts with repeated lines removed, set of
        normalized lines that were stripped)
    """
    if len(page_texts) < _HEADER_FOOTER_MIN_DOC_PAGES:
        return page_texts, set()

    pages_lines = [text.splitlines() for text in page_texts]
    pages_keys = [[_candidate_key(line) for line in lines] for lines in pages_lines]
    threshold = max(
        _HEADER_FOOTER_MIN_PAGES,
        math.ceil(len(page_texts) * _HEADER_FOOTER_MIN_FRACTION),
    )

    def running(positions: list[list[int]]) -> set[str]:
        """The keys at *positions* (per page) that pass the page counts."""
        counts: dict[str, int] = {}
        repeats: dict[str, int] = {}
        for keys, indices in zip(pages_keys, positions, strict=True):
            seen = {keys[i] for i in indices if keys[i]}
            for key in seen:
                counts[key] = counts.get(key, 0) + 1
                if keys.count(key) > 1:
                    repeats[key] = repeats.get(key, 0) + 1
        return {
            key
            for key, count in counts.items()
            if count >= threshold and repeats.get(key, 0) * 2 < count
        }

    repeated = running([_boundary_line_indices(lines) for lines in pages_lines])
    if not repeated:
        return page_texts, set()

    # The third line of an edge whose two outer lines both go.
    inner: list[list[int]] = [[] for _ in pages_lines]
    if third_line:
        for page_index, lines in enumerate(pages_lines):
            keys = pages_keys[page_index]
            for run in _edge_runs(lines, _BOUNDARY_LINE_COUNT + 1):
                if len(run) > _BOUNDARY_LINE_COUNT and all(
                    keys[i] in repeated for i in run[:_BOUNDARY_LINE_COUNT]
                ):
                    inner[page_index].append(run[_BOUNDARY_LINE_COUNT])
    repeated_inner = running(inner) if any(inner) else set()

    result: list[str] = []
    stripped: set[str] = set()
    for page_index, lines in enumerate(pages_lines):
        keys = pages_keys[page_index]
        drop: set[int] = set()
        for idx in _boundary_line_indices(lines):
            if keys[idx] in repeated:
                drop.add(idx)
                stripped.add(keys[idx])
        for idx in inner[page_index]:
            if keys[idx] in repeated_inner:
                drop.add(idx)
                stripped.add(keys[idx])
        if drop:
            result.append(
                "\n".join(line for i, line in enumerate(lines) if i not in drop)
            )
        else:
            result.append(page_texts[page_index])
    return result, stripped


_PAGE_MARKER_RE = re.compile(r"(<!-- (?:Page|Slide) (?:number: ?)?\d+ -->)")


def strip_repeated_marker_lines(content: str) -> str:
    """Apply :func:`strip_repeated_page_lines` to marker-delimited pages.

    The blocks after each ``<!-- Page number: N -->`` / ``<!-- Slide number:
    N -->`` marker are the pages; text before the first marker is left alone.
    A slide footer can be three lines (date, footer text, slide number), so
    the third line inside two removed ones is examined too, in the same pass.
    """
    parts = _PAGE_MARKER_RE.split(content)
    original = parts[2::2]
    pages, _ = strip_repeated_page_lines(original, third_line=True)
    for i, (old, new) in enumerate(zip(original, pages, strict=True)):
        if new is not old:
            # Keep the blank lines that separate a page from the next marker.
            parts[2 + 2 * i] = new.rstrip("\n") + old[len(old.rstrip("\n")) :]
    return "".join(parts)
