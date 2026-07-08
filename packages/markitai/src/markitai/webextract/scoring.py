"""Main-content candidate selection, ported from defuddle.

``select_best_candidate`` mirrors defuddle's ``findMainContent``: score all
matches of an ordered entry-point selector list (earlier selectors carry a
higher base weight), prefer the deepest high-priority child over a wrapper
that only wins on sibling noise, detect listing pages, and fall back to
block-element scoring and old-style table layouts.

``score_candidate`` mirrors ``ContentScorer.scoreElement``.
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, Tag

from markitai.webextract.constants import (
    FOOTNOTE_INLINE_REFERENCES,
    FOOTNOTE_LIST_SELECTORS,
)
from markitai.webextract.utils import count_words

# Ordered entry-point selectors (defuddle ENTRY_POINT_ELEMENTS). Earlier
# entries get a higher base score; 'body' ensures there is always a match.
ENTRY_POINT_SELECTORS: tuple[str, ...] = (
    "#post",
    ".post-content",
    ".post-body",
    ".article-content",
    "#article-content",
    ".js-article-content",
    ".article_post",
    ".article-wrapper",
    ".entry-content",
    ".content-article",
    ".instapaper_body",
    ".post",
    ".markdown-body",
    "article",
    '[role="article"]',
    "main",
    '[role="main"]',
    ".article-body",
    "#content",
    "body",
)

_BLOCK_ELEMENTS = (
    "div",
    "section",
    "article",
    "main",
    "aside",
    "header",
    "footer",
    "nav",
)

# No leading \b — text can concatenate adjacent elements without whitespace.
_DATE_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*)",
    re.IGNORECASE,
)
# Case-sensitive "By" + capitalized name.
_BYLINE_RE = re.compile(r"\bBy\s+[A-Z]")

_CONTENT_CLASS_TOKENS = ("content", "article", "post")


def _class_str(el: Tag) -> str:
    raw = el.get("class")
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw
    return " ".join(raw)


def _attr_str(el: Tag, name: str) -> str:
    value = el.get(name)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(value)


def _is_table_layout(table: Tag) -> bool:
    """Old-style content-layout table: wide, centered, or content-classed."""
    try:
        width = int(_attr_str(table, "width") or "0")
    except ValueError:
        width = 0
    style = _attr_str(table, "style")
    style_width = 0
    match = re.search(r"width\s*:\s*(\d+)px", style)
    if match:
        style_width = int(match.group(1))
    table_class = _class_str(table).lower()
    if (
        width > 400
        or style_width > 400
        or _attr_str(table, "align") == "center"
        or "content" in table_class
        or "article" in table_class
    ):
        return True
    # Multi-column layout: a row with 2+ cells where at least one has an
    # explicit width attribute.
    for row in table.find_all("tr"):
        cells = [c for c in row.children if isinstance(c, Tag) and c.name == "td"]
        if len(cells) >= 2 and any(c.get("width") for c in cells):
            return True
    return False


def score_candidate(node: Tag) -> float:
    """Score a DOM node as likely main content.

    Port of defuddle ``ContentScorer.scoreElement``: text density,
    paragraph count, commas, image-density penalty, position bonus,
    date/author indicators, content class tokens, footnote presence,
    nested-table penalty, table-cell bonus, and a multiplicative link
    density scale (capped at 0.5).
    """
    text = node.get_text()
    words = count_words(text)
    score = float(words)

    paragraphs = len(node.find_all("p"))
    score += paragraphs * 10

    score += text.count(",")

    images = len(node.find_all("img"))
    image_density = images / (words or 1)
    score -= image_density * 3

    # Position bonus (center/right elements)
    style = _attr_str(node, "style")
    align = _attr_str(node, "align")
    if "float: right" in style or "text-align: right" in style or align == "right":
        score += 5

    if _DATE_RE.search(text):
        score += 10
    if _BYLINE_RE.search(text):
        score += 10

    class_name = _class_str(node).lower()
    if any(token in class_name for token in _CONTENT_CLASS_TOKENS):
        score += 15

    if node.select_one(FOOTNOTE_INLINE_REFERENCES) is not None:
        score += 10
    if node.select_one(FOOTNOTE_LIST_SELECTORS) is not None:
        score += 10

    nested_tables = len(node.find_all("table"))
    score -= nested_tables * 5

    # Table cells in old-style content-layout tables get a center-cell bonus
    if node.name == "td":
        parent_table = node.find_parent("table")
        if parent_table is not None and _is_table_layout(parent_table):
            all_cells = parent_table.find_all("td")
            try:
                cell_index = all_cells.index(node)
            except ValueError:
                cell_index = -1
            if 0 < cell_index < len(all_cells) - 1:
                score += 10

    # Link density as a multiplier — scales the score down proportionally,
    # capped at 0.5 to avoid over-penalizing link-heavy content.
    link_text_len = sum(len(a.get_text()) for a in node.find_all("a"))
    link_density = min(link_text_len / (len(text) or 1), 0.5)
    score *= 1 - link_density

    return score


def _find_content_by_scoring(soup: BeautifulSoup) -> Tag | None:
    """Score all block elements and return the best positive scorer."""
    best: Tag | None = None
    best_score = 0.0
    for el in soup.find_all(list(_BLOCK_ELEMENTS)):
        score = score_candidate(el)
        if score > best_score:
            best = el
            best_score = score
    return best


def _find_table_based_content(soup: BeautifulSoup) -> Tag | None:
    """Detect old-style table layouts and return the best content cell."""
    tables = soup.find_all("table")
    if not any(_is_table_layout(t) for t in tables):
        return None

    best_cell: Tag | None = None
    best_score = 50.0  # findBestElement min score
    for cell in soup.find_all("td"):
        score = score_candidate(cell)
        if score > best_score:
            best_cell = cell
            best_score = score
    if best_cell is None:
        return None

    # If there's more text outside the best cell than inside it, tables are
    # peripheral (TOC, intro boxes, data tables) — not the main container.
    body = soup.body or soup
    if count_words(best_cell.get_text()) * 2 < count_words(body.get_text()):
        return None
    return best_cell


def select_best_candidate(soup: BeautifulSoup) -> Tag | None:
    """Return the main-content element (port of defuddle findMainContent)."""
    candidates: list[tuple[Tag, float, int]] = []
    total = len(ENTRY_POINT_SELECTORS)
    for index, selector in enumerate(ENTRY_POINT_SELECTORS):
        for el in soup.select(selector):
            base = (total - index) * 40
            candidates.append((el, base + score_candidate(el), index))

    if not candidates:
        return _find_content_by_scoring(soup)

    candidates.sort(key=lambda c: -c[1])

    # If we only matched body, try table-based detection
    if len(candidates) == 1 and candidates[0][0].name == "body":
        table_content = _find_table_based_content(soup)
        if table_content is not None:
            return table_content

    # If the top candidate contains a child candidate that matched a
    # higher-priority selector, prefer the most specific (deepest) child —
    # unless the parent holds multiple candidates at that selector index,
    # which indicates a listing/portfolio page.
    top_el, _top_score, _top_index = candidates[0]
    top_words = count_words(top_el.get_text())
    best_el, best_index = top_el, _top_index
    for child_el, _child_score, child_index in candidates[1:]:
        child_words = count_words(child_el.get_text())
        # NOTE: divergence from defuddle — also require the child to hold a
        # meaningful share of the top candidate's text. Defuddle relies on
        # site extractors (e.g. Substack) to catch pages where a small
        # embedded card matches a high-priority selector; without those,
        # a 50-word floor alone promotes the card over the real article.
        if (
            child_index < best_index
            and best_el in child_el.parents
            and child_words > 50
            and child_words * 4 >= top_words
        ):
            siblings_at_index = 0
            for other_el, _s, other_index in candidates:
                if other_index == child_index and (
                    other_el is top_el or top_el in other_el.parents
                ):
                    siblings_at_index += 1
                    if siblings_at_index > 1:
                        break
            if siblings_at_index > 1:
                continue
            best_el, best_index = child_el, child_index

    return best_el
