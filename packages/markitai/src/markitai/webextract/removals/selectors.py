"""Remove elements matching exact and partial selectors."""

from __future__ import annotations

import re
from functools import lru_cache

from bs4 import Tag

from markitai.webextract.constants import (
    EXACT_SELECTORS,
    EXACT_SELECTORS_JOINED,
    FOOTNOTE_LIST_SELECTORS,
    HIDDEN_EXACT_SELECTOR,
    HIDDEN_EXACT_SKIP_SELECTOR,
    PARTIAL_SELECTOR_ANCHORED_REGEX,
    PARTIAL_SELECTOR_REGEX,
    TEST_ATTRIBUTES,
)
from markitai.webextract.selectors import matches as _css_matches
from markitai.webextract.selectors import select
from markitai.webextract.selectors import select_one as _css_select_one
from markitai.webextract.utils import count_words, has_responsive_show_class

_ID_DELIMITER_RE = re.compile(r"[\s_\-:.]")
_HEADING_NAMES = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_TABLE_PARTS = frozenset({"thead", "tbody", "tfoot", "tr", "th", "td"})


def remove_by_selectors(
    root: Tag,
    main_content: Tag | None,
    *,
    use_partial: bool = True,
    skip_hidden_exact: bool = False,
) -> int:
    """Remove elements matching known non-content selectors.

    Args:
        root: Content root element.
        main_content: Main content element (protected from removal).
        use_partial: Whether to also use partial attribute matching.
        skip_hidden_exact: Keep elements matched only by the hidden-element
            exact selectors, plus dialogs inside hidden subtrees. Set on
            retries with hidden-element removal disabled so pages that
            reveal an ``aria-hidden`` overlay at runtime keep their content
            (defuddle issue 232).

    Returns:
        Number of elements removed.
    """
    to_remove: list[Tag] = []
    seen_ids: set[int] = set()

    # Phase 1: Exact CSS selectors (single joined query for performance)
    try:
        for el in select(root, EXACT_SELECTORS_JOINED):
            eid = id(el)
            if eid in seen_ids:
                continue
            if _skip_hidden_match(el, skip_hidden_exact):
                continue
            if _should_protect(el, main_content):
                continue
            if _is_article_aside(el):
                continue
            to_remove.append(el)
            seen_ids.add(eid)
    except Exception:
        # Fallback: query individually if joined selector fails
        for selector in EXACT_SELECTORS:
            try:
                for el in root.select(selector):
                    eid = id(el)
                    if eid in seen_ids:
                        continue
                    if _skip_hidden_match(el, skip_hidden_exact):
                        continue
                    if _should_protect(el, main_content):
                        continue
                    if _is_article_aside(el):
                        continue
                    to_remove.append(el)
                    seen_ids.add(eid)
            except Exception:
                continue

    # Phase 2: Partial attribute matching
    if use_partial:
        for el in root.find_all(True):
            eid = id(el)
            if eid in seen_ids:
                continue
            # Like defuddle, collect actual attribute matches first. Protection
            # checks walk ancestors/subtrees; running them for every DOM node
            # is especially expensive on math and syntax-highlighted pages.
            if not _matches_partial(el):
                continue
            if _should_protect(el, main_content):
                continue
            if not _protected_for_partial(el):
                to_remove.append(el)
                seen_ids.add(eid)

    # Remove in reverse document order to avoid parent-before-child issues
    removed = 0
    decomposed: set[int] = set()
    abstract = _article_abstract(root) if to_remove else None
    for el in to_remove:
        eid = id(el)
        if eid in decomposed:
            continue
        # Check if any ancestor was already decomposed
        if any(id(p) in decomposed for p in el.parents if isinstance(p, Tag)):
            continue
        if abstract is not None and any(p is el for p in abstract.parents):
            # A removed block never takes the article's abstract with it
            # (martinfowler.com's "frontMatter": abstract, date, author bios)
            el.replace_with(abstract.extract())
            abstract = None
        else:
            el.decompose()
        decomposed.add(eid)
        removed += 1

    return removed


_ABSTRACT_SELECTOR = '[class~="abstract"], [itemprop="abstract"], [role="doc-abstract"]'


def _prose_words(el: Tag) -> tuple[int, int]:
    """(words, words inside links) of an element's text."""
    words = count_words(el.get_text(" ", strip=True))
    link_words = sum(count_words(a.get_text(" ", strip=True)) for a in el.find_all("a"))
    return words, link_words


def _article_abstract(root: Tag) -> Tag | None:
    """The first element marked as the abstract that reads as one.

    Marked by class ``abstract``, ``itemprop="abstract"`` or
    ``role="doc-abstract"``; prose (at least 15 words, few of them links).
    Only the first: later ones are more likely other articles' teasers.
    """
    for candidate in select(root, _ABSTRACT_SELECTOR):
        words, link_words = _prose_words(candidate)
        if words >= 15 and link_words <= 0.3 * words:
            return candidate
        return None
    return None


def _is_article_aside(el: Tag) -> bool:
    """Whether an aside is a pull-out box of the article, not page chrome.

    The aside and sidebar selectors are for a page's side column, which
    stands beside the article and is mostly links. An aside set among the
    article's own paragraphs and written as prose belongs to the article
    (martinfowler.com's sidebars: "How big is a microservice?", ...).
    """
    classes = el.get("class")
    class_list = classes if isinstance(classes, list) else str(classes or "").split()
    if el.name != "aside" and "sidebar" not in (c.lower() for c in class_list):
        return False
    parent = el.parent
    if not isinstance(parent, Tag):
        return False
    prose_siblings = sum(
        1
        for child in parent.find_all("p", recursive=False)
        if count_words(child.get_text(" ", strip=True)) >= 15
    )
    if prose_siblings < 2:
        return False
    words, link_words = _prose_words(el)
    return words >= 30 and link_words <= 0.3 * words


def _skip_hidden_match(el: Tag, skip_hidden_exact: bool) -> bool:
    """Check if an exact-selector match must be kept for hidden-content retries.

    Mirrors defuddle ``removeBySelector``: an element with a responsive
    show class (e.g. "hidden sm:flex") is always kept; when
    ``skip_hidden_exact`` is set (hidden-element removal disabled),
    elements matching the hidden exact selectors are kept, as are
    ``role="dialog"`` elements inside a hidden subtree — pages that
    reveal an aria-hidden overlay at runtime keep their article.
    """
    try:
        is_hidden_match = _css_matches(el, HIDDEN_EXACT_SELECTOR)
    except Exception:
        return False
    classes = el.get("class")
    class_str = " ".join(classes) if isinstance(classes, list) else str(classes or "")
    if is_hidden_match and has_responsive_show_class(class_str):
        return True
    if not skip_hidden_exact:
        return False
    if is_hidden_match:
        return True
    role = str(el.get("role") or "").lower()
    if role != "dialog":
        return False
    return any(
        isinstance(p, Tag) and p.name != "[document]" and _matches_hidden_skip(p)
        for p in el.parents
    )


def _matches_hidden(el: Tag) -> bool:
    """Check if element matches the hidden-element exact selectors."""
    try:
        return _css_matches(el, HIDDEN_EXACT_SELECTOR)
    except Exception:
        return False


def _matches_hidden_skip(el: Tag) -> bool:
    """Check if element matches the unguarded hidden selectors."""
    try:
        return _css_matches(el, HIDDEN_EXACT_SKIP_SELECTOR)
    except Exception:
        return False


def _should_protect(el: Tag, main_content: Tag | None) -> bool:
    """Check if an element should be protected from removal."""
    # Protect main content and its ancestors
    if main_content is not None:
        if el is main_content:
            return True
        if _is_ancestor_of(el, main_content):
            return True

    # Protect elements inside <pre> or <code>
    for parent in el.parents:
        if isinstance(parent, Tag) and parent.name in ("pre", "code"):
            return True

    # Protect footnote list containers, their parents/children, and the
    # backref links generated by standardize_footnotes (mirrors defuddle
    # removals/selectors.ts).
    try:
        if _css_matches(el, FOOTNOTE_LIST_SELECTORS):
            return True
        if _css_select_one(el, FOOTNOTE_LIST_SELECTORS) is not None:
            return True
        parent = el.parent
        if (
            isinstance(parent, Tag)
            and parent.name != "[document]"
            and _css_matches(parent, FOOTNOTE_LIST_SELECTORS)
        ):
            return True
        classes = el.get("class")
        if (
            isinstance(classes, list)
            and "footnote-backref" in classes
            and el.find_parent(id="footnotes") is not None
        ):
            return True
    except Exception:
        pass

    # NOTE: divergence from defuddle — the rows and cells of a data table
    # are never page chrome. Pandoc writes <tr class="header"> for every
    # table's head row, which ".header:not(.banner)" removed, leaving the
    # Markdown table with an empty header row. A table with <th> cells is
    # a data table; layout tables (old sites' <td id="header">) have none.
    # Rows hidden by class still go, like any hidden element.
    if el.name in _TABLE_PARTS:
        table = el.find_parent("table")
        if (
            table is not None
            and table.find("th") is not None
            and not _matches_hidden(el)
        ):
            return True

    # Protect <header> elements that are direct children of article/main/section.
    # The bare "header" selector is intended to remove site-level page headers,
    # not article-level headers that contain the h1 and byline.
    if el.name == "header":
        parent = el.parent
        if isinstance(parent, Tag) and parent.name in ("article", "main", "section"):
            return True

    return False


def _is_ancestor_of(ancestor: Tag, descendant: Tag) -> bool:
    """Check if ancestor contains descendant."""
    return any(parent is ancestor for parent in descendant.parents)


def _protected_for_partial(el: Tag) -> bool:
    """Check if a partial-selector match must be kept anyway.

    Partial selectors are fuzzy; defuddle skips them for code content:
    elements that are — or contain — ``<pre>`` blocks (e.g. a code tab
    group whose class happens to contain "dropdown"), and elements inside
    syntax-highlighting wrappers.
    """
    if el.name in ("pre", "code"):
        return True
    if el.find("pre") is not None:
        return True
    for ancestor in el.parents:
        if not isinstance(ancestor, Tag):
            break
        classes = ancestor.get("class")
        if isinstance(classes, list) and any(
            "language-" in c or "syntax-" in c for c in classes
        ):
            return True
    return False


@lru_cache(maxsize=8192)
def _partial_hit(text: str) -> bool:
    """PARTIAL_SELECTOR_REGEX over an attribute value; the same class and
    data-* strings recur on most elements of a page."""
    return PARTIAL_SELECTOR_REGEX.search(text) is not None


def _matches_partial(el: Tag) -> bool:
    """Check if element matches any partial selector pattern.

    Follows defuddle's matching rules:

    - class and ``data-*`` test attributes are substring-matched.
    - Headings only check class — their ids are auto-generated slugs and
      their data-testid values (e.g. "article-header") cause false
      positives.
    - A delimited id (e.g. "feedback-form") is substring-matched like
      class, but a delimiter-less id is usually a content anchor of
      concatenated heading words (e.g. "loopsandfeedback") — substring
      matching would wrongly strip it, so it must equal a selector token
      outright.
    """
    attrs = el.attrs
    if not attrs:
        return False
    is_heading = el.name in _HEADING_NAMES

    for attr_name in TEST_ATTRIBUTES:
        if attr_name == "id":
            continue
        if is_heading and attr_name != "class":
            continue
        value = attrs.get(attr_name)
        if value is None:
            continue
        # class is a list in BeautifulSoup
        if isinstance(value, list):
            text = " ".join(value)
        else:
            text = str(value)
        if text and _partial_hit(text):
            return True

    if is_heading:
        return False
    id_value = attrs.get("id")
    if isinstance(id_value, str) and id_value:
        if _ID_DELIMITER_RE.search(id_value):
            return _partial_hit(id_value)
        return bool(PARTIAL_SELECTOR_ANCHORED_REGEX.match(id_value))
    return False
