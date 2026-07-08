"""Locate where the prose body starts inside the main content element.

Ported from defuddle ``content-boundary.ts``. ``find_content_start`` returns
a DOM-shape proxy for "here is where the article actually begins", which
pre-content removal heuristics use as the authoritative above/below check —
replacing ad-hoc ``content_text.find(text) < N`` byte-offset thresholds.
"""

from __future__ import annotations

import re

from bs4 import Tag

from markitai.webextract.utils import count_words, normalize_text

_DATE_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"|\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    re.IGNORECASE,
)
_BYLINE_RE = re.compile(r"^by\s+\S", re.IGNORECASE)
_SENTENCE_PUNCT_RE = re.compile(r"[.!?]")
_HIDDEN_CLASS_RE = re.compile(r"\b(?:isHidden(?:-[A-Za-z0-9_]+)?|is-hidden)\b")

_CANDIDATE_TAGS = frozenset({"p", "div", "section", "article", "blockquote", "font"})
# Leaf-like candidates are preferred over container candidates so a top-level
# wrapper whose text happens to qualify doesn't outvote the real paragraph.
_LEAF_CANDIDATE_TAGS = frozenset({"p", "blockquote", "font"})

_SKIP_ANCESTOR_TAGS = frozenset({"aside", "nav", "header", "footer", "form"})
_DIALOG_ROLES = frozenset({"dialog", "alertdialog"})
_PROSE_MIN_WORDS = 7


def _find_title_element(main_content: Tag, title: str) -> Tag | None:
    normalized_title = normalize_text(title)
    if not normalized_title:
        return None
    for heading in main_content.find_all(["h1", "h2"]):
        if normalize_text(heading.get_text()) == normalized_title:
            return heading
    return None


def _link_text_length(el: Tag) -> int:
    return sum(len(a.get_text()) for a in el.find_all("a"))


def _has_skip_ancestor(el: Tag) -> bool:
    for parent in el.parents:
        if not isinstance(parent, Tag):
            continue
        if parent.name in _SKIP_ANCESTOR_TAGS:
            return True
        if parent.get("role") in _DIALOG_ROLES:
            return True
    return False


def _is_prose_block(el: Tag) -> bool:
    if el.name not in _CANDIDATE_TAGS:
        return False
    if _has_skip_ancestor(el):
        return False
    class_attr = " ".join(el.get("class") or [])
    if _HIDDEN_CLASS_RE.search(class_attr):
        return False
    # A dialog inside the subtree pollutes the element's text with modal
    # copy; a script/style descendant pollutes it with source code.
    if el.select_one('[role="dialog"], [role="alertdialog"]') is not None:
        return False
    if el.find(["script", "style"]) is not None:
        return False

    text = el.get_text().strip()
    if not text:
        return False
    words = count_words(text)
    if words < _PROSE_MIN_WORDS:
        return False
    if not _SENTENCE_PUNCT_RE.search(text):
        return False
    # "By Jane Smith, reporter" — a short text starting with "By" is a byline.
    # Real prose that happens to start with "By" tends to be longer.
    if _BYLINE_RE.search(text) and words < 15:
        return False
    if _DATE_RE.search(text) and words < 20:
        return False
    if _link_text_length(el) > len(text) * 0.7:
        return False
    # A DIV whose text comes from spans/labels (cookie banners, button
    # groups) can pass the word-count bar without containing article prose.
    if el.name == "div" and el.find("p") is None:
        return False

    return True


def find_content_start(main_content: Tag, title: str) -> Tag | None:
    """Best candidate for "this element is the start of the prose body".

    Anchor on the title (h1/h2 matching the normalized page title) if
    present, then walk forward in document order for the first prose-length
    block. Returns ``None`` when no candidate qualifies; callers should
    treat that as "no signal" rather than a removal opportunity.
    """
    title_el = _find_title_element(main_content, title)

    # Single tree walk that records the first leaf-candidate hit and,
    # separately, the first container-candidate hit. Prefer the leaf — a
    # top-level wrapper would otherwise outvote a real paragraph inside it.
    started = title_el is None
    leaf_hit: Tag | None = None
    container_hit: Tag | None = None
    for node in main_content.descendants:
        if not isinstance(node, Tag):
            continue
        if not started:
            if node is title_el:
                started = True
            continue
        if _is_prose_block(node):
            if node.name in _LEAF_CANDIDATE_TAGS:
                leaf_hit = node
                break
            if container_hit is None:
                container_hit = node

    if leaf_hit is not None:
        return leaf_hit
    if container_hit is not None:
        # Drill down through wrapper containers to find the most specific
        # qualifying block. At each level, if exactly one child qualifies,
        # descend into it. Stop when multiple children qualify (the content
        # area has been reached) or none do.
        result = container_hit
        while True:
            qualifying_child: Tag | None = None
            multiple = False
            for child in result.children:
                if isinstance(child, Tag) and _is_prose_block(child):
                    if qualifying_child is not None:
                        multiple = True
                        break
                    qualifying_child = child
            if qualifying_child is not None and not multiple:
                result = qualifying_child
            else:
                break
        return result

    # If we anchored on the title and found nothing after it, retry from top.
    if title_el is not None:
        return find_content_start(main_content, "")
    return None


def element_precedes(a: Tag, b: Tag) -> bool:
    """True when ``a`` strictly precedes ``b`` in document order.

    Containment counts as preceding (a containing b → True), matching DOM
    ``compareDocumentPosition`` FOLLOWING semantics for this use case.
    Detached elements compare False.
    """
    if a is b:
        return False
    if a.parent is None or b.parent is None:
        return False
    a_chain: list[Tag] = [a]
    for parent in a.parents:
        if isinstance(parent, Tag):
            a_chain.append(parent)
    a_index = {id(node): i for i, node in enumerate(a_chain)}
    if id(b) in a_index:
        return False  # b contains a
    prev = b
    for anc in b.parents:
        if not isinstance(anc, Tag):
            continue
        i = a_index.get(id(anc))
        if i is not None:
            if i == 0:
                return True  # a contains b
            a_branch = a_chain[i - 1]
            for child in anc.children:
                if child is a_branch:
                    return True
                if child is prev:
                    return False
            return False
        prev = anc
    return False  # disconnected trees


def is_above_content_start(el: Tag, boundary: Tag | None) -> bool:
    """True when ``el`` sits before ``boundary`` in document order.

    Safe when ``boundary`` is None (returns False — treat as "don't know").
    """
    if boundary is None:
        return False
    return element_precedes(el, boundary)
