"""Remove elements matching exact and partial selectors."""

from __future__ import annotations

from bs4 import Tag

from markitai.webextract.constants import (
    EXACT_SELECTORS,
    PARTIAL_SELECTOR_REGEX,
    TEST_ATTRIBUTES,
)


def remove_by_selectors(
    root: Tag,
    main_content: Tag | None,
    *,
    use_partial: bool = True,
) -> int:
    """Remove elements matching known non-content selectors.

    Args:
        root: Content root element.
        main_content: Main content element (protected from removal).
        use_partial: Whether to also use partial attribute matching.

    Returns:
        Number of elements removed.
    """
    to_remove: list[Tag] = []

    # Phase 1: Exact CSS selectors
    for selector in EXACT_SELECTORS:
        try:
            for el in root.select(selector):
                if _should_protect(el, main_content):
                    continue
                to_remove.append(el)
        except Exception:  # noqa: BLE001
            # Some selectors may not be supported by BeautifulSoup
            continue

    # Phase 2: Partial attribute matching
    if use_partial:
        for el in root.find_all(True):
            if el in to_remove:
                continue
            if _should_protect(el, main_content):
                continue
            if _matches_partial(el):
                to_remove.append(el)

    # Remove in reverse document order to avoid parent-before-child issues
    removed = 0
    decomposed: set[int] = set()
    for el in to_remove:
        eid = id(el)
        if eid in decomposed:
            continue
        # Check if any ancestor was already decomposed
        if any(id(p) in decomposed for p in el.parents if isinstance(p, Tag)):
            continue
        el.decompose()
        decomposed.add(eid)
        removed += 1

    return removed


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

    return False


def _is_ancestor_of(ancestor: Tag, descendant: Tag) -> bool:
    """Check if ancestor contains descendant."""
    return any(parent is ancestor for parent in descendant.parents)


def _matches_partial(el: Tag) -> bool:
    """Check if element matches any partial selector pattern."""
    attrs = el.attrs
    if not attrs:
        return False

    for attr_name in TEST_ATTRIBUTES:
        value = attrs.get(attr_name)
        if value is None:
            continue
        # class is a list in BeautifulSoup
        if isinstance(value, list):
            text = " ".join(value)
        else:
            text = str(value)
        if text and PARTIAL_SELECTOR_REGEX.search(text):
            return True
    return False
