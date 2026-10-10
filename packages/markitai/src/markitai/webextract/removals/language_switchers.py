"""Language switchers: the page's other-language editions as a list of links.

MediaWiki's Vector skin (Wikipedia, Wiktionary, Wikivoyage, ArchWiki) puts a
dropdown of the article in every other language next to the title, inside
the content root; it came out as ``[ ]`` (the dropdown's checkbox) and a
bulleted list of up to hundreds of language names before the article.

A switcher is recognised by HTML's own marker for "this link is the page in
another language", ``hreflang``, not by a site's class names: a list of at
least three links, nearly all with ``hreflang``, whose texts are short
(language names) and which holds little else. The removal takes the
smallest wrapper that adds nothing but the switcher's own controls (a
checkbox, a "33 languages" label, an "Edit links" link) and stops at
anything with a heading, media, a table, a paragraph or more links. A list
that is most of the page (a portal listing its languages) is kept.
"""

from __future__ import annotations

from bs4 import Tag

from markitai.webextract.utils import count_words

_MIN_LINKS = 3
_LINK_SHARE = 0.8
# Language names: "English", "Norsk bokmål", "Srpskohrvatski / српскохрватски"
_MAX_NAME_WORDS = 4
# Text in the list besides its links ("Languages:", the current language)
_MAX_LABEL_WORDS = 3
# What a wrapper may add around the list: "33 languages", "Edit links"
_MAX_WRAPPER_WORDS = 8
_MAX_WRAPPER_LINKS = 2
_NEVER_REMOVE = frozenset({"html", "body", "main", "article", "[document]"})
_CONTENT_TAGS = [
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "p",
    "img",
    "picture",
    "video",
    "audio",
    "table",
    "pre",
    "blockquote",
    "figure",
]


def remove_language_switchers(root: Tag) -> int:
    """Remove language switcher blocks from *root*.

    Returns:
        Number of blocks removed.
    """
    links = [a for a in root.find_all("a", hreflang=True) if a.get("href")]
    if len(links) < _MIN_LINKS:
        return 0
    removed = 0
    checked: set[int] = set()
    root_words: int | None = None
    for link in links:
        if link.__dict__.get("_decomposed") or link.parent is None:
            continue
        candidate = _list_around(link, root)
        if candidate is None or id(candidate) in checked:
            continue
        checked.add(id(candidate))
        if not _is_language_list(candidate):
            continue
        block = _switcher_block(candidate, root)
        if root_words is None:
            root_words = _words(root)
        if _words(block) * 2 > root_words:
            continue  # the list is the page
        block.decompose()
        removed += 1
    return removed


def _words(el: Tag) -> int:
    return count_words(el.get_text(" ", strip=True))


def _links(el: Tag) -> list[Tag]:
    return [a for a in el.find_all("a") if a.get("href")]


def _list_around(link: Tag, root: Tag) -> Tag | None:
    """The nearest ancestor below *root* holding at least three hreflang links."""
    parent = link.parent
    while isinstance(parent, Tag) and parent is not root:
        if len(parent.find_all("a", hreflang=True, limit=_MIN_LINKS)) >= _MIN_LINKS:
            return parent
        parent = parent.parent
    return None


def _is_language_list(candidate: Tag) -> bool:
    links = _links(candidate)
    language_links = [a for a in links if a.get("hreflang")]
    if len(language_links) < _MIN_LINKS:
        return False
    if len(language_links) < _LINK_SHARE * len(links):
        return False
    name_words = sorted(_words(a) for a in language_links)
    if name_words[len(name_words) // 2] > _MAX_NAME_WORDS:
        return False  # titles of articles, not names of languages
    link_words = sum(_words(a) for a in links)
    return _words(candidate) - link_words <= _MAX_LABEL_WORDS


def _switcher_block(candidate: Tag, root: Tag) -> Tag:
    """The largest wrapper of *candidate* that holds only the switcher."""
    base_words = _words(candidate)
    base_links = len(_links(candidate))
    block = candidate
    parent = block.parent
    while (
        isinstance(parent, Tag)
        and parent is not root
        and parent.name not in _NEVER_REMOVE
    ):
        if any(not _inside(el, candidate) for el in parent.find_all(_CONTENT_TAGS)):
            break
        if _words(parent) - base_words > _MAX_WRAPPER_WORDS:
            break
        if len(_links(parent)) - base_links > _MAX_WRAPPER_LINKS:
            break
        block = parent
        parent = parent.parent
    return block


def _inside(el: Tag, container: Tag) -> bool:
    return any(parent is container for parent in el.parents)
