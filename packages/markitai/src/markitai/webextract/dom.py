from __future__ import annotations

from importlib.util import find_spec

from bs4 import BeautifulSoup, Tag

from markitai.webextract.preprocess import preprocess_html


def default_parser() -> str:
    """Best available BeautifulSoup parser (lxml when installed)."""
    return "lxml" if find_spec("lxml") is not None else "html.parser"


def attr_str(el: Tag, name: str) -> str:
    """One attribute as a string, whether BeautifulSoup returns one or a list.

    Class-like attributes come back as a list, everything else as a string,
    and a missing attribute as None. Three modules each grew their own copy
    of this before it lived anywhere shared.
    """
    value = el.get(name)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(value)


def parse_fragment(html: str) -> BeautifulSoup:
    """Parse an HTML fragment, preferring lxml for speed.

    Unlike :func:`parse_html` this applies no preprocessing — callers are
    mid-pipeline over already-cleaned HTML. lxml wraps fragments in
    ``<html><body>`` and relocates head-only tags into ``<head>``; the
    body's children are hoisted into a bare tree so the result serializes
    exactly like an ``html.parser`` fragment parse. If lxml did route
    content into ``<head>`` (shouldn't happen for extracted article HTML),
    fall back to ``html.parser`` rather than drop it.

    Args:
        html: HTML fragment (no ``<html>``/``<body>`` wrapper expected).

    Returns:
        Parsed BeautifulSoup fragment tree.
    """
    parser = default_parser()
    if parser == "html.parser":
        return BeautifulSoup(html, "html.parser")
    soup = BeautifulSoup(html, "lxml")
    head = soup.head
    if soup.body is None or (head is not None and head.contents):
        return BeautifulSoup(html, "html.parser")
    fragment = BeautifulSoup("", "html.parser")
    for child in list(soup.body.contents):
        fragment.append(child)
    return fragment


def parse_html(html: str) -> BeautifulSoup:
    """Parse HTML using the best available parser.

    Applies raw HTML preprocessing (shadow DOM flattening, ``<wbr>`` removal,
    ``<noscript>`` promotion) before handing the markup to BeautifulSoup,
    then resolves lazy-loading ``<noscript>`` image fallbacks in the parsed
    tree (mirrors defuddle, which runs ``_resolveNoscriptImages`` on the
    live document before any extraction path).

    Args:
        html: Raw HTML content.

    Returns:
        Parsed BeautifulSoup document.
    """
    html = preprocess_html(html)
    soup = BeautifulSoup(html, default_parser())
    resolve_noscript_images(soup)
    return soup


def resolve_noscript_images(soup: BeautifulSoup) -> None:
    """Swap lazy-load placeholder images for their ``<noscript>`` originals.

    Next.js-style lazy images render a ``data:`` placeholder ``<img>`` next
    to a ``<noscript>`` holding the real one. Replace the placeholder's
    ``src``/``srcset`` (matched by alt text), or promote the noscript image
    when no JS-hydrated sibling exists in a lazy-image context. Ported from
    defuddle ``defuddle.ts`` ``_resolveNoscriptImages``.
    """
    for noscript in soup.find_all("noscript"):
        noscript_img = noscript.find("img")
        if noscript_img is None:
            # Some parsers keep noscript content as raw text — re-parse it.
            inner = noscript.get_text()
            if "<img" not in inner:
                continue
            fragment = BeautifulSoup(inner, "html.parser")
            noscript_img = fragment.find("img")
        if not isinstance(noscript_img, Tag):
            continue

        real_src = str(noscript_img.get("src") or "")
        if not real_src or real_src.startswith("data:"):
            continue

        alt = noscript_img.get("alt")
        parent = noscript.parent
        if not isinstance(parent, Tag):
            continue

        matched = False
        for img in parent.find_all("img", recursive=False):
            src = str(img.get("src") or "")
            if not src.startswith("data:"):
                continue
            # Match by alt text; require a non-empty alt to avoid false hits
            if not alt or img.get("alt") != alt:
                continue
            img["src"] = real_src
            srcset = str(noscript_img.get("srcset") or "")
            if srcset:
                img["srcset"] = srcset
            matched = True
            break

        # No matching sibling img — promote the noscript img directly, but
        # only in lazy-loading contexts (not tracking pixels) and only when
        # no JS-hydrated image already exists.
        if not matched and _is_lazy_image_context(noscript):
            container = noscript.find_parent("figure") or parent
            has_real_image = any(
                str(img.get("src") or "")
                and not str(img.get("src") or "").startswith("data:")
                for img in container.find_all("img")
                if img.find_parent("noscript") is None
            )
            if not has_real_image:
                promoted = BeautifulSoup(str(noscript_img), "html.parser").find("img")
                if isinstance(promoted, Tag):
                    noscript.insert_before(promoted)


def _is_lazy_image_context(noscript: Tag) -> bool:
    """Check if a ``<noscript>`` sits in a lazy-image wrapper.

    Distinguishes lazy-loading image fallbacks from standalone tracking
    pixels that must not be promoted.
    """
    if noscript.find_parent("figure") is not None:
        return True
    parent = noscript.parent
    if not isinstance(parent, Tag):
        return False
    for sibling in parent.find_all(True, recursive=False):
        if sibling is noscript:
            continue
        classes = sibling.get("class")
        joined = " ".join(classes).lower() if isinstance(classes, list) else ""
        if "lazy" in joined:
            return True
    parent_classes = parent.get("class")
    parent_cls = (
        " ".join(parent_classes).lower() if isinstance(parent_classes, list) else ""
    )
    return any(t in parent_cls for t in ("image", "img", "picture", "photo", "media"))
