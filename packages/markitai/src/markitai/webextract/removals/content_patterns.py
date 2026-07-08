"""Remove text-pattern noise: bylines, read-time, related/newsletter/CTA blocks.

Faithful port of defuddle ``removals/content-patterns.ts`` — pattern order,
thresholds, and guards mirror the original. Pre-content checks anchor on
:func:`markitai.webextract.content_boundary.find_content_start` instead of
byte offsets wherever defuddle does.
"""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from bs4 import Tag

from markitai.webextract.content_boundary import (
    element_precedes,
    find_content_start,
    is_above_content_start,
)
from markitai.webextract.utils import count_words, normalize_text

_DATE_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}"
    r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*"
    r"|\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    re.IGNORECASE,
)
_RELATIVE_TIME_RE = re.compile(
    r"\b\d+\s+(?:second|minute|hour|day|week|month|year)s?\s+ago\b", re.IGNORECASE
)
_READ_TIME_RE = re.compile(
    r"\d+\s*min(?:ute)?s?\s+read\b|(?:read(?:ing)?\s+time)\s*:?\s*\d+\s*min(?:ute)?s?\b",
    re.IGNORECASE,
)
_STARTS_WITH_BY_RE = re.compile(r"^(?:posted\s+)?by\s+\S", re.IGNORECASE)
_METADATA_LABEL_RE = re.compile(
    r"^(?:date|published|updated|posted|from|to|subject)\s*:", re.IGNORECASE
)
_BOILERPLATE_PATTERNS = [
    re.compile(p, re.IGNORECASE if flag else 0)
    for p, flag in [
        (
            r"^This (?:article|story|piece) (?:appeared|was published|originally appeared) in\b",
            True,
        ),
        (r"^A version of this (?:article|story) (?:appeared|was published) in\b", True),
        (r"^Originally (?:published|appeared) (?:in|on|at)\b", True),
        (r"^Any re-?use permitted\b", True),
        (r"^©\s*(?:Copyright\s+)?\d{4}", True),
        (r"^Comments?$", True),
        (r"^Leave a (?:comment|reply)$", True),
        (r"^Loading\.{3}$", False),
        (r"^Affiliate links\b.*\b(?:earn|commission)", True),
        (r"\bRead our Comment Policy\b", True),
        (r"^Thank you for (?:being part of|joining) our community\b", True),
    ]
]
_NEWSLETTER_RE = re.compile(
    r"\bsubscribe\b[\s\S]{0,40}\bnewsletter\b"
    r"|\bnewsletter\b[\s\S]{0,40}\bsubscribe\b"
    r"|\bsign[- ]up\b[\s\S]{0,80}\b(?:newsletter|email alert)"
    r"|\b(?:don[’']?t (?:want to )?miss|never miss)\b"
    r"[\s\S]{0,80}\b(?:latest|best|exclusive|reports?|updates?|source)",
    re.IGNORECASE,
)
_SOCIAL_COUNTER_RE = re.compile(
    r"^\d+\s+(?:Likes?|Comments?|Shares?|Retweets?|Reposts?|Restacks?)$", re.IGNORECASE
)
_TIMEZONE_WIDGET_RE = re.compile(r"^current time in$", re.IGNORECASE)
_PINNED_LABEL_RE = re.compile(r"^pinned$", re.IGNORECASE)
_AUTHOR_CONTACT_LABEL_RE = re.compile(
    r"^(?:written by|(?:author|contact|reporter|correspondent)s?)$", re.IGNORECASE
)
_SHARE_AUTHOR_LABEL_RE = re.compile(
    r"^(?:share|follow|authors?|written\s+by)$", re.IGNORECASE
)
_EMAIL_RE = re.compile(r"[\w.-]+@[\w.-]+\.\w+")
_PHONE_RE = re.compile(r"\(?\d{3}\)?[\s.‑–-]?\d{3}[\s.‑–-]?\d{4}")
_RELATED_HEADING_RE = re.compile(
    r"^(?:related (?:posts?|articles?|content|stories|reads?|reading)"
    r"|you (?:might|may|could) (?:also )?(?:like|enjoy|be interested in)"
    r"|read (?:next|more|also)|further reading|see also"
    r"|more (?:from .*|from|articles?|posts?|like this)|more to (?:read|explore)"
    r"|explore more|about (?:the )?author"
    r"|latest (?:news|events?|posts?|articles?|stories)"
    r"(?:\s*[&+]\s*(?:news|events?|posts?|articles?|stories))?)$",
    re.IGNORECASE,
)
# CTA headings that are never real content — safe to remove even as direct children
_CTA_HEADING_RE = re.compile(
    r"^(?:subscribe|sign up|follow us|share this|stay (?:updated|connected)"
    r"|join (?:us|our)|search (?:the |our )?"
    r"(?:site|blog|archives?|newsroom|website|catalog|store|shop|database))$",
    re.IGNORECASE,
)
_RELATED_INTRO_RE = re.compile(r"^for more (?:on|about)\b", re.IGNORECASE)
_TOC_HEADING_RE = re.compile(
    r"^(?:table of )?contents$|^on this page$|^in this (?:article|guide|post)$",
    re.IGNORECASE,
)
_SENTENCE_PUNCT_RE = re.compile(r"[.!?]")
_SENTENCE_PUNCT_SPACE_RE = re.compile(r"[.!?]\s")
_SENTENCE_END_RE = re.compile(r"[.!?]$")
_CAMEL_BOUNDARY_RE = re.compile(r"([a-z])([A-Z])")

# Shared date/number patterns for stripping metadata text.
_METADATA_STRIP_BASE = [
    re.compile(
        r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?"
        r"|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:Mon(?:day)?|Tue(?:s(?:day)?)?|Wed(?:nesday)?|Thu(?:rs(?:day)?)?"
        r"|Fri(?:day)?|Sat(?:urday)?|Sun(?:day)?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b\d+(?:st|nd|rd|th)?\b"),
    re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"),
]
# Read-time: strip everything including whitespace (expect empty residual)
_READ_TIME_STRIP_PATTERNS = [
    *_METADATA_STRIP_BASE,
    re.compile(r"\bmin(?:ute)?s?\b", re.IGNORECASE),
    re.compile(r"\bread(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\btime\b", re.IGNORECASE),
    re.compile(r"\bestimated\b", re.IGNORECASE),
    re.compile(r"[/|·•—–\-,:.\s]+"),
]
# Byline: preserve spaces so name words can be split
_BYLINE_STRIP_PATTERNS = [
    *_METADATA_STRIP_BASE,
    re.compile(r"\bby\b", re.IGNORECASE),
    re.compile(r"[/|·•—–\-,]+"),
]

# Content element selectors — presence indicates real article content.
# Mirrors defuddle CONTENT_ELEMENT_SELECTOR (constants.ts).
_CONTENT_ELEMENT_CSS = (
    "math, [data-mathml], .katex, .katex-mathml, .katex-display, "
    ".MathJax, .MathJax_Display, .MathJax_SVG, mjx-container, "
    "pre, code, table, img, picture, video, blockquote, figure"
)
# Minus img/picture — author avatars are common in metadata widgets.
_CONTENT_ELEMENT_NO_IMG_CSS = (
    "math, [data-mathml], .katex, .katex-mathml, .katex-display, "
    ".MathJax, .MathJax_Display, .MathJax_SVG, mjx-container, "
    "pre, code, table, video, blockquote, figure"
)

_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})


def _text(el: Tag) -> str:
    return el.get_text().strip()


def _attr(el: Tag, name: str) -> str:
    value = el.get(name)
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(value)


def _gone(el: Tag) -> bool:
    return bool(getattr(el, "decomposed", False)) or el.parent is None


def _tag_children(el: Tag) -> list[Tag]:
    return [c for c in el.children if isinstance(c, Tag)]


def _next_tag_sibling(el: Tag) -> Tag | None:
    sib = el.next_sibling
    while sib is not None and not isinstance(sib, Tag):
        sib = sib.next_sibling
    return sib


def _prev_tag_sibling(el: Tag) -> Tag | None:
    sib = el.previous_sibling
    while sib is not None and not isinstance(sib, Tag):
        sib = sib.previous_sibling
    return sib


def _is_or_contains_heading(el: Tag) -> bool:
    return el.name in _HEADING_TAGS or el.find(list(_HEADING_TAGS)) is not None


def _has_content_element(el: Tag, *, include_images: bool = True) -> bool:
    css = _CONTENT_ELEMENT_CSS if include_images else _CONTENT_ELEMENT_NO_IMG_CSS
    return el.select_one(css) is not None


def _is_newsletter_element(el: Tag, max_words: int) -> bool:
    text = _text(el)
    words = count_words(text)
    if words < 2 or words > max_words:
        return False
    if _has_content_element(el):
        return False
    # Adjacent element text may concatenate without whitespace; normalize
    # camelCase boundaries and curly apostrophes before matching.
    normalized = _CAMEL_BOUNDARY_RE.sub(r"\1 \2", text)
    normalized = normalized.replace("‘", "'").replace("’", "'")
    return _NEWSLETTER_RE.search(normalized) is not None


def _walk_up_to_wrapper(el: Tag, text: str, main_content: Tag) -> Tag:
    target = el
    while (
        isinstance(target.parent, Tag)
        and target.parent is not main_content
        and _text(target.parent) == text
    ):
        target = target.parent
    return target


def _remove_trailing_siblings(element: Tag, remove_self: bool) -> int:
    removed = 0
    sibling = _next_tag_sibling(element)
    while sibling is not None:
        nxt = _next_tag_sibling(sibling)
        if sibling.get("id") == "footnotes":
            sibling = nxt
            continue
        sibling.decompose()
        removed += 1
        sibling = nxt
    if remove_self:
        element.decompose()
        removed += 1
    return removed


def _remove_trailing_with_cascade(target: Tag, main_content: Tag) -> int:
    """Remove target and following siblings, cascading up each ancestor level."""
    ancestors: list[Tag] = []
    anc = target.parent
    while isinstance(anc, Tag) and anc is not main_content:
        ancestors.append(anc)
        anc = anc.parent
    removed = _remove_trailing_siblings(target, True)
    for ancestor in ancestors:
        removed += _remove_trailing_siblings(ancestor, False)
    return removed


def _walk_up_isolated(el: Tag, main_content: Tag) -> Tag:
    """Highest ancestor whose preceding siblings hold ≤ 10 words total."""
    target = el
    while isinstance(target.parent, Tag) and target.parent is not main_content:
        preceding_words = 0
        sib = _prev_tag_sibling(target)
        while sib is not None:
            preceding_words += count_words(sib.get_text())
            if preceding_words > 10:
                break
            sib = _prev_tag_sibling(sib)
        if preceding_words > 10:
            break
        target = target.parent
    return target


def _remove_thin_preceding_section(target: Tag) -> int:
    """Remove a thin CTA/promo block immediately before a related section."""
    prev_sib = _prev_tag_sibling(target)
    if prev_sib is None:
        return 0
    if count_words(prev_sib.get_text()) >= 50:
        return 0
    if _has_content_element(prev_sib):
        return 0
    # If preceded by a heading it's the body of a named section, not a CTA.
    before_prev = _prev_tag_sibling(prev_sib)
    if before_prev is not None and _is_or_contains_heading(before_prev):
        return 0
    prev_sib.decompose()
    return 1


def _remove_hero_header(main_content: Tag, content_start: Tag | None) -> int:
    """Remove wrappers grouping heading + <time> + hero image above the body.

    Walks up from a pre-content <time> to the largest ancestor that contains
    both a heading and a <time> but has < 30 words of non-metadata prose.
    """
    for time_el in main_content.find_all("time"):
        if not is_above_content_start(time_el, content_start):
            continue

        best_block: Tag | None = None
        current = time_el.parent
        while isinstance(current, Tag) and current is not main_content:
            if current.find(["h1", "h2"]) is not None and current.find("time"):
                total_words = count_words(_text(current))
                # Count words in metadata elements, deduping nested ones.
                metadata_els: list[Tag] = []
                for el in current.select("h1, h2, h3, time, [aria-label]"):
                    if not any(existing in el.parents for existing in metadata_els):
                        metadata_els.append(el)
                metadata_words = sum(count_words(el.get_text()) for el in metadata_els)
                prose_words = total_words - metadata_words
                if prose_words < 30:
                    best_block = current
                else:
                    break
            current = current.parent

        if best_block is not None:
            best_block.decompose()
            return 1
    return 0


def _is_breadcrumb_list(list_el: Tag) -> bool:
    """Detect a breadcrumb (Home › Posts › Title) posing as the first list."""
    list_items = list_el.find_all("li")
    if len(list_items) < 2 or len(list_items) > 8:
        return False

    list_links = list_el.find_all("a")
    if len(list_links) < 1 or len(list_links) >= len(list_items):
        return False
    if list_el.find(["img", "p", "figure", "blockquote"]) is not None:
        return False

    # Breadcrumb items are short labels; content lists have longer prose.
    for item in list_items:
        if count_words(item.get_text()) > 8:
            return False

    has_breadcrumb_link = False
    for a in list_links:
        href = _attr(a, "href")
        if href.startswith("http") or href.startswith("//"):
            return False
        if href == "/" or re.fullmatch(r"/[a-zA-Z0-9_-]+/?", href):
            has_breadcrumb_link = True
        if len([w for w in a.get_text().strip().split() if w]) > 5:
            return False
    return has_breadcrumb_link


def remove_eyebrow_label(main_content: Tag) -> int:
    """Remove short category labels immediately preceding the first heading.

    E.g. "Blog post", "Announcements" — presentational taxonomy labels that
    don't belong in extracted content. Runs before selector removal so the
    h1 anchor is still present on pages that strip title classes.
    """
    first_heading = main_content.find("h1") or main_content.find("h2")
    if first_heading is None:
        return 0

    # Walk up through wrappers where the heading is the first child, so we
    # match eyebrows appearing as siblings of an h1 ancestor.
    current: Tag = first_heading
    while (
        isinstance(current.parent, Tag)
        and current.parent is not main_content
        and _prev_tag_sibling(current) is None
    ):
        current = current.parent
    prev = _prev_tag_sibling(current)
    if prev is None:
        return 0

    text = _text(prev)
    words = count_words(text)
    if words < 1 or words > 6:
        return 0
    if len(text) > 40:
        return 0
    if _SENTENCE_PUNCT_RE.search(text):
        return 0
    if _DATE_RE.search(text):
        return 0
    if prev.select_one(
        "img, picture, video, iframe, figure, table, pre, code, time, [datetime], "
        "h1, h2, h3, h4, h5, h6, ul, ol, blockquote"
    ):
        return 0

    prev.decompose()
    return 1


def _remove_breadcrumb_list(root: Tag) -> int:
    first_list = root.find(["ul", "ol"])
    if first_list is None or not _is_breadcrumb_list(first_list):
        return 0
    target: Tag = first_list
    while (
        isinstance(target.parent, Tag)
        and target.parent is not root
        and len(_tag_children(target.parent)) == 1
    ):
        target = target.parent
    target.decompose()
    return 1


def _remove_promo_banner_links(root: Tag) -> int:
    """Remove promotional block <a> elements appearing before the first h1."""
    first_h1 = root.find("h1")
    if first_h1 is None:
        return 0
    removed = 0
    for link in root.select("a[href]"):
        if _gone(link) or _gone(first_h1):
            continue
        if not element_precedes(link, first_h1):
            continue
        if link.find("div") is None:
            continue
        if link.find(["img", "picture", "video"]) is not None:
            continue
        text = _text(link)
        if count_words(text) > 25:
            continue
        if _SENTENCE_PUNCT_SPACE_RE.search(text):
            continue
        link.decompose()
        removed += 1
    return removed


def _remove_listen_widgets(root: Tag, is_pre_content: _PreContentCheck) -> int:
    """Remove "Listen to this article" TTS widgets and pre-content players."""
    removed = 0
    for media in root.find_all(["audio", "video"]):
        if _gone(media):
            continue
        if not media.get("src") and media.find("source") is None:
            continue

        container: Tag = media
        while (
            isinstance(container.parent, Tag)
            and container.parent is not root
            and count_words(_text(container.parent)) <= 25
        ):
            container = container.parent

        container_text = _text(container)
        is_listen_widget = bool(
            re.search(
                r"\blisten\s+to\s+(?:this\s+)?(?:article|story|post|episode|podcast)\b",
                container_text,
                re.IGNORECASE,
            )
        )
        # Pre-content audio/video in a short container is almost always a
        # TTS widget — real media embeds appear within the article body.
        is_pre_content_player = (
            not is_listen_widget
            and is_pre_content(container)
            and count_words(container_text) <= 25
        )
        if is_listen_widget or is_pre_content_player:
            container.decompose()
            removed += 1
    return removed


def _remove_orphan_toc_headings(root: Tag) -> int:
    """Remove ToC-labelled headings whose list is already gone.

    NOTE: markitai-specific — mobile-style pruning (a markitai extension)
    can remove a ToC list while leaving its "Table of Contents" heading
    behind; a ToC label with no following list is always noise.
    """
    removed = 0
    for heading in list(root.find_all(list(_HEADING_TAGS))):
        if _gone(heading):
            continue
        if not _TOC_HEADING_RE.match(_text(heading)):
            continue
        nxt = _next_tag_sibling(heading)
        if nxt is not None and (
            nxt.name in ("ul", "ol") or nxt.find(["ul", "ol"]) is not None
        ):
            continue
        heading.decompose()
        removed += 1
    return removed


def _remove_toc(root: Tag, content_text: str, url: str) -> int:
    """Remove tables of contents — same-page anchor link lists near the top."""
    parsed_url = None
    if url:
        try:
            parsed_url = urlparse(url)
        except ValueError:
            parsed_url = None

    for list_el in root.find_all(["ul", "ol"]):
        if _gone(list_el):
            continue
        if list_el.find_parent(id="footnotes") is not None:
            continue

        list_text = _text(list_el)
        list_pos = content_text.find(list_text[:60])
        if list_pos < 0 or list_pos > len(content_text) * 0.3:
            continue

        links = list_el.select("a[href]")
        if len(links) < 3:
            continue
        if _has_content_element(list_el):
            continue

        anchor_count = 0
        for link in links:
            href = _attr(link, "href")
            if href.startswith("#"):
                anchor_count += 1
            elif parsed_url is not None and "#" in href:
                try:
                    resolved = urlparse(urljoin(url, href))
                except ValueError:
                    continue
                if (
                    resolved.path == parsed_url.path
                    and resolved.hostname == parsed_url.hostname
                ):
                    anchor_count += 1

        if anchor_count < 3 or anchor_count / len(links) < 0.8:
            continue

        target: Tag = list_el
        while (
            isinstance(target.parent, Tag)
            and target.parent is not root
            and len(_tag_children(target.parent)) == 1
        ):
            target = target.parent

        removed = 1
        # Remove an adjacent preceding heading if it's a ToC label
        prev_el = _prev_tag_sibling(target)
        if prev_el is not None and prev_el.name in _HEADING_TAGS:
            if _TOC_HEADING_RE.match(_text(prev_el)):
                prev_el.decompose()
                removed += 1

        # Remove surrounding HR separators that framed the ToC
        prev_sib = _prev_tag_sibling(target)
        next_sib = _next_tag_sibling(target)
        target.decompose()
        if prev_sib is not None and prev_sib.name == "hr":
            prev_sib.decompose()
            removed += 1
        if next_sib is not None and next_sib.name == "hr":
            next_sib.decompose()
            removed += 1
        return removed
    return 0


class _PreContentCheck:
    """Callable wrapper: is the element above the content-start boundary?"""

    def __init__(self, content_start: Tag | None) -> None:
        self._content_start = content_start

    def __call__(self, el: Tag) -> bool:
        return is_above_content_start(el, self._content_start)


def _remove_metadata_candidates(
    root: Tag,
    content_text: str,
    is_pre_content: _PreContentCheck,
    normalized_title: str,
    normalized_desc: str,
) -> int:
    """Single pass over short elements for all metadata-removal checks."""
    removed = 0
    byline_found = False
    author_date_found = False

    for el in list(root.find_all(["p", "span", "div", "time"])):
        if _gone(el):
            continue

        text = _text(el)
        words = count_words(text)
        # All checks target short metadata elements.
        if words > 15 or words == 0:
            continue
        if el.find_parent(["pre", "code"]) is not None:
            continue

        tag = el.name
        has_date = _DATE_RE.search(text) is not None

        # Timezone widgets (e.g. NYT live blogs): label is a child of a
        # container that also holds timezone entries.
        if _TIMEZONE_WIDGET_RE.match(text) and content_text.find(text) <= 300:
            target: Tag = el
            if isinstance(target.parent, Tag) and target.parent is not root:
                target = target.parent
            target.decompose()
            removed += 1
            continue

        # Standalone "Pinned" labels (live blog pinned post markers).
        if words == 1 and _PINNED_LABEL_RE.match(text):
            el.decompose()
            removed += 1
            continue

        # Pre-content elements duplicating the page title or description —
        # already extracted as metadata fields.
        duplicated = False
        for normalized in (normalized_title, normalized_desc):
            if (
                normalized
                and words >= 3
                and is_pre_content(el)
                and normalize_text(text) == normalized
            ):
                el.decompose()
                removed += 1
                duplicated = True
                break
        if duplicated or _gone(el):
            continue

        # Article metadata header blocks (date/category divs, relative
        # timestamps) near the top of content.
        if (
            tag in ("div", "p")
            and 1 <= words <= 10
            and (has_date or _RELATIVE_TIME_RE.search(text))
            and not _METADATA_LABEL_RE.match(text)
            and not _SENTENCE_PUNCT_RE.search(text)
            and is_pre_content(el)
        ):
            blocks = el.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
            if not any(count_words(b.get_text()) > 8 for b in blocks):
                el.decompose()
                removed += 1
                continue

        # Category/topic badge blocks: small containers holding only an
        # image link and a category name link.
        if (
            tag == "div"
            and 1 <= words <= 5
            and not _SENTENCE_PUNCT_RE.search(text)
            and is_pre_content(el)
            and el.find("img") is not None
        ):
            links = el.select("a[href]")
            if links:
                link_text_len = sum(len(_text(a)) for a in links)
                if link_text_len / (len(text) or 1) >= 0.8:
                    el.decompose()
                    removed += 1
                    continue

        # Standalone "By [Name]" author bylines near the start of content.
        if (
            not byline_found
            and _STARTS_WITH_BY_RE.match(text)
            and words >= 2
            and not _SENTENCE_END_RE.search(text)
            and is_pre_content(el)
        ):
            target = _walk_up_to_wrapper(el, text, root)
            target.decompose()
            removed += 1
            byline_found = True
            continue

        # Read time metadata ("8 min read", "Mar 4th 2026 | 3 min read").
        # With a date: any position, no block children. Without: short text
        # near the start.
        if _READ_TIME_RE.search(text) and (
            len(el.find_all(["p", "div", "section", "article"])) == 0
            if has_date
            else (words <= 5 and is_pre_content(el))
        ):
            cleaned = text
            for pattern in _READ_TIME_STRIP_PATTERNS:
                cleaned = pattern.sub("", cleaned)
            if not cleaned.strip():
                target = el if has_date else _walk_up_to_wrapper(el, text, root)
                target.decompose()
                removed += 1
                continue

        # Author + date bylines (name + date, any order) near the start.
        if (
            not author_date_found
            and 2 <= words <= 10
            and has_date
            and not _METADATA_LABEL_RE.match(text)
            and is_pre_content(el)
        ):
            residual = text
            for pattern in _BYLINE_STRIP_PATTERNS:
                residual = pattern.sub("", residual)
            residual = residual.strip()
            if residual:
                name_words = [w for w in residual.split() if w]
                if 1 <= len(name_words) <= 4 and all(
                    w[:1].isupper() for w in name_words
                ):
                    target = _walk_up_to_wrapper(el, text, root)
                    target.decompose()
                    removed += 1
                    author_date_found = True
                    continue

        # Standalone date elements near the start of content.
        if has_date and words <= 5 and is_pre_content(el):
            residual = text
            for pattern in _METADATA_STRIP_BASE:
                residual = pattern.sub("", residual)
            residual = re.sub(r"[,\s/\-]+", "", residual).strip()
            if not residual:
                target = _walk_up_to_wrapper(el, text, root)
                target.decompose()
                removed += 1
                continue

    return removed


def _remove_boundary_dates(root: Tag, content_text: str) -> int:
    """Remove standalone <time> elements near the start or end of content.

    A <time> in its own paragraph at the boundary is metadata (publish
    date), but <time> inline within prose is preserved.
    """
    removed = 0
    for time_el in list(root.find_all("time")):
        if _gone(time_el):
            continue
        # Walk up through inline/formatting wrappers only; a <p> that only
        # wraps this time is included, then the walk stops.
        target: Tag = time_el
        target_text = _text(target)
        while isinstance(target.parent, Tag) and target.parent is not root:
            parent_tag = target.parent.name
            parent_text = _text(target.parent)
            if parent_tag == "p" and parent_text == target_text:
                target = target.parent
                break
            if parent_tag in ("i", "em", "span", "b", "strong", "small") and (
                parent_text == target_text
            ):
                target = target.parent
                target_text = parent_text
                continue
            break
        text = _text(target)
        if count_words(text) > 10:
            continue
        pos = content_text.find(text)
        dist_from_end = len(content_text) - (pos + len(text))
        if pos > 200 and dist_from_end > 200:
            continue
        target.decompose()
        removed += 1
    return removed


def _remove_metadata_lists(root: Tag, content_text: str) -> int:
    """Remove short label/value lists (date, reading time, author, share)
    near content boundaries; <dl> author blocks allowed as single-item."""
    removed = 0
    for list_el in list(root.find_all(["ul", "ol", "dl"])):
        if _gone(list_el):
            continue
        if list_el.find_parent(id="footnotes") is not None:
            continue
        # NOTE: divergence from defuddle — GitHub-style task lists are short
        # and punctuation-free but are real content markitai preserves.
        if list_el.select_one('input[type="checkbox"]') is not None:
            continue
        is_dl = list_el.name == "dl"
        items = [
            c for c in _tag_children(list_el) if c.name == ("dd" if is_dl else "li")
        ]
        min_items = 1 if is_dl else 2
        if len(items) < min_items or len(items) > 8:
            continue

        list_text = _text(list_el)
        list_pos = content_text.find(list_text)
        dist_from_end = len(content_text) - (list_pos + len(list_text))
        if list_pos > 500 and dist_from_end > 500:
            continue

        # Lists introduced by a heading or a "…:" paragraph are content.
        prev_sibling = _prev_tag_sibling(list_el)
        if prev_sibling is not None:
            if _is_or_contains_heading(prev_sibling):
                continue
            if _text(prev_sibling).endswith(":"):
                continue

        is_metadata = True
        for item in items:
            item_text = _text(item)
            if count_words(item_text) > 8 or _SENTENCE_END_RE.search(item_text):
                is_metadata = False
                break
        if not is_metadata:
            continue
        if count_words(list_text) > 30:
            continue

        target = _walk_up_to_wrapper(list_el, list_text, root)
        target.decompose()
        removed += 1
    return removed


def _remove_section_breadcrumbs(root: Tag, url: str) -> int:
    """Remove parent-path breadcrumbs and bare back-navigation links."""
    try:
        parsed_url = urlparse(url) if url else None
    except ValueError:
        parsed_url = None
    # JS `new URL(...)` normalizes an empty path to "/"; urlparse doesn't.
    url_path = (parsed_url.path or "/") if parsed_url else ""
    if not url_path:
        return 0

    removed = 0
    first_heading = root.find(["h1", "h2", "h3"])
    elements = list(root.find_all(["div", "span", "p", "a"]))
    for el in elements:
        if _gone(el):
            continue
        if el.name == "a" and not el.get("href"):
            continue
        text = _text(el)
        if count_words(text) > 10:
            continue
        # Must be leaf-ish (no block children)
        if el.find(["p", "div", "section", "article"]) is not None:
            continue
        # Bare <a> embedded in flowing prose is skipped — unless it appears
        # before the first heading (back-nav links in page headers).
        if el.name == "a" and isinstance(el.parent, Tag) and el.parent is not root:
            parent_text = _text(el.parent)
            if parent_text != text:
                if el.find_parent("p") is not None:
                    continue
                if first_heading is None or _gone(first_heading):
                    continue
                if not element_precedes(el, first_heading):
                    continue
        link = el if el.name == "a" else el.select_one("a[href]")
        if link is None:
            continue
        try:
            link_path = urlparse(urljoin(url, _attr(link, "href"))).path or "/"
        except ValueError:
            continue
        # Also catch index.html links to a parent directory (../index.html)
        link_dir = re.sub(r"/[^/]*$", "/", link_path)
        basename = link_path.rsplit("/", 1)[-1]
        is_parent_index = bool(
            re.fullmatch(r"index\.(html?|php)", basename, re.IGNORECASE)
        ) and url_path.startswith(link_dir)
        if (
            link_path != "/"
            and link_path != url_path
            and (url_path.startswith(link_path) or is_parent_index)
        ):
            el.decompose()
            removed += 1
    return removed


def _remove_trailing_external_link_lists(root: Tag, url: str) -> int:
    """Remove a trailing heading + list of purely off-site links."""
    try:
        parsed_url = urlparse(url) if url else None
    except ValueError:
        parsed_url = None
    page_host = (parsed_url.hostname or "").removeprefix("www.") if parsed_url else ""
    if not page_host:
        return 0

    removed = 0
    for heading in list(root.find_all(["h2", "h3", "h4", "h5", "h6"])):
        if _gone(heading):
            continue
        list_el = _next_tag_sibling(heading)
        if list_el is None or list_el.name not in ("ul", "ol"):
            continue
        items = [c for c in _tag_children(list_el) if c.name == "li"]
        if len(items) < 2:
            continue

        # The list must be the last meaningful block at every ancestor level.
        trailing_content = False
        check_el: Tag | None = list_el
        while isinstance(check_el, Tag) and check_el is not root:
            sibling = _next_tag_sibling(check_el)
            while sibling is not None:
                if _text(sibling):
                    trailing_content = True
                    break
                sibling = _next_tag_sibling(sibling)
            if trailing_content:
                break
            check_el = check_el.parent if isinstance(check_el.parent, Tag) else None
        if trailing_content:
            continue

        # Every list item must be primarily a link pointing off-site.
        all_external = True
        for item in items:
            links = item.select("a[href]")
            if not links:
                all_external = False
                break
            item_text = _text(item)
            link_text_len = 0
            for link in links:
                link_text_len += len(_text(link))
                try:
                    link_host = (
                        urlparse(urljoin(url, _attr(link, "href"))).hostname or ""
                    ).removeprefix("www.")
                except ValueError:
                    continue
                if link_host == page_host:
                    all_external = False
                    break
            if not all_external:
                break
            if link_text_len < len(item_text) * 0.6:
                all_external = False
                break
        if not all_external:
            continue

        list_el.decompose()
        heading.decompose()
        removed += 2
    return removed


def _remove_trailing_related_posts(root: Tag) -> int:
    """Remove a trailing container of short, link-dense paragraphs."""
    children = _tag_children(root)
    last_child = children[-1] if children else None
    while last_child is not None and last_child.name in ("hr", "br"):
        last_child = _prev_tag_sibling(last_child)
    if last_child is None or last_child.name not in ("section", "div", "aside"):
        return 0

    paras: list[Tag] = []
    has_non_para = False
    for child in _tag_children(last_child):
        if not _text(child):
            continue
        if child.name == "p":
            paras.append(child)
        elif child.name != "br":
            has_non_para = True
            break
    if len(paras) < 2 or has_non_para:
        return 0

    for p in paras:
        text = re.sub(r"\s+", " ", _text(p))
        links = p.select("a[href]")
        if not links:
            return 0
        link_text_len = sum(len(_text(a)) for a in links)
        if link_text_len / (len(text) or 1) <= 0.6:
            return 0
        non_link_text = text
        for link in links:
            link_text = _text(link)
            if link_text:
                non_link_text = non_link_text.replace(link_text, "")
        if _SENTENCE_PUNCT_RE.search(non_link_text):
            return 0

    last_child.decompose()
    return 1


def _remove_trailing_thin_sections(root: Tag) -> int:
    """Remove trailing heading-bearing children with very little prose.

    Typically CTAs, newsletter prompts, or promo sections partially
    stripped by prior removal steps. Only runs on substantial documents.
    """
    total_words = count_words(root.get_text())
    if total_words <= 300:
        return 0

    trailing_els: list[Tag] = []
    trailing_words = 0
    children = _tag_children(root)
    for child in reversed(children):
        # Skip the standardized footnotes container
        if child.get("id") == "footnotes":
            continue
        # An <hr> is a content boundary — include it and stop walking
        if child.name == "hr":
            trailing_els.append(child)
            break
        # Exclude SVG text (path data) from word counts — it's not prose.
        svg_words = sum(count_words(svg.get_text()) for svg in child.find_all("svg"))
        words = count_words(_text(child)) - svg_words
        if words > 25:
            break
        trailing_words += words
        trailing_els.append(child)

    if not trailing_els or trailing_words >= total_words * 0.15:
        return 0
    has_heading = any(_is_or_contains_heading(el) for el in trailing_els)
    has_content = any(_has_content_element(el) for el in trailing_els)
    # Multiple prose paragraphs indicate a conclusion, not a CTA block.
    prose_paragraphs = sum(
        1 for el in trailing_els if el.name == "p" and count_words(el.get_text()) > 5
    )
    if not has_heading or has_content or prose_paragraphs >= 2:
        return 0
    for el in trailing_els:
        el.decompose()
    return len(trailing_els)


def _remove_boilerplate(root: Tag) -> int:
    """Remove end-of-article boilerplate and truncate trailing non-content."""
    removed = 0
    full_text = root.get_text()
    for el in list(root.find_all(["p", "div", "span", "section"])):
        if _gone(el):
            continue
        if el.find_parent(["pre", "code"]) is not None:
            continue
        text = _text(el)
        words = count_words(text)
        if words > 50 or words < 1:
            continue

        for pattern in _BOILERPLATE_PATTERNS:
            if pattern.search(text):
                # Walk up to an ancestor that has next siblings to truncate.
                target: Tag = el
                while isinstance(target.parent, Tag) and target.parent is not root:
                    if _next_tag_sibling(target) is not None:
                        break
                    target = target.parent

                # Only truncate if substantial content precedes the match.
                target_text = target.get_text()
                target_pos = full_text.find(target_text)
                if target_pos < 200:
                    # Walk-up reached a high-level wrapper. If the original
                    # element is a trailing orphan, remove it directly.
                    if target is not el and _next_tag_sibling(el) is None:
                        el.decompose()
                        removed += 1
                    break

                removed += _remove_trailing_with_cascade(target, root)
                break
    return removed


def _remove_related_sections(root: Tag, content_text: str) -> int:
    """Remove "Related posts" / "Read next" / CTA sections by heading text."""
    for heading in root.find_all(["h2", "h3", "h4", "h5", "h6"]):
        if _gone(heading):
            continue
        heading_text = _text(heading)
        is_cta = _CTA_HEADING_RE.match(heading_text) is not None
        if not is_cta and _RELATED_HEADING_RE.match(heading_text) is None:
            continue
        # Must appear after substantial content
        if content_text.find(heading_text) < 500:
            continue

        target = _walk_up_isolated(heading, root)
        if target is heading:
            # Direct child — only remove CTA headings (never real content)
            if not is_cta:
                continue
            return _remove_trailing_siblings(heading, True)
        removed = _remove_thin_preceding_section(target)
        return removed + _remove_trailing_with_cascade(target, root)
    return 0


def _remove_related_intros(root: Tag) -> int:
    """Remove orphaned "For more on/about ..." intro paragraphs."""
    removed = 0
    for el in list(root.find_all("p")):
        if _gone(el):
            continue
        text = _text(el)
        if not _RELATED_INTRO_RE.match(text):
            continue
        if count_words(text) > 20:
            continue
        if _has_content_element(el):
            continue
        el.decompose()
        removed += 1
    return removed


def _remove_related_card_grids(root: Tag, content_text: str) -> int:
    """Remove related-post card grids lacking a detectable heading."""
    content_word_count = count_words(content_text)
    for el in root.find_all("div"):
        if _gone(el):
            continue
        children = _tag_children(el)
        if len(children) < 2:
            continue

        # Each card must contain an image and either a heading or a link.
        card_count = sum(
            1
            for c in children
            if c.find(["img", "picture"]) is not None
            and (c.find(["h2", "h3", "h4"]) is not None or c.select_one("a[href]"))
        )
        if card_count < 2 or card_count < len(children) * 0.7:
            continue

        # Must appear after substantial content
        first_text = _text(children[0])[:30]
        if len(first_text) < 5 or content_text.find(first_text) < 500:
            continue

        # Skip grids whose text is a large share of total content.
        grid_words = count_words(el.get_text())
        if content_word_count > 0 and grid_words / content_word_count > 0.3:
            continue

        target = _walk_up_isolated(el, root)
        if target is el:
            continue

        removed = _remove_thin_preceding_section(target)
        return removed + _remove_trailing_siblings(target, True)
    return 0


def _remove_newsletter_sections(root: Tag) -> int:
    """Remove newsletter signup sections identified by their text content."""
    for el in root.find_all(["div", "section", "aside"]):
        if _gone(el):
            continue
        if el.find_parent(["pre", "code"]) is not None:
            continue
        if not _is_newsletter_element(el, 60):
            continue

        # Walk up while the newsletter is the only or near-only child.
        el_words = count_words(_text(el))
        target: Tag = el
        while isinstance(target.parent, Tag) and target.parent is not root:
            parent_words = count_words(_text(target.parent))
            if parent_words > el_words * 2 + 15:
                break
            target = target.parent
        target.decompose()
        return 1
    return 0


def _remove_newsletter_lists(root: Tag) -> int:
    """Remove <ul> elements whose only content is newsletter signup links."""
    for el in root.find_all("ul"):
        if _gone(el):
            continue
        if _is_newsletter_element(el, 30):
            el.decompose()
            return 1
    return 0


def _remove_author_contact_blocks(root: Tag, content_text: str) -> int:
    """Remove "Written by"/"Contact" blocks with emails or phone numbers."""
    for el in root.find_all(["div", "section"]):
        if _gone(el):
            continue
        text = _text(el)
        words = count_words(text)
        if words < 2 or words > 40:
            continue

        pos = content_text.find(text[:60])
        if pos < 0:
            continue
        if len(content_text) - (pos + len(text)) > 300:
            continue

        has_label = any(
            _AUTHOR_CONTACT_LABEL_RE.match(_text(child))
            for child in el.find_all(["div", "span", "p", "dt", "dd", "li"])
        )
        if not has_label:
            continue
        if not (
            _EMAIL_RE.search(text)
            or _PHONE_RE.search(text)
            or el.select_one('a[href^="mailto:"]')
        ):
            continue

        target = _walk_up_isolated(el, root)
        target.decompose()
        return 1
    return 0


def _remove_author_share_widgets(root: Tag) -> int:
    """Remove short "Author"/"Share"/"Written by" metadata widgets."""
    removed = 0
    for el in list(root.find_all(["p", "span", "div"])):
        if _gone(el):
            continue
        if not _SHARE_AUTHOR_LABEL_RE.match(_text(el)):
            continue

        container: Tag = el
        while (
            isinstance(container.parent, Tag)
            and container.parent is not root
            and count_words(_text(container.parent)) <= 15
        ):
            container = container.parent

        # Images excluded from the content check — avatars are common here.
        if _has_content_element(container, include_images=False):
            continue
        container.decompose()
        removed += 1
    return removed


def _remove_social_counters(root: Tag, content_text: str) -> int:
    """Remove social engagement counters ("9 Likes", "3 Comments")."""
    removed = 0
    for el in list(root.find_all(["a", "p", "div", "span"])):
        if _gone(el):
            continue
        text = _text(el)
        if not _SOCIAL_COUNTER_RE.match(text):
            continue
        if el.name == "a" and el.get("href"):
            continue
        if el.name != "a":
            pos = content_text.find(text)
            if len(content_text) - (pos + len(text)) > 200:
                continue
        target = _walk_up_to_wrapper(el, text, root)
        target.decompose()
        removed += 1
    return removed


def _remove_trailing_tag_blocks(root: Tag, content_text: str) -> int:
    """Remove trailing tag/category link blocks (link-only short divs)."""
    removed = 0
    for el in list(root.find_all("div")):
        if _gone(el):
            continue
        text = _text(el)
        words = count_words(text)
        if words < 1 or words > 10:
            continue
        if _SENTENCE_PUNCT_RE.search(text):
            continue
        if _has_content_element(el):
            continue

        pos = content_text.find(text)
        if pos < 0:
            continue
        if len(content_text) - (pos + len(text)) > 300:
            continue

        links = el.select("a[href]")
        if not links:
            continue
        link_text_len = sum(len(_text(a)) for a in links)
        if link_text_len / (len(text) or 1) < 0.8:
            continue
        el.decompose()
        removed += 1
    return removed


def remove_content_patterns(
    root: Tag,
    *,
    url: str = "",
    title: str = "",
    description: str = "",
) -> int:
    """Remove metadata/noise patterns from content by text shape.

    Mirrors defuddle ``removeByContentPattern``: breadcrumbs, promo banners,
    hero headers, TTS widgets, tables of contents, bylines, read-time and
    date metadata, metadata lists, back-navigation, trailing related/CTA/
    newsletter/tag sections, boilerplate sentences, and social counters.

    Args:
        root: Content root element.
        url: Canonical page URL (enables path-based breadcrumb checks).
        title: Page title from metadata (anchors the content boundary).
        description: Page description (for duplicate-text removal).

    Returns:
        Number of elements removed.
    """
    # Structural anchor for "where the prose body starts" — the
    # authoritative above/below check for pre-content heuristics.
    content_start = find_content_start(root, title)
    is_pre_content = _PreContentCheck(content_start)
    normalized_title = normalize_text(title) if title else ""
    normalized_desc = normalize_text(description) if description else ""

    removed = 0
    removed += _remove_breadcrumb_list(root)
    removed += _remove_promo_banner_links(root)
    removed += _remove_hero_header(root, content_start)
    removed += _remove_listen_widgets(root, is_pre_content)

    content_text = root.get_text()
    removed += _remove_toc(root, content_text, url)
    removed += _remove_orphan_toc_headings(root)
    removed += _remove_metadata_candidates(
        root, content_text, is_pre_content, normalized_title, normalized_desc
    )
    removed += _remove_boundary_dates(root, content_text)
    removed += _remove_metadata_lists(root, content_text)
    removed += _remove_section_breadcrumbs(root, url)
    removed += _remove_trailing_external_link_lists(root, url)
    removed += _remove_trailing_related_posts(root)
    removed += _remove_trailing_thin_sections(root)
    removed += _remove_boilerplate(root)
    removed += _remove_related_sections(root, content_text)
    removed += _remove_related_intros(root)
    removed += _remove_related_card_grids(root, content_text)
    removed += _remove_newsletter_sections(root)
    removed += _remove_newsletter_lists(root)
    removed += _remove_author_contact_blocks(root, content_text)
    removed += _remove_author_share_widgets(root)
    removed += _remove_social_counters(root, content_text)
    removed += _remove_trailing_tag_blocks(root, content_text)
    return removed
