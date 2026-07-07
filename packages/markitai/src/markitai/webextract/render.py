from __future__ import annotations

"""Canonical HTML renderer for semantic extraction models.

Converts a ``SemanticExtraction`` (containing a ``ConversationThread``) into
a structured HTML fragment.  This intermediate HTML representation exists so
that:

- Markitai can snapshot-test the extraction structure.
- HTML-level quality checks can be applied before Markdown conversion.
- A single shared Markdown renderer can consume both semantic and non-semantic
  pages uniformly.

This is the **only** place where threaded extractors may define their
presentation structure.  Downstream code must not re-implement this logic.
"""

from datetime import datetime
from html import escape

from markitai.webextract.semantics import (
    ConversationItem,
    ConversationThread,
    EmbeddedQuote,
    MediaAttachment,
)
from markitai.webextract.types import SemanticExtraction

_MAX_REPLY_DEPTH = 50

# Twitter API legacy format, e.g. "Wed Dec 24 10:00:00 +0000 2025"
_TWITTER_DATE_FORMAT = "%a %b %d %H:%M:%S %z %Y"


def render_semantic_content(extraction: SemanticExtraction) -> str:
    """Render a ``SemanticExtraction`` to a canonical HTML fragment.

    Returns an empty string when the extraction carries no structured semantic
    data (i.e. ``extraction.thread`` is ``None``).

    Args:
        extraction: The semantic extraction to render.

    Returns:
        A string of canonical HTML.  The outermost element is an
        ``<article>`` wrapping the thread title and all conversation items.
        Returns ``""`` if there is no thread to render.
    """
    if extraction.thread is None:
        return ""

    return _render_thread(extraction.thread)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_thread(thread: ConversationThread) -> str:
    """Render a full ConversationThread to an HTML article.

    By default, the H1 title and main item author meta are rendered.  For
    social posts (tweets), callers may set ``show_title_in_body=False``
    and ``show_author_meta=False`` on the thread — the title and author
    are already in the source frontmatter, and removing them from the body
    saves tokens for LLM consumption (matching defuddle's approach).

    Args:
        thread: The conversation thread to render.

    Returns:
        HTML string for the thread.
    """
    parts: list[str] = []
    parts.append("<article>")
    if thread.show_title_in_body:
        parts.append(f"<h1>{escape(thread.title)}</h1>")
    parts.append(_render_item(thread.main_item, show_meta=thread.show_author_meta))
    for continuation in thread.continuation_items:
        parts.append("<hr>")
        parts.append(_render_item(continuation, show_meta=False))
    if thread.items:
        parts.append("<h2>Comments</h2>")
        for item in _iter_top_level_items(thread):
            parts.append(_render_item_tree(item, thread.items))
    parts.append("</article>")
    return "\n".join(parts)


def _format_display_date(timestamp: str) -> str:
    """Format a timestamp string as a short display date (``YYYY-MM-DD``).

    Accepts ISO-8601 (with a trailing ``Z`` or offset) and the Twitter API
    legacy format (``"Wed Dec 24 10:00:00 +0000 2025"``).  Falls back to the
    raw string when the timestamp cannot be parsed.

    Args:
        timestamp: The raw timestamp string.

    Returns:
        A ``YYYY-MM-DD`` date string, or the raw input if unparseable.
    """
    raw = timestamp.strip()
    if not raw:
        return raw
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    try:
        return datetime.strptime(raw, _TWITTER_DATE_FORMAT).date().isoformat()
    except ValueError:
        return raw


def _author_display(name: str | None, handle: str | None) -> str:
    """Combine display name and @handle into a single author string."""
    if name and handle:
        return f"{name} {handle}"
    return name or handle or ""


def _render_meta_line(
    author: str,
    timestamp: str | None,
    css_class: str = "item-meta",
) -> str | None:
    """Render the ``**author** · date`` metadata line for an item.

    Args:
        author: Combined author display string (may be empty).
        timestamp: Raw timestamp string, if known.
        css_class: CSS class for the wrapping ``<p>``.

    Returns:
        An HTML ``<p>`` string, or ``None`` when there is nothing to show.
    """
    pieces: list[str] = []
    if author:
        pieces.append(f"<strong>{escape(author)}</strong>")
    if timestamp:
        pieces.append(f"<time>{escape(_format_display_date(timestamp))}</time>")
    if not pieces:
        return None
    return f'  <p class="{css_class}">{" · ".join(pieces)}</p>'


def _render_text_paragraphs(text: str, indent: str = "  ") -> list[str]:
    """Render plain text as one ``<p>`` element per non-empty line.

    Lines starting with ``- `` are escaped to ``\\- `` so that MarkItDown
    does not interpret them as Markdown list items (tweet text).
    """
    result: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        # Escape leading "- " to prevent Markdown list parsing
        if stripped.startswith("- "):
            stripped = "\\- " + stripped[2:]
        result.append(f"{indent}<p>{escape(stripped)}</p>")
    return result


def _render_media(media: MediaAttachment, indent: str = "  ") -> list[str]:
    """Render a media attachment to HTML.

    Images render as ``<img>``.  Videos and GIFs render as a link to the
    media resource (labelled by type), preceded by the poster image when
    one is available.  When no real ``https`` media URL is known (e.g. the
    DOM only exposes a session-local ``blob:`` handle), only the poster is
    rendered and the useless link is omitted.

    Args:
        media: The media attachment.
        indent: Leading indentation for emitted lines.

    Returns:
        List of HTML lines.
    """
    alt = escape(media.alt)
    url = escape(media.url)
    parts: list[str] = []
    if media.media_type in ("video", "gif"):
        label = "GIF" if media.media_type == "gif" else "Video"
        if media.poster:
            poster = escape(media.poster)
            parts.append(
                f'{indent}<div class="media-attachment">'
                f'<img src="{poster}" alt="{alt}"></div>'
            )
        if media.url.startswith("http"):
            parts.append(
                f'{indent}<p class="media-attachment"><a href="{url}">{label}</a></p>'
            )
    else:
        parts.append(
            f'{indent}<div class="media-attachment"><img src="{url}" alt="{alt}"></div>'
        )
    return parts


def _render_quote(quote: EmbeddedQuote) -> list[str]:
    """Render an embedded quote to a blockquote HTML fragment."""
    parts: list[str] = ['  <blockquote class="quoted-item">']
    meta = _render_meta_line(
        _author_display(quote.author_name, quote.author_handle),
        quote.timestamp,
        css_class="quote-meta",
    )
    if meta:
        parts.append("  " + meta)
    if quote.text:
        parts.extend(_render_text_paragraphs(quote.text, indent="    "))
    for media in quote.media:
        parts.extend(_render_media(media, indent="    "))
    if quote.url:
        parts.append(
            f'    <p><a href="{escape(quote.url)}">{escape(quote.url)}</a></p>'
        )
    parts.append("  </blockquote>")
    return parts


def _render_item(item: ConversationItem, *, show_meta: bool = True) -> str:
    """Render a single ConversationItem to an HTML div.

    Args:
        item: The conversation item to render.
        show_meta: Whether to emit the author/date metadata line.  Thread
            continuation posts by the same author suppress it.

    Returns:
        HTML string for the item.
    """
    parts: list[str] = []
    parts.append(f'<div class="conversation-item" data-id="{escape(item.id)}">')

    if show_meta:
        meta = _render_meta_line(
            _author_display(item.author_name, item.author_handle),
            item.timestamp,
        )
        if meta:
            parts.append(meta)

    if item.html:
        parts.append(f'  <div class="item-body">{item.html}</div>')
    elif item.text:
        parts.extend(_render_text_paragraphs(item.text))

    for media in item.media:
        parts.extend(_render_media(media))

    if item.card_url:
        card_label = item.card_title or item.card_url
        parts.append(
            f'  <p class="card-link">'
            f'<a href="{escape(item.card_url)}">{escape(card_label)}</a></p>'
        )

    if item.quoted_item is not None:
        parts.extend(_render_quote(item.quoted_item))

    parts.append("</div>")
    return "\n".join(parts)


def _iter_top_level_items(thread: ConversationThread) -> list[ConversationItem]:
    """Return replies that should render directly under the comments heading."""
    item_ids = {item.id for item in thread.items}
    root_id = thread.main_item.id
    top_level: list[ConversationItem] = []
    for item in thread.items:
        if item.parent_id in (None, "", root_id) or item.parent_id not in item_ids:
            top_level.append(item)
    return top_level


def _iter_child_items(
    parent_id: str,
    items: list[ConversationItem],
) -> list[ConversationItem]:
    """Return direct child items in their original source order."""
    return [item for item in items if item.parent_id == parent_id]


def _render_item_tree(
    item: ConversationItem,
    items: list[ConversationItem],
    depth: int = 0,
) -> str:
    """Render a conversation item along with any nested replies.

    Stops recursing beyond ``_MAX_REPLY_DEPTH`` to prevent stack overflow
    on deeply nested threads (e.g. Reddit, HackerNews).
    """
    parts = [_render_item(item)]
    if depth < _MAX_REPLY_DEPTH:
        children = _iter_child_items(item.id, items)
        if children:
            parts.append('<blockquote class="reply-thread">')
            for child in children:
                parts.append(_render_item_tree(child, items, depth + 1))
            parts.append("</blockquote>")
    return "\n".join(parts)
