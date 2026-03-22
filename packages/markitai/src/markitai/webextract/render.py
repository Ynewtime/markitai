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

from html import escape

from markitai.webextract.semantics import ConversationItem, ConversationThread
from markitai.webextract.types import SemanticExtraction

_MAX_REPLY_DEPTH = 50


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

    Args:
        thread: The conversation thread to render.

    Returns:
        HTML string for the thread.
    """
    parts: list[str] = []
    parts.append("<article>")
    parts.append(f"<h1>{escape(thread.title)}</h1>")
    parts.append(_render_item(thread.main_item))
    if thread.items:
        parts.append("<h2>Comments</h2>")
        for item in _iter_top_level_items(thread):
            parts.append(_render_item_tree(item, thread.items))
    parts.append("</article>")
    return "\n".join(parts)


def _render_item(item: ConversationItem) -> str:
    """Render a single ConversationItem to an HTML div.

    Args:
        item: The conversation item to render.

    Returns:
        HTML string for the item.
    """
    parts: list[str] = []
    parts.append(f'<div class="conversation-item" data-id="{escape(item.id)}">')

    if item.author_handle or item.author_name:
        author = item.author_handle or item.author_name or ""
        parts.append(f'  <span class="author">{escape(author)}</span>')

    if item.timestamp:
        parts.append(f"  <time>{escape(item.timestamp)}</time>")

    if item.html:
        parts.append(f'  <div class="item-body">{item.html}</div>')
    elif item.text:
        parts.append(f'  <div class="item-body">{escape(item.text)}</div>')

    for media in item.media:
        alt = escape(media.alt)
        url = escape(media.url)
        parts.append(
            f'  <div class="media-attachment"><img src="{url}" alt="{alt}"></div>'
        )

    if item.quoted_item is not None:
        q = item.quoted_item
        parts.append('  <blockquote class="quoted-item">')
        if q.author_handle or q.author_name:
            qauthor = q.author_handle or q.author_name or ""
            parts.append(f'    <span class="author">{escape(qauthor)}</span>')
        if q.text:
            parts.append(f"    <p>{escape(q.text)}</p>")
        if q.url:
            parts.append(f'    <a href="{escape(q.url)}">{escape(q.url)}</a>')
        parts.append("  </blockquote>")

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
