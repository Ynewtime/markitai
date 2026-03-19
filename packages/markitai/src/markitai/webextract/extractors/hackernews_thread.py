from __future__ import annotations

"""Extractor for Hacker News thread pages.

Implements the shared resolver pattern to prove the ConversationThread /
render_semantic_content abstraction applies to HN-style threaded discussions,
not just X/Twitter, GitHub, and Reddit.
"""

from bs4 import BeautifulSoup, Tag

from markitai.webextract.render import render_semantic_content
from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.semantics import ConversationItem, ConversationThread
from markitai.webextract.types import ContentProfile, SemanticExtraction


class HackerNewsThreadExtractor:
    """Extract Hacker News story threads into a ConversationThread.

    Targets news.ycombinator.com item URLs.  Parses the story text (if any)
    as the ``main_item`` and collects comments as ``items``, inferring
    parent-child relationships from the indent level of each comment row.

    HN HTML structure targeted:
    - Story title: ``<span class="titleline"><a>Title</a></span>``
    - Story text: ``<span class="commtext">`` in the first non-comment td
    - Comments: ``<tr class="athing comtr" id="...">``
      - Author: ``<a class="hnuser">username</a>``
      - Text: ``<span class="commtext">``
      - Indent depth: ``<td class="ind" indent="N">`` attribute
    """

    name = "hackernews_thread"

    def matches_url(self, url: str) -> bool:
        """Match Hacker News item/thread URLs.

        Args:
            url: Source URL to test.

        Returns:
            True when the URL is a news.ycombinator.com thread item page.
        """
        return "news.ycombinator.com" in url and "item?id=" in url

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the main HN table as a minimal root (legacy path).

        This method exists for protocol compliance. The resolver path (via
        ``resolve()``) is preferred and produces richer output.

        Args:
            soup: Parsed HTML of the page.

        Returns:
            The ``#hnmain`` table Tag, or ``None``.
        """
        hnmain = soup.find("table", id="hnmain")
        return hnmain if isinstance(hnmain, Tag) else None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve a Hacker News thread page into a structured ResolvedPage.

        Finds the story title and optional text, parses them as the
        ``main_item``, and collects comment rows as reply
        ``ConversationItem``s using indent depth to infer parent-child
        relationships.

        Args:
            soup: Parsed HTML of the page.
            url: Source URL.

        Returns:
            A ``ResolvedPage`` with semantic content and metadata overrides.
        """
        title = _extract_story_title(soup)
        story_text_html, story_text = _extract_story_text(soup)
        story_author = _extract_story_author(soup)

        # If no story text, use the title as the body so main_item is non-empty
        main_item = ConversationItem(
            id="story",
            author_name=story_author,
            text=story_text or title,
            html=story_text_html,
        )

        comment_items = _collect_comments(soup)

        thread = ConversationThread(
            title=title,
            main_item=main_item,
            items=comment_items,
        )

        semantic = SemanticExtraction(thread=thread)
        content_html = render_semantic_content(semantic)

        metadata_overrides: dict[str, object] = {
            "title": title,
            "site": "Hacker News",
        }
        if story_author:
            metadata_overrides["author"] = story_author

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            semantic=semantic,
            diagnostics={
                "hn_resolve": "success",
                "content_profile": ContentProfile.DISCUSSION_THREAD.value,
                "extractor_name": self.name,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_story_title(soup: BeautifulSoup) -> str:
    """Extract the story title from the HN page.

    Args:
        soup: Parsed page HTML.

    Returns:
        Story title string, or a generic fallback.
    """
    titleline = soup.find("span", class_="titleline")
    if isinstance(titleline, Tag):
        link = titleline.find("a")
        if isinstance(link, Tag):
            text = link.get_text(strip=True)
            if text:
                return text

    # Fallback: page <title>
    title_tag = soup.find("title")
    if isinstance(title_tag, Tag):
        text = title_tag.get_text(strip=True)
        # Strip " | Hacker News" suffix
        if " | Hacker News" in text:
            text = text[: text.rfind(" | Hacker News")].strip()
        if text:
            return text

    return "Hacker News Thread"


def _extract_story_text(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract the story body text, if present (Ask HN / Show HN with text).

    Args:
        soup: Parsed page HTML.

    Returns:
        A tuple of (html_string, plain_text).
    """
    # Story text lives in a <span class="commtext"> outside the comment tree,
    # or in a <td class="title"> adjacent to the first athing row.
    # We look for the first .commtext that is NOT inside a .comtr row.
    for span in soup.find_all("span", class_="commtext"):
        if not isinstance(span, Tag):
            continue
        # Check if this span is inside a .comtr (comment row)
        parent = span.parent
        inside_comment = False
        while parent is not None and not isinstance(parent, BeautifulSoup):
            if isinstance(parent, Tag):
                classes = parent.get("class") or []
                if isinstance(classes, list) and "comtr" in classes:
                    inside_comment = True
                    break
            parent = parent.parent  # type: ignore[assignment]

        if not inside_comment:
            return str(span), span.get_text(separator=" ", strip=True)

    return "", ""


def _extract_story_author(soup: BeautifulSoup) -> str | None:
    """Extract the story submitter username from the page.

    Args:
        soup: Parsed page HTML.

    Returns:
        The submitter username, or ``None`` if not found.
    """
    # The submitter appears in the .subline span (subtext row)
    subline = soup.find("span", class_="subline")
    if isinstance(subline, Tag):
        user_tag = subline.find("a", class_="hnuser")
        if isinstance(user_tag, Tag):
            return user_tag.get_text(strip=True)
    return None


def _collect_comments(soup: BeautifulSoup) -> list[ConversationItem]:
    """Collect all comments from the HN thread.

    Infers parent-child relationships from the ``indent`` attribute on each
    comment's ``<td class="ind">`` cell.  Maintains a depth stack to resolve
    the nearest ancestor at each depth level.

    Args:
        soup: Parsed page HTML.

    Returns:
        Ordered list of ConversationItems for all visible comments.
    """
    comment_rows = soup.find_all("tr", class_="comtr")
    items: list[ConversationItem] = []
    # Stack maps depth -> comment_id of the most recent item at that depth
    depth_stack: dict[int, str] = {}

    for row in comment_rows:
        if not isinstance(row, Tag):
            continue

        comment_id = str(row.get("id", f"comment-{len(items)}"))

        # Determine indent depth from the ind <td> attribute
        ind_td = row.find("td", class_="ind")
        depth = 0
        if isinstance(ind_td, Tag):
            raw_indent = ind_td.get("indent")
            if isinstance(raw_indent, str) and raw_indent.isdigit():
                depth = int(raw_indent)
            elif isinstance(raw_indent, list) and raw_indent:
                val = raw_indent[0]
                if isinstance(val, str) and val.isdigit():
                    depth = int(val)

        # Author
        author: str | None = None
        user_tag = row.find("a", class_="hnuser")
        if isinstance(user_tag, Tag):
            author = user_tag.get_text(strip=True)

        # Comment text
        body_html = ""
        body_text = ""
        commtext = row.find("span", class_="commtext")
        if isinstance(commtext, Tag):
            body_html = str(commtext)
            body_text = commtext.get_text(separator=" ", strip=True)

        # Resolve parent from depth stack
        parent_id: str | None = None
        if depth > 0 and (depth - 1) in depth_stack:
            parent_id = depth_stack[depth - 1]

        items.append(
            ConversationItem(
                id=comment_id,
                author_name=author,
                text=body_text,
                html=body_html,
                parent_id=parent_id,
            )
        )

        # Update the depth stack for this depth level
        depth_stack[depth] = comment_id
        # Invalidate any deeper levels (new branch at this depth)
        to_remove = [d for d in depth_stack if d > depth]
        for d in to_remove:
            del depth_stack[d]

    return items
