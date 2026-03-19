from __future__ import annotations

"""Extractor for Reddit post thread pages (old.reddit.com HTML structure).

Implements the shared resolver pattern to prove the ConversationThread /
render_semantic_content abstraction applies to Reddit-style threaded
discussions, not just X/Twitter and GitHub.
"""

from bs4 import BeautifulSoup, Tag

from markitai.webextract.render import render_semantic_content
from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.semantics import ConversationItem, ConversationThread
from markitai.webextract.types import ContentProfile, SemanticExtraction


class RedditPostExtractor:
    """Extract Reddit post threads into a ConversationThread.

    Targets old.reddit.com post URLs.  Parses the self-post body as the
    ``main_item`` and collects top-level and nested comments as ``items``,
    discarding sidebar content (subreddit info, subscription prompts, etc.).

    Reddit HTML structure targeted:
    - Post title: ``<a class="title">`` inside ``.thing.link``
    - Post body: ``<div class="md">`` inside ``.expando``
    - Post author: ``<a class="author">`` inside ``.thing.link``
    - Comments: ``<div class="thing comment" data-fullname="t1_...">``
      - Author: ``<a class="author">``
      - Body: ``<div class="md">`` inside ``.usertext-body``
      - Nested comments: ``.child`` wrapping nested ``.thing.comment`` elements
    - Sidebar: ``<div class="side">`` (excluded)
    """

    name = "reddit_post"

    def matches_url(self, url: str) -> bool:
        """Match Reddit post URLs.

        Args:
            url: Source URL to test.

        Returns:
            True when the URL is a reddit.com post or comments page.
        """
        return "reddit.com" in url and ("/comments/" in url or "/r/" in url)

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the main content area (legacy path).

        This method exists for protocol compliance. The resolver path (via
        ``resolve()``) is preferred and produces richer output.

        Args:
            soup: Parsed HTML of the page.

        Returns:
            The main content div, or ``None``.
        """
        content = soup.find("div", class_="content")
        return content if isinstance(content, Tag) else None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve a Reddit post page into a structured ResolvedPage.

        Finds the post title and body, parses them as the ``main_item``, and
        collects all comment blocks as reply ``ConversationItem``s.  Sidebar
        content is excluded by restricting parsing to ``div.content``.

        Args:
            soup: Parsed HTML of the page.
            url: Source URL.

        Returns:
            A ``ResolvedPage`` with semantic content and metadata overrides.
        """
        # Remove sidebar before any parsing to avoid noise
        for side in soup.find_all("div", class_="side"):
            if isinstance(side, Tag):
                side.decompose()

        title = _extract_post_title(soup)
        post_author = _extract_post_author(soup)
        post_body_html, post_body_text = _extract_post_body(soup)

        main_item = ConversationItem(
            id="post-body",
            author_name=post_author,
            text=post_body_text,
            html=post_body_html,
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
            "site": "Reddit",
        }
        if post_author:
            metadata_overrides["author"] = post_author

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            semantic=semantic,
            diagnostics={
                "reddit_resolve": "success",
                "content_profile": ContentProfile.DISCUSSION_THREAD.value,
                "extractor_name": self.name,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_post_title(soup: BeautifulSoup) -> str:
    """Extract the post title from the page.

    Tries multiple selectors in priority order.

    Args:
        soup: Parsed page HTML.

    Returns:
        The post title string, or a generic fallback.
    """
    # Old Reddit: <a class="title"> inside .thing.link
    title_tag = soup.find("a", class_="title")
    if isinstance(title_tag, Tag):
        text = title_tag.get_text(strip=True)
        if text:
            return text

    # New Reddit or fallback: <h1> in post header
    h1 = soup.find("h1")
    if isinstance(h1, Tag):
        text = h1.get_text(strip=True)
        if text:
            return text

    return "Reddit Post"


def _extract_post_author(soup: BeautifulSoup) -> str | None:
    """Extract the post author from the post header.

    Args:
        soup: Parsed page HTML.

    Returns:
        The post author username, or ``None`` if not found.
    """
    # Old Reddit: .thing.link > .entry > .tagline > a.author
    thing = soup.find("div", class_="thing")
    if isinstance(thing, Tag):
        author_tag = thing.find("a", class_="author")
        if isinstance(author_tag, Tag):
            return author_tag.get_text(strip=True)
    return None


def _extract_post_body(soup: BeautifulSoup) -> tuple[str, str]:
    """Extract the self-post body HTML and text.

    Args:
        soup: Parsed page HTML.

    Returns:
        A tuple of (html_string, plain_text).
    """
    # Old Reddit: .expando > .md (self-post body)
    expando = soup.find("div", class_="expando")
    if isinstance(expando, Tag):
        md_div = expando.find("div", class_="md")
        if isinstance(md_div, Tag):
            return str(md_div), md_div.get_text(separator=" ", strip=True)

    return "", ""


def _collect_comments(soup: BeautifulSoup) -> list[ConversationItem]:
    """Collect all comments from the comment section.

    Traverses nested comment structures, preserving parent-child relationships
    via the ``parent_id`` field on each ``ConversationItem``.

    Args:
        soup: Parsed page HTML.

    Returns:
        Ordered list of ConversationItems for all visible comments.
    """
    commentarea = soup.find("div", class_="commentarea")
    if not isinstance(commentarea, Tag):
        return []

    items: list[ConversationItem] = []
    _collect_comment_nodes(commentarea, parent_id=None, items=items)
    return items


def _collect_comment_nodes(
    scope: Tag,
    parent_id: str | None,
    items: list[ConversationItem],
) -> None:
    """Recursively collect comment items from a DOM subtree.

    Args:
        scope: The parent Tag to search within (commentarea or child div).
        parent_id: The parent comment ID, or ``None`` for top-level comments.
        items: Accumulated list to append ConversationItems to.
    """
    # Find direct child .thing.comment elements (not grandchildren)
    for child in scope.children:
        if not isinstance(child, Tag):
            continue

        # Recurse into sitetable/nestedlisting wrappers transparently
        classes = child.get("class") or []
        if isinstance(classes, list):
            class_set = set(classes)
        else:
            class_set = {classes}

        if "sitetable" in class_set or "nestedlisting" in class_set:
            _collect_comment_nodes(child, parent_id, items)
            continue

        # Process a .thing.comment element
        if "thing" in class_set and "comment" in class_set:
            fullname = child.get("data-fullname")
            comment_id = str(fullname) if fullname else f"comment-{len(items)}"

            entry = child.find("div", class_="entry", recursive=False)
            if not isinstance(entry, Tag):
                # Try non-recursive fallback
                entry = child.find("div", class_="entry")

            author: str | None = None
            body_html = ""
            body_text = ""

            if isinstance(entry, Tag):
                author_tag = entry.find("a", class_="author")
                if isinstance(author_tag, Tag):
                    author = author_tag.get_text(strip=True)

                usertext = entry.find("div", class_="usertext-body")
                if isinstance(usertext, Tag):
                    md_div = usertext.find("div", class_="md")
                    if isinstance(md_div, Tag):
                        body_html = str(md_div)
                        body_text = md_div.get_text(separator=" ", strip=True)

            items.append(
                ConversationItem(
                    id=comment_id,
                    author_name=author,
                    text=body_text,
                    html=body_html,
                    parent_id=parent_id,
                )
            )

            # Recurse into .child divs to collect nested replies
            if isinstance(entry, Tag):
                child_div = entry.find("div", class_="child")
                if isinstance(child_div, Tag):
                    _collect_comment_nodes(child_div, comment_id, items)
