"""Extractor for GitHub issue and pull request thread pages.

Implements the same resolver pattern as XTweetExtractor to prove that the
shared ConversationThread / render_semantic_content abstraction is not
X/Twitter-specific.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.render import render_semantic_content
from markitai.webextract.resolver import ResolvedPage
from markitai.webextract.semantics import ConversationItem, ConversationThread
from markitai.webextract.types import ContentProfile, SemanticExtraction


class GitHubThreadExtractor:
    """Extract GitHub issue or pull request threads into a ConversationThread.

    Targets github.com issue and pull request URLs.  Parses the issue body as
    the ``main_item`` and collects subsequent comments as ``items`` in the
    thread, discarding sidebar content (Assignees, Labels, Milestone, etc.).

    GitHub HTML structure targeted:
    - Issue title: ``<bdi class="js-issue-title">`` or
      ``<h1 class="gh-header-title"><span>``
    - Issue/comment bodies: ``<div class="comment-body">``
    - Author per block: ``<a class="author">username</a>``
    - Timestamp per block: ``<relative-time datetime="...">``
    - Sidebar: ``<div class="Layout-sidebar">`` (excluded)
    """

    name = "github_thread"

    def matches_url(self, url: str) -> bool:
        """Match GitHub issue and pull request URLs.

        Args:
            url: Source URL to test.

        Returns:
            True when the URL is a github.com issue or PR page.
        """
        return "github.com" in url and ("/issues/" in url or "/pull/" in url)

    def extract_root(self, soup: BeautifulSoup) -> Tag | None:
        """Return the first comment-body as a minimal root (legacy path).

        This method exists for protocol compliance. The resolver path (via
        ``resolve()``) is preferred and produces richer output.

        Args:
            soup: Parsed HTML of the page.

        Returns:
            The first ``<div class="comment-body">`` tag, or ``None``.
        """
        body = soup.find("div", class_="comment-body")
        return body if isinstance(body, Tag) else None

    def resolve(self, soup: BeautifulSoup, url: str) -> ResolvedPage:
        """Resolve a GitHub issue/PR page into a structured ResolvedPage.

        Finds the issue title, parses the first ``comment-body`` as the
        ``main_item``, and collects remaining ``comment-body`` blocks as
        reply ``ConversationItem``s.  Sidebar content is excluded by
        restricting parsing to the main layout column.

        Args:
            soup: Parsed HTML of the page.
            url: Source URL.

        Returns:
            A ``ResolvedPage`` with semantic content and metadata overrides.
        """
        title = _extract_issue_title(soup)

        # Restrict to main column to avoid sidebar noise
        main_col = _find_main_column(soup)
        scope: BeautifulSoup | Tag = main_col if main_col is not None else soup

        comment_blocks = _find_comment_blocks(scope)
        if not comment_blocks:
            return ResolvedPage(
                diagnostics={"github_resolve": "no_comment_bodies_found"},
            )

        main_item = _parse_comment_block(comment_blocks[0], item_id="issue-body")
        reply_items = [
            _parse_comment_block(block, item_id=f"comment-{i}")
            for i, block in enumerate(comment_blocks[1:], start=1)
        ]

        thread = ConversationThread(
            title=title,
            main_item=main_item,
            items=reply_items,
        )

        semantic = SemanticExtraction(thread=thread)
        content_html = render_semantic_content(semantic)

        metadata_overrides: dict[str, object] = {
            "title": title,
            "site": "GitHub",
        }
        if main_item.author_name:
            metadata_overrides["author"] = main_item.author_name

        return ResolvedPage(
            content_html=content_html,
            metadata_overrides=metadata_overrides,
            semantic=semantic,
            diagnostics={
                "github_resolve": "success",
                "content_profile": ContentProfile.DISCUSSION_ISSUE.value,
                "extractor_name": self.name,
            },
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_issue_title(soup: BeautifulSoup) -> str:
    """Extract the issue or PR title from the page.

    Tries multiple selectors in priority order.

    Args:
        soup: Parsed page HTML.

    Returns:
        The issue title string, or a generic fallback.
    """
    # GitHub new UI: <bdi class="js-issue-title">
    bdi = soup.find("bdi", class_="js-issue-title")
    if isinstance(bdi, Tag):
        return bdi.get_text(strip=True)

    # Older UI: <h1 class="gh-header-title"><span>…</span>
    h1 = soup.find("h1", class_="gh-header-title")
    if isinstance(h1, Tag):
        span = h1.find("span")
        if isinstance(span, Tag):
            return span.get_text(strip=True)
        return h1.get_text(strip=True)

    # Minimal fixture / older structure: <div class="gh-header-show"><h1>Title</h1>
    header_show = soup.find(class_="gh-header-show")
    if isinstance(header_show, Tag):
        h1_in_header = header_show.find("h1")
        if isinstance(h1_in_header, Tag):
            # Strip issue number suffix if present (e.g. "#42")
            text = h1_in_header.get_text(" ", strip=True)
            # Remove trailing " #NNN" issue/PR number
            import re

            text = re.sub(r"\s*#\d+\s*$", "", text).strip()
            if text:
                return text

    return "GitHub Issue"


def _find_main_column(soup: BeautifulSoup) -> Tag | None:
    """Find the main content column, excluding the sidebar.

    Args:
        soup: Parsed page HTML.

    Returns:
        The main layout column Tag, or ``None`` if not found.
    """
    # New GitHub layout uses Layout-main
    main = soup.find("div", class_="Layout-main")
    if isinstance(main, Tag):
        return main

    # Older layout: look for the div that contains js-timeline-item elements
    # but NOT the sidebar
    timeline = soup.find("div", class_="js-issues-results")
    if isinstance(timeline, Tag):
        return timeline

    return None


def _find_comment_blocks(scope: BeautifulSoup | Tag) -> list[Tag]:
    """Find all comment body blocks within the given scope.

    Args:
        scope: The BeautifulSoup or Tag to search within.

    Returns:
        Ordered list of ``<div class="comment-body">`` Tags.
    """
    blocks = scope.find_all("div", class_="comment-body")
    return [b for b in blocks if isinstance(b, Tag)]


def _parse_comment_block(block: Tag, item_id: str) -> ConversationItem:
    """Parse a single comment body block into a ConversationItem.

    Looks for the author link and timestamp in the nearest enclosing
    timeline-comment or timeline-comment-header sibling structure.

    Args:
        block: The ``<div class="comment-body">`` Tag.
        item_id: Stable identifier for the resulting ConversationItem.

    Returns:
        A populated ConversationItem.
    """
    # Walk up to find the timeline-comment container to locate author/timestamp
    author_name: str | None = None
    timestamp: str | None = None

    container = block.parent
    while container is not None and not isinstance(container, BeautifulSoup):
        author_tag = container.find("a", class_="author")
        if isinstance(author_tag, Tag):
            author_name = author_tag.get_text(strip=True)
            break
        container = container.parent  # type: ignore[assignment]

    # Look for relative-time in the same container
    if container is not None and not isinstance(container, BeautifulSoup):
        rt = container.find("relative-time")
        if isinstance(rt, Tag):
            raw_dt = rt.get("datetime")
            if isinstance(raw_dt, str):
                timestamp = raw_dt
            elif isinstance(raw_dt, list):
                timestamp = raw_dt[0] if raw_dt else None

    text = block.get_text(separator=" ", strip=True)
    html_content = str(block)

    return ConversationItem(
        id=item_id,
        author_name=author_name,
        text=text,
        html=html_content,
        timestamp=timestamp,
    )
