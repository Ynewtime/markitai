from __future__ import annotations

"""Tests for render_semantic_content() in render.py."""

from markitai.webextract.markdown import render_markdown
from markitai.webextract.render import render_semantic_content
from markitai.webextract.semantics import (
    ConversationItem,
    ConversationThread,
    EmbeddedQuote,
    MediaAttachment,
)
from markitai.webextract.types import SemanticExtraction


def _make_thread(
    *,
    title: str = "Post by @ixiaowenz",
    main_text: str = "Main post content",
    main_handle: str = "@ixiaowenz",
    replies: list[ConversationItem] | None = None,
) -> ConversationThread:
    main = ConversationItem(
        id="root",
        author_name="Wen Z",
        author_handle=main_handle,
        text=main_text,
        html=f"<p>{main_text}</p>",
    )
    return ConversationThread(
        title=title,
        main_item=main,
        items=replies or [],
    )


class TestRenderSemanticContent:
    """Tests for render_semantic_content()."""

    def test_render_thread_emits_article_element(self) -> None:
        thread = _make_thread()
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "<article" in html

    def test_render_thread_emits_conversation_item_class(self) -> None:
        thread = _make_thread()
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "conversation-item" in html

    def test_render_thread_emits_clean_main_item_then_followups(self) -> None:
        """Main item appears first; follow-up items appear after."""
        reply = ConversationItem(
            id="reply-1",
            parent_id="root",
            text="A follow-up reply",
        )
        thread = _make_thread(replies=[reply])
        html = render_semantic_content(SemanticExtraction(thread=thread))

        # Both main item and replies are present
        assert "conversation-item" in html
        assert "Main post content" in html
        assert "A follow-up reply" in html

        # Main item must appear before the reply
        main_pos = html.index("Main post content")
        reply_pos = html.index("A follow-up reply")
        assert main_pos < reply_pos

    def test_render_thread_preserves_quote_as_quoted_item_class(self) -> None:
        """Embedded quotes are marked with the quoted-item CSS class."""
        quote = EmbeddedQuote(
            author_handle="@other",
            text="A quoted tweet",
            url="https://x.com/other/status/999",
        )
        main = ConversationItem(
            id="root",
            author_handle="@ixiaowenz",
            text="Look at this",
            quoted_item=quote,
        )
        thread = ConversationThread(
            title="Post by @ixiaowenz",
            main_item=main,
        )
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "quoted-item" in html
        assert "A quoted tweet" in html

    def test_render_thread_preserves_media_as_img_tag(self) -> None:
        """Media attachments render as <img> elements."""
        media = MediaAttachment(
            url="https://example.com/photo.jpg",
            alt="A sunset photo",
        )
        main = ConversationItem(
            id="root",
            author_handle="@ixiaowenz",
            text="Beautiful",
            media=[media],
        )
        thread = ConversationThread(
            title="Post by @ixiaowenz",
            main_item=main,
        )
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "<img" in html
        assert "https://example.com/photo.jpg" in html

    def test_render_thread_preserves_quote_and_media_as_structured_children(
        self,
    ) -> None:
        """Regression: both quoted-item and img must be present together."""
        quote = EmbeddedQuote(author_handle="@other", text="Quoted content")
        media = MediaAttachment(url="https://example.com/img.jpg", alt="photo")
        main = ConversationItem(
            id="root",
            author_handle="@ixiaowenz",
            text="Check this",
            quoted_item=quote,
            media=[media],
        )
        thread = ConversationThread(title="Post by @ixiaowenz", main_item=main)
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "quoted-item" in html
        assert "<img" in html

    def test_render_no_thread_returns_empty_string(self) -> None:
        """SemanticExtraction with no thread renders to an empty string."""
        html = render_semantic_content(SemanticExtraction())
        assert html == ""

    def test_main_shows_author_meta_by_default(self) -> None:
        """Main item author meta is rendered by default (backward compat)."""
        thread = _make_thread(main_handle="@testuser")
        html = render_semantic_content(SemanticExtraction(thread=thread))
        # Author/date meta line SHOULD be in main item body by default
        assert "@testuser</strong>" in html

    def test_main_shows_title_by_default(self) -> None:
        """Thread title H1 is rendered by default (backward compat)."""
        thread = _make_thread(title="Post by @alice")
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "<h1>" in html

    def test_multiple_media_attachments_all_rendered(self) -> None:
        """All media attachments in a single item are rendered."""
        media1 = MediaAttachment(url="https://example.com/a.jpg")
        media2 = MediaAttachment(url="https://example.com/b.jpg")
        main = ConversationItem(
            id="root",
            author_handle="@user",
            text="Two images",
            media=[media1, media2],
        )
        thread = ConversationThread(title="Post by @user", main_item=main)
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "https://example.com/a.jpg" in html
        assert "https://example.com/b.jpg" in html

    def test_nested_reply_becomes_nested_markdown_quote(self) -> None:
        """Replies to replies must retain visible hierarchy in markdown."""
        parent = ConversationItem(
            id="reply-1",
            parent_id="root",
            author_handle="@parent",
            text="Parent reply",
        )
        child = ConversationItem(
            id="reply-2",
            parent_id="reply-1",
            author_handle="@child",
            text="Nested reply",
        )
        thread = _make_thread(replies=[parent, child])

        html = render_semantic_content(SemanticExtraction(thread=thread))
        markdown = render_markdown(html)

        assert "> **@child**" in markdown
        assert "> Nested reply" in markdown


class TestRenderDefuddleParity:
    """Rendering features ported from defuddle's Twitter extractor."""

    def test_main_omits_author_meta_when_disabled(self) -> None:
        """Main item author meta is omitted when show_author_meta=False."""
        main = ConversationItem(
            id="root",
            author_name="Baoyu",
            author_handle="@dotey",
            text="Hello",
            timestamp="2025-12-24T10:00:00.000Z",
        )
        thread = ConversationThread(
            title="Post by @dotey",
            main_item=main,
            show_title_in_body=False,
            show_author_meta=False,
        )
        html = render_semantic_content(SemanticExtraction(thread=thread))
        markdown = render_markdown(html)
        assert "**Baoyu @dotey** · 2025-12-24" not in markdown

    def test_comment_items_still_show_author_meta(self) -> None:
        """Comment/reply items should still show author/date meta."""
        main = ConversationItem(
            id="root",
            author_handle="@author",
            text="Main post",
        )
        reply = ConversationItem(
            id="r1",
            parent_id="root",
            author_handle="@commenter",
            text="Nice!",
            timestamp="2025-12-25T10:00:00.000Z",
        )
        thread = ConversationThread(
            title="Post by @author", main_item=main, items=[reply]
        )
        html = render_semantic_content(SemanticExtraction(thread=thread))
        markdown = render_markdown(html)
        assert "**@commenter** · 2025-12-25" in markdown

    def test_quote_items_still_show_author_meta(self) -> None:
        """Quoted tweets should still show author/date meta."""
        quote = EmbeddedQuote(
            author_handle="@quser",
            text="Quoted",
            timestamp="2025-12-24T10:00:00+00:00",
        )
        main = ConversationItem(
            id="root",
            author_handle="@dotey",
            text="Hello",
            quoted_item=quote,
        )
        thread = ConversationThread(title="Post by @dotey", main_item=main)
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "<time>2025-12-24</time>" in html

    def test_text_newlines_become_paragraphs(self) -> None:
        main = ConversationItem(
            id="root",
            author_handle="@u",
            text="First paragraph.\nSecond paragraph.",
        )
        thread = ConversationThread(title="Post by @u", main_item=main)
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "<p>First paragraph.</p>" in html
        assert "<p>Second paragraph.</p>" in html
        markdown = render_markdown(html)
        assert "First paragraph.\n\nSecond paragraph." in markdown

    def test_video_media_renders_poster_and_link(self) -> None:
        media = MediaAttachment(
            url="https://video.twimg.com/vid.mp4",
            media_type="video",
            poster="https://pbs.twimg.com/thumb.jpg",
        )
        main = ConversationItem(id="root", text="Watch", media=[media])
        thread = ConversationThread(title="Post by @u", main_item=main)
        markdown = render_markdown(
            render_semantic_content(SemanticExtraction(thread=thread))
        )
        assert "![](https://pbs.twimg.com/thumb.jpg)" in markdown
        assert "[Video](https://video.twimg.com/vid.mp4)" in markdown

    def test_card_link_rendered(self) -> None:
        main = ConversationItem(
            id="root",
            text="Check this",
            card_url="https://t.co/card123",
            card_title="Example Site",
        )
        thread = ConversationThread(title="Post by @u", main_item=main)
        markdown = render_markdown(
            render_semantic_content(SemanticExtraction(thread=thread))
        )
        assert "[Example Site](https://t.co/card123)" in markdown

    def test_quote_renders_author_media_and_permalink(self) -> None:
        quote = EmbeddedQuote(
            author_name="OpenAI",
            author_handle="@OpenAI",
            text="Announcement",
            url="https://x.com/OpenAI/status/111",
            timestamp="2025-12-23T08:00:00.000Z",
            media=[MediaAttachment(url="https://pbs.twimg.com/media/Q.jpg", alt="pic")],
        )
        main = ConversationItem(id="root", text="Quoting", quoted_item=quote)
        thread = ConversationThread(title="Post by @u", main_item=main)
        markdown = render_markdown(
            render_semantic_content(SemanticExtraction(thread=thread))
        )
        assert "> **OpenAI @OpenAI** · 2025-12-23" in markdown
        assert "![pic](https://pbs.twimg.com/media/Q.jpg)" in markdown
        assert "https://x.com/OpenAI/status/111" in markdown

    def test_continuation_items_rendered_with_hr_before_comments(self) -> None:
        continuation = ConversationItem(id="c1", text="Part two of the thread.")
        reply = ConversationItem(id="r1", text="A comment.")
        main = ConversationItem(id="root", author_handle="@u", text="Part one.")
        thread = ConversationThread(
            title="Post by @u",
            main_item=main,
            items=[reply],
            continuation_items=[continuation],
        )
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "<hr>" in html
        # Continuation appears before the Comments heading
        assert html.index("Part two of the thread.") < html.index("Comments")
        markdown = render_markdown(html)
        assert "Part two of the thread." in markdown
        assert "A comment." in markdown


class TestRenderDepthLimit:
    """Tests for reply nesting depth limit."""

    def test_deeply_nested_thread_does_not_crash(self) -> None:
        """A thread with 200+ levels of nesting should not hit RecursionError."""
        items = []
        for i in range(200):
            parent = "root" if i == 0 else f"item_{i - 1}"
            items.append(
                ConversationItem(
                    id=f"item_{i}",
                    author_name=f"user_{i}",
                    text=f"Reply level {i}",
                    parent_id=parent,
                )
            )

        thread = _make_thread(replies=items)
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "item_0" in html
        assert "Reply level 0" in html

    def test_items_beyond_depth_limit_are_truncated(self) -> None:
        """Items beyond MAX_REPLY_DEPTH should not produce nested blockquotes."""
        from markitai.webextract.render import _MAX_REPLY_DEPTH

        depth = _MAX_REPLY_DEPTH + 10
        items = []
        for i in range(depth):
            parent = "root" if i == 0 else f"item_{i - 1}"
            items.append(
                ConversationItem(
                    id=f"item_{i}",
                    author_name=f"user_{i}",
                    text=f"Reply level {i}",
                    parent_id=parent,
                )
            )

        thread = _make_thread(replies=items)
        html = render_semantic_content(SemanticExtraction(thread=thread))

        nesting_count = html.count('class="reply-thread"')
        assert nesting_count <= _MAX_REPLY_DEPTH
