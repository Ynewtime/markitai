from __future__ import annotations

"""Tests for render_semantic_content() in render.py."""

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

    def test_render_includes_author_handle(self) -> None:
        """Author handle is present in the rendered output."""
        thread = _make_thread(main_handle="@testuser")
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "@testuser" in html

    def test_render_includes_thread_title(self) -> None:
        """Thread title appears somewhere in the rendered HTML."""
        thread = _make_thread(title="Post by @alice")
        html = render_semantic_content(SemanticExtraction(thread=thread))
        assert "Post by @alice" in html

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
