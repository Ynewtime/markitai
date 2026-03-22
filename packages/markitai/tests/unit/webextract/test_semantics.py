from __future__ import annotations

"""Tests for the shared semantic model (ConversationThread, ConversationItem, etc.)."""


from markitai.webextract.semantics import (
    ConversationItem,
    ConversationThread,
    EmbeddedQuote,
    MediaAttachment,
)
from markitai.webextract.types import SemanticExtraction


class TestMediaAttachment:
    """Tests for the MediaAttachment dataclass."""

    def test_basic_construction_with_url(self) -> None:
        m = MediaAttachment(url="https://example.com/img.jpg")
        assert m.url == "https://example.com/img.jpg"

    def test_alt_defaults_to_empty_string(self) -> None:
        m = MediaAttachment(url="https://example.com/img.jpg")
        assert m.alt == ""

    def test_media_type_defaults_to_image(self) -> None:
        m = MediaAttachment(url="https://example.com/img.jpg")
        assert m.media_type == "image"

    def test_custom_media_type(self) -> None:
        m = MediaAttachment(url="https://example.com/clip.gif", media_type="gif")
        assert m.media_type == "gif"

    def test_custom_alt(self) -> None:
        m = MediaAttachment(url="https://example.com/img.jpg", alt="A cat")
        assert m.alt == "A cat"


class TestEmbeddedQuote:
    """Tests for the EmbeddedQuote dataclass."""

    def test_all_fields_default_to_none_or_empty(self) -> None:
        q = EmbeddedQuote()
        assert q.author_name is None
        assert q.author_handle is None
        assert q.text == ""
        assert q.url is None

    def test_construction_with_all_fields(self) -> None:
        q = EmbeddedQuote(
            author_name="Jane Doe",
            author_handle="@jane",
            text="Some quoted text",
            url="https://x.com/jane/status/123",
        )
        assert q.author_name == "Jane Doe"
        assert q.author_handle == "@jane"
        assert q.text == "Some quoted text"
        assert q.url == "https://x.com/jane/status/123"


class TestConversationItem:
    """Tests for the ConversationItem dataclass."""

    def test_requires_id(self) -> None:
        item = ConversationItem(id="abc123")
        assert item.id == "abc123"

    def test_optional_fields_default_to_none(self) -> None:
        item = ConversationItem(id="abc123")
        assert item.author_name is None
        assert item.author_handle is None
        assert item.timestamp is None
        assert item.parent_id is None
        assert item.quoted_item is None

    def test_text_and_html_default_to_empty(self) -> None:
        item = ConversationItem(id="abc123")
        assert item.text == ""
        assert item.html == ""

    def test_media_defaults_to_empty_list(self) -> None:
        item = ConversationItem(id="abc123")
        assert item.media == []

    def test_full_construction(self) -> None:
        media = MediaAttachment(url="https://example.com/img.jpg", alt="photo")
        quote = EmbeddedQuote(author_handle="@other", text="Quoted")
        item = ConversationItem(
            id="tweet-1",
            author_name="Wen Z",
            author_handle="@ixiaowenz",
            text="Hello world",
            html="<p>Hello world</p>",
            timestamp="2026-03-19T10:00:00Z",
            parent_id=None,
            quoted_item=quote,
            media=[media],
        )
        assert item.id == "tweet-1"
        assert item.author_handle == "@ixiaowenz"
        assert item.quoted_item is quote
        assert item.media[0] is media

    def test_parent_id_links_reply_to_parent(self) -> None:
        item = ConversationItem(id="reply-1", parent_id="root-tweet")
        assert item.parent_id == "root-tweet"


class TestConversationThread:
    """Tests for the ConversationThread dataclass."""

    def _make_main_item(self) -> ConversationItem:
        return ConversationItem(
            id="root",
            author_name="Wen Z",
            author_handle="@ixiaowenz",
            text="Main post content",
        )

    def test_basic_construction(self) -> None:
        main = self._make_main_item()
        thread = ConversationThread(title="Post by @ixiaowenz", main_item=main)
        assert thread.title == "Post by @ixiaowenz"
        assert thread.main_item is main

    def test_items_defaults_to_empty_list(self) -> None:
        thread = ConversationThread(
            title="Post by @ixiaowenz",
            main_item=self._make_main_item(),
        )
        assert thread.items == []

    def test_items_can_hold_replies(self) -> None:
        main = self._make_main_item()
        reply = ConversationItem(id="reply-1", parent_id="root", text="Reply text")
        thread = ConversationThread(
            title="Post by @ixiaowenz",
            main_item=main,
            items=[reply],
        )
        assert len(thread.items) == 1
        assert thread.items[0].id == "reply-1"

    def test_thread_items_can_encode_nested_reply_relationships(self) -> None:
        """Reply items carry parent_id to represent tree structure."""
        main = ConversationItem(id="root", parent_id=None, text="Root post")
        reply = ConversationItem(id="reply-1", parent_id="root", text="A reply")
        thread = ConversationThread(
            title="Nested",
            main_item=main,
            items=[reply],
        )
        assert thread.items[0].parent_id == "root"


class TestSemanticExtractionWithThread:
    """Tests that SemanticExtraction.thread accepts ConversationThread."""

    def test_semantic_extraction_accepts_conversation_thread(self) -> None:
        main = ConversationItem(id="root", author_handle="@ixiaowenz", text="Post")
        thread = ConversationThread(title="Post by @ixiaowenz", main_item=main)
        sem = SemanticExtraction(thread=thread)
        assert sem.thread is thread
        assert sem.thread is not None
        assert sem.thread.main_item.author_handle == "@ixiaowenz"

    def test_semantic_extraction_thread_none_by_default(self) -> None:
        sem = SemanticExtraction()
        assert sem.thread is None
