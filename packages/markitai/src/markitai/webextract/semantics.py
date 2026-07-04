from __future__ import annotations

"""Shared semantic model for threaded web content extractions.

Provides the canonical data structures used by site-specific extractors
(e.g. X/Twitter, GitHub Discussions) to represent threaded conversations.
These models are renderer-agnostic; the ``render`` module consumes them to
produce canonical HTML fragments.
"""

from dataclasses import dataclass, field


@dataclass(slots=True)
class MediaAttachment:
    """A media file attached to a conversation item.

    Attributes:
        url: Absolute URL of the media resource.
        alt: Alt text for the media (empty string if absent).
        media_type: Kind of media: ``"image"``, ``"video"``, or ``"gif"``.
        poster: Thumbnail/poster image URL for video media (empty if absent).
    """

    url: str
    alt: str = ""
    media_type: str = "image"
    poster: str = ""


@dataclass(slots=True)
class EmbeddedQuote:
    """A quoted post embedded within a conversation item.

    Attributes:
        author_name: Display name of the quoted author.
        author_handle: @handle of the quoted author.
        text: Plain text of the quoted content.
        url: Permalink to the original quoted post.
        timestamp: ISO-8601 timestamp string of the quoted post, if known.
        media: Media attachments belonging to the quoted post.
    """

    author_name: str | None = None
    author_handle: str | None = None
    text: str = ""
    url: str | None = None
    timestamp: str | None = None
    media: list[MediaAttachment] = field(default_factory=list)


@dataclass(slots=True)
class ConversationItem:
    """A single post or reply within a conversation thread.

    The ``id`` field is stable and unique within a thread; it is used by
    ``parent_id`` on sibling items to express reply relationships.

    Attributes:
        id: Stable identifier for this item (e.g. tweet ID or comment ID).
        author_name: Display name of the item's author.
        author_handle: @handle of the item's author.
        text: Plain text body of the item.
        html: HTML body of the item (may be empty if only ``text`` is known).
        timestamp: ISO-8601 timestamp string, if available.
        parent_id: ``id`` of the parent item this item replies to; ``None``
            for top-level items.
        quoted_item: An embedded quote referenced from this item.
        media: Media attachments associated with this item.
        card_url: Link-preview card URL attached to the item, if any.
        card_title: Human-readable title of the link-preview card.
    """

    id: str
    author_name: str | None = None
    author_handle: str | None = None
    text: str = ""
    html: str = ""
    timestamp: str | None = None
    parent_id: str | None = None
    quoted_item: EmbeddedQuote | None = None
    media: list[MediaAttachment] = field(default_factory=list)
    card_url: str | None = None
    card_title: str = ""


@dataclass(slots=True)
class ConversationThread:
    """A threaded conversation rooted at a single main post.

    The semantic model retains full tree structure via ``parent_id`` on each
    item. Renderers may choose to flatten items for Markdown readability.

    Attributes:
        title: Human-readable title for the thread (e.g. ``"Post by @user"``).
        main_item: The root post that started the conversation.
        continuation_items: The main author's own follow-up posts that
            continue the root post (e.g. an X thread). Rendered as part of
            the post body, separated by horizontal rules, not as comments.
        items: Additional items in the thread (replies, comments, etc.).
            Items express reply relationships via their ``parent_id`` field.
    """

    title: str
    main_item: ConversationItem
    items: list[ConversationItem] = field(default_factory=list)
    continuation_items: list[ConversationItem] = field(default_factory=list)
