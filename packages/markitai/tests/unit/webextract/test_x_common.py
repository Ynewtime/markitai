"""Tests for X/Twitter shared parsing utilities (defuddle-parity features)."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag

from markitai.webextract.extractors.x_common import parse_tweet_article


def _article(html: str) -> Tag:
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    assert isinstance(article, Tag)
    return article


_QUOTED_TWEET_HTML = """
<article data-testid="tweet">
  <div data-testid="User-Name">
    <a href="/dotey"><span>Baoyu</span></a>
    <a href="/dotey"><span>@dotey</span></a>
  </div>
  <div data-testid="tweetText"><span>Main tweet text</span></div>
  <div data-testid="tweetPhoto">
    <img src="https://pbs.twimg.com/media/MAIN?format=jpg&name=small" alt="main img"/>
  </div>
  <div aria-labelledby="id__aaa id__bbb">
    <div data-testid="User-Name">
      <div><span>OpenAI</span></div>
      <div><span>@OpenAI</span><span>&#183;</span>
        <time datetime="2025-12-23T08:00:00.000Z">Dec 23</time></div>
    </div>
    <div data-testid="tweetText"><span>Quoted tweet text</span></div>
    <div data-testid="tweetPhoto">
      <img src="https://pbs.twimg.com/media/QUOTE?format=jpg&name=small" alt="Image"/>
    </div>
  </div>
  <time datetime="2025-12-24T10:00:00.000Z">10:00 AM</time>
</article>
"""


class TestQuotedTweet:
    def test_quote_detected_via_aria_labelledby(self) -> None:
        item = parse_tweet_article(_article(_QUOTED_TWEET_HTML), tweet_id="1")
        assert item.quoted_item is not None
        assert item.quoted_item.author_name == "OpenAI"
        assert item.quoted_item.author_handle == "@OpenAI"
        assert item.quoted_item.text == "Quoted tweet text"
        assert item.quoted_item.timestamp == "2025-12-23T08:00:00.000Z"

    def test_quote_media_not_attributed_to_parent(self) -> None:
        item = parse_tweet_article(_article(_QUOTED_TWEET_HTML), tweet_id="1")
        parent_urls = [m.url for m in item.media]
        assert parent_urls == ["https://pbs.twimg.com/media/MAIN?format=jpg&name=large"]
        assert item.quoted_item is not None
        quote_urls = [m.url for m in item.quoted_item.media]
        assert quote_urls == ["https://pbs.twimg.com/media/QUOTE?format=jpg&name=large"]

    def test_quote_text_does_not_leak_into_parent_text(self) -> None:
        item = parse_tweet_article(_article(_QUOTED_TWEET_HTML), tweet_id="1")
        assert "Quoted tweet text" not in item.text

    def test_main_timestamp_not_taken_from_quote(self) -> None:
        item = parse_tweet_article(_article(_QUOTED_TWEET_HTML), tweet_id="1")
        assert item.timestamp == "2025-12-24T10:00:00.000Z"


class TestMedia:
    def test_image_url_upgraded_to_large(self) -> None:
        html = """
        <article data-testid="tweet">
          <div data-testid="tweetText"><span>t</span></div>
          <div data-testid="tweetPhoto">
            <img src="https://pbs.twimg.com/media/X?format=jpg&name=900x900" alt="pic"/>
          </div>
        </article>
        """
        item = parse_tweet_article(_article(html))
        assert (
            item.media[0].url == "https://pbs.twimg.com/media/X?format=jpg&name=large"
        )

    def test_video_extracted_with_poster(self) -> None:
        html = """
        <article data-testid="tweet">
          <div data-testid="tweetText"><span>t</span></div>
          <video poster="https://pbs.twimg.com/thumb.jpg"
                 src="https://video.twimg.com/vid.mp4"></video>
        </article>
        """
        item = parse_tweet_article(_article(html))
        assert len(item.media) == 1
        assert item.media[0].media_type == "video"
        assert item.media[0].url == "https://video.twimg.com/vid.mp4"
        assert item.media[0].poster == "https://pbs.twimg.com/thumb.jpg"

    def test_avatar_and_emoji_images_skipped(self) -> None:
        html = """
        <article data-testid="tweet">
          <div data-testid="tweetText"><span>t</span></div>
          <img src="https://pbs.twimg.com/profile_images/1/avatar.jpg" alt=""/>
          <img src="https://abs-0.twimg.com/emoji/v2/svg/1f600.svg" alt="grin"/>
        </article>
        """
        item = parse_tweet_article(_article(html))
        assert item.media == []


class TestCard:
    def test_card_link_extracted_with_aria_label_title(self) -> None:
        html = """
        <article data-testid="tweet">
          <div data-testid="tweetText"><span>t</span></div>
          <div data-testid="card.wrapper">
            <a href="https://t.co/card123" aria-label="Example Site
Some description">
              <img src="https://pbs.twimg.com/card_img/999?name=small" alt=""/>
            </a>
          </div>
        </article>
        """
        item = parse_tweet_article(_article(html))
        assert item.card_url == "https://t.co/card123"
        assert item.card_title == "Example Site"
        # Card thumbnail must not become a media attachment
        assert item.media == []


class TestText:
    def test_emoji_images_replaced_with_alt_text(self) -> None:
        html = """
        <article data-testid="tweet">
          <div data-testid="tweetText"><span>Hello </span>
            <img src="https://abs-0.twimg.com/emoji/v2/svg/1f600.svg" alt="&#128512;"/>
          </div>
        </article>
        """
        item = parse_tweet_article(_article(html))
        assert "\U0001f600" in item.text

    def test_newlines_inside_text_nodes_preserved_as_paragraphs(self) -> None:
        html = (
            '<article data-testid="tweet">'
            '<div data-testid="tweetText">'
            "<span>First paragraph.\n\nSecond paragraph.</span>"
            "</div></article>"
        )
        item = parse_tweet_article(_article(html))
        assert item.text == "First paragraph.\nSecond paragraph."

    def test_link_display_text_used(self) -> None:
        html = (
            '<article data-testid="tweet">'
            '<div data-testid="tweetText">'
            "<span>See </span>"
            '<a href="https://t.co/xyz"><span>example.com/page…</span></a>'
            "</div></article>"
        )
        item = parse_tweet_article(_article(html))
        assert item.text == "See example.com/page…"


class TestAuthorFallback:
    def test_quoted_tweet_author_without_links(self) -> None:
        html = """
        <article data-testid="tweet">
          <div data-testid="User-Name">
            <div><span>Display Name</span></div>
            <div><span>@handle</span><span>&#183;</span><span>Dec 23</span></div>
          </div>
          <div data-testid="tweetText"><span>t</span></div>
        </article>
        """
        item = parse_tweet_article(_article(html))
        assert item.author_name == "Display Name"
        assert item.author_handle == "@handle"
