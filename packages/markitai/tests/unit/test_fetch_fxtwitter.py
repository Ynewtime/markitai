"""Tests for FxTwitter API fetch enrichment."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from markitai.fetch_fxtwitter import (
    _build_conversation_thread,
    _extract_twitter_url_parts,
    fetch_with_fxtwitter,
)


class TestExtractTwitterUrlParts:
    def test_x_com_status_url(self) -> None:
        result = _extract_twitter_url_parts(
            "https://x.com/zty0826/status/2035899567837978794"
        )
        assert result is not None
        user, tweet_id = result
        assert user == "zty0826"
        assert tweet_id == "2035899567837978794"

    def test_twitter_com_status_url(self) -> None:
        result = _extract_twitter_url_parts(
            "https://twitter.com/elonmusk/status/123456789"
        )
        assert result is not None
        user, tweet_id = result
        assert user == "elonmusk"
        assert tweet_id == "123456789"

    def test_non_status_url_returns_none(self) -> None:
        assert _extract_twitter_url_parts("https://x.com/zty0826") is None

    def test_non_twitter_url_returns_none(self) -> None:
        assert _extract_twitter_url_parts("https://github.com/user/repo") is None


MOCK_FXTWITTER_RESPONSE = {
    "code": 200,
    "tweet": {
        "text": "This is a test tweet with some content.",
        "author": {"name": "Test User", "screen_name": "testuser"},
        "created_at": "2026-03-23T10:00:00.000Z",
        "media": {
            "all": [
                {
                    "type": "photo",
                    "url": "https://pbs.twimg.com/media/test.jpg",
                    "width": 800,
                    "height": 600,
                }
            ]
        },
    },
}


class TestBuildConversationThread:
    def test_basic_tweet(self) -> None:
        thread = _build_conversation_thread(
            MOCK_FXTWITTER_RESPONSE["tweet"],
            tweet_id="123",
            url="https://x.com/testuser/status/123",
        )
        assert thread.main_item.author_name == "Test User"
        assert thread.main_item.author_handle == "@testuser"
        assert "test tweet" in thread.main_item.text
        assert len(thread.main_item.media) == 1

    def test_tweet_with_quoted_tweet(self) -> None:
        data = {
            **MOCK_FXTWITTER_RESPONSE["tweet"],
            "quote": {
                "text": "Quoted tweet.",
                "author": {"name": "Q", "screen_name": "quser"},
            },
        }
        thread = _build_conversation_thread(
            data, tweet_id="456", url="https://x.com/testuser/status/456"
        )
        assert thread.main_item.quoted_item is not None
        assert thread.main_item.quoted_item.author_handle == "@quser"


@pytest.mark.asyncio
class TestFetchWithFxtwitter:
    async def test_successful_fetch(self) -> None:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        # httpx Response.json() is synchronous, not async
        mock_response.json = lambda: MOCK_FXTWITTER_RESPONSE
        mock_response.raise_for_status = lambda: None

        with patch("markitai.fetch_fxtwitter.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.return_value = mock_response
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance
            result = await fetch_with_fxtwitter("https://x.com/testuser/status/123")
            assert result is not None
            assert result.strategy_used == "fxtwitter"
            assert "test tweet" in result.content
            assert "@testuser" in result.content

    async def test_non_twitter_url_returns_none(self) -> None:
        result = await fetch_with_fxtwitter("https://github.com/user/repo")
        assert result is None

    async def test_api_failure_returns_none(self) -> None:
        with patch("markitai.fetch_fxtwitter.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.side_effect = Exception("timeout")
            client_instance.__aenter__ = AsyncMock(return_value=client_instance)
            client_instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = client_instance
            result = await fetch_with_fxtwitter("https://x.com/testuser/status/123")
            assert result is None

    async def test_api_failure_logs_url_and_traceback(self) -> None:
        """Regression: failure log must include the URL and the exception."""
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(
            lambda m: messages.append(str(m)), level="DEBUG", format="{message}"
        )
        try:
            with patch("markitai.fetch_fxtwitter.httpx.AsyncClient") as MockClient:
                client_instance = AsyncMock()
                client_instance.get.side_effect = Exception("timeout")
                client_instance.__aenter__ = AsyncMock(return_value=client_instance)
                client_instance.__aexit__ = AsyncMock(return_value=False)
                MockClient.return_value = client_instance
                result = await fetch_with_fxtwitter("https://x.com/testuser/status/123")
                assert result is None
        finally:
            logger.remove(sink_id)

        combined = "\n".join(messages)
        assert "https://x.com/testuser/status/123" in combined
        assert "timeout" in combined


@pytest.mark.asyncio
async def test_dispatch_strategy_tries_fxtwitter_before_playwright() -> None:
    """_dispatch_strategy should attempt FxTwitter before Playwright for x.com URLs."""
    from markitai.config import FetchConfig
    from markitai.fetch_types import FetchResult, FetchStrategy

    mock_fxtwitter_result = FetchResult(
        content="# FxTwitter content",
        strategy_used="fxtwitter",
        title="Test tweet",
        url="https://x.com/user/status/123",
    )

    with patch(
        "markitai.fetch_fxtwitter.fetch_with_fxtwitter",
        new_callable=AsyncMock,
        return_value=mock_fxtwitter_result,
    ) as mock_fx:
        from markitai.fetch import _dispatch_strategy

        result, _ = await _dispatch_strategy(
            url="https://x.com/user/status/123",
            strategy=FetchStrategy.PLAYWRIGHT,
            config=FetchConfig(),
            explicit_strategy=False,
            screenshot_kwargs={},
            screenshot_config=None,
            screenshot_dir=None,
            renderer=None,
        )
        mock_fx.assert_called_once()
        assert result.strategy_used == "fxtwitter"
