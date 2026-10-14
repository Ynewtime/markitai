"""X's t.co shortlinks resolved by one bounded, cached HEAD request each."""

from __future__ import annotations

import asyncio

import httpx
import pytest


class TestShortlinks:
    @pytest.fixture(autouse=True)
    def _fresh_cache(self):
        from markitai import fetch_shortlinks

        fetch_shortlinks._resolved.clear()
        yield
        fetch_shortlinks._resolved.clear()

    async def test_resolved_once_and_replaced_everywhere(self):
        from markitai.fetch_shortlinks import expand_shortlinks

        calls: list[str] = []

        def answer(request: httpx.Request) -> httpx.Response:
            calls.append(f"{request.method} {request.url}")
            if request.url.path == "/AbCd1234":
                return httpx.Response(
                    301, headers={"location": "https://example.org/lamp-benchmarks"}
                )
            return httpx.Response(200)

        transport = httpx.MockTransport(answer)
        markdown = "See <https://t.co/AbCd1234> or [here](https://t.co/AbCd1234); https://t.co/Zz99Yy88"
        out = await expand_shortlinks(markdown, transport=transport)
        assert out == (
            "See <https://example.org/lamp-benchmarks> or "
            "[here](https://example.org/lamp-benchmarks); https://t.co/Zz99Yy88"
        )
        assert sorted(calls) == [
            "HEAD https://t.co/AbCd1234",
            "HEAD https://t.co/Zz99Yy88",
        ]
        # Kept for the process, failures included: nothing is asked again
        assert await expand_shortlinks(markdown, transport=transport) == out
        assert len(calls) == 2

    async def test_bounded_in_links_and_time(self, monkeypatch):
        from markitai import fetch_shortlinks

        monkeypatch.setattr(fetch_shortlinks, "BUDGET_S", 0.2)
        asked: list[str] = []

        async def slow(request: httpx.Request) -> httpx.Response:
            asked.append(str(request.url))
            await asyncio.sleep(5)
            return httpx.Response(301, headers={"location": "https://example.org/"})

        links = " ".join(f"https://t.co/Link{n:04d}" for n in range(30))
        out = await fetch_shortlinks.expand_shortlinks(
            links, transport=httpx.MockTransport(slow)
        )
        assert out == links
        assert len(asked) <= fetch_shortlinks.CONCURRENCY
        assert fetch_shortlinks.MAX_LINKS == 20

    async def test_a_redirect_back_to_tco_is_not_a_destination(self):
        from markitai.fetch_shortlinks import expand_shortlinks

        transport = httpx.MockTransport(
            lambda _request: httpx.Response(
                301, headers={"location": "https://t.co/Other123"}
            )
        )
        assert await expand_shortlinks(
            "https://t.co/Loop1234", transport=transport
        ) == ("https://t.co/Loop1234")

    def test_only_x_pages_are_expanded(self):
        from markitai.fetch_shortlinks import is_x_url

        assert is_x_url("https://x.com/user/status/1")
        assert is_x_url("https://mobile.twitter.com/user/status/1")
        assert not is_x_url("https://example.com/x.com/status/1")
