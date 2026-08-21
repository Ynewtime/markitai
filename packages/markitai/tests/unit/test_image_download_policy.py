"""Network policy for `download_url_images`: proxying and SSRF.

Two defects, one call path:

1. `_detect_proxy_for_images()` was a third, private copy of proxy detection
   that had never heard of NO_PROXY. Image downloads therefore ignored the
   exception list every other fetch honours.
2. The image URLs come from *converted page content*. A public page that
   embeds `<img src="http://192.168.1.1/...">` made markitai fetch that
   address on the attacker's behalf — a second-order SSRF that walks straight
   past the private-network gate `markitai serve` applies to submitted URLs.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import httpx
import pytest
from PIL import Image

from markitai.config import ImageConfig


def _png(width: int = 400, height: int = 400) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), "red").save(buffer, format="PNG")
    return buffer.getvalue()


class _RecordingTransport:
    """Stand-in for httpx.AsyncClient that records every URL requested."""

    def __init__(self) -> None:
        self.requested: list[str] = []
        self.client_kwargs: list[dict[str, Any]] = []
        self.redirects: dict[str, str] = {}

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        transport = self
        real_init = httpx.AsyncClient.__init__

        def fake_init(client: httpx.AsyncClient, **kwargs: Any) -> None:
            transport.client_kwargs.append(kwargs)
            real_init(client, **kwargs)

        async def fake_get(_client, url, **kwargs: Any):
            url = str(url)
            transport.requested.append(url)
            target = transport.redirects.get(url)
            if target is not None:
                if kwargs.get("follow_redirects"):
                    return _Response(200, _png(), url=target)
                return _Response(302, b"", url=url, location=target)
            return _Response(200, _png(), url=url)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", fake_init)
        monkeypatch.setattr(httpx.AsyncClient, "get", fake_get)


class _Response:
    def __init__(
        self, status_code: int, content: bytes, *, url: str, location: str | None = None
    ) -> None:
        self.status_code = status_code
        self.content = content
        self.url = url
        self.headers = {"content-type": "image/png"}
        if location:
            self.headers["location"] = location

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)  # type: ignore[arg-type]

    @property
    def is_redirect(self) -> bool:
        return self.status_code in (301, 302, 303, 307, 308)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> _RecordingTransport:
    recorder = _RecordingTransport()
    recorder.install(monkeypatch)
    return recorder


async def _download(tmp_path: Path, markdown: str, base_url: str, **kwargs: Any):
    from markitai.image import download_url_images

    return await download_url_images(
        markdown=markdown,
        output_dir=tmp_path,
        base_url=base_url,
        config=ImageConfig(),
        **kwargs,
    )


class TestSecondOrderSsrf:
    """A public page must not be able to aim the fetcher at the LAN."""

    @pytest.mark.parametrize(
        "target",
        [
            "http://192.168.1.1/admin.png",
            "http://10.0.0.5/x.png",
            "http://127.0.0.1:8080/x.png",
            "http://169.254.169.254/latest/meta-data/x.png",  # cloud metadata
            "http://localhost/x.png",
            "http://nas.local/x.png",
        ],
    )
    async def test_private_target_from_public_page_is_refused(
        self, tmp_path: Path, transport: _RecordingTransport, target: str
    ) -> None:
        markdown = f"![shot]({target})"
        result = await _download(tmp_path, markdown, "https://example.com/post")

        assert transport.requested == [], "the request must never leave the process"
        assert result.failed_urls == [target]
        assert result.downloaded_paths == []
        assert result.updated_markdown == markdown

    async def test_public_target_from_public_page_is_downloaded(
        self, tmp_path: Path, transport: _RecordingTransport
    ) -> None:
        target = "https://cdn.example.com/a.png"
        result = await _download(
            tmp_path, f"![a]({target})", "https://example.com/post"
        )

        assert transport.requested == [target]
        assert len(result.downloaded_paths) == 1
        assert result.failed_urls == []

    async def test_intranet_page_may_load_its_own_intranet_images(
        self, tmp_path: Path, transport: _RecordingTransport
    ) -> None:
        """CLI users converting an intranet wiki are not the threat model.

        The page itself was already a private target the user chose, so its
        images are consistent with that intent — and `markitai serve` already
        refuses private *page* URLs from non-loopback peers, so this branch is
        unreachable for a remote attacker.
        """
        target = "http://wiki.corp/logo.png"
        result = await _download(tmp_path, f"![l]({target})", "http://wiki.corp/page")

        assert transport.requested == [target]
        assert len(result.downloaded_paths) == 1

    async def test_relative_image_on_an_intranet_page_still_works(
        self, tmp_path: Path, transport: _RecordingTransport
    ) -> None:
        result = await _download(
            tmp_path, "![l](/static/logo.png)", "http://192.168.1.50/page"
        )

        assert transport.requested == ["http://192.168.1.50/static/logo.png"]
        assert len(result.downloaded_paths) == 1

    async def test_explicit_override_allows_private_targets(
        self, tmp_path: Path, transport: _RecordingTransport
    ) -> None:
        target = "http://192.168.1.1/x.png"
        result = await _download(
            tmp_path,
            f"![a]({target})",
            "https://example.com",
            allow_private_targets=True,
        )

        assert transport.requested == [target]
        assert len(result.downloaded_paths) == 1

    async def test_explicit_override_can_also_tighten_an_intranet_page(
        self, tmp_path: Path, transport: _RecordingTransport
    ) -> None:
        result = await _download(
            tmp_path,
            "![a](http://192.168.1.1/x.png)",
            "http://wiki.corp/page",
            allow_private_targets=False,
        )

        assert transport.requested == []
        assert len(result.failed_urls) == 1

    async def test_redirect_into_the_private_range_is_refused(
        self, tmp_path: Path, transport: _RecordingTransport
    ) -> None:
        """A public URL that 302s to the LAN is the same attack, one hop later."""
        entry = "https://evil.example.com/pixel.png"
        transport.redirects[entry] = "http://169.254.169.254/latest/meta-data/"

        result = await _download(tmp_path, f"![a]({entry})", "https://example.com")

        assert result.downloaded_paths == []
        assert result.failed_urls == [entry]
        assert "169.254.169.254" not in "".join(transport.requested)

    async def test_non_http_scheme_is_refused(
        self, tmp_path: Path, transport: _RecordingTransport
    ) -> None:
        markdown = "![a](ftp://example.com/x.png)"
        result = await _download(tmp_path, markdown, "https://example.com")

        assert transport.requested == []
        assert result.downloaded_paths == []


class TestProxyResolution:
    """One proxy entry point, NO_PROXY included."""

    def test_private_proxy_helper_is_gone(self) -> None:
        import markitai.image as image_module

        assert not hasattr(image_module, "_detect_proxy_for_images")

    async def test_proxy_is_applied_to_a_normal_host(
        self, tmp_path: Path, transport: _RecordingTransport, monkeypatch
    ) -> None:
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
        monkeypatch.delenv("NO_PROXY", raising=False)
        monkeypatch.delenv("no_proxy", raising=False)
        _reset_proxy_cache()

        await _download(
            tmp_path, "![a](https://cdn.example.com/a.png)", "https://example.com"
        )

        proxies = [kwargs.get("proxy") for kwargs in transport.client_kwargs]
        assert "http://127.0.0.1:7890" in proxies

    async def test_no_proxy_exempts_the_image_host(
        self, tmp_path: Path, transport: _RecordingTransport, monkeypatch
    ) -> None:
        """The whole point: image downloads honour the exception list too."""
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
        monkeypatch.setenv("NO_PROXY", "cdn.example.com")
        _reset_proxy_cache()

        await _download(
            tmp_path, "![a](https://cdn.example.com/a.png)", "https://example.com"
        )

        proxies = [kwargs.get("proxy") for kwargs in transport.client_kwargs]
        assert proxies, "no client was constructed"
        assert all(proxy is None for proxy in proxies), proxies

    async def test_no_proxy_decision_is_per_url_not_per_batch(
        self, tmp_path: Path, transport: _RecordingTransport, monkeypatch
    ) -> None:
        """One document can mix exempt and proxied hosts."""
        monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
        monkeypatch.setenv("NO_PROXY", "exempt.example.com")
        _reset_proxy_cache()

        markdown = (
            "![a](https://exempt.example.com/a.png)\n"
            "![b](https://proxied.example.com/b.png)"
        )
        await _download(tmp_path, markdown, "https://example.com")

        proxies = {kwargs.get("proxy") for kwargs in transport.client_kwargs}
        assert proxies == {None, "http://127.0.0.1:7890"}


def _reset_proxy_cache() -> None:
    """Drop the process-wide proxy detection cache between env manipulations."""
    from markitai.fetch_session import get_default_session

    session = get_default_session()
    session.detected_proxy = None
    session.detected_proxy_bypass = ""
