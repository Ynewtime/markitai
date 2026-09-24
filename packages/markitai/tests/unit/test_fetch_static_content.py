"""Static strategy: charset, redirect base URL and non-HTML routing.

Regression tests for E2E findings: meta-charset pages decoded as mojibake,
relative links resolved against the pre-redirect URL, PDFs/DOCX/XLSX served
without an extension handed to markitdown (or treated as HTML), and short
plain-text documents failing AUTO as "requires JavaScript".
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.config import FetchConfig
from markitai.fetch import fetch_url
from markitai.fetch_http import StaticHttpResponse
from markitai.fetch_strategies.static import (
    _response_suffix,
    fetch_with_static_conditional,
)
from markitai.fetch_types import FetchError, FetchStrategy

FIXTURES = Path(__file__).parent.parent / "fixtures"
PARA = (
    "This paragraph is part of a long article used to exercise the static "
    "extractor. It has enough words to count as real prose, with several "
    "sentences that describe nothing in particular but read naturally. "
)


def _client(response: StaticHttpResponse) -> MagicMock:
    client = MagicMock()
    client.name = "httpx"
    client.get = AsyncMock(return_value=response)
    return client


async def _conditional(response: StaticHttpResponse, url: str):
    with (
        patch("markitai.fetch_strategies.static._detect_proxy", return_value=""),
        patch(
            "markitai.fetch_strategies.static.get_static_http_client",
            return_value=_client(response),
        ),
    ):
        return await fetch_with_static_conditional(url)


def _article(title: str, body: str, head: str = "") -> str:
    return (
        f"<!doctype html><html><head>{head}<title>{title}</title></head>"
        f"<body><article><h1>{title}</h1>{body}</article></body></html>"
    )


class TestCharsetEndToEnd:
    @pytest.mark.asyncio
    async def test_meta_charset_page_without_header_charset(self):
        html = _article(
            "中文页面",
            "".join(
                f"<p>第{i}段：朱镕基总理的讲话内容，需要足够长的正文才能通过提取。</p>"
                for i in range(8)
            ),
            head='<meta charset="gbk">',
        )
        response = StaticHttpResponse(
            content=html.encode("gbk"),
            status_code=200,
            headers={"content-type": "text/html"},
            url="https://example.cn/a",
        )
        result = (await _conditional(response, "https://example.cn/a")).result
        assert result is not None
        assert "朱镕基总理的讲话内容" in result.content
        assert "�" not in result.content

    @pytest.mark.asyncio
    async def test_markitdown_fallback_gets_the_decoded_text(self):
        """A page too short for native extraction still decodes correctly."""
        html = '<html><head><meta charset="gbk"></head><body><p>朱镕基讲话</p></body></html>'
        response = StaticHttpResponse(
            content=html.encode("gbk"),
            status_code=200,
            headers={"content-type": "text/html"},
            url="https://example.cn/short",
        )
        result = (await _conditional(response, "https://example.cn/short")).result
        assert result is not None
        assert result.metadata["converter"] == "markitdown"
        assert "朱镕基讲话" in result.content

    @pytest.mark.asyncio
    async def test_plain_text_gbk_is_reencoded_before_conversion(self):
        response = StaticHttpResponse(
            content="纯文本：朱镕基。\n".encode("gbk"),
            status_code=200,
            headers={"content-type": "text/plain; charset=gbk"},
            url="https://example.cn/note",
        )
        result = (await _conditional(response, "https://example.cn/note")).result
        assert result is not None
        assert "朱镕基" in result.content
        assert result.metadata["content_kind"] == "text"


class TestRedirectBaseUrl:
    @pytest.mark.asyncio
    async def test_relative_links_resolve_against_final_url(self):
        body = "".join(f"<p>{PARA}</p>" for _ in range(5)) + (
            '<p>See <a href="page2">next</a> <img src="img.png" alt="d"></p>'
        )
        response = StaticHttpResponse(
            content=_article("Docs", body).encode(),
            status_code=200,
            headers={"content-type": "text/html; charset=utf-8"},
            url="https://example.com/docs/",  # after the /docs -> /docs/ 301
        )
        result = (await _conditional(response, "https://example.com/docs")).result
        assert result is not None
        assert "(https://example.com/docs/page2)" in result.content
        assert "(https://example.com/docs/img.png)" in result.content
        assert result.url == "https://example.com/docs"
        assert result.final_url == "https://example.com/docs/"


class TestNonHtmlRouting:
    @pytest.mark.parametrize(
        ("content_type", "disposition", "url", "body", "suffix"),
        [
            ("application/pdf", None, "https://x/doc", b"", ".pdf"),
            (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document",
                None,
                "https://x/get?id=1",
                b"",
                ".docx",
            ),
            (
                "application/octet-stream",
                'attachment; filename="report.xlsx"',
                "https://x/download",
                b"",
                ".xlsx",
            ),
            (
                "application/octet-stream",
                "attachment; filename*=UTF-8''%E6%8A%A5%E5%91%8A.pdf",
                "https://x/download",
                b"",
                ".pdf",
            ),
            ("text/plain", None, "https://x/README.md", b"", ".md"),
            ("text/plain", None, "https://x/notes", b"", ".txt"),
            ("", None, "https://x/blob", b"%PDF-1.7\n", ".pdf"),
            ("text/html; charset=utf-8", None, "https://x/file.pdf", b"", ".html"),
            ("", None, "https://x/", b"<html></html>", ".html"),
        ],
    )
    def test_suffix_selection(self, content_type, disposition, url, body, suffix):
        assert _response_suffix(content_type, disposition, (url,), body)[0] == suffix

    def test_zip_container_is_sniffed(self):
        body = (FIXTURES / "sample.docx").read_bytes()
        assert _response_suffix("", None, ("https://x/blob",), body)[0] == ".docx"

    @staticmethod
    def _zip(mimetype: bytes) -> bytes:
        import io
        import zipfile

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("mimetype", mimetype)
            archive.writestr("content.xml", "<x/>")
        return buffer.getvalue()

    def test_zip_mimetype_entry_is_sniffed(self):
        body = self._zip(b"application/epub+zip")
        assert _response_suffix("", None, ("https://x/blob",), body)[0] == ".epub"

    def test_oversized_zip_mimetype_entry_is_not_inflated(self):
        """A deflate bomb posing as the mimetype entry is skipped unread."""
        body = self._zip(b"application/epub+zip" + b" " * (64 * 1024 * 1024))
        assert len(body) < 1024 * 1024  # compresses to almost nothing
        with patch("zipfile.ZipFile.read", side_effect=AssertionError("inflated")):
            suffix = _response_suffix("", None, ("https://x/blob",), body)[0]
        assert suffix != ".epub"

    @pytest.mark.asyncio
    async def test_pdf_url_uses_markitai_pdf_converter(self):
        response = StaticHttpResponse(
            content=(FIXTURES / "sample.pdf").read_bytes(),
            status_code=200,
            headers={"content-type": "application/pdf"},
            url="https://example.com/paper",
        )
        with patch("markitai.fetch_strategies._shared._get_markitdown") as markitdown:
            result = (await _conditional(response, "https://example.com/paper")).result
        markitdown.assert_not_called()
        assert result is not None
        assert "<!-- Page number: 1 -->" in result.content  # markitai PDF output
        assert result.metadata["converter"] == "markitai"
        assert result.metadata["content_kind"] == "document"

    @pytest.mark.asyncio
    async def test_docx_without_extension_is_not_treated_as_html(self):
        response = StaticHttpResponse(
            content=(FIXTURES / "sample.docx").read_bytes(),
            status_code=200,
            headers={
                "content-type": "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            },
            url="https://example.com/export",
        )
        result = (await _conditional(response, "https://example.com/export")).result
        assert result is not None
        assert "PK" not in result.content[:10]
        assert "Markitai Snapshot Fixture" in result.content

    @pytest.mark.asyncio
    async def test_xlsx_behind_octet_stream_and_disposition(self):
        response = StaticHttpResponse(
            content=(FIXTURES / "sample.xlsx").read_bytes(),
            status_code=200,
            headers={
                "content-type": "application/octet-stream",
                "content-disposition": 'attachment; filename="people.xlsx"',
            },
            url="https://example.com/download",
        )
        result = (await _conditional(response, "https://example.com/download")).result
        assert result is not None
        assert "|" in result.content  # a Markdown table, not binary noise
        assert "First Name" in result.content


class TestAutoNonHtml:
    @pytest.mark.asyncio
    async def test_short_plain_text_is_accepted_without_spa_learning(self):
        response = StaticHttpResponse(
            content=b"Just a short note.\n",
            status_code=200,
            headers={"content-type": "text/plain; charset=utf-8"},
            url="https://example.com/note.txt",
        )
        spa_cache = MagicMock()
        spa_cache.is_known_spa.return_value = False
        with (
            patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
            patch("markitai.fetch_strategies.static._detect_proxy", return_value=""),
            patch(
                "markitai.fetch_strategies.static.get_static_http_client",
                return_value=_client(response),
            ),
            patch(
                "markitai.fetch_strategies.playwright.PlaywrightRunner.fetch",
                new_callable=AsyncMock,
            ) as browser,
        ):
            result = await fetch_url(
                "https://example.com/note.txt",
                FetchStrategy.AUTO,
                FetchConfig(),
                skip_read_cache=True,
            )
        assert "Just a short note." in result.content
        assert result.strategy_used == "static"
        browser.assert_not_awaited()
        spa_cache.record_spa_domain.assert_not_called()

    @pytest.mark.asyncio
    async def test_short_html_still_learns_spa(self):
        response = StaticHttpResponse(
            content=b"<html><body><div id='root'>Loading...</div></body></html>",
            status_code=200,
            headers={"content-type": "text/html"},
            url="https://spa.example.com/",
        )
        spa_cache = MagicMock()
        spa_cache.is_known_spa.return_value = False
        config = FetchConfig()
        config.policy.strategy_priority = ["static"]
        with (
            patch("markitai.fetch.get_spa_domain_cache", return_value=spa_cache),
            patch("markitai.fetch_strategies.static._detect_proxy", return_value=""),
            patch(
                "markitai.fetch_strategies.static.get_static_http_client",
                return_value=_client(response),
            ),
            pytest.raises(FetchError, match="requires JavaScript"),
        ):
            await fetch_url(
                "https://spa.example.com/",
                FetchStrategy.AUTO,
                config,
                skip_read_cache=True,
            )
        spa_cache.record_spa_domain.assert_called_once_with("https://spa.example.com/")
