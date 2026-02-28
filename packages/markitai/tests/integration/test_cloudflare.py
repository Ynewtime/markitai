"""Integration tests for Cloudflare BR + Workers AI.

These tests call real CF APIs and require credentials.
They are protected by TWO skip mechanisms:

1. @pytest.mark.network — CI can exclude with: -m "not network"
2. requires_cf_credentials fixture — auto-skips when env vars missing

Run explicitly:
    pytest tests/integration/test_cloudflare.py -m network -v

Requires env vars:
    CLOUDFLARE_API_TOKEN  — CF API token with BR Edit + Workers AI Read
    CLOUDFLARE_ACCOUNT_ID — CF account ID
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.network, pytest.mark.slow]

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def requires_cf_credentials():
    """Skip all tests in this module when CF credentials are not configured."""
    token = os.environ.get("CLOUDFLARE_API_TOKEN")
    account = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if not token or not account:
        pytest.skip(
            "Skipping CF integration test: "
            "CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID env vars required"
        )


@pytest.fixture
def cf_token() -> str:
    return os.environ["CLOUDFLARE_API_TOKEN"]


@pytest.fixture
def cf_account() -> str:
    return os.environ["CLOUDFLARE_ACCOUNT_ID"]


class TestCloudflareBRIntegration:
    """CF Browser Rendering /markdown API integration tests."""

    @pytest.mark.asyncio
    async def test_br_basic_url_fetch(self, cf_token, cf_account):
        """Fetch a simple URL via CF BR and get markdown."""
        from markitai.fetch import fetch_with_cloudflare

        result = await fetch_with_cloudflare(
            url="https://example.com",
            api_token=cf_token,
            account_id=cf_account,
        )
        assert result.content
        assert result.strategy_used == "cloudflare"

    @pytest.mark.asyncio
    async def test_br_with_user_agent(self, cf_token, cf_account):
        """CF BR respects custom userAgent."""
        from markitai.fetch import fetch_with_cloudflare

        result = await fetch_with_cloudflare(
            url="https://httpbin.org/user-agent",
            api_token=cf_token,
            account_id=cf_account,
            user_agent="MarkitaiTest/1.0",
        )
        assert "MarkitaiTest" in result.content

    @pytest.mark.asyncio
    async def test_br_with_cache_ttl(self, cf_token, cf_account):
        """CF BR cache_ttl > 0 returns cached result."""
        from markitai.fetch import fetch_with_cloudflare

        result = await fetch_with_cloudflare(
            url="https://example.com",
            api_token=cf_token,
            account_id=cf_account,
            cache_ttl=60,
        )
        assert result.content


class TestCloudflareConverterIntegration:
    """CF Workers AI toMarkdown integration tests."""

    @pytest.mark.asyncio
    async def test_convert_pdf(self, cf_token, cf_account):
        """Convert PDF fixture via CF toMarkdown."""
        from markitai.converter.cloudflare import CloudflareConverter

        pdf_path = FIXTURES_DIR / "sample.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Fixture not found: {pdf_path}")

        converter = CloudflareConverter(api_token=cf_token, account_id=cf_account)
        result = await converter.convert_async(pdf_path)
        assert result.markdown
        assert result.metadata["converter"] == "cloudflare-tomarkdown"

    @pytest.mark.asyncio
    async def test_convert_docx(self, cf_token, cf_account):
        """Convert DOCX fixture via CF toMarkdown."""
        from markitai.converter.cloudflare import CloudflareConverter

        docx_path = FIXTURES_DIR / "sample.docx"
        if not docx_path.exists():
            pytest.skip(f"Fixture not found: {docx_path}")

        converter = CloudflareConverter(api_token=cf_token, account_id=cf_account)
        result = await converter.convert_async(docx_path)
        assert result.markdown
