"""Fake-IP compatibility is limited to consented remote URL sharing."""

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from markitai.config import FetchConfig
from markitai.fetch import _ensure_external_strategy_allowed
from markitai.fetch_policy import (
    assess_url_for_remote,
    public_network_only,
    resolve_public_hostname_addresses,
)
from markitai.fetch_types import FetchError

URL = "https://example.com/article"
FAKE_IPS = ("198.18.0.158", "2001:2::9e")
PUBLIC_IPS = ("93.184.216.34", "2606:4700::1111")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "addresses",
    [FAKE_IPS, (FAKE_IPS[0],), (FAKE_IPS[1],), (FAKE_IPS[0], PUBLIC_IPS[1])],
)
async def test_fake_ip_requires_independent_public_confirmation(addresses):
    public = AsyncMock(return_value=PUBLIC_IPS)
    assessment = await assess_url_for_remote(
        URL, resolver=AsyncMock(return_value=addresses), public_resolver=public
    )
    assert assessment.allowed
    public.assert_awaited_once_with("example.com")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url, addresses",
    [
        (URL, (FAKE_IPS[0], "10.0.0.1")),
        (URL, (FAKE_IPS[1], "::1")),
        (URL, ("192.168.1.1",)),
        (URL, ("100.64.0.1",)),
        (URL, ("fd00::1",)),
        (URL, ("invalid",)),
        ("https://198.18.0.158/article", FAKE_IPS),
        ("https://[2001:2::9e]/article", FAKE_IPS),
        ("https://portal.local/article", FAKE_IPS),
        ("https://example.com/article?token=secret", FAKE_IPS),
    ],
)
async def test_private_targets_never_use_public_dns(url, addresses):
    public = AsyncMock(return_value=PUBLIC_IPS)
    assessment = await assess_url_for_remote(
        url, resolver=AsyncMock(return_value=addresses), public_resolver=public
    )
    assert not assessment.allowed
    public.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("answer", [(), ("10.0.0.1",), FAKE_IPS, (*PUBLIC_IPS, "::1")])
async def test_non_public_verification_fails_closed(answer):
    assessment = await assess_url_for_remote(
        URL,
        resolver=AsyncMock(return_value=FAKE_IPS),
        public_resolver=AsyncMock(return_value=answer),
    )
    assert not assessment.allowed


@pytest.mark.asyncio
async def test_verification_error_fails_closed():
    assessment = await assess_url_for_remote(
        URL,
        resolver=AsyncMock(return_value=FAKE_IPS),
        public_resolver=AsyncMock(side_effect=OSError("DNS unavailable")),
    )
    assert assessment.reason == "hostname_resolution_failed"


@pytest.mark.asyncio
async def test_normal_dns_needs_no_extra_network_request():
    public = AsyncMock()
    assessment = await assess_url_for_remote(
        URL, resolver=AsyncMock(return_value=PUBLIC_IPS), public_resolver=public
    )
    assert assessment.allowed
    public.assert_not_awaited()


@pytest.mark.asyncio
async def test_default_and_serve_policies_still_reject_fake_ip():
    resolver = AsyncMock(return_value=FAKE_IPS)
    assert not (await assess_url_for_remote(URL, resolver=resolver)).allowed
    public = AsyncMock(return_value=PUBLIC_IPS)
    token = public_network_only.set(True)
    try:
        assert not (
            await assess_url_for_remote(URL, resolver=resolver, public_resolver=public)
        ).allowed
    finally:
        public_network_only.reset(token)
    public.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("strategy", ["defuddle", "jina", "cloudflare"])
@pytest.mark.parametrize("explicit, consent", [(True, "never"), (False, "always")])
async def test_remote_strategy_verifies_fake_ip_after_consent(
    strategy, explicit, consent
):
    with (
        patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            AsyncMock(return_value=FAKE_IPS),
        ),
        patch(
            "markitai.fetch_policy.resolve_public_hostname_addresses",
            AsyncMock(return_value=PUBLIC_IPS),
        ) as public,
    ):
        await _ensure_external_strategy_allowed(
            URL,
            strategy,
            config=FetchConfig(remote_consent=consent),
            allow_pattern_override=explicit,
        )
    public.assert_awaited_once_with("example.com")


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_by", ["consent", "environment", "local_pattern"])
async def test_remote_restrictions_prevent_public_dns(monkeypatch, blocked_by):
    config = FetchConfig(
        remote_consent="never" if blocked_by == "consent" else "always"
    )
    if blocked_by == "environment":
        monkeypatch.setenv("MARKITAI_NO_REMOTE_FETCH", "1")
    elif blocked_by == "local_pattern":
        config.policy.local_only_patterns = ["example.com"]
    with (
        patch(
            "markitai.fetch_policy.resolve_hostname_addresses",
            AsyncMock(return_value=FAKE_IPS),
        ),
        patch(
            "markitai.fetch_policy.resolve_public_hostname_addresses",
            AsyncMock(return_value=PUBLIC_IPS),
        ) as public,
        pytest.raises(FetchError),
    ):
        await _ensure_external_strategy_allowed(URL, "defuddle", config=config)
    public.assert_not_awaited()


@pytest.mark.asyncio
async def test_doh_queries_both_families_and_ignores_cname():
    requests = []

    def respond(request):
        requests.append(request)
        record_type = int(request.url.params["type"])
        address = PUBLIC_IPS[0 if record_type == 1 else 1]
        return httpx.Response(
            200,
            json={
                "Status": 0,
                "Answer": [
                    {"type": 5, "data": "cdn.example.com"},
                    {"type": record_type, "data": address},
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    with patch("httpx.AsyncClient", return_value=client):
        assert await resolve_public_hostname_addresses("example.com") == PUBLIC_IPS
    assert {r.url.params["type"] for r in requests} == {"1", "28"}
    assert all(r.url.params["name"] == "example.com" for r in requests)
    assert all(r.headers["Accept"] == "application/dns-json" for r in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_response",
    [
        httpx.Response(503),
        httpx.Response(302, headers={"Location": "https://other.example/dns-query"}),
        httpx.Response(200, text="not JSON"),
        httpx.Response(200, json={"Status": 2}),
        httpx.Response(200, json={"Status": 0, "Answer": None}),
        httpx.Response(
            200, json={"Status": 0, "Answer": [{"type": 28, "data": "bad-ip"}]}
        ),
        httpx.Response(
            200, json={"Status": 0, "Answer": [{"type": 28, "data": "1.1.1.1"}]}
        ),
    ],
)
async def test_doh_failure_in_either_family_rejects_partial_success(bad_response):
    def respond(request):
        if request.url.params["type"] == "28":
            return bad_response
        return httpx.Response(
            200,
            json={
                "Status": 0,
                "Answer": [
                    {"type": 1, "data": PUBLIC_IPS[0]},
                ],
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(respond), follow_redirects=False
    )
    with patch("httpx.AsyncClient", return_value=client), pytest.raises(OSError):
        await resolve_public_hostname_addresses("example.com")
