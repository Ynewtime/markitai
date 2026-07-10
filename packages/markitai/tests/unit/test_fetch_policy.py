"""Tests for fetch policy engine."""

import pytest

from markitai.constants import (
    ALL_FETCH_STRATEGIES,
    EXTERNAL_STRATEGIES,
    LOCAL_STRATEGIES,
)
from markitai.fetch_policy import (
    ALL_STRATEGIES,
    FetchPolicyEngine,
    assess_url_for_remote,
    is_private_or_local_domain,
    match_local_only,
    parse_no_proxy,
    url_contains_credentials,
)


def test_policy_prefers_playwright_for_known_spa_domain() -> None:
    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="x.com",
        known_spa=True,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    # SPA pages need JS rendering: local browser first, remote fallbacks after
    assert decision.order[0] == "playwright"
    assert decision.order[-1] == "static"


def test_policy_prefers_static_for_normal_domain() -> None:
    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="example.com",
        known_spa=False,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    # Local-first default: static leads, remote strategies are the tail
    assert decision.order[0] == "static"


def test_policy_prefers_local_strategies_for_localhost() -> None:
    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="localhost:8000",
        known_spa=False,
        explicit_strategy=None,
        fallback_patterns=[],
        policy_enabled=True,
    )
    assert decision.order == ["static", "playwright"]
    assert decision.reason == "private_or_local"


@pytest.mark.parametrize(
    "domain",
    [
        "127.1",
        "127.0.1",
        "0x7f.0.0.1",
        "2130706433",
        "100.64.0.1",
        "224.0.0.1",
        "localhost.",
        "localhost.:8000",
        "printer.local.",
    ],
)
def test_private_guard_rejects_numeric_aliases_and_terminal_dot(
    domain: str,
) -> None:
    assert is_private_or_local_domain(domain) is True


@pytest.mark.parametrize("domain", ["8.8.8.8", "example.com", "dead.beef"])
def test_private_guard_keeps_public_hosts_available(domain: str) -> None:
    assert is_private_or_local_domain(domain) is False


@pytest.mark.parametrize(
    "url",
    [
        "https://user:password@example.com/private",
        "https://example.com/file?token=secret",
        "https://example.com/file?X-Amz-Credential=id&X-Amz-Signature=sig",
        "https://example.com/reset?api_key=secret",
        "https://example.com/callback#access_token=secret",
        "https://example.com/account?session_id=secret",
        "https://example.com/account?sid=secret",
        "https://example.com/account?auth_key=secret",
        "https://example.com/account?access_key=secret",
        "https://example.com/account?ticket=secret",
        "https://example.com/account?pwd=secret",
        "https://example.com/sso?SAMLResponse=secret",
        "https://example.com/reset/550e8400-e29b-41d4-a716-446655440000",
        "https://example.com/api-key/550e8400-e29b-41d4-a716-446655440000",
        ("https://example.com/download/AbCDef0123456789_-AbCDef0123456789"),
    ],
)
def test_url_credentials_are_hard_local_only(url: str) -> None:
    assert url_contains_credentials(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/article?page=2#section",
        "https://example.com/releases/markitai-0.20.0-py3-none-any.whl",
        "https://example.com/articles/how-to-use-python-3-14-in-2026",
        "https://example.com/reset/complete.html",
    ],
)
def test_normal_public_url_is_not_treated_as_credentials(url: str) -> None:
    assert url_contains_credentials(url) is False


@pytest.mark.asyncio
async def test_remote_assessment_rejects_private_dns_answer() -> None:
    async def resolve_private(_hostname: str) -> tuple[str, ...]:
        return ("93.184.216.34", "10.0.0.25")

    assessment = await assess_url_for_remote(
        "https://portal.example.com/article",
        resolver=resolve_private,
    )

    assert assessment.allowed is False
    assert assessment.reason == "non_global_address"


@pytest.mark.asyncio
async def test_remote_assessment_allows_ordinary_public_url() -> None:
    async def resolve_public(_hostname: str) -> tuple[str, ...]:
        return ("93.184.216.34", "2606:4700::1111")

    assessment = await assess_url_for_remote(
        "https://example.com/articles/getting-started",
        resolver=resolve_public,
    )

    assert assessment.allowed is True
    assert assessment.reason is None


class TestFetchPolicyEngine:
    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_default_order(self) -> None:
        decision = self.engine.decide("example.com", False, None, [], True)
        assert decision.order == [
            "static",
            "playwright",
            "defuddle",
            "jina",
            "cloudflare",
        ]
        assert decision.reason == "default"

    def test_default_order_ignores_jina_key(self) -> None:
        """has_jina_key no longer changes ordering (local-first order is fixed)."""
        decision = self.engine.decide(
            "example.com", False, None, [], True, has_jina_key=True
        )
        assert decision.order == [
            "static",
            "playwright",
            "defuddle",
            "jina",
            "cloudflare",
        ]

    def test_spa_order(self) -> None:
        decision = self.engine.decide("twitter.com", True, None, ["twitter.com"], True)
        assert decision.order == [
            "playwright",
            "defuddle",
            "jina",
            "cloudflare",
            "static",
        ]
        assert decision.reason == "spa_or_pattern"

    def test_explicit_strategy(self) -> None:
        decision = self.engine.decide(
            "example.com", False, "static", [], True, has_jina_key=True
        )
        assert decision.order == ["static"]

    def test_disabled_policy(self) -> None:
        decision = self.engine.decide("example.com", False, None, [], False)
        assert decision.order == [
            "static",
            "playwright",
            "defuddle",
            "jina",
            "cloudflare",
        ]
        assert decision.reason == "disabled"

    def test_disabled_policy_still_keeps_private_hosts_local_only(self) -> None:
        decision = self.engine.decide("127.0.0.1:3000", False, None, [], False)
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "private_or_local"

    def test_all_strategies_present_in_default(self) -> None:
        decision = self.engine.decide("example.com", False, None, [], True)
        assert set(decision.order) == set(ALL_STRATEGIES)


class TestDomainPreferStrategy:
    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_prefer_jina(self) -> None:
        decision = self.engine.decide(
            "docs.python.org", False, None, [], True, domain_prefer_strategy="jina"
        )
        assert decision.order[0] == "jina"
        assert decision.reason == "domain_prefer_jina"
        assert len(decision.order) == 5

    def test_prefer_playwright(self) -> None:
        decision = self.engine.decide(
            "example.com",
            False,
            None,
            [],
            True,
            domain_prefer_strategy="playwright",
        )
        assert decision.order[0] == "playwright"
        assert decision.reason == "domain_prefer_playwright"

    def test_prefer_overrides_spa(self) -> None:
        """Domain preference should override SPA detection."""
        decision = self.engine.decide(
            "twitter.com",
            True,
            None,
            ["twitter.com"],
            True,
            domain_prefer_strategy="static",
        )
        assert decision.order[0] == "static"
        assert decision.reason == "domain_prefer_static"

    def test_explicit_overrides_prefer(self) -> None:
        """Explicit strategy should override domain preference."""
        decision = self.engine.decide(
            "example.com",
            False,
            "playwright",
            [],
            True,
            domain_prefer_strategy="jina",
        )
        assert decision.order == ["playwright"]

    def test_no_prefer_strategy(self) -> None:
        """When prefer_strategy is None, behavior is unchanged."""
        decision = self.engine.decide(
            "example.com", False, None, [], True, domain_prefer_strategy=None
        )
        assert decision.order == [
            "static",
            "playwright",
            "defuddle",
            "jina",
            "cloudflare",
        ]
        assert decision.reason == "default"


def test_strategy_constants_are_consistent() -> None:
    """LOCAL + EXTERNAL = ALL, no overlap."""
    assert set(LOCAL_STRATEGIES) | set(EXTERNAL_STRATEGIES) == set(ALL_FETCH_STRATEGIES)
    assert set(LOCAL_STRATEGIES) & set(EXTERNAL_STRATEGIES) == set()


class TestMatchLocalOnly:
    def test_exact_domain_match(self) -> None:
        assert match_local_only("internal.corp.com", ["internal.corp.com"]) is True

    def test_exact_domain_match_ignores_dns_terminal_dot(self) -> None:
        assert match_local_only("internal.corp.com.", ["internal.corp.com"]) is True

    def test_domain_no_match(self) -> None:
        assert match_local_only("example.com", ["internal.corp.com"]) is False

    def test_suffix_with_leading_dot(self) -> None:
        assert match_local_only("app.internal.com", [".internal.com"]) is True

    def test_suffix_does_not_match_exact(self) -> None:
        assert match_local_only("internal.com", [".internal.com"]) is False

    def test_wildcard_prefix(self) -> None:
        assert match_local_only("app.internal.com", ["*.internal.com"]) is True

    def test_ip_exact_match(self) -> None:
        assert match_local_only("10.0.1.5", ["10.0.1.5"]) is True

    def test_cidr_match(self) -> None:
        assert match_local_only("10.0.1.5", ["10.0.0.0/8"]) is True

    def test_cidr_no_match(self) -> None:
        assert match_local_only("192.168.1.1", ["10.0.0.0/8"]) is False

    def test_ipv6_cidr(self) -> None:
        assert match_local_only("fd12::1", ["fd00::/8"]) is True

    def test_localhost_special(self) -> None:
        assert match_local_only("localhost", ["localhost"]) is True

    def test_localhost_with_port(self) -> None:
        assert match_local_only("localhost:3000", ["localhost"]) is True

    def test_empty_patterns(self) -> None:
        assert match_local_only("example.com", []) is False

    def test_multiple_patterns_any_match(self) -> None:
        patterns = [".corp.com", "10.0.0.0/8"]
        assert match_local_only("app.corp.com", patterns) is True
        assert match_local_only("10.1.2.3", patterns) is True
        assert match_local_only("example.com", patterns) is False


class TestParseNoProxy:
    def test_comma_separated(self) -> None:
        result = parse_no_proxy("localhost,.internal.com,10.0.0.0/8")
        assert result == ["localhost", ".internal.com", "10.0.0.0/8"]

    def test_strips_whitespace(self) -> None:
        result = parse_no_proxy(" localhost , .corp.com ")
        assert result == ["localhost", ".corp.com"]

    def test_empty_string(self) -> None:
        assert parse_no_proxy("") == []

    def test_none(self) -> None:
        assert parse_no_proxy(None) == []

    def test_filters_empty_entries(self) -> None:
        result = parse_no_proxy("localhost,,,.corp.com")
        assert result == ["localhost", ".corp.com"]

    def test_wildcard_star_only(self) -> None:
        result = parse_no_proxy("*")
        assert result == ["*"]


class TestIsPrivateOrLocalDomain:
    """Domains that must never be sent to remote extraction services."""

    def test_public_domains_are_not_private(self) -> None:
        assert is_private_or_local_domain("x.com") is False
        assert is_private_or_local_domain("www.bilibili.com") is False

    def test_localhost_and_suffixes(self) -> None:
        assert is_private_or_local_domain("localhost") is True
        assert is_private_or_local_domain("app.localhost:3000") is True
        assert is_private_or_local_domain("nas.local") is True
        assert is_private_or_local_domain("wiki.corp") is True

    def test_private_ips(self) -> None:
        assert is_private_or_local_domain("10.0.1.5") is True
        assert is_private_or_local_domain("192.168.1.1:8080") is True
        assert is_private_or_local_domain("127.0.0.1") is True
        assert is_private_or_local_domain("[::1]") is True

    def test_userinfo_credentials_treated_as_private(self) -> None:
        """URLs carrying credentials in the netloc (user:pass@host, user@host)
        must be treated as private: the credential string itself is the
        secret, so it must never reach a remote extraction service."""
        assert is_private_or_local_domain("user:secret@example.com") is True
        assert is_private_or_local_domain("admin@example.com") is True


class TestStrategyPriorityOverride:
    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_global_priority_overrides_default(self) -> None:
        decision = self.engine.decide(
            "example.com",
            False,
            None,
            [],
            True,
            global_strategy_priority=["static", "playwright"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "global_priority"

    def test_domain_priority_overrides_global(self) -> None:
        decision = self.engine.decide(
            "example.com",
            False,
            None,
            [],
            True,
            global_strategy_priority=["static", "playwright"],
            domain_strategy_priority=["defuddle", "static"],
        )
        assert decision.order == ["defuddle", "static"]
        assert decision.reason == "domain_priority"

    def test_domain_priority_overrides_prefer(self) -> None:
        decision = self.engine.decide(
            "example.com",
            False,
            None,
            [],
            True,
            domain_prefer_strategy="jina",
            domain_strategy_priority=["static"],
        )
        assert decision.order == ["static"]
        assert decision.reason == "domain_priority"

    def test_explicit_still_wins_over_all(self) -> None:
        decision = self.engine.decide(
            "example.com",
            False,
            "playwright",
            [],
            True,
            global_strategy_priority=["static"],
            domain_strategy_priority=["defuddle"],
        )
        assert decision.order == ["playwright"]

    def test_global_priority_bypasses_spa(self) -> None:
        decision = self.engine.decide(
            "twitter.com",
            True,
            None,
            ["twitter.com"],
            True,
            global_strategy_priority=["static", "playwright"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "global_priority"


class TestLocalOnlyExemption:
    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_matching_domain_forces_local_strategies(self) -> None:
        decision = self.engine.decide(
            "app.internal.com",
            False,
            None,
            [],
            True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "local_only_pattern"

    def test_cidr_match_forces_local(self) -> None:
        decision = self.engine.decide(
            "10.0.1.5",
            False,
            None,
            [],
            True,
            local_only_patterns=["10.0.0.0/8"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "local_only_pattern"

    def test_non_matching_domain_uses_normal_order(self) -> None:
        decision = self.engine.decide(
            "example.com",
            False,
            None,
            [],
            True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order[0] == "static"

    def test_local_only_overrides_global_priority(self) -> None:
        decision = self.engine.decide(
            "app.internal.com",
            False,
            None,
            [],
            True,
            global_strategy_priority=["defuddle", "jina", "static"],
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["static", "playwright"]
        assert decision.reason == "local_only_pattern"

    def test_local_only_overrides_domain_priority(self) -> None:
        decision = self.engine.decide(
            "app.internal.com",
            False,
            None,
            [],
            True,
            domain_strategy_priority=["defuddle"],
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["static", "playwright"]

    def test_explicit_strategy_overrides_local_only(self) -> None:
        decision = self.engine.decide(
            "app.internal.com",
            False,
            "defuddle",
            [],
            True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["defuddle"]

    def test_wildcard_star_matches_all(self) -> None:
        decision = self.engine.decide(
            "example.com",
            False,
            None,
            [],
            True,
            local_only_patterns=["*"],
        )
        assert decision.order == ["static", "playwright"]

    def test_spa_domain_with_local_only_stays_local(self) -> None:
        decision = self.engine.decide(
            "spa.internal.com",
            True,
            None,
            [],
            True,
            local_only_patterns=[".internal.com"],
        )
        assert decision.order == ["playwright", "static"]
        assert "local_only" in decision.reason
