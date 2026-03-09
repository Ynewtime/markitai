"""Tests for fetch policy engine."""

from markitai.constants import (
    ALL_FETCH_STRATEGIES,
    EXTERNAL_STRATEGIES,
    LOCAL_STRATEGIES,
)
from markitai.fetch_policy import (
    ALL_STRATEGIES,
    FetchPolicyEngine,
    match_local_only,
    parse_no_proxy,
)


def test_policy_prefers_defuddle_for_known_spa_domain() -> None:
    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="x.com",
        known_spa=True,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    # defuddle and jina lead even for SPA (they may have server-side rendering)
    assert decision.order[:2] == ["defuddle", "jina"]
    assert "playwright" in decision.order


def test_policy_prefers_defuddle_for_normal_domain() -> None:
    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="example.com",
        known_spa=False,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    assert decision.order[0] == "defuddle"


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


class TestFetchPolicyEngine:
    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_default_order(self) -> None:
        decision = self.engine.decide("example.com", False, None, [], True)
        assert decision.order == [
            "defuddle",
            "jina",
            "static",
            "playwright",
            "cloudflare",
        ]
        assert decision.reason == "default"

    def test_default_order_ignores_jina_key(self) -> None:
        """has_jina_key no longer changes ordering (defuddle+jina always first)."""
        decision = self.engine.decide(
            "example.com", False, None, [], True, has_jina_key=True
        )
        assert decision.order == [
            "defuddle",
            "jina",
            "static",
            "playwright",
            "cloudflare",
        ]

    def test_spa_order(self) -> None:
        decision = self.engine.decide("twitter.com", True, None, ["twitter.com"], True)
        assert decision.order == [
            "defuddle",
            "jina",
            "playwright",
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
            "defuddle",
            "jina",
            "static",
            "playwright",
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
            "defuddle",
            "jina",
            "static",
            "playwright",
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
        assert decision.order[0] == "defuddle"

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
