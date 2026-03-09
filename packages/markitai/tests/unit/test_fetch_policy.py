"""Tests for fetch policy engine."""

from markitai.fetch_policy import ALL_STRATEGIES, FetchPolicyEngine


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
