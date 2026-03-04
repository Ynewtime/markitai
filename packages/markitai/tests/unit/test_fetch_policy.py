"""Tests for fetch policy engine."""

from markitai.fetch_policy import FetchPolicyEngine


def test_policy_prefers_browser_for_known_spa_domain() -> None:
    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="x.com",
        known_spa=True,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    assert decision.order[:2] == ["playwright", "cloudflare"]


def test_policy_keeps_static_first_for_normal_domain() -> None:
    engine = FetchPolicyEngine()
    decision = engine.decide(
        domain="example.com",
        known_spa=False,
        explicit_strategy=None,
        fallback_patterns=["x.com"],
        policy_enabled=True,
    )
    assert decision.order[0] == "static"


class TestFetchPolicyEngine:
    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_default_order_no_jina_key(self) -> None:
        decision = self.engine.decide("example.com", False, None, [], True)
        assert decision.order == ["static", "playwright", "cloudflare", "jina"]
        assert decision.reason == "default"

    def test_default_order_with_jina_key(self) -> None:
        decision = self.engine.decide(
            "example.com", False, None, [], True, has_jina_key=True
        )
        assert decision.order == ["static", "jina", "playwright", "cloudflare"]
        assert decision.reason == "default_jina_key"

    def test_spa_order_no_jina_key(self) -> None:
        decision = self.engine.decide("twitter.com", True, None, ["twitter.com"], True)
        assert decision.order == ["playwright", "cloudflare", "jina", "static"]

    def test_spa_order_with_jina_key(self) -> None:
        decision = self.engine.decide(
            "twitter.com", True, None, ["twitter.com"], True, has_jina_key=True
        )
        assert decision.order == ["playwright", "jina", "cloudflare", "static"]
        assert decision.reason == "spa_jina_key"

    def test_explicit_strategy_ignores_jina_key(self) -> None:
        decision = self.engine.decide(
            "example.com", False, "static", [], True, has_jina_key=True
        )
        assert decision.order == ["static"]

    def test_disabled_policy_no_jina_key(self) -> None:
        decision = self.engine.decide("example.com", False, None, [], False)
        assert decision.order == ["static", "playwright", "cloudflare", "jina"]
        assert decision.reason == "disabled"

    def test_disabled_policy_with_jina_key(self) -> None:
        decision = self.engine.decide(
            "example.com", False, None, [], False, has_jina_key=True
        )
        assert decision.order == ["static", "jina", "playwright", "cloudflare"]
        assert decision.reason == "disabled_jina_key"


class TestDomainPreferStrategy:
    def setup_method(self) -> None:
        self.engine = FetchPolicyEngine()

    def test_prefer_jina(self) -> None:
        decision = self.engine.decide(
            "docs.python.org", False, None, [], True, domain_prefer_strategy="jina"
        )
        assert decision.order[0] == "jina"
        assert decision.reason == "domain_prefer_jina"
        assert len(decision.order) == 4

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
        assert decision.order == ["static", "playwright", "cloudflare", "jina"]
        assert decision.reason == "default"
