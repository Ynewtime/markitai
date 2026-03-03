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
