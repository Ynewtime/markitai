"""Tests for Playwright domain profile configuration and resolution."""

from __future__ import annotations

from markitai.config import DomainProfileConfig


def test_domain_profile_skip_auto_scroll_default_false() -> None:
    profile = DomainProfileConfig()
    assert profile.skip_auto_scroll is False


def test_domain_profile_skip_auto_scroll_can_be_set() -> None:
    profile = DomainProfileConfig(skip_auto_scroll=True)
    assert profile.skip_auto_scroll is True


def test_domain_profile_reject_resource_patterns_default_none() -> None:
    profile = DomainProfileConfig()
    assert profile.reject_resource_patterns is None


def test_domain_profile_reject_resource_patterns_can_be_set() -> None:
    profile = DomainProfileConfig(reject_resource_patterns=["**/ads/**"])
    assert profile.reject_resource_patterns == ["**/ads/**"]


def test_profile_overrides_propagate_skip_auto_scroll() -> None:
    """skip_auto_scroll from domain profile must be propagated to fetch kwargs."""
    from markitai.fetch import _resolve_playwright_profile_overrides

    profiles = {
        "x.com": DomainProfileConfig(
            skip_auto_scroll=True,
            wait_for_selector='[data-testid="tweet"]',
            extra_wait_ms=500,
        ),
    }
    overrides = _resolve_playwright_profile_overrides(
        "https://x.com/user/status/123", profiles
    )
    assert overrides.get("skip_auto_scroll") is True
    assert overrides.get("wait_for_selector") == '[data-testid="tweet"]'
    assert overrides.get("extra_wait_ms") == 500


def test_profile_overrides_propagate_reject_resource_patterns() -> None:
    """reject_resource_patterns from domain profile must be propagated."""
    from markitai.fetch import _resolve_playwright_profile_overrides

    profiles = {
        "x.com": DomainProfileConfig(
            reject_resource_patterns=["**/analytics/**", "**/*.mp4"],
        ),
    }
    overrides = _resolve_playwright_profile_overrides(
        "https://x.com/user/status/123", profiles
    )
    assert overrides.get("reject_resource_patterns") == ["**/analytics/**", "**/*.mp4"]


def test_fetch_method_accepts_skip_auto_scroll() -> None:
    """PlaywrightRenderer.fetch() must accept skip_auto_scroll parameter."""
    import inspect

    from markitai.fetch_playwright import PlaywrightRenderer

    sig = inspect.signature(PlaywrightRenderer.fetch)
    assert "skip_auto_scroll" in sig.parameters
    assert sig.parameters["skip_auto_scroll"].default is False


def test_profile_overrides_no_match_returns_empty() -> None:
    """Non-matching domain returns empty overrides."""
    from markitai.fetch import _resolve_playwright_profile_overrides

    profiles = {
        "x.com": DomainProfileConfig(skip_auto_scroll=True),
    }
    overrides = _resolve_playwright_profile_overrides(
        "https://github.com/repo", profiles
    )
    assert overrides == {}
