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
