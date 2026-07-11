"""Shared fetch helpers used by both the orchestrator and the strategies.

This module is a leaf below both ``markitai.fetch`` (orchestration) and
``markitai.fetch_strategies`` (per-strategy implementations). Helpers that
both layers need live here so strategy modules never import
``markitai.fetch`` — the cycle-free direction (``fetch`` ->
``fetch_strategies`` -> ``fetch_support``) is enforced by import-linter.

The public import path for these helpers remains ``markitai.fetch``, which
re-exports them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from markitai.fetch_session import get_default_session

if TYPE_CHECKING:
    from markitai.config import FetchConfig, PlaywrightConfig


def _detect_proxy(force_recheck: bool = False) -> str:
    """Detect proxy settings from environment, system config, or common local ports.

    Detection order:
    1. Environment variables: HTTPS_PROXY, HTTP_PROXY, ALL_PROXY
    2. System proxy settings (Windows registry / macOS scutil)
    3. Probe common proxy ports on localhost

    Args:
        force_recheck: Force re-detection even if cached

    Returns:
        Proxy URL string (e.g., "http://127.0.0.1:7890") or empty string if no proxy
    """
    return get_default_session().detect_proxy(force_recheck)


def _url_to_session_key(url: str) -> str:
    """Extract session key (domain) from URL."""
    from urllib.parse import urlparse

    return urlparse(url).netloc.lower()


def _resolve_playwright_profile_overrides(
    url: str, domain_profiles: dict[str, Any]
) -> dict[str, Any]:
    """Resolve domain-specific Playwright overrides from config."""
    from urllib.parse import urlparse

    from markitai.domain_profiles import BUILTIN_DOMAIN_PROFILES

    domain = urlparse(url).netloc.lower()
    profile = domain_profiles.get(domain)
    if not profile:
        profile = BUILTIN_DOMAIN_PROFILES.get(domain)
    if not profile:
        return {}

    out: dict[str, Any] = {}
    if profile.wait_for_selector:
        out["wait_for_selector"] = profile.wait_for_selector
    if profile.wait_for:
        out["wait_for"] = profile.wait_for
    if profile.extra_wait_ms is not None:
        out["extra_wait_ms"] = profile.extra_wait_ms
    if profile.skip_auto_scroll:
        out["skip_auto_scroll"] = profile.skip_auto_scroll
    if profile.reject_resource_patterns:
        out["reject_resource_patterns"] = profile.reject_resource_patterns
    return out


def _get_playwright_advanced_kwargs(pw: PlaywrightConfig) -> dict[str, Any]:
    """Extract advanced Playwright kwargs from config, omitting None values."""
    return {
        k: v
        for k, v in {
            "wait_for_selector": pw.wait_for_selector,
            "cookies": pw.cookies,
            "reject_resource_patterns": pw.reject_resource_patterns,
            "extra_http_headers": pw.extra_http_headers,
            "user_agent": pw.user_agent,
            "http_credentials": pw.http_credentials,
        }.items()
        if v is not None
    }


def _get_playwright_fetch_kwargs(
    url: str,
    config: FetchConfig,
    screenshot_config: Any | None = None,
    output_dir: Any | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Resolve all Playwright fetch arguments including domain profile overrides."""
    profile_overrides = _resolve_playwright_profile_overrides(
        url, config.domain_profiles
    )

    kwargs = {
        "timeout": config.playwright.timeout,
        "wait_for": profile_overrides.get("wait_for", config.playwright.wait_for),
        "extra_wait_ms": profile_overrides.get(
            "extra_wait_ms", config.playwright.extra_wait_ms
        ),
        "proxy": _detect_proxy() if getattr(config, "auto_proxy", True) else None,
        "screenshot_config": screenshot_config,
        "output_dir": output_dir,
        "renderer": renderer,
    }

    # Session persistence
    if config.playwright.session_mode == "domain_persistent":
        kwargs["session_key"] = _url_to_session_key(url)
        kwargs["persist_context"] = True

    advanced_kwargs = _get_playwright_advanced_kwargs(config.playwright)
    kwargs.update(advanced_kwargs)
    kwargs.update(profile_overrides)

    return kwargs
