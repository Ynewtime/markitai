"""Fetch policy engine for determining URL fetch strategies.

Strategy priority rationale (v0.9.0):
  defuddle → jina → static → playwright → cloudflare

- defuddle: Free, no auth, best content cleaning (article extraction + frontmatter).
- jina: Free tier (20 RPM), good JS rendering, structured JSON output.
- static: Fastest but no content cleaning or JS rendering.
- playwright: Full JS rendering but heavy dependency, slow.
- cloudflare: Requires CF account, rate-limited.

NOTE: defuddle's rate limit is undocumented. If it turns out to be restrictive,
consider swapping defuddle and static in the default order.
"""

from __future__ import annotations

import ipaddress
from dataclasses import dataclass

ALL_STRATEGIES = ["defuddle", "jina", "static", "playwright", "cloudflare"]
LOCAL_ONLY_STRATEGIES = ["static", "playwright"]


def _extract_host(domain: str) -> str:
    """Extract host from a netloc-like domain string."""
    if domain.startswith("[") and "]" in domain:
        return domain[1 : domain.index("]")]
    return domain.split(":", 1)[0]


def is_private_or_local_domain(domain: str) -> bool:
    """Return True for localhost, private IPs, and common intranet-only hosts."""
    host = _extract_host(domain).strip().lower()
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    if host.endswith((".local", ".internal", ".lan", ".home", ".corp")):
        return True
    if "." not in host:
        return True

    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_unspecified
    )


@dataclass
class FetchDecision:
    """Decision from the fetch policy engine."""

    order: list[str]
    reason: str


class FetchPolicyEngine:
    """Engine to decide the order of fetch strategies based on URL and config."""

    def decide(
        self,
        domain: str,
        known_spa: bool,
        explicit_strategy: str | None,
        fallback_patterns: list[str],
        policy_enabled: bool,
        has_jina_key: bool = False,
        domain_prefer_strategy: str | None = None,
    ) -> FetchDecision:
        """Decide the fetch strategy order."""
        if explicit_strategy and explicit_strategy != "auto":
            return FetchDecision(
                order=[explicit_strategy], reason=f"explicit_{explicit_strategy}"
            )

        is_fallback_domain = any(
            domain == p or domain.endswith(f".{p}") for p in fallback_patterns
        )

        if is_private_or_local_domain(domain):
            order = (
                ["playwright", "static"]
                if known_spa or is_fallback_domain
                else LOCAL_ONLY_STRATEGIES.copy()
            )
            return FetchDecision(order=order, reason="private_or_local")

        # Domain-specific preference (highest priority after explicit)
        if domain_prefer_strategy:
            remaining = [s for s in ALL_STRATEGIES if s != domain_prefer_strategy]
            return FetchDecision(
                order=[domain_prefer_strategy] + remaining,
                reason=f"domain_prefer_{domain_prefer_strategy}",
            )

        if not policy_enabled:
            return FetchDecision(
                order=["defuddle", "jina", "static", "playwright", "cloudflare"],
                reason="disabled",
            )

        # For SPA/JS-heavy sites, playwright goes earlier but defuddle/jina still lead
        # because they may have server-side rendering on their end
        if known_spa or is_fallback_domain:
            return FetchDecision(
                order=["defuddle", "jina", "playwright", "cloudflare", "static"],
                reason="spa_or_pattern",
            )

        # Default: content-cleaning APIs first, then local strategies
        return FetchDecision(
            order=["defuddle", "jina", "static", "playwright", "cloudflare"],
            reason="default",
        )
