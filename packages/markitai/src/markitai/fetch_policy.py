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

from markitai.constants import ALL_FETCH_STRATEGIES, LOCAL_STRATEGIES

ALL_STRATEGIES = list(ALL_FETCH_STRATEGIES)
LOCAL_ONLY_STRATEGIES = list(LOCAL_STRATEGIES)


def _extract_host(domain: str) -> str:
    """Extract host from a netloc-like domain string."""
    if domain.startswith("[") and "]" in domain:
        return domain[1 : domain.index("]")]
    # Bare IPv6 (multiple colons, e.g. "fd12::1") — return as-is
    if domain.count(":") > 1:
        return domain
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


def parse_no_proxy(value: str | None) -> list[str]:
    """Parse a NO_PROXY-style comma-separated string into a list of patterns."""
    if not value:
        return []
    return [p.strip() for p in value.split(",") if p.strip()]


def match_local_only(domain: str, patterns: list[str]) -> bool:
    """Check if domain matches any local-only exemption pattern.

    Supports NO_PROXY syntax:
    - Domain exact: ``internal.corp.com``
    - Suffix: ``.internal.com`` (matches subdomains only)
    - Wildcard: ``*.internal.com`` (same as ``.internal.com``)
    - IP exact: ``10.0.1.5``
    - CIDR: ``10.0.0.0/8``, ``fd00::/8``
    - Star: ``*`` (matches everything)
    - Special: ``localhost``
    """
    if not patterns:
        return False

    host = _extract_host(domain).strip().lower()
    if not host:
        return False

    for pattern in patterns:
        p = pattern.strip().lower()
        if not p:
            continue

        # Wildcard: match everything
        if p == "*":
            return True

        # Normalize *.foo.com → .foo.com
        if p.startswith("*."):
            p = p[1:]  # "*.foo.com" → ".foo.com"

        # Suffix match: .foo.com matches sub.foo.com but not foo.com
        if p.startswith("."):
            if host.endswith(p):
                return True
            continue

        # CIDR match
        if "/" in p:
            try:
                network = ipaddress.ip_network(p, strict=False)
                host_ip = ipaddress.ip_address(host)
                if host_ip in network:
                    return True
            except ValueError:
                pass
            continue

        # Exact match (domain or IP)
        if host == p:
            return True

    return False


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
        global_strategy_priority: list[str] | None = None,
        domain_strategy_priority: list[str] | None = None,
        local_only_patterns: list[str] | None = None,
    ) -> FetchDecision:
        """Decide the fetch strategy order.

        Priority chain (highest to lowest):
        1. Explicit strategy (CLI flag) -- single strategy, no fallback
        2. Local-only exemption -- security: restrict to local strategies
        3. Private/local domain -- hardcoded local detection
        4. Domain strategy_priority -- per-domain full override
        5. Domain prefer_strategy -- per-domain single promotion
        6. Global strategy_priority -- global full override
        7. SPA/JS-required pattern -- browser-first order
        8. Default order
        """
        # 1. Explicit strategy always wins
        if explicit_strategy and explicit_strategy != "auto":
            return FetchDecision(
                order=[explicit_strategy], reason=f"explicit_{explicit_strategy}"
            )

        is_fallback_domain = any(
            domain == p or domain.endswith(f".{p}") for p in fallback_patterns
        )

        # 2. Local-only exemption (security: skip external APIs)
        if local_only_patterns and match_local_only(domain, local_only_patterns):
            order = (
                ["playwright", "static"]
                if known_spa or is_fallback_domain
                else LOCAL_ONLY_STRATEGIES.copy()
            )
            return FetchDecision(order=order, reason="local_only_pattern")

        # 3. Private/local domain (hardcoded detection)
        if is_private_or_local_domain(domain):
            order = (
                ["playwright", "static"]
                if known_spa or is_fallback_domain
                else LOCAL_ONLY_STRATEGIES.copy()
            )
            return FetchDecision(order=order, reason="private_or_local")

        # 4. Per-domain strategy_priority (full override)
        if domain_strategy_priority:
            return FetchDecision(
                order=list(domain_strategy_priority), reason="domain_priority"
            )

        # 5. Per-domain prefer_strategy (single promotion)
        if domain_prefer_strategy:
            remaining = [s for s in ALL_STRATEGIES if s != domain_prefer_strategy]
            return FetchDecision(
                order=[domain_prefer_strategy] + remaining,
                reason=f"domain_prefer_{domain_prefer_strategy}",
            )

        # 6. Global strategy_priority (full override)
        if global_strategy_priority:
            return FetchDecision(
                order=list(global_strategy_priority), reason="global_priority"
            )

        if not policy_enabled:
            return FetchDecision(
                order=ALL_STRATEGIES.copy(),
                reason="disabled",
            )

        # 7. SPA/JS-heavy: defuddle & jina still lead
        # Known SPAs and fallback-pattern domains often need JS rendering,
        # but defuddle and jina may have server-side extraction that works
        # without a browser. Try them first (fast), fall back to playwright.
        if known_spa or is_fallback_domain:
            return FetchDecision(
                order=["defuddle", "jina", "playwright", "cloudflare", "static"],
                reason="spa_or_pattern",
            )

        # 8. Default
        return FetchDecision(
            order=ALL_STRATEGIES.copy(),
            reason="default",
        )
