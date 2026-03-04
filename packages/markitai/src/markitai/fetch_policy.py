"""Fetch policy engine for determining URL fetch strategies."""

from dataclasses import dataclass


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

        # Domain-specific preference (highest priority after explicit)
        if domain_prefer_strategy:
            all_strategies = ["static", "playwright", "cloudflare", "jina"]
            remaining = [s for s in all_strategies if s != domain_prefer_strategy]
            return FetchDecision(
                order=[domain_prefer_strategy] + remaining,
                reason=f"domain_prefer_{domain_prefer_strategy}",
            )

        if not policy_enabled:
            if has_jina_key:
                return FetchDecision(
                    order=["static", "jina", "playwright", "cloudflare"],
                    reason="disabled_jina_key",
                )
            return FetchDecision(
                order=["static", "playwright", "cloudflare", "jina"], reason="disabled"
            )

        # Fallback patterns might be exact domains or glob-like. For simplicity in the initial policy
        # we check if the domain is directly in fallback_patterns or ends with a pattern.
        # However, fallback_patterns logic is handled upstream. For now, we trust `known_spa` or `domain in fallback_patterns`.
        is_fallback_domain = any(
            domain == p or domain.endswith(f".{p}") for p in fallback_patterns
        )

        if known_spa or is_fallback_domain:
            if has_jina_key:
                return FetchDecision(
                    order=["playwright", "jina", "cloudflare", "static"],
                    reason="spa_jina_key",
                )
            return FetchDecision(
                order=["playwright", "cloudflare", "jina", "static"],
                reason="spa_or_pattern",
            )

        if has_jina_key:
            return FetchDecision(
                order=["static", "jina", "playwright", "cloudflare"],
                reason="default_jina_key",
            )

        return FetchDecision(
            order=["static", "playwright", "cloudflare", "jina"], reason="default"
        )
