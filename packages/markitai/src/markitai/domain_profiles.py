"""Built-in domain profiles for common sites.

Separated into its own module to avoid circular imports between
``constants.py`` (imported by ``config.py``) and ``config.py``
(needed to construct ``DomainProfileConfig`` instances).
"""

from __future__ import annotations

from markitai.config import DomainProfileConfig


def resolve_domain_profile(
    domain: str, profiles: dict[str, DomainProfileConfig]
) -> DomainProfileConfig | None:
    """Merge explicit user fields over built-in tuning without mutating either."""
    builtin = BUILTIN_DOMAIN_PROFILES.get(domain)
    user = profiles.get(domain)
    if builtin is None:
        return user
    if user is None:
        return builtin
    return builtin.model_copy(update=user.model_dump(exclude_unset=True))


_X_COM_PROFILE = DomainProfileConfig(
    wait_for_selector='article[data-tweet-id], [data-testid="tweet"]',
    wait_for="domcontentloaded",
    extra_wait_ms=500,
    skip_auto_scroll=True,
    reject_resource_patterns=[
        "**/analytics/**",
        "**/ads/**",
        "**/tracking/**",
        "**/*.mp4",
    ],
)

BUILTIN_DOMAIN_PROFILES: dict[str, DomainProfileConfig] = {
    "x.com": _X_COM_PROFILE,
    "twitter.com": _X_COM_PROFILE,
    "github.com": DomainProfileConfig(
        wait_for_selector=".markdown-body",
        wait_for="domcontentloaded",
        extra_wait_ms=300,
        skip_auto_scroll=True,
    ),
}
