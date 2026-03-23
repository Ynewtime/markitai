"""Built-in domain profiles for common sites.

Separated into its own module to avoid circular imports between
``constants.py`` (imported by ``config.py``) and ``config.py``
(needed to construct ``DomainProfileConfig`` instances).
"""

from __future__ import annotations

from markitai.config import DomainProfileConfig

_X_COM_PROFILE = DomainProfileConfig(
    wait_for_selector='[data-testid="tweet"]',
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
