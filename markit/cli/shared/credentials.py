"""Shared credential utilities for CLI commands."""

import os
from typing import Any

from markit.config.constants import PROVIDER_API_KEY_ENV_VARS


def get_effective_api_key(config: Any) -> str | None:
    """Get API key from config, checking env vars if needed.

    Args:
        config: A config object with optional api_key, api_key_env, and provider attributes

    Returns:
        The resolved API key or None if not found
    """
    api_key = getattr(config, "api_key", None)
    if not api_key:
        api_key_env = getattr(config, "api_key_env", None)
        if api_key_env:
            api_key = os.environ.get(api_key_env)

    if not api_key:
        provider_name = config.provider
        env_var = PROVIDER_API_KEY_ENV_VARS.get(provider_name)
        if env_var:
            api_key = os.environ.get(env_var)
    return api_key


def get_unique_credentials(
    settings,
) -> list[tuple[str, str | None, str | None, str, str | None]]:
    """Get unique credentials from settings.

    Deduplicates credentials from both new-style credentials and legacy providers.

    Args:
        settings: MarkIt settings object

    Returns:
        List of (provider_name, api_key, base_url, display_name, credential_id) tuples
    """
    creds = []
    seen = set()

    # 1. Process new credentials
    for cred in settings.llm.credentials:
        api_key = get_effective_api_key(cred)
        key = (cred.provider, api_key, cred.base_url)
        if key not in seen:
            seen.add(key)
            creds.append((cred.provider, api_key, cred.base_url, cred.id, cred.id))

    # 2. Process legacy providers (de-duplicate)
    for config in settings.llm.providers:
        api_key = get_effective_api_key(config)
        key = (config.provider, api_key, config.base_url)
        if key not in seen:
            seen.add(key)
            display_name = config.name or config.provider
            # Legacy providers don't have a credential ID
            creds.append((config.provider, api_key, config.base_url, display_name, None))

    return creds
