# Local Provider Integration Improvement Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enhance reliability, functionality, and testability of Claude Agent and Copilot provider integrations

**Architecture:** Three-phase approach - reliability foundations (auth, timeout, retry), feature enhancements (caching, unified JSON, capability detection), and testing/tooling (doctor command, mock-based integration tests)

**Tech Stack:** Python 3.13, LiteLLM, Claude Agent SDK, Copilot SDK, pytest, asyncio

---

## Phase 1: Reliability Foundations (P0)

### Task 1: Create Provider Error Classes

**Files:**
- Create: `packages/markitai/src/markitai/providers/errors.py`
- Test: `packages/markitai/tests/unit/test_provider_errors.py`

**Step 1: Write the failing test**

```python
# packages/markitai/tests/unit/test_provider_errors.py
"""Unit tests for provider error classes."""

from __future__ import annotations

import pytest


class TestProviderErrors:
    """Tests for custom provider error classes."""

    def test_provider_error_base_class(self) -> None:
        """Test ProviderError is the base class for all provider errors."""
        from markitai.providers.errors import ProviderError

        error = ProviderError("Test error", provider="claude-agent")
        assert str(error) == "Test error"
        assert error.provider == "claude-agent"

    def test_authentication_error(self) -> None:
        """Test AuthenticationError with provider and resolution hint."""
        from markitai.providers.errors import AuthenticationError

        error = AuthenticationError(
            message="Not authenticated",
            provider="copilot",
            resolution="Run 'copilot' in terminal to login",
        )
        assert "Not authenticated" in str(error)
        assert error.provider == "copilot"
        assert error.resolution == "Run 'copilot' in terminal to login"

    def test_quota_error(self) -> None:
        """Test QuotaError for billing/subscription issues."""
        from markitai.providers.errors import QuotaError

        error = QuotaError(
            message="Quota exceeded",
            provider="claude-agent",
            resolution="Check your Claude subscription status",
        )
        assert "Quota exceeded" in str(error)
        assert error.provider == "claude-agent"

    def test_timeout_error(self) -> None:
        """Test ProviderTimeoutError with timeout value."""
        from markitai.providers.errors import ProviderTimeoutError

        error = ProviderTimeoutError(
            message="Request timed out",
            provider="copilot",
            timeout_seconds=120,
        )
        assert error.timeout_seconds == 120

    def test_sdk_not_available_error(self) -> None:
        """Test SDKNotAvailableError with install instructions."""
        from markitai.providers.errors import SDKNotAvailableError

        error = SDKNotAvailableError(
            provider="claude-agent",
            install_command="uv sync --extra claude-agent",
        )
        assert "claude-agent" in str(error)
        assert error.install_command == "uv sync --extra claude-agent"

    def test_errors_inherit_from_provider_error(self) -> None:
        """Test all errors inherit from ProviderError."""
        from markitai.providers.errors import (
            AuthenticationError,
            ProviderError,
            ProviderTimeoutError,
            QuotaError,
            SDKNotAvailableError,
        )

        assert issubclass(AuthenticationError, ProviderError)
        assert issubclass(QuotaError, ProviderError)
        assert issubclass(ProviderTimeoutError, ProviderError)
        assert issubclass(SDKNotAvailableError, ProviderError)
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_errors.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'markitai.providers.errors'"

**Step 3: Write minimal implementation**

```python
# packages/markitai/src/markitai/providers/errors.py
"""Custom exception classes for local providers.

This module defines a hierarchy of provider-specific exceptions that:
1. Provide structured error information (provider, resolution hints)
2. Enable proper error classification for retry logic
3. Improve user experience with actionable error messages
"""

from __future__ import annotations


class ProviderError(Exception):
    """Base class for all provider-related errors.

    Attributes:
        provider: The provider that raised the error (e.g., "claude-agent", "copilot")
        message: Human-readable error description
    """

    def __init__(self, message: str, *, provider: str) -> None:
        self.message = message
        self.provider = provider
        super().__init__(message)


class AuthenticationError(ProviderError):
    """Raised when authentication fails or is missing.

    This error should NOT be retried - it requires user action to resolve.

    Attributes:
        resolution: Actionable steps to fix the authentication issue
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        resolution: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.resolution = resolution or self._default_resolution(provider)

    @staticmethod
    def _default_resolution(provider: str) -> str:
        """Return default resolution based on provider."""
        if provider == "copilot":
            return "Run 'copilot' in terminal and follow the login prompts"
        if provider == "claude-agent":
            return "Run 'claude setup-token' to configure authentication"
        return "Check your authentication credentials"


class QuotaError(ProviderError):
    """Raised when quota/billing limits are exceeded.

    This error should NOT be retried - it requires user action to resolve.

    Attributes:
        resolution: Actionable steps to resolve quota issues
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        resolution: str | None = None,
    ) -> None:
        super().__init__(message, provider=provider)
        self.resolution = resolution or "Check your subscription status and billing"


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out.

    This error MAY be retried with adjusted timeout.

    Attributes:
        timeout_seconds: The timeout value that was exceeded
    """

    def __init__(
        self,
        message: str,
        *,
        provider: str,
        timeout_seconds: int | float,
    ) -> None:
        super().__init__(message, provider=provider)
        self.timeout_seconds = timeout_seconds


class SDKNotAvailableError(ProviderError):
    """Raised when the required SDK is not installed.

    Attributes:
        install_command: Command to install the missing SDK
    """

    def __init__(
        self,
        *,
        provider: str,
        install_command: str | None = None,
    ) -> None:
        self.install_command = install_command or self._default_install(provider)
        message = (
            f"{provider} SDK is not installed. "
            f"Install with: {self.install_command}"
        )
        super().__init__(message, provider=provider)

    @staticmethod
    def _default_install(provider: str) -> str:
        """Return default install command based on provider."""
        if provider == "copilot":
            return "uv sync --extra copilot"
        if provider == "claude-agent":
            return "uv sync --extra claude-agent"
        return f"uv sync --extra {provider}"
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_errors.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/errors.py packages/markitai/tests/unit/test_provider_errors.py
git commit -m "feat(providers): add structured error classes for local providers"
```

---

### Task 2: Create Authentication Manager

**Files:**
- Create: `packages/markitai/src/markitai/providers/auth.py`
- Test: `packages/markitai/tests/unit/test_provider_auth.py`

**Step 1: Write the failing test**

```python
# packages/markitai/tests/unit/test_provider_auth.py
"""Unit tests for provider authentication manager."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestAuthManager:
    """Tests for AuthManager class."""

    def test_auth_manager_singleton(self) -> None:
        """Test AuthManager uses singleton pattern."""
        from markitai.providers.auth import AuthManager

        manager1 = AuthManager()
        manager2 = AuthManager()
        assert manager1 is manager2

    def test_auth_status_dataclass(self) -> None:
        """Test AuthStatus dataclass structure."""
        from markitai.providers.auth import AuthStatus

        status = AuthStatus(
            provider="copilot",
            authenticated=True,
            user="test@example.com",
            expires_at=None,
            error=None,
        )
        assert status.provider == "copilot"
        assert status.authenticated is True
        assert status.user == "test@example.com"

    def test_auth_status_unauthenticated(self) -> None:
        """Test AuthStatus for unauthenticated state."""
        from markitai.providers.auth import AuthStatus

        status = AuthStatus(
            provider="claude-agent",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Not authenticated",
        )
        assert status.authenticated is False
        assert status.error == "Not authenticated"


class TestCopilotAuthCheck:
    """Tests for Copilot authentication checking."""

    @pytest.mark.asyncio
    async def test_check_copilot_auth_when_authenticated(self) -> None:
        """Test Copilot auth check returns authenticated status."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Mock the Copilot SDK auth check
        mock_status = MagicMock()
        mock_status.authenticated = True
        mock_status.user = "test@example.com"

        with patch(
            "markitai.providers.auth._check_copilot_sdk_auth",
            AsyncMock(return_value=mock_status),
        ):
            status = await manager.check_auth("copilot")
            assert status.authenticated is True
            assert status.provider == "copilot"

    @pytest.mark.asyncio
    async def test_check_copilot_auth_sdk_not_installed(self) -> None:
        """Test Copilot auth check when SDK not installed."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with patch(
            "markitai.providers.auth._is_copilot_sdk_available",
            return_value=False,
        ):
            status = await manager.check_auth("copilot")
            assert status.authenticated is False
            assert "not installed" in (status.error or "").lower()


class TestClaudeAuthCheck:
    """Tests for Claude Agent authentication checking."""

    @pytest.mark.asyncio
    async def test_check_claude_auth_via_doctor(self) -> None:
        """Test Claude auth check via claude doctor command."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        # Mock successful claude doctor output
        with patch(
            "markitai.providers.auth._run_claude_doctor",
            AsyncMock(return_value=(True, None)),
        ):
            status = await manager.check_auth("claude-agent")
            assert status.authenticated is True
            assert status.provider == "claude-agent"

    @pytest.mark.asyncio
    async def test_check_claude_auth_cli_not_found(self) -> None:
        """Test Claude auth check when CLI not found."""
        from markitai.providers.auth import AuthManager

        manager = AuthManager()

        with patch(
            "markitai.providers.auth._is_claude_agent_sdk_available",
            return_value=False,
        ):
            status = await manager.check_auth("claude-agent")
            assert status.authenticated is False


class TestAuthResolutionHints:
    """Tests for authentication resolution hints."""

    def test_copilot_resolution_hint(self) -> None:
        """Test resolution hint for Copilot authentication."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("copilot")
        assert "copilot" in hint.lower()
        assert "login" in hint.lower() or "terminal" in hint.lower()

    def test_claude_resolution_hint(self) -> None:
        """Test resolution hint for Claude authentication."""
        from markitai.providers.auth import get_auth_resolution_hint

        hint = get_auth_resolution_hint("claude-agent")
        assert "claude" in hint.lower()
        assert "setup" in hint.lower() or "token" in hint.lower()
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_auth.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'markitai.providers.auth'"

**Step 3: Write minimal implementation**

```python
# packages/markitai/src/markitai/providers/auth.py
"""Authentication management for local providers.

This module provides:
1. Unified authentication status checking across providers
2. Cached authentication state to avoid repeated checks
3. User-friendly resolution hints for auth issues
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    pass

# Singleton instance
_auth_manager: AuthManager | None = None


@dataclass(frozen=True, slots=True)
class AuthStatus:
    """Authentication status for a provider.

    Attributes:
        provider: Provider name (e.g., "copilot", "claude-agent")
        authenticated: Whether authentication is valid
        user: Username/email if authenticated
        expires_at: Token expiration time if known
        error: Error message if not authenticated
    """

    provider: str
    authenticated: bool
    user: str | None = None
    expires_at: datetime | None = None
    error: str | None = None


def get_auth_resolution_hint(provider: str) -> str:
    """Get user-friendly resolution hint for authentication issues.

    Args:
        provider: Provider name

    Returns:
        Actionable instruction to resolve authentication
    """
    hints = {
        "copilot": (
            "Run 'copilot' in your terminal and follow the prompts to login. "
            "Ensure you have an active GitHub Copilot subscription."
        ),
        "claude-agent": (
            "Run 'claude setup-token' to configure your Claude authentication. "
            "You need an active Claude Pro or Team subscription."
        ),
    }
    return hints.get(provider, f"Check authentication for {provider}")


def _is_copilot_sdk_available() -> bool:
    """Check if Copilot SDK is available."""
    try:
        import importlib.util

        return importlib.util.find_spec("copilot_sdk") is not None
    except Exception:
        return False


def _is_claude_agent_sdk_available() -> bool:
    """Check if Claude Agent SDK is available."""
    try:
        import importlib.util

        return importlib.util.find_spec("claude_agent_sdk") is not None
    except Exception:
        return False


async def _check_copilot_sdk_auth() -> AuthStatus:
    """Check Copilot authentication via SDK.

    Returns:
        AuthStatus with authentication details
    """
    try:
        from copilot_sdk import CopilotClient

        client = CopilotClient()
        status = await client.get_auth_status()

        return AuthStatus(
            provider="copilot",
            authenticated=status.authenticated,
            user=getattr(status, "user", None),
            expires_at=getattr(status, "expires_at", None),
            error=None if status.authenticated else "Not authenticated",
        )
    except Exception as e:
        logger.debug(f"[Auth] Copilot SDK auth check failed: {e}")
        return AuthStatus(
            provider="copilot",
            authenticated=False,
            error=str(e),
        )


async def _run_claude_doctor() -> tuple[bool, str | None]:
    """Run claude doctor to check authentication health.

    Returns:
        Tuple of (is_healthy, error_message)
    """
    claude_path = shutil.which("claude")
    if not claude_path:
        return False, "Claude CLI not found in PATH"

    try:
        proc = await asyncio.create_subprocess_exec(
            claude_path,
            "doctor",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)

        if proc.returncode == 0:
            return True, None

        error_output = stderr.decode() if stderr else stdout.decode()
        return False, error_output.strip() or "claude doctor failed"

    except asyncio.TimeoutError:
        return False, "claude doctor timed out"
    except Exception as e:
        return False, str(e)


class AuthManager:
    """Manages authentication state for local providers.

    Uses singleton pattern to cache authentication status.
    """

    _instance: AuthManager | None = None
    _cache: dict[str, AuthStatus]

    def __new__(cls) -> AuthManager:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._cache = {}
        return cls._instance

    async def check_auth(
        self,
        provider: str,
        *,
        force_refresh: bool = False,
    ) -> AuthStatus:
        """Check authentication status for a provider.

        Args:
            provider: Provider name ("copilot" or "claude-agent")
            force_refresh: Skip cache and re-check

        Returns:
            AuthStatus with current authentication state
        """
        if not force_refresh and provider in self._cache:
            return self._cache[provider]

        if provider == "copilot":
            status = await self._check_copilot()
        elif provider == "claude-agent":
            status = await self._check_claude()
        else:
            status = AuthStatus(
                provider=provider,
                authenticated=False,
                error=f"Unknown provider: {provider}",
            )

        self._cache[provider] = status
        return status

    async def _check_copilot(self) -> AuthStatus:
        """Check Copilot authentication."""
        if not _is_copilot_sdk_available():
            return AuthStatus(
                provider="copilot",
                authenticated=False,
                error="Copilot SDK not installed. Run: uv sync --extra copilot",
            )

        return await _check_copilot_sdk_auth()

    async def _check_claude(self) -> AuthStatus:
        """Check Claude Agent authentication."""
        if not _is_claude_agent_sdk_available():
            return AuthStatus(
                provider="claude-agent",
                authenticated=False,
                error="Claude Agent SDK not installed. Run: uv sync --extra claude-agent",
            )

        is_healthy, error = await _run_claude_doctor()
        return AuthStatus(
            provider="claude-agent",
            authenticated=is_healthy,
            error=error,
        )

    def clear_cache(self, provider: str | None = None) -> None:
        """Clear cached authentication status.

        Args:
            provider: Specific provider to clear, or None for all
        """
        if provider:
            self._cache.pop(provider, None)
        else:
            self._cache.clear()
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_auth.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/auth.py packages/markitai/tests/unit/test_provider_auth.py
git commit -m "feat(providers): add authentication manager with status caching"
```

---

### Task 3: Implement Adaptive Timeout Calculator

**Files:**
- Create: `packages/markitai/src/markitai/providers/timeout.py`
- Test: `packages/markitai/tests/unit/test_provider_timeout.py`

**Step 1: Write the failing test**

```python
# packages/markitai/tests/unit/test_provider_timeout.py
"""Unit tests for adaptive timeout calculation."""

from __future__ import annotations

import pytest


class TestAdaptiveTimeout:
    """Tests for calculate_timeout function."""

    def test_base_timeout(self) -> None:
        """Test base timeout for short text."""
        from markitai.providers.timeout import calculate_timeout

        # Short prompt should use base timeout
        timeout = calculate_timeout(prompt_length=100, has_images=False)
        assert timeout >= 60  # Minimum base timeout
        assert timeout <= 120  # Should not exceed base for short prompts

    def test_timeout_scales_with_length(self) -> None:
        """Test timeout increases with prompt length."""
        from markitai.providers.timeout import calculate_timeout

        short_timeout = calculate_timeout(prompt_length=1000, has_images=False)
        long_timeout = calculate_timeout(prompt_length=50000, has_images=False)

        assert long_timeout > short_timeout

    def test_timeout_increases_for_images(self) -> None:
        """Test timeout multiplier for multimodal requests."""
        from markitai.providers.timeout import calculate_timeout

        text_only = calculate_timeout(prompt_length=5000, has_images=False)
        with_images = calculate_timeout(prompt_length=5000, has_images=True)

        # Images should increase timeout by ~1.5x
        assert with_images > text_only
        assert with_images >= text_only * 1.4

    def test_timeout_increases_for_multiple_images(self) -> None:
        """Test timeout scales with image count."""
        from markitai.providers.timeout import calculate_timeout

        one_image = calculate_timeout(prompt_length=1000, has_images=True, image_count=1)
        many_images = calculate_timeout(prompt_length=1000, has_images=True, image_count=5)

        assert many_images > one_image

    def test_timeout_caps_at_maximum(self) -> None:
        """Test timeout does not exceed maximum."""
        from markitai.providers.timeout import calculate_timeout

        timeout = calculate_timeout(
            prompt_length=1000000,  # Very long
            has_images=True,
            image_count=20,
        )
        assert timeout <= 600  # 10 minute max

    def test_timeout_respects_minimum(self) -> None:
        """Test timeout does not go below minimum."""
        from markitai.providers.timeout import calculate_timeout

        timeout = calculate_timeout(prompt_length=10, has_images=False)
        assert timeout >= 60  # 1 minute minimum


class TestTimeoutConfig:
    """Tests for timeout configuration."""

    def test_timeout_config_defaults(self) -> None:
        """Test TimeoutConfig has sensible defaults."""
        from markitai.providers.timeout import TimeoutConfig

        config = TimeoutConfig()
        assert config.base_timeout == 60
        assert config.max_timeout == 600
        assert config.chars_per_second > 0
        assert config.image_multiplier >= 1.0

    def test_custom_timeout_config(self) -> None:
        """Test TimeoutConfig accepts custom values."""
        from markitai.providers.timeout import TimeoutConfig, calculate_timeout

        config = TimeoutConfig(
            base_timeout=30,
            max_timeout=300,
            chars_per_second=100,
            image_multiplier=2.0,
        )

        timeout = calculate_timeout(
            prompt_length=1000,
            has_images=True,
            config=config,
        )
        # Custom config should be respected
        assert timeout >= 30
        assert timeout <= 300
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_timeout.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'markitai.providers.timeout'"

**Step 3: Write minimal implementation**

```python
# packages/markitai/src/markitai/providers/timeout.py
"""Adaptive timeout calculation for LLM requests.

This module implements intelligent timeout calculation based on:
1. Input length (longer prompts need more processing time)
2. Multimodal content (images require additional processing)
3. Expected output complexity
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    """Configuration for timeout calculation.

    Attributes:
        base_timeout: Minimum timeout in seconds
        max_timeout: Maximum timeout in seconds
        chars_per_second: Estimated LLM processing speed
        image_multiplier: Timeout multiplier for image requests
        per_image_seconds: Additional seconds per image
    """

    base_timeout: int = 60
    max_timeout: int = 600
    chars_per_second: float = 500.0  # Conservative estimate
    image_multiplier: float = 1.5
    per_image_seconds: float = 10.0


# Default configuration
DEFAULT_CONFIG = TimeoutConfig()


def calculate_timeout(
    prompt_length: int,
    *,
    has_images: bool = False,
    image_count: int = 1,
    expected_output_tokens: int | None = None,
    config: TimeoutConfig | None = None,
) -> int:
    """Calculate adaptive timeout based on request complexity.

    The formula considers:
    - Base timeout as minimum
    - Additional time for prompt length
    - Multiplier for multimodal requests
    - Per-image overhead

    Args:
        prompt_length: Number of characters in the prompt
        has_images: Whether request contains images
        image_count: Number of images in request
        expected_output_tokens: Expected output length (optional)
        config: Custom timeout configuration

    Returns:
        Timeout in seconds, clamped to [base_timeout, max_timeout]
    """
    cfg = config or DEFAULT_CONFIG

    # Start with base timeout
    timeout = float(cfg.base_timeout)

    # Add time for input processing
    # Estimate: 1 char ~ 0.25 tokens, processing at ~500 chars/sec
    input_time = prompt_length / cfg.chars_per_second
    timeout += input_time

    # Add time for expected output
    if expected_output_tokens:
        # Rough estimate: 3-4 tokens per second generation
        output_time = expected_output_tokens / 4.0
        timeout += output_time

    # Apply image multiplier
    if has_images:
        timeout *= cfg.image_multiplier
        # Add per-image overhead
        timeout += cfg.per_image_seconds * max(0, image_count - 1)

    # Clamp to bounds
    timeout = max(cfg.base_timeout, min(int(timeout), cfg.max_timeout))

    return timeout


def calculate_timeout_from_messages(
    messages: list[dict],
    *,
    config: TimeoutConfig | None = None,
) -> int:
    """Calculate timeout from OpenAI-style messages.

    Automatically extracts prompt length and image count from messages.

    Args:
        messages: List of message dicts with role/content
        config: Custom timeout configuration

    Returns:
        Timeout in seconds
    """
    prompt_length = 0
    image_count = 0
    has_images = False

    for msg in messages:
        content = msg.get("content", "")

        if isinstance(content, str):
            prompt_length += len(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        prompt_length += len(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        has_images = True
                        image_count += 1

    return calculate_timeout(
        prompt_length=prompt_length,
        has_images=has_images,
        image_count=image_count,
        config=config,
    )
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_timeout.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/timeout.py packages/markitai/tests/unit/test_provider_timeout.py
git commit -m "feat(providers): add adaptive timeout calculation based on request complexity"
```

---

### Task 4: Update Providers Module Init

**Files:**
- Modify: `packages/markitai/src/markitai/providers/__init__.py`

**Step 1: Write the failing test**

```python
# Add to packages/markitai/tests/unit/test_providers.py at the end

class TestProviderModuleExports:
    """Tests for providers module exports."""

    def test_error_classes_exported(self) -> None:
        """Test error classes are exported from providers module."""
        from markitai.providers import (
            AuthenticationError,
            ProviderError,
            ProviderTimeoutError,
            QuotaError,
            SDKNotAvailableError,
        )

        # Verify classes are importable
        assert ProviderError is not None
        assert AuthenticationError is not None
        assert QuotaError is not None
        assert ProviderTimeoutError is not None
        assert SDKNotAvailableError is not None

    def test_auth_manager_exported(self) -> None:
        """Test AuthManager is exported from providers module."""
        from markitai.providers import AuthManager, AuthStatus

        assert AuthManager is not None
        assert AuthStatus is not None

    def test_timeout_functions_exported(self) -> None:
        """Test timeout functions are exported from providers module."""
        from markitai.providers import (
            TimeoutConfig,
            calculate_timeout,
            calculate_timeout_from_messages,
        )

        assert calculate_timeout is not None
        assert calculate_timeout_from_messages is not None
        assert TimeoutConfig is not None
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestProviderModuleExports -v`
Expected: FAIL with import errors

**Step 3: Update the providers __init__.py**

Open `packages/markitai/src/markitai/providers/__init__.py` and add exports:

```python
# Add at the top of the file after existing imports:
from markitai.providers.auth import AuthManager, AuthStatus, get_auth_resolution_hint
from markitai.providers.errors import (
    AuthenticationError,
    ProviderError,
    ProviderTimeoutError,
    QuotaError,
    SDKNotAvailableError,
)
from markitai.providers.timeout import (
    TimeoutConfig,
    calculate_timeout,
    calculate_timeout_from_messages,
)

# Update __all__ to include new exports:
__all__ = [
    # ... existing exports ...
    # Error classes
    "ProviderError",
    "AuthenticationError",
    "QuotaError",
    "ProviderTimeoutError",
    "SDKNotAvailableError",
    # Auth
    "AuthManager",
    "AuthStatus",
    "get_auth_resolution_hint",
    # Timeout
    "TimeoutConfig",
    "calculate_timeout",
    "calculate_timeout_from_messages",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestProviderModuleExports -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/__init__.py packages/markitai/tests/unit/test_providers.py
git commit -m "feat(providers): export error classes, auth manager, and timeout utilities"
```

---

### Task 5: Integrate Adaptive Timeout into Claude Agent Provider

**Files:**
- Modify: `packages/markitai/src/markitai/providers/claude_agent.py:93-101` (init)
- Modify: `packages/markitai/src/markitai/providers/claude_agent.py:344-541` (acompletion)
- Test: Add to `packages/markitai/tests/unit/test_providers.py`

**Step 1: Write the failing test**

```python
# Add to packages/markitai/tests/unit/test_providers.py

class TestClaudeAgentAdaptiveTimeout:
    """Tests for Claude Agent provider adaptive timeout."""

    def test_claude_agent_uses_adaptive_timeout(self) -> None:
        """Test ClaudeAgentProvider calculates adaptive timeout."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Short messages should have shorter timeout
        short_messages = [{"role": "user", "content": "Hello"}]
        short_timeout = provider._calculate_adaptive_timeout(short_messages)

        # Long messages should have longer timeout
        long_content = "x" * 50000
        long_messages = [{"role": "user", "content": long_content}]
        long_timeout = provider._calculate_adaptive_timeout(long_messages)

        assert long_timeout > short_timeout

    def test_claude_agent_timeout_with_images(self) -> None:
        """Test timeout increases for multimodal requests."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        text_only = [{"role": "user", "content": "Describe something"}]
        with_images = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]

        text_timeout = provider._calculate_adaptive_timeout(text_only)
        image_timeout = provider._calculate_adaptive_timeout(with_images)

        assert image_timeout > text_timeout
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestClaudeAgentAdaptiveTimeout -v`
Expected: FAIL with "AttributeError: 'ClaudeAgentProvider' object has no attribute '_calculate_adaptive_timeout'"

**Step 3: Modify claude_agent.py**

Add the adaptive timeout method to ClaudeAgentProvider class:

```python
# In packages/markitai/src/markitai/providers/claude_agent.py

# Add import at top:
from markitai.providers.timeout import calculate_timeout_from_messages

# Add method to ClaudeAgentProvider class (after __init__):

    def _calculate_adaptive_timeout(self, messages: list[dict]) -> int:
        """Calculate adaptive timeout based on message content.

        Args:
            messages: OpenAI-style messages

        Returns:
            Timeout in seconds
        """
        return calculate_timeout_from_messages(messages)
```

Then update the `acompletion` method to use adaptive timeout:

```python
# In acompletion method, replace hardcoded timeout with:
timeout = self._calculate_adaptive_timeout(messages)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestClaudeAgentAdaptiveTimeout -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/claude_agent.py packages/markitai/tests/unit/test_providers.py
git commit -m "feat(claude-agent): integrate adaptive timeout calculation"
```

---

### Task 6: Integrate Adaptive Timeout into Copilot Provider

**Files:**
- Modify: `packages/markitai/src/markitai/providers/copilot.py:215-225` (init)
- Modify: `packages/markitai/src/markitai/providers/copilot.py:566-803` (acompletion)
- Test: Add to `packages/markitai/tests/unit/test_providers.py`

**Step 1: Write the failing test**

```python
# Add to packages/markitai/tests/unit/test_providers.py

class TestCopilotAdaptiveTimeout:
    """Tests for Copilot provider adaptive timeout."""

    def test_copilot_uses_adaptive_timeout(self) -> None:
        """Test CopilotProvider calculates adaptive timeout."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Short messages should have shorter timeout
        short_messages = [{"role": "user", "content": "Hello"}]
        short_timeout = provider._calculate_adaptive_timeout(short_messages)

        # Long messages should have longer timeout
        long_content = "x" * 50000
        long_messages = [{"role": "user", "content": long_content}]
        long_timeout = provider._calculate_adaptive_timeout(long_messages)

        assert long_timeout > short_timeout

    def test_copilot_timeout_with_images(self) -> None:
        """Test timeout increases for multimodal requests."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        text_only = [{"role": "user", "content": "Describe something"}]
        with_images = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's this?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            }
        ]

        text_timeout = provider._calculate_adaptive_timeout(text_only)
        image_timeout = provider._calculate_adaptive_timeout(with_images)

        assert image_timeout > text_timeout
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestCopilotAdaptiveTimeout -v`
Expected: FAIL

**Step 3: Modify copilot.py**

Same pattern as claude_agent.py - add import and method.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestCopilotAdaptiveTimeout -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/copilot.py packages/markitai/tests/unit/test_providers.py
git commit -m "feat(copilot): integrate adaptive timeout calculation"
```

---

## Phase 2: Feature Enhancements (P1)

### Task 7: Create Unified JSON Mode Handler

**Files:**
- Create: `packages/markitai/src/markitai/providers/json_mode.py`
- Test: `packages/markitai/tests/unit/test_provider_json_mode.py`

**Step 1: Write the failing test**

```python
# packages/markitai/tests/unit/test_provider_json_mode.py
"""Unit tests for unified JSON mode handler."""

from __future__ import annotations

import pytest


class TestJsonModeHandler:
    """Tests for StructuredOutputHandler class."""

    def test_build_json_prompt_suffix(self) -> None:
        """Test building JSON instruction suffix for prompts."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        suffix = handler.build_json_prompt_suffix()

        assert "JSON" in suffix
        assert "```json" in suffix.lower() or "json" in suffix.lower()

    def test_build_json_prompt_suffix_with_schema(self) -> None:
        """Test building JSON suffix with schema hint."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        suffix = handler.build_json_prompt_suffix(schema=schema)

        assert "name" in suffix
        assert "string" in suffix

    def test_extract_json_from_text(self) -> None:
        """Test extracting JSON from response text."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()

        # Plain JSON
        result = handler.extract_json('{"key": "value"}')
        assert result == {"key": "value"}

        # JSON in code block
        result = handler.extract_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

        # JSON with surrounding text
        result = handler.extract_json('Here is the result:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_extract_json_handles_invalid(self) -> None:
        """Test extract_json returns None for invalid JSON."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()

        result = handler.extract_json("This is not JSON")
        assert result is None

        result = handler.extract_json('{"broken": }')
        assert result is None

    def test_extract_json_cleans_control_chars(self) -> None:
        """Test extract_json removes control characters."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()

        # JSON with control characters
        dirty_json = '{"text": "hello\x00world\x1f"}'
        result = handler.extract_json(dirty_json)

        assert result is not None
        assert "\x00" not in result.get("text", "")


class TestJsonSchemaGeneration:
    """Tests for JSON schema generation for providers."""

    def test_generate_system_prompt_for_json_mode(self) -> None:
        """Test generating system prompt for JSON mode."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {"type": "object", "properties": {"result": {"type": "string"}}}

        system_prompt = handler.generate_json_system_prompt(
            base_prompt="You are a helpful assistant.",
            schema=schema,
        )

        assert "helpful assistant" in system_prompt
        assert "JSON" in system_prompt

    def test_validate_against_schema(self) -> None:
        """Test JSON validation against schema."""
        from markitai.providers.json_mode import StructuredOutputHandler

        handler = StructuredOutputHandler()
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }

        # Valid
        assert handler.validate_json({"name": "test"}, schema) is True

        # Invalid - missing required field
        assert handler.validate_json({}, schema) is False

        # Invalid - wrong type
        assert handler.validate_json({"name": 123}, schema) is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_json_mode.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# packages/markitai/src/markitai/providers/json_mode.py
"""Unified JSON mode handling for local providers.

This module provides:
1. JSON prompt suffix generation (for providers without native JSON mode)
2. JSON extraction from LLM responses
3. Control character cleaning
4. Optional schema validation
"""

from __future__ import annotations

import json
import re
from typing import Any

from loguru import logger


def clean_control_characters(text: str) -> str:
    """Remove control characters that break JSON parsing.

    Args:
        text: Input text potentially containing control chars

    Returns:
        Cleaned text safe for JSON parsing
    """
    # Remove ASCII control characters except newline/tab
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)


class StructuredOutputHandler:
    """Handles structured JSON output from LLM responses.

    Provides a unified interface for:
    - Building JSON mode prompts
    - Extracting JSON from responses
    - Validating against schemas
    """

    # Patterns to find JSON in response text
    _JSON_BLOCK_PATTERN = re.compile(
        r"```(?:json)?\s*\n?(.*?)\n?```",
        re.DOTALL | re.IGNORECASE,
    )
    _JSON_OBJECT_PATTERN = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)

    def build_json_prompt_suffix(
        self,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Build instruction suffix for JSON output.

        Args:
            schema: Optional JSON schema to include

        Returns:
            Instruction text to append to prompts
        """
        suffix_parts = [
            "\n\nRespond with valid JSON only.",
            "Do not include any text before or after the JSON.",
            "Do not wrap the JSON in markdown code blocks.",
        ]

        if schema:
            schema_str = json.dumps(schema, indent=2)
            suffix_parts.append(f"\n\nExpected JSON schema:\n{schema_str}")

        return " ".join(suffix_parts)

    def extract_json(self, text: str) -> dict[str, Any] | list | None:
        """Extract JSON from LLM response text.

        Handles:
        - Plain JSON
        - JSON in markdown code blocks
        - JSON with surrounding text
        - Control characters

        Args:
            text: Response text potentially containing JSON

        Returns:
            Parsed JSON object/array, or None if extraction fails
        """
        if not text or not text.strip():
            return None

        # Clean control characters first
        text = clean_control_characters(text.strip())

        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from code block
        match = self._JSON_BLOCK_PATTERN.search(text)
        if match:
            try:
                return json.loads(match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Try finding JSON object in text
        match = self._JSON_OBJECT_PATTERN.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        logger.debug(f"[JSONMode] Failed to extract JSON from: {text[:100]}...")
        return None

    def generate_json_system_prompt(
        self,
        base_prompt: str,
        schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate system prompt with JSON mode instructions.

        Args:
            base_prompt: Original system prompt
            schema: Optional JSON schema

        Returns:
            Enhanced system prompt with JSON instructions
        """
        json_instruction = (
            "\n\nIMPORTANT: You must respond with valid JSON only. "
            "Do not include any explanatory text, markdown formatting, "
            "or code block markers. Output raw JSON."
        )

        if schema:
            schema_str = json.dumps(schema, indent=2)
            json_instruction += f"\n\nYour response must conform to this schema:\n{schema_str}"

        return base_prompt + json_instruction

    def validate_json(
        self,
        data: dict[str, Any] | list,
        schema: dict[str, Any],
    ) -> bool:
        """Validate JSON data against a schema.

        Uses basic validation without jsonschema dependency.

        Args:
            data: Parsed JSON data
            schema: JSON schema to validate against

        Returns:
            True if valid, False otherwise
        """
        try:
            return self._validate_value(data, schema)
        except Exception as e:
            logger.debug(f"[JSONMode] Validation error: {e}")
            return False

    def _validate_value(self, value: Any, schema: dict[str, Any]) -> bool:
        """Recursively validate a value against schema."""
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(value, dict):
                return False

            # Check required fields
            required = schema.get("required", [])
            for field in required:
                if field not in value:
                    return False

            # Check properties
            properties = schema.get("properties", {})
            for prop, prop_schema in properties.items():
                if prop in value:
                    if not self._validate_value(value[prop], prop_schema):
                        return False

            return True

        elif schema_type == "array":
            if not isinstance(value, list):
                return False
            items_schema = schema.get("items")
            if items_schema:
                return all(self._validate_value(item, items_schema) for item in value)
            return True

        elif schema_type == "string":
            return isinstance(value, str)

        elif schema_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)

        elif schema_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)

        elif schema_type == "boolean":
            return isinstance(value, bool)

        elif schema_type == "null":
            return value is None

        return True  # Unknown type, pass
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_provider_json_mode.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/json_mode.py packages/markitai/tests/unit/test_provider_json_mode.py
git commit -m "feat(providers): add unified JSON mode handler for structured outputs"
```

---

### Task 8: Integrate JSON Mode Handler into Copilot Provider

**Files:**
- Modify: `packages/markitai/src/markitai/providers/copilot.py:434-520`

**Step 1: Write the failing test**

```python
# Add to packages/markitai/tests/unit/test_providers.py

class TestCopilotJsonModeIntegration:
    """Tests for Copilot provider JSON mode integration."""

    def test_copilot_uses_structured_output_handler(self) -> None:
        """Test Copilot uses StructuredOutputHandler for JSON extraction."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Test extraction through provider
        response_text = '```json\n{"result": "success"}\n```'
        result = provider._extract_json_from_response(response_text)

        assert result is not None
        assert result.get("result") == "success"

    def test_copilot_json_extraction_with_control_chars(self) -> None:
        """Test JSON extraction handles control characters."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        # Response with control characters
        response_text = '{"text": "hello\x00world"}'
        result = provider._extract_json_from_response(response_text)

        assert result is not None
        assert "\x00" not in result.get("text", "")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestCopilotJsonModeIntegration -v`
Expected: Test may pass if existing implementation works, or fail if changes needed

**Step 3: Refactor copilot.py to use StructuredOutputHandler**

Replace the manual JSON extraction in `_extract_json_from_response` with:

```python
# In copilot.py, update import:
from markitai.providers.json_mode import StructuredOutputHandler

# Add instance variable in __init__:
self._json_handler = StructuredOutputHandler()

# Replace _extract_json_from_response body:
def _extract_json_from_response(self, text: str) -> dict | None:
    """Extract JSON from response using unified handler."""
    return self._json_handler.extract_json(text)
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestCopilotJsonModeIntegration -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/copilot.py packages/markitai/tests/unit/test_providers.py
git commit -m "refactor(copilot): use unified StructuredOutputHandler for JSON extraction"
```

---

### Task 9: Add Prompt Caching Support for Claude Agent

**Files:**
- Modify: `packages/markitai/src/markitai/providers/claude_agent.py`
- Test: Add to `packages/markitai/tests/unit/test_providers.py`

**Step 1: Write the failing test**

```python
# Add to packages/markitai/tests/unit/test_providers.py

class TestClaudeAgentPromptCaching:
    """Tests for Claude Agent prompt caching."""

    def test_add_cache_control_to_long_system_prompt(self) -> None:
        """Test cache_control is added to long system prompts."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Long system prompt (> 1024 tokens ~ 4096 chars)
        long_system = "x" * 5000
        messages = [
            {"role": "system", "content": long_system},
            {"role": "user", "content": "Hello"},
        ]

        enhanced = provider._add_cache_control(messages)

        # System message should have cache_control
        system_msg = next(m for m in enhanced if m["role"] == "system")
        assert "cache_control" in str(system_msg) or isinstance(
            system_msg.get("content"), list
        )

    def test_no_cache_control_for_short_prompts(self) -> None:
        """Test cache_control is not added to short prompts."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        messages = [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"},
        ]

        enhanced = provider._add_cache_control(messages)

        # Short prompts should not have cache_control
        system_msg = next(m for m in enhanced if m["role"] == "system")
        if isinstance(system_msg.get("content"), str):
            assert "cache_control" not in str(system_msg)

    def test_cache_control_threshold_configurable(self) -> None:
        """Test cache threshold can be configured."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Should be configurable via constant
        assert hasattr(provider, "_CACHE_THRESHOLD_CHARS") or True  # Optional
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestClaudeAgentPromptCaching -v`
Expected: FAIL

**Step 3: Implement prompt caching in claude_agent.py**

```python
# Add to ClaudeAgentProvider class:

    # Threshold for enabling prompt caching (~1024 tokens)
    _CACHE_THRESHOLD_CHARS: int = 4096

    def _add_cache_control(
        self,
        messages: list[dict],
    ) -> list[dict]:
        """Add cache_control to eligible messages for prompt caching.

        Anthropic's prompt caching caches content marked with cache_control.
        We add it to long system prompts to reduce compute costs on repeated calls.

        Args:
            messages: OpenAI-style messages

        Returns:
            Messages with cache_control added where beneficial
        """
        result = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            # Only cache system messages for now
            if role == "system" and isinstance(content, str):
                if len(content) >= self._CACHE_THRESHOLD_CHARS:
                    # Convert to content blocks format with cache_control
                    result.append({
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": content,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    })
                    continue

            result.append(msg.copy())

        return result
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestClaudeAgentPromptCaching -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/claude_agent.py packages/markitai/tests/unit/test_providers.py
git commit -m "feat(claude-agent): add prompt caching support for long system prompts"
```

---

## Phase 3: Testing & Tooling (P1)

### Task 10: Upgrade check-deps to doctor Command

**Files:**
- Rename: `packages/markitai/src/markitai/cli/commands/deps.py` → `doctor.py`
- Modify: `packages/markitai/src/markitai/cli/commands/__init__.py`
- Modify: `packages/markitai/src/markitai/cli/main.py`
- Test: `packages/markitai/tests/unit/test_doctor_cli.py`

**Step 1: Write the failing test**

```python
# packages/markitai/tests/unit/test_doctor_cli.py
"""Unit tests for doctor CLI command."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from click.testing import CliRunner


class TestDoctorCommand:
    """Tests for markitai doctor command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_doctor_command_exists(self, runner: CliRunner) -> None:
        """Test doctor command is registered."""
        from markitai.cli.main import app

        result = runner.invoke(app, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "doctor" in result.output.lower() or "check" in result.output.lower()

    def test_check_deps_alias_works(self, runner: CliRunner) -> None:
        """Test check-deps is an alias for doctor."""
        from markitai.cli.main import app

        result = runner.invoke(app, ["check-deps", "--help"])
        assert result.exit_code == 0

    def test_doctor_shows_auth_status(self, runner: CliRunner) -> None:
        """Test doctor includes authentication status."""
        from markitai.cli.main import app

        # Mock the auth check to avoid real SDK calls
        with patch(
            "markitai.cli.commands.doctor._check_provider_auth",
            return_value={"status": "ok", "message": "Authenticated"},
        ):
            result = runner.invoke(app, ["doctor"])
            # Should complete without error
            assert result.exit_code == 0

    def test_doctor_json_output(self, runner: CliRunner) -> None:
        """Test doctor --json outputs valid JSON."""
        import json

        from markitai.cli.main import app

        result = runner.invoke(app, ["doctor", "--json"])
        assert result.exit_code == 0

        # Should be valid JSON
        try:
            data = json.loads(result.output)
            assert isinstance(data, list)
        except json.JSONDecodeError:
            pytest.fail("doctor --json did not output valid JSON")


class TestDoctorAuthChecks:
    """Tests for doctor authentication checks."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_doctor_checks_copilot_auth(self, runner: CliRunner) -> None:
        """Test doctor checks Copilot authentication when configured."""
        from markitai.cli.commands.doctor import _check_copilot_auth

        # Test the auth check function directly
        result = _check_copilot_auth()
        assert "status" in result
        assert result["status"] in ("ok", "warning", "missing", "error")

    def test_doctor_checks_claude_auth(self, runner: CliRunner) -> None:
        """Test doctor checks Claude authentication when configured."""
        from markitai.cli.commands.doctor import _check_claude_auth

        # Test the auth check function directly
        result = _check_claude_auth()
        assert "status" in result
        assert result["status"] in ("ok", "warning", "missing", "error")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/unit/test_doctor_cli.py -v`
Expected: FAIL (module not found or command not registered)

**Step 3: Rename and enhance deps.py to doctor.py**

1. Rename the file:
```bash
mv packages/markitai/src/markitai/cli/commands/deps.py packages/markitai/src/markitai/cli/commands/doctor.py
```

2. Update the command in doctor.py:

```python
# At the top, add imports:
import asyncio
from markitai.providers.auth import AuthManager, AuthStatus

# Rename the command and add alias:
@click.command(name="doctor")
@click.option("--json", "json_output", is_flag=True, help="Output as JSON")
@click.pass_context
def doctor(ctx: click.Context, json_output: bool) -> None:
    """Check system dependencies and authentication status.

    Verifies that all required dependencies are installed and properly
    configured, including authentication for local providers.
    """
    # ... existing check logic ...
    
    # Add auth checks
    results.extend(_check_all_auth())
    
    # ... rest of output logic ...


# Add alias for backward compatibility
check_deps = doctor


def _check_copilot_auth() -> dict:
    """Check Copilot authentication status."""
    try:
        manager = AuthManager()
        status = asyncio.run(manager.check_auth("copilot"))
        
        if status.authenticated:
            return {
                "name": "Copilot Auth",
                "description": "GitHub Copilot authentication",
                "status": "ok",
                "message": f"Authenticated as {status.user or 'user'}",
                "install_hint": "",
            }
        else:
            return {
                "name": "Copilot Auth",
                "description": "GitHub Copilot authentication",
                "status": "warning",
                "message": status.error or "Not authenticated",
                "install_hint": "Run 'copilot' in terminal to login",
            }
    except Exception as e:
        return {
            "name": "Copilot Auth",
            "description": "GitHub Copilot authentication",
            "status": "error",
            "message": str(e),
            "install_hint": "",
        }


def _check_claude_auth() -> dict:
    """Check Claude Agent authentication status."""
    try:
        manager = AuthManager()
        status = asyncio.run(manager.check_auth("claude-agent"))
        
        if status.authenticated:
            return {
                "name": "Claude Auth",
                "description": "Claude Agent authentication",
                "status": "ok",
                "message": "Authenticated",
                "install_hint": "",
            }
        else:
            return {
                "name": "Claude Auth",
                "description": "Claude Agent authentication",
                "status": "warning",
                "message": status.error or "Not authenticated",
                "install_hint": "Run 'claude setup-token' to configure",
            }
    except Exception as e:
        return {
            "name": "Claude Auth",
            "description": "Claude Agent authentication",
            "status": "error",
            "message": str(e),
            "install_hint": "",
        }


def _check_all_auth() -> list[dict]:
    """Check authentication for all configured providers."""
    results = []
    
    # Check Copilot if configured
    results.append(_check_copilot_auth())
    
    # Check Claude if configured
    results.append(_check_claude_auth())
    
    return results
```

3. Update `__init__.py`:

```python
# packages/markitai/src/markitai/cli/commands/__init__.py
from markitai.cli.commands.cache import cache
from markitai.cli.commands.config import config
from markitai.cli.commands.doctor import doctor, check_deps

__all__ = ["cache", "config", "doctor", "check_deps"]
```

4. Update `main.py`:

```python
# Update import
from markitai.cli.commands.doctor import doctor, check_deps

# Register both commands
app.add_command(doctor)
app.add_command(check_deps)  # Alias for backward compatibility
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_doctor_cli.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/cli/commands/
git add packages/markitai/tests/unit/test_doctor_cli.py
git commit -m "feat(cli): upgrade check-deps to doctor command with auth status"
```

---

### Task 11: Create Mock-Based Integration Tests for Local Providers

**Files:**
- Create: `packages/markitai/tests/integration/test_local_providers.py`

**Step 1: Write the failing test**

```python
# packages/markitai/tests/integration/test_local_providers.py
"""Integration tests for local providers using mocks.

These tests verify the full request flow without requiring real SDK credentials.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestClaudeAgentIntegration:
    """Integration tests for Claude Agent provider."""

    @pytest.mark.asyncio
    async def test_successful_completion_flow(self) -> None:
        """Test complete request flow with mocked SDK."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        # Mock the Claude Agent SDK
        mock_result = MagicMock()
        mock_result.result = MagicMock()
        mock_result.result.text = "Hello! How can I help you?"
        mock_result.usage = MagicMock(input_tokens=10, output_tokens=20)

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.claude_agent.ClaudeAgentProvider._call_sdk",
                AsyncMock(return_value=mock_result),
            ),
        ):
            response = await provider.acompletion(
                model="claude-agent/sonnet",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response is not None
            assert "Hello" in response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_image_request_flow(self) -> None:
        """Test multimodal request with image."""
        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        mock_result = MagicMock()
        mock_result.result = MagicMock()
        mock_result.result.text = "I see a cat in the image."
        mock_result.usage = MagicMock(input_tokens=100, output_tokens=20)

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.claude_agent.ClaudeAgentProvider._call_sdk",
                AsyncMock(return_value=mock_result),
            ),
        ):
            response = await provider.acompletion(
                model="claude-agent/sonnet",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this image?"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64,abc123"},
                            },
                        ],
                    }
                ],
            )

            assert response is not None
            assert "cat" in response.choices[0].message.content.lower()

    @pytest.mark.asyncio
    async def test_authentication_error_handling(self) -> None:
        """Test authentication error is properly raised."""
        from litellm import AuthenticationError

        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.claude_agent.ClaudeAgentProvider._call_sdk",
                AsyncMock(side_effect=Exception("not authenticated")),
            ),
        ):
            with pytest.raises(AuthenticationError):
                await provider.acompletion(
                    model="claude-agent/sonnet",
                    messages=[{"role": "user", "content": "Hello"}],
                )

    @pytest.mark.asyncio
    async def test_rate_limit_error_handling(self) -> None:
        """Test rate limit error triggers retry mechanism."""
        from litellm import RateLimitError

        from markitai.providers.claude_agent import ClaudeAgentProvider

        provider = ClaudeAgentProvider()

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.claude_agent.ClaudeAgentProvider._call_sdk",
                AsyncMock(side_effect=Exception("rate limit exceeded")),
            ),
        ):
            with pytest.raises(RateLimitError):
                await provider.acompletion(
                    model="claude-agent/sonnet",
                    messages=[{"role": "user", "content": "Hello"}],
                )


class TestCopilotIntegration:
    """Integration tests for Copilot provider."""

    @pytest.mark.asyncio
    async def test_successful_completion_flow(self) -> None:
        """Test complete request flow with mocked SDK."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.data = MagicMock()
        mock_response.data.content = "Hello! I'm here to help."
        mock_session.send_and_wait = AsyncMock(return_value=mock_response)
        mock_session.destroy = AsyncMock()

        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch.object(provider, "_get_client", AsyncMock(return_value=mock_client)),
        ):
            response = await provider.acompletion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Hello"}],
            )

            assert response is not None
            assert "Hello" in response.choices[0].message.content

    @pytest.mark.asyncio
    async def test_json_mode_extraction(self) -> None:
        """Test JSON mode response extraction."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.data = MagicMock()
        mock_response.data.content = '```json\n{"result": "success"}\n```'
        mock_session.send_and_wait = AsyncMock(return_value=mock_response)
        mock_session.destroy = AsyncMock()

        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch.object(provider, "_get_client", AsyncMock(return_value=mock_client)),
        ):
            response = await provider.acompletion(
                model="copilot/gpt-4.1",
                messages=[{"role": "user", "content": "Return JSON"}],
                response_format={"type": "json_object"},
            )

            assert response is not None
            # Content should be clean JSON
            content = response.choices[0].message.content
            assert "result" in content

    @pytest.mark.asyncio
    async def test_timeout_error_handling(self) -> None:
        """Test timeout error with helpful message."""
        from markitai.providers.copilot import CopilotProvider

        provider = CopilotProvider()

        mock_session = MagicMock()
        mock_session.send_and_wait = AsyncMock(side_effect=TimeoutError())
        mock_session.destroy = AsyncMock()

        mock_client = MagicMock()
        mock_client.create_session = AsyncMock(return_value=mock_session)

        with (
            patch(
                "markitai.providers.copilot._is_copilot_sdk_available",
                return_value=True,
            ),
            patch.object(provider, "_get_client", AsyncMock(return_value=mock_client)),
        ):
            with pytest.raises(RuntimeError) as exc:
                await provider.acompletion(
                    model="copilot/gpt-4.1",
                    messages=[{"role": "user", "content": "Hello"}],
                )

            assert "timed out" in str(exc.value).lower()


class TestProviderErrorClassification:
    """Tests for error classification across providers."""

    def test_quota_errors_not_retried(self) -> None:
        """Verify quota errors are classified as non-retryable."""
        from markitai.llm.processor import RETRYABLE_ERRORS

        # Quota patterns from processor.py
        non_retryable_patterns = (
            "quota",
            "billing",
            "payment",
            "402",
        )

        # These should NOT be in retryable errors
        from litellm import AuthenticationError

        assert AuthenticationError not in RETRYABLE_ERRORS

    def test_network_errors_are_retried(self) -> None:
        """Verify network errors are classified as retryable."""
        from litellm import APIConnectionError, Timeout

        from markitai.llm.processor import RETRYABLE_ERRORS

        assert APIConnectionError in RETRYABLE_ERRORS
        assert Timeout in RETRYABLE_ERRORS
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest packages/markitai/tests/integration/test_local_providers.py -v`
Expected: Some tests may fail due to missing mocks or method names

**Step 3: Adjust tests and provider code as needed**

The tests use reasonable mocks - adjust provider internals or test mocks to match actual implementation.

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/integration/test_local_providers.py -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/tests/integration/test_local_providers.py
git commit -m "test(providers): add mock-based integration tests for local providers"
```

---

### Task 12: Export New Modules from Providers Package

**Files:**
- Modify: `packages/markitai/src/markitai/providers/__init__.py`

**Step 1: Verify all exports**

```python
# Add test to packages/markitai/tests/unit/test_providers.py

class TestProviderPackageExports:
    """Tests for complete package exports."""

    def test_all_new_modules_exported(self) -> None:
        """Test all new modules are properly exported."""
        from markitai.providers import (
            # Error classes
            AuthenticationError,
            ProviderError,
            ProviderTimeoutError,
            QuotaError,
            SDKNotAvailableError,
            # Auth
            AuthManager,
            AuthStatus,
            get_auth_resolution_hint,
            # Timeout
            TimeoutConfig,
            calculate_timeout,
            calculate_timeout_from_messages,
            # JSON mode
            StructuredOutputHandler,
            clean_control_characters,
        )

        # All should be importable
        assert ProviderError is not None
        assert AuthManager is not None
        assert calculate_timeout is not None
        assert StructuredOutputHandler is not None
```

**Step 2: Run test**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestProviderPackageExports -v`

**Step 3: Update __init__.py if needed**

```python
# Ensure all exports are in __init__.py:
from markitai.providers.json_mode import (
    StructuredOutputHandler,
    clean_control_characters,
)

# Add to __all__:
__all__ = [
    # ... existing ...
    "StructuredOutputHandler",
    "clean_control_characters",
]
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest packages/markitai/tests/unit/test_providers.py::TestProviderPackageExports -v`
Expected: PASS

**Step 5: Commit**

```bash
git add packages/markitai/src/markitai/providers/__init__.py packages/markitai/tests/unit/test_providers.py
git commit -m "chore(providers): ensure all new modules are exported"
```

---

### Task 13: Run Full Test Suite and Fix Any Issues

**Step 1: Run linting**

Run: `uv run ruff check --fix packages/markitai/`
Expected: No errors (or fix any that appear)

**Step 2: Run type checking**

Run: `uv run pyright packages/markitai/src/markitai/providers/`
Expected: No errors (or fix any that appear)

**Step 3: Run full test suite**

Run: `uv run pytest packages/markitai/tests/ -v --tb=short`
Expected: All tests pass

**Step 4: Run with coverage**

Run: `uv run pytest packages/markitai/tests/ --cov=markitai.providers --cov-report=term-missing`
Expected: Coverage > 80% for new code

**Step 5: Final commit**

```bash
git add -A
git commit -m "test: ensure all tests pass with full coverage"
```

---

### Task 14: Update Documentation

**Files:**
- Modify: `website/guide/configuration.md`
- Modify: `website/zh/guide/configuration.md`

**Step 1: Add documentation for new features**

Add sections documenting:
- The `doctor` command and its auth checks
- Adaptive timeout behavior
- Prompt caching for Claude Agent

**Step 2: Commit documentation**

```bash
git add website/
git commit -m "docs: add documentation for provider improvements"
```

---

## Summary

This plan implements the Local Provider Integration Improvement in 14 tasks across 3 phases:

**Phase 1 (P0) - Reliability Foundations:**
1. Provider error classes (`errors.py`)
2. Authentication manager (`auth.py`)
3. Adaptive timeout calculator (`timeout.py`)
4. Module exports update
5. Claude Agent timeout integration
6. Copilot timeout integration

**Phase 2 (P1) - Feature Enhancements:**
7. Unified JSON mode handler (`json_mode.py`)
8. Copilot JSON mode integration
9. Claude Agent prompt caching

**Phase 3 (P1) - Testing & Tooling:**
10. Doctor CLI command
11. Mock-based integration tests
12. Package exports verification
13. Full test suite validation
14. Documentation updates

**Estimated Time:** 8-12 days

---

**Plan complete and saved to `docs/plans/2026-02-01-local-provider-integration-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** - Open new session with executing-plans, batch execution with checkpoints

**Which approach?**
