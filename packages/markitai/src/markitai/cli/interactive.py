"""Interactive CLI mode for guided setup.

This module provides an interactive TUI for users who prefer guided
configuration over command-line flags.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ProviderDetectionResult:
    """Result of LLM provider auto-detection."""

    provider: str
    model: str
    authenticated: bool
    source: str  # "cli", "env", "config"


def _check_claude_auth() -> bool:
    """Check if Claude CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("claude-agent"))
        return status.authenticated
    except Exception:
        return False


def _check_copilot_auth() -> bool:
    """Check if Copilot CLI is authenticated."""
    from markitai.providers.auth import AuthManager

    auth_manager = AuthManager()
    try:
        status = asyncio.run(auth_manager.check_auth("copilot"))
        return status.authenticated
    except Exception:
        return False


def detect_llm_provider() -> ProviderDetectionResult | None:
    """Auto-detect available LLM provider.

    Detection priority:
    1. Claude CLI (if installed and authenticated)
    2. Copilot CLI (if installed and authenticated)
    3. ANTHROPIC_API_KEY environment variable
    4. OPENAI_API_KEY environment variable
    5. GEMINI_API_KEY environment variable

    Returns:
        ProviderDetectionResult if a provider is found, None otherwise.
    """
    # 1. Check Claude CLI
    if shutil.which("claude"):
        if _check_claude_auth():
            return ProviderDetectionResult(
                provider="claude-agent",
                model="claude-agent/sonnet",
                authenticated=True,
                source="cli",
            )

    # 2. Check Copilot CLI
    if shutil.which("copilot"):
        if _check_copilot_auth():
            return ProviderDetectionResult(
                provider="copilot",
                model="copilot/gpt-4o",
                authenticated=True,
                source="cli",
            )

    # 3. Check environment variables
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ProviderDetectionResult(
            provider="anthropic",
            model="anthropic/claude-sonnet-4-20250514",
            authenticated=True,
            source="env",
        )

    if os.environ.get("OPENAI_API_KEY"):
        return ProviderDetectionResult(
            provider="openai",
            model="openai/gpt-4o",
            authenticated=True,
            source="env",
        )

    if os.environ.get("GEMINI_API_KEY"):
        return ProviderDetectionResult(
            provider="gemini",
            model="gemini/gemini-2.0-flash",
            authenticated=True,
            source="env",
        )

    return None


@dataclass
class InteractiveSession:
    """State for an interactive CLI session."""

    input_path: Path | None = None
    input_type: str = "file"  # "file", "directory", "url"
    output_dir: Path = field(default_factory=lambda: Path("./output"))
    enable_llm: bool = False
    enable_alt: bool = False
    enable_desc: bool = False
    enable_ocr: bool = False
    enable_screenshot: bool = False
    provider_result: ProviderDetectionResult | None = None
