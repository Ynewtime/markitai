"""E2E test configuration and fixtures.

These tests require external services or API keys.
Skip tests automatically when dependencies are not available.
"""

import os

import pytest


# Register e2e marker
def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring external services")
    config.addinivalue_line("markers", "ollama: tests requiring local Ollama service")
    config.addinivalue_line("markers", "openai: tests requiring OPENAI_API_KEY")
    config.addinivalue_line("markers", "anthropic: tests requiring ANTHROPIC_API_KEY")
    config.addinivalue_line("markers", "gemini: tests requiring GOOGLE_API_KEY")
    config.addinivalue_line("markers", "openrouter: tests requiring OPENROUTER_API_KEY")


# API key fixtures
@pytest.fixture
def openai_api_key() -> str | None:
    """Get OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY")


@pytest.fixture
def anthropic_api_key() -> str | None:
    """Get Anthropic API key from environment."""
    return os.environ.get("ANTHROPIC_API_KEY")


@pytest.fixture
def google_api_key() -> str | None:
    """Get Google API key from environment."""
    return os.environ.get("GOOGLE_API_KEY")


@pytest.fixture
def openrouter_api_key() -> str | None:
    """Get OpenRouter API key from environment."""
    return os.environ.get("OPENROUTER_API_KEY")


# Skip decorators for convenience
requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

requires_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)

requires_gemini = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set"
)

requires_openrouter = pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not set"
)
