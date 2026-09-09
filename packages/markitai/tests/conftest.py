"""Pytest configuration and fixtures."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from markitai.config import LLMConfig, PromptsConfig


# =============================================================================
# Path Fixtures
# =============================================================================


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the path to the fixtures directory."""
    return Path(__file__).parent / "fixtures"


# =============================================================================
# Sample Content Fixtures
# =============================================================================


@pytest.fixture
def sample_config_dict() -> dict:
    """Return sample configuration dictionary."""
    return {
        "output": {
            "dir": "./custom_output",
            "on_conflict": "overwrite",
        },
        "llm": {
            "enabled": True,
            "concurrency": 5,
        },
        "image": {
            "compress": True,
            "quality": 90,
        },
    }


@pytest.fixture(autouse=True)
def _isolate_global_cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep the default PersistentCache out of the real ``~/.markitai``.

    Many tests build ``LLMProcessor`` without ``cache_global_dir``; the
    default would open — and write test entries into — the user's real
    ``~/.markitai/cache.db``. Explicit ``global_dir`` arguments are
    unaffected.
    """
    monkeypatch.setattr(
        "markitai.llm.cache.DEFAULT_GLOBAL_CACHE_DIR",
        str(tmp_path / "global-cache"),
    )


@pytest.fixture(autouse=True)
def _isolate_user_config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point ``Path.home()`` at a temp dir so no test reads the developer's
    real ``~/.markitai``.

    ``cli/main.py`` and ``serve/app.py`` load ``~/.markitai/.env`` (into
    ``os.environ``) at import/startup and ``ConfigManager`` reads
    ``~/.markitai/config.json``. A developer with real LLM keys and a model
    config configured would otherwise pollute provider-detection,
    capabilities, and doctor tests that assume a clean environment — the
    app re-loads the keys after a test deletes them, so deleting env vars
    alone is not enough. Redirecting home makes the whole suite hermetic.

    Tests that intentionally probe config resolution set ``MARKITAI_CONFIG``
    or a config path themselves and are unaffected (their own monkeypatch
    runs after this fixture).
    """
    fake_home = tmp_path / "fake-home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.setenv("HOME", str(fake_home))  # expanduser("~") follows too
    monkeypatch.setenv("USERPROFILE", str(fake_home))  # ...on Windows, via this one
    # DEFAULT_USER_CONFIG_DIR is computed at import time from Path.home();
    # re-point it so ConfigManager/auth/init never see the developer's real
    # ~/.markitai/config.json either.
    monkeypatch.setattr(
        "markitai.config.ConfigManager.DEFAULT_USER_CONFIG_DIR",
        fake_home / ".markitai",
    )
    # Scrub any keys the module-level load_dotenv already placed in os.environ.
    # The provider map is the source of truth; the rest are read directly by
    # provider code that is not in it.
    from markitai.constants import PROVIDER_API_KEY_ENV

    for key in {
        *PROVIDER_API_KEY_ENV.values(),
        "MISTRAL_API_KEY",
        "CLOUDFLARE_API_TOKEN",
    }:
        monkeypatch.delenv(key, raising=False)


# =============================================================================
# Fetch Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _reset_remote_fetch_consent(monkeypatch: pytest.MonkeyPatch):
    """Isolate remote consent and keep unit tests independent of live DNS."""
    from markitai import fetch, fetch_policy

    async def resolve_public_test_host(_hostname: str) -> tuple[str, ...]:
        return ("93.184.216.34",)

    monkeypatch.delenv("MARKITAI_NO_REMOTE_FETCH", raising=False)
    monkeypatch.setattr(
        fetch_policy,
        "resolve_hostname_addresses",
        resolve_public_test_host,
    )
    fetch.reset_remote_consent()
    fetch.reset_explicit_fallback_decision()
    yield
    fetch.reset_remote_consent()
    fetch.reset_explicit_fallback_decision()


# =============================================================================
# CLI Fixtures
# =============================================================================


@pytest.fixture
def cli_runner():
    """Return a CLI test runner."""
    from click.testing import CliRunner

    return CliRunner()


# =============================================================================
# LLM Configuration Fixtures
# =============================================================================


@pytest.fixture
def llm_config() -> "LLMConfig":
    """Return a test LLM configuration."""
    from markitai.config import LiteLLMParams, LLMConfig, ModelConfig

    return LLMConfig(
        enabled=True,
        model_list=[
            ModelConfig(
                model_name="default",
                litellm_params=LiteLLMParams(
                    model="openai/gpt-4o-mini",
                    api_key="test-key",
                ),
            ),
        ],
        concurrency=2,
    )


@pytest.fixture
def prompts_config() -> "PromptsConfig":
    """Return a test prompts configuration."""
    from markitai.config import PromptsConfig

    return PromptsConfig()


# =============================================================================
# Test File Fixtures
# =============================================================================


@pytest.fixture
def sample_txt_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    txt_file = tmp_path / "sample.txt"
    txt_file.write_text("# Test Document\n\nThis is test content.", encoding="utf-8")
    return txt_file


@pytest.fixture
def sample_md_file(tmp_path: Path) -> Path:
    """Create a sample markdown file for testing."""
    md_file = tmp_path / "sample.md"
    md_file.write_text(
        "# Test Document\n\nThis is test content.\n\n## Section 1\n\nSome text here.\n",
        encoding="utf-8",
    )
    return md_file


# =============================================================================
# Image Test Utilities
# =============================================================================


@pytest.fixture
def create_test_image():
    """Factory fixture for creating test images.

    Usage:
        def test_something(create_test_image):
            png_bytes = create_test_image(100, 100, "red")
    """
    import io

    from PIL import Image

    def _create(width: int = 100, height: int = 100, color: str = "red") -> bytes:
        img = Image.new("RGB", (width, height), color)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

    return _create


@pytest.fixture
def sample_png_bytes() -> bytes:
    """Return minimal valid PNG bytes for testing."""
    # Minimal 1x1 red PNG
    return bytes(
        [
            0x89,
            0x50,
            0x4E,
            0x47,
            0x0D,
            0x0A,
            0x1A,
            0x0A,  # PNG signature
            0x00,
            0x00,
            0x00,
            0x0D,
            0x49,
            0x48,
            0x44,
            0x52,  # IHDR chunk
            0x00,
            0x00,
            0x00,
            0x01,
            0x00,
            0x00,
            0x00,
            0x01,  # 1x1 pixel
            0x08,
            0x02,
            0x00,
            0x00,
            0x00,
            0x90,
            0x77,
            0x53,
            0xDE,
            0x00,
            0x00,
            0x00,
            0x0C,
            0x49,
            0x44,
            0x41,
            0x54,
            0x08,
            0xD7,
            0x63,
            0xF8,
            0xCF,
            0xC0,
            0x00,
            0x00,
            0x00,
            0x03,
            0x00,
            0x01,
            0x00,
            0x05,
            0xFE,
            0xD4,
            0xEF,
            0x00,
            0x00,
            0x00,
            0x00,
            0x49,
            0x45,
            0x4E,
            0x44,
            0xAE,
            0x42,
            0x60,
            0x82,
        ]
    )
