"""Pytest configuration and fixtures."""

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

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


@pytest.fixture
def tmp_output(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    output = tmp_path / "output"
    output.mkdir()
    return output


# =============================================================================
# Sample Content Fixtures
# =============================================================================


@pytest.fixture
def sample_markdown() -> str:
    """Return sample markdown content for testing."""
    return """# Test Document

This is a test document with some content.

## Section 1

Some text in section 1.

- Item 1
- Item 2
- Item 3

## Section 2

A table:

| Column 1 | Column 2 |
|----------|----------|
| Value 1  | Value 2  |
"""


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


# =============================================================================
# Mock Helpers
# =============================================================================


@pytest.fixture
def mock_llm_response():
    """Factory fixture for creating mock LLM responses.

    Usage:
        def test_something(mock_llm_response):
            response = mock_llm_response(content="Hello", model="gpt-4")
    """

    def _create(
        content: str = "Response",
        model: str = "test-model",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
    ) -> MagicMock:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=content))]
        mock_response.model = model
        mock_response.usage = MagicMock(
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
        )
        return mock_response

    return _create
