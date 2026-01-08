"""Tests for configuration module."""

from pathlib import Path

import pytest


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Run settings tests in an isolated directory without markit.toml."""
    # Change to temp directory so no markit.toml is found
    monkeypatch.chdir(tmp_path)
    # Clear any cached settings
    from markit.config.settings import get_settings

    get_settings.cache_clear()
    yield tmp_path
    # Clean up cache after test
    get_settings.cache_clear()


class TestMarkitSettings:
    """Tests for MarkitSettings."""

    def test_default_settings(self, isolated_settings):  # noqa: ARG002
        """Test default settings values."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        assert settings.log_level == "INFO"
        assert settings.log_file is None
        assert settings.state_file == ".markit-state.json"

    def test_output_settings(self):
        """Test output configuration defaults."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        assert settings.output.default_dir == "output"
        assert settings.output.on_conflict == "rename"
        assert settings.output.create_assets_subdir is True

    def test_image_settings(self):
        """Test image configuration defaults."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        assert settings.image.enable_compression is True
        assert settings.image.enable_analysis is False
        assert settings.image.png_optimization_level == 2
        assert settings.image.jpeg_quality == 85
        assert settings.image.max_dimension == 2048

    def test_pdf_settings(self):
        """Test PDF configuration defaults."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        assert settings.pdf.engine == "pymupdf4llm"
        assert settings.pdf.extract_images is True
        assert settings.pdf.ocr_enabled is False

    def test_enhancement_settings(self):
        """Test enhancement configuration defaults."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        assert settings.enhancement.enabled is False
        assert settings.enhancement.add_frontmatter is True
        assert settings.enhancement.fix_heading_levels is True

    def test_concurrency_settings(self):
        """Test concurrency configuration defaults."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        assert settings.concurrency.file_workers == 4
        assert settings.concurrency.image_workers == 8
        assert settings.concurrency.llm_workers == 5

    def test_environment_variable_override(self, monkeypatch):
        """Test that environment variables override defaults."""
        from markit.config.settings import reload_settings

        monkeypatch.setenv("MARKIT_LOG_LEVEL", "DEBUG")

        settings = reload_settings()

        assert settings.log_level == "DEBUG"

    def test_get_output_dir(self):
        """Test get_output_dir method."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        # Default path
        output_dir = settings.get_output_dir()
        assert output_dir == Path("output")

        # With base path
        base = Path("/some/base")
        output_dir = settings.get_output_dir(base)
        assert output_dir == Path("/some/base/output")


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_empty_providers(self, isolated_settings):  # noqa: ARG002
        """Test empty providers list."""
        from markit.config.settings import MarkitSettings

        settings = MarkitSettings()

        assert settings.llm.providers == []
        assert settings.llm.default_provider is None

    def test_provider_config(self):
        """Test LLM provider configuration."""
        from markit.config.settings import LLMProviderConfig

        config = LLMProviderConfig(
            provider="openai",
            model="gpt-5.2",
            api_key="test-key",
            timeout=30,
        )

        assert config.provider == "openai"
        assert config.model == "gpt-5.2"
        assert config.api_key == "test-key"
        assert config.timeout == 30
        assert config.max_retries == 3  # default


class TestConstants:
    """Tests for constants module."""

    def test_supported_extensions(self):
        """Test supported extensions are defined."""
        from markit.config.constants import SUPPORTED_EXTENSIONS

        assert ".docx" in SUPPORTED_EXTENSIONS
        assert ".pptx" in SUPPORTED_EXTENSIONS
        assert ".xlsx" in SUPPORTED_EXTENSIONS
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".txt" in SUPPORTED_EXTENSIONS

    def test_markitdown_formats(self):
        """Test MarkItDown formats are defined."""
        from markit.config.constants import MARKITDOWN_FORMATS

        assert ".docx" in MARKITDOWN_FORMATS
        assert ".pptx" in MARKITDOWN_FORMATS
        assert ".xlsx" in MARKITDOWN_FORMATS
        assert ".pdf" in MARKITDOWN_FORMATS

    def test_legacy_formats(self):
        """Test legacy formats are defined."""
        from markit.config.constants import LEGACY_FORMATS

        assert ".doc" in LEGACY_FORMATS
        assert ".ppt" in LEGACY_FORMATS
        assert ".xls" in LEGACY_FORMATS

    def test_llm_providers(self):
        """Test LLM providers are defined."""
        from markit.config.constants import LLM_PROVIDERS

        assert "openai" in LLM_PROVIDERS
        assert "anthropic" in LLM_PROVIDERS
        assert "gemini" in LLM_PROVIDERS
        assert "ollama" in LLM_PROVIDERS
        assert "openrouter" in LLM_PROVIDERS
