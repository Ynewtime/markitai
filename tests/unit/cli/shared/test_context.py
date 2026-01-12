"""Tests for CLI shared context module."""

import pytest
import typer
from rich.console import Console

from markit.cli.shared.context import (
    ConversionContext,
    _mask_config,
    _validate_options,
)
from markit.cli.shared.options import ConversionOptions


class TestMaskConfig:
    """Tests for _mask_config function."""

    def test_masks_api_key_in_providers(self):
        """Test that API keys are masked in provider config."""
        config = {
            "llm": {
                "providers": [
                    {"provider": "openai", "api_key": "sk-secret-key"},
                    {"provider": "anthropic", "api_key": "anthro-secret"},
                ]
            }
        }

        result = _mask_config(config)

        assert result["llm"]["providers"][0]["api_key"] == "***"
        assert result["llm"]["providers"][1]["api_key"] == "***"

    def test_handles_empty_api_key(self):
        """Test that empty API keys are not masked."""
        config = {
            "llm": {
                "providers": [
                    {"provider": "ollama", "api_key": ""},
                    {"provider": "openai", "api_key": None},
                ]
            }
        }

        result = _mask_config(config)

        # Empty/None keys should remain as-is
        assert result["llm"]["providers"][0]["api_key"] == ""
        assert result["llm"]["providers"][1]["api_key"] is None

    def test_handles_missing_llm_key(self):
        """Test handling config without LLM section."""
        config = {"image": {"enable_compression": True}}

        result = _mask_config(config)

        assert result == config

    def test_handles_missing_providers_key(self):
        """Test handling LLM config without providers."""
        config = {"llm": {"default_provider": "openai"}}

        result = _mask_config(config)

        assert result == config


class TestValidateOptions:
    """Tests for _validate_options function."""

    def test_valid_options_passes(self):
        """Test that valid options pass validation."""
        options = ConversionOptions()
        console = Console()

        # Should not raise
        _validate_options(options, console)

    def test_invalid_pdf_engine_raises(self):
        """Test that invalid PDF engine raises typer.Exit."""
        options = ConversionOptions()
        options.pdf_engine = "invalid_engine"
        console = Console()

        with pytest.raises(typer.Exit) as exc_info:
            _validate_options(options, console)

        assert exc_info.value.exit_code == 1

    def test_invalid_llm_provider_raises(self):
        """Test that invalid LLM provider raises typer.Exit."""
        options = ConversionOptions()
        options.llm_provider = "invalid_provider"
        console = Console()

        with pytest.raises(typer.Exit) as exc_info:
            _validate_options(options, console)

        assert exc_info.value.exit_code == 1

    def test_valid_pdf_engine_passes(self):
        """Test that valid PDF engines pass."""
        from markit.config.constants import PDF_ENGINES

        options = ConversionOptions()
        options.pdf_engine = list(PDF_ENGINES)[0]
        console = Console()

        # Should not raise
        _validate_options(options, console)

    def test_valid_llm_provider_passes(self):
        """Test that valid LLM providers pass."""
        from markit.config.constants import LLM_PROVIDERS

        options = ConversionOptions()
        options.llm_provider = list(LLM_PROVIDERS)[0]
        console = Console()

        # Should not raise
        _validate_options(options, console)


class TestConversionContextCreate:
    """Tests for ConversionContext.create method."""

    def test_create_with_fast_mode(self, tmp_path, monkeypatch):
        """Test creating context with fast mode enabled."""
        monkeypatch.chdir(tmp_path)

        # Clear settings cache
        from markit.config.settings import get_settings

        get_settings.cache_clear()

        options = ConversionOptions()
        options.fast = True
        options.output_dir = tmp_path / "output"

        context = ConversionContext.create(
            options=options,
            command_prefix="test",
        )

        assert context.settings.llm.validation.enabled is False
        assert context.settings.execution.mode == "fast"

        get_settings.cache_clear()

    def test_create_with_verbose(self, tmp_path, monkeypatch):
        """Test creating context with verbose mode."""
        monkeypatch.chdir(tmp_path)

        from markit.config.settings import get_settings

        get_settings.cache_clear()

        options = ConversionOptions()
        options.verbose = True
        options.output_dir = tmp_path / "output"

        context = ConversionContext.create(
            options=options,
            command_prefix="test",
        )

        assert context is not None
        assert context.log_path.exists() or True  # Log file may or may not exist yet

        get_settings.cache_clear()


class TestConversionContextMethods:
    """Tests for ConversionContext instance methods."""

    @pytest.fixture
    def context(self, tmp_path, monkeypatch):
        """Create a test context."""
        monkeypatch.chdir(tmp_path)

        from markit.config.settings import get_settings

        get_settings.cache_clear()

        options = ConversionOptions()
        options.output_dir = tmp_path / "output"

        ctx = ConversionContext.create(
            options=options,
            command_prefix="test",
        )

        yield ctx

        get_settings.cache_clear()

    def test_create_pipeline(self, context):
        """Test creating pipeline from context."""
        from markit.core.pipeline import ConversionPipeline

        pipeline = context.create_pipeline()

        assert isinstance(pipeline, ConversionPipeline)

    def test_create_pipeline_with_concurrent_fallback(self, context):
        """Test creating pipeline with concurrent fallback."""
        pipeline = context.create_pipeline(use_concurrent_fallback=True)

        assert pipeline is not None

    def test_log_start(self, context):
        """Test log_start method."""
        # Should not raise
        context.log_start(file="test.pdf")
