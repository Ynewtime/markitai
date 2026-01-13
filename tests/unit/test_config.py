"""Tests for configuration module."""

from pathlib import Path

import pytest


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Run settings tests in an isolated directory without markit.yaml."""
    # Change to temp directory so no markit.yaml is found
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
        assert settings.log_dir == ".logs"
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


class TestYamlConfigLoading:
    """Tests for YAML configuration loading."""

    def test_yaml_config_loading(self, tmp_path, monkeypatch):
        """Test that settings are loaded from markit.yaml file."""
        # Create a test YAML config
        config_content = """
log_level: "DEBUG"
state_file: ".custom-state.json"

llm:
  credentials:
    - id: "test-cred"
      provider: "openai"
      api_key: "test-key"
  models:
    - name: "Test Model"
      model: "gpt-4"
      credential_id: "test-cred"

image:
  enable_compression: false
  max_dimension: 1024
"""
        config_file = tmp_path / "markit.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        # Change to temp directory
        monkeypatch.chdir(tmp_path)

        # Clear cache and reload
        from markit.config.settings import get_settings

        get_settings.cache_clear()

        settings = get_settings()

        assert settings.log_level == "DEBUG"
        assert settings.state_file == ".custom-state.json"
        assert settings.image.enable_compression is False
        assert settings.image.max_dimension == 1024
        assert len(settings.llm.credentials) == 1
        assert settings.llm.credentials[0].id == "test-cred"
        assert len(settings.llm.models) == 1
        assert settings.llm.models[0].name == "Test Model"

        # Clean up
        get_settings.cache_clear()

    def test_settings_uses_yaml_config_source(self):
        """Test that MarkitSettings uses YamlConfigSettingsSource."""
        from pydantic_settings import YamlConfigSettingsSource

        from markit.config.settings import MarkitSettings

        # Check that the settings_customise_sources method returns YamlConfigSettingsSource
        sources = MarkitSettings.settings_customise_sources(MarkitSettings, None, None, None, None)  # type: ignore[arg-type]

        # Find YamlConfigSettingsSource in sources
        yaml_source_found = False
        for source in sources:
            if source is not None and isinstance(source, YamlConfigSettingsSource):
                yaml_source_found = True
                break

        assert yaml_source_found, "YamlConfigSettingsSource should be in settings sources"


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

    def test_credential_config(self):
        """Test LLM credential configuration."""
        from markit.config.settings import LLMCredentialConfig

        cred = LLMCredentialConfig(
            id="openai-main",
            provider="openai",
            api_key="sk-test-key",
            base_url="https://custom.api.com",
        )

        assert cred.id == "openai-main"
        assert cred.provider == "openai"
        assert cred.api_key == "sk-test-key"
        assert cred.base_url == "https://custom.api.com"

    def test_credential_config_with_env_var(self):
        """Test LLM credential configuration with environment variable."""
        from markit.config.settings import LLMCredentialConfig

        cred = LLMCredentialConfig(
            id="anthropic-main",
            provider="anthropic",
            api_key_env="MY_ANTHROPIC_KEY",
        )

        assert cred.id == "anthropic-main"
        assert cred.provider == "anthropic"
        assert cred.api_key is None
        assert cred.api_key_env == "MY_ANTHROPIC_KEY"

    def test_model_config(self):
        """Test LLM model configuration."""
        from markit.config.settings import LLMModelConfig

        model = LLMModelConfig(
            name="GPT-4o",
            model="gpt-4o",
            credential_id="openai-main",
            capabilities=["text", "vision"],
            timeout=120,
        )

        assert model.name == "GPT-4o"
        assert model.model == "gpt-4o"
        assert model.credential_id == "openai-main"
        assert model.capabilities == ["text", "vision"]
        assert model.timeout == 120
        assert model.max_retries == 3  # default

    def test_model_config_defaults(self):
        """Test LLM model configuration defaults."""
        from markit.config.settings import LLMModelConfig

        model = LLMModelConfig(
            name="Test",
            model="test-model",
            credential_id="test-cred",
        )

        assert model.capabilities is None  # Defaults to None (optimistic)
        assert model.timeout == 300  # DEFAULT_MAX_REQUEST_TIMEOUT
        assert model.max_retries == 3

    def test_llm_config_with_credentials_and_models(self):
        """Test LLMConfig with new credentials and models schema."""
        from markit.config.settings import LLMConfig, LLMCredentialConfig, LLMModelConfig

        llm_config = LLMConfig(
            credentials=[
                LLMCredentialConfig(id="openai-1", provider="openai", api_key="sk-1"),
                LLMCredentialConfig(id="anthropic-1", provider="anthropic", api_key="sk-2"),
            ],
            models=[
                LLMModelConfig(
                    name="GPT-4o",
                    model="gpt-4o",
                    credential_id="openai-1",
                    capabilities=["text", "vision"],
                ),
                LLMModelConfig(
                    name="Claude Sonnet",
                    model="claude-sonnet-4",
                    credential_id="anthropic-1",
                    capabilities=["text", "vision"],
                ),
            ],
        )

        assert len(llm_config.credentials) == 2
        assert len(llm_config.models) == 2
        assert llm_config.credentials[0].id == "openai-1"
        assert llm_config.models[0].name == "GPT-4o"
        assert llm_config.models[1].credential_id == "anthropic-1"

    def test_llm_config_mixed_legacy_and_new(self):
        """Test LLMConfig supporting both legacy providers and new schema."""
        from markit.config.settings import (
            LLMConfig,
            LLMCredentialConfig,
            LLMModelConfig,
            LLMProviderConfig,
        )

        llm_config = LLMConfig(
            # Legacy provider
            providers=[
                LLMProviderConfig(provider="ollama", model="llama3.2-vision"),
            ],
            # New schema
            credentials=[
                LLMCredentialConfig(id="openai-1", provider="openai", api_key="sk-1"),
            ],
            models=[
                LLMModelConfig(name="GPT-4o", model="gpt-4o", credential_id="openai-1"),
            ],
        )

        assert len(llm_config.providers) == 1
        assert len(llm_config.credentials) == 1
        assert len(llm_config.models) == 1

    def test_provider_config_all_providers(self):
        """Test LLMProviderConfig accepts all supported provider types."""
        from markit.config.settings import LLMProviderConfig

        providers = ["openai", "anthropic", "gemini", "ollama", "openrouter"]
        for provider in providers:
            config = LLMProviderConfig(provider=provider, model="test-model")  # type: ignore[arg-type]
            assert config.provider == provider

    def test_credential_config_all_providers(self):
        """Test LLMCredentialConfig accepts all supported provider types."""
        from markit.config.settings import LLMCredentialConfig

        providers = ["openai", "anthropic", "gemini", "ollama", "openrouter"]
        for provider in providers:
            config = LLMCredentialConfig(id=f"{provider}-cred", provider=provider)  # type: ignore[arg-type]
            assert config.provider == provider


class TestConfigCommands:
    """Tests for config CLI commands."""

    def test_config_init_creates_file(self, tmp_path):
        """Test that config init creates a configuration file."""
        from typer.testing import CliRunner

        from markit.cli.commands.config import config_app

        runner = CliRunner()
        config_file = tmp_path / "markit.yaml"

        result = runner.invoke(config_app, ["init", "--path", str(config_file)])

        assert result.exit_code == 0
        assert config_file.exists()
        assert "Created config file at:" in result.stdout

    def test_config_init_contains_required_sections(self, tmp_path):
        """Test that generated config contains all required sections."""
        from typer.testing import CliRunner

        from markit.cli.commands.config import config_app

        runner = CliRunner()
        config_file = tmp_path / "markit.yaml"

        runner.invoke(config_app, ["init", "--path", str(config_file)])

        content = config_file.read_text(encoding="utf-8")

        # Check for global settings
        assert 'log_level: "INFO"' in content
        assert 'state_file: ".markit-state.json"' in content

        # Check for new schema documentation (commented example)
        assert "credentials:" in content
        assert "models:" in content
        assert "credential_id:" in content

        # Check for provider types in commented example
        assert 'provider: "openai"' in content
        assert 'provider: "anthropic"' in content
        # Supported providers listed in comment
        assert "openai, anthropic, gemini, ollama, openrouter" in content

        # Check for image section
        assert "image:" in content
        assert "enable_compression: true" in content
        assert "max_dimension: 2048" in content

        # Check for concurrency section
        assert "concurrency:" in content
        assert "file_workers: 4" in content
        assert "image_workers: 8" in content
        assert "llm_workers: 5" in content

        # Check for pdf section
        assert "pdf:" in content
        assert 'engine: "pymupdf4llm"' in content

        # Check for enhancement section
        assert "enhancement:" in content
        assert "chunk_size: 32000" in content

        # Check for output section
        assert "output:" in content
        assert 'default_dir: "output"' in content
        assert 'on_conflict: "rename"' in content

    def test_config_init_refuses_overwrite_without_force(self, tmp_path):
        """Test that config init refuses to overwrite existing file."""
        from typer.testing import CliRunner

        from markit.cli.commands.config import config_app

        runner = CliRunner()
        config_file = tmp_path / "markit.yaml"
        config_file.write_text("existing content", encoding="utf-8")

        result = runner.invoke(config_app, ["init", "--path", str(config_file)])

        assert result.exit_code == 1
        assert "already exists" in result.stdout
        assert config_file.read_text(encoding="utf-8") == "existing content"

    def test_config_init_force_overwrites(self, tmp_path):
        """Test that config init with --force overwrites existing file."""
        from typer.testing import CliRunner

        from markit.cli.commands.config import config_app

        runner = CliRunner()
        config_file = tmp_path / "markit.yaml"
        config_file.write_text("existing content", encoding="utf-8")

        result = runner.invoke(config_app, ["init", "--path", str(config_file), "--force"])

        assert result.exit_code == 0
        assert "Created config file at:" in result.stdout
        assert config_file.read_text(encoding="utf-8") != "existing content"
        assert "[image]" not in config_file.read_text(encoding="utf-8")  # It should be YAML now
        assert "image:" in config_file.read_text(encoding="utf-8")

    def test_default_config_template_matches_example(self, tmp_path):
        """Test that generated config aligns with markit.example.yaml structure."""
        from typer.testing import CliRunner

        from markit.cli.commands.config import DEFAULT_CONFIG_TEMPLATE, config_app

        runner = CliRunner()
        config_file = tmp_path / "markit.yaml"

        runner.invoke(config_app, ["init", "--path", str(config_file)])

        content = config_file.read_text(encoding="utf-8")

        # Verify content matches the template
        assert content == DEFAULT_CONFIG_TEMPLATE

        # Key structural elements
        assert "# MarkIt Configuration" in content
        assert "markit provider add" in content  # Instructions for adding providers


class TestPromptConfig:
    """Tests for PromptConfig.get_prompt method."""

    def test_get_prompt_from_custom_file(self, tmp_path):
        """Test get_prompt loads from custom file path."""
        from markit.config.settings import PromptConfig

        # Create custom prompt file
        custom_file = tmp_path / "my_prompt.md"
        custom_file.write_text("Custom image prompt", encoding="utf-8")

        # Use new *_prompt_file naming
        config = PromptConfig(image_analysis_prompt_file=str(custom_file))
        result = config.get_prompt("image_analysis")

        assert result == "Custom image prompt"

    def test_get_prompt_from_user_prompts_dir(self, tmp_path):
        """Test get_prompt loads from user prompts directory."""
        from markit.config.settings import PromptConfig

        # Create prompts dir with language-suffixed file
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "enhancement_zh.md").write_text("用户增强提示", encoding="utf-8")

        config = PromptConfig(prompts_dir=str(prompts_dir), output_language="zh")
        result = config.get_prompt("enhancement")

        assert result == "用户增强提示"

    def test_get_prompt_from_legacy_inline(self, tmp_path):
        """Test get_prompt uses legacy inline custom prompt."""
        from markit.config.settings import PromptConfig

        # Use empty prompts_dir so custom file doesn't exist
        config = PromptConfig(
            prompts_dir=str(tmp_path / "nonexistent"),
            custom_summary_prompt="Legacy summary prompt",
        )
        result = config.get_prompt("summary")

        assert result == "Legacy summary prompt"

    def test_get_prompt_from_package_builtin(self):
        """Test get_prompt falls back to package built-in."""
        from markit.config.settings import PromptConfig

        config = PromptConfig(output_language="en")
        result = config.get_prompt("image_analysis")

        assert result is not None
        assert len(result) > 0

    def test_get_prompt_auto_language_defaults_to_zh(self, tmp_path):
        """Test get_prompt with auto language defaults to zh."""
        from markit.config.settings import PromptConfig

        # Create zh prompt file
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "summary_zh.md").write_text("中文摘要提示", encoding="utf-8")

        config = PromptConfig(prompts_dir=str(prompts_dir), output_language="auto")
        result = config.get_prompt("summary")

        assert result == "中文摘要提示"

    def test_get_prompt_nonexistent_type_returns_none(self, tmp_path):
        """Test get_prompt returns None for nonexistent prompt type."""
        from markit.config.settings import PromptConfig

        config = PromptConfig(prompts_dir=str(tmp_path / "nonexistent"))
        result = config.get_prompt("nonexistent_type")

        assert result is None

    def test_get_prompt_custom_file_not_exists_fallback(self, tmp_path):
        """Test get_prompt falls back when custom file doesn't exist."""
        from markit.config.settings import PromptConfig

        # Use new *_prompt_file naming
        config = PromptConfig(
            image_analysis_prompt_file=str(tmp_path / "nonexistent.md"),
            output_language="en",
        )
        result = config.get_prompt("image_analysis")

        # Should fall back to package built-in
        assert result is not None


class TestLLMConfigResolver:
    """Tests for LLMConfigResolver class."""

    def test_resolve_no_cli_args(self):
        """Test resolve returns base config when no CLI args."""
        from markit.config.settings import LLMConfig, LLMConfigResolver

        base_config = LLMConfig()
        result = LLMConfigResolver.resolve(base_config)

        assert result is base_config

    def test_resolve_cli_provider_creates_new(self):
        """Test resolve creates new provider when CLI provider specified."""
        from markit.config.settings import LLMConfig, LLMConfigResolver

        base_config = LLMConfig()
        result = LLMConfigResolver.resolve(base_config, cli_provider="anthropic")

        assert len(result.providers) == 1
        assert result.providers[0].provider == "anthropic"

    def test_resolve_cli_provider_with_model(self):
        """Test resolve creates provider with custom model."""
        from markit.config.settings import LLMConfig, LLMConfigResolver

        base_config = LLMConfig()
        result = LLMConfigResolver.resolve(
            base_config, cli_provider="openai", cli_model="gpt-4o-mini"
        )

        assert len(result.providers) == 1
        assert result.providers[0].provider == "openai"
        assert result.providers[0].model == "gpt-4o-mini"

    def test_resolve_updates_existing_provider(self):
        """Test resolve updates existing provider's model."""
        from markit.config.settings import LLMConfig, LLMConfigResolver, LLMProviderConfig

        base_config = LLMConfig(
            providers=[
                LLMProviderConfig(provider="openai", model="gpt-4"),
                LLMProviderConfig(provider="anthropic", model="claude-sonnet-4"),
            ]
        )
        result = LLMConfigResolver.resolve(
            base_config, cli_provider="anthropic", cli_model="claude-opus-4"
        )

        # anthropic should be moved to front with new model
        assert len(result.providers) == 2
        assert result.providers[0].provider == "anthropic"
        assert result.providers[0].model == "claude-opus-4"

    def test_resolve_model_only_updates_first_provider(self):
        """Test resolve updates first provider's model when only model specified."""
        from markit.config.settings import LLMConfig, LLMConfigResolver, LLMProviderConfig

        base_config = LLMConfig(
            providers=[
                LLMProviderConfig(provider="openai", model="gpt-4"),
            ]
        )
        result = LLMConfigResolver.resolve(base_config, cli_model="gpt-4o")

        assert result.providers[0].model == "gpt-4o"

    def test_resolve_model_only_no_providers(self):
        """Test resolve with model only but no providers does nothing."""
        from markit.config.settings import LLMConfig, LLMConfigResolver

        base_config = LLMConfig()
        result = LLMConfigResolver.resolve(base_config, cli_model="gpt-4o")

        assert len(result.providers) == 0

    def test_resolve_uses_default_model(self):
        """Test resolve uses default model for provider."""
        from markit.config.settings import LLMConfig, LLMConfigResolver

        base_config = LLMConfig()
        result = LLMConfigResolver.resolve(base_config, cli_provider="gemini")

        assert result.providers[0].provider == "gemini"
        # Should use default model from constants
        assert result.providers[0].model != ""


class TestValidationConfig:
    """Tests for ValidationConfig class."""

    def test_default_values(self):
        """Test default validation config values."""
        from markit.config.settings import ValidationConfig

        config = ValidationConfig()

        assert config.enabled is True
        assert config.retry_count == 2
        assert config.on_failure == "warn"

    def test_custom_values(self):
        """Test custom validation config values."""
        from markit.config.settings import ValidationConfig

        config = ValidationConfig(
            enabled=False,
            retry_count=5,
            on_failure="fail",
        )

        assert config.enabled is False
        assert config.retry_count == 5
        assert config.on_failure == "fail"

    def test_on_failure_options(self):
        """Test all on_failure options."""
        from markit.config.settings import ValidationConfig

        for option in ["warn", "skip", "fail"]:
            config = ValidationConfig(on_failure=option)  # type: ignore[arg-type]
            assert config.on_failure == option


class TestModelCostConfig:
    """Tests for ModelCostConfig class."""

    def test_default_values(self):
        """Test default model cost values."""
        from markit.config.settings import ModelCostConfig

        config = ModelCostConfig()

        assert config.input_per_1m == 0.0
        assert config.output_per_1m == 0.0
        assert config.cached_input_per_1m is None

    def test_custom_values(self):
        """Test custom model cost values."""
        from markit.config.settings import ModelCostConfig

        config = ModelCostConfig(
            input_per_1m=2.5,
            output_per_1m=10.0,
            cached_input_per_1m=1.25,
        )

        assert config.input_per_1m == 2.5
        assert config.output_per_1m == 10.0
        assert config.cached_input_per_1m == 1.25


class TestExecutionConfig:
    """Tests for ExecutionConfig class."""

    def test_default_values(self):
        """Test default execution config values."""
        from markit.config.settings import ExecutionConfig

        config = ExecutionConfig()

        assert config.mode == "default"
        assert config.fast_max_fallback == 1
        assert config.fast_skip_validation is True

    def test_fast_mode(self):
        """Test fast mode configuration."""
        from markit.config.settings import ExecutionConfig

        config = ExecutionConfig(
            mode="fast",
            fast_max_fallback=3,
            fast_skip_validation=False,
        )

        assert config.mode == "fast"
        assert config.fast_max_fallback == 3
        assert config.fast_skip_validation is False


class TestLibreOfficeConfig:
    """Tests for LibreOfficeConfig class."""

    def test_default_values(self):
        """Test default LibreOffice config values."""
        from markit.config.settings import LibreOfficeConfig

        config = LibreOfficeConfig()

        assert config.pool_size == 8
        assert config.profile_base_dir == ".markit-lo-profiles"
        assert config.reset_after_failures == 3
        assert config.reset_after_uses == 100

    def test_custom_values(self):
        """Test custom LibreOffice config values."""
        from markit.config.settings import LibreOfficeConfig

        config = LibreOfficeConfig(
            pool_size=8,
            profile_base_dir="/custom/profiles",
            reset_after_failures=5,
            reset_after_uses=50,
        )

        assert config.pool_size == 8
        assert config.profile_base_dir == "/custom/profiles"
        assert config.reset_after_failures == 5
        assert config.reset_after_uses == 50

    def test_pool_size_constraints(self):
        """Test pool size constraints."""
        from pydantic import ValidationError

        from markit.config.settings import LibreOfficeConfig

        # Valid range
        config = LibreOfficeConfig(pool_size=32)
        assert config.pool_size == 32

        # Below minimum
        with pytest.raises(ValidationError):
            LibreOfficeConfig(pool_size=0)

        # Above maximum
        with pytest.raises(ValidationError):
            LibreOfficeConfig(pool_size=33)


class TestLLMConfigAdvanced:
    """Advanced tests for LLMConfig settings."""

    def test_concurrent_fallback_settings(self):
        """Test concurrent fallback settings."""
        from markit.config.settings import LLMConfig

        config = LLMConfig(
            concurrent_fallback_enabled=True,
            concurrent_fallback_timeout=10,
            max_request_timeout=300,
        )

        assert config.concurrent_fallback_enabled is True
        assert config.concurrent_fallback_timeout == 10
        assert config.max_request_timeout == 300

    def test_validation_settings(self):
        """Test validation settings in LLMConfig."""
        from markit.config.settings import LLMConfig, ValidationConfig

        config = LLMConfig(validation=ValidationConfig(enabled=False, on_failure="skip"))

        assert config.validation.enabled is False
        assert config.validation.on_failure == "skip"


class TestModelConfigWithCost:
    """Tests for LLMModelConfig with cost configuration."""

    def test_model_with_cost(self):
        """Test model config with cost information."""
        from markit.config.settings import LLMModelConfig, ModelCostConfig

        model = LLMModelConfig(
            name="GPT-4o",
            model="gpt-4o",
            credential_id="openai-1",
            cost=ModelCostConfig(
                input_per_1m=5.0,
                output_per_1m=15.0,
            ),
        )

        assert model.cost is not None
        assert model.cost.input_per_1m == 5.0
        assert model.cost.output_per_1m == 15.0

    def test_model_without_cost(self):
        """Test model config without cost is None."""
        from markit.config.settings import LLMModelConfig

        model = LLMModelConfig(
            name="Test",
            model="test",
            credential_id="test-cred",
        )

        assert model.cost is None


class TestConstants:
    """Tests for constants module."""

    def test_default_config_file_is_yaml(self):
        """Test that default config file uses YAML format."""
        from markit.config.constants import DEFAULT_CONFIG_FILE

        assert DEFAULT_CONFIG_FILE == "markit.yaml"
        assert DEFAULT_CONFIG_FILE.endswith(".yaml")

    def test_config_locations_use_yaml(self):
        """Test that config locations use YAML format."""
        from markit.config.constants import CONFIG_LOCATIONS

        for location in CONFIG_LOCATIONS:
            assert str(location).endswith(".yaml"), f"Config location {location} should use .yaml"

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
