"""Tests for configuration module."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from markitai.config import (
    BUILTIN_PRESETS,
    ConfigManager,
    DomainProfileConfig,
    EnvVarNotFoundError,
    FetchPolicyConfig,
    LiteLLMParams,
    MarkitaiConfig,
    PresetConfig,
    get_preset,
    resolve_env_value,
)


class TestResolveEnvValue:
    """Tests for resolve_env_value function."""

    def test_resolve_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test resolving env: syntax."""
        monkeypatch.setenv("TEST_API_KEY", "test-value-123")
        result = resolve_env_value("env:TEST_API_KEY")
        assert result == "test-value-123"

    def test_resolve_missing_env_var_strict(self) -> None:
        """Test resolving missing environment variable in strict mode (default)."""
        with pytest.raises(EnvVarNotFoundError) as exc_info:
            resolve_env_value("env:NONEXISTENT_VAR_12345")
        assert exc_info.value.var_name == "NONEXISTENT_VAR_12345"

    def test_resolve_missing_env_var_non_strict(self) -> None:
        """Test resolving missing environment variable in non-strict mode."""
        result = resolve_env_value("env:NONEXISTENT_VAR_12345", strict=False)
        assert result is None

    def test_resolve_plain_value(self) -> None:
        """Test that plain values are returned unchanged."""
        result = resolve_env_value("plain-value")
        assert result == "plain-value"


class TestLiteLLMParamsResolvedApiBase:
    """Tests for LiteLLMParams.get_resolved_api_base()."""

    def test_api_base_none(self) -> None:
        """Test returns None when api_base is not configured."""
        params = LiteLLMParams(model="openai/gpt-4o")
        assert params.get_resolved_api_base() is None

    def test_api_base_plain_url(self) -> None:
        """Test plain URL is returned unchanged."""
        params = LiteLLMParams(
            model="openai/gpt-4o", api_base="https://api.example.com/v1"
        )
        assert params.get_resolved_api_base() == "https://api.example.com/v1"

    def test_api_base_env_syntax(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test env:VAR_NAME syntax resolves to environment variable."""
        monkeypatch.setenv("TEST_API_BASE", "https://my-proxy.example.com/v1")
        params = LiteLLMParams(model="openai/gpt-4o", api_base="env:TEST_API_BASE")
        assert params.get_resolved_api_base() == "https://my-proxy.example.com/v1"

    def test_api_base_env_missing_strict(self) -> None:
        """Test raises EnvVarNotFoundError when env var missing in strict mode."""
        params = LiteLLMParams(
            model="openai/gpt-4o", api_base="env:NONEXISTENT_BASE_URL_12345"
        )
        with pytest.raises(EnvVarNotFoundError) as exc_info:
            params.get_resolved_api_base(strict=True)
        assert exc_info.value.var_name == "NONEXISTENT_BASE_URL_12345"

    def test_api_base_env_missing_non_strict(self) -> None:
        """Test returns None when env var missing in non-strict mode."""
        params = LiteLLMParams(
            model="openai/gpt-4o", api_base="env:NONEXISTENT_BASE_URL_12345"
        )
        result = params.get_resolved_api_base(strict=False)
        assert result is None


class TestMarkitaiConfig:
    """Tests for MarkitaiConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MarkitaiConfig()

        assert config.output.dir == "./output"
        assert config.output.on_conflict == "rename"
        assert config.output.allow_symlinks is False
        assert config.llm.enabled is False
        assert config.llm.concurrency == 10
        assert config.image.compress is True
        assert config.batch.scan_max_depth == 5
        assert config.batch.scan_max_files == 10000
        assert config.image.quality == 75
        assert config.image.format == "jpeg"
        assert config.ocr.enabled is False
        assert config.log.level == "INFO"
        # Cloudflare config defaults
        assert config.fetch.cloudflare.api_token is None
        assert config.fetch.cloudflare.account_id is None
        assert config.fetch.cloudflare.convert_enabled is False

    def test_custom_values(self, sample_config_dict: dict) -> None:
        """Test configuration with custom values."""
        config = MarkitaiConfig.model_validate(sample_config_dict)

        assert config.output.dir == "./custom_output"
        assert config.output.on_conflict == "overwrite"
        assert config.llm.enabled is True
        assert config.llm.concurrency == 5
        assert config.image.quality == 90

    def test_image_quality_validation(self) -> None:
        """Test image quality validation bounds."""
        # Valid quality
        config = MarkitaiConfig.model_validate({"image": {"quality": 50}})
        assert config.image.quality == 50

        # Invalid quality (too low)
        with pytest.raises(ValueError):
            MarkitaiConfig.model_validate({"image": {"quality": 0}})

        # Invalid quality (too high)
        with pytest.raises(ValueError):
            MarkitaiConfig.model_validate({"image": {"quality": 101}})


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_load_default_config(self) -> None:
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.load()

        assert isinstance(config, MarkitaiConfig)
        assert config.output.dir == "./output"

    def test_load_from_file(self, tmp_path: Path, sample_config_dict: dict) -> None:
        """Test loading configuration from file."""
        config_file = tmp_path / "markitai.json"
        config_file.write_text(json.dumps(sample_config_dict))

        manager = ConfigManager()
        config = manager.load(config_path=config_file)

        assert config.output.dir == "./custom_output"
        assert config.llm.enabled is True

    def test_load_from_env_var(
        self, tmp_path: Path, sample_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading configuration from MARKITAI_CONFIG env var."""
        config_file = tmp_path / "env_config.json"
        config_file.write_text(json.dumps(sample_config_dict))

        monkeypatch.setenv("MARKITAI_CONFIG", str(config_file))

        manager = ConfigManager()
        config = manager.load()

        assert config.output.dir == "./custom_output"

    def test_get_nested_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test getting nested configuration values."""
        # Change to tmp_path to avoid loading project's markitai.json
        monkeypatch.chdir(tmp_path)

        manager = ConfigManager()
        manager.load()

        assert manager.get("llm.enabled") is False
        assert manager.get("image.quality") == 75  # Default value
        assert manager.get("nonexistent.key", "default") == "default"

    def test_set_nested_value(self) -> None:
        """Test setting nested configuration values."""
        manager = ConfigManager()
        manager.load()

        manager.set("llm.enabled", True)
        assert manager.config.llm.enabled is True

        manager.set("image.quality", 75)
        assert manager.config.image.quality == 75

    def test_save_config(self, tmp_path: Path) -> None:
        """Test saving configuration to file."""
        manager = ConfigManager()
        manager.load()
        manager.set("llm.enabled", True)

        save_path = tmp_path / "saved_config.json"
        manager.save(save_path)

        # Load and verify
        with open(save_path) as f:
            saved_data = json.load(f)

        assert saved_data["llm"]["enabled"] is True


class TestPresetConfig:
    """Tests for PresetConfig and preset system."""

    def test_preset_config_defaults(self) -> None:
        """Test PresetConfig default values."""
        preset = PresetConfig()
        assert preset.llm is False
        assert preset.ocr is False
        assert preset.alt is False
        assert preset.desc is False
        assert preset.screenshot is False

    def test_preset_config_custom(self) -> None:
        """Test PresetConfig with custom values."""
        preset = PresetConfig(llm=True, alt=True, desc=True)
        assert preset.llm is True
        assert preset.alt is True
        assert preset.desc is True
        assert preset.ocr is False
        assert preset.screenshot is False


class TestBuiltinPresets:
    """Tests for built-in presets."""

    def test_rich_preset_definition(self) -> None:
        """Test rich preset enables all features."""
        preset = BUILTIN_PRESETS["rich"]
        assert preset.llm is True
        assert preset.alt is True
        assert preset.desc is True
        assert preset.screenshot is True

    def test_standard_preset_definition(self) -> None:
        """Test standard preset enables LLM features without screenshot."""
        preset = BUILTIN_PRESETS["standard"]
        assert preset.llm is True
        assert preset.alt is True
        assert preset.desc is True
        assert preset.screenshot is False

    def test_minimal_preset_definition(self) -> None:
        """Test minimal preset disables all features."""
        preset = BUILTIN_PRESETS["minimal"]
        assert preset.llm is False
        assert preset.alt is False
        assert preset.desc is False
        assert preset.screenshot is False

    def test_all_builtin_presets_exist(self) -> None:
        """Test all expected built-in presets exist."""
        expected = {"rich", "standard", "minimal"}
        assert set(BUILTIN_PRESETS.keys()) == expected


class TestGetPreset:
    """Tests for get_preset function."""

    def test_get_builtin_preset(self) -> None:
        """Test getting built-in preset."""
        preset = get_preset("rich")
        assert preset is not None
        assert preset.llm is True
        assert preset.screenshot is True

    def test_get_nonexistent_preset(self) -> None:
        """Test getting nonexistent preset returns None."""
        preset = get_preset("nonexistent")
        assert preset is None

    def test_get_custom_preset_from_config(self) -> None:
        """Test getting custom preset from config."""
        custom_preset = PresetConfig(llm=True, ocr=True)
        config = MarkitaiConfig(presets={"custom": custom_preset})

        preset = get_preset("custom", config)
        assert preset is not None
        assert preset.llm is True
        assert preset.ocr is True

    def test_custom_preset_overrides_builtin(self) -> None:
        """Test custom preset with same name overrides built-in."""
        # Create a custom 'rich' preset with different values
        custom_rich = PresetConfig(llm=False, alt=False)
        config = MarkitaiConfig(presets={"rich": custom_rich})

        preset = get_preset("rich", config)
        assert preset is not None
        assert preset.llm is False  # Overridden
        assert preset.alt is False  # Overridden

    def test_fallback_to_builtin_when_not_in_config(self) -> None:
        """Test fallback to built-in when preset not in config."""
        config = MarkitaiConfig(presets={"custom": PresetConfig()})

        # 'rich' not in config.presets, should fall back to built-in
        preset = get_preset("rich", config)
        assert preset is not None
        assert preset.llm is True  # Built-in value


def test_fetch_policy_defaults_are_user_friendly() -> None:
    from markitai.config import MarkitaiConfig

    cfg = MarkitaiConfig()
    assert cfg.fetch.policy.enabled is True
    assert cfg.fetch.policy.max_strategy_hops == 5
    assert cfg.fetch.playwright.session_mode == "isolated"
    assert cfg.fetch.playwright.session_ttl_seconds == 600


def test_fetch_config_accepts_domain_profile_overrides() -> None:
    from markitai.config import MarkitaiConfig

    cfg = MarkitaiConfig.model_validate(
        {
            "fetch": {
                "domain_profiles": {
                    "x.com": {
                        "wait_for_selector": '[data-testid="tweetText"]',
                        "wait_for": "domcontentloaded",
                        "extra_wait_ms": 1200,
                    }
                }
            }
        }
    )

    assert (
        cfg.fetch.domain_profiles["x.com"].wait_for_selector
        == '[data-testid="tweetText"]'
    )


class TestFetchPolicyConfigValidation:
    """Tests for FetchPolicyConfig new fields."""

    def test_defaults(self) -> None:
        cfg = FetchPolicyConfig()
        assert cfg.strategy_priority is None
        assert cfg.local_only_patterns == []
        assert cfg.inherit_no_proxy is True

    def test_valid_strategy_priority(self) -> None:
        cfg = FetchPolicyConfig(strategy_priority=["static", "playwright"])
        assert cfg.strategy_priority == ["static", "playwright"]

    def test_invalid_strategy_name(self) -> None:
        with pytest.raises(ValidationError, match="invalid_strategy"):
            FetchPolicyConfig(strategy_priority=["static", "invalid"])

    def test_duplicate_strategies(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            FetchPolicyConfig(strategy_priority=["static", "static"])

    def test_empty_strategy_priority_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            FetchPolicyConfig(strategy_priority=[])

    def test_valid_local_only_patterns(self) -> None:
        cfg = FetchPolicyConfig(
            local_only_patterns=[".corp.com", "10.0.0.0/8", "localhost"]
        )
        assert len(cfg.local_only_patterns) == 3

    def test_empty_pattern_rejected(self) -> None:
        with pytest.raises(ValidationError, match="empty"):
            FetchPolicyConfig(local_only_patterns=[""])

    def test_invalid_cidr_rejected(self) -> None:
        with pytest.raises(ValidationError, match="CIDR"):
            FetchPolicyConfig(local_only_patterns=["999.0.0.0/8"])


class TestDomainProfileStrategyPriority:
    """Tests for DomainProfileConfig.strategy_priority."""

    def test_default_none(self) -> None:
        cfg = DomainProfileConfig()
        assert cfg.strategy_priority is None

    def test_valid_priority(self) -> None:
        cfg = DomainProfileConfig(strategy_priority=["static"])
        assert cfg.strategy_priority == ["static"]

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValidationError, match="invalid_strategy"):
            DomainProfileConfig(strategy_priority=["bogus"])

    def test_duplicate_rejected(self) -> None:
        with pytest.raises(ValidationError, match="duplicate"):
            DomainProfileConfig(strategy_priority=["static", "static"])
