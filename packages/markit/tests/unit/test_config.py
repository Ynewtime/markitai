"""Tests for configuration module."""

import json
from pathlib import Path

import pytest

from markit.config import (
    BUILTIN_PRESETS,
    ConfigManager,
    EnvVarNotFoundError,
    MarkitConfig,
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


class TestMarkitConfig:
    """Tests for MarkitConfig model."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        config = MarkitConfig()

        assert config.output.dir == "./output"
        assert config.output.on_conflict == "rename"
        assert config.output.allow_symlinks is False
        assert config.llm.enabled is False
        assert config.llm.concurrency == 10
        assert config.image.compress is True
        assert config.batch.scan_max_depth == 5
        assert config.batch.scan_max_files == 10000
        assert config.image.quality == 85
        assert config.image.format == "jpeg"
        assert config.ocr.enabled is False
        assert config.log.level == "DEBUG"

    def test_custom_values(self, sample_config_dict: dict) -> None:
        """Test configuration with custom values."""
        config = MarkitConfig.model_validate(sample_config_dict)

        assert config.output.dir == "./custom_output"
        assert config.output.on_conflict == "overwrite"
        assert config.llm.enabled is True
        assert config.llm.concurrency == 5
        assert config.image.quality == 90

    def test_image_quality_validation(self) -> None:
        """Test image quality validation bounds."""
        # Valid quality
        config = MarkitConfig.model_validate({"image": {"quality": 50}})
        assert config.image.quality == 50

        # Invalid quality (too low)
        with pytest.raises(ValueError):
            MarkitConfig.model_validate({"image": {"quality": 0}})

        # Invalid quality (too high)
        with pytest.raises(ValueError):
            MarkitConfig.model_validate({"image": {"quality": 101}})


class TestConfigManager:
    """Tests for ConfigManager class."""

    def test_load_default_config(self) -> None:
        """Test loading default configuration."""
        manager = ConfigManager()
        config = manager.load()

        assert isinstance(config, MarkitConfig)
        assert config.output.dir == "./output"

    def test_load_from_file(self, tmp_path: Path, sample_config_dict: dict) -> None:
        """Test loading configuration from file."""
        config_file = tmp_path / "markit.json"
        config_file.write_text(json.dumps(sample_config_dict))

        manager = ConfigManager()
        config = manager.load(config_path=config_file)

        assert config.output.dir == "./custom_output"
        assert config.llm.enabled is True

    def test_load_from_env_var(
        self, tmp_path: Path, sample_config_dict: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading configuration from MARKIT_CONFIG env var."""
        config_file = tmp_path / "env_config.json"
        config_file.write_text(json.dumps(sample_config_dict))

        monkeypatch.setenv("MARKIT_CONFIG", str(config_file))

        manager = ConfigManager()
        config = manager.load()

        assert config.output.dir == "./custom_output"

    def test_get_nested_value(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test getting nested configuration values."""
        # Change to tmp_path to avoid loading project's markit.json
        monkeypatch.chdir(tmp_path)

        manager = ConfigManager()
        manager.load()

        assert manager.get("llm.enabled") is False
        assert manager.get("image.quality") == 85  # Default value
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
        config = MarkitConfig(presets={"custom": custom_preset})

        preset = get_preset("custom", config)
        assert preset is not None
        assert preset.llm is True
        assert preset.ocr is True

    def test_custom_preset_overrides_builtin(self) -> None:
        """Test custom preset with same name overrides built-in."""
        # Create a custom 'rich' preset with different values
        custom_rich = PresetConfig(llm=False, alt=False)
        config = MarkitConfig(presets={"rich": custom_rich})

        preset = get_preset("rich", config)
        assert preset is not None
        assert preset.llm is False  # Overridden
        assert preset.alt is False  # Overridden

    def test_fallback_to_builtin_when_not_in_config(self) -> None:
        """Test fallback to built-in when preset not in config."""
        config = MarkitConfig(presets={"custom": PresetConfig()})

        # 'rich' not in config.presets, should fall back to built-in
        preset = get_preset("rich", config)
        assert preset is not None
        assert preset.llm is True  # Built-in value
