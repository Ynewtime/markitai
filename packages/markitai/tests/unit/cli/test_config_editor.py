"""Tests for interactive config editor schema introspection."""

from __future__ import annotations

from markitai.config import MarkitaiConfig


def test_extract_settings_returns_flat_list() -> None:
    from markitai.cli.config_editor import extract_editable_settings

    cfg = MarkitaiConfig()
    settings = extract_editable_settings(cfg)
    assert len(settings) > 0
    for s in settings:
        assert "key" in s
        assert "value" in s
        assert "field_type" in s
        assert "description" in s


def test_extract_settings_includes_known_keys() -> None:
    from markitai.cli.config_editor import extract_editable_settings

    cfg = MarkitaiConfig()
    settings = extract_editable_settings(cfg)
    keys = {s["key"] for s in settings}
    assert "output.dir" in keys
    assert "llm.enabled" in keys
    assert "image.quality" in keys
    assert "log.level" in keys
    assert "cache.enabled" in keys


def test_extract_settings_literal_has_choices() -> None:
    from markitai.cli.config_editor import extract_editable_settings

    cfg = MarkitaiConfig()
    settings = extract_editable_settings(cfg)
    level_setting = next(s for s in settings if s["key"] == "log.level")
    assert level_setting["field_type"] == "literal"
    assert "choices" in level_setting
    assert "INFO" in level_setting["choices"]
    assert "DEBUG" in level_setting["choices"]


def test_extract_settings_skips_complex_types() -> None:
    from markitai.cli.config_editor import extract_editable_settings

    cfg = MarkitaiConfig()
    settings = extract_editable_settings(cfg)
    keys = {s["key"] for s in settings}
    assert "llm.model_list" not in keys
    assert "presets" not in keys


def test_format_value_for_display() -> None:
    from markitai.cli.config_editor import format_display_value

    assert format_display_value(True) == "true"
    assert format_display_value(False) == "false"
    assert format_display_value(None) == "—"
    assert format_display_value(85) == "85"
    assert format_display_value("./output") == "./output"


def test_build_choices_returns_questionary_choices() -> None:
    """build_choices should return a list of questionary Choice objects."""
    from markitai.cli.config_editor import build_choices, extract_editable_settings

    cfg = MarkitaiConfig()
    settings = extract_editable_settings(cfg)
    choices = build_choices(settings)
    assert len(choices) > 0
    for c in choices:
        assert hasattr(c, "title")
        assert hasattr(c, "value")


def test_extract_settings_bool_before_int() -> None:
    """Bool fields must be detected as 'bool', not 'int' (bool is subclass of int)."""
    from markitai.cli.config_editor import extract_editable_settings

    cfg = MarkitaiConfig()
    settings = extract_editable_settings(cfg)
    enabled = next(s for s in settings if s["key"] == "llm.enabled")
    assert enabled["field_type"] == "bool"
    assert enabled["value"] is False


def test_extract_settings_optional_str() -> None:
    """Optional[str] fields should still be editable as 'str'."""
    from markitai.cli.config_editor import extract_editable_settings

    cfg = MarkitaiConfig(output={"dir": "./output"})
    settings = extract_editable_settings(cfg)
    keys = {s["key"] for s in settings}
    assert "output.dir" in keys
    dir_setting = next(s for s in settings if s["key"] == "output.dir")
    assert dir_setting["field_type"] == "str"
