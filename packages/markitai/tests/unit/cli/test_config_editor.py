"""Tests for interactive config editor schema introspection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
    from markitai.config import OutputConfig

    cfg = MarkitaiConfig(output=OutputConfig(dir="./output"))
    settings = extract_editable_settings(cfg)
    keys = {s["key"] for s in settings}
    assert "output.dir" in keys
    dir_setting = next(s for s in settings if s["key"] == "output.dir")
    assert dir_setting["field_type"] == "str"


class TestPromptNewValueCancel:
    """Pressing Esc/Ctrl-C in any sub-prompt should return _CANCEL."""

    def _make_setting(self, field_type: str, **kwargs):
        base = {
            "key": "test.key",
            "value": kwargs.get("value"),
            "field_type": field_type,
            "description": "",
        }
        base.update(kwargs)
        return base

    def _mock_ask_none(self):
        """Return a mock questionary prompt whose .ask() returns None (cancel)."""
        mock_prompt = MagicMock()
        mock_prompt.ask.return_value = None
        return mock_prompt

    @patch("questionary.text")
    def test_cancel_str_prompt(self, mock_text) -> None:
        from markitai.cli.config_editor import _CANCEL, _prompt_new_value

        mock_text.return_value = self._mock_ask_none()
        result = _prompt_new_value(self._make_setting("str", value="hello"))
        assert result is _CANCEL

    @patch("questionary.text")
    def test_cancel_int_prompt(self, mock_text) -> None:
        from markitai.cli.config_editor import _CANCEL, _prompt_new_value

        mock_text.return_value = self._mock_ask_none()
        result = _prompt_new_value(self._make_setting("int", value=42))
        assert result is _CANCEL

    @patch("questionary.text")
    def test_cancel_float_prompt(self, mock_text) -> None:
        from markitai.cli.config_editor import _CANCEL, _prompt_new_value

        mock_text.return_value = self._mock_ask_none()
        result = _prompt_new_value(self._make_setting("float", value=3.14))
        assert result is _CANCEL

    @patch("questionary.select")
    def test_cancel_bool_prompt(self, mock_select) -> None:
        from markitai.cli.config_editor import _CANCEL, _prompt_new_value

        mock_select.return_value = self._mock_ask_none()
        result = _prompt_new_value(self._make_setting("bool", value=True))
        assert result is _CANCEL

    @patch("questionary.select")
    def test_cancel_literal_prompt(self, mock_select) -> None:
        from markitai.cli.config_editor import _CANCEL, _prompt_new_value

        mock_select.return_value = self._mock_ask_none()
        result = _prompt_new_value(
            self._make_setting("literal", value="INFO", choices=["INFO", "DEBUG"])
        )
        assert result is _CANCEL

    @patch("questionary.text")
    def test_str_prompt_passes_key_bindings(self, mock_text) -> None:
        """text() prompts must receive key_bindings for Esc support."""
        mock_text.return_value = self._mock_ask_none()
        from markitai.cli.config_editor import _prompt_new_value

        _prompt_new_value(self._make_setting("str", value="test"))
        _, kwargs = mock_text.call_args
        assert "key_bindings" in kwargs

    @patch("questionary.select")
    def test_bool_uses_select_not_confirm(self, mock_select) -> None:
        """Bool fields should use select() (supports Esc), not confirm()."""
        mock_select.return_value = self._mock_ask_none()
        from markitai.cli.config_editor import _prompt_new_value

        _prompt_new_value(self._make_setting("bool", value=False))
        mock_select.assert_called_once()


def test_add_esc_to_question_injects_escape_binding() -> None:
    """_add_esc_to_question should merge Esc key binding into a select Question."""
    import sys

    import pytest

    if sys.platform == "win32":
        pytest.skip("questionary.select() requires a console on Windows")

    import questionary
    from prompt_toolkit.keys import Keys

    from markitai.cli.config_editor import _add_esc_to_question

    q = questionary.select("test", choices=["a", "b"], use_jk_keys=False)
    original_bindings = q.application.key_bindings
    assert original_bindings is not None

    # Esc should NOT be bound before injection
    esc_before = [b for b in original_bindings.bindings if b.keys == (Keys.Escape,)]
    assert len(esc_before) == 0

    _add_esc_to_question(q)

    # Esc SHOULD be bound after injection
    merged_bindings = q.application.key_bindings
    assert merged_bindings is not None
    esc_after = [b for b in merged_bindings.bindings if b.keys == (Keys.Escape,)]
    assert len(esc_after) == 1


class TestFuzzyMatch:
    """Tests for fuzzy_match pure function."""

    def test_exact_match(self) -> None:
        from markitai.cli.config_editor import fuzzy_match

        matched, _score = fuzzy_match("output", "output.dir ./output")
        assert matched is True

    def test_fuzzy_order_preserved(self) -> None:
        from markitai.cli.config_editor import fuzzy_match

        matched, _ = fuzzy_match("odir", "output.dir ./output")
        assert matched is True

    def test_no_match(self) -> None:
        from markitai.cli.config_editor import fuzzy_match

        matched, _ = fuzzy_match("zzz", "output.dir ./output")
        assert matched is False

    def test_case_insensitive(self) -> None:
        from markitai.cli.config_editor import fuzzy_match

        matched, _ = fuzzy_match("OUTPUT", "output.dir ./output")
        assert matched is True

    def test_empty_query_matches_all(self) -> None:
        from markitai.cli.config_editor import fuzzy_match

        matched, score = fuzzy_match("", "anything")
        assert matched is True
        assert score == 0

    def test_consecutive_chars_score_better(self) -> None:
        from markitai.cli.config_editor import fuzzy_match

        _, score_consecutive = fuzzy_match("out", "output.dir ./output")
        _, score_spread = fuzzy_match("odr", "output.dir ./output")
        assert score_consecutive < score_spread

    def test_query_longer_than_text(self) -> None:
        from markitai.cli.config_editor import fuzzy_match

        matched, _ = fuzzy_match("very long query", "short")
        assert matched is False


class TestEditorValidatesBeforeSave:
    """Regression: invalid values must never be persisted by the editor."""

    def _run_editor(self, monkeypatch, config_file, key, new_value) -> None:
        import markitai.cli.config_editor as ce

        monkeypatch.setenv("MARKITAI_CONFIG", str(config_file))

        selections = iter([key, None])
        monkeypatch.setattr(ce, "_select_setting", lambda *_args: next(selections))
        monkeypatch.setattr(ce, "_prompt_new_value", lambda *_args: new_value)
        monkeypatch.setattr(ce, "_get_cursor_row", lambda: 0)

        ce.run_config_editor()

    def test_invalid_value_not_saved(self, tmp_path, monkeypatch) -> None:
        """image.quality=500 must be rejected, not written to disk."""
        import json

        from markitai.config import ConfigManager

        config_file = tmp_path / "markitai.json"
        config_file.write_text(json.dumps({"image": {"quality": 80}}))

        self._run_editor(monkeypatch, config_file, "image.quality", 500)

        saved = json.loads(config_file.read_text())
        assert saved["image"]["quality"] == 80
        # The file must still load cleanly afterwards
        cfg = ConfigManager().load(config_path=config_file)
        assert cfg.image.quality == 80

    def test_valid_value_saved(self, tmp_path, monkeypatch) -> None:
        import json

        config_file = tmp_path / "markitai.json"
        config_file.write_text(json.dumps({"image": {"quality": 80}}))

        self._run_editor(monkeypatch, config_file, "image.quality", 60)

        saved = json.loads(config_file.read_text())
        assert saved["image"]["quality"] == 60
