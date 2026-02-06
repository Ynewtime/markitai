"""Tests for i18n module."""

from __future__ import annotations

from unittest.mock import patch

from markitai.cli import i18n


class TestDetectLanguage:
    """Tests for language detection."""

    def setup_method(self) -> None:
        """Reset language cache before each test."""
        i18n._lang = None

    def test_detect_language_from_markitai_lang_zh(self) -> None:
        """Should detect Chinese from MARKITAI_LANG."""
        with patch.dict("os.environ", {"MARKITAI_LANG": "zh_CN"}, clear=True):
            assert i18n.detect_language() == "zh"

    def test_detect_language_from_markitai_lang_en(self) -> None:
        """Should detect English from MARKITAI_LANG."""
        with patch.dict("os.environ", {"MARKITAI_LANG": "en_US"}, clear=True):
            assert i18n.detect_language() == "en"

    def test_detect_language_from_lang(self) -> None:
        """Should detect language from LANG when MARKITAI_LANG not set."""
        with patch.dict("os.environ", {"LANG": "zh_CN.UTF-8"}, clear=True):
            assert i18n.detect_language() == "zh"

    def test_detect_language_from_lc_all(self) -> None:
        """Should detect language from LC_ALL when LANG not set."""
        with patch.dict("os.environ", {"LC_ALL": "zh_TW.UTF-8"}, clear=True):
            assert i18n.detect_language() == "zh"

    def test_markitai_lang_overrides_lang(self) -> None:
        """MARKITAI_LANG should override LANG."""
        with patch.dict(
            "os.environ",
            {"MARKITAI_LANG": "en", "LANG": "zh_CN.UTF-8"},
            clear=True,
        ):
            assert i18n.detect_language() == "en"

    def test_default_to_english(self) -> None:
        """Should default to English when no language set."""
        with patch.dict("os.environ", {}, clear=True):
            assert i18n.detect_language() == "en"

    def test_case_insensitive(self) -> None:
        """Should be case insensitive."""
        with patch.dict("os.environ", {"MARKITAI_LANG": "ZH"}, clear=True):
            assert i18n.detect_language() == "zh"


class TestGetLanguage:
    """Tests for get_language with caching."""

    def setup_method(self) -> None:
        """Reset language cache before each test."""
        i18n._lang = None

    def test_get_language_caches_result(self) -> None:
        """Should cache the detected language."""
        with patch.dict("os.environ", {"MARKITAI_LANG": "zh"}, clear=True):
            result1 = i18n.get_language()
            assert result1 == "zh"
            assert i18n._lang == "zh"

        # Even with different env, should return cached value
        with patch.dict("os.environ", {"MARKITAI_LANG": "en"}, clear=True):
            result2 = i18n.get_language()
            assert result2 == "zh"  # Still cached


class TestSetLanguage:
    """Tests for set_language override."""

    def setup_method(self) -> None:
        """Reset language cache before each test."""
        i18n._lang = None

    def test_set_language_overrides_detection(self) -> None:
        """Should override detected language."""
        with patch.dict("os.environ", {"MARKITAI_LANG": "en"}, clear=True):
            i18n.set_language("zh")
            assert i18n.get_language() == "zh"

    def test_set_language_updates_cache(self) -> None:
        """Should update cached language."""
        i18n.set_language("en")
        assert i18n._lang == "en"
        i18n.set_language("zh")
        assert i18n._lang == "zh"


class TestTranslate:
    """Tests for translation function."""

    def setup_method(self) -> None:
        """Reset language cache before each test."""
        i18n._lang = None

    def test_translate_english(self) -> None:
        """Should return English text."""
        i18n.set_language("en")
        assert i18n.t("success") == "completed"

    def test_translate_chinese(self) -> None:
        """Should return Chinese text."""
        i18n.set_language("zh")
        assert i18n.t("success") == "完成"

    def test_translate_with_format_args(self) -> None:
        """Should support format arguments."""
        i18n.set_language("en")
        result = i18n.t("doctor.summary", passed=3, optional=1)
        assert "3 required passed" in result
        assert "1 optional missing" in result

    def test_translate_with_format_args_chinese(self) -> None:
        """Should support format arguments in Chinese."""
        i18n.set_language("zh")
        result = i18n.t("doctor.summary", passed=3, optional=1)
        assert "3 必需通过" in result
        assert "1 可选缺失" in result

    def test_unknown_key_returns_key(self) -> None:
        """Should return key itself for unknown translations."""
        i18n.set_language("en")
        assert i18n.t("unknown.key") == "unknown.key"

    def test_translate_cache_command(self) -> None:
        """Should translate cache command texts."""
        i18n.set_language("zh")
        assert i18n.t("cache.title") == "缓存统计"
        result = i18n.t("cache.cleared", count=5)
        assert "5" in result
        assert "缓存" in result

    def test_translate_config_command(self) -> None:
        """Should translate config command texts."""
        i18n.set_language("zh")
        assert i18n.t("config.title") == "配置来源"
        assert i18n.t("config.valid") == "配置有效"


class TestTextsCompleteness:
    """Tests to verify all required texts are defined."""

    def test_all_required_keys_exist(self) -> None:
        """All required keys should exist in TEXTS."""
        required_keys = [
            "success",
            "failed",
            "warning",
            "error",
            "enabled",
            "disabled",
            "not_found",
            "installed",
            "missing",
            "total",
            "doctor.title",
            "doctor.required",
            "doctor.optional",
            "doctor.auth",
            "doctor.summary",
            "doctor.all_good",
            "doctor.fix_hint",
            "cache.title",
            "cache.llm",
            "cache.spa",
            "cache.proxy",
            "cache.entries",
            "cache.cleared",
            "cache.no_entries",
            "config.title",
            "config.cli_args",
            "config.env_vars",
            "config.local_file",
            "config.user_file",
            "config.defaults",
            "config.highest",
            "config.lowest",
            "config.loaded",
            "config.created",
            "config.valid",
        ]

        for key in required_keys:
            assert key in i18n.TEXTS, f"Missing translation key: {key}"

    def test_all_keys_have_both_languages(self) -> None:
        """All keys should have both English and Chinese translations."""
        for key, translations in i18n.TEXTS.items():
            assert "en" in translations, f"Missing English for: {key}"
            assert "zh" in translations, f"Missing Chinese for: {key}"
