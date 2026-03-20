"""Tests for cli/processors/validators.py validation functions."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markitai.cli.processors.validators import (
    _check_copilot_unsupported_models,
    check_playwright_for_urls,
    check_vision_model_config,
    warn_case_sensitivity_mismatches,
)


def _make_model_config(
    model: str, weight: float = 1.0, supports_vision: bool | None = None
) -> MagicMock:
    """Create a mock ModelConfig for testing."""
    mc = MagicMock()
    mc.litellm_params.model = model
    mc.litellm_params.weight = weight
    if supports_vision is not None:
        mc.model_info = MagicMock()
        mc.model_info.supports_vision = supports_vision
    else:
        mc.model_info = None
    return mc


def _make_cfg(
    llm_enabled: bool = True,
    model_list: list | None = None,
    alt_enabled: bool = False,
    desc_enabled: bool = False,
) -> MagicMock:
    """Create a mock configuration for testing."""
    cfg = MagicMock()
    cfg.llm.enabled = llm_enabled
    cfg.llm.model_list = model_list or []
    cfg.image.alt_enabled = alt_enabled
    cfg.image.desc_enabled = desc_enabled
    return cfg


class TestCheckCopilotUnsupportedModels:
    """Tests for _check_copilot_unsupported_models."""

    def test_warns_on_copilot_o1(self):
        """Should warn about copilot/o1 models."""
        model_list = [_make_model_config("copilot/o1-preview", weight=1)]
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            _check_copilot_unsupported_models(model_list, console)
            mock_ui.warning.assert_called_once()
            assert "o1-preview" in mock_ui.warning.call_args[0][0]

    def test_warns_on_copilot_o3(self):
        """Should warn about copilot/o3 models."""
        model_list = [_make_model_config("copilot/o3-mini", weight=1)]
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            _check_copilot_unsupported_models(model_list, console)
            mock_ui.warning.assert_called_once()

    def test_no_warning_for_weight_zero(self):
        """Should NOT warn about disabled models (weight=0)."""
        model_list = [_make_model_config("copilot/o1", weight=0)]
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            _check_copilot_unsupported_models(model_list, console)
            mock_ui.warning.assert_not_called()

    def test_no_warning_for_supported_models(self):
        """Should not warn about supported copilot models."""
        model_list = [
            _make_model_config("copilot/claude-sonnet-4.5"),
            _make_model_config("copilot/gpt-4o"),
        ]
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            _check_copilot_unsupported_models(model_list, console)
            mock_ui.warning.assert_not_called()

    def test_no_warning_for_empty_list(self):
        """Should handle empty model list."""
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            _check_copilot_unsupported_models([], console)
            mock_ui.warning.assert_not_called()


class TestCheckVisionModelConfig:
    """Tests for check_vision_model_config."""

    def test_skips_if_no_image_analysis(self):
        """Should return early if alt and desc are both disabled."""
        cfg = _make_cfg(alt_enabled=False, desc_enabled=False)
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            check_vision_model_config(cfg, console)
            # No vision-specific warnings should be emitted
            for call in mock_ui.warning.call_args_list:
                assert "vision" not in call[0][0].lower()
                assert "No vision" not in call[0][0]

    def test_warns_if_llm_disabled_but_alt_enabled(self):
        """Should warn that alt/desc requires LLM."""
        cfg = _make_cfg(llm_enabled=False, alt_enabled=True)
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            check_vision_model_config(cfg, console)
            mock_ui.warning.assert_called()
            assert "LLM" in mock_ui.warning.call_args[0][0]

    def test_config_override_supports_vision(self):
        """Config override (supports_vision=True) should be detected."""
        model = _make_model_config("openai/gpt-4o-mini", supports_vision=True)
        cfg = _make_cfg(alt_enabled=True, model_list=[model])
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            check_vision_model_config(cfg, console)
            # No "No vision-capable models" warning
            for call in mock_ui.warning.call_args_list:
                assert "No vision-capable" not in call[0][0]

    def test_local_provider_auto_vision(self):
        """Local provider models (claude-agent/) should be auto-detected as vision-capable."""
        model = _make_model_config("claude-agent/sonnet")
        cfg = _make_cfg(alt_enabled=True, model_list=[model])
        console = MagicMock()

        with (
            patch("markitai.cli.processors.validators.ui") as mock_ui,
            patch("markitai.providers.is_local_provider_model", return_value=True),
        ):
            check_vision_model_config(cfg, console)
            for call in mock_ui.warning.call_args_list:
                assert "No vision-capable" not in call[0][0]

    def test_warns_no_vision_models(self):
        """Should warn when no vision-capable models are detected."""
        model = _make_model_config("openai/gpt-3.5-turbo", supports_vision=False)
        cfg = _make_cfg(alt_enabled=True, model_list=[model])
        console = MagicMock()

        with (
            patch("markitai.cli.processors.validators.ui") as mock_ui,
            patch("markitai.providers.is_local_provider_model", return_value=False),
            patch(
                "markitai.llm.get_model_info_cached",
                return_value={"supports_vision": False},
            ),
        ):
            check_vision_model_config(cfg, console)
            warning_texts = [c[0][0] for c in mock_ui.warning.call_args_list]
            assert any("No vision-capable" in t for t in warning_texts)

    def test_truncates_long_model_list(self):
        """Model list with >3 models should show truncated display."""
        models = [_make_model_config(f"openai/model-{i}") for i in range(5)]
        cfg = _make_cfg(alt_enabled=True, model_list=models)
        console = MagicMock()

        with (
            patch("markitai.cli.processors.validators.ui") as mock_ui,
            patch("markitai.providers.is_local_provider_model", return_value=False),
            patch(
                "markitai.llm.get_model_info_cached",
                return_value={"supports_vision": False},
            ),
        ):
            check_vision_model_config(cfg, console)
            step_texts = [c[0][0] for c in mock_ui.step.call_args_list]
            assert any("+2 more" in t for t in step_texts)


class TestCheckPlaywrightForUrls:
    """Tests for check_playwright_for_urls."""

    def test_skips_for_static_strategy(self):
        """Should skip check for STATIC fetch strategy."""
        from markitai.fetch import FetchStrategy

        cfg = MagicMock()
        cfg.fetch.strategy = FetchStrategy.STATIC
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            check_playwright_for_urls(cfg, console)
            mock_ui.warning.assert_not_called()

    def test_skips_for_jina_strategy(self):
        """Should skip check for JINA fetch strategy."""
        from markitai.fetch import FetchStrategy

        cfg = MagicMock()
        cfg.fetch.strategy = FetchStrategy.JINA
        console = MagicMock()

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            check_playwright_for_urls(cfg, console)
            mock_ui.warning.assert_not_called()

    def test_warns_if_playwright_unavailable(self):
        """Should warn when Playwright is not available."""
        from markitai.fetch import FetchStrategy

        cfg = MagicMock()
        cfg.fetch.strategy = FetchStrategy.AUTO
        console = MagicMock()

        with (
            patch("markitai.cli.processors.validators.ui") as mock_ui,
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=False,
            ),
        ):
            check_playwright_for_urls(cfg, console)
            mock_ui.warning.assert_called_once()
            assert "Playwright" in mock_ui.warning.call_args[0][0]

    def test_no_warning_if_playwright_available(self):
        """Should not warn when Playwright is available."""
        from markitai.fetch import FetchStrategy

        cfg = MagicMock()
        cfg.fetch.strategy = FetchStrategy.AUTO
        console = MagicMock()

        with (
            patch("markitai.cli.processors.validators.ui") as mock_ui,
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=True,
            ),
        ):
            check_playwright_for_urls(cfg, console)
            mock_ui.warning.assert_not_called()


class TestWarnCaseSensitivityMismatches:
    """Tests for warn_case_sensitivity_mismatches."""

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fnmatch is case-insensitive on Windows"
    )
    def test_detects_case_mismatch(self, tmp_path: Path):
        """Should detect files that would match pattern if case-insensitive."""
        files = [tmp_path / "IMAGE.JPG"]
        patterns = ["*.jpg"]

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches(files, tmp_path, patterns)
            mock_ui.warning.assert_called_once()

    def test_no_warning_for_exact_match(self, tmp_path: Path):
        """Should not warn when file exactly matches pattern."""
        files = [tmp_path / "photo.jpg"]
        patterns = ["*.jpg"]

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches(files, tmp_path, patterns)
            mock_ui.warning.assert_not_called()

    def test_no_warning_for_no_match(self, tmp_path: Path):
        """Should not warn when file doesn't match even case-insensitively."""
        files = [tmp_path / "document.pdf"]
        patterns = ["*.jpg"]

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches(files, tmp_path, patterns)
            mock_ui.warning.assert_not_called()

    def test_empty_files(self, tmp_path: Path):
        """Should handle empty file list."""
        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches([], tmp_path, ["*.jpg"])
            mock_ui.warning.assert_not_called()

    def test_empty_patterns(self, tmp_path: Path):
        """Should handle empty pattern list."""
        files = [tmp_path / "IMAGE.JPG"]

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches(files, tmp_path, [])
            mock_ui.warning.assert_not_called()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fnmatch is case-insensitive on Windows"
    )
    def test_file_outside_input_dir(self, tmp_path: Path):
        """Files outside input_dir should use filename as fallback."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        files = [other_dir / "IMAGE.JPG"]
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        patterns = ["*.jpg"]

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches(files, input_dir, patterns)
            # Should still detect mismatch via filename fallback
            mock_ui.warning.assert_called_once()

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fnmatch is case-insensitive on Windows"
    )
    def test_max_3_examples_per_pattern(self, tmp_path: Path):
        """Should show at most 3 examples per pattern."""
        files = [tmp_path / f"IMAGE{i}.JPG" for i in range(5)]
        patterns = ["*.jpg"]

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches(files, tmp_path, patterns)
            step_texts = [c[0][0] for c in mock_ui.step.call_args_list]
            assert any("2 more" in t for t in step_texts)

    @pytest.mark.skipif(
        sys.platform == "win32", reason="fnmatch is case-insensitive on Windows"
    )
    def test_backslash_pattern_normalized(self, tmp_path: Path):
        """Backslash in pattern should be normalized to forward slash."""
        subdir = tmp_path / "sub"
        subdir.mkdir()
        files = [subdir / "IMAGE.JPG"]
        patterns = ["sub\\*.jpg"]

        with patch("markitai.cli.processors.validators.ui") as mock_ui:
            warn_case_sensitivity_mismatches(files, tmp_path, patterns)
            mock_ui.warning.assert_called_once()
