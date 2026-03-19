"""Tests for --kreuzberg converter engine override."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


class TestKreuzbergEngineOverride:
    def test_kreuzberg_flag_sets_config(self):
        """--kreuzberg should set kreuzberg_convert_enabled in config."""
        from markitai.config import FetchConfig

        cfg = FetchConfig()
        assert cfg.kreuzberg_convert_enabled is False

    def test_kreuzberg_override_in_workflow(self, tmp_path: Path):
        """When kreuzberg_convert_enabled=True, workflow should use KreuzbergConverter."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, validate_and_detect_format

        # Create a test file
        test_file = tmp_path / "test.docx"
        test_file.write_bytes(b"fake docx content")

        cfg = MarkitaiConfig()
        cfg.fetch.kreuzberg_convert_enabled = True

        ctx = ConversionContext(
            input_path=test_file,
            output_dir=tmp_path / "output",
            config=cfg,
        )

        with (
            patch("importlib.util.find_spec", return_value=MagicMock()),
            patch("markitai.converter.kreuzberg.KreuzbergConverter") as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            result = validate_and_detect_format(ctx, max_size=500_000_000)

        assert result.success
        assert ctx.converter is not None

    def test_kreuzberg_not_installed_gives_error(self, tmp_path: Path):
        """When kreuzberg not installed, should return clear error."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, validate_and_detect_format

        test_file = tmp_path / "test.tsv"
        test_file.write_text("a\tb\n1\t2\n")

        cfg = MarkitaiConfig()
        cfg.fetch.kreuzberg_convert_enabled = True

        ctx = ConversionContext(
            input_path=test_file,
            output_dir=tmp_path / "output",
            config=cfg,
        )

        with patch("importlib.util.find_spec", return_value=None):
            result = validate_and_detect_format(ctx, max_size=500_000_000)

        assert not result.success
        assert "kreuzberg" in result.error.lower()

    def test_no_flag_uses_default_converter(self, tmp_path: Path):
        """Without --kreuzberg, normal converter registry should be used."""
        from markitai.config import MarkitaiConfig
        from markitai.workflow.core import ConversionContext, validate_and_detect_format

        test_file = tmp_path / "test.html"
        test_file.write_text("<html><body><p>test</p></body></html>")

        cfg = MarkitaiConfig()
        # kreuzberg_convert_enabled defaults to False

        ctx = ConversionContext(
            input_path=test_file,
            output_dir=tmp_path / "output",
            config=cfg,
        )

        result = validate_and_detect_format(ctx, max_size=500_000_000)
        assert result.success
        # Should NOT be KreuzbergConverter
        converter_name = type(ctx.converter).__name__
        assert converter_name != "KreuzbergConverter"
