"""Tests for the KreuzbergConverter and registration logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markitai.converter.base import (
    EXTENSION_MAP,
    ConvertResult,
    FileFormat,
    _converter_registry,
    detect_format,
)


class TestFormatDetection:
    """Tests for format detection of kreuzberg-handled extensions."""

    def test_detect_new_formats(self) -> None:
        """Verify all new extensions map to the correct FileFormat."""
        cases = {
            "book.epub": FileFormat.EPUB,
            "document.rtf": FileFormat.RTF,
            "page.html": FileFormat.HTML,
            "page.htm": FileFormat.HTM,
            "mail.eml": FileFormat.EML,
            "mail.msg": FileFormat.MSG,
            "doc.rst": FileFormat.RST,
            "notes.org": FileFormat.ORG,
            "notebook.ipynb": FileFormat.IPYNB,
            "data.tsv": FileFormat.TSV,
            "paper.tex": FileFormat.TEX,
            "anim.gif": FileFormat.GIF,
            "bitmap.bmp": FileFormat.BMP,
            "scan.tiff": FileFormat.TIFF,
            "scan2.tif": FileFormat.TIFF,
            "page.xhtml": FileFormat.XHTML,
        }
        for filename, expected_format in cases.items():
            assert detect_format(filename) == expected_format, (
                f"Expected {filename} -> {expected_format}"
            )

    def test_extension_map_has_new_entries(self) -> None:
        """Verify EXTENSION_MAP contains all new extensions."""
        expected_extensions = [
            ".epub",
            ".rtf",
            ".html",
            ".htm",
            ".eml",
            ".msg",
            ".rst",
            ".org",
            ".ipynb",
            ".tsv",
            ".tex",
            ".gif",
            ".bmp",
            ".tiff",
            ".tif",
            ".xhtml",
            ".csv",
            ".xml",
            ".ods",
            ".odt",
            ".svg",
            ".numbers",
        ]
        for ext in expected_extensions:
            assert ext in EXTENSION_MAP, f"Missing extension in EXTENSION_MAP: {ext}"


class TestKreuzbergConverter:
    """Tests for KreuzbergConverter (kreuzberg is always mocked)."""

    def _make_mock_kreuzberg(
        self, content: str = "# Hello", metadata: dict | None = None
    ) -> tuple[MagicMock, MagicMock, MagicMock]:
        """Build a mock kreuzberg module with extract_file_sync and ExtractionConfig.

        Returns:
            Tuple of (mock_module, mock_extract_file_sync, mock_ExtractionConfig).
        """
        mock_result = MagicMock()
        mock_result.content = content
        mock_result.metadata = metadata

        mock_extract = MagicMock(return_value=mock_result)
        mock_config_cls = MagicMock()

        mock_module = MagicMock()
        mock_module.extract_file_sync = mock_extract
        mock_module.ExtractionConfig = mock_config_cls

        return mock_module, mock_extract, mock_config_cls

    def test_convert_returns_markdown(self, tmp_path: Path) -> None:
        """extract_file_sync result.content should become ConvertResult.markdown."""
        mock_module, _, _ = self._make_mock_kreuzberg(content="# Hello")

        with patch.dict("sys.modules", {"kreuzberg": mock_module}):
            from markitai.converter.kreuzberg import KreuzbergConverter

            converter = KreuzbergConverter()
            test_file = tmp_path / "doc.epub"
            test_file.write_text("")
            result = converter.convert(test_file)

        assert isinstance(result, ConvertResult)
        assert result.markdown == "# Hello"

    def test_convert_passes_markdown_config(self, tmp_path: Path) -> None:
        """extract_file_sync must be called with ExtractionConfig(output_format='markdown')."""
        mock_module, mock_extract, mock_config_cls = self._make_mock_kreuzberg()

        with patch.dict("sys.modules", {"kreuzberg": mock_module}):
            from markitai.converter.kreuzberg import KreuzbergConverter

            converter = KreuzbergConverter()
            test_file = tmp_path / "doc.rtf"
            test_file.write_text("")
            converter.convert(test_file)

        mock_config_cls.assert_called_once_with(output_format="markdown")
        mock_extract.assert_called_once()
        call_kwargs = mock_extract.call_args
        assert call_kwargs.kwargs.get("config") == mock_config_cls.return_value

    def test_convert_includes_metadata(self, tmp_path: Path) -> None:
        """ConvertResult.metadata should contain source, format, and converter info."""
        mock_module, _, _ = self._make_mock_kreuzberg(
            content="text", metadata={"pages": 3}
        )

        with patch.dict("sys.modules", {"kreuzberg": mock_module}):
            from markitai.converter.kreuzberg import KreuzbergConverter

            converter = KreuzbergConverter()
            test_file = tmp_path / "report.html"
            test_file.write_text("")
            result = converter.convert(test_file)

        assert result.metadata["source"] == str(test_file)
        assert result.metadata["format"] == "HTML"
        assert result.metadata["converter"] == "kreuzberg"
        assert result.metadata["kreuzberg_metadata"] == {"pages": 3}

    def test_convert_metadata_omits_kreuzberg_metadata_when_empty(
        self, tmp_path: Path
    ) -> None:
        """When kreuzberg returns no metadata, 'kreuzberg_metadata' key should be absent."""
        mock_module, _, _ = self._make_mock_kreuzberg(content="x", metadata=None)

        with patch.dict("sys.modules", {"kreuzberg": mock_module}):
            from markitai.converter.kreuzberg import KreuzbergConverter

            converter = KreuzbergConverter()
            test_file = tmp_path / "notes.rst"
            test_file.write_text("")
            result = converter.convert(test_file)

        assert "kreuzberg_metadata" not in result.metadata

    def test_convert_error_handling(self, tmp_path: Path) -> None:
        """When extract_file_sync raises, it should be wrapped in RuntimeError."""
        mock_module, mock_extract, _ = self._make_mock_kreuzberg()
        mock_extract.side_effect = ValueError("corrupt file")

        with patch.dict("sys.modules", {"kreuzberg": mock_module}):
            from markitai.converter.kreuzberg import KreuzbergConverter

            converter = KreuzbergConverter()
            test_file = tmp_path / "bad.epub"
            test_file.write_text("")

            with pytest.raises(RuntimeError, match="kreuzberg failed to extract"):
                converter.convert(test_file)

    def test_convert_import_error_when_not_installed(self, tmp_path: Path) -> None:
        """When kreuzberg is not importable, convert() should raise ImportError."""
        # Remove kreuzberg from sys.modules so the lazy import inside convert() fails
        with patch.dict("sys.modules", {"kreuzberg": None}):
            from markitai.converter.kreuzberg import KreuzbergConverter

            converter = KreuzbergConverter()
            test_file = tmp_path / "doc.odt"
            test_file.write_text("")

            with pytest.raises(ImportError, match="kreuzberg is required"):
                converter.convert(test_file)

    def test_supported_formats_list(self) -> None:
        """KreuzbergConverter.supported_formats should contain expected formats."""
        from markitai.converter.kreuzberg import KreuzbergConverter

        expected = {
            FileFormat.TSV,
            FileFormat.XML,
            FileFormat.ODS,
            FileFormat.ODT,
            FileFormat.RTF,
            FileFormat.RST,
            FileFormat.ORG,
            FileFormat.TEX,
            FileFormat.EML,
        }
        assert set(KreuzbergConverter.supported_formats) == expected

    def test_convert_images_always_empty(self, tmp_path: Path) -> None:
        """KreuzbergConverter should always return an empty images list."""
        mock_module, _, _ = self._make_mock_kreuzberg(content="data")

        with patch.dict("sys.modules", {"kreuzberg": mock_module}):
            from markitai.converter.kreuzberg import KreuzbergConverter

            converter = KreuzbergConverter()
            test_file = tmp_path / "sheet.csv"
            test_file.write_text("")
            result = converter.convert(test_file)

        assert result.images == []
        assert result.has_images is False


class TestKreuzbergRegistration:
    """Tests for register_kreuzberg_converters()."""

    def test_register_skips_when_not_installed(self) -> None:
        """When kreuzberg is not installed, no formats should be registered."""
        from markitai.converter.kreuzberg import (
            KREUZBERG_FORMATS,
            register_kreuzberg_converters,
        )

        # Snapshot existing registry state for kreuzberg formats
        pre_state = {fmt: _converter_registry.get(fmt) for fmt in KREUZBERG_FORMATS}

        with patch("importlib.util.find_spec", return_value=None):
            register_kreuzberg_converters()

        # Registry should be unchanged for all kreuzberg formats
        for fmt in KREUZBERG_FORMATS:
            assert _converter_registry.get(fmt) == pre_state[fmt], (
                f"Registry changed for {fmt} even though kreuzberg is not installed"
            )

    def test_register_when_installed(self) -> None:
        """When kreuzberg is installed, formats without existing converters get registered."""
        from markitai.converter.kreuzberg import (
            KREUZBERG_FORMATS,
            KreuzbergConverter,
            register_kreuzberg_converters,
        )

        # Clear kreuzberg formats from registry to test fresh registration
        saved = {}
        for fmt in KREUZBERG_FORMATS:
            if fmt in _converter_registry:
                saved[fmt] = _converter_registry.pop(fmt)

        try:
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                register_kreuzberg_converters()

            for fmt in KREUZBERG_FORMATS:
                assert _converter_registry.get(fmt) is KreuzbergConverter, (
                    f"{fmt} was not registered with KreuzbergConverter"
                )
        finally:
            # Restore original registry state
            for fmt in KREUZBERG_FORMATS:
                if fmt in saved:
                    _converter_registry[fmt] = saved[fmt]
                elif fmt in _converter_registry:
                    del _converter_registry[fmt]

    def test_register_does_not_override_existing(self) -> None:
        """Formats that already have a converter should not be replaced."""
        from markitai.converter.kreuzberg import (
            KREUZBERG_FORMATS,
            KreuzbergConverter,
            register_kreuzberg_converters,
        )

        # Pick a format and register a dummy converter for it
        test_fmt = KREUZBERG_FORMATS[0]

        class DummyConverter:
            pass

        saved = _converter_registry.get(test_fmt)
        _converter_registry[test_fmt] = DummyConverter  # type: ignore[assignment]

        try:
            with patch("importlib.util.find_spec", return_value=MagicMock()):
                register_kreuzberg_converters()

            # The dummy converter must still be in place
            assert _converter_registry[test_fmt] is DummyConverter, (
                "register_kreuzberg_converters() replaced an existing converter"
            )

            # But other (previously unregistered) formats should get kreuzberg
            # Pick one that was not pre-registered
            other_fmts = [
                f
                for f in KREUZBERG_FORMATS
                if f != test_fmt
                and f not in _converter_registry
                or _converter_registry.get(f) is KreuzbergConverter
            ]
            if other_fmts:
                assert _converter_registry.get(other_fmts[0]) is KreuzbergConverter
        finally:
            # Restore
            if saved is not None:
                _converter_registry[test_fmt] = saved
            else:
                _converter_registry.pop(test_fmt, None)
