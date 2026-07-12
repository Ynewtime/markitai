"""Tests for the pure-Python legacy XLS converter (converter/office.py).

Legacy .xls used to go through the Office-app upgrade chain (COM /
LibreOffice / AppleScript). It now converts directly through MarkItDown's
xlrd path — these tests lock that no Office machinery is involved.
"""

from __future__ import annotations

from pathlib import Path

from markitai.converter import get_converter
from markitai.converter.base import FileFormat
from markitai.converter.legacy import COM_CONFIGS, LegacyOfficeConverter
from markitai.converter.office import OfficeConverter, XlsConverter


class TestXlsRouting:
    def test_xls_routes_to_pure_python_converter(self) -> None:
        converter = get_converter("sheet.xls")
        assert type(converter) is XlsConverter
        assert isinstance(converter, OfficeConverter)
        assert not isinstance(converter, LegacyOfficeConverter)
        assert converter.supported_formats == [FileFormat.XLS]

    def test_xls_left_the_upgrade_chain(self) -> None:
        # Regression lock: neither Windows COM nor the LegacyOfficeConverter
        # target map may quietly re-adopt .xls.
        assert ".xls" not in COM_CONFIGS
        assert ".xls" not in LegacyOfficeConverter.TARGET_FORMAT


class TestXlsConversion:
    def test_converts_fixture_without_office(self, fixtures_dir: Path) -> None:
        # Runs everywhere (CI included): pandas + xlrd only, no Office app,
        # no LibreOffice, no subprocess.
        result = XlsConverter().convert(fixtures_dir / "legacy" / "sample.xls")

        assert "## Sheet1" in result.markdown
        assert "| First Name | Last Name |" in result.markdown
        assert result.markdown.count("|") > 100  # full table came through
        assert result.metadata["format"] == "XLS"
        assert result.metadata["converter"] == "markitdown"
