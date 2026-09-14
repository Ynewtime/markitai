"""Dedicated readers must preserve an explicitly supplied Office adapter."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from markitai.converter.office import DocxConverter, PptxConverter, XlsxConverter


@pytest.mark.parametrize(
    "converter_type,extension",
    [(DocxConverter, "docx"), (XlsxConverter, "xlsx"), (PptxConverter, "pptx")],
)
def test_office_reader_respects_supplied_adapter(
    fixtures_dir, converter_type, extension
):
    path = fixtures_dir / f"sample.{extension}"
    adapter = Mock()
    adapter.convert.return_value = SimpleNamespace(markdown="# Custom", title="Custom")
    converter = converter_type()
    converter._markitdown = adapter

    result = converter.convert(path)

    assert result.markdown == "# Custom"
    assert result.metadata["title"] == "Custom"
    adapter.convert.assert_called_once_with(path, keep_data_uris=True)
