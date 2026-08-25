"""Tests for the anydoc-backed legacy converters (converter/legacy.py).

.doc/.ppt conversion went from three-platform Office automation (Windows
COM / macOS AppleScript / LibreOffice CLI) to the bundled-Rust anydoc
backend (markitai[legacy] extra). These tests lock the routing, the real
fixture conversions, and the error mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from markitai.converter import get_converter
from markitai.converter.legacy import DocConverter, PptConverter, _load_anydoc
from markitai.utils.errors import ConversionError, MissingDependencyError


class TestLegacyRouting:
    def test_doc_routes_to_anydoc_converter(self) -> None:
        assert type(get_converter("letter.doc")) is DocConverter

    def test_ppt_routes_to_anydoc_converter(self) -> None:
        assert type(get_converter("deck.ppt")) is PptConverter


class TestAnyDocConversion:
    """Real fixture conversions through the anydoc Rust backend."""

    def test_doc_fixture_structure(self, fixtures_dir: Path) -> None:
        result = DocConverter().convert(fixtures_dir / "legacy" / "sample.doc")

        assert result.metadata["original_format"] == "DOC"
        assert result.metadata["backend"] == "anydoc"
        # Headings survive as markdown headings (the Office-automation chain
        # flattened them), and the table keeps GFM structure.
        assert "# Lorem ipsum dolor sit amet" in result.markdown
        assert "| --- | --- | --- |" in result.markdown
        assert "HYPERLINK" not in result.markdown  # no field-code leakage

    def test_ppt_fixture_text(self, fixtures_dir: Path) -> None:
        result = PptConverter().convert(fixtures_dir / "legacy" / "sample.ppt")

        assert result.metadata["original_format"] == "PPT"
        assert "Lorem ipsum" in result.markdown


class TestAnyDocErrors:
    def test_missing_extra_raises_actionable_error(self) -> None:
        with (
            patch.dict(sys.modules, {"anydoc": None}),
            pytest.raises(MissingDependencyError, match=r"markitai\[legacy\]"),
        ):
            _load_anydoc()

    def test_convert_without_extra_raises(self, tmp_path: Path) -> None:
        doc = tmp_path / "x.doc"
        doc.write_bytes(b"not a real doc")
        with (
            patch.dict(sys.modules, {"anydoc": None}),
            pytest.raises(MissingDependencyError),
        ):
            DocConverter().convert(doc)

    def test_encrypted_maps_to_clear_message(self, tmp_path: Path) -> None:
        import anydoc

        doc = tmp_path / "secret.doc"
        doc.write_bytes(b"encrypted")
        with (
            patch.object(
                sys.modules["anydoc"],
                "to_markdown",
                side_effect=anydoc.EncryptedError("encrypted"),
            ),
            pytest.raises(ConversionError, match="password-protected"),
        ):
            DocConverter().convert(doc)

    def test_malformed_maps_to_conversion_error(self, tmp_path: Path) -> None:
        import anydoc

        doc = tmp_path / "broken.doc"
        doc.write_bytes(b"junk")
        with (
            patch.object(
                sys.modules["anydoc"],
                "to_markdown",
                side_effect=anydoc.MalformedError("bad ole"),
            ),
            pytest.raises(ConversionError, match="Failed to convert broken.doc"),
        ):
            DocConverter().convert(doc)
