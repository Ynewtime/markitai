"""Tests for HEIC/HEIF/AVIF input support (optional pillow-heif extra)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import markitai.converter.heif as heif_mod
from markitai.config import MarkitaiConfig
from markitai.converter import FileFormat, detect_format, get_converter
from markitai.converter.base import IMAGE_ONLY_FORMATS
from markitai.converter.heif import is_heif_container
from markitai.converter.image import ImageConverter


def _ftyp(brand: bytes) -> bytes:
    """Synthesise a minimal ISO-BMFF ftyp box with the given major brand."""
    return b"\x00\x00\x00\x18ftyp" + brand + b"\x00" * 12


class TestHeifSniff:
    def test_detects_known_brands(self) -> None:
        for brand in (b"heic", b"heix", b"mif1", b"msf1", b"avif", b"avis", b"avcs"):
            assert is_heif_container(_ftyp(brand)), brand

    def test_rejects_non_heif(self) -> None:
        assert not is_heif_container(b"")
        assert not is_heif_container(b"hello world")
        assert not is_heif_container(b"\x89PNG\r\n\x1a\n0000")
        assert not is_heif_container(_ftyp(b"xxxx"))
        # Truncated: shorter than the 12-byte sniff window
        assert not is_heif_container(b"\x00\x00\x00\x18ftyphe")


class TestHeifRegistration:
    def test_detect_formats(self) -> None:
        assert detect_format("photo.heic") == FileFormat.HEIC
        assert detect_format("photo.HEIC") == FileFormat.HEIC
        assert detect_format("photo.heif") == FileFormat.HEIF
        assert detect_format("photo.avif") == FileFormat.AVIF

    def test_image_converter_supports_heif_formats(self) -> None:
        assert FileFormat.HEIC in ImageConverter.supported_formats
        assert FileFormat.HEIF in ImageConverter.supported_formats
        assert FileFormat.AVIF in ImageConverter.supported_formats

    def test_heif_formats_are_image_only(self) -> None:
        assert FileFormat.HEIC in IMAGE_ONLY_FORMATS
        assert FileFormat.HEIF in IMAGE_ONLY_FORMATS
        assert FileFormat.AVIF in IMAGE_ONLY_FORMATS

    def test_get_converter_for_heic(self) -> None:
        converter = get_converter("photo.heic")
        assert isinstance(converter, ImageConverter)


class TestHeifNotInstalled:
    """Error path when the markitai[heif] extra is missing."""

    def test_actionable_error_names_extra(self, tmp_path: Path) -> None:
        heic = tmp_path / "photo.heic"
        heic.write_bytes(_ftyp(b"heic") + b"\x00" * 64)

        config = MarkitaiConfig()
        config.ocr.enabled = False
        converter = ImageConverter(config)

        with (
            patch.object(heif_mod, "_opener_registered", False),
            patch.dict(sys.modules, {"pillow_heif": None}),
            pytest.raises(ImportError, match=r"markitai\[heif\]"),
        ):
            converter.convert(heic, output_dir=tmp_path / "out")

    def test_mislabeled_non_heif_file_does_not_require_extra(
        self, tmp_path: Path
    ) -> None:
        # A renamed PNG with a .heic extension sniffs as non-HEIF and is
        # handled by plain Pillow — no pillow-heif needed.
        fake = tmp_path / "actually-a-png.heic"
        Image.new("RGB", (8, 8), "red").save(fake, format="PNG")

        config = MarkitaiConfig()
        config.ocr.enabled = False
        converter = ImageConverter(config)

        with (
            patch.object(heif_mod, "_opener_registered", False),
            patch.dict(sys.modules, {"pillow_heif": None}),
        ):
            result = converter.convert(fake, output_dir=tmp_path / "out")

        assert ".png" in result.metadata["asset_path"]


class TestHeifDecode:
    """Decode tests — require pillow-heif (markitai[heif])."""

    @pytest.fixture
    def sample_heic(self, tmp_path: Path) -> Path:
        pillow_heif = pytest.importorskip("pillow_heif")
        pillow_heif.register_heif_opener()
        heic = tmp_path / "photo.heic"
        Image.new("RGB", (64, 32), "green").save(heic, format="HEIF")
        return heic

    def test_generated_heic_sniffs_as_heif(self, sample_heic: Path) -> None:
        assert is_heif_container(sample_heic.read_bytes()[:12])

    def test_convert_transcodes_to_png_asset(
        self, sample_heic: Path, tmp_path: Path
    ) -> None:
        config = MarkitaiConfig()
        config.ocr.enabled = False
        converter = ImageConverter(config)
        out = tmp_path / "out"

        result = converter.convert(sample_heic, output_dir=out)

        asset_path = result.metadata["asset_path"]
        assert asset_path.endswith(".png")
        # markdown references the PNG, never the raw .heic
        assert asset_path in result.markdown
        assert ".heic" not in result.markdown
        png_file = out / asset_path
        assert png_file.exists()
        with Image.open(png_file) as img:
            assert img.format == "PNG"
            assert img.size == (64, 32)

    def test_exif_orientation_applied_post_decode(self, tmp_path: Path) -> None:
        pillow_heif = pytest.importorskip("pillow_heif")
        pillow_heif.register_heif_opener()

        heic = tmp_path / "rotated.heic"
        exif = Image.Exif()
        exif[274] = 6  # Orientation: rotate 90 CW to display upright
        Image.new("RGB", (64, 32), "blue").save(
            heic, format="HEIF", exif=exif.tobytes()
        )

        dest = tmp_path / "rotated.png"
        heif_mod.decode_to_png(heic, dest)

        with Image.open(dest) as img:
            assert img.size == (32, 64)  # dimensions swapped by transpose

    def test_ocr_runs_on_decoded_png(self, sample_heic: Path, tmp_path: Path) -> None:
        config = MarkitaiConfig()
        config.ocr.enabled = True
        config.llm.enabled = False
        converter = ImageConverter(config)

        with patch("markitai.ocr.OCRProcessor") as MockOCR:
            mock_processor = MagicMock()
            mock_processor.recognize_to_markdown.return_value = "OCR text"
            MockOCR.return_value = mock_processor

            result = converter.convert(sample_heic, output_dir=tmp_path / "out")

        ocr_arg = mock_processor.recognize_to_markdown.call_args[0][0]
        assert Path(ocr_arg).suffix == ".png"
        assert "OCR text" in result.markdown

    def test_convert_without_output_dir(self, sample_heic: Path) -> None:
        config = MarkitaiConfig()
        config.ocr.enabled = False
        converter = ImageConverter(config)

        result = converter.convert(sample_heic)

        assert result.markdown.startswith("# photo")
