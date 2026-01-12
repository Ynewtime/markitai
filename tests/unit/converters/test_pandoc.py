"""Tests for Pandoc converter module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from markit.converters.pandoc import (
    PandocConverter,
    PandocTableConverter,
    check_pandoc_available,
    get_pandoc_version,
)
from markit.exceptions import ConversionError


class TestPandocConverterInit:
    """Tests for PandocConverter initialization."""

    def test_default_init(self):
        """Test default initialization."""
        converter = PandocConverter()

        assert converter.extract_media is True
        assert converter.standalone is False
        assert converter.extra_args == []
        assert converter._pandoc_path is None

    def test_custom_init(self):
        """Test custom initialization."""
        converter = PandocConverter(
            extract_media=False,
            standalone=True,
            extra_args=["--toc", "--columns=80"],
        )

        assert converter.extract_media is False
        assert converter.standalone is True
        assert converter.extra_args == ["--toc", "--columns=80"]

    def test_converter_name(self):
        """Test converter name."""
        converter = PandocConverter()
        assert converter.name == "pandoc"

    def test_supported_extensions(self):
        """Test supported extensions."""
        converter = PandocConverter()

        assert converter.supports(".docx")
        assert converter.supports(".pptx")
        assert converter.supports(".html")
        assert converter.supports(".rst")
        assert converter.supports(".csv")
        assert not converter.supports(".pdf")
        assert not converter.supports(".xyz")


class TestPandocConverterCheckPandoc:
    """Tests for _check_pandoc method."""

    def test_check_pandoc_found(self):
        """Test when Pandoc is found."""
        converter = PandocConverter()

        with patch("shutil.which", return_value="/usr/bin/pandoc"):
            result = converter._check_pandoc()

            assert result is True
            assert converter._pandoc_path == "/usr/bin/pandoc"

    def test_check_pandoc_not_found(self):
        """Test when Pandoc is not found."""
        converter = PandocConverter()

        with patch("shutil.which", return_value=None):
            result = converter._check_pandoc()

            assert result is False

    def test_check_pandoc_cached(self):
        """Test that Pandoc path is cached."""
        converter = PandocConverter()
        converter._pandoc_path = "/cached/path"

        result = converter._check_pandoc()

        assert result is True
        assert converter._pandoc_path == "/cached/path"

    def test_check_pandoc_cached_empty(self):
        """Test cached empty path returns False."""
        converter = PandocConverter()
        converter._pandoc_path = ""

        result = converter._check_pandoc()

        assert result is False


class TestPandocConverterConvert:
    """Tests for PandocConverter.convert method."""

    @pytest.mark.asyncio
    async def test_convert_invalid_file(self, tmp_path):
        """Test convert raises error for invalid file."""
        converter = PandocConverter()
        invalid_path = tmp_path / "nonexistent.docx"

        with pytest.raises(ConversionError):
            await converter.convert(invalid_path)

    @pytest.mark.asyncio
    async def test_convert_unsupported_format(self, tmp_path):
        """Test convert raises error for unsupported format."""
        converter = PandocConverter()
        unsupported = tmp_path / "file.xyz"
        unsupported.touch()

        with pytest.raises(ConversionError):
            await converter.convert(unsupported)

    @pytest.mark.asyncio
    async def test_convert_pandoc_not_installed(self, tmp_path):
        """Test convert raises error when Pandoc not installed."""
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        converter = PandocConverter()

        with patch.object(converter, "_check_pandoc", return_value=False):
            with pytest.raises(ConversionError) as exc_info:
                await converter.convert(docx_path)

            assert "Pandoc is not installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_convert_success(self, tmp_path):
        """Test successful conversion."""
        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        converter = PandocConverter(extract_media=False)

        mock_result = MagicMock()
        mock_result.stdout = "# Converted Content\n\nSome text."
        mock_result.returncode = 0

        with patch.object(converter, "_check_pandoc", return_value=True):
            converter._pandoc_path = "/usr/bin/pandoc"
            with patch("subprocess.run", return_value=mock_result):
                result = await converter.convert(docx_path)

                assert result.markdown is not None
                assert result.metadata["converter"] == "pandoc"

    @pytest.mark.asyncio
    async def test_convert_subprocess_error(self, tmp_path):
        """Test conversion handles subprocess error."""
        import subprocess

        docx_path = tmp_path / "test.docx"
        docx_path.touch()

        converter = PandocConverter()

        error = subprocess.CalledProcessError(1, "pandoc")
        error.stderr = "Pandoc error message"

        with patch.object(converter, "_check_pandoc", return_value=True):
            converter._pandoc_path = "/usr/bin/pandoc"
            with patch("subprocess.run", side_effect=error):
                with pytest.raises(ConversionError) as exc_info:
                    await converter.convert(docx_path)

                assert "Pandoc error" in str(exc_info.value)


class TestPandocConverterBuildCommand:
    """Tests for _build_command method."""

    def test_build_command_basic(self, tmp_path):
        """Test basic command building."""
        converter = PandocConverter(extract_media=False)
        converter._pandoc_path = "/usr/bin/pandoc"

        input_file = tmp_path / "input.docx"
        output_file = tmp_path / "output.md"
        media_dir = tmp_path / "media"

        cmd = converter._build_command(input_file, output_file, media_dir)

        assert cmd[0] == "/usr/bin/pandoc"
        assert str(input_file) in cmd
        assert "-o" in cmd
        assert str(output_file) in cmd
        assert "-t" in cmd
        assert "gfm" in cmd
        assert "--wrap=none" in cmd
        assert "--extract-media" not in cmd

    def test_build_command_with_media_extraction(self, tmp_path):
        """Test command with media extraction."""
        converter = PandocConverter(extract_media=True)
        converter._pandoc_path = "/usr/bin/pandoc"

        input_file = tmp_path / "input.docx"
        output_file = tmp_path / "output.md"
        media_dir = tmp_path / "media"

        cmd = converter._build_command(input_file, output_file, media_dir)

        assert "--extract-media" in cmd

    def test_build_command_standalone(self, tmp_path):
        """Test command with standalone flag."""
        converter = PandocConverter(standalone=True)
        converter._pandoc_path = "/usr/bin/pandoc"

        input_file = tmp_path / "input.docx"
        output_file = tmp_path / "output.md"
        media_dir = tmp_path / "media"

        cmd = converter._build_command(input_file, output_file, media_dir)

        assert "-s" in cmd

    def test_build_command_extra_args(self, tmp_path):
        """Test command with extra arguments."""
        converter = PandocConverter(extra_args=["--toc", "--columns=80"])
        converter._pandoc_path = "/usr/bin/pandoc"

        input_file = tmp_path / "input.docx"
        output_file = tmp_path / "output.md"
        media_dir = tmp_path / "media"

        cmd = converter._build_command(input_file, output_file, media_dir)

        assert "--toc" in cmd
        assert "--columns=80" in cmd


class TestPandocConverterCollectMedia:
    """Tests for _collect_media method."""

    def test_collect_media_images(self, tmp_path):
        """Test collecting image files."""
        converter = PandocConverter()

        media_dir = tmp_path / "media"
        media_dir.mkdir()

        # Create mock image files
        (media_dir / "image1.png").write_bytes(b"png data")
        (media_dir / "image2.jpg").write_bytes(b"jpg data")
        (media_dir / "document.txt").write_text("text file")

        source_file = tmp_path / "source.docx"

        images = converter._collect_media(media_dir, source_file)

        assert len(images) == 2
        assert any(img.format == "png" for img in images)
        assert any(img.format == "jpg" for img in images)

    def test_collect_media_nested(self, tmp_path):
        """Test collecting images from nested directories."""
        converter = PandocConverter()

        media_dir = tmp_path / "media"
        (media_dir / "subdir").mkdir(parents=True)

        (media_dir / "image1.png").write_bytes(b"png data")
        (media_dir / "subdir" / "image2.gif").write_bytes(b"gif data")

        source_file = tmp_path / "source.docx"

        images = converter._collect_media(media_dir, source_file)

        assert len(images) == 2

    def test_collect_media_empty_directory(self, tmp_path):
        """Test collecting from empty directory."""
        converter = PandocConverter()

        media_dir = tmp_path / "media"
        media_dir.mkdir()

        source_file = tmp_path / "source.docx"

        images = converter._collect_media(media_dir, source_file)

        assert len(images) == 0


class TestPandocConverterUpdateImageRefs:
    """Tests for _update_image_refs method."""

    def test_update_image_refs(self):
        """Test updating image references."""
        from markit.converters.base import ExtractedImage

        converter = PandocConverter()

        images = [
            ExtractedImage(
                data=b"data",
                format="png",
                filename="image_001.png",
                source_document=Path("doc.docx"),
                position=1,
                original_path="subdir/original.png",
            )
        ]

        markdown = "![Image](media/subdir/original.png)"

        result = converter._update_image_refs(markdown, images)

        assert "assets/image_001.png" in result
        assert "media/subdir/original.png" not in result

    def test_update_image_refs_no_original_path(self):
        """Test updating refs when no original path."""
        from markit.converters.base import ExtractedImage

        converter = PandocConverter()

        images = [
            ExtractedImage(
                data=b"data",
                format="png",
                filename="image_001.png",
                source_document=Path("doc.docx"),
                position=1,
            )
        ]

        markdown = "![Image](media/image.png)"

        result = converter._update_image_refs(markdown, images)

        # Should not change
        assert result == markdown


class TestPandocConverterGetVersion:
    """Tests for _get_pandoc_version method."""

    def test_get_pandoc_version_success(self):
        """Test getting Pandoc version."""
        converter = PandocConverter()
        converter._pandoc_path = "/usr/bin/pandoc"

        mock_result = MagicMock()
        mock_result.stdout = "pandoc 3.1.2\nCompiled with..."

        with patch("subprocess.run", return_value=mock_result):
            version = converter._get_pandoc_version()

            assert version == "3.1.2"

    def test_get_pandoc_version_error(self):
        """Test getting version when error occurs."""
        converter = PandocConverter()
        converter._pandoc_path = "/usr/bin/pandoc"

        with patch("subprocess.run", side_effect=Exception("Error")):
            version = converter._get_pandoc_version()

            assert version == "unknown"


class TestPandocTableConverter:
    """Tests for PandocTableConverter class."""

    def test_init(self):
        """Test initialization."""
        with patch("shutil.which", return_value="/usr/bin/pandoc"):
            converter = PandocTableConverter()
            assert converter._pandoc_path == "/usr/bin/pandoc"

    def test_csv_to_markdown_no_pandoc(self):
        """Test CSV conversion when Pandoc not available."""
        with patch("shutil.which", return_value=None):
            converter = PandocTableConverter()

            with pytest.raises(RuntimeError, match="Pandoc not available"):
                converter.csv_to_markdown("a,b\n1,2")

    def test_csv_to_markdown_success(self):
        """Test CSV to Markdown conversion."""
        mock_result = MagicMock()
        mock_result.stdout = "| a | b |\n| --- | --- |\n| 1 | 2 |"

        with (
            patch("shutil.which", return_value="/usr/bin/pandoc"),
            patch("subprocess.run", return_value=mock_result),
        ):
            converter = PandocTableConverter()
            result = converter.csv_to_markdown("a,b\n1,2")

            assert "| a | b |" in result

    def test_html_table_to_markdown_no_pandoc(self):
        """Test HTML table conversion when Pandoc not available."""
        with patch("shutil.which", return_value=None):
            converter = PandocTableConverter()

            with pytest.raises(RuntimeError, match="Pandoc not available"):
                converter.html_table_to_markdown("<table><tr><td>A</td></tr></table>")

    def test_html_table_to_markdown_success(self):
        """Test HTML table to Markdown conversion."""
        mock_result = MagicMock()
        mock_result.stdout = "| A |\n| --- |"

        with (
            patch("shutil.which", return_value="/usr/bin/pandoc"),
            patch("subprocess.run", return_value=mock_result),
        ):
            converter = PandocTableConverter()
            result = converter.html_table_to_markdown("<table><tr><td>A</td></tr></table>")

            assert "| A |" in result


class TestCheckPandocAvailable:
    """Tests for check_pandoc_available function."""

    def test_check_pandoc_available_true(self):
        """Test when Pandoc is available."""
        with patch("shutil.which", return_value="/usr/bin/pandoc"):
            assert check_pandoc_available() is True

    def test_check_pandoc_available_false(self):
        """Test when Pandoc is not available."""
        with patch("shutil.which", return_value=None):
            assert check_pandoc_available() is False


class TestGetPandocVersion:
    """Tests for get_pandoc_version function."""

    def test_get_pandoc_version_success(self):
        """Test getting version when Pandoc available."""
        mock_result = MagicMock()
        mock_result.stdout = "pandoc 3.1.2\nMore info..."

        with (
            patch("shutil.which", return_value="/usr/bin/pandoc"),
            patch("subprocess.run", return_value=mock_result),
        ):
            version = get_pandoc_version()

            assert version == "3.1.2"

    def test_get_pandoc_version_not_installed(self):
        """Test getting version when not installed."""
        with patch("shutil.which", return_value=None):
            version = get_pandoc_version()

            assert version is None

    def test_get_pandoc_version_error(self):
        """Test getting version when error occurs."""
        with (
            patch("shutil.which", return_value="/usr/bin/pandoc"),
            patch("subprocess.run", side_effect=Exception("Error")),
        ):
            version = get_pandoc_version()

            assert version is None
