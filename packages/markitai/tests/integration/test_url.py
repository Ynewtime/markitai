"""Integration tests for URL conversion and batch processing."""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from click.testing import CliRunner
from PIL import Image

from markitai.cli import app
from markitai.image import (
    _get_extension_from_url,
    _sanitize_image_filename,
    download_url_images,
)
from markitai.urls import (
    find_url_list_files,
    is_url_list_file,
    parse_url_list,
)
from markitai.utils.mime import get_extension_from_mime


def _create_test_png(width: int = 100, height: int = 100) -> bytes:
    """Create a valid PNG image for testing."""
    img = Image.new("RGB", (width, height), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


# =============================================================================
# URL List Parsing Tests
# =============================================================================


class TestUrlListParsing:
    """Tests for URL list file parsing."""

    def test_is_url_list_file_true(self, tmp_path: Path):
        """Test .urls file detection."""
        url_file = tmp_path / "test.urls"
        url_file.touch()
        assert is_url_list_file(url_file) is True

    def test_is_url_list_file_false(self, tmp_path: Path):
        """Test non-.urls file detection."""
        txt_file = tmp_path / "test.txt"
        txt_file.touch()
        assert is_url_list_file(txt_file) is False

    def test_parse_plain_text_format(self, tmp_path: Path):
        """Test parsing plain text URL list."""
        url_file = tmp_path / "urls.urls"
        url_file.write_text(
            """# Comment line
https://example.com
https://test.org custom_name

# Another comment
https://foo.bar
"""
        )

        entries = parse_url_list(url_file)
        assert len(entries) == 3
        assert entries[0].url == "https://example.com"
        assert entries[0].output_name is None
        assert entries[1].url == "https://test.org"
        assert entries[1].output_name == "custom_name"
        assert entries[2].url == "https://foo.bar"

    def test_parse_json_array_format(self, tmp_path: Path):
        """Test parsing JSON array URL list."""
        url_file = tmp_path / "urls.urls"
        url_file.write_text('["https://a.com", "https://b.com"]')

        entries = parse_url_list(url_file)
        assert len(entries) == 2
        assert entries[0].url == "https://a.com"
        assert entries[1].url == "https://b.com"

    def test_parse_json_objects_format(self, tmp_path: Path):
        """Test parsing JSON objects URL list."""
        url_file = tmp_path / "urls.urls"
        url_file.write_text(
            '[{"url": "https://c.com", "output_name": "custom"}, {"url": "https://d.com"}]'
        )

        entries = parse_url_list(url_file)
        assert len(entries) == 2
        assert entries[0].url == "https://c.com"
        assert entries[0].output_name == "custom"
        assert entries[1].url == "https://d.com"
        assert entries[1].output_name is None

    def test_parse_empty_file(self, tmp_path: Path):
        """Test parsing empty URL list."""
        url_file = tmp_path / "empty.urls"
        url_file.write_text("")

        entries = parse_url_list(url_file)
        assert len(entries) == 0

    def test_parse_invalid_urls_skipped(self, tmp_path: Path):
        """Test that invalid URLs are skipped with warning."""
        url_file = tmp_path / "urls.urls"
        url_file.write_text(
            """https://valid.com
not-a-url
https://also-valid.org
"""
        )

        entries = parse_url_list(url_file)
        assert len(entries) == 2
        assert entries[0].url == "https://valid.com"
        assert entries[1].url == "https://also-valid.org"

    def test_parse_file_not_found(self, tmp_path: Path):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            parse_url_list(tmp_path / "nonexistent.urls")

    def test_find_url_list_files(self, tmp_path: Path):
        """Test finding .urls files in directory."""
        # Create nested structure
        (tmp_path / "a.urls").touch()
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        (subdir / "b.urls").touch()
        (tmp_path / "other.txt").touch()

        files = find_url_list_files(tmp_path)
        assert len(files) == 2
        assert any(f.name == "a.urls" for f in files)
        assert any(f.name == "b.urls" for f in files)


# =============================================================================
# URL Image Download Tests
# =============================================================================


class TestUrlImageDownload:
    """Tests for URL image downloading."""

    def test_get_extension_from_content_type(self):
        """Test content-type to extension mapping."""
        assert get_extension_from_mime("image/jpeg") == ".jpg"
        assert get_extension_from_mime("image/png") == ".png"
        assert get_extension_from_mime("image/webp") == ".webp"
        assert get_extension_from_mime("image/gif") == ".gif"
        assert get_extension_from_mime("image/svg+xml") == ".svg"
        assert get_extension_from_mime("text/html") == ".jpg"  # default

    def test_get_extension_from_url(self):
        """Test URL to extension extraction."""
        assert _get_extension_from_url("https://example.com/image.jpg") == ".jpg"
        assert _get_extension_from_url("https://example.com/image.PNG") == ".png"
        assert _get_extension_from_url("https://example.com/image.webp?v=1") == ".webp"
        assert _get_extension_from_url("https://example.com/no-extension") is None

    def test_sanitize_image_filename(self):
        """Test filename sanitization."""
        assert _sanitize_image_filename("normal") == "normal"
        assert _sanitize_image_filename("has:colon") == "has_colon"
        assert _sanitize_image_filename("has/slash") == "has_slash"
        assert _sanitize_image_filename("a" * 200, max_length=50) == "a" * 50
        assert _sanitize_image_filename("") == "image"

    @pytest.mark.asyncio
    async def test_download_url_images_no_images(self, tmp_path: Path):
        """Test download with no images in markdown."""
        from markitai.config import ImageConfig

        markdown = "# Test\n\nNo images here."
        config = ImageConfig()

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        assert result.updated_markdown == markdown
        assert result.downloaded_paths == []
        assert result.failed_urls == []

    @pytest.mark.asyncio
    async def test_download_url_images_skips_data_uris(self, tmp_path: Path):
        """Test that data: URIs are not downloaded."""
        from markitai.config import ImageConfig

        markdown = "![Image](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==)"
        config = ImageConfig()

        result = await download_url_images(
            markdown=markdown,
            output_dir=tmp_path,
            base_url="https://example.com",
            config=config,
        )

        # Should be unchanged (data URIs are skipped)
        assert result.updated_markdown == markdown
        assert result.downloaded_paths == []

    @pytest.mark.asyncio
    async def test_download_url_images_with_mock(self, tmp_path: Path):
        """Test image download with mocked HTTP response."""
        from markitai.config import ImageConfig

        markdown = "# Test\n\n![Alt](https://example.com/test.png)"
        config = ImageConfig()

        # Mock httpx response with valid PNG data
        mock_response = MagicMock()
        mock_response.content = _create_test_png(100, 100)  # Valid 100x100 PNG
        mock_response.headers = {"content-type": "image/png"}
        mock_response.raise_for_status = MagicMock()

        with patch("markitai.image.httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client_class.return_value = mock_client

            result = await download_url_images(
                markdown=markdown,
                output_dir=tmp_path,
                base_url="https://example.com",
                config=config,
                source_name="test_page",
            )

        # Verify image was "downloaded"
        assert len(result.downloaded_paths) == 1
        # Image naming format: {base_name}.{idx:04d}.{extension}
        assert "assets/test_page.0001." in result.updated_markdown
        assert result.failed_urls == []


# =============================================================================
# CLI URL Tests
# =============================================================================


class TestUrlListCli:
    """Tests for .urls file auto-detection."""

    # Uses cli_runner from conftest.py

    def test_url_list_dry_run(self, cli_runner: CliRunner, tmp_path: Path):
        """Test .urls file with --dry-run."""
        url_file = tmp_path / "test.urls"
        url_file.write_text("https://example.com\nhttps://test.org")
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(
            app,
            [
                str(url_file),
                "-o",
                str(output_dir),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "URLs: 2" in result.output or "Dry Run" in result.output

    def test_url_list_with_custom_names(self, cli_runner: CliRunner, tmp_path: Path):
        """Test .urls file with custom output names."""
        url_file = tmp_path / "test.urls"
        url_file.write_text(
            '[{"url": "https://example.com", "output_name": "my_custom_name"}]'
        )
        output_dir = tmp_path / "output"

        result = cli_runner.invoke(
            app,
            [
                str(url_file),
                "-o",
                str(output_dir),
                "--dry-run",
            ],
        )

        assert result.exit_code == 0
        assert "my_custom_name" in result.output


class TestBatchWithUrlsFiles:
    """Tests for batch processing with .urls files in input directory."""

    # Uses cli_runner from conftest.py

    def test_batch_finds_urls_files(self, cli_runner: CliRunner, tmp_path: Path):
        """Test that batch mode finds .urls files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create a .urls file
        url_file = input_dir / "my_urls.urls"
        url_file.write_text("https://example.com")

        result = cli_runner.invoke(
            app,
            [str(input_dir), "-o", str(output_dir), "--dry-run"],
        )

        assert result.exit_code == 0
        assert "1 URLs" in result.output or "URL list" in result.output

    def test_batch_mixed_files_and_urls(self, cli_runner: CliRunner, tmp_path: Path):
        """Test batch with both files and .urls files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        # Create a text file
        (input_dir / "doc.txt").write_text("# Test Document")

        # Create a .urls file
        (input_dir / "urls.urls").write_text("https://example.com")

        result = cli_runner.invoke(
            app,
            [str(input_dir), "-o", str(output_dir), "--dry-run"],
        )

        assert result.exit_code == 0
        # Should mention both files and URLs
        assert "1 files" in result.output or "file" in result.output.lower()
