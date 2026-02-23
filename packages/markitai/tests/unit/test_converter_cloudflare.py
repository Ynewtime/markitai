"""Tests for Cloudflare Workers AI toMarkdown converter."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from markitai.converter.base import FileFormat, detect_format


class TestCloudflareFormats:
    """Tests for CF-specific file format support."""

    def test_svg_format_exists(self):
        assert FileFormat.SVG.value == "svg"
        assert detect_format("image.svg") == FileFormat.SVG

    def test_csv_format_exists(self):
        assert FileFormat.CSV.value == "csv"
        assert detect_format("data.csv") == FileFormat.CSV

    def test_xml_format_exists(self):
        assert FileFormat.XML.value == "xml"
        assert detect_format("feed.xml") == FileFormat.XML

    def test_ods_format_exists(self):
        assert FileFormat.ODS.value == "ods"
        assert detect_format("sheet.ods") == FileFormat.ODS

    def test_odt_format_exists(self):
        assert FileFormat.ODT.value == "odt"
        assert detect_format("doc.odt") == FileFormat.ODT

    def test_numbers_format_exists(self):
        assert FileFormat.NUMBERS.value == "numbers"
        assert detect_format("budget.numbers") == FileFormat.NUMBERS

    def test_existing_formats_unchanged(self):
        """Existing formats still work."""
        assert detect_format("report.pdf") == FileFormat.PDF
        assert detect_format("doc.docx") == FileFormat.DOCX
        assert detect_format("photo.jpg") == FileFormat.JPG


class TestCloudflareConverter:
    """Tests for CloudflareConverter."""

    def test_supported_formats(self):
        """Converter declares all CF-supported formats."""
        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token="test", account_id="test")
        # Must include all CF toMarkdown supported formats
        assert FileFormat.PDF in converter.supported_formats
        assert FileFormat.DOCX in converter.supported_formats
        assert FileFormat.XLSX in converter.supported_formats
        assert FileFormat.CSV in converter.supported_formats
        assert FileFormat.SVG in converter.supported_formats
        assert FileFormat.XML in converter.supported_formats
        assert FileFormat.ODS in converter.supported_formats
        assert FileFormat.ODT in converter.supported_formats
        assert FileFormat.NUMBERS in converter.supported_formats
        # Images
        assert FileFormat.JPG in converter.supported_formats
        assert FileFormat.JPEG in converter.supported_formats
        assert FileFormat.PNG in converter.supported_formats
        assert FileFormat.WEBP in converter.supported_formats
        # NOT supported by CF toMarkdown
        assert FileFormat.PPTX not in converter.supported_formats
        assert FileFormat.PPT not in converter.supported_formats
        assert FileFormat.DOC not in converter.supported_formats

    def test_will_incur_cost(self):
        """Image formats cost Neurons, others are free."""
        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token="test", account_id="test")
        assert converter.will_incur_cost(Path("photo.jpg")) is True
        assert converter.will_incur_cost(Path("image.png")) is True
        assert converter.will_incur_cost(Path("icon.svg")) is True
        assert converter.will_incur_cost(Path("report.pdf")) is False
        assert converter.will_incur_cost(Path("data.xlsx")) is False
        assert converter.will_incur_cost(Path("data.csv")) is False

    @pytest.mark.asyncio
    async def test_convert_pdf_success(self):
        """Successful PDF conversion via CF API."""
        from markitai.converter.cloudflare import CloudflareConverter

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [
                {
                    "name": "report.pdf",
                    "format": "markdown",
                    "mimetype": "application/pdf",
                    "tokens": 4231,
                    "data": "# report.pdf\n## Metadata\n- Title=Annual Report\n\n## Contents\n### Page 1\nContent here.",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
            patch("aiofiles.open") as mock_aiofiles,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            # Mock aiofiles
            mock_file = AsyncMock()
            mock_file.read = AsyncMock(return_value=b"%PDF-1.4 fake content")
            mock_aiofiles_ctx = AsyncMock()
            mock_aiofiles_ctx.__aenter__.return_value = mock_file
            mock_aiofiles_ctx.__aexit__.return_value = None
            mock_aiofiles.return_value = mock_aiofiles_ctx

            converter = CloudflareConverter(
                api_token="test-token", account_id="test-account"
            )
            result = await converter.convert_async(Path("report.pdf"))

        assert "Annual Report" in result.markdown
        assert result.metadata["converter"] == "cloudflare-tomarkdown"
        assert result.metadata["tokens"] == 4231

    @pytest.mark.asyncio
    async def test_convert_api_error(self):
        """CF API error in result raises exception."""
        from markitai.converter.cloudflare import CloudflareConverter

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": [
                {
                    "name": "bad.pdf",
                    "format": "error",
                    "mimetype": "application/pdf",
                    "error": "Failed to parse PDF",
                }
            ]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)

        with (
            patch("markitai.fetch._detect_proxy", return_value=""),
            patch("httpx.AsyncClient") as mock_client_class,
            patch("aiofiles.open") as mock_aiofiles,
        ):
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_client
            mock_ctx.__aexit__.return_value = None
            mock_client_class.return_value = mock_ctx

            mock_file = AsyncMock()
            mock_file.read = AsyncMock(return_value=b"fake content")
            mock_aiofiles_ctx = AsyncMock()
            mock_aiofiles_ctx.__aenter__.return_value = mock_file
            mock_aiofiles_ctx.__aexit__.return_value = None
            mock_aiofiles.return_value = mock_aiofiles_ctx

            converter = CloudflareConverter(
                api_token="test-token", account_id="test-account"
            )
            with pytest.raises(RuntimeError, match="Failed to parse PDF"):
                await converter.convert_async(Path("bad.pdf"))

    @pytest.mark.asyncio
    async def test_convert_no_credentials(self):
        """Missing credentials raises clear error."""
        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token=None, account_id="test")
        with pytest.raises(
            RuntimeError, match="Cloudflare API token and account ID required"
        ):
            await converter.convert_async(Path("report.pdf"))


class TestCloudflareConverterIntegration:
    """Tests for CF converter integration with workflow."""

    def test_cf_converter_not_registered_by_default(self):
        """CF converter does NOT auto-register (won't override local converters)."""
        from markitai.converter.base import get_converter

        # PDF should still use local PdfConverter, not CloudflareConverter
        converter = get_converter("report.pdf")
        assert converter is not None
        assert type(converter).__name__ != "CloudflareConverter"

    def test_cf_formats_detected(self):
        """New formats (svg, csv, etc.) are detected but have no local converter."""
        from markitai.converter.base import get_converter

        assert detect_format("data.csv") == FileFormat.CSV
        assert get_converter("data.csv") is None  # No local converter

    def test_cf_converter_supports_new_formats(self):
        """CloudflareConverter handles formats that local converters don't."""
        from markitai.converter.cloudflare import CloudflareConverter

        converter = CloudflareConverter(api_token="test", account_id="test")
        # Formats with no local converter
        assert converter.can_convert("data.csv")
        assert converter.can_convert("feed.xml")
        assert converter.can_convert("sheet.ods")
        assert converter.can_convert("doc.odt")
        assert converter.can_convert("budget.numbers")
        assert converter.can_convert("icon.svg")
