"""Cloudflare Workers AI toMarkdown converter.

Converts local files to Markdown via CF's REST API. Supports PDF, Office,
images, CSV, XML, ODF, and more. Most formats are free; image conversion
consumes Workers AI Neurons quota (10,000/day free).

API docs: https://developers.cloudflare.com/workers-ai/markdown-conversion/
"""

from __future__ import annotations

import logging
import mimetypes
from pathlib import Path
from typing import Any

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
)

logger = logging.getLogger(__name__)

# Formats supported by CF toMarkdown API
CF_SUPPORTED_FORMATS = [
    FileFormat.PDF,
    FileFormat.DOCX,
    FileFormat.XLSX,
    FileFormat.XLS,
    FileFormat.JPEG,
    FileFormat.JPG,
    FileFormat.PNG,
    FileFormat.WEBP,
    FileFormat.SVG,
    FileFormat.CSV,
    FileFormat.XML,
    FileFormat.ODS,
    FileFormat.ODT,
    FileFormat.NUMBERS,
]

# Image formats that consume Workers AI Neurons quota
_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".svg"}

# Extension to MIME type overrides (mimetypes module may not know all)
_MIME_OVERRIDES = {
    ".numbers": "application/vnd.apple.numbers",
    ".ods": "application/vnd.oasis.opendocument.spreadsheet",
    ".odt": "application/vnd.oasis.opendocument.text",
    ".et": "application/vnd.ms-excel",
    ".xlsm": "application/vnd.ms-excel.sheet.macroEnabled.12",
    ".xlsb": "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
}


class CloudflareConverter(BaseConverter):
    """Converter using Cloudflare Workers AI toMarkdown API.

    Pricing (fact-checked 2026-02-23):
    - PDF, Office, HTML, XML, CSV: FREE
    - Images (JPG/PNG/WEBP/SVG): Consumes Neurons (10,000/day free, then $0.011/1K)
    """

    supported_formats = CF_SUPPORTED_FORMATS

    def __init__(
        self,
        api_token: str | None = None,
        account_id: str | None = None,
        config: Any = None,
    ):
        super().__init__(config=config)
        self.api_token = api_token
        self.account_id = account_id

    def will_incur_cost(self, path: Path) -> bool:
        """Check if converting this file will consume Neurons quota."""
        return path.suffix.lower() in _IMAGE_EXTENSIONS

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Synchronous convert — raises NotImplementedError.

        CF toMarkdown is a network API; use convert_async() instead.
        The workflow layer calls convert_async() when the converter supports it.
        """
        raise NotImplementedError(
            "CloudflareConverter requires async execution. "
            "Use convert_async() or ensure the workflow uses async dispatch."
        )

    async def convert_async(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        """Convert a local file via CF Workers AI toMarkdown API.

        Args:
            input_path: Path to the local file
            output_dir: Unused (CF returns markdown directly, no extracted images)

        Returns:
            ConvertResult with markdown content and metadata

        Raises:
            RuntimeError: If credentials missing or CF API returns error
        """
        import httpx

        if not self.api_token or not self.account_id:
            raise RuntimeError(
                "Cloudflare API token and account ID required. "
                "Set in config: fetch.cloudflare.api_token and fetch.cloudflare.account_id"
            )

        # Determine MIME type with fallback.
        ext = input_path.suffix.lower()
        mime = _MIME_OVERRIDES.get(ext)
        if not mime:
            mime, _ = mimetypes.guess_type(str(input_path))
        if not mime:
            logger.warning(
                f"Cannot determine MIME type for {input_path.name}, "
                "falling back to application/octet-stream"
            )
            mime = "application/octet-stream"

        endpoint = (
            f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}"
            f"/ai/tomarkdown"
        )

        if self.will_incur_cost(input_path):
            logger.info(
                f"Image conversion uses Workers AI Neurons quota: {input_path.name}"
            )

        logger.debug(f"Converting {input_path.name} via CF toMarkdown (MIME: {mime})")

        from markitai.fetch import _detect_proxy

        proxy_url = _detect_proxy()
        proxy_config = proxy_url if proxy_url else None

        async with httpx.AsyncClient(
            timeout=60.0,
            proxy=proxy_config,
        ) as client:
            # Read file into memory asynchronously to avoid blocking the event loop.
            import aiofiles

            async with aiofiles.open(input_path, "rb") as f:
                file_content = await f.read()

            response = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {self.api_token}"},
                files={"files": (input_path.name, file_content, mime)},
            )
            response.raise_for_status()

            data = response.json()
            results = data.get("result", [])

            if not results:
                raise RuntimeError(
                    f"CF toMarkdown returned empty result for: {input_path.name}"
                )

            result = results[0]

            if result.get("format") == "error":
                raise RuntimeError(
                    f"CF toMarkdown failed for {input_path.name}: {result.get('error')}"
                )

            markdown = result.get("data", "")
            tokens = result.get("tokens")

            logger.debug(
                f"CF toMarkdown success: {input_path.name}"
                f" ({len(markdown)} chars"
                f"{f', ~{tokens} tokens' if tokens else ''})"
            )

            return ConvertResult(
                markdown=markdown,
                metadata={
                    "converter": "cloudflare-tomarkdown",
                    "tokens": tokens,
                    "mimetype": result.get("mimetype"),
                    "source_name": result.get("name"),
                },
            )
