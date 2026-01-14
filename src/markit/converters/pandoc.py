"""Pandoc-based document converter."""

import shutil
import subprocess
import tempfile
from pathlib import Path

import anyio

from markit.converters.base import BaseConverter, ConversionResult, ExtractedImage
from markit.exceptions import ConversionError
from markit.utils.logging import get_logger

log = get_logger(__name__)


class PandocConverter(BaseConverter):
    """Document converter using Pandoc.

    Pandoc is a universal document converter that supports many formats.
    It serves as a reliable fallback when other converters fail.

    Supported input formats:
    - docx, pptx, odt, odp, ods
    - html, epub, rst, textile
    - mediawiki, latex, rtf
    """

    name = "pandoc"
    supported_extensions = {
        ".docx",
        ".pptx",
        ".odt",
        ".odp",
        ".ods",
        ".html",
        ".htm",
        ".epub",
        ".rst",
        ".textile",
        ".mediawiki",
        ".latex",
        ".tex",
        ".rtf",
        ".csv",
    }

    def __init__(
        self,
        extract_media: bool = True,
        standalone: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        """Initialize the Pandoc converter.

        Args:
            extract_media: Extract embedded media files
            standalone: Generate standalone document
            extra_args: Additional Pandoc arguments
        """
        self.extract_media = extract_media
        self.standalone = standalone
        self.extra_args = extra_args or []
        self._pandoc_path: str | None = None

    async def convert(self, file_path: Path) -> ConversionResult:
        """Convert a document to Markdown using Pandoc.

        Args:
            file_path: Path to the input document

        Returns:
            ConversionResult with markdown content and extracted images
        """
        if not await self.validate(file_path):
            raise ConversionError(
                file_path,
                f"Invalid file or unsupported format: {file_path}",
            )

        # Check Pandoc availability
        if not self._check_pandoc():
            raise ConversionError(
                file_path,
                "Pandoc is not installed or not found in PATH",
            )

        log.info("Converting with Pandoc", file=str(file_path))

        try:
            result = await anyio.to_thread.run_sync(  # type: ignore[attr-defined]
                self._convert_sync,
                file_path,
            )
            return result
        except subprocess.CalledProcessError as e:
            log.error(
                "Pandoc conversion failed",
                file=str(file_path),
                error=e.stderr,
            )
            raise ConversionError(file_path, f"Pandoc error: {e.stderr}", cause=e) from e
        except Exception as e:
            log.error(
                "Pandoc conversion failed",
                file=str(file_path),
                error=str(e),
            )
            raise ConversionError(file_path, str(e), cause=e) from e

    def _check_pandoc(self) -> bool:
        """Check if Pandoc is available."""
        if self._pandoc_path is not None:
            return bool(self._pandoc_path)

        self._pandoc_path = shutil.which("pandoc")
        if self._pandoc_path:
            log.debug("Found Pandoc", path=self._pandoc_path)
            return True

        log.warning("Pandoc not found in PATH")
        return False

    def _convert_sync(self, file_path: Path) -> ConversionResult:
        """Synchronous conversion using Pandoc."""
        images: list[ExtractedImage] = []

        # Create temp directory for media extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            media_dir = temp_path / "media"
            output_file = temp_path / "output.md"

            # Build Pandoc command
            cmd = self._build_command(file_path, output_file, media_dir)

            log.debug("Running Pandoc", command=" ".join(cmd))

            # Run Pandoc with explicit UTF-8 for Windows compatibility
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=True,
            )

            # Read output
            if output_file.exists():
                markdown = output_file.read_text(encoding="utf-8")
            else:
                # Pandoc might output to stdout
                markdown = result.stdout

            # Collect extracted media
            if self.extract_media and media_dir.exists():
                images = self._collect_media(media_dir, file_path)

                # Update image references in markdown
                markdown = self._update_image_refs(markdown, images)

        return ConversionResult(
            markdown=markdown,
            images=images,
            metadata={
                "converter": self.name,
                "pandoc_version": self._get_pandoc_version(),
            },
        )

    def _build_command(
        self,
        input_file: Path,
        output_file: Path,
        media_dir: Path,
    ) -> list[str]:
        """Build Pandoc command."""
        cmd = [
            self._pandoc_path or "pandoc",
            str(input_file),
            "-o",
            str(output_file),
            "-t",
            "gfm",  # GitHub Flavored Markdown
            "--wrap=none",  # Don't wrap lines
        ]

        if self.extract_media:
            media_dir.mkdir(parents=True, exist_ok=True)
            cmd.extend(["--extract-media", str(media_dir)])

        if self.standalone:
            cmd.append("-s")

        # Add extra arguments
        cmd.extend(self.extra_args)

        return cmd

    def _collect_media(
        self,
        media_dir: Path,
        source_file: Path,
    ) -> list[ExtractedImage]:
        """Collect extracted media files."""
        images = []
        position = 0

        for media_file in media_dir.rglob("*"):
            if media_file.is_file():
                suffix = media_file.suffix.lower()
                if suffix in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}:
                    position += 1

                    # Generate new filename
                    filename = f"image_{position:03d}{suffix}"

                    extracted = ExtractedImage(
                        data=media_file.read_bytes(),
                        format=suffix.lstrip("."),
                        filename=filename,
                        source_document=source_file,
                        position=position,
                        original_path=str(media_file.relative_to(media_dir)),
                    )
                    images.append(extracted)

        return images

    def _update_image_refs(
        self,
        markdown: str,
        images: list[ExtractedImage],
    ) -> str:
        """Update image references to use new filenames."""
        for img in images:
            if img.original_path:
                # Replace original path with new filename
                old_ref = f"media/{img.original_path}"
                new_ref = f"assets/{img.filename}"
                markdown = markdown.replace(old_ref, new_ref)

        return markdown

    def _get_pandoc_version(self) -> str:
        """Get Pandoc version string."""
        try:
            result = subprocess.run(
                [self._pandoc_path or "pandoc", "--version"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            # First line is "pandoc X.Y.Z"
            first_line = result.stdout.split("\n")[0]
            return first_line.replace("pandoc ", "")
        except Exception:
            return "unknown"


class PandocTableConverter:
    """Convert tables between formats using Pandoc."""

    def __init__(self):
        self._pandoc_path = shutil.which("pandoc")

    def csv_to_markdown(self, csv_content: str) -> str:
        """Convert CSV to Markdown table."""
        if not self._pandoc_path:
            raise RuntimeError("Pandoc not available")

        result = subprocess.run(
            [self._pandoc_path, "-f", "csv", "-t", "gfm"],
            input=csv_content,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return result.stdout

    def html_table_to_markdown(self, html_content: str) -> str:
        """Convert HTML table to Markdown."""
        if not self._pandoc_path:
            raise RuntimeError("Pandoc not available")

        result = subprocess.run(
            [self._pandoc_path, "-f", "html", "-t", "gfm"],
            input=html_content,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
        return result.stdout


def check_pandoc_available() -> bool:
    """Check if Pandoc is installed and available."""
    return shutil.which("pandoc") is not None


def get_pandoc_version() -> str | None:
    """Get Pandoc version or None if not installed."""
    pandoc_path = shutil.which("pandoc")
    if not pandoc_path:
        return None

    try:
        result = subprocess.run(
            [pandoc_path, "--version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        first_line = result.stdout.split("\n")[0]
        return first_line.replace("pandoc ", "")
    except Exception:
        return None
