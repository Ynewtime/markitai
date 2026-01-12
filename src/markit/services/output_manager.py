"""Output management service for writing conversion results.

This service handles all file I/O for conversion results, extracted
from ConversionPipeline to enable independent testing and reuse.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from markit.exceptions import ConversionError
from markit.utils.logging import get_logger

if TYPE_CHECKING:
    from markit.converters.base import ConversionResult
    from markit.image.analyzer import ImageAnalysis

log = get_logger(__name__)


class OutputManager:
    """Manages output writing: files, assets, conflict resolution.

    This service handles all file I/O for conversion results, extracted
    from ConversionPipeline to enable independent testing and reuse.

    Features:
    - Markdown file writing with conflict resolution
    - Asset directory management
    - Image description file generation
    - Configurable conflict resolution strategies
    """

    def __init__(
        self,
        on_conflict: Literal["skip", "overwrite", "rename"] = "rename",
        create_assets_subdir: bool = True,
        generate_image_descriptions: bool = False,
    ) -> None:
        """Initialize the output manager.

        Args:
            on_conflict: Strategy for handling existing files
                - "skip": Raise error if file exists
                - "overwrite": Overwrite existing file
                - "rename": Add numeric suffix to filename
            create_assets_subdir: Create assets/ subdirectory for images
            generate_image_descriptions: Generate .md files for image descriptions
        """
        self.on_conflict = on_conflict
        self.create_assets_subdir = create_assets_subdir
        self.generate_image_descriptions = generate_image_descriptions

    async def write_output(
        self,
        input_file: Path,
        output_dir: Path,
        result: "ConversionResult",
        image_info_list: list[Any] | None = None,
        skip_images: bool = False,
    ) -> Path:
        """Write conversion result to output directory.

        Args:
            input_file: Original input file
            output_dir: Output directory
            result: Conversion result containing markdown and images
            image_info_list: List of processed image info (with analysis results)
            skip_images: If True, skip writing images (they were already written
                        immediately during analysis)

        Returns:
            Path to the output markdown file
        """
        import anyio

        # Determine output file path (preserve original extension for clarity)
        output_file = output_dir / f"{input_file.name}.md"

        # Handle conflicts
        output_file = self.resolve_conflict(output_file)

        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write markdown content
        async with await anyio.open_file(output_file, "w", encoding="utf-8") as f:
            await f.write(result.markdown)

        log.info("Output written", path=str(output_file))

        # Write images if any (skip if already written during analysis)
        if result.images and self.create_assets_subdir and not skip_images:
            assets_dir = output_dir / "assets"
            assets_dir.mkdir(exist_ok=True)

            # Build a lookup for image analysis
            analysis_lookup: dict[str, ImageAnalysis | None] = {}
            if image_info_list:
                for info in image_info_list:
                    if hasattr(info, "filename") and hasattr(info, "analysis"):
                        analysis_lookup[info.filename] = info.analysis

            images_written = 0
            descriptions_written = 0
            for image in result.images:
                image_path = assets_dir / image.filename
                async with await anyio.open_file(image_path, "wb") as f:
                    await f.write(image.data)
                images_written += 1

                # Write image description .md file if enabled
                if self.generate_image_descriptions:
                    analysis = analysis_lookup.get(image.filename)
                    if analysis:
                        from datetime import UTC

                        md_content = self.generate_image_description_md(
                            image.filename, analysis, datetime.now(UTC)
                        )
                        md_path = assets_dir / f"{image.filename}.md"
                        async with await anyio.open_file(md_path, "w", encoding="utf-8") as f:
                            await f.write(md_content)
                        descriptions_written += 1

            # Log summary instead of individual files
            if images_written > 0:
                log.debug(
                    "Assets written",
                    images=images_written,
                    descriptions=descriptions_written,
                    output_dir=str(assets_dir),
                )

        return output_file

    def resolve_conflict(self, output_path: Path) -> Path:
        """Resolve output file conflicts based on settings.

        Args:
            output_path: Desired output path

        Returns:
            Resolved output path (may be renamed)

        Raises:
            ConversionError: If strategy is "skip" and file exists
        """
        if not output_path.exists():
            return output_path

        if self.on_conflict == "overwrite":
            return output_path
        elif self.on_conflict == "skip":
            raise ConversionError(
                output_path,
                f"Output file already exists: {output_path}",
            )
        elif self.on_conflict == "rename":
            counter = 1
            stem = output_path.stem
            suffix = output_path.suffix
            parent = output_path.parent

            while True:
                new_path = parent / f"{stem}_{counter}{suffix}"
                if not new_path.exists():
                    return new_path
                counter += 1
        else:
            return output_path

    def generate_image_description_md(
        self,
        filename: str,
        analysis: "ImageAnalysis",
        generated_at: datetime,
    ) -> str:
        """Generate markdown content for image description file.

        Args:
            filename: Image filename
            analysis: Image analysis result
            generated_at: Timestamp when the description was generated

        Returns:
            Markdown content for the image description file
        """
        lines = [
            "---",
            f"source_image: {filename}",
            f"image_type: {analysis.image_type}",
            f"generated_at: {generated_at.isoformat()}",
        ]

        # Add knowledge graph metadata if available
        if hasattr(analysis, "knowledge_meta") and analysis.knowledge_meta:
            km = analysis.knowledge_meta
            if hasattr(km, "entities") and km.entities:
                entities_str = ", ".join(km.entities)
                lines.append(f"entities: [{entities_str}]")
            if hasattr(km, "topics") and km.topics:
                topics_str = ", ".join(km.topics)
                lines.append(f"topics: [{topics_str}]")
            if hasattr(km, "domain") and km.domain:
                lines.append(f"domain: {km.domain}")

        lines.extend(
            [
                "---",
                "",
                "# Image Description",
                "",
                "## Alt Text",
                "",
                analysis.alt_text,
                "",
                "## Detailed Description",
                "",
                analysis.detailed_description,
            ]
        )

        # Add detected text if available
        if analysis.detected_text:
            lines.extend(
                [
                    "",
                    "## Detected Text",
                    "",
                    analysis.detected_text,
                ]
            )

        # Add knowledge graph section if available
        if hasattr(analysis, "knowledge_meta") and analysis.knowledge_meta:
            km = analysis.knowledge_meta
            if hasattr(km, "relationships") and km.relationships:
                lines.extend(
                    [
                        "",
                        "## Relationships",
                        "",
                    ]
                )
                for rel in km.relationships:
                    lines.append(f"- {rel}")

        return "\n".join(lines) + "\n"

    async def write_markdown_only(
        self,
        output_file: Path,
        markdown_content: str,
    ) -> Path:
        """Write only markdown content without handling images.

        A convenience method for simple markdown output.

        Args:
            output_file: Output file path
            markdown_content: Markdown content to write

        Returns:
            Path to the output file
        """
        import anyio

        # Handle conflicts
        output_file = self.resolve_conflict(output_file)

        # Ensure parent directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write content
        async with await anyio.open_file(output_file, "w", encoding="utf-8") as f:
            await f.write(markdown_content)

        log.info("Markdown written", path=str(output_file))
        return output_file
