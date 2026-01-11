"""YAML Frontmatter handling for Markdown documents.

Provides utilities for adding, parsing, and modifying YAML frontmatter
in Markdown documents.
"""

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from markit.utils.logging import get_logger

log = get_logger(__name__)


# Regex pattern for detecting frontmatter
FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n",
    re.DOTALL,
)


@dataclass
class Frontmatter:
    """Represents YAML frontmatter data."""

    title: str | None = None
    processed: str | None = None
    description: str | None = None
    source: str | None = None
    # Knowledge graph metadata fields
    entities: list[str] | None = None
    topics: list[str] | None = None
    domain: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        result = {}

        if self.title:
            result["title"] = self.title
        if self.processed:
            result["processed"] = self.processed
        if self.description:
            result["description"] = self.description
        if self.source:
            result["source"] = self.source
        # Knowledge graph fields
        if self.entities:
            result["entities"] = self.entities
        if self.topics:
            result["topics"] = self.topics
        if self.domain:
            result["domain"] = self.domain

        # Add extra fields
        result.update(self.extra)

        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Frontmatter":
        """Create from dictionary."""
        known_fields = {
            "title",
            "processed",
            "description",
            "source",
            "entities",
            "topics",
            "domain",
        }
        extra = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            title=data.get("title"),
            processed=data.get("processed"),
            description=data.get("description"),
            source=data.get("source"),
            entities=data.get("entities"),
            topics=data.get("topics"),
            domain=data.get("domain"),
            extra=extra,
        )

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        data = self.to_dict()
        return yaml.dump(
            data,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )


class FrontmatterHandler:
    """Handle YAML frontmatter in Markdown documents."""

    def __init__(self):
        """Initialize the handler."""
        pass

    def has_frontmatter(self, markdown: str) -> bool:
        """Check if markdown has frontmatter.

        Args:
            markdown: Markdown content

        Returns:
            True if frontmatter exists
        """
        return bool(FRONTMATTER_PATTERN.match(markdown))

    def parse(self, markdown: str) -> tuple[Frontmatter | None, str]:
        """Parse frontmatter from markdown.

        Args:
            markdown: Markdown content

        Returns:
            Tuple of (Frontmatter or None, content without frontmatter)
        """
        match = FRONTMATTER_PATTERN.match(markdown)

        if not match:
            return None, markdown

        yaml_content = match.group(1)
        content = markdown[match.end() :]

        try:
            data = yaml.safe_load(yaml_content)
            if data is None:
                data = {}
            frontmatter = Frontmatter.from_dict(data)
            return frontmatter, content
        except yaml.YAMLError as e:
            log.warning("Failed to parse frontmatter", error=str(e))
            return None, markdown

    def extract(self, markdown: str) -> dict[str, Any]:
        """Extract frontmatter as dictionary.

        Args:
            markdown: Markdown content

        Returns:
            Frontmatter data as dictionary (empty if no frontmatter)
        """
        frontmatter, _ = self.parse(markdown)
        if frontmatter:
            return frontmatter.to_dict()
        return {}

    def remove(self, markdown: str) -> str:
        """Remove frontmatter from markdown.

        Args:
            markdown: Markdown content

        Returns:
            Markdown content without frontmatter
        """
        _, content = self.parse(markdown)
        return content.lstrip()

    def add(
        self,
        markdown: str,
        frontmatter: Frontmatter | dict[str, Any],
        replace: bool = True,
    ) -> str:
        """Add frontmatter to markdown.

        Args:
            markdown: Markdown content
            frontmatter: Frontmatter to add
            replace: Replace existing frontmatter if present

        Returns:
            Markdown with frontmatter
        """
        # Convert dict to Frontmatter if needed
        if isinstance(frontmatter, dict):
            frontmatter = Frontmatter.from_dict(frontmatter)

        # Remove existing frontmatter if replacing
        if replace and self.has_frontmatter(markdown):
            _, content = self.parse(markdown)
        else:
            content = markdown

        # Build frontmatter block
        yaml_str = frontmatter.to_yaml()
        frontmatter_block = f"---\n{yaml_str}---\n\n"

        return frontmatter_block + content.lstrip()

    def update(
        self,
        markdown: str,
        updates: dict[str, Any],
    ) -> str:
        """Update frontmatter fields.

        Args:
            markdown: Markdown content
            updates: Fields to update

        Returns:
            Markdown with updated frontmatter
        """
        existing, content = self.parse(markdown)

        if existing:
            # Update existing frontmatter
            data = existing.to_dict()
            data.update(updates)
            frontmatter = Frontmatter.from_dict(data)
        else:
            # Create new frontmatter
            frontmatter = Frontmatter.from_dict(updates)

        return self.add(content, frontmatter, replace=False)


def create_frontmatter(
    source_file: Path,
    summary: str | None = None,
    extra: dict[str, Any] | None = None,
) -> Frontmatter:
    """Create standard frontmatter for a converted document.

    Args:
        source_file: Original source file
        summary: Document summary (optional)
        extra: Additional frontmatter fields

    Returns:
        Frontmatter object
    """
    return Frontmatter(
        title=_escape_yaml_string(source_file.stem),
        processed=datetime.now().strftime("%Y-%m-%d"),
        description=_escape_yaml_string(summary) if summary else None,
        source=source_file.name,
        extra=extra or {},
    )


def inject_frontmatter(
    markdown: str,
    source_file: Path,
    summary: str | None = None,
) -> str:
    """Inject standard frontmatter into markdown.

    Args:
        markdown: Markdown content
        source_file: Original source file
        summary: Document summary (optional)

    Returns:
        Markdown with frontmatter
    """
    handler = FrontmatterHandler()
    frontmatter = create_frontmatter(source_file, summary)
    return handler.add(markdown, frontmatter)


def _escape_yaml_string(value: str | None) -> str | None:
    """Escape special characters in YAML string.

    Args:
        value: String to escape

    Returns:
        Escaped string or None
    """
    if value is None:
        return None

    # Replace problematic characters
    value = value.replace("\\", "\\\\")
    value = value.replace('"', '\\"')
    value = value.replace("\n", " ")
    value = value.replace("\r", "")

    return value


@dataclass
class KnowledgeGraphData:
    """Knowledge graph metadata for image descriptions."""

    entities: list[str] | None = None
    relationships: list[str] | None = None
    topics: list[str] | None = None
    domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        if self.entities:
            result["entities"] = self.entities
        if self.relationships:
            result["relationships"] = self.relationships
        if self.topics:
            result["topics"] = self.topics
        if self.domain:
            result["domain"] = self.domain
        return result


class ImageDescriptionFrontmatter:
    """Handle frontmatter for image description files."""

    def create(
        self,
        image_filename: str,
        image_type: str,
        alt_text: str,
        detailed_description: str,
        detected_text: str | None = None,
        knowledge_graph: KnowledgeGraphData | None = None,
    ) -> str:
        """Create frontmatter for image description markdown.

        Args:
            image_filename: Name of the image file
            image_type: Type of image (diagram, photo, etc.)
            alt_text: Short alt text
            detailed_description: Detailed description
            detected_text: OCR-detected text
            knowledge_graph: Knowledge graph metadata

        Returns:
            Complete markdown document for image description
        """
        frontmatter_data: dict[str, Any] = {
            "source_image": image_filename,
            "image_type": image_type,
            "generated_at": datetime.now().strftime("%Y-%m-%d"),
        }

        # Add knowledge graph metadata to frontmatter
        if knowledge_graph:
            kg_dict = knowledge_graph.to_dict()
            frontmatter_data.update(kg_dict)

        yaml_str = yaml.dump(
            frontmatter_data,
            default_flow_style=False,
            allow_unicode=True,
        )

        content_parts = [
            f"---\n{yaml_str}---\n",
            "\n# Image Description\n",
            "\n## Short Description\n",
            f"\n{alt_text}\n",
            "\n## Detailed Description\n",
            f"\n{detailed_description}\n",
        ]

        if detected_text:
            content_parts.extend(
                [
                    "\n## Detected Text\n",
                    f"\n{detected_text}\n",
                ]
            )

        # Add knowledge graph section if available
        if knowledge_graph and knowledge_graph.relationships:
            content_parts.extend(
                [
                    "\n## Relationships\n",
                    "\n",
                ]
            )
            for rel in knowledge_graph.relationships:
                content_parts.append(f"- {rel}\n")

        return "".join(content_parts)
