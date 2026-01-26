"""Prompt management module."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from markitai.config import PromptsConfig


# Built-in prompts directory
BUILTIN_PROMPTS_DIR = Path(__file__).parent


class PromptManager:
    """Manager for loading and rendering prompts."""

    # Available prompt names
    PROMPT_NAMES = (
        "cleaner",
        "frontmatter",
        "image_caption",
        "image_description",
        "image_analysis",
        "page_content",
        "document_enhance",
        "document_process",
        "document_enhance_complete",
        "url_enhance",
    )

    def __init__(self, config: PromptsConfig | None = None) -> None:
        """
        Initialize prompt manager.

        Args:
            config: Optional prompts configuration
        """
        self.config = config
        self._cache: dict[str, str] = {}

    def get_prompt(self, name: str, **variables: str) -> str:
        """
        Get a prompt by name with variables substituted.

        Args:
            name: Prompt name (cleaner, frontmatter, image_caption, image_description)
            **variables: Variables to substitute in the template

        Returns:
            Rendered prompt string

        Raises:
            ValueError: If prompt name is not valid
        """
        if name not in self.PROMPT_NAMES:
            raise ValueError(
                f"Unknown prompt: {name}. Valid names: {', '.join(self.PROMPT_NAMES)}"
            )

        template = self._load_prompt(name)
        return self._render(template, **variables)

    def _load_prompt(self, name: str) -> str:
        """
        Load prompt template from file.

        Priority:
        1. Config-specified path
        2. Custom directory
        3. Built-in prompts
        """
        # Check cache first
        if name in self._cache:
            return self._cache[name]

        template = None

        # 1. Check config-specified path
        if self.config:
            config_path = getattr(self.config, name, None)
            if config_path:
                path = Path(config_path).expanduser()
                if path.exists():
                    template = path.read_text(encoding="utf-8")

        # 2. Check custom directory
        if template is None and self.config:
            custom_dir = Path(self.config.dir).expanduser()
            custom_path = custom_dir / f"{name}.md"
            if custom_path.exists():
                template = custom_path.read_text(encoding="utf-8")

        # 3. Fall back to built-in
        if template is None:
            builtin_path = BUILTIN_PROMPTS_DIR / f"{name}.md"
            if builtin_path.exists():
                template = builtin_path.read_text(encoding="utf-8")
            else:
                raise FileNotFoundError(f"Built-in prompt not found: {name}")

        # Cache the template
        self._cache[name] = template
        return template

    def _render(self, template: str, **variables: str) -> str:
        """
        Render a template with variables.

        Uses simple {variable} substitution.
        """
        result = template

        # Add default variables
        if "timestamp" not in variables:
            variables["timestamp"] = datetime.now().astimezone().isoformat()

        for key, value in variables.items():
            result = result.replace(f"{{{key}}}", str(value))

        return result

    def clear_cache(self) -> None:
        """Clear the prompt cache."""
        self._cache.clear()

    def list_prompts(self) -> dict[str, str]:
        """
        List all available prompts with their sources.

        Returns:
            Dict mapping prompt names to their source paths
        """
        result = {}

        for name in self.PROMPT_NAMES:
            source = "built-in"

            # Check config-specified path
            if self.config:
                config_path = getattr(self.config, name, None)
                if config_path:
                    path = Path(config_path).expanduser()
                    if path.exists():
                        source = str(path)
                        result[name] = source
                        continue

                # Check custom directory
                custom_dir = Path(self.config.dir).expanduser()
                custom_path = custom_dir / f"{name}.md"
                if custom_path.exists():
                    source = str(custom_path)

            result[name] = source

        return result
