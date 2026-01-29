"""Tests for prompts module."""

from pathlib import Path

import pytest

from markitai.config import PromptsConfig
from markitai.prompts import BUILTIN_PROMPTS_DIR, PromptManager


class TestPromptManager:
    """Tests for PromptManager class."""

    def test_get_builtin_prompt(self) -> None:
        """Test getting a built-in prompt."""
        manager = PromptManager()
        prompt = manager.get_prompt("cleaner_system")

        assert "Markdown" in prompt

    def test_get_all_builtin_prompts(self) -> None:
        """Test that all built-in prompts exist."""
        manager = PromptManager()

        for name in PromptManager.PROMPT_NAMES:
            prompt = manager.get_prompt(name, content="test", source="test.txt")
            assert len(prompt) > 0

    def test_prompt_caching(self) -> None:
        """Test that prompts are cached."""
        manager = PromptManager()

        # First call loads from file
        prompt1 = manager.get_prompt("cleaner_system")
        assert "cleaner_system" in manager._cache

        # Second call uses cache
        prompt2 = manager.get_prompt("cleaner_system")
        assert prompt1 == prompt2

    def test_clear_cache(self) -> None:
        """Test clearing prompt cache."""
        manager = PromptManager()

        manager.get_prompt("cleaner_system")
        assert len(manager._cache) > 0

        manager.clear_cache()
        assert len(manager._cache) == 0

    def test_invalid_prompt_name(self) -> None:
        """Test error on invalid prompt name."""
        manager = PromptManager()

        with pytest.raises(ValueError, match="Unknown prompt"):
            manager.get_prompt("invalid_name")

    def test_variable_substitution(self) -> None:
        """Test variable substitution in prompts."""
        manager = PromptManager()

        prompt = manager.get_prompt(
            "cleaner_user",
            content="my content",
            source="test.docx",
        )

        assert "my content" in prompt

    def test_custom_prompt_from_config(self, tmp_path: Path) -> None:
        """Test loading custom prompt from config path."""
        # Create custom prompt
        custom_prompt = tmp_path / "my_cleaner_system.md"
        custom_prompt.write_text("Custom cleaner system prompt")

        config = PromptsConfig(cleaner_system=str(custom_prompt))
        manager = PromptManager(config)

        prompt = manager.get_prompt("cleaner_system")

        assert "Custom cleaner system prompt" in prompt

    def test_custom_prompt_from_directory(self, tmp_path: Path) -> None:
        """Test loading custom prompt from prompts directory."""
        # Create custom prompt in directory
        prompts_dir = tmp_path / "prompts"
        prompts_dir.mkdir()
        (prompts_dir / "cleaner_system.md").write_text("Dir cleaner system: {content}")

        config = PromptsConfig(dir=str(prompts_dir))
        manager = PromptManager(config)

        prompt = manager.get_prompt("cleaner_system", content="test")

        assert "Dir cleaner system" in prompt

    def test_list_prompts(self) -> None:
        """Test listing available prompts."""
        manager = PromptManager()
        prompts = manager.list_prompts()

        assert len(prompts) == len(PromptManager.PROMPT_NAMES)
        for name in PromptManager.PROMPT_NAMES:
            assert name in prompts
            assert prompts[name] == "built-in"


class TestBuiltinPrompts:
    """Tests for built-in prompt files."""

    def test_builtin_prompts_exist(self) -> None:
        """Test that all built-in prompt files exist."""
        for name in PromptManager.PROMPT_NAMES:
            path = BUILTIN_PROMPTS_DIR / f"{name}.md"
            assert path.exists(), f"Missing built-in prompt: {name}"

    def test_cleaner_system_prompt_content(self) -> None:
        """Test cleaner system prompt has required elements."""
        path = BUILTIN_PROMPTS_DIR / "cleaner_system.md"
        content = path.read_text(encoding="utf-8")

        assert "Markdown" in content

    def test_cleaner_user_prompt_content(self) -> None:
        """Test cleaner user prompt has content placeholder."""
        path = BUILTIN_PROMPTS_DIR / "cleaner_user.md"
        content = path.read_text(encoding="utf-8")

        assert "{content}" in content

    def test_image_caption_system_prompt_content(self) -> None:
        """Test image caption system prompt content."""
        path = BUILTIN_PROMPTS_DIR / "image_caption_system.md"
        content = path.read_text(encoding="utf-8")

        assert "alt" in content.lower() or "描述" in content

    def test_image_description_system_prompt_content(self) -> None:
        """Test image description system prompt content."""
        path = BUILTIN_PROMPTS_DIR / "image_description_system.md"
        content = path.read_text(encoding="utf-8")

        assert "描述" in content or "describe" in content.lower()
