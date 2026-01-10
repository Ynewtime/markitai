"""Tests for capabilities inference utility."""

import pytest

from markit.utils.capabilities import EMBEDDING_PATTERNS, VISION_PATTERNS, infer_capabilities


class TestInferCapabilities:
    """Tests for infer_capabilities function."""

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            # Vision-capable models
            ("gpt-4-vision-preview", ["text", "vision"]),
            ("gpt-4o", ["text", "vision"]),
            ("gpt-4o-mini", ["text", "vision"]),
            ("gpt-4.5", ["text", "vision"]),
            ("gpt-4.5-preview", ["text", "vision"]),
            ("gpt-5", ["text", "vision"]),
            ("gpt-5-turbo", ["text", "vision"]),
            ("claude-3-opus", ["text", "vision"]),
            ("claude-3-sonnet", ["text", "vision"]),
            ("claude-3-haiku", ["text", "vision"]),
            ("claude-sonnet-4", ["text", "vision"]),
            ("claude-opus-4", ["text", "vision"]),
            ("gemini-pro", ["text", "vision"]),
            ("gemini-1.5-pro", ["text", "vision"]),
            ("gemini-2.0-flash", ["text", "vision"]),
            ("llava:latest", ["text", "vision"]),
            ("bakllava:7b", ["text", "vision"]),
            ("yi-vl-34b", ["text", "vision"]),
            ("qwen-vl-max", ["text", "vision"]),
            ("some-vision-model", ["text", "vision"]),
            # Text-only models
            ("gpt-3.5-turbo", ["text"]),
            ("gpt-4-turbo", ["text"]),
            ("claude-2", ["text"]),
            ("llama3:70b", ["text"]),
            ("mistral:7b", ["text"]),
            ("deepseek-v2", ["text"]),
            ("qwen2:72b", ["text"]),
        ],
    )
    def test_vision_detection(self, model_id: str, expected: list[str]):
        """Test that vision capability is correctly detected."""
        result = infer_capabilities(model_id)
        assert sorted(result) == sorted(expected)

    @pytest.mark.parametrize(
        "model_id,expected",
        [
            # Embedding models
            ("text-embedding-ada-002", ["embedding"]),
            ("text-embedding-3-small", ["embedding"]),
            ("text-embedding-3-large", ["embedding"]),
            ("bge-large-en", ["embedding"]),
            ("bge-m3", ["embedding"]),
            ("e5-large-v2", ["embedding"]),
            ("e5-mistral-7b-instruct", ["embedding"]),
        ],
    )
    def test_embedding_detection(self, model_id: str, expected: list[str]):
        """Test that embedding models are correctly identified."""
        result = infer_capabilities(model_id)
        assert sorted(result) == sorted(expected)

    def test_case_insensitivity(self):
        """Test that model ID matching is case-insensitive."""
        assert infer_capabilities("GPT-4O") == infer_capabilities("gpt-4o")
        assert infer_capabilities("CLAUDE-3-OPUS") == infer_capabilities("claude-3-opus")
        assert infer_capabilities("TEXT-EMBEDDING-3-LARGE") == infer_capabilities(
            "text-embedding-3-large"
        )

    def test_unknown_model_defaults_to_text(self):
        """Test that unknown models default to text capability only."""
        assert infer_capabilities("unknown-model-xyz") == ["text"]
        assert infer_capabilities("some-random-llm") == ["text"]

    def test_returns_sorted_unique_list(self):
        """Test that result is sorted and deduplicated."""
        result = infer_capabilities("gpt-4o")
        assert result == sorted(result)
        assert len(result) == len(set(result))


class TestCapabilityPatterns:
    """Tests for capability pattern constants."""

    def test_vision_patterns_not_empty(self):
        """Test that vision patterns are defined."""
        assert len(VISION_PATTERNS) > 0

    def test_embedding_patterns_not_empty(self):
        """Test that embedding patterns are defined."""
        assert len(EMBEDDING_PATTERNS) > 0

    def test_vision_patterns_are_valid_regex(self):
        """Test that all vision patterns are valid regex."""
        import re

        for pattern in VISION_PATTERNS:
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern '{pattern}': {e}")

    def test_embedding_patterns_are_valid_regex(self):
        """Test that all embedding patterns are valid regex."""
        import re

        for pattern in EMBEDDING_PATTERNS:
            try:
                re.compile(pattern)
            except re.error as e:
                pytest.fail(f"Invalid regex pattern '{pattern}': {e}")
