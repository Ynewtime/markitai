"""Unit tests for adaptive timeout calculation.

The timeout calculator provides intelligent timeout estimation based on
request complexity, including prompt length, expected output, and image presence.
"""

from __future__ import annotations

import pytest


class TestTimeoutConfig:
    """Tests for TimeoutConfig dataclass."""

    def test_timeout_config_defaults(self) -> None:
        """Test that TimeoutConfig has correct default values."""
        from markitai.providers.timeout import TimeoutConfig

        config = TimeoutConfig()
        assert config.base_timeout == 60
        assert config.max_timeout == 600
        assert config.chars_per_second == 500.0
        assert config.image_multiplier == 1.5
        assert config.per_image_seconds == 10.0

    def test_timeout_config_custom_values(self) -> None:
        """Test that TimeoutConfig accepts custom values."""
        from markitai.providers.timeout import TimeoutConfig

        config = TimeoutConfig(
            base_timeout=30,
            max_timeout=300,
            chars_per_second=1000.0,
            image_multiplier=2.0,
            per_image_seconds=15.0,
        )
        assert config.base_timeout == 30
        assert config.max_timeout == 300
        assert config.chars_per_second == 1000.0
        assert config.image_multiplier == 2.0
        assert config.per_image_seconds == 15.0

    def test_timeout_config_is_frozen(self) -> None:
        """Test that TimeoutConfig is immutable (frozen)."""
        from markitai.providers.timeout import TimeoutConfig

        config = TimeoutConfig()
        with pytest.raises(AttributeError):
            config.base_timeout = 100  # type: ignore[misc]

    def test_timeout_config_uses_slots(self) -> None:
        """Test that TimeoutConfig uses __slots__ for memory efficiency."""
        from markitai.providers.timeout import TimeoutConfig

        config = TimeoutConfig()
        assert hasattr(config, "__slots__") or not hasattr(config, "__dict__")


class TestCalculateTimeout:
    """Tests for calculate_timeout function."""

    def test_base_timeout_for_short_text(self) -> None:
        """Test that short text returns base timeout."""
        from markitai.providers.timeout import calculate_timeout

        # Very short prompt should return base timeout (60s)
        timeout = calculate_timeout(prompt_length=100)
        assert timeout == 60

    def test_timeout_scales_with_prompt_length(self) -> None:
        """Test that timeout increases with prompt length."""
        from markitai.providers.timeout import calculate_timeout

        short_timeout = calculate_timeout(prompt_length=1000)
        long_timeout = calculate_timeout(prompt_length=50000)

        assert long_timeout > short_timeout

    def test_timeout_increases_for_images(self) -> None:
        """Test that image requests get 1.5x multiplier."""
        from markitai.providers.timeout import calculate_timeout

        text_timeout = calculate_timeout(prompt_length=10000)
        image_timeout = calculate_timeout(prompt_length=10000, has_images=True)

        # Image timeout should be approximately 1.5x text timeout
        assert image_timeout > text_timeout
        # Check the multiplier is applied (allowing for base_timeout floor)
        assert image_timeout >= int(text_timeout * 1.5) - 1

    def test_timeout_increases_for_multiple_images(self) -> None:
        """Test that multiple images add additional seconds."""
        from markitai.providers.timeout import calculate_timeout

        one_image = calculate_timeout(
            prompt_length=10000, has_images=True, image_count=1
        )
        five_images = calculate_timeout(
            prompt_length=10000, has_images=True, image_count=5
        )

        # 4 additional images * 10s per image = 40s more
        assert five_images > one_image
        assert five_images >= one_image + 40

    def test_timeout_caps_at_maximum(self) -> None:
        """Test that timeout never exceeds max_timeout."""
        from markitai.providers.timeout import calculate_timeout

        # Very long prompt that would exceed max
        timeout = calculate_timeout(
            prompt_length=1_000_000,
            has_images=True,
            image_count=100,
            expected_output_tokens=100_000,
        )
        assert timeout == 600  # Default max_timeout

    def test_timeout_respects_minimum(self) -> None:
        """Test that timeout never goes below base_timeout."""
        from markitai.providers.timeout import calculate_timeout

        # Tiny request should still get base timeout
        timeout = calculate_timeout(prompt_length=0)
        assert timeout == 60  # Default base_timeout

    def test_timeout_with_expected_output_tokens(self) -> None:
        """Test that expected_output_tokens adds to timeout."""
        from markitai.providers.timeout import calculate_timeout

        without_output = calculate_timeout(prompt_length=1000)
        with_output = calculate_timeout(prompt_length=1000, expected_output_tokens=4000)

        # 4000 tokens / 4.0 = 1000s additional (clamped to max)
        assert with_output > without_output

    def test_timeout_with_custom_config(self) -> None:
        """Test that custom TimeoutConfig is respected."""
        from markitai.providers.timeout import TimeoutConfig, calculate_timeout

        config = TimeoutConfig(
            base_timeout=30,
            max_timeout=120,
            chars_per_second=250.0,
            image_multiplier=2.0,
            per_image_seconds=20.0,
        )

        timeout = calculate_timeout(prompt_length=5000, config=config)
        # 30 (base) + 5000/250 (20s) = 50s
        assert timeout >= 30
        assert timeout <= 120

    def test_timeout_returns_integer(self) -> None:
        """Test that timeout is always an integer."""
        from markitai.providers.timeout import calculate_timeout

        timeout = calculate_timeout(prompt_length=1234)
        assert isinstance(timeout, int)


class TestCalculateTimeoutFromMessages:
    """Tests for calculate_timeout_from_messages function."""

    def test_text_only_messages(self) -> None:
        """Test timeout calculation with text-only messages."""
        from markitai.providers.timeout import calculate_timeout_from_messages

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        timeout = calculate_timeout_from_messages(messages)
        assert timeout >= 60  # At least base timeout

    def test_multimodal_messages_with_images(self) -> None:
        """Test timeout calculation with image content."""
        from markitai.providers.timeout import calculate_timeout_from_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,abc123"},
                    },
                ],
            }
        ]

        timeout = calculate_timeout_from_messages(messages)
        # Should detect the image and apply multiplier
        assert timeout >= 60

    def test_multimodal_messages_with_multiple_images(self) -> None:
        """Test timeout calculation with multiple images."""
        from markitai.providers.timeout import calculate_timeout_from_messages

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/1.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/2.png"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/3.png"},
                    },
                ],
            }
        ]

        single_image_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/1.png"},
                    },
                ],
            }
        ]

        multi_timeout = calculate_timeout_from_messages(messages)
        single_timeout = calculate_timeout_from_messages(single_image_messages)

        # Multiple images should have higher timeout
        assert multi_timeout >= single_timeout

    def test_empty_messages(self) -> None:
        """Test timeout calculation with empty message list."""
        from markitai.providers.timeout import calculate_timeout_from_messages

        timeout = calculate_timeout_from_messages([])
        assert timeout == 60  # Should return base timeout

    def test_custom_config_with_messages(self) -> None:
        """Test that custom config is passed through."""
        from markitai.providers.timeout import (
            TimeoutConfig,
            calculate_timeout_from_messages,
        )

        config = TimeoutConfig(base_timeout=45, max_timeout=180)
        messages = [{"role": "user", "content": "Hello"}]

        timeout = calculate_timeout_from_messages(messages, config=config)
        assert timeout >= 45
        assert timeout <= 180

    def test_long_conversation(self) -> None:
        """Test timeout scales with conversation length."""
        from markitai.providers.timeout import calculate_timeout_from_messages

        short_conversation = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        long_conversation = [
            {"role": "user", "content": "x" * 10000},
            {"role": "assistant", "content": "y" * 10000},
            {"role": "user", "content": "z" * 10000},
        ]

        short_timeout = calculate_timeout_from_messages(short_conversation)
        long_timeout = calculate_timeout_from_messages(long_conversation)

        assert long_timeout >= short_timeout

    def test_mixed_content_types(self) -> None:
        """Test messages with mixed string and list content."""
        from markitai.providers.timeout import calculate_timeout_from_messages

        messages = [
            {"role": "system", "content": "You are helpful."},  # String content
            {
                "role": "user",
                "content": [  # List content
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                ],
            },
        ]

        timeout = calculate_timeout_from_messages(messages)
        assert timeout >= 60
