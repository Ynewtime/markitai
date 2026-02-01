"""Adaptive timeout calculation for LLM requests.

This module provides intelligent timeout estimation based on request complexity,
including prompt length, expected output tokens, and image presence.

Example:
    >>> from markitai.providers.timeout import calculate_timeout
    >>> # Simple text request
    >>> timeout = calculate_timeout(prompt_length=1000)
    >>> # Multimodal request with images
    >>> timeout = calculate_timeout(prompt_length=5000, has_images=True, image_count=3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class TimeoutConfig:
    """Configuration for timeout calculation.

    Attributes:
        base_timeout: Minimum timeout in seconds (default: 60)
        max_timeout: Maximum timeout in seconds (default: 600, i.e., 10 minutes)
        chars_per_second: Estimated LLM processing speed (default: 500.0)
        image_multiplier: Timeout multiplier for image requests (default: 1.5)
        per_image_seconds: Additional seconds per image beyond the first (default: 10.0)
    """

    base_timeout: int = 60
    max_timeout: int = 600
    chars_per_second: float = 500.0
    image_multiplier: float = 1.5
    per_image_seconds: float = 10.0


def calculate_timeout(
    prompt_length: int,
    *,
    has_images: bool = False,
    image_count: int = 1,
    expected_output_tokens: int | None = None,
    config: TimeoutConfig | None = None,
) -> int:
    """Calculate adaptive timeout based on request complexity.

    The timeout is calculated using the following formula:
    1. Start with base_timeout
    2. Add prompt_length / chars_per_second for input processing
    3. Add expected_output_tokens / 4.0 if provided (generation time estimate)
    4. If has_images: multiply by image_multiplier
    5. Add per_image_seconds for each additional image (beyond the first)
    6. Clamp result to [base_timeout, max_timeout]

    Args:
        prompt_length: Total character count of the prompt/messages
        has_images: Whether the request contains image content
        image_count: Number of images in the request (default: 1)
        expected_output_tokens: Expected output token count (optional)
        config: Custom timeout configuration (uses defaults if None)

    Returns:
        Calculated timeout in seconds (integer)

    Example:
        >>> calculate_timeout(5000)  # Short text
        60
        >>> calculate_timeout(100000, has_images=True, image_count=5)
        290
    """
    if config is None:
        config = TimeoutConfig()

    # Start with base timeout
    timeout = float(config.base_timeout)

    # Add time for input processing
    timeout += prompt_length / config.chars_per_second

    # Add time for expected output generation
    if expected_output_tokens is not None:
        # Rough estimate: 4 chars per token, same processing speed
        timeout += expected_output_tokens / 4.0

    # Apply image multiplier
    if has_images:
        timeout *= config.image_multiplier

        # Add additional time for multiple images (beyond the first)
        if image_count > 1:
            additional_images = image_count - 1
            timeout += additional_images * config.per_image_seconds

    # Clamp to [base_timeout, max_timeout]
    timeout = max(config.base_timeout, min(config.max_timeout, timeout))

    return int(timeout)


def calculate_timeout_from_messages(
    messages: list[dict[str, Any]],
    *,
    config: TimeoutConfig | None = None,
) -> int:
    """Calculate timeout from OpenAI-style messages.

    Extracts prompt length and image count from the message list,
    then delegates to calculate_timeout.

    Supports both simple string content and multimodal content arrays
    (as used by OpenAI's Vision API and compatible providers).

    Args:
        messages: List of OpenAI-style message dicts with 'role' and 'content'
        config: Custom timeout configuration (uses defaults if None)

    Returns:
        Calculated timeout in seconds (integer)

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ... ]
        >>> calculate_timeout_from_messages(messages)
        60
    """
    total_length = 0
    image_count = 0
    has_images = False

    for message in messages:
        content = message.get("content")
        if content is None:
            continue

        if isinstance(content, str):
            # Simple text content
            total_length += len(content)
        elif isinstance(content, list):
            # Multimodal content array
            for item in content:
                if not isinstance(item, dict):
                    continue

                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text", "")
                    if isinstance(text, str):
                        total_length += len(text)
                elif item_type == "image_url":
                    has_images = True
                    image_count += 1

    return calculate_timeout(
        prompt_length=total_length,
        has_images=has_images,
        image_count=max(1, image_count),  # At least 1 if has_images
        config=config,
    )


__all__ = [
    "TimeoutConfig",
    "calculate_timeout",
    "calculate_timeout_from_messages",
]
