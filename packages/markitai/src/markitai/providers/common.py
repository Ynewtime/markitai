"""Shared utilities for custom LLM providers.

Common functions used by claude_agent and copilot providers, extracted
to avoid code duplication.
"""

from __future__ import annotations

from typing import Any


def has_images(messages: list[Any]) -> bool:
    """Check if messages contain any image content.

    Args:
        messages: OpenAI-style message list

    Returns:
        True if any message contains image content
    """
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Convert OpenAI-style messages to a single prompt string (text only).

    Args:
        messages: List of message dicts with 'role' and 'content' keys

    Returns:
        Combined prompt string
    """
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        # Handle content that's a list (multimodal messages) - extract text only
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            content = "\n".join(text_parts)

        if role == "system":
            parts.append(f"<system>\n{content}\n</system>")
        elif role == "assistant":
            parts.append(f"<assistant>\n{content}\n</assistant>")
        else:
            # user or other roles
            parts.append(content)

    return "\n\n".join(parts)


UNSUPPORTED_PARAMS = frozenset(
    {
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "presence_penalty",
        "frequency_penalty",
        "logit_bias",
        "seed",
        "n",
    }
)
