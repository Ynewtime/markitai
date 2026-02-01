"""Claude Agent SDK provider for LiteLLM.

This provider uses the Claude Agent SDK to make LLM requests through
the Claude Code CLI authentication, allowing users to use their Claude
subscription directly.

Usage:
    In configuration, use "claude-agent/<model>" as the model name:

    {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "claude-agent/sonnet"
                    }
                }
            ]
        }
    }

Supported models:
    - Aliases (recommended): sonnet, opus, haiku, inherit
      (automatically resolves to latest version via LiteLLM database)
    - Full model strings: claude-sonnet-4-5-20250929, claude-opus-4-5-20251101, etc.

Supported API providers (via environment variables):
    - Anthropic API (default)
    - Amazon Bedrock: CLAUDE_CODE_USE_BEDROCK=1
    - Google Vertex AI: CLAUDE_CODE_USE_VERTEX=1
    - Microsoft Foundry: CLAUDE_CODE_USE_FOUNDRY=1

Requirements:
    - claude-agent-sdk package: uv add claude-agent-sdk
    - Claude Code CLI installed and authenticated: https://docs.anthropic.com/claude-code

Limitations:
    - Does not support streaming responses (in this implementation)
    - Requires Claude Code CLI to be installed and authenticated
    - Image size limit: 5MB (API limitation)
"""

from __future__ import annotations

import asyncio
import importlib.util
import time
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.providers.timeout import calculate_timeout_from_messages

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse

try:
    import litellm
    from litellm.exceptions import AuthenticationError, RateLimitError
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.utils import Usage

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    CustomLLM = object  # type: ignore[misc, assignment]
    AuthenticationError = Exception  # type: ignore[misc, assignment]
    RateLimitError = Exception  # type: ignore[misc, assignment]


def _is_claude_agent_sdk_available() -> bool:
    """Check if Claude Agent SDK is available using importlib.

    This is the correct way to check for optional dependencies without
    actually importing them or triggering side effects.

    Returns:
        True if claude-agent-sdk is installed, False otherwise
    """
    return importlib.util.find_spec("claude_agent_sdk") is not None


class ClaudeAgentProvider(CustomLLM):  # type: ignore[misc]
    """Custom LiteLLM provider using Claude Agent SDK.

    This provider enables using Claude through the Claude Code CLI
    authentication, which uses subscription credits rather than API credits.

    Supports multimodal input (text and images) via streaming input.
    """

    def __init__(self, timeout: int = 120) -> None:
        """Initialize the provider.

        Args:
            timeout: Request timeout in seconds
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM not available. Install with: uv add litellm")
        self.timeout = timeout

    def _has_images(self, messages: list[dict[str, Any]]) -> bool:
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

    def _calculate_adaptive_timeout(self, messages: list[dict[str, Any]]) -> int:
        """Calculate adaptive timeout based on message content.

        Uses the timeout module to estimate appropriate timeout based on
        prompt length, image count, and other factors.

        Args:
            messages: OpenAI-style messages

        Returns:
            Timeout in seconds
        """
        return calculate_timeout_from_messages(messages)

    def _messages_to_prompt(self, messages: list[dict[str, Any]]) -> str:
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

    def _convert_content_to_sdk_format(
        self, content: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert OpenAI-style content to Claude Agent SDK format.

        Args:
            content: OpenAI-style content (string or list of content blocks)

        Returns:
            List of content blocks in Claude SDK format
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]

        sdk_content: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                continue

            part_type = part.get("type", "")
            if part_type == "text":
                sdk_content.append({"type": "text", "text": part.get("text", "")})
            elif part_type == "image_url":
                # Convert OpenAI image_url format to Claude SDK format
                image_url = part.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:"):
                    # Parse data URL: data:image/jpeg;base64,<data>
                    try:
                        header, data = url.split(",", 1)
                        media_type = header.split(":")[1].split(";")[0]
                        sdk_content.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": data,
                                },
                            }
                        )
                    except (ValueError, IndexError):
                        logger.warning("[ClaudeAgent] Invalid image data URL format")
                else:
                    # URL-based image - not directly supported, log warning
                    logger.warning(
                        "[ClaudeAgent] URL-based images not supported, only base64"
                    )

        return sdk_content

    async def _messages_to_stream(
        self, messages: list[dict[str, Any]]
    ) -> AsyncIterator[dict[str, Any]]:
        """Convert OpenAI-style messages to streaming input for Claude Agent SDK.

        According to the SDK documentation, streaming input should yield complete
        user message objects with content in Claude SDK format.

        Args:
            messages: OpenAI-style message list with potential image content

        Yields:
            Message dicts compatible with Claude Agent SDK streaming input
        """
        # Combine all messages into one user message for SDK
        all_content: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Convert content to SDK format
            sdk_content = self._convert_content_to_sdk_format(content)

            # For system/assistant messages, wrap in role tags as text
            if role == "system":
                text_content = "\n".join(
                    block.get("text", "")
                    for block in sdk_content
                    if block.get("type") == "text"
                )
                all_content.append(
                    {"type": "text", "text": f"<system>\n{text_content}\n</system>\n\n"}
                )
            elif role == "assistant":
                text_content = "\n".join(
                    block.get("text", "")
                    for block in sdk_content
                    if block.get("type") == "text"
                )
                all_content.append(
                    {
                        "type": "text",
                        "text": f"<assistant>\n{text_content}\n</assistant>\n\n",
                    }
                )
            else:
                # User messages - include all content (text and images)
                all_content.extend(sdk_content)

        # Yield as a single user message per SDK docs
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": all_content,
            },
        }

    def _extract_usage(self, message: Any, usage_info: dict[str, Any]) -> None:
        """Extract usage information from a ResultMessage.

        Args:
            message: ResultMessage from Claude Agent SDK
            usage_info: Dict to populate with usage data
        """
        if hasattr(message, "usage") and message.usage:
            usage_dict = message.usage
            if isinstance(usage_dict, dict):
                usage_info.update(usage_dict)
            else:
                # Handle object-style usage
                usage_info.update(
                    {
                        "input_tokens": getattr(usage_dict, "input_tokens", 0),
                        "output_tokens": getattr(usage_dict, "output_tokens", 0),
                        "cache_read_input_tokens": getattr(
                            usage_dict, "cache_read_input_tokens", 0
                        ),
                        "cache_creation_input_tokens": getattr(
                            usage_dict, "cache_creation_input_tokens", 0
                        ),
                    }
                )

    def _convert_response_format(
        self, response_format: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Convert OpenAI-style response_format to Claude Agent SDK output_format.

        Supports:
        - {"type": "json_object"} -> basic JSON mode
        - {"type": "json_schema", "json_schema": {"schema": {...}}} -> structured output

        Args:
            response_format: OpenAI/LiteLLM response_format parameter

        Returns:
            Claude Agent SDK output_format dict or None
        """
        if not response_format:
            return None

        format_type = response_format.get("type")

        if format_type == "json_schema":
            # Full structured output with schema
            json_schema = response_format.get("json_schema", {})
            schema = json_schema.get("schema")
            if schema:
                return {"type": "json_schema", "schema": schema}

        if format_type == "json_object":
            # Basic JSON mode - use a permissive schema
            return {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                },
            }

        return None

    # Parameters that are not supported by Claude Agent SDK
    # These are silently ignored with DEBUG logging
    _UNSUPPORTED_PARAMS = frozenset(
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

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using Claude Agent SDK.

        Supports multimodal input (text and images) and structured outputs.

        Args:
            model: Model identifier (e.g., "claude-agent/sonnet")
            messages: OpenAI-style message list (supports multimodal content)
            **kwargs: Additional parameters:
                - response_format: OpenAI-style response format for JSON/structured output
                - Other LLM params (max_tokens, temperature, etc.) are ignored
                  as Claude Agent SDK manages these internally.

        Returns:
            LiteLLM ModelResponse

        Raises:
            RuntimeError: If SDK is not available or authentication fails
        """
        # Log ignored parameters at DEBUG level
        ignored_params = [k for k in kwargs if k in self._UNSUPPORTED_PARAMS]
        if ignored_params:
            logger.debug(f"[ClaudeAgent] Ignoring unsupported params: {ignored_params}")

        if not _is_claude_agent_sdk_available():
            raise RuntimeError(
                "Claude Agent SDK not installed. Install with: uv add claude-agent-sdk"
            )

        # Import SDK components only when needed (lazy import for optional dependency)
        # These imports are guarded by the availability check above
        import claude_agent_sdk  # type: ignore[import-not-found]
        import claude_agent_sdk.types  # type: ignore[import-not-found]

        # Extract model name from provider prefix
        model_name = model.replace("claude-agent/", "")

        # Check if messages contain images
        has_images = self._has_images(messages)

        if has_images:
            logger.debug("[ClaudeAgent] Using streaming input for multimodal content")
            # Use streaming input for multimodal messages
            prompt: str | AsyncIterator[dict[str, Any]] = self._messages_to_stream(
                messages
            )
        else:
            prompt = self._messages_to_prompt(messages)
            logger.debug(
                f"[ClaudeAgent] Calling model={model_name}, prompt_length={len(prompt)}"
            )

        # Convert response_format to output_format for structured outputs
        response_format = kwargs.get("response_format")
        output_format = self._convert_response_format(response_format)

        # Calculate adaptive timeout based on message content
        timeout = self._calculate_adaptive_timeout(messages)
        logger.debug(f"[ClaudeAgent] Using adaptive timeout: {timeout}s")

        start_time = time.time()
        result_text = ""
        structured_output: dict[str, Any] | None = None
        usage_info: dict[str, Any] = {}
        total_cost_usd: float = 0.0

        try:
            # Build options dict
            options_kwargs: dict[str, Any] = {
                "allowed_tools": [],
                "permission_mode": "bypassPermissions",
                "max_turns": 1,
                "model": model_name,
                "timeout": timeout,
            }

            # Add output_format if structured output is requested
            if output_format:
                options_kwargs["output_format"] = output_format
                logger.debug("[ClaudeAgent] Using structured output mode")

            options = claude_agent_sdk.ClaudeAgentOptions(**options_kwargs)

            # Use ClaudeSDKClient for all requests (more reliable than query())
            # For images, use streaming input; for text, pass string directly
            # See: docs/reference/claude_streaming_vs_single_mode.md
            async with claude_agent_sdk.ClaudeSDKClient(options) as client:
                await client.query(prompt)
                async for message in client.receive_response():
                    if isinstance(message, claude_agent_sdk.types.AssistantMessage):
                        for block in message.content:
                            if isinstance(block, claude_agent_sdk.types.TextBlock):
                                result_text += block.text
                    elif isinstance(message, claude_agent_sdk.types.ResultMessage):
                        self._extract_usage(message, usage_info)
                        if hasattr(message, "total_cost_usd"):
                            total_cost_usd = (
                                getattr(message, "total_cost_usd", 0.0) or 0.0
                            )
                        if (
                            hasattr(message, "structured_output")
                            and message.structured_output
                        ):
                            structured_output = message.structured_output

        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            logger.error(f"[ClaudeAgent] Error: {error_msg}")

            # Check for quota/billing errors (should NOT be retried)
            quota_patterns = [
                "quota",
                "billing",
                "payment",
                "subscription",
                "402",
                "insufficient",
                "credit",
            ]
            if any(pattern in error_msg_lower for pattern in quota_patterns):
                raise AuthenticationError(
                    f"Claude Agent quota/billing error: {error_msg}",
                    "claude-agent",
                    model_name,
                )

            # Check for authentication errors (should NOT be retried)
            auth_patterns = [
                "not authenticated",
                "authentication",
                "unauthorized",
                "401",
                "api key",
            ]
            if any(pattern in error_msg_lower for pattern in auth_patterns):
                raise AuthenticationError(
                    f"Claude Agent authentication error: {error_msg}",
                    "claude-agent",
                    model_name,
                )

            # Rate limit errors (LiteLLM will handle retry)
            if "rate limit" in error_msg_lower or "429" in error_msg_lower:
                raise RateLimitError(
                    f"Claude Agent rate limit: {error_msg}",
                    "claude-agent",
                    model_name,
                )

            raise RuntimeError(f"Claude Agent SDK error: {e}")

        elapsed = time.time() - start_time

        cost_str = f", cost=${total_cost_usd:.4f}" if total_cost_usd > 0 else ""
        logger.debug(
            f"[ClaudeAgent] Completed in {elapsed:.2f}s, "
            f"response_length={len(result_text)}{cost_str}"
        )

        # Build usage object
        input_tokens = usage_info.get("input_tokens", 0)
        output_tokens = usage_info.get("output_tokens", 0)

        # Determine content for response
        # If structured output is available, serialize it as JSON string
        if structured_output is not None:
            import json

            response_content = json.dumps(structured_output, ensure_ascii=False)
        else:
            response_content = result_text

        response = litellm.ModelResponse(
            id=f"claude-agent-{uuid.uuid4().hex[:12]}",
            choices=[
                {
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            model=model,  # Keep full model ID with prefix for llm_usage tracking
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

        # Store cost and structured output in hidden params for downstream retrieval
        # The SDK returns total_cost_usd based on actual API pricing
        response._hidden_params = {
            "total_cost_usd": total_cost_usd,
            "structured_output": structured_output,
        }

        return response

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Sync completion wrapper.

        Note: Prefer using acompletion() in async contexts for better performance.
        This method is provided for compatibility with sync code only.

        Args:
            model: Model identifier
            messages: OpenAI-style message list
            **kwargs: Additional parameters

        Returns:
            LiteLLM ModelResponse

        Raises:
            RuntimeError: If called from within a running event loop
        """
        # Check if we're already in an async context
        try:
            asyncio.get_running_loop()
            # If we get here, there's a running loop - can't use run_until_complete
            raise RuntimeError(
                "Cannot call sync completion() from within an async context. "
                "Please use acompletion() instead."
            )
        except RuntimeError as e:
            # "no running event loop" means we can safely use asyncio.run()
            if "no running event loop" not in str(e):
                raise

        # Use asyncio.run() which properly creates and cleans up the event loop
        return asyncio.run(self.acompletion(model, messages, **kwargs))
