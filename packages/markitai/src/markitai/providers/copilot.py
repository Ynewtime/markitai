"""GitHub Copilot SDK provider for LiteLLM.

This provider uses the GitHub Copilot SDK to make LLM requests through
the Copilot CLI, allowing users to use their GitHub Copilot subscription.

Usage:
    In configuration, use "copilot/<model>" as the model name:

    {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "copilot/gpt-4.1"
                    }
                }
            ]
        }
    }

Supported Models:
    - OpenAI: gpt-4.1 (GPT-5 series not yet supported, see below)
    - Anthropic: claude-sonnet-4, claude-sonnet-4.5, claude-opus-4.5, claude-haiku-4.5
    - Google: gemini-2.5-pro, gemini-3-flash, gemini-3-pro
    - Availability depends on your Copilot subscription
    - See docs/archive/deps/github-copilot-supported-models.md for full list

Known Limitations:
    - GPT-5 series models (gpt-5, gpt-5.1, gpt-5.2, etc.) are NOT supported yet.
      The Copilot SDK/CLI has a compatibility issue with OpenAI's reasoning models
      which require 'max_completion_tokens' instead of 'max_tokens'.
      Use non-GPT-5 models or direct OpenAI API for these models.

Requirements:
    - github-copilot-sdk package: pip install github-copilot-sdk
    - Copilot CLI installed and authenticated: https://docs.github.com/en/copilot

Error Handling:
    - FileNotFoundError: CLI not installed or not in PATH
    - ConnectionError: Cannot connect to Copilot server
    - TimeoutError: Request timeout (configurable via timeout parameter)
    - RuntimeError: Authentication failed, rate limit, or other SDK errors

Limitations:
    - Does not support streaming responses (in this implementation)
    - Requires Copilot CLI to be installed and authenticated
    - JSON mode is emulated via prompt engineering (SDK has no native support)
"""

from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import re
import tempfile
import time
import uuid
from typing import TYPE_CHECKING, Any

from loguru import logger

# PIL for image resizing (optional, graceful fallback)
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

if TYPE_CHECKING:
    from litellm.types.utils import ModelResponse

try:
    import litellm
    from litellm.llms.custom_llm import CustomLLM
    from litellm.types.utils import Usage

    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False
    CustomLLM = object  # type: ignore[misc, assignment]


def _is_copilot_sdk_available() -> bool:
    """Check if GitHub Copilot SDK is available using importlib.

    This is the correct way to check for optional dependencies without
    actually importing them or triggering side effects.

    Returns:
        True if github-copilot-sdk is installed, False otherwise
    """
    return importlib.util.find_spec("copilot") is not None


# Cache for resolved Copilot CLI path
_copilot_cli_cache: dict[str, str | None] = {}


def _find_copilot_cli() -> str | None:
    """Find the Copilot CLI executable path.

    On Windows, the CLI may be installed via npm/pnpm in locations not in
    the default subprocess PATH. This function searches common locations.

    Returns:
        Full path to copilot CLI, or None if not found
    """
    import shutil
    import sys
    from pathlib import Path

    cache_key = "copilot"
    if cache_key in _copilot_cli_cache:
        return _copilot_cli_cache[cache_key]

    # First, check if already in PATH
    cli_path = shutil.which("copilot")
    if cli_path:
        _copilot_cli_cache[cache_key] = cli_path
        logger.debug(f"[Copilot] Found CLI in PATH: {cli_path}")
        return cli_path

    # On Windows, search common npm/pnpm global locations
    if sys.platform == "win32":
        home = Path.home()
        search_paths = [
            # pnpm global
            home / "AppData" / "Local" / "pnpm" / "copilot.CMD",
            home / "AppData" / "Local" / "pnpm" / "copilot.cmd",
            # npm global (Roaming)
            home / "AppData" / "Roaming" / "npm" / "copilot.CMD",
            home / "AppData" / "Roaming" / "npm" / "copilot.cmd",
            # Scoop
            home / "scoop" / "shims" / "copilot.cmd",
            # Chocolatey
            Path("C:/ProgramData/chocolatey/bin/copilot.exe"),
            # WinGet
            home
            / "AppData"
            / "Local"
            / "Microsoft"
            / "WinGet"
            / "Packages"
            / "GitHub.Copilot_*"
            / "copilot.exe",
        ]

        for path in search_paths:
            # Handle glob patterns
            if "*" in str(path):
                matches = list(path.parent.glob(path.name))
                if matches:
                    for match in matches:
                        copilot_exe = match / "copilot.exe" if match.is_dir() else match
                        if copilot_exe.exists():
                            result = str(copilot_exe)
                            _copilot_cli_cache[cache_key] = result
                            logger.debug(f"[Copilot] Found CLI at: {result}")
                            return result
            elif path.exists():
                result = str(path)
                _copilot_cli_cache[cache_key] = result
                logger.debug(f"[Copilot] Found CLI at: {result}")
                return result

    _copilot_cli_cache[cache_key] = None
    logger.debug("[Copilot] CLI not found in common locations")
    return None


class CopilotProvider(CustomLLM):  # type: ignore[misc]
    """Custom LiteLLM provider using GitHub Copilot SDK.

    This provider enables using LLMs through the Copilot CLI
    authentication, which uses GitHub Copilot subscription credits.

    Supports multimodal input (text and images) via attachments.
    """

    # Copilot view tool has a pixel size limit for images (~2000x2000)
    # Images larger than this will fail to be processed
    MAX_IMAGE_DIMENSION = 2000

    # GPT-5 series models are NOT supported by Copilot SDK yet.
    # The SDK/CLI has a compatibility issue with OpenAI's reasoning models
    # which require 'max_completion_tokens' instead of 'max_tokens'.
    # This causes 400 errors from the API.
    UNSUPPORTED_MODELS = frozenset(
        {
            "gpt-5",
            "gpt-5.1",
            "gpt-5.2",
            "gpt-5-mini",
            "gpt-5.1-mini",
            "gpt-5.2-mini",
            "gpt-5-codex",
            "gpt-5.1-codex",
            "gpt-5.2-codex",
            "o1",
            "o1-mini",
            "o1-preview",
            "o3",
            "o3-mini",
        }
    )

    def __init__(self, timeout: int = 120) -> None:
        """Initialize the provider.

        Args:
            timeout: Request timeout in seconds
        """
        if not LITELLM_AVAILABLE:
            raise RuntimeError(
                "LiteLLM not available. Install with: pip install litellm"
            )
        self.timeout = timeout
        self._client: Any = None
        self._temp_files: list[str] = []

    def _resize_image_if_needed(self, image_path: str) -> str:
        """Resize image if it exceeds Copilot's dimension limit.

        Copilot's view tool has a ~2000x2000 pixel limit. Images larger
        than this will not be processed correctly.

        Args:
            image_path: Path to the image file

        Returns:
            Path to the (possibly resized) image file
        """
        if not PIL_AVAILABLE:
            logger.debug("[Copilot] PIL not available, skipping image resize check")
            return image_path

        try:
            with Image.open(image_path) as img:
                width, height = img.size
                max_dim = max(width, height)

                if max_dim <= self.MAX_IMAGE_DIMENSION:
                    return image_path

                # Calculate new dimensions maintaining aspect ratio
                scale = self.MAX_IMAGE_DIMENSION / max_dim
                new_width = int(width * scale)
                new_height = int(height * scale)

                logger.debug(
                    f"[Copilot] Resizing image from {width}x{height} to "
                    f"{new_width}x{new_height} (Copilot limit: {self.MAX_IMAGE_DIMENSION}px)"
                )

                # Resize and save to new temp file
                resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

                # Determine format from original
                img_format = img.format or "JPEG"
                ext = (
                    ".jpg" if img_format.upper() == "JPEG" else f".{img_format.lower()}"
                )

                fd, resized_path = tempfile.mkstemp(
                    suffix=ext, prefix="copilot_resized_"
                )
                self._temp_files.append(resized_path)
                os.close(fd)

                # Save with appropriate format
                if img_format.upper() == "JPEG":
                    resized.save(resized_path, format="JPEG", quality=85)
                else:
                    resized.save(resized_path, format=img_format)

                return resized_path

        except Exception as e:
            logger.warning(f"[Copilot] Failed to check/resize image: {e}")
            return image_path

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

    def _extract_images(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Extract text prompt and image attachments from messages.

        Args:
            messages: OpenAI-style message list with potential image content

        Returns:
            Tuple of (text_prompt, attachments_list)
        """
        text_parts = []
        attachments = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Add role prefix for non-user messages
            if role == "system":
                text_parts.append("<system>")
            elif role == "assistant":
                text_parts.append("<assistant>")

            # Handle multimodal content
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        part_type = part.get("type", "")
                        if part_type == "text":
                            text_parts.append(part.get("text", ""))
                        elif part_type == "image_url":
                            # Convert OpenAI image_url format to file attachment
                            image_url = part.get("image_url", {})
                            url = image_url.get("url", "")
                            if url.startswith("data:"):
                                # Parse data URL: data:image/jpeg;base64,<data>
                                try:
                                    header, data = url.split(",", 1)
                                    media_type = header.split(":")[1].split(";")[0]
                                    # Determine file extension
                                    ext_map = {
                                        "image/jpeg": ".jpg",
                                        "image/png": ".png",
                                        "image/gif": ".gif",
                                        "image/webp": ".webp",
                                    }
                                    ext = ext_map.get(media_type, ".jpg")

                                    # Save to temp file
                                    fd, temp_path = tempfile.mkstemp(
                                        suffix=ext, prefix="copilot_img_"
                                    )
                                    # Add to cleanup list immediately to prevent leaks
                                    # even if base64 decode fails
                                    self._temp_files.append(temp_path)
                                    try:
                                        os.write(fd, base64.b64decode(data))
                                    finally:
                                        os.close(fd)

                                    # Resize image if needed (Copilot has ~2000px limit)
                                    resized_path = self._resize_image_if_needed(
                                        temp_path
                                    )
                                    attachments.append(
                                        {"type": "file", "path": resized_path}
                                    )
                                except (ValueError, IndexError, OSError) as e:
                                    logger.warning(
                                        f"[Copilot] Invalid image data URL format: {e}"
                                    )
                            else:
                                # URL-based image - pass as-is if SDK supports it
                                attachments.append({"type": "url", "url": url})
            else:
                # Simple text content
                text_parts.append(str(content))

            # Close role tags
            if role == "system":
                text_parts.append("</system>")
            elif role == "assistant":
                text_parts.append("</assistant>")

        return "\n\n".join(text_parts), attachments

    def _cleanup_temp_files(self) -> None:
        """Clean up temporary image files."""
        for path in self._temp_files:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError as e:
                logger.warning(f"[Copilot] Failed to delete temp file {path}: {e}")
        self._temp_files.clear()

    def _build_json_prompt_suffix(
        self, response_format: dict[str, Any] | None
    ) -> str | None:
        """Build a prompt suffix to request JSON output.

        Since Copilot SDK doesn't support native JSON mode, we use prompt
        engineering to request JSON output format.

        Args:
            response_format: OpenAI-style response_format parameter

        Returns:
            Prompt suffix string or None if not JSON mode
        """
        if not response_format:
            return None

        format_type = response_format.get("type")

        if format_type == "json_schema":
            json_schema = response_format.get("json_schema", {})
            schema = json_schema.get("schema")
            if schema:
                schema_str = json.dumps(schema, indent=2, ensure_ascii=False)
                return (
                    "\n\n[IMPORTANT: You MUST respond with valid JSON only. "
                    "No markdown code blocks, no explanations, just pure JSON. "
                    f"The response must conform to this JSON Schema:\n{schema_str}]"
                )

        if format_type == "json_object":
            return (
                "\n\n[IMPORTANT: You MUST respond with valid JSON only. "
                "No markdown code blocks, no explanations, just pure JSON.]"
            )

        return None

    def _extract_json_from_response(self, text: str) -> str:
        """Extract JSON from response text.

        Handles cases where the model wraps JSON in markdown code blocks
        or includes extra text.

        Args:
            text: Raw response text

        Returns:
            Extracted JSON string (or original text if extraction fails)
        """
        # Try to extract from markdown code block first
        # Match ```json ... ``` or ``` ... ```
        code_block_match = re.search(
            r"```(?:json)?\s*\n?([\s\S]*?)\n?```", text, re.IGNORECASE
        )
        if code_block_match:
            extracted = code_block_match.group(1).strip()
            # Validate it's valid JSON
            try:
                json.loads(extracted)
                return extracted
            except json.JSONDecodeError:
                pass

        # Try to find JSON object or array directly
        # Look for { ... } or [ ... ]
        text_stripped = text.strip()

        # If starts with { or [, try to parse as-is
        if text_stripped.startswith(("{", "[")):
            try:
                json.loads(text_stripped)
                return text_stripped
            except json.JSONDecodeError:
                pass

        # Try to find embedded JSON object
        json_match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if json_match:
            try:
                json.loads(json_match.group(1))
                return json_match.group(1)
            except json.JSONDecodeError:
                pass

        # Return original text if no JSON found
        return text

    async def _get_client(self) -> Any:
        """Get or create Copilot client.

        Automatically finds the Copilot CLI path on Windows where npm/pnpm
        global installations may not be in the subprocess PATH.

        Returns:
            CopilotClient instance

        Raises:
            FileNotFoundError: If Copilot CLI is not installed
        """
        if self._client is None:
            from copilot import CopilotClient  # type: ignore[import-not-found]

            # Find CLI path (especially important on Windows)
            cli_path = _find_copilot_cli()
            if cli_path:
                logger.debug(f"[Copilot] Using CLI at: {cli_path}")
                self._client = CopilotClient({"cli_path": cli_path})
            else:
                # Let SDK try default 'copilot' command
                self._client = CopilotClient()

            await self._client.start()
        return self._client

    # Parameters that are not supported by Copilot SDK
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
        """Async completion using GitHub Copilot SDK.

        Supports multimodal input (text and images) and JSON mode (emulated).

        Args:
            model: Model identifier (e.g., "copilot/gpt-4.1")
            messages: OpenAI-style message list (supports multimodal content)
            **kwargs: Additional parameters:
                - response_format: OpenAI-style response format for JSON output
                  (emulated via prompt engineering, not native SDK support)
                - Other LLM params (max_tokens, temperature, etc.) are ignored
                  as Copilot SDK manages these internally.

        Returns:
            LiteLLM ModelResponse

        Raises:
            RuntimeError: If SDK is not available or request fails
        """
        # Log ignored parameters at DEBUG level
        ignored_params = [k for k in kwargs if k in self._UNSUPPORTED_PARAMS]
        if ignored_params:
            logger.debug(f"[Copilot] Ignoring unsupported params: {ignored_params}")

        if not _is_copilot_sdk_available():
            raise RuntimeError(
                "GitHub Copilot SDK 未安装。请执行: pip install github-copilot-sdk"
            )

        # Extract model name from provider prefix
        model_name = model.replace("copilot/", "")

        # Check if model is in the unsupported list (GPT-5 series, o1/o3)
        if model_name in self.UNSUPPORTED_MODELS:
            logger.warning(
                f"[Copilot] Model '{model_name}' is not supported by Copilot SDK. "
                "GPT-5/o1/o3 series models require 'max_completion_tokens' which "
                "the Copilot SDK does not support yet. Consider using a different model."
            )

        # Check for JSON mode request
        response_format = kwargs.get("response_format")
        json_prompt_suffix = self._build_json_prompt_suffix(response_format)
        is_json_mode = json_prompt_suffix is not None

        # Check if messages contain images
        has_images = self._has_images(messages)

        # Extract prompt and attachments (temp files created here)
        try:
            if has_images:
                prompt, attachments = self._extract_images(messages)
                logger.debug(
                    f"[Copilot] Calling model={model_name} with {len(attachments)} image(s)"
                )
            else:
                prompt = self._messages_to_prompt(messages)
                attachments = []
                logger.debug(
                    f"[Copilot] Calling model={model_name}, prompt_length={len(prompt)}"
                )

            # Append JSON format instructions if requested
            if json_prompt_suffix:
                prompt += json_prompt_suffix
                logger.debug(
                    "[Copilot] Using emulated JSON mode via prompt engineering"
                )
        except Exception as e:
            # Clean up temp files if extraction fails
            self._cleanup_temp_files()
            raise RuntimeError(f"消息预处理失败: {e}")

        start_time = time.time()
        result_text = ""
        session = None

        try:
            client = await self._get_client()

            # Create session without tools for pure LLM completion
            session = await client.create_session(
                {
                    "model": model_name,
                    "streaming": False,
                }
            )

            # Build request payload
            request_payload: dict[str, Any] = {"prompt": prompt}
            if attachments:
                request_payload["attachments"] = attachments

            # Send prompt and wait for response
            response = await asyncio.wait_for(
                session.send_and_wait(request_payload),
                timeout=self.timeout,
            )

            # Extract content from response
            if hasattr(response, "data") and hasattr(response.data, "content"):
                result_text = response.data.content
            else:
                result_text = str(response)

        except FileNotFoundError as e:
            # CLI not installed or not found in PATH
            logger.error(f"[Copilot] CLI not found: {e}")
            raise RuntimeError(
                "Copilot CLI 未安装或不在 PATH 中。"
                "请参考: https://docs.github.com/en/copilot/how-tos/set-up/install-copilot-cli"
            )
        except ConnectionError as e:
            # Server connection failed
            logger.error(f"[Copilot] Connection failed: {e}")
            raise RuntimeError(f"无法连接 Copilot 服务器: {e}")
        except TimeoutError:
            raise RuntimeError(
                f"Copilot 请求超时 ({self.timeout} 秒)。请检查网络连接或增加超时时间。"
            )
        except Exception as e:
            # Log and re-raise with context
            error_msg = str(e)
            logger.error(f"[Copilot] Error: {error_msg}")

            # Check for common error patterns
            if "not authenticated" in error_msg.lower():
                raise RuntimeError(
                    "Copilot CLI 未认证。请执行 'copilot auth login' 进行登录。"
                )
            elif "rate limit" in error_msg.lower():
                raise RuntimeError("已达到 Copilot API 速率限制，请稍后重试。")

            raise RuntimeError(f"GitHub Copilot SDK 错误: {e}")
        finally:
            # Clean up session (release resources)
            if session is not None:
                try:
                    await session.destroy()
                except Exception as e:
                    logger.debug(f"[Copilot] Session destroy warning: {e}")

            # Clean up temporary files
            self._cleanup_temp_files()

        elapsed = time.time() - start_time

        logger.debug(
            f"[Copilot] Completed in {elapsed:.2f}s, response_length={len(result_text)}"
        )

        # Extract JSON if JSON mode was requested
        response_content = result_text
        if is_json_mode:
            response_content = self._extract_json_from_response(result_text)
            if response_content != result_text:
                logger.debug("[Copilot] Extracted JSON from response")

        # Count tokens using tiktoken (for OpenAI models) or estimation (for others)
        from markitai.providers import count_tokens

        input_tokens = count_tokens(prompt, model_name)
        output_tokens = count_tokens(result_text, model_name)

        # Calculate estimated cost
        from markitai.providers import calculate_copilot_cost

        cost_result = calculate_copilot_cost(model_name, input_tokens, output_tokens)

        response = litellm.ModelResponse(
            id=f"copilot-{uuid.uuid4().hex[:12]}",
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

        # Store cost in hidden params for downstream retrieval
        # Note: All Copilot costs are ESTIMATED (subscription-based billing)
        response._hidden_params = {
            "total_cost_usd": cost_result.cost_usd,
            "cost_is_estimated": cost_result.is_estimated,
            "cost_source": cost_result.source,
        }

        if cost_result.cost_usd > 0:
            source_info = (
                f" (via {cost_result.source})"
                if cost_result.source != "litellm"
                else ""
            )
            logger.debug(
                f"[Copilot] Estimated cost: ${cost_result.cost_usd:.6f}{source_info}"
            )

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

    async def close(self) -> None:
        """Close the Copilot client connection."""
        # Clean up any remaining temp files
        self._cleanup_temp_files()

        if self._client is not None:
            try:
                await self._client.stop()
            except Exception as e:
                logger.warning(f"[Copilot] Error closing client: {e}")
            finally:
                self._client = None
