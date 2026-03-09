"""ChatGPT Responses API provider for LiteLLM.

This provider bypasses LiteLLM's broken chatgpt/ routing (which uses
ChatGPTConfig -> /chat/completions -> 403 Forbidden) and calls the
Responses API directly via httpx.

Usage:
    In configuration, use "chatgpt/<model>" as the model name:

    {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "chatgpt/codex-mini"
                    }
                }
            ]
        }
    }

Requirements:
    - LiteLLM with chatgpt authenticator support
    - Authenticated via Device Code Flow (auto-triggered on first use)

Limitations:
    - Does not support streaming responses (in this implementation)
    - max_tokens, temperature, top_p are NOT sent (backend rejects them)
"""

from __future__ import annotations

import platform
import uuid
from typing import TYPE_CHECKING, Any

import httpx
from loguru import logger

from markitai.providers.auth import get_auth_resolution_hint
from markitai.providers.common import UNSUPPORTED_PARAMS, sync_completion
from markitai.providers.errors import AuthenticationError, ProviderError
from markitai.providers.oauth_display import (
    DeviceCodeInterceptor,
    show_oauth_success,
)

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


def _import_authenticator() -> type:
    """Import the Authenticator class from LiteLLM's chatgpt module.

    Returns:
        The Authenticator class.

    Raises:
        ImportError: If the chatgpt authenticator module is not available.
    """
    from litellm.llms.chatgpt.authenticator import Authenticator

    return Authenticator


class ChatGPTProvider(CustomLLM):  # type: ignore[misc]
    """Custom LiteLLM provider using the ChatGPT Responses API.

    This provider bypasses LiteLLM's broken chatgpt/ routing by calling
    the Responses API endpoint directly via httpx, using LiteLLM's
    built-in Authenticator for Device Code Flow authentication.
    """

    # Parameters not supported by ChatGPT Responses API
    _UNSUPPORTED_PARAMS = UNSUPPORTED_PARAMS

    def __init__(self) -> None:
        """Initialize the provider."""
        if not LITELLM_AVAILABLE:
            raise RuntimeError("LiteLLM not available. Install with: uv add litellm")
        self._authenticator: Any | None = None
        self._client: httpx.AsyncClient | None = None

    def _get_authenticator(self) -> Any:
        """Get or create the LiteLLM Authenticator instance.

        Returns:
            Authenticator instance for ChatGPT OAuth.

        Raises:
            AuthenticationError: If the authenticator module is not available.
        """
        if self._authenticator is not None:
            return self._authenticator

        try:
            authenticator_cls = _import_authenticator()
        except ImportError as e:
            raise AuthenticationError(
                f"ChatGPT authenticator not available: {e}. "
                "Ensure LiteLLM is installed with chatgpt support.",
                provider="chatgpt",
                resolution_hint=(
                    "The LiteLLM chatgpt module is not available.\n"
                    "Upgrade LiteLLM: uv add litellm --upgrade"
                ),
            ) from e

        self._authenticator = authenticator_cls()
        return self._authenticator

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the httpx async client.

        Returns:
            httpx.AsyncClient instance.
        """
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    async def _stream_response(
        self,
        url: str,
        request_body: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[str, int, int]:
        """Make a streaming request and collect the full response.

        The ChatGPT Responses API requires stream=true. We collect text deltas
        and extract usage from the final response.completed event.

        Args:
            url: API endpoint URL.
            request_body: JSON request body.
            headers: HTTP headers.

        Returns:
            Tuple of (result_text, input_tokens, output_tokens).

        Raises:
            AuthenticationError: On 401/403 responses.
            ProviderError: On other API errors.
        """
        import json as json_mod

        text_parts: list[str] = []
        input_tokens = 0
        output_tokens = 0

        try:
            client = await self._get_client()
            async with client.stream(
                "POST", url, json=request_body, headers=headers
            ) as response:
                if response.status_code >= 400:
                    # Read the error body
                    error_body = await response.aread()
                    error_text = error_body.decode("utf-8", errors="replace")
                    try:
                        error_data = json_mod.loads(error_text)
                        error_text = (
                            error_data.get("error", {}).get("message", "")
                            or error_data.get("detail", "")
                            or str(error_data)
                        )
                    except Exception:
                        pass

                    if response.status_code in (401, 403):
                        raise AuthenticationError(
                            f"ChatGPT API authentication error "
                            f"({response.status_code}): {error_text}",
                            provider="chatgpt",
                            resolution_hint=get_auth_resolution_hint("chatgpt"),
                        )
                    raise ProviderError(
                        f"ChatGPT API error ({response.status_code}): {error_text}",
                        provider="chatgpt",
                    )

                # Parse SSE stream
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if not data_str:
                        continue
                    try:
                        event = json_mod.loads(data_str)
                    except json_mod.JSONDecodeError:
                        continue

                    event_type = event.get("type", "")

                    if event_type == "response.output_text.delta":
                        text_parts.append(event.get("delta", ""))

                    elif event_type == "response.completed":
                        # Final event with full response and usage
                        resp = event.get("response", {})
                        usage = resp.get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)
                        output_tokens = usage.get("output_tokens", 0)

                        # Also extract full text from completed response
                        # as a fallback if we missed deltas
                        if not text_parts:
                            for item in resp.get("output", []):
                                if item.get("type") == "message":
                                    for cb in item.get("content", []):
                                        if cb.get("type") == "output_text":
                                            text_parts.append(cb.get("text", ""))

        except (AuthenticationError, ProviderError):
            raise
        except httpx.ConnectError as e:
            raise ProviderError(
                f"ChatGPT connection error: {e}",
                provider="chatgpt",
            ) from e
        except Exception as e:
            raise ProviderError(
                f"ChatGPT request failed: {e}",
                provider="chatgpt",
            ) from e

        return "".join(text_parts), input_tokens, output_tokens

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str, list[dict[str, Any]]]:
        """Convert Chat Completions messages to Responses API format.

        Extracts system messages into `instructions` (top-level field required
        by Responses API) and converts remaining messages into `input` format.

        Args:
            messages: OpenAI Chat Completions style messages.

        Returns:
            Tuple of (instructions, input_messages).
        """
        instructions_parts: list[str] = []
        result: list[dict[str, Any]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # System messages → instructions field
            if role == "system":
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            instructions_parts.append(part.get("text", ""))
                continue

            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                converted_parts: list[dict[str, Any]] = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type", "")
                    if part_type == "text":
                        converted_parts.append(
                            {"type": "input_text", "text": part.get("text", "")}
                        )
                    elif part_type == "image_url":
                        image_url = part.get("image_url", {})
                        url = image_url.get("url", "")
                        converted_parts.append(
                            {"type": "input_image", "image_url": url}
                        )
                result.append({"role": role, "content": converted_parts})
            else:
                result.append({"role": role, "content": str(content)})

        instructions = "\n\n".join(instructions_parts) if instructions_parts else ""
        return instructions, result

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using ChatGPT Responses API.

        Args:
            model: Model identifier (e.g., "chatgpt/codex-mini").
            messages: OpenAI-style message list.
            **kwargs: Additional parameters (most are ignored).

        Returns:
            LiteLLM ModelResponse.

        Raises:
            AuthenticationError: If authentication fails.
            ProviderError: If the API request fails.
        """
        # Log ignored parameters at TRACE level to keep file logs compact
        ignored_params = [k for k in kwargs if k in self._UNSUPPORTED_PARAMS]
        if ignored_params:
            logger.trace(f"[ChatGPT] Ignoring unsupported params: {ignored_params}")

        # Strip provider prefix from model name
        model_name = model.replace("chatgpt/", "")

        # Authenticate — intercept device code stdout from LiteLLM
        import sys

        try:
            authenticator = self._get_authenticator()
            interceptor = DeviceCodeInterceptor()
            original_stdout = sys.stdout
            sys.stdout = interceptor  # type: ignore[assignment]
            try:
                access_token = authenticator.get_access_token()
            finally:
                sys.stdout = original_stdout
            if interceptor.displayed:
                show_oauth_success("chatgpt")
        except AuthenticationError:
            raise
        except Exception as e:
            hint = get_auth_resolution_hint("chatgpt")
            raise AuthenticationError(
                f"ChatGPT authentication failed: {e}",
                provider="chatgpt",
                resolution_hint=hint,
            ) from e

        # Get API base URL
        try:
            api_base = authenticator.get_api_base()
        except Exception:
            api_base = "https://chatgpt.com/backend-api/codex"

        # Get account ID for headers (optional)
        try:
            account_id = authenticator.get_account_id()
        except Exception:
            account_id = None

        # Convert messages to Responses API format
        instructions, converted_input = self._convert_messages(messages)

        # Build request body (Responses API format)
        # instructions is required; stream is required by the backend
        session_id = str(uuid.uuid4())
        request_body: dict[str, Any] = {
            "model": model_name,
            "instructions": instructions or "You are a helpful assistant.",
            "input": converted_input,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
        }

        # Build headers (match Codex CLI conventions)
        os_info = f"{platform.system()} {platform.release()}; {platform.machine()}"
        headers: dict[str, str] = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "accept": "text/event-stream",
            "originator": "codex_cli_rs",
            "user-agent": f"codex_cli_rs/markitai ({os_info})",
            "session_id": session_id,
        }
        if account_id:
            headers["ChatGPT-Account-Id"] = account_id

        url = f"{api_base}/responses"

        logger.debug(
            f"[ChatGPT] Calling model={model_name}, messages={len(messages)}, url={url}"
        )

        # Make streaming API call and collect the final response
        result_text, input_tokens, output_tokens = await self._stream_response(
            url, request_body, headers
        )

        logger.debug(
            f"[ChatGPT] Completed, response_length={len(result_text)}, "
            f"tokens={input_tokens}+{output_tokens}"
        )

        # Build ModelResponse
        model_response = litellm.ModelResponse(
            id=f"chatgpt-{uuid.uuid4().hex[:12]}",
            choices=[
                {
                    "message": {"role": "assistant", "content": result_text},
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

        # Estimate cost using LiteLLM pricing for the underlying model
        # ChatGPT is subscription-based, but we estimate equivalent API cost
        from markitai.providers import estimate_model_cost

        cost_result = estimate_model_cost(model_name, input_tokens, output_tokens)
        model_response._hidden_params = {
            "total_cost_usd": cost_result.cost_usd,
            "cost_is_estimated": True,
            "cost_source": cost_result.source
            if cost_result.cost_usd > 0
            else "chatgpt_subscription",
        }

        return model_response

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Sync completion wrapper.

        Args:
            model: Model identifier.
            messages: OpenAI-style message list.
            **kwargs: Additional parameters.

        Returns:
            LiteLLM ModelResponse.
        """
        return sync_completion(self, model, messages, **kwargs)

    async def close(self) -> None:
        """Close the httpx client."""
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.warning(f"[ChatGPT] Error closing client: {e}")
            finally:
                self._client = None
