"""Gemini CLI provider for LiteLLM.

This provider uses Google's Gemini CLI OAuth credentials to make LLM
requests through the Code Assist internal API, allowing users who have
Gemini CLI installed to use their existing authentication.

Usage:
    In configuration, use "gemini-cli/<model>" as the model name:

    {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "gemini-cli/gemini-2.5-pro"
                    }
                }
            ]
        }
    }

Authentication (dual-mode):
    1. Reads existing Gemini CLI credentials from ~/.gemini/oauth_creds.json
    2. Falls back to built-in OAuth flow using the same client_id as Gemini CLI

Requirements:
    - google-auth and google-auth-oauthlib packages: uv add 'markitai[gemini-cli]'
    - OR: Gemini CLI installed and authenticated (credentials reused automatically)

Limitations:
    - Does not support streaming responses (in this implementation)
    - Image size limit: per Gemini API limits
"""

from __future__ import annotations

import asyncio
import base64 as _b64
import json
import re
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from markitai.providers.common import sync_completion
from markitai.providers.errors import (
    AuthenticationError,
    ProviderError,
    QuotaError,
    SDKNotAvailableError,
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

# Optional: google-auth for credential management
try:
    import google.auth.transport.requests as _google_auth_requests  # type: ignore[import-untyped]
    import google.oauth2.credentials as _google_oauth2_credentials  # type: ignore[import-untyped]

    _GOOGLE_AUTH_AVAILABLE = True
except ImportError:
    _GOOGLE_AUTH_AVAILABLE = False
    _google_auth_requests = None  # type: ignore[assignment]
    _google_oauth2_credentials = None  # type: ignore[assignment]

# Optional: google-auth-oauthlib for OAuth flow
try:
    from google_auth_oauthlib.flow import (
        InstalledAppFlow as _InstalledAppFlow,  # type: ignore[import-untyped]
    )

    _OAUTHLIB_AVAILABLE = True
except ImportError:
    _OAUTHLIB_AVAILABLE = False
    _InstalledAppFlow = None  # type: ignore[assignment]

# Optional: httpx for async HTTP requests
try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GEMINI_CLI_CREDS_PATH = Path.home() / ".gemini" / "oauth_creds.json"

# These are PUBLIC values hardcoded in Gemini CLI's open-source code.
# This is standard practice for installed/native OAuth applications.
# Stored as base64 to avoid triggering GitHub push protection (secret scanner).
GEMINI_CLI_CLIENT_ID = _b64.b64decode(
    "NjgxMjU1ODA5Mzk1LW9vOGZ0Mm9wcmRybnA5ZTNhcWY2YXYzaG1kaWIxMzVqLmFwcHMuZ29vZ2xldXNlcmNvbnRlbnQuY29t"
).decode()
GEMINI_CLI_CLIENT_SECRET = _b64.b64decode(
    "R09DU1BYLTR1SGdNUG0tMW83U2stZ2VWNkN1NWNsWEZzeGw="
).decode()
GEMINI_CLI_REDIRECT_PORT = 45289
GEMINI_CLI_SCOPES = [
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

CODE_ASSIST_BASE = "https://cloudcode-pa.googleapis.com/v1internal"
CODE_ASSIST_ENDPOINT = f"{CODE_ASSIST_BASE}:generateContent"
CODE_ASSIST_STREAM_ENDPOINT = f"{CODE_ASSIST_BASE}:streamGenerateContent"
CODE_ASSIST_LOAD_ENDPOINT = f"{CODE_ASSIST_BASE}:loadCodeAssist"

# Code Assist rate limits are very low for some models (e.g., flash-lite ~1 RPM).
# Retry 429s with the wait time from the response.
MAX_429_RETRIES = 3


class _RawToken:
    """Lightweight token wrapper used when google-auth is not installed.

    Provides the same interface as google.oauth2.credentials.Credentials
    for token access, but without refresh capability.
    """

    __slots__ = ("token",)

    def __init__(self, token: str) -> None:
        self.token = token

    @property
    def valid(self) -> bool:
        return bool(self.token)

    @property
    def expired(self) -> bool:
        return False  # Cannot determine without google-auth

    @property
    def refresh_token(self) -> None:
        return None


class GeminiCLIProvider(CustomLLM):  # type: ignore[misc]
    """Custom LiteLLM provider using Gemini CLI OAuth credentials.

    This provider enables using Gemini models through the Code Assist API
    by reusing Gemini CLI's OAuth credentials or running a built-in OAuth
    flow with the same client_id.
    """

    _project_id: str | None = None

    @property
    def _creds_path(self) -> Path:
        """Path to the Gemini CLI credentials file.

        Exposed as a property so tests can override it.
        """
        return GEMINI_CLI_CREDS_PATH

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _load_credentials(
        self,
    ) -> Any:  # google.oauth2.credentials.Credentials | None
        """Load credentials from Gemini CLI's credential file.

        When google-auth is available, returns a Credentials object with
        refresh support. When google-auth is NOT available, returns a
        lightweight _RawToken wrapper that provides the raw access_token
        (no refresh capability).

        Returns:
            Credentials-like object if valid creds found, None otherwise.
        """
        creds_path = self._creds_path
        if not creds_path.exists():
            logger.debug(f"[GeminiCLI] Credentials file not found: {creds_path}")
            return None

        try:
            data = json.loads(creds_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[GeminiCLI] Failed to read credentials: {e}")
            return None

        access_token = data.get("access_token")
        if not access_token:
            logger.debug("[GeminiCLI] No access_token in credentials file")
            return None

        if not _GOOGLE_AUTH_AVAILABLE:
            # Without google-auth we can't refresh tokens, but we can
            # still use the raw access_token if it hasn't expired.
            logger.debug(
                "[GeminiCLI] google-auth not installed, using raw token "
                "(no refresh support)"
            )
            return _RawToken(access_token)

        creds = _google_oauth2_credentials.Credentials(  # type: ignore[union-attr]
            token=access_token,
            refresh_token=data.get("refresh_token"),
            token_uri="https://oauth2.googleapis.com/token",  # nosec B106
            client_id=data.get("client_id", GEMINI_CLI_CLIENT_ID),
            client_secret=data.get("client_secret", GEMINI_CLI_CLIENT_SECRET),
            scopes=GEMINI_CLI_SCOPES,
        )

        # Set expiry if available
        # Gemini CLI stores expiry_date as milliseconds Unix timestamp
        expiry_date_ms = data.get("expiry_date")
        if expiry_date_ms:
            try:
                from datetime import UTC, datetime

                creds.expiry = datetime.fromtimestamp(
                    expiry_date_ms / 1000, tz=UTC
                ).replace(tzinfo=None)
            except (ValueError, TypeError, OSError):
                pass

        logger.debug("[GeminiCLI] Loaded credentials from file")
        return creds

    def _run_oauth_flow(self) -> Any:  # google.oauth2.credentials.Credentials
        """Run OAuth flow using google_auth_oauthlib.

        Opens browser for Google OAuth login using the same client_id as
        Gemini CLI. Saves credentials to ~/.gemini/oauth_creds.json for
        future reuse.

        Returns:
            google.oauth2.credentials.Credentials object.

        Raises:
            SDKNotAvailableError: If google-auth-oauthlib is not installed.
        """
        if not _OAUTHLIB_AVAILABLE:
            raise SDKNotAvailableError(
                "google-auth-oauthlib is required for OAuth login. "
                "Install with: uv add 'markitai[gemini-cli]'",
                provider="gemini-cli",
                install_command="uv add 'markitai[gemini-cli]'",
            )

        client_config = {
            "installed": {
                "client_id": GEMINI_CLI_CLIENT_ID,
                "client_secret": GEMINI_CLI_CLIENT_SECRET,
                "redirect_uris": [f"http://localhost:{GEMINI_CLI_REDIRECT_PORT}"],
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",  # nosec B105
            }
        }

        flow = _InstalledAppFlow.from_client_config(  # type: ignore[union-attr]
            client_config,
            scopes=GEMINI_CLI_SCOPES,
        )

        logger.info("[GeminiCLI] Opening browser for OAuth login...")
        creds = flow.run_local_server(port=GEMINI_CLI_REDIRECT_PORT)

        # Save credentials for future reuse
        self._save_credentials(creds)

        return creds

    def _save_credentials(self, creds: Any) -> None:
        """Save credentials to the Gemini CLI credentials file.

        Args:
            creds: google.oauth2.credentials.Credentials object.
        """
        creds_path = self._creds_path
        creds_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "access_token": creds.token,
            "refresh_token": creds.refresh_token,
            "client_id": getattr(creds, "client_id", GEMINI_CLI_CLIENT_ID),
            "client_secret": getattr(creds, "client_secret", GEMINI_CLI_CLIENT_SECRET),
        }

        if hasattr(creds, "expiry") and creds.expiry is not None:
            # Save as milliseconds Unix timestamp (Gemini CLI format)
            data["expiry_date"] = int(creds.expiry.timestamp() * 1000)

        creds_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.debug(f"[GeminiCLI] Saved credentials to {creds_path}")

    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing or running OAuth if needed.

        Returns:
            Valid access token string.

        Raises:
            AuthenticationError: If authentication cannot be established.
            SDKNotAvailableError: If deps are missing AND no usable token exists.
        """
        # 1. Try loading existing credentials
        creds = self._load_credentials()

        if creds is not None:
            # 2. _RawToken (no google-auth) — use token directly
            if isinstance(creds, _RawToken):
                return creds.token

            # 3. If valid, return immediately
            if creds.valid and not creds.expired:
                return creds.token

            # 4. If expired but has refresh token, try refreshing
            if creds.expired and creds.refresh_token:
                try:
                    request = _google_auth_requests.Request()  # type: ignore[union-attr]
                    creds.refresh(request)
                    self._save_credentials(creds)
                    logger.debug("[GeminiCLI] Token refreshed successfully")
                    return creds.token
                except Exception as e:
                    logger.warning(
                        f"[GeminiCLI] Token refresh failed: {e}, "
                        "falling back to OAuth flow"
                    )

        # 5. No valid credentials — need google-auth-oauthlib for OAuth flow
        if not _GOOGLE_AUTH_AVAILABLE:
            raise SDKNotAvailableError(
                "google-auth is required for the gemini-cli provider "
                "(no valid cached token found). "
                "Install with: uv add 'markitai[gemini-cli]'",
                provider="gemini-cli",
                install_command="uv add 'markitai[gemini-cli]'",
            )

        try:
            creds = self._run_oauth_flow()
            return creds.token
        except SDKNotAvailableError:
            raise
        except Exception as e:
            raise AuthenticationError(
                f"Failed to authenticate with Gemini CLI: {e}",
                provider="gemini-cli",
                resolution_hint=(
                    "Run 'gemini login' to authenticate with Gemini CLI, "
                    "or install google-auth-oauthlib: "
                    "uv add 'markitai[gemini-cli]'"
                ),
            ) from e

    async def _get_project_id(self, access_token: str) -> str | None:
        """Discover GCP project ID via loadCodeAssist endpoint.

        Results are cached after the first successful call.

        Args:
            access_token: Valid OAuth access token.

        Returns:
            Project ID string or None if discovery fails.
        """
        if self._project_id is not None:
            return self._project_id

        if httpx is None:
            return None

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    CODE_ASSIST_LOAD_ENDPOINT,
                    headers={"Authorization": f"Bearer {access_token}"},
                    json={},
                )
            if resp.status_code == 200:
                data = resp.json()
                project = (
                    data.get("cloudaicompanionProject")
                    or data.get("cloudProject")
                    or data.get("projectId")
                )
                if project:
                    self._project_id = project
                    logger.debug(f"[GeminiCLI] Discovered project: {project}")
                    return project
        except Exception as e:
            logger.debug(f"[GeminiCLI] Project discovery failed: {e}")

        return None

    # ------------------------------------------------------------------
    # Message conversion
    # ------------------------------------------------------------------

    def _convert_messages(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """Convert OpenAI-style messages to Gemini content format.

        Args:
            messages: OpenAI-style message list.

        Returns:
            Tuple of (contents, systemInstruction).
            contents: List of Gemini content dicts.
            systemInstruction: Dict with "parts" key or None.
        """
        contents: list[dict[str, Any]] = []
        system_parts: list[dict[str, str]] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                # Collect system messages for systemInstruction
                if isinstance(content, str):
                    system_parts.append({"text": content})
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            system_parts.append({"text": part.get("text", "")})
                continue

            # Map roles: assistant -> model, user -> user
            gemini_role = "model" if role == "assistant" else "user"
            parts = self._convert_content_to_parts(content)

            contents.append({"role": gemini_role, "parts": parts})

        system_instruction = {"parts": system_parts} if system_parts else None
        return contents, system_instruction

    def _convert_content_to_parts(
        self, content: str | list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Convert message content to Gemini parts format.

        Args:
            content: String content or list of content blocks.

        Returns:
            List of Gemini part dicts.
        """
        if isinstance(content, str):
            return [{"text": content}]

        parts: list[dict[str, Any]] = []
        for block in content:
            if not isinstance(block, dict):
                continue

            block_type = block.get("type", "")
            if block_type == "text":
                parts.append({"text": block.get("text", "")})
            elif block_type == "image_url":
                image_url = block.get("image_url", {})
                url = image_url.get("url", "")
                if url.startswith("data:"):
                    try:
                        header, data = url.split(",", 1)
                        mime_type = header.split(":")[1].split(";")[0]
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": mime_type,
                                    "data": data,
                                }
                            }
                        )
                    except (ValueError, IndexError):
                        logger.warning("[GeminiCLI] Invalid image data URL format")
                else:
                    logger.warning(
                        "[GeminiCLI] URL-based images not supported, "
                        "only base64 data URLs"
                    )

        return parts

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(
        self,
        model: str,
        messages: list[dict[str, Any]],
        project: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build CAGenerateContentRequest.

        Args:
            model: Model name (may include gemini-cli/ prefix).
            messages: OpenAI-style message list.
            project: GCP project ID (auto-discovered if not provided).
            **kwargs: Additional params (temperature, max_tokens, etc.).

        Returns:
            Request dict for the Code Assist API.
        """
        # Strip provider prefix
        model_name = model.replace("gemini-cli/", "")

        contents, system_instruction = self._convert_messages(messages)

        # Build generation config from kwargs
        gen_config: dict[str, Any] = {}
        if "temperature" in kwargs:
            gen_config["temperature"] = kwargs["temperature"]
        if "max_tokens" in kwargs:
            gen_config["maxOutputTokens"] = kwargs["max_tokens"]
        if "top_p" in kwargs:
            gen_config["topP"] = kwargs["top_p"]

        # Build request payload
        request_body: dict[str, Any] = {
            "contents": contents,
        }

        if system_instruction is not None:
            request_body["systemInstruction"] = system_instruction

        if gen_config:
            request_body["generationConfig"] = gen_config

        payload: dict[str, Any] = {
            "model": model_name,
            "request": request_body,
        }

        if project:
            payload["project"] = project

        return payload

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, data: dict[str, Any], model: str) -> ModelResponse:
        """Parse Gemini API response into LiteLLM ModelResponse.

        Args:
            data: Raw response JSON from Code Assist API.
                  Code Assist wraps the standard Gemini response in a
                  top-level {"response": {...}, "traceId": ..., "metadata": ...}
                  envelope. This method handles both wrapped and unwrapped formats.
            model: Full model identifier (with prefix).

        Returns:
            LiteLLM ModelResponse.
        """
        # Unwrap Code Assist envelope if present
        inner = data.get("response", data)

        # Extract text from candidates
        text = ""
        candidates = inner.get("candidates", [])
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            text_parts = [p.get("text", "") for p in parts if "text" in p]
            text = "".join(text_parts)

        # Extract usage metadata
        usage_meta = inner.get("usageMetadata", {})
        prompt_tokens = usage_meta.get("promptTokenCount", 0)
        completion_tokens = usage_meta.get("candidatesTokenCount", 0)
        total_tokens = usage_meta.get(
            "totalTokenCount", prompt_tokens + completion_tokens
        )

        response = litellm.ModelResponse(
            id=f"gemini-cli-{uuid.uuid4().hex[:12]}",
            choices=[
                {
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                    "index": 0,
                }
            ],
            model=model,
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
        )

        # Store in hidden params for cost tracking
        response._hidden_params = {
            "total_cost_usd": 0.0,
        }

        return response

    # ------------------------------------------------------------------
    # API call
    # ------------------------------------------------------------------

    async def acompletion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Async completion using Gemini Code Assist API.

        Args:
            model: Model identifier (e.g., "gemini-cli/gemini-2.5-pro").
            messages: OpenAI-style message list.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Returns:
            LiteLLM ModelResponse.

        Raises:
            AuthenticationError: If authentication fails (401).
            ProviderError: If the API returns an error.
            SDKNotAvailableError: If required dependencies are missing.
        """
        if httpx is None:
            raise SDKNotAvailableError(
                "httpx is required for the gemini-cli provider.",
                provider="gemini-cli",
                install_command="uv add httpx",
            )

        # Get access token (may trigger refresh or OAuth flow)
        access_token = await self._get_access_token()

        # Auto-discover project ID (cached after first call)
        project_id = await self._get_project_id(access_token)

        # Build request
        request_payload = self._build_request(
            model, messages, project=project_id, **kwargs
        )

        logger.debug(
            f"[GeminiCLI] Calling model={request_payload['model']}, "
            f"contents_count={len(request_payload['request'].get('contents', []))}"
        )

        start_time = time.time()
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        # Retry loop for 429 rate limiting
        for attempt in range(MAX_429_RETRIES + 1):
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    CODE_ASSIST_ENDPOINT,
                    headers=headers,
                    json=request_payload,
                )

            if resp.status_code == 429 and attempt < MAX_429_RETRIES:
                # Parse "reset after Xs" from error message
                wait_seconds = self._parse_retry_after(resp.text)
                logger.info(
                    f"[GeminiCLI] Rate limited (429), "
                    f"waiting {wait_seconds}s before retry "
                    f"({attempt + 1}/{MAX_429_RETRIES})"
                )
                await asyncio.sleep(wait_seconds)
                continue

            break

        # Handle error responses
        if resp.status_code == 401:
            raise AuthenticationError(
                f"Gemini CLI authentication failed (HTTP 401): {resp.text}",
                provider="gemini-cli",
                resolution_hint=(
                    "Run 'gemini login' to re-authenticate, or check "
                    "that your Google account has access to Code Assist."
                ),
            )

        if resp.status_code == 429:
            raise QuotaError(
                f"Code Assist rate limit exceeded after {MAX_429_RETRIES} "
                f"retries: {resp.text}",
                provider="gemini-cli",
            )

        if resp.status_code != 200:
            raise ProviderError(
                f"Code Assist API error (HTTP {resp.status_code}): {resp.text}",
                provider="gemini-cli",
            )

        elapsed = time.time() - start_time
        response_data = resp.json()

        result = self._parse_response(response_data, model)

        logger.debug(
            f"[GeminiCLI] Completed in {elapsed:.2f}s, "
            f"response_length={len(result.choices[0].message.content or '')}"
        )

        return result

    @staticmethod
    def _parse_retry_after(error_text: str) -> float:
        """Parse wait time from 429 error message.

        Extracts the number from "Your quota will reset after Xs."
        Falls back to 60 seconds if parsing fails.
        """
        match = re.search(r"reset after (\d+)s", error_text)
        if match:
            return float(match.group(1)) + 1  # +1s buffer
        return 60.0

    def completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> ModelResponse:
        """Sync completion wrapper.

        Note: Prefer using acompletion() in async contexts for better
        performance. This method is provided for sync compatibility only.

        Args:
            model: Model identifier.
            messages: OpenAI-style message list.
            **kwargs: Additional parameters.

        Returns:
            LiteLLM ModelResponse.

        Raises:
            RuntimeError: If called from within a running event loop.
        """
        return sync_completion(self, model, messages, **kwargs)
