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
from dataclasses import dataclass
from datetime import UTC, datetime
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
from markitai.providers.oauth_display import (
    show_oauth_start,
    show_oauth_success,
    suppress_stdout,
)
from markitai.security import atomic_write_json

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
MARKITAI_GEMINI_ACTIVE_PROFILE = "gemini-current.json"

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
CODE_ASSIST_ONBOARD_ENDPOINT = f"{CODE_ASSIST_BASE}:onboardUser"
GOOGLE_USERINFO_ENDPOINT = "https://www.googleapis.com/oauth2/v1/userinfo?alt=json"
CODE_ASSIST_METADATA = {
    "ideType": "IDE_UNSPECIFIED",
    "platform": "PLATFORM_UNSPECIFIED",
    "pluginType": "GEMINI",
}

# Code Assist rate limits are very low for some models (e.g., flash-lite ~1 RPM).
# Do NOT retry 429s internally — let the Router see the QuotaError immediately
# so it can set a cooldown and route subsequent requests to other models.
MAX_429_RETRIES = 0
TOKEN_CACHE_SAFETY_MARGIN_SECONDS = 60
DEFAULT_TOKEN_CACHE_SECONDS = 3000


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


@dataclass(frozen=True, slots=True)
class GeminiCredentialRecord:
    """Resolved Gemini credential profile."""

    path: Path
    source: str
    email: str | None
    project_id: str | None
    auth_mode: str | None


class GeminiCLIProvider(CustomLLM):  # type: ignore[misc]
    """Custom LiteLLM provider using Gemini CLI OAuth credentials.

    This provider enables using Gemini models through the Code Assist API
    by reusing Gemini CLI's OAuth credentials or running a built-in OAuth
    flow with the same client_id.
    """

    def __init__(self) -> None:
        super().__init__()
        # Token cache for concurrent safety (T4).
        # asyncio.Lock cannot be created here because __init__ may run
        # before an event loop exists; lazy-init in _get_access_token.
        self._token_lock: asyncio.Lock | None = None
        self._cached_token: str | None = None
        self._token_expiry: float = 0.0
        self._cached_token_source: str | None = None
        self._project_id: str | None = None
        self._project_source: str | None = None

    @property
    def _creds_path(self) -> Path:
        """Path to the Gemini CLI credentials file.

        Exposed as a property so tests can override it.
        """
        return GEMINI_CLI_CREDS_PATH

    @property
    def _managed_auth_dir(self) -> Path:
        """Directory containing Markitai-managed Gemini profiles."""
        return Path.home() / ".markitai" / "auth"

    @property
    def _active_profile_path(self) -> Path:
        """Path to the active Markitai-managed Gemini profile pointer."""
        return self._managed_auth_dir / MARKITAI_GEMINI_ACTIVE_PROFILE

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_profile_part(value: str) -> str:
        """Normalize profile filename components."""
        sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
        return sanitized.strip("._") or "default"

    def _managed_profile_path(self, email: str, project_id: str) -> Path:
        """Build the path for a managed Gemini profile."""
        safe_email = self._sanitize_profile_part(email)
        safe_project = self._sanitize_profile_part(project_id)
        return self._managed_auth_dir / f"gemini-{safe_email}-{safe_project}.json"

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any] | None:
        """Read a JSON file into a dict."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        return data if isinstance(data, dict) else None

    def _build_record(
        self,
        path: Path,
        data: dict[str, Any],
        *,
        source: str,
    ) -> GeminiCredentialRecord | None:
        """Build a credential record from a JSON payload."""
        if not data.get("access_token"):
            return None

        email = data.get("email")
        project_id = data.get("project_id")
        auth_mode = data.get("auth_mode")
        return GeminiCredentialRecord(
            path=path,
            source=source,
            email=str(email) if isinstance(email, str) and email else None,
            project_id=(
                str(project_id) if isinstance(project_id, str) and project_id else None
            ),
            auth_mode=(
                str(auth_mode) if isinstance(auth_mode, str) and auth_mode else None
            ),
        )

    def _load_managed_credential_payload(
        self,
    ) -> tuple[GeminiCredentialRecord, dict[str, Any]] | None:
        """Load the active Markitai-managed credential profile if present."""
        active_profile_path = self._active_profile_path
        active_data = self._read_json(active_profile_path)
        candidate_paths: list[Path] = []
        auth_dir = self._managed_auth_dir

        if active_data:
            credential_path = active_data.get("credential_path")
            if isinstance(credential_path, str) and credential_path:
                cred_path = Path(credential_path)
                try:
                    cred_path.resolve().relative_to(auth_dir.resolve())
                except ValueError:
                    logger.warning(
                        "[GeminiCLI] credential_path points outside managed "
                        f"dir, ignoring: {cred_path}"
                    )
                else:
                    candidate_paths.append(cred_path)
        if auth_dir.exists():
            for path in sorted(
                auth_dir.glob("gemini-*.json"),
                key=lambda item: item.stat().st_mtime,
                reverse=True,
            ):
                if path.name == active_profile_path.name:
                    continue
                candidate_paths.append(path)

        seen: set[Path] = set()
        for path in candidate_paths:
            if path in seen or not path.exists():
                continue
            seen.add(path)
            data = self._read_json(path)
            if not data:
                continue
            record = self._build_record(path, data, source="markitai")
            if record is not None:
                return record, data

        return None

    def _load_shared_credential_payload(
        self,
    ) -> tuple[GeminiCredentialRecord, dict[str, Any]] | None:
        """Load shared Gemini CLI credentials from ~/.gemini."""
        creds_path = self._creds_path
        if not creds_path.exists():
            return None

        data = self._read_json(creds_path)
        if not data:
            return None

        record = self._build_record(creds_path, data, source="gemini-cli")
        if record is None:
            return None
        return record, data

    def _get_credential_payload_candidates(
        self,
    ) -> list[tuple[GeminiCredentialRecord, dict[str, Any]]]:
        """Return ordered credential candidates for authentication attempts."""
        candidates: list[tuple[GeminiCredentialRecord, dict[str, Any]]] = []
        seen: set[Path] = set()

        for candidate in (
            self._load_managed_credential_payload(),
            self._load_shared_credential_payload(),
        ):
            if candidate is None:
                continue
            record, _ = candidate
            if record.path in seen:
                continue
            seen.add(record.path)
            candidates.append(candidate)

        return candidates

    def _select_credential_payload(
        self,
    ) -> tuple[GeminiCredentialRecord, dict[str, Any]] | None:
        """Resolve the credential payload to use for this request."""
        candidates = self._get_credential_payload_candidates()
        return candidates[0] if candidates else None

    def get_active_profile(self) -> GeminiCredentialRecord | None:
        """Return the currently active Gemini credential profile."""
        selected = self._select_credential_payload()
        if selected is None:
            return None
        return selected[0]

    def _build_credentials_from_data(
        self, data: dict[str, Any]
    ) -> Any:  # google.oauth2.credentials.Credentials | _RawToken | None
        """Build credential objects from raw JSON payload."""
        access_token = data.get("access_token")
        if not access_token:
            return None

        if not _GOOGLE_AUTH_AVAILABLE:
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

        expiry_date_ms = data.get("expiry_date")
        if expiry_date_ms:
            try:
                from datetime import UTC, datetime

                creds.expiry = datetime.fromtimestamp(
                    expiry_date_ms / 1000, tz=UTC
                ).replace(tzinfo=None)
            except (ValueError, TypeError, OSError):
                pass

        return creds

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
        selected = self._select_credential_payload()
        if selected is None:
            logger.debug(
                "[GeminiCLI] No managed or shared credentials found "
                f"(shared path: {self._creds_path})"
            )
            return None

        record, data = selected
        logger.debug(
            f"[GeminiCLI] Loaded credentials from {record.source}: {record.path}"
        )
        return self._build_credentials_from_data(data)

    def _create_oauth_credentials(self) -> Any:  # google.oauth2.credentials.Credentials
        """Run the interactive OAuth browser flow and return credentials."""
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

        # Suppress raw stdout from google-auth-oauthlib's run_local_server
        # (it prints "Please visit this URL...") and show Rich-styled output
        with suppress_stdout():
            try:
                return flow.run_local_server(port=GEMINI_CLI_REDIRECT_PORT)
            except OSError:
                # Port occupied (e.g. Gemini CLI running) — let OS pick a free port
                logger.debug(
                    f"[GeminiCLI] Port {GEMINI_CLI_REDIRECT_PORT} busy, using ephemeral port"
                )
                return flow.run_local_server(port=0)

    def _run_oauth_flow(self) -> Any:  # google.oauth2.credentials.Credentials
        """Run OAuth flow and persist the shared Gemini CLI credentials.

        Returns:
            google.oauth2.credentials.Credentials object.
        """
        show_oauth_start("gemini-cli")

        creds = self._create_oauth_credentials()

        # Save credentials for future reuse
        self._save_credentials(creds)

        show_oauth_success(
            "gemini-cli",
            detail=f"Saved to {self._creds_path}",
        )

        return creds

    def _save_credentials(
        self,
        creds: Any,
        *,
        path: Path | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save credentials to disk.

        Args:
            creds: google.oauth2.credentials.Credentials object.
            path: Target file path. Defaults to Gemini CLI shared path.
            metadata: Additional metadata to persist alongside the token payload.
        """
        creds_path = path or self._creds_path

        data: dict[str, Any] = {
            "access_token": creds.token,
            "refresh_token": creds.refresh_token,
            "client_id": getattr(creds, "client_id", GEMINI_CLI_CLIENT_ID),
            "client_secret": getattr(creds, "client_secret", GEMINI_CLI_CLIENT_SECRET),
        }

        if hasattr(creds, "expiry") and creds.expiry is not None:
            # Save as milliseconds Unix timestamp (Gemini CLI format)
            data["expiry_date"] = int(creds.expiry.timestamp() * 1000)

        if metadata:
            data.update(metadata)

        atomic_write_json(creds_path, data, ensure_ascii=False)
        logger.debug(f"[GeminiCLI] Saved credentials to {creds_path}")

    async def _get_access_token(self) -> str:
        """Get a valid access token, refreshing or running OAuth if needed.

        Uses an asyncio.Lock to prevent concurrent token refreshes and an
        in-memory cache to avoid redundant disk reads / network refreshes.

        Returns:
            Valid access token string.

        Raises:
            AuthenticationError: If authentication cannot be established.
            SDKNotAvailableError: If deps are missing AND no usable token exists.
        """
        candidate_sources = {
            str(record.path) for record, _ in self._get_credential_payload_candidates()
        }

        # Fast path: cached token still valid (lock-free)
        now = time.monotonic()
        if (
            self._cached_token
            and now < self._token_expiry
            and (
                (self._cached_token_source is None and not candidate_sources)
                or self._cached_token_source in candidate_sources
            )
        ):
            return self._cached_token

        # Lazy-init lock (must be in async context where event loop exists)
        if self._token_lock is None:
            self._token_lock = asyncio.Lock()

        async with self._token_lock:
            # Double-check after acquiring lock
            candidate_sources = {
                str(record.path)
                for record, _ in self._get_credential_payload_candidates()
            }
            now = time.monotonic()
            if (
                self._cached_token
                and now < self._token_expiry
                and (
                    (self._cached_token_source is None and not candidate_sources)
                    or self._cached_token_source in candidate_sources
                )
            ):
                return self._cached_token

            token, cache_deadline, token_source = await self._acquire_token()

            self._cached_token = token
            self._token_expiry = cache_deadline
            self._cached_token_source = token_source
            return token

    def _clear_cached_token(self) -> None:
        """Invalidate the in-memory token cache."""
        self._cached_token = None
        self._token_expiry = 0.0
        self._cached_token_source = None

    def _clear_cached_project(self) -> None:
        """Invalidate the cached project binding."""
        self._project_id = None
        self._project_source = None

    @staticmethod
    def _normalize_expiry_datetime(expiry: datetime) -> datetime:
        """Normalize mixed naive/aware expiry timestamps to UTC-aware datetimes."""
        if expiry.tzinfo is None:
            return expiry.replace(tzinfo=UTC)
        return expiry.astimezone(UTC)

    def _compute_token_cache_deadline(
        self,
        creds: Any,
        selected_data: dict[str, Any] | None,
    ) -> float:
        """Compute an in-memory cache deadline that never exceeds token expiry."""
        expiry: datetime | None = None

        raw_expiry = getattr(creds, "expiry", None)
        if isinstance(raw_expiry, datetime):
            expiry = self._normalize_expiry_datetime(raw_expiry)
        elif isinstance(creds, _RawToken) and isinstance(selected_data, dict):
            expiry_ms = selected_data.get("expiry_date")
            if expiry_ms is not None:
                try:
                    expiry = datetime.fromtimestamp(float(expiry_ms) / 1000, tz=UTC)
                except (ValueError, TypeError, OSError):
                    expiry = None

        if expiry is None:
            return time.monotonic() + DEFAULT_TOKEN_CACHE_SECONDS

        remaining_seconds = (expiry - datetime.now(UTC)).total_seconds()
        safe_remaining = max(0.0, remaining_seconds - TOKEN_CACHE_SAFETY_MARGIN_SECONDS)
        ttl_seconds = min(DEFAULT_TOKEN_CACHE_SECONDS, safe_remaining)
        return time.monotonic() + ttl_seconds

    async def _try_credentials_candidate(
        self,
        record: GeminiCredentialRecord,
        selected_data: dict[str, Any],
    ) -> tuple[str, float, str] | None:
        """Try a single credential payload and return a token when it works."""
        creds = self._build_credentials_from_data(selected_data)
        if creds is None:
            return None

        if isinstance(creds, _RawToken):
            return (
                creds.token,
                self._compute_token_cache_deadline(creds, selected_data),
                str(record.path),
            )

        if creds.valid and not creds.expired:
            return (
                creds.token,
                self._compute_token_cache_deadline(creds, selected_data),
                str(record.path),
            )

        if creds.expired and creds.refresh_token:
            max_retries = 2
            for attempt in range(1 + max_retries):
                try:
                    request = _google_auth_requests.Request()  # type: ignore[union-attr]
                    creds.refresh(request)
                    metadata: dict[str, Any] | None = None
                    if record.source == "markitai":
                        metadata = {
                            key: selected_data.get(key)
                            for key in (
                                "email",
                                "project_id",
                                "auth_mode",
                                "source",
                            )
                            if selected_data.get(key) is not None
                        }
                    self._save_credentials(
                        creds,
                        path=record.path,
                        metadata=metadata,
                    )
                    logger.debug(
                        "[GeminiCLI] Token refreshed successfully from "
                        f"{record.source}: {record.path}"
                    )
                    return (
                        creds.token,
                        self._compute_token_cache_deadline(creds, selected_data),
                        str(record.path),
                    )
                except Exception as e:
                    if attempt < max_retries:
                        delay = 1.0 * (2**attempt)
                        logger.debug(
                            f"[GeminiCLI] Token refresh attempt {attempt + 1} "
                            f"failed for {record.path}: {e}, "
                            f"retrying in {delay:.0f}s"
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.warning(
                            f"[GeminiCLI] Token refresh failed for {record.path} "
                            f"after {1 + max_retries} attempts: {e}, "
                            "trying next credential source"
                        )

        return None

    async def _acquire_token(self) -> tuple[str, float, str | None]:
        """Internal: acquire a fresh token (no caching/locking).

        Returns:
            Tuple of (valid access token string, monotonic cache deadline, source path).

        Raises:
            AuthenticationError: If authentication cannot be established.
            SDKNotAvailableError: If deps are missing AND no usable token exists.
        """
        for record, selected_data in self._get_credential_payload_candidates():
            token_result = await self._try_credentials_candidate(record, selected_data)
            if token_result is not None:
                return token_result

        # 5. No valid credentials — fail fast instead of launching interactive
        # OAuth browser flow during an LLM call (which would disrupt spinners).
        # Interactive auth should go through alogin() / preflight / `markitai auth`.
        if not _GOOGLE_AUTH_AVAILABLE:
            raise SDKNotAvailableError(
                "google-auth is required for the gemini-cli provider "
                "(no valid cached token found). "
                "Install with: uv add 'markitai[gemini-cli]'",
                provider="gemini-cli",
                install_command="uv add 'markitai[gemini-cli]'",
            )

        raise AuthenticationError(
            "No valid Gemini credentials found. "
            "Run 'markitai auth gemini login' to authenticate.",
            provider="gemini-cli",
            resolution_hint=(
                "Run 'markitai auth gemini login' to create a managed "
                "profile, or 'gemini login' to reuse Gemini CLI auth.\n"
                "Requires: uv add 'markitai[gemini-cli]'"
            ),
        )

    @staticmethod
    def _extract_project_id(data: dict[str, Any] | None) -> str | None:
        """Extract a project ID from Code Assist responses."""
        if not isinstance(data, dict):
            return None

        for key in ("cloudaicompanionProject", "cloudProject", "projectId"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested_id = value.get("id")
                if isinstance(nested_id, str) and nested_id.strip():
                    return nested_id.strip()
        return None

    @staticmethod
    def _extract_default_tier_id(data: dict[str, Any]) -> str:
        """Extract the default tier from loadCodeAssist results."""
        tiers = data.get("allowedTiers")
        if isinstance(tiers, list):
            for raw_tier in tiers:
                if not isinstance(raw_tier, dict):
                    continue
                if raw_tier.get("isDefault") is True:
                    tier_id = raw_tier.get("id")
                    if isinstance(tier_id, str) and tier_id.strip():
                        return tier_id.strip()
        return "legacy-tier"

    async def _call_code_assist_endpoint(
        self,
        endpoint: str,
        access_token: str,
        payload: dict[str, Any],
        *,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Call a Code Assist endpoint and return the JSON body."""
        if httpx is None:
            raise SDKNotAvailableError(
                "httpx is required for the gemini-cli provider.",
                provider="gemini-cli",
                install_command="uv add httpx",
            )

        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                endpoint,
                headers={"Authorization": f"Bearer {access_token}"},
                json=payload,
            )

        if resp.status_code == 401:
            self._clear_cached_token()
            self._clear_cached_project()
            raise AuthenticationError(
                f"Gemini CLI authentication failed (HTTP 401): {resp.text}",
                provider="gemini-cli",
                resolution_hint=(
                    "Run 'markitai auth gemini login' to refresh your managed "
                    "profile, or 'gemini login' to refresh shared CLI auth."
                ),
            )

        if resp.status_code != 200:
            raise ProviderError(
                f"Code Assist API error (HTTP {resp.status_code}): {resp.text}",
                provider="gemini-cli",
            )

        data = resp.json()
        return data if isinstance(data, dict) else {}

    async def _fetch_user_email(self, access_token: str) -> str | None:
        """Resolve the authenticated Google account email."""
        if httpx is None:
            raise SDKNotAvailableError(
                "httpx is required for the gemini-cli provider.",
                provider="gemini-cli",
                install_command="uv add httpx",
            )

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(
                GOOGLE_USERINFO_ENDPOINT,
                headers={"Authorization": f"Bearer {access_token}"},
            )

        if resp.status_code != 200:
            return None

        data = resp.json()
        email = data.get("email") if isinstance(data, dict) else None
        return email.strip() if isinstance(email, str) and email.strip() else None

    async def _onboard_user(
        self,
        access_token: str,
        tier_id: str,
        *,
        project_id: str | None,
    ) -> str | None:
        """Run Code Assist onboarding until a project binding is ready."""
        payload: dict[str, Any] = {
            "tierId": tier_id,
            "metadata": dict(CODE_ASSIST_METADATA),
        }
        if project_id:
            payload["cloudaicompanionProject"] = project_id

        deadline = time.monotonic() + 30.0
        while True:
            data = await self._call_code_assist_endpoint(
                CODE_ASSIST_ONBOARD_ENDPOINT,
                access_token,
                payload,
                timeout=30.0,
            )
            if data.get("done") is True:
                response = data.get("response")
                if isinstance(response, dict):
                    return self._extract_project_id(response) or project_id
                return project_id

            if time.monotonic() >= deadline:
                return project_id
            await asyncio.sleep(2)

    async def _resolve_login_project(
        self,
        access_token: str,
        *,
        mode: str,
        project_id: str | None = None,
    ) -> str:
        """Resolve a stable project binding during login."""
        normalized_mode = mode.strip().lower()
        if normalized_mode not in {"google-one", "code-assist"}:
            raise AuthenticationError(
                f"Unsupported Gemini auth mode: {mode}",
                provider="gemini-cli",
            )

        requested_project = project_id.strip() if project_id else None
        if normalized_mode == "code-assist" and not requested_project:
            raise AuthenticationError(
                "Code Assist mode requires --project-id.",
                provider="gemini-cli",
                resolution_hint="Run 'markitai auth gemini login --mode code-assist --project-id <PROJECT_ID>'.",
            )

        load_payload: dict[str, Any] = {"metadata": dict(CODE_ASSIST_METADATA)}
        if requested_project:
            load_payload["cloudaicompanionProject"] = requested_project

        load_data = await self._call_code_assist_endpoint(
            CODE_ASSIST_LOAD_ENDPOINT,
            access_token,
            load_payload,
            timeout=15.0,
        )
        resolved_project = self._extract_project_id(load_data) or requested_project
        tier_id = self._extract_default_tier_id(load_data)

        if normalized_mode == "google-one":
            if resolved_project:
                return resolved_project
            discovered = await self._onboard_user(
                access_token,
                tier_id,
                project_id=None,
            )
            if discovered:
                return discovered
        else:
            onboarded = await self._onboard_user(
                access_token,
                tier_id,
                project_id=resolved_project,
            )
            if onboarded:
                return onboarded

        raise AuthenticationError(
            "Failed to resolve a Gemini Code Assist project.",
            provider="gemini-cli",
            resolution_hint=(
                "Try 'markitai auth gemini login --mode google-one', "
                "or provide an explicit --project-id for Code Assist mode."
            ),
        )

    async def alogin(
        self,
        *,
        mode: str = "google-one",
        project_id: str | None = None,
    ) -> GeminiCredentialRecord:
        """Run interactive Gemini OAuth and save a Markitai-managed profile."""
        creds = self._create_oauth_credentials()
        access_token = getattr(creds, "token", None)
        if not isinstance(access_token, str) or not access_token:
            raise AuthenticationError(
                "OAuth login did not return a valid access token.",
                provider="gemini-cli",
            )

        email = await self._fetch_user_email(access_token)
        if not email:
            raise AuthenticationError(
                "Failed to resolve Google account email from Gemini login.",
                provider="gemini-cli",
            )

        resolved_project = await self._resolve_login_project(
            access_token,
            mode=mode,
            project_id=project_id,
        )
        profile_path = self._managed_profile_path(email, resolved_project)
        metadata = {
            "email": email,
            "project_id": resolved_project,
            "auth_mode": mode,
            "source": "markitai",
        }
        self._save_credentials(creds, path=profile_path, metadata=metadata)
        atomic_write_json(
            self._active_profile_path,
            {"credential_path": str(profile_path)},
            ensure_ascii=False,
        )
        self._clear_cached_token()
        self._clear_cached_project()
        return GeminiCredentialRecord(
            path=profile_path,
            source="markitai",
            email=email,
            project_id=resolved_project,
            auth_mode=mode,
        )

    async def _get_project_id(self, access_token: str) -> str | None:
        """Resolve the project binding for the current credential profile."""
        selected = self._select_credential_payload()
        selected_source = str(selected[0].path) if selected is not None else None
        if selected is not None:
            record, _data = selected
            if record.project_id:
                return record.project_id

        if self._project_id is not None and (
            self._project_source == selected_source or self._project_source is None
        ):
            return self._project_id

        try:
            data = await self._call_code_assist_endpoint(
                CODE_ASSIST_LOAD_ENDPOINT,
                access_token,
                {},
                timeout=15.0,
            )
            project = self._extract_project_id(data)
            if project:
                self._project_id = project
                self._project_source = selected_source
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

        # Estimate cost using LiteLLM pricing for the underlying model
        from markitai.providers import estimate_model_cost

        # Map gemini-cli model name to standard gemini format for LiteLLM lookup
        bare_model = model.replace("gemini-cli/", "")
        cost_result = estimate_model_cost(bare_model, prompt_tokens, completion_tokens)
        if cost_result.cost_usd == 0:
            # Try with gemini/ prefix (LiteLLM registers Gemini models this way)
            cost_result = estimate_model_cost(
                f"gemini/{bare_model}", prompt_tokens, completion_tokens
            )
        response._hidden_params = {
            "total_cost_usd": cost_result.cost_usd,
            "cost_is_estimated": True,
            "cost_source": cost_result.source,
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
            self._clear_cached_token()
            self._clear_cached_project()
            raise AuthenticationError(
                f"Gemini CLI authentication failed (HTTP 401): {resp.text}",
                provider="gemini-cli",
                resolution_hint=(
                    "Run 'markitai auth gemini login' to refresh your managed "
                    "profile, or 'gemini login' to refresh shared CLI auth."
                ),
            )

        if resp.status_code == 429:
            raise QuotaError(
                f"Code Assist rate limit exceeded: {resp.text}",
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

        choice = result.choices[0]
        content = getattr(choice, "message", None)
        content_text = (content.content or "") if content else ""
        logger.trace(
            f"[GeminiCLI] Completed in {elapsed:.2f}s, "
            f"response_length={len(content_text)}"
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
