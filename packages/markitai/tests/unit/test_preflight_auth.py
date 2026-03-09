"""Tests for pre-flight auth check."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from markitai.providers import preflight_auth_check
from markitai.providers.auth import AuthStatus


def _make_model_config(model: str, weight: int = 1) -> MagicMock:
    """Create a mock ModelConfig."""
    mc = MagicMock()
    mc.litellm_params.model = model
    mc.litellm_params.weight = weight
    return mc


class TestPreflightAuthCheck:
    """Tests for preflight_auth_check function."""

    async def test_checks_gemini_cli_provider(self) -> None:
        """Checks auth for gemini-cli models with weight > 0."""
        configs = [_make_model_config("gemini-cli/gemini-2.5-pro", weight=1)]

        mock_status = AuthStatus(
            provider="gemini-cli",
            authenticated=True,
            user="test@example.com",
            expires_at=None,
            error=None,
        )

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock(return_value=mock_status)
            results = await preflight_auth_check(configs)

        assert len(results) == 1
        assert results[0].provider == "gemini-cli"
        assert results[0].authenticated

    async def test_checks_chatgpt_provider(self) -> None:
        """Checks auth for chatgpt models with weight > 0."""
        configs = [_make_model_config("chatgpt/codex-mini", weight=1)]

        mock_status = AuthStatus(
            provider="chatgpt",
            authenticated=False,
            user=None,
            expires_at=None,
            error="Auth file not found",
        )

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock(return_value=mock_status)
            results = await preflight_auth_check(configs)

        assert len(results) == 1
        assert results[0].provider == "chatgpt"
        assert not results[0].authenticated

    async def test_skips_weight_zero_models(self) -> None:
        """Models with weight=0 are not checked."""
        configs = [
            _make_model_config("gemini-cli/gemini-2.5-pro", weight=0),
            _make_model_config("chatgpt/codex-mini", weight=0),
        ]

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock()
            results = await preflight_auth_check(configs)

        assert results == []
        instance.check_auth.assert_not_called()

    async def test_skips_non_local_providers(self) -> None:
        """Non-local providers (openai, gemini, etc.) are not checked."""
        configs = [
            _make_model_config("openai/gpt-4o", weight=1),
            _make_model_config("gemini/gemini-2.5-flash", weight=1),
        ]

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock()
            results = await preflight_auth_check(configs)

        assert results == []
        instance.check_auth.assert_not_called()

    async def test_checks_copilot_provider(self) -> None:
        """Checks auth for copilot models with weight > 0."""
        configs = [_make_model_config("copilot/gpt-4.1", weight=1)]

        mock_status = AuthStatus(
            provider="copilot",
            authenticated=True,
            user="github-user",
            expires_at=None,
            error=None,
        )

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock(return_value=mock_status)
            results = await preflight_auth_check(configs)

        assert len(results) == 1
        assert results[0].provider == "copilot"
        assert results[0].authenticated

    async def test_logs_copilot_credentials_detected_without_runtime_verification(
        self,
    ) -> None:
        """Copilot preflight should not overstate config-only auth as runtime-ready."""
        configs = [_make_model_config("copilot/gpt-4.1", weight=1)]

        mock_status = AuthStatus(
            provider="copilot",
            authenticated=True,
            user="github-user",
            expires_at=None,
            error=None,
            details={"source": "config", "verification": "credentials-only"},
        )

        with (
            patch("markitai.providers.AuthManager") as MockManager,
            patch("markitai.providers.logger.debug") as mock_debug,
        ):
            instance = MockManager.return_value
            instance.check_auth = AsyncMock(return_value=mock_status)
            results = await preflight_auth_check(configs)

        assert len(results) == 1
        mock_debug.assert_called_once_with(
            "[Preflight] copilot credentials detected for github-user "
            "(runtime capability not verified)"
        )

    async def test_checks_claude_agent_provider(self) -> None:
        """Checks auth for claude-agent models with weight > 0."""
        configs = [_make_model_config("claude-agent/sonnet", weight=1)]

        mock_status = AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user="claude-user",
            expires_at=None,
            error=None,
        )

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock(return_value=mock_status)
            results = await preflight_auth_check(configs)

        assert len(results) == 1
        assert results[0].provider == "claude-agent"
        assert results[0].authenticated

    async def test_checks_all_local_providers(self) -> None:
        """All local providers checked when configured with weight > 0."""
        configs = [
            _make_model_config("gemini-cli/gemini-2.5-pro", weight=1),
            _make_model_config("chatgpt/codex-mini", weight=1),
            _make_model_config("claude-agent/sonnet", weight=1),
            _make_model_config("copilot/gpt-4.1", weight=1),
            _make_model_config("openai/gpt-4o", weight=1),  # skipped
        ]

        async def fake_check_auth(
            provider: str, force_refresh: bool = False
        ) -> AuthStatus:
            return AuthStatus(
                provider=provider,
                authenticated=True,
                user="test",
                expires_at=None,
                error=None,
            )

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock(side_effect=fake_check_auth)
            results = await preflight_auth_check(configs)

        assert len(results) == 4
        providers_checked = {r.provider for r in results}
        assert providers_checked == {"gemini-cli", "chatgpt", "claude-agent", "copilot"}

    async def test_deduplicates_same_provider(self) -> None:
        """Multiple models from same provider only trigger one check."""
        configs = [
            _make_model_config("gemini-cli/gemini-2.5-pro", weight=1),
            _make_model_config("gemini-cli/gemini-2.5-flash", weight=2),
        ]

        mock_status = AuthStatus(
            provider="gemini-cli",
            authenticated=True,
            user="test",
            expires_at=None,
            error=None,
        )

        with patch("markitai.providers.AuthManager") as MockManager:
            instance = MockManager.return_value
            instance.check_auth = AsyncMock(return_value=mock_status)
            results = await preflight_auth_check(configs)

        assert len(results) == 1
        instance.check_auth.assert_called_once()

    async def test_empty_model_list(self) -> None:
        """Empty model list returns empty results."""
        results = await preflight_auth_check([])
        assert results == []
