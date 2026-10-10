"""Claude Code runs on the subscription, never silently on ANTHROPIC_API_KEY.

In non-interactive mode, which the Agent SDK uses, Claude Code sends
ANTHROPIC_API_KEY whenever it is set, even with a Claude subscription signed
in (checked against Claude Code 2.1.293 with a local stub server: x-api-key
with the variable set, the OAuth bearer token with it empty). With all
provider keys loaded, every claude-agent/ request was billed per token.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from markitai.providers import auth


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in (
        "ANTHROPIC_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(auth.sys, "platform", "linux")  # no macOS Keychain probe
    return tmp_path


def _sign_in(home: Path) -> None:
    creds = home / ".claude" / ".credentials.json"
    creds.parent.mkdir(parents=True)
    creds.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "fake",
                    "refreshToken": "fake",
                    "expiresAt": 4102444800000,
                    "subscriptionType": "max",
                }
            }
        ),
        encoding="utf-8",
    )


class TestClaudeCodeEnv:
    def test_subscription_and_key_hides_the_key(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _sign_in(home)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        assert auth.claude_code_env() == {"ANTHROPIC_API_KEY": ""}

    def test_setup_token_counts_as_a_subscription(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "fake")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        assert auth.claude_code_env() == {"ANTHROPIC_API_KEY": ""}

    def test_key_only_keeps_the_key(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a login the key is Claude Code's only credential."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        assert auth.claude_code_env() == {}

    def test_no_key_changes_nothing(self, home: Path) -> None:
        _sign_in(home)
        assert auth.claude_code_env() == {}


class TestClaudeAuthStatus:
    def test_key_only_is_authenticated_but_billed_per_token(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        with patch.object(
            auth, "_claude_cli_auth_status", return_value={"loggedIn": False}
        ):
            status = auth._check_claude_credentials_auth()
        assert status.authenticated
        assert status.details == {"source": "anthropic-api-key", "billing": "api_key"}

    def test_subscription_file_reports_the_plan(self, home: Path) -> None:
        _sign_in(home)
        with patch.object(auth, "_claude_cli_email", return_value=None):
            status = auth._check_claude_credentials_auth()
        assert status.details is not None
        assert status.details["billing"] == "subscription"
        assert status.details["subscription"] == "max"

    def test_cli_status_is_asked_without_the_key(
        self, home: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`claude auth status` reports the key, not the login, while it is set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
        run = MagicMock(return_value=MagicMock(returncode=0, stdout="{}"))
        with (
            patch.object(auth, "_resolve_cli_path", return_value="/usr/bin/claude"),
            patch("subprocess.run", run),
        ):
            auth._claude_cli_auth_status()
        assert "ANTHROPIC_API_KEY" not in run.call_args.kwargs["env"]


class TestDetection:
    def test_key_only_claude_code_is_not_detected(self) -> None:
        """It is the Anthropic API (anthropic/), not a subscription."""
        from markitai.providers import detect

        status = auth.AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user="ANTHROPIC_API_KEY",
            expires_at=None,
            error=None,
            details={"source": "anthropic-api-key", "billing": "api_key"},
        )
        with patch.object(auth.AuthManager, "check_auth", return_value=status):
            assert detect._check_claude_auth() is False

    def test_subscription_is_detected(self) -> None:
        from markitai.providers import detect

        status = auth.AuthStatus(
            provider="claude-agent",
            authenticated=True,
            user="me",
            expires_at=None,
            error=None,
            details={"source": "credentials-file", "billing": "subscription"},
        )
        with patch.object(auth.AuthManager, "check_auth", return_value=status):
            assert detect._check_claude_auth() is True


class TestProviderPassesTheOverride:
    async def test_claude_code_runs_with_the_key_hidden(self) -> None:
        from markitai.providers.claude_agent import ClaudeAgentProvider

        captured: dict[str, Any] = {}

        class FakeOptions:
            def __init__(self, **kwargs: Any) -> None:
                captured.update(kwargs)

        class FakeClient:
            def __init__(self, options: FakeOptions) -> None:
                pass

            async def __aenter__(self) -> FakeClient:
                return self

            async def __aexit__(self, *exc: object) -> None:
                return None

            async def query(self, prompt: object) -> None:
                return None

            async def receive_response(self) -> Any:
                for message in ():
                    yield message

        sdk_types = types.ModuleType("claude_agent_sdk.types")
        for name in ("AssistantMessage", "TextBlock", "ResultMessage"):
            setattr(sdk_types, name, type(name, (), {}))
        sdk = types.ModuleType("claude_agent_sdk")
        sdk.ClaudeAgentOptions = FakeOptions  # type: ignore[attr-defined]
        sdk.ClaudeSDKClient = FakeClient  # type: ignore[attr-defined]
        sdk.types = sdk_types  # type: ignore[attr-defined]

        with (
            patch(
                "markitai.providers.claude_agent._is_claude_agent_sdk_available",
                return_value=True,
            ),
            patch(
                "markitai.providers.claude_agent.claude_code_env",
                return_value={"ANTHROPIC_API_KEY": ""},
            ),
            patch.dict(
                sys.modules,
                {"claude_agent_sdk": sdk, "claude_agent_sdk.types": sdk_types},
            ),
        ):
            await ClaudeAgentProvider().acompletion(
                model="claude-agent/haiku",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert captured["env"] == {"ANTHROPIC_API_KEY": ""}
        assert captured["model"] == "haiku"
