"""Tests for the interaction port (dependency-inversion seam to the CLI).

The port exists so fetch/batch never import markitai.cli: domain code talks
to the user through ``markitai.ports.get_interaction()``, and the CLI injects
a live-display-aware implementation at startup.
"""

from unittest.mock import patch

import pytest

from markitai import ports
from markitai.ports import InteractionPort, StdioInteraction


@pytest.fixture(autouse=True)
def _restore_default_interaction():
    """Keep the process-wide port injection isolated per test."""
    original = ports.get_interaction()
    yield
    ports.set_interaction(original)


class FakeInteraction:
    """Scriptable port double."""

    def __init__(self, *, promptable: bool = True, answer: bool = True) -> None:
        self.promptable = promptable
        self.answer = answer
        self.notices: list[str] = []
        self.questions: list[str] = []

    def can_prompt(self) -> bool:
        return self.promptable

    def notify(self, message: str) -> None:
        self.notices.append(message)

    def confirm(
        self, question: str, *, default: bool = False, preamble: str | None = None
    ) -> bool:
        self.questions.append(question)
        return self.answer


class TestStdioInteraction:
    def test_is_the_default_port(self) -> None:
        assert isinstance(ports.get_interaction(), StdioInteraction)

    def test_satisfies_the_protocol(self) -> None:
        assert isinstance(StdioInteraction(), InteractionPort)
        assert isinstance(FakeInteraction(), InteractionPort)

    def test_can_prompt_follows_stdin_tty(self) -> None:
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = True
            assert StdioInteraction().can_prompt() is True
            mock_stdin.isatty.return_value = False
            assert StdioInteraction().can_prompt() is False

    def test_notify_writes_to_stderr_via_click(self) -> None:
        with patch("click.echo") as mock_echo:
            StdioInteraction().notify("privacy notice")
        mock_echo.assert_called_once_with("privacy notice", err=True)

    def test_confirm_asks_on_stderr_with_preamble(self) -> None:
        with (
            patch("click.confirm", return_value=True) as mock_confirm,
            patch("click.echo") as mock_echo,
        ):
            answer = StdioInteraction().confirm(
                "Proceed?", default=False, preamble="Context first."
            )
        assert answer is True
        mock_echo.assert_called_once_with("Context first.", err=True)
        mock_confirm.assert_called_once_with("Proceed?", default=False, err=True)

    def test_confirm_without_preamble_emits_no_echo(self) -> None:
        with (
            patch("click.confirm", return_value=False),
            patch("click.echo") as mock_echo,
        ):
            answer = StdioInteraction().confirm("Proceed?", default=True)
        assert answer is False
        mock_echo.assert_not_called()


class TestInjection:
    def test_set_interaction_replaces_the_process_port(self) -> None:
        fake = FakeInteraction()
        ports.set_interaction(fake)
        assert ports.get_interaction() is fake


class TestFetchUsesThePort:
    """Prove the dependency really inverted: fetch consults the injected port."""

    def test_resolve_remote_consent_ask_uses_injected_port(self) -> None:
        from markitai.config import FetchConfig
        from markitai.fetch import reset_remote_consent, resolve_remote_consent

        fake = FakeInteraction(promptable=True, answer=True)
        ports.set_interaction(fake)
        reset_remote_consent()
        try:
            config = FetchConfig(remote_consent="ask")
            assert resolve_remote_consent(config, ["jina"]) is True
            assert len(fake.questions) == 1
        finally:
            reset_remote_consent()

    def test_resolve_remote_consent_ask_denied_without_prompt_capability(
        self,
    ) -> None:
        from markitai.config import FetchConfig
        from markitai.fetch import reset_remote_consent, resolve_remote_consent

        fake = FakeInteraction(promptable=False, answer=True)
        ports.set_interaction(fake)
        reset_remote_consent()
        try:
            config = FetchConfig(remote_consent="ask")
            assert resolve_remote_consent(config, ["jina"]) is False
            assert fake.questions == []
        finally:
            reset_remote_consent()

    def test_disclosure_routes_through_notify(self) -> None:
        import markitai.fetch as fetch_mod
        from markitai.fetch import disclose_remote_use

        fake = FakeInteraction()
        ports.set_interaction(fake)
        original = fetch_mod.get_default_session().consent.disclosure_emitted
        fetch_mod.get_default_session().consent.disclosure_emitted = False
        try:
            disclose_remote_use(["jina"])
            assert len(fake.notices) == 1
            assert "remote extraction services" in fake.notices[0]
        finally:
            fetch_mod.get_default_session().consent.disclosure_emitted = original

    def test_explicit_fallback_refusal_prompt_uses_injected_port(self) -> None:
        from markitai.fetch import (
            _should_fallback_after_refusal,
            reset_explicit_fallback_decision,
        )

        fake = FakeInteraction(promptable=True, answer=False)
        ports.set_interaction(fake)
        reset_explicit_fallback_decision()
        try:
            assert _should_fallback_after_refusal("jina", "auth required") is False
            assert len(fake.questions) == 1
        finally:
            reset_explicit_fallback_decision()


class TestConsoleInteraction:
    """The CLI implementation pauses the live display around every touchpoint."""

    def test_confirm_pauses_live_display(self) -> None:
        from markitai.cli.ui import ConsoleInteraction

        with (
            patch("markitai.cli.ui.suspend_active_live") as mock_suspend,
            patch("click.confirm", return_value=True),
        ):
            ConsoleInteraction().confirm("Proceed?", default=False)
        mock_suspend.assert_called_once()

    def test_notify_pauses_live_display(self) -> None:
        from markitai.cli.ui import ConsoleInteraction

        with (
            patch("markitai.cli.ui.suspend_active_live") as mock_suspend,
            patch("click.echo"),
        ):
            ConsoleInteraction().notify("notice")
        mock_suspend.assert_called_once()
