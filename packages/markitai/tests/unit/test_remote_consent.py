"""Tests for remote-fetch consent (fetch.remote_consent / MARKITAI_NO_REMOTE_FETCH)."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from markitai.config import FetchConfig
from markitai.fetch import (
    FetchError,
    FetchResult,
    _fetch_with_fallback,
    reset_remote_consent,
    resolve_remote_consent,
    set_remote_consent,
    set_remote_consent_prompt_allowed,
)

VALID_CONTENT = (
    "# Article\n\n"
    "This is valid content with enough text to pass the content validation "
    "threshold which requires at least 100 characters of clean text after "
    "removing all markdown syntax elements like headers, links, and images."
)


class TestDefaultConsentBehavior:
    """Default policy: public URLs go to remote services without a prompt.

    Private/local/credentialed URLs never reach the consent gate at all —
    is_private_or_local_domain() strips remote strategies from the chain
    before it (see fetch.py). So the remaining consent question only
    concerns public URLs, which default to allowed with a one-time
    disclosure log instead of an interactive prompt.
    """

    def test_default_is_always(self) -> None:
        assert FetchConfig().remote_consent == "always"

    def test_default_allows_remote_without_prompt(self) -> None:
        with patch("click.confirm") as mock_confirm:
            assert resolve_remote_consent(FetchConfig()) is True
            mock_confirm.assert_not_called()

    def test_always_logs_disclosure_once_per_process(self) -> None:
        """The first remote use logs an INFO disclosure; later calls don't."""
        from loguru import logger

        messages: list[str] = []
        sink_id = logger.add(lambda m: messages.append(str(m)), level="INFO")
        try:
            assert resolve_remote_consent(FetchConfig()) is True
            assert resolve_remote_consent(FetchConfig()) is True
        finally:
            logger.remove(sink_id)
        disclosures = [m for m in messages if "remote extraction" in m.lower()]
        assert len(disclosures) == 1, (
            f"expected exactly one disclosure log, got: {disclosures}"
        )


class TestAskPromptWording:
    """The interactive prompt must explain sequencing and name the actual
    services in the chain — not read like a simultaneous broadcast."""

    def _run_prompt(self, services: list[str]) -> str:
        config = FetchConfig(remote_consent="ask")
        texts: list[str] = []

        def fake_confirm(text: str, *args: object, **kwargs: object) -> bool:
            texts.append(text)
            return True

        def fake_echo(text: str = "", *args: object, **kwargs: object) -> None:
            texts.append(str(text))

        with (
            patch("sys.stdin") as mock_stdin,
            patch("click.confirm", side_effect=fake_confirm),
            patch("click.echo", side_effect=fake_echo),
        ):
            mock_stdin.isatty.return_value = True
            assert resolve_remote_consent(config, services=services) is True
        return " ".join(texts)

    def test_names_chain_services_and_sequencing(self) -> None:
        text = self._run_prompt(["defuddle", "jina"])
        assert "defuddle.md" in text
        assert "Jina" in text
        assert "one at a time" in text
        assert "Cloudflare" not in text, "services not in the chain must not be listed"

    def test_includes_cloudflare_when_in_chain(self) -> None:
        text = self._run_prompt(["defuddle", "jina", "cloudflare"])
        assert "Cloudflare (your account)" in text


class TestResolveRemoteConsent:
    """Tests for resolve_remote_consent decision logic."""

    def test_always_grants_consent(self) -> None:
        config = FetchConfig(remote_consent="always")
        assert resolve_remote_consent(config) is True

    def test_never_denies_consent(self) -> None:
        config = FetchConfig(remote_consent="never")
        assert resolve_remote_consent(config) is False

    def test_ask_non_interactive_denies_consent(self) -> None:
        config = FetchConfig(remote_consent="ask")
        with patch("sys.stdin") as mock_stdin:
            mock_stdin.isatty.return_value = False
            assert resolve_remote_consent(config) is False

    def test_env_var_overrides_always(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MARKITAI_NO_REMOTE_FETCH", "1")
        config = FetchConfig(remote_consent="always")
        assert resolve_remote_consent(config) is False

    @pytest.mark.parametrize("value", ["0", "false", "no", "", "  "])
    def test_env_var_falsy_values_ignored(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("MARKITAI_NO_REMOTE_FETCH", value)
        config = FetchConfig(remote_consent="always")
        assert resolve_remote_consent(config) is True

    def test_ask_interactive_prompts_once_and_caches_yes(self) -> None:
        config = FetchConfig(remote_consent="ask")
        with (
            patch("sys.stdin") as mock_stdin,
            patch("click.confirm", return_value=True) as mock_confirm,
        ):
            mock_stdin.isatty.return_value = True
            assert resolve_remote_consent(config) is True
            # Second call must reuse the cached decision (once per process)
            assert resolve_remote_consent(config) is True
            mock_confirm.assert_called_once()

    def test_ask_interactive_default_no_denies(self) -> None:
        config = FetchConfig(remote_consent="ask")
        with (
            patch("sys.stdin") as mock_stdin,
            patch("click.confirm", return_value=False) as mock_confirm,
        ):
            mock_stdin.isatty.return_value = True
            assert resolve_remote_consent(config) is False
            assert resolve_remote_consent(config) is False
            mock_confirm.assert_called_once()

    def test_prompt_disallowed_skips_prompt_even_on_tty(self) -> None:
        """--quiet disables the prompt: no consent, no interaction."""
        config = FetchConfig(remote_consent="ask")
        set_remote_consent_prompt_allowed(False)
        with (
            patch("sys.stdin") as mock_stdin,
            patch("click.confirm") as mock_confirm,
        ):
            mock_stdin.isatty.return_value = True
            assert resolve_remote_consent(config) is False
            mock_confirm.assert_not_called()

    def test_seeded_consent_skips_prompt(self) -> None:
        """Explicit -s defuddle/jina/cloudflare seeds consent for the run."""
        config = FetchConfig()
        set_remote_consent(True)
        with patch("click.confirm") as mock_confirm:
            assert resolve_remote_consent(config) is True
            mock_confirm.assert_not_called()

    def test_reset_clears_cached_decision(self) -> None:
        set_remote_consent(True)
        config = FetchConfig(remote_consent="never")
        assert resolve_remote_consent(config) is True
        reset_remote_consent()
        assert resolve_remote_consent(config) is False

    def test_never_decision_cached_for_process(self) -> None:
        """The decision is resolved once; later config changes don't apply."""
        assert resolve_remote_consent(FetchConfig(remote_consent="never")) is False
        assert resolve_remote_consent(FetchConfig(remote_consent="always")) is False

    def test_prompt_suspends_active_live_display(self) -> None:
        """The consent prompt must pause an active StageList Live.

        With the Live running, rich proxies sys.stderr; click's prompt then
        loses its "[y/N]" suffix (FileProxy.flush prints with markup enabled)
        and the Enter echo desyncs Live's cursor so spinner frames stack.
        """
        import io
        import sys

        from rich.console import Console
        from rich.file_proxy import FileProxy

        from markitai.cli import ui

        stages = ui.StageList(
            enabled=True,
            transient=False,
            console=Console(file=io.StringIO(), force_terminal=True, width=80),
        )
        stages.start()
        seen: dict[str, bool] = {}

        def fake_confirm(*args: object, **kwargs: object) -> bool:
            seen["live_started"] = bool(stages._live and stages._live.is_started)
            seen["stderr_proxied"] = isinstance(sys.stderr, FileProxy)
            return True

        try:
            stages.advance("render", "Rendering...")
            with (
                patch("sys.stdin") as mock_stdin,
                patch("click.confirm", side_effect=fake_confirm),
            ):
                mock_stdin.isatty.return_value = True
                assert resolve_remote_consent(FetchConfig(remote_consent="ask")) is True

            assert seen["live_started"] is False, (
                "Live must be suspended while the consent prompt is shown"
            )
            assert seen["stderr_proxied"] is False, (
                "the prompt must write to the real stderr, not rich's proxy"
            )
            assert stages._live is not None and stages._live.is_started, (
                "Live must resume after the prompt"
            )
        finally:
            stages.stop()


class TestFallbackChainConsentGate:
    """Tests for the consent gate in the auto fallback chain."""

    def _static_result(self, url: str) -> FetchResult:
        return FetchResult(content=VALID_CONTENT, strategy_used="static", url=url)

    @pytest.mark.asyncio
    async def test_no_consent_skips_remote_strategies(self) -> None:
        """ask + non-interactive: defuddle/jina never receive the URL."""
        url = "https://example.com/article"
        config = FetchConfig()  # remote_consent defaults to "ask"

        with (
            patch("sys.stdin") as mock_stdin,
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                return_value=self._static_result(url),
            ) as mock_static,
            patch(
                "markitai.fetch.fetch_with_defuddle", new_callable=AsyncMock
            ) as mock_defuddle,
            patch(
                "markitai.fetch.fetch_with_jina", new_callable=AsyncMock
            ) as mock_jina,
        ):
            mock_stdin.isatty.return_value = False
            result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "static"
        mock_static.assert_called_once()
        mock_defuddle.assert_not_called()
        mock_jina.assert_not_called()

    @pytest.mark.asyncio
    async def test_interactive_ask_no_prompt_when_local_succeeds(self) -> None:
        """Lazy consent: with ask + interactive TTY, a fetch satisfied by
        the local-first chain (static wins) must never prompt the user."""
        url = "https://example.com/article"
        config = FetchConfig()  # remote_consent defaults to "ask"

        def _fail_if_prompted(*args: object, **kwargs: object) -> bool:
            raise AssertionError("consent prompt fired despite static success")

        with (
            patch("sys.stdin") as mock_stdin,
            patch("click.confirm", side_effect=_fail_if_prompted),
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                return_value=self._static_result(url),
            ) as mock_static,
            patch(
                "markitai.fetch.fetch_with_defuddle", new_callable=AsyncMock
            ) as mock_defuddle,
        ):
            mock_stdin.isatty.return_value = True
            result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "static"
        mock_static.assert_called_once()
        mock_defuddle.assert_not_called()

    @pytest.mark.asyncio
    async def test_always_uses_remote_strategies(self) -> None:
        """remote_consent=always keeps remote strategies in the chain: when
        the local-first strategies fail, defuddle runs without any prompt."""
        url = "https://example.com/article"
        config = FetchConfig(remote_consent="always")
        defuddle_result = FetchResult(
            content=VALID_CONTENT, strategy_used="defuddle", url=url
        )

        with (
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                side_effect=FetchError("static failed"),
            ),
            patch(
                "markitai.fetch_playwright.is_playwright_available",
                return_value=False,
            ),
            patch(
                "markitai.fetch.fetch_with_defuddle",
                new_callable=AsyncMock,
                return_value=defuddle_result,
            ) as mock_defuddle,
        ):
            result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "defuddle"
        mock_defuddle.assert_called_once()

    @pytest.mark.asyncio
    async def test_env_var_skips_remote_despite_always(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        url = "https://example.com/article"
        monkeypatch.setenv("MARKITAI_NO_REMOTE_FETCH", "1")
        config = FetchConfig(remote_consent="always")

        with (
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                return_value=self._static_result(url),
            ),
            patch(
                "markitai.fetch.fetch_with_defuddle", new_callable=AsyncMock
            ) as mock_defuddle,
        ):
            result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "static"
        mock_defuddle.assert_not_called()

    @pytest.mark.asyncio
    async def test_explicit_remote_strategy_bypasses_gate(self) -> None:
        """An explicit remote strategy counts as consent (no prompt, no skip)."""
        url = "https://example.com/article"
        config = FetchConfig(strategy="jina")  # remote_consent stays "ask"
        jina_result = FetchResult(content=VALID_CONTENT, strategy_used="jina", url=url)

        with (
            patch("sys.stdin") as mock_stdin,
            patch(
                "markitai.fetch.fetch_with_jina",
                new_callable=AsyncMock,
                return_value=jina_result,
            ) as mock_jina,
            patch("click.confirm") as mock_confirm,
        ):
            mock_stdin.isatty.return_value = True
            result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "jina"
        mock_jina.assert_called_once()
        mock_confirm.assert_not_called()

    @pytest.mark.asyncio
    async def test_all_remote_priority_falls_back_to_local(self) -> None:
        """If the configured chain is remote-only, denial falls back to local."""
        url = "https://example.com/article"
        config = FetchConfig(remote_consent="never")
        config.policy.strategy_priority = ["defuddle", "jina"]

        with (
            patch(
                "markitai.fetch.fetch_with_static",
                new_callable=AsyncMock,
                return_value=self._static_result(url),
            ) as mock_static,
            patch(
                "markitai.fetch.fetch_with_defuddle", new_callable=AsyncMock
            ) as mock_defuddle,
        ):
            result = await _fetch_with_fallback(url, config)

        assert result.strategy_used == "static"
        mock_static.assert_called_once()
        mock_defuddle.assert_not_called()


class TestRemoteConsentConfig:
    """Tests for the fetch.remote_consent config field."""

    def test_invalid_value_rejected(self) -> None:
        with pytest.raises(ValueError):
            FetchConfig(remote_consent="sometimes")  # type: ignore[arg-type]
