"""Tests for the shared actionable-error guidance helpers."""

from __future__ import annotations

import sys
from pathlib import Path

from markitai.utils.guidance import (
    cloudflare_credentials_error,
    config_location_hint,
    format_actionable_error,
    jina_api_key_hint,
    playwright_browser_missing_error,
    playwright_package_missing_error,
)


class TestConfigLocationHint:
    """Tests for config_location_hint()."""

    def test_uses_loaded_config_path(self, monkeypatch, tmp_path: Path) -> None:
        """When a config file is loaded, its concrete path is shown."""
        from markitai.config import config_manager

        cfg_file = tmp_path / "markitai.json"
        cfg_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(config_manager, "_config_path", cfg_file)

        hint = config_location_hint()
        assert str(cfg_file) in hint
        assert hint.startswith("Config file:")

    def test_resolves_candidates_when_not_loaded(
        self, monkeypatch, tmp_path: Path
    ) -> None:
        """Without a loaded config, the search-chain resolution is used."""
        from markitai.config import ConfigManager, config_manager

        cfg_file = tmp_path / "markitai.json"
        cfg_file.write_text("{}", encoding="utf-8")
        monkeypatch.setattr(config_manager, "_config_path", None)
        monkeypatch.setattr(
            ConfigManager,
            "_resolve_config_path",
            lambda *_a, **_k: cfg_file,
        )

        assert str(cfg_file) in config_location_hint()

    def test_points_to_init_when_no_config_exists(self, monkeypatch) -> None:
        """With no config anywhere, the hint names the candidates and init."""
        from markitai.config import ConfigManager, config_manager

        monkeypatch.setattr(config_manager, "_config_path", None)
        monkeypatch.setattr(
            ConfigManager,
            "_resolve_config_path",
            lambda *_a, **_k: None,
        )

        hint = config_location_hint()
        assert "./markitai.json" in hint
        assert "~/.markitai/config.json" in hint
        assert "markitai init" in hint


class TestFormatActionableError:
    """Tests for format_actionable_error()."""

    def test_numbers_steps_and_indents_continuations(self, monkeypatch) -> None:
        from markitai.config import ConfigManager, config_manager

        monkeypatch.setattr(config_manager, "_config_path", None)
        monkeypatch.setattr(
            ConfigManager,
            "_resolve_config_path",
            lambda *_a, **_k: None,
        )

        block = format_actionable_error(
            "Something is missing.",
            ["first step\ncontinuation line", "second step"],
        )
        lines = block.splitlines()
        assert lines[0] == "Something is missing."
        assert "To fix:" in lines
        assert "  1. first step" in lines
        assert "     continuation line" in lines
        assert "  2. second step" in lines
        # Config hint appended by default
        assert any(line.startswith("Config file:") for line in lines)

    def test_config_hint_can_be_disabled(self) -> None:
        block = format_actionable_error("problem", ["step"], include_config_hint=False)
        assert "Config file:" not in block


class TestCloudflareCredentialsError:
    """The Cloudflare credentials error must be fully actionable."""

    def test_contains_all_required_elements(self) -> None:
        block = cloudflare_credentials_error()
        # Kept for backwards-compatible matching in callers/tests
        assert "Cloudflare API token and account ID required" in block
        # Where to obtain credentials
        assert "https://dash.cloudflare.com/profile/api-tokens" in block
        assert "Browser Rendering" in block
        assert "Workers AI" in block
        # Account ID location
        assert "Account ID" in block
        # Copy-pasteable fix commands
        assert "markitai config set fetch.cloudflare.api_token <token>" in block
        assert "markitai config set fetch.cloudflare.account_id <account-id>" in block
        assert "CLOUDFLARE_API_TOKEN" in block
        assert "CLOUDFLARE_ACCOUNT_ID" in block
        # Config file location line
        assert "Config file:" in block


class TestJinaApiKeyHint:
    def test_contains_config_and_env_commands(self) -> None:
        hint = jina_api_key_hint()
        assert "markitai config set fetch.jina.api_key <key>" in hint
        assert "JINA_API_KEY" in hint
        assert "https://jina.ai/reader" in hint


class TestPlaywrightErrors:
    def test_package_missing_lists_install_paths(self) -> None:
        block = playwright_package_missing_error()
        assert "pip install 'markitai[browser]'" in block
        assert "uv add playwright" in block
        assert "uv tool install --force 'markitai[all]'" in block
        assert "markitai doctor --fix" in block

    def test_browser_missing_explains_two_part_install(self) -> None:
        block = playwright_browser_missing_error()
        # Root cause explanation: package vs browser download
        assert "separate one-time download" in block
        assert "markitai doctor --fix" in block
        # Exact command for the running environment
        assert f'"{sys.executable}" -m playwright install chromium' in block
        # uv tool environment command (verified invocation)
        # The 'uv tool run --from markitai[all]' hint was removed: it triggers
        # a uv warning and resolves an ephemeral env whose playwright version
        # may differ from the tool env; the python -m form targets it exactly
        assert "-m playwright install chromium" in block
        # Dev checkout command (also asserted by fetch_playwright launch test)
        assert "uv run playwright install chromium" in block
