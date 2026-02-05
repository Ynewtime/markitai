"""Unit tests for cache CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from markitai.cli.commands.cache import (
    cache,
    cache_clear,
    cache_spa_domains,
    cache_stats,
    format_size,
)


class TestFormatSize:
    """Tests for format_size function."""

    @pytest.mark.parametrize(
        "bytes_val,expected",
        [
            (0, "0 B"),
            (1, "1 B"),
            (512, "512 B"),
            (1023, "1023 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1024 * 1024, "1.00 MB"),
            (1024 * 1024 * 10, "10.00 MB"),
            (1024 * 1024 * 1024, "1024.00 MB"),
        ],
    )
    def test_format_size(self, bytes_val: int, expected: str) -> None:
        """Test format_size with various byte values."""
        assert format_size(bytes_val) == expected

    def test_format_size_boundary_kb(self) -> None:
        """Test KB boundary (1024 bytes)."""
        assert format_size(1023) == "1023 B"
        assert format_size(1024) == "1.0 KB"

    def test_format_size_boundary_mb(self) -> None:
        """Test MB boundary (1024 * 1024 bytes)."""
        assert format_size(1024 * 1024 - 1) == "1024.0 KB"
        assert format_size(1024 * 1024) == "1.00 MB"


class TestCacheStatsUnifiedUI:
    """Tests for cache stats unified UI."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_cache_stats_unified_ui(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test cache stats uses unified UI."""
        mock_cfg = MagicMock()
        mock_cfg.cache.enabled = True
        mock_cfg.cache.global_dir = str(tmp_path)
        mock_cfg.cache.max_size_bytes = 100 * 1024 * 1024

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_cfg

            result = runner.invoke(cache, ["stats"])

            assert result.exit_code == 0
            assert "\u25c6" in result.output  # Title marker (diamond)


class TestCacheStatsCommand:
    """Tests for cache stats CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.cache.enabled = True
        config.cache.global_dir = str(tmp_path)
        config.cache.max_size_bytes = 100 * 1024 * 1024
        return config

    def test_stats_no_cache_exists(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats when no cache exists."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats)

            assert result.exit_code == 0
            # Now uses unified UI with title marker
            assert "\u25c6" in result.output  # Title marker (diamond)

    def test_stats_json_format(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats with JSON output format."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats, ["--json"])

            assert result.exit_code == 0
            # Should be valid JSON
            data = json.loads(result.output)
            assert "enabled" in data
            assert data["enabled"] is True

    def test_stats_with_existing_cache(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats with existing cache."""
        from markitai.constants import DEFAULT_CACHE_DB_FILENAME
        from markitai.llm import SQLiteCache

        # Create a cache with some entries
        cache_path = tmp_path / DEFAULT_CACHE_DB_FILENAME
        cache_obj = SQLiteCache(cache_path, mock_config.cache.max_size_bytes)
        cache_obj.set("test_key", "test_content", "test_response", model="test-model")
        # SQLiteCache uses connection context managers, no explicit close needed

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats)

            assert result.exit_code == 0
            # Now uses unified UI with bullet points for info
            assert "\u2022" in result.output  # Info marker (bullet)


class TestCacheClearCommand:
    """Tests for cache clear CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.cache.enabled = True
        config.cache.global_dir = str(tmp_path)
        config.cache.max_size_bytes = 100 * 1024 * 1024
        return config

    def test_clear_aborted_without_yes(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test clear is aborted when user doesn't confirm."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_clear, input="n\n")

            assert result.exit_code == 0
            assert "Aborted" in result.output

    def test_clear_with_yes_flag(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test clear with --yes flag skips confirmation."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_clear, ["--yes"])

            assert result.exit_code == 0
            # Either cleared something or nothing to clear (supports both en/zh)
            assert (
                "Cleared" in result.output
                or "No cache entries" in result.output
                or "已清理" in result.output
                or "无缓存可清理" in result.output
            )


class TestCacheSpaDomainsCommand:
    """Tests for cache spa-domains CLI command."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_spa_domains_empty(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test spa-domains with no learned domains."""
        with patch("markitai.fetch.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.list_domains.return_value = []
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains)

            assert result.exit_code == 0
            assert "No learned SPA domains" in result.output

    def test_spa_domains_json_format(self, runner: CliRunner) -> None:
        """Test spa-domains with JSON output."""
        with patch("markitai.fetch.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.list_domains.return_value = [
                {"domain": "example.com", "hits": 5, "expired": False}
            ]
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains, ["--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert len(data) == 1
            assert data[0]["domain"] == "example.com"

    def test_spa_domains_clear(self, runner: CliRunner) -> None:
        """Test spa-domains --clear."""
        with patch("markitai.fetch.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.clear.return_value = 3
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains, ["--clear"])

            assert result.exit_code == 0
            assert "Cleared 3" in result.output
            mock_cache.clear.assert_called_once()

    def test_spa_domains_clear_json_format(self, runner: CliRunner) -> None:
        """Test spa-domains --clear with JSON output."""
        with patch("markitai.fetch.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.clear.return_value = 5
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains, ["--clear", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert data["cleared"] == 5

    def test_spa_domains_with_entries(self, runner: CliRunner) -> None:
        """Test spa-domains with multiple entries using unified UI."""
        with patch("markitai.fetch.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.list_domains.return_value = [
                {
                    "domain": "twitter.com",
                    "hits": 10,
                    "learned_at": "2026-01-15T10:00:00",
                    "last_hit": "2026-01-20T15:30:00",
                    "expired": False,
                },
                {
                    "domain": "old-spa.com",
                    "hits": 2,
                    "learned_at": "2025-11-01T08:00:00",
                    "last_hit": "2025-11-05T09:00:00",
                    "expired": True,
                },
            ]
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains)

            assert result.exit_code == 0
            assert "Learned SPA Domains" in result.output
            assert "2 total" in result.output or "2 总计" in result.output
            assert "twitter.com" in result.output
            assert "old-spa.com" in result.output
            # Now uses unified UI with checkmark for active and warning for expired
            assert "\u2713" in result.output  # Success checkmark for active
            assert "!" in result.output  # Warning for expired

    def test_spa_domains_with_missing_fields(self, runner: CliRunner) -> None:
        """Test spa-domains handles entries with missing optional fields."""
        with patch("markitai.fetch.get_spa_domain_cache") as mock_get_cache:
            mock_cache = MagicMock()
            mock_cache.list_domains.return_value = [
                {
                    "domain": "minimal.com",
                    "hits": 1,
                    "expired": False,
                    # Missing learned_at and last_hit
                },
            ]
            mock_get_cache.return_value = mock_cache

            result = runner.invoke(cache_spa_domains)

            assert result.exit_code == 0
            assert "minimal.com" in result.output


class TestCacheStatsVerboseMode:
    """Tests for cache stats verbose mode."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.cache.enabled = True
        config.cache.global_dir = str(tmp_path)
        config.cache.max_size_bytes = 100 * 1024 * 1024
        return config

    def test_stats_verbose_with_cache(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats --verbose with existing cache entries."""
        from markitai.constants import DEFAULT_CACHE_DB_FILENAME
        from markitai.llm import SQLiteCache

        # Create a cache with multiple entries
        cache_path = tmp_path / DEFAULT_CACHE_DB_FILENAME
        cache = SQLiteCache(cache_path, mock_config.cache.max_size_bytes)
        cache.set("prompt1", "content1", '{"title": "Test 1"}', model="gpt-4")
        cache.set("prompt2", "content2", '{"title": "Test 2"}', model="gpt-4")
        cache.set("prompt3", "content3", '{"title": "Test 3"}', model="claude-3")

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats, ["--verbose"])

            assert result.exit_code == 0
            # Should show model breakdown and entries

    def test_stats_verbose_json_format(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats --verbose --json includes detailed data."""
        from markitai.constants import DEFAULT_CACHE_DB_FILENAME
        from markitai.llm import SQLiteCache

        cache_path = tmp_path / DEFAULT_CACHE_DB_FILENAME
        cache = SQLiteCache(cache_path, mock_config.cache.max_size_bytes)
        cache.set("prompt1", "content1", '{"data": "test"}', model="gpt-4")

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats, ["--verbose", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            assert "cache" in data
            if data["cache"]:
                assert "by_model" in data["cache"]
                assert "entries" in data["cache"]

    def test_stats_verbose_limit_option(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test stats --verbose --limit controls number of entries shown."""
        from markitai.constants import DEFAULT_CACHE_DB_FILENAME
        from markitai.llm import SQLiteCache

        cache_path = tmp_path / DEFAULT_CACHE_DB_FILENAME
        cache = SQLiteCache(cache_path, mock_config.cache.max_size_bytes)
        # Create more entries than limit
        for i in range(10):
            cache.set(f"prompt{i}", f"content{i}", f'{{"data": "{i}"}}', model="test")

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats, ["--verbose", "--limit", "5", "--json"])

            assert result.exit_code == 0
            data = json.loads(result.output)
            if data["cache"] and "entries" in data["cache"]:
                assert len(data["cache"]["entries"]) <= 5


class TestCacheStatsErrorHandling:
    """Tests for cache stats error handling."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    def test_stats_cache_error(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test stats handles cache initialization errors gracefully."""
        mock_config = MagicMock()
        mock_config.cache.enabled = True
        mock_config.cache.global_dir = str(tmp_path)
        mock_config.cache.max_size_bytes = 100 * 1024 * 1024

        # Create a corrupted cache file
        from markitai.constants import DEFAULT_CACHE_DB_FILENAME

        cache_path = tmp_path / DEFAULT_CACHE_DB_FILENAME
        cache_path.write_text("not a valid sqlite database")

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats)

            # Should not crash, might show error
            assert result.exit_code == 0

    def test_stats_disabled_cache(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test stats when cache is disabled."""
        mock_config = MagicMock()
        mock_config.cache.enabled = False
        mock_config.cache.global_dir = str(tmp_path)
        mock_config.cache.max_size_bytes = 100 * 1024 * 1024

        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_stats)

            assert result.exit_code == 0
            # Should indicate cache is disabled or not found


class TestCacheClearWithSpaDomains:
    """Tests for cache clear with SPA domains option."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_config(self, tmp_path: Path) -> MagicMock:
        """Create a mock config."""
        config = MagicMock()
        config.cache.enabled = True
        config.cache.global_dir = str(tmp_path)
        config.cache.max_size_bytes = 100 * 1024 * 1024
        return config

    def test_clear_with_spa_domains(
        self, runner: CliRunner, mock_config: MagicMock, tmp_path: Path
    ) -> None:
        """Test clear --include-spa-domains clears both cache and SPA domains."""
        from markitai.constants import DEFAULT_CACHE_DB_FILENAME
        from markitai.llm import SQLiteCache

        # Create a cache with entries
        cache_path = tmp_path / DEFAULT_CACHE_DB_FILENAME
        cache = SQLiteCache(cache_path, mock_config.cache.max_size_bytes)
        cache.set("prompt", "content", "response", model="test")

        with (
            patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager,
            patch("markitai.fetch.get_spa_domain_cache") as mock_spa,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_spa_cache = MagicMock()
            mock_spa_cache.clear.return_value = 3
            mock_spa.return_value = mock_spa_cache

            result = runner.invoke(cache_clear, ["--yes", "--include-spa-domains"])

            assert result.exit_code == 0
            assert "Cleared" in result.output or "已清理" in result.output
            mock_spa_cache.clear.assert_called_once()

    def test_clear_confirmation_includes_spa_mention(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test clear --include-spa-domains confirmation mentions SPA domains."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_clear, ["--include-spa-domains"], input="n\n")

            assert result.exit_code == 0
            assert "SPA domains" in result.output
            assert "Aborted" in result.output

    def test_clear_with_confirmed_input(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test clear proceeds when user confirms."""
        with patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager:
            MockConfigManager.return_value.load.return_value = mock_config

            result = runner.invoke(cache_clear, input="y\n")

            assert result.exit_code == 0
            # Should proceed with clearing

    def test_clear_spa_domains_error_handling(
        self, runner: CliRunner, mock_config: MagicMock
    ) -> None:
        """Test clear handles SPA domain clearing errors gracefully."""
        with (
            patch("markitai.cli.commands.cache.ConfigManager") as MockConfigManager,
            patch("markitai.fetch.get_spa_domain_cache") as mock_spa,
        ):
            MockConfigManager.return_value.load.return_value = mock_config
            mock_spa.side_effect = Exception("SPA cache error")

            result = runner.invoke(cache_clear, ["--yes", "--include-spa-domains"])

            assert result.exit_code == 0
            assert "Failed to clear SPA domains" in result.output
