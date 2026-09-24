"""The root ``-c`` / ``--config-json`` apply to subcommands too.

``markitai -c alt.json config get x`` used to read the default config chain,
and ``config set`` wrote to ``~/.markitai/config.json`` instead of alt.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click import unstyle
from click.testing import CliRunner

from markitai.cli.main import app
from markitai.config import ConfigManager


@pytest.fixture
def configs(tmp_path: Path) -> tuple[Path, Path]:
    """A user config and an alternative one with different values."""
    user = ConfigManager.DEFAULT_USER_CONFIG_DIR / "config.json"
    user.parent.mkdir(parents=True, exist_ok=True)
    user.write_text(json.dumps({"output": {"dir": "./home_out"}}), encoding="utf-8")
    alt = tmp_path / "alt.json"
    alt.write_text(
        json.dumps(
            {
                "output": {"dir": "./alt_out"},
                "cache": {"global_dir": str(tmp_path / "alt-cache")},
            }
        ),
        encoding="utf-8",
    )
    return user, alt


def _flat(output: str) -> str:
    """Rich-click's error panel wraps at the terminal width (narrower under
    xdist): drop the borders and line breaks before matching a phrase."""
    return " ".join(unstyle(output).replace("│", " ").split())


def _invoke(*args: str):
    return CliRunner().invoke(app, list(args))


class TestConfigSubcommands:
    def test_get_reads_the_root_config_file(self, configs) -> None:
        _user, alt = configs

        result = _invoke("-c", str(alt), "config", "get", "output.dir")

        assert result.exit_code == 0, result.output
        assert result.output.strip() == "./alt_out"

    def test_get_without_c_keeps_the_default_chain(self, configs) -> None:
        result = _invoke("config", "get", "output.dir")

        assert result.exit_code == 0, result.output
        assert result.output.strip() == "./home_out"

    def test_get_and_list_apply_config_json(self, configs) -> None:
        _user, alt = configs
        overrides = '{"output": {"dir": "inline"}}'

        got = _invoke(
            "-c", str(alt), "--config-json", overrides, "config", "get", "output.dir"
        )
        listed = _invoke("--config-json", overrides, "config", "list")

        assert got.output.strip() == "inline"
        assert '"inline"' in listed.output

    def test_set_writes_the_root_config_file(self, configs) -> None:
        user, alt = configs

        result = _invoke("-c", str(alt), "config", "set", "output.dir", "./changed")

        assert result.exit_code == 0, result.output
        assert json.loads(alt.read_text())["output"]["dir"] == "./changed"
        assert json.loads(user.read_text())["output"]["dir"] == "./home_out"
        # Minimal-diff save: the other keys of alt.json survive.
        assert "global_dir" in json.loads(alt.read_text())["cache"]

    def test_set_refuses_config_json(self, configs) -> None:
        user, alt = configs
        before = (user.read_text(), alt.read_text())

        result = _invoke(
            "-c",
            str(alt),
            "--config-json",
            "{}",
            "config",
            "set",
            "output.dir",
            "./x",
        )

        assert result.exit_code == 2
        assert "cannot be saved" in unstyle(result.output)
        assert (user.read_text(), alt.read_text()) == before

    def test_path_marks_the_cli_source_as_loaded(self, configs) -> None:
        _user, alt = configs

        result = _invoke("-c", str(alt), "config", "path")

        assert result.exit_code == 0, result.output
        lines = unstyle(result.output).splitlines()
        cli_row = next(line for line in lines if "CLI arguments" in line)
        user_row = next(line for line in lines if "~/.markitai/config.json" in line)
        assert "loaded" in cli_row and "-c" in cli_row
        assert "loaded" not in user_row
        assert "alt.json" in result.output

    def test_path_without_c_marks_the_user_file(self, configs) -> None:
        result = _invoke("config", "path")

        lines = unstyle(result.output).splitlines()
        cli_row = next(line for line in lines if "CLI arguments" in line)
        user_row = next(line for line in lines if "~/.markitai/config.json" in line)
        assert "loaded" not in cli_row
        assert "loaded" in user_row

    def test_validate_without_argument_checks_the_root_config(
        self, tmp_path: Path
    ) -> None:
        broken = tmp_path / "broken.json"
        broken.write_text('{"image": {"quality": 500}}', encoding="utf-8")

        result = _invoke("-c", str(broken), "config", "validate")

        assert result.exit_code == 1

    def test_set_creates_the_root_config_file_when_missing(
        self, configs, tmp_path: Path
    ) -> None:
        """``-c new.json config set`` names the file to create.

        ``-c`` used to be ``click.Path(exists=True)``, so this was rejected
        before config set's "write to -c even when it does not exist" path
        could run.
        """
        user, _alt = configs
        new = tmp_path / "fresh" / "new.json"

        result = _invoke("-c", str(new), "config", "set", "output.dir", "./made")

        assert result.exit_code == 0, result.output
        assert json.loads(new.read_text())["output"]["dir"] == "./made"
        assert json.loads(user.read_text())["output"]["dir"] == "./home_out"

    def test_edit_accepts_a_missing_root_config_file(
        self, configs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Path | None] = []
        monkeypatch.setattr("markitai.cli.config_editor.run_config_editor", seen.append)
        new = tmp_path / "new.json"

        result = _invoke("-c", str(new), "config", "edit")

        assert result.exit_code == 0, result.output
        assert seen == [new]

    @pytest.mark.parametrize(
        "args",
        [
            ("config", "get", "output.dir"),
            ("config", "list"),
            ("config", "path"),
            ("config", "validate"),
            ("cache", "stats", "--json"),
        ],
    )
    def test_readers_still_require_an_existing_root_config(
        self, configs, tmp_path: Path, args: tuple[str, ...]
    ) -> None:
        missing = tmp_path / "missing.json"

        result = _invoke("-c", str(missing), *args)

        assert result.exit_code == 2, result.output
        assert "does not exist" in _flat(result.output)
        assert not missing.exists()

    def test_bad_config_json_is_rejected_for_subcommands(self, configs) -> None:
        result = _invoke("--config-json", "[1]", "config", "get", "output.dir")

        assert result.exit_code == 2
        assert "expected a JSON object" in unstyle(result.output)


class TestCacheSubcommands:
    def test_stats_reads_cache_dir_from_the_root_config(
        self, configs, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _user, alt = configs
        seen: list[Path] = []

        import markitai.cli.commands.cache as cache_cmd

        real = cache_cmd._fetch_cache_path

        def spy(cfg):
            seen.append(Path(cfg.cache.global_dir))
            return real(cfg)

        monkeypatch.setattr(cache_cmd, "_fetch_cache_path", spy)

        result = _invoke("-c", str(alt), "cache", "stats", "--json")

        assert result.exit_code == 0, result.output
        assert seen == [tmp_path / "alt-cache"]


class TestConversionConfig:
    def test_conversion_rejects_a_missing_config_file(self, tmp_path: Path) -> None:
        """Only config set/edit may name a missing -c; a conversion must not
        silently fall back to defaults."""
        note = tmp_path / "note.txt"
        note.write_text("hello", encoding="utf-8")
        missing = tmp_path / "missing.json"

        result = _invoke("-c", str(missing), str(note), "-o", str(tmp_path / "out"))

        assert result.exit_code == 2, result.output
        output = _flat(result.output)
        assert "does not exist" in output
        assert "--config" in output
        assert not (tmp_path / "out").exists()
