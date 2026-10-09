"""A crash logged with its traceback must not print variable values.

Loguru's default sink option ``diagnose=True`` prints every frame's local
variables under a traceback, so a crash in a conversion or LLM step would
write the API key and the document text those frames held to stderr and the
log file. Every markitai sink keeps the traceback (frames, source lines, the
exception) without the values, and the log patcher masks credentials in the
exception's own text.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from loguru import logger

# Built at run time so no key-shaped literal sits in the repository. The key
# has no known prefix: only its value in the environment can mask it.
KEY = "markitai-dummy-" + "Z4k8Rw1N" * 4
DOC_TEXT = "CONFIDENTIAL-DOC-TEXT the synthetic invoice listed 4217 blue widgets"

SRC = Path(__file__).resolve().parents[2] / "src" / "markitai"

# Runs the real CLI on one file with the text converter replaced by one that
# crashes while its locals hold the key and the document; the conversion step
# logs that traceback at DEBUG to the log file.
_CLI_DRIVER = """
import os
import runpy
import sys

import markitai.converter.text as text


def convert(self, input_path, output_dir=None):
    api_key = os.environ["OPENAI_API_KEY"]
    document = input_path.read_text(encoding="utf-8")
    raise RuntimeError(f"converter crashed (key {api_key})")


text.TextConverter.convert = convert
sys.argv = ["markitai", *sys.argv[1:]]
runpy.run_module("markitai", run_name="__main__")
"""

# Sets up CLI logging (console + file) and logs an ERROR crash, which reaches
# both stderr and the log file.
_LOG_DRIVER = """
import os
import sys

from loguru import logger

from markitai.cli.logging_config import setup_logging


def call_provider(api_key, document):
    page = document.splitlines()[0]
    raise ValueError(f"request failed for key {api_key}")


def run_step():
    try:
        call_provider(os.environ["OPENAI_API_KEY"], sys.argv[2])
    except ValueError as e:
        raise RuntimeError("step crashed") from e


setup_logging(verbose=False, log_dir=sys.argv[1], log_level="DEBUG")
try:
    run_step()
except RuntimeError:
    logger.exception("[Test] unexpected crash")
"""


def _env() -> dict[str, str]:
    return {
        **os.environ,
        "OPENAI_API_KEY": KEY,
        "NO_COLOR": "1",
        "PYTHON_DOTENV_DISABLED": "1",
        "MARKITAI_LOG_DIR": "",
    }


def _run(script: Path, *args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        capture_output=True,
        text=True,
        env=_env(),
        cwd=cwd,
        timeout=180,
    )


def _assert_no_values(text: str) -> None:
    assert KEY not in text
    assert "Z4k8Rw1N" not in text
    assert "CONFIDENTIAL-DOC-TEXT" not in text
    assert "4217 blue widgets" not in text


def _log_text(log_dir: Path) -> str:
    files = list(log_dir.rglob("*.log"))
    assert files, "no log file written"
    return "\n".join(f.read_text(encoding="utf-8") for f in files)


class TestCrashTracebacks:
    def test_cli_conversion_crash(self, tmp_path: Path) -> None:
        note = tmp_path / "note.md"
        note.write_text(f"# Note\n\n{DOC_TEXT}.\n", encoding="utf-8")
        logs = tmp_path / "logs"
        config = tmp_path / "config.json"
        config.write_text(
            json.dumps(
                {
                    "llm": {"enabled": False},
                    "cache": {"enabled": False},
                    "log": {"dir": str(logs), "level": "DEBUG"},
                }
            ),
            encoding="utf-8",
        )
        script = tmp_path / "cli_driver.py"
        script.write_text(_CLI_DRIVER, encoding="utf-8")

        run = _run(
            script, str(note), "-o", str(tmp_path / "out"), "-c", str(config),
            cwd=tmp_path,
        )  # fmt: skip

        log = _log_text(logs)
        # The traceback itself is still there, with the exception's text
        assert "Traceback" in log
        assert "in convert" in log
        assert "RuntimeError: converter crashed (key [REDACTED])" in log
        for text in (run.stdout, run.stderr, log):
            _assert_no_values(text)

    def test_logged_crash_on_stderr_and_file(self, tmp_path: Path) -> None:
        logs = tmp_path / "logs"
        script = tmp_path / "log_driver.py"
        script.write_text(_LOG_DRIVER, encoding="utf-8")

        run = _run(script, str(logs), f"# Note\n{DOC_TEXT}", cwd=tmp_path)

        log = _log_text(logs)
        for text in (run.stderr, log):
            assert "[Test] unexpected crash" in text
            assert "in call_provider" in text
            assert "ValueError: request failed for key [REDACTED]" in text
            assert "RuntimeError: step crashed" in text
        for text in (run.stdout, run.stderr, log):
            _assert_no_values(text)


def _crash(api_key: str, document: str) -> None:
    page = document.upper()
    raise RuntimeError(f"crashed after {len(page)} characters")


class TestHostSinks:
    """``markitai-mcp`` and ``serve`` set up their own sinks."""

    @pytest.fixture(autouse=True)
    def _reset_logger(self) -> Iterator[None]:
        yield
        logger.remove()

    def _log_crash(self) -> None:
        try:
            _crash(KEY, DOC_TEXT)
        except RuntimeError:
            logger.exception("[Test] host crash")

    def test_mcp_host_logging(self, capsys: pytest.CaptureFixture[str]) -> None:
        from markitai.api import _configure_host_logging

        _configure_host_logging()
        self._log_crash()

        err = capsys.readouterr().err
        assert "RuntimeError: crashed after" in err
        assert "in _crash" in err
        _assert_no_values(err)

    async def test_serve_lifespan_logging(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pytest.importorskip("fastapi")
        from markitai.config import MarkitaiConfig
        from markitai.serve import create_app

        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("OPENAI_API_KEY", KEY)
        cfg = MarkitaiConfig()
        cfg.cache.enabled = False
        cfg.cache.global_dir = str(tmp_path / "cache")
        app = create_app(
            static_dir=tmp_path / "no-static",
            jobs_root=tmp_path / "jobs",
            config=cfg,
            configure_logging=True,
        )
        async with app.router.lifespan_context(app):
            self._log_crash()
            logger.error("[Test] failed with key {}", KEY)

        err = capsys.readouterr().err
        assert "RuntimeError: crashed after" in err
        assert "in _crash" in err
        assert "[Test] failed with key [REDACTED]" in err
        _assert_no_values(err)

    def test_patcher_masks_the_exception_chain(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from markitai.utils.url_redaction import redact_log_record

        logger.remove()
        logger.configure(patcher=redact_log_record)
        logger.add(sys.stderr, format="{message}", diagnose=False)
        # Not literal in the raise lines: the traceback shows their source
        url = "https://alice:" + "s3cret@example.test/x?sig=abc"
        header = "Authorization: Bearer " + "abcdefgh12345678"
        try:
            try:
                raise OSError(f"GET {url}")
            except OSError as inner:
                raise RuntimeError(header) from inner
        except RuntimeError:
            logger.exception("[Test] chained")
        finally:
            logger.configure(patcher=None)

        err = capsys.readouterr().err
        assert "s3cret" not in err and "sig=abc" not in err
        assert "abcdefgh12345678" not in err
        assert "OSError: GET https://example.test/x" in err
        assert "RuntimeError: Authorization: Bearer [REDACTED]" in err


def test_every_sink_hides_variable_values() -> None:
    """Every ``logger.add`` in markitai passes ``diagnose=False``."""
    missing: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                diagnose = next(
                    (k.value for k in node.keywords if k.arg == "diagnose"), None
                )
                if not (isinstance(diagnose, ast.Constant) and diagnose.value is False):
                    missing.append(f"{path.relative_to(SRC)}:{node.lineno}")
    assert not missing, f"logger.add without diagnose=False: {', '.join(missing)}"
