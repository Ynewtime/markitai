"""The CLI with a missing or refused API key, end to end, in a subprocess.

Each run has its own HOME (config in ``~/.markitai/config.json``) and sends
its LLM requests to a loopback stub: ``DEEPSEEK_API_BASE`` points there, so
a request the key check let through would reach the stub, never DeepSeek.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from markitai.constants import PROVIDER_API_KEY_ENV_NAMES
from tests.local_http import LocalHTTPServer

AUTH_FAILS = "Authentication Fails (governor)"
FAKE_KEY = "kv-" + "cli-test-not-a-key-" * 2


class _Stub(BaseHTTPRequestHandler):
    """``/ok/...`` answers a chat completion; any other path is DeepSeek's 401."""

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        self.server.paths.append(self.path)  # type: ignore[attr-defined]
        if self.path.startswith("/ok"):
            status, payload = (
                200,
                {
                    "id": "chatcmpl-stub",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "stub",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {
                                "role": "assistant",
                                "content": "Cleaned by stub.",
                            },
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 5,
                        "completion_tokens": 3,
                        "total_tokens": 8,
                    },
                },
            )
        else:
            status, payload = (
                401,
                {
                    "error": {
                        "message": AUTH_FAILS,
                        "type": "authentication_error",
                        "param": None,
                        "code": None,
                    }
                },
            )
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        pass


@pytest.fixture
def stub() -> Iterator[tuple[str, list[str]]]:
    server = LocalHTTPServer(("127.0.0.1", 0), _Stub)
    server.paths = []  # type: ignore[attr-defined]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server.paths  # type: ignore[attr-defined]
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    inputs = tmp_path / "in"
    inputs.mkdir()
    for name in ("a.md", "b.md"):
        (inputs / name).write_text(
            f"# {name}\n\nA paragraph about markitai.\n", encoding="utf-8"
        )
    return tmp_path


def _write_config(home: Path, *models: dict[str, Any]) -> None:
    config_dir = home / ".markitai"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "llm": {
                    "model_list": [
                        {"model_name": "default", "litellm_params": params}
                        for params in models
                    ]
                },
                "cache": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )


def _run(
    workspace: Path, stub_url: str, *args: str, **env: str
) -> subprocess.CompletedProcess[str]:
    home = workspace / "home"
    base = {
        k: v
        for k, v in os.environ.items()
        if k not in {n for names in PROVIDER_API_KEY_ENV_NAMES.values() for n in names}
        and k != "MODEL"
    }
    return subprocess.run(
        [sys.executable, "-m", "markitai", *args],
        capture_output=True,
        # UTF-8 as the CLI writes it; text=True decodes ANSI on Windows
        encoding="utf-8",
        errors="replace",
        cwd=workspace,
        timeout=180,
        env={
            **base,
            "HOME": str(home),
            "USERPROFILE": str(home),
            "NO_COLOR": "1",
            "PYTHON_DOTENV_DISABLED": "1",
            # Where a DeepSeek request would go if one were sent
            "DEEPSEEK_API_BASE": stub_url,
            **env,
        },
    )


def _lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.strip()]


class TestMissingKey:
    def test_single_file_says_it_once_and_sends_nothing(
        self, workspace: Path, stub: tuple[str, list[str]]
    ) -> None:
        url, paths = stub
        _write_config(workspace / "home", {"model": "deepseek/deepseek-flash"})
        result = _run(workspace, url, "in/a.md", "--llm", "--json", "-o", "out")

        assert result.returncode == 1
        [line] = _lines(result.stderr)
        assert line.startswith("Error: DEEPSEEK_API_KEY is not set")
        assert "DEEPSEEK_API_KEY=your_key" in line
        envelope = json.loads(result.stdout)
        assert envelope["ok"] is False
        assert envelope["error"].startswith("DEEPSEEK_API_KEY is not set")
        assert envelope["items"] == []
        assert paths == []
        assert not (workspace / "out").exists()

    def test_batch_stops_before_converting(
        self, workspace: Path, stub: tuple[str, list[str]]
    ) -> None:
        url, paths = stub
        _write_config(workspace / "home", {"model": "deepseek/deepseek-flash"})
        result = _run(workspace, url, "in", "--llm", "-o", "out")

        assert result.returncode == 1
        [line] = _lines(result.stderr)
        assert line.startswith("Error: DEEPSEEK_API_KEY is not set")
        assert paths == []

    def test_pool_leaves_the_keyless_model_out(
        self, workspace: Path, stub: tuple[str, list[str]]
    ) -> None:
        url, paths = stub
        _write_config(
            workspace / "home",
            {"model": "deepseek/deepseek-flash"},
            {"model": "openrouter/stub/model", "api_base": f"{url}/ok"},
        )
        result = _run(
            workspace,
            url,
            "in",
            "--llm",
            "--pure",
            "-o",
            "out",
            OPENROUTER_API_KEY=FAKE_KEY,
        )

        assert result.returncode == 0, result.stderr
        left_out = [line for line in _lines(result.stderr) if "left out" in line]
        assert len(left_out) == 1
        assert "deepseek/deepseek-flash is left out of the model pool" in left_out[0]
        assert "DEEPSEEK_API_KEY is not set" in left_out[0]
        assert paths and all(path.startswith("/ok") for path in paths)
        assert "Cleaned by stub." in (workspace / "out" / "a.md.llm.md").read_text(
            encoding="utf-8"
        )


class TestRefusedKey:
    def test_single_file_prints_the_failure_once(
        self, workspace: Path, stub: tuple[str, list[str]]
    ) -> None:
        url, paths = stub
        _write_config(workspace / "home", {"model": "deepseek/deepseek-flash"})
        result = _run(
            workspace,
            url,
            "in/a.md",
            "--llm",
            "--json",
            "-o",
            "out",
            DEEPSEEK_API_KEY=FAKE_KEY,
        )

        assert result.returncode == 0, result.stderr
        stderr = result.stderr
        assert stderr.count(AUTH_FAILS) == 1
        assert stderr.count("rejected the API key in DEEPSEEK_API_KEY") == 1
        assert "DEEPSEEK_API_KEY=your_key" in stderr
        for repeated in (
            "Authentication failed for model",
            "LLM processing failed",
            "using fallback frontmatter",
            "llm.on_failure",
        ):
            assert repeated not in stderr
        assert FAKE_KEY not in stderr
        [item] = json.loads(result.stdout)["items"]
        assert item["status"] == "completed"
        [warning] = item["warnings"]
        assert "rejected the API key in DEEPSEEK_API_KEY" in warning
        assert paths == ["/chat/completions"]

    def test_batch_report_carries_the_warning(
        self, workspace: Path, stub: tuple[str, list[str]]
    ) -> None:
        url, paths = stub
        _write_config(workspace / "home", {"model": "deepseek/deepseek-flash"})
        result = _run(
            workspace, url, "in", "--llm", "-o", "out", DEEPSEEK_API_KEY=FAKE_KEY
        )

        assert result.returncode == 0, result.stderr
        assert result.stderr.count(AUTH_FAILS) == 2  # one line per file
        [report] = (workspace / "out" / ".markitai" / "reports").glob("*.json")
        documents = json.loads(report.read_text(encoding="utf-8"))["documents"]
        for entry in documents.values():
            assert entry["status"] == "completed"
            [warning] = entry["warnings"]
            assert "rejected the API key in DEEPSEEK_API_KEY" in warning
        # A refused key is not sent again once every model has refused it
        assert 1 <= len(paths) <= 2
