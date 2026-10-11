"""Ctrl-C a real `markitai <dir> --llm` batch, then `--resume` it.

End to end: the CLI runs as a subprocess against a loopback OpenAI-compatible
stub (``openai/gpt-4o-mini``, a model with a known price, so the cost comes
from LiteLLM's table) and is stopped with a real SIGINT while its second
document waits on the model. The resumed run's report and summary must
count the first document's spend as well as its own.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import pytest

from tests.local_http import LocalHTTPServer

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="SIGINT to a process group is POSIX; Windows needs a console for "
    "CTRL_BREAK, and markitai does not handle SIGBREAK",
)

# gpt-4o-mini at $0.15/$0.60 per million tokens: $0.027 per request
PROMPT_TOKENS, COMPLETION_TOKENS = 100_000, 20_000
REQUEST_COST = PROMPT_TOKENS * 0.15e-6 + COMPLETION_TOKENS * 0.60e-6


def _instance(schema: dict[str, Any], defs: dict[str, Any], name: str) -> Any:
    """A value matching *schema* (enough for markitai's response models)."""
    if "$ref" in schema:
        return _instance(defs[schema["$ref"].split("/")[-1]], defs, name)
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"]
            return _instance((options or schema[key])[0], defs, name)
    kind = schema.get("type")
    if kind == "object" or "properties" in schema:
        return {
            prop: _instance(sub, defs, prop)
            for prop, sub in schema.get("properties", {}).items()
        }
    if kind == "array":
        return ["stub", "test", "markitai"]
    if kind == "integer":
        return 1
    if kind == "number":
        return 1.0
    if kind == "boolean":
        return True
    if name == "cleaned_markdown":
        return "# Document\n\nStub text of the document."
    return "A stub description of the document."


class _Stub(LocalHTTPServer):
    answered = 0
    hold_after = 1 << 30  # requests beyond this many wait for `release`
    release: threading.Event
    waiting: threading.Event


class _Handler(BaseHTTPRequestHandler):
    server: _Stub

    def do_POST(self) -> None:  # noqa: N802 (http.server naming)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.server.answered >= self.server.hold_after:
            self.server.waiting.set()
            self.server.release.wait(30)
        self.server.answered += 1
        message: dict[str, Any] = {"role": "assistant"}
        if body.get("tools"):
            function = body["tools"][0]["function"]
            params = function.get("parameters", {})
            arguments = _instance(params, params.get("$defs", {}), "")
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": "call_stub",
                    "type": "function",
                    "function": {
                        "name": function["name"],
                        "arguments": json.dumps(arguments),
                    },
                }
            ]
        elif (body.get("response_format") or {}).get("type") == "json_schema":
            schema = body["response_format"]["json_schema"]["schema"]
            message["content"] = json.dumps(
                _instance(schema, schema.get("$defs", {}), "")
            )
        else:
            message["content"] = "# Document\n\nStub text of the document."
        reply = json.dumps(
            {
                "id": "chatcmpl-stub",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": PROMPT_TOKENS,
                    "completion_tokens": COMPLETION_TOKENS,
                    "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
                },
            }
        ).encode()
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(reply)))
            self.end_headers()
            self.wfile.write(reply)
        except OSError:
            pass  # the interrupted client is gone

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass


@pytest.fixture
def stub() -> Iterator[_Stub]:
    server = _Stub(("127.0.0.1", 0), _Handler)
    server.release = threading.Event()
    server.waiting = threading.Event()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.release.set()
        server.shutdown()
        server.server_close()


def _wait_for(condition, timeout: float = 60.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_resume_after_sigint_reports_the_whole_batch(
    tmp_path: Path, stub: _Stub
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    for name in ("a", "b", "c"):
        (docs / f"{name}.md").write_text(f"# {name.upper()}\n\nText of {name}.\n")
    out = tmp_path / "out"
    home = tmp_path / "home"
    (home / ".markitai").mkdir(parents=True)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "llm": {
                    "enabled": True,
                    "model_list": [
                        {
                            "model_name": "default",
                            "litellm_params": {
                                "model": "openai/gpt-4o-mini",
                                "api_key": "sk-fake",
                                "api_base": f"http://127.0.0.1:{stub.server_port}/v1",
                            },
                        }
                    ],
                },
                "cache": {"enabled": False},
                "batch": {"concurrency": 1},
            }
        )
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "MARKITAI_HOME": str(home / ".markitai"),
        "NO_COLOR": "1",
    }
    env.pop("OPENAI_API_KEY", None)
    command = [
        sys.executable,
        "-m",
        "markitai",
        str(docs),
        "-o",
        str(out),
        "--llm",
        "-c",
        str(config),
        "--no-record-history",
    ]

    # First run: a.md is answered, b.md's request is held, then Ctrl-C
    stub.hold_after = 1
    proc = subprocess.Popen(
        command,
        env=env,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        assert _wait_for(lambda: stub.waiting.is_set() or proc.poll() is not None)
        assert proc.poll() is None, proc.communicate()[1].decode()
        assert _wait_for(lambda: len(list(out.glob("*.llm.md"))) == 1)
        time.sleep(0.3)
        os.killpg(proc.pid, signal.SIGINT)
        _stdout, stderr = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    assert proc.returncode != 0
    assert "--resume" in stderr.decode()
    (state_file,) = (out / ".markitai" / "states").glob("*.state.json")
    documents = json.loads(state_file.read_text(encoding="utf-8"))["documents"]
    done_first = [k for k, v in documents.items() if v["status"] == "completed"]
    assert len(done_first) == 1
    assert documents[done_first[0]]["cost_usd"] == pytest.approx(REQUEST_COST)

    # Second run: --resume finishes the other two
    stub.hold_after = 1 << 30
    stub.release.set()
    resumed = subprocess.run(
        [*command, "--resume"],
        env=env,
        cwd=tmp_path,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=120,
    )
    stderr_text = resumed.stderr.decode()
    assert resumed.returncode == 0, stderr_text
    assert re.search(
        r"Done: 3 files \(\d+:\d\d, \$0\.081 total, \$0\.054 this run\)",
        stderr_text,
    ), stderr_text

    (report_file,) = (out / ".markitai" / "reports").glob("*.report.json")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    usage = report["llm_usage"]
    assert usage["requests"] == 3
    assert usage["cost_usd"] == pytest.approx(3 * REQUEST_COST)
    assert usage["this_run"]["requests"] == 2
    assert usage["this_run"]["cost_usd"] == pytest.approx(2 * REQUEST_COST)
    first = report["documents"][done_first[0]]
    assert first["earlier_run"] is True
    assert first["cost_usd"] == pytest.approx(REQUEST_COST)
    assert first["started_at"] and first["completed_at"] and first["duration"]
    assert sum(u["requests"] for u in first["llm_usage"].values()) == 1
