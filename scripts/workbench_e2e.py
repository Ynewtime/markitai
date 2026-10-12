#!/usr/bin/env python3
"""Drive the web workbench in a headless browser against `markitai serve`.

Everything runs locally and offline: an OpenAI-compatible fake LLM and a page
server start in this process, the inputs are generated, and the server gets a
throwaway HOME and MARKITAI_HOME (your ~/.markitai is never read or written).
For each viewport width it walks the main flows with Playwright/Chromium and
screenshots every state:

    convert page, a refused URL as a bottom notice strip, staged files +
    URLs, live progress, finished list, a conversion in the peek window
    (centred; full screen on a phone) with its Details, Preview and Compare
    tabs (original page beside the Markdown, C to switch), Up/Down and the
    header arrows between items, the window docked beside the list and back
    (a per-browser choice), retry of a failed item, a failed action reported
    in the window, New conversion and stop remaining (cancel), download all
    from the "⋯" menu (ticket; on a phone Stop from the header of the
    full-screen windows and on its own row under the Conversions title, a
    long URL title cut in the middle), an inline SVG drawing in the preview,
    settings model test (fake LLM), every settings page and Convert as one
    centred column, the provider picker (groups, brand marks, search), a
    saved address shown without its credentials, a provider beyond the first
    eight connected with a key and an address, Vertex AI connected without an
    address (its login note; Bedrock keeps its address; its saved connection
    offers no key and asks for no model list), history after a
    reload, token prompt

Notices must be strips at the bottom (never the page-wide banners the UI
had before) and leave by themselves.

It fails on a page error, a 5xx answer or a step whose check does not hold.

Usage (needs the serve and browser extras, and the browser for Playwright:
``python -m playwright install chromium``, or ``webkit`` / ``firefox`` for
``--browser``)::

    uv run python scripts/workbench_e2e.py --out /tmp/workbench-e2e
    uv run --with google-auth python scripts/workbench_e2e.py   # + the Vertex AI card
    uv run python scripts/workbench_e2e.py --browser webkit --theme dark
    uv run python scripts/workbench_e2e.py --widths 1440 --bundle path/to/unpacked-bundle

Results: ``OUT/<NN>-<state>-<width>.png``, ``OUT/results.json`` and ``OUT/INDEX.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import signal
import socket
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOKEN = "e2e-token-0000000000000000"
LLM_DELAY_S = 0.8
SLOW_PAGE_S = 2.5
# A saved connection whose address carries credentials: settings must show
# it masked and never send these parts to the browser
CONNECTION_ID = "00000000-0000-4000-8000-0000000000c1"
BASE_SECRETS = ("e2e-user", "e2e-pass-0000", "e2e-base-secret-0000")
# The server's own OpenAI key, behind the environment card
ENV_KEY = ("OPENAI_API_KEY", "sk-e2e-env-0000")
# The settings API's refusal of a Vertex AI address (the UI translates it by
# this wording)
VERTEX_REFUSAL = (
    "api_base: Vertex AI models send the server's own Google credentials to any "
    "address, so here they can only use Vertex AI's own endpoint; set another "
    "address in the config file"
)


DRAWING = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="240" height="60" viewBox="0 0 240 60">'
    '<rect width="240" height="60" style="fill:#2563eb"/>'
    '<text x="16" y="36" fill="#ffffff">Q3 drawing</text>'
    '<script>alert("drawing script ran")</script></svg>'
)


def saved_base(llm_port: int) -> str:
    return (
        f"http://e2e-user:e2e-pass-0000@127.0.0.1:{llm_port}/v1"
        "?api_key=e2e-base-secret-0000&v=1"
    )


def shown_base(llm_port: int) -> str:
    return f"http://\u2026@127.0.0.1:{llm_port}/v1?api_key=\u2026&v=1"


# ---------------------------------------------------------------------------
# Offline helpers: a fake OpenAI-compatible LLM and a page server
# ---------------------------------------------------------------------------

_DOC_MARKERS = [
    re.compile(r"<document>\n?(.*?)\n?</document>", re.DOTALL),
    re.compile(r"Please process the following document:\s*---\s*(.*)", re.DOTALL),
    re.compile(r"Please optimize the following Markdown:\s*(.*)", re.DOTALL),
]


def _texts(messages: list[dict[str, Any]], role: str | None = None) -> list[str]:
    out: list[str] = []
    for message in messages:
        if role is not None and message.get("role") != role:
            continue
        content = message.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
    return out


def _document(messages: list[dict[str, Any]]) -> str:
    for text in _texts(messages, "user"):
        for pattern in _DOC_MARKERS:
            match = pattern.search(text)
            if match:
                return match.group(1).strip()
    users = _texts(messages, "user")
    return users[-1].strip() if users else "OK"


def _synth(
    name: str, schema: dict[str, Any], defs: dict[str, Any], doc: str, depth: int = 0
) -> Any:
    if "$ref" in schema:
        return _synth(
            name, defs.get(schema["$ref"].split("/")[-1], {}), defs, doc, depth
        )
    for key in ("anyOf", "oneOf"):
        if key in schema:
            options = [s for s in schema[key] if s.get("type") != "null"] or schema[key]
            return _synth(name, options[0], defs, doc, depth)
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), "string")
    lname = name.lower()
    if kind == "object" or "properties" in schema:
        return {
            k: _synth(k, v, defs, doc, depth + 1)
            for k, v in schema.get("properties", {}).items()
        }
    if kind == "array":
        if lname == "tags":
            return ["sample", "workbench"]
        return (
            []
            if depth > 3
            else [
                _synth(name.rstrip("s"), schema.get("items", {}), defs, doc, depth + 1)
            ]
        )
    if kind in ("integer", "number"):
        return 1
    if kind == "boolean":
        return False
    if "markdown" in lname or lname in ("content", "text", "cleaned"):
        return doc
    if "description" in lname:
        return "A synthetic description from the offline fake LLM."
    if "title" in lname:
        return "Sample document"
    return "Sample figure" if ("alt" in lname or "caption" in lname) else "fake"


def _answer(body: dict[str, Any]) -> tuple[str | None, list[dict[str, Any]] | None]:
    messages = body.get("messages") or []
    doc = _document(messages)
    for text in _texts(messages):
        at = text.find("json_schema:")
        if at >= 0 and not body.get("tools"):
            try:
                schema, _ = json.JSONDecoder().raw_decode(text[text.find("{", at) :])
            except ValueError:
                continue
            value = json.dumps(_synth("result", schema, schema.get("$defs", {}), doc))
            if (body.get("response_format") or {}).get("type") in (
                "json_object",
                "json_schema",
            ):
                return value, None
            return f"```json\n{value}\n```", None
    tools = body.get("tools") or []
    if tools:
        function = tools[0].get("function", {})
        params = function.get("parameters", {})
        arguments = _synth(
            function.get("name", "r"), params, params.get("$defs", {}), doc
        )
        call = {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": function.get("name", "r"),
                "arguments": json.dumps(arguments),
            },
        }
        return None, [call]
    fmt = body.get("response_format") or {}
    if fmt.get("type") == "json_schema":
        schema = fmt.get("json_schema", {}).get("schema", {})
        return json.dumps(_synth("result", schema, schema.get("$defs", {}), doc)), None
    if fmt.get("type") == "json_object":
        return json.dumps(
            {"caption": "Sample figure", "description": "Synthetic."}
        ), None
    return doc, None


class _FakeLLM(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args: Any) -> None:
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path.rstrip("/").endswith("/models"):
            self._json(
                200,
                {"object": "list", "data": [{"id": "fake-model", "object": "model"}]},
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        body = json.loads(raw or b"{}")
        time.sleep(LLM_DELAY_S)
        content, calls = _answer(body)
        usage = {
            "prompt_tokens": max(1, len(raw) // 4),
            "completion_tokens": 20,
            "total_tokens": max(1, len(raw) // 4) + 20,
        }
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if calls:
            message["tool_calls"] = calls
        if body.get("stream"):
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("connection", "close")
            self.end_headers()
            delta: dict[str, Any] = {"role": "assistant"}
            if calls:
                delta["tool_calls"] = [dict(c, index=0) for c in calls]
            else:
                delta["content"] = content
            for choice in (
                {"index": 0, "delta": delta, "finish_reason": None},
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "tool_calls" if calls else "stop",
                },
            ):
                chunk = {
                    "id": "c",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": body.get("model"),
                    "choices": [choice],
                    "usage": usage,
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.close_connection = True
            return
        self._json(
            200,
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
                "usage": usage,
            },
        )


class _Pages(BaseHTTPRequestHandler):
    """``/page/<n>.html`` answers at once, ``/slow/[<dir>/…]<n>.html`` after
    a pause."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        match = re.fullmatch(r"/(page|slow)/(?:[\w-]+/)*(\w+)\.html", self.path)
        if not match:
            self.send_response(404)
            self.end_headers()
            return
        if match.group(1) == "slow":
            time.sleep(SLOW_PAGE_S)
        rows = "".join(
            f"<tr><td>Row {n}</td><td>{n * 7}</td></tr>" for n in range(1, 8)
        )
        html = (
            f"<!doctype html><html><head><title>Page {match.group(2)}</title></head><body><article>"
            f"<h1>Page {match.group(2)}</h1><p>A local page for the workbench end-to-end run. "
            "It has a paragraph, a list and a table so the Markdown has some structure.</p>"
            "<ul><li>First point</li><li>Second point</li></ul>"
            f"<table><thead><tr><th>Name</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>"
            "</article></body></html>"
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(html)))
        self.end_headers()
        self.wfile.write(html)


class _LocalServer(ThreadingHTTPServer):
    """ThreadingHTTPServer that binds without a host name lookup.

    ``HTTPServer.server_bind`` calls ``socket.getfqdn`` on the address,
    which took over 25 s on macOS CI runners.
    """

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, int]:
    server = _LocalServer(("127.0.0.1", 0), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1]


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


# ---------------------------------------------------------------------------
# Inputs and server
# ---------------------------------------------------------------------------


def make_inputs(directory: Path, pages_port: int) -> dict[str, Path]:
    import pymupdf
    from PIL import Image, ImageDraw

    directory.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    for index, height in enumerate((180, 320, 260, 410)):
        draw.rectangle(
            (120 + index * 180, 520 - height, 220 + index * 180, 520),
            fill=(37, 99, 235),
        )
    photo = directory / "bar-chart.png"
    image.save(photo)
    pdf = pymupdf.open()
    for number in range(1, 4):
        page = pdf.new_page(width=612, height=792)
        page.insert_text((72, 90), f"Quarterly report, part {number}", fontsize=22)
        page.insert_text(
            (72, 130), "Revenue grew in every region this quarter.", fontsize=12
        )
        y = 170
        for row in (
            "Region      Revenue   Growth",
            "North       1,200     4%",
            "South       980       6%",
            "East        1,540     3%",
        ):
            page.insert_text((72, y), row, fontsize=11, fontname="cour")
            y += 18
        if number == 1:
            # An extracted picture the preview draws through the item's
            # image route
            page.insert_image(pymupdf.Rect(72, 260, 372, 460), filename=str(photo))
    report = directory / "quarterly-report.pdf"
    pdf.save(report)
    notes = directory / "meeting-notes.md"
    # The drawing must render in the preview, its script never
    notes.write_text(
        "# Meeting notes\n\n- Ship 1.2.1\n- Review the workbench\n\n```python\nprint('hello')\n```\n\n"
        f"{DRAWING}\n",
        encoding="utf-8",
    )
    broken = directory / "damaged.pdf"
    broken.write_bytes(b"%PDF-1.7\n% this file is not a PDF\n")
    blob = directory / "archive.bin"
    blob.write_bytes(bytes(range(256)))
    url_list = directory / "reading.urls"
    url_list.write_text(
        f"# pages to read\nhttp://127.0.0.1:{pages_port}/page/alpha.html  Alpha page\nhttp://127.0.0.1:{pages_port}/page/beta.html\n",
        encoding="utf-8",
    )
    return {
        "pdf": report,
        "png": photo,
        "md": notes,
        "broken": broken,
        "bin": blob,
        "urls": url_list,
    }


@contextlib.contextmanager
def run_server(
    home: Path,
    port: int,
    llm_port: int,
    bundle: Path | None,
    log: Path,
    shutdown: dict[str, Any],
) -> Iterator[str]:
    """Run serve in *home* until the block ends, then stop it like a user does.

    How the stop went (graceful or not, how long, processes it left behind)
    is written into *shutdown*.
    """
    (home / ".markitai").mkdir(parents=True, exist_ok=True)
    config = {
        "llm": {
            "model_list": [
                {
                    "model_name": "default",
                    "litellm_params": {
                        "model": "openai/fake-model",
                        "api_key": "sk-e2e-fake-0000",
                        "api_base": f"http://127.0.0.1:{llm_port}/v1",
                    },
                    "model_info": {"id": "00000000-0000-4000-8000-0000000000e2"},
                }
            ],
            "providers": [
                {
                    "id": CONNECTION_ID,
                    "provider": "custom",
                    "api_key": "sk-e2e-conn-0000",
                    "api_base": saved_base(llm_port),
                }
            ],
        },
        "cache": {"enabled": False},
        "batch": {"url_concurrency": 1},
    }
    (home / ".markitai" / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper()
        in {
            "PATH",
            "SYSTEMROOT",
            "SYSTEMDRIVE",
            "TEMP",
            "TMP",
            "WINDIR",
            "COMSPEC",
            "PATHEXT",
            "LANG",
            "PLAYWRIGHT_BROWSERS_PATH",
            "LOCALAPPDATA",
        }
    }
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        MARKITAI_HOME=str(home / ".markitai"),
        MARKITAI_SERVE_TOKEN=TOKEN,
        NO_PROXY="127.0.0.1,localhost",
        # Anything bound for the internet (the environment card's model list
        # at api.openai.com) goes to a proxy that is not there: it fails here
        HTTPS_PROXY="http://127.0.0.1:9",
        PYTHONUTF8="1",
    )
    env[ENV_KEY[0]] = ENV_KEY[1]
    if bundle is None:
        command = [
            sys.executable,
            "-m",
            "markitai",
            "serve",
            "--port",
            str(port),
            "--no-open",
        ]
    else:
        command = [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; from markitai.serve import create_app; "
                "from markitai.cli.commands.serve import _run_server; "
                f"_run_server(create_app(static_dir=Path(sys.argv[1]), token={TOKEN!r}), '127.0.0.1', {port})"
            ),
            str(bundle),
        ]
    # Its own process group (Windows) or session: the stop signal below goes
    # to the server, and the terminal's Ctrl-C does not reach it twice
    group: dict[str, Any] = (
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
        if sys.platform == "win32"
        else {"start_new_session": True}
    )
    with log.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            command, cwd=home, env=env, stdout=sink, stderr=subprocess.STDOUT, **group
        )
        base = f"http://127.0.0.1:{port}"
        try:
            import urllib.request

            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"server exited with {process.returncode}; see {log}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"{base}/api/capabilities", timeout=1
                    ) as answer:
                        if answer.status == 200:
                            break
                except OSError:
                    time.sleep(0.3)
            else:
                raise RuntimeError("server did not start")
            yield base
        finally:
            shutdown.update(stop_server(process))


def _alive(process: Any) -> bool:
    import psutil

    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def stop_server(process: subprocess.Popen[bytes]) -> dict[str, Any]:
    """Stop serve with Ctrl-C (Ctrl-Break on Windows) and wait for its children.

    ``terminate()`` is not a stop on Windows: it kills the process outright
    (under ``uv run`` only the venv's python.exe launcher, which then kills
    the server through its job object), so the server cannot stop its
    workers, and its spawned workers break away from that job. A stopped
    server must leave no process behind: one that does keeps its working
    directory (the throwaway home) open, and Windows then refuses to delete
    it. Leftovers are reported, then killed.
    """
    import psutil

    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    started = time.monotonic()
    graceful, note = True, ""
    try:
        process.send_signal(
            signal.CTRL_BREAK_EVENT if sys.platform == "win32" else signal.SIGINT
        )
        process.wait(timeout=30)
    except (OSError, ValueError) as error:  # Windows: no console to signal in
        graceful, note = False, f"could not signal it ({error}); killed"
    except subprocess.TimeoutExpired:
        graceful, note = False, "still running 30 s after the signal; killed"
    if process.poll() is None:
        for child in children:
            with contextlib.suppress(psutil.Error):
                child.kill()
        process.kill()
        process.wait(timeout=30)
    seconds = round(time.monotonic() - started, 1)
    # A stopped server stops its children; give them a moment to exit
    deadline = time.monotonic() + 10
    leftovers = [child for child in children if _alive(child)]
    while leftovers and time.monotonic() < deadline:
        time.sleep(0.2)
        leftovers = [child for child in leftovers if _alive(child)]
    described = []
    for child in leftovers:
        with contextlib.suppress(psutil.Error):
            described.append(f"{child.pid} {' '.join(child.cmdline())[:160]}")
            child.kill()
    return {
        "graceful": graceful,
        "seconds": seconds,
        "children": len(children),
        "leftovers": described,
        "note": note,
    }


def shutdown_step(shutdown: dict[str, Any]) -> dict[str, Any]:
    """The server's stop as a step of the report."""
    if not shutdown:
        return {
            "state": "shutdown",
            "ok": False,
            "note": "the server was not stopped",
            "screenshot": "",
            "seconds": 0,
        }
    leftovers = shutdown["leftovers"]
    ok = shutdown["graceful"] and not leftovers
    if ok:
        note = (
            f"stopped by {'Ctrl-Break' if sys.platform == 'win32' else 'Ctrl-C'} "
            f"in {shutdown['seconds']} s; its {shutdown['children']} child "
            "process(es) ended with it"
        )
    else:
        note = "; ".join(
            part
            for part in (
                shutdown["note"],
                f"left running: {', '.join(leftovers)}" if leftovers else "",
            )
            if part
        )
    return {
        "state": "shutdown",
        "ok": ok,
        "note": note,
        "screenshot": "",
        "seconds": shutdown["seconds"],
    }


def remove_tree(path: Path, timeout: float = 15) -> str | None:
    """Delete *path*, retrying while Windows still has a file in it open.

    Returns:
        None once it is gone, else the last error.
    """
    deadline = time.monotonic() + timeout
    while True:
        try:
            shutil.rmtree(path)
            return None
        except FileNotFoundError:
            return None
        except OSError as error:
            if time.monotonic() >= deadline:
                return f"{type(error).__name__}: {error}"
            time.sleep(0.5)


# ---------------------------------------------------------------------------
# The browser run
# ---------------------------------------------------------------------------


@dataclass
class Run:
    out: Path
    width: int
    steps: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    api: list[tuple[str, str, int]] = field(default_factory=list)
    jobs: list[str] = field(default_factory=list)
    counter: int = 0
    current: str = "start"
    page_errors: list[tuple[str, str]] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    navigation_cancels: list[str] = field(default_factory=list)

    def page_error(self, error: Any) -> None:
        self.page_errors.append((self.current, str(error)))

    def request_failed(self, request: Any) -> None:
        # WebKit "Load request cancelled", Chromium net::ERR_ABORTED,
        # Firefox NS_BINDING_ABORTED: the page went away, nothing refused it
        failure = str(request.failure or "")
        if re.search(r"cancel|abort", failure, re.IGNORECASE):
            self.cancelled.append(request.url)

    def settle_page_errors(self, engine: str) -> None:
        """Page errors become run errors, except WebKit's cancelled fetches.

        Playwright's WebKit reports every console error of the page as a
        page error, and WebKit logs a fetch that a navigation cancelled as
        "Fetch API cannot load <url> due to access control checks", although
        the app caught it and nothing refused it (Safari only logs it). Such
        an error counts as a cancellation when that URL's request failed as
        cancelled; a real access-control failure does not fail that way.
        """
        cancelled = list(self.cancelled)
        for state, message in self.page_errors:
            match = re.search(r"(/api/\S*) due to access control checks", message)
            url = (
                next((u for u in cancelled if u.endswith(match.group(1))), None)
                if engine == "webkit" and match
                else None
            )
            if url is not None:
                cancelled.remove(url)
                self.navigation_cancels.append(f"{state}: {match.group(1)}")
            else:
                self.errors.append(f"pageerror during {state}: {message}")

    def shot(self, page: Any, state: str) -> str:
        self.counter += 1
        name = f"{self.counter:02d}-{state}-{self.width}.png"
        page.screenshot(path=str(self.out / name))
        return name

    def step(self, page: Any, state: str, check: Callable[[], str | None]) -> None:
        started = time.monotonic()
        self.current = state
        try:
            note = check() or ""
            ok = True
        except Exception as error:  # noqa: BLE001 - a failed check is recorded, the run goes on
            note, ok = f"{type(error).__name__}: {error}"[:400], False
        try:
            shot = self.shot(page, state)
        except Exception as error:  # noqa: BLE001 - a lost page is a failed step
            shot, ok = "", False
            note = f"{note} (screenshot failed: {type(error).__name__}: {error})"[:400]
        self.steps.append(
            {
                "state": state,
                "ok": ok,
                "note": note,
                "screenshot": shot,
                "seconds": round(time.monotonic() - started, 1),
            }
        )
        print(f"[{self.width}] {'ok  ' if ok else 'FAIL'} {state}: {note}", flush=True)

    def saw(
        self, method: str, pattern: str, status: int | None = None, since: int = 0
    ) -> bool:
        """Whether an API answer matched, among those from index *since* on."""
        return any(
            m == method and re.search(pattern, path) and (status is None or s == status)
            for m, path, s in self.api[since:]
        )


def _wait(
    page: Any,
    predicate: Callable[[], bool],
    timeout: float = 60,
    what: str = "condition",
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        page.wait_for_timeout(200)
    raise TimeoutError(f"timed out waiting for {what}")


def _items(page: Any) -> Any:
    return page.get_by_role("treeitem")


def _item(page: Any, name: str) -> Any:
    """One row of the list by its file name or URL (the UI may break a long
    name in two for its middle ellipsis, so spaces may sit anywhere)."""
    return page.get_by_role("treeitem", name=_spaced(name)).first


def _spaced(name: str) -> re.Pattern[str]:
    # "/" is escaped too: Playwright passes the pattern on as a /.../ literal
    return re.compile(r"\s*".join(re.escape(char).replace("/", r"\/") for char in name))


def _visible(locator: Any) -> Any | None:
    for index in range(locator.count()):
        candidate = locator.nth(index)
        if candidate.is_visible():
            return candidate
    return None


def _click_visible(locator: Any, timeout: float = 30) -> None:
    """Click the first visible match, one inside an open sheet or dialog
    first (a twin behind the sheet is visible but covered)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        sheet = _visible(locator.page.get_by_role("dialog"))
        for scope in ([sheet.locator(locator)] if sheet is not None else []) + [
            locator
        ]:
            for index in range(scope.count()):
                candidate = scope.nth(index)
                if candidate.is_visible() and candidate.is_enabled():
                    candidate.click()
                    return
        time.sleep(0.2)
    raise TimeoutError("no visible, enabled match to click")


def _dismiss(page: Any) -> None:
    """Close sheets and dialogs (the phone layout opens a summary sheet)."""
    for _ in range(3):
        dialogs = page.get_by_role("dialog")
        if not any(dialogs.nth(i).is_visible() for i in range(dialogs.count())):
            return
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)


def _heading(page: Any, name: str) -> Any:
    return page.get_by_role("heading", name=_spaced(name)).first


def _peek(page: Any) -> Any:
    """The open conversion: a window over the list (a dialog), or docked
    beside it (a region); full screen on a phone."""
    return page.locator(".peek").first


def _peek_title(page: Any) -> str:
    return re.sub(r"\s+", "", page.locator("#peek-title").inner_text())


def _tab(page: Any, name: str) -> Any:
    return page.locator(".peek-tabs").get_by_role("tab", name=name, exact=True)


def _show_tab(page: Any, name: str) -> None:
    tab = _tab(page, name)
    tab.click()
    _wait(
        page,
        lambda: tab.get_attribute("aria-selected") == "true",
        timeout=10,
        what=f"the {name} tab",
    )


def _close_window(page: Any) -> None:
    """Escape closes the window. Its key handler is registered a frame after
    the window appears, so give it that frame first."""
    page.wait_for_timeout(300)
    page.keyboard.press("Escape")
    _peek(page).wait_for(state="detached", timeout=10000)


def _window_placement(page: Any, phone: bool) -> str:
    """A window centred in the viewport on a wide screen, the whole screen on
    a phone."""
    box, settled, view = _peek(page).bounding_box(), None, page.viewport_size
    for _ in range(20):  # after its slide-in (a phone's comes from below)
        page.wait_for_timeout(100)
        settled, box = box, _peek(page).bounding_box()
        if box == settled:
            break
    if box is None or view is None:
        raise AssertionError("the window has no box")
    if phone:
        if box["x"] > 1 or box["y"] > 1 or box["width"] < view["width"] - 1:
            raise AssertionError(f"the window does not fill the phone: {box}")
        if box["height"] < view["height"] - 1:
            raise AssertionError(f"the window does not fill the phone: {box}")
        return "full screen"
    across = box["x"] + box["width"] / 2 - view["width"] / 2
    down = box["y"] + box["height"] / 2 - view["height"] / 2
    if abs(across) > 3 or abs(down) > 3:
        raise AssertionError(f"the window is not centred: {box}")
    return f"centred, {box['width']:.0f}x{box['height']:.0f}"


def _centred_column(page: Any) -> str:
    """The page's column sits in the middle of the main area (its heading
    row spans the column; Convert has none)."""
    heads = page.locator("#main .page-head")
    column = heads.first if heads.count() else page.locator("#main .page").first
    gaps = column.evaluate(
        """page => {
            const main = document.getElementById("main");
            const r = page.getBoundingClientRect(), m = main.getBoundingClientRect();
            return [r.left - m.left, m.left + main.clientWidth - r.right, r.width];
        }"""
    )
    left, right, wide = gaps
    if abs(left - right) > 2:
        raise AssertionError(f"the column is off centre: {left:.0f} / {right:.0f} px")
    return f"{wide:.0f} px column, {left:.0f} px each side"


def _control_height(locator: Any, phone: bool) -> str:
    """One control scale: 28 px buttons, 36 px on a phone."""
    box = locator.bounding_box()
    expected = 36 if phone else 28
    if box is None or abs(box["height"] - expected) > 1:
        raise AssertionError(
            f"a button is {box and box['height']} px high, not {expected}"
        )
    return f"{expected} px buttons"


def _header_stop(page: Any) -> Any:
    """The window's Stop remaining: a button of its header (above its tabs
    or body), within the screen, never folded into "⋯"."""
    window = _peek(page)
    window.wait_for(timeout=10000)
    _window_placement(page, True)  # after its slide-in from below
    stop = window.get_by_role("button", name=re.compile(r"^Stop remaining"))
    stop.first.wait_for(timeout=10000)
    box, frame, view = (
        stop.first.bounding_box(),
        window.bounding_box(),
        page.viewport_size,
    )
    if box is None or frame is None or view is None:
        raise AssertionError("Stop remaining has no box")
    if box["y"] < 0 or box["y"] + box["height"] > min(frame["y"] + 120, view["height"]):
        raise AssertionError(f"Stop remaining is not in the window's header: {box}")
    if box["x"] < 0 or box["x"] + box["width"] > view["width"]:
        raise AssertionError(f"Stop remaining is off the screen: {box}")
    return stop.first


def _fits(locator: Any) -> bool:
    """Whether *locator*'s text is drawn whole (nothing cut by an ellipsis)."""
    return bool(locator.evaluate("e => e.scrollWidth <= e.clientWidth + 1"))


def _phone_conversions_head(page: Any) -> str:
    """The phone Conversions header while items wait: the title whole, then
    the status, then Stop remaining on a row of its own (bcdc180 squeezed
    the title to "Co" and wrapped the status one word per line)."""
    head = page.locator(".conv-head")
    title, summary = head.locator("h1"), head.locator(".conv-summary")
    stop = head.get_by_role("button", name=re.compile(r"^Stop remaining"))
    stop.wait_for(timeout=10000)
    page.wait_for_timeout(300)
    boxes = [x.bounding_box() for x in (title, summary, stop)]
    if any(b is None for b in boxes):
        raise AssertionError(f"Conversions header boxes: {boxes}")
    t, s, b = boxes
    if not _fits(title) or t["width"] < 80:
        raise AssertionError(f"the Conversions title is cut: {t}")
    line = float(summary.evaluate("e => parseFloat(getComputedStyle(e).lineHeight)"))
    if s["y"] < t["y"] + t["height"] - 1 or s["height"] > 2 * line + 2:
        raise AssertionError(f"the status is squeezed: {s} (line {line})")
    if b["y"] < s["y"] + s["height"] - 1:
        raise AssertionError(f"Stop remaining is not on its own row: {b}, status {s}")
    return (
        f"title whole ({t['width']:.0f} px), status in "
        f"{round(s['height'] / line)} line(s), Stop on its own row"
    )


def _middle_ellipsis(page: Any, host: str, end: str) -> str:
    """A phone window's URL title: on its own row under the header buttons,
    shortened in the middle (the host and the end of the path stay)."""
    title = page.locator("#peek-title")
    head = title.locator(".file-name-head")
    tail = title.locator(".file-name-tail")
    if not head.count() or not tail.count():
        raise AssertionError("the window title has no middle ellipsis parts")
    if not head.inner_text().startswith(f"http://{host}"):
        raise AssertionError(f"the title starts {head.inner_text()!r}")
    if not tail.inner_text().endswith(end) or not _fits(tail):
        raise AssertionError(f"the title ends {tail.inner_text()!r}")
    full = title.locator(".file-name").get_attribute("title") or ""
    if head.inner_text() + tail.inner_text() != full:
        raise AssertionError(f"the title's parts do not make {full!r}")
    if not tail.is_visible() or not head.is_visible():
        raise AssertionError("a part of the title is hidden")
    close = _peek(page).get_by_role("button", name="Close").bounding_box()
    box = title.bounding_box()
    if close is None or box is None or box["y"] < close["y"] + close["height"] - 1:
        raise AssertionError(f"the title is not on its own row: {box}, Close {close}")
    if _fits(head):
        raise AssertionError(f"the title {full!r} is not shortened")
    return f"title cut in the middle, ends {tail.inner_text()!r} on its own row"


#: The last URL of the Stop step: long enough that a phone window's title
#: has to shorten it (in the middle: the host and its file name stay)
LAST_WAITING = "slow/annual-reports/2026/reading-list/s5.html"


def _phone_stop(page: Any, run: Run) -> str:
    """Stop reached on a phone while the full-screen windows cover the list."""
    _header_stop(page)
    run.shot(page, "phone-stop-submission")
    title = " ".join(page.locator("#peek-title").inner_text().split())
    _click_visible(page.get_by_role("button", name="Close"))
    _peek(page).wait_for(state="detached", timeout=10000)
    head = _phone_conversions_head(page)
    run.shot(page, "phone-stop-list")
    # The last URL waits behind the others (one URL at a time)
    _item(page, LAST_WAITING).click()
    _heading(page, LAST_WAITING).wait_for()
    stop = _header_stop(page)
    named = _middle_ellipsis(page, "127.0.0.1", "/s5.html")
    run.shot(page, "phone-stop-item")
    stop.click()
    return (
        f"the header of {title}'s window and of a waiting item's window "
        f"({named}); Conversions: {head}"
    )


#: The notices of the UI before the bundle of 2026-10-11 (rounded boxes with
#: a bar and a close button, page-wide banners): none may be left
OLD_NOTICES = ".notice, .notes-region, .note-close"


def _no_banners(page: Any) -> None:
    left = page.locator(OLD_NOTICES).count()
    if left:
        raise AssertionError(f"{left} page-wide notice(s) of the old UI on the page")


def _strip(page: Any, tone: str, text: str | re.Pattern[str]) -> Any:
    """The notice strip of *tone* saying *text*, once it is shown."""
    strip = page.locator(f".note-strips .note-strip.is-{tone}").filter(has_text=text)
    strip.first.wait_for(timeout=15000)
    return strip.first


def _at_the_bottom(page: Any, strip: Any) -> str:
    """Where a strip sits: full width above the tab bar on a phone, a narrow
    stack in the bottom-right corner on a wide screen; one line either way."""
    box, settled = strip.bounding_box(), None
    for _ in range(20):
        page.wait_for_timeout(100)
        settled, box = box, strip.bounding_box()
        if box == settled:
            break
    view = page.viewport_size
    if box is None or view is None:
        raise AssertionError("the strip has no box")
    width, height = view["width"], view["height"]
    bottom, right = box["y"] + box["height"], box["x"] + box["width"]
    if box["height"] > 60:
        raise AssertionError(f"the strip is {box['height']:.0f} px high, not one line")
    if bottom > height or box["y"] < height * 0.6:
        raise AssertionError(f"the strip is not at the bottom: {box}")
    if width < 600:
        if box["width"] < width - 2:
            raise AssertionError(f"the strip is not full width: {box}")
        bar = _visible(page.get_by_role("navigation"))
        top = bar.bounding_box() if bar is not None else None
        if top is not None and top["y"] > height * 0.6 and bottom > top["y"] + 1:
            raise AssertionError(f"the strip covers the tab bar: {box} / {top}")
        return f"full width at y={box['y']:.0f}, above the tab bar"
    if right < width - 48 or height - bottom > 48 or box["width"] > 420:
        raise AssertionError(f"the strip is not in the bottom-right corner: {box}")
    return f"{box['width']:.0f} px wide, {width - right:.0f} px from the right edge"


def _google_auth() -> bool:
    """Whether the server (this interpreter) can sign Vertex AI requests."""
    try:
        return importlib.util.find_spec("google.auth") is not None
    except ModuleNotFoundError:
        return False


def _api(page: Any, base: str, method: str, path: str, **kwargs: Any) -> Any:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    return page.request.fetch(f"{base}{path}", method=method, headers=headers, **kwargs)


def _open(page: Any, navigate: Callable[[], Any]) -> None:
    """Navigate (goto, reload) and wait for the app's capabilities answer.

    The app asks for /api/capabilities as it starts; navigating again before
    that answer cancels the request, which WebKit logs as an error. A user
    reloading in Safari sees nothing; a run should not cause it.
    """
    with page.expect_response(
        lambda response: response.url.endswith("/api/capabilities"), timeout=30000
    ):
        navigate()


def browse(
    base: str,
    width: int,
    out: Path,
    inputs: dict[str, Path],
    pages_port: int,
    engine: str = "chromium",
    *,
    llm_port: int,
    config: Path,
    theme: str = "light",
) -> Run:
    from playwright.sync_api import sync_playwright

    run = Run(out=out, width=width)
    phone = width < 600
    with sync_playwright() as playwright:
        browser = getattr(playwright, engine).launch()
        context = browser.new_context(
            viewport={"width": width, "height": 844 if phone else 900},
            device_scale_factor=2 if phone else 1,
            accept_downloads=True,
            color_scheme=theme,
        )
        page = context.new_page()
        page.on("pageerror", run.page_error)
        page.on("requestfailed", run.request_failed)

        def record(response: Any) -> None:
            if "/api/" in response.url:
                path = response.url.replace(base, "")
                run.api.append((response.request.method, path, response.status))
                if (
                    response.request.method == "POST"
                    and path == "/api/jobs"
                    and response.status == 201
                ):
                    with contextlib.suppress(Exception):
                        run.jobs.append(response.json()["job_id"])
                if response.status >= 500:
                    run.errors.append(
                        f"{response.request.method} {path} -> {response.status}"
                    )

        page.on("response", record)

        # 1. The convert page, signed in by the URL fragment
        def home() -> str:
            _open(page, lambda: page.goto(f"{base}/#token={TOKEN}"))
            page.get_by_role("button", name="Convert").wait_for()
            if "#token" in page.url:
                raise AssertionError("the token stayed in the address bar")
            return "token taken from the fragment and removed"

        run.step(page, "convert", home)

        # 1b. A refused URL: a notice strip at the bottom, gone by itself
        def notice() -> str:
            box = page.get_by_role("textbox", name=re.compile("Paste URLs"))
            box.fill("ftp://example.com/report.pdf")
            page.get_by_role("button", name="Convert", exact=True).click()
            strip = _strip(page, "error", "ftp://example.com/report.pdf")
            where = _at_the_bottom(page, strip)
            run.shot(page, "notice-strip-shown")
            _no_banners(page)
            if run.saw("POST", r"^/api/jobs$"):
                raise AssertionError("the refused URL was sent")
            # A strip stays while the pointer is on it: move away, then it
            # leaves after its 6 s
            page.mouse.move(1, 1)
            strip.wait_for(state="detached", timeout=12000)
            box.fill("")
            return f"ftp:// refused in a bottom strip ({where}) that left by itself"

        run.step(page, "notice-strip", notice)

        # 2. Stage files, a URL list and a pasted URL
        def stage() -> str:
            page.locator(
                "input[type=file]:not([webkitdirectory])"
            ).first.set_input_files(
                [
                    str(inputs[key])
                    for key in ("pdf", "png", "md", "broken", "bin", "urls")
                ]
            )
            page.get_by_role("textbox", name=re.compile("Paste URLs")).fill(
                f"http://127.0.0.1:{pages_port}/slow/gamma.html"
            )
            page.wait_for_timeout(400)
            return "6 files (with a .urls list) and 1 URL staged"

        run.step(page, "staged", stage)

        # 3. Convert: live progress
        def progress() -> str:
            page.get_by_role("button", name="Convert", exact=True).click()
            _wait(
                page, lambda: run.saw("POST", r"^/api/jobs$", 201), what="job creation"
            )
            _wait(page, lambda: run.saw("GET", r"/events$"), what="the event stream")
            page.wait_for_timeout(700)
            return "job created, event stream open"

        run.step(page, "progress", progress)

        def finished() -> str:
            _wait(
                page,
                lambda: (
                    bool(run.jobs)
                    and all(
                        _api(page, base, "GET", f"/api/jobs/{job}").json()["status"]
                        == "done"
                        for job in run.jobs
                    )
                ),
                timeout=120,
                what="the job to finish",
            )
            _wait(
                page,
                lambda: (
                    not re.search(
                        r"Converting|Queued|Waiting",
                        page.get_by_role("tree").first.aria_snapshot(),
                    )
                ),
                timeout=30,
                what="the list to show it",
            )
            names = re.sub(r"\s+", "", " ".join(_items(page).all_inner_texts()))
            for expected in (
                "quarterly-report.pdf",
                "meeting-notes.md",
                "damaged.pdf",
                "page/alpha.html",
                "page/beta.html",
                "slow/gamma.html",
            ):
                if expected not in names:
                    raise AssertionError(
                        f"{expected} missing from the list: {names[:300]}"
                    )
            snapshot = _api(page, base, "GET", f"/api/jobs/{run.jobs[-1]}").json()
            named = [
                item["output_name"]
                for item in snapshot["items"]
                if item["kind"] == "url"
            ]
            if "Alpha page.md" not in named:
                raise AssertionError(f"the .urls name was not used: {named}")
            return f"{len(snapshot['items'])} items finished; the .urls list became URL items"

        run.step(page, "finished", finished)

        # 4. The PDF opens in the peek window on Details
        def peek() -> str:
            _dismiss(page)
            before = len(run.api)
            _item(page, "quarterly-report.pdf").click()
            _heading(page, "quarterly-report.pdf").wait_for()
            page.get_by_role("dialog", name=_spaced("quarterly-report.pdf")).wait_for()
            tabs = [
                " ".join(t.split())
                for t in page.locator(".peek-tabs [role=tab]").all_inner_texts()
            ]
            if tabs != ["Details", "Preview", "Compare"]:
                raise AssertionError(f"tabs: {tabs}")
            if _tab(page, "Details").get_attribute("aria-selected") != "true":
                raise AssertionError("the window did not open on Details")
            page.locator(".peek").get_by_text("No model used").first.wait_for()
            _wait(
                page,
                lambda: run.saw("GET", r"/result\?max_bytes=16384", 200, since=before),
                what="the result excerpt",
            )
            where = _window_placement(page, phone)
            position = page.locator(".peek-pos")
            shown = (
                f"; position {position.inner_text().strip()!r}"
                if position.count()
                else ""
            )
            return f"window {where} on Details (sizes, cost, excerpt){shown}"

        run.step(page, "peek-details", peek)

        # 5. Preview tab
        def preview() -> str:
            _show_tab(page, "Preview")
            page.get_by_role("tab", name=re.compile("Rendered|Source")).first.wait_for()
            page.locator("#peek-panel").get_by_text(
                "Quarterly report, part 1"
            ).first.wait_for()
            # The PDF's picture, read through the item's own image route
            # (contract v1.6) and drawn, not "[Image: preview unavailable]"
            _wait(
                page,
                lambda: run.saw("GET", r"/items/i\d+/assets/.+\.(png|jpe?g)$", 200),
                what="the extracted picture from the item's image route",
            )
            drawn = page.locator(
                "#peek-panel [role=tabpanel]:not([hidden]) .doc-body img"
            ).first
            _wait(
                page,
                lambda: (
                    drawn.count() == 1
                    and drawn.evaluate("img => img.complete && img.naturalWidth > 0")
                ),
                what="the picture drawn in the preview",
            )
            if page.get_by_text("preview unavailable").count():
                raise AssertionError("a picture is shown as unavailable")
            return "Preview tab: rendered Markdown, its extracted picture drawn"

        run.step(page, "peek-preview", preview)

        # 6. Compare tab: the original page beside the Markdown; C switches
        def compare() -> str:
            _show_tab(page, "Compare")
            _wait(
                page,
                lambda: run.saw("GET", r"/items/i\d+/original$", 200),
                what="the original descriptor",
            )
            _wait(
                page,
                lambda: run.saw("GET", r"/original/pages/1\?width=\d+", 200),
                what="page 1 as an image",
            )
            page.wait_for_timeout(800)
            run.shot(page, "peek-compare-shown")
            page.wait_for_timeout(300)  # the key handler follows the tab a frame later
            page.keyboard.press("c")
            _wait(
                page,
                lambda: _tab(page, "Preview").get_attribute("aria-selected") == "true",
                timeout=10,
                what="C to switch back to Preview",
            )
            page.wait_for_timeout(300)
            page.keyboard.press("c")
            _wait(
                page,
                lambda: _tab(page, "Compare").get_attribute("aria-selected") == "true",
                timeout=10,
                what="C to switch to Compare",
            )
            return (
                "Compare tab: PDF page image and text layer; C toggles Preview/Compare"
            )

        run.step(page, "peek-compare", compare)

        # 6b. Up/Down (and the header's arrows) move between items
        def arrows() -> str:
            _show_tab(page, "Details")
            first = _peek_title(page)
            # The arrows move unless the focus is in the tabs or a document
            page.locator(".peek").focus()
            page.keyboard.press("ArrowDown")
            _wait(page, lambda: _peek_title(page) != first, timeout=10, what="Down")
            second = _peek_title(page)
            position = page.locator(".peek-pos")
            at = position.inner_text().strip() if position.count() else ""
            page.wait_for_timeout(300)
            page.locator(".peek").focus()
            page.keyboard.press("ArrowUp")
            _wait(page, lambda: _peek_title(page) == first, timeout=10, what="Up")
            _click_visible(page.get_by_role("button", name="Next item"))
            _wait(page, lambda: _peek_title(page) == second, timeout=10, what="Next")
            _click_visible(page.get_by_role("button", name="Previous item"))
            _wait(page, lambda: _peek_title(page) == first, timeout=10, what="Previous")
            if not phone and not re.fullmatch(r"2 of \d+", at):
                raise AssertionError(f"position after Down: {at!r}")
            _close_window(page)
            return (
                f"Down/Up and the header arrows: {first} <-> {second}"
                + (f" ({at})" if at else "")
                + "; Escape closed the window"
            )

        run.step(page, "peek-arrows", arrows)

        # 6c. Dock the window beside the list (a per-browser choice); phones
        # have no dock and fill the screen
        def dock() -> str:
            if phone:
                _item(page, "quarterly-report.pdf").click()
                _heading(page, "quarterly-report.pdf").wait_for()
                if page.get_by_role("button", name="Dock beside the list").count():
                    raise AssertionError("a phone offers the dock")
                where = _window_placement(page, phone)
                _click_visible(page.get_by_role("button", name="Close"))
                _peek(page).wait_for(state="detached", timeout=10000)
                return f"no dock on a phone; the window is {where}; Close (←) returns"
            _item(page, "quarterly-report.pdf").click()
            _heading(page, "quarterly-report.pdf").wait_for()
            page.get_by_role("button", name="Dock beside the list").click()
            region = page.locator(".peek.is-docked")
            region.wait_for(timeout=10000)
            if region.get_attribute("role") != "region":
                raise AssertionError("the docked window is not a region")
            if page.get_by_role("dialog").count():
                raise AssertionError("docked, a dialog is still open")
            tree, side = (
                page.get_by_role("tree").bounding_box(),
                _peek(page).bounding_box(),
            )
            if not tree or not side or side["x"] < tree["x"] + tree["width"] - 1:
                raise AssertionError(f"not beside the list: {tree} / {side}")
            stored = page.evaluate("() => localStorage.getItem('markitai.peek')")
            if stored != "docked":
                raise AssertionError(f"markitai.peek = {stored!r}")
            # Docked, the list's arrows select and the window follows
            _item(page, "quarterly-report.pdf").focus()
            page.keyboard.press("ArrowDown")
            _wait(
                page,
                lambda: _peek_title(page) != "quarterly-report.pdf",
                timeout=10,
                what="the docked window to follow the list",
            )
            followed = _peek_title(page)
            focused = page.locator("[role=treeitem]:focus")
            try:
                _wait(
                    page,
                    lambda: (
                        focused.count() == 1
                        and re.sub(r"\s+", "", focused.inner_text()).find(followed) >= 0
                    ),
                    timeout=5,
                    what="the focus to follow the selection",
                )
            except TimeoutError:
                raise AssertionError(
                    f"Down selected {followed} but the focus is on "
                    f"{focused.all_inner_texts() if focused.count() else 'no row'}"
                ) from None
            page.keyboard.press("ArrowUp")
            _wait(
                page,
                lambda: _peek_title(page) == "quarterly-report.pdf",
                timeout=10,
                what="the docked window to follow back",
            )
            page.wait_for_timeout(600)  # its excerpt, before the reload
            run.shot(page, "peek-docked")
            _open(page, page.reload)
            _items(page).first.wait_for()
            if not region.is_visible():
                _item(page, "quarterly-report.pdf").click()
            region.wait_for(timeout=10000)
            page.wait_for_timeout(600)
            page.get_by_role("button", name="Open as a window").click()
            page.get_by_role("dialog", name=_spaced("quarterly-report.pdf")).wait_for()
            stored_after = page.evaluate("() => localStorage.getItem('markitai.peek')")
            if stored_after != "window":
                raise AssertionError(f"markitai.peek = {stored_after!r}")
            _close_window(page)
            return (
                f"docked beside the list (list arrows select: {followed}), "
                "kept over a reload, back to a window"
            )

        run.step(page, "peek-docked", dock)

        # 7. Retry a failed item
        def retry() -> str:
            _open(page, lambda: page.goto(f"{base}/jobs"))
            _dismiss(page)
            _item(page, "damaged.pdf").click()
            _heading(page, "damaged.pdf").wait_for()
            _click_visible(
                page.get_by_role("button", name=re.compile(r"^(Retry|Run again)$"))
            )
            _wait(
                page,
                lambda: run.saw("POST", r"/items/i\d+/retry$", 202),
                what="the retry",
            )
            page.wait_for_timeout(2000)
            _dismiss(page)
            _item(page, "archive.bin").click()
            _heading(page, "archive.bin").wait_for()
            page.wait_for_timeout(300)
            offered = page.get_by_role(
                "button", name=re.compile(r"^(Retry|Run again)$")
            )
            if any(offered.nth(i).is_visible() for i in range(offered.count())):
                raise AssertionError("retry offered for an unsupported file")
            return "failed PDF retried; no retry offered for the unsupported .bin"

        run.step(page, "retry", retry)

        # 7b. A failed action on one conversion: reported in its inspector,
        # not as a notice (another tab deleted it before Retry was pressed)
        def action_error() -> str:
            found = [
                (job, item)
                for job in run.jobs
                for item in _api(page, base, "GET", f"/api/jobs/{job}").json()["items"]
                if item["name"] == "damaged.pdf"
            ]
            if not found:
                raise AssertionError("damaged.pdf not found through the API")
            job, item = found[0]

            def status() -> str:
                items = _api(page, base, "GET", f"/api/jobs/{job}").json()["items"]
                return next(
                    i["status"] for i in items if i["item_id"] == item["item_id"]
                )

            _wait(
                page,
                lambda: status() in ("done", "error"),
                timeout=60,
                what="damaged.pdf's retry to end",
            )
            _dismiss(page)
            _item(page, "damaged.pdf").click()
            _heading(page, "damaged.pdf").wait_for()
            reason = _visible(page.locator(".insp-line.is-error, .problem.is-error"))
            if reason is None:
                raise AssertionError("the inspector does not show why it failed")
            gone = _api(
                page, base, "DELETE", f"/api/jobs/{job}/items/{item['item_id']}"
            )
            if gone.status != 204:
                raise AssertionError(f"delete: {gone.status} {gone.text()[:200]}")
            before = len(run.api)
            _click_visible(
                page.get_by_role("button", name=re.compile(r"^(Retry|Run again)$"))
            )
            _wait(
                page,
                lambda: run.saw("POST", r"/items/i\d+/retry$", 404, since=before),
                what="the refused retry",
            )
            line = page.locator(".action-error-line").filter(has_text="Retry failed")
            line.first.wait_for(timeout=10000)
            run.shot(page, "inspector-error-line")
            if page.locator(".note-strip").filter(has_text="Retry failed").count():
                raise AssertionError("the failed retry also raised a notice")
            _no_banners(page)
            text = " ".join(line.first.inner_text().split())
            # The list still shows the deleted row until it is read again
            _open(page, page.reload)
            _items(page).first.wait_for()
            return f"inspector says {text!r}; no notice raised"

        run.step(page, "inspector-error", action_error)

        # 8. New conversion (the Conversions composer is gone), then Stop
        # remaining
        def stop() -> str:
            _dismiss(page)
            new = page.get_by_role("button", name="New conversion")
            scale = _control_height(new, phone)
            new.click()
            page.wait_for_url(re.compile(r"/(\?.*)?$"), timeout=10000)
            page.get_by_role("textbox", name=re.compile("Paste URLs")).fill(
                "\n".join(
                    [
                        f"http://127.0.0.1:{pages_port}/slow/s{n}.html"
                        for n in range(1, 5)
                    ]
                    + [f"http://127.0.0.1:{pages_port}/{LAST_WAITING}"]
                )
            )

            def created() -> int:
                return sum(
                    1
                    for m, p, s in run.api
                    if m == "POST" and p == "/api/jobs" and s == 201
                )

            before = created()
            page.get_by_role("button", name="Convert", exact=True).click()
            _wait(page, lambda: created() > before, what="the second job")
            page.wait_for_timeout(1200)
            where = "the list's header"
            if phone:
                # A phone opens the new submission full screen over the list:
                # Stop stays a button in that window's header, then in a
                # waiting item's window
                where = _phone_stop(page, run)
            else:
                _click_visible(
                    page.get_by_role("button", name=re.compile(r"^Stop remaining"))
                )
            _wait(
                page, lambda: run.saw("POST", r"/cancel$", 202), what="the stop request"
            )
            page.wait_for_timeout(SLOW_PAGE_S * 1000 + 1500)
            _dismiss(page)
            texts = " ".join(_items(page).all_inner_texts())
            if not re.search(r"Stopped|Cancelled", texts):
                raise AssertionError(f"no stopped item shown: {texts[:300]}")
            return (
                f"New conversion led to Convert ({scale}); queued URLs stopped "
                f"from {where}, the running one finished"
            )

        run.step(page, "stopped", stop)

        # 9. Download all, through a single-use ticket
        def download_all() -> str:
            _dismiss(page)
            page.locator(".conv-head").get_by_role(
                "button", name="More actions"
            ).click()
            with page.expect_download(timeout=60000) as info:
                page.get_by_role("menuitem", name="Download all").click()
            download = info.value
            data = Path(download.path()).read_bytes()
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
            if not names:
                raise AssertionError("empty archive")
            if not run.saw("POST", r"^/api/download-tickets$", 201):
                raise AssertionError("the download did not use a ticket")
            return f"from ⋯: {download.suggested_filename}: {len(names)} files, {len(data)} bytes via a ticket"

        run.step(page, "download-all", download_all)

        # 10. An LLM job (through the API) shows an unknown cost, never $0
        def llm_cost() -> str:
            created = _api(
                page,
                base,
                "POST",
                "/api/jobs",
                multipart={
                    "urls": json.dumps(
                        [f"http://127.0.0.1:{pages_port}/page/llm.html"]
                    ),
                    "options": json.dumps({"llm": True}),
                    "label": "LLM check",
                },
            )
            if created.status != 201:
                raise AssertionError(
                    f"LLM job: {created.status} {created.text()[:200]}"
                )
            job_id = created.json()["job_id"]
            _wait(
                page,
                lambda: (
                    _api(page, base, "GET", f"/api/jobs/{job_id}").json()["status"]
                    == "done"
                ),
                timeout=90,
                what="the LLM job",
            )
            item = _api(page, base, "GET", f"/api/jobs/{job_id}").json()["items"][0]
            if (
                not item["llm_enhanced"]
                or (item.get("pricing") or {}).get("cost_status") != "unknown"
            ):
                raise AssertionError(f"LLM item: {item}")
            _open(page, page.reload)
            _dismiss(page)
            _item(page, "page/llm.html").click()
            _heading(page, "llm.html").wait_for()
            cost = page.locator(
                "xpath=//dt[normalize-space()='Cost']/following-sibling::dd[1]"
            )
            text = " ".join(c for c in cost.all_inner_texts() if c.strip())
            if re.search(r"\$0(?![.,]\d*[1-9])", text) or "unknown" not in text.lower():
                raise AssertionError(f"cost shown as: {text[:300]}")
            return f"enhanced by the fake model; cost shown as {text.strip()!r}"

        run.step(page, "llm-cost-unknown", llm_cost)

        # 11. A drawing in the Markdown renders in the preview, sanitized
        def drawing() -> str:
            dialogs: list[str] = []
            page.on(
                "dialog",
                lambda dialog: (dialogs.append(dialog.message), dialog.dismiss()),
            )
            _open(page, lambda: page.goto(f"{base}/jobs"))
            _item(page, "meeting-notes.md").click()
            _heading(page, "meeting-notes.md").wait_for()
            _show_tab(page, "Preview")
            svg = page.locator("svg", has_text="Q3 drawing").first
            svg.wait_for(timeout=15000)
            if svg.locator("script").count():
                raise AssertionError("the drawing kept its script")
            fill = svg.locator("rect").first.get_attribute("fill")
            if fill != "#2563eb":
                raise AssertionError(f"the drawing's style paint became fill={fill!r}")
            page.wait_for_timeout(500)
            if dialogs:
                raise AssertionError(f"a script ran: {dialogs}")
            _close_window(page)
            return "inline SVG drawn in the Rendered view, its paint kept, its script removed"

        run.step(page, "preview-drawing", drawing)

        # 12. Settings: test the configured model
        def settings() -> str:
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name=re.compile(r"^Test ")))
            _wait(
                page,
                lambda: run.saw("POST", r"^/api/settings/llm/test$", 200),
                what="the model test",
            )
            page.wait_for_timeout(LLM_DELAY_S * 1000 + 800)
            return "model test answered by the fake LLM"

        run.step(page, "settings-test", settings)

        # 12a. Every settings page, and Convert, is one centred column
        def settings_pages() -> str:
            notes = []
            for path, title in (
                ("/settings/models", "Models"),
                ("/settings/appearance", "Appearance and language"),
                ("/settings/access", "Access token"),
                ("/", None),
            ):
                _open(page, lambda path=path: page.goto(f"{base}{path}"))
                if title is not None:
                    page.get_by_role("heading", name=title, level=1).wait_for()
                else:
                    page.get_by_role("button", name="Convert", exact=True).wait_for()
                page.wait_for_timeout(400)
                name = path.strip("/").replace("settings/", "") or "convert"
                run.shot(page, f"page-{name}")
                notes.append(f"{name}: {_centred_column(page)}")
            return "; ".join(notes)

        run.step(page, "settings-pages", settings_pages)

        # 12b. Add models: the provider picker groups Python's cards
        def picker() -> str:
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name="Add models"))
            panel = page.locator(".provider-picker")
            panel.wait_for()
            tiles = panel.locator(".provider-tile")
            # The brand marks load with the models settings: each tile's mark
            # replaces its blank placeholder
            _wait(
                page,
                lambda: (
                    tiles.count() >= 8
                    and panel.locator(".provider-tile svg.brand-icon").count()
                    == tiles.count()
                ),
                timeout=20,
                what="the common providers with their brand marks",
            )
            names = [" ".join(n.split()) for n in tiles.all_inner_texts()]
            common = [
                "OpenAI",
                "Anthropic",
                "Gemini",
                "DeepSeek",
                "OpenRouter",
                "Azure OpenAI",
                "Ollama",
                "Mistral AI",
            ]
            if names != common:
                raise AssertionError(f"Common shows {names}")
            chips = [
                " ".join(c.split())
                for c in panel.locator(".filter-chip").all_inner_texts()
            ]
            # The groups of the Rust server's picker: every provider this
            # build routes has a card (Subscriptions needs a runtime's extra)
            subscriptions = any(
                importlib.util.find_spec(name) is not None
                for name in ("claude_agent_sdk", "copilot")
            )
            expected_chips = [
                "All",
                "Clouds",
                "Inference",
                "Gateways",
                "Local",
                *(["Subscriptions"] if subscriptions else []),
                "Model makers",
            ]
            if chips != expected_chips:
                raise AssertionError(f"filters: {chips}")
            total = panel.get_by_role("searchbox").get_attribute("placeholder") or ""
            if not re.search(r"\b(8[5-9]|9\d) providers\b", total):
                raise AssertionError(f"the search offers {total!r}")
            panel.locator(".filter-chip").filter(has_text="Local").click()
            on = " ".join(panel.locator(".filter-chip.is-on").all_inner_texts()).strip()
            items = panel.locator(".provider-item")
            _wait(
                page,
                lambda: (
                    "OpenAI-compatible endpoint"
                    in [
                        " ".join(items.nth(i).inner_text().split())
                        for i in range(items.count())
                        if items.nth(i).is_visible()
                    ]
                ),
                timeout=5,
                what="Local to list the OpenAI-compatible endpoint",
            )
            # On a phone the sticky action bar covers the list's end until the
            # page scrolls: a trial click proves the row can be reached
            items.filter(has_text="OpenAI-compatible endpoint").first.click(
                trial=True, timeout=5000
            )
            run.shot(page, "provider-picker-local")
            if on != "Local":
                raise AssertionError(f"the filter shown as on is {on!r}")
            search = panel.get_by_role("searchbox")
            search.fill("ollama_chat")
            _wait(
                page,
                lambda: (
                    [" ".join(n.split()) for n in tiles.all_inner_texts()] == ["Ollama"]
                ),
                timeout=5,
                what="ollama_chat to find Ollama",
            )
            search.fill("self-hosted")
            _wait(
                page,
                lambda: (
                    "OpenAI-compatible endpoint" in " ".join(panel.all_inner_texts())
                ),
                timeout=5,
                what="self-hosted to find the OpenAI-compatible endpoint",
            )
            search.fill("")
            _no_banners(page)
            return (
                f"{total}; Common: {len(names)} providers with marks; "
                f"filters {' · '.join(chips)}; search by prefix and alias"
            )

        run.step(page, "provider-picker", picker)
        _dismiss(page)

        # 13. Settings: a saved address shows no credentials, and saving a
        # new key with the address left as shown keeps the stored address
        def masked_address() -> str:
            shown = shown_base(llm_port)
            credentials = f"/api/settings/llm/providers/{CONNECTION_ID}/credentials"
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name="Add models"))
            _click_visible(
                page.locator(".provider-pick").filter(
                    has_text="OpenAI-compatible endpoint"
                )
            )
            _wait(
                page,
                lambda: run.saw("GET", f"^{re.escape(credentials)}$", 200),
                what="the saved connection",
            )
            row = page.locator(".conn-row").filter(has_text="API base")
            _wait(
                page,
                lambda: row.locator(".conn-value").inner_text().strip() == shown,
                timeout=10,
                what=f"the address shown as {shown}",
            )
            leaks = [secret for secret in BASE_SECRETS if secret in page.content()]
            for path in (
                "/api/settings/llm",
                "/api/settings/llm/providers",
                credentials,
            ):
                body = _api(page, base, "GET", path).text()
                leaks += [
                    f"{path}: {secret}" for secret in BASE_SECRETS if secret in body
                ]
            if leaks:
                raise AssertionError(f"credentials reached the browser: {leaks}")
            row.get_by_role("button", name="Edit").click()
            field = page.locator("#conn-base")
            if field.input_value() != shown:
                raise AssertionError(f"address field holds {field.input_value()!r}")
            noted = (
                _visible(page.get_by_text("Parts shown as … are hidden")) is not None
            )
            # The form focuses the address a frame after it opens; typing the
            # key before that lands in the address (WebKit lost that race)
            # (a locator, not wait_for_function: the page's CSP forbids eval)
            page.locator("#conn-base:focus").wait_for(timeout=5000)
            key = page.locator("#conn-base-key")
            key.fill("sk-e2e-conn-0001")
            if field.input_value() != shown or key.input_value() != "sk-e2e-conn-0001":
                raise AssertionError("the key was not typed into its own field")
            before = len(run.api)
            page.locator("#conn-base-form").get_by_role("button", name="Save").click()
            _wait(
                page,
                lambda: run.saw(
                    "PATCH", f"^/api/settings/llm/providers/{CONNECTION_ID}$", 200
                ),
                what="the saved key",
            )
            # After a save the editor reloads the connection and its models;
            # leaving before that cancels them, which WebKit logs as errors
            _wait(
                page,
                lambda: (
                    run.saw("GET", f"^{re.escape(credentials)}$", 200, since=before)
                    and run.saw(
                        "POST", r"^/api/settings/llm/model-discovery$", since=before
                    )
                ),
                what="the editor's reload after the save",
            )
            stored = json.loads(config.read_text(encoding="utf-8"))["llm"]["providers"]
            if stored[0].get("api_base") != saved_base(llm_port):
                raise AssertionError("saving changed the stored address")
            if stored[0].get("api_key") != "sk-e2e-conn-0001":
                raise AssertionError("the new key was not saved")
            return (
                f"address shown as {shown}"
                + (" with the hidden-parts note" if noted else "")
                + "; a new key saved with it kept the stored address"
            )

        run.step(page, "settings-masked-address", masked_address)

        # 13b. Add models: a provider beyond markitai's first eight connects
        # with a key and an address of one's own, and lists its models (Groq,
        # pointed at the fake LLM: nothing leaves the machine)
        def connect_provider() -> str:
            address = f"http://127.0.0.1:{llm_port}/v1"
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name="Add models"))
            panel = page.locator(".provider-picker")
            panel.wait_for()
            panel.get_by_role("searchbox").fill("groq")
            _click_visible(panel.locator(".provider-item").filter(has_text="Groq"))
            # The key variable under the title; the default address is the
            # optional address field's placeholder
            facts = page.locator(".provider-facts")
            default_base = page.get_by_placeholder("https://api.groq.com/openai/v1")
            _wait(
                page,
                lambda: (
                    facts.count() == 1
                    and "GROQ_API_KEY" in facts.inner_text()
                    and default_base.count() == 1
                ),
                timeout=10,
                what="Groq's address and key variable",
            )
            page.get_by_placeholder("Paste the key").fill("gsk-e2e-0000-1111")
            default_base.fill(address)
            before = len(run.api)
            page.get_by_role("button", name="Load models").click()
            _wait(
                page,
                lambda: run.saw(
                    "POST", r"^/api/settings/llm/model-discovery$", 200, since=before
                ),
                what="Groq's model list",
            )
            option = page.locator(".picker-option").filter(has_text="groq/fake-model")
            option.wait_for(timeout=10000)
            option.locator("input[type=checkbox]").check()
            # Its own routing group, so later steps' jobs keep their model
            page.get_by_label("Routing group").fill("e2e-groq")
            run.shot(page, "provider-connect-groq")
            before = len(run.api)
            page.locator(".form-actions").get_by_role("button", name="Save").click()
            _wait(
                page,
                lambda: (
                    run.saw(
                        "POST",
                        r"^/api/settings/llm/deployments/batch$",
                        200,
                        since=before,
                    )
                    # The page reloads the providers after a save; leaving
                    # before that cancels the fetch (WebKit logs it)
                    and run.saw(
                        "GET", r"^/api/settings/llm/providers", 200, since=before
                    )
                ),
                what="the saved models",
            )
            stored = json.loads(config.read_text(encoding="utf-8"))["llm"]
            groq = [p for p in stored["providers"] if p.get("provider") == "groq"]
            if [(p.get("api_key"), p.get("api_base")) for p in groq] != [
                ("gsk-e2e-0000-1111", address)
            ]:
                raise AssertionError(f"saved connection: {groq}")
            added = [
                m
                for m in stored["model_list"]
                if m["litellm_params"]["model"] == "groq/fake-model"
            ]
            if [m["model_name"] for m in added] != ["e2e-groq"]:
                raise AssertionError(f"saved models: {added}")
            if "gsk-e2e-0000-1111" in page.content():
                raise AssertionError("the key came back to the page")
            return "Groq connected with a key and an address; its list loaded; one model saved"

        run.step(page, "settings-connect-provider", connect_provider)

        # 13c. Add models from an environment card: the server's own
        # OPENAI_API_KEY. Its list is OpenAI's (unavailable here, see the
        # proxy above), a model typed by hand is saved with the variable's
        # name, and the card becomes that saved connection
        def environment_card() -> str:
            variable, secret = ENV_KEY
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name="Add models"))
            panel = page.locator(".provider-picker")
            panel.wait_for()
            row = panel.locator(".provider-row").filter(has_text=f"Using {variable}")
            before = len(run.api)
            _click_visible(row.locator(".provider-pick"))
            _wait(
                page,
                lambda: run.saw(
                    "POST", r"^/api/settings/llm/model-discovery$", 200, since=before
                ),
                what="the environment card's model list",
            )
            page.get_by_label("Add a model by ID").fill("gpt-e2e-env")
            page.get_by_role("button", name="Add", exact=True).click()
            page.get_by_label("Routing group").fill("e2e-env")
            run.shot(page, "provider-environment-card")
            before = len(run.api)
            page.locator(".form-actions").get_by_role("button", name="Save").click()
            _wait(
                page,
                lambda: (
                    run.saw(
                        "POST",
                        r"^/api/settings/llm/deployments/batch$",
                        200,
                        since=before,
                    )
                    and run.saw(
                        "GET", r"^/api/settings/llm/providers", 200, since=before
                    )
                ),
                what="the model saved from the environment card",
            )
            text = config.read_text(encoding="utf-8")
            stored = json.loads(text)["llm"]
            added = [
                m
                for m in stored["model_list"]
                if m["litellm_params"]["model"] == "openai/gpt-e2e-env"
            ]
            if [(m["model_name"], m["litellm_params"]) for m in added] != [
                (
                    "e2e-env",
                    {"model": "openai/gpt-e2e-env", "api_key": f"env:{variable}"},
                )
            ]:
                raise AssertionError(f"saved models: {added}")
            linked = [
                p
                for p in stored["providers"]
                if p["id"] == added[0]["model_info"].get("provider_id")
            ]
            if [
                (p["provider"], p.get("api_key"), p.get("api_base")) for p in linked
            ] != [("openai", f"env:{variable}", None)]:
                raise AssertionError(f"saved connection: {linked}")
            if secret in text or secret in page.content():
                raise AssertionError("the environment's key was copied")
            return f"a model added from the {variable} card; saved as env:{variable}"

        run.step(page, "settings-environment-card", environment_card)

        # 13d. Vertex AI signs in with the server's Google login at its own
        # endpoint: Add models offers it no address (the settings API refuses
        # one, contract v1.9) and says so; Bedrock keeps its address. The
        # card needs google-auth (``uv run --with google-auth``)
        def vertex_ai() -> str:
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name="Add models"))
            panel = page.locator(".provider-picker")
            panel.wait_for()
            search = panel.get_by_role("searchbox")
            detail = page.locator(".provider-detail")
            address = detail.locator("input[type=url]")
            note = page.locator(".login-note")
            # Bedrock: optional key and address, no login note
            search.fill("bedrock")
            _click_visible(
                panel.locator(".provider-item").filter(
                    has_text=re.compile(r"^\s*Amazon Bedrock\s*$")
                )
            )
            detail.wait_for(timeout=10000)
            if address.count() != 1 or note.count():
                raise AssertionError(
                    f"Bedrock: {address.count()} address field(s), "
                    f"{note.count()} login note(s)"
                )
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name="Add models"))
            panel.wait_for()
            search.fill("vertex")
            offered = panel.locator(".provider-item").filter(has_text="Vertex AI")
            page.wait_for_timeout(300)
            if not _google_auth():
                if offered.count():
                    raise AssertionError("Vertex AI offered without google-auth")
                return (
                    "Bedrock offers its address; Vertex AI not offered (no google-auth)"
                )
            _click_visible(offered)
            note.wait_for(timeout=10000)
            said = " ".join(note.inner_text().split())
            if said != "Uses this server's Google login at Vertex AI's own endpoint.":
                raise AssertionError(f"the login note says {said!r}")
            if address.count() or detail.locator("input").count():
                raise AssertionError("Vertex AI offers a key or an address")
            # The form may draw again as it settles (a phone did), dropping
            # what was typed: type until Add takes it
            typed, add = (
                page.get_by_label("Add a model by ID"),
                page.get_by_role("button", name="Add", exact=True),
            )
            for _ in range(10):
                typed.fill("gemini-2.5-flash")
                page.wait_for_timeout(200)
                if typed.input_value() == "gemini-2.5-flash" and add.is_enabled():
                    break
            add.click()
            page.get_by_label("Routing group").fill("e2e-vertex")
            run.shot(page, "provider-connect-vertex")
            before = len(run.api)
            page.locator(".form-actions").get_by_role("button", name="Save").click()
            _wait(
                page,
                lambda: (
                    run.saw(
                        "POST",
                        r"^/api/settings/llm/deployments/batch$",
                        200,
                        since=before,
                    )
                    and run.saw(
                        "GET", r"^/api/settings/llm/providers", 200, since=before
                    )
                ),
                what="the saved Vertex AI model",
            )
            stored = json.loads(config.read_text(encoding="utf-8"))["llm"]
            added = [m for m in stored["model_list"] if m["model_name"] == "e2e-vertex"]
            if [m["litellm_params"] for m in added] != [
                {"model": "vertex_ai/gemini-2.5-flash"}
            ]:
                raise AssertionError(f"saved models: {added}")
            # Neither key nor address: no connection is stored for it (the
            # list shows the model as its own configured card)
            linked = [
                p
                for p in stored.get("providers", [])
                if p["id"] == added[0]["model_info"].get("provider_id")
            ]
            if any(p.get("api_key") or p.get("api_base") for p in linked):
                raise AssertionError(f"saved connection: {linked}")
            # The saved connection shows its address row without Edit or Reset
            opened = len(run.api)
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            _click_visible(page.get_by_role("button", name="Add models"))
            panel.wait_for()
            row = panel.locator(".provider-row").filter(has_text="Vertex AI")
            _click_visible(row.locator(".provider-pick"))
            base_row = page.locator(".conn-row").filter(has_text="API base")
            base_row.wait_for(timeout=10000)
            page.wait_for_timeout(300)
            if base_row.get_by_role("button").count():
                raise AssertionError(
                    f"the saved connection offers {base_row.get_by_role('button').all_inner_texts()}"
                )
            # ... no key to add (it signs with the Google login), and no
            # model-list request or "could not be loaded" (manual-only)
            key_row = page.locator(".conn-row").filter(has_text="API key")
            if key_row.count():
                raise AssertionError(
                    f"the saved connection offers a key: {key_row.all_inner_texts()}"
                )
            page.wait_for_timeout(700)
            if page.get_by_text("Models could not be loaded").count():
                raise AssertionError("the saved connection says models could not load")
            if run.saw("POST", r"^/api/settings/llm/model-discovery$", since=opened):
                raise AssertionError("the saved connection asked for a model list")
            run.shot(page, "provider-saved-vertex")
            # An API caller naming an address gets the wording the UI translates
            refused = _api(
                page,
                base,
                "POST",
                "/api/settings/llm/deployments/batch",
                data={
                    "expected_revision": _api(
                        page, base, "GET", "/api/settings/llm"
                    ).json()["revision"],
                    "deployments": [
                        {
                            "model_name": "e2e-vertex-address",
                            "model": "vertex_ai/gemini-2.5-flash",
                            "provider": "vertex_ai",
                            "api_base": "https://caller.example/v1",
                        }
                    ],
                },
            )
            detail_text = (
                refused.json().get("detail") if refused.status == 422 else None
            )
            if detail_text != VERTEX_REFUSAL:
                raise AssertionError(
                    f"address refused as {refused.status}: {detail_text!r}"
                )
            return (
                "Bedrock offers its address; Vertex AI: no key or address, the "
                "login note, a model saved without an address, no Edit, key or "
                "model-list error on its connection; an address through the API "
                "is 422 as worded"
            )

        run.step(page, "settings-vertex-ai", vertex_ai)

        # 14. History survives a reload
        def history() -> str:
            before = len(run.api)
            _open(page, lambda: page.goto(f"{base}/jobs"))
            # Reloading before the list's answer would cancel it (see above)
            _wait(
                page,
                lambda: run.saw("GET", r"^/api/history\?include=items$", since=before),
                what="the history",
            )
            _open(page, page.reload)
            _items(page).first.wait_for()
            count = _items(page).count()
            if count < 8:
                raise AssertionError(f"only {count} rows after a reload")
            if not run.saw("GET", r"^/api/history\?include=items$", 200):
                raise AssertionError("history not read with include=items")
            return f"{count} rows restored from history"

        run.step(page, "history", history)

        # 15. A new tab without the token: settings ask for it
        def token_prompt() -> str:
            fresh = browser.new_context(
                viewport={"width": width, "height": 844 if phone else 900},
                device_scale_factor=2 if phone else 1,
                color_scheme=theme,
            )
            other = fresh.new_page()
            other.on("response", record)
            other.on("pageerror", run.page_error)
            other.on("requestfailed", run.request_failed)
            try:
                _open(other, lambda: other.goto(f"{base}/settings/models"))
                other.wait_for_timeout(1500)
                if not run.saw("GET", r"^/api/settings/llm", 401):
                    models = other.get_by_role("link", name="Models")
                    if models.count():
                        _click_visible(models)
                _wait(
                    other,
                    lambda: run.saw("GET", r"^/api/settings/llm", 401),
                    timeout=15,
                    what="the 401",
                )
                _wait(
                    other,
                    lambda: (
                        _visible(other.get_by_text(re.compile("token", re.IGNORECASE)))
                        is not None
                    ),
                    timeout=15,
                    what="the token message",
                )
                # The token is asked for on the page, not in a banner
                _no_banners(other)
            finally:
                run.shot(other, "token-required")
                fresh.close()
            return "settings without the token answer 401 and the UI asks for the token"

        run.step(page, "token-prompt-checked", token_prompt)
        browser.close()
    run.settle_page_errors(engine)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(tempfile.gettempdir()) / "markitai-workbench-e2e",
    )
    parser.add_argument(
        "--widths", default="1440,390", help="viewport widths, comma-separated"
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        help="serve this unpacked bundle instead of the vendored one",
    )
    parser.add_argument(
        "--browser",
        choices=("chromium", "webkit", "firefox"),
        default="chromium",
        help="browser engine (webkit is Safari's; install it with playwright install)",
    )
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default="light",
        help="the browser's colour scheme (the workbench follows it by default)",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    _, llm_port = _serve(_FakeLLM)
    _, pages_port = _serve(_Pages)
    results: dict[str, Any] = {
        "widths": {},
        "bundle": None,
        "browser": args.browser,
        "theme": args.theme,
    }
    interrupted: BaseException | None = None
    scratch = Path(tempfile.mkdtemp(prefix="markitai-e2e-"))
    try:
        inputs = make_inputs(scratch / "inputs", pages_port)
        for width in (int(w) for w in args.widths.split(",")):
            entry: dict[str, Any] = {"steps": [], "errors": [], "api_calls": 0}
            results["widths"][width] = entry
            shutdown: dict[str, Any] = {}
            try:
                with run_server(
                    scratch / f"home-{width}",
                    _free_port(),
                    llm_port,
                    args.bundle,
                    args.out / f"server-{width}.log",
                    shutdown,
                ) as base:
                    caps = json.loads(
                        __import__("urllib.request")
                        .request.urlopen(f"{base}/api/capabilities")
                        .read()
                    )
                    results["version"] = caps.get("version")
                    run = browse(
                        base,
                        width,
                        args.out,
                        inputs,
                        pages_port,
                        args.browser,
                        llm_port=llm_port,
                        config=scratch / f"home-{width}" / ".markitai" / "config.json",
                        theme=args.theme,
                    )
                    entry.update(
                        steps=run.steps,
                        errors=run.errors,
                        api_calls=len(run.api),
                        navigation_cancels=run.navigation_cancels,
                    )
            except Exception as error:  # noqa: BLE001 - recorded; the next width runs
                entry["errors"].append(
                    f"run aborted: {type(error).__name__}: {error}"[:400]
                )
                print(traceback.format_exc(), file=sys.stderr, flush=True)
            entry["steps"].append(shutdown_step(shutdown))
            print(
                f"[{width}] {'ok  ' if entry['steps'][-1]['ok'] else 'FAIL'} "
                f"shutdown: {entry['steps'][-1]['note']}",
                flush=True,
            )
    except (Exception, KeyboardInterrupt) as error:  # noqa: BLE001 - still write what ran
        interrupted = error
        results["aborted"] = f"{type(error).__name__}: {error}"[:400]
    finally:
        # Every server and its children are stopped by now: on Windows a
        # process left with a file or its working directory in here makes
        # the delete fail, which is a leak to report, not to crash on
        cleanup_error = remove_tree(scratch)
        if cleanup_error:
            results["cleanup_error"] = cleanup_error
    failed = (
        interrupted is not None
        or bool(results.get("cleanup_error"))
        or any(
            data["errors"] or not all(step["ok"] for step in data["steps"])
            for data in results["widths"].values()
        )
    )
    from importlib.resources import files

    manifest_path = (
        args.bundle / "manifest.json"
        if args.bundle
        else Path(str(files("markitai.serve") / "workbench" / "manifest.json"))
    )
    if manifest_path.is_file():
        results["bundle"] = json.loads(manifest_path.read_text(encoding="utf-8")).get(
            "source_commit"
        )
    (args.out / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    lines = [
        "# markitai serve: web workbench end-to-end run",
        "",
        f"markitai {results.get('version')}, workbench bundle `{results['bundle']}`, {sys.platform}, Python {sys.version.split()[0]}, {args.browser}, {args.theme}.",
        "Offline: fake LLM, local pages, generated inputs, throwaway HOME.",
        "",
        "| Width | State | Result | Note | Screenshot |",
        "|---|---|---|---|---|",
    ]
    for width, data in results["widths"].items():
        for step in data["steps"]:
            shot = step["screenshot"]
            lines.append(
                f"| {width} | {step['state']} | {'ok' if step['ok'] else '**FAIL**'} | {step['note']} | {f'[{shot}]({shot})' if shot else ''} |"
            )
        for error in data["errors"]:
            lines.append(f"| {width} | error | **FAIL** | {error} | |")
        if data.get("navigation_cancels"):
            lines.append(
                f"| {width} | cancelled | ok | requests a navigation cancelled "
                f"(WebKit logs them as access-control errors): "
                f"{', '.join(data['navigation_cancels'])} | |"
            )
    for key, state in (("cleanup_error", "cleanup"), ("aborted", "aborted")):
        if results.get(key):
            lines.append(f"| | {state} | **FAIL** | {results[key]} | |")
    (args.out / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{'FAILED' if failed else 'passed'}: {args.out / 'INDEX.md'}")
    if isinstance(interrupted, KeyboardInterrupt):
        raise interrupted
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
