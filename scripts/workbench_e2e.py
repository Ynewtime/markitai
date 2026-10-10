#!/usr/bin/env python3
"""Drive the web workbench in a headless browser against `markitai serve`.

Everything runs locally and offline: an OpenAI-compatible fake LLM and a page
server start in this process, the inputs are generated, and the server gets a
throwaway HOME and MARKITAI_HOME (your ~/.markitai is never read or written).
For each viewport width it walks the main flows with Playwright/Chromium and
screenshots every state:

    convert page, staged files + URLs, live progress, finished list,
    inspector, preview, compare (original page beside the Markdown),
    retry of a failed item, stop remaining (cancel), download all (ticket),
    settings model test (fake LLM), history after a reload, token prompt

It fails on a page error, a 5xx answer or a step whose check does not hold.

Usage (needs the serve and browser extras, and the browser for Playwright:
``python -m playwright install chromium``, or ``webkit`` / ``firefox`` for
``--browser``)::

    uv run python scripts/workbench_e2e.py --out /tmp/workbench-e2e
    uv run python scripts/workbench_e2e.py --widths 1440 --bundle path/to/unpacked-bundle

Results: ``OUT/<NN>-<state>-<width>.png``, ``OUT/results.json`` and ``OUT/INDEX.md``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

TOKEN = "e2e-token-0000000000000000"
LLM_DELAY_S = 0.8
SLOW_PAGE_S = 2.5

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
    """``/page/<n>.html`` answers at once, ``/slow/<n>.html`` after a pause."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        match = re.fullmatch(r"/(page|slow)/(\w+)\.html", self.path)
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


def _serve(handler: type[BaseHTTPRequestHandler]) -> tuple[ThreadingHTTPServer, int]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
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
    report = directory / "quarterly-report.pdf"
    pdf.save(report)
    image = Image.new("RGB", (900, 600), "white")
    draw = ImageDraw.Draw(image)
    for index, height in enumerate((180, 320, 260, 410)):
        draw.rectangle(
            (120 + index * 180, 520 - height, 220 + index * 180, 520),
            fill=(37, 99, 235),
        )
    photo = directory / "bar-chart.png"
    image.save(photo)
    notes = directory / "meeting-notes.md"
    notes.write_text(
        "# Meeting notes\n\n- Ship 1.2.1\n- Review the workbench\n\n```python\nprint('hello')\n```\n",
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
    home: Path, port: int, llm_port: int, bundle: Path | None, log: Path
) -> Iterator[str]:
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
            ]
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
        PYTHONUTF8="1",
    )
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
    with log.open("w", encoding="utf-8") as sink:
        process = subprocess.Popen(
            command, cwd=home, env=env, stdout=sink, stderr=subprocess.STDOUT
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
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()


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

    def shot(self, page: Any, state: str) -> str:
        self.counter += 1
        name = f"{self.counter:02d}-{state}-{self.width}.png"
        page.screenshot(path=str(self.out / name))
        return name

    def step(self, page: Any, state: str, check: Callable[[], str | None]) -> None:
        started = time.monotonic()
        try:
            note = check() or ""
            ok = True
        except Exception as error:  # noqa: BLE001 - a failed check is recorded, the run goes on
            note, ok = f"{type(error).__name__}: {error}"[:400], False
        shot = self.shot(page, state)
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

    def saw(self, method: str, pattern: str, status: int | None = None) -> bool:
        return any(
            m == method and re.search(pattern, path) and (status is None or s == status)
            for m, path, s in self.api
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


def _api(page: Any, base: str, method: str, path: str, **kwargs: Any) -> Any:
    headers = {"Authorization": f"Bearer {TOKEN}"}
    return page.request.fetch(f"{base}{path}", method=method, headers=headers, **kwargs)


def browse(
    base: str,
    width: int,
    out: Path,
    inputs: dict[str, Path],
    pages_port: int,
    engine: str = "chromium",
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
        )
        page = context.new_page()
        page.on("pageerror", lambda error: run.errors.append(f"pageerror: {error}"))

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
            page.goto(f"{base}/#token={TOKEN}")
            page.get_by_role("button", name="Convert").wait_for()
            if "#token" in page.url:
                raise AssertionError("the token stayed in the address bar")
            return "token taken from the fragment and removed"

        run.step(page, "convert", home)

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

        # 4. Inspector of the PDF
        def inspector() -> str:
            _dismiss(page)
            _item(page, "quarterly-report.pdf").click()
            _heading(page, "quarterly-report.pdf").wait_for()
            page.get_by_text("No model used").first.wait_for()
            _wait(
                page,
                lambda: run.saw("GET", r"/result\?max_bytes=16384", 200),
                what="the result excerpt",
            )
            return "inspector shows source size, output, cost and the result excerpt"

        run.step(page, "inspector", inspector)

        # 5. Preview
        def preview() -> str:
            _click_visible(page.get_by_role("button", name="Preview", exact=True))
            page.get_by_role("tab", name=re.compile("Rendered|Source")).first.wait_for()
            page.get_by_text("Quarterly report, part 1").first.wait_for()
            return "rendered Markdown in the preview"

        run.step(page, "preview", preview)

        # 6. Compare: the original page beside the Markdown
        def compare() -> str:
            _click_visible(page.get_by_role("button", name="Compare", exact=True))
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
            return "PDF page image and text layer loaded"

        run.step(page, "compare", compare)
        page.keyboard.press("Escape")
        page.wait_for_timeout(300)
        page.keyboard.press("Escape")

        # 7. Retry a failed item
        def retry() -> str:
            page.goto(f"{base}/jobs")
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

        # 8. Stop remaining
        def stop() -> str:
            _dismiss(page)
            page.get_by_role("textbox", name=re.compile("Paste URLs")).fill(
                "\n".join(
                    f"http://127.0.0.1:{pages_port}/slow/s{n}.html" for n in range(1, 6)
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
            return "queued URLs stopped, the running one finished"

        run.step(page, "stopped", stop)

        # 9. Download all, through a single-use ticket
        def download_all() -> str:
            _dismiss(page)
            with page.expect_download(timeout=60000) as info:
                _click_visible(page.get_by_role("button", name="Download all"))
            download = info.value
            data = Path(download.path()).read_bytes()
            names = zipfile.ZipFile(io.BytesIO(data)).namelist()
            if not names:
                raise AssertionError("empty archive")
            if not run.saw("POST", r"^/api/download-tickets$", 201):
                raise AssertionError("the download did not use a ticket")
            return f"{download.suggested_filename}: {len(names)} files, {len(data)} bytes via a ticket"

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
            page.reload()
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

        # 11. Settings: test the configured model
        def settings() -> str:
            page.goto(f"{base}/settings/models")
            _click_visible(page.get_by_role("button", name=re.compile(r"^Test ")))
            _wait(
                page,
                lambda: run.saw("POST", r"^/api/settings/llm/test$", 200),
                what="the model test",
            )
            page.wait_for_timeout(LLM_DELAY_S * 1000 + 800)
            return "model test answered by the fake LLM"

        run.step(page, "settings-test", settings)

        # 12. History survives a reload
        def history() -> str:
            page.goto(f"{base}/jobs")
            page.reload()
            _items(page).first.wait_for()
            count = _items(page).count()
            if count < 8:
                raise AssertionError(f"only {count} rows after a reload")
            if not run.saw("GET", r"^/api/history\?include=items$", 200):
                raise AssertionError("history not read with include=items")
            return f"{count} rows restored from history"

        run.step(page, "history", history)

        # 13. A new tab without the token: settings ask for it
        def token_prompt() -> str:
            fresh = browser.new_context(
                viewport={"width": width, "height": 844 if phone else 900},
                device_scale_factor=2 if phone else 1,
            )
            other = fresh.new_page()
            other.on("response", record)
            other.on(
                "pageerror", lambda error: run.errors.append(f"pageerror: {error}")
            )
            try:
                other.goto(f"{base}/settings/models")
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
            finally:
                run.shot(other, "token-required")
                fresh.close()
            return "settings without the token answer 401 and the UI asks for the token"

        run.step(page, "token-prompt-checked", token_prompt)
        browser.close()
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
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    _, llm_port = _serve(_FakeLLM)
    _, pages_port = _serve(_Pages)
    results: dict[str, Any] = {"widths": {}, "bundle": None, "browser": args.browser}
    failed = False
    with tempfile.TemporaryDirectory(prefix="markitai-e2e-") as scratch:
        inputs = make_inputs(Path(scratch) / "inputs", pages_port)
        for width in (int(w) for w in args.widths.split(",")):
            home = Path(scratch) / f"home-{width}"
            with run_server(
                home,
                _free_port(),
                llm_port,
                args.bundle,
                args.out / f"server-{width}.log",
            ) as base:
                caps = json.loads(
                    __import__("urllib.request")
                    .request.urlopen(f"{base}/api/capabilities")
                    .read()
                )
                run = browse(base, width, args.out, inputs, pages_port, args.browser)
            results["widths"][width] = {
                "steps": run.steps,
                "errors": run.errors,
                "api_calls": len(run.api),
            }
            results["version"] = caps.get("version")
            failed = (
                failed or bool(run.errors) or not all(step["ok"] for step in run.steps)
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
        f"markitai {results.get('version')}, workbench bundle `{results['bundle']}`, {sys.platform}, Python {sys.version.split()[0]}, {args.browser}.",
        "Offline: fake LLM, local pages, generated inputs, throwaway HOME.",
        "",
        "| Width | State | Result | Note | Screenshot |",
        "|---|---|---|---|---|",
    ]
    for width, data in results["widths"].items():
        for step in data["steps"]:
            lines.append(
                f"| {width} | {step['state']} | {'ok' if step['ok'] else '**FAIL**'} | {step['note']} | [{step['screenshot']}]({step['screenshot']}) |"
            )
        for error in data["errors"]:
            lines.append(f"| {width} | error | **FAIL** | {error} | |")
    (args.out / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{'FAILED' if failed else 'passed'}: {args.out / 'INDEX.md'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
