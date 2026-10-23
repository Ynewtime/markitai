#!/usr/bin/env python3
"""Drive the web workbench in a headless browser against `markitai serve`.

Everything runs locally and offline: an OpenAI-compatible fake LLM and a page
server start in this process, the inputs are generated, and the server gets a
throwaway HOME and MARKITAI_HOME (your ~/.markitai is never read or written).
For each viewport width it walks the main flows with Playwright/Chromium and
screenshots every state:

    convert page, a refused URL as a bottom error toast, staged files +
    URLs, live progress, finished list, a conversion in the peek window
    (centred; full screen on a phone) opening on Preview, with its Compare and
    Details tabs (original page beside the Markdown, C to switch), the header's
    Download opening Preview's Files (the original first, Download all (.zip)
    through the item's archive), Up/Down and the header arrows between items,
    a URL item's Compare framing the page as fetched (its style drawn, its
    script never run, nothing loaded from the site; hovering a block on either
    side outlines both; the Markdown's https picture drawn in Compare and
    Preview, from a local site whose certificate a local test CA issued), the window at full
    screen and back (F; a per-browser choice), retry of a failed item, a failed action reported
    in one error toast, New conversion and stop remaining (cancel), download all
    from the "⋯" menu (ticket; on a phone Stop from the header of the
    full-screen windows and on its own row under the Conversions title, a
    long URL title cut in the middle), an inline SVG drawing in the preview,
    settings model test (fake LLM) and a model whose key the server lacks
    (the test names the variable), every settings page and Convert as one
    centred column, the provider picker (groups, brand marks, search), a
    saved address shown without its credentials, a provider beyond the first
    eight connected with a key and an address, Vertex AI connected without an
    address (its login note; Bedrock keeps its address; its saved connection
    offers no key and asks for no model list), a hand-written DeepSeek entry
    served by DEEPSEEK_API_KEY (one saved row marked Added, Edit "From
    environment variable", its unreachable list said why), history after a
    reload, token prompt, three conversions deleted in one request (rows and a
    submission header checked, the confirmation), a user token made on
    Settings → Access token and its own tab (only its conversions, its own
    Models and Edit config without the server's file, "Admin token needed" on
    Access token), Settings → Edit config (secrets
    masked, the JSON check, Review changes, Save writing exactly that change,
    a file changed behind the page refused as stale; a CRLF file without a
    final line break saved with only the changed line different), and Settings → Access
    token → Regenerate (the old token refused, another tab holding it too).
    With ``--axe axe.min.js`` axe-core checks every screenshot's page (the
    page's own policy kept) and a violation fails the run

Notices must be toasts at the bottom (never the page-wide banners or the
strips the UI had before): errors stay until closed, a failed action on one
conversion is one toast that trying again replaces.

It fails on a page error, a 5xx answer or a step whose check does not hold.

Usage (needs the serve and browser extras, and the browser for Playwright:
``python -m playwright install chromium``, or ``webkit`` / ``firefox`` for
``--browser``)::

    uv run python scripts/workbench_e2e.py --out /tmp/workbench-e2e
    uv run --with google-auth python scripts/workbench_e2e.py   # + the Vertex AI card
    uv run python scripts/workbench_e2e.py --browser webkit --theme dark
    uv run python scripts/workbench_e2e.py --widths 1440 --bundle path/to/unpacked-bundle
    uv run python scripts/workbench_e2e.py --axe path/to/axe.min.js

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
# The user token the admin makes in the run (contract v1.13 access levels)
USER_LABEL = "Reading group laptop"
# axe-core's source (--axe): every screenshot is checked, violations fail
AXE_SOURCE = ""
LLM_DELAY_S = 0.8
SLOW_PAGE_S = 2.5
# A saved connection whose address carries credentials: settings must show
# it masked and never send these parts to the browser
CONNECTION_ID = "00000000-0000-4000-8000-0000000000c1"
# Models the fake LLM refuses, with the status and the words it answers
REFUSALS = {
    "fake-401": (401, "Authentication Fails (e2e)"),
    "fake-404": (404, "The model fake-404 does not exist"),
    "fake-429": (429, "Rate limit reached for requests (e2e)"),
    "fake-500": (500, "Upstream overloaded (e2e)"),
}
BASE_SECRETS = ("e2e-user", "e2e-pass-0000", "e2e-base-secret-0000")
# The server's own OpenAI key, behind the environment card
ENV_KEY = ("OPENAI_API_KEY", "sk-e2e-env-0000")
# A model entry without a key, as a user wrote deepseek/deepseek-flash
HAND_MODEL = "openai/fake-hand"
# The case: a DeepSeek entry written by hand, at DeepSeek's own
# endpoint, served by the server's DEEPSEEK_API_KEY (one saved row, contract
# v1.11). Its list goes to api.deepseek.com, which the dead proxy refuses.
# Conversions balance over every configured model, so its chat requests go
# to the fake LLM through DEEPSEEK_API_BASE (LiteLLM reads it), and its ID
# has no price, as the cost step expects
DEEPSEEK_MODEL = "deepseek/deepseek-e2e"
DEEPSEEK_KEY = ("DEEPSEEK_API_KEY", "sk-e2e-deepseek-0000")
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

#: A Mermaid block the preview draws as a diagram (bundle 7b2355b, Pele), and
#: one that does not parse, which stays a code block with a note
MERMAID = (
    "```mermaid\nflowchart LR\n"
    "  draft[Draft notes] --> review[Review] --> ship[Ship 1.3.0]\n```"
)
BROKEN_MERMAID = "```mermaid\nflowchart LR\n  draft --> [unclosed\n```"


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


#: The Authorization header of each model list the fake LLM served
MODEL_LIST_KEYS: list[str] = []


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
            MODEL_LIST_KEYS.append(self.headers.get("authorization") or "")
            self._json(
                200,
                {"object": "list", "data": [{"id": "fake-model", "object": "model"}]},
            )
        else:
            self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("content-length") or 0))
        body = json.loads(raw or b"{}")
        refusal = REFUSALS.get(str(body.get("model")))
        if refusal is not None:
            # A model test's refusal, as an OpenAI-compatible provider says it
            status, words = refusal
            self._json(status, {"error": {"message": words, "type": "e2e"}})
            return
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


#: Paths the page server was asked for: the framed page (contract v1.12)
#: runs no script and loads nothing over plain HTTP
PAGE_REQUESTS: list[str] = []
#: The site's files over HTTPS (``https://127.0.0.1:<port>``, a self-signed
#: certificate the browser is told to accept): the framed page loads them.
#: Empty when no certificate could be made (no ``cryptography``).
SITE_ASSETS = ""
# The base64 SHA-256 of the HTTPS site's public key, trusted by Chromium
SITE_SPKI = ""
ASSET_REQUESTS: list[str] = []


class _Pages(BaseHTTPRequestHandler):
    """``/page/<n>.html`` answers at once, ``/slow/[<dir>/…]<n>.html`` after
    a pause. Each page carries an inline style, a script (it would ask for
    ``/script-ran``), a picture and style sheet from the site over HTTP, and
    another picture and style sheet from it over HTTPS (:data:`SITE_ASSETS`)."""

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        PAGE_REQUESTS.append(self.path)
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
            f"<!doctype html><html><head><title>Page {match.group(2)}</title>"
            "<style>h1{color:rgb(200, 0, 0)}</style>"
            '<link rel="stylesheet" href="/site.css"></head><body><article>'
            '<script>document.body.dataset.ran="yes";fetch("/script-ran")</script>'
            '<img src="/remote.png" alt="Remote picture">'
            + (
                f'<link rel="stylesheet" href="{SITE_ASSETS}/site.css">'
                f'<img src="{SITE_ASSETS}/site.png" alt="Site picture">'
                if SITE_ASSETS
                else ""
            )
            + f"<h1>Page {match.group(2)}</h1><p>A local page for the workbench end-to-end run. "
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


class _SiteAssets(BaseHTTPRequestHandler):
    """The site's picture (``/site.png``) and style sheet (``/site.css``)."""

    @staticmethod
    def _png() -> bytes:
        """A 4 × 4 green PNG."""
        import struct
        import zlib

        def chunk(kind: bytes, data: bytes) -> bytes:
            body = kind + data
            return (
                struct.pack(">I", len(data))
                + body
                + struct.pack(">I", zlib.crc32(body))
            )

        rows = b"".join(b"\x00" + b"\x00\x78\x00" * 4 for _ in range(4))
        header = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
        return (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b"")
        )

    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        ASSET_REQUESTS.append(self.path)
        body, kind = {
            "/site.png": (self._png(), "image/png"),
            "/site.css": (b"article{border-top:6px solid rgb(0, 120, 0)}", "text/css"),
        }.get(self.path, (b"", ""))
        self.send_response(200 if kind else 404)
        if kind:
            self.send_header("content-type", kind)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _serve_https(
    handler: type[BaseHTTPRequestHandler], directory: Path
) -> tuple[ThreadingHTTPServer, int, str] | None:
    """*handler* over HTTPS on 127.0.0.1, its certificate issued by a local CA.

    A fresh test CA (``ca.crt``) signs the site's certificate for 127.0.0.1.
    Returns the server, its port and the base64 SHA-256 of the site key's
    SubjectPublicKeyInfo, which Chromium is told to trust
    (``--ignore-certificate-errors-spki-list``): exactly this site, nothing
    else. Other engines fall back to ignoring certificate errors.
    """
    import base64
    import datetime
    import hashlib
    import ssl

    try:
        from cryptography import x509
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.x509.oid import NameOID
    except ImportError:
        return None
    import ipaddress

    now = datetime.datetime.now(datetime.UTC)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name(
        [x509.NameAttribute(NameOID.COMMON_NAME, "markitai e2e test CA")]
    )
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    certificate = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "127.0.0.1")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]
            ),
            critical=False,
        )
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "ca.crt").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    cert_file, key_file = directory / "site.crt", directory / "site.key"
    cert_file.write_bytes(
        certificate.public_bytes(serialization.Encoding.PEM)
        + ca.public_bytes(serialization.Encoding.PEM)
    )
    key_file.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    spki = base64.b64encode(
        hashlib.sha256(
            key.public_key().public_bytes(
                serialization.Encoding.DER,
                serialization.PublicFormat.SubjectPublicKeyInfo,
            )
        ).digest()
    ).decode()
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(cert_file, key_file)
    server = _LocalServer(("127.0.0.1", 0), handler)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, server.server_address[1], spki


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
        "# Meeting notes\n\n- Ship 1.3.0\n- Review the workbench\n\n```python\nprint('hello')\n```\n\n"
        f"{DRAWING}\n\n{MERMAID}\n\n{BROKEN_MERMAID}\n",
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
                },
                # Written by hand: no key, the server's OPENAI_API_KEY serves
                # it, as LiteLLM does for its conversions.
                # Its own address text: Edit opens the first connection of
                # the model's provider at the model's address, and the others
                # here are at 127.0.0.1
                {
                    "model_name": "e2e-hand",
                    "litellm_params": {
                        "model": HAND_MODEL,
                        "api_base": f"http://localhost:{llm_port}/v1",
                    },
                },
                {
                    "model_name": "e2e-deepseek",
                    "litellm_params": {"model": DEEPSEEK_MODEL},
                },
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
    env[DEEPSEEK_KEY[0]] = DEEPSEEK_KEY[1]
    env["DEEPSEEK_API_BASE"] = f"http://127.0.0.1:{llm_port}/v1"
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
    axe_checked: int = 0

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
        if AXE_SOURCE:
            self.check_accessibility(page, name)
        return name

    def check_accessibility(self, page: Any, shot: str) -> None:
        """Run axe-core on the page as shot; a violation is a run error.

        Injected through evaluate, so the page keeps its own policy (a
        script tag would be refused by it).
        """
        try:
            # Contrast is measured on what is drawn: let fades and slides end
            # (not spinners, which run until their work is done)
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline and page.evaluate(
                "document.getAnimations().some((a) => a.playState === 'running' && "
                "a.effect?.getTiming().iterations !== Infinity)"
            ):
                page.wait_for_timeout(100)
            page.evaluate(AXE_SOURCE)
            violations = page.evaluate(
                "async () => (await axe.run(document, {resultTypes: ['violations']}))"
                ".violations.map((v) => ({id: v.id, impact: v.impact, "
                "nodes: v.nodes.length, target: v.nodes[0]?.target}))"
            )
        except Exception as error:  # noqa: BLE001 - recorded as a failure
            violations = [{"id": "axe-failed", "error": str(error)[:200]}]
        self.axe_checked += 1
        with (self.out / f"axe-{self.width}.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"shot": shot, "violations": violations}) + "\n")
        for violation in violations:
            self.errors.append(f"axe {shot}: {json.dumps(violation)}"[:400])

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
    # 46a0489 draws a name cut to fit, a sizer and a spoken copy: the whole
    # name is the drawn one's title
    title = page.locator("#peek-title")
    name = title.locator(".mid-name")
    text = (
        name.first.get_attribute("title") or "" if name.count() else title.inner_text()
    )
    return re.sub(r"\s+", "", text)


def _tab(page: Any, name: str) -> Any:
    """A tab of the window's own row (Preview, Compare, Details); Preview's
    views share the row on a wide screen but are not in this tab list."""
    return page.locator(".peek-tablist").get_by_role("tab", name=name, exact=True)


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
    """The Conversions header's buttons: 28 px; on a phone one compact 32 px
    size whose tap area an invisible margin keeps at 44 px or more."""
    box = locator.bounding_box()
    expected = 32 if phone else 28
    if box is None or abs(box["height"] - expected) > 1:
        raise AssertionError(
            f"a button is {box and box['height']} px high, not {expected}"
        )
    if not phone:
        return f"{expected} px buttons"
    reach = locator.evaluate(
        "b => { const s = getComputedStyle(b, '::before');"
        " return s.content === 'none' ? 0 : b.getBoundingClientRect().height"
        " - parseFloat(s.top) - parseFloat(s.bottom); }"
    )
    if reach < 44:
        raise AssertionError(f"the button's tap area is {reach} px high, under 44")
    return f"{expected} px buttons with a {reach:.0f} px tap area"


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
    shortened in the middle at folder boundaries (46a0489: one drawn line
    with the "…" in it, the host and the last segment kept)."""
    title = page.locator("#peek-title")
    name = title.locator(".mid-name")
    line = name.locator(".mid-line")
    if not name.count() or not line.count():
        raise AssertionError("the window title is not a middle-cut name")
    full = name.get_attribute("title") or ""
    drawn = line.inner_text()
    if not drawn.startswith(f"http://{host}"):
        raise AssertionError(f"the title starts {drawn!r}")
    if not drawn.endswith(end):
        raise AssertionError(f"the title ends {drawn!r}")
    if drawn == full or "…" not in drawn:
        raise AssertionError(f"the title {full!r} is not shortened: {drawn!r}")
    # Cut at a folder: what follows the "…" is a whole tail of the address
    after = drawn.split("…", 1)[1]
    if not after.startswith("/") or not full.endswith(after):
        raise AssertionError(f"the title is not cut at a folder: {drawn!r}")
    if not _fits(name) or not line.is_visible():
        raise AssertionError(f"the drawn title overflows: {drawn!r}")
    close = _peek(page).get_by_role("button", name="Close").bounding_box()
    box = title.bounding_box()
    if close is None or box is None or box["y"] < close["y"] + close["height"] - 1:
        raise AssertionError(f"the title is not on its own row: {box}, Close {close}")
    return f"title cut at a folder, {drawn!r}, on its own row"


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
OLD_NOTICES = ".notice, .notes-region, .note-close, .note-strip"
# The notice toasts' close button (en.ts `dismiss`)
DISMISS = "Dismiss"


def _no_banners(page: Any) -> None:
    left = page.locator(OLD_NOTICES).count()
    if left:
        raise AssertionError(f"{left} page-wide notice(s) of the old UI on the page")


def _strip(page: Any, tone: str, text: str | re.Pattern[str]) -> Any:
    """The notice toast of *tone* saying *text*, once it is shown."""
    strip = page.locator(f".toasts .toast.is-{tone}").filter(has_text=text)
    strip.first.wait_for(timeout=15000)
    return strip.first


def _at_the_bottom(page: Any, strip: Any) -> str:
    """Where a toast sits: centred directly above the tab bar on a phone
    (12 px from each edge, at most 440 px wide), a 380 px card 24 px from
    the bottom-right corner on a wide screen; a card of a line or two."""
    box, settled = strip.bounding_box(), None
    for _ in range(20):
        page.wait_for_timeout(100)
        settled, box = box, strip.bounding_box()
        if box == settled:
            break
    view = page.viewport_size
    if box is None or view is None:
        raise AssertionError("the toast has no box")
    width, height = view["width"], view["height"]
    bottom, right = box["y"] + box["height"], box["x"] + box["width"]
    if box["height"] > 90:
        raise AssertionError(
            f"the toast is {box['height']:.0f} px high, not a short card"
        )
    if bottom > height or box["y"] < height * 0.6:
        raise AssertionError(f"the toast is not at the bottom: {box}")
    if width < 600:
        expected = min(width - 24, 440)
        if (
            abs(box["width"] - expected) > 2
            or abs(box["x"] - (width - box["width"]) / 2) > 2
        ):
            raise AssertionError(f"the toast is not centred at full width: {box}")
        bar = _visible(page.get_by_role("navigation"))
        top = bar.bounding_box() if bar is not None else None
        # The tab bar counts only where it is drawn on top: a full-screen
        # window covers it, and the toast then sits 12 px above the bottom
        shown = (
            top is not None
            and top["y"] > height * 0.6
            and page.evaluate(
                "y => !!document.elementFromPoint(4, y)?.closest('nav')",
                top["y"] + top["height"] / 2,
            )
        )
        if shown and top is not None:
            if bottom > top["y"] + 1:
                raise AssertionError(f"the toast covers the tab bar: {box} / {top}")
            return f"centred, {box['width']:.0f} px wide at y={box['y']:.0f}, above the tab bar"
        if abs(height - bottom - 12) > 2:
            raise AssertionError(f"the toast is not 12 px above the bottom: {box}")
        return f"centred, {box['width']:.0f} px wide, 12 px above the bottom (tab bar covered)"
    if (
        abs(width - right - 24) > 2
        or abs(height - bottom - 24) > 2
        or abs(box["width"] - 380) > 2
    ):
        raise AssertionError(
            f"the toast is not the 380 px card in the bottom-right corner: {box}"
        )
    return f"{box['width']:.0f} px wide, {width - right:.0f} px from the right edge"


def _offered_strategies(page: Any, base: str) -> tuple[list[str], bool]:
    """The fetch strategies the composer's Options → Advanced offers, and
    whether it offers a files backend at all."""
    _open(page, lambda: page.goto(f"{base}/"))
    page.get_by_role("button", name="Options").first.click()
    advanced = page.get_by_role("button", name="Advanced")
    advanced.wait_for(timeout=10000)
    if advanced.get_attribute("aria-expanded") != "true":
        advanced.click()
    group = page.get_by_role("group", name="URL", exact=True)
    group.wait_for(timeout=10000)
    choices = [
        " ".join(t.split()) for t in group.get_by_role("button").all_inner_texts()
    ]
    files = page.get_by_role("group", name="Files", exact=True).count() > 0
    page.keyboard.press("Escape")
    return choices, files


def _google_auth() -> bool:
    """Whether the server (this interpreter) can sign Vertex AI requests."""
    try:
        return importlib.util.find_spec("google.auth") is not None
    except ModuleNotFoundError:
        return False


def _hover_links(page: Any, frame: Any) -> str:
    """Hovering a block on either side of Compare outlines both (v1.13).

    Markdown → page: the frame's paired block gets ``data-markitai-linked``
    (the workbench's own style draws its outline); page → Markdown: the
    Markdown block gets ``is-linked``.
    """
    markdown = page.locator("#peek-panel .doc-body h1").first
    markdown.hover()
    heading = frame.locator("h1[data-markitai-linked]")
    heading.wait_for(timeout=10000)
    outline = heading.evaluate("h => getComputedStyle(h).outlineStyle")
    if outline != "solid":
        raise AssertionError(
            f"the page's heading is linked but not outlined ({outline})"
        )
    frame.locator("article p").first.hover()
    linked = page.locator("#peek-panel .doc-body p.is-linked")
    linked.first.wait_for(timeout=10000)
    text = linked.first.inner_text().strip()
    if not text.startswith("A local page for the workbench end-to-end run."):
        raise AssertionError(f"hovering the page's paragraph linked {text!r}")
    page.mouse.move(0, 0)
    return (
        "hover links both ways (Markdown heading → the page's h1 outlined; the "
        "page's paragraph → the Markdown paragraph)"
    )


def _web_picture(scope: Any, alt: str) -> str:
    """The Markdown's absolute https picture is drawn, not a placeholder (v1.13)."""
    picture = scope.locator(f'.doc-body img[alt="{alt}"]').first
    picture.wait_for(timeout=10000)
    picture.scroll_into_view_if_needed()
    source = picture.get_attribute("src") or ""
    if not source.startswith(f"{SITE_ASSETS}/"):
        raise AssertionError(f"the web picture points at {source!r}")
    if picture.get_attribute("referrerpolicy") != "no-referrer":
        raise AssertionError("the web picture sends a referrer")
    deadline = time.monotonic() + 10
    while picture.evaluate("img => img.naturalWidth") != 4:
        if time.monotonic() > deadline:
            raise AssertionError(f"the web picture {source} did not load")
        scope.page.wait_for_timeout(200)
    return f"its https picture drawn from the site ({source}, no referrer)"


def _submit_files(base: str, token: str, files: list[tuple[str, bytes]]) -> str:
    """Convert *files* (minimal preset, no model) as *token*; the job id once done."""
    import urllib.request
    import uuid

    boundary = uuid.uuid4().hex
    parts = [
        f'--{boundary}\r\nContent-Disposition: form-data; name="files"; '
        f'filename="{name}"\r\nContent-Type: text/markdown\r\n\r\n'.encode()
        + content
        + b"\r\n"
        for name, content in files
    ]
    for key, value in (
        ("urls", "[]"),
        ("options", json.dumps({"preset": "minimal", "llm": False})),
    ):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"'
            f"\r\n\r\n{value}\r\n".encode()
        )

    def call(method: str, path: str, body: bytes | None = None) -> Any:
        headers = {"Authorization": f"Bearer {token}"}
        if body is not None:
            headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        request = urllib.request.Request(
            f"{base}{path}", data=body, method=method, headers=headers
        )
        with urllib.request.urlopen(request, timeout=60) as answer:
            return json.loads(answer.read())

    job = call("POST", "/api/jobs", b"".join(parts) + f"--{boundary}--\r\n".encode())[
        "job_id"
    ]
    deadline = time.monotonic() + 60
    while call("GET", f"/api/jobs/{job}")["status"] == "running":
        if time.monotonic() > deadline:
            raise TimeoutError(f"job {job} did not finish")
        time.sleep(0.2)
    return job


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
        # Chromium trusts the site's key from the local test CA, and only
        # it; the other engines take no such switch and ignore the errors
        pinned = engine == "chromium" and bool(SITE_SPKI)
        browser = getattr(playwright, engine).launch(
            args=[f"--ignore-certificate-errors-spki-list={SITE_SPKI}"]
            if pinned
            else []
        )
        context = browser.new_context(
            viewport={"width": width, "height": 844 if phone else 900},
            device_scale_factor=2 if phone else 1,
            accept_downloads=True,
            color_scheme=theme,
            ignore_https_errors=not pinned,
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
            # An error stays until it is closed (done and warning notices
            # leave after 5 s): still there well after that, pointer away
            page.mouse.move(1, 1)
            page.wait_for_timeout(6500)
            if not strip.is_visible():
                raise AssertionError(
                    "the error left by itself; errors stay until closed"
                )
            strip.get_by_role("button", name=DISMISS).click()
            strip.wait_for(state="detached", timeout=5000)
            box.fill("")
            return (
                f"ftp:// refused in a bottom toast ({where}) that stayed until closed"
            )

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

        # 4. The PDF opens in the peek window on its document (Preview first,
        # Details last); Details holds the sizes, cost and result excerpt
        def peek() -> str:
            _dismiss(page)
            before = len(run.api)
            _item(page, "quarterly-report.pdf").click()
            _heading(page, "quarterly-report.pdf").wait_for()
            page.get_by_role("dialog", name=_spaced("quarterly-report.pdf")).wait_for()
            tabs = [
                " ".join(t.split())
                for t in page.locator(".peek-tablist [role=tab]").all_inner_texts()
            ]
            if tabs != ["Preview", "Compare", "Details"]:
                raise AssertionError(f"tabs: {tabs}")
            if _tab(page, "Preview").get_attribute("aria-selected") != "true":
                raise AssertionError("the window did not open on Preview")
            _show_tab(page, "Details")
            page.locator(".peek").get_by_text("No model used").first.wait_for()
            # Details starts the result from what Preview already loaded (the
            # bundle asks for a 16 KiB excerpt only when it has nothing yet):
            # its start is shown, and the result was read once
            excerpt = page.locator(".peek .excerpt-body")
            excerpt.wait_for(timeout=15000)
            _wait(
                page,
                lambda: bool((excerpt.text_content() or "").strip()),
                what="the start of the result in Details",
            )
            reads = [
                call
                for call in run.api[before:]
                if call[0] == "GET" and re.search(r"/items/i\d+/result(\?|$)", call[1])
            ]
            if len(reads) != 1 or reads[0][2] != 200:
                raise AssertionError(f"the result was read as {reads}")
            where = _window_placement(page, phone)
            position = page.locator(".peek-pos")
            shown = (
                f"; position {position.inner_text().strip()!r}"
                if position.count()
                else ""
            )
            return f"window {where} opened on Preview; Details has sizes, cost, excerpt{shown}"

        run.step(page, "peek-details", peek)

        # 5. Preview tab
        def preview() -> str:
            _show_tab(page, "Preview")
            # Its views: tabs at the end of the window's tab row, or on a
            # phone one compact switch named after the view ("Rendered, ...,
            # Preview mode")
            page.get_by_role("tab", name=re.compile("Rendered|Source")).or_(
                page.get_by_role(
                    "button", name=re.compile(r"^Rendered\b.*Preview mode")
                )
            ).first.wait_for()
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

        # 6a. The header's Download opens Preview → Files: the original
        # first, then every result file, the total, and Download all (.zip)
        # through the item's archive (contract v1.11 section 6.7)
        def peek_files() -> str:
            _show_tab(page, "Details")
            page.wait_for_timeout(300)
            if phone:
                _show_tab(page, "Preview")
                page.get_by_role(
                    "button", name=re.compile(r"Preview mode")
                ).first.click()
                page.get_by_role("menuitemradio", name=re.compile(r"^Files")).or_(
                    page.get_by_role("menuitem", name=re.compile(r"^Files"))
                ).or_(
                    page.get_by_role("option", name=re.compile(r"^Files"))
                ).first.click()
            else:
                # v1.20: Download is a split button; its caret lists the
                # formats and Choose files… (Preview's Files)
                page.locator(".peek-head").get_by_role(
                    "button", name="Download as", exact=True
                ).click()
                menu = page.locator(".peek-head .download-menu")
                menu.wait_for(timeout=10000)
                entries = " | ".join(
                    " ".join(text.split())
                    for text in menu.get_by_role("menuitem").all_inner_texts()
                )
                for wanted in ("Markdown", "JSON", "All files", "Choose files"):
                    if wanted not in entries:
                        raise AssertionError(f"Download lists {entries!r}")
                menu.get_by_role("menuitem", name=re.compile(r"^Choose files")).click()
            panel = page.locator("#peek-panel [role=tabpanel]:not([hidden]) .files")
            panel.wait_for(timeout=15000)
            first = panel.locator(".file-list li").first
            first_text = " ".join(first.inner_text().split())
            if "is-original" not in (first.get_attribute("class") or "") or not (
                "quarterly-report.pdf" in first_text and "Original" in first_text
            ):
                raise AssertionError(f"the first file is {first_text!r}")
            paths = panel.locator(
                ".file-list li:not(.is-original) .file-path"
            ).all_inner_texts()
            total = " ".join(panel.locator(".files-total").inner_text().split())
            if not total.startswith(f"{len(paths) + 1} files"):
                raise AssertionError(
                    f"{total!r} for {len(paths)} files and the original"
                )
            run.shot(page, "peek-files")
            with (
                page.expect_download(timeout=60000) as info,
                page.expect_request(
                    lambda r: r.url.endswith("/api/download-tickets")
                ) as ticket,
            ):
                panel.get_by_role("button", name="Download all (.zip)").click()
            asked = json.loads(ticket.value.post_data or "{}").get("path", "")
            if not re.fullmatch(r"/api/jobs/\w+/items/i\d+/archive", asked):
                raise AssertionError(f"Download all asked for {asked!r}")
            download = info.value
            names = zipfile.ZipFile(
                io.BytesIO(Path(download.path()).read_bytes())
            ).namelist()
            if "original/quarterly-report.pdf" not in names or not set(paths) <= set(
                names
            ):
                raise AssertionError(f"the archive holds {names}, the list {paths}")
            return (
                f"Files: the original, {len(paths)} result files, {total!r}; "
                f"Download all: {download.suggested_filename} ({len(names)} entries, "
                "original/ included) via the item's archive and a ticket"
            )

        run.step(page, "peek-files", peek_files)

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

        # 6c. A web page's original is the page the server fetched, framed:
        # its inline style drawn and the site's HTTPS picture and style sheet
        # loaded, its script never run, nothing loaded over plain HTTP
        # (contract v1.12 section 7). v1.13: the frame shares the workbench's
        # origin, so hovering a block on either side outlines both, and the
        # Markdown's own https picture shows beside it
        def compare_page() -> str:
            row = _items(page).filter(has_text=re.compile("alpha", re.IGNORECASE)).first
            # The window reads the original's descriptor as it opens (it
            # decides whether Compare is offered), before Compare is shown
            before = len(run.api)
            asked = len(PAGE_REQUESTS)
            assets_asked = len(ASSET_REQUESTS)
            row.click()
            _peek(page).wait_for()
            _show_tab(page, "Compare")
            _wait(
                page,
                lambda: run.saw("GET", r"/items/i\d+/original$", 200, since=before),
                what="the URL item's original",
            )
            frame_element = page.locator("#peek-panel iframe")
            frame_element.wait_for(timeout=15000)
            sandbox = frame_element.get_attribute("sandbox")
            if (
                sandbox
                != "allow-same-origin allow-popups allow-popups-to-escape-sandbox"
            ):
                raise AssertionError(f"the frame's sandbox is {sandbox!r}")
            if frame_element.get_attribute("referrerpolicy") != "no-referrer":
                raise AssertionError("the frame sends a referrer")
            source = frame_element.get_attribute("src") or ""
            if not re.search(r"/items/i\d+/original/view\?ticket=", source):
                raise AssertionError(f"the frame loads {source!r}")
            frame = page.frame_locator("#peek-panel iframe")
            heading = frame.locator("h1")
            heading.wait_for(timeout=15000)
            title = heading.inner_text().strip()
            if title != "Page alpha":
                raise AssertionError(f"the framed page shows {title!r}")
            colour = heading.evaluate("h => getComputedStyle(h).color")
            if colour != "rgb(200, 0, 0)":
                raise AssertionError(f"the page's own style not drawn: {colour}")
            ran = frame.locator("body").get_attribute("data-ran")
            page.wait_for_timeout(1000)
            picture = frame.locator('img[alt="Remote picture"]').evaluate(
                "img => img.naturalWidth"
            )
            site = "no HTTPS site files (no cryptography)"
            if SITE_ASSETS:
                site_picture = frame.locator('img[alt="Site picture"]').evaluate(
                    "img => img.naturalWidth"
                )
                rule = frame.locator("article").evaluate(
                    "a => getComputedStyle(a).borderTopColor"
                )
                loaded = sorted(set(ASSET_REQUESTS[assets_asked:]))
                if site_picture != 4 or rule != "rgb(0, 120, 0)":
                    raise AssertionError(
                        f"the site's HTTPS files did not load: picture width "
                        f"{site_picture}, style sheet colour {rule}, asked {loaded}"
                    )
                site = f"the site's HTTPS picture and style sheet loaded ({loaded})"
            fetched = [
                path
                for path in PAGE_REQUESTS[asked:]
                if path in ("/script-ran", "/remote.png", "/site.css")
            ]
            if ran is not None or fetched or picture:
                raise AssertionError(
                    f"the frame ran a script ({ran!r}) or loaded {fetched} "
                    f"(picture width {picture})"
                )
            linked = _hover_links(page, frame)
            remote = "no HTTPS site (no cryptography)"
            if SITE_ASSETS:
                remote = _web_picture(page.locator("#peek-panel"), "Site picture")
            link = (
                page.locator(".peek-head, #peek-panel")
                .get_by_role("link", name=re.compile("Open the page"))
                .or_(page.get_by_role("menuitem", name=re.compile("Open the page")))
            )
            run.shot(page, "peek-compare-page")
            has_link = link.count() > 0
            if SITE_ASSETS:
                _show_tab(page, "Preview")
                _web_picture(page.locator("#peek-panel"), "Site picture")
                run.shot(page, "peek-preview-web-picture")
                remote += "; Preview draws it too"
            note = (
                f"Compare frames the fetched page (sandbox {sandbox!r}, ticket "
                f"URL): '{title}' in its own red, {site}, no script ran, nothing "
                f"loaded over plain HTTP; {linked}; Markdown side: {remote}"
            )
            if not phone and not has_link:
                raise AssertionError("no Open the page link")
            _close_window(page)
            return note

        run.step(page, "peek-compare-page", compare_page)

        # 6d. The window grows to the whole viewport and back (F or its
        # button), remembered per browser; a stored "docked" (the layout
        # earlier versions had) opens as the window. A phone is always full
        def full_screen() -> str:
            if phone:
                _item(page, "quarterly-report.pdf").click()
                _heading(page, "quarterly-report.pdf").wait_for()
                if page.get_by_role("button", name="Full screen").count():
                    raise AssertionError("a phone offers Full screen")
                where = _window_placement(page, phone)
                _click_visible(page.get_by_role("button", name="Close"))
                _peek(page).wait_for(state="detached", timeout=10000)
                return f"no Full screen on a phone; the window is {where}"
            page.evaluate("() => localStorage.setItem('markitai.peek', 'docked')")
            _open(page, page.reload)
            _items(page).first.wait_for()
            _item(page, "quarterly-report.pdf").click()
            page.get_by_role("dialog", name=_spaced("quarterly-report.pdf")).wait_for()
            stored = page.evaluate("() => localStorage.getItem('markitai.peek')")
            if stored != "window" or page.locator(".peek.is-full").count():
                raise AssertionError(f"a stored 'docked' opened as {stored!r}")
            page.get_by_role("button", name="Full screen").click()
            full = page.locator(".peek.is-full")
            full.wait_for(timeout=10000)
            page.wait_for_timeout(400)
            box = full.bounding_box() or {}
            viewport = page.viewport_size or {"width": width, "height": 900}
            if (
                box.get("width", 0) < viewport["width"] - 2
                or box.get("height", 0) < viewport["height"] - 2
            ):
                raise AssertionError(f"full screen is {box} in {viewport}")
            if page.evaluate("() => localStorage.getItem('markitai.peek')") != "full":
                raise AssertionError("full screen not remembered")
            run.shot(page, "peek-full-screen")
            _open(page, page.reload)
            _items(page).first.wait_for()
            if not full.is_visible():
                _item(page, "quarterly-report.pdf").click()
            full.wait_for(timeout=10000)
            page.wait_for_timeout(300)  # the key handler registers a frame later
            page.locator(".peek").focus()
            page.keyboard.press("f")
            _wait(
                page,
                lambda: full.count() == 0,
                timeout=10,
                what="F to leave full screen",
            )
            stored_after = page.evaluate("() => localStorage.getItem('markitai.peek')")
            if stored_after != "window":
                raise AssertionError(f"markitai.peek = {stored_after!r}")
            _close_window(page)
            return (
                f"a stored 'docked' opens the window; Full screen fills "
                f"{int(box['width'])}x{int(box['height'])}, kept over a reload; F "
                "returns to the window"
            )

        run.step(page, "peek-full-screen", full_screen)

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
            # A failed action on one conversion is one error toast that stays
            # until closed, naming the action and the server's reason; trying
            # again replaces it instead of stacking a second one
            line = _strip(page, "error", "Retry failed")
            where = _at_the_bottom(page, line)
            text = " ".join(line.inner_text().split())
            if "no longer part of the job" not in text:
                raise AssertionError(f"the toast does not say why: {text!r}")
            run.shot(page, "inspector-error-line")
            _no_banners(page)
            if not _heading(page, "damaged.pdf").is_visible():
                raise AssertionError("the window left its conversion")
            again = len(run.api)
            _click_visible(
                page.get_by_role("button", name=re.compile(r"^(Retry|Run again)$"))
            )
            _wait(
                page,
                lambda: run.saw("POST", r"/items/i\d+/retry$", 404, since=again),
                what="the second refused retry",
            )
            page.wait_for_timeout(500)
            shown = (
                page.locator(".toasts .toast").filter(has_text="Retry failed").count()
            )
            if shown != 1:
                raise AssertionError(
                    f"{shown} 'Retry failed' toasts after trying again"
                )
            line.get_by_role("button", name=DISMISS).click()
            line.wait_for(state="detached", timeout=5000)
            # The list still shows the deleted row until it is read again
            _open(page, page.reload)
            _items(page).first.wait_for()
            return f"one error toast ({where}) says {text!r}; trying again replaced it"

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
            _show_tab(page, "Details")  # the window opens on Preview
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

        # 11b. A Mermaid block is drawn as a diagram by the renderer the app
        # loads only for it (/ui/mermaid.js); one that does not parse stays
        # code with a note, and the Markdown itself keeps both blocks
        def mermaid() -> str:
            chunks: list[tuple[int, str]] = []

            def chunk(response: Any) -> None:
                if response.url.endswith("/ui/mermaid.js"):
                    chunks.append(
                        (response.status, response.header_value("content-type") or "")
                    )

            page.on("response", chunk)
            try:
                with page.expect_response(
                    lambda response: (
                        re.search(r"/items/i\d+/result", response.url) is not None
                    ),
                    timeout=30000,
                ) as result:
                    _open(page, lambda: page.goto(f"{base}/jobs"))
                    _item(page, "meeting-notes.md").click()
                markdown = result.value.text()
                _heading(page, "meeting-notes.md").wait_for()
                _show_tab(page, "Preview")
                body = page.locator(
                    "#peek-panel [role=tabpanel]:not([hidden]) .doc-body"
                ).first
                svg = body.locator("figure.md-diagram > svg").first
                svg.wait_for(timeout=20000)
                labels = svg.text_content() or ""
                for label in ("Draft notes", "Review", "Ship 1.3.0"):
                    if label not in labels:
                        raise AssertionError(f"the diagram lacks {label!r}: {labels!r}")
                for tag in ("script", "foreignObject", "style", "image"):
                    if svg.locator(tag).count():
                        raise AssertionError(f"the diagram kept a <{tag}>")
                if body.locator("figure.md-diagram").count() != 1:
                    raise AssertionError("expected exactly one drawn diagram")
                note = body.locator("p.diagram-note").first
                note.wait_for(timeout=10000)
                note_text = note.inner_text()
                if not note_text.startswith("Diagram not drawn"):
                    raise AssertionError(f"unexpected note {note_text!r}")
                kept = body.locator("pre.has-diagram-note").first
                if "draft --> [unclosed" not in (kept.text_content() or ""):
                    raise AssertionError("the broken block is not shown as its code")
                if markdown.count("```mermaid") != 2 or "flowchart LR" not in markdown:
                    raise AssertionError("the converted Markdown lost a mermaid block")
                # The diagram above is drawn, so the chunk ran; Chromium
                # reports a copy from its memory cache as a 200 without
                # headers (the server's own answer is checked by the tests)
                if not any(
                    status == 304
                    or (status == 200 and ("javascript" in kind or not kind))
                    for status, kind in chunks
                ):
                    raise AssertionError(f"/ui/mermaid.js answers {chunks}")
            finally:
                page.remove_listener("response", chunk)
            # The open window, the diagram in view (the step's own shot comes
            # after the window closed)
            svg.evaluate("node => node.scrollIntoView({block: 'center'})")
            page.wait_for_timeout(300)
            page.screenshot(path=str(out / f"preview-mermaid-open-{width}.png"))
            _close_window(page)
            return (
                f"Mermaid block drawn as an SVG diagram in Rendered (/ui/mermaid.js "
                f"{chunks[0][0]}); the broken one kept as code with {note_text!r}; "
                "the Markdown keeps both blocks"
            )

        run.step(page, "preview-mermaid", mermaid)

        # 11c. Contract v1.20 (options A): Output's JSON switch, the local OCR
        # language while OCR runs without LLM, Advanced → Pages with OCR
        # Every Page only (Python always writes page markers, so it announces
        # no page_markers); a conversion with them sends format, ocr_lang and
        # ocr_all_pages, publishes <name>.json among its files, and Download's
        # JSON entry saves the item's document
        def options_a() -> str:
            _open(page, lambda: page.goto(f"{base}/"))
            _click_visible(page.get_by_role("button", name="Options"))
            json_switch = page.get_by_role("switch", name="JSON", exact=True)
            json_switch.wait_for(timeout=10000)
            if json_switch.get_attribute("aria-checked") != "false":
                raise AssertionError("JSON is on before it was asked for")
            ocr = page.get_by_role("switch", name="OCR", exact=True)
            if ocr.get_attribute("aria-checked") != "true":
                ocr.click()
            picker = page.locator(".opt-pick").first
            picker.wait_for(timeout=10000)
            shown = " ".join(picker.inner_text().split())
            if "Automatic" not in shown:
                raise AssertionError(f"the OCR language reads {shown!r}")
            picker.click()
            languages = page.locator(".opt-pick-menu [role=menuitemradio]")
            languages.first.wait_for(timeout=10000)
            if languages.count() != 10:
                raise AssertionError(f"{languages.count()} OCR languages offered")
            languages.nth(4).click()  # ko: en, zh, zh_tw, ja, ko, ...
            picked = " ".join(picker.inner_text().split())
            advanced = page.get_by_role("button", name="Advanced")
            if advanced.get_attribute("aria-expanded") != "true":
                advanced.click()
            pages_row = page.get_by_role("group", name="Pages", exact=True)
            pages_row.wait_for(timeout=10000)
            if pages_row.get_by_role("switch", name="Page Markers").count():
                raise AssertionError("Python offers Page Markers")
            every_page = pages_row.get_by_role("switch", name="OCR Every Page")
            every_page.click()
            json_switch.click()
            run.shot(page, "options-a")

            def reset() -> None:
                # The composer keeps its choices: later steps start from none
                _open(page, lambda: page.goto(f"{base}/"))
                _click_visible(page.get_by_role("button", name="Options"))
                for name in ("OCR Every Page", "OCR", "JSON"):
                    switch = page.get_by_role("switch", name=name, exact=True)
                    if (
                        switch.count()
                        and switch.get_attribute("aria-checked") == "true"
                    ):
                        switch.click()
                page.keyboard.press("Escape")

            page.keyboard.press("Escape")
            if phone:
                reset()
                return (
                    f"JSON switch, OCR language {shown!r} → {picked!r} (10 offered), "
                    "Pages with OCR Every Page and no Page Markers"
                )
            name = "options-a-notes.txt"
            note = Path(inputs["md"]).parent / name
            note.write_text(
                "# Options A\n\nA table follows.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
            )
            page.locator(
                "input[type=file]:not([webkitdirectory])"
            ).first.set_input_files([str(note)])
            with page.expect_response(
                lambda r: r.url.endswith("/api/jobs") and r.request.method == "POST",
                timeout=30000,
            ) as created:
                page.get_by_role("button", name="Convert", exact=True).click()
            job = created.value.json()["job_id"]
            # The job keeps the options it was sent (a body carrying a file
            # is not readable from the page's request)
            saved = (
                _api(page, base, "GET", f"/api/jobs/{job}").json().get("options") or {}
            )
            wanted = {"format": "json", "ocr_lang": "ko", "ocr_all_pages": True}
            if {key: saved.get(key) for key in wanted} != wanted:
                raise AssertionError(f"the job's options are {saved}")
            _wait(
                page,
                lambda: all(
                    item.get("finished_at")
                    for item in _api(page, base, "GET", f"/api/jobs/{job}").json()[
                        "items"
                    ]
                ),
                what="the JSON conversion",
            )
            item_id = _api(page, base, "GET", f"/api/jobs/{job}").json()["items"][0][
                "item_id"
            ]
            artifacts = [
                artifact["relpath"]
                for artifact in _api(
                    page, base, "GET", f"/api/jobs/{job}/items/{item_id}/result"
                ).json()["artifacts"]
            ]
            if artifacts[:2] != [f"{name}.md", f"{name}.json"]:
                raise AssertionError(f"the item's files are {artifacts}")
            _dismiss(page)
            _open(page, lambda: page.goto(f"{base}/jobs"))
            _item(page, name).click()
            _heading(page, name).wait_for()
            page.locator(".peek-head").get_by_role(
                "button", name="Download as", exact=True
            ).click()
            with page.expect_download(timeout=60000) as info:
                page.locator(".peek-head .download-menu").get_by_role(
                    "menuitem", name=re.compile(r"^JSON")
                ).click()
            document = json.loads(Path(info.value.path()).read_text(encoding="utf-8"))
            if (
                document.get("schema_version") != "1.0"
                or document["source"]["path"] != name
                or document["tables"][0]["rows"] != [["1", "2"]]
            ):
                raise AssertionError(f"the downloaded document is {document}")
            _close_window(page)
            reset()
            return (
                f"JSON switch, OCR language {shown!r} → {picked!r} (10 offered), "
                "Pages with OCR Every Page and no Page Markers; the job kept "
                f"format/ocr_lang/ocr_all_pages, its files {artifacts[:2]}, "
                f"Download → JSON saved {info.value.suggested_filename} "
                f"(schema {document['schema_version']}, its table read)"
            )

        run.step(page, "options-a", options_a)

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

        # 12b. Why a model test failed, one model per reason (contract v1.10):
        # the server's answer and the workbench's translated block. A model
        # whose key the server lacks (no MISTRAL_API_KEY here) sends nothing;
        # the others are refused by the fake LLM or
        # aimed at a port nothing listens on. A timeout (15 s) is left to the
        # unit tests.
        def settings_test_reasons() -> str:
            cases = [
                # model, how it is saved, reason, title, provider's words
                (
                    "mistral/mistral-small-latest",
                    {"provider": "mistral"},
                    "no_key",
                    "No API key",
                    "MISTRAL_API_KEY is not set on the server, so nothing was sent",
                ),
                *(
                    (
                        f"openai/{name}",
                        {
                            "api_key": "sk-e2e-fake-0000",
                            "api_base": f"http://127.0.0.1:{llm_port}/v1",
                        },
                        reason,
                        title,
                        "Model connection returned HTTP {}: {}".format(*REFUSALS[name]),
                    )
                    for name, reason, title in (
                        ("fake-401", "key_rejected", "Key rejected"),
                        ("fake-404", "model_not_found", "Model not found"),
                        ("fake-429", "rate_limited", "Rate limited or out of quota"),
                        ("fake-500", "provider_error", "Provider error"),
                    )
                ),
                (
                    "openai/fake-unreachable",
                    {
                        "api_key": "sk-e2e-dead-0000",
                        "api_base": "http://127.0.0.1:9/v1",
                    },
                    "network",
                    "Provider unreachable",
                    "Model connection request failed",
                ),
            ]
            revision = _api(page, base, "GET", "/api/settings/llm").json()["revision"]
            saved = _api(
                page,
                base,
                "POST",
                "/api/settings/llm/deployments/batch",
                data={
                    "expected_revision": revision,
                    "deployments": [
                        {"model_name": "e2e-reasons", "model": model, **how}
                        for model, how, *_ in cases
                    ],
                },
            )
            if saved.status != 200:
                raise AssertionError(
                    f"saving the models: {saved.status} {saved.text()}"
                )
            added = [
                d["deployment_id"]
                for d in saved.json()["deployments"]
                if d["routing_group"] == "e2e-reasons"
            ]
            notes = []
            try:
                _open(page, lambda: page.goto(f"{base}/settings/models"))
                for model, _how, reason, title, words in cases:
                    before = len(run.api)
                    with page.expect_response(
                        lambda r: r.url.endswith("/api/settings/llm/test")
                    ) as answered:
                        _click_visible(
                            page.get_by_role("button", name=f"Test {model}", exact=True)
                        )
                    answer = answered.value.json()
                    if answer.get("reason") != reason or "provider_detail" in answer:
                        raise AssertionError(f"{model}: the server answered {answer}")
                    if not answer["detail"].startswith(words):
                        raise AssertionError(f"{model}: detail {answer['detail']!r}")
                    # One error toast per test (a new test replaces it): the
                    # translated reason and the model as its title, what to do
                    # as its text, the server's detail folded under Details
                    block = _strip(page, "error", f"Test of {model}")
                    # the bold title ends with the language's lead (". ")
                    heading = " ".join(
                        (
                            block.locator(".toast-text strong").text_content() or ""
                        ).split()
                    ).removesuffix(".")
                    if heading != f"{title} · Test of {model}":
                        raise AssertionError(
                            f"{model}: the toast is titled {heading!r}"
                        )
                    shown = block.locator(".toast-detail p").text_content() or ""
                    if shown != answer["detail"]:
                        raise AssertionError(
                            f"{model}: the provider's message shows {shown!r}"
                        )
                    if (
                        page.locator(".toasts .toast")
                        .filter(has_text="Test of")
                        .count()
                        != 1
                    ):
                        raise AssertionError(f"{model}: an earlier test's toast stayed")
                    if reason == "no_key":
                        if answer.get("key_variable") != "MISTRAL_API_KEY":
                            raise AssertionError(f"{model}: {answer}")
                        block.get_by_text(
                            "MISTRAL_API_KEY is not set on the server. Set it where "
                            "markitai serve runs"
                        ).wait_for()
                        row = page.locator(".model-test-fail").filter(
                            has_text="MISTRAL_API_KEY"
                        )
                        row.wait_for()
                        tests = [
                            call
                            for call in run.api[before:]
                            if call[1].startswith("/api/settings/llm/test")
                        ]
                        if len(tests) != 1:
                            raise AssertionError(f"model tests sent: {tests}")
                        run.shot(page, "settings-test-no-key")
                    elif "key_variable" in answer:
                        raise AssertionError(f"{model}: {answer}")
                    notes.append(f"{reason}: {title!r}")
            finally:
                revision = _api(page, base, "GET", "/api/settings/llm").json()[
                    "revision"
                ]
                for deployment_id in added:
                    gone = _api(
                        page,
                        base,
                        "DELETE",
                        f"/api/settings/llm/deployments/{deployment_id}"
                        f"?expected_revision={revision}",
                    )
                    revision = gone.json().get("revision", revision)
            return (
                "each reason shown translated with the server's detail under "
                "Provider's message; " + ", ".join(notes)
            )

        run.step(page, "settings-test-reasons", settings_test_reasons)

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
            shown = [" ".join(n.split()) for n in tiles.all_inner_texts()]
            # A provider with a saved row is marked Added (contract v1.11)
            names = [name.removesuffix(" Added") for name in shown]
            added = [
                name.removesuffix(" Added") for name in shown if name.endswith(" Added")
            ]
            if added != ["OpenAI", "DeepSeek"]:
                raise AssertionError(f"Common marks {added} as added")
            # One provider, one saved row: DeepSeek's environment card folded
            # into the hand-written entry at its own endpoint
            deepseek_rows = [
                " ".join(text.split())
                for text in panel.locator(".provider-row").all_inner_texts()
                if "DeepSeek" in text
            ]
            if (
                len(deepseek_rows) != 1
                or f"Using {DEEPSEEK_KEY[0]}" not in deepseek_rows[0]
            ):
                raise AssertionError(f"DeepSeek's saved rows: {deepseek_rows}")
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
                f"{total}; Common: {len(names)} providers with marks, "
                f"{' and '.join(added)} marked Added; one DeepSeek row "
                f"({deepseek_rows[0]}); filters {' · '.join(chips)}; search by "
                "prefix and alias"
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
            # The key field says where else its key can come from (v1.11);
            # the default address is the optional address field's placeholder
            hint = page.get_by_text(
                "Or set GROQ_API_KEY where markitai serve runs and restart it; "
                "the provider then appears under Saved, with no key to paste."
            )
            default_base = page.get_by_placeholder("https://api.groq.com/openai/v1")
            _wait(
                page,
                lambda: hint.count() == 1 and default_base.count() == 1,
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

        # 13c'. A model written into the config without a key, served by the
        # server's own OPENAI_API_KEY (a hand-written deepseek/deepseek-flash
        # with DEEPSEEK_API_KEY, 2026-10-12): its connection shows the
        # variable, not "Not set", and its model list loads with that key
        def environment_key() -> str:
            variable, secret = ENV_KEY
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            before, served = len(run.api), len(MODEL_LIST_KEYS)
            with page.expect_response(
                lambda r: r.url.endswith("/api/settings/llm/model-discovery")
            ) as listed:
                _click_visible(
                    page.get_by_role("button", name=f"Edit {HAND_MODEL}", exact=True)
                )
            answer = listed.value.json()
            if answer.get("status") != "ok" or "openai/fake-model" not in [
                m["model"] for m in answer.get("models", [])
            ]:
                raise AssertionError(f"the model list: {answer}")
            if not run.saw(
                "GET",
                r"^/api/settings/llm/providers/[^/]+/credentials$",
                200,
                since=before,
            ):
                raise AssertionError("the connection was not loaded")
            if MODEL_LIST_KEYS[served:] != [f"Bearer {secret}"]:
                raise AssertionError(
                    f"model list requests carried {MODEL_LIST_KEYS[served:]}, not "
                    f"the environment's (the page sent {listed.value.request.post_data})"
                )
            key_row = page.locator(".conn-row").filter(has_text="API key")
            key_row.get_by_text(f"From environment variable {variable}").wait_for(
                timeout=10000
            )
            if _visible(page.get_by_text("Models could not be loaded")) is not None:
                raise AssertionError("the editor says the models could not be loaded")
            page.get_by_text("fake-model").first.wait_for(timeout=10000)
            run.shot(page, "settings-environment-key")
            if secret in page.content():
                raise AssertionError("the environment's key reached the page")
            return (
                f"{HAND_MODEL} (no key in the config): API key 'From environment "
                f"variable {variable}', its list loaded with that key"
            )

        run.step(page, "settings-environment-key", environment_key)

        # 13c''. The case itself: a DeepSeek entry written by hand at
        # DeepSeek's endpoint, served by DEEPSEEK_API_KEY. Edit names the
        # variable; its list, sent with that key, cannot reach DeepSeek here
        # and the page says why (contract v1.11 reasons)
        def deepseek_key() -> str:
            variable, secret = DEEPSEEK_KEY
            _open(page, lambda: page.goto(f"{base}/settings/models"))
            with page.expect_response(
                lambda r: r.url.endswith("/api/settings/llm/model-discovery")
            ) as listed:
                _click_visible(
                    page.get_by_role(
                        "button", name=f"Edit {DEEPSEEK_MODEL}", exact=True
                    )
                )
            answer = listed.value.json()
            if (answer.get("status"), answer.get("reason")) != (
                "unavailable",
                "network",
            ):
                raise AssertionError(f"the model list: {answer}")
            if not answer.get("detail", "").startswith(
                "Model discovery request failed"
            ):
                raise AssertionError(f"the detail: {answer.get('detail')!r}")
            key_row = page.locator(".conn-row").filter(has_text="API key")
            key_row.get_by_text(f"From environment variable {variable}").wait_for(
                timeout=10000
            )
            page.get_by_text(
                "The server could not reach the provider", exact=False
            ).first.wait_for(timeout=10000)
            run.shot(page, "settings-deepseek-key")
            if secret in page.content():
                raise AssertionError("the environment's key reached the page")
            return (
                f"{DEEPSEEK_MODEL} (hand-written, no key): 'From environment variable "
                f"{variable}'; its list unreachable here: reason network, "
                f"{answer['detail']!r}"
            )

        run.step(page, "settings-deepseek-key", deepseek_key)

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

        # 15b. Several conversions deleted at once (contract v1.13 §6.8): two
        # rows and a submission's header checked, Delete, the confirmation;
        # one POST /api/history/delete, no single DELETE
        def batch_delete() -> str:
            pair = _submit_files(
                base,
                TOKEN,
                [
                    ("delete-me-1.md", b"# Delete me 1\n\nFirst throwaway.\n"),
                    ("delete-me-2.md", b"# Delete me 2\n\nSecond throwaway.\n"),
                ],
            )
            single = _submit_files(
                base, TOKEN, [("delete-me-3.md", b"# Delete me 3\n\nThird.\n")]
            )
            _open(page, lambda: page.goto(f"{base}/jobs"))
            third = page.locator(".conv-list .row", has_text="delete-me-3")
            third.first.wait_for(timeout=15000)
            head = (
                page.locator(".conv-list .group-head")
                .filter(
                    has=page.locator(
                        "xpath=following-sibling::li[1][contains(., 'delete-me-1')]"
                    )
                )
                .first
            )
            calls: list[str] = []

            def deleting(request: Any) -> None:
                path = request.url.replace(base, "")
                if request.method == "DELETE" or path == "/api/history/delete":
                    calls.append(f"{request.method} {path}")

            page.on("request", deleting)
            try:
                if phone:
                    # Select mode: a tap (here a click) checks a row
                    _click_visible(page.locator(".select-mode-btn"))
                    third.first.click()
                    head.click()
                else:
                    third.first.focus()
                    page.keyboard.press(" ")
                    head.locator(".row-check").click()
                count = page.locator(".select-count").inner_text().strip()
                checked = page.locator(".conv-list [aria-checked=true]").count()
                if count != "3 selected" or checked != 4:
                    raise AssertionError(f"{count!r}, {checked} rows aria-checked")
                run.shot(page, "batch-delete-selected")
                page.locator(".select-bar .select-delete").click()
                dialog = page.get_by_role("alertdialog").filter(
                    has_text="Delete 3 conversions?"
                )
                dialog.wait_for(timeout=10000)
                text = dialog.inner_text()
                if (
                    "Their Markdown, pictures and kept originals are deleted too. "
                    "This cannot be undone." not in text
                ):
                    raise AssertionError(f"the confirmation says {text!r}")
                run.shot(page, "batch-delete-confirm")
                with page.expect_response(
                    lambda r: r.url.endswith("/api/history/delete")
                ) as answered:
                    dialog.get_by_role("button", name="Delete", exact=True).click()
                body = answered.value.json()
                page.locator(".toasts .toast").filter(
                    has_text="3 conversions deleted"
                ).first.wait_for(timeout=15000)
                page.wait_for_timeout(500)
            finally:
                page.remove_listener("request", deleting)
            if calls != ["POST /api/history/delete"]:
                raise AssertionError(f"the page sent {calls}")
            sent = json.loads(answered.value.request.post_data or "{}")
            # A submission whose every item is checked goes as a job
            named = set(sent.get("jobs", [])) | {
                entry["job_id"] for entry in sent.get("items", [])
            }
            if pair not in sent.get("jobs", []) or named != {pair, single}:
                raise AssertionError(f"the request named {sent}")
            if (body["deleted"], body["failed"]) != (2, 0):
                raise AssertionError(f"the answer was {body}")
            left = _api(page, base, "GET", "/api/history?include=items").json()
            names = [
                item["name"]
                for entry in left
                for item in entry.get("items", [])
                if item["name"].startswith("delete-me")
            ]
            if names or page.locator(".conv-list .row", has_text="delete-me").count():
                raise AssertionError(f"still listed: {names}")
            return (
                f"'3 selected' (4 rows aria-checked: 3 + the submission header), the "
                f"confirmation, one request {calls[0]} with {json.dumps(sent)}, "
                "'3 conversions deleted', none listed after"
            )

        run.step(page, "batch-delete", batch_delete)

        # 15c. A user token, made by the admin on Settings → Access token: it
        # converts and sees only its own conversions, and the settings pages
        # say the admin's token is needed (contract v1.13 §2)
        user: dict[str, str] = {}

        def user_token() -> str:
            _open(page, lambda: page.goto(f"{base}/settings/access"))
            field = page.get_by_label("Name", exact=True)
            field.wait_for(timeout=15000)
            field.fill(USER_LABEL)
            with page.expect_response(
                lambda r: (
                    r.url.endswith("/api/access-tokens") and r.request.method == "POST"
                )
            ) as answered:
                page.get_by_role("button", name="Create token").click()
            created = answered.value.json()
            if answered.value.status != 201 or not re.fullmatch(
                r"mku_[0-9a-f]{64}", created.get("token", "")
            ):
                raise AssertionError(f"create answered {answered.value.status}")
            user.update(token=created["token"], id=created["id"])
            page.get_by_text(created["token"], exact=True).first.wait_for(timeout=10000)
            page.get_by_text(f"Token for “{USER_LABEL}” created").first.wait_for()
            # The list waits for Done while the new token is shown; after it
            # the token is gone from the page (shown once)
            page.get_by_role("button", name="Done", exact=True).click()
            page.get_by_text(created["token"], exact=True).first.wait_for(
                state="detached", timeout=10000
            )
            listed = page.locator(".token-list li").filter(has_text=USER_LABEL)
            listed.first.wait_for(timeout=10000)
            hint = created["token"][-4:]
            if hint not in listed.first.inner_text():
                raise AssertionError(
                    "the list does not show the token's last characters"
                )
            return (
                f"user token '{USER_LABEL}' created: shown once with its link, "
                f"listed as …{hint}"
            )

        run.step(page, "user-token", user_token)

        def user_session() -> str:
            if not user:
                raise AssertionError("no user token")
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
                own = _submit_files(
                    base,
                    user["token"],
                    [("field-notes.md", b"# Field notes\n\nThe user's own file.\n")],
                )
                _open(other, lambda: other.goto(f"{base}/jobs#token={user['token']}"))
                other.locator(".conv-list .row", has_text="field-notes").first.wait_for(
                    timeout=15000
                )
                other.wait_for_timeout(800)
                rows = other.locator(".conv-list .row").count()
                if rows != 1:
                    raise AssertionError(f"the user's list shows {rows} rows")
                run.shot(other, "user-conversions")
                admins = _api(page, base, "GET", "/api/history").json()
                hidden = other.request.fetch(
                    f"{base}/api/jobs/{admins[-1]['job_id']}",
                    headers={"Authorization": f"Bearer {user['token']}"},
                )
                if hidden.status != 404:
                    raise AssertionError(f"another's job answered {hidden.status}")
                # Its own Models (contract v1.15) and Edit config (v1.16):
                # the user's models and the admin's shared ones, never the
                # server's configuration file; Access token stays the admin's
                # The admin shares none of its models here: the user's page
                # is its own empty one, without the admin's private models
                _open(other, lambda: other.goto(f"{base}/settings/models"))
                other.get_by_text(
                    "Add a model with your own API key to use LLM features"
                ).first.wait_for(timeout=15000)
                leaked = [
                    name
                    for name in ("openai/fake-model", "e2e-hand", "e2e-deepseek")
                    if other.get_by_text(name).count()
                ]
                if leaked or other.get_by_text("config.json").count():
                    raise AssertionError(
                        f"the user's Models page shows the admin's {leaked or 'file'}"
                    )
                run.shot(other, "user-settings")
                _open(other, lambda: other.goto(f"{base}/settings/config"))
                other.get_by_text(
                    "Your models, their connections and your weights for shared models."
                ).first.wait_for(timeout=15000)
                area = other.locator("#config-text")
                area.wait_for(timeout=15000)
                if "url_concurrency" in area.input_value():
                    raise AssertionError(
                        "the user's Edit config shows the server's file"
                    )
                if (
                    other.get_by_role("link", name="Access token").count()
                    or other.get_by_role("button", name="Access token").count()
                ):
                    raise AssertionError("the user's tab offers Access token")
                _open(other, lambda: other.goto(f"{base}/settings/access"))
                other.get_by_text("Admin token needed").first.wait_for(timeout=15000)
                _open(other, lambda: other.goto(f"{base}/settings/appearance"))
                other.get_by_text(
                    f"This tab uses the user token “{USER_LABEL}”"
                ).first.wait_for(timeout=15000)
                run.shot(other, "user-appearance")
                # Owner decision 3: a user token's conversions never use a
                # remote service, so its composer offers only the local
                # strategies and no files backend (the server refuses the
                # rest with 403 remote_processing_forbidden); the admin's
                # composer still offers all six
                offered, files = _offered_strategies(other, base)
                if offered != ["Auto", "Static", "Playwright"] or files:
                    raise AssertionError(
                        f"the user's composer offers {offered}, files backend: {files}"
                    )
                run.shot(other, "user-strategies")
                admin_offered, admin_files = _offered_strategies(page, base)
                if (
                    admin_offered
                    != [
                        "Auto",
                        "Static",
                        "Playwright",
                        "Defuddle",
                        "Jina",
                        "Cloudflare",
                    ]
                    or not admin_files
                ):
                    raise AssertionError(
                        f"the admin's composer offers {admin_offered}, files: {admin_files}"
                    )
            finally:
                fresh.close()
            return (
                f"the user sees 1 row (its own job {own}), another's job is 404, "
                "Models and Edit config are its own (no server file), Access token "
                "says 'Admin token needed', Appearance names the token; its composer "
                "offers only Auto/Static/Playwright and no files backend (the "
                "admin's all six)"
            )

        run.step(page, "user-session", user_session)

        # 15d. Settings → Edit config: the file's own text with its secrets
        # masked, JSON checked as typed, Review changes as a line diff, Save
        # config writing exactly that change (secrets kept on disk), and a
        # file changed behind the page refused as stale (contract v1.13 §8)
        def edit_config() -> str:
            before = config.read_text(encoding="utf-8")
            _open(page, lambda: page.goto(f"{base}/settings/config"))
            area = page.locator("#config-text")
            area.wait_for(timeout=15000)
            shown = area.input_value()
            leaked = [
                secret
                for secret in ("sk-e2e-fake-0000", "sk-e2e-conn-0000", *BASE_SECRETS)
                if secret in shown
            ]
            if leaked or "…" not in shown:
                raise AssertionError(f"secrets shown: {leaked}")
            masked = page.locator(".config-note").filter(has_text="shown masked")
            masked.first.wait_for(timeout=5000)
            if '"url_concurrency": 1' not in shown:
                raise AssertionError("the editor does not show the file")
            area.fill(shown.replace('"url_concurrency": 1', '"url_concurrency": 1,', 1))
            check = page.locator("#config-check")
            _wait(
                page,
                lambda: "Not valid JSON at line" in check.inner_text(),
                timeout=5,
                what="the JSON check",
            )
            if not page.locator(".config-review-btn").is_disabled():
                raise AssertionError("Review is offered for broken JSON")
            run.shot(page, "edit-config-invalid")
            edited = shown.replace('"url_concurrency": 1', '"url_concurrency": 2', 1)
            area.fill(edited)
            _wait(
                page,
                lambda: check.inner_text().strip() == "Valid JSON · unsaved changes",
                timeout=5,
                what="the JSON check to pass",
            )
            page.locator(".config-review-btn").click()
            diff = page.locator(".config-diff")
            diff.wait_for(timeout=5000)
            # The rows' text (the blocks are drawn lazily, so not innerText)
            changed = diff.evaluate(
                "d => [...d.querySelectorAll('.diff-line.is-del .diff-text, "
                ".diff-line.is-add .diff-text')].map((e) => e.textContent.trim())"
            )
            if changed != ['"url_concurrency": 1', '"url_concurrency": 2']:
                raise AssertionError(f"the diff changes {changed}")
            run.shot(page, "edit-config-diff")
            with page.expect_response(
                lambda r: (
                    r.url.endswith("/api/settings/config") and r.request.method == "PUT"
                )
            ) as answered:
                page.get_by_role("button", name="Save config").click()
            if answered.value.status != 200:
                raise AssertionError(f"save answered {answered.value.status}")
            # A save that needs a restart says so in a warning toast
            _strip(page, "warning", "restart the server for the other changes")
            after = config.read_text(encoding="utf-8")
            expected = before.replace('"url_concurrency": 1', '"url_concurrency": 2', 1)
            if after != expected:
                raise AssertionError(
                    "the saved file is not the original with that one change"
                )
            run.shot(page, "edit-config-saved")
            # Someone edits the file behind the page
            config.write_text(
                after.replace('"url_concurrency": 2', '"url_concurrency": 3', 1),
                encoding="utf-8",
            )
            again = area.input_value()
            area.fill(again.replace('"enabled": false', '"enabled": true', 1))
            page.locator(".config-review-btn").click()
            with page.expect_response(
                lambda r: (
                    r.url.endswith("/api/settings/config") and r.request.method == "PUT"
                )
            ) as refused:
                page.get_by_role("button", name="Save config").click()
            failure = page.locator(".config-failure")
            failure.wait_for(timeout=10000)
            if (
                refused.value.status != 409
                or "The file changed since you opened it" not in (failure.inner_text())
            ):
                raise AssertionError(
                    f"stale save answered {refused.value.status}: {failure.inner_text()!r}"
                )
            run.shot(page, "edit-config-stale")
            with page.expect_response(lambda r: r.url.endswith("/api/settings/config")):
                failure.get_by_role("button", name="Load current file").click()
            _wait(
                page,
                lambda: (
                    '"url_concurrency": 3' in page.locator("#config-text").input_value()
                ),
                timeout=10,
                what="the current file",
            )
            return (
                "secrets masked in the file's own text; broken JSON named by line and "
                "column with Review off; the diff showed 1 → 2; Save wrote exactly that "
                "change (keys kept on disk) and said a restart is needed; a file "
                "changed behind the page answered 409 and Load current file read it"
            )

        run.step(page, "edit-config", edit_config)

        # 15e. A config file written on Windows: CRLF breaks and no final line
        # break. The page edits with LF breaks and Save puts the file's own
        # back, so one changed value changes only its line (contract v1.13
        # clarification, 2026-10-14)
        def edit_config_crlf() -> str:
            crlf = config.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            crlf = crlf.rstrip(b"\r\n")
            if b'"url_concurrency": 3' not in crlf:
                raise AssertionError("the file is not as the previous step left it")
            config.write_bytes(crlf)
            with page.expect_response(
                lambda r: (
                    r.url.endswith("/api/settings/config") and r.request.method == "GET"
                )
            ) as read:
                _open(page, lambda: page.goto(f"{base}/settings/config"))
            sent = read.value.json()["text"]
            if "\r\n" not in sent or sent.endswith("\n"):
                raise AssertionError("GET did not send the file's own line breaks")
            area = page.locator("#config-text")
            area.wait_for(timeout=15000)
            page.locator(".config-note").filter(
                has_text="This file uses CRLF line endings; saving keeps them."
            ).first.wait_for(timeout=5000)
            shown = area.input_value()
            area.fill(shown.replace('"url_concurrency": 3', '"url_concurrency": 4', 1))
            check = page.locator("#config-check")
            _wait(
                page,
                lambda: check.inner_text().strip() == "Valid JSON · unsaved changes",
                timeout=5,
                what="the JSON check to pass",
            )
            page.locator(".config-review-btn").click()
            diff = page.locator(".config-diff")
            diff.wait_for(timeout=5000)
            changed = diff.evaluate(
                "d => [...d.querySelectorAll('.diff-line.is-del .diff-text, "
                ".diff-line.is-add .diff-text')].map((e) => e.textContent.trim())"
            )
            if changed != ['"url_concurrency": 3', '"url_concurrency": 4']:
                raise AssertionError(f"Review shows {changed}")
            run.shot(page, "edit-config-crlf-diff")
            with page.expect_response(
                lambda r: (
                    r.url.endswith("/api/settings/config") and r.request.method == "PUT"
                )
            ) as answered:
                page.get_by_role("button", name="Save config").click()
            if answered.value.status != 200:
                raise AssertionError(f"save answered {answered.value.status}")
            _strip(page, "warning", "restart the server for the other changes")
            after = config.read_bytes()
            expected = crlf.replace(b'"url_concurrency": 3', b'"url_concurrency": 4', 1)
            if after != expected:
                lone = after.replace(b"\r\n", b"").count(b"\n")
                raise AssertionError(
                    f"the saved file differs beyond that line ({lone} LF breaks, "
                    f"ends with {after[-4:]!r})"
                )
            run.shot(page, "edit-config-crlf-saved")
            breaks = crlf.count(b"\r\n")
            return (
                f"a CRLF file without a final break ({breaks} breaks): GET "
                "sent its breaks, the page said it keeps CRLF, Review showed only 3 → 4, "
                "and Save wrote the original bytes with that one line changed, still "
                "CRLF and still without a final break"
            )

        run.step(page, "edit-config-crlf", edit_config_crlf)

        # 16. Settings → Access token → Regenerate token: this tab keeps the
        # new token (shown once, with its link and QR code); the old token,
        # another tab holding it and its tickets stop working (contract v1.11)
        def regenerate() -> str:
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
                _open(other, lambda: other.goto(f"{base}/#token={TOKEN}"))
                other.get_by_role("button", name="Convert").wait_for()
                _open(page, lambda: page.goto(f"{base}/settings/access"))
                page.get_by_role("button", name="Sign out of this tab").wait_for()
                _click_visible(page.get_by_role("button", name="Regenerate token…"))
                with page.expect_response(
                    lambda r: r.url.endswith("/api/access-token/regenerate")
                ) as answered:
                    page.get_by_role("button", name="Regenerate", exact=True).click()
                body = answered.value.json()
                new = body["token"]
                if answered.value.status != 200 or new == TOKEN:
                    raise AssertionError(f"regenerate answered {answered.value.status}")
                page.get_by_text(new, exact=True).first.wait_for(timeout=10000)
                page.get_by_label("QR code of the sign-in link").wait_for(timeout=10000)
                page.get_by_text(
                    "Lasts until the server stops; the next start uses "
                    "MARKITAI_SERVE_TOKEN again"
                ).first.wait_for(timeout=10000)
                link = f"{base}/#token={new}"
                if link not in page.content():
                    raise AssertionError(f"the sign-in link {link} is not shown")
                run.shot(page, "access-regenerated")
                old = _api(page, base, "GET", "/api/settings/llm")
                renewed = page.request.fetch(
                    f"{base}/api/settings/llm",
                    headers={"Authorization": f"Bearer {new}"},
                )
                if (old.status, renewed.status) != (401, 200):
                    raise AssertionError(
                        f"old token {old.status}, new token {renewed.status}"
                    )
                # This tab keeps working with the new token
                before = len(run.api)
                _open(page, lambda: page.goto(f"{base}/settings/models"))
                _wait(
                    page,
                    lambda: run.saw("GET", r"^/api/settings/llm$", 200, since=before),
                    timeout=15,
                    what="this tab's settings with the new token",
                )
                # The other tab, still holding the old token, is refused
                before = len(run.api)
                _open(other, lambda: other.goto(f"{base}/jobs"))
                _wait(
                    other,
                    lambda: run.saw("GET", r"^/api/", 401, since=before),
                    timeout=15,
                    what="the other tab's old token refused",
                )
                run.shot(other, "access-other-tab")
            finally:
                fresh.close()
            return (
                "Regenerate token: new token shown once with its link and QR code; "
                "the old token answers 401 (another tab holding it too), the new one "
                "200, and this tab stays signed in"
            )

        run.step(page, "access-regenerate", regenerate)
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
        "--axe",
        type=Path,
        help="axe.min.js (axe-core): check every screenshot's page, fail on a violation",
    )
    parser.add_argument(
        "--theme",
        choices=("light", "dark"),
        default="light",
        help="the browser's colour scheme (the workbench follows it by default)",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    global AXE_SOURCE
    if args.axe is not None:
        AXE_SOURCE = args.axe.read_text(encoding="utf-8")
    _, llm_port = _serve(_FakeLLM)
    _, pages_port = _serve(_Pages)
    global SITE_ASSETS, SITE_SPKI
    scratch = Path(tempfile.mkdtemp(prefix="markitai-e2e-"))
    assets = _serve_https(_SiteAssets, scratch / "tls")
    if assets is not None:
        SITE_ASSETS = f"https://127.0.0.1:{assets[1]}"
        SITE_SPKI = assets[2]
    results: dict[str, Any] = {
        "widths": {},
        "bundle": None,
        "browser": args.browser,
        "theme": args.theme,
    }
    interrupted: BaseException | None = None
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
                        axe_checked=run.axe_checked,
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
        if data.get("axe_checked"):
            lines.append(
                f"| {width} | axe | ok | axe-core checked {data['axe_checked']} "
                "screenshots' pages (violations would be listed as errors above) | |"
            )
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
