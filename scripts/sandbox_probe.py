#!/usr/bin/env python3
"""Probe the framed web page's policy in Chromium against `markitai serve`.

Contract v1.13 §7: ``…/original/view`` answers under ``sandbox
allow-same-origin allow-popups allow-popups-to-escape-sandbox; default-src
'none'; …`` and the workbench frames it with the same sandbox keywords, so
the workbench's own script can read the page's blocks while the page itself
runs nothing. This probe checks that in a real browser:

1. A local "hostile" site serves a page holding ``<script>`` (inline and
   ``src``), ``onload``/``onerror`` handlers, ``javascript:`` links (HTML and
   SVG, ``_self`` and ``_blank``), ``<meta http-equiv=refresh>``, forms
   (``_self`` and ``_blank``), ``<iframe>`` and ``<object>``. Every one of
   them would ask that site for ``/hit/<what>``.
2. An in-process Python server (throwaway HOME, ``--no-auth`` on loopback)
   converts the page's URL with the static fetcher, which keeps the page.
3. Chromium opens a parent page on the server's origin (fulfilled by the
   browser, never served) that frames ``…/original/view`` exactly as the
   workbench does, waits, clicks every link and button in the frame.

It passes when the site saw no ``/hit/`` request, no popup opened, the frame
still shows the page (no refresh, no navigation), its title is unchanged,
and the parent can read the frame's DOM and add a style to it (the reason for
``allow-same-origin``).

``--control`` grants scripts on purpose (``allow-scripts`` in the header and
the frame, ``script-src 'unsafe-inline'``) and expects the probe to fail: it
shows the probe would notice a page that runs.

Usage (needs the serve and browser extras, and Chromium for Playwright)::

    uv run python scripts/sandbox_probe.py [--out probe.json] [--control]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import socketserver
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HITS: list[str] = []

HOSTILE = """<!doctype html><html><head><title>Hostile page</title>
<meta http-equiv="refresh" content="3;url=/hit/meta-refresh">
<script>fetch('/hit/script');document.title='RAN'</script>
<script src="/hit/script-src"></script>
</head><body onload="fetch('/hit/onload')">
<h1 id="h">Hostile page</h1><p>A paragraph the workbench may read.</p>
<img src="x" onerror="fetch('/hit/onerror')">
<a id="js" href="javascript:fetch('/hit/js-link')">js</a>
<a id="jsself" target="_self" href="javascript:fetch('/hit/js-self')">js self</a>
<a id="jsblank" target="_blank" href="javascript:fetch('/hit/js-blank')">js blank</a>
<form id="f" action="/hit/form" method="get"><input name="a" value="1">
<button id="fb">send</button></form>
<form id="f2" action="/hit/form-blank" target="_blank"><button id="fb2">send</button></form>
<iframe src="/hit/inner-iframe"></iframe><object data="/hit/object"></object>
<embed src="/hit/embed">
<svg width="80" height="30"><a href="javascript:fetch('/hit/svg')">
<text id="svgt" y="20">svg link</text></a></svg>
</body></html>"""

#: How the workbench frames the page (contract v1.13 §7).
FRAME_SANDBOX = "allow-same-origin allow-popups allow-popups-to-escape-sandbox"


class _Site(BaseHTTPRequestHandler):
    def log_message(self, *args: Any) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/hit/"):
            HITS.append(self.path)
            self.send_response(204)
            self.end_headers()
            return
        body = HOSTILE.encode()
        self.send_response(200 if self.path == "/page.html" else 404)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_POST = do_GET


class _LocalServer(ThreadingHTTPServer):
    """Binds without a host name lookup (slow on macOS runners)."""

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = self.server_address[:2]


def _playwright_browsers(home: Path) -> Path:
    """Where Playwright installs browsers by default for *home*."""
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", home)) / "ms-playwright"
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "ms-playwright"
    return home / ".cache" / "ms-playwright"


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _start_markitai(home: Path) -> tuple[Any, int]:
    import uvicorn
    from markitai.config import MarkitaiConfig
    from markitai.serve import create_app

    cfg = MarkitaiConfig()
    cfg.cache.enabled = False
    cfg.cache.global_dir = str(home / "cache")
    app = create_app(
        static_dir=home / "no-static",
        jobs_root=home / "jobs",
        config=cfg,
        config_path=home / "config.json",
        configure_logging=False,
    )
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.monotonic() + 30
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the server did not start")
        time.sleep(0.05)
    return server, port


def _convert(base: str, url: str) -> tuple[str, str]:
    """Convert *url*; its item's view path and the policy it is served under."""
    import httpx

    with httpx.Client(base_url=base, timeout=60) as client:
        created = client.post(
            "/api/jobs",
            files=[
                ("urls", (None, json.dumps([url]))),
                ("options", (None, json.dumps({"strategy": "static"}))),
            ],
        )
        created.raise_for_status()
        job_id = created.json()["job_id"]
        deadline = time.monotonic() + 60
        while True:
            job = client.get(f"/api/jobs/{job_id}").json()
            if job["status"] != "running":
                break
            if time.monotonic() > deadline:
                raise RuntimeError("the conversion did not finish")
            time.sleep(0.1)
        item = job["items"][0]
        descriptor = client.get(
            f"/api/jobs/{job_id}/items/{item['item_id']}/original"
        ).json()
        if descriptor.get("view") is None:
            raise RuntimeError(f"no kept page: {item.get('error')} {descriptor}")
        policy = client.get(descriptor["view"]).headers["content-security-policy"]
        return descriptor["view"], policy


def probe(out: Path | None = None, *, control: bool = False) -> dict[str, Any]:
    from playwright.sync_api import sync_playwright

    frame_sandbox = FRAME_SANDBOX
    if control:
        from markitai.serve import originals

        originals.VIEW_POLICY = originals.VIEW_POLICY.replace(
            "sandbox ", "sandbox allow-scripts ", 1
        ).replace(
            "default-src 'none';", "default-src 'none'; script-src 'unsafe-inline';"
        )
        frame_sandbox = f"allow-scripts {FRAME_SANDBOX}"

    with tempfile.TemporaryDirectory(prefix="markitai-probe-") as scratch:
        home = Path(scratch)
        # Never read the operator's ~/.markitai (its .env, its config); keep
        # Playwright's browsers where they were installed
        os.environ.setdefault(
            "PLAYWRIGHT_BROWSERS_PATH", str(_playwright_browsers(Path.home()))
        )
        os.environ["HOME"] = os.environ["USERPROFILE"] = str(home)
        os.environ.pop("MARKITAI_CONFIG", None)
        site = _LocalServer(("127.0.0.1", 0), _Site)
        site.daemon_threads = True
        threading.Thread(target=site.serve_forever, daemon=True).start()
        site_url = f"http://127.0.0.1:{site.server_address[1]}/page.html"
        server, port = _start_markitai(home)
        base = f"http://127.0.0.1:{port}"
        try:
            view, policy = _convert(base, site_url)
            HITS.clear()
            parent = (
                "<!doctype html><html><head><title>Workbench</title></head><body>"
                f'<iframe id="fr" sandbox="{frame_sandbox}" referrerpolicy="no-referrer" '
                f'src="{view}" style="width:800px;height:600px"></iframe></body></html>'
            )
            result: dict[str, Any] = {"view": view, "policy": policy}
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                context = browser.new_context()
                popups: list[str] = []
                context.route(
                    f"{base}/probe",
                    lambda route: route.fulfill(
                        status=200,
                        content_type="text/html; charset=utf-8",
                        headers={
                            # The workbench's own policy (contract §1)
                            "content-security-policy": (
                                "default-src 'none'; script-src 'self'; "
                                "style-src 'self'; connect-src 'self'; "
                                "img-src 'self' blob: https:; font-src 'self'; "
                                "frame-src 'self'; object-src 'none'; "
                                "base-uri 'none'; form-action 'self'; "
                                "frame-ancestors 'none'"
                            )
                        },
                        body=parent,
                    ),
                )
                page = context.new_page()
                # Every page opened from here on is the frame's popup
                clicked = ["(loading)"]
                context.on(
                    "page", lambda popup: popups.append(f"{clicked[0]}: {popup.url}")
                )
                page.goto(f"{base}/probe")
                page.wait_for_timeout(1500)
                frame = page.frame_locator("#fr")
                result["title_at_load"] = page.evaluate(
                    "(() => { const d = document.getElementById('fr').contentDocument;"
                    " return d ? d.title : null })()"
                )
                result["parent_reads"] = page.evaluate(
                    "(() => { const d = document.getElementById('fr').contentDocument;"
                    " const h = d && d.getElementById('h');"
                    " return h ? h.textContent : null })()"
                )
                result["parent_styles"] = page.evaluate(
                    "(() => { const d = document.getElementById('fr').contentDocument;"
                    " if (!d || !d.getElementById('h')) return null;"
                    " const s = d.createElement('style');"
                    " s.textContent = '[data-markitai-linked]{outline:3px solid red}';"
                    " d.head.append(s); const h = d.getElementById('h');"
                    " h.setAttribute('data-markitai-linked', '');"
                    " return getComputedStyle(h).outlineStyle })()"
                )
                for selector in ("#js", "#jsself", "#jsblank", "#fb", "#fb2", "#svgt"):
                    clicked[0] = selector
                    try:
                        frame.locator(selector).click(timeout=1500)
                    except Exception as error:  # noqa: BLE001 - a click that cannot land
                        result.setdefault("click_errors", {})[selector] = type(
                            error
                        ).__name__
                    page.wait_for_timeout(300)
                page.wait_for_timeout(2500)
                result["frame_title"] = page.evaluate(
                    "(() => { const d = document.getElementById('fr').contentDocument;"
                    " return d ? d.title : null })()"
                )
                result["frame_url"] = next(
                    (each.url for each in page.frames if each != page.main_frame), ""
                )
                result["parent_title"] = page.title()
                result["popups"] = popups
                browser.close()
            result["hits"] = list(HITS)
        finally:
            server.should_exit = True
            site.shutdown()
            time.sleep(0.3)
    failures = []
    if result["hits"]:
        failures.append(f"the page reached its site: {result['hits']}")
    if result["popups"]:
        failures.append(f"a popup opened: {result['popups']}")
    if result["parent_reads"] != "Hostile page":
        failures.append(
            f"the workbench cannot read the frame: {result['parent_reads']}"
        )
    if result["parent_styles"] != "solid":
        failures.append("the workbench cannot outline a block in the frame")
    for key in ("title_at_load", "frame_title"):
        if result[key] != "Hostile page":
            failures.append(f"the page's script ran: {key} {result[key]!r}")
    if not result["frame_url"].endswith(view):
        failures.append(f"the frame navigated: {result['frame_url']}")
    if "allow-scripts" in result["policy"] or "script-src" in result["policy"]:
        failures.append(f"the policy allows script: {result['policy']}")
    result["control"] = control
    result["failures"] = failures
    result["ok"] = not failures
    if out is not None:
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    from loguru import logger

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, help="write the result as JSON here")
    parser.add_argument(
        "--control",
        action="store_true",
        help="grant scripts on purpose; succeed only when the probe catches them",
    )
    args = parser.parse_args(argv)
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    result = probe(args.out, control=args.control)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] != args.control else 1


if __name__ == "__main__":
    sys.exit(main())
