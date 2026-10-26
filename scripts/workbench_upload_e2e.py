#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["playwright", "pillow"]
# ///
"""The workbench's folder upload, end to end, in a real Chromium.

Starts ``markitai serve`` (``--bin``, default ``target/debug/markitai``) with
its own HOME and MARKITAI_HOME in a temporary directory, opens the Convert page
in Chinese and checks:

1. folder: with the caret in the URL line, the Folder tool picks a folder of
   24 files (about 9 MB: Word, Excel, HTML and PDF fixtures from this repository
   and a generated 6 MB image-only PDF), then Enter is pressed (the key that
   confirms a system picker). Nothing is sent: no ``POST /api/jobs`` before
   Convert, the files wait in the list and Convert is enabled. Convert then sends
   them in exactly one request, the job is created with 24 items and the page
   follows its events. The time from the click to the service's answer is printed.
2. errors: a proxy's 524 (an HTML page, no reason) and a refusal in the
   service's JSON shape each end in a notice that says why, in Chinese; the 524
   offers Retry, which sends the same files once more and creates the job.
3. refused (with ``--nginx``): the page goes through nginx, which buffers the
   upload as Cloudflare does. A second nginx server sends the server a Host
   without its port, so the server refuses the page's origin before reading the
   9 MB; the server must still read it, so that nginx relays the 403 and its
   reason instead of a 502 (closing early resets the connection). Four tries.

``--ui DIR`` serves the workbench from a ``node build.mjs`` output instead of
the one built into the binary (for testing sources before ``dist/`` is
rebuilt). ``--shots DIR`` saves screenshots at 1280 and 390 px wide.
``--nginx [PATH]`` runs checks 1 and 2 through nginx too (on ``--port`` + 1;
check 3 uses + 2). ``--throttle KBPS`` limits the browser's upload rate (Chrome's network
emulation) so the progress notice can be seen. Needs Playwright's Chromium
(``playwright install chromium``, or ``PLAYWRIGHT_BROWSERS_PATH``).

    uv run scripts/workbench_upload_e2e.py [--bin PATH] [--ui DIR] [--shots DIR]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, Request, Route, sync_playwright

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "crates/markitai-core/src/formats"
SAMPLES = [
    "native/fixtures/legacy-doc-chart.doc",
    "native/fixtures/textedit-word97-long.doc",
    "native/fixtures/textedit-word97-lists.doc",
    "native/fixtures/textedit-word97.doc",
    "native/fixtures/office-images/xlsx-complex.xlsx",
    "pdf/fixtures/page-chrome/how-to-cjk-footer.pdf",
    "pdf/fixtures/table-beside-prose.pdf",
    "pdf/fixtures/default-table.pdf",
    "pdf/fixtures/rtl-text.pdf",
    "pdf/fixtures/borderless-table.pdf",
    "pdf/fixtures/wrapped-table.pdf",
    "pdf/fixtures/cjk-course/form-grid.pdf",
    "pdf/fixtures/cjk-course/long-sentences.pdf",
    "pdf/fixtures/cjk-course/heading-over-rule.pdf",
    "pdf/fixtures/cjk-course/nul-separated.pdf",
]
FOLDER_FILES = 24
LEAD = "无法开始转换"


def make_folder(folder: Path) -> int:
    """24 files of about 9 MB; returns their total size."""
    from PIL import Image

    folder.mkdir(parents=True)
    for name in SAMPLES:
        shutil.copy(FIXTURES / name, folder)
    for index in range(4):
        page = f"<html><body><h1>Notes {index}</h1>" + "<p>Plain paragraph text.</p>" * 400 + "</body></html>"
        (folder / f"notes-{index}.html").write_text(page, encoding="utf-8")
    # Noise does not compress: a scan-sized image-only PDF and four smaller ones.
    pages = [Image.effect_noise((2480, 3508), 40).convert("RGB") for _ in range(2)]
    pages[0].save(folder / "扫描件-scan.pdf", save_all=True, append_images=pages[1:], quality=70, resolution=300)
    for index in range(4):
        Image.effect_noise((900, 1200), 50).convert("RGB").save(folder / f"photo-{index}.pdf", quality=75)
    files = sorted(folder.iterdir())
    assert len(files) == FOLDER_FILES, len(files)
    return sum(path.stat().st_size for path in files)


def start_server(binary: Path, home: Path, port: int) -> tuple[subprocess.Popen[bytes], str]:
    (home / "home").mkdir(parents=True)
    (home / "mh").mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": str(home / "home"),
        "MARKITAI_HOME": str(home / "mh"),
        "MARKITAI_LANG": "zh",
        "LANG": "C.UTF-8",
    }
    server = subprocess.Popen(
        [str(binary), "serve", "--port", str(port), "--no-open"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
    )
    assert server.stdout is not None
    seen = b""
    deadline = time.time() + 30
    while b"#token=" not in seen:
        chunk = server.stdout.read1(4096)
        if not chunk or time.time() > deadline:
            server.kill()
            sys.exit(f"serve did not start:\n{seen.decode(errors='replace')}")
        seen += chunk
    token = re.search(rb"#token=([0-9a-f]+)", seen)
    assert token is not None
    return server, token.group(1).decode()


class Watch:
    """Job submissions the page sends."""

    def __init__(self, page: Page) -> None:
        self.posts: list[Request] = []
        page.on("request", self.seen)

    def seen(self, request: Request) -> None:
        if request.method == "POST" and request.url.endswith("/api/jobs"):
            self.posts.append(request)


def open_page(browser, origin: str, token: str, width: int, ui: Path | None) -> Page:
    phone = width < 600
    context = browser.new_context(
        viewport={"width": width, "height": 860 if not phone else 844},
        device_scale_factor=2 if phone else 1,
        is_mobile=phone,
        has_touch=phone,
        locale="zh-CN",
    )
    page = context.new_page()
    if ui is not None:

        def local(route: Route) -> None:
            name = route.request.url.split("/ui/", 1)[1].split("?", 1)[0]
            path = ui / name
            if path.is_file():
                route.fulfill(path=str(path))
            else:
                route.continue_()

        page.route(re.compile(r".*/ui/[^/]+$"), local)
        page.route(re.compile(r"^[^?#]*/(#.*)?$"), lambda route: route.fulfill(path=str(ui / "index.html")))
    page.goto(f"{origin}/#token={token}")
    page.wait_for_selector("label.upload-folder input", state="attached")
    page.wait_for_load_state("networkidle")
    return page


def pick_folder(page: Page, folder: Path) -> None:
    page.locator("textarea.url-box").click()
    with page.expect_file_chooser() as chooser:
        page.locator("label.upload-folder").click()
    chooser.value.set_files(str(folder))
    # The key that confirms the system picker lands in the page.
    page.locator("textarea.url-box").focus()
    page.keyboard.press("Enter")


def toast_text(page: Page) -> str:
    return " | ".join(page.locator(".toast").all_inner_texts())


def check_folder(page: Page, token: str, folder: Path, size: int, shots: Path | None, tag: str, throttle: int | None) -> float:
    watch = Watch(page)
    pick_folder(page, folder)
    page.wait_for_timeout(2000)
    assert not watch.posts, f"{len(watch.posts)} job request(s) before Convert"
    assert page.locator(".staged-title").inner_text().strip(), "no staged list"
    convert = page.locator("button.tool-convert")
    assert convert.is_enabled(), "Convert is disabled with a folder waiting"
    if shots:
        page.screenshot(path=str(shots / f"picked-{tag}.png"))
    if throttle:
        cdp = page.context.new_cdp_session(page)
        cdp.send("Network.enable")
        cdp.send(
            "Network.emulateNetworkConditions",
            {"offline": False, "latency": 20, "downloadThroughput": -1, "uploadThroughput": throttle * 1024},
        )
    started = time.perf_counter()
    with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith("/api/jobs"), timeout=300_000) as answer:
        convert.click()
        if shots and throttle:
            page.wait_for_function("() => /正在上传 [3-9]\\d%/.test(document.querySelector('.toasts')?.innerText ?? '')", timeout=120_000)
            page.screenshot(path=str(shots / f"uploading-{tag}.png"))
    elapsed = time.perf_counter() - started
    response = answer.value
    assert response.status == 201, f"HTTP {response.status}: {response.text()}"
    assert len(watch.posts) == 1, f"{len(watch.posts)} job requests"
    created = response.json()
    assert len(created["items"]) == FOLDER_FILES, len(created["items"])
    # The service received every byte of every file.
    snapshot = page.request.get(response.url + "/" + created["job_id"], headers={"Authorization": f"Bearer {token}"}).json()
    received = sum(item.get("source_size") or 0 for item in snapshot["items"])
    assert received == size, f"the service received {received} of {size} bytes"
    page.wait_for_url(re.compile(r"/jobs"), timeout=10_000)
    page.wait_for_timeout(1500)
    assert len(watch.posts) == 1, f"{len(watch.posts)} job requests after the job was created"
    if shots:
        page.screenshot(path=str(shots / f"created-{tag}.png"))
    return elapsed


def check_errors(page: Page, folder: Path, shots: Path | None, tag: str) -> None:
    page.goto(page.url.split("/jobs")[0] + "/")
    page.wait_for_selector("label.upload-folder input", state="attached")
    watch = Watch(page)
    small = folder / "notes-0.html"
    page.locator("label.upload-main input").set_input_files(str(small))
    convert = page.locator("button.tool-convert")

    def gateway(route: Route) -> None:
        route.fulfill(status=524, content_type="text/html", body="<html><body>A timeout occurred</body></html>")

    page.route("**/api/jobs", gateway)
    convert.click()
    page.wait_for_function(f"() => (document.querySelector('.toasts')?.innerText ?? '').includes('{LEAD}')")
    text = toast_text(page)
    assert "HTTP 524" in text and "代理" in text, text
    retry = page.locator(".toast .note-action", has_text="重试")
    assert retry.count() == 1, text
    if shots:
        page.screenshot(path=str(shots / f"error-524-{tag}.png"))

    page.unroute("**/api/jobs", gateway)
    page.route(
        "**/api/jobs",
        lambda route: route.fulfill(
            status=422,
            content_type="application/json",
            body='{"detail":"too many job items","code":"invalid_request","reason":"too_many_items"}',
        ),
    )
    convert.click()
    page.wait_for_function("() => (document.querySelector('.toasts')?.innerText ?? '').includes('超出服务上限')")
    text = toast_text(page)
    assert LEAD in text, text
    assert page.locator(".toast .note-action", has_text="重试").count() == 0, "a refusal offers Retry"
    if shots:
        page.screenshot(path=str(shots / f"error-422-{tag}.png"))

    # Retry after a gateway failure sends the same files again and creates the job.
    page.unroute("**/api/jobs")
    page.route("**/api/jobs", gateway)
    convert.click()
    page.locator(".toast .note-action", has_text="重试").wait_for()
    page.unroute("**/api/jobs", gateway)
    before = len(watch.posts)
    with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith("/api/jobs")) as answer:
        page.locator(".toast .note-action", has_text="重试").click()
    assert answer.value.status == 201, answer.value.status
    assert len(watch.posts) == before + 1
    page.wait_for_url(re.compile(r"/jobs"), timeout=10_000)


NGINX = """daemon off; pid {dir}/nginx.pid; error_log {dir}/error.log;
events {{}}
http {{
  client_body_temp_path {dir}/body; proxy_temp_path {dir}/proxy; access_log off; client_max_body_size 0;
  server {{ listen 127.0.0.1:{via}; location / {{ proxy_pass http://127.0.0.1:{port};
    proxy_http_version 1.1; proxy_buffering off; proxy_set_header Host $http_host; }} }}
  server {{ listen 127.0.0.1:{refusing}; location / {{ proxy_pass http://127.0.0.1:{port};
    proxy_http_version 1.1; proxy_buffering off; proxy_set_header Host $host; }} }}
}}
"""


LOGS: Path | None = None


def start_nginx(binary: str, scratch: Path, port: int) -> subprocess.Popen[bytes]:
    """nginx on port + 1 (a plain proxy) and port + 2 (one whose Host drops the port)."""
    global LOGS
    folder = LOGS = scratch / "nginx"
    folder.mkdir()
    (folder / "nginx.conf").write_text(NGINX.format(dir=folder, port=port, via=port + 1, refusing=port + 2))
    proxy = subprocess.Popen([binary, "-p", str(folder), "-c", str(folder / "nginx.conf"), "-e", str(folder / "error.log")])
    time.sleep(1)
    if proxy.poll() is not None:
        sys.exit(f"nginx did not start: {(folder / 'error.log').read_text()}")
    return proxy


def check_refused(page: Page, folder: Path, tag: str, tries: int = 4) -> None:
    for attempt in range(tries):
        page.goto(page.url.split("#")[0].split("/jobs")[0].rstrip("/") + "/")
        page.wait_for_selector("label.upload-folder input", state="attached")
        with page.expect_file_chooser() as chooser:
            page.locator("label.upload-folder").click()
        chooser.value.set_files(str(folder))
        with page.expect_response(lambda response: response.request.method == "POST" and response.url.endswith("/api/jobs"), timeout=120_000) as answer:
            page.locator("button.tool-convert").click()
        page.wait_for_function(f"() => (document.querySelector('.toasts')?.innerText ?? '').includes('{LEAD}')")
        text = toast_text(page)
        if answer.value.status != 403:
            log = (LOGS / "error.log").read_text() if LOGS else ""
            raise AssertionError(f"{tag} try {attempt + 1}: HTTP {answer.value.status} · {text}\n{log}")
        assert "其他网站" in text, text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=ROOT / "target/debug/markitai")
    parser.add_argument("--port", type=int, default=3741)
    parser.add_argument("--origin", help="open the page here (a reverse proxy in front of --port)")
    parser.add_argument("--ui", type=Path, help="serve the workbench from this build output")
    parser.add_argument("--shots", type=Path)
    parser.add_argument("--throttle", type=int, help="upload rate in KB/s")
    parser.add_argument("--nginx", nargs="?", const="nginx", help="also test through nginx (this binary)")
    parser.add_argument("--only", choices=["folder", "errors", "refused"])
    args = parser.parse_args()
    if args.shots:
        args.shots.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mk-upload-e2e-") as scratch:
        folder = Path(scratch) / "samples"
        size = make_folder(folder)
        server, token = start_server(args.bin.resolve(), Path(scratch) / "run", args.port)
        proxy = start_nginx(args.nginx, Path(scratch), args.port) if args.nginx else None
        origin = args.origin or f"http://127.0.0.1:{args.port + 1 if proxy else args.port}"
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch()
                for width, tag in [(1280, "desktop"), (390, "phone")]:
                    if args.only in (None, "folder"):
                        page = open_page(browser, origin, token, width, args.ui)
                        elapsed = check_folder(page, token, folder, size, args.shots, tag, args.throttle)
                        print(f"{tag}: folder ok · {FOLDER_FILES} files, {size / 1e6:.1f} MB in one request, Convert → 201 in {elapsed:.2f} s")
                        page.context.close()
                    if args.only not in ("folder", "refused"):
                        page = open_page(browser, origin, token, width, args.ui)
                        check_errors(page, folder, args.shots, tag)
                        print(f"{tag}: errors ok · 524 and 422 say why, Retry creates the job")
                        page.context.close()
                    if proxy and args.only in (None, "refused"):
                        page = open_page(browser, f"http://127.0.0.1:{args.port + 2}", token, width, args.ui)
                        check_refused(page, folder, tag)
                        print(f"{tag}: refused ok · a 9 MB upload refused before it was read gets the server's 403 and reason through nginx")
                        page.context.close()
                browser.close()
        finally:
            if proxy:
                proxy.terminate()
                proxy.wait(timeout=30)
            server.terminate()
            server.wait(timeout=60)


if __name__ == "__main__":
    main()
