#!/usr/bin/env python3
"""The desktop app end to end: its e2e build driven over mygo's automation socket.

    python3.13 scripts/desktop_e2e.py ROUND [--only NAME,...] [--app PATH]

Build the app first, around a markitai that has `serve --supervised`:

    python3.13 scripts/desktop_build.py stage --archive markitai-v<version>-aarch64-apple-darwin.tar.gz
    python3.13 scripts/desktop_build.py build --e2e --platform darwin/arm64

Scenarios (all by default):

  ready        the workbench loads from 127.0.0.1, on a port from 23600 to 23699,
               with the token in sessionStorage and out of the address;
               capabilities answer 200 with the token, 401 without, supervised
               true, no update; no Access token or update button in the
               sidebar; serve.log names neither the token nor an address; a
               link to another site the user follows opens in the browser and
               the window stays, a page's own navigation elsewhere goes
               nowhere; a stop-now link in the workbench does nothing
  convert      a synthetic DataTransfer drop and Convert: the job finishes with
               the expected Markdown
  download     a blob download and a ticket download land where the save panel
               answered, with the right content
  pdf          Download as > PDF... > Print... in the preview writes a %PDF file
               and the page receives afterprint
  port         a second launch serves the same origin, the first launch's log
               moved to serve.previous.log; with that port taken, a third
               launch takes another from 23600 to 23699 and remembers it
  quit-idle    SIGTERM to the app: the app exits 0, serve drains, its PID is
               gone, its control file removed
  quit-uncounted
               SIGTERM while serve cannot answer (SIGSTOP): the app asks with
               no count; Cancel keeps it working; with nothing unfinished,
               Quit from the menu then asks nothing
  quit-busy-when-done, quit-busy-now, quit-busy-then-now
               SIGTERM while a URL conversion is held open by a local page: the
               question answers "Quit When Done" (the Stopping page offers Stop
               Now alone, the app waits, serve drains and the conversion
               finishes in it), "Stop and Quit" (serve stops at once with the
               conversion held; the next start says it was interrupted), or
               "Quit When Done" and then the Stopping page's Stop Now (serve
               drains, then stops at once; interrupted)
  crash        SIGKILL to the app while a conversion is held: its input closed,
               serve drains and keeps running the conversion; a launch
               meanwhile shows that the server stopped; once serve has
               finished and released the home, Try Again brings the workbench
               on the same origin, the conversion done
  home-busy    another `markitai serve` holds the home: the Failed page shows
               why, with Try Again and Show Log; once it stops, Try Again
               brings the workbench, its serve's output after the failed one's
               in the launch's serve.log

Each scenario runs the app with its own HOME, MARKITAI_HOME and app data in
.local/desktop-e2e/ROUND/<scenario>/ and leaves there result.json, the app's
output, screenshots, and the app's data and serve logs. ROUND/run.json names
the app, the bundled markitai, the staged archive and the commit. Save panels
and the busy-quit question answer through the e2e build's environment, which
also records what would open in the browser (desktop/e2e.go); the native panel
and dialog are checked by hand.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import base64
import datetime
import hashlib
import json
import os
import platform
import plistlib
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import zipfile

from isolation import install_state_guard, isolated_env

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "desktop/build/e2e/darwin-arm64/Markitai.app"
EVIDENCE = ROOT / ".local/desktop-e2e"
TOKEN_KEY = "markitai.service-token"
READY_TIMEOUT = 90
# The page a held conversion fetches: enough text for a plain fetch to keep.
SLOW_PAGE = ("<!doctype html><html><head><title>Held page</title></head><body><h1>Held page</h1>"
             + "".join(f"<p>Paragraph {n} of a page the desktop e2e holds open until the app quits.</p>"
                       for n in range(1, 21))
             + "</body></html>").encode()
DROPPED = "desktop-e2e.html"
DROPPED_HTML = ("<!doctype html><html><head><title>Desktop e2e</title></head>"
                "<body><h1>Desktop e2e</h1><p>hello desktop</p></body></html>")
# Proxy settings would send the held page's fetch elsewhere.
PROXIES = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# Where the app looks for a port first (desktop/port.go).
APP_PORTS = range(23600, 23700)
# What serve writes to serve.log (MARKITAI_LANG=en).
LISTENING = "Markitai server listening on http://127.0.0.1:"
STOPPED_BY_APP = "The app that started this server stops it through standard input."
DRAINING = "Server stopping: draining active conversions and saving history."
STOPPING_NOW = "Server stopping now: running conversions are abandoned"
# The e2e build's name (desktop/e2e.go), which its application menu shows.
APP_NAME = "Markitai E2E"
# The status pages' buttons (desktop/window.go).
RETRY, SHOW_LOG, STOP_NOW = "markitai-desktop:retry", "markitai-desktop:show-log", "markitai-desktop:stop-now"


class Failure(Exception):
    """A scenario cannot go on."""


class AutomationError(Exception):
    """The automation socket refused a call or went away (the page may be navigating)."""


def wait_for(what, probe, timeout, interval=0.2):
    """probe's first truthy answer; calls that fail while a page loads are retried."""
    deadline = time.monotonic() + timeout
    last = None
    while True:
        try:
            value = probe()
        except (AutomationError, OSError, ValueError) as error:
            value, last = None, error
        if value:
            return value
        if time.monotonic() > deadline:
            raise Failure(f"no {what} within {timeout} s" + (f" (last error: {last})" if last else ""))
        time.sleep(interval)


def eventually(what, probe, timeout):
    """wait_for's answer, or None when it times out."""
    try:
        return wait_for(what, probe, timeout)
    except Failure:
        return None


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def is_markitai(pid):
    command = subprocess.run(["ps", "-o", "comm=", "-p", str(pid)], capture_output=True, text=True).stdout
    return Path(command.strip()).name == "markitai"


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def http(url, token=None, timeout=10):
    """(status, body) of a GET on loopback, with the token as a Bearer header."""
    request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    try:
        with LOOPBACK.open(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


class Automation:
    """mygo's automation plugin: newline-delimited JSON calls over a Unix socket."""

    def __init__(self, path, timeout=30):
        def connect():
            attempt = socket.socket(socket.AF_UNIX)
            try:
                attempt.connect(path)
            except OSError:
                attempt.close()
                raise
            return attempt
        self.socket = wait_for("automation socket", connect, timeout, 0.1)
        self.socket.settimeout(120)
        self.stream = self.socket.makefile("rw", encoding="utf-8", newline="\n")
        self.sent = 0

    def call(self, method, **params):
        self.sent += 1
        try:
            self.stream.write(json.dumps({"id": self.sent, "method": method, "params": params}) + "\n")
            self.stream.flush()
            while True:
                line = self.stream.readline()
                if not line:
                    raise AutomationError("the automation socket closed")
                answer = json.loads(line)
                if answer.get("id") == self.sent:
                    break
        except OSError as error:
            raise AutomationError(str(error)) from error
        if "error" in answer:
            raise AutomationError(f"{method}: {answer['error']}")
        return answer.get("result") or {}

    def window(self):
        windows = self.call("windows").get("windows") or []
        if not windows:
            raise AutomationError("no window")
        return windows[0]["id"]

    def eval(self, script):
        return self.call("eval", window=self.window(), script=script).get("value")

    def close(self):
        self.socket.close()


# What the window shows: a status page (about:blank, body data-phase) or the workbench.
PAGE_STATE = f"""
let token = null;
try {{ token = sessionStorage.getItem({json.dumps(TOKEN_KEY)}); }} catch (_) {{}}
const body = document.body;
return {{url: location.href, phase: body?.dataset.phase ?? null, cause: body?.dataset.cause ?? null,
        tail: document.querySelector("pre")?.textContent ?? "", token: token !== null,
        sidebar: !!document.querySelector('a[href="/settings/models"]'),
        actions: body?.dataset.phase ? [...document.querySelectorAll(".actions a")].map((a) => a.getAttribute("href"))
                                     : null}};
"""
AUTH = f"const H = {{Authorization: 'Bearer ' + sessionStorage.getItem({json.dumps(TOKEN_KEY)})}};"


class App:
    """One launch of the e2e build in a scenario's sandbox."""

    def __init__(self, scenario, index, **environment):
        self.scenario = scenario
        self.index = index
        executable = scenario.run.app / "Contents/MacOS/Markitai"
        socket_name = f"a{index}.sock"
        self.output = open(scenario.dir / f"app-{index}.log", "wb")
        self.process = subprocess.Popen(
            [str(executable), "-AppleLanguages", "(en)"], cwd=scenario.dir, stdin=subprocess.DEVNULL,
            stdout=self.output, stderr=subprocess.STDOUT, start_new_session=True,
            env=scenario.environment(MYGO_AUTOMATION=socket_name, MARKITAI_DESKTOP_E2E=str(scenario.app_data),
                                     **environment))
        self.socket = os.path.relpath(scenario.dir / socket_name)
        self.ui = None
        print(f"  launch {index}: pid {self.process.pid}", flush=True)

    def connect(self):
        self.ui = Automation(self.socket)

    def page(self):
        state = self.ui.eval(PAGE_STATE)
        if not isinstance(state, dict):
            raise AutomationError(f"no page state yet: {state!r}")
        return state

    def ready(self):
        """The workbench's origin once it has rendered with the token; a Failed page fails."""
        def probe():
            if self.process.poll() is not None:
                raise Failure(f"the app exited with {self.process.returncode} before it was ready")
            state = self.page()
            if state["phase"] == "failed":
                raise Failure(f"the app shows Failed ({state['cause']}): {state['tail'][-800:]}")
            return state if state["url"].startswith("http://127.0.0.1:") and state["token"] and state["sidebar"] else None
        state = wait_for("ready workbench", probe, READY_TIMEOUT)
        self.ui.call("idle", window=self.ui.window(), timeout=10000)
        address = urlsplit(state["url"])
        return f"{address.scheme}://{address.netloc}"

    def phase(self, phase, timeout):
        return wait_for(f"{phase} page", lambda: (state := self.page())["phase"] == phase and state, timeout)

    def token(self):
        return self.ui.eval(f"return sessionStorage.getItem({json.dumps(TOKEN_KEY)})")

    def capture(self, name):
        if self.ui is None:
            return
        try:
            png = self.ui.call("capture", window=self.ui.window())["png"]
            (self.scenario.dir / f"{name}.png").write_bytes(base64.b64decode(png))
        except (AutomationError, KeyError, ValueError) as error:
            print(f"  (no {name} screenshot: {error})")

    def signal(self, number):
        self.process.send_signal(number)

    def exited(self, timeout):
        try:
            return self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            raise Failure(f"the app (pid {self.process.pid}) still runs {timeout} s later") from None

    def quit(self):
        """SIGTERM, which mygo takes as Quit (as Cmd-Q), and the app's exit status."""
        self.signal(signal.SIGTERM)
        code = self.exited(30)
        self.close()
        return code

    def quit_from_menu(self):
        """Quit from the application menu, as Cmd-Q does, and the app's exit status: after one
        SIGTERM, mygo ends the app at once on the next."""
        try:
            self.ui.call("menu", window=self.ui.window(), path=[APP_NAME, f"Quit {APP_NAME}"])
        except AutomationError:
            pass  # the app may have quit before answering
        code = self.exited(30)
        self.close()
        return code

    def close(self):
        if self.ui is not None:
            self.ui.close()
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait(10)
        self.output.close()


class HeldPage:
    """A local page whose GET is held until released: a conversion that stays running."""

    def __init__(self):
        self.requested, self.release = threading.Event(), threading.Event()
        held = self

        class Handler(BaseHTTPRequestHandler):
            def do_HEAD(self):
                self.answer(include_body=False)

            def do_GET(self):
                if self.path == "/held.html":
                    held.requested.set()
                    held.release.wait(300)
                self.answer(include_body=True)

            def answer(self, include_body):
                if self.path != "/held.html":
                    self.send_error(404)
                    return
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(SLOW_PAGE)))
                    self.end_headers()
                    if include_body:
                        self.wfile.write(SLOW_PAGE)
                except OSError:
                    pass

            def log_message(self, *arguments):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/held.html"

    def close(self):
        self.release.set()
        self.server.shutdown()
        self.server.server_close()


class OtherSite:
    """A site on another loopback origin that a window allowed to go there would load: it lists the
    paths asked of it."""

    def __init__(self):
        self.asked = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                site.asked.append(self.path)
                body = b"<!doctype html><html><head><title>Elsewhere</title></head><body>Elsewhere</body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *arguments):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.origin = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class Scenario:
    def __init__(self, run, name):
        self.run, self.name = run, name
        self.dir = run.dir / name
        self.dir.mkdir(parents=True)
        self.home, self.markitai_home, self.app_data = self.dir / "home", self.dir / "markitai-home", self.dir / "app"
        self.base = isolated_env(self.home, self.markitai_home)
        # A login shell in an empty HOME would start zsh's new-user setup.
        (self.home / ".zshrc").write_text("# desktop e2e\n", encoding="utf-8")
        self.checks, self.notes, self.apps, self.others, self.sites = [], [], [], [], []
        self.serve_pids = set()

    def environment(self, **extra):
        """The sandbox's environment as Finder would start the app: no terminal, so the app
        reads a login shell's; nothing of the developer's markitai, mygo or proxies."""
        environment = {key: value for key, value in self.base.items()
                       if not key.startswith(("MARKITAI_", "MYGO_")) and key.lower() not in PROXIES
                       and key not in {"TERM", "TERM_PROGRAM", "ZDOTDIR"}}
        environment.update(MARKITAI_HOME=str(self.markitai_home), SHELL="/bin/zsh", LANG="en_US.UTF-8",
                           MARKITAI_LANG="en", **extra)
        return environment

    def launch(self, **environment):
        app = App(self, len(self.apps) + 1, **environment)
        self.apps.append(app)
        app.connect()
        return app

    def check(self, name, ok, detail=""):
        self.checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f": {detail}" if detail != "" else ""), flush=True)
        return bool(ok)

    def note(self, text):
        self.notes.append(text)
        print(f"  note {text}", flush=True)

    def require(self, name, ok, detail=""):
        if not self.check(name, ok, detail):
            raise Failure(name)

    def control(self):
        """serve's control file: its PID (never the secret beside it), or None."""
        try:
            pid = json.loads((self.markitai_home / "serve/control").read_text(encoding="utf-8"))["pid"]
        except (OSError, ValueError, KeyError, TypeError):
            return None
        self.serve_pids.add(pid)
        return pid

    def serve_ended(self, pid, timeout):
        """Whether serve's control file went with it; serve itself must end."""
        wait_for(f"serve {pid} to end", lambda: not alive(pid), timeout)
        return bool(eventually("serve's control file to go",
                               lambda: not (self.markitai_home / "serve/control").exists(), 5))

    def log(self, name="serve.log"):
        """The app's serve.log (this launch's) or serve.previous.log, "" when there is none."""
        try:
            return (self.app_data / "logs" / name).read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""

    def drained(self, name="serve.log"):
        """Whether serve's output ends with its drain line, as a drain that exits 0 leaves it: one
        that then fails says why after it, and one stopped at once (129) says that instead."""
        lines = self.log(name).strip().splitlines()
        return bool(lines) and lines[-1].endswith(DRAINING) and STOPPING_NOW not in self.log(name)

    def cleanup(self):
        for site in self.sites:
            site.close()
        for app in self.apps:
            app.close()
        for process in self.others:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.control()
        for pid in self.serve_pids:
            if alive(pid) and is_markitai(pid):
                print(f"  stopping serve {pid}, still running after the scenario")
                os.kill(pid, signal.SIGTERM)
                try:
                    wait_for(f"serve {pid} to end", lambda pid=pid: not alive(pid), 10)
                except Failure:
                    os.kill(pid, signal.SIGKILL)

    def saved(self, suffix, timeout=30):
        """The file the save panel answered, once its size holds still."""
        folder = self.app_data / "saved"
        sizes = {}

        def probe():
            for path in sorted(folder.glob(f"*{suffix}")) if folder.exists() else []:
                size = path.stat().st_size
                if size and sizes.get(path) == size:
                    return path
                sizes[path] = size
            return None
        return wait_for(f"saved {suffix} file", probe, timeout, 0.5)


def history(origin, token):
    """The finished jobs: the history lists no running one."""
    status, body = http(f"{origin}/api/history?include=items", token)
    if status != 200:
        raise Failure(f"GET /api/history answered {status}")
    return json.loads(body)


def snapshot(origin, token, job):
    status, body = http(f"{origin}/api/jobs/{job}", token)
    if status != 200:
        raise Failure(f"GET /api/jobs/{job} answered {status}")
    return json.loads(body)


def finished(origin, token, job=None, name=None, timeout=60):
    """(job, item) of the finished item named `name` or in `job`; an error fails."""
    def probe():
        for entry in history(origin, token):
            for item in entry.get("items") or []:
                if (job is None or entry["job_id"] == job) and (name is None or name in (item.get("name") or "")):
                    if item.get("status") == "error":
                        raise Failure(f"item {item.get('name')} failed: {json.dumps(item)[:600]}")
                    if item.get("status") == "done":
                        return entry["job_id"], item["item_id"]
        return None
    return wait_for(f"finished item {name or job}", probe, timeout, 0.5)


def convert_dropped(scenario, app, origin, token):
    """Drop DROPPED on the Convert page and press Convert; (job, item, markdown)."""
    clicked = app.ui.eval(f"""
        const dt = new DataTransfer();
        dt.items.add(new File([{json.dumps(DROPPED_HTML)}], {json.dumps(DROPPED)}, {{type: "text/html"}}));
        for (const type of ["dragenter", "dragover", "drop"])
          window.dispatchEvent(new DragEvent(type, {{dataTransfer: dt, bubbles: true, cancelable: true}}));
        for (let i = 0; i < 100; i++) {{
          const button = [...document.querySelectorAll("button")]
            .find((b) => b.textContent.trim() === "Convert" && !b.closest("nav, aside") && !b.disabled);
          if (button) {{ button.click(); return true; }}
          await new Promise((r) => setTimeout(r, 100));
        }}
        return false;
    """)
    scenario.require("the drop enabled Convert", clicked)
    job, item = finished(origin, token, name=DROPPED)
    status, body = http(f"{origin}/api/jobs/{job}/items/{item}/result", token)
    markdown = json.loads(body).get("markdown", "") if status == 200 else ""
    scenario.require("the conversion's Markdown", status == 200 and "# Desktop e2e" in markdown
                     and "hello desktop" in markdown, f"{status} {markdown[:200]!r}")
    return job, item, markdown


def ready(s):
    app = s.launch()
    origin = app.ready()
    app.capture("ready")
    address = urlsplit(origin)
    s.check("the workbench is on 127.0.0.1, on a port from 23600 to 23699",
            address.hostname == "127.0.0.1" and address.port in APP_PORTS, origin)
    token = app.token()
    s.check("the token is in sessionStorage", bool(token))
    log = s.log()
    s.check("serve.log names neither the token nor an address, and says the app stops serve",
            LISTENING in log and STOPPED_BY_APP in log and bool(token) and token not in log and "#token=" not in log
            and "Access token" not in log and "Open in your browser" not in log, log.strip().splitlines())
    status, body = http(f"{origin}/api/capabilities", token)
    capabilities = json.loads(body) if status == 200 else {}
    s.check("capabilities with the token: 200", status == 200, status)
    s.check("capabilities say supervised", capabilities.get("supervised") is True, capabilities.get("supervised"))
    s.check("capabilities offer no update", not capabilities.get("update"), capabilities.get("update"))
    status, _ = http(f"{origin}/api/capabilities")
    s.check("capabilities without the token: 401", status == 401, status)
    page = app.ui.eval("""return {access: !!document.querySelector('a[href="/settings/access"]'),
                                  update: !!document.querySelector(".upd-icon, .update-check"),
                                  address: location.href.includes("token")};""")
    s.check("no Access token in the sidebar", not page["access"])
    s.check("no update button", not page["update"])
    s.check("the token is out of the address", not page["address"])
    elsewhere = OtherSite()
    s.sites.append(elsewhere)
    external = s.app_data / "external"
    followed = f"{elsewhere.origin}/x"
    app.ui.eval(f"""const a = document.createElement("a");
        a.href = {json.dumps(followed)};
        document.body.append(a); a.click(); a.remove(); return true;""")
    opened = eventually("the followed link in the browser", lambda: external.is_file() and external.read_text(), 10)
    time.sleep(2)
    s.check("a link to another site the user follows opens in the browser; the window stays on the workbench",
            opened == followed + "\n" and app.page()["url"].startswith(origin) and not elsewhere.asked,
            f"browser: {opened!r}, the site was asked for {elsewhere.asked}")
    # Past 1 s: WebKit counts a timer set within 1 s of a gesture (the eval) as that gesture.
    app.ui.eval(f"""setTimeout(() => {{ location.href = {json.dumps(elsewhere.origin + "/y")}; }}, 1500);
        return true;""")
    time.sleep(4)
    opened = external.read_text() if external.is_file() else ""
    s.check("a page's own navigation elsewhere goes nowhere, not to the browser either",
            opened == followed + "\n" and app.page()["url"].startswith(origin) and not elsewhere.asked,
            f"browser: {opened!r}, the site was asked for {elsewhere.asked}")
    pid = s.control()
    app.ui.eval("""const a = document.createElement("a");
        a.href = "markitai-desktop:stop-now"; a.target = "_blank";
        document.body.append(a); a.click(); a.remove(); return true;""")
    time.sleep(2)
    s.check("a stop-now link in the workbench does nothing", app.page()["url"].startswith(origin) and alive(pid),
            pid)
    s.check("the app quits with 0", app.quit() == 0)


def convert(s):
    app = s.launch()
    origin = app.ready()
    convert_dropped(s, app, origin, app.token())
    app.capture("converted")
    app.quit()


def download(s):
    app = s.launch()
    origin = app.ready()
    token = app.token()
    job, item, markdown = convert_dropped(s, app, origin, token)
    status = app.ui.eval(AUTH + f"""
        const r = await fetch("/api/jobs/{job}/items/{item}/result", {{headers: H}});
        const {{markdown}} = await r.json();
        const a = document.createElement("a");
        a.href = URL.createObjectURL(new Blob([markdown], {{type: "text/markdown"}}));
        a.download = "desktop-e2e-blob.md";
        document.body.append(a); a.click(); a.remove();
        return r.status;
    """)
    s.require("the page read the result", status == 200, status)
    saved = s.saved(".md")
    s.check("the blob download is saved where the panel answered, with the Markdown",
            saved.read_text(encoding="utf-8") == markdown, saved.name)
    status = app.ui.eval(AUTH + f"""
        const r = await fetch("/api/download-tickets", {{method: "POST",
          headers: {{...H, "Content-Type": "application/json"}},
          body: JSON.stringify({{path: "/api/jobs/{job}/items/{item}/archive"}})}});
        if (!r.ok) return r.status;
        const a = document.createElement("a");
        a.href = (await r.json()).url;
        a.download = "desktop-e2e-ticket.zip";
        document.body.append(a); a.click(); a.remove();
        return r.status;
    """)
    s.require("a download ticket", status == 201, status)
    archive = s.saved(".zip")
    with zipfile.ZipFile(archive) as bundle:
        texts = [bundle.read(name).decode("utf-8", "replace") for name in bundle.namelist() if name.endswith(".md")]
    s.check("the ticket download is saved, an archive holding the Markdown",
            any("hello desktop" in text for text in texts), f"{archive.name}: {len(texts)} Markdown file(s)")
    app.quit()


EXPORT_PDF = """
window.__e2eAfterprint = 0;
addEventListener("afterprint", () => { window.__e2eAfterprint += 1; });
const until = async (find) => {
  for (let i = 0; i < 150; i++) { const found = find(); if (found) return found; await new Promise((r) => setTimeout(r, 100)); }
  return null;
};
const tools = await until(() => document.querySelector(".peek-tools"));
if (!tools) return "no preview";
const opener = tools.querySelector(".split-caret") ?? tools.querySelector('button[aria-haspopup="menu"]');
if (!opener) return "no Download as menu";
opener.click();
const entry = await until(() => [...document.querySelectorAll('[role="menuitem"]')]
  .find((e) => e.querySelector(".menu-ext")?.textContent.trim() === ".pdf" && !e.disabled));
if (!entry) return "no PDF... entry";
entry.click();
const print = await until(() => document.querySelector(".pdf-card .btn-primary"));
if (!print) return "no Print... button";
print.click();
return "printing";
"""


def pdf(s):
    app = s.launch()
    origin = app.ready()
    job, item, _ = convert_dropped(s, app, origin, app.token())
    app.ui.eval(f"location.assign('/jobs?sel={job}/{item}&view=preview'); return true;")
    wait_for("the preview", lambda: app.ui.eval("return !!document.querySelector('.peek-tools')"), 30)
    app.ui.call("idle", window=app.ui.window(), timeout=10000)
    app.capture("preview")
    step = app.ui.eval(EXPORT_PDF)
    if not s.check("Download as > PDF... > Print... in the preview", step == "printing", step):
        step = app.ui.eval("""window.__e2eAfterprint = 0;
            addEventListener("afterprint", () => { window.__e2eAfterprint += 1; });
            document.title = "desktop-e2e-print"; window.print(); return "window.print";""")
    saved = s.saved(".pdf")
    s.check(f"{step} saved a PDF", saved.read_bytes().startswith(b"%PDF"), f"{saved.name}, {saved.stat().st_size} bytes")
    after = eventually("afterprint", lambda: app.ui.eval("return window.__e2eAfterprint"), 10)
    restored = eventually("the page's restore", lambda: app.ui.eval(
        "return !document.body.classList.contains('is-printing')"), 5)
    s.check("the page received afterprint and restored itself", after and restored, f"afterprint x{after or 0}")
    app.quit()


def remembered_port(s):
    return json.loads((s.app_data / "data/desktop.json").read_text(encoding="utf-8")).get("port")


def press(app, href):
    """Clicks the status page's button for href; whether the page has it."""
    return app.ui.eval(f"""const a = document.querySelector('.actions a[href="{href}"]');
        if (!a) return false; a.click(); return true;""")


def port(s):
    first = s.launch()
    origin = first.ready()
    chosen = urlsplit(origin).port
    s.check("the first launch takes a port from 23600 to 23699", chosen in APP_PORTS, origin)
    first.quit()
    s.check("the app remembered its port", remembered_port(s) == chosen, remembered_port(s))
    second = s.launch()
    again = second.ready()
    s.check("the second launch serves the same origin", again == origin, f"{origin} then {again}")
    log, previous = s.log(), s.log("serve.previous.log")
    s.check("the first launch's log is now serve.previous.log, and this launch's serve.log its own",
            s.drained("serve.previous.log") and previous.count(LISTENING) == 1 and log.count(LISTENING) == 1
            and DRAINING not in log, f"previous: {previous.count(LISTENING)} start(s), now: {log.count(LISTENING)}")
    second.quit()
    with socket.socket() as taken:
        # The port's last connections may wait out TIME_WAIT; listening still holds it.
        taken.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        taken.bind(("127.0.0.1", chosen))
        taken.listen()
        third = s.launch()
        moved = urlsplit(third.ready()).port
        third.quit()
    s.check("with that port taken, a launch takes another from 23600 to 23699",
            moved != chosen and moved in APP_PORTS, f"{chosen} then {moved}")
    s.check("and remembers it", remembered_port(s) == moved, remembered_port(s))


def quit_idle(s):
    app = s.launch()
    app.ready()
    pid = s.control()
    s.require("serve wrote its control file", pid is not None and alive(pid), pid)
    s.check("the app quits with 0 on SIGTERM", app.quit() == 0)
    s.check("serve ended and removed its control file", s.serve_ended(pid, 10), pid)
    s.check("serve's log ends with its drain, as a server that exits 0 leaves it", s.drained(),
            s.log().strip().splitlines()[-1:])
    s.check("with nothing unfinished, the app asked nothing", not (s.app_data / "asked").exists())


def quit_uncounted(s):
    app = s.launch(MARKITAI_DESKTOP_E2E_QUIT="cancel")
    origin = app.ready()
    pid = s.control()
    s.require("serve wrote its control file", pid is not None and alive(pid), pid)
    asked = s.app_data / "asked"
    os.kill(pid, signal.SIGSTOP)
    try:
        app.signal(signal.SIGTERM)
        answer = eventually("the quit question", lambda: asked.is_file() and asked.read_text(), 10)
    finally:
        os.kill(pid, signal.SIGCONT)
    s.check("with serve unable to count, the app asks anyway, with no count", answer == "unknown", answer)
    time.sleep(1)
    s.check("Cancel keeps the app, serve and the workbench",
            app.process.poll() is None and alive(pid) and app.page()["url"].startswith(origin))
    asked.unlink(missing_ok=True)
    s.check("with nothing unfinished, a quit from the menu asks nothing and the app exits 0",
            app.quit_from_menu() == 0 and not asked.exists())
    s.check("serve ended and its log ends with its drain", s.serve_ended(pid, 10) and s.drained(), pid)


def hold_a_conversion(s, app, origin, token):
    """A URL job of a HeldPage, its item running; (held, job)."""
    held = HeldPage()
    s.sites.append(held)
    created = app.ui.eval(AUTH + f"""
        const r = await fetch("/api/jobs", {{method: "POST", headers: H,
          body: new URLSearchParams({{urls: JSON.stringify([{json.dumps(held.url)}])}})}});
        return {{status: r.status, job: (await r.json()).job_id ?? null}};
    """)
    s.require("a URL job of the held page", created["status"] == 201 and created["job"], created)
    job = created["job"]
    s.require("serve fetches the held page", held.requested.wait(30))
    s.require("the job's item runs", wait_for("running item", lambda: any(
        item.get("status") in ("queued", "running") for item in snapshot(origin, token, job)["items"]), 10))
    return held, job


def ask_to_quit(s, choice):
    """A launch whose quit question answers `choice`, told to quit while it holds a conversion;
    (app, origin, held, job, serve's PID)."""
    app = s.launch(MARKITAI_DESKTOP_E2E_QUIT=choice)
    origin = app.ready()
    held, job = hold_a_conversion(s, app, origin, app.token())
    pid = s.control()
    app.signal(signal.SIGTERM)
    asked = s.app_data / "asked"
    answer = eventually("the quit question", lambda: asked.is_file() and asked.read_text(), 10)
    s.check("the app asked about the one unfinished conversion", answer == "1", answer)
    return app, origin, held, job, pid


def interrupted(s, origin, job):
    """The next launch, on the same origin, finds the held conversion interrupted."""
    again = s.launch()
    s.check("the next launch is ready on the same origin", again.ready() == origin)
    items = snapshot(origin, again.token(), job).get("items") or []
    s.check("it marks the stopped conversion interrupted", items and all(
        item.get("status") == "error" and item.get("error_code") == "interrupted" for item in items),
        [(item.get("status"), item.get("error_code")) for item in items])
    again.quit()


def quit_busy_when_done(s):
    app, origin, held, job, pid = ask_to_quit(s, "when-done")
    page = app.phase("stopping", 15)
    app.capture("stopping")
    s.check("Quit When Done: the Stopping page offers Stop Now alone", page["actions"] == [STOP_NOW],
            page["actions"])
    s.check("the app waits while serve drains, its conversion held",
            eventually("serve's drain", lambda: DRAINING in s.log(), 5) and app.process.poll() is None
            and alive(pid))
    held.release.set()
    s.check("the app quits with 0 once serve has drained", app.exited(90) == 0)
    s.check("serve ended and removed its control file", s.serve_ended(pid, 15), pid)
    s.check("serve's log ends with its drain, as a server that exits 0 leaves it", s.drained(),
            s.log().strip().splitlines()[-1:])
    again = s.launch()
    s.check("the next launch is ready on the same origin", again.ready() == origin)
    s.require("the held conversion finished in the first serve (a restart reads a cut-off job as "
              "interrupted)", finished(origin, again.token(), job=job, timeout=10), job)
    again.quit()


def quit_busy_now(s):
    app, origin, held, job, pid = ask_to_quit(s, "now")
    s.check("Stop and Quit: the app quits with 0", app.exited(20) == 0)
    s.check("serve ended while its conversion was held", eventually(f"serve {pid} to end",
                                                                   lambda: not alive(pid), 10), pid)
    log = s.log()
    s.check("serve's log says it stopped at once, not that it drained", STOPPING_NOW in log and DRAINING not in log,
            log.strip().splitlines()[-1:])
    interrupted(s, origin, job)


def quit_busy_then_now(s):
    app, origin, held, job, pid = ask_to_quit(s, "when-done")
    app.phase("stopping", 15)
    s.require("serve drains", eventually("serve's drain", lambda: DRAINING in s.log(), 5))
    s.require("Stop Now is on the Stopping page", press(app, STOP_NOW))
    s.check("Stop Now: the app quits with 0 while the conversion is held", app.exited(20) == 0)
    s.check("serve ended", eventually(f"serve {pid} to end", lambda: not alive(pid), 10), pid)
    log = s.log()
    s.check("serve's log says it drained, then stopped at once",
            DRAINING in log and STOPPING_NOW in log and log.index(DRAINING) < log.index(STOPPING_NOW),
            log.strip().splitlines()[-2:])
    interrupted(s, origin, job)


def crash(s):
    app = s.launch()
    origin = app.ready()
    pid = s.control()
    s.require("serve wrote its control file", pid is not None and alive(pid), pid)
    held, job = hold_a_conversion(s, app, origin, app.token())
    app.signal(signal.SIGKILL)
    app.exited(10)
    s.check("with the app killed, its input closed: serve drains, running its held conversion",
            eventually("serve's drain", lambda: DRAINING in s.log(), 10) and alive(pid), pid)
    again = s.launch()
    page = again.phase("failed", READY_TIMEOUT)
    again.capture("failed-while-draining")
    s.check("a launch meanwhile shows that its server stopped, with Try Again and Show Log",
            page["cause"] == "exited" and page["actions"] == [RETRY, SHOW_LOG],
            [page["cause"], page["actions"], *page["tail"].strip().splitlines()[-1:]])
    held.release.set()
    started = time.monotonic()
    s.check("serve then finished, drained and released the home", s.serve_ended(pid, 60),
            f"{time.monotonic() - started:.1f} s")
    s.check("its log, now serve.previous.log, ends with its drain", s.drained("serve.previous.log"),
            s.log("serve.previous.log").strip().splitlines()[-1:])
    s.require("Try Again is on the Failed page", press(again, RETRY))
    wait_for("the Failed page to go", lambda: again.page()["phase"] != "failed", 10)
    s.check("Try Again brings the workbench on the same origin", again.ready() == origin)
    s.require("the held conversion finished in the first serve (a restart reads a cut-off job as "
              "interrupted)", finished(origin, again.token(), job=job, timeout=10), job)
    again.quit()


def home_busy(s):
    program = s.run.app / "Contents/Resources/bin/markitai"
    other = subprocess.Popen(
        [str(program), "serve", "--host", "127.0.0.1", "--port", str(free_port()), "--no-open"],
        cwd=s.markitai_home, stdin=subprocess.DEVNULL, stdout=open(s.dir / "other-serve.log", "wb"),
        stderr=subprocess.STDOUT, start_new_session=True, env=s.environment(MARKITAI_NO_UPDATE_CHECK="1"))
    s.others.append(other)
    wait_for("the other serve's control file", lambda: (pid := s.control()) is not None and alive(pid), 30)
    app = s.launch()
    page = app.phase("failed", READY_TIMEOUT)
    app.capture("failed")
    refusal = page["tail"].strip().splitlines()[-1] if page["tail"].strip() else None
    s.check("the Failed page says serve exited, with its log", page["cause"] == "exited" and refusal, refusal)
    s.check("it offers Try Again and Show Log", page["actions"] == [RETRY, SHOW_LOG], page["actions"])
    s.check("the other serve still runs", other.poll() is None)
    other.terminate()
    other.wait(15)
    s.require("Try Again is on the Failed page", press(app, RETRY))
    wait_for("the Failed page to go", lambda: app.page()["phase"] != "failed", 10)
    s.require("Try Again brought the workbench", app.ready())
    app.capture("ready-after-retry")
    log = s.log()
    s.check("the launch's serve.log holds the refused serve's output, then that of the one Try Again started",
            bool(refusal) and refusal in log and LISTENING in log and log.index(refusal) < log.index(LISTENING),
            log.strip().splitlines())
    app.quit()


SCENARIOS = {
    "ready": ready, "convert": convert, "download": download, "pdf": pdf, "port": port,
    "quit-idle": quit_idle, "quit-uncounted": quit_uncounted, "quit-busy-when-done": quit_busy_when_done,
    "quit-busy-now": quit_busy_now, "quit-busy-then-now": quit_busy_then_now, "crash": crash,
    "home-busy": home_busy,
}


class Run:
    def __init__(self, name, app):
        self.dir, self.app = EVIDENCE / name, app


def identity(run, guard):
    """What ran: the app, the markitai inside it, how it was staged, the checkout."""
    app = run.app
    program = app / "Contents/Resources/bin/markitai"
    info = plistlib.loads((app / "Contents/Info.plist").read_bytes())
    stamp = ROOT / "desktop/resources/.stage.json"
    git = lambda *arguments: subprocess.run(["git", "--no-optional-locks", *arguments], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip()
    with_home = run.dir / "version-home"
    version = subprocess.run([str(program), "--version"], capture_output=True, text=True, timeout=60,
                             env=isolated_env(with_home / "home", with_home / "markitai-home")).stdout.strip()
    return {
        "command": [Path(sys.executable).name, *sys.argv],
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": f"{platform.system()} {platform.release()} {platform.machine()}",
        "app": {"path": app.relative_to(ROOT).as_posix() if app.is_relative_to(ROOT) else app.name,
                "bundle": info.get("CFBundleIdentifier"), "version": info.get("CFBundleShortVersionString"),
                "sha256": hashlib.sha256((app / "Contents/MacOS/Markitai").read_bytes()).hexdigest()},
        "markitai": {"version": version, "sha256": hashlib.sha256(program.read_bytes()).hexdigest()},
        "staged": json.loads(stamp.read_text(encoding="utf-8")) if stamp.exists() else None,
        "commit": git("rev-parse", "HEAD"), "changed_paths": len(git("status", "--porcelain").splitlines()),
        "fixtures": f"generated by this script: {DROPPED} and a held local page",
        "guard": guard,
    }


def run_scenario(run, name):
    print(f"{name}", flush=True)
    s = Scenario(run, name)
    started, error = time.monotonic(), None
    try:
        SCENARIOS[name](s)
    except Failure as failure:
        error = str(failure)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
        (s.dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        if error and s.apps and s.apps[-1].process.poll() is None:
            s.apps[-1].capture("failure")
        s.cleanup()
    if error:
        print(f"  FAIL {error}", flush=True)
    result = {"scenario": name, "ok": error is None and all(check["ok"] for check in s.checks),
              "error": error, "checks": s.checks, "notes": s.notes, "seconds": round(time.monotonic() - started, 1)}
    (s.dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def loopback_only(event, arguments):
    if event == "socket.connect":
        address = arguments[1]
        return isinstance(address, (str, bytes)) or address[0] in ("127.0.0.1", "localhost")
    if event == "socket.getaddrinfo":
        return arguments[0] in ("127.0.0.1", "localhost", None)
    return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("round", help="evidence goes to .local/desktop-e2e/ROUND/")
    parser.add_argument("--only", help="comma-separated scenarios: " + ", ".join(SCENARIOS))
    parser.add_argument("--app", type=Path, default=APP, help="the e2e build's Markitai.app")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.round):
        parser.error("ROUND is a plain name: letters, digits, '.', '_' or '-'")
    names = args.only.split(",") if args.only else list(SCENARIOS)
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {', '.join(unknown)}")
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise SystemExit("The desktop e2e runs the darwin/arm64 build, on an Apple silicon Mac")
    app = args.app.resolve()
    if not os.access(app / "Contents/MacOS/Markitai", os.X_OK):
        raise SystemExit(f"No e2e build at {app}; run desktop_build.py stage, then build --e2e --platform darwin/arm64")
    run = Run(args.round, app)
    taken = [name for name in names if (run.dir / name).exists()]
    if taken:
        raise SystemExit(f"{run.dir} already holds {', '.join(taken)}; evidence is kept, so pick another ROUND")
    # Automation sockets are named relative to the repository: macOS allows 104 bytes.
    os.chdir(ROOT)
    guard = install_state_guard("this script's Python file and network events: loopback only, no real "
                                "~/.markitai; the app and serve get a sandboxed HOME and MARKITAI_HOME",
                                allow_network=loopback_only)
    run.dir.mkdir(parents=True, exist_ok=True)
    record = identity(run, {key: guard[key] for key in ("scope", "self_tests")})
    results = record["results"] = [run_scenario(run, name) for name in names]
    record["blocked"] = {"state": guard["blocked_state_events"], "network": guard["blocked_network_events"]}
    run_file = run.dir / "run.json"
    runs = json.loads(run_file.read_text(encoding="utf-8"))["runs"] if run_file.exists() else []
    run_file.write_text(json.dumps({"runs": [*runs, record]}, indent=2) + "\n", encoding="utf-8")
    failed = [result["scenario"] for result in results if not result["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} scenarios passed; evidence in "
          f"{run.dir.relative_to(ROOT)}" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
