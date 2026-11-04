#!/usr/bin/env python3
"""Markitai Web end to end: its e2e build driven over mygo's automation socket.

    python3.13 scripts/desktop_web_e2e.py ROUND [--only NAME,...] [--app PATH]

It runs on an Apple silicon Mac, on x86_64 Linux in a Hyprland session, and on x86-64 Windows in a
desktop session. Build the app first, around a markitai that has `serve --supervised`:

    python3.13 scripts/desktop_web_build.py stage --archive markitai-v<version>-aarch64-apple-darwin.tar.gz
    python3.13 scripts/desktop_web_build.py build --e2e --platform darwin/arm64

and on Linux the same with x86_64-unknown-linux-gnu and linux/amd64, on Windows with
x86_64-pc-windows-msvc (a .zip) and windows/amd64.

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
  download     a blob download, and in the next launch a ticket download, land
               where the save panel answered, with the right content
  pdf          Download as > PDF... > Print... in the preview writes a %PDF file
               of A4 pages and the page receives afterprint; the bundled
               markitai reads the file as the page's text, without the
               workbench's sidebar
  port         a second launch serves the same origin, the first launch's log
               moved to serve.previous.log; with that port taken, a third
               launch takes another from 23600 to 23699 and remembers it
  quit-idle    SIGTERM to the app (on Windows, WM_CLOSE to its window): the app
               exits 0, serve drains, its PID is gone, its control file removed
  quit-uncounted
               a quit (SIGTERM, which mygo takes as Cmd-Q, on macOS; closing
               the window on Linux, where SIGTERM asks nothing; WM_CLOSE on
               Windows) while serve cannot answer (suspended): the app asks with
               no count; Cancel keeps it working; with nothing unfinished,
               Quit from the menu (on Linux and Windows, closing the window)
               then asks nothing
  quit-busy-when-done, quit-busy-now, quit-busy-then-now
               that quit while a URL conversion is held open by a local page: the
               question answers "Quit When Done" (the Stopping page offers Stop
               Now alone, the app waits, serve drains and the conversion
               finishes in it), "Stop and Quit" (serve stops at once with the
               conversion held; the next start says it was interrupted), or
               "Quit When Done" and then the Stopping page's Stop Now (serve
               drains, then stops at once; interrupted)
  crash        SIGKILL to the app (on Windows, TerminateProcess) while a
               conversion is held: its input closed,
               serve drains and keeps running the conversion; a launch
               meanwhile shows that the server stopped; once serve has
               finished and released the home, Try Again brings the workbench
               on the same origin, the conversion done
  home-busy    another `markitai serve` holds the home: the Failed page shows
               why, with Try Again and Show Log; once it stops, Try Again
               brings the workbench, its serve's output after the failed one's
               in the launch's serve.log
  history      the history's Back from the workbench to the Starting page it
               replaced: the app shows the workbench again, not a stale page
  look         the workbench's language and theme reach the app: web.json keeps
               the theme, and the next launch's pages take the language
               (navigator.language), as do the application menu's items on
               macOS, back to the system's when it follows it
  session-end  Linux only: SIGTERM, as a session that ends sends it, while a
               URL conversion is held: the app asks nothing, quits as Quit When
               Done does (the Stopping page), and the conversion finishes
  logout       Windows only: signing out while a URL conversion is held, as
               WM_QUERYENDSESSION and WM_ENDSESSION to the app's windows: no
               window holds the session up, the app asks nothing, serve has
               stopped at once by the time the app answers WM_ENDSESSION, after
               which Windows ends the process (the next start says interrupted)

Each scenario runs the app with its own HOME (on Linux with the XDG data,
config, cache and state directories in it, on Windows with APPDATA,
LOCALAPPDATA and TEMP), MARKITAI_HOME and app data in
.local/desktop-web-e2e/ROUND/<scenario>/ and leaves there result.json, the app's
output, screenshots, and the app's data and serve logs. ROUND/run.json names
the app, the bundled markitai, the staged archive and the commit. Save panels
and the busy-quit question answer through the e2e build's environment, which
also records what would open in the browser (desktop/web/e2e.go); the native panel
and dialog are checked by hand.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
import argparse
import base64
import ctypes
import datetime
import hashlib
import json
import os
import platform
import plistlib
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from typing import NamedTuple
import urllib.error
import urllib.request
import zipfile

from isolation import install_state_guard, isolated_env

ROOT = Path(__file__).resolve().parents[1]


class Build(NamedTuple):
    """The e2e build of a system the e2e runs on, and how the app runs there."""
    platform: str
    path: str  # from the checkout
    executable: str  # within path, as the bundled markitai is
    program: str
    arguments: tuple  # for English
    variables: dict
    # Variables naming where the app and its web view keep data, each made a directory in the sandbox's HOME.
    data: dict
    shell: str  # the login shell the app reads its environment from; "" where it reads none
    quit: str  # how a user quits with nothing unfinished: "menu", or closing the window with "hyprctl" or "close"


BUILDS = {
    ("Darwin", "arm64"): Build("darwin/arm64", "desktop/web/build/e2e/darwin-arm64/Markitai Web.app",
                               "Contents/MacOS/Markitai Web", "Contents/Resources/bin/markitai",
                               ("-AppleLanguages", "(en)"), {}, {}, "/bin/zsh", "menu"),
    # No application menu on Linux; its window's class is the app's name, dev.markitai.web.e2e (e2e.go).
    ("Linux", "x86_64"): Build("linux/amd64", "desktop/web/build/e2e/linux-amd64", "markitai-web", "bin/markitai", (),
                               {"LANGUAGE": "en"},
                               {"XDG_CONFIG_HOME": ".config", "XDG_DATA_HOME": ".local/share",
                                "XDG_CACHE_HOME": ".cache", "XDG_STATE_HOME": ".local/state"}, "/bin/bash", "hyprctl"),
    # The app takes Explorer's environment, as the process's own.
    ("Windows", "AMD64"): Build("windows/amd64", "desktop/web/build/e2e/windows-amd64", "Markitai Web.exe",
                                "bin/markitai.exe", (), {},
                                {"APPDATA": "AppData/Roaming", "LOCALAPPDATA": "AppData/Local",
                                 "TEMP": "AppData/Local/Temp", "TMP": "AppData/Local/Temp"}, "", "close"),
}
BUILD = BUILDS.get((platform.system(), platform.machine()))
# Its real path: serve refuses a MARKITAI_HOME reached through a symbolic link (output.allow_symlinks).
EVIDENCE = (ROOT / ".local/desktop-web-e2e").resolve()
TOKEN_KEY = "markitai.service-token"
READY_TIMEOUT = 90
# The page a held conversion fetches: enough text for a plain fetch to keep.
SLOW_PAGE = ("<!doctype html><html><head><title>Held page</title></head><body><h1>Held page</h1>"
             + "".join(f"<p>Paragraph {n} of a page the web e2e holds open until the app quits.</p>"
                       for n in range(1, 21))
             + "</body></html>").encode()
DROPPED = "web-e2e.html"
DROPPED_HTML = ("<!doctype html><html><head><title>Web e2e</title></head>"
                "<body><h1>Web e2e</h1><p>hello web</p></body></html>")
# Proxy settings would send the held page's fetch elsewhere.
PROXIES = {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}
LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))
# Where the app looks for a port first (desktop/web/port.go).
APP_PORTS = range(23600, 23700)
# What serve writes to serve.log (MARKITAI_LANG=en).
LISTENING = "Markitai server listening on http://127.0.0.1:"
STOPPED_BY_APP = "The app that started this server stops it through standard input."
DRAINING = "Server stopping: draining active conversions and saving history."
STOPPING_NOW = "Server stopping now: running conversions are abandoned"
# The e2e build's name (desktop/web/e2e.go), which its application menu shows.
APP_NAME = "Markitai Web E2E"
# A4 in PDF points; Export PDF lays out on it (desktop/web/window.go).
A4 = (595.28, 841.89)
# Words of the workbench's sidebar, which an export of the preview must not print.
SIDEBAR = ("Conversions", "Settings")
# The status pages' buttons (desktop/web/window.go).
RETRY, SHOW_LOG, STOP_NOW = "markitai-web:retry", "markitai-web:show-log", "markitai-web:stop-now"


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


class Posix:
    """Processes on macOS and Linux, through signals; the app quits on SIGTERM, as mygo takes it."""
    request = "SIGTERM"

    def alive(self, pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def program(self, pid):
        command = subprocess.run(["ps", "-o", "comm=", "-p", str(pid)], capture_output=True, text=True).stdout
        return Path(command.strip()).name

    def suspend(self, pid):
        os.kill(pid, signal.SIGSTOP)

    def resume(self, pid):
        os.kill(pid, signal.SIGCONT)

    def terminate(self, pid):
        os.kill(pid, signal.SIGTERM)

    def kill(self, pid):
        os.kill(pid, signal.SIGKILL)

    def request_quit(self, process):
        process.send_signal(signal.SIGTERM)

    def connect(self, path):
        attempt = socket.socket(socket.AF_UNIX)
        try:
            attempt.connect(path)
        except OSError:
            attempt.close()
            raise
        return attempt


class Windows:
    """Processes on Windows, through the Win32 API: there are no signals, and os.kill would end the
    process whatever it sends. The system asks an app to quit with WM_CLOSE to its windows, as taskkill
    without /F does; Python has no AF_UNIX there, so the automation socket is opened through Winsock."""
    request = "WM_CLOSE"
    QUERY, SUSPEND_RESUME, TERMINATE = 0x1000, 0x0800, 0x0001
    STILL_ACTIVE, ACCESS_DENIED = 259, 5
    WM_CLOSE, WM_SYSCOMMAND, SC_CLOSE, GW_OWNER = 0x0010, 0x0112, 0xF060, 4
    WM_QUERYENDSESSION, WM_ENDSESSION, ENDSESSION_LOGOFF, SMTO_BLOCK = 0x0011, 0x0016, 0x80000000, 0x0001
    AF_UNIX, INVALID_SOCKET = 1, 2 ** (8 * struct.calcsize("P")) - 1

    def __init__(self):
        from ctypes import wintypes
        self.kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.ntdll = ctypes.WinDLL("ntdll")
        self.ws2 = ctypes.WinDLL("ws2_32", use_last_error=True)
        self.kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel32.OpenProcess.restype = wintypes.HANDLE
        self.kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self.kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self.kernel32.QueryFullProcessImageNameW.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR,
                                                            ctypes.POINTER(wintypes.DWORD)]
        self.kernel32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.ntdll.NtSuspendProcess.argtypes = self.ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
        self.each_window = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        self.user32.EnumWindows.argtypes = [self.each_window, wintypes.LPARAM]
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.IsWindowVisible.argtypes = [wintypes.HWND]
        self.user32.GetWindow.argtypes = [wintypes.HWND, wintypes.UINT]
        self.user32.GetWindow.restype = wintypes.HWND
        self.user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        self.user32.SendMessageTimeoutW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM,
                                                    wintypes.UINT, wintypes.UINT, ctypes.POINTER(ctypes.c_size_t)]
        self.user32.SendMessageTimeoutW.restype = ctypes.c_ssize_t
        self.ws2.socket.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.ws2.socket.restype = ctypes.c_size_t
        self.ws2.connect.argtypes = [ctypes.c_size_t, ctypes.c_void_p, ctypes.c_int]
        self.ws2.closesocket.argtypes = [ctypes.c_size_t]
        self.dword = wintypes.DWORD

    def handle(self, pid, access):
        """A handle to pid with `access`, or None and the Win32 error."""
        handle = self.kernel32.OpenProcess(access, False, pid)
        return (handle, 0) if handle else (None, ctypes.get_last_error())

    def call(self, pid, access, act):
        handle, error = self.handle(pid, access)
        if handle is None:
            raise ctypes.WinError(error)
        try:
            return act(handle)
        finally:
            self.kernel32.CloseHandle(handle)

    def alive(self, pid):
        handle, error = self.handle(pid, self.QUERY)
        if handle is None:
            return error == self.ACCESS_DENIED
        try:
            code = self.dword()
            return bool(self.kernel32.GetExitCodeProcess(handle, ctypes.byref(code))) and code.value == self.STILL_ACTIVE
        finally:
            self.kernel32.CloseHandle(handle)

    def program(self, pid):
        def name(handle):
            buffer, size = ctypes.create_unicode_buffer(1024), self.dword(1024)
            return Path(buffer.value).stem if self.kernel32.QueryFullProcessImageNameW(
                handle, 0, buffer, ctypes.byref(size)) else ""
        try:
            return self.call(pid, self.QUERY, name)
        except OSError:
            return ""

    def suspend(self, pid):
        self.call(pid, self.SUSPEND_RESUME, self.ntdll.NtSuspendProcess)

    def resume(self, pid):
        self.call(pid, self.SUSPEND_RESUME, self.ntdll.NtResumeProcess)

    def terminate(self, pid):
        self.call(pid, self.TERMINATE, lambda handle: self.kernel32.TerminateProcess(handle, 1))

    kill = terminate

    def windows(self, pid, hidden=False):
        """pid's visible top-level windows that no other window owns: the app's, not mygo's hidden one;
        with hidden, all its top-level windows, mygo's among them."""
        found = []

        def each(window, _):
            owner = self.dword()
            self.user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
            if owner.value == pid and (hidden or self.user32.IsWindowVisible(window) and not self.user32.GetWindow(
                    window, self.GW_OWNER)):
                found.append(window)
            return True
        self.user32.EnumWindows(self.each_window(each), 0)
        if not found:
            raise Failure(f"no window of pid {pid}")
        return found

    def request_quit(self, process):
        for window in self.windows(process.pid):
            self.user32.PostMessageW(window, self.WM_CLOSE, 0, 0)

    def close(self, process):
        """What the window's close button and Alt+F4 send."""
        for window in self.windows(process.pid):
            self.user32.PostMessageW(window, self.WM_SYSCOMMAND, self.SC_CLOSE, 0)

    def end_session(self, process, timeout):
        """What signing out sends each top-level window of the process, as Windows sends it: whether
        the session may end, then that it ends. The answers to the question, None where unanswered."""
        def send(window, message, wparam):
            result = ctypes.c_size_t()
            sent = self.user32.SendMessageTimeoutW(window, message, wparam, self.ENDSESSION_LOGOFF, self.SMTO_BLOCK,
                                                   int(timeout * 1000), ctypes.byref(result))
            return result.value if sent else None
        windows = self.windows(process.pid, hidden=True)
        answers = [send(window, self.WM_QUERYENDSESSION, 0) for window in windows]
        for window in windows:
            send(window, self.WM_ENDSESSION, 1)
        return answers

    def connect(self, path):
        class Address(ctypes.Structure):
            _fields_ = [("family", ctypes.c_ushort), ("path", ctypes.c_char * 108)]
        handle = self.ws2.socket(self.AF_UNIX, socket.SOCK_STREAM, 0)
        if handle == self.INVALID_SOCKET:
            raise ctypes.WinError(ctypes.get_last_error())
        address = Address(self.AF_UNIX, os.fsencode(path))
        if self.ws2.connect(handle, ctypes.byref(address), ctypes.sizeof(address)):
            error = ctypes.get_last_error()
            self.ws2.closesocket(handle)
            raise ctypes.WinError(error)
        return socket.socket(self.AF_UNIX, socket.SOCK_STREAM, 0, fileno=handle)


SYSTEM = Windows() if os.name == "nt" else Posix()


def alive(pid):
    return SYSTEM.alive(pid)


def is_markitai(pid):
    return SYSTEM.program(pid) == "markitai"


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


def sandbox(home, markitai_home):
    """isolated_env's HOME and MARKITAI_HOME, with the build's data directories made in HOME."""
    environment = isolated_env(home, markitai_home)
    for name, relative in BUILD.data.items():
        directory = Path(home).absolute() / relative
        directory.mkdir(parents=True, exist_ok=True)
        environment[name] = str(directory)
    return environment


class Automation:
    """mygo's automation plugin: newline-delimited JSON calls over a Unix socket."""

    def __init__(self, path, timeout=30):
        self.socket = wait_for("automation socket", lambda: SYSTEM.connect(path), timeout, 0.1)
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

    def __init__(self, scenario, index, english=True, **environment):
        """english: the arguments that make the app English whatever it chose before (BUILD)."""
        self.scenario = scenario
        self.index = index
        executable = scenario.run.app / BUILD.executable
        socket_name = f"a{index}.sock"
        self.output = open(scenario.dir / f"app-{index}.log", "wb")
        self.process = subprocess.Popen(
            [str(executable), *(BUILD.arguments if english else ())], cwd=scenario.dir, stdin=subprocess.DEVNULL,
            stdout=self.output, stderr=subprocess.STDOUT, start_new_session=True,
            env=scenario.environment(MYGO_AUTOMATION=socket_name, MARKITAI_WEB_E2E=str(scenario.app_data),
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

    def request_quit(self):
        """The system's request to quit: SIGTERM, which mygo takes as Quit (as Cmd-Q), or WM_CLOSE."""
        SYSTEM.request_quit(self.process)

    def ask_quit(self):
        """A quit that asks about unfinished conversions, without waiting: SIGTERM on macOS, which
        mygo takes as Cmd-Q; on Linux, where SIGTERM is the session's end and asks nothing (the
        app's session_linux.go), closing the window; WM_CLOSE on Windows."""
        if BUILD.quit == "hyprctl":
            self.close_window()
        else:
            SYSTEM.request_quit(self.process)

    def close_window(self):
        """Hyprland's close of the app's window, as Super+W does."""
        closed = subprocess.run(["hyprctl", "dispatch",
                                 f'hl.dsp.window.close({{ window = "pid:{self.process.pid}" }})'],
                                capture_output=True, text=True, timeout=10)
        if closed.returncode or closed.stdout.strip() != "ok":
            raise Failure(f"hyprctl could not close the window: {closed.stdout}{closed.stderr}".strip())

    def exited(self, timeout):
        try:
            return self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            raise Failure(f"the app (pid {self.process.pid}) still runs {timeout} s later") from None

    def quit(self):
        """The system's request to quit, and the app's exit status."""
        self.request_quit()
        code = self.exited(30)
        self.close()
        return code

    def quit_as_user(self):
        """Quit as a user does, and the app's exit status: after one SIGTERM, mygo ends the app at
        once on the next. Quit from the application menu, as Cmd-Q does; on Linux, which has no
        application menu, the compositor closes the window (Hyprland's close, as Super+W does); on
        Windows the window's close command, as its close button sends."""
        try:
            if BUILD.quit == "menu":
                self.ui.call("menu", window=self.ui.window(), path=[APP_NAME, f"Quit {APP_NAME}"])
            elif BUILD.quit == "close":
                SYSTEM.close(self.process)
            else:
                self.close_window()
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
    paths asked of it, each with the headers that say what kind of load asked."""

    KINDS = ("Sec-Purpose", "Purpose", "Sec-Fetch-Mode", "Sec-Fetch-Dest", "Sec-Fetch-User")

    def __init__(self):
        self.asked = []
        site = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                kind = ", ".join(f"{name}: {self.headers[name]}" for name in OtherSite.KINDS if self.headers[name])
                site.asked.append(f"{self.path} ({kind})" if kind else self.path)
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
        self.base = sandbox(self.home, self.markitai_home)
        # A login shell in an empty HOME would start zsh's new-user setup.
        (self.home / ".zshrc").write_text("# web e2e\n", encoding="utf-8")
        self.checks, self.notes, self.apps, self.others, self.sites = [], [], [], [], []
        self.serve_pids = set()

    def environment(self, **extra):
        """The sandbox's environment as Finder would start the app: no terminal, so the app
        reads a login shell's; nothing of the developer's markitai, mygo or proxies."""
        environment = {key: value for key, value in self.base.items()
                       if not key.startswith(("MARKITAI_", "MYGO_")) and key.lower() not in PROXIES
                       and key not in {"TERM", "TERM_PROGRAM", "ZDOTDIR"}}
        environment.update(MARKITAI_HOME=str(self.markitai_home), LANG="en_US.UTF-8", MARKITAI_LANG="en",
                           **BUILD.variables, **extra)
        if BUILD.shell:
            environment["SHELL"] = BUILD.shell
        return environment

    def launch(self, english=True, **environment):
        app = App(self, len(self.apps) + 1, english, **environment)
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
        for stale in self.dir.glob("*.sock"):
            stale.unlink(missing_ok=True)  # a killed app's: on Windows a file that other programs cannot stat
        self.control()
        for pid in self.serve_pids:
            if alive(pid) and is_markitai(pid):
                print(f"  stopping serve {pid}, still running after the scenario")
                SYSTEM.terminate(pid)
                try:
                    wait_for(f"serve {pid} to end", lambda pid=pid: not alive(pid), 10)
                except Failure:
                    SYSTEM.kill(pid)

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
    scenario.require("the conversion's Markdown", status == 200 and "# Web e2e" in markdown
                     and "hello web" in markdown, f"{status} {markdown[:200]!r}")
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
    at = app.page()["url"]
    s.check("a link to another site the user follows opens in the browser; the window stays on the workbench",
            opened == followed + "\n" and at.startswith(origin) and not elsewhere.asked,
            f"browser: {opened!r}, the window at {at}, the site was asked for {elsewhere.asked}")
    # Past 1 s: WebKit counts a timer set within 1 s of a gesture (the eval) as that gesture.
    app.ui.eval(f"""setTimeout(() => {{ location.href = {json.dumps(elsewhere.origin + "/y")}; }}, 1500);
        return true;""")
    time.sleep(4)
    opened = external.read_text() if external.is_file() else ""
    at = app.page()["url"]
    s.check("a page's own navigation elsewhere goes nowhere, not to the browser either",
            opened == followed + "\n" and at.startswith(origin) and not elsewhere.asked,
            f"browser: {opened!r}, the window at {at}, the site was asked for {elsewhere.asked}")
    pid = s.control()
    app.ui.eval("""const a = document.createElement("a");
        a.href = "markitai-web:stop-now"; a.target = "_blank";
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
        a.download = "web-e2e-blob.md";
        document.body.append(a); a.click(); a.remove();
        return r.status;
    """)
    s.require("the page read the result", status == 200, status)
    saved = s.saved(".md")
    s.check("the blob download is saved where the panel answered, with the Markdown",
            saved.read_text(encoding="utf-8") == markdown, saved.name)
    # WebView2, as Chrome, asks before a page's second download with no input of the user's between
    # (multiple automatic downloads). The workbench downloads once per click; a script run by the
    # driver is no click, so the ticket download has a launch of its own.
    app.quit()
    app = s.launch()
    app.ready()
    status = app.ui.eval(AUTH + f"""
        const r = await fetch("/api/download-tickets", {{method: "POST",
          headers: {{...H, "Content-Type": "application/json"}},
          body: JSON.stringify({{path: "/api/jobs/{job}/items/{item}/archive"}})}});
        if (!r.ok) return r.status;
        const a = document.createElement("a");
        a.href = (await r.json()).url;
        a.download = "web-e2e-ticket.zip";
        document.body.append(a); a.click(); a.remove();
        return r.status;
    """)
    s.require("a download ticket", status == 201, status)
    archive = s.saved(".zip")
    with zipfile.ZipFile(archive) as bundle:
        texts = [bundle.read(name).decode("utf-8", "replace") for name in bundle.namelist() if name.endswith(".md")]
    s.check("the ticket download is saved, an archive holding the Markdown",
            any("hello web" in text for text in texts), f"{archive.name}: {len(texts)} Markdown file(s)")
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
            document.title = "web-e2e-print"; window.print(); return "window.print";""")
    saved = s.saved(".pdf")
    data = saved.read_bytes()
    s.check(f"{step} saved a PDF", data.startswith(b"%PDF"), f"{saved.name}, {len(data)} bytes")
    boxes = [(float(x1) - float(x0), float(y1) - float(y0)) for x0, y0, x1, y1 in re.findall(
        rb"/MediaBox\s*\[\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*\]", data)]
    s.check("its pages are A4", boxes and all(abs(w - A4[0]) < 1 and abs(h - A4[1]) < 1 for w, h in boxes), boxes)
    after = eventually("afterprint", lambda: app.ui.eval("return window.__e2eAfterprint"), 10)
    restored = eventually("the page's restore", lambda: app.ui.eval(
        "return !document.body.classList.contains('is-printing')"), 5)
    s.check("the page received afterprint and restored itself", after and restored, f"afterprint x{after or 0}")
    app.quit()
    read = subprocess.run([str(s.run.app / BUILD.program), str(saved)], capture_output=True, text=True, timeout=120,
                          cwd=s.dir, stdin=subprocess.DEVNULL, env=s.environment(MARKITAI_NO_UPDATE_CHECK="1"))
    (s.dir / "saved-pdf.md").write_text(read.stdout + read.stderr, encoding="utf-8")
    s.check("the bundled markitai reads the PDF as the page's text, without the workbench's sidebar",
            read.returncode == 0 and "Web e2e" in read.stdout and "hello web" in read.stdout
            and not any(word in read.stdout for word in SIDEBAR),
            f"exit {read.returncode}; sidebar words: {[word for word in SIDEBAR if word in read.stdout]}")


def remembered_port(s):
    return json.loads((s.app_data / "data/web.json").read_text(encoding="utf-8")).get("port")


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
        # The port's last connections may wait out TIME_WAIT; listening still holds it. Windows binds
        # past TIME_WAIT by itself, and its SO_REUSEADDR would let another listener in too.
        if os.name != "nt":
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
    s.check(f"the app quits with 0 on {SYSTEM.request}", app.quit() == 0)
    s.check("serve ended and removed its control file", s.serve_ended(pid, 10), pid)
    s.check("serve's log ends with its drain, as a server that exits 0 leaves it", s.drained(),
            s.log().strip().splitlines()[-1:])
    s.check("with nothing unfinished, the app asked nothing", not (s.app_data / "asked").exists())


def quit_uncounted(s):
    app = s.launch(MARKITAI_WEB_E2E_QUIT="cancel")
    origin = app.ready()
    pid = s.control()
    s.require("serve wrote its control file", pid is not None and alive(pid), pid)
    asked = s.app_data / "asked"
    SYSTEM.suspend(pid)
    try:
        app.ask_quit()
        answer = eventually("the quit question", lambda: asked.is_file() and asked.read_text(), 10)
    finally:
        SYSTEM.resume(pid)
    s.check("with serve unable to count, the app asks anyway, with no count", answer == "unknown", answer)
    time.sleep(1)
    s.check("Cancel keeps the app, serve and the workbench",
            app.process.poll() is None and alive(pid) and app.page()["url"].startswith(origin))
    asked.unlink(missing_ok=True)
    s.check("with nothing unfinished, " + ("a quit from the menu" if BUILD.quit == "menu" else "closing the window")
            + " asks nothing and the app exits 0", app.quit_as_user() == 0 and not asked.exists())
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
    app = s.launch(MARKITAI_WEB_E2E_QUIT=choice)
    origin = app.ready()
    held, job = hold_a_conversion(s, app, origin, app.token())
    pid = s.control()
    app.ask_quit()
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
    app.process.kill()
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


def history_back(s):
    """W2: the window's history holds the Starting page the workbench replaced; Back to it shows
    the workbench again (the app repaints what it shows), never the stale page."""
    app = s.launch()
    origin = app.ready()
    token = app.token()
    app.ui.eval("setTimeout(() => history.back(), 100); return true;")
    time.sleep(3)
    state = wait_for("the workbench again", lambda: (state := app.page())["url"].startswith(origin)
                     and state["phase"] is None and state["sidebar"] and state, 15)
    app.capture("after-back")
    s.check("Back from the workbench shows the workbench again, with the token", bool(state) and state["token"],
            state)
    s.check("the same server serves it", app.token() == token)
    app.ui.eval("setTimeout(() => history.forward(), 100); return true;")
    time.sleep(2)
    state = app.page()
    s.check("Forward leaves the workbench in place", state["url"].startswith(origin) and state["phase"] is None,
            state["url"])
    s.check("the app quits with 0", app.quit() == 0)


def look(s):
    """Decision 2: the workbench's own language and theme become the app's."""
    if BUILD.quit != "menu":
        return look_here(s)
    # On macOS mygo keeps the app's language in the app's own defaults, through cfprefsd, which takes
    # neither HOME nor CFFIXED_USER_HOME: the account's defaults of the e2e build's identifier, apart
    # from the installed app's. The scenario leaves them as it found them: without a language.
    import pwd  # macOS alone comes here
    domain = plistlib.loads((s.run.app / "Contents/Info.plist").read_bytes())["CFBundleIdentifier"]
    forget = lambda: subprocess.run(["defaults", "delete", domain, "AppleLanguages"], capture_output=True)
    forget()
    try:
        look_here(s)
    finally:
        forget()
        # Deleting the last key leaves an empty property list, which the scenario made, in the
        # account's home: not HOME, which may be a sandbox's.
        plist = Path(pwd.getpwuid(os.getuid()).pw_dir) / "Library/Preferences" / f"{domain}.plist"
        try:
            if plistlib.loads(plist.read_bytes()) == {}:
                plist.unlink()
        except (OSError, ValueError, plistlib.InvalidFileException):
            pass
        s.check("the e2e build's defaults keep no language after the scenario",
                "AppleLanguages" not in subprocess.run(["defaults", "read", domain], capture_output=True,
                                                       text=True).stdout, "no file" if not plist.exists() else plist)


def look_here(s):
    memory = s.app_data / "data" / "web.json"

    def kept_theme():
        try:
            return json.loads(memory.read_text(encoding="utf-8")).get("theme")
        except (OSError, ValueError):
            return None

    app = s.launch()
    app.ready()
    app.ui.eval('localStorage.setItem("markitai.lang", "zh"); localStorage.setItem("markitai.theme", "dark"); '
                'return true;')
    s.check("web.json keeps the workbench's dark theme", eventually("the theme kept", lambda: kept_theme() == "dark",
                                                                   10), kept_theme())
    s.check("the app quits with 0", app.quit() == 0)
    # The e2e build clears the web view's data at each launch: the next workbench has no language of its
    # own and takes the page's, which is the app's.
    again = s.launch(english=False)
    again.ready()
    language = again.ui.eval("return navigator.language")
    s.check("the next launch's pages take the language the workbench chose", str(language).lower().startswith("zh"),
            language)
    shown = again.ui.eval("return document.documentElement.lang")
    s.check("and its workbench shows Chinese", str(shown).lower().startswith("zh"), shown)
    if MENUS:
        s.check("and the app's menus are in Chinese", chosen_in_menu(again, ["显示", "实际大小"]))
    s.check("a workbench that follows the system theme leaves web.json without one",
            eventually("the theme dropped", lambda: kept_theme() is None, 10), kept_theme())
    again.ui.eval('localStorage.setItem("markitai.lang", "en"); return true;')
    time.sleep(1)
    s.check("the app quits with 0", again.quit() == 0)
    third = s.launch(english=False)
    third.ready()
    language = third.ui.eval("return navigator.language")
    s.check("English chosen in the workbench is the next launch's language", str(language).lower().startswith("en"),
            language)
    if MENUS:
        s.check("and the app's menus are in English", chosen_in_menu(third, ["View", "Actual Size"]))
    s.check("the app quits with 0", third.quit() == 0)


# Only the Mac's app has an application menu.
MENUS = BUILD is not None and BUILD.platform.startswith("darwin")


def chosen_in_menu(app, path):
    """Whether the application menu has the item at path, by its titles: chosen as the menu bar does."""
    try:
        app.ui.call("menu", window=app.ui.window(), path=path)
        return True
    except Exception:  # noqa: BLE001 - no such item: the menu speaks another language
        return False


def session_end(s):
    """L5: SIGTERM on Linux, as a session that ends sends it before SIGKILL, asks nothing."""
    app = s.launch(MARKITAI_WEB_E2E_QUIT="now")
    origin = app.ready()
    held, job = hold_a_conversion(s, app, origin, app.token())
    pid = s.control()
    SYSTEM.terminate(app.process.pid)
    page = app.phase("stopping", 15)
    app.capture("stopping")
    s.check("SIGTERM asks nothing", not (s.app_data / "asked").exists())
    s.check("the app quits as Quit When Done: the Stopping page offers Stop Now alone", page["actions"] == [STOP_NOW],
            page["actions"])
    s.check("serve drains, its conversion held",
            eventually("serve's drain", lambda: DRAINING in s.log(), 5) and app.process.poll() is None and alive(pid))
    held.release.set()
    s.check("the app quits with 0 once serve has drained", app.exited(90) == 0)
    s.check("serve ended and removed its control file", s.serve_ended(pid, 15), pid)
    again = s.launch()
    s.check("the next launch is ready on the same origin", again.ready() == origin)
    s.require("the held conversion finished", finished(origin, again.token(), job=job, timeout=10), job)
    again.quit()


def logout(s):
    app = s.launch()
    origin = app.ready()
    held, job = hold_a_conversion(s, app, origin, app.token())
    pid = s.control()
    answers = SYSTEM.end_session(app.process, 20)
    s.check("every window of the app lets the session end", bool(answers) and all(answers), answers)
    s.check("by the time the app has answered that the session ends, serve has ended, its conversion held",
            not alive(pid), pid)
    s.check("the app asked nothing", not (s.app_data / "asked").exists())
    # Windows ends the process once the session ends, as it then does to every process of it.
    s.note(f"the app's process {'still ran' if app.process.poll() is None else 'had exited'}; ended as the "
           "session ends")
    if app.process.poll() is None:
        SYSTEM.terminate(app.process.pid)
    app.exited(10)
    log = s.log()
    s.check("serve's log says it stopped at once, not that it drained", STOPPING_NOW in log and DRAINING not in log,
            log.strip().splitlines()[-1:])
    interrupted(s, origin, job)


def home_busy(s):
    program = s.run.app / BUILD.program
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
    "home-busy": home_busy, "history": history_back, "look": look,
}
if os.name == "nt":
    SCENARIOS["logout"] = logout
if BUILD and BUILD.quit == "hyprctl":
    SCENARIOS["session-end"] = session_end


class Run:
    def __init__(self, name, app):
        self.dir, self.app = EVIDENCE / name, app


def identity(run, guard):
    """What ran: the app, the markitai inside it, how it was staged, the checkout."""
    app = run.app
    program = app / BUILD.program
    plist = app / "Contents/Info.plist"
    info = plistlib.loads(plist.read_bytes()) if plist.exists() else {}
    stamp = ROOT / "desktop/web/resources/.stage.json"
    git = lambda *arguments: subprocess.run(["git", "--no-optional-locks", *arguments], cwd=ROOT,
                                            capture_output=True, text=True).stdout.strip()
    with_home = run.dir / "version-home"
    version = subprocess.run([str(program), "--version"], capture_output=True, text=True, timeout=60,
                             env=sandbox(with_home / "home", with_home / "markitai-home")).stdout.strip()
    return {
        "command": [Path(sys.executable).name, *sys.argv],
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "system": f"{platform.system()} {platform.release()} {platform.machine()}",
        "app": {"path": app.relative_to(ROOT).as_posix() if app.is_relative_to(ROOT) else app.name,
                "bundle": info.get("CFBundleIdentifier"), "version": info.get("CFBundleShortVersionString"),
                "sha256": hashlib.sha256((app / BUILD.executable).read_bytes()).hexdigest()},
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
    parser.add_argument("round", help="evidence goes to .local/desktop-web-e2e/ROUND/")
    parser.add_argument("--only", help="comma-separated scenarios: " + ", ".join(SCENARIOS))
    parser.add_argument("--app", type=Path, help="the e2e build: its Markitai Web.app, or its directory on Linux "
                                                     "and Windows")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.round):
        parser.error("ROUND is a plain name: letters, digits, '.', '_' or '-'")
    names = args.only.split(",") if args.only else list(SCENARIOS)
    unknown = [name for name in names if name not in SCENARIOS]
    if unknown:
        parser.error(f"unknown scenarios: {', '.join(unknown)}")
    if BUILD is None:
        raise SystemExit("The Markitai Web e2e runs the darwin/arm64 build on an Apple silicon Mac, the linux/amd64 "
                         "build on x86_64 Linux, or the windows/amd64 build on x86-64 Windows")
    app = (args.app or ROOT / BUILD.path).resolve()
    if not os.access(app / BUILD.executable, os.X_OK):
        raise SystemExit(f"No e2e build at {app}; run desktop_web_build.py stage, then build --e2e --platform "
                         f"{BUILD.platform}")
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
          f"{run.dir.relative_to(ROOT) if run.dir.is_relative_to(ROOT) else run.dir}"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
