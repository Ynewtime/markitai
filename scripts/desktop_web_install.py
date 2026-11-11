#!/usr/bin/env python3
"""Markitai Web as it is installed: what a release build's package must do once a user has it.

  run --package PATH [--evidence DIR]
        Install the package a release build made for this system, start the
        installed app in a HOME and MARKITAI_HOME of the check's own, and see
        what a user gets: the bundled `markitai serve` listening, a window on
        the screen, a file converted (as convert does), the app quitting with
        0 and its server drained. Then remove the app. Each check is a line,
        "ok" or "FAIL", and any FAIL makes the exit status 1. --evidence
        keeps the lines (install.log), the app's output (installed-app.log),
        its serve.log (installed-serve.log) and the screen once the window is
        up (installed-ready.png).
        macOS: PATH is the disk image. Markitai Web.app goes from it to
        /Applications, as dragging it there copies it; the check does not
        start when one is there or runs already. The image was never
        downloaded, so Gatekeeper is not asked about an app that is not
        signed for other Macs.
  convert --pid PID --origin http://127.0.0.1:PORT
        Convert a file with the `markitai serve` an installed Markitai Web
        started (PID is that server's, the app's child). The release build
        has no automation socket and keeps the server's token to itself and
        its child, so the token comes from where the app put it, the child's
        environment (MARKITAI_SERVE_TOKEN), read as the same user. The file,
        a small HTML page, goes to POST /api/jobs as the workbench sends one;
        the finished item's Markdown must be the page's heading and text.

Nothing here builds: scripts/desktop_web_build.py builds the app and its
packages. .github/workflows/desktop-web.yml runs `run` on macOS; on Windows
its own step installs the app and calls `convert`.
"""
from pathlib import Path
from typing import Callable, NamedTuple
import argparse
import errno
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid

# The app's name (desktop/web/mygo.json): its bundle, and its folders for logs and data.
APP_NAME = "Markitai Web"
# Where a user drags the app from its disk image.
APPLICATIONS = Path("/Applications")
# What serve writes to serve.log (MARKITAI_LANG=en).
LISTENING = re.compile(r"Markitai server listening on (http://127\.0\.0\.1:\d+)")
DRAINING = "Server stopping: draining active conversions"
# How long the app has to start its server, to show its window and keep its port, and to quit; and
# how long the window is left to paint the workbench before the screen is kept.
START_SECONDS, WINDOW_SECONDS, QUIT_SECONDS, PAINT_SECONDS = 90, 30, 30, 5
# The variable the app gives its server the token in (desktop/web/env.go).
TOKEN_VARIABLE = "MARKITAI_SERVE_TOKEN"
# The file an installed app's server converts, and what its Markdown must hold.
PAGE_NAME = "installed.html"
PAGE = ("<!doctype html><html><head><title>Installed</title></head>"
        "<body><h1>Installed app</h1><p>hello installed</p></body></html>").encode()
EXPECTED = ("# Installed app", "hello installed")
# A proxy must not carry a request for this computer.
LOOPBACK = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Failure(Exception):
    """What an installed app did not do."""


def _strings(block, encoding):
    """A process's NAME=value strings from `block`, NUL-separated and ended by an empty one."""
    environment = {}
    # Decoded first: in UTF-16 a NUL's two bytes also end and begin its neighbours.
    for entry in block.decode(encoding, "replace").split("\0"):
        if not entry:
            break
        name, found, value = entry.partition("=")
        # Windows keeps each drive's directory as "=C:=C:\...".
        if found and name:
            environment[name] = value
    return environment


def _linux_environment(pid):
    return _strings(Path(f"/proc/{pid}/environ").read_bytes(), "utf-8")


def _darwin_process(pid):
    """(program, arguments, environment) of process `pid`, from KERN_PROCARGS2: the argument count,
    the program's path, padding, the arguments, the environment."""
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    def sysctl(names, buffer, size):
        mib = (ctypes.c_int * len(names))(*names)
        if libc.sysctl(mib, len(names), buffer, ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), f"sysctl {names}: {os.strerror(ctypes.get_errno())}")

    limit = ctypes.c_int()
    sysctl((1, 8), ctypes.byref(limit), ctypes.c_size_t(ctypes.sizeof(limit)))  # CTL_KERN, KERN_ARGMAX
    buffer = ctypes.create_string_buffer(limit.value)
    deadline = time.monotonic() + 2
    while True:
        size = ctypes.c_size_t(limit.value)
        try:
            sysctl((1, 49, pid), buffer, size)  # CTL_KERN, KERN_PROCARGS2
            break
        except OSError as error:
            # EIO while the process starts a program, a fraction of a millisecond: a launcher, such as
            # a framework's python3, becomes the program it starts.
            if error.errno != errno.EIO or time.monotonic() > deadline:
                raise
            time.sleep(0.01)
    data = buffer.raw[:size.value]
    try:
        count = struct.unpack("i", data[:4])[0]
        start = data.index(b"\0", 4)
        program = data[4:start].decode("utf-8", "replace")
        while start < len(data) and data[start] == 0:
            start += 1
        arguments = []
        for _ in range(count):
            end = data.index(b"\0", start)
            arguments.append(data[start:end].decode("utf-8", "replace"))
            start = end + 1
    except (ValueError, struct.error) as error:
        raise OSError(f"KERN_PROCARGS2 of process {pid} is not laid out as expected: {error}") from error
    return program, arguments, _strings(data[start:], "utf-8")


def _darwin_environment(pid):
    return _darwin_process(pid)[2]


def _windows_environment(pid):
    """The environment block the process's PEB points to; this Python and the process are 64-bit."""
    import ctypes
    from ctypes import wintypes

    if struct.calcsize("P") != 8:
        raise OSError("reading a process's environment needs a 64-bit Python")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    ntdll = ctypes.WinDLL("ntdll")
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.ReadProcessMemory.restype = wintypes.BOOL
    kernel32.ReadProcessMemory.argtypes = (wintypes.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t,
                                           ctypes.POINTER(ctypes.c_size_t))
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    ntdll.NtQueryInformationProcess.restype = ctypes.c_long
    ntdll.NtQueryInformationProcess.argtypes = (wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong,
                                                ctypes.POINTER(ctypes.c_ulong))

    class BasicInformation(ctypes.Structure):
        _fields_ = [("exit_status", ctypes.c_void_p), ("peb", ctypes.c_void_p), ("affinity", ctypes.c_void_p),
                    ("priority", ctypes.c_void_p), ("pid", ctypes.c_void_p), ("parent", ctypes.c_void_p)]

    handle = kernel32.OpenProcess(0x0400 | 0x0010, False, pid)  # PROCESS_QUERY_INFORMATION | PROCESS_VM_READ
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        def read(address, size):
            buffer, done = ctypes.create_string_buffer(size), ctypes.c_size_t()
            if not kernel32.ReadProcessMemory(handle, address, buffer, size, ctypes.byref(done)) or done.value != size:
                raise ctypes.WinError(ctypes.get_last_error())
            return buffer.raw

        def pointer(address):
            return struct.unpack("<Q", read(address, 8))[0]

        basic = BasicInformation()
        status = ntdll.NtQueryInformationProcess(handle, 0, ctypes.byref(basic), ctypes.sizeof(basic), None)
        if status:
            raise OSError(f"NtQueryInformationProcess answered {status & 0xFFFFFFFF:#010x}")
        # PEB.ProcessParameters, then RTL_USER_PROCESS_PARAMETERS.Environment and .EnvironmentSize.
        parameters = pointer(basic.peb + 0x20)
        block = read(pointer(parameters + 0x80), pointer(parameters + 0x3F0))
    finally:
        kernel32.CloseHandle(handle)
    return _strings(block, "utf-16-le")


def process_environment(pid):
    """The environment process `pid`, one of this user's, was started with."""
    if sys.platform == "win32":
        return _windows_environment(pid)
    if sys.platform == "darwin":
        return _darwin_environment(pid)
    return _linux_environment(pid)


def serve_token(pid):
    try:
        environment = process_environment(pid)
    except OSError as error:
        raise Failure(f"process {pid}'s environment cannot be read: {error}") from error
    token = environment.get(TOKEN_VARIABLE)
    if not token:
        raise Failure(f"process {pid} has no {TOKEN_VARIABLE}: it is not a server Markitai Web started")
    return token


def multipart(files):
    """(Content-Type, body) of POST /api/jobs for `files`, (name, bytes) pairs."""
    boundary = "markitai-" + uuid.uuid4().hex
    parts = [(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="{name}"\r\n'
              "Content-Type: application/octet-stream\r\n\r\n").encode() + data + b"\r\n" for name, data in files]
    return f"multipart/form-data; boundary={boundary}", b"".join(parts) + f"--{boundary}--\r\n".encode()


def call(origin, token, path, body=None, content_type=None):
    """(status, body) of a request to the server at `origin` with its token."""
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    request = urllib.request.Request(origin + path, data=body, headers=headers)
    try:
        with LOOPBACK.open(request, timeout=30) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except OSError as error:
        raise Failure(f"{origin}{path}: {error}") from error


def convert(origin, token, timeout=120):
    """PAGE converted by the server at `origin`; its Markdown."""
    content_type, body = multipart([(PAGE_NAME, PAGE)])
    status, answer = call(origin, token, "/api/jobs", body, content_type)
    if status != 201:
        raise Failure(f"POST /api/jobs answered {status}: {answer[:300]!r}")
    job = json.loads(answer)["job_id"]
    deadline = time.monotonic() + timeout
    item = None
    while item is None:
        status, answer = call(origin, token, "/api/history?include=items")
        if status != 200:
            raise Failure(f"GET /api/history answered {status}: {answer[:300]!r}")
        # The history lists a job once it has finished.
        for entry in json.loads(answer):
            for listed in entry.get("items") or []:
                if entry.get("job_id") != job:
                    continue
                if listed.get("status") == "error":
                    raise Failure(f"{PAGE_NAME} failed to convert: {json.dumps(listed)[:600]}")
                if listed.get("status") == "done":
                    item = listed["item_id"]
        if item is None:
            if time.monotonic() > deadline:
                raise Failure(f"job {job} did not finish in {timeout} s")
            time.sleep(0.5)
    status, answer = call(origin, token, f"/api/jobs/{job}/items/{item}/result")
    markdown = json.loads(answer).get("markdown", "") if status == 200 else ""
    if status != 200 or not all(text in markdown for text in EXPECTED):
        raise Failure(f"{PAGE_NAME}'s Markdown is not the page's: {status} {markdown[:300]!r}")
    return markdown


def converted(pid, origin):
    """What the server `pid`, an installed app's, at `origin` made of PAGE, in a line."""
    try:
        markdown = convert(origin.rstrip("/"), serve_token(pid))
    except (ValueError, LookupError, TypeError, AttributeError) as error:
        raise Failure(f"{origin} did not answer as the workbench's API does: {error!r}") from error
    lines = " / ".join(line.strip() for line in markdown.splitlines() if line.strip())
    return f"process {pid} at {origin} converted {PAGE_NAME}: {lines}"


class Report:
    """A run's lines, on standard output and in the evidence's install.log, and how many checks failed."""

    def __init__(self, evidence=None):
        self.failures = 0
        self.lines = None
        if evidence is not None:
            evidence.mkdir(parents=True, exist_ok=True)
            self.lines = open(evidence / "install.log", "w", encoding="utf-8")

    def log(self, text):
        print(text, flush=True)
        if self.lines is not None:
            self.lines.write(text + "\n")
            self.lines.flush()

    def check(self, ok, what):
        """A check's line; whether it held."""
        self.log(("ok   " if ok else "FAIL ") + what)
        self.failures += not ok
        return bool(ok)

    def close(self):
        if self.lines is not None:
            self.lines.close()


def command(arguments, timeout=120):
    """A command's standard output. One that fails, or is not done in `timeout` s, is a Failure with what it
    printed."""
    arguments = [str(argument) for argument in arguments]
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, errors="replace", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Failure(f"{' '.join(arguments)}: {error}") from error
    if result.returncode:
        printed = (result.stdout + result.stderr).strip()[-600:]
        raise Failure(f"{' '.join(arguments)} exited {result.returncode}: {printed}")
    return result.stdout


def wait_for(probe, seconds, process=None):
    """probe()'s first true value within `seconds`; None when there is none, or `process` ends before one."""
    deadline = time.monotonic() + seconds
    while True:
        found = probe()
        if found:
            return found
        if time.monotonic() > deadline or (process is not None and process.poll() is not None):
            return None
        time.sleep(0.25)


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def process_command(pid):
    """(program, arguments) process `pid`, one of this user's, was started with."""
    if sys.platform == "darwin":
        return _darwin_process(pid)[:2]
    raise OSError(f"a process's command is not read on {sys.platform}")


def started_by(pid):
    """(pid, program, arguments) of each process that `pid` started and that still runs."""
    found = []
    for line in command(["ps", "-axo", "pid=,ppid="]).splitlines():
        child, parent = (int(number) for number in line.split())
        if parent == pid:
            try:
                found.append((child, *process_command(child)))
            except OSError:
                pass  # it ended meanwhile, as the login shell the app reads its environment from does
    return found


def listening(log):
    """Where serve.log says the server listens, http://127.0.0.1:PORT; None until it says so."""
    try:
        found = LISTENING.search(log.read_text(encoding="utf-8", errors="replace"))
    except FileNotFoundError:
        return None
    return found.group(1) if found else None


def user_environment(home, markitai_home, shell):
    """This process's environment for an installed app, as a desktop starts one: a HOME and MARKITAI_HOME
    of the check's own, the login shell the app reads its environment from, serve's messages in English,
    and nothing of a terminal's or of a developer's markitai and mygo."""
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("MARKITAI_", "MYGO_")) and name not in {"TERM", "TERM_PROGRAM", "ZDOTDIR"}}
    environment.update(HOME=str(home), MARKITAI_HOME=str(markitai_home), MARKITAI_LANG="en", SHELL=shell)
    return environment


def _darwin_windows(pid):
    """(width, height) of each of process `pid`'s windows on the screen, as the window server lists them."""
    import ctypes

    graphics = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
    foundation = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")

    class Rect(ctypes.Structure):
        _fields_ = [(name, ctypes.c_double) for name in ("x", "y", "width", "height")]

    graphics.CGWindowListCopyWindowInfo.restype = ctypes.c_void_p
    graphics.CGWindowListCopyWindowInfo.argtypes = (ctypes.c_uint32, ctypes.c_uint32)
    graphics.CGRectMakeWithDictionaryRepresentation.restype = ctypes.c_bool
    graphics.CGRectMakeWithDictionaryRepresentation.argtypes = (ctypes.c_void_p, ctypes.POINTER(Rect))
    foundation.CFArrayGetCount.restype = ctypes.c_long
    foundation.CFArrayGetCount.argtypes = (ctypes.c_void_p,)
    foundation.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    foundation.CFArrayGetValueAtIndex.argtypes = (ctypes.c_void_p, ctypes.c_long)
    foundation.CFDictionaryGetValue.restype = ctypes.c_void_p
    foundation.CFDictionaryGetValue.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    foundation.CFNumberGetValue.restype = ctypes.c_bool
    foundation.CFNumberGetValue.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p)
    foundation.CFRelease.argtypes = (ctypes.c_void_p,)

    def value(entry, key):
        return foundation.CFDictionaryGetValue(entry, ctypes.c_void_p.in_dll(graphics, key))

    def number(entry, key):
        found, held = value(entry, key), ctypes.c_int64()
        # kCFNumberSInt64Type
        return held.value if found and foundation.CFNumberGetValue(found, 4, ctypes.byref(held)) else None

    # kCGWindowListOptionOnScreenOnly | kCGWindowListExcludeDesktopElements, from kCGNullWindowID
    listed = graphics.CGWindowListCopyWindowInfo(1 | 16, 0)
    if not listed:
        raise OSError("the window server lists no windows to this process")
    try:
        sizes = []
        for index in range(foundation.CFArrayGetCount(listed)):
            entry = foundation.CFArrayGetValueAtIndex(listed, index)
            bounds, rect = value(entry, "kCGWindowBounds"), Rect()
            # Layer 0 holds the windows of apps; their menus, the Dock and the menu bar are above it.
            if (number(entry, "kCGWindowOwnerPID") == pid and number(entry, "kCGWindowLayer") == 0 and bounds
                    and graphics.CGRectMakeWithDictionaryRepresentation(bounds, ctypes.byref(rect))):
                sizes.append((int(rect.width), int(rect.height)))
    finally:
        foundation.CFRelease(listed)
    return sizes


def _darwin_screenshot(path):
    """The screen as a PNG at `path`; why not, when the system does not show it to this process."""
    try:
        result = subprocess.run(["screencapture", "-x", str(path)], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired) as error:
        return str(error)
    if result.returncode or not path.is_file():
        return (result.stdout + result.stderr).strip() or f"screencapture exited {result.returncode}"
    return None


class Installed(NamedTuple):
    """An installed Markitai Web, as its system lays it out."""
    program: Path  # the app
    markitai: Path  # the markitai it starts
    logs: Path  # where it keeps serve.log, from HOME
    data: Path  # where it keeps web.json, from HOME
    shell: str  # a new user's login shell, which the app reads its environment from
    windows: Callable  # the sizes of a process's windows on the screen
    screenshot: Callable  # the screen as a PNG at a path; why not, when there is none


def exercise(app, report, work, evidence=None):
    """Start the installed `app` with a HOME of the check's own in `work`, and check what a user gets from
    it: its server, its window, a conversion, and a quit that leaves nothing running."""
    home, markitai_home = work / "home", work / "markitai-home"
    home.mkdir()
    markitai_home.mkdir()
    # A login shell in an empty HOME would start zsh's new-user setup.
    (home / ".zshrc").write_text("# the install check\n", encoding="utf-8")
    serve_log = home / app.logs / "serve.log"
    servers = []
    with open((work if evidence is None else evidence) / "installed-app.log", "wb") as output:
        # In a session of its own, without a terminal, as a desktop starts it: the app then reads its
        # environment from a login shell.
        process = subprocess.Popen([str(app.program)], cwd=work, stdin=subprocess.DEVNULL, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True,
                                   env=user_environment(home, markitai_home, app.shell))
        report.log(f"started: process {process.pid}, {app.program}")
        try:
            origin = wait_for(lambda: listening(serve_log), START_SECONDS, process)
            if not report.check(origin, f"the installed app's serve listens ({app.logs / 'serve.log'} in HOME)"):
                status = process.poll()
                report.log("the app " + (f"exited {status}" if status is not None
                                         else f"had no server listening {START_SECONDS} s later"))
                return
            servers = [found for found in started_by(process.pid) if Path(found[1]).name == app.markitai.name]
            for pid, program, arguments in servers:
                report.log(f"serve: process {pid}, {program} {' '.join(arguments[1:])}")
            bundled = (len(servers) == 1 and os.path.realpath(servers[0][1]) == os.path.realpath(app.markitai)
                       and "serve --supervised" in " ".join(servers[0][2]))
            report.check(bundled, f"its child is the bundled {app.markitai.name} serve --supervised")
            try:
                windows = wait_for(lambda: app.windows(process.pid), WINDOW_SECONDS, process)
                seen = ", ".join(f"{width}x{height}" for width, height in windows or []) or "none"
            except OSError as error:
                windows, seen = None, str(error)
            report.check(windows, f"the app shows a window ({seen})")
            if evidence is not None:
                time.sleep(PAINT_SECONDS)
                refused = app.screenshot(evidence / "installed-ready.png")
                report.log("screen: installed-ready.png" if refused is None else f"screen: not kept ({refused})")
            report.check(wait_for(lambda: (home / app.data / "web.json").is_file(), WINDOW_SECONDS, process),
                         f"the port kept in {app.data / 'web.json'} in HOME")
            if bundled:
                # The release build has no automation socket: the page goes to the app's server with the
                # token the app gave it, as the workbench in the window sends one.
                try:
                    report.log("convert: " + converted(servers[0][0], origin))
                    converts = True
                except Failure as failure:
                    report.log(f"convert: {failure}")
                    converts = False
                report.check(converts, "the installed app's serve converts a file")
            # As a user's Quit: mygo takes the signal as the menu's Quit.
            process.send_signal(signal.SIGTERM)
            try:
                status = process.wait(QUIT_SECONDS)
            except subprocess.TimeoutExpired:
                status = None
            report.check(status == 0, "the app quits with 0 (" + (f"it still runs {QUIT_SECONDS} s later"
                                                                  if status is None else str(status)) + ")")
            if bundled:
                report.check(wait_for(lambda: not alive(servers[0][0]), 10), "its serve has ended")
            report.check(DRAINING in serve_log.read_text(encoding="utf-8", errors="replace"), "serve drained")
        finally:
            # What still runs goes with the app's session, and a server that left it by itself.
            for pid in [-process.pid, *(pid for pid, _, _ in servers)]:
                try:
                    os.kill(pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            process.wait(30)
            if evidence is not None and serve_log.is_file():
                shutil.copyfile(serve_log, evidence / "installed-serve.log")


def detach(image):
    """Detach the disk image attached at `image`; whether it went. A scan or an index of the new volume may
    hold it for a moment, and after a few seconds it is detached all the same."""
    for attempt in range(6):
        force = ["-force"] if attempt == 5 else []
        if subprocess.run(["hdiutil", "detach", str(image), *force], capture_output=True).returncode == 0:
            return True
        time.sleep(1)
    return False


def run_darwin(package, report, evidence=None, applications=APPLICATIONS):
    """The disk image `package`'s app copied to `applications` as dragging it to the link beside it does,
    exercised there and removed."""
    installed = applications / f"{APP_NAME}.app"
    if os.path.lexists(installed):
        raise Failure(f"{installed} is already there: the check installs the app and removes it, and replaces none")
    listed = [line.split(None, 1) for line in command(["ps", "-axo", "pid=,comm="]).splitlines()]
    running = [process[0] for process in listed if len(process) == 2 and Path(process[1]).name == APP_NAME]
    if running:
        raise Failure(f"{APP_NAME} already runs (process {', '.join(running)}): a second one hands over to it "
                      "and exits")
    # Its real path: serve refuses a MARKITAI_HOME reached through a symbolic link, as /var and /tmp are.
    work = Path(tempfile.mkdtemp(prefix="markitai-installed-")).resolve()
    image = work / "image"
    image.mkdir()
    try:
        command(["hdiutil", "attach", "-nobrowse", "-readonly", "-mountpoint", image, package])
        try:
            held = sorted(entry.name for entry in image.iterdir() if not entry.name.startswith("."))
            link = image / "Applications"
            report.check(held == ["Applications", installed.name] and link.is_symlink()
                         and os.readlink(link) == "/Applications",
                         f"the disk image holds {installed.name} beside a link to /Applications ({', '.join(held)})")
            command(["ditto", image / installed.name, installed])
        finally:
            if not detach(image):
                report.log(f"the disk image stayed attached at {image}")
        resources = installed / "Contents/Resources"
        app = Installed(installed / "Contents/MacOS" / APP_NAME, resources / "bin/markitai",
                        Path("Library/Logs") / APP_NAME, Path("Library/Application Support") / APP_NAME,
                        "/bin/zsh", _darwin_windows, _darwin_screenshot)
        if report.check(os.access(app.program, os.X_OK) and os.access(app.markitai, os.X_OK)
                        and (resources / "licenses").is_dir(),
                        f"the app is in {applications} with its markitai and licences"):
            exercise(app, report, work, evidence)
    finally:
        if os.path.lexists(installed):
            shutil.rmtree(installed)
            report.log(f"removed: {installed}")
        shutil.rmtree(work, ignore_errors=True)


def run(package, evidence=None):
    """`run`'s exit status: 1 when a check failed."""
    if sys.platform != "darwin":
        raise SystemExit("run installs the disk image on macOS; on Windows, .github/workflows/desktop-web.yml's "
                         "own step installs the app")
    report = Report(evidence)
    try:
        if not package.is_file():
            raise Failure(f"no package at {package}")
        report.log(f"package: {package.name}, {package.stat().st_size} bytes, "
                   f"sha256 {hashlib.sha256(package.read_bytes()).hexdigest()}")
        report.log(f"system: macOS {platform.mac_ver()[0]} {platform.machine()}")
        run_darwin(package, report, evidence)
    except Failure as failure:
        report.check(False, str(failure))
    if report.failures:
        report.log(f"{report.failures} check(s) failed")
    report.close()
    return 1 if report.failures else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    running = commands.add_parser("run", help="install this system's package, run the installed app, remove it")
    running.add_argument("--package", type=Path, required=True, help="the release build's: on macOS, the .dmg")
    running.add_argument("--evidence", type=Path,
                         help="where to keep install.log, the app's output, its serve.log and the screen")
    converting = commands.add_parser("convert", help="convert a file with the server an installed app started")
    converting.add_argument("--pid", type=int, required=True, help="the app's `markitai serve`")
    converting.add_argument("--origin", required=True, help="where it listens, as serve.log says: http://127.0.0.1:PORT")
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args.package.resolve(), args.evidence.resolve() if args.evidence else None)
    # What failed goes to standard output with the rest: the Windows step logs this command's lines.
    try:
        print(converted(args.pid, args.origin))
    except Failure as failure:
        print(f"FAIL {failure}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
