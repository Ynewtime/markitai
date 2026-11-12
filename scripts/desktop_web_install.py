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
        Linux: PATH is the Debian package, or the .tar.gz with install.sh
        beside it. apt-get installs the package with its dependencies
        alone, as root (through sudo, which must not ask, when the check is
        not root), the app runs as /usr/bin/markitai-web, and apt-get
        removes the package and leaves what it depends on. install.sh
        installs the archive in the check's HOME, the app runs as
        ~/.local/bin/markitai-web, and install.sh --uninstall removes it
        and must leave its settings. The app needs an X display (DISPLAY),
        where the check finds its window and closes it as a window manager
        does for a user; a Wayland session's display is kept from the app.
        The check does not start when the package is installed or the app
        runs already.
  convert --pid PID --origin http://127.0.0.1:PORT
        Convert a file with the `markitai serve` an installed Markitai Web
        started (PID is that server's, the app's child). The release build
        has no automation socket and keeps the server's token to itself and
        its child, so the token comes from where the app put it, the child's
        environment (MARKITAI_SERVE_TOKEN), read as the same user. The file,
        a small HTML page, goes to POST /api/jobs as the workbench sends one;
        the finished item's Markdown must be the page's heading and text.

Nothing here builds: scripts/desktop_web_build.py builds the app and its
packages. .github/workflows/desktop-web.yml runs `run` on macOS and, under
Xvfb, on Linux; on Windows its own step installs the app and calls
`convert`.
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
import zlib

# The app's name (desktop/web/mygo.json): its bundle, and its folders for logs and data.
APP_NAME = "Markitai Web"
# Where a user drags the app from its disk image.
APPLICATIONS = Path("/Applications")
# On Linux, desktop/web/mygo.json's slug, which names the package, its folder and its command, and its
# identifier, which names the app's folders in HOME and its entry in the applications menu.
SLUG = "markitai-web"
APP_ID = "dev.markitai.web"
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


def _linux_process(pid):
    """(program, arguments) of process `pid`: the file it runs, and the arguments it was started with."""
    program = os.readlink(f"/proc/{pid}/exe")
    arguments = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
    return program, [argument.decode("utf-8", "replace") for argument in arguments]


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


def command(arguments, timeout=120, environment=None, with_errors=False):
    """A command's standard output, and its standard error with it for `with_errors`, run in `environment`
    when there is one. One that fails, or is not done in `timeout` s, is a Failure with what it printed."""
    arguments = [str(argument) for argument in arguments]
    try:
        result = subprocess.run(arguments, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT if with_errors else subprocess.PIPE, text=True,
                                errors="replace", timeout=timeout, env=environment)
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Failure(f"{' '.join(arguments)}: {error}") from error
    if result.returncode:
        printed = (result.stdout + (result.stderr or "")).strip()[-600:]
        raise Failure(f"{' '.join(arguments)} exited {result.returncode}: {printed}")
    return result.stdout


def as_root(arguments, timeout=900):
    """command() as root: through sudo, which must not ask for a password, when this process is not root."""
    return command([*([] if os.geteuid() == 0 else ["sudo", "-n"]), *arguments], timeout, with_errors=True)


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
    if sys.platform.startswith("linux"):
        return _linux_process(pid)
    raise OSError(f"a process's command is not read on {sys.platform}")


def started_by(pid):
    """(pid, program, arguments) of each process that `pid` started and that still runs."""
    found = []
    for line in command(["ps", "-A", "-o", "pid=,ppid="]).splitlines():
        child, parent = (int(number) for number in line.split())
        if parent == pid:
            try:
                found.append((child, *process_command(child)))
            except OSError:
                pass  # it ended meanwhile, as the login shell the app reads its environment from does
    return found


def processes_named(name):
    """The numbers of the processes whose program is named `name`."""
    listed = [line.split(None, 1) for line in command(["ps", "-A", "-o", "pid=,comm="]).splitlines()]
    return [process[0] for process in listed if len(process) == 2 and Path(process[1]).name == name]


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
    and nothing of a terminal's or of a developer's markitai and mygo. Nothing either that takes the
    app's folders out of that HOME, as a runner's XDG_CONFIG_HOME does, and no Wayland display: the check
    finds the app's window on the X display."""
    dropped = {"TERM", "TERM_PROGRAM", "ZDOTDIR", "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME",
               "XDG_STATE_HOME", "WAYLAND_DISPLAY"}
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("MARKITAI_", "MYGO_")) and name not in dropped}
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


def _darwin_quit(process):
    """Quit the app as its user does from its menu: mygo takes this signal as the menu's Quit."""
    process.send_signal(signal.SIGTERM)
    return "SIGTERM, which the app takes as its menu's Quit"


class _Display:
    """The X display DISPLAY names, through libX11: a process's windows on it, and the screen."""
    # libX11 calls it for as long as this process lives.
    errors = None

    def __init__(self):
        import ctypes

        self.ctypes = ctypes
        try:
            x11 = self.x11 = ctypes.CDLL("libX11.so.6")
        except OSError as error:
            raise OSError(f"no libX11 to look at the display with: {error}") from error
        pointer, number, window = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulong
        long_at, window_at = ctypes.POINTER(ctypes.c_ulong), ctypes.POINTER(window)
        for name, result, arguments in (
                ("XOpenDisplay", pointer, (ctypes.c_char_p,)),
                ("XCloseDisplay", number, (pointer,)),
                ("XDefaultRootWindow", window, (pointer,)),
                ("XInternAtom", ctypes.c_ulong, (pointer, ctypes.c_char_p, number)),
                ("XQueryTree", number, (pointer, window, window_at, window_at, ctypes.POINTER(window_at),
                                        ctypes.POINTER(ctypes.c_uint))),
                ("XGetWindowProperty", number, (pointer, window, ctypes.c_ulong, ctypes.c_long, ctypes.c_long, number,
                                                ctypes.c_ulong, long_at, ctypes.POINTER(number), long_at, long_at,
                                                ctypes.POINTER(long_at))),
                ("XGetWindowAttributes", number, (pointer, window, pointer)),
                ("XSendEvent", number, (pointer, window, number, ctypes.c_long, pointer)),
                ("XFlush", number, (pointer,)),
                ("XGetImage", pointer, (pointer, window, number, number, ctypes.c_uint, ctypes.c_uint, ctypes.c_ulong,
                                        number)),
                ("XFree", number, (pointer,))):
            function = getattr(x11, name)
            function.restype, function.argtypes = result, arguments

        class Attributes(ctypes.Structure):  # XWindowAttributes
            _fields_ = [(name, number) for name in ("x", "y", "width", "height", "border_width", "depth")] + [
                ("visual", pointer), ("root", window), ("window_class", number), ("bit_gravity", number),
                ("win_gravity", number), ("backing_store", number), ("backing_planes", ctypes.c_ulong),
                ("backing_pixel", ctypes.c_ulong), ("save_under", number), ("colormap", ctypes.c_ulong),
                ("map_installed", number), ("map_state", number), ("all_event_masks", ctypes.c_long),
                ("your_event_mask", ctypes.c_long), ("do_not_propagate_mask", ctypes.c_long),
                ("override_redirect", number), ("screen", pointer)]

        class Message(ctypes.Structure):  # XClientMessageEvent, its data as longs
            _fields_ = [("type", number), ("serial", ctypes.c_ulong), ("send_event", number), ("display", pointer),
                        ("window", window), ("message_type", ctypes.c_ulong), ("format", number),
                        ("data", ctypes.c_long * 5)]

        class Event(ctypes.Union):  # XEvent, as large as Xlib's
            _fields_ = [("message", Message), ("pad", ctypes.c_long * 24)]

        class Image(ctypes.Structure):  # XImage, up to what tells its pixels apart
            _fields_ = [(name, number) for name in ("width", "height", "xoffset", "format")] + [("data", pointer)] + [
                (name, number) for name in ("byte_order", "bitmap_unit", "bitmap_bit_order", "bitmap_pad", "depth",
                                            "bytes_per_line", "bits_per_pixel")] + [
                (name, ctypes.c_ulong) for name in ("red_mask", "green_mask", "blue_mask")]

        self.Attributes, self.Event, self.Image = Attributes, Event, Image
        if _Display.errors is None:
            # An error the server answers, as for a window that went meanwhile, fails that request and
            # no more: Xlib's own handler ends the process.
            handler = ctypes.CFUNCTYPE(number, pointer, pointer)
            _Display.errors = handler(lambda display, event: 0)
            x11.XSetErrorHandler.restype, x11.XSetErrorHandler.argtypes = pointer, (handler,)
            x11.XSetErrorHandler(_Display.errors)
        self.display = x11.XOpenDisplay(None)
        if not self.display:
            raise OSError(f"no X display opens: DISPLAY is {os.environ.get('DISPLAY') or 'not set'}")

    def close(self):
        self.x11.XCloseDisplay(self.display)

    def atom(self, name):
        return self.x11.XInternAtom(self.display, name.encode(), 0)

    def shown(self, pid):
        """(window, width, height) of each window process `pid` has on the screen: one that names the
        process as its own (_NET_WM_PID, as GTK's do) and is mapped with all it is inside."""
        ctypes, x11 = self.ctypes, self.x11
        owner = self.atom("_NET_WM_PID")
        found, queue = [], [x11.XDefaultRootWindow(self.display)]
        while queue:
            window = queue.pop()
            root, parent, count = ctypes.c_ulong(), ctypes.c_ulong(), ctypes.c_uint()
            children = ctypes.POINTER(ctypes.c_ulong)()
            if x11.XQueryTree(self.display, window, ctypes.byref(root), ctypes.byref(parent), ctypes.byref(children),
                              ctypes.byref(count)):
                if count.value:
                    queue += children[:count.value]
                if children:
                    x11.XFree(children)
            kind, size, items, left = ctypes.c_ulong(), ctypes.c_int(), ctypes.c_ulong(), ctypes.c_ulong()
            value = ctypes.POINTER(ctypes.c_ulong)()
            # A CARDINAL of 32 bits, which Xlib hands over as a C long.
            read = x11.XGetWindowProperty(self.display, window, owner, 0, 1, 0, 6, ctypes.byref(kind),
                                          ctypes.byref(size), ctypes.byref(items), ctypes.byref(left),
                                          ctypes.byref(value))
            held = value[0] if read == 0 and value and size.value == 32 and items.value == 1 else None
            if value:
                x11.XFree(value)
            attributes = self.Attributes()
            # IsViewable; GTK also keeps a window of one pixel, never mapped, that leads its others.
            if (held == pid and x11.XGetWindowAttributes(self.display, window, ctypes.byref(attributes))
                    and attributes.map_state == 2 and attributes.width > 1 and attributes.height > 1):
                found.append((window, attributes.width, attributes.height))
        return found

    def ask_to_close(self, window):
        """What a window manager sends a window its user closes: the client message WM_DELETE_WINDOW."""
        event = self.Event()
        message = event.message
        message.type, message.send_event, message.display, message.window = 33, 1, self.display, window
        message.message_type, message.format = self.atom("WM_PROTOCOLS"), 32
        message.data[0], message.data[1] = self.atom("WM_DELETE_WINDOW"), 0  # CurrentTime
        self.x11.XSendEvent(self.display, window, 0, 0, self.ctypes.byref(event))
        self.x11.XFlush(self.display)

    def screen(self):
        """(width, height, rows of RGB bytes) of the screen, as the server holds it."""
        ctypes, x11 = self.ctypes, self.x11
        root, attributes = x11.XDefaultRootWindow(self.display), self.Attributes()
        if not x11.XGetWindowAttributes(self.display, root, ctypes.byref(attributes)):
            raise OSError("the X server does not say how large its screen is")
        # AllPlanes, ZPixmap
        image = x11.XGetImage(self.display, root, 0, 0, attributes.width, attributes.height, 2 ** 64 - 1, 2)
        if not image:
            raise OSError("the X server gives no image of its screen")
        held = ctypes.cast(image, ctypes.POINTER(self.Image)).contents
        try:
            masks = (held.red_mask, held.green_mask, held.blue_mask)
            # Blue, green, red and a spare byte, as a 24-bit screen's image is on a little-endian machine.
            if (held.bits_per_pixel, held.byte_order, masks) != (32, 0, (0xFF0000, 0xFF00, 0xFF)):
                raise OSError(f"the screen's image is not 32 bits a pixel with blue first: {held.bits_per_pixel} "
                              f"bits, byte order {held.byte_order}, masks {', '.join(map(hex, masks))}")
            data = ctypes.string_at(held.data, held.bytes_per_line * held.height)
            rows = []
            for line in range(held.height):
                pixels = data[line * held.bytes_per_line:line * held.bytes_per_line + 4 * held.width]
                row = bytearray(3 * held.width)
                row[0::3], row[1::3], row[2::3] = pixels[2::4], pixels[1::4], pixels[0::4]
                rows.append(bytes(row))
            return held.width, held.height, rows
        finally:
            # As XDestroyImage does for an image XGetImage made.
            x11.XFree(held.data)
            x11.XFree(image)


def png(width, height, rows):
    """A PNG of `rows`, each `width` pixels of red, green and blue bytes."""
    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    # 8 bits a sample, colour type 2 (RGB), each row with filter 0 (none).
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(b"".join(b"\0" + row for row in rows), 6)) + chunk(b"IEND", b""))


def _x11_windows(pid):
    """(width, height) of each of process `pid`'s windows on the X display."""
    display = _Display()
    try:
        return [(width, height) for _, width, height in display.shown(pid)]
    finally:
        display.close()


def _x11_screenshot(path):
    """The X display's screen as a PNG at `path`; why not, when it gives none."""
    try:
        display = _Display()
        try:
            width, height, rows = display.screen()
        finally:
            display.close()
    except OSError as error:
        return str(error)
    path.write_bytes(png(width, height, rows))
    return None


def _x11_quit(process):
    """Close the app's windows as a window manager does for its user, which quits the app: Linux has no
    menu to quit it from, and SIGTERM is there the session's end, which asks nothing."""
    display = _Display()
    try:
        windows = display.shown(process.pid)
        for window, _, _ in windows:
            display.ask_to_close(window)
    finally:
        display.close()
    if not windows:
        raise OSError("the app has no window on the display to close")
    return f"{len(windows)} window(s) asked to close, as a window manager asks for a user"


class Installed(NamedTuple):
    """An installed Markitai Web, as its system lays it out."""
    program: Path  # the app
    markitai: Path  # the markitai it starts
    logs: Path  # where it keeps serve.log, from HOME
    data: Path  # where it keeps web.json, from HOME
    shell: str  # a new user's login shell, which the app reads its environment from
    windows: Callable  # the sizes of a process's windows on the screen
    screenshot: Callable  # the screen as a PNG at a path; why not, when there is none
    quit: Callable  # quits the app, a process, as its user does; how, in a few words


def homes(work):
    """A HOME and a MARKITAI_HOME of the check's own, in `work`: made, or there from an install."""
    home, markitai_home = work / "home", work / "markitai-home"
    home.mkdir(exist_ok=True)
    markitai_home.mkdir(exist_ok=True)
    return home, markitai_home


def exercise(app, report, work, evidence=None):
    """Start the installed `app` with a HOME of the check's own in `work`, and check what a user gets from
    it: its server, its window, a conversion, and a quit that leaves nothing running."""
    home, markitai_home = homes(work)
    if Path(app.shell).name == "zsh":
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
            try:
                report.log("quit: " + app.quit(process))
            except OSError as error:
                report.log(f"quit: {error}")
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


def not_running(name):
    """A Failure when the app, a program named `name`, runs already: its second start hands over to the
    first and exits."""
    found = processes_named(name)
    if found:
        raise Failure(f"{APP_NAME} already runs (process {', '.join(found)}): a second one hands over to it "
                      "and exits")


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
    not_running(APP_NAME)
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
                        "/bin/zsh", _darwin_windows, _darwin_screenshot, _darwin_quit)
        if report.check(os.access(app.program, os.X_OK) and os.access(app.markitai, os.X_OK)
                        and (resources / "licenses").is_dir(),
                        f"the app is in {applications} with its markitai and licences"):
            exercise(app, report, work, evidence)
    finally:
        if os.path.lexists(installed):
            shutil.rmtree(installed)
            report.log(f"removed: {installed}")
        shutil.rmtree(work, ignore_errors=True)


def linux_app(program, markitai):
    """Markitai Web on Linux, started as `program` with its markitai at `markitai`: its settings and its
    server's log are in .config/APP_ID of HOME (docs/desktop-web.md), and bash is a new user's shell."""
    folder = Path(".config") / APP_ID
    return Installed(program, markitai, folder / "logs", folder, "/bin/bash", _x11_windows, _x11_screenshot,
                     _x11_quit)


def entry_runs(entry, program):
    """Whether the desktop entry at `entry` starts `program`."""
    try:
        lines = entry.read_text(encoding="utf-8").splitlines()
    except OSError:
        return False
    return any(line in (f"Exec={program}", f'Exec="{program}"') for line in lines)


def installed_whole(folder, link, entry):
    """Whether the app is in `folder` with its markitai and licences, `link` is its command and `entry`
    starts it."""
    program = folder / SLUG
    return (os.access(program, os.X_OK) and os.access(folder / "bin/markitai", os.X_OK)
            and (folder / "licenses").is_dir() and os.path.realpath(link) == os.path.realpath(program)
            and entry_runs(entry, program))


def debian_installed(name):
    """Whether dpkg has the package `name` installed, or left half way."""
    try:
        status = command(["dpkg-query", "-W", "-f", "${db:Status-Status}", name]).strip()
    except Failure:
        return False  # dpkg knows no such package
    return status not in ("", "not-installed", "config-files")


def run_debian(package, report, evidence=None, root=Path("/")):
    """The Debian package `package` installed by apt-get with what it depends on and no more, exercised as
    the command it installs, and removed. `root` is where the system's files are."""
    folder, link = root / "opt" / SLUG, root / "usr/bin" / SLUG
    entry = root / "usr/share/applications" / f"{APP_ID}.desktop"
    if debian_installed(SLUG) or os.path.lexists(folder):
        raise Failure(f"{SLUG} is installed already: the check installs the package and removes it, and "
                      "replaces none")
    not_running(SLUG)
    apt = ["env", "DEBIAN_FRONTEND=noninteractive", "apt-get"]
    # Its real path: serve refuses a MARKITAI_HOME reached through a symbolic link.
    work = Path(tempfile.mkdtemp(prefix="markitai-installed-")).resolve()
    try:
        report.log("depends: " + command(["dpkg-deb", "-f", package, "Depends"]).strip())
        try:
            as_root([*apt, "install", "-y", "--no-install-recommends", package])
            report.log("WebKitGTK: " + command(["dpkg-query", "-W", "-f", "${Version}", "libwebkit2gtk-4.1-0"]))
            if report.check(installed_whole(folder, link, entry),
                            f"apt-get installs the app in {folder} with its markitai and licences, the command "
                            f"{link} and the entry {entry.name}"):
                exercise(linux_app(link, folder / "bin/markitai"), report, work, evidence)
        finally:
            if debian_installed(SLUG):
                try:
                    as_root([*apt, "remove", "-y", SLUG])
                    report.check(not any(os.path.lexists(path) for path in (folder, link, entry)),
                                 "apt-get remove removes the app, its command and its entry")
                except Failure as failure:
                    report.check(False, str(failure))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_archive(package, report, evidence=None):
    """The archive `package` installed by the install.sh beside it for the check's HOME, exercised as the
    command it links there, and removed by install.sh --uninstall."""
    script = package.parent / "install.sh"
    if not script.is_file():
        raise Failure(f"no install.sh beside {package.name}: it installs the archive beside it")
    not_running(SLUG)
    work = Path(tempfile.mkdtemp(prefix="markitai-installed-")).resolve()
    try:
        home, markitai_home = homes(work)
        folder, link = home / ".local" / f"{SLUG}.app", home / ".local/bin" / SLUG
        entry = home / ".local/share/applications" / f"{APP_ID}.desktop"
        app = linux_app(link, folder / "bin/markitai")
        environment = user_environment(home, markitai_home, app.shell)

        def install(*arguments):
            printed = command(["sh", script, *arguments], environment=environment, with_errors=True)
            lines = [line.strip().replace(str(home), "~") for line in printed.splitlines() if line.strip()]
            report.log(" ".join(["install.sh", *arguments]) + ": " + " / ".join(lines))

        install()
        if report.check(installed_whole(folder, link, entry),
                        f"install.sh installs the app in ~/.local/{folder.name} with its markitai and licences, "
                        f"the command ~/.local/bin/{SLUG} and the entry {entry.name}"):
            exercise(app, report, work, evidence)
        kept = [path for path in (home / app.data / "web.json", home / app.logs / "serve.log") if path.is_file()]
        install("--uninstall")
        report.check(not any(os.path.lexists(path) for path in (folder, link, entry)),
                     "install.sh --uninstall removes the app, its command and its entry")
        if kept:
            report.check(all(path.is_file() for path in kept),
                         f"and leaves the app's settings and its server's log ({app.data} in HOME)")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def run_linux(package, report, evidence=None):
    if not os.environ.get("DISPLAY"):
        raise Failure("the app needs an X display and DISPLAY names none: .github/workflows/desktop-web.yml "
                      "runs the check under xvfb-run")
    if package.name.endswith(".deb"):
        run_debian(package, report, evidence)
    elif package.name.endswith(".tar.gz"):
        run_archive(package, report, evidence)
    else:
        raise Failure(f"{package.name} is neither the Debian package nor the .tar.gz that install.sh installs")


def linux_name():
    """The distribution's name, as os-release has it."""
    for file in ("/etc/os-release", "/usr/lib/os-release"):
        try:
            found = re.search(r'^PRETTY_NAME="?([^"\n]*)"?$', Path(file).read_text(encoding="utf-8"), re.M)
        except OSError:
            continue
        if found:
            return found.group(1)
    return "Linux"


def run(package, evidence=None):
    """`run`'s exit status: 1 when a check failed."""
    if sys.platform == "darwin":
        system, install = f"macOS {platform.mac_ver()[0]} {platform.machine()}", run_darwin
    elif sys.platform.startswith("linux"):
        system = f"{linux_name()} {platform.machine()}, X display {os.environ.get('DISPLAY') or 'none'}"
        install = run_linux
    else:
        raise SystemExit("run installs the disk image on macOS, and the Debian package or install.sh's archive "
                         "on Linux; on Windows, .github/workflows/desktop-web.yml's own step installs the app")
    report = Report(evidence)
    try:
        if not package.is_file():
            raise Failure(f"no package at {package}")
        report.log(f"package: {package.name}, {package.stat().st_size} bytes, "
                   f"sha256 {hashlib.sha256(package.read_bytes()).hexdigest()}")
        report.log(f"system: {system}")
        install(package, report, evidence)
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
    running.add_argument("--package", type=Path, required=True,
                         help="the release build's: on macOS, the .dmg; on Linux, the .deb, or the .tar.gz beside "
                              "install.sh")
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
