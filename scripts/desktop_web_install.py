#!/usr/bin/env python3
"""Markitai Web as it is installed: what a release build's package must do once a user has it.

  convert --pid PID --origin http://127.0.0.1:PORT
        Convert a file with the `markitai serve` an installed Markitai Web
        started (PID is that server's, the app's child). The release build
        has no automation socket and keeps the server's token to itself and
        its child, so the token comes from where the app put it, the child's
        environment (MARKITAI_SERVE_TOKEN), read as the same user. The file,
        a small HTML page, goes to POST /api/jobs as the workbench sends one;
        the finished item's Markdown must be the page's heading and text.

Nothing here builds or installs: scripts/desktop_web_build.py builds the app
and its packages, and .github/workflows/desktop-web.yml installs them.
"""
from pathlib import Path
import argparse
import json
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import uuid

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


def _darwin_environment(pid):
    """KERN_PROCARGS2: the argument count, the program's path, padding, the arguments, the environment."""
    import ctypes
    import ctypes.util

    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

    def sysctl(names, buffer, size):
        mib = (ctypes.c_int * len(names))(*names)
        if libc.sysctl(mib, len(names), buffer, ctypes.byref(size), None, 0):
            raise OSError(ctypes.get_errno(), f"sysctl {names}: {os.strerror(ctypes.get_errno())}")

    limit = ctypes.c_int()
    sysctl((1, 8), ctypes.byref(limit), ctypes.c_size_t(ctypes.sizeof(limit)))  # CTL_KERN, KERN_ARGMAX
    buffer, size = ctypes.create_string_buffer(limit.value), ctypes.c_size_t(limit.value)
    sysctl((1, 49, pid), buffer, size)  # CTL_KERN, KERN_PROCARGS2
    data = buffer.raw[:size.value]
    count = struct.unpack("i", data[:4])[0]
    start = data.index(b"\0", 4)
    while start < len(data) and data[start] == 0:
        start += 1
    for _ in range(count):
        start = data.index(b"\0", start) + 1
    return _strings(data[start:], "utf-8")


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    converting = commands.add_parser("convert", help="convert a file with the server an installed app started")
    converting.add_argument("--pid", type=int, required=True, help="the app's `markitai serve`")
    converting.add_argument("--origin", required=True, help="where it listens, as serve.log says: http://127.0.0.1:PORT")
    args = parser.parse_args(argv)
    # What failed goes to standard output with the rest: the Windows step logs this command's lines.
    try:
        markdown = convert(args.origin.rstrip("/"), serve_token(args.pid))
    except Failure as failure:
        print(f"FAIL {failure}")
        return 1
    lines = " / ".join(line.strip() for line in markdown.splitlines() if line.strip())
    print(f"process {args.pid} at {args.origin} converted {PAGE_NAME}: {lines}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
