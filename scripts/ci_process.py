"""Run a CI stage with complete file logs and content-free progress events."""
import os
import signal
import subprocess
import time


class _WindowsTree:
    """Handles to a process and its descendants, found in Toolhelp snapshots.

    A held handle keeps its process ID from being reused, so a later snapshot's
    parent IDs still point into the tree; a child is accepted only if it was
    created after the parent it names. taskkill's exit code cannot confirm a
    stop by itself: it fails when a member exits while the tree is being
    killed (a console host, a launcher's job object), and it cannot see what
    the tree starts afterwards.
    """

    SYNCHRONIZE, PROCESS_TERMINATE, QUERY_LIMITED = 0x00100000, 0x0001, 0x1000

    def __init__(self, pid):
        import ctypes
        from ctypes import wintypes

        class Entry(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", wintypes.WCHAR * 260)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateToolhelp32Snapshot": (wintypes.HANDLE, [wintypes.DWORD, wintypes.DWORD]),
            "Process32FirstW": (wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(Entry)]),
            "Process32NextW": (wintypes.BOOL, [wintypes.HANDLE, ctypes.POINTER(Entry)]),
            "OpenProcess": (wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]),
            "GetProcessTimes": (wintypes.BOOL, [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4),
            "WaitForSingleObject": (wintypes.DWORD, [wintypes.HANDLE, wintypes.DWORD]),
            "TerminateProcess": (wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
            "CloseHandle": (wintypes.BOOL, [wintypes.HANDLE]),
        }
        for name, (result, arguments) in signatures.items():
            function = getattr(kernel32, name)
            function.restype, function.argtypes = result, arguments
        self._ctypes, self._wintypes, self._kernel32, self._entry = ctypes, wintypes, kernel32, Entry
        self._held = {}
        handle = self._open(pid)
        if handle is None:
            raise ctypes.WinError(ctypes.get_last_error())
        self._held[pid] = (handle, self._created(handle))
        self.collect()

    def _open(self, pid):
        handle = self._kernel32.OpenProcess(self.SYNCHRONIZE | self.PROCESS_TERMINATE | self.QUERY_LIMITED, False, pid)
        return handle or None

    def _created(self, handle):
        times = [self._wintypes.FILETIME() for _ in range(4)]
        if not self._kernel32.GetProcessTimes(handle, *(self._ctypes.byref(time) for time in times)):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        return (times[0].dwHighDateTime << 32) | times[0].dwLowDateTime

    def _parents(self):
        snapshot = self._kernel32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
        if snapshot in (None, self._wintypes.HANDLE(-1).value):
            raise self._ctypes.WinError(self._ctypes.get_last_error())
        try:
            entry = self._entry(dwSize=self._ctypes.sizeof(self._entry))
            parents = {}
            more = self._kernel32.Process32FirstW(snapshot, self._ctypes.byref(entry))
            while more:
                parents[entry.th32ProcessID] = entry.th32ParentProcessID
                more = self._kernel32.Process32NextW(snapshot, self._ctypes.byref(entry))
            return parents
        finally:
            self._kernel32.CloseHandle(snapshot)

    def collect(self):
        """Hold every descendant a snapshot shows; return how many were new."""
        added = 0
        parents = self._parents()
        found = True
        while found:
            found = False
            for pid, parent in parents.items():
                if pid in self._held or parent not in self._held or pid == parent:
                    continue
                handle = self._open(pid)
                if handle is None:
                    continue  # Already gone, or not ours to stop.
                try:
                    created = self._created(handle)
                except OSError:
                    created = None
                if created is None or created < self._held[parent][1]:
                    self._kernel32.CloseHandle(handle)
                    continue
                self._held[pid] = (handle, created)
                added += 1
                found = True
        return added

    def _running(self):
        return [handle for handle, _ in self._held.values() if self._kernel32.WaitForSingleObject(handle, 0) != 0]

    def stop(self, timeout):
        """Terminate whatever still runs, including descendants started meanwhile;
        return how many held processes are still running at the end."""
        import time
        deadline = time.monotonic() + timeout
        for _ in range(8):  # Each round only adds processes started since the last.
            self.collect()
            running = self._running()
            if not running:
                return 0
            for handle in running:
                self._kernel32.TerminateProcess(handle, 1)
            for handle in running:
                remaining = max(0, int((deadline - time.monotonic()) * 1000))
                self._kernel32.WaitForSingleObject(handle, remaining)
            if time.monotonic() >= deadline:
                break
        self.collect()
        return len(self._running())

    def close(self):
        for handle, _ in self._held.values():
            self._kernel32.CloseHandle(handle)
        self._held.clear()


def _stop_tree(process, windows_tree=None):
    """Stop only descendants of the stage we launched, including compilers."""
    cleanup = {"tree_termination": "unverified"}
    if os.name == "nt":
        # Killing the root first would lose its child relationship on Windows.
        # The tree is recorded before taskkill runs and checked afterwards.
        tree = None
        try:
            tree = (windows_tree or _WindowsTree)(process.pid)
        except (OSError, AttributeError) as error:
            cleanup["inspection_error_type"] = type(error).__name__
        try:
            try:
                result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                        timeout=15, check=False)
                cleanup["taskkill_exit_code"] = result.returncode
            except (OSError, subprocess.TimeoutExpired) as error:
                cleanup["error_type"] = type(error).__name__
            if tree is None:
                if cleanup.get("taskkill_exit_code") == 0:
                    cleanup["tree_termination"] = "confirmed"
            else:
                cleanup["still_running"] = tree.stop(timeout=15)
                if cleanup["still_running"] == 0:
                    cleanup["tree_termination"] = "confirmed"
        except OSError as error:
            cleanup["inspection_error_type"] = type(error).__name__
        finally:
            if tree is not None:
                tree.close()
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            cleanup["tree_termination"] = "confirmed"
        except ProcessLookupError:
            cleanup["tree_termination"] = "already_exited"
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)
    return cleanup


def run_logged(command, *, cwd, env, log, timeout, progress, heartbeat=30):
    """Never stream child text, commands or environment values to CI stdout.

    Full output stays in the existing evidence log. A heartbeat reports elapsed
    time, log byte count and output idle time; idle output does not trigger a kill.
    A hard stage deadline leaves time for the workflow's always-upload step.
    """
    if timeout <= 0 or heartbeat <= 0:
        raise ValueError("CI time limits must be positive")
    started = time.monotonic()
    last_output = started
    last_size = 0
    with log.open("wb") as stream:
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT,
                                   start_new_session=os.name != "nt")
        try:
            while True:
                elapsed = time.monotonic() - started
                remaining = timeout - elapsed
                if remaining <= 0:
                    progress("timeout", duration_seconds=elapsed,
                             log_bytes=log.stat().st_size, timeout_seconds=timeout)
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = process.wait(timeout=min(heartbeat, remaining))
                    return subprocess.CompletedProcess(command, code)
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    size = log.stat().st_size
                    if size != last_size:
                        last_output = now
                        last_size = size
                    progress("running", duration_seconds=now - started,
                             log_bytes=size, output_idle_seconds=now - last_output,
                             timeout_seconds=timeout)
        except BaseException:
            cleanup = _stop_tree(process)
            progress("cleanup", **cleanup)
            raise


def write_summary(record, destination):
    """Expose stage timings without child output or diagnostic exception text."""
    import re
    def label(value):
        # Stage/status identifiers are authored by our harness; remain defensive.
        return re.sub(r"[^a-zA-Z0-9_-]", "?", str(value))[:80]
    lines = ["## Native package stages", "",
             f"Result: **{label(record['status'])}**", "",
             "| Stage | Seconds | Exit |", "| --- | ---: | ---: |"]
    for step in record["steps"]:
        duration = float(step["duration_seconds"])
        code = step["exit_code"]
        lines.append(f"| {label(step['name'])} | {duration:.1f} | {code if isinstance(code, int) else 'interrupted'} |")
    lines += ["", "Complete output and source/artifact identities are retained in the job artifact.", ""]
    with open(destination, "a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
