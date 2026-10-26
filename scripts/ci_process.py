"""Run a CI stage with complete file logs and content-free progress events."""
import os
import signal
import subprocess
import time


class _WindowsJob:
    """A Job Object that holds a stage and every process it starts (Windows).

    The stage is created suspended and joins the job before it runs, so no
    descendant can start outside it, and the job permits no breakaway, so a
    launcher's own job (silent breakaway) cannot take children out of it.
    Walking parent IDs or `taskkill /T` misses a grandchild whose parent has
    exited, and taskkill is slow enough on a loaded host for a child to keep
    working meanwhile; terminating the job reaches every member at once.
    With `kill_on_close`, closing the job's last handle (or the end of the
    process holding it) terminates its members as well.
    """

    CREATE_SUSPENDED = 0x00000004
    PROCESS_TERMINATE, PROCESS_SET_QUOTA, PROCESS_SUSPEND_RESUME = 0x0001, 0x0100, 0x0800
    BASIC_ACCOUNTING = 1  # JobObjectBasicAccountingInformation
    EXTENDED_LIMITS = 9  # JobObjectExtendedLimitInformation
    KILL_ON_JOB_CLOSE = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    def __init__(self, kill_on_close=False):
        import ctypes
        from ctypes import wintypes

        class Accounting(ctypes.Structure):
            _fields_ = [("TotalUserTime", ctypes.c_int64), ("TotalKernelTime", ctypes.c_int64),
                        ("ThisPeriodTotalUserTime", ctypes.c_int64), ("ThisPeriodTotalKernelTime", ctypes.c_int64),
                        ("TotalPageFaultCount", wintypes.DWORD), ("TotalProcesses", wintypes.DWORD),
                        ("ActiveProcesses", wintypes.DWORD), ("TotalTerminatedProcesses", wintypes.DWORD)]

        # JOBOBJECT_BASIC_LIMIT_INFORMATION and JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        # (IoInfo is IO_COUNTERS, six ULONGLONGs); 144 bytes on 64-bit Windows.
        class BasicLimits(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        ntdll = ctypes.WinDLL("ntdll")
        signatures = [
            (kernel32.CreateJobObjectW, wintypes.HANDLE, [wintypes.LPVOID, wintypes.LPCWSTR]),
            (kernel32.AssignProcessToJobObject, wintypes.BOOL, [wintypes.HANDLE, wintypes.HANDLE]),
            (kernel32.TerminateJobObject, wintypes.BOOL, [wintypes.HANDLE, wintypes.UINT]),
            (kernel32.QueryInformationJobObject, wintypes.BOOL,
             [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD, wintypes.LPDWORD]),
            (kernel32.SetInformationJobObject, wintypes.BOOL,
             [wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD]),
            (kernel32.OpenProcess, wintypes.HANDLE, [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]),
            (kernel32.CloseHandle, wintypes.BOOL, [wintypes.HANDLE]),
            # Resumes a process created suspended (all its threads), as psutil does.
            (ntdll.NtResumeProcess, ctypes.c_long, [wintypes.HANDLE]),
        ]
        for function, result, arguments in signatures:
            function.restype, function.argtypes = result, arguments
        self._ctypes, self._kernel32, self._ntdll, self._accounting = ctypes, kernel32, ntdll, Accounting
        self._handle = kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        if kill_on_close:
            limits = ExtendedLimits()
            limits.BasicLimitInformation.LimitFlags = self.KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(self._handle, self.EXTENDED_LIMITS, ctypes.byref(limits),
                                                    ctypes.sizeof(limits)):
                error = self._error()
                self.close()
                raise error

    def _error(self):
        return self._ctypes.WinError(self._ctypes.get_last_error())

    def adopt(self, pid):
        """Put the suspended process `pid` in the job, then let it run. Returns
        whether it joined; it runs either way. Raises only if it cannot resume."""
        process = self._kernel32.OpenProcess(
            self.PROCESS_TERMINATE | self.PROCESS_SET_QUOTA | self.PROCESS_SUSPEND_RESUME, False, pid)
        if not process:
            raise self._error()
        try:
            joined = bool(self._kernel32.AssignProcessToJobObject(self._handle, process))
            if self._ntdll.NtResumeProcess(process) < 0:
                raise OSError("The stage could not be resumed")
            return joined
        finally:
            self._kernel32.CloseHandle(process)

    def counts(self):
        """(processes still running in the job, processes it has ever held)."""
        info = self._accounting()
        if not self._kernel32.QueryInformationJobObject(self._handle, self.BASIC_ACCOUNTING,
                                                        self._ctypes.byref(info), self._ctypes.sizeof(info), None):
            raise self._error()
        return info.ActiveProcesses, info.TotalProcesses

    def stop(self, timeout):
        """Terminate every process in the job, again for any started meanwhile,
        until none runs or `timeout` passes; return how many still run."""
        deadline = time.monotonic() + timeout
        while True:
            if not self._kernel32.TerminateJobObject(self._handle, 1):
                raise self._error()
            running, _ = self.counts()
            if not running or time.monotonic() >= deadline:
                return running
            time.sleep(0.02)

    def close(self):
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _stop_tree(process, job=None):
    """Stop only the stage we launched and its descendants, including compilers.

    On Windows the stage's job (see _WindowsJob) is terminated and counted;
    without one, `taskkill /T` is the fallback and only its exit code speaks.
    """
    started = time.monotonic()
    cleanup = {"tree_termination": "unverified"}
    if os.name == "nt" and job is not None:
        cleanup["job"] = True
        try:
            cleanup["still_running"] = job.stop(timeout=15)
            cleanup["job_processes"] = job.counts()[1]
            if cleanup["still_running"] == 0:
                cleanup["tree_termination"] = "confirmed"
        except OSError as error:
            cleanup["error_type"] = type(error).__name__
    elif os.name == "nt":
        # Killing the root first would lose its child relationship on Windows.
        cleanup["job"] = False
        try:
            result = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                    timeout=15, check=False)
            cleanup["taskkill_exit_code"] = result.returncode
            if result.returncode == 0:
                cleanup["tree_termination"] = "confirmed"
        except (OSError, subprocess.TimeoutExpired) as error:
            cleanup["error_type"] = type(error).__name__
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
            cleanup["tree_termination"] = "confirmed"
        except ProcessLookupError:
            cleanup["tree_termination"] = "already_exited"
    if process.poll() is None:
        process.kill()
    process.wait(timeout=15)
    cleanup["stop_seconds"] = round(time.monotonic() - started, 3)
    return cleanup


def _windows_job(kill_on_close=False):
    """A new job for a stage, or None where Windows cannot make one."""
    try:
        return _WindowsJob(kill_on_close)
    except (OSError, AttributeError):
        return None


def run_logged(command, *, cwd, env, log, timeout, progress, heartbeat=30):
    """Never stream child text, commands or environment values to CI stdout.

    Full output stays in the existing evidence log. A heartbeat reports elapsed
    time, log byte count and output idle time; idle output does not trigger a kill.
    A hard stage deadline leaves time for the workflow's always-upload step.
    On Windows the stage runs in its own job, so a stop reaches all it started.
    """
    if timeout <= 0 or heartbeat <= 0:
        raise ValueError("CI time limits must be positive")
    started = time.monotonic()
    last_output = started
    last_size = 0
    job = _windows_job() if os.name == "nt" else None
    try:
        with log.open("wb") as stream:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=stream,
                                       stderr=subprocess.STDOUT,
                                       start_new_session=os.name != "nt",
                                       creationflags=_WindowsJob.CREATE_SUSPENDED if job else 0)
            if job is not None:
                try:
                    joined = job.adopt(process.pid)
                except BaseException:
                    process.kill()
                    process.wait()
                    raise
                if not joined:
                    job.close()
                    job = None
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
                cleanup = _stop_tree(process, job)
                progress("cleanup", **cleanup)
                raise
    finally:
        if job is not None:
            job.close()


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
