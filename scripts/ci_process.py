"""Run a CI stage with complete file logs and content-free progress events."""
import os
import signal
import subprocess
import time


def _stop_tree(process):
    """Stop only descendants of the stage we launched, including compilers."""
    cleanup = {"tree_termination": "unverified"}
    if os.name == "nt":
        # Killing the root first would lose its child relationship on Windows.
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
