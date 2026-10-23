"""Real tiny subprocesses exercise CI progress, deadlines and log retention."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ci_process import run_logged, _stop_tree, write_summary


class CIProcessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.log = self.root / "stage.log"
        self.events = []

    def run_stage(self, script, **options):
        return run_logged([sys.executable, "-c", script], cwd=self.root,
                          env=dict(os.environ, HOME=str(self.root), MARKITAI_HOME=str(self.root)),
                          log=self.log, timeout=options.pop("timeout", 5), heartbeat=.05,
                          progress=lambda event, **fields: self.events.append({"event": event, **fields}),
                          **options)

    def test_heartbeat_preserves_output_without_echoing_it(self):
        result = self.run_stage("import time; print('synthetic-secret', flush=True); time.sleep(.3)")
        self.assertEqual(result.returncode, 0)
        self.assertIn(b"synthetic-secret", self.log.read_bytes())
        self.assertGreaterEqual(len(self.events), 2)
        self.assertNotIn("synthetic-secret", json.dumps(self.events))
        self.assertTrue(all(e["event"] == "running" for e in self.events))
        self.assertTrue(any(e["output_idle_seconds"] > 0 for e in self.events))
        self.assertTrue(any(e["log_bytes"] > 0 for e in self.events))

    def test_nonzero_exit_and_complete_evidence(self):
        result = self.run_stage("import sys; print('failure detail', file=sys.stderr); sys.exit(7)")
        self.assertEqual(result.returncode, 7)
        self.assertIn(b"failure detail", self.log.read_bytes())

    def test_deadline_stops_child_and_keeps_partial_log(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_stage("import time; print('started',flush=True); time.sleep(20)", timeout=.6)
        self.assertIn(b"started", self.log.read_bytes())
        self.assertEqual(self.events[-2]["event"], "timeout")
        self.assertEqual(self.events[-1]["event"], "cleanup")

    def test_missing_executable_keeps_empty_log(self):
        with self.assertRaises(FileNotFoundError):
            run_logged([str(self.root / "missing")], cwd=self.root, env={}, log=self.log,
                       timeout=1, progress=lambda *args, **kwargs: None)
        self.assertEqual(self.log.read_bytes(), b"")

    def windows_stop(self, job=None, taskkill=0):
        """_stop_tree as on Windows: with the stage's job, or with taskkill's result."""
        class Process:
            pid = 12345
            def poll(self): return None
            def kill(self): calls.append("kill")
            def wait(self, **kwargs): calls.append("wait")

        def run(*args, **kwargs):
            calls.append(args[0])
            if isinstance(taskkill, Exception):
                raise taskkill
            return subprocess.CompletedProcess(args[0], taskkill)

        calls = []
        with patch("ci_process.os.name", "nt"), patch("ci_process.subprocess.run", side_effect=run):
            cleanup = _stop_tree(Process(), job)
        self.assertIsInstance(cleanup.pop("stop_seconds"), float)
        return cleanup, calls

    def test_windows_stop_terminates_the_stage_job(self):
        class Job:
            def __init__(self, running=0, error=None):
                self.running, self.error = running, error
            def stop(self, timeout):
                if self.error:
                    raise self.error
                return self.running
            def counts(self): return self.running, 4
        cleanup, calls = self.windows_stop(Job())
        self.assertEqual(cleanup, {"tree_termination": "confirmed", "job": True, "still_running": 0, "job_processes": 4})
        # No taskkill: the job holds every process the stage started; the root is then reaped.
        self.assertEqual(calls, ["kill", "wait"])
        cleanup, _ = self.windows_stop(Job(running=1))
        self.assertEqual(cleanup, {"tree_termination": "unverified", "job": True, "still_running": 1, "job_processes": 4})
        cleanup, calls = self.windows_stop(Job(error=OSError("synthetic-secret")))
        self.assertEqual(cleanup, {"tree_termination": "unverified", "job": True, "error_type": "OSError"})
        self.assertEqual(calls, ["kill", "wait"])

    def test_windows_stop_without_a_job_falls_back_to_taskkill(self):
        cleanup, calls = self.windows_stop(taskkill=0)
        self.assertEqual(cleanup, {"tree_termination": "confirmed", "job": False, "taskkill_exit_code": 0})
        self.assertEqual(calls, [["taskkill", "/PID", "12345", "/T", "/F"], "kill", "wait"])

    def test_windows_stage_starts_suspended_in_its_job(self):
        class Job:
            def __init__(self, joined=True, resumes=True):
                self.joined, self.resumes = joined, resumes
            def adopt(self, pid):
                calls.append(("adopt", pid))
                if not self.resumes:
                    raise OSError("synthetic-secret")
                return self.joined
            def close(self): calls.append("close")

        class Process:
            pid = 12345
            def __init__(self, command, **kwargs):
                calls.append(("popen", kwargs["creationflags"], kwargs["start_new_session"]))
            def wait(self, timeout=None):
                calls.append("wait")
                return 0
            def kill(self): calls.append("kill")

        for job, expected in [
            (Job(), [("popen", 4, False), ("adopt", 12345), "wait", "close"]),
            # A process the job refused still runs; its stop falls back to taskkill.
            (Job(joined=False), [("popen", 4, False), ("adopt", 12345), "close", "wait"]),
            # Without a job the stage starts as usual.
            (None, [("popen", 0, False), "wait"]),
        ]:
            with self.subTest(job=job and vars(job)):
                calls = []
                with (patch("ci_process.os.name", "nt"), patch("ci_process._windows_job", return_value=job),
                      patch("ci_process.subprocess.Popen", Process)):
                    result = run_logged(["stage"], cwd=self.root, env={}, log=self.log, timeout=5,
                                        progress=lambda *args, **kwargs: None)
                self.assertEqual(result.returncode, 0)
                self.assertEqual(calls, expected)
        # A stage that cannot be resumed is killed rather than left suspended.
        calls = []
        with (patch("ci_process.os.name", "nt"), patch("ci_process._windows_job", return_value=Job(resumes=False)),
              patch("ci_process.subprocess.Popen", Process), self.assertRaises(OSError)):
            run_logged(["stage"], cwd=self.root, env={}, log=self.log, timeout=5, progress=lambda *args, **kwargs: None)
        self.assertEqual(calls, [("popen", 4, False), ("adopt", 12345), "kill", "wait", "close"])

    def test_summary_has_timings_without_child_output_or_error_details(self):
        summary = self.root / "summary.md"
        write_summary({"status": "failed", "error": "synthetic-secret", "steps": [
            {"name": "build", "duration_seconds": 12.34, "exit_code": 7,
             "command": ["synthetic-secret"], "error_type": "OSError"},
            {"name": "bad|stage\n", "duration_seconds": 0, "exit_code": None},
        ]}, summary)
        text = summary.read_text()
        self.assertIn("| build | 12.3 | 7 |", text)
        self.assertIn("| bad?stage? | 0.0 | interrupted |", text)
        self.assertNotIn("synthetic-secret", text)

    def test_windows_cleanup_failure_is_explicit(self):
        for taskkill in [1, OSError("synthetic-secret"), subprocess.TimeoutExpired(["taskkill"], 15)]:
            with self.subTest(taskkill=type(taskkill).__name__):
                cleanup, calls = self.windows_stop(taskkill=taskkill)
                self.assertEqual(cleanup["tree_termination"], "unverified")
                self.assertNotIn("synthetic-secret", json.dumps(cleanup))
                self.assertEqual(calls[-2:], ["kill", "wait"])

    def test_timeout_kills_descendant_tree(self):
        # A direct child, a grandchild whose parent has already exited (invisible to
        # `taskkill /T` and to parent-ID walks) and a process that keeps starting
        # children each append to their own file. Once the stage is stopped, none may
        # grow: that holds however long the stop took, and shows any survivor.
        (self.root / "tree.py").write_text(TREE, encoding="utf-8")
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_stage("import runpy; runpy.run_path('tree.py', run_name='__main__')", timeout=8)
        record = self.events[-1] if self.events else None
        self.assertIn(b"ready", self.log.read_bytes(), record)
        self.assertEqual(record["event"], "cleanup", record)
        self.assertEqual(record["tree_termination"], "confirmed", record)
        if os.name == "nt":
            self.assertIs(record.get("job"), True, record)
        sizes = lambda: {name: (self.root / name).stat().st_size for name in BEATING}
        stopped = sizes()
        self.assertTrue(all(stopped.values()), (stopped, record))
        import time
        time.sleep(1.0)
        self.assertEqual(sizes(), stopped, ("a process of the stopped stage kept running", record))


# The stage for test_timeout_kills_descendant_tree; each role is one process.
BEATING = ["child", "orphan", "late"]
TREE = """import subprocess, sys, time
from pathlib import Path
role = sys.argv[1] if len(sys.argv) > 1 else "root"
start = lambda *arguments: subprocess.Popen([sys.executable, "tree.py", *arguments])
if role == "beat":
    with open(sys.argv[2], "ab", buffering=0) as stream:
        for _ in range(600):
            stream.write(b".")
            time.sleep(0.05)
elif role == "middle":
    start("beat", "orphan")
elif role == "spawner":
    for _ in range(40):
        start("beat", "late")
        time.sleep(0.25)
else:
    start("beat", "child")
    start("middle").wait()
    start("spawner")
    while not all(Path(name).exists() and Path(name).stat().st_size for name in ["child", "orphan", "late"]):
        time.sleep(0.01)
    print("ready", flush=True)
    time.sleep(60)
"""


if __name__ == "__main__":
    unittest.main()
