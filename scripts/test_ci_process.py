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

    def test_windows_stop_uses_owned_pid_tree_before_root_kill(self):
        class Process:
            pid = 12345
            def poll(self): return None
            def kill(self): calls.append("kill")
            def wait(self, **kwargs): calls.append("wait")
        calls = []
        def taskkill(*args, **kwargs):
            calls.append(args[0])
            return subprocess.CompletedProcess(args[0], 0)
        with patch("ci_process.os.name", "nt"), patch("ci_process.subprocess.run", side_effect=taskkill):
            result = _stop_tree(Process())
        self.assertEqual(result, {"tree_termination": "confirmed", "taskkill_exit_code": 0})
        self.assertEqual(calls, [["taskkill", "/PID", "12345", "/T", "/F"], "kill", "wait"])

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
        class Process:
            pid = 12345
            def poll(self): return None
            def kill(self): pass
            def wait(self, **kwargs): pass
        for result in [subprocess.CompletedProcess([], 1), OSError("synthetic-secret"),
                       subprocess.TimeoutExpired(["taskkill"], 15)]:
            with self.subTest(result=type(result).__name__):
                kwargs = {"side_effect": result} if isinstance(result, Exception) else {"return_value": result}
                with patch("ci_process.os.name", "nt"), patch("ci_process.subprocess.run", **kwargs):
                    cleanup = _stop_tree(Process())
                self.assertEqual(cleanup["tree_termination"], "unverified")
                self.assertNotIn("synthetic-secret", json.dumps(cleanup))

    def test_timeout_kills_descendant_tree(self):
        # The handshake proves the grandchild is running before the timeout;
        # a surviving grandchild would subsequently create the failure marker.
        script = "import subprocess,sys,time; from pathlib import Path; subprocess.Popen([sys.executable,'-c',\"import time; from pathlib import Path; Path('ready').write_text('ready'); time.sleep(3); Path('leaked').write_text('bad')\"]); exec(\"while not Path('ready').exists(): time.sleep(.01)\"); print('ready',flush=True); time.sleep(20)"
        with self.assertRaises(subprocess.TimeoutExpired):
            self.run_stage(script, timeout=1.5)
        self.assertTrue((self.root / "ready").exists())
        self.assertIn(b"ready", self.log.read_bytes())
        self.assertEqual(self.events[-1]["tree_termination"], "confirmed")
        import time
        # Wait longer than the child delay even if it became ready just
        # before the timeout; slow Windows spawn must not yield a false pass.
        time.sleep(3.2)
        self.assertFalse((self.root / "leaked").exists())


if __name__ == "__main__":
    unittest.main()
