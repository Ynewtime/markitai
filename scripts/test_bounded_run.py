"""bounded_run: the helper tests' children can't hold a test past their timeout."""
import os
import signal
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

import bounded_run

# Starts `beat` (appends to a file every 50 ms, holding the inherited output),
# says so and then waits for `seconds`.
PARENT = """import subprocess, sys, time
beat = "import sys, time\\nwith open(sys.argv[1], 'ab', buffering=0) as s:\\n    for _ in range(400): s.write(b'.'); time.sleep(0.05)"
child = subprocess.Popen([sys.executable, "-c", beat, sys.argv[1]])
open(sys.argv[2], "w").write(str(child.pid))
print("started", flush=True)
time.sleep(float(sys.argv[3]))
"""


class BoundedRunTests(unittest.TestCase):
    def setUp(self):
        bounded_run.watch(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.beats, self.pid = self.root / "beats", self.root / "pid"

    def parent(self, seconds, timeout):
        return bounded_run.run([sys.executable, "-c", PARENT, str(self.beats), str(self.pid), str(seconds)],
                               text=True, timeout=timeout)

    def assert_beat_stops(self):
        deadline = time.monotonic() + 10
        while not (self.beats.exists() and self.beats.stat().st_size):
            self.assertLess(time.monotonic(), deadline, "the beat never started")
            time.sleep(0.05)
        time.sleep(0.3)  # a kill takes a moment to land
        size = self.beats.stat().st_size
        time.sleep(0.5)
        self.assertEqual(self.beats.stat().st_size, size, "a process the child started kept running")

    def test_a_process_left_holding_the_output_does_not_hold_the_test(self):
        started = time.monotonic()
        result = self.parent(0, timeout=15)
        self.assertLess(time.monotonic() - started, 10)
        self.assertEqual((result.returncode, result.stdout), (0, "started\n"))
        if os.name == "nt":  # No session to clear once the child is gone: the test stops it.
            os.kill(int(self.pid.read_text()), signal.SIGTERM)
        else:
            self.assert_beat_stops()

    def test_a_timeout_stops_the_child_and_what_it_started(self):
        started = time.monotonic()
        with self.assertRaises(subprocess.TimeoutExpired) as raised:
            self.parent(60, timeout=2)
        self.assertLess(time.monotonic() - started, 20)
        self.assertIn("stdout: ...started", str(raised.exception))
        self.assert_beat_stops()

    def test_input_text_and_check(self):
        echo = [sys.executable, "-c",
                "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read().upper() + b'\\r\\n'); sys.exit(3)"]
        result = bounded_run.run(echo, input="ab", text=True)
        self.assertEqual((result.returncode, result.stdout, result.stderr), (3, "AB\n", ""))
        self.assertEqual(bounded_run.run(echo, input=b"ab").stdout, b"AB\r\n")
        with self.assertRaises(subprocess.CalledProcessError):
            bounded_run.output(echo, input=b"ab")
        # Nothing typed: the child reads end of input, never the developer's terminal.
        self.assertEqual(bounded_run.run(echo, text=True).stdout, "\n")


if __name__ == "__main__":
    unittest.main()
