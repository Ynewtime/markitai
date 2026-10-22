"""Python markitai's error and shutdown helpers, kept by the binding."""

import asyncio
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import markitai
import markitai.api


class CompatibilityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Every child runs with its own home, user state and language.
        self.environment = dict(os.environ, HOME=str(self.root / "user"),
                                MARKITAI_HOME=str(self.root / "state"), MARKITAI_LANG="en")
        for name in ("USERPROFILE", "APPDATA", "LOCALAPPDATA"):
            if name in os.environ:
                self.environment[name] = str(self.root / "user")

    def run_python(self, code, *arguments):
        return subprocess.run([sys.executable, "-c", code, *arguments], env=self.environment,
                              cwd=self.root, capture_output=True, timeout=120)

    def test_a_bad_configuration_raises_config_file_error_a_value_error(self):
        self.assertTrue(issubclass(markitai.ConfigFileError, ValueError))
        self.assertIs(markitai.api.ConfigFileError, markitai.ConfigFileError)
        self.assertIn("ConfigFileError", markitai.__all__)
        source = self.root / "note.md"
        source.write_text("# Note\n", encoding="utf-8")
        with self.assertRaises(markitai.ConfigFileError) as caught:
            markitai.convert(source, config={"output": {"on_conflict": "sideways"}}, llm=False)
        self.assertEqual(caught.exception.code, "config_error")
        self.assertIn("on_conflict", str(caught.exception))

    def test_close_shared_clients_is_an_awaitable_no_op(self):
        from markitai.fetch import close_shared_clients

        self.assertIsNone(asyncio.run(close_shared_clients()))

    def test_finalize_process_flushes_and_exits_with_the_code(self):
        run = self.run_python("import sys\nfrom markitai.utils.shutdown import finalize_process\n"
                              "print('written', end='')\nsys.stderr.write('noted')\n"
                              "finalize_process(3)\nprint('never')")
        self.assertEqual((run.returncode, run.stdout, run.stderr), (3, b"written", b"noted"))


if __name__ == "__main__":
    unittest.main()
