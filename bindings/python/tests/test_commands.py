"""The wheel's commands and extras, which replace Python markitai's.

The metadata checks read the installed distribution, so they run against an
installed wheel and are skipped from a source checkout.
"""

import importlib.metadata
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path

import markitai

COMMANDS = {
    "markitai": "markitai._cli:markitai",
    "mkai": "markitai._cli:mkai",
    "markitai-mcp": "markitai._cli:markitai_mcp",
}


def installed():
    try:
        return importlib.metadata.distribution("markitai")
    except importlib.metadata.PackageNotFoundError:
        return None


class CommandTests(unittest.TestCase):
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

    def test_the_commands_run_the_native_program_by_their_names(self):
        version = markitai.__version__
        for name, function in (("markitai", "markitai"), ("mkai", "mkai"),
                               ("markitai-mcp", "markitai_mcp")):
            run = self.run_python(f"import sys\nsys.argv[1:] = ['--version']\n"
                                  f"from markitai._cli import {function}\n{function}()\n"
                                  "raise SystemExit('the program returned')")
            self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
            self.assertIn(version, run.stdout.decode())
        # The MCP name is the MCP server, its usage under that name.
        run = self.run_python("import sys\nsys.argv[1:] = ['--help']\n"
                              "from markitai._cli import markitai_mcp\nmarkitai_mcp()")
        self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
        self.assertIn("markitai-mcp [OPTIONS]", run.stdout.decode())
        # An unknown option is the program's usage error, status 2.
        run = self.run_python("import sys\nsys.argv[1:] = ['--no-such-option']\n"
                              "from markitai._cli import markitai\nmarkitai()")
        self.assertEqual(run.returncode, 2)
        self.assertIn("--no-such-option", run.stderr.decode(errors="replace"))

    def test_the_program_converts_a_file(self):
        source = self.root / "note.md"
        source.write_text("# Note\n\nWords.\n", encoding="utf-8")
        output = self.root / "out"
        run = self.run_python("import sys\nsys.argv[1:] = sys.argv[1:]\n"
                              "from markitai._cli import markitai\nmarkitai()",
                              str(source), "-o", str(output))
        self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
        self.assertIn("Words.", (output / "note.md.md").read_text(encoding="utf-8"))

    @unittest.skipIf(installed() is None, "the package is not installed")
    def test_the_wheel_declares_the_commands_and_python_markitai_extras(self):
        distribution = installed()
        scripts = {entry.name: entry.value for entry in distribution.entry_points
                   if entry.group == "console_scripts"}
        self.assertEqual(scripts, COMMANDS)
        extras = set(distribution.metadata.get_all("Provides-Extra") or [])
        self.assertEqual({"all", "browser", "claude-agent", "copilot", "extra-fetch", "heif",
                          "legacy", "mcp", "ocr", "serve", "svg"}, extras)
        self.assertEqual([requirement for requirement in distribution.requires or []
                          if "extra ==" not in requirement], [])

    @unittest.skipIf(installed() is None, "the package is not installed")
    def test_the_installed_commands_run(self):
        folder = Path(sysconfig.get_path("scripts"))
        for name in COMMANDS:
            command = shutil.which(name, path=str(folder))
            if command is None:
                self.skipTest(f"{name} is not in {folder}")
            run = subprocess.run([command, "--version"], env=self.environment, cwd=self.root,
                                 capture_output=True, timeout=120)
            self.assertEqual(run.returncode, 0, run.stderr.decode(errors="replace"))
            self.assertIn(markitai.__version__, run.stdout.decode())


if __name__ == "__main__":
    unittest.main()
