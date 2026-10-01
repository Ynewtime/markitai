"""`import markitai` stays cheap: heavy modules load when a call needs them.

Each check runs in a fresh interpreter, because the test process itself has
long since imported asyncio, dataclasses and the rest.
"""

import ast
import dataclasses
import os
import pickle
import subprocess
import sys
import tempfile
import typing
import unittest
from pathlib import Path

import markitai

# The standard-library modules that made up almost all of the 23 ms
# `import markitai` once took: asyncio about 13 ms, dataclasses (with inspect
# and ast) and json (with re) about 5 and 3 ms, pathlib and typing 1 ms each.
HEAVY = {"asyncio", "dataclasses", "inspect", "ast", "json", "re", "pathlib", "typing", "copy",
         "markitai.config", "markitai._records"}


def run_python(code, root, *, stdin=None):
    environment = dict(os.environ, MARKITAI_HOME=str(Path(root) / "state"))
    run = subprocess.run([sys.executable, "-c", code], env=environment, input=stdin,
                         capture_output=True, timeout=120)
    if run.returncode != 0:
        raise AssertionError(run.stderr.decode(errors="replace")[-4000:])
    return run.stdout.decode()


def new_modules(statements, root):
    """Modules `statements` import on top of an interpreter that has run `site`.

    The probe avoids importing anything itself, so a module it reports was
    imported by the statements.
    """
    code = "\n".join([
        "import sys",
        "before = set(sys.modules)",
        statements,
        "sys.stdout.write(repr(sorted(set(sys.modules) - before)))",
    ])
    return set(ast.literal_eval(run_python(code, root).strip().splitlines()[-1]))


class LazyImportTests(unittest.TestCase):
    def assertNoneLoaded(self, new, names, message=""):
        # Name the offenders only; the full module list drowns the failure.
        self.assertEqual(new & set(names), set(), message or "imported too early")

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "doc.md"
        self.source.write_text("# Lazy\n\nBody 你好\n", encoding="utf-8")

    def test_import_loads_only_the_package_itself(self):
        new = new_modules("import markitai", self.root)
        self.assertGreaterEqual(new, {"markitai", "markitai._native", "markitai.api"})
        self.assertNoneLoaded(new, HEAVY, "import markitai pulled in heavy modules")
        # Nothing but the package's own modules and `__future__`, which its
        # annotations need; a new import here should be a deliberate choice.
        self.assertEqual({name for name in new if not name.startswith("markitai")} - {"__future__"}, set())

    def test_lazy_names_are_listed_resolvable_and_identical(self):
        code = "\n".join([
            "import sys",
            "import markitai",
            "lazy = ('ConversionOutput', 'ConversionUsage', 'OutputProfileName', 'MarkitaiConfig', 'config')",
            "assert all(name in dir(markitai) for name in lazy + ('api', '__version__', 'convert'))",
            "assert 'dataclasses' not in sys.modules and 'markitai.config' not in sys.modules",
            "from markitai import ConversionOutput, ConversionUsage, OutputProfileName",
            "assert 'dataclasses' in sys.modules and 'markitai.config' not in sys.modules",
            "from markitai.api import ConfigModel",
            "assert ConversionOutput is markitai.api.ConversionOutput is markitai.ConversionOutput",
            "assert ConversionUsage is markitai.api.ConversionUsage",
            "assert OutputProfileName is markitai.api.OutputProfileName",
            "assert ConfigModel.__module__ == 'markitai.api'",
            "config = markitai.config",
            "assert config is sys.modules['markitai.config']",
            "assert markitai.MarkitaiConfig is config.MarkitaiConfig",
            "namespace = {}",
            "exec('from markitai import *', namespace)",
            "assert set(markitai.__all__) <= set(namespace), set(markitai.__all__) - set(namespace)",
            "for module in (markitai, markitai.api):",
            "    try:",
            "        module.no_such_name",
            "    except AttributeError as error:",
            "        assert 'no_such_name' in str(error)",
            "    else:",
            "        raise AssertionError('missing attribute did not raise')",
            "print('ok')",
        ])
        self.assertEqual(run_python(code, self.root).strip(), "ok")

    def test_configuration_models_load_with_the_configuration_not_the_package(self):
        for statements in ("import markitai\nfrom markitai import MarkitaiConfig",
                           "import markitai\nimport markitai.config",
                           "from markitai.config import PresetConfig"):
            with self.subTest(statements=statements):
                new = new_modules(statements, self.root)
                self.assertIn("markitai.config", new)
                self.assertNoneLoaded(new, {"asyncio", "markitai._records"})
        # Annotations only: configuration models need no typing import.
        new = new_modules("import markitai\nfrom markitai import MarkitaiConfig", self.root)
        self.assertNoneLoaded(new, {"typing"})

    def test_synchronous_conversion_never_imports_asyncio_or_the_configuration(self):
        new = new_modules("\n".join([
            "import markitai",
            f"result = markitai.convert({str(self.source)!r}, config={{}}, llm=False)",
            "assert 'Body 你好' in result.markdown, result",
            "try:",
            f"    markitai.convert({str(self.root / 'missing.md')!r}, config={{}}, llm=False)",
            "except FileNotFoundError:",
            "    pass",
            "else:",
            "    raise AssertionError('a missing file converted')",
        ]), self.root)
        self.assertNoneLoaded(new, {"asyncio", "markitai.config"})
        self.assertIn("markitai._records", new)

    def test_failures_without_accounting_need_no_records(self):
        new = new_modules("\n".join([
            "import markitai",
            "try:",
            f"    markitai.convert({str(self.root / 'missing.md')!r}, config={{}}, llm=False)",
            "except FileNotFoundError:",
            "    pass",
        ]), self.root)
        self.assertNoneLoaded(new, {"dataclasses", "markitai._records"})

    def test_first_use_from_many_threads_at_once(self):
        # Every lazy path races on a fresh interpreter: conversions, a failing
        # conversion, and attribute access that loads the records or the models.
        code = "\n".join([
            "import sys, threading",
            "import markitai",
            f"source = {str(self.source)!r}",
            f"missing = {str(self.root / 'missing.md')!r}",
            "barrier = threading.Barrier(9)",
            "outcomes = [None] * 9",
            "def work(index):",
            "    try:",
            "        barrier.wait(30)",
            "        if index < 4:",
            "            result = markitai.convert(source, config={}, llm=False)",
            "            assert isinstance(result, markitai.ConversionOutput)",
            "            assert isinstance(result.usage, markitai.api.ConversionUsage)",
            "        elif index < 6:",
            "            try:",
            "                markitai.convert(missing, config={}, llm=False)",
            "            except FileNotFoundError:",
            "                pass",
            "            else:",
            "                raise AssertionError('missing file converted')",
            "        elif index < 8:",
            "            assert markitai.ConversionUsage(requests=2).requests == 2",
            "            assert markitai.MarkitaiConfig().llm.enabled is False",
            "        else:",
            "            assert markitai.config.MarkitaiConfig is markitai.MarkitaiConfig",
            "            assert markitai.api.ConversionOutput is markitai.ConversionOutput",
            "        outcomes[index] = 'ok'",
            "    except BaseException as error:",
            "        outcomes[index] = repr(error)",
            "threads = [threading.Thread(target=work, args=(i,)) for i in range(9)]",
            "for thread in threads: thread.start()",
            "for thread in threads: thread.join()",
            "print(outcomes)",
        ])
        outcomes = ast.literal_eval(run_python(code, self.root).strip().splitlines()[-1])
        self.assertEqual(outcomes, ["ok"] * 9)

    def test_event_loop_guard_and_aconvert_after_a_lazy_import(self):
        # The package is imported first, as a synchronous caller would; the
        # guard must still see a loop that runs once asyncio has been imported.
        code = "\n".join([
            "import markitai",
            "import asyncio",
            f"source = {str(self.source)!r}",
            "async def main():",
            "    try:",
            "        markitai.convert(source, config={}, llm=False)",
            "    except RuntimeError as error:",
            "        assert 'aconvert' in str(error)",
            "    else:",
            "        raise AssertionError('convert ran inside a running loop')",
            "    result = await markitai.aconvert(source, config={}, llm=False)",
            "    assert 'Body 你好' in result.markdown",
            "asyncio.run(main())",
            "print(markitai.convert(source, config={}, llm=False).source == source)",
        ])
        self.assertEqual(run_python(code, self.root).strip(), "True")


class RecordIdentityTests(unittest.TestCase):
    """The moved result records still present as `markitai.api` dataclasses."""

    def test_records_are_dataclasses_of_the_api_module(self):
        for cls in (markitai.ConversionOutput, markitai.ConversionUsage):
            self.assertTrue(dataclasses.is_dataclass(cls))
            self.assertEqual(cls.__module__, "markitai.api")
            self.assertEqual(cls.__qualname__, cls.__name__)
        self.assertEqual(
            [field.name for field in dataclasses.fields(markitai.ConversionOutput)],
            ["source", "markdown", "llm_markdown", "frontmatter", "output_path", "llm_output_path",
             "assets", "screenshots", "images", "usage", "skip_reason", "duration", "warnings"])
        self.assertEqual(
            [field.name for field in dataclasses.fields(markitai.ConversionUsage)],
            ["cost_usd", "requests", "input_tokens", "output_tokens", "by_model"])
        # The string annotations resolve through `markitai.api`, as
        # typing.get_type_hints and pydantic look them up for a dataclass.
        hints = typing.get_type_hints(markitai.ConversionOutput)
        self.assertEqual(hints["output_path"], typing.Optional[Path])
        self.assertIs(hints["usage"], markitai.ConversionUsage)
        self.assertEqual(typing.get_args(markitai.OutputProfileName), ("rag", "obsidian", "okf"))
        self.assertIn("markitai.api.ConversionOutput", repr(markitai.ConversionOutput))

    def test_results_pickle_by_their_api_module_into_a_fresh_interpreter(self):
        original = markitai.ConversionOutput(
            source="a.md", markdown="# A", output_path=Path("out/a.md"),
            usage=markitai.ConversionUsage.from_usage_dict(0.5, {"m": {"requests": 2}}))
        blob = pickle.dumps(original)
        self.assertIn(b"markitai.api", blob)
        self.assertEqual(pickle.loads(blob), original)
        with tempfile.TemporaryDirectory() as root:
            code = "\n".join([
                "import sys",
                "blob = sys.stdin.buffer.read()",
                "import markitai",
                "assert 'dataclasses' not in sys.modules",
                "import pickle",
                "value = pickle.loads(blob)",
                "print(value.markdown, value.usage.requests, value.output_path)",
            ])
            output = run_python(code, root, stdin=blob)
            self.assertEqual(output.strip(), f"# A 2 {Path('out/a.md')}")


if __name__ == "__main__":
    unittest.main()
