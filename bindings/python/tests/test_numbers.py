"""Installed-native Numbers package parity; fixtures retain their upstream MIT bytes."""
import asyncio
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import markitai

FIXTURES = {
    "test-1.numbers": "b9e9772b2d2866c26d773fe46a173c373c7dc1dd3df6cefc7f0253b6ab50d4c3",
    "test-formats.numbers": "9b3ba4b52b2eb3ffd1e05602ab7da954abb2da7f37e68b147725d31777e1cda3",
}


class NumbersTests(unittest.TestCase):
    def setUp(self):
        fixtures = Path(os.environ.get("MARKITAI_TEST_NUMBERS_FIXTURES", "")) if os.environ.get("MARKITAI_TEST_NUMBERS_FIXTURES") else Path(__file__).resolve().parents[3] / "crates/markitai-core/src/formats/numbers/fixtures"
        self.inputs = [(name, (fixtures / name).read_bytes()) for name in FIXTURES]
        for name, data in self.inputs:
            self.assertEqual(hashlib.sha256(data).hexdigest(), FIXTURES[name])
        temp = tempfile.TemporaryDirectory(prefix="markitai-python-numbers-")
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        original_cwd = Path.cwd()
        self.addCleanup(os.chdir, original_cwd)
        os.chdir(self.root)
        env = patch.dict(os.environ, {"MARKITAI_HOME": str(self.root / "state")})
        env.start()
        self.addCleanup(env.stop)
        self.home = os.environ.get("HOME")
        self.options = dict(config={
            "llm": {"enabled": False}, "ocr": {"enabled": False},
            "screenshot": {"enabled": False}, "cache": {"enabled": False},
            "image": {"alt_enabled": False, "desc_enabled": False},
            "history": {"record": False}, "prompts": {"dir": str(self.root / "prompts")},
        }, llm=False, ocr=False, screenshot=False, alt=False, desc=False)

    def bundle(self, name, data):
        zipped = self.root / name
        zipped.write_bytes(data)
        bundle = self.root / ("目录-" + name.upper())
        bundle.mkdir()
        with zipfile.ZipFile(zipped) as archive:
            self.assertEqual(len(archive.namelist()), len(set(archive.namelist())))
            for entry in archive.infolist():
                relative = Path(entry.filename)
                self.assertFalse(relative.is_absolute())
                self.assertNotIn("..", relative.parts)
                self.assertNotIn("\\", entry.filename)
                target = bundle / relative
                if entry.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as output:
                        output.write(archive.read(entry))
        return zipped, bundle

    def test_sync_and_async_directory_match_zip_and_written_result(self):
        for name, data in self.inputs:
            with self.subTest(fixture=name):
                zipped, bundle = self.bundle(name, data)
                baseline = markitai.convert(zipped, **self.options)
                sync = markitai.convert(bundle, **self.options)
                asynchronous = asyncio.run(markitai.aconvert(bundle, **self.options))
                self.assertTrue(baseline.markdown)
                for result in (sync, asynchronous):
                    self.assertIsInstance(result, markitai.ConversionOutput)
                    self.assertEqual(result.source, str(bundle))
                    self.assertEqual(result.markdown, baseline.markdown)
                    self.assertEqual(result.warnings, baseline.warnings)
                    self.assertEqual(result.usage.requests, 0)
                    self.assertIsNone(result.output_path)
                    self.assertEqual(result.assets, [])
                written = markitai.convert(bundle, output_dir=self.root / ("out-" + name), **self.options)
                self.assertEqual(written.output_path.name, bundle.name + ".md")
                self.assertTrue(written.output_path.read_text().endswith(baseline.markdown))
        self.assertEqual(os.environ.get("HOME"), self.home)

    def test_other_directories_old_xml_and_visual_modes_keep_explicit_errors(self):
        _, bundle = self.bundle(*self.inputs[0])
        ordinary, old = self.root / "ordinary", self.root / "old.numbers"
        ordinary.mkdir()
        old.mkdir()
        (old / "index.xml").write_text("<document><table>OLD_XML_MUST_NOT_BE_BATCHED</table></document>")
        for source, overrides in [(ordinary, {}), (old, {}), (bundle, {"ocr": True}), (bundle, {"screenshot": True})]:
            with self.subTest(source=source, overrides=overrides):
                options = dict(self.options, **overrides)
                expected = IsADirectoryError if source == ordinary else markitai.ConversionError
                for invoke in (
                    lambda: markitai.convert(source, **options),
                    lambda: asyncio.run(markitai.aconvert(source, **options)),
                ):
                    with self.assertRaises(expected) as caught:
                        invoke()
                    self.assertIsNone(caught.exception.usage)
                    if source != ordinary:
                        if overrides.get("ocr") or overrides.get("screenshot"):
                            self.assertEqual(str(caught.exception), "Numbers complete-sheet screenshots are not supported by this LibreOffice adapter; native Numbers table reading remains available without screenshot or OCR options")
                        else:
                            self.assertIn("Numbers", str(caught.exception))
                        self.assertEqual(caught.exception.code, "unsupported")


if __name__ == "__main__":
    unittest.main()
