"""Tailwind palette attribution: exact notice transport and embedded-value drift; no builds."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from tailwind_attribution import EMBEDDED, NAMES, stage_tailwind_files, tailwind_files
from ci_packages import package_attribution

ROOT = Path(__file__).resolve().parents[1]
SHADES = ["50", "100", "200", "300", "400", "500", "600", "700", "800", "900", "950"]


class TailwindAttributionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.directory = self.source / "licenses/tailwind"
        self.directory.mkdir(parents=True)
        license = b"Authored MIT license transport fixture\n"
        values = [f'"#{index:06x}"' for index in range(11)]
        self.table = (
            "const PALETTE: &[(&str, [&str; 11])] = &[\n"
            f'    ("slate", [{", ".join(values)}]),\n'
            f'    ("rose", [{", ".join(reversed(values))}]),\n'
            "];\n"
        )
        lines = "".join(f"slate {shade} #{index:06x}\n" for index, shade in enumerate(SHADES))
        lines += "".join(f"rose {shade} #{10 - index:06x}\n" for index, shade in enumerate(SHADES))
        (self.directory / "LICENSE").write_bytes(license)
        (self.directory / "README.md").write_bytes(b"Authored palette notice\n")
        (self.directory / "provenance.json").write_text(json.dumps({
            "schema_version": 1, "license_file": "LICENSE", "embedded_in": EMBEDDED,
            "license_sha256": hashlib.sha256(license).hexdigest(), "palettes": ["slate", "rose"],
            "shades": SHADES, "palette_sha256": hashlib.sha256(lines.encode()).hexdigest(),
        }))
        self.embedded = self.source / EMBEDDED
        self.embedded.parent.mkdir(parents=True)
        self.embedded.write_text("// reader\n" + self.table)

    def test_staging_preserves_the_complete_notice_and_refuses_overwrite(self):
        expected = tailwind_files(self.source)
        self.assertEqual(set(expected), {"licenses/tailwind/" + name for name in NAMES})
        target = self.root / "go"
        record = stage_tailwind_files(self.source, target)
        for name, content in expected.items():
            self.assertEqual((target / name).read_bytes(), content)
            self.assertEqual(record[name]["sha256"], hashlib.sha256(content).hexdigest())
        with self.assertRaises(FileExistsError):
            stage_tailwind_files(self.source, target)

    def test_a_changed_license_manifest_or_embedded_palette_cannot_ship(self):
        for path, change in [
            (self.directory / "LICENSE", lambda text: text + "changed"),
            (self.embedded, lambda text: text.replace('"#000001"', '"#000002"', 1)),
            (self.embedded, lambda text: text.replace('("rose"', '("pink"')),
            (self.embedded, lambda text: text.replace('"#00000a"]', ']', 1)),
            (self.embedded, lambda text: text.replace("const PALETTE", "const COLOURS")),
            (self.directory / "provenance.json", lambda text: text.replace('"embedded_in"', '"elsewhere"')),
        ]:
            # Bytes, not text: a text write on Windows would restore LF files as CRLF.
            old = path.read_bytes()
            path.write_bytes(change(old.decode("utf-8")).encode("utf-8"))
            with self.assertRaises(RuntimeError):
                tailwind_files(self.source)
            path.write_bytes(old)
        tailwind_files(self.source)

    @unittest.skipIf(os.name == "nt", "symlink privileges are host-dependent")
    def test_a_symlink_cannot_redirect_the_notice_or_the_embedded_table(self):
        for path in [self.directory / "LICENSE", self.embedded]:
            outside = self.root / ("outside-" + path.name)
            shutil.copyfile(path, outside)
            path.unlink()
            path.symlink_to(outside)
            with self.assertRaisesRegex(RuntimeError, "regular"):
                tailwind_files(self.source)
            path.unlink()
            shutil.copyfile(outside, path)

    def test_the_repository_palette_matches_its_tailwind_source(self):
        files = tailwind_files(ROOT)
        provenance = json.loads(files["licenses/tailwind/provenance.json"])
        self.assertEqual((provenance["package"], provenance["version"]), ("tailwindcss", "3.4.19"))
        self.assertEqual(len(provenance["palettes"]) * len(provenance["shades"]), 242)
        self.assertIn(b"Copyright (c) Tailwind Labs, Inc.", files["licenses/tailwind/LICENSE"])
        common = package_attribution(ROOT)
        for name, content in files.items():
            self.assertEqual(common[name], content)


if __name__ == "__main__":
    unittest.main()
