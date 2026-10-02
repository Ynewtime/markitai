"""Real source notices and destructive counterexamples, entirely offline."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import tempfile
import unittest

from portable_attribution import HAYRO_NAMES, PADDLE_NAMES, MAX_FILE, portable_files, stage_portable_files

ROOT = Path(__file__).resolve().parents[1]


class PortableAttributionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        for name in ["hayro", "paddleocr"]:
            shutil.copytree(ROOT / "licenses" / name, self.root / "source/licenses" / name)
        self.source = self.root / "source"

    def test_all_nine_real_notices_keep_their_original_bytes(self):
        files = portable_files(self.source)
        expected = {"licenses/hayro/" + name for name in HAYRO_NAMES} | {"licenses/paddleocr/" + name for name in PADDLE_NAMES}
        self.assertEqual(set(files), expected)
        record = stage_portable_files(self.source, self.root / "package")
        self.assertEqual(set(record), expected)
        for name, content in files.items():
            self.assertEqual(content, (ROOT / name).read_bytes())
            self.assertEqual((self.root / "package" / name).read_bytes(), content)
            self.assertEqual(record[name], {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()})
        self.assertFalse(any(name.endswith(".onnx") for name in files))
        with self.assertRaises(FileExistsError):
            stage_portable_files(self.source, self.root / "package")

    def test_each_missing_notice_is_rejected(self):
        for relative in portable_files(self.source):
            path = self.source / relative
            content = path.read_bytes()
            path.unlink()
            with self.assertRaises(FileNotFoundError):
                portable_files(self.source)
            path.write_bytes(content)

    def test_same_length_license_damage_is_rejected(self):
        for name in ["PaddleOCR-LICENSE", "RapidOCR-LICENSE"]:
            path = self.source / "licenses/paddleocr" / name
            original = path.read_bytes()
            path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            with self.assertRaisesRegex(RuntimeError, "bytes differ"):
                portable_files(self.source)
            path.write_bytes(original)

    def test_manifest_cannot_change_the_license_inventory(self):
        path = self.source / "licenses/paddleocr/provenance.json"
        original = json.loads(path.read_bytes())
        for replacement in [[], original["licenses"] * 2, [{"file": "../outside"}], [{"file": []}]]:
            changed = dict(original, licenses=replacement)
            path.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(RuntimeError):
                portable_files(self.source)

    def test_oversized_and_nonregular_notices_are_rejected(self):
        path = self.source / "licenses/hayro/CGATS_LICENSE.txt"
        path.write_bytes(b"x" * (MAX_FILE + 1))
        with self.assertRaisesRegex(RuntimeError, "bounded regular"):
            portable_files(self.source)
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(RuntimeError, "bounded regular"):
            portable_files(self.source)

    def test_redirected_file_and_directory_are_rejected(self):
        path = self.source / "licenses/hayro/CMAP_LICENSE.txt"
        external = self.root / "outside"
        external.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(external)
        except OSError as error:
            self.skipTest(f"This host cannot create a test symlink: {error}")
        with self.assertRaisesRegex(RuntimeError, "bounded regular"):
            portable_files(self.source)
        path.unlink()
        path.write_bytes(external.read_bytes())
        shutil.move(self.source / "licenses/hayro", self.root / "external-notices")
        (self.source / "licenses/hayro").symlink_to(self.root / "external-notices", target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "ordinary source directories"):
            portable_files(self.source)
        self.assertEqual(external.read_bytes(), (ROOT / "licenses/hayro/CMAP_LICENSE.txt").read_bytes())


if __name__ == "__main__":
    unittest.main()
