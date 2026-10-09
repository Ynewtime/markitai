"""Offline attribution counterexamples; no build, npm, installation or providers."""
from decimal import Decimal
from pathlib import Path
import base64
import csv
import hashlib
import io
import json
import tarfile
import tempfile
import unittest
import zipfile

from ci_packages import supplement_wheel_licenses, verify_node_licenses
from pricing_attribution import NAMES, pricing_files, stage_pricing_files


class PricingAttributionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        directory = self.source / "licenses/pricing"
        directory.mkdir(parents=True)
        # Authored stand-ins test byte transport, never substitute a real notice.
        license_text = b"Authored fixture notice\r\nCopyright fixture\r\n"
        rows = b'{"fixture-model":{"input_cost_per_token":0.000001}}\n'
        (directory / "LiteLLM-LICENSE").write_bytes(license_text)
        (directory / "source-rows.json").write_bytes(rows)
        (directory / "README.md").write_text("Authored provenance fixture\n")
        (directory / "provenance.json").write_text(json.dumps({
            "schema_version":1, "license_file":"LiteLLM-LICENSE",
            "license_sha256":hashlib.sha256(license_text).hexdigest(),
            "source_rows_sha256":hashlib.sha256(rows).hexdigest(),
            "rows":[{"model":"fixture-model"}],
        }))
        self.directory = directory

    def test_exact_notice_rows_and_provenance_are_staged_independently(self):
        expected = pricing_files(self.source)
        target = self.root / "package"
        record = stage_pricing_files(self.source, target)
        self.assertEqual(len(record), 4)
        for path, content in expected.items():
            self.assertEqual((target / path).read_bytes(), content)
            self.assertEqual(record[path], {"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()})
        self.assertEqual(pricing_files(self.source), expected)
        with self.assertRaises(FileExistsError):
            stage_pricing_files(self.source, target)

    def test_corrupt_notice_or_rows_and_symlink_are_rejected(self):
        for name in ["LiteLLM-LICENSE", "source-rows.json"]:
            path = self.directory / name
            original = path.read_bytes()
            path.write_bytes(original + b"altered")
            with self.assertRaisesRegex(RuntimeError, "digest"):
                pricing_files(self.source)
            path.write_bytes(original)
        path = self.directory / "README.md"
        original = path.read_bytes()
        elsewhere = self.root / "elsewhere"
        elsewhere.write_bytes(original)
        path.unlink()
        try:
            path.symlink_to(elsewhere)
        except (OSError, NotImplementedError):
            self.skipTest("host cannot create symlinks")
        with self.assertRaisesRegex(RuntimeError, "regular"):
            pricing_files(self.source)

    def test_npm_archive_requires_each_exact_nested_source_file(self):
        expected = pricing_files(self.source)
        for mode in ["complete", "missing", "wrong", "duplicate"]:
            package = self.root / (mode + ".tgz")
            with tarfile.open(package, "w:gz") as archive:
                for index, (name, content) in enumerate(expected.items()):
                    if mode == "missing" and index == 0:
                        continue
                    value = content + b"bad" if mode == "wrong" and index == 0 else content
                    for _ in range(2 if mode == "duplicate" and index == 0 else 1):
                        member = tarfile.TarInfo("package/" + name)
                        member.size = len(value)
                        archive.addfile(member, io.BytesIO(value))
            if mode == "complete":
                verify_node_licenses(package, expected)
            else:
                with self.assertRaises(RuntimeError):
                    verify_node_licenses(package, expected)

    def test_wheel_preserves_nested_paths_and_rebuilds_record_hashes(self):
        source = self.root / "source.whl"
        target = self.root / "target.whl"
        expected = pricing_files(self.source)
        expected["vendor/web/provenance.json"] = b'{"independent":"web"}\n'
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("markitai/__init__.py", "fixture\n")
            archive.writestr("markitai-1.dist-info/RECORD", "markitai/__init__.py,,\nmarkitai-1.dist-info/RECORD,,\n")
        supplement_wheel_licenses(source, target, expected)
        with zipfile.ZipFile(target) as archive:
            records = {row[0]:row[1:] for row in csv.reader(io.StringIO(archive.read("markitai-1.dist-info/RECORD").decode()))}
            for name, content in expected.items():
                path = "markitai-1.dist-info/licenses/" + name
                self.assertEqual(archive.read(path), content)
                digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).rstrip(b"=").decode()
                self.assertEqual(records[path], ["sha256=" + digest, str(len(content))])
            self.assertNotEqual(archive.read("markitai-1.dist-info/licenses/licenses/pricing/provenance.json"), archive.read("markitai-1.dist-info/licenses/vendor/web/provenance.json"))
        self.assertEqual(set(pricing_files(self.source)), {"licenses/pricing/" + name for name in NAMES})


class RepositoryPricingRowsTests(unittest.TestCase):
    def test_each_retained_object_matches_its_recorded_source_range_and_rates(self):
        directory = Path(__file__).resolve().parents[1] / "licenses/pricing"
        provenance = json.loads((directory / "provenance.json").read_bytes())
        text = (directory / "source-rows.json").read_bytes().decode()
        # Recover each retained object's exact bytes, not a re-serialization.
        decoder, objects, index = json.JSONDecoder(), {}, text.index("{") + 1
        while (index := text.find('"', index)) != -1:
            key, index = json.decoder.scanstring(text, index + 1)
            start = text.index("{", index)
            _, index = decoder.raw_decode(text, start)
            objects[key] = text[start:index].encode()
        self.assertEqual(provenance["snapshot"], "litellm-%s-selected-%s" % (provenance["version"], provenance["captured_date"]))
        self.assertEqual(list(objects), [row["model"] for row in provenance["rows"]])
        for row in provenance["rows"]:
            raw = objects[row["model"]]
            self.assertEqual(len(raw), row["source_byte_end"] - row["source_byte_start"], row["model"])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), row["source_object_sha256"], row["model"])
            source = json.loads(raw, parse_float=Decimal)
            self.assertEqual(set(row["usd_per_token_decimal"]), set(row["picodollars_per_token"]))
            for field, decimal in row["usd_per_token_decimal"].items():
                self.assertEqual(str(source[field]), decimal, (row["model"], field))
                self.assertEqual(Decimal(decimal) * 10**12, row["picodollars_per_token"][field], (row["model"], field))


if __name__ == "__main__":
    unittest.main()
