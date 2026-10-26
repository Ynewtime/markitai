"""Offline license-provenance counterexamples; never compile or download."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
import unittest
import tarfile
import zipfile

from license_overlay import DISK, MAX_FILE, MAX_FILES, _relative, stage_overlay, upstream_files
from ci_packages import supplement_wheel_licenses, verify_node_licenses
from package_go_static import bundle_licenses, inventory, package_archive, unpack_verified

VENDOR = Path(__file__).resolve().parents[1] / "licenses/upstream"
# Windows refuses symlinks to accounts without Developer Mode or administrator rights.
ERROR_PRIVILEGE_NOT_HELD = 1314
REFRESHED = {"selectors@0.41.0", "napi@3.15.0", "napi-derive@3.6.13", "napi-sys@3.4.1",
             "napi-derive-backend@6.1.6"}
# Lock-refresh tuples of 2026-10-23 (brotli 9, macOS objc2, Android verifier).
LOCK_REFRESH = {"alloc-stdlib@0.3.0", "objc2@0.6.5", "jni@0.22.4", "jni-macros@0.22.4",
                "jni-sys-macros@0.4.1", "rustls-platform-verifier-android@0.2.0"}


class MemoryFiles:
    """`license_overlay.Disk` over a dict of `{absolute Path: bytes}`.

    The counterexamples stage the overlay hundreds of times; on Windows every
    file operation is slow, so they read and change the fixture in memory. The
    disk walk itself (symlinks, special files) is tested on a real tree.
    """

    def __init__(self, store):
        self.store = store

    def under(self, root):
        prefix = os.path.join(str(root), "")
        return [path for path in self.store if str(path).startswith(prefix)]

    def exists(self, path):
        return path in self.store or bool(self.under(path))

    def listing(self, root):
        found = self.under(root)
        if not found:
            raise RuntimeError("License evidence root must be a real directory")
        if len(found) > MAX_FILES * 4:
            raise RuntimeError("License evidence must not contain symlinks or excessive entries")
        start = len(os.path.join(str(root), ""))
        return {str(path)[start:].replace(os.sep, "/"): (path, len(self.store[path])) for path in found}

    def read(self, key):
        return self.store[key][:MAX_FILE + 1]

    def package(self, directory, name):
        path = directory / str(_relative(name))
        if path not in self.store:
            raise FileNotFoundError(path)
        if len(self.store[path]) > MAX_FILE:
            raise RuntimeError("License evidence must be a bounded regular file")
        return self.store[path]

    def write_new(self, path, raw):
        if path in self.store:
            raise FileExistsError(path)
        self.store[path] = raw


class LicenseOverlayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The fixture (the evidence and the registry sources), read once into
        # memory; each test changes its own copy (see MemoryFiles).
        contents = {}
        for path in VENDOR.rglob("*"):
            if path.is_file():
                contents[PurePosixPath("repository/licenses/upstream", path.relative_to(VENDOR).as_posix())] = path.read_bytes()
        manifest = json.loads((VENDOR / "manifest.json").read_text(encoding="utf-8"))
        for item in manifest["packages"]:
            label = f"{item['name']}-{item['version']}"
            for name in [".cargo_vcs_info.json", "Cargo.toml.orig"]:
                contents[PurePosixPath("registry", label, name)] = (VENDOR / "local-evidence" / label / name).read_bytes()
            if item["name"] in {"nom-language", "tract-extra"}:
                with tarfile.open(VENDOR / "source-archives" / (label + ".crate")) as archive:
                    for member in archive:
                        name = member.name.removeprefix(label + "/")
                        if name.startswith("src/") and name.endswith(".rs"):
                            contents[PurePosixPath("registry", label, name)] = archive.extractfile(member).read()
        cls.contents = contents

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.files = MemoryFiles({self.root / name: raw for name, raw in self.contents.items()})
        self.repository = self.root / "repository"
        self.vendor = self.repository / "licenses/upstream"
        self.manifest = json.loads(self.read_text(self.vendor / "manifest.json"))
        self.packages = []
        self.node_packages = []
        self.heif_packages = []
        self.qpp_packages = []
        self.avif_packages = []
        # Versions added after the dependency majors of 2026-10-22.
        self.refresh_packages = []
        self.lock_packages = []
        for item in self.manifest["packages"]:
            source = self.root / "registry" / f"{item['name']}-{item['version']}"
            if item["id"].rsplit("#", 1)[-1] in LOCK_REFRESH:
                target_packages = self.lock_packages
            elif item["id"].rsplit("#", 1)[-1] in REFRESHED:
                target_packages = self.refresh_packages
            elif item["name"] in {"napi", "napi-build", "napi-derive", "napi-sys"}:
                target_packages = self.node_packages
            elif item["name"].startswith("heifer"):
                target_packages = self.heif_packages
            elif item["name"].startswith("qppocr"):
                target_packages = self.qpp_packages
            elif item["name"] == "atomig-macro":
                target_packages = self.avif_packages
            else:
                target_packages = self.packages
            target_packages.append({"id": item["id"], "name": item["name"], "version": item["version"],
                                  "source": "registry+fixture", "license": item["declared_license"],
                                  "license_file": None, "repository": item["repository"],
                                  "manifest_path": str(source / "Cargo.toml")})
        self.destination = self.root / "module/licenses/upstream"

    def stage(self, packages):
        """stage_overlay on this test's fixture, in memory."""
        return stage_overlay(self.vendor, self.destination, packages, files=self.files)

    def write(self, path, content):
        self.files.store[Path(path)] = content.encode("utf-8") if isinstance(content, str) else content

    def read(self, path):
        """The fixture's bytes, or a file a disk test's packaging wrote."""
        path = Path(path)
        return self.files.store[path] if path in self.files.store else path.read_bytes()

    def read_text(self, path):
        return self.read(path).decode("utf-8")

    def remove(self, path):
        del self.files.store[Path(path)]

    def exists(self, path):
        return self.files.exists(Path(path)) or Path(path).exists()

    def remove_tree(self, root):
        for path in self.files.under(Path(root)):
            del self.files.store[path]

    def materialize(self):
        """Write the fixture to disk, for the tests of the disk walk and of packaging."""
        for path, raw in self.files.store.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)

    def seal(self, name):
        """Rehash one changed test input to exercise semantic checks separately."""
        raw = self.read(self.vendor / name)
        path = self.vendor / "inventory.json"
        value = json.loads(self.read_text(path))
        row = next(entry for entry in value["files"] if entry["path"] == name)
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.write(path, json.dumps(value))

    def save_manifest(self):
        self.write((self.vendor / "manifest.json"), json.dumps(self.manifest))
        self.seal("manifest.json")

    def test_complete_text_and_notice_are_distinct_and_all_provenance_is_archived(self):
        ffi = self.repository / "crates/ffi"
        ffi.mkdir(parents=True)
        for name in ["LICENSE", "NOTICE"]:
            self.write((self.repository / name), "Fixture root notice\n")
        root_package = {"id": "fixture-ffi", "name": "markitai-ffi", "version": "1",
                        "source": None, "license": "MIT", "manifest_path": str(ffi / "Cargo.toml")}
        metadata = {"packages": [root_package] + self.packages,
                    "resolve": {"nodes": [{"id": "fixture-ffi", "deps": [{"pkg": p["id"]} for p in self.packages]}]
                                + [{"id": p["id"], "deps": []} for p in self.packages]}}
        sysroot = self.root / "rust"
        toolchain = sysroot / "share/doc/rust/LICENSE-test"
        toolchain.parent.mkdir(parents=True)
        self.write(toolchain, "Fixture toolchain notice\n")
        self.destination.parent.parent.mkdir(parents=True)
        self.materialize()  # packaging reads the repository on disk
        record = bundle_licenses(metadata, self.destination.parent, self.repository, sysroot)
        self.assertEqual(record["legal_review"], "not_performed")
        self.assertEqual(record["upstream_overlay"]["complete_text_packages"], 17)
        self.assertEqual(record["upstream_overlay"]["notice_only_packages"], 0)
        self.assertEqual(record["unresolved"], [])
        self.assertEqual(record["upstream_overlay"]["historical_text_packages"], 3)
        self.assertEqual(record["upstream_overlay"]["legal_review"], "not_performed")
        historical = [p for p in record["dependencies"]
                      if p.get("upstream_evidence", {}).get("historical_provenance")]
        self.assertEqual({p["name"] for p in historical}, {"objc2", "objc2-encode", "objc2-foundation"})
        copied = [text for p in record["dependencies"] for text in p["texts"] if text.get("origin") == "verified_upstream_overlay"]
        self.assertEqual(len(copied), 26)
        for text in copied:
            raw = self.read(self.destination.parent.parent / text["path"])
            self.assertEqual(raw, self.read(text["source"]))
            self.assertEqual(hashlib.sha256(raw).hexdigest(), text["sha256"])
        self.assertEqual(inventory(self.vendor), inventory(self.destination))
        archive = self.root / "licenses.tgz"
        expected = package_archive(self.destination.parent.parent, archive)
        unpack_verified(archive, self.root / "installed", expected)
        self.assertEqual(inventory(self.root / "installed/licenses/upstream"), inventory(self.vendor))

    def test_current_summary_is_derived_while_collection_history_stays_separate(self):
        self.assertEqual(self.manifest["unresolved_full_license_text"], [])
        historical = self.manifest["historical_collection"]["unresolved_full_license_text"]
        self.assertEqual(len(historical), 9)
        self.assertEqual(self.manifest["historical_collection"]["source"], "input-manifest.json")
        result = self.stage(self.packages)
        self.assertEqual(result["record"]["complete_text_packages"], 17)
        self.assertEqual(result["record"]["historical_text_packages"], 3)
        self.assertEqual(result["record"]["legal_review"], "not_performed")
        copied = json.loads(self.read_text(self.destination / "manifest.json"))
        self.assertEqual(copied["historical_collection"]["unresolved_full_license_text"], historical)
        self.assertEqual(copied["unresolved_full_license_text"], [])

    def test_resealed_stale_unresolved_or_count_summary_fails_before_copying(self):
        original = json.loads(json.dumps(self.manifest))
        variants = [("unresolved", None), ("historical_text_packages", 0),
                    ("full_license_text_packages", 14), ("notice_only_packages", 3),
                    ("requested_packages", 14), ("exact_commit_manifest_matches", 14),
                    ("exact_commit_manifest_matches", 17), ("reviewed_publication_version_stamps", 0),
                    ("overlay_files", 24), ("source_archives", True)]
        for field, value in variants:
            with self.subTest(field=field):
                self.manifest = json.loads(json.dumps(original))
                if field == "unresolved":
                    self.manifest["unresolved_full_license_text"] = self.manifest["historical_collection"]["unresolved_full_license_text"]
                else:
                    self.manifest["summary"][field] = value
                self.save_manifest()
                with self.assertRaisesRegex(RuntimeError, "summary differs"):
                    self.stage(self.packages)
                self.assertFalse(self.exists(self.destination))

    def test_a_new_version_cannot_inherit_old_version_license_evidence(self):
        future = dict(self.packages[0], id="registry+fixture#alloc-stdlib@99.0", version="99.0")
        result = self.stage([future])
        self.assertEqual(result["packages"], {})
        self.assertEqual(result["record"]["matched_packages"], 0)

    def test_current_package_vcs_original_and_license_must_match(self):
        for kind in ["vcs", "original", "license"]:
            with self.subTest(kind=kind):
                package = dict(self.packages[0])
                directory = Path(package["manifest_path"]).parent
                name = ".cargo_vcs_info.json" if kind == "vcs" else "Cargo.toml.orig"
                previous = self.read(directory / name)
                if kind == "license":
                    package["license"] = "MIT"
                else:
                    self.write((directory / name), previous + b"\n")
                with self.assertRaisesRegex(RuntimeError, "differs"):
                    self.stage([package])
                self.assertFalse(self.exists(self.destination))
                self.write((directory / name), previous)

    def test_changed_asset_bytes_fail_before_copying(self):
        asset = self.manifest["packages"][0]["assets"][0]["path"]
        self.write((self.vendor / asset), b"substituted text")
        with self.assertRaisesRegex(RuntimeError, "identity differs"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))
        self.seal(asset)
        with self.assertRaisesRegex(RuntimeError, "identity differs"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_notice_classification_and_upstream_url_are_checked(self):
        notice = next(p for p in self.manifest["packages"] if p["name"] == "objc2")
        notice["complete_license_options_present"] = []
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "cannot claim"):
            self.stage(self.packages)
        notice["complete_license_options_present"] = ["MIT"]
        notice["assets"][0]["source_url"] = "https://raw.githubusercontent.com/example/other/main/LICENSE"
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "URL or hash"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_notice_extraction_must_equal_exact_source_byte_range(self):
        selector = next(p for p in self.manifest["packages"] if p["name"] == "selectors")
        selector["assets"][0]["source_byte_range"]["end_exclusive"] -= 1
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "source bytes"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_inventory_rejects_symlinks(self):
        self.materialize()  # the disk walk itself
        source = self.vendor / "README.md"
        outside = self.root / "readme"
        source.rename(outside)
        try:
            source.symlink_to(outside)
        except OSError as error:
            if getattr(error, "winerror", None) != ERROR_PRIVILEGE_NOT_HELD:
                raise
            self.skipTest("this Windows account may not create symlinks")
        with self.assertRaisesRegex(RuntimeError, "symlinks"):
            stage_overlay(self.vendor, self.destination, self.packages, files=DISK)
        self.assertFalse(self.destination.exists())

    def test_inventory_rejects_traversal_and_unlisted_files(self):
        self.write((self.vendor / "unreviewed.txt"), "extra")
        with self.assertRaisesRegex(RuntimeError, "not exact"):
            self.stage(self.packages)
        self.remove(self.vendor / "unreviewed.txt")
        value = json.loads(self.read_text(self.vendor / "inventory.json"))
        value["files"][0]["path"] = "../escape"
        self.write((self.vendor / "inventory.json"), json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_boolean_sizes_and_duplicate_manifest_keys_are_not_accepted(self):
        value = json.loads(self.read_text(self.vendor / "inventory.json"))
        previous = json.dumps(value)
        value["files"][0]["bytes"] = True
        self.write((self.vendor / "inventory.json"), json.dumps(value))
        with self.assertRaisesRegex(RuntimeError, "identity"):
            self.stage(self.packages)
        self.write((self.vendor / "inventory.json"), previous)
        raw = self.read_text(self.vendor / "manifest.json")
        self.write((self.vendor / "manifest.json"), raw.replace('"schema": 1,', '"schema": 1, "schema": 1,', 1))
        self.seal("manifest.json")
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))


    def authority(self):
        entry = next(p for p in self.manifest["packages"] if p["name"] == "dispatch2")
        asset = next(a for a in entry["assets"] if a.get("source_kind") == "linked_authority")
        return entry, asset

    def test_authority_url_and_license_option_are_not_general_allowlists(self):
        entry, asset = self.authority()
        for field, value in [("source_url", "https://www.apache.org.example/LICENSE-2.0.txt"),
                             ("source_url", "https://www.apache.org/licenses/LICENSE-2.0.txt?variant=other"),
                             ("license_option", "MIT"),
                             ("source_kind", "external_url"),
                             ("notice_asset", "overlay/licenses/selectors-0.38.0/NOTICE-MPL-2.0.txt")]:
            with self.subTest(field=field, value=value):
                old = asset.get(field)
                asset[field] = value
                self.save_manifest()
                with self.assertRaises(RuntimeError):
                    self.stage(self.packages)
                self.assertFalse(self.exists(self.destination))
                asset[field] = old
        self.save_manifest()

    def test_resealed_authority_substitution_still_fails_reviewed_digest(self):
        _, asset = self.authority()
        raw = self.read(self.vendor / asset["source_path"]) + b"\nSubstituted terms\n"
        for name in [asset["source_path"], asset["path"]]:
            self.write((self.vendor / name), raw)
            self.seal(name)
        asset.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                     source_sha256=hashlib.sha256(raw).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "reviewed text hash"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_authority_requires_the_link_in_the_verified_version_notice(self):
        entry, _ = self.authority()
        notice = entry["assets"][0]
        raw = self.read(self.vendor / notice["source_path"]).replace(
            b"[Apache-2.0]: https://www.apache.org/licenses/LICENSE-2.0",
            b"[Apache-2.0]: https://example.invalid/unreviewed")
        for name in [notice["source_path"], notice["path"]]:
            self.write((self.vendor / name), raw)
            self.seal(name)
        notice.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                      source_sha256=hashlib.sha256(raw).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "does not link"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_historical_original_closes_text_gap_without_claiming_legal_review(self):
        result = self.stage(self.packages + self.lock_packages)
        self.assertEqual(result["record"]["historical_text_packages"], 4)
        self.assertEqual(result["record"]["legal_review"], "not_performed")
        for entry in self.manifest["packages"]:
            if not entry.get("supplemental"):
                continue
            record = result["packages"][entry["id"]]
            self.assertTrue(record["complete_text"])
            self.assertTrue(record["historical_provenance"])
            self.assertIsNone(record["full_text_gap"])
            supplemental = next(t for t in record["texts"] if t.get("source_kind") == "historical_notice")
            self.assertTrue(supplemental["provenance_verified"])
            self.assertEqual(supplemental["legal_review"], "not_performed")
            self.assertIn(b"Copyright (c) Steven Sheldon", self.read(supplemental["source"]))
        self.remove_tree(self.destination)
        entry = next(p for p in self.manifest["packages"] if p.get("supplemental"))
        entry["supplemental"][0]["legal_review"] = "approved"
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "reviewed scope"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_historical_scope_cannot_expand_to_other_versions_or_current_commit(self):
        entry = next(p for p in self.manifest["packages"] if p.get("supplemental"))
        asset = entry["supplemental"][0]
        for target, field, value in [(entry, "version", "99.0"),
                                     (entry, "declared_license", "Apache-2.0"),
                                     (asset, "source_kind", "same_commit"),
                                     (asset, "provenance_verified", False),
                                     (asset, "proofs", asset["proofs"][:-1])]:
            with self.subTest(field=field):
                old = target[field]
                target[field] = value
                self.save_manifest()
                with self.assertRaises(RuntimeError):
                    self.stage(self.packages)
                self.assertFalse(self.exists(self.destination))
                target[field] = old
        self.save_manifest()

    def test_resealed_historical_terms_cannot_replace_original_copyright(self):
        entry = next(p for p in self.manifest["packages"] if p.get("supplemental"))
        asset = entry["supplemental"][0]
        raw = self.read(self.vendor / asset["source_path"]).replace(
            b"Copyright (c) Steven Sheldon", b"Copyright (c) <year> <copyright holders>")
        for name in [asset["source_path"], asset["path"]]:
            self.write((self.vendor / name), raw)
            self.seal(name)
        asset.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                     source_sha256=hashlib.sha256(raw).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "Historical original URL or reviewed text hash"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_resealed_current_notice_cannot_drop_apple_sdk_discussion(self):
        entry = next(p for p in self.manifest["packages"] if p.get("supplemental"))
        notice = entry["assets"][0]
        raw = self.read(self.vendor / notice["source_path"]).split(b"## Apple SDKs")[0]
        # The pinned root notice is shared by dispatch2 and objc2. Reseal all
        # references so rejection tests the reviewed historical notice, not a
        # stale digest in an earlier package.
        self.write((self.vendor / notice["source_path"]), raw)
        self.seal(notice["source_path"])
        for package in self.manifest["packages"]:
            for asset in package["assets"]:
                if asset.get("source_path") == notice["source_path"]:
                    self.write((self.vendor / asset["path"]), raw)
                    self.seal(asset["path"])
                    asset.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                                 source_sha256=hashlib.sha256(raw).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "current-version MIT notice differs"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_current_package_mit_declaration_is_required_even_with_historical_text(self):
        entry = next(p for p in self.manifest["packages"] if p.get("supplemental"))
        suffix = entry["path_in_vcs"] + "/Cargo.toml"
        upstream = next(p for p in entry["evidence"] if p["path"].startswith("upstream/") and p["path"].endswith(suffix))
        original = next(p for p in entry["evidence"] if p["path"].endswith("/Cargo.toml.orig"))
        raw = self.read(self.vendor / upstream["path"]).replace(b'license = "MIT"', b'license = "Apache-2.0"')
        for proof in [upstream, original]:
            self.write((self.vendor / proof["path"]), raw)
            proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            self.seal(proof["path"])
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "current package MIT declaration"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_missing_or_resealed_historical_provenance_is_not_accepted(self):
        entry = next(p for p in self.manifest["packages"] if p.get("supplemental"))
        proof = entry["supplemental"][0]["proofs"][0]
        raw = self.read(self.vendor / proof["path"]) + b"changed"
        self.write((self.vendor / proof["path"]), raw)
        proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.seal(proof["path"])
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "Historical provenance"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_source_archive_matches_exact_package_and_is_not_only_a_link(self):
        result = self.stage(self.packages)
        source = result["record"]["source_archives"][0]
        archive = self.destination.parent.parent / source["path"]
        self.assertEqual(hashlib.sha256(self.read(archive)).hexdigest(), source["sha256"])
        with tarfile.open(fileobj=io.BytesIO(self.read(archive))) as contents:
            self.assertEqual(len(contents.getmembers()), 22)
            original = contents.extractfile("selectors-0.38.0/Cargo.toml.orig").read()
            self.assertEqual(original, self.read(self.vendor / "local-evidence/selectors-0.38.0/Cargo.toml.orig"))
        self.remove_tree(self.destination)
        row = self.manifest["source_archives"][0]
        raw = self.read(self.vendor / row["path"]) + b"changed"
        self.write((self.vendor / row["path"]), raw)
        row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.seal(row["path"])
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "reviewed exact package"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_parent_terms_close_both_gaps_with_truthful_manifest_provenance(self):
        result = self.stage(self.packages)
        self.assertEqual(result["record"]["exact_commit_manifest_matches"], 39)
        self.assertEqual(result["record"]["reviewed_publication_version_stamps"], 1)
        expected = {"nom-language": ("raw_exact_match", ["LICENSE"], 7),
                    "tract-extra": ("reviewed_publication_version_stamp",
                                    ["LICENSE", "LICENSE-MIT", "LICENSE-APACHE"], 6)}
        for name, (provenance, terms, members) in expected.items():
            entry = next(p for p in self.manifest["packages"] if p["name"] == name)
            record = result["packages"][entry["id"]]
            self.assertEqual(record["manifest_provenance"], provenance)
            self.assertTrue(record["complete_text"])
            self.assertIsNone(record["full_text_gap"])
            self.assertFalse(record["historical_provenance"])
            self.assertEqual([Path(text["source"]).name for text in record["texts"]], terms)
            archive = next(a for a in self.manifest["source_archives"] if a["package"] == name)
            with tarfile.open(fileobj=io.BytesIO(self.read(self.vendor / archive["path"]))) as source:
                self.assertEqual(len(source.getmembers()), members)
                prefix = f"{name}-{entry['version']}/"
                for filename in ["Cargo.toml.orig", ".cargo_vcs_info.json"]:
                    self.assertEqual(source.extractfile(prefix + filename).read(),
                                     self.read(self.vendor / "local-evidence" / prefix / filename))
            if name == "tract-extra":
                original = self.read(self.vendor / "local-evidence/tract-extra-0.23.8/Cargo.toml.orig")
                upstream = self.read(self.vendor / f"upstream/sonos/tract/{entry['commit']}/extra/Cargo.toml")
                self.assertNotEqual(original, upstream)
                self.assertEqual(upstream.replace(b'\nversion = "0.23.8-pre"\n', b'\nversion = "0.23.8"\n'), original)
                vcs = json.loads(self.read(self.vendor / "local-evidence/tract-extra-0.23.8/.cargo_vcs_info.json"))
                self.assertIs(vcs["git"]["dirty"], True)
        self.assertEqual(result["record"]["legal_review"], "not_performed")

    def test_publication_stamp_is_not_a_general_dirty_version_or_identity_allowlist(self):
        entry = next(p for p in self.manifest["packages"] if p["name"] == "tract-extra")
        variants = [("id", entry["id"] + "-other"), ("version", "0.23.9"),
                    ("commit", "0" * 40), ("repository", "https://github.com/example/tract"),
                    ("path_in_vcs", "other"), ("declared_license", "MIT"),
                    ("package_manifest_matches_published_original", True),
                    ("publication_version_stamp", None),
                    ("publication_version_stamp", {**entry["publication_version_stamp"], "legal_review": "approved"})]
        for field, value in variants:
            with self.subTest(field=field, value=value):
                old = entry[field]
                entry[field] = value
                self.save_manifest()
                with self.assertRaises(RuntimeError):
                    self.stage(self.packages)
                self.assertFalse(self.exists(self.destination))
                entry[field] = old
        self.save_manifest()

    def test_new_archives_require_original_vcs_and_both_raw_manifests(self):
        entry = next(p for p in self.manifest["packages"] if p["name"] == "tract-extra")
        names = [f"upstream/sonos/tract/{entry['commit']}/extra/Cargo.toml",
                 "local-evidence/tract-extra-0.23.8/Cargo.toml.orig",
                 "local-evidence/tract-extra-0.23.8/.cargo_vcs_info.json"]
        for name in names:
            previous = self.read(self.vendor / name)
            variants = [previous + b"\n"]
            if name.endswith(".cargo_vcs_info.json"):
                value = json.loads(previous)
                value["git"]["dirty"] = False
                variants.append(json.dumps(value).encode())
            for raw in variants:
                with self.subTest(name=name, raw=raw):
                    proof = next(p for p in entry["evidence"] if p["path"] == name)
                    old = dict(proof)
                    self.write((self.vendor / name), raw)
                    proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                    self.seal(name)
                    self.save_manifest()
                    with self.assertRaises(RuntimeError):
                        self.stage(self.packages)
                    self.assertFalse(self.exists(self.destination))
                    self.write((self.vendor / name), previous)
                    proof.clear()
                    proof.update(old)
                    self.seal(name)
            self.save_manifest()

    def test_resealed_upstream_or_current_rust_changes_do_not_inherit_stamp(self):
        entry = next(p for p in self.manifest["packages"] if p["name"] == "tract-extra")
        name = f"upstream/sonos/tract/{entry['commit']}/extra/src/lib.rs"
        proof = next(p for p in entry["evidence"] if p["path"] == name)
        previous, old = self.read(self.vendor / name), dict(proof)
        raw = previous + b"\n// changed source\n"
        self.write((self.vendor / name), raw)
        proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.seal(name)
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "Rust source differs"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))
        self.write((self.vendor / name), previous)
        proof.clear()
        proof.update(old)
        self.seal(name)
        self.save_manifest()
        package = next(p for p in self.packages if p["name"] == "tract-extra")
        current = Path(package["manifest_path"]).parent / "src/lib.rs"
        self.write(current, raw)
        with self.assertRaisesRegex(RuntimeError, "Current package Rust source differs"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_resealed_parent_terms_and_workspace_claims_cannot_change(self):
        for package in ["nom-language", "tract-extra"]:
            entry = next(p for p in self.manifest["packages"] if p["name"] == package)
            for field in ["license_inherited_from_workspace", "version_inherited_from_workspace"]:
                entry[field] = True
                self.save_manifest()
                with self.assertRaisesRegex(RuntimeError, "parent license package scope"):
                    self.stage(self.packages)
                self.assertFalse(self.exists(self.destination))
                entry[field] = False
            for asset in entry["assets"]:
                previous, old = self.read(self.vendor / asset["path"]), dict(asset)
                raw = previous + b"\nSubstituted terms\n"
                for name in [asset["path"], asset["source_path"]]:
                    self.write((self.vendor / name), raw)
                    self.seal(name)
                proof = next(p for p in entry["evidence"] if p["path"] == asset["source_path"])
                old_proof = dict(proof)
                proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                asset.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                             source_sha256=hashlib.sha256(raw).hexdigest())
                self.save_manifest()
                with self.assertRaisesRegex(RuntimeError, "parent original terms hash"):
                    self.stage(self.packages)
                self.assertFalse(self.exists(self.destination))
                for name in [asset["path"], asset["source_path"]]:
                    self.write((self.vendor / name), previous)
                    self.seal(name)
                asset.clear()
                asset.update(old)
                proof.clear()
                proof.update(old_proof)
                self.save_manifest()

    def test_all_three_archives_are_required_and_digest_pinned_before_copying(self):
        original = json.loads(json.dumps(self.manifest))
        for package in ["nom-language", "tract-extra"]:
            self.manifest = json.loads(json.dumps(original))
            self.manifest["source_archives"] = [a for a in self.manifest["source_archives"] if a["package"] != package]
            self.save_manifest()
            with self.assertRaisesRegex(RuntimeError, "archive inventory is incomplete"):
                self.stage(self.packages)
            self.assertFalse(self.exists(self.destination))
            self.manifest = json.loads(json.dumps(original))
            row = next(a for a in self.manifest["source_archives"] if a["package"] == package)
            previous = self.read(self.vendor / row["path"])
            raw = previous + b"changed"
            self.write((self.vendor / row["path"]), raw)
            row.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
            self.seal(row["path"])
            self.save_manifest()
            with self.assertRaisesRegex(RuntimeError, "reviewed exact package"):
                self.stage(self.packages)
            self.assertFalse(self.exists(self.destination))
            self.write((self.vendor / row["path"]), previous)
            self.seal(row["path"])
        self.manifest = original
        self.save_manifest()

    def test_stamp_cannot_be_reclassified_as_raw_exact_or_copied_to_another_package(self):
        nom = next(p for p in self.manifest["packages"] if p["name"] == "nom-language")
        tract = next(p for p in self.manifest["packages"] if p["name"] == "tract-extra")
        nom["publication_version_stamp"] = tract["publication_version_stamp"]
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "Raw manifest match has inconsistent provenance"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_resealed_normalized_tract_manifest_cannot_be_claimed_as_raw_exact(self):
        entry = next(p for p in self.manifest["packages"] if p["name"] == "tract-extra")
        name = f"upstream/sonos/tract/{entry['commit']}/extra/Cargo.toml"
        raw = self.read(self.vendor / "local-evidence/tract-extra-0.23.8/Cargo.toml.orig")
        self.write((self.vendor / name), raw)
        proof = next(p for p in entry["evidence"] if p["path"] == name)
        proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.seal(name)
        entry["package_manifest_matches_published_original"] = True
        del entry["publication_version_stamp"]
        self.manifest["summary"].update(exact_commit_manifest_matches=17, reviewed_publication_version_stamps=0)
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "reviewed stamp"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_wheel_and_npm_payloads_include_verified_offline_source_and_notices(self):
        files = upstream_files(self.repository, files=self.files)
        source_name = "licenses/upstream/source-archives/selectors-0.38.0.crate"
        self.assertEqual(files[source_name], self.read(self.vendor / "source-archives/selectors-0.38.0.crate"))
        source = self.root / "base.whl"
        wheel = self.root / "final.whl"
        with zipfile.ZipFile(source, "w") as archive:
            archive.writestr("markitai/__init__.py", "fixture")
            archive.writestr("markitai-1.dist-info/RECORD", "markitai/__init__.py,,\nmarkitai-1.dist-info/RECORD,,\n")
        supplement_wheel_licenses(source, wheel, files)
        with zipfile.ZipFile(wheel) as archive:
            for name, raw in files.items():
                self.assertEqual(archive.read("markitai-1.dist-info/licenses/" + name), raw)
        npm = self.root / "module.tgz"
        with tarfile.open(npm, "w:gz", compresslevel=1) as archive:
            for name, raw in files.items():
                member = tarfile.TarInfo("package/" + name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        verify_node_licenses(npm, files)
        # A distributor cannot silently omit the covered-source copy.
        with tarfile.open(npm, "w:gz", compresslevel=1) as archive:
            for name, raw in files.items():
                if name == source_name:
                    continue
                member = tarfile.TarInfo("package/" + name)
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
        with self.assertRaises(RuntimeError):
            verify_node_licenses(npm, files)

    def test_node_terms_are_extra_to_the_original_seventeen_package_closure(self):
        result = self.stage(self.packages)
        self.assertEqual(len(self.packages), 17)
        self.assertEqual(len(self.node_packages), 4)
        self.assertEqual(result["record"]["matched_packages"], 17)
        self.assertEqual(result["record"]["complete_text_packages"], 17)
        self.assertTrue(all(not key.rsplit("#", 1)[-1].startswith("napi") for key in result["packages"]))
        self.remove_tree(self.destination)
        napi = [p for p in self.refresh_packages if p["name"].startswith("napi")]
        result = self.stage(self.packages + self.node_packages + napi)
        self.assertEqual(result["record"]["matched_packages"], 25)
        self.assertEqual(result["record"]["complete_text_packages"], 25)
        self.assertEqual(result["record"]["exact_commit_manifest_matches"], 39)
        self.assertEqual(result["record"]["reviewed_publication_version_stamps"], 1)
        self.assertEqual(len(result["record"]["source_archives"]), 18)
        for entry in self.manifest["packages"]:
            if not entry["name"].startswith("napi"):
                continue
            self.assertFalse(entry["license_inherited_from_workspace"])
            self.assertFalse(entry["version_inherited_from_workspace"])
            record = result["packages"][entry["id"]]
            self.assertEqual(record["manifest_provenance"], "raw_exact_match")
            self.assertFalse(record["historical_provenance"])
            self.assertIsNone(record["full_text_gap"])
            self.assertEqual(len(record["texts"]), 1)
            raw = self.read(record["texts"][0]["source"])
            self.assertEqual(raw.count(b"MIT License"), 2)
            self.assertIn(b"Copyright (c) 2020-present LongYinan", raw)
            self.assertIn(b"Copyright (c) 2018 GitHub", raw)
            self.assertEqual(hashlib.sha256(raw).hexdigest(), "3f1ce66533302df3a32edbfdfc0b78f0dd34659e4c1f5817162e5ea3c2297215")
        self.assertEqual(result["record"]["legal_review"], "not_performed")

    def test_heifer_terms_cover_the_workspace_inherited_license_of_each_crate(self):
        self.assertEqual(len(self.heif_packages), 4)
        result = self.stage(self.packages + self.heif_packages)
        self.assertEqual(result["record"]["matched_packages"], 21)
        for package in self.heif_packages:
            entry = next(p for p in self.manifest["packages"] if p["id"] == package["id"])
            self.assertTrue(entry["license_inherited_from_workspace"])
            record = result["packages"][package["id"]]
            self.assertEqual(record["manifest_provenance"], "raw_exact_match")
            self.assertTrue(record["complete_text"])
            self.assertEqual(sorted(Path(text["path"]).name for text in record["texts"]),
                             ["LICENSE-APACHE", "LICENSE-MIT"])

    def test_qppocr_terms_are_the_published_commit_root_licenses(self):
        self.assertEqual(len(self.qpp_packages), 3)
        result = self.stage(self.packages + self.qpp_packages)
        self.assertEqual(result["record"]["matched_packages"], 20)
        for package in self.qpp_packages:
            entry = next(p for p in self.manifest["packages"] if p["id"] == package["id"])
            self.assertNotIn("vcs_record", entry)
            vcs = json.loads(self.read_text(self.vendor / "local-evidence" / f"{package['name']}-{package['version']}"
                              / ".cargo_vcs_info.json"))
            self.assertEqual((vcs["git"]["sha1"], vcs["path_in_vcs"]), (entry["commit"], entry["path_in_vcs"]))
            record = result["packages"][package["id"]]
            self.assertEqual(record["manifest_provenance"], "raw_exact_match")
            self.assertTrue(record["complete_text"])
            self.assertEqual(sorted(Path(text["path"]).name for text in record["texts"]),
                             ["LICENSE-APACHE", "LICENSE-MIT"])

    def test_every_package_needs_its_published_vcs_record(self):
        # The evidence names the commit the crate says it was published from.
        name = "local-evidence/qppocr-core-0.3.1/.cargo_vcs_info.json"
        self.remove(self.vendor / name)
        value = json.loads(self.read_text(self.vendor / "inventory.json"))
        value["files"] = [row for row in value["files"] if row["path"] != name]
        self.write((self.vendor / "inventory.json"), json.dumps(value))
        entry = next(p for p in self.manifest["packages"] if p["name"] == "qppocr-core")
        entry["evidence"] = [proof for proof in entry["evidence"] if proof["path"] != name]
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "Missing exact-version package evidence"):
            self.stage(self.packages)
        self.assertFalse(self.exists(self.destination))

    def test_a_qppocr_crate_whose_published_vcs_record_differs_is_rejected(self):
        directory = Path(self.qpp_packages[0]["manifest_path"]).parent
        self.write((directory / ".cargo_vcs_info.json"), "{}")
        with self.assertRaisesRegex(RuntimeError, "Current package source differs"):
            self.stage(self.packages + self.qpp_packages)

    def test_selectors_0_41_takes_the_reviewed_mpl_terms_through_its_own_notice(self):
        package = next(p for p in self.refresh_packages if p["name"] == "selectors")
        result = self.stage(self.packages + [package])
        self.assertEqual(result["record"]["matched_packages"], 18)
        record = result["packages"][package["id"]]
        self.assertEqual(record["manifest_provenance"], "raw_exact_match")
        self.assertTrue(record["complete_text"])
        self.assertEqual(record["commit"], "34d25f741765a0bdb894f8271852e633299352d8")
        self.assertEqual(sorted((Path(text["path"]).name, text["source_kind"]) for text in record["texts"]),
                         [("LICENSE-MPL-2.0.txt", "linked_authority"), ("NOTICE-MPL-2.0.txt", "same_commit")])
        archives = {(row["package"], row["version"]) for row in result["record"]["source_archives"]}
        self.assertIn(("selectors", "0.41.0"), archives)
        # The 0.38.0 terms do not stand in for 0.41.0, nor the reverse.
        old = next(p for p in self.packages if p["name"] == "selectors")
        self.assertNotEqual(result["packages"][old["id"]]["commit"], record["commit"])

    def test_selectors_0_41_notice_must_be_its_own_commit_bytes(self):
        entry = next(p for p in self.manifest["packages"] if p["id"].endswith("#selectors@0.41.0"))
        notice = entry["assets"][0]
        notice["source_path"] = notice["source_path"].replace(entry["commit"], "572ecba2d1600e7c3d490586692a209faf703baa")
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "outside the attributed commit"):
            self.stage(self.refresh_packages)
        self.assertFalse(self.exists(self.destination))

    def test_refreshed_napi_tuples_keep_both_original_copyrights_from_their_commit(self):
        napi = [p for p in self.refresh_packages if p["name"].startswith("napi")]
        self.assertEqual(sorted(p["name"] for p in napi),
                         ["napi", "napi-derive", "napi-derive-backend", "napi-sys"])
        result = self.stage(self.node_packages + napi)
        self.assertEqual(result["record"]["matched_packages"], 8)
        for package in napi:
            record = result["packages"][package["id"]]
            self.assertEqual(record["commit"], "d9cb526c15e63ba30e06b4d169068c9854180595")
            self.assertEqual(record["manifest_provenance"], "raw_exact_match")
            self.assertEqual(len(record["texts"]), 1)
            raw = self.read(self.destination.parent.parent / record["texts"][0]["path"])
            self.assertEqual(raw.count(b"MIT License"), 2)
        archives = {(row["package"], row["version"]) for row in result["record"]["source_archives"]}
        self.assertTrue({(p["name"], p["version"]) for p in napi} <= archives)

    def test_a_refreshed_napi_tuple_cannot_borrow_the_older_commit(self):
        entry = next(p for p in self.manifest["packages"] if p["id"].endswith("#napi-sys@3.4.1"))
        entry["commit"] = "37109f133043aff379a38edfb8b1548e7e9309e5"
        self.save_manifest()
        with self.assertRaises(RuntimeError):
            self.stage(self.refresh_packages)
        self.assertFalse(self.exists(self.destination))

    def test_lock_refresh_tuples_take_their_own_commit_root_terms(self):
        self.assertEqual(sorted(p["name"] for p in self.lock_packages),
                         ["alloc-stdlib", "jni", "jni-macros", "jni-sys-macros", "objc2",
                          "rustls-platform-verifier-android"])
        result = self.stage(self.packages + self.lock_packages)
        self.assertEqual(result["record"]["matched_packages"], 23)
        self.assertEqual(result["record"]["complete_text_packages"], 23)
        expected = {"alloc-stdlib": ["LICENSE"], "objc2": ["LICENSE-MIT-HISTORICAL.txt", "LICENSE.md"]}
        for package in self.lock_packages:
            entry = next(p for p in self.manifest["packages"] if p["id"] == package["id"])
            self.assertEqual(entry["license_inherited_from_workspace"], package["name"] == "jni")
            vcs = json.loads(self.read_text(self.vendor / "local-evidence" / f"{package['name']}-{package['version']}"
                              / ".cargo_vcs_info.json"))
            self.assertEqual((vcs["git"]["sha1"], vcs["path_in_vcs"]), (entry["commit"], entry["path_in_vcs"]))
            record = result["packages"][package["id"]]
            self.assertEqual(record["manifest_provenance"], "raw_exact_match")
            self.assertTrue(record["complete_text"])
            self.assertEqual(record["historical_provenance"], package["name"] == "objc2")
            self.assertEqual(sorted(Path(text["path"]).name for text in record["texts"]),
                             expected.get(package["name"], ["LICENSE-APACHE", "LICENSE-MIT"]))
        archives = {(row["package"], row["version"]) for row in result["record"]["source_archives"]}
        self.assertTrue({(p["name"], p["version"]) for p in self.lock_packages} <= archives)

    def test_only_jni_may_claim_workspace_license_inheritance(self):
        for name, claim in [("jni-macros", True), ("jni", False)]:
            entry = next(p for p in self.manifest["packages"] if p["name"] == name)
            entry["license_inherited_from_workspace"] = claim
            self.save_manifest()
            with self.assertRaisesRegex(RuntimeError, "parent license package scope"):
                self.stage(self.lock_packages)
            self.assertFalse(self.exists(self.destination))
            entry["license_inherited_from_workspace"] = not claim
        self.save_manifest()

    def test_objc2_0_6_5_historical_text_needs_its_own_commit_history(self):
        old = next(p for p in self.manifest["packages"] if p["id"].endswith("#objc2@0.6.4"))
        entry = next(p for p in self.manifest["packages"] if p["id"].endswith("#objc2@0.6.5"))
        entry["supplemental"][0]["proofs"][0] = dict(old["supplemental"][0]["proofs"][0])
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "target-anchored provenance"):
            self.stage(self.lock_packages)
        self.assertFalse(self.exists(self.destination))

    def test_atomig_macro_terms_are_the_published_commit_root_licenses(self):
        self.assertEqual(len(self.avif_packages), 1)
        result = self.stage(self.packages + self.avif_packages)
        self.assertEqual(result["record"]["matched_packages"], 18)
        package = self.avif_packages[0]
        entry = next(p for p in self.manifest["packages"] if p["id"] == package["id"])
        self.assertFalse(entry["license_inherited_from_workspace"])
        self.assertEqual(entry["path_in_vcs"], "atomig-macro")
        record = result["packages"][package["id"]]
        self.assertEqual(record["manifest_provenance"], "raw_exact_match")
        self.assertTrue(record["complete_text"])
        self.assertEqual(sorted(Path(text["path"]).name for text in record["texts"]),
                         ["LICENSE-APACHE", "LICENSE-MIT"])

    def test_node_parent_child_and_vcs_cannot_be_resealed_into_other_identity(self):
        entry = next(p for p in self.manifest["packages"] if p["name"] == "napi-build")
        root = f"upstream/napi-rs/napi-rs/{entry['commit']}/"
        original = "local-evidence/napi-build-2.6.0/"
        names = [root + "Cargo.toml", root + "crates/build/Cargo.toml",
                 original + ".cargo_vcs_info.json"]
        for name in names:
            with self.subTest(name=name):
                previous = self.read(self.vendor / name)
                proof = next(p for p in entry["evidence"] if p["path"] == name)
                old_proof = dict(proof)
                raw = previous + b"\n"
                self.write((self.vendor / name), raw)
                proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                self.seal(name)
                self.save_manifest()
                with self.assertRaises(RuntimeError):
                    self.stage(self.node_packages)
                self.assertFalse(self.exists(self.destination))
                self.write((self.vendor / name), previous)
                proof.clear()
                proof.update(old_proof)
                self.seal(name)
                self.save_manifest()

    def test_node_original_terms_cannot_drop_a_copyright_even_after_resealing(self):
        entry = next(p for p in self.manifest["packages"] if p["name"] == "napi-build")
        asset = entry["assets"][0]
        previous = self.read(self.vendor / asset["path"])
        raw = previous.split(b"\nMIT License\n", 1)[0] + b"\n"
        self.assertNotEqual(raw, previous)
        for name in [asset["path"], asset["source_path"]]:
            self.write((self.vendor / name), raw)
            self.seal(name)
        proof = next(p for p in entry["evidence"] if p["path"] == asset["source_path"])
        proof.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        asset.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(),
                     source_sha256=hashlib.sha256(raw).hexdigest())
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "parent original terms hash"):
            self.stage(self.node_packages)
        self.assertFalse(self.exists(self.destination))

    def test_node_scope_and_original_archives_remain_fixed(self):
        original = json.loads(json.dumps(self.manifest))
        for field, value in [("license_inherited_from_workspace", True),
                             ("version_inherited_from_workspace", True),
                             ("declared_license", "Apache-2.0"),
                             ("repository", "https://github.com/example/napi-rs"),
                             ("path_in_vcs", "other"), ("version", "3.14.1"),
                             ("commit", "0" * 40)]:
            with self.subTest(field=field):
                self.manifest = json.loads(json.dumps(original))
                entry = next(p for p in self.manifest["packages"] if p["name"] == "napi")
                entry[field] = value
                self.save_manifest()
                with self.assertRaises(RuntimeError):
                    self.stage(self.node_packages)
                self.assertFalse(self.exists(self.destination))
        self.manifest = json.loads(json.dumps(original))
        self.manifest["source_archives"] = [a for a in self.manifest["source_archives"] if a["package"] != "napi"]
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "archive inventory is incomplete"):
            self.stage(self.node_packages)
        self.assertFalse(self.exists(self.destination))
        self.manifest = json.loads(json.dumps(original))
        archive = next(a for a in self.manifest["source_archives"] if a["package"] == "napi")
        raw = self.read(self.vendor / archive["path"]) + b"changed"
        self.write((self.vendor / archive["path"]), raw)
        archive.update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        self.seal(archive["path"])
        self.save_manifest()
        with self.assertRaisesRegex(RuntimeError, "reviewed exact package"):
            self.stage(self.node_packages)
        self.assertFalse(self.exists(self.destination))

    def test_node_runtime_metadata_must_match_published_vcs_and_original(self):
        package = self.node_packages[0]
        source = Path(package["manifest_path"]).parent
        for name in [".cargo_vcs_info.json", "Cargo.toml.orig"]:
            previous = self.read(source / name)
            self.write((source / name), previous + b"\n")
            with self.assertRaisesRegex(RuntimeError, "Current package source differs"):
                self.stage([package])
            self.assertFalse(self.exists(self.destination))
            self.write((source / name), previous)

    def test_node_payload_verifier_checks_both_original_copyrights_byte_for_byte(self):
        files = upstream_files(self.repository, files=self.files)
        names = ["licenses/upstream/overlay/licenses/" + name + "/LICENSE" for name in
                 ["napi-3.14.0", "napi-build-2.6.0", "napi-derive-3.6.10", "napi-sys-3.4.0",
                  "napi-3.15.0", "napi-derive-3.6.13", "napi-sys-3.4.1", "napi-derive-backend-6.1.6"]]
        for name in names:
            self.assertEqual(files[name], self.read(self.repository / name))
            self.assertEqual(files[name].count(b"MIT License"), 2)
        npm = self.root / "node.tgz"
        def archive_with(changed=None):
            with tarfile.open(npm, "w:gz", compresslevel=1) as archive:
                for name, raw in files.items():
                    if name == changed:
                        raw = raw.split(b"\nMIT License\n", 1)[0] + b"\n"
                    member = tarfile.TarInfo("package/" + name)
                    member.size = len(raw)
                    archive.addfile(member, io.BytesIO(raw))
        archive_with()
        verify_node_licenses(npm, files)
        for name in names:
            archive_with(name)
            with self.assertRaises(RuntimeError):
                verify_node_licenses(npm, files)


if __name__ == "__main__":
    unittest.main()
