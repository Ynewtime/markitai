"""The committed web workbench build (crates/markitai-cli/src/server/web/dist).

`cargo build` embeds dist/ without Node, so these checks keep it honest: every
file matches manifest.json, the manifest's source digest matches the sources
(stale output fails without Node), compressed twins decode to their originals,
the vendored files match vendor/web/provenance.json, and with Node and the
pinned toolchain installed the bundle is rebuilt and compared byte for byte.
"""

import gzip
import hashlib
import json
import pathlib
import re
import shutil
import subprocess
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
WEB = ROOT / "crates/markitai-cli/src/server/web"
DIST = WEB / "dist"
VENDOR = ROOT / "vendor/web"
WEB_RS = ROOT / "crates/markitai-cli/src/server/web.rs"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def source_files():
    """The files build.mjs hashes, in its order."""
    files = []

    def walk(directory):
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.is_dir():
                walk(path)
            elif not path.name.endswith(".test.ts"):
                files.append(path)

    walk(WEB / "src")
    walk(WEB / "public")
    files.extend(WEB / name for name in ["build.mjs", "package-lock.json", "tsconfig.json"])
    return files


def source_digest():
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(path.relative_to(WEB).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class WebDistTest(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads((DIST / "manifest.json").read_text())

    def test_every_built_file_matches_the_manifest(self):
        listed = set(self.manifest["files"])
        present = {path.name for path in DIST.iterdir() if path.name != "manifest.json"}
        self.assertEqual(present, listed)
        for name, record in self.manifest["files"].items():
            data = (DIST / name).read_bytes()
            self.assertEqual(len(data), record["bytes"], name)
            self.assertEqual(sha256(data), record["sha256"], name)

    def test_the_build_is_not_older_than_its_sources(self):
        self.assertEqual(
            self.manifest["source_digest"],
            source_digest(),
            "dist/ is stale: run `npm ci && node build.mjs` in crates/markitai-cli/src/server/web",
        )

    def test_compressed_twins_decode_to_their_originals(self):
        for name in ["app.js", "app.css"]:
            self.assertEqual(gzip.decompress((DIST / f"{name}.gz").read_bytes()), (DIST / name).read_bytes(), name)
        for name in ["marked.js", "purify.js"]:
            original = (VENDOR / name).read_bytes()
            self.assertEqual(gzip.decompress((DIST / f"{name}.gz").read_bytes()), original, name)
            self.assertEqual(self.manifest["vendor"][name], sha256(original), name)

    def test_the_server_embeds_only_listed_files(self):
        source = WEB_RS.read_text()
        embedded = set(re.findall(r'dist!\("([^"]+)"\)', source))
        self.assertTrue(embedded, "web.rs embeds nothing from dist/")
        self.assertTrue(embedded <= set(self.manifest["files"]), embedded - set(self.manifest["files"]))
        for name in re.findall(r'vendor!\("([^"]+)"\)', source):
            self.assertTrue((VENDOR / name).is_file(), name)
        # The page loads exactly these, all same-origin.
        html = (DIST / "index.html").read_text()
        for reference in re.findall(r'(?:src|href)="([^"]+)"', html):
            self.assertTrue(reference.startswith("/ui/"), reference)
        self.assertNotIn("style=", html)

    def test_vendored_files_match_their_provenance(self):
        entries = json.loads((VENDOR / "provenance.json").read_text())
        by_file = {entry["file"]: entry for entry in entries}
        for path in VENDOR.iterdir():
            if path.name == "provenance.json":
                continue
            self.assertIn(path.name, by_file, f"{path.name} has no provenance entry")
            data = path.read_bytes()
            self.assertEqual(by_file[path.name]["bytes"], len(data), path.name)
            self.assertEqual(by_file[path.name]["sha256"], sha256(data), path.name)
        for entry in entries:
            target = (VENDOR / entry["file"]).resolve()
            self.assertTrue(target.is_file(), entry["file"])
            self.assertEqual(entry["sha256"], sha256(target.read_bytes()), entry["file"])
            self.assertTrue(entry["source"].startswith("https://registry.npmjs.org/"), entry["file"])

    @unittest.skipUnless(shutil.which("node") and (WEB / "node_modules/esbuild").is_dir(), "Node or the pinned toolchain is not installed")
    def test_a_rebuild_reproduces_dist(self):
        result = subprocess.run(["node", "build.mjs", "--check"], cwd=WEB, capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is not installed")
    def test_the_workbench_unit_tests_and_css_scale_pass(self):
        tests = sorted(str(path.relative_to(WEB)) for path in (WEB / "src").rglob("*.test.ts"))
        self.assertTrue(tests)
        for command in [["node", "--test", *tests], ["node", "scripts/check-css-scale.mjs"]]:
            result = subprocess.run(command, cwd=WEB, capture_output=True, text=True, timeout=300)
            self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])


if __name__ == "__main__":
    unittest.main()
