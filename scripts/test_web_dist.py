"""The committed web workbench build (crates/markitai-cli/src/server/web/dist).

`cargo build` embeds dist/ without Node, so these checks keep it honest: every
file matches manifest.json, the manifest's source digest matches the sources
(stale output fails without Node), compressed twins decode to their originals,
the vendored files match vendor/web/provenance.json, and with Node and the
pinned toolchain installed the bundle is rebuilt and compared byte for byte,
and the workbench tests, the CSS scale check and the colour-contrast check run.
"""

import gzip
import hashlib
import json
import pathlib
import re
import shutil
import contextlib
import io
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import bounded_run
import check
from web_deps import stale_web_dependencies

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
        bounded_run.watch(self)
        self.manifest = json.loads((DIST / "manifest.json").read_text(encoding="utf-8"))

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
        source = WEB_RS.read_text(encoding="utf-8")
        embedded = set(re.findall(r'dist!\("([^"]+)"\)', source))
        self.assertTrue(embedded, "web.rs embeds nothing from dist/")
        self.assertTrue(embedded <= set(self.manifest["files"]), embedded - set(self.manifest["files"]))
        for name in re.findall(r'vendor!\("([^"]+)"\)', source):
            self.assertTrue((VENDOR / name).is_file(), name)
        # The page loads exactly these, all same-origin.
        html = (DIST / "index.html").read_text(encoding="utf-8")
        for reference in re.findall(r'(?:src|href)="([^"]+)"', html):
            self.assertTrue(reference.startswith("/ui/"), reference)
        self.assertNotIn("style=", html)

    def test_vendored_files_match_their_provenance(self):
        entries = json.loads((VENDOR / "provenance.json").read_text(encoding="utf-8"))
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
            # A package that ships no licence text names its repository's file at a pinned commit.
            pinned = re.fullmatch(r"https://raw\.githubusercontent\.com/[\w.-]+/[\w.-]+/[0-9a-f]{40}/[\w./-]+", entry["source"])
            self.assertTrue(entry["source"].startswith("https://registry.npmjs.org/") or pinned, entry["file"])

    def test_pele_licence_evidence_follows_the_locked_version(self):
        # package.json takes any pele 1.x from 1.9.0, so `npm update pele` can move the lock: the licence copy,
        # its provenance entry and the NOTICE line must then be refreshed to the version the build bundles.
        locked = json.loads((WEB / "package-lock.json").read_text(encoding="utf-8"))["packages"]["node_modules/pele"]
        entry = next(entry for entry in json.loads((VENDOR / "provenance.json").read_text(encoding="utf-8")) if entry["file"] == "pele-LICENSE")
        stale = "refresh vendor/web/pele-LICENSE, its provenance entry and NOTICE for the locked pele (see docs/web-ui.md)"
        self.assertEqual(entry["version"], locked["version"], stale)
        self.assertEqual(entry["package_integrity"], locked["integrity"], stale)
        self.assertEqual(entry["source"], locked["resolved"], stale)
        self.assertIn(f"Pele {locked['version']} (MIT", (ROOT / "NOTICE").read_text(encoding="utf-8"), stale)
        installed = WEB / "node_modules/pele"
        if (installed / "LICENSE").is_file() and json.loads((installed / "package.json").read_text(encoding="utf-8"))["version"] == locked["version"]:
            self.assertEqual((VENDOR / "pele-LICENSE").read_bytes(), (installed / "LICENSE").read_bytes(), stale)

    @unittest.skipUnless(shutil.which("node") and (WEB / "node_modules/esbuild").is_dir(), "Node or the pinned toolchain is not installed")
    def test_a_rebuild_reproduces_dist(self):
        result = bounded_run.run(["node", "build.mjs", "--check"], cwd=WEB, encoding="utf-8", errors="replace", timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which("node"), "Node is not installed")
    def test_the_workbench_unit_tests_css_scale_and_contrast_pass(self):
        tests = sorted(str(path.relative_to(WEB)) for path in (WEB / "src").rglob("*.test.ts"))
        self.assertTrue(tests)
        # The tests import the locked packages; without them two files fail only by "test failed".
        stale = stale_web_dependencies(WEB)
        if stale:
            self.fail(f"run `npm ci` in {WEB.relative_to(ROOT).as_posix()} (scripts/check.py does when needed): "
                      + "; ".join(stale[:3]) + (f"; {len(stale) - 3} more" if len(stale) > 3 else ""))
        for command in [
            ["node", "--test", *tests],
            ["node", "scripts/check-css-scale.mjs"],
            ["node", "scripts/check-contrast.mjs"],
        ]:
            result = bounded_run.run(command, cwd=WEB, encoding="utf-8", errors="replace", timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout[-4000:] + result.stderr[-4000:])


class WebDependencyTest(unittest.TestCase):
    """node_modules against package-lock.json, and check.py's `npm ci` when they differ."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.web = pathlib.Path(temporary.name)
        (self.web / "package-lock.json").write_text(json.dumps({"lockfileVersion": 3, "packages": {
            "": {"version": "1.0.0"},
            "node_modules/preact": {"version": "11.0.1", "dev": True},
            "node_modules/@esbuild/aix-ppc64": {"version": "0.28.2", "dev": True, "optional": True, "os": ["aix"]},
            "node_modules/a/node_modules/b": {"version": "2.0.0", "dev": True},
            "node_modules/local": {"resolved": "../local", "link": True},
        }}), encoding="utf-8")

    def install(self, key, version):
        (self.web / key).mkdir(parents=True, exist_ok=True)
        (self.web / key / "package.json").write_text(json.dumps({"version": version}), encoding="utf-8")

    def test_missing_or_other_versions_are_named_and_other_platforms_may_be_absent(self):
        self.assertEqual(stale_web_dependencies(self.web), ["preact is not installed", "b is not installed"])
        self.install("node_modules/preact", "10.29.8")
        self.install("node_modules/a/node_modules/b", "2.0.0")
        self.assertEqual(stale_web_dependencies(self.web), ["preact 10.29.8 is installed, the lockfile has 11.0.1"])
        self.install("node_modules/preact", "11.0.1")
        self.install("node_modules/@esbuild/aix-ppc64", "0.27.0")
        self.assertEqual(stale_web_dependencies(self.web), ["@esbuild/aix-ppc64 0.27.0 is installed, the lockfile has 0.28.2"])
        self.install("node_modules/@esbuild/aix-ppc64", "0.28.2")
        self.assertEqual(stale_web_dependencies(self.web), [])

    def run_check(self, tools, stale):
        runs, out, err = [], io.StringIO(), io.StringIO()
        with (patch("check.WEB", self.web), patch("check.stale_web_dependencies", return_value=stale),
              patch("check.shutil.which", side_effect=lambda name, path=None: tools.get(name)),
              patch("check.subprocess.run", side_effect=lambda command, **kwargs: runs.append((command, kwargs)) or
                    subprocess.CompletedProcess(command, 0)),
              contextlib.redirect_stdout(out), contextlib.redirect_stderr(err)):
            code = check.install_web_dependencies(self.web.parent, {"PATH": "synthetic"})
        return code, runs, out.getvalue() + err.getvalue()

    def test_check_runs_npm_ci_only_when_the_packages_differ(self):
        tools = {"node": "/opt/node/bin/node", "npm": "/opt/node/bin/npm"}
        self.assertEqual(self.run_check(tools, []), (0, [], ""))
        code, runs, said = self.run_check(tools, ["pele is not installed", "preact is not installed"])
        self.assertEqual(code, 0)
        self.assertEqual([(command, kwargs["cwd"]) for command, kwargs in runs], [(["/opt/node/bin/npm", "ci"], self.web)])
        self.assertIn("differs from package-lock.json (pele is not installed and 1 more); running npm ci", said)
        # Without npm it says what to do and stops the gate; without Node the web tests skip.
        code, runs, said = self.run_check({"node": tools["node"]}, ["pele is not installed"])
        self.assertEqual((code, runs), (1, []))
        self.assertIn("npm is not on PATH", said)
        self.assertEqual(self.run_check({}, ["pele is not installed"]), (0, [], ""))


if __name__ == "__main__":
    unittest.main()
