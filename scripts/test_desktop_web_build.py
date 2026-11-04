"""desktop_web_build.py with synthetic archives and Go modules; no Go, mygo or network."""
from pathlib import Path
import contextlib
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import tempfile
import unittest
import zipfile

import bounded_run
import desktop_web_build
from desktop_web_build import (APP_DIR, PLATFORMS, SHIPPED, check_artifacts, e2e_project, layout, mygo_build_command,
                               stage)
from release_build import TARGETS, archive_name

ROOT = Path(__file__).resolve().parents[1]
VERSION = "9.8.7"
TARGET = "aarch64-apple-darwin"


def tar_archive(path, version=VERSION):
    members = [("markitai", 0o755, f"#!/bin/sh\necho markitai {version}\n".encode()),
               ("LICENSE", 0o644, b"MIT"), ("NOTICE", 0o644, b"notice"),
               ("licenses/upstream/inventory.json", 0o644, b"{}"),
               ("vendor/web/preact-LICENSE", 0o644, b"preact"), ("README.md", 0o644, b"# guide")]
    with tarfile.open(path, "w:gz") as bundle:
        for name, mode, data in members:
            info = tarfile.TarInfo(name)
            info.mode, info.size = mode, len(data)
            bundle.addfile(info, io.BytesIO(data))
        for name in ["mkai", "markitai-mcp"]:
            info = tarfile.TarInfo(name)
            info.type, info.linkname, info.mode = tarfile.SYMTYPE, "markitai", 0o777
            bundle.addfile(info)
    return path


def zip_archive(path):
    with zipfile.ZipFile(path, "w") as bundle:
        for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe"]:
            bundle.writestr(name, b"MZ")
        bundle.writestr("LICENSE", b"MIT")
        bundle.writestr("NOTICE", b"notice")
    return path


class Checkout:
    """A synthetic repository root: Cargo.toml, desktop/web/mygo.json, a Go root and one module."""

    def __init__(self, directory, cargo=VERSION, app=VERSION):
        self.root = Path(directory) / "repo"
        (self.root / APP_DIR).mkdir(parents=True)
        (self.root / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{cargo}"\n', encoding="utf-8")
        mygo = json.loads((ROOT / APP_DIR / "mygo.json").read_text(encoding="utf-8"))
        mygo["version"] = app
        (self.root / APP_DIR / "mygo.json").write_text(json.dumps(mygo, indent=2) + "\n", encoding="utf-8")
        self.goroot = Path(directory) / "goroot"
        self.goroot.mkdir()
        (self.goroot / "LICENSE").write_bytes(b"Go BSD")
        (self.goroot / "PATENTS").write_bytes(b"Go patents")
        module = Path(directory) / "mod/mygo@v0.7.4"
        module.mkdir(parents=True)
        (module / "LICENSE").write_bytes(b"mygo MIT")
        (module / "go.mod").write_bytes(b"module github.com/egoist/mygo\n")
        self.modules = [("github.com/egoist/mygo", "v0.7.4", str(module))]
        self.archives = Path(directory) / "archives"
        self.archives.mkdir()

    def stage(self, archive, sums=None, modules=None, host_system="none/none"):
        with contextlib.redirect_stdout(io.StringIO()):
            return stage(archive, sums, root=self.root, modules=self.modules if modules is None else modules,
                         goroot=self.goroot, host_system=host_system)

    @property
    def resources(self):
        return self.root / APP_DIR / "resources"


class StageTests(unittest.TestCase):
    def setUp(self):
        bounded_run.watch(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.repo = Checkout(directory.name)
        self.directory = Path(directory.name)

    def archive(self, version=VERSION, target=TARGET):
        return tar_archive(self.repo.archives / archive_name(version, target))

    def test_stage_places_the_program_and_every_licence(self):
        stamp = self.repo.stage(self.archive())
        program = self.repo.resources / "darwin-arm64/bin/markitai"
        if os.name != "nt":  # Windows files have no executable bit
            self.assertEqual(stat.S_IMODE(program.stat().st_mode), 0o755)
        files = sorted(path.relative_to(self.repo.resources).as_posix()
                       for path in self.repo.resources.rglob("*") if path.is_file())
        self.assertEqual(files, [
            ".stage.json",
            "darwin-arm64/bin/markitai",
            "licenses/go/README.md",
            "licenses/go/github.com/egoist/mygo/LICENSE",
            "licenses/go/go/LICENSE",
            "licenses/go/go/PATENTS",
            "licenses/markitai/LICENSE",
            "licenses/markitai/NOTICE",
            "licenses/markitai/licenses/upstream/inventory.json",
            "licenses/markitai/vendor/web/preact-LICENSE",
        ])
        readme = (self.repo.resources / "licenses/go/README.md").read_text(encoding="utf-8")
        self.assertIn("github.com/egoist/mygo v0.7.4", readme)
        self.assertEqual((stamp["version"], stamp["platform"]), (VERSION, "darwin/arm64"))

    def test_restage_starts_from_nothing(self):
        self.repo.stage(self.archive())
        (self.repo.resources / "darwin-arm64/bin/stale").write_bytes(b"x")
        self.repo.stage(zip_archive(self.repo.archives / archive_name(VERSION, "x86_64-pc-windows-msvc")))
        self.assertFalse((self.repo.resources / "darwin-arm64").exists())
        program = self.repo.resources / "windows-amd64/bin/markitai.exe"
        self.assertEqual(program.read_bytes(), b"MZ")
        self.assertEqual(json.loads((self.repo.resources / ".stage.json").read_text())["platform"], "windows/amd64")

    def test_checksums_are_checked(self):
        archive = self.archive()
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        sums = self.directory / "SHA256SUMS"
        sums.write_text(f"{digest}  {archive.name}\n{'0' * 64}  other.tar.gz\n", encoding="utf-8")
        self.repo.stage(archive, sums)
        sums.write_text(f"{'0' * 64}  {archive.name}\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "differs"):
            self.repo.stage(archive, sums)
        sums.write_text(f"{digest}  other.tar.gz\n", encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "does not list"):
            self.repo.stage(archive, sums)

    def test_an_archive_of_another_version_is_refused(self):
        with self.assertRaisesRegex(SystemExit, "not a markitai 9.8.7 release archive"):
            self.repo.stage(self.archive(version="9.8.6"))
        renamed = self.repo.archives / "markitai.tar.gz"
        shutil.copyfile(self.archive(), renamed)
        with self.assertRaises(SystemExit):
            self.repo.stage(renamed)
        self.assertFalse(self.repo.resources.exists())

    def test_mygo_json_must_name_the_workspace_version(self):
        repo = Checkout(self.directory / "drift", cargo=VERSION, app="9.8.6")
        with self.assertRaisesRegex(SystemExit, "desktop/web/mygo.json says 9.8.6, Cargo.toml 9.8.7"):
            repo.stage(tar_archive(repo.archives / archive_name(VERSION, TARGET)))

    def test_a_program_that_names_another_version_is_refused_where_it_runs(self):
        host = desktop_web_build.host_platform()
        target = PLATFORMS[host] if host in ("darwin/arm64", "linux/amd64") else TARGET
        archive = tar_archive(self.repo.archives / archive_name(VERSION, target), version="9.8.6")
        if target != TARGET or host == "darwin/arm64":
            with self.assertRaisesRegex(SystemExit, "printed 'markitai 9.8.6'"):
                self.repo.stage(archive, host_system=host)
        self.repo.stage(archive)

    def test_a_module_without_a_licence_fails(self):
        bare = self.directory / "mod/bare@v1.0.0"
        bare.mkdir(parents=True)
        (bare / "README.md").write_bytes(b"no licence")
        with self.assertRaisesRegex(SystemExit, "example.com/bare v1.0.0: no licence file"):
            self.repo.stage(self.archive(), modules=[*self.repo.modules, ("example.com/bare", "v1.0.0", str(bare))])

    def test_a_module_without_a_directory_fails(self):
        with self.assertRaisesRegex(SystemExit, "example.com/vendored v1.0.0: go list gave no module directory"):
            self.repo.stage(self.archive(), modules=[*self.repo.modules, ("example.com/vendored", "v1.0.0", "")])

    def test_a_failed_stage_keeps_the_resources_staged_before(self):
        before = self.repo.stage(self.archive())
        with self.assertRaises(SystemExit):
            self.repo.stage(zip_archive(self.repo.archives / archive_name(VERSION, "x86_64-pc-windows-msvc")),
                            modules=[*self.repo.modules, ("example.com/vendored", "v1.0.0", "")])
        self.assertEqual(json.loads((self.repo.resources / ".stage.json").read_text()), before)
        self.assertTrue((self.repo.resources / "darwin-arm64/bin/markitai").is_file())


class RegistryTests(unittest.TestCase):
    def test_platforms_are_the_release_targets(self):
        self.assertEqual(sorted(PLATFORMS.values()), sorted(TARGETS))
        self.assertLessEqual(SHIPPED, PLATFORMS.keys())
        for platform, target in PLATFORMS.items():
            goos, goarch = platform.split("/")
            self.assertIn({"darwin": "apple-darwin", "linux": "linux-gnu", "windows": "windows-msvc"}[goos], target)
            self.assertTrue(target.startswith({"arm64": "aarch64-", "amd64": "x86_64-"}[goarch]))

    def test_no_artifact_takes_the_cli_archive_prefix(self):
        check_artifacts(["Markitai Web.app", "markitai-web-2.3.2-arm64.dmg", "SHA256SUMS"])
        with self.assertRaisesRegex(SystemExit, "markitai-v2.3.2-arm64.dmg"):
            check_artifacts(["Markitai Web.app", "markitai-v2.3.2-arm64.dmg"])

    def test_the_disk_image_is_named_by_the_slug(self):
        mygo = json.loads((ROOT / APP_DIR / "mygo.json").read_text(encoding="utf-8"))
        self.assertEqual((mygo["name"], mygo["slug"], mygo["identifier"]),
                         ("Markitai Web", "markitai-web", "dev.markitai.web"))
        self.assertFalse(mygo["slug"].startswith(desktop_web_build.CLI_ASSET_PREFIX))

    def test_linux_packages_are_named_by_the_slug(self):
        app, program, packages = layout("linux/amd64", "2.3.2")
        self.assertEqual((app, program), ("markitai-web", "bin/markitai"))
        self.assertEqual(packages, {"markitai-web_2.3.2_amd64.deb", "markitai-web-2.3.2-linux-amd64.tar.gz"})
        check_artifacts([app, *packages])
        self.assertIn("markitai-web_2.4.0~rc.1_amd64.deb", layout("linux/amd64", "2.4.0-rc.1")[2])

    def test_the_windows_installer_is_named_by_the_slug(self):
        app, program, packages = layout("windows/amd64", "2.3.2")
        self.assertEqual((app, program), ("Markitai Web.exe", "bin/markitai.exe"))
        self.assertEqual(packages, {"markitai-web-setup-2.3.2-amd64.exe"})
        check_artifacts([app, *packages])

    def test_the_e2e_build_keeps_the_inspector_and_its_own_directory(self):
        arguments, environment = mygo_build_command("darwin/arm64", e2e=True)
        self.assertEqual(arguments[-6:], ["-debug", "-skip-dmg", "-skip-installer", "-o", "../e2e",
                                          desktop_web_build.E2E_PROJECT])
        app = ROOT / APP_DIR
        self.assertEqual(os.path.normpath(app / arguments[-1] / arguments[-2]), str(app / "build/e2e"))
        self.assertEqual(environment, {"GOFLAGS": "-tags=e2e"})
        _, environment = mygo_build_command("darwin/arm64", e2e=True, goflags="-mod=mod")
        self.assertEqual(environment, {"GOFLAGS": "-mod=mod -tags=e2e"})
        arguments, environment = mygo_build_command("darwin/arm64", e2e=False)
        self.assertNotIn("-debug", arguments)
        self.assertEqual(environment, {})


class E2EProjectTests(unittest.TestCase):
    def test_the_e2e_build_has_its_own_identifier_and_the_app_s_package_and_resources(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        repo = Checkout(directory.name)
        app = repo.root / APP_DIR
        production = (app / "mygo.json").read_bytes()
        (app / "build/e2e-project").mkdir(parents=True)
        (app / "build/e2e-project/stale").write_bytes(b"x")
        (app / "resources").mkdir()  # as stage leaves it
        project = e2e_project(repo.root)
        self.assertEqual(project, app / desktop_web_build.E2E_PROJECT)
        self.assertFalse((project / "stale").exists())
        config = json.loads((project / "mygo.json").read_text(encoding="utf-8"))
        expected = json.loads(production)
        self.assertEqual(config.pop("identifier"), expected.pop("identifier") + ".e2e")
        self.assertEqual((project / config.pop("main")).resolve(), app.resolve())
        self.assertEqual((project / config.pop("icon")).resolve(), (app / expected.pop("icon")).resolve())
        self.assertEqual(config, expected)
        # The same directory, however the link names it (Windows: a junction, maybe in 8.3 names).
        self.assertTrue(os.path.samefile(project / "resources", app / "resources"))
        self.assertEqual((app / "mygo.json").read_bytes(), production)


if __name__ == "__main__":
    unittest.main()
