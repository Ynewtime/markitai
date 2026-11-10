"""desktop_web_build.py with synthetic archives, Go modules and mygo builds; no Go, mygo or network."""
from pathlib import Path
from unittest import mock
import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import unittest
import zipfile

import bounded_run
import desktop_web_build
from desktop_web_build import (APP_DIR, DEB_DESCRIPTION, ENTRY_KEYS, PLATFORMS, SHIPPED, STAMP, build,
                               check_artifacts, e2e_project, layout, mygo_build_command, read_ar, stage)
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
        (self.root / "Cargo.toml").write_text(f'[workspace.package]\nversion = "{cargo}"\nlicense = "MIT"\n',
                                              encoding="utf-8")
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
        self.assertEqual(packages, {"markitai-web_2.3.2_amd64.deb", "markitai-web-2.3.2-linux-amd64.tar.gz",
                                    "install.sh", "PKGBUILD", "markitai-web-256x256.png", "markitai-web-512x512.png"})
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


# install.sh as mygo build writes it, with only what --uninstall runs.
INSTALL_SH = """#!/bin/sh
set -eu

app_name='Markitai Web'
name='markitai-web'

fail() {
\techo "install.sh: $*" >&2
\texit 1
}

main() {
\tlinux "$@"
}

linux() {
\tapp_dir="$HOME/.local/$name.app"
\tbin_dir="$HOME/.local/bin"
\tdata_home="${XDG_DATA_HOME:-$HOME/.local/share}"
\tdesktop_file="$data_home/applications/$name.desktop"
\tmime_file="$data_home/mime/packages/$name.xml"
\tif [ "${1:-}" = --uninstall ]; then
\t\tuninstall
\t\treturn
\tfi
\tfail "this sample only uninstalls"
}

uninstall() {
\t[ -d "$app_dir" ] || fail "$app_name is not installed in $app_dir"
\tif [ -f "$desktop_file" ] && grep -qF "$app_dir/" "$desktop_file"; then
\t\trm -f "$desktop_file" "$mime_file"
\tfi
\trm -rf "$app_dir"
\tupdate_databases
\techo "Uninstalled $app_name. Its settings and data are still in place."
}

update_databases() {
\tif command -v update-desktop-database >/dev/null 2>&1; then
\t\tupdate-desktop-database "$data_home/applications" >/dev/null 2>&1 || true
\tfi
}

main "$@"
"""


def tar_gz(members, prefix="", owner=""):
    """A .tar.gz of (name, mode, bytes, or None for a directory, or a link target str)."""
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w:gz", format=tarfile.USTAR_FORMAT) as bundle:
        for name, mode, data in members:
            info = tarfile.TarInfo(prefix + name)
            info.mode, info.uname, info.gname, info.mtime = mode, owner, owner, 1_790_000_000
            if data is None:
                info.type = tarfile.DIRTYPE
            elif isinstance(data, str):
                info.type, info.linkname = tarfile.SYMTYPE, data
            else:
                info.size = len(data)
            bundle.addfile(info, io.BytesIO(data) if isinstance(data, bytes) else None)
    return out.getvalue()


def read_tar(data):
    """name -> (mode, bytes, or None for a directory, or a link target str) of a .tar.gz."""
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as bundle:
        return {member.name: (member.mode, bundle.extractfile(member).read() if member.isfile()
                              else member.linkname if member.issym() else None) for member in bundle.getmembers()}


def entry_keys(text):
    return dict(line.split("=", 1) for line in text.splitlines() if "=" in line)


class MygoLinux:
    """What mygo build writes for linux/amd64, in small: the app and its packages, each with the
    desktop entry of linuxDesktopEntry for APP_DIR/mygo.json, ending with `tail`."""

    def __init__(self, root, tail=""):
        self.root = root
        self.tail = tail
        self.made = {}

    def entry(self, run):
        config = json.loads((self.root / APP_DIR / "mygo.json").read_text(encoding="utf-8"))
        linux = config["linux"]
        return (f"[Desktop Entry]\nType=Application\nName={config['name']}\nComment={linux['comment']}\n"
                f"Exec={run}\nIcon=markitai-web\nCategories={';'.join(linux['categories'])};\n"
                f"Terminal=false\n{self.tail}").encode()

    def __call__(self, arguments, cwd, environment=None, capture=False):
        out = self.root / APP_DIR / ("build/e2e" if "-debug" in arguments else "build") / "linux-amd64"
        out.mkdir(parents=True)
        config = json.loads((self.root / APP_DIR / "mygo.json").read_text(encoding="utf-8"))
        app = [("bin", 0o755, None), ("bin/markitai", 0o755, b"#!/bin/sh\n"), ("licenses", 0o755, None),
               ("licenses/markitai", 0o755, None), ("licenses/markitai/LICENSE", 0o644, b"MIT"),
               ("markitai-web", 0o755, b"\x7fELF"), ("markitai-web.desktop", 0o644, self.entry("markitai-web")),
               ("markitai-web.png", 0o644, b"png")]
        for name, mode, data in app:
            if data is None:
                (out / name).mkdir()
            else:
                (out / name).write_bytes(data)
                (out / name).chmod(mode)
        archive = tar_gz(app)
        icons = {"256x256": b"png 256", "512x512": b"png 512"}
        data = [(name, 0o755, None) for name in ["opt", "opt/markitai-web", "usr", "usr/share",
                                                  "usr/share/applications"]]
        data += [(f"opt/markitai-web/{name}", mode, body) for name, mode, body in app
                 if name not in ("markitai-web.desktop", "markitai-web.png")]
        data += [("usr/bin", 0o755, None), ("usr/bin/markitai-web", 0o777, "../../opt/markitai-web/markitai-web"),
                 ("usr/share/applications/markitai-web.desktop", 0o644, self.entry("/opt/markitai-web/markitai-web"))]
        data += [(f"usr/share/icons/hicolor/{size}/apps/markitai-web.png", 0o644, body) for size, body in icons.items()]
        md5sums = "".join(f"{hashlib.md5(body).hexdigest()}  {name}\n" for name, _, body in data
                          if isinstance(body, bytes)).encode()
        control = (f"Package: markitai-web\nVersion: {VERSION}\nArchitecture: amd64\n"
                   f"Maintainer: {config['linux']['maintainer']}\nInstalled-Size: 1\n"
                   "Depends: libgtk-3-0 | libgtk-3-0t64, libwebkit2gtk-4.1-0\nSection: utils\nPriority: optional\n"
                   f"Description: {config['linux']['comment']}\n Markitai Web is a desktop application.\n").encode()
        deb = b"!<arch>\n"
        for name, body in [("debian-binary", b"2.0\n"),
                           ("control.tar.gz", tar_gz([("control", 0o644, control), ("md5sums", 0o644, md5sums)],
                                                     "./", "root")),
                           ("data.tar.gz", tar_gz(data, "./", "root"))]:
            deb += f"{name:<16}{1_790_000_000:<12}{0:<6}{0:<6}{'100644':<8}{len(body):<10}`\n".encode() + body
            deb += b"\n" if len(body) % 2 else b""
        entry = self.entry("/opt/markitai-web/markitai-web").decode()
        sources = {f"markitai-web-{VERSION}-linux-amd64.tar.gz": archive,
                   **{f"markitai-web-{size}.png": body for size, body in icons.items()}}
        pkgbuild = (
            f"# Maintainer: {config['linux']['maintainer']}\n"
            "# Written by mygo build next to the app's archive: makepkg -si packages\n"
            "# the app for Arch Linux and installs it.\n"
            f"pkgname='markitai-web'\npkgver='{VERSION}'\npkgrel=1\npkgdesc='{config['linux']['comment']}'\n"
            "arch=('x86_64')\nlicense=('LicenseRef-custom')\ndepends=('gtk3' 'webkit2gtk-4.1')\n"
            "options=('!strip' '!debug')\n"
            f"source=({' '.join(repr(name) for name in sources)})\n"
            f"sha256sums=({' '.join(repr(hashlib.sha256(body).hexdigest()) for body in sources.values())})\n\n"
            "package() {\n\tcd \"$srcdir\"\n\tinstall -d \"$pkgdir\"/opt/markitai-web\n"
            "\tcp -dR -- bin licenses markitai-web \"$pkgdir\"/opt/markitai-web/\n"
            "\t# The modes of the Debian package: 755 for directories and programs, 644\n\t# for the other files.\n"
            "\tfind \"$pkgdir\"/opt/markitai-web -type d -exec chmod 755 {} +\n"
            "\tinstall -d \"$pkgdir\"/usr/bin\n"
            "\tln -s ../../opt/markitai-web/markitai-web \"$pkgdir\"/usr/bin/markitai-web\n"
            f"\tprintf '%s' '{entry}' |\n"
            "\t\tinstall -Dm644 /dev/stdin \"$pkgdir\"/usr/share/applications/markitai-web.desktop\n"
            + "".join(f"\tinstall -Dm644 markitai-web-{size}.png \"$pkgdir\"/usr/share/icons/hicolor/{size}/apps/"
                      "markitai-web.png\n" for size in icons) + "}\n").encode()
        packages = {**sources, f"markitai-web_{VERSION}_amd64.deb": deb, "PKGBUILD": pkgbuild,
                    "install.sh": INSTALL_SH.encode()}
        for name, body in packages.items():
            (out / name).write_bytes(body)
        (out / "install.sh").chmod(0o755)
        self.made = {path.name: path.read_bytes() for path in out.iterdir() if path.is_file()}


class LinuxPackageTests(unittest.TestCase):
    def setUp(self):
        bounded_run.watch(self)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.repo = Checkout(directory.name)
        self.repo.resources.mkdir()
        (self.repo.resources / STAMP).write_text(json.dumps({"version": VERSION, "platform": "linux/amd64"}),
                                                 encoding="utf-8")

    def build(self, e2e=False, tail=""):
        self.mygo = MygoLinux(self.repo.root, tail)
        with mock.patch.object(desktop_web_build, "go", self.mygo), contextlib.redirect_stdout(io.StringIO()):
            return build("linux/amd64", e2e, root=self.repo.root)

    def assert_entry(self, data, run):
        keys = entry_keys(data.decode("utf-8"))
        self.assertEqual((keys["Comment"], keys["Categories"]),
                         ("Convert documents and web pages to Markdown", "Office;Utility;"))
        for key, value in ENTRY_KEYS.items():
            self.assertEqual(keys[key], value, key)
        self.assertEqual((keys["StartupWMClass"], keys["Exec"], keys["Icon"]),
                         ("dev.markitai.web", run, "markitai-web"))

    def test_every_package_installs_the_finished_entry_as_the_app_id(self):
        out = self.build()
        self.assert_entry((out / "markitai-web.desktop").read_bytes(), "markitai-web")

        archive = (out / f"markitai-web-{VERSION}-linux-amd64.tar.gz").read_bytes()
        files = read_tar(archive)
        self.assert_entry(files["markitai-web.desktop"][1], "markitai-web")
        self.assertEqual(files["bin/markitai"], (0o755, b"#!/bin/sh\n"))
        self.assertEqual(files["licenses/markitai"], (0o755, None))

        script = (out / "install.sh").read_text(encoding="utf-8")
        self.assertIn('desktop_file="$data_home/applications/dev.markitai.web.desktop"', script)

        pkgbuild = (out / "PKGBUILD").read_text(encoding="utf-8")
        self.assertIn(f"sha256sums=('{hashlib.sha256(archive).hexdigest()}' ", pkgbuild)
        self.assertIn("license=('MIT')", pkgbuild)
        self.assertIn('\tln -s ../../../opt/markitai-web/licenses "$pkgdir"/usr/share/licenses/markitai-web\n',
                      pkgbuild)
        inline = pkgbuild.split("printf '%s' '")[1].split("' |\n")[0]
        self.assert_entry(inline.encode(), "/opt/markitai-web/markitai-web")
        self.assertIn('install -Dm644 /dev/stdin "$pkgdir"/usr/share/applications/dev.markitai.web.desktop\n', pkgbuild)
        self.assertNotIn("applications/markitai-web.desktop", pkgbuild)

        members = read_ar((out / f"markitai-web_{VERSION}_amd64.deb").read_bytes(), "deb")
        self.assertEqual([header[:16].strip() for header, _ in members],
                         [b"debian-binary", b"control.tar.gz", b"data.tar.gz"])
        control, data = read_tar(members[1][1]), read_tar(members[2][1])
        self.assertNotIn("./usr/share/applications/markitai-web.desktop", data)
        self.assert_entry(data["./usr/share/applications/dev.markitai.web.desktop"][1],
                                 "/opt/markitai-web/markitai-web")
        self.assertEqual(data["./usr/bin/markitai-web"], (0o777, "../../opt/markitai-web/markitai-web"))
        listed = dict(reversed(line.split("  ", 1)) for line in control["./md5sums"][1].decode().splitlines())
        self.assertEqual(listed, {name[2:]: hashlib.md5(body).hexdigest() for name, (_, body) in data.items()
                                  if isinstance(body, bytes)})
        fields = control["./control"][1].decode()
        self.assertIn("Maintainer: Ynewtime <27655771+Ynewtime@users.noreply.github.com>\n", fields)
        description = fields.split("Description: Convert documents and web pages to Markdown\n")[1]
        self.assertTrue(all(line.startswith(" ") and len(line) <= 80 for line in description.splitlines()))
        self.assertEqual(" ".join(line.strip() for line in description.splitlines()), DEB_DESCRIPTION)
        self.assertTrue(DEB_DESCRIPTION.startswith("Markitai Web "))

    def test_the_e2e_build_keeps_the_app_and_no_package(self):
        out = self.build(e2e=True)
        self.assertEqual(out, self.repo.root / APP_DIR / "build/e2e/linux-amd64")
        self.assertEqual(sorted(path.name for path in out.iterdir()),
                         ["bin", "licenses", "markitai-web", "markitai-web.desktop", "markitai-web.png"])
        self.assertTrue((out / "bin/markitai").is_file())

    def test_a_package_mygo_writes_differently_fails_and_stays_as_written(self):
        cases = {
            "Maintainer is not 'Name <email>'": ("linux", "maintainer", "Markitai Web"),
        }
        production = (self.repo.root / APP_DIR / "mygo.json").read_text(encoding="utf-8")
        for message, (section, key, value) in cases.items():
            with self.subTest(message):
                config = json.loads(production)
                config[section][key] = value
                (self.repo.root / APP_DIR / "mygo.json").write_text(json.dumps(config), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, re.escape(message)):
                    self.build()
                out = self.repo.root / APP_DIR / "build/linux-amd64"
                self.assertEqual({path.name: path.read_bytes() for path in out.iterdir() if path.is_file()},
                                 self.mygo.made)
        (self.repo.root / APP_DIR / "mygo.json").write_text(production, encoding="utf-8")
        with self.assertRaisesRegex(SystemExit, "mygo build now writes StartupWMClass"):
            self.build(tail="StartupWMClass=markitai-web\n")
        with self.assertRaisesRegex(SystemExit, "the desktop entry has 2 Comment lines, not one"):
            self.build(tail="Comment=Again\n")

    def test_uninstall_removes_the_mime_cache_it_leaves_with_no_type(self):
        if os.name == "nt" or not shutil.which("sh"):
            self.skipTest("install.sh runs on Linux; this needs a POSIX sh")
        script = self.build() / "install.sh"
        tools = self.directory / "tools"
        tools.mkdir()
        (tools / "update-desktop-database").write_text(
            "#!/bin/sh\n{ echo '[MIME Cache]'; grep -l '^MimeType=' \"$1\"/*.desktop 2>/dev/null"
            " | sed 's|.*/|text/plain=|'; } >\"$1/mimeinfo.cache\"\n", encoding="utf-8")
        (tools / "update-desktop-database").chmod(0o755)
        for other in ["", "MimeType=text/plain;\n"]:
            with self.subTest(other_entry=bool(other)):
                home = self.directory / f"home{len(other)}"
                app = home / ".local/markitai-web.app"
                applications = home / ".local/share/applications"
                app.mkdir(parents=True)
                applications.mkdir(parents=True)
                (applications / "dev.markitai.web.desktop").write_text(f'Exec="{app}/markitai-web"\n', encoding="utf-8")
                (applications / "other.desktop").write_text(f"[Desktop Entry]\n{other}", encoding="utf-8")
                (applications / "mimeinfo.cache").write_text("[MIME Cache]\n", encoding="utf-8")
                environment = {"PATH": f"{tools}{os.pathsep}{os.environ['PATH']}", "HOME": str(home),
                               "MARKITAI_HOME": str(home / ".markitai")}
                result = bounded_run.run(["sh", str(script), "--uninstall"], env=environment, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertFalse(app.exists())
                self.assertEqual(sorted(path.name for path in applications.iterdir()),
                                 ["mimeinfo.cache", "other.desktop"] if other else ["other.desktop"])


if __name__ == "__main__":
    unittest.main()
