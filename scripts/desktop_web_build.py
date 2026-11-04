#!/usr/bin/env python3
"""Build Markitai Web (desktop/web/) around a released markitai CLI.

  stage --archive PATH [--sums SHA256SUMS]
        Replace the Git-ignored desktop/web/resources/ with the archive's
        markitai, at <goos>-<goarch>/bin/, and the app's licences: the
        archive's attribution under licenses/markitai/, Go's and every linked
        module's under licenses/go/. With --sums the archive's SHA-256 must
        be the one SHA256SUMS lists.
  build --platform darwin/arm64|linux/amd64|windows/amd64 [--e2e]
        `go tool mygo build` of the staged app in
        desktop/web/build/<goos>-<goarch>/: Markitai Web.app and
        markitai-web-<version>-arm64.dmg for macOS; the markitai-web
        executable with bin/ and licenses/ beside it,
        markitai-web_<version>_amd64.deb,
        markitai-web-<version>-linux-amd64.tar.gz with install.sh, and a
        PKGBUILD for Linux, each installing the desktop entry as
        dev.markitai.web.desktop (finish_linux); Markitai Web.exe
        with bin/ and licenses/ beside it and the installer
        markitai-web-setup-<version>-amd64.exe for Windows. --e2e makes
        the -debug build with the automation plugin, without the disk image,
        the installer or the Linux packages, in desktop/web/build/e2e/, for
        scripts/desktop_web_e2e.py: from the project
        desktop/web/build/e2e-project/, desktop/web/mygo.json with its own
        identifier (".e2e" added), so it keeps no web view data where an
        installed Markitai Web does.
  vet   gofmt, go vet and go test of desktop/web/ for darwin, linux and
        windows; tests run on this system and compile for the others.

The input is a release-shaped CLI archive (`release_build.py cli`, or the
GitHub release's); nothing here downloads. The app's version is the
workspace's: `release_build.py prepare-version` writes desktop/web/mygo.json,
and stage refuses an archive when Cargo.toml, mygo.json and the archive name
disagree. No artifact may be named like a CLI archive (markitai-v...), which
the CLI's updater and setup scripts match.
"""
from pathlib import Path
import argparse
import hashlib
import io
import json
import os
import platform as host
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import tomllib

from package_cli_target import workspace_version
from release_build import TARGETS, archive_attribution, archive_name, extract_release_archive

ROOT = Path(__file__).resolve().parents[1]
# The app's Go module and mygo project, from the checkout.
APP_DIR = "desktop/web"
# mygo platform -> release target; the platform, "/" made "-", names mygo's
# resource directory (resources/darwin-arm64).
PLATFORMS = {
    "darwin/arm64": "aarch64-apple-darwin",
    "darwin/amd64": "x86_64-apple-darwin",
    "linux/amd64": "x86_64-unknown-linux-gnu",
    "linux/arm64": "aarch64-unknown-linux-gnu",
    "windows/amd64": "x86_64-pc-windows-msvc",
    "windows/arm64": "aarch64-pc-windows-msvc",
}
# Built and verified this round; the others stage and compile only.
SHIPPED = {"darwin/arm64", "linux/amd64", "windows/amd64"}
CLI_ASSET_PREFIX = "markitai-v"
# Licence files at a Go module's root.
LICENCE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "PATENTS")
# What stage records beside the resources; hidden, so mygo does not ship it.
STAMP = ".stage.json"
# The e2e build's mygo project, within APP_DIR, and what it adds to the app's identifier.
E2E_PROJECT = "build/e2e-project"
E2E_IDENTIFIER = ".e2e"
# APP_DIR/mygo.json's slug, which names the artifacts, and identifier, the app's on Linux
# (toolkit_linux.go): its window's class and the name of its installed desktop entry.
SLUG = "markitai-web"
APP_ID = "dev.markitai.web"
# The Linux entry is Markitai Native's: keys mygo build writes from APP_DIR/mygo.json, which must
# be Native's, by the field that sets each, and keys mygo build writes none of, taken as they are.
NATIVE_ENTRY = "desktop/native/linux/data/dev.markitai.native.desktop"
SAME_AS_NATIVE = {"Comment": "linux.comment", "Categories": "linux.categories"}
FROM_NATIVE = ("Comment[zh_CN]", "Keywords", "StartupNotify")
# The Debian package's long description, where mygo build writes "Markitai Web is a desktop application."
DEB_DESCRIPTION = ("Markitai Web opens the markitai workbench in a window of its own. The markitai bundled with it "
                   "serves the workbench on this computer and converts documents, web pages and images to "
                   "Markdown, as the CLI of the same version does.")


def resources_dir(platform):
    return platform.replace("/", "-")


def program_name(platform):
    return "markitai.exe" if platform.startswith("windows/") else "markitai"


def versions(root):
    """(Cargo workspace version, desktop/web/mygo.json version)."""
    cargo = workspace_version((root / "Cargo.toml").read_text(encoding="utf-8"))
    app = json.loads((root / APP_DIR / "mygo.json").read_text(encoding="utf-8"))["version"]
    return cargo, app


def app_version(root):
    cargo, app = versions(root)
    if cargo != app:
        raise SystemExit(f"{APP_DIR}/mygo.json says {app}, Cargo.toml {cargo}; "
                         f"run release_build.py prepare-version or fix {APP_DIR}/mygo.json")
    return app


def archive_platform(archive, version):
    """The platform whose release archive of `version` is named like `archive`."""
    for platform, target in PLATFORMS.items():
        if archive.name == archive_name(version, target):
            return platform
    raise SystemExit(f"{archive.name} is not a markitai {version} release archive "
                     f"(expected {archive_name(version, PLATFORMS['darwin/arm64'])} or another target's)")


def check_sum(archive, sums):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    for line in sums.read_text(encoding="utf-8").splitlines():
        listed, _, name = line.partition(" ")
        if name.lstrip(" *") == archive.name:
            if listed != digest:
                raise SystemExit(f"{archive.name}: SHA-256 {digest} differs from {sums.name}'s {listed}")
            return
    raise SystemExit(f"{sums.name} does not list {archive.name}")


def host_platform():
    system = {"Darwin": "darwin", "Linux": "linux", "Windows": "windows"}.get(host.system())
    machine = {"arm64": "arm64", "aarch64": "arm64", "x86_64": "amd64", "AMD64": "amd64"}.get(host.machine())
    return f"{system}/{machine}"


def probe_version(program, version):
    """`markitai --version` must name `version`; run with its own HOME and MARKITAI_HOME."""
    with tempfile.TemporaryDirectory() as work:
        environment = dict(os.environ, HOME=work, USERPROFILE=work, APPDATA=work, LOCALAPPDATA=work, TEMP=work,
                           TMP=work, MARKITAI_HOME=str(Path(work) / "state"), MARKITAI_LANG="en")
        try:
            result = subprocess.run([str(program), "--version"], capture_output=True, text=True, timeout=60,
                                    env=environment, cwd=work)
        except subprocess.TimeoutExpired:
            raise SystemExit(f"{program} --version did not answer within 60 s") from None
    printed = result.stdout.strip()
    if printed != f"markitai {version}":
        stderr = result.stderr.strip()[-2000:]
        raise SystemExit(f"{program} --version printed {printed!r} and exited {result.returncode}, "
                         f"not 'markitai {version}'" + (f"\n{stderr}" if stderr else ""))


def go(arguments, cwd, environment=None, capture=False):
    command = ["go", *arguments]
    print("+ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=environment, check=False, text=True,
                            stdout=subprocess.PIPE if capture else None)
    if result.returncode:
        raise SystemExit(f"{' '.join(command)} exited {result.returncode}")
    return result.stdout


def go_environment(platform, **extra):
    goos, goarch = platform.split("/")
    return dict(os.environ, GOOS=goos, GOARCH=goarch, CGO_ENABLED="0", **extra)


def go_modules(root, platform):
    """(path, version, directory) of every module linked into the app for `platform`."""
    listed = go(["list", "-deps", "-f",
                 "{{with .Module}}{{if not .Main}}{{.Path}}\t{{.Version}}\t{{.Dir}}{{end}}{{end}}", "."],
                cwd=root / APP_DIR, environment=go_environment(platform), capture=True)
    return sorted({tuple(line.split("\t")) for line in listed.splitlines() if line})


def go_root():
    return Path(go(["env", "GOROOT"], cwd=ROOT, capture=True).strip())


def licence_files(directory):
    return sorted(path for path in Path(directory).iterdir()
                  if path.is_file() and path.name.upper().startswith(LICENCE_PREFIXES))


def stage_go_licences(destination, modules, goroot):
    """Go's licence and each module's, with a README listing them; a module without one fails."""
    rows = []
    for owner, version, directory in [("Go", "", goroot), *modules]:
        if not str(directory):
            raise SystemExit(f"{owner} {version}: go list gave no module directory; run go mod download")
        found = licence_files(directory)
        if not found:
            raise SystemExit(f"{owner} {version}: no licence file in {directory}")
        target = destination / ("go" if owner == "Go" else owner)
        target.mkdir(parents=True, exist_ok=True)
        for path in found:
            shutil.copyfile(path, target / path.name)
        where = target.relative_to(destination).as_posix()
        rows.append(f"- {owner} {version}".rstrip() + ": " + ", ".join(f"{where}/{path.name}" for path in found))
    (destination / "README.md").write_text(
        "# Go licences\n\nThe Markitai Web app is a Go program. It links the Go standard library and "
        "these modules:\n\n" + "\n".join(rows) + "\n", encoding="utf-8")


def stage(archive, sums=None, *, root=ROOT, modules=None, goroot=None, host_system=None):
    """Replace desktop/web/resources with the archive's program and the app's licences. The new
    resources are made apart and replace the old ones only once complete."""
    version = app_version(root)
    platform = archive_platform(archive, version)
    if sums is not None:
        check_sum(archive, sums)
    resources = root / APP_DIR / "resources"
    with tempfile.TemporaryDirectory() as work:
        staged = Path(work) / "resources"
        extracted = Path(work) / "archive"
        extract_release_archive(archive, extracted)
        name = program_name(platform)
        program = staged / resources_dir(platform) / "bin" / name
        program.parent.mkdir(parents=True)
        shutil.copyfile(extracted / name, program)
        program.chmod(0o755)
        if (host_system or host_platform()) == platform:
            probe_version(program, version)
        for relative, data in archive_attribution(archive).items():
            path = staged / "licenses/markitai" / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        stage_go_licences(staged / "licenses/go",
                          go_modules(root, platform) if modules is None else modules,
                          go_root() if goroot is None else goroot)
        stamp = {"version": version, "platform": platform, "archive": archive.name,
                 "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
        (staged / STAMP).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
        if resources.exists():
            shutil.rmtree(resources)
        shutil.move(staged, resources)
    print(f"Staged markitai {version} for {platform} in {resources.relative_to(root)}")
    return stamp


def check_artifacts(names):
    taken = sorted(name for name in names if name.startswith(CLI_ASSET_PREFIX))
    if taken:
        raise SystemExit(f"Markitai Web's artifacts must not be named like CLI archives ({CLI_ASSET_PREFIX}...): "
                         + ", ".join(taken))


def mygo_build_command(platform, e2e, goflags=""):
    """`go tool mygo build` arguments, run in APP_DIR, and the extra environment; the e2e build of
    E2E_PROJECT keeps the inspector the automation plugin needs, makes no disk image or installer
    (mygo build has no flag to skip the Linux packages, which build deletes), goes to build/e2e and
    adds its tag to `goflags`: mygo build takes tags from GOFLAGS only."""
    arguments = ["tool", "mygo", "build", "-platform", platform]
    if e2e:
        return (arguments + ["-debug", "-skip-dmg", "-skip-installer", "-o", "../e2e", E2E_PROJECT],
                {"GOFLAGS": f"{goflags} -tags=e2e".strip()})
    return arguments, {}


def e2e_project(root=ROOT):
    """Write E2E_PROJECT: APP_DIR's mygo.json under the identifier plus E2E_IDENTIFIER, building the
    APP_DIR package with its resources. WKWebView keeps website data (localStorage, caches) by the
    bundle identifier outside HOME, so the e2e build must not share an installed app's."""
    app = root / APP_DIR
    project = app / E2E_PROJECT
    if project.exists():
        shutil.rmtree(project)
    project.mkdir(parents=True)
    config = json.loads((app / "mygo.json").read_text(encoding="utf-8"))
    back = os.path.relpath(app, project)
    config.update(identifier=config["identifier"] + E2E_IDENTIFIER, main=back,
                  icon=os.path.join(back, config["icon"]))
    (project / "mygo.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    try:
        (project / "resources").symlink_to(os.path.join(back, "resources"), target_is_directory=True)
    except OSError as error:
        # Without Developer Mode or the symlink privilege, a junction, which mygo reads the same.
        if os.name != "nt" or error.winerror != 1314:
            raise
        subprocess.run(["cmd.exe", "/d", "/c", "mklink", "/J", str(project / "resources"), str(app / "resources")],
                       check=True, stdout=subprocess.DEVNULL)
    return project


def deb_version(version):
    """mygo's Debian version of `version`: a pre-release sorts before its release with "~"."""
    release, dash, pre = version.removeprefix("v").partition("-")
    return (release + ("~" + pre.replace("-", ".") if dash else "")).replace("+", "~")


def layout(platform, version):
    """What `mygo build` writes to APP_DIR/build/<goos>-<goarch>/ for `platform`: the app as it runs,
    its markitai relative to that directory, and the packages, with the files they are made from,
    that only a release build keeps. On Linux mygo build makes them with the e2e build too, and build
    deletes them there; the app's icon and desktop entry beside it are the app's, as install.sh
    installs it."""
    goos, goarch = platform.split("/")
    program = program_name(platform)
    if goos == "darwin":
        return "Markitai Web.app", f"Markitai Web.app/Contents/Resources/bin/{program}", {
            f"{SLUG}-{version}-{goarch}.dmg"}
    if goos == "linux":
        return SLUG, f"bin/{program}", {
            f"{SLUG}_{deb_version(version)}_{goarch}.deb", f"{SLUG}-{version}-linux-{goarch}.tar.gz",
            "install.sh", "PKGBUILD", f"{SLUG}-256x256.png", f"{SLUG}-512x512.png"}
    if goos == "windows":
        return "Markitai Web.exe", f"bin/{program}", {f"{SLUG}-setup-{version}-{goarch}.exe"}
    raise SystemExit(f"No app layout for {platform}")


def native_entry(root):
    """Markitai Native's desktop entry, key -> value."""
    lines = (root / NATIVE_ENTRY).read_text(encoding="utf-8").splitlines()
    return dict(line.split("=", 1) for line in lines if "=" in line)


def desktop_entry(text, native):
    """text, a desktop entry as mygo build writes it, with Native's keys and the window's class."""
    lines = text.rstrip("\n").split("\n")
    for key, field in SAME_AS_NATIVE.items():
        if [line for line in lines if line.startswith(key + "=")] != [f"{key}={native[key]}"]:
            raise SystemExit(f"the desktop entry's {key} is not Markitai Native's {native[key]!r}: "
                             f"make {field} of {APP_DIR}/mygo.json match")
    for key in (*FROM_NATIVE, "StartupWMClass"):
        if any(line.startswith(key + "=") for line in lines):
            raise SystemExit(f"mygo build now writes {key} in the desktop entry: drop it from desktop_web_build.py")
    lines.insert(lines.index("Comment=" + native["Comment"]) + 1, "Comment[zh_CN]=" + native["Comment[zh_CN]"])
    lines += [f"Keywords={native['Keywords']}", f"StartupNotify={native['StartupNotify']}",
              f"StartupWMClass={APP_ID}"]
    return "\n".join(lines) + "\n"


def replace_once(text, old, new, where):
    if text.count(old) != 1:
        raise SystemExit(f"{where}: expected {old!r} once, as mygo build writes it")
    return text.replace(old, new)


def rewrite_tar(data, changes, where):
    """data, a .tar.gz, with each regular file `changes` names as its change(body) -> (name, body)
    makes it, and the other members as they were. Each named file must be there."""
    out, seen = io.BytesIO(), set()
    with (tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as source,
          tarfile.open(fileobj=out, mode="w:gz", format=tarfile.PAX_FORMAT) as target):
        for member in source.getmembers():
            body = source.extractfile(member).read() if member.isfile() else None
            if member.isfile() and member.name in changes:
                seen.add(member.name)
                member.name, body = changes[member.name](body)
                member.size = len(body)
            target.addfile(member, None if body is None else io.BytesIO(body))
    if seen != changes.keys():
        raise SystemExit(f"{where}: no {', '.join(sorted(changes.keys() - seen))}, as mygo build writes it")
    return out.getvalue()


def read_ar(data, where):
    """The (header, body) members of an ar archive, as a Debian package is."""
    if not data.startswith(b"!<arch>\n"):
        raise SystemExit(f"{where} is not an ar archive")
    members, at = [], 8
    while at < len(data):
        header = data[at:at + 60]
        size = int(header[48:58])
        members.append((header, data[at + 60:at + 60 + size]))
        at += 60 + size + size % 2
    return members


def write_ar(members):
    out = bytearray(b"!<arch>\n")
    for header, body in members:
        out += header[:48] + str(len(body)).encode().ljust(10) + header[58:60] + body
        if len(body) % 2:
            out += b"\n"
    return bytes(out)


def finish_linux(out, version, root=ROOT):
    """mygo build's Linux packages in `out` as Native's are. The app runs as APP_ID (toolkit_linux.go),
    but mygo build names the desktop entry after the slug, which cannot hold a dot, and writes no
    Comment[zh_CN], Keywords, StartupNotify or StartupWMClass (linuxDesktopEntry, cmd/mygo/build.go).
    Every copy of the entry gets those keys: beside the app and in the .tar.gz, where install.sh and
    mygo's updater read it by the slug's name, and where install.sh, the Debian package and the
    PKGBUILD install it, as APP_ID.desktop. The Debian package gets the app's long description, the
    PKGBUILD the workspace's licence, which mygo build reads from a package.json the app does not
    have, with the app's licences in /usr/share/licenses/SLUG, and install.sh --uninstall removes the
    MIME cache it leaves with no type in it. It fails rather than guess when mygo build writes any
    of them differently, and writes nothing until all of them hold."""
    native = native_entry(root)
    with open(root / "Cargo.toml", "rb") as file:
        licence = tomllib.load(file)["workspace"]["package"]["license"]
    goarch = out.name.removeprefix("linux-")
    template = f"{SLUG}.desktop"
    installed = f"usr/share/applications/{SLUG}.desktop"
    renamed = f"usr/share/applications/{APP_ID}.desktop"

    def entry(body):
        return desktop_entry(body.decode("utf-8"), native).encode("utf-8")

    written = {out / template: entry((out / template).read_bytes())}
    archive = out / f"{SLUG}-{version}-linux-{goarch}.tar.gz"
    before = archive.read_bytes()
    written[archive] = rewrite_tar(before, {template: lambda body: (template, entry(body))}, archive.name)

    script = out / "install.sh"
    text = replace_once(script.read_bytes().decode("utf-8"), 'desktop_file="$data_home/applications/$name.desktop"',
                        f'desktop_file="$data_home/applications/{APP_ID}.desktop"', script.name)
    written[script] = replace_once(text, '\tupdate_databases\n\techo "Uninstalled $app_name.', (
        '\tupdate_databases\n'
        "\t# update-desktop-database, as install.sh ran it, made the cache; with no entry left that\n"
        "\t# lists a type, it holds only its header.\n"
        '\tcache="$data_home/applications/mimeinfo.cache"\n'
        "\tif [ -f \"$cache\" ] && [ \"$(cat \"$cache\")\" = '[MIME Cache]' ]; then\n"
        '\t\trm -f "$cache"\n'
        "\tfi\n"
        '\techo "Uninstalled $app_name.'), script.name).encode("utf-8")

    pkgbuild = out / "PKGBUILD"
    text = replace_once(pkgbuild.read_bytes().decode("utf-8"), "license=('LicenseRef-custom')",
                        f"license=('{licence}')", pkgbuild.name)
    copy = f'\tcp -dR -- bin licenses {SLUG} "$pkgdir"/opt/{SLUG}/\n'
    text = replace_once(text, copy, copy + '\tinstall -d "$pkgdir"/usr/share/licenses\n'
                        f'\tln -s ../../../opt/{SLUG}/licenses "$pkgdir"/usr/share/licenses/{SLUG}\n', pkgbuild.name)
    inline = re.search(r"printf '%s' '(\[Desktop Entry\]\n.*?)' \|", text, re.S)
    if not inline:
        raise SystemExit(f"{pkgbuild.name}: no desktop entry as mygo build writes it")
    quoted = desktop_entry(inline.group(1), native).replace("'", "'\\''")
    text = replace_once(text[:inline.start(1)] + quoted + text[inline.end(1):], f'"$pkgdir"/{installed}',
                        f'"$pkgdir"/{renamed}', pkgbuild.name)
    written[pkgbuild] = replace_once(text, hashlib.sha256(before).hexdigest(),
                                     hashlib.sha256(written[archive]).hexdigest(), pkgbuild.name).encode("utf-8")

    deb = out / f"{SLUG}_{deb_version(version)}_{goarch}.deb"
    members = read_ar(deb.read_bytes(), deb.name)
    if [header[:16].decode().strip() for header, _ in members] != ["debian-binary", "control.tar.gz", "data.tar.gz"]:
        raise SystemExit(f"{deb.name}: not the members mygo build writes")
    data = rewrite_tar(members[2][1], {"./" + installed: lambda body: ("./" + renamed, entry(body))}, deb.name)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as files:
        digest = hashlib.md5(files.extractfile("./" + renamed).read()).hexdigest()

    def in_control(body):
        text = body.decode("utf-8")
        if not re.search(r"^Maintainer: [^<>\n]+ <[^<>@\s]+@[^<>\s]+>$", text, re.M):
            raise SystemExit(f"{deb.name}: its Maintainer is not 'Name <email>': set linux.maintainer of "
                             f"{APP_DIR}/mygo.json")
        description = "".join(f" {line}\n" for line in textwrap.wrap(DEB_DESCRIPTION, 76))
        return "./control", replace_once(text, " Markitai Web is a desktop application.\n", description,
                                         f"{deb.name} control").encode("utf-8")

    def in_md5sums(body):
        text, found = re.subn(rf"^[0-9a-f]{{32}}  {re.escape(installed)}$", f"{digest}  {renamed}",
                              body.decode("utf-8"), flags=re.M)
        if found != 1:
            raise SystemExit(f"{deb.name}: md5sums names {installed} {found} times")
        return "./md5sums", text.encode("utf-8")

    control = rewrite_tar(members[1][1], {"./control": in_control, "./md5sums": in_md5sums}, deb.name)
    written[deb] = write_ar([members[0], (members[1][0], control), (members[2][0], data)])
    for path, body in written.items():
        path.write_bytes(body)


def build(platform, e2e=False, *, root=ROOT):
    if platform not in SHIPPED:
        raise SystemExit(f"This round builds {', '.join(sorted(SHIPPED))}, not {platform}")
    version = app_version(root)
    resources = root / APP_DIR / "resources"
    stamp_path = resources / STAMP
    stamp = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.exists() else None
    if stamp is None or (stamp.get("version"), stamp.get("platform")) != (version, platform):
        held = "nothing staged" if stamp is None else f"{stamp.get('platform')} {stamp.get('version')}"
        raise SystemExit(f"{APP_DIR}/resources holds {held}, not {platform} {version}; run: desktop_web_build.py "
                         f"stage --archive {archive_name(version, PLATFORMS[platform])}")
    arguments, extra = mygo_build_command(platform, e2e, os.environ.get("GOFLAGS", ""))
    out = root / APP_DIR / ("build/e2e" if e2e else "build") / resources_dir(platform)
    if out.exists():
        shutil.rmtree(out)
    if e2e:
        e2e_project(root)
    go(arguments, cwd=root / APP_DIR, environment=dict(os.environ, **extra))
    app, program, packages = layout(platform, version)
    if e2e:
        for name in packages:
            (out / name).unlink(missing_ok=True)
    names = sorted(path.name for path in out.iterdir())
    check_artifacts(names)
    if app not in names or packages & set(names) != (set() if e2e else packages):
        raise SystemExit(f"{out} has {names}; expected {app} and {'none' if e2e else 'all'} of {sorted(packages)}")
    bundled = out / program
    if not os.access(bundled, os.X_OK):
        raise SystemExit(f"{bundled} is missing or not executable")
    if platform.startswith("linux/") and not e2e:
        finish_linux(out, version, root)
    for name in names:
        print(out / name)
    return out


def vet(root=ROOT):
    app = root / APP_DIR
    gofmt = go_root() / "bin" / ("gofmt.exe" if os.name == "nt" else "gofmt")
    unformatted = subprocess.run([str(gofmt), "-l", "."], cwd=app, capture_output=True, text=True,
                                 check=True).stdout.split()
    if unformatted:
        raise SystemExit("gofmt would change: " + ", ".join(unformatted))
    native = host_platform()
    for system in ["darwin", "linux", "windows"]:
        platform = native if native.startswith(system + "/") else f"{system}/amd64"
        environment = go_environment(platform)
        go(["vet", "./..."], cwd=app, environment=environment)
        if platform == native:
            go(["vet", "-tags", "e2e", "./..."], cwd=app, environment=environment)
            go(["test", "-count=1", "./..."], cwd=app, environment=environment)
        else:
            with tempfile.TemporaryDirectory() as work:
                go(["test", "-c", "-o", str(Path(work) / "web.test"), "."], cwd=app,
                   environment=environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage",
                                  help=f"put a release archive's markitai and the licences in {APP_DIR}/resources")
    staging.add_argument("--archive", type=Path, required=True)
    staging.add_argument("--sums", type=Path, help="the release's SHA256SUMS")
    building = commands.add_parser("build", help="build the app from the staged resources")
    building.add_argument("--platform", required=True, choices=sorted(SHIPPED))
    building.add_argument("--e2e", action="store_true", help="the -debug build scripts/desktop_web_e2e.py drives")
    commands.add_parser("vet", help="gofmt, go vet and go test for darwin, linux and windows")
    args = parser.parse_args(argv)
    if args.command == "stage":
        stage(args.archive.resolve(), args.sums.resolve() if args.sums else None)
    elif args.command == "build":
        build(args.platform, args.e2e)
    else:
        vet()
    return 0


if __name__ == "__main__":
    sys.exit(main())
