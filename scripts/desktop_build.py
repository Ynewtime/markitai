#!/usr/bin/env python3
"""Build the Markitai desktop app (desktop/) around a released markitai CLI.

  stage --archive PATH [--sums SHA256SUMS]
        Replace the Git-ignored desktop/resources/ with the archive's
        markitai, at <goos>-<goarch>/bin/, and the app's licences: the
        archive's attribution under licenses/markitai/, Go's and every linked
        module's under licenses/go/. With --sums the archive's SHA-256 must
        be the one SHA256SUMS lists.
  build --platform darwin/arm64 [--e2e]
        `go tool mygo build` of the staged app: desktop/build/darwin-arm64/
        Markitai.app and markitai-desktop-<version>-arm64.dmg. --e2e makes
        the -debug build with the automation plugin, without a disk image,
        in desktop/build/e2e/, for scripts/desktop_e2e.py.
  vet   gofmt, go vet and go test of desktop/ for darwin, linux and windows;
        tests run on this system and compile for the others.

The input is a release-shaped CLI archive (`release_build.py cli`, or the
GitHub release's); nothing here downloads. The app's version is the
workspace's: `release_build.py prepare-version` writes desktop/mygo.json, and
stage refuses an archive when Cargo.toml, mygo.json and the archive name
disagree. No artifact may be named like a CLI archive (markitai-v...), which
the CLI's updater and setup scripts match.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform as host
import shutil
import subprocess
import sys
import tempfile

from package_cli_target import workspace_version
from release_build import TARGETS, archive_attribution, archive_name, extract_release_archive

ROOT = Path(__file__).resolve().parents[1]
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
SHIPPED = {"darwin/arm64"}
CLI_ASSET_PREFIX = "markitai-v"
# Licence files at a Go module's root.
LICENCE_PREFIXES = ("LICENSE", "LICENCE", "COPYING", "NOTICE", "PATENTS")
# What stage records beside the resources; hidden, so mygo does not ship it.
STAMP = ".stage.json"


def resources_dir(platform):
    return platform.replace("/", "-")


def program_name(platform):
    return "markitai.exe" if platform.startswith("windows/") else "markitai"


def versions(root):
    """(Cargo workspace version, desktop/mygo.json version)."""
    cargo = workspace_version((root / "Cargo.toml").read_text(encoding="utf-8"))
    app = json.loads((root / "desktop/mygo.json").read_text(encoding="utf-8"))["version"]
    return cargo, app


def app_version(root):
    cargo, app = versions(root)
    if cargo != app:
        raise SystemExit(f"desktop/mygo.json says {app}, Cargo.toml {cargo}; "
                         "run release_build.py prepare-version or fix desktop/mygo.json")
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
        environment = dict(os.environ, HOME=work, USERPROFILE=work, MARKITAI_HOME=str(Path(work) / "state"),
                           MARKITAI_LANG="en")
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
                cwd=root / "desktop", environment=go_environment(platform), capture=True)
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
        "# Go licences\n\nThe Markitai desktop app is a Go program. It links the Go standard library and "
        "these modules:\n\n" + "\n".join(rows) + "\n", encoding="utf-8")


def stage(archive, sums=None, *, root=ROOT, modules=None, goroot=None, host_system=None):
    """Replace desktop/resources with the archive's program and the app's licences. The new
    resources are made apart and replace the old ones only once complete."""
    version = app_version(root)
    platform = archive_platform(archive, version)
    if sums is not None:
        check_sum(archive, sums)
    resources = root / "desktop/resources"
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
        raise SystemExit(f"Desktop artifacts must not be named like CLI archives ({CLI_ASSET_PREFIX}...): "
                         + ", ".join(taken))


def mygo_build_command(platform, e2e, goflags=""):
    """`go tool mygo build` arguments and the extra environment; the e2e build keeps the
    inspector the automation plugin needs, goes to build/e2e and adds its tag to `goflags`:
    mygo build takes tags from GOFLAGS only."""
    arguments = ["tool", "mygo", "build", "-platform", platform]
    if e2e:
        return arguments + ["-debug", "-skip-dmg", "-o", "build/e2e"], {"GOFLAGS": f"{goflags} -tags=e2e".strip()}
    return arguments, {}


def build(platform, e2e=False, *, root=ROOT):
    if platform not in SHIPPED:
        raise SystemExit(f"This round builds {', '.join(sorted(SHIPPED))}, not {platform}")
    version = app_version(root)
    resources = root / "desktop/resources"
    stamp_path = resources / STAMP
    stamp = json.loads(stamp_path.read_text(encoding="utf-8")) if stamp_path.exists() else None
    if stamp is None or (stamp.get("version"), stamp.get("platform")) != (version, platform):
        held = "nothing staged" if stamp is None else f"{stamp.get('platform')} {stamp.get('version')}"
        raise SystemExit(f"desktop/resources holds {held}, not {platform} {version}; run: desktop_build.py "
                         f"stage --archive {archive_name(version, PLATFORMS[platform])}")
    arguments, extra = mygo_build_command(platform, e2e, os.environ.get("GOFLAGS", ""))
    out = root / "desktop" / ("build/e2e" if e2e else "build") / resources_dir(platform)
    if out.exists():
        shutil.rmtree(out)
    go(arguments, cwd=root / "desktop", environment=dict(os.environ, **extra))
    names = sorted(path.name for path in out.iterdir())
    check_artifacts(names)
    goarch = platform.split("/")[1]
    expected = {"Markitai.app"} | (set() if e2e else {f"markitai-desktop-{version}-{goarch}.dmg"})
    if not expected <= set(names):
        raise SystemExit(f"{out} has {names}; expected {sorted(expected)}")
    bundled = out / "Markitai.app/Contents/Resources/bin" / program_name(platform)
    if not os.access(bundled, os.X_OK):
        raise SystemExit(f"{bundled} is missing or not executable")
    for name in names:
        print(out / name)
    return out


def vet(root=ROOT):
    desktop = root / "desktop"
    gofmt = go_root() / "bin" / ("gofmt.exe" if os.name == "nt" else "gofmt")
    unformatted = subprocess.run([str(gofmt), "-l", "."], cwd=desktop, capture_output=True, text=True,
                                 check=True).stdout.split()
    if unformatted:
        raise SystemExit("gofmt would change: " + ", ".join(unformatted))
    native = host_platform()
    for system in ["darwin", "linux", "windows"]:
        platform = native if native.startswith(system + "/") else f"{system}/amd64"
        environment = go_environment(platform)
        go(["vet", "./..."], cwd=desktop, environment=environment)
        if platform == native:
            go(["vet", "-tags", "e2e", "./..."], cwd=desktop, environment=environment)
            go(["test", "-count=1", "./..."], cwd=desktop, environment=environment)
        else:
            with tempfile.TemporaryDirectory() as work:
                go(["test", "-c", "-o", str(Path(work) / "desktop.test"), "."], cwd=desktop,
                   environment=environment)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    staging = commands.add_parser("stage", help="put a release archive's markitai and the licences in desktop/resources")
    staging.add_argument("--archive", type=Path, required=True)
    staging.add_argument("--sums", type=Path, help="the release's SHA256SUMS")
    building = commands.add_parser("build", help="build the app from the staged resources")
    building.add_argument("--platform", required=True, choices=sorted(SHIPPED))
    building.add_argument("--e2e", action="store_true", help="the -debug build scripts/desktop_e2e.py drives")
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
