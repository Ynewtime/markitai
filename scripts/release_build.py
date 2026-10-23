"""Build, check and describe the artifacts of a markitai release.

`.github/workflows/release.yml` runs every subcommand; each also runs locally:

  version   check the tag, Cargo, Python and server.json versions agree
  notes     the tag's section of CHANGELOG.md, the GitHub Release notes
  cli       build the CLI for one target and write its release archive
  wheel     build the Python wheel for one target (licenses supplemented)
  sdist     write the source distribution, with the vendored crates it needs
  smoke     extract a release archive and exercise the installed CLI
  smoke-wheel  install a wheel into a new venv and exercise it
  smoke-sdist  build a wheel from the sdist with pip and exercise it
  sums      write SHA256SUMS for the release assets

Archives are `markitai-v<version>-<target>.tar.gz` (`.zip` on Windows) with
the executables at the root: `markitai` and relative `mkai`/`markitai-mcp`
links on Unix, three EXEs on Windows, beside the license notices and offline
guides `ci_packages.py` defines. The setup scripts (`scripts/install/`) read
these names. Nothing here signs or publishes.
"""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from ci_packages import (apply_path_remap, cli_attribution, doctor_probe, identity, mcp_probe,
                         package_attribution, pinned_maturin, supplement_wheel_licenses,
                         windows_cli_build, write_cli_zip, write_single_binary_tar)
from cli_documentation import cli_documentation
from executable_identity import WINDOWS_TARGETS, verify_target_executable, verify_windows_cli_runtime
from package_cli_target import macho_minimum, workspace_version

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "aarch64-apple-darwin": "macosx_11_0_arm64",
    "x86_64-apple-darwin": "macosx_11_0_x86_64",
    "x86_64-unknown-linux-gnu": "manylinux_2_28_x86_64",
    "aarch64-unknown-linux-gnu": "manylinux_2_28_aarch64",
    "x86_64-pc-windows-msvc": "win_amd64",
    "aarch64-pc-windows-msvc": "win_arm64",
}
# The oldest glibc the Linux builds (in the manylinux_2_28 image) may require.
GLIBC_FLOOR = (2, 28)
MACOS_MINIMUM = "11.0"
SERVER_NAME = "io.github.Ynewtime/markitai"
# What maturin's sdist leaves out but a build reads: the vendored crates the
# workspace patches in (Cargo.toml [patch.crates-io]; maturin follows path
# dependencies, not patches) and the files crates embed from outside their
# own directories (the workbench's vendored web assets, pricing provenance).
SDIST_EXTRA = ["vendor/anydoc", "vendor/htmd", "vendor/iwork", "vendor/lopdf", "vendor/pdf-inspector",
               "vendor/web", "licenses"]


def python_version(cargo):
    """The PEP 440 form of a Cargo version: 2.0.0-dev -> 2.0.0.dev0, 2.0.0-rc.1 -> 2.0.0rc1."""
    match = re.fullmatch(r"(\d+\.\d+\.\d+)(?:-(dev|alpha|beta|rc)(?:\.(\d+))?)?", cargo)
    if not match:
        raise ValueError(f"Not a release version: {cargo}")
    base, label, number = match.groups()
    if not label:
        return base
    suffix = {"dev": ".dev", "alpha": "a", "beta": "b", "rc": "rc"}[label]
    return f"{base}{suffix}{number or 0}"


def pyproject_version(text):
    section = None
    for line in text.splitlines():
        heading = re.match(r"\s*\[([^\]]+)\]\s*$", line)
        if heading:
            section = heading.group(1).strip()
            continue
        value = re.match(r'\s*version\s*=\s*"([^"]+)"', line)
        if section == "project" and value:
            return value.group(1)
    raise RuntimeError("bindings/python/pyproject.toml has no [project] version")


def check_versions(root, tag=None):
    """The versions every artifact carries, or an error naming the mismatch.

    server.json may lag on a development version (its committed entry names
    the next release); a tag must match it as well as Cargo and Python.
    """
    cargo = workspace_version((root / "Cargo.toml").read_text(encoding="utf-8"))
    python = pyproject_version((root / "bindings/python/pyproject.toml").read_text(encoding="utf-8"))
    errors = []
    if python != python_version(cargo):
        errors.append(f"bindings/python/pyproject.toml says {python}, Cargo.toml {cargo} "
                      f"(expected {python_version(cargo)})")
    server = json.loads((root / "server.json").read_text(encoding="utf-8"))
    errors.extend(server_errors(server, python_version(cargo) if tag else None))
    if tag is not None and tag != f"v{cargo}":
        errors.append(f"Tag {tag} does not name the Cargo version {cargo} (expected v{cargo})")
    if errors:
        raise SystemExit("\n".join(errors))
    return {"version": cargo, "python_version": python, "prerelease": "-" in cargo,
            "server_version": server.get("version")}


def server_errors(server, version=None):
    """What the MCP Registry entry gets wrong for this repository's PyPI package."""
    errors = []
    if server.get("name") != SERVER_NAME:
        errors.append(f"server.json name is not {SERVER_NAME}")
    packages = server.get("packages")
    if not isinstance(packages, list) or len(packages) != 1:
        return errors + ["server.json must list exactly one package"]
    package = packages[0]
    if (package.get("registryType"), package.get("identifier")) != ("pypi", "markitai"):
        errors.append("server.json must point at the PyPI package markitai")
    if package.get("version") != server.get("version"):
        errors.append("server.json version and package version differ")
    if version is not None and server.get("version") != version:
        errors.append(f"server.json says {server.get('version')}, the release is {version}")
    readme = ROOT / "bindings/python/README.md"
    if f"mcp-name: {SERVER_NAME}" not in readme.read_text(encoding="utf-8"):
        errors.append("bindings/python/README.md lacks the MCP Registry's mcp-name marker")
    return errors


def release_notes(changelog, version):
    """The body of `## [version]` up to the next version heading."""
    lines = changelog.splitlines()
    heading = re.compile(r"^## \[" + re.escape(version) + r"\](?:\s|$)")
    for start, line in enumerate(lines):
        if heading.match(line):
            end = next((index for index in range(start + 1, len(lines)) if lines[index].startswith("## [")),
                       len(lines))
            body = "\n".join(lines[start + 1:end]).strip()
            if body:
                return body + "\n"
            break
    raise SystemExit(f"CHANGELOG.md has no section for {version}")


def archive_name(version, target):
    return f"markitai-v{version}-{target}" + (".zip" if target in WINDOWS_TARGETS else ".tar.gz")


def glibc_requirement(path):
    """The newest GLIBC_x.y symbol version an ELF executable names."""
    found = {tuple(int(part) for part in match.split(b"."))
             for match in re.findall(rb"GLIBC_(\d+\.\d+(?:\.\d+)?)\x00", path.read_bytes())}
    return max(found) if found else None


def isolated_environment(work):
    for name in ["state", "home", "tmp"]:
        (work / name).mkdir(exist_ok=True)
    environment = dict(os.environ, MARKITAI_HOME=str(work / "state"), HOME=str(work / "home"),
                       USERPROFILE=str(work / "home"), MARKITAI_LANG="en", TMPDIR=str(work / "tmp"),
                       TMP=str(work / "tmp"), TEMP=str(work / "tmp"))
    for key in ["PYTHONPATH", "PYTHONHOME", "CARGO_BUILD_TARGET", "MARKITAI_CONFIG"]:
        environment.pop(key, None)
    return environment


def build_environment():
    """The caller's toolchain with build paths remapped out of the binaries."""
    environment = dict(os.environ)
    environment.setdefault("CARGO_HOME", str(Path.home() / ".cargo"))
    environment.setdefault("RUSTUP_HOME", str(Path.home() / ".rustup"))
    environment.pop("CARGO_BUILD_TARGET", None)
    # The build time the binaries report is the commit's, as for a reproduced build.
    if "SOURCE_DATE_EPOCH" not in environment:
        environment["SOURCE_DATE_EPOCH"] = capture(["git", "log", "-1", "--format=%ct"], cwd=ROOT).strip()
    apply_path_remap(environment, ROOT)
    return environment


def run(command, **options):
    command = [str(part) for part in command]
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, **options)


def capture(command, **options):
    """A command's standard output; a failure shows what it printed."""
    result = subprocess.run([str(part) for part in command], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=300, **options)
    if result.returncode:
        raise SystemExit(f"{command[0]} exited {result.returncode}\n--- stdout\n{result.stdout}"
                         f"\n--- stderr\n{result.stderr}")
    return result.stdout


def host_target():
    return next(line.split(": ", 1)[1] for line in capture(["rustc", "-vV"]).splitlines()
                if line.startswith("host: "))


def build_cli(target, output):
    if target not in TARGETS:
        raise SystemExit(f"Not a release target: {target}")
    if host_target() != target:
        raise SystemExit(f"Release archives are built natively; this host is {host_target()}, not {target}")
    output.mkdir(parents=True, exist_ok=True)
    environment = build_environment()
    windows = target in WINDOWS_TARGETS
    if windows:
        command, environment, release = windows_cli_build(ROOT, target, environment)
    else:
        if target.endswith("-apple-darwin"):
            environment.setdefault("MACOSX_DEPLOYMENT_TARGET", MACOS_MINIMUM)
        command = ["cargo", "build", "--release", "--locked", "-p", "markitai-cli", "--bin", "markitai",
                   "--target", target]
        base = Path(environment.get("CARGO_TARGET_DIR", ROOT / "target"))
        release = (base if base.is_absolute() else ROOT / base) / target / "release"
    run(command, cwd=ROOT, env=environment)
    binary = release / ("markitai.exe" if windows else "markitai")
    info = verify_target_executable(binary, target)
    if windows:
        verify_windows_cli_runtime(binary, target)
    elif target.endswith("-apple-darwin"):
        minimum = macho_minimum(binary)
        if minimum != environment["MACOSX_DEPLOYMENT_TARGET"]:
            raise SystemExit(f"The executable records macOS {minimum}, not {environment['MACOSX_DEPLOYMENT_TARGET']}")
        info["minimum_macos"] = minimum
    else:
        glibc = glibc_requirement(binary)
        if glibc is None or glibc > GLIBC_FLOOR:
            raise SystemExit(f"The executable needs glibc {glibc}; release builds stay at or below "
                             f"{'.'.join(map(str, GLIBC_FLOOR))} (build in the manylinux_2_28 image)")
        info["glibc"] = ".".join(map(str, glibc))
    version = workspace_version((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    archive = output / archive_name(version, target)
    if archive.exists():
        archive.unlink()
    licenses = cli_attribution(ROOT, package_attribution(ROOT))
    documentation = cli_documentation(ROOT)
    if windows:
        # One optimized executable under all three names; argv[0] selects MCP mode.
        write_cli_zip(binary, binary, archive, licenses, True, documentation)
    else:
        write_single_binary_tar(binary, archive, licenses, documentation)
    print(json.dumps({"archive": archive.name, **identity(archive), "executable": {**identity(binary), **info}},
                     indent=2), flush=True)
    return archive


def _safe_member(name):
    path = PurePosixPath(name)
    return (name and not path.is_absolute() and path.as_posix() == name.rstrip("/")
            and not any(part in {"", ".", ".."} for part in path.parts) and "\\" not in name)


def extract_release_archive(archive, destination):
    """Check a release archive's shape the way a user receives it, then extract it.

    Only the CLI names are executables; the only links are the Unix aliases.
    """
    destination.mkdir(parents=True)
    if archive.name.endswith(".zip"):
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            if len(names) != len(set(names)) or not all(_safe_member(name) for name in names):
                raise SystemExit("The ZIP has duplicate or unsafe member names")
            for required in ["markitai.exe", "mkai.exe", "markitai-mcp.exe", "LICENSE", "NOTICE"]:
                if required not in names:
                    raise SystemExit(f"The ZIP lacks {required} at its root")
            for item in bundle.infolist():
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise SystemExit("A Windows ZIP carries no links")
            bundle.extractall(destination)
        return {"members": len(names)}
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or not all(_safe_member(name) for name in names):
            raise SystemExit("The archive has duplicate or unsafe member names")
        for member in members:
            if member.issym():
                if member.name not in {"mkai", "markitai-mcp"} or member.linkname != "markitai":
                    raise SystemExit(f"Unexpected link {member.name} -> {member.linkname}")
            elif not member.isfile():
                raise SystemExit(f"Unexpected member type: {member.name}")
            elif member.mode & 0o111 and member.name != "markitai":
                raise SystemExit(f"Only markitai is executable, not {member.name}")
        for required in ["markitai", "mkai", "markitai-mcp", "LICENSE", "NOTICE"]:
            if required not in names:
                raise SystemExit(f"The archive lacks {required} at its root")
        if bundle.getmember("markitai").mode & 0o777 != 0o755:
            raise SystemExit("markitai is not mode 0755")
        bundle.extractall(destination, filter="tar")
    return {"members": len(names)}


def smoke_commands(markitai, mkai, mcp, version, work, environment, windows):
    """The installed-package probes ci_packages runs, on whichever entry points the package installs."""
    results = {}
    for name, command in [("markitai", markitai), ("mkai", mkai)]:
        reported = capture([command, "--version"], cwd=work, env=environment).strip()
        if reported.split()[-1:] != [version]:
            raise SystemExit(f"{name} --version reported {reported!r}, not {version}")
        results[name] = reported
    help_text = capture([mcp, "--help"], cwd=work, env=environment)
    if "mcp" not in help_text.lower() or "Commands:" in help_text:
        raise SystemExit("markitai-mcp did not select the MCP command")
    source = work / "source.md"
    source.write_text("# Release package\n\nHello 世界.\n", encoding="utf-8")
    converted = subprocess.run([str(markitai), str(source), "--pure"], cwd=work, env=environment,
                               capture_output=True, timeout=300).stdout.decode("utf-8")
    if converted.replace("\r\n", "\n") != source.read_text(encoding="utf-8"):
        raise SystemExit("markitai changed plain Unicode Markdown")
    results["mcp"] = mcp_probe(mcp, work, environment, work / "mcp.log")["tools"]
    doctor = doctor_probe(markitai, work, environment, work / "doctor.log")
    results["doctor"] = {"exit_code": doctor["exit_code"],
                         "checks": {name: check.get("status") for name, check in doctor["checks"].items()}}
    return results


def smoke_archive(archive, target):
    version = workspace_version((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    if archive.name != archive_name(version, target):
        raise SystemExit(f"{archive.name} is not {archive_name(version, target)}")
    windows = target in WINDOWS_TARGETS
    with tempfile.TemporaryDirectory(prefix="markitai-smoke-") as directory:
        work = Path(directory)
        install = work / "安装 with spaces"
        shape = extract_release_archive(archive, install)
        extension = ".exe" if windows else ""
        names = [install / (name + extension) for name in ["markitai", "mkai", "markitai-mcp"]]
        verify_target_executable(names[0], target)
        if windows:
            for path in names:
                verify_windows_cli_runtime(path, target)
        results = smoke_commands(*names, version, work, isolated_environment(work), windows)
    print(json.dumps({"archive": archive.name, **identity(archive), **shape, "probes": results}, indent=2,
                     ensure_ascii=False))


def build_wheel(target, output):
    if target not in TARGETS:
        raise SystemExit(f"Not a release target: {target}")
    pinned = pinned_maturin((ROOT / "bindings/python/pyproject.toml").read_text(encoding="utf-8"))
    found = capture(["maturin", "--version"]).split()
    if found[-1:] != [pinned]:
        raise SystemExit(f"Release wheels are built with maturin {pinned}, not {' '.join(found)}")
    output.mkdir(parents=True, exist_ok=True)
    environment = build_environment()
    command = ["maturin", "build", "--release", "--locked", "--target", target,
               "--interpreter", sys.executable]
    if target.endswith("-apple-darwin"):
        environment.setdefault("MACOSX_DEPLOYMENT_TARGET", MACOS_MINIMUM)
    if "-linux-" in target:
        command += ["--compatibility", "manylinux_2_28"]
    with tempfile.TemporaryDirectory(prefix="markitai-wheel-") as directory:
        run([*command, "--out", directory], cwd=ROOT / "bindings/python", env=environment)
        built = list(Path(directory).glob("*.whl"))
        if len(built) != 1:
            raise SystemExit(f"maturin wrote {len(built)} wheels")
        expected = f"-cp310-abi3-{TARGETS[target]}.whl"
        if not built[0].name.endswith(expected):
            raise SystemExit(f"{built[0].name} is not a {expected[1:]} wheel")
        wheel = output / built[0].name
        if wheel.exists():
            wheel.unlink()
        supplement_wheel_licenses(built[0], wheel, package_attribution(ROOT), ROOT)
    print(json.dumps({"wheel": wheel.name, **identity(wheel)}, indent=2), flush=True)
    return wheel


def smoke_wheel(wheel, python, run_tests):
    version = workspace_version((ROOT / "Cargo.toml").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(prefix="markitai-wheel-smoke-") as directory:
        work = Path(directory)
        environment = isolated_environment(work)
        venv = work / "venv"
        run([python, "-m", "venv", venv], env=environment)
        windows = os.name == "nt"
        scripts = venv / ("Scripts" if windows else "bin")
        interpreter = scripts / ("python.exe" if windows else "python")
        run([interpreter, "-m", "pip", "install", "--no-index", "--no-deps", "--disable-pip-version-check", wheel],
            env=environment)
        extension = ".exe" if windows else ""
        commands = [scripts / (name + extension) for name in ["markitai", "mkai", "markitai-mcp"]]
        results = smoke_commands(*commands, version, work, environment, windows)
        probe = ("import markitai, sys; result = markitai.convert(sys.argv[1]);"
                 " print(markitai.__version__); print(result.markdown, end='')")
        source = work / "api.md"
        source.write_text("# Python API\n\nHello 世界.\n", encoding="utf-8")
        # Python writes a pipe in the locale's encoding (cp1252 on Windows),
        # which has no 世界; the probe reads its answer as UTF-8.
        lines = capture([interpreter, "-c", probe, source], cwd=work,
                        env={**environment, "PYTHONIOENCODING": "utf-8"}).splitlines()
        if lines[0] != version or "Hello 世界." not in "\n".join(lines[1:]):
            raise SystemExit(f"The Python API answered {lines!r}")
        results["python_api"] = lines[0]
        if run_tests:
            environment["MARKITAI_TEST_NUMBERS_FIXTURES"] = str(
                ROOT / "crates/markitai-core/src/formats/numbers/fixtures")
            run([interpreter, "-m", "unittest", "discover", "-s", ROOT / "bindings/python/tests", "-v"],
                cwd=work, env=environment)
            results["python_tests"] = "passed"
    print(json.dumps({"wheel": wheel.name, **identity(wheel), "probes": results}, indent=2, ensure_ascii=False))


def smoke_sdist(sdist, python):
    """pip builds a wheel from the sdist alone (Rust needed), then the wheel's probes run."""
    with tempfile.TemporaryDirectory(prefix="markitai-sdist-smoke-") as directory:
        work = Path(directory)
        environment = build_environment()
        # Built outside the checkout, so only what the sdist carries is used.
        environment["CARGO_TARGET_DIR"] = str(work / "target")
        run([python, "-m", "pip", "wheel", "--no-deps", "--no-binary", "markitai", "--wheel-dir", work / "wheels",
             sdist], cwd=work, env=environment)
        built = list((work / "wheels").glob("markitai-*.whl"))
        if len(built) != 1:
            raise SystemExit(f"pip built {len(built)} wheels from the sdist")
        smoke_wheel(built[0], python, run_tests=False)


def tracked_files(directory):
    names = subprocess.check_output(["git", "ls-files", "-z", "--", directory], cwd=ROOT)
    return [os.fsdecode(name) for name in names.split(b"\0") if name]


def build_sdist(output):
    """maturin's sdist plus SDIST_EXTRA and the license notices.

    maturin 1.15 copies each crate's own files and its path dependencies,
    so its sdist alone cannot build.
    """
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="markitai-sdist-") as directory:
        run(["maturin", "sdist", "--out", directory], cwd=ROOT / "bindings/python")
        built = list(Path(directory).glob("*.tar.gz"))
        if len(built) != 1:
            raise SystemExit(f"maturin wrote {len(built)} sdists")
        sdist = output / built[0].name
        if sdist.exists():
            sdist.unlink()
        prefix = built[0].name[:-len(".tar.gz")]
        extra = {relative: (ROOT / relative).read_bytes()
                 for directory in SDIST_EXTRA for relative in tracked_files(directory)}
        for name, content in package_attribution(ROOT).items():
            extra.setdefault(name, content)
        with tarfile.open(built[0], "r:gz") as original, tarfile.open(sdist, "x:gz", format=tarfile.PAX_FORMAT) as result:
            present = set()
            for member in original.getmembers():
                present.add(member.name)
                result.addfile(member, original.extractfile(member) if member.isfile() else None)
            for relative, content in sorted(extra.items()):
                name = f"{prefix}/{relative}"
                if name in present:
                    continue
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                result.addfile(info, io.BytesIO(content))
    print(json.dumps({"sdist": sdist.name, **identity(sdist), "added": len(extra)}, indent=2), flush=True)
    return sdist


def expected_assets(version):
    """Every file a release carries besides SHA256SUMS: archives, wheels, the sdist."""
    python = python_version(version)
    names = {archive_name(version, target) for target in TARGETS}
    names |= {f"markitai-{python}-cp310-abi3-{tag}.whl" for tag in TARGETS.values()}
    names.add(f"markitai-{python}.tar.gz")
    return names


def write_sums(directory, version=None):
    """`sha256sum -c`-compatible lines for every file but SHA256SUMS itself;
    with `version`, exactly the assets that release carries."""
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    if not files:
        raise SystemExit(f"No release assets in {directory}")
    if version is not None:
        found = {path.name for path in files}
        expected = expected_assets(version)
        if found != expected:
            raise SystemExit(f"Release assets differ: missing {sorted(expected - found)}, "
                             f"unexpected {sorted(found - expected)}")
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in files]
    (directory / "SHA256SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")
    print("".join(lines), end="")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    version = commands.add_parser("version")
    version.add_argument("--tag")
    version.add_argument("--github-output", type=Path)
    notes = commands.add_parser("notes")
    notes.add_argument("--version", required=True)
    notes.add_argument("--output", type=Path, required=True)
    for name in ["cli", "wheel"]:
        command = commands.add_parser(name)
        command.add_argument("--target", required=True)
        command.add_argument("--output", type=Path, required=True)
    sdist = commands.add_parser("sdist")
    sdist.add_argument("--output", type=Path, required=True)
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--target", required=True)
    smoke.add_argument("archive", type=Path)
    wheel_smoke = commands.add_parser("smoke-wheel")
    wheel_smoke.add_argument("--python", default=sys.executable)
    wheel_smoke.add_argument("--no-tests", action="store_true", help="skip bindings/python/tests")
    wheel_smoke.add_argument("wheel", type=Path)
    sdist_smoke = commands.add_parser("smoke-sdist")
    sdist_smoke.add_argument("--python", default=sys.executable)
    sdist_smoke.add_argument("sdist", type=Path)
    sums = commands.add_parser("sums")
    sums.add_argument("--version", help="require exactly this release's assets")
    sums.add_argument("directory", type=Path)
    args = parser.parse_args(argv)

    if args.command == "version":
        result = check_versions(ROOT, args.tag)
        print(json.dumps(result, indent=2))
        if args.github_output:
            with args.github_output.open("a", encoding="utf-8") as stream:
                for key, value in result.items():
                    stream.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")
    elif args.command == "notes":
        args.output.write_text(release_notes((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), args.version),
                               encoding="utf-8")
    elif args.command == "cli":
        build_cli(args.target, args.output.resolve())
    elif args.command == "wheel":
        build_wheel(args.target, args.output.resolve())
    elif args.command == "sdist":
        build_sdist(args.output.resolve())
    elif args.command == "smoke":
        smoke_archive(args.archive.resolve(), args.target)
    elif args.command == "smoke-wheel":
        smoke_wheel(args.wheel.resolve(), args.python, not args.no_tests)
    elif args.command == "smoke-sdist":
        smoke_sdist(args.sdist.resolve(), args.python)
    elif args.command == "sums":
        write_sums(args.directory, args.version)
    return 0


if __name__ == "__main__":
    sys.exit(main())
