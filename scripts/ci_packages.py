"""Build native artifacts and test installed Python/Node packages on one host.

The manifest records executed steps and omissions. It is not a release signature,
a portability promise, or proof that another matrix runner succeeded.
"""
from pathlib import Path
import argparse
import base64
import csv
import datetime
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

from pricing_attribution import pricing_files
from codex_attribution import codex_files
from license_overlay import upstream_files


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def source_snapshot(root, paths):
    """Hash source bytes and symlink text, including already-dirty files."""
    files = {}
    for relative in sorted(set(paths)):
        path = root / relative
        if path.is_symlink():
            value = os.fsencode(os.readlink(path))
            if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
                raise RuntimeError(f"Source symlink is not an in-repository file: {relative}")
            files[relative] = {"kind": "symlink", "sha256": hashlib.sha256(value).hexdigest(),
                               "target": identity(path)}
        elif path.is_file():
            files[relative] = {"kind": "file", **identity(path),
                               "executable": bool(path.stat().st_mode & 0o111)}
        else:
            raise RuntimeError(f"Source file is missing or not a regular file: {relative}")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files}


def npm_command(npm, node, host_platform=sys.platform):
    """Avoid cmd.exe quoting and CreateProcess failures for npm.cmd on Windows."""
    if host_platform != "win32" or Path(npm).suffix.lower() not in {".cmd", ".bat"}:
        return [str(npm)]
    if not node:
        raise RuntimeError("node is required to invoke the Windows npm CLI")
    for directory in dict.fromkeys([Path(npm).parent, Path(node).parent]):
        cli = directory / "node_modules" / "npm" / "bin" / "npm-cli.js"
        if cli.is_file():
            return [str(node), str(cli)]
    raise RuntimeError("Cannot locate npm-cli.js beside npm.cmd or node; install a standard Node/npm distribution")


def verify_node_licenses(package, licenses):
    with tarfile.open(package, "r:gz") as bundle:
        for name, expected in licenses.items():
            members = [entry for entry in bundle.getmembers() if entry.name == f"package/{name}"]
            if len(members) != 1 or not members[0].isfile():
                raise RuntimeError(f"Node package is missing regular {name}")
            if bundle.extractfile(members[0]).read() != expected:
                raise RuntimeError(f"Node package {name} differs from the source")


def stage_node_licenses(directory, licenses):
    """Preserve original evidence names despite npm's nested *.orig default."""
    overrides = {}
    for name, content in licenses.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        if target.suffix == ".orig":
            overrides.setdefault(target.parent, []).append("!" + target.name)
    for parent, entries in overrides.items():
        (parent / ".npmignore").write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")
    # npm propagates default exclusions through explicit nested file entries.
    # Listing their private staging directories lets the leaf override apply.
    return sorted(parent.relative_to(directory).as_posix() for parent in overrides)


def supplement_wheel_licenses(source, destination, licenses):
    """Keep the original wheel and record a separate, explicitly supplemented wheel.

    Native and Python bytes remain unchanged. This adds files to dist-info and
    rebuilds RECORD; it does not claim the project declares PEP 639 license-files.
    """
    with zipfile.ZipFile(source) as original:
        names = original.namelist()
        if len(names) != len(set(names)):
            raise RuntimeError("Wheel contains duplicate paths")
        records = [name for name in names if name.endswith(".dist-info/RECORD")]
        if len(records) != 1:
            raise RuntimeError("Wheel must contain exactly one RECORD")
        record = records[0]
        if record + ".jws" in names or record + ".p7s" in names:
            raise RuntimeError("Refusing to change a signed wheel")
        directory = record.rsplit("/", 1)[0]
        additions = {f"{directory}/licenses/{name}": value for name, value in licenses.items()}
        rows = []
        with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as result:
            for info in original.infolist():
                if info.filename == record:
                    continue
                if info.filename in additions:
                    if original.read(info.filename) != additions.pop(info.filename):
                        raise RuntimeError(f"Wheel already has conflicting {info.filename}")
                digest = hashlib.sha256()
                size = 0
                with original.open(info) as incoming, result.open(info, "w") as outgoing:
                    for block in iter(lambda: incoming.read(1024 * 1024), b""):
                        outgoing.write(block)
                        digest.update(block)
                        size += len(block)
                encoded = base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")
                rows.append([info.filename, f"sha256={encoded}", str(size)])
            for name, value in additions.items():
                result.writestr(name, value)
                encoded = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b"=").decode("ascii")
                rows.append([name, f"sha256={encoded}", str(len(value))])
            rows.append([record, "", ""])
            buffer = io.StringIO(newline="")
            csv.writer(buffer, lineterminator="\n").writerows(rows)
            result.writestr(record, buffer.getvalue().encode("utf-8"))
    return {"operation": "add_dist_info_licenses_and_rebuild_RECORD", "original": identity(source),
            "supplemented": identity(destination), "license_directory": directory + "/licenses"}


def package_attribution(root):
    """Attribution every package carries: the project's license and notice,
    pricing data, Codex and vendored upstream licenses."""
    licenses = {name: (root / name).read_bytes() for name in ["LICENSE", "NOTICE"]}
    licenses.update(pricing_files(root))
    licenses.update(codex_files(root))
    licenses.update(upstream_files(root))
    return licenses


def cli_attribution(root, licenses):
    """The CLI archives' attribution: the package attribution and the
    licenses of the web libraries the CLI embeds."""
    cli_licenses = dict(licenses)
    for name in [
        "marked-LICENSE",
        "DOMPurify-LICENSE",
        "preact-LICENSE",
        "phosphor-LICENSE",
        "Inter-OFL.txt",
        "provenance.json",
    ]:
        relative = "vendor/web/" + name
        cli_licenses[relative] = (root / relative).read_bytes()
    return cli_licenses


def _cli_license_paths(licenses):
    from pathlib import PurePosixPath
    for name in licenses:
        path = PurePosixPath(name)
        if (not name or path.is_absolute() or path.as_posix() != name
                or any(part in {"", ".", ".."} for part in path.parts)
                or "\\" in name or path.parts[0] in {"markitai", "mkai", "markitai-mcp"}):
            raise RuntimeError("Unsafe CLI attribution path")


def write_single_binary_tar(binary, destination, licenses):
    """Unix delivery: one regular executable, relative aliases, exact attribution."""
    _cli_license_paths(licenses)
    if binary.is_symlink() or not binary.is_file():
        raise RuntimeError("CLI delivery requires a regular executable")
    with tarfile.open(destination, "x:gz") as archive:
        member = tarfile.TarInfo("markitai")
        member.size = binary.stat().st_size
        member.mode = 0o755
        with binary.open("rb") as stream:
            archive.addfile(member, stream)
        for name in ["mkai", "markitai-mcp"]:
            alias = tarfile.TarInfo(name)
            alias.type = tarfile.SYMTYPE
            alias.linkname = "markitai"
            alias.mode = 0o777
            archive.addfile(alias)
        for name, content in licenses.items():
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def extract_single_binary_tar(archive_path, destination, binary, licenses):
    """Validate the complete inventory before extracting our narrowly shaped tar."""
    _cli_license_paths(licenses)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        expected = {"markitai", "mkai", "markitai-mcp", *licenses}
        if len(members) != len(expected) or {m.name for m in members} != expected:
            raise RuntimeError("Single-binary CLI archive has an unexpected member inventory")
        by_name = {member.name: member for member in members}
        executable = by_name["markitai"]
        if (not executable.isfile() or executable.size != binary.stat().st_size
                or executable.mode != 0o755):
            raise RuntimeError("Single-binary CLI archive has an invalid executable")
        for name in ["mkai", "markitai-mcp"]:
            alias = by_name[name]
            if not alias.issym() or alias.linkname != "markitai":
                raise RuntimeError("CLI aliases must be relative symlinks to markitai")
        for name, content in licenses.items():
            member = by_name[name]
            if (not member.isfile() or member.mode != 0o644 or member.size != len(content)
                    or archive.extractfile(member).read() != content):
                raise RuntimeError("CLI attribution is missing or differs from source")
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / "markitai"
        with target.open("xb") as stream:
            shutil.copyfileobj(archive.extractfile(executable), stream)
        target.chmod(0o755)
        if identity(target) != identity(binary):
            raise RuntimeError("Archived CLI executable differs from the frozen binary")
        for name, content in licenses.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
            if target.read_bytes() != content:
                raise RuntimeError("Extracted CLI attribution differs from source")
        for name in ["mkai", "markitai-mcp"]:
            (destination / name).symlink_to("markitai")
            if os.readlink(destination / name) != "markitai" or identity(destination / name) != identity(binary):
                raise RuntimeError("Extracted CLI alias does not address the frozen executable")
    return {"archive": identity(archive_path), "executable": identity(destination / "markitai"),
            "aliases": [{"path": name, "target": "markitai", "kind": "symlink"}
                        for name in ["mkai", "markitai-mcp"]],
            "attribution": {name: identity(destination / name) for name in licenses}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-host", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    local = root / ".local"
    local.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="ci-packages-", dir=local))
    state = work / "state"
    state.mkdir()
    environment = dict(os.environ, MARKITAI_HOME=str(state), PYO3_PYTHON=sys.executable)
    for key in ["PYTHONPATH", "PYTHONHOME", "NODE_PATH", "NODE_OPTIONS"]:
        environment.pop(key, None)
    environment["MARKITAI_TEST_NUMBERS_FIXTURES"] = str(root / "crates/markitai-core/src/formats/numbers/fixtures")
    # This script tests native builds. A foreign Cargo target cannot be loaded
    # into the Python/Node processes running on this host.
    if environment.get("CARGO_BUILD_TARGET"):
        raise SystemExit("Unset CARGO_BUILD_TARGET for native package validation")
    target = Path(environment.get("CARGO_TARGET_DIR", root / "target"))
    if not target.is_absolute():
        target = root / target
    # The current Go source binding names this directory in its cgo directives.
    # Do not accidentally test a stale FFI from there after building elsewhere.
    if target.resolve() != (root / "target").resolve():
        raise SystemExit("Native package validation currently requires CARGO_TARGET_DIR=target for Go/cgo")
    release = target / "release"
    record = {
        "schema": 1,
        "status": "running",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "expected_host": args.expected_host,
        "python": sys.version,
        "work_directory": str(work),
        "steps": [],
        "artifacts": {},
        "limitations": ["No signing, publishing or other-host result is implied."],
    }

    def git(*arguments):
        return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()

    def snapshot():
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
        return source_snapshot(root, [os.fsdecode(name) for name in names.split(b"\0") if name])

    def run(name, command, cwd=root, env=None):
        command = [str(value) for value in command]
        log = output / "logs" / f"{len(record['steps']):02}-{name}.log"
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=cwd, env=env or environment,
                                    stdout=stream, stderr=subprocess.STDOUT)
        step = {"name": name, "command": command, "cwd": str(cwd),
                "exit_code": result.returncode, "log": str(log.relative_to(output)),
                "log_identity": identity(log)}
        record["steps"].append(step)
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")
        return log.read_text(encoding="utf-8", errors="replace")

    def artifact(path):
        record["artifacts"][str(path.relative_to(output))] = identity(path)

    try:
        record["source_revision"] = git("rev-parse", "HEAD")
        record["source_status_before"] = git("status", "--porcelain")
        if record["source_status_before"]:
            raise RuntimeError("Package validation requires a clean source checkout")
        record["source_before"] = snapshot()
        licenses = package_attribution(root)
        compiler = run("compiler", ["rustc", "-vV"])
        record["compiler"] = compiler
        host = next(line.split(": ", 1)[1] for line in compiler.splitlines()
                    if line.startswith("host: "))
        if host != args.expected_host:
            raise RuntimeError(f"Expected {args.expected_host}, rustc reports {host}")
        run("build", ["cargo", "build", "--workspace", "--release", "--locked"])
        record["source_after_workspace_build"] = snapshot()
        if record["source_before"] != record["source_after_workspace_build"]:
            raise RuntimeError("Source bytes changed during the workspace build")
        extension = ".exe" if sys.platform == "win32" else ""
        binary = release / ("markitai" + extension)
        version = run("version", [binary, "--version"]).strip().split()[-1]

        cli_licenses = cli_attribution(root, licenses)
        archive = output / f"markitai-{version}-{host}.zip"
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
            for name in ["markitai", "mkai"]:
                bundle.write(release / (name + extension), name + extension)
            for name, content in cli_licenses.items():
                bundle.writestr(name, content)
            if os.name == "nt":
                bundle.writestr("markitai-mcp.cmd", b'@echo off\r\n"%~dp0markitai.exe" mcp %*\r\n')
            else:
                alias = zipfile.ZipInfo("markitai-mcp")
                alias.create_system = 3
                alias.external_attr = (0o120777 << 16)
                bundle.writestr(alias, b"markitai")
        artifact(archive)
        extracted = work / "cli"
        with zipfile.ZipFile(archive) as bundle:
            if os.name == "nt":
                bundle.extractall(extracted)
                launcher = extracted / "markitai-mcp.cmd"
                if launcher.read_bytes() != b'@echo off\r\n"%~dp0markitai.exe" mcp %*\r\n':
                    raise RuntimeError("Windows MCP launcher differs from its declared forwarding command")
                record["cli_mcp_alias"] = {"kind": "cmd_forwarder", "identity": identity(launcher), "executed": False}
            else:
                alias = bundle.getinfo("markitai-mcp")
                if alias.external_attr >> 16 != 0o120777 or bundle.read(alias) != b"markitai":
                    raise RuntimeError("ZIP MCP alias is not the declared relative symlink")
                for member in bundle.infolist():
                    if member.filename != "markitai-mcp":
                        bundle.extract(member, extracted)
                for name in ["markitai", "mkai"]:
                    (extracted / name).chmod(0o755)
                (extracted / "markitai-mcp").symlink_to("markitai")
                if identity(extracted / "markitai-mcp") != identity(binary):
                    raise RuntimeError("ZIP MCP alias differs from the retained executable")
                record["cli_mcp_alias"] = {"kind": "symlink", "target": "markitai", "identity": identity(extracted / "markitai-mcp")}
                help_text = run("zip-mcp-alias-help", [extracted / "markitai-mcp", "--help"], cwd=extracted)
                if "mcp" not in help_text.lower() or "Commands:" in help_text:
                    raise RuntimeError("ZIP MCP alias did not select the MCP subcommand")
                record["cli_mcp_alias"]["executed"] = True
        for name, content in licenses.items():
            if (extracted / name).read_bytes() != content:
                raise RuntimeError(f"Archived CLI {name} differs from pricing/project attribution")
        if os.name != "nt":
            single = output / f"markitai-{version}-{host}-single-binary.tar.gz"
            write_single_binary_tar(binary, single, cli_licenses)
            artifact(single)
            unpacked = work / "single-binary-cli"
            record["single_binary_cli"] = extract_single_binary_tar(single, unpacked, binary, cli_licenses)
            alias_version = run("single-binary-alias-version", [unpacked / "mkai", "--version"], cwd=unpacked).strip()
            if alias_version.split()[-1] != version:
                raise RuntimeError("Archived mkai alias reported a different version")
            record["single_binary_cli"]["alias_version"] = alias_version
            help_text = run("single-binary-mcp-alias-help", [unpacked / "markitai-mcp", "--help"], cwd=unpacked)
            if "mcp" not in help_text.lower() or "Commands:" in help_text:
                raise RuntimeError("Single-binary MCP alias did not select the MCP subcommand")
            record["single_binary_cli"]["mcp_alias_executed"] = True
        source = work / "source.md"
        source.write_text("# Native package\n\nHello 世界.\n", encoding="utf-8")
        text = run("archived-cli", [extracted / ("markitai" + extension), source, "--pure"], cwd=work)
        if text != source.read_text(encoding="utf-8"):
            raise RuntimeError("Archived CLI changed plain Unicode content")
        record["doctor"] = json.loads(run("doctor", [binary, "doctor", "--json"], cwd=work))

        native = output / "native"
        native.mkdir()
        libraries = {
            "darwin": ("libmarkitai_ffi.dylib", "libmarkitai_node.dylib"),
            "linux": ("libmarkitai_ffi.so", "libmarkitai_node.so"),
            "win32": ("markitai_ffi.dll", "markitai_node.dll"),
        }[sys.platform]
        ffi = native / libraries[0]
        shutil.copy2(release / libraries[0], ffi)
        shutil.copy2(root / "bindings/c/markitai.h", native / "markitai.h")
        artifact(ffi)
        artifact(native / "markitai.h")
        for name, content in licenses.items():
            target = native / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            artifact(target)

        node_source = work / "node-package"
        node_source.mkdir()
        for name in ["index.cjs", "index.d.ts", "package.json"]:
            shutil.copy2(root / "bindings/node" / name, node_source / name)
        evidence_directories = stage_node_licenses(node_source, licenses)
        package_json = node_source / "package.json"
        package = json.loads(package_json.read_text(encoding="utf-8"))
        package["files"] = list(dict.fromkeys([*package.get("files", []), *licenses, *evidence_directories]))
        package_json.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
        record["node_staging"] = {"operation": "include_attribution_and_preserve_nested_orig_evidence", "package_json": identity(package_json)}
        shutil.copy2(release / libraries[1], node_source / "markitai.node")
        npm = shutil.which("npm")
        if not npm:
            raise RuntimeError("npm is required for native Node package validation")
        npm = npm_command(npm, shutil.which("node"))
        packed = json.loads(run("node-pack", [*npm, "pack", "--json", "--pack-destination", output], cwd=node_source))
        node_package = output / packed[0]["filename"]
        verify_node_licenses(node_package, licenses)
        artifact(node_package)
        consumer = work / "node-consumer"
        consumer.mkdir()
        run("node-install", [*npm, "install", "--ignore-scripts", "--no-audit", "--no-fund",
                             "--package-lock=false", node_package], cwd=consumer)
        installed_node = consumer / "node_modules" / package["name"]
        for name in ["index.cjs", "index.d.ts", "markitai.node", *licenses]:
            if identity(installed_node / name) != identity(node_source / name):
                raise RuntimeError(f"Installed Node package {name} differs from packed source")
        record["installed_node_native"] = identity(installed_node / "markitai.node")
        tests = (root / "bindings/node/test.cjs").read_text(encoding="utf-8")
        if tests.count("require('./index.cjs')") != 1:
            raise RuntimeError("Node test entrypoint changed; update the installed-package harness")
        (consumer / "test.cjs").write_text(tests.replace("require('./index.cjs')", "require('markitai')"), encoding="utf-8")
        run("installed-node-tests", ["node", "--test", "test.cjs"], cwd=consumer)

        wheels = output / "python"
        wheels.mkdir()
        raw_wheels = wheels / "original"
        raw_wheels.mkdir()
        maturin = shutil.which("maturin")
        if not maturin:
            raise RuntimeError("Install maturin>=1.9,<2 before package validation")
        run("python-wheel", [maturin, "build", "--release", "--locked", "--out", raw_wheels,
                             "--interpreter", sys.executable], cwd=root / "bindings/python")
        record["source_after_wheel_build"] = snapshot()
        if record["source_before"] != record["source_after_wheel_build"]:
            raise RuntimeError("Source bytes changed during the wheel build")
        built = list(raw_wheels.glob("*.whl"))
        if len(built) != 1:
            raise RuntimeError("Expected exactly one native wheel")
        artifact(built[0])
        wheel = wheels / built[0].name
        record["wheel_supplement"] = supplement_wheel_licenses(built[0], wheel, licenses)
        artifact(wheel)
        venv = work / "python-env"
        run("python-environment", [sys.executable, "-m", "venv", venv])
        python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        run("python-install", [python, "-m", "pip", "install", "--no-index", "--no-deps", wheel])
        python_probe = """import hashlib, importlib.metadata, json, pathlib, sys, markitai
from markitai import _native
distribution = importlib.metadata.distribution('markitai')
files = {}
for item in distribution.files:
    if '.dist-info/licenses/' in str(item):
        path = pathlib.Path(distribution.locate_file(item))
        key = str(item).split('.dist-info/licenses/', 1)[1]
        files[key] = {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
native = pathlib.Path(_native.__file__)
print(json.dumps({'module': markitai.__file__, 'prefix': sys.prefix, 'licenses': files,
    'native': {'file': native.name, 'bytes': native.stat().st_size, 'sha256': hashlib.sha256(native.read_bytes()).hexdigest()}}))
"""
        installed_python = json.loads(run("installed-python-provenance", [python, "-c", python_probe], cwd=work))
        if not Path(installed_python["module"]).resolve().is_relative_to(venv.resolve()):
            raise RuntimeError("Python imported markitai outside the installed environment")
        for name in licenses:
            if installed_python["licenses"].get(name) != identity(root / name):
                raise RuntimeError(f"Installed wheel {name} differs from the source")
        with zipfile.ZipFile(wheel) as bundle:
            native_names = [name for name in bundle.namelist()
                            if name == "markitai/" + installed_python["native"]["file"]]
            if len(native_names) != 1:
                raise RuntimeError("Installed Python native extension is not present in the wheel")
            native_bytes = bundle.read(native_names[0])
        if (installed_python["native"]["sha256"] != hashlib.sha256(native_bytes).hexdigest()
                or installed_python["native"]["bytes"] != len(native_bytes)):
            raise RuntimeError("Installed Python extension differs from its wheel")
        record["installed_python"] = installed_python
        run("installed-python-tests", [python, "-m", "unittest", "discover", "-s",
                                       root / "bindings/python/tests", "-v"], cwd=work)

        if os.name == "nt":
            record["limitations"].append("Windows Go/cgo import-library distribution is not validated by this job.")
            record["go"] = "not_run_windows_cgo_distribution_pending"
        else:
            run("go-tests", ["go", "test", "-race", "./..."], cwd=root / "bindings/go")
            record["go"] = "source_package_race_tests_passed"
        metadata = json.loads(run("cargo-metadata", ["cargo", "metadata", "--locked", "--format-version", "1"]))
        licenses = [{key: package.get(key) for key in ["name", "version", "license", "license_file", "source"]}
                    for package in metadata["packages"]]
        inventory = output / "dependency-licenses.json"
        inventory.write_text(json.dumps(licenses, indent=2) + "\n", encoding="utf-8")
        artifact(inventory)
        record["limitations"].append("Dependency license metadata is an inventory, not a completed redistribution review.")
        record["source_status_after"] = git("status", "--porcelain")
        record["source_after"] = snapshot()
        if (record["source_revision"] != git("rev-parse", "HEAD")
                or record["source_status_before"] != record["source_status_after"]
                or record["source_before"] != record["source_after"]):
            raise RuntimeError("Repository state changed during package validation")
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
    finally:
        if "source_before" in record and "source_after" not in record:
            try:
                record["source_after"] = snapshot()
                record["source_status_after"] = git("status", "--porcelain")
                record["source_revision_after"] = git("rev-parse", "HEAD")
                record["source_changed"] = (record["source_before"] != record["source_after"]
                    or record["source_status_after"] != record["source_status_before"]
                    or record["source_revision_after"] != record["source_revision"])
            except Exception as error:
                record["source_after_error"] = str(error)
        record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / "evidence.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": record["status"], "evidence": str(output / "evidence.json")}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
