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
import queue
import stat
import subprocess
import threading
import sys
import tarfile
import tempfile
import zipfile

from pricing_attribution import pricing_files
from codex_attribution import codex_files
from license_overlay import upstream_files
from portable_attribution import portable_files
from executable_identity import verify_target_executable
from cli_documentation import cli_documentation, validate_documentation


def identity(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def write_cli_zip(binary, alternate, archive, licenses, windows, documentation=None):
    """The MCP entry is a directly executable name on every supported host."""
    payloads = _cli_payloads(licenses, documentation)
    for path in [binary, alternate]:
        if not stat.S_ISREG(path.lstat().st_mode):
            raise RuntimeError("CLI ZIP inputs must be regular non-symlink files")
    before = [identity(path) for path in [binary, alternate]]
    extension = ".exe" if windows else ""
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(binary, "markitai" + extension)
        bundle.write(alternate, "mkai" + extension)
        for name, content in payloads.items():
            bundle.writestr(name, content)
        if windows:
            # arg0 selects the MCP command in the shared executable. A copy
            # needs no cmd.exe, shell quoting, extra process or new crate.
            bundle.write(binary, "markitai-mcp.exe")
        else:
            alias = zipfile.ZipInfo("markitai-mcp")
            alias.create_system = 3
            alias.external_attr = (0o120777 << 16)
            bundle.writestr(alias, b"markitai")
    if before != [identity(path) for path in [binary, alternate]]:
        raise RuntimeError("CLI executable bytes changed during ZIP packaging")


def extract_cli_zip(archive_path, destination, binary, alternate, licenses, windows, documentation=None):
    """Validate every member before creating a fresh extraction directory."""
    payloads = _cli_payloads(licenses, documentation)
    extension = ".exe" if windows else ""
    executable_names = {"markitai" + extension: binary, "mkai" + extension: alternate}
    alias_name = "markitai-mcp" + extension
    expected = {*executable_names, alias_name, *payloads}
    with zipfile.ZipFile(archive_path) as bundle:
        members = bundle.infolist()
        if len(members) != len(expected) or {item.filename for item in members} != expected:
            raise RuntimeError("CLI ZIP inventory differs from the declared files")
        for item in members:
            kind = stat.S_IFMT(item.external_attr >> 16)
            if item.filename == alias_name and not windows:
                if kind != stat.S_IFLNK or bundle.read(item) != b"markitai":
                    raise RuntimeError("CLI ZIP MCP alias is not the declared relative symlink")
                continue
            if item.is_dir() or kind not in {0, stat.S_IFREG}:
                raise RuntimeError("CLI ZIP member is not a regular file")
            source = executable_names.get(item.filename)
            if item.filename == alias_name:
                source = binary
            if source is None:
                content = payloads[item.filename]
                if item.file_size != len(content) or bundle.read(item) != content:
                    raise RuntimeError("CLI ZIP payload differs from source")
            else:
                expected_identity = identity(source)
                if item.file_size != expected_identity["bytes"]:
                    raise RuntimeError("CLI ZIP executable size differs from the frozen binary")
                digest = hashlib.sha256()
                with bundle.open(item) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if {"bytes": item.file_size, "sha256": digest.hexdigest()} != expected_identity:
                    raise RuntimeError("CLI ZIP executable differs from the frozen binary")
        destination.mkdir(parents=True, exist_ok=False)
        for item in members:
            target = destination / item.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            if item.filename == alias_name and not windows:
                target.symlink_to("markitai")
            else:
                with bundle.open(item) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if item.filename in executable_names or item.filename == alias_name:
                    target.chmod(0o755)
        for name, source in executable_names.items():
            if identity(destination / name) != identity(source):
                raise RuntimeError("Extracted CLI executable differs from source")
        if identity(destination / alias_name) != identity(binary):
            raise RuntimeError("Extracted MCP alias differs from source")
        for name, content in payloads.items():
            if (destination / name).read_bytes() != content:
                raise RuntimeError("Extracted CLI payload differs from source")
    return {"archive": identity(archive_path),
            "executables": {name: identity(destination / name) for name in executable_names},
            "mcp_alias": {"kind": "executable_copy" if windows else "symlink", "target": "markitai" + extension,
                          "identity": identity(destination / alias_name)},
            "attribution": {name: identity(destination / name) for name in licenses},
            "documentation": {name: identity(destination / name) for name in (documentation or {})}}


def mcp_probe(launcher, cwd, environment, log, timeout=30):
    """Exercise the installed executable over real JSON-RPC stdin/stdout."""
    messages = queue.Queue()
    stderr = bytearray()
    process = subprocess.Popen([str(launcher)], cwd=cwd, env=environment,
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def read_stdout():
        try:
            total = 0
            while line := process.stdout.readline(65537):
                total += len(line)
                if len(line) > 65536 or total > 1024 * 1024:
                    raise RuntimeError("MCP probe output exceeded its bound")
                messages.put(json.loads(line))
        except Exception as error:
            messages.put(error)
        finally:
            messages.put(None)

    def read_stderr():
        while block := process.stderr.read(8192):
            if len(stderr) < 1024 * 1024:
                stderr.extend(block[:1024 * 1024 - len(stderr)])

    tasks = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for task in tasks:
        task.start()
    received = []

    def send(value):
        process.stdin.write(json.dumps(value).encode("utf-8") + b"\n")
        process.stdin.flush()

    def response(identifier):
        value = messages.get(timeout=timeout)
        if isinstance(value, Exception):
            raise value
        if not isinstance(value, dict) or value.get("jsonrpc") != "2.0" or value.get("id") != identifier or "error" in value:
            raise RuntimeError("MCP probe did not receive its successful JSON-RPC response")
        received.append(value)
        return value.get("result")

    try:
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2025-11-25", "capabilities": {},
            "clientInfo": {"name": "markitai-package-probe", "version": "1"}}})
        result = response(1)
        if not isinstance(result, dict) or result.get("protocolVersion") != "2025-11-25":
            raise RuntimeError("MCP initialize negotiated an unexpected protocol")
        send({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
        result = response(2)
        tools = result.get("tools") if isinstance(result, dict) else None
        names = [item.get("name") for item in tools] if isinstance(tools, list) and all(isinstance(t, dict) for t in tools) else None
        if names != ["convert_document", "convert_url", "batch_convert", "job_status"]:
            raise RuntimeError("MCP installed tool inventory is incomplete or unexpected")
        process.stdin.close()
        if process.wait(timeout=timeout):
            raise RuntimeError("MCP installed executable failed after EOF")
        return {"ran": True, "command": [str(launcher)], "cwd": str(cwd),
                "protocol_version": "2025-11-25", "tools": names, "exit_code": process.returncode}
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()
        for task in tasks:
            task.join(timeout=5)
        for stream in [process.stdin, process.stdout, process.stderr]:
            stream.close()
        log.write_bytes(json.dumps({"responses": received, "exit_code": process.returncode},
                                   indent=2, ensure_ascii=False).encode("utf-8") + b"\n" + bytes(stderr))


def doctor_probe(binary, cwd, environment, log):
    """Record readiness separately from packaging; never install dependencies."""
    result = subprocess.run([str(binary), "doctor", "--json"], cwd=cwd, env=environment,
                            capture_output=True, timeout=120)
    log.write_bytes(result.stdout + b"\n" + result.stderr)
    checks = json.loads(result.stdout)
    statuses = {"ok", "warning", "missing", "error"}
    if (result.returncode not in {0, 1} or not isinstance(checks, dict) or not checks
            or any(not isinstance(check, dict) or check.get("status") not in statuses for check in checks.values())):
        raise RuntimeError("Installed doctor did not return a valid diagnostic report")
    return {"ran": True, "command": [str(binary), "doctor", "--json"], "cwd": str(cwd),
            "exit_code": result.returncode, "configuration_ready": result.returncode == 0,
            "checks": checks,
            "limitations": [] if result.returncode == 0 else ["The isolated configuration has unmet requirements; no repair/download was attempted."]}


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
    pricing data, Codex, portable engines and vendored upstream licenses."""
    licenses = {name: (root / name).read_bytes() for name in ["LICENSE", "NOTICE"]}
    licenses.update(pricing_files(root))
    licenses.update(codex_files(root))
    licenses.update(upstream_files(root))
    licenses.update(portable_files(root))
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
        if (not name or not path.parts or path.is_absolute() or path.as_posix() != name
                or any(part in {"", ".", ".."} for part in path.parts)
                or "\\" in name or "\0" in name or ":" in name or any(part.endswith((".", " ")) for part in path.parts)
                or path.parts[0].lower() in {"markitai", "mkai", "markitai-mcp", "markitai.exe", "mkai.exe", "markitai-mcp.exe"}):
            raise RuntimeError("Unsafe CLI attribution path")


def _cli_payloads(licenses, documentation):
    _cli_license_paths(licenses)
    if documentation is not None:
        validate_documentation(documentation)
    payloads = dict(licenses)
    for name, content in (documentation or {}).items():
        if name in payloads:
            raise RuntimeError("CLI documentation collides with attribution")
        payloads[name] = content
    folded = set()
    for name, content in payloads.items():
        if not isinstance(content, bytes):
            raise RuntimeError("CLI payload must contain bytes")
        key = name.casefold()
        if key in folded:
            raise RuntimeError("CLI payload paths collide on a case-insensitive filesystem")
        folded.add(key)
    for key in folded:
        parts = key.split("/")
        if any("/".join(parts[:end]) in folded for end in range(1, len(parts))):
            raise RuntimeError("CLI payload file collides with another payload's parent directory")
    return payloads


def write_single_binary_tar(binary, destination, licenses, documentation=None):
    """Unix delivery: one regular executable, relative aliases, exact attribution."""
    payloads = _cli_payloads(licenses, documentation)
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
        for name, content in payloads.items():
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def extract_single_binary_tar(archive_path, destination, binary, licenses, documentation=None):
    """Validate the complete inventory before extracting our narrowly shaped tar."""
    payloads = _cli_payloads(licenses, documentation)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        expected = {"markitai", "mkai", "markitai-mcp", *payloads}
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
        for name, content in payloads.items():
            member = by_name[name]
            if (not member.isfile() or member.mode != 0o644 or member.size != len(content)
                    or archive.extractfile(member).read() != content):
                raise RuntimeError("CLI payload is missing or differs from source")
        destination.mkdir(parents=True, exist_ok=False)
        target = destination / "markitai"
        with target.open("xb") as stream:
            shutil.copyfileobj(archive.extractfile(executable), stream)
        target.chmod(0o755)
        if identity(target) != identity(binary):
            raise RuntimeError("Archived CLI executable differs from the frozen binary")
        for name, content in payloads.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(content)
            if target.read_bytes() != content:
                raise RuntimeError("Extracted CLI payload differs from source")
        for name in ["mkai", "markitai-mcp"]:
            (destination / name).symlink_to("markitai")
            if os.readlink(destination / name) != "markitai" or identity(destination / name) != identity(binary):
                raise RuntimeError("Extracted CLI alias does not address the frozen executable")
    return {"archive": identity(archive_path), "executable": identity(destination / "markitai"),
            "aliases": [{"path": name, "target": "markitai", "kind": "symlink"}
                        for name in ["mkai", "markitai-mcp"]],
            "attribution": {name: identity(destination / name) for name in licenses},
            "documentation": {name: identity(destination / name) for name in (documentation or {})}}


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
    isolated_home = work / "home"
    isolated_home.mkdir()
    (work / "tmp").mkdir()
    npm_config = work / "npmrc"
    npm_config.write_text("", encoding="utf-8")
    # Help assertions use English independently of a Windows runner's UI language.
    environment = dict(os.environ, MARKITAI_HOME=str(state), PYO3_PYTHON=sys.executable,
                       HOME=str(isolated_home), MARKITAI_LANG="en", TMPDIR=str(work / "tmp"),
                       TMP=str(work / "tmp"), TEMP=str(work / "tmp"),
                       npm_config_userconfig=str(npm_config))
    environment.setdefault("CARGO_HOME", str(Path.home() / ".cargo"))
    environment.setdefault("RUSTUP_HOME", str(Path.home() / ".rustup"))
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
        record["cli_executable"] = {**identity(binary), **verify_target_executable(binary, host)}
        alternate = release / ("mkai" + extension)
        verify_target_executable(alternate, host)
        version = run("version", [binary, "--version"]).strip().split()[-1]

        cli_licenses = cli_attribution(root, licenses)
        cli_docs = cli_documentation(root)
        archive = output / f"markitai-{version}-{host}.zip"
        write_cli_zip(binary, release / ("mkai" + extension), archive, cli_licenses, os.name == "nt", cli_docs)
        artifact(archive)
        extracted = work / "CLI 安装 with spaces"
        record["cli_zip"] = extract_cli_zip(archive, extracted, binary, alternate, cli_licenses, os.name == "nt", cli_docs)
        if os.name == "nt":
            for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe"]:
                verify_target_executable(extracted / name, host)
        alias_version = run("zip-alias-version", [extracted / ("mkai" + extension), "--version"], cwd=extracted).strip()
        if alias_version.split()[-1:] != [version]:
            raise RuntimeError("Archived mkai reported a different version")
        launcher = extracted / ("markitai-mcp" + extension)
        help_text = run("zip-mcp-alias-help", [launcher, "--help"], cwd=extracted)
        if "mcp" not in help_text.lower() or "Commands:" in help_text:
            raise RuntimeError("Archived MCP executable did not select its own command")
        if os.name == "nt":
            alias_version = run("zip-mcp-alias-version", [launcher, "--version"], cwd=extracted).strip()
            if alias_version != f"markitai-mcp {version}":
                raise RuntimeError("Windows MCP executable did not report its own name and version")
        record["cli_mcp_alias"] = {**record["cli_zip"]["mcp_alias"], "executed": True}
        protocol_log = output / "logs" / "installed-mcp-protocol.log"
        record["cli_mcp_protocol"] = mcp_probe(launcher, extracted, environment, protocol_log)
        record["cli_mcp_protocol"]["log_identity"] = identity(protocol_log)
        if os.name != "nt":
            single = output / f"markitai-{version}-{host}-single-binary.tar.gz"
            write_single_binary_tar(binary, single, cli_licenses, cli_docs)
            artifact(single)
            unpacked = work / "single-binary-cli"
            record["single_binary_cli"] = extract_single_binary_tar(single, unpacked, binary, cli_licenses, cli_docs)
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
        doctor_log = output / "logs" / "installed-doctor.log"
        record["doctor"] = doctor_probe(extracted / ("markitai" + extension), work, environment, doctor_log)
        record["doctor"]["log_identity"] = identity(doctor_log)

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
