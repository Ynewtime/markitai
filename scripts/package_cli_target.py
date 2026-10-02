"""Package a cross-target Unix CLI or either native Windows MSVC64 CLI.

Unix targets retain the single-binary tar and relative aliases. Windows targets
require their own native host, produce a ZIP with three real EXE entries, and
verify PE architecture before exercising the installed CLI and MCP protocol.
Every package carries complete portable-engine attribution. Executed probes
use isolated HOME/state and report dependency readiness without installing it.

Apple builds default to macOS 11.0 and read the minimum back from load commands.
The record is not signing, publishing or evidence from another target's hardware.
"""
from pathlib import Path
import argparse
import datetime
import json
import os
import platform
import re
import struct
import subprocess
import sys
import tempfile

from ci_packages import (cli_attribution, doctor_probe, extract_cli_zip, extract_single_binary_tar,
                         identity, mcp_probe, package_attribution, source_snapshot, write_cli_zip,
                         write_single_binary_tar)
from executable_identity import (MACHO, ELF, WINDOWS_TARGETS, executable_arch,
                                 verify_target_executable)


def validate_target(target, host, host_platform):
    """Keep native Windows support explicit; do not imply cross-link acceptance."""
    windows = "windows" in target
    if windows:
        if target not in WINDOWS_TARGETS:
            raise SystemExit("Windows CLI packaging supports only x86_64/aarch64-pc-windows-msvc")
        if host_platform != "win32" or target != host:
            raise SystemExit("Windows CLI packaging requires a matching native Windows host")
    elif host_platform == "win32":
        raise SystemExit("Unix CLI packaging requires a Unix host")
    elif target == host:
        raise SystemExit("The host target is packaged and tested by ci_packages.py")
    return windows


def macho_minimum(path):
    """The minimum macOS a thin 64-bit Mach-O executable records, else None."""
    data = path.read_bytes()
    if data[:4] != b"\xcf\xfa\xed\xfe" or len(data) < 32:
        return None
    count = struct.unpack_from("<I", data, 16)[0]
    offset = 32
    for _ in range(count):
        if offset + 8 > len(data):
            return None
        command, size = struct.unpack_from("<II", data, offset)
        # LC_BUILD_VERSION names a platform (1, macOS) before its minimum;
        # LC_VERSION_MIN_MACOSX names the minimum alone.
        if command == 0x32 and offset + 16 <= len(data) and struct.unpack_from("<I", data, offset + 8)[0] == 1:
            encoded = struct.unpack_from("<I", data, offset + 12)[0]
        elif command == 0x24 and offset + 12 <= len(data):
            encoded = struct.unpack_from("<I", data, offset + 8)[0]
        else:
            if size < 8:
                return None
            offset += size
            continue
        major, minor, patch = encoded >> 16, (encoded >> 8) & 0xFF, encoded & 0xFF
        return f"{major}.{minor}" + (f".{patch}" if patch else "")
    return None


def workspace_version(manifest):
    """The `[workspace.package]` version of the root manifest."""
    section = None
    for line in manifest.splitlines():
        heading = re.match(r"\s*\[([^\]]+)\]\s*$", line)
        if heading:
            section = heading.group(1).strip()
            continue
        value = re.match(r'\s*version\s*=\s*"([^"]+)"\s*$', line)
        if section == "workspace.package" and value:
            return value.group(1)
    raise RuntimeError("The root manifest has no [workspace.package] version")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Cross-target Unix triple, or this Windows host's MSVC64 triple")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[a-z0-9_]+(-[a-z0-9_.]+){2,3}", args.target):
        raise SystemExit(f"Not a target triple: {args.target}")
    expected_arch = args.target.split("-", 1)[0]
    if expected_arch not in {*MACHO.values(), *ELF.values()}:
        raise SystemExit(f"Unsupported architecture: {expected_arch}")
    host = next(line.split(": ", 1)[1] for line in
                subprocess.check_output(["rustc", "-vV"], text=True).splitlines() if line.startswith("host: "))
    windows = validate_target(args.target, host, sys.platform)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    local = root / ".local"
    local.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cli-target-", dir=local))
    for name in ["state", "home", "tmp"]:
        (work / name).mkdir()
    environment = dict(os.environ, MARKITAI_HOME=str(work / "state"), HOME=str(work / "home"),
                       MARKITAI_LANG="en", TMPDIR=str(work / "tmp"), TMP=str(work / "tmp"), TEMP=str(work / "tmp"))
    environment.setdefault("CARGO_HOME", str(Path.home() / ".cargo"))
    environment.setdefault("RUSTUP_HOME", str(Path.home() / ".rustup"))
    environment.pop("CARGO_BUILD_TARGET", None)
    if args.target.endswith("-apple-darwin"):
        environment.setdefault("MACOSX_DEPLOYMENT_TARGET", "11.0")
    target_dir = Path(environment.get("CARGO_TARGET_DIR", root / "target"))
    if not target_dir.is_absolute():
        target_dir = root / target_dir
    record = {
        "schema": 1,
        "status": "running",
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": platform.platform(),
        "host": host,
        "target": args.target,
        "work_directory": str(work),
        "steps": [],
        "limitations": ["No signing, notarization, publishing or target-hardware result is implied."],
    }
    if "MACOSX_DEPLOYMENT_TARGET" in environment:
        record["deployment_target"] = environment["MACOSX_DEPLOYMENT_TARGET"]

    def snapshot():
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root)
        return source_snapshot(root, [os.fsdecode(name) for name in names.split(b"\0") if name])

    def run(name, command, cwd=root):
        command = [str(value) for value in command]
        log = output / "logs" / f"{len(record['steps']):02}-{name}.log"
        with log.open("wb") as stream:
            result = subprocess.run(command, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        record["steps"].append({"name": name, "command": command, "cwd": str(cwd),
                                "exit_code": result.returncode, "log": str(log.relative_to(output)),
                                "log_identity": identity(log)})
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {log}")
        return log.read_text(encoding="utf-8", errors="replace")

    try:
        record["source_revision"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                                            text=True).strip()
        record["source_status_before"] = subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                                                 text=True).strip()
        if record["source_status_before"]:
            raise RuntimeError("Target packaging requires a clean source checkout")
        record["source_before"] = snapshot()
        record["compiler"] = run("compiler", ["rustc", "-vV"])
        run("build", ["cargo", "build", "--release", "--locked", "-p", "markitai-cli", "--bins", "--target", args.target])
        record["source_after_build"] = snapshot()
        if record["source_before"] != record["source_after_build"]:
            raise RuntimeError("Source bytes changed during the build")

        extension = ".exe" if windows else ""
        binary = target_dir / args.target / "release" / ("markitai" + extension)
        executable = verify_target_executable(binary, args.target)
        version = workspace_version((root / "Cargo.toml").read_text(encoding="utf-8"))
        record["executable"] = {**identity(binary), **executable, "version": version}
        if args.target.endswith("-apple-darwin"):
            record["executable"]["minimum_macos"] = macho_minimum(binary)
            if record["executable"]["minimum_macos"] != record["deployment_target"]:
                raise RuntimeError(f"Executable records macOS {record['executable']['minimum_macos']}, "
                                   f"not the deployment target {record['deployment_target']}")
            signature = subprocess.run(["codesign", "-dv", str(binary)], capture_output=True, text=True)
            record["executable"]["code_signature"] = (
                "none" if "not signed" in signature.stderr else signature.stderr.strip().splitlines()[-1:])
        cli_licenses = cli_attribution(root, package_attribution(root))
        unpacked = work / "CLI 安装 with spaces"
        if windows:
            alternate = target_dir / args.target / "release" / "mkai.exe"
            verify_target_executable(alternate, args.target)
            archive = output / f"markitai-{version}-{args.target}.zip"
            write_cli_zip(binary, alternate, archive, cli_licenses, True)
            record["cli_zip"] = extract_cli_zip(archive, unpacked, binary, alternate, cli_licenses, True)
            for name in ["markitai.exe", "mkai.exe", "markitai-mcp.exe"]:
                verify_target_executable(unpacked / name, args.target)
        else:
            archive = output / f"markitai-{version}-{args.target}-single-binary.tar.gz"
            write_single_binary_tar(binary, archive, cli_licenses)
            record["single_binary_cli"] = extract_single_binary_tar(archive, unpacked, binary, cli_licenses)
        record["artifacts"] = {archive.name: identity(archive)}

        try:
            reported = subprocess.run([str(unpacked / ("markitai" + extension)), "--version"], cwd=work, env=environment,
                                      capture_output=True, text=True, encoding="utf-8", timeout=60)
        except OSError as error:
            if windows:
                raise RuntimeError("Native Windows installed CLI could not execute") from error
            record["executed"] = {"ran": False, "reason": f"{type(error).__name__}: {error}"}
        else:
            if reported.returncode or reported.stdout.split()[-1:] != [version]:
                raise RuntimeError(f"Archived executable reported {reported.stdout!r}")
            alias = run("alias-version", [unpacked / ("mkai" + extension), "--version"], cwd=unpacked).strip()
            if alias.split()[-1] != version:
                raise RuntimeError("Archived mkai alias reported a different version")
            help_text = run("mcp-alias-help", [unpacked / ("markitai-mcp" + extension), "--help"], cwd=unpacked)
            if "mcp" not in help_text.lower() or "Commands:" in help_text:
                raise RuntimeError("Archived MCP alias did not select the MCP subcommand")
            source = work / "source.md"
            source.write_text("# Target package\n\nHello 世界.\n", encoding="utf-8")
            text = run("round-trip", [unpacked / ("markitai" + extension), source, "--pure"], cwd=work)
            if text != source.read_text(encoding="utf-8"):
                raise RuntimeError("Archived executable changed plain Unicode content")
            launcher = unpacked / ("markitai-mcp" + extension)
            if windows:
                own_version = run("mcp-alias-version", [launcher, "--version"], cwd=unpacked).strip()
                if own_version != f"markitai-mcp {version}":
                    raise RuntimeError("Windows MCP executable did not report its own name and version")
            protocol_log = output / "logs" / "installed-mcp-protocol.log"
            protocol = mcp_probe(launcher, unpacked, environment, protocol_log)
            protocol["log_identity"] = identity(protocol_log)
            doctor_log = output / "logs" / "installed-doctor.log"
            doctor = doctor_probe(unpacked / ("markitai" + extension), work, environment, doctor_log)
            doctor["log_identity"] = identity(doctor_log)
            record["executed"] = {"ran": True, "version": reported.stdout.strip(), "alias_version": alias,
                                  "mcp_protocol": protocol, "doctor": doctor}
        record["source_after"] = snapshot()
        if record["source_before"] != record["source_after"]:
            raise RuntimeError("Source bytes changed during packaging")
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (output / "record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print(json.dumps({"status": record["status"], "artifacts": record["artifacts"],
                      "executed": record["executed"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
