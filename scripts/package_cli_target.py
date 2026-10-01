"""Build and package the single-binary CLI for one non-host Cargo target.

`ci_packages.py` validates native builds only. This script cross-builds
`markitai-cli` in release mode for another target (such as x86-64 macOS on
Apple silicon), writes the same single-binary archive as the native run (one
executable, `mkai` and `markitai-mcp` relative symlinks, the CLI's
attribution), re-reads it through the same inventory check, and records the
executable's identity and instruction set. When this host can execute the
target (x86-64 macOS under Rosetta 2), the archived executable and its aliases
also run: version, MCP alias selection, a Markdown round trip and
`doctor --json`. Otherwise the record says it was not run.

An Apple target builds for macOS 11.0, the oldest system the arm64 build
names, unless `MACOSX_DEPLOYMENT_TARGET` is set; Rust's x86-64 default
(10.12) would claim systems nothing was tested on. The minimum the
executable records is read back from its load commands.

The record is not a signature, a notarization or evidence from the target's
own hardware.
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

from ci_packages import (cli_attribution, extract_single_binary_tar, identity, package_attribution,
                         source_snapshot, write_single_binary_tar)

# Mach-O CPU types and ELF machines by the architecture a target triple names.
MACHO = {0x01000007: "x86_64", 0x0100000C: "aarch64"}
ELF = {62: "x86_64", 183: "aarch64"}


def executable_arch(path):
    """The instruction set of a thin 64-bit Mach-O or ELF executable, else None."""
    with path.open("rb") as stream:
        header = stream.read(20)
    if len(header) < 20:
        return None
    if header[:4] == b"\xcf\xfa\xed\xfe":
        return MACHO.get(struct.unpack_from("<I", header, 4)[0])
    if header[:4] == b"\x7fELF" and header[4] == 2 and header[5] == 1:
        return ELF.get(struct.unpack_from("<H", header, 18)[0])
    return None


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
    parser.add_argument("--target", required=True, help="Cargo target triple other than the host's")
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
    if args.target == host:
        raise SystemExit("The host target is packaged and tested by ci_packages.py")
    if os.name == "nt" or "windows" in args.target:
        raise SystemExit("Windows archives are packaged by ci_packages.py on Windows")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "logs").mkdir()
    local = root / ".local"
    local.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="cli-target-", dir=local))
    environment = dict(os.environ, MARKITAI_HOME=str(work / "state"))
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
        run("build", ["cargo", "build", "--release", "--locked", "-p", "markitai-cli", "--target", args.target])
        record["source_after_build"] = snapshot()
        if record["source_before"] != record["source_after_build"]:
            raise RuntimeError("Source bytes changed during the build")

        binary = target_dir / args.target / "release" / "markitai"
        arch = executable_arch(binary)
        if arch != expected_arch:
            raise RuntimeError(f"Built executable is {arch}, not {expected_arch}")
        version = workspace_version((root / "Cargo.toml").read_text(encoding="utf-8"))
        record["executable"] = {**identity(binary), "arch": arch, "version": version}
        if args.target.endswith("-apple-darwin"):
            record["executable"]["minimum_macos"] = macho_minimum(binary)
            if record["executable"]["minimum_macos"] != record["deployment_target"]:
                raise RuntimeError(f"Executable records macOS {record['executable']['minimum_macos']}, "
                                   f"not the deployment target {record['deployment_target']}")
            signature = subprocess.run(["codesign", "-dv", str(binary)], capture_output=True, text=True)
            record["executable"]["code_signature"] = (
                "none" if "not signed" in signature.stderr else signature.stderr.strip().splitlines()[-1:])
        cli_licenses = cli_attribution(root, package_attribution(root))
        archive = output / f"markitai-{version}-{args.target}-single-binary.tar.gz"
        write_single_binary_tar(binary, archive, cli_licenses)
        unpacked = work / "single-binary-cli"
        record["single_binary_cli"] = extract_single_binary_tar(archive, unpacked, binary, cli_licenses)
        record["artifacts"] = {archive.name: identity(archive)}

        try:
            reported = subprocess.run([str(unpacked / "markitai"), "--version"], cwd=work, env=environment,
                                      capture_output=True, text=True, timeout=60)
        except OSError as error:
            record["executed"] = {"ran": False, "reason": f"{type(error).__name__}: {error}"}
        else:
            if reported.returncode or reported.stdout.split()[-1:] != [version]:
                raise RuntimeError(f"Archived executable reported {reported.stdout!r}")
            alias = run("alias-version", [unpacked / "mkai", "--version"], cwd=unpacked).strip()
            if alias.split()[-1] != version:
                raise RuntimeError("Archived mkai alias reported a different version")
            help_text = run("mcp-alias-help", [unpacked / "markitai-mcp", "--help"], cwd=unpacked)
            if "mcp" not in help_text.lower() or "Commands:" in help_text:
                raise RuntimeError("Archived MCP alias did not select the MCP subcommand")
            source = work / "source.md"
            source.write_text("# Target package\n\nHello 世界.\n", encoding="utf-8")
            text = run("round-trip", [unpacked / "markitai", source, "--pure"], cwd=work)
            if text != source.read_text(encoding="utf-8"):
                raise RuntimeError("Archived executable changed plain Unicode content")
            doctor = json.loads(run("doctor", [unpacked / "markitai", "doctor", "--json"], cwd=work))
            record["executed"] = {"ran": True, "version": reported.stdout.strip(), "alias_version": alias,
                                  "doctor_keys": sorted(doctor) if isinstance(doctor, dict) else None}
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
