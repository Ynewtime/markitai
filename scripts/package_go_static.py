"""Stage and exercise a prebuilt, self-contained Go static package for this host.

Native darwin/arm64 and linux/amd64 with glibc are the packaged targets; a
package is staged and proven only on its own target. This driver never builds
Rust or downloads dependencies. A coordinator-supplied build record binds the
archive, compiler probe, metadata and clean source tree. License collection is
mechanical evidence, not a completed redistribution review.
"""
from pathlib import Path
import argparse
import datetime
import gzip
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tarfile
from typing import NamedTuple

from license_overlay import stage_overlay
from pricing_attribution import stage_pricing_files
from codex_attribution import stage_codex_files
from portable_attribution import stage_portable_files

MAX_NOTICE = 16 * 1024 * 1024
SOURCE_NAMES = ["go.mod", "markitai.go", "markitai_test.go", "linkage_test.go", "link_dynamic.go",
                "link_static_darwin_arm64.go", "link_static_linux_amd64.go", "link_static_unsupported.go"]


class Target(NamedTuple):
    triple: str
    goos: str
    goarch: str

    @property
    def directory(self):
        """The module directory holding this target's archive, as the linkage file names it."""
        return f"{self.goos}_{self.goarch}"


TARGETS = {target.triple: target for target in [
    Target("aarch64-apple-darwin", "darwin", "arm64"),
    Target("x86_64-unknown-linux-gnu", "linux", "amd64"),
]}

# A Linux consumer may load only the C library's own objects and the GCC
# unwinder that Rust's standard library names, from the system directories.
EM_X86_64 = 62
SYSTEM_SONAMES = frozenset(["libc.so.6", "libm.so.6", "libpthread.so.0", "libdl.so.2", "librt.so.1",
                            "libutil.so.1", "libresolv.so.2", "libgcc_s.so.1", "ld-linux-x86-64.so.2"])
SYSTEM_DIRECTORIES = ("/lib/x86_64-linux-gnu/", "/usr/lib/x86_64-linux-gnu/", "/lib64/", "/usr/lib64/")
VDSO = "linux-vdso.so.1"


def glibc_version():
    """The running C library's glibc version, or None for another C library."""
    try:
        value = os.confstr("CS_GNU_LIBC_VERSION")
    except (AttributeError, ValueError, OSError):
        return None
    match = re.fullmatch(r"glibc (\d+\.\d+(?:\.\d+)?)", value or "")
    return match.group(1) if match else None


def host_target(system, machine, glibc):
    """The packaged target this host builds natively, or None."""
    if system == "darwin" and machine == "arm64":
        return TARGETS["aarch64-apple-darwin"]
    if system.startswith("linux") and machine == "x86_64" and glibc:
        return TARGETS["x86_64-unknown-linux-gnu"]
    return None


def identity(path):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise RuntimeError(f"Expected a regular non-symlink file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Unexpected staged symlink: {path}")
        if path.is_dir():
            continue
        files[path.relative_to(root).as_posix()] = identity(path)
    return files


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def copy_verified(source, destination):
    expected = identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Path(source).open("rb") as incoming, destination.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
    if identity(source) != expected or identity(destination) != expected:
        raise RuntimeError("Input changed during package copying")
    return expected


def parse_native_flags(text):
    matches = re.findall(r"(?:^|\n)(?:note: )?native-static-libs:\s*([^\n]+)", text)
    values = {tuple(shlex.split(value)) for value in matches}
    if len(values) != 1:
        raise RuntimeError("Expected one unambiguous native-static-libs compiler note")
    flags = next(iter(values))
    dependencies = set()
    index = 0
    while index < len(flags):
        token = flags[index]
        if token == "-framework" and index + 1 < len(flags):
            index += 1
            name = flags[index]
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise RuntimeError("Invalid framework in compiler note")
            dependencies.add(("framework", name))
        elif re.fullmatch(r"-l[A-Za-z0-9_+.-]+", token):
            dependencies.add(("library", token[2:]))
        else:
            raise RuntimeError(f"Unsupported native-static-libs token: {token}")
        index += 1
    if not dependencies:
        raise RuntimeError("Empty native-static-libs compiler note")
    return sorted(dependencies)


def validate_linkage(text, required, directory="darwin_arm64"):
    lines = [line.removeprefix("#cgo LDFLAGS: ") for line in text.splitlines()
             if line.startswith("#cgo LDFLAGS: ")]
    prefix = f"${{SRCDIR}}/native/{directory}/libmarkitai_ffi.a "
    if len(lines) != 1 or not lines[0].startswith(prefix):
        raise RuntimeError("Static linkage must name the packaged archive explicitly")
    configured = parse_native_flags("native-static-libs: " + lines[0][len(prefix):])
    if not set(map(tuple, required)).issubset(set(configured)):
        raise RuntimeError("Static Go linkage omits a compiler-reported native dependency")
    return configured


def elf_archive_objects(path, machine):
    """The number of objects in a GNU ar archive, each of which must be a 64-bit
    little-endian relocatable ELF object for `machine`."""
    count = 0
    with Path(path).open("rb") as stream:
        end = os.fstat(stream.fileno()).st_size
        if stream.read(8) != b"!<arch>\n":
            raise RuntimeError("Static archive is not a regular ar archive")
        while header := stream.read(60):
            if len(header) != 60 or header[58:] != b"`\n" or not header[48:58].strip().isdigit():
                raise RuntimeError("Malformed static archive member header")
            size, start = int(header[48:58]), stream.tell()
            if start + size > end:
                raise RuntimeError("Truncated static archive member")
            # The symbol index and long-name table are the only other members.
            if header[:16].rstrip(b" ") not in {b"/", b"//", b"/SYM64/"}:
                ident = stream.read(20)
                if (len(ident) < 20 or ident[:6] != b"\x7fELF\x02\x01"
                        or struct.unpack_from("<HH", ident, 16) != (1, machine)):
                    raise RuntimeError("Static archive member is not a relocatable 64-bit object for the target")
                count += 1
            stream.seek(start + size + size % 2)
    if not count:
        raise RuntimeError("Static archive contains no objects")
    return count


def elf_dynamic_entries(text):
    """NEEDED, RPATH and RUNPATH values from `readelf -d` output."""
    entries = {"NEEDED": [], "RPATH": [], "RUNPATH": []}
    for kind, value in re.findall(r"\((NEEDED|RPATH|RUNPATH)\)[^\[\n]*\[([^\]\n]*)\]", text):
        entries[kind].append(value)
    return entries


def loader_resolution(text):
    """(name, path) pairs from `ldd` output: the vDSO has the path "" and an
    unresolved library None."""
    resolved = []
    for line in filter(None, (line.strip() for line in text.splitlines())):
        if match := re.fullmatch(r"(\S+) => (\S+) \(0x[0-9a-f]+\)", line):
            resolved.append((match[1], match[2]))
        elif match := re.fullmatch(r"(\S+) => not found", line):
            resolved.append((match[1], None))
        elif match := re.fullmatch(r"(\S+) \(0x[0-9a-f]+\)", line):
            name = match[1]
            resolved.append((Path(name).name, name) if name.startswith("/") else (name, ""))
        else:
            raise RuntimeError(f"Unrecognized loader output: {line}")
    return resolved


def linked_libraries(dynamic):
    """The libraries a consumer's `readelf -d` output names, which must all be
    system libraries, without an rpath or runpath."""
    entries = elf_dynamic_entries(dynamic)
    if entries["RPATH"] or entries["RUNPATH"]:
        raise RuntimeError("Static consumer unexpectedly retains an rpath")
    needed = entries["NEEDED"]
    if not needed or any(name not in SYSTEM_SONAMES for name in needed):
        raise RuntimeError("Consumer retains a non-system dynamic library dependency")
    return needed


def loaded_libraries(loader, needed):
    """The (name, path) pairs `ldd` resolves, each a system library from a system
    directory, covering every linked library."""
    resolved = loader_resolution(loader)
    for name, path in resolved:
        if path is None:
            raise RuntimeError(f"Consumer dependency {name} is not found")
        if path == "" and name == VDSO:
            continue
        if name not in SYSTEM_SONAMES or Path(path).name != name or not path.startswith(SYSTEM_DIRECTORIES):
            raise RuntimeError(f"Consumer loads a non-system library: {name} => {path}")
    if not set(needed) <= {name for name, _ in resolved}:
        raise RuntimeError("The loader output omits a linked dependency")
    return [list(pair) for pair in resolved]


def symbol_versions(text):
    """The GLIBC and GCC symbol versions `objdump -T` output requires, the
    newest glibc among them, and the symbols requiring that version."""
    if "GLIBC_PRIVATE" in text:
        raise RuntimeError("Consumer binds a private glibc symbol")
    numeric = lambda version: tuple(map(int, version.split(".")))
    versions, symbols = {"GLIBC": set(), "GCC": set()}, {}
    for line in text.splitlines():
        for namespace, version in re.findall(r"\b(GLIBC|GCC)_(\d+(?:\.\d+)+)\b", line):
            versions[namespace].add(version)
            if namespace == "GLIBC":
                symbols.setdefault(version, set()).add(line.split()[-1])
    if not versions["GLIBC"]:
        raise RuntimeError("No glibc symbol versions found")
    newest = max(versions["GLIBC"], key=numeric)
    return {"versions": {key: sorted(value, key=numeric) for key, value in versions.items()},
            "glibc_minimum": newest, "glibc_minimum_symbols": sorted(symbols[newest])}


def dependency_closure(metadata):
    packages = {package["id"]: package for package in metadata["packages"]}
    roots = [key for key, package in packages.items() if package["name"] == "markitai-ffi"
             and package.get("source") is None]
    if len(roots) != 1 or not metadata.get("resolve"):
        raise RuntimeError("Full resolved metadata with one local markitai-ffi is required")
    nodes = {node["id"]: node for node in metadata["resolve"]["nodes"]}
    visited, pending = set(), roots.copy()
    while pending:
        key = pending.pop()
        if key in visited:
            continue
        if key not in packages or key not in nodes:
            raise RuntimeError("Incomplete resolved dependency graph")
        visited.add(key)
        # Including build, dev and target-conditional dependencies deliberately
        # over-approximates the linked closure; it must not be called an SBOM of
        # code proven reachable in the final executable.
        pending.extend(dependency["pkg"] for dependency in nodes[key]["deps"])
    return [packages[key] for key in sorted(visited)]


def notice_candidates(directory, declared):
    candidates = set()
    count = 0
    for path in directory.rglob("*"):
        count += 1
        if count > 100_000:
            raise RuntimeError("Dependency source inventory exceeds the limit")
        name = path.name.lower()
        in_notices = any(part.lower() in {"licenses", "licences", "license", "licence"}
                         for part in path.relative_to(directory).parts[:-1])
        if ((in_notices or name.startswith(("license", "licence", "copying", "notice", "copyright")))
                and path.suffix.lower() not in {".rs", ".py", ".go", ".c", ".h", ".js", ".json"}):
            if path.is_symlink():
                raise RuntimeError("Dependency notice is a symlink")
            if path.is_file():
                candidates.add(path)
    if declared:
        path = directory / declared
        if not path.resolve().is_relative_to(directory.resolve()):
            raise RuntimeError("Declared dependency license escapes its package")
        if path.is_file():
            candidates.add(path)
    return sorted(candidates)


def bundle_licenses(metadata, destination, root, sysroot):
    records, unresolved = [], []
    packages = dependency_closure(metadata)
    overlay = stage_overlay(root / "licenses/upstream", destination / "upstream", packages)
    for package in packages:
        directory = Path(package["manifest_path"]).parent
        if not directory.is_dir():
            raise RuntimeError("Dependency source directory is unavailable")
        # Workspace sources share root notices. Avoid recursively copying the
        # entire checkout as though it were one dependency's license directory.
        local = package.get("source") is None and directory.resolve().is_relative_to(root.resolve())
        if local and not directory.resolve().is_relative_to((root / "vendor").resolve()):
            candidates = [root / "LICENSE", root / "NOTICE"]
        else:
            candidates = notice_candidates(directory, package.get("license_file"))
        label = re.sub(r"[^A-Za-z0-9_.-]", "_", f"{package['name']}-{package['version']}")
        label += "-" + hashlib.sha256(package["id"].encode()).hexdigest()[:10]
        files = []
        for source in candidates:
            base = root if local and source in {root / "LICENSE", root / "NOTICE"} else directory
            relative = source.relative_to(base)
            if identity(source)["bytes"] > MAX_NOTICE:
                raise RuntimeError("Dependency license text exceeds the 16 MiB limit")
            target = destination / label / relative
            value = copy_verified(source, target)
            files.append({"source": str(source), "path": target.relative_to(destination.parent).as_posix(), **value})
        source_texts = bool(files)
        supplemental = overlay["packages"].get(package["id"])
        if supplemental:
            files.extend(supplemental["texts"])
        row = {"id": package["id"], "name": package["name"], "version": package["version"],
               "license_expression": package.get("license"), "manifest": str(directory / "Cargo.toml"),
               "source": package.get("source"), "texts": files}
        if supplemental:
            row["upstream_evidence"] = {key: supplemental[key] for key in ["content_kind", "commit", "full_text_gap", "historical_provenance"]}
        if not source_texts and not (supplemental and supplemental["complete_text"]):
            reason = supplemental["full_text_gap"] if supplemental else "No license/copyright/notice text found in the available package source"
            unresolved.append({"id": package["id"], "reason": reason})
        records.append(row)
    rust_directory = sysroot / "share/doc/rust"
    standard = []
    for source in notice_candidates(rust_directory, None) if rust_directory.is_dir() else []:
        if identity(source)["bytes"] > MAX_NOTICE:
            continue
        target = destination / "rust-toolchain" / source.relative_to(rust_directory)
        value = copy_verified(source, target)
        standard.append({"source": str(source), "path": target.relative_to(destination.parent).as_posix(), **value})
    if not standard:
        unresolved.append({"id": "rust-toolchain", "reason": "Toolchain license texts were not found under share/doc/rust"})
    return {"scope": "Conservative resolved Cargo closure, including non-runtime and other-target dependencies; toolchain notices are separate",
            "legal_review": "not_performed", "dependencies": records, "upstream_overlay": overlay["record"],
            "rust_toolchain_texts": standard, "unresolved": unresolved}


def package_archive(module, destination):
    before = inventory(module)
    with destination.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for relative in before:
                path = module / relative
                member = tarfile.TarInfo("markitai-go/" + relative)
                member.size = before[relative]["bytes"]
                member.mode = 0o644
                with path.open("rb") as incoming:
                    archive.addfile(member, incoming)
    if inventory(module) != before:
        raise RuntimeError("Staged files changed while archiving")
    return before


def unpack_verified(archive, destination, expected):
    destination.mkdir(parents=True, exist_ok=False)
    seen = set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            prefix = "markitai-go/"
            if not member.isfile() or not member.name.startswith(prefix):
                raise RuntimeError("Unexpected static-package archive member")
            relative = member.name[len(prefix):]
            if relative not in expected or relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
                raise RuntimeError("Invalid or duplicate static-package archive path")
            seen.add(relative)
            if member.size != expected[relative]["bytes"]:
                raise RuntimeError("Static-package archive member size differs")
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
            if identity(target) != expected[relative]:
                raise RuntimeError("Static-package archive member bytes differ")
    if seen != set(expected):
        raise RuntimeError("Static-package archive is incomplete")


def verify_build(root, source_revision, record, inputs):
    if record.get("status") != "passed" or record.get("source_unchanged") is not True:
        raise RuntimeError("The native build record must report a successful unchanged-source build")
    if not re.fullmatch(r"[0-9a-f]{40}", source_revision):
        raise RuntimeError("A full source revision is required")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if revision != source_revision or revision != record.get("source_revision"):
        raise RuntimeError("Source revision differs from the frozen build record")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root):
        raise RuntimeError("Static package validation requires a clean source checkout")
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).split(b"\0")
    actual = {os.fsdecode(path): identity(root / os.fsdecode(path)) for path in paths if path}
    if actual != record.get("source_files"):
        raise RuntimeError("Source files differ from the frozen build record")
    for key, path in inputs.items():
        if identity(path) != record.get(key):
            raise RuntimeError(f"{key} differs from the frozen build record")
    return actual


CONSUMER = r'''package main

import (
    "encoding/json"
    "fmt"
    "os"
    "sync"
    markitai "markitai.local/go"
)

func main() {
    if len(os.Args) != 2 { panic("one input path required") }
    options := &markitai.Options{LLM: markitai.Bool(false), OCR: markitai.Bool(false), Screenshot: markitai.Bool(false),
        Config: map[string]any{"cache": map[string]any{"enabled": false}, "history": map[string]any{"record": false}}}
    failures := make(chan error, 24)
    var workers sync.WaitGroup
    for i := 0; i < 24; i++ {
        workers.Add(1)
        go func() {
            defer workers.Done()
            output, err := markitai.Convert(os.Args[1], options)
            if err == nil && (output.Markdown != "# Static Go\n\n独立消费者 🌍\n" || output.OutputPath != nil || output.Usage.Requests != 0) {
                err = fmt.Errorf("unexpected native result")
            }
            failures <- err
        }()
    }
    workers.Wait()
    close(failures)
    for err := range failures { if err != nil { panic(err) } }
    request, err := json.Marshal(map[string]any{"source": os.Args[1]+".missing", "options": options})
    if err != nil { panic(err) }
    raw, err := markitai.ConvertJSON(request)
    if err != nil { panic(err) }
    var result struct { OK bool `json:"ok"`; Error struct { Code string `json:"code"` } `json:"error"` }
    if err = json.Unmarshal(raw, &result); err != nil { panic(err) }
    if result.OK || result.Error.Code != "not_found" { panic("missing-file contract differs") }
    if err = json.NewEncoder(os.Stdout).Encode(map[string]any{"version":markitai.Version(),"concurrent_conversions":24,"json_error":"not_found"}); err != nil { panic(err) }
}
'''


STATIC_TEXT = {
    "darwin": ("This package requires native macOS arm64, Go with cgo and Xcode Command Line Tools.",
               "macOS frameworks remain system dependencies."),
    "linux": ("This package requires native Linux x86-64 with glibc, Go with cgo and a C toolchain (gcc).",
              "glibc and the GCC runtime (libgcc_s) remain system dependencies. The executable\n"
              "needs at least the newest glibc symbol version it binds (see `objdump -T`),\n"
              "which follows the glibc it was linked against. Adding `-Wl,--gc-sections` to\n"
              "CGO_LDFLAGS drops unused archive code from the executable."),
}

LIMITATIONS = {
    "darwin": ["Only this native macOS arm64 host is exercised; no Linux, Windows or minimum-macOS execution is implied.",
               "Markitai is statically linked; macOS system frameworks and optional browser/Office installations remain external."],
    "linux": ["Only this native Linux x86-64 glibc host is exercised; other distributions, older glibc releases, musl and arm64 are not implied.",
              "Markitai is statically linked; glibc, libgcc_s and optional browser/Office installations remain external."],
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["archive", "native-static-libs", "metadata", "build-record", "output"]:
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--expected-host", help="Rust target triple of this host (default: detected)")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    host_glibc = glibc_version() if sys.platform.startswith("linux") else None
    target = host_target(sys.platform, platform.machine(), host_glibc)
    if target is None or args.expected_host not in {None, target.triple}:
        parser.error("Static package delivery is validated only on native darwin/arm64 and linux/amd64 with glibc, "
                     "each for its own target; other hosts are unsupported")
    inputs = {"archive": args.archive.resolve(), "native_static_libs": args.native_static_libs.resolve(),
              "metadata": args.metadata.resolve()}
    build_record_path = args.build_record.resolve()
    build_record_identity = identity(build_record_path)
    build = json.loads(build_record_path.read_text())
    verify_build(root, args.source_revision, build, inputs)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    work = output / "work"
    work.mkdir(mode=0o700)
    (output / "logs").mkdir()
    environment = {key: os.environ[key] for key in ["PATH", "LANG", "LC_ALL", "TZ", "DEVELOPER_DIR", "SDKROOT",
                                                    "RUSTUP_TOOLCHAIN", "CARGO_HOME", "RUSTUP_HOME"] if key in os.environ}
    environment.setdefault("CARGO_HOME", str(Path.home() / ".cargo"))
    environment.setdefault("RUSTUP_HOME", str(Path.home() / ".rustup"))
    for name in ["home", "state", "tmp", "gocache", "gomodcache"]:
        (work / name).mkdir(mode=0o700)
    environment.update(HOME=str(work / "home"), MARKITAI_HOME=str(work / "state"), MARKITAI_LANG="en",
                       TMPDIR=str(work / "tmp"), TMP=str(work / "tmp"), TEMP=str(work / "tmp"),
                       GOCACHE=str(work / "gocache"), GOMODCACHE=str(work / "gomodcache"),
                       GOENV="off", GOWORK="off", GOTOOLCHAIN="local", GOPROXY="off", GOSUMDB="off", CGO_ENABLED="1")
    record = {"schema": 1, "status": "running", "source_revision": args.source_revision,
              "host": target.triple, "platform": platform.platform(),
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "build_record": {"path": str(build_record_path), **build_record_identity},
              "inputs": {key: {"path": str(path), **identity(path)} for key, path in inputs.items()},
              "steps": [], "limitations": LIMITATIONS[target.goos] +
              ["The license bundle is mechanically collected, not a completed legal review."]}
    if host_glibc:
        record["host_glibc"] = host_glibc

    def run(name, command, cwd=work, env=None, timeout=600):
        command = [str(value) for value in command]
        log = output / "logs" / f"{len(record['steps']):02}-{name}.log"
        with log.open("xb") as stream:
            child = subprocess.Popen(command, cwd=cwd, env=env or environment,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = child.wait(timeout=timeout)
            except BaseException:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
                raise
        record["steps"].append({"name": name, "command": command, "cwd": str(cwd), "exit_code": code,
                                "log": str(log.relative_to(output)), **identity(log)})
        if code:
            raise RuntimeError(f"{name} failed; inspect {log.relative_to(output)}")
        return log.read_text(errors="replace")

    try:
        native = parse_native_flags(inputs["native_static_libs"].read_text())
        linkage = root / "bindings/go" / f"link_static_{target.directory}.go"
        record["native_dependencies"] = native
        record["configured_dependencies"] = validate_linkage(linkage.read_text(), native, target.directory)
        if target.goos == "darwin":
            if run("archive-architecture", ["lipo", "-archs", inputs["archive"]]).strip() != "arm64":
                raise RuntimeError("Static archive is not a single arm64 artifact")
        else:
            record["archive_objects"] = elf_archive_objects(inputs["archive"], EM_X86_64)
        record["go_version"] = run("go-version", ["go", "version"]).strip()
        if run("go-platform", ["go", "env", "GOOS", "GOARCH", "CGO_ENABLED"]).split() != [target.goos, target.goarch, "1"]:
            raise RuntimeError("Go host or cgo setting differs from the package target")
        if target.goos == "darwin":
            # Invoking Apple's selected clang by absolute path bypasses the
            # system driver shim's SDK discovery. Preserve an explicit SDKROOT or
            # resolve the selected macOS SDK before building cgo's runtime headers.
            if not environment.get("SDKROOT"):
                environment["SDKROOT"] = run("macos-sdk-path", ["xcrun", "--sdk", "macosx", "--show-sdk-path"]).strip()
            if not (Path(environment["SDKROOT"]) / "usr/include/stdlib.h").is_file():
                raise RuntimeError("The selected macOS SDK does not provide standard C headers")
            record["macos_sdk_root"] = environment["SDKROOT"]
            environment["CC"] = run("c-compiler-path", ["xcrun", "--find", "clang"]).strip()
        else:
            compiler = shutil.which(run("c-compiler-name", ["go", "env", "CC"]).strip(), path=environment.get("PATH"))
            if not compiler:
                raise RuntimeError("Go's C compiler is not on PATH")
            environment["CC"] = compiler
        record["c_compiler"] = run("c-compiler", [environment["CC"], "--version"])
        record["rust_compiler"] = run("rust-compiler", ["rustc", "-vV"])
        if f"host: {target.triple}" not in record["rust_compiler"].splitlines():
            raise RuntimeError("The Rust compiler's host differs from the package target")
        sysroot = Path(run("rust-sysroot", ["rustc", "--print", "sysroot"]).strip())
        module = work / "staged-module"
        module.mkdir()
        for name in SOURCE_NAMES:
            copy_verified(root / "bindings/go" / name, module / name)
        copy_verified(root / "bindings/c/markitai.h", module / "native/include/markitai.h")
        copy_verified(inputs["archive"], module / "native" / target.directory / "libmarkitai_ffi.a")
        for name in ["LICENSE", "NOTICE"]:
            copy_verified(root / name, module / name)
        pricing_record = stage_pricing_files(root, module)
        record["pricing_attribution"] = pricing_record
        codex_record = stage_codex_files(root, module)
        record["codex_attribution"] = codex_record
        portable_record = stage_portable_files(root, module)
        record["portable_attribution"] = portable_record
        license_record = bundle_licenses(json.loads(inputs["metadata"].read_text()), module / "licenses", root, sysroot)
        license_record["pricing_attribution"] = pricing_record
        license_record["codex_attribution"] = codex_record
        license_record["portable_attribution"] = portable_record
        json_file(module / "licenses.json", license_record)
        record["unresolved_licenses"] = license_record["unresolved"]
        record["license_overlay"] = license_record["upstream_overlay"]
        requirement, system = STATIC_TEXT[target.goos]
        (module / "STATIC.md").write_text(f"# Go static package\n\n{requirement}\nUse `go build -tags markitai_static` (or `go test -tags markitai_static`).\nThe archive is linked into your executable; do not ship a Markitai dynamic library.\n{system} `licenses.json` records mechanical\nnotice collection and any missing texts; it is not a completed legal review.\n\nThe module name is currently markitai.local/go; consumers can use a local\n`replace markitai.local/go => /path/to/unpacked/module` in their go.mod.\n", encoding="utf-8")
        manifest = {"schema": 1, "source_revision": args.source_revision, "host": target.triple,
                    "build_record": record["build_record"], "inputs": record["inputs"],
                    "native_dependencies": native, "files": inventory(module),
                    "scope": "files excludes this manifest itself; compressed archive identity is recorded externally"}
        json_file(module / "manifest.json", manifest)
        archive = output / f"markitai-go-{target.goos}-{target.goarch}.tar.gz"
        expected = package_archive(module, archive)
        installed = work / "installed-module"
        unpack_verified(archive, installed, expected)
        record["package"] = {"path": archive.name, **identity(archive)}
        record["installed_files"] = expected
        test_environment = dict(environment, MARKITAI_TEST_NUMBERS_FIXTURES=str(root / "crates/markitai-core/src/formats/numbers/fixtures"))
        run("installed-go-tests", ["go", "test", "-race", "-v", "-count=1", "-tags", "markitai_static", "./..."], cwd=installed, env=test_environment)
        consumer = work / "consumer"
        consumer.mkdir()
        (consumer / "go.mod").write_text("module static-consumer\n\ngo 1.23\n\nrequire markitai.local/go v0.0.0\nreplace markitai.local/go => ../installed-module\n")
        (consumer / "main.go").write_text(CONSUMER)
        executable = consumer / "consumer"
        run("consumer-build", ["go", "build", "-tags", "markitai_static", "-trimpath", "-ldflags=-linkmode=external", "-o", executable, "."], cwd=consumer)
        linked = {}
        if target.goos == "darwin":
            dependencies = run("consumer-dependencies", ["otool", "-L", executable])
            libraries = [line.strip().split(" (", 1)[0] for line in dependencies.splitlines()[1:]]
            if not libraries or any(not path.startswith(("/System/Library/Frameworks/", "/usr/lib/")) for path in libraries):
                raise RuntimeError("Consumer retains a non-system dynamic library dependency")
            loads = run("consumer-load-commands", ["otool", "-l", executable])
            if "LC_RPATH" in loads:
                raise RuntimeError("Static consumer unexpectedly retains an rpath")
        else:
            libraries = linked_libraries(run("consumer-dynamic-section", ["readelf", "-d", executable]))
            linked = {"resolved_libraries": loaded_libraries(run("consumer-loader", ["ldd", executable]), libraries),
                      **symbol_versions(run("consumer-symbol-versions", ["objdump", "-T", executable]))}
        relocated = work / "relocated"
        relocated.mkdir()
        copy_verified(executable, relocated / "consumer")
        (relocated / "consumer").chmod(0o755)
        (relocated / "source.md").write_text("# Static Go\n\n独立消费者 🌍\n", encoding="utf-8")
        result = json.loads(run("relocated-consumer", [relocated / "consumer", relocated / "source.md"], cwd=relocated))
        if result.get("concurrent_conversions") != 24 or result.get("json_error") != "not_found" or not result.get("version"):
            raise RuntimeError("Relocated consumer did not prove its native calls")
        record["consumer"] = {"result": result, "binary": identity(executable), "dynamic_libraries": libraries,
                              **linked, "files": inventory(relocated)}
        if inventory(installed) != expected:
            raise RuntimeError("Installed package changed during consumer verification")
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
    finally:
        try:
            verify_build(root, args.source_revision, build, inputs)
            if identity(build_record_path) != build_record_identity:
                raise RuntimeError("Frozen build record changed")
            record["identity_after"] = "unchanged"
        except Exception as error:
            record["status"] = "failed"
            record["identity_after"] = str(error)
        record["finished_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json_file(output / "record.json", record)
    print(json.dumps({"status": record["status"], "record": str(output / "record.json"),
                      "unresolved_licenses": len(record.get("unresolved_licenses", []))}))
    return 0 if record["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
