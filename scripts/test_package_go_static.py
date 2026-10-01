"""Static-package counterexamples; no native compiler, conversion or installation."""
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from package_go_static import (EM_X86_64, TARGETS, bundle_licenses, dependency_closure,
                              elf_archive_objects, host_target, identity, inventory,
                              linked_libraries, loaded_libraries, package_archive,
                              parse_native_flags, symbol_versions, unpack_verified,
                              validate_linkage, verify_build)

ROOT = Path(__file__).resolve().parents[1]
# Compiler notes as rustc 1.98.1 printed them for each packaged target.
NOTES = {
    "aarch64-apple-darwin": "note: native-static-libs: -framework Vision -framework Foundation -framework ImageIO "
                            "-framework CoreGraphics -framework CoreFoundation -lobjc -framework Foundation "
                            "-framework CoreFoundation -liconv -lSystem -lc -lm",
    "x86_64-unknown-linux-gnu": "note: native-static-libs: -lgcc_s -lutil -lrt -lpthread -lm -ldl -lc",
}


def elf_object(machine=EM_X86_64, kind=1, header=b"\x7fELF\x02\x01", extra=b""):
    ident = header + bytes(16 - len(header))
    return ident + struct.pack("<HH", kind, machine) + bytes(44) + extra


def ar_archive(members, magic=b"!<arch>\n"):
    data = bytearray(magic)
    for name, payload in members:
        data += f"{name:<16}{0:<12}{0:<6}{0:<6}{644:<8}{len(payload):<10}`\n".encode()
        data += payload + (b"\n" if len(payload) % 2 else b"")
    return bytes(data)


# A Go consumer's dynamic section and loader output on Ubuntu 24.04 amd64. Go
# names the loader itself; native kernels also list the vDSO, which Rosetta
# does not provide.
READELF = """
Dynamic section at offset 0x20ff650 contains 27 entries:
  Tag        Type                         Name/Value
 0x0000000000000001 (NEEDED)             Shared library: [libgcc_s.so.1]
 0x0000000000000001 (NEEDED)             Shared library: [libm.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [libc.so.6]
 0x0000000000000001 (NEEDED)             Shared library: [ld-linux-x86-64.so.2]
 0x000000000000000c (INIT)               0x405000
 0x000000006ffffffe (VERNEED)            0x4024d0
"""
LDD = """\tlinux-vdso.so.1 (0x00007ffc2b3f9000)
\tlibgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fffff791000)
\tlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f3c1a4e7000)
\tlibc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fffff495000)
\t/lib64/ld-linux-x86-64.so.2 (0x00007ffffffc6000)
"""

class StaticGoPackageTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_compiler_note_keeps_only_explicit_system_dependencies(self):
        raw = "Compiling markitai\nnote: native-static-libs: -framework Vision -lobjc -lc -lc\nFinished\n"
        self.assertEqual(parse_native_flags(raw), [("framework", "Vision"), ("library", "c"), ("library", "objc")])
        for bad in ["Finished", "native-static-libs: ", "native-static-libs: -L/private -lfoo",
                    "native-static-libs: -Wl,-rpath,/repo", "native-static-libs: -framework ../bad",
                    "native-static-libs: -lc\nnative-static-libs: -lm"]:
            with self.subTest(raw=bad), self.assertRaises(RuntimeError):
                parse_native_flags(bad)

    def test_static_linkage_cannot_fall_back_to_a_dynamic_library(self):
        required = parse_native_flags("native-static-libs: -framework Vision -lobjc")
        prefix = "#cgo LDFLAGS: ${SRCDIR}/native/darwin_arm64/libmarkitai_ffi.a "
        self.assertIn(("framework", "Vision"), validate_linkage(prefix + "-framework Vision -lobjc -lc", required))
        for wrong in ["#cgo LDFLAGS: -L${SRCDIR}/native -lmarkitai_ffi -framework Vision -lobjc",
                      prefix + "-lobjc", prefix + "-framework Vision -lobjc -L/tmp"]:
            with self.assertRaises(RuntimeError):
                validate_linkage(wrong, required)

    def test_each_packaged_target_is_chosen_only_on_its_own_native_host(self):
        self.assertEqual(host_target("darwin", "arm64", None).triple, "aarch64-apple-darwin")
        self.assertEqual(host_target("linux", "x86_64", "2.39").triple, "x86_64-unknown-linux-gnu")
        for host in [("linux", "x86_64", None), ("linux", "aarch64", "2.39"), ("darwin", "x86_64", None),
                     ("win32", "AMD64", None), ("freebsd14", "amd64", None)]:
            with self.subTest(host=host):
                self.assertIsNone(host_target(*host))
        self.assertEqual({target.directory for target in TARGETS.values()}, {"darwin_arm64", "linux_amd64"})

    def test_repository_linkage_names_each_archive_and_its_recorded_compiler_dependencies(self):
        for triple, target in TARGETS.items():
            with self.subTest(target=triple):
                text = (ROOT / f"bindings/go/link_static_{target.directory}.go").read_text()
                required = parse_native_flags(NOTES[triple])
                self.assertTrue(set(required) <= set(validate_linkage(text, required, target.directory)))
                other = next(each for each in TARGETS.values() if each != target).directory
                with self.assertRaisesRegex(RuntimeError, "archive explicitly"):
                    validate_linkage(text, required, other)
        linux = parse_native_flags(NOTES["x86_64-unknown-linux-gnu"])
        prefix = "#cgo LDFLAGS: ${SRCDIR}/native/linux_amd64/libmarkitai_ffi.a "
        with self.assertRaisesRegex(RuntimeError, "omits"):
            validate_linkage(prefix + "-lutil -lrt -lpthread -lm -ldl -lc", linux, "linux_amd64")

    def test_linux_archive_holds_only_relocatable_x86_64_objects(self):
        good = [("/", b"\0\0\0\0"), ("//", b"long_member_name.rcgu.o/\n"), ("a.o/", elf_object(extra=b"x")),
                ("/0", elf_object())]
        path = self.root / "lib.a"
        path.write_bytes(ar_archive(good))
        self.assertEqual(elf_archive_objects(path, EM_X86_64), 2)
        bad = {"other machine": ar_archive([("a.o/", elf_object(machine=183))]),
               "shared object": ar_archive([("a.o/", elf_object(kind=3))]),
               "32-bit": ar_archive([("a.o/", elf_object(header=b"\x7fELF\x01\x01"))]),
               "big-endian": ar_archive([("a.o/", elf_object(header=b"\x7fELF\x02\x02"))]),
               "not an object": ar_archive([("a.o/", elf_object()), ("notes.txt/", b"plain text, not ELF" * 4)]),
               "short member": ar_archive([("a.o/", b"\x7fELF")]),
               "thin archive": ar_archive([("a.o/", elf_object())], magic=b"!<thin>\n"),
               "no objects": ar_archive([("/", b"\0\0\0\0")]),
               "truncated": ar_archive([("a.o/", elf_object())])[:-8],
               "bad header": ar_archive([("a.o/", elf_object())]).replace(b"`\n", b"\n\n", 1)}
        for name, data in bad.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                path.write_bytes(data)
                elf_archive_objects(path, EM_X86_64)

    def test_linux_consumer_links_only_system_libraries_without_rpath(self):
        self.assertEqual(linked_libraries(READELF),
                         ["libgcc_s.so.1", "libm.so.6", "libc.so.6", "ld-linux-x86-64.so.2"])
        needed = " 0x0000000000000001 (NEEDED)             Shared library: [{}]\n"
        bad = {"runpath": " 0x000000000000001d (RUNPATH)            Library runpath: [/repo/target/release]\n",
               "rpath": " 0x000000000000000f (RPATH)              Library rpath: [$ORIGIN]\n",
               "markitai library": needed.format("libmarkitai_ffi.so"),
               "other system library": needed.format("libz.so.1")}
        for name, line in bad.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                linked_libraries(READELF + line)
        with self.assertRaises(RuntimeError):
            linked_libraries("There is no dynamic section in this file.\n")

    def test_linux_consumer_loads_its_libraries_from_system_directories(self):
        needed = linked_libraries(READELF)
        resolved = loaded_libraries(LDD, needed)
        self.assertIn(["ld-linux-x86-64.so.2", "/lib64/ld-linux-x86-64.so.2"], resolved)
        self.assertIn(["linux-vdso.so.1", ""], resolved)
        bad = {"not found": LDD.replace("libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f3c1a4e7000)",
                                        "libm.so.6 => not found"),
               "private copy": LDD.replace("/lib/x86_64-linux-gnu/libc.so.6", "/home/ci/lib/libc.so.6"),
               "renamed copy": LDD.replace("/lib/x86_64-linux-gnu/libm.so.6", "/lib/x86_64-linux-gnu/libc.so.6"),
               "system copy of markitai": LDD + "\tlibmarkitai_ffi.so => /usr/lib/x86_64-linux-gnu/libmarkitai_ffi.so (0x00007f3c1a000000)\n",
               "unresolved needed": LDD.replace("\tlibm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007f3c1a4e7000)\n", ""),
               "pathless library": LDD + "\tlibfoo.so.1 (0x00007f3c1a000000)\n",
               "unknown line": LDD + "\tstatically linked\n"}
        for name, loader in bad.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                loaded_libraries(loader, needed)

    def test_symbol_versions_name_the_newest_glibc_numerically(self):
        table = """
0000000000000000      DF *UND*\t0000000000000000 (GLIBC_2.2.5) free
0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.9   pipe2
0000000000000000  w   DF *UND*\t0000000000000000  GLIBC_2.34  __libc_start_main
0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.34  pthread_create
0000000000000000      DF *UND*\t0000000000000000 (GCC_3.0)    _Unwind_Resume
0000000000000000      DF *UND*\t0000000000000000  GLIBC_2.17  clock_gettime
"""
        result = symbol_versions(table)
        self.assertEqual(result["glibc_minimum"], "2.34")
        self.assertEqual(result["glibc_minimum_symbols"], ["__libc_start_main", "pthread_create"])
        self.assertEqual(result["versions"], {"GLIBC": ["2.2.5", "2.9", "2.17", "2.34"], "GCC": ["3.0"]})
        for bad in ["0000000000000000      DF *UND*\t0000000000000000  GLIBC_PRIVATE _dl_x\n" + table,
                    "0000000000000000      DF *UND*\t0000000000000000  Base  free\n"]:
            with self.assertRaises(RuntimeError):
                symbol_versions(bad)

    def metadata(self):
        repository = self.root / "repository"
        source = repository / "crates/markitai-ffi"
        source.mkdir(parents=True)
        (repository / "LICENSE").write_bytes(b"root license\n")
        (repository / "NOTICE").write_bytes(b"root attribution\n")
        package = self.root / "registry/dependency"
        package.mkdir(parents=True)
        (package / "LICENSE-MIT").write_bytes(b"first copyright\r\nMIT text\r\n")
        (package / "LICENSE-other-bits").write_bytes(b"bundled original terms\n")
        (package / "terms.txt").write_bytes(b"declared license file\n")
        (package / "license.rs").write_text("// This is code, not license evidence")
        missing = self.root / "registry/missing"
        missing.mkdir()
        packages = [
            {"id": "ffi", "name": "markitai-ffi", "version": "1", "source": None,
             "license": "MIT", "license_file": None, "manifest_path": str(source / "Cargo.toml")},
            {"id": "dep", "name": "dependency", "version": "2", "source": "registry+fixture",
             "license": "MIT", "license_file": "terms.txt", "manifest_path": str(package / "Cargo.toml")},
            {"id": "missing", "name": "missing", "version": "3", "source": "registry+fixture",
             "license": "MIT", "license_file": None, "manifest_path": str(missing / "Cargo.toml")},
            {"id": "unrelated", "name": "not-linked", "version": "4", "source": "registry+fixture"},
        ]
        metadata = {"packages": packages, "resolve": {"nodes": [
            {"id": "ffi", "deps": [{"pkg": "dep"}, {"pkg": "missing"}]},
            {"id": "dep", "deps": [{"pkg": "missing"}]}, {"id": "missing", "deps": []}]}}
        return repository, metadata

    def test_license_collection_preserves_texts_and_exposes_missing_sources(self):
        repository, metadata = self.metadata()
        sysroot = self.root / "rust"
        text = sysroot / "share/doc/rust/COPYRIGHT-library.html"
        text.parent.mkdir(parents=True)
        text.write_bytes(b"<p>Toolchain original notice</p>\n")
        (text.parent / "licenses").mkdir()
        (text.parent / "licenses/MIT.txt").write_bytes(b"Original toolchain MIT text\n")
        stage = self.root / "module"
        stage.mkdir()
        record = bundle_licenses(metadata, stage / "licenses", repository, sysroot)
        self.assertEqual({item["id"] for item in record["dependencies"]}, {"ffi", "dep", "missing"})
        self.assertEqual(record["legal_review"], "not_performed")
        self.assertEqual([item["id"] for item in record["unresolved"]], ["missing"])
        dependency = next(item for item in record["dependencies"] if item["id"] == "dep")
        self.assertEqual(len(dependency["texts"]), 3)
        for item in dependency["texts"] + record["rust_toolchain_texts"]:
            self.assertEqual((stage / item["path"]).read_bytes(), Path(item["source"]).read_bytes())
            self.assertEqual(identity(stage / item["path"])["sha256"], item["sha256"])
        self.assertEqual(len(record["rust_toolchain_texts"]), 2)
        metadata["resolve"]["nodes"].pop()
        with self.assertRaisesRegex(RuntimeError, "Incomplete"):
            dependency_closure(metadata)

    def test_declared_notice_cannot_escape_its_package(self):
        repository, metadata = self.metadata()
        metadata["packages"][1]["license_file"] = "../../outside.txt"
        with self.assertRaisesRegex(RuntimeError, "escapes"):
            bundle_licenses(metadata, self.root / "licenses", repository, self.root / "rust")

    def test_archive_is_relocatable_exact_and_rejects_missing_or_changed_members(self):
        module = self.root / "module"
        (module / "native/include").mkdir(parents=True)
        (module / "markitai.go").write_bytes(b"package markitai\n")
        (module / "native/include/markitai.h").write_bytes(bytes(range(256)))
        expected = inventory(module)
        archive = self.root / "package.tgz"
        self.assertEqual(package_archive(module, archive), expected)
        installed = self.root / "installed"
        unpack_verified(archive, installed, expected)
        self.assertEqual(inventory(installed), expected)
        second = self.root / "second.tgz"
        package_archive(module, second)
        self.assertEqual(archive.read_bytes(), second.read_bytes())
        with self.assertRaises(FileExistsError):
            unpack_verified(archive, installed, expected)
        wrong = {**expected, "missing.go": identity(module / "markitai.go")}
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            unpack_verified(archive, self.root / "missing", wrong)
        (module / "markitai.go").write_bytes(b"changed payload\n")
        package_archive(module, self.root / "changed.tgz")
        with self.assertRaisesRegex(RuntimeError, "differs"):
            unpack_verified(self.root / "changed.tgz", self.root / "changed", expected)

    def test_archive_links_and_duplicates_are_not_installable(self):
        for kind in ["symlink", "duplicate"]:
            archive = self.root / f"{kind}.tgz"
            expected = {"file": {"bytes": 1, "sha256": hashlib.sha256(b"x").hexdigest()}}
            with tarfile.open(archive, "w:gz") as bundle:
                entry = tarfile.TarInfo("markitai-go/file")
                if kind == "symlink":
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = "../../outside"
                    bundle.addfile(entry)
                else:
                    entry.size = 1
                    bundle.addfile(entry, io.BytesIO(b"x"))
                    bundle.addfile(entry, io.BytesIO(b"x"))
            with self.assertRaises(RuntimeError):
                unpack_verified(archive, self.root / kind, expected)

    def test_build_identity_checks_bytes_not_only_a_supplied_revision(self):
        source = self.root / "file.rs"
        source.write_bytes(b"original")
        library = self.root / "library.a"
        library.write_bytes(b"archive")
        build = {"source_revision": "a" * 40, "status": "passed", "source_unchanged": True,
                 "source_files": {"file.rs": identity(source)}, "archive": identity(library)}
        def git(command, **kwargs):
            return {"rev-parse": "a" * 40, "status": b"", "ls-files": b"file.rs\0"}[command[1]]
        with patch("package_go_static.subprocess.check_output", side_effect=git):
            self.assertEqual(verify_build(self.root, "a" * 40, build, {"archive": library}), build["source_files"])
            source.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "Source files"):
                verify_build(self.root, "a" * 40, build, {"archive": library})
            source.write_bytes(b"original")
            library.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "archive differs"):
                verify_build(self.root, "a" * 40, build, {"archive": library})


if __name__ == "__main__":
    unittest.main()
