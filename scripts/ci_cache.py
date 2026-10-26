"""Keep the check workflow's Cargo cache to what is slow to build and safe to reuse.

The cache holds `target/` and Cargo's downloads. Third-party crates are keyed by
Cargo.lock, so Cargo reuses them as they are. The vendored crates under
`vendor/` are path dependencies: after a checkout their files are newer than
any cached artifact and Cargo would rebuild them all. `restore` gives them an
old modification time, but only when the cache was saved from the same
`vendor/` tree (`git rev-parse HEAD:vendor`), so a changed vendored crate always
rebuilds. The workspace's own crates always rebuild.

    python scripts/ci_cache.py key       # the dependencies' part of the cache key
    python scripts/ci_cache.py restore   # after the cache restore, before cargo
    python scripts/ci_cache.py prune     # before the cache save

`prune` removes the workspace crates' artifacts and executables (rebuilt by every
run), and, when the vendored crates were rebuilt here, their artifacts from the
cache's older tree, then records this tree. It never deletes a source file.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

STATE = ".ci-cache.json"
# The time vendored sources are given; any cached artifact is newer.
OLD = 946684800  # 2000-01-01
# The artifact directories Cargo reuses; everything else in a profile is per-run output.
KEPT = ("deps", "build", ".fingerprint")


def dependency_key(root):
    """A hash of what the cached artifacts depend on: the third-party packages Cargo.lock pins and the
    manifests, leaving out `version = ` lines, so a release commit (which bumps the workspace version)
    keeps the key."""
    digest = hashlib.sha256()
    for package in (root / "Cargo.lock").read_text(encoding="utf-8").split("[[package]]"):
        if "\nsource = " in package:
            digest.update(package.encode())
    manifests = [root / "Cargo.toml", *sorted((root / "crates").glob("*/Cargo.toml")),
                 *sorted((root / ".cargo").glob("config*"))]
    for manifest in manifests:
        text = manifest.read_text(encoding="utf-8")
        digest.update(manifest.relative_to(root).as_posix().encode() + b"\0")
        digest.update(re.sub(r"(?m)^version = .*$", "", text).encode())
    return digest.hexdigest()[:32]


def git(root, *arguments):
    return subprocess.run(["git", *arguments], cwd=root, capture_output=True, text=True, check=True).stdout


def vendor_tree(root):
    return git(root, "rev-parse", "HEAD:vendor").strip()


def crate_names(root):
    """`(workspace, vendored)`: the names a crate's artifacts start with ('-' as '_'), from `cargo metadata`."""
    metadata = json.loads(subprocess.run(["cargo", "metadata", "--format-version", "1", "--locked", "--offline"],
                                         cwd=root, capture_output=True, text=True, check=True).stdout)
    members = set(metadata["workspace_members"])
    vendor = (root / "vendor").resolve()
    workspace, vendored = set(), set()
    for package in metadata["packages"]:
        if package["id"] in members:
            names = [package["name"], *(target["name"] for target in package["targets"])]
            workspace |= {name.replace("-", "_") for name in names}
        elif Path(package["manifest_path"]).resolve().is_relative_to(vendor):
            # A dependency builds only its library and build script, named after it.
            names = [package["name"], *(target["name"] for target in package["targets"]
                                        if not {"bin", "example", "test", "bench"} & set(target["kind"]))]
            vendored |= {name.replace("-", "_") for name in names}
    return workspace, vendored


def crate_of(entry):
    """`libfoo_bar-0123abcd.rlib`, `foo-bar-0123abcd` or `foo_bar-0123abcd.d` -> `foo_bar`."""
    stem = entry.rsplit("-", 1)[0] if "-" in entry else entry.split(".", 1)[0]
    return stem.removeprefix("lib").replace("-", "_")


def profiles(target):
    """The profile directories under `target` (`debug`, `<triple>/debug`, ...)."""
    found = []
    for directory in [target, *(path for path in target.iterdir() if path.is_dir())] if target.is_dir() else []:
        found += [path for path in directory.iterdir() if path.is_dir() and (path / ".fingerprint").is_dir()]
    return found


def remove(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def restore(root, target):
    state_file = target / STATE
    if not state_file.is_file():
        print("ci_cache: no cached build; everything builds")
        state = {}
    else:
        state = json.loads(state_file.read_text(encoding="utf-8"))
    current = vendor_tree(root)
    reused = state.get("vendor") == current
    if reused:
        for name in git(root, "ls-files", "-z", "vendor").split("\0"):
            if name and (root / name).is_file() and not (root / name).is_symlink():
                os.utime(root / name, (OLD, OLD))
        print(f"ci_cache: the cache was built from this vendor/ tree ({current[:12]}); its vendored crates are reused")
    elif state:
        print(f"ci_cache: the cache was built from vendor/ {state.get('vendor', '?')[:12]}, this is {current[:12]}; "
              "the vendored crates rebuild")
    if state:
        state_file.write_text(json.dumps({**state, "restored": time.time(), "vendor_reused": reused}),
                              encoding="utf-8")
    return 0


def prune(root, target):
    state_file = target / STATE
    state = json.loads(state_file.read_text(encoding="utf-8")) if state_file.is_file() else {}
    workspace, vendored = crate_names(root)
    # Vendored artifacts older than this run belong to the cache's other vendor/ tree.
    stale_before = state["restored"] if state.get("restored") and not state.get("vendor_reused") else None
    removed = 0
    for entry in target.iterdir() if target.is_dir() else []:
        if entry.name != STATE and entry.name != "CACHEDIR.TAG" and not profiles(entry) \
                and not (entry / ".fingerprint").is_dir():
            remove(entry)
    for profile in profiles(target):
        for entry in profile.iterdir():
            if entry.name not in KEPT:
                remove(entry)
        for kept in KEPT:
            for entry in (profile / kept).iterdir() if (profile / kept).is_dir() else []:
                crate = crate_of(entry.name)
                if crate in workspace or (stale_before and crate in vendored
                                          and entry.stat().st_mtime < stale_before):
                    remove(entry)
                    removed += 1
    state_file.write_text(json.dumps({"vendor": vendor_tree(root)}), encoding="utf-8")
    print(f"ci_cache: removed {removed} workspace{' and stale vendored' if stale_before else ''} artifacts")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["key", "restore", "prune"])
    parser.add_argument("--target", help="the Cargo target directory (default: target)")
    options = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    target = Path(options.target) if options.target else root / "target"
    if options.command == "key":
        print(dependency_key(root))
        return 0
    return (restore if options.command == "restore" else prune)(root, target)


if __name__ == "__main__":
    sys.exit(main())
