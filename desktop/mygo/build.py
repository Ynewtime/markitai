#!/usr/bin/env python3.13
"""Build Markitai Mygo: the markitai_app core library, then the mygo app.

    python3.13 desktop/mygo/build.py                    core, then the app for this machine, in desktop/mygo/build/
    python3.13 desktop/mygo/build.py --platform linux/amd64
    python3.13 desktop/mygo/build.py --no-core          reuse the core library built before
    python3.13 desktop/mygo/build.py --check            core, strings, go vet and go test; builds no app
    python3.13 desktop/mygo/build.py --e2e              a test build with mygo's automation plugin, in desktop/mygo/build/e2e/

Python 3.13 or newer, standard library only, on macOS, Windows or Linux.
Needs cargo (the repository's toolchain) and Go (desktop/mygo/go.mod's
version). --check never writes a tracked file and runs the tests with HOME,
MARKITAI_HOME and every user folder in a temporary folder.
"""
from pathlib import Path
import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import tomllib

HERE = Path(__file__).resolve().parent
DESKTOP = HERE.parent
REPO = DESKTOP.parent
MINIMUM_MACOS = "13.0"

LIBRARY = {"darwin": "libmarkitai_app.dylib", "windows": "markitai_app.dll", "linux": "libmarkitai_app.so"}
RUST_TARGET = {
    "darwin/arm64": "aarch64-apple-darwin",
    "darwin/amd64": "x86_64-apple-darwin",
    "windows/amd64": "x86_64-pc-windows-msvc",
    "linux/amd64": "x86_64-unknown-linux-gnu",
    "linux/arm64": "aarch64-unknown-linux-gnu",
}
COPIES = {
    DESKTOP / "strings.json": HERE / "internal" / "l10n" / "strings.json",
    DESKTOP / "icon.png": HERE / "internal" / "brand" / "icon.png",
}


def run(*args, cwd=HERE, env=None):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)


def host_platform():
    goos = {"darwin": "darwin", "win32": "windows"}.get(sys.platform, "linux")
    arch = platform.machine().lower()
    return f"{goos}/{ {'x86_64': 'amd64', 'amd64': 'amd64', 'aarch64': 'arm64', 'arm64': 'arm64'}[arch] }"


def target_root():
    return Path(os.environ.get("CARGO_TARGET_DIR") or REPO / "target")


def core_library(plat):
    """Where cargo leaves the core for plat, and the cargo command that builds it there.

    Windows links the C runtime statically, as the CLI does, so the app needs
    no Visual C++ redistributable; that build keeps its own target folder so
    other builds keep their flags. The host's other builds use target/release.
    """
    goos = plat.split("/")[0]
    command = ["cargo", "rustc", "-p", "markitai-app", "--release", "--locked", "--lib", "--crate-type", "cdylib"]
    env = dict(os.environ)
    if goos == "windows":
        target_dir = target_root() / "mygo-windows"
        flags = env.pop("CARGO_ENCODED_RUSTFLAGS", None)
        flags = flags.split("\x1f") if flags else env.pop("RUSTFLAGS", "").split()
        env["CARGO_ENCODED_RUSTFLAGS"] = "\x1f".join([*flags, "-Ctarget-feature=+crt-static"])
        command += ["--target", RUST_TARGET[plat], "--target-dir", target_dir]
        out = target_dir / RUST_TARGET[plat] / "release"
    elif plat == host_platform():
        out = target_root() / "release"
    else:
        command += ["--target", RUST_TARGET[plat]]
        out = target_root() / RUST_TARGET[plat] / "release"
    if goos == "darwin":
        # Without it the library's install name is its absolute path under target/.
        command += ["--", "-C", "link-arg=-Wl,-install_name,@rpath/libmarkitai_app.dylib"]
        env["MACOSX_DEPLOYMENT_TARGET"] = MINIMUM_MACOS
        if "SDKROOT" not in env and shutil.which("xcrun"):
            env["SDKROOT"] = subprocess.run(["xcrun", "--show-sdk-path"], check=True, capture_output=True, text=True).stdout.strip()
    return out / LIBRARY[goos], command, env


def stage_core(plat, build):
    """Builds the core unless build is False, and stages it in resources/<goos>-<goarch>/."""
    library, command, env = core_library(plat)
    if build:
        run(*command, cwd=REPO, env=env)
    if not library.exists():
        raise SystemExit(f"{library} is missing: build the core first (drop --no-core)")
    staged = HERE / "resources" / plat.replace("/", "-") / library.name
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(library, staged)
    print(f"staged {staged.relative_to(REPO)}  {staged.stat().st_size} bytes", flush=True)
    return staged


def check_version():
    with open(REPO / "Cargo.toml", "rb") as file:
        workspace = tomllib.load(file)["workspace"]["package"]["version"]
    app = json.loads((HERE / "mygo.json").read_text(encoding="utf-8"))["version"]
    if app != workspace:
        raise SystemExit(f"desktop/mygo/mygo.json says version {app}, the workspace {workspace}")


def copies(write):
    """The module's copies of files outside it, which go:embed cannot reach."""
    for source, copy in COPIES.items():
        if copy.exists() and copy.read_bytes() == source.read_bytes():
            continue
        if not write:
            raise SystemExit(f"{copy.relative_to(REPO)} differs from {source.relative_to(REPO)}: run desktop/mygo/build.py, which copies it")
        shutil.copyfile(source, copy)
        print(f"copied {source.relative_to(REPO)} to {copy.relative_to(REPO)}", flush=True)


def isolated_env(root, library):
    """go test never sees the real home, markitai home or user folders."""
    env = dict(os.environ)
    for key, value in (("GOMODCACHE", "env GOMODCACHE"), ("GOCACHE", "env GOCACHE"), ("GOPATH", "env GOPATH")):
        env.setdefault(key, subprocess.run(["go", *value.split()], cwd=HERE, check=True, capture_output=True, text=True).stdout.strip())
    home = root / "home"
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        MARKITAI_HOME=str(home / ".markitai"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_CACHE_HOME=str(root / "cache"),
        XDG_STATE_HOME=str(root / "state"),
        APPDATA=str(root / "AppData" / "Roaming"),
        LOCALAPPDATA=str(root / "AppData" / "Local"),
        TERM="dumb",
        MARKITAI_APP_LIB=str(library),
        CGO_ENABLED="0",
    )
    for key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME", "APPDATA", "LOCALAPPDATA"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def check(build_core):
    plat = host_platform()
    library = stage_core(plat, build_core)
    run(sys.executable, DESKTOP / "scripts" / "strings.py", "--check", cwd=REPO)
    copies(write=False)
    check_version()
    with tempfile.TemporaryDirectory(prefix="markitai-mygo-check-") as root:
        env = isolated_env(Path(root), library)
        run("go", "vet", "./...", env=env)
        run("go", "test", "-count=1", "./...", env=env)


def build(plat, build_core, e2e):
    """The release app, or with e2e a production build that keeps mygo's inspector and links its
    automation plugin (e2e.go), into build/e2e/: never ship it."""
    stage_core(plat, build_core)
    copies(write=True)
    check_version()
    env = dict(os.environ, CGO_ENABLED="0")
    command = ["go", "tool", "mygo", "build", "-platform", plat, "-skip-installer"]
    out = HERE / "build"
    if e2e:
        out = out / "e2e"
        env.update(GOFLAGS="-tags=e2e", MYGO_INSPECTOR="1")
        command += ["-o", out]
    if plat.startswith("darwin/"):
        command.append("-skip-dmg")
    run(*command, env=env)
    for path in sorted((out / plat.replace("/", "-")).rglob("*")):
        if path.is_file() and not path.is_symlink():
            print(f"{path.relative_to(HERE)}  {path.stat().st_size} bytes")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="core, strings, go vet and go test; no app")
    parser.add_argument("--no-core", action="store_true", help="reuse the core library cargo built before")
    parser.add_argument("--platform", choices=sorted(RUST_TARGET), help="GOOS/GOARCH to build for (default: this machine)")
    parser.add_argument("--e2e", action="store_true", help="a test build with mygo's automation plugin, in build/e2e/")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 13):
        raise SystemExit("build.py needs Python 3.13 or newer")
    if args.check:
        if args.platform or args.e2e:
            raise SystemExit("--check tests this machine's platform; drop --platform and --e2e")
        check(build_core=not args.no_core)
    else:
        build(args.platform or host_platform(), build_core=not args.no_core, e2e=args.e2e)


if __name__ == "__main__":
    main(sys.argv[1:])
