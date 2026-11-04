#!/usr/bin/env python3
"""Build Markitai for GNOME on Linux and stage its install tree.

    python3 native/linux/build.py           release binary, staged in native/linux/build/
    python3 native/linux/build.py --check   strings, tests and clippy

Python 3.13 or newer, standard library only. Linux distributions ship it as
python3 rather than python3.13.

The staged tree is an install prefix: bin/markitai-app,
share/applications, share/icons/hicolor and share/metainfo. Copy it to
/usr/local or ~/.local, or run build/bin/markitai-app in place.
"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
NATIVE = HERE.parent
APP_ID = "dev.markitai.Markitai"


def run(*args, env=None):
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], cwd=HERE, env=env, check=True)


def isolated_env(root):
    """Tests never see the real home, markitai home or XDG folders."""
    home = Path.home()
    env = dict(os.environ)
    env.setdefault("CARGO_HOME", str(home / ".cargo"))
    env.setdefault("RUSTUP_HOME", str(home / ".rustup"))
    env.update(
        HOME=str(root / "home"),
        MARKITAI_HOME=str(root / "home" / ".markitai"),
        XDG_DATA_HOME=str(root / "data"),
        XDG_CONFIG_HOME=str(root / "config"),
        XDG_CACHE_HOME=str(root / "cache"),
        XDG_STATE_HOME=str(root / "state"),
    )
    for key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "XDG_STATE_HOME"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def check():
    run(sys.executable, NATIVE / "scripts" / "strings.py", "--check")
    with tempfile.TemporaryDirectory(prefix="markitai-linux-check-") as root:
        env = isolated_env(Path(root))
        run("cargo", "test", "--locked", env=env)
        run("cargo", "clippy", "--locked", "--all-targets", "--", "-D", "warnings", env=env)
    for tool, args in (
        ("desktop-file-validate", [HERE / "data" / f"{APP_ID}.desktop"]),
        ("appstreamcli", ["validate", "--no-net", HERE / "data" / f"{APP_ID}.metainfo.xml"]),
    ):
        if shutil.which(tool):
            run(tool, *args)
        else:
            print(f"skip: {tool} is not installed", flush=True)


def target_dir():
    metadata = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps", "--locked"],
        cwd=HERE, check=True, capture_output=True, text=True,
    )
    return Path(json.loads(metadata.stdout)["target_directory"])


def build():
    run("cargo", "build", "--release", "--locked")
    binary = target_dir() / "release" / "markitai-app"
    stage = HERE / "build"
    if stage.exists():
        shutil.rmtree(stage)
    data = HERE / "data"
    files = {
        stage / "bin" / "markitai-app": binary,
        stage / "share" / "applications" / f"{APP_ID}.desktop": data / f"{APP_ID}.desktop",
        stage / "share" / "metainfo" / f"{APP_ID}.metainfo.xml": data / f"{APP_ID}.metainfo.xml",
    }
    for icon in sorted((data / "icons").rglob("*.png")):
        files[stage / "share" / "icons" / icon.relative_to(data / "icons")] = icon
    for target, source in files.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    (stage / "bin" / "markitai-app").chmod(0o755)
    for target in sorted(files):
        print(f"{target.relative_to(HERE)}  {target.stat().st_size} bytes")


def main(argv):
    if argv not in ([], ["--check"]):
        raise SystemExit(__doc__)
    if not sys.platform.startswith("linux"):
        raise SystemExit("build.py runs on Linux")
    if sys.version_info < (3, 13):
        raise SystemExit("build.py needs Python 3.13 or newer")
    if argv == ["--check"]:
        check()
    else:
        build()


if __name__ == "__main__":
    main(sys.argv[1:])
