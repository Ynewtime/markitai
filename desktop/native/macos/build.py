#!/usr/bin/env python3.13
"""Build Markitai Native.app, the macOS app (desktop/DESIGN.md section 2).

    python3.13 desktop/native/macos/build.py            build desktop/native/macos/build/Markitai Native.app
    python3.13 desktop/native/macos/build.py --check    also run the tests, isolated
    python3.13 desktop/native/macos/build.py --no-core  reuse target/release/libmarkitai_app.a

Needs cargo (with the repository's toolchain), the Command Line Tools'
Swift, sips, iconutil and codesign.
"""
from pathlib import Path
import json
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
import tomllib

PACKAGE = Path(__file__).resolve().parent
REPOSITORY = PACKAGE.parents[2]
LIBRARY = REPOSITORY / "target" / "release" / "libmarkitai_app.a"
READER = "markitai-vision-reader"
APP = PACKAGE / "build" / "Markitai Native.app"
EXECUTABLE = "Markitai Native"
BUNDLE_ID = "dev.markitai.native"
MINIMUM_SYSTEM = "13.0"


def run(*command, cwd=PACKAGE, env=None):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], cwd=cwd, env=env, check=True)


def output(*command, cwd=PACKAGE):
    return subprocess.run([str(part) for part in command], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def build_core():
    """The core as a static library and Vision's reader (docs/bindings.md), built for the
    app's minimum system. The reader builds alone, so it takes none of the app's features."""
    env = dict(os.environ, MACOSX_DEPLOYMENT_TARGET=MINIMUM_SYSTEM)
    run("cargo", "rustc", "-p", "markitai-app", "--release", "--lib", "--crate-type", "staticlib", cwd=REPOSITORY, env=env)
    run("cargo", "build", "-p", READER, "--release", "--locked", cwd=REPOSITORY, env=env)


def workspace_version():
    with open(REPOSITORY / "Cargo.toml", "rb") as file:
        return tomllib.load(file)["workspace"]["package"]["version"]


def strings():
    return json.loads((REPOSITORY / "desktop" / "strings.json").read_text(encoding="utf-8"))


def info_plist(version, copy):
    return {
        "CFBundleDevelopmentRegion": "en",
        "CFBundleDisplayName": copy["app.name.native"]["en"],
        "CFBundleExecutable": EXECUTABLE,
        "CFBundleIconFile": "Markitai",
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleLocalizations": ["en", "zh-Hans"],
        "CFBundleName": copy["app.name.native"]["en"],
        "CFBundlePackageType": "APPL",
        "CFBundleShortVersionString": version,
        "CFBundleVersion": version,
        "CFBundleDocumentTypes": [{
            "CFBundleTypeName": copy["app.name.native"]["en"],
            "CFBundleTypeRole": "Viewer",
            "LSHandlerRank": "None",
            "LSItemContentTypes": ["public.data"],
        }],
        "LSApplicationCategoryType": "public.app-category.productivity",
        "LSMinimumSystemVersion": MINIMUM_SYSTEM,
        "NSDocumentsFolderUsageDescription": copy["plist.documents"]["en"],
        "NSDesktopFolderUsageDescription": copy["plist.desktop"]["en"],
        "NSDownloadsFolderUsageDescription": copy["plist.downloads"]["en"],
        "NSHighResolutionCapable": True,
        "NSHumanReadableCopyright": copy["settings.about.copyright"]["en"],
        "NSPrincipalClass": "NSApplication",
        "NSSupportsAutomaticTermination": False,
        "NSSupportsSuddenTermination": False,
    }


def localized_info(copy, language):
    """InfoPlist.strings: the usage descriptions in each language."""
    keys = {
        "NSDocumentsFolderUsageDescription": "plist.documents",
        "NSDesktopFolderUsageDescription": "plist.desktop",
        "NSDownloadsFolderUsageDescription": "plist.downloads",
        "NSHumanReadableCopyright": "settings.about.copyright",
    }
    lines = [f'"{plist_key}" = {json.dumps(copy[key][language], ensure_ascii=False)};' for plist_key, key in keys.items()]
    return "\n".join(lines) + "\n"


def icon(resources, work):
    iconset = work / "Markitai.iconset"
    iconset.mkdir()
    source = REPOSITORY / "desktop" / "icon.png"
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            name = f"icon_{size}x{size}{'@2x' if scale == 2 else ''}.png"
            pixels = size * scale
            run("sips", "-z", pixels, pixels, source, "--out", iconset / name)
    run("iconutil", "-c", "icns", iconset, "-o", resources / "Markitai.icns")


def assemble():
    # Swift Build links the app with the deployment target as its SDK
    # version, which would run it in macOS 13 compatibility mode; name the
    # SDK it was built with so AppKit gives it the current system's design.
    sdk = output("xcrun", "--show-sdk-version")
    platform = ["-Xlinker", "-platform_version", "-Xlinker", "macos", "-Xlinker", MINIMUM_SYSTEM, "-Xlinker", sdk]
    run("swift", "build", "-c", "release", "--product", "Markitai", *platform)
    binary = Path(output("swift", "build", "-c", "release", "--show-bin-path")) / "Markitai"
    build = output("vtool", "-show-build", binary)
    if f"sdk {sdk}" not in build or f"minos {MINIMUM_SYSTEM}" not in build:
        raise SystemExit(f"{binary} has the wrong build version:\n{build}")
    copy = strings()
    if APP.exists():
        shutil.rmtree(APP)
    contents = APP / "Contents"
    resources = contents / "Resources"
    (contents / "MacOS").mkdir(parents=True)
    resources.mkdir()
    shutil.copy2(binary, contents / "MacOS" / EXECUTABLE)
    reader = contents / "MacOS" / READER
    shutil.copy2(LIBRARY.with_name(READER), reader)
    with open(contents / "Info.plist", "wb") as file:
        plistlib.dump(info_plist(workspace_version(), copy), file)
    (contents / "PkgInfo").write_text("APPL????")
    for language in ("en", "zh-Hans"):
        folder = resources / f"{language}.lproj"
        folder.mkdir()
        (folder / "InfoPlist.strings").write_text(localized_info(copy, language), encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="markitai-icon-") as work:
        icon(resources, Path(work))
    run("codesign", "--force", "--sign", "-", "--timestamp=none", reader)
    run("codesign", "--force", "--sign", "-", "--timestamp=none", APP)
    run("codesign", "--verify", "--strict", APP)
    run("plutil", "-lint", contents / "Info.plist")
    print(APP)


def check():
    """The tests, with HOME, the Foundation home and MARKITAI_HOME isolated,
    and no provider keys or model from this shell. They compile with this
    shell's HOME, where the compiler finds the Testing macros. Swift Build's
    explicit module scan intermittently leaves out the Command Line Tools'
    Testing macro plugin, so the build names its folder. Every target builds
    with complete concurrency checking (Package.swift), and here a Swift
    warning fails the check."""
    run(sys.executable, REPOSITORY / "desktop" / "scripts" / "strings.py", "--check")
    plugins = Path(output("xcrun", "--find", "swift")).parents[1] / "lib" / "swift" / "host" / "plugins" / "testing"
    run("swift", "build", "--build-tests", "-Xswiftc", "-plugin-path", "-Xswiftc", plugins, "-Xswiftc", "-warnings-as-errors")
    with tempfile.TemporaryDirectory(prefix="markitai-macos-check-") as root:
        root = Path(root).resolve()
        home, markitai, temporary = root / "home", root / "markitai-home", root / "tmp"
        for folder in (home, markitai, temporary):
            folder.mkdir()
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(home),
            "CFFIXED_USER_HOME": str(home),
            "MARKITAI_HOME": str(markitai),
            "TMPDIR": str(temporary) + "/",
            "LANG": "en_US.UTF-8",
        }
        if "DEVELOPER_DIR" in os.environ:
            env["DEVELOPER_DIR"] = os.environ["DEVELOPER_DIR"]
        run("swift", "test", "--skip-build", env=env)


def main():
    if "--no-core" not in sys.argv[1:]:
        build_core()
    for built in (LIBRARY, LIBRARY.with_name(READER)):
        if not built.exists():
            raise SystemExit(f"{built} is missing: run without --no-core")
    assemble()
    if "--check" in sys.argv[1:]:
        check()


if __name__ == "__main__":
    main()
