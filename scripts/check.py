"""Run the native development gate without requiring a Unix shell."""
from pathlib import Path
import os
import shutil
import subprocess
import sys

from ci_packages import npm_command
from isolation import isolated_env
from web_deps import WEB, stale_web_dependencies

# Each interpreter names its own executable.
PROBES = {"python3": ["-c", "import sys; print(sys.executable)"], "node": ["-p", "process.execPath"]}


def unshimmed_path(path):
    """Put real interpreters first: HOME-relative shims (mise, asdf) stop resolving once HOME moves."""
    directories = []
    for tool, arguments in PROBES.items():
        found = shutil.which(tool, path=path)
        if not found:
            continue
        try:
            real = subprocess.run([found, *arguments], capture_output=True, text=True,
                                  timeout=120, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if real and Path(real).resolve() != Path(found).resolve():
            directories.append(str(Path(real).parent))
    return os.pathsep.join(directories + ([path] if path else []))


def install_web_dependencies(root, environment):
    """Run `npm ci` for the workbench when its packages differ from package-lock.json.

    The workbench tests import them (Preact, Pele), so a checkout that has never
    run `npm ci`, or ran it before the lockfile changed, would otherwise fail
    them with module-not-found errors. Without Node the web tests skip.
    """
    node = shutil.which("node", path=environment["PATH"])
    stale = stale_web_dependencies(WEB) if node else []
    if not stale:
        return 0
    where = WEB.relative_to(root).as_posix()
    more = f" and {len(stale) - 1} more" if len(stale) > 1 else ""
    print(f"check.py: {where}/node_modules differs from package-lock.json ({stale[0]}{more}); running npm ci",
          flush=True)
    npm = shutil.which("npm", path=environment["PATH"])
    if not npm:
        print(f"check.py: npm is not on PATH; install Node.js with npm, or run `npm ci` in {where}", file=sys.stderr)
        return 1
    code = subprocess.run([*npm_command(npm, node), "ci"], cwd=WEB, env=environment).returncode
    if code:
        print(f"check.py: npm ci failed in {where} (exit {code})", file=sys.stderr)
    return code


def test_skips(environment):
    """`--skip` filters for the cargo tests from MARKITAI_CHECK_SKIP (space-separated names)."""
    names = environment.get("MARKITAI_CHECK_SKIP", "").split()
    return ["--", *(argument for name in names for argument in ("--skip", name))] if names else []


def main():
    root = Path(__file__).resolve().parents[1]
    local = root / ".local"
    environment = isolated_env(local / "test-user-home", local / "test-home")
    environment["PATH"] = unshimmed_path(os.environ.get("PATH", ""))
    commands = [
        ["cargo", "fmt", "--all", "--check"],
        # Every test binary runs even after a failure, so one run reports them all.
        ["cargo", "test", "--workspace", "--locked", "--no-fail-fast", *test_skips(os.environ)],
        ["cargo", "clippy", "--workspace", "--all-targets", "--locked", "--", "-D", "warnings"],
        # The helper tests include the workbench's, which need its locked npm packages.
        lambda: install_web_dependencies(root, environment),
        [sys.executable, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py"],
    ]
    for command in commands:
        code = command() if callable(command) else subprocess.run(command, cwd=root, env=environment).returncode
        if code:
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
