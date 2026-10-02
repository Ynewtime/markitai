"""Run the native development gate without requiring a Unix shell."""
from pathlib import Path
import os
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    state = root / ".local" / "test-home"
    state.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, MARKITAI_HOME=str(state))
    commands = [
        ["cargo", "fmt", "--all", "--check"],
        # Every test binary runs even after a failure, so one run reports them all.
        ["cargo", "test", "--workspace", "--locked", "--no-fail-fast"],
        ["cargo", "clippy", "--workspace", "--all-targets", "--locked", "--", "-D", "warnings"],
        [sys.executable, "-m", "unittest", "discover", "-s", "scripts", "-p", "test_*.py"],
    ]
    for command in commands:
        result = subprocess.run(command, cwd=root, env=environment)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main())
