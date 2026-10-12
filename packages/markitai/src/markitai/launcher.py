"""The console scripts' entry point (``markitai``, ``mkai``, ``python -m markitai``).

It answers ``markitai --version`` before the CLI loads: the CLI imports about
430 modules (click, Rich, pydantic, the config models), which on Windows
costs about 2 ms each, and more under x64 emulation on Arm (runner-win:
1.9 s for ``--version``). Everything else goes to ``markitai.cli.main``.

``MARKITAI_TIMINGS=1`` prints where a run's time went to stderr when it
ends: the CLI's imports, the command, and the steps a command records
(``markitai doctor`` records each check). Standard library only, so it
costs nothing when off.
"""

from __future__ import annotations

import os
import sys
import time

_START = time.perf_counter()
_steps: list[tuple[str, float]] = []


def timings_enabled() -> bool:
    """Whether ``MARKITAI_TIMINGS`` asks for the timing report."""
    return os.environ.get("MARKITAI_TIMINGS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def record(name: str, seconds: float) -> None:
    """Note how long one step took, for the timing report."""
    _steps.append((name, seconds))


def report() -> None:
    """Print the timing report to stderr when ``MARKITAI_TIMINGS`` is set."""
    if not timings_enabled():
        return
    total = time.perf_counter() - _START
    lines = [f"markitai timings (since the entry point, {len(sys.modules)} modules):"]
    lines += [f"  {name:<34} {seconds:8.3f} s" for name, seconds in _steps]
    lines.append(f"  {'total':<34} {total:8.3f} s")
    try:
        sys.stderr.write("\n".join(lines) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def main() -> None:
    """Run markitai: ``--version`` at once, anything else through the CLI."""
    if sys.argv[1:] in (["--version"], ["-V"]):
        from markitai import __version__

        sys.stdout.write(f"markitai {__version__}\n")
        sys.stdout.flush()
        record("version", time.perf_counter() - _START)
        report()
        return
    from markitai.cli.main import main as cli_main

    record("imports (CLI)", time.perf_counter() - _START)
    cli_main()
