"""The wheel's `markitai`, `mkai` and `markitai-mcp` commands.

Each runs the native command-line program, the one the `markitai`
executable runs, in this process, which then ends with the program's exit
status: nothing of the interpreter runs after it.
"""

import signal
import sys

from . import _native


def _run(name: str) -> None:
    # The program handles interrupts itself, as the executable does; the
    # interpreter's handlers would only set a flag nobody reads.
    for signum in ("SIGINT", "SIGBREAK"):
        if hasattr(signal, signum):
            signal.signal(getattr(signal, signum), signal.SIG_DFL)
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.flush()
    # The program name selects the command as the executables' names do:
    # `markitai-mcp` is the MCP server.
    _native.run_cli([name, *sys.argv[1:]])


def markitai() -> None:
    """`markitai`."""
    _run("markitai")


def mkai() -> None:
    """`mkai`, the short name of `markitai`."""
    _run("mkai")


def markitai_mcp() -> None:
    """`markitai-mcp`, the MCP server (`markitai mcp`)."""
    _run("markitai-mcp")
