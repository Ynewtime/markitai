"""``markitai-mcp`` without the mcp extra says how to install it."""

from __future__ import annotations

import subprocess
import sys


def test_the_script_explains_the_missing_extra_instead_of_a_traceback() -> None:
    code = (
        "import sys; sys.modules['mcp'] = None; from markitai.mcp import main; main()"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60
    )
    assert done.returncode == 1
    assert "needs the mcp extra" in done.stderr
    assert "markitai[mcp]" in done.stderr
    assert "Traceback" not in done.stderr
