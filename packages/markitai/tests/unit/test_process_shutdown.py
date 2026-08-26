"""The CLI's exit status must survive its own dependencies' teardown.

markitdown pulls Magika, which pulls onnxruntime, into every markitai
process — a plain `.txt` conversion loads it too. Under load onnxruntime's
C++ statics abort at interpreter shutdown ("recursive_mutex lock failed"),
which turned a conversion that had already written its output into exit code
134, and skipped the atexit handler that removes markitai's tracked temp
directories. ``finalize_process`` ends the run before that teardown.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


class TestFinalizeProcess:
    def test_reports_the_requested_status(self) -> None:
        for code in (0, 1, 10):
            proc = _run(f"""
                from markitai.utils.shutdown import finalize_process
                finalize_process({code})
            """)
            assert proc.returncode == code

    def test_flushes_buffered_output_before_leaving(self) -> None:
        """os._exit skips the flush the normal exit path would have done."""
        proc = _run("""
            import sys
            from markitai.utils.shutdown import finalize_process

            sys.stdout.write("buffered")  # no newline, no flush
            finalize_process(0)
        """)
        assert proc.stdout == "buffered"

    def test_removes_tracked_temp_dirs(self) -> None:
        """The atexit handler never runs, so finalize must do this itself."""
        proc = _run("""
            from markitai.utils.paths import create_tracked_temp_dir
            from markitai.utils.shutdown import finalize_process

            temp_dir = create_tracked_temp_dir()
            print(temp_dir, flush=True)
            finalize_process(0)
        """)
        assert proc.returncode == 0
        assert not Path(proc.stdout.strip()).exists()


class TestTrackedTempDirs:
    def test_cleanup_is_idempotent(self, tmp_path: Path) -> None:
        """finalize_process calls it, and atexit may call it again."""
        from markitai.utils.paths import (
            cleanup_tracked_temp_dirs,
            create_tracked_temp_dir,
        )

        temp_dir = create_tracked_temp_dir()
        assert temp_dir.is_dir()

        cleanup_tracked_temp_dirs()
        cleanup_tracked_temp_dirs()

        assert not temp_dir.exists()


def test_console_scripts_go_through_the_finalizing_entry_point() -> None:
    """`markitai.cli:app` alone would exit through interpreter shutdown."""
    scripts = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))["project"][
        "scripts"
    ]
    assert scripts["markitai"] == "markitai.cli:main"
    assert scripts["mkai"] == "markitai.cli:main"
