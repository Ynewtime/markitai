"""Concurrent first use of markitdown must not hit an import deadlock.

``markitdown`` and ``markitdown.converters`` import each other; threads that
first touch them through different doors used to fail with
``_DeadlockError: deadlock detected by _ModuleLock('markitdown.converters')``
(seen in one of five 36-file batch runs). The race needs a fresh interpreter
(an already imported package cannot deadlock) and its timing varies, so the
test starts several interpreters; unguarded, the same script fails nearly
every time (verified while writing it).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

_SCRIPT = textwrap.dedent(
    """
    import sys, threading
    from pathlib import Path

    from markitai.converter import markitdown_ext
    from markitai.converter._patches import apply_markitdown_cell_patches

    csv = Path(sys.argv[1])

    def door_converter():
        markitdown_ext._convert(csv)

    def door_patch():
        apply_markitdown_cell_patches()

    def door_epub_chapter():
        try:
            markitdown_ext._epub_chapter_markdown(b"<p>x</p>", "c.xhtml", None)
        except ImportError:
            raise
        except Exception:
            pass  # only the import matters here

    doors = [door_converter, door_patch, door_epub_chapter] * 3
    errors = []

    def run(door):
        try:
            door()
        except BaseException as exc:
            errors.append(repr(exc))

    threads = [threading.Thread(target=run, args=(d,)) for d in doors]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("\\n".join(errors))
    sys.exit(1 if errors else 0)
    """
)


@pytest.mark.parametrize("attempt", range(6))
def test_threads_first_importing_markitdown_do_not_deadlock(
    tmp_path: Path, attempt: int
) -> None:
    pytest.importorskip("markitdown")
    sample = tmp_path / "t.csv"
    sample.write_text("a,b\n1,2\n", encoding="utf-8")
    done = subprocess.run(
        [sys.executable, "-I", "-c", _SCRIPT, str(sample)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stdout + done.stderr
