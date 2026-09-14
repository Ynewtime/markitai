"""Informational notices persist across processes without persisting consent."""

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest

from markitai.notices import notify_once

SCRIPT = """
import sys
from pathlib import Path
Path.home = staticmethod(lambda: Path(sys.argv[1]))
from loguru import logger
logger.remove()
from markitai.fetch_consent import disclose_remote_use, resolve_remote_consent
from markitai.vision_consent import ensure_vlm_ocr_disclosed
if sys.argv[2] == "fetch":
    disclose_remote_use()
elif sys.argv[2] == "vision":
    ensure_vlm_ocr_disclosed(None)
else:
    assert not sys.stdin.isatty(), "ask test requires noninteractive stdin"
    print(resolve_remote_consent("ask"))
"""


def run_notice(home, mode):
    return subprocess.run(
        [sys.executable, "-c", SCRIPT, str(home), mode],
        # Windows NUL is a character device and reports isatty() == True.
        input="",
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )


@pytest.mark.parametrize("mode", ["fetch", "vision"])
def test_notice_is_one_short_line_across_fresh_processes(tmp_path, mode):
    first = run_notice(tmp_path, mode)
    second = run_notice(tmp_path, mode)
    assert first.stdout == second.stdout == ""
    assert len(first.stderr.splitlines()) == 1
    assert len(first.stderr.strip()) <= 80
    assert second.stderr == ""


def test_seen_notice_does_not_authorize_remote_calls(tmp_path):
    run_notice(tmp_path, "fetch")
    assert run_notice(tmp_path, "ask").stdout.strip() == "False"


def test_concurrent_processes_emit_once(tmp_path):
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", SCRIPT, str(tmp_path), "fetch"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    outputs = [p.communicate(timeout=20) for p in processes]
    assert all(p.returncode == 0 for p in processes)
    assert sum("Remote services" in err for _, err in outputs) == 1


def test_failed_delivery_does_not_record_seen_notice():
    with pytest.raises(RuntimeError, match="delivery"):
        notify_once(
            "test-failure", "message", Mock(side_effect=RuntimeError("delivery"))
        )
    notify = Mock()
    assert notify_once("test-failure", "message", notify) is True
    notify.assert_called_once_with("message")


def test_unwritable_notice_state_does_not_block_delivery():
    notify = Mock()
    with patch("markitai.notices.os.open", side_effect=PermissionError):
        assert notify_once("test-permissions", "message", notify) is True
    notify.assert_called_once_with("message")
