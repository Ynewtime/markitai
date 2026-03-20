"""Tests for terminal image protocol detection and rendering."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from markitai.utils.terminal_image import Protocol, detect_protocol, render_inline_image


class TestDetectProtocol:
    """detect_protocol() returns the best available terminal image protocol."""

    def test_returns_none_when_not_tty(self) -> None:
        with patch("sys.stdout") as mock_stdout:
            mock_stdout.isatty.return_value = False
            assert detect_protocol() is None

    def test_returns_kitty_when_kitty_pid_set(self) -> None:
        env = {"KITTY_PID": "12345", "TERM": "", "TERM_PROGRAM": ""}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.KITTY

    def test_returns_kitty_when_term_is_xterm_kitty(self) -> None:
        env = {"TERM": "xterm-kitty", "TERM_PROGRAM": ""}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.KITTY

    def test_returns_iterm2_when_term_program_is_iterm(self) -> None:
        env = {"TERM_PROGRAM": "iTerm.app", "TERM": "xterm-256color"}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.ITERM2

    def test_returns_iterm2_for_wezterm(self) -> None:
        env = {"TERM_PROGRAM": "WezTerm", "TERM": "xterm-256color"}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.ITERM2

    def test_returns_none_for_unknown_terminal(self) -> None:
        env = {"TERM": "xterm-256color", "TERM_PROGRAM": "unknown"}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() is None

    def test_kitty_takes_priority_over_iterm2(self) -> None:
        """When both Kitty and iTerm2 signals are present, prefer Kitty."""
        env = {"KITTY_PID": "12345", "TERM_PROGRAM": "WezTerm", "TERM": ""}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.KITTY


class TestRenderInlineImage:
    """render_inline_image() produces terminal escape sequences."""

    def test_kitty_output_starts_with_apc(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = render_inline_image(img, Protocol.KITTY)
        assert result.startswith("\033_G")

    def test_kitty_output_ends_with_st(self, tmp_path: Path) -> None:
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)

        result = render_inline_image(img, Protocol.KITTY)
        assert result.endswith("\033\\")

    def test_iterm2_output_contains_protocol_header(self, tmp_path: Path) -> None:
        img = tmp_path / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)

        result = render_inline_image(img, Protocol.ITERM2)
        assert "\033]1337;File=" in result
        assert "inline=1" in result

    def test_kitty_multi_chunk_for_large_image(self, tmp_path: Path) -> None:
        """Images larger than KITTY_CHUNK_SIZE should produce multiple chunks."""
        img = tmp_path / "large.png"
        # Create image larger than 4096 bytes of base64 (~3072 raw bytes)
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 4000)

        result = render_inline_image(img, Protocol.KITTY)
        # Should have intermediate chunks with m=1 and final with m=0
        assert "m=1;" in result
        assert "m=0;" in result

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.png"
        with pytest.raises(FileNotFoundError):
            render_inline_image(missing, Protocol.KITTY)
