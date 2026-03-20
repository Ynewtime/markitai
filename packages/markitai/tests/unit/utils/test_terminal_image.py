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

    def test_returns_kitty_for_ghostty_via_term(self) -> None:
        env = {"TERM": "xterm-ghostty", "TERM_PROGRAM": "ghostty"}
        with (
            patch("sys.stdout") as mock_stdout,
            patch.dict("os.environ", env, clear=True),
        ):
            mock_stdout.isatty.return_value = True
            assert detect_protocol() == Protocol.KITTY

    def test_returns_kitty_for_ghostty_via_term_program(self) -> None:
        env = {"TERM": "xterm-256color", "TERM_PROGRAM": "ghostty"}
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

    def _make_png(
        self, tmp_path: Path, name: str = "test.png", size: int = 100
    ) -> Path:
        """Create a minimal PNG file for testing."""
        import io

        from PIL import Image

        img = Image.new("RGB", (size, size), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        path = tmp_path / name
        path.write_bytes(buf.getvalue())
        return path

    def _make_jpeg(
        self, tmp_path: Path, name: str = "test.jpg", size: int = 100
    ) -> Path:
        """Create a minimal JPEG file for testing."""
        import io

        from PIL import Image

        img = Image.new("RGB", (size, size), color="blue")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        path = tmp_path / name
        path.write_bytes(buf.getvalue())
        return path

    def test_kitty_output_starts_with_apc(self, tmp_path: Path) -> None:
        img = self._make_png(tmp_path)
        result = render_inline_image(img, Protocol.KITTY)
        assert result.startswith("\033_G")

    def test_kitty_output_ends_with_st(self, tmp_path: Path) -> None:
        img = self._make_png(tmp_path)
        result = render_inline_image(img, Protocol.KITTY)
        assert result.endswith("\033\\")

    def test_kitty_uses_f100_png_format(self, tmp_path: Path) -> None:
        img = self._make_png(tmp_path)
        result = render_inline_image(img, Protocol.KITTY)
        assert "f=100" in result

    def test_kitty_converts_jpeg_to_png(self, tmp_path: Path) -> None:
        """JPEG input should be converted to PNG for Kitty protocol."""
        img = self._make_jpeg(tmp_path)
        result = render_inline_image(img, Protocol.KITTY)
        assert result.startswith("\033_G")
        assert "f=100" in result

    def test_iterm2_output_contains_protocol_header(self, tmp_path: Path) -> None:
        img = self._make_png(tmp_path)
        result = render_inline_image(img, Protocol.ITERM2)
        assert "\033]1337;File=" in result
        assert "inline=1" in result

    def test_iterm2_converts_jpeg_to_png(self, tmp_path: Path) -> None:
        """JPEG should be converted to PNG for iTerm2 cross-terminal compat."""
        img = self._make_jpeg(tmp_path)
        result = render_inline_image(img, Protocol.ITERM2)
        assert "\033]1337;File=" in result

    def test_kitty_multi_chunk_for_large_image(self, tmp_path: Path) -> None:
        """Images producing >4096 bytes of base64 should use multiple chunks."""
        import io

        # Random noise image — solid colors compress too small
        import os

        from PIL import Image

        img = Image.frombytes("RGB", (200, 200), os.urandom(200 * 200 * 3))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        path = tmp_path / "noisy.png"
        path.write_bytes(buf.getvalue())

        result = render_inline_image(path, Protocol.KITTY)
        assert "m=1;" in result
        assert "m=0;" in result

    def test_large_image_resized(self, tmp_path: Path) -> None:
        """Images wider than MAX_INLINE_WIDTH should be resized down."""
        from markitai.utils.terminal_image import MAX_INLINE_WIDTH

        # Create a wide image
        wide_img = self._make_png(tmp_path, name="wide.png", size=2000)

        # Render and compare — resized image should produce less data
        small_img = self._make_png(tmp_path, name="small.png", size=MAX_INLINE_WIDTH)

        result_wide = render_inline_image(wide_img, Protocol.KITTY)
        result_small = render_inline_image(small_img, Protocol.KITTY)

        # Wide image (2000px) when resized to 800px should produce
        # similar or smaller output than an 800px image
        assert len(result_wide) <= len(result_small) * 1.5

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "nonexistent.png"
        with pytest.raises(FileNotFoundError):
            render_inline_image(missing, Protocol.KITTY)
