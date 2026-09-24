"""Tests for the Outlook .msg converter's body recovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from markitai.converter import markitdown_ext
from markitai.converter.base import ConvertResult
from markitai.converter.markitdown_ext import MsgConverter
from markitai.notices import is_user_notice

FIXTURES = Path(__file__).parent.parent / "fixtures"

_HEADERS_ONLY = "# Email Message\n\n**Subject:** Hi\n\n## Content"


def _properties(**codepages: int) -> bytes:
    """A top-level MAPI property stream holding the given PT_LONG values."""
    tags = {"message": 0x3FFD0003, "internet": 0x3FDE0003}
    data = bytearray(32)
    for name, value in codepages.items():
        data += tags[name].to_bytes(4, "little") + bytes(4)
        data += value.to_bytes(4, "little") + bytes(4)
    return bytes(data)


class _FakeOle:
    """Stand-in for olefile.OleFileIO over an in-memory stream table."""

    streams: dict[str, bytes] = {}

    def __init__(self, _path: str) -> None:
        pass

    def __enter__(self) -> _FakeOle:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def exists(self, name: str) -> bool:
        return name in self.streams

    def openstream(self, name: str) -> Any:
        import io

        return io.BytesIO(self.streams[name])


@pytest.fixture
def fake_msg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Convert a .msg whose streams the test supplies; markitdown is stubbed."""
    import olefile

    path = tmp_path / "mail.msg"
    path.write_bytes(b"")
    monkeypatch.setattr(olefile, "OleFileIO", _FakeOle)
    monkeypatch.setattr(
        markitdown_ext, "_convert", lambda _p: ConvertResult(markdown=_HEADERS_ONLY)
    )

    def run(streams: dict[str, bytes]) -> tuple[str, list[str]]:
        _FakeOle.streams = streams
        records: list[Any] = []
        sink_id = logger.add(lambda m: records.append(m.record), level="WARNING")
        try:
            markdown = MsgConverter().convert(path).markdown
        finally:
            logger.remove(sink_id)
        return markdown, [r["message"] for r in records if is_user_notice(r)]

    return run


class TestMsgBodyRecovery:
    def test_html_only_fixture_keeps_its_body(self) -> None:
        markdown = MsgConverter().convert(FIXTURES / "sample.msg").markdown

        content = markdown.split("## Content", 1)[1]
        assert "This is a message" in content

    def test_ansi_body_uses_the_message_codepage(self, fake_msg) -> None:
        markdown, notices = fake_msg(
            {
                "__properties_version1.0": _properties(message=1251),
                "__substg1.0_1000001E": "Привет".encode("cp1251") + b"\x00",
            }
        )

        assert markdown == f"{_HEADERS_ONLY}\n\nПривет"
        assert notices == []

    def test_html_body_is_converted_to_markdown(self, fake_msg) -> None:
        html = "<html><body><p>Hello <b>world</b></p></body></html>"
        markdown, notices = fake_msg(
            {
                "__properties_version1.0": _properties(internet=65001),
                "__substg1.0_10130102": html.encode(),
            }
        )

        body = markdown.split("## Content", 1)[1]
        assert "Hello **world**" in body
        assert "<b>" not in body
        assert notices == []

    def test_unicode_body_is_left_to_markitdown(self, fake_msg) -> None:
        markdown, notices = fake_msg(
            {
                "__substg1.0_1000001F": "Body".encode("utf-16-le"),
                "__substg1.0_10130102": b"<p>ignored</p>",
            }
        )

        assert markdown == _HEADERS_ONLY
        assert notices == []

    def test_no_body_raises_a_user_notice(self, fake_msg) -> None:
        markdown, notices = fake_msg({"__substg1.0_1000001F": b""})

        assert markdown == _HEADERS_ONLY
        assert len(notices) == 1
        assert "No readable body in mail.msg" in notices[0]
        assert "no plain-text or HTML body" in notices[0]

    def test_rtf_only_body_names_the_reason(self, fake_msg) -> None:
        _, notices = fake_msg({"__substg1.0_10090102": b"LZFu..."})

        assert len(notices) == 1
        assert "RTF" in notices[0]
