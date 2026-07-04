"""Tests for the native EML email converter."""

from __future__ import annotations

import io
from email.message import EmailMessage
from pathlib import Path

from PIL import Image

from markitai.converter import FileFormat, detect_format, get_converter
from markitai.converter.eml import EmlConverter, _sanitize_header
from markitai.image import ImageProcessor


def _png_bytes(size: tuple[int, int] = (32, 32), color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _write_eml(tmp_path: Path, msg: EmailMessage, name: str = "mail.eml") -> Path:
    path = tmp_path / name
    path.write_bytes(bytes(msg))
    return path


def _base_message(**headers: str) -> EmailMessage:
    msg = EmailMessage()
    defaults = {
        "From": "Alice <alice@example.com>",
        "To": "Bob <bob@example.com>",
        "Date": "Fri, 03 Jul 2026 10:00:00 +0000",
        "Subject": "Hello",
    }
    defaults.update(headers)
    for key, value in defaults.items():
        msg[key] = value
    return msg


class TestEmlRegistration:
    def test_detect_format(self) -> None:
        assert detect_format("message.eml") == FileFormat.EML

    def test_get_converter_returns_native_eml_converter(self, tmp_path: Path) -> None:
        msg = _base_message()
        msg.set_content("hi")
        path = _write_eml(tmp_path, msg)
        converter = get_converter(path)
        assert isinstance(converter, EmlConverter)


class TestEmlSimpleText:
    def test_plain_text_email(self, tmp_path: Path) -> None:
        msg = _base_message(Subject="Weekly sync notes")
        msg.set_content("First paragraph.\n\nSecond paragraph.")
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        assert result.markdown.startswith("# Email Message")
        assert "**From:** Alice \\<alice@example.com\\>" in result.markdown
        assert "**To:** Bob \\<bob@example.com\\>" in result.markdown
        assert "**Date:** Fri, 03 Jul 2026 10:00:00 +0000" in result.markdown
        assert "**Subject:** Weekly sync notes" in result.markdown
        assert "## Content" in result.markdown
        assert "First paragraph." in result.markdown
        assert "Second paragraph." in result.markdown
        assert result.metadata["converter"] == "eml"
        assert result.metadata["title"] == "Weekly sync notes"

    def test_cc_header_rendered_when_present(self, tmp_path: Path) -> None:
        msg = _base_message(Cc="Carol <carol@example.com>")
        msg.set_content("body")
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        assert "**Cc:** Carol \\<carol@example.com\\>" in result.markdown

    def test_email_without_body(self, tmp_path: Path) -> None:
        msg = _base_message()
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        assert "## Content" in result.markdown


class TestEmlHtmlBody:
    def test_html_body_converted_to_markdown(self, tmp_path: Path) -> None:
        msg = _base_message(Subject="HTML mail")
        msg.set_content("Plain fallback text.")
        msg.add_alternative(
            "<html><body><h1>Report</h1>"
            "<p>Revenue grew <b>12%</b> quarter over quarter.</p>"
            "<ul><li>Alpha</li><li>Beta</li></ul></body></html>",
            subtype="html",
        )
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        # HTML part preferred over plain text
        assert "Plain fallback text." not in result.markdown
        assert "Revenue grew" in result.markdown
        assert "**12%**" in result.markdown
        assert "Alpha" in result.markdown
        # No raw HTML tags leaked into the body
        assert "<body>" not in result.markdown
        assert "<ul>" not in result.markdown

    def test_plain_fallback_when_no_html_part(self, tmp_path: Path) -> None:
        msg = _base_message()
        msg.set_content("Only plain text here.")
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        assert "Only plain text here." in result.markdown


class TestEmlAttachments:
    def test_image_attachment_flows_into_base64_pipeline(self, tmp_path: Path) -> None:
        png = _png_bytes()
        msg = _base_message()
        msg.set_content("See attached chart.")
        msg.add_attachment(png, maintype="image", subtype="png", filename="chart.png")
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        assert "## Attachments" in result.markdown
        assert "data:image/png;base64," in result.markdown
        # The data URI must match the standard extraction pipeline so
        # workflow saves it to assets and vision/alt analysis picks it up.
        extracted = ImageProcessor().extract_base64_images(result.markdown)
        assert len(extracted) == 1
        alt_text, mime_type, data = extracted[0]
        assert alt_text == "chart.png"
        assert mime_type == "image/png"
        assert data == png

    def test_non_image_attachment_listed_with_name_and_size(
        self, tmp_path: Path
    ) -> None:
        msg = _base_message()
        msg.set_content("body")
        msg.add_attachment(
            b"x" * 2048,
            maintype="application",
            subtype="pdf",
            filename="report.pdf",
        )
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        assert "## Attachments" in result.markdown
        assert "- report.pdf (2.0 KB)" in result.markdown
        assert "data:" not in result.markdown


class TestEmlNestedMessage:
    def test_nested_rfc822_rendered_as_quoted_section(self, tmp_path: Path) -> None:
        inner = _base_message(
            **{"From": "Carol <carol@example.com>", "Subject": "Nested hello"}
        )
        inner.set_content("Nested body text.")

        outer = _base_message(Subject="Fwd: Nested hello")
        outer.set_content("Forwarding this.")
        outer.add_attachment(inner)
        path = _write_eml(tmp_path, outer)

        result = EmlConverter().convert(path)

        assert "### Attached message" in result.markdown
        assert "> **From:** Carol \\<carol@example.com\\>" in result.markdown
        assert "> Nested body text." in result.markdown

    def test_nested_recursion_capped_at_one_level(self, tmp_path: Path) -> None:
        # innermost message with an image attachment — at depth 1 its
        # attachments must be listed by name only (no data URIs, no deeper
        # quoted sections).
        innermost = _base_message(Subject="Level 2")
        innermost.set_content("Deepest body.")

        inner = _base_message(Subject="Level 1")
        inner.set_content("Middle body.")
        inner.add_attachment(
            _png_bytes(), maintype="image", subtype="png", filename="deep.png"
        )
        inner.add_attachment(innermost)

        outer = _base_message(Subject="Level 0")
        outer.set_content("Top body.")
        outer.add_attachment(inner)
        path = _write_eml(tmp_path, outer)

        result = EmlConverter().convert(path)

        assert "> Middle body." in result.markdown
        # Depth cap: nested message's attachments are names only
        assert "deep.png" in result.markdown
        assert "data:image/png" not in result.markdown
        assert "Deepest body." not in result.markdown


class TestEmlCharsets:
    def test_gb2312_body_decoded(self, tmp_path: Path) -> None:
        body = "你好，世界"  # 你好,世界
        payload = body.encode("gb2312")
        import base64 as b64

        raw = (
            b"From: zhang@example.com\r\n"
            b"To: li@example.com\r\n"
            b"Subject: =?gb2312?B?" + b64.b64encode("问候".encode("gb2312")) + b"?=\r\n"
            b"Content-Type: text/plain; charset=gb2312\r\n"
            b"Content-Transfer-Encoding: base64\r\n"
            b"\r\n" + b64.b64encode(payload) + b"\r\n"
        )
        path = tmp_path / "gb2312.eml"
        path.write_bytes(raw)

        result = EmlConverter().convert(path)

        assert body in result.markdown
        assert "**Subject:** 问候" in result.markdown

    def test_unknown_charset_falls_back_to_replacement(self, tmp_path: Path) -> None:
        raw = (
            b"From: a@example.com\r\n"
            b"Subject: odd charset\r\n"
            b"Content-Type: text/plain; charset=x-no-such-charset\r\n"
            b"\r\n"
            b"hello bytes\r\n"
        )
        path = tmp_path / "weird.eml"
        path.write_bytes(raw)

        result = EmlConverter().convert(path)

        assert "hello bytes" in result.markdown


class TestEmlHeaderSanitization:
    def test_sanitize_header_escapes_html(self) -> None:
        assert (
            _sanitize_header("<script>alert(1)</script>")
            == "\\<script\\>alert(1)\\</script\\>"
        )

    def test_sanitize_header_strips_control_chars(self) -> None:
        assert _sanitize_header("line1\r\nInjected: header") == "line1 Injected: header"

    def test_no_raw_html_passthrough_in_headers(self, tmp_path: Path) -> None:
        msg = _base_message(Subject='Sale <img src=x onerror="alert(1)"> now')
        msg.set_content("body")
        path = _write_eml(tmp_path, msg)

        result = EmlConverter().convert(path)

        # Every angle bracket in the rendered header is backslash-escaped;
        # no unescaped <img ...> tag survives.
        import re

        assert re.search(r"(?<!\\)<img", result.markdown) is None
        assert "\\<img src=x" in result.markdown
