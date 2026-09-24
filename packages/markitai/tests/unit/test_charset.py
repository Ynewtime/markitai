"""WHATWG-style charset resolution for fetched bodies (markitai.utils.charset)."""

from __future__ import annotations

import codecs

import pytest

from markitai.fetch_http import StaticHttpResponse
from markitai.utils.charset import (
    charset_from_content_type,
    decode_body,
    resolve_charset_label,
    sniff_meta_charset,
)


class TestResolveCharsetLabel:
    @pytest.mark.parametrize(
        ("label", "codec"),
        [
            ("gb2312", "gb18030"),
            ("GBK", "gb18030"),
            ("x-gbk", "gb18030"),
            ("Shift_JIS", "cp932"),
            ("sjis", "cp932"),
            ("iso-8859-1", "cp1252"),
            ("latin1", "cp1252"),
            ("US-ASCII", "cp1252"),
            ("ascii", "cp1252"),
            ("euc-kr", "cp949"),
            ("big5", "big5hkscs"),
            ("iso-8859-9", "cp1254"),
            ("tis-620", "cp874"),
            ("utf8", "utf-8"),
            (" 'UTF-8' ", "utf-8"),
            ("latin-1", "cp1252"),  # Python alias, widened via the codec name
            ("koi8-r", "koi8_r"),
        ],
    )
    def test_labels_widen_to_what_browsers_decode(self, label, codec):
        assert resolve_charset_label(label) == codec

    @pytest.mark.parametrize("label", ["", None, "no-such-charset", "iso-2022-kr"])
    def test_unknown_and_replacement_labels_are_ignored(self, label):
        assert resolve_charset_label(label) is None

    def test_content_type_parameter(self):
        assert charset_from_content_type('text/html; charset="GB2312"') == "gb18030"
        assert charset_from_content_type("text/html") is None
        assert charset_from_content_type(None) is None


class TestMetaPrescan:
    def test_meta_charset_attribute(self):
        assert sniff_meta_charset(b'<html><head><meta charset="gbk">') == "gb18030"

    def test_http_equiv_content_type(self):
        head = b'<meta http-equiv="Content-Type" content="text/html; charset=euc-kr">'
        assert sniff_meta_charset(head) == "cp949"

    def test_content_without_http_equiv_is_not_a_declaration(self):
        assert sniff_meta_charset(b'<meta content="text/html; charset=gbk">') is None

    def test_commented_out_meta_is_skipped(self):
        head = b'<!-- <meta charset="gbk"> --><meta charset="shift_jis">'
        assert sniff_meta_charset(head) == "cp932"

    def test_only_the_first_1024_bytes_count(self):
        head = b"<html>" + b" " * 2000 + b'<meta charset="gbk">'
        assert sniff_meta_charset(head) is None

    def test_utf16_declaration_in_document_means_utf8(self):
        assert sniff_meta_charset(b'<meta charset="utf-16">') == "utf-8"


class TestDecodeBody:
    def test_meta_charset_without_header_charset(self):
        """text/html + <meta charset=gbk> used to be decoded as UTF-8."""
        html = '<meta charset="gbk"><p>朱镕基讲话全文</p>'
        text, codec = decode_body(html.encode("gbk"), "text/html")
        assert "朱镕基讲话全文" in text
        assert codec == "gb18030"

    def test_gb2312_label_with_gbk_only_characters(self):
        """镕 is not in GB2312; strict gb2312 decoding failed the whole page."""
        text, _ = decode_body(
            "<p>朱镕基</p>".encode("gbk"), "text/html; charset=gb2312"
        )
        assert text == "<p>朱镕基</p>"

    def test_shift_jis_label_with_cp932_extensions(self):
        text, _ = decode_body("①髙橋".encode("cp932"), "text/html; charset=Shift_JIS")
        assert text == "①髙橋"

    def test_iso_8859_1_label_keeps_windows_1252_punctuation(self):
        body = "“quote” — €5 and it’s fine".encode("cp1252")
        text, _ = decode_body(body, "text/html; charset=iso-8859-1")
        assert text == "“quote” — €5 and it’s fine"

    def test_undefined_windows_1252_bytes_map_to_c1_controls(self):
        text, _ = decode_body(b"a\x81b", "text/plain; charset=windows-1252")
        assert text == "a\x81b"

    def test_bom_beats_header(self):
        text, codec = decode_body("﻿中文".encode(), "text/html; charset=gbk")
        assert text == "中文"
        assert codec == "utf-8"

    def test_header_beats_meta(self):
        body = '<meta charset="gbk"><p>中文</p>'.encode()
        text, codec = decode_body(body, "text/html; charset=utf-8")
        assert "中文" in text
        assert codec == "utf-8"

    def test_wrong_header_yields_to_matching_meta(self):
        body = '<meta charset="gbk"><p>中文内容</p>'.encode("gbk")
        text, codec = decode_body(body, "text/html; charset=utf-8")
        assert "中文内容" in text
        assert codec == "gb18030"

    def test_plain_text_is_not_prescanned_but_detected(self):
        body = "纯文本内容，中文测试一下这个检测器是否可以正确工作。".encode("gbk")
        text, _ = decode_body(body, "text/plain", html=False)
        assert "纯文本内容" in text

    def test_plain_text_header_charset(self):
        text, _ = decode_body("中文".encode("gbk"), "text/plain; charset=gbk")
        assert text == "中文"

    def test_utf8_without_any_declaration(self):
        text, codec = decode_body("<p>日本語</p>".encode(), None)
        assert text == "<p>日本語</p>"
        assert codec == "utf-8"


class TestDecodeBodyCorruption:
    """A few corrupt bytes must not throw the whole page to windows-1252."""

    PAGE = "中文内容测试，这是一个页面。" * 20

    def test_utf8_with_one_bad_byte_stays_utf8(self):
        body = f"<html><body>{self.PAGE}".encode() + b"\xff</body></html>"
        text, codec = decode_body(body, "text/html")
        assert codec == "utf-8"
        assert self.PAGE in text
        assert text.count("\ufffd") == 1

    def test_utf8_truncated_sequence_stays_utf8(self):
        body = f"<p>{self.PAGE}</p>".encode()
        body = body[:-10] + body[-9:]  # drop one byte of a multi-byte char
        text, codec = decode_body(body, None)
        assert codec == "utf-8"
        assert "中文内容测试" in text

    def test_undeclared_gbk_with_bad_byte_stays_gbk(self):
        body = f"<html><body>{self.PAGE}".encode("gbk") + b"\x80</body></html>"
        text, codec = decode_body(body, "text/html")
        assert codec == "gb18030"
        assert self.PAGE in text

    def test_undeclared_big5_with_bad_byte_stays_big5(self):
        page = "中文內容測試，這是一個頁面。" * 20
        text, codec = decode_body(page.encode("big5") + b"\x81", "text/html")
        assert codec == "big5hkscs"
        assert page in text

    def test_late_meta_with_bad_byte_uses_that_meta(self):
        body = (
            b"<html><head>"
            + b" " * 1100
            + b'<meta charset="gbk"></head><body>'
            + self.PAGE.encode("gbk")
            + b"\xff</body></html>"
        )
        text, codec = decode_body(body, "text/html")
        assert codec == "gb18030"
        assert self.PAGE in text

    def test_bom_beats_meta(self):
        body = codecs.BOM_UTF8 + '<meta charset="gbk"><p>中文</p>'.encode()
        text, codec = decode_body(body, "text/html")
        assert codec == "utf-8"
        assert "<p>中文</p>" in text

    def test_utf16_bom_beats_meta(self):
        body = codecs.BOM_UTF16_LE + '<meta charset="gbk"><p>中文</p>'.encode(
            "utf-16-le"
        )
        text, codec = decode_body(body, "text/html")
        assert codec == "utf-16-le"
        assert "<p>中文</p>" in text


class TestDecodeBodyDetection:
    def test_short_latin1_text_is_not_utf16(self):
        text, codec = decode_body("Résumé".encode("latin-1"), "text/plain", html=False)
        assert text == "Résumé"
        assert codec == "cp1252"

    def test_long_latin1_text_is_not_utf16(self):
        body = ("Résumé " * 100).encode("latin-1")
        text, _ = decode_body(body, "text/plain", html=False)
        assert text == "Résumé " * 100

    def test_short_latin1_csv_keeps_windows_1252(self):
        csv = "name,city\nJoão,São Paulo\nJosé,Brasília\n"
        text, codec = decode_body(csv.encode("latin-1"), "text/csv", html=False)
        assert text == csv
        assert codec == "cp1252"

    def test_long_latin1_csv_keeps_windows_1252(self):
        csv = "name,city\n" + "João,São Paulo\nJosé,Brasília\nAna,Maceió\n" * 30
        text, codec = decode_body(csv.encode("latin-1"), "text/csv", html=False)
        assert text == csv
        assert codec == "cp1252"

    def test_french_text_keeps_windows_1252(self):
        body = "Le café est très bon. Déjà vu, à la façon de Noël. " * 20
        text, _ = decode_body(body.encode("cp1252"), "text/plain", html=False)
        assert text == body

    def test_polish_windows_1250_is_still_detected(self):
        body = (
            "Zażółć gęślą jaźń, może łódź. Ślę pozdrowienia z Łodzi. W Polsce "
            "mówi się po polsku, a w Czechach po czesku. Pięćdziesiąt złotych. "
        ) * 10
        text, codec = decode_body(body.encode("cp1250"), "text/plain", html=False)
        assert codec == "cp1250"
        assert text == body

    def test_cyrillic_windows_1251_is_still_detected(self):
        body = "Привет, это тестовая страница на русском языке. " * 10
        text, codec = decode_body(body.encode("cp1251"), "text/html")
        assert codec == "cp1251"
        assert text == body


class TestStaticHttpResponse:
    def test_encoding_is_widened(self):
        response = StaticHttpResponse(
            content=b"",
            status_code=200,
            headers={"content-type": "text/html; charset=gb2312"},
            url="https://example.cn",
        )
        assert response.encoding == "gb18030"

    def test_text_uses_meta_prescan(self):
        response = StaticHttpResponse(
            content='<meta charset="shift_jis"><p>①髙橋</p>'.encode("cp932"),
            status_code=200,
            headers={"content-type": "text/html"},
            url="https://example.jp",
        )
        assert "①髙橋" in response.text
