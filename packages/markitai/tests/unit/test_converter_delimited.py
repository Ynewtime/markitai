"""Tests for the native TSV converter."""

from __future__ import annotations

from pathlib import Path

from markitai.converter.delimited import TsvConverter

FIXTURES = Path(__file__).parent.parent / "fixtures"


class TestTsvConverter:
    def test_fixture_becomes_a_markdown_table(self) -> None:
        result = TsvConverter().convert(FIXTURES / "sample.tsv")
        lines = result.markdown.splitlines()

        assert lines[0].startswith("| Employee_ID | First_Name |")
        assert lines[1] == "| " + " | ".join(["---"] * 12) + " |"
        assert "| E001 | James | Anderson |" in lines[2]
        assert result.metadata["format"] == "TSV"
        assert result.images == []

    def test_quoted_tab_stays_inside_one_cell(self, tmp_path: Path) -> None:
        path = tmp_path / "quoted.tsv"
        path.write_text('a\tb\n"one\ttwo"\tthree\n', encoding="utf-8")

        markdown = TsvConverter().convert(path).markdown

        assert markdown.splitlines()[2] == "| one\ttwo | three |"

    def test_pipes_are_escaped_and_rows_padded(self, tmp_path: Path) -> None:
        path = tmp_path / "pipes.tsv"
        path.write_text("a\tb\tc\nx|y\tz\n", encoding="utf-8")

        lines = TsvConverter().convert(path).markdown.splitlines()

        assert lines[2] == "| x\\|y | z |  |"

    def test_comma_file_without_tabs_is_sniffed(self, tmp_path: Path) -> None:
        path = tmp_path / "commas.tsv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")

        lines = TsvConverter().convert(path).markdown.splitlines()

        assert lines[0] == "| a | b |"
        assert lines[2] == "| 1 | 2 |"

    def test_blank_rows_are_dropped(self, tmp_path: Path) -> None:
        path = tmp_path / "blanks.tsv"
        path.write_text("a\tb\n\n1\t2\n\n", encoding="utf-8")

        assert len(TsvConverter().convert(path).markdown.splitlines()) == 3

    def test_empty_file_yields_empty_markdown(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.tsv"
        path.write_text("", encoding="utf-8")

        assert TsvConverter().convert(path).markdown == ""

    def test_unpaired_quote_does_not_raise(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.tsv"
        path.write_bytes(b'a\tb\n"unclosed\tvalue\n')

        result = TsvConverter().convert(path)

        assert isinstance(result.markdown, str)
        assert "error" not in result.metadata or result.markdown.startswith(">")

    def test_quoted_multiline_cell_keeps_its_line_break(self, tmp_path: Path) -> None:
        path = tmp_path / "multiline.tsv"
        path.write_text(
            'name\tcommand\nbuild\t"echo a\necho b"\nlast\tone\n', encoding="utf-8"
        )

        lines = TsvConverter().convert(path).markdown.splitlines()

        assert lines[2] == "| build | echo a<br>echo b |"
        assert lines[3] == "| last | one |"

    def test_crlf_multiline_cell_uses_br_without_carriage_return(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "crlf.tsv"
        path.write_bytes(b'a\tb\r\n"x\r\ny"\tz\r\n')

        lines = TsvConverter().convert(path).markdown.splitlines()

        assert lines[0] == "| a | b |"
        assert lines[2] == "| x<br>y | z |"

    def test_utf8_bom_does_not_stick_to_first_header(self, tmp_path: Path) -> None:
        path = tmp_path / "bom.tsv"
        path.write_bytes("﻿a\tb\n1\t2\n".encode())

        assert TsvConverter().convert(path).markdown.splitlines()[0] == "| a | b |"

    def test_unpaired_quote_spoils_one_row_not_the_rest(self, tmp_path: Path) -> None:
        """Stream parsing let one stray quote swallow every later row."""
        path = tmp_path / "stray.tsv"
        path.write_text(
            'name\tnote\nalpha\t"unclosed note\nbeta\ttwo\ngamma\tthree\n',
            encoding="utf-8",
        )

        lines = TsvConverter().convert(path).markdown.splitlines()

        assert lines[0] == "| name | note |"
        assert lines[2] == "| alpha | unclosed note |"
        assert lines[3:] == ["| beta | two |", "| gamma | three |"]
