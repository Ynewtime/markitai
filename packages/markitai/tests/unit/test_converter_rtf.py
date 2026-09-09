"""Tests for the native RTF converter."""

from __future__ import annotations

from pathlib import Path

import pytest

from markitai.converter.base import FileFormat, load_converter_class
from markitai.converter.rtf import RtfConverter
from markitai.utils.errors import ConversionError

FIXTURES = Path(__file__).parent.parent / "fixtures"

_PREAMBLE = rb"{\rtf1\ansi\ansicpg1252\deff0{\fonttbl{\f0\fswiss Helvetica;}}"


def _convert(tmp_path: Path, body: bytes, *, preamble: bytes = _PREAMBLE) -> str:
    """Write ``preamble + body + '}'`` as an .rtf file and convert it."""
    path = tmp_path / "sample.rtf"
    path.write_bytes(preamble + body + b"}")
    return RtfConverter().convert(path).markdown


class TestRtfFixture:
    def test_heading_hierarchy_matches_the_document_outline(self) -> None:
        markdown = RtfConverter().convert(FIXTURES / "sample.rtf").markdown
        headings = [line for line in markdown.splitlines() if line.startswith("#")]

        assert headings == [
            "# Comprehensive Extraction Test Document",
            "## First Section",
            "### Subsection 1.1",
            "## Second Section",
            "### Table Section",
            "### Another Table",
            "## Third Section",
            "### Subsection 3.1",
            "## Final Section",
        ]

    def test_paragraphs_and_tables_survive(self) -> None:
        markdown = RtfConverter().convert(FIXTURES / "sample.rtf").markdown

        assert "This is a regular text paragraph in the first section." in markdown
        assert "| Header 1 | Header 2 | Header 3 |" in markdown
        assert "| --- | --- | --- |" in markdown
        assert "| Cell 3A | Cell 3B | Cell 3C |" in markdown
        assert "| Orange | $0.75 | 15 |" in markdown
        assert markdown.endswith(
            "The extraction should capture everything listed here completely.\n"
        )

    def test_metadata_names_the_converter_and_the_first_heading(self) -> None:
        result = RtfConverter().convert(FIXTURES / "sample.rtf")

        assert result.metadata["format"] == "RTF"
        assert result.metadata["converter"] == "rtf"
        assert result.metadata["title"] == "Comprehensive Extraction Test Document"
        assert result.images == []

    def test_the_registry_reaches_this_converter(self) -> None:
        assert load_converter_class(FileFormat.RTF) is RtfConverter


class TestLists:
    def test_word_97_pntext_bullets(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\pard\plain Shopping list:\par}
{\pard\plain{\*\pn\pnlvlblt\pnf1\pnindent360{\pntxtb\'b7}}
{\pntext\f1\'b7\tab}Milk\par}
{\pard\plain{\*\pn\pnlvlblt\pnf1\pnindent360{\pntxtb\'b7}}
{\pntext\f1\'b7\tab}Bread\par}
{\pard\plain Done.\par}
""",
        )

        assert markdown == "Shopping list:\n\n- Milk\n- Bread\n\nDone.\n"

    def test_pntext_numbers_keep_their_own_numbering(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\pard\plain{\*\pn\pnlvlbody\pndec\pnstart1}{\pntext\f0 3.\tab}Third\par}
{\pard\plain{\*\pn\pnlvlbody\pndec}{\pntext\f0 4.\tab}Fourth\par}
""",
        )

        assert markdown == "3. Third\n4. Fourth\n"

    def test_ls_numbered_list_nests_by_ilvl(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\*\listtable{\list\listtemplateid1
{\listlevel\levelnfc0\levelstartat1{\leveltext\'02\'00.;}}
{\listlevel\levelnfc0{\leveltext\'02\'01.;}}\listid101}}
{\*\listoverridetable{\listoverride\listid101\listoverridecount0\ls1}}
{\pard\plain\ls1\ilvl0 Alpha\par}
{\pard\plain\ls1\ilvl1 Alpha one\par}
{\pard\plain\ls1\ilvl1 Alpha two\par}
{\pard\plain\ls1\ilvl0 Beta\par}
""",
        )

        assert markdown == ("1. Alpha\n  1. Alpha one\n  2. Alpha two\n2. Beta\n")

    def test_levelnfc23_is_a_bullet_list(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\*\listtable{\list{\listlevel\levelnfc23{\leveltext\'01\u183 ?;}}\listid7}}
{\*\listoverridetable{\listoverride\listid7\listoverridecount0\ls2}}
{\pard\plain\ls2\ilvl0 One\par}
{\pard\plain\ls2\ilvl0 Two\par}
""",
        )

        assert markdown == "- One\n- Two\n"


class TestTables:
    def test_two_by_three_table(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
\trowd\cellx2000\cellx4000\cellx6000
{\pard\intbl A\cell}{\pard\intbl B\cell}{\pard\intbl C\cell}\row
\trowd\cellx2000\cellx4000\cellx6000
{\pard\intbl 1\cell}{\pard\intbl 2\cell}{\pard\intbl 3\cell}\row
{\pard After the table.\par}
""",
        )

        assert markdown == (
            "| A | B | C |\n| --- | --- | --- |\n| 1 | 2 | 3 |\n\nAfter the table.\n"
        )

    def test_pipes_in_a_cell_are_escaped(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
\trowd\cellx2000\cellx4000
{\pard\intbl a|b\cell}{\pard\intbl c\cell}\row
""",
        )

        assert markdown.splitlines()[0] == "| a\\|b | c |"

    def test_a_nested_table_is_flattened_into_its_cell(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
\trowd\cellx2000\cellx4000
{\pard\intbl outer\par}{\pard\intbl inner\nestcell}{\*\nesttableprops\nestrow}\cell
{\pard\intbl right\cell}\row
""",
        )

        assert markdown.splitlines()[0] == "| outer<br>inner | right |"


class TestCharacterFormatting:
    def test_bold_italic_and_strike_runs_spanning_groups(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard plain {\b bold} and {\i italic} and {\b\i both}"
            rb" and {\strike gone} back to plain\par}",
        )

        assert markdown == (
            "plain **bold** and *italic* and ***both*** and ~~gone~~ back to plain\n"
        )

    def test_adjacent_runs_of_the_same_format_merge(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard {\b bo}{\b ld} text\par}")

        assert markdown == "**bold** text\n"

    def test_bold_whitespace_never_becomes_empty_emphasis(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard {\b }{\b  }kept\par}")

        assert "****" not in markdown
        assert markdown == "kept\n"

    def test_formatting_does_not_leak_past_its_group(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard {\b bold}plain\par}")

        assert markdown == "**bold**plain\n"

    def test_line_is_a_hard_break_and_tab_is_four_spaces(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard one\line two\tab three\par}")

        assert markdown == "one  \ntwo    three\n"

    def test_hidden_text_is_dropped(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard shown {\v hidden}text\par}")

        assert markdown == "shown text\n"


class TestEncoding:
    def test_unicode_escape_wins_over_its_ansi_fallback(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard\uc1 caf\u233 ? and \u8212 - here\par}",
        )

        assert markdown == "café and — here\n"

    def test_uc2_skips_two_fallback_characters(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard\uc2 \u8212 --dash\par}")

        assert markdown == "—dash\n"

    def test_surrogate_pairs_recombine(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard\uc1 \u-10179 ?\u-8694 ?\par}",
        )

        assert markdown == "\U0001f60a\n"

    def test_cp1252_hex_bytes(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard caf\'e9 na\'efve r\'e9sum\'e9\par}")

        assert markdown == "café naïve résumé\n"

    def test_gbk_document_decodes_multibyte_pairs(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard\f0 \'d6\'d0\'ce\'c4 test\par}",
            preamble=(
                rb"{\rtf1\ansi\ansicpg936\deff0"
                rb"{\fonttbl{\f0\fnil\fcharset134 SimSun;}}"
            ),
        )

        assert markdown == "中文 test\n"

    def test_font_charset_overrides_the_document_codepage(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard\f0 \'e9\par}{\pard\f1 \'d6\'d0\par}",
            preamble=(
                rb"{\rtf1\ansi\ansicpg1252\deff0"
                rb"{\fonttbl{\f0\fswiss Helvetica;}"
                rb"{\f1\fnil\fcharset134 SimSun;}}"
            ),
        )

        assert markdown == "é\n\n中\n"


class TestHeadings:
    def test_outline_levels_become_hashes(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\pard\outlinelevel0 Top\par}
{\pard\outlinelevel2 Deep\par}
{\pard Body.\par}
""",
        )

        assert markdown == "# Top\n\n### Deep\n\nBody.\n"

    def test_stylesheet_names_supply_the_level(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\stylesheet{\s0\snext0 Normal;}{\s1\sbasedon0 heading 1;}
{\s2\sbasedon0 heading 2;}{\s15 Title;}}
{\pard\s15 Doc Title\par}
{\pard\s1 Chapter\par}
{\pard\s2 Section\par}
{\pard\s0 Body.\par}
""",
        )

        assert markdown == "# Doc Title\n\n# Chapter\n\n## Section\n\nBody.\n"

    def test_a_short_bold_oversized_paragraph_is_promoted(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\pard\plain\fs24 Some ordinary body text that runs on for a while.\par}
{\pard\plain\b\fs36 A Big Bold Heading\par}
{\pard\plain\fs24 More ordinary body text follows here.\par}
""",
        )

        assert "# A Big Bold Heading" in markdown

    def test_a_long_bold_paragraph_stays_a_paragraph(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\pard\plain\fs24 Some ordinary body text that runs on for a while.\par}
{\pard\plain\b\fs36 This bold sentence is far too long to be a heading and so
it must remain an ordinary paragraph of the document.\par}
""",
        )

        assert not markdown.splitlines()[-1].startswith("#")

    def test_headings_do_not_carry_redundant_bold(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard\outlinelevel0\b Title\par}")

        assert markdown == "# Title\n"


class TestFieldsAndDestinations:
    def test_hyperlink_fields_become_markdown_links(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"""
{\pard See {\field{\*\fldinst HYPERLINK "https://example.com/a"}
{\fldrslt\ul\cf2 the site}} now.\par}
""",
        )

        assert markdown == "See [the site](https://example.com/a) now.\n"

    def test_other_fields_keep_only_their_result(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard Page {\field{\*\fldinst PAGE }{\fldrslt 7}} of 9.\par}",
        )

        assert markdown == "Page 7 of 9.\n"

    def test_headers_footers_footnotes_and_pictures_are_dropped(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "dropped.rtf"
        path.write_bytes(
            _PREAMBLE
            + rb"""
{\info{\title My Title}{\author Nobody}}
{\header\pard Page header\par}
{\footer\pard Page footer\par}
{\pard Body text{\footnote\pard A footnote.\par} continues.\par}
{\pard{\*\shppict{\pict\pngblip 89504e47}}
{\nonshppict{\pict\wmetafile8 0102}}Caption.\par}
"""
            + b"}"
        )
        result = RtfConverter().convert(path)

        assert result.markdown == "Body text continues.\n\nCaption.\n"
        assert result.metadata["title"] == "My Title"
        assert result.metadata["dropped_images"] == 1

    def test_unknown_starred_destinations_are_skipped(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard keep{\*\madeupdest drop this}ing\par}",
        )

        assert markdown == "keeping\n"


class TestMalformedInput:
    def test_unbalanced_braces(self, tmp_path: Path) -> None:
        path = tmp_path / "broken.rtf"
        path.write_bytes(rb"{\rtf1\ansi{\pard hello\par}")

        with pytest.raises(ConversionError, match="unbalanced group"):
            RtfConverter().convert(path)

    def test_a_stray_closing_brace(self, tmp_path: Path) -> None:
        path = tmp_path / "extra.rtf"
        path.write_bytes(rb"{\rtf1\ansi\pard hi\par}}")

        with pytest.raises(ConversionError, match="unbalanced group"):
            RtfConverter().convert(path)

    def test_a_file_that_is_not_rtf(self, tmp_path: Path) -> None:
        path = tmp_path / "plain.rtf"
        path.write_bytes(b"just some text, no header at all")

        with pytest.raises(ConversionError, match="not an RTF document"):
            RtfConverter().convert(path)

    def test_undecodable_bytes_are_replaced_rather_than_raising(
        self, tmp_path: Path
    ) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard ok \'ff\'fe done\par}",
            preamble=rb"{\rtf1\ansi\ansicpg936\deff0{\fonttbl{\f0\fcharset134 X;}}",
        )

        assert markdown.startswith("ok ")
        assert markdown.endswith("done\n")


class TestNormalisation:
    def test_blank_paragraphs_collapse(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            rb"{\pard one\par}{\pard\par}{\pard\par}{\pard two\par}",
        )

        assert markdown == "one\n\ntwo\n"

    def test_an_empty_document_yields_empty_markdown(self, tmp_path: Path) -> None:
        assert _convert(tmp_path, rb"{\pard\par}") == ""

    def test_sect_and_page_end_the_paragraph(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, rb"{\pard one\page two\sect three\par}")

        assert markdown == "one\n\ntwo\n\nthree\n"
