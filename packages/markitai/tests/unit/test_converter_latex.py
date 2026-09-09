"""Tests for the native LaTeX converter."""

from __future__ import annotations

from pathlib import Path

from markitai.converter.latex import TexConverter

FIXTURES = Path(__file__).parent.parent / "fixtures"


def _convert(tmp_path: Path, source: str) -> str:
    path = tmp_path / "doc.tex"
    path.write_text(source, encoding="utf-8")
    return TexConverter().convert(path).markdown


class TestTexConverter:
    def test_fixture_keeps_only_the_body(self) -> None:
        result = TexConverter().convert(FIXTURES / "sample.tex")

        assert result.markdown.strip() == "Hello World from LaTeX!"
        assert result.metadata["format"] == "TEX"

    def test_preamble_is_dropped_and_title_kept(self, tmp_path: Path) -> None:
        path = tmp_path / "titled.tex"
        path.write_text(
            "\\documentclass{article}\n"
            "\\usepackage{amsmath}\n"
            "\\title{On Conversion}\n"
            "\\begin{document}\n\\maketitle\nBody text.\n\\end{document}\n",
            encoding="utf-8",
        )

        result = TexConverter().convert(path)

        assert result.metadata["title"] == "On Conversion"
        assert result.markdown.strip() == "Body text."

    def test_sections_become_headings(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            "\\begin{document}\n\\section{One}\ntext\n"
            "\\subsection{Two}\n\\subsubsection{Three}\n\\end{document}\n",
        )

        assert "# One" in markdown
        assert "## Two" in markdown
        assert "### Three" in markdown

    def test_inline_formatting(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            "\\begin{document}\n"
            "\\textbf{bold} \\emph{em} \\textit{it} \\texttt{code}\n"
            "\\end{document}\n",
        )

        assert markdown.strip() == "**bold** *em* *it* `code`"

    def test_lists_become_markdown_lists(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            "\\begin{document}\n"
            "\\begin{itemize}\n\\item apples\n\\item pears\n\\end{itemize}\n"
            "\\begin{enumerate}\n\\item first\n\\end{enumerate}\n"
            "\\end{document}\n",
        )

        assert "- apples" in markdown
        assert "- pears" in markdown
        assert "1. first" in markdown

    def test_nested_lists_are_indented(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            "\\begin{document}\n\\begin{itemize}\n\\item outer\n"
            "\\begin{itemize}\n\\item inner\n\\end{itemize}\n"
            "\\end{itemize}\n\\end{document}\n",
        )

        assert "- outer" in markdown
        assert "  - inner" in markdown

    def test_verbatim_becomes_a_fence_and_keeps_percent(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            "\\begin{document}\n\\begin{verbatim}\n50% done\n\\end{verbatim}\n"
            "\\end{document}\n",
        )

        assert markdown.strip() == "```\n50% done\n```"

    def test_comments_go_but_escaped_percent_stays(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path,
            "\\begin{document}\nkept 50\\% here % dropped\n\\end{document}\n",
        )

        assert markdown.strip() == "kept 50% here"

    def test_labels_go_and_unknown_commands_keep_their_text(
        self, tmp_path: Path
    ) -> None:
        markdown = _convert(
            tmp_path,
            "\\begin{document}\n\\label{sec:a}\\textsc{Kept} words\n\\end{document}\n",
        )

        assert markdown.strip() == "Kept words"

    def test_double_backslash_breaks_the_line(self, tmp_path: Path) -> None:
        markdown = _convert(
            tmp_path, "\\begin{document}\none \\\\ two\n\\end{document}"
        )

        assert "one" in markdown.splitlines()[0]
        assert "two" in markdown.splitlines()[1]

    def test_unbalanced_braces_do_not_raise(self, tmp_path: Path) -> None:
        markdown = _convert(tmp_path, "\\begin{document}\n\\textbf{oops\n")

        assert "oops" in markdown
