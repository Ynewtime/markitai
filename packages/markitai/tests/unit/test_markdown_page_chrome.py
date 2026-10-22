"""Running header/footer removal in the shared markdown cleanup.

``normalize_markdown`` runs on every output. Its header/footer pass splits
on page/slide markers and applies ``strip_repeated_page_lines``, the PDF
converter's rule: only the first/last two lines of each page, a line must
repeat on at least 60% of pages (and at least 3), documents under 4 pages
are left alone, and headings and table rows are never removed. Only short
lines are candidates, and a line whose form repeats within its own pages is
body text. Marker pages also examine the third line inside two removed ones,
so a three-line slide footer comes off whole, in one pass that does not eat
further into the body.
"""

from __future__ import annotations

from markitai.utils.markdown_quality import normalize_markdown


def _pages(bodies: list[str], marker: str = "Page number") -> str:
    return "\n\n".join(
        f"<!-- {marker}: {i} -->\n\n{body}" for i, body in enumerate(bodies, 1)
    )


class TestRunningHeadersAndFooters:
    def test_running_header_and_footer_removed(self) -> None:
        bodies = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
        content = _pages(
            [
                f"ACME Confidential\n\n{body} opens.\n\n{body} goes on.\n\n"
                f"{body} ends.\n\nAcme Corp\n{i}"
                for i, body in enumerate(bodies, 1)
            ]
        )

        cleaned = normalize_markdown(content)

        assert "ACME Confidential" not in cleaned
        assert "Acme Corp" not in cleaned
        for i, body in enumerate(bodies, 1):
            assert f"{body} opens.\n\n{body} goes on.\n\n{body} ends." in cleaned
            assert f"<!-- Page number: {i} -->" in cleaned

    def test_three_line_slide_footer_removed(self) -> None:
        """PowerPoint footers can be date + text + slide number."""
        topics = ["Plan", "Risks", "Budget", "Team"]
        content = _pages(
            [
                f"# {topic}\n\nAbout {topic.lower()}.\n\nFTD\nFREE TEST DATA\n{i}"
                for i, topic in enumerate(topics, 1)
            ],
            marker="Slide number",
        )

        cleaned = normalize_markdown(content)

        assert "FTD" not in cleaned
        assert "FREE TEST DATA" not in cleaned
        for topic in topics:
            assert f"# {topic}\n\nAbout {topic.lower()}." in cleaned

    def test_body_line_matching_a_footer_is_kept(self) -> None:
        content = _pages(
            [
                "Intro.\n\nHistory.\n\nAcme Corp\n\nwas founded in 1999.\n\n"
                "More text.\n\nAcme Corp\n1",
                "Revenue grew.\n\nAcme Corp\n2",
                "Costs stayed flat.\n\nAcme Corp\n3",
                "Outlook is good.\n\nAcme Corp\n4",
            ]
        )

        cleaned = normalize_markdown(content)

        assert cleaned.count("Acme Corp") == 1
        assert "Acme Corp\n\nwas founded in 1999." in cleaned

    def test_line_on_only_half_the_slides_is_kept(self) -> None:
        content = _pages(
            [
                "# Plan\n\nShip it.\n\nQuestions?",
                "# Risks\n\nNone.\n\nQuestions?",
                "# Budget\n\nSmall.",
                "# Team\n\nFour people.",
            ],
            marker="Slide number",
        )

        cleaned = normalize_markdown(content)

        assert cleaned.count("Questions?") == 2

    def test_table_rows_and_headings_are_kept(self) -> None:
        content = _pages(
            [
                f"## Totals\n\n| Item | Cost |\n| --- | --- |\n| Item {i} | {i} |\n| Total | n/a |"
                for i in range(1, 5)
            ]
        )

        cleaned = normalize_markdown(content)

        assert cleaned.count("| Total | n/a |") == 4
        assert cleaned.count("## Totals") == 4

    def test_short_documents_are_left_alone(self) -> None:
        content = _pages([f"Body {i}.\n\nAcme Corp\n{i}" for i in range(1, 4)])

        cleaned = normalize_markdown(content)

        assert cleaned.count("Acme Corp") == 3

    def test_numbered_body_lines_are_kept(self) -> None:
        """Lines that differ only in their numbers looked like one running
        line to the digit-blind match: a 4-page scan read by OCR kept 3 of
        its 48 lines."""
        content = _pages(
            [
                "\n".join(
                    f"Page {page} line {line}: The quick brown fox jumps over "
                    f"the lazy dog {page}{line:02d}."
                    for line in range(1, 13)
                )
                for page in range(1, 5)
            ]
        )

        cleaned = normalize_markdown(content)

        assert cleaned.count("The quick brown fox") == 48

    def test_numbered_paragraphs_of_a_long_report_are_kept(self) -> None:
        paragraph = (
            "The committee reviewed section {}. The annual plan describes the "
            "library, its reading rooms and the budget lines for staff."
        )
        content = _pages(
            [
                f"# Chapter {page}\n\n"
                + "\n\n".join(paragraph.format(f"{page}.{k}") for k in range(1, 10))
                for page in range(1, 61)
            ]
        )

        cleaned = normalize_markdown(content)

        assert cleaned.count("The committee reviewed") == 540

    def test_long_lines_are_never_running_lines(self) -> None:
        notice = (
            "This report was prepared for the members of the board and is "
            "shared with them under the terms of the annual agreement."
        )
        bodies = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
        content = _pages(
            [f"{notice}\n\n{body} opens.\n\n{body} ends." for body in bodies]
        )

        cleaned = normalize_markdown(content)

        assert cleaned.count(notice) == 5

    def test_removal_does_not_cascade_into_the_body(self) -> None:
        """Each round used to look again at the lines the last one uncovered;
        one pass takes at most three lines from an edge."""
        bodies = ["Alpha", "Bravo", "Charlie", "Delta", "Echo"]
        content = _pages(
            [
                "Acme Corp\n\nAnnual report\n\nDraft\n\nSee below.\n\n"
                f"{body} one.\n\n{body} two.\n\n{body} three."
                for body in bodies
            ]
        )

        cleaned = normalize_markdown(content)

        for line in ("Acme Corp", "Annual report", "Draft"):
            assert line not in cleaned
        assert cleaned.count("See below.") == 5
