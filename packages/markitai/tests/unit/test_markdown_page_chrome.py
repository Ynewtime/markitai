"""Running header/footer removal in the shared markdown cleanup.

``normalize_markdown`` runs on every output. Its header/footer pass splits
on page/slide markers and applies ``strip_repeated_page_lines``, the PDF
converter's rule: only the first/last two lines of each page, a line must
repeat on at least 60% of pages (and at least 3), documents under 4 pages
are left alone, and headings and table rows are never removed. Marker
pages get a second round, so a three-line slide footer comes off whole.
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
