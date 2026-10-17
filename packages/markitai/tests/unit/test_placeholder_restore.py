"""Structural placeholders an answer dropped: put back, or one named retry.

A 1.2.1 check: ``default-table.pdf --llm`` failed with Gemini
Flash-Lite and OpenRouter's Claude Haiku alias: "LLM answer dropped 1/1
structural placeholders", the ``__MARKITAI_PAGENUM_0__`` that opens the
one-page PDF, before its title. The whole cleanup was thrown away.
"""

from __future__ import annotations

import pytest

from markitai.llm.content import restore_dropped_placeholders
from markitai.llm.engine import LLMEnhancementDegradedError, track_llm_spend
from tests.unit.test_llm_spend import (
    IMG,
    P0,
    P1,
    TABLE_PAGE,
    TWO_PAGES,
    _answer,
    _body,
    _processor,
    _Router,
)


class TestRestoreDroppedPlaceholders:
    SOURCE = _body(TABLE_PAGE)

    def test_page_marker_that_opens_the_document_goes_back(self) -> None:
        answer = _body(TABLE_PAGE).removeprefix(P0).lstrip()
        restored, missing = restore_dropped_placeholders(self.SOURCE, answer, [P0])
        assert missing == []
        assert restored.startswith(P0 + "\n\n# Language comparison")

    @pytest.mark.parametrize(
        "rewritten",
        [
            "**MARKITAI_PAGENUM_0**",
            "\\_\\_MARKITAI\\_PAGENUM\\_0\\_\\_",
            "MARKITAI_PAGENUM_0",
        ],
    )
    def test_placeholder_rewritten_as_markdown_in_place(self, rewritten: str) -> None:
        answer = _body(TABLE_PAGE).replace(P0, rewritten)
        restored, missing = restore_dropped_placeholders(self.SOURCE, answer, [P0])
        assert missing == []
        assert restored == _body(TABLE_PAGE)

    def test_reformatted_neighbours_still_anchor(self) -> None:
        answer = "## Language Comparison\n\nA table in the browser's default style."
        restored, missing = restore_dropped_placeholders(self.SOURCE, answer, [P0])
        assert missing == []
        assert restored.startswith(P0)

    @pytest.mark.parametrize(
        "answer",
        [
            # the title the marker stood before was rewritten
            "# Comparing languages\n\nA table.",
            # something the source did not have now opens the answer
            "Here is the cleaned document:\n\n# Language comparison\n\nA table.",
            # the placeholder was kept, but moved below the title
            "# Language comparison\n\n**MARKITAI_PAGENUM_0**\n\nA table.",
        ],
    )
    def test_ambiguous_positions_stay_missing(self, answer: str) -> None:
        _, missing = restore_dropped_placeholders(self.SOURCE, answer, [P0])
        assert missing == [P0]

    def test_marker_between_kept_paragraphs_goes_back(self) -> None:
        source = f"{P0}\n\n# One\n\nAlpha\n\n{P1}\n\n# Two\n\nBeta"
        restored, missing = restore_dropped_placeholders(
            source, "# One\n\nAlpha\n\n# Two\n\nBeta", [P0, P1]
        )
        assert missing == []
        assert restored == source

    @pytest.mark.parametrize(
        "answer",
        [
            "# One\n\nAlpha Beta",  # the pages were merged
            "# One\n\nAlpha\n\nBeta",  # page 2's heading was dropped with it
            "# One\n\n# Two\n\nBeta",  # page 1's text was dropped with it
        ],
    )
    def test_content_lost_next_to_a_marker_is_not_repaired(self, answer: str) -> None:
        source = f"{P0}\n\n# One\n\nAlpha\n\n{P1}\n\n# Two\n\nBeta"
        _, missing = restore_dropped_placeholders(source, answer, [P0, P1])
        assert P1 in missing

    def test_image_written_back_elsewhere_is_not_accepted(self) -> None:
        image = "![cover](.markitai/assets/a.jpeg)"
        source = f"{IMG}\n\nFirst.\n\nSecond."
        restored, missing = restore_dropped_placeholders(
            source, f"First.\n\nSecond.\n\n{image}", [IMG], {IMG: image}
        )
        assert missing == [IMG]
        restored, missing = restore_dropped_placeholders(
            source, f"{image}\n\nFirst.\n\nSecond.", [IMG], {IMG: image}
        )
        assert missing == []
        assert restored == source

    def test_inline_placeholder_is_never_guessed(self) -> None:
        source = f"Before {IMG} after."
        _, missing = restore_dropped_placeholders(source, "Before after.", [IMG])
        assert missing == [IMG]


class TestDocumentCall:
    async def test_one_page_pdf_answer_without_its_marker_completes(self) -> None:
        cleaned = _body(TABLE_PAGE).removeprefix(P0).lstrip()
        router = _Router(_answer(cleaned))
        processor = _processor(router)

        result, _frontmatter = await processor.process_document(
            TABLE_PAGE, "default-table.pdf"
        )

        assert result.startswith("<!-- Page number: 1 -->")
        assert "|Python|1991|Dynamic|General purpose|" in result
        assert len(router.requests) == 1

    async def test_unrestorable_loss_gets_one_retry_naming_the_placeholder(
        self,
    ) -> None:
        merged = _body(TWO_PAGES).replace(f"{P1}\n\n# The buses\n\n", "")
        router = _Router(_answer(merged), _answer(_body(TWO_PAGES)))
        processor = _processor(router)

        with track_llm_spend() as spend:
            result, _ = await processor.process_document(TWO_PAGES, "two.pdf")

        assert "<!-- Page number: 2 -->" in result
        assert len(router.requests) == 2
        retry_system = router.requests[1][0]["content"]
        assert "Previous Answer Rejected" in retry_system
        assert f"`{P1}`, on its own line, right before: # The buses" in retry_system
        assert P1 not in router.requests[0][0]["content"]
        # Both answers were billed
        assert spend.usage["deepseek-flash"]["requests"] == 2
        assert spend.cost == pytest.approx(0.02)

    async def test_a_real_loss_is_still_rejected_after_the_retry(self) -> None:
        merged = _body(TWO_PAGES).replace(f"{P1}\n\n# The buses\n\n", "")
        router = _Router(_answer(merged))
        processor = _processor(router)

        with (
            track_llm_spend() as spend,
            pytest.raises(LLMEnhancementDegradedError) as exc_info,
        ):
            await processor.process_document(TWO_PAGES, "two.pdf")

        assert "1/2 structural placeholders (page 2 marker)" in str(exc_info.value)
        assert "the one retry lost them too" in str(exc_info.value)
        assert exc_info.value.cleaned_markdown == TWO_PAGES
        assert spend.usage["deepseek-flash"]["requests"] == 2
        assert spend.cost == pytest.approx(0.02)

    async def test_no_retry_without_room_in_the_request_budget(self) -> None:
        merged = _body(TWO_PAGES).replace(f"{P1}\n\n# The buses\n\n", "")
        router = _Router(_answer(merged))
        processor = _processor(router, max_requests_per_document=1)

        with pytest.raises(
            LLMEnhancementDegradedError, match="structural placeholders"
        ):
            await processor.process_document(TWO_PAGES, "two.pdf")

        assert len(router.requests) == 1
        assert processor.get_context_cost("two.pdf") == pytest.approx(0.01)
