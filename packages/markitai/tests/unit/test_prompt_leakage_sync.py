"""Anti-rot: leakage detectors must stay tied to the live prompt text.

A prompt-leakage detector can only fire on text the prompts actually
contain. When the prompt corpus was translated to English, the Chinese
detectors kept running against nothing at all — a silent failure nobody
could see. These tests make that visible: every retained detector must
match at least one line of the current prompts, so rewording a prompt
fails here instead of quietly disarming the guard.
"""

from __future__ import annotations

import re

import markitai.llm.content as content_module
from markitai.llm.document import (
    PAGE_LABEL_TEMPLATE,
    PROMPT_LEAKAGE_MARKERS,
    PURE_MODE_RULES,
    SCREENSHOT_LABEL,
    STANDARD_MODE_RULES,
    VISION_METADATA_SECTION,
    VISION_TAIL_REMINDER,
)
from markitai.llm.vision import VISION_PROMPT_FRAGMENTS
from markitai.prompts import BUILTIN_PROMPTS_DIR
from markitai.workflow.helpers import PROMPT_LEAKAGE_KEY_PATTERNS

# In-code prompt fragments that are sent to the model alongside the .md
# templates. Anything added here must also reach the model.
IN_CODE_PROMPT_TEXTS: tuple[str, ...] = (
    STANDARD_MODE_RULES,
    PURE_MODE_RULES,
    SCREENSHOT_LABEL,
    PAGE_LABEL_TEMPLATE,
    VISION_TAIL_REMINDER,
    VISION_METADATA_SECTION,
    *VISION_PROMPT_FRAGMENTS,
)


def prompt_corpus() -> str:
    """All prompt text the pipeline can send to a model."""
    template_texts = [
        path.read_text(encoding="utf-8")
        for path in sorted(BUILTIN_PROMPTS_DIR.glob("*.md"))
    ]
    return "\n".join([*template_texts, *IN_CODE_PROMPT_TEXTS])


def prompt_corpus_lines() -> list[str]:
    """Stripped, non-empty lines of the prompt corpus."""
    return [line.strip() for line in prompt_corpus().splitlines() if line.strip()]


class TestCorpusHarness:
    """The corpus itself must be real, or every check below is vacuous."""

    def test_corpus_covers_templates_and_code_fragments(self) -> None:
        corpus = prompt_corpus()

        assert len(prompt_corpus_lines()) > 100
        # A built-in template...
        assert "You are a professional Markdown formatting assistant." in corpus
        # ...and an in-code fragment
        assert "REMINDER: Preserve ALL __MARKITAI_*__ placeholders" in corpus

    def test_a_dead_pattern_is_reported_as_dead(self) -> None:
        """Positive control: the check can actually fail."""
        dead = re.compile(r"^【核心原则】.*$")

        assert not any(dead.match(line) for line in prompt_corpus_lines())


class TestLeakageDetectorsMatchLivePrompts:
    """Every retained detector must have something to detect."""

    def test_every_frontmatter_leakage_pattern_matches_prompt_text(self) -> None:
        lines = prompt_corpus_lines()
        dead = [
            pattern.pattern
            for pattern in content_module._PROMPT_LEAKAGE_PATTERNS
            if not any(pattern.match(line) for line in lines)
        ]

        assert not dead, (
            "These frontmatter leakage patterns match no line of any current "
            f"prompt, so they can never fire: {dead}. Either delete them or "
            "update them to the reworded prompt."
        )

    def test_every_frontmatter_key_pattern_matches_prompt_text(self) -> None:
        """``normalize_frontmatter`` filters hallucinated YAML *keys*.

        The key is a fragment the model echoed out of the prompt, so the
        pattern is searched anywhere in a prompt line — same rule as the
        ``re.search(..., IGNORECASE)`` the filter itself uses.
        """
        lines = prompt_corpus_lines()
        dead = [
            pattern
            for pattern in PROMPT_LEAKAGE_KEY_PATTERNS
            if not any(re.search(pattern, line, re.IGNORECASE) for line in lines)
        ]

        assert not dead, (
            "These frontmatter key patterns match no line of any current "
            f"prompt, so no model can echo them back: {dead}. Either delete "
            "them or update them to the reworded prompt."
        )

    def test_every_document_leakage_marker_matches_prompt_text(self) -> None:
        corpus = prompt_corpus()
        dead = [marker for marker in PROMPT_LEAKAGE_MARKERS if marker not in corpus]

        assert not dead, (
            "These prompt-leakage markers no longer appear in any prompt, so "
            f"they can never fire: {dead}. Either delete them or update them "
            "to the reworded prompt."
        )

    def test_document_leakage_markers_are_not_empty(self) -> None:
        """Guard against the marker list silently shrinking to nothing."""
        assert len(PROMPT_LEAKAGE_MARKERS) > 0
