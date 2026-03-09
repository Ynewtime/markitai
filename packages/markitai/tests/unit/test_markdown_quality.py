from __future__ import annotations


def test_normalize_markdown_reuses_existing_cleanup_primitives() -> None:
    from markitai.utils.markdown_quality import normalize_markdown

    markdown = "[Title\n\nDescription](/docs)\n\n__MARKITAI_FILE_ASSET__\n\n# Heading"
    cleaned = normalize_markdown(markdown)

    assert "[Title](/docs)" in cleaned
    assert "__MARKITAI" not in cleaned
    assert cleaned.endswith("\n")
