"""A pipe table has one delimiter row; a repeated one is not a data row."""

from __future__ import annotations

from markitai.utils.markdown_quality import normalize_markdown
from markitai.utils.text import drop_stray_table_delimiters

TABLE = """| Tool | What it does |
| --- | --- |
| --- | --- |
| lamp.sh | Shows the lamp's state; |
| :--- | ---: |
| dim.sh | Dims it. |"""


def test_repeated_delimiter_rows_are_dropped():
    assert drop_stray_table_delimiters(TABLE) == (
        "| Tool | What it does |\n"
        "| --- | --- |\n"
        "| lamp.sh | Shows the lamp's state; |\n"
        "| dim.sh | Dims it. |"
    )


def test_each_table_keeps_its_own_delimiter():
    two = "| a |\n| --- |\n| 1 |\n\n| b |\n| --- |\n| 2 |"
    assert drop_stray_table_delimiters(two) == two


def test_fenced_code_is_left_alone():
    fenced = "```\n| a |\n| --- |\n| --- |\n```"
    assert drop_stray_table_delimiters(fenced) == fenced
    tilde = "~~~~\n| a |\n| --- |\n~~~\n| --- |\n~~~~"
    assert drop_stray_table_delimiters(tilde) == tilde


def test_a_cell_with_dashes_and_text_stays():
    table = "| a | b |\n| --- | --- |\n| --- | text |"
    assert drop_stray_table_delimiters(table) == table


def test_the_shared_cleanup_applies_it():
    # Base and LLM output both pass normalize_markdown (llm/document.py
    # format_llm_output, workflow/helpers.py)
    assert "| --- | --- |\n| --- | --- |" not in normalize_markdown(TABLE)
