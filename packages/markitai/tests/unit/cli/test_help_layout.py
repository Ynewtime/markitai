"""`markitai --help` is the product's first screen; keep it curated.

rich-click renders any option missing from OPTION_GROUPS in an unnamed
"Options" panel that it prints *first*. The three `--llm-batch*` flags were
never added to a group, so the first thing every new user read — above
output, LLM, OCR and everything else — was the half-price Batch API, the
most specialised feature in the tool.
"""

from __future__ import annotations

import rich_click

from markitai.cli import app


def _declared_long_options() -> set[str]:
    """Primary spellings only.

    A ``--no-*`` counterpart is a secondary opt on the same parameter and
    rich-click renders it on that parameter's row, so it needs no group of
    its own.
    """
    return {
        option
        for param in app.params
        for option in param.opts
        if option.startswith("--")
    }


def _grouped_options() -> set[str]:
    groups = rich_click.rich_click.OPTION_GROUPS["markitai"]
    # "options" is NotRequired: a panel declared for its help text alone has
    # no key here, and indexing it would raise rather than report.
    return {option for group in groups for option in group.get("options", [])}


def test_every_option_belongs_to_a_named_panel() -> None:
    ungrouped = sorted(_declared_long_options() - _grouped_options())
    assert not ungrouped, (
        "these options have no panel, so rich-click prints them first, above "
        f"the curated groups: {ungrouped}"
    )


def test_no_panel_advertises_an_option_that_does_not_exist() -> None:
    # --help is click's own, added at render time rather than declared.
    phantom = sorted(_grouped_options() - _declared_long_options() - {"--help"})
    assert not phantom, f"panels list options the CLI does not define: {phantom}"


def test_examples_only_promise_what_the_cli_does() -> None:
    """The help text once offered `markitai <youtube url>  # Convert YouTube
    video`. Nothing in the codebase converts a video: markitai reads YouTube
    only as an *embed* inside a fetched page, which it rewrites to a link.
    """
    assert app.help is not None
    assert "youtube" not in app.help.lower()
