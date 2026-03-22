"""Tests for CLI framework option consistency."""

from __future__ import annotations

import click

from markitai.cli.framework import MarkitaiGroup
from markitai.cli.main import app


class TestOptionsWithValues:
    """Verify _OPTIONS_WITH_VALUES stays in sync with actual CLI options."""

    def test_all_value_options_are_registered(self) -> None:
        """Every option that takes a value must appear in _OPTIONS_WITH_VALUES."""
        value_options: set[str] = set()
        for param in app.params:
            if isinstance(param, click.Option) and not param.is_flag:
                for opt in param.opts + param.secondary_opts:
                    value_options.add(opt)

        missing = value_options - MarkitaiGroup._OPTIONS_WITH_VALUES
        assert not missing, (
            f"Options taking values but missing from _OPTIONS_WITH_VALUES: {missing}. "
            f"Add them to MarkitaiGroup._OPTIONS_WITH_VALUES in framework.py."
        )

    def test_no_stale_entries_in_options_with_values(self) -> None:
        """_OPTIONS_WITH_VALUES should not contain entries that don't exist or are flags."""
        value_options: set[str] = set()
        for param in app.params:
            if isinstance(param, click.Option) and not param.is_flag:
                for opt in param.opts + param.secondary_opts:
                    value_options.add(opt)

        extra = MarkitaiGroup._OPTIONS_WITH_VALUES - value_options
        assert not extra, (
            f"Stale entries in _OPTIONS_WITH_VALUES (not value-taking options): {extra}. "
            f"Remove them from MarkitaiGroup._OPTIONS_WITH_VALUES in framework.py."
        )
