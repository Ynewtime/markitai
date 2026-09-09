from __future__ import annotations

import pytest

from markitai.webextract.utils import has_responsive_show_class


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        ("hidden md:block", True),
        ("hidden min-[600px]:block", True),
        ("hidden max-[900px]:flex", True),
        ("hidden", False),
        ("md:hidden", False),
    ],
)
def test_arbitrary_breakpoints_count_as_responsive_show(
    classes: str, expected: bool
) -> None:
    assert has_responsive_show_class(classes) is expected
