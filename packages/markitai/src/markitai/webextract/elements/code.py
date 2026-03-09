from __future__ import annotations

from bs4 import BeautifulSoup, Tag


def normalize_code_blocks(root: Tag) -> None:
    """Normalize code blocks in place.

    Args:
        root: Content root.
    """

    for code in list(root.find_all("code")):
        classes = code.get("class", [])
        style = str(code.get("style", ""))
        if "white-space: pre" in style and code.parent and code.parent.name != "pre":
            pre = BeautifulSoup("", "html.parser").new_tag("pre")
            code.wrap(pre)
        if classes:
            code["class"] = classes
