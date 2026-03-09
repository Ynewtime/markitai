from __future__ import annotations

from bs4 import BeautifulSoup, Tag


def normalize_footnotes(root: Tag) -> None:
    """Normalize footnotes into an ordered list.

    Args:
        root: Content root.
    """

    container = root.find(class_="footnotes")
    if not isinstance(container, Tag):
        return

    soup = BeautifulSoup("", "html.parser")
    ol = soup.new_tag("ol")
    for note in list(container.find_all(id=True)):
        li = soup.new_tag("li")
        text = note.get_text(" ", strip=True).replace(" ↩", "")
        if ". " in text:
            text = text.split(". ", 1)[1]
        li.string = text.strip()
        ol.append(li)

    section = soup.new_tag("section")
    section.append(ol)
    container.replace_with(section)
