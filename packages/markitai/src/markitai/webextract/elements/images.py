from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag


def normalize_images(root: Tag, base_url: str) -> None:
    """Normalize lazy-loaded images and captions.

    Args:
        root: Content root.
        base_url: Base URL for relative asset resolution.
    """

    for img in list(root.find_all("img")):
        src = img.get("src") or img.get("data-src") or img.get("data-original")
        if src:
            img["src"] = urljoin(base_url, str(src))
        caption = img.find_next_sibling(class_="caption")
        if isinstance(caption, Tag):
            soup = BeautifulSoup("", "html.parser")
            figure = soup.new_tag("figure")
            caption_text = caption.get_text(" ", strip=True)
            img.replace_with(figure)
            caption.extract()
            figure.append(img)
            figcaption = soup.new_tag("figcaption")
            figcaption.string = caption_text
            figure.append(figcaption)
