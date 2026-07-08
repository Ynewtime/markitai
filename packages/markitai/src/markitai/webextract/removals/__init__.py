from __future__ import annotations

from bs4 import Tag

from markitai.webextract.removals.content_patterns import (
    remove_content_patterns,
    remove_eyebrow_label,
)
from markitai.webextract.removals.hidden import remove_hidden_elements
from markitai.webextract.removals.scoring import score_and_remove
from markitai.webextract.removals.selectors import remove_by_selectors
from markitai.webextract.removals.small_images import remove_small_images


def apply_removals(
    root: Tag,
    main_content: Tag | None = None,
    *,
    use_partial_selectors: bool = True,
    use_hidden_removal: bool = True,
    use_scoring: bool = True,
    use_content_patterns: bool = True,
    url: str = "",
    title: str = "",
    description: str = "",
) -> dict[str, int]:
    """Apply all removal stages to the content root.

    Args:
        root: Content root element.
        main_content: The identified main content element (protected from removal).
        use_partial_selectors: Whether to use partial selector matching.
        use_hidden_removal: Whether to remove hidden elements.
        use_scoring: Whether to score and remove non-content blocks.
        use_content_patterns: Whether to remove text-pattern noise
            (disabled on the listing-page retry, mirroring defuddle).
        url: Canonical page URL (path-based breadcrumb/link checks).
        title: Page title from metadata (anchors the content boundary).
        description: Page description (duplicate-text removal).

    Returns:
        Dict mapping removal stage names to count of elements removed.
    """
    stats: dict[str, int] = {}
    stats["small_images"] = remove_small_images(root)
    if use_hidden_removal:
        stats["hidden"] = remove_hidden_elements(root)
    # Eyebrow removal runs before selector removal (mirrors defuddle) so the
    # h1 anchor is still present on pages that strip title classes.
    stats["eyebrow"] = remove_eyebrow_label(root)
    stats["selectors"] = remove_by_selectors(
        root, main_content, use_partial=use_partial_selectors
    )
    if use_scoring:
        stats["scoring"] = score_and_remove(root)
    if use_content_patterns:
        stats["content_patterns"] = remove_content_patterns(
            root, url=url, title=title, description=description
        )
    return stats
