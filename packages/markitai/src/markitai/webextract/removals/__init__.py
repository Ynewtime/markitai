from __future__ import annotations

from bs4 import Tag

from markitai.webextract.removals.content_patterns import remove_content_patterns
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
) -> dict[str, int]:
    """Apply all removal stages to the content root.

    Args:
        root: Content root element.
        main_content: The identified main content element (protected from removal).
        use_partial_selectors: Whether to use partial selector matching.
        use_hidden_removal: Whether to remove hidden elements.
        use_scoring: Whether to score and remove non-content blocks.

    Returns:
        Dict mapping removal stage names to count of elements removed.
    """
    stats: dict[str, int] = {}
    stats["small_images"] = remove_small_images(root)
    if use_hidden_removal:
        stats["hidden"] = remove_hidden_elements(root)
    stats["selectors"] = remove_by_selectors(
        root, main_content, use_partial=use_partial_selectors
    )
    if use_scoring:
        stats["scoring"] = score_and_remove(root)
    stats["content_patterns"] = remove_content_patterns(root)
    return stats
