from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag


@dataclass(slots=True)
class _CandidateStats:
    """Pre-computed statistics for a candidate node.

    Avoids repeated subtree traversals by computing text, paragraph count,
    link count, and comma count in a single pass.
    """

    words: int
    paragraphs: int
    links: int
    commas: int
    link_text_len: int
    total_text_len: int


def _compute_stats(node: Tag) -> _CandidateStats:
    """Compute candidate statistics in a single pass.

    Args:
        node: Candidate tag.

    Returns:
        Pre-computed statistics.
    """
    text = node.get_text(" ", strip=True)
    words = len(text.split())
    commas = text.count(",")
    total_text_len = len(text)

    paragraphs = 0
    links = 0
    link_text_len = 0
    for descendant in node.descendants:
        if isinstance(descendant, Tag):
            if descendant.name == "p":
                paragraphs += 1
            elif descendant.name == "a":
                links += 1
                link_text_len += len(descendant.get_text(strip=True))

    return _CandidateStats(
        words=words,
        paragraphs=paragraphs,
        links=links,
        commas=commas,
        link_text_len=link_text_len,
        total_text_len=total_text_len,
    )


def select_best_candidate(soup: BeautifulSoup) -> Tag | None:
    """Return the highest-scoring content candidate.

    Args:
        soup: Parsed HTML document.

    Returns:
        Best scoring tag, if any.
    """

    best: Tag | None = None
    best_score = float("-inf")

    for candidate in soup.find_all(["article", "main", "section", "div"]):
        score = score_candidate(candidate)
        if score > best_score:
            best = candidate
            best_score = score

    return best


def score_candidate(node: Tag, stats: _CandidateStats | None = None) -> float:
    """Score a DOM node as likely main content.

    Args:
        node: Candidate tag.
        stats: Optional pre-computed statistics. If None, computed on demand.

    Returns:
        Numeric score.
    """
    if stats is None:
        stats = _compute_stats(node)

    raw_classes = node.get("class")
    classes: str = " ".join(raw_classes) if isinstance(raw_classes, list) else ""

    score = float(stats.words)
    score += stats.paragraphs * 20
    score += stats.commas  # prose indicator
    if node.name == "article":
        score += 40
    if any(token in classes for token in ("article", "content", "post", "story")):
        score += 30

    # Multiplicative link density scaling (defuddle approach)
    if stats.total_text_len > 0:
        link_density = min(stats.link_text_len / stats.total_text_len, 0.5)
        score *= 1 - link_density
    return score
