from __future__ import annotations

from dataclasses import dataclass

from bs4 import BeautifulSoup, Tag


@dataclass(slots=True)
class _CandidateStats:
    """Pre-computed statistics for a candidate node.

    Avoids repeated subtree traversals by computing text, paragraph count,
    and link count in a single pass.
    """

    words: int
    paragraphs: int
    links: int


def _compute_stats(node: Tag) -> _CandidateStats:
    """Compute candidate statistics in a single pass.

    Calls get_text() once and counts <p>/<a> tags once, rather than
    invoking find_all() separately for each metric.

    Args:
        node: Candidate tag.

    Returns:
        Pre-computed statistics.
    """
    text = node.get_text(" ", strip=True)
    words = len(text.split())

    paragraphs = 0
    links = 0
    for descendant in node.descendants:
        if isinstance(descendant, Tag):
            if descendant.name == "p":
                paragraphs += 1
            elif descendant.name == "a":
                links += 1

    return _CandidateStats(words=words, paragraphs=paragraphs, links=links)


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

    classes: str = " ".join(node.get("class", []))  # type: ignore[arg-type]

    score = float(stats.words)
    score += stats.paragraphs * 20
    if node.name == "article":
        score += 40
    if any(token in classes for token in ("article", "content", "post", "story")):
        score += 30
    if stats.links and stats.words:
        score -= min(40.0, (stats.links / max(stats.words, 1)) * 200)
    return score
