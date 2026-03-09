from __future__ import annotations

from bs4 import BeautifulSoup, Tag


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


def score_candidate(node: Tag) -> float:
    """Score a DOM node as likely main content.

    Args:
        node: Candidate tag.

    Returns:
        Numeric score.
    """

    text = node.get_text(" ", strip=True)
    words = len(text.split())
    paragraphs = len(node.find_all("p"))
    links = len(node.find_all("a"))
    classes = " ".join(node.get("class", []))

    score = float(words)
    score += paragraphs * 20
    if node.name == "article":
        score += 40
    if any(token in classes for token in ("article", "content", "post", "story")):
        score += 30
    if links and words:
        score -= min(40.0, (links / max(words, 1)) * 200)
    return score
