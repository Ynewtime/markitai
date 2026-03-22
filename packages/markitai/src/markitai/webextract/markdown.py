from __future__ import annotations

"""Narrow Markdown fidelity layer for canonical HTML content.

This module adds a thin pre- and post-processing layer around MarkItDown's
HTML-to-Markdown conversion.  It handles cases that MarkItDown does not
preserve well on its own:

- ``<figure>``/``<figcaption>`` pairs: captions are kept beneath the image.
- ``srcset`` attributes: the highest-resolution URL is selected as ``src``.
- Social/video embed ``<iframe>`` elements: reduced to plain canonical links.
- Post-processing: collapse excessive blank lines and strip trailing whitespace.

The module deliberately does **not** rewrite Markdown rendering from scratch;
it only pre/post-processes around MarkItDown.
"""

import re
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_markdown(html: str, *, md_instance: object | None = None) -> str:
    """Convert canonical HTML to high-fidelity Markdown.

    Applies pre-processing to preserve captions and canonicalize embeds,
    delegates to MarkItDown for the core conversion, then post-processes
    the output.

    Args:
        html: Canonical HTML fragment to convert.
        md_instance: Optional pre-created MarkItDown instance.  A new one is
            created when ``None``.

    Returns:
        Clean Markdown string.
    """
    if not html or not html.strip():
        return ""
    html = _preprocess_for_markdown(html)
    markdown = _html_to_markdown(html, md_instance)
    return _postprocess_markdown(markdown)


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------


def _preprocess_for_markdown(html: str) -> str:
    """Apply HTML mutations that improve Markdown fidelity.

    Transformations (in order):
    1. Pick best image URL from ``srcset`` attributes.
    2. Canonicalize social/video embed ``<iframe>`` elements to ``<a>`` links.
    3. Preserve ``<figure>``/``<figcaption>`` structure so captions survive.

    Args:
        html: Input HTML string.

    Returns:
        Mutated HTML string ready for MarkItDown.
    """
    soup = BeautifulSoup(html, "html.parser")
    _resolve_srcset(soup)
    _canonicalize_embeds(soup)
    _preserve_figure_captions(soup)
    return str(soup)


def _resolve_srcset(soup: BeautifulSoup) -> None:
    """Replace each ``img[srcset]`` src with the highest-resolution candidate.

    The srcset value is a comma-separated list of ``url width`` or ``url dpr``
    descriptors.  We pick the candidate with the largest numeric width/dpr
    descriptor.  Falls back to the first candidate when no descriptors are
    present.

    Args:
        soup: BeautifulSoup tree (mutated in place).
    """
    for img in soup.find_all("img"):
        if not isinstance(img, Tag):
            continue
        srcset = img.get("srcset", "")
        if not srcset:
            continue
        best_url = _pick_best_srcset_url(str(srcset))
        if best_url:
            img["src"] = best_url
        del img["srcset"]


def _pick_best_srcset_url(srcset: str) -> str | None:
    """Return the URL with the highest numeric descriptor from a srcset string.

    Args:
        srcset: Raw srcset attribute value.

    Returns:
        Best URL string, or ``None`` when srcset is empty.
    """
    best_url: str | None = None
    best_value: float = -1.0

    for candidate in srcset.split(","):
        candidate = candidate.strip()
        if not candidate:
            continue
        parts = candidate.split()
        url = parts[0]
        if len(parts) >= 2:
            descriptor = parts[1].lower().rstrip("wx")
            try:
                value = float(descriptor)
            except ValueError:
                value = 0.0
        else:
            value = 0.0

        if best_url is None or value > best_value:
            best_url = url
            best_value = value

    return best_url


# ---------------------------------------------------------------------------
# Embed canonicalization
# ---------------------------------------------------------------------------

# Patterns: (compiled regex matching iframe src, canonical URL builder)
_YOUTUBE_EMBED_RE = re.compile(
    r"^https?://(?:www\.)?youtube(?:-nocookie)?\.com/embed/([A-Za-z0-9_-]+)",
    re.IGNORECASE,
)
_VIMEO_EMBED_RE = re.compile(
    r"^https?://player\.vimeo\.com/video/(\d+)",
    re.IGNORECASE,
)
_TWITTER_EMBED_RE = re.compile(
    r"^https?://(?:twitter|x)\.com/",
    re.IGNORECASE,
)


def _canonicalize_embeds(soup: BeautifulSoup) -> None:
    """Replace embed ``<iframe>`` elements with canonical ``<a>`` links.

    Handles YouTube, Vimeo, and Twitter/X embeds.  Unknown iframes are left
    intact.

    Args:
        soup: BeautifulSoup tree (mutated in place).
    """
    for iframe in soup.find_all("iframe"):
        if not isinstance(iframe, Tag):
            continue
        src = str(iframe.get("src", "")).strip()
        if not src:
            continue
        canonical = _embed_src_to_canonical(src)
        if canonical is not None:
            link = soup.new_tag("a", href=canonical)
            link.string = canonical
            iframe.replace_with(link)


def _embed_src_to_canonical(src: str) -> str | None:
    """Convert an embed iframe src URL to a canonical link.

    Args:
        src: The ``src`` attribute value of an ``<iframe>``.

    Returns:
        Canonical URL string, or ``None`` if not a recognised embed pattern.
    """
    # YouTube
    yt_match = _YOUTUBE_EMBED_RE.match(src)
    if yt_match:
        video_id = yt_match.group(1)
        return f"https://www.youtube.com/watch?v={video_id}"

    # Vimeo
    vm_match = _VIMEO_EMBED_RE.match(src)
    if vm_match:
        video_id = vm_match.group(1)
        return f"https://vimeo.com/{video_id}"

    # Twitter / X  — keep the URL as-is (already canonical)
    if _TWITTER_EMBED_RE.match(src):
        # Normalise twitter.com → x.com for consistency
        parsed = urlparse(src)
        host = parsed.netloc.lower()
        if "twitter.com" in host:
            host = host.replace("twitter.com", "x.com")
        canonical = urlunparse(parsed._replace(netloc=host))
        return canonical

    return None


# ---------------------------------------------------------------------------
# Figure / caption preservation
# ---------------------------------------------------------------------------


def _preserve_figure_captions(soup: BeautifulSoup) -> None:
    """Ensure ``<figcaption>`` text survives MarkItDown conversion.

    MarkItDown drops ``<figcaption>`` elements.  We rewrite each
    ``<figure>`` so the caption text is preserved as an ``<em>`` paragraph
    immediately after the image, which MarkItDown renders faithfully.

    Args:
        soup: BeautifulSoup tree (mutated in place).
    """
    for figure in soup.find_all("figure"):
        if not isinstance(figure, Tag):
            continue
        figcaption = figure.find("figcaption")
        if figcaption is None or not isinstance(figcaption, Tag):
            continue
        caption_text = figcaption.get_text(strip=True)
        if not caption_text:
            continue

        # Remove the figcaption from its current position
        figcaption.decompose()

        # Append an <em><p> after the figure's image (or at end of figure)
        caption_tag = soup.new_tag("p")
        em_tag = soup.new_tag("em")
        em_tag.string = caption_text
        caption_tag.append(em_tag)

        # Insert after figure, or at end of figure if no parent
        parent = figure.parent
        if parent is not None:
            figure.insert_after(caption_tag)
        else:
            figure.append(caption_tag)


# ---------------------------------------------------------------------------
# HTML → Markdown (delegate to MarkItDown)
# ---------------------------------------------------------------------------


def _html_to_markdown(html: str, md_instance: object | None = None) -> str:
    """Convert HTML to Markdown using MarkItDown.

    Args:
        html: HTML string to convert.
        md_instance: Optional pre-created MarkItDown instance.

    Returns:
        Markdown text from MarkItDown.
    """
    import io

    from markitdown import MarkItDown, StreamInfo

    if md_instance is None:
        md_instance = MarkItDown()

    stream = io.BytesIO(html.encode("utf-8"))
    result = md_instance.convert_stream(  # type: ignore[union-attr]
        stream,
        file_extension=".html",
        stream_info=StreamInfo(
            mimetype="text/html",
            extension=".html",
            charset="utf-8",
        ),
    )
    return result.text_content if result and result.text_content else ""


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

_EXCESS_BLANK_LINES_RE = re.compile(r"\n{4,}")
_ORPHAN_SEPARATOR_RE = re.compile(r"^\s*[·|—–•]\s*$", re.MULTILINE)


def _postprocess_markdown(markdown: str) -> str:
    """Apply post-processing to the raw MarkItDown output.

    Transformations:
    1. Strip trailing whitespace from every line.
    2. Remove lines containing only separator characters.
    3. Collapse runs of 4+ newlines to at most 2 blank lines (3 newlines).

    Args:
        markdown: Raw Markdown string from MarkItDown.

    Returns:
        Cleaned Markdown string.
    """
    # Strip trailing whitespace per line
    lines = [line.rstrip() for line in markdown.splitlines()]
    markdown = "\n".join(lines)

    # Remove lines containing only separator characters
    markdown = _ORPHAN_SEPARATOR_RE.sub("", markdown)

    # Collapse excessive blank lines (>2 consecutive blank lines → 2)
    markdown = _EXCESS_BLANK_LINES_RE.sub("\n\n\n", markdown)

    return markdown
