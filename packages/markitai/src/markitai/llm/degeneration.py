"""Degeneration guard for VLM outputs.

Vision-model extraction can degenerate into repetition loops where the model
emits the same line or chunk endlessly at the end of the output. The logits
are not reachable through litellm (no ``no_repeat_ngram`` processors), so this
module post-checks generated text: detect a degenerate tail, truncate it
(keeping one instance of the repeated unit), and let callers skip persisting
the salvaged response to cache so a clean retry is possible on the next run.
"""

from __future__ import annotations

from loguru import logger

# Minimum length (chars) of a repeated chunk for the chunk detector.
DEFAULT_MIN_NGRAM_CHARS = 20
# Minimum consecutive repeats of a chunk to flag degeneration.
DEFAULT_MIN_REPEATS = 4
# Minimum consecutive identical non-empty lines to flag degeneration.
DEFAULT_MIN_LINE_REPEATS = 6

# Degeneration is a tail property; cap the scanned region for very large texts.
_MAX_SCAN_CHARS = 200_000


def _detect_repeated_lines(text: str, min_line_repeats: int) -> int | None:
    """Detect the same non-empty line repeated at the tail of the text.

    Blank lines between repeats are ignored, so both ``line\\nline`` and
    ``line\\n\\nline`` loops are caught. Lines that differ in content (e.g.
    markdown table rows or list items) never match.

    Returns:
        Offset just after the first line of the trailing run (so slicing at
        the offset keeps one instance), or None if no run is found.
    """
    lines = text.splitlines(keepends=True)
    starts: list[int] = []
    pos = 0
    for line in lines:
        starts.append(pos)
        pos += len(line)

    unit: str | None = None
    count = 0
    first_idx = -1
    for i in range(len(lines) - 1, -1, -1):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if unit is None:
            unit = stripped
            count = 1
            first_idx = i
        elif stripped == unit:
            count += 1
            first_idx = i
        else:
            break

    if unit is None or count < min_line_repeats:
        return None
    return starts[first_idx] + len(lines[first_idx])


def _detect_repeated_chunk(
    text: str, min_ngram_chars: int, min_repeats: int
) -> int | None:
    """Detect a chunk of >= min_ngram_chars repeated at the tail of the text.

    Finds the longest periodic suffix via the KMP prefix function of the
    reversed tail (O(n)); a trailing partial repeat (generation cut mid-unit)
    is covered by the periodicity check.

    Returns:
        Offset just after the first full unit of the periodic suffix (so
        slicing at the offset keeps one instance), or None.
    """
    tail = text[-_MAX_SCAN_CHARS:]
    base = len(text) - len(tail)
    min_len = min_ngram_chars * min_repeats
    n = len(tail)
    if n < min_len:
        return None

    rev = tail[::-1]
    pi = [0] * n
    k = 0
    for i in range(1, n):
        while k and rev[i] != rev[k]:
            k = pi[k - 1]
        if rev[i] == rev[k]:
            k += 1
        pi[i] = k

    # Prefix of rev of length L == suffix of text of length L; smallest period
    # is L - pi[L-1]. Scan longest-first so the full degenerate run is found.
    for i in range(n - 1, min_len - 2, -1):
        length = i + 1
        period = length - pi[i]
        if period >= min_ngram_chars and period * min_repeats <= length:
            return base + n - length + period
    return None


def detect_trailing_repetition(
    text: str,
    min_ngram_chars: int = DEFAULT_MIN_NGRAM_CHARS,
    min_repeats: int = DEFAULT_MIN_REPEATS,
    min_line_repeats: int = DEFAULT_MIN_LINE_REPEATS,
) -> int | None:
    """Detect degenerate repetition at the tail of a text.

    Combines two detectors:
    - Repeated line: the same non-empty line >= min_line_repeats times
      consecutively at the tail (catches short-unit loops). Preferred when
      it fires because its cut lands exactly on a line boundary.
    - Repeated chunk: a unit of >= min_ngram_chars chars repeated
      >= min_repeats times consecutively at the tail (catches long and
      multi-line loops, including a partial final repeat). Its cut may land
      on a rotation of the semantic unit.

    Legitimate repetition (markdown table rows or list items with differing
    content) is not flagged because the repeated units are not identical.

    Args:
        text: Text to inspect (typically VLM output).
        min_ngram_chars: Minimum repeated-chunk length in characters.
        min_repeats: Minimum consecutive chunk repeats.
        min_line_repeats: Minimum consecutive identical non-empty lines.

    Returns:
        Offset where the redundant tail starts (text[:offset] keeps exactly
        one instance of the repeated unit), or None if no degeneration.
    """
    if not text:
        return None
    line_offset = _detect_repeated_lines(text, min_line_repeats)
    if line_offset is not None:
        return line_offset
    return _detect_repeated_chunk(text, min_ngram_chars, min_repeats)


def truncate_degenerate_tail(
    text: str,
    *,
    context: str = "",
    stage: str = "",
) -> tuple[str, bool]:
    """Detect and truncate a degenerate repetition tail in VLM output.

    Returns:
        Tuple of (text, truncated). When truncated is True the redundant tail
        was removed (one instance of the repeated unit is kept) and the caller
        should skip persisting the response to cache so a clean retry on the
        next run is not poisoned.
    """
    if not text:
        return text, False
    offset = detect_trailing_repetition(text)
    if offset is None or offset <= 0 or offset >= len(text):
        return text, False
    logger.warning(
        "[{}] {}: VLM output degenerated into trailing repetition; "
        "truncating {} of {} chars (keeping one instance of the repeated unit)",
        context or "unknown",
        stage or "vlm",
        len(text) - offset,
        len(text),
    )
    return text[:offset].rstrip(), True
