"""Text processing utilities for Markitai."""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any
from urllib.parse import quote

from pydantic import ValidationError

from markitai.utils.errors import SelfExplanatoryError
from markitai.utils.secret_redaction import redact_secrets


def markdown_image_reference(alt: str, path: str) -> str:
    """Build a CommonMark image reference safe for spaces and Unicode paths."""
    escaped_alt = alt.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")
    encoded_path = quote(path, safe="/._~-")
    return f"![{escaped_alt}]({encoded_path})"


def normalize_identifier_key(key: str) -> str:
    """Fold snake_case, camelCase and header-style spellings onto one form.

    ``apiKey``, ``api-key`` and ``API_KEY`` all become ``api_key``, so a
    secret-name rule can be written once and still match however the key
    was spelled. Used by config redaction and URL query redaction, which
    had grown identical private copies.
    """
    snake_key = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", key)
    snake_key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", snake_key)
    return re.sub(r"[^a-z0-9]+", "_", snake_key.lower()).strip("_")


_SECRET_KEY_WORDS = frozenset(
    {
        "authorization",
        "cookie",
        "cookies",
        "credential",
        "credentials",
        "passwd",
        "password",
        "secret",
    }
)


def is_secret_config_key(key: str) -> bool:
    """Whether a config key conventionally holds secret material.

    The rule ``markitai config show`` redacts with and the serve Edit config
    page masks with: an api key in any spelling, a token, or a word naming
    authorization, cookies, credentials, a password or a secret.
    """
    normalized = normalize_identifier_key(key)
    parts = set(normalized.split("_"))

    if normalized == "apikey" or normalized.endswith("_apikey"):
        return True
    if "api" in parts and "key" in parts:
        return True
    if "token" in parts:  # Deliberately does not match non-secret ``max_tokens``.
        return True
    return bool(parts & _SECRET_KEY_WORDS)


def is_header_map_config_key(key: str) -> bool:
    """Whether a config key names an HTTP header map (``extra_http_headers``)."""
    normalized = normalize_identifier_key(key)
    return normalized == "extra_http_headers" or normalized.endswith(
        "_extra_http_headers"
    )


def is_api_base_config_key(key: str) -> bool:
    """Whether a config key names an endpoint address (``api_base``)."""
    normalized = normalize_identifier_key(key)
    return normalized == "api_base" or normalized.endswith("_api_base")


def image_ref_pattern(name: str) -> str:
    """Regex matching an image reference to *name*, encoded or raw.

    ``markdown_image_reference`` writes asset destinations percent-encoded
    (spaces/CJK → %XX), so matching only the raw filename silently dropped
    alt-text updates for those names; match either form.
    """
    raw = re.escape(name)
    encoded = re.escape(quote(name, safe="/._~-"))
    alt = raw if raw == encoded else f"(?:{encoded}|{raw})"
    return rf"!\[[^\]]*\]\([^)]*{alt}\)"


def clean_control_characters(text: str, preserve_whitespace: bool = True) -> str:
    """Remove control characters from string.

    Control characters (ASCII 0x00-0x1F and 0x7F DEL) can cause JSON parsing
    errors when LLMs include them in their output. This function removes them
    while optionally preserving common whitespace characters.

    Args:
        text: Input text to clean
        preserve_whitespace: If True, preserve newline, carriage return, and tab
            characters. If False, remove all control characters.

    Returns:
        Text with control characters removed

    Example:
        >>> clean_control_characters("hello\\x00world")
        'helloworld'
        >>> clean_control_characters("line1\\nline2")
        'line1\\nline2'
    """
    if preserve_whitespace:
        # Keep \n (0x0A), \r (0x0D), \t (0x09); remove 0x00-0x1F and 0x7F
        return "".join(
            c for c in text if (ord(c) >= 32 and ord(c) != 0x7F) or c in "\n\r\t"
        )
    return "".join(c for c in text if 32 <= ord(c) != 0x7F)


# Pre-compiled patterns for repair_json_string
_JSON_CODEBLOCK_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def repair_json_string(text: str) -> str | None:
    """Attempt to repair common JSON syntax errors from LLM output.

    The single JSON repair primitive in the codebase. It belongs to the
    bottom rung of the structured-output staircase
    (``markitai.llm.structured``) — the one tier where a model hand-writes
    JSON into its answer text. Everything above that tier has the provider
    producing the JSON, so nothing there needs repairing.

    Tries multiple repair strategies in sequence:
    1. Parse as-is
    2. Extract from markdown code blocks
    3. Trim to JSON object/array boundaries
    4. Remove trailing commas
    5. Close unclosed brackets/braces (truncated output)

    Args:
        text: Raw text that should contain JSON

    Returns:
        Repaired JSON string, or None if repair is not possible
    """
    import json

    # Control characters an LLM emits mid-string break every strategy below
    text = clean_control_characters(text)

    # Strategy 1: Try as-is
    try:
        json.loads(text)
        return text
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from markdown code blocks
    match = _JSON_CODEBLOCK_RE.search(text)
    if match:
        candidate = match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            text = candidate

    # Strategy 3: Trim surrounding prose to the outermost JSON boundaries.
    # Arrays count: a model asked for a list often answers with a bare one.
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        return None
    text = text[min(starts) :]
    end = max(text.rfind("}"), text.rfind("]"))
    if end >= 0:
        candidate = text[: end + 1]
        try:
            json.loads(candidate)
            return candidate
        except (json.JSONDecodeError, ValueError):
            text = candidate

    # Strategy 4: Remove trailing commas before } or ]
    repaired = _TRAILING_COMMA_RE.sub(r"\1", text)
    try:
        json.loads(repaired)
        return repaired
    except (json.JSONDecodeError, ValueError):
        text = repaired

    # Strategy 5: Close unclosed brackets/braces (handles truncated output)
    open_braces = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")
    if open_brackets > 0 or open_braces > 0:
        # Try closing with a quote first (in case truncated inside a string value)
        for prefix in ['"', ""]:
            closed = text + prefix
            if open_brackets > 0:
                closed += "]" * open_brackets
            if open_braces > 0:
                closed += "}" * open_braces
            try:
                json.loads(closed)
                return closed
            except (json.JSONDecodeError, ValueError):
                continue

    return None


def format_error_message(error: Any, max_length: int = 200) -> str:
    """Format an exception for user-friendly logging.

    Extracts the core error message without full traceback.
    For LiteLLM and other chained exceptions, extracts the most relevant message.

    The exception class name is prefixed (``ValueError: ...``) because for an
    unexpected failure the type is the most useful diagnostic we have.
    :class:`~markitai.utils.errors.SelfExplanatoryError` subclasses opt out:
    their messages already name the problem and the fix, so the prefix is
    noise (see that module for the rationale).

    Args:
        error: Exception object or any value
        max_length: Maximum message length before truncation

    Returns:
        A concise, user-friendly error message without traceback
    """
    if error is None:
        return "Unknown error"

    # pydantic's ValidationError.__str__ is a multi-line block per field
    # (message, input value repr, and a docs URL) that survives the
    # newline-collapse/truncate below as a confusing, often mid-field-cut
    # fragment. Summarize as "N error(s) for Model: field (type), ...".
    if isinstance(error, ValidationError):
        errors = error.errors()
        fields = [
            f"{'.'.join(str(x) for x in err.get('loc', ())) or '?'} "
            f"({err.get('type', 'invalid')})"
            for err in errors[:5]
        ]
        summary = ", ".join(fields)
        if len(errors) > 5:
            summary += f", +{len(errors) - 5} more"
        model_name = getattr(error, "title", None) or "model"
        return f"ValidationError: {error.error_count()} error(s) for {model_name}: {summary}"

    # Get the exception class name
    exc_type = type(error).__name__

    # For BaseException and its subclasses, extract just the message
    if isinstance(error, BaseException):
        # Get the direct message (args[0] if available)
        if error.args:
            msg = str(error.args[0])
        else:
            msg = str(error)

        # Self-explanatory errors own their wording: keep it verbatim and skip
        # the root-cause walk below, which would happily swap an actionable
        # install hint for the terser "No module named 'x'" it was raised from.
        self_explanatory = isinstance(error, SelfExplanatoryError)

        if not self_explanatory:
            # For chained exceptions, find the root cause message.
            # Only follow __cause__ (explicit 'raise X from Y') — never __context__
            # (implicit chaining), which often leads to wrapper exceptions like
            # RetryError whose messages are opaque (e.g., "<Future at 0x...>").
            root_error = error
            while hasattr(root_error, "__cause__") and root_error.__cause__ is not None:
                root_error = root_error.__cause__

            # If we found a root cause, use its message
            if root_error is not error:
                root_type = type(root_error).__name__
                if root_error.args:
                    root_msg = str(root_error.args[0])
                else:
                    root_msg = str(root_error)
                # Use root message if it's more informative
                if root_msg and len(root_msg) < len(msg):
                    msg = root_msg
                    exc_type = root_type
                    # Adopting a self-explanatory cause adopts its rendering too
                    self_explanatory = isinstance(root_error, SelfExplanatoryError)

        # Clean up the message - remove traceback if embedded
        if "Traceback (most recent call last):" in msg:
            # Keep only the part before the traceback
            msg = msg.split("Traceback (most recent call last):")[0].strip()

        # Never print a credential an error echoes (a provider's 401)
        msg = redact_secrets(msg)

        # Replace newlines with spaces to keep log output on single line
        msg = re.sub(r"\s*\n\s*", " ", msg)
        # Collapse multiple spaces into one
        msg = re.sub(r"\s{2,}", " ", msg).strip()

        # Remove common prefixes that duplicate exception type
        for prefix in [f"{exc_type}: ", "Error: ", "Exception: "]:
            if msg.startswith(prefix):
                msg = msg[len(prefix) :]

        # Truncate if too long
        if len(msg) > max_length:
            msg = msg[:max_length] + "..."

        # Return formatted message with exception type. A self-explanatory
        # error with no message left falls back to its type — an empty string
        # would be worse than a noisy one.
        if msg:
            return msg if self_explanatory else f"{exc_type}: {msg}"
        return exc_type

    # For non-exceptions, just convert to string
    result = str(error)
    if len(result) > max_length:
        result = result[:max_length] + "..."
    return result


def preview_items_for_log(items: Sequence[str], max_items: int = 3) -> str:
    """Format a compact preview for list-like log fields.

    Args:
        items: Items to preview
        max_items: Maximum number of items to render inline

    Returns:
        A comma-separated preview with an optional ``+N more`` suffix
    """
    if not items:
        return ""

    preview = ", ".join(items[:max_items])
    remaining = len(items) - max_items
    if remaining > 0:
        return f"{preview}, +{remaining} more"
    return preview


def clean_residual_placeholders(content: str) -> str:
    """Remove residual MARKITAI placeholders from content.

    Some placeholders may leak into the output, especially in image references
    like `![](__MARKITAI_FILE_ASSET__)`. This function cleans them up.

    Args:
        content: Markdown content with potential residual placeholders

    Returns:
        Cleaned content
    """
    if "__MARKITAI_" not in content:
        return content

    # Remove standalone placeholder lines
    content = re.sub(r"^__MARKITAI_[A-Z_]+_?\d*__\s*$", "", content, flags=re.MULTILINE)

    # Remove image references with placeholder URLs
    content = re.sub(r"!\[[^\]]*\]\(__MARKITAI_[A-Z_]+_?\d*__\)\s*\n?", "", content)

    # Remove any other inline placeholders
    content = re.sub(r"__MARKITAI_[A-Z_]+_?\d*__", "", content)

    return content


_TABLE_DELIMITER_ROW = re.compile(r"^\s*\|(?:\s*:?-{3,}:?\s*\|)+\s*$")
_FENCE_OPEN = re.compile(r"^\s{0,3}(`{3,}|~{3,})")


def drop_stray_table_delimiters(content: str) -> str:
    """Remove delimiter rows (``| --- | --- |``) a table already has.

    A pipe table has exactly one delimiter row, right after its header. A
    model rewriting a table sometimes repeats it, and the repeat renders as
    a data row of dashes. Lines inside fenced code are left alone.
    """
    if "---" not in content or "|" not in content:
        return content
    lines = content.split("\n")
    kept: list[str] = []
    fence: str | None = None
    row = -1  # index of the current line within its table block; -1 outside
    for line in lines:
        opener = _FENCE_OPEN.match(line)
        if fence is not None:
            if (
                opener
                and opener.group(1)[0] == fence[0]
                and len(opener.group(1)) >= len(fence)
            ):
                fence = None
            kept.append(line)
            continue
        if opener:
            fence = opener.group(1)
            row = -1
            kept.append(line)
            continue
        if not line.lstrip().startswith("|"):
            row = -1
            kept.append(line)
            continue
        row += 1
        if row >= 2 and _TABLE_DELIMITER_ROW.match(line):
            row -= 1
            continue
        kept.append(line)
    return "\n".join(kept)


# A line or paragraph separator, vertical tab, NEL or lone CR in a table row:
# a renderer takes it for a line end and the table ends there (U+2028 runs
# every later row together as one paragraph), so in a cell it is a <br>
_ROW_BREAK = re.compile("[\r\v\f\x1c-\x1e\x85\u2028\u2029]")
_CELL_EDGE = re.compile(r"(?<!\\)\|")


def _cell_with_breaks(cell: str) -> str:
    parts = [part.strip() for part in _ROW_BREAK.split(cell)]
    joined = "<br>".join(part for part in parts if part)
    return f" {joined} " if joined else " "


def table_cell_breaks(content: str) -> str:
    """Write each line break left inside a table row as ``<br>``.

    The table writers already do (``webextract.table_cells``); this repairs
    what still arrives, a model's answer above all: U+2028/U+2029, a
    vertical tab, NEL or a lone CR in a row. Runs become one ``<br>``, none
    stands at a cell's edges, and a break after the row's last ``|`` is
    dropped. Lines inside fenced code are left alone.
    """
    if not _ROW_BREAK.search(content) or "|" not in content:
        return content
    lines = content.split("\n")
    fence: str | None = None
    for index, line in enumerate(lines):
        opener = _FENCE_OPEN.match(line)
        if fence is not None:
            if (
                opener
                and opener.group(1)[0] == fence[0]
                and len(opener.group(1)) >= len(fence)
            ):
                fence = None
            continue
        if opener:
            fence = opener.group(1)
            continue
        ending = "\r" if line.endswith("\r") else ""
        row = line.removesuffix("\r")
        if not row.lstrip().startswith("|") or not _ROW_BREAK.search(row):
            continue
        segments = _CELL_EDGE.split(row)
        first, *cells, last = segments
        first = _ROW_BREAK.sub("", first)
        cells = [
            _cell_with_breaks(cell) if _ROW_BREAK.search(cell) else cell
            for cell in cells
        ]
        if _ROW_BREAK.search(last):
            last = _cell_with_breaks(last).rstrip() if last.strip() else ""
        lines[index] = "|".join([first, *cells, last]) + ending
    return "\n".join(lines)


def normalize_markdown_whitespace(content: str) -> str:
    """Normalize whitespace in markdown content.

    - Fix malformed image references (double alt text, empty paths)
    - Ensure headers (#) have one blank line before and after
    - Merge 3+ consecutive blank lines into 2 blank lines
    - Ensure consistent line endings
    - Strip trailing whitespace from lines

    Note: Header normalization is markdown-aware and correctly handles
    nested fenced code blocks (e.g., ```` containing ```).
    Indented code blocks (4-space) are not detected.

    Args:
        content: Markdown content to normalize

    Returns:
        Normalized content
    """
    # Fix malformed image references first
    content = fix_malformed_image_refs(content)

    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in content.split("\n")]

    # Normalize header spacing (markdown-aware, skip code blocks)
    result_lines: list[str] = []
    code_block_char: str | None = None  # '`' or '~'
    code_block_count: int = 0  # Number of fence chars that opened the block

    for i, line in enumerate(lines):
        # Check for code fence (``` or ~~~, possibly more)
        fence_match = (
            re.match(r"^(`{3,}|~{3,})", line)
            if line.startswith(("```", "~~~"))
            else None
        )
        if fence_match:
            fence = fence_match.group(1)
            fence_char = fence[0]
            fence_count = len(fence)

            if code_block_char is None:
                # Start of code block
                code_block_char = fence_char
                code_block_count = fence_count
            elif fence_char == code_block_char and fence_count >= code_block_count:
                # End of code block (same char, count >= opening)
                code_block_char = None
                code_block_count = 0
            # else: fence inside code block, ignore

        # Only process headers and slide comments outside code blocks
        in_code_block = code_block_char is not None
        if in_code_block or not line.startswith(("#", "<!--")):
            result_lines.append(line)
            continue

        # ATX headers: 1-6 # followed by space or end of line
        # Excludes: #hashtag, #123, #! (shebang)
        is_atx_header = line.startswith("#") and bool(re.match(r"^#{1,6}(\s|$)", line))

        # Slide comments: <!-- Slide number: X -->
        is_slide_comment = line.startswith("<!--") and bool(
            re.match(r"^<!--\s*Slide\s+(number:\s*)?\d+\s*-->", line)
        )

        needs_spacing = is_atx_header or is_slide_comment

        if not in_code_block and needs_spacing:
            # Add blank line before if needed
            if result_lines and result_lines[-1] != "":
                result_lines.append("")
            result_lines.append(line)
            # Add blank line after if next line is not empty
            if i + 1 < len(lines) and lines[i + 1] != "":
                result_lines.append("")
        else:
            result_lines.append(line)

    content = "\n".join(result_lines)

    # Merge 3+ consecutive blank lines into 2 (keep one blank line between blocks)
    if "\n\n\n" in content:
        content = re.sub(r"\n{3,}", "\n\n", content)

    # Ensure single newline at end
    return content.strip() + "\n"


def fix_malformed_image_refs(content: str) -> str:
    """Fix malformed image references in markdown.

    Handles common LLM output errors:
    1. Double alt text: `![alt1]![alt2](path)` -> `![alt2](path)`
    2. Empty path: `![alt](assets/)` or `![alt]()` -> removed
    3. Extra closing parenthesis: `![alt](path))` -> `![alt](path)`

    Args:
        content: Markdown content with potential malformed image refs

    Returns:
        Content with fixed image references
    """
    if "![" not in content:
        return content

    # Fix double alt text: ![alt1]![alt2](path) -> ![alt2](path)
    # This happens when LLM generates two consecutive alt texts
    content = re.sub(
        r"!\[[^\]]*\](!\[[^\]]*\]\([^)]+\))",
        r"\1",
        content,
    )

    # Remove images with empty paths: ![alt](assets/) or ![alt]()
    content = re.sub(r"!\[[^\]]*\]\((?:\.markitai/assets/)?\)\s*\n?", "", content)

    # Fix extra closing parenthesis: ![alt](path)) -> ![alt](path)
    content = re.sub(r"(!\[[^\]]*\]\([^)]+\))\)+", r"\1", content)

    return content


def extract_asset_image_names(markdown: str) -> list[str]:
    """Extract basenames of ``.markitai/assets/`` image refs from markdown.

    Returns names in document order, deduplicated. This is the reliable way
    to discover which asset files belong to a document: refs are written by
    the converter itself, so they always match on-disk names even when the
    extractor sanitizes the source filename (e.g. pymupdf4llm rewrites
    spaces and parentheses, so a prefix glob on the input name matches
    nothing).

    Args:
        markdown: Markdown content with image references

    Returns:
        List of asset file basenames referenced in the markdown
    """
    from urllib.parse import unquote

    from markitai.constants import ASSETS_REL_PATH

    pattern = re.compile(rf"!\[[^\]]*\]\({re.escape(ASSETS_REL_PATH)}[/\\]([^)]+)\)")
    seen: set[str] = set()
    names: list[str] = []
    for match in pattern.finditer(markdown):
        name = unquote(match.group(1).strip())
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def fix_broken_markdown_links(content: str) -> str:
    """Fix broken markdown links where text and URL are split by newlines.

    Common pattern from web scraping:
    [Title text

    Description text](/url)

    Should become: [Title text](/url)

    Args:
        content: Markdown content with potentially broken links

    Returns:
        Content with fixed links
    """
    # Apply fix iteratively until no more changes
    prev_content = ""
    while prev_content != content:
        prev_content = content
        content = _join_split_link_texts(content)

    return content


def _join_split_link_texts(content: str) -> str:
    r"""One pass of :func:`fix_broken_markdown_links`, in linear time.

    What ``re.sub(r"\[([^\]]*?)\n+([^\]]*?)\]\(([^)]+)\)", "[\1](\3)")``
    did (group 1 stripped), without its backtracking: from each ``[`` the
    pattern tried every line break as the split and ran to the end of the
    text from each, so a long document with an unclosed ``[`` (an OCR'd
    book) took most of a minute. A link text runs from a ``[`` to the first
    ``]`` after it, so of the ``[`` before one ``]`` only the first can
    start a match, and it matches when a line break comes before that ``]``
    and ``(url)`` right after it.
    """
    parts: list[str] = []
    copied = pos = 0
    while (start := content.find("[", pos)) >= 0:
        close = content.find("]", start + 1)
        if close < 0:
            break
        newline = content.find("\n", start + 1, close)
        end = content.find(")", close + 2) if newline >= 0 else -1
        if content.startswith("(", close + 1) and end > close + 2:
            first_line = content[start + 1 : newline].strip()
            parts += [
                content[copied:start],
                f"[{first_line}]({content[close + 2 : end]})",
            ]
            copied = pos = end + 1
        else:
            pos = close + 1
    parts.append(content[copied:])
    return "".join(parts)


def dedupe_paragraphs(content: str, min_length: int = 100) -> str:
    """Remove duplicate paragraphs from content.

    Useful for cleaning browser-fetched content where the same text
    may appear multiple times (e.g., Twitter/X pages with repeated
    content in og:title, aria-label, and main content).

    Args:
        content: Markdown content with potential duplicate paragraphs
        min_length: Minimum paragraph length to consider for deduplication.
            Shorter paragraphs are always kept (to avoid removing headers, etc.)

    Returns:
        Content with duplicate paragraphs removed (first occurrence kept)
    """
    # Split by double newlines (paragraph separator)
    paragraphs = re.split(r"\n\n+", content)

    seen_paragraphs: set[str] = set()
    result_paragraphs: list[str] = []

    for para in paragraphs:
        para_stripped = para.strip()

        # Skip empty paragraphs
        if not para_stripped:
            continue

        # Normalize for comparison: collapse whitespace
        normalized = re.sub(r"\s+", " ", para_stripped)

        # Short paragraphs: always keep (headers, short lines, etc.)
        if len(normalized) < min_length:
            result_paragraphs.append(para)
            continue

        # Long paragraphs: dedupe
        if normalized not in seen_paragraphs:
            seen_paragraphs.add(normalized)
            result_paragraphs.append(para)
        # else: skip duplicate

    return "\n\n".join(result_paragraphs)


def dedupe_long_text_blocks(content: str, min_length: int = 50) -> str:
    """Remove duplicate long text blocks from content.

    More aggressive deduplication for social media content where the same
    long text appears multiple times in different formatting contexts
    (e.g., Twitter aria-label, og:title, and main content).

    This function:
    1. Finds all "long text blocks" (continuous text >= min_length chars)
    2. Removes duplicate occurrences, keeping the first one
    3. Also handles cases where text is prefixed with usernames/metadata

    Args:
        content: Markdown content with potential duplicate text blocks
        min_length: Minimum text length to consider for deduplication

    Returns:
        Content with duplicate text blocks removed
    """
    lines = content.split("\n")
    result_lines: list[str] = []
    seen_texts: list[str] = []  # Use list to preserve order for substring matching

    for line in lines:
        line_stripped = line.strip()

        # Skip short lines, markdown syntax, or empty lines
        if not line_stripped or len(line_stripped) < min_length:
            result_lines.append(line)
            continue

        # Skip lines that are primarily markdown syntax
        if line_stripped.startswith(("#", "!", "[", "|", "-", "*", ">", "```", "<!--")):
            result_lines.append(line)
            continue

        # Normalize text for comparison
        # Remove leading username patterns (Twitter/X format)
        normalized = re.sub(r"^[A-Za-z\s]+@\w+\s+", "", line_stripped)
        # Remove @ mentions at start
        normalized = re.sub(r"^@\w+\s*", "", normalized)
        # Remove timestamps and metrics
        normalized = re.sub(r"\d+:\d+\s*(AM|PM|am|pm)?\s*·?\s*", "", normalized)
        normalized = re.sub(
            r"\d+\s*(replies|reposts|likes|views|bookmarks)[,\s]*",
            "",
            normalized,
            flags=re.I,
        )
        normalized = re.sub(r"\s+", " ", normalized).strip()

        # If still long enough after normalization
        if len(normalized) >= min_length:
            # Check if this text is a duplicate or substring of seen text
            is_duplicate = False
            for seen in seen_texts:
                # Check both directions: new is in seen, or seen is in new
                if normalized in seen or seen in normalized:
                    is_duplicate = True
                    break

            if not is_duplicate:
                seen_texts.append(normalized)
                result_lines.append(line)
            # else: skip duplicate
        else:
            result_lines.append(line)

    return "\n".join(result_lines)


def page_ranges(pages: Iterable[int]) -> str:
    """Page numbers as ranges: ``"3–5, 9"`` for pages 3, 4, 5 and 9
    (ascending, an en dash, no repeats), as every page list in a warning
    reads (workbench contract v1.16)."""
    numbers = sorted(set(pages))
    parts: list[str] = []
    index = 0
    while index < len(numbers):
        end = index
        while end + 1 < len(numbers) and numbers[end + 1] == numbers[end] + 1:
            end += 1
        first, last = numbers[index], numbers[end]
        parts.append(str(first) if first == last else f"{first}–{last}")
        index = end + 1
    return ", ".join(parts)
