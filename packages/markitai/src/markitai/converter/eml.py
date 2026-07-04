"""Native EML email converter (Python stdlib ``email`` + HTML pipeline).

Zero new dependencies: parsing uses ``email`` with ``policy.default`` (which
handles RFC 2047 header decoding and per-part charsets), HTML bodies are run
through the same webextract → markitdown HTML pipeline used for .html files,
and image attachments are embedded as base64 data URIs so they flow into the
standard ExtractedImage pipeline (assets extraction, compression, and LLM
vision/alt analysis).

Output shape mirrors markitdown's Outlook .msg converter (``# Email Message``
header block + ``## Content``) for consistency across email formats.
"""

from __future__ import annotations

import base64
import re
from email import message_from_bytes
from email.message import EmailMessage
from email.policy import default as _DEFAULT_POLICY
from pathlib import Path

from loguru import logger

from markitai.converter.base import (
    BaseConverter,
    ConvertResult,
    FileFormat,
    register_converter,
)

_HEADER_FIELDS = ("From", "To", "Cc", "Date", "Subject")

# Maximum nesting depth for message/rfc822 attachments (mirror xberg's
# recursive attachment extraction, capped at one level deep).
_MAX_NESTED_DEPTH = 1


def _strip_control_chars(value: str) -> str:
    """Remove control characters (CR/LF header-injection artifacts) and
    collapse whitespace."""
    cleaned = "".join(ch if ch.isprintable() or ch == " " else " " for ch in value)
    return " ".join(cleaned.split())


def _sanitize_header(value: str) -> str:
    """Sanitize a header value for safe markdown rendering.

    Strips control characters (so injected CR/LF can't fabricate extra
    header lines) and escapes angle brackets so values like
    ``Alice <alice@example.com>`` — or injected ``<script>`` tags — are
    rendered as literal text, never raw HTML.
    """
    return _strip_control_chars(value).replace("<", "\\<").replace(">", "\\>")


def _sanitize_alt_text(name: str) -> str:
    """Make an attachment filename safe inside ``![alt](...)``."""
    return re.sub(r"[\[\]()]", "_", _strip_control_chars(name))


def _format_size(num_bytes: int) -> str:
    """Human-readable attachment size."""
    if num_bytes >= 1024 * 1024:
        return f"{num_bytes / (1024 * 1024):.1f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes} B"


def _part_text(part: EmailMessage) -> str:
    """Decode a text part, tolerating unknown/broken charsets."""
    try:
        content = part.get_content()
        if isinstance(content, str):
            return content
        if isinstance(content, bytes):
            return content.decode("utf-8", errors="replace")
        return str(content)
    except (LookupError, UnicodeDecodeError, KeyError) as exc:
        logger.debug("[EmlConverter] charset decode fallback: {}", exc)
        payload = part.get_payload(decode=True)
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return ""


def _html_to_markdown(html: str, source: str) -> str:
    """Convert an HTML email body through the repo's HTML→MD pipeline.

    Same strategy as the .html file converter: webextract first, markitdown
    fallback when webextract output is unavailable or too thin.
    """
    try:
        from markitai.webextract import (
            extract_web_content,
            is_native_extraction_acceptable,
        )

        extracted = extract_web_content(html, source)
        if is_native_extraction_acceptable(extracted):
            return extracted.markdown
        logger.debug(
            "[EmlConverter] webextract output too short, falling back to markitdown"
        )
    except Exception as exc:
        logger.debug(
            "[EmlConverter] webextract failed, falling back to markitdown: {}", exc
        )

    import io

    from markitdown import MarkItDown, StreamInfo

    result = MarkItDown().convert_stream(
        io.BytesIO(html.encode("utf-8")),
        stream_info=StreamInfo(extension=".html", mimetype="text/html"),
    )
    return result.markdown


def _render_header_block(msg: EmailMessage) -> str:
    lines = []
    for field in _HEADER_FIELDS:
        raw = msg.get(field)
        if raw is None:
            continue
        value = _sanitize_header(str(raw))
        if value:
            lines.append(f"**{field}:** {value}")
    return "\n".join(lines)


def _render_body(msg: EmailMessage, source: str) -> str:
    """Render the message body: prefer text/html (via HTML→MD pipeline),
    fall back to text/plain as-is."""
    body_part = msg.get_body(preferencelist=("html", "plain"))
    if body_part is None:
        return ""
    text = _part_text(body_part)  # type: ignore[arg-type]
    if not text.strip():
        return ""
    if body_part.get_content_type() == "text/html":
        try:
            return _html_to_markdown(text, source).strip()
        except Exception as exc:
            logger.warning("[EmlConverter] HTML body conversion failed: {}", exc)
            return text.strip()
    return text.strip()


def _quote_block(markdown: str) -> str:
    """Render markdown as a blockquote (for nested messages)."""
    return "\n".join(
        f"> {line}" if line.strip() else ">" for line in markdown.splitlines()
    )


def _render_attachments(msg: EmailMessage, source: str, depth: int) -> str:
    """Render the attachments section.

    - image/* parts become base64 data URIs (standard image pipeline)
    - message/rfc822 parts are rendered inline as a quoted section
      (one level deep)
    - everything else is listed by name + size
    """
    sections: list[str] = []
    listing: list[str] = []

    for idx, part in enumerate(msg.iter_attachments()):
        ctype = part.get_content_type()
        filename = part.get_filename() or f"attachment_{idx}"

        if ctype == "message/rfc822" and depth < _MAX_NESTED_DEPTH:
            try:
                nested = part.get_payload(0)
                nested_md = _render_message(
                    nested,  # type: ignore[arg-type]
                    source=f"{source}#nested-{idx}",
                    depth=depth + 1,
                )
                sections.append(
                    f"### Attached message: {_sanitize_header(filename)}\n\n"
                    + _quote_block(nested_md)
                )
                continue
            except Exception as exc:
                logger.warning(
                    "[EmlConverter] failed to render nested message '{}': {}",
                    filename,
                    exc,
                )
                # fall through to plain listing

        payload = part.get_payload(decode=True)
        data = payload if isinstance(payload, bytes) else b""

        if ctype.startswith("image/") and data:
            alt = _sanitize_alt_text(filename)
            b64 = base64.b64encode(data).decode("ascii")
            listing.append(f"![{alt}](data:{ctype};base64,{b64})")
        else:
            listing.append(
                f"- {_sanitize_header(filename)} ({_format_size(len(data))})"
            )

    if listing:
        sections.insert(0, "\n\n".join(listing))
    if not sections:
        return ""
    return "## Attachments\n\n" + "\n\n".join(sections)


def _render_message(msg: EmailMessage, source: str, depth: int = 0) -> str:
    """Render a full message (headers + body + attachments) to markdown."""
    parts = ["# Email Message"]

    headers = _render_header_block(msg)
    if headers:
        parts.append(headers)

    body = _render_body(msg, source)
    parts.append("## Content\n\n" + body if body else "## Content")

    if depth < _MAX_NESTED_DEPTH:
        attachments = _render_attachments(msg, source, depth)
        if attachments:
            parts.append(attachments)
    else:
        # Depth cap reached: list attachment names only, no recursion.
        names = [
            f"- {_sanitize_header(part.get_filename() or f'attachment_{idx}')}"
            for idx, part in enumerate(msg.iter_attachments())
        ]
        if names:
            parts.append("## Attachments\n\n" + "\n".join(names))

    return "\n\n".join(parts)


@register_converter(FileFormat.EML)
class EmlConverter(BaseConverter):
    """Converter for RFC 822 .eml email files using the stdlib email parser."""

    supported_formats = [FileFormat.EML]

    def convert(
        self, input_path: Path, output_dir: Path | None = None
    ) -> ConvertResult:
        input_path = Path(input_path)
        logger.debug("[EmlConverter] Converting: {}", input_path.name)

        msg = message_from_bytes(input_path.read_bytes(), policy=_DEFAULT_POLICY)
        markdown = _render_message(
            msg,  # type: ignore[arg-type]
            source=f"file://{input_path.resolve()}",
        )

        metadata: dict = {
            "source": str(input_path),
            "format": "EML",
            "converter": "eml",
        }
        subject = msg.get("Subject")
        if subject:
            title = _strip_control_chars(str(subject))
            if title:
                metadata["title"] = title

        return ConvertResult(markdown=markdown, images=[], metadata=metadata)
