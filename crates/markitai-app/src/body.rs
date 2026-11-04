//! A written Markdown file without its front matter: what the conversion's
//! `markdown` field held (desktop/DESIGN.md 3.3).

use markitai_core::output::split_frontmatter;

/// The block goes when the core's own reader takes it for front matter
/// (`split_frontmatter`: a `---` line, a YAML mapping, a `---` line). A
/// leading byte-order mark is not text, as when the core reads a text file,
/// and CRLF counts as one line break, so a file an editor saved again with
/// Windows line endings still loses its block. The body keeps its bytes.
pub(crate) fn body(text: &str) -> &str {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    if !text.contains("\r\n") {
        return split_frontmatter(text).1;
    }
    let unix = text.replace("\r\n", "\n");
    let consumed = unix.len() - split_frontmatter(&unix).1.len();
    let bytes = text.as_bytes();
    let mut at = 0;
    for _ in 0..consumed {
        at += if bytes[at..].starts_with(b"\r\n") {
            2
        } else {
            1
        };
    }
    &text[at..]
}
