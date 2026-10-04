use crate::{Asset, Document, Error, Result};
use anydoc::model::{Block, CellSlot, ImageSource, Inline, LinkTarget};
use std::collections::BTreeSet;

#[path = "office.rs"]
mod office;
pub(crate) use office::extract_presentation_count;
#[path = "native/compound.rs"]
mod compound;
#[path = "native/line.rs"]
mod line;
use line::{Line, Literal, Place};
#[cfg(test)]
#[path = "native/office_content_tests.rs"]
mod office_content_tests;
#[path = "office_meta.rs"]
mod office_meta;
#[path = "pdf.rs"]
pub(super) mod pdf;
#[cfg(test)]
#[path = "native/slides_tests.rs"]
mod slides_tests;

/// Adjacent text runs of one style as one run. A reader can split runs per
/// character (RTF `\\u` escapes) or wherever the source splits them (Word's
/// revision marks); rendered apart, a bold word becomes `**a****b**`, which
/// breaks the word and the joining of Arabic-script letters.
fn merged_runs(values: &[Inline]) -> std::borrow::Cow<'_, [Inline]> {
    let splits = values.windows(2).any(|pair| {
        matches!(pair, [Inline::Text { style: a, .. }, Inline::Text { style: b, .. }] if a == b)
    });
    if !splits {
        return std::borrow::Cow::Borrowed(values);
    }
    let mut merged: Vec<Inline> = Vec::with_capacity(values.len());
    for value in values {
        if let (
            Some(Inline::Text { text, style }),
            Inline::Text {
                text: next,
                style: next_style,
            },
        ) = (merged.last_mut(), value)
            && style == next_style
        {
            text.push_str(next);
            continue;
        }
        merged.push(value.clone());
    }
    std::borrow::Cow::Owned(merged)
}

/// Whether a block shows anything: text, an image, a rule, code.
fn has_content(block: &Block) -> bool {
    match block {
        Block::Heading { content, .. } | Block::Paragraph(content) => {
            content.iter().any(|inline| match inline {
                Inline::Anchor(_) | Inline::LineBreak => false,
                Inline::Text { text, .. } => !text.trim().is_empty(),
                Inline::Link { content, .. } => !anydoc::model::inlines_to_plain_text(content)
                    .trim()
                    .is_empty(),
                _ => true,
            })
        }
        Block::List(list) => list
            .items
            .iter()
            .any(|item| item.blocks.iter().any(has_content)),
        Block::BlockQuote(blocks) => blocks.iter().any(has_content),
        Block::Table(table) => table.grid.iter().flatten().any(
            |slot| matches!(slot, CellSlot::Origin(cell) if cell.blocks.iter().any(has_content)),
        ),
        _ => true,
    }
}

fn conversion_error(error: anydoc::ConvertError) -> Error {
    Error::Conversion(format!("Native document conversion failed: {error}"))
}

fn conversion_error_for_format(error: anydoc::ConvertError, format: anydoc::Format) -> Error {
    let hint = if format == anydoc::Format::Excel
        && matches!(
            &error,
            anydoc::ConvertError::ResourceLimit {
                limit: "max_xml_nodes",
                ..
            }
        ) {
        "; this workbook has an XML part larger than the 2,000,000-node reading limit. Export the needed sheets as CSV, or split the workbook into smaller XLSX files and convert them separately"
    } else {
        ""
    };
    Error::Conversion(format!("Native document conversion failed: {error}{hint}"))
}

/// Text escaped without knowing what stands around it, as the PDF and PPTX
/// writers use it: every `\`, `*`, `_`, `[`, `]` and backtick, and `<` or `&`
/// that would open an HTML tag or an entity. The document renderer escapes in
/// context instead (`line.rs`).
fn escape(text: &str) -> String {
    let mut output = String::new();
    for (index, ch) in text.char_indices() {
        let rest = &text[index + ch.len_utf8()..];
        // Text that opens an HTML tag, comment or entity is raw HTML to a
        // Markdown renderer, which hides the `<int>` of `std::vector<int>`
        // or turns a literal `&copy;` into a symbol.
        let html = match ch {
            '<' => rest
                .chars()
                .next()
                .is_some_and(|next| next.is_ascii_alphabetic() || matches!(next, '/' | '!' | '?')),
            '&' => {
                rest.starts_with('#')
                    || rest
                        .find(|c: char| !c.is_ascii_alphanumeric())
                        .is_some_and(|end| end > 0 && rest[end..].starts_with(';'))
            }
            _ => false,
        };
        if html || matches!(ch, '\\' | '*' | '_' | '[' | ']' | '`') {
            output.push('\\');
        }
        output.push(ch);
    }
    output
}

/// A paragraph's lines with the mark that would make Markdown read a line as
/// a block of its own written as text: an ATX heading (`# text`), a bullet
/// (`- item`, `+ item`, `* item`), an ordered item (`1. item`, `1) item`), a
/// quote (`> text`), a thematic break (`---`, `***`, `___`), a code fence
/// (```` ``` ````, `~~~`) or a link reference definition (`[1]: …`), up to
/// three spaces in; on a line after the first, also a setext underline (`==`,
/// `--`) or a table's delimiter row (`| --- |`), which would turn the line
/// above into a heading or a table header. A document's own line starting so
/// (a shell comment, a dash before a remark, a year and a full stop) is text
/// in its source. A line indented four spaces or more is a code block already
/// and stays as it is.
fn literal_heading_marks(text: &str) -> String {
    text.split('\n')
        .enumerate()
        .map(|(number, line)| {
            let indent = line.len() - line.trim_start_matches(' ').len();
            let rest = &line[indent..];
            if indent > 3 {
                return line.to_owned();
            }
            let ends = |after: &str| after.is_empty() || after.starts_with([' ', '\t']);
            let hashes = rest.bytes().take_while(|&b| b == b'#').count();
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            let heading = (1..=6).contains(&hashes) && ends(&rest[hashes..]);
            let bullet = rest.starts_with(['-', '+', '*']) && ends(&rest[1..]);
            let rule = ['-', '*', '_'].into_iter().any(|mark| {
                rest.matches(mark).count() >= 3
                    && rest.chars().all(|c| c == mark || c == ' ' || c == '\t')
            });
            let ordered = (1..=9).contains(&digits)
                && rest[digits..].starts_with(['.', ')'])
                && ends(&rest[digits + 1..]);
            let fence = rest.starts_with("~~~")
                || (rest.starts_with("```") && !rest.trim_start_matches('`').contains('`'));
            let definition = defines_link(rest);
            let shown = rest.trim_end_matches([' ', '\t']);
            let underline = number > 0
                && !shown.is_empty()
                && (shown.chars().all(|c| c == '=') || shown.chars().all(|c| c == '-'));
            let delimiter_row = number > 0
                && shown.contains('|')
                && shown.contains('-')
                && shown
                    .chars()
                    .all(|c| matches!(c, '|' | '-' | ':' | ' ' | '\t'));
            // The marks to escape, as byte offsets in `rest`. Where the mark is
            // an emphasis or code delimiter (`***`, `___`, `~~~`, ```` ``` ````),
            // all of it: the rest of the run would still pair with a
            // delimiter elsewhere in the paragraph (`\___` leaves `__`).
            let first = rest.chars().next().unwrap_or(' ');
            let marks: Vec<usize> = if (rule || fence) && matches!(first, '*' | '_' | '~' | '`') {
                if rule {
                    rest.match_indices(first).map(|(at, _)| at).collect()
                } else {
                    (0..rest.len() - rest.trim_start_matches(first).len()).collect()
                }
            } else if heading
                || bullet
                || rule
                || fence
                || definition
                || underline
                || delimiter_row
                || rest.starts_with('>')
            {
                vec![0]
            } else if ordered {
                vec![digits]
            } else {
                Vec::new()
            };
            if marks.is_empty() {
                return line.to_owned();
            }
            let mut written = line[..indent].to_owned();
            let mut from = 0;
            for at in marks {
                written.push_str(&rest[from..at]);
                written.push('\\');
                from = at;
            }
            written.push_str(&rest[from..]);
            written
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Whether a line reads as a link reference definition (`[label]: …`): a
/// label of at least one character, without an unescaped bracket, closed by
/// `]:`. A footnote's (`[^1]: …`) is the renderer's own or an EPUB's
/// written mark; text that would start one has its bracket escaped already.
fn defines_link(line: &str) -> bool {
    let Some(label) = line
        .strip_prefix('[')
        .filter(|label| !label.starts_with('^'))
    else {
        return false;
    };
    let bytes = label.as_bytes();
    let mut at = 0;
    while at < bytes.len() {
        match bytes[at] {
            b'\\' => at += 2,
            b'[' => return false,
            b']' => return at > 0 && bytes.get(at + 1) == Some(&b':'),
            _ => at += 1,
        }
    }
    false
}

fn destination(value: &str) -> String {
    value
        .replace(' ', "%20")
        .replace('(', "%28")
        .replace(')', "%29")
        .replace('<', "%3C")
        .replace('>', "%3E")
        .replace(['\r', '\n'], "")
}

fn image_extension(mime: &str, origin: &str) -> String {
    let known = match mime {
        "image/png" => "png",
        "image/jpeg" => "jpg",
        "image/gif" => "gif",
        "image/svg+xml" => "svg",
        "image/webp" => "webp",
        "image/bmp" => "bmp",
        "image/tiff" => "tiff",
        "image/avif" => "avif",
        _ => "",
    };
    if !known.is_empty() {
        return known.into();
    }
    std::path::Path::new(origin)
        .extension()
        .and_then(|s| s.to_str())
        .filter(|ext| ext.len() <= 12 && ext.chars().all(|c| c.is_ascii_alphanumeric()))
        .unwrap_or("bin")
        .to_ascii_lowercase()
}

struct Renderer<'a> {
    asset_names: &'a [String],
    merged_cells: bool,
    /// Whether the blocks being written are a table cell's (see
    /// [`Renderer::cell_text`]): inline content only, its lines joined with
    /// `<br>`.
    in_cell: bool,
    anchors: BTreeSet<String>,
    extension: &'a str,
}

fn references(blocks: &[Block], anchors: &mut BTreeSet<String>, assets: &mut BTreeSet<usize>) {
    fn inlines(values: &[Inline], anchors: &mut BTreeSet<String>, assets: &mut BTreeSet<usize>) {
        for value in values {
            if let Inline::Link { content, target } = value {
                if let LinkTarget::Anchor(anchor) = target {
                    anchors.insert(anchor.clone());
                }
                inlines(content, anchors, assets);
            }
            if let Inline::Image {
                source: ImageSource::Asset(id),
                ..
            } = value
            {
                assets.insert(id.0);
            }
        }
    }
    for block in blocks {
        match block {
            Block::Heading { content, .. } | Block::Paragraph(content) => {
                inlines(content, anchors, assets)
            }
            Block::List(list) => {
                for item in &list.items {
                    references(&item.blocks, anchors, assets);
                }
            }
            Block::BlockQuote(blocks) => references(blocks, anchors, assets),
            Block::Table(table) => {
                for row in &table.grid {
                    for slot in row {
                        if let CellSlot::Origin(cell) = slot {
                            references(&cell.blocks, anchors, assets);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

fn heading_without_bold(content: &[Inline]) -> Vec<Inline> {
    content
        .iter()
        .map(|inline| match inline {
            Inline::Text { text, style } => {
                let mut style = *style;
                style.bold = false;
                Inline::Text {
                    text: text.clone(),
                    style,
                }
            }
            Inline::Link { content, target } => Inline::Link {
                content: heading_without_bold(content),
                target: target.clone(),
            },
            other => other.clone(),
        })
        .collect()
}

/// Whether CommonMark counts a character as punctuation for deciding if an
/// emphasis marker can open or close: any Unicode punctuation or symbol.
fn is_punctuation(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    static PUNCTUATION: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    PUNCTUATION
        .get_or_init(|| regex::Regex::new(r"^[\p{P}\p{S}]$").expect("a fixed pattern"))
        .is_match(c.encode_utf8(&mut [0; 4]))
}

/// The first character a rendered inline starts with, as far as it matters to
/// the emphasis around the one before it: whitespace, or something that is
/// punctuation (a marker, a bracket, an opening `<`).
fn leading_char(value: &Inline) -> Option<char> {
    match value {
        Inline::Text { text, style } => {
            let first = text.chars().next()?;
            if first.is_whitespace() {
                Some(first)
            } else if style.code || style.bold || style.italic || style.strike || style.underline {
                Some('*')
            } else {
                Some(first)
            }
        }
        Inline::LineBreak => Some(' '),
        _ => Some('['),
    }
}

/// Splits an emphasised run's text into the punctuation to leave outside its
/// markers at either end, and the text between. CommonMark lets a marker open
/// emphasis only where the character after it is not punctuation, unless the
/// one before it is whitespace or punctuation, and close it only under the
/// mirror rule. `**（注意）**后续` in Chinese text, whose words have no spaces
/// around them, opens and closes nothing and shows its asterisks; the
/// punctuation written outside (`（**注意**）后续`) does. Only the ends beside a
/// letter or digit move, so `**Note:** text` is unchanged. Whitespace the
/// moved punctuation leaves at an end of the text moves out with it, since a
/// marker beside whitespace neither opens nor closes (`**< **|` would show
/// its asterisks). A run of nothing but punctuation and whitespace gets no
/// emphasis, and comes back as the leading part.
fn emphasis_edges(text: &str, before: Option<char>, after: Option<char>) -> (&str, &str, &str) {
    let word = |c: Option<char>| c.is_some_and(|c| !c.is_whitespace() && !is_punctuation(c));
    let mut start = 0;
    if word(before) {
        start = text.len() - text.trim_start_matches(is_punctuation).len();
    }
    let mut end = text.len();
    if word(after) {
        end = text[start..].trim_end_matches(is_punctuation).len() + start;
    }
    let core = text[start..end].trim();
    if core.is_empty() {
        return (text, "", "");
    }
    let start = start + (text[start..end].len() - text[start..end].trim_start().len());
    let end = start + core.len();
    (&text[..start], core, &text[end..])
}

/// What each line break among `values` is written as, as an HTML page's
/// `<br>` is: the first break of a run of them (with nothing `shows` between)
/// carries the run and the others are `None`. In running text one break is a
/// hard break and two or more end the paragraph (`<br><br>` in a cell); a
/// run at the edge of the content shows nothing (`""`); in a heading or a
/// link's text a break is a space.
fn break_plan(
    values: &[Inline],
    place: Place,
    shows: impl Fn(&Inline) -> bool + Copy,
) -> Vec<Option<&'static str>> {
    let mut plan = vec![None; values.len()];
    let mut index = 0;
    while index < values.len() {
        if !matches!(values[index], Inline::LineBreak) {
            index += 1;
            continue;
        }
        let (mut count, mut last) = (0, index);
        for (at, value) in values.iter().enumerate().skip(index) {
            if shows(value) {
                break;
            }
            if matches!(value, Inline::LineBreak) {
                count += 1;
                last = at;
            }
        }
        let edge = !values[..index].iter().any(&shows) || !values[last + 1..].iter().any(&shows);
        plan[index] = Some(match place {
            _ if edge => "",
            Place::OneLine => " ",
            Place::Text | Place::Cell if count > 1 => "\n\n",
            Place::Text => "\\\n",
            Place::Cell => "\n",
        });
        index = last + 1;
    }
    plan
}

/// The content of a heading that cannot be one (a table cell has no block
/// structure), emphasised so it still stands apart from the text around it.
fn heading_as_emphasis(content: &[Inline]) -> Vec<Inline> {
    content
        .iter()
        .map(|inline| match inline {
            Inline::Text { text, style } => {
                let mut style = *style;
                style.bold = true;
                Inline::Text {
                    text: text.clone(),
                    style,
                }
            }
            Inline::Link { content, target } => Inline::Link {
                content: heading_as_emphasis(content),
                target: target.clone(),
            },
            other => other.clone(),
        })
        .collect()
}

/// A heading's text with a closing sequence written as text: `## Issue #`
/// would end in an optional closing `#` that Markdown drops.
fn literal_closing_hashes(text: &str) -> String {
    let kept = text.trim_end_matches('#');
    if kept.len() < text.len() && (kept.is_empty() || kept.ends_with([' ', '\t'])) {
        format!("{kept}\\{}", &text[kept.len()..])
    } else {
        text.to_owned()
    }
}

/// A link's destination as written, or `None` where the link is not written
/// as one (no destination, or a scheme other than web, mail and telephone):
/// its text then stands in the line as it is.
fn link_destination(target: &LinkTarget) -> Option<String> {
    let target = match target {
        LinkTarget::Anchor(anchor) => format!("#{}", destination(anchor)),
        LinkTarget::External(url) | LinkTarget::Relative(url) => destination(url),
    };
    let safe = url::Url::parse(&target)
        .map(|url| matches!(url.scheme(), "http" | "https" | "mailto" | "tel"))
        .unwrap_or(true);
    (!target.is_empty() && safe).then_some(target)
}

/// `values` with the text of each link not written as one in its place, and
/// the line breaks at either edge of a written link's text moved before or
/// after the link, where they break the line as a `<br>` at the edge of an
/// HTML page's `<a>` does (inside, a link's text is one line). Judged in
/// the line around them, a link's breaks follow the same rules as any
/// other: `a<br><br>` inside a link ends the paragraph, not the link's text.
fn link_edges(values: &[Inline]) -> std::borrow::Cow<'_, [Inline]> {
    if !values
        .iter()
        .any(|value| matches!(value, Inline::Link { .. }))
    {
        return std::borrow::Cow::Borrowed(values);
    }
    let edge = |inline: &&Inline| match inline {
        Inline::LineBreak => true,
        Inline::Text { text, .. } => text.trim().is_empty(),
        _ => false,
    };
    let breaks = |inlines: &[Inline]| {
        inlines
            .iter()
            .filter(|inline| matches!(inline, Inline::LineBreak))
            .cloned()
            .collect::<Vec<_>>()
    };
    let mut output = Vec::with_capacity(values.len());
    for value in values {
        let Inline::Link { content, target } = value else {
            output.push(value.clone());
            continue;
        };
        let content = link_edges(content);
        if link_destination(target).is_none() {
            output.extend(content.iter().cloned());
            continue;
        }
        // An edge moves out only where it holds a break; blank text alone
        // stays the link's own.
        let mut lead = content.iter().take_while(edge).count();
        if !content[..lead]
            .iter()
            .any(|i| matches!(i, Inline::LineBreak))
        {
            lead = 0;
        }
        let mut trail = content[lead..].iter().rev().take_while(edge).count();
        if !content[content.len() - trail..]
            .iter()
            .any(|i| matches!(i, Inline::LineBreak))
        {
            trail = 0;
        }
        output.extend(breaks(&content[..lead]));
        output.push(Inline::Link {
            content: content[lead..content.len() - trail].to_vec(),
            target: target.clone(),
        });
        output.extend(breaks(&content[content.len() - trail..]));
    }
    std::borrow::Cow::Owned(output)
}

/// Text on one line: inside a heading, a link's text or `![…]`, a line break
/// would end the heading or let the next line start a block of its own.
fn one_line(text: &str) -> std::borrow::Cow<'_, str> {
    if text.contains(['\n', '\r']) {
        text.split_whitespace().collect::<Vec<_>>().join(" ").into()
    } else {
        text.into()
    }
}

impl Renderer<'_> {
    /// Inline content as Markdown where it stands: a paragraph's lines, or a
    /// table cell's in a cell.
    fn inlines(&self, values: &[Inline]) -> String {
        self.inlines_in(
            values,
            if self.in_cell {
                Place::Cell
            } else {
                Place::Text
            },
        )
    }

    fn inlines_in(&self, values: &[Inline], place: Place) -> String {
        let mut line = Line::default();
        self.write_inlines(values, place, false, &mut line);
        line.finish(place)
    }

    /// Where an image's link points: its asset's published name, or its
    /// external address; `None` when only its description can be written.
    fn image_target(&self, source: &ImageSource) -> Option<String> {
        match source {
            ImageSource::Asset(id) => self
                .asset_names
                .get(id.0)
                .map(|name| format!(".markitai/assets/{name}")),
            ImageSource::External(url) => Some(destination(url)),
            ImageSource::Unavailable => None,
        }
    }

    /// Whether an inline shows anything where it stands, as it is written: a
    /// line break, an anchor, blank text or an image with neither a picture
    /// nor a description does not.
    fn shows(&self, value: &Inline) -> bool {
        match value {
            Inline::Anchor(_) | Inline::LineBreak => false,
            Inline::Text { text, .. } => !text.trim().is_empty(),
            Inline::Link { content, .. } => content.iter().any(|inline| self.shows(inline)),
            Inline::Image { alt, source } => {
                !alt.trim().is_empty() || self.image_target(source).is_some()
            }
            _ => true,
        }
    }

    /// Writes `values` into `line`; `label` when they are a link's text.
    fn write_inlines(&self, values: &[Inline], place: Place, label: bool, line: &mut Line) {
        let edged = link_edges(values);
        let values = merged_runs(&edged);
        let breaks = break_plan(&values, place, |value| self.shows(value));
        let plain = Literal {
            label,
            ..Literal::default()
        };
        for (index, value) in values.iter().enumerate() {
            match value {
                Inline::Text { text, style } => {
                    let text = if place == Place::OneLine {
                        one_line(text)
                    } else {
                        text.as_str().into()
                    };
                    if text.trim().is_empty() {
                        line.text(&text, plain);
                        continue;
                    }
                    let trimmed = text.trim();
                    let prefix = &text[..text.len() - text.trim_start().len()];
                    let suffix = &text[text.trim_end().len()..];
                    let emphasised = !style.code && (style.bold || style.italic || style.strike);
                    // A link already shows as underlined; Word's Hyperlink style
                    // would otherwise wrap every link label in tags.
                    let underline = style.underline && !label;
                    let (lead, core, trail) = if emphasised {
                        let before = prefix.chars().next_back().or_else(|| line.last_char());
                        let after = suffix.chars().next().or_else(|| {
                            values[index + 1..]
                                .iter()
                                .find(|next| !matches!(next, Inline::Anchor(_)))
                                .and_then(leading_char)
                        });
                        // The generated underline tags already separate
                        // emphasis from neighbouring prose. Preserve the
                        // source's emphasis on punctuation inside those tags.
                        if underline {
                            emphasis_edges(trimmed, Some('>'), Some('<'))
                        } else {
                            emphasis_edges(trimmed, before, after)
                        }
                    } else {
                        ("", trimmed, "")
                    };
                    // Nothing but punctuation left to emphasise: it stays plain.
                    let marked = !core.is_empty();
                    let markers = [
                        (style.strike, "~~"),
                        (style.italic, "*"),
                        (style.bold, "**"),
                    ];
                    line.text(prefix, plain);
                    // Underline has no Markdown delimiter. Its HTML wrapper
                    // includes punctuation moved outside emphasis markers.
                    if underline {
                        line.markup("<u>");
                    }
                    line.text(lead, plain);
                    for (on, marker) in markers {
                        if on && marked {
                            line.markup(marker);
                        }
                    }
                    if style.code {
                        let max_ticks =
                            trimmed.split(|c| c != '`').map(str::len).max().unwrap_or(0);
                        let ticks = "`".repeat(max_ticks + 1);
                        line.markup(&if trimmed.starts_with('`') || trimmed.ends_with('`') {
                            format!("{ticks} {trimmed} {ticks}")
                        } else {
                            format!("{ticks}{trimmed}{ticks}")
                        });
                    } else {
                        line.text(
                            core,
                            Literal {
                                label,
                                emphasis: marked && (style.bold || style.italic),
                                strike: marked && style.strike,
                            },
                        );
                    }
                    for (on, marker) in markers.into_iter().rev() {
                        if on && marked {
                            line.markup(marker);
                        }
                    }
                    line.text(trail, plain);
                    if underline {
                        line.markup("</u>");
                    }
                    line.text(suffix, plain);
                }
                Inline::Link { content, target } => match link_destination(target) {
                    // A link's text is one line: split over lines, normal
                    // output's link repair would keep only its first.
                    Some(target) => {
                        line.markup("[");
                        self.write_inlines(content, Place::OneLine, true, line);
                        line.markup(&format!("]({target})"));
                    }
                    // `link_edges` has put such a link's text in its place.
                    None => self.write_inlines(content, place, label, line),
                },
                Inline::Image { alt, source } => {
                    let alt = one_line(alt);
                    if let Some(target) = self.image_target(source) {
                        line.markup("![");
                        line.text(
                            &alt,
                            Literal {
                                label: true,
                                ..Literal::default()
                            },
                        );
                        line.markup(&format!("]({target})"));
                    } else {
                        line.text(&alt, plain);
                    }
                }
                Inline::Anchor(anchor) => {
                    if !self.anchors.contains(anchor) {
                        continue;
                    }
                    let anchor = anchor
                        .replace('&', "&amp;")
                        .replace('"', "&quot;")
                        .replace('<', "&lt;");
                    line.markup(&format!("<a id=\"{anchor}\"></a>"));
                }
                Inline::NoteRef(id) => line.markup(&format!("[^{}]", destination(id))),
                Inline::LineBreak => {
                    if let Some(written) = breaks[index] {
                        line.line_break(written);
                    }
                }
                Inline::Math(text) => line.markup(&format!("${text}$")),
                Inline::Checkbox(checked) => line.markup(if *checked { "[x] " } else { "[ ] " }),
            }
        }
    }

    /// The anchors a heading carries that some link targets, as inlines to
    /// write before the heading, and the heading's content without them.
    fn lifted_anchors(&self, own: Option<&str>, content: &[Inline]) -> (Vec<Inline>, Vec<Inline>) {
        fn strip(
            renderer: &Renderer<'_>,
            values: &[Inline],
            targets: &mut Vec<Inline>,
        ) -> Vec<Inline> {
            values
                .iter()
                .filter_map(|value| match value {
                    Inline::Anchor(anchor) => {
                        if renderer.anchors.contains(anchor)
                            && !targets
                                .iter()
                                .any(|seen| matches!(seen, Inline::Anchor(a) if a == anchor))
                        {
                            targets.push(Inline::Anchor(anchor.clone()));
                        }
                        None
                    }
                    Inline::Link { content, target } => Some(Inline::Link {
                        content: strip(renderer, content, targets),
                        target: target.clone(),
                    }),
                    other => Some(other.clone()),
                })
                .collect()
        }
        let mut targets = Vec::new();
        if let Some(anchor) = own.filter(|anchor| self.anchors.contains(*anchor)) {
            targets.push(Inline::Anchor(anchor.to_owned()));
        }
        let content = strip(self, content, &mut targets);
        (targets, content)
    }

    /// Whether a document table only lays out content: no row holds two
    /// non-empty cells, and a cell holds a table or several blocks with
    /// content (a web page saved as a document, spacing its comments with
    /// empty columns). Its cells are then the document's blocks. A table of
    /// single paragraphs stays a table even with empty columns (a form to
    /// fill in), and spreadsheets keep every table as a table.
    fn is_layout_table(&self, table: &anydoc::model::Table) -> bool {
        if !matches!(
            self.extension,
            "doc" | "docx" | "docm" | "odt" | "rtf" | "epub"
        ) {
            return false;
        }
        let filled = |slot: &CellSlot| matches!(slot, CellSlot::Origin(cell) if cell.blocks.iter().any(has_content));
        if table
            .grid
            .iter()
            .any(|row| row.iter().filter(|slot| filled(slot)).count() > 1)
        {
            return false;
        }
        table.grid.iter().flatten().any(|slot| {
            matches!(slot, CellSlot::Origin(cell)
                if cell.blocks.iter().filter(|block| has_content(block)).count() > 1
                    || cell.blocks.iter().any(|block| matches!(block, Block::Table(_))))
        })
    }

    /// A cell's blocks as the text of one Markdown table cell. A table
    /// nested in the cell cannot be written as a Markdown table: each of its
    /// rows becomes a line of its cells' text.
    fn cell_text(&mut self, blocks: &[Block]) -> String {
        let outer = std::mem::replace(&mut self.in_cell, true);
        let text = self.cell_blocks(blocks);
        self.in_cell = outer;
        text
    }

    fn cell_blocks(&mut self, blocks: &[Block]) -> String {
        let mut parts = Vec::new();
        for block in blocks {
            let text = match block {
                // A cell has no block structure: a heading is emphasised text
                // (a `#` would show as text).
                Block::Heading {
                    anchor, content, ..
                } => {
                    let mut inlines = Vec::new();
                    if let Some(anchor) = anchor
                        .as_ref()
                        .filter(|anchor| self.anchors.contains(*anchor))
                    {
                        inlines.push(Inline::Anchor(anchor.clone()));
                    }
                    inlines.extend(heading_as_emphasis(content));
                    self.inlines_in(&inlines, Place::OneLine)
                        .trim_end()
                        .to_owned()
                }
                Block::Table(table) => table
                    .grid
                    .iter()
                    .filter_map(|row| {
                        let cells = row
                            .iter()
                            .filter_map(|slot| match slot {
                                CellSlot::Origin(cell) => Some(self.cell_text(&cell.blocks)),
                                CellSlot::Covered { .. } => None,
                            })
                            .map(|text| text.trim().to_owned())
                            .filter(|text| !text.is_empty())
                            .collect::<Vec<_>>();
                        (!cells.is_empty()).then(|| cells.join(" "))
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
                other => self.blocks(std::slice::from_ref(other)),
            };
            // A block keeps its own leading spaces (code indentation); the
            // caller trims the cell as a whole.
            if !text.trim().is_empty() {
                parts.push(text.trim_end().to_owned());
            }
        }
        parts.join("\n\n")
    }

    fn blocks(&mut self, blocks: &[Block]) -> String {
        // The reference spreadsheet converters write each sheet as a heading
        // line immediately followed by its table.
        let sheet_layout = matches!(self.extension, "xlsx" | "xlsm" | "xls");
        let mut joined = String::new();
        let mut previous_heading = false;
        // The last heading's text, which a sheet's merged title may repeat.
        let mut last_heading = String::new();
        for block in blocks {
            // Whether a table's merged title rows were written above it.
            let mut lifted_title = false;
            let rendered = match block {
                Block::Heading {
                    level,
                    anchor,
                    content,
                } => {
                    let plain_bold;
                    let content = if self.extension == "rtf" {
                        plain_bold = heading_without_bold(content);
                        &plain_bold
                    } else {
                        content
                    };
                    // A target a link names is written on a line of its own
                    // before the heading: inside it, `## <a id="x"></a>Title`
                    // puts markup into the words a reader or a search sees.
                    let (targets, content) = self.lifted_anchors(anchor.as_deref(), content);
                    // A heading is one line: a line break in the source (a
                    // Word heading with a soft return) is a space, since the
                    // words after the break would leave it as a paragraph.
                    last_heading = self
                        .inlines_in(&content, Place::OneLine)
                        .trim_end()
                        .to_owned();
                    let heading = format!(
                        "{} {}",
                        "#".repeat(usize::from((*level).clamp(1, 6))),
                        literal_closing_hashes(&last_heading)
                    );
                    if targets.is_empty() {
                        heading
                    } else {
                        format!("{}\n{heading}", self.inlines(&targets))
                    }
                }
                // A cell holds inline content only: a line that starts like
                // a block is text there.
                Block::Paragraph(values) if self.in_cell => {
                    self.inlines(values).trim_end().to_owned()
                }
                Block::Paragraph(values) => literal_heading_marks(&self.inlines(values))
                    .trim_end()
                    .to_owned(),
                Block::CodeBlock { lang, text } => {
                    super::text::fence(text, lang.as_deref().unwrap_or(""))
                }
                Block::Math(text) => format!("$$\n{text}\n$$"),
                Block::Rule => "---".into(),
                Block::BlockQuote(blocks) => self
                    .blocks(blocks)
                    .lines()
                    .map(|line| format!("> {line}"))
                    .collect::<Vec<_>>()
                    .join("\n"),
                Block::List(list) => {
                    let mut items: Vec<ListLine> = Vec::new();
                    for (index, item) in list.items.iter().enumerate() {
                        let bullet = if matches!(self.extension, "doc" | "odt") {
                            "-"
                        } else {
                            "*"
                        };
                        let marker = match item.marker_label.as_deref().map(str::trim) {
                            // A source's own label must still read as a
                            // Markdown marker: RTF list text gives `1` and
                            // `•`, which would leave the items one paragraph.
                            Some(label)
                                if !label.is_empty()
                                    && label.bytes().all(|b| b.is_ascii_digit()) =>
                            {
                                format!("{label}.")
                            }
                            Some(
                                "•" | "◦" | "▪" | "●" | "○" | "■" | "□" | "·" | "‣" | "⁃" | "–"
                                | "o",
                            ) => bullet.into(),
                            Some(label) if !label.is_empty() => label.to_owned(),
                            _ if list.ordered() => list.marker.label(list.start + index as u64),
                            _ => bullet.into(),
                        };
                        let content = self.blocks(&item.blocks);
                        if content.trim().is_empty() {
                            continue;
                        }
                        // Whether Markdown reads the marker as one (a number
                        // and a stop, or a bullet); a label such as `a)`,
                        // `(1)` or `一、` is only text at the head of a line.
                        let digits = marker.bytes().take_while(u8::is_ascii_digit).count();
                        let markdown_marker = matches!(marker.as_str(), "*" | "-" | "+")
                            || ((1..=9).contains(&digits)
                                && matches!(&marker[digits..], "." | ")"));
                        // Text under a marker Markdown does not read is not
                        // list content; indented four columns or more after a
                        // blank line it would be a code block.
                        let width = marker.chars().count() + 1;
                        let indent = " ".repeat(if markdown_marker { width } else { width.min(3) });
                        let mut lines = content.lines();
                        let mut item_text = format!("{marker} {}", lines.next().unwrap_or(""));
                        for line in lines {
                            item_text.push_str(&format!("\n{indent}{line}"));
                        }
                        items.push(ListLine {
                            text: item_text.trim_end().to_owned(),
                            markdown_marker,
                            // A bullet, or a number one, may start a list
                            // right under a paragraph's line.
                            interrupts: markdown_marker
                                && (digits == 0 || marker[..digits].parse() == Ok(1u64)),
                            ends_in_paragraph: matches!(
                                item.blocks.iter().rev().find(|block| has_content(block)),
                                Some(Block::Paragraph(_))
                            ),
                        });
                    }
                    let mut joined = String::new();
                    let mut previous: Option<&ListLine> = None;
                    for item in &items {
                        if let Some(previous) = previous {
                            joined.push_str(list_separator(previous, item, self.in_cell));
                        }
                        joined.push_str(&item.text);
                        previous = Some(item);
                    }
                    joined
                }
                Block::Table(table) if self.is_layout_table(table) => table
                    .grid
                    .iter()
                    .flatten()
                    .filter_map(|slot| match slot {
                        CellSlot::Origin(cell) => Some(self.blocks(&cell.blocks)),
                        CellSlot::Covered { .. } => None,
                    })
                    .filter(|text| !text.trim().is_empty())
                    .collect::<Vec<_>>()
                    .join("\n\n"),
                Block::Table(table) => {
                    let mut rows = table
                        .grid
                        .iter()
                        .map(|row| {
                            row.iter()
                                .map(|slot| match slot {
                                    CellSlot::Covered { .. } => String::new(),
                                    CellSlot::Origin(cell) => {
                                        table_cell(self.cell_text(&cell.blocks).trim())
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    if self.extension == "ods" {
                        // Columns after the last one that holds text in any
                        // row are the sheet's declared width or the span of a
                        // merged title, not data; interior empty columns stay.
                        let width = rows
                            .iter()
                            .filter_map(|row| row.iter().rposition(|cell| !cell.is_empty()))
                            .max()
                            .map(|last| last + 1)
                            .unwrap_or(0);
                        for row in &mut rows {
                            row.truncate(width);
                        }
                    }
                    let sheet = matches!(self.extension, "ods" | "xlsx" | "xlsm" | "xls" | "xlsb");
                    // A sheet's merged title rows are text above its table,
                    // where the real header row follows them.
                    let (titles, body) = if sheet {
                        sheet_titles(table, &rows)
                    } else {
                        (Vec::new(), 0)
                    };
                    self.merged_cells |= table.grid[body.min(table.grid.len())..]
                        .iter()
                        .flatten()
                        .any(|slot| matches!(slot, CellSlot::Origin(cell) if cell.row_span > 1 || cell.col_span > 1));
                    // A sheet's first row is its header, as the reference's
                    // spreadsheet readers take it, whether or not the source
                    // marks it; a blank header row would only push it down.
                    // A document's table has a header only when the document
                    // declares one (Word's repeated header row); otherwise
                    // the header line is blank and every row is data.
                    let header = sheet || table.header_rows > 0;
                    let markdown = super::text::table(&rows[body..], header)
                        .trim_end()
                        .to_owned();
                    // A title the sheet's heading already gives is said once.
                    let titles = titles
                        .into_iter()
                        .filter(|title| !(previous_heading && last_heading == *title))
                        .collect::<Vec<_>>();
                    lifted_title = !titles.is_empty();
                    if titles.is_empty() {
                        markdown
                    } else {
                        format!("{}\n\n{markdown}", titles.join("\n\n"))
                    }
                }
            };
            if !rendered.is_empty() {
                let table = matches!(block, Block::Table(_));
                if !joined.is_empty() {
                    joined.push_str(
                        if sheet_layout && previous_heading && table && !lifted_title {
                            "\n"
                        } else {
                            "\n\n"
                        },
                    );
                }
                joined.push_str(&rendered);
                previous_heading = matches!(block, Block::Heading { .. });
            }
        }
        joined
    }

    /// A presentation's slides as the PPTX reader writes them: each slide
    /// behind a `<!-- Slide number: N -->` line, blank slides included, and a
    /// blank line between slides. `starts` holds the index in `blocks` where
    /// each slide begins; blocks before the first start belong to no slide
    /// and come first, unnumbered.
    fn slides(&mut self, blocks: &[Block], starts: &[usize]) -> String {
        let bounded = |index: usize| {
            starts
                .get(index)
                .map_or(blocks.len(), |&at| at.min(blocks.len()))
        };
        let mut pages = Vec::with_capacity(starts.len() + 1);
        let head = self.blocks(&blocks[..bounded(0)]);
        if !head.is_empty() {
            pages.push(head);
        }
        for index in 0..starts.len() {
            let start = bounded(index);
            let end = bounded(index + 1).max(start);
            let content = self.blocks(&blocks[start..end]);
            let mut page = format!("<!-- Slide number: {} -->", index + 1);
            if !content.is_empty() {
                page.push('\n');
                page.push_str(&content);
            }
            pages.push(page);
        }
        pages.join("\n\n")
    }
}

/// A list item as written, with what decides how it joins the item before.
struct ListLine {
    text: String,
    /// Its marker is one Markdown reads (`*`, `1.`); a label such as `a)`,
    /// `(1)` or `一、` is only text at the head of a line.
    markdown_marker: bool,
    /// Its marker may start a list right under a paragraph's line.
    interrupts: bool,
    /// Its last block is a paragraph, which a hard break can continue.
    ends_in_paragraph: bool,
}

/// What stands between two list items. Markdown items follow each other on
/// the next line. A line that starts with a label Markdown does not read (or
/// a number other than one, under a paragraph) would continue the line
/// before it: a hard break keeps it a line of its own after a paragraph, and
/// a blank line after anything else (a fence or a table row cannot end with
/// a backslash). In a table cell every item is a line of the cell.
fn list_separator(previous: &ListLine, next: &ListLine, in_cell: bool) -> &'static str {
    if in_cell || (next.markdown_marker && (previous.markdown_marker || next.interrupts)) {
        "\n"
    } else if !next.markdown_marker && previous.ends_in_paragraph {
        "\\\n"
    } else {
        "\n\n"
    }
}

/// A cell's Markdown as one table cell, in one pass: inline Markdown is kept,
/// `|` is escaped, and each line (a line break, a paragraph, a list item; see
/// [`Place::Cell`]) is joined with `<br>`.
fn table_cell(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    let mut lines = text.split('\n').peekable();
    let mut first = true;
    while let Some(line) = lines.next() {
        if !first {
            out.push_str("<br>");
        }
        first = false;
        let line = if lines.peek().is_some() {
            line.strip_suffix("  ").unwrap_or(line)
        } else {
            line
        };
        for (index, part) in line.split('|').enumerate() {
            if index > 0 {
                out.push_str("\\|");
            }
            out.push_str(part);
        }
    }
    out
}

/// A sheet's leading title rows, as their text, and the row its table starts
/// at. A title row holds one cell, merged across every column that holds
/// content in the sheet; the row its titles lead to (past blank rows) must
/// be the real header: two cells or more, none merged. Without both, the
/// table starts at its first row, which is then its header as before.
fn sheet_titles(table: &anydoc::model::Table, rows: &[Vec<String>]) -> (Vec<String>, usize) {
    let filled = |row: &[String]| row.iter().filter(|cell| !cell.is_empty()).count();
    // Most sheets have no title: their first row holds several cells.
    if rows.first().is_none_or(|row| filled(row) != 1) {
        return (Vec::new(), 0);
    }
    let first = rows
        .iter()
        .filter_map(|row| row.iter().position(|cell| !cell.is_empty()))
        .min();
    let last = rows
        .iter()
        .filter_map(|row| row.iter().rposition(|cell| !cell.is_empty()))
        .max();
    let (Some(first), Some(last)) = (first, last) else {
        return (Vec::new(), 0);
    };
    let mut titles = Vec::new();
    let mut at = 0;
    while let Some(row) = rows.get(at)
        && filled(row) == 1
        && let Some(column) = row.iter().position(|cell| !cell.is_empty())
        && matches!(table.grid.get(at).and_then(|slots| slots.get(column)),
            Some(CellSlot::Origin(cell)) if cell.row_span == 1 && cell.col_span > 1
                && column <= first && column + cell.col_span as usize > last)
    {
        titles.push(row[column].clone());
        at += 1;
        while rows.get(at).is_some_and(|row| filled(row) == 0) {
            at += 1;
        }
    }
    let header = rows.get(at).is_some_and(|row| filled(row) >= 2)
        && table.grid.get(at).is_some_and(|slots| {
            slots.iter().all(|slot| {
                matches!(slot, CellSlot::Origin(cell) if cell.row_span == 1 && cell.col_span == 1)
            })
        });
    if titles.is_empty() || !header {
        return (Vec::new(), 0);
    }
    (titles, at)
}

/// Markdown for the blocks an embedded object holds, which carry no assets,
/// notes or anchors of the presentation around them (`office.rs`).
fn object_markdown(blocks: &[Block]) -> String {
    Renderer {
        asset_names: &[],
        merged_cells: false,
        in_cell: false,
        anchors: BTreeSet::new(),
        extension: "pptx",
    }
    .blocks(blocks)
}

pub(super) fn extract(bytes: &[u8], extension: &str) -> Result<Document> {
    // A template reads as the document it makes.
    let extension = super::document_extension(extension);
    let format = anydoc::Format::from_extension(extension)
        .ok_or_else(|| Error::Unsupported(format!("Unsupported format: {extension}")))?;
    if format == anydoc::Format::Pdf {
        return pdf::extract(bytes);
    }
    if format == anydoc::Format::Pptx {
        return office::extract_presentation(bytes);
    }
    // A compound file that fails for an unused, malformed mini stream or a
    // FAT one sector short (the macOS Word 97 exporter writes both) is read
    // from a repaired copy; if the copy fails too, the original error stands.
    let (mut parsed, repaired) = match anydoc::to_document(bytes, format) {
        Err(error) if error.to_string().contains("not an OLE2 compound file") => {
            match compound::repaired(bytes)
                .and_then(|repaired| Some((anydoc::to_document(&repaired, format).ok()?, repaired)))
            {
                Some((parsed, repaired)) => (parsed, Some(repaired)),
                None => return Err(conversion_error_for_format(error, format)),
            }
        }
        result => (
            result.map_err(|error| conversion_error_for_format(error, format))?,
            None,
        ),
    };
    let bytes = repaired.as_deref().unwrap_or(bytes);
    let metadata = office_meta::read(bytes, extension);
    if metadata.sheets.len() == 1 && !matches!(parsed.blocks.first(), Some(Block::Heading { .. })) {
        parsed.blocks.insert(
            0,
            Block::heading(2, vec![Inline::plain(metadata.sheets[0].clone())]),
        );
    }
    if extension == "epub" && metadata.title().is_some_and(|title| matches!(parsed.blocks.first(), Some(Block::Heading { content, .. }) if anydoc::model::inlines_to_plain_text(content) == title)) {
        parsed.blocks.remove(0);
    }
    let names = parsed
        .assets
        .iter()
        .map(|asset| {
            format!(
                "asset-{}.{}",
                asset.id.0 + 1,
                image_extension(&asset.media_type, &asset.origin_part)
            )
        })
        .collect::<Vec<_>>();
    let mut anchors = BTreeSet::new();
    let mut used_assets = BTreeSet::new();
    references(&parsed.blocks, &mut anchors, &mut used_assets);
    for note in &parsed.notes {
        references(&note.blocks, &mut anchors, &mut used_assets);
    }
    let mut renderer = Renderer {
        asset_names: &names,
        merged_cells: false,
        in_cell: false,
        anchors,
        extension,
    };
    let mut markdown = if parsed.slide_starts.is_empty() {
        renderer.blocks(&parsed.blocks)
    } else {
        renderer.slides(&parsed.blocks, &parsed.slide_starts)
    };
    let preamble = metadata.preamble();
    if !preamble.is_empty() {
        markdown = format!("{preamble}\n\n{markdown}");
    }
    for note in &parsed.notes {
        let text = renderer.blocks(&note.blocks);
        let mut lines = text.lines();
        // The text of a note follows a space that Word writes after the mark.
        markdown.push_str(&format!(
            "\n\n[^{}]: {}",
            destination(&note.id),
            lines.next().unwrap_or("").trim_start()
        ));
        for line in lines {
            markdown.push_str(&format!("\n    {line}"));
        }
    }
    // The reference's legacy Word/PowerPoint (anydoc), RTF and OpenDocument
    // output ends with a newline; its DOCX, XLS/XLSX and EPUB output does not.
    if matches!(extension, "doc" | "ppt" | "rtf" | "odt" | "ods") && !markdown.is_empty() {
        markdown.push('\n');
    }
    let mut warnings = metadata.warnings.clone();
    // What the reader left out on purpose, such as a Word document's
    // embedded part in a format it does not convert.
    warnings.append(&mut parsed.warnings);
    if repaired.is_some() {
        warnings.push("The compound file's allocation tables were malformed (an unused mini stream or a short FAT, as the macOS Word 97 exporter writes); it was read from a repaired copy.".into());
    }
    if renderer.merged_cells {
        warnings.push("Merged table cells are represented by their origin cell with empty covered cells in Markdown.".into());
    }
    let mut document = Document {
        markdown,
        warnings,
        ..Document::default()
    };
    document
        .metadata
        .insert("converter".into(), "anydoc".into());
    if let Some(title) = metadata.title().filter(|s| !s.is_empty()) {
        document.metadata.insert("title".into(), title.into());
    }
    for (asset, name) in parsed.assets.into_iter().zip(names) {
        if !used_assets.contains(&asset.id.0) {
            continue;
        }
        document.assets.push(Asset {
            name,
            bytes: asset.bytes,
        });
    }
    Ok(document)
}

#[cfg(test)]
#[path = "native/docx_tests.rs"]
mod docx_tests;

#[cfg(test)]
#[path = "native/odt_rtf_tests.rs"]
mod odt_rtf_tests;

#[cfg(test)]
#[path = "native/line_tests.rs"]
mod line_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_line_markdown_would_read_as_a_heading_keeps_its_hash_as_text() {
        for (line, written) in [
            ("- and then it rained.", "\\- and then it rained."),
            ("+ plus", "\\+ plus"),
            ("2024. A good year", "2024\\. A good year"),
            ("1) first", "1\\) first"),
            ("> quoted", "\\> quoted"),
            ("---", "\\---"),
            ("- - -", "\\- - -"),
            ("-dash", "-dash"),
            ("3.5 kg", "3.5 kg"),
            ("1234567890. too long", "1234567890. too long"),
            ("# shell comment", "\\# shell comment"),
            ("  ## two in", "  \\## two in"),
            ("#", "\\#"),
            ("a\n# b", "a\n\\# b"),
            ("#hashtag", "#hashtag"),
            ("####### seven", "####### seven"),
            ("    # indented code", "    # indented code"),
            ("C# and F#", "C# and F#"),
        ] {
            assert_eq!(literal_heading_marks(line), written, "{line:?}");
        }
    }
    use anydoc::model::{AssetId, Cell, Style, Table, TableKind};

    #[test]
    fn xlsx_node_limit_preserves_the_error_and_offers_content_export_remedies() {
        let limit = || anydoc::ConvertError::ResourceLimit {
            limit: "max_xml_nodes",
            detail: "part exceeds 2000000 xml nodes".into(),
        };
        let error = conversion_error_for_format(limit(), anydoc::Format::Excel).to_string();
        assert!(error.contains("resource limit exceeded (max_xml_nodes)"));
        assert!(error.contains("Export the needed sheets as CSV"));
        assert!(error.contains("split the workbook"));
        assert!(
            !conversion_error_for_format(limit(), anydoc::Format::Docx)
                .to_string()
                .contains("CSV")
        );
        assert!(
            !conversion_error_for_format(anydoc::ConvertError::Encrypted, anydoc::Format::Excel)
                .to_string()
                .contains("CSV")
        );
    }

    #[test]
    fn rtf_ends_with_a_newline_and_headings_drop_trailing_spaces_like_the_reference() {
        let rtf = br"{\rtf1\ansi{\stylesheet{\s1 heading 1;}}{\pard\s1 Title with space \par}{\pard Body text.\par}}";
        let document = extract(rtf, "rtf").unwrap();
        assert!(
            document.markdown.ends_with("Body text.\n"),
            "{:?}",
            document.markdown
        );
        assert!(
            !document.markdown.contains(" \n"),
            "{:?}",
            document.markdown
        );
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "docx",
        };
        assert_eq!(
            renderer.blocks(&[Block::heading(1, vec![Inline::plain("Heading ")])]),
            "# Heading"
        );
    }

    #[test]
    fn spreadsheet_sheet_heading_is_followed_directly_by_its_table_like_the_reference() {
        for (extension, bytes, separator) in [
            (
                "xlsx",
                &include_bytes!("../office_render/fixtures/whole-workbook.xlsx")[..],
                "\n",
            ),
            (
                "xls",
                &include_bytes!("../office_render/fixtures/whole-workbook.xls")[..],
                "\n",
            ),
            // ODS is not one of the reference's XLSX/XLS converters.
            (
                "ods",
                &include_bytes!("../office_render/fixtures/whole-workbook.ods")[..],
                "\n\n",
            ),
        ] {
            let doc = extract(bytes, extension).unwrap();
            assert!(
                doc.markdown
                    .starts_with(&format!("## Wide 宽表{separator}| ")),
                "{extension}: {}",
                &doc.markdown[..doc.markdown.len().min(80)]
            );
        }
    }
    #[test]
    fn document_renderer_keeps_assets_math_notes_and_styles() {
        let names = vec!["asset-1.png".into()];
        let mut renderer = Renderer {
            asset_names: &names,
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "docx",
        };
        let result = renderer.blocks(&[
            Block::heading(1, vec![Inline::plain("Title")]),
            Block::Paragraph(vec![
                Inline::Text {
                    text: " bold ".into(),
                    style: Style {
                        bold: true,
                        ..Style::PLAIN
                    },
                },
                Inline::Image {
                    alt: "figure".into(),
                    source: ImageSource::Asset(AssetId(0)),
                },
                Inline::NoteRef("1".into()),
                Inline::Math("x^2".into()),
            ]),
            Block::Table(Table::from_rows(
                vec![vec![Cell::from_inlines(vec![Inline::plain("a|b")])]],
                0,
                TableKind::Data,
            )),
        ]);
        assert!(result.starts_with("# Title"));
        assert!(result.contains(" **bold** "));
        assert!(result.contains("![figure](.markitai/assets/asset-1.png)[^1]$x^2$"));
        assert!(result.contains("a\\|b"));
    }
    #[test]
    fn rtf_converts_locally_and_bad_pdf_fails() {
        let doc = extract(
            br"{\rtf1\ansi Hello \b world\b0\par Second paragraph}",
            "rtf",
        )
        .unwrap();
        assert!(doc.markdown.contains("**world**"));
        assert!(extract(b"this is not a PDF", "pdf").is_err());
    }

    #[test]
    fn referenced_anchors_survive_and_unused_anchors_disappear() {
        let blocks = vec![
            Block::Heading {
                level: 1,
                anchor: Some("target".into()),
                content: vec![Inline::plain("Target")],
            },
            Block::Paragraph(vec![
                Inline::Anchor("unused".into()),
                Inline::Link {
                    content: vec![Inline::plain("jump")],
                    target: LinkTarget::Anchor("target".into()),
                },
            ]),
        ];
        let mut anchors = BTreeSet::new();
        references(&blocks, &mut anchors, &mut BTreeSet::new());
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors,
            extension: "doc",
        };
        let markdown = renderer.blocks(&blocks);
        assert!(markdown.contains("<a id=\"target\"></a>\n# Target"));
        assert!(markdown.contains("[jump](#target)"));
        assert!(!markdown.contains("unused"));
    }

    #[test]
    fn a_bookmark_inside_a_heading_is_written_on_a_line_before_it_and_only_when_linked() {
        // Word bookmarks a heading's text: the anchor sits among its inlines.
        let heading = |anchor: &str, title: &str| Block::Heading {
            level: 2,
            anchor: None,
            content: vec![Inline::Anchor(anchor.into()), Inline::plain(title)],
        };
        let blocks = vec![
            Block::Paragraph(vec![Inline::Link {
                content: vec![Inline::plain("Section Two")],
                target: LinkTarget::Anchor("section_two".into()),
            }]),
            heading("section_two", "Section Two"),
            heading("_Toc99", "Never linked"),
        ];
        let mut anchors = BTreeSet::new();
        references(&blocks, &mut anchors, &mut BTreeSet::new());
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors,
            extension: "docx",
        };
        assert_eq!(
            renderer.blocks(&blocks),
            "[Section Two](#section_two)\n\n<a id=\"section_two\"></a>\n## Section Two\n\n## Never linked"
        );
        // Several bookmarks of one heading share its line, once each.
        let both = Block::Heading {
            level: 1,
            anchor: Some("a".into()),
            content: vec![
                Inline::Anchor("a".into()),
                Inline::plain("Title"),
                Inline::Anchor("b".into()),
            ],
        };
        renderer.anchors = ["a".to_owned(), "b".to_owned()].into();
        assert_eq!(
            renderer.blocks(&[both]),
            "<a id=\"a\"></a><a id=\"b\"></a>\n# Title"
        );
    }

    #[test]
    fn document_and_sheet_table_headers_match_their_format_contracts() {
        let table = Block::Table(Table::from_rows(
            vec![
                vec![
                    Cell::from_inlines(vec![Inline::plain("Name")]),
                    Cell::default(),
                ],
                vec![
                    Cell::from_inlines(vec![Inline::plain("Value")]),
                    Cell::default(),
                ],
            ],
            1,
            TableKind::Data,
        ));
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "docx",
        };
        // A document's table is headed by the row it declares as its header.
        assert_eq!(
            renderer.blocks(std::slice::from_ref(&table)),
            "| Name |  |\n| --- | --- |\n| Value |  |"
        );
        // Without one the header line is blank and every row is data.
        let undeclared = Block::Table(Table::from_rows(
            vec![
                vec![
                    Cell::from_inlines(vec![Inline::plain("Name")]),
                    Cell::default(),
                ],
                vec![
                    Cell::from_inlines(vec![Inline::plain("Value")]),
                    Cell::default(),
                ],
            ],
            0,
            TableKind::Data,
        ));
        assert!(
            renderer
                .blocks(std::slice::from_ref(&undeclared))
                .starts_with("|  |  |\n| --- | --- |\n| Name |  |")
        );
        renderer.extension = "ods";
        assert_eq!(
            renderer.blocks(std::slice::from_ref(&table)),
            "| Name |\n| --- |\n| Value |"
        );
        // A workbook sheet's first row is its header whatever the source marks.
        let unmarked = Block::Table(Table::from_rows(
            vec![
                vec![
                    Cell::from_inlines(vec![Inline::plain("Name")]),
                    Cell::from_inlines(vec![Inline::plain("Size")]),
                ],
                vec![
                    Cell::from_inlines(vec![Inline::plain("Alpha")]),
                    Cell::from_inlines(vec![Inline::plain("1")]),
                ],
            ],
            0,
            TableKind::Data,
        ));
        renderer.extension = "xlsx";
        assert_eq!(
            renderer.blocks(&[unmarked]),
            "| Name | Size |\n| --- | --- |\n| Alpha | 1 |"
        );
    }

    #[test]
    fn source_list_labels_render_as_markdown_markers() {
        use anydoc::model::{List, ListItem, MarkerKind};
        // RTF list text labels items `1`, `2` and `•`; kept verbatim, the
        // items would read as one paragraph.
        let item = |label: &str, text: &str| ListItem {
            blocks: vec![Block::Paragraph(vec![Inline::plain(text)])],
            marker_label: Some(label.into()),
        };
        let list = |marker, items| {
            Block::List(List {
                marker,
                start: 1,
                items,
            })
        };
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "rtf",
        };
        assert_eq!(
            renderer.blocks(&[list(
                MarkerKind::Decimal,
                vec![
                    item("1", "First"),
                    item("2", "Second"),
                    item("1-a)", "Composite")
                ]
            )]),
            // A label Markdown does not read (`1-a)`) is text at the head of a
            // line, which the hard break keeps from joining the line before.
            "1. First\n2. Second\\\n1-a) Composite"
        );
        assert_eq!(
            renderer.blocks(&[list(MarkerKind::Bullet, vec![item("•", "Point")])]),
            "* Point"
        );
    }

    #[test]
    fn runs_split_inside_a_word_render_as_one_run() {
        // RTF writes each `\\u` character as a run of its own; the bold
        // Persian word must not become one bold span per letter.
        let bold = anydoc::model::Style {
            bold: true,
            ..Default::default()
        };
        let run = |text: &str, style| Inline::Text {
            text: text.into(),
            style,
        };
        let renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "rtf",
        };
        let plain = anydoc::model::Style::default();
        assert_eq!(
            renderer.inlines(&[
                run("م", bold),
                run("ع", bold),
                run("رفی", bold),
                run(" and ", plain),
                run("Ba", bold),
                run("ses", bold),
            ]),
            "**معرفی** and **Bases**"
        );
    }

    #[test]
    fn a_layout_table_is_unwrapped_and_a_nested_table_becomes_cell_lines() {
        let paragraph = |text: &str| Block::Paragraph(vec![Inline::plain(text)]);
        // A web page saved as a document: an empty spacer column beside a
        // comment of two paragraphs, wrapped in a one-column container.
        let comment = Block::Table(Table::from_rows(
            vec![vec![
                Cell::default(),
                Cell::new(vec![
                    paragraph("commenter 2 hours ago"),
                    paragraph("A reply."),
                ]),
            ]],
            0,
            TableKind::Data,
        ));
        let container = Block::Table(Table::from_rows(
            vec![vec![Cell::new(vec![comment])]],
            0,
            TableKind::Data,
        ));
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "odt",
        };
        assert_eq!(
            renderer.blocks(std::slice::from_ref(&container)),
            "commenter 2 hours ago\n\nA reply."
        );
        // A data table holding a small table keeps its rows; the nested
        // table's rows become lines of the cell.
        let nested = Block::Table(Table::from_rows(
            vec![
                vec![
                    Cell::from_inlines(vec![Inline::plain("x")]),
                    Cell::from_inlines(vec![Inline::plain("1")]),
                ],
                vec![
                    Cell::from_inlines(vec![Inline::plain("y")]),
                    Cell::from_inlines(vec![Inline::plain("2")]),
                ],
            ],
            0,
            TableKind::Data,
        ));
        let data = Block::Table(Table::from_rows(
            vec![vec![
                Cell::from_inlines(vec![Inline::plain("Point")]),
                Cell::new(vec![nested]),
            ]],
            0,
            TableKind::Data,
        ));
        assert_eq!(
            renderer.blocks(std::slice::from_ref(&data)),
            "|  |  |\n| --- | --- |\n| Point | x 1<br>y 2 |"
        );
        // Spreadsheets keep their tables whatever they hold.
        renderer.extension = "ods";
        assert!(renderer.blocks(&[container]).starts_with('|'));
    }

    #[test]
    fn sheet_trimming_drops_trailing_empty_columns_but_keeps_interior_ones() {
        // A title merged over five columns above three columns of data (the
        // shape of the reference's ODS fixture) is three columns wide: the
        // span and the sheet's declared width are not data.
        let wide = Block::Table(Table {
            grid: vec![
                vec![
                    CellSlot::Origin(Cell::spanning(
                        vec![Block::Paragraph(vec![Inline::plain("Cups")])],
                        5,
                        1,
                    )),
                    CellSlot::Covered {
                        origin_row: 0,
                        origin_col: 0,
                    },
                    CellSlot::Covered {
                        origin_row: 0,
                        origin_col: 0,
                    },
                    CellSlot::Covered {
                        origin_row: 0,
                        origin_col: 0,
                    },
                    CellSlot::Covered {
                        origin_row: 0,
                        origin_col: 0,
                    },
                ],
                vec![
                    CellSlot::Origin(Cell::from_inlines(vec![Inline::plain("Team")])),
                    CellSlot::Origin(Cell::new(vec![])),
                    CellSlot::Origin(Cell::from_inlines(vec![Inline::plain("Count")])),
                    CellSlot::Origin(Cell::new(vec![])),
                    CellSlot::Origin(Cell::new(vec![])),
                ],
            ],
            header_rows: 1,
            kind: TableKind::Data,
        });
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "ods",
        };
        // The merged title is text above the table (see
        // `a_merged_sheet_title_above_its_header_is_not_the_header`).
        assert_eq!(
            renderer.blocks(std::slice::from_ref(&wide)),
            "Cups\n\n| Team |  | Count |\n| --- | --- | --- |"
        );
        // Another format keeps the declared columns.
        renderer.extension = "docx";
        assert!(
            renderer
                .blocks(&[wide])
                .starts_with("| Cups |  |  |  |  |\n| --- | --- | --- | --- | --- |\n")
        );
    }

    #[test]
    fn a_table_cell_escapes_pipes_and_writes_breaks_without_their_spaces() {
        assert_eq!(table_cell("a | b  \nc\n\nd  "), "a \\| b<br>c<br><br>d  ");
        assert_eq!(table_cell("plain"), "plain");
        assert_eq!(table_cell(""), "");
    }

    #[test]
    fn a_merged_sheet_title_above_its_header_is_not_the_header() {
        let text = |value: &str| CellSlot::Origin(Cell::from_inlines(vec![Inline::plain(value)]));
        let covered = || CellSlot::Covered {
            origin_row: 0,
            origin_col: 0,
        };
        let title = |value: &str, span: u32| {
            CellSlot::Origin(Cell::spanning(
                vec![Block::Paragraph(vec![Inline::plain(value)])],
                span,
                1,
            ))
        };
        let sheet = |rows: Vec<Vec<CellSlot>>| {
            Block::Table(Table {
                grid: rows,
                header_rows: 0,
                kind: TableKind::Data,
            })
        };
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "xlsx",
        };
        let report = sheet(vec![
            vec![title("Q1 report", 3), covered(), covered()],
            vec![text(""), text(""), text("")],
            vec![text("Team"), text("City"), text("Cups")],
            vec![text("Blues"), text("STL"), text("1")],
        ]);
        let heading = Block::heading(2, vec![Inline::plain("Sheet1")]);
        assert_eq!(
            renderer.blocks(&[heading, report.clone()]),
            "## Sheet1\n\nQ1 report\n\n| Team | City | Cups |\n| --- | --- | --- |\n| Blues | STL | 1 |"
        );
        // The title is lifted out with its merge, which no longer warns.
        assert!(!renderer.merged_cells);
        // A title the sheet's heading already gives is not repeated, and the
        // table follows its heading directly, as every sheet does.
        let named = Block::heading(2, vec![Inline::plain("Q1 report")]);
        assert_eq!(
            renderer.blocks(&[named, report]),
            "## Q1 report\n| Team | City | Cups |\n| --- | --- | --- |\n| Blues | STL | 1 |"
        );
        // A merge over some columns only groups them, and a title above a
        // row that is not a header stays the header.
        for grid in [
            vec![
                vec![title("Q1", 2), covered(), text("")],
                vec![text("Jan"), text("Feb"), text("Total")],
            ],
            vec![
                vec![title("Notes", 2), covered()],
                vec![text("only one"), text("")],
            ],
        ] {
            assert!(renderer.blocks(&[sheet(grid)]).starts_with("| "));
        }
        // A document's merged first row is never lifted.
        renderer.extension = "docx";
        let report = sheet(vec![
            vec![title("Q1 report", 2), covered()],
            vec![text("Team"), text("City")],
        ]);
        assert!(renderer.blocks(&[report]).starts_with("|  |  |\n"));
    }

    #[test]
    fn sheet_trimming_does_not_affect_later_tables() {
        let merged = Block::Table(Table {
            grid: vec![vec![
                CellSlot::Origin(Cell::spanning(
                    vec![Block::Paragraph(vec![Inline::plain("Title")])],
                    2,
                    1,
                )),
                CellSlot::Covered {
                    origin_row: 0,
                    origin_col: 0,
                },
                CellSlot::Origin(Cell::new(vec![])),
            ]],
            header_rows: 1,
            kind: TableKind::Data,
        });
        let plain = Block::Table(Table::from_rows(
            vec![vec![
                Cell::from_inlines(vec![Inline::plain("One")]),
                Cell::new(vec![]),
            ]],
            1,
            TableKind::Data,
        ));
        let mut renderer = Renderer {
            asset_names: &[],
            merged_cells: false,
            in_cell: false,
            anchors: BTreeSet::new(),
            extension: "ods",
        };
        assert_eq!(
            renderer.blocks(&[merged, plain]),
            "| Title |\n| --- |\n\n| One |\n| --- |"
        );
        assert!(renderer.merged_cells);
    }

    #[test]
    fn image_references_inside_table_links_are_retained() {
        let blocks = [Block::Table(Table::from_rows(
            vec![vec![Cell::from_inlines(vec![Inline::Link {
                content: vec![Inline::Image {
                    alt: "Figure".into(),
                    source: ImageSource::Asset(AssetId(2)),
                }],
                target: LinkTarget::External("https://example.test".into()),
            }])]],
            0,
            TableKind::Data,
        ))];
        let mut assets = BTreeSet::new();
        references(&blocks, &mut BTreeSet::new(), &mut assets);
        assert_eq!(assets.into_iter().collect::<Vec<_>>(), [2]);
    }

    #[test]
    fn text_that_reads_as_html_is_escaped() {
        // Code pasted into a document: a renderer would take `<int>` and
        // `<iostream>` for tags and `&copy;` for an entity, and show neither.
        let rtf = br"{\rtf1\ansi #include <iostream>\par std::vector<int> v; a < b, a<3 & b\par &copy; &#169; AT&T; x&y\par}";
        let doc = extract(rtf, "rtf").unwrap();
        assert_eq!(
            doc.markdown,
            "#include \\<iostream>\n\nstd::vector\\<int> v; a < b, a<3 & b\n\n\\&copy; \\&#169; AT\\&T; x&y\n"
        );
    }

    #[test]
    fn a_title_set_by_hand_in_a_word_97_file_is_a_heading() {
        // The exporter saves `<h1>` as a bold 24-point paragraph over 12-point
        // text, with no heading style or outline level (anydoc's
        // `shared::visual`); the other paragraphs stay as they were.
        let doc = extract(include_bytes!("native/fixtures/textedit-word97.doc"), "doc").unwrap();
        assert!(
            doc.markdown
                .starts_with("# Compound repair\n\nThis paragraph was written"),
            "{}",
            doc.markdown
        );
        assert!(
            doc.markdown
                .contains("\n\n- First listed point\n- Second listed point\n\n"),
            "{}",
            doc.markdown
        );
        assert_eq!(doc.markdown.matches('#').count(), 1, "{}", doc.markdown);
    }
}

#[cfg(test)]
#[path = "native/office_image_tests.rs"]
mod office_image_tests;
