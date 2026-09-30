use crate::{Asset, Document, Error, Result};
use anydoc::model::{Block, CellSlot, ImageSource, Inline, LinkTarget};
use std::collections::BTreeSet;

#[path = "office.rs"]
mod office;
pub(crate) use office::extract_presentation_count;
#[path = "native/compound.rs"]
mod compound;
#[path = "office_meta.rs"]
mod office_meta;
#[path = "pdf.rs"]
pub(super) mod pdf;

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

fn escape(text: &str) -> String {
    let mut output = String::new();
    for ch in text.chars() {
        if matches!(ch, '\\' | '*' | '_' | '[' | ']' | '`') {
            output.push('\\');
        }
        output.push(ch);
    }
    output
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

impl Renderer<'_> {
    fn inlines(&self, values: &[Inline]) -> String {
        let values = merged_runs(values);
        let mut output = String::new();
        for value in values.iter() {
            match value {
                Inline::Text { text, style } => {
                    if text.trim().is_empty() {
                        output.push_str(text);
                        continue;
                    }
                    let trimmed = text.trim();
                    let prefix = &text[..text.len() - text.trim_start().len()];
                    let suffix = &text[text.trim_end().len()..];
                    let mut rendered = if style.code {
                        let max_ticks =
                            trimmed.split(|c| c != '`').map(str::len).max().unwrap_or(0);
                        let ticks = "`".repeat(max_ticks + 1);
                        if trimmed.starts_with('`') || trimmed.ends_with('`') {
                            format!("{ticks} {trimmed} {ticks}")
                        } else {
                            format!("{ticks}{trimmed}{ticks}")
                        }
                    } else {
                        escape(trimmed)
                    };
                    if style.bold {
                        rendered = format!("**{rendered}**");
                    }
                    if style.italic {
                        rendered = format!("*{rendered}*");
                    }
                    if style.strike {
                        rendered = format!("~~{rendered}~~");
                    }
                    output.push_str(prefix);
                    output.push_str(&rendered);
                    output.push_str(suffix);
                }
                Inline::Link { content, target } => {
                    let label = self.inlines(content);
                    let target = match target {
                        LinkTarget::Anchor(anchor) => format!("#{}", destination(anchor)),
                        LinkTarget::External(url) | LinkTarget::Relative(url) => destination(url),
                    };
                    let safe = url::Url::parse(&target)
                        .map(|url| matches!(url.scheme(), "http" | "https" | "mailto" | "tel"))
                        .unwrap_or(true);
                    if target.is_empty() || !safe {
                        output.push_str(&label);
                    } else {
                        output.push_str(&format!("[{label}]({target})"));
                    }
                }
                Inline::Image { alt, source } => {
                    let target = match source {
                        ImageSource::Asset(id) => self
                            .asset_names
                            .get(id.0)
                            .map(|name| format!(".markitai/assets/{name}")),
                        ImageSource::External(url) => Some(destination(url)),
                        ImageSource::Unavailable => None,
                    };
                    if let Some(target) = target {
                        output.push_str(&format!("![{}]({target})", escape(alt)));
                    } else {
                        output.push_str(&escape(alt));
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
                    output.push_str(&format!("<a id=\"{anchor}\"></a>"));
                }
                Inline::NoteRef(id) => output.push_str(&format!("[^{}]", destination(id))),
                Inline::LineBreak => output.push_str("  \n"),
                Inline::Math(text) => output.push_str(&format!("${text}$")),
                Inline::Checkbox(checked) => {
                    output.push_str(if *checked { "[x] " } else { "[ ] " })
                }
            }
        }
        output
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
        let mut parts = Vec::new();
        for block in blocks {
            let text = match block {
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
        for block in blocks {
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
                    let heading = format!(
                        "{} {}",
                        "#".repeat(usize::from((*level).clamp(1, 6))),
                        self.inlines(content).trim_end()
                    );
                    if let Some(anchor) = anchor
                        .as_ref()
                        .filter(|anchor| self.anchors.contains(*anchor))
                    {
                        format!(
                            "{}\n{heading}",
                            self.inlines(&[Inline::Anchor(anchor.clone())])
                        )
                    } else {
                        heading
                    }
                }
                Block::Paragraph(values) => self.inlines(values).trim_end().to_owned(),
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
                Block::List(list) => list
                    .items
                    .iter()
                    .enumerate()
                    .filter_map(|(index, item)| {
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
                            return None;
                        }
                        let indent = " ".repeat(marker.len() + 1);
                        let mut lines = content.lines();
                        let mut item_text = format!("{marker} {}", lines.next().unwrap_or(""));
                        for line in lines {
                            item_text.push_str(&format!("\n{indent}{line}"));
                        }
                        Some(item_text.trim_end().to_owned())
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
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
                                        self.merged_cells |= cell.row_span > 1 || cell.col_span > 1;
                                        // Keep inline Markdown, escape table syntax only.
                                        self.cell_text(&cell.blocks)
                                            .trim()
                                            .replace('|', "\\|")
                                            .replace('\n', "<br>")
                                    }
                                })
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    if self.extension == "ods" {
                        let content_width = rows
                            .iter()
                            .filter_map(|row| row.iter().rposition(|cell| !cell.is_empty()))
                            .max()
                            .map(|last| last + 1)
                            .unwrap_or(0);
                        let span_width = table
                            .grid
                            .iter()
                            .flat_map(|row| row.iter().enumerate())
                            .filter_map(|(column, slot)| match slot {
                                CellSlot::Origin(cell)
                                    if cell.col_span > 1 || cell.row_span > 1 =>
                                {
                                    Some(column.saturating_add(cell.col_span as usize))
                                }
                                _ => None,
                            })
                            .max()
                            .unwrap_or(0);
                        let width = content_width.max(span_width);
                        for row in &mut rows {
                            row.truncate(width);
                        }
                    }
                    // A sheet's first row is its header, as the reference's
                    // spreadsheet readers take it, whether or not the source
                    // marks it; a blank header row would only push it down.
                    let header = if matches!(self.extension, "docx" | "docm") {
                        false
                    } else {
                        matches!(self.extension, "ods" | "xlsx" | "xlsm" | "xls" | "xlsb")
                            || table.header_rows > 0
                    };
                    super::text::table(&rows, header).trim_end().to_owned()
                }
            };
            if !rendered.is_empty() {
                let table = matches!(block, Block::Table(_));
                if !joined.is_empty() {
                    joined.push_str(if sheet_layout && previous_heading && table {
                        "\n"
                    } else {
                        "\n\n"
                    });
                }
                joined.push_str(&rendered);
                previous_heading = matches!(block, Block::Heading { .. });
            }
        }
        joined
    }
}

pub(super) fn extract(bytes: &[u8], extension: &str) -> Result<Document> {
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
                None => return Err(conversion_error(error)),
            }
        }
        result => (result.map_err(conversion_error)?, None),
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
        anchors,
        extension,
    };
    let mut markdown = renderer.blocks(&parsed.blocks);
    let preamble = metadata.preamble();
    if !preamble.is_empty() {
        markdown = format!("{preamble}\n\n{markdown}");
    }
    for note in &parsed.notes {
        let text = renderer.blocks(&note.blocks);
        let mut lines = text.lines();
        markdown.push_str(&format!(
            "\n\n[^{}]: {}",
            destination(&note.id),
            lines.next().unwrap_or("")
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
mod tests {
    use super::*;
    use anydoc::model::{AssetId, Cell, Style, Table, TableKind};
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
            anchors,
            extension: "doc",
        };
        let markdown = renderer.blocks(&blocks);
        assert!(markdown.contains("<a id=\"target\"></a>\n# Target"));
        assert!(markdown.contains("[jump](#target)"));
        assert!(!markdown.contains("unused"));
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
            anchors: BTreeSet::new(),
            extension: "docx",
        };
        assert!(
            renderer
                .blocks(std::slice::from_ref(&table))
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
            "1. First\n2. Second\n1-a) Composite"
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
    fn sheet_trimming_preserves_merged_extent_without_affecting_later_tables() {
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
            anchors: BTreeSet::new(),
            extension: "ods",
        };
        assert_eq!(
            renderer.blocks(&[merged, plain]),
            "| Title |  |\n| --- | --- |\n\n| One |\n| --- |"
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
}
