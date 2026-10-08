use crate::opc;
use std::collections::BTreeMap;
use std::io::{Cursor, Read};

const MAX_METADATA_BYTES: u64 = 16 * 1024 * 1024;
const MAX_METADATA_ELEMENTS: usize = 100_000;
// Descendant text is retained by its ancestors for EPUB mixed-content fields.
// Bound the total copies, including completed elements, rather than each field.
const MAX_METADATA_TEXT_BYTES: usize = MAX_METADATA_BYTES as usize;

#[derive(Default)]
pub(super) struct Metadata {
    pub sheets: Vec<String>,
    pub book: BTreeMap<String, Vec<String>>,
    pub warnings: Vec<String>,
}

impl Metadata {
    pub fn title(&self) -> Option<&str> {
        self.book.get("title")?.first().map(String::as_str)
    }

    pub fn preamble(&self) -> String {
        [
            "title",
            "authors",
            "language",
            "publisher",
            "date",
            "description",
            "identifier",
        ]
        .into_iter()
        .filter_map(|key| {
            let values = self.book.get(key)?;
            let value = if key == "authors" {
                values.join(", ")
            } else {
                values.first()?.clone()
            };
            if value.is_empty() {
                return None;
            }
            let label = format!("{}{}", key[..1].to_ascii_uppercase(), &key[1..]);
            Some(format!("**{label}:** {value}"))
        })
        .collect::<Vec<_>>()
        .join("\n")
    }
}

fn zip_text(zip: &mut opc::Zip, name: &str) -> Result<String, String> {
    String::from_utf8(zip_bytes(zip, name)?).map_err(|e| e.to_string())
}

fn zip_bytes(zip: &mut opc::Zip, name: &str) -> Result<Vec<u8>, String> {
    zip.read(name, MAX_METADATA_BYTES)?
        .ok_or_else(|| format!("metadata part {name} is missing"))
}

#[derive(Default)]
struct Element {
    name: String,
    attributes: BTreeMap<String, String>,
    text: String,
}

fn append_text(stack: &mut [Element], text: &str, remaining: &mut usize) -> Result<(), String> {
    let bytes = text
        .len()
        .checked_mul(stack.len())
        .filter(|bytes| *bytes <= *remaining)
        .ok_or("metadata text exceeds the aggregate size limit")?;
    *remaining -= bytes;
    for item in stack {
        item.text.push_str(text);
    }
    Ok(())
}

fn elements(xml: &str) -> Result<Vec<Element>, String> {
    elements_with_limits(xml, MAX_METADATA_TEXT_BYTES, MAX_METADATA_ELEMENTS)
}

fn elements_with_limits(
    xml: &str,
    mut remaining_text: usize,
    max_elements: usize,
) -> Result<Vec<Element>, String> {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_str(xml);
    let mut stack: Vec<Element> = Vec::new();
    let mut output = Vec::new();
    let mut element_count = 0usize;
    loop {
        let event = reader.read_event().map_err(|e| e.to_string())?;
        let empty = matches!(event, Event::Empty(_));
        match event {
            Event::Start(event) | Event::Empty(event) => {
                if element_count >= max_elements {
                    return Err("metadata element count exceeds the limit".into());
                }
                element_count += 1;
                let mut element = Element {
                    name: String::from_utf8_lossy(event.local_name().as_ref()).into_owned(),
                    ..Element::default()
                };
                for attr in event.attributes() {
                    let attr = attr.map_err(|e| e.to_string())?;
                    element.attributes.insert(
                        String::from_utf8_lossy(attr.key.local_name().as_ref()).into_owned(),
                        attr.decoded_and_normalized_value(
                            quick_xml::XmlVersion::Implicit1_0,
                            reader.decoder(),
                        )
                        .map_err(|e| e.to_string())?
                        .into_owned(),
                    );
                }
                if empty {
                    output.push(element);
                } else {
                    if stack.len() >= 256 {
                        return Err("metadata nesting exceeds the limit".into());
                    }
                    stack.push(element);
                }
            }
            Event::Text(text) => {
                let decoded = text.decode().map_err(|e| e.to_string())?;
                let decoded = quick_xml::escape::unescape(&decoded).map_err(|e| e.to_string())?;
                append_text(&mut stack, &decoded, &mut remaining_text)?;
            }
            Event::CData(text) => {
                let decoded = text.decode().map_err(|e| e.to_string())?;
                append_text(&mut stack, &decoded, &mut remaining_text)?;
            }
            Event::GeneralRef(reference) => {
                let name = reference.decode().map_err(|e| e.to_string())?;
                let entity = format!("&{name};");
                let decoded = quick_xml::escape::unescape(&entity).map_err(|e| e.to_string())?;
                append_text(&mut stack, &decoded, &mut remaining_text)?;
            }
            Event::End(_) => {
                if let Some(mut item) = stack.pop() {
                    // Trim without allocating another copy of a large field.
                    item.text.truncate(item.text.trim_end().len());
                    let leading = item.text.len() - item.text.trim_start().len();
                    item.text.drain(..leading);
                    output.push(item);
                }
            }
            Event::DocType(_) => {
                return Err("document types are not accepted in package metadata".into());
            }
            Event::Eof => break,
            _ => {}
        }
    }
    if !stack.is_empty() {
        return Err("incomplete package metadata".into());
    }
    Ok(output)
}

fn read_zip(bytes: &[u8], extension: &str) -> Result<Metadata, String> {
    let mut archive = opc::Zip::open(bytes, opc::MAX_ENTRIES)?;
    let mut result = Metadata::default();
    if extension == "epub" {
        let container = elements(&zip_text(&mut archive, "META-INF/container.xml")?)?;
        let path = container
            .iter()
            .find(|el| el.name == "rootfile")
            .and_then(|el| el.attributes.get("full-path"))
            .ok_or("EPUB package path missing")?;
        let package = elements(&zip_text(&mut archive, path)?)?;
        for element in package {
            let key = match element.name.as_str() {
                "creator" => "authors",
                "title" | "language" | "publisher" | "date" | "description" | "identifier" => {
                    &element.name
                }
                _ => continue,
            };
            if !element.text.is_empty() {
                result
                    .book
                    .entry(key.into())
                    .or_default()
                    .push(element.text);
            }
        }
    } else {
        let path = zip_text(&mut archive, "_rels/.rels")
            .ok()
            .and_then(|xml| elements(&xml).ok())
            .and_then(|parts| {
                parts
                    .into_iter()
                    .find(|el| {
                        el.name == "Relationship"
                            && el
                                .attributes
                                .get("Type")
                                .is_some_and(|t| t.ends_with("/officeDocument"))
                    })
                    .and_then(|el| el.attributes.get("Target").cloned())
            })
            .map(|target| opc::resolve("", &target))
            .transpose()?
            .unwrap_or_else(|| {
                if extension == "xlsb" {
                    "xl/workbook.bin".into()
                } else {
                    "xl/workbook.xml".into()
                }
            });
        if extension == "xlsb" {
            return xlsb_sheet_names(&zip_bytes(&mut archive, &path)?);
        }
        for sheet in elements(&zip_text(&mut archive, &path)?)?
            .into_iter()
            .filter(|el| el.name == "sheet")
        {
            if sheet
                .attributes
                .get("state")
                .is_some_and(|s| s != "visible")
            {
                result.warnings.push(format!(
                    "Hidden worksheet {:?} is omitted by the native spreadsheet reader.",
                    sheet
                        .attributes
                        .get("name")
                        .map(String::as_str)
                        .unwrap_or("")
                ));
                continue;
            }
            if let Some(name) = sheet.attributes.get("name") {
                result.sheets.push(name.clone());
            }
        }
    }
    Ok(result)
}

/// What a Word package holds that the Markdown leaves out and a reader would
/// want to know about: its review comments, which carry text of their own.
/// Missing optional comments are normal. A present but unreadable comments
/// part must not silently look like a document with no review comments.
fn read_docx(bytes: &[u8]) -> Metadata {
    let mut archive = match opc::Zip::open(bytes, opc::MAX_ENTRIES) {
        Ok(archive) => archive,
        // The document reader reports a package that is not a ZIP itself, but
        // it converts some that `opc::Zip` refuses (a repeated name, more
        // entries): their comments are present but unread, not absent.
        Err(message) => {
            return match zip::ZipArchive::new(Cursor::new(bytes)) {
                Ok(raw) if raw.file_names().any(|name| name == "word/comments.xml") => Metadata {
                    warnings: vec![format!(
                        "Word review comments could not be recovered: {message}"
                    )],
                    ..Metadata::default()
                },
                _ => Metadata::default(),
            };
        }
    };
    if !archive
        .archive
        .file_names()
        .any(|name| name == "word/comments.xml")
    {
        return Metadata::default();
    }
    match zip_text(&mut archive, "word/comments.xml").and_then(|xml| elements(&xml)) {
        Ok(elements) => comment_warning(
            elements
                .iter()
                .filter(|element| element.name == "comment")
                .count(),
        ),
        Err(message) => Metadata {
            warnings: vec![format!(
                "Word review comments could not be recovered: {message}"
            )],
            ..Metadata::default()
        },
    }
}

/// An OpenDocument text's comments (`office:annotation`), which the
/// Markdown leaves out as it does Word's. Counted while streaming
/// `content.xml`, without building it, since the document reader builds it
/// anyway. As for Word, comments that cannot be counted are reported rather
/// than taken for none.
fn read_odt(bytes: &[u8]) -> Metadata {
    let mut archive = match opc::Zip::open(bytes, opc::MAX_ENTRIES) {
        Ok(archive) => archive,
        // The document reader reports an invalid package itself.
        Err(_) if zip::ZipArchive::new(Cursor::new(bytes)).is_err() => {
            return Metadata::default();
        }
        // It converts some packages that `opc::Zip` refuses, as for Word.
        Err(message) => {
            return Metadata {
                warnings: vec![format!(
                    "OpenDocument comments could not be recovered: {message}"
                )],
                ..Metadata::default()
            };
        }
    };
    let count = zip_text(&mut archive, "content.xml").and_then(|xml| {
        let mut reader = quick_xml::Reader::from_str(&xml);
        let mut count = 0usize;
        loop {
            match reader.read_event().map_err(|e| e.to_string())? {
                quick_xml::events::Event::Start(event) | quick_xml::events::Event::Empty(event)
                    if event.local_name().as_ref() == b"annotation" =>
                {
                    count += 1;
                }
                quick_xml::events::Event::Eof => break,
                _ => {}
            }
        }
        Ok(count)
    });
    match count {
        Ok(count) => comment_warning(count),
        Err(message) => Metadata {
            warnings: vec![format!(
                "OpenDocument comments could not be recovered: {message}"
            )],
            ..Metadata::default()
        },
    }
}

fn comment_warning(count: usize) -> Metadata {
    let mut result = Metadata::default();
    if count > 0 {
        result.warnings.push(format!(
            "The document has {count} review comment{}; comments are not included in the Markdown.",
            if count == 1 { "" } else { "s" }
        ));
    }
    result
}

fn biff_sheet_names(bytes: &[u8]) -> Result<Metadata, String> {
    let mut result = Metadata::default();
    let mut position = 0;
    let mut unicode = true;
    while position + 4 <= bytes.len() {
        let id = u16::from_le_bytes([bytes[position], bytes[position + 1]]);
        let len = usize::from(u16::from_le_bytes([
            bytes[position + 2],
            bytes[position + 3],
        ]));
        position += 4;
        let data = bytes
            .get(position..position + len)
            .ok_or("truncated workbook metadata")?;
        position += len;
        if id == 0x0809 && data.len() >= 2 {
            unicode = u16::from_le_bytes([data[0], data[1]]) >= 0x0600;
        }
        if id == 0x000a {
            break;
        }
        if id != 0x0085 || data.len() < 7 {
            continue;
        }
        if data[5] != 0 {
            continue;
        }
        let len = usize::from(data[6]);
        let name = if unicode {
            let flag = *data.get(7).ok_or("truncated worksheet name")?;
            if flag & 1 != 0 {
                let encoded = data
                    .get(8..8 + len * 2)
                    .ok_or("truncated UTF-16 worksheet name")?;
                String::from_utf16(
                    &encoded
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .map(|v| u16::from_le_bytes([v[0], v[1]]))
                        .collect::<Vec<_>>(),
                )
                .map_err(|e| e.to_string())?
            } else {
                data.get(8..8 + len)
                    .ok_or("truncated worksheet name")?
                    .iter()
                    .map(|&byte| char::from(byte))
                    .collect()
            }
        } else {
            encoding_rs::WINDOWS_1252
                .decode(data.get(7..7 + len).ok_or("truncated worksheet name")?)
                .0
                .into_owned()
        };
        if data[4] != 0 {
            result.warnings.push(format!(
                "Hidden worksheet {name:?} is omitted by the native spreadsheet reader."
            ));
        } else {
            result.sheets.push(name);
        }
    }
    Ok(result)
}

/// Sheet names from an XLSB `workbook.bin`, a stream of records (MS-XLSB
/// 2.1.4) whose type and size are little-endian groups of seven bits. Each
/// BrtBundleSh names a sheet: its state (1 hidden, 2 very hidden, which the
/// spreadsheet reader omits), tab id, relationship id and name.
fn xlsb_sheet_names(bytes: &[u8]) -> Result<Metadata, String> {
    const BRT_BUNDLE_SH: usize = 156;
    fn number(bytes: &[u8], position: &mut usize, groups: usize) -> Result<usize, String> {
        let mut value = 0;
        for group in 0..groups {
            let byte = *bytes.get(*position).ok_or("truncated workbook record")?;
            *position += 1;
            value |= usize::from(byte & 0x7f) << (7 * group);
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err("invalid workbook record header".into())
    }
    fn u32_at(data: &[u8], position: &mut usize) -> Result<u32, String> {
        let value = data
            .get(*position..*position + 4)
            .ok_or("truncated worksheet record")?;
        *position += 4;
        Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
    }
    /// An XLWideString, or none for the null XLNullableWideString.
    fn wide(data: &[u8], position: &mut usize) -> Result<Option<String>, String> {
        let units = u32_at(data, position)?;
        if units == u32::MAX {
            return Ok(None);
        }
        let end = usize::try_from(units)
            .ok()
            .and_then(|units| units.checked_mul(2))
            .and_then(|length| position.checked_add(length))
            .ok_or("truncated worksheet name")?;
        let encoded = data.get(*position..end).ok_or("truncated worksheet name")?;
        *position = end;
        let units: Vec<u16> = encoded
            .as_chunks::<2>()
            .0
            .iter()
            .map(|unit| u16::from_le_bytes(*unit))
            .collect();
        String::from_utf16(&units)
            .map(Some)
            .map_err(|e| e.to_string())
    }
    let mut result = Metadata::default();
    let mut position = 0;
    while position < bytes.len() {
        let id = number(bytes, &mut position, 2)?;
        let size = number(bytes, &mut position, 4)?;
        let data = position
            .checked_add(size)
            .and_then(|end| bytes.get(position..end))
            .ok_or("truncated workbook record")?;
        position += size;
        if id != BRT_BUNDLE_SH {
            continue;
        }
        let mut field = 0;
        let state = u32_at(data, &mut field)?;
        u32_at(data, &mut field)?;
        wide(data, &mut field)?;
        let name = wide(data, &mut field)?.ok_or("worksheet has no name")?;
        if matches!(state, 1 | 2) {
            result.warnings.push(format!(
                "Hidden worksheet {name:?} is omitted by the native spreadsheet reader."
            ));
        } else {
            result.sheets.push(name);
        }
    }
    Ok(result)
}

pub(super) fn read(bytes: &[u8], extension: &str) -> Metadata {
    let parsed = match extension {
        "docx" | "docm" => return read_docx(bytes),
        "odt" => return read_odt(bytes),
        "xlsx" | "xlsm" | "xlsb" | "epub" => read_zip(bytes, extension),
        "xls" => (|| {
            let mut compound =
                cfb::CompoundFile::open(Cursor::new(bytes)).map_err(|e| e.to_string())?;
            let name = if compound.is_stream("/Workbook") {
                "/Workbook"
            } else {
                "/Book"
            };
            let stream = compound.open_stream(name).map_err(|e| e.to_string())?;
            let mut workbook = Vec::new();
            stream
                .take(MAX_METADATA_BYTES + 1)
                .read_to_end(&mut workbook)
                .map_err(|e| e.to_string())?;
            if workbook.len() as u64 > MAX_METADATA_BYTES {
                return Err("workbook metadata exceeds the size limit".into());
            }
            biff_sheet_names(&workbook)
        })(),
        _ => return Metadata::default(),
    };
    parsed.unwrap_or_else(|message| Metadata {
        warnings: vec![format!(
            "Package metadata could not be recovered: {message}"
        )],
        ..Metadata::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn archive(parts: &[(&str, &str)]) -> Vec<u8> {
        let mut zip = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, content) in parts {
            zip.start_file(*name, zip::write::SimpleFileOptions::default())
                .unwrap();
            zip.write_all(content.as_bytes()).unwrap();
        }
        zip.finish().unwrap().into_inner()
    }

    #[test]
    fn xlsx_names_follow_package_relationships_and_report_hidden_sheets() {
        let bytes = archive(&[
            (
                "_rels/.rels",
                "<Relationships><Relationship Type='urn:test/officeDocument' Target='/custom/book.xml'/></Relationships>",
            ),
            (
                "custom/book.xml",
                "<workbook><sheets><sheet name='A &amp; B'/><sheet name='Hidden' state='veryHidden'/></sheets></workbook>",
            ),
        ]);
        let metadata = read(&bytes, "xlsx");
        assert_eq!(metadata.sheets, ["A & B"]);
        assert_eq!(metadata.warnings.len(), 1);
        assert!(metadata.warnings[0].contains("Hidden"));
    }

    #[test]
    fn the_workbook_target_is_resolved_as_a_relationship_reference() {
        let rels = |target: &str| {
            format!(
                "<Relationships><Relationship Type='urn:test/officeDocument' Target='{target}'/></Relationships>"
            )
        };
        let book = "<workbook><sheets><sheet name='One'/></sheets></workbook>";
        let escaped = rels("./custom/my%20book.xml");
        let bytes = archive(&[("_rels/.rels", &escaped), ("custom/my book.xml", book)]);
        assert_eq!(read(&bytes, "xlsx").sheets, ["One"]);
        let outside = rels("../custom/book.xml");
        let bytes = archive(&[("_rels/.rels", &outside), ("custom/book.xml", book)]);
        let metadata = read(&bytes, "xlsx");
        assert!(metadata.sheets.is_empty());
        assert!(metadata.warnings[0].contains("escapes package root"));
    }

    #[test]
    fn epub_preamble_retains_source_field_order_and_authors() {
        let bytes = archive(&[
            (
                "META-INF/container.xml",
                "<container><rootfiles><rootfile full-path='book/package.opf'/></rootfiles></container>",
            ),
            (
                "book/package.opf",
                "<package><metadata xmlns:dc='urn:dc'><dc:creator>A</dc:creator><dc:title>Book</dc:title><dc:creator>B</dc:creator><dc:language>zh</dc:language><dc:identifier>ID</dc:identifier></metadata></package>",
            ),
        ]);
        let metadata = read(&bytes, "epub");
        assert_eq!(metadata.title(), Some("Book"));
        assert_eq!(
            metadata.preamble(),
            "**Title:** Book\n**Authors:** A, B\n**Language:** zh\n**Identifier:** ID"
        );
    }
    #[test]
    fn package_entities_and_book_fields_are_preserved() {
        let values = elements("<metadata xmlns:dc='urn:dc'><dc:title>A &amp; B</dc:title><dc:creator>作者</dc:creator><dc:identifier id='a'>urn:one</dc:identifier></metadata>").unwrap();
        assert_eq!(values[0].text, "A & B");
        assert_eq!(values[1].text, "作者");
        assert_eq!(values[2].attributes["id"], "a");
        assert!(elements("<!DOCTYPE x><x/>").is_err());
    }

    #[test]
    fn mixed_content_preserves_ancestor_text_without_truncation() {
        let values = elements("<metadata><description> 前 <b>bold &amp; &#x4E2D;</b><![CDATA[<tail>]]> 后 </description></metadata>").unwrap();
        let description = values
            .iter()
            .find(|item| item.name == "description")
            .unwrap();
        assert_eq!(description.text, "前 bold & 中<tail> 后");
        assert_eq!(values.last().unwrap().text, description.text);
    }

    #[test]
    fn aggregate_text_budget_counts_ancestors_and_completed_siblings() {
        // Four copies of "abc" across two siblings and their parent: 12 bytes.
        let xml = "<a><b>abc</b><c>abc</c></a>";
        assert!(elements_with_limits(xml, 12, 3).is_ok());
        let error = elements_with_limits(xml, 11, 3).err().unwrap();
        assert!(error.contains("aggregate size limit"));
        // Three event kinds must consume the same shared byte budget.
        for xml in [
            "<a><b>abc</b></a>",
            "<a><b><![CDATA[abc]]></b></a>",
            "<a><b>&#x4E2D;</b></a>",
        ] {
            assert!(elements_with_limits(xml, 6, 2).is_ok());
            assert!(elements_with_limits(xml, 5, 2).is_err());
        }
    }

    #[test]
    fn metadata_limits_reject_amplification_before_large_allocations() {
        let xml = format!(
            "{}{}{}",
            "<a>".repeat(255),
            "x".repeat(1024),
            "</a>".repeat(255)
        );
        assert!(
            elements_with_limits(&xml, 4096, 256)
                .err()
                .unwrap()
                .contains("aggregate size limit")
        );
        assert!(elements_with_limits("<a><b/><c/></a>", 0, 3).is_ok());
        assert!(
            elements_with_limits("<a><b/><c/></a>", 0, 2)
                .err()
                .unwrap()
                .contains("element count")
        );
        assert!(elements_with_limits("<a><b></b><c/></a>", 0, 2).is_err());
    }

    #[test]
    fn docx_comment_parts_warn_on_limits_or_invalid_xml_but_not_when_absent() {
        let missing = archive(&[("word/document.xml", "<document/>")]);
        assert!(read(&missing, "docx").warnings.is_empty());
        let valid = archive(&[(
            "word/comments.xml",
            "<comments><comment><text>review</text></comment></comments>",
        )]);
        assert_eq!(read(&valid, "docx").warnings.len(), 1);
        assert!(read(&valid, "docx").warnings[0].contains("1 review comment"));
        let expanded = format!(
            "<comments><comment>{}{}{}</comment></comments>",
            "<a>".repeat(250),
            "x".repeat(70_000),
            "</a>".repeat(250)
        );
        for (xml, reason) in [
            ("<comments><comment>", "incomplete package metadata"),
            (expanded.as_str(), "aggregate size limit"),
        ] {
            let bytes = archive(&[("word/comments.xml", xml)]);
            let metadata = read(&bytes, "docx");
            assert_eq!(metadata.warnings.len(), 1);
            assert!(metadata.warnings[0].contains("Word review comments could not be recovered"));
            assert!(metadata.warnings[0].contains(reason));
        }
    }

    #[test]
    fn comments_in_a_package_only_the_document_reader_accepts_are_reported() {
        let repeat = |bytes: Vec<u8>, from: &[u8], to: &[u8]| {
            let mut bytes = bytes;
            for at in 0..=bytes.len() - from.len() {
                if bytes[at..].starts_with(from) {
                    bytes[at..at + to.len()].copy_from_slice(to);
                }
            }
            bytes
        };
        let docx = repeat(
            archive(&[
                ("word/comments.xml", "<comments><comment/></comments>"),
                ("word/document.xml", "<document/>"),
                ("word/commentz.xml", "<comments/>"),
            ]),
            b"word/commentz.xml",
            b"word/comments.xml",
        );
        let metadata = read(&docx, "docx");
        assert_eq!(metadata.warnings.len(), 1, "{:?}", metadata.warnings);
        assert!(metadata.warnings[0].contains("Word review comments could not be recovered"));
        assert!(metadata.warnings[0].contains("repeats an entry name"));
        let plain = repeat(
            archive(&[
                ("word/document.xml", "<document/>"),
                ("word/documenz.xml", ""),
            ]),
            b"word/documenz.xml",
            b"word/document.xml",
        );
        assert!(read(&plain, "docx").warnings.is_empty());
        let odt = repeat(
            archive(&[("content.xml", "<x/>"), ("contenz.xml", "<x/>")]),
            b"contenz.xml",
            b"content.xml",
        );
        let metadata = read(&odt, "odt");
        assert_eq!(metadata.warnings.len(), 1, "{:?}", metadata.warnings);
        assert!(metadata.warnings[0].contains("OpenDocument comments could not be recovered"));
        assert!(read(b"not a zip", "docx").warnings.is_empty());
        assert!(read(b"not a zip", "odt").warnings.is_empty());
    }

    #[test]
    fn odt_comments_warn_when_counted_or_unreadable() {
        let content = |body: &str| {
            format!(
                "<office:document-content xmlns:office=\"urn:oasis:names:tc:opendocument:xmlns:office:1.0\"><office:body>{body}</office:body></office:document-content>"
            )
        };
        let none = archive(&[("content.xml", &content("<p/>"))]);
        assert!(read(&none, "odt").warnings.is_empty());
        let two = archive(&[(
            "content.xml",
            &content("<office:annotation/><office:annotation><p/></office:annotation>"),
        )]);
        assert_eq!(read(&two, "odt").warnings.len(), 1);
        assert!(read(&two, "odt").warnings[0].contains("2 review comments"));
        let malformed = archive(&[(
            "content.xml",
            "<office:document-content><office:annotation/></office:body>",
        )]);
        let metadata = read(&malformed, "odt");
        assert_eq!(metadata.warnings.len(), 1, "{:?}", metadata.warnings);
        assert!(metadata.warnings[0].contains("OpenDocument comments could not be recovered"));
    }

    #[test]
    fn a_truncated_xlsb_workbook_record_is_reported() {
        // Record type 0 declares five bytes and holds two.
        let bytes = archive(&[("xl/workbook.bin", "\u{0}\u{5}ab")]);
        let metadata = read(&bytes, "xlsb");
        assert!(metadata.sheets.is_empty());
        assert_eq!(metadata.warnings.len(), 1);
        assert!(metadata.warnings[0].contains("truncated workbook record"));
    }

    #[test]
    fn biff_unicode_sheet_name_and_hidden_status() {
        let mut bytes = vec![0x09, 0x08, 2, 0, 0, 6];
        let mut record = vec![0, 0, 0, 0, 0, 0, 2, 1];
        for ch in "数据".encode_utf16() {
            record.extend(ch.to_le_bytes());
        }
        bytes.extend([0x85, 0, record.len() as u8, 0]);
        bytes.extend(&record);
        assert_eq!(biff_sheet_names(&bytes).unwrap().sheets, ["数据"]);
        bytes[14] = 1;
        assert!(biff_sheet_names(&bytes).unwrap().sheets.is_empty());
        assert_eq!(biff_sheet_names(&bytes).unwrap().warnings.len(), 1);
    }
}
