//! Local Outlook compound-message reader. Attachment paths are data, never files to open.

use crate::{Asset, Document, Error, Result};
use std::collections::{BTreeMap, HashSet};
use std::io::{Cursor, Read};
use std::path::{Path, PathBuf};

const PROPERTIES: &str = "__properties_version1.0";
const MAX_INPUT: usize = 256 * 1024 * 1024;
const MAX_TOTAL_READ: u64 = 128 * 1024 * 1024;
const MAX_TEXT: u64 = 16 * 1024 * 1024;
const MAX_ATTACHMENT: u64 = 64 * 1024 * 1024;
const MAX_PROPERTIES: u64 = 1024 * 1024;
const MAX_SUBOBJECTS: usize = 1024;

fn error(message: impl std::fmt::Display) -> Error {
    Error::Conversion(format!("MSG conversion failed: {message}"))
}

#[derive(Default)]
struct Properties {
    values: BTreeMap<u32, [u8; 8]>,
    ambiguous: HashSet<u16>,
}

impl Properties {
    fn parse(bytes: &[u8], header: usize) -> Result<Self> {
        Self::parse_inner(bytes, header, false)
    }

    fn parse_attachment(bytes: &[u8], header: usize) -> Result<Self> {
        Self::parse_inner(bytes, header, true)
    }

    fn parse_inner(bytes: &[u8], header: usize, attachment: bool) -> Result<Self> {
        if bytes.len() < header {
            return Err(error("property stream is shorter than its header"));
        }
        let (entries, remainder) = bytes[header..].as_chunks::<16>();
        let mut values = BTreeMap::new();
        let mut ambiguous = HashSet::new();
        for entry in entries {
            let tag = u32::from_le_bytes(entry[..4].try_into().unwrap());
            let value: [u8; 8] = entry[8..16].try_into().unwrap();
            if let Some(previous) = values.insert(tag, value)
                && previous != value
            {
                let id = (tag >> 16) as u16;
                if attachment && matches!(id, 0x7ffe | 0x3714) {
                    // A damaged classification cannot authorize dropping an
                    // original, but must not discard readable by-value data.
                    ambiguous.insert(id);
                } else {
                    return Err(error(format!("conflicting property {tag:08X}")));
                }
            }
        }
        // Some producers pad the stream past its final complete entry.
        if remainder.iter().any(|byte| *byte != 0) {
            return Err(error("truncated property entry"));
        }
        Ok(Self { values, ambiguous })
    }

    fn integer(&self, id: u16) -> Option<u32> {
        let value = self.values.get(&((u32::from(id) << 16) | 0x0003))?;
        Some(u32::from_le_bytes(value[..4].try_into().unwrap()))
    }

    fn time(&self, id: u16) -> Option<u64> {
        self.values
            .get(&((u32::from(id) << 16) | 0x0040))
            .map(|bytes| u64::from_le_bytes(*bytes))
    }

    fn typed(&self, id: u16, kind: u16) -> Option<&[u8; 8]> {
        let first = u32::from(id) << 16;
        if self.ambiguous.contains(&id) || self.values.range(first..=(first | 0xffff)).count() != 1
        {
            return None;
        }
        self.values.get(&(first | u32::from(kind)))
    }

    fn boolean(&self, id: u16) -> Option<bool> {
        let value = self.typed(id, 0x000b)?;
        match u16::from_le_bytes(value[..2].try_into().unwrap()) {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        }
    }

    fn has_id(&self, id: u16) -> bool {
        let first = u32::from(id) << 16;
        self.values.range(first..=(first | 0xffff)).next().is_some()
    }

    fn inline_candidate(&self) -> (bool, bool) {
        let hidden = self.boolean(0x7ffe);
        let flags = self
            .typed(0x3714, 0x0003)
            .map(|value| u32::from_le_bytes(value[..4].try_into().unwrap()));
        let invalid = (self.has_id(0x7ffe) && hidden.is_none())
            || (self.has_id(0x3714) && flags.is_none())
            || flags.is_some_and(|flags| flags & 0x5 == 0x5);
        // Hidden and rendered-in-HTML are independent properties. Neither a
        // missing property nor HTML-invisible data can imply a body-only image.
        let inline =
            hidden == Some(true) && flags.is_some_and(|flags| flags & 0x4 != 0 && flags & 0x1 == 0);
        (inline, invalid)
    }
}

struct Reader<'a> {
    compound: cfb::CompoundFile<Cursor<&'a [u8]>>,
    total_read: u64,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() > MAX_INPUT {
            return Err(error("input exceeds the 256 MiB limit"));
        }
        let compound = cfb::CompoundFile::open(Cursor::new(bytes)).map_err(error)?;
        if !compound.is_stream(format!("/{PROPERTIES}")) {
            return Err(error("compound file has no top-level message properties"));
        }
        if compound.walk().take(16385).count() > 16384 {
            return Err(error("compound message exceeds the entry limit"));
        }
        Ok(Self {
            compound,
            total_read: 0,
        })
    }

    fn read(&mut self, path: &Path, limit: u64) -> Result<Option<Vec<u8>>> {
        let entry = match self.compound.entry(path) {
            Ok(entry) => entry,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(error(e)),
        };
        if !entry.is_stream() {
            return Err(error(format!("{} is not a stream", path.display())));
        }
        if entry.len() > limit {
            return Err(error(format!(
                "{} exceeds its {limit}-byte limit",
                path.display()
            )));
        }
        let remaining = MAX_TOTAL_READ.saturating_sub(self.total_read);
        if entry.len() > remaining {
            return Err(error("message exceeds the total decoded-stream budget"));
        }
        let mut bytes = Vec::new();
        self.compound
            .open_stream(path)
            .map_err(error)?
            .take(limit.min(remaining) + 1)
            .read_to_end(&mut bytes)
            .map_err(error)?;
        if bytes.len() as u64 > limit.min(remaining) {
            return Err(error("stream exceeds the permitted read size"));
        }
        self.total_read += bytes.len() as u64;
        Ok(Some(bytes))
    }

    fn properties(&mut self, storage: &Path, header: usize) -> Result<Properties> {
        let bytes = self
            .read(&storage.join(PROPERTIES), MAX_PROPERTIES)?
            .ok_or_else(|| error(format!("{} has no properties stream", storage.display())))?;
        Properties::parse(&bytes, header)
    }

    fn attachment_properties(&mut self, storage: &Path) -> Result<Properties> {
        let bytes = self
            .read(&storage.join(PROPERTIES), MAX_PROPERTIES)?
            .ok_or_else(|| error(format!("{} has no properties stream", storage.display())))?;
        Properties::parse_attachment(&bytes, 8)
    }

    fn binary(&mut self, storage: &Path, id: u16, limit: u64) -> Result<Option<Vec<u8>>> {
        self.read(&storage.join(format!("__substg1.0_{id:04X}0102")), limit)
    }

    fn string(
        &mut self,
        storage: &Path,
        id: u16,
        codepage: Option<u32>,
        warnings: &mut Vec<String>,
    ) -> Result<String> {
        if let Some(bytes) =
            self.read(&storage.join(format!("__substg1.0_{id:04X}001F")), MAX_TEXT)?
        {
            let text = unicode(&bytes)?;
            if !text.is_empty() {
                return Ok(text);
            }
        }
        let Some(bytes) =
            self.read(&storage.join(format!("__substg1.0_{id:04X}001E")), MAX_TEXT)?
        else {
            return Ok(String::new());
        };
        decode_ansi(&bytes, codepage, warnings)
    }

    fn objects(&self, prefix: &str) -> Result<Vec<PathBuf>> {
        let mut objects = self
            .compound
            .read_storage("/")
            .map_err(error)?
            .filter(|entry| entry.is_storage() && entry.name().starts_with(prefix))
            .map(|entry| entry.path().to_owned())
            .take(MAX_SUBOBJECTS + 1)
            .collect::<Vec<_>>();
        if objects.len() > MAX_SUBOBJECTS {
            return Err(error(format!(
                "more than {MAX_SUBOBJECTS} message subobjects"
            )));
        }
        objects.sort();
        Ok(objects)
    }
}

fn unicode(bytes: &[u8]) -> Result<String> {
    if !bytes.len().is_multiple_of(2) {
        return Err(error("UTF-16 property has an odd byte length"));
    }
    let units = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect::<Vec<_>>();
    String::from_utf16(&units)
        .map(|text| {
            text.trim_start_matches('\u{feff}')
                .trim_end_matches('\0')
                .trim()
                .to_owned()
        })
        .map_err(|_| error("invalid UTF-16 property"))
}

fn codepage_encoding(codepage: u32) -> Option<&'static encoding_rs::Encoding> {
    let label = match codepage {
        65001 => "utf-8",
        932 => "shift_jis",
        936 => "gbk",
        949 => "euc-kr",
        950 => "big5",
        50220..=50222 => "iso-2022-jp",
        51932 => "euc-jp",
        51949 => "euc-kr",
        54936 => "gb18030",
        874 => "windows-874",
        1250 => "windows-1250",
        1251 => "windows-1251",
        1252 => "windows-1252",
        1253 => "windows-1253",
        1254 => "windows-1254",
        1255 => "windows-1255",
        1256 => "windows-1256",
        1257 => "windows-1257",
        1258 => "windows-1258",
        20866 => "koi8-r",
        21866 => "koi8-u",
        28592 => "iso-8859-2",
        28595 => "iso-8859-5",
        28597 => "iso-8859-7",
        28599 => "iso-8859-9",
        28605 => "iso-8859-15",
        _ => return None,
    };
    encoding_rs::Encoding::for_label(label.as_bytes())
}

fn decode_ansi(bytes: &[u8], codepage: Option<u32>, warnings: &mut Vec<String>) -> Result<String> {
    let end = bytes
        .iter()
        .rposition(|byte| *byte != 0)
        .map_or(0, |last| last + 1);
    let bytes = &bytes[..end];
    if let Some(codepage) = codepage {
        if codepage == 28591 {
            return Ok(bytes
                .iter()
                .map(|&byte| char::from(byte))
                .collect::<String>()
                .trim()
                .to_owned());
        }
        if codepage == 20127 && bytes.is_ascii() {
            return Ok(String::from_utf8_lossy(bytes).trim().to_owned());
        }
        if let Some(encoding) = codepage_encoding(codepage) {
            let (text, malformed) = encoding.decode_without_bom_handling(bytes);
            if !malformed {
                return Ok(text.trim().to_owned());
            }
            warnings.push(format!("MSG code page {codepage} could not decode a text property; UTF-8/Windows-1252 fallback was used."));
        } else {
            warnings.push(format!(
                "MSG code page {codepage} is unsupported; UTF-8/Windows-1252 fallback was used."
            ));
        }
    }
    if let Ok(text) = std::str::from_utf8(bytes) {
        return Ok(text.trim().to_owned());
    }
    // Windows-1252 maps every byte, so this fallback cannot fail.
    let (text, _) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(bytes);
    Ok(text.trim().to_owned())
}

fn recipient_headers(
    reader: &mut Reader<'_>,
    codepage: Option<u32>,
    warnings: &mut Vec<String>,
) -> Result<BTreeMap<u32, Vec<String>>> {
    let mut groups = BTreeMap::<u32, Vec<String>>::new();
    for path in reader.objects("__recip_version1.0_#")? {
        let recipient = (|| {
            let properties = reader.properties(&path, 8)?;
            let kind = properties.integer(0x0c15).unwrap_or(1);
            let mut address = reader.string(&path, 0x39fe, codepage, warnings)?;
            if address.is_empty() {
                address = reader.string(&path, 0x3003, codepage, warnings)?;
            }
            let name = reader.string(&path, 0x3001, codepage, warnings)?;
            let display = if address.is_empty() || name == address {
                name
            } else if name.is_empty() {
                address
            } else {
                format!("{name} <{address}>")
            };
            Ok::<_, Error>((kind, display))
        })();
        match recipient {
            Ok((kind, display)) if !display.is_empty() => {
                groups.entry(kind).or_default().push(display)
            }
            Ok(_) => {}
            Err(e) => warnings.push(format!(
                "MSG recipient {} could not be read: {e}",
                path.display()
            )),
        }
    }
    Ok(groups)
}

struct Attachment {
    asset: Asset,
    label: String,
    cid: String,
    inline_candidate: bool,
}

fn resolve_content_ids(
    html: &str,
    attachments: &[Attachment],
    warnings: &mut Vec<String>,
) -> String {
    static ATTRIBUTES: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let pattern = ATTRIBUTES.get_or_init(|| {
        regex::Regex::new(r#"(?i)(\b(?:src|href)\s*=\s*)(?:"([^"]*)"|'([^']*)'|([^\s>]+))"#)
            .expect("static HTML attribute pattern")
    });
    pattern
        .replace_all(html, |captures: &regex::Captures<'_>| {
            let value = captures
                .get(2)
                .or_else(|| captures.get(3))
                .or_else(|| captures.get(4))
                .unwrap()
                .as_str();
            if !value
                .get(..4)
                .is_some_and(|scheme| scheme.eq_ignore_ascii_case("cid:"))
            {
                return captures[0].to_owned();
            }
            let id = &value[4..];
            if let Some(attachment) = attachments
                .iter()
                .find(|attachment| !attachment.cid.is_empty() && attachment.cid == id)
            {
                format!(
                    "{}\".markitai/assets/{}\"",
                    &captures[1], attachment.asset.name
                )
            } else {
                warnings.push(format!("MSG HTML refers to missing content ID {id:?}."));
                captures[0].to_owned()
            }
        })
        .into_owned()
}

fn safe_name(name: &str, index: usize) -> String {
    format!("msg-{}-{}", index + 1, crate::output_name::attachment(name))
}

fn attachments(
    reader: &mut Reader<'_>,
    codepage: Option<u32>,
    warnings: &mut Vec<String>,
) -> Result<Vec<Attachment>> {
    let mut attachments = Vec::new();
    for (index, path) in reader
        .objects("__attach_version1.0_#")?
        .into_iter()
        .enumerate()
    {
        let result = (|| {
            let properties = reader.attachment_properties(&path)?;
            let method = properties.integer(0x3705).unwrap_or(1);
            if method != 1 {
                return Err(error(format!(
                    "attachment method {method} is not supported (only by-value binary attachments are extracted)"
                )));
            }
            let mut label = reader.string(&path, 0x3707, codepage, warnings)?;
            if label.is_empty() {
                label = reader.string(&path, 0x3704, codepage, warnings)?;
            }
            if label.is_empty() {
                label = format!("Attachment {}", index + 1);
            }
            let cid = reader
                .string(&path, 0x3712, codepage, warnings)?
                .trim_matches(['<', '>'])
                .to_owned();
            let bytes = reader
                .binary(&path, 0x3701, MAX_ATTACHMENT)?
                .ok_or_else(|| error("attachment has no by-value data stream"))?;
            let name = safe_name(&label, index);
            let (inline_candidate, invalid) = properties.inline_candidate();
            if invalid {
                warnings.push(format!(
                    "MSG attachment {} has invalid or conflicting inline classification; its original data was retained as a download.",
                    index + 1
                ));
            }
            Ok::<_, Error>(Attachment {
                asset: Asset { name, bytes },
                label,
                cid,
                inline_candidate,
            })
        })();
        match result {
            Ok(attachment) => attachments.push(attachment),
            Err(e) => warnings.push(format!(
                "MSG attachment {} was not extracted: {e}",
                index + 1
            )),
        }
    }
    Ok(attachments)
}

fn header(value: &str) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('<', "\\<")
        .replace('>', "\\>")
}

fn attachment_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('[', "\\[")
        .replace(']', "\\]")
        .replace(['\r', '\n'], " ")
}

fn filetime(value: u64) -> Option<String> {
    if value == 0 {
        return None;
    }
    let seconds = i64::try_from(value / 10_000_000)
        .ok()?
        .checked_sub(11_644_473_600)?;
    chrono::DateTime::from_timestamp(seconds, ((value % 10_000_000) * 100) as u32)
        .map(|date| date.to_rfc3339_opts(chrono::SecondsFormat::AutoSi, true))
}

/// Extract a compound Outlook message without Outlook, Python or external files.
pub(super) fn extract(bytes: &[u8]) -> Result<Document> {
    extract_with_attachments(bytes).map(|(document, _)| document)
}

pub(super) fn extract_with_attachments(bytes: &[u8]) -> Result<(Document, HashSet<String>)> {
    let mut reader = Reader::new(bytes)?;
    let root = Path::new("/");
    let properties = reader.properties(root, 32)?;
    let codepage = properties.integer(0x3ffd);
    let mut document = Document::default();
    let subject = reader.string(root, 0x0037, codepage, &mut document.warnings)?;
    let mut from = reader.string(root, 0x0c1f, codepage, &mut document.warnings)?;
    for id in [0x5d01, 0x0c1a] {
        if from.is_empty() {
            from = reader.string(root, id, codepage, &mut document.warnings)?;
        }
    }
    let recipients = recipient_headers(&mut reader, codepage, &mut document.warnings)?;
    let mut display = Vec::new();
    for (id, kind, key) in [(0x0e04, 1, "to"), (0x0e03, 2, "cc"), (0x0e02, 3, "bcc")] {
        let mut value = reader.string(root, id, codepage, &mut document.warnings)?;
        if value.is_empty() {
            value = recipients
                .get(&kind)
                .map(|items| items.join("; "))
                .unwrap_or_default();
        }
        if !value.is_empty() {
            document.metadata.insert(key.into(), value.clone().into());
        }
        display.push(value);
    }
    if !subject.is_empty() {
        document
            .metadata
            .insert("title".into(), subject.clone().into());
    }
    if !from.is_empty() {
        document.metadata.insert("from".into(), from.clone().into());
    }
    if let Some(date) = properties
        .time(0x0039)
        .or_else(|| properties.time(0x0e06))
        .and_then(filetime)
    {
        document.metadata.insert("date".into(), date.into());
    }

    let plain = reader.string(root, 0x1000, codepage, &mut document.warnings)?;
    let html = if plain.is_empty() {
        reader
            .binary(root, 0x1013, MAX_TEXT)?
            .map(|bytes| decode_ansi(&bytes, properties.integer(0x3fde), &mut document.warnings))
            .transpose()?
            .unwrap_or_default()
    } else {
        String::new()
    };
    let attachments = attachments(&mut reader, codepage, &mut document.warnings)?;
    let mut shown = HashSet::new();
    let body = if !plain.is_empty() {
        plain
    } else if !html.is_empty() {
        let requested: HashSet<_> = crate::output_profiles::html_image_references(&html)
            .into_iter()
            .filter_map(|target| {
                target
                    .get(..4)
                    .filter(|scheme| scheme.eq_ignore_ascii_case("cid:"))
                    .map(|_| target[4..].to_owned())
            })
            .collect();
        let html = resolve_content_ids(&html, &attachments, &mut document.warnings);
        let body = super::html::fragment(&html)?;
        let images: HashSet<_> = crate::output_profiles::image_references(&body)
            .iter()
            .filter_map(|target| crate::image_enrichment::asset_name(target))
            .collect();
        for attachment in &attachments {
            if !attachment.cid.is_empty()
                && requested.contains(&attachment.cid)
                && images.contains(&attachment.asset.name)
            {
                shown.insert(attachment.asset.name.clone());
            }
        }
        body
    } else {
        let reason = if reader.compound.is_stream(root.join("__substg1.0_10090102")) {
            "its body is only stored as RTF, which this MSG reader does not yet decode"
        } else {
            "it has no plain-text or HTML body"
        };
        document.warnings.push(format!("MSG has no readable body: {reason}; only headers and supported attachments were converted."));
        String::new()
    };
    let headers = [("From", &from), ("To", &display[0]), ("Subject", &subject)]
        .into_iter()
        .filter(|(_, value)| !value.is_empty())
        .map(|(name, value)| format!("**{name}:** {}", header(value)))
        .collect::<Vec<_>>();
    document.markdown = "# Email Message".into();
    if !headers.is_empty() {
        document
            .markdown
            .push_str(&format!("\n\n{}", headers.join("\n")));
    }
    document
        .markdown
        .push_str(&format!("\n\n## Content\n\n{}", body.trim()));
    let mut originals = HashSet::new();
    for attachment in attachments {
        let destination = format!(".markitai/assets/{}", attachment.asset.name);
        if !attachment.inline_candidate || !shown.contains(&attachment.asset.name) {
            originals.insert(attachment.asset.name.clone());
            document.markdown.push_str(&format!(
                "\n\n[{}]({destination})",
                attachment_label(&attachment.label)
            ));
        }
        document.assets.push(attachment.asset);
    }
    document
        .metadata
        .insert("converter".into(), "native-msg".into());
    Ok((document, originals))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn property_stream(header: usize, properties: &[(u32, u64)]) -> Vec<u8> {
        let mut bytes = vec![0; header];
        for &(tag, value) in properties {
            bytes.extend(tag.to_le_bytes());
            bytes.extend(0u32.to_le_bytes());
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn utf16(value: &str) -> Vec<u8> {
        value.encode_utf16().flat_map(u16::to_le_bytes).collect()
    }

    fn message(streams: Vec<(&str, Vec<u8>)>) -> Vec<u8> {
        let mut compound = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
        for (path, bytes) in streams {
            compound
                .create_storage_all(Path::new(path).parent().unwrap())
                .unwrap();
            compound
                .create_stream(path)
                .unwrap()
                .write_all(&bytes)
                .unwrap();
        }
        compound.into_inner().into_inner()
    }

    #[test]
    fn html_only_message_keeps_legacy_layout_and_stores_date_separately() {
        let bytes = message(vec![
            (
                "/__properties_version1.0",
                property_stream(
                    32,
                    &[(0x00390040, 133_485_408_000_000_000), (0x3fde0003, 65001)],
                ),
            ),
            ("/__substg1.0_0037001F", utf16("Subject")),
            ("/__substg1.0_0C1F001F", utf16("sender@example.test")),
            ("/__substg1.0_0E04001F", utf16("recipient@example.test")),
            (
                "/__substg1.0_10130102",
                b"<p>Hello <b>world</b></p>".to_vec(),
            ),
        ]);
        let document = extract(&bytes).unwrap();
        assert_eq!(
            document.markdown,
            "# Email Message\n\n**From:** sender@example.test\n**To:** recipient@example.test\n**Subject:** Subject\n\n## Content\n\nHello **world**"
        );
        assert_eq!(document.metadata["title"], "Subject");
        assert_eq!(document.metadata["date"], "2024-01-01T00:00:00Z");
        assert!(document.warnings.is_empty());
    }

    #[test]
    fn declared_ansi_codepage_and_unicode_body_precedence() {
        let mut streams = vec![
            (
                "/__properties_version1.0",
                property_stream(32, &[(0x3ffd0003, 1251)]),
            ),
            (
                "/__substg1.0_1000001E",
                vec![0xcf, 0xf0, 0xe8, 0xe2, 0xe5, 0xf2, 0],
            ),
        ];
        assert!(
            extract(&message(streams.clone()))
                .unwrap()
                .markdown
                .ends_with("Привет")
        );
        streams.push(("/__substg1.0_1000001F", utf16("正文\0")));
        assert!(
            extract(&message(streams))
                .unwrap()
                .markdown
                .ends_with("正文")
        );
    }

    #[test]
    fn recipient_fallback_and_attachment_bytes_are_preserved() {
        let bytes = message(vec![
            ("/__properties_version1.0", property_stream(32, &[])),
            ("/__substg1.0_1000001F", utf16("Body")),
            (
                "/__recip_version1.0_#00000000/__properties_version1.0",
                property_stream(8, &[(0x0c150003, 1)]),
            ),
            (
                "/__recip_version1.0_#00000000/__substg1.0_3001001F",
                utf16("Reader"),
            ),
            (
                "/__recip_version1.0_#00000000/__substg1.0_39FE001F",
                utf16("reader@example.test"),
            ),
            (
                "/__attach_version1.0_#00000000/__properties_version1.0",
                property_stream(8, &[(0x37050003, 1)]),
            ),
            (
                "/__attach_version1.0_#00000000/__substg1.0_3707001F",
                utf16("../report.txt"),
            ),
            (
                "/__attach_version1.0_#00000000/__substg1.0_37010102",
                vec![0, 1, 255],
            ),
        ]);
        let document = extract(&bytes).unwrap();
        assert!(
            document
                .markdown
                .contains("**To:** Reader \\<reader@example.test\\>")
        );
        assert_eq!(document.assets[0].name, "msg-1-report.txt");
        assert_eq!(document.assets[0].bytes, [0, 1, 255]);
        assert!(
            document
                .markdown
                .contains("(.markitai/assets/msg-1-report.txt)")
        );
    }

    #[test]
    fn malformed_properties_unicode_and_foreign_compounds_fail() {
        assert!(extract(b"not an OLE file").is_err());
        assert!(extract(&message(vec![("/other", vec![1])])).is_err());
        assert!(Properties::parse(&[1; 31], 32).is_err());
        assert!(Properties::parse(&[vec![0; 32], vec![1]].concat(), 32).is_err());
        assert!(unicode(&[1]).is_err());
        assert!(unicode(&[0, 0xd8]).is_err());
        assert!(Properties::parse(&[vec![0; 32], vec![0; 4]].concat(), 32).is_ok());
    }

    #[test]
    fn only_attachment_classification_conflicts_can_retain_readable_payloads() {
        let hidden = property_stream(8, &[(0x7ffe000b, 0), (0x7ffe000b, 1)]);
        assert!(Properties::parse(&hidden, 8).is_err());
        let properties = Properties::parse_attachment(&hidden, 8).unwrap();
        assert_eq!(properties.ambiguous, HashSet::from([0x7ffe]));
        assert_eq!(properties.inline_candidate(), (false, true));
        let flags = property_stream(8, &[(0x7ffe000b, 1), (0x37140003, 0), (0x37140003, 4)]);
        assert!(Properties::parse(&flags, 8).is_err());
        let properties = Properties::parse_attachment(&flags, 8).unwrap();
        assert_eq!(properties.ambiguous, HashSet::from([0x3714]));
        assert_eq!(properties.inline_candidate(), (false, true));
        // Method, other fixed properties and truncated records stay strict.
        assert!(
            Properties::parse_attachment(
                &property_stream(8, &[(0x37050003, 1), (0x37050003, 2)]),
                8,
            )
            .is_err()
        );
        assert!(Properties::parse_attachment(&[vec![0; 8], vec![1]].concat(), 8).is_err());
    }

    #[test]
    fn inline_classification_requires_unambiguous_typed_hidden_and_html_flags() {
        for (values, expected) in [
            (vec![(0x7ffe000b, 1), (0x37140003, 4)], (true, false)),
            (vec![(0x7ffe000b, 1), (0x37140003, 6)], (true, false)),
            (vec![(0x7ffe000b, 0), (0x37140003, 4)], (false, false)),
            (vec![(0x7ffe000b, 1)], (false, false)),
            (vec![(0x37140003, 4)], (false, false)),
            (vec![(0x7ffe000b, 2), (0x37140003, 4)], (false, true)),
            (vec![(0x7ffe0003, 1), (0x37140003, 4)], (false, true)),
            (
                vec![(0x7ffe000b, 1), (0x7ffe0003, 1), (0x37140003, 4)],
                (false, true),
            ),
            (vec![(0x7ffe000b, 1), (0x3714000b, 4)], (false, true)),
            (vec![(0x7ffe000b, 1), (0x37140003, 5)], (false, true)),
        ] {
            let properties = Properties::parse_attachment(&property_stream(8, &values), 8).unwrap();
            assert_eq!(properties.inline_candidate(), expected, "{values:x?}");
        }
    }

    #[test]
    fn rtf_only_and_external_attachments_are_explicit() {
        let bytes = message(vec![
            ("/__properties_version1.0", property_stream(32, &[])),
            ("/__substg1.0_0037001F", utf16("Headers survive")),
            ("/__substg1.0_10090102", b"LZFu".to_vec()),
            (
                "/__attach_version1.0_#00000000/__properties_version1.0",
                property_stream(8, &[(0x37050003, 2)]),
            ),
        ]);
        let document = extract(&bytes).unwrap();
        assert_eq!(document.warnings.len(), 2);
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("RTF"))
        );
        assert!(
            document
                .warnings
                .iter()
                .any(|warning| warning.contains("attachment method 2"))
        );
        assert!(document.assets.is_empty());
        assert!(document.markdown.contains("Headers survive"));
    }

    #[test]
    fn stream_limits_are_checked_before_copying_payloads() {
        let bytes = message(vec![
            ("/__properties_version1.0", property_stream(32, &[])),
            ("/payload", vec![1; 12]),
        ]);
        let mut reader = Reader::new(&bytes).unwrap();
        assert!(reader.read(Path::new("/payload"), 8).is_err());
        assert_eq!(reader.total_read, 0);
    }

    #[test]
    fn content_ids_match_whole_attributes_and_report_missing_parts() {
        let attachments = vec![
            Attachment {
                asset: Asset {
                    name: "first.png".into(),
                    bytes: vec![],
                },
                label: "First".into(),
                cid: "logo".into(),
                inline_candidate: false,
            },
            Attachment {
                asset: Asset {
                    name: "second.png".into(),
                    bytes: vec![],
                },
                label: "Second".into(),
                cid: "logo2".into(),
                inline_candidate: false,
            },
        ];
        let mut warnings = Vec::new();
        let html = resolve_content_ids(
            "<p>cid:logo stays text</p><img src='cid:logo2'><img src=\"CID:logo\"><img src=cid:missing>",
            &attachments,
            &mut warnings,
        );
        assert!(html.contains("<p>cid:logo stays text</p>"));
        assert!(html.contains("src=\".markitai/assets/second.png\""));
        assert!(html.contains("src=\".markitai/assets/first.png\""));
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].contains("missing"));
    }
}
