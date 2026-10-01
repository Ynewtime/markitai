//! markitai: embedded parts (`w:altChunk`).
//!
//! A Word document may hold a part in another format where a `w:altChunk`
//! stands, for Word to convert when it opens the document: programs that
//! build documents (mail merges, report generators, the Open XML SDK's
//! "alternative format import") write HTML, RTF, plain text or whole Word
//! documents that way. Upstream ignored the element, so all of that content
//! was lost. Each part is read where it stands, by its content type (from
//! `[Content_Types].xml`, else its extension, else its first bytes):
//!
//! - HTML and XHTML (`text/html`, `application/xhtml+xml`) through the
//!   shared HTML reader, once [`html_as_xml`] has made well-formed XML of it;
//!   its pictures from `data:` URLs, and from the archive when it is
//! - a web archive (MHT: `message/rfc822`, `multipart/related`, as the
//!   html-docx-js library writes every document): its HTML page, its
//!   quoted-printable or base64 parts decoded, pictures found by
//!   `Content-Location` or `cid:`;
//! - RTF (`application/rtf`, `text/rtf`) through the RTF reader;
//! - plain text (`text/plain`), a paragraph per line;
//! - a Word document (the WordprocessingML main-part types, a ZIP package)
//!   through this reader, at most [`MAX_DEPTH`] documents deep, its reads
//!   counting against the outer package's decompression budget.
//!
//! An embedded document's images join the document's assets, its notes
//! follow the document's notes, and its anchors and note ids are scoped to
//! the part so they cannot meet the document's own. A part in any other
//! format (Word's XML formats, `application/xml`), nested too deep, missing
//! or unreadable adds nothing and a warning (see [`Embedded::finish`]); a
//! resource limit is an error, as everywhere.

use super::content::Ctx;
use crate::error::ConvertError;
use crate::model::{
    AnchorId, AssetId, Block, CellSlot, Document, ImageSource, Inline, LinkTarget, Note,
};
use crate::package::xml::{Element, ns, parse_xml};
use crate::shared::html::{HtmlCtx, Stylesheet, to_blocks};
use crate::shared::uri::is_absolute_uri;
use std::collections::BTreeMap;

/// Most Word documents embedded inside one another that are read.
pub(super) const MAX_DEPTH: u32 = 3;

/// What the document's embedded parts add besides their blocks.
#[derive(Debug, Default)]
pub(super) struct Embedded {
    /// The notes of embedded documents, their ids scoped.
    notes: Vec<Note>,
    /// Parts read so far, which scopes their ids.
    read: usize,
    /// Parts left out, by why.
    skipped: BTreeMap<Skipped, usize>,
    /// What embedded documents left out themselves.
    nested: Vec<String>,
}

/// Why an embedded part adds nothing.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Skipped {
    /// Its relationship or part is missing.
    Missing,
    /// Its format is not read (the content type, or a description).
    Format(String),
    /// A Word document more than [`MAX_DEPTH`] deep.
    TooDeep,
    /// Its reader found it unusable.
    Unreadable,
}

impl Embedded {
    /// Move the embedded documents' notes after `notes`, and say what was
    /// left out: one sentence per reason, then what the embedded documents
    /// said themselves.
    pub(super) fn finish(self, notes: &mut Vec<Note>) -> Vec<String> {
        notes.extend(self.notes);
        let mut warnings = Vec::new();
        for (why, count) in self.skipped {
            let (parts, its) = if count == 1 {
                ("1 embedded part".to_string(), "its")
            } else {
                (format!("{count} embedded parts"), "their")
            };
            let why = match why {
                Skipped::Missing => "is missing from the package".to_string(),
                Skipped::Format(format) => {
                    format!("is in a format that is not converted ({format})")
                }
                Skipped::TooDeep => {
                    format!("is a Word document nested more than {MAX_DEPTH} documents deep")
                }
                Skipped::Unreadable => "could not be read".to_string(),
            };
            warnings.push(format!(
                "{parts} of the document (w:altChunk) {}; {its} content is not in the Markdown.",
                if count == 1 { why } else { why.replacen("is ", "are ", 1) }
            ));
        }
        warnings.extend(self.nested);
        warnings
    }
}

/// The blocks of the part a `w:altChunk` embeds, in its place.
pub(super) fn blocks(elem: &Element, ctx: &Ctx) -> Result<Vec<Block>, ConvertError> {
    let target = match elem.attr(ns::R, "id") {
        Some(id) => ctx.rel_part(id)?,
        None => None,
    };
    let Some((part, bytes)) = target else {
        skip(ctx, Skipped::Missing);
        return Ok(Vec::new());
    };
    let scope = {
        let mut embedded = ctx.embedded.borrow_mut();
        embedded.read += 1;
        format!("chunk{}-", embedded.read)
    };
    let content_type = content_type(ctx, &part)?;
    let read = match kind(content_type.as_deref(), &part, &bytes) {
        Kind::Html => html(&bytes, None, None, ctx, &scope),
        Kind::Mht => mht(&bytes, ctx, &scope),
        Kind::Rtf => {
            crate::formats::rtf::parse(&bytes).and_then(|doc| merge(doc, ctx, &part, &scope))
        }
        Kind::Text => Ok(text(&bytes)),
        Kind::Word if ctx.chunk_depth >= MAX_DEPTH => {
            skip(ctx, Skipped::TooDeep);
            return Ok(Vec::new());
        }
        Kind::Word => {
            let spent = ctx.pkg.borrow().total_read();
            super::parse_at(&bytes, ctx.chunk_depth + 1, spent).and_then(|(doc, total)| {
                ctx.pkg.borrow_mut().charge(total)?;
                merge(doc, ctx, &part, &scope)
            })
        }
        Kind::Other(format) => {
            log::debug!("altChunk {part} is {format}, which is not read");
            skip(ctx, Skipped::Format(format));
            return Ok(Vec::new());
        }
    };
    match read {
        Ok(blocks) => Ok(blocks),
        Err(error @ ConvertError::ResourceLimit { .. }) => Err(error),
        Err(error) => {
            log::warn!("skipping unreadable altChunk {part}: {error}");
            skip(ctx, Skipped::Unreadable);
            Ok(Vec::new())
        }
    }
}

fn skip(ctx: &Ctx, why: Skipped) {
    *ctx.embedded.borrow_mut().skipped.entry(why).or_default() += 1;
}

/// The formats an embedded part can be in.
#[derive(Debug, PartialEq, Eq)]
enum Kind {
    Html,
    /// A web archive (MHT): a MIME message holding an HTML page and the
    /// pictures it shows.
    Mht,
    Rtf,
    Text,
    Word,
    /// Anything else, described for the warning.
    Other(String),
}

/// What a part holds, by its declared content type, else its extension,
/// else its first bytes.
fn kind(content_type: Option<&str>, part: &str, bytes: &[u8]) -> Kind {
    let declared = content_type
        .map(|value| value.split(';').next().unwrap_or("").trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());
    let zip = bytes.starts_with(b"PK\x03\x04");
    if let Some(declared) = declared {
        return match declared.as_str() {
            "text/html" | "application/xhtml+xml" => Kind::Html,
            "application/rtf" | "text/rtf" => Kind::Rtf,
            "text/plain" => Kind::Text,
            word if zip
                && (word.starts_with(
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.",
                ) || word.starts_with("application/vnd.ms-word.")) =>
            {
                Kind::Word
            }
            "message/rfc822" | "multipart/related" => Kind::Mht,
            _ => Kind::Other(declared),
        };
    }
    let extension = part.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
    match extension.as_str() {
        "htm" | "html" | "xhtml" => Kind::Html,
        "rtf" => Kind::Rtf,
        "txt" => Kind::Text,
        "docx" | "docm" | "dotx" | "dotm" if zip => Kind::Word,
        "mht" | "mhtml" => Kind::Mht,
        _ if bytes.starts_with(b"{\\rtf") => Kind::Rtf,
        _ if zip => Kind::Word,
        _ => Kind::Other(if extension.is_empty() { "unknown".into() } else { extension }),
    }
}

/// The content type `[Content_Types].xml` gives a part: its `Override`,
/// else the `Default` for its extension.
fn content_type(ctx: &Ctx, part: &str) -> Result<Option<String>, ConvertError> {
    let Some(types) = ctx.pkg.borrow_mut().optional_xml_part("[Content_Types].xml")? else {
        return Ok(None);
    };
    let Some(types) = types.child_elems().find(|e| e.local == "Types") else {
        return Ok(None);
    };
    let name = format!("/{}", part.trim_start_matches('/'));
    let extension = part.rsplit_once('.').map(|(_, e)| e).unwrap_or("");
    let overridden = types.child_elems().find(|e| {
        e.local == "Override"
            && e.attr_any("PartName").is_some_and(|value| value.eq_ignore_ascii_case(&name))
    });
    let default = || {
        types.child_elems().find(|e| {
            e.local == "Default"
                && e.attr_any("Extension")
                    .is_some_and(|value| value.eq_ignore_ascii_case(extension))
        })
    };
    Ok(overridden.or_else(default).and_then(|e| e.attr_any("ContentType")).map(str::to_string))
}

/// A document read from an embedded part as blocks of this one: its images
/// added to the document's assets, its notes to the embedded notes, its
/// anchors and note ids scoped by `scope`, and what it left out to the
/// warnings.
fn merge(doc: Document, ctx: &Ctx, part: &str, scope: &str) -> Result<Vec<Block>, ConvertError> {
    let mut assets = Vec::with_capacity(doc.assets.len());
    for asset in doc.assets {
        let origin = format!("{part}!{}", asset.origin_part);
        assets.push(ctx.add_asset(asset.media_type, origin, &asset.bytes)?);
    }
    let rescope = |inline: &mut Inline| match inline {
        Inline::Image { source, .. } => {
            if let ImageSource::Asset(AssetId(index)) = *source {
                *source = assets
                    .get(index)
                    .map_or(ImageSource::Unavailable, |&id| ImageSource::Asset(id));
            }
        }
        Inline::Anchor(id)
        | Inline::NoteRef(id)
        | Inline::Link { target: LinkTarget::Anchor(id), .. } => {
            *id = format!("{scope}{id}");
        }
        _ => {}
    };
    let mut blocks = doc.blocks;
    each_inline(&mut blocks, scope, &rescope);
    let mut embedded = ctx.embedded.borrow_mut();
    for mut note in doc.notes {
        each_inline(&mut note.blocks, scope, &rescope);
        note.id = format!("{scope}{}", note.id);
        embedded.notes.push(note);
    }
    embedded.nested.extend(doc.warnings);
    Ok(blocks)
}

/// Apply `f` to every inline in `blocks`, nested ones included, and scope
/// each heading's anchor by `scope`.
fn each_inline(blocks: &mut [Block], scope: &str, f: &impl Fn(&mut Inline)) {
    fn inlines(values: &mut [Inline], f: &impl Fn(&mut Inline)) {
        for inline in values {
            f(inline);
            if let Inline::Link { content, .. } = inline {
                inlines(content, f);
            }
        }
    }
    for block in blocks {
        match block {
            Block::Heading { content, anchor, .. } => {
                if let Some(anchor) = anchor {
                    *anchor = format!("{scope}{anchor}");
                }
                inlines(content, f);
            }
            Block::Paragraph(content) => inlines(content, f),
            Block::List(list) => {
                for item in &mut list.items {
                    each_inline(&mut item.blocks, scope, f);
                }
            }
            Block::Table(table) => {
                for slot in table.grid.iter_mut().flatten() {
                    if let CellSlot::Origin(cell) = slot {
                        each_inline(&mut cell.blocks, scope, f);
                    }
                }
            }
            Block::BlockQuote(inner) => each_inline(inner, scope, f),
            Block::CodeBlock { .. } | Block::Rule | Block::Math(_) => {}
        }
    }
}

/// Plain text as a paragraph per non-blank line.
fn text(bytes: &[u8]) -> Vec<Block> {
    decode(bytes, None)
        .lines()
        .map(str::trim_end)
        .filter(|line| !line.trim().is_empty())
        .map(|line| Block::Paragraph(vec![Inline::plain(line)]))
        .collect()
}

/// Text in the encoding its byte-order mark, or `declared`, names; else
/// UTF-8 when it is that, else Windows-1252.
fn decode(bytes: &[u8], declared: Option<&str>) -> String {
    if let Some((encoding, bom)) = encoding_rs::Encoding::for_bom(bytes) {
        return encoding.decode_without_bom_handling(&bytes[bom..]).0.into_owned();
    }
    if let Some(encoding) =
        declared.and_then(|label| encoding_rs::Encoding::for_label(label.as_bytes()))
    {
        return encoding.decode_without_bom_handling(bytes).0.into_owned();
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => encoding_rs::WINDOWS_1252.decode_without_bom_handling(bytes).0.into_owned(),
    }
}

/// The encoding an HTML part declares near its start: a `<meta charset>`,
/// a `content="…; charset=…"` or an XML declaration's `encoding`.
fn declared_charset(bytes: &[u8]) -> Option<String> {
    let head = String::from_utf8_lossy(&bytes[..bytes.len().min(2048)]).to_ascii_lowercase();
    let at = match head.find("charset=") {
        Some(at) => at + "charset=".len(),
        None => {
            let xml = head.find("<?xml")?;
            xml + head[xml..].find("encoding=")? + "encoding=".len()
        }
    };
    let value: String = head[at..]
        .trim_start_matches(['"', '\''])
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '.'))
        .collect();
    (!value.is_empty()).then_some(value)
}

/// An HTML part's blocks: its text in `charset`, else the encoding it
/// declares; its pictures from `resources`, when it came in a web archive.
fn html(
    bytes: &[u8],
    charset: Option<&str>,
    resources: Option<&[Resource]>,
    ctx: &Ctx,
    scope: &str,
) -> Result<Vec<Block>, ConvertError> {
    let declared = charset.map(str::to_string).or_else(|| declared_charset(bytes));
    let text = decode(bytes, declared.as_deref());
    let tree = parse_xml(html_as_xml(&text).as_bytes())?;
    let html = tree.child_elems().find(|e| e.local == "html");
    let body = html.and_then(|html| html.child_elems().find(|e| e.local == "body"));
    let mut css = Stylesheet::default();
    let mut stack: Vec<&Element> = tree.child_elems().collect();
    while let Some(elem) = stack.pop() {
        if elem.local == "style" {
            css.add(&elem.text());
        } else if elem.local != "body" {
            stack.extend(elem.child_elems());
        }
    }
    let hooks = ChunkHtml { ctx, scope, resources: resources.unwrap_or_default() };
    to_blocks(body.or(html).unwrap_or(&tree), &css, &hooks)
}

/// How an embedded HTML part's links, images and anchors resolve: it has
/// no package of its own, so only absolute URLs, its own fragments and
/// `data:` images do.
struct ChunkHtml<'c, 'a, 'b> {
    ctx: &'c Ctx<'a, 'b>,
    scope: &'c str,
    /// The other parts of the web archive the page came in.
    resources: &'c [Resource],
}

impl HtmlCtx for ChunkHtml<'_, '_, '_> {
    fn link_target(&self, href: &str) -> Option<LinkTarget> {
        if href.is_empty() {
            return None;
        }
        if let Some(fragment) = href.strip_prefix('#') {
            return Some(LinkTarget::Anchor(self.anchor_id(fragment)));
        }
        Some(if is_absolute_uri(href) {
            LinkTarget::External(href.to_string())
        } else {
            LinkTarget::Relative(href.to_string())
        })
    }

    fn image_source(&self, src: &str) -> Result<Option<ImageSource>, ConvertError> {
        if let Some(data) = src.strip_prefix("data:") {
            let Some((head, payload)) = data.split_once(',') else {
                return Ok(Some(ImageSource::Unavailable));
            };
            let media = head.split(';').next().unwrap_or("").trim();
            let bytes = if head.ends_with(";base64") { base64(payload) } else { None };
            return match bytes {
                Some(bytes) if media.starts_with("image/") => {
                    let origin = format!("{}data-{}", self.scope, bytes.len());
                    let id = self.ctx.add_asset(media.to_string(), origin, &bytes)?;
                    Ok(Some(ImageSource::Asset(id)))
                }
                _ => Ok(Some(ImageSource::Unavailable)),
            };
        }
        // A picture the web archive holds, by its location or content id.
        let cid = src.strip_prefix("cid:");
        let held = self.resources.iter().find(|resource| {
            resource.location.as_deref() == Some(src)
                || cid.is_some_and(|cid| resource.id.as_deref() == Some(cid))
        });
        if let Some(resource) = held {
            if !resource.media.starts_with("image/") {
                return Ok(Some(ImageSource::Unavailable));
            }
            let origin = format!("{}{src}", self.scope);
            let id = self.ctx.add_asset(resource.media.clone(), origin, &resource.bytes)?;
            return Ok(Some(ImageSource::Asset(id)));
        }
        if src.starts_with("http://") || src.starts_with("https://") {
            return Ok(Some(ImageSource::External(src.to_string())));
        }
        Ok(Some(ImageSource::Unavailable))
    }

    fn anchor_id(&self, raw: &str) -> AnchorId {
        format!("{}{raw}", self.scope)
    }
}

/// A part of a web archive: its media type, `charset` parameter, where the
/// page refers to it (`Content-Location`, `Content-ID`) and its decoded
/// bytes.
#[derive(Debug, Default)]
struct Resource {
    media: String,
    charset: Option<String>,
    location: Option<String>,
    id: Option<String>,
    bytes: Vec<u8>,
}

/// Most multipart bodies inside one another a web archive is read through.
const MAX_MIME_DEPTH: usize = 4;

/// A web archive's blocks: its first HTML part, read with the archive's
/// pictures. One with no HTML part is unreadable.
fn mht(bytes: &[u8], ctx: &Ctx, scope: &str) -> Result<Vec<Block>, ConvertError> {
    let mut parts = Vec::new();
    mime_parts(bytes, 0, &mut parts);
    let Some(page) = parts.iter().position(|part| part.media == "text/html") else {
        return Err(ConvertError::malformed("a web archive with no HTML page"));
    };
    let page = parts.remove(page);
    html(&page.bytes, page.charset.as_deref(), Some(&parts), ctx, scope)
}

/// The leaf parts of a MIME entity, decoded; multipart bodies are followed
/// down to [`MAX_MIME_DEPTH`].
fn mime_parts(entity: &[u8], depth: usize, out: &mut Vec<Resource>) {
    let (headers, body) = split_headers(entity);
    let header = |name: &str| {
        headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    };
    let content_type = header("content-type").unwrap_or("text/plain");
    let media = content_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
    let parameter = |name: &str| {
        content_type.split(';').skip(1).find_map(|param| {
            let (key, value) = param.split_once('=')?;
            key.trim()
                .eq_ignore_ascii_case(name)
                .then(|| value.trim().trim_matches('"').to_string())
        })
    };
    let boundary = parameter("boundary").filter(|_| media.starts_with("multipart/"));
    if let Some(boundary) = boundary.filter(|_| depth < MAX_MIME_DEPTH) {
        let delimiter = format!("--{boundary}");
        let mut rest = body;
        let mut started = false;
        while let Some(at) = find_line(rest, delimiter.as_bytes()) {
            if started {
                mime_parts(trim_line_end(&rest[..at]), depth + 1, out);
            }
            started = true;
            rest = &rest[at + delimiter.len()..];
            if rest.starts_with(b"--") {
                return;
            }
            rest = rest.iter().position(|&b| b == b'\n').map_or(&[][..], |end| &rest[end + 1..]);
        }
        return;
    }
    let encoding = header("content-transfer-encoding").unwrap_or("").trim().to_ascii_lowercase();
    let bytes = match encoding.as_str() {
        "base64" => base64(&String::from_utf8_lossy(body)).unwrap_or_default(),
        "quoted-printable" => quoted_printable(body),
        _ => body.to_vec(),
    };
    out.push(Resource {
        media,
        charset: parameter("charset"),
        location: header("content-location").map(|value| value.trim().to_string()),
        id: header("content-id").map(|value| value.trim().trim_matches(['<', '>']).to_string()),
        bytes,
    });
}

/// A MIME entity's headers (folded lines unfolded) and its body.
fn split_headers(entity: &[u8]) -> (Vec<(String, String)>, &[u8]) {
    let mut headers: Vec<(String, String)> = Vec::new();
    let mut rest = entity;
    while !rest.is_empty() {
        let end = rest.iter().position(|&b| b == b'\n').map_or(rest.len(), |end| end + 1);
        let line = String::from_utf8_lossy(trim_line_end(&rest[..end])).into_owned();
        rest = &rest[end..];
        if line.trim().is_empty() {
            break;
        }
        match headers.last_mut() {
            Some((_, value)) if line.starts_with([' ', '\t']) => {
                value.push(' ');
                value.push_str(line.trim());
            }
            _ => {
                if let Some((name, value)) = line.split_once(':') {
                    headers.push((name.trim().to_string(), value.trim().to_string()));
                }
            }
        }
    }
    (headers, rest)
}

/// Where the first line of `text` that starts with `delimiter` begins.
fn find_line(text: &[u8], delimiter: &[u8]) -> Option<usize> {
    let mut at = 0;
    loop {
        if text[at..].starts_with(delimiter) {
            return Some(at);
        }
        at += text[at..].iter().position(|&b| b == b'\n')? + 1;
    }
}

/// A line without its line break.
fn trim_line_end(line: &[u8]) -> &[u8] {
    let line = line.strip_suffix(b"\n").unwrap_or(line);
    line.strip_suffix(b"\r").unwrap_or(line)
}

/// Quoted-printable bytes decoded: `=XX` is a byte, `=` at a line's end a
/// soft break.
fn quoted_printable(text: &[u8]) -> Vec<u8> {
    let hex = |b: Option<&u8>| b.and_then(|&b| (b as char).to_digit(16));
    let mut out = Vec::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        if text[i] != b'=' {
            out.push(text[i]);
            i += 1;
        } else if text[i + 1..].starts_with(b"\r\n") {
            i += 3;
        } else if text[i + 1..].starts_with(b"\n") {
            i += 2;
        } else if let (Some(high), Some(low)) = (hex(text.get(i + 1)), hex(text.get(i + 2))) {
            out.push((high * 16 + low) as u8);
            i += 3;
        } else {
            out.push(b'=');
            i += 1;
        }
    }
    out
}

/// Standard base64 (whitespace ignored, padding optional); `None` for any
/// other character.
fn base64(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut buffer, mut bits) = (0u32, 0u32);
    for byte in text.bytes() {
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b' ' | b'\t' | b'\r' | b'\n' => continue,
            _ => return None,
        };
        buffer = (buffer << 6) | u32::from(value);
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buffer >> bits) as u8);
            buffer &= (1 << bits) - 1;
        }
    }
    Some(out)
}

/// Most elements [`html_as_xml`] keeps open at once, below the XML
/// reader's nesting limit.
const MAX_HTML_DEPTH: usize = 200;

/// Elements HTML never closes.
const VOID: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
    "track", "wbr",
];

/// Elements a `p` cannot hold: opening one closes an open paragraph.
const BLOCKS: &[&str] = &[
    "address",
    "article",
    "aside",
    "blockquote",
    "center",
    "div",
    "dl",
    "fieldset",
    "figure",
    "footer",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "ul",
];

/// HTML as well-formed XML the XML reader takes: tag and attribute names in
/// lower case, every element closed (void ones, and those HTML closes by
/// itself: a paragraph before a block, an item before the next item, a cell
/// or row before the next), stray end tags dropped, attribute values quoted,
/// a lone `&` or `<` escaped; comments, the doctype, conditional comments,
/// processing instructions, scripts and prefixed Office markup (`o:p`,
/// `v:shape`, whose text stays) left out.
pub(super) fn html_as_xml(html: &str) -> String {
    let mut out = String::with_capacity(html.len() + html.len() / 8);
    let mut open: Vec<String> = Vec::new();
    let mut rest = html;
    while let Some(at) = rest.find('<') {
        escape(&rest[..at], &mut out);
        rest = &rest[at..];
        if let Some(after) = rest.strip_prefix("<!--") {
            rest = after.find("-->").map_or("", |end| &after[end + 3..]);
            continue;
        }
        if rest.starts_with("<!") || rest.starts_with("<?") {
            rest = rest.find('>').map_or("", |end| &rest[end + 1..]);
            continue;
        }
        let closing = rest.starts_with("</");
        let start = if closing { 2 } else { 1 };
        let name_len = rest[start..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | ':' | '.')))
            .unwrap_or(rest.len() - start);
        if !rest[start..].starts_with(|c: char| c.is_ascii_alphabetic()) {
            out.push_str("&lt;");
            rest = &rest[1..];
            continue;
        }
        let name = rest[start..start + name_len].to_ascii_lowercase();
        let tag = &rest[start + name_len..];
        let end = tag_end(tag);
        let attributes = &tag[..end];
        rest = tag.get(end + 1..).unwrap_or("");
        if name.contains(':') {
            continue;
        }
        if closing {
            if let Some(at) = open.iter().rposition(|open| *open == name) {
                for name in open.drain(at..).rev() {
                    close(&name, &mut out);
                }
            }
            continue;
        }
        if matches!(name.as_str(), "script" | "style" | "textarea" | "title") {
            let end = find_ignoring_case(rest, &format!("</{name}")).unwrap_or(rest.len());
            if name == "style" {
                out.push_str("<style>");
                escape(&rest[..end], &mut out);
                out.push_str("</style>");
            } else if name == "title" {
                out.push_str("<title>");
                escape(&rest[..end], &mut out);
                out.push_str("</title>");
            }
            rest = &rest[end..];
            rest = rest.find('>').map_or("", |at| &rest[at + 1..]);
            continue;
        }
        implied_ends(&name, &mut open, &mut out);
        let void = VOID.contains(&name.as_str()) || attributes.trim_end().ends_with('/');
        // Past MAX_HTML_DEPTH open elements a tag is left out (its text
        // stays), which bounds the work per tag and keeps the XML reader's
        // depth limit from failing the document.
        if !void && open.len() >= MAX_HTML_DEPTH {
            continue;
        }
        out.push('<');
        out.push_str(&name);
        write_attributes(attributes, &mut out);
        if void {
            out.push_str("/>");
        } else {
            out.push('>');
            open.push(name);
        }
    }
    escape(rest, &mut out);
    for name in open.into_iter().rev() {
        close(&name, &mut out);
    }
    out
}

fn close(name: &str, out: &mut String) {
    out.push_str("</");
    out.push_str(name);
    out.push('>');
}

/// Close what HTML closes by itself before an element named `name` opens.
fn implied_ends(name: &str, open: &mut Vec<String>, out: &mut String) {
    // The nearest open element of `ends`, unless one of `bounds` opened
    // after it.
    let nearest = |open: &[String], ends: &[&str], bounds: &[&str]| {
        open.iter()
            .rposition(|open| ends.contains(&open.as_str()) || bounds.contains(&open.as_str()))
            .filter(|&at| ends.contains(&open[at].as_str()))
    };
    let at = match name {
        "li" => nearest(open, &["li"], &["ul", "ol", "table", "div"]),
        "dt" | "dd" => nearest(open, &["dt", "dd"], &["dl", "table", "div"]),
        "tr" => nearest(open, &["tr"], &["table"]),
        "td" | "th" => nearest(open, &["td", "th"], &["tr", "table"]),
        "tbody" | "thead" | "tfoot" => nearest(open, &["tbody", "thead", "tfoot"], &["table"]),
        "option" => nearest(open, &["option"], &["select"]),
        _ => None,
    };
    let at = at.or_else(|| {
        BLOCKS
            .contains(&name)
            .then(|| {
                nearest(open, &["p"], &["li", "td", "th", "div", "blockquote", "table", "body"])
            })
            .flatten()
    });
    if let Some(at) = at {
        for name in open.drain(at..).rev() {
            close(&name, out);
        }
    }
}

/// Where a tag's text (after its name) ends: its `>`, outside quotes.
fn tag_end(tag: &str) -> usize {
    let mut quote = None;
    for (at, c) in tag.char_indices() {
        match (quote, c) {
            (None, '"' | '\'') => quote = Some(c),
            (Some(q), c) if c == q => quote = None,
            (None, '>') => return at,
            _ => {}
        }
    }
    tag.len()
}

fn find_ignoring_case(text: &str, needle: &str) -> Option<usize> {
    let needle = needle.as_bytes();
    text.as_bytes().windows(needle.len()).position(|window| window.eq_ignore_ascii_case(needle))
}

/// The attributes of a tag as XML: unprefixed names in lower case, once
/// each, values quoted and escaped.
fn write_attributes(source: &str, out: &mut String) {
    let mut seen: Vec<String> = Vec::new();
    let mut rest = source.trim_start_matches('/');
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == '/');
        let name_len = rest
            .find(|c: char| c.is_whitespace() || matches!(c, '=' | '/' | '>' | '"' | '\''))
            .unwrap_or(rest.len());
        if name_len == 0 {
            return;
        }
        let name = rest[..name_len].to_ascii_lowercase();
        rest = rest[name_len..].trim_start();
        let mut value = "";
        if let Some(after) = rest.strip_prefix('=') {
            let after = after.trim_start();
            match after.chars().next() {
                Some(q @ ('"' | '\'')) => {
                    let end = after[1..].find(q).map_or(after.len(), |end| end + 1);
                    value = &after[1..end];
                    rest = after.get(end + 1..).unwrap_or("");
                }
                _ => {
                    let end = after.find(char::is_whitespace).unwrap_or(after.len());
                    value = &after[..end];
                    rest = &after[end..];
                }
            }
        }
        let valid = name.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
            && name.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        if valid && !seen.contains(&name) {
            let mut escaped = String::with_capacity(value.len());
            escape(value, &mut escaped);
            out.push(' ');
            out.push_str(&name);
            out.push_str("=\"");
            out.push_str(&escaped.replace('"', "&quot;"));
            out.push('"');
            seen.push(name);
        }
    }
}

/// Text as XML character data: a `<` and an `&` that starts no reference
/// escaped.
fn escape(text: &str, out: &mut String) {
    let mut rest = text;
    while let Some(at) = rest.find(['&', '<']) {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        if rest.starts_with('<') {
            out.push_str("&lt;");
        } else if starts_reference(rest) {
            out.push('&');
        } else {
            out.push_str("&amp;");
        }
        rest = &rest[1..];
    }
    out.push_str(rest);
}

/// Whether `text` (starting with `&`) is a character reference: `&name;`,
/// `&#123;` or `&#x1F;`.
fn starts_reference(text: &str) -> bool {
    let body = &text[1..];
    let Some(end) = body.find(';').filter(|&end| end <= 32) else {
        return false;
    };
    let name = &body[..end];
    match name.strip_prefix('#') {
        Some(number) => match number.strip_prefix(['x', 'X']) {
            Some(hex) => !hex.is_empty() && hex.chars().all(|c| c.is_ascii_hexdigit()),
            None => !number.is_empty() && number.chars().all(|c| c.is_ascii_digit()),
        },
        None => {
            name.starts_with(|c: char| c.is_ascii_alphabetic())
                && name.chars().all(|c| c.is_ascii_alphanumeric())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn html_becomes_well_formed_xml() {
        let html = r#"<!DOCTYPE html><HTML><head><meta charset=utf-8><title>T</title>
            <style>p { color: red } a > b {}</style><script>if (a < b) alert(1)</script></head>
            <body><!-- note --><P class=lead>One & two<br>three<p>Four <o:p>&nbsp;</o:p></p>
            <ul><li>a<li>b</ul><table><tr><td>1<td>2<tr><td>3</table>
            <img src="x.png" alt='A "quoted" one'></div><![if !supportLists]>x<![endif]> 1 < 2</body></html>"#;
        let xml = html_as_xml(html);
        assert_eq!(
            xml.split_whitespace().collect::<Vec<_>>().join(" "),
            "<html><head><meta charset=\"utf-8\"/><title>T</title> <style>p { color: red } a > b {}</style>\
             </head> <body><p class=\"lead\">One &amp; two<br/>three</p><p>Four &nbsp;</p> \
             <ul><li>a</li><li>b</li></ul><table><tr><td>1</td><td>2</td></tr><tr><td>3</td></tr></table> \
             <img src=\"x.png\" alt=\"A &quot;quoted&quot; one\"/>x 1 &lt; 2</body></html>"
        );
        assert!(parse_xml(xml.as_bytes()).is_ok());
    }

    #[test]
    fn deep_nesting_keeps_its_text_within_the_xml_depth_limit() {
        let html = format!("{}deep text{}", "<div><span>".repeat(5_000), "<p>more".repeat(5_000));
        let tree = parse_xml(html_as_xml(&html).as_bytes()).unwrap();
        let text = tree.text();
        assert!(text.starts_with("deep textmore"), "{}", &text[..40]);
        assert_eq!(text.matches("more").count(), 5_000);
    }

    #[test]
    fn attribute_values_keep_their_references() {
        let mut out = String::new();
        write_attributes(
            r#" href="a?x=1&amp;y=2&z" checked data-x=1 HREF="dup" x:y="no""#,
            &mut out,
        );
        assert_eq!(out, r#" href="a?x=1&amp;y=2&amp;z" checked="" data-x="1""#);
    }

    #[test]
    fn content_types_and_extensions_name_the_format() {
        let word =
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml";
        assert_eq!(kind(Some("text/html; charset=utf-8"), "a.bin", b""), Kind::Html);
        assert_eq!(kind(Some("application/rtf"), "a.bin", b""), Kind::Rtf);
        assert_eq!(kind(Some(word), "a.docx", b"PK\x03\x04"), Kind::Word);
        assert_eq!(kind(Some(word), "a.docx", b"not a zip"), Kind::Other(word.into()));
        assert_eq!(kind(Some("message/rfc822"), "a.mht", b""), Kind::Mht);
        assert_eq!(kind(None, "a.txt", b"x"), Kind::Text);
        assert_eq!(kind(None, "a.bin", b"{\\rtf1"), Kind::Rtf);
        assert_eq!(kind(None, "a.mht", b""), Kind::Mht);
        assert_eq!(kind(None, "a", b""), Kind::Other("unknown".into()));
    }

    #[test]
    fn text_decodes_by_its_mark_or_declaration() {
        assert_eq!(decode(b"\xef\xbb\xbfcaf\xc3\xa9", None), "café");
        assert_eq!(decode(b"\xff\xfeh\x00i\x00", None), "hi");
        assert_eq!(decode(b"caf\xe9", None), "café");
        assert_eq!(decode("日本".as_bytes(), None), "日本");
        assert_eq!(
            declared_charset(br#"<meta http-equiv=x content="text/html; charset=windows-1251">"#)
                .as_deref(),
            Some("windows-1251")
        );
        assert_eq!(
            declared_charset(br#"<?xml version="1.0" encoding="ISO-8859-1"?><html>"#).as_deref(),
            Some("iso-8859-1")
        );
        assert_eq!(base64("aGk=").as_deref(), Some(&b"hi"[..]));
        assert_eq!(base64("aG\nk"), Some(b"hi".to_vec()));
        assert_eq!(base64("a*"), None);
    }
}
