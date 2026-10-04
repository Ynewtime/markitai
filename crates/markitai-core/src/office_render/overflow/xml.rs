//! Small bounded index for insertion into already validated UTF-8 XML.
use super::{Result, failure};
use quick_xml::{events::Event, name::ResolveResult};
use std::{collections::BTreeMap, time::Instant};

pub(super) const LIMIT: usize = 32 * 1024 * 1024;
// This accounts for retained index payload, not allocator overhead or RSS.
// Repeated inherited namespaces count for every retained copy.
const INDEX_BUDGET: usize = 64 * 1024 * 1024;
const MAX_ATTRIBUTES: usize = 800_000;
struct IndexBudget {
    remaining: usize,
    attributes: usize,
}
impl IndexBudget {
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.remaining = self
            .remaining
            .checked_sub(bytes)
            .ok_or_else(|| failure("workbook XML exceeds cumulative index budget"))?;
        Ok(())
    }
    fn retain(&mut self, text: &str) -> Result<String> {
        self.charge(text.len())?;
        Ok(text.to_owned())
    }
    fn attribute(&mut self) -> Result<()> {
        if self.attributes >= MAX_ATTRIBUTES {
            return Err(failure("workbook XML exceeds cumulative attribute limit"));
        }
        self.charge(
            std::mem::size_of::<((String, String), String)>() + 4 * std::mem::size_of::<usize>(),
        )?;
        self.attributes += 1;
        Ok(())
    }
}
#[derive(Debug)]
pub(super) struct Node {
    pub start: usize,
    pub open_end: usize,
    pub close_start: usize,
    pub end: usize,
    pub depth: usize,
    pub namespace: String,
    pub local: String,
    pub attributes: BTreeMap<(String, String), String>,
    pub empty: bool,
}
impl Node {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attribute("", name)
    }
    pub fn attribute(&self, ns: &str, name: &str) -> Option<&str> {
        self.attributes
            .get(&(ns.into(), name.into()))
            .map(String::as_str)
    }
}
fn namespace(value: ResolveResult<'_>, budget: &mut IndexBudget) -> Result<String> {
    match value {
        ResolveResult::Bound(uri) => budget.retain(
            std::str::from_utf8(uri.as_ref()).map_err(|_| failure("invalid workbook namespace"))?,
        ),
        ResolveResult::Unbound => Ok(String::new()),
        ResolveResult::Unknown(_) => Err(failure("unbound workbook XML namespace")),
    }
}
pub(super) fn index(bytes: &[u8], deadline: Instant) -> Result<Vec<Node>> {
    index_with_budget(bytes, deadline, INDEX_BUDGET)
}
fn index_with_budget(bytes: &[u8], deadline: Instant, limit: usize) -> Result<Vec<Node>> {
    let mut budget = IndexBudget {
        remaining: limit,
        attributes: 0,
    };
    if bytes.len() > LIMIT || std::str::from_utf8(bytes).is_err() {
        return Err(failure("workbook XML must be UTF-8 and at most 32 MiB"));
    }
    let offset = if bytes.starts_with(b"\xef\xbb\xbf") {
        3
    } else {
        0
    };
    let mut reader = quick_xml::NsReader::from_reader(&bytes[offset..]);
    let mut nodes: Vec<Node> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    let mut roots = 0;
    loop {
        super::check_deadline(deadline)?;
        let start = reader.buffer_position() as usize + offset;
        let event = reader
            .read_event()
            .map_err(|_| failure("invalid workbook XML"))?;
        let end = reader.buffer_position() as usize + offset;
        let empty = matches!(event, Event::Empty(_));
        match event {
            Event::Decl(declaration) => {
                if start != offset {
                    return Err(failure("invalid workbook XML declaration position"));
                }
                declaration
                    .xml_version()
                    .map_err(|_| failure("invalid workbook XML declaration"))?;
                // Inspect every pseudo-attribute so duplicate/conflicting encoding
                // declarations cannot hide behind the first value returned by encoding().
                let content = std::str::from_utf8(declaration.as_ref())
                    .map_err(|_| failure("invalid workbook XML declaration"))?;
                let declaration = quick_xml::events::BytesStart::from_content(content, 3);
                for attribute in declaration.attributes() {
                    super::check_deadline(deadline)?;
                    let attribute =
                        attribute.map_err(|_| failure("invalid workbook XML declaration"))?;
                    if attribute.key.as_ref() == b"encoding"
                        && !attribute.value.as_ref().eq_ignore_ascii_case(b"UTF-8")
                    {
                        return Err(failure("unsupported workbook XML encoding: UTF-8 required"));
                    }
                }
            }
            Event::Start(element) | Event::Empty(element) => {
                let start = start
                    + bytes[start..end]
                        .iter()
                        .position(|b| *b == b'<')
                        .ok_or_else(|| failure("invalid XML element offset"))?;
                if nodes.len() >= 200_000 || stack.len() >= 128 {
                    return Err(failure("workbook XML exceeds node or nesting limit"));
                }
                budget.charge(std::mem::size_of::<Node>() + std::mem::size_of::<usize>())?;
                if stack.is_empty() {
                    roots += 1;
                }
                let ns = namespace(
                    reader.resolver().resolve_element(element.name()).0,
                    &mut budget,
                )?;
                let local = budget.retain(
                    std::str::from_utf8(element.local_name().as_ref())
                        .map_err(|_| failure("invalid workbook element"))?,
                )?;
                let mut attributes = BTreeMap::new();
                for attribute in element.attributes() {
                    super::check_deadline(deadline)?;
                    let attribute = attribute.map_err(|_| failure("invalid workbook attribute"))?;
                    budget.attribute()?;
                    let (resolved, local) = reader.resolver().resolve_attribute(attribute.key);
                    let key = (
                        namespace(resolved, &mut budget)?,
                        budget.retain(
                            std::str::from_utf8(local.as_ref())
                                .map_err(|_| failure("invalid workbook attribute"))?,
                        )?,
                    );
                    // UTF-8 validation and rejection of custom entities mean XML
                    // reference decoding/normalization cannot grow this length.
                    // Reserve before decoding, which may itself allocate.
                    let reserved = attribute.value.len();
                    budget.charge(reserved)?;
                    let value = attribute
                        .normalized_value(quick_xml::XmlVersion::Implicit1_0)
                        .map_err(|_| failure("invalid workbook attribute value"))?;
                    if let std::borrow::Cow::Owned(ref owned) = value {
                        budget.charge(owned.capacity().saturating_sub(reserved))?;
                    }
                    let value = value.into_owned();
                    if attributes.insert(key, value).is_some() {
                        return Err(failure("duplicate workbook attribute"));
                    }
                }
                let next = nodes.len();
                nodes.push(Node {
                    start,
                    open_end: end,
                    close_start: end,
                    end,
                    depth: stack.len(),
                    namespace: ns,
                    local,
                    attributes,
                    empty,
                });
                if !empty {
                    stack.push(next);
                }
            }
            Event::End(_) => {
                let node = stack
                    .pop()
                    .ok_or_else(|| failure("unbalanced workbook XML"))?;
                nodes[node].close_start = start;
                nodes[node].end = end;
            }
            Event::Text(text) if stack.is_empty() && !text.iter().all(u8::is_ascii_whitespace) => {
                return Err(failure("text outside workbook XML root"));
            }
            Event::DocType(_) => return Err(failure("workbook document types are not accepted")),
            Event::Eof => break,
            _ => {}
        }
    }
    if roots != 1 || !stack.is_empty() {
        return Err(failure("workbook XML must have one complete root"));
    }
    Ok(nodes)
}
/// Insert children without serializing any existing node or namespace prefix.
pub(super) fn append(bytes: &[u8], node: &Node, children: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(bytes.len().saturating_add(children.len() + 128));
    if node.empty {
        let opening = &bytes[node.start..node.open_end];
        if !opening.ends_with(b"/>") {
            return Err(failure("invalid empty workbook XML element"));
        }
        let name_end = opening[1..]
            .iter()
            .position(|b| b.is_ascii_whitespace() || *b == b'/' || *b == b'>')
            .map(|n| n + 1)
            .ok_or_else(|| failure("invalid XML opening"))?;
        out.extend_from_slice(&bytes[..node.open_end - 2]);
        out.push(b'>');
        out.extend_from_slice(children.as_bytes());
        out.extend_from_slice(b"</");
        out.extend_from_slice(&opening[1..name_end]);
        out.push(b'>');
        out.extend_from_slice(&bytes[node.end..]);
    } else {
        out.extend_from_slice(&bytes[..node.close_start]);
        out.extend_from_slice(children.as_bytes());
        out.extend_from_slice(&bytes[node.close_start..]);
    }
    if out.len() > LIMIT {
        return Err(failure("modified workbook XML exceeds 32 MiB"));
    }
    Ok(out)
}
pub(super) fn insert(bytes: &[u8], position: usize, addition: &str) -> Result<Vec<u8>> {
    if bytes
        .len()
        .checked_add(addition.len())
        .is_none_or(|n| n > LIMIT)
    {
        return Err(failure("modified workbook XML exceeds 32 MiB"));
    }
    let mut out = Vec::with_capacity(bytes.len() + addition.len());
    out.extend_from_slice(&bytes[..position]);
    out.extend_from_slice(addition.as_bytes());
    out.extend_from_slice(&bytes[position..]);
    Ok(out)
}

#[cfg(test)]
mod budget_tests {
    use super::*;
    use std::time::Duration;
    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(10)
    }

    #[test]
    fn inherited_element_namespace_is_charged_per_retained_copy() {
        let uri = format!("urn:test:{}", "x".repeat(4096));
        let input = format!("<root xmlns='{uri}'>{}</root>", "<child/>".repeat(40));
        assert!(input.len() < 8192);
        let error = index_with_budget(input.as_bytes(), deadline(), 32 * 1024).unwrap_err();
        assert!(error.to_string().contains("cumulative index budget"));
        assert_eq!(index(input.as_bytes(), deadline()).unwrap().len(), 41);
    }
    #[test]
    fn inherited_attribute_namespace_is_charged_per_retained_copy() {
        let uri = format!("urn:test:{}", "x".repeat(4096));
        let input = format!(
            "<root xmlns:p='{uri}'>{}</root>",
            "<child p:value='v'/>".repeat(40)
        );
        assert!(input.len() < 8192);
        let error = index_with_budget(input.as_bytes(), deadline(), 32 * 1024).unwrap_err();
        assert!(error.to_string().contains("cumulative index budget"));
        let nodes = index(input.as_bytes(), deadline()).unwrap();
        assert_eq!(nodes[1].attribute(&uri, "value"), Some("v"));
    }
    #[test]
    fn attribute_decode_is_not_attempted_without_budget() {
        // An unknown reference would report an attribute-value error if decoded.
        let input = b"<r a='&unknown;'/>";
        let limit = std::mem::size_of::<Node>()
            + std::mem::size_of::<usize>()
            + 1
            + std::mem::size_of::<((String, String), String)>()
            + 4 * std::mem::size_of::<usize>()
            + 1;

        assert!(
            index_with_budget(input, deadline(), limit)
                .unwrap_err()
                .to_string()
                .contains("cumulative index budget")
        );
        assert!(
            index(input, deadline())
                .unwrap_err()
                .to_string()
                .contains("invalid workbook attribute value")
        );
    }
    #[test]
    fn many_small_attributes_cannot_bypass_cumulative_metadata_budget() {
        let attributes = (0..200).map(|i| format!(" a{i}=''")).collect::<String>();
        let input = format!("<r{attributes}/>");
        assert!(input.len() < 2048);
        assert!(
            index_with_budget(input.as_bytes(), deadline(), 4096)
                .unwrap_err()
                .to_string()
                .contains("cumulative index budget")
        );
        assert_eq!(
            index(input.as_bytes(), deadline()).unwrap()[0]
                .attributes
                .len(),
            200
        );
    }
    #[test]
    fn global_attribute_limit_is_checked_before_reservation() {
        let mut budget = IndexBudget {
            remaining: usize::MAX,
            attributes: MAX_ATTRIBUTES,
        };
        assert!(
            budget
                .attribute()
                .unwrap_err()
                .to_string()
                .contains("cumulative attribute limit")
        );
        assert_eq!(budget.remaining, usize::MAX);
    }
    #[test]
    fn namespace_normalization_chinese_bom_and_append_are_preserved() {
        let input = "\u{feff}<p:r xmlns:p='urn:中文' xmlns:q='urn:属性' q:a='汉字 &amp; &#x1F600;' plain='x\t y'><p:c/></p:r>";
        let nodes = index(input.as_bytes(), deadline()).unwrap();
        assert_eq!(nodes[0].namespace, "urn:中文");
        assert_eq!(nodes[0].attribute("urn:属性", "a"), Some("汉字 & 😀"));
        assert_eq!(nodes[0].attr("plain"), Some("x  y"));
        let modified = append(input.as_bytes(), &nodes[1], "<p:d/>").unwrap();
        let expected = input.replace("<p:c/>", "<p:c><p:d/></p:c>");
        assert_eq!(modified, expected.as_bytes());
        assert_eq!(index(&modified, deadline()).unwrap().len(), 3);
    }
    #[test]
    fn utf8_declarations_with_or_without_bom_preserve_unicode_and_append() {
        for bom in ["", "\u{feff}"] {
            for encoding in ["", " encoding='UTF-8'", " encoding='utf-8'"] {
                let input = format!("{bom}<?xml version='1.0'{encoding}?><r a='中文'/>");
                let nodes = index(input.as_bytes(), deadline()).unwrap();
                assert_eq!(nodes[0].attr("a"), Some("中文"));
                let changed = append(input.as_bytes(), &nodes[0], "<c/>").unwrap();
                assert_eq!(
                    changed,
                    input
                        .replace("<r a='中文'/>", "<r a='中文'><c/></r>")
                        .as_bytes()
                );
                assert_eq!(index(&changed, deadline()).unwrap().len(), 2);
            }
        }
    }
    #[test]
    fn non_utf8_declarations_are_rejected_even_when_bytes_are_utf8() {
        for bom in ["", "\u{feff}"] {
            for encoding in ["ISO-8859-1", "UTF-16", "UTF-16LE", "UTF-32", "windows-1252"] {
                let input =
                    format!("{bom}<?xml version='1.0' encoding='{encoding}'?><r a='中文'/>");
                assert!(
                    index(input.as_bytes(), deadline())
                        .unwrap_err()
                        .to_string()
                        .contains("unsupported workbook XML encoding: UTF-8 required")
                );
            }
        }
    }
    #[test]
    fn malformed_conflicting_or_misplaced_declarations_are_rejected() {
        for input in [
            "<?xml version='1.0' encoding='UTF-8' encoding='ISO-8859-1'?><r/>",
            "<?xml version='1.0' encoding='UTF-8' encoding='utf-8'?><r/>",
            "<?xml version='1.0' encoding=UTF-8?><r/>",
            "<?xml encoding='UTF-8'?><r/>",
            "<!--comment--><?xml version='1.0' encoding='UTF-8'?><r/>",
            "<r><?xml version='1.0' encoding='UTF-8'?></r>",
            "<?xml version='1.0'?><?xml version='1.0'?><r/>",
        ] {
            assert!(index(input.as_bytes(), deadline()).is_err(), "{input}");
        }
        assert!(index(b"\xff\xfe<r/>", deadline()).is_err());
    }
}
