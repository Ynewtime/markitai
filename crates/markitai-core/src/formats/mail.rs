//! Header text, link labels and HTML Content-ID binding shared by the EML and
//! MSG readers.

use crate::{Result, output_profiles};
use std::collections::HashMap;

/// One header line's value: controls become spaces, whitespace collapses and
/// angle brackets cannot open raw HTML.
pub(super) fn safe_header(value: &str) -> String {
    value
        .chars()
        .map(|ch| if ch.is_control() { ' ' } else { ch })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .replace('<', "\\<")
        .replace('>', "\\>")
}

/// Safe inside `[...]` and `![...]`, as the reference sanitizes alt text. A
/// backslash is escaped first, so a trailing one cannot escape the closing `]`.
pub(super) fn link_text(label: &str) -> String {
    safe_header(&label.replace('\\', "\\\\")).replace(['[', ']', '(', ')'], "_")
}

/// A Content-ID header value without its optional angle brackets, or `None`
/// when it is empty, too long, or holds whitespace, controls or stray brackets.
pub(super) fn header_id(value: &str) -> Option<String> {
    let value = value.trim();
    let value = if let Some(value) = value.strip_prefix('<') {
        value.strip_suffix('>')?
    } else {
        value
    };
    if value.is_empty()
        || value.len() > 1024
        || value
            .chars()
            .any(|ch| ch.is_control() || ch.is_whitespace() || matches!(ch, '<' | '>'))
    {
        return None;
    }
    Some(value.to_owned())
}

/// The Content-ID a `cid:` URI names, percent-decoded once.
pub(super) fn uri_id(value: &str) -> Option<String> {
    if !value
        .get(..4)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("cid:"))
    {
        return None;
    }
    let mut bytes = Vec::with_capacity(value.len() - 4);
    let value = value.as_bytes();
    let mut index = 4;
    while index < value.len() {
        if value[index] == b'%' {
            let first = char::from(*value.get(index + 1)?).to_digit(16)?;
            let second = char::from(*value.get(index + 2)?).to_digit(16)?;
            bytes.push((first * 16 + second) as u8);
            index += 3;
        } else {
            bytes.push(value[index]);
            index += 1;
        }
    }
    header_id(std::str::from_utf8(&bytes).ok()?)
}

/// Content-IDs of a message's candidate parts. Exact spelling wins; an ASCII
/// case-insensitive match is accepted only when it names one part.
#[derive(Default)]
pub(super) struct ContentIds {
    exact: HashMap<String, Vec<usize>>,
    folded: HashMap<String, Vec<usize>>,
}

impl ContentIds {
    /// Record part `index` under its Content-ID header; invalid IDs are ignored.
    pub(super) fn insert(&mut self, header: &str, index: usize) {
        if let Some(cid) = header_id(header) {
            self.folded
                .entry(cid.to_ascii_lowercase())
                .or_default()
                .push(index);
            self.exact.entry(cid).or_default().push(index);
        }
    }

    /// The one part a `cid:` URI names, if exactly one does.
    pub(super) fn lookup(&self, uri: &str) -> Option<usize> {
        let id = uri_id(uri)?;
        let values = self
            .exact
            .get(&id)
            .or_else(|| self.folded.get(&id.to_ascii_lowercase()))?;
        match values.as_slice() {
            [value] => Some(*value),
            _ => None,
        }
    }
}

/// Render an HTML body with its `cid:` image targets bound before sanitizing.
/// `bind` maps a target to an asset name. An unbound target survives the
/// sanitizer through a collision-checked placeholder and is restored as its
/// original URI; `unresolved` receives it, shortened, for the warning.
pub(super) fn html_with_content_ids(
    html: &str,
    mut bind: impl FnMut(&str) -> Option<String>,
    unresolved: impl Fn(&str) -> String,
    warnings: &mut Vec<String>,
) -> Result<String> {
    let mut mapped = HashMap::new();
    let mut placeholders = HashMap::new();
    let references = output_profiles::html_image_references(html);
    let mut nonce = 0usize;
    let prefix = loop {
        let prefix = format!(".markitai-mail-unresolved-{nonce}-");
        if !html.contains(&prefix) && !references.iter().any(|uri| uri.contains(&prefix)) {
            break prefix;
        }
        nonce += 1;
    };
    for target in references {
        if !target
            .get(..4)
            .is_some_and(|scheme| scheme.eq_ignore_ascii_case("cid:"))
        {
            continue;
        }
        if let Some(name) = bind(&target) {
            mapped.insert(target, format!(".markitai/assets/{name}"));
        } else {
            let placeholder = format!("{prefix}{}", placeholders.len());
            warnings.push(unresolved(&target.chars().take(180).collect::<String>()));
            placeholders.insert(placeholder.clone(), target.clone());
            mapped.insert(target, placeholder);
        }
    }
    let html = output_profiles::rewrite_html_image_targets(html, &mapped);
    let markdown = super::html::fragment(&html)?;
    Ok(output_profiles::rewrite_image_uri_targets(
        &markdown,
        &placeholders,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn link_text_escapes_backslashes_brackets_angles_and_controls() {
        assert_eq!(link_text("a\\"), "a\\\\");
        assert_eq!(link_text("[a](b)"), "_a__b_");
        assert_eq!(link_text("a<b>\u{0}\u{1b}\r\n  c"), "a\\<b\\> c");
        assert_eq!(safe_header("x\u{0}y\u{1b}z <q>"), "x y z \\<q\\>");
    }

    #[test]
    fn content_ids_decode_once_and_refuse_ambiguous_case_folding() {
        let mut ids = ContentIds::default();
        ids.insert("<Logo@Example>", 0);
        ids.insert("<Case>", 1);
        ids.insert("<case>", 2);
        ids.insert("<bad id>", 3);
        assert_eq!(ids.lookup("CID:%3Clogo%40example%3E"), Some(0));
        assert_eq!(ids.lookup("cid:Case"), Some(1));
        assert_eq!(ids.lookup("cid:CASE"), None);
        assert_eq!(ids.lookup("cid:bad%20id"), None);
        assert_eq!(ids.lookup("cid:bad%XZ"), None);
        assert_eq!(ids.lookup("logo@example"), None);
        assert_eq!(uri_id("cid:part%252Fid").as_deref(), Some("part%2Fid"));
        assert_eq!(header_id("<part%2Fid>").as_deref(), Some("part%2Fid"));
        assert_eq!(header_id("<unbalanced"), None);
    }
}
