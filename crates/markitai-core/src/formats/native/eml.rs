//! MIME body selection and scoped Content-ID resolution without filesystem I/O.

use crate::formats::mail::{self, ContentIds, header_id, link_text, safe_header};
use crate::{Asset, Document, Error, Result};
use mail_parser::{Message, MessagePart, MimeHeaders, PartType};
use std::collections::{HashMap, HashSet};

const MAX_INPUT: usize = 100 * 1024 * 1024;
const MAX_PARTS: usize = 4096;
const MAX_DEPTH: usize = 64;
const MAX_DECODED: usize = 128 * 1024 * 1024;

fn error(message: &str) -> Error {
    Error::Conversion(format!("EML: {message}"))
}

fn attachment_name(part: &MessagePart<'_>, index: usize) -> String {
    let original = part.attachment_name().unwrap_or("attachment.bin");
    // Filename headers are labels, never source paths. Both mail readers use
    // the same portable leaf policy and preserve readable Unicode.
    let name = crate::output_name::attachment(original);
    format!("email-{index}-{name}")
}

fn attached(part: &MessagePart<'_>) -> bool {
    // RFC 2183 section 2.8 treats an unknown declared disposition as an
    // attachment. Only an absent declaration or literal inline is a body resource.
    part.content_disposition()
        .is_some_and(|value| !value.c_type.eq_ignore_ascii_case("inline"))
}

fn validate(message: &Message<'_>) -> Result<Vec<Option<usize>>> {
    let mut total_parts = 0usize;
    let mut total_bytes = 0usize;
    let mut messages = vec![(message, 0)];
    while let Some((message, depth)) = messages.pop() {
        if depth > MAX_DEPTH {
            return Err(error("nested message depth exceeds 64"));
        }
        total_parts = total_parts
            .checked_add(message.parts.len())
            .ok_or_else(|| error("part count overflow"))?;
        if total_parts > MAX_PARTS {
            return Err(error("MIME part count exceeds 4096"));
        }
        for part in &message.parts {
            if let PartType::Message(nested) = &part.body {
                messages.push((nested, depth + 1));
            } else {
                total_bytes = total_bytes
                    .checked_add(part.contents().len())
                    .ok_or_else(|| error("decoded content length overflow"))?;
            }
        }
        if total_bytes > MAX_DECODED {
            return Err(error("decoded MIME content exceeds 128 MiB"));
        }
    }
    let mut parents = vec![None; message.parts.len()];
    let mut seen = HashSet::new();
    let mut pending = vec![(0usize, 0usize)];
    while let Some((id, depth)) = pending.pop() {
        if depth > MAX_DEPTH {
            return Err(error("multipart nesting exceeds 64"));
        }
        if !seen.insert(id) {
            return Err(error("MIME tree repeats a part"));
        }
        let part = message
            .parts
            .get(id)
            .ok_or_else(|| error("MIME tree references a missing part"))?;
        if let PartType::Multipart(children) = &part.body {
            for &child in children.iter().rev() {
                let child = child as usize;
                *parents
                    .get_mut(child)
                    .ok_or_else(|| error("MIME child index is invalid"))? = Some(id);
                pending.push((child, depth + 1));
            }
        }
    }
    Ok(parents)
}

fn related_root(
    message: &Message<'_>,
    part: &MessagePart<'_>,
    children: &[u32],
    warnings: &mut Vec<String>,
) -> Option<u32> {
    let first = children.first().copied()?;
    let Some(start) = part.content_type().and_then(|ct| ct.attribute("start")) else {
        return Some(first);
    };
    let matches: Vec<_> = header_id(start)
        .into_iter()
        .flat_map(|start| {
            children.iter().copied().filter(move |&id| {
                message
                    .parts
                    .get(id as usize)
                    .and_then(MimeHeaders::content_id)
                    .and_then(header_id)
                    .as_deref()
                    == Some(start.as_str())
            })
        })
        .collect();
    if matches.len() == 1 {
        Some(matches[0])
    } else {
        warnings.push("EML multipart/related start does not identify exactly one root; the first member was used.".into());
        Some(first)
    }
}

fn body_part(message: &Message<'_>, warnings: &mut Vec<String>) -> Option<usize> {
    let mut plain = None;
    let mut pending = vec![0usize];
    while let Some(id) = pending.pop() {
        let part = message.parts.get(id)?;
        if attached(part) {
            continue;
        }
        match &part.body {
            PartType::Html(_) => return Some(id),
            PartType::Text(_)
                if part.content_type().is_none() || part.is_content_type("text", "plain") =>
            {
                plain.get_or_insert(id);
            }
            PartType::Multipart(children) => {
                if part.is_content_type("multipart", "related") {
                    if let Some(root) = related_root(message, part, children, warnings) {
                        pending.push(root as usize);
                    }
                } else {
                    pending.extend(children.iter().rev().map(|id| *id as usize));
                }
            }
            // An attached RFC 822 message cannot become the outer message body.
            _ => (),
        }
    }
    plain
}

fn related_scope(message: &Message<'_>, parents: &[Option<usize>], mut id: usize) -> usize {
    loop {
        if message.parts[id].is_content_type("multipart", "related") {
            return id;
        }
        match parents[id] {
            Some(parent) => id = parent,
            None => return 0,
        }
    }
}

/// Content-IDs of the parts in the selected body's related scope.
fn content_ids(message: &Message<'_>, parents: &[Option<usize>], body: usize) -> ContentIds {
    let scope = related_scope(message, parents, body);
    let mut ids = ContentIds::default();
    for (id, part) in message.parts.iter().enumerate() {
        if related_scope(message, parents, id) == scope
            && let Some(cid) = part.content_id()
        {
            ids.insert(cid, id);
        }
    }
    ids
}

fn image_part(part: &MessagePart<'_>) -> bool {
    part.content_type()
        .is_some_and(|ct| ct.c_type.eq_ignore_ascii_case("image"))
        && matches!(part.body, PartType::Binary(_) | PartType::InlineBinary(_))
        && !part.is_encoding_problem
        && !part.contents().is_empty()
}

fn html_body(
    html: &str,
    message: &Message<'_>,
    body: usize,
    parents: &[Option<usize>],
    names: &mut HashMap<usize, String>,
    assets: &mut Vec<Asset>,
    warnings: &mut Vec<String>,
) -> Result<(String, HashSet<usize>)> {
    let ids = content_ids(message, parents, body);
    // The image parts the body shows, which the attachment listing omits.
    let mut bound = HashSet::new();
    let markdown = mail::html_with_content_ids(
        html,
        |target| {
            let id = ids
                .lookup(target)
                .filter(|id| image_part(&message.parts[*id]))?;
            bound.insert(id);
            Some(
                names
                    .entry(id)
                    .or_insert_with(|| {
                        let name = attachment_name(&message.parts[id], assets.len() + 1);
                        assets.push(Asset {
                            name: name.clone(),
                            bytes: message.parts[id].contents().to_vec(),
                        });
                        name
                    })
                    .clone(),
            )
        },
        |target| {
            format!(
                "EML image reference {target:?} has no unambiguous image Content-ID with valid transfer encoding in its MIME related scope; the reference was retained."
            )
        },
        warnings,
    )?;
    Ok((markdown, bound))
}

/// The reference's header block, `## Content` and body; an empty body leaves
/// the bare heading.
fn message_head(message: &Message<'_>, body: &str) -> String {
    let mut headers = Vec::new();
    for name in ["From", "To", "Cc", "Date", "Subject"] {
        for value in message.header_as(name, mail_parser::HeaderForm::Text) {
            if let Some(value) = value.as_text() {
                let date = (name == "Date")
                    .then(|| message.date())
                    .flatten()
                    .map(|date| {
                        date.to_rfc822().replacen(
                            &format!(", {} ", date.day),
                            &format!(", {:02} ", date.day),
                            1,
                        )
                    });
                let value = safe_header(date.as_deref().unwrap_or(value));
                if !value.is_empty() {
                    headers.push(format!("**{name}:** {value}"));
                }
            }
        }
    }
    let mut markdown = "# Email Message".to_owned();
    if !headers.is_empty() {
        markdown.push_str(&format!("\n\n{}", headers.join("\n")));
    }
    let body = body.trim();
    if body.is_empty() {
        markdown.push_str("\n\n## Content");
    } else {
        markdown.push_str(&format!("\n\n## Content\n\n{body}"));
    }
    markdown
}

/// The reference's attachment label: its filename, else `attachment_N`
/// counted from zero, with controls collapsed.
fn attachment_label(part: &MessagePart<'_>, position: usize) -> String {
    let label = part
        .attachment_name()
        .map(|name| {
            name.chars()
                .map(|ch| if ch.is_control() { ' ' } else { ch })
                .collect::<String>()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|name| !name.is_empty());
    label.unwrap_or_else(|| format!("attachment_{position}"))
}

/// The reference's human-readable attachment size.
fn size(bytes: usize) -> String {
    if bytes >= 1024 * 1024 {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

fn quote(markdown: &str) -> String {
    markdown
        .lines()
        .map(|line| {
            if line.trim().is_empty() {
                ">".to_owned()
            } else {
                format!("> {line}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// One nested message at the reference's depth limit: headers, body without
/// Content-ID binding, and attachment names only. It owns no assets; the
/// original message stays downloadable.
fn nested_message(message: &Message<'_>) -> Result<String> {
    let body_id = body_part(message, &mut Vec::new());
    let body = match body_id
        .and_then(|id| message.parts.get(id))
        .map(|part| &part.body)
    {
        Some(PartType::Html(html)) => crate::formats::html::fragment(html)?,
        Some(PartType::Text(text)) => text.to_string(),
        _ => String::new(),
    };
    let mut markdown = message_head(message, &body);
    let names: Vec<String> = message
        .attachments
        .iter()
        .map(|&id| id as usize)
        .filter(|id| Some(*id) != body_id)
        .enumerate()
        .filter_map(|(position, id)| {
            message
                .parts
                .get(id)
                .map(|part| attachment_label(part, position))
        })
        .map(|label| format!("- {}", safe_header(&label)))
        .collect();
    if !names.is_empty() {
        markdown.push_str("\n\n## Attachments\n\n");
        markdown.push_str(&names.join("\n"));
    }
    Ok(markdown)
}

pub(super) fn extract(bytes: &[u8]) -> Result<Document> {
    extract_with_attachments(bytes).map(|(document, _)| document)
}

/// Original downloads are identified by MIME/body provenance, before Markdown
/// can contain literal or author-supplied references to an attachment path.
pub(super) fn extract_with_attachments(bytes: &[u8]) -> Result<(Document, HashSet<String>)> {
    if bytes.len() > MAX_INPUT {
        return Err(error("input exceeds 100 MiB"));
    }
    let message = mail_parser::MessageParser::default()
        .parse(bytes)
        .ok_or_else(|| error("malformed message"))?;
    let parents = validate(&message)?;
    let mut warnings = Vec::new();
    let selected = body_part(&message, &mut warnings);
    let mut assets = Vec::new();
    let mut names = HashMap::new();
    let mut listed = Vec::new();
    // Image parts the body shows through their Content-ID.
    let mut bound = HashSet::new();
    let mut attachment_ids: Vec<_> = message.attachments.iter().map(|&id| id as usize).collect();
    // The parser's body lists only classify literal `attachment`. Include
    // declared non-inline text that it could otherwise omit from attachments.
    // Part indices keep MIME order; multipart containers have no leaf payload.
    attachment_ids.extend(message.parts.iter().enumerate().filter_map(|(id, part)| {
        (attached(part) && matches!(&part.body, PartType::Html(_) | PartType::Text(_)))
            .then_some(id)
    }));
    attachment_ids.sort_unstable();
    attachment_ids.dedup();
    for id in attachment_ids {
        if Some(id) == selected {
            continue;
        }
        let part = message
            .parts
            .get(id)
            .ok_or_else(|| error("attachment index is invalid"))?;
        let index = assets.len() + 1;
        let name = attachment_name(part, index);
        names.insert(id, name.clone());
        assets.push(Asset {
            name: name.clone(),
            bytes: part.contents().to_vec(),
        });
        listed.push((id, name));
        if part.is_encoding_problem {
            warnings.push(format!("EML attachment {index} has malformed transfer encoding; its recovered bytes were retained without CID image binding."));
        }
    }
    let body = match selected.and_then(|id| message.parts.get(id).map(|part| (id, part))) {
        Some((id, part)) => {
            if part.is_encoding_problem {
                warnings.push(
                    "EML body has malformed transfer encoding; parser-recovered text was retained."
                        .into(),
                );
            }
            if let Some(charset) = part.content_type().and_then(|ct| ct.attribute("charset"))
                && encoding_rs::Encoding::for_label(charset.as_bytes()).is_none()
            {
                warnings.push("EML body declares an unknown charset; the MIME parser's UTF-8 fallback was used.".into());
            }
            match &part.body {
                PartType::Html(html) => {
                    let (markdown, shown) = html_body(
                        html,
                        &message,
                        id,
                        &parents,
                        &mut names,
                        &mut assets,
                        &mut warnings,
                    )?;
                    bound = shown;
                    markdown
                }
                PartType::Text(text) => text.to_string(),
                _ => String::new(),
            }
        }
        None => String::new(),
    };
    let mut markdown = message_head(&message, &body);
    let mut listing = Vec::new();
    let mut sections = Vec::new();
    let mut originals = HashSet::new();
    for (position, (id, name)) in listed.iter().enumerate() {
        let part = &message.parts[*id];
        // A displayed CID is only a body resource when it is not explicitly
        // an attachment. Explicit attachments still need their original download.
        if bound.contains(id) && !attached(part) {
            continue;
        }
        originals.insert(name.clone());
        let label = attachment_label(part, position);
        let target = format!(".markitai/assets/{name}");
        if let PartType::Message(nested) = &part.body {
            // As in the reference, one nested level is quoted; the original
            // message also stays downloadable below.
            sections.push(format!(
                "### Attached message: {}\n\n{}",
                safe_header(&label),
                quote(&nested_message(nested)?)
            ));
        }
        // Download originals, including explicitly attached CID images and
        // parser-recovered malformed transfer payloads, independently of previews.
        let mut item = format!(
            "- [{}]({target}) ({})",
            link_text(&label),
            size(part.contents().len())
        );
        // Valid image MIME attachments remain visible to normal image analysis,
        // even without a body CID binding. The original download stays separate;
        // the mail preparation route copies only this image use to a preview.
        if !bound.contains(id) && image_part(part) {
            item.push_str(&format!("\n  ![{}]({target})", link_text(&label)));
        }
        listing.push(item);
    }
    if !listing.is_empty() {
        sections.insert(0, listing.join("\n\n"));
    }
    if !sections.is_empty() {
        markdown.push_str("\n\n## Attachments\n\n");
        markdown.push_str(&sections.join("\n\n"));
    }
    let mut metadata = serde_json::Map::new();
    let title = message.subject().unwrap_or("");
    if !title.is_empty() {
        metadata.insert("title".into(), title.into());
    }
    if let Some(date) = message.date() {
        metadata.insert("date".into(), date.to_rfc3339().into());
    }
    Ok((
        Document {
            markdown,
            metadata,
            assets,
            warnings,
        },
        originals,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output_profiles;
    use base64::Engine;

    fn image(color: [u8; 3]) -> Vec<u8> {
        let mut bytes = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(image::RgbImage::from_pixel(80, 80, image::Rgb(color)))
            .write_to(&mut bytes, image::ImageFormat::Png)
            .unwrap();
        bytes.into_inner()
    }
    fn part(headers: &str, body: &[u8]) -> String {
        format!(
            "{headers}\r\nContent-Transfer-Encoding: base64\r\n\r\n{}\r\n",
            base64::engine::general_purpose::STANDARD.encode(body)
        )
    }
    fn multipart(kind: &str, boundary: &str, parts: &[String]) -> String {
        let mut value = format!("Content-Type: multipart/{kind}; boundary=\"{boundary}\"\r\n\r\n");
        for part in parts {
            value.push_str(&format!("--{boundary}\r\n{part}"));
        }
        value.push_str(&format!("--{boundary}--\r\n"));
        value
    }
    fn message(body: String) -> Vec<u8> {
        format!("MIME-Version: 1.0\r\nSubject: Fixture\r\n{body}").into_bytes()
    }
    fn refs(doc: &Document) -> Vec<String> {
        output_profiles::image_references(&doc.markdown)
    }

    #[test]
    fn plain_body_and_ordinary_attachment_use_the_reference_listing_with_a_download_link() {
        let bytes = message(multipart(
            "mixed",
            "outer",
            &[
                part(
                    "Content-Type: text/plain; charset=utf-8",
                    b"First line\nSecond line",
                ),
                part(
                    "Content-Type: application/octet-stream\r\nContent-Disposition: attachment; filename=notes.bin",
                    b"original attachment",
                ),
            ],
        ));
        let doc = extract(&bytes).unwrap();
        assert_eq!(
            doc.markdown,
            "# Email Message\n\n**Subject:** Fixture\n\n## Content\n\nFirst line\nSecond line\n\n## Attachments\n\n- [notes.bin](.markitai/assets/email-1-notes.bin) (19 B)"
        );
        assert_eq!(doc.assets.len(), 1);
        assert_eq!(doc.assets[0].bytes, b"original attachment");
        assert!(refs(&doc).is_empty());
    }

    #[test]
    fn attachment_section_follows_the_reference_labels_sizes_and_nesting_limit() {
        let inner = multipart(
            "mixed",
            "inner",
            &[
                part(
                    "Content-Type: text/plain",
                    b"Inner body\n\nsecond paragraph",
                ),
                part(
                    "Content-Type: image/png\r\nContent-Disposition: attachment; filename=deep.png",
                    &image([7, 8, 9]),
                ),
            ],
        );
        let nested = format!(
            "Content-Type: message/rfc822\r\nContent-Disposition: attachment; filename=\"fw (1).eml\"\r\n\r\nSubject: Inner\r\n{inner}"
        );
        let doc = extract(&message(multipart(
            "mixed",
            "outer",
            &[
                part("Content-Type: text/plain", b"  "),
                part(
                    "Content-Type: image/png\r\nContent-Disposition: attachment; filename=\"chart [v2] (final).png\"",
                    &image([1, 1, 1]),
                ),
                part("Content-Type: application/octet-stream", &vec![0; 1536]),
                part(
                    "Content-Type: application/pdf\r\nContent-Disposition: attachment; filename=\"a<b>.pdf\"",
                    &vec![0; 3 * 1024 * 1024 / 2],
                ),
                nested,
            ],
        )))
        .unwrap();
        let (head, attachments) = doc.markdown.split_once("\n\n## Attachments\n\n").unwrap();
        assert!(head.ends_with("\n\n## Content"), "{head}");
        let listing: Vec<_> = attachments.split("\n\n").take(4).collect();
        assert!(listing[0].starts_with(
            "- [chart _v2_ _final_.png](.markitai/assets/email-1-chart__v2___final_.png) ("
        ));
        assert_eq!(
            listing[1],
            "- [attachment_1](.markitai/assets/email-2-attachment.bin) (1.5 KB)"
        );
        assert_eq!(
            listing[2],
            "- [a\\<b\\>.pdf](.markitai/assets/email-3-a_b_.pdf) (1.5 MB)"
        );
        assert!(listing[3].starts_with("- [fw _1_.eml](.markitai/assets/email-4-fw__1_.eml) ("));
        assert!(attachments.ends_with(
            "### Attached message: fw (1).eml\n\n> # Email Message\n>\n> **Subject:** Inner\n>\n> ## Content\n>\n> Inner body\n>\n> second paragraph\n>\n> ## Attachments\n>\n> - deep.png"
        ), "{attachments}");
        // Only the outer message's attachments become assets; the nested
        // image stays inside its downloadable message.
        assert_eq!(doc.assets.len(), 4);
    }

    #[test]
    fn html_is_preferred_to_plain_but_never_taken_from_attached_messages_or_html_attachments() {
        let nested = "Content-Type: message/rfc822\r\nContent-Disposition: attachment; filename=forward.eml\r\n\r\nSubject: Nested\r\nContent-Type: text/html\r\n\r\n<p>Not outer body</p>\r\n".to_owned();
        let bytes = message(multipart(
            "mixed",
            "outer",
            &[
                nested,
                part(
                    "Content-Type: text/html\r\nContent-Disposition: attachment; filename=example.html",
                    b"<p>Not body attachment</p>",
                ),
                multipart(
                    "alternative",
                    "choice",
                    &[
                        part("Content-Type: text/plain", b"Plain fallback"),
                        part(
                            "Content-Type: text/html",
                            b"<h2>Chosen</h2><p>HTML <b>body</b>.</p>",
                        ),
                    ],
                ),
            ],
        ));
        let doc = extract(&bytes).unwrap();
        assert!(
            doc.markdown.contains("## Chosen\n\nHTML **body**."),
            "{}",
            doc.markdown
        );
        let (content, attachments) = doc.markdown.split_once("\n## Attachments\n").unwrap();
        for absent in ["Not outer body", "Not body attachment", "Plain fallback"] {
            assert!(!content.contains(absent));
        }
        // The attached message is quoted one level, as in the reference, and
        // an HTML attachment is only listed.
        assert!(
            attachments.contains("### Attached message: forward.eml\n\n> # Email Message\n>\n> **Subject:** Nested\n>\n> ## Content\n>\n> Not outer body"),
            "{attachments}"
        );
        assert!(!attachments.contains("Not body attachment"));
        assert!(
            attachments.contains("- [example.html](.markitai/assets/email-2-example.html) (26 B)")
        );
        assert!(attachments.contains("- [forward.eml](.markitai/assets/email-1-forward.eml) ("));
        assert!(
            doc.assets
                .iter()
                .any(|asset| asset.name.ends_with("forward.eml"))
        );
        assert!(
            doc.assets
                .iter()
                .any(|asset| asset.name.ends_with("example.html"))
        );
    }

    #[test]
    fn related_start_selects_root_and_cid_decodes_scheme_entities_angles_and_percent_once() {
        let pixels = image([20, 90, 170]);
        let bytes = message(multipart("related; start=\"<chosen>\"", "outer", &[
            part("Content-Type: text/html\r\nContent-ID: <unused>", b"<p>Wrong root</p>"),
            part("Content-Type: image/png\r\ncOnTent-Id: <Logo@Example>\r\nContent-Disposition: inline; filename=chart.png", &pixels),
            part("Content-Type: text/html\r\nContent-ID: <chosen>", b"<p>Right root</p><img alt='Figure' src='CID:%3CLogo%40Example%3E'><img alt='Second' src='cid:&lt;logo@example&gt;'>"),
        ]));
        let doc = extract(&bytes).unwrap();
        assert!(doc.markdown.contains("Right root"));
        assert!(!doc.markdown.contains("Wrong root"));
        let target = refs(&doc);
        assert_eq!(target.len(), 1, "{doc:#?}");
        let name = target[0].strip_prefix(".markitai/assets/").unwrap();
        assert_eq!(
            doc.assets
                .iter()
                .find(|asset| asset.name == name)
                .unwrap()
                .bytes,
            pixels
        );
        assert!(doc.markdown.contains("![Figure]"));
        assert!(doc.markdown.contains("![Second]"));
        assert!(
            !doc.warnings
                .iter()
                .any(|warning| warning.contains("Content-ID"))
        );
    }

    #[test]
    fn unknown_dispositions_cannot_select_a_text_or_html_attachment_as_the_body() {
        let bytes = message(multipart(
            "mixed",
            "unknown-disposition",
            &[
                part(
                    "Content-Type: text/html\r\nContent-Disposition: x-project-download; filename=example.html",
                    b"<p>Attached HTML only.</p>",
                ),
                part(
                    "Content-Type: text/plain\r\nContent-Disposition: X-PROJECT-DOWNLOAD; filename=notes.txt",
                    b"Attached plain text only.",
                ),
                part("Content-Type: text/plain", b"The actual message body."),
            ],
        ));
        let (doc, originals) = extract_with_attachments(&bytes).unwrap();
        let (body, listing) = doc.markdown.split_once("\n\n## Attachments\n\n").unwrap();
        assert!(body.contains("The actual message body."));
        assert!(!body.contains("Attached HTML only."));
        assert!(!body.contains("Attached plain text only."));
        assert!(listing.contains("[example.html]"));
        assert!(listing.contains("[notes.txt]"));
        assert_eq!(originals.len(), 2);
        assert_eq!(doc.assets.len(), 2);
        assert_eq!(doc.assets[0].bytes, b"<p>Attached HTML only.</p>");
        assert_eq!(doc.assets[1].bytes, b"Attached plain text only.");
        // Text assets use the parser's charset-decoded UTF-8 buffer, unlike
        // binary attachments' byte-exact transfer-decoded data.
        let latin1 = message(multipart(
            "mixed",
            "text-encoding",
            &[
                part(
                    "Content-Type: text/plain; charset=iso-8859-1\r\nContent-Disposition: x-project-download; filename=latin1.txt",
                    b"caf\xe9",
                ),
                part("Content-Type: text/plain", b"The actual message body."),
            ],
        ));
        let (doc, originals) = extract_with_attachments(&latin1).unwrap();
        assert!(doc.markdown.contains("The actual message body."));
        assert_eq!(originals.len(), 1);
        assert_eq!(doc.assets.len(), 1);
        assert_eq!(doc.assets[0].bytes, "café".as_bytes());
    }

    #[test]
    fn an_explicit_attachment_keeps_a_download_when_the_body_shows_its_content_id() {
        let pixels = image([20, 90, 170]);
        let orphan = image([200, 30, 30]);
        let bytes = message(multipart(
            "mixed",
            "outer",
            &[
                multipart(
                    "related",
                    "inner",
                    &[
                        part(
                            "Content-Type: text/html",
                            b"<p>Body.</p><img src='cid:img1' alt='Logo'><img src='cid:img1' alt='Again'>",
                        ),
                        part(
                            "Content-Type: image/png\r\nContent-ID: <img1>\r\nContent-Disposition: attachment; filename=inline.png",
                            &pixels,
                        ),
                    ],
                ),
                part(
                    "Content-Type: image/png\r\nContent-Disposition: attachment; filename=unreferenced.png",
                    &orphan,
                ),
                part(
                    "Content-Type: text/csv\r\nContent-Disposition: attachment; filename=data.csv",
                    b"a,b\n1,2\n",
                ),
            ],
        ));
        let (doc, originals) = extract_with_attachments(&bytes).unwrap();
        let (content, attachments) = doc.markdown.split_once("\n\n## Attachments\n\n").unwrap();
        assert_eq!(content.matches("![Logo](.markitai/assets/").count(), 1);
        assert_eq!(content.matches("![Again](.markitai/assets/").count(), 1);
        // Explicit disposition retains the download even when the body uses it.
        assert!(attachments.starts_with("- [inline.png]("), "{attachments}");
        assert!(
            attachments.contains("- [unreferenced.png]("),
            "{attachments}"
        );
        assert!(attachments.contains("- [data.csv]("), "{attachments}");
        assert!(originals.contains("email-1-inline.png"));
        // The shown image's asset is the body's and holds its bytes.
        let shown = refs(&doc)
            .into_iter()
            .find(|target| target.contains("inline.png"))
            .unwrap();
        let name = shown.strip_prefix(".markitai/assets/").unwrap();
        assert_eq!(
            doc.assets
                .iter()
                .find(|asset| asset.name == name)
                .unwrap()
                .bytes,
            pixels
        );
        // A genuinely inline resource used by the body keeps the single asset.
        let only = message(multipart(
            "related",
            "inner",
            &[
                part(
                    "Content-Type: text/html",
                    b"<p>Body.</p><img src='cid:img1'>",
                ),
                part(
                    "Content-Type: image/png\r\nContent-ID: <img1>\r\nContent-Disposition: inline; filename=inline.png",
                    &pixels,
                ),
            ],
        ));
        let (doc, originals) = extract_with_attachments(&only).unwrap();
        assert!(!doc.markdown.contains("## Attachments"), "{}", doc.markdown);
        assert_eq!(refs(&doc).len(), 1);
        assert!(originals.is_empty());
        let unspecified = message(multipart(
            "related",
            "no-disposition",
            &[
                part("Content-Type: text/html", b"<img src='cid:img1'>"),
                part(
                    "Content-Type: image/png; name=inline.png\r\nContent-ID: <img1>",
                    &pixels,
                ),
            ],
        ));
        let (doc, originals) = extract_with_attachments(&unspecified).unwrap();
        assert!(!doc.markdown.contains("## Attachments"), "{}", doc.markdown);
        assert_eq!(refs(&doc).len(), 1);
        assert!(originals.is_empty());
    }

    #[test]
    fn sibling_related_sets_cannot_supply_or_override_each_others_cids() {
        let red = image([220, 10, 10]);
        let green = image([10, 220, 10]);
        let first = multipart("related", "first", &[
            part("Content-Type: text/html", b"<p>First body.</p><img src='cid:shared' alt='First'><img src='cid:second-only' alt='Unavailable'>"),
            part("Content-Type: image/png\r\nContent-ID: <shared>\r\nContent-Disposition: inline; filename=red.png", &red),
        ]);
        let second = multipart(
            "related",
            "second",
            &[
                part("Content-Type: text/html", b"<p>Not chosen second body.</p>"),
                part(
                    "Content-Type: image/png\r\nContent-ID: <shared>\r\nContent-Disposition: inline; filename=green.png",
                    &green,
                ),
                part(
                    "Content-Type: image/png\r\nContent-ID: <second-only>\r\nContent-Disposition: inline; filename=other.png",
                    &green,
                ),
            ],
        );
        let doc = extract(&message(multipart("mixed", "outer", &[first, second]))).unwrap();
        assert!(doc.markdown.contains("First body."));
        assert!(!doc.markdown.contains("Not chosen second body."));
        let target = refs(&doc)
            .into_iter()
            .find(|target| target.starts_with(".markitai/assets/"))
            .unwrap();
        let asset = doc
            .assets
            .iter()
            .find(|asset| target.ends_with(&asset.name))
            .unwrap();
        assert_eq!(asset.bytes, red);
        assert!(doc.markdown.contains("![Unavailable](cid:second-only)"));
        assert!(
            doc.warnings
                .iter()
                .any(|warning| warning.contains("second-only"))
        );
    }

    #[test]
    fn duplicate_missing_and_nonimage_ids_keep_original_image_labels_without_wrong_binding() {
        let pixels = image([80, 90, 100]);
        let doc = extract(&message(multipart("related", "outer", &[
            part("Content-Type: text/html", b"<p>Body</p><img src='cid:duplicate' alt='Duplicate'><img src='cid:missing' alt='Missing'><img src='cid:download' alt='Ordinary'>"),
            part("Content-Type: image/png\r\nContent-ID: <duplicate>\r\nContent-Disposition: inline; filename=one.png", &pixels),
            part("Content-Type: image/png\r\nContent-ID: <duplicate>\r\nContent-Disposition: inline; filename=two.png", &pixels),
            part("Content-Type: application/octet-stream\r\nContent-ID: <download>\r\nContent-Disposition: attachment; filename=looks-like-image.png", &pixels),
        ]))).unwrap();
        assert_eq!(
            refs(&doc),
            [
                "cid:duplicate",
                "cid:missing",
                "cid:download",
                ".markitai/assets/email-1-one.png",
                ".markitai/assets/email-2-two.png"
            ]
        );
        for label in ["Duplicate", "Missing", "Ordinary"] {
            assert!(doc.markdown.contains(&format!("![{label}]")));
        }
        assert_eq!(
            doc.warnings
                .iter()
                .filter(|warning| warning.contains("Content-ID"))
                .count(),
            3
        );
        assert_eq!(doc.assets.len(), 3);
        assert!(doc.markdown.contains(&format!(
            "- [looks-like-image.png](.markitai/assets/email-3-looks-like-image.png) ({:.1} KB)",
            pixels.len() as f64 / 1024.0
        )) || doc.markdown.contains(&format!(
            "- [looks-like-image.png](.markitai/assets/email-3-looks-like-image.png) ({} B)",
            pixels.len()
        )));
    }

    #[test]
    fn cid_lookup_prefers_exact_case_and_refuses_ambiguous_casefolding() {
        let bytes = message(multipart("related", "outer", &[
            part("Content-Type: text/html", b"<p>Case tests</p><img alt='Exact' src='cid:Case'><img alt='Ambiguous' src='cid:CASE'>"),
            part("Content-Type: image/png\r\nContent-ID: <Case>\r\nContent-Disposition: inline; filename=upper.png", &image([1,2,3])),
            part("Content-Type: image/png\r\nContent-ID: <case>\r\nContent-Disposition: inline; filename=lower.png", &image([3,2,1])),
        ]));
        let doc = extract(&bytes).unwrap();
        assert!(
            doc.markdown
                .contains("![Exact](.markitai/assets/email-1-upper.png)")
        );
        assert!(doc.markdown.contains("![Ambiguous](cid:CASE)"));
        assert_eq!(
            doc.warnings
                .iter()
                .filter(|warning| warning.contains("Content-ID"))
                .count(),
            1
        );
    }

    #[test]
    fn ordinary_html_text_code_comments_and_unrelated_attributes_are_not_cid_resources() {
        let html = br#"<p>Literal ![example](cid:chart) and cid:chart.</p><pre><code>&lt;img src="cid:chart"&gt;</code></pre><!-- <img src='cid:chart'> --><p title="<img src='cid:chart'>">Title</p><script>const sample = '<img src="cid:chart">';</script><img src='cid:chart' alt='Visible'>"#;
        let doc = extract(&message(multipart("related", "outer", &[
            part("Content-Type: text/html", html),
            part("Content-Type: image/png\r\nContent-ID: <chart>\r\nContent-Disposition: inline; filename=chart.png", &image([1,2,3])),
        ]))).unwrap();
        assert!(doc.markdown.contains("cid:chart"));
        assert!(doc.markdown.contains("<img src=\"cid:chart\">"));
        assert!(
            doc.markdown
                .contains("![Visible](.markitai/assets/email-1-chart.png)")
        );
        // Only the real `img` is bound; the image the body shows is not listed
        // again under the attachments.
        assert_eq!(
            doc.markdown
                .matches(".markitai/assets/email-1-chart.png")
                .count(),
            1
        );
        assert!(!doc.markdown.contains("## Attachments"));
        assert!(!doc.markdown.contains("const sample"));
    }

    #[test]
    fn mime_charset_quoted_printable_and_encoded_attachment_names_are_decoded_safely() {
        let body = "Content-Type: text/html; charset=iso-8859-1\r\nContent-Transfer-Encoding: quoted-printable\r\n\r\n<p>caf=E9</p><img src=3D\"cid:chart\" alt=3D\"Caf=E9\">\r\n".to_owned();
        let doc = extract(&message(multipart("related", "outer", &[
            body,
            part("Content-Type: image/png\r\nContent-ID: <chart>\r\nContent-Disposition: inline; filename*=utf-8''..%2F..%2Fchart%20%C3%A9.png", &image([4,5,6])),
        ]))).unwrap();
        assert!(doc.markdown.contains("café"));
        assert!(doc.markdown.contains("![Café]"));
        assert_eq!(doc.assets.len(), 1);
        assert!(doc.assets[0].name.starts_with("email-1-"));
        assert!(!doc.assets[0].name.contains(['/', '\\', ' ', '<', '>']));
        assert!(doc.assets[0].name.ends_with(".png"));
        assert!(doc.warnings.is_empty(), "{:?}", doc.warnings);
    }

    #[test]
    fn multibyte_east_asian_charsets_decode_subjects_and_bodies() {
        // GB2312 subject and body (base64), Big5, Shift_JIS, EUC-KR bodies
        // (8bit) and ISO-2022-JP (7bit escape sequences).
        let gb_subject = b"\xb2\xe2\xca\xd4\xd6\xf7\xcc\xe2"; // 测试主题
        let gb_body = b"\xc4\xe3\xba\xc3\xa3\xac\xca\xc0\xbd\xe7"; // 你好，世界
        let base64 = |bytes: &[u8]| {
            use base64::Engine as _;
            base64::engine::general_purpose::STANDARD.encode(bytes)
        };
        let mail = format!(
            "From: a@example.com\r\nSubject: =?gb2312?B?{}?=\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset=gb2312\r\nContent-Transfer-Encoding: base64\r\n\r\n{}\r\n",
            base64(gb_subject),
            base64(gb_body)
        );
        let doc = extract(mail.as_bytes()).unwrap();
        assert!(
            doc.markdown.contains("**Subject:** 测试主题"),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("你好，世界"), "{}", doc.markdown);
        for (charset, encoded, expected) in [
            ("big5", &b"\xa4\xa4\xa4\xe5"[..], "中文"),
            (
                "shift_jis",
                &b"\x82\xb1\x82\xf1\x82\xc9\x82\xbf\x82\xcd"[..],
                "こんにちは",
            ),
            ("euc-kr", &b"\xc7\xd1\xb1\xb9\xbe\xee"[..], "한국어"),
            ("iso-2022-jp", &b"\x1b$B$3$s$K$A$O\x1b(B"[..], "こんにちは"),
        ] {
            let mut mail = format!(
                "From: a@example.com\r\nSubject: s\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset={charset}\r\nContent-Transfer-Encoding: 8bit\r\n\r\n"
            )
            .into_bytes();
            mail.extend_from_slice(encoded);
            mail.extend_from_slice(b"\r\n");
            let doc = extract(&mail).unwrap();
            assert!(
                doc.markdown.contains(expected),
                "{charset}: {}",
                doc.markdown
            );
            assert!(
                !doc.markdown.contains('\u{fffd}'),
                "{charset}: {}",
                doc.markdown
            );
        }
    }

    #[test]
    fn malformed_transfer_encoding_and_invalid_cid_escapes_cannot_create_image_bindings() {
        let doc = extract(&message(multipart("related", "outer", &[
            part("Content-Type: text/html", b"<p>Surviving body.</p><img src='cid:bad' alt='Bad bytes'><img src='cid:%0Aevil' alt='Bad ID'><img src='cid:bad%XZ' alt='Bad escape'>"),
            "Content-Type: image/png\r\nContent-ID: <bad>\r\nContent-Disposition: inline; filename=bad.png\r\nContent-Transfer-Encoding: base64\r\n\r\n!not base64!\r\n".into(),
        ]))).unwrap();
        assert!(doc.markdown.contains("Surviving body."));
        assert_eq!(refs(&doc), ["cid:bad", "cid:%0Aevil", "cid:bad%XZ"]);
        assert!(
            doc.markdown
                .contains("- [bad.png](.markitai/assets/email-1-bad.png) (")
        );
        assert!(
            doc.warnings
                .iter()
                .any(|warning| warning.contains("malformed transfer encoding"))
        );
    }

    #[test]
    fn unresolved_cid_placeholders_cannot_capture_an_entity_encoded_original_target() {
        let doc = extract(&message(part("Content-Type: text/html", b"<p>Body.</p><img src='.markitai&#45;mail-unresolved-0-0' alt='Original'><img src='cid:missing' alt='Missing'>"))).unwrap();
        assert_eq!(refs(&doc), [".markitai-mail-unresolved-0-0", "cid:missing"]);
        assert!(
            doc.markdown
                .contains("![Original](.markitai-mail-unresolved-0-0)")
        );
        assert!(doc.markdown.contains("![Missing](cid:missing)"));
    }

    #[test]
    fn excessive_multipart_depth_is_rejected_before_recursive_rendering() {
        let mut body = part("Content-Type: text/plain", b"Deep text");
        for index in 0..=MAX_DEPTH {
            body = multipart("mixed", &format!("depth-{index}"), &[body]);
        }
        let error = extract(&message(body)).unwrap_err();
        assert!(error.to_string().contains("nesting exceeds 64"), "{error}");
    }
}
