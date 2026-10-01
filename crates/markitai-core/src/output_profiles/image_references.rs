//! Image-only operations reuse the profile parser's literal and URI boundaries.
use super::*;
use std::ops::Range;

struct DefinitionSource {
    target: String,
    title: String,
}

enum Target {
    Direct(Range<usize>, TargetSyntax),
    Srcset(Range<usize>),
    Indirect {
        whole: Range<usize>,
        alt: String,
        title: String,
    },
}

enum Caption {
    Markdown(Range<usize>),
    Reference {
        whole: Range<usize>,
        suffix: String,
    },
    Wiki {
        range: Range<usize>,
        insert: bool,
    },
    Html {
        tag: usize,
        range: Range<usize>,
        insert: bool,
    },
    None,
}

struct ImageUse {
    uri: String,
    target: Target,
    caption: Caption,
}

/// Return syntax-unescaped URI values in appearance order, without decoding
/// percent escapes or dropping query/fragment data. Duplicate values occur once.
pub(crate) fn image_references(markdown: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    uses(markdown)
        .into_iter()
        .filter_map(|image| seen.insert(image.uri.clone()).then_some(image.uri))
        .collect()
}

/// Localize only actual image targets; ordinary links sharing a definition stay
/// untouched. Map keys are complete original URIs, values are raw local paths.
pub(crate) fn rewrite_image_targets(markdown: &str, paths: &HashMap<String, String>) -> String {
    rewrite_targets(markdown, uses(markdown), paths, false)
}

/// Restore complete URI targets while retaining their percent escapes, queries
/// and fragments. Callers supply URIs, not raw filesystem paths.
pub(crate) fn rewrite_image_uri_targets(markdown: &str, uris: &HashMap<String, String>) -> String {
    rewrite_targets(markdown, uses(markdown), uris, true)
}

/// Inspect raw HTML before sanitization without interpreting Markdown syntax.
pub(crate) fn html_image_references(html: &str) -> Vec<String> {
    let mut seen = HashSet::new();
    raw_html_uses(html)
        .into_iter()
        .filter_map(|image| seen.insert(image.uri.clone()).then_some(image.uri))
        .collect()
}

pub(crate) fn rewrite_html_image_targets(html: &str, paths: &HashMap<String, String>) -> String {
    rewrite_targets(html, raw_html_uses(html), paths, false)
}

fn raw_html_uses(html: &str) -> Vec<ImageUse> {
    let mut output = Vec::new();
    let mut index = 0;
    while let Some(relative) = html[index..].find('<') {
        index += relative;
        let tail = &html[index..];
        if let Some(end) = html_literal_end(tail) {
            index += end;
        } else if let Some(reference) = html_reference(tail) {
            if reference.name.eq_ignore_ascii_case("plaintext") {
                break;
            }
            if ["textarea", "title", "xmp", "iframe", "noembed", "noframes"]
                .iter()
                .any(|name| reference.name.eq_ignore_ascii_case(name))
            {
                index += html_closing(tail, reference.tag_end, reference.name)
                    .map_or(tail.len(), |(_, end)| end);
            } else {
                html_uses(tail, index, &reference, &mut output);
                index += reference.tag_end;
            }
        } else if let Some(end) = html_tag_end(tail) {
            index += end;
        } else {
            index += 1;
        }
    }
    output
}

fn rewrite_targets(
    markdown: &str,
    images: Vec<ImageUse>,
    paths: &HashMap<String, String>,
    uri: bool,
) -> String {
    let edits = images
        .into_iter()
        .filter_map(|image| {
            let next = paths.get(&image.uri)?;
            if next == &image.uri {
                return None;
            }
            Some(match image.target {
                Target::Direct(range, syntax) => {
                    let value = match syntax {
                        TargetSyntax::Html if uri => html_escape(&uri_destination(next, false)),
                        _ if uri => uri_destination(next, false),
                        TargetSyntax::Markdown => destination(next, false),
                        TargetSyntax::Html => html_escape(&html_file_path(next)),
                        TargetSyntax::Wiki => uri_file_path(next, true),
                    };
                    (range, value)
                }
                Target::Srcset(range) => (
                    range,
                    if uri {
                        html_escape(&uri_destination(next, true))
                    } else {
                        srcset_destination(TargetRewrite {
                            path: next,
                            suffix: "",
                        })
                    },
                ),
                Target::Indirect { whole, alt, title } => (
                    whole,
                    format!(
                        "![{alt}]({}{title})",
                        if uri {
                            uri_destination(next, false)
                        } else {
                            destination(next, false)
                        }
                    ),
                ),
            })
        })
        .collect();
    apply_edits(markdown, edits)
}

fn uri_destination(value: &str, srcset: bool) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        if ch.is_whitespace()
            || ch.is_control()
            || matches!(
                ch,
                '(' | ')' | '<' | '>' | '\\' | '"' | '\'' | '[' | ']' | '|'
            )
            || (srcset && ch == ',')
        {
            let mut bytes = [0; 4];
            for byte in ch.encode_utf8(&mut bytes).as_bytes() {
                use std::fmt::Write;
                write!(output, "%{byte:02X}").expect("writing to String cannot fail");
            }
        } else {
            output.push(ch);
        }
    }
    output
}

pub(crate) fn replace_image_alts(markdown: &str, captions: &HashMap<String, String>) -> String {
    let mut html_tags = HashSet::new();
    let edits = uses(markdown)
        .into_iter()
        .filter_map(|image| {
            let caption = captions.get(&image.uri)?;
            let caption = caption.split_whitespace().collect::<Vec<_>>().join(" ");
            if caption.is_empty() {
                return None;
            }
            Some(match image.caption {
                Caption::Markdown(range) => (range, markdown_caption(&caption)),
                Caption::Reference { whole, suffix } => {
                    (whole, format!("![{}]{suffix}", markdown_caption(&caption)))
                }
                Caption::Wiki { range, insert } => {
                    let caption = caption
                        .replace('|', "&#124;")
                        .replace('[', "&#91;")
                        .replace(']', "&#93;");
                    (range, format!("{}{caption}", if insert { "|" } else { "" }))
                }
                Caption::Html { tag, range, insert } if html_tags.insert(tag) => (
                    range,
                    format!(
                        "{}alt=\"{}\"",
                        if insert { " " } else { "" },
                        html_escape(&caption)
                    ),
                ),
                Caption::None | Caption::Html { .. } => return None,
            })
        })
        .collect();
    apply_edits(markdown, edits)
}

fn markdown_caption(value: &str) -> String {
    let mut output = String::with_capacity(value.len());
    for ch in value.chars() {
        if matches!(ch, '\\' | '[' | ']' | '`' | '*' | '_' | '<' | '>') {
            output.push('\\');
        }
        output.push(ch);
    }
    output
}

fn apply_edits(source: &str, mut edits: Vec<(Range<usize>, String)>) -> String {
    crate::sort::by_key(&mut edits, |(range, _)| range.start);
    let mut output = String::with_capacity(source.len());
    let mut cursor = 0;
    for (range, next) in edits {
        if range.start < cursor {
            continue;
        }
        output.push_str(&source[cursor..range.start]);
        output.push_str(&next);
        cursor = range.end;
    }
    output.push_str(&source[cursor..]);
    output
}

fn uses(markdown: &str) -> Vec<ImageUse> {
    let (_, body) = crate::output::split_frontmatter(markdown);
    let body_offset = markdown.len() - body.len();
    let mut definitions = HashMap::new();
    let mut context = LiteralContext::default();
    let mut cursor = body_offset;
    while let Some((line, literal)) = next_content(markdown, &mut cursor, &mut context) {
        if !literal && let Some(def) = definition(line) {
            let title = line[def.end + usize::from(def.angle)..].trim();
            definitions
                .entry(def.label)
                .or_insert_with(|| DefinitionSource {
                    target: unescape(def.target),
                    title: if title.is_empty() {
                        String::new()
                    } else {
                        format!(" {title}")
                    },
                });
        }
    }
    let mut output = Vec::new();
    let mut context = LiteralContext::default();
    let mut cursor = body_offset;
    loop {
        let offset = cursor;
        let Some((line, literal)) = next_content(markdown, &mut cursor, &mut context) else {
            break;
        };
        if !literal && definition(line).is_none() {
            scan(line, offset, &definitions, &mut output, 0);
        }
    }
    output
}

fn scan(
    text: &str,
    base: usize,
    definitions: &HashMap<String, DefinitionSource>,
    output: &mut Vec<ImageUse>,
    depth: usize,
) {
    if depth >= 64 {
        return;
    }
    let mut index = 0;
    while index < text.len() {
        let tail = &text[index..];
        let offset = base + index;
        if let Some(rest) = tail.strip_prefix('\\') {
            index += 1 + rest.chars().next().map_or(0, char::len_utf8);
        } else if let Some(end) = code_span_end(tail).or_else(|| html_literal_end(tail)) {
            index += end;
        } else if let Some(reference) = html_reference(tail) {
            html_uses(tail, offset, &reference, output);
            index += reference.tag_end;
        } else if let Some(end) = html_tag_end(tail) {
            index += end;
        } else if let Some(inner) = tail
            .strip_prefix("![[")
            .and_then(|rest| rest.find("]]").map(|end| &rest[..end]))
        {
            let (target, alias) = inner
                .split_once('|')
                .map_or((inner, None), |(target, alias)| (target, Some(alias)));
            let caption_start = offset + 3 + target.len();
            if !target.is_empty() {
                output.push(ImageUse {
                    uri: target.to_owned(),
                    target: Target::Direct(
                        offset + 3..offset + 3 + target.len(),
                        TargetSyntax::Wiki,
                    ),
                    caption: Caption::Wiki {
                        range: if let Some(alias) = alias {
                            caption_start + 1..caption_start + 1 + alias.len()
                        } else {
                            caption_start..caption_start
                        },
                        insert: alias.is_none(),
                    },
                });
            }
            index += inner.len() + 5;
        } else if let Some(reference) = reference(tail) {
            if reference.image {
                output.push(ImageUse {
                    uri: unescape(reference.target),
                    target: Target::Direct(
                        offset + reference.target_start..offset + reference.target_end,
                        TargetSyntax::Markdown,
                    ),
                    caption: Caption::Markdown(offset + 2..offset + 2 + reference.alt.len()),
                });
            } else {
                scan(reference.alt, offset + 1, definitions, output, depth + 1);
            }
            index += reference.end;
        } else if let Some((name, alt, end, image)) = reference_use(tail) {
            if image {
                if let Some(def) = definitions.get(&name) {
                    let suffix = &tail[alt.len() + 3..end];
                    output.push(ImageUse {
                        uri: def.target.clone(),
                        target: Target::Indirect {
                            whole: offset..offset + end,
                            alt: alt.into(),
                            title: def.title.clone(),
                        },
                        caption: Caption::Reference {
                            whole: offset..offset + end,
                            suffix: if suffix.is_empty() || suffix == "[]" {
                                format!("[{alt}]")
                            } else {
                                suffix.into()
                            },
                        },
                    });
                }
            } else {
                scan(alt, offset + 1, definitions, output, depth + 1);
            }
            index += end;
        } else {
            index += tail.chars().next().unwrap().len_utf8();
        }
    }
}

fn html_uses(text: &str, offset: usize, reference: &HtmlReference<'_>, output: &mut Vec<ImageUse>) {
    for attribute in &reference.attributes {
        if !attribute.active || !reference.image_attribute(attribute.kind) {
            continue;
        }
        let raw = &text[attribute.value.clone()];
        let decoded = DecodedHtml::new(raw);
        let ranges = if attribute.kind == HtmlAttributeKind::Srcset {
            srcset_candidates(&decoded.text)
                .filter(|candidate| candidate.valid)
                .map(|candidate| candidate.url)
                .collect::<Vec<_>>()
        } else {
            let start = decoded.text.len() - decoded.text.trim_start_matches(html_space).len();
            let end = decoded.text.trim_end_matches(html_space).len();
            if start >= end {
                continue;
            }
            std::iter::once(start..end).collect()
        };
        for range in ranges {
            let start = offset + attribute.value.start + decoded.raw_offset(range.start);
            let end = offset + attribute.value.start + decoded.raw_offset(range.end);
            let caption = if reference.name.eq_ignore_ascii_case("img") {
                match reference
                    .attributes
                    .iter()
                    .find(|attr| attr.active && attr.kind == HtmlAttributeKind::Alt)
                {
                    Some(alt) => Caption::Html {
                        tag: offset,
                        range: offset + alt.start..offset + alt.end,
                        insert: false,
                    },
                    None => Caption::Html {
                        tag: offset,
                        range: offset + 1 + reference.name.len()..offset + 1 + reference.name.len(),
                        insert: true,
                    },
                }
            } else {
                Caption::None
            };
            let target = if attribute.kind == HtmlAttributeKind::Srcset {
                Target::Srcset(start..end)
            } else {
                Target::Direct(start..end, TargetSyntax::Html)
            };
            output.push(ImageUse {
                uri: decoded.text[range].to_owned(),
                target,
                caption,
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn map(entries: &[(&str, &str)]) -> HashMap<String, String> {
        entries
            .iter()
            .map(|(key, value)| ((*key).into(), (*value).into()))
            .collect()
    }

    #[test]
    fn complete_image_uris_preserve_queries_escapes_and_literal_boundaries() {
        let text = "![a](https://x.test/a%20b.png?variant=1&k=2) [![b](b\\(c\\).png)](outer)\n![Ref][id] ![[wiki.png|old]]\n[id]: <a%23b.png> \"title\"\n<img src='https://x.test/img?v=2&amp;x=3' srcset='one.png 1x, two.png 2x'>\n<video poster='poster.png' src='video.mp4'></video>\n<!-- ![hidden](comment.png) -->\n`![code](inline.png)`\n```md\n![code](fenced.png)\n```\n";
        assert_eq!(
            image_references(text),
            [
                "https://x.test/a%20b.png?variant=1&k=2",
                "b(c).png",
                "a%23b.png",
                "wiki.png",
                "https://x.test/img?v=2&x=3",
                "one.png",
                "two.png",
                "poster.png"
            ]
        );
    }

    #[test]
    fn localizing_reference_images_never_rewrites_shared_download_links() {
        let text = "![old][id] [download][id]\n[id]: <https://x/a.png?size=1> \"Keep title\"\n![other](https://x/a.png?size=2)\n";
        let rewritten = rewrite_image_targets(
            text,
            &map(&[("https://x/a.png?size=1", ".markitai/assets/a #1.png")]),
        );
        assert_eq!(
            rewritten,
            "![old](.markitai/assets/a%20%231.png \"Keep title\") [download][id]\n[id]: <https://x/a.png?size=1> \"Keep title\"\n![other](https://x/a.png?size=2)\n"
        );
    }

    #[test]
    fn replacing_captions_preserves_reference_identity_and_nested_link_destination() {
        let text =
            "![id] ![id][] ![old][id] [![old](a.png \"t\")](jump)\n[id]: a.png \"ref title\"\n";
        let result = replace_image_alts(text, &map(&[("a.png", "new [label]\nnext")]));
        assert_eq!(
            result,
            "![new \\[label\\] next][id] ![new \\[label\\] next][id] ![new \\[label\\] next][id] [![new \\[label\\] next](a.png \"t\")](jump)\n[id]: a.png \"ref title\"\n"
        );
    }

    #[test]
    fn html_caption_updates_handle_missing_boolean_duplicate_and_multiline_attributes() {
        let text = "<IMG\nsrc='a.png'> <img alt src=a.png> <img alt='old' ALT='inactive' src=a.png>\n<img srcset='one.png 1x, a.png 2x'>\n";
        let result = replace_image_alts(text, &map(&[("a.png", "new & caption")]));
        assert_eq!(
            result.matches("alt=\"new&#32;&amp;&#32;caption\"").count(),
            4
        );
        assert!(result.contains("ALT='inactive'"));
        assert!(result.contains("srcset='one.png 1x, a.png 2x'"));
        assert_eq!(image_references(&result), ["a.png", "one.png"]);
    }

    #[test]
    fn html_srcset_localization_maps_entities_without_touching_invalid_candidates() {
        let text =
            "<img srcset=\"a.png?v=1&amp;x=2 1x, a.png?v=2 2x, ignored.png 1x 2x\" alt='keep'>";
        let result = rewrite_image_targets(
            text,
            &map(&[
                ("a.png?v=1&x=2", "assets/one.png"),
                ("a.png?v=2", "assets/two.png"),
                ("ignored.png", "bad"),
            ]),
        );
        assert_eq!(
            result,
            "<img srcset=\"assets/one.png 1x, assets/two.png 2x, ignored.png 1x 2x\" alt='keep'>"
        );
    }

    #[test]
    fn localized_srcset_encodes_candidate_delimiters() {
        let result = rewrite_image_targets(
            "<img srcset='old.png 1x'>",
            &map(&[("old.png", "assets/a b,.png")]),
        );
        assert_eq!(result, "<img srcset='assets/a%20b%2C.png 1x'>");
    }

    #[test]
    fn raw_html_images_do_not_interpret_markdown_or_attribute_examples() {
        let html = "<p>![literal](cid:markdown) `<img src='cid:real'>`</p><div data-example=\"<img src='cid:attribute'>\"></div><textarea><img src='cid:textarea'></textarea><script><img src='cid:script'></script><!-- <img src='cid:comment'> --><pre><img src='cid:pre'></pre><img srcset='cid:one 1x, cid:two 2x'>";
        assert_eq!(
            html_image_references(html),
            ["cid:real", "cid:one", "cid:two"]
        );
        let changed = rewrite_html_image_targets(
            html,
            &map(&[
                ("cid:real", "assets/a.png"),
                ("cid:one", "assets/one, x.png"),
                ("cid:markdown", "wrong"),
            ]),
        );
        assert!(changed.contains("![literal](cid:markdown)"));
        assert!(changed.contains("`<img src='assets/a.png'>`"));
        assert!(changed.contains("assets/one%2C%20x.png 1x, cid:two 2x"));
        assert!(changed.contains("<textarea><img src='cid:textarea'></textarea>"));
    }

    #[test]
    fn restored_uris_preserve_encoding_and_only_change_image_uses() {
        let source = "![direct](placeholder) ![ref][id] [download][id] `![code](placeholder)`\n[id]: placeholder \"title\"\n<img src='placeholder'><img srcset='placeholder 1x'>";
        let result =
            rewrite_image_uri_targets(source, &map(&[("placeholder", "cid:a%2Fb?x=1&y=2#frame")]));
        assert!(result.contains("![direct](cid:a%2Fb?x=1&y=2#frame)"));
        assert!(result.contains("![ref](cid:a%2Fb?x=1&y=2#frame \"title\") [download][id]"));
        assert!(result.contains("`![code](placeholder)`\n[id]: placeholder \"title\""));
        assert!(result.contains("src='cid:a%2Fb?x&#61;1&amp;y&#61;2#frame'"));
        assert!(result.contains("srcset='cid:a%2Fb?x&#61;1&amp;y&#61;2#frame 1x'"));
        assert_eq!(image_references(&result), ["cid:a%2Fb?x=1&y=2#frame"]);
        let result =
            rewrite_image_uri_targets("<img srcset='old 2x'>", &map(&[("old", "cid:a,b %23\n")]));
        assert_eq!(result, "<img srcset='cid:a%2Cb%20%23%0A 2x'>");
    }

    #[test]
    fn frontmatter_examples_are_not_document_images() {
        let text = "---\nexample: '![sample](not-an-image.png)'\n---\n\n![body](body.png)\n";
        assert_eq!(image_references(text), ["body.png"]);
        let result = replace_image_alts(
            text,
            &map(&[("body.png", "new"), ("not-an-image.png", "wrong")]),
        );
        assert!(result.contains("example: '![sample](not-an-image.png)'"));
        assert!(result.ends_with("![new](body.png)\n"));
    }
}
