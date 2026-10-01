use super::Attribute;
use crate::Result;
use scraper::{ElementRef, Html, Node};
use std::collections::{HashMap, HashSet};

const MAX_NODES: usize = 200_000;
const MAX_IDS: usize = 512;
const MAX_OPERATIONS: usize = 256;
const MAX_SCRIPT_BYTES: usize = 2 * 1024 * 1024;
const MAX_CHAIN: usize = 16;
const MAX_DEPTH: usize = 128;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Boundary,
    Segment,
    Snapshot,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Instruction {
    target: String,
    segment: String,
    kind: Kind,
}

fn stream_id(id: &str, prefix: &str) -> bool {
    id.strip_prefix(prefix).is_some_and(|part| {
        !part.is_empty() && part.len() <= 16 && part.bytes().all(|b| b.is_ascii_hexdigit())
    })
}

fn hidden(element: ElementRef<'_>) -> bool {
    super::is_hidden(element)
        || element.value().attribute("aria-hidden") == Some("true")
        || element
            .value()
            .classes()
            .any(|c| matches!(c, "hidden" | "invisible"))
}

fn inert_ancestor(element: ElementRef<'_>) -> bool {
    element
        .ancestors()
        .filter_map(ElementRef::wrap)
        .any(|ancestor| {
            matches!(
                ancestor.value().name(),
                "template" | "noscript" | "iframe" | "object" | "embed"
            )
        })
}

fn snapshot_context(element: ElementRef<'_>) -> bool {
    !inert_ancestor(element)
        && !element
            .ancestors()
            .filter_map(ElementRef::wrap)
            .any(|ancestor| matches!(ancestor.value().name(), "head" | "pre" | "code"))
}

fn segment(element: ElementRef<'_>) -> bool {
    element.value().name() == "div"
        && element.value().attribute("hidden").is_some()
        && element.value().attribute("style").is_none()
        && element.value().attribute("aria-hidden").is_none()
        && !element
            .value()
            .classes()
            .any(|class| matches!(class, "hidden" | "invisible"))
        && !inert_ancestor(element)
}

fn placeholder(element: ElementRef<'_>) -> bool {
    element.value().name() == "template"
        && !inert_ancestor(element)
        && element
            .value()
            .attrs
            .iter()
            .all(|(name, _)| name.local.as_ref() == "id")
        && element.children().all(|node| match node.value() {
            Node::Fragment => node
                .children()
                .all(|child| matches!(child.value(), Node::Text(t) if t.trim().is_empty())),
            Node::Text(text) => text.trim().is_empty(),
            _ => false,
        })
}

fn executable_script(element: ElementRef<'_>) -> bool {
    element.value().name() == "script"
        && !inert_ancestor(element)
        && element.value().attribute("type").is_none_or(|kind| {
            matches!(
                kind.trim().to_ascii_lowercase().as_str(),
                "" | "module"
                    | "text/javascript"
                    | "application/javascript"
                    | "text/ecmascript"
                    | "application/ecmascript"
            )
        })
}

struct Index<'a> {
    elements: HashMap<String, ElementRef<'a>>,
    duplicates: HashSet<String>,
    calls: Vec<Instruction>,
    has_script: bool,
    has_marker: bool,
}

impl<'a> Index<'a> {
    fn new(document: &'a Html) -> Option<Self> {
        let mut index = Self {
            elements: HashMap::new(),
            duplicates: HashSet::new(),
            calls: Vec::new(),
            has_script: false,
            has_marker: false,
        };
        let mut scripts = Vec::new();
        for (count, node) in document.root_element().descendants().enumerate() {
            if count >= MAX_NODES {
                return None;
            }
            if let Node::Comment(comment) = node.value() {
                index.has_marker |= matches!(comment.comment.as_ref(), "$" | "$?" | "$!" | "/$");
            }
            let Some(element) = ElementRef::wrap(node) else {
                continue;
            };
            if let Some(id) = element.value().attribute("id")
                && ["B:", "S:", "P:"]
                    .iter()
                    .any(|prefix| stream_id(id, prefix))
            {
                if index.elements.insert(id.to_owned(), element).is_some() {
                    index.duplicates.insert(id.to_owned());
                }
                if index.elements.len() > MAX_IDS {
                    return None;
                }
            }
            if executable_script(element) {
                index.has_script = true;
                if element.value().attribute("src").is_none() {
                    scripts.push(element);
                }
            }
        }
        if index.elements.is_empty() {
            return Some(index);
        }
        let mut script_bytes = 0usize;
        for script in scripts {
            let mut source = String::new();
            for text in script.text() {
                script_bytes = script_bytes.checked_add(text.len())?;
                if script_bytes > MAX_SCRIPT_BYTES {
                    return None;
                }
                source.push_str(text);
            }
            index.calls.extend(calls(&source));
            if index.calls.len() > MAX_OPERATIONS {
                return None;
            }
        }
        Some(index)
    }

    fn get(&self, id: &str) -> Option<ElementRef<'a>> {
        (!self.duplicates.contains(id))
            .then(|| self.elements.get(id).copied())
            .flatten()
    }

    fn eligible(&self, instruction: &Instruction) -> bool {
        self.get(&instruction.target).is_some_and(placeholder)
            && self.get(&instruction.segment).is_some_and(segment)
    }
}

// Recognize complete top-level instruction statements, not arbitrary JavaScript
// substrings. Definitions, strings, comments and conditional calls are inert.
fn calls(source: &str) -> Vec<Instruction> {
    let bytes = source.as_bytes();
    let mut result = Vec::new();
    let mut position = 0;
    let mut start = 0;
    let mut nesting = Vec::new();
    while position < bytes.len() {
        match bytes[position] {
            b'`' => return Vec::new(),
            b'\'' | b'"' => {
                let quote = bytes[position];
                position += 1;
                while position < bytes.len() && bytes[position] != quote {
                    if bytes[position] == b'\\' {
                        position += 1;
                    }
                    position += 1;
                }
                if position >= bytes.len() {
                    return Vec::new();
                }
            }
            b'/' if bytes.get(position + 1) == Some(&b'/') => {
                let comment_start = position;
                position += 2;
                while position < bytes.len() && bytes[position] != b'\n' {
                    position += 1;
                }
                if nesting.is_empty() && source[start..comment_start].trim().is_empty() {
                    start = position;
                }
                continue;
            }
            b'/' if bytes.get(position + 1) == Some(&b'*') => {
                let comment_start = position;
                position += 2;
                while position + 1 < bytes.len() && &bytes[position..position + 2] != b"*/" {
                    position += 1;
                }
                if position + 1 >= bytes.len() {
                    return Vec::new();
                }
                position += 2;
                if nesting.is_empty() && source[start..comment_start].trim().is_empty() {
                    start = position;
                }
                continue;
            }
            // A regex literal needs a JavaScript parser. Refuse that script
            // rather than accidentally treating a quoted example as a call.
            b'/' => return Vec::new(),
            b'(' | b'[' | b'{' => {
                nesting.push(bytes[position]);
                if nesting.len() > 64 {
                    return Vec::new();
                }
            }
            b')' | b']' | b'}' => {
                let expected = match bytes[position] {
                    b')' => b'(',
                    b']' => b'[',
                    _ => b'{',
                };
                if nesting.pop() != Some(expected) {
                    return Vec::new();
                }
            }
            b';' if nesting.is_empty() => {
                if let Some(call) = instruction(&source[start..position]) {
                    result.push(call);
                }
                start = position + 1;
            }
            _ => {}
        }
        position += 1;
    }
    if !nesting.is_empty() {
        return Vec::new();
    }
    if let Some(call) = instruction(&source[start..]) {
        result.push(call);
    }
    result
}

fn instruction(source: &str) -> Option<Instruction> {
    fn quoted(input: &str) -> Option<(&str, &str)> {
        let input = input.trim_start();
        let quote = input.as_bytes().first().copied()?;
        if !matches!(quote, b'\'' | b'"') {
            return None;
        }
        let end = input[1..].find(char::from(quote))? + 1;
        let text = &input[1..end];
        if text.contains('\\') {
            return None;
        }
        Some((text, &input[end + 1..]))
    }
    let source = source.trim();
    let (kind, rest) = if let Some(rest) = source.strip_prefix("$RC") {
        (Kind::Boundary, rest)
    } else {
        (Kind::Segment, source.strip_prefix("$RS")?)
    };
    let rest = rest.trim_start().strip_prefix('(')?;
    let (first, rest) = quoted(rest)?;
    let (second, rest) = quoted(rest.trim_start().strip_prefix(',')?)?;
    if rest.trim() != ")" {
        return None;
    }
    let (target, source) = if kind == Kind::Boundary {
        (first, second)
    } else {
        (second, first)
    };
    if !stream_id(source, "S:")
        || !stream_id(target, if kind == Kind::Boundary { "B:" } else { "P:" })
    {
        return None;
    }
    Some(Instruction {
        target: target.to_owned(),
        segment: source.to_owned(),
        kind,
    })
}

fn semantic(element: ElementRef<'_>) -> bool {
    matches!(element.value().name(), "article" | "main")
        || element
            .value()
            .attribute("role")
            .is_some_and(|v| v.split_ascii_whitespace().any(|p| p == "main"))
        || element
            .value()
            .attribute("itemprop")
            .is_some_and(|v| v.split_ascii_whitespace().any(|p| p == "articleBody"))
        || element
            .value()
            .attribute("id")
            .into_iter()
            .chain(element.value().classes())
            .any(|name| {
                matches!(
                    name.to_ascii_lowercase().as_str(),
                    "postbody"
                        | "post-body"
                        | "post-content"
                        | "article-body"
                        | "article-content"
                        | "entry-content"
                        | "story-body"
                )
            })
}

fn units(text: &str) -> usize {
    let mut result = 0usize;
    let mut word = false;
    for character in text.chars() {
        let east_asian = matches!(character as u32, 0x3040..=0x30ff | 0x3400..=0x9fff | 0xac00..=0xd7af | 0xf900..=0xfaff | 0x20000..=0x323af);
        if east_asian {
            result += 1;
            word = false;
        } else if character.is_alphanumeric() {
            if !word {
                result += 1;
            }
            word = true;
        } else {
            word = false;
        }
    }
    result
}

#[derive(Default)]
struct Content {
    units: usize,
    semantic_units: usize,
    paragraphs: usize,
}

fn content(root: ElementRef<'_>, ignore_root_hidden: bool) -> Option<Content> {
    let mut result = Content::default();
    let mut paragraphs = Vec::<usize>::new();
    let mut stack = vec![(root, false, None, 0usize)];
    while let Some((element, in_article, paragraph, depth)) = stack.pop() {
        if depth > MAX_DEPTH {
            return None;
        }
        if (hidden(element) && !(ignore_root_hidden && element == root))
            || matches!(
                element.value().name(),
                "head" | "script" | "style" | "template" | "nav" | "footer" | "form" | "button"
            )
        {
            continue;
        }
        let in_article = in_article || semantic(element);
        let paragraph = if in_article && element.value().name() == "p" {
            paragraphs.push(0);
            Some(paragraphs.len() - 1)
        } else {
            paragraph
        };
        for child in element.children() {
            match child.value() {
                Node::Text(text) => {
                    let size = units(text);
                    result.units += size;
                    if in_article {
                        result.semantic_units += size;
                    }
                    if let Some(index) = paragraph {
                        paragraphs[index] += size;
                    }
                }
                Node::Element(_) => {
                    stack.push((ElementRef::wrap(child)?, in_article, paragraph, depth + 1))
                }
                _ => {}
            }
        }
    }
    result.paragraphs = paragraphs.into_iter().filter(|count| *count >= 10).count();
    Some(result)
}

fn snapshot(document: &Html, index: &Index<'_>) -> Option<Vec<Instruction>> {
    if index.has_script || index.has_marker || !index.duplicates.is_empty() {
        return None;
    }
    let visible = content(document.root_element(), false)?.units;
    if visible > 80 {
        return None;
    }
    let mut candidate = None;
    for (id, element) in &index.elements {
        if !stream_id(id, "S:") || !segment(*element) || !snapshot_context(*element) {
            continue;
        }
        let score = content(*element, true)?;
        if score.semantic_units < 200
            || score.semantic_units < visible.saturating_mul(4)
            || score.paragraphs < 3
        {
            continue;
        }
        if candidate.replace(id.clone()).is_some() {
            return None;
        }
    }
    let mut segment_id = candidate?;
    let mut seen = HashSet::new();
    let mut instructions = Vec::new();
    for _ in 0..MAX_CHAIN {
        if !seen.insert(segment_id.clone()) {
            return None;
        }
        let element = index.get(&segment_id)?;
        if !segment(element) || !snapshot_context(element) {
            return None;
        }
        // Only top-level transport wrappers may be removed. A hidden ordinary
        // ancestor is intentional visibility, not part of the transport chain.
        if element
            .ancestors()
            .filter_map(ElementRef::wrap)
            .any(|parent| hidden(parent))
        {
            return None;
        }
        let suffix = segment_id.strip_prefix("S:")?;
        let targets: Vec<_> = [format!("B:{suffix}"), format!("P:{suffix}")]
            .into_iter()
            .filter(|id| index.get(id).is_some())
            .collect();
        if targets.len() != 1 {
            return None;
        }
        let target = index.get(&targets[0])?;
        if !placeholder(target) || !snapshot_context(target) {
            return None;
        }
        let mut outer = None;
        for ancestor in target.ancestors().filter_map(ElementRef::wrap) {
            if hidden(ancestor) {
                let id = ancestor.value().attribute("id")?;
                if outer.is_some() || !stream_id(id, "S:") || !segment(ancestor) {
                    return None;
                }
                outer = Some(id.to_owned());
            }
        }
        instructions.push(Instruction {
            target: targets[0].clone(),
            segment: segment_id,
            kind: Kind::Snapshot,
        });
        match outer {
            Some(id) => segment_id = id,
            None => return Some(instructions),
        }
    }
    None
}

fn plan(document: &Html) -> Vec<Instruction> {
    let Some(index) = Index::new(document) else {
        return Vec::new();
    };
    if index.elements.is_empty() {
        return Vec::new();
    }
    if !index.has_script {
        return snapshot(document, &index).unwrap_or_default();
    }
    let mut targets: HashMap<&str, HashSet<&str>> = HashMap::new();
    let mut sources: HashMap<&str, HashSet<&str>> = HashMap::new();
    for call in &index.calls {
        targets
            .entry(&call.target)
            .or_default()
            .insert(&call.segment);
        sources
            .entry(&call.segment)
            .or_default()
            .insert(&call.target);
    }
    index
        .calls
        .iter()
        .filter(|call| {
            index.eligible(call)
                && targets[call.target.as_str()].len() == 1
                && sources[call.segment.as_str()].len() == 1
        })
        .cloned()
        .collect()
}

/// Restore recognized transport containers before selecting a page's article.
/// Ineligible or over-limit snapshots retain their original DOM unchanged.
pub(super) fn restore(document: &mut Html) -> Result<()> {
    let instructions = plan(document);
    if instructions.is_empty() {
        return Ok(());
    }
    let ids: HashMap<_, _> = document
        .root_element()
        .descendants()
        .filter_map(ElementRef::wrap)
        .filter_map(|element| {
            element
                .value()
                .attribute("id")
                .filter(|id| {
                    ["B:", "S:", "P:"]
                        .iter()
                        .any(|prefix| stream_id(id, prefix))
                })
                .map(|id| (id.to_owned(), element.id()))
        })
        .collect();
    for instruction in instructions {
        let (Some(&target_id), Some(&segment_id)) =
            (ids.get(&instruction.target), ids.get(&instruction.segment))
        else {
            continue;
        };
        let prepared = (|| {
            let target = document.tree.get(target_id)?;
            let source = document.tree.get(segment_id)?;
            if !target
                .ancestors()
                .any(|node| matches!(node.value(), Node::Document))
                || !source
                    .ancestors()
                    .any(|node| matches!(node.value(), Node::Document))
                || target.ancestors().any(|node| node.id() == segment_id)
                || source.ancestors().any(|node| node.id() == target_id)
            {
                return None;
            }
            let mut remove = vec![target_id];
            let mut start = None;
            let mut insertion = target_id;
            if instruction.kind == Kind::Boundary {
                let previous = target
                    .prev_siblings()
                    .find(|node| !matches!(node.value(), Node::Text(t) if t.trim().is_empty()))?;
                if !matches!(previous.value(), Node::Comment(c) if c.comment.as_ref() == "$?") {
                    return None;
                }
                let mut depth = 0usize;
                let mut end = None;
                for sibling in target.next_siblings() {
                    if let Node::Comment(comment) = sibling.value() {
                        match comment.comment.as_ref() {
                            "$" | "$?" | "$!" => depth += 1,
                            "/$" if depth == 0 => {
                                end = Some(sibling.id());
                                break;
                            }
                            "/$" => depth -= 1,
                            _ => {}
                        }
                    }
                    remove.push(sibling.id());
                }
                insertion = end?;
                start = Some(previous.id());
            }
            let removing: HashSet<_> = remove.iter().copied().collect();
            if source
                .ancestors()
                .any(|ancestor| removing.contains(&ancestor.id()))
                || removing.contains(&segment_id)
            {
                return None;
            }
            let children: Vec<_> = source.children().map(|child| child.id()).collect();
            Some((remove, insertion, start, children))
        })();
        let Some((remove, insertion, start, children)) = prepared else {
            continue;
        };
        for child in children {
            document
                .tree
                .get_mut(insertion)
                .expect("validated insertion")
                .insert_id_before(child);
        }
        for node in remove {
            document
                .tree
                .get_mut(node)
                .expect("validated fallback")
                .detach();
        }
        document
            .tree
            .get_mut(segment_id)
            .expect("validated segment")
            .detach();
        if let Some(start) = start
            && let Node::Comment(comment) = document
                .tree
                .get_mut(start)
                .expect("validated boundary")
                .value()
        {
            comment.comment = "$".into();
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn restore_html(source: &str) -> Html {
        let mut document = Html::parse_document(&format!("<body>{source}</body>"));
        restore(&mut document).unwrap();
        document
    }

    fn body(document: &Html) -> String {
        let root = document.root_element();
        super::super::render_clean(root, None).unwrap()
    }

    fn prose() -> String {
        format!(
            "<article><h1>Recovered heading</h1>{}</article>",
            (0..4)
                .map(|index| format!(
                    "<p>Paragraph {index} {}.</p>",
                    "substantial article prose ".repeat(20)
                ))
                .collect::<String>()
        )
    }

    #[test]
    fn completed_boundary_replaces_nested_fallback_and_keeps_hidden_descendants() {
        let document = restore_html(
            r#"<main><!--$?--><template id="B:0"></template><p>loading</p><!--$?--><template id="B:1"></template>nested<!--/$--><!--/$--></main><div hidden id="S:0"><h1>Complete</h1><p>Body</p><p hidden>secret</p></div><script>$RC("B:0","S:0")</script>"#,
        );
        let output = body(&document);
        assert!(output.contains("Complete") && output.contains("Body"));
        assert!(
            !output.contains("loading") && !output.contains("nested") && !output.contains("secret")
        );
    }

    #[test]
    fn segment_instructions_compose_with_boundary_in_order() {
        let document = restore_html(
            r#"<main><!--$?--><template id="B:a"></template>waiting<!--/$--></main><div hidden id="S:a"><template id="P:b"></template></div><div hidden id="S:b"><p>Nested result</p></div><script>$RC=function(a,b){let marker="/$";};$RS('S:b','P:b');$RC('B:a','S:a');</script>"#,
        );
        assert!(body(&document).contains("Nested result"));
        assert!(!body(&document).contains("waiting"));
    }

    #[test]
    fn quoted_commented_conditional_and_error_instructions_are_inert() {
        for call in [
            r#"const example='$RC("B:0","S:0")';"#,
            r#"/* $RC("B:0","S:0") */"#,
            r#"if(false) $RC("B:0","S:0");"#,
            r#"$RC("B:0","S:0","error")"#,
            r#"const re=/$RC("B:0","S:0")/;"#,
            "$RC(\"B:0\",\"S:0\"); \"unterminated\\",
        ] {
            let source = format!(
                r#"<main><!--$?--><template id="B:0"></template>waiting<!--/$--></main><div hidden id="S:0"><p>secret</p></div><script>{call}</script>"#
            );
            let output = body(&restore_html(&source));
            assert!(output.contains("waiting"), "{call}");
            assert!(!output.contains("secret"), "{call}");
        }
    }

    #[test]
    fn inert_templates_do_not_supply_instructions_or_transport_content() {
        let source = format!(
            r#"<head><template id="B:0"></template></head><body><div hidden id="S:0">{}</div></body>"#,
            prose()
        );
        let mut document = Html::parse_document(&source);
        let before = document.html();
        restore(&mut document).unwrap();
        assert_eq!(document.html(), before, "snapshot target in head");
        let waiting = r#"<main><!--$?--><template id="B:0"></template>waiting<!--/$--></main>"#;
        let source = format!(
            r#"{waiting}<div hidden id="S:0"><p>secret</p></div><template><script>$RC("B:0","S:0")</script></template>"#
        );
        assert!(body(&restore_html(&source)).contains("waiting"));
        assert!(!body(&restore_html(&source)).contains("secret"));
        let source = format!(
            r#"<template id="B:0"></template><template><div hidden id="S:0">{}</div></template>"#,
            prose()
        );
        assert!(!body(&restore_html(&source)).contains("Recovered heading"));
        for tag in ["pre", "code", "object"] {
            let source = format!(
                r#"<{tag}><template id="B:0"></template></{tag}><div hidden id="S:0">{}</div>"#,
                prose()
            );
            let mut document = Html::parse_document(&source);
            let before = document.html();
            restore(&mut document).unwrap();
            assert_eq!(document.html(), before, "snapshot target in {tag}");
        }
    }

    #[test]
    fn duplicate_ids_and_incomplete_boundaries_remain_untouched() {
        for boundary in [
            r#"<!--$?--><template id="B:0"></template>waiting"#,
            r#"<!--$?--><template id="B:0"></template><template id="B:0"></template>waiting<!--/$-->"#,
            r#"<!--$!--><template id="B:0"></template>waiting<!--/$-->"#,
        ] {
            let source = format!(
                r#"<main>{boundary}</main><div hidden id="S:0"><p>secret</p></div><script>$RC("B:0","S:0")</script>"#
            );
            let output = body(&restore_html(&source));
            assert!(output.contains("waiting"));
            assert!(!output.contains("secret"));
        }
    }

    #[test]
    fn stripped_snapshot_recovers_only_unique_article_chain() {
        let source = format!(
            r#"<template id="B:1"></template><div hidden id="S:1"><div><template id="B:3"></template></div></div><div hidden id="S:3"><template id="P:5"></template></div><div hidden id="S:5">{}<p hidden>secret</p><template id="B:7"></template></div><div hidden id="S:7"><p>Unrelated recommendation</p></div>"#,
            prose()
        );
        let document = restore_html(&source);
        let output = body(&document);
        assert!(output.contains("Recovered heading") && output.contains("Paragraph 3"));
        assert!(!output.contains("secret") && !output.contains("Unrelated recommendation"));
        assert_eq!(
            document
                .root_element()
                .descendants()
                .filter_map(ElementRef::wrap)
                .filter(|e| e.value().attribute("id") == Some("S:7"))
                .count(),
            1
        );
    }

    #[test]
    fn sparse_snapshot_uses_cjk_units_and_rejects_substantial_visible_cjk() {
        let article = format!(
            "<article>{}</article>",
            format!("<p>{}</p>", "这是一段明确的文章正文内容。".repeat(12)).repeat(3)
        );
        let transport =
            format!(r#"<template id="B:0"></template><div hidden id="S:0">{article}</div>"#);
        let recovered = body(&restore_html(&transport));
        assert!(recovered.contains("这是一段"));
        let visible = format!(
            "<main><p>{}</p></main>",
            "已有完整可见正文内容。".repeat(12)
        );
        let output = body(&restore_html(&(visible + &transport)));
        assert!(output.contains("已有完整"));
        assert!(!output.contains("这是一段"));
    }

    #[test]
    fn snapshot_rejects_ambiguous_unpaired_and_intentionally_hidden_content() {
        for source in [
            format!(
                r#"<template id="B:0"></template><div hidden id="S:0">{}</div><template id="B:1"></template><div hidden id="S:1">{}</div>"#,
                prose(),
                prose()
            ),
            format!(r#"<div hidden id="S:0">{}</div>"#, prose()),
            format!(
                r#"<template id="B:0"></template><template id="P:0"></template><div hidden id="S:0">{}</div>"#,
                prose()
            ),
            format!(
                r#"<section hidden><template id="B:0"></template></section><div hidden id="S:0">{}</div>"#,
                prose()
            ),
            format!(
                r#"<template id="B:0"></template><div hidden id="S:0"><section hidden>{}</section></div>"#,
                prose()
            ),
            format!(
                r#"<template id="B:0"></template><div hidden class="hidden" id="S:0">{}</div>"#,
                prose()
            ),
            format!(
                r#"<template id="B:0"></template><div hidden class="invisible" id="S:0">{}</div>"#,
                prose()
            ),
            format!(
                r#"<template id="B:0"></template><div hidden id="S:0">{}</div><div hidden id="S:0"></div>"#,
                prose()
            ),
            format!(
                r#"<template id="B:0"></template><div hidden id="S:0">{}</div><script>startApp()</script>"#,
                prose()
            ),
            format!(
                r#"<!--$?--><template id="B:0"></template><div hidden id="S:0">{}</div>"#,
                prose()
            ),
        ] {
            assert!(!body(&restore_html(&source)).contains("Recovered heading"));
        }
    }

    #[test]
    fn unrelated_hidden_prose_and_cyclic_chains_are_not_promoted() {
        let source = format!(
            r#"<template id="B:0"></template><div hidden id="S:0"><div>{}</div></div>"#,
            "<p>unstructured prose words </p>".repeat(200)
        );
        assert!(!body(&restore_html(&source)).contains("unstructured"));
        let cyclic = format!(
            r#"<div hidden id="S:0">{}<template id="B:1"></template></div><div hidden id="S:1"><template id="B:0"></template></div>"#,
            prose()
        );
        assert!(!body(&restore_html(&cyclic)).contains("Recovered heading"));
    }
}
