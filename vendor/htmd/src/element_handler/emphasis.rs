use crate::{
    Element,
    element_handler::{HandlerResult, Handlers},
    serialize_if_faithful,
    text_util::{StripWhitespace, concat_strings},
};

pub(super) fn emphasis_handler(
    handlers: &dyn Handlers,
    element: Element,
    marker: &str,
) -> Option<HandlerResult> {
    serialize_if_faithful!(handlers, element, 0);
    let content = handlers.walk_children(element.node).content;
    if content.is_empty() {
        return None;
    }
    // markitai: inside an element that already writes this emphasis
    // (`<b><strong>x</strong></b>`) the markers would double (`****x****`,
    // which reads as an empty span between literal asterisks), so the inner
    // element adds none and the outer one wraps the whole.
    if inside_same_emphasis(element.node, marker) {
        return Some(content.into());
    }
    // markitai: emphasis cannot span a blank line, so emphasis around
    // paragraphs (`<b>a<p>b</p></b>`, or an unclosed `<b>` the parser carries
    // into the next block) wraps each paragraph alone; written once around
    // them all, its markers were left as literal asterisks.
    if content.trim().contains("\n\n") && !content.contains("```") && !content.contains("~~~") {
        let wrapped: Vec<String> = content
            .split("\n\n")
            .map(|chunk| wrap_paragraph(chunk, marker))
            .collect();
        return Some(wrapped.join("\n\n").into());
    }
    // Note: this is whitespace, NOT document whitespace, per the
    // [Commonmark spec](https://spec.commonmark.org/0.31.2/#emphasis-and-strong-emphasis).
    let (content, leading_whitespace) = content.strip_leading_whitespace();
    let (content, trailing_whitespace) = content.strip_trailing_whitespace();
    if content.is_empty() {
        return None;
    }
    let content = concat_strings!(
        leading_whitespace.unwrap_or(""),
        marker,
        content,
        marker,
        trailing_whitespace.unwrap_or("")
    );
    Some(content.into())
}

/// markitai: one paragraph of an emphasis that spans several, between
/// `marker`s; a blank chunk, or one that starts a block of its own (a heading,
/// quote, list item, table row or indented code), is left as it is.
fn wrap_paragraph(chunk: &str, marker: &str) -> String {
    let (text, leading) = chunk.strip_leading_whitespace();
    let (text, trailing) = text.strip_trailing_whitespace();
    let marked = |at: usize| matches!(text.as_bytes().get(at), None | Some(b' ' | b'\t'));
    let ordered = {
        let digits = text.bytes().take_while(u8::is_ascii_digit).count();
        digits > 0 && matches!(text.as_bytes().get(digits), Some(b'.' | b')')) && marked(digits + 1)
    };
    let bullet = text.starts_with(['-', '+', '*']) && marked(1);
    let block = text.is_empty()
        || ordered
        || bullet
        || text.starts_with("---")
        || leading.is_some_and(|space| space.contains("    "))
        || text.starts_with(['#', '>', '|']);
    if block {
        return chunk.to_string();
    }
    concat_strings!(
        leading.unwrap_or(""),
        marker,
        text,
        marker,
        trailing.unwrap_or("")
    )
}

/// markitai: whether an ancestor in the same block writes the same emphasis
/// `marker` (`**` for `b` and `strong`, `*` for `i` and `em`). A block element
/// between them ends the search: the inner one then stands in its own
/// paragraph and keeps its markers.
fn inside_same_emphasis(node: crate::NodeRef<'_>, marker: &str) -> bool {
    let names: &[&str] = if marker == "**" {
        &["b", "strong"]
    } else {
        &["i", "em"]
    };
    let mut current = crate::node_util::get_parent_node(node);
    while let Some(parent) = current {
        let Some(tag) = crate::node_util::get_node_tag_name(parent) else {
            return false;
        };
        if names.contains(&tag) {
            return true;
        }
        if crate::dom_walker::is_block_element(tag) {
            return false;
        }
        current = crate::node_util::get_parent_node(parent);
    }
    false
}
