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
