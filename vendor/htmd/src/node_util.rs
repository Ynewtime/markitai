// markitai: these helpers read the scraper tree (`NodeRef`) instead of
// markup5ever_rcdom's `Rc<Node>`; children come from the conversion's
// `Handlers`, which knows the adjacent elements merged while walking.
use scraper::Node;

use crate::{NodeRef, element_handler::Handlers};

pub(crate) fn get_node_tag_name<'a>(node: NodeRef<'a>) -> Option<&'a str> {
    match node.value() {
        // markitai: a fragment root stands where markup5ever_rcdom put a
        // document node (a template's contents, a parsed fragment).
        Node::Document | Node::Fragment => Some("html"),
        Node::Element(element) => Some(element.name()),
        _ => None,
    }
}

pub(crate) fn get_parent_node(node: NodeRef<'_>) -> Option<NodeRef<'_>> {
    node.parent()
}

// Check to see if node's parent's tag name matches the provided string.
pub(crate) fn parent_tag_name_equals(node: NodeRef<'_>, tag_names: &[&str]) -> bool {
    if let Some(parent) = get_parent_node(node)
        && let Some(actual_tag_name) = get_node_tag_name(parent)
        && tag_names.contains(&actual_tag_name)
    {
        true
    } else {
        false
    }
}

pub(crate) fn get_node_children<'a>(
    handlers: &dyn Handlers,
    node: NodeRef<'a>,
) -> Vec<NodeRef<'a>> {
    handlers.node_children(node)
}
