// markitai: reads the scraper tree through `Handlers`.
use scraper::Node;

use crate::{
    Element,
    element_handler::{HandlerResult, Handlers},
    serialize_if_faithful,
    text_util::concat_strings,
};

pub(super) fn span_handler(handlers: &dyn Handlers, element: Element) -> Option<HandlerResult> {
    // See if this contains math: `<span class="math math-inline/display>text-only content</span>`.
    if element.attrs.len() == 1
        && let Some((name, value)) = element.attrs.iter().next()
        && *name.local == *"class"
        && let children = handlers.node_children(element.node)
        && children.len() == 1
        && let Node::Text(_) = children[0].value()
    {
        let contents = handlers.node_text(children[0]);
        if **value == *"math math-inline" {
            return Some(concat_strings!("$", contents.as_ref(), "$").into());
        }

        if **value == *"math math-display" {
            return Some(concat_strings!("$$", contents.as_ref(), "$$").into());
        }
    }

    // Always serialize as HTML if we're in faithful mode.
    serialize_if_faithful!(handlers, element, -1);

    // Otherwise, just return the contents.
    let content = handlers.walk_children(element.node).content;
    let content = content.trim_matches('\n');

    Some(content.into())
}
