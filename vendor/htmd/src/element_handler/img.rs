use crate::{
    Element,
    element_handler::{HandlerResult, Handlers},
    serialize_if_faithful,
    text_util::{JoinOnStringIterator, TrimDocumentWhitespace, concat_strings},
};

pub(super) fn img_handler(handlers: &dyn Handlers, element: Element) -> Option<HandlerResult> {
    let mut link: Option<String> = None;
    let mut alt: Option<String> = None;
    let mut title: Option<String> = None;
    // markitai: scraper's attributes come in name order, so of `href` and
    // `src` (both the link) `src` wins whatever the source order.
    for (name, value) in element.attrs.iter() {
        let name = &name.local;
        if name == "href" {
            link = Some(value.to_string())
        } else if name == "src" {
            link = Some(value.to_string());
        } else if name == "alt" {
            alt = Some(value.to_string());
        } else if name == "title" {
            title = Some(value.to_string());
        } else {
            serialize_if_faithful!(handlers, element, 0);
        }
    }

    link.as_ref()?;

    // markitai: the lines of an alt text or title are one sentence wrapped in
    // the source, so they join with a space (upstream kept the line breaks,
    // which end an image's text at its first line once the output is
    // repaired as prose), and an empty title is no title: `![](a.png "")`
    // carries nothing.
    let process_alt_title = |text: String| {
        text.lines()
            .map(|line| line.trim_document_whitespace().replace('"', "\\\""))
            .filter(|line| !line.is_empty())
            .join(" ")
    };

    // Handle new lines in alt
    let alt = alt.map(process_alt_title);

    // Handle new lines in title
    let title = title
        .map(process_alt_title)
        .filter(|title| !title.is_empty());

    let link = link.map(|text| text.replace('(', "\\(").replace(')', "\\)"));

    let has_spaces_in_link = link.as_ref().is_some_and(|link| link.contains(' '));

    let md = concat_strings!(
        "![",
        alt.as_ref().unwrap_or(&String::new()),
        "](",
        if has_spaces_in_link { "<" } else { "" },
        link.as_ref().unwrap_or(&String::new()),
        if has_spaces_in_link { ">" } else { "" },
        title
            .as_ref()
            .map_or(String::new(), |t| concat_strings!(" \"", t, "\"")),
        ")"
    );
    Some(md.into())
}
