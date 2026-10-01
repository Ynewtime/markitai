//! Callouts and alerts as Obsidian-style blockquotes, as the reference's web
//! extraction renders them: Obsidian callouts, GitHub alerts, Bootstrap alerts,
//! callout asides and Hugo/Docsy admonitions.
use super::Attribute;
use scraper::{ElementRef, Node};

const ADMONITION_TYPES: [&str; 14] = [
    "info",
    "warning",
    "note",
    "tip",
    "danger",
    "caution",
    "important",
    "abstract",
    "success",
    "question",
    "failure",
    "bug",
    "example",
    "quote",
];

pub(super) struct Callout<'a> {
    pub kind: String,
    pub fold: &'static str,
    pub title: String,
    /// Whose children form the body; its own hidden state is the fold state.
    pub content: ElementRef<'a>,
    /// A title element inside the body that the marker replaces.
    pub omit: Option<ElementRef<'a>>,
}

fn classes<'a>(element: ElementRef<'a>) -> impl Iterator<Item = &'a str> {
    element
        .value()
        .attribute("class")
        .unwrap_or("")
        .split_whitespace()
}

fn has_class(element: ElementRef<'_>, class: &str) -> bool {
    classes(element).any(|value| value == class)
}

/// `prefix` followed by word characters at the start of a class, as the
/// reference's `re.match(r"prefix(\w+)")` reads it.
fn class_suffix<'a>(class: &'a str, prefix: &str, ignore_case: bool) -> Option<&'a str> {
    let head = class.get(..prefix.len())?;
    let matches = if ignore_case {
        head.eq_ignore_ascii_case(prefix)
    } else {
        head == prefix
    };
    if !matches {
        return None;
    }
    let rest = &class[prefix.len()..];
    let end = rest
        .char_indices()
        .find(|(_, ch)| !(ch.is_alphanumeric() || *ch == '_'))
        .map_or(rest.len(), |(index, _)| index);
    (end > 0).then(|| &rest[..end])
}

fn descendants_with_class<'a>(
    element: ElementRef<'a>,
    wanted: &'a [&'a str],
) -> impl Iterator<Item = ElementRef<'a>> + 'a {
    element
        .descendants()
        .skip(1)
        .filter_map(ElementRef::wrap)
        .filter(move |node| wanted.iter().any(|class| has_class(*node, class)))
}

/// Visible text pieces, each stripped, joined by `separator`.
fn text(element: ElementRef<'_>, separator: &str) -> String {
    element
        .descendants()
        .filter_map(|node| match node.value() {
            Node::Text(text) => Some(text.trim()),
            _ => None,
        })
        .filter(|piece| !piece.is_empty())
        .collect::<Vec<_>>()
        .join(separator)
}

fn obsidian(element: ElementRef<'_>) -> Option<Callout<'_>> {
    if element.value().name() != "div" || !has_class(element, "callout") {
        return None;
    }
    let kind = element.value().attribute("data-callout")?;
    let content = element
        .children()
        .filter_map(ElementRef::wrap)
        .find(|child| child.value().name() == "div" && has_class(*child, "callout-content"))?;
    let kind = if !kind.is_empty()
        && kind
            .chars()
            .all(|ch| ch.is_alphanumeric() || ch == '_' || ch == '-')
    {
        kind.to_owned()
    } else {
        "note".to_owned()
    };
    let title = descendants_with_class(element, &["callout-title-inner"])
        .next()
        .map(|title| text(title, " "))
        .unwrap_or_default();
    let fold = match element.value().attribute("data-callout-fold") {
        Some("-") => "-",
        Some("+") => "+",
        _ if has_class(element, "is-collapsed") => "-",
        _ if has_class(element, "is-collapsible") => "+",
        _ => "",
    };
    Some(Callout {
        kind,
        fold,
        title,
        content,
        omit: None,
    })
}

fn github(element: ElementRef<'_>) -> Option<Callout<'_>> {
    if !has_class(element, "markdown-alert") {
        return None;
    }
    let kind = classes(element)
        .find_map(|class| class_suffix(class, "markdown-alert-", false))
        .map_or("note".to_owned(), str::to_lowercase);
    Some(Callout {
        kind,
        fold: "",
        title: String::new(),
        content: element,
        omit: descendants_with_class(element, &["markdown-alert-title"]).next(),
    })
}

fn bootstrap(element: ElementRef<'_>) -> Option<Callout<'_>> {
    if !has_class(element, "alert") {
        return None;
    }
    let kind = classes(element)
        .find_map(|class| {
            class_suffix(class, "alert-", false).filter(|kind| *kind != "dismissible")
        })
        .map_or("note".to_owned(), str::to_lowercase);
    Some(Callout {
        kind,
        fold: "",
        title: String::new(),
        content: element,
        omit: descendants_with_class(element, &["alert-heading", "alert-title"]).next(),
    })
}

fn aside(element: ElementRef<'_>) -> Option<Callout<'_>> {
    if element.value().name() != "aside"
        || !element
            .value()
            .attribute("class")
            .is_some_and(|value| value.to_lowercase().contains("callout"))
    {
        return None;
    }
    let kind = classes(element)
        .find_map(|class| class_suffix(class, "callout-", true))
        .map_or("note".to_owned(), str::to_lowercase);
    Some(Callout {
        kind,
        fold: "",
        title: String::new(),
        content: element,
        omit: None,
    })
}

fn admonition(element: ElementRef<'_>) -> Option<Callout<'_>> {
    if !has_class(element, "admonition") || element.value().attribute("data-callout").is_some() {
        return None;
    }
    let kind = classes(element)
        .find(|class| ADMONITION_TYPES.contains(class))
        .unwrap_or("note")
        .to_owned();
    let title = descendants_with_class(element, &["admonition-title"]).next();
    let content = descendants_with_class(element, &["admonition-content"])
        .next()
        .or_else(|| descendants_with_class(element, &["details-content"]).next())
        .unwrap_or(element);
    Some(Callout {
        kind,
        fold: "",
        // Source whitespace is kept: the reference's piece-joining would read
        // `Helpful <em>tip</em>` as `Helpfultip`.
        title: title
            .map(|title| {
                title
                    .text()
                    .collect::<String>()
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default(),
        content,
        omit: title,
    })
}

pub(super) fn detect(element: ElementRef<'_>) -> Option<Callout<'_>> {
    obsidian(element)
        .or_else(|| github(element))
        .or_else(|| bootstrap(element))
        .or_else(|| aside(element))
        .or_else(|| admonition(element))
}

/// Whether `element` is the title a surrounding callout's marker replaces.
pub(super) fn replaced_title(element: ElementRef<'_>) -> bool {
    [
        "markdown-alert-title",
        "alert-heading",
        "alert-title",
        "admonition-title",
    ]
    .into_iter()
    .any(|class| has_class(element, class))
        && element
            .ancestors()
            .filter_map(ElementRef::wrap)
            .find_map(detect)
            .is_some_and(|callout| callout.omit == Some(element))
}

/// The marker line: `[!type]fold Title`, the title defaulting to the
/// capitalized type.
pub(super) fn marker_title(callout: &Callout<'_>) -> String {
    if !callout.title.is_empty() {
        return callout.title.clone();
    }
    let mut chars = callout.kind.chars();
    chars.next().map_or(String::new(), |first| {
        first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect()
    })
}
