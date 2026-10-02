//! Forms, footers and controls: which of them are page text.
//!
//! A page writes its interface in the same markup as its text. Controls
//! (`input`, `select`, `textarea`, `button`) are never text. A form is text
//! unless it is an entry box: a search, sign-in, newsletter or comment form,
//! known by its fields and names, with little prose of its own. ASP.NET pages
//! wrap the whole page in one form, and old Reddit wraps every post body in
//! one, so a form's prose is never dropped for the element alone. A footer is
//! the page's own (its address, copyright and links) unless it belongs to an
//! article, section, figure or quotation, whose footers carry a note, a source
//! or an attribution.

use super::Attribute;
use scraper::{ElementRef, Node};

/// An entry box says no more than this many words around its fields ...
const MAX_BOX_WORDS: usize = 20;
/// ... and a box that names itself (a search field, a password, an e-mail
/// address, `class="newsletter"`) no more than this many.
const MAX_NAMED_BOX_WORDS: usize = 60;

/// Whether an element and its content are left out of the page's text:
/// scripts, styles, frames, navigation, controls, a page's own footer and an
/// entry box.
pub(super) fn left_out(element: ElementRef<'_>) -> bool {
    match element.value().name() {
        "script" | "style" | "nav" | "button" | "input" | "select" | "textarea" | "iframe"
        | "object" | "embed" | "head" | "template" | "noscript" => true,
        "footer" | "form" => page_box(element),
        _ => false,
    }
}

/// Whether a `footer` is the page's own or a `form` is an entry box (see the
/// module documentation); any other element is not.
pub(super) fn page_box(element: ElementRef<'_>) -> bool {
    match element.value().name() {
        "footer" => page_footer(element),
        "form" => entry_form(element),
        _ => false,
    }
}

/// A footer that belongs to the page, not to a part of it: its nearest
/// sectioning ancestor is the body (or an aside, a navigation block, a
/// dialog, a table cell), or it says it is the page's (`role="contentinfo"`).
/// An article, section, figure or quotation that holds the page's `main`
/// region is a wrapper of the whole page, and its footer is the page's.
fn page_footer(footer: ElementRef<'_>) -> bool {
    if footer
        .value()
        .attribute("role")
        .is_some_and(|role| role.trim().eq_ignore_ascii_case("contentinfo"))
    {
        return true;
    }
    for ancestor in footer.ancestors().filter_map(ElementRef::wrap) {
        match ancestor.value().name() {
            "article" | "section" | "figure" | "blockquote" => {
                return ancestor
                    .descendants()
                    .filter_map(ElementRef::wrap)
                    .any(|node| {
                        node.value().name() == "main"
                            || node
                                .value()
                                .attribute("role")
                                .is_some_and(|role| role.trim().eq_ignore_ascii_case("main"))
                    });
            }
            "aside" | "nav" | "details" | "dialog" | "fieldset" | "td" | "th" | "body" => {
                return true;
            }
            _ => {}
        }
    }
    true
}

/// Whether a word of a form's class, id, name or action names an entry box:
/// a word that starts with one of the names (`searchform`, `comment-form`,
/// `wp-comments-post.php`, `mc-embedded-subscribe-form`).
fn names_entry_box(form: ElementRef<'_>) -> bool {
    const NAMES: [&str; 9] = [
        "search",
        "login",
        "signin",
        "signup",
        "register",
        "subscribe",
        "newsletter",
        "comment",
        "password",
    ];
    let value = form.value();
    ["class", "id", "name", "action"]
        .iter()
        .filter_map(|attribute| value.attribute(attribute))
        .flat_map(|value| value.split(|c: char| !c.is_ascii_alphanumeric()))
        .any(|word| {
            NAMES.iter().any(|name| {
                word.get(..name.len())
                    .is_some_and(|start| start.eq_ignore_ascii_case(name))
            })
        })
        || value
            .attribute("role")
            .is_some_and(|role| role.trim().eq_ignore_ascii_case("search"))
}

/// A search, sign-in, newsletter or comment form: it has visible fields and
/// few words of prose, or it names itself an entry box (by a search, password
/// or e-mail field, or by its names) and still says little. Hidden fields
/// (an ASP.NET view state, old Reddit's hidden editor) are not fields.
fn entry_form(form: ElementRef<'_>) -> bool {
    let mut named = names_entry_box(form);
    let mut fields = false;
    let mut words = 0usize;
    let mut stack = vec![form];
    while let Some(element) = stack.pop() {
        if element != form && super::is_hidden(element) {
            continue;
        }
        let value = element.value();
        match value.name() {
            "input" => {
                let kind = value.attribute("type").unwrap_or("text").trim();
                if kind.eq_ignore_ascii_case("hidden") {
                    continue;
                }
                fields = true;
                let search_name = ["name", "id"].iter().any(|attribute| {
                    value.attribute(attribute).is_some_and(|name| {
                        matches!(
                            name.to_ascii_lowercase().as_str(),
                            "q" | "s" | "search" | "query" | "keyword" | "keywords"
                        )
                    })
                });
                named |= search_name
                    || ["password", "search", "email"]
                        .iter()
                        .any(|named_kind| kind.eq_ignore_ascii_case(named_kind));
                continue;
            }
            "select" | "textarea" | "button" => {
                fields = true;
                continue;
            }
            "script" | "style" | "template" | "noscript" => continue,
            _ => {}
        }
        for child in element.children() {
            match child.value() {
                Node::Text(text) => {
                    words = words.saturating_add(text.split_whitespace().count());
                    if words > MAX_NAMED_BOX_WORDS {
                        return false;
                    }
                }
                Node::Element(_) => stack.extend(ElementRef::wrap(child)),
                _ => {}
            }
        }
    }
    (named && words <= MAX_NAMED_BOX_WORDS) || (fields && words <= MAX_BOX_WORDS)
}

/// For a checkbox that opens a list item (a task list: GitHub's
/// `<li><input type="checkbox" checked disabled> Done</li>`, also inside the
/// item's paragraph or label), whether it is checked. `None` for any other
/// input, and for a checkbox after the item's text.
pub(super) fn task_checkbox(input: ElementRef<'_>) -> Option<bool> {
    let value = input.value();
    if value.name() != "input"
        || !value
            .attribute("type")
            .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("checkbox"))
    {
        return None;
    }
    let item = input
        .ancestors()
        .filter_map(ElementRef::wrap)
        .find(|ancestor| !matches!(ancestor.value().name(), "p" | "label" | "span" | "div"))
        .filter(|ancestor| ancestor.value().name() == "li")?;
    // Nothing shows before the checkbox in the item.
    for node in item.descendants() {
        if node.id() == input.id() {
            return Some(value.attribute("checked").is_some());
        }
        match node.value() {
            Node::Text(text) if !text.trim().is_empty() => return None,
            Node::Element(element) if matches!(element.name(), "img" | "input" | "svg") => {
                return None;
            }
            _ => {}
        }
    }
    None
}

/// Whether an element is the attribution of the quotation it ends: a
/// `footer` of a `blockquote` (`<footer>— Name</footer>`) or Bootstrap's
/// `blockquote-footer` line.
pub(super) fn quote_attribution(element: ElementRef<'_>) -> bool {
    let value = element.value();
    let marked = value.name() == "footer"
        || value
            .class_names()
            .any(|class| class.eq_ignore_ascii_case("blockquote-footer"));
    marked
        && element
            .ancestors()
            .filter_map(ElementRef::wrap)
            .find(|ancestor| {
                matches!(
                    ancestor.value().name(),
                    "blockquote" | "article" | "section" | "figure" | "aside" | "body"
                )
            })
            .is_some_and(|ancestor| ancestor.value().name() == "blockquote")
}

/// Whether an attribution's text opens with a dash already (`— Name`,
/// `-- Name`, `~ Name`).
pub(super) fn opens_with_dash(text: &str) -> bool {
    text.trim_start()
        .starts_with(['—', '–', '―', '-', '~', '⸺'])
}

#[cfg(test)]
mod tests {
    use super::*;
    use scraper::{Html, Selector};

    fn first<'a>(document: &'a Html, query: &str) -> ElementRef<'a> {
        document
            .select(&Selector::parse(query).unwrap())
            .next()
            .unwrap()
    }

    #[test]
    fn a_page_footer_is_left_out_and_a_part_s_footer_is_kept() {
        let document = Html::parse_document(
            r#"<body><article><p>Body.</p><footer id="a">Article note</footer>
            <section><footer id="s">Section note</footer></section>
            <figure><img src="x.png"><footer id="f">Source</footer></figure>
            <blockquote><p>Quote.</p><footer id="q">Name</footer></blockquote>
            <footer id="c" role="contentinfo">Page</footer></article>
            <aside><footer id="aside">Aside</footer></aside>
            <main><footer id="m">Main footer</footer></main>
            <footer id="page">Copyright</footer>
            <section id="wrap"><main><p>x</p></main><footer id="w">Site</footer></section></body>"#,
        );
        for kept in ["#a", "#s", "#f", "#q"] {
            assert!(!left_out(first(&document, kept)), "{kept}");
        }
        for dropped in ["#c", "#aside", "#m", "#page", "#w"] {
            assert!(left_out(first(&document, dropped)), "{dropped}");
        }
    }

    #[test]
    fn entry_boxes_are_left_out_and_forms_with_prose_are_kept() {
        let prose = "The article says a great deal in this sentence and the next. ".repeat(6);
        let document = Html::parse_document(&format!(
            r#"<body>
            <form id="search"><input type="search" name="q"><button>Go</button></form>
            <form id="login"><label>User <input name="u"></label><label>Password <input type="password"></label></form>
            <form id="news" class="newsletter-form"><p>Get the latest posts delivered to your inbox every week.</p><input type="email"><button>Subscribe</button></form>
            <form id="fields"><label>Name <input type="text"></label><select><option>One</option></select><textarea>Area</textarea></form>
            <form id="aspnetForm"><input type="hidden" name="__VIEWSTATE"><div><p>{prose}</p><input type="submit" value="Go"></div></form>
            <form id="usertext" class="usertext"><input type="hidden" name="thing_id"><div class="md"><p>A short comment.</p></div><div style="display: none"><textarea>A short comment.</textarea><button>save</button></div></form>
            <form id="plain"><h2>Inside</h2><p>Short.</p></form>
            <form id="searchpage" role="search"><input type="search"><p>{prose}</p></form>
            </body>"#
        ));
        for dropped in ["#search", "#login", "#news", "#fields"] {
            assert!(left_out(first(&document, dropped)), "{dropped}");
        }
        for kept in ["#aspnetForm", "#usertext", "#plain", "#searchpage"] {
            assert!(!left_out(first(&document, kept)), "{kept}");
        }
        for control in ["input", "select", "textarea", "button"] {
            assert!(left_out(first(&document, control)), "{control}");
        }
    }

    #[test]
    fn a_checkbox_opening_a_list_item_is_a_task() {
        let document = Html::parse_document(
            r#"<ul><li id="done"><input type="checkbox" checked disabled> Done</li>
            <li id="open"><p><input type="checkbox"> Open</p></li>
            <li id="late">Text <input type="checkbox"></li>
            <li id="radio"><input type="radio"> Choice</li></ul>
            <p><input type="checkbox" id="toggle"> Menu</p>"#,
        );
        let input = |item: &str| first(&document, &format!("{item} input"));
        assert_eq!(task_checkbox(input("#done")), Some(true));
        assert_eq!(task_checkbox(input("#open")), Some(false));
        assert_eq!(task_checkbox(input("#late")), None);
        assert_eq!(task_checkbox(input("#radio")), None);
        assert_eq!(task_checkbox(first(&document, "#toggle")), None);
    }

    #[test]
    fn a_quotation_s_footer_is_its_attribution() {
        let document = Html::parse_document(
            r#"<blockquote><p>Quote.</p><footer id="f">— <cite>Name</cite></footer><p class="blockquote-footer" id="b">Someone</p></blockquote>
            <article><footer id="a">Note</footer></article>"#,
        );
        assert!(quote_attribution(first(&document, "#f")));
        assert!(quote_attribution(first(&document, "#b")));
        assert!(!quote_attribution(first(&document, "#a")));
        assert!(opens_with_dash("  — Name") && opens_with_dash("-- Name"));
        assert!(!opens_with_dash("Name"));
    }
}
