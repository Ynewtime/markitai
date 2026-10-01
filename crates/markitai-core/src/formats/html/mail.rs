//! Web mail threads saved from a browser. A mail client folds each message's
//! quoted history away behind a control ("Show trimmed content"): the earlier
//! messages it repeats stand on the page as messages of their own. Only a
//! message body of the client (Gmail's `a3s`) is read this way; an article or
//! a mail fragment quoting a reply keeps it.

use super::has_class as class;
use scraper::ElementRef;

/// Gmail's message body.
const BODY: &str = "a3s";

/// Quoted history inside a message body, or the control that unfolds it:
/// Gmail's quote (`gmail_quote`) and its "On … wrote:" line (`gmail_attr`), a
/// quote right after that line, Apple Mail's quote (`blockquote type=cite`),
/// Gmail's trimmed content (`adL`) and a control (`role=button`).
fn quoted(element: ElementRef<'_>) -> bool {
    let value = element.value();
    class(element, "gmail_quote")
        || class(element, "gmail_attr")
        || class(element, "adL")
        || value.attr("role") == Some("button")
        || (value.name() == "blockquote"
            && (value
                .attr("type")
                .is_some_and(|kind| kind.eq_ignore_ascii_case("cite"))
                || element
                    .prev_siblings()
                    .find_map(ElementRef::wrap)
                    .is_some_and(|previous| class(previous, "gmail_attr"))))
}

/// Outlook's header of the message replied to (`divRplyFwdMsg`, with Gmail's
/// `m_…` prefix in a Gmail body): that message follows it.
fn reply_header(element: ElementRef<'_>) -> bool {
    element
        .value()
        .attr("id")
        .is_some_and(|id| id.ends_with("divRplyFwdMsg"))
}

/// The quoted history of every web mail message body under `root`.
pub(super) fn quoted_history(root: ElementRef<'_>) -> Vec<ElementRef<'_>> {
    let mut found = Vec::new();
    let mut stack = vec![root];
    while let Some(element) = stack.pop() {
        if !class(element, BODY) {
            stack.extend(element.child_elements());
            continue;
        }
        let mut inside: Vec<_> = element.child_elements().collect();
        while let Some(element) = inside.pop() {
            if quoted(element) {
                found.push(element);
            } else if reply_header(element) {
                // The rule Outlook draws above the header, the header and the
                // earlier message after it.
                found.extend(
                    element
                        .prev_siblings()
                        .find_map(ElementRef::wrap)
                        .filter(|rule| rule.value().name() == "hr"),
                );
                found.push(element);
                found.extend(element.next_siblings().filter_map(ElementRef::wrap));
            } else {
                inside.extend(element.child_elements());
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use scraper::Html;

    /// The text of each element left out, sorted.
    fn quoted_text(html: &str) -> Vec<String> {
        let document = Html::parse_document(html);
        let mut found: Vec<String> = quoted_history(document.root_element())
            .into_iter()
            .map(super::super::plain)
            .collect();
        found.sort();
        found
    }

    #[test]
    fn a_gmail_message_folds_its_quoted_history_and_controls() {
        let html = r#"<div role="main"><div class="adn" data-message-id="1"><div class="a3s aiL">
            <div dir="ltr">Thanks, see you then.</div>
            <div class="gmail_quote"><div class="gmail_attr">On Wed, Jane wrote:</div><blockquote class="gmail_quote">Earlier message.</blockquote></div>
            <div class="gmail_attr">On Thu, Alex wrote:</div><blockquote style="margin:0">Styled quote.</blockquote>
            <blockquote type="cite">Apple quote.</blockquote>
            <div class="yj6qo"><div role="button" aria-label="Show trimmed content"></div></div>
            <div class="adL"><div>-- Signature</div></div></div></div></div>"#;
        assert_eq!(
            quoted_text(html),
            [
                "",
                "-- Signature",
                "Apple quote.",
                "On Thu, Alex wrote:",
                "On Wed, Jane wrote:Earlier message.",
                "Styled quote.",
            ]
        );
    }

    #[test]
    fn an_outlook_reply_header_folds_the_message_after_it() {
        let html = r#"<div class="a3s"><div>My answer.</div><div id="m_123appendonsend"></div><hr><div id="m_123divRplyFwdMsg">From: Jane Sent: Monday</div><div>The original message.</div></div>"#;
        assert_eq!(
            quoted_text(html),
            ["", "From: Jane Sent: Monday", "The original message."]
        );
    }

    #[test]
    fn quotes_outside_a_mail_body_stay() {
        // An article about email, a quoted reply in a forum post and a quote
        // of a mail client's markup in an ordinary page are content.
        for html in [
            r#"<article><p>Gmail marks quotes with gmail_quote.</p><div class="gmail_quote"><p>On Wed, Jane wrote:</p><blockquote>A quote.</blockquote></div></article>"#,
            r#"<div class="post"><blockquote type="cite">A cited reply.</blockquote><p>My answer.</p></div>"#,
            r#"<main><div role="button">Expand</div><div id="divRplyFwdMsg">From: Jane</div><p>Text.</p></main>"#,
        ] {
            assert!(quoted_text(html).is_empty(), "{html}");
        }
        // The message itself, and a quote that is not marked as history.
        let html = r#"<div class="a3s"><div>Reply text.</div><blockquote><p>A quoted poem.</p></blockquote></div>"#;
        assert!(quoted_text(html).is_empty());
    }
}
