//! Word field-instruction parsing (DOC, DOCX, RTF), via a small
//! case-insensitive lexer with known switch arities.

use crate::model::{Inline, LinkTarget, inlines_are_empty};

/// Field accumulator: instruction text before the separator, result after.
#[derive(Default)]
pub struct FieldFrame {
    pub instr: String,
    pub in_result: bool,
    pub inlines: Vec<Inline>,
    /// markitai: the form field data the field's start carries, if any.
    pub form: Option<FormField>,
}

/// Finish a field: wrap the result in a link when the instruction is a
/// hyperlink, otherwise pass the result content through.
pub fn field_result(instr: &str, content: Vec<Inline>) -> Vec<Inline> {
    match hyperlink_target(instr) {
        Some(target) if !inlines_are_empty(&content) => {
            vec![Inline::Link { content, target }]
        }
        _ => content,
    }
}

/// markitai: the kind of a legacy form field (Word's `w:ffData`, RTF's
/// `\*\formfield` with `\fftype`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FormKind {
    Text,
    CheckBox,
    DropDown,
}

/// markitai: a legacy form field's own data. A check box and a drop-down
/// list keep their state here and not in the field result, which Word leaves
/// empty: the box is drawn and the chosen entry shown from this data, and
/// without it a filled-in form lost every answer but its text fields.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FormField {
    pub kind: Option<FormKind>,
    /// The current state: a check box's 0 or 1, a drop-down list's entry
    /// index; RTF writes 25 for "not set", which falls back to `default`.
    pub result: Option<i32>,
    pub default: Option<i32>,
    /// A drop-down list's entries, in order.
    pub entries: Vec<String>,
}

impl FormField {
    /// What the field shows: `☒` or `☐` for a check box, the chosen entry of
    /// a drop-down list (the first when none is chosen, as Word shows it).
    /// `None` for a text field (its result is its text) and when `instr`
    /// does not name the matching form field.
    pub fn shown(&self, instr: &str) -> Option<String> {
        let keyword = instr.split_whitespace().next()?;
        match self.kind? {
            FormKind::CheckBox if keyword.eq_ignore_ascii_case("FORMCHECKBOX") => {
                let state = self.result.filter(|r| matches!(r, 0 | 1)).or(self.default);
                Some(if state.unwrap_or(0) != 0 { "☒" } else { "☐" }.to_string())
            }
            FormKind::DropDown if keyword.eq_ignore_ascii_case("FORMDROPDOWN") => {
                let entry = |index: Option<i32>| {
                    index.and_then(|i| usize::try_from(i).ok()).filter(|&i| i < self.entries.len())
                };
                let index = entry(self.result).or(entry(self.default)).unwrap_or(0);
                self.entries.get(index).map(|e| e.trim().to_string()).filter(|e| !e.is_empty())
            }
            _ => None,
        }
    }
}

/// markitai: finish a field that may be a form field: a check box or a
/// drop-down list whose result shows nothing reads as its state (see
/// [`FormField::shown`]); everything else goes through [`field_result`].
pub fn form_field_result(
    instr: &str,
    form: Option<&FormField>,
    content: Vec<Inline>,
) -> Vec<Inline> {
    if inlines_are_empty(&content)
        && let Some(text) = form.and_then(|form| form.shown(instr))
    {
        let mut content = content;
        content.push(Inline::Text { text, style: crate::model::Style::PLAIN });
        return content;
    }
    field_result(instr, content)
}

#[derive(Debug, PartialEq)]
enum Token {
    Word(String),
    Switch(char),
}

/// Tokenize a field instruction: quoted strings (with `\"` and `\\`
/// escapes), backslash switches, bare words.
fn tokenize(instr: &str) -> Vec<Token> {
    let mut out = Vec::new();
    let mut chars = instr.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_whitespace() {
            chars.next();
        } else if c == '"' {
            chars.next();
            let mut word = String::new();
            while let Some(c) = chars.next() {
                match c {
                    '"' => break,
                    '\\' => match chars.next() {
                        Some(esc @ ('"' | '\\')) => word.push(esc),
                        Some(other) => {
                            word.push('\\');
                            word.push(other);
                        }
                        None => break,
                    },
                    c => word.push(c),
                }
            }
            out.push(Token::Word(word));
        } else if c == '\\' {
            chars.next();
            if let Some(&switch) = chars.peek() {
                chars.next();
                out.push(Token::Switch(switch.to_ascii_lowercase()));
            }
        } else {
            let mut word = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() {
                    break;
                }
                word.push(c);
                chars.next();
            }
            out.push(Token::Word(word));
        }
    }
    out
}

/// Switches of the HYPERLINK field that take an argument.
fn hyperlink_switch_takes_arg(switch: char) -> bool {
    matches!(switch, 'l' | 'o' | 't')
}

/// Interpret a HYPERLINK field instruction as a link target.
pub fn hyperlink_target(instr: &str) -> Option<LinkTarget> {
    let mut tokens = tokenize(instr).into_iter().peekable();
    match tokens.next() {
        Some(Token::Word(w)) if w.eq_ignore_ascii_case("HYPERLINK") => {}
        _ => return None,
    }
    let mut url: Option<String> = None;
    let mut anchor: Option<String> = None;
    while let Some(token) = tokens.next() {
        match token {
            Token::Word(w) => {
                if url.is_none() && !w.trim().is_empty() {
                    url = Some(w.trim().to_string());
                }
            }
            Token::Switch(s) => {
                // A switch's argument is the next token only when it is a
                // word; a following switch means the argument was omitted.
                let arg = if hyperlink_switch_takes_arg(s)
                    && matches!(tokens.peek(), Some(Token::Word(_)))
                {
                    match tokens.next() {
                        Some(Token::Word(w)) => Some(w),
                        _ => None,
                    }
                } else {
                    None
                };
                if s == 'l'
                    && let Some(a) = arg
                    && !a.trim().is_empty()
                {
                    anchor = Some(a.trim().to_string());
                }
            }
        }
    }
    match (url, anchor) {
        (Some(url), Some(frag)) => Some(classify(format!("{url}#{frag}"))),
        (Some(url), None) => Some(classify(url)),
        (None, Some(frag)) => Some(LinkTarget::Anchor(frag)),
        (None, None) => None,
    }
}

/// Classify an OPC relationship target as a link target. External-mode
/// targets are parsed like hyperlink field targets (scheme detection);
/// internal-mode targets stay package-relative.
pub fn classify_rel_target(external: bool, target: &str) -> LinkTarget {
    if external { classify(target.to_string()) } else { LinkTarget::Relative(target.to_string()) }
}

fn classify(url: String) -> LinkTarget {
    if let Some(anchor) = url.strip_prefix('#') {
        return LinkTarget::Anchor(anchor.to_string());
    }
    if crate::shared::uri::is_absolute_uri(&url) {
        LinkTarget::External(url)
    } else {
        LinkTarget::Relative(url)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoted_url() {
        assert_eq!(
            hyperlink_target(r#" HYPERLINK "https://e.com/a b" "#),
            Some(LinkTarget::External("https://e.com/a b".into()))
        );
    }

    #[test]
    fn case_insensitive_keyword() {
        assert_eq!(
            hyperlink_target(r#"hyperlink "https://e.com""#),
            Some(LinkTarget::External("https://e.com".into()))
        );
    }

    #[test]
    fn anchor_only() {
        assert_eq!(
            hyperlink_target(r#"HYPERLINK \l "sec2""#),
            Some(LinkTarget::Anchor("sec2".into()))
        );
    }

    #[test]
    fn url_plus_anchor() {
        assert_eq!(
            hyperlink_target(r#"HYPERLINK "https://e.com/p" \l "frag""#),
            Some(LinkTarget::External("https://e.com/p#frag".into()))
        );
    }

    #[test]
    fn switch_with_argument_not_mistaken_for_url() {
        assert_eq!(
            hyperlink_target(r#"HYPERLINK \o "tooltip text" "https://e.com""#),
            Some(LinkTarget::External("https://e.com".into()))
        );
    }

    #[test]
    fn escaped_quotes_in_target() {
        assert_eq!(
            hyperlink_target(r#"HYPERLINK "https://e.com/\"q\"""#),
            Some(LinkTarget::External(r#"https://e.com/"q""#.into()))
        );
    }

    #[test]
    fn relative_target() {
        assert_eq!(
            hyperlink_target(r#"HYPERLINK "docs/readme.docx""#),
            Some(LinkTarget::Relative("docs/readme.docx".into()))
        );
    }

    #[test]
    fn argless_switch_does_not_swallow_following_switch() {
        assert_eq!(
            hyperlink_target(r#"HYPERLINK "https://e.com/p" \o \l "frag""#),
            Some(LinkTarget::External("https://e.com/p#frag".into()))
        );
    }

    #[test]
    fn non_hyperlink_fields_ignored() {
        assert_eq!(hyperlink_target("PAGEREF _Toc123 \\h"), None);
    }
}
