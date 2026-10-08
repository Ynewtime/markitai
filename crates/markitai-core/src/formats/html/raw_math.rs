//! TeX source left in a page's text as raw delimiters (`$x$`, `$$x$$`,
//! `\(x\)`, `\[x\]`), the markup MathJax and KaTeX auto-render read at view
//! time. The text is math, not prose: its backslashes, brackets and
//! underscores must reach the Markdown as written, so the cleaner writes it as
//! math (like a recognized wrapper) instead of letting the Markdown escaping
//! rewrite it.
//!
//! The rule is conservative, since a dollar sign usually is a price:
//!
//! - `$$ ... $$` and `\[ ... \]` are display math. The text between may span
//!   lines but not be blank. `\[ ... \]` also has to look like math (see
//!   [`looks_like_math`]), so a bracketed word (`\[options\]`) stays text.
//!   Whether a display expression is a block of its own or inline math (as
//!   the reference writes one inside a sentence) is the caller's to decide.
//! - `\( ... \)` and `$ ... $` are inline math on one line. The text between
//!   has to look like math. `$` follows Pandoc's rule: no space right after
//!   the opening dollar or right before the closing one, and no digit right
//!   after the closing one, so `$5 and $10` and `US$5-$10` are prices.
//! - An escaped dollar (`\$`) is a dollar sign and ends nothing.
//! - Nothing inside code is read (the caller passes only prose), and nothing
//!   spans two text nodes, so a delimiter in another element than its partner
//!   is not a pair.

/// A run of a text node.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum Piece<'a> {
    Text(&'a str),
    /// TeX source without its delimiters, trimmed.
    Math {
        latex: &'a str,
        display: bool,
    },
}

/// The longest inline expression read, in bytes: a dollar sign far from its
/// partner is not its partner.
const MAX_INLINE: usize = 400;

/// The pieces of `text` when it holds raw math; `None` when it holds none.
pub(super) fn split(text: &str) -> Option<Vec<Piece<'_>>> {
    let bytes = text.as_bytes();
    if !bytes.iter().any(|byte| matches!(byte, b'$' | b'\\')) {
        return None;
    }
    let mut pieces = Vec::new();
    let mut plain = 0;
    let mut at = 0;
    let (mut parens, mut brackets, mut dollar_pairs) = Default::default();
    while at < bytes.len() {
        let found = match bytes[at] {
            b'\\' => match bytes.get(at + 1) {
                Some(b'(') => inline(text, at + 2, "\\)", &mut parens),
                Some(b'[') => display(text, at + 2, "\\]", true, &mut brackets),
                // An escaped backslash or dollar sign is text.
                Some(b'\\' | b'$') => {
                    at += 2;
                    continue;
                }
                _ => None,
            },
            b'$' if bytes.get(at + 1) == Some(&b'$') => {
                let found = display(text, at + 2, "$$", false, &mut dollar_pairs);
                if found.is_none() {
                    // `$$` that closes nothing: both dollar signs are text.
                    at += 2;
                    continue;
                }
                found
            }
            b'$' => dollars(text, at),
            _ => None,
        };
        if let Some((end, latex, display)) = found {
            flush(text, &mut pieces, plain, at);
            pieces.push(Piece::Math { latex, display });
            at = end;
            plain = end;
        } else {
            at += 1;
        }
    }
    if pieces.is_empty() {
        return None;
    }
    flush(text, &mut pieces, plain, text.len());
    Some(pieces)
}

fn flush<'a>(text: &'a str, pieces: &mut Vec<Piece<'a>>, from: usize, to: usize) {
    if from < to {
        pieces.push(Piece::Text(&text[from..to]));
    }
}

/// Where a closer next occurs, as last searched. The scan only moves
/// forward, so a closer found stays the next one until the scan passes it,
/// and one not found is never found: each byte is searched once, not once per
/// opener before it.
#[derive(Default)]
struct Closer(Option<Option<usize>>);

impl Closer {
    fn find(&mut self, text: &str, start: usize, close: &str) -> Option<usize> {
        match self.0 {
            Some(Some(found)) if found >= start => Some(found),
            Some(None) => None,
            _ => *self
                .0
                .insert(text[start..].find(close).map(|length| start + length)),
        }
    }
}

/// A display expression whose content starts at `start` and which `close`
/// ends: the end of the closing delimiter, the trimmed content and `true`.
fn display<'a>(
    text: &'a str,
    start: usize,
    close: &str,
    needs_math: bool,
    closer: &mut Closer,
) -> Option<(usize, &'a str, bool)> {
    let length = closer.find(text, start, close)? - start;
    let latex = text[start..start + length].trim();
    (!latex.is_empty() && (!needs_math || looks_like_math(latex))).then_some((
        start + length + close.len(),
        latex,
        true,
    ))
}

/// An inline `\( ... \)` expression on one line.
fn inline<'a>(
    text: &'a str,
    start: usize,
    close: &str,
    closer: &mut Closer,
) -> Option<(usize, &'a str, bool)> {
    let length = closer.find(text, start, close)? - start;
    let latex = text[start..start + length].trim();
    (length <= MAX_INLINE
        && !latex.is_empty()
        && !latex.contains(['\n', '\r'])
        && looks_like_math(latex))
    .then_some((start + length + close.len(), latex, false))
}

/// A `$ ... $` expression opened at `open`, by Pandoc's rule.
fn dollars(text: &str, open: usize) -> Option<(usize, &str, bool)> {
    let bytes = text.as_bytes();
    let first = *bytes.get(open + 1)?;
    if first.is_ascii_whitespace() {
        return None;
    }
    let mut at = open + 1;
    while at < bytes.len() && at - open <= MAX_INLINE {
        match bytes[at] {
            b'\n' | b'\r' => return None,
            // An escaped character, `\$` included, ends nothing.
            b'\\' => at += 2,
            b'$' => {
                let before = bytes[at - 1];
                let after = bytes.get(at + 1).copied();
                // A dollar sign that cannot close ends the search: it opens
                // another expression, or is a price.
                if before.is_ascii_whitespace() || after.is_some_and(|byte| byte.is_ascii_digit()) {
                    return None;
                }
                let latex = &text[open + 1..at];
                return looks_like_math(latex).then_some((at + 1, latex, false));
            }
            _ => at += 1,
        }
    }
    None
}

/// Whether text between delimiters reads as math rather than as a word or
/// phrase: a short symbol (`x`, `n`, `ij`) or one with a TeX or operator
/// character in it.
fn looks_like_math(latex: &str) -> bool {
    latex.chars().count() <= 3
        || latex.contains([
            '\\', '^', '_', '{', '}', '=', '+', '<', '>', '|', '(', ')', '[', ']', ',', '\'',
        ])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn math(text: &str) -> Vec<(String, bool)> {
        split(text)
            .unwrap_or_default()
            .into_iter()
            .filter_map(|piece| match piece {
                Piece::Math { latex, display } => Some((latex.to_owned(), display)),
                Piece::Text(_) => None,
            })
            .collect()
    }

    #[test]
    fn the_four_delimiters_are_read_with_their_text_around_them() {
        assert_eq!(
            split(r"Say $\mathbf{x}_1$ of length \(n\) and:\[a^2=b\]"),
            Some(vec![
                Piece::Text("Say "),
                Piece::Math {
                    latex: r"\mathbf{x}_1",
                    display: false
                },
                Piece::Text(" of length "),
                Piece::Math {
                    latex: "n",
                    display: false
                },
                Piece::Text(" and:"),
                Piece::Math {
                    latex: "a^2=b",
                    display: true
                },
            ])
        );
        // Display math may span lines; its content is trimmed.
        assert_eq!(
            math("$$\n\\begin{aligned}\na &= [x_1, x_2] \\\\\n\\end{aligned}\n$$"),
            vec![(
                "\\begin{aligned}\na &= [x_1, x_2] \\\\\n\\end{aligned}".to_owned(),
                true
            )]
        );
    }

    #[test]
    fn prices_shell_variables_and_loose_dollars_are_text() {
        for text in [
            "$5 and $10",
            "Tickets cost $5, or $10 at the door, $20 for two.",
            "US$5-$10 and $5/$10",
            "pay $ 5 $ now",
            "$HOME/$USER and $PATH:$HOME",
            "a lone $ sign",
            r"\$x_1\$ is an escaped pair",
            "line one $x\ny$ line two",
            "a $$ that closes nothing",
            "an empty $$$$ pair",
            r"match \[options\] and \(some words\)",
            "a $b",
        ] {
            assert_eq!(split(text), None, "{text}");
        }
    }

    #[test]
    fn inline_dollars_need_a_non_space_inside_and_no_digit_after() {
        assert_eq!(
            math("$x$ and $y_i$"),
            vec![("x".into(), false), ("y_i".into(), false)]
        );
        // A digit right after the closing dollar makes it a price.
        assert_eq!(math("costs $a+b$1"), vec![]);
        // A space before the closing dollar, or after the opening one, too.
        assert_eq!(math("$a+b $ and $ a+b$"), vec![]);
        // An escaped dollar inside stays part of the expression.
        assert_eq!(math(r"$a\$b_1$"), vec![(r"a\$b_1".to_owned(), false)]);
    }

    #[test]
    fn display_and_inline_math_share_a_text_with_prose() {
        assert_eq!(
            split("before $$ x_1 $$ after $y_2$."),
            Some(vec![
                Piece::Text("before "),
                Piece::Math {
                    latex: "x_1",
                    display: true
                },
                Piece::Text(" after "),
                Piece::Math {
                    latex: "y_2",
                    display: false
                },
                Piece::Text("."),
            ])
        );
    }

    #[test]
    fn openers_without_closers_are_read_in_linear_time() {
        // Each opener searched the rest of the text for its closer, so a long
        // run of openers once took quadratic time: minutes for these texts.
        let (sender, received) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for opener in ["\\(", "\\["] {
                let text = opener.repeat(1 << 20);
                sender.send(split(&text).is_none()).unwrap();
            }
        });
        for _ in 0..2 {
            let none = received
                .recv_timeout(std::time::Duration::from_secs(20))
                .expect("the split finishes");
            assert!(none);
        }
    }

    #[test]
    fn non_ascii_text_around_math_is_cut_at_character_boundaries() {
        let pieces = split("设 $x_1$ 为长度 \\(n\\) 的序列").unwrap();
        assert_eq!(pieces.len(), 5);
        assert_eq!(pieces[0], Piece::Text("设 "));
        assert_eq!(pieces[4], Piece::Text(" 的序列"));
    }
}
