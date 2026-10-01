//! Superscript and subscript runs (`w:vertAlign`).
//!
//! markitai: Markdown has no syntax for a raised or lowered run, and dropping
//! the formatting turns the exponent in "10⁻³" into the subtraction "10-3" and
//! the "2" of "H₂O" into a number beside the letter. Unicode has superscript
//! and subscript forms of the characters that matter (digits, signs,
//! parentheses and a few letters), and text written in them reads the same as
//! a plain text reader and as a Markdown renderer.

/// Which way a run is raised or lowered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Script {
    Superscript,
    Subscript,
}

impl Script {
    /// The script a `w:vertAlign` value names.
    pub fn from_value(value: &str) -> Option<Script> {
        match value {
            "superscript" => Some(Script::Superscript),
            "subscript" => Some(Script::Subscript),
            _ => None,
        }
    }

    /// `text` in the script's Unicode forms. `None` when some character has
    /// no such form, in which case the text is better left at the baseline
    /// than half converted ("1st" would become "1ˢt").
    pub fn convert(self, text: &str) -> Option<String> {
        let mut out = String::with_capacity(text.len() * 3);
        let mut converted = false;
        for c in text.chars() {
            if c.is_whitespace() {
                out.push(c);
                continue;
            }
            out.push(match self {
                Script::Superscript => superscript(c)?,
                Script::Subscript => subscript(c)?,
            });
            converted = true;
        }
        converted.then_some(out)
    }
}

fn superscript(c: char) -> Option<char> {
    Some(match c {
        '0' => '⁰',
        '1' => '¹',
        '2' => '²',
        '3' => '³',
        '4' => '⁴',
        '5' => '⁵',
        '6' => '⁶',
        '7' => '⁷',
        '8' => '⁸',
        '9' => '⁹',
        '+' => '⁺',
        '-' | '−' | '–' => '⁻',
        '=' => '⁼',
        '(' => '⁽',
        ')' => '⁾',
        'n' => 'ⁿ',
        'i' => 'ⁱ',
        _ => return None,
    })
}

fn subscript(c: char) -> Option<char> {
    Some(match c {
        '0' => '₀',
        '1' => '₁',
        '2' => '₂',
        '3' => '₃',
        '4' => '₄',
        '5' => '₅',
        '6' => '₆',
        '7' => '₇',
        '8' => '₈',
        '9' => '₉',
        '+' => '₊',
        '-' | '−' | '–' => '₋',
        '=' => '₌',
        '(' => '₍',
        ')' => '₎',
        'a' => 'ₐ',
        'e' => 'ₑ',
        'h' => 'ₕ',
        'i' => 'ᵢ',
        'j' => 'ⱼ',
        'k' => 'ₖ',
        'l' => 'ₗ',
        'm' => 'ₘ',
        'n' => 'ₙ',
        'o' => 'ₒ',
        'p' => 'ₚ',
        'r' => 'ᵣ',
        's' => 'ₛ',
        't' => 'ₜ',
        'u' => 'ᵤ',
        'v' => 'ᵥ',
        'x' => 'ₓ',
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digits_signs_and_parentheses_have_both_forms() {
        assert_eq!(Script::Superscript.convert("-3").as_deref(), Some("⁻³"));
        assert_eq!(Script::Superscript.convert("2+").as_deref(), Some("²⁺"));
        assert_eq!(Script::Subscript.convert("2").as_deref(), Some("₂"));
        assert_eq!(Script::Subscript.convert("(10)").as_deref(), Some("₍₁₀₎"));
    }

    #[test]
    fn letters_convert_only_where_unicode_has_them() {
        assert_eq!(Script::Superscript.convert("n").as_deref(), Some("ⁿ"));
        assert_eq!(Script::Subscript.convert("max").as_deref(), Some("ₘₐₓ"));
        // `st` and `eff` have no complete superscript or subscript form.
        assert_eq!(Script::Superscript.convert("st"), None);
        assert_eq!(Script::Subscript.convert("eff"), None);
    }

    #[test]
    fn whitespace_passes_through_and_a_blank_run_converts_to_nothing() {
        assert_eq!(Script::Superscript.convert("2 ").as_deref(), Some("² "));
        assert_eq!(Script::Superscript.convert(" "), None);
    }

    #[test]
    fn vertical_alignment_values_name_a_script() {
        assert_eq!(Script::from_value("superscript"), Some(Script::Superscript));
        assert_eq!(Script::from_value("subscript"), Some(Script::Subscript));
        assert_eq!(Script::from_value("baseline"), None);
    }
}
