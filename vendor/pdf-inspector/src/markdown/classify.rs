//! Line classification: captions, lists, code detection.

/// Check if text is a figure/table caption or source citation
pub(crate) fn is_caption_line(text: &str) -> bool {
    let trimmed = text.trim();

    // Caption prefixes that always match (always followed by identifiers)
    let always_prefixes = [
        "Figura ",
        "Fig. ",
        "Fig ",
        "Tabela ",
        "Source:",
        "Fonte:",
        "Source ",
        "Fonte ",
        "Note:",
        "Nota:",
        "Chart ",
        "Gráfico ",
        "Graph ",
        "Diagram ",
        "Image ",
        "Imagem ",
        "Photo ",
        "Foto ",
    ];
    for prefix in &always_prefixes {
        if trimmed.starts_with(prefix) {
            return true;
        }
    }

    // "Figure" and "Table" need a digit/reference after them to distinguish
    // captions ("Table 1", "Figure 3.2") from headings ("Table of Contents")
    for prefix in ["Figure ", "Table "] {
        if let Some(rest) = trimmed.strip_prefix(prefix) {
            if rest
                .trim_start()
                .starts_with(|c: char| c.is_ascii_digit() || c == '(' || c == '#')
            {
                return true;
            }
        }
    }

    // Check case-insensitive patterns — require digit or punctuation after
    // prefix to avoid matching "Table of Contents" or "Figure drawing" etc.
    let lower = trimmed.to_lowercase();
    for pfx in ["figure ", "table "] {
        if let Some(rest) = lower.strip_prefix(pfx) {
            if rest
                .trim_start()
                .starts_with(|c: char| c.is_ascii_digit() || c == '(' || c == '#')
            {
                return true;
            }
        }
    }
    if lower.starts_with("source:") {
        return true;
    }

    false
}

/// Check if text starts with an unambiguous bullet marker (●, •, ○, ◦).
///
/// Narrower than [`is_list_item`]: it excludes numbered/lettered patterns
/// like `1.` or `a)`, which legitimately appear as section headings in many
/// documents. Used by the heading classifier to reject bullet lines without
/// also demoting numbered headings.
/// Whether a line starts with a check box or check mark and a space
/// (markitai): `✅ True HEPA filter`, `☐ Not yet`, a checklist's item. The
/// mark is the item's state, so it stays in the item's text.
pub(crate) fn starts_with_check_mark(text: &str) -> bool {
    let mut chars = text.trim_start().chars();
    let Some(mark) = chars.next() else {
        return false;
    };
    if !matches!(
        mark,
        '\u{2705}'
            | '\u{2714}'
            | '\u{2713}'
            | '\u{2611}'
            | '\u{2610}'
            | '\u{2612}'
            | '\u{2717}'
            | '\u{2718}'
            | '\u{274C}'
            | '\u{274E}'
            | '\u{2B1C}'
            | '\u{25A2}'
    ) {
        return false;
    }
    match chars.next() {
        Some('\u{FE0E}' | '\u{FE0F}') => chars.next().is_some_and(char::is_whitespace),
        Some(next) => next.is_whitespace(),
        None => false,
    }
}

/// markitai: the dingbats a symbol font's bullets map to (`❖` is Word's
/// Wingdings diamond), read as list markers like `•`.
const DINGBAT_BULLETS: [char; 6] = ['❖', '◆', '◇', '■', '▪', '➢'];

/// markitai: whether `text` starts with a dingbat bullet and a space.
fn starts_with_dingbat_bullet(text: &str) -> bool {
    let mut chars = text.chars();
    chars.next().is_some_and(|c| DINGBAT_BULLETS.contains(&c))
        && chars.next().is_some_and(char::is_whitespace)
}

pub(crate) fn starts_with_bullet_marker(text: &str) -> bool {
    let trimmed = text.trim_start();
    starts_with_check_mark(trimmed)
        || starts_with_dingbat_bullet(trimmed)
        || trimmed.starts_with("• ")
        || trimmed.starts_with("● ")
        || trimmed.starts_with("○ ")
        || trimmed.starts_with("◦ ")
        || trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
}

/// Check if text looks like a list item
pub(crate) fn is_list_item(text: &str) -> bool {
    let trimmed = text.trim_start();

    // Bullet patterns
    if starts_with_check_mark(trimmed)
        || starts_with_dingbat_bullet(trimmed)
        || trimmed.starts_with("• ")
        || trimmed.starts_with("- ")
        || trimmed.starts_with("* ")
        || trimmed.starts_with("○ ")
        || trimmed.starts_with("● ")
        || trimmed.starts_with("◦ ")
    {
        return true;
    }

    // Numbered list patterns: "1.", "1)", "(1)", "a.", "a)"
    let first_chars: String = trimmed.chars().take(5).collect();
    if first_chars.contains(|c: char| c.is_ascii_digit()) {
        // Check for "1.", "1)", "10."
        if let Some(idx) = first_chars.find(['.', ')']) {
            let prefix = &first_chars[..idx];
            if prefix.chars().all(|c| c.is_ascii_digit()) {
                return true;
            }
        }
    }

    // markitai: CJK enumerators, "1．", "1）", "1、", "（1）": one to three
    // digits and a full-width stop, closing parenthesis or enumeration comma.
    if full_width_enumerator(trimmed).is_some() {
        return true;
    }

    // Letter list: "a.", "a)", "(a)"
    let mut chars = trimmed.chars();
    if let (Some(first), Some(second)) = (chars.next(), chars.next()) {
        if first.is_ascii_alphabetic() && (second == '.' || second == ')') {
            return true;
        }
        if first == '(' && chars.next() == Some(')') {
            return true;
        }
    }

    false
}

/// markitai: the length of a CJK enumerator opening `text` ("1．", "12）",
/// "3、", "（1）") followed by the item's text, which Markdown does not read
/// as a list marker.
pub(crate) fn full_width_enumerator(text: &str) -> Option<usize> {
    let (open, rest) = match text.strip_prefix('（') {
        Some(rest) => (true, rest),
        None => (false, text),
    };
    let digits = rest.chars().take_while(char::is_ascii_digit).count();
    if !(1..=3).contains(&digits) {
        return None;
    }
    let mut after = rest[digits..].chars();
    let close = after.next()?;
    let closes = if open {
        close == '）'
    } else {
        matches!(close, '．' | '）' | '、')
    };
    let body = after.as_str().trim_start();
    // "1．5" is a number with a full-width decimal point.
    let decimal = close == '．' && body.starts_with(|c: char| c.is_ascii_digit());
    (closes && !body.is_empty() && !decimal).then(|| text.len() - after.as_str().len())
}

/// markitai: the number of words a line holds: its words set
/// apart by spaces, a run of CJK text counting a word for every three
/// characters (CJK text sets no spaces between words, and a 70-character
/// sentence is one run).
pub(crate) fn text_words(text: &str) -> usize {
    text.split_whitespace()
        .map(|word| {
            let cjk = word
                .chars()
                .filter(|&c| crate::text_utils::is_cjk_char(c))
                .count();
            cjk.div_ceil(3).max(1)
        })
        .sum()
}

/// markitai: a line that reads as a sentence, not a heading: it ends with,
/// or holds within, a CJK sentence end (`。`, `！`, `？`, `；`), ends with a
/// CJK comma, the sentence running on, or runs on past a comma for a dozen
/// characters or more.
pub(crate) fn reads_as_sentence(text: &str) -> bool {
    let text = text.trim();
    let ends = |c: char| matches!(c, '。' | '！' | '？' | '；');
    text.ends_with(|c: char| ends(c) || matches!(c, '，' | '、'))
        || text
            .char_indices()
            .any(|(at, c)| ends(c) && !text[at + c.len_utf8()..].trim().is_empty())
        || (text.contains('，') && text.chars().count() >= 12)
}

/// markitai: a labelled field, `label：value`, rather than a heading: a
/// short label (up to twelve characters, no sentence inside) before a
/// full-width colon, and a value after it (`报名人：示例学员`, `班级类型：周末班`).
pub(crate) fn reads_as_field(text: &str) -> bool {
    let Some((label, value)) = text.trim().split_once('：') else {
        return false;
    };
    let label = label.trim();
    !label.is_empty()
        && label.chars().count() <= 12
        && !label.contains(['。', '，', '；'])
        && !value.trim().is_empty()
}

/// Format list item to markdown
pub(crate) fn format_list_item(text: &str) -> String {
    let trimmed = text.trim_start();

    // markitai: a CJK enumerator is kept as the item's text, after a
    // Markdown marker (Markdown reads no "1．" or "1）" as one).
    if full_width_enumerator(trimmed).is_some() {
        return format!("- {trimmed}");
    }

    // markitai: a check mark is the item's state and stays in its text.
    if starts_with_check_mark(trimmed) {
        return format!("- {trimmed}");
    }

    // Convert various bullet styles to markdown
    // Note: bullet characters like • are multi-byte in UTF-8, use char indices
    if starts_with_dingbat_bullet(trimmed) {
        let mut chars = trimmed.chars();
        chars.next();
        return format!("- {}", chars.as_str().trim_start());
    }
    for bullet in &['•', '○', '●', '◦'] {
        if let Some(rest) = trimmed.strip_prefix(*bullet) {
            return format!("- {}", rest.trim_start());
        }
        // Bullet inside a leading style run (e.g. "**● Label:** rest" or
        // "<u>● Label</u>"). The run wraps both the marker and the following
        // label because both carry the style in the PDF. The marker must move
        // outside the wrapper so markdown still sees a list item.
        for wrapper in ["**", "*", "<u>"] {
            if let Some(after_open) = trimmed.strip_prefix(wrapper) {
                if let Some(rest) = after_open.strip_prefix(*bullet) {
                    return format!("- {}{}", wrapper, rest.trim_start());
                }
            }
        }
    }

    if trimmed.starts_with("- ") || trimmed.starts_with("* ") {
        return trimmed.to_string();
    }

    // Keep numbered lists as-is (markdown supports them)
    trimmed.to_string()
}

/// Check if text looks like code
pub(crate) fn is_code_like(text: &str) -> bool {
    let trimmed = text.trim();

    // Code patterns
    let code_patterns = [
        // Language keywords
        "import ",
        "export ",
        "from ",
        "const ",
        "let ",
        "var ",
        "function ",
        "class ",
        "def ",
        "pub fn ",
        "fn ",
        "async fn ",
        "impl ",
        // Syntax patterns
        "=> ",
        "-> ",
        ":: ",
        ":= ",
    ];

    for pattern in &code_patterns {
        if trimmed.starts_with(pattern) {
            return true;
        }
    }

    // Check for code-like syntax
    let special_chars: usize = trimmed
        .chars()
        .filter(|c| matches!(c, '{' | '}' | '(' | ')' | '[' | ']' | ';' | '=' | '<' | '>'))
        .count();

    if special_chars >= 3 && trimmed.len() < 200 {
        return true;
    }

    // Ends with semicolon or braces
    if trimmed.ends_with(';') || trimmed.ends_with('{') || trimmed.ends_with('}') {
        return true;
    }

    false
}

/// True when a line's text is essentially all monospace (≥90% by character
/// count). Code lines are wholly monospace; anything less is prose carrying
/// mono-styled fragments — a URL sidebar, or a sentence quoting an inline
/// code literal — and fencing it would split paragraphs mid-sentence.
/// Any-item matching was safe only while items carried opaque font resource
/// names that never matched the monospace patterns; items now carry real
/// family names.
pub(crate) fn line_is_monospace(line: &crate::types::TextLine) -> bool {
    monospace_share_reaches_code(line, true)
}

/// [`line_is_monospace`] for a line inside an open code block: a URL in the
/// code (`baseUrl="https://…"`) is code there, not a mono-set link, and must
/// not cut the block in two.
pub(crate) fn line_continues_code(line: &crate::types::TextLine) -> bool {
    monospace_share_reaches_code(line, false)
}

fn monospace_share_reaches_code(line: &crate::types::TextLine, links_are_prose: bool) -> bool {
    let mut monospace_chars = 0usize;
    let mut total_chars = 0usize;
    for item in &line.items {
        let text = item.text.trim();
        let chars = text.chars().count();
        total_chars += chars;
        // Hyperlinks and underlined text set in a mono face are link
        // styling, not code — a URL sidebar must not fence lyric lines. A
        // run that merely contains a URL (`curl http://localhost/paid`) is
        // a command, not a link.
        let looks_like_link = links_are_prose
            && (item.is_underline
                || matches!(item.item_type, crate::types::ItemType::Link(_))
                || is_bare_url(text));
        if is_monospace_font(&item.font) && !looks_like_link {
            monospace_chars += chars;
        }
    }
    total_chars > 0 && monospace_chars * 10 >= total_chars * 9
}

/// Whether a run is nothing but a URL: one word starting with a scheme
/// (`https://…`) or `www.`, optionally in angle brackets. A word that only
/// contains one (`baseUrl="https://…"`) is code or prose, not a link.
pub(crate) fn is_bare_url(text: &str) -> bool {
    let text = text.strip_prefix('<').unwrap_or(text);
    if text.contains(char::is_whitespace) {
        return false;
    }
    if text.starts_with("www.") {
        return true;
    }
    text.find("://").is_some_and(|at| {
        let scheme = &text[..at];
        scheme.starts_with(|c: char| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '.' | '-'))
    })
}

/// Check if font name indicates monospace
pub(crate) fn is_monospace_font(font_name: &str) -> bool {
    let lower = font_name.to_lowercase();
    // "Monotype" is a foundry prefix on proportional faces (Monotype
    // Corsiva, Monotype Garamond) — it must not satisfy the generic "mono"
    // token below.
    if lower.contains("monotype") {
        return false;
    }
    let patterns = [
        "courier",
        "consolas",
        "monaco",
        "menlo",
        "mono",
        "fixed",
        "terminal",
        "typewriter",
        "source code",
        "fira code",
        "jetbrains",
        "inconsolata",
        "dejavu sans mono",
        "liberation mono",
    ];

    patterns.iter().any(|p| lower.contains(p))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monotype_foundry_faces_are_not_monospace() {
        // "Monotype" is a foundry prefix on proportional faces; the generic
        // "mono" token must not classify them as code fonts.
        assert!(!is_monospace_font("MonotypeCorsiva"));
        assert!(!is_monospace_font("ABCDEF+Monotype-Garamond"));
        assert!(is_monospace_font("RobotoMono-Regular"));
        assert!(is_monospace_font("PTMono"));
        assert!(is_monospace_font("Courier"));
    }

    #[test]
    fn format_list_item_plain_bullet() {
        assert_eq!(format_list_item("● Item"), "- Item");
        assert_eq!(format_list_item("• Item"), "- Item");
    }

    #[test]
    fn format_list_item_bullet_inside_underline() {
        // Fully-underlined bullet line: the marker must move outside the
        // <u> wrapper so markdown still renders a list item.
        assert_eq!(format_list_item("<u>● Item text</u>"), "- <u>Item text</u>");
    }

    #[test]
    fn format_list_item_bullet_inside_bold() {
        // PDF that uses bold font for both the marker and the label produces
        // a single bold run like "**● Label:** rest"; the bullet must still
        // be stripped and the bold wrapper preserved on the label.
        assert_eq!(
            format_list_item("**● Fraud: Willing cooperation;**"),
            "- **Fraud: Willing cooperation;**"
        );
        assert_eq!(
            format_list_item("**● Label:** rest of line"),
            "- **Label:** rest of line"
        );
        assert_eq!(format_list_item("*● Italic:* rest"), "- *Italic:* rest");
    }

    #[test]
    fn format_list_item_already_dash() {
        assert_eq!(format_list_item("- existing"), "- existing");
    }

    #[test]
    fn a_bare_url_starts_with_its_scheme() {
        assert!(is_bare_url("https://example.com/a"));
        assert!(is_bare_url("<https://example.com/a>"));
        assert!(is_bare_url("www.example.com"));
        assert!(!is_bare_url("baseUrl=\"https://example.com/a\""));
        assert!(!is_bare_url("curl https://example.com/a"));
        assert!(!is_bare_url("://missing-scheme"));
    }

    #[test]
    fn check_marks_start_list_items_and_stay_in_their_text() {
        assert!(is_list_item("\u{2705} True HEPA filter"));
        assert!(is_list_item("\u{2714}\u{FE0F} Done"));
        assert!(starts_with_bullet_marker("\u{2610} Not yet"));
        assert_eq!(
            format_list_item("\u{2705} True HEPA filter"),
            "- \u{2705} True HEPA filter"
        );
        // A mark without a space after it, or another symbol, is text.
        assert!(!is_list_item("\u{2705}done"));
        assert!(!is_list_item("\u{2192} next step"));
    }

    #[test]
    fn is_list_item_with_bullet_space() {
        assert!(is_list_item("● Item"));
        assert!(is_list_item("• Item"));
        assert!(is_list_item("- Item"));
    }

    #[test]
    fn dingbat_bullets_open_list_items() {
        // markitai: Word's Wingdings bullets, mapped to dingbats.
        for bullet in ['❖', '◆', '◇', '■', '▪', '➢'] {
            let item = format!("{bullet} 课程内容以本说明为准");
            assert!(is_list_item(&item), "{item}");
            assert_eq!(format_list_item(&item), "- 课程内容以本说明为准");
        }
        assert!(!is_list_item("■■ 图例"));
    }

    #[test]
    fn cjk_enumerators_open_list_items() {
        for item in [
            "1．报名学员范围",
            "1）2类书法课程",
            "3、学习期间",
            "（4）其他",
            "12） 集训",
        ] {
            assert!(is_list_item(item), "{item}");
            assert_eq!(format_list_item(item), format!("- {item}"));
        }
        // A number and its unit, a date, or nothing after the marker.
        for text in ["2026年05月21日", "1．", "1．5倍", "1234．五", "（1"] {
            assert!(!is_list_item(text), "{text}");
        }
    }

    #[test]
    fn cjk_sentences_and_fields_are_no_headings() {
        assert_eq!(text_words("Annual Report 2025"), 3);
        // Seventy characters of one sentence are many words.
        let sentence = "本中心办学资质与服务评价符合要求，您可以登录官方网站查看最近季度办学资质年检结果和学员服务评价等全部相关信息";
        assert!(text_words(sentence) > 15);
        assert!(text_words("社区书法艺术与传统绘画研修进阶课程（周末）说明") <= 15);
        assert!(reads_as_sentence(
            "报名人已确认学员符合课程简章所述的报名条件。"
        ));
        assert!(reads_as_sentence(
            "如有隐瞒或填写不实，足以影响学习中心正常安排的，"
        ));
        assert!(reads_as_sentence("第一句。第二句"));
        assert!(reads_as_sentence(
            "本中心办学资质与服务评价符合要求，您可以登录"
        ));
        assert!(!reads_as_sentence("一、总则，范围"));
        assert!(!reads_as_sentence("学习中心办学资质告知和服务评价"));
        assert!(!reads_as_sentence("Results."));
        assert!(reads_as_field("报名人：示例学员 证件类型：学生证"));
        assert!(reads_as_field("首期材料费合计：960.50"));
        assert!(!reads_as_field("说明："));
        assert!(!reads_as_field("Chapter 1: Introduction"));
        assert!(!reads_as_field(
            "这是一个很长很长的不像标签的句子开头部分：值"
        ));
    }
}
