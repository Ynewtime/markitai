//! Link annotations over a page's text runs.
//!
//! A `/Link` annotation is a box on the page with an action; the text it
//! covers is ordinary page text. The runs a link covers become link items
//! (`ItemType::Link`, which the page reader also gives a run that is link
//! styling) carrying a Markdown destination, which the layout pass renders as
//! `[text](target)`. Only web and mail addresses are carried; the underline a
//! browser draws under any link is its styling, not emphasis, and is dropped.
use pdf_inspector::{TextItem, types::ItemType};
use std::sync::Arc;

/// A link annotation's box on the page, with its target when Markdown may
/// carry it.
pub(super) struct LinkBox {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
    target: Option<Arc<str>>,
}

impl LinkBox {
    /// The boxes of a page's link annotation items.
    pub(super) fn collect(items: &[TextItem]) -> Vec<Self> {
        items
            .iter()
            .filter_map(|item| {
                let ItemType::Link(uri) = &item.item_type else {
                    return None;
                };
                let (x0, x1) = (
                    item.x.min(item.x + item.width),
                    item.x.max(item.x + item.width),
                );
                let (y0, y1) = (
                    item.y.min(item.y + item.height),
                    item.y.max(item.y + item.height),
                );
                [x0, x1, y0, y1]
                    .iter()
                    .all(|n| n.is_finite())
                    .then(|| Self {
                        x0,
                        y0,
                        x1,
                        y1,
                        target: target(uri),
                    })
            })
            .collect()
    }

    /// Whether the middle of `item`'s height is inside the box.
    fn on_line(&self, item: &TextItem) -> bool {
        let middle = item.y + item.height / 2.;
        middle >= self.y0 - 1. && middle <= self.y1 + 1.
    }

    /// How much of `item`'s width the box covers.
    fn overlap(&self, item: &TextItem) -> f32 {
        (item.x + item.width).min(self.x1 + 1.) - item.x.max(self.x0 - 1.)
    }
}

/// The destination a link annotation's URI may carry into Markdown: a web or
/// mail address, written as a link destination (spaces and angle brackets
/// encoded, parentheses and backslashes escaped). Another scheme
/// (`javascript:`, `file:`, `data:`), a relative or local reference and a URI
/// holding control characters or other white space carry none.
pub(super) fn target(uri: &str) -> Option<Arc<str>> {
    let uri = uri.trim();
    let (scheme, rest) = uri.split_once(':')?;
    let allowed = match scheme.to_ascii_lowercase().as_str() {
        "http" | "https" => rest.len() > 2 && rest.starts_with("//"),
        "mailto" => rest.contains('@'),
        _ => false,
    };
    if !allowed {
        return None;
    }
    let mut destination = String::with_capacity(uri.len());
    for ch in uri.chars() {
        match ch {
            ' ' => destination.push_str("%20"),
            '<' => destination.push_str("%3C"),
            '>' => destination.push_str("%3E"),
            '(' | ')' | '\\' => {
                destination.push('\\');
                destination.push(ch);
            }
            _ if ch.is_control() || ch.is_whitespace() => return None,
            _ => destination.push(ch),
        }
    }
    Some(destination.into())
}

/// Mark the runs `links` cover. A run belongs to a link when the middle of
/// its height is inside the link's box and four fifths of its width overlap
/// it; punctuation ending the run outside the box (`see the [manual].`)
/// stays outside the link. A run only partly inside links (a sentence the
/// extractor merged with its linked words) is split at word boundaries:
/// a word belongs to the link holding its middle and half its width, and
/// sentence punctuation and unpaired quotation marks or brackets at either
/// end of the linked words stay outside. Where the words stand inside the run
/// is estimated from typical glyph widths, as the extractor keeps no glyph
/// positions; the split is made only when the words placed in each link span
/// most of its box (they overlap by half their union), so a run the estimate
/// cannot place keeps its text unlinked. A run whose underline the links
/// explain (four fifths of its width under links, with targets or without)
/// loses the underline: it is link styling.
pub(super) fn apply(items: &mut Vec<TextItem>, links: &[LinkBox]) {
    if links.is_empty() {
        return;
    }
    let mut output = Vec::with_capacity(items.len());
    for mut item in std::mem::take(items) {
        let on_line: Vec<&LinkBox> = links
            .iter()
            .filter(|link| link.on_line(&item) && link.overlap(&item) > 0.)
            .collect();
        if on_line.is_empty() || item.width <= 0. || !item.width.is_finite() {
            output.push(item);
            continue;
        }
        if covered(&item, &on_line) >= item.width * 0.8 {
            item.is_underline = false;
        }
        let targets: Vec<&LinkBox> = on_line
            .into_iter()
            .filter(|link| link.target.is_some())
            .collect();
        if targets.is_empty() {
            output.push(item);
            continue;
        }
        let full = targets
            .iter()
            .enumerate()
            .map(|(index, link)| (index, link.overlap(&item)))
            .filter(|(_, overlap)| *overlap >= item.width * 0.8)
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(index, _)| index);
        let owners = match full {
            Some(index) => Some(whole(&item, targets[index], index)),
            None => words(&item, &targets),
        };
        match owners {
            Some(owners) => output.extend(pieces(item, &owners, &targets)),
            None => output.push(item),
        }
    }
    *items = output;
}

/// The width under the union of `links` within `item`.
fn covered(item: &TextItem, links: &[&LinkBox]) -> f32 {
    let mut spans: Vec<(f32, f32)> = links
        .iter()
        .map(|link| {
            (
                item.x.max(link.x0 - 1.),
                (item.x + item.width).min(link.x1 + 1.),
            )
        })
        .filter(|(start, end)| end > start)
        .collect();
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut total = 0.;
    let mut reach = f32::NEG_INFINITY;
    for (start, end) in spans {
        let start = start.max(reach);
        if end > start {
            total += end - start;
        }
        reach = reach.max(end);
    }
    total
}

/// Relative advance of a character, in thousandths of an em: Helvetica's
/// widths for ASCII, an em for wide East Asian characters and an average
/// letter otherwise. Proportional faces differ in detail, but much less
/// than all characters from one another.
fn advance(c: char) -> f32 {
    match c {
        ' ' | '!' | ',' | '.' | '/' | ':' | ';' | 'I' | '[' | ']' | 'f' | 't' | '\\' => 278.,
        'i' | 'j' | 'l' | '\'' | '|' => 222.,
        '"' => 355.,
        '(' | ')' | '-' | 'r' | '`' => 333.,
        '*' => 389.,
        '+' | '<' | '=' | '>' | '~' => 584.,
        'm' | 'M' => 833.,
        'w' => 722.,
        'W' => 944.,
        '@' => 1015.,
        '%' => 889.,
        '&' | 'A' | 'B' | 'E' | 'K' | 'P' | 'S' | 'V' | 'X' | 'Y' => 667.,
        'C' | 'D' | 'H' | 'N' | 'R' | 'U' => 722.,
        'F' | 'T' | 'Z' => 611.,
        'G' | 'O' | 'Q' => 778.,
        'J' | 'c' | 'k' | 's' | 'v' | 'x' | 'y' | 'z' => 500.,
        'L' => 556.,
        '\u{2014}' => 1000.,
        '\u{2013}' | '\u{00b7}' | '\u{2026}' => 556.,
        '\u{2018}' | '\u{2019}' => 222.,
        '\u{201c}' | '\u{201d}' => 333.,
        c if is_wide(c) => 1000.,
        _ => 556.,
    }
}

/// Characters an East Asian face sets an em wide.
fn is_wide(c: char) -> bool {
    matches!(c as u32, 0x1100..=0x115F | 0x2E80..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF | 0xFE30..=0xFE4F | 0xFF00..=0xFF60 | 0xFFE0..=0xFFE6 | 0x20000..=0x3FFFD)
}

/// Each character boundary's estimated position in `item` (one more than
/// its characters).
fn positions(item: &TextItem) -> Vec<f32> {
    let mut total = 0.;
    let mut sums = vec![0.];
    for c in item.text.chars() {
        total += advance(c);
        sums.push(total);
    }
    let scale = if total > 0. { item.width / total } else { 0. };
    sums.into_iter().map(|sum| item.x + sum * scale).collect()
}

/// Sentence punctuation that follows a link rather than ending its text.
fn trailing(c: char) -> bool {
    matches!(
        c,
        '.' | ',' | ';' | ':' | '!' | '?' | '\u{2026}' | '\u{00b7}' | '\u{2014}' | '\u{2013}'
    )
}

/// The opening mark a closing quotation mark or bracket pairs with.
fn opener(c: char) -> Option<char> {
    match c {
        ')' => Some('('),
        ']' => Some('['),
        '}' => Some('{'),
        '\u{201d}' => Some('\u{201c}'),
        '\u{2019}' => Some('\u{2018}'),
        '\u{00bb}' => Some('\u{00ab}'),
        '"' => Some('"'),
        _ => None,
    }
}

/// The closing mark an opening quotation mark or bracket pairs with.
fn closer(c: char) -> Option<char> {
    match c {
        '(' => Some(')'),
        '[' => Some(']'),
        '{' => Some('}'),
        '\u{201c}' => Some('\u{201d}'),
        '\u{2018}' => Some('\u{2019}'),
        '\u{00ab}' => Some('\u{00bb}'),
        '"' => Some('"'),
        _ => None,
    }
}

/// Unlink the punctuation at either end of the characters `owners` gives
/// link `index`, from `start` to `end`: trailing sentence punctuation, and a
/// quotation mark or bracket whose partner is not inside. `keep` says whether
/// a character may stay linked anyway (the box covers it).
fn trim(
    chars: &[char],
    owners: &mut [Option<usize>],
    (start, end): (usize, usize),
    keep: &dyn Fn(usize) -> bool,
) {
    // Whether `mark`, paired with `partner`, has no partner inside the span.
    let unpaired = |span: &[char], mark: char, partner: char| {
        let count = |c: char| span.iter().filter(|&&x| x == c).count();
        if mark == partner {
            count(mark) % 2 == 1
        } else {
            count(partner) < count(mark)
        }
    };
    let mut end = end;
    while end > start {
        let c = chars[end - 1];
        let lone = opener(c).is_some_and(|open| unpaired(&chars[start..end], c, open));
        if !(trailing(c) || lone) || keep(end - 1) {
            break;
        }
        owners[end - 1] = None;
        end -= 1;
    }
    let mut first = start;
    while first < end {
        let c = chars[first];
        let lone = closer(c).is_some_and(|close| unpaired(&chars[first..end], c, close));
        if !lone || keep(first) {
            break;
        }
        owners[first] = None;
        first += 1;
    }
}

/// Owners for a run a link covers as a whole: every character, except
/// punctuation at either end that lies outside the box.
fn whole(item: &TextItem, link: &LinkBox, index: usize) -> Vec<Option<usize>> {
    let chars: Vec<char> = item.text.chars().collect();
    let at = positions(item);
    let mut owners = vec![Some(index); chars.len()];
    let inside = |i: usize| {
        let middle = (at[i] + at[i + 1]) / 2.;
        middle >= link.x0 && middle <= link.x1
    };
    let (start, end) = span(&chars);
    trim(&chars, &mut owners, (start, end), &inside);
    owners
}

/// The first and last non-white characters (end exclusive).
fn span(chars: &[char]) -> (usize, usize) {
    let start = chars.iter().position(|c| !c.is_whitespace()).unwrap_or(0);
    let end = chars
        .iter()
        .rposition(|c| !c.is_whitespace())
        .map_or(start, |i| i + 1);
    (start, end)
}

/// Owners for a run links cover in part: each word in the link holding its
/// middle and half its width (see [`apply`]); `None` when no word is placed
/// in a link or a link's words do not span its box.
fn words(item: &TextItem, links: &[&LinkBox]) -> Option<Vec<Option<usize>>> {
    let chars: Vec<char> = item.text.chars().collect();
    let at = positions(item);
    let mut ranges: Vec<(usize, usize, Option<usize>)> = Vec::new();
    let mut index = 0;
    while index < chars.len() {
        if chars[index].is_whitespace() {
            index += 1;
            continue;
        }
        let start = index;
        while index < chars.len() && !chars[index].is_whitespace() {
            index += 1;
        }
        let (left, right) = (at[start], at[index]);
        let middle = (left + right) / 2.;
        let owner = links.iter().position(|link| {
            let inside = right.min(link.x1) - left.max(link.x0);
            middle >= link.x0 - 1. && middle <= link.x1 + 1. && inside * 2. >= right - left
        });
        ranges.push((start, index, owner));
    }
    // A word of punctuation alone belongs to a link only inside it.
    for number in 0..ranges.len() {
        let (start, end, owner) = ranges[number];
        if owner.is_some() && chars[start..end].iter().all(|c| !c.is_alphanumeric()) {
            let before = number.checked_sub(1).and_then(|n| ranges[n].2);
            let after = ranges.get(number + 1).and_then(|r| r.2);
            if before != owner || after != owner {
                ranges[number].2 = None;
            }
        }
    }
    if ranges.iter().all(|r| r.2.is_none()) {
        return None;
    }
    let mut owners: Vec<Option<usize>> = vec![None; chars.len()];
    for (number, &(start, end, owner)) in ranges.iter().enumerate() {
        owners[start..end].fill(owner);
        // The spaces after a word go with it.
        let next = ranges.get(number + 1).map_or(chars.len(), |r| r.0);
        if ranges.get(number + 1).is_some_and(|r| r.2 == owner) {
            owners[end..next].fill(owner);
        }
    }
    // Each link's words, trimmed, must span most of its box.
    for (number, link) in links.iter().enumerate() {
        let Some(start) = owners.iter().position(|o| *o == Some(number)) else {
            continue;
        };
        let end = owners
            .iter()
            .rposition(|o| *o == Some(number))
            .map_or(start, |i| i + 1);
        if owners[start..end].iter().any(|o| *o != Some(number)) {
            // One link's words with another's between: no clear reading.
            return None;
        }
        trim(&chars, &mut owners, (start, end), &|_| false);
        let Some(start) = owners.iter().position(|o| *o == Some(number)) else {
            continue;
        };
        let end = owners
            .iter()
            .rposition(|o| *o == Some(number))
            .map_or(start, |i| i + 1);
        let (left, right) = (at[start], at[end]);
        let common = right.min(link.x1) - left.max(link.x0);
        let union = right.max(link.x1) - left.min(link.x0);
        if union <= 0. || common * 2. < union {
            return None;
        }
    }
    Some(owners)
}

/// `item` cut where the owner of its characters changes, each piece a link
/// item of its owner's target or text as before. A piece keeps the spaces
/// after its text; geometry follows the estimated positions.
fn pieces(item: TextItem, owners: &[Option<usize>], links: &[&LinkBox]) -> Vec<TextItem> {
    // Spaces between pieces take the owner before them, except at the end
    // of a link, where they leave it.
    let chars: Vec<(usize, char)> = item.text.char_indices().collect();
    let mut owners = owners.to_vec();
    for i in 1..owners.len() {
        if chars[i].1.is_whitespace() && owners[i].is_some() && owners[i] != owners[i - 1] {
            owners[i] = owners[i - 1];
        }
    }
    let target = |owner: Option<usize>| owner.and_then(|o| links[o].target.as_ref());
    let first = owners.first().copied().flatten();
    if owners.iter().all(|o| *o == first) {
        let mut item = item;
        if let Some(target) = target(first) {
            item.item_type = ItemType::Link(target.to_string());
        }
        return vec![item];
    }
    let at = positions(&item);
    let byte = |index: usize| chars.get(index).map_or(item.text.len(), |c| c.0);
    let mut output = Vec::new();
    let mut start = 0;
    for index in 1..=owners.len() {
        if index < owners.len() && owners[index] == owners[start] {
            continue;
        }
        let mut piece = item.clone();
        piece.text = item.text[byte(start)..byte(index)].to_owned();
        piece.x = at[start];
        piece.width = at[index] - at[start];
        if let Some(target) = target(owners[start]) {
            piece.item_type = ItemType::Link(target.to_string());
        }
        output.push(piece);
        start = index;
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(text: &str, x: f32, width: f32) -> TextItem {
        TextItem {
            text: text.into(),
            x,
            y: 700.,
            width,
            height: 12.,
            font: "Helvetica".into(),
            font_tag: "F1".into(),
            legacy_symbol_rewrite: false,
            font_size: 12.,
            page: 1,
            is_bold: false,
            is_italic: false,
            font_weight: None,
            bold_source: None,
            fixed_pitch: Some(false),
            fill_color: None,
            stroke_color: None,
            render_mode: None,
            is_underline: false,
            is_strikeout: false,
            rotation: 0.,
            advance_known: true,
            item_type: ItemType::Text,
            mcid: None,
            baseline_shift: 0.,
        }
    }

    fn link(x0: f32, x1: f32, uri: &str) -> LinkBox {
        LinkBox {
            x0,
            y0: 697.,
            x1,
            y1: 711.,
            target: target(uri),
        }
    }

    /// The pieces' texts with their targets.
    fn read(items: &[TextItem]) -> Vec<(String, Option<String>)> {
        items
            .iter()
            .map(|item| {
                let target = match &item.item_type {
                    ItemType::Link(target) => Some(target.clone()),
                    _ => None,
                };
                (item.text.clone(), target)
            })
            .collect()
    }

    #[test]
    fn only_web_and_mail_targets_are_carried() {
        assert_eq!(
            target("https://example.com/a b(c)").as_deref(),
            Some("https://example.com/a%20b\\(c\\)")
        );
        assert_eq!(
            target(" mailto:a@example.com ").as_deref(),
            Some("mailto:a@example.com")
        );
        for uri in [
            "javascript:alert(1)",
            "file:///etc/passwd",
            "data:text/html,x",
            "/relative/page",
            "#section",
            "https:",
            "http://exa\nmple.com",
            "mailto:nobody",
        ] {
            assert_eq!(target(uri), None, "{uri}");
        }
    }

    #[test]
    fn a_run_inside_a_link_becomes_a_link_item_and_loses_its_underline() {
        let mut item = run("link text", 100., 40.);
        item.is_underline = true;
        let mut items = vec![run("Second ", 60., 38.), item];
        apply(&mut items, &[link(99.5, 140.5, "https://example.com/x")]);
        assert_eq!(
            read(&items),
            [
                ("Second ".into(), None),
                ("link text".into(), Some("https://example.com/x".into()))
            ]
        );
        assert!(!items[1].is_underline);
    }

    #[test]
    fn a_link_without_a_carried_target_only_drops_the_underline() {
        let mut item = run("Home", 72., 29.);
        item.is_underline = true;
        let mut items = vec![item];
        apply(&mut items, &[link(72., 101.3, "javascript:void(0)")]);
        assert_eq!(read(&items), [("Home".into(), None)]);
        assert!(!items[0].is_underline);
    }

    #[test]
    fn punctuation_after_a_link_stays_outside_it() {
        // "Uncategorized." with the box ending before the full stop.
        let item = run("Uncategorized.", 100., 80.);
        let mut items = vec![item];
        apply(
            &mut items,
            &[link(100., 77. + 100. - 2., "https://example.com/c")],
        );
        assert_eq!(
            read(&items),
            [
                ("Uncategorized".into(), Some("https://example.com/c".into())),
                (".".into(), None)
            ]
        );
    }

    #[test]
    fn linked_words_inside_a_merged_run_are_split_out() {
        // "See this report for supporting analysis." with the box over
        // "this report", placed by the glyph widths the estimate uses.
        let text = "See this report for supporting analysis.";
        let item = run(text, 0., 240.);
        let at = positions(&item);
        let (start, end) = (4, 15);
        let mut items = vec![item];
        apply(
            &mut items,
            &[link(at[start], at[end], "https://example.com/report")],
        );
        assert_eq!(
            read(&items),
            [
                ("See ".into(), None),
                (
                    "this report".into(),
                    Some("https://example.com/report".into())
                ),
                (" for supporting analysis.".into(), None),
            ]
        );
        // The pieces keep the run's extent between them.
        assert!((items[0].x - 0.).abs() < 1e-3);
        let last = &items[2];
        assert!((last.x + last.width - 240.).abs() < 1e-3);
    }

    #[test]
    fn a_comma_after_linked_words_is_not_linked() {
        let text = "I work at Voodoo, a French company";
        let item = run(text, 0., 200.);
        let at = positions(&item);
        let mut items = vec![item];
        // The box covers "Voodoo" only.
        apply(&mut items, &[link(at[10], at[16], "http://voodoo.io/")]);
        assert_eq!(
            read(&items),
            [
                ("I work at ".into(), None),
                ("Voodoo".into(), Some("http://voodoo.io/".into())),
                (", a French company".into(), None),
            ]
        );
    }

    #[test]
    fn a_box_the_words_do_not_span_leaves_the_run_unlinked() {
        // A box over a sliver of one long word: no word is half inside.
        let item = run("Supercalifragilistic expialidocious", 0., 200.);
        let mut items = vec![item];
        apply(&mut items, &[link(50., 60., "https://example.com/")]);
        assert_eq!(
            read(&items),
            [("Supercalifragilistic expialidocious".into(), None)]
        );
        // A box much wider than the word it holds is no reading either.
        let item = run("a b c", 0., 30.);
        let mut items = vec![item];
        apply(&mut items, &[link(10., 200., "https://example.com/")]);
        assert_eq!(read(&items), [("a b c".into(), None)]);
    }
}
