//! Chinese recognition aids: a second, enlarged reading of small text, and
//! recovery of characters the recognizer drops inside a line.

use super::Line;

/// Lines whose median height is below this many pixels are read again enlarged.
const SMALL_LINE: f32 = 24.0;
/// The line height, in pixels, that enlargement aims for.
const TARGET_LINE: f32 = 32.0;
const MIN_FACTOR: f32 = 1.25;
const MAX_FACTOR: f32 = 4.0;
/// A character box this many times wider than the line's median Han character
/// box may hold a character that is missing from the recognized text.
const WIDE: f64 = 1.6;
/// Han characters a line needs before its widths are compared.
pub(super) const MIN_HAN: usize = 4;
/// Recognized characters on each side of a wide box that the second reading covers.
const CONTEXT: usize = 4;
/// Characters on each side of a wide box that the second reading must repeat.
const ANCHOR: usize = 2;
/// Second readings of wide character regions attempted per image.
#[cfg(target_os = "macos")]
pub(super) const MAX_REREADS: usize = 32;

/// Whether the Chinese aids apply to a Vision recognition language.
pub(super) fn applies(language: &str) -> bool {
    ["zh-Hans", "zh-Hant"]
        .iter()
        .any(|tag| language.eq_ignore_ascii_case(tag))
}

/// CJK unified ideographs, extension A and compatibility ideographs.
pub(super) fn han(character: char) -> bool {
    matches!(character, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}')
}

/// The factor for reading a `width` by `height` image again, enlarged, when
/// the lines holding Han characters are small: at most `max_pixels` result.
pub(super) fn enlargement(lines: &[Line], width: u32, height: u32, max_pixels: u64) -> Option<f32> {
    let mut heights: Vec<f32> = lines
        .iter()
        .filter(|line| line.text.chars().any(han))
        .map(|line| line.bounds[3] - line.bounds[1])
        .collect();
    if heights.is_empty() {
        return None;
    }
    heights.sort_by(f32::total_cmp);
    let median = heights[heights.len() / 2].max(1.0);
    if median >= SMALL_LINE {
        return None;
    }
    let pixels = (u64::from(width) * u64::from(height)).max(1) as f64;
    let room = (max_pixels as f64 / pixels).sqrt() as f32;
    let factor = (TARGET_LINE / median).min(MAX_FACTOR).min(room);
    (factor >= MIN_FACTOR).then_some(factor)
}

/// A recognized character whose box is wide enough to hide a dropped one, and
/// the normalized region `[x, y, width, height]` that a second reading covers.
#[derive(Debug, PartialEq)]
pub(super) struct Suspect {
    pub index: usize,
    pub region: [f64; 4],
}

/// Wide Han character boxes in one recognized line. `boxes` holds each
/// character's normalized `[x, y, width, height]`, when the engine gave one.
pub(super) fn suspects(characters: &[char], boxes: &[Option<[f64; 4]>]) -> Vec<Suspect> {
    if characters.len() != boxes.len() {
        return Vec::new();
    }
    let han_box = |index: usize| boxes[index].filter(|_| han(characters[index]));
    let mut widths: Vec<f64> = (0..characters.len())
        .filter_map(|index| han_box(index).map(|b| b[2]))
        .collect();
    if widths.len() < MIN_HAN {
        return Vec::new();
    }
    widths.sort_by(f64::total_cmp);
    let median = widths[widths.len() / 2];
    let mut found = Vec::new();
    for index in 0..characters.len() {
        let Some(own) = han_box(index).filter(|b| b[2] > WIDE * median) else {
            continue;
        };
        let (mut left, mut bottom) = (own[0], own[1]);
        let (mut right, mut top) = (own[0] + own[2], own[1] + own[3]);
        // Each side extends over its neighbours up to one the engine gave no box.
        for before in [true, false] {
            for neighbour in context(characters, index, CONTEXT, before) {
                let Some(b) = boxes[neighbour] else { break };
                left = left.min(b[0]);
                bottom = bottom.min(b[1]);
                right = right.max(b[0] + b[2]);
                top = top.max(b[1] + b[3]);
            }
        }
        let (pad_x, pad_y) = (0.15 * median, 0.2 * own[3]);
        let x = (left - pad_x).clamp(0.0, 1.0);
        let y = (bottom - pad_y).clamp(0.0, 1.0);
        let region = [
            x,
            y,
            (right + pad_x).clamp(0.0, 1.0) - x,
            (top + pad_y).clamp(0.0, 1.0) - y,
        ];
        if region[2] > 0.0 && region[3] > 0.0 {
            found.push(Suspect { index, region });
        }
    }
    found
}

/// Up to `count` indices of non-blank characters before (or after) `index`,
/// nearest first.
fn context(characters: &[char], index: usize, count: usize, before: bool) -> Vec<usize> {
    let indices: Box<dyn Iterator<Item = usize>> = if before {
        Box::new((0..index).rev())
    } else {
        Box::new(index + 1..characters.len())
    };
    indices
        .filter(|&i| !characters[i].is_whitespace())
        .take(count)
        .collect()
}

/// The one Han character that a second `reading` of a suspect's region places
/// directly before or after `characters[index]`, with up to two neighbouring
/// characters on each side read identically, as an insertion index into
/// `characters`. Any other or ambiguous reading keeps the line unchanged.
pub(super) fn insertion(characters: &[char], index: usize, reading: &str) -> Option<(usize, char)> {
    let reading: Vec<char> = reading.chars().filter(|c| !c.is_whitespace()).collect();
    let mut left: Vec<char> = context(characters, index, ANCHOR, true)
        .into_iter()
        .map(|i| characters[i])
        .collect();
    left.reverse();
    let right: Vec<char> = context(characters, index, ANCHOR, false)
        .into_iter()
        .map(|i| characters[i])
        .collect();
    let own = characters[index];
    let span = left.len() + right.len() + 2;
    let mut found: Option<(usize, char)> = None;
    for window in reading.windows(span) {
        let (head, rest) = window.split_at(left.len());
        let (pair, tail) = rest.split_at(2);
        if head != left.as_slice() || tail != right.as_slice() {
            continue;
        }
        let candidate = if pair[1] == own && han(pair[0]) {
            (index, pair[0])
        } else if pair[0] == own && han(pair[1]) {
            (index + 1, pair[1])
        } else {
            continue;
        };
        // The same insertion found twice (a repeated reading) is still one.
        match found {
            None => found = Some(candidate),
            Some(previous) if previous == candidate => {}
            Some(_) => return None,
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, top: f32, bottom: f32) -> Line {
        Line {
            text: text.into(),
            confidence: 1.0,
            bounds: [0.0, top, 100.0, bottom],
        }
    }

    #[test]
    fn small_han_lines_are_enlarged_toward_the_target_height_within_bounds() {
        let small = [
            line("小字号的中文", 0.0, 16.0),
            line("第二行中文", 20.0, 36.0),
            line("Large English heading", 40.0, 140.0),
        ];
        assert_eq!(enlargement(&small, 400, 200, 32_000_000), Some(2.0));
        // Latin lines do not count, however small.
        let latin = [
            line("small print", 0.0, 10.0),
            line("more print", 12.0, 22.0),
        ];
        assert_eq!(enlargement(&latin, 400, 200, 32_000_000), None);
        // Text at or above the threshold is read once.
        let large = [line("常规字号", 0.0, 24.0), line("常规字号", 30.0, 54.0)];
        assert_eq!(enlargement(&large, 400, 200, 32_000_000), None);
        let tiny = [line("极小", 0.0, 4.0)];
        assert_eq!(enlargement(&tiny, 400, 200, 32_000_000), Some(4.0));
        // The pixel limit caps the factor, and a negligible factor is skipped.
        assert_eq!(enlargement(&small, 400, 200, 80_000 * 9 / 4), Some(1.5));
        assert_eq!(enlargement(&small, 400, 200, 80_000 * 3 / 2), None);
        assert_eq!(enlargement(&[], 400, 200, 32_000_000), None);
    }

    #[test]
    fn only_chinese_recognition_languages_use_the_aids() {
        assert!(applies("zh-Hans") && applies("zh-Hant") && applies("ZH-HANS"));
        assert!(!applies("en-US") && !applies("ja-JP") && !applies("zh"));
        assert!(han('的') && han('㐀') && han('豈'));
        assert!(!han('a') && !han('，') && !han('ア') && !han('한'));
    }

    fn boxes(widths: &[f64]) -> Vec<Option<[f64; 4]>> {
        let mut x = 0.0;
        widths
            .iter()
            .map(|&width| {
                let b = [x, 0.5, width, 0.1];
                x += width;
                (width > 0.0).then_some(b)
            })
            .collect()
    }

    #[test]
    fn a_double_width_han_box_is_a_suspect_with_its_context_region() {
        let characters: Vec<char> = "采用参考版单一环境代".chars().collect();
        let mut widths = vec![0.05; characters.len()];
        widths[4] = 0.10;
        let found = suspects(&characters, &boxes(&widths));
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].index, 4);
        // Four characters each side (0..=8), padded by 0.15 median width and
        // 0.2 box height, clamped to the image.
        let [x, y, w, h] = found[0].region;
        assert!(
            x.abs() < 1e-9 && (w - (0.50 + 0.0075)).abs() < 1e-9,
            "{x} {w}"
        );
        assert!(
            (y - 0.48).abs() < 1e-9 && (h - 0.14).abs() < 1e-9,
            "{y} {h}"
        );
        // Uniform widths, wide Latin or punctuation boxes, and short lines are not suspects.
        let uniform = vec![0.05; characters.len()];
        assert!(suspects(&characters, &boxes(&uniform)).is_empty());
        let mixed: Vec<char> = "中文 PDF 页面内容".chars().collect();
        let mut widths = vec![0.05; mixed.len()];
        widths[3] = 0.2;
        assert!(suspects(&mixed, &boxes(&widths)).is_empty());
        let short: Vec<char> = "很宽的字".chars().collect();
        assert!(suspects(&short[..3], &boxes(&[0.05, 0.2, 0.05])).is_empty());
        // Mismatched geometry is ignored rather than indexed.
        assert!(suspects(&characters, &boxes(&[0.05; 3])).is_empty());
    }

    #[test]
    fn context_stops_at_a_character_without_a_box() {
        let characters: Vec<char> = "甲乙丙丁戊己庚辛".chars().collect();
        let mut widths = vec![0.05; characters.len()];
        widths[5] = 0.12;
        widths[2] = 0.0;
        let found = suspects(&characters, &boxes(&widths));
        assert_eq!(found.len(), 1);
        // 丁 and 戊 widen the region on the left; 丙 has no box, so 乙 is not
        // reached. The right side still reaches 辛.
        let [left, _, width, _] = found[0].region;
        assert!((left - (0.10 - 0.0075)).abs() < 1e-9, "{left}");
        assert!((left + width - (0.42 + 0.0075)).abs() < 1e-9, "{width}");
    }

    #[test]
    fn a_dropped_character_is_inserted_only_with_matching_anchors() {
        let characters: Vec<char> = "采用参考版单一环境".chars().collect();
        // 的 read before 单 (index 5), after 版 (index 4).
        assert_eq!(
            insertion(&characters, 4, "用参考版的单一环"),
            Some((5, '的'))
        );
        assert_eq!(
            insertion(&characters, 5, "参考版的单一环境"),
            Some((5, '的'))
        );
        // Whitespace in the reading and around the wide box is ignored.
        let spaced: Vec<char> = "圆角裁剪 PDF".chars().collect();
        assert_eq!(insertion(&spaced, 3, "角 裁剪的 PD"), Some((4, '的')));
        // Same text, a changed anchor, a non-Han insertion or two different
        // insertions: no change.
        assert_eq!(insertion(&characters, 4, "用参考版单一环"), None);
        assert_eq!(insertion(&characters, 4, "用参考板的单一环"), None);
        assert_eq!(insertion(&characters, 4, "用参看版的单一环"), None);
        assert_eq!(insertion(&characters, 4, "用参考版的单二环"), None);
        assert_eq!(insertion(&characters, 4, "用参考版，单一环"), None);
        assert_eq!(insertion(&characters, 5, "参考版，单一环境"), None);
        assert_eq!(insertion(&characters, 4, "参考版的单一参考版之单一"), None);
        // The same insertion read twice is still one insertion.
        assert_eq!(
            insertion(&characters, 4, "参考版的单一 参考版的单一"),
            Some((5, '的'))
        );
        // A doubled character is one insertion, whichever side it is read on.
        let doubled: Vec<char> = "我们看一下这个".chars().collect();
        assert_eq!(insertion(&doubled, 2, "我们看看一下"), Some((2, '看')));
        // At the start of a line there is no left anchor.
        assert_eq!(insertion(&characters, 0, "的采用参"), Some((0, '的')));
    }
}
