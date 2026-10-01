//! Chinese, Japanese and Korean recognition aids: a second, enlarged reading
//! of small text or of text the recognizer missed entirely, and recovery of
//! characters it drops inside a line.

use super::Line;

/// Lines whose median height is below this many pixels are read again enlarged.
const SMALL_LINE: f32 = 24.0;
/// The line height, in pixels, that enlargement aims for.
const TARGET_LINE: f32 = 32.0;
const MIN_FACTOR: f32 = 1.25;
const MAX_FACTOR: f32 = 4.0;
/// A line found without text counts only when it is at least this many times
/// as wide as it is tall: a few characters, not a speck in a picture.
const LINE_SHAPE: f32 = 3.0;
/// A character box this many times wider than the line's median letter box
/// may hold a character that is missing from the recognized text.
const WIDE: f64 = 1.6;
/// Letters a line needs before its widths are compared.
pub(super) const MIN_LETTERS: usize = 4;
/// Recognized characters on each side of a wide box that the second reading covers.
const CONTEXT: usize = 4;
/// Characters on each side of a wide box that the second reading must repeat.
const ANCHOR: usize = 2;
/// Second readings of wide character regions attempted per image.
#[cfg(target_os = "macos")]
pub(super) const MAX_REREADS: usize = 32;

/// A writing system whose letters each fill one square, full-width body, so
/// that the recognizer boxes them about equally wide.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum Script {
    /// Han characters.
    Chinese,
    /// Han characters and kana. Small kana and the prolonged sound mark are
    /// drawn narrower, but Vision boxes them as wide as the other letters.
    Japanese,
    /// Hangul syllables. Vision's Korean recognizer does not read Hanja.
    Korean,
}

/// The script the aids measure for a Vision recognition language; other
/// languages take no aid.
pub(super) fn script(language: &str) -> Option<Script> {
    let is = |tag: &str| language.eq_ignore_ascii_case(tag);
    if is("zh-Hans") || is("zh-Hant") {
        Some(Script::Chinese)
    } else if is("ja-JP") {
        Some(Script::Japanese)
    } else if is("ko-KR") {
        Some(Script::Korean)
    } else {
        None
    }
}

impl Script {
    /// Whether `character` is one of this script's full-width letters: the
    /// characters whose boxes are compared and that may be inserted.
    pub(super) fn letter(self, character: char) -> bool {
        match self {
            Script::Chinese => han(character),
            Script::Japanese => han(character) || kana(character),
            Script::Korean => hangul(character),
        }
    }
}

/// CJK unified ideographs, extension A and compatibility ideographs.
pub(super) fn han(character: char) -> bool {
    matches!(character, '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}' | '\u{f900}'..='\u{faff}')
}

/// Full-width hiragana and katakana, small kana, the prolonged sound mark,
/// the kana iteration marks and the ideographic iteration mark 々; not the
/// katakana middle dot (punctuation) or half-width katakana.
fn kana(character: char) -> bool {
    matches!(
        character,
        '\u{3041}'..='\u{3096}'
            | '\u{309d}'..='\u{309f}'
            | '\u{30a1}'..='\u{30fa}'
            | '\u{30fc}'..='\u{30ff}'
            | '\u{31f0}'..='\u{31ff}'
            | '々'
    )
}

/// Precomposed Hangul syllables, not conjoining or compatibility jamo.
fn hangul(character: char) -> bool {
    matches!(character, '\u{ac00}'..='\u{d7a3}')
}

/// The factor for reading a `width` by `height` image again, enlarged, when
/// the lines holding the script's letters are small: at most `max_pixels`
/// result. Korean text is not enlarged: Vision reads small Hangul as well at
/// its own size, and enlarged copies changed vowels and dropped punctuation.
pub(super) fn enlargement(
    lines: &[Line],
    script: Script,
    width: u32,
    height: u32,
    max_pixels: u64,
) -> Option<f32> {
    if script == Script::Korean {
        return None;
    }
    let heights = lines
        .iter()
        .filter(|line| line.text.chars().any(|c| script.letter(c)))
        .map(|line| line.bounds[3] - line.bounds[1])
        .collect();
    factor(heights, width, height, max_pixels)
}

/// The factor for reading a `width` by `height` image again, enlarged, when
/// the first reading found no text but another reading found small `lines`
/// (pixel `[width, height]`): at most `max_pixels` result.
pub(super) fn missed(lines: &[[f32; 2]], width: u32, height: u32, max_pixels: u64) -> Option<f32> {
    let heights = lines
        .iter()
        .filter(|[long, tall]| *long >= LINE_SHAPE * *tall)
        .map(|[_, tall]| *tall)
        .collect();
    factor(heights, width, height, max_pixels)
}

/// The enlargement factor for text lines of these pixel `heights` in a
/// `width` by `height` image, when their median is small: toward the target
/// height, by at least the minimum factor, within `max_pixels`.
fn factor(mut heights: Vec<f32>, width: u32, height: u32, max_pixels: u64) -> Option<f32> {
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

/// Wide letter boxes in one recognized line. `boxes` holds each character's
/// normalized `[x, y, width, height]`, when the engine gave one.
pub(super) fn suspects(
    characters: &[char],
    boxes: &[Option<[f64; 4]>],
    script: Script,
) -> Vec<Suspect> {
    if characters.len() != boxes.len() {
        return Vec::new();
    }
    let letter_box = |index: usize| boxes[index].filter(|_| script.letter(characters[index]));
    let mut widths: Vec<f64> = (0..characters.len())
        .filter_map(|index| letter_box(index).map(|b| b[2]))
        .collect();
    if widths.len() < MIN_LETTERS {
        return Vec::new();
    }
    crate::sort::by(&mut widths, f64::total_cmp);
    let median = widths[widths.len() / 2];
    // Vision boxes the first syllable of a Korean line up to about twice as
    // wide as the others: three quarters of the wide Korean boxes measured.
    let first = characters.iter().position(|c| !c.is_whitespace());
    let mut found = Vec::new();
    for index in 0..characters.len() {
        if script == Script::Korean && Some(index) == first {
            continue;
        }
        let Some(own) = letter_box(index).filter(|b| b[2] > WIDE * median) else {
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

/// The one letter that a second `reading` of a suspect's region places
/// directly before or after `characters[index]`, with up to two neighbouring
/// characters on each side read identically, as an insertion index into
/// `characters`. Any other or ambiguous reading keeps the line unchanged.
pub(super) fn insertion(
    characters: &[char],
    index: usize,
    reading: &str,
    script: Script,
) -> Option<(usize, char)> {
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
        let candidate = if pair[1] == own && script.letter(pair[0]) {
            (index, pair[0])
        } else if pair[0] == own && script.letter(pair[1]) {
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

    /// The enlargement factor for these lines in a 400 by 200 image.
    fn enlarged(lines: &[Line], script: Script, max_pixels: u64) -> Option<f32> {
        enlargement(lines, script, 400, 200, max_pixels)
    }

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
        assert_eq!(enlarged(&small, Script::Chinese, 32_000_000), Some(2.0));
        // Latin lines do not count, however small.
        let latin = [
            line("small print", 0.0, 10.0),
            line("more print", 12.0, 22.0),
        ];
        assert_eq!(enlarged(&latin, Script::Chinese, 32_000_000), None);
        // Text at or above the threshold is read once.
        let large = [line("常规字号", 0.0, 24.0), line("常规字号", 30.0, 54.0)];
        assert_eq!(enlarged(&large, Script::Chinese, 32_000_000), None);
        let tiny = [line("极小", 0.0, 4.0)];
        assert_eq!(enlarged(&tiny, Script::Chinese, 32_000_000), Some(4.0));
        // The pixel limit caps the factor, and a negligible factor is skipped.
        assert_eq!(enlarged(&small, Script::Chinese, 80_000 * 9 / 4), Some(1.5));
        assert_eq!(enlarged(&small, Script::Chinese, 80_000 * 3 / 2), None);
        assert_eq!(enlarged(&[], Script::Chinese, 32_000_000), None);
    }

    #[test]
    fn chinese_japanese_and_korean_recognition_languages_use_the_aids() {
        assert_eq!(script("zh-Hans"), Some(Script::Chinese));
        assert_eq!(script("ZH-HANT"), Some(Script::Chinese));
        assert_eq!(script("ja-jp"), Some(Script::Japanese));
        assert_eq!(script("ko-KR"), Some(Script::Korean));
        for other in ["en-US", "zh", "ja", "ko", "vi-VN", "ar-SA"] {
            assert_eq!(script(other), None, "{other}");
        }
        assert!(han('的') && han('㐀') && han('豈'));
        assert!(!han('a') && !han('，') && !han('ア') && !han('한'));
    }

    #[test]
    fn each_script_measures_its_own_full_width_letters() {
        // Japanese: Han and kana, with small kana, the prolonged sound mark and
        // the iteration marks.
        for c in ['漢', 'あ', 'ア', 'ゃ', 'ッ', 'ー', 'ゝ', 'ヶ', 'ㇰ', '々'] {
            assert!(Script::Japanese.letter(c), "{c}");
        }
        for c in ['・', 'ｱ', '゛', '、', 'a', '1', '한'] {
            assert!(!Script::Japanese.letter(c), "{c}");
        }
        // Korean: precomposed Hangul syllables only.
        for c in ['가', '한', '글', '힣'] {
            assert!(Script::Korean.letter(c), "{c}");
        }
        for c in ['ㄱ', 'ᄀ', '漢', 'あ', 'a', '.'] {
            assert!(!Script::Korean.letter(c), "{c}");
        }
        // Chinese keeps Han characters only.
        assert!(Script::Chinese.letter('漢'));
        for c in ['あ', 'ア', 'ー', '々', '한'] {
            assert!(!Script::Chinese.letter(c), "{c}");
        }
    }

    #[test]
    fn small_japanese_lines_are_enlarged_but_korean_lines_are_not() {
        let kana = [
            line("ひらがなとカタカナ", 0.0, 16.0),
            line("ちいさいもじ", 20.0, 36.0),
        ];
        assert_eq!(enlarged(&kana, Script::Japanese, 32_000_000), Some(2.0));
        // Chinese measures Han lines only.
        assert_eq!(enlarged(&kana, Script::Chinese, 32_000_000), None);
        let hangul = [
            line("작은 글자", 0.0, 16.0),
            line("한국어 문장", 20.0, 36.0),
        ];
        assert_eq!(enlarged(&hangul, Script::Korean, 32_000_000), None);
        // Lines found without text follow the same rule, by their heights.
        let missed = |lines: &[[f32; 2]], max_pixels| super::missed(lines, 400, 200, max_pixels);
        let small = [[213.0, 16.0], [167.0, 16.0], [363.0, 40.0]];
        assert_eq!(missed(&small, 32_000_000), Some(2.0));
        assert_eq!(missed(&[[41.0, 8.0]], 32_000_000), Some(4.0));
        assert_eq!(missed(&[[300.0, 24.0], [300.0, 30.0]], 32_000_000), None);
        assert_eq!(missed(&small, 80_000 * 3 / 2), None);
        assert_eq!(missed(&[], 32_000_000), None);
        // A speck in a picture is not a line of text: a small line must be at
        // least three times as wide as it is tall.
        assert_eq!(missed(&[[36.0, 23.0]], 32_000_000), None);
        assert_eq!(missed(&[[47.0, 16.0]], 32_000_000), None);
        assert_eq!(missed(&[[48.0, 16.0]], 32_000_000), Some(2.0));
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
        let found = suspects(&characters, &boxes(&widths), Script::Chinese);
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
        assert!(suspects(&characters, &boxes(&uniform), Script::Chinese).is_empty());
        let mixed: Vec<char> = "中文 PDF 页面内容".chars().collect();
        let mut widths = vec![0.05; mixed.len()];
        widths[3] = 0.2;
        assert!(suspects(&mixed, &boxes(&widths), Script::Chinese).is_empty());
        let short: Vec<char> = "很宽的字".chars().collect();
        assert!(suspects(&short[..3], &boxes(&[0.05, 0.2, 0.05]), Script::Chinese).is_empty());
        // Mismatched geometry is ignored rather than indexed.
        assert!(suspects(&characters, &boxes(&[0.05; 3]), Script::Chinese).is_empty());
    }

    #[test]
    fn the_median_width_is_the_middle_of_the_sorted_widths() {
        // The wide box sits in the middle of the line, where an unsorted
        // "median" would take it as the typical width.
        let characters: Vec<char> = "采用参考版单一环境".chars().collect();
        let widths = [0.03, 0.03, 0.03, 0.03, 0.10, 0.04, 0.04, 0.04, 0.04];
        let found = suspects(&characters, &boxes(&widths), Script::Chinese);
        assert_eq!(found.iter().map(|s| s.index).collect::<Vec<_>>(), [4]);
    }

    #[test]
    fn context_stops_at_a_character_without_a_box() {
        let characters: Vec<char> = "甲乙丙丁戊己庚辛".chars().collect();
        let mut widths = vec![0.05; characters.len()];
        widths[5] = 0.12;
        widths[2] = 0.0;
        let found = suspects(&characters, &boxes(&widths), Script::Chinese);
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
            insertion(&characters, 4, "用参考版的单一环", Script::Chinese),
            Some((5, '的'))
        );
        assert_eq!(
            insertion(&characters, 5, "参考版的单一环境", Script::Chinese),
            Some((5, '的'))
        );
        // Whitespace in the reading and around the wide box is ignored.
        let spaced: Vec<char> = "圆角裁剪 PDF".chars().collect();
        assert_eq!(
            insertion(&spaced, 3, "角 裁剪的 PD", Script::Chinese),
            Some((4, '的'))
        );
        // Same text, a changed anchor, a non-Han insertion or two different
        // insertions: no change.
        assert_eq!(
            insertion(&characters, 4, "用参考版单一环", Script::Chinese),
            None
        );
        assert_eq!(
            insertion(&characters, 4, "用参考板的单一环", Script::Chinese),
            None
        );
        assert_eq!(
            insertion(&characters, 4, "用参看版的单一环", Script::Chinese),
            None
        );
        assert_eq!(
            insertion(&characters, 4, "用参考版的单二环", Script::Chinese),
            None
        );
        assert_eq!(
            insertion(&characters, 4, "用参考版，单一环", Script::Chinese),
            None
        );
        assert_eq!(
            insertion(&characters, 5, "参考版，单一环境", Script::Chinese),
            None
        );
        assert_eq!(
            insertion(&characters, 4, "参考版的单一参考版之单一", Script::Chinese),
            None
        );
        // The same insertion read twice is still one insertion.
        assert_eq!(
            insertion(&characters, 4, "参考版的单一 参考版的单一", Script::Chinese),
            Some((5, '的'))
        );
        // A doubled character is one insertion, whichever side it is read on.
        let doubled: Vec<char> = "我们看一下这个".chars().collect();
        assert_eq!(
            insertion(&doubled, 2, "我们看看一下", Script::Chinese),
            Some((2, '看'))
        );
        // At the start of a line there is no left anchor.
        assert_eq!(
            insertion(&characters, 0, "的采用参", Script::Chinese),
            Some((0, '的'))
        );
    }

    #[test]
    fn wide_kana_and_hangul_boxes_are_suspects_in_their_own_script() {
        // Vision drops 箱 after 段ボール and stretches the box of ル over it.
        let japanese: Vec<char> = "本を段ボールに詰めました".chars().collect();
        let mut widths = vec![0.05; japanese.len()];
        widths[5] = 0.11;
        // Vision often boxes the first character of a line up to 1.5 times as
        // wide as the others; that is not a suspect.
        widths[0] = 0.07;
        let found = suspects(&japanese, &boxes(&widths), Script::Japanese);
        assert_eq!(found.iter().map(|s| s.index).collect::<Vec<_>>(), [5]);
        // In Chinese only the Han characters count, and ル is not one.
        assert!(suspects(&japanese, &boxes(&widths), Script::Chinese).is_empty());
        // Word spaces have no box and are skipped; 왕 covers a dropped 릉.
        let korean: Vec<char> = "오래된 왕과 사찰이 남아".chars().collect();
        let mut widths: Vec<f64> = korean
            .iter()
            .map(|c| if c.is_whitespace() { 0.0 } else { 0.05 })
            .collect();
        widths[4] = 0.1;
        // The first syllable of a Korean line is often boxed this wide; it is
        // not a suspect, while the first Han character of a Chinese line is.
        widths[0] = 0.1;
        let found = suspects(&korean, &boxes(&widths), Script::Korean);
        assert_eq!(found.iter().map(|s| s.index).collect::<Vec<_>>(), [4]);
        assert!(suspects(&korean, &boxes(&widths), Script::Japanese).is_empty());
        let indented: Vec<char> = " 오래된 왕과 사찰이".chars().collect();
        let mut widths: Vec<f64> = indented
            .iter()
            .map(|c| if c.is_whitespace() { 0.0 } else { 0.05 })
            .collect();
        widths[1] = 0.1;
        assert!(suspects(&indented, &boxes(&widths), Script::Korean).is_empty());
        let chinese: Vec<char> = "采用参考版单一环境".chars().collect();
        let mut widths = vec![0.05; chinese.len()];
        widths[0] = 0.1;
        let found = suspects(&chinese, &boxes(&widths), Script::Chinese);
        assert_eq!(found.iter().map(|s| s.index).collect::<Vec<_>>(), [0]);
        // A wide Latin box in a Korean line is not a suspect.
        let mixed: Vec<char> = "한국어 PDF 문서 변환".chars().collect();
        let mut widths: Vec<f64> = mixed
            .iter()
            .map(|c| if c.is_whitespace() { 0.0 } else { 0.05 })
            .collect();
        widths[5] = 0.2;
        assert!(suspects(&mixed, &boxes(&widths), Script::Korean).is_empty());
    }

    #[test]
    fn each_script_inserts_only_its_own_letters() {
        let japanese: Vec<char> = "本を段ボールに詰めました".chars().collect();
        assert_eq!(
            insertion(&japanese, 5, "段ボール箱に詰め", Script::Japanese),
            Some((6, '箱'))
        );
        // A dropped kana is a letter in Japanese, not in Chinese.
        let particle: Vec<char> = "日本語文章には漢字".chars().collect();
        assert_eq!(
            insertion(&particle, 2, "日本語の文章に", Script::Japanese),
            Some((3, 'の'))
        );
        assert_eq!(
            insertion(&particle, 2, "日本語の文章に", Script::Chinese),
            None
        );
        // Korean anchors skip word spaces; Hanja read into the gap is not inserted.
        let korean: Vec<char> = "오래된 왕과 사찰이".chars().collect();
        assert_eq!(
            insertion(&korean, 4, "래된 왕릉과 사찰", Script::Korean),
            Some((5, '릉'))
        );
        assert_eq!(
            insertion(&korean, 4, "래된 왕陵과 사찰", Script::Korean),
            None
        );
        let chinese: Vec<char> = "采用参考版单一环境".chars().collect();
        assert_eq!(
            insertion(&chinese, 4, "参考版의单一", Script::Chinese),
            None
        );
    }
}
