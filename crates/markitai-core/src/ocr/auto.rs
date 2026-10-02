//! The default language policy. When `ocr.lang` is left at `en`, an image is
//! read as English first. Vision's English recognizer reads Latin and Cyrillic
//! script well and returns nothing, or a few low-confidence symbols, for
//! Chinese, Japanese and Korean text. When the reading shows that, the image
//! is read again as Chinese (which also reads Latin, Japanese and Traditional
//! Chinese), then as Korean, and as Japanese when the Chinese reading holds
//! kana. These functions judge readings; `ocr` runs them.

use super::{Line, cjk};
use cjk::Script;

/// A line the recognizer is less sure of than this is doubtful. Vision reports
/// 1.0 for nearly every line of English it reads, and 0.3 or 0.5 for what it
/// makes of Chinese, Japanese or Korean text.
const SURE: f32 = 0.9;
/// One doubtful line in this many makes the English reading doubtful, and one
/// in two fails it.
const DOUBTFUL_IN: usize = 4;
/// A sure English reading of at most this many lines is short: the English
/// recognizer also returns confident Latin fragments for lines that mix Latin
/// with Chinese, Japanese or Korean (`PDF, Word, Excel SEAT.` for a line
/// of Chinese with three Latin words), which only another reading tells from
/// English. The cost is one more reading of an image with at most two lines.
const SHORT: usize = 2;

/// What an English reading shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Verdict {
    /// Lines the recognizer is sure of: the reading stands.
    Sound,
    /// Sure of its lines, but there are only a few: read again as Chinese.
    Short,
    /// A quarter or more of its lines are doubtful: read again as Chinese.
    Doubtful,
    /// No text, or half of the lines doubtful: the text is probably of another
    /// script. Read again as Chinese and as Korean.
    Failed,
}

fn present(lines: &[Line]) -> impl Iterator<Item = &Line> {
    lines.iter().filter(|line| !line.text.trim().is_empty())
}

/// Judge an English reading. An empty reading failed: English never reads the
/// text of Chinese, Japanese or Korean as anything, and a blank image costs
/// two more cheap readings to learn that.
pub(super) fn judge(lines: &[Line]) -> Verdict {
    let total = present(lines).count();
    let doubtful = present(lines).filter(|line| line.confidence < SURE).count();
    if total == 0 || doubtful * 2 >= total {
        Verdict::Failed
    } else if doubtful * DOUBTFUL_IN >= total {
        Verdict::Doubtful
    } else if total <= SHORT {
        Verdict::Short
    } else {
        Verdict::Sound
    }
}

/// A reading holds text of a script when it has at least this many letters of
/// it, which are at least one in [`SHARE_IN`] of its letters and digits, and
/// whose confidence adds up to at least [`MIN_STRENGTH`]: a few characters
/// among English, or among symbols, are noise, not text.
const MIN_LETTERS: usize = 4;
const SHARE_IN: usize = 4;
const MIN_STRENGTH: f32 = 3.0;
/// Letters of lines the recognizer is at least this sure of are credible.
/// Vision's Chinese recognizer reports 0.5 or 1.0 for text it reads at an
/// ordinary size, and 0.3 for most of what it makes of Korean.
const CREDIBLE: f32 = 0.45;
/// A Chinese reading with this many credible letters settles the language.
/// With fewer, a Korean reading is tried too and the stronger one kept: the
/// Chinese recognizer makes confident Han and kana of Hangul, but never more
/// than 30 credible letters of it in 184 Korean images, while 27 of 416
/// Chinese and Japanese images hold fewer.
#[cfg(target_os = "macos")]
pub(super) const CONVINCING: usize = 32;
/// Kana in a Chinese reading mean Japanese text when there are at least this
/// many, and they are at least one in [`KANA_IN`] of the credible letters.
const MIN_KANA: usize = 8;
const KANA_IN: usize = 5;

fn letters(lines: &[Line], script: Script) -> impl Iterator<Item = (&Line, usize)> {
    present(lines).map(move |line| {
        (
            line,
            line.text.chars().filter(|c| script.letter(*c)).count(),
        )
    })
}

/// The letters of `script` in the lines the recognizer is fairly sure of.
pub(super) fn credible(lines: &[Line], script: Script) -> usize {
    letters(lines, script)
        .filter(|(line, _)| line.confidence >= CREDIBLE)
        .map(|(_, count)| count)
        .sum()
}

/// The letters of `script`, each weighed by the confidence of its line: how
/// much text of that script a reading holds. Readings of the same image in
/// different languages are compared by it.
pub(super) fn strength(lines: &[Line], script: Script) -> f32 {
    letters(lines, script)
        .map(|(line, count)| count as f32 * line.confidence)
        .sum()
}

/// Whether a reading holds text of `script`.
pub(super) fn reads(lines: &[Line], script: Script) -> bool {
    let count: usize = letters(lines, script).map(|(_, count)| count).sum();
    let all: usize = present(lines)
        .map(|line| line.text.chars().filter(|c| c.is_alphanumeric()).count())
        .sum();
    count >= MIN_LETTERS && count * SHARE_IN >= all && strength(lines, script) >= MIN_STRENGTH
}

/// Whether a Chinese reading is of Japanese text: it holds kana, which
/// Chinese text does not.
pub(super) fn japanese(lines: &[Line]) -> bool {
    let kana: usize = present(lines)
        .filter(|line| line.confidence >= CREDIBLE)
        .map(|line| line.text.chars().filter(|c| cjk::kana(*c)).count())
        .sum();
    kana >= MIN_KANA && kana * KANA_IN >= credible(lines, Script::Japanese)
}

/// Whether to warn about an image that no reading could read: the English
/// reading failed, and it found text lines (at least half of them doubtful) or
/// another reading found text (`elsewhere`). A blank page, or a photograph
/// without text, found none.
pub(super) fn unread(verdict: Verdict, english: &[Line], elsewhere: bool) -> bool {
    verdict == Verdict::Failed && (elsewhere || present(english).next().is_some())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(text: &str, confidence: f32) -> Line {
        Line {
            text: text.into(),
            confidence,
            bounds: [0.0, 0.0, 100.0, 10.0],
        }
    }

    /// Four lines of English with these confidences.
    fn english(confidences: [f32; 4]) -> Vec<Line> {
        [
            "The quick brown fox jumps over the lazy dog.",
            "Pack my box with five dozen liquor jugs 1234567890.",
            "Price list: $4.50, $3.20, 12/03/2026 (net)",
            "Local OCR runs on this machine.",
        ]
        .iter()
        .zip(confidences)
        .map(|(text, confidence)| line(text, confidence))
        .collect()
    }

    #[test]
    fn english_is_doubtful_at_one_doubtful_line_in_four_and_failed_at_one_in_two() {
        assert_eq!(judge(&english([1.0; 4])), Verdict::Sound);
        assert_eq!(judge(&english([1.0, 0.5, 1.0, 1.0])), Verdict::Doubtful);
        assert_eq!(judge(&english([1.0, 0.5, 0.3, 1.0])), Verdict::Failed);
        // One doubtful line in five is not enough.
        let mut five = english([1.0, 0.5, 1.0, 1.0]);
        five.push(line("Another sure line", 1.0));
        assert_eq!(judge(&five), Verdict::Sound);
        // Nothing found failed, and blank lines are not lines.
        assert_eq!(judge(&[]), Verdict::Failed);
        assert_eq!(judge(&[line("  ", 1.0)]), Verdict::Failed);
        let blank = [
            line("a b c", 1.0),
            line("", 0.3),
            line("d e f", 1.0),
            line("g", 1.0),
        ];
        assert_eq!(judge(&blank), Verdict::Sound);
        assert_eq!(
            judge(&[line("Name: Alice", 1.0), line("**7: 02", 0.3)]),
            Verdict::Failed
        );
    }

    #[test]
    fn low_confidence_is_what_english_makes_of_other_scripts() {
        // What this system's English recognizer returned for Chinese text.
        assert_eq!(judge(&[line("#Æ[× •", 0.3)]), Verdict::Failed);
        assert_eq!(judge(&[line("Markitai OCR", 0.5)]), Verdict::Failed);
        // Other scripts' letters are not a failure when the recognizer is sure.
        let russian = line("Съешь же ещё этих мягких французских булок", 1.0);
        assert_eq!(judge(&[russian]), Verdict::Short);
    }

    #[test]
    fn a_confident_reading_of_at_most_two_lines_is_short() {
        // What the English recognizer returned for lines of Chinese with Latin words.
        for text in [
            "macOS E PDF REA, AH OCR FAR J",
            "·L‡o",
            "I PDF, Word 5EA, W Markdown X4.",
        ] {
            assert_eq!(judge(&[line(text, 1.0)]), Verdict::Short, "{text}");
        }
        let two = english([1.0; 4]);
        assert_eq!(judge(&two[..2]), Verdict::Short);
        assert_eq!(judge(&two[..3]), Verdict::Sound);
    }

    #[test]
    fn a_script_is_read_when_it_has_letters_a_fair_share_and_strength() {
        let chinese = [
            line("本地文字识别在这台机器上运行，不需要任何网络。", 1.0),
            line("支持 PDF、Word、Excel 与图片转换。", 0.5),
        ];
        assert_eq!(credible(&chinese, Script::Chinese), 28);
        assert_eq!(strength(&chinese, Script::Chinese), 24.5);
        assert!(reads(&chinese, Script::Japanese) && !reads(&chinese, Script::Korean));
        assert!(!japanese(&chinese));
        // Text at a size the recognizer doubts (0.3) is text, but weak.
        let doubted = [line("双然合口进天叫发号创壹号叫全号对", 0.3)];
        assert_eq!(credible(&doubted, Script::Chinese), 0);
        assert!(reads(&doubted, Script::Japanese));
        assert!(!reads(&[line("双然合口", 0.3)], Script::Japanese));
        // A few hallucinated letters among English are not Chinese text.
        let noise = [
            line("The quick brown fox jumps over the lazy dog 双然合口", 1.0),
            line("Pack my box with five dozen liquor jugs", 1.0),
        ];
        assert_eq!(credible(&noise, Script::Chinese), 4);
        assert!(!reads(&noise, Script::Chinese));
        // Each script counts its own letters.
        let korean = [line("로컬 문자 인식은 이 컴퓨터에서 실행되며", 1.0)];
        assert_eq!(credible(&korean, Script::Korean), 17);
        assert_eq!(credible(&korean, Script::Chinese), 0);
        assert!(reads(&korean, Script::Korean) && !reads(&korean, Script::Japanese));
        // Three letters are never enough.
        assert!(!reads(&[line("你好吗", 1.0)], Script::Chinese));
        assert!(reads(&[line("你好世界", 1.0)], Script::Chinese));
        // Of two readings of one image, the one with more text is stronger.
        let hallucinated = [line("双然合口进天叫发号创壹号叫全号对", 0.3)];
        let hangul = [line(
            "비가 그친 공원에서는 아이들이 물웅덩이를 뛰어넘으며",
            1.0,
        )];
        assert!(strength(&hangul, Script::Korean) > strength(&hallucinated, Script::Japanese));
    }

    #[test]
    fn kana_in_a_chinese_reading_mean_japanese() {
        let text = [line(
            "ローカルの文字認識はこのマシン上で動作し、ネットワークは不要です。",
            1.0,
        )];
        assert!(reads(&text, Script::Japanese) && japanese(&text));
        // One stray kana among Chinese letters is not Japanese.
        let stray = [line(
            "本地文字识别在这台机器上运行，不需要任何网络ミ。",
            1.0,
        )];
        assert!(!japanese(&stray));
        // Neither are a few kana among many kanji.
        let kanji = [line(
            "日本語文字認識処理機械学習環境設定変換出力ああいうえお",
            1.0,
        )];
        assert!(!japanese(&kanji));
        // Kana the recognizer doubts do not count.
        assert!(!japanese(&[line(
            "ローカルの文字認識はこのマシン上で動作し",
            0.3
        )]));
    }

    #[test]
    fn a_failed_search_warns_only_when_text_was_seen_but_not_read() {
        // English found only doubtful lines, or half of them.
        let garbage = [line("#Æ[× •", 0.3)];
        assert!(unread(judge(&garbage), &garbage, false));
        let half = english([1.0, 1.0, 0.5, 0.5]);
        assert!(unread(judge(&half), &half, false));
        // English found nothing: a blank page, unless another reading found text.
        assert!(!unread(judge(&[]), &[], false));
        assert!(unread(judge(&[]), &[], true));
        // Sure, doubtful or short readings that nothing replaced are English text.
        for lines in [
            english([1.0; 4]),
            english([1.0, 1.0, 1.0, 0.5]),
            english([1.0; 4]).into_iter().take(2).collect(),
        ] {
            assert!(!unread(judge(&lines), &lines, true));
        }
    }
}
