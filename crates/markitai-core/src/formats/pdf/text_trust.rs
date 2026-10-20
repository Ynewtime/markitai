//! Whether a page's text layer says what the page shows.
//!
//! A text layer can be present and still wrong, and the page reader's own
//! checks (undecodable runs, consonant soup and enciphered text, see
//! `pdf-inspector`'s `text_quality`) do not see every way it goes wrong.
//! These checks read the text the page reader gives for a page:
//!
//! - **A broken font map** that reached the text: private-use, replacement,
//!   unassigned or control code points making up a fifth of the page (the
//!   reader already leaves out runs that are mostly such codes; this is what
//!   remains of them spread through the runs it kept).
//! - **Raw glyph codes** written out as `(cid:12)`, as a PDF made from
//!   pdfminer's output carries them.
//! - **Mojibake**: UTF-8 text that went through Windows-1252 on its way into
//!   the PDF (`cafÃ©`, `â€œ`).
//! - **Right-to-left text in reverse order.** The reader puts a line's runs
//!   into reading order, so the storage order alone is no fault; what is
//!   checked is the text it gives, by the letters that can only end a word
//!   (Hebrew final forms, Arabic `ة` and `ى`) turning up at the start of
//!   words.
//!
//! The idea of checking a page's text layer by the script it shows comes from
//! Indra OCR (datalab-to/indra); this is markitai's own implementation of it,
//! after Python markitai's `converter/pdf_text_trust.py`.

use regex::Regex;
use std::sync::OnceLock;
use unicode_properties::{GeneralCategory, GeneralCategoryGroup, UnicodeGeneralCategory};

/// The reason of a page whose text layer passed the checks.
pub(crate) const TRUSTED: &str = "text layer trusted";
/// The reason of a page with no text to read.
pub(crate) const NO_TEXT_LAYER: &str = "no text layer";
/// The prefix of every reason a text layer failed for.
pub(crate) const FAILED: &str = "text layer failed the script check: ";

/// Pages with fewer characters than this (whitespace included) have no
/// text layer to judge.
pub(super) const MIN_CHARS: usize = 50;

// A broken font map: at least this many unreadable characters, and this
// share of the page's non-blank characters. Bullets and icons drawn from a
// symbol font are a few percent of a page at most.
const BAD_MIN_CHARS: usize = 10;
const BAD_MIN_SHARE: f64 = 0.2;
// (cid:NN) glyph codes: at least this many, covering this share of the text.
const CID_MIN: usize = 3;
const CID_MIN_SHARE: f64 = 0.2;
// Mojibake: at least this many words that are UTF-8 read as Windows-1252,
// and this share of the words that hold a non-ASCII character.
const MOJIBAKE_MIN_WORDS: usize = 3;
const MOJIBAKE_MIN_SHARE: f64 = 0.3;
// Right-to-left text in reverse order: the right-to-left letters are at
// least this share of the page's letters, and reversed-order evidence beats
// reading-order evidence by this factor in at least this many words.
const RTL_MIN_SHARE: f64 = 0.2;
const RTL_MIN_EVIDENCE: usize = 3;
const RTL_MARGIN: usize = 2;

/// Why a page's text layer failed the checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Failure {
    /// What is wrong, without its measure, so pages that fail alike share
    /// one warning: `mojibake, UTF-8 read as Windows-1252`.
    pub kind: &'static str,
    /// The same with its measure: `mojibake, UTF-8 read as Windows-1252 (5 words)`.
    pub detail: String,
    /// The writing system the page shows, when the check knows it (a
    /// reversed Hebrew layer: `Hebrew`). Whether OCR can read the page
    /// depends on it; `None` is a script OCR is taken to read.
    pub script: Option<&'static str>,
}

impl Failure {
    fn new(kind: &'static str, measure: Option<String>, script: Option<&'static str>) -> Self {
        let detail = match measure {
            Some(measure) => format!("{kind} ({measure})"),
            None => kind.to_owned(),
        };
        Self {
            kind,
            detail,
            script,
        }
    }

    /// The reason the reader gives for the page: [`FAILED`] and its kind.
    pub fn reason(&self) -> String {
        format!("{FAILED}{}", self.kind)
    }
}

// --- writing systems --------------------------------------------------------

// Code point ranges -> script, sorted by start. Han, kana and the CJK marks
// are one script here: one recognizer reads them, and a Japanese line mixes
// them. The Indic and Southeast Asian blocks are named by their block.
const SCRIPT_RANGES: &[(u32, u32, &str)] = &[
    (0x0370, 0x03FF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"),
    (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"),
    (0x0700, 0x074F, "Syriac"),
    (0x0750, 0x077F, "Arabic"),
    (0x0780, 0x07BF, "Thaana"),
    (0x08A0, 0x08FF, "Arabic"),
    (0x0900, 0x097F, "Devanagari"),
    (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"),
    (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"),
    (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"),
    (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
    (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"),
    (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"),
    (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1100, 0x11FF, "Hangul"),
    (0x1200, 0x139F, "Ethiopic"),
    (0x1780, 0x17FF, "Khmer"),
    (0x1800, 0x18AF, "Mongolian"),
    (0x1C80, 0x1C8F, "Cyrillic"),
    (0x1C90, 0x1CBF, "Georgian"),
    (0x1E00, 0x1EFF, "Latin"),
    (0x1F00, 0x1FFF, "Greek"),
    (0x2E80, 0x2FDF, "CJK"),
    (0x3040, 0x30FF, "CJK"),
    (0x3130, 0x318F, "Hangul"),
    (0x31F0, 0x31FF, "CJK"),
    (0x3400, 0x4DBF, "CJK"),
    (0x4E00, 0x9FFF, "CJK"),
    (0xAC00, 0xD7AF, "Hangul"),
    (0xF900, 0xFAFF, "CJK"),
    (0xFB1D, 0xFB4F, "Hebrew"),
    (0xFB50, 0xFDFF, "Arabic"),
    (0xFE70, 0xFEFF, "Arabic"),
    (0xFF21, 0xFF3A, "Latin"),
    (0xFF41, 0xFF5A, "Latin"),
    (0xFF66, 0xFF9F, "CJK"),
    (0x20000, 0x3134F, "CJK"),
];

fn is_letter(ch: char) -> bool {
    ch.general_category_group() == GeneralCategoryGroup::Letter
}

/// The writing system of a letter; `None` for anything else (digits, signs,
/// math letters such as 𝑥, letters of a script without a name here).
pub(crate) fn char_script(ch: char) -> Option<&'static str> {
    if !is_letter(ch) {
        return None;
    }
    let code = ch as u32;
    // ASCII, Latin-1, Latin Extended-A/B, IPA
    if code < 0x2B0 {
        return Some("Latin");
    }
    let index = SCRIPT_RANGES.partition_point(|&(start, _, _)| start <= code);
    let (start, end, name) = *SCRIPT_RANGES.get(index.checked_sub(1)?)?;
    (start..=end).contains(&code).then_some(name)
}

/// The writing system most letters of `text` belong to (of a tie, the one
/// met first), or `None`.
pub(crate) fn dominant_script(text: &str) -> Option<&'static str> {
    let mut counts: Vec<(&'static str, usize)> = Vec::new();
    for script in text.chars().filter_map(char_script) {
        match counts.iter_mut().find(|(name, _)| *name == script) {
            Some((_, count)) => *count += 1,
            None => counts.push((script, 1)),
        }
    }
    let mut best: Option<(&'static str, usize)> = None;
    for (name, count) in counts {
        if best.is_none_or(|(_, most)| count > most) {
            best = Some((name, count));
        }
    }
    best.map(|(name, _)| name)
}

/// The writing systems the recognizer for a normalized `ocr.lang` spelling
/// reads: the default policy (`en`) reads Latin, Chinese and Japanese, then
/// Korean; a single-language model reads its own script and Latin; the
/// multilingual model, every other language, Latin and CJK. An unknown
/// language reads as the multilingual one: it fails on its own when OCR runs.
pub(crate) fn readable_scripts(spelling: &str) -> &'static [&'static str] {
    let language = spelling.split('-').next().unwrap_or(spelling);
    let named = |names: &[&str]| names.contains(&spelling) || names.contains(&language);
    if spelling == "en" {
        &["Latin", "CJK", "Hangul"]
    } else if named(&["ko", "korean"]) {
        &["Latin", "Hangul"]
    } else if named(&["ar", "arabic", "ars", "fa", "ur", "ug"]) {
        &["Latin", "Arabic"]
    } else if named(&["th", "thai"]) {
        &["Latin", "Thai"]
    } else if named(&["el", "greek"]) {
        &["Latin", "Greek"]
    } else if spelling == "sr-cyrl"
        || named(&[
            "eslav",
            "ru",
            "uk",
            "be",
            "russian",
            "ukrainian",
            "cyrillic",
            "bg",
            "mk",
            "kk",
            "ky",
            "mn",
            "tg",
        ])
    {
        &["Latin", "Cyrillic"]
    } else if named(&["latin"]) {
        &["Latin"]
    } else if named(&["devanagari", "hi", "mr", "ne", "sa"]) {
        &["Latin", "Devanagari"]
    } else if named(&["ta", "tamil"]) {
        &["Latin", "Tamil"]
    } else if named(&["te", "telugu"]) {
        &["Latin", "Telugu"]
    } else {
        &["Latin", "CJK"]
    }
}

/// The `ocr.lang` value whose recognizer reads `script`, if any.
pub(crate) fn lang_for_script(script: &str) -> Option<&'static str> {
    Some(match script {
        "Latin" => "en",
        "CJK" => "zh",
        "Hangul" => "ko",
        "Arabic" => "ar",
        "Thai" => "th",
        "Cyrillic" => "cyrillic",
        "Greek" => "el",
        "Devanagari" => "devanagari",
        "Tamil" => "ta",
        "Telugu" => "te",
        _ => return None,
    })
}

// --- the checks -------------------------------------------------------------

/// A character no font map should produce for text: replacement,
/// private-use, unassigned or control. The Microsoft symbol-font range
/// (U+F020–U+F0FF, Symbol and Wingdings bytes) is left out: icons and
/// bullets, not letters.
fn unreadable(ch: char) -> bool {
    if ch == '\u{FFFD}' {
        return true;
    }
    if ('\u{F020}'..='\u{F0FF}').contains(&ch) {
        return false;
    }
    match ch.general_category() {
        GeneralCategory::PrivateUse | GeneralCategory::Unassigned | GeneralCategory::Surrogate => {
            true
        }
        GeneralCategory::Control => !matches!(ch, '\t' | '\r' | '\n'),
        _ => false,
    }
}

/// The Windows-1252 byte of a character, or the Latin-1 byte of a C1
/// control (a byte Windows-1252 leaves undefined passes through as one).
fn windows_1252(ch: char) -> Option<u8> {
    const HIGH: [char; 32] = [
        '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8D}', 'Ž',
        '\u{8F}', '\u{90}', '‘', '’', '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9D}',
        'ž', 'Ÿ',
    ];
    let code = ch as u32;
    if code < 0x80 || (0xA0..=0xFF).contains(&code) {
        return Some(code as u8);
    }
    if let Some(index) = HIGH.iter().position(|&high| high == ch) {
        return Some(0x80 + index as u8);
    }
    (0x80..=0x9F).contains(&code).then_some(code as u8)
}

/// What a word reads as when it is UTF-8 read as Windows-1252, if it is.
fn unmangled(word: &str) -> Option<String> {
    let bytes: Vec<u8> = word.chars().map(windows_1252).collect::<Option<_>>()?;
    let fixed = String::from_utf8(bytes).ok()?;
    (fixed != word).then_some(fixed)
}

/// Hebrew points and accents, which split no word.
fn hebrew_mark(ch: char) -> bool {
    ('\u{0591}'..='\u{05BD}').contains(&ch)
        || matches!(
            ch,
            '\u{05BF}' | '\u{05C1}' | '\u{05C2}' | '\u{05C4}' | '\u{05C5}' | '\u{05C7}'
        )
}

/// Arabic harakat and the superscript alef, which split no word.
fn arabic_mark(ch: char) -> bool {
    matches!(ch, '\u{064B}'..='\u{0670}')
}

/// An Arabic presentation form (Presentation Forms-B) as the letters it
/// stands for; other characters as themselves.
fn arabic_base(ch: char, out: &mut String) {
    // (first form, forms, letter), in code point order.
    const FORMS: &[(u32, u32, char)] = &[
        (0xFE80, 1, 'ء'),
        (0xFE81, 2, 'آ'),
        (0xFE83, 2, 'أ'),
        (0xFE85, 2, 'ؤ'),
        (0xFE87, 2, 'إ'),
        (0xFE89, 4, 'ئ'),
        (0xFE8D, 2, 'ا'),
        (0xFE8F, 4, 'ب'),
        (0xFE93, 2, 'ة'),
        (0xFE95, 4, 'ت'),
        (0xFE99, 4, 'ث'),
        (0xFE9D, 4, 'ج'),
        (0xFEA1, 4, 'ح'),
        (0xFEA5, 4, 'خ'),
        (0xFEA9, 2, 'د'),
        (0xFEAB, 2, 'ذ'),
        (0xFEAD, 2, 'ر'),
        (0xFEAF, 2, 'ز'),
        (0xFEB1, 4, 'س'),
        (0xFEB5, 4, 'ش'),
        (0xFEB9, 4, 'ص'),
        (0xFEBD, 4, 'ض'),
        (0xFEC1, 4, 'ط'),
        (0xFEC5, 4, 'ظ'),
        (0xFEC9, 4, 'ع'),
        (0xFECD, 4, 'غ'),
        (0xFED1, 4, 'ف'),
        (0xFED5, 4, 'ق'),
        (0xFED9, 4, 'ك'),
        (0xFEDD, 4, 'ل'),
        (0xFEE1, 4, 'م'),
        (0xFEE5, 4, 'ن'),
        (0xFEE9, 4, 'ه'),
        (0xFEED, 2, 'و'),
        (0xFEEF, 2, 'ى'),
        (0xFEF1, 4, 'ي'),
    ];
    let code = ch as u32;
    if (0xFEF5..=0xFEFC).contains(&code) {
        out.push('ل');
        out.push(['آ', 'أ', 'إ', 'ا'][((code - 0xFEF5) / 2) as usize]);
        return;
    }
    let index = FORMS.partition_point(|&(start, _, _)| start <= code);
    match index.checked_sub(1).map(|index| FORMS[index]) {
        Some((start, count, letter)) if code < start + count => out.push(letter),
        _ => out.push(ch),
    }
}

/// The runs of at least two characters of `letter` in `text`.
fn words(text: &str, letter: impl Fn(char) -> bool) -> Vec<Vec<char>> {
    let mut found = Vec::new();
    let mut word = Vec::new();
    for ch in text.chars().chain(std::iter::once(' ')) {
        if letter(ch) {
            word.push(ch);
        } else if !word.is_empty() {
            if word.len() >= 2 {
                found.push(std::mem::take(&mut word));
            }
            word.clear();
        }
    }
    found
}

/// `Hebrew` or `Arabic` when the text reads in reverse order.
fn reversed_rtl(text: &str, letters: usize) -> Option<&'static str> {
    if letters == 0 {
        return None;
    }
    let hebrew_text: String = text.chars().filter(|&ch| !hebrew_mark(ch)).collect();
    let hebrew = words(&hebrew_text, |ch| ('\u{05D0}'..='\u{05EA}').contains(&ch));
    let (mut normal, mut backwards) = (0, 0);
    let mut rtl_letters: usize = hebrew.iter().map(Vec::len).sum();
    for word in &hebrew {
        let (first, last) = (word[0], word[word.len() - 1]);
        normal += usize::from("ךםןףץ".contains(last));
        backwards += usize::from("ךםןףץ".contains(first)) + usize::from("כמנפצ".contains(last));
    }
    let mut script = "Hebrew";
    let mut arabic_text = String::with_capacity(text.len());
    for ch in text.chars().filter(|&ch| !arabic_mark(ch)) {
        arabic_base(ch, &mut arabic_text);
    }
    let arabic = words(&arabic_text, |ch| ('\u{0621}'..='\u{064A}').contains(&ch));
    let arabic_letters: usize = arabic.iter().map(Vec::len).sum();
    if arabic_letters > rtl_letters {
        script = "Arabic";
        rtl_letters = arabic_letters;
        (normal, backwards) = (0, 0);
        // ta marbuta and alef maksura only ever end a word; the article ال
        // opens one.
        for word in &arabic {
            let (first, last) = (word[0], word[word.len() - 1]);
            let long = word.len() >= 4;
            normal += usize::from(matches!(last, 'ة' | 'ى'))
                + usize::from(long && word[..2] == ['ا', 'ل']);
            backwards += usize::from(matches!(first, 'ة' | 'ى'))
                + usize::from(long && word[word.len() - 2..] == ['ل', 'ا']);
        }
    }
    (rtl_letters as f64 >= RTL_MIN_SHARE * letters as f64
        && backwards >= RTL_MIN_EVIDENCE
        && backwards > RTL_MARGIN * normal)
        .then_some(script)
}

fn cid_codes() -> &'static Regex {
    static CID: OnceLock<Regex> = OnceLock::new();
    CID.get_or_init(|| Regex::new(r"\(cid:\d+\)").expect("valid pattern"))
}

/// The verdict on a page's text: `None` when it passes, else why it fails.
/// The caller judges only a page with at least [`MIN_CHARS`] of text.
pub(crate) fn judge(text: &str) -> Option<Failure> {
    let visible: Vec<char> = text.chars().filter(|ch| !ch.is_whitespace()).collect();
    let total = visible.len().max(1);

    let bad = visible.iter().filter(|&&ch| unreadable(ch)).count();
    if bad >= BAD_MIN_CHARS && bad as f64 >= BAD_MIN_SHARE * total as f64 {
        return Some(Failure::new(
            "a broken font map",
            Some(format!(
                "{}% of the characters are private-use, replacement or control codes",
                (bad * 100 + total / 2) / total
            )),
            None,
        ));
    }

    let cids: Vec<&str> = cid_codes().find_iter(text).map(|m| m.as_str()).collect();
    let cid_chars: usize = cids.iter().map(|code| code.chars().count()).sum();
    if cids.len() >= CID_MIN && cid_chars as f64 >= CID_MIN_SHARE * total as f64 {
        return Some(Failure::new(
            "raw glyph codes instead of text",
            Some(format!("{} (cid:NN) codes", cids.len())),
            None,
        ));
    }

    let (mut found, mut candidates) = (0, 0);
    let mut decoded = String::new();
    for word in text.split_whitespace().filter(|word| !word.is_ascii()) {
        candidates += 1;
        if let Some(fixed) = unmangled(word) {
            found += 1;
            decoded.push_str(&fixed);
            decoded.push(' ');
        }
    }
    if found >= MOJIBAKE_MIN_WORDS && found as f64 >= MOJIBAKE_MIN_SHARE * candidates as f64 {
        return Some(Failure::new(
            "mojibake, UTF-8 read as Windows-1252",
            Some(format!("{found} words")),
            dominant_script(&decoded),
        ));
    }

    let letters = visible.iter().filter(|&&ch| is_letter(ch)).count();
    if let Some(script) = reversed_rtl(text, letters) {
        return Some(Failure::new(
            "right-to-left text in reverse order",
            None,
            Some(script),
        ));
    }
    None
}

/// The advice for pages whose text layer failed in a `script` OCR with
/// `ocr.lang` `lang` does not read (with `--ocr` on or not).
pub(crate) fn unread_script_advice(script: Option<&str>, lang: &str, ocr: bool) -> String {
    let script = script.unwrap_or("this script");
    let other = lang_for_script(script);
    match (ocr, other) {
        (true, Some(other)) => format!(
            "OCR with ocr.lang {lang} does not read {script}, so the text layer is kept; set ocr.lang to {other} to OCR it"
        ),
        (true, None) => {
            format!("OCR with ocr.lang {lang} does not read {script}, so the text layer is kept")
        }
        (false, Some(other)) => {
            format!("consider re-running with --ocr and ocr.lang set to {other}")
        }
        (false, None) => format!("no OCR language reads {script}, so it is kept as extracted"),
    }
}
