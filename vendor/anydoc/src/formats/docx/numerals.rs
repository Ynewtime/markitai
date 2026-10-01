//! Number formats of `w:numFmt` that count in characters other than Latin
//! digits and letters.
//!
//! markitai: only decimal, letter and roman numbering was known; every other
//! format fell back to Arabic digits, so a Chinese outline's "一、" "二、" and
//! "（一）" read "1、" "2、" and "（1）", and an enclosed number "①" read "1".

/// A numeral system of the CJK and enclosed-digit formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Numeral {
    /// 一 二 三 … 十 十一 … (`chineseCounting` and its Taiwanese and
    /// Japanese counterparts).
    Counting,
    /// 壹 贰 叁 … 拾 (`chineseLegalSimplified`).
    LegalSimplified,
    /// 壹 貳 參 … 拾 (`ideographLegalTraditional`, `japaneseLegal`).
    LegalTraditional,
    /// 一 二 三 … 一〇, one character for each digit (`ideographDigital`).
    Digits,
    /// 甲 乙 丙 丁 … (`ideographTraditional`).
    HeavenlyStems,
    /// 子 丑 寅 卯 … (`ideographZodiac`).
    EarthlyBranches,
    /// ① ② ③ … (`decimalEnclosedCircle`).
    EnclosedCircle,
    /// ⑴ ⑵ ⑶ … (`decimalEnclosedParen`).
    EnclosedParen,
    /// ⒈ ⒉ ⒊ … (`decimalEnclosedFullstop`).
    EnclosedFullstop,
    /// １ ２ ３ … (`decimalFullWidth`).
    FullWidth,
    /// 01 02 03 … (`decimalZero`).
    ZeroPadded,
    /// 1st 2nd 3rd … (`ordinal`).
    Ordinal,
}

impl Numeral {
    /// The numeral system a `w:numFmt` value names, when it is one of these.
    pub fn from_format(format: &str) -> Option<Numeral> {
        Some(match format {
            "chineseCounting"
            | "chineseCountingThousand"
            | "taiwaneseCounting"
            | "taiwaneseCountingThousand"
            | "japaneseCounting" => Numeral::Counting,
            "chineseLegalSimplified" => Numeral::LegalSimplified,
            "ideographLegalTraditional" | "japaneseLegal" => Numeral::LegalTraditional,
            "ideographDigital" | "taiwaneseDigital" | "japaneseDigitalTenThousand" => {
                Numeral::Digits
            }
            "ideographTraditional" => Numeral::HeavenlyStems,
            "ideographZodiac" | "ideographZodiacTraditional" => Numeral::EarthlyBranches,
            "decimalEnclosedCircle" | "decimalEnclosedCircleChinese" => Numeral::EnclosedCircle,
            "decimalEnclosedParen" => Numeral::EnclosedParen,
            "decimalEnclosedFullstop" => Numeral::EnclosedFullstop,
            "decimalFullWidth" | "decimalFullWidth2" => Numeral::FullWidth,
            "decimalZero" => Numeral::ZeroPadded,
            "ordinal" => Numeral::Ordinal,
            _ => return None,
        })
    }

    /// The ordinal `n` written in this system, without the level's
    /// punctuation. A number beyond what the system can write is Arabic.
    pub fn ordinal(self, n: u64) -> String {
        let text = match self {
            Numeral::Counting => positional(n, &COUNTING, &['十', '百', '千'], true),
            Numeral::LegalSimplified => {
                positional(n, &LEGAL_SIMPLIFIED, &['拾', '佰', '仟'], false)
            }
            Numeral::LegalTraditional => {
                positional(n, &LEGAL_TRADITIONAL, &['拾', '佰', '仟'], false)
            }
            Numeral::Digits => Some(
                n.to_string()
                    .chars()
                    .map(|digit| IDEOGRAPH_DIGITS[digit.to_digit(10).unwrap_or(0) as usize])
                    .collect(),
            ),
            Numeral::HeavenlyStems => {
                pick(n, &['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'])
            }
            Numeral::EarthlyBranches => {
                pick(n, &['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'])
            }
            // The circled numbers sit in three blocks of the Unicode chart.
            Numeral::EnclosedCircle => run_of(0x2460, 1, 20, n)
                .or_else(|| run_of(0x3251, 21, 35, n))
                .or_else(|| run_of(0x32B1, 36, 50, n)),
            Numeral::EnclosedParen => run_of(0x2474, 1, 20, n),
            Numeral::EnclosedFullstop => run_of(0x2488, 1, 20, n),
            Numeral::FullWidth => Some(
                n.to_string()
                    .chars()
                    .filter_map(|digit| char::from_u32(0xFF10 + digit.to_digit(10)?))
                    .collect(),
            ),
            Numeral::ZeroPadded => Some(format!("{n:02}")),
            Numeral::Ordinal => Some(english_ordinal(n)),
        };
        text.unwrap_or_else(|| n.to_string())
    }
}

const COUNTING: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
const LEGAL_SIMPLIFIED: [char; 10] = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖'];
const LEGAL_TRADITIONAL: [char; 10] = ['零', '壹', '貳', '參', '肆', '伍', '陸', '柒', '捌', '玖'];
const IDEOGRAPH_DIGITS: [char; 10] = ['〇', '一', '二', '三', '四', '五', '六', '七', '八', '九'];

/// Ordinal `n` written as the character `n - first_n` places after `first`,
/// when `n` lies in `first_n..=last_n`.
fn run_of(first: u32, first_n: u64, last_n: u64, n: u64) -> Option<String> {
    if !(first_n..=last_n).contains(&n) {
        return None;
    }
    char::from_u32(first + (n - first_n) as u32).map(String::from)
}

/// The `n`th of `symbols` (1-based).
fn pick(n: u64, symbols: &[char]) -> Option<String> {
    let index = usize::try_from(n.checked_sub(1)?).ok()?;
    symbols.get(index).map(|symbol| symbol.to_string())
}

/// 1–9999 in a positional system with tens, hundreds and thousands units:
/// one 零 stands for any run of zeros between two digits, and a bare ten is
/// 十 rather than 一十 where the system allows it.
fn positional(n: u64, digits: &[char; 10], units: &[char; 3], bare_ten: bool) -> Option<String> {
    if n == 0 {
        return Some(digits[0].to_string());
    }
    if n > 9999 {
        return None;
    }
    let places = [
        (n / 1000, Some(units[2])),
        (n / 100 % 10, Some(units[1])),
        (n / 10 % 10, Some(units[0])),
        (n % 10, None),
    ];
    let mut out = String::new();
    let mut gap = false;
    for (digit, unit) in places {
        if digit == 0 {
            gap = !out.is_empty();
            continue;
        }
        if gap {
            out.push(digits[0]);
            gap = false;
        }
        if !(bare_ten && digit == 1 && unit == Some(units[0]) && out.is_empty()) {
            out.push(digits[digit as usize]);
        }
        out.extend(unit);
    }
    Some(out)
}

fn english_ordinal(n: u64) -> String {
    let suffix = match (n % 100, n % 10) {
        (11..=13, _) => "th",
        (_, 1) => "st",
        (_, 2) => "nd",
        (_, 3) => "rd",
        _ => "th",
    };
    format!("{n}{suffix}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_writes_chinese_numerals() {
        let written: Vec<String> =
            [1, 9, 10, 11, 19, 20, 21, 99, 100, 101, 110, 111, 1000, 1001, 1010, 2024]
                .into_iter()
                .map(|n| Numeral::Counting.ordinal(n))
                .collect();
        assert_eq!(
            written,
            [
                "一",
                "九",
                "十",
                "十一",
                "十九",
                "二十",
                "二十一",
                "九十九",
                "一百",
                "一百零一",
                "一百一十",
                "一百一十一",
                "一千",
                "一千零一",
                "一千零一十",
                "二千零二十四",
            ]
        );
        // Beyond four digits the system is not attempted.
        assert_eq!(Numeral::Counting.ordinal(10000), "10000");
    }

    #[test]
    fn legal_numerals_keep_the_leading_digit_of_ten() {
        assert_eq!(Numeral::LegalSimplified.ordinal(10), "壹拾");
        assert_eq!(Numeral::LegalSimplified.ordinal(23), "贰拾叁");
        assert_eq!(Numeral::LegalTraditional.ordinal(2), "貳");
    }

    #[test]
    fn digit_by_digit_and_cyclic_systems() {
        assert_eq!(Numeral::Digits.ordinal(10), "一〇");
        assert_eq!(Numeral::HeavenlyStems.ordinal(3), "丙");
        assert_eq!(Numeral::HeavenlyStems.ordinal(11), "11");
        assert_eq!(Numeral::EarthlyBranches.ordinal(12), "亥");
    }

    #[test]
    fn enclosed_and_padded_digits() {
        assert_eq!(Numeral::EnclosedCircle.ordinal(1), "①");
        assert_eq!(Numeral::EnclosedCircle.ordinal(20), "⑳");
        assert_eq!(Numeral::EnclosedCircle.ordinal(21), "㉑");
        assert_eq!(Numeral::EnclosedCircle.ordinal(36), "㊱");
        assert_eq!(Numeral::EnclosedCircle.ordinal(51), "51");
        assert_eq!(Numeral::EnclosedParen.ordinal(2), "⑵");
        assert_eq!(Numeral::EnclosedFullstop.ordinal(3), "⒊");
        assert_eq!(Numeral::FullWidth.ordinal(12), "１２");
        assert_eq!(Numeral::ZeroPadded.ordinal(7), "07");
        assert_eq!(Numeral::ZeroPadded.ordinal(12), "12");
    }

    #[test]
    fn english_ordinals_take_their_suffix() {
        let written: Vec<String> = [1, 2, 3, 4, 11, 12, 13, 21, 22, 101, 111]
            .into_iter()
            .map(|n| Numeral::Ordinal.ordinal(n))
            .collect();
        assert_eq!(
            written,
            ["1st", "2nd", "3rd", "4th", "11th", "12th", "13th", "21st", "22nd", "101st", "111th"]
        );
    }

    #[test]
    fn formats_name_their_numeral_system() {
        assert_eq!(Numeral::from_format("chineseCounting"), Some(Numeral::Counting));
        assert_eq!(
            Numeral::from_format("decimalEnclosedCircleChinese"),
            Some(Numeral::EnclosedCircle)
        );
        assert_eq!(Numeral::from_format("decimal"), None);
        assert_eq!(Numeral::from_format("lowerRoman"), None);
    }
}
