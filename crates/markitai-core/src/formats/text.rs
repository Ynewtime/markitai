use crate::{Document, Error, Result};
use encoding_rs::{BIG5, EUC_JP, EUC_KR, Encoding, GB18030, SHIFT_JIS};
use serde_json::Value;

pub(super) fn decode(bytes: &[u8]) -> Result<String> {
    decode_with(bytes, |_| None).map(|(text, _)| text)
}

/// Decode plain text or a delimited table, which carry no encoding label.
///
/// BOM-marked and valid UTF-8 input keeps the single validation pass of
/// [`decode`]. Other bytes are offered to the East Asian legacy encodings that
/// local tools still write (Chinese and Japanese Excel and Notepad exports);
/// see [`guess_legacy`]. The second value is a warning when the encoding
/// could not be told apart.
pub(super) fn decode_legacy(bytes: &[u8]) -> Result<(String, Option<String>)> {
    decode_with(bytes, guess_legacy)
}

/// Plain text or Markdown bytes as a document.
pub(super) fn plain(bytes: &[u8]) -> Result<Document> {
    let (markdown, warning) = decode_legacy(bytes)?;
    Ok(Document {
        markdown,
        warnings: warning.into_iter().collect(),
        ..Document::default()
    })
}

/// CSV or TSV bytes as a table document. `default` is the delimiter the
/// extension names; the decoded text may show another (see [`sniff_delimiter`]).
pub(super) fn delimited_bytes(bytes: &[u8], default: u8) -> Result<Document> {
    let (source, warning) = decode_legacy(bytes)?;
    let mut document = delimited(&source, sniff_delimiter(&source, default))?;
    document.warnings.extend(warning);
    Ok(document)
}

/// Delimiters a delimited text file is read with, in the order that breaks a
/// tie after the extension's own.
const DELIMITERS: [u8; 4] = *b",\t;|";
/// Records the delimiter is judged on.
const SNIFF_RECORDS: usize = 50;
/// Bytes the judged records are cut from, so one long line costs little.
const SNIFF_BYTES: usize = 64 * 1024;
/// Share of the judged records that must have the most common field count, as
/// a numerator over [`SHARE_DEN`]: a title line or a total row may differ.
const SHARE_NUM: usize = 4;
const SHARE_DEN: usize = 5;

/// What one delimiter makes of the first records.
struct Reading {
    delimiter: u8,
    /// The most common field count (the larger one on a tie).
    width: usize,
    /// Records with that count, and records judged.
    agree: usize,
    records: usize,
    /// Whether every field holding a comma reads as a number written with a
    /// decimal comma (`1.234,50`, `99,00`), and at least one does.
    decimal_commas: bool,
}

/// The delimiter of delimited text, which carries none of its own: a
/// semicolon-separated export from a European Excel, a tab-separated
/// "Unicode Text" file named `.csv`, a pipe-separated dump.
///
/// Each of `,` `\t` `;` `|` splits the first [`SNIFF_RECORDS`] records,
/// quote-aware (a delimiter inside a quoted field does not split it). A
/// delimiter qualifies when it splits records into more than one field and
/// at least four in five records have its most common field count. The
/// qualifying delimiter whose count the most records share wins, then the
/// one that splits into more fields, then `default` (the extension's: `,` for
/// `.csv`, a tab for `.tsv`). When none qualifies `default` stays.
///
/// Commas inside a semicolon or tab file are decimal separators when every
/// field holding one reads as such a number; the comma then never competes,
/// so `José;Zürich;1.234,50` keeps `1.234,50` whole even in a file whose
/// every row has one amount.
pub(super) fn sniff_delimiter(source: &str, default: u8) -> u8 {
    let sample = sniff_sample(source);
    let truncated = sample.len() < source.len();
    let readings: Vec<Reading> = DELIMITERS
        .iter()
        .filter_map(|&delimiter| reading(sample, delimiter, truncated))
        .collect();
    let decimal_commas = readings
        .iter()
        .any(|r| r.delimiter != b',' && qualifies(r) && r.decimal_commas);
    let mut best: Option<&Reading> = None;
    for candidate in readings.iter().filter(|r| qualifies(r)) {
        if candidate.delimiter == b',' && decimal_commas {
            continue;
        }
        let better = match best {
            None => true,
            Some(current) => {
                // Shares compared exactly, as cross products.
                let (a, b) = (
                    candidate.agree * current.records,
                    current.agree * candidate.records,
                );
                a > b
                    || (a == b && candidate.width > current.width)
                    || (a == b
                        && candidate.width == current.width
                        && candidate.delimiter == default)
            }
        };
        if better {
            best = Some(candidate);
        }
    }
    best.map_or(default, |reading| reading.delimiter)
}

fn qualifies(reading: &Reading) -> bool {
    reading.width > 1 && reading.agree * SHARE_DEN >= reading.records * SHARE_NUM
}

/// The first [`SNIFF_RECORDS`] lines of `source`, at most [`SNIFF_BYTES`]
/// and cut at a line end where there is one; a quoted field may still run
/// past the cut, which the caller then leaves out of the judgement.
fn sniff_sample(source: &str) -> &str {
    let mut end = source.len().min(SNIFF_BYTES);
    while !source.is_char_boundary(end) {
        end -= 1;
    }
    let head = &source[..end];
    let mut lines = 0;
    for (at, byte) in head.bytes().enumerate() {
        if byte == b'\n' {
            lines += 1;
            if lines == SNIFF_RECORDS {
                return &head[..=at];
            }
        }
    }
    if end < source.len()
        && let Some(at) = head.rfind('\n')
    {
        return &head[..=at];
    }
    head
}

fn reading(sample: &str, delimiter: u8, truncated: bool) -> Option<Reading> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .flexible(true)
        .from_reader(sample.as_bytes());
    let mut widths = Vec::new();
    let mut commas = 0usize;
    let mut decimal = true;
    for record in reader.records().take(SNIFF_RECORDS) {
        let Ok(record) = record else {
            break;
        };
        // A blank line is no record of the table.
        if record.len() == 1 && record[0].trim().is_empty() {
            continue;
        }
        widths.push(record.len());
        for field in record.iter().filter(|field| field.contains(',')) {
            commas += 1;
            decimal &= is_decimal_comma_number(field);
        }
    }
    // The last record of a cut sample may be a quoted field cut short.
    if truncated {
        widths.pop();
    }
    if widths.is_empty() {
        return None;
    }
    let mut counts: std::collections::BTreeMap<usize, usize> = std::collections::BTreeMap::new();
    for &width in &widths {
        *counts.entry(width).or_default() += 1;
    }
    let (width, agree) = counts
        .iter()
        .max_by_key(|&(&width, &count)| (count, width))
        .map(|(&width, &count)| (width, count))?;
    Some(Reading {
        delimiter,
        width,
        agree,
        records: widths.len(),
        decimal_commas: decimal && commas > 0,
    })
}

/// A number written with a decimal comma: an optional sign or currency
/// symbol, digits optionally grouped by `.`, spaces or apostrophes, one
/// comma, then digits (`-1.234,50`, `99,00`, `€ 12,5`, `12,5 %`).
fn is_decimal_comma_number(field: &str) -> bool {
    let field = field.trim().trim_start_matches(['-', '+', '(']);
    let field = field.trim_end_matches([')', '%', ' ', '€', '$', '£']);
    let field = field.trim_start_matches(['€', '$', '£', ' ']);
    let Some((whole, fraction)) = field.split_once(',') else {
        return false;
    };
    !whole.is_empty()
        && !fraction.is_empty()
        && whole.starts_with(|c: char| c.is_ascii_digit())
        && whole
            .chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | ' ' | '\'' | '\u{a0}'))
        && fraction.chars().all(|c| c.is_ascii_digit())
}

fn decode_with(
    bytes: &[u8],
    legacy: impl FnOnce(&[u8]) -> Option<(String, Option<String>)>,
) -> Result<(String, Option<String>)> {
    let (encoding, offset) = Encoding::for_bom(bytes).unwrap_or((encoding_rs::UTF_8, 0));
    let bytes = &bytes[offset..];
    let (decoded, malformed) = encoding.decode_without_bom_handling(bytes);
    if !malformed {
        return Ok((decoded.into_owned(), None));
    }
    if offset != 0 {
        return Err(Error::Conversion(
            "Invalid Unicode text after byte-order mark".into(),
        ));
    }
    if let Some(guessed) = legacy(bytes) {
        return Ok(guessed);
    }
    // Legacy Western text is a supported input; never silently replace bytes.
    let (decoded, malformed) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(bytes);
    if malformed {
        Err(Error::Conversion(
            "Input text is not valid UTF-8, UTF-16 or Windows-1252".into(),
        ))
    } else {
        Ok((decoded.into_owned(), None))
    }
}

/// Multibyte encodings tried for unlabeled non-UTF-8 text, in the order that
/// breaks an exact score tie.
const LEGACY: [&Encoding; 5] = [GB18030, BIG5, SHIFT_JIS, EUC_JP, EUC_KR];
/// Minimum share of a reading's non-ASCII characters in its frequent repertoire.
const LEGACY_SCORE: f64 = 0.7;
/// Minimum lead of the best reading over every other one.
const LEGACY_MARGIN: f64 = 0.2;
/// Hangul syllables that make a reading confined to the Hangul rows Korean.
const HANGUL_MIN: usize = 12;
/// Big5 characters after which a reading without a low trail byte is not Big5.
const BIG5_MIN: usize = 8;

/// Choose a multibyte reading of non-UTF-8 text, or `None` for Windows-1252.
///
/// Readings with any invalid or unmapped sequence are rejected; Western text
/// rarely survives, because an accented letter followed by a space, digit or
/// punctuation is invalid in all five encodings. Each survivor is scored by
/// the share of its non-ASCII characters in the frequent repertoire of its
/// script (see [`Profile`]), and wins only with [`LEGACY_SCORE`] and a
/// [`LEGACY_MARGIN`] over the other readings, Windows-1252 included (see
/// [`western`]). A plausible reading without that margin keeps Windows-1252
/// with a warning, unless the Western reading is at least as plausible.
fn guess_legacy(bytes: &[u8]) -> Option<(String, Option<String>)> {
    let mut best: Option<(f64, &'static Encoding, std::borrow::Cow<'_, str>)> = None;
    let mut runner_up = 0.0_f64;
    for encoding in LEGACY {
        let Some(text) = encoding.decode_without_bom_handling_and_without_replacement(bytes) else {
            continue;
        };
        let profile = Profile::read(encoding, bytes);
        if encoding == EUC_KR && profile.hangul_only() {
            return Some((text.into_owned(), None));
        }
        let score = profile.score();
        match &best {
            Some((top, ..)) if score <= *top => runner_up = runner_up.max(score),
            _ => {
                if let Some((top, ..)) = best.replace((score, encoding, text)) {
                    runner_up = runner_up.max(top);
                }
            }
        }
    }
    let (score, encoding, text) = best?;
    if score < LEGACY_SCORE {
        return None;
    }
    let western = western(bytes);
    if score - runner_up.max(western) >= LEGACY_MARGIN {
        return Some((text.into_owned(), None));
    }
    if western >= score {
        return None;
    }
    let (text, _) = encoding_rs::WINDOWS_1252.decode_without_bom_handling(bytes);
    Some((
        text.into_owned(),
        Some(format!(
            "Text is not UTF-8 and its encoding could not be determined; it was read as Windows-1252, but it may be {}.",
            encoding.name()
        )),
    ))
}

/// Character classes of one multibyte reading, from its byte structure.
#[derive(Default)]
struct Profile {
    /// Non-ASCII characters.
    total: usize,
    /// Punctuation and, except in EUC-JP, the first-level ideographs.
    frequent: usize,
    /// Kana in EUC-JP, KS X 1001 Hangul in EUC-KR.
    script: usize,
    /// EUC-JP first-level kanji, which count only beside kana: Chinese reads
    /// as kana-free EUC-JP, and Japanese text without kana is rare.
    kanji: usize,
    /// Big5 characters with a trail byte below 0x7F. About two in five Big5
    /// characters have one; GB2312, JIS X 0208 and KS X 1001 text, which also
    /// reads as Big5, never does.
    low_trail: usize,
}

#[derive(PartialEq)]
enum Class {
    Other,
    Frequent,
    Script,
    Kanji,
}

impl Profile {
    fn read(encoding: &'static Encoding, bytes: &[u8]) -> Self {
        let mut profile = Self::default();
        let mut i = 0;
        while let Some(&lead) = bytes.get(i) {
            if lead < 0x80 {
                i += 1;
                continue;
            }
            let trail = bytes.get(i + 1).copied().unwrap_or(0);
            let width = if encoding == GB18030 {
                match (lead, trail) {
                    (0x80 | 0xFF, _) => 1,
                    (_, 0x30..=0x39) => 4,
                    _ => 2,
                }
            } else if encoding == SHIFT_JIS {
                if matches!(lead, 0x80 | 0xA1..=0xDF) {
                    1
                } else {
                    2
                }
            } else if encoding == EUC_JP && lead == 0x8F {
                3
            } else {
                2
            };
            let class = if width == 2 {
                class(encoding, lead, trail)
            } else {
                Class::Other
            };
            profile.total += 1;
            profile.frequent += usize::from(class == Class::Frequent);
            profile.script += usize::from(class == Class::Script);
            profile.kanji += usize::from(class == Class::Kanji);
            profile.low_trail += usize::from(encoding == BIG5 && trail < 0x7F);
            i += width;
        }
        if encoding == BIG5 && profile.total >= BIG5_MIN && profile.low_trail == 0 {
            profile.frequent = 0;
        }
        profile
    }

    fn score(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        let kanji = if self.script > 0 { self.kanji } else { 0 };
        (self.frequent + self.script + kanji) as f64 / self.total as f64
    }

    /// KS X 1001 puts its 2,350 Hangul syllables in rows B0-C8, which GB2312
    /// and JIS X 0208 fill with their first ideographs, so Korean also reads
    /// as plausible Chinese or Japanese. Chinese and Japanese text of any
    /// length uses later ideograph rows or kana too, so a reading made only of
    /// these syllables and punctuation is Korean.
    fn hangul_only(&self) -> bool {
        self.script >= HANGUL_MIN && self.frequent + self.script == self.total
    }
}

/// The plausibility of the Windows-1252 reading: the share of non-ASCII bytes
/// that stand alone between ASCII bytes and read as a letter or common
/// punctuation, as accented letters, curly quotes and dashes do. Multibyte
/// text puts its non-ASCII bytes in runs or in rarely used symbols.
fn western(bytes: &[u8]) -> f64 {
    let mut total = 0_usize;
    let mut alone = 0_usize;
    for (i, &byte) in bytes.iter().enumerate() {
        if byte < 0x80 {
            continue;
        }
        total += 1;
        alone += usize::from(
            (i == 0 || bytes[i - 1] < 0x80)
                && bytes.get(i + 1).is_none_or(|&next| next < 0x80)
                // Unassigned, modifier and rarely written symbol bytes.
                && !matches!(
                    byte,
                    0x81 | 0x83 | 0x86..=0x89 | 0x8B | 0x8D | 0x8F | 0x90 | 0x98 | 0x9B | 0x9D
                        | 0xA4 | 0xA6 | 0xA8 | 0xAC | 0xAD | 0xAF | 0xB8
                ),
        );
    }
    alone as f64 / total.max(1) as f64
}

/// The part of a two-byte character's set that ordinary text in its script
/// mostly uses: punctuation, kana, Hangul and first-level (most frequent)
/// ideographs.
fn class(encoding: &'static Encoding, lead: u8, trail: u8) -> Class {
    let high_trail = (0xA1..=0xFE).contains(&trail);
    let frequent = if encoding == GB18030 {
        // GB2312 punctuation and full-width forms, level-1 hanzi.
        high_trail && matches!(lead, 0xA1 | 0xA3 | 0xB0..=0xD7)
    } else if encoding == BIG5 {
        // Punctuation and level-1 hanzi.
        matches!(lead, 0xA1 | 0xA4..=0xC6)
    } else if encoding == SHIFT_JIS {
        // Punctuation, hiragana, katakana and level-1 kanji.
        matches!(lead, 0x81..=0x83 | 0x88..=0x98)
    } else if !high_trail {
        false
    } else if encoding == EUC_JP {
        match lead {
            0xA4 | 0xA5 => return Class::Script,
            0xB0..=0xCF => return Class::Kanji,
            lead => lead == 0xA1,
        }
    } else {
        // EUC-KR.
        match lead {
            0xB0..=0xC8 => return Class::Script,
            lead => lead == 0xA1,
        }
    };
    if frequent {
        Class::Frequent
    } else {
        Class::Other
    }
}

pub(super) fn cell(value: &str) -> String {
    value
        .trim()
        .replace('\\', "\\\\")
        .replace('|', "\\|")
        .replace("\r\n", "<br>")
        .replace(['\n', '\r'], "<br>")
}

pub(super) fn table(rows: &[Vec<String>], header: bool) -> String {
    let width = rows.iter().map(Vec::len).max().unwrap_or(0);
    if width == 0 {
        return String::new();
    }
    let row_text = |row: &[String]| {
        format!(
            "| {} |\n",
            (0..width)
                .map(|i| row.get(i).cloned().unwrap_or_default())
                .collect::<Vec<_>>()
                .join(" | ")
        )
    };
    let mut output = String::new();
    let (first, rest) = if header {
        (rows[0].clone(), &rows[1..])
    } else {
        (vec![String::new(); width], rows)
    };
    output.push_str(&row_text(&first));
    output.push_str(&row_text(&vec!["---".into(); width]));
    for row in rest {
        output.push_str(&row_text(row));
    }
    output
}

pub(super) fn delimited(source: &str, delimiter: u8) -> Result<Document> {
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(false)
        .flexible(true)
        .from_reader(source.as_bytes());
    let rows = reader
        .records()
        .map(|row| {
            row.map(|row| row.iter().map(cell).collect::<Vec<_>>())
                .map_err(|e| Error::Conversion(format!("Invalid delimited text: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    // The reference cuts CSV rows to the header's width and leaves `|` and line
    // breaks raw, which loses fields and breaks the table; every row is kept
    // whole and escaped instead.
    let markdown = table(&rows, true)
        .lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n");
    Ok(Document {
        markdown,
        ..Document::default()
    })
}

fn source_text(value: &Value) -> Result<String> {
    match value {
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => parts
            .iter()
            .map(|part| {
                part.as_str()
                    .ok_or_else(|| Error::Conversion("Notebook source must contain strings".into()))
            })
            .collect::<Result<Vec<_>>>()
            .map(|parts| parts.concat()),
        _ => Err(Error::Conversion(
            "Notebook cell source must be a string or an array of strings".into(),
        )),
    }
}

pub(super) fn fence(source: &str, language: &str) -> String {
    let longest = source.split(|c| c != '`').map(str::len).max().unwrap_or(0);
    let marker = "`".repeat(longest.saturating_add(1).max(3));
    format!(
        "{marker}{language}\n{}\n{marker}",
        source.trim_end_matches('\n')
    )
}

#[path = "native/notebook.rs"]
mod notebook_reader;

/// A notebook's cells, with what the code cells printed and drew (see
/// `native/notebook.rs`).
pub(super) fn notebook(source: &str) -> Result<Document> {
    notebook_reader::read(source)
}

pub(super) fn json(source: &str) -> Result<Document> {
    let value: Value = serde_json::from_str(source)?;
    Ok(Document {
        markdown: fence(&serde_json::to_string_pretty(&value)?, "json"),
        ..Document::default()
    })
}

pub(super) fn xml(source: &str) -> Result<Document> {
    use quick_xml::events::Event;
    let mut reader = quick_xml::Reader::from_str(source);
    let mut depth = 0usize;
    let mut roots = 0usize;
    let mut blocks = Vec::new();
    let mut pending = String::new();
    let malformed =
        |error: &dyn std::fmt::Display| Error::Conversion(format!("Malformed XML: {error}"));
    loop {
        let event = reader.read_event().map_err(|e| malformed(&e))?;
        match &event {
            Event::Start(_) | Event::Empty(_) | Event::End(_) | Event::Eof => {
                if !pending.trim().is_empty() {
                    if depth == 0 {
                        return Err(Error::Conversion(
                            "XML text outside document element".into(),
                        ));
                    }
                    blocks.push(pending.trim().to_owned());
                }
                pending.clear();
            }
            _ => (),
        }
        let empty = matches!(&event, Event::Empty(_));
        match event {
            Event::Start(element) | Event::Empty(element) => {
                if depth == 0 {
                    roots += 1;
                }
                let level = depth + 1;
                if level > 256 {
                    return Err(Error::Conversion("XML nesting exceeds 256 elements".into()));
                }
                let local_name = element.local_name();
                let name = reader
                    .decoder()
                    .decode(local_name.as_ref())
                    .map_err(|e| malformed(&e))?;
                if level <= 6 {
                    blocks.push(format!("{} {name}", "#".repeat(level)));
                } else {
                    blocks.push(format!("{}- **{name}**", "  ".repeat(level - 7)));
                }
                for attr in element.attributes() {
                    let attr = attr.map_err(|e| malformed(&e))?;
                    if attr.key.as_ref() == b"xmlns" || attr.key.as_ref().starts_with(b"xmlns:") {
                        continue;
                    }
                    let key = attr.key.local_name();
                    let key = reader
                        .decoder()
                        .decode(key.as_ref())
                        .map_err(|e| malformed(&e))?;
                    let value = attr
                        .decoded_and_normalized_value(
                            quick_xml::XmlVersion::Implicit1_0,
                            reader.decoder(),
                        )
                        .map_err(|e| malformed(&e))?;
                    blocks.push(format!("{key}: {value}"));
                }
                if !empty {
                    depth += 1;
                }
            }
            Event::End(_) => {
                depth = depth.saturating_sub(1);
            }
            Event::Text(text) => pending.push_str(
                &text
                    .xml_content(quick_xml::XmlVersion::Implicit1_0)
                    .map_err(|e| malformed(&e))?,
            ),
            Event::CData(text) => pending.push_str(
                &text
                    .xml_content(quick_xml::XmlVersion::Implicit1_0)
                    .map_err(|e| malformed(&e))?,
            ),
            Event::GeneralRef(reference) => {
                let name = reference.decode().map_err(|e| malformed(&e))?;
                let encoded = format!("&{name};");
                pending
                    .push_str(&quick_xml::escape::unescape(&encoded).map_err(|e| malformed(&e))?);
            }
            Event::Eof => break,
            Event::DocType(_) => {
                return Err(Error::Conversion(
                    "XML document types and external entities are not supported".into(),
                ));
            }
            _ => (),
        }
    }
    if depth != 0 || roots != 1 {
        return Err(Error::Conversion(
            "XML must contain one complete document element".into(),
        ));
    }
    let source = source.trim();
    if source.len() < 20 * 1024 {
        blocks.push(fence(source, "xml"));
    }
    Ok(Document {
        markdown: blocks.join("\n\n"),
        ..Document::default()
    })
}

#[path = "native/eml.rs"]
mod eml;

pub(super) fn email(bytes: &[u8]) -> Result<Document> {
    eml::extract(bytes)
}

pub(super) fn email_with_attachments(
    bytes: &[u8],
) -> Result<(Document, std::collections::HashSet<String>)> {
    eml::extract_with_attachments(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encodings_keep_non_ascii_text() {
        assert_eq!(decode(b"caf\xe9").unwrap(), "café");
        assert_eq!(decode(&[0xff, 0xfe, 0x2d, 0x4e]).unwrap(), "中");
        assert_eq!(decode(b"\xef\xbb\xbfhello").unwrap(), "hello");
        assert!(decode(&[0xff, 0xfe, 0x00]).is_err());
    }
    fn legacy(text: &str, encoding: &'static Encoding) -> (String, Option<String>) {
        let (bytes, _, unmappable) = encoding.encode(text);
        assert!(!unmappable, "{text}");
        decode_legacy(&bytes).unwrap()
    }

    #[test]
    fn legacy_text_detects_east_asian_encodings() {
        let cases: [(&str, &'static Encoding); 6] = [
            ("姓名,城市\n张三,北京\n李四,上海\n", encoding_rs::GBK),
            ("你好，世界！这是一个测试文件。\n", GB18030),
            ("繁體中文內容測試編碼偵測。\n", BIG5),
            ("こんにちは世界。これはテストです。\n", SHIFT_JIS),
            ("日本語のテキストです。漢字とかなを含みます。\n", EUC_JP),
            ("안녕하세요, 세계! 이것은 인코딩 테스트입니다.\n", EUC_KR),
        ];
        for (text, encoding) in cases {
            assert_eq!(
                legacy(text, encoding),
                (text.to_owned(), None),
                "{}",
                encoding.name()
            );
        }
        // Mostly ASCII with a few Han characters, and a single one.
        let mixed = "id,name,city,note\n1,Alice,Paris,ok\n2,Bob,London,ok\n3,王伟,Beijing,ok\n";
        assert_eq!(legacy(mixed, encoding_rs::GBK).0, mixed);
        assert_eq!(legacy("中\n", encoding_rs::GBK).0, "中\n");
        // Katakana-only Shift_JIS and a short Korean line with Hangul only.
        assert_eq!(legacy("カタカナ\n", SHIFT_JIS).0, "カタカナ\n");
        let korean = "대한민국 서울특별시 종로구\n";
        assert_eq!(legacy(korean, EUC_KR).0, korean);
    }

    #[test]
    fn legacy_text_keeps_western_text_as_windows_1252() {
        for text in [
            "café naïve résumé\n",
            "Müller wohnt in Zürich und fährt mit dem Fahrrad zur Arbeit.\n",
            "A informação é essencial para a comunicação.\n",
            "Let’s say “hello” — it won’t hurt…\n",
            "Temperature: 21 °C\n",
            "¡Comencemos!\n",
            "Æsir og Ásynjur\n",
            "Nom;Prénom\nDupont;Hélène\nMüller;Jürgen\n",
            "ação\n",
            "Zürich",
        ] {
            assert_eq!(
                legacy(text, encoding_rs::WINDOWS_1252),
                (text.to_owned(), None),
                "{text}"
            );
        }
    }

    #[test]
    fn legacy_text_warns_when_readings_tie() {
        // Two Hangul syllables are also two frequent GB2312 hanzi.
        let (text, warning) = legacy("한국\n", EUC_KR);
        assert_eq!(
            text,
            encoding_rs::WINDOWS_1252
                .decode_without_bom_handling(&EUC_KR.encode("한국\n").0)
                .0
        );
        assert!(warning.unwrap().contains("could not be determined"));
        // Undecodable in every multibyte encoding: Windows-1252 without a warning.
        assert_eq!(
            decode_legacy(b"caf\xe9 \x81").unwrap(),
            ("caf\u{e9} \u{81}".into(), None)
        );
        let windows_1252 =
            |bytes: &[u8]| (encoding_rs::WINDOWS_1252.decode(bytes).0.into_owned(), None);
        // A reading with one frequent hanzi in two is too weak, even unopposed.
        let weak = b"x \xd6\xd0\xe7\xe3 y";
        assert_eq!(decode_legacy(weak).unwrap(), windows_1252(weak));
        // A truncated last character rejects the GBK reading entirely.
        let (gbk, ..) = encoding_rs::GBK.encode("中文内容在这里");
        let truncated = [&gbk[..], b"\xd6"].concat();
        assert_eq!(decode_legacy(&truncated).unwrap(), windows_1252(&truncated));
    }

    #[test]
    fn western_reading_counts_lone_letters_and_punctuation() {
        assert_eq!(western(b"caf\xe9 na\xefve \x93q\x94"), 1.0);
        assert_eq!(western(b"a\xd6\xd0b"), 0.0);
        assert_eq!(western(b"\xd6\xd0"), 0.0);
        // The florin and unassigned bytes are not Western prose.
        assert_eq!(western(b"\x83J \x81"), 0.0);
        assert_eq!(western(b"plain"), 0.0);
    }

    #[test]
    fn legacy_documents_carry_the_warning() {
        let (bytes, ..) = EUC_KR.encode("한국");
        let csv = delimited_bytes(&[b"a,b\n", &bytes[..], b",x\n"].concat(), b',').unwrap();
        assert_eq!(csv.warnings.len(), 1);
        let txt = plain(&bytes).unwrap();
        assert_eq!(txt.warnings.len(), 1);
        assert!(plain("中文".as_bytes()).unwrap().warnings.is_empty());
    }

    #[test]
    fn csv_keeps_every_field_and_escapes_table_delimiters() {
        let doc = delimited("name,body\n\"a,b\",\"line\nbreak | text\"\nlast\n", b',').unwrap();
        assert_eq!(
            doc.markdown,
            "| name | body |\n| --- | --- |\n| a,b | line<br>break \\| text |\n| last |  |"
        );
        assert_eq!(
            delimited("a,b\n1\n2,3,4\n", b',').unwrap().markdown,
            "| a | b |  |\n| --- | --- | --- |\n| 1 |  |  |\n| 2 | 3 | 4 |"
        );
        assert!(delimited("", b',').unwrap().markdown.is_empty());
    }
    #[test]
    fn tsv_escapes_cells_and_retains_surplus_columns() {
        let doc = delimited("a\tb\nx|y\t\"line\nbreak\"\textra\n", b'\t').unwrap();
        assert!(doc.markdown.contains("| a | b |  |"));
        assert!(doc.markdown.contains("x\\|y | line<br>break | extra"));
    }
    #[test]
    fn the_delimiter_is_sniffed_from_the_first_records() {
        let sniff = |text: &str, default: u8| sniff_delimiter(text, default) as char;
        // A European Excel export, decimal commas kept whole.
        let euro = "Name;Ville;Montant\nJosé;Zürich;1.234,50\nÉlodie;São Paulo;99,00\n";
        assert_eq!(sniff(euro, b','), ';');
        assert_eq!(
            delimited_bytes(euro.as_bytes(), b',').unwrap().markdown,
            "| Name | Ville | Montant |\n| --- | --- | --- |\n| José | Zürich | 1.234,50 |\n| Élodie | São Paulo | 99,00 |"
        );
        // Without a header every row splits the same on its decimal comma
        // too; decimal commas never compete with the semicolon.
        assert_eq!(sniff("A;B;1,5\nC;D;2,5\n", b','), ';');
        assert_eq!(sniff("A;1,5;2,5\nB;3,5;4,5\n", b','), ';');
        assert_eq!(sniff("a\t1,5\nb\t2,25\n", b','), '\t');
        // Tabs in a `.csv`, commas in a `.tsv`, semicolons in a `.tsv`, pipes.
        assert_eq!(sniff("Name\tCity\nJosé\tZürich\n", b','), '\t');
        assert_eq!(sniff("a,b,c\n1,2,3\n", b'\t'), ',');
        assert_eq!(sniff("Name;Ville;Montant\nJosé;Zürich;1,234\n", b'\t'), ';');
        assert_eq!(sniff("a|b|c\n1|2|3\n4|5|6\n", b','), '|');
        // Quoted delimiters do not split, and ragged comma rows stay commas.
        assert_eq!(sniff("id,text\n1,\"a;b;c\"\n2,\"d;e\"\n", b','), ',');
        assert_eq!(sniff("a,b,c\n1,2\n3,4,5,6\n\n7,8,9\n", b','), ',');
        // Ambiguous or single-column text keeps the extension's delimiter.
        assert_eq!(sniff("a,b;c\n1,2;3\n", b','), ',');
        assert_eq!(sniff("a,b;c\n1,2;3\n", b';'), ';');
        assert_eq!(sniff("name\nAlice\nBob\n", b'\t'), '\t');
        assert_eq!(sniff("", b','), ',');
        // A title line before the records does not outvote them.
        let titled = "Report 2026\nx;y;z\n1;2;3\n4;5;6\n7;8;9\n10;11;12\n";
        assert_eq!(sniff(titled, b','), ';');
        // A quoted field the sample cuts short is left out of the count.
        let mut long = "a;b\n".repeat(60);
        long.push_str(&format!("\"{}", "x\n".repeat(100)));
        assert_eq!(sniff(&long, b','), ';');
    }

    #[test]
    fn a_utf16_unicode_text_export_named_csv_splits_on_tabs() {
        let mut bytes = vec![0xff, 0xfe];
        for unit in "Name\tCity\nJosé\tZürich\nLi\t北京\n".encode_utf16() {
            bytes.extend(unit.to_le_bytes());
        }
        assert_eq!(
            delimited_bytes(&bytes, b',').unwrap().markdown,
            "| Name | City |\n| --- | --- |\n| José | Zürich |\n| Li | 北京 |"
        );
        assert!(is_decimal_comma_number("-1.234,50") && is_decimal_comma_number("€ 12,5"));
        assert!(!is_decimal_comma_number("Paris, France") && !is_decimal_comma_number("1,"));
    }

    #[test]
    fn notebook_preserves_all_cell_types_and_long_fences() {
        let doc = notebook(r##"{"metadata":{"title":"Notebook","language_info":{"name":"python"}},"cells":[{"cell_type":"markdown","source":["# Title\n","text"]},{"cell_type":"code","source":"print('```')\n"},{"cell_type":"raw","source":"raw"}]}"##).unwrap();
        assert!(doc.markdown.contains("````python\nprint('```')\n````"));
        assert!(doc.markdown.ends_with("```\nraw\n```"));
        assert_eq!(doc.metadata["title"], "Notebook");
        assert!(notebook(r#"{"cells":null}"#).is_err());
    }
    #[test]
    fn xml_rejects_broken_or_external_entity_input() {
        assert!(
            xml("<root><child>text</child></root>")
                .unwrap()
                .markdown
                .starts_with("# root\n\n## child\n\ntext")
        );
        assert!(xml("<root>").is_err());
        assert!(xml("<a/><b/>").is_err());
        assert!(xml("<!DOCTYPE a SYSTEM 'https://example.test/x'><a/>").is_err());
        assert!(xml("outside<a/>").is_err());
        assert!(xml("<a>&unknown;</a>").is_err());
        let mixed =
            xml("<x:root xmlns:x=\"urn:test\" a=\"1&amp;2\">left &lt;<x:child/>right</x:root>")
                .unwrap();
        assert!(
            mixed
                .markdown
                .starts_with("# root\n\na: 1&2\n\nleft <\n\n## child\n\nright\n\n```xml")
        );
    }
    #[test]
    fn email_decodes_mime_and_subject() {
        let doc = email(b"Subject: =?UTF-8?B?5Lit5paH?=\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nHello body").unwrap();
        assert_eq!(doc.metadata["title"], "中文");
        assert!(doc.markdown.contains("Hello body"));
        let doc = email(b"From: from@example.test\nSubject: Example\nContent-Type: text/plain\n\nFirst line\nSecond line").unwrap();
        assert_eq!(
            doc.markdown,
            "# Email Message\n\n**From:** from@example.test\n**Subject:** Example\n\n## Content\n\nFirst line\nSecond line"
        );
    }
}
