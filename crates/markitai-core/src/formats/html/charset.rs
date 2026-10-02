//! The character encoding of HTML bytes that arrive without a transport label.
//!
//! The HTML standard decides it from a byte-order mark, then from a `<meta>`
//! declaration found by its prescan of the first 1,024 bytes. One deliberate
//! difference: bytes that are valid UTF-8 and not plain ASCII are read as
//! UTF-8 whatever they declare, because pages re-saved as UTF-8 (a browser's
//! DOM capture, an editor's conversion) keep their old declaration, while
//! legacy text practically never forms valid UTF-8. The `replacement`
//! encoding (ISO-2022-KR, HZ-GB-2312) is ignored instead of turning the page
//! into one replacement character.

use encoding_rs::{Encoding, UTF_8, WINDOWS_1252, X_USER_DEFINED};

/// Bytes the prescan reads.
const PRESCAN: usize = 1024;

/// The non-UTF-8 encoding to read unlabeled HTML bytes with, or `None` when
/// the caller's BOM/UTF-8 handling applies.
pub(crate) fn legacy_encoding(bytes: &[u8]) -> Option<&'static Encoding> {
    if Encoding::for_bom(bytes).is_some() {
        return None;
    }
    let encoding = prescan(&bytes[..bytes.len().min(PRESCAN)]).filter(|e| *e != UTF_8)?;
    // ASCII-only bytes read the same in every declared encoding except
    // ISO-2022-JP, which is why they follow the declaration.
    (bytes.is_ascii() || std::str::from_utf8(bytes).is_err()).then_some(encoding)
}

/// The text of a fetched HTML response and, when the encoding was uncertain
/// or the bytes did not fit it, a warning.
///
/// A BOM decides first. Otherwise the first declaration the bytes actually
/// fit wins: the HTTP `charset`, then a `<meta>` declaration, then UTF-8.
/// Many servers send `utf-8` for bytes that are really GBK, and many old
/// pages declare nothing. When no declaration fits, a declared legacy
/// encoding is kept (a few corrupt sequences cost characters, not the page),
/// UTF-8 with only a sprinkling of bad bytes stays UTF-8, and the rest is
/// read like a local text file, by the East Asian detection of plain-text
/// input, which says so when it cannot tell Windows-1252 from a multibyte
/// reading.
pub(crate) fn decode_fetched<'a>(
    bytes: &'a [u8],
    header_charset: Option<&str>,
) -> (std::borrow::Cow<'a, str>, Option<String>) {
    if Encoding::for_bom(bytes).is_some() {
        return (UTF_8.decode(bytes).0, None);
    }
    let header = header_charset.and_then(|value| label(value.trim_matches(['"', '\'']).as_bytes()));
    let mut declared: Vec<&'static Encoding> = Vec::new();
    for encoding in [header, legacy_encoding(bytes), Some(UTF_8)]
        .into_iter()
        .flatten()
    {
        if !declared.contains(&encoding) {
            declared.push(encoding);
        }
    }
    for encoding in &declared {
        if let Some(text) = encoding.decode_without_bom_handling_and_without_replacement(bytes) {
            return (text, None);
        }
    }
    if let Some(encoding) = declared.iter().find(|encoding| **encoding != UTF_8) {
        let (text, _) = encoding.decode_without_bom_handling(bytes);
        return (
            text,
            Some(format!(
                "HTML declared as {} contains invalid byte sequences; they were replaced with U+FFFD.",
                encoding.name()
            )),
        );
    }
    // Only UTF-8 is left, and the bytes are not valid UTF-8.
    let (lossy, _) = UTF_8.decode_without_bom_handling(bytes);
    let non_ascii = bytes.iter().filter(|byte| !byte.is_ascii()).count();
    let invalid = bytes
        .utf8_chunks()
        .filter(|chunk| !chunk.invalid().is_empty())
        .count();
    if invalid * 100 < non_ascii {
        return (
            lossy,
            Some(format!(
                "HTML is UTF-8 with {invalid} invalid byte sequence{}; they were replaced with U+FFFD.",
                if invalid == 1 { "" } else { "s" }
            )),
        );
    }
    match super::super::text::decode_legacy(bytes) {
        Ok((text, warning)) => (text.into(), warning),
        Err(_) => (
            lossy,
            Some(
                "HTML is not valid UTF-8 and its encoding could not be determined; invalid bytes were replaced with U+FFFD."
                    .into(),
            ),
        ),
    }
}

fn is_space(byte: u8) -> bool {
    matches!(byte, 0x09 | 0x0A | 0x0C | 0x0D | 0x20)
}

/// The standard's "prescan a byte stream to determine its encoding".
fn prescan(bytes: &[u8]) -> Option<&'static Encoding> {
    let mut i = 0;
    while let Some(rest) = bytes.get(i..).filter(|rest| !rest.is_empty()) {
        if rest.starts_with(b"<!--") {
            // The closing dashes may overlap the opening ones (`<!-->`).
            i += 2 + rest[2..].windows(3).position(|w| w == b"-->")? + 3;
        } else if rest.len() > 5
            && rest[..5].eq_ignore_ascii_case(b"<meta")
            && (is_space(rest[5]) || rest[5] == b'/')
        {
            i += 6;
            if let Some(encoding) = meta(bytes, &mut i) {
                return Some(encoding);
            }
            i += 1;
        } else if rest[0] == b'<'
            && match rest.get(1) {
                Some(b'/') => rest.get(2).is_some_and(u8::is_ascii_alphabetic),
                next => next.is_some_and(u8::is_ascii_alphabetic),
            }
        {
            i += rest.iter().position(|&b| is_space(b) || b == b'>')?;
            while attribute(bytes, &mut i).is_some() {}
            i += 1;
        } else if rest.starts_with(b"<!") || rest.starts_with(b"</") || rest.starts_with(b"<?") {
            i += 1 + rest[1..].iter().position(|&b| b == b'>')? + 1;
        } else {
            i += 1;
        }
    }
    None
}

/// The attributes of one `<meta>` element; the declared encoding when they
/// form a `charset` declaration or a `Content-Type` pragma.
fn meta(bytes: &[u8], i: &mut usize) -> Option<&'static Encoding> {
    let mut seen: Vec<Vec<u8>> = Vec::new();
    let mut got_pragma = false;
    let mut need_pragma = None;
    // `None` until declared; `Some(None)` for an unknown `charset` label.
    let mut charset = None;
    while let Some((name, value)) = attribute(bytes, i) {
        if seen.contains(&name) {
            continue;
        }
        match name.as_slice() {
            b"http-equiv" => got_pragma |= value == b"content-type",
            b"content" if charset.is_none() => {
                if let Some(found) = content_charset(&value) {
                    charset = Some(Some(found));
                    need_pragma = Some(true);
                }
            }
            b"charset" => {
                charset = Some(label(&value));
                need_pragma = Some(false);
            }
            _ => (),
        }
        seen.push(name);
    }
    match need_pragma? {
        true if !got_pragma => None,
        _ => charset.flatten(),
    }
}

/// The standard's "get an attribute": a lowercased name and value, or `None`
/// at the end of the tag or of the prescanned bytes.
fn attribute(bytes: &[u8], i: &mut usize) -> Option<(Vec<u8>, Vec<u8>)> {
    let skip_spaces = |i: &mut usize| {
        while bytes.get(*i).copied().is_some_and(is_space) {
            *i += 1;
        }
    };
    while bytes.get(*i).is_some_and(|&b| is_space(b) || b == b'/') {
        *i += 1;
    }
    if *bytes.get(*i)? == b'>' {
        return None;
    }
    let mut name = Vec::new();
    let mut value = Vec::new();
    loop {
        match *bytes.get(*i)? {
            b'=' if !name.is_empty() => break,
            b if is_space(b) => {
                skip_spaces(i);
                if bytes.get(*i) != Some(&b'=') {
                    return Some((name, value));
                }
                break;
            }
            b'/' | b'>' => return Some((name, value)),
            b => name.push(b.to_ascii_lowercase()),
        }
        *i += 1;
    }
    *i += 1;
    skip_spaces(i);
    match *bytes.get(*i)? {
        quote @ (b'"' | b'\'') => loop {
            *i += 1;
            let b = *bytes.get(*i)?;
            if b == quote {
                *i += 1;
                return Some((name, value));
            }
            value.push(b.to_ascii_lowercase());
        },
        b'>' => return Some((name, value)),
        b => value.push(b.to_ascii_lowercase()),
    }
    loop {
        *i += 1;
        match *bytes.get(*i)? {
            b if is_space(b) || b == b'>' => return Some((name, value)),
            b => value.push(b.to_ascii_lowercase()),
        }
    }
}

/// The standard's "extract a character encoding from a meta element" for a
/// `content` value such as `text/html; charset=gbk`.
fn content_charset(value: &[u8]) -> Option<&'static Encoding> {
    let mut i = 0;
    loop {
        i += value[i..].windows(7).position(|w| w == b"charset")? + 7;
        while value.get(i).copied().is_some_and(is_space) {
            i += 1;
        }
        if value.get(i) != Some(&b'=') {
            continue;
        }
        i += 1;
        while value.get(i).copied().is_some_and(is_space) {
            i += 1;
        }
        let rest = &value[i..];
        return match *rest.first()? {
            quote @ (b'"' | b'\'') => {
                let end = rest[1..].iter().position(|&b| b == quote)?;
                label(&rest[1..=end])
            }
            _ => {
                let end = rest
                    .iter()
                    .position(|&b| is_space(b) || b == b';')
                    .unwrap_or(rest.len());
                label(&rest[..end])
            }
        };
    }
}

/// An encoding label with the prescan's substitutions: UTF-16 declarations
/// (impossible in an ASCII-compatible prescan) mean UTF-8 and `x-user-defined`
/// means Windows-1252.
fn label(value: &[u8]) -> Option<&'static Encoding> {
    let encoding = Encoding::for_label_no_replacement(value)?;
    Some(if encoding == X_USER_DEFINED {
        WINDOWS_1252
    } else {
        encoding.output_encoding()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use encoding_rs::{BIG5, GBK, ISO_2022_JP, SHIFT_JIS, WINDOWS_1251};

    fn declared(source: &[u8]) -> Option<&'static str> {
        prescan(source).map(Encoding::name)
    }

    #[test]
    fn prescan_reads_charset_and_content_type_declarations() {
        assert_eq!(declared(b"<meta charset=gbk>"), Some("GBK"));
        assert_eq!(declared(b"<META CHARSET='Big5'>"), Some("Big5"));
        assert_eq!(declared(b"<meta/charset=\"shift_jis\">"), Some("Shift_JIS"));
        assert_eq!(
            declared(b"<meta http-equiv=\"Content-Type\" content=\"text/html; charset=gb2312\">"),
            Some("GBK")
        );
        assert_eq!(
            declared(
                b"<meta content='text/html;charset = \"windows-1251\"' http-equiv=content-type>"
            ),
            Some("windows-1251")
        );
        // The first valid declaration wins; a later one is not read.
        assert_eq!(
            declared(b"<meta charset=euc-kr><meta charset=gbk>"),
            Some("EUC-KR")
        );
    }

    #[test]
    fn prescan_requires_the_pragma_and_a_known_label() {
        assert_eq!(declared(b"<meta content=\"text/html; charset=gbk\">"), None);
        assert_eq!(
            declared(b"<meta http-equiv=refresh content=\"0; charset=gbk\">"),
            None
        );
        assert_eq!(
            declared(b"<meta charset=klingon><meta charset=big5>"),
            Some("Big5")
        );
        // A repeated attribute keeps its first value.
        assert_eq!(declared(b"<meta charset=gbk charset=big5>"), Some("GBK"));
        // `charset` overrides an earlier `content` declaration.
        assert_eq!(
            declared(b"<meta http-equiv=content-type content=\"charset=big5\" charset=gbk>"),
            Some("GBK")
        );
    }

    #[test]
    fn prescan_skips_comments_other_tags_and_attribute_values() {
        assert_eq!(declared(b"<!-- <meta charset=gbk> --><p>"), None);
        assert_eq!(declared(b"<!-- a > b <meta charset=gbk> -->"), None);
        assert_eq!(declared(b"<!--><meta charset=gbk>"), Some("GBK"));
        assert_eq!(
            declared(b"<title x=\"<meta charset=gbk>\"><meta charset=big5>"),
            Some("Big5")
        );
        assert_eq!(
            declared(b"<?xml version=\"1.0\"?><!DOCTYPE html><meta charset=sjis>"),
            Some("Shift_JIS")
        );
        assert_eq!(declared(b"<metadata charset=gbk>"), None);
        let mut late = vec![b' '; PRESCAN];
        late.extend_from_slice(b"<meta charset=gbk>");
        assert_eq!(legacy_encoding(&late), None);
    }

    #[test]
    fn prescan_applies_the_standard_label_substitutions() {
        assert_eq!(declared(b"<meta charset=utf-16le>"), Some("UTF-8"));
        assert_eq!(
            declared(b"<meta charset=x-user-defined>"),
            Some("windows-1252")
        );
        assert_eq!(declared(b"<meta charset=iso-2022-kr>"), None);
        assert_eq!(declared(b"<meta charset=latin1>"), Some("windows-1252"));
    }

    #[test]
    fn legacy_encoding_yields_to_a_bom_and_to_valid_utf8_text() {
        assert_eq!(
            legacy_encoding(b"<meta charset=gbk><p>\xd6\xd0</p>"),
            Some(GBK)
        );
        assert_eq!(
            legacy_encoding(b"<meta charset=big5><p>\xa4\xa4</p>"),
            Some(BIG5)
        );
        assert_eq!(
            legacy_encoding(b"<meta charset=sjis><p>\x82\xa0</p>"),
            Some(SHIFT_JIS)
        );
        assert_eq!(
            legacy_encoding(b"<meta charset=cp1251><p>\xcf\xf0</p>"),
            Some(WINDOWS_1251)
        );
        // ISO-2022-JP is seven-bit, so it is also valid UTF-8.
        assert_eq!(
            legacy_encoding(b"<meta charset=iso-2022-jp><p>\x1b$B$3\x1b(B</p>"),
            Some(ISO_2022_JP)
        );
        assert_eq!(
            legacy_encoding("<meta charset=gbk><p>中文</p>".as_bytes()),
            None
        );
        assert_eq!(
            legacy_encoding(b"\xef\xbb\xbf<meta charset=gbk><p>\xd6\xd0"),
            None
        );
        assert_eq!(
            legacy_encoding(b"<meta charset=utf-8><p>\xd6\xd0</p>"),
            None
        );
        assert_eq!(legacy_encoding(b"<p>\xd6\xd0</p>"), None);
    }

    const CHINESE: &str = "这是一段用于测试字符编码的中文文本，包含常用汉字和标点符号。第二段：风急天高猿啸哀，渚清沙白鸟飞回。无边落木萧萧下，不尽长江滚滚来。";

    const TRADITIONAL: &str = "這是一段用於測試字元編碼的中文文字，包含常用漢字和標點符號。第二段：風急天高猿嘯哀，渚清沙白鳥飛回。無邊落木蕭蕭下，不盡長江滾滾來。";

    fn page_of(text: &str, meta: &str, encoding: &'static Encoding) -> Vec<u8> {
        let source =
            format!("<html><head>{meta}<title>t</title></head><body><p>{text}</p></body></html>");
        let (bytes, _, unmappable) = encoding.encode(&source);
        assert!(!unmappable);
        bytes.into_owned()
    }

    fn page(meta: &str, encoding: &'static Encoding) -> Vec<u8> {
        page_of(CHINESE, meta, encoding)
    }

    #[test]
    fn a_fetched_page_keeps_the_first_declaration_its_bytes_fit() {
        // A correct header wins, even over a contradicting `<meta>`.
        let big5 = page_of(TRADITIONAL, "<meta charset=gbk>", BIG5);
        let (text, warning) = decode_fetched(&big5, Some("big5"));
        assert!(text.contains(TRADITIONAL) && warning.is_none());
        let utf8 = format!("<meta charset=gbk><p>{CHINESE}</p>");
        let (text, warning) = decode_fetched(utf8.as_bytes(), Some("UTF-8"));
        assert!(text.contains(CHINESE) && warning.is_none());
        // Quoted, padded and unknown labels.
        let gbk = page("", GBK);
        for header in ["\"gbk\"", " 'GB2312' "] {
            let (text, warning) = decode_fetched(&gbk, Some(header));
            assert!(text.contains(CHINESE) && warning.is_none(), "{header}");
        }
        // A single-byte header is trusted: it never fails to decode.
        let (text, warning) = decode_fetched(&gbk, Some("windows-1252"));
        assert!(!text.contains(CHINESE) && warning.is_none());
        // A BOM overrides the label.
        let mut bom = b"\xef\xbb\xbf".to_vec();
        bom.extend_from_slice(utf8.as_bytes());
        let (text, warning) = decode_fetched(&bom, Some("gbk"));
        assert!(text.contains(CHINESE) && warning.is_none());
    }

    #[test]
    fn a_utf8_label_that_the_bytes_contradict_yields_to_a_matching_declaration_or_detection() {
        // Header utf-8, `<meta>` gbk: the declaration that fits wins quietly.
        let with_meta = page("<meta charset=gbk>", GBK);
        let (text, warning) = decode_fetched(&with_meta, Some("utf-8"));
        assert!(text.contains(CHINESE) && warning.is_none());
        // Header utf-8 or nothing at all, no `<meta>`: detected.
        let bare = page("", GBK);
        for header in [Some("utf-8"), None] {
            let (text, warning) = decode_fetched(&bare, header);
            assert!(text.contains(CHINESE), "{header:?}: {text}");
            assert!(warning.is_none(), "{header:?}: {warning:?}");
        }
        // Traditional Chinese without any declaration is never garbled
        // silently: either Big5 is recognized or the page says it is unsure.
        let bare = page_of(TRADITIONAL, "", BIG5);
        let (text, warning) = decode_fetched(&bare, None);
        assert!(text.contains(TRADITIONAL) || warning.is_some(), "{text}");
        let (japanese, _, _) =
            SHIFT_JIS.encode("<p>日本語のテキストです。これは文字コードの検出を試すための、少し長めの文章になっています。</p>");
        let (text, warning) = decode_fetched(&japanese, Some("utf-8"));
        assert!(text.contains("日本語のテキストです"), "{text}");
        assert!(warning.is_none());
    }

    #[test]
    fn uncertain_or_partly_corrupt_bytes_keep_a_reading_and_a_warning() {
        // A little Western text without a declaration is Windows-1252.
        let (text, warning) = decode_fetched(b"<p>caf\xe9 au lait</p>", None);
        assert_eq!(text, "<p>café au lait</p>");
        assert!(warning.is_none());
        // A stray bad byte in a long UTF-8 page stays UTF-8.
        let mut mostly = format!("<p>{}</p>", CHINESE.repeat(4)).into_bytes();
        mostly.insert(3, 0xff);
        let (text, warning) = decode_fetched(&mostly, Some("utf-8"));
        assert!(text.contains(CHINESE) && text.contains('\u{fffd}'));
        assert!(warning.unwrap().contains("1 invalid byte sequence;"));
        // A declared legacy encoding with a corrupt tail is kept, with a warning.
        let (text, warning) = decode_fetched(b"<meta charset=shift_jis><p>\x82\xa0\x82</p>", None);
        assert!(text.contains("あ\u{fffd}"));
        assert!(warning.unwrap().starts_with("HTML declared as Shift_JIS"));
        // The `replacement` encodings are not labels.
        let (text, warning) = decode_fetched(CHINESE.as_bytes(), Some("hz-gb-2312"));
        assert!(text.contains(CHINESE) && warning.is_none());
    }

    #[test]
    fn local_html_bytes_follow_the_declaration() {
        use super::super::extract_html_bytes;
        let page = |charset: &str, text: &str, encoding: &'static Encoding| {
            let source = format!(
                "<html><head><meta charset=\"{charset}\"><title>t</title></head><body><p>{text}</p></body></html>"
            );
            let (bytes, _, unmappable) = encoding.encode(&source);
            assert!(!unmappable);
            extract_html_bytes(&bytes).unwrap()
        };
        for (charset, text, encoding) in [
            ("gbk", "中文内容在这里，测试编码检测。", GBK),
            ("gb2312", "简体中文", GBK),
            ("big5", "繁體中文內容", BIG5),
            ("shift_jis", "日本語のテキストです。", SHIFT_JIS),
            ("windows-1251", "Привет, мир!", WINDOWS_1251),
            ("euc-kr", "안녕하세요", encoding_rs::EUC_KR),
        ] {
            let document = page(charset, text, encoding);
            assert_eq!(document.markdown, text, "{charset}");
            assert!(document.warnings.is_empty());
        }
        // Undeclared legacy bytes are read like plain text.
        let (gbk, ..) = GBK.encode("<p>中文内容在这里，测试编码检测。</p>");
        assert_eq!(
            extract_html_bytes(&gbk).unwrap().markdown,
            "中文内容在这里，测试编码检测。"
        );
        assert_eq!(
            extract_html_bytes(b"<p>caf\xe9</p>").unwrap().markdown,
            "café"
        );
        // Bytes invalid in the declared encoding are replaced with a warning.
        let document = extract_html_bytes(b"<meta charset=shift_jis><p>\x82\xa0\x82</p>").unwrap();
        assert_eq!(document.markdown, "あ\u{fffd}");
        assert_eq!(document.warnings.len(), 1);
    }
}
