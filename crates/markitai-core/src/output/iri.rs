//! Web addresses as people read them: IRIs (RFC 3987) in front matter.
//!
//! A page address arrives percent-encoded (a browser copies
//! `/posts/20251223-ai-%E5%BA%94%E7%94%A8`, the URL parser encodes what was
//! typed, and a page's `rel=canonical` is often written that way), which
//! hides every non-ASCII word of it. Front matter shows it with those
//! characters decoded (`/posts/20251223-ai-应用`): the same address, since
//! turning an IRI back into a URI percent-encodes each non-ASCII character as
//! its UTF-8 bytes (RFC 3987 §3.1), which gives the original back. Only
//! escapes of non-ASCII characters are decoded; `%20`, `%2F`, `%3F`, `%25`
//! and every other ASCII escape stay, so the path, query and fragment keep
//! their boundaries. Characters RFC 3987 leaves out of an IRI (bidirectional
//! controls, spaces and other separators, control and private-use
//! characters) and bytes that are not UTF-8 stay encoded too. The authority
//! is left as written. Python markitai 1.3.0 writes the same
//! (`utils/iri.py`).

use unicode_properties::{GeneralCategoryGroup, UnicodeGeneralCategory};

/// A non-ASCII character an IRI may hold as itself.
fn readable(ch: char) -> bool {
    // Separators (no-break and ideographic spaces), controls, format
    // characters (the bidirectional controls RFC 3987 §4.1 forbids), private
    // use and unassigned code points stay encoded, as do the last two code
    // points of every plane, which RFC 3987's `ucschar` leaves out.
    !matches!(
        ch.general_category_group(),
        GeneralCategoryGroup::Separator | GeneralCategoryGroup::Other
    ) && (ch as u32 & 0xFFFE) != 0xFFFE
}

fn hex(byte: u8) -> Option<u8> {
    (byte as char).to_digit(16).map(|digit| digit as u8)
}

/// One run of `%XX` escapes with its readable non-ASCII characters decoded.
fn decode_run(run: &str, out: &mut String) {
    let escapes: Vec<&str> = (0..run.len())
        .step_by(3)
        .map(|at| &run[at..at + 3])
        .collect();
    let data: Vec<u8> = escapes
        .iter()
        .map(|escape| {
            let bytes = escape.as_bytes();
            hex(bytes[1]).unwrap_or(0) << 4 | hex(bytes[2]).unwrap_or(0)
        })
        .collect();
    let mut index = 0;
    while index < data.len() {
        let size = match data[index] {
            0xC2..=0xDF => 2,
            0xE0..=0xEF => 3,
            0xF0..=0xF4 => 4,
            _ => 0,
        };
        let decoded = (size > 0 && index + size <= data.len())
            .then(|| std::str::from_utf8(&data[index..index + size]).ok())
            .flatten()
            .and_then(|text| text.chars().next());
        match decoded {
            Some(ch) => {
                if readable(ch) {
                    out.push(ch);
                } else {
                    escapes[index..index + size]
                        .iter()
                        .for_each(|escape| out.push_str(escape));
                }
                index += size;
            }
            // ASCII, or a byte that starts no UTF-8 character.
            None => {
                out.push_str(escapes[index]);
                index += 1;
            }
        }
    }
}

/// `url` with its percent-encoded non-ASCII characters shown, if it is an
/// `http` or `https` address; anything else (a file name, a `mailto:`, a
/// value that is no URL) comes back unchanged.
pub(crate) fn readable_url(url: &str) -> String {
    if !url.contains('%') {
        return url.to_owned();
    }
    let Some((scheme, rest)) = url.split_once("://") else {
        return url.to_owned();
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return url.to_owned();
    }
    let authority = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    if authority == 0 {
        return url.to_owned();
    }
    let start = scheme.len() + 3 + authority;
    let (head, tail) = url.split_at(start);
    let mut out = String::with_capacity(url.len());
    out.push_str(head);
    let bytes = tail.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        let mut end = index;
        while end + 2 < bytes.len()
            && bytes[end] == b'%'
            && hex(bytes[end + 1]).is_some()
            && hex(bytes[end + 2]).is_some()
        {
            end += 3;
        }
        if end > index {
            decode_run(&tail[index..end], &mut out);
            index = end;
        } else {
            let ch = tail[index..].chars().next().expect("index on a character");
            out.push(ch);
            index += ch.len_utf8();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const ENCODED: &str = concat!(
        "https://ynewtime.com/posts/20251223-ai-%E5%BA%94%E7%94%A8%E6%A1%88%E4%BE%8B",
        "%E7%B3%BB%E5%88%972chrome-%E6%8F%92%E4%BB%B6/"
    );
    const READABLE: &str = "https://ynewtime.com/posts/20251223-ai-应用案例系列2chrome-插件/";

    /// RFC 3987 §3.1: each non-ASCII character as its UTF-8 escapes.
    fn as_uri(iri: &str) -> String {
        let mut uri = String::new();
        for ch in iri.chars() {
            if ch.is_ascii() {
                uri.push(ch);
            } else {
                let mut buffer = [0; 4];
                for byte in ch.encode_utf8(&mut buffer).bytes() {
                    uri.push_str(&format!("%{byte:02X}"));
                }
            }
        }
        uri
    }

    #[test]
    fn the_owners_address_reads_as_written_and_round_trips() {
        assert_eq!(readable_url(ENCODED), READABLE);
        assert_eq!(as_uri(&readable_url(ENCODED)), ENCODED);
        let url = "https://x.test/%E4%B8%AD/%F0%9F%98%80?a=%E6%97%A5%E6%9C%AC#%C3%A9";
        assert_eq!(readable_url(url), "https://x.test/中/😀?a=日本#é");
        assert_eq!(as_uri(&readable_url(url)), url);
    }

    #[test]
    fn only_readable_non_ascii_escapes_are_decoded() {
        for (url, expected) in [
            // ASCII escapes keep their meaning: space, slash, &, %, ?, #.
            (
                "https://x.test/a%20b%2Fc?q=%E4%B8%AD%26x#%E6%96%87",
                "https://x.test/a%20b%2Fc?q=中%26x#文",
            ),
            ("https://x.test/100%25", "https://x.test/100%25"),
            ("https://x.test/%e4%b8%ad", "https://x.test/中"),
            // Bidi controls, ideographic space, private use, non-UTF-8 bytes,
            // a cut sequence: kept as escapes.
            ("https://x.test/%E2%80%8Fa", "https://x.test/%E2%80%8Fa"),
            ("https://x.test/%E3%80%80", "https://x.test/%E3%80%80"),
            ("https://x.test/%EF%80%83", "https://x.test/%EF%80%83"),
            ("https://x.test/%FF%E4%B8", "https://x.test/%FF%E4%B8"),
            ("https://x.test/%E4%B8%AD%E4", "https://x.test/中%E4"),
            ("https://x.test/%E4%B8%A", "https://x.test/%E4%B8%A"),
            // The authority stays as written (punycode, port).
            (
                "https://xn--fiq.test:8443/%E4%B8%AD",
                "https://xn--fiq.test:8443/中",
            ),
            ("HTTPS://x.test?%E4%B8%AD", "HTTPS://x.test?中"),
            // Not a web address: unchanged.
            ("report%E4%B8%AD.pdf", "report%E4%B8%AD.pdf"),
            ("ftp://x.test/%E4%B8%AD", "ftp://x.test/%E4%B8%AD"),
            ("https://x.test/plain", "https://x.test/plain"),
        ] {
            assert_eq!(readable_url(url), expected, "{url}");
        }
    }
}
