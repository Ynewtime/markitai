//! `w:sym` symbol characters.
//!
//! markitai: Word stores a character picked from the Symbol dialog as a font
//! name and a code point, and for the legacy symbol fonts the code sits in the
//! private-use block (`F0xx`, the font's byte value plus `F000`). Without a
//! mapping to Unicode the mark (a tick, an arrow, a Greek letter typed in the
//! Symbol font) is lost from the text.

/// The character a `w:sym` element shows, when it maps to Unicode.
///
/// `font` is the element's `w:font`, `code` its `w:char` (hexadecimal).
/// Characters outside the private-use block are already Unicode. A legacy
/// symbol font's private-use code is looked up in that font's table; fonts
/// without a table, and unassigned codes, give `None` rather than a
/// character the reader could not display.
pub fn symbol_char(font: &str, code: &str) -> Option<char> {
    let code = u32::from_str_radix(code.trim(), 16).ok()?;
    if !(0xF000..=0xF0FF).contains(&code) {
        // A private-use code outside the symbol-font block has no meaning
        // without the font; anything else is a Unicode scalar value.
        return char::from_u32(code).filter(|c| !is_private_use(*c) && !c.is_control());
    }
    let byte = (code & 0xFF) as u8;
    match font.trim().to_ascii_lowercase().as_str() {
        "symbol" => symbol_font(byte),
        "wingdings" => wingdings(byte),
        _ => None,
    }
}

/// markitai: whether text set in `font` shows that font's glyphs rather than
/// the characters it holds: the legacy symbol fonts with a table here.
/// (LibreOffice's OpenSymbol is a Unicode font and is not one of them.)
pub fn is_symbol_font(font: &str) -> bool {
    symbol_face(font).is_some()
}

/// markitai: the symbol font `font` names, as [`symbol_text`] takes it.
pub fn symbol_face(font: &str) -> Option<&'static str> {
    let font = font.trim();
    if font.eq_ignore_ascii_case("symbol") {
        Some("Symbol")
    } else if font.eq_ignore_ascii_case("wingdings") {
        Some("Wingdings")
    } else {
        None
    }
}

/// markitai: the text a run set in symbol font `font` shows. Word and
/// LibreOffice keep such text as the font's byte values, either as
/// characters `U+0020`-`U+00FF` (text typed in the font: `a` in Symbol is
/// α, `ü` in Wingdings a tick) or in the private-use block `F020`-`F0FF`;
/// each maps through the font's table, and characters it has no entry for,
/// or that lie elsewhere (a real α), stay as they are. Before, the reader
/// wrote such text as the Latin letters or private-use characters stored.
pub fn symbol_text(font: &str, text: &str) -> String {
    text.chars()
        .map(|c| match c as u32 {
            code @ (0x20..=0xFF | 0xF020..=0xF0FF) => {
                symbol_char(font, &format!("F0{:02X}", code & 0xFF)).unwrap_or(c)
            }
            _ => c,
        })
        .collect()
}

fn is_private_use(c: char) -> bool {
    matches!(c as u32, 0xE000..=0xF8FF | 0xF0000..=0x10FFFF)
}

/// The Adobe Symbol encoding, as Microsoft's Symbol font uses it.
fn symbol_font(byte: u8) -> Option<char> {
    let code: u32 = match byte {
        0x20..=0x21
        | 0x23
        | 0x25..=0x26
        | 0x28..=0x29
        | 0x2B..=0x2C
        | 0x2E..=0x3F
        | 0x5B
        | 0x5D
        | 0x5F
        | 0x7B..=0x7D => byte as u32,
        0x22 => 0x2200, // for all
        0x24 => 0x2203, // there exists
        0x27 => 0x220B, // contains as member
        0x2A => 0x2217, // asterisk operator
        0x2D => 0x2212, // minus
        0x40 => 0x2245, // congruent
        0x41 => 0x0391, // Alpha
        0x42 => 0x0392,
        0x43 => 0x03A7, // Chi
        0x44 => 0x0394, // Delta
        0x45 => 0x0395,
        0x46 => 0x03A6, // Phi
        0x47 => 0x0393, // Gamma
        0x48 => 0x0397,
        0x49 => 0x0399,
        0x4A => 0x03D1, // theta symbol
        0x4B => 0x039A,
        0x4C => 0x039B,
        0x4D => 0x039C,
        0x4E => 0x039D,
        0x4F => 0x039F,
        0x50 => 0x03A0,
        0x51 => 0x0398, // Theta
        0x52 => 0x03A1,
        0x53 => 0x03A3,
        0x54 => 0x03A4,
        0x55 => 0x03A5,
        0x56 => 0x03C2, // final sigma
        0x57 => 0x03A9, // Omega
        0x58 => 0x039E,
        0x59 => 0x03A8,
        0x5A => 0x0396,
        0x5C => 0x2234, // therefore
        0x5E => 0x22A5, // perpendicular
        0x60 => 0x203E, // radical extender
        0x61 => 0x03B1,
        0x62 => 0x03B2,
        0x63 => 0x03C7,
        0x64 => 0x03B4,
        0x65 => 0x03B5,
        0x66 => 0x03C6,
        0x67 => 0x03B3,
        0x68 => 0x03B7,
        0x69 => 0x03B9,
        0x6A => 0x03D5, // phi symbol
        0x6B => 0x03BA,
        0x6C => 0x03BB,
        0x6D => 0x03BC,
        0x6E => 0x03BD,
        0x6F => 0x03BF,
        0x70 => 0x03C0,
        0x71 => 0x03B8,
        0x72 => 0x03C1,
        0x73 => 0x03C3,
        0x74 => 0x03C4,
        0x75 => 0x03C5,
        0x76 => 0x03D6, // pi symbol
        0x77 => 0x03C9,
        0x78 => 0x03BE,
        0x79 => 0x03C8,
        0x7A => 0x03B6,
        0x7E => 0x223C, // tilde operator
        0xA0 => 0x20AC, // euro
        0xA1 => 0x03D2, // upsilon with hook
        0xA2 => 0x2032, // prime
        0xA3 => 0x2264, // less than or equal
        0xA4 => 0x2044, // fraction slash
        0xA5 => 0x221E, // infinity
        0xA6 => 0x0192, // florin
        0xA7 => 0x2663, // club
        0xA8 => 0x2666, // diamond
        0xA9 => 0x2665, // heart
        0xAA => 0x2660, // spade
        0xAB => 0x2194, // left right arrow
        0xAC => 0x2190,
        0xAD => 0x2191,
        0xAE => 0x2192,
        0xAF => 0x2193,
        0xB0 => 0x00B0,
        0xB1 => 0x00B1,
        0xB2 => 0x2033, // double prime
        0xB3 => 0x2265, // greater than or equal
        0xB4 => 0x00D7,
        0xB5 => 0x221D, // proportional to
        0xB6 => 0x2202, // partial differential
        0xB7 => 0x2022, // bullet
        0xB8 => 0x00F7,
        0xB9 => 0x2260,
        0xBA => 0x2261,
        0xBB => 0x2248,
        0xBC => 0x2026,
        0xBF => 0x21B5, // carriage return arrow
        0xC0 => 0x2135, // aleph
        0xC1 => 0x2111,
        0xC2 => 0x211C,
        0xC3 => 0x2118,
        0xC4 => 0x2297, // circled times
        0xC5 => 0x2295, // circled plus
        0xC6 => 0x2205, // empty set
        0xC7 => 0x2229,
        0xC8 => 0x222A,
        0xC9 => 0x2283,
        0xCA => 0x2287,
        0xCB => 0x2284,
        0xCC => 0x2282,
        0xCD => 0x2286,
        0xCE => 0x2208,
        0xCF => 0x2209,
        0xD0 => 0x2220, // angle
        0xD1 => 0x2207, // nabla
        0xD2 | 0xE2 => 0x00AE,
        0xD3 | 0xE3 => 0x00A9,
        0xD4 | 0xE4 => 0x2122,
        0xD5 => 0x220F, // product
        0xD6 => 0x221A, // square root
        0xD7 => 0x22C5, // dot operator
        0xD8 => 0x00AC,
        0xD9 => 0x2227,
        0xDA => 0x2228,
        0xDB => 0x21D4,
        0xDC => 0x21D0,
        0xDD => 0x21D1,
        0xDE => 0x21D2,
        0xDF => 0x21D3,
        0xE0 => 0x25CA, // lozenge
        0xE1 => 0x27E8, // left angle bracket
        0xE5 => 0x2211, // summation
        0xF1 => 0x27E9, // right angle bracket
        0xF2 => 0x222B, // integral
        // Bracket and integral pieces, which build large delimiters from
        // several glyphs, have no character of their own.
        _ => return None,
    };
    char::from_u32(code)
}

/// The Wingdings characters documents use as bullets, ticks and arrows. The
/// font's pictographs (hands, flowers, clocks) have no reliable Unicode
/// counterpart and give `None`.
fn wingdings(byte: u8) -> Option<char> {
    let code: u32 = match byte {
        0x28 => 0x260E, // telephone
        0x4A => 0x263A, // smiling face
        0x4C => 0x2639, // frowning face
        0x6C => 0x25CF, // black circle
        0x6E => 0x25A0, // black square
        0x6F => 0x25A1, // white square
        0x71 => 0x2751, // shadowed white square
        0x75 => 0x25C6, // black diamond
        0x76 => 0x2756, // black diamond minus white x
        0xA7 => 0x25AA, // black small square
        0xA8 => 0x25FB, // white medium square
        0xD8 => 0x27A2, // three-d top-lighted rightwards arrowhead
        0xE0 => 0x2192, // rightwards arrow
        0xFB => 0x2717, // ballot x
        0xFC => 0x2713, // check mark
        0xFD => 0x2612, // ballot box with x
        0xFE => 0x2611, // ballot box with check
        _ => return None,
    };
    char::from_u32(code)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unicode_codes_are_themselves() {
        assert_eq!(symbol_char("Arial", "00A9"), Some('©'));
        assert_eq!(symbol_char("MS Gothic", "2611"), Some('☑'));
    }

    #[test]
    fn symbol_font_codes_map_to_unicode() {
        assert_eq!(symbol_char("Symbol", "F061"), Some('α'));
        assert_eq!(symbol_char("symbol", "F0B7"), Some('•'));
        assert_eq!(symbol_char("Symbol", "F0AE"), Some('→'));
        assert_eq!(symbol_char("Symbol", "F0E5"), Some('∑'));
        // Large-delimiter pieces have no character.
        assert_eq!(symbol_char("Symbol", "F0EB"), None);
    }

    #[test]
    fn wingdings_codes_map_to_common_marks() {
        assert_eq!(symbol_char("Wingdings", "F0FC"), Some('✓'));
        assert_eq!(symbol_char("Wingdings", "F0A7"), Some('▪'));
        assert_eq!(symbol_char("Wingdings", "F0E0"), Some('→'));
        // A pictograph, and a font this table does not cover.
        assert_eq!(symbol_char("Wingdings", "F021"), None);
        assert_eq!(symbol_char("Webdings", "F0FC"), None);
    }

    #[test]
    fn malformed_codes_and_private_use_are_dropped() {
        assert_eq!(symbol_char("Symbol", "zz"), None);
        assert_eq!(symbol_char("Arial", "E123"), None);
        assert_eq!(symbol_char("Arial", "0007"), None);
    }
}
