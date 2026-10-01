//! RTF prelude tables parsed into typed definitions: the font table (with
//! per-font charsets), the style sheet, and the list/list-override tables.

use crate::formats::rtf::lexer::{Lexer, Token, destination_groups};
use crate::shared::blockstyle::{self, BlockStyle};
use crate::shared::code::{is_fixed_pitch_code_face, is_monospace};
use crate::shared::delta::StyleDelta;
use crate::shared::list::MarkerKind;
use crate::shared::numbering::{NumberPattern, NumberText};
use std::collections::{HashMap, HashSet};

pub const LIST_LEVELS: usize = 9;

#[derive(Debug, Clone)]
pub struct ListLevelDef {
    pub marker: Option<MarkerKind>,
    pub start: u64,
    /// `\leveltext` with `\levelnumbers` placeholders and `\levellegal`.
    pub pattern: NumberPattern,
}

impl Default for ListLevelDef {
    fn default() -> Self {
        ListLevelDef { marker: Some(MarkerKind::Bullet), start: 1, pattern: Default::default() }
    }
}

/// Build a number pattern from a `\leveltext` payload (first byte = length,
/// then the characters) and the 1-based placeholder positions listed in
/// `\levelnumbers` (each placeholder byte is a zero-based level index).
fn build_pattern(
    text: &[u8],
    positions: &[u8],
    enc: &'static encoding_rs::Encoding,
) -> Vec<NumberText> {
    let Some((&count, rest)) = text.split_first() else {
        return Vec::new();
    };
    let chars = &rest[..rest.len().min(count as usize)];
    let mut out: Vec<NumberText> = Vec::new();
    let mut literal: Vec<u8> = Vec::new();
    let flush = |literal: &mut Vec<u8>, out: &mut Vec<NumberText>| {
        if !literal.is_empty() {
            let (s, _, _) = enc.decode(literal);
            out.push(NumberText::Literal(s.into_owned()));
            literal.clear();
        }
    };
    for (i, &b) in chars.iter().enumerate() {
        if b <= 8 && positions.contains(&((i + 1) as u8)) {
            flush(&mut literal, &mut out);
            out.push(NumberText::Level(b));
        } else {
            literal.push(b);
        }
    }
    flush(&mut literal, &mut out);
    out
}

#[derive(Debug, Clone)]
pub struct ListDef {
    pub levels: [ListLevelDef; LIST_LEVELS],
}

#[derive(Debug, Clone, Copy, Default)]
pub struct StyleDef {
    pub outline: Option<u8>,
    pub delta: StyleDelta,
    /// The block container the style's name designates.
    pub block: Option<BlockStyle>,
    /// markitai: the text size the style sets (`\fs`), in half-points.
    pub size: Option<crate::shared::visual::Size>,
}

#[derive(Debug, Default)]
pub struct Prelude {
    /// Font number -> encoding, from `\fcharsetN`.
    pub fonts: HashMap<i32, &'static encoding_rs::Encoding>,
    /// markitai: the numbers of the monospaced fonts (see
    /// [`crate::shared::code`]).
    pub mono_fonts: HashSet<i32>,
    /// Paragraph style id (`\sN`) -> definition.
    pub styles: HashMap<i32, StyleDef>,
    /// `\lsN` -> resolved list definition (through the override table).
    pub lists: HashMap<i32, ListDef>,
}

pub fn parse_prelude(bytes: &[u8], default_encoding: &'static encoding_rs::Encoding) -> Prelude {
    let mut prelude = Prelude::default();

    for group in destination_groups(bytes, "fonttbl") {
        parse_fonttbl(group, &mut prelude.fonts, default_encoding);
        mono_fonts(group, &mut prelude.mono_fonts, default_encoding);
    }
    for group in destination_groups(bytes, "stylesheet") {
        parse_stylesheet(group, &mut prelude.styles, default_encoding);
    }
    let mut by_list_id: HashMap<i32, ListDef> = HashMap::new();
    for group in destination_groups(bytes, "listtable") {
        parse_listtable(group, &mut by_list_id, default_encoding);
    }
    for group in destination_groups(bytes, "listoverridetable") {
        parse_overrides(group, &by_list_id, &mut prelude.lists, default_encoding);
    }
    prelude
}

/// Collector for the byte payloads of `\leveltext` / `\levelnumbers`
/// destinations inside one level group.
#[derive(Default)]
struct LevelTextCollector {
    /// Which destination is currently receiving bytes.
    active: Option<bool>, // true = leveltext, false = levelnumbers
    text: Vec<u8>,
    numbers: Vec<u8>,
    legal: bool,
}

impl LevelTextCollector {
    fn byte(&mut self, b: u8) {
        match self.active {
            Some(true) => self.text.push(b),
            Some(false) => self.numbers.push(b),
            None => {}
        }
    }

    /// The finished pattern for an ordered level; bullets carry glyph text,
    /// not a pattern.
    fn finish(
        &mut self,
        marker: Option<MarkerKind>,
        enc: &'static encoding_rs::Encoding,
    ) -> NumberPattern {
        let pattern = if marker.is_some_and(MarkerKind::ordered) {
            NumberPattern { text: build_pattern(&self.text, &self.numbers, enc), legal: self.legal }
        } else {
            NumberPattern::default()
        };
        *self = LevelTextCollector::default();
        pattern
    }
}

fn parse_fonttbl(
    group: &[u8],
    fonts: &mut HashMap<i32, &'static encoding_rs::Encoding>,
    default_encoding: &'static encoding_rs::Encoding,
) {
    let mut lexer = Lexer::new(group);
    let mut current: Option<i32> = None;
    while let Some(token) = lexer.next_token() {
        if let Token::Word { name, param } = token {
            match name {
                "f" => current = param,
                "fcharset" => {
                    if let (Some(f), Some(cs)) = (current, param) {
                        fonts.insert(f, charset_encoding(cs, default_encoding));
                    }
                }
                _ => {}
            }
        }
    }
}

/// markitai: a font the font table is declaring.
struct FontDecl {
    id: i32,
    /// The group depth its `\f` stands at; its name is the text there.
    depth: usize,
    name: Vec<u8>,
    /// The name ended at its `;`.
    named: bool,
    charset: Option<i32>,
    /// Declared fixed-pitch: `\fmodern` or `\fprq1`.
    fixed: bool,
}

/// markitai: the monospaced fonts of a font table (see
/// [`crate::shared::code`]): a monospaced name, or a fixed-pitch declaration
/// (`\fmodern`, `\fprq1`) on a face that is not CJK (by its charset or
/// name). Word writes each font in a group of its own (`{\f2\fmodern
/// \fcharset0\fprq1{\*\panose ...}Courier New;}`); TextEdit writes them one
/// after another (`\f2\fnil\fcharset0 Menlo-Regular;`).
fn mono_fonts(
    group: &[u8],
    mono: &mut HashSet<i32>,
    default_encoding: &'static encoding_rs::Encoding,
) {
    let finish = |font: FontDecl, mono: &mut HashSet<i32>| {
        let encoding = font
            .charset
            .map_or(default_encoding, |charset| charset_encoding(charset, default_encoding));
        let (name, _, _) = encoding.decode(&font.name);
        let name = name.trim();
        let cjk =
            font.charset.is_some_and(|charset| matches!(charset, 128 | 129 | 130 | 134 | 136));
        if is_monospace(name) || (font.fixed && !cjk && is_fixed_pitch_code_face(name)) {
            mono.insert(font.id);
        }
    };
    let mut lexer = Lexer::new(group);
    let mut depth = 0usize;
    let mut current: Option<FontDecl> = None;
    while let Some(token) = lexer.next_token() {
        match token {
            Token::Open => depth += 1,
            Token::Close => {
                depth = depth.saturating_sub(1);
                if current.as_ref().is_some_and(|font| font.depth > depth)
                    && let Some(font) = current.take()
                {
                    finish(font, mono);
                }
            }
            Token::Word { name: "f", param: Some(id) } => {
                if let Some(font) = current.take() {
                    finish(font, mono);
                }
                current = Some(FontDecl {
                    id,
                    depth,
                    name: Vec::new(),
                    named: false,
                    charset: None,
                    fixed: false,
                });
            }
            Token::Word { name, param } => {
                if let Some(font) = current.as_mut().filter(|font| font.depth == depth) {
                    match name {
                        "fmodern" => font.fixed = true,
                        "fprq" => font.fixed |= param == Some(1),
                        "fcharset" => font.charset = param,
                        _ => {}
                    }
                }
            }
            Token::Byte(b) | Token::Hex(b) => {
                if let Some(font) =
                    current.as_mut().filter(|font| font.depth == depth && !font.named)
                {
                    if b == b';' {
                        font.named = true;
                    } else {
                        font.name.push(b);
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(font) = current {
        finish(font, mono);
    }
}

/// The null style id: `\sbasedon222` means "based on nothing".
const NULL_STYLE: i32 = 222;

fn parse_stylesheet(
    group: &[u8],
    styles: &mut HashMap<i32, StyleDef>,
    enc: &'static encoding_rs::Encoding,
) {
    let mut lexer = Lexer::new(group);
    let mut depth = 0usize;
    let mut current: Option<(i32, StyleDef, Option<i32>)> = None;
    let mut raw: HashMap<i32, (StyleDef, Option<i32>)> = HashMap::new();
    // A style's name is the text at the end of its group, before the `;`.
    let mut name: Vec<u8> = Vec::new();
    while let Some(token) = lexer.next_token() {
        match token {
            Token::Open => depth += 1,
            Token::Close => {
                if depth == 1
                    && let Some((id, mut def, base)) = current.take()
                {
                    let (text, _, _) = enc.decode(&name);
                    def.block = blockstyle::from_style_name(text.trim_end_matches(';'));
                    raw.insert(id, (def, base));
                }
                name.clear();
                depth = depth.saturating_sub(1);
            }
            Token::Hex(b) | Token::Byte(b) if depth == 1 && current.is_some() => name.push(b),
            Token::Word { name: word, param } => match word {
                "s" => {
                    current = Some((param.unwrap_or(0), StyleDef::default(), None));
                    name.clear();
                }
                "sbasedon" => {
                    if let Some((_, _, base)) = current.as_mut() {
                        *base = param.filter(|&b| b != NULL_STYLE);
                    }
                }
                "outlinelevel" => {
                    if let (Some((_, def, _)), Some(n)) = (current.as_mut(), param)
                        && (0..9).contains(&n)
                    {
                        def.outline = Some((n + 1) as u8);
                    }
                }
                "b" => {
                    if let Some((_, def, _)) = current.as_mut() {
                        def.delta.bold = Some(param != Some(0));
                    }
                }
                "i" => {
                    if let Some((_, def, _)) = current.as_mut() {
                        def.delta.italic = Some(param != Some(0));
                    }
                }
                // markitai: the style's text size.
                "fs" => {
                    if let Some((_, def, _)) = current.as_mut() {
                        def.size = Some(param.unwrap_or(24).clamp(1, 3276) as u32);
                    }
                }
                _ => {}
            },
            _ => {}
        }
    }
    // Resolve every \sbasedon chain root-to-leaf: the child's own settings
    // win over inherited ones. A cycle is bounded by the visited set and
    // resolves from the acyclic prefix.
    for &id in raw.keys() {
        let mut chain: Vec<&StyleDef> = Vec::new();
        let mut seen: std::collections::HashSet<i32> = std::collections::HashSet::new();
        let mut cursor = Some(id);
        while let Some(cur) = cursor {
            if !seen.insert(cur) {
                log::warn!("style inheritance cycle at rtf style {cur}");
                break;
            }
            let Some((def, base)) = raw.get(&cur) else { break };
            chain.push(def);
            cursor = *base;
        }
        let mut resolved = StyleDef::default();
        for def in chain.iter().rev() {
            resolved.delta = resolved.delta.merge(def.delta);
            resolved.outline = def.outline.or(resolved.outline);
            resolved.block = def.block.or(resolved.block);
            resolved.size = def.size.or(resolved.size);
        }
        styles.insert(id, resolved);
    }
}

fn parse_listtable(
    group: &[u8],
    by_list_id: &mut HashMap<i32, ListDef>,
    enc: &'static encoding_rs::Encoding,
) {
    let mut lexer = Lexer::new(group);
    let mut depth = 0usize;
    let mut list_depth: Option<usize> = None;
    let mut levels: [ListLevelDef; LIST_LEVELS] = std::array::from_fn(|_| ListLevelDef::default());
    let mut level_index = 0usize;
    let mut in_level = false;
    let mut list_id: Option<i32> = None;
    let mut collector = LevelTextCollector::default();
    while let Some(token) = lexer.next_token() {
        match token {
            Token::Open => depth += 1,
            Token::Close => {
                // A leveltext/levelnumbers destination ends at its group.
                collector.active = None;
                if in_level && Some(depth) == list_depth.map(|d| d + 1) {
                    in_level = false;
                    if level_index < LIST_LEVELS {
                        levels[level_index].pattern =
                            collector.finish(levels[level_index].marker, enc);
                    }
                    level_index += 1;
                }
                if list_depth == Some(depth) {
                    if let Some(id) = list_id.take() {
                        by_list_id.insert(id, ListDef { levels: levels.clone() });
                    }
                    levels = std::array::from_fn(|_| ListLevelDef::default());
                    level_index = 0;
                    list_depth = None;
                }
                depth = depth.saturating_sub(1);
            }
            Token::Word { name, param } => match name {
                "list" => {
                    list_depth = Some(depth);
                    list_id = None;
                    levels = std::array::from_fn(|_| ListLevelDef::default());
                    level_index = 0;
                }
                "listid" => {
                    if list_depth.is_some() {
                        list_id = param;
                    }
                }
                "listlevel" => {
                    in_level = true;
                    collector = LevelTextCollector::default();
                }
                "levelnfc" | "levelnfcn" => {
                    if in_level && level_index < LIST_LEVELS {
                        levels[level_index].marker = marker_for_nfc(param.unwrap_or(0));
                    }
                }
                "levelstartat" => {
                    if in_level
                        && level_index < LIST_LEVELS
                        && let Some(n) = param
                    {
                        levels[level_index].start = n.max(0) as u64;
                    }
                }
                "leveltext" if in_level => collector.active = Some(true),
                "levelnumbers" if in_level => collector.active = Some(false),
                "levellegal" if in_level => collector.legal = param != Some(0),
                _ => {}
            },
            Token::Hex(b) | Token::Byte(b) => collector.byte(b),
            _ => {}
        }
    }
}

/// Raw override number text: (`\leveltext` bytes, `\levelnumbers` bytes, legal).
type RawLevelText = (Vec<u8>, Vec<u8>, bool);

fn parse_overrides(
    group: &[u8],
    by_list_id: &HashMap<i32, ListDef>,
    lists: &mut HashMap<i32, ListDef>,
    enc: &'static encoding_rs::Encoding,
) {
    let mut lexer = Lexer::new(group);
    let mut depth = 0usize;
    let mut over_depth: Option<usize> = None;
    let mut lfo_depth: Option<usize> = None;
    let mut list_id: Option<i32> = None;
    let mut ls: Option<i32> = None;
    // Per-level override records (`\lfolevel` groups in level order): an
    // overridden start and/or an overriding format (embedded `\listlevel`
    // with its own number text).
    let mut level_index = 0usize;
    let mut starts: [Option<u64>; LIST_LEVELS] = [None; LIST_LEVELS];
    let mut markers: [Option<Option<MarkerKind>>; LIST_LEVELS] = [None; LIST_LEVELS];
    let mut texts: [Option<RawLevelText>; LIST_LEVELS] = std::array::from_fn(|_| None);
    let mut collector = LevelTextCollector::default();
    let mut flush = |ls: &mut Option<i32>,
                     list_id: &mut Option<i32>,
                     starts: &mut [Option<u64>; LIST_LEVELS],
                     markers: &mut [Option<Option<MarkerKind>>; LIST_LEVELS],
                     texts: &mut [Option<RawLevelText>; LIST_LEVELS]| {
        if let (Some(ls_n), Some(id)) = (ls.take(), list_id.take()) {
            let mut def = by_list_id.get(&id).cloned().unwrap_or_else(|| ListDef {
                levels: std::array::from_fn(|_| ListLevelDef::default()),
            });
            for level in 0..LIST_LEVELS {
                if let Some(m) = markers[level] {
                    def.levels[level].marker = m;
                }
                if let Some(s) = starts[level] {
                    def.levels[level].start = s;
                }
                if let Some((text, numbers, legal)) = &texts[level]
                    && def.levels[level].marker.is_some_and(MarkerKind::ordered)
                {
                    def.levels[level].pattern =
                        NumberPattern { text: build_pattern(text, numbers, enc), legal: *legal };
                }
            }
            lists.insert(ls_n, def);
        }
        *starts = [None; LIST_LEVELS];
        *markers = [None; LIST_LEVELS];
        *texts = std::array::from_fn(|_| None);
    };
    while let Some(token) = lexer.next_token() {
        match token {
            Token::Open => depth += 1,
            Token::Close => {
                collector.active = None;
                if lfo_depth == Some(depth) {
                    lfo_depth = None;
                    if level_index < LIST_LEVELS && !collector.text.is_empty() {
                        texts[level_index] = Some((
                            std::mem::take(&mut collector.text),
                            std::mem::take(&mut collector.numbers),
                            collector.legal,
                        ));
                    }
                    collector = LevelTextCollector::default();
                    level_index += 1;
                }
                if over_depth == Some(depth) {
                    flush(&mut ls, &mut list_id, &mut starts, &mut markers, &mut texts);
                    level_index = 0;
                    over_depth = None;
                }
                depth = depth.saturating_sub(1);
            }
            Token::Word { name, param } => match name {
                "listoverride" => {
                    // A previous override group left unclosed still counts.
                    flush(&mut ls, &mut list_id, &mut starts, &mut markers, &mut texts);
                    over_depth = Some(depth);
                    level_index = 0;
                }
                "listid" if over_depth.is_some() => list_id = param,
                "ls" if over_depth.is_some() => ls = param,
                "lfolevel" => {
                    lfo_depth = Some(depth);
                    collector = LevelTextCollector::default();
                }
                "levelstartat" if lfo_depth.is_some() => {
                    if level_index < LIST_LEVELS
                        && let Some(n) = param
                    {
                        starts[level_index] = Some(n.max(0) as u64);
                    }
                }
                "levelnfc" | "levelnfcn" if lfo_depth.is_some() => {
                    if level_index < LIST_LEVELS {
                        markers[level_index] = Some(marker_for_nfc(param.unwrap_or(0)));
                    }
                }
                "leveltext" if lfo_depth.is_some() => collector.active = Some(true),
                "levelnumbers" if lfo_depth.is_some() => collector.active = Some(false),
                "levellegal" if lfo_depth.is_some() => collector.legal = param != Some(0),
                _ => {}
            },
            Token::Hex(b) | Token::Byte(b) => collector.byte(b),
            _ => {}
        }
    }
    flush(&mut ls, &mut list_id, &mut starts, &mut markers, &mut texts);
}

/// MS-OSHARED numbering formats -> marker kinds. `None` = no number.
fn marker_for_nfc(nfc: i32) -> Option<MarkerKind> {
    match nfc {
        0 => Some(MarkerKind::Decimal),
        1 => Some(MarkerKind::UpperRoman),
        2 => Some(MarkerKind::LowerRoman),
        3 => Some(MarkerKind::UpperAlpha),
        4 => Some(MarkerKind::LowerAlpha),
        23 => Some(MarkerKind::Bullet),
        255 => None,
        _ => Some(MarkerKind::Decimal),
    }
}

/// `\fcharsetN` -> encoding.
///
/// markitai: charset 0 is ANSI, which is Windows-1252 whatever `\ansicpg`
/// says; only 1 (`DEFAULT_CHARSET`) defers to the document code page.
/// TextEdit, saving on a Chinese system, declares `\ansicpg936` but writes
/// the bytes of its charset-0 fonts in 1252 (`Apple\'92s`, `\'bd` for ½),
/// and the code page turned them into CJK characters. Charset 77 is Mac
/// Roman.
fn charset_encoding(
    charset: i32,
    default_encoding: &'static encoding_rs::Encoding,
) -> &'static encoding_rs::Encoding {
    match charset {
        0 => encoding_rs::WINDOWS_1252,
        1 => default_encoding,
        77 => encoding_rs::MACINTOSH,
        128 => encoding_rs::SHIFT_JIS,
        129 => encoding_rs::EUC_KR,
        134 => encoding_rs::GBK,
        136 => encoding_rs::BIG5,
        161 => encoding_rs::WINDOWS_1253,
        162 => encoding_rs::WINDOWS_1254,
        163 => encoding_rs::WINDOWS_1258,
        177 => encoding_rs::WINDOWS_1255,
        178..=180 => encoding_rs::WINDOWS_1256,
        186 => encoding_rs::WINDOWS_1257,
        204 => encoding_rs::WINDOWS_1251,
        222 => encoding_rs::WINDOWS_874,
        238 => encoding_rs::WINDOWS_1250,
        _ => default_encoding,
    }
}

pub fn codepage_encoding(cp: u32) -> &'static encoding_rs::Encoding {
    match cp {
        932 => encoding_rs::SHIFT_JIS,
        936 => encoding_rs::GBK,
        949 => encoding_rs::EUC_KR,
        950 => encoding_rs::BIG5,
        1250 => encoding_rs::WINDOWS_1250,
        1251 => encoding_rs::WINDOWS_1251,
        1253 => encoding_rs::WINDOWS_1253,
        1254 => encoding_rs::WINDOWS_1254,
        1255 => encoding_rs::WINDOWS_1255,
        1256 => encoding_rs::WINDOWS_1256,
        1257 => encoding_rs::WINDOWS_1257,
        1258 => encoding_rs::WINDOWS_1258,
        874 => encoding_rs::WINDOWS_874,
        _ => encoding_rs::WINDOWS_1252,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// markitai: the monospaced fonts of a font table.
    fn mono(table: &str) -> Vec<i32> {
        let mut found = HashSet::new();
        mono_fonts(table.as_bytes(), &mut found, encoding_rs::WINDOWS_1252);
        let mut found: Vec<i32> = found.into_iter().collect();
        found.sort_unstable();
        found
    }

    #[test]
    fn monospaced_fonts_are_named_or_declared_fixed_pitch() {
        // TextEdit: one font after another, PostScript names.
        assert_eq!(
            mono(
                r"\f0\froman\fcharset0 Times-Bold;\f1\fnil\fcharset0 Menlo-Regular;\f2\fmodern\fcharset0 Courier;
                \f3\fnil\fcharset0 Monaco;\f4\fnil\fcharset0 HelveticaNeue;"
            ),
            [1, 2, 3]
        );
        // Word: a group per font, the name after a panose group; a fixed
        // pitch (`\fprq1`) or the modern family marks an unnamed code face,
        // unless its charset or name is CJK.
        assert_eq!(
            mono(
                r"{\f0\froman\fcharset0\fprq2{\*\panose 02020603050405020304}Times New Roman;}
                {\f1\fswiss\fcharset0\fprq1 Iosevka;}{\f2\fmodern\fcharset0 Letter Gothic;}
                {\f3\fmodern\fcharset128\fprq1 MS Mincho;}{\f4\fmodern\fcharset0\fprq1 MS Gothic Western;}
                {\f5\fnil\fcharset134\fprq1 \'cb\'ce\'cc\'e5;}{\f6\fswiss\fcharset0\fprq2 Arial{\*\falt  Mono};}
                {\f7\fmodern\fcharset128\fprq1 BIZ UDGothic;}"
            ),
            [1, 2]
        );
    }
}
