//! RTF frontend: a position-explicit lexer feeding a state machine, with the
//! font/style/list tables parsed into typed definitions up front. Numbering
//! comes from the list tables - never guessed from label text. Page
//! headers/footers are excluded (fixed policy).

mod lexer;
mod table;
mod tables;

use crate::error::ConvertError;
use crate::formats::docx::scripts::Script;
use crate::formats::docx::symbols::symbol_char;
use crate::model::{Block, Document, Inline, Note, NoteKind, Style, inlines_are_empty};
use crate::package::xml::{Element, Node, ns};
use crate::shared::blockstyle::{BlockStyle, StyledRun};
use crate::shared::code::{
    MonoShare, RunFonts, drop_line_gutters, has_image, listing_tables, without_code,
};
use crate::shared::delta::rebase_emphasis;
use crate::shared::fields::{FormField, FormKind, field_result, form_field_result};
use crate::shared::list::{ListEntry, ListKey, MarkerKind, continuation_level, flush_list};
use crate::shared::math::{math_lines, omath_para_to_tex};
use crate::shared::tabs::{self, Stops, TabRows};
use crate::shared::text::clean_text;
use crate::shared::typed_lists::{Indent, TypedLists, opens_with_bullet};
use crate::shared::visual::{Looks, ParaSize, Size};
use lexer::{Lexer, Token};
use std::collections::HashMap;
use std::rc::Rc;
use table::TableState;
use tables::{LIST_LEVELS, Prelude, codepage_encoding, parse_prelude};

pub fn parse(bytes: &[u8]) -> Result<Document, ConvertError> {
    if !bytes.starts_with(b"{\\rtf") {
        return Err(ConvertError::malformed("not an RTF file"));
    }
    // The code page must be known before the font table decodes; scan the
    // header for \ansicpg first.
    let default_encoding = scan_codepage(bytes);
    let prelude = parse_prelude(bytes, default_encoding);
    // markitai: text in a monospaced font is code, unless that font sets
    // most of the text; the document is then read again without code (see
    // `crate::shared::code`).
    let code_fonts = !prelude.mono_fonts.is_empty();
    let mut parser = Parser::new(bytes, prelude, default_encoding);
    parser.code_fonts = code_fonts;
    parser.run()?;
    if code_fonts && !parser.share.sets_code_apart() {
        parser = Parser::new(bytes, parse_prelude(bytes, default_encoding), default_encoding);
        parser.run()?;
    }
    parser.finish()
}

fn scan_codepage(bytes: &[u8]) -> &'static encoding_rs::Encoding {
    // \ansicpg sits in the header, but generator comments and extra header
    // words can push it past any fixed prefix; the lexer scan is linear and
    // stops at the first match.
    let mut lexer = Lexer::new(bytes);
    while let Some(token) = lexer.next_token() {
        if let Token::Word { name: "ansicpg", param: Some(cp) } = token {
            return codepage_encoding(cp.max(0) as u32);
        }
    }
    encoding_rs::WINDOWS_1252
}

#[derive(Clone, Copy, PartialEq)]
enum Capture {
    None,
    ListText,
    FieldInstr,
    Bookmark,
    /// Inside a `\pict` destination: bytes route to the picture collector.
    Pict,
    /// Inside a math zone: text routes to the math element under
    /// construction.
    Math,
    /// markitai: inside a form field's `\*\ffl`: text is a drop-down
    /// list's entry.
    FormEntry,
    /// markitai: inside a shape property's name (`\sn`) or value (`\sv`).
    PropName,
    PropValue,
}

#[derive(Clone, Copy)]
struct CharState {
    style: Style,
    font: Option<i32>,
    uc_skip: u32,
    in_table: bool,
    itap: usize,
    ilvl: usize,
    ls: Option<i32>,
    legacy_list: Option<MarkerKind>,
    outline: Option<u8>,
    /// The block container this paragraph's style names.
    block: Option<BlockStyle>,
    /// Emphasis the paragraph style itself carries, subtracted from headings.
    style_base: Style,
    suppress: bool,
    capture: Capture,
    note: Option<NoteKind>,
    /// markitai: `\super` or `\sub` text, until `\nosupersub` or `\plain`.
    script: Option<Script>,
    /// markitai: the text size (`\fs`), in half-points.
    size: Size,
    /// markitai: the paragraph's tab stops (`\tx`, until `\pard`), and the
    /// alignment (`\tqr`, `\tqc`, `\tqdec`) and leader (`\tldot`, ...) the
    /// next stop takes; see [`crate::shared::tabs`].
    stops: Stops,
    tab_align: &'static str,
    tab_leader: bool,
    /// markitai: the paragraph's indents (`\li`, `\fi`, until `\pard`); see
    /// [`crate::shared::typed_lists`].
    indent: Indent,
    /// markitai: hidden text (`\v`) and text whose deletion is tracked
    /// (`\deleted`), until turned off or `\plain`. Neither is shown, as for
    /// Word's `w:vanish` and `w:del`.
    hidden: bool,
    deleted: bool,
    /// markitai: inside `\upr`, the `suppress` and `capture` its group
    /// opened with; its `\ud` destination reads with them again.
    upr: Option<(bool, Capture)>,
    /// markitai: inside a destination left out as a whole (headers,
    /// footers, comments, document information): the destinations that
    /// show text inside a suppressed one (a shape's `\shptxt`, an object's
    /// `\result`, `\shppict`, a footnote) stay suppressed there.
    excluded: bool,
}

impl Default for CharState {
    fn default() -> Self {
        CharState {
            style: Style::PLAIN,
            font: None,
            uc_skip: 1,
            in_table: false,
            itap: 1,
            ilvl: 0,
            ls: None,
            legacy_list: None,
            outline: None,
            block: None,
            style_base: Style::PLAIN,
            suppress: false,
            capture: Capture::None,
            note: None,
            script: None,
            size: 24,
            stops: Stops::default(),
            tab_align: "left",
            tab_leader: false,
            indent: Indent::default(),
            hidden: false,
            deleted: false,
            upr: None,
            excluded: false,
        }
    }
}

struct FieldFrame {
    depth: usize,
    instr: String,
    start: usize,
    /// markitai: the field's `\*\formfield` data, and whether the field
    /// started in shown text (a hidden check box shows nothing).
    form: Option<FormField>,
    shown: bool,
}

struct NoteFrame {
    depth: usize,
    start: usize,
    kind: NoteKind,
}

/// Per-instance numbering counters with restart-on-shallower semantics.
#[derive(Default)]
struct Counters {
    state: HashMap<i32, ([u64; LIST_LEVELS], [bool; LIST_LEVELS])>,
}

impl Counters {
    fn next(&mut self, ls: i32, level: usize, start: u64) -> u64 {
        let level = level.min(LIST_LEVELS - 1);
        let (values, started) = self.state.entry(ls).or_default();
        let value = if started[level] { values[level].saturating_add(1) } else { start };
        values[level] = value;
        started[level] = true;
        for s in started.iter_mut().skip(level + 1) {
            *s = false;
        }
        value
    }

    /// Advance a table-defined list level and render its composite label
    /// against the live counter values.
    fn next_labeled(
        &mut self,
        ls: i32,
        level: usize,
        levels: &[tables::ListLevelDef; LIST_LEVELS],
    ) -> (u64, Option<String>) {
        let level = level.min(LIST_LEVELS - 1);
        let def = &levels[level];
        let value = self.next(ls, level, def.start);
        let (values, started) = self.state.entry(ls).or_default();
        let label = def.marker.and_then(|marker| {
            crate::shared::numbering::composite_label(
                &def.pattern,
                marker,
                value,
                |l| levels[l.min(LIST_LEVELS - 1)].marker.unwrap_or(MarkerKind::Decimal),
                |l| {
                    let l = l.min(LIST_LEVELS - 1);
                    if started[l] { values[l] } else { levels[l].start }
                },
            )
        });
        (value, label)
    }

    /// Pin a counter to a number taken from the source (legacy `\pn` labels).
    fn seed(&mut self, ls: i32, level: usize, value: u64) {
        let level = level.min(LIST_LEVELS - 1);
        let (values, started) = self.state.entry(ls).or_default();
        values[level] = value;
        started[level] = true;
        for s in started.iter_mut().skip(level + 1) {
            *s = false;
        }
    }
}

/// Byte-level text decoding: pending code-page bytes, `\uN` fallback skips,
/// and surrogate pairing.
struct TextDecoder {
    default_encoding: &'static encoding_rs::Encoding,
    pending: Vec<u8>,
    skip: u32,
    surrogate: Option<u16>,
}

impl TextDecoder {
    fn new(default_encoding: &'static encoding_rs::Encoding) -> Self {
        TextDecoder { default_encoding, pending: Vec::new(), skip: 0, surrogate: None }
    }

    /// Buffer one text byte, honoring an active `\uN` fallback skip.
    fn byte(&mut self, b: u8) {
        if self.skip > 0 {
            self.skip -= 1;
        } else {
            self.pending.push(b);
        }
    }

    /// True when an active fallback skip consumed the character.
    fn skip_char(&mut self) -> bool {
        if self.skip > 0 {
            self.skip -= 1;
            true
        } else {
            false
        }
    }

    /// Decode and take the buffered bytes with the current font's encoding.
    fn take_pending(&mut self, encoding: Option<&'static encoding_rs::Encoding>) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let bytes = std::mem::take(&mut self.pending);
        let (text, _, _) = encoding.unwrap_or(self.default_encoding).decode(&bytes);
        Some(text.into_owned())
    }

    /// markitai: the buffered bytes of text in symbol font `font`: each its
    /// glyph's character where [`symbol_char`] maps it, else decoded with
    /// `encoding` as other text is.
    fn take_symbols(
        &mut self,
        font: &str,
        encoding: Option<&'static encoding_rs::Encoding>,
    ) -> Option<String> {
        if self.pending.is_empty() {
            return None;
        }
        let encoding = encoding.unwrap_or(self.default_encoding);
        let mut text = String::new();
        for byte in std::mem::take(&mut self.pending) {
            match symbol_char(font, &format!("F0{byte:02X}")) {
                Some(c) => text.push(c),
                None => text.push_str(&encoding.decode(&[byte]).0),
            }
        }
        Some(text)
    }

    /// `\uN`: the completed scalar (surrogate pairs are held and combined),
    /// arming the `uc_skip`-length fallback skip.
    fn unicode(&mut self, param: Option<i32>, uc_skip: u32) -> Option<char> {
        // A new \u ends the previous one's fallback range.
        self.skip = 0;
        let n = param?;
        let code = if n < 0 { (n + 65536) as u32 } else { n as u32 };
        let out = match (self.surrogate.take(), code as u16) {
            // High surrogate: hold for its pair.
            (None, unit @ 0xD800..=0xDBFF) => {
                self.surrogate = Some(unit);
                None
            }
            (Some(high), low @ 0xDC00..=0xDFFF) => {
                let combined = 0x10000 + ((high as u32 - 0xD800) << 10) + (low as u32 - 0xDC00);
                char::from_u32(combined)
            }
            (_, _) => char::from_u32(code),
        };
        self.skip = uc_skip;
        out
    }
}

/// An accumulating `\pict` destination: the payload arrives as ASCII hex
/// characters (or one `\binN` run), typed by a format control word.
#[derive(Default)]
struct PictState {
    /// Group depth of the `\pict` destination itself: payload bytes live at
    /// this depth only (subgroups like `\*\picprop` carry properties).
    depth: usize,
    hex: Vec<u8>,
    binary: Option<Vec<u8>>,
    /// (media type, extension) from `\pngblip`/`\jpegblip`/`\emfblip`/
    /// `\wmetafile`; `None` = unsupported format.
    format: Option<(&'static str, &'static str)>,
    /// markitai: its alt text (`wzDescription` in `\*\picprop`).
    alt: String,
}

impl PictState {
    /// The decoded payload bytes.
    fn payload(self) -> Vec<u8> {
        if let Some(binary) = self.binary {
            return binary;
        }
        let mut out = Vec::with_capacity(self.hex.len() / 2);
        let mut high: Option<u8> = None;
        for &b in &self.hex {
            let Some(digit) = (b as char).to_digit(16) else {
                continue;
            };
            match high.take() {
                Some(h) => out.push((h << 4) | digit as u8),
                None => high = Some(digit as u8),
            }
        }
        out
    }
}

/// An accumulating `\mmath` zone. Its groups mirror OMML elements, each
/// named by its first control word (`\mf` is `m:f`), so they are built into
/// the same element tree the OOXML readers hand the converter: a property
/// group's text is its value, a run's text is its content, and a property
/// written as a parameter word on the run (`\msty2`) is a child element.
struct MathState {
    /// Group depth of the zone's own group.
    depth: usize,
    /// Open groups, innermost last, each with the depth it opened at. The
    /// first is the zone itself.
    open: Vec<(usize, Element)>,
}

impl MathState {
    fn new(depth: usize) -> Self {
        MathState { depth, open: vec![(depth, math_elem("zone"))] }
    }

    fn open_group(&mut self, depth: usize) {
        self.open.push((depth, math_elem("")));
    }

    /// Attach the groups opened deeper than `depth` to their parents. A
    /// group that named no element is transparent.
    fn close_groups(&mut self, depth: usize) {
        while self.open.len() > 1 && self.open.last().is_some_and(|(d, _)| *d > depth) {
            let (_, elem) = self.open.pop().unwrap();
            let (_, parent) = self.open.last_mut().unwrap();
            if elem.local.is_empty() {
                parent.children.extend(elem.children);
            } else {
                parent.children.push(Node::Elem(elem));
            }
        }
    }

    /// A math control word inside a group: the first names the group, a
    /// later one is a property of it, its parameter the value.
    fn word(&mut self, name: &str, param: Option<i32>) {
        let Some((_, elem)) = self.open.last_mut() else { return };
        if elem.local.is_empty() {
            elem.local = name.to_string();
            return;
        }
        let mut prop = math_elem(name);
        if let Some(param) = param {
            prop.children.push(Node::Text(param.to_string()));
        }
        elem.children.push(Node::Elem(prop));
    }

    fn push_text(&mut self, text: String) {
        if let Some((_, elem)) = self.open.last_mut() {
            elem.children.push(Node::Text(text));
        }
    }

    /// The zone's equations, and whether they are a math paragraph
    /// (`\moMathPara`), which displays on its own line.
    fn finish(mut self) -> (Vec<String>, bool) {
        self.close_groups(0);
        let (_, zone) = self.open.remove(0);
        match zone.find(ns::M, "oMathPara") {
            Some(para) => (omath_para_to_tex(para), true),
            None => (omath_para_to_tex(&zone), false),
        }
    }
}

fn math_elem(local: &str) -> Element {
    Element {
        ns: Some(Rc::from(ns::M)),
        local: local.to_string(),
        attrs: Vec::new(),
        children: Vec::new(),
    }
}

/// Every OMML element name, as its rtf control word without the `m` prefix.
const OMML_NAMES: &[&str] = &[
    "acc",
    "accPr",
    "aln",
    "alnScr",
    "argPr",
    "argSz",
    "bar",
    "barPr",
    "baseJc",
    "begChr",
    "borderBox",
    "borderBoxPr",
    "box",
    "boxPr",
    "brk",
    "brkBin",
    "brkBinSub",
    "cGp",
    "cGpRule",
    "chr",
    "count",
    "cSp",
    "ctrlPr",
    "d",
    "defJc",
    "deg",
    "degHide",
    "den",
    "diff",
    "dispDef",
    "dPr",
    "e",
    "endChr",
    "eqArr",
    "eqArrPr",
    "f",
    "fName",
    "fPr",
    "func",
    "funcPr",
    "groupChr",
    "groupChrPr",
    "grow",
    "hideBot",
    "hideLeft",
    "hideRight",
    "hideTop",
    "interSp",
    "intLim",
    "intraSp",
    "jc",
    "lim",
    "limLoc",
    "limLow",
    "limLowPr",
    "limUpp",
    "limUppPr",
    "lit",
    "lMargin",
    "m",
    "mathFont",
    "mathPr",
    "maxDist",
    "mc",
    "mcJc",
    "mcPr",
    "mcs",
    "mPr",
    "mr",
    "nary",
    "naryLim",
    "naryPr",
    "noBreak",
    "nor",
    "num",
    "objDist",
    "oMath",
    "oMathPara",
    "oMathParaPr",
    "opEmu",
    "phant",
    "phantPr",
    "plcHide",
    "pos",
    "postSp",
    "preSp",
    "r",
    "rad",
    "radPr",
    "rMargin",
    "rPr",
    "rSp",
    "rSpRule",
    "scr",
    "sepChr",
    "show",
    "shp",
    "smallFrac",
    "sPre",
    "sPrePr",
    "sSub",
    "sSubPr",
    "sSubSup",
    "sSubSupPr",
    "sSup",
    "sSupPr",
    "strikeBLTR",
    "strikeH",
    "strikeTLBR",
    "strikeV",
    "sty",
    "sub",
    "subHide",
    "sup",
    "supHide",
    "t",
    "transp",
    "type",
    "vertJc",
    "wrapIndent",
    "wrapRight",
    "zeroAsc",
    "zeroDesc",
    "zeroWid",
];

/// Field, note, bookmark, and list-label destination state: frames open at a
/// group depth and close when the group stack unwinds past it.
#[derive(Default)]
struct Destinations {
    fields: Vec<FieldFrame>,
    note_frames: Vec<NoteFrame>,
    notes: Vec<Note>,
    bookmark: String,
    /// Captured `\listtext` label for the current paragraph.
    listtext: Option<String>,
    /// The `\pict` destination currently accumulating, if any.
    pict: Option<PictState>,
    /// The math zone currently accumulating, if any.
    math: Option<MathState>,
    /// The current paragraph holds a finished math paragraph.
    math_display: bool,
    /// markitai: the shapes (`\shp`, `\shpgrp`) and shape properties
    /// (`\sp`) open around the text being read, innermost last.
    shapes: Vec<ShapeState>,
    props: Vec<PropState>,
}

/// markitai: a shape being read. Its properties sit in `\*\shpinst` and
/// a copy for readers without shapes in `\shprslt`: a picture shape keeps
/// its picture in the `pib` property (the copy is a Windows metafile of
/// it), WordArt its words in `gtextUNICODE`, and any shape its alt text in
/// `wzDescription`. Upstream read only the copy, so a floating picture came
/// out as the metafile, or not at all, and WordArt and alt text were lost.
struct ShapeState {
    depth: usize,
    alt: String,
    word_art: String,
    /// Its `pib` picture was read, so `\shprslt` is a copy of it.
    picture: bool,
    /// Its text box (`\shptxt`) was read.
    text: bool,
    /// Where in the paragraph its picture went, for alt text that follows it.
    image: Option<usize>,
}

/// markitai: a shape property (`{\sp{\sn name}{\sv value}}`) being read.
struct PropState {
    depth: usize,
    name: String,
    value: String,
}

impl Destinations {
    /// Close field frames opened deeper than `depth`, folding their results
    /// back into the inline stream.
    fn close_fields(&mut self, depth: usize, inlines: &mut Vec<Inline>) {
        while self.fields.last().is_some_and(|f| f.depth > depth) {
            let frame = self.fields.pop().unwrap();
            let start = frame.start.min(inlines.len());
            let content: Vec<Inline> = inlines.drain(start..).collect();
            // markitai: a check box or drop-down list shows its state.
            let form = frame.form.as_ref().filter(|_| frame.shown);
            inlines.extend(form_field_result(&frame.instr, form, content));
        }
    }

    /// markitai: a paragraph ends inside open fields. A hyperlink result can
    /// span paragraphs (TextEdit links a whole card: a blank paragraph, then
    /// the title and summary), and the text before each paragraph mark
    /// would otherwise leave the field unlinked, closing it with nothing in
    /// the current paragraph. Each paragraph keeps its part of the link, as
    /// a word processor shows it; the rest of the field starts afresh.
    fn split_open_links(&mut self, inlines: &mut Vec<Inline>) {
        for frame in self.fields.iter_mut().rev() {
            let start = frame.start.min(inlines.len());
            if crate::shared::fields::hyperlink_target(&frame.instr).is_some() {
                let content: Vec<Inline> = inlines.drain(start..).collect();
                inlines.extend(field_result(&frame.instr, content));
            }
        }
        for frame in &mut self.fields {
            frame.start = 0;
        }
    }

    /// Close note frames opened deeper than `depth`, replacing their content
    /// with a reference to the collected note.
    fn close_notes(&mut self, depth: usize, inlines: &mut Vec<Inline>) {
        while self.note_frames.last().is_some_and(|f| f.depth > depth) {
            let frame = self.note_frames.pop().unwrap();
            let start = frame.start.min(inlines.len());
            let content: Vec<Inline> = inlines.drain(start..).collect();
            if !inlines_are_empty(&content) {
                let id = format!("rtf{}", self.notes.len());
                self.notes.push(Note {
                    id: id.clone(),
                    kind: frame.kind,
                    blocks: vec![Block::Paragraph(content)],
                });
                inlines.push(Inline::NoteRef(id));
            }
        }
    }

    /// Emit a completed bookmark capture as an anchor.
    fn close_bookmark(&mut self, still_capturing: bool, inlines: &mut Vec<Inline>) {
        if !still_capturing && !self.bookmark.is_empty() {
            let name = std::mem::take(&mut self.bookmark);
            let name = name.trim().to_string();
            if !name.is_empty() {
                inlines.push(Inline::Anchor(name));
            }
        }
    }
}

/// Destinations whose content is excluded: headers/footers by fixed policy,
/// metadata and binary destinations because they carry no document text.
const SUPPRESSED_DESTINATIONS: &[&str] = &[
    "fonttbl",
    "colortbl",
    "stylesheet",
    "info",
    "object",
    "header",
    "footer",
    "headerl",
    "headerr",
    "headerf",
    "footerl",
    "footerr",
    "footerf",
    "ftnsep",
    "ftnsepc",
    "aftnsep",
    "aftnsepc",
    "xmlnstbl",
    "themedata",
    "colorschememapping",
    "datastore",
    "latentstyles",
    "listtable",
    "listoverridetable",
    "rsidtbl",
    "generator",
    "filetbl",
    "revtbl",
    "datafield",
    "bkmkend",
    "annotation",
    "atnid",
    "atnauthor",
    "template",
    "defchp",
    "defpap",
    "panose",
    "falt",
    "objdata",
    "blipuid",
    "nonshppict",
    "wgrffmtfilter",
    "pgdsctbl",
    "docvar",
    "shpinst",
    "background",
    "userprops",
    "operator",
    "author",
    "title",
    "subject",
    "keywords",
    "doccomm",
    "creatim",
    "revtim",
    "printim",
    // markitai: index and contents entries (`{\xe ...}`, `{\tc ...}`) are
    // hidden marks.
    "xe",
    "tc",
    "tcn",
];

struct Parser<'a> {
    lexer: Lexer<'a>,
    stack: Vec<CharState>,
    state: CharState,
    prelude: Prelude,
    decoder: TextDecoder,
    recovered: bool,

    inlines: Vec<Inline>,
    blocks: Vec<Block>,
    list_run: Vec<ListEntry>,
    /// markitai: empty paragraphs came after the list's last paragraph (they
    /// close it unless an item's continuation follows), and where the last
    /// paragraph of body text before it starts its lines, in twips.
    list_gap: bool,
    body_left: i32,
    styled: StyledRun,
    counters: Counters,
    table: TableState,
    dest: Destinations,
    assets: crate::shared::assets::AssetSink,
    /// markitai: the depth whose `\nestrow` was the last table control, with
    /// no `\itap` or `\pard` since; see the `nestcell` arm.
    nested_row_closed: Option<usize>,
    /// markitai: what the body's text looks like, and the sizes of the
    /// paragraph being read (see [`crate::shared::visual`]).
    looks: Looks,
    para_size: ParaSize,
    /// markitai: whether monospaced text is code (see
    /// [`crate::shared::code`]), how much of the body's text is monospaced,
    /// the runs of the paragraph being read, and the font of text no `\f`
    /// names (`\deff`).
    code_fonts: bool,
    share: MonoShare,
    para_fonts: RunFonts,
    default_font: Option<i32>,
    /// markitai: the body's paragraphs that may be rows of a table set with
    /// tab stops (see [`crate::shared::tabs`]), and its plain paragraphs and
    /// their indents (see [`crate::shared::typed_lists`]).
    tab_rows: TabRows,
    typed_lists: TypedLists,
    /// markitai: a picture attachment (`\NeXTGraphic`) just ended; the
    /// attachment character TextEdit writes after it is no text.
    attachment: bool,
}

impl<'a> Parser<'a> {
    fn new(
        bytes: &'a [u8],
        prelude: Prelude,
        default_encoding: &'static encoding_rs::Encoding,
    ) -> Self {
        Parser {
            lexer: Lexer::new(bytes),
            stack: Vec::new(),
            state: CharState::default(),
            prelude,
            decoder: TextDecoder::new(default_encoding),
            recovered: false,
            inlines: Vec::new(),
            blocks: Vec::new(),
            list_run: Vec::new(),
            list_gap: false,
            body_left: 0,
            styled: StyledRun::default(),
            counters: Counters::default(),
            table: TableState::new(),
            dest: Destinations::default(),
            assets: crate::shared::assets::AssetSink::new(),
            nested_row_closed: None,
            looks: Looks::default(),
            para_size: ParaSize::default(),
            code_fonts: false,
            share: MonoShare::default(),
            para_fonts: RunFonts::default(),
            default_font: None,
            tab_rows: TabRows::default(),
            typed_lists: TypedLists::default(),
            attachment: false,
        }
    }

    fn run(&mut self) -> Result<(), ConvertError> {
        while let Some(token) = self.lexer.next_token() {
            match token {
                Token::Open => {
                    self.flush_pending();
                    self.stack.push(self.state);
                    if self.state.capture == Capture::Math
                        && let Some(math) = &mut self.dest.math
                    {
                        math.open_group(self.stack.len());
                    }
                }
                Token::Close => {
                    self.flush_pending();
                    match self.stack.pop() {
                        Some(prev) => self.state = prev,
                        None => self.recovered = true,
                    }
                    let depth = self.stack.len();
                    self.dest.close_fields(depth, &mut self.inlines);
                    self.dest.close_notes(depth, &mut self.inlines);
                    self.dest
                        .close_bookmark(self.state.capture == Capture::Bookmark, &mut self.inlines);
                    // The pict destination finishes when the stack unwinds
                    // past its own group (inner property groups close first).
                    if self.dest.pict.as_ref().is_some_and(|p| self.stack.len() < p.depth) {
                        self.finish_pict()?;
                    }
                    // markitai: and shape properties and shapes likewise.
                    while self.dest.props.last().is_some_and(|p| self.stack.len() < p.depth) {
                        self.finish_prop();
                    }
                    while self.dest.shapes.last().is_some_and(|s| self.stack.len() < s.depth) {
                        self.finish_shape();
                    }
                    if let Some(math) = &mut self.dest.math {
                        if self.stack.len() < math.depth {
                            self.finish_math();
                        } else {
                            math.close_groups(self.stack.len());
                        }
                    }
                }
                Token::Word { name, param } => self.control_word(name, param)?,
                Token::Symbol(b) => self.control_symbol(b),
                Token::Hex(b) | Token::Byte(b) => {
                    if self.state.capture == Capture::Pict {
                        if let Some(pict) = &mut self.dest.pict
                            && self.stack.len() == pict.depth
                        {
                            pict.hex.push(b);
                        }
                    } else if self.accepts_text() {
                        self.decoder.byte(b);
                    }
                }
                Token::Bin(payload) => {
                    if self.state.capture == Capture::Pict
                        && let Some(pict) = &mut self.dest.pict
                        && self.stack.len() == pict.depth
                    {
                        pict.binary = Some(payload.to_vec());
                    } else {
                        log::debug!("skipping {} bytes of embedded binary data", payload.len());
                    }
                }
            }
        }
        if !self.stack.is_empty() {
            self.recovered = true;
        }
        if self.recovered {
            log::warn!("recovered unbalanced rtf groups");
        }
        self.flush_pending();
        self.finish_math();
        self.end_paragraph()
    }

    fn accepts_text(&self) -> bool {
        !self.state.excluded && (self.state.capture != Capture::None || !self.state.suppress)
    }

    fn control_symbol(&mut self, b: u8) {
        match b {
            b'~' => self.push_char('\u{a0}'),
            b'-' => {}
            b'_' => self.push_char('-'),
            b'*' => self.state.suppress = true,
            b'\\' | b'{' | b'}' => self.push_char(b as char),
            _ => {}
        }
    }

    /// Dispatch a control word to its subsystem handler; unhandled words
    /// that name a suppressed destination silence their group.
    fn control_word(&mut self, word: &str, param: Option<i32>) -> Result<(), ConvertError> {
        if self.math_control(word, param) {
            return Ok(());
        }
        if self.text_control(word, param)? {
            return Ok(());
        }
        if self.table_control(word, param)? {
            return Ok(());
        }
        if self.list_control(word, param) {
            return Ok(());
        }
        if self.object_control(word, param) {
            return Ok(());
        }
        if SUPPRESSED_DESTINATIONS.contains(&word) {
            self.flush_pending();
            self.state.suppress = true;
            self.state.capture = Capture::None;
            // markitai: a shape's properties and an object's data hold
            // destinations that show text; everything else is left out
            // whole.
            if !matches!(word, "shpinst" | "object") {
                self.state.excluded = true;
            }
        }
        Ok(())
    }

    /// Character- and paragraph-level controls: text encoding, styling,
    /// paragraph and line breaks, and special characters.
    fn text_control(&mut self, word: &str, param: Option<i32>) -> Result<bool, ConvertError> {
        let on = param != Some(0);
        match word {
            "u" => {
                if self.accepts_text() {
                    self.flush_pending();
                    if let Some(c) = self.decoder.unicode(param, self.state.uc_skip) {
                        let text = self.symbol_unicode(c);
                        self.push_text(text);
                    }
                }
            }
            "uc" => self.state.uc_skip = param.unwrap_or(1).max(0) as u32,
            "f" => {
                self.flush_pending();
                self.state.font = param;
            }
            "b" => self.set_style(|s| s.bold = on),
            // markitai: the text size, for headings set by hand.
            "fs" => {
                self.flush_pending();
                self.state.size = param.unwrap_or(24).clamp(1, 3276) as Size;
            }
            "i" => self.set_style(|s| s.italic = on),
            "strike" | "striked" => self.set_style(|s| s.strike = on),
            "plain" => {
                self.flush_pending();
                let font = self.state.font;
                self.state.style = Style::PLAIN;
                self.state.font = font;
                self.state.script = None;
                self.state.size = 24;
                self.state.hidden = false;
                self.state.deleted = false;
            }
            // markitai: hidden text and tracked deletions show nothing.
            "v" => {
                self.flush_pending();
                self.state.hidden = on;
            }
            "deleted" => {
                self.flush_pending();
                self.state.deleted = on;
            }
            // markitai: `{\upr{ansi}{\*\ud{unicode}}}` gives text twice, in
            // the code page and in Unicode (characters the code page lacks
            // are `?` in the first); only the Unicode one is read.
            "upr" => {
                self.flush_pending();
                self.state.upr = Some((self.state.suppress, self.state.capture));
                self.state.suppress = true;
                self.state.capture = Capture::None;
            }
            "ud" => {
                if let Some((suppress, capture)) = self.state.upr.take() {
                    self.flush_pending();
                    self.state.suppress = suppress;
                    self.state.capture = capture;
                }
            }
            // markitai: raised and lowered text in its Unicode forms where
            // every character has one ("x₁", "library¹"), as for Word's
            // `w:vertAlign`; see `crate::formats::docx::scripts`.
            "super" | "sub" | "nosupersub" => {
                self.flush_pending();
                self.state.script = match word {
                    _ if param == Some(0) => None,
                    "super" => Some(Script::Superscript),
                    "sub" => Some(Script::Subscript),
                    _ => None,
                };
            }
            "s" => {
                // Paragraph style: outline level for headings plus its
                // formatting delta as the new base.
                let def = param.and_then(|id| self.prelude.styles.get(&id).copied());
                if let Some(def) = def {
                    self.flush_pending();
                    self.state.outline = def.outline;
                    self.state.block = def.block;
                    self.state.style = def.delta.apply(self.state.style);
                    self.state.style_base = self.state.style;
                    // markitai: the style's text size.
                    if let Some(size) = def.size {
                        self.state.size = size;
                    }
                }
            }
            "par" | "sect" => {
                self.flush_pending();
                // markitai: a hidden or deleted paragraph mark joins its
                // paragraph to the next one, as Word shows it.
                let shown = !self.hidden();
                if shown && self.state.note.is_some() {
                    self.inlines.push(Inline::LineBreak);
                } else if shown && !self.state.suppress {
                    self.end_paragraph()?;
                }
            }
            "pard" => {
                self.flush_pending();
                self.nested_row_closed = None;
                self.state.in_table = false;
                self.state.itap = 1;
                self.state.ilvl = 0;
                self.state.ls = None;
                self.state.legacy_list = None;
                self.state.outline = None;
                self.state.block = None;
                self.state.style_base = Style::PLAIN;
                // markitai: and the tab stops and indents.
                self.state.stops = Stops::default();
                self.state.tab_align = "left";
                self.state.tab_leader = false;
                self.state.indent = Indent::default();
            }
            // markitai: the paragraph's indents, for the lists typed by
            // hand.
            "li" | "lin" => self.state.indent.left = param.unwrap_or(0),
            "fi" => self.state.indent.first_line = param.unwrap_or(0),
            // markitai: the paragraph's tab stops, for the tables set with
            // them; a bar tab only draws a line.
            "tx" => {
                let align = std::mem::replace(&mut self.state.tab_align, "left");
                let leader = std::mem::take(&mut self.state.tab_leader);
                if let Some(position) = param {
                    self.state.stops.add(position, align, leader);
                }
            }
            "tb" => {
                self.state.tab_align = "left";
                self.state.tab_leader = false;
            }
            "tqr" => self.state.tab_align = "right",
            "tqc" => self.state.tab_align = "center",
            "tqdec" => self.state.tab_align = "decimal",
            "tldot" | "tlmdot" | "tlhyph" | "tlul" | "tlth" | "tleq" => {
                self.state.tab_leader = true;
            }
            // markitai: the font of text no `\f` names.
            "deff" => self.default_font = param,
            // \page and \column break the flow without ending the
            // paragraph; the page they start is unrepresentable, the word
            // boundary they carry is not.
            "line" | "lbr" | "page" | "column" => {
                self.flush_pending();
                if !self.state.suppress && !self.hidden() {
                    self.inlines.push(Inline::LineBreak);
                }
            }
            "tab" => self.push_tab(),
            "emdash" => self.push_char('\u{2014}'),
            "endash" => self.push_char('\u{2013}'),
            "lquote" => self.push_char('\u{2018}'),
            "rquote" => self.push_char('\u{2019}'),
            "ldblquote" => self.push_char('\u{201c}'),
            "rdblquote" => self.push_char('\u{201d}'),
            "bullet" => self.push_char('\u{2022}'),
            "enspace" | "emspace" | "qmspace" => self.push_char(' '),
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// Table controls, delegated to the per-depth [`TableState`].
    fn table_control(&mut self, word: &str, param: Option<i32>) -> Result<bool, ConvertError> {
        match word {
            "intbl" => self.state.in_table = true,
            "itap" => {
                self.nested_row_closed = None;
                self.state.itap = param.unwrap_or(1).clamp(0, 8) as usize;
                if self.state.itap > 1 {
                    self.state.in_table = true;
                }
            }
            "trowd" => {
                if self.table_active() {
                    let depth = self.state.itap.max(1);
                    self.table.begin_row(depth);
                }
            }
            "trhdr" => {
                if self.table_active() {
                    let depth = self.state.itap.max(1);
                    self.table.mark_header_row(depth);
                }
            }
            "clmgf" => self.pending_cell_prop(|p| p.merge_first = true),
            "clmrg" => self.pending_cell_prop(|p| p.merge_cont = true),
            "clvmgf" => self.pending_cell_prop(|p| p.vmerge_first = true),
            "clvmrg" => self.pending_cell_prop(|p| p.vmerge_cont = true),
            "cellx" => {
                if self.table_active() {
                    let depth = self.state.itap.max(1);
                    self.table.declare_cell(depth, param.unwrap_or(0) as i64);
                }
            }
            "cell" => {
                self.flush_pending();
                if self.table_active() {
                    self.end_cell(1)?;
                }
            }
            "nestcell" => {
                self.flush_pending();
                if self.table_active() {
                    let mut depth = self.state.itap.max(2);
                    // markitai: a `\nestcell` straight after a `\nestrow`,
                    // with no `\itap` between them, closes the cell that
                    // holds the finished table. TextEdit ends a table nested
                    // two deep with `\nestcell \lastrow\nestrow\nestcell
                    // \nestrow` and never writes `\itap2` again; read at
                    // depth 3, the outer cell never closed and every row of
                    // the inner table (Hacker News comments) was lost.
                    if self.nested_row_closed.take() == Some(depth) && depth > 2 {
                        depth -= 1;
                        self.state.itap = depth;
                    }
                    self.end_cell(depth)?;
                }
            }
            "row" => {
                self.flush_pending();
                if self.table_active() {
                    self.end_row(1)?;
                }
            }
            "nestrow" => {
                self.flush_pending();
                if self.table_active() {
                    let depth = self.state.itap.max(2);
                    self.end_row(depth)?;
                    self.nested_row_closed = Some(depth);
                }
            }
            // Nested row properties arrive in a `{\*\nesttableprops ...}`
            // destination; its \trowd/\cellx/\nestrow must still act.
            "nesttableprops" => self.state.suppress = self.state.excluded,
            _ => return Ok(false),
        }
        Ok(true)
    }

    /// List and outline controls.
    fn list_control(&mut self, word: &str, param: Option<i32>) -> bool {
        match word {
            "outlinelevel" => {
                if let Some(n) = param
                    && (0..9).contains(&n)
                {
                    self.state.outline = Some((n + 1) as u8);
                }
            }
            "ilvl" => self.state.ilvl = param.unwrap_or(0).clamp(0, 8) as usize,
            "ls" => self.state.ls = param,
            "listtext" | "pntext" => {
                self.flush_pending();
                self.state.capture = Capture::ListText;
                if self.dest.listtext.is_none() {
                    self.dest.listtext = Some(String::new());
                }
            }
            "pnlvlblt" => self.state.legacy_list = Some(MarkerKind::Bullet),
            "pnlvlbody" | "pndec" => self.state.legacy_list = Some(MarkerKind::Decimal),
            _ => return false,
        }
        true
    }

    /// Field, footnote, and bookmark controls.
    fn object_control(&mut self, word: &str, param: Option<i32>) -> bool {
        match word {
            "field" => {
                self.flush_pending();
                if !self.state.suppress {
                    self.dest.fields.push(FieldFrame {
                        depth: self.stack.len(),
                        instr: String::new(),
                        start: self.inlines.len(),
                        form: None,
                        shown: !self.hidden(),
                    });
                }
            }
            // markitai: a legacy form field's data, inside its instruction:
            // the kind (`\fftype` 0 text, 1 check box, 2 drop-down list),
            // the state (`\ffres`, else `\ffdefres`) and a list's entries
            // (`\*\ffl`); see [`FormField`]. The field's name, default text,
            // help and macros are not shown.
            "formfield" => {
                if let Some(frame) = self.dest.fields.last_mut() {
                    frame.form = Some(FormField::default());
                }
            }
            "fftype" | "ffres" | "ffdefres" => {
                if let Some(form) = self.dest.fields.last_mut().and_then(|f| f.form.as_mut()) {
                    match word {
                        "fftype" => {
                            form.kind = match param {
                                Some(1) => Some(FormKind::CheckBox),
                                Some(2) => Some(FormKind::DropDown),
                                _ => Some(FormKind::Text),
                            }
                        }
                        "ffres" => form.result = param,
                        _ => form.default = param,
                    }
                }
            }
            "ffl" => {
                self.flush_pending();
                if let Some(form) = self.dest.fields.last_mut().and_then(|f| f.form.as_mut()) {
                    form.entries.push(String::new());
                    self.state.capture = Capture::FormEntry;
                }
            }
            "ffname" | "ffdeftext" | "ffformat" | "ffhelptext" | "ffstattext" | "ffentrymcr"
            | "ffexitmcr" => {
                self.flush_pending();
                self.state.capture = Capture::None;
                self.state.suppress = true;
            }
            "fldinst" => {
                self.flush_pending();
                if !self.dest.fields.is_empty() {
                    self.state.capture = Capture::FieldInstr;
                }
                self.state.suppress = true;
            }
            "fldrslt" => {
                self.flush_pending();
                self.state.capture = Capture::None;
            }
            "footnote" => {
                self.flush_pending();
                self.state.note = Some(NoteKind::Footnote);
                self.state.suppress = self.state.excluded;
                self.state.capture = Capture::None;
                self.dest.note_frames.push(NoteFrame {
                    depth: self.stack.len(),
                    start: self.inlines.len(),
                    kind: NoteKind::Footnote,
                });
            }
            "ftnalt" => {
                // Marks the enclosing \footnote as an endnote.
                if let Some(frame) = self.dest.note_frames.last_mut() {
                    frame.kind = NoteKind::Endnote;
                }
            }
            "chftn" => {}
            "bkmkstart" => {
                self.flush_pending();
                self.state.capture = Capture::Bookmark;
                self.state.suppress = self.state.excluded;
            }
            // `{\*\shppict {\pict ...}}` wraps the preferred picture; the
            // `\nonshppict` fallback duplicate stays suppressed. markitai:
            // these destinations show nothing inside a header or footer
            // (they did: a letterhead's logo or text box ran into the body).
            "shppict" => self.state.suppress = self.state.excluded,
            // `\shpinst` itself is suppressed (shape properties), but its
            // `\shptxt` destination holds the shape's real text.
            "shptxt" => {
                self.state.suppress = self.state.excluded;
                if let Some(shape) = self.dest.shapes.last_mut() {
                    shape.text = true;
                }
            }
            // Likewise `\object` is suppressed (class names, `\objdata`
            // payload), but `\result` is the object's displayable rendering.
            "result" => self.state.suppress = self.state.excluded,
            // markitai: a picture TextEdit attached (`{{\NeXTGraphic name
            // \width..}\'ac}` in an RTFD's text) names a file beside the RTF,
            // which a reader of the RTF alone does not have: its file name
            // and the attachment character after it are no text (they were
            // read into the sentence).
            "NeXTGraphic" => {
                self.flush_pending();
                self.state.suppress = true;
                self.state.excluded = true;
                self.state.capture = Capture::None;
                self.attachment = true;
            }
            // markitai: shapes and their properties (see `ShapeState`).
            "shp" | "shpgrp" => {
                if !self.state.excluded {
                    self.dest.shapes.push(ShapeState {
                        depth: self.stack.len(),
                        alt: String::new(),
                        word_art: String::new(),
                        picture: false,
                        text: false,
                        image: None,
                    });
                }
            }
            "sp" => {
                self.flush_pending();
                self.state.suppress = true;
                self.state.capture = Capture::None;
                self.dest.props.push(PropState {
                    depth: self.stack.len(),
                    name: String::new(),
                    value: String::new(),
                });
            }
            "sn" | "sv" => {
                self.flush_pending();
                self.state.suppress = true;
                self.state.capture = Capture::None;
                if let Some(prop) = self.dest.props.last() {
                    if word == "sn" {
                        self.state.capture = Capture::PropName;
                    } else if prop.name.trim() == "pib" {
                        // The shape's picture is a `\pict` in the value.
                        self.state.suppress = self.state.excluded;
                    } else {
                        self.state.capture = Capture::PropValue;
                    }
                }
            }
            // The copy of a shape whose picture or WordArt was read.
            "shprslt" => {
                if self.dest.shapes.last().is_some_and(|shape| {
                    shape.picture || (!shape.word_art.trim().is_empty() && !shape.text)
                }) {
                    self.state.suppress = true;
                }
            }
            "pict" => {
                // A pict inside a suppressed destination (the nonshppict
                // fallback, excluded headers) is not extracted, nor
                // (markitai) a hidden or deleted one.
                if !self.state.suppress && !self.hidden() {
                    self.flush_pending();
                    self.state.capture = Capture::Pict;
                    self.dest.pict =
                        Some(PictState { depth: self.stack.len(), ..PictState::default() });
                }
            }
            "pngblip" => self.set_pict_format(Some(("image/png", "png"))),
            "jpegblip" => self.set_pict_format(Some(("image/jpeg", "jpg"))),
            "emfblip" => self.set_pict_format(Some(("image/emf", "emf"))),
            "wmetafile" => self.set_pict_format(Some(("image/wmf", "wmf"))),
            "macpict" | "dibitmap" | "wbitmap" => self.set_pict_format(None),
            _ => return false,
        }
        true
    }

    /// Math zone controls: `\mmath` opens a zone, and inside one every math
    /// control word names the group it starts or sets a property on it.
    /// The `\mmathPict` fallback picture is skipped.
    fn math_control(&mut self, word: &str, param: Option<i32>) -> bool {
        if self.dest.math.is_none() {
            if word != "mmath" {
                return false;
            }
            if !self.state.suppress && !self.hidden() {
                self.flush_pending();
                self.state.capture = Capture::Math;
                self.dest.math = Some(MathState::new(self.stack.len()));
            }
            return true;
        }
        if self.state.capture != Capture::Math {
            return false;
        }
        if word == "mmathPict" {
            self.flush_pending();
            self.state.capture = Capture::None;
            self.state.suppress = true;
            return true;
        }
        match word.strip_prefix('m') {
            Some(name) if OMML_NAMES.contains(&name) => {
                self.flush_pending();
                if let Some(math) = &mut self.dest.math {
                    math.word(name, param);
                }
                true
            }
            _ => false,
        }
    }

    fn finish_math(&mut self) {
        let Some(math) = self.dest.math.take() else {
            return;
        };
        let (lines, display) = math.finish();
        for (i, tex) in lines.into_iter().enumerate() {
            if i > 0 {
                self.inlines.push(Inline::LineBreak);
            }
            self.inlines.push(Inline::Math(tex));
        }
        if display {
            self.dest.math_display = true;
        }
    }

    fn set_pict_format(&mut self, format: Option<(&'static str, &'static str)>) {
        if self.state.capture == Capture::Pict
            && let Some(pict) = &mut self.dest.pict
        {
            pict.format = format;
        }
    }

    /// Finalize a closed `\pict` destination: retain the payload as an
    /// asset and emit its inline reference. Unsupported formats degrade
    /// with a log.
    fn finish_pict(&mut self) -> Result<(), ConvertError> {
        let Some(mut pict) = self.dest.pict.take() else {
            return Ok(());
        };
        let Some((media_type, extension)) = pict.format else {
            log::debug!("skipping picture in an unsupported format");
            return Ok(());
        };
        let pict_alt = std::mem::take(&mut pict.alt);
        let bytes = pict.payload();
        if bytes.is_empty() {
            return Ok(());
        }
        let part = format!("pict/{}.{extension}", self.assets.assets.len());
        let id = self.assets.add(media_type.to_string(), part, &bytes)?;
        // markitai: its alt text, else its shape's (a picture inside the
        // shape's text box is not the shape's); a shape's `pib` picture
        // makes the shape's copy redundant.
        let mut alt = clean_text(pict_alt.trim());
        if let Some(shape) = self.dest.shapes.last_mut().filter(|shape| !shape.text) {
            if alt.is_empty() {
                alt = clean_text(shape.alt.trim());
            }
            shape.image = Some(self.inlines.len());
            if self.dest.props.last().is_some_and(|prop| prop.name.trim() == "pib") {
                shape.picture = true;
            }
        }
        self.inlines.push(Inline::Image { alt, source: crate::model::ImageSource::Asset(id) });
        Ok(())
    }

    /// markitai: a shape property was read (see [`ShapeState`]).
    fn finish_prop(&mut self) {
        let Some(prop) = self.dest.props.pop() else {
            return;
        };
        match prop.name.trim() {
            // A `\*\picprop` describes its picture, a `\*\shpinst` its shape.
            "wzDescription" => match &mut self.dest.pict {
                Some(pict) if pict.depth < prop.depth => pict.alt = prop.value,
                _ => {
                    if let Some(shape) = self.dest.shapes.last_mut() {
                        shape.alt = prop.value;
                    }
                }
            },
            "gtextUNICODE" => {
                if let Some(shape) = self.dest.shapes.last_mut() {
                    shape.word_art = prop.value;
                }
            }
            _ => {}
        }
    }

    /// markitai: a shape ended: WordArt shows its words where the shape
    /// stands, and alt text read after the shape's picture goes onto it.
    fn finish_shape(&mut self) {
        let Some(shape) = self.dest.shapes.pop() else {
            return;
        };
        let words = clean_text(shape.word_art.trim());
        if !words.is_empty() && !shape.text && !self.hidden() {
            self.inlines.push(Inline::Text { text: words, style: Style::PLAIN });
        }
        let alt = clean_text(shape.alt.trim());
        if !alt.is_empty()
            && let Some(Inline::Image { alt: image_alt, .. }) =
                shape.image.and_then(|i| self.inlines.get_mut(i))
            && image_alt.is_empty()
        {
            *image_alt = alt;
        }
    }

    /// Table controls act outside suppressed groups and note bodies.
    fn table_active(&self) -> bool {
        !self.state.suppress && self.state.note.is_none()
    }

    fn pending_cell_prop(&mut self, f: impl FnOnce(&mut table::CellProp)) {
        if self.table_active() {
            let depth = self.state.itap.max(1);
            f(self.table.pending_prop(depth));
        }
    }

    fn set_style(&mut self, f: impl FnOnce(&mut Style)) {
        self.flush_pending();
        f(&mut self.state.style);
    }

    fn push_char(&mut self, c: char) {
        if !self.accepts_text() {
            return;
        }
        if self.decoder.skip_char() {
            return;
        }
        self.flush_pending();
        self.push_text(c.to_string());
    }

    /// markitai: a tab. In body text it is kept until the tables set with
    /// tab stops are found (see `crate::shared::tabs`); anywhere else it
    /// is the space it was.
    fn push_tab(&mut self) {
        if self.state.capture != Capture::None || self.state.suppress {
            self.push_char(' ');
            return;
        }
        if self.decoder.skip_char() || self.hidden() {
            return;
        }
        self.flush_pending();
        let style = self.text_style();
        self.inlines.push(tabs::tab(style));
    }

    /// markitai: whether the text being read is hidden (`\v`) or deleted
    /// (`\deleted`), which Word does not show.
    fn hidden(&self) -> bool {
        self.state.hidden || self.state.deleted
    }

    /// markitai: a `\uN` character in a symbol font: Word writes such a
    /// font's glyph as its byte in the private-use block (`\u-3913` is
    /// `F0B7`, the Symbol bullet), mapped as the font's bytes are (see
    /// [`crate::formats::docx::symbols::symbol_text`]).
    fn symbol_unicode(&self, c: char) -> String {
        match self.state.font.and_then(|f| self.prelude.symbol_fonts.get(&f).copied()) {
            Some(font) => {
                crate::formats::docx::symbols::symbol_text(font, c.encode_utf8(&mut [0; 4]))
            }
            None => c.to_string(),
        }
    }

    /// markitai: whether the text being read is set in a monospaced font.
    fn font_mono(&self) -> bool {
        self.state.font.or(self.default_font).is_some_and(|f| self.prelude.mono_fonts.contains(&f))
    }

    /// markitai: the style of the text being read: monospaced text is code.
    fn text_style(&self) -> Style {
        Style {
            code: self.state.style.code || (self.code_fonts && self.font_mono()),
            ..self.state.style
        }
    }

    fn flush_pending(&mut self) {
        let encoding = self.state.font.and_then(|f| self.prelude.fonts.get(&f).copied());
        // markitai: a symbol font's bytes are its glyphs (Word writes a
        // Wingdings square bullet as `\'a7`, not `§`); those the DOCX
        // reader's table maps become their characters, the rest decode as
        // before.
        let symbol = self.state.font.and_then(|f| self.prelude.symbol_fonts.get(&f).copied());
        let text = match symbol {
            Some(font) => self.decoder.take_symbols(font, encoding),
            None => self.decoder.take_pending(encoding),
        };
        if let Some(text) = text {
            self.push_text(text);
        }
    }

    fn push_text(&mut self, text: String) {
        let text = clean_text(&text);
        if text.is_empty() {
            return;
        }
        match self.state.capture {
            Capture::ListText => {
                if let Some(lt) = &mut self.dest.listtext {
                    lt.push_str(&text);
                }
            }
            Capture::FieldInstr => {
                if let Some(f) = self.dest.fields.last_mut() {
                    f.instr.push_str(&text);
                }
            }
            Capture::Bookmark => self.dest.bookmark.push_str(&text),
            Capture::PropName | Capture::PropValue => {
                if let Some(prop) = self.dest.props.last_mut() {
                    match self.state.capture {
                        Capture::PropName => prop.name.push_str(&text),
                        _ => prop.value.push_str(&text),
                    }
                }
            }
            Capture::FormEntry => {
                if let Some(entry) = self
                    .dest
                    .fields
                    .last_mut()
                    .and_then(|f| f.form.as_mut())
                    .and_then(|form| form.entries.last_mut())
                {
                    entry.push_str(&text);
                }
            }
            // Picture payload bytes are collected raw in the token loop.
            Capture::Pict => {}
            Capture::Math => {
                if let Some(math) = &mut self.dest.math {
                    math.push_text(text);
                }
            }
            Capture::None => {
                if !self.state.suppress && !self.hidden() {
                    // markitai: the character a picture attachment leaves.
                    let text = if std::mem::take(&mut self.attachment) {
                        match text.strip_prefix(['\u{AC}', '\u{FFFC}']) {
                            Some("") => return,
                            Some(rest) => rest.to_string(),
                            None => text,
                        }
                    } else {
                        text
                    };
                    let text =
                        self.state.script.and_then(|script| script.convert(&text)).unwrap_or(text);
                    // markitai: a note's text is not the body's; text set in
                    // a monospaced font is code.
                    if self.state.note.is_none() {
                        self.looks.text(&mut self.para_size, self.state.size, &text);
                        let mono = self.font_mono();
                        self.share.text(&text, mono);
                        self.para_fonts.run(text.trim().is_empty(), mono);
                    }
                    self.inlines.push(Inline::Text { text, style: self.text_style() });
                }
            }
        }
    }

    /// Flush the finished top-level table, if any, into the block stream.
    fn flush_top_table(&mut self) -> Result<(), ConvertError> {
        if let Some(block) = self.table.take_table(1)? {
            self.flush_runs();
            self.blocks.push(block);
        }
        Ok(())
    }

    fn end_paragraph(&mut self) -> Result<(), ConvertError> {
        self.dest.split_open_links(&mut self.inlines);
        let inlines = std::mem::take(&mut self.inlines);
        // markitai: the sizes and fonts this paragraph's text gathered.
        let size = std::mem::take(&mut self.para_size);
        let fonts = std::mem::take(&mut self.para_fonts);
        let listtext = self.dest.listtext.take();
        let math_display = std::mem::take(&mut self.dest.math_display);

        if self.state.in_table {
            let depth = self.state.itap.max(1);
            if let Some(inlines) = self.cell_list_entry(depth, inlines, listtext.as_deref())? {
                // markitai: a blank line of a listing in the cell.
                if self.code_fonts
                    && self.state.block.is_none()
                    && inlines_are_empty(&inlines)
                    && self.font_mono()
                {
                    self.table.push_cell_blank_line(depth)?;
                } else {
                    self.table.push_cell_paragraph(depth, self.state.block, inlines)?;
                }
            }
            return Ok(());
        }
        self.flush_top_table()?;

        // A styled container absorbs its blank paragraphs: they are the
        // blank lines of a code block.
        if let Some(style) = self.state.block {
            self.close_list();
            self.styled.push(style, inlines, &mut self.blocks);
            return Ok(());
        }
        // markitai: a paragraph all in a monospaced font (its text runs, or
        // without text the font at its mark: a blank line of a listing) is a
        // line of code, unless it is a heading or a list item; one holding
        // an image or a displayed formula never is.
        let mono = self.code_fonts
            && !math_display
            && !has_image(&inlines)
            && fonts.all_mono().unwrap_or_else(|| inlines_are_empty(&inlines) && self.font_mono());
        let listed = self.state.ls.is_some()
            || self.state.legacy_list.is_some()
            || listtext.as_deref().is_some_and(|text| !text.trim().is_empty());
        // markitai: one opening with a bullet is an item of a list typed by
        // hand (see `crate::shared::typed_lists`).
        if mono && self.state.outline.is_none() && !listed && !opens_with_bullet(&inlines) {
            self.close_list();
            self.styled.push(BlockStyle::Code, inlines, &mut self.blocks);
            return Ok(());
        }
        // markitai: an empty paragraph after a list closes it, as before,
        // unless an item's continuation follows.
        if inlines_are_empty(&inlines) {
            if self.list_run.is_empty() {
                self.flush_runs();
            } else {
                self.list_gap = true;
            }
            return Ok(());
        }
        // Numbering identity comes from the list tables; the captured label
        // text only seeds legacy Word-95 numbering. Numbered headings
        // advance the sequence and keep their number visible.
        let entry = self.list_entry(listtext.as_deref());
        if let Some(level) = self.state.outline {
            self.flush_runs();
            let mut content = inlines;
            rebase_emphasis(&mut content, self.state.style_base);
            // markitai: a heading set in a monospaced font is no code.
            if mono {
                without_code(&mut content);
            }
            if let Some((key, _, number, label)) = &entry
                && key.marker.ordered()
            {
                let text =
                    format!("{} ", label.clone().unwrap_or_else(|| key.marker.label(*number)));
                content.insert(0, Inline::Text { text, style: Style::PLAIN });
            }
            self.blocks.push(Block::Heading { level, anchor: None, content });
            return Ok(());
        }
        if let Some((key, level, number, label)) = entry {
            if self.list_gap {
                self.flush_runs();
            }
            self.list_run.push(ListEntry {
                level,
                key,
                number,
                label,
                blocks: vec![Block::Paragraph(inlines)],
                indent: Some(self.state.indent.left),
                continues: false,
            });
            return Ok(());
        }
        // markitai: a paragraph set in as far as an item's text continues
        // it (see `crate::shared::list::continuation_level`); `\pard`
        // resets the indent, so body text after the list starts at the
        // body's.
        if !math_display
            && let Some(level) =
                continuation_level(&self.list_run, self.state.indent.left, self.body_left)
        {
            self.list_run.push(ListEntry::continuation(level, vec![Block::Paragraph(inlines)]));
            self.list_gap = false;
            return Ok(());
        }
        self.flush_runs();
        self.body_left = self.state.indent.left;
        match math_lines(&inlines).filter(|_| math_display) {
            Some(lines) => self.blocks.extend(lines.into_iter().map(Block::Math)),
            None => {
                let row = tabs::has_tab(&inlines);
                self.blocks.push(Block::Paragraph(inlines));
                // markitai: a plain paragraph may turn out to be a heading
                // set by hand, a row of a table set with tab stops, or an
                // item of a list typed by hand.
                self.looks.paragraph(self.blocks.len() - 1, size);
                if row {
                    self.tab_rows.paragraph(self.blocks.len() - 1, self.state.stops);
                }
                self.typed_lists.paragraph(self.blocks.len() - 1, self.state.indent);
            }
        }
        Ok(())
    }

    /// A captured `\listtext` as a literal marker label: trimmed, non-empty.
    fn trimmed_label(listtext: Option<&str>) -> Option<String> {
        listtext.map(str::trim).filter(|t| !t.is_empty()).map(str::to_string)
    }

    #[allow(clippy::type_complexity)]
    fn list_entry(
        &mut self,
        listtext: Option<&str>,
    ) -> Option<(ListKey, usize, u64, Option<String>)> {
        if let Some(ls) = self.state.ls {
            let level = self.state.ilvl;
            if let Some(list) = self.prelude.lists.get(&ls) {
                let def = &list.levels[level.min(LIST_LEVELS - 1)];
                let marker = def.marker?;
                let (number, label) = if marker.ordered() {
                    self.counters.next_labeled(ls, level, &list.levels)
                } else {
                    (0, None)
                };
                return Some((ListKey { instance: ls as u64, marker }, level, number, label));
            }
            // No table definition behind the \ls: the captured \listtext is
            // the only surviving evidence of the real marker; carry it as
            // the item's literal label over a bullet base.
            return Some((
                ListKey { instance: ls as u64, marker: MarkerKind::Bullet },
                level,
                0,
                Self::trimmed_label(listtext),
            ));
        }
        if let Some(marker) = self.state.legacy_list {
            let number = if marker.ordered() {
                // Legacy \pn numbering: the label text carries the number,
                // clamped so a crafted label cannot overflow the counter.
                let parsed = listtext.and_then(|t| {
                    let digits: String = t.chars().take_while(|c| c.is_ascii_digit()).collect();
                    digits.parse::<u64>().ok().map(|n| n.min(u32::MAX as u64))
                });
                match parsed {
                    Some(n) => {
                        self.counters.seed(i32::MAX, self.state.ilvl, n);
                        n
                    }
                    None => self.counters.next(i32::MAX, self.state.ilvl, 1),
                }
            } else {
                0
            };
            return Some((ListKey { instance: u64::MAX, marker }, self.state.ilvl, number, None));
        }
        // A bare \listtext with no list state still marks a list paragraph
        // (some producers omit \ls); its text is the literal marker.
        if listtext.is_some_and(|t| !t.trim().is_empty()) {
            return Some((
                ListKey { instance: u64::MAX - 1, marker: MarkerKind::Bullet },
                self.state.ilvl,
                0,
                Self::trimmed_label(listtext),
            ));
        }
        None
    }

    fn end_cell(&mut self, depth: usize) -> Result<(), ConvertError> {
        self.dest.split_open_links(&mut self.inlines);
        let inlines = std::mem::take(&mut self.inlines);
        // markitai: a cell's paragraph is never a heading set by hand, nor a
        // line of code.
        self.para_size = ParaSize::default();
        self.para_fonts = RunFonts::default();
        let listtext = self.dest.listtext.take();
        let inlines =
            self.cell_list_entry(depth, inlines, listtext.as_deref())?.unwrap_or_default();
        self.table.end_cell(depth, self.state.block, inlines)
    }

    /// markitai: a paragraph inside a table cell that belongs to a list
    /// joins the cell's list, numbered as in the body; any other paragraph
    /// is handed back. A heading or a styled block is never a list item.
    fn cell_list_entry(
        &mut self,
        depth: usize,
        inlines: Vec<Inline>,
        listtext: Option<&str>,
    ) -> Result<Option<Vec<Inline>>, ConvertError> {
        if self.state.block.is_some() || self.state.outline.is_some() || inlines_are_empty(&inlines)
        {
            return Ok(Some(inlines));
        }
        let Some((key, level, number, label)) = self.list_entry(listtext) else {
            return Ok(Some(inlines));
        };
        let blocks = vec![Block::Paragraph(inlines)];
        let entry = ListEntry { level, key, number, label, blocks, indent: None, continues: false };
        self.table.push_cell_list_entry(depth, entry)?;
        Ok(None)
    }

    fn end_row(&mut self, depth: usize) -> Result<(), ConvertError> {
        if self.table.has_pending_cell(depth) || !inlines_are_empty(&self.inlines) {
            self.end_cell(depth)?;
        }
        self.table.end_row(depth);
        Ok(())
    }

    /// Close every open block run before something else is emitted.
    fn flush_runs(&mut self) {
        self.styled.flush(&mut self.blocks);
        self.close_list();
    }

    /// markitai: place the open list, the empty paragraphs after it read.
    fn close_list(&mut self) {
        flush_list(&mut self.blocks, &mut self.list_run);
        self.list_gap = false;
    }

    fn finish(mut self) -> Result<Document, ConvertError> {
        for depth in (1..=self.table.depth()).rev() {
            if self.table.has_partial_row(depth) {
                self.end_row(depth)?;
            }
        }
        // Collapse any dangling nested tables outward, then flush.
        self.table.collapse_nested()?;
        self.flush_top_table()?;
        self.flush_runs();
        // markitai: tables set with tab stops, headings set by hand and lists
        // typed by hand (see `crate::shared::tabs`, `crate::shared::visual`
        // and `crate::shared::typed_lists`); a table laying out a listing is
        // its code, and a numbered listing keeps its code, not its line
        // numbers (see `crate::shared::code`).
        let mut blocks = self.blocks;
        let mut notes = self.dest.notes;
        tabs::finish(self.tab_rows, self.typed_lists, Some(self.looks), &mut blocks, &mut notes);
        for blocks in
            std::iter::once(&mut blocks).chain(notes.iter_mut().map(|note| &mut note.blocks))
        {
            listing_tables(blocks);
            drop_line_gutters(blocks);
        }
        Ok(Document {
            blocks,
            notes,
            assets: self.assets.assets,
            slide_starts: Vec::new(),
            warnings: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_list_table_keeps_listtext_marker() {
        // A \ls with no list-table definition degrades to a bullet, but the
        // captured \listtext (the marker Word displays) must stay visible.
        let doc = parse(
            br"{\rtf1 \pard{\listtext 1.\tab}\ls5 one\par \pard{\listtext 2.\tab}\ls5 two\par}",
        )
        .unwrap();
        let Block::List(list) = &doc.blocks[0] else { panic!("{:?}", doc.blocks) };
        assert_eq!(list.items[0].marker_label.as_deref(), Some("1."));
        assert_eq!(list.items[1].marker_label.as_deref(), Some("2."));
    }

    #[test]
    fn mid_paragraph_page_and_column_breaks_keep_the_word_boundary() {
        // \page and \column carry no paragraph mark: without a break of
        // their own the text on either side would run together.
        for src in [r"{\rtf1 Alfa\page Beta\par}", r"{\rtf1 Alfa\column Beta\par}"] {
            let markdown = crate::to_markdown_bytes(src.as_bytes(), crate::Format::Rtf).unwrap();
            assert_eq!(markdown, "Alfa\\\nBeta\n", "source: {src}");
        }
    }

    #[test]
    fn backslash_before_a_line_break_breaks_the_paragraph() {
        // The paragraph mark Cocoa's RTF writer emits instead of `\par`.
        for src in
            ["{\\rtf1 Alpha\\\nBeta\\\nGamma\\\n}", "{\\rtf1 Alpha\\\r\nBeta\\\rGamma\\\r\n}"]
        {
            let markdown = crate::to_markdown_bytes(src.as_bytes(), crate::Format::Rtf).unwrap();
            assert_eq!(markdown, "Alpha\n\nBeta\n\nGamma\n", "source: {src:?}");
        }
    }

    #[test]
    fn body_text_before_a_table_stays_out_of_the_first_cell() {
        let src = "{\\rtf1 Intro\\\n\\trowd\\cellx2000\\cellx4000 A\\cell B\\cell\\row}";
        let markdown = crate::to_markdown_bytes(src.as_bytes(), crate::Format::Rtf).unwrap();
        assert!(markdown.starts_with("Intro\n\n|"), "{markdown}");
    }

    #[test]
    fn styled_runs_stop_before_tables_and_work_inside_cells() {
        let src = br"{\rtf1\ansi{\stylesheet{\s0 Normal;}{\s1 Source Code;}}\pard\plain\s1 before table\par\trowd\cellx2000\pard\plain\intbl\s1 first\par second\cell\row}";
        let doc = parse(src).unwrap();
        let [Block::CodeBlock { text: before, .. }, Block::Table(table)] = &doc.blocks[..] else {
            panic!("unexpected block order: {:?}", doc.blocks)
        };
        assert_eq!(before, "before table");
        let crate::model::CellSlot::Origin(cell) = &table.grid[0][0] else {
            panic!("expected an origin cell")
        };
        let [Block::CodeBlock { text: inside, .. }] = &cell.blocks[..] else {
            panic!("unexpected cell blocks: {:?}", cell.blocks)
        };
        assert_eq!(inside, "first\nsecond");
    }

    #[test]
    fn pict_hex_payload_becomes_an_asset() {
        let doc = parse(br"{\rtf1 before {\pict\pngblip 89504e470d0a1a0a} after}").unwrap();
        assert_eq!(doc.assets.len(), 1, "assets: {:?}", doc.assets);
        assert_eq!(doc.assets[0].media_type, "image/png");
        assert!(doc.assets[0].bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn pict_property_subgroups_do_not_contaminate_the_payload() {
        let doc = parse(
            br"{\rtf1 {\pict{\*\picprop{\sp{\sn wzDescription}{\sv abcdef}}}\pngblip 89504e470d0a1a0a}}",
        )
        .unwrap();
        assert_eq!(doc.assets.len(), 1, "assets: {:?}", doc.assets);
        assert!(doc.assets[0].bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn nonshppict_fallback_is_not_extracted_twice() {
        let doc = parse(
            br"{\rtf1 {\*\shppict{\pict\pngblip 89504e47}}{\nonshppict{\pict\wmetafile8 0102}}}",
        )
        .unwrap();
        assert_eq!(doc.assets.len(), 1, "only the preferred picture: {:?}", doc.assets);
        assert_eq!(doc.assets[0].media_type, "image/png");
    }

    fn markdown(src: &str) -> String {
        crate::to_markdown_bytes(src.as_bytes(), crate::Format::Rtf).unwrap()
    }

    fn cell_blocks(block: &Block, row: usize, column: usize) -> &[Block] {
        let Block::Table(table) = block else { panic!("expected a table: {block:?}") };
        let crate::model::CellSlot::Origin(cell) = &table.grid[row][column] else {
            panic!("expected an origin cell")
        };
        &cell.blocks
    }

    // markitai: charset 0 is Windows-1252 under any `\ansicpg`.
    #[test]
    fn ansi_fonts_read_their_bytes_as_windows_1252_under_a_cjk_code_page() {
        // As TextEdit saves on a Chinese system: the document code page is
        // 936, the charset-0 font's bytes are 1252, the charset-134 font's
        // bytes are GBK.
        let src = r"{\rtf1\ansi\ansicpg936{\fonttbl\f0\froman\fcharset0 Times-Roman;\f1\fnil\fcharset134 STSongti-SC-Regular;}
\f0 Apple\'92s \'bd caf\'e9 10\'9616\par
\f1 \'d6\'d0\'ce\'c4\'a1\'af\par}";
        assert_eq!(markdown(src), "Apple’s ½ café 10–16\n\n中文’\n");
        // The default charset still follows the code page.
        let src =
            r"{\rtf1\ansi\ansicpg1251{\fonttbl\f0\fnil\fcharset1 Arial;}\f0 \'cf\'f0\'e8\par}";
        assert_eq!(markdown(src), "При\n");
    }

    // markitai: TextEdit's nested-table ending.
    #[test]
    fn a_nestcell_after_a_nestrow_closes_the_cell_holding_the_nested_table() {
        // A table nested two deep, as TextEdit writes a web page's comment
        // thread: each depth-2 row holds a depth-3 table, and the depth-2
        // cell is closed by a `\nestcell` with no `\itap2` before it.
        let src = r"{\rtf1
\itap1\trowd\cellx8640
\itap2\trowd\cellx8640
\pard\intbl\itap2 Title\nestcell \lastrow\nestrow
\itap2\trowd\cellx8640
\itap3\trowd\cellx4320\cellx8640
\pard\intbl\itap3 \nestcell
\pard\intbl\itap3 First comment\nestcell \lastrow\nestrow\nestcell \nestrow
\itap2\trowd\cellx8640
\itap3\trowd\cellx4320\cellx8640
\pard\intbl\itap3 \nestcell
\pard\intbl\itap3 Second comment\nestcell \lastrow\nestrow\nestcell \lastrow\nestrow\cell \lastrow\row
\pard After\par}";
        let doc = parse(src.as_bytes()).unwrap();
        let [outer, Block::Paragraph(after)] = &doc.blocks[..] else {
            panic!("unexpected blocks: {:?}", doc.blocks)
        };
        assert_eq!(crate::model::inlines_to_plain_text(after), "After");
        let [middle] = cell_blocks(outer, 0, 0) else { panic!("{outer:?}") };
        let Block::Table(table) = middle else { panic!("{middle:?}") };
        assert_eq!(table.grid.len(), 3, "title row and one row per comment: {table:?}");
        for (row, comment) in [(1, "First comment"), (2, "Second comment")] {
            let [inner] = cell_blocks(middle, row, 0) else { panic!("{middle:?}") };
            let [Block::Paragraph(text)] = cell_blocks(inner, 0, 1) else { panic!("{inner:?}") };
            assert_eq!(crate::model::inlines_to_plain_text(text), comment);
        }
    }

    // markitai: text after a nested table stays after it.
    #[test]
    fn text_after_a_nested_table_follows_it_in_the_cell() {
        // Word's shape: the nested row properties trail the row in
        // `\nesttableprops`, and the outer cell continues at `\itap1`.
        let src = r"{\rtf1
\trowd\cellx3000\cellx6000
\pard\intbl\itap1 Outer A\cell
\pard\intbl\itap1 Before inner\par
\pard\intbl\itap2 Inner 1\nestcell Inner 2\nestcell
{\*\nesttableprops\trowd\cellx1000\cellx2000\nestrow}{\nonesttables\par}
\pard\intbl\itap1 After inner\cell
\trowd\cellx3000\cellx6000\row}";
        let doc = parse(src.as_bytes()).unwrap();
        let [table] = &doc.blocks[..] else { panic!("unexpected blocks: {:?}", doc.blocks) };
        let [Block::Paragraph(before), Block::Table(_), Block::Paragraph(after)] =
            cell_blocks(table, 0, 1)
        else {
            panic!("{table:?}")
        };
        assert_eq!(crate::model::inlines_to_plain_text(before), "Before inner");
        assert_eq!(crate::model::inlines_to_plain_text(after), "After inner");
    }

    // markitai: a hyperlink result spanning paragraphs.
    #[test]
    fn a_link_spanning_paragraphs_links_each_paragraph() {
        let src = "{\\rtf1 {\\field{\\*\\fldinst{HYPERLINK \"https://example.com/related\"}}{\\fldrslt \\pard \\\n\\pard example.com {\\b Related Project}\\\nA similar project\\\n}}\\pard After\\par}";
        assert_eq!(
            markdown(src),
            "[example.com **Related Project**](https://example.com/related)\n\n\
             [A similar project](https://example.com/related)\n\nAfter\n"
        );
        // A link inside one paragraph is unchanged.
        let src = r#"{\rtf1 See {\field{\*\fldinst{HYPERLINK "https://example.com"}}{\fldrslt this}} now\par}"#;
        assert_eq!(markdown(src), "See [this](https://example.com) now\n");
    }

    // markitai: lists inside table cells.
    #[test]
    fn list_paragraphs_inside_a_cell_form_a_list() {
        let src = r"{\rtf1{\*\listtable{\list\listtemplateid1{\listlevel\levelnfc23\levelstartat1{\leveltext\'01舦 ;}{\levelnumbers;}}\listid1}{\list\listtemplateid2{\listlevel\levelnfc0\levelstartat1{\leveltext\'02\'00.;}{\levelnumbers\'01;}}\listid2}}
{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}{\listoverride\listid2\listoverridecount0\ls2}}
\trowd\cellx4000\cellx8000
\pard\intbl Posted a follow-up.\par
\pard\intbl\ls1\ilvl0 {\listtext 舦 }Preserve linked text\par
\ls1\ilvl0 {\listtext 舦 }Check block content\cell
\pard\intbl\ls2\ilvl0 {\listtext 1.}One\par
\ls2\ilvl0 {\listtext 2.}Two\cell\row}";
        let doc = parse(src.as_bytes()).unwrap();
        let [table] = &doc.blocks[..] else { panic!("unexpected blocks: {:?}", doc.blocks) };
        let [Block::Paragraph(intro), Block::List(bullets)] = cell_blocks(table, 0, 0) else {
            panic!("{table:?}")
        };
        assert_eq!(crate::model::inlines_to_plain_text(intro), "Posted a follow-up.");
        assert!(!bullets.marker.ordered());
        assert_eq!(bullets.items.len(), 2);
        let [Block::List(numbers)] = cell_blocks(table, 0, 1) else { panic!("{table:?}") };
        assert!(numbers.marker.ordered());
        assert_eq!((numbers.start, numbers.items.len()), (1, 2));
    }

    // markitai: `\super` and `\sub`.
    #[test]
    fn raised_and_lowered_text_uses_unicode_forms_where_it_has_them() {
        let src = r"{\rtf1 library{\super 1} and x{\sub 1}, H\sub 2\nosupersub O, 1{\super st}, 10\super -3\plain\par}";
        assert_eq!(markdown(src), "library¹ and x₁, H₂O, 1st, 10⁻³\n");
    }

    // markitai: headings set by hand (see `crate::shared::visual`).
    #[test]
    fn bold_paragraphs_set_above_the_body_size_are_headings_ranked_by_size() {
        let prose = r"\f0\b0\fs24 Body text long enough to outweigh every heading together, as the \
            text of a document does: a few sentences set at one ordinary size.\par ";
        let body = [
            r"\pard\b\fs48 The Title\par ",
            prose,
            r"\pard\b\fs36 A Section\par ",
            prose,
            r"\pard\intbl\b\fs36 Cell\cell\row\pard ",
            r"\b0\fs26 let x = 1\par ",
            r"{\b\fs28 A Subsection}\par ",
            r"\plain\b A plain reset\par ",
            prose,
        ]
        .concat();
        let markdown = markdown(&format!(r"{{\rtf1\ansi {body}}}"));
        assert_eq!(
            markdown.lines().filter(|line| line.starts_with('#')).collect::<Vec<_>>(),
            ["# The Title", "## A Section", "### A Subsection"],
            "{markdown}"
        );
        assert!(markdown.contains("Cell"), "{markdown}");
        let outlined = format!(r"{{\rtf1\ansi {body}\pard\outlinelevel0 Real\par}}");
        let markdown = crate::to_markdown_bytes(outlined.as_bytes(), crate::Format::Rtf).unwrap();
        assert_eq!(
            markdown.lines().filter(|line| line.starts_with('#')).collect::<Vec<_>>(),
            ["# Real"],
            "{markdown}"
        );
    }

    /// markitai: a document's blocks, described (see
    /// `crate::shared::code::describe`).
    fn described(rtf: &str) -> Vec<String> {
        crate::shared::code::describe(&parse(rtf.as_bytes()).unwrap().blocks)
    }

    /// A font table as TextEdit writes it (PostScript names, one font after
    /// another) and as Word does (a group per font, a fixed-pitch face).
    const FONTS: &str = r"{\fonttbl\f0\froman\fcharset0 Times-Roman;\f1\fnil\fcharset0 Menlo-Regular;
        {\f2\fmodern\fcharset0\fprq1{\*\panose 02070309020205020404}Letter Gothic;}
        {\f3\fmodern\fcharset128\fprq1 MS Gothic;}}";

    #[test]
    fn text_in_a_monospaced_font_is_code() {
        let rtf = format!(
            r"{{\rtf1\ansi\deff0{FONTS}
\pard\outlinelevel0\f1 Setup\par
\pard\f0 Run \f1 make\f0  or \f2 cargo\f0 ; \f3 kanji\f0  is body text.\par
\pard\f1 fn main() \{{\par
\par
    go();\par
\}}\par
\pard\ls1\ilvl0{{\listtext 1.\tab}}listed\par
\pard\intbl celled\cell\f0 prose cell\cell\row
\pard\f1\pard 1\par
\pard\f0 And then the prose of the document goes on for a while in its own face.\par}}"
        );
        assert_eq!(
            described(&rtf),
            [
                "h1:Setup",
                "p:Run `make` or `cargo`; kanji is body text.",
                "code:fn main() {\n\n    go();\n}",
                "list:p:`listed`",
                "table:p:`celled`|p:prose cell",
                "code:1",
                "p:And then the prose of the document goes on for a while in its own face.",
            ]
        );
    }

    #[test]
    fn the_default_font_and_a_pictures_paragraph() {
        // Text no `\f` names is in the `\deff` font; a paragraph holding a
        // picture is never a line of code, which would keep only its text.
        let rtf = format!(
            r"{{\rtf1\ansi\deff1{FONTS}
\pard ls -la\par
\pard fig.{{\pict\pngblip 89504e47}}\par
\pard\f0 And then the prose of the document goes on for a while in its own face.\par}}"
        );
        assert_eq!(
            described(&rtf),
            [
                "code:ls -la",
                "p:`fig.`",
                "p:And then the prose of the document goes on for a while in its own face.",
            ]
        );
    }

    #[test]
    fn a_document_set_in_a_monospaced_font_is_not_code() {
        // Courier sets most of the text: it is the typewriter's face.
        let rtf = r"{\rtf1\ansi{\fonttbl{\f0\fmodern\fcharset0\fprq1 Courier New;}{\f1\froman Times;}}
\pard\f0 INT. KITCHEN - NIGHT\par
\pard She opens the door.\par
\pard\f1 Page 1\par}";
        assert_eq!(described(rtf), ["p:INT. KITCHEN - NIGHT", "p:She opens the door.", "p:Page 1"]);
    }

    #[test]
    fn a_listing_table_and_numbered_lines_are_code() {
        let rtf = format!(
            r"{{\rtf1\ansi{FONTS}
\pard\f0 A listing as a highlighter lays it out, and one numbered line by line.\par
\trowd\cellx1000\cellx8000
\pard\intbl\f1 1\par 2\cell a = 1\par b = 2\cell\row
\pard\f1 1\par x\par 2\par y\par
\pard\f0 Then the prose of the document goes on in the body face for long enough.\par}}"
        );
        assert_eq!(
            described(&rtf),
            [
                "p:A listing as a highlighter lays it out, and one numbered line by line.",
                "code:a = 1\nb = 2",
                "code:x\ny",
                "p:Then the prose of the document goes on in the body face for long enough.",
            ]
        );
    }

    // markitai: TextEdit's picture attachments.
    #[test]
    fn picture_attachments_leave_no_file_name() {
        let src = r"{\rtf1\ansi\ansicpg1252\cocoartf2822{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
\pard\f0 Before {{\NeXTGraphic Pasted Graphic.png \width2000 \height1000 \appleattachmentpadding0 \appleembedmode1 \appleaqc
}\'ac}after.\par
\pard Price \'ac 5 stays.\par}";
        assert_eq!(markdown(src), "Before after.\n\nPrice ¬ 5 stays.\n");
    }

    // markitai: a blank line inside a listing's cell.
    #[test]
    fn a_blank_line_of_a_listing_in_a_cell_is_kept() {
        let rtf = format!(
            r"{{\rtf1\ansi{FONTS}
\pard\f0 A listing with a blank line, as TextEdit saves a highlighter's table.\par
\trowd\cellx1000\cellx8000
\pard\intbl\f1 1\par 2\par 3\cell #include <stdio.h>\par
\par
int main();\par
\cell\row
\pard\f0 A table of values follows.\par
\trowd\cellx4000\cellx8000
\pard\intbl\f0 Name\cell\f1 value\par
\par
\f0 note\cell\row
\pard\f0 Then the prose of the document goes on in the body face for long enough.\par}}"
        );
        assert_eq!(
            described(&rtf),
            [
                "p:A listing with a blank line, as TextEdit saves a highlighter's table.",
                "code:#include <stdio.h>\n\nint main();",
                "p:A table of values follows.",
                "table:p:Name|p:`value`;p:;p:note",
                "p:Then the prose of the document goes on in the body face for long enough.",
            ]
        );
        let text = markdown(&rtf);
        assert!(text.contains("| Name | `value`<br>note |"), "{text}");
    }

    #[test]
    fn columns_set_with_tab_stops_are_a_table() {
        let row = |stops: &str, text: &str| format!(r"\pard{stops} {text}\par ");
        let cols = r"\tx2880\tqr\tx5760";
        let dots = r"\tqr\tldot\tx8640";
        let rtf = [
            r"{\rtf1\ansi ",
            &row(cols, r"Item\tab Qty\tab Price"),
            &row(cols, r"Apple\tab 3\tab 1.20"),
            &row(cols, r"Pear\tab 12\tab 0.50"),
            &row(dots, r"Part 1\tab Intro\tab Page one"),
            &row(dots, r"Part 2\tab Methods\tab Page two"),
            &row(dots, r"Part 3\tab Results\tab Page three"),
            // `\pard` clears the stops: these rows are at the default ones.
            &row("", r"North\tab Ann\tab 1a"),
            &row("", r"South\tab Ben\tab 2b"),
            &row("", r"West\tab Cleo\tab 3c"),
            // Two columns at the default stops stay text.
            &row("", r"Oak\tab Ash"),
            &row("", r"Elm\tab Yew"),
            &row("", r"Fir\tab Box"),
            r"\pard\outlinelevel0 A\tab B\tab C\par ",
            r"\pard{\footnote Note\tab text}\par}",
        ]
        .concat();
        let doc = parse(rtf.as_bytes()).unwrap();
        assert_eq!(
            crate::shared::code::describe(&doc.blocks),
            [
                "table:p:Item|p:Qty|p:Price/p:Apple|p:3|p:1.20/p:Pear|p:12|p:0.50",
                "p:Part 1 Intro Page one",
                "p:Part 2 Methods Page two",
                "p:Part 3 Results Page three",
                "table:p:North|p:Ann|p:1a/p:South|p:Ben|p:2b/p:West|p:Cleo|p:3c",
                "p:Oak Ash",
                "p:Elm Yew",
                "p:Fir Box",
                "h1:A B C",
                "p:",
            ]
        );
        assert_eq!(crate::shared::code::describe(&doc.notes[0].blocks), ["p:Note text"]);
    }

    #[test]
    fn indents_and_symbol_fonts_in_a_list_typed_by_hand() {
        // markitai: `\li` and `\fi` place the bullet until `\pard`; a
        // Wingdings square's byte is the square, not `§`; a bullet line in
        // a monospaced font is an item.
        let read = |body: &str| {
            described(&format!(
                r"{{\rtf1\ansi{{\fonttbl{{\f0\froman\fcharset0 Times;}}{{\f1\fnil\fcharset2 Wingdings;}}
                {{\f2\fnil\fcharset0 Menlo-Regular;}}}}{body}
                \pard\f0 And then the prose of the document goes on for a while.\par}}"
            ))
        };
        let both = |a: &str, b: &str| {
            read(&format!(r"\pard{a} \bullet\tab a\par \pard{b} \bullet\tab b\par"))
        };
        assert_eq!(both("", r"\li360\fi-360")[0], "list:p:a|p:b");
        assert_eq!(both("", r"\li720")[0], "list:p:a;list:p:b");
        // `\pard` resets the indent: `c` is back beside `a`, not under it.
        let doc = parse(
            br"{\rtf1 \pard \bullet\tab a\par \pard\li720 \bullet\tab b\par \pard \bullet\tab c\par}",
        )
        .unwrap();
        let [Block::List(list)] = &doc.blocks[..] else { panic!("{:?}", doc.blocks) };
        assert_eq!(list.items.len(), 2, "{list:?}");
        assert_eq!(read(r"\pard {\f1\'a7}\tab a\par \pard {\f1\'a7}\tab b\par")[0], "list:p:a|p:b");
        assert_eq!(read(r"\pard\f2 \bullet\tab npm test\par")[0], "list:p:`npm test`");
        // Text in a symbol font that the table has no glyph for decodes as
        // before.
        assert_eq!(read(r"\pard {\f1 J} {\f1 A} smiles\par")[0], "p:☺ A smiles");
    }

    /// markitai: Word's list table: a numbered level and a bulleted one
    /// under it.
    const LISTS: &str = r"{\*\listtable{\list\listtemplateid1
        {\listlevel\levelnfc0\levelstartat1{\leveltext\'02\'00.;}{\levelnumbers\'01;}\fi-360\li720}
        {\listlevel\levelnfc23\levelstartat1{\leveltext\'01舦 ?;}{\levelnumbers;}\fi-360\li1440}
        \listid1}}{\*\listoverridetable{\listoverride\listid1\listoverridecount0\ls1}}";

    #[test]
    fn a_paragraph_set_in_to_an_items_text_continues_the_item() {
        let one = r"\pard\fi-360\li720\ls1\ilvl0 {\listtext 1.\tab}";
        let two = r"\pard\fi-360\li720\ls1\ilvl0 {\listtext 2.\tab}";
        let nested = r"\pard\fi-360\li1440\ls1\ilvl1 {\listtext \bullet\tab}";
        let body = [
            r"\pard Before the list.\par ",
            one,
            r"One.\par \pard\li720 More of one.\par \pard\par \pard\li720 After a blank line.\par ",
            two,
            r"Two.\par ",
            nested,
            r"Two a.\par \pard\li1440 More of two a.\par \pard\li720 More of two.\par ",
            r"\pard Body after.\par \pard\li720 An indented note.\par",
        ]
        .concat();
        let doc = described(&format!(r"{{\rtf1\ansi {LISTS}{body}}}"));
        assert_eq!(
            doc,
            [
                "p:Before the list.",
                "list:p:One.;p:More of one.;p:After a blank line.|p:Two.;list:p:Two a.;p:More of two a.;p:More of two.",
                "p:Body after.",
                "p:An indented note.",
            ]
        );
        // An empty paragraph between items closes the list as before, and
        // a body set in as far as the list's text continues nothing.
        let body = [r"\pard\li720 Body set in.\par ", one, r"One.\par \pard\par ", two].concat()
            + r"Two.\par \pard\li720 Body again.\par";
        let doc = parse(format!(r"{{\rtf1\ansi {LISTS}{body}}}").as_bytes()).unwrap();
        let shown = crate::shared::code::describe(&doc.blocks);
        assert_eq!(shown, ["p:Body set in.", "list:p:One.", "list:p:Two.", "p:Body again."]);
        let starts: Vec<u64> = doc
            .blocks
            .iter()
            .filter_map(|block| match block {
                Block::List(list) => Some(list.start),
                _ => None,
            })
            .collect();
        assert_eq!(starts, [1, 2]);
    }

    // markitai: hidden and deleted text.
    #[test]
    fn hidden_and_deleted_text_is_left_out() {
        let src = r"{\rtf1\ansi
\pard Shown {\v hidden in a group} shown.\par
\pard Before \v toggled\v0  after.\par
\pard Kept {\deleted\revauthdel1 removed }and {\revised inserted} text.\par
\pard Reset \v gone\plain  back.\par
\pard {\v Hidden paragraph mark.\par}Joined to the next.\par
\pard Visible part{\v  hidden\par} joined.\par
\pard {\deleted Deleted paragraph.\par}
\pard {\v\tab\line hidden{\pict\pngblip 89504e47}}Tail.\par}";
        assert_eq!(
            markdown(src),
            "Shown  shown.\n\nBefore  after.\n\nKept and inserted text.\n\nReset  back.\n\n\
             Joined to the next.\n\nVisible part joined.\n\nTail.\n"
        );
    }

    // markitai: index and contents entries.
    #[test]
    fn index_entries_are_not_text() {
        let src = r"{\rtf1\ansi
\pard Term{\xe {\v Hidden entry}} defined.\par
\pard Old{\xe Plain entry{\txe See also}} style.\par
\pard {\tc Contents entry}Chapter.\par}";
        assert_eq!(markdown(src), "Term defined.\n\nOld style.\n\nChapter.\n");
    }

    // markitai: `\upr` keeps the Unicode reading.
    #[test]
    fn upr_reads_its_unicode_text() {
        let u = |n: i32| format!("{}u{n}", '\\');
        let src = format!(
            r"{{\rtf1\ansi\pard A {{\upr{{Caf\'e9 ?}}{{\*\ud{{Caf{}? {}?}}}}}} B.\par
\pard {{\*\bkmkstart {{\upr{{n?}}{{\*\ud{{n{}?}}}}}}}}C.\par
{{\info{{\title {{\upr{{T}}{{\*\ud{{Title}}}}}}}}}}{{\*\mystery {{\upr{{M}}{{\*\ud{{Mystery}}}}}}}}\pard D.\par}}",
            u(233),
            u(20320),
            u(241)
        );
        let doc = parse(src.as_bytes()).unwrap();
        let text = crate::to_markdown_bytes(src.as_bytes(), crate::Format::Rtf).unwrap();
        assert!(text.starts_with("A Café 你 B.\n\n"), "{text}");
        assert!(!text.contains("Title"), "a suppressed destination stays suppressed: {text}");
        assert!(!text.contains("Mystery"), "an unknown destination stays suppressed: {text}");
        let Block::Paragraph(second) = &doc.blocks[1] else { panic!("{:?}", doc.blocks) };
        assert!(
            second.iter().any(|i| matches!(i, Inline::Anchor(name) if name == "nñ")),
            "{second:?}"
        );
    }

    // markitai: legacy form fields.
    #[test]
    fn form_fields_show_check_boxes_and_chosen_entries() {
        fn field(data: &str, instr: &str) -> String {
            format!(
                r"{{\field{{\*\fldinst {{{instr} {{\*\formfield{{{data}{{\*\ffname F}}}}}}}}}}{{\fldrslt }}}}"
            )
        }
        let body = [
            r"\pard Typed: {\field{\*\fldinst {FORMTEXT {\*\formfield{\fftype0{\*\ffname T}{\*\ffdeftext Default}}}}}{\fldrslt Jane}}.\par "
                .to_string(),
            format!(r"\pard Checked: {}.\par ", field(r"\fftype1\ffres1\ffdefres0", "FORMCHECKBOX")),
            format!(r"\pard By default: {}.\par ", field(r"\fftype1\ffres25\ffdefres1", "FORMCHECKBOX")),
            format!(r"\pard Unset: {}.\par ", field(r"\fftype1\ffres25", "FORMCHECKBOX")),
            format!(
                r"\pard Chosen: {}.\par ",
                field(r"\fftype2\ffres1{\*\ffl Red}{\*\ffl Green}{\*\ffl Blue}", "FORMDROPDOWN")
            ),
            format!(
                r"\pard Default: {}.\par ",
                field(r"\fftype2\ffres25\ffdefres2{\*\ffl S}{\*\ffl M}{\*\ffl L}", "FORMDROPDOWN")
            ),
            format!(r"\pard Not a form: {}.\par ", field(r"\fftype1\ffres1", "PAGE")),
            format!(r"\pard Hidden: {{\v {}}}.\par", field(r"\fftype1\ffres1", "FORMCHECKBOX")),
        ]
        .concat();
        assert_eq!(
            markdown(&format!(r"{{\rtf1\ansi {body}}}")),
            "Typed: Jane.\n\nChecked: ☒.\n\nBy default: ☒.\n\nUnset: ☐.\n\nChosen: Green.\n\n\
             Default: L.\n\nNot a form: .\n\nHidden: .\n"
        );
    }

    // markitai: `\uN` in a symbol font.
    #[test]
    fn symbol_font_unicode_reads_as_the_fonts_character() {
        let src = r"{\rtf1\ansi{\fonttbl{\f0\froman Times;}{\f1\ftech\fcharset2 Symbol;}{\f2\fnil\fcharset2 Wingdings;}}
\pard\f0 A {\f1 \u-3913\'b7} B {\f2 \u-3844\'fc} C {\f0 \u-3913?} D.\par}";
        assert_eq!(markdown(src), "A • B ✓ C \u{F0B7} D.\n");
    }

    // markitai: what a header or footer holds stays out of the body.
    #[test]
    fn header_and_footer_shapes_pictures_and_notes_stay_out() {
        let src = r"{\rtf1\ansi
{\header \pard Header {\shp{\*\shpinst{\sp{\sn shapeType}{\sv 202}}{\shptxt \pard Header box\par}}}{\*\shppict{\pict\pngblip 89504e470d0a1a0a}}{\object\objemb{\*\objdata 0102}{\result Header object}}{\*\bkmkstart hdr}\par}
{\footer \pard {\footnote Footer note}{\shp{\*\shpinst{\sp{\sn gtextUNICODE}{\sv DRAFT}}}}\par}
\pard Body {\shp{\*\shpinst{\shptxt \pard Body box\par}}}text.\par}";
        let doc = parse(src.as_bytes()).unwrap();
        assert!(doc.assets.is_empty() && doc.notes.is_empty(), "{doc:?}");
        // Not even the header's bookmark.
        let anchors = crate::shared::code::describe(&doc.blocks);
        assert!(
            doc.blocks.iter().all(|b| match b {
                Block::Paragraph(inlines) =>
                    !inlines.iter().any(|i| matches!(i, Inline::Anchor(_))),
                _ => true,
            }),
            "{anchors:?} {:?}",
            doc.blocks
        );
        assert_eq!(
            crate::render::markdown::document_to_markdown(&doc),
            // The text box's paragraph mark ends the body's paragraph, as
            // upstream read it.
            "Body Body box\n\ntext.\n"
        );
    }

    // markitai: a shape's picture, alt text and WordArt.
    #[test]
    fn shape_pictures_alt_text_and_word_art_are_read() {
        let png = "89504e470d0a1a0a0000000d49484452";
        let wmf = "0100090000";
        let src = r"{\rtf1\ansi
\pard Inline {\*\shppict{\pict{\*\picprop{\sp{\sn wzDescription}{\sv A red square}}{\sp{\sn wzName}{\sv Picture 1}}}\pngblip PNG}}{\nonshppict{\pict\wmetafile8 WMF}} end.\par
\pard Floating {\shp{\*\shpinst{\sp{\sn shapeType}{\sv 75}}{\sp{\sn pib}{\sv {\pict\pngblip PNG}}}{\sp{\sn wzDescription}{\sv A floating square}}}{\shprslt\par\pard {\pict\wmetafile8 WMF}\par}}end.\par
\pard Copy only {\shp{\*\shpinst{\sp{\sn shapeType}{\sv 75}}{\sp{\sn wzDescription}{\sv Metafile copy}}}{\shprslt {\pict\wmetafile8 WMF}}}end.\par
\pard {\shp{\*\shpinst{\sp{\sn shapeType}{\sv 136}}{\sp{\sn gtextUNICODE}{\sv Grand Opening}}{\sp{\sn gtextFont}{\sv Arial Black}}}{\shprslt{\pict\wmetafile8 0200090000}}}\par
\pard Hidden {\v{\shp{\*\shpinst{\sp{\sn gtextUNICODE}{\sv Secret}}}}}art.\par
\pard Boxed {\shp{\*\shpinst{\sp{\sn wzDescription}{\sv Box alt}}{\shptxt \pard Inside {\*\shppict{\pict\pngblip PNG}} box\par}}}end.\par}"
            .replace("PNG", png)
            .replace("WMF", wmf);
        let doc = parse(src.as_bytes()).unwrap();
        let types: Vec<&str> = doc.assets.iter().map(|a| a.media_type.as_str()).collect();
        // The inline picture, the floating shape's own picture (not its
        // metafile copy), the copy of the shape with no picture of its own,
        // not the WordArt's copy, and the picture in a text box.
        assert_eq!(types, ["image/png", "image/png", "image/wmf", "image/png"]);
        // The renderer writes a picture as its alt text.
        let text = crate::render::markdown::document_to_markdown(&doc);
        let lines: Vec<&str> = text.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(
            lines,
            [
                "Inline A red square end.",
                "Floating A floating squareend.",
                "Copy only Metafile copyend.",
                "Grand Opening",
                "Hidden art.",
                // The text box's alt text is not its picture's.
                "Boxed Inside  box",
                "end.",
            ]
        );
    }
}
