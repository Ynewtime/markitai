//! Block and inline walking for WordprocessingML parts.

use crate::error::ConvertError;
use crate::formats::docx::code::{is_monospace, without_line_gutter};
use crate::formats::docx::numbering::{Counters, Numbering};
use crate::formats::docx::scripts::Script;
use crate::formats::docx::styles::{Styles, on_off, rpr_delta, run_font, run_size};
use crate::model::{
    Block, Cell, GridBuilder, ImageSource, Inline, LinkTarget, Style, TableKind, inlines_are_empty,
};
use crate::package::Package;
use crate::package::relationships::{RelTarget, Relationships, TargetMode, rel_target_bytes};
use crate::package::xml::{Element, ns};
use crate::shared::blockstyle::{BlockStyle, StyledRun};
use crate::shared::code::{RunFonts, without_code};
use crate::shared::delta::rebase_emphasis;
use crate::shared::fields::{FieldFrame, field_result};
use crate::shared::list::{ListEntry, ListKey, continuation_level, flush_list, unmarked_level};
use crate::shared::math::{omath_para_to_tex, omath_to_tex};
use crate::shared::tabs::{self, Stops, TabRows};
use crate::shared::text::clean_text;
use crate::shared::typed_lists::{Indent, TypedLists, opens_with_bullet};
use crate::shared::visual::{Looks, ParaSize, Size};
use std::cell::RefCell;
use std::collections::HashMap;

/// Namespaces whose markup this frontend understands; `mc:Choice` branches
/// requiring anything else fall back to `mc:Fallback`.
const SUPPORTED_NS: &[&str] = &[
    ns::W,
    ns::A,
    ns::PIC,
    ns::WP,
    ns::MC,
    ns::CHART,
    ns::DGM,
    ns::VML,
    ns::O_VML,
    ns::WPS,
    ns::WPG,
];

use crate::shared::assets::AssetSink;

pub(super) struct Ctx<'a, 'b> {
    pub pkg: &'b RefCell<Package<'a>>,
    pub rels: Relationships,
    pub base_part: String,
    pub styles: &'b Styles<'b>,
    pub numbering: &'b Numbering,
    pub counters: &'b RefCell<Counters>,
    pub assets: &'b RefCell<AssetSink>,
    /// markitai: whether monospaced runs are code (see `super::code`), and
    /// how many table cells deep the content being read sits.
    pub code_fonts: bool,
    pub cell_depth: std::cell::Cell<u32>,
    /// markitai: what the body's text looks like (see
    /// [`crate::shared::visual`]), and how many block containers deep (the
    /// body, a text box, a cell) the content being read sits. Notes are not
    /// looked at.
    pub looks: Option<&'b RefCell<Looks>>,
    pub block_depth: std::cell::Cell<u32>,
    /// markitai: the body's paragraphs that could be rows of a table set
    /// with tab stops (see [`crate::shared::tabs`]); while it is set, a tab
    /// is read as [`tabs::TAB`]. Notes are not looked at.
    pub tabs: Option<&'b RefCell<TabRows>>,
    /// markitai: the body's plain paragraphs and their indents, for the
    /// lists typed by hand (see [`crate::shared::typed_lists`]).
    pub lists: Option<&'b RefCell<TypedLists>>,
    /// markitai: how many `w:altChunk`s deep this package sits (0 for the
    /// document itself), and what its embedded parts add to the document
    /// (see [`super::altchunk`]).
    pub chunk_depth: u32,
    pub embedded: &'b RefCell<super::altchunk::Embedded>,
}

/// markitai: content read one level deeper (inside a table cell, or a block
/// container: the body, a text box, a cell), for as long as it lives.
struct InCell<'c>(&'c std::cell::Cell<u32>);

impl<'c> InCell<'c> {
    fn enter(depth: &'c std::cell::Cell<u32>) -> Self {
        depth.set(depth.get() + 1);
        InCell(depth)
    }
}

impl Drop for InCell<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

impl<'a, 'b> Ctx<'a, 'b> {
    /// The same document-wide dependencies scoped to another package part
    /// (notes parts): only the relationships and base path change.
    pub fn for_part(&self, rels: Relationships, base_part: String) -> Ctx<'a, 'b> {
        Ctx {
            pkg: self.pkg,
            rels,
            base_part,
            styles: self.styles,
            numbering: self.numbering,
            counters: self.counters,
            assets: self.assets,
            code_fonts: self.code_fonts,
            cell_depth: Default::default(),
            looks: None,
            block_depth: Default::default(),
            tabs: None,
            lists: None,
            chunk_depth: self.chunk_depth,
            embedded: self.embedded,
        }
    }

    /// Load an internal relationship target's bytes, resolved against this
    /// part. Failures degrade (log + `None`) per the unified policy;
    /// resource-limit errors always propagate.
    pub(super) fn rel_part(&self, rel_id: &str) -> Result<Option<RelTarget>, ConvertError> {
        rel_target_bytes(self.pkg, &self.rels, &self.base_part, rel_id)
    }

    pub(super) fn add_asset(
        &self,
        media_type: String,
        part: String,
        bytes: &[u8],
    ) -> Result<crate::model::AssetId, ConvertError> {
        self.assets.borrow_mut().add(media_type, part, bytes)
    }

    /// Pick the branch of an `mc:AlternateContent` to process.
    fn alternate_branch<'e>(&self, alt: &'e Element) -> Option<&'e Element> {
        crate::shared::mc::alternate_branch(alt, SUPPORTED_NS)
    }
}

pub(super) enum ParaKind {
    Heading {
        level: u8,
        /// The visible number of a numbered heading (with its trailing
        /// separator), prepended to the content: headings have no native
        /// numbering in the output.
        label: Option<String>,
        /// The heading style's own emphasis, subtracted from its runs.
        base: Style,
    },
    ListItem {
        ilvl: usize,
        key: ListKey,
        number: u64,
        label: Option<String>,
    },
    /// A paragraph whose style names a block container.
    Styled(BlockStyle),
    Plain,
}

/// markitai: where a paragraph sits, for the list before it and the lists
/// typed by hand.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct Place {
    /// Where its lines start, in twips, attribute by attribute from its own
    /// `w:ind`, else its numbering level's, else its style's; `None` in a
    /// table of contents or an index.
    pub indent: Option<Indent>,
    /// The level it is numbered at, when that level's marker shows nothing.
    pub unmarked: Option<usize>,
}

/// Block runs a following paragraph may extend: a list being built, and a
/// styled container. Only one is ever open, so starting either closes the
/// other.
#[derive(Default)]
pub(super) struct Runs {
    list: Vec<ListEntry>,
    styled: StyledRun,
    /// markitai: empty paragraphs came after the list's last paragraph. They
    /// close the list, as before, unless what follows continues an item.
    gap: bool,
    /// markitai: where the last paragraph of body text before the list
    /// starts its lines, in twips (see [`continuation_level`]).
    body_left: i32,
}

impl Runs {
    fn flush(&mut self, blocks: &mut Vec<Block>) {
        self.styled.flush(blocks);
        flush_list(blocks, &mut self.list);
        self.gap = false;
    }

    /// markitai: the level of the open list's item a plain paragraph with
    /// content continues: one numbered at a level that shows no marker, or
    /// one set in as far as an item's text (see [`continuation_level`]).
    fn continues(&self, pieces: &[Piece], place: Place) -> Option<usize> {
        let empty =
            pieces.iter().all(|piece| matches!(piece, Piece::Inlines(i) if inlines_are_empty(i)));
        if self.list.is_empty() || empty {
            return None;
        }
        match place.unmarked {
            Some(level) => unmarked_level(&self.list, level),
            None => continuation_level(&self.list, place.indent?.left, self.body_left),
        }
    }
}

/// A paragraph's content in source order: inline runs interleaved with
/// block attachments (text boxes, charts) at their anchor positions.
pub(super) enum Piece {
    Inlines(Vec<Inline>),
    Blocks(Vec<Block>),
}

pub(super) fn parse_blocks(parent: &Element, ctx: &Ctx) -> Result<Vec<Block>, ConvertError> {
    // markitai: see `Ctx::block_depth`.
    let _nested = InCell::enter(&ctx.block_depth);
    let mut blocks: Vec<Block> = Vec::new();
    let mut runs = Runs::default();
    collect_blocks(parent, ctx, &mut blocks, &mut runs)?;
    runs.flush(&mut blocks);
    // markitai: a numbered listing keeps its code, not its line numbers.
    for block in &mut blocks {
        if let Block::CodeBlock { text, .. } = block
            && let Some(code) = without_line_gutter(text)
        {
            *text = code;
        }
    }
    Ok(blocks)
}

fn collect_blocks(
    parent: &Element,
    ctx: &Ctx,
    blocks: &mut Vec<Block>,
    runs: &mut Runs,
) -> Result<(), ConvertError> {
    for child in parent.child_elems() {
        if child.is(ns::MC, "AlternateContent") {
            if let Some(branch) = ctx.alternate_branch(child) {
                collect_blocks(branch, ctx, blocks, runs)?;
            }
            continue;
        }
        if child.is(ns::M, "oMathPara") || child.is(ns::M, "oMath") {
            runs.flush(blocks);
            blocks.extend(omath_para_to_tex(child).into_iter().map(Block::Math));
            continue;
        }
        if child.ns.as_deref().is_none_or(|n| n != ns::W) {
            continue;
        }
        match child.local.as_str() {
            "p" => {
                let (kind, pieces, size, place) = parse_paragraph(child, ctx)?;
                // markitai: a plain paragraph of the body itself may turn
                // out to be a heading set by hand.
                let plain = matches!(kind, ParaKind::Plain)
                    && matches!(&pieces[..], [Piece::Inlines(_)])
                    && ctx.block_depth.get() == 1
                    && ctx.cell_depth.get() == 0;
                // markitai: and it may be a row of a table set with tab stops.
                let row = plain
                    && matches!(&pieces[..], [Piece::Inlines(inlines)] if tabs::has_tab(inlines));
                // markitai: unless it continues an item of the list before it.
                let plain = !emit_paragraph(kind, pieces, place, blocks, runs) && plain;
                let row = plain && row;
                if plain && let Some(looks) = ctx.looks {
                    looks.borrow_mut().paragraph(blocks.len() - 1, size);
                }
                if row
                    && let Some(tab_rows) = ctx.tabs
                    && let Some(stops) = paragraph_stops(child, ctx.styles)?
                {
                    tab_rows.borrow_mut().paragraph(blocks.len() - 1, stops);
                }
                // markitai: and an item of a list typed by hand.
                if plain
                    && let Some(lists) = ctx.lists
                    && let Some(indent) = place.indent
                {
                    lists.borrow_mut().paragraph(blocks.len() - 1, indent);
                }
            }
            "tbl" => {
                runs.flush(blocks);
                blocks.extend(parse_table(child, ctx)?);
            }
            "sdt" => {
                if let Some(content) = child.find(ns::W, "sdtContent") {
                    collect_blocks(content, ctx, blocks, runs)?;
                }
            }
            "customXml" => collect_blocks(child, ctx, blocks, runs)?,
            // markitai: an embedded part, read where it stands.
            "altChunk" => {
                runs.flush(blocks);
                blocks.extend(super::altchunk::blocks(child, ctx)?);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Place a paragraph. markitai: true when a plain one went into the open
/// list (an item's continuation, or an empty paragraph after it) instead of
/// being the last of `blocks`.
fn emit_paragraph(
    kind: ParaKind,
    pieces: Vec<Piece>,
    place: Place,
    blocks: &mut Vec<Block>,
    runs: &mut Runs,
) -> bool {
    match kind {
        ParaKind::ListItem { ilvl, key, number, label } => {
            runs.styled.flush(blocks);
            // markitai: an empty paragraph between items closes the list as
            // it did before.
            if std::mem::take(&mut runs.gap) {
                flush_list(blocks, &mut runs.list);
            }
            let item = pieces_into_blocks(pieces);
            runs.list.push(ListEntry {
                level: ilvl,
                key,
                number,
                label,
                blocks: item,
                indent: place.indent.map(|indent| indent.left),
                continues: false,
            });
        }
        ParaKind::Styled(style) => {
            flush_list(blocks, &mut runs.list);
            runs.gap = false;
            // markitai: an empty paragraph of a code block is a blank line of
            // it (upstream dropped it, joining the lines around it).
            if !pieces.iter().any(|piece| matches!(piece, Piece::Inlines(_))) {
                runs.styled.push(style, Vec::new(), blocks);
            }
            for piece in pieces {
                match piece {
                    Piece::Inlines(inlines) => runs.styled.push(style, inlines, blocks),
                    Piece::Blocks(attachments) => {
                        runs.styled.flush(blocks);
                        blocks.extend(attachments);
                    }
                }
            }
        }
        ParaKind::Heading { level, label, base } => {
            runs.flush(blocks);
            let mut label = label;
            let mut emitted_heading = false;
            for piece in pieces {
                match piece {
                    Piece::Inlines(mut content) if !inlines_are_empty(&content) => {
                        rebase_emphasis(&mut content, base);
                        if !emitted_heading {
                            if let Some(label) = label.take() {
                                content
                                    .insert(0, Inline::Text { text: label, style: Style::PLAIN });
                            }
                            blocks.push(Block::Heading { level, anchor: None, content });
                            emitted_heading = true;
                        } else {
                            // Attachments cannot be nested in a heading, so
                            // subsequent text becomes a paragraph.
                            blocks.push(Block::Paragraph(content));
                        }
                    }
                    Piece::Inlines(_) => {}
                    Piece::Blocks(attachments) => blocks.extend(attachments),
                }
            }
        }
        ParaKind::Plain => {
            // markitai: a paragraph that continues an item of the open list
            // goes into it; an empty one after the list waits to see whether
            // one follows (it shows nothing either way).
            if let Some(level) = runs.continues(&pieces, place) {
                runs.list.push(ListEntry::continuation(level, pieces_into_blocks(pieces)));
                runs.gap = false;
                return true;
            }
            if pieces.is_empty() && !runs.list.is_empty() {
                runs.gap = true;
                return true;
            }
            runs.flush(blocks);
            if !pieces.is_empty() {
                runs.body_left = place.indent.map_or(0, |indent| indent.left);
            }
            blocks.extend(pieces_into_blocks(pieces));
        }
    }
    false
}

fn parse_paragraph(
    p: &Element,
    ctx: &Ctx,
) -> Result<(ParaKind, Vec<Piece>, ParaSize, Place), ConvertError> {
    let ppr = p.find(ns::W, "pPr");
    let pstyle_id = ppr.and_then(|pr| pr.find(ns::W, "pStyle")).and_then(|e| e.attr(ns::W, "val"));

    // Direct paragraph properties overlay the style chain: an explicit
    // outlineLvl of 9 ("no outline level") turns a style heading off.
    let direct_outline: Option<Option<u8>> = ppr
        .and_then(|pr| pr.find(ns::W, "outlineLvl"))
        .and_then(|e| e.attr(ns::W, "val"))
        .and_then(|v| v.parse::<u8>().ok())
        .map(|l| if l < 9 { Some(l + 1) } else { None });
    let style_heading = match pstyle_id {
        Some(id) => ctx.styles.heading_level(id)?,
        None => None,
    };
    let heading = direct_outline.or(style_heading).flatten();
    let style_block = match pstyle_id {
        Some(id) => ctx.styles.block_style(id)?,
        None => None,
    };

    // Numbering resolves independently of heading semantics: a numbered
    // heading advances its sequence and keeps its number visible.
    let Numbered { item: numbering, level } = resolve_numbering(ppr, pstyle_id, ctx)?;
    // markitai: where its lines start, and whether it is numbered at a level
    // that shows no marker, for the list before it.
    let place = Place {
        indent: paragraph_indent(p, ctx.styles, level)?,
        unmarked: level.filter(|level| !level.marked).map(|level| level.ilvl),
    };

    // Toggle properties: the paragraph style chain's true-parity flips the
    // docDefaults base. Headings use the same resolution as body text.
    let parity = match pstyle_id {
        Some(id) => ctx.styles.run_toggles(id)?,
        None => Default::default(),
    };
    let paragraph_level = parity.apply_over(ctx.styles.doc_defaults);

    let kind = match heading {
        Some(level) => {
            let label = match &numbering {
                Some((_, key, number, label)) if key.marker.ordered() => {
                    Some(format!("{} ", label.clone().unwrap_or_else(|| key.marker.label(*number))))
                }
                _ => None,
            };
            ParaKind::Heading { level, label, base: paragraph_level }
        }
        None => match numbering {
            Some((ilvl, key, number, label)) => ParaKind::ListItem { ilvl, key, number, label },
            None => match style_block {
                Some(style) => ParaKind::Styled(style),
                None => ParaKind::Plain,
            },
        },
    };

    let mut walker = InlineWalker::new(ctx, paragraph_level);
    // markitai: a paragraph style can hide its text like a run can.
    walker.hidden = match pstyle_id {
        Some(id) => ctx.styles.run_hidden(id)?.unwrap_or(false),
        None => false,
    };
    // markitai: the font the paragraph's style, or the default one, sets
    // its runs in.
    walker.font = match pstyle_id.or(ctx.styles.default_paragraph) {
        Some(id) => ctx.styles.style_font(id)?,
        None => None,
    }
    .or(ctx.styles.default_font);
    // markitai: the size the paragraph's style, or the default one, sets
    // its runs in (Word shows 10 points where nothing names one).
    walker.size = match pstyle_id.or(ctx.styles.default_paragraph) {
        Some(id) => ctx.styles.style_size(id)?,
        None => None,
    }
    .or(ctx.styles.default_size)
    .unwrap_or(20);
    walker.walk(p)?;
    // markitai: a paragraph whose text is all monospaced is a line of code,
    // except in a table cell, where no code block can go. Without text the
    // paragraph mark's font decides (a blank line of a listing). A heading
    // set in such a font is typography, not code.
    let mono = walker.fonts.all_mono().unwrap_or_else(|| {
        ctx.code_fonts
            && ppr
                .and_then(|pr| pr.find(ns::W, "rPr"))
                .and_then(run_font)
                .or(walker.font)
                .is_some_and(is_monospace)
    });
    let size = walker.para_size;
    let mut pieces = walker.finish();
    // markitai: in the body, a monospaced paragraph opening with a bullet is
    // an item of a list typed by hand, not a line of code.
    let bullet = ctx.lists.is_some()
        && ctx.block_depth.get() == 1
        && matches!(&pieces[..], [Piece::Inlines(inlines)] if opens_with_bullet(inlines));
    let kind = match kind {
        ParaKind::Plain if mono && ctx.cell_depth.get() == 0 && !bullet => {
            ParaKind::Styled(BlockStyle::Code)
        }
        ParaKind::Heading { .. } if mono => {
            for piece in &mut pieces {
                if let Piece::Inlines(inlines) = piece {
                    without_code(inlines);
                }
            }
            kind
        }
        kind => kind,
    };
    Ok((kind, pieces, size, place))
}

/// markitai: count a run of a paragraph in `fonts` (see
/// [`crate::shared::code::RunFonts`]): a run with no text is not counted,
/// and one of only spaces is blank.
fn count_run(fonts: &mut RunFonts, run: &Element, mono: bool) {
    let mut texts = run.child_elems().filter(|c| c.is(ns::W, "t")).peekable();
    if texts.peek().is_none() {
        return;
    }
    let blank = !texts.any(|t| !t.text().trim().is_empty());
    fonts.run(blank, mono);
}

/// markitai: the tab stops a paragraph is set at: its style's, through
/// `basedOn`, then its own `w:tabs` (a `clear` stop removes an inherited
/// one; a bar tab only draws a line). `None` for a paragraph of a table of
/// contents or an index, whose tabs lead to page numbers.
fn paragraph_stops(p: &Element, styles: &Styles) -> Result<Option<Stops>, ConvertError> {
    let ppr = p.find(ns::W, "pPr");
    let style = ppr
        .and_then(|pr| pr.find(ns::W, "pStyle"))
        .and_then(|e| e.attr(ns::W, "val"))
        .or(styles.default_paragraph);
    let mut set = std::collections::BTreeMap::new();
    if let Some(id) = style {
        if styles.style_name(id).is_some_and(lists_pages) {
            return Ok(None);
        }
        for tabs in styles.style_tabs(id)?.into_iter().rev() {
            tab_stops(tabs, &mut set);
        }
    }
    if let Some(tabs) = ppr.and_then(|pr| pr.find(ns::W, "tabs")) {
        tab_stops(tabs, &mut set);
    }
    let mut stops = Stops::default();
    for (position, (align, leader)) in set {
        stops.add(position, align, leader);
    }
    Ok(Some(stops))
}

/// markitai: apply a `w:tabs` element to the stops set so far, by position.
fn tab_stops(tabs: &Element, set: &mut std::collections::BTreeMap<i64, (&'static str, bool)>) {
    for tab in tabs.child_elems().filter(|c| c.is(ns::W, "tab")) {
        let Some(position) = tab.attr(ns::W, "pos").and_then(|v| v.trim().parse::<i64>().ok())
        else {
            continue;
        };
        let align = match tab.attr(ns::W, "val").unwrap_or("left") {
            "clear" => {
                set.remove(&position);
                continue;
            }
            "bar" => continue,
            "end" | "right" => "right",
            "center" => "center",
            "decimal" => "decimal",
            "num" => "num",
            _ => "left",
        };
        let leader = !matches!(tab.attr(ns::W, "leader"), None | Some("none"));
        set.insert(position, (align, leader));
    }
}

/// markitai: where a paragraph's lines start, for the lists typed by hand
/// (see [`crate::shared::typed_lists`]) and an item's continuations: its
/// own `w:ind`, then its numbering level's (`level`), then its style's
/// through `basedOn`, attribute by attribute, as Word applies them. A
/// hanging indent wins over a first-line one, and character units over
/// twips, as in Word; `textutil` writes the first line's offset as
/// `w:first-line`. `None` for a paragraph of a table of contents or an
/// index, which lists pages.
fn paragraph_indent(
    p: &Element,
    styles: &Styles,
    level: Option<NumLevel>,
) -> Result<Option<Indent>, ConvertError> {
    let ppr = p.find(ns::W, "pPr");
    let style = ppr
        .and_then(|pr| pr.find(ns::W, "pStyle"))
        .and_then(|e| e.attr(ns::W, "val"))
        .or(styles.default_paragraph);
    let direct = ppr.and_then(|pr| pr.find(ns::W, "ind"));
    let mut inherited: Vec<&Element> = Vec::new();
    if let Some(id) = style {
        if styles.style_name(id).is_some_and(lists_pages) {
            return Ok(None);
        }
        inherited.extend(styles.style_indents(id)?);
    }
    let left = direct
        .and_then(ind_left)
        .or(level.and_then(|level| level.left))
        .or_else(|| inherited.iter().find_map(|ind| ind_left(ind)));
    let first_line = direct
        .and_then(ind_first_line)
        .or(level.and_then(|level| level.first_line))
        .or_else(|| inherited.iter().find_map(|ind| ind_first_line(ind)));
    Ok(Some(Indent { left: left.unwrap_or(0), first_line: first_line.unwrap_or(0) }))
}

/// markitai: the left indent a `w:ind` sets, in twips.
pub(super) fn ind_left(ind: &Element) -> Option<i32> {
    indent_value(ind, &["startChars", "leftChars"], &["start", "left"])
}

/// markitai: the first line's offset a `w:ind` sets, in twips (negative
/// for a hanging indent).
pub(super) fn ind_first_line(ind: &Element) -> Option<i32> {
    indent_value(ind, &["hangingChars"], &["hanging"])
        .map(|hanging| -hanging)
        .or_else(|| indent_value(ind, &["firstLineChars"], &["firstLine", "first-line"]))
}

/// markitai: one `w:ind` length in twips: a non-zero count in hundredths
/// of a character (taken as a 12-point em), else a length in twips or with
/// its unit.
fn indent_value(ind: &Element, chars: &[&str], lengths: &[&str]) -> Option<i32> {
    chars
        .iter()
        .find_map(|name| {
            let hundredths = ind.attr(ns::W, name)?.trim().parse::<i32>().ok()?;
            (hundredths != 0).then(|| hundredths.saturating_mul(12) / 5)
        })
        .or_else(|| {
            lengths
                .iter()
                .find_map(|name| crate::shared::typed_lists::twips(ind.attr(ns::W, name)?))
        })
}

/// markitai: a paragraph style of a table of contents or an index.
fn lists_pages(name: &str) -> bool {
    let name = name.trim().to_ascii_lowercase();
    name.starts_with("toc ")
        || name.starts_with("index ")
        || matches!(name.as_str(), "table of figures" | "table of authorities")
}

/// Resolve a paragraph's effective numbering per ECMA-376: the direct
/// `numPr` children are tri-state and merge property-by-property with the
/// style-inherited `numPr` (a missing `numId`/`ilvl` inherits; an explicit
/// `numId` of 0 suppresses). Returns the level, list identity, effective
/// number, and composite label, advancing the instance counters.
/// markitai: and the level the paragraph is numbered at, whether its marker
/// shows or not, with that level's own indent.
fn resolve_numbering(
    ppr: Option<&Element>,
    pstyle_id: Option<&str>,
    ctx: &Ctx,
) -> Result<Numbered, ConvertError> {
    let direct = ppr.and_then(|pr| pr.find(ns::W, "numPr"));
    let direct_num_id: Option<u64> = direct
        .and_then(|numpr| numpr.find(ns::W, "numId"))
        .and_then(|e| e.attr(ns::W, "val"))
        .and_then(|v| v.parse().ok());
    let direct_ilvl: Option<usize> = direct
        .and_then(|numpr| numpr.find(ns::W, "ilvl"))
        .and_then(|e| e.attr(ns::W, "val"))
        .and_then(|v| v.parse().ok());

    let num_id = match direct_num_id {
        Some(id) => Some(id),
        None => match pstyle_id {
            Some(id) => ctx.styles.style_num_pr(id)?,
            None => None,
        },
    };
    let Some(num_id) = num_id else {
        return Ok(Numbered::default());
    };
    if num_id == 0 {
        // numId 0 is explicitly suppressed numbering.
        return Ok(Numbered::default());
    }
    let Some(instance) = ctx.numbering.instance(num_id) else {
        log::debug!("paragraph references undefined numbering instance {num_id}");
        return Ok(Numbered::default());
    };
    let ilvl = match (direct_ilvl, pstyle_id) {
        (Some(l), _) => l,
        // Style-referenced numbering carries no usable ilvl (§17.3.1.19);
        // the level comes from the abstract levels' pStyle bindings.
        (None, Some(id)) => ctx.styles.style_numbering_level(id, instance)?.unwrap_or(0),
        (None, None) => 0,
    };
    let def = &instance.levels[ilvl.min(crate::formats::docx::numbering::LEVELS - 1)];
    let level = Some(NumLevel {
        ilvl,
        marked: def.marker.is_some(),
        left: def.left,
        first_line: def.first_line,
    });
    // A numFmt of "none" is explicitly suppressed numbering.
    let Some(marker) = def.marker else {
        return Ok(Numbered { item: None, level });
    };
    let (number, label) = if marker.ordered() {
        ctx.counters.borrow_mut().next(num_id, ilvl, instance)
    } else {
        (0, None)
    };
    Ok(Numbered { item: Some((ilvl, ListKey { instance: num_id, marker }, number, label)), level })
}

/// A paragraph's numbering (see [`resolve_numbering`]).
#[derive(Default)]
struct Numbered {
    /// The level, list identity, effective number and composite label of a
    /// numbered paragraph whose marker shows.
    item: Option<(usize, ListKey, u64, Option<String>)>,
    /// markitai: the level it is numbered at, marker or not.
    level: Option<NumLevel>,
}

/// markitai: a numbering level a paragraph is set at.
#[derive(Debug, Clone, Copy)]
struct NumLevel {
    ilvl: usize,
    /// Whether its marker shows (not `none`, nor a bullet of spaces).
    marked: bool,
    /// Its own indent, in twips (see `numbering::LevelDef`).
    left: Option<i32>,
    first_line: Option<i32>,
}

struct InlineWalker<'a, 'b, 'e> {
    ctx: &'e Ctx<'a, 'b>,
    base: Style,
    /// markitai: whether the paragraph style hides the text of its runs.
    hidden: bool,
    /// markitai: the font the paragraph style sets runs in, and the fonts of
    /// the runs read so far.
    font: Option<&'b str>,
    fonts: RunFonts,
    /// markitai: the size the paragraph style sets runs in, and the sizes
    /// of the visible text read so far.
    size: Size,
    para_size: ParaSize,
    pieces: Vec<Piece>,
    current: Vec<Inline>,
    fields: Vec<FieldFrame>,
}

impl<'a, 'b, 'e> InlineWalker<'a, 'b, 'e> {
    fn new(ctx: &'e Ctx<'a, 'b>, base: Style) -> Self {
        InlineWalker {
            ctx,
            base,
            hidden: false,
            font: None,
            fonts: RunFonts::default(),
            size: 20,
            para_size: ParaSize::default(),
            pieces: Vec::new(),
            current: Vec::new(),
            fields: Vec::new(),
        }
    }

    /// A walker for content nested in this one (a hyperlink, a simple
    /// field), which inherits the paragraph's formatting.
    fn nested(&self) -> Self {
        let mut inner = InlineWalker::new(self.ctx, self.base);
        inner.hidden = self.hidden;
        inner.font = self.font;
        inner.size = self.size;
        inner
    }

    fn push(&mut self, inline: Inline) {
        match self.fields.last_mut() {
            Some(f) if f.in_result => f.inlines.push(inline),
            Some(_) => {}
            None => self.current.push(inline),
        }
    }

    /// Attach block content at the current position in run order.
    fn push_blocks(&mut self, blocks: Vec<Block>) {
        if blocks.is_empty() {
            return;
        }
        if !self.current.is_empty() {
            self.pieces.push(Piece::Inlines(std::mem::take(&mut self.current)));
        }
        self.pieces.push(Piece::Blocks(blocks));
    }

    fn walk(&mut self, elem: &Element) -> Result<(), ConvertError> {
        for child in elem.child_elems() {
            if child.is(ns::MC, "AlternateContent") {
                if let Some(branch) = self.ctx.alternate_branch(child) {
                    self.walk(branch)?;
                }
                continue;
            }
            if child.is(ns::M, "oMathPara") {
                // A math paragraph is displayed on its own line.
                self.push_blocks(omath_para_to_tex(child).into_iter().map(Block::Math).collect());
                continue;
            }
            if child.is(ns::M, "oMath") {
                let tex = omath_to_tex(child);
                if !tex.is_empty() {
                    self.push(Inline::Math(tex));
                }
                continue;
            }
            if child.ns.as_deref().is_none_or(|n| n != ns::W) {
                continue;
            }
            match child.local.as_str() {
                "pPr" => {}
                "r" => self.walk_run(child)?,
                "hyperlink" => {
                    let target = self.hyperlink_link_target(child);
                    let mut inner = self.nested();
                    inner.walk(child)?;
                    self.fonts.add(inner.fonts);
                    self.para_size.merge(inner.para_size);
                    let (content, attachments) = split_pieces(inner.finish());
                    if let Some(target) = target {
                        // An empty label still keeps a resolved target: the
                        // renderer shows the URL as the link text.
                        self.push(Inline::Link { content, target });
                    } else {
                        for inline in content {
                            self.push(inline);
                        }
                    }
                    self.push_blocks(attachments);
                }
                "fldSimple" => {
                    let instr = child.attr(ns::W, "instr").unwrap_or("").to_string();
                    let mut inner = self.nested();
                    inner.walk(child)?;
                    self.fonts.add(inner.fonts);
                    self.para_size.merge(inner.para_size);
                    let (content, attachments) = split_pieces(inner.finish());
                    self.push_field_result(&instr, content);
                    self.push_blocks(attachments);
                }
                "bookmarkStart" => {
                    if let Some(name) = child.attr(ns::W, "name")
                        && name != "_GoBack"
                    {
                        self.push(Inline::Anchor(name.to_string()));
                    }
                }
                "sdt" => {
                    if let Some(content) = child.find(ns::W, "sdtContent") {
                        self.walk(content)?;
                    }
                }
                // `moveTo` is moved-in text — part of the final document;
                // `customXml` wraps ordinary run content.
                "smartTag" | "ins" | "bdo" | "dir" | "moveTo" | "customXml" => self.walk(child)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn hyperlink_link_target(&self, link: &Element) -> Option<LinkTarget> {
        if let Some(id) = link.attr_qualified(ns::R, "id")
            && let Some(rel) = self.ctx.rels.get(id)
        {
            return Some(crate::shared::fields::classify_rel_target(
                rel.mode == TargetMode::External,
                &rel.target,
            ));
        }
        link.attr(ns::W, "anchor").map(|a| LinkTarget::Anchor(a.to_string()))
    }

    fn walk_run(&mut self, run: &Element) -> Result<(), ConvertError> {
        let mut hidden = self.hidden;
        let mut script = None;
        let style = match run.find(ns::W, "rPr") {
            Some(rpr) => {
                // Character-style chain: another toggle layer over the
                // paragraph-level value. Direct formatting is absolute.
                let char_style = rpr.find(ns::W, "rStyle").and_then(|e| e.attr(ns::W, "val"));
                let char_parity = match char_style {
                    Some(id) => self.ctx.styles.run_toggles(id)?,
                    None => Default::default(),
                };
                // markitai: hidden text (`w:vanish`) follows direct
                // formatting, then the character style, then the paragraph's.
                hidden = match on_off(rpr, "vanish") {
                    Some(direct) => direct,
                    None => match char_style {
                        Some(id) => self.ctx.styles.run_hidden(id)?.unwrap_or(hidden),
                        None => hidden,
                    },
                };
                // markitai: raised or lowered text, likewise.
                script = match rpr.find(ns::W, "vertAlign").and_then(|e| e.attr(ns::W, "val")) {
                    Some(value) => Script::from_value(value),
                    None => match char_style {
                        Some(id) => self.ctx.styles.run_script(id)?.flatten(),
                        None => None,
                    },
                };
                let with_char = char_parity.apply_over(self.base);
                rpr_delta(rpr).apply(with_char)
            }
            None => self.base,
        };
        // markitai: a run set in a monospaced font is code: its own font,
        // else its character style's, else the paragraph's.
        let mut style = style;
        if self.ctx.code_fonts && !hidden {
            let rpr = run.find(ns::W, "rPr");
            let font = match rpr.and_then(run_font) {
                Some(font) => Some(font),
                None => match rpr
                    .and_then(|rpr| rpr.find(ns::W, "rStyle"))
                    .and_then(|e| e.attr(ns::W, "val"))
                {
                    Some(id) => self.ctx.styles.style_font(id)?,
                    None => None,
                }
                .or(self.font),
            };
            let mono = font.is_some_and(is_monospace);
            count_run(&mut self.fonts, run, mono);
            style.code |= mono;
        }
        // markitai: the run's size: its own, else its character style's,
        // else the paragraph's.
        let size = match run.find(ns::W, "rPr") {
            Some(rpr) => match run_size(rpr) {
                Some(size) => Some(size),
                None => match rpr.find(ns::W, "rStyle").and_then(|e| e.attr(ns::W, "val")) {
                    Some(id) => self.ctx.styles.style_size(id)?,
                    None => None,
                },
            },
            None => None,
        }
        .unwrap_or(self.size);
        self.walk_run_content(run, style, hidden, script, size)
    }

    /// markitai: text set at `size` was read; a field's instructions are not
    /// visible.
    fn saw_text(&mut self, size: Size, text: &str) {
        if let Some(looks) = self.ctx.looks
            && self.fields.last().is_none_or(|field| field.in_result)
        {
            looks.borrow_mut().text(&mut self.para_size, size, text);
        }
    }

    fn walk_run_content(
        &mut self,
        run: &Element,
        style: Style,
        hidden: bool,
        script: Option<Script>,
        size: Size,
    ) -> Result<(), ConvertError> {
        for child in run.child_elems() {
            if child.is(ns::MC, "AlternateContent") {
                if let Some(branch) = self.ctx.alternate_branch(child) {
                    self.walk_run_content(branch, style, hidden, script, size)?;
                }
                continue;
            }
            let in_w = child.ns.as_deref().is_some_and(|n| n == ns::W);
            if !in_w {
                continue;
            }
            // markitai: a hidden run shows nothing, but its field marks still
            // open and close the fields around it.
            if hidden && !matches!(child.local.as_str(), "fldChar" | "instrText") {
                continue;
            }
            match child.local.as_str() {
                "t" => {
                    // Run edges carry the spacing between words in documents
                    // that never mark xml:space, and XML leaves unmarked
                    // whitespace to the application, so it is kept.
                    let text = clean_text(child.text().as_ref());
                    // markitai: a raised or lowered run in its Unicode forms
                    // where it has them ("10⁻³", "H₂O").
                    let text = script.and_then(|script| script.convert(&text)).unwrap_or(text);
                    if !text.is_empty() {
                        self.saw_text(size, &text);
                        self.push(Inline::Text { text, style });
                    }
                }
                // markitai: a tab of the body is kept until the tables set
                // with tab stops are found (see `crate::shared::tabs`).
                "tab" if self.ctx.tabs.is_some() => self.push(tabs::tab(Style::PLAIN)),
                "tab" | "ptab" => self.push(Inline::Text { text: " ".into(), style: Style::PLAIN }),
                // Markdown has no pages or columns, but every w:br still
                // separates the runs around it: dropping a page break
                // outright would join the words on either side. One left at
                // the end of a paragraph is trimmed when the block renders.
                "br" => self.push(Inline::LineBreak),
                "cr" => self.push(Inline::LineBreak),
                // markitai: a non-breaking hyphen is a hyphen; dropping it ran
                // "e-mail" together. A soft hyphen shows nothing unless a line
                // breaks there, so it stays out of the text.
                "noBreakHyphen" => {
                    self.saw_text(size, "-");
                    self.push(Inline::Text { text: "-".into(), style });
                }
                // markitai: the base text of a phonetic guide is the text; the
                // guide (`w:rt`, furigana or pinyin) only annotates it. Leaving
                // the whole element unread lost the words themselves.
                "ruby" => {
                    if let Some(base) = child.find(ns::W, "rubyBase") {
                        self.walk(base)?;
                    }
                }
                // markitai: a character picked from Word's Symbol dialog.
                "sym" => {
                    if let (Some(font), Some(code)) =
                        (child.attr(ns::W, "font"), child.attr(ns::W, "char"))
                        && let Some(symbol) = super::symbols::symbol_char(font, code)
                    {
                        let text = symbol.to_string();
                        self.saw_text(size, &text);
                        self.push(Inline::Text { text, style });
                    }
                }
                "footnoteReference" => {
                    if let Some(id) = child.attr(ns::W, "id") {
                        self.push(Inline::NoteRef(format!("fn{id}")));
                    }
                }
                "endnoteReference" => {
                    if let Some(id) = child.attr(ns::W, "id") {
                        self.push(Inline::NoteRef(format!("en{id}")));
                    }
                }
                "drawing" | "pict" | "object" => self.walk_drawing(child)?,
                "fldChar" => match child.attr(ns::W, "fldCharType") {
                    Some("begin") => self.fields.push(FieldFrame::default()),
                    Some("separate") => {
                        if let Some(f) = self.fields.last_mut() {
                            f.in_result = true;
                        }
                    }
                    Some("end") => {
                        if let Some(FieldFrame { instr, inlines, .. }) = self.fields.pop() {
                            self.push_field_result(&instr, inlines);
                        }
                    }
                    _ => {}
                },
                "instrText" => {
                    if let Some(f) = self.fields.last_mut() {
                        f.instr.push_str(&child.text());
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Drawings, VML picts, and embedded objects: text boxes become block
    /// attachments at this position; images/charts/diagrams/objects resolve
    /// through relationships.
    fn walk_drawing(&mut self, elem: &Element) -> Result<(), ConvertError> {
        // Text boxes first: their content is the drawing's content.
        let mut boxes = Vec::new();
        collect_text_boxes(elem, &mut boxes);
        if !boxes.is_empty() {
            let mut blocks = Vec::new();
            for tb in boxes {
                blocks.extend(parse_blocks(tb, self.ctx)?);
            }
            self.push_blocks(blocks);
            return Ok(());
        }

        // markitai: WordArt (a VML `v:textpath`) keeps its words in an
        // attribute, where no text-box content is to be found; a title set in
        // WordArt was left out of the document.
        let word_art: Vec<String> = elem
            .descendants(ns::VML, "textpath")
            .filter_map(|path| path.attr_any("string"))
            .map(clean_text)
            .filter(|text| !text.trim().is_empty())
            .collect();
        if !word_art.is_empty() {
            self.push(Inline::Text { text: word_art.join(" "), style: Style::PLAIN });
            return Ok(());
        }

        let descr = elem
            .first_descendant(ns::WP, "docPr")
            .and_then(|d| d.attr(ns::WP, "descr"))
            .map(clean_text)
            .unwrap_or_default();

        // Charts and SmartArt render their textual content.
        if let Some(chart_ref) = elem.first_descendant(ns::CHART, "chart")
            && let Some(rel_id) = chart_ref.attr_qualified(ns::R, "id")
        {
            self.push_blocks(self.chart_blocks(rel_id)?);
            return Ok(());
        }
        if let Some(rel_ids) = elem.first_descendant(ns::DGM, "relIds")
            && let Some(rel_id) = rel_ids.attr_qualified(ns::R, "dm")
        {
            self.push_blocks(self.diagram_blocks(rel_id)?);
            return Ok(());
        }

        // Embedded OLE objects come before images: standard Word OLE markup
        // carries a VML preview image next to the o:OLEObject, and the
        // object's identity and payload must win over its preview.
        if let Some(ole) = elem.first_descendant(ns::O_VML, "OLEObject") {
            let prog_id = ole.attr(ns::O_VML, "ProgID").unwrap_or("object").to_string();
            let alt = if descr.trim().is_empty() {
                format!("Embedded object: {prog_id}")
            } else {
                descr.clone()
            };
            let source = match ole.attr_qualified(ns::R, "id") {
                Some(rel_id) => match self.ctx.rel_part(rel_id)? {
                    Some((part, bytes)) => Some(ImageSource::Asset(self.ctx.add_asset(
                        "application/vnd.ms-ole-object".into(),
                        part,
                        &bytes,
                    )?)),
                    None => None,
                },
                None => None,
            };
            self.push(Inline::Image { alt, source: source.unwrap_or(ImageSource::Unavailable) });
            return Ok(());
        }

        // Bitmap images: DrawingML blip or VML imagedata.
        let image_rel = elem
            .first_descendant(ns::A, "blip")
            .and_then(|b| {
                b.attr_qualified(ns::R, "embed").or_else(|| b.attr_qualified(ns::R, "link"))
            })
            .or_else(|| {
                elem.first_descendant(ns::VML, "imagedata")
                    .and_then(|i| i.attr_qualified(ns::R, "id"))
            });
        if let Some(rel_id) = image_rel {
            // External-mode targets (`r:link`) become external image
            // sources; embedded targets are retained as assets.
            let source = crate::shared::assets::rel_image_source(
                self.ctx.pkg,
                &self.ctx.rels,
                &self.ctx.base_part,
                self.ctx.assets,
                rel_id,
            )?;
            match source {
                Some(source) => self.push(Inline::Image { alt: descr, source }),
                None => {
                    if !descr.trim().is_empty() {
                        self.push(Inline::Image { alt: descr, source: ImageSource::Unavailable });
                    }
                }
            }
            return Ok(());
        }

        if !descr.trim().is_empty() {
            self.push(Inline::Image { alt: descr, source: ImageSource::Unavailable });
        }
        Ok(())
    }

    /// Textual extraction of a chart part via `shared::drawingml`.
    fn chart_blocks(&self, rel_id: &str) -> Result<Vec<Block>, ConvertError> {
        let Some((part, bytes)) = self.ctx.rel_part(rel_id)? else {
            return Ok(Vec::new());
        };
        match crate::package::xml::parse_xml(&bytes) {
            Ok(root) => Ok(crate::shared::drawingml::chart_blocks(&root)),
            Err(e) if e.is_fatal() => Err(e),
            Err(e) => {
                log::warn!("skipping corrupt chart part {part}: {e}");
                Ok(Vec::new())
            }
        }
    }

    /// Textual extraction of a SmartArt data part via `shared::drawingml`.
    fn diagram_blocks(&self, rel_id: &str) -> Result<Vec<Block>, ConvertError> {
        let Some((part, bytes)) = self.ctx.rel_part(rel_id)? else {
            return Ok(Vec::new());
        };
        match crate::package::xml::parse_xml(&bytes) {
            Ok(root) => Ok(crate::shared::drawingml::diagram_blocks(&root)),
            Err(e) if e.is_fatal() => Err(e),
            Err(e) => {
                log::warn!("skipping corrupt diagram part {part}: {e}");
                Ok(Vec::new())
            }
        }
    }

    fn push_field_result(&mut self, instr: &str, content: Vec<Inline>) {
        for inline in field_result(instr, content) {
            self.push(inline);
        }
    }

    fn finish(mut self) -> Vec<Piece> {
        while let Some(frame) = self.fields.pop() {
            for inline in frame.inlines {
                self.push(inline);
            }
        }
        if !self.current.is_empty() {
            self.pieces.push(Piece::Inlines(self.current));
        }
        self.pieces
    }
}

fn split_pieces(pieces: Vec<Piece>) -> (Vec<Inline>, Vec<Block>) {
    let mut inlines = Vec::new();
    let mut blocks = Vec::new();
    for piece in pieces {
        match piece {
            Piece::Inlines(mut i) => inlines.append(&mut i),
            Piece::Blocks(b) => blocks.extend(b),
        }
    }
    (inlines, blocks)
}

fn pieces_into_blocks(pieces: Vec<Piece>) -> Vec<Block> {
    let mut blocks = Vec::new();
    for piece in pieces {
        match piece {
            // Visually empty inlines still become a paragraph: a
            // bookmark-only one carries the anchor link resolution binds to.
            Piece::Inlines(inlines) => blocks.push(Block::Paragraph(inlines)),
            Piece::Blocks(attachments) => blocks.extend(attachments),
        }
    }
    blocks
}

/// Find text-box content (`w:txbxContent`) in a drawing or VML pict, skipping
/// `mc:Fallback` so AlternateContent shapes aren't collected twice.
fn collect_text_boxes<'e>(elem: &'e Element, out: &mut Vec<&'e Element>) {
    for child in elem.child_elems() {
        if child.is(ns::MC, "Fallback") {
            continue;
        }
        if child.is(ns::W, "txbxContent") {
            out.push(child);
        } else {
            collect_text_boxes(child, out);
        }
    }
}

// ---------------------------------------------------------------------------
// Tables

struct TcInfo<'e> {
    elem: Option<&'e Element>,
    /// Legacy `hMerge` continuation cells folded into this origin.
    merged: Vec<&'e Element>,
    col_span: usize,
    row_span: usize,
    /// vMerge continuation: this position belongs to the origin above.
    covered: bool,
}

impl TcInfo<'_> {
    /// Empty single-column filler for `gridBefore`/`gridAfter` positions.
    fn filler() -> Self {
        TcInfo { elem: None, merged: Vec::new(), col_span: 1, row_span: 1, covered: false }
    }
}

/// A `w:trPr` grid filler count (`gridBefore`/`gridAfter`).
fn grid_filler(trpr: Option<&Element>, name: &str) -> usize {
    trpr.and_then(|p| p.find(ns::W, name))
        .and_then(|e| e.attr(ns::W, "val"))
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
        .min(1000)
}

pub(super) fn parse_table(tbl: &Element, ctx: &Ctx) -> Result<Vec<Block>, ConvertError> {
    // Collect the raw cell matrix first so vertical merges can be resolved
    // into row spans before the grid is built. gridBefore/gridAfter filler
    // materializes as empty cells so every cell keeps its grid column.
    let mut matrix: Vec<Vec<TcInfo>> = Vec::new();
    // markitai: a row whose deletion is tracked (`w:trPr/w:del`) is gone once
    // the changes are accepted, as its text already is; keeping it left an
    // empty row in the table.
    let live_rows: Vec<&Element> = tbl
        .find_all(ns::W, "tr")
        .filter(|tr| tr.find(ns::W, "trPr").and_then(|p| p.find(ns::W, "del")).is_none())
        .collect();
    for &tr in &live_rows {
        let trpr = tr.find(ns::W, "trPr");
        let mut row = Vec::new();
        row.extend((0..grid_filler(trpr, "gridBefore")).map(|_| TcInfo::filler()));
        collect_row_cells(tr, &mut row);
        row.extend((0..grid_filler(trpr, "gridAfter")).map(|_| TcInfo::filler()));
        matrix.push(row);
    }
    // Active vertical-merge chains by grid column.
    let mut active: HashMap<usize, (usize, usize)> = HashMap::new();
    let rows_len = matrix.len();
    for r in 0..rows_len {
        let mut col = 0usize;
        let mut next_active: Vec<(usize, (usize, usize))> = Vec::new();
        for i in 0..matrix[r].len() {
            let (covered, span) = (matrix[r][i].covered, matrix[r][i].col_span);
            if covered {
                if let Some(&(orow, oidx)) = active.get(&col) {
                    matrix[orow][oidx].row_span += 1;
                    for c in col..col + span {
                        next_active.push((c, (orow, oidx)));
                    }
                } else {
                    matrix[r][i].covered = false; // stray continuation
                }
            } else {
                for c in col..col + span {
                    next_active.push((c, (r, i)));
                }
            }
            col += span;
        }
        active = next_active.into_iter().collect();
    }

    // tblHeader is ST_OnOff: an explicit false value is not a header row.
    let header_rows = live_rows
        .iter()
        .take_while(|tr| tr.find(ns::W, "trPr").and_then(|p| on_off(p, "tblHeader")) == Some(true))
        .count();

    let mut builder = GridBuilder::new();
    let _in_cell = InCell::enter(&ctx.cell_depth);
    for row in &matrix {
        builder.next_row();
        for tc in row {
            if tc.covered {
                for _ in 0..tc.col_span {
                    builder.covered();
                }
            } else {
                let mut blocks = match tc.elem {
                    Some(elem) => parse_blocks(elem, ctx)?,
                    None => Vec::new(),
                };
                for merged in &tc.merged {
                    blocks.extend(parse_blocks(merged, ctx)?);
                }
                builder.place(Cell::spanning(blocks, tc.col_span as u32, tc.row_span as u32))?;
            }
        }
    }
    let mut table = builder.finish(TableKind::Data);
    if table.grid.is_empty() {
        return Ok(Vec::new());
    }
    // markitai: the rows the table declares as its header (`w:tblHeader`, a
    // row repeated at the top of each page) and no others. A Word table
    // has no other notion of a header row, and reading one from the types
    // of the columns below made a first row of data a header.
    table.header_rows = header_rows.min(table.grid.len());
    Ok(vec![Block::Table(table)])
}

fn collect_row_cells<'e>(parent: &'e Element, cells: &mut Vec<TcInfo<'e>>) {
    for child in parent.child_elems() {
        if child.ns.as_deref().is_none_or(|n| n != ns::W) {
            continue;
        }
        match child.local.as_str() {
            "tc" => {
                let tcpr = child.find(ns::W, "tcPr");
                let covered = tcpr
                    .and_then(|p| p.find(ns::W, "vMerge"))
                    .is_some_and(|v| !matches!(v.attr(ns::W, "val"), Some("restart")));
                let col_span: usize = tcpr
                    .and_then(|p| p.find(ns::W, "gridSpan"))
                    .and_then(|e| e.attr(ns::W, "val"))
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(1)
                    .clamp(1, 1000);
                // Legacy hMerge: a non-restart hMerge cell folds into the
                // preceding origin, widening its span.
                let hmerge_cont = tcpr
                    .and_then(|p| p.find(ns::W, "hMerge"))
                    .is_some_and(|h| !matches!(h.attr(ns::W, "val"), Some("restart")));
                if hmerge_cont
                    && let Some(prev) = cells.last_mut()
                    && !prev.covered
                {
                    prev.col_span += col_span;
                    prev.merged.push(child);
                    continue;
                }
                cells.push(TcInfo {
                    elem: Some(child),
                    merged: Vec::new(),
                    col_span,
                    row_span: 1,
                    covered,
                });
            }
            "sdt" => {
                if let Some(content) = child.find(ns::W, "sdtContent") {
                    collect_row_cells(content, cells);
                }
            }
            "customXml" => collect_row_cells(child, cells),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::MarkerKind;

    fn text(value: &str) -> Piece {
        Piece::Inlines(vec![Inline::plain(value)])
    }

    fn assert_text(block: &Block, expected: &str) {
        let Block::Paragraph(inlines) = block else { panic!("expected paragraph: {block:?}") };
        assert_eq!(crate::model::inlines_to_plain_text(inlines), expected);
    }

    #[test]
    fn heading_attachments_keep_source_order() {
        let mut blocks = Vec::new();
        emit_paragraph(
            ParaKind::Heading { level: 2, label: None, base: Style::PLAIN },
            vec![text("before"), Piece::Blocks(vec![Block::Rule]), text("after")],
            Place::default(),
            &mut blocks,
            &mut Runs::default(),
        );
        let [Block::Heading { content, .. }, Block::Rule, after] = &blocks[..] else {
            panic!("unexpected blocks: {blocks:?}")
        };
        assert_eq!(crate::model::inlines_to_plain_text(content), "before");
        assert_text(after, "after");
    }

    #[test]
    fn list_attachments_keep_source_order_inside_the_item() {
        let mut blocks = Vec::new();
        let mut runs = Runs::default();
        emit_paragraph(
            ParaKind::ListItem {
                ilvl: 0,
                key: ListKey { instance: 1, marker: MarkerKind::Bullet },
                number: 0,
                label: None,
            },
            vec![text("before"), Piece::Blocks(vec![Block::Rule]), text("after")],
            Place::default(),
            &mut blocks,
            &mut runs,
        );
        runs.flush(&mut blocks);
        let [Block::List(list)] = &blocks[..] else { panic!("unexpected blocks: {blocks:?}") };
        let [before, Block::Rule, after] = &list.items[0].blocks[..] else {
            panic!("unexpected item: {:?}", list.items[0].blocks)
        };
        assert_text(before, "before");
        assert_text(after, "after");
    }

    #[test]
    fn a_bookmark_only_paragraph_keeps_its_anchor() {
        // `inlines_are_empty` is true of a lone anchor, but dropping the
        // paragraph would strip the target a link resolves against.
        let mut blocks = Vec::new();
        emit_paragraph(
            ParaKind::Plain,
            vec![Piece::Inlines(vec![Inline::Anchor("mark".into())])],
            Place::default(),
            &mut blocks,
            &mut Runs::default(),
        );
        let [Block::Paragraph(inlines)] = &blocks[..] else { panic!("{blocks:?}") };
        assert!(matches!(inlines[..], [Inline::Anchor(_)]), "{inlines:?}");
    }

    #[test]
    fn an_empty_list_item_carries_no_blocks() {
        let mut blocks = Vec::new();
        let mut runs = Runs::default();
        emit_paragraph(
            ParaKind::ListItem {
                ilvl: 0,
                key: ListKey { instance: 1, marker: MarkerKind::Bullet },
                number: 0,
                label: None,
            },
            Vec::new(),
            Place::default(),
            &mut blocks,
            &mut runs,
        );
        runs.flush(&mut blocks);
        let [Block::List(list)] = &blocks[..] else { panic!("{blocks:?}") };
        assert!(list.items[0].blocks.is_empty(), "{:?}", list.items[0].blocks);
    }

    #[test]
    fn attachments_split_styled_runs_in_place() {
        let mut blocks = Vec::new();
        let mut runs = Runs::default();
        emit_paragraph(
            ParaKind::Styled(BlockStyle::Code),
            vec![text("before"), Piece::Blocks(vec![Block::Rule]), text("after")],
            Place::default(),
            &mut blocks,
            &mut runs,
        );
        runs.flush(&mut blocks);
        let [
            Block::CodeBlock { text: before, .. },
            Block::Rule,
            Block::CodeBlock { text: after, .. },
        ] = &blocks[..]
        else {
            panic!("unexpected blocks: {blocks:?}")
        };
        assert_eq!(before, "before");
        assert_eq!(after, "after");
    }
}
