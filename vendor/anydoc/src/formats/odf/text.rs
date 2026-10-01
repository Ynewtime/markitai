//! Block and inline walking for ODF text content.

use crate::error::ConvertError;
use crate::formats::docx::scripts::Script;
use crate::formats::odf::styles::{LIST_LEVELS, OdfStyles, parse_start};
use crate::formats::odf::table::parse_table;
use crate::model::{
    Block, ImageSource, Inline, LinkTarget, List, ListItem, Note, NoteKind, Style,
    inlines_are_empty,
};
use crate::package::Package;
use crate::package::xml::{Element, Node, ns};
use crate::shared::assets::{AssetSink, media_type_for};
use crate::shared::blockstyle::StyledRun;
use crate::shared::delta::{StyleDelta, rebase_emphasis};
use crate::shared::math::mathml_to_tex;
use crate::shared::text::{clean_text, collapse_ws};
use crate::shared::visual::{Looks, ParaSize, Size};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

pub struct Ctx<'a, 'b> {
    pub styles: &'b OdfStyles<'b>,
    pub pkg: &'b RefCell<Package<'a>>,
    pub assets: &'b RefCell<AssetSink>,
    pub notes: RefCell<Vec<Note>>,
    /// Continuation counters per (list style, depth): the next number a
    /// `text:continue-numbering` list resumes at.
    list_counters: RefCell<HashMap<(String, usize), u64>>,
    /// Next number per list `xml:id`, resolved by `text:continue-list`.
    list_ids: RefCell<HashMap<String, u64>>,
    /// Heading numbering state for `text:outline-style` (values, started).
    heading_counters: RefCell<([u64; LIST_LEVELS], [bool; LIST_LEVELS])>,
    /// markitai: in a text document, what its body text looks like (see
    /// [`crate::shared::visual`]); the sizes of the paragraph being read;
    /// how many containers or lists deep the content being read sits; and
    /// how many notes deep (a note's text is not the body's).
    pub looks: Option<RefCell<Looks>>,
    para_size: Cell<ParaSize>,
    depth: Cell<u32>,
    note_depth: Cell<u32>,
}

impl<'a, 'b> Ctx<'a, 'b> {
    pub fn new(
        styles: &'b OdfStyles<'b>,
        pkg: &'b RefCell<Package<'a>>,
        assets: &'b RefCell<AssetSink>,
    ) -> Self {
        Ctx {
            styles,
            pkg,
            assets,
            notes: RefCell::new(Vec::new()),
            list_counters: Default::default(),
            list_ids: Default::default(),
            heading_counters: Default::default(),
            looks: None,
            para_size: Default::default(),
            depth: Default::default(),
            note_depth: Default::default(),
        }
    }

    /// markitai: visible text set at `size` was read.
    fn saw_text(&self, size: Size, text: &str) {
        if let Some(looks) = &self.looks
            && self.note_depth.get() == 0
        {
            let mut para = self.para_size.get();
            looks.borrow_mut().text(&mut para, size, text);
            self.para_size.set(para);
        }
    }
}

/// markitai: content read one container, list or note deeper, for as long
/// as it lives.
struct Deeper<'c>(&'c Cell<u32>);

impl<'c> Deeper<'c> {
    fn enter(depth: &'c Cell<u32>) -> Self {
        depth.set(depth.get() + 1);
        Deeper(depth)
    }
}

impl Drop for Deeper<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

pub fn parse_container(parent: &Element, ctx: &Ctx) -> Result<Vec<Block>, ConvertError> {
    let _deeper = Deeper::enter(&ctx.depth);
    let mut blocks = Vec::new();
    let mut run = StyledRun::default();
    for child in parent.child_elems() {
        parse_block_elem(child, ctx, &mut blocks, &mut run)?;
    }
    run.flush(&mut blocks);
    Ok(blocks)
}

fn parse_block_elem(
    elem: &Element,
    ctx: &Ctx,
    blocks: &mut Vec<Block>,
    run: &mut StyledRun,
) -> Result<(), ConvertError> {
    let in_text = elem.ns.as_deref().is_some_and(|n| n == ns::TEXT);
    if in_text {
        // Only a run of same-styled paragraphs continues a container.
        if !elem.is(ns::TEXT, "p") {
            run.flush(blocks);
        }
        match elem.local.as_str() {
            "h" => {
                let level = elem
                    .attr(ns::TEXT, "outline-level")
                    .and_then(|v| v.parse::<u8>().ok())
                    .unwrap_or(1);
                let (inlines, boxes, _) = parse_inline_content(elem, ctx)?;
                if !inlines_are_empty(&inlines) {
                    let mut content = inlines;
                    rebase_emphasis(&mut content, paragraph_base(elem, ctx)?.resolve());
                    // ODF outline links target headings by their text; carry
                    // it as the heading's anchor id (without the number).
                    let anchor = Some(crate::model::inlines_to_plain_text(&content));
                    if let Some(label) = heading_label(elem, level, ctx) {
                        content.insert(0, Inline::Text { text: label, style: Style::PLAIN });
                    }
                    blocks.push(Block::Heading { level, anchor, content });
                }
                blocks.extend(boxes);
                return Ok(());
            }
            "p" => {
                let (inlines, boxes, size) = parse_inline_content(elem, ctx)?;
                let style =
                    elem.attr(ns::TEXT, "style-name").and_then(|n| ctx.styles.block_style(n));
                match style {
                    Some(style) => run.push(style, inlines, blocks),
                    None => {
                        run.flush(blocks);
                        blocks.push(Block::Paragraph(inlines));
                        // markitai: a plain paragraph of the body itself may
                        // turn out to be a heading set by hand.
                        if boxes.is_empty()
                            && ctx.depth.get() == 1
                            && let Some(looks) = &ctx.looks
                        {
                            looks.borrow_mut().paragraph(blocks.len() - 1, size);
                        }
                    }
                }
                if !boxes.is_empty() {
                    // A frame is not part of the container's text.
                    run.flush(blocks);
                    blocks.extend(boxes);
                }
                return Ok(());
            }
            "list" => {
                blocks.extend(parse_list(elem, ctx, 0, None, &[])?);
                return Ok(());
            }
            // markitai: a section's paragraphs are the body's own, read into
            // the same blocks (as upstream's separate container read them,
            // its open run closing at the section's end), so a heading set
            // by hand inside one is found too.
            "section" => {
                let mut inner = StyledRun::default();
                for child in elem.child_elems() {
                    parse_block_elem(child, ctx, blocks, &mut inner)?;
                }
                inner.flush(blocks);
                return Ok(());
            }
            "index-body" | "index-title" => {
                blocks.extend(parse_container(elem, ctx)?);
                return Ok(());
            }
            "table-of-content" | "alphabetical-index" | "bibliography" | "illustration-index" => {
                // The stored index body is real document text (the generated
                // entries are written into `text:index-body`).
                blocks.extend(parse_container(elem, ctx)?);
                return Ok(());
            }
            _ => return Ok(()),
        }
    }
    if elem.is(ns::TABLE, "table") {
        run.flush(blocks);
        blocks.extend(parse_table(elem, ctx)?);
    }
    Ok(())
}

/// One `text:list` -> blocks: list headers render without markers (as
/// plain blocks alongside the list), and every `text:start-value` restart
/// after the first item splits the run into a new list with that start.
/// `ancestors` carries the enclosing levels' current numbers so composite
/// labels (`text:display-levels` > 1) render the full chain.
fn parse_list(
    elem: &Element,
    ctx: &Ctx,
    depth: usize,
    inherited_style: Option<&str>,
    ancestors: &[u64],
) -> Result<Vec<Block>, ConvertError> {
    // markitai: a list's paragraphs are not the body's own (see `Ctx::depth`).
    let _deeper = Deeper::enter(&ctx.depth);
    let style_name = elem.attr(ns::TEXT, "style-name").or(inherited_style);
    let level = ctx.styles.list_level(style_name.unwrap_or(""), depth);
    let ordered = level.marker.ordered();

    let counter_key = (style_name.unwrap_or("").to_string(), depth);
    let mut start = level.start;
    if ordered {
        // Continuation identity: an explicit target list (`continue-list`
        // by xml:id) wins over the style-scoped `continue-numbering`.
        if let Some(target) = elem.attr(ns::TEXT, "continue-list") {
            if let Some(&resume) = ctx.list_ids.borrow().get(target) {
                start = resume;
            }
        } else if elem.attr(ns::TEXT, "continue-numbering") == Some("true")
            && let Some(&resume) = ctx.list_counters.borrow().get(&counter_key)
        {
            start = resume;
        }
    }

    let mut out: Vec<Block> = Vec::new();
    let mut current = List { marker: level.marker, start, items: Vec::new() };
    let mut next = start;
    let mut first_item = true;
    let flush = |current: &mut List, out: &mut Vec<Block>, start: u64| {
        let done =
            std::mem::replace(current, List { marker: current.marker, start, items: Vec::new() });
        if !done.items.is_empty() {
            out.push(Block::List(done));
        }
    };
    for item in elem.child_elems() {
        let header = item.is(ns::TEXT, "list-header");
        if !(header || item.is(ns::TEXT, "list-item")) {
            continue;
        }
        // Establish this item's number before walking children: nested
        // lists render composite labels against the ancestor chain.
        if !header
            && ordered
            && let Some(sv) = item.attr(ns::TEXT, "start-value").and_then(parse_start)
        {
            if first_item {
                current.start = sv;
            } else {
                flush(&mut current, &mut out, sv);
            }
        }
        let number = current.start.saturating_add(current.items.len() as u64);
        let mut chain = ancestors.to_vec();
        chain.push(number);
        let mut item_blocks = Vec::new();
        let mut item_run = StyledRun::default();
        for child in item.child_elems() {
            if child.is(ns::TEXT, "list") {
                item_run.flush(&mut item_blocks);
                item_blocks.extend(parse_list(child, ctx, depth + 1, style_name, &chain)?);
            } else {
                parse_block_elem(child, ctx, &mut item_blocks, &mut item_run)?;
            }
        }
        item_run.flush(&mut item_blocks);
        if header {
            // A list header has no marker: its blocks sit next to the list
            // and do not consume a number.
            flush(&mut current, &mut out, next);
            out.extend(item_blocks);
            continue;
        }
        first_item = false;
        let label = item_label(ctx, style_name, depth, &chain);
        current.items.push(ListItem { blocks: item_blocks, marker_label: label });
        next = current.start.saturating_add(current.items.len() as u64);
    }
    flush(&mut current, &mut out, next);
    if ordered && !out.is_empty() {
        ctx.list_counters.borrow_mut().insert(counter_key, next);
        if let Some(id) = elem.attr_qualified(ns::XML, "id") {
            ctx.list_ids.borrow_mut().insert(id.to_string(), next);
        }
    }
    Ok(out)
}

/// A list item's composite marker label (`num-prefix`/`num-suffix`/
/// `display-levels`); `None` when the default `n.` label is faithful.
fn item_label(ctx: &Ctx, style_name: Option<&str>, depth: usize, chain: &[u64]) -> Option<String> {
    let levels = ctx.styles.list_levels(style_name?)?;
    let depth = depth.min(LIST_LEVELS - 1);
    let lvl = &levels[depth];
    if !lvl.marker.ordered() {
        return None;
    }
    crate::shared::numbering::composite_label(
        &lvl.pattern(depth),
        lvl.marker,
        *chain.last().unwrap_or(&1),
        |l| levels[l.min(LIST_LEVELS - 1)].marker,
        |l| chain.get(l).copied().unwrap_or_else(|| levels[l.min(LIST_LEVELS - 1)].start),
    )
}

/// ODF heading numbering: the `text:outline-style` level formats with the
/// heading's `restart-numbering`/`start-value`/`is-list-header` controls.
/// Advances the outline sequence; `None` for unnumbered headings.
fn heading_label(elem: &Element, level: u8, ctx: &Ctx) -> Option<String> {
    let levels = ctx.styles.outline_levels()?;
    let idx = (level.max(1) as usize - 1).min(LIST_LEVELS - 1);
    let lvl = &levels[idx];
    if !lvl.marker.ordered() {
        return None;
    }
    // An unnumbered heading displays no number and consumes none.
    if elem.attr(ns::TEXT, "is-list-header") == Some("true") {
        return None;
    }
    let (values, started) = &mut *ctx.heading_counters.borrow_mut();
    let restart = elem.attr(ns::TEXT, "restart-numbering") == Some("true");
    let explicit = elem.attr(ns::TEXT, "start-value").and_then(parse_start);
    let value = match explicit {
        Some(v) => v,
        None if started[idx] && !restart => values[idx].saturating_add(1),
        None => lvl.start,
    };
    values[idx] = value;
    started[idx] = true;
    for deeper in started.iter_mut().skip(idx + 1) {
        *deeper = false;
    }
    let label = crate::shared::numbering::composite_label(
        &lvl.pattern(idx),
        lvl.marker,
        value,
        |l| levels[l.min(LIST_LEVELS - 1)].marker,
        |l| {
            let l = l.min(LIST_LEVELS - 1);
            if started[l] { values[l] } else { levels[l].start }
        },
    );
    // Headings have no native numbering in the output, so the default label
    // is rendered too.
    Some(format!("{} ", label.unwrap_or_else(|| lvl.marker.label(value))))
}

/// markitai: how a run's text shows beyond emphasis: raised or lowered
/// (written in Unicode super/subscript forms where every character has
/// one), or hidden (left out).
#[derive(Clone, Copy, Default)]
struct RunMarks {
    script: Option<Script>,
    hidden: bool,
    /// markitai: the text size, in half-points.
    size: Size,
}

impl RunMarks {
    /// The marks of text styled `name` in `family`, over the enclosing ones:
    /// the nearest specification in the style's chain wins.
    fn under(self, ctx: &Ctx, family: &str, name: &str) -> RunMarks {
        RunMarks {
            script: ctx.styles.script(family, name).unwrap_or(self.script),
            hidden: ctx.styles.hidden(family, name).unwrap_or(self.hidden),
            size: ctx
                .styles
                .font_size(family, name)
                .map_or(self.size, |size| size.within(self.size)),
        }
    }
}

/// Inline content of a paragraph plus block attachments (text boxes) that
/// were anchored in it.
///
/// markitai: and the sizes of its visible text.
fn parse_inline_content(
    elem: &Element,
    ctx: &Ctx,
) -> Result<(Vec<Inline>, Vec<Block>, ParaSize), ConvertError> {
    let base = paragraph_base(elem, ctx)?;
    let unstyled = RunMarks { size: ctx.styles.base_size(), ..RunMarks::default() };
    let marks = match elem.attr(ns::TEXT, "style-name") {
        Some(name) => unstyled.under(ctx, "paragraph", name),
        None => unstyled,
    };
    let mut out = Vec::new();
    let mut boxes = Vec::new();
    // A text box read inside the paragraph gathers its own sizes.
    let outer = ctx.para_size.take();
    walk_inlines(elem, ctx, base, marks, &mut out, &mut boxes)?;
    let size = ctx.para_size.replace(outer);
    Ok((out, boxes, size))
}

/// The style a paragraph's runs cascade from. An unstyled paragraph still sits
/// on the family's default style.
fn paragraph_base(elem: &Element, ctx: &Ctx) -> Result<StyleDelta, ConvertError> {
    ctx.styles.delta("paragraph", elem.attr(ns::TEXT, "style-name").unwrap_or(""))
}

fn walk_inlines(
    elem: &Element,
    ctx: &Ctx,
    delta: StyleDelta,
    marks: RunMarks,
    out: &mut Vec<Inline>,
    boxes: &mut Vec<Block>,
) -> Result<(), ConvertError> {
    let style = delta.resolve();
    for node in &elem.children {
        match node {
            Node::Text(t) => {
                // markitai: hidden text shows nothing.
                if marks.hidden {
                    continue;
                }
                let text = collapse_ws(&clean_text(t));
                // markitai: raised or lowered text ("x₁", "library¹").
                let text = marks.script.and_then(|script| script.convert(&text)).unwrap_or(text);
                if !text.is_empty() {
                    // markitai: visible text and its size.
                    ctx.saw_text(marks.size, &text);
                    out.push(Inline::Text { text, style });
                }
            }
            Node::Elem(child) => {
                // markitai: a comment (`office:annotation`, its author, date
                // and text) is not document text; read as plain children it
                // ran into the sentence it annotates. Comments stay out, as
                // for Word documents.
                if child.is(ns::OFFICE, "annotation") || child.is(ns::OFFICE, "annotation-end") {
                    continue;
                }
                let in_text = child.ns.as_deref().is_some_and(|n| n == ns::TEXT);
                if in_text {
                    match child.local.as_str() {
                        "span" => {
                            let (merged, marks) = match child.attr(ns::TEXT, "style-name") {
                                Some(name) => (
                                    delta.merge(ctx.styles.delta("text", name)?),
                                    marks.under(ctx, "text", name),
                                ),
                                None => (delta, marks),
                            };
                            walk_inlines(child, ctx, merged, marks, out, boxes)?;
                            continue;
                        }
                        // markitai: the base text of a phonetic guide is the
                        // text; the guide (`text:ruby-text`, furigana or
                        // pinyin) only annotates it and ran into the words.
                        "ruby" => {
                            if let Some(base) = child.find(ns::TEXT, "ruby-base") {
                                walk_inlines(base, ctx, delta, marks, out, boxes)?;
                            }
                            continue;
                        }
                        "a" => {
                            let href = child.attr(ns::XLINK, "href").unwrap_or("");
                            let mut content = Vec::new();
                            walk_inlines(child, ctx, delta, marks, &mut content, boxes)?;
                            match classify_href(href) {
                                Some(target) if !inlines_are_empty(&content) => {
                                    out.push(Inline::Link { content, target })
                                }
                                _ => out.append(&mut content),
                            }
                            continue;
                        }
                        "s" => {
                            let n = child
                                .attr(ns::TEXT, "c")
                                .and_then(|v| v.parse::<usize>().ok())
                                .unwrap_or(1);
                            out.push(Inline::Text {
                                text: " ".repeat(n.min(20)),
                                style: Style::PLAIN,
                            });
                            continue;
                        }
                        "tab" => {
                            out.push(Inline::Text { text: " ".into(), style: Style::PLAIN });
                            continue;
                        }
                        "line-break" => {
                            out.push(Inline::LineBreak);
                            continue;
                        }
                        "bookmark" | "bookmark-start" => {
                            if let Some(name) = child.attr(ns::TEXT, "name") {
                                out.push(Inline::Anchor(name.to_string()));
                            }
                            continue;
                        }
                        "note" => {
                            let idx = ctx.notes.borrow().len();
                            let id = child
                                .attr(ns::TEXT, "id")
                                .map(String::from)
                                .unwrap_or_else(|| format!("odt{idx}"));
                            let kind = match child.attr(ns::TEXT, "note-class") {
                                Some("endnote") => NoteKind::Endnote,
                                _ => NoteKind::Footnote,
                            };
                            let note_blocks = match child.find(ns::TEXT, "note-body") {
                                Some(b) => {
                                    // markitai: see `Ctx::note_depth`.
                                    let _in_note = Deeper::enter(&ctx.note_depth);
                                    parse_container(b, ctx)?
                                }
                                None => Vec::new(),
                            };
                            ctx.notes.borrow_mut().push(Note {
                                id: id.clone(),
                                kind,
                                blocks: note_blocks,
                            });
                            out.push(Inline::NoteRef(id));
                            continue;
                        }
                        "annotation" | "tracked-changes" | "soft-page-break" => continue,
                        _ => {}
                    }
                }
                if child.is(ns::DRAW, "frame") {
                    walk_frame(child, ctx, out, boxes)?;
                    continue;
                }
                walk_inlines(child, ctx, delta, marks, out, boxes)?;
            }
        }
    }
    Ok(())
}

/// A draw:frame inline in text: an image (resolved into the asset store) or
/// a text box (attached as blocks after the paragraph).
pub(super) fn walk_frame(
    frame: &Element,
    ctx: &Ctx,
    out: &mut Vec<Inline>,
    boxes: &mut Vec<Block>,
) -> Result<(), ConvertError> {
    if let Some(text_box) = frame.find(ns::DRAW, "text-box") {
        boxes.extend(parse_container(text_box, ctx)?);
        return Ok(());
    }
    if let Some(object) = frame.find(ns::DRAW, "object")
        && let Some(tex) = formula_tex(ctx, object)?
    {
        out.push(Inline::Math(tex));
        return Ok(());
    }
    if let Some(object) = frame.find(ns::DRAW, "object")
        && let Some(chart) = chart_blocks(ctx, object)?
    {
        boxes.extend(chart);
        return Ok(());
    }
    let alt = frame
        .first_descendant(ns::SVG_COMPAT, "title")
        .map(|t| t.text())
        .or_else(|| frame.first_descendant(ns::SVG_COMPAT, "desc").map(|d| d.text()))
        .unwrap_or_default();
    let alt = clean_text(alt.trim());
    if let Some(image) = frame.first_descendant(ns::DRAW, "image") {
        let href = image.attr(ns::XLINK, "href").unwrap_or("");
        let source = load_image(ctx, href)?;
        if source.is_some() || !alt.is_empty() {
            out.push(Inline::Image { alt, source: source.unwrap_or(ImageSource::Unavailable) });
        }
        return Ok(());
    }
    if !alt.is_empty() {
        out.push(Inline::Image { alt, source: ImageSource::Unavailable });
    }
    Ok(())
}

/// The LaTeX of a `draw:object` holding a formula: MathML inline in the
/// object, or in the `content.xml` of the object directory it links to.
/// `None` for any other embedded object.
pub(super) fn formula_tex(ctx: &Ctx, object: &Element) -> Result<Option<String>, ConvertError> {
    if let Some(math) = object.find(ns::MATHML, "math") {
        return Ok(Some(mathml_to_tex(math)).filter(|t| !t.is_empty()));
    }
    let href = object.attr(ns::XLINK, "href").unwrap_or("");
    if href.is_empty() || crate::shared::uri::is_absolute_uri(href) {
        return Ok(None);
    }
    let target = match crate::package::path::resolve("content.xml", &format!("{href}/content.xml"))
    {
        Ok(t) => t,
        Err(e) => {
            log::warn!("skipping unresolvable object reference {href:?}: {e}");
            return Ok(None);
        }
    };
    let Some(tree) = ctx.pkg.borrow_mut().optional_xml_part(&target.path)? else {
        return Ok(None);
    };
    let Some(math) = tree.first_descendant(ns::MATHML, "math") else {
        return Ok(None);
    };
    Ok(Some(mathml_to_tex(math)).filter(|t| !t.is_empty()))
}

/// markitai: the data of a `draw:object` holding a chart. A chart document
/// keeps the values it plots in a table of its own (`chart:chart`'s
/// `table:table`: series names across the first row, categories down the
/// first column), which is returned after the chart's title, the first row
/// as the header, in place of the replacement picture. Upstream read only
/// the picture, usually a metafile with no text. `None` for any other
/// object; an unreadable object degrades to `None` like a formula.
pub(super) fn chart_blocks(
    ctx: &Ctx,
    object: &Element,
) -> Result<Option<Vec<Block>>, ConvertError> {
    let href = object.attr(ns::XLINK, "href").unwrap_or("");
    if href.is_empty() || crate::shared::uri::is_absolute_uri(href) {
        return Ok(None);
    }
    let Ok(target) = crate::package::path::resolve("content.xml", &format!("{href}/content.xml"))
    else {
        return Ok(None);
    };
    let Some(tree) = ctx.pkg.borrow_mut().optional_xml_part(&target.path)? else {
        return Ok(None);
    };
    let Some(chart) = tree.first_descendant(ns::ODF_CHART, "chart") else {
        return Ok(None);
    };
    let mut blocks = Vec::new();
    if let Some(title) = chart.find(ns::ODF_CHART, "title") {
        let text = title
            .find_all(ns::TEXT, "p")
            .map(|p| collapse_ws(&clean_text(&p.text())).trim().to_string())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ");
        if !text.is_empty() {
            blocks.push(Block::Paragraph(vec![Inline::Text { text, style: Style::PLAIN }]));
        }
    }
    if let Some(table) = chart.find(ns::TABLE, "table") {
        for block in parse_table(table, ctx)? {
            match block {
                Block::Table(mut table) => {
                    table.header_rows = table.header_rows.max(1).min(table.grid.len());
                    blocks.push(Block::Table(table));
                }
                other => blocks.push(other),
            }
        }
    }
    Ok(Some(blocks))
}

/// Failures degrade (log + `None`) per the unified policy; resource-limit
/// errors always propagate.
fn load_image(ctx: &Ctx, href: &str) -> Result<Option<ImageSource>, ConvertError> {
    if href.is_empty() {
        return Ok(None);
    }
    if crate::shared::uri::is_absolute_uri(href) {
        return Ok(Some(ImageSource::External(href.to_string())));
    }
    let target = match crate::package::path::resolve("content.xml", href) {
        Ok(t) => t,
        Err(e) => {
            log::warn!("skipping unresolvable image reference {href:?}: {e}");
            return Ok(None);
        }
    };
    match ctx.pkg.borrow_mut().optional_part(&target.path)? {
        Some(bytes) => {
            let media = media_type_for(&target.path);
            let id = ctx.assets.borrow_mut().add(media, target.path, &bytes)?;
            Ok(Some(ImageSource::Asset(id)))
        }
        None => {
            log::warn!("image part {} is missing", target.path);
            Ok(None)
        }
    }
}

/// ODF hrefs: external URLs, package-relative paths, or `#target` internal
/// references (with `|outline`-style suffixes on generated links).
fn classify_href(href: &str) -> Option<LinkTarget> {
    if href.is_empty() {
        return None;
    }
    if let Some(fragment) = href.strip_prefix('#') {
        let target = fragment.split('|').next().unwrap_or(fragment);
        if target.is_empty() {
            return None;
        }
        return Some(LinkTarget::Anchor(crate::package::path::decode_fragment(target)));
    }
    if crate::shared::uri::is_absolute_uri(href) {
        Some(LinkTarget::External(href.to_string()))
    } else {
        Some(LinkTarget::Relative(href.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn internal_href_fragments_are_percent_decoded() {
        assert_eq!(
            classify_href("#caf%C3%A9%20menu"),
            Some(LinkTarget::Anchor("café menu".into()))
        );
    }
}
