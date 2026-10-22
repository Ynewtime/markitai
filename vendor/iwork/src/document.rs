//! Document-level view over a package: components, media, text and styles.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use crate::create::OFFSET_SLOTS;
use crate::iwa::{self, ArchiveObject};
use crate::package::Package;
use crate::pb::{Message, Value};
use crate::style::{self, CreatedStyle, StyleDeletion, StyleKind, StyleUse, TextStyle};
use crate::table::{cell_type, CellValue};
use crate::text;
use crate::{Error, Refusal};

/// Which app wrote the document.
///
/// All three share the container, the IWA framing and most object types; the
/// kind only affects which document-level archives are present.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Pages,
    Numbers,
    Keynote,
    Unknown,
}

impl Kind {
    pub fn from_extension(path: &std::path::Path) -> Kind {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            // A template is the same package as a document, and the extension
            // is the only thing that says which app owns it — so the three
            // template extensions are read as their app's documents. A
            // `.template` renamed `.pages` opens in Pages; two fixtures in this
            // corpus are exactly that.
            Some("pages") | Some("template") => Kind::Pages,
            Some("numbers") | Some("nmbtemplate") => Kind::Numbers,
            Some("key") | Some("kth") => Kind::Keynote,
            _ => Kind::Unknown,
        }
    }

    /// The document extension for this app — what a template of it saves as.
    pub fn extension(self) -> Option<&'static str> {
        match self {
            Kind::Pages => Some("pages"),
            Kind::Numbers => Some("numbers"),
            Kind::Keynote => Some("key"),
            Kind::Unknown => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Pages => "Pages",
            Kind::Numbers => "Numbers",
            Kind::Keynote => "Keynote",
            Kind::Unknown => "unknown",
        }
    }
}

/// One entry of `TSP.PackageMetadata.components`.
#[derive(Debug, Clone)]
pub struct Component {
    pub identifier: u64,
    /// `preferred_name`, e.g. `"Document"` or `"Tables/DataList"`.
    pub preferred_name: String,
    /// `file_name` when the component has its own stream, e.g.
    /// `"Tables/DataList-874443-2"`. Empty when it shares the stream named by
    /// `preferred_name`.
    pub file_name: String,
    /// Number of `external_references` entries — a rough measure of how central
    /// the component is to the document.
    pub external_reference_count: usize,
}

impl Component {
    /// Package entry holding this component's objects.
    ///
    /// Verified against a 96-component Numbers document: every component
    /// resolves as `Index/{file_name or preferred_name}.iwa`.
    pub fn stream_name(&self) -> String {
        let stem = if self.file_name.is_empty() {
            &self.preferred_name
        } else {
            &self.file_name
        };
        format!("Index/{stem}.iwa")
    }
}

/// One entry of `TSP.PackageMetadata.datas` — the media registry.
#[derive(Debug, Clone)]
pub struct DataFile {
    /// Identifier drawables use to refer to this media. Not a package object id.
    pub identifier: u64,
    /// Name the file had when it was placed into the document.
    pub original_name: String,
    /// Name under `Data/`. Empty for assets that live in the app's own theme
    /// bundle and are therefore not copied into the package.
    pub stored_name: String,
    /// Path within the app's theme bundle, when this is a theme asset.
    pub asset_path: String,
    /// 20-byte digest iWork uses to deduplicate media.
    pub digest: Vec<u8>,
}

impl DataFile {
    /// Package entry holding the bytes, if they are stored in the package.
    pub fn entry_name(&self) -> Option<String> {
        (!self.stored_name.is_empty()).then(|| format!("Data/{}", self.stored_name))
    }
}

/// Which component owns what, and what each component already declares —
/// everything [`Document::undeclared_references`] needs, gathered in one pass.
#[derive(Default)]
struct ComponentIndex {
    /// Package entry name to the component stored there.
    by_stream: BTreeMap<String, u64>,
    /// Object identifier to the component it lives in.
    by_object: BTreeMap<u64, u64>,
    /// Component identifiers, which are also their root objects' identifiers.
    roots: BTreeSet<u64>,
    /// The two components every reader has open before it reads anything else:
    /// `Document` and `DocumentMetadata`. References to *their* roots are the
    /// only cross-component references iWork leaves undeclared, in all 23
    /// documents of this corpus and in every one of the three apps.
    always_loaded: BTreeSet<u64>,
    /// Objects each component declares in its `external_references`.
    declared: BTreeMap<u64, BTreeSet<u64>>,
}

/// The side-table archives a batch of cell writes is holding open.
///
/// A `TableDataList` is one object holding every string, format or control a
/// table uses, so interning a string decodes and re-encodes the whole list —
/// and a cell write does that up to three times. Over a batch that is the same
/// quadratic cost the tiles have: filling a table with 5000 distinct strings
/// decodes a 5000-entry list 15000 times. Here each list is decoded on first
/// use, every cell of the batch works on the decoded form, and it is written
/// back once, by [`ListCache::flush`].
///
/// Nothing is written until the flush, so a batch that refuses part way leaves
/// the lists as they were without the caller's snapshot even being consulted.
#[derive(Default)]
struct ListCache {
    open: std::collections::BTreeMap<u64, Message>,
    dirty: std::collections::BTreeSet<u64>,
    /// What each string list already holds, and the next key it can hand out.
    /// Interning without it means scanning the list for the text and again for
    /// the highest key, decoding every entry both times — per cell.
    strings: std::collections::BTreeMap<u64, StringIndex>,
    /// The format key borrowed for each slot, which depends only on the table
    /// as the batch found it — so it is looked up once rather than scanned out
    /// of every cell of the table for every cell written.
    donors: Vec<(crate::table::FormatSlot, Option<u32>)>,
}

/// What a `TableDataList` of strings holds, so that interning is a lookup.
#[derive(Default)]
struct StringIndex {
    by_text: std::collections::BTreeMap<Vec<u8>, u32>,
    /// The next key the list can hand out — a high-water mark, as the app keeps
    /// it, never reused downwards.
    next: u64,
}

impl ListCache {
    fn read(&mut self, document: &Document, list: u64) -> Result<&Message, Error> {
        if let std::collections::btree_map::Entry::Vacant(slot) = self.open.entry(list) {
            slot.insert(document.archive_of(list)?);
        }
        Ok(self.open.get(&list).expect("just inserted"))
    }

    fn write(&mut self, document: &Document, list: u64) -> Result<&mut Message, Error> {
        self.read(document, list)?;
        self.dirty.insert(list);
        Ok(self.open.get_mut(&list).expect("just inserted"))
    }

    /// The text-to-key index of one `STRING` list, read once.
    fn string_index(&mut self, document: &Document, list: u64) -> Result<&mut StringIndex, Error> {
        if !self.strings.contains_key(&list) {
            let archive = self.read(document, list)?;
            let mut by_text = std::collections::BTreeMap::new();
            let mut highest = 0u64;
            for value in archive.all(3) {
                let Value::Bytes(raw) = value else { continue };
                let Some(entry) = crate::pb::decode_nested(raw) else {
                    continue;
                };
                let Some(key) = entry.varint(1) else { continue };
                highest = highest.max(key);
                if let Some(text) = entry.bytes(3) {
                    by_text.insert(text.to_vec(), key as u32);
                }
            }
            // Above the mark *and* above every key present, because the mark is
            // only trustworthy while the app is the one maintaining it.
            let next = archive
                .varint(2)
                .unwrap_or(0)
                .max(highest.saturating_add(1))
                .max(1);
            self.strings.insert(list, StringIndex { by_text, next });
        }
        Ok(self.strings.get_mut(&list).expect("just inserted"))
    }

    /// Put every list this batch changed back into the document.
    fn flush(self, document: &mut Document) -> Result<(), Error> {
        for list in &self.dirty {
            let archive = self.open.get(list).expect("dirty lists are open");
            document.set_archive(*list, archive)?;
        }
        Ok(())
    }
}

/// One text storage, held for editing — see [`Document::text_mut`].
///
/// Forwarding and nothing more: every method calls the `Document` method of the
/// same name with the storage already named, so the remapping every edit does —
/// style runs, hyperlinks, list levels, anchored drawables, comment anchors —
/// is the same code and the same refusals behind both.
pub struct TextHandle<'a> {
    document: &'a mut Document,
    storage: u64,
}

impl TextHandle<'_> {
    /// The storage's identifier, for a caller that needs it elsewhere.
    pub fn identifier(&self) -> u64 {
        self.storage
    }

    /// The text as it is now.
    pub fn read(&self) -> Result<String, Error> {
        self.document.storage_text(self.storage)
    }

    /// Replace the whole text.
    pub fn set(&mut self, text: &str) -> Result<TextEdit, Error> {
        self.document.set_text(self.storage, text)
    }

    /// Insert at a character index, in UTF-16 code units.
    pub fn insert(&mut self, at: u64, text: &str) -> Result<TextEdit, Error> {
        self.document.insert_text(self.storage, at, text)
    }

    /// Delete a range, in UTF-16 code units.
    pub fn delete(&mut self, range: Range<u64>) -> Result<TextEdit, Error> {
        self.document.delete_text(self.storage, range)
    }

    /// Replace a range with other text.
    pub fn replace(&mut self, range: Range<u64>, text: &str) -> Result<TextEdit, Error> {
        self.document.replace_text(self.storage, range, text)
    }

    /// Add a paragraph at the end, with the newline that separates it from
    /// whatever is already there.
    pub fn append(&mut self, text: &str) -> Result<TextEdit, Error> {
        let end = crate::text::length(&self.read()?);
        match end {
            0 => self.insert(0, text),
            end => self.insert(end, &format!("\n{text}")),
        }
    }

    /// Point a range of text at a text style. See
    /// [`Document::apply_text_style`].
    pub fn style(&mut self, range: Range<u64>, style: u64) -> Result<(), Error> {
        self.document
            .apply_text_style(self.storage, range, style)
            .map(|_| ())
    }

    /// Where each paragraph begins and ends, in UTF-16 code units.
    pub fn paragraphs(&self) -> Result<Vec<Range<u64>>, Error> {
        self.document.paragraph_ranges(self.storage)
    }
}

/// One slide, held for writing — see [`Document::slide_mut`].
///
/// Forwarding and nothing more, with one thing of its own: it knows which
/// storage a placeholder lays out, so `title` and `body` are writes a caller
/// can make without going through the slide's object graph to find them.
pub struct SlideHandle<'a> {
    document: &'a mut Document,
    slide: u64,
}

impl SlideHandle<'_> {
    /// The `KN.SlideArchive` identifier.
    pub fn identifier(&self) -> u64 {
        self.slide
    }

    /// The slide as it is now.
    pub fn read(&self) -> crate::keynote::Slide {
        self.document
            .slides()
            .into_iter()
            .find(|slide| slide.identifier == self.slide)
            .expect("the handle resolved this slide")
    }

    /// The storage a placeholder lays out, or why there is none to write.
    fn placeholder(&self, which: &str) -> Result<u64, Error> {
        let slide = self.read();
        let placeholder = match which {
            "title" => slide.title.as_ref(),
            _ => slide.body.as_ref(),
        };
        placeholder
            .and_then(|placeholder| placeholder.storage)
            .ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!(
                        "slide {} is built on the layout {:?}, which gives it no {which} — a \
                     placeholder comes from the layout, and a text box put here instead would \
                     be a box and not a {which}",
                        slide.number.unwrap_or(slide.index + 1),
                        slide.layout_name
                    ),
                )
            })
    }

    /// Write the title placeholder's text.
    pub fn title(&mut self, text: &str) -> Result<TextEdit, Error> {
        let storage = self.placeholder("title")?;
        self.document.set_text(storage, text)
    }

    /// Write the body placeholder's text.
    pub fn body(&mut self, text: &str) -> Result<TextEdit, Error> {
        let storage = self.placeholder("body")?;
        self.document.set_text(storage, text)
    }

    /// Write the presenter notes. See [`Document::set_presenter_notes`].
    pub fn notes(&mut self, text: &str) -> Result<TextEdit, Error> {
        self.document.set_presenter_notes(self.slide, text)
    }

    /// Leave the slide out of the show, or put it back.
    pub fn skip(&mut self, skipped: bool) -> Result<bool, Error> {
        self.document.set_slide_skipped(self.slide, skipped)
    }

    /// Move the slide to a position in the deck, counting from 0.
    pub fn move_to(&mut self, index: usize) -> Result<usize, Error> {
        self.document.move_slide(self.slide, index)
    }

    /// Copy the slide, straight after itself.
    pub fn duplicate(&mut self) -> Result<crate::keynote::SlideCopy, Error> {
        self.document.duplicate_slide(self.slide)
    }

    /// Give the slide a transition — by the app's name for the effect or by the
    /// identifier on the wire, and `"none"` takes it away.
    ///
    /// The duration and delay the app writes unless a caller says otherwise;
    /// everything subtler is [`Document::set_transition`] and a
    /// [`crate::keynote::TransitionEdit`] of one's own.
    /// The duration and delay are the app's own unless
    /// [`SlideHandle::transition_with`] says otherwise — which is the call to
    /// use for those, because `(Some(1.5), None)` is two `Option<f64>` in a row
    /// and nothing but the reader's memory says which is which.
    pub fn transition(&mut self, effect: &str) -> Result<crate::keynote::Transition, Error> {
        self.transition_with(&crate::keynote::TransitionEdit {
            effect: effect.to_string(),
            ..Default::default()
        })
    }

    /// A transition with every field named — duration, delay, the automatic
    /// flag and the direction the app's own writer never emits.
    pub fn transition_with(
        &mut self,
        edit: &crate::keynote::TransitionEdit,
    ) -> Result<crate::keynote::Transition, Error> {
        self.document.set_transition(self.slide, edit)
    }

    /// Animate one of the slide's drawables on, or off.
    ///
    /// The effect is the kind's default — see [`crate::keynote::BuildEdit`] for
    /// the rest, and note that a *build* effect is not a transition effect.
    pub fn add_build(
        &mut self,
        drawable: u64,
        kind: crate::keynote::BuildKind,
    ) -> Result<u64, Error> {
        self.document.add_build(
            self.slide,
            drawable,
            &crate::keynote::BuildEdit {
                kind,
                ..Default::default()
            },
        )
    }

    /// Put a text box on the slide, in a frame given in points.
    ///
    /// The frame is named rather than positional: `(100.0, 120.0), (600.0,
    /// 120.0)` is two tuples of the same type whose order is the whole meaning,
    /// and `Frame { x, y, width, height }` cannot be given in the wrong order.
    pub fn add_text_box(
        &mut self,
        text: &str,
        frame: crate::drawable::Frame,
    ) -> Result<u64, Error> {
        let slide = self.slide.to_string();
        self.document.add_text_box(
            &slide,
            text,
            (frame.x, frame.y),
            (frame.width, frame.height),
        )
    }

    /// Put a picture on the slide, in a frame — PNG or JPEG bytes.
    pub fn add_image(
        &mut self,
        bytes: &[u8],
        name: &str,
        frame: crate::drawable::Frame,
    ) -> Result<u64, Error> {
        let slide = self.slide.to_string();
        self.document.add_image(
            &slide,
            bytes,
            name,
            (frame.x, frame.y),
            Some((frame.width, frame.height)),
        )
    }

    /// The same, drawn at the picture's own pixel size.
    pub fn add_image_at(&mut self, bytes: &[u8], name: &str, x: f32, y: f32) -> Result<u64, Error> {
        let slide = self.slide.to_string();
        self.document.add_image(&slide, bytes, name, (x, y), None)
    }

    /// Put a table on the slide.
    pub fn add_table(&mut self, name: &str, rows: usize, columns: usize) -> Result<u64, Error> {
        let slide = self.slide.to_string();
        self.document.add_table(&slide, name, rows, columns)
    }
}

/// One table, held for writing — see [`Document::table_mut`].
///
/// A forwarding layer and nothing more: every method here calls the `Document`
/// method of the same name with the table already resolved, so there is one
/// implementation of each write and one set of refusals behind both.
pub struct TableHandle<'a> {
    document: &'a mut Document,
    /// The table's identifier as text, which is what the write path takes and
    /// what makes it unambiguous whatever the caller named.
    table: String,
}

impl TableHandle<'_> {
    /// The table as it is now — the snapshot to read many cells from.
    pub fn read(&self) -> crate::table::Table {
        self.document
            .table(&self.table)
            .expect("the handle resolved this table")
    }

    /// The table's identifier, for a caller that wants to keep it.
    pub fn identifier(&self) -> u64 {
        self.table.parse().expect("resolved from an identifier")
    }

    /// One cell's value. Reading many is [`TableHandle::read`] — this decodes
    /// the table each time it is called.
    pub fn value(&self, cell: impl Into<crate::table::CellRef>) -> Result<CellValue, Error> {
        let (row, column) = cell.into().resolve()?;
        Ok(self.read().value(row, column))
    }

    /// Write one cell, and hand back what it held.
    pub fn set(
        &mut self,
        cell: impl Into<crate::table::CellRef>,
        value: impl Into<CellValue>,
    ) -> Result<CellValue, Error> {
        let (row, column) = cell.into().resolve()?;
        self.document
            .set_cell(&self.table, row, column, value.into())
    }

    /// Write a rectangle of values, the first landing at `at`.
    ///
    /// The batch write: one pass over each tile, each list and each header
    /// bucket however many cells are given, and all or nothing if one is
    /// refused. See [`Document::set_block`].
    pub fn set_block<V: Into<CellValue> + Clone>(
        &mut self,
        at: impl Into<crate::table::CellRef>,
        rows: &[Vec<V>],
    ) -> Result<usize, Error> {
        let at = at.into().resolve()?;
        let rows: Vec<Vec<CellValue>> = rows
            .iter()
            .map(|row| row.iter().cloned().map(Into::into).collect())
            .collect();
        self.document.set_block(&self.table, at, &rows)
    }

    /// Give a cell a formula, with the value it shows until the app
    /// recalculates. See [`Document::set_formula`].
    pub fn formula(
        &mut self,
        cell: impl Into<crate::table::CellRef>,
        formula: &str,
        value: impl Into<CellValue>,
    ) -> Result<(), Error> {
        let (row, column) = cell.into().resolve()?;
        self.document
            .set_formula(&self.table, row, column, formula, value.into())
    }

    /// Write an amount as money, in a currency named by its code.
    ///
    /// Two decimals unless `decimals` says otherwise — see
    /// [`Document::set_currency`], which this calls.
    pub fn currency(
        &mut self,
        cell: impl Into<crate::table::CellRef>,
        amount: impl Into<CellValue>,
        code: &str,
    ) -> Result<(), Error> {
        self.currency_with(cell, amount, code, Some(2))
    }

    /// The same, saying how many decimal places the amount shows.
    pub fn currency_with(
        &mut self,
        cell: impl Into<crate::table::CellRef>,
        amount: impl Into<CellValue>,
        code: &str,
        decimals: Option<u8>,
    ) -> Result<(), Error> {
        let (row, column) = cell.into().resolve()?;
        let amount = match amount.into() {
            CellValue::Number(decimal) | CellValue::Currency(decimal) => decimal,
            other => {
                return Err(Error::refused(
                    Refusal::UnwritableValue,
                    format!("{}: {} is not an amount", self.table, other.kind()),
                ))
            }
        };
        self.document
            .set_currency(&self.table, [(row, column, amount)], code, decimals)?;
        Ok(())
    }

    /// Give cells a data format — one cell, or a range of them at once.
    ///
    /// See [`Document::set_format`], which this calls with the range's cells.
    pub fn format(
        &mut self,
        cells: impl Into<crate::table::CellRange>,
        format: &crate::table::Format,
    ) -> Result<usize, Error> {
        let cells = cells.into().cells()?;
        self.document.set_format(&self.table, cells, format)
    }

    /// What the table sorts by, replacing whatever it sorted by before.
    ///
    /// The rules are stored, not applied: see [`Document::set_sort_rules`].
    pub fn sort_by(&mut self, rules: &[crate::table::SortRule]) -> Result<(), Error> {
        self.document.set_sort_rules(&self.table, rules)
    }

    /// A column's width in points, or `None` for the table's default.
    ///
    /// The axis is in the name rather than in the reader's memory: `width(0,
    /// …)` could be a column's width or a row's, and the compiler cannot tell
    /// the difference between two `usize`s.
    pub fn column_width(&mut self, column: usize, points: Option<f32>) -> Result<(), Error> {
        self.document.set_column_width(&self.table, column, points)
    }

    /// A row's height in points, or `None` for the table's default.
    pub fn row_height(&mut self, row: usize, points: Option<f32>) -> Result<(), Error> {
        self.document.set_row_height(&self.table, row, points)
    }

    pub fn insert_row(&mut self, at: usize) -> Result<(), Error> {
        self.document.insert_row(&self.table, at)
    }

    pub fn delete_row(&mut self, at: usize) -> Result<(), Error> {
        self.document.delete_row(&self.table, at)
    }

    pub fn insert_column(&mut self, at: usize) -> Result<(), Error> {
        self.document.insert_column(&self.table, at)
    }

    pub fn delete_column(&mut self, at: usize) -> Result<(), Error> {
        self.document.delete_column(&self.table, at)
    }

    /// Merge a rectangle of cells into one — `merge("B2:D2")`.
    ///
    /// A range and not a corner with two counts: which of `(1, 3)` is the rows
    /// and which the columns is something a caller has to remember and the
    /// compiler cannot check, and `"B2:D2"` says it in the notation the app
    /// itself uses.
    pub fn merge(&mut self, cells: impl Into<crate::table::CellRange>) -> Result<(), Error> {
        let rectangle = cells.into().resolve()?;
        self.document.merge_cells(
            &self.table,
            rectangle.top_left.0,
            rectangle.top_left.1,
            rectangle.rows(),
            rectangle.columns(),
        )
    }

    pub fn unmerge(&mut self, cell: impl Into<crate::table::CellRef>) -> Result<(), Error> {
        let (row, column) = cell.into().resolve()?;
        self.document.unmerge_cells(&self.table, row, column)
    }
}

/// What a write to one table needs that does not change from cell to cell: the
/// tile map and the side tables every row of it shares.
struct TableSite {
    /// How many rows one tile covers.
    tile_size: usize,
    /// Tile index — `row / tile_size` — to the object holding it.
    tiles: std::collections::BTreeMap<usize, u64>,
    /// `TableDataList` for interned strings, for data formats, and for the
    /// control specs a deleted cell has to give up its reference to.
    strings: Option<u64>,
    formats: Option<u64>,
    controls: Option<u64>,
    /// The `FORMULA` list, which a filled formula takes another reference in.
    formulas: Option<u64>,
    /// `TST.HeaderStorageBucket`s carrying the per-row and per-column cell
    /// counts, which change when a cell appears or disappears.
    row_bucket: Vec<u64>,
    column_bucket: Option<u64>,
}

/// One row of a batch: the row's index in the table, and its cells by column.
type TileRowCells = (usize, Vec<(usize, CellValue)>);

/// One row of a decoded tile: its cells, and where its `TileRowInfo` sits.
struct TileRow {
    /// Index of the `TileRowInfo` **among the tile's fields** — or, when
    /// `fresh`, the position a new one is inserted at.
    position: usize,
    /// The row has no `TileRowInfo` yet, because a row with no cells has none.
    fresh: bool,
    /// The row's index within its tile.
    index: usize,
    wide: bool,
    /// Every cell of the row, one slot per offset entry.
    records: Vec<Option<Vec<u8>>>,
}

/// Where each row of a decoded tile sits among its fields.
///
/// A `TST.Tile` carries its rows as repeated field 5, keyed by a
/// `tile_row_index` *inside* the message, so finding one row means decoding
/// every field until it turns up. Doing that per row is quadratic in the tile —
/// 256 rows of a full tile decode 32 000 row messages between them — so a batch
/// reads the keys once and looks rows up by index afterwards.
struct TileIndex {
    /// `tile_row_index` to the field's position in the archive.
    positions: std::collections::BTreeMap<usize, usize>,
    /// How many columns a row of this tile names, which is what a new row is
    /// given: Numbers pads the array to 255 entries whatever the table's width,
    /// and a reader steps through the padding.
    slots: usize,
    wide: bool,
}

impl TileIndex {
    fn of(archive: &Message) -> TileIndex {
        let mut positions = std::collections::BTreeMap::new();
        let mut slots = 0;
        for (at, field) in archive.fields.iter().enumerate() {
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            if field.number != 5 {
                continue;
            }
            let Some(info) = crate::pb::decode_nested(raw) else {
                continue;
            };
            if let Some(index) = info.varint(1) {
                positions.insert(index as usize, at);
            }
            slots = slots.max(info.bytes(7).map(|offsets| offsets.len() / 2).unwrap_or(0));
        }
        TileIndex {
            positions,
            slots: if slots == 0 { OFFSET_SLOTS } else { slots },
            wide: archive.varint(8).unwrap_or(0) != 0,
        }
    }

    /// Slice one row out of the tile.
    ///
    /// A row with no stored cells has no `TileRowInfo` at all, so this cannot
    /// fail to find one: it says where a new one would go instead, and the
    /// first cell written into the row brings it into being.
    fn row(&self, archive: &Message, tile_row: usize, where_: &str) -> Result<TileRow, Error> {
        let Some(&position) = self.positions.get(&tile_row) else {
            // Where the new entry goes: in ascending `tile_row_index` order,
            // which is how every tile in the corpus keeps them.
            let position = self
                .positions
                .range(tile_row + 1..)
                .next()
                .map(|(_, &at)| at)
                .or_else(|| self.positions.values().next_back().map(|&last| last + 1))
                .or_else(|| archive.fields.iter().position(|field| field.number > 5))
                .unwrap_or(archive.fields.len());
            return Ok(TileRow {
                position,
                fresh: true,
                index: tile_row,
                wide: self.wide,
                records: vec![None; self.slots],
            });
        };
        TileRow::at(archive, position, tile_row, self.wide, where_)
    }

    /// Record that a row was brought into being at `position`, which moves
    /// every field after it along one.
    fn inserted(&mut self, tile_row: usize, position: usize) {
        for at in self.positions.values_mut() {
            if *at >= position {
                *at += 1;
            }
        }
        self.positions.insert(tile_row, position);
    }
}

impl TileRow {
    /// Read the row whose `TileRowInfo` is field `position` of the tile.
    fn at(
        archive: &Message,
        position: usize,
        tile_row: usize,
        wide: bool,
        where_: &str,
    ) -> Result<TileRow, Error> {
        let Value::Bytes(raw) = &archive.fields[position].value else {
            unreachable!("the position was found by matching on Bytes")
        };
        let info = crate::pb::decode_nested(raw)
            .ok_or_else(|| Error::Format(format!("{where_}: row storage does not decode")))?;
        let (Some(buffer), Some(offsets)) = (info.bytes(6), info.bytes(7)) else {
            return Err(Error::Format(format!(
                "{where_}: row {tile_row} of this tile has no version-5 cell storage"
            )));
        };
        let wide = wide || info.varint(8).unwrap_or(0) != 0;
        let mut records: Vec<Option<Vec<u8>>> = vec![None; offsets.len() / 2];
        for (at, bytes) in crate::table::row_cells(buffer, offsets, wide)
            .into_iter()
            .flatten()
        {
            records[at] = Some(bytes.to_vec());
        }
        Ok(TileRow {
            position,
            fresh: false,
            index: tile_row,
            wide,
            records,
        })
    }

    /// Lay the row back into its tile, bringing it into being if it is new.
    ///
    /// The one fallible step is the offset arithmetic: a row grown past what a
    /// 16-bit offset can address is refused rather than truncated.
    fn put(&self, archive: &mut Message, where_: &str) -> Result<(), Error> {
        let (buffer, offsets) = crate::table::encode_row(&self.records, self.wide)
            .map_err(|e| Error::Format(format!("{where_}: {e}")))?;
        let cells = self.records.iter().filter(|r| r.is_some()).count() as u64;
        if self.fresh {
            // The row is brought into being in the shape `Document::new` writes
            // and all three apps open: the pre-BNC pair (3, 4) present and
            // empty, because they are `required` in the schema and meaningless
            // in this storage version, and `numrows` counting one more row.
            let mut info = Message::default();
            info.set_in_order(1, Value::Varint(self.index as u64));
            info.set_in_order(2, Value::Varint(cells));
            info.set_in_order(3, Value::Bytes(Vec::new()));
            info.set_in_order(4, Value::Bytes(offsets.clone()));
            info.set_in_order(5, Value::Varint(5));
            info.set_in_order(6, Value::Bytes(buffer));
            info.set_in_order(7, Value::Bytes(offsets));
            archive.fields.insert(
                self.position,
                crate::pb::Field {
                    number: 5,
                    value: Value::Bytes(info.encode()),
                },
            );
            let rows = archive.varint(4).unwrap_or(0) + 1;
            archive.set(4, Value::Varint(rows));
            return Ok(());
        }
        let Value::Bytes(raw) = &archive.fields[self.position].value else {
            unreachable!("the position was found by matching on Bytes")
        };
        let mut info = crate::pb::decode_nested(raw)
            .ok_or_else(|| Error::Format(format!("{where_}: row storage does not decode")))?;
        info.set(2, Value::Varint(cells));
        info.set(6, Value::Bytes(buffer));
        info.set(7, Value::Bytes(offsets));
        archive.fields[self.position].value = Value::Bytes(info.encode());
        Ok(())
    }
}

/// Everything [`Document::set_cell`] has to find before it can write a byte.
///
/// A cell is addressed by row and column, but it is *stored* as a slice of one
/// tile's row buffer, and everything it says about itself is a key into a side
/// table that lives in another stream again. Gathering the addresses first
/// keeps the write itself short enough to read.
struct CellSite {
    /// `TST.Tile` holding the row.
    tile: u64,
    /// Index of the row's `TileRowInfo` **among the tile's fields**, not its
    /// row number: the field is what gets replaced — or, when `fresh_row`, the
    /// position the new one is inserted at.
    tile_row: usize,
    /// The row's index *within its tile*, which is what a `TileRowInfo` stores.
    tile_row_index: usize,
    /// Whether the row has no `TileRowInfo` yet, so that storing a record means
    /// bringing the row into being rather than rewriting it.
    fresh_row: bool,
    /// Whether this row's offsets count groups of four bytes.
    wide: bool,
    row: usize,
    column: usize,
    /// Every cell of the row, one slot per offset entry.
    records: Vec<Option<Vec<u8>>>,
    /// The target cell's bytes, if it has any.
    record: Option<Vec<u8>>,
    /// The `FORMULA` list, which a filled formula takes another reference in.
    formulas: Option<u64>,
    /// `TST.HeaderStorageBucket`s carrying the per-row and per-column cell
    /// counts, which change when a cell appears or disappears.
    row_bucket: Vec<u64>,
    column_bucket: Option<u64>,
}

/// One reference-count change to a `TableDataList`, queued during planning so
/// that every fallible lookup is behind us before any of them touches a byte.
enum ListMutation {
    /// Take a reference to `text`, interning it if it is not already there. The
    /// key is decided while planning so the rewritten record can name it; `fresh`
    /// says whether applying it appends a new entry or bumps an existing one.
    Intern {
        list: u64,
        text: String,
        key: u32,
        fresh: bool,
    },
    /// Take one more reference to an entry that already exists.
    Retain { list: u64, key: u32 },
    /// Give up one reference, dropping the entry at zero.
    Release { list: u64, key: u32 },
}

impl ListMutation {
    fn list(&self) -> u64 {
        match self {
            ListMutation::Intern { list, .. }
            | ListMutation::Retain { list, .. }
            | ListMutation::Release { list, .. } => *list,
        }
    }
}

/// A planned cell write: the record to store — or `None` to delete it — and the
/// reference-count changes that go with it.
struct CellWrite {
    record: Option<crate::table::CellRecord>,
    mutations: Vec<ListMutation>,
}

/// A row re-encoded and ready to write, held apart from the tile so the encoding
/// (which can fail) happens before the tile is touched (which does not).
struct StagedRow {
    cells: u64,
    buffer: Vec<u8>,
    offsets: Vec<u8>,
    /// Columns that gained a stored cell, and columns that lost one — the
    /// header buckets count cells, so they move by these and not by the number
    /// of values written.
    appeared: Vec<usize>,
    disappeared: Vec<usize>,
}

/// A planned row insert: the objects it rewrites, each with its new archive, so
/// the whole write is proven before any of it is applied.
/// Reproduce a merge's node array, for the test that compares it with the app's.
#[doc(hidden)]
pub fn merge_range_node_for_test(
    table: &crate::table::Table,
    row: usize,
    column: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<u8>, Error> {
    merge_range_node(table, row, column, rows, columns)
}

/// The node array a merge is stored as: an absolute range, wrapped in `SUM`.
///
/// Built through the formula parser, so the colon tract's two encodings — plain
/// `int32` in the tract's lists, against the zigzag of a single cell — are the
/// ones [`crate::formula_parse`] was measured into. What the parser does not
/// write is the cross-table reference the app puts on a merge's range: field 28
/// naming **this table's own** `base_owner_uid`, which is what the four merges
/// of `numbers-formats.numbers` carry.
fn merge_range_node(
    table: &crate::table::Table,
    row: usize,
    column: usize,
    rows: usize,
    columns: usize,
) -> Result<Vec<u8>, Error> {
    let cell = |row: usize, column: usize| {
        format!(
            "${}${}",
            crate::formula::column_letters(column as i64),
            row + 1
        )
    };
    let text = format!(
        "=SUM({}:{})",
        cell(row, column),
        cell(row + rows - 1, column + columns - 1)
    );
    let ast = crate::formula_parse::parse(&text, (0, 0))
        .map_err(|e| Error::Format(format!("{}: {text}: {e}", table.name)))?;
    let mut nodes = ast.nodes;
    let mut tract = nodes
        .first()
        .ok_or_else(|| Error::Format(format!("{}: {text} parsed to nothing", table.name)))?
        .message()
        .clone();
    let mut uuid = Message::default();
    uuid.set_in_order(2, Value::Varint(table.base_uid.lower & 0xffff_ffff));
    uuid.set_in_order(3, Value::Varint(table.base_uid.lower >> 32));
    uuid.set_in_order(4, Value::Varint(table.base_uid.upper & 0xffff_ffff));
    uuid.set_in_order(5, Value::Varint(table.base_uid.upper >> 32));
    let mut extra = Message::default();
    extra.set_in_order(1, Value::Bytes(uuid.encode()));
    tract.set_in_order(28, Value::Bytes(extra.encode()));
    nodes[0] = crate::formula::Node::decode(tract)
        .ok_or_else(|| Error::Format(format!("{}: the range node is not a node", table.name)))?;
    Ok(crate::formula::Ast { nodes }.encode())
}

/// Every object a row or column delete rewrites, and its new form.
struct DeletePlan {
    touched: Vec<u64>,
    model: u64,
    model_archive: Message,
    tiles: Vec<(u64, Message)>,
    /// The row bucket holding the deleted row, or the single column bucket.
    bucket: u64,
    bucket_archive: Message,
    uid_map: u64,
    uid_map_archive: Message,
}

/// Every object a column insert rewrites, and its new form.
struct ColumnInsertPlan {
    touched: Vec<u64>,
    model: u64,
    model_archive: Message,
    /// One entry per tile: the tile, and its rows with a gap opened.
    tiles: Vec<(u64, Message)>,
    column_bucket: u64,
    bucket_archive: Message,
    uid_map: u64,
    uid_map_archive: Message,
}

struct RowInsertPlan {
    touched: Vec<u64>,
    model: u64,
    model_archive: Message,
    /// One entry per tile the insert rewrites — more than one when the shift
    /// pushes a row over a tile boundary.
    tiles: Vec<(u64, Message)>,
    row_bucket: u64,
    bucket_archive: Message,
    uid_map: u64,
    uid_map_archive: Message,
}

/// The node kinds that name cells — the ones a row insert has to reason about.
fn is_reference_node(kind: u32) -> bool {
    use crate::formula::node;
    matches!(
        kind,
        node::LOCAL_CELL_REFERENCE
            | node::CROSS_TABLE_CELL_REFERENCE
            | node::COLON
            | node::REFERENCE_ERROR
            | node::CELL_REFERENCE
            | node::COLON_WITH_UIDS
            | node::REFERENCE_ERROR_WITH_UIDS
            | node::UID_REFERENCE
            | node::LINKED_CELL_REF
            | node::LINKED_COLUMN_REF
            | node::LINKED_ROW_REF
            | node::CATEGORY_REF
            | node::COLON_TRACT
            | node::VIEW_TRACT_REF
            | node::INTERSECTION
            | node::SPILL_RANGE
    )
}

/// Of those, the ones that can carry a table id and so reach another table.
fn is_cross_table_node(kind: u32) -> bool {
    use crate::formula::node;
    matches!(
        kind,
        node::CROSS_TABLE_CELL_REFERENCE
            | node::COLON_WITH_UIDS
            | node::REFERENCE_ERROR_WITH_UIDS
            | node::UID_REFERENCE
            | node::LINKED_CELL_REF
            | node::LINKED_COLUMN_REF
            | node::LINKED_ROW_REF
            | node::CATEGORY_REF
            | node::COLON_TRACT
            | node::VIEW_TRACT_REF
    )
}

/// Encode a `TSP.UUID` as `{1: lower, 2: upper}`.
fn encode_uuid(uuid: crate::table::Uuid) -> Vec<u8> {
    let mut message = Message::default();
    message.set_in_order(1, Value::Varint(uuid.lower));
    message.set_in_order(2, Value::Varint(uuid.upper));
    message.encode()
}

/// Mint a row UUID unique among the table's existing row UUIDs.
///
/// Row UUIDs collide across tables, so uniqueness is only ever needed within the
/// one table; the value is never verified against the app, which exposes no row
/// UUID, so any high-entropy value the table does not already hold will do. It is
/// derived deterministically from the table's identity and the insertion index
/// so a given insert reproduces the same document, and bumped until it does not
/// collide.
fn mint_row_uuid(
    existing: &BTreeSet<crate::table::Uuid>,
    table: &crate::table::Table,
    at: usize,
) -> crate::table::Uuid {
    // splitmix64 seeded from the table id and the insertion point.
    let mut state = 0x9E37_79B9_7F4A_7C15u64
        ^ (at as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
        ^ table.identifier.wrapping_mul(0x2545_F491_4F6C_DD1D);
    for byte in table.table_id.bytes() {
        state = state
            .wrapping_mul(0x0000_0100_0000_01B3)
            .wrapping_add(u64::from(byte));
    }
    let mut next = move || {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    };
    loop {
        let candidate = crate::table::Uuid {
            lower: next(),
            upper: next(),
        };
        if candidate != crate::table::Uuid::default() && !existing.contains(&candidate) {
            return candidate;
        }
    }
}

/// What [`Document::new`] gives a new spreadsheet: the sheet, the table and the
/// size Numbers' own blank document has.
const DEFAULT_SHEET: &str = "Sheet 1";
const DEFAULT_TABLE: &str = "Table 1";
const DEFAULT_ROWS: usize = 22;
const DEFAULT_COLUMNS: usize = 7;

/// The slide a new deck starts with, in points — 16:9, which is what Keynote
/// offers first.
const DEFAULT_SLIDE_SIZE: (f32, f32) = (1920.0, 1080.0);

/// The sizes a table this crate makes may have.
fn check_table_size(rows: usize, columns: usize) -> Result<(), Error> {
    if rows == 0 || columns == 0 {
        return Err(Error::Format(
            "a table needs at least one row and one column".into(),
        ));
    }
    if columns > MAX_COLUMNS {
        return Err(Error::Format(format!(
            "{columns} columns: a new table is capped at {MAX_COLUMNS}, the number of cell \
             offsets Numbers writes into a row"
        )));
    }
    Ok(())
}

/// How wide a table [`Document::new_spreadsheet`] will make.
///
/// A row carries 255 cell offsets whatever the table's width — see
/// [`crate::create`] — and a table wider than that would need an offset array
/// no document in this corpus has.
const MAX_COLUMNS: usize = 255;

/// Every slot a cell record can carry a format key in.
/// Byte 7 of a cell record, the high half of `extras`: `0x08` on every
/// currency cell in the corpus and `0x00` on everything else. What it means is
/// not established here; what is established is that the app sets it when a
/// cell becomes money and this crate has to as well.
const CURRENCY_BYTE_SEVEN: u16 = 0x0800;

const ALL_SLOTS: [crate::table::FormatSlot; 6] = [
    crate::table::FormatSlot::Number,
    crate::table::FormatSlot::Currency,
    crate::table::FormatSlot::Date,
    crate::table::FormatSlot::Duration,
    crate::table::FormatSlot::Text,
    crate::table::FormatSlot::Boolean,
];

/// `TSK.FormatStructArchive.format_type` for the automatic format.
///
/// The one format that suits every slot: a text cell points at it and Numbers
/// calls the cell text, a number cell points at it and Numbers calls the cell
/// automatic. See [`crate::table::CellFormat::of`].
const AUTOMATIC_FORMAT: u64 = 260;

/// `TSWP.HyperlinkFieldArchive` — `{1: {1: uuid}, 2: url}`.
pub const TYPE_HYPERLINK_FIELD: u32 = 2032;

/// One attribute table of one storage, as [`Document::storages`] found it.
#[derive(Debug, Clone)]
pub struct StorageTable {
    pub field: u32,
    pub name: &'static str,
    pub anchoring: crate::text::Anchoring,
    pub entries: usize,
}

/// A `TSWP.StorageArchive` and everything anchored into it.
#[derive(Debug, Clone)]
pub struct StorageInventory {
    pub identifier: u64,
    pub stream: String,
    /// `StorageArchive.kind`: 0 body, 1 header, 2 footnote, 3 text box, 4 note,
    /// 5 cell, 6 unclassified, 7 table of contents, 8 undefined. The default is
    /// 3, so a storage with no field 1 is a text box.
    pub kind: u64,
    /// Length of the text in UTF-16 code units.
    pub length: u64,
    pub paragraphs: usize,
    pub tables: Vec<StorageTable>,
    /// A length-delimited field that is not a table this crate knows — the
    /// reason an edit on this storage would be refused.
    pub unknown_field: Option<u32>,
}

impl StorageInventory {
    pub fn kind_name(&self) -> &'static str {
        match self.kind {
            0 => "body",
            1 => "header",
            2 => "footnote",
            3 => "text box",
            4 => "note",
            5 => "cell",
            6 => "unclassified",
            7 => "table of contents",
            _ => "undefined",
        }
    }
}

/// A hyperlink or other smart field, and the text it covers.
#[derive(Debug, Clone)]
pub struct SmartField {
    pub storage: u64,
    /// The field archive — 2031 placeholder, 2032 hyperlink, 2034 date and
    /// time, 2036 mail merge, and the rest of 2031–2042.
    pub object: u64,
    pub message_type: u32,
    /// Characters the field covers, in UTF-16 code units.
    pub range: Range<u64>,
    /// Field 2 of the archive. On a hyperlink (2032) it is the **URL**; on a
    /// placeholder (2031) there is none; on a mail-merge field (2036) it is the
    /// contacts property the field stands for, which is why this is not called
    /// `url`.
    pub payload: Option<String>,
    pub text: String,
}

/// One paragraph's list state — see [`Document::list_paragraphs`].
#[derive(Debug, Clone)]
pub struct ListParagraph {
    pub range: Range<u64>,
    /// Indent depth, counted from 0. Stored as `first` in the paragraph-data
    /// table.
    pub level: u64,
    /// `TSWP.ListStyleArchive` in force, if the storage names one.
    pub style: Option<u64>,
    pub text: String,
}

/// The style over a run, resolved — see [`Document::style_of_run`].
#[derive(Debug, Clone)]
pub struct ResolvedStyle {
    /// The object the run points at, named or not.
    pub style: u64,
    /// Nearest ancestor with a name, which is the style a user would say the
    /// text is in.
    pub named: Option<u64>,
    pub name: Option<String>,
    /// Set on the anonymous archives iWork writes for directly formatted text.
    pub is_variation: bool,
    /// `override_count`, field 10 — how many properties the archive claims to
    /// override. Not maintained by this crate, and reported as it is found.
    pub override_count: Option<u64>,
    /// Properties set anywhere between the run and its named style, named where
    /// this crate knows the field and as `bag.field` where it does not.
    pub overrides: Vec<String>,
}

/// What one text edit did — see [`Document::replace_text`].
#[derive(Debug, Clone)]
pub struct TextEdit {
    pub storage: u64,
    /// Where it happened and how much moved, in UTF-16 code units.
    pub edit: crate::text::Edit,
    /// What became of the attribute tables.
    pub report: crate::text::EditReport,
}

/// A run of styled text, held by `TSWP.StorageArchive`.
#[derive(Debug, Clone)]
pub struct TextStorage {
    /// Object identifier — the handle `Document::set_text` takes.
    pub identifier: u64,
    /// Stream the object lives in.
    pub stream: String,
    pub text: String,
}

/// An open iWork document.
///
/// Object streams are decoded on open and re-encoded on save, so a
/// `open` → `save` cycle reproduces every object byte for byte.
pub struct Document {
    package: Package,
    kind: Kind,
    /// Decoded object streams, keyed by package entry name.
    streams: BTreeMap<String, Vec<ArchiveObject>>,
}

impl Document {
    pub fn open(path: impl AsRef<std::path::Path>) -> Result<Document, Error> {
        let path = path.as_ref();
        let package = Package::read(path)?;
        let mut document = Document::from_package(package)?;
        // Trust the extension when there is one; fall back to structure.
        let by_extension = Kind::from_extension(path);
        if by_extension != Kind::Unknown {
            document.kind = by_extension;
        }
        Ok(document)
    }

    /// A new, empty document — the one call here that needs no file at all.
    ///
    /// [`Document::from_template`] answers "create a document" by copying one
    /// that works, and needs Apple's software installed to have something to
    /// copy. This is the other half of the promise: the bytes come from
    /// [`crate::create`], which writes the least document the app was measured
    /// accepting — eleven objects for Pages, where the app's own Blank template
    /// has 569.
    ///
    /// The document is in memory; [`Document::save`] puts it somewhere, and its
    /// identity is minted here, so two calls make two documents.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// let mut doc = iwork::Document::new(iwork::Kind::Pages)?;
    /// doc.append_paragraph("Hello")?;
    /// doc.save("Hello.pages")?;
    ///
    /// let sheet = iwork::Document::new(iwork::Kind::Numbers)?;
    /// sheet.save("Blank.numbers")?;
    ///
    /// let deck = iwork::Document::new(iwork::Kind::Keynote)?;
    /// deck.save("Blank.key")?;
    /// # Ok(()) }
    /// ```
    pub fn new(kind: Kind) -> Result<Document, Error> {
        Document::new_on(kind, crate::create::Paper::default())
    }

    /// [`Document::new`], choosing the paper a Pages document starts on.
    ///
    /// Numbers has no paper — a sheet is sized in its own archive — so this is
    /// [`Document::new`] for it; [`Document::new_spreadsheet`] is the call that
    /// takes a spreadsheet's dimensions.
    pub fn new_on(kind: Kind, paper: crate::create::Paper) -> Result<Document, Error> {
        let blueprint = match kind {
            Kind::Pages => crate::create::pages(paper),
            Kind::Numbers => {
                crate::create::numbers(DEFAULT_SHEET, DEFAULT_TABLE, DEFAULT_ROWS, DEFAULT_COLUMNS)
            }
            Kind::Keynote => crate::create::keynote(DEFAULT_SLIDE_SIZE),
            Kind::Unknown => {
                return Err(Error::Format(
                    "a document has to be a Pages, Numbers or Keynote one".into(),
                ))
            }
        };
        Document::assemble(kind, blueprint)
    }

    /// Turn a finished blueprint into a document.
    fn assemble(kind: Kind, blueprint: crate::create::Blueprint) -> Result<Document, Error> {
        let mut document = Document::from_package(blueprint.finish())?;
        document.kind = kind;
        // Every reference that leaves the component it is written in has to be
        // declared, and working it out from the objects beats maintaining a list
        // while building them — the same call an edit makes.
        document.declare_external_references();
        // A table made from nothing has a haunted owner in its model and no
        // `TSCE` owner anywhere, and a table with no cell owner can hold no
        // formula — `set_formula` refused every one of them by name. Giving the
        // owners here is what makes a document made from nothing a spreadsheet
        // rather than a grid of literals.
        crate::calc::give_tables_their_owners(&mut document)?;
        // The owners went in after the package was assembled, so the engine's
        // stream no longer matches the bytes the document was built from. A
        // document nobody has edited has to report no changed streams —
        // `changed_streams` is how a caller sees what an edit touched, and how
        // several tests prove a refused edit touched nothing.
        document.package = document.saved_package();
        Ok(document)
    }

    /// A new spreadsheet with one sheet and one table of a chosen size.
    ///
    /// [`Document::new`] makes the table Numbers itself makes — 22 rows by 7
    /// columns — and this is for a caller who knows what they are about to put
    /// in it. Every cell of the table can be written from the start:
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// use iwork::table::CellValue;
    /// let mut doc = iwork::Document::new_spreadsheet("Sales", "Q1", 4, 3)?;
    /// doc.set_cell("Q1", 0, 0, CellValue::Text("Region".into()))?;
    /// doc.save("Sales.numbers")?;
    /// # Ok(()) }
    /// ```
    ///
    /// The width is capped at 255 columns, which is the number of cell offsets
    /// Numbers writes into a row whatever the table's width; a wider table is
    /// refused rather than written with an offset array this crate has never
    /// seen the app produce.
    pub fn new_spreadsheet(
        sheet: &str,
        table: &str,
        rows: usize,
        columns: usize,
    ) -> Result<Document, Error> {
        if rows == 0 || columns == 0 {
            return Err(Error::Format(
                "a table needs at least one row and one column".into(),
            ));
        }
        if columns > MAX_COLUMNS {
            return Err(Error::Format(format!(
                "{columns} columns: a new table is capped at {MAX_COLUMNS}, the number of cell \
                 offsets Numbers writes into a row"
            )));
        }
        Document::assemble(
            Kind::Numbers,
            crate::create::numbers(sheet, table, rows, columns),
        )
    }

    /// Make a new document out of a template bundle — `.template`,
    /// `.nmbtemplate` or `.kth`.
    ///
    /// This is the copy-don't-synthesise answer to "create a document". A
    /// template *is* a document package: the same entries, the same object
    /// graph, opened by the same code. What it is not is a *new* document, and
    /// the three things that differ were measured against documents Pages,
    /// Numbers and Keynote made from templates they ship:
    ///
    /// 1. **A fresh identity, lineage included.** `documentUUID`, `shareUUID`,
    ///    `privateUUID`, `versionUUID`, `revision` and
    ///    `Metadata/DocumentIdentifier` are new — and so is
    ///    `stableDocumentUUID`, which is where this differs from
    ///    [`Document::save_as_new`]. A document from a template is not a copy
    ///    of the template; the app gives it a lineage beginning with itself
    ///    ([`crate::metadata::Lineage`]).
    /// 2. **The template it came from, recorded.**
    ///    `TSA.DocumentArchive.template_identifier` is set when the path says
    ///    which template this is — a bundle inside an app's
    ///    `SharedSupport/Templates` — and left alone when it does not, because
    ///    the identifier of a template somewhere else is not something this
    ///    crate can know.
    /// 3. **Nothing else.** No view state to clear: not one of the 901
    ///    templates the three apps ship has an `Index/ViewState.iwa`, a
    ///    `Metadata/BuildVersionHistory.plist` or a preview. The app writes
    ///    those the first time it saves, and a document without them opens —
    ///    which the corpus already proves, two of its fixtures being template
    ///    bundles under a document's name.
    ///
    /// The result is a document in memory: [`Document::save`] writes it
    /// wherever it should live, and saving it twice from two calls gives two
    /// documents with two identities.
    pub fn from_template(path: impl AsRef<std::path::Path>) -> Result<Document, Error> {
        let path = path.as_ref();
        let mut package = Package::read(path)?;
        crate::metadata::assign_identity(&mut package, crate::metadata::Lineage::Fresh)?;
        // A new document starts in the single-file form, whatever shape the
        // template happened to be stored in. The directory form is a thing a
        // user chooses for a document that has grown large (`Change File Type`),
        // not something a template's on-disk layout should decide for them.
        package.form = crate::package::Form::SingleFile;
        let mut document = Document::from_package(package)?;
        let by_extension = Kind::from_extension(path);
        if by_extension != Kind::Unknown {
            document.kind = by_extension;
        }
        if let Some(identifier) = template_identifier(path) {
            crate::metadata::set_template_identifier(&mut document, &identifier)?;
        }
        Ok(document)
    }

    pub fn from_package(package: Package) -> Result<Document, Error> {
        // Before anything tries to Snappy-decompress a stream. A
        // password-protected package is a package whose `Index/*.iwa` are
        // ciphertext, and reporting that as "unexpected chunk marker 0x56" is
        // the least useful thing this crate could say about it.
        if let Some(encryption) = crate::metadata::encryption(&package) {
            return Err(Error::Encrypted {
                hint: encryption.hint,
            });
        }
        let mut streams = BTreeMap::new();
        for name in package.iwa_names() {
            let raw = package.get(&name).expect("name came from the package");
            let objects = iwa::parse(raw).map_err(|e| Error::Format(format!("{name}: {e}")))?;
            streams.insert(name, objects);
        }
        let kind = detect_kind(&package, &streams);
        Ok(Document {
            package,
            kind,
            streams,
        })
    }

    pub fn kind(&self) -> Kind {
        self.kind
    }

    pub fn package(&self) -> &Package {
        &self.package
    }

    pub fn stream_names(&self) -> impl Iterator<Item = &str> {
        self.streams.keys().map(String::as_str)
    }

    pub fn objects(&self) -> impl Iterator<Item = (&str, &ArchiveObject)> {
        self.streams
            .iter()
            .flat_map(|(name, objects)| objects.iter().map(move |o| (name.as_str(), o)))
    }

    pub fn object(&self, identifier: u64) -> Option<(&str, &ArchiveObject)> {
        self.objects().find(|(_, o)| o.identifier == identifier)
    }

    /// The `TSP.PackageMetadata` object, which indexes components and media.
    fn package_metadata(&self) -> Option<Message> {
        for (_, object) in self.objects() {
            if object.message_type() == crate::TYPE_PACKAGE_METADATA {
                return Message::decode(object.payload()).ok();
            }
        }
        None
    }

    pub fn components(&self) -> Vec<Component> {
        let Some(meta) = self.package_metadata() else {
            return Vec::new();
        };
        meta.all(3)
            .filter_map(|value| {
                let Value::Bytes(raw) = value else {
                    return None;
                };
                let c = Message::decode(raw).ok()?;
                Some(Component {
                    identifier: c.varint(1).unwrap_or(0),
                    preferred_name: utf8(c.bytes(2)),
                    file_name: utf8(c.bytes(3)),
                    external_reference_count: c.all(6).count(),
                })
            })
            .collect()
    }

    pub fn data_files(&self) -> Vec<DataFile> {
        let Some(meta) = self.package_metadata() else {
            return Vec::new();
        };
        meta.all(4)
            .filter_map(|value| {
                let Value::Bytes(raw) = value else {
                    return None;
                };
                let d = Message::decode(raw).ok()?;
                Some(DataFile {
                    identifier: d.varint(1).unwrap_or(0),
                    digest: d.bytes(2).unwrap_or_default().to_vec(),
                    original_name: utf8(d.bytes(3)),
                    stored_name: utf8(d.bytes(4)),
                    asset_path: utf8(d.bytes(5)),
                })
            })
            .collect()
    }

    /// Objects that carry a version patch — `MessageInfo.type == 0`.
    ///
    /// The archive format lets one object hold its message **and older
    /// encodings of it as patches**: a first message of the real type, then
    /// further messages of type `0`, each naming a `base_message_index`, an app
    /// version it is meant for, and the fields to drop from the base before
    /// merging. The first message is the newest encoding and the one the
    /// current app reads; the patches exist so that an older Numbers opening
    /// the same file gets a shape it understands.
    ///
    /// **What 15.3.1 actually writes, over this whole corpus and after an edit
    /// made by the app itself: exactly one patched object per Numbers document,
    /// and none at all in Pages or Keynote.** It is the `TN.UIStateArchive` in
    /// the view-state component, with three patches for 11.0, 10.1 and 10.0,
    /// each dropping field 28 from the base and supplying its own. No table
    /// archive — no tile, no data list, no model — carries one.
    ///
    /// So the rule this crate follows is short. Read the first message and
    /// ignore the patches, which is what the app does. Never write one. And
    /// **never rewrite the first message of an object that has them**, because
    /// the patches would then describe the object as it used to be, which is a
    /// document that says two different things depending on who opens it.
    /// [`Document::set_cell`] refuses on that ground rather than silently
    /// producing one.
    pub fn patched_objects(&self) -> Vec<(u64, usize)> {
        self.objects()
            .filter_map(|(_, object)| {
                let patches = object
                    .messages
                    .iter()
                    .filter(|m| m.message_type == 0)
                    .count();
                (patches > 0).then_some((object.identifier, patches))
            })
            .collect()
    }

    /// Highest object identifier the document has ever allocated.
    ///
    /// New objects must take identifiers above this, and the field must be
    /// bumped to match, or iWork will hand out identifiers that already exist.
    pub fn last_object_identifier(&self) -> Option<u64> {
        self.package_metadata()?.varint(1)
    }

    /// Every non-empty text storage in the document, in stream order.
    pub fn text_storages(&self) -> Vec<TextStorage> {
        let mut out = Vec::new();
        for (stream, object) in self.objects() {
            if object.message_type() != crate::TYPE_STORAGE {
                continue;
            }
            let Ok(storage) = Message::decode(object.payload()) else {
                continue;
            };
            let text = text::read(&storage);
            if !text::has_content(&text) {
                continue;
            }
            out.push(TextStorage {
                identifier: object.identifier,
                stream: stream.to_string(),
                text,
            });
        }
        out
    }

    /// Replace the contents of one text storage.
    ///
    /// A full-range [`Document::replace_text`], which is what "select all and
    /// type" is. The first paragraph keeps its style and the paragraphs of the
    /// new text are given entries of their own where the storage had one per
    /// paragraph before; nothing is left pointing at a character that is not
    /// there any more.
    pub fn set_text(&mut self, identifier: u64, new_text: &str) -> Result<TextEdit, Error> {
        let length = text::length(&self.storage_text(identifier)?);
        self.replace_text(identifier, 0..length, new_text)
    }

    /// The body of a Pages document — the storage its text goes in.
    ///
    /// A word-processing document has exactly one: `StorageArchive.kind` 0, the
    /// storage `TP.DocumentArchive` field 4 points at. Headers, footers, text
    /// boxes, table cells and footnotes are storages too, and none of them is
    /// this one.
    pub fn body_storage(&self) -> Option<u64> {
        self.structure()?.body_storage
    }

    /// Refuse an edit to "the document body" when the document does not draw
    /// one.
    ///
    /// **A page-layout document has a body storage and never shows it.** Its
    /// words live in text boxes, and the app's Document Body switch is off —
    /// which is what the mode *is*. The storage is still there, holding the
    /// `U+0004` that starts the first section, so an append into it succeeds at
    /// the byte level and produces text nobody will ever see: measured, by
    /// appending a paragraph to `pages-layout.pages`, opening it in Pages and
    /// asking for every word in the document. Forty lines came back and the new
    /// paragraph was not among them.
    ///
    /// So the concept "the document's body" is refused here rather than
    /// written into a void. The storage itself is still reachable by identifier
    /// through [`Document::text_mut`], for a caller who means exactly that.
    fn refuse_if_body_is_not_drawn(&self) -> Result<(), Error> {
        let Some(structure) = self.structure() else {
            return Ok(());
        };
        if structure.mode == crate::pages::Mode::PageLayout {
            return Err(Error::refused(
                Refusal::NotDrawn,
                "this is a page-layout document: it has a body storage and the app never draws \
                 it, so text put there is text nobody sees. Its words live in text boxes — \
                 `add_text_box` makes one — and the storage itself is reachable by identifier \
                 if that is really what was meant",
            ));
        }
        Ok(())
    }

    /// Add a paragraph to the end of a Pages document's body.
    ///
    /// The exceljs-shaped call: a document from [`Document::new`] has an empty
    /// body and this is how text gets into it. A paragraph is a run of text
    /// ending at a newline, so this appends one — with the newline in front of
    /// it when there is already text to separate from, and without when the
    /// body is empty and the new text *is* the first paragraph.
    ///
    /// Everything anchored into the body moves with the insert, exactly as it
    /// does for [`Document::insert_text`]: this is that call with the index
    /// worked out.
    pub fn append_paragraph(&mut self, text: &str) -> Result<TextEdit, Error> {
        self.refuse_if_body_is_not_drawn()?;
        let Some(storage) = self.body_storage() else {
            return Err(Error::refused(
                Refusal::Missing,
                format!(
                    "a {} document has no body to append a paragraph to",
                    self.kind.as_str()
                ),
            ));
        };
        let existing = self.storage_text(storage)?;
        let at = text::length(&existing);
        let addition = match existing.is_empty() {
            true => text.to_string(),
            false => format!("\n{text}"),
        };
        self.insert_text(storage, at, &addition)
    }

    /// Insert text at a character index, moving everything anchored past it.
    ///
    /// `at` counts **UTF-16 code units**, like every character index in the
    /// format; see [`Document::replace_text`].
    pub fn insert_text(
        &mut self,
        identifier: u64,
        at: u64,
        new_text: &str,
    ) -> Result<TextEdit, Error> {
        self.replace_text(identifier, at..at, new_text)
    }

    /// Delete a range of a storage's text, moving everything anchored past it
    /// and dropping what was inside it.
    ///
    /// `range` counts **UTF-16 code units**; see [`Document::replace_text`].
    pub fn delete_text(&mut self, identifier: u64, range: Range<u64>) -> Result<TextEdit, Error> {
        self.replace_text(identifier, range, "")
    }

    /// Replace `range` of a storage's text with `new_text`, remapping every
    /// attribute table the storage carries.
    ///
    /// **Indices are UTF-16 code units**, which is what the format counts:
    /// `"🎬"` is two, and an index between its halves is
    /// [`Error::SplitSurrogate`] rather than a corrupted string.
    ///
    /// Everything anchored into the storage moves with the text — paragraph,
    /// list, character and layout styles, smart fields and hyperlinks, language
    /// and dictation runs, tracked insertions and deletions, comment and pencil
    /// annotations, bookmarks, drop caps, and the paragraph-start bookkeeping.
    /// [`crate::text::TABLES`] is the whole inventory and [`crate::text::Edit`]
    /// is what each kind does. Where the rules came from — a probe per rule,
    /// with Pages performing the edit — is in `FORMAT.md` §Text.
    ///
    /// What it refuses rather than damaging:
    ///
    /// * a range covering the character an object is anchored to
    ///   ([`Error::AnchoredObject`]) — an image, a footnote mark, a section
    ///   break. Pages deletes the object; this crate cannot, so it declines;
    /// * a storage carrying a table this crate does not know
    ///   ([`Error::UnknownAttributeTable`]);
    /// * text containing `U+FFFC`, `U+0004` or `U+0005`
    ///   ([`Error::UnwritableCharacter`]), which stand for objects;
    /// * an index inside a surrogate pair, or outside the text.
    pub fn replace_text(
        &mut self,
        identifier: u64,
        range: Range<u64>,
        new_text: &str,
    ) -> Result<TextEdit, Error> {
        if let Some(character) = new_text.chars().find(|c| text::UNWRITABLE.contains(c)) {
            return Err(Error::UnwritableCharacter { character });
        }

        for (name, objects) in self.streams.iter_mut() {
            for object in objects.iter_mut() {
                if object.identifier != identifier || object.message_type() != crate::TYPE_STORAGE {
                    continue;
                }
                let mut storage = Message::decode(object.payload())
                    .map_err(|e| Error::Format(format!("{name}: storage {identifier}: {e}")))?;

                let old = text::read(&storage);
                let length = text::length(&old);
                let (start, end) = (range.start, range.end.max(range.start));
                if end > length {
                    return Err(Error::TextRange {
                        storage: identifier,
                        index: end,
                        length,
                    });
                }
                let (Some(from), Some(to)) = (
                    text::utf16_offset(&old, start),
                    text::utf16_offset(&old, end),
                ) else {
                    let index = if text::utf16_offset(&old, start).is_none() {
                        start
                    } else {
                        end
                    };
                    return Err(Error::SplitSurrogate {
                        storage: identifier,
                        index,
                    });
                };

                let edit = text::Edit {
                    at: start,
                    removed: end - start,
                    inserted: text::length(new_text),
                };
                // Everything `text::apply` requires of its caller, in one
                // question: the tables it can place and decode, the text it can
                // read back byte for byte, change tracking, and what this
                // particular edit would destroy. See [`text::refusal`].
                if let Some(refusal) = text::refusal(&storage, edit) {
                    return Err(refusal.into_error(identifier));
                }

                let mut text = String::with_capacity(old.len() - (to - from) + new_text.len());
                text.push_str(&old[..from]);
                text.push_str(new_text);
                text.push_str(&old[to..]);

                let report = text::apply(&mut storage, edit, &text);
                let payload = storage.encode();
                if payload != object.payload() {
                    object.messages[0].payload = payload;
                }
                return Ok(TextEdit {
                    storage: identifier,
                    edit,
                    report,
                });
            }
        }
        Err(Error::NoSuchObject(identifier))
    }

    // -- what a storage carries ----------------------------------------------

    /// Every `TSWP.StorageArchive` in the document, with the attribute tables it
    /// carries — the inventory an edit has to remap.
    ///
    /// Unlike [`Document::text_storages`] this reports the placeholders too, and
    /// the storages whose whole contents are a `U+FFFC`, because those are
    /// exactly the ones with something anchored into them.
    pub fn storages(&self) -> Vec<StorageInventory> {
        let mut out = Vec::new();
        for (stream, object) in self.objects() {
            if object.message_type() != crate::TYPE_STORAGE {
                continue;
            }
            let Ok(storage) = Message::decode(object.payload()) else {
                continue;
            };
            let text = text::read(&storage);
            let mut tables = Vec::new();
            for spec in text::TABLES {
                let Some(decoded) = storage.bytes(spec.field).and_then(crate::pb::decode_nested)
                else {
                    continue;
                };
                tables.push(StorageTable {
                    field: spec.field,
                    name: spec.name,
                    anchoring: spec.anchoring,
                    entries: text::entry_indices(&decoded, spec.anchoring).len(),
                });
            }
            out.push(StorageInventory {
                identifier: object.identifier,
                stream: stream.to_string(),
                kind: storage.varint(1).unwrap_or(3),
                length: text::length(&text),
                paragraphs: text::paragraph_ranges(&text).len(),
                unknown_field: text::unknown_table(&storage),
                tables,
            });
        }
        out
    }

    /// Every smart field in the document — hyperlinks above all, but also the
    /// placeholder, date, page-number and mail-merge fields that share the table.
    ///
    /// A smart field is a *run*: `StorageArchive` field 11 holds an entry
    /// carrying the field's object at the character it starts on, and the run
    /// reaches to the next entry — which for a field that does not run to the
    /// end of the text is an entry with **no object**, a terminator. Both shapes
    /// occur in Apple's own templates, and one link in `46_Business_Modern…`
    /// runs to the end with no terminator at all.
    pub fn smart_fields(&self) -> Vec<SmartField> {
        let mut out = Vec::new();
        for (_, object) in self.objects() {
            if object.message_type() != crate::TYPE_STORAGE {
                continue;
            }
            let Ok(storage) = Message::decode(object.payload()) else {
                continue;
            };
            let Some(table) = storage.bytes(11).and_then(crate::pb::decode_nested) else {
                continue;
            };
            let text = text::read(&storage);
            let length = text::length(&text);
            let entries = text::entry_indices(&table, text::Anchoring::Run);
            for (position, (index, target)) in entries.iter().enumerate() {
                let Some(target) = target else { continue };
                let end = entries
                    .get(position + 1)
                    .map(|(next, _)| *next)
                    .unwrap_or(length);
                let (_, field) = self.object(*target).unzip();
                let archive = field.and_then(|o| Message::decode(o.payload()).ok());
                out.push(SmartField {
                    storage: object.identifier,
                    object: *target,
                    message_type: field.map(|o| o.message_type()).unwrap_or(0),
                    range: *index..end,
                    payload: archive
                        .as_ref()
                        .and_then(|a| a.bytes(2))
                        .map(|b| String::from_utf8_lossy(b).into_owned()),
                    text: slice(&text, *index..end),
                });
            }
        }
        out
    }

    /// Point a hyperlink at a different address.
    ///
    /// `TSWP.HyperlinkFieldArchive` (2032) is `{1: {1: uuid}, 2: url}` — the
    /// whole of the link is that one string, so changing it is changing one
    /// field of one object and nothing else. What the *text* says is separate
    /// and stays as it was; a link reading "example.com" and pointing at
    /// somewhere else is a document the app writes happily.
    pub fn set_link_url(&mut self, identifier: u64, url: &str) -> Result<(), Error> {
        for (name, objects) in self.streams.iter_mut() {
            for object in objects.iter_mut() {
                if object.identifier != identifier {
                    continue;
                }
                if object.message_type() != TYPE_HYPERLINK_FIELD {
                    return Err(Error::Format(format!(
                        "object {identifier} is a {} archive, not a hyperlink (2032)",
                        object.message_type()
                    )));
                }
                let mut archive = Message::decode(object.payload())
                    .map_err(|e| Error::Format(format!("{name}: link {identifier}: {e}")))?;
                archive.set_in_order(2, Value::Bytes(url.as_bytes().to_vec()));
                object.messages[0].payload = archive.encode();
                return Ok(());
            }
        }
        Err(Error::NoSuchObject(identifier))
    }

    /// The list state of every paragraph of a storage: its level, and the list
    /// style in force.
    ///
    /// Two tables together say it, keyed on the same paragraph starts. Field 6
    /// (`table_para_data`) carries `{character_index, first, second}` and
    /// **`first` is the level**, counted from 0; field 7 (`table_list_style`)
    /// points at the `TSWP.ListStyleArchive`. Both are sparse — a paragraph with
    /// no entry keeps whatever the paragraph before it had — which is why they
    /// are read forward rather than looked up.
    ///
    /// Read off Apple's own templates: `60_Academic_Modern_PM` has one storage
    /// whose five paragraphs are levels 0 to 4 with a style override each, and
    /// `04_Real_Estate_Flyer` has three named list styles and a level change in
    /// one storage of fourteen paragraphs.
    pub fn list_paragraphs(&self, identifier: u64) -> Result<Vec<ListParagraph>, Error> {
        let storage = self.archive_of(identifier)?;
        let text = text::read(&storage);
        let levels = storage
            .bytes(6)
            .and_then(crate::pb::decode_nested)
            .map(|t| text::para_data(&t))
            .unwrap_or_default();
        let styles = storage
            .bytes(7)
            .and_then(crate::pb::decode_nested)
            .map(|t| text::entry_indices(&t, text::Anchoring::Paragraph))
            .unwrap_or_default();

        // Both tables are sparse and both carry forward: a paragraph with no
        // entry, and a paragraph whose entry names nothing, keep what the
        // paragraph before them had.
        let carried = |entries: &[(u64, Option<u64>)], at: u64| {
            entries
                .iter()
                .rev()
                .find(|(index, value)| *index <= at && value.is_some())
                .and_then(|(_, value)| *value)
        };
        Ok(text::paragraph_ranges(&text)
            .into_iter()
            .map(|range| ListParagraph {
                level: carried(&levels, range.start).unwrap_or(0),
                style: carried(&styles, range.start),
                text: slice(&text, range.clone()),
                range,
            })
            .collect())
    }

    /// The style in force over one character, resolved to the named style it
    /// comes from and what the run overrides on top of it.
    ///
    /// Most runs in a real document do not point at a named style. They point at
    /// a **variation**: an anonymous archive carrying [`style::IS_VARIATION`], a
    /// parent, and a property bag holding only what differs. That is what iWork
    /// writes when text is formatted directly rather than by picking a style
    /// from the list, and it is why editing "Title" can leave the title looking
    /// exactly as it did.
    ///
    /// This walks the parent chain to the first archive with a name and reports
    /// both ends: which named style the run ultimately is, and which properties
    /// the chain below it sets.
    pub fn style_of_run(
        &self,
        storage: u64,
        index: u64,
        kind: StyleKind,
    ) -> Result<Option<ResolvedStyle>, Error> {
        let archive = self.archive_of(storage)?;
        let Some(table) = archive
            .bytes(kind.attribute_table())
            .and_then(crate::pb::decode_nested)
        else {
            return Ok(None);
        };
        let anchoring = text::table(kind.attribute_table())
            .map(|t| t.anchoring)
            .unwrap_or(text::Anchoring::Run);
        let entries = text::entry_indices(&table, anchoring);
        // A paragraph entry with no object means "whatever was in force here"
        // — it is what Pages writes for a paragraph an edit has just created —
        // so resolving one walks back to the last entry that names a style. A
        // *run* entry with no object is a terminator and means exactly what it
        // says: no attribute from here on.
        let resolved = match anchoring {
            text::Anchoring::Paragraph => entries
                .iter()
                .rev()
                .find(|(at, target)| *at <= index && target.is_some()),
            _ => entries.iter().rev().find(|(at, _)| *at <= index),
        };
        let Some(identifier) = resolved.and_then(|(_, target)| *target) else {
            return Ok(None);
        };

        // What the *variations* set on the way down, and nothing from the named
        // style itself — a named style's bags carry every property it has,
        // defaults included, and calling those overrides would say a paragraph
        // overrides sixty things when it overrides one.
        let mut overrides: Vec<String> = Vec::new();
        let mut named = None;
        let mut name = None;
        let mut walker = Some(identifier);
        let mut seen = 0;
        while let (Some(current), true) = (walker, seen < 16) {
            seen += 1;
            let Some(style) = self.text_style(current) else {
                break;
            };
            if style.name.is_some() {
                named = Some(current);
                name = style.name.clone();
                break;
            }
            for bag in [11u32, 12u32] {
                let Some(properties) = style.archive.bytes(bag).and_then(crate::pb::decode_nested)
                else {
                    continue;
                };
                for field in &properties.fields {
                    let path = [bag, field.number];
                    let label = style::property::BY_NAME
                        .iter()
                        .find(|(_, known, _)| *known == path)
                        .map(|(label, _, _)| label.to_string())
                        .unwrap_or_else(|| format!("{bag}.{}", field.number));
                    if !overrides.contains(&label) {
                        overrides.push(label);
                    }
                }
            }
            walker = style.parent;
        }

        let archive = self.archive_of(identifier)?;
        Ok(Some(ResolvedStyle {
            style: identifier,
            named,
            name,
            is_variation: style::get_path(&archive, style::IS_VARIATION)
                .is_some_and(|v| v != Value::Varint(0)),
            override_count: match style::get_path(&archive, style::OVERRIDE_COUNT) {
                Some(Value::Varint(n)) => Some(n),
                _ => None,
            },
            overrides,
        }))
    }

    // -- tables --------------------------------------------------------------

    /// Every table in the document, in object order.
    ///
    /// Tables are a cross-app archive: Numbers puts them on sheets, Pages on
    /// pages and Keynote on slides, and all three write the same `TST` object
    /// graph. See [`crate::table`] for what a table is made of.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let doc = iwork::Document::open("Budget.numbers")?;
    /// for table in doc.tables() {
    ///     println!("{} — {}×{}", table.name, table.rows, table.columns);
    ///     println!("{}", table.value(1, 0).to_text());
    /// }
    /// # Ok(()) }
    /// ```
    pub fn tables(&self) -> Vec<crate::table::Table> {
        let mut tables = crate::table::tables(self);
        for table in &mut tables {
            crate::table::resolve_rich_text(self, table);
        }
        tables
    }

    /// Every sheet of a Numbers document, in the order the tabs show them.
    ///
    /// **A sheet is not a grid.** This is where a Numbers document parts
    /// company with the spreadsheet everyone pictures: a sheet is a *canvas*,
    /// and what it holds is a list of drawables — any number of tables, and
    /// charts, shapes, images and text boxes beside them. Two tables on one
    /// sheet is ordinary, and they may share nothing but the page they are
    /// drawn on. So this reports the sheet and what is on it, and
    /// [`crate::table::Sheet::tables`] narrows that list to the tables rather than pretending
    /// the sheet is one.
    ///
    /// Pages and Keynote have no sheets — a table there hangs off a page or a
    /// slide — so this is empty for them and [`Document::tables`] is the way in.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let doc = iwork::Document::open("Budget.numbers")?;
    /// for sheet in doc.sheets() {
    ///     println!("{} — {} drawable(s)", sheet.name, sheet.drawables.len());
    ///     for table in sheet.tables(&doc) {
    ///         println!("  table {} — {}×{}", table.name, table.rows, table.columns);
    ///     }
    /// }
    /// # Ok(()) }
    /// ```
    pub fn sheets(&self) -> Vec<crate::table::Sheet> {
        crate::table::sheets(self)
    }

    /// One sheet by name or by identifier.
    pub fn sheet(&self, wanted: &str) -> Option<crate::table::Sheet> {
        let by_id: Option<u64> = wanted.parse().ok();
        self.sheets()
            .into_iter()
            .find(|sheet| Some(sheet.identifier) == by_id || sheet.name == wanted)
    }

    /// One table by its `TST.TableInfoArchive` identifier, or by name.
    pub fn table(&self, wanted: &str) -> Option<crate::table::Table> {
        let by_id: Option<u64> = wanted.parse().ok();
        self.tables()
            .into_iter()
            .find(|t| Some(t.identifier) == by_id || t.name == wanted)
    }

    /// Every named cell format the document defines.
    ///
    /// Custom formats are document-scoped, not table-scoped: cells across every
    /// table reach into one list by UUID. See [`crate::table::CustomFormat`].
    pub fn custom_formats(&self) -> Vec<crate::table::CustomFormat> {
        crate::table::custom_formats(self)
    }

    /// Write a value into a cell that already exists.
    ///
    /// The narrow, provable operation: the grid keeps its shape, every row and
    /// column keeps its identity, and nothing that indexes rows by position —
    /// a category's group nodes, a filter's hidden state — has anything to
    /// re-point. What moves is the tile's storage for one row, and with it the
    /// interned string and format entries the cell reaches into.
    ///
    /// What travels with the value:
    ///
    /// * the cell's **data format**, because value and format travel together.
    ///   A cell that keeps its type keeps its format key untouched; a cell that
    ///   changes type is given the key another cell of the new type already
    ///   uses in this table, and its old slot's key is released — which is what
    ///   Numbers 15.3.1 does, observed by writing a number over a text cell.
    /// * everything this crate does not model. Bytes 2–5, byte 6's
    ///   chosen-format bits, byte 7, any trailing bytes, the cell and text
    ///   style keys, the control definition and — the one that would be
    ///   invisible until someone looked at the document — the **conditional
    ///   style and rule keys**, which are how a highlighted cell knows it is
    ///   highlighted.
    ///
    /// What it refuses, by name rather than by writing something plausible:
    /// a cell holding a formula (removing one means editing `TSCE`), a
    /// rich-text cell (its text is a `TSWP` storage, not a table string), a
    /// cell covered by a merge, and any object carrying version patches — see
    /// [`Document::patched_objects`]. A row with no stored cells is *not*
    /// refused: the write brings the row into being.
    pub fn set_cell(
        &mut self,
        wanted: &str,
        row: usize,
        column: usize,
        value: CellValue,
    ) -> Result<CellValue, Error> {
        let table = self.table_for_write(wanted)?;
        let previous = table.value(row, column);
        self.write_cells(&table, vec![(row, column, value)])?;
        Ok(previous)
    }

    /// Write many cells of one table in a single pass.
    ///
    /// This is [`Document::set_cell`] for a block of values, and the two share
    /// every rule and every refusal — `set_cell` is one cell handed to this.
    /// What the batch adds is the two things a loop of single writes cannot
    /// give:
    ///
    /// * **One pass over the row.** A cell is stored in its row's
    ///   `TileRowInfo`, so writing a row one cell at a time decodes and
    ///   re-encodes that row — and re-reads every table in the document to
    ///   resolve the name — once per cell. The cost of filling a table that way
    ///   is quadratic in its size, measured: 500 cells in 0.2s, 2000 in 3.0s,
    ///   and a spreadsheet's worth of cells in minutes. Grouped by row it is
    ///   one decode, one re-encode and one table resolution however wide the
    ///   row.
    /// * **All or nothing.** A refused cell leaves the document byte for byte
    ///   as it was — including the cells of the same batch that had already
    ///   been written. A loop of `set_cell` stops half-applied instead.
    ///
    /// Cells may be given in any order and may span any number of rows; naming
    /// the same cell twice is refused rather than silently resolved, because
    /// which of the two values was meant is not this crate's to guess.
    pub fn set_cells(
        &mut self,
        wanted: &str,
        cells: impl IntoIterator<Item = (usize, usize, CellValue)>,
    ) -> Result<usize, Error> {
        let table = self.table_for_write(wanted)?;
        self.write_cells(&table, cells.into_iter().collect())
    }

    /// Write a rectangular block of values, `rows[0][0]` landing at `at`.
    ///
    /// The shape a caller with a table of data has: rows of values, written
    /// into the table at a corner. Short rows leave the cells past their end
    /// alone rather than emptying them — a row of three values written into a
    /// five-column table changes three cells — because a value nobody gave is
    /// not the same as a value someone cleared. Pass [`CellValue::Empty`] to
    /// clear.
    pub fn set_block(
        &mut self,
        wanted: &str,
        at: (usize, usize),
        rows: &[Vec<CellValue>],
    ) -> Result<usize, Error> {
        let cells: Vec<(usize, usize, CellValue)> = rows
            .iter()
            .enumerate()
            .flat_map(|(row, values)| {
                values
                    .iter()
                    .enumerate()
                    .map(move |(column, value)| (at.0 + row, at.1 + column, value.clone()))
            })
            .collect();
        self.set_cells(wanted, cells)
    }

    /// Every cell write in this crate, single or batched, goes through here.
    ///
    /// Two halves, in this order and for the reason the whole write path is
    /// built around: every refusal that can be decided from the request alone
    /// is decided *first*, over the whole batch, before a byte moves. What can
    /// only be found while planning a cell — a donor format that is not there,
    /// a string key that will not fit — is caught with the streams snapshotted,
    /// and the snapshot is what makes a refused batch a no-op rather than a
    /// half-written table.
    fn write_cells(
        &mut self,
        table: &crate::table::Table,
        cells: Vec<(usize, usize, CellValue)>,
    ) -> Result<usize, Error> {
        self.write_cells_with(table, cells, ListCache::default())
    }

    /// The same, with a cache the caller has already put something in.
    ///
    /// One caller does: [`Document::set_currency`] interns the currency format
    /// and seeds it as the *donor* for the currency slot, so the cells it then
    /// writes point at that format rather than hunting the table for one to
    /// borrow — which a table holding no money has none of.
    fn write_cells_with(
        &mut self,
        table: &crate::table::Table,
        cells: Vec<(usize, usize, CellValue)>,
        cache: ListCache,
    ) -> Result<usize, Error> {
        let mut by_row: std::collections::BTreeMap<usize, Vec<(usize, CellValue)>> =
            std::collections::BTreeMap::new();
        for (row, column, value) in cells {
            let where_ = format!(
                "{} {}",
                table.name,
                crate::table::CellRef::Index(row, column)
            );
            if row >= table.rows || column >= table.columns {
                return Err(Error::refused(
                    Refusal::OutOfBounds,
                    format!("{where_}: the table is {}×{}", table.rows, table.columns),
                ));
            }
            if let Some(merge) = table.merge_covering(row, column) {
                if (merge.row, merge.column) != (row, column) {
                    return Err(Error::refused(
                        Refusal::Merged,
                        format!(
                            "{where_}: covered by the merge that begins at row {} column {}",
                            merge.row, merge.column
                        ),
                    ));
                }
            }
            match value {
                CellValue::Empty
                | CellValue::Text(_)
                | CellValue::Number(_)
                | CellValue::Currency(_)
                | CellValue::Bool(_)
                | CellValue::Date(_)
                | CellValue::Duration(_) => {}
                other => {
                    return Err(Error::refused(
                        Refusal::UnwritableValue,
                        format!("{where_}: this crate does not write {} cells", other.kind()),
                    ))
                }
            }
            let row_cells = by_row.entry(row).or_default();
            if row_cells.iter().any(|(at, _)| *at == column) {
                return Err(Error::refused(
                    Refusal::NotACell,
                    format!("{where_}: named twice in one write"),
                ));
            }
            row_cells.push((column, value));
        }

        let site = self.table_site(table)?;
        let restore = self.streams.clone();
        match self.apply_cell_batch(table, &site, by_row, cache) {
            Ok(written) => Ok(written),
            Err(e) => {
                self.streams = restore;
                Err(e)
            }
        }
    }

    /// Apply a validated batch: one decode per tile, one rewrite per bucket.
    ///
    /// The grouping is what makes a batch cheap. A cell lives in its row's
    /// `TileRowInfo`, a row in its tile, and a tile holds up to 256 rows, so
    /// decoding the tile per cell — which is what a loop of single writes does —
    /// re-reads the whole tile for every value. Here the tile is decoded once,
    /// every row of it is laid back in, and the object is written once. The
    /// header buckets go the same way: their per-row and per-column cell counts
    /// are accumulated and applied in one pass each, rather than a decode and a
    /// re-encode per cell that appears.
    ///
    /// Planning and applying stay interleaved per cell, because they have to
    /// be: two cells given two new strings need two keys, and the second key is
    /// only free once the first has been taken. Every fallible step of a cell
    /// still happens before that cell's reference counts move, and the caller's
    /// snapshot covers a later cell refusing.
    fn apply_cell_batch(
        &mut self,
        table: &crate::table::Table,
        site: &TableSite,
        by_row: std::collections::BTreeMap<usize, Vec<(usize, CellValue)>>,
        mut cache: ListCache,
    ) -> Result<usize, Error> {
        let mut by_tile: std::collections::BTreeMap<usize, Vec<TileRowCells>> =
            std::collections::BTreeMap::new();
        for (row, cells) in by_row {
            by_tile
                .entry(row / site.tile_size)
                .or_default()
                .push((row, cells));
        }

        // The patched-object list is a scan of every object in the document, so
        // it is read once for the batch rather than once for every cell.
        let patched = self.patched_objects();
        let mut written = 0;
        let mut row_delta: std::collections::BTreeMap<usize, i64> = Default::default();
        let mut column_delta: std::collections::BTreeMap<usize, i64> = Default::default();

        for (_, rows) in by_tile {
            let first = rows.first().expect("a tile with no rows is not grouped").0;
            let where_ = format!("{} r{first}", table.name);
            let tile = self.tile_for_row(site, first, &where_)?;
            let mut archive = self.archive_of(tile)?;
            let mut index = TileIndex::of(&archive);
            let mut touched_tile = false;

            for (row, cells) in rows {
                let mut in_tile = index.row(&archive, row % site.tile_size, &where_)?;
                let mut changed = false;
                for (column, value) in cells {
                    let where_ = format!(
                        "{} {}",
                        table.name,
                        crate::table::CellRef::Index(row, column)
                    );
                    let slots = in_tile.records.len();
                    if column >= slots {
                        return Err(Error::refused(
                            Refusal::OutOfBounds,
                            format!("{where_}: the row's offset array names only {slots} columns"),
                        ));
                    }
                    let current = in_tile.records[column].clone();
                    let old = match &current {
                        Some(bytes) => crate::table::decode_cell(bytes)
                            .map_err(|e| Error::Format(format!("{where_}: {e}")))?,
                        None => crate::table::CellRecord {
                            version: 5,
                            ..crate::table::CellRecord::default()
                        },
                    };
                    if old.formula_id.is_some() {
                        return Err(Error::refused(
                            Refusal::HoldsFormula,
                            format!(
                                "{where_}: holds a formula, and taking one out means editing the \
                             calculation engine — not this phase's job"
                            ),
                        ));
                    }
                    if old.cell_type == cell_type::RICH_TEXT || old.rich_id.is_some() {
                        return Err(Error::refused(
                            Refusal::HoldsRichText,
                            format!(
                                "{where_}: holds rich text, whose words are in a TSWP storage \
                             rather than in the table"
                            ),
                        ));
                    }
                    // Nothing to do, and nothing to damage.
                    if current.is_none() && value == CellValue::Empty {
                        continue;
                    }

                    let write =
                        self.rewrite_record(&mut cache, table, site, old, &value, &where_)?;

                    // Everything this will rewrite has to be free of version
                    // patches, not just the tile — the string, format and
                    // control lists and the header buckets are set_archive'd
                    // too, and a patch left over one of those would describe the
                    // cell as it used to be.
                    let mut touched: Vec<u64> =
                        write.mutations.iter().map(ListMutation::list).collect();
                    let had = current.is_some();
                    let has = write.record.is_some();
                    if had != has {
                        touched.extend(site.row_bucket.iter().copied());
                        touched.extend(site.column_bucket);
                    }
                    touched.push(tile);
                    for id in touched {
                        if patched.iter().any(|&(at, _)| at == id) {
                            return Err(Error::refused(
                                Refusal::Patched,
                                format!(
                                    "{where_}: object {id} carries version patches, and rewriting \
                                 it would leave them describing the cell as it used to be"
                                ),
                            ));
                        }
                    }

                    // Prove the record encodes before any reference count moves.
                    let encoded = match &write.record {
                        Some(record) => Some(
                            record
                                .encode()
                                .map_err(|e| Error::Format(format!("{where_}: {e}")))?,
                        ),
                        None => None,
                    };
                    for mutation in &write.mutations {
                        self.apply_mutation(&mut cache, mutation)?;
                    }
                    match (had, has) {
                        (false, true) => {
                            *row_delta.entry(row).or_default() += 1;
                            *column_delta.entry(column).or_default() += 1;
                        }
                        (true, false) => {
                            *row_delta.entry(row).or_default() -= 1;
                            *column_delta.entry(column).or_default() -= 1;
                        }
                        _ => {}
                    }
                    in_tile.records[column] = encoded;
                    changed = true;
                    written += 1;
                }
                if changed {
                    in_tile.put(&mut archive, &where_)?;
                    if in_tile.fresh {
                        index.inserted(in_tile.index, in_tile.position);
                    }
                    touched_tile = true;
                }
            }
            if touched_tile {
                self.set_archive(tile, &archive)?;
            }
        }

        cache.flush(self)?;
        if written > 0 {
            self.step_row_counts(site, &row_delta)?;
            if let Some(bucket) = site.column_bucket {
                self.step_bucket_counts(bucket, &column_delta)?;
            }
        }
        Ok(written)
    }

    /// Move the per-row cell counts, each in whichever bucket holds its row.
    ///
    /// A table's rows are counted across a *list* of `HeaderStorageBucket`s
    /// (its columns in a single one), so which bucket a row belongs to is
    /// looked up once for the whole batch rather than per row — the lookup
    /// decodes the bucket.
    fn step_row_counts(
        &mut self,
        site: &TableSite,
        deltas: &std::collections::BTreeMap<usize, i64>,
    ) -> Result<(), Error> {
        if deltas.is_empty() {
            return Ok(());
        }
        let mut per_bucket: std::collections::BTreeMap<
            u64,
            std::collections::BTreeMap<usize, i64>,
        > = Default::default();
        let mut holds: Vec<(u64, std::collections::BTreeSet<usize>)> = Vec::new();
        for &bucket in &site.row_bucket {
            holds.push((bucket, self.bucket_indices(bucket)));
        }
        for (&row, &delta) in deltas {
            if delta == 0 {
                continue;
            }
            let bucket = holds
                .iter()
                .find(|(_, indices)| indices.contains(&row))
                .map(|(bucket, _)| *bucket)
                .or_else(|| site.row_bucket.first().copied());
            if let Some(bucket) = bucket {
                *per_bucket
                    .entry(bucket)
                    .or_default()
                    .entry(row)
                    .or_default() += delta;
            }
        }
        for (bucket, deltas) in per_bucket {
            self.step_bucket_counts(bucket, &deltas)?;
        }
        Ok(())
    }

    /// Resolve a table for a *write*, where an ambiguous name is an error.
    ///
    /// Reading by name may settle for the first match; writing must not. Apple's
    /// own `numbers-pivot.numbers` carries two tables called `Sales` on different
    /// sheets, and silently editing the first one is how a caller writes into the
    /// wrong table without ever being told there was a choice. A name that names
    /// more than one table is refused with the identifiers to disambiguate by.
    fn table_for_write(&self, wanted: &str) -> Result<crate::table::Table, Error> {
        let by_id: Option<u64> = wanted.parse().ok();
        let mut matches = self
            .tables()
            .into_iter()
            .filter(|t| Some(t.identifier) == by_id || t.name == wanted);
        let first = matches.next().ok_or_else(|| {
            Error::refused(Refusal::NotFound, format!("no table called '{wanted}'"))
        })?;
        // An id names one table; only a name can be ambiguous.
        if by_id.is_none() {
            let also: Vec<u64> = matches.map(|t| t.identifier).collect();
            if !also.is_empty() {
                let mut ids = vec![first.identifier];
                ids.extend(also);
                let list = ids
                    .iter()
                    .map(u64::to_string)
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(Error::refused(
                    Refusal::Ambiguous,
                    format!(
                        "'{wanted}' names {} tables ({list}) — write by identifier to say which",
                        ids.len()
                    ),
                ));
            }
        }
        Ok(first)
    }

    /// Refuse the write if any object it would rewrite carries version patches.
    fn refuse_if_patched(&self, objects: &[u64], where_: &str) -> Result<(), Error> {
        let patched = self.patched_objects();
        for &id in objects {
            if patched.iter().any(|&(patched, _)| patched == id) {
                return Err(Error::refused(
                    Refusal::Patched,
                    format!(
                        "{where_}: object {id} carries version patches, and rewriting it would \
                     leave them describing the cell as it used to be"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// What a write to one table needs that does not change from cell to cell.
    ///
    /// Reading it is the expensive half of a cell write — the model, the data
    /// store, the tile map and the side-table references — so a batch reads it
    /// once and every cell of every row shares it.
    fn table_site(&self, table: &crate::table::Table) -> Result<TableSite, Error> {
        let where_ = &table.name;
        let model = self.archive_of(table.model)?;
        let store = model
            .bytes(4)
            .and_then(crate::pb::decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the model has no data store")))?;
        let list = |field: u32| store.bytes(field).and_then(crate::table::reference);

        let tiles = store
            .bytes(3)
            .and_then(crate::pb::decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the data store has no tiles")))?;
        let tile_size = tiles.varint(2).unwrap_or(256) as usize;
        if tile_size == 0 {
            return Err(Error::Format(format!(
                "{where_}: the tile storage says a tile covers no rows at all"
            )));
        }
        let mut by_index = std::collections::BTreeMap::new();
        for value in tiles.all(1) {
            let Value::Bytes(raw) = value else { continue };
            let Some(entry) = crate::pb::decode_nested(raw) else {
                continue;
            };
            let (Some(index), Some(tile)) = (
                entry.varint(1),
                entry.bytes(2).and_then(crate::table::reference),
            ) else {
                continue;
            };
            by_index.insert(index as usize, tile);
        }

        Ok(TableSite {
            tile_size,
            tiles: by_index,
            strings: list(4),
            formats: list(22),
            controls: list(21),
            formulas: list(6),
            row_bucket: store
                .bytes(1)
                .and_then(crate::pb::decode_nested)
                .map(|storage| {
                    storage
                        .all(2)
                        .filter_map(|value| match value {
                            Value::Bytes(raw) => crate::table::reference(raw),
                            _ => None,
                        })
                        .collect::<Vec<u64>>()
                })
                .unwrap_or_default(),
            column_bucket: store.bytes(2).and_then(crate::table::reference),
        })
    }

    /// The tile covering `row`, refused if it carries version patches.
    fn tile_for_row(&self, site: &TableSite, row: usize, where_: &str) -> Result<u64, Error> {
        let tile = *site.tiles.get(&(row / site.tile_size)).ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{where_}: no tile covers row {row}"),
            )
        })?;
        if self.patched_objects().iter().any(|&(id, _)| id == tile) {
            return Err(Error::refused(
                Refusal::Patched,
                format!(
                    "{where_}: tile {tile} carries version patches, and rewriting it would \
                 leave them describing the cell as it used to be"
                ),
            ));
        }
        Ok(tile)
    }

    /// Where a cell's bytes live, and which side tables reach it.
    fn cell_site(
        &self,
        table: &crate::table::Table,
        row: usize,
        column: usize,
    ) -> Result<CellSite, Error> {
        let where_ = format!(
            "{} {}",
            table.name,
            crate::table::CellRef::Index(row, column)
        );
        let site = self.table_site(table)?;
        let tile = self.tile_for_row(&site, row, &where_)?;
        let archive = self.archive_of(tile)?;
        let tile_row = row % site.tile_size;
        let in_tile = TileIndex::of(&archive).row(&archive, tile_row, &where_)?;
        if column >= in_tile.records.len() {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!(
                    "{where_}: the row's offset array names only {} columns",
                    in_tile.records.len()
                ),
            ));
        }
        let record = in_tile.records[column].clone();
        Ok(CellSite {
            tile,
            tile_row: in_tile.position,
            tile_row_index: tile_row,
            fresh_row: in_tile.fresh,
            wide: in_tile.wide,
            records: in_tile.records,
            record,
            column,
            // The *absolute* row: the header bucket counts cells by the row's
            // index in the table, not by its index in the tile.
            row,
            formulas: site.formulas,
            row_bucket: site.row_bucket,
            column_bucket: site.column_bucket,
        })
    }

    /// Plan the new record and the reference-count changes it implies.
    ///
    /// **Pure**: it reads the document to decide what to do, but writes nothing.
    /// Every fallible decision — the string key, the donor format — is made here,
    /// so that [`Document::set_cell`] can prove the whole write will succeed
    /// before applying any of it. That is what keeps a refused write from leaving
    /// a half-maintained side table: an interned string with no cell, or a format
    /// entry whose refcount no longer matches, either of which `audit` rejects in
    /// a document this crate wrote itself.
    fn rewrite_record(
        &self,
        cache: &mut ListCache,
        table: &crate::table::Table,
        site: &TableSite,
        old: crate::table::CellRecord,
        value: &CellValue,
        where_: &str,
    ) -> Result<CellWrite, Error> {
        use crate::table::FormatSlot;

        let mut record = old.clone();
        record.version = 5;
        record.decimal = None;
        record.double = None;
        record.seconds = None;
        record.string_id = None;
        record.rich_id = None;
        record.formula_error_id = None;

        let slot = match value {
            CellValue::Empty => None,
            CellValue::Text(_) => Some(FormatSlot::Text),
            CellValue::Number(_) => Some(FormatSlot::Number),
            CellValue::Currency(_) => Some(FormatSlot::Currency),
            CellValue::Bool(_) => Some(FormatSlot::Boolean),
            CellValue::Date(_) => Some(FormatSlot::Date),
            CellValue::Duration(_) => Some(FormatSlot::Duration),
            _ => unreachable!("set_cell rejected every other value"),
        };
        record.cell_type = match value {
            CellValue::Empty => cell_type::EMPTY,
            CellValue::Text(_) => cell_type::TEXT,
            CellValue::Number(_) => cell_type::NUMBER,
            CellValue::Currency(_) => cell_type::CURRENCY,
            CellValue::Bool(_) => cell_type::BOOL,
            CellValue::Date(_) => cell_type::DATE,
            CellValue::Duration(_) => cell_type::DURATION,
            _ => unreachable!("set_cell rejected every other value"),
        };
        // Byte 7 is `0x08` on every currency cell in the corpus and `0x00`
        // everywhere else, and nothing here knows what it means — so it is
        // written as the app writes it and cleared when the cell stops being
        // money. Measured on a cell this crate wrote and Numbers converted:
        // `extras` went from `0x0000` to `0x0802`.
        record.extras = match value {
            CellValue::Currency(_) => record.extras | CURRENCY_BYTE_SEVEN,
            _ => record.extras & !CURRENCY_BYTE_SEVEN,
        };
        match value {
            CellValue::Number(decimal) | CellValue::Currency(decimal) => {
                record.decimal = Some(*decimal)
            }
            CellValue::Bool(flag) => record.double = Some(if *flag { 1.0 } else { 0.0 }),
            CellValue::Duration(seconds) => record.double = Some(*seconds),
            CellValue::Date(seconds) => record.seconds = Some(*seconds),
            _ => {}
        }

        let mut mutations = Vec::new();

        // The string table first, so that rewriting a cell with the text it
        // already held reuses its entry — its key is taken and given up in the
        // same breath, and the net is nothing at all.
        if let CellValue::Text(text) = value {
            let list = site.strings.ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!("{where_}: the table has no string list"),
                )
            })?;
            let (key, fresh) = self.planned_string_key(cache, list, text, where_)?;
            record.string_id = Some(key);
            mutations.push(ListMutation::Intern {
                list,
                text: text.clone(),
                key,
                fresh,
            });
        }
        if let Some(key) = old.string_id {
            let list = site.strings.ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!("{where_}: the table has no string list"),
                )
            })?;
            mutations.push(ListMutation::Release { list, key });
        }

        // An emptied cell keeps no record at all — the app deletes it, and the
        // covered half of a merge shows the same thing. The deleted record was
        // holding references in *every* slot it carried a key for, not just the
        // one the format kind named, and each one has to be given back or the
        // list's refcount stays too high — which is exactly a document this
        // crate's own `audit` then rejects. Its control-spec reference goes the
        // same way.
        if slot.is_none() {
            if let Some(list) = site.formats {
                for slot in ALL_SLOTS {
                    if let Some(key) = old.format_id_in(slot) {
                        mutations.push(ListMutation::Release { list, key });
                    }
                }
            }
            if let (Some(list), Some(key)) = (site.controls, old.control_id) {
                mutations.push(ListMutation::Release { list, key });
            }
            return Ok(CellWrite {
                record: None,
                mutations,
            });
        }
        let slot = slot.expect("checked above");

        if record.format_id_in(slot).is_none() {
            let list = site.formats.ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!("{where_}: the table has no format list"),
                )
            })?;
            let donor = match cache.donors.iter().find(|(at, _)| *at == slot) {
                Some((_, donor)) => *donor,
                None => {
                    let found = self.donor_format(cache, table, list, slot);
                    cache.donors.push((slot, found));
                    found
                }
            };
            let donor = donor.ok_or_else(|| {
                Error::refused(
                    Refusal::NoDonorFormat,
                    format!(
                        "{where_}: no cell in this table carries a {slot:?} format to copy, \
                     and one invented here would be a format the document never defined"
                    ),
                )
            })?;
            mutations.push(ListMutation::Retain { list, key: donor });
            record.set_format_id_in(slot, Some(donor));

            // The slot the cell used to be in loses its key, and byte 6 loses
            // the claim that the user chose that format. Observed: a text cell
            // written over with a number in Numbers 15.3.1 comes back carrying
            // the number key only.
            if let Some(previous) = old.current_format() {
                if previous != slot {
                    if let Some(key) = old.format_id_in(previous) {
                        mutations.push(ListMutation::Release { list, key });
                    }
                    record.set_format_id_in(previous, None);
                    record.forget_explicit_format(previous);
                }
            }
        }
        record.format_kind = Some(crate::table::format_kind_of(slot));
        // Money is a format the caller *chose*, and byte 6 is where that is
        // said: the app's own conversion set the currency bit, and without it
        // this crate's reader — and the app's inspector — call the cell
        // automatic. Every other value type leaves byte 6 as it found it,
        // because writing a number into a cell is not choosing a format for it.
        if matches!(value, CellValue::Currency(_)) {
            record.set_explicit_format(crate::table::FormatSlot::Currency);
        }
        Ok(CellWrite {
            record: Some(record),
            mutations,
        })
    }

    /// The format key a cell of this slot borrows, or `None` if there is none
    /// to borrow honestly.
    ///
    /// A cell whose format nobody chose is the better loan: its key names the
    /// plain format for the slot rather than, say, the percentage a neighbour
    /// was given — so the smallest `(chosen, key)` wins.
    ///
    /// When no cell of the table carries a format for the slot at all, the
    /// answer depends on what the table looks like otherwise. That is every
    /// cell of a table `Document::new` just made, and it is also the header of
    /// a currency column being turned into a number — and those two want
    /// opposite answers. What separates them is whether the table shows any
    /// sign of formatting: a table whose every format is *automatic* has
    /// nothing to contradict, so the automatic entry is borrowed; a table
    /// holding a currency format somewhere has been formatted by someone, and
    /// guessing which of their formats this value wants is not this crate's to
    /// do.
    fn donor_format(
        &self,
        cache: &mut ListCache,
        table: &crate::table::Table,
        list: u64,
        slot: crate::table::FormatSlot,
    ) -> Option<u32> {
        table
            .cells()
            .iter()
            .filter_map(|cell| {
                cell.record
                    .format_id_in(slot)
                    .map(|key| (cell.record.explicit_format().is_some(), key))
            })
            .min()
            .map(|(_, key)| key)
            .or_else(|| self.automatic_format(cache, list, table))
    }

    /// The key of the automatic format in a format list, if it holds one.
    ///
    /// `format_type` 260 is the one format that reads sensibly whatever the
    /// value is — it is what Numbers itself gives a cell nobody has formatted,
    /// text cells included. Nothing else in the list will do: a currency format
    /// borrowed for a number is a format the caller did not ask for, and one
    /// this crate would then have written into a document silently.
    fn automatic_format(
        &self,
        cache: &mut ListCache,
        list: u64,
        table: &crate::table::Table,
    ) -> Option<u32> {
        let entries = crate::table::DataList::decode(cache.read(self, list).ok()?).entries;
        let format_type = |key: u32| {
            entries
                .get(&key)
                .and_then(|entry| entry.format.as_ref())
                .and_then(|format| format.varint(1))
        };
        // Any format a cell actually uses that is not automatic is a decision
        // someone made about this table, and the end of this.
        let formatted = table.cells().iter().any(|cell| {
            ALL_SLOTS.iter().any(|slot| {
                cell.record
                    .format_id_in(*slot)
                    .and_then(format_type)
                    .is_some_and(|kind| kind != AUTOMATIC_FORMAT)
            })
        });
        if formatted {
            return None;
        }
        entries
            .iter()
            .find(|(_, entry)| {
                entry
                    .format
                    .as_ref()
                    .and_then(|format| format.varint(1))
                    .is_some_and(|kind| kind == AUTOMATIC_FORMAT)
            })
            .map(|(key, _)| *key)
    }

    /// Apply one planned reference-count change, in the cache.
    ///
    /// Nothing reaches the document until [`ListCache::flush`], which is what
    /// keeps a batch's hundredth string from re-encoding the list a hundred
    /// times — and what leaves the lists untouched when a later cell refuses.
    fn apply_mutation(&self, cache: &mut ListCache, mutation: &ListMutation) -> Result<(), Error> {
        match mutation {
            ListMutation::Intern {
                list,
                text,
                key,
                fresh,
            } => self.commit_intern(cache, *list, text, *key, *fresh),
            ListMutation::Retain { list, key } => self.step_list_entry(cache, *list, *key, 1),
            ListMutation::Release { list, key } => self.step_list_entry(cache, *list, *key, -1),
        }
    }

    /// Lay the row's records back out, without touching the tile.
    ///
    /// The last fallible step of a row's write: a row grown past what a 16-bit
    /// offset can address fails here, before the tile is rewritten.
    fn stage_row(
        &self,
        site: &CellSite,
        appeared: Vec<usize>,
        disappeared: Vec<usize>,
    ) -> Result<StagedRow, Error> {
        let (buffer, offsets) = crate::table::encode_row(&site.records, site.wide)
            .map_err(|e| Error::Format(format!("{}: {e}", site.row)))?;
        let cells = site.records.iter().filter(|r| r.is_some()).count() as u64;
        Ok(StagedRow {
            cells,
            buffer,
            offsets,
            appeared,
            disappeared,
        })
    }

    /// Put the staged row back into its tile, and keep the cell counts.
    ///
    /// The infallible half: everything it reaches for — the tile, the header
    /// buckets — was decoded while the site was found, so this does not refuse a
    /// write that `stage_store` and the list mutations have already committed to.
    fn write_store(&mut self, site: &CellSite, staged: StagedRow) -> Result<(), Error> {
        let mut tile = self.archive_of(site.tile)?;
        if site.fresh_row {
            // The row is brought into being in the shape `Document::new` writes
            // and all three apps open: the pre-BNC pair (3, 4) present and
            // empty, because they are `required` in the schema and meaningless
            // in this storage version, and `numrows` counting one more row.
            let mut info = Message::default();
            info.set_in_order(1, Value::Varint(site.tile_row_index as u64));
            info.set_in_order(2, Value::Varint(staged.cells));
            info.set_in_order(3, Value::Bytes(Vec::new()));
            info.set_in_order(4, Value::Bytes(staged.offsets.clone()));
            info.set_in_order(5, Value::Varint(5));
            info.set_in_order(6, Value::Bytes(staged.buffer));
            info.set_in_order(7, Value::Bytes(staged.offsets));
            tile.fields.insert(
                site.tile_row,
                crate::pb::Field {
                    number: 5,
                    value: Value::Bytes(info.encode()),
                },
            );
            let rows = tile.varint(4).unwrap_or(0) + 1;
            tile.set(4, Value::Varint(rows));
        } else {
            let Value::Bytes(raw) = &tile.fields[site.tile_row].value else {
                unreachable!("cell_site found this field by matching on Bytes")
            };
            let mut info = crate::pb::decode_nested(raw)
                .ok_or_else(|| Error::Format("row storage does not decode".to_string()))?;
            info.set(2, Value::Varint(staged.cells));
            info.set(6, Value::Bytes(staged.buffer));
            info.set(7, Value::Bytes(staged.offsets));
            tile.fields[site.tile_row].value = Value::Bytes(info.encode());
        }
        self.set_archive(site.tile, &tile)?;

        let step = staged.appeared.len() as i64 - staged.disappeared.len() as i64;
        if step != 0 {
            let bucket = site
                .row_bucket
                .iter()
                .copied()
                .find(|&id| self.bucket_has(id, site.row))
                .or_else(|| site.row_bucket.first().copied());
            if let Some(bucket) = bucket {
                self.step_bucket_count(bucket, site.row, step)?;
            }
        }
        if let Some(bucket) = site.column_bucket {
            for column in &staged.appeared {
                self.step_bucket_count(bucket, *column, 1)?;
            }
            for column in &staged.disappeared {
                self.step_bucket_count(bucket, *column, -1)?;
            }
        }
        Ok(())
    }

    /// Every row or column index this `TST.HeaderStorageBucket` has an entry
    /// for — read once, so a batch does not decode the bucket per row.
    fn bucket_indices(&self, bucket: u64) -> std::collections::BTreeSet<usize> {
        let Ok(archive) = self.archive_of(bucket) else {
            return Default::default();
        };
        archive
            .all(2)
            .filter_map(|value| match value {
                Value::Bytes(raw) => crate::pb::decode_nested(raw)?.varint(1),
                _ => None,
            })
            .map(|index| index as usize)
            .collect()
    }

    /// Does this `TST.HeaderStorageBucket` have an entry for `index`?
    fn bucket_has(&self, bucket: u64, index: usize) -> bool {
        self.archive_of(bucket).is_ok_and(|archive| {
            archive.all(2).any(|value| match value {
                Value::Bytes(raw) => crate::pb::decode_nested(raw)
                    .and_then(|entry| entry.varint(1))
                    .is_some_and(|at| at as usize == index),
                _ => false,
            })
        })
    }

    /// Move a row's or column's stored cell count, adding the entry if the row
    /// or column had none — which is what an all-empty column has.
    fn step_bucket_count(&mut self, bucket: u64, index: usize, step: i64) -> Result<(), Error> {
        let mut one = std::collections::BTreeMap::new();
        one.insert(index, step);
        self.step_bucket_counts(bucket, &one)
    }

    /// The same, for many indices at once: one decode and one re-encode of the
    /// bucket however many rows or columns moved.
    fn step_bucket_counts(
        &mut self,
        bucket: u64,
        deltas: &std::collections::BTreeMap<usize, i64>,
    ) -> Result<(), Error> {
        if deltas.values().all(|&delta| delta == 0) {
            return Ok(());
        }
        let mut archive = self.archive_of(bucket)?;
        let mut left: std::collections::BTreeMap<usize, i64> = deltas.clone();
        for field in archive.fields.iter_mut() {
            if field.number != 2 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let Some(mut entry) = crate::pb::decode_nested(raw) else {
                continue;
            };
            let index = entry.varint(1).unwrap_or(u64::MAX) as usize;
            let Some(step) = left.remove(&index) else {
                continue;
            };
            if step == 0 {
                continue;
            }
            let count = entry.varint(4).unwrap_or(0) as i64 + step;
            entry.set_in_order(4, Value::Varint(count.max(0) as u64));
            field.value = Value::Bytes(entry.encode());
        }
        // A row or column the bucket has no entry for is one that held no cells
        // — an all-empty column has none — so a cell appearing there adds one.
        for (index, step) in left {
            if step <= 0 {
                continue;
            }
            let mut entry = Message::default();
            entry.set_in_order(1, Value::Varint(index as u64));
            entry.set_in_order(2, Value::Fixed32(0f32.to_le_bytes()));
            entry.set_in_order(3, Value::Varint(0));
            entry.set_in_order(4, Value::Varint(step as u64));
            archive.append_in_order(2, Value::Bytes(entry.encode()));
        }
        self.set_archive(bucket, &archive)
    }

    /// Decide the key interning `text` will use — without writing anything.
    ///
    /// Returns the key and whether it is a fresh entry. Numbers hands out the
    /// **smallest free key** and keeps `nextListID` as a high-water mark that
    /// only rises — observed by watching keys 6 and 13 be freed and then handed
    /// out again while the field stayed at 15. This takes the simpler half of
    /// that: a key at or above the mark can collide with nothing.
    ///
    /// A `TableDataList` key is a **32-bit** field, and key 0 means "absent". A
    /// list whose mark has reached the 32-bit ceiling can mint no more keys
    /// without either wrapping to 0 (silently losing the value) or colliding, so
    /// this refuses rather than truncates — the write fails cleanly and the list
    /// is left as it was.
    fn planned_string_key(
        &self,
        cache: &mut ListCache,
        list: u64,
        text: &str,
        where_: &str,
    ) -> Result<(u32, bool), Error> {
        let index = cache.string_index(self, list)?;
        if let Some(&key) = index.by_text.get(text.as_bytes()) {
            return Ok((key, false));
        }
        // Computed in 64 bits so the 32-bit ceiling is a refusal and not a wrap.
        let candidate = index.next;
        let key = u32::try_from(candidate).map_err(|_| {
            Error::Format(format!(
                "{where_}: the string list's next key {candidate} does not fit the 32 bits a \
                 TableDataList key has, so a new string cannot be interned"
            ))
        })?;
        Ok((key, true))
    }

    /// Apply an interning decided by [`Document::planned_string_key`].
    ///
    /// `fresh` says which branch the plan chose: bump an existing entry's
    /// refcount, or append a new one under `key` and raise the mark past it.
    fn commit_intern(
        &self,
        cache: &mut ListCache,
        list: u64,
        text: &str,
        key: u32,
        fresh: bool,
    ) -> Result<(), Error> {
        let archive = cache.write(self, list)?;
        if !fresh {
            for field in archive.fields.iter_mut() {
                if field.number != 3 {
                    continue;
                }
                let Value::Bytes(raw) = &field.value else {
                    continue;
                };
                let Some(mut entry) = crate::pb::decode_nested(raw) else {
                    continue;
                };
                if entry.bytes(3) != Some(text.as_bytes()) {
                    continue;
                }
                entry.set_in_order(2, Value::Varint(entry.varint(2).unwrap_or(0) + 1));
                field.value = Value::Bytes(entry.encode());
                return Ok(());
            }
            // The entry planning found is gone — impossible between plan and
            // apply, since nothing touches the list in between.
            return Err(Error::Format(format!(
                "string entry for {text:?} vanished before it could be interned"
            )));
        }

        let mut entry = Message::default();
        entry.set_in_order(1, Value::Varint(u64::from(key)));
        entry.set_in_order(2, Value::Varint(1));
        entry.set_in_order(3, Value::Bytes(text.as_bytes().to_vec()));
        archive.set_in_order(2, Value::Varint(u64::from(key) + 1));
        archive.append_in_order(3, Value::Bytes(entry.encode()));
        if let Some(index) = cache.strings.get_mut(&list) {
            index.by_text.insert(text.as_bytes().to_vec(), key);
            index.next = index.next.max(u64::from(key) + 1);
        }
        Ok(())
    }

    /// Move a `TableDataList` entry's reference count, dropping it at zero.
    ///
    /// Dropping is what the app does: emptying the one cell that held a string
    /// removed its entry outright, and the key was later handed out again to a
    /// different string.
    ///
    /// Retaining an entry that is not there is a *refusal*, not a no-op: the
    /// only way to reach one is a plan made against a stale reading of the
    /// list, and writing the cell anyway would leave it naming a key nothing
    /// defines — which is precisely what `Table::audit` reports and what this
    /// crate must not produce.
    fn step_list_entry(
        &self,
        cache: &mut ListCache,
        list: u64,
        key: u32,
        step: i64,
    ) -> Result<(), Error> {
        let archive = cache.write(self, list)?;
        let mut drop = None;
        let mut found = false;
        for (at, field) in archive.fields.iter_mut().enumerate() {
            if field.number != 3 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let Some(mut entry) = crate::pb::decode_nested(raw) else {
                continue;
            };
            if entry.varint(1).unwrap_or(0) as u32 != key {
                continue;
            }
            found = true;
            let count = entry.varint(2).unwrap_or(0) as i64 + step;
            if count <= 0 {
                drop = Some(at);
            } else {
                entry.set_in_order(2, Value::Varint(count as u64));
                field.value = Value::Bytes(entry.encode());
            }
            break;
        }
        if !found && step > 0 {
            return Err(Error::Format(format!(
                "list {list} has no entry {key} to take a reference to"
            )));
        }
        if let Some(at) = drop {
            // The entry goes, so the index must forget the text — a later cell
            // in the same batch given that text interns it afresh rather than
            // naming a key nothing defines any more.
            let gone = match &archive.fields[at].value {
                Value::Bytes(raw) => crate::pb::decode_nested(raw)
                    .and_then(|entry| entry.bytes(3).map(<[u8]>::to_vec)),
                _ => None,
            };
            archive.fields.remove(at);
            if let (Some(text), Some(index)) = (gone, cache.strings.get_mut(&list)) {
                index.by_text.remove(&text);
            }
        }
        Ok(())
    }

    // -- working with one text storage ---------------------------------------

    /// A handle for editing one text storage, without carrying its identifier
    /// around.
    ///
    /// Text is where Pages and Keynote live, and every text call in this crate
    /// takes a `TSWP.StorageArchive` identifier — a number a caller has to find
    /// first and then thread through every edit. This resolves it once:
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let mut doc = iwork::Document::open("Report.pages")?;
    /// let mut body = doc.body_mut()?;
    /// body.append("A new paragraph")?;
    /// body.insert(12, "eingeschoben ")?;
    /// println!("{}", body.read()?);
    /// # Ok(()) }
    /// ```
    ///
    /// Indices are **UTF-16 code units**, which is what iWork counts text in
    /// and what every range here means — the one thing a handle cannot make go
    /// away.
    pub fn text_mut(&mut self, storage: u64) -> Result<TextHandle<'_>, Error> {
        // Resolve now, so a wrong identifier is an error here rather than on
        // the first edit.
        self.storage_text(storage)?;
        Ok(TextHandle {
            document: self,
            storage,
        })
    }

    /// The document body of a Pages document, as a handle.
    ///
    /// Pages only: a Numbers or Keynote document has no body, and says so.
    pub fn body_mut(&mut self) -> Result<TextHandle<'_>, Error> {
        self.refuse_if_body_is_not_drawn()?;
        let storage = self.body_storage().ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!(
                    "a {} document has no body text — a Pages document does, and a deck's words \
                 are on its slides",
                    self.kind().as_str()
                ),
            )
        })?;
        self.text_mut(storage)
    }

    // -- working with one slide ----------------------------------------------

    /// A handle for one slide of a deck, by position or by identifier.
    ///
    /// **A slide is not a page with a title slot.** What a slide can hold is
    /// decided by the *layout* it is built on: the placeholders that layout
    /// defines — title, body, slide number, object — and any drawables the
    /// slide owns besides. So [`SlideHandle::title`] writes the title
    /// placeholder when the layout has one and refuses by name when it does
    /// not, rather than inventing a text box and calling it a title.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let mut doc = iwork::Document::open("Talk.key")?;
    /// let mut slide = doc.slide_mut(0)?;
    /// slide.title("Quarterly review")?;
    /// slide.notes("Remember the numbers are provisional")?;
    /// slide.transition("dissolve")?;
    /// # Ok(()) }
    /// ```
    pub fn slide_mut(
        &mut self,
        slide: impl Into<crate::keynote::SlideRef>,
    ) -> Result<SlideHandle<'_>, Error> {
        let wanted = slide.into();
        let slides = self.slides();
        if slides.is_empty() {
            return Err(Error::refused(
                Refusal::NotFound,
                format!("a {} document has no slides", self.kind().as_str()),
            ));
        }
        let found = match wanted {
            crate::keynote::SlideRef::Index(index) => slides.get(index).map(|s| s.identifier),
            crate::keynote::SlideRef::Identifier(id) => slides
                .iter()
                .find(|s| s.identifier == id || s.node == id)
                .map(|s| s.identifier),
        };
        let identifier = found.ok_or_else(|| {
            Error::refused(
                Refusal::OutOfBounds,
                format!("no slide {wanted} — the deck has {} of them", slides.len()),
            )
        })?;
        Ok(SlideHandle {
            document: self,
            slide: identifier,
        })
    }

    // -- working with one table ----------------------------------------------

    /// A handle for writing to one table, by any of the ways a table can be
    /// named.
    ///
    /// Everything it does, the methods on [`Document`] already do; what it adds
    /// is that the table is resolved **once** and named in one place, and that
    /// a cell can be given as `"B3"`:
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// let mut doc = iwork::Document::new_spreadsheet("Sales", "Q1", 9, 3)?;
    /// let mut q1 = doc.table_mut("Q1")?;
    /// q1.set("A1", "Region")?;
    /// q1.set("B1", 1_240)?;
    /// q1.set("C1", 184_300.0)?;
    /// q1.formula("B9", "=SUM(B1:B8)", 1_240)?;
    /// q1.column_width(0, Some(210.0))?;
    /// # Ok(()) }
    /// ```
    ///
    /// **A table is addressed within its sheet when it has to be.** A Numbers
    /// sheet holds any number of tables, and two sheets may hold tables of the
    /// same name — `numbers-pivot.numbers` has a `Sales` on each of its two
    /// sheets — so a bare name that is not unique is refused, and
    /// `("Pivot Table Basics", "Sales")` or the identifier says which.
    ///
    /// The handle stores no cells, so it cannot go stale: every call reads what
    /// is there now. That also means reading cell by cell re-reads the table
    /// each time — [`TableHandle::read`] hands back a snapshot for that.
    pub fn table_mut<'a>(
        &'a mut self,
        table: impl Into<crate::table::TableRef<'a>>,
    ) -> Result<TableHandle<'a>, Error> {
        let wanted = table.into();
        let identifier = self.resolve_table(&wanted)?;
        Ok(TableHandle {
            document: self,
            table: identifier.to_string(),
        })
    }

    /// Which table a [`crate::table::TableRef`] names, refusing an ambiguous
    /// one by name.
    fn resolve_table(&self, wanted: &crate::table::TableRef<'_>) -> Result<u64, Error> {
        use crate::table::TableRef;
        let tables = self.tables();
        let matches: Vec<&crate::table::Table> = match wanted {
            TableRef::Identifier(id) => tables.iter().filter(|t| t.identifier == *id).collect(),
            TableRef::Name(name) => tables.iter().filter(|t| t.name == *name).collect(),
            TableRef::On(sheet, name) => tables
                .iter()
                .filter(|t| t.name == *name && t.sheet.as_deref() == Some(*sheet))
                .collect(),
        };
        match matches.len() {
            1 => Ok(matches[0].identifier),
            0 => Err(Error::Format(format!("no table {wanted}"))),
            _ => {
                let where_ = matches
                    .iter()
                    .map(|t| match &t.sheet {
                        Some(sheet) => format!("{} on {sheet:?}", t.identifier),
                        None => t.identifier.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(", ");
                Err(Error::refused(
                    Refusal::Ambiguous,
                    format!(
                        "{wanted} names {} tables ({where_}) — a sheet holds any number of tables \
                     and two sheets may hold one name, so say which: (\"sheet\", \"table\") or \
                     the identifier",
                        matches.len()
                    ),
                ))
            }
        }
    }

    /// Make a chart follow a table — the *mediator*, which is what turns a
    /// picture of some numbers into a chart of a table.
    ///
    /// The grid the chart draws stays what it was: in Numbers it is a **cache**
    /// of what the mediator's formulas last evaluated to. What this adds is the
    /// formulas — one per series and per label, each a reference to the table
    /// wrapped in function 175 — and the owner that makes the calculation
    /// engine know about them. See [`crate::chart::bind_chart`].
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let mut doc = iwork::Document::open("Budget.numbers")?;
    /// use iwork::chart::ChartBinding;
    /// doc.bind_chart(
    ///     905245,
    ///     "Umsatz",
    ///     &ChartBinding {
    ///         series: vec!["B2:B13".into(), "C2:C13".into()],
    ///         row_labels: vec!["A2:A13".into()],
    ///         column_labels: vec!["B1".into(), "C1".into()],
    ///         series_by_row: false,
    ///     },
    /// )?;
    /// # Ok(()) }
    /// ```
    pub fn bind_chart(
        &mut self,
        chart: u64,
        table: &str,
        binding: &crate::chart::ChartBinding,
    ) -> Result<u64, Error> {
        let table = self.table_for_write(table)?;
        crate::chart::bind_chart(self, chart, &table, binding)
    }

    // -- sizes and formats ---------------------------------------------------

    /// Set a column's width in points, or put it back to the table's default.
    ///
    /// A row or column's size lives in its `TST.HeaderStorageBucket` entry, and
    /// **a literal `0` there means "the table's default"** — which is what
    /// every row and column of the corpus carries except the one column a user
    /// dragged to 150 and the one row dragged to 40. `None` writes that zero.
    ///
    /// The table's *drawable* is not touched, and measurement says it must not
    /// be: every table in the corpus has a frame 494pt wide whatever its
    /// columns — three of them, seven of them, one of them 150pt — so the app
    /// lays the table out from the column widths rather than from the frame.
    pub fn set_column_width(
        &mut self,
        wanted: &str,
        column: usize,
        points: Option<f32>,
    ) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        if column >= table.columns {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!(
                    "{} c{column}: the table has {} column(s)",
                    table.name, table.columns
                ),
            ));
        }
        let site = self.table_site(&table)?;
        let bucket = site.column_bucket.ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{}: the table has no column header", table.name),
            )
        })?;
        self.refuse_if_patched(&[bucket], &format!("{} c{column}", table.name))?;
        self.set_bucket_size(bucket, column, points.unwrap_or(0.0))
    }

    /// Set a row's height in points, or put it back to the table's default.
    ///
    /// The row half of [`Document::set_column_width`], and the one asymmetry is
    /// Apple's: a table's rows are counted across a *list* of buckets where its
    /// columns share one, so the row's own bucket has to be found first.
    pub fn set_row_height(
        &mut self,
        wanted: &str,
        row: usize,
        points: Option<f32>,
    ) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        if row >= table.rows {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!("{} r{row}: the table has {} row(s)", table.name, table.rows),
            ));
        }
        let site = self.table_site(&table)?;
        let bucket = site
            .row_bucket
            .iter()
            .copied()
            .find(|&id| self.bucket_has(id, row))
            .or_else(|| site.row_bucket.first().copied())
            .ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!("{}: the table has no row header", table.name),
                )
            })?;
        self.refuse_if_patched(&[bucket], &format!("{} r{row}", table.name))?;
        self.set_bucket_size(bucket, row, points.unwrap_or(0.0))
    }

    /// Write field 2 of one `HeaderStorageBucket` entry, adding the entry if the
    /// row or column has none — which is what a row holding no cells has.
    fn set_bucket_size(&mut self, bucket: u64, index: usize, points: f32) -> Result<(), Error> {
        if !points.is_finite() || points < 0.0 {
            return Err(Error::refused(
                Refusal::NotACell,
                format!("{points} is not a size a row or column can have"),
            ));
        }
        let mut archive = self.archive_of(bucket)?;
        for field in archive.fields.iter_mut() {
            if field.number != 2 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let Some(mut entry) = crate::pb::decode_nested(raw) else {
                continue;
            };
            if entry.varint(1).unwrap_or(u64::MAX) as usize != index {
                continue;
            }
            entry.set_in_order(2, Value::Fixed32(points.to_le_bytes()));
            field.value = Value::Bytes(entry.encode());
            return self.set_archive(bucket, &archive);
        }
        let mut entry = Message::default();
        entry.set_in_order(1, Value::Varint(index as u64));
        entry.set_in_order(2, Value::Fixed32(points.to_le_bytes()));
        entry.set_in_order(3, Value::Varint(0));
        entry.set_in_order(4, Value::Varint(0));
        archive.append_in_order(2, Value::Bytes(entry.encode()));
        self.set_archive(bucket, &archive)
    }

    /// Give cells a data format — the number of decimals, a currency, a date
    /// pattern.
    ///
    /// A format is a `TSK.FormatStructArchive` interned in the table's FORMAT
    /// list, and the archives written here are the ones Numbers wrote for the
    /// same formats in `numbers-formats.numbers`, field for field: a number is
    /// `{1: 256, 2: decimals, 4: 0, 5: 0}`, a currency the same with `{3: code,
    /// 6: 0}`, a date `{1: 261, 14: pattern}`. `253` is the decimal count that
    /// means "as many as it takes", which is what the app writes for a format
    /// nobody has pinned.
    ///
    /// Three things travel with the format, and leaving any of them out gives a
    /// cell the app draws the old way:
    ///
    /// 1. the key goes in the **slot the value type uses** — a percentage is a
    ///    number format, a currency is its own slot;
    /// 2. `format_kind` (the `0x1000` payload) names that slot, or the format
    ///    is inert;
    /// 3. **byte 6's bit is set**, which is the only thing separating a cell
    ///    Numbers calls "number" from one it calls "automatic".
    ///
    /// Refused by name: a format whose family the cell's value cannot use (a
    /// date pattern on a text cell), an empty cell, a cell holding rich text,
    /// and a table with no format list.
    pub fn set_format(
        &mut self,
        wanted: &str,
        cells: impl IntoIterator<Item = (usize, usize)>,
        format: &crate::table::Format,
    ) -> Result<usize, Error> {
        let table = self.table_for_write(wanted)?;
        let site = self.table_site(&table)?;
        let list = site.formats.ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{}: the table has no format list", table.name),
            )
        })?;
        let archive = format.archive()?;

        let cells: Vec<(usize, usize)> = cells.into_iter().collect();
        let mut slots = Vec::with_capacity(cells.len());
        for &(row, column) in &cells {
            let where_ = format!(
                "{} {}",
                table.name,
                crate::table::CellRef::Index(row, column)
            );
            if row >= table.rows || column >= table.columns {
                return Err(Error::refused(
                    Refusal::OutOfBounds,
                    format!("{where_}: the table is {}×{}", table.rows, table.columns),
                ));
            }
            let cell = table.cell(row, column).ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!(
                        "{where_}: is empty, and a format on nothing is not something the app \
                     writes — give the cell a value first"
                    ),
                )
            })?;
            if cell.record.cell_type == cell_type::RICH_TEXT {
                return Err(Error::refused(
                    Refusal::HoldsRichText,
                    format!(
                        "{where_}: holds rich text, whose formatting is its storage's business"
                    ),
                ));
            }
            // The slot the cell's value actually uses — `format_kind` when it
            // has one, and what its type can use otherwise.
            let slot = cell
                .record
                .current_format()
                .or_else(|| crate::table::Format::slot_of(cell.record.cell_type))
                .ok_or_else(|| {
                    Error::Format(format!(
                        "{where_}: holds {} and this crate cannot say which format slot that \
                         uses",
                        table.value(row, column).kind()
                    ))
                })?;
            if !format.suits(slot) {
                return Err(Error::refused(
                    Refusal::WrongSlot,
                    format!(
                        "{where_}: holds {} and uses the {slot:?} format slot, so a {} format \
                     would sit in the file and never be drawn — the slot follows the value's \
                     type, and changing it is a value write",
                        table.value(row, column).kind(),
                        format.name()
                    ),
                ));
            }
            slots.push(slot);
        }
        self.refuse_if_patched(&[list], &table.name)?;

        let restore = self.streams.clone();
        match self.apply_format(&table, &site, &cells, &slots, list, archive) {
            Ok(written) => Ok(written),
            Err(e) => {
                self.streams = restore;
                Err(e)
            }
        }
    }

    /// The applying half of [`Document::set_format`], behind the snapshot.
    fn apply_format(
        &mut self,
        table: &crate::table::Table,
        site: &TableSite,
        cells: &[(usize, usize)],
        slots: &[crate::table::FormatSlot],
        list: u64,
        archive: Message,
    ) -> Result<usize, Error> {
        let mut cache = ListCache::default();
        // One entry however many cells take it: a format is interned exactly as
        // a string is, and an identical archive already in the list is reused
        // rather than written twice.
        let payload = archive.encode();
        let key = match self.matching_format(&mut cache, list, &payload)? {
            Some(key) => key,
            // No references yet: every cell of the batch takes its own below,
            // including the first — which is what keeps the count and the cells
            // agreeing, and what `Table::audit` checks.
            None => self.define_list_entry(&mut cache, list, 6, payload.clone(), 0)?,
        };

        let mut written = 0;
        for (&(row, column), &slot) in cells.iter().zip(slots) {
            let where_ = format!(
                "{} {}",
                table.name,
                crate::table::CellRef::Index(row, column)
            );
            let tile = self.tile_for_row(site, row, &where_)?;
            let mut tile_archive = self.archive_of(tile)?;
            let index = TileIndex::of(&tile_archive);
            let mut in_tile = index.row(&tile_archive, row % site.tile_size, &where_)?;
            let Some(bytes) = in_tile.records.get(column).cloned().flatten() else {
                return Err(Error::Format(format!("{where_}: has no stored cell")));
            };
            let mut record = crate::table::decode_cell(&bytes)
                .map_err(|e| Error::Format(format!("{where_}: {e}")))?;

            // The cell keeps every format it was ever given, so only the slot
            // being written gives its key up — and a cell already carrying this
            // exact format is left alone, references and bytes both.
            let previous = record.format_id_in(slot);
            if previous == Some(key) {
                continue;
            }
            if let Some(previous) = previous {
                self.step_list_entry(&mut cache, list, previous, -1)?;
            }
            self.step_list_entry(&mut cache, list, key, 1)?;
            record.set_format_id_in(slot, Some(key));
            record.set_explicit_format(slot);
            record.format_kind = Some(crate::table::format_kind_of(slot));
            in_tile.records[column] = Some(
                record
                    .encode()
                    .map_err(|e| Error::Format(format!("{where_}: {e}")))?,
            );
            in_tile.put(&mut tile_archive, &where_)?;
            self.set_archive(tile, &tile_archive)?;
            written += 1;
        }
        cache.flush(self)?;
        Ok(written)
    }

    /// The key of a format entry whose archive is byte for byte this one.
    fn matching_format(
        &self,
        cache: &mut ListCache,
        list: u64,
        payload: &[u8],
    ) -> Result<Option<u32>, Error> {
        let archive = cache.read(self, list)?;
        for value in archive.all(3) {
            let Value::Bytes(raw) = value else { continue };
            let Some(entry) = crate::pb::decode_nested(raw) else {
                continue;
            };
            if entry.bytes(6) == Some(payload) {
                return Ok(entry.varint(1).map(|key| key as u32));
            }
        }
        Ok(None)
    }

    /// Write amounts as **money**, in a currency the caller names.
    ///
    /// A currency cell is a *value type*, not a format: `type 10`, the amount
    /// in the decimal payload, and the currency's format in the cell's currency
    /// slot. That is why a currency format applied to a plain number cell is
    /// ignored by the app — the cell is still a number — and why this exists
    /// beside [`Document::set_format`] rather than inside it.
    ///
    /// Measured, on a cell this crate wrote and Numbers converted through its
    /// own inspector: the record's type went from 2 to **10**, `extras` from
    /// `0x0000` to **`0x0802`** — byte 6's currency bit, and byte 7's `0x08`
    /// whose meaning is not established here — `format_kind` to the currency
    /// slot, and **the number slot's key was dropped**, which is what this does
    /// too. The format the app interned for `CHF` is byte for byte the one
    /// [`crate::table::Format::Currency`] writes.
    ///
    /// The format is interned once for the whole batch and seeded as the slot's
    /// donor, so a table holding no money can be given a money column — which
    /// borrowing from the table, the way every other format is found, cannot
    /// do.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let mut doc = iwork::Document::new_spreadsheet("Sales", "Q1", 4, 3)?;
    /// let mut q1 = doc.table_mut("Q1")?;
    /// q1.currency("C2", 184_300.0, "CHF")?;
    /// # Ok(()) }
    /// ```
    pub fn set_currency(
        &mut self,
        wanted: &str,
        cells: impl IntoIterator<Item = (usize, usize, crate::table::Decimal)>,
        code: &str,
        decimals: Option<u8>,
    ) -> Result<usize, Error> {
        let table = self.table_for_write(wanted)?;
        let site = self.table_site(&table)?;
        let list = site.formats.ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{}: the table has no format list", table.name),
            )
        })?;
        self.refuse_if_patched(&[list], &table.name)?;
        let format = crate::table::Format::Currency {
            code: code.to_string(),
            decimals,
        };
        let payload = format.archive()?.encode();

        let mut cache = ListCache::default();
        // The same entry a second money column reuses, and the same one the app
        // would have interned.
        let key = match self.matching_format(&mut cache, list, &payload)? {
            Some(key) => key,
            None => self.define_list_entry(&mut cache, list, 6, payload, 0)?,
        };
        // Seeding the donor is what makes this work on a table with no money in
        // it: the planner takes a reference per cell, so the count follows.
        cache
            .donors
            .push((crate::table::FormatSlot::Currency, Some(key)));

        let cells: Vec<(usize, usize, CellValue)> = cells
            .into_iter()
            .map(|(row, column, amount)| (row, column, CellValue::Currency(amount)))
            .collect();
        self.write_cells_with(&table, cells, cache)
    }

    // -- how a table is organised --------------------------------------------

    /// Give a table its **sort rules**, replacing whatever it had.
    ///
    /// A sort rule is a column and a direction, and the whole list lives inline
    /// in `TableModelArchive` field 44 — `TST.TableSortOrderArchive`, which is
    /// `{1: type, 2: repeated {1: column, 2: descending}}`. A table with no
    /// rules still carries the archive with its type alone, which is what an
    /// empty list writes.
    ///
    /// **The rules are not the order of the rows.** Numbers keeps them as *what
    /// to sort by* and applies them when asked — through Sort Now, or
    /// continuously when the table is set to sort itself. Nothing here moves a
    /// row: [`Document::insert_row`] and [`Document::delete_row`] are what do
    /// that, and a table whose rows this crate reorders would disagree with its
    /// own rules the moment the app looked.
    ///
    /// The `type` the table already carries is kept. Every one in the corpus is
    /// `0`, which is "the whole table" — the other values name a row range and
    /// nothing here has seen one.
    pub fn set_sort_rules(
        &mut self,
        wanted: &str,
        rules: &[crate::table::SortRule],
    ) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!("{}: sort rules", table.name);
        for rule in rules {
            if rule.column >= table.columns {
                return Err(Error::refused(
                    Refusal::OutOfBounds,
                    format!(
                        "{where_}: column {} of a table with {}",
                        rule.column, table.columns
                    ),
                ));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        for rule in rules {
            if !seen.insert(rule.column) {
                return Err(Error::refused(
                    Refusal::NotACell,
                    format!(
                        "{where_}: column {} is named twice, and which of the two directions \
                         was meant is not this crate's to guess",
                        rule.column
                    ),
                ));
            }
        }
        self.refuse_if_patched(&[table.model], &where_)?;

        let mut archive = self.archive_of(table.model)?;
        let existing = archive
            .bytes(crate::table::model_field::SORT_ORDER)
            .and_then(crate::pb::decode_nested);
        let mut order = Message::default();
        // The type the table already had, or "the whole table" for one that had
        // no archive at all.
        order.set_in_order(
            1,
            Value::Varint(existing.as_ref().and_then(|o| o.varint(1)).unwrap_or(0)),
        );
        for rule in rules {
            let mut entry = Message::default();
            entry.set_in_order(1, Value::Varint(rule.column as u64));
            entry.set_in_order(2, Value::Varint(u64::from(rule.descending)));
            order.append_in_order(2, Value::Bytes(entry.encode()));
        }
        archive.set_in_order(
            crate::table::model_field::SORT_ORDER,
            Value::Bytes(order.encode()),
        );
        self.set_archive_of(table.model, &archive)
    }

    /// Turn a table's filters on or off, and say whether **all** its rules must
    /// match or **any** of them.
    ///
    /// The two switches of the Organise pane, and the two varints at the head
    /// of `TST.FilterSetArchive`: `{1: match_any, 2: enabled}`. The rules
    /// themselves are not touched — writing one means writing the `TSCE`
    /// formula Numbers compiles a filter into, and this corpus carries exactly
    /// one of those to learn from, so a rule built here would be an
    /// extrapolation from a single sample. Refused by name below.
    ///
    /// **The switch does not by itself change which rows are visible**, and that
    /// was measured rather than assumed. With the filter written off, Numbers
    /// opened the document, edited a cell and saved: the switch came back
    /// *off*, and the ten rows the filter had hidden were still hidden. Which
    /// rows are hidden is **stored**, in the per-row hiding state and the
    /// row hidden-state extent, and the app recomputes it when the filter is
    /// next touched in its own interface — not when the document opens. The
    /// same shape as a formula's cached value and a chart's grid: a cache the
    /// app maintains, not one it rebuilds on load.
    ///
    /// So this writes what the Organise pane shows, and nothing else. Clearing
    /// the hidden state as well would mean rewriting the UUID-keyed extent that
    /// [`Document::insert_row`] refuses to maintain for exactly the same
    /// reason.
    pub fn set_filter_enabled(
        &mut self,
        wanted: &str,
        enabled: bool,
        match_any: Option<bool>,
    ) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!("{}: filter", table.name);
        let filter = table.filter.as_ref().ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!(
                    "{where_}: the table has no filter set with rules in it, and a filter \
                     made from nothing would need the TSCE formula Numbers compiles one into \
                     — this corpus has one of those to learn from, which is not enough"
                ),
            )
        })?;
        self.refuse_if_patched(&[filter.identifier], &where_)?;

        let mut archive = self.archive_of(filter.identifier)?;
        // Field 2's default is **true**, so "on" is written explicitly rather
        // than left out: an absent field and a false one are different things
        // everywhere else in this format and the same thing here only by
        // accident.
        archive.set_in_order(2, Value::Varint(u64::from(enabled)));
        if let Some(any) = match_any {
            archive.set_in_order(1, Value::Varint(u64::from(any)));
        }
        self.set_archive_of(filter.identifier, &archive)
    }

    /// Change the number a conditional highlight compares against.
    ///
    /// See [`crate::table::set_conditional_threshold`]: the threshold is kept
    /// twice on each rule and the rules twice over in the set, and all of it is
    /// rewritten together.
    pub fn set_conditional_threshold(
        &mut self,
        set: u64,
        rule: usize,
        value: crate::table::Decimal,
    ) -> Result<(), Error> {
        crate::table::set_conditional_threshold(self, set, rule, value)
    }

    // -- writing a formula ---------------------------------------------------

    /// Give a cell a formula, written as text.
    ///
    /// `=SUM(B2:B4)` — the thing a spreadsheet library is for, and the one
    /// place in this crate that builds a `TSCE` AST from nothing.
    /// [`crate::formula_parse`] is the parser, and every node it emits was
    /// copied from a formula the app wrote: the shapes were dumped out of
    /// `numbers-formulas.numbers` node by node and are reproduced field for
    /// field, down to the list node a parenthesis writes and the `5: 1` on
    /// every colon tract.
    ///
    /// **The value is the caller's**, as it is for [`Document::fill_formula`]
    /// and for the same reason: Numbers shows what is written in the cell until
    /// something the formula reads changes, and this crate evaluates nothing.
    /// Give the answer the formula would produce, or an answer the app will
    /// correct the moment a precedent moves.
    ///
    /// The formula is *live*: its cell is registered in the calculation
    /// engine's dependency graph ([`crate::calc`]), so the app recalculates it
    /// whenever a cell it reads changes.
    ///
    /// Refused by name: a cell that already holds a formula (removing one is
    /// `TSCE` surgery this crate does not do), a rich-text cell, a
    /// merge-covered cell, an empty value, a table with no formula list, any
    /// object carrying version patches — and, from the parser, everything whose
    /// dependency edges this crate cannot write: another table, a whole row or
    /// column, a header name, a function it does not know.
    pub fn set_formula(
        &mut self,
        wanted: &str,
        row: usize,
        column: usize,
        text: &str,
        value: CellValue,
    ) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!(
            "{} {}",
            table.name,
            crate::table::CellRef::Index(row, column)
        );
        if row >= table.rows || column >= table.columns {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!("{where_}: the table is {}×{}", table.rows, table.columns),
            ));
        }
        if let Some(merge) = table.merge_covering(row, column) {
            if (merge.row, merge.column) != (row, column) {
                return Err(Error::refused(
                    Refusal::Merged,
                    format!(
                        "{where_}: covered by the merge that begins at row {} column {}",
                        merge.row, merge.column
                    ),
                ));
            }
        }
        if value == CellValue::Empty {
            return Err(Error::refused(
                Refusal::UnwritableValue,
                format!(
                    "{where_}: a formula cell cannot be empty — the value is the answer the app \
                 shows until it recalculates, and this crate evaluates nothing"
                ),
            ));
        }

        // The formula first, because it is what the caller most likely got
        // wrong, and nothing has moved yet when it is refused.
        let ast = crate::formula_parse::parse(text, (column as i64, row as i64))
            .map_err(|e| Error::refused(Refusal::NotACell, format!("{where_}: {e}")))?;
        let formula = crate::formula::Formula::from_ast(ast);
        let precedents = crate::calc::precedents_of(&formula, row, column).ok_or_else(|| {
            Error::refused(
                Refusal::FormulaReference,
                format!(
                    "{where_}: the formula reads something this crate cannot resolve to cells \
                     of this table, and a formula the engine only half knows about is one the \
                     app would recalculate wrongly"
                ),
            )
        })?;

        let site = self.cell_site(&table, row, column)?;
        let list = site.formulas.ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{where_}: the table has no formula list"),
            )
        })?;
        let old = match &site.record {
            Some(bytes) => crate::table::decode_cell(bytes)
                .map_err(|e| Error::Format(format!("{where_}: {e}")))?,
            None => crate::table::CellRecord {
                version: 5,
                ..crate::table::CellRecord::default()
            },
        };
        if old.formula_id.is_some() {
            return Err(Error::refused(
                Refusal::HoldsFormula,
                format!(
                    "{where_}: already holds a formula, and replacing one means taking its \
                 dependency edges out of the calculation engine first"
                ),
            ));
        }
        if old.cell_type == cell_type::RICH_TEXT || old.rich_id.is_some() {
            return Err(Error::refused(
                Refusal::HoldsRichText,
                format!(
                    "{where_}: holds rich text, whose words are in a TSWP storage rather than in \
                 the table"
                ),
            ));
        }
        self.refuse_if_patched(&[list, site.tile], &where_)?;

        let table_site = self.table_site(&table)?;
        let mut cache = ListCache::default();
        let write = self.rewrite_record(&mut cache, &table, &table_site, old, &value, &where_)?;
        let mut record = write.record.ok_or_else(|| {
            Error::refused(
                Refusal::UnwritableValue,
                format!("{where_}: a formula cell cannot be empty"),
            )
        })?;
        // One reference, taken by the cell this write is about.
        let key = self.define_list_entry(&mut cache, list, 5, formula.encode(), 1)?;
        record.formula_id = Some(key);

        let had = site.record.is_some();
        let mut site = site;
        site.records[site.column] = Some(
            record
                .encode()
                .map_err(|e| Error::Format(format!("{where_}: {e}")))?,
        );
        let appeared = if had { Vec::new() } else { vec![site.column] };
        let staged = self.stage_row(&site, appeared, Vec::new())?;
        for mutation in &write.mutations {
            self.apply_mutation(&mut cache, mutation)?;
        }
        cache.flush(self)?;
        self.write_store(&site, staged)?;
        crate::calc::register_formula(
            self,
            &table,
            &crate::calc::Dependency {
                row,
                column,
                precedents,
            },
        )?;
        Ok(())
    }

    /// Put a new entry in a `TableDataList` and hand back its key.
    ///
    /// The generic half of interning: an entry is `{1: key, 2: refcount}` plus
    /// one payload field whose number says what kind of list it is — 5 for a
    /// formula, as 3 is for a string. The key is the same arithmetic
    /// [`Document::planned_string_key`] does: above the list's own mark *and*
    /// above every key present, because the mark is only trustworthy while the
    /// app is the one maintaining it. `references` is the refcount the entry
    /// starts with: one when the caller is the single user of it, zero when
    /// each user takes its own.
    fn define_list_entry(
        &self,
        cache: &mut ListCache,
        list: u64,
        field: u32,
        payload: Vec<u8>,
        references: u64,
    ) -> Result<u32, Error> {
        let archive = cache.write(self, list)?;
        let highest = archive
            .all(3)
            .filter_map(|value| match value {
                Value::Bytes(raw) => crate::pb::decode_nested(raw)?.varint(1),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let candidate = archive
            .varint(2)
            .unwrap_or(0)
            .max(highest.saturating_add(1))
            .max(1);
        let key = u32::try_from(candidate).map_err(|_| {
            Error::Format(format!(
                "list {list}: the next key {candidate} does not fit the 32 bits a \
                 TableDataList key has"
            ))
        })?;
        let mut entry = Message::default();
        entry.set_in_order(1, Value::Varint(u64::from(key)));
        entry.set_in_order(2, Value::Varint(references));
        entry.set_in_order(field, Value::Bytes(payload));
        archive.set_in_order(2, Value::Varint(u64::from(key) + 1));
        archive.append_in_order(3, Value::Bytes(entry.encode()));
        Ok(key)
    }

    // -- filling a formula ---------------------------------------------------

    /// Give a cell the formula another cell in the same table holds.
    ///
    /// This is *fill*, and it is the only way to write a formula here. A
    /// formula lives in the table's `FORMULA` list and a cell holds its key;
    /// the entry names no host cell, which is why one entry serves a whole
    /// filled column and why giving another cell the same key is a real fill
    /// rather than a trick — a **relative** reference in the formula is
    /// relative to whichever cell holds it, so `=B2+1` in C2 is `=B3+1` in C3,
    /// exactly as the app's own fill does.
    ///
    /// **The value is the caller's**, and it has to be: Numbers shows what is
    /// written in the cell until something the formula reads changes, and this
    /// crate evaluates nothing. `None` says "the source's value", which is
    /// refused unless every reference in the formula is absolute — the one case
    /// where the source's answer is the target's too.
    ///
    /// The formula is *live*: the cell is registered in the calculation
    /// engine's dependency graph ([`crate::calc`]), so the app recalculates it
    /// whenever a cell it reads changes. Watched, with `=B1+B2` filled from
    /// `=A1+A2`: 0 on opening, 12 after `B1=5, B2=7`, 8 after `B1=1`.
    ///
    /// Refused by name: a source with no formula, a target that already holds
    /// one, a formula that reads another table or a whole row or column (whose
    /// dependency edges this crate does not write), an out-of-range cell, a
    /// merge-covered target, a table with no formula list, and any object
    /// carrying version patches.
    pub fn fill_formula(
        &mut self,
        wanted: &str,
        from: (usize, usize),
        to: (usize, usize),
        value: Option<CellValue>,
    ) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!(
            "{} {} → {}",
            table.name,
            crate::table::CellRef::Index(from.0, from.1),
            crate::table::CellRef::Index(to.0, to.1)
        );
        for (row, column) in [from, to] {
            if row >= table.rows || column >= table.columns {
                return Err(Error::refused(
                    Refusal::OutOfBounds,
                    format!("{where_}: the table is {}×{}", table.rows, table.columns),
                ));
            }
        }
        if from == to {
            return Err(Error::refused(
                Refusal::NotACell,
                format!("{where_}: the same cell"),
            ));
        }
        if let Some(merge) = table.merge_covering(to.0, to.1) {
            if (merge.row, merge.column) != to {
                return Err(Error::Format(format!(
                    "{where_}: the target is covered by a merge"
                )));
            }
        }

        let source = self.cell_site(&table, from.0, from.1)?;
        let source_record = source
            .record
            .as_ref()
            .ok_or_else(|| Error::Format(format!("{where_}: the source cell is empty")))?;
        let source_record = crate::table::decode_cell(source_record)
            .map_err(|e| Error::Format(format!("{where_}: {e}")))?;
        let formula = source_record
            .formula_id
            .ok_or_else(|| Error::Format(format!("{where_}: the source holds no formula")))?;

        // What the formula will read from where it is going. The engine has to
        // be told, or the app never recalculates the cell — and if this crate
        // cannot work out every cell it reads, registering half a graph would
        // be worse than refusing.
        let ast = table
            .formula_cells()
            .into_iter()
            .find(|(row, column, _)| (*row, *column) == from)
            .map(|(_, _, formula)| formula)
            .ok_or_else(|| {
                Error::Format(format!("{where_}: the source's formula does not decode"))
            })?;
        let precedents = crate::calc::precedents_of(ast, to.0, to.1).ok_or_else(|| {
            Error::refused(
                Refusal::FormulaReference,
                format!(
                    "{where_}: the formula reads something this crate cannot resolve to cells of \
                 this table — another table, a whole row or column, or a stored #REF! — and a \
                 formula the engine only half knows about is one the app would recalculate \
                 wrongly"
                ),
            )
        })?;

        let site = self.cell_site(&table, to.0, to.1)?;
        let old = match &site.record {
            Some(bytes) => crate::table::decode_cell(bytes)
                .map_err(|e| Error::Format(format!("{where_}: {e}")))?,
            None => crate::table::CellRecord {
                version: 5,
                ..crate::table::CellRecord::default()
            },
        };
        if old.formula_id.is_some() {
            return Err(Error::refused(
                Refusal::HoldsFormula,
                format!(
                    "{where_}: the target already holds a formula, and taking one out means \
                 editing the calculation engine"
                ),
            ));
        }
        if old.cell_type == crate::table::cell_type::RICH_TEXT || old.rich_id.is_some() {
            return Err(Error::refused(
                Refusal::HoldsRichText,
                format!("{where_}: the target holds rich text"),
            ));
        }
        let list = site.formulas.ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{where_}: the table has no formula list"),
            )
        })?;
        self.refuse_if_patched(&[site.tile, list], &where_)?;

        // The value the target will *show*. Numbers shows whatever is written
        // here until something the formula reads changes, so it is the
        // caller's: `None` says "the source's", which is the right answer only
        // when the formula cannot mean anything different where it is going.
        let value = match value {
            Some(value) => value,
            None => {
                self.refuse_if_the_answer_would_move(ast, &where_)?;
                table.value(from.0, from.1)
            }
        };
        match value {
            CellValue::Empty
            | CellValue::Text(_)
            | CellValue::Number(_)
            | CellValue::Bool(_)
            | CellValue::Date(_)
            | CellValue::Duration(_) => {}
            other => {
                return Err(Error::refused(
                    Refusal::UnwritableValue,
                    format!("{where_}: this crate does not write {} cells", other.kind()),
                ))
            }
        }

        // The value goes in through the same planner every other write uses —
        // string interning, format borrowing, reference counts and all — and
        // the formula key is the one thing added on top.
        let table_site = self.table_site(&table)?;
        let mut cache = ListCache::default();
        let write = self.rewrite_record(&mut cache, &table, &table_site, old, &value, &where_)?;
        let mut record = write.record.ok_or_else(|| {
            Error::refused(
                Refusal::UnwritableValue,
                format!("{where_}: a formula cell cannot be empty"),
            )
        })?;
        record.formula_id = Some(formula);
        let mut mutations = write.mutations;
        mutations.push(ListMutation::Retain { list, key: formula });

        let had = site.record.is_some();
        let mut site = site;
        site.records[site.column] = Some(
            record
                .encode()
                .map_err(|e| Error::Format(format!("{where_}: {e}")))?,
        );
        let appeared = if had { Vec::new() } else { vec![site.column] };
        let staged = self.stage_row(&site, appeared, Vec::new())?;
        for mutation in &mutations {
            self.apply_mutation(&mut cache, mutation)?;
        }
        cache.flush(self)?;
        self.write_store(&site, staged)?;
        crate::calc::register_formula(
            self,
            &table,
            &crate::calc::Dependency {
                row: to.0,
                column: to.1,
                precedents,
            },
        )?;
        Ok(())
    }

    /// Refuse a fill whose answer would differ from the value it copies.
    ///
    /// **Numbers does not recalculate a filled cell.** The value in the record
    /// is what the app shows, on opening and after saving, and it goes on
    /// showing it: a cell filled with a formula whose cached value belongs to
    /// another row was watched reporting that other row's answer, and still
    /// reporting it after Numbers had loaded the document and written it back.
    /// Editing a cell the formula depends on does not help either — the
    /// engine's dependency graph has no edge for a cell this crate added, so
    /// the recalculation never reaches it.
    ///
    /// What follows is a rule, not a workaround: a fill is allowed exactly when
    /// the formula's answer cannot depend on where the formula sits — every
    /// reference **absolute** on both axes, and no whole-row or whole-column
    /// reference. Then the value copied is the value the target computes, and
    /// the document is right the moment it opens. A relative reference is
    /// refused by name, because the alternative is a document showing a number
    /// that is wrong.
    fn refuse_if_the_answer_would_move(
        &self,
        formula: &crate::formula::Formula,
        where_: &str,
    ) -> Result<(), Error> {
        use crate::formula::Axis;
        for node in &formula.ast.nodes {
            let Some(reference) = node.reference() else {
                continue;
            };
            if reference.is_error {
                return Err(Error::Format(format!(
                    "{where_}: the source's formula holds a stored #REF!"
                )));
            }
            for axis in [
                reference.column,
                reference.row,
                reference.column_end,
                reference.row_end,
            ] {
                match axis {
                    Axis::Absolute(_) => {}
                    Axis::Relative(_) => {
                        return Err(Error::Format(format!(
                            "{where_}: the source's formula has a relative reference, so its \
                             answer moves with it — and Numbers shows the value this crate \
                             writes rather than recalculating, so the filled cell would show \
                             the source's answer for ever. FORMAT.md §9 has the measurement."
                        )))
                    }
                    Axis::Unbounded => {
                        return Err(Error::Format(format!(
                            "{where_}: the source's formula names a whole row or column, whose \
                             answer depends on where the formula sits"
                        )))
                    }
                }
            }
        }
        Ok(())
    }

    // -- inserting a row -----------------------------------------------------

    /// Insert an empty row before index `at`; `at == rows` appends.
    ///
    /// Transactional in the same way as [`Document::set_cell`]: the whole write
    /// is planned — and every case this crate cannot maintain safely is refused
    /// by name — before a single byte moves, so a refused insert leaves the
    /// document byte for byte as it was.
    ///
    /// **What it supports.** A plain rectangular table, over any number of
    /// tiles: the row count is bumped, every `TileRowInfo` at or below the
    /// insertion is shifted down — **across tile boundaries where the shift
    /// crosses one**, since a row's absolute index is
    /// `tileid * tile_size + tile_row_index` — the row header bucket's per-row
    /// entries are shifted and one is added for the new empty row, and the
    /// `ColumnRowUIDMapArchive` gains a fresh per-table-unique row UUID at the
    /// new index (rebuilt sorted by UUID, the way the app keeps it). The new
    /// row has no `TileRowInfo` and no cells — an empty row has none — so
    /// [`Document::set_cell`] is what fills it afterwards.
    ///
    /// **What it refuses, by name**, because no corpus fixture proves the
    /// bookkeeping and a wrong guess corrupts silently:
    ///
    /// * a table that fills every tile it has — the row would need a tile of
    ///   its own, which is a new object *and* a new component;
    /// * a categorised, pivoted or filtered table, or one with hidden or
    ///   collapsed rows — the group nodes address rows by index and the hidden
    ///   state by UUID, two schemes that must stay in step, and nothing here has
    ///   both to verify against;
    /// * a table carrying conditional highlighting (its rule ranges are row
    ///   addressed);
    /// * a merge at or straddling the insertion point — a merge is stored as an
    ///   absolute-row formula that nothing here rewrites;
    /// * any table whose formulas reference this table at or below the insertion
    ///   point — inserting a row shifts what those references mean, and rewriting
    ///   a `TSCE` AST is a phase of its own;
    /// * an object the write would touch that carries version patches.
    pub fn insert_row(&mut self, wanted: &str, at: usize) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!("{}: insert row at {at}", table.name);

        if at > table.rows {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!(
                    "{where_}: the table has {} row(s), so a row goes in at 0..={}",
                    table.rows, table.rows
                ),
            ));
        }

        // Everything unproven is refused here, before a byte moves.
        if !table.categories.is_empty() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table is categorised, and a category's group nodes address \
                 rows by index — maintaining them across an insert is unverified here"
                ),
            ));
        }
        if table.pivot.is_some() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                "{where_}: the table is a pivot table, whose rows the app builds from its source"
            ),
            ));
        }
        if table.filter.is_some() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table is filtered, and a filter's hidden state addresses rows \
                 by UUID — maintaining it across an insert is unverified here"
                ),
            ));
        }
        if !table.conditional_styles.is_empty() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table carries conditional highlighting, whose rule ranges are \
                 row addressed"
                ),
            ));
        }
        if table.footer_rows > 0 {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table has {} footer row(s), and where an inserted row falls \
                 relative to a footer is not something any fixture here can verify",
                    table.footer_rows
                ),
            ));
        }
        if !table.row_states.user_hidden.is_empty()
            || !table.row_states.filtered.is_empty()
            || !table.row_states.collapsed_groups.is_empty()
            || table.row_extents.iter().any(|extent| extent.hidden())
        {
            return Err(Error::refused(
                Refusal::Organised,
                format!("{where_}: the table has hidden or collapsed rows, addressed by UUID"),
            ));
        }
        if let Some(merge) = table.merges.iter().find(|m| m.row + m.rows > at) {
            return Err(Error::refused(
                Refusal::Merged,
                format!(
                    "{where_}: a merge at row {} column {} would shift or straddle the new row, \
                 and a merge is stored as an absolute-row formula this crate does not rewrite",
                    merge.row, merge.column
                ),
            ));
        }
        if let Some(reason) = self.row_insert_would_break_a_formula(&table, at) {
            return Err(Error::Format(format!("{where_}: {reason}")));
        }

        let plan = self.plan_row_insert(&table, at, &where_)?;
        self.refuse_if_patched(&plan.touched, &where_)?;

        // Commit: every object was decoded during planning, so nothing below
        // refuses a write already committed to.
        self.set_archive(plan.model, &plan.model_archive)?;
        for (tile, archive) in &plan.tiles {
            self.set_archive(*tile, archive)?;
        }
        self.set_archive(plan.row_bucket, &plan.bucket_archive)?;
        self.set_archive(plan.uid_map, &plan.uid_map_archive)?;
        Ok(())
    }

    /// Locate every object a row delete rewrites and build each one's new form.
    fn plan_row_delete(
        &self,
        table: &crate::table::Table,
        at: usize,
        where_: &str,
    ) -> Result<DeletePlan, Error> {
        use crate::pb::decode_nested;
        let model = table.model;
        let mut model_archive = self.archive_of(model)?;
        let store = model_archive
            .bytes(4)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the model has no data store")))?;

        let tiles = store
            .bytes(3)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the data store has no tiles")))?;
        let tile_size = tiles.varint(2).unwrap_or(256) as usize;
        let tile_entries: Vec<(u64, u64)> = tiles
            .all(1)
            .filter_map(|value| match value {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .filter_map(|entry| {
                let id = entry.varint(1)?;
                let reference = entry.bytes(2).and_then(crate::table::reference)?;
                Some((id, reference))
            })
            .collect();
        if tile_entries.is_empty() {
            return Err(Error::refused(
                Refusal::Missing,
                format!("{where_}: the table has no tiles"),
            ));
        }
        for (position, (tile_id, _)) in tile_entries.iter().enumerate() {
            if *tile_id != position as u64 {
                return Err(Error::Format(format!(
                    "{where_}: tile {position} is numbered {tile_id}, and a row's absolute \
                     index is its tile's number times {tile_size} plus its index within it"
                )));
            }
        }

        let header_storage = store.bytes(1).and_then(decode_nested).ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{where_}: the data store has no row headers"),
            )
        })?;
        let buckets: Vec<u64> = header_storage
            .all(2)
            .filter_map(|value| match value {
                Value::Bytes(raw) => crate::table::reference(raw),
                _ => None,
            })
            .collect();
        if buckets.len() != 1 {
            return Err(Error::Format(format!(
                "{where_}: the row headers span {} buckets, not one",
                buckets.len()
            )));
        }
        let bucket = buckets[0];
        let uid_map = model_archive
            .bytes(46)
            .and_then(crate::table::reference)
            .ok_or_else(|| {
                Error::Format(format!(
                    "{where_}: the model has no ColumnRowUIDMap, so the row's identity has \
                     nowhere to be taken from"
                ))
            })?;
        self.refuse_if_row_extent_keys_rows(&model_archive, where_)?;

        let tiles = self.tiles_with_deleted_row(&tile_entries, tile_size, at, where_)?;
        let bucket_archive = self.bucket_without(bucket, at, where_)?;
        let uid_map_archive = self.uid_map_without(uid_map, at, true, where_)?;
        model_archive.set(6, Value::Varint((table.rows - 1) as u64));

        let mut touched = vec![model, bucket, uid_map];
        touched.extend(tiles.iter().map(|(tile, _)| *tile));
        Ok(DeletePlan {
            touched,
            model,
            model_archive,
            tiles,
            bucket,
            bucket_archive,
            uid_map,
            uid_map_archive,
        })
    }

    /// Every tile with row `at` gone and the rows below it shifted up one.
    ///
    /// The insert's mirror, and it crosses tile boundaries the same way: the
    /// rows are gathered by absolute index, the one at `at` is dropped, the
    /// rest shift, and each is laid back out into the tile it now belongs to.
    /// `numrows` (field 4) is recounted per tile, because it counts the
    /// `TileRowInfo`s a tile holds and one of them has gone.
    fn tiles_with_deleted_row(
        &self,
        tiles: &[(u64, u64)],
        tile_size: usize,
        at: usize,
        where_: &str,
    ) -> Result<Vec<(u64, Message)>, Error> {
        use crate::pb::decode_nested;
        let mut rows: Vec<(usize, Message)> = Vec::new();
        let mut archives: Vec<(u64, Message)> = Vec::new();
        for (number, tile) in tiles {
            let archive = self.archive_of(*tile)?;
            for value in archive.all(5) {
                let Value::Bytes(raw) = value else { continue };
                let info = decode_nested(raw).ok_or_else(|| {
                    Error::Format(format!("{where_}: a tile row does not decode"))
                })?;
                let within = info.varint(1).unwrap_or(0) as usize;
                if within >= tile_size {
                    return Err(Error::Format(format!(
                        "{where_}: a row of tile {number} is at index {within}, past the \
                         {tile_size} rows a tile holds"
                    )));
                }
                rows.push((*number as usize * tile_size + within, info));
            }
            archives.push((*tile, archive));
        }

        let mut per_tile: BTreeMap<usize, Vec<(usize, Message)>> = BTreeMap::new();
        for (index, mut info) in rows {
            let index = match index.cmp(&at) {
                std::cmp::Ordering::Equal => continue, // the row that goes
                std::cmp::Ordering::Greater => index - 1,
                std::cmp::Ordering::Less => index,
            };
            let (tile, within) = (index / tile_size, index % tile_size);
            info.set(1, Value::Varint(within as u64));
            per_tile.entry(tile).or_default().push((within, info));
        }

        let mut out = Vec::new();
        for (position, (tile, mut archive)) in archives.into_iter().enumerate() {
            let mut mine = per_tile.remove(&position).unwrap_or_default();
            mine.sort_by_key(|(within, _)| *within);
            archive.clear(5);
            for (_, info) in &mine {
                archive.append_in_order(5, Value::Bytes(info.encode()));
            }
            archive.set(4, Value::Varint(mine.len() as u64));
            out.push((tile, archive));
        }
        Ok(out)
    }

    /// Locate every object a column delete rewrites and build its new form.
    fn plan_column_delete(
        &self,
        table: &crate::table::Table,
        at: usize,
        where_: &str,
    ) -> Result<DeletePlan, Error> {
        use crate::pb::decode_nested;
        let model = table.model;
        let mut model_archive = self.archive_of(model)?;
        let store = model_archive
            .bytes(4)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the model has no data store")))?;
        let tiles = store
            .bytes(3)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the data store has no tiles")))?;
        let tile_ids: Vec<u64> = tiles
            .all(1)
            .filter_map(|value| match value {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .filter_map(|entry| entry.bytes(2).and_then(crate::table::reference))
            .collect();
        if tile_ids.is_empty() {
            return Err(Error::refused(
                Refusal::Missing,
                format!("{where_}: the table has no tiles"),
            ));
        }
        let bucket = store
            .bytes(2)
            .and_then(crate::table::reference)
            .ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!("{where_}: the data store has no column headers"),
                )
            })?;
        let uid_map = model_archive
            .bytes(46)
            .and_then(crate::table::reference)
            .ok_or_else(|| Error::Format(format!("{where_}: the model has no ColumnRowUIDMap")))?;

        let mut plan_tiles = Vec::new();
        for tile in &tile_ids {
            plan_tiles.push((*tile, self.tile_with_deleted_column(*tile, at, where_)?));
        }
        let bucket_archive = self.bucket_without(bucket, at, where_)?;
        let uid_map_archive = self.uid_map_without(uid_map, at, false, where_)?;
        model_archive.set(7, Value::Varint((table.columns - 1) as u64));

        let mut touched = vec![model, bucket, uid_map];
        touched.extend(&tile_ids);
        Ok(DeletePlan {
            touched,
            model,
            model_archive,
            tiles: plan_tiles,
            bucket,
            bucket_archive,
            uid_map,
            uid_map_archive,
        })
    }

    /// Every row of one tile with column `at` taken out.
    ///
    /// A `-1` takes its place at the end of the offset array, so the array
    /// keeps the length it arrived with: its padding is what a reader steps
    /// through, and shortening it would say the table is narrower than it is.
    fn tile_with_deleted_column(
        &self,
        tile: u64,
        at: usize,
        where_: &str,
    ) -> Result<Message, Error> {
        use crate::pb::decode_nested;
        let mut archive = self.archive_of(tile)?;
        for field in archive.fields.iter_mut() {
            if field.number != 5 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let mut info = decode_nested(raw)
                .ok_or_else(|| Error::Format(format!("{where_}: a tile row does not decode")))?;
            let buffer = info.bytes(6).unwrap_or(&[]).to_vec();
            let offsets = info.bytes(7).unwrap_or(&[]).to_vec();
            if offsets.is_empty() {
                continue;
            }
            let wide = info.varint(8).unwrap_or(0) != 0;
            let entries = offsets.len() / 2;
            if at >= entries {
                return Err(Error::Format(format!(
                    "{where_}: a row's offset array holds {entries} column(s), so there is no \
                     column {at} in it"
                )));
            }
            let mut records: Vec<Option<Vec<u8>>> = vec![None; entries];
            for (column, bytes) in crate::table::row_cells(&buffer, &offsets, wide)
                .into_iter()
                .flatten()
            {
                records[column] = Some(bytes.to_vec());
            }
            records.remove(at);
            records.push(None);
            let cells = records.iter().filter(|record| record.is_some()).count() as u64;
            let (buffer, offsets) = crate::table::encode_row(&records, wide)
                .map_err(|e| Error::Format(format!("{where_}: {e}")))?;
            info.set(2, Value::Varint(cells));
            info.set(6, Value::Bytes(buffer));
            info.set(7, Value::Bytes(offsets));
            field.value = Value::Bytes(info.encode());
        }
        Ok(archive)
    }

    /// A header bucket with the entry at `at` gone and those after it shifted.
    fn bucket_without(&self, bucket: u64, at: usize, where_: &str) -> Result<Message, Error> {
        use crate::pb::decode_nested;
        let mut archive = self.archive_of(bucket)?;
        let mut drop = None;
        for (position, field) in archive.fields.iter_mut().enumerate() {
            if field.number != 2 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let mut entry = decode_nested(raw).ok_or_else(|| {
                Error::Format(format!("{where_}: a header entry does not decode"))
            })?;
            let index = entry.varint(1).unwrap_or(0) as usize;
            match index.cmp(&at) {
                std::cmp::Ordering::Equal => drop = Some(position),
                std::cmp::Ordering::Greater => {
                    entry.set(1, Value::Varint((index - 1) as u64));
                    field.value = Value::Bytes(entry.encode());
                }
                std::cmp::Ordering::Less => {}
            }
        }
        if let Some(position) = drop {
            archive.fields.remove(position);
        }
        Ok(archive)
    }

    /// The `ColumnRowUIDMapArchive` with one line's UUID taken out.
    ///
    /// The three arrays are rebuilt and re-sorted by the 128-bit value, the
    /// order the app keeps them in — the same rebuild an insert does, with a
    /// UUID removed instead of minted. `row` picks the half: fields 4/5/6 for
    /// rows, 1/2/3 for columns.
    fn uid_map_without(
        &self,
        uid_map: u64,
        at: usize,
        row: bool,
        where_: &str,
    ) -> Result<Message, Error> {
        use crate::table::Uuid;
        let (uuids_field, index_field, position_field) = if row { (4, 5, 6) } else { (1, 2, 3) };
        let mut archive = self.archive_of(uid_map)?;

        let uuids: Vec<Uuid> = archive
            .all(uuids_field)
            .filter_map(|value| match value {
                Value::Bytes(raw) => crate::pb::decode_nested(raw).map(|m| Uuid::decode(&m)),
                _ => None,
            })
            .collect();
        let indices: Vec<u64> = archive
            .all(index_field)
            .filter_map(|value| match value {
                Value::Varint(n) => Some(*n),
                _ => None,
            })
            .collect();
        if uuids.len() != indices.len() {
            return Err(Error::Format(format!(
                "{where_}: the UUID map has {} UUID(s) and {} index(es); a shape this crate \
                 did not write is not safe to rebuild",
                uuids.len(),
                indices.len()
            )));
        }

        let mut pairs: Vec<(Uuid, usize)> = Vec::new();
        for (&uuid, &index) in uuids.iter().zip(&indices) {
            let index = index as usize;
            match index.cmp(&at) {
                std::cmp::Ordering::Equal => continue,
                std::cmp::Ordering::Greater => pairs.push((uuid, index - 1)),
                std::cmp::Ordering::Less => pairs.push((uuid, index)),
            }
        }
        pairs.sort_by_key(|(uuid, _)| (uuid.upper, uuid.lower));

        let count = pairs.len();
        let mut uid_for_index = vec![0u32; count];
        for (position, (_, index)) in pairs.iter().enumerate() {
            if *index >= count {
                return Err(Error::Format(format!(
                    "{where_}: the UUID map names index {index} of {count}"
                )));
            }
            uid_for_index[*index] = position as u32;
        }

        archive.clear(uuids_field);
        archive.clear(index_field);
        archive.clear(position_field);
        for (uuid, _) in &pairs {
            archive.append_in_order(uuids_field, Value::Bytes(encode_uuid(*uuid)));
        }
        for (_, index) in &pairs {
            archive.append_in_order(index_field, Value::Varint(*index as u64));
        }
        for &position in &uid_for_index {
            archive.append_in_order(position_field, Value::Varint(u64::from(position)));
        }
        Ok(archive)
    }

    // -- merging cells -------------------------------------------------------

    /// Merge a rectangle of cells into one.
    ///
    /// **A merge is stored nowhere near the cells it covers.** The covered
    /// cells simply stop having records — not even a `spanCellType` one — and
    /// what says they are merged is a *formula*, one per range, in the formula
    /// store of the table's `merge_owner` (`TableModelArchive` field 47). This
    /// writes the shape the app writes, taken from the four merges of
    /// `numbers-formats.numbers`:
    ///
    /// ```text
    /// FormulaStorePair { 1: index, 2: FormulaArchive { 1: nodes } }
    ///   COLON_TRACT { 28: {1: CFUUID(the table's base_owner_uid)},
    ///                 33: {1,1,1,1}, 40: {3: columns, 4: rows, 5: 1} }
    ///   FUNCTION    { 2: 168 (SUM), 3: 1 }
    /// ```
    ///
    /// The range is written **absolute** on both axes, which is what the four
    /// sticky bits say, and the covered cells are emptied through the ordinary
    /// cell writer — so every string, format and control key they held is given
    /// back and the lists' reference counts still match.
    ///
    /// Refused by name: a range off the table, a range of one cell, a range
    /// overlapping a merge the table already has, a covered cell holding a
    /// formula, a categorised, filtered, pivoted or conditionally highlighted
    /// table, and any object carrying version patches. **Not** refused, and
    /// worth knowing: a formula elsewhere that reads a covered cell keeps its
    /// cached value and reads an empty cell the next time the app
    /// recalculates — the same staleness any cell write causes.
    pub fn merge_cells(
        &mut self,
        wanted: &str,
        row: usize,
        column: usize,
        rows: usize,
        columns: usize,
    ) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!(
            "{}: merge {} {rows}×{columns}",
            table.name,
            crate::table::CellRef::Index(row, column)
        );
        if rows == 0 || columns == 0 {
            return Err(Error::refused(
                Refusal::NotACell,
                format!("{where_}: an empty range"),
            ));
        }
        if rows * columns == 1 {
            return Err(Error::refused(
                Refusal::NotACell,
                format!(
                    "{where_}: one cell is not a merge to make — the app writes such a range only \
                 when a merge is being taken apart"
                ),
            ));
        }
        if row + rows > table.rows || column + columns > table.columns {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!("{where_}: the table is {}×{}", table.rows, table.columns),
            ));
        }
        // A merge spans both axes, so both axes' hidden states are its
        // business — unlike a delete, which only ever moves one.
        self.refuse_if_organised(&table, true, &where_)?;
        self.refuse_if_organised(&table, false, &where_)?;
        for merge in &table.merges {
            let overlaps = row < merge.row + merge.rows
                && merge.row < row + rows
                && column < merge.column + merge.columns
                && merge.column < column + columns;
            if overlaps {
                return Err(Error::refused(
                    Refusal::Merged,
                    format!(
                        "{where_}: it overlaps the merge at row {} column {}",
                        merge.row, merge.column
                    ),
                ));
            }
        }

        let model = table.model;
        self.refuse_if_patched(&[model], &where_)?;
        let mut model_archive = self.archive_of(model)?;
        let mut owner = model_archive
            .bytes(47)
            .and_then(crate::pb::decode_nested)
            .ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!(
                        "{where_}: the table has no merge owner, and one made from nothing would \
                     need an owner UUID this crate cannot derive"
                    ),
                )
            })?;
        let node = merge_range_node(&table, row, column, rows, columns)?;

        // Everything under the merge but its own top-left cell loses its value,
        // which is what the app leaves behind — and goes through the ordinary
        // writer, so the lists' reference counts follow.
        let covered: Vec<(usize, usize, CellValue)> = (row..row + rows)
            .flat_map(|r| (column..column + columns).map(move |c| (r, c)))
            .filter(|&(r, c)| (r, c) != (row, column))
            .map(|(r, c)| (r, c, CellValue::Empty))
            .collect();
        let restore = self.streams.clone();
        if let Err(e) = self.write_cells(&table, covered) {
            self.streams = restore;
            return Err(e);
        }

        let mut store = owner
            .bytes(2)
            .and_then(crate::pb::decode_nested)
            .unwrap_or_default();
        let index = store.varint(2).unwrap_or(0);
        let mut formula = Message::default();
        formula.set_in_order(1, Value::Bytes(node));
        let mut pair = Message::default();
        pair.set_in_order(1, Value::Varint(index));
        pair.set_in_order(2, Value::Bytes(formula.encode()));
        store.set_in_order(2, Value::Varint(index + 1));
        store.append_in_order(3, Value::Bytes(pair.encode()));
        owner.set_in_order(2, Value::Bytes(store.encode()));
        model_archive.set(47, Value::Bytes(owner.encode()));
        if let Err(e) = self.set_archive(model, &model_archive) {
            self.streams = restore;
            return Err(e);
        }
        Ok(())
    }

    /// Take a merge apart, leaving its cells where they are — which is empty,
    /// the merge having emptied them.
    ///
    /// The formula naming the range goes out of the merge owner's store;
    /// `next_formula_index` is left alone, because it is a high-water mark and
    /// the app only ever raises it.
    pub fn unmerge_cells(&mut self, wanted: &str, row: usize, column: usize) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!(
            "{}: unmerge {}",
            table.name,
            crate::table::CellRef::Index(row, column)
        );
        let merge = table
            .merges
            .iter()
            .find(|merge| merge.row == row && merge.column == column)
            .ok_or_else(|| {
                Error::refused(
                    Refusal::NotFound,
                    format!("{where_}: no merge begins there"),
                )
            })?;
        let node = merge_range_node(&table, merge.row, merge.column, merge.rows, merge.columns)?;

        let model = table.model;
        self.refuse_if_patched(&[model], &where_)?;
        let mut model_archive = self.archive_of(model)?;
        let mut owner = model_archive
            .bytes(47)
            .and_then(crate::pb::decode_nested)
            .ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!("{where_}: the table has no merge owner"),
                )
            })?;
        let mut store = owner
            .bytes(2)
            .and_then(crate::pb::decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the merge owner has no store")))?;

        // The pair whose formula holds this range, found by the bytes of the
        // range node rather than by index — the indices are the app's and this
        // crate does not renumber them.
        let position = store.fields.iter().position(|field| {
            let Value::Bytes(raw) = &field.value else {
                return false;
            };
            field.number == 3
                && crate::pb::decode_nested(raw)
                    .and_then(|pair| pair.bytes(2).and_then(crate::pb::decode_nested))
                    .and_then(|formula| formula.bytes(1).map(<[u8]>::to_vec))
                    .is_some_and(|nodes| nodes == node)
        });
        let Some(position) = position else {
            return Err(Error::Format(format!(
                "{where_}: the merge owner holds no formula for that range in the shape this \
                 crate writes, so removing one would be guesswork"
            )));
        };
        store.fields.remove(position);
        owner.set_in_order(2, Value::Bytes(store.encode()));
        model_archive.set(47, Value::Bytes(owner.encode()));
        self.set_archive(model, &model_archive)
    }

    // -- deleting a row or a column ------------------------------------------

    /// Delete row `at`, with everything in it.
    ///
    /// The mirror of [`Document::insert_row`] and then some: a deleted row
    /// takes its cells with it, and a cell's **references have to be given
    /// back** — every string, format and control key it held — or the list's
    /// reference counts stop matching the cells that point at them, which is
    /// exactly what [`crate::table::Table::audit`] reports.
    ///
    /// What moves is what an insert moves, backwards: `number_of_rows`, every
    /// `TileRowInfo` below `at` (across tile boundaries, so a row can change
    /// tiles), the row header bucket, and the row half of the
    /// `ColumnRowUIDMapArchive` — rebuilt and re-sorted by UUID, the order the
    /// app keeps it in.
    ///
    /// Refused by name, each before a byte moves: the table's last row, a
    /// header or footer row (whose count would have to change with it), a
    /// categorised, filtered, pivoted or conditionally highlighted table, a
    /// table with hidden or collapsed rows, a merge at or below the row, a row
    /// holding a formula, any formula anywhere that names the row or a range
    /// across it, and any object carrying version patches.
    pub fn delete_row(&mut self, wanted: &str, at: usize) -> Result<(), Error> {
        self.delete_line(wanted, at, true)
    }

    /// Delete column `at`, with everything in it.
    ///
    /// The column half of [`Document::delete_row`], and the same asymmetry an
    /// insert has: a row is an object and a column is one entry in every row's
    /// offset array, so this rewrites every row of every tile — slice at the
    /// offsets, drop the slot, lay the row back out. The array keeps the length
    /// it arrived with, a `-1` taking the place at the end, because the padding
    /// is what a reader steps through.
    pub fn delete_column(&mut self, wanted: &str, at: usize) -> Result<(), Error> {
        self.delete_line(wanted, at, false)
    }

    /// Both deletes: the refusals, the plan, and the commit.
    fn delete_line(&mut self, wanted: &str, at: usize, row: bool) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let what = if row { "row" } else { "column" };
        let where_ = format!("{}: delete {what} {at}", table.name);
        let (count, headers, footers) = match row {
            true => (table.rows, table.header_rows, table.footer_rows),
            false => (table.columns, table.header_columns, 0),
        };

        if at >= count {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!("{where_}: the table has {count} {what}(s)"),
            ));
        }
        if count == 1 {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!(
                    "{where_}: it is the table's only {what}, and a table with none is not \
                 something the app makes"
                ),
            ));
        }
        if at < headers as usize {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: it is one of the table's {headers} header {what}(s), and whether \
                 the count follows the {what} or the next one becomes a header is not \
                 something any fixture here can settle"
                ),
            ));
        }
        if footers > 0 && at >= count - footers as usize {
            return Err(Error::refused(
                Refusal::Organised,
                format!("{where_}: it is one of the table's {footers} footer row(s)"),
            ));
        }
        self.refuse_if_organised(&table, row, &where_)?;
        if let Some(merge) = table.merges.iter().find(|m| {
            if row {
                m.row + m.rows > at
            } else {
                m.column + m.columns > at
            }
        }) {
            return Err(Error::refused(
                Refusal::Merged,
                format!(
                "{where_}: a merge at row {} column {} would shift or straddle it, and a merge \
                 is stored as an absolute formula this crate does not rewrite",
                merge.row, merge.column
            ),
            ));
        }
        if let Some((formula_row, formula_column, _)) = table
            .formula_cells()
            .into_iter()
            .find(|(r, c, _)| if row { *r == at } else { *c == at })
        {
            return Err(Error::refused(
                Refusal::HoldsFormula,
                format!(
                    "{where_}: r{formula_row}c{formula_column} holds a formula, and removing one \
                 means taking its dependency edges out of the calculation engine"
                ),
            ));
        }
        if let Some(reason) = self.delete_would_break_a_formula(&table, at, row) {
            return Err(Error::Format(format!("{where_}: {reason}")));
        }

        // The references the deleted cells hold, which have to be given back —
        // and the cell counts the *other* axis keeps, one per cell that goes.
        let site = self.table_site(&table)?;
        let mut released: Vec<(u64, u32)> = Vec::new();
        let mut across: std::collections::BTreeMap<usize, i64> = Default::default();
        for cell in table.cells() {
            if if row {
                cell.row != at
            } else {
                cell.column != at
            } {
                continue;
            }
            *across
                .entry(if row { cell.column } else { cell.row })
                .or_default() -= 1;
            for (list, key) in [
                (site.strings, cell.record.string_id),
                (site.controls, cell.record.control_id),
            ]
            .into_iter()
            .chain(
                ALL_SLOTS
                    .iter()
                    .map(|slot| (site.formats, cell.record.format_id_in(*slot))),
            ) {
                if let (Some(list), Some(key)) = (list, key) {
                    released.push((list, key));
                }
            }
        }

        let plan = match row {
            true => self.plan_row_delete(&table, at, &where_)?,
            false => self.plan_column_delete(&table, at, &where_)?,
        };
        let mut touched = plan.touched.clone();
        touched.extend(released.iter().map(|(list, _)| *list));
        self.refuse_if_patched(&touched, &where_)?;

        // Commit: every object was decoded during planning.
        let mut cache = ListCache::default();
        for (list, key) in &released {
            self.step_list_entry(&mut cache, *list, *key, -1)?;
        }
        cache.flush(self)?;
        self.set_archive(plan.model, &plan.model_archive)?;
        for (tile, archive) in &plan.tiles {
            self.set_archive(*tile, archive)?;
        }
        self.set_archive(plan.bucket, &plan.bucket_archive)?;
        self.set_archive(plan.uid_map, &plan.uid_map_archive)?;
        // A deleted row takes one cell out of each column that had one, and a
        // deleted column one out of each row — the counts the *other* axis's
        // header keeps, which `Table::audit` checks against the cells present.
        if row {
            if let Some(bucket) = site.column_bucket {
                self.step_bucket_counts(bucket, &across)?;
            }
        } else {
            self.step_row_counts(&site, &across)?;
        }
        Ok(())
    }

    /// The organisation a row or column delete cannot maintain.
    ///
    /// The same list an insert refuses, and for the same reasons: a category's
    /// group nodes address rows by index, a filter's hidden state by UUID, a
    /// pivot's rows are the app's own, a conditional rule's range is addressed
    /// either way — and no fixture here proves any of them across a delete.
    fn refuse_if_organised(
        &self,
        table: &crate::table::Table,
        row: bool,
        where_: &str,
    ) -> Result<(), Error> {
        if !table.categories.is_empty() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table is categorised, and a category's groups are addressed by \
                 row index"
                ),
            ));
        }
        if table.pivot.is_some() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table is a pivot table, whose shape the app builds from its \
                 source"
                ),
            ));
        }
        if table.filter.is_some() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table is filtered, and a filter names the column it tests and \
                 the rows it hides"
                ),
            ));
        }
        if !table.conditional_styles.is_empty() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table carries conditional highlighting, whose rule ranges are \
                 row and column addressed"
                ),
            ));
        }
        let (states, extents) = match row {
            true => (&table.row_states, &table.row_extents),
            false => (&table.column_states, &table.column_extents),
        };
        if !states.user_hidden.is_empty()
            || !states.filtered.is_empty()
            || !states.collapsed_groups.is_empty()
            || extents.iter().any(|extent| extent.hidden())
        {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                "{where_}: the table has hidden or collapsed rows or columns, addressed by UUID"
            ),
            ));
        }
        Ok(())
    }

    /// Whether any formula anywhere would be broken by the delete.
    ///
    /// Stricter than the insert's check, and it has to be: an insert leaves
    /// every referenced cell in existence, while a delete **takes cells away**,
    /// so a reference to the deleted line becomes a `#REF!` however it was
    /// written. Refused: a bounded reference that names the line, a range that
    /// spans it, a relative reference whose host and referent fall on opposite
    /// sides of it, and any reference this crate cannot read.
    fn delete_would_break_a_formula(
        &self,
        target: &crate::table::Table,
        at: usize,
        row: bool,
    ) -> Option<String> {
        use crate::formula::Axis;
        let at = at as i64;
        let target_base = target.base_uid;
        for t in self.tables() {
            let same_table = t.model == target.model;
            for (host_row, host_column, formula) in t.formula_cells() {
                let host = if row { host_row } else { host_column } as i64;
                for node in &formula.ast.nodes {
                    if !is_reference_node(node.kind) {
                        continue;
                    }
                    let Some(reference) = node.reference() else {
                        if same_table || is_cross_table_node(node.kind) {
                            return Some(format!(
                                "table {} holds a formula with a reference this crate cannot \
                                 analyse ({}), so the delete might silently break it",
                                t.name,
                                crate::formula::node::name(node.kind)
                            ));
                        }
                        continue;
                    };
                    if reference.is_error {
                        continue;
                    }
                    let into_target = match reference.table {
                        None => same_table,
                        Some(uid) => {
                            uid == target_base && target_base != crate::table::Uuid::default()
                        }
                    };
                    if !into_target {
                        continue;
                    }
                    let cross = reference.table.is_some();
                    let (begin, end) = match row {
                        true => (reference.row, reference.row_end),
                        false => (reference.column, reference.column_end),
                    };
                    // Resolve both ends against the host, where they are bounded.
                    let resolve = |axis: Axis| match axis {
                        Axis::Unbounded => None,
                        Axis::Absolute(index) => Some(index),
                        Axis::Relative(_) if cross => Some(at), // unresolvable: refuse
                        Axis::Relative(offset) => Some(host + offset),
                    };
                    let (Some(first), Some(last)) = (resolve(begin), resolve(end)) else {
                        // A whole row or column: the axis is unbounded and the
                        // app keeps it whatever leaves the table.
                        continue;
                    };
                    let (low, high) = (first.min(last), first.max(last));
                    if (low..=high).contains(&at) {
                        return Some(format!(
                            "table {} has a formula whose reference names this {} or a range \
                             across it",
                            t.name,
                            if row { "row" } else { "column" }
                        ));
                    }
                    // Both ends outside, but a relative reference that crosses
                    // the line shifts by one and its host may not.
                    for index in [first, last] {
                        if (host < at) != (index < at) {
                            return Some(format!(
                                "table {} has a formula whose reference would shift relative \
                                 to what it names",
                                t.name
                            ));
                        }
                    }
                }
            }
        }
        None
    }

    // -- inserting a column --------------------------------------------------

    /// Insert an empty column before index `at`; `at == columns` appends.
    ///
    /// Transactional the same way [`Document::insert_row`] is: the whole write
    /// is planned, and every case this crate cannot maintain safely is refused
    /// by name, before a byte moves.
    ///
    /// **A column is not a row turned sideways.** A row is a `TileRowInfo` of
    /// its own and inserting one shifts whole objects; a column exists only as
    /// *one entry in every row's offset array*, so inserting one rewrites every
    /// row of every tile — slice the row into its per-column records, open a
    /// gap at `at`, lay them back out, and every offset after the gap moves.
    /// That is also why a column insert is **not** limited to a single tile the
    /// way a row insert is: the work is per row, and a tile boundary is a row
    /// boundary, so a second tile is more of the same rather than something new.
    ///
    /// The offset array keeps the length it arrived with — Numbers pads it to
    /// 255 entries whatever the table's width, and that padding is what a
    /// reader steps through — so the last entry falls off the end as the gap
    /// opens. A table already 255 columns wide is refused rather than losing a
    /// column out of the back.
    ///
    /// **What it refuses, by name**, for the same reasons the row insert does:
    /// a categorised, pivoted or filtered table; conditional highlighting;
    /// hidden columns; a merge at or straddling the insertion; any formula
    /// whose reference to this table names a column at or after it; and any
    /// object the write would touch that carries version patches.
    pub fn insert_column(&mut self, wanted: &str, at: usize) -> Result<(), Error> {
        let table = self.table_for_write(wanted)?;
        let where_ = format!("{}: insert column at {at}", table.name);

        if at > table.columns {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!(
                    "{where_}: the table has {} column(s), so a column goes in at 0..={}",
                    table.columns, table.columns
                ),
            ));
        }
        if table.columns >= MAX_COLUMNS {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!(
                "{where_}: the table already has {} columns, which is what a row's offset array \
                 holds — one more would fall off the end of every row",
                table.columns
            ),
            ));
        }
        if !table.categories.is_empty() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table is categorised, and a category is built on a column — \
                 which column it is is stored by index"
                ),
            ));
        }
        if table.pivot.is_some() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table is a pivot table, whose columns the app builds from its \
                 source"
                ),
            ));
        }
        if table.filter.is_some() {
            return Err(Error::refused(
                Refusal::Organised,
                format!("{where_}: the table is filtered, and a filter names the column it tests"),
            ));
        }
        if !table.conditional_styles.is_empty() {
            return Err(Error::refused(
                Refusal::Organised,
                format!(
                    "{where_}: the table carries conditional highlighting, whose rule ranges are \
                 column addressed"
                ),
            ));
        }
        if !table.column_states.user_hidden.is_empty()
            || table.column_extents.iter().any(|extent| extent.hidden())
        {
            return Err(Error::refused(
                Refusal::Organised,
                format!("{where_}: the table has hidden columns, addressed by UUID"),
            ));
        }
        if let Some(merge) = table.merges.iter().find(|m| m.column + m.columns > at) {
            return Err(Error::refused(
                Refusal::Merged,
                format!(
                "{where_}: a merge at row {} column {} would shift or straddle the new column, \
                 and a merge is stored as an absolute formula this crate does not rewrite",
                merge.row, merge.column
            ),
            ));
        }
        if let Some(reason) = self.column_insert_would_break_a_formula(&table, at) {
            return Err(Error::Format(format!("{where_}: {reason}")));
        }

        let plan = self.plan_column_insert(&table, at, &where_)?;
        self.refuse_if_patched(&plan.touched, &where_)?;

        self.set_archive(plan.model, &plan.model_archive)?;
        for (tile, archive) in &plan.tiles {
            self.set_archive(*tile, archive)?;
        }
        self.set_archive(plan.column_bucket, &plan.bucket_archive)?;
        self.set_archive(plan.uid_map, &plan.uid_map_archive)?;
        Ok(())
    }

    /// Would inserting a column at `at` shift what a formula refers to?
    ///
    /// The column-axis twin of
    /// [`Document::row_insert_would_break_a_formula`], and the reasoning is the
    /// same: a whole-*row* reference (column axis `Unbounded`) is unaffected, an
    /// absolute column is safe only left of the insertion, and a relative one is
    /// safe only when host and referent fall on the same side of it.
    fn column_insert_would_break_a_formula(
        &self,
        target: &crate::table::Table,
        at: usize,
    ) -> Option<String> {
        use crate::formula::Axis;
        let at = at as i64;
        let target_base = target.base_uid;
        for t in self.tables() {
            let same_table = t.model == target.model;
            for (_row, column, formula) in t.formula_cells() {
                let host_column = column as i64;
                for node in &formula.ast.nodes {
                    if !is_reference_node(node.kind) {
                        continue;
                    }
                    let Some(reference) = node.reference() else {
                        if same_table || is_cross_table_node(node.kind) {
                            return Some(format!(
                                "table {} holds a formula with a reference this crate cannot \
                                 analyse ({}), so an inserted column might silently break it",
                                t.name,
                                crate::formula::node::name(node.kind)
                            ));
                        }
                        continue;
                    };
                    if reference.is_error {
                        continue;
                    }
                    let (into_target, cross) = match reference.table {
                        None => (same_table, false),
                        Some(uid) => (
                            uid == target_base && target_base != crate::table::Uuid::default(),
                            true,
                        ),
                    };
                    if !into_target {
                        continue;
                    }
                    for axis in [reference.column, reference.column_end] {
                        let unsafe_reason = match axis {
                            Axis::Unbounded => None,
                            Axis::Absolute(c) => {
                                (c >= at).then_some("names a column at or after it")
                            }
                            Axis::Relative(_) if cross => {
                                Some("is a relative cross-table reference")
                            }
                            Axis::Relative(offset) => {
                                let referent = host_column + offset;
                                ((host_column < at) != (referent < at))
                                    .then_some("would shift relative to the column it names")
                            }
                        };
                        if let Some(reason) = unsafe_reason {
                            return Some(format!(
                                "table {} has a formula whose reference to this table {reason}",
                                t.name
                            ));
                        }
                    }
                }
            }
        }
        None
    }

    /// Locate every object a column insert rewrites and build each one's new
    /// form. Pure, like [`Document::plan_row_insert`].
    fn plan_column_insert(
        &self,
        table: &crate::table::Table,
        at: usize,
        where_: &str,
    ) -> Result<ColumnInsertPlan, Error> {
        use crate::pb::decode_nested;
        let model = table.model;
        let mut model_archive = self.archive_of(model)?;
        let store = model_archive
            .bytes(4)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the model has no data store")))?;

        let tiles = store
            .bytes(3)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the data store has no tiles")))?;
        let tile_ids: Vec<u64> = tiles
            .all(1)
            .filter_map(|value| match value {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .filter_map(|entry| entry.bytes(2).and_then(crate::table::reference))
            .collect();
        if tile_ids.is_empty() {
            return Err(Error::refused(
                Refusal::Missing,
                format!("{where_}: the table has no tiles"),
            ));
        }

        // The column headers are **one bucket**, not a list of them: the
        // row/column asymmetry at `DataStore` 1 and 2 is Apple's, not a
        // simplification here.
        let column_bucket = store
            .bytes(2)
            .and_then(crate::table::reference)
            .ok_or_else(|| {
                Error::refused(
                    Refusal::Missing,
                    format!("{where_}: the data store has no column headers"),
                )
            })?;

        let uid_map = model_archive
            .bytes(46)
            .and_then(crate::table::reference)
            .ok_or_else(|| {
                Error::Format(format!(
                    "{where_}: the model has no ColumnRowUIDMap, so a new column cannot be \
                     given a UUID"
                ))
            })?;

        let mut plan_tiles = Vec::new();
        for tile in &tile_ids {
            plan_tiles.push((*tile, self.tile_with_inserted_column(*tile, at, where_)?));
        }
        let bucket_archive = self.bucket_with_inserted_column(column_bucket, at, where_)?;
        let uid_map_archive = self.uid_map_with_inserted_column(uid_map, at, table, where_)?;
        model_archive.set(7, Value::Varint((table.columns + 1) as u64));

        let mut touched = vec![model, column_bucket, uid_map];
        touched.extend(&tile_ids);
        Ok(ColumnInsertPlan {
            touched,
            model,
            model_archive,
            tiles: plan_tiles,
            column_bucket,
            bucket_archive,
            uid_map,
            uid_map_archive,
        })
    }

    /// Every row of one tile with a gap opened at column `at`.
    ///
    /// The row is sliced into its per-column records, a `None` is spliced in,
    /// and the whole row is laid back out — which moves every offset after the
    /// gap. The array is truncated back to the length it arrived with, because
    /// its padding is what a reader steps through and lengthening it would say
    /// the table is wider than it is.
    fn tile_with_inserted_column(
        &self,
        tile: u64,
        at: usize,
        where_: &str,
    ) -> Result<Message, Error> {
        use crate::pb::decode_nested;
        let mut archive = self.archive_of(tile)?;
        for field in archive.fields.iter_mut() {
            if field.number != 5 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let mut info = decode_nested(raw)
                .ok_or_else(|| Error::Format(format!("{where_}: a tile row does not decode")))?;
            let buffer = info.bytes(6).unwrap_or(&[]).to_vec();
            let offsets = info.bytes(7).unwrap_or(&[]).to_vec();
            if offsets.is_empty() {
                continue;
            }
            let wide = info.varint(8).unwrap_or(0) != 0;
            let entries = offsets.len() / 2;
            if at >= entries {
                return Err(Error::Format(format!(
                    "{where_}: a row's offset array holds {entries} column(s), so there is no \
                     column {at} in it"
                )));
            }
            let mut records: Vec<Option<Vec<u8>>> = vec![None; entries];
            for (column, bytes) in crate::table::row_cells(&buffer, &offsets, wide)
                .into_iter()
                .flatten()
            {
                records[column] = Some(bytes.to_vec());
            }
            records.insert(at, None);
            // The one that falls off the back must be empty, or a cell would be
            // lost — which the width check above has already made impossible.
            if records.pop().flatten().is_some() {
                return Err(Error::Format(format!(
                    "{where_}: the last column of a row holds a cell, so opening a gap would \
                     push it off the end of the offset array"
                )));
            }
            let (buffer, offsets) = crate::table::encode_row(&records, wide)
                .map_err(|e| Error::Format(format!("{where_}: {e}")))?;
            info.set(6, Value::Bytes(buffer));
            info.set(7, Value::Bytes(offsets));
            field.value = Value::Bytes(info.encode());
        }
        // `maxColumn` (1), `maxRow` (2) and `numCells` (3) are **dead**: all
        // three are 0 on every tile in the corpus, including a tile holding
        // 2411 cells across nine columns and 256 rows. A gap opened in the
        // middle of a row changes none of them, and writing a number into a
        // field the app writes as zero would be inventing one.
        Ok(archive)
    }

    /// The column-header bucket with entries at or after `at` shifted right,
    /// and a zero-cell entry added for the inserted column.
    fn bucket_with_inserted_column(
        &self,
        bucket: u64,
        at: usize,
        where_: &str,
    ) -> Result<Message, Error> {
        use crate::pb::decode_nested;
        let mut archive = self.archive_of(bucket)?;
        for field in archive.fields.iter_mut() {
            if field.number != 2 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let mut entry = decode_nested(raw).ok_or_else(|| {
                Error::Format(format!("{where_}: a column-header entry does not decode"))
            })?;
            let index = entry.varint(1).unwrap_or(0) as usize;
            if index >= at {
                entry.set(1, Value::Varint((index + 1) as u64));
                field.value = Value::Bytes(entry.encode());
            }
        }
        let mut entry = Message::default();
        entry.set_in_order(1, Value::Varint(at as u64));
        // A literal 0 is "still at the table's default width", which is what a
        // column nobody has dragged carries.
        entry.set_in_order(2, Value::Fixed32(0f32.to_le_bytes()));
        entry.set_in_order(3, Value::Varint(0));
        entry.set_in_order(4, Value::Varint(0));
        archive.append_in_order(2, Value::Bytes(entry.encode()));
        Ok(archive)
    }

    /// The `ColumnRowUIDMapArchive` with a fresh column UUID at index `at`.
    ///
    /// The column half is fields 1, 2 and 3 — the same three arrays as the row
    /// half at 4, 5 and 6, and rebuilt the same way: shift, mint, sort by the
    /// 128-bit value. The row half is untouched.
    fn uid_map_with_inserted_column(
        &self,
        uid_map: u64,
        at: usize,
        table: &crate::table::Table,
        where_: &str,
    ) -> Result<Message, Error> {
        use crate::table::Uuid;
        let mut archive = self.archive_of(uid_map)?;

        let uuids: Vec<Uuid> = archive
            .all(1)
            .filter_map(|value| match value {
                Value::Bytes(raw) => crate::pb::decode_nested(raw).map(|m| Uuid::decode(&m)),
                _ => None,
            })
            .collect();
        let indices: Vec<u64> = archive
            .all(2)
            .filter_map(|value| match value {
                Value::Varint(n) => Some(*n),
                _ => None,
            })
            .collect();
        if uuids.len() != indices.len() {
            return Err(Error::Format(format!(
                "{where_}: the UUID map has {} column UUID(s) and {} index(es); a shape this \
                 crate did not write is not safe to rebuild",
                uuids.len(),
                indices.len()
            )));
        }

        let mut pairs: Vec<(Uuid, usize)> = uuids
            .iter()
            .zip(&indices)
            .map(|(&uuid, &index)| {
                let index = index as usize;
                (uuid, if index >= at { index + 1 } else { index })
            })
            .collect();
        let existing: BTreeSet<Uuid> = pairs.iter().map(|(uuid, _)| *uuid).collect();
        // The row minter, seeded by the same table and the same index — a
        // column UUID has to be unique within the table, not unlike a row's,
        // and the seed differs because the arrays it is checked against do.
        let new_uuid = mint_row_uuid(&existing, table, at);
        pairs.push((new_uuid, at));
        pairs.sort_by_key(|(uuid, _)| (uuid.upper, uuid.lower));

        let count = pairs.len();
        let mut uid_for_index = vec![0u32; count];
        for (position, (_, index)) in pairs.iter().enumerate() {
            uid_for_index[*index] = position as u32;
        }

        archive.clear(1);
        archive.clear(2);
        archive.clear(3);
        for (uuid, _) in &pairs {
            archive.append_in_order(1, Value::Bytes(encode_uuid(*uuid)));
        }
        for (_, index) in &pairs {
            archive.append_in_order(2, Value::Varint(*index as u64));
        }
        for &position in &uid_for_index {
            archive.append_in_order(3, Value::Varint(u64::from(position)));
        }
        Ok(archive)
    }

    /// Would inserting a row at `at` shift what a formula refers to?
    ///
    /// A formula's cell references are stored as indexes (absolute) or offsets
    /// from the host cell (relative), and nothing here rewrites a `TSCE` AST. So
    /// every formula that could name a row of the target table is examined:
    ///
    /// * a whole-column reference (the row axis is `Unbounded`) is unaffected;
    /// * a stored `#REF!` is already broken and stays so;
    /// * an **absolute** row `r` is safe only when `r < at` — otherwise the cell
    ///   it names moves out from under it;
    /// * a **relative** row is safe only when host and referent fall on the same
    ///   side of `at`, so that both shift together or neither does.
    ///
    /// Anything this crate cannot classify — a reference node whose shape it does
    /// not read — is treated as unsafe rather than assumed harmless.
    fn row_insert_would_break_a_formula(
        &self,
        target: &crate::table::Table,
        at: usize,
    ) -> Option<String> {
        use crate::formula::Axis;
        let at = at as i64;
        let target_base = target.base_uid;
        for t in self.tables() {
            let same_table = t.model == target.model;
            for (row, _column, formula) in t.formula_cells() {
                let host_row = row as i64;
                for node in &formula.ast.nodes {
                    if !is_reference_node(node.kind) {
                        continue;
                    }
                    let Some(reference) = node.reference() else {
                        // A reference this crate does not read. In the target
                        // table it may name a target row; in another table, only
                        // a cross-table-capable node could reach the target.
                        if same_table || is_cross_table_node(node.kind) {
                            return Some(format!(
                                "table {} holds a formula with a reference this crate cannot \
                                 analyse ({}), so an inserted row might silently break it",
                                t.name,
                                crate::formula::node::name(node.kind)
                            ));
                        }
                        continue;
                    };
                    if reference.is_error {
                        continue;
                    }
                    let (into_target, cross) = match reference.table {
                        None => (same_table, false),
                        Some(uid) => (
                            uid == target_base && target_base != crate::table::Uuid::default(),
                            true,
                        ),
                    };
                    if !into_target {
                        continue;
                    }
                    for axis in [reference.row, reference.row_end] {
                        let unsafe_reason = match axis {
                            Axis::Unbounded => None,
                            Axis::Absolute(r) => (r >= at).then_some("names a row at or below it"),
                            Axis::Relative(_) if cross => {
                                Some("is a relative cross-table reference")
                            }
                            Axis::Relative(offset) => {
                                let referent = host_row + offset;
                                ((host_row < at) != (referent < at))
                                    .then_some("would shift relative to the row it names")
                            }
                        };
                        if let Some(reason) = unsafe_reason {
                            return Some(format!(
                                "table {} has a formula whose reference to this table {reason}",
                                t.name
                            ));
                        }
                    }
                }
            }
        }
        None
    }

    /// Locate every object a row insert rewrites and build each one's new form.
    ///
    /// Pure: it reads the document and returns the archives to store, so
    /// [`Document::insert_row`] can prove the whole write before applying any of
    /// it. Every fallible step — tiles that do not run 0, 1, 2…, a table that
    /// fills them, a missing UUID map, a
    /// hidden-state extent that keys rows — refuses here.
    fn plan_row_insert(
        &self,
        table: &crate::table::Table,
        at: usize,
        where_: &str,
    ) -> Result<RowInsertPlan, Error> {
        use crate::pb::decode_nested;
        let model = table.model;
        let mut model_archive = self.archive_of(model)?;
        let store = model_archive
            .bytes(4)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the model has no data store")))?;

        // One tile, covering row 0 up, with room for one more row.
        let tiles = store
            .bytes(3)
            .and_then(decode_nested)
            .ok_or_else(|| Error::Format(format!("{where_}: the data store has no tiles")))?;
        let tile_size = tiles.varint(2).unwrap_or(256) as usize;
        let tile_entries: Vec<(u64, u64)> = tiles
            .all(1)
            .filter_map(|value| match value {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .filter_map(|entry| {
                let id = entry.varint(1)?;
                let reference = entry.bytes(2).and_then(crate::table::reference)?;
                Some((id, reference))
            })
            .collect();
        if tile_entries.is_empty() {
            return Err(Error::refused(
                Refusal::Missing,
                format!("{where_}: the table has no tiles"),
            ));
        }
        // Tile *k* covers rows `k * tile_size ..`, so the tiles have to be the
        // run 0, 1, 2… for an absolute row index to mean anything.
        for (position, (tile_id, _)) in tile_entries.iter().enumerate() {
            if *tile_id != position as u64 {
                return Err(Error::Format(format!(
                    "{where_}: tile {position} is numbered {tile_id}, and a row's absolute index \
                     is its tile's number times {tile_size} plus its index within it"
                )));
            }
        }
        // The capacity the tiles that exist provide. A row pushed past it would
        // need a tile of its own — a new object *and* a new component — which
        // nothing in the corpus shows the app making mid-table.
        let capacity = tile_entries.len() * tile_size;
        if table.rows >= capacity {
            return Err(Error::Format(format!(
                "{where_}: the table fills its {} tile(s) of {tile_size} rows, so a new row \
                 would have to open another one",
                tile_entries.len()
            )));
        }

        // One row-header bucket.
        let header_storage = store.bytes(1).and_then(decode_nested).ok_or_else(|| {
            Error::refused(
                Refusal::Missing,
                format!("{where_}: the data store has no row headers"),
            )
        })?;
        let buckets: Vec<u64> = header_storage
            .all(2)
            .filter_map(|value| match value {
                Value::Bytes(raw) => crate::table::reference(raw),
                _ => None,
            })
            .collect();
        if buckets.len() != 1 {
            return Err(Error::Format(format!(
                "{where_}: the row headers span {} buckets, not one",
                buckets.len()
            )));
        }
        let row_bucket = buckets[0];

        // The UUID map, and the check that the hidden-state extent keys no rows.
        let uid_map = model_archive
            .bytes(46)
            .and_then(crate::table::reference)
            .ok_or_else(|| {
                Error::Format(format!(
                "{where_}: the model has no ColumnRowUIDMap, so a new row cannot be given a UUID"
            ))
            })?;
        self.refuse_if_row_extent_keys_rows(&model_archive, where_)?;

        let tiles = self.tiles_with_inserted_row(&tile_entries, tile_size, at, where_)?;
        let bucket_archive = self.bucket_with_inserted_row(row_bucket, at, where_)?;
        let uid_map_archive = self.uid_map_with_inserted_row(uid_map, at, table, where_)?;
        model_archive.set(6, Value::Varint((table.rows + 1) as u64));

        let mut touched = vec![model, row_bucket, uid_map];
        touched.extend(tiles.iter().map(|(tile, _)| *tile));
        Ok(RowInsertPlan {
            touched,
            model,
            model_archive,
            tiles,
            row_bucket,
            bucket_archive,
            uid_map,
            uid_map_archive,
        })
    }

    /// Refuse if the row hidden-state extent (model field 70) carries any
    /// `base_hidden_states` entries.
    ///
    /// A plain table's extent has none; a filtered or hand-hidden one carries an
    /// entry per row, each keyed by a row UUID, and inserting a row would mean
    /// minting an entry there too — which no corpus fixture proves.
    fn refuse_if_row_extent_keys_rows(&self, model: &Message, where_: &str) -> Result<(), Error> {
        use crate::pb::decode_nested;
        let Some(owner) = model.bytes(70).and_then(decode_nested) else {
            return Ok(());
        };
        for states in owner.all(2) {
            let Value::Bytes(raw) = states else { continue };
            let Some(states) = decode_nested(raw) else {
                continue;
            };
            // Field 3 is the row extent; field 2 within it is base_hidden_states.
            if let Some(extent) = states.bytes(3).and_then(decode_nested) {
                if extent.get(2).is_some() {
                    return Err(Error::Format(format!(
                        "{where_}: the row hidden-state extent keys rows by UUID, and updating \
                         it across an insert is unverified here"
                    )));
                }
            }
        }
        Ok(())
    }

    /// The tile with every `TileRowInfo` at or below `at` shifted down one row.
    ///
    /// The inserted row is empty, so it gets no `TileRowInfo` — and `numrows`
    /// (field 4), which counts them, does not change.
    /// Every tile with the rows at or below `at` shifted down one.
    ///
    /// **A shift can cross a tile boundary.** A row's absolute index is
    /// `tileid * tile_size + tile_row_index`, so the last row of tile 0 moving
    /// down one becomes the first row of tile 1 — the `TileRowInfo` has to
    /// leave one object and join another. So this does not edit tiles in
    /// place: it gathers every row of every tile by absolute index, shifts the
    /// indexes, and lays the rows back out into the tile each now belongs to.
    ///
    /// Only field 5 and field 4 are written. `maxColumn`, `maxRow` and
    /// `numCells` (1, 2, 3) are dead — `0` on every tile in the corpus,
    /// including one holding 2411 cells — and everything else in the tile is
    /// left exactly as it was.
    fn tiles_with_inserted_row(
        &self,
        tiles: &[(u64, u64)],
        tile_size: usize,
        at: usize,
        where_: &str,
    ) -> Result<Vec<(u64, Message)>, Error> {
        use crate::pb::decode_nested;

        // Gather: every row of every tile, by the index it has in the table.
        let mut rows: Vec<(usize, Message)> = Vec::new();
        let mut archives: Vec<(u64, Message)> = Vec::new();
        for (number, tile) in tiles {
            let archive = self.archive_of(*tile)?;
            for value in archive.all(5) {
                let Value::Bytes(raw) = value else { continue };
                let info = decode_nested(raw).ok_or_else(|| {
                    Error::Format(format!("{where_}: a tile row does not decode"))
                })?;
                let within = info.varint(1).unwrap_or(0) as usize;
                if within >= tile_size {
                    return Err(Error::Format(format!(
                        "{where_}: a row of tile {number} is at index {within}, past the \
                         {tile_size} rows a tile holds"
                    )));
                }
                rows.push((*number as usize * tile_size + within, info));
            }
            archives.push((*tile, archive));
        }

        // Shift, and lay back out into the tile each row now belongs to.
        let mut per_tile: BTreeMap<usize, Vec<(usize, Message)>> = BTreeMap::new();
        for (index, mut info) in rows {
            let index = if index >= at { index + 1 } else { index };
            let (tile, within) = (index / tile_size, index % tile_size);
            if tile >= tiles.len() {
                return Err(Error::Format(format!(
                    "{where_}: the shift would push a row into tile {tile}, which does not exist"
                )));
            }
            info.set(1, Value::Varint(within as u64));
            per_tile.entry(tile).or_default().push((within, info));
        }

        let mut out = Vec::new();
        for (position, (tile, mut archive)) in archives.into_iter().enumerate() {
            let mut mine = per_tile.remove(&position).unwrap_or_default();
            mine.sort_by_key(|(within, _)| *within);
            archive.clear(5);
            for (_, info) in &mine {
                archive.append_in_order(5, Value::Bytes(info.encode()));
            }
            // Field 4 counts the `TileRowInfo`s, and a row that moved out of
            // this tile is one it no longer counts.
            archive.set(4, Value::Varint(mine.len() as u64));
            out.push((tile, archive));
        }
        Ok(out)
    }

    /// The row-header bucket with entries at or below `at` shifted down, and a
    /// zero-cell entry added for the inserted row.
    fn bucket_with_inserted_row(
        &self,
        bucket: u64,
        at: usize,
        where_: &str,
    ) -> Result<Message, Error> {
        use crate::pb::decode_nested;
        let mut archive = self.archive_of(bucket)?;
        for field in archive.fields.iter_mut() {
            if field.number != 2 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let mut entry = decode_nested(raw).ok_or_else(|| {
                Error::Format(format!("{where_}: a row-header entry does not decode"))
            })?;
            let index = entry.varint(1).unwrap_or(0) as usize;
            if index >= at {
                entry.set(1, Value::Varint((index + 1) as u64));
                field.value = Value::Bytes(entry.encode());
            }
        }
        // The new empty row: default height, visible, no cells — the same shape
        // `set_cell` writes when it first gives a row an entry.
        let mut entry = Message::default();
        entry.set_in_order(1, Value::Varint(at as u64));
        entry.set_in_order(2, Value::Fixed32(0f32.to_le_bytes()));
        entry.set_in_order(3, Value::Varint(0));
        entry.set_in_order(4, Value::Varint(0));
        archive.append_in_order(2, Value::Bytes(entry.encode()));
        Ok(archive)
    }

    /// The `ColumnRowUIDMapArchive` with a fresh row UUID at index `at`.
    ///
    /// The row half (fields 4/5/6) is rebuilt from scratch: every existing row's
    /// index is shifted down past `at`, a new per-table-unique UUID is minted for
    /// the new row, and the three arrays are re-emitted sorted by UUID — the
    /// order the app keeps them in. The column half (fields 1/2/3) is untouched.
    fn uid_map_with_inserted_row(
        &self,
        uid_map: u64,
        at: usize,
        table: &crate::table::Table,
        where_: &str,
    ) -> Result<Message, Error> {
        use crate::table::Uuid;
        let mut archive = self.archive_of(uid_map)?;

        let uuids: Vec<Uuid> = archive
            .all(4)
            .filter_map(|value| match value {
                Value::Bytes(raw) => crate::pb::decode_nested(raw).map(|m| Uuid::decode(&m)),
                _ => None,
            })
            .collect();
        let indices: Vec<u64> = archive
            .all(5)
            .filter_map(|value| match value {
                Value::Varint(n) => Some(*n),
                _ => None,
            })
            .collect();
        if uuids.len() != indices.len() {
            return Err(Error::Format(format!(
                "{where_}: the UUID map has {} row UUID(s) and {} index(es); a shape this crate \
                 did not write is not safe to rebuild",
                uuids.len(),
                indices.len()
            )));
        }

        let mut pairs: Vec<(Uuid, usize)> = uuids
            .iter()
            .zip(&indices)
            .map(|(&uuid, &index)| {
                let index = index as usize;
                (uuid, if index >= at { index + 1 } else { index })
            })
            .collect();
        let existing: BTreeSet<Uuid> = pairs.iter().map(|(uuid, _)| *uuid).collect();
        let new_uuid = mint_row_uuid(&existing, table, at);
        pairs.push((new_uuid, at));
        // The app sorts by the 128-bit value, high half first.
        pairs.sort_by_key(|(uuid, _)| (uuid.upper, uuid.lower));

        let count = pairs.len();
        let mut uid_for_index = vec![0u32; count];
        for (position, (_, index)) in pairs.iter().enumerate() {
            uid_for_index[*index] = position as u32;
        }

        archive.clear(4);
        archive.clear(5);
        archive.clear(6);
        for (uuid, _) in &pairs {
            archive.append_in_order(4, Value::Bytes(encode_uuid(*uuid)));
        }
        for (_, index) in &pairs {
            archive.append_in_order(5, Value::Varint(*index as u64));
        }
        for &position in &uid_for_index {
            archive.append_in_order(6, Value::Varint(u64::from(position)));
        }
        Ok(archive)
    }

    // -- drawables -----------------------------------------------------------

    /// Every drawable in the document — images, shapes, text boxes, lines,
    /// movies, groups, tables and charts — in stream and z-order.
    ///
    /// See [`crate::drawable`] for what a drawable is made of and why the
    /// enumeration does not assume how deep the geometry sits.
    pub fn drawables(&self) -> Vec<crate::drawable::Drawable> {
        crate::drawable::drawables(self)
    }

    /// Every chart in the document, with its type, its placement, its private
    /// copy of its data and — in Numbers — the table references that feed it.
    ///
    /// See [`crate::chart`] for why those are two different things.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let doc = iwork::Document::open("Report.numbers")?;
    /// for chart in doc.charts() {
    ///     println!("{} on {}", chart.type_label(), chart.placement.as_str());
    ///     for series in chart.series() {
    ///         println!("  {:?}", series.name);
    ///     }
    /// }
    /// # Ok(()) }
    /// ```
    pub fn charts(&self) -> Vec<crate::chart::Chart> {
        crate::chart::charts(self)
    }

    /// Rewrite the private grid a chart draws.
    ///
    /// See [`crate::chart::set_chart_data`]: a Numbers chart fed by a table is
    /// refused, because its grid is a cache of formulas this crate does not
    /// evaluate.
    pub fn set_chart_data(
        &mut self,
        chart: u64,
        data: &crate::chart::ChartData,
    ) -> Result<(), Error> {
        crate::chart::set_chart_data(self, chart, data)
    }

    /// Put a copy of a chart the document already has somewhere else, drawing
    /// the data given.
    ///
    /// See [`crate::chart::add_chart`]: a chart is copied rather than invented,
    /// because a dozen theme styles stand behind it and none of them can be
    /// made up honestly.
    pub fn add_chart(
        &mut self,
        container: &str,
        from: u64,
        data: &crate::chart::ChartData,
        position: (f32, f32),
        size: (f32, f32),
    ) -> Result<u64, Error> {
        crate::chart::add_chart(self, container, from, data, position, size)
    }

    /// One drawable by object identifier.
    pub fn drawable(&self, identifier: u64) -> Option<crate::drawable::Drawable> {
        self.drawables()
            .into_iter()
            .find(|d| d.identifier == identifier)
    }

    /// A drawable's fill, stroke, shadow, reflection and opacity, resolved up
    /// the style chain.
    /// Every `TST.CellStyleArchive` in the document — how its tables' cells
    /// are painted.
    ///
    /// Numbers names these by role in the stylesheet
    /// (`tableCell-0-headerRowStyle`, `-bodyStyle`, and so on) and every table
    /// built on the same table style shares them.
    pub fn cell_styles(&self) -> Vec<crate::table::CellStyle> {
        crate::table::cell_styles(self)
    }

    /// Paint a cell style, or with `None` stop it painting.
    ///
    /// See [`crate::table::set_cell_style_fill`]: this repaints **every**
    /// table whose style names the cell style, because that is what a shared
    /// style is. The fill is the only property of a cell style this crate
    /// writes; the rest of the archive is left exactly as it was.
    pub fn set_cell_style_fill(
        &mut self,
        style: u64,
        colour: Option<crate::drawable::Color>,
    ) -> Result<(), Error> {
        crate::table::set_cell_style_fill(self, style, colour)
    }

    /// Paint a drawable, or with `None` leave it painting nothing.
    ///
    /// The identifier is the **drawable**, not its style. The first paint
    /// gives it a style of its own, parented to the one it was sharing, which
    /// is what the app does and the only form that survives the app's own
    /// save — see [`crate::drawable::set_fill`].
    pub fn set_object_fill(
        &mut self,
        drawable: u64,
        colour: Option<crate::drawable::Color>,
    ) -> Result<(), Error> {
        crate::drawable::set_fill(self, drawable, colour)
    }

    /// Outline a drawable: colour, and width in points.
    ///
    /// See [`crate::drawable::set_stroke`].
    pub fn set_object_stroke(
        &mut self,
        drawable: u64,
        colour: crate::drawable::Color,
        width: f32,
    ) -> Result<(), Error> {
        crate::drawable::set_stroke(self, drawable, colour, width)
    }

    /// How opaque a drawable is, 0.0 to 1.0.
    ///
    /// See [`crate::drawable::set_opacity`].
    pub fn set_object_opacity(&mut self, drawable: u64, opacity: f32) -> Result<(), Error> {
        crate::drawable::set_opacity(self, drawable, opacity)
    }

    pub fn object_style(&self, identifier: u64) -> Option<crate::drawable::ObjectStyle> {
        crate::drawable::object_style(self, identifier)
    }

    /// Move or resize a drawable.
    ///
    /// The rectangle is the one the **app** reports, which for a masked image
    /// is the mask's window and not the picture's own rectangle. Give `None`
    /// for either half to leave it alone.
    ///
    /// What travels with it, because the app maintains it and a document that
    /// does not is inconsistent with every other document:
    ///
    /// * an **unmasked** media drawable's `originalSize`, which Keynote and
    ///   Pages both rewrote to the new size when a script resized an image. A
    ///   masked image's `originalSize` is left alone: the app fills it with the
    ///   mask window rather than the picture there, and not even consistently
    ///   (see the body of this method), so there is no size to rewrite it to;
    /// * a mask's geometry and its path source's natural size. Resizing a
    ///   masked image scales the whole assembly by one factor: Pages, asked to
    ///   make a 475-point-wide masked photo 300 wide, multiplied the picture's
    ///   own size, the mask's offset, the mask's size and the mask path's
    ///   natural size by 300/475 and moved the picture so that
    ///   `image.position + mask.position` still landed on the frame's corner.
    ///   That is what this reproduces, with the horizontal and vertical factors
    ///   taken separately — which reduces to what was observed whenever the
    ///   resize is proportional, and is **unverified** when it is not, because
    ///   the app would not perform one: every image in the corpus has its
    ///   aspect ratio locked.
    ///
    /// Rotation, the geometry flags and everything else in the archive are left
    /// exactly as they were.
    ///
    /// Refused by name: an object carrying version patches (see
    /// [`Document::patched_objects`]), and resizing something whose current
    /// width or height is zero, where the scale factor is not a number. A
    /// **locked** drawable is not refused — the lock is a rule the app's UI
    /// keeps, not one the format keeps — but it is reported, because the app
    /// will not let a user undo the move by hand.
    pub fn set_geometry(
        &mut self,
        identifier: u64,
        position: Option<(f32, f32)>,
        size: Option<(f32, f32)>,
    ) -> Result<crate::drawable::GeometryChange, Error> {
        use crate::drawable::Frame;

        let drawable = self
            .drawable(identifier)
            .ok_or(Error::NoSuchObject(identifier))?;
        let mask = drawable.mask().and_then(|id| self.drawable(id));
        let before = drawable.frame(mask.as_ref());
        for object in [Some(identifier), mask.as_ref().map(|m| m.identifier)]
            .into_iter()
            .flatten()
        {
            if self.patched_objects().iter().any(|&(id, _)| id == object) {
                return Err(Error::refused(
                    Refusal::Patched,
                    format!(
                        "drawable {identifier}: object {object} carries version patches, and \
                     rewriting it would leave them describing where it used to be"
                    ),
                ));
            }
        }

        let after = Frame {
            x: position.map(|p| p.0).unwrap_or(before.x),
            y: position.map(|p| p.1).unwrap_or(before.y),
            width: size.map(|s| s.0).unwrap_or(before.width),
            height: size.map(|s| s.1).unwrap_or(before.height),
        };
        let resizing = after.width != before.width || after.height != before.height;
        if resizing && (before.width == 0.0 || before.height == 0.0) {
            return Err(Error::Format(format!(
                "drawable {identifier}: it is {} × {} and there is no factor that scales \
                 zero to something else",
                before.width, before.height
            )));
        }
        let (sx, sy) = if resizing {
            (after.width / before.width, after.height / before.height)
        } else {
            (1.0, 1.0)
        };

        // The requested rectangle is the *reported* one, whose origin is the
        // rotated bounding box's corner. Turn it back into the unrotated
        // origin the archive stores, which is the inverse of what
        // `Drawable::frame` does and is the identity at zero degrees.
        let (extent_x, extent_y) =
            crate::drawable::rotated_extent(after.width, after.height, drawable.geometry.angle);
        let base = Frame {
            x: after.x + extent_x / 2.0 - after.width / 2.0,
            y: after.y + extent_y / 2.0 - after.height / 2.0,
            width: after.width,
            height: after.height,
        };

        let mut rewritten = Vec::new();
        let mut geometry = drawable.geometry;
        match &mask {
            Some(mask) => {
                geometry.width *= sx;
                geometry.height *= sy;
                let mut mask_geometry = mask.geometry;
                mask_geometry.x *= sx;
                mask_geometry.y *= sy;
                mask_geometry.width = base.width;
                mask_geometry.height = base.height;
                geometry.x = base.x - mask_geometry.x;
                geometry.y = base.y - mask_geometry.y;
                self.write_geometry(mask.identifier, &mask.path, mask_geometry)?;
                if resizing {
                    self.scale_path_source(mask, base.width, base.height)?;
                }
                rewritten.push(mask.identifier);
            }
            None => {
                geometry.x = base.x;
                geometry.y = base.y;
                geometry.width = base.width;
                geometry.height = base.height;
            }
        }
        self.write_geometry(identifier, &drawable.path, geometry)?;
        if resizing && mask.is_none() && drawable.path_source.is_some() {
            self.scale_path_source(&drawable, base.width, base.height)?;
        }
        rewritten.push(identifier);

        // Media keeps its placed size beside its geometry, and the app moves
        // the two together — but only for an *unmasked* picture. For a masked
        // one `originalSize` is not the picture's own size and the corpus does
        // not agree on what it is: Pages wrote the full picture there for the
        // report photo (511.86 × 466.13, mask 475 × 383) yet the mask window
        // for the book cover (324 × 216, mask 324 × 216), and Keynote wrote the
        // mask window too (160 × 120, image 160 × 160). No single rule fits, so
        // — finding 1 having moved the crop test off this field — the safe
        // choice is to leave a masked image's `originalSize` exactly as it was.
        if resizing && drawable.kind.is_media() && mask.is_none() {
            let mut archive = self.archive_of(identifier)?;
            let field = match drawable.kind {
                crate::drawable::Kind::Image => crate::drawable::image_field::ORIGINAL_SIZE,
                _ => 20,
            };
            let path: Vec<u32> = drawable.path[..drawable.path.len().saturating_sub(1)].to_vec();
            let mut body = if path.is_empty() {
                archive.clone()
            } else {
                match style::get_path(&archive, &path) {
                    Some(Value::Bytes(raw)) => crate::pb::decode_nested(&raw).unwrap_or_default(),
                    _ => Message::default(),
                }
            };
            if body.get(field).is_some() {
                let mut point = body
                    .bytes(field)
                    .and_then(crate::pb::decode_nested)
                    .unwrap_or_default();
                point.set_in_order(1, Value::Fixed32(geometry.width.to_le_bytes()));
                point.set_in_order(2, Value::Fixed32(geometry.height.to_le_bytes()));
                body.set_in_order(field, Value::Bytes(point.encode()));
                if path.is_empty() {
                    archive = body;
                } else {
                    style::set_path(&mut archive, &path, Some(Value::Bytes(body.encode())))
                        .map_err(|e| Error::Format(format!("drawable {identifier}: {e}")))?;
                }
                self.set_archive(identifier, &archive)?;
            }
        }

        Ok(crate::drawable::GeometryChange {
            drawable: identifier,
            before,
            after,
            mask: mask.map(|m| m.identifier),
            rewritten,
        })
    }

    /// Swap the bytes an image is drawn from.
    ///
    /// `target` is an image drawable or a media-registry identifier, as
    /// `iwork drawables` and `iwork media` print them. The bytes are written
    /// into the package, the `DataInfo` is brought back into step — digest,
    /// name, byte length and recorded pixel size — and every drawable showing
    /// that picture has its `naturalSize`, its traced outline and its
    /// "was replaced" flag updated, which is what Keynote did when a script
    /// gave one of its images a new file.
    ///
    /// **What is deliberately not touched: the frame.** The app re-fits a
    /// replaced picture by scaling it to fill the old window and cropping the
    /// overflow with a mask — an 8 × 8 frame given a 32 × 24 picture came back
    /// as a 10.67 × 8 image behind an 8 × 8 mask offset by 1.33. This does the
    /// simpler thing and leaves the geometry alone, so a picture of a different
    /// shape is drawn stretched; [`crate::MediaReplacement::aspect_note`] says
    /// so, and `set_geometry` is how to fix it.
    ///
    /// **What it refuses, and why this method exists at all.** A drawable can
    /// carry state that sits *between* the stored pixels and what is drawn: a
    /// crop, a mask shaped like something other than a rectangle, an Instant
    /// Alpha knockout path, tone and colour adjustments, cached renderings of
    /// the old pixels, a traced outline of them. None of it is in the file
    /// being swapped in, and none of it can be recomputed here. Replacing bytes
    /// under any of it produces a document that opens, reports the same
    /// geometry through AppleScript, passes every structural check — and
    /// renders the wrong thing. So it is refused by name, with
    /// [`Error::NonDestructiveEdit`] listing what was found. An *identity* mask
    /// — the whole picture, no crop, which is what the app installs when it
    /// replaces an image — is not an objection.
    pub fn replace_media(
        &mut self,
        target: u64,
        bytes: &[u8],
        preferred_name: &str,
        pixel_size: Option<(f32, f32)>,
    ) -> Result<crate::media::MediaReplacement, Error> {
        use crate::drawable::image_field;
        use crate::media::{self, field as data_field};

        let drawables = self.drawables();
        let data = match drawables
            .iter()
            .find(|d| d.identifier == target)
            .and_then(|d| d.media.as_ref())
            .and_then(|m| m.data)
        {
            Some(data) => data,
            None => target,
        };
        let file = self
            .data_files()
            .into_iter()
            .find(|d| d.identifier == data)
            .ok_or_else(|| Error::Format(format!("no media registered as {data}")))?;

        // The edit state first, and before anything else that could refuse:
        // it is the answer a caller most needs to hear, and a theme asset that
        // is *also* cropped should say so.
        let users: Vec<&crate::drawable::Drawable> = drawables
            .iter()
            .filter(|d| d.media.as_ref().and_then(|m| m.data) == Some(data))
            .collect();
        for drawable in &users {
            let objections = drawable
                .edit_state
                .as_ref()
                .map(|state| state.objections())
                .unwrap_or_default();
            if !objections.is_empty() {
                return Err(Error::NonDestructiveEdit {
                    drawable: drawable.identifier,
                    reasons: objections,
                });
            }
        }

        let was = file.entry_name().ok_or_else(|| {
            Error::Format(format!(
                "media {data} ({}) lives in the app's theme bundle rather than in the \
                 document, so there are no bytes here to replace",
                file.original_name
            ))
        })?;

        let new_size = match pixel_size.or_else(|| media::pixel_size(bytes)) {
            Some(size) => size,
            None => {
                return Err(Error::Format(
                    "the replacement is not a PNG or a JPEG, so its pixel size cannot be \
                     read here — pass it explicitly"
                        .into(),
                ))
            }
        };
        let digest = media::sha1(bytes);
        let now = format!("Data/{}", media::stored_name(preferred_name, data));

        // The old picture's size, for the stretched-aspect warning. The
        // `DataInfo`'s recorded pixel size is the first choice, but only four
        // of the corpus's 203 stored files carry one; the drawable's
        // `naturalSize` carries it on every one of the 69 images, so it is the
        // fallback that makes the warning fire at all.
        let old_natural = users
            .iter()
            .filter_map(|d| d.media.as_ref().and_then(|m| m.natural_size))
            .next();

        // The registry entry first: if the metadata cannot be rewritten,
        // nothing has been touched.
        let mut old_pixel_size = None;
        let mut pixel_size_recorded = false;
        let metadata = self
            .objects()
            .find(|(_, object)| object.message_type() == crate::TYPE_PACKAGE_METADATA)
            .map(|(_, object)| object.identifier)
            .ok_or_else(|| Error::Format("no TSP.PackageMetadata".into()))?;
        let mut archive = self.archive_of(metadata)?;
        let mut found = false;
        for field in &mut archive.fields {
            if field.number != 4 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let Some(mut info) = crate::pb::decode_nested(raw) else {
                continue;
            };
            if info.varint(data_field::IDENTIFIER) != Some(data) {
                continue;
            }
            old_pixel_size = media::attribute_pixel_size(&info);
            info.set_in_order(data_field::DIGEST, Value::Bytes(digest.to_vec()));
            info.set_in_order(
                data_field::PREFERRED_FILE_NAME,
                Value::Bytes(preferred_name.as_bytes().to_vec()),
            );
            info.set_in_order(
                data_field::FILE_NAME,
                Value::Bytes(
                    now.strip_prefix("Data/")
                        .unwrap_or(&now)
                        .as_bytes()
                        .to_vec(),
                ),
            );
            if info.get(data_field::MATERIALIZED_LENGTH).is_some() {
                info.set_in_order(
                    data_field::MATERIALIZED_LENGTH,
                    Value::Varint(bytes.len() as u64),
                );
            }
            pixel_size_recorded =
                media::set_attribute_pixel_size(&mut info, new_size.0, new_size.1);
            field.value = Value::Bytes(info.encode());
            found = true;
            break;
        }
        if !found {
            return Err(Error::Format(format!(
                "the media registry has no entry {data}"
            )));
        }
        self.set_archive(metadata, &archive)?;

        // Then the bytes, keeping the entry where it was in the package.
        match self
            .package
            .entries
            .iter_mut()
            .find(|(name, _)| *name == was)
        {
            Some(entry) => {
                entry.0 = now.clone();
                entry.1 = bytes.to_vec();
            }
            None => self.package.set(&now, bytes.to_vec()),
        }

        // Then everything that has to agree with the new picture.
        let mut updated = Vec::new();
        for drawable in &users {
            if drawable.kind != crate::drawable::Kind::Image {
                continue;
            }
            let mut archive = self.archive_of(drawable.identifier)?;
            let body_path: Vec<u32> =
                drawable.path[..drawable.path.len().saturating_sub(1)].to_vec();
            let mut body = if body_path.is_empty() {
                archive.clone()
            } else {
                match style::get_path(&archive, &body_path) {
                    Some(Value::Bytes(raw)) => crate::pb::decode_nested(&raw).unwrap_or_default(),
                    _ => continue,
                }
            };
            let mut point = body
                .bytes(image_field::NATURAL_SIZE)
                .and_then(crate::pb::decode_nested)
                .unwrap_or_default();
            point.set_in_order(1, Value::Fixed32(new_size.0.to_le_bytes()));
            point.set_in_order(2, Value::Fixed32(new_size.1.to_le_bytes()));
            body.set_in_order(image_field::NATURAL_SIZE, Value::Bytes(point.encode()));

            // The traced outline is the picture's own rectangle and the app
            // rewrites it with the picture. Anything else was refused above.
            if body.get(image_field::TRACED_PATH).is_some() {
                body.set_in_order(
                    image_field::TRACED_PATH,
                    Value::Bytes(
                        crate::drawable::natural_rectangle(new_size.0, new_size.1).encode(),
                    ),
                );
            }
            let flags = body.varint(image_field::FLAGS).unwrap_or(0)
                | u64::from(crate::drawable::media_flag::WAS_REPLACED);
            body.set_in_order(image_field::FLAGS, Value::Varint(flags));

            if body_path.is_empty() {
                archive = body;
            } else {
                style::set_path(&mut archive, &body_path, Some(Value::Bytes(body.encode())))
                    .map_err(|e| Error::Format(format!("drawable {}: {e}", drawable.identifier)))?;
            }
            self.set_archive(drawable.identifier, &archive)?;
            updated.push(drawable.identifier);
        }

        // Prefer the registry's recorded size, fall back to the drawable's
        // naturalSize, so the comparison is made on all but the images that
        // carry neither.
        let old_size = old_pixel_size.or(old_natural);
        let aspect_changed = old_size.is_some_and(|(w, h)| {
            w > 0.0
                && h > 0.0
                && ((w / h) - (new_size.0 / new_size.1)).abs() > 0.01 * (w / h).max(0.01)
        });
        Ok(crate::media::MediaReplacement {
            data,
            was,
            now,
            digest,
            bytes: bytes.len(),
            old_size,
            new_pixel_size: new_size,
            pixel_size_recorded,
            drawables: updated,
            aspect_changed,
        })
    }

    /// Write a geometry back at the field path the drawable was found through.
    fn write_geometry(
        &mut self,
        identifier: u64,
        path: &[u32],
        geometry: crate::drawable::Geometry,
    ) -> Result<(), Error> {
        let mut archive = self.archive_of(identifier)?;
        let mut full: Vec<u32> = path.to_vec();
        full.push(crate::drawable::field::GEOMETRY);
        let Some(Value::Bytes(raw)) = style::get_path(&archive, &full) else {
            return Err(Error::Format(format!(
                "drawable {identifier}: no geometry at field {}",
                full.iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(".")
            )));
        };
        let mut message = crate::pb::decode_nested(&raw).ok_or_else(|| {
            Error::Format(format!("drawable {identifier}: geometry does not decode"))
        })?;
        geometry.write_into(&mut message);
        style::set_path(&mut archive, &full, Some(Value::Bytes(message.encode())))
            .map_err(|e| Error::Format(format!("drawable {identifier}: {e}")))?;
        self.set_archive(identifier, &archive)
    }

    /// Resize a shape's or a mask's path source with the object.
    ///
    /// **A shape's size lives in two places**, which is the thing this exists
    /// for. Told to make a 200 × 200 Keynote shape 444 × 128, the app rewrote
    /// the geometry *and* the bezier path source — its natural size and every
    /// one of the rectangle's corners — and a document with only the geometry
    /// changed opens with the app still reporting 200 × 200. Nothing about that
    /// is visible from the archive alone; it took writing one and asking.
    ///
    /// **A mask is the exception, and it is the app's exception rather than a
    /// simplification here.** Asked to resize the cropped photo in the Pages
    /// report, Pages changed the mask path's natural size from 475 × 383 to
    /// 300 × 241.89 and left every point of the path exactly where it was —
    /// while Keynote, resizing a shape, moved both. So masks get the natural
    /// size only, shapes get both, and each matches what the app that owns it
    /// wrote. The points are additionally left alone when they are plainly not
    /// drawn in the frame's coordinates, which is the case for a line.
    fn scale_path_source(
        &mut self,
        drawable: &crate::drawable::Drawable,
        width: f32,
        height: f32,
    ) -> Result<(), Error> {
        let identifier = drawable.identifier;
        let mut archive = self.archive_of(identifier)?;
        // The path source hangs off the concrete class — field 3 of a shape,
        // field 2 of a mask — which is one level above the drawable archive.
        let body_path: Vec<u32> = drawable.path[..drawable.path.len().saturating_sub(1)].to_vec();
        let field = match drawable.kind {
            crate::drawable::Kind::Mask => 2u32,
            _ => 3,
        };
        let mut body = if body_path.is_empty() {
            archive.clone()
        } else {
            match style::get_path(&archive, &body_path) {
                Some(Value::Bytes(raw)) => crate::pb::decode_nested(&raw).unwrap_or_default(),
                _ => return Ok(()),
            }
        };
        let Some(mut source) = body.bytes(field).and_then(crate::pb::decode_nested) else {
            return Ok(());
        };

        // The natural size sits at a different field per path-source kind: 3
        // for the point and scalar sources, 2 for the bezier ones, 1 for a
        // callout. Only the arm that is set is touched.
        for (arm, natural) in [(3u32, 3u32), (4, 3), (5, 2), (6, 1), (8, 2)] {
            let Some(mut arm_body) = source.bytes(arm).and_then(crate::pb::decode_nested) else {
                continue;
            };
            let Some(mut point) = arm_body.bytes(natural).and_then(crate::pb::decode_nested) else {
                continue;
            };
            let was = (float_field(&point, 1), float_field(&point, 2));
            point.set_in_order(1, Value::Fixed32(width.to_le_bytes()));
            point.set_in_order(2, Value::Fixed32(height.to_le_bytes()));
            arm_body.set_in_order(natural, Value::Bytes(point.encode()));

            if matches!(arm, 5 | 8) && drawable.kind != crate::drawable::Kind::Mask {
                let factor = |before: f32, after: f32| {
                    if before.abs() < f32::EPSILON {
                        1.0
                    } else {
                        after / before
                    }
                };
                scale_path_points(
                    &mut arm_body,
                    3,
                    was,
                    (factor(was.0, width), factor(was.1, height)),
                );
            }

            source.set_in_order(arm, Value::Bytes(arm_body.encode()));
            body.set_in_order(field, Value::Bytes(source.encode()));
            if body_path.is_empty() {
                archive = body;
            } else {
                style::set_path(&mut archive, &body_path, Some(Value::Bytes(body.encode())))
                    .map_err(|e| Error::Format(format!("drawable {identifier}: {e}")))?;
            }
            return self.set_archive(identifier, &archive);
        }
        Ok(())
    }

    // -- text styles ---------------------------------------------------------

    /// Every character, paragraph and list style in the document.
    ///
    /// These are the objects the attribute tables of a `TSWP.StorageArchive`
    /// point at; see [`crate::style`] for what is and is not known about their
    /// contents.
    pub fn text_styles(&self) -> Vec<TextStyle> {
        let mut out = Vec::new();
        for (stream, object) in self.objects() {
            let Some(message) = object.messages.first() else {
                continue;
            };
            let Some(kind) = StyleKind::from_message_type(message.message_type) else {
                continue;
            };
            let Ok(archive) = Message::decode(&message.payload) else {
                continue;
            };
            out.push(TextStyle {
                identifier: object.identifier,
                stream: stream.to_string(),
                kind,
                name: style::string_at(&archive, style::NAME),
                style_identifier: style::string_at(&archive, style::STYLE_IDENTIFIER),
                parent: style::reference_at(&archive, style::PARENT),
                stylesheet: style::reference_at(&archive, style::STYLESHEET),
                labels: style::labels(&archive),
                archive,
            });
        }
        out
    }

    pub fn text_style(&self, identifier: u64) -> Option<TextStyle> {
        self.text_styles()
            .into_iter()
            .find(|s| s.identifier == identifier)
    }

    /// Every run of text that points at a style, across the whole document.
    ///
    /// This reads all six attribute tables rather than only the one belonging
    /// to the style's kind, so a style used somewhere unexpected still shows up.
    pub fn text_style_usage(&self, identifier: u64) -> Vec<StyleUse> {
        let mut out = Vec::new();
        for (stream, object) in self.objects() {
            let Some(message) = object.messages.first() else {
                continue;
            };
            if message.message_type != crate::TYPE_STORAGE {
                continue;
            }
            let Ok(storage) = Message::decode(&message.payload) else {
                continue;
            };
            let length = text::length(&text::read(&storage));
            for &table in text::ATTRIBUTE_TABLES {
                let Some(table_message) = storage.bytes(table).and_then(crate::pb::decode_nested)
                else {
                    continue;
                };
                let runs = style::runs(&table_message);
                for (i, run) in runs.iter().enumerate() {
                    if run.style != Some(identifier) {
                        continue;
                    }
                    let end = runs.get(i + 1).map(|next| next.start).unwrap_or(length);
                    out.push(StyleUse {
                        storage: object.identifier,
                        stream: stream.to_string(),
                        table,
                        range: run.start..end.max(run.start),
                    });
                }
            }
        }
        out
    }

    /// Copy a style, giving the copy a new name and a new object identifier.
    ///
    /// Styles are created by copying rather than by synthesis, for the reason
    /// `FORMAT.md` gives for whole documents: the style graph is large, iWork is
    /// unforgiving about dangling references, and a style that already works is
    /// a far better starting point than a guess at the schema. The copy lands in
    /// the template's stream, right after it, and is listed in every stylesheet
    /// that listed the template by plain reference.
    ///
    /// The identifier comes from above `TSP.PackageMetadata` field 1, which is
    /// then bumped, so iWork will not later hand the same number to something
    /// else.
    ///
    /// **A copy of a variation style does not get the name.** Named styles and
    /// variations are different things: a named style carries a name at
    /// [`style::NAME`] and an identifier at [`style::STYLE_IDENTIFIER`], while a
    /// variation carries neither, a parent, and a flag saying it is one. Naming
    /// the copy of a variation produces an object that claims to be a variation,
    /// has a name, has no identifier, and is listed among the named styles —
    /// and Pages crashes on opening the document. It was doing exactly that
    /// until a document that crashed showed up.
    ///
    /// The copy is still made, still listed, and still usable — it is simply
    /// anonymous, which is what a variation is. `CreatedStyle::name` reports
    /// whether the name was applied. Turning a variation into a named style
    /// properly would mean synthesising a style identifier and clearing the
    /// variation flag, which is more invention than this crate is willing to do
    /// without a document to check it against.
    pub fn create_text_style(&mut self, template: u64, name: &str) -> Result<CreatedStyle, Error> {
        let source = self
            .text_style(template)
            .ok_or(Error::NoSuchStyle(template))?;
        let (stream, index) = self.locate(template).expect("the style was just found");
        let identifier = self.next_object_identifier();

        let mut archive = source.archive.clone();
        let applied_name = source.name.is_some().then(|| name.to_string());
        if applied_name.is_some() {
            style::set_path(
                &mut archive,
                style::NAME,
                Some(Value::Bytes(name.as_bytes().to_vec())),
            )
            .map_err(|e| Error::Format(format!("style {template}: {e}")))?;
        }

        // Bump the high-water mark first: if there is no package metadata to
        // bump, no object is added at all.
        self.set_last_object_identifier(identifier)?;

        let mut object = self.streams[&stream][index].clone();
        object.identifier = identifier;
        object.messages[0].payload = archive.encode();
        self.streams
            .get_mut(&stream)
            .expect("stream came from the document")
            .insert(index + 1, object);

        // List the copy wherever the template is listed — in the template's own
        // stylesheet, which the template names. See `style::clone_registrations`
        // for why anywhere-a-reference-looks-right is not good enough.
        let mut registrations_cloned = 0;
        if let Some(stylesheet) = source.stylesheet {
            if let Ok(mut sheet) = self.archive_of(stylesheet) {
                registrations_cloned = style::clone_registrations(&mut sheet, template, identifier);
                // A style is listed twice: once among the stylesheet's styles,
                // once under its parent. Both, or the copy is an orphan.
                if let Some(parent) = source.parent {
                    if style::clone_sibling(&mut sheet, parent, template, identifier) {
                        registrations_cloned += 1;
                    }
                }
                if registrations_cloned > 0 {
                    self.set_archive(stylesheet, &sheet)?;
                }
            }
        }
        self.declare_external_references();

        Ok(CreatedStyle {
            identifier,
            template,
            stream,
            registrations_cloned,
            name: applied_name,
        })
    }

    /// Rewrite a style's name.
    ///
    /// Naming an unnamed variation style is allowed and gives it a name; the
    /// field is created if it is not there.
    pub fn rename_text_style(&mut self, identifier: u64, name: &str) -> Result<(), Error> {
        self.set_text_style_property(
            identifier,
            style::NAME,
            Some(Value::Bytes(name.as_bytes().to_vec())),
        )
    }

    /// Edit a style archive directly.
    ///
    /// The escape hatch for everything this crate does not model: the archive
    /// arrives decoded into wire fields and is re-encoded in place afterwards.
    pub fn update_text_style(
        &mut self,
        identifier: u64,
        edit: impl FnOnce(&mut Message),
    ) -> Result<(), Error> {
        let style = self
            .text_style(identifier)
            .ok_or(Error::NoSuchStyle(identifier))?;
        let mut archive = style.archive;
        edit(&mut archive);
        self.set_archive(identifier, &archive)
    }

    /// Set one field of a style, addressed by a path of field numbers, or
    /// remove it when `value` is `None`.
    ///
    /// `iwork style <file> <id>` prints those paths. Nothing here knows what a
    /// given field *means* — see [`crate::style`].
    pub fn set_text_style_property(
        &mut self,
        identifier: u64,
        path: &[u32],
        value: Option<Value>,
    ) -> Result<(), Error> {
        let style = self
            .text_style(identifier)
            .ok_or(Error::NoSuchStyle(identifier))?;
        let mut archive = style.archive;
        style::set_path(&mut archive, path, value)
            .map_err(|e| Error::Format(format!("style {identifier}: {e}")))?;
        self.set_archive(identifier, &archive)
    }

    /// Set the colour of a style's text, everywhere the style keeps it.
    ///
    /// A style does not have one text colour. It has up to four — the font
    /// colour, the fill drawn inside the glyphs, and the underline and
    /// strikethrough colours that follow the text — and Pages writes all of
    /// them together. Setting only the font colour leaves the fill behind, and
    /// the fill is what is drawn: a title whose `11.7` says red and whose
    /// `11.46.1` still says black renders black. That was not a guess; it took
    /// a document that came back with a bigger title and no red.
    ///
    /// Returns how many were set. **Zero means the style keeps no colour at
    /// all** — nothing was written, because a colour this crate invents is
    /// missing fields that make Pages refuse the document. Copy the colour from
    /// a style that has one, with [`Document::copy_text_style_property`].
    ///
    /// Channels are `0.0..=1.0`, as the format stores them.
    pub fn set_text_style_color(
        &mut self,
        identifier: u64,
        red: f32,
        green: f32,
        blue: f32,
        alpha: f32,
    ) -> Result<usize, Error> {
        let style = self
            .text_style(identifier)
            .ok_or(Error::NoSuchStyle(identifier))?;
        let mut archive = style.archive;
        let mut set = 0;
        for path in style::property::TEXT_COLOR_PATHS {
            let Some(Value::Bytes(raw)) = style::get_path(&archive, path) else {
                continue;
            };
            let Some(mut colour) = crate::pb::decode_nested(&raw) else {
                continue;
            };
            if !style::is_color(&colour) {
                continue;
            }
            style::set_channels(&mut colour, red, green, blue, alpha);
            style::set_path(&mut archive, path, Some(Value::Bytes(colour.encode())))
                .map_err(|e| Error::Format(format!("style {identifier}: {e}")))?;
            set += 1;
        }
        if set > 0 {
            self.set_archive(identifier, &archive)?;
        }
        Ok(set)
    }

    /// Copy one property subtree from another style.
    ///
    /// The way to give a style a property whose container it does not have.
    /// [`Document::set_text_style_property`] will not invent a container,
    /// because a container it invents holds only the fields it was asked for —
    /// a colour written as `{r, g, b}`, with no model and no alpha, crashes
    /// Pages on opening. Lifting a whole working subtree across avoids the
    /// question: the copy is a colour that a real document already contains.
    ///
    /// Take the colour from a style that has one, then change the channels:
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let mut doc = iwork::Document::open("Report.pages")?;
    /// use iwork::style::property;
    /// doc.copy_text_style_property(3712, 3801, property::FONT_COLOR)?;
    /// doc.set_text_style_property(3801, property::RED,
    ///     Some(iwork::pb::Value::Fixed32(0.85f32.to_le_bytes())))?;
    /// # Ok(()) }
    /// ```
    pub fn copy_text_style_property(
        &mut self,
        from: u64,
        to: u64,
        path: &[u32],
    ) -> Result<(), Error> {
        let source = self.text_style(from).ok_or(Error::NoSuchStyle(from))?;
        let value = style::get_path(&source.archive, path).ok_or_else(|| {
            Error::Format(format!(
                "style {from} has no field {} to copy",
                dotted(path)
            ))
        })?;
        let target = self.text_style(to).ok_or(Error::NoSuchStyle(to))?;
        let mut archive = target.archive;
        // The parent of the leaf has to exist in the target; copy it whole if
        // the target lacks it, which is the case this method is for.
        if let Some((leaf, parents)) = path.split_last() {
            if !parents.is_empty() && style::get_path(&archive, parents).is_none() {
                let container = style::get_path(&source.archive, parents).ok_or_else(|| {
                    Error::Format(format!("style {from} has no field {}", dotted(parents)))
                })?;
                style::set_path(&mut archive, parents, Some(container))
                    .map_err(|e| Error::Format(format!("style {to}: {e}")))?;
            }
            let _ = leaf;
        }
        style::set_path(&mut archive, path, Some(value))
            .map_err(|e| Error::Format(format!("style {to}: {e}")))?;
        self.set_archive(to, &archive)
    }

    /// Remove a style, and every reference to it this crate can account for.
    ///
    /// Runs that use it are pointed at `replace_with`, or dropped when that is
    /// `None` — a dropped run lets the preceding one extend over its text.
    /// References from stylesheets are removed.
    ///
    /// If anything *else* in the document still refers to the style — another
    /// style naming it as a parent, a keyed stylesheet entry, an archive this
    /// crate does not model — the delete is refused with [`Error::StyleInUse`]
    /// and the document is left untouched. iWork is unforgiving about dangling
    /// references, so a delete that cannot be completed cleanly is not
    /// completed at all.
    pub fn delete_text_style(
        &mut self,
        identifier: u64,
        replace_with: Option<u64>,
    ) -> Result<StyleDeletion, Error> {
        let style = self
            .text_style(identifier)
            .ok_or(Error::NoSuchStyle(identifier))?;
        if let Some(replacement) = replace_with {
            let other = self
                .text_style(replacement)
                .ok_or(Error::NoSuchStyle(replacement))?;
            if other.kind != style.kind {
                return Err(Error::Format(format!(
                    "cannot replace a {} style with a {} style",
                    style.kind.as_str(),
                    other.kind.as_str()
                )));
            }
        }

        let mut deletion = StyleDeletion {
            identifier,
            runs_repointed: 0,
            runs_dropped: 0,
            registrations_removed: 0,
        };
        let mut edits: Vec<(u64, Message)> = Vec::new();
        let mut still_referenced: Vec<u64> = Vec::new();

        // Work out the whole edit before touching anything, so a refusal leaves
        // the document exactly as it was.
        for (_, object) in self.objects() {
            if object.identifier == identifier {
                continue;
            }
            let Some(message) = object.messages.first() else {
                continue;
            };
            let Ok(archive) = Message::decode(&message.payload) else {
                continue;
            };
            if style::count_references(&archive, identifier) == 0 {
                continue;
            }

            let mut edited = archive.clone();
            let mut touched = 0;
            if message.message_type == crate::TYPE_STORAGE {
                for &table in text::ATTRIBUTE_TABLES {
                    let Some(mut runs) = edited.bytes(table).and_then(crate::pb::decode_nested)
                    else {
                        continue;
                    };
                    let changed = style::repoint(&mut runs, identifier, replace_with);
                    if changed > 0 {
                        edited.set(table, Value::Bytes(runs.encode()));
                        touched += changed;
                    }
                }
                match replace_with {
                    Some(_) => deletion.runs_repointed += touched,
                    None => deletion.runs_dropped += touched,
                }
            }
            // Unlist it, but only from the stylesheet it says it belongs to.
            // Other objects hold bare references that are positions rather than
            // memberships — a Keynote slide's outline levels, say — and pulling
            // one out of those would shift the rest. Anything still holding a
            // reference after this is reported below and the delete is refused,
            // which is the right outcome for a style that is genuinely in use.
            if Some(object.identifier) == style.stylesheet {
                let removed = style::remove_registrations(&mut edited, identifier)
                    + style::remove_sibling(&mut edited, identifier);
                deletion.registrations_removed += removed;
                touched += removed;
            }

            if style::count_references(&edited, identifier) > 0 {
                still_referenced.push(object.identifier);
            } else if touched > 0 {
                edits.push((object.identifier, edited));
            }
        }

        if !still_referenced.is_empty() {
            return Err(Error::StyleInUse {
                identifier,
                references: still_referenced,
            });
        }

        for (target, archive) in edits {
            self.set_archive(target, &archive)?;
        }
        let (stream, index) = self.locate(identifier).expect("the style was just found");
        self.streams
            .get_mut(&stream)
            .expect("stream came from the document")
            .remove(index);
        // `TSP.PackageMetadata` field 1 is a high-water mark, not a count, so it
        // stays where it is — lowering it would let iWork reuse the identifier.
        Ok(deletion)
    }

    /// Point a range of one storage's text at a style.
    ///
    /// The range is in UTF-16 code units, the unit run indices are counted in.
    /// Which attribute table is edited follows from the style's kind. Paragraph
    /// and list styles apply to whole paragraphs, so give them ranges from
    /// [`Document::paragraph_ranges`] rather than arbitrary offsets.
    ///
    /// A range reaching past the end of the text is [`Error::TextRange`], as it
    /// is for [`Document::replace_text`]. Clamping it instead is how
    /// `500..600` came to restyle the only paragraph of a 54-unit body and
    /// report success.
    pub fn apply_text_style(
        &mut self,
        storage: u64,
        range: Range<u64>,
        style_identifier: u64,
    ) -> Result<(), Error> {
        let kind = self
            .text_style(style_identifier)
            .ok_or(Error::NoSuchStyle(style_identifier))?
            .kind;
        let mut archive = self.storage_archive(storage)?;
        let body = text::read(&archive);
        let length = text::length(&body);
        let table_field = kind.attribute_table();

        let end = range.end.max(range.start);
        if end > length {
            return Err(Error::TextRange {
                storage,
                index: end,
                length,
            });
        }

        // A paragraph style applies to whole paragraphs, so the range grows to
        // the paragraphs it touches. Writing a paragraph run inside a paragraph
        // is not a smaller version of the same edit — it is a table iWork never
        // writes, and Keynote's own parser is documented as rendering a text box
        // 2^16 points tall and then crashing on one.
        let range = match kind {
            StyleKind::Paragraph | StyleKind::List => {
                let paragraphs = text::paragraph_ranges(&body);
                let start = paragraphs
                    .iter()
                    .rev()
                    .find(|p| p.start <= range.start)
                    .map(|p| p.start)
                    .unwrap_or(range.start);
                let end = paragraphs
                    .iter()
                    .find(|p| p.end >= range.end)
                    .map(|p| p.end)
                    .unwrap_or(range.end);
                start..end.max(start)
            }
            StyleKind::Character => range,
        };

        let mut table = match archive.bytes(table_field) {
            Some(raw) => crate::pb::decode_nested(raw).ok_or_else(|| {
                Error::Format(format!(
                    "storage {storage}: field {table_field} is not an attribute table"
                ))
            })?,
            None => Message::default(),
        };
        style::apply(&mut table, range, style_identifier, length);
        // In field order: a storage that had no table of this kind would
        // otherwise get one appended after every other field, and iWork writes
        // its fields ascending everywhere anyone has looked. Appending gave a
        // storage the field order 1, …, 28, 8.
        archive.set_in_order(table_field, Value::Bytes(table.encode()));
        self.set_archive(storage, &archive)?;
        // The style usually lives in the document stylesheet and the storage in
        // the document body — different components, so the reference has to be
        // declared before iWork will resolve it.
        self.declare_external_references();
        Ok(())
    }

    /// Character ranges of the paragraphs in one storage, in UTF-16 code units.
    pub fn paragraph_ranges(&self, storage: u64) -> Result<Vec<Range<u64>>, Error> {
        Ok(text::paragraph_ranges(&self.storage_text(storage)?))
    }

    /// Text of one storage by identifier, placeholders and all.
    ///
    /// [`Document::text_storages`] hides storages whose contents are only an
    /// anchor for something else; this does not, because a caller addressing a
    /// storage by identifier has already chosen it.
    pub fn storage_text(&self, storage: u64) -> Result<String, Error> {
        Ok(text::read(&self.storage_archive(storage)?))
    }

    /// The Pages document spine: mode, paper, sections, headers and footers,
    /// page templates, threads, contents lists, footnotes and bookmarks.
    ///
    /// `None` for a document with no `TP.DocumentArchive`, which is every
    /// Numbers and Keynote document.
    pub fn structure(&self) -> Option<crate::pages::Structure> {
        crate::pages::structure(self)
    }

    /// Sections of a Pages document, with the range of body text each covers.
    pub fn sections(&self) -> Vec<crate::pages::Section> {
        self.structure().map(|s| s.sections).unwrap_or_default()
    }

    /// Every header and footer storage, named by section, template page and
    /// zone.
    ///
    /// The storage identifier each one carries is what [`Document::set_text`]
    /// takes: a header is a `TSWP.StorageArchive` like any other, and editing
    /// one goes through the same remapping as editing the body.
    pub fn header_footers(&self) -> Vec<crate::pages::HeaderFooter> {
        self.structure()
            .map(|s| s.header_footers)
            .unwrap_or_default()
    }

    /// Column layouts in force over one storage's paragraphs.
    pub fn column_layouts(&self, storage: u64) -> Vec<crate::pages::ColumnLayout> {
        crate::pages::column_layouts(self, storage)
    }

    // -- the Keynote show ----------------------------------------------------

    /// The deck: theme, slide size, every slide with its layout, placeholders,
    /// presenter notes, transition and build counts, and every slide layout the
    /// theme carries.
    ///
    /// `None` for a document with no `KN.ShowArchive`, which is every Pages and
    /// Numbers document.
    pub fn show(&self) -> Option<crate::keynote::Show> {
        crate::keynote::show(self)
    }

    /// The slides of a Keynote document, in deck order.
    pub fn slides(&self) -> Vec<crate::keynote::Slide> {
        self.show().map(|s| s.slides).unwrap_or_default()
    }

    /// The slide layouts a Keynote document's theme carries, in the order the
    /// app lists them.
    pub fn slide_layouts(&self) -> Vec<crate::keynote::Layout> {
        self.show().map(|s| s.layouts).unwrap_or_default()
    }

    /// Skip a slide, or stop skipping it. Returns whether the flag changed.
    ///
    /// The slide may be named by its `KN.SlideArchive` or by its
    /// `KN.SlideNodeArchive`; both are printed by `iwork slides`.
    pub fn set_slide_skipped(&mut self, slide: u64, skipped: bool) -> Result<bool, Error> {
        crate::keynote::set_slide_skipped(self, slide, skipped)
    }

    /// Move a slide to another position in the deck, counting from 0. Returns
    /// where it landed.
    pub fn move_slide(&mut self, slide: u64, to: usize) -> Result<usize, Error> {
        crate::keynote::move_slide(self, slide, to)
    }

    /// Copy a slide, and put the copy straight after it.
    ///
    /// See [`crate::keynote::duplicate_slide`] for what a copy has to touch and
    /// how that was measured.
    pub fn duplicate_slide(&mut self, slide: u64) -> Result<crate::keynote::SlideCopy, Error> {
        crate::keynote::duplicate_slide(self, slide)
    }

    /// Add a table to a sheet a Numbers document already has.
    ///
    /// The sheet is named by its object identifier or by its name. The table
    /// borrows the styles a table already in the document uses, so it looks
    /// like its neighbours rather than like something invented here — and a
    /// document with no table at all is refused for exactly that reason.
    ///
    /// Every cell of it can be written from the start.
    pub fn add_table(
        &mut self,
        sheet: &str,
        name: &str,
        rows: usize,
        columns: usize,
    ) -> Result<u64, Error> {
        self.add_table_at(sheet, name, rows, columns, (0.0, 0.0))
    }

    /// The same, anywhere a drawable can go — a Numbers sheet, a Keynote
    /// slide or a Pages page (`page 2`) — at a position in points.
    ///
    /// A table is a drawable like any other, so the containment rules are the
    /// ones in [`Document::add_text_box`]: a sheet and a slide own theirs and
    /// are named as the parent, a page owns nothing.
    pub fn add_table_at(
        &mut self,
        container: &str,
        name: &str,
        rows: usize,
        columns: usize,
        position: (f32, f32),
    ) -> Result<u64, Error> {
        check_table_size(rows, columns)?;
        crate::table::add_table(self, container, name, rows, columns, position)
    }

    /// Add a sheet, with one table on it, to a Numbers document.
    ///
    /// The new sheet is modelled on one the document already has — its style,
    /// its margins, its zoom — with its own name and nothing on it but the
    /// table.
    pub fn add_sheet(
        &mut self,
        name: &str,
        table: &str,
        rows: usize,
        columns: usize,
    ) -> Result<u64, Error> {
        check_table_size(rows, columns)?;
        crate::table::add_sheet(self, name, table, rows, columns)
    }

    /// Put a text box on a Keynote slide, a Numbers sheet or a Pages page.
    ///
    /// The container is named by identifier, a sheet by its name, or a page as
    /// `page 2`, counting from one; the position and size are in points, as
    /// everything geometric here is. What the box is drawn with comes from the
    /// document — the style of a text shape already on it, or the theme's
    /// text-box preset — so a box added to a deck made from a theme looks like
    /// the theme.
    ///
    /// Returns the drawable, which [`Document::set_geometry`] can then move.
    pub fn add_text_box(
        &mut self,
        container: &str,
        text: &str,
        position: (f32, f32),
        size: (f32, f32),
    ) -> Result<u64, Error> {
        crate::drawable::add_text_box(self, container, text, position, size)
    }

    /// Put an image on a Keynote slide, a Numbers sheet or a Pages page.
    ///
    /// The bytes are copied into the package and registered; `size` is the
    /// rectangle in points, and `None` draws the picture at its own pixel
    /// size. PNG and JPEG only — see [`crate::drawable::add_image`].
    pub fn add_image(
        &mut self,
        container: &str,
        bytes: &[u8],
        preferred_name: &str,
        position: (f32, f32),
        size: Option<(f32, f32)>,
    ) -> Result<u64, Error> {
        crate::drawable::add_image(self, container, bytes, preferred_name, position, size)
    }

    /// Add a file to the package and a `TSP.DataInfo` naming it to the media
    /// registry.
    ///
    /// The registry is refcounted (§7), so this is the allocating half of what
    /// [`Document::replace_media`] rewrites in place: a fresh identifier, the
    /// raw SHA-1 of the bytes, the two names, the byte length, and the pixel
    /// size in the image attributes the app reads it from.
    pub(crate) fn register_media(
        &mut self,
        data: u64,
        bytes: &[u8],
        preferred_name: &str,
        pixel_size: (f32, f32),
    ) -> Result<String, Error> {
        use crate::media::field as data_field;

        let metadata = self
            .objects()
            .find(|(_, object)| object.message_type() == crate::TYPE_PACKAGE_METADATA)
            .map(|(_, object)| object.identifier)
            .ok_or_else(|| Error::Format("no TSP.PackageMetadata".into()))?;
        let stored = crate::media::stored_name(preferred_name, data);
        let digest = crate::media::sha1(bytes);

        let mut info = Message::default();
        info.set_in_order(data_field::IDENTIFIER, Value::Varint(data));
        info.set_in_order(data_field::DIGEST, Value::Bytes(digest.to_vec()));
        info.set_in_order(
            data_field::PREFERRED_FILE_NAME,
            Value::Bytes(preferred_name.as_bytes().to_vec()),
        );
        info.set_in_order(
            data_field::FILE_NAME,
            Value::Bytes(stored.as_bytes().to_vec()),
        );
        // `TSP.DataAttributes { 100: TSD.ImageDataAttributes { 1: size } }` —
        // the extension is how a picture's pixels get recorded at all, and
        // `replace_media` will not update a size the entry does not carry.
        let size = crate::pb::Message {
            fields: vec![crate::pb::Field {
                number: 1,
                value: Value::Bytes(
                    crate::pb::Message {
                        fields: vec![
                            crate::pb::Field {
                                number: 1,
                                value: Value::Fixed32(pixel_size.0.to_le_bytes()),
                            },
                            crate::pb::Field {
                                number: 2,
                                value: Value::Fixed32(pixel_size.1.to_le_bytes()),
                            },
                        ],
                    }
                    .encode(),
                ),
            }],
        };
        info.set_in_order(
            data_field::ATTRIBUTES,
            Value::Bytes(
                crate::pb::Message {
                    fields: vec![crate::pb::Field {
                        number: data_field::IMAGE_ATTRIBUTES,
                        value: Value::Bytes(size.encode()),
                    }],
                }
                .encode(),
            ),
        );
        info.set_in_order(
            data_field::MATERIALIZED_LENGTH,
            Value::Varint(bytes.len() as u64),
        );

        let mut archive = self.archive_of(metadata)?;
        archive.append_in_order(4, Value::Bytes(info.encode()));
        self.set_archive(metadata, &archive)?;
        self.package.set(&format!("Data/{stored}"), bytes.to_vec());
        Ok(stored)
    }

    /// Put a shape on a Keynote slide, a Numbers sheet or a Pages page.
    ///
    /// The same archive as [`Document::add_text_box`] with a different path,
    /// and the text may be empty. See [`crate::drawable::Outline`] for the
    /// three outlines this crate can draw.
    pub fn add_shape(
        &mut self,
        container: &str,
        outline: crate::drawable::Outline,
        text: &str,
        position: (f32, f32),
        size: (f32, f32),
    ) -> Result<u64, Error> {
        crate::drawable::add_shape(self, container, outline, text, position, size)
    }

    /// Attach a comment to a range of text.
    ///
    /// The range is in UTF-16 code units and half-open. See
    /// [`crate::annotations::add_comment`] for what it writes and what it
    /// refuses.
    pub fn add_comment(
        &mut self,
        storage: u64,
        start: u64,
        end: u64,
        edit: &crate::annotations::CommentEdit,
    ) -> Result<u64, Error> {
        crate::annotations::add_comment(self, storage, start, end, edit)
    }

    /// Animate a drawable on or off a slide.
    ///
    /// See [`crate::keynote::add_build`]: the effect is a *build* identifier,
    /// not a transition one, and the drawable has to be on the slide.
    pub fn add_build(
        &mut self,
        slide: u64,
        drawable: u64,
        edit: &crate::keynote::BuildEdit,
    ) -> Result<u64, Error> {
        crate::keynote::add_build(self, slide, drawable, edit)
    }

    /// Give a slide a transition, or take its transition away.
    ///
    /// See [`crate::keynote::set_transition`]: the effect may be named the way
    /// `iwork slides` prints it or by its archive identifier, and `"none"`
    /// removes it.
    pub fn set_transition(
        &mut self,
        slide: u64,
        edit: &crate::keynote::TransitionEdit,
    ) -> Result<crate::keynote::Transition, Error> {
        crate::keynote::set_transition(self, slide, edit)
    }

    /// Add a slide to the end of a Keynote deck.
    ///
    /// Drawn from one of the deck's own layouts — `None` takes the first, which
    /// is what Keynote's New Slide does — so the slide looks like the theme
    /// rather than like something this crate invented. It is empty: the
    /// layout's placeholders are what a new slide shows, and putting text in
    /// one is a separate job.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// let mut deck = iwork::Document::open("Deck.key")?;
    /// let slide = deck.add_slide(None)?;
    /// println!("slide {} is number {:?}", slide.identifier, slide.number);
    /// deck.save("Deck-with-one-more.key")?;
    /// # Ok(()) }
    /// ```
    pub fn add_slide(&mut self, layout: Option<u64>) -> Result<crate::keynote::Slide, Error> {
        crate::keynote::add_slide(self, layout)
    }

    /// Replace a slide's presenter notes.
    ///
    /// Notes are an ordinary `TSWP.StorageArchive` of kind 4, so the text edit
    /// is [`Document::set_text`] on the storage `iwork slides` prints — "which
    /// storage" being the only hard part.
    ///
    /// It also keeps the node's `hasNote` (field 8) in step with the text.
    /// Across all 83 slide nodes in the corpus `hasNote == 1` exactly when the
    /// note storage is non-empty — an app invariant, and one [`Document::problems`]
    /// now enforces — so writing text into an empty-note slide must set the
    /// flag, and clearing it must unset it, or the flag comes to say the
    /// opposite of what the storage holds.
    pub fn set_presenter_notes(&mut self, slide: u64, text: &str) -> Result<TextEdit, Error> {
        let show = self
            .show()
            .ok_or_else(|| Error::refused(Refusal::NotFound, "not a Keynote document"))?;
        let found = show.slide(slide).ok_or(Error::NoSuchObject(slide))?;
        let node = found.node;
        // A slide with no notes is given some: two objects, which is all a note
        // is — see [`crate::keynote::add_note`]. A deck made from nothing has
        // none of them, and refusing here made such a deck unable to carry a
        // word of speaker notes.
        let storage = match found.note_storage {
            Some(storage) => storage,
            None => crate::keynote::add_note(self, slide)?,
        };

        let edit = self.set_text(storage, text)?;

        // `hasNote` follows the text. Only rewrite the node when the flag
        // actually changes, so replacing one non-empty note with another still
        // touches the slide stream alone.
        let has_note = !text.is_empty();
        let mut archive = self.archive_of(node)?;
        let was = archive
            .varint(crate::keynote::node_field::HAS_NOTE)
            .is_some_and(|v| v != 0);
        if was != has_note {
            archive.set_in_order(
                crate::keynote::node_field::HAS_NOTE,
                Value::Varint(u64::from(has_note)),
            );
            self.set_archive_of(node, &archive)?;
        }
        Ok(edit)
    }

    /// Comments, their authors, and tracked changes.
    ///
    /// Present in every document — the author storage always is — and empty in
    /// every document this crate has been shown. See [`crate::annotations`] for
    /// what that means and what is therefore unverified.
    pub fn annotations(&self) -> crate::annotations::Annotations {
        crate::annotations::annotations(self)
    }

    /// Everything about the object graph that looks wrong.
    ///
    /// Written to be run against an *unedited* document first: whatever it says
    /// about a file iWork itself produced is a bug in this function, not in the
    /// file. Its value is the difference — a check that passes on the original
    /// and fails after an edit has found something the edit broke.
    ///
    /// It cannot say a document will open. It can say a document is missing an
    /// object something points at, which is the failure `FORMAT.md` warns about
    /// and the one this crate is most likely to cause.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let mut seen: BTreeMap<u64, usize> = BTreeMap::new();
        for (_, object) in self.objects() {
            *seen.entry(object.identifier).or_default() += 1;
        }
        for (identifier, count) in seen.iter().filter(|(_, n)| **n > 1) {
            problems.push(format!("object {identifier} is defined {count} times"));
        }

        // An entry name is a path this crate will one day join to a directory —
        // saving in the package form does exactly that — so a name that is not
        // a plain relative path is a document trying to write somewhere it does
        // not belong. All 928 packages examined here have `Index/…`,
        // `Metadata/…`, `Data/…` and the four root files, and nothing else.
        for name in self.package.names() {
            let plain = !name.is_empty()
                && !name.starts_with('/')
                && std::path::Path::new(name)
                    .components()
                    .all(|c| matches!(c, std::path::Component::Normal(_)));
            if !plain {
                problems.push(format!(
                    "package entry {name:?} is not a plain relative path, and would escape the \
                     document if it were written as a package"
                ));
            }
        }

        // **The media registry is keyed by content.** iWork stores one file
        // per distinct set of bytes and points every user at it; a package
        // carrying the same content twice is one the app aborts on rather
        // than opens — Keynote dies inside `TSPersistence`, before it draws,
        // with nothing said to the user. Found by a deck this crate wrote,
        // which `iwork check` had called clean.
        let mut by_digest: BTreeMap<Vec<u8>, Vec<String>> = BTreeMap::new();
        for file in self.data_files() {
            if file.digest.is_empty() {
                continue;
            }
            by_digest
                .entry(file.digest.clone())
                .or_default()
                .push(file.stored_name.clone());
        }
        for (digest, names) in by_digest.iter().filter(|(_, names)| names.len() > 1) {
            let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
            problems.push(format!(
                "{} media files hold the same bytes (sha1 {hex}): {} — the registry is keyed by \
                 content, and a duplicate is what the app aborts on",
                names.len(),
                names.join(", ")
            ));
        }

        let highest = seen.keys().copied().max().unwrap_or(0);
        match self.last_object_identifier() {
            Some(mark) if mark < highest => problems.push(format!(
                "package metadata says the highest identifier is {mark}, but {highest} exists"
            )),
            None => problems.push("no TSP.PackageMetadata".to_string()),
            _ => {}
        }

        // A stream this session added — a copied Keynote slide is one — is in
        // `streams` and not yet in the package, and is not missing. Asking the
        // package alone reported every duplicate as broken until it was saved.
        for component in self.components() {
            let name = component.stream_name();
            if !self.package.contains(&name) && !self.streams.contains_key(&name) {
                problems.push(format!(
                    "component {} points at missing {name}",
                    component.identifier
                ));
            }
        }

        let styles: BTreeMap<u64, StyleKind> = self
            .text_styles()
            .into_iter()
            .map(|s| (s.identifier, s.kind))
            .collect();

        for (stream, object) in self.objects() {
            let Some(message) = object.messages.first() else {
                continue;
            };
            if message.message_type != crate::TYPE_STORAGE {
                continue;
            }
            let Ok(storage) = Message::decode(&message.payload) else {
                problems.push(format!("storage {} does not decode", object.identifier));
                continue;
            };
            let body = text::read(&storage);
            let length = text::length(&body);
            let starts: Vec<u64> = text::paragraph_ranges(&body)
                .into_iter()
                .map(|r| r.start)
                .collect();

            // A field this crate cannot place is not a fault in the document —
            // it is a gap in the crate, and one that makes an edit unsafe. Say
            // so, because `check` is where a caller finds out before writing.
            if let Some(field) = text::unknown_table(&storage) {
                problems.push(format!(
                    "{stream} storage {}: field {field} is not an attribute table \
                     this crate knows, so an edit to this storage is refused",
                    object.identifier
                ));
            }
            // A table this crate knows by number and cannot decode is worse: an
            // edit would remap every other table in the storage and step over
            // this one, leaving it anchored to characters that have moved. The
            // remapper refuses; `check` is where a caller finds out first.
            if let Some(field) = text::undecodable_table(&storage) {
                problems.push(format!(
                    "{stream} storage {}: field {field} ({}) does not decode as an \
                     attribute table, so an edit to this storage is refused",
                    object.identifier,
                    text::table(field).map(|t| t.name).unwrap_or("unknown")
                ));
            }
            // And text that is not UTF-8 is read lossily, which is fine for a
            // reader and fatal for a writer.
            if !text::text_is_utf8(&storage) {
                problems.push(format!(
                    "{stream} storage {}: the text is not valid UTF-8, so an edit to \
                     this storage is refused",
                    object.identifier
                ));
            }

            for spec in text::TABLES {
                let field = spec.field;
                let Some(table) = storage.bytes(field).and_then(crate::pb::decode_nested) else {
                    continue;
                };
                let where_ = format!("{stream} storage {} {}", object.identifier, spec.name);
                if spec.anchoring == text::Anchoring::Range {
                    // These carry explicit ranges rather than run starts, so
                    // the rules are different: a range has to fit in the text
                    // and cover something.
                    for (location, span) in text::ranges(&table) {
                        if location + span > length {
                            problems.push(format!(
                                "{where_}: range {location}..{} runs past the text, \
                                 which is {length} long",
                                location + span
                            ));
                        }
                        if span == 0 {
                            problems.push(format!("{where_}: range at {location} is empty"));
                        }
                    }
                    continue;
                }
                let entries = text::entry_indices(&table, spec.anchoring);
                let mut previous: Option<u64> = None;
                for (index, _) in &entries {
                    if previous.is_some_and(|p| *index <= p) {
                        problems.push(format!(
                            "{where_}: entry index {index} does not increase (after {})",
                            previous.unwrap()
                        ));
                    }
                    previous = Some(*index);
                    if *index > length {
                        problems.push(format!(
                            "{where_}: entry at {index} but the text is {length} long"
                        ));
                    }
                    // A paragraph attribute may only sit where a paragraph
                    // begins, or at the very end of the text — the slot the
                    // style of a paragraph not yet typed comes from. Keynote
                    // renders a text box 2^16 points tall and then crashes on a
                    // paragraph run past the end of the text.
                    if spec.anchoring == text::Anchoring::Paragraph
                        && *index != length
                        && !starts.contains(index)
                    {
                        problems.push(format!(
                            "{where_}: entry at {index} is neither a paragraph start \
                             nor the end of the text"
                        ));
                    }
                }
                // A run or paragraph table describes the text from its first
                // entry onward, so one that starts late leaves characters with
                // no attribute at all. An anchor table is a set of points and
                // has no such obligation.
                if let Some((first, _)) = entries.first() {
                    if *first != 0 && spec.anchoring != text::Anchoring::Character {
                        problems.push(format!(
                            "{where_}: the first entry is at {first}, so characters \
                             0..{first} have no attribute"
                        ));
                    }
                }
                for (index, target) in &entries {
                    let where_ = format!("{stream} storage {} table {field}", object.identifier);
                    let _ = index;
                    let Some(target) = target else { continue };
                    let target = *target;
                    if self.object(target).is_none() {
                        problems.push(format!("{where_}: run points at missing object {target}"));
                    } else if let Some(kind) = styles.get(&target) {
                        if kind.attribute_table() != field {
                            problems.push(format!(
                                "{where_}: run points at a {} style, which belongs in table {}",
                                kind.as_str(),
                                kind.attribute_table()
                            ));
                        }
                    }
                }
            }
        }

        // Style archives must resolve their parent and stylesheet, and a style
        // listed in a stylesheet must also be grouped under its parent.
        for style in self.text_styles() {
            for (what, target) in [("parent", style.parent), ("stylesheet", style.stylesheet)] {
                let Some(target) = target else { continue };
                if target != 0 && self.object(target).is_none() {
                    problems.push(format!(
                        "style {}: {what} {target} does not exist",
                        style.identifier
                    ));
                }
            }
            let (Some(parent), Some(sheet)) = (style.parent, style.stylesheet) else {
                continue;
            };
            let Ok(archive) = self.archive_of(sheet) else {
                continue;
            };
            if style::count_references(&archive, style.identifier) == 1 {
                problems.push(format!(
                    "style {}: listed in stylesheet {sheet} but not grouped under \
                     its parent {parent}",
                    style.identifier
                ));
            }
        }

        // Drawables and the media registry check each other. Every rule here
        // is one the apps keep across the whole corpus, and one a writer can
        // break without the document failing to open.
        problems.extend(self.media_problems());

        for (component, object, target) in self.undeclared_references() {
            problems.push(format!(
                "object {object} refers to {target} in another component, which \
                 component {component} does not declare"
            ));
        }

        // The Pages spine checks itself: sections that begin on a break,
        // section templates with their three zones each, threads whose boxes
        // exist. See `structure_problems`.
        problems.extend(self.structure_problems());

        // So does the Keynote show: the slide tree against the nodes, the nodes
        // against their slides, layouts that resolve, placeholders whose kind
        // matches the field that names them. See `show_problems`.
        problems.extend(self.show_problems());

        // Tables check themselves: keys that resolve, refcounts that match the
        // cells, cell counts that match the records. See `Table::audit` for why
        // each of those is a rule.
        for table in self.tables() {
            for problem in &table.problems {
                problems.push(format!(
                    "table {} {}: {problem}",
                    table.identifier, table.name
                ));
            }
            problems.extend(table.audit());
        }

        // Charts check themselves too, and the rules are the ones a chart-aware
        // writer could break silently. See `chart_problems`.
        problems.extend(self.chart_problems());
        problems
    }

    /// Everything about a chart that looks wrong.
    ///
    /// Four rules, kept by every chart in the corpus and by every chart in the
    /// 69 bundled templates that have one, and each of them breakable by an
    /// edit that changes a chart's data without changing everything else:
    ///
    /// * **The grid is rectangular and its names fit it.** One row name per
    ///   row, one column name per column. A chart whose names and grid disagree
    ///   draws a series with no label, or labels a series that is not there.
    /// * **A sparse array's indices are inside its count.** `count` is the
    ///   logical number of series and the entries are the overrides; an index
    ///   at or past the count points at a series that does not exist.
    /// * **A style a chart names exists**, and is of the type the slot is for —
    ///   a chart style is a 5022, a series style a 5028.
    /// * **A mediator's back-reference points at the chart that names it.** The
    ///   binding is written both ways, and a chart whose mediator names another
    ///   chart follows the wrong table.
    fn chart_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let types: BTreeMap<u64, u32> = self
            .objects()
            .map(|(_, object)| (object.identifier, object.message_type()))
            .collect();

        for chart in self.charts() {
            let grid = &chart.grid;
            let width = grid.column_count();
            if grid.rows.iter().any(|row| row.len() != width) {
                problems.push(format!(
                    "chart {}: the grid is not rectangular",
                    chart.identifier
                ));
            }
            if grid.row_names.len() != grid.rows.len() {
                problems.push(format!(
                    "chart {}: {} row names for {} rows",
                    chart.identifier,
                    grid.row_names.len(),
                    grid.rows.len()
                ));
            }
            if grid.column_names.len() != width {
                problems.push(format!(
                    "chart {}: {} column names for {width} columns",
                    chart.identifier,
                    grid.column_names.len()
                ));
            }

            for (which, array) in [
                ("private series styles", &chart.series_private_styles),
                ("series non-styles", &chart.series_non_styles),
            ] {
                for (index, _) in &array.entries {
                    if *index >= array.count {
                        problems.push(format!(
                            "chart {}: {which} has an entry at {index} in an array of {}",
                            chart.identifier, array.count
                        ));
                    }
                }
            }

            let mut expect = |slot: &str, target: Option<u64>, kind: u32| {
                let Some(target) = target else { return };
                match types.get(&target) {
                    None => problems.push(format!(
                        "chart {}: {slot} points at missing object {target}",
                        chart.identifier
                    )),
                    Some(found) if *found != kind => problems.push(format!(
                        "chart {}: {slot} points at object {target} of type {found}, not {kind}",
                        chart.identifier
                    )),
                    Some(_) => {}
                }
            };
            expect(
                "chart style",
                chart.chart_style,
                crate::chart::TYPE_CHART_STYLE,
            );
            expect(
                "chart non-style",
                chart.chart_non_style,
                crate::chart::TYPE_CHART_NON_STYLE,
            );
            expect(
                "legend style",
                chart.legend_style,
                crate::chart::TYPE_LEGEND_STYLE,
            );
            expect(
                "legend non-style",
                chart.legend_non_style,
                crate::chart::TYPE_LEGEND_NON_STYLE,
            );
            for target in &chart.series_theme_styles {
                expect(
                    "series theme style",
                    Some(*target),
                    crate::chart::TYPE_SERIES_STYLE,
                );
            }
            for (_, target) in &chart.series_private_styles.entries {
                expect(
                    "private series style",
                    Some(*target),
                    crate::chart::TYPE_SERIES_STYLE,
                );
            }
            for (_, target) in &chart.series_non_styles.entries {
                expect(
                    "series non-style",
                    Some(*target),
                    crate::chart::TYPE_SERIES_NON_STYLE,
                );
            }
            for target in chart
                .value_axis_styles
                .iter()
                .chain(chart.category_axis_styles.iter())
            {
                expect("axis style", Some(*target), crate::chart::TYPE_AXIS_STYLE);
            }
            for target in chart
                .value_axis_non_styles
                .iter()
                .chain(chart.category_axis_non_styles.iter())
            {
                expect(
                    "axis non-style",
                    Some(*target),
                    crate::chart::TYPE_AXIS_NON_STYLE,
                );
            }

            // The mediator names the chart back, at field 1 of the base
            // `TSCH.ChartMediatorArchive`.
            if let Some(mediator) = chart.mediator {
                let back = self
                    .objects()
                    .find(|(_, object)| object.identifier == mediator)
                    .and_then(|(_, object)| Message::decode(object.payload()).ok())
                    .and_then(|archive| {
                        let base = match archive.bytes(1).and_then(crate::pb::decode_nested) {
                            Some(base) => base,
                            None => archive,
                        };
                        base.bytes(1)
                            .and_then(crate::pb::decode_nested)
                            .and_then(|r| r.varint(1))
                    });
                match back {
                    Some(target) if target != chart.identifier => problems.push(format!(
                        "chart {}: its mediator {mediator} names chart {target}",
                        chart.identifier
                    )),
                    _ => {}
                }
            }
        }
        problems
    }

    /// What a Keynote show has to hold, and what the phase-8a writes could
    /// break.
    ///
    /// Nothing here fires on a document Keynote wrote.
    ///
    /// * **Every entry in the slide tree is a `KN.SlideNodeArchive`, and no
    ///   node is in it twice.** The tree is a positional list of references;
    ///   a reorder that duplicated an entry would show the same slide twice and
    ///   the app would still open the deck.
    /// * **Every node names a slide, and the slide exists and is a
    ///   `KN.SlideArchive`.** A node whose slide did not load is what an
    ///   undeclared slide component looks like from the outside: the app counts
    ///   the slide and answers `missing value` for everything on it.
    /// * **Every slide names a layout, and the layout is one the theme lists.**
    ///   A `template_slide` pointing at anything else is a slide with no
    ///   master, which is how the app reports a copy gone wrong.
    /// * **A placeholder's kind matches the field that names it.** Field 5 is a
    ///   title (kind 2), 6 a body (3), 20 a slide number (1), 30 an object
    ///   well (4) — true of all 86 placeholders in `keynote-deck` and of every
    ///   one in the other two decks.
    /// * **A slide's own objects are in its own component.** A slide whose
    ///   placeholders or note storage live in another stream is a copy that
    ///   shared what it should have duplicated.
    /// * **Two slides never share a stream.** Each slide is its own component;
    ///   a copy that wrote into the original's stream would open, and the two
    ///   would be one slide as far as loading goes.
    fn show_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let Some(show) = self.show() else {
            return problems;
        };
        let layouts: BTreeSet<u64> = show.layouts.iter().map(|l| l.identifier).collect();
        let components: BTreeSet<u64> = self.components().iter().map(|c| c.identifier).collect();
        let mut seen_nodes: BTreeSet<u64> = BTreeSet::new();
        let mut seen_streams: BTreeMap<String, u64> = BTreeMap::new();

        for slide in &show.slides {
            // Every slide stream needs a `TSP.ComponentInfo` keyed by its slide
            // archive's id, or nothing declares it to the document — the
            // quietest way a duplicate can fail. The existing component→stream
            // check does not catch this: a slide with no component at all has
            // no entry to check.
            if !components.contains(&slide.identifier) {
                problems.push(format!(
                    "slide {} has no TSP.ComponentInfo, so nothing declares its stream {}",
                    slide.identifier, slide.stream
                ));
            }
            // `hasNote` (node field 8) is 1 exactly when the note storage holds
            // text — zero mismatches across the corpus. A blind reader of the
            // flag, and any tool that trusts it, is wrong the moment an edit
            // sets the text without the flag or the flag without the text.
            if let Ok(node) = self.archive_of(slide.node) {
                let has_note_flag = node
                    .varint(crate::keynote::node_field::HAS_NOTE)
                    .is_some_and(|v| v != 0);
                if has_note_flag == slide.notes.is_empty() {
                    problems.push(format!(
                        "slide {}: node {} says hasNote={} but its note storage is {}",
                        slide.identifier,
                        slide.node,
                        u8::from(has_note_flag),
                        if slide.notes.is_empty() {
                            "empty"
                        } else {
                            "non-empty"
                        }
                    ));
                }
            }
            if !seen_nodes.insert(slide.node) {
                problems.push(format!(
                    "slide node {} is in the slide tree more than once",
                    slide.node
                ));
            }
            match self.object(slide.node) {
                Some((_, object)) if object.message_type() == crate::keynote::TYPE_SLIDE_NODE => {}
                Some((_, object)) => problems.push(format!(
                    "slide tree entry {} is type {}, not a KN.SlideNodeArchive",
                    slide.node,
                    object.message_type()
                )),
                None => problems.push(format!("slide tree names missing node {}", slide.node)),
            }
            match self.object(slide.identifier) {
                Some((_, object)) if object.message_type() == crate::keynote::TYPE_SLIDE => {}
                Some((_, object)) => problems.push(format!(
                    "slide node {} names {}, which is type {}, not a KN.SlideArchive",
                    slide.node,
                    slide.identifier,
                    object.message_type()
                )),
                None => problems.push(format!(
                    "slide node {} names missing slide {}",
                    slide.node, slide.identifier
                )),
            }
            match slide.layout {
                Some(layout) if layouts.contains(&layout) => {}
                Some(layout) => problems.push(format!(
                    "slide {} is built on {layout}, which is not a layout the theme lists",
                    slide.identifier
                )),
                None => problems.push(format!("slide {} names no layout", slide.identifier)),
            }
            for (place, wanted, field) in [
                (&slide.title, crate::keynote::PlaceholderKind::Title, 5),
                (&slide.body, crate::keynote::PlaceholderKind::Body, 6),
                (
                    &slide.slide_number,
                    crate::keynote::PlaceholderKind::SlideNumber,
                    20,
                ),
                (&slide.object, crate::keynote::PlaceholderKind::Object, 30),
            ] {
                let Some(place) = place else { continue };
                if place.kind != wanted {
                    problems.push(format!(
                        "slide {}: field {field} names placeholder {}, whose kind is {}, not {}",
                        slide.identifier,
                        place.identifier,
                        place.kind.as_str(),
                        wanted.as_str()
                    ));
                }
            }
            if !slide.stream.is_empty() {
                if let Some(other) = seen_streams.insert(slide.stream.clone(), slide.identifier) {
                    problems.push(format!(
                        "slides {other} and {} are both in {}",
                        slide.identifier, slide.stream
                    ));
                }
                for object in slide
                    .texts
                    .iter()
                    .map(|t| t.storage)
                    .chain(slide.note)
                    .chain(slide.drawables.iter().copied())
                {
                    if let Some((stream, _)) = self.object(object) {
                        if stream != slide.stream {
                            problems.push(format!(
                                "slide {} owns {object}, which is in {stream}, not {}",
                                slide.identifier, slide.stream
                            ));
                        }
                    }
                }
            }
        }
        problems
    }

    /// Everything about a Pages document's structure that looks wrong.
    ///
    /// Four rules, each kept by every Pages document in the corpus **and by
    /// all 640 bundled Pages templates**, and each one breakable by an edit
    /// that moves text without thinking about what is anchored to it:
    ///
    /// * **A section that does not begin at 0 begins after a `U+0004`.** The
    ///   break is what makes the section, and an entry that has drifted off one
    ///   is a section boundary the app will not draw.
    /// * **A section's template pages are section templates.** Field 23, 24
    ///   and 25 of `TP.SectionArchive` reach `TP.SectionTemplateArchive`s, and
    ///   a section with none has no headers, footers or page background at all.
    /// * **A section template has three headers and three footers, and every
    ///   one is a storage of kind 1.** Never any other count: 3144 section
    ///   templates across the bundled templates and 66 in this corpus.
    /// * **A thread's storage and boxes exist.** A linked-text-box thread that
    ///   names a missing box is a flow with a hole in it.
    fn structure_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let Some(structure) = self.structure() else {
            return problems;
        };
        let body: Vec<u16> = structure
            .body_storage
            .and_then(|id| self.storage_text(id).ok())
            .unwrap_or_default()
            .encode_utf16()
            .collect();

        for section in &structure.sections {
            if section.start > 0 {
                let before = body.get(section.start as usize - 1).copied();
                if before != Some(0x0004) {
                    problems.push(format!(
                        "section {} begins at character {} and the character before it \
                         is not the U+0004 that starts a section",
                        section.identifier, section.start
                    ));
                }
            }
            if section.templates.iter().all(Option::is_none) {
                problems.push(format!(
                    "section {} has no first, even or odd section template",
                    section.identifier
                ));
            }
            for template in section.templates.iter().flatten() {
                match self.object(*template) {
                    Some((_, object))
                        if object.message_type() == crate::pages::TYPE_SECTION_TEMPLATE => {}
                    Some((_, object)) => problems.push(format!(
                        "section {}: page template {template} is type {}, not a \
                         TP.SectionTemplateArchive",
                        section.identifier,
                        object.message_type()
                    )),
                    None => problems.push(format!(
                        "section {}: page template {template} does not exist",
                        section.identifier
                    )),
                }
            }
        }

        let mut zones: BTreeMap<(u64, bool), usize> = BTreeMap::new();
        for entry in &structure.header_footers {
            *zones
                .entry((entry.section_template, entry.footer))
                .or_default() += 1;
            match self.object(entry.storage) {
                // A zone Pages has nothing to put in is a **null reference** —
                // `{1: 0}`, object zero, which is no object. Watched: Pages
                // resaving a document made from nothing writes three of those
                // per section template rather than three empty storages.
                _ if entry.storage == 0 => {}
                Some((_, object)) if object.message_type() == crate::TYPE_STORAGE => {
                    let kind = Message::decode(object.payload())
                        .ok()
                        .and_then(|m| m.varint(1));
                    if kind != Some(1) {
                        problems.push(format!(
                            "{} {} of section template {} is a storage of kind {}, not 1",
                            entry.kind(),
                            entry.storage,
                            entry.section_template,
                            kind.map(|k| k.to_string()).unwrap_or_else(|| "none".into())
                        ));
                    }
                }
                _ => problems.push(format!(
                    "{} {} of section template {} is not a text storage",
                    entry.kind(),
                    entry.storage,
                    entry.section_template
                )),
            }
        }
        for ((template, footer), count) in zones {
            if count != 3 {
                problems.push(format!(
                    "section template {template} has {count} {}(s), not three",
                    if footer { "footer" } else { "header" }
                ));
            }
        }

        for thread in &structure.threads {
            for object in thread.storage.iter().chain(thread.boxes.iter()) {
                if self.object(*object).is_none() {
                    problems.push(format!(
                        "linked-text-box thread {} names missing object {object}",
                        thread.identifier
                    ));
                }
            }
        }

        // A footnote is a chain of three: a `table_footnote` entry on the
        // `U+000E` mark, the `TSWP.FootnoteReferenceAttachmentArchive` it
        // names, and the kind-2 storage that archive contains. **The mark is
        // the only route in.** Delete its character and the other two stay in
        // the document with nothing reaching them — which is exactly what
        // `iwork delete-text` did while it believed a footnote mark was a
        // `U+FFFC`, and which nothing here could see afterwards. Both halves
        // are cheap to check and both are checked, because an orphan is the
        // signature of that damage.
        let anchored: BTreeSet<u64> = structure
            .footnotes
            .iter()
            .filter_map(|note| note.attachment)
            .collect();
        let bodies: BTreeSet<u64> = structure
            .footnotes
            .iter()
            .filter_map(|note| note.body)
            .collect();
        for (stream, object) in self.objects() {
            if object.message_type() == crate::pages::TYPE_FOOTNOTE_REFERENCE
                && !anchored.contains(&object.identifier)
            {
                problems.push(format!(
                    "{stream} footnote attachment {} is not anchored in any text, so \
                     the note it carries cannot be reached",
                    object.identifier
                ));
            }
            if object.message_type() == crate::TYPE_STORAGE && !bodies.contains(&object.identifier)
            {
                let kind = Message::decode(object.payload())
                    .ok()
                    .and_then(|m| m.varint(1));
                if kind == Some(2) {
                    problems.push(format!(
                        "{stream} storage {} is a footnote body (kind 2) that no \
                         footnote mark contains",
                        object.identifier
                    ));
                }
            }
        }
        problems
    }

    /// Everything about drawables and media that looks wrong.
    ///
    /// Five rules, each of them kept by every document in the corpus and each
    /// of them breakable by a writer without the app noticing at open time:
    ///
    /// * **A data reference resolves.** A drawable naming media the registry
    ///   does not list draws nothing, silently.
    /// * **A stored file is there and its digest is its bytes.** The digest is
    ///   a raw SHA-1 and iWork uses it to recognise the file; writing new bytes
    ///   under an old digest is accepted when the document opens and is a lie
    ///   afterwards.
    /// * **`MessageInfo.data_references` lists what the payload uses.** The
    ///   framing carries a packed list of the data ids inside each object, and
    ///   across the corpus it is exactly the set the archive names — 12
    ///   drawables in two documents, no extras and none missing.
    /// * **A mask's parent is the image that names it.** The link is written
    ///   both ways and the pair is meaningless if they disagree.
    /// * **A drawable's parent exists.**
    fn media_problems(&self) -> Vec<String> {
        let mut problems = Vec::new();
        let files = self.data_files();
        let known: BTreeSet<u64> = files.iter().map(|f| f.identifier).collect();

        for file in &files {
            let Some(entry) = file.entry_name() else {
                continue;
            };
            let Some(bytes) = self.package.get(&entry) else {
                problems.push(format!(
                    "media {} names {entry}, which the package does not have",
                    file.identifier
                ));
                continue;
            };
            if !file.digest.is_empty() && file.digest != crate::media::sha1(bytes) {
                problems.push(format!(
                    "media {}: the digest is not the SHA-1 of {entry}",
                    file.identifier
                ));
            }
        }

        let drawables = self.drawables();
        for drawable in &drawables {
            if let Some(parent) = drawable.parent {
                if self.object(parent).is_none() {
                    problems.push(format!(
                        "drawable {} has parent {parent}, which does not exist",
                        drawable.identifier
                    ));
                }
            }
            if let Some(mask) = drawable.mask() {
                match self.drawable(mask) {
                    Some(mask_drawable) if mask_drawable.parent == Some(drawable.identifier) => {}
                    Some(mask_drawable) => problems.push(format!(
                        "image {} is masked by {mask}, whose parent is {:?}",
                        drawable.identifier, mask_drawable.parent
                    )),
                    None => problems.push(format!(
                        "image {} is masked by {mask}, which does not exist",
                        drawable.identifier
                    )),
                }
            }

            let used: BTreeSet<u64> = drawable
                .media
                .iter()
                .flat_map(|media| [media.data, media.poster])
                .flatten()
                .collect();
            for data in &used {
                if !known.contains(data) {
                    problems.push(format!(
                        "drawable {} refers to media {data}, which the registry does not list",
                        drawable.identifier
                    ));
                }
            }
            if used.is_empty() {
                continue;
            }
            let Some((_, object)) = self.object(drawable.identifier) else {
                continue;
            };
            let declared: BTreeSet<u64> = object
                .messages
                .iter()
                .flat_map(|message| message.extra.iter())
                .filter(|field| field.number == 6)
                .filter_map(|field| match &field.value {
                    Value::Bytes(raw) => Some(raw),
                    _ => None,
                })
                .flat_map(|raw| {
                    let mut reader = crate::pb::Reader::new(raw);
                    let mut out = Vec::new();
                    while !reader.done() {
                        match reader.varint() {
                            Ok(value) => out.push(value),
                            Err(_) => break,
                        }
                    }
                    out
                })
                .collect();
            let missing: Vec<String> = used
                .difference(&declared)
                .map(u64::to_string)
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                problems.push(format!(
                    "drawable {} uses media {} without listing it in the object's \
                     data references",
                    drawable.identifier,
                    missing.join(", ")
                ));
            }
        }

        // -- identity, which lives outside the object graph ------------------
        //
        // Three places say what the document is, and an edit that rewrote one
        // of them and not the others would leave a file whose parts disagree.
        // What is checked is the **agreement**, never the presence: a document
        // with no `Properties.plist` is not this crate's doing, while three
        // that contradict each other is exactly what a bad `save_as_new` looks
        // like.
        match crate::metadata::Properties::read(&self.package) {
            Err(e) => problems.push(format!("{}: {e}", crate::metadata::PROPERTIES)),
            Ok(None) => {}
            Ok(Some(properties)) => {
                let identifier = self
                    .package
                    .get(crate::metadata::DOCUMENT_IDENTIFIER)
                    .map(|raw| String::from_utf8_lossy(raw).trim().to_string());
                if identifier.is_some() && identifier != properties.document_uuid {
                    problems.push(format!(
                        "{} says {:?} and Properties.plist says {:?}",
                        crate::metadata::DOCUMENT_IDENTIFIER,
                        identifier.unwrap_or_default(),
                        properties.document_uuid.unwrap_or_default()
                    ));
                } else if properties.share_uuid != properties.document_uuid {
                    problems.push(format!(
                        "shareUUID {:?} is not the documentUUID {:?}",
                        properties.share_uuid.unwrap_or_default(),
                        properties.document_uuid.unwrap_or_default()
                    ));
                }
                if let (Some(revision), Some(version)) =
                    (&properties.revision, &properties.version_uuid)
                {
                    if !revision.ends_with(version) {
                        problems.push(format!(
                            "revision {revision} does not end in the versionUUID {version}"
                        ));
                    }
                }
            }
        }

        // A comment or a tracked change that points at an author the storage
        // does not list is a dangling reference the object walk above cannot
        // see, because both ends exist.
        let annotations = self.annotations();
        for comment in &annotations.comments {
            if let Some(author) = comment.author {
                if !annotations.authors.iter().any(|a| a.identifier == author) {
                    problems.push(format!(
                        "comment {} names author {author}, which is not an annotation author",
                        comment.identifier
                    ));
                }
            }
        }
        problems
    }

    // -- components ----------------------------------------------------------

    /// References that leave the component they are written in without being
    /// declared in it, as `(component, referring object, target)`.
    ///
    /// A package is loaded component by component. Within a component an
    /// identifier resolves against the objects in the same stream; to reach an
    /// object in another component, the referring component must list it in its
    /// `TSP.ComponentInfo.external_references` as
    /// `{1: target's component, 2: target}`. iWork keeps that list exact: across
    /// five Pages documents and one Keynote document — some five thousand
    /// objects — every cross-component reference is declared and this returns
    /// nothing.
    ///
    /// An undeclared reference does not always crash. A Pages document that
    /// pointed a paragraph at an undeclared style opened with the paragraph
    /// simply unstyled, as though the edit had never been made. That silence is
    /// the reason to check rather than to trust the file opening.
    ///
    /// **A reference to a component's root object needs declaring too**, and
    /// this used to say it did not. The component identifier *is* the root
    /// object's identifier, so it looked like naming a component rather than
    /// crossing into one — but Keynote's `Index/Document.iwa` declares
    /// `{component_identifier: <slide>}` for every slide component its nodes
    /// point at, 475 of them in a five-slide deck, and a slide component that
    /// nothing declares does not load: the app opens the deck, counts the extra
    /// slide, and answers `missing value` for its layout, its title and its
    /// body. That is what a slide copied without the declaration looks like,
    /// and it is how the rule was found.
    ///
    /// Two components are exempt, and only two: `Document` and
    /// `DocumentMetadata`, whose roots are object 1 and object 71 and which
    /// every reader has open before it reads anything else. With those two
    /// excused the rule costs nothing on documents the apps wrote — over all 23
    /// documents of this corpus it reports **zero** undeclared references,
    /// because iWork already declares every one of the rest.
    pub fn undeclared_references(&self) -> Vec<(u64, u64, u64)> {
        /// `MessageInfo.data_references` (6), packed varints, over every
        /// message of one object.
        fn data_references(object: &crate::iwa::ArchiveObject) -> BTreeSet<u64> {
            let mut out = BTreeSet::new();
            for field in object
                .messages
                .iter()
                .flat_map(|message| message.extra.iter())
                .filter(|field| field.number == 6)
            {
                let Value::Bytes(raw) = &field.value else {
                    continue;
                };
                let mut reader = crate::pb::Reader::new(raw);
                while !reader.done() {
                    match reader.varint() {
                        Ok(value) => out.insert(value),
                        Err(_) => break,
                    };
                }
            }
            out
        }

        let Some(index) = self.component_index() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for (stream, object) in self.objects() {
            let Some(&from) = index.by_stream.get(stream) else {
                continue;
            };
            let mut targets = Vec::new();
            for message in &object.messages {
                if let Ok(archive) = Message::decode(&message.payload) {
                    targets.extend(style::references(&archive));
                }
            }
            targets.sort_unstable();
            targets.dedup();
            // A `TSP.DataReference` is `{1: identifier}` and so is a
            // `TSP.Reference`: nothing in the bytes tells them apart, and the
            // two identifier spaces overlap. Keynote proves it — resaving a
            // deck this crate wrote, it gave a slide thumbnail the *data*
            // identifier 1041 while object 1041 was an image, and
            // `KN.SlideNodeArchive.thumbnails` (16) then read as a reference
            // to that image in another component. The file settles it: the
            // object's own `MessageInfo.data_references` names which of these
            // identifiers are data.
            let data = data_references(object);
            for target in targets {
                if data.contains(&target) {
                    continue;
                }
                let Some(&to) = index.by_object.get(&target) else {
                    continue;
                };
                if to == from || index.always_loaded.contains(&target) {
                    continue;
                }
                if !index
                    .declared
                    .get(&from)
                    .is_some_and(|d| d.contains(&target))
                {
                    out.push((from, object.identifier, target));
                }
            }
        }
        out
    }

    /// Declare every reference [`Document::undeclared_references`] reports, so
    /// that the objects an edit points at can actually be loaded.
    ///
    /// Nothing is remembered between calls: the declarations are derived from
    /// the objects as they now stand, so this cannot drift out of step with the
    /// edits, and running it twice adds nothing the second time. Existing
    /// declarations are never removed — iWork's own are kept as written, and a
    /// declaration for an object that is no longer referenced is inert.
    ///
    /// Returns the number of declarations added.
    pub fn declare_external_references(&mut self) -> usize {
        let missing = self.undeclared_references();
        if missing.is_empty() {
            return 0;
        }
        let Some(index) = self.component_index() else {
            return 0;
        };
        let Some((stream, position)) = self
            .objects()
            .find(|(_, o)| o.message_type() == crate::TYPE_PACKAGE_METADATA)
            .map(|(s, o)| (s.to_string(), o.identifier))
            .and_then(|(s, id)| self.locate(id).map(|(_, i)| (s, i)))
        else {
            return 0;
        };

        let object = &mut self
            .streams
            .get_mut(&stream)
            .expect("stream came from the document")[position];
        let message = object.messages.first_mut().expect("metadata has a payload");
        let Ok(mut metadata) = Message::decode(&message.payload) else {
            return 0;
        };

        let mut added = 0;
        for field in &mut metadata.fields {
            if field.number != 3 {
                continue;
            }
            let Value::Bytes(raw) = &field.value else {
                continue;
            };
            let Ok(mut info) = Message::decode(raw) else {
                continue;
            };
            let Some(component) = info.varint(1) else {
                continue;
            };
            let mut wanted: Vec<u64> = missing
                .iter()
                .filter(|(from, _, _)| *from == component)
                .map(|(_, _, target)| *target)
                .collect();
            wanted.sort_unstable();
            wanted.dedup();
            for target in wanted {
                let mut entry = Message::default();
                entry.set(1, Value::Varint(index.by_object[&target]));
                // A component's *root* is declared by naming the component and
                // nothing else — `{component_identifier}` with no
                // `object_identifier` — which is the form iWork writes for
                // every one of them.
                if !index.roots.contains(&target) {
                    entry.set(2, Value::Varint(target));
                }
                info.append_in_order(6, Value::Bytes(entry.encode()));
                added += 1;
            }
            if added > 0 {
                field.value = Value::Bytes(info.encode());
            }
        }
        if added > 0 {
            message.payload = metadata.encode();
        }
        added
    }

    /// Which component each stream and each object belongs to, and what each
    /// component already declares.
    fn component_index(&self) -> Option<ComponentIndex> {
        let metadata = self.package_metadata()?;
        let mut index = ComponentIndex::default();
        for component in self.components() {
            index.roots.insert(component.identifier);
            if matches!(
                component.preferred_name.as_str(),
                "Document" | "DocumentMetadata"
            ) {
                index.always_loaded.insert(component.identifier);
            }
            index
                .by_stream
                .insert(component.stream_name(), component.identifier);
        }
        for (stream, object) in self.objects() {
            if let Some(&component) = index.by_stream.get(stream) {
                index.by_object.insert(object.identifier, component);
            }
        }
        for value in metadata.all(3) {
            let Value::Bytes(raw) = value else { continue };
            let Ok(info) = Message::decode(raw) else {
                continue;
            };
            let Some(component) = info.varint(1) else {
                continue;
            };
            let declared = index.declared.entry(component).or_default();
            for value in info.all(6) {
                let Value::Bytes(raw) = value else { continue };
                if let Ok(entry) = Message::decode(raw) {
                    // `{component_identifier, object_identifier}` declares one
                    // object; `{component_identifier}` alone declares the
                    // component's root, whose identifier is the component's.
                    match entry.varint(2) {
                        Some(target) => declared.insert(target),
                        None => match entry.varint(1) {
                            Some(component) => declared.insert(component),
                            None => continue,
                        },
                    };
                }
            }
        }
        Some(index)
    }

    /// Identifier a new object may take: above every identifier in use and above
    /// the package's own high-water mark.
    pub fn next_object_identifier(&self) -> u64 {
        let highest = self
            .objects()
            .map(|(_, object)| object.identifier)
            .max()
            .unwrap_or(0);
        highest.max(self.last_object_identifier().unwrap_or(0)) + 1
    }

    // -- internals -----------------------------------------------------------

    /// Stream and position of an object, for the methods that rewrite one.
    fn locate(&self, identifier: u64) -> Option<(String, usize)> {
        self.streams.iter().find_map(|(name, objects)| {
            objects
                .iter()
                .position(|object| object.identifier == identifier)
                .map(|index| (name.clone(), index))
        })
    }

    /// Decode any object's first message, at the wire level.
    ///
    /// The escape hatch for a caller that knows a message this crate does not
    /// model: the fields come back as they were written, in the order they were
    /// written, and an unrecognised one is a field like any other.
    pub fn archive(&self, identifier: u64) -> Result<Message, Error> {
        self.archive_of(identifier)
    }

    /// Decode any object's first message.
    fn archive_of(&self, identifier: u64) -> Result<Message, Error> {
        let (name, object) = self
            .object(identifier)
            .ok_or(Error::NoSuchObject(identifier))?;
        let message = object
            .messages
            .first()
            .ok_or_else(|| Error::Format(format!("object {identifier} carries no message")))?;
        Message::decode(&message.payload)
            .map_err(|e| Error::Format(format!("{name}: object {identifier}: {e}")))
    }

    fn storage_archive(&self, identifier: u64) -> Result<Message, Error> {
        let (name, object) = self
            .object(identifier)
            .ok_or(Error::NoSuchObject(identifier))?;
        if object.message_type() != crate::TYPE_STORAGE {
            return Err(Error::Format(format!(
                "object {identifier} is not a text storage"
            )));
        }
        Message::decode(object.payload())
            .map_err(|e| Error::Format(format!("{name}: storage {identifier}: {e}")))
    }

    /// `Document::set_archive`, for the modules that model one app.
    /// Write any object's archive back, at the wire level.
    ///
    /// The escape hatch [`Document::archive`] is the read half of: a caller
    /// that understands a message this crate does not can edit it and put it
    /// back. Nothing checks what it means — `iwork check` and
    /// [`Document::problems`] are what say whether the result is still a
    /// document.
    pub fn set_archive_for(&mut self, identifier: u64, archive: &Message) -> Result<(), Error> {
        self.set_archive(identifier, archive)
    }

    pub(crate) fn set_archive_of(
        &mut self,
        identifier: u64,
        archive: &Message,
    ) -> Result<(), Error> {
        self.set_archive(identifier, archive)
    }

    /// Every object of one stream, cloned — the raw material of a component
    /// copy.
    pub(crate) fn stream_objects(&self, name: &str) -> Vec<ArchiveObject> {
        self.streams.get(name).cloned().unwrap_or_default()
    }

    /// Add a stream the package does not have yet.
    ///
    /// [`Document::save`] writes any stream whose framed bytes differ from the
    /// package's, and a stream the package has never held differs from nothing
    /// — so a new entry appears with no further ceremony. Adding one twice is
    /// refused rather than silently replacing a component.
    pub(crate) fn add_stream(
        &mut self,
        name: &str,
        objects: Vec<ArchiveObject>,
    ) -> Result<(), Error> {
        if self.streams.contains_key(name) {
            return Err(Error::Format(format!("{name} is already in the package")));
        }
        self.streams.insert(name.to_string(), objects);
        Ok(())
    }

    /// Add an object to the stream that holds `neighbour`, straight after it.
    ///
    /// The framing — version numbers, the message-type header — is the
    /// neighbour's, which is the copy-don't-synthesise rule applied to the one
    /// part of an object this crate would otherwise have to invent.
    pub(crate) fn add_object_after(
        &mut self,
        neighbour: u64,
        identifier: u64,
        message_type: u32,
        archive: &Message,
    ) -> Result<(), Error> {
        self.add_object_after_with(neighbour, identifier, message_type, archive, &[])
    }

    /// The same, declaring the media the new object refers to.
    ///
    /// `MessageInfo.data_references` (6) is the object's own list of the
    /// `TSP.DataInfo` identifiers its payload names, and an image that does not
    /// declare its picture is a document `Document::problems` calls out —
    /// rightly, since the app uses the list to decide what to load.
    pub(crate) fn add_object_after_with(
        &mut self,
        neighbour: u64,
        identifier: u64,
        message_type: u32,
        archive: &Message,
        data_references: &[u64],
    ) -> Result<(), Error> {
        if self.object(identifier).is_some() {
            return Err(Error::Format(format!(
                "object {identifier} is already in the document"
            )));
        }
        let (stream, index) = self
            .locate(neighbour)
            .ok_or(Error::NoSuchObject(neighbour))?;
        let mut object = self.streams[&stream][index].clone();
        object.identifier = identifier;
        // The neighbour is borrowed for its *shape*, not its contents. Its
        // `ArchiveInfo` and `MessageInfo` extras are about the neighbour —
        // which objects and which media it names, and, in fields 7 to 11,
        // whether it is a version patch of something. Carrying those onto a
        // new object claims media it does not use and makes it a patch of a
        // message that has nothing to do with it.
        object.extra.clear();
        object.messages.truncate(1);
        let message = object
            .messages
            .first_mut()
            .ok_or_else(|| Error::Format(format!("object {neighbour} carries no message")))?;
        message.extra.clear();
        message.message_type = message_type;
        message.payload = archive.encode();
        if !data_references.is_empty() {
            let mut packed = Vec::new();
            for data in data_references {
                crate::pb::write_varint(&mut packed, *data);
            }
            message.extra.push(crate::pb::Field {
                number: 6,
                value: Value::Bytes(packed),
            });
        }
        self.streams
            .get_mut(&stream)
            .expect("stream came from the document")
            .insert(index + 1, object);
        Ok(())
    }

    /// Edit `TSP.PackageMetadata` at the wire level.
    pub(crate) fn update_package_metadata(
        &mut self,
        edit: impl FnOnce(&mut Message),
    ) -> Result<(), Error> {
        let identifier = self
            .objects()
            .find(|(_, object)| object.message_type() == crate::TYPE_PACKAGE_METADATA)
            .map(|(_, object)| object.identifier)
            .ok_or_else(|| Error::Format("no TSP.PackageMetadata".into()))?;
        let mut metadata = self.archive_of(identifier)?;
        edit(&mut metadata);
        self.set_archive(identifier, &metadata)
    }

    fn set_archive(&mut self, identifier: u64, archive: &Message) -> Result<(), Error> {
        let (stream, index) = self
            .locate(identifier)
            .ok_or(Error::NoSuchObject(identifier))?;
        let object = &mut self
            .streams
            .get_mut(&stream)
            .expect("stream came from the document")[index];
        let message = object
            .messages
            .first_mut()
            .ok_or_else(|| Error::Format(format!("object {identifier} carries no message")))?;
        message.payload = archive.encode();
        Ok(())
    }

    /// Raise `TSP.PackageMetadata` field 1 so iWork does not reissue an
    /// identifier this crate has already handed out.
    pub(crate) fn set_last_object_identifier(&mut self, value: u64) -> Result<(), Error> {
        let metadata = self
            .objects()
            .find(|(_, object)| object.message_type() == crate::TYPE_PACKAGE_METADATA)
            .map(|(_, object)| object.identifier)
            .ok_or_else(|| {
                Error::Format(
                    "no TSP.PackageMetadata: cannot allocate an object identifier safely".into(),
                )
            })?;
        let (stream, index) = self.locate(metadata).expect("the object was just found");
        let object = &mut self
            .streams
            .get_mut(&stream)
            .expect("stream came from the document")[index];
        let message = object
            .messages
            .first_mut()
            .expect("the message type came from this message");
        let mut archive = Message::decode(&message.payload)
            .map_err(|e| Error::Format(format!("package metadata: {e}")))?;
        if archive.varint(1).unwrap_or(0) < value {
            archive.set(1, Value::Varint(value));
            message.payload = archive.encode();
        }
        Ok(())
    }

    /// Write the package out, re-encoding only the streams that changed.
    ///
    /// A stream whose objects still frame to the bytes they were read from
    /// keeps its **original entry, byte for byte** — not merely an equivalent
    /// one. That matters beyond saving work: re-compressing an untouched stream
    /// moves every Snappy block boundary in it, so a document edited in one
    /// place would otherwise differ from the original everywhere, and there
    /// would be no way to tell an intended change from an incidental one by
    /// looking at the file. Editing one style in a 97-stream Numbers document
    /// now rewrites one stream.
    ///
    /// Nothing is remembered to make this work: the comparison is against the
    /// bytes actually in the package, so it cannot drift out of step with the
    /// edits the way a dirty flag can.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> Result<(), Error> {
        self.saved_package().write(path)
    }

    /// Save under a **fresh document identity**, so the copy and the original
    /// are two documents rather than two versions of one.
    ///
    /// `documentUUID`, `shareUUID`, `privateUUID` and `versionUUID` are
    /// replaced, `revision` follows the new version, and
    /// `Metadata/DocumentIdentifier` is rewritten to match — the exact set
    /// Pages' own Save As was measured changing. `stableDocumentUUID` is kept,
    /// because that is what says the copy came from this document.
    ///
    /// Everything else is [`Document::save`]: the object streams are the same
    /// bytes, and `Metadata/BuildVersionHistory.plist` is left alone.
    ///
    /// Plain [`Document::save`] does the opposite and keeps the identity, which
    /// is what an edit-in-place wants — and which byte-identity already proves,
    /// since a no-op save reproduces `Properties.plist` exactly.
    pub fn save_as_new(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<crate::metadata::NewIdentity, Error> {
        let mut package = self.saved_package();
        let identity = crate::metadata::assign_new_identity(&mut package)?;
        package.write(path)?;
        Ok(identity)
    }

    /// The `preview*.jpg` entries: pictures of the document, drawn by the app.
    ///
    /// They are what the Finder and Quick Look show, and nothing else: no
    /// object stream in 928 packages — the corpus and all 901 bundled templates
    /// — so much as contains the string `preview`.
    pub fn previews(&self) -> Vec<&str> {
        self.package
            .names()
            .filter(|name| name.starts_with("preview") && !name.contains('/'))
            .collect()
    }

    /// Remove the previews, and return how many went.
    ///
    /// **This crate does not do it on its own.** An edit makes a preview stale
    /// — it is a rendering of the document as it was, and nothing here can
    /// draw a page — but leaving it is what iWork itself does between saves,
    /// and removing it would break the byte-identity a no-op save promises. See
    /// `FORMAT.md` §1 for the whole argument.
    ///
    /// What makes the choice available rather than dangerous: a document
    /// **without** previews is an ordinary document. All 901 template bundles
    /// have none, two fixtures are exactly that and the apps open them; a
    /// password-protected package has none either. And the app puts them back
    /// on its own next save — three documents made from templates had no
    /// previews, were opened and saved by Pages, Numbers and Keynote, and came
    /// back with three each.
    pub fn strip_previews(&mut self) -> usize {
        let names: Vec<String> = self.previews().iter().map(|n| n.to_string()).collect();
        for name in &names {
            self.package.remove(name);
        }
        names.len()
    }

    /// The package [`Document::save`] would write.
    fn saved_package(&self) -> Package {
        let mut package = self.package.clone();
        for (name, objects) in &self.streams {
            let framed = iwa::serialize_stream(objects);
            if !self.stream_matches(name, &framed) {
                package.set(name, iwa::compress(&framed));
            }
        }
        package
    }

    /// The package's metadata plists and the document-level fields of its root
    /// archive: identity, locale, template, build history, custom-format list.
    pub fn metadata(&self) -> Result<crate::metadata::Metadata, Error> {
        crate::metadata::metadata(self)
    }

    /// Streams whose objects no longer match the bytes they were read from —
    /// exactly the entries [`Document::save`] would rewrite.
    pub fn changed_streams(&self) -> Vec<&str> {
        self.streams
            .iter()
            .filter(|(name, objects)| !self.stream_matches(name, &iwa::serialize_stream(objects)))
            .map(|(name, _)| name.as_str())
            .collect()
    }

    /// Does the package still hold exactly these framed bytes for `name`?
    fn stream_matches(&self, name: &str, framed: &[u8]) -> bool {
        self.package
            .get(name)
            .and_then(|raw| iwa::decompress(raw).ok())
            .is_some_and(|original| original == framed)
    }
}

/// A float field, or zero.
fn float_field(message: &Message, number: u32) -> f32 {
    match message.get(number) {
        Some(Value::Fixed32(bytes)) => f32::from_le_bytes(*bytes),
        _ => 0.0,
    }
}

/// Scale every point of a baked path, when the path is drawn in the space the
/// natural size describes.
///
/// **Every point of every element, not the first of each.** A path element's
/// control points all live in field 2, and a `curveTo` (kind 4) carries three
/// of them — two control points and the endpoint. Reading only the first
/// occurrence measured the extent from partial data and, worse, left a scaled
/// curve with two of its three points at the old scale, deforming the outline.
fn scale_path_points(body: &mut Message, field: u32, was: (f32, f32), scale: (f32, f32)) {
    let Some(mut path) = body.bytes(field).and_then(crate::pb::decode_nested) else {
        return;
    };
    let mut extent = (0.0f32, 0.0f32);
    let mut points: Vec<Message> = Vec::new();
    for value in path.all(1) {
        let Value::Bytes(raw) = value else { return };
        let Some(element) = crate::pb::decode_nested(raw) else {
            return;
        };
        for point in element.all(2).filter_map(|v| match v {
            Value::Bytes(raw) => crate::pb::decode_nested(raw),
            _ => None,
        }) {
            extent.0 = extent.0.max(float_field(&point, 1));
            extent.1 = extent.1.max(float_field(&point, 2));
        }
        points.push(element);
    }
    // Not drawn in the frame's coordinates: leave it exactly as it is.
    let close = |a: f32, b: f32| (a - b).abs() <= (a.abs().max(b.abs()) * 1e-3).max(1e-3);
    if !close(extent.0, was.0) || !close(extent.1, was.1) {
        return;
    }
    for element in &mut points {
        let scaled: Vec<Value> = element
            .all(2)
            .filter_map(|v| match v {
                Value::Bytes(raw) => crate::pb::decode_nested(raw),
                _ => None,
            })
            .map(|mut point| {
                point.set_in_order(
                    1,
                    Value::Fixed32((float_field(&point, 1) * scale.0).to_le_bytes()),
                );
                point.set_in_order(
                    2,
                    Value::Fixed32((float_field(&point, 2) * scale.1).to_le_bytes()),
                );
                Value::Bytes(point.encode())
            })
            .collect();
        element.clear(2);
        for point in scaled {
            element.append_in_order(2, point);
        }
    }
    path.clear(1);
    for element in points {
        path.append_in_order(1, Value::Bytes(element.encode()));
    }
    body.set_in_order(field, Value::Bytes(path.encode()));
}

fn dotted(path: &[u32]) -> String {
    path.iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(".")
}

fn utf8(bytes: Option<&[u8]>) -> String {
    bytes
        .map(|b| String::from_utf8_lossy(b).into_owned())
        .unwrap_or_default()
}

/// The characters of `text` in a UTF-16 range, as far as the range is real.
fn slice(text: &str, range: Range<u64>) -> String {
    let from = crate::text::utf16_offset(text, range.start).unwrap_or(text.len());
    let to = crate::text::utf16_offset(text, range.end).unwrap_or(text.len());
    text.get(from..to).unwrap_or_default().to_string()
}

/// What the apps call a template they ship, derived from where it lives.
///
/// `.../Pages.app/Contents/SharedSupport/Templates/04_Real_Estate_Flyer/ISO.template`
/// is `Application/04_Real_Estate_Flyer/ISO`, which is the identifier
/// `pages-lists.pages` — a document Pages made from that very bundle — carries,
/// and the id AppleScript addresses templates by. Anything outside an app's own
/// `Templates` directory returns `None`: a template in the user's library has an
/// identifier too, and it is not this, so nothing is claimed about it.
fn template_identifier(path: &std::path::Path) -> Option<String> {
    let components: Vec<String> = path
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    let templates = components.iter().position(|c| c == "Templates")?;
    if templates < 3 {
        return None;
    }
    let inside_an_app = components[..templates]
        .iter()
        .rev()
        .take(3)
        .any(|c| c.ends_with(".app"));
    let shared = components[templates - 1] == "SharedSupport";
    if !inside_an_app || !shared {
        return None;
    }
    let rest = &components[templates + 1..];
    let (last, leading) = rest.split_last()?;
    let stem = last.rsplit_once('.').map(|(s, _)| s).unwrap_or(last);
    let mut parts = vec!["Application".to_string()];
    parts.extend(leading.iter().cloned());
    parts.push(stem.to_string());
    Some(parts.join("/"))
}

/// Identify the app from the object graph alone.
///
/// Used when there is no filename to go by, and it cannot lean on the root
/// archive's type alone: **Numbers and Keynote both put message type 1 at
/// object 1.** The app-level archives are numbered per app, so a `1` means
/// `TN.DocumentArchive` in one and `KN.DocumentArchive` in the other, and a
/// Keynote deck read by root type alone comes back as a spreadsheet.
///
/// The components are what separate them, and they are unambiguous: Numbers
/// spreads tables over `Index/Tables/`, Keynote writes a stream per slide and
/// per master. Pages is the one app whose root type is its own — 10000
/// `TP.DocumentArchive`.
fn detect_kind(package: &Package, streams: &BTreeMap<String, Vec<ArchiveObject>>) -> Kind {
    if package.names().any(|n| n.starts_with("Index/Tables/")) {
        return Kind::Numbers;
    }
    if package
        .names()
        .any(|n| n.starts_with("Index/Slide") || n.starts_with("Index/TemplateSlide"))
    {
        return Kind::Keynote;
    }
    let root_type = streams
        .get("Index/Document.iwa")
        .and_then(|objects| objects.iter().find(|o| o.identifier == 1))
        .map(ArchiveObject::message_type);
    match root_type {
        Some(10000) => Kind::Pages,
        // Ambiguous on its own, but a Keynote package has slide streams and has
        // already been caught above.
        Some(1) => Kind::Numbers,
        _ => Kind::Unknown,
    }
}

#[cfg(test)]
mod path_scaling_tests {
    use super::{float_field, scale_path_points};
    use crate::pb::{decode_nested, Message, Value};

    fn point(x: f32, y: f32) -> Message {
        let mut p = Message::default();
        p.set_in_order(1, Value::Fixed32(x.to_le_bytes()));
        p.set_in_order(2, Value::Fixed32(y.to_le_bytes()));
        p
    }

    fn element(kind: u64, points: &[(f32, f32)]) -> Value {
        let mut e = Message::default();
        e.set_in_order(1, Value::Varint(kind));
        for (x, y) in points {
            e.append_in_order(2, Value::Bytes(point(*x, *y).encode()));
        }
        Value::Bytes(e.encode())
    }

    fn curve_points(element: &Message) -> Vec<(f32, f32)> {
        element
            .all(2)
            .filter_map(|v| match v {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .map(|p| (float_field(&p, 1), float_field(&p, 2)))
            .collect()
    }

    /// A `curveTo` carries three control points in field 2. The old reader took
    /// only the first, so a scaled curve kept two points at the old scale and
    /// the outline deformed. Every point must move.
    #[test]
    fn a_curve_scales_all_three_of_its_control_points() {
        let mut path = Message::default();
        path.append_in_order(1, element(1, &[(0.0, 0.0)]));
        path.append_in_order(1, element(4, &[(50.0, 80.0), (100.0, 40.0), (100.0, 80.0)]));
        let mut body = Message::default();
        body.set_in_order(3, Value::Bytes(path.encode()));

        scale_path_points(&mut body, 3, (100.0, 80.0), (2.0, 0.5));

        let scaled = body.bytes(3).and_then(decode_nested).unwrap();
        let elements: Vec<Message> = scaled
            .all(1)
            .filter_map(|v| match v {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .collect();
        // The moveTo is untouched — (0, 0) scales to (0, 0).
        assert_eq!(curve_points(&elements[0]), vec![(0.0, 0.0)]);
        // The curveTo keeps all three points, and every one is scaled.
        assert_eq!(
            curve_points(&elements[1]),
            vec![(100.0, 40.0), (200.0, 20.0), (200.0, 40.0)]
        );
    }

    /// The extent is measured from every point too, so a curve whose control
    /// points fall outside the natural box does not fool the "is this drawn in
    /// the frame's coordinates" guard into leaving a real path unscaled.
    #[test]
    fn the_extent_sees_every_control_point() {
        let mut path = Message::default();
        // The endpoint is at 60, but a control point reaches 100 — the natural
        // width. Reading only the first point would have measured 60 and, since
        // 60 != 100, refused to scale.
        path.append_in_order(1, element(4, &[(100.0, 40.0), (80.0, 80.0), (60.0, 80.0)]));
        let mut body = Message::default();
        body.set_in_order(3, Value::Bytes(path.encode()));

        scale_path_points(&mut body, 3, (100.0, 80.0), (2.0, 1.0));

        let scaled = body.bytes(3).and_then(decode_nested).unwrap();
        let element = scaled
            .all(1)
            .filter_map(|v| match v {
                Value::Bytes(raw) => decode_nested(raw),
                _ => None,
            })
            .next()
            .unwrap();
        assert_eq!(
            curve_points(&element),
            vec![(200.0, 40.0), (160.0, 80.0), (120.0, 80.0)]
        );
    }
}
