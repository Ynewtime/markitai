//! Read and write Apple iWork documents — Pages, Numbers and Keynote.
//!
//! All three apps share one format, in four layers:
//!
//! 1. a package — a ZIP whose entries are all *stored*, or a directory holding
//!    the same entries ([`package`]);
//! 2. `Index/*.iwa` streams framed as raw Snappy blocks ([`iwa`]);
//! 3. a flat stream of length-delimited protobuf objects ([`iwa`], [`pb`]);
//! 4. an object graph whose shape depends on the app ([`document`], [`style`]).
//!
//! Apple does not publish the `.proto` definitions and the message type is the
//! only thing identifying a payload's schema, so this crate works at the
//! protobuf **wire level**: objects are decoded into fields and re-encoded in
//! place. That is enough to inspect a document and rewrite parts of it without
//! knowing every message definition, and it means an unrecognised object is
//! carried through untouched rather than lost.
//!
//! ```no_run
//! # fn main() -> Result<(), iwork::Error> {
//! use iwork::table::Format;
//!
//! let mut doc = iwork::Document::new_spreadsheet("Sales", "Q1", 4, 3)?;
//! let mut q1 = doc.table_mut("Q1")?;
//! q1.set_block("A1", &[vec!["Region", "Units"]])?;
//! q1.set("A2", "Zürich")?;
//! q1.set("B2", 1_240)?;
//! q1.set("A3", "Genève")?;
//! q1.set("B3", 980)?;
//! // A real formula: Numbers recalculates it when a figure above it changes.
//! // The value is what the cell shows until it does — nothing here evaluates.
//! q1.formula("B4", "=SUM(B2:B3)", 2_220)?;
//! q1.format("B4", &Format::Number { decimals: Some(0) })?;
//! doc.save("Sales.numbers")?;
//! # Ok(()) }
//! ```
//!
//! Pages and Keynote are addressed the same way — by the thing rather than by
//! its object identifier. [`Document::body_mut`] hands back the Pages body,
//! [`Document::text_mut`] any storage, and [`Document::slide_mut`] a slide by
//! its position in the deck or by identifier:
//!
//! ```no_run
//! # fn main() -> Result<(), iwork::Error> {
//! # let mut deck = iwork::Document::open("Talk.key")?;
//! let mut slide = deck.slide_mut(0)?;
//! slide.title("Quarterly review")?;
//! slide.notes("The numbers are provisional")?;
//! slide.transition("dissolve")?;
//! # Ok(()) }
//! ```
//!
//! **A slide is not a page with a title slot.** What a slide can hold is
//! decided by the *layout* it is built on: a title is a placeholder that layout
//! defines. A deck made from nothing has a layout that defines none, so
//! `title` refuses by name there and says why, rather than putting a text box
//! on the slide and calling it a title.
//!
//! **A sheet is not a grid.** This is the one place a Numbers document refuses
//! the shape a spreadsheet library usually assumes: a sheet is a *canvas*
//! holding any number of tables, with charts, shapes and images beside them.
//! [`Document::sheets`] reports the sheets and what is drawn on each;
//! [`Document::tables`] reports every table in the document whatever holds it,
//! which is also how a table on a Pages page or a Keynote slide is reached.
//! A table is named by a name that is unique, or by its sheet and name, or by
//! its identifier — `numbers-pivot.numbers` has a `Sales` on each of two
//! sheets, and a bare `"Sales"` is refused rather than guessed at.
//!
//! A document is *created* two ways. [`Document::from_template`] copies one
//! that works — a template bundle is a package like any other — and needs
//! Apple's software installed to have something to copy. [`Document::new`]
//! writes one from nothing, for any of the three apps, which is allowed here
//! only because it was measured: see [`create`].
//!
//! ```no_run
//! # fn main() -> Result<(), iwork::Error> {
//! let mut doc = iwork::Document::open("Report.pages")?;
//! println!("{} document", doc.kind().as_str());
//! for storage in doc.text_storages() {
//!     println!("{}: {}", storage.identifier, storage.text);
//! }
//! // Editing text remaps everything anchored into the storage — style runs,
//! // hyperlinks, list levels, anchored drawables ([`text`]). Indices are
//! // UTF-16 code units.
//! doc.insert_text(6083, 12, "eingeschoben ")?;
//! doc.delete_text(6083, 40..55)?;
//! doc.set_text(6083, "A new headline")?;
//!
//! // Text styles are shared objects a range of text points at ([`style`]).
//! for style in doc.text_styles() {
//!     println!("{} {} {:?}", style.identifier, style.kind.as_str(), style.name);
//! }
//! let kicker = doc.create_text_style(3712, "Kicker")?;
//! doc.set_text_style_property(
//!     kicker.identifier,
//!     iwork::style::property::FONT_SIZE,
//!     Some(iwork::pb::Value::Fixed32(18f32.to_le_bytes())),
//! )?;
//! doc.apply_text_style(6083, 0..8, kicker.identifier)?;
//!
//! doc.save("Report-edited.pages")?;
//! # Ok(()) }
//! ```
//!
//! # Where things are
//!
//! Start at [`Document`]: it opens a package, hands out handles, and saves.
//! Everything below it is grouped by what part of a document it is about, and
//! each module's own documentation is where the format is written down.
//!
//! **What a caller reaches for**
//!
//! | | |
//! |---|---|
//! | [`document`] | [`Document`] itself, and the handles: [`document::TableHandle`], [`document::TextHandle`], [`document::SlideHandle`] |
//! | [`table`] | cells, values, formats, ranges, sheets — [`table::CellValue`], [`table::CellRef`], [`table::CellRange`], [`table::Format`], [`table::Sheet`] |
//! | [`text`] | what an edit to a storage has to remap, and the rules it obeys |
//! | [`style`] | text styles, which are shared objects a range of text points at |
//! | [`keynote`] | the show: slides, layouts, transitions, builds, playback |
//! | [`pages`] | sections, headers and footers, page templates, the two document modes |
//! | [`chart`] | the private grid, and the mediator that makes a chart follow a table |
//! | [`drawable`] | anything drawn: shapes, images, tables, charts, their geometry and style |
//! | [`annotations`] | comments, their authors and anchors, and tracked changes |
//!
//! **What a document is made of**
//!
//! | | |
//! |---|---|
//! | [`package`] | the ZIP or directory a document *is* |
//! | [`iwa`] | the Snappy framing and the object stream inside it |
//! | [`pb`] | the protobuf wire level every archive is read at |
//! | [`plist`] | the metadata plists beside the object streams |
//! | [`metadata`] | identity, lineage, locale, template, build history |
//! | [`media`] | the media registry and its reference counts |
//! | [`registry`] | message type numbers, and the evidence for each |
//!
//! **The calculation engine**
//!
//! | | |
//! |---|---|
//! | [`formula`] | reading a formula: the AST, its node types, the reference model |
//! | [`formula_parse`] | the other direction — `=SUM(B2:B4)` to the nodes the app writes |
//! | [`calc`] | the dependency graph, and the owners without which nothing recalculates |
//! | [`create`] | documents from nothing: what the apps were measured needing |
//!
//! # What this crate will not do
//!
//! It does not lay a document out, render it, or evaluate a formula. A cell
//! given a formula carries the value the caller supplies until the app
//! recalculates; a chart draws the grid it was given. Where a write could be
//! made to *look* right without being right, it is refused instead — with a
//! [`Refusal`] saying which kind of wrong it would have been.

pub mod annotations;
pub mod calc;
pub mod chart;
pub mod create;
pub mod document;
pub mod drawable;
pub mod formula;
pub mod formula_parse;
pub mod iwa;
pub mod keynote;
pub mod media;
pub mod metadata;
pub mod package;
pub mod pages;
pub mod pb;
pub mod plist;
pub mod registry;
pub mod style;
pub mod table;
pub mod text;

pub use annotations::{Annotations, Author, Change, Comment};
pub use chart::{Chart, DataReferences, Grid, GridValue, Series};
pub use document::{Component, DataFile, Document, Kind, TextEdit, TextStorage};
pub use drawable::{Drawable, Geometry, Placement};
pub use formula::{Ast, Formula, Node, Reference};
pub use keynote::{Layout, Placeholder, Show, Slide, SlideCopy, Transition};
pub use media::MediaReplacement;
pub use package::{Form, Package};
pub use style::{CreatedStyle, Label, StyleDeletion, StyleKind, StyleUse, TextStyle};
pub use table::{Cell, CellControl, CellFormat, CellValue, Merge, Table};

/// `TSWP.StorageArchive` — a run of styled text. Same in all three apps.
pub const TYPE_STORAGE: u32 = 2001;
/// `TSP.PackageMetadata` — the component and media index.
pub const TYPE_PACKAGE_METADATA: u32 = 11006;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Zip(zip::result::ZipError),
    /// The package parsed as a ZIP but its contents are not a document we
    /// understand. Carries a human-readable reason.
    Format(String),
    NoSuchObject(u64),
    NoSuchStyle(u64),
    /// A style could not be deleted because references to it would have been
    /// left dangling. Carries the objects that still refer to it.
    StyleInUse {
        identifier: u64,
        references: Vec<u64>,
    },
    /// An image's bytes were not replaced, because something between the stored
    /// pixels and the render was computed from the old ones — a crop, a shaped
    /// mask, an Instant Alpha path, tone adjustments, a cached rendering, a
    /// traced outline. Swapping the bytes under any of those leaves a document
    /// that opens, reports the same geometry, and draws the wrong thing.
    NonDestructiveEdit {
        drawable: u64,
        reasons: Vec<String>,
    },
    /// A character index was outside the storage's text.
    TextRange {
        storage: u64,
        index: u64,
        length: u64,
    },
    /// A character index landed between the halves of a surrogate pair —
    /// inside an emoji, in other words. Indices count UTF-16 code units, and an
    /// edit at half a character would leave two unpaired surrogates.
    SplitSurrogate {
        storage: u64,
        index: u64,
    },
    /// The deleted range covered the character an object is anchored to: the
    /// `U+FFFC` an image stands in, the `U+000E` a footnote mark stands on, the
    /// `U+0004` a section begins after. Deleting it means deleting the object —
    /// Pages does exactly that, removing the drawable, its mask, its z-order
    /// entry and its media registration — and this crate will not, so it
    /// refuses.
    AnchoredObject {
        storage: u64,
        index: u64,
        table: &'static str,
        object: Option<u64>,
    },
    /// The deleted range covered the `U+0004` a section begins after, which
    /// would merge two sections into one. Which of the two keeps its page
    /// templates, headers, footers, background and guides is not something any
    /// probe here could establish — Pages refuses to delete a section from a
    /// script — so this crate refuses rather than choose.
    SectionBreak {
        storage: u64,
        /// Character index of the break itself.
        index: u64,
        /// The `TP.SectionArchive` that begins after it.
        section: Option<u64>,
    },
    /// The storage carries a length-delimited field that is not one of the
    /// attribute tables this crate knows. It may well be one, and remapping it
    /// by guesswork is how an edit silently damages a document.
    UnknownAttributeTable {
        storage: u64,
        field: u32,
    },
    /// The storage carries an attribute table this crate knows by number whose
    /// bytes it cannot decode and re-encode unchanged. Every *other* table
    /// would be remapped and this one skipped, leaving it pointing into text
    /// that has moved — a quieter corruption than refusing.
    UndecodableAttributeTable {
        storage: u64,
        field: u32,
    },
    /// The storage's text is not valid UTF-8. Reading it is lossy and writing
    /// the lossy reading back would replace every ill-formed sequence with
    /// `U+FFFD` and shift every index after it, so an edit is refused.
    InvalidText {
        storage: u64,
    },
    /// The storage carries `table_insertion` (21) or `table_deletion` (22):
    /// change tracking is on and there are changes in this text. A tracked
    /// deletion leaves its characters *in* the storage, so an edit through one
    /// is not the run remap it looks like — see [`annotations`]. Nothing here
    /// can make Pages produce one to watch, so this crate declines.
    TrackedChanges {
        storage: u64,
        field: u32,
    },
    /// Text to be written contains a character that only means something with
    /// an object behind it — see [`text::UNWRITABLE`].
    UnwritableCharacter {
        character: char,
    },
    /// The package is password-protected. Its object streams and its media are
    /// ciphertext; this crate does not decrypt and will not write one.
    Encrypted {
        /// The password hint, from `.iwph`, when the document carries one.
        hint: Option<String>,
    },
    /// A write this crate declined, with **why** in a form a program can act
    /// on.
    ///
    /// Refusing precisely is what this crate does instead of guessing, and
    /// until now the reason was prose: a caller wanting to skip merge-covered
    /// cells, grow a table that was too small, or stop at a formula had to
    /// match on the text of a message. [`Refusal`] is that reason as a value;
    /// `detail` is the sentence, unchanged, and is what `Display` prints.
    Refused {
        reason: Refusal,
        detail: String,
    },
}

/// Why a write was declined — see [`Error::Refused`].
///
/// Each of these is a case where writing *something* was possible and only one
/// of the possibilities was right. They are grouped by what a caller can do
/// about them: change the request ([`Refusal::OutOfBounds`],
/// [`Refusal::NotACell`], [`Refusal::Ambiguous`]), write something else first
/// ([`Refusal::NoDonorFormat`], [`Refusal::WrongSlot`]), or stop
/// ([`Refusal::HoldsFormula`], [`Refusal::Organised`], [`Refusal::Patched`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The row, column or cell is outside the table.
    OutOfBounds,
    /// The name given is not a cell reference, a range, or a thing that exists.
    NotACell,
    /// A name that matches more than one table — a sheet holds any number of
    /// them, and two sheets may hold one name.
    Ambiguous,
    /// The cell is covered by a merge that begins somewhere else.
    Merged,
    /// The cell holds a formula, and taking one out means editing the
    /// calculation engine.
    HoldsFormula,
    /// The cell's text is a `TSWP` storage rather than a table string.
    HoldsRichText,
    /// The value's kind is not one this crate writes.
    UnwritableValue,
    /// The table carries no format of the kind being written that could be
    /// copied, and inventing one would be a format the document never defined.
    NoDonorFormat,
    /// The format's slot is not the one the cell's value uses, so the app would
    /// never draw it.
    WrongSlot,
    /// The table is categorised, filtered, pivoted, conditionally highlighted,
    /// or has hidden or collapsed rows — organisation this crate cannot
    /// maintain across the edit.
    Organised,
    /// A formula somewhere names what the edit would move or remove.
    FormulaReference,
    /// The object carries version patches, which would go on describing it as
    /// it used to be.
    Patched,
    /// The document has the thing but the app never draws it — a page-layout
    /// document's body.
    NotDrawn,
    /// The document, table, slide or storage named is not there.
    NotFound,
    /// The structure a write would need is missing and cannot be invented: a
    /// layout with no placeholder, a table with no formula list, a deck with no
    /// notes.
    Missing,
}

impl Error {
    /// A refusal with its reason and its sentence.
    pub fn refused(reason: Refusal, detail: impl Into<String>) -> Error {
        Error::Refused {
            reason,
            detail: detail.into(),
        }
    }

    /// Why this write was declined, when it was declined for a reason a
    /// program can act on.
    ///
    /// ```no_run
    /// # fn main() -> Result<(), iwork::Error> {
    /// # let mut doc = iwork::Document::open("Budget.numbers")?;
    /// use iwork::Refusal;
    /// let mut table = doc.table_mut("Q1")?;
    /// match table.set("B3", 42) {
    ///     Err(e) if e.refusal() == Some(Refusal::Merged) => {} // skip it
    ///     Err(e) if e.refusal() == Some(Refusal::OutOfBounds) => {} // grow first
    ///     other => { other?; }
    /// }
    /// # Ok(()) }
    /// ```
    pub fn refusal(&self) -> Option<Refusal> {
        match self {
            Error::Refused { reason, .. } => Some(*reason),
            _ => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "{e}"),
            Error::Zip(e) => write!(f, "{e}"),
            Error::Format(m) => write!(f, "{m}"),
            // The sentence, and only the sentence: a refusal reads exactly as
            // it did before it had a reason attached.
            Error::Refused { detail, .. } => write!(f, "{detail}"),
            Error::NoSuchObject(id) => write!(f, "no object with identifier {id}"),
            Error::NoSuchStyle(id) => write!(f, "no text style with identifier {id}"),
            Error::NonDestructiveEdit { drawable, reasons } => write!(
                f,
                "drawable {drawable} carries edit state a replacement would falsify: {}",
                reasons.join("; ")
            ),
            Error::TextRange {
                storage,
                index,
                length,
            } => write!(
                f,
                "storage {storage}: character {index} is outside its text, which is \
                 {length} UTF-16 code unit(s) long"
            ),
            Error::SplitSurrogate { storage, index } => write!(
                f,
                "storage {storage}: character {index} is the second half of a surrogate \
                 pair — indices count UTF-16 code units and an edit may not split one"
            ),
            Error::AnchoredObject {
                storage,
                index,
                table,
                object,
            } => write!(
                f,
                "storage {storage}: character {index} anchors {} in {table}, and deleting \
                 it means deleting that object from the whole document — which this crate \
                 does not do",
                match object {
                    Some(id) => format!("object {id}"),
                    None => "something".to_string(),
                }
            ),
            Error::SectionBreak {
                storage,
                index,
                section,
            } => write!(
                f,
                "storage {storage}: character {index} is the U+0004 that begins {}, \
                 and deleting it merges two sections — which of the two keeps its \
                 page templates, headers, footers, background and guides is not \
                 known, because Pages will not perform the edit for anyone to watch",
                match section {
                    Some(id) => format!("section {id}"),
                    None => "a section".to_string(),
                }
            ),
            Error::UnknownAttributeTable { storage, field } => write!(
                f,
                "storage {storage}: field {field} is not an attribute table this crate \
                 knows, and an edit would have to guess how its entries are anchored"
            ),
            Error::UndecodableAttributeTable { storage, field } => write!(
                f,
                "storage {storage}: field {field} ({}) does not decode as an attribute \
                 table, so an edit would remap every other table and leave this one \
                 pointing at characters that have moved",
                text::table(*field)
                    .map(|t| t.name)
                    .unwrap_or("an attribute table")
            ),
            Error::InvalidText { storage } => write!(
                f,
                "storage {storage}: the text is not valid UTF-8, and an edit would write \
                 back a lossy reading of it — replacing every ill-formed sequence with \
                 U+FFFD and moving every index after it"
            ),
            Error::TrackedChanges { storage, field } => write!(
                f,
                "storage {storage} carries {} — change tracking is on and this text has \
                 tracked changes in it. A tracked deletion keeps its characters, so an \
                 edit through one is not a plain remap, and nothing available here can \
                 make the app perform one to be watched",
                text::table(*field)
                    .map(|t| t.name)
                    .unwrap_or("a change table")
            ),
            Error::Encrypted { hint } => write!(
                f,
                "the document is password-protected ({}); its object streams are \
                 ciphertext and this crate does not decrypt",
                match hint {
                    Some(hint) => format!("hint: {hint}"),
                    None => "no hint".to_string(),
                }
            ),
            Error::UnwritableCharacter { character } => write!(
                f,
                "U+{:04X} stands for an object rather than for itself, and this crate \
                 will not write one into text",
                *character as u32
            ),
            Error::StyleInUse {
                identifier,
                references,
            } => {
                let list: Vec<String> = references.iter().map(u64::to_string).collect();
                write!(
                    f,
                    "text style {identifier} is still referenced by {} object(s): {}",
                    references.len(),
                    list.join(", ")
                )
            }
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Error {
        Error::Io(e)
    }
}

impl From<zip::result::ZipError> for Error {
    fn from(e: zip::result::ZipError) -> Error {
        Error::Zip(e)
    }
}
