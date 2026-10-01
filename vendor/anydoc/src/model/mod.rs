//! Information-preserving document model.
//!
//! Only fully resolved content lives here: format frontends resolve style
//! cascades, numbering, and references before constructing these types.
//! A [`Document`] is self-contained - embedded assets carry their bytes, so it
//! stays usable after the source archive is gone.

mod asset;
mod block;
mod inline;
mod link;
mod list;
mod style;
mod table;

pub use asset::{Asset, AssetId};
pub use block::Block;
pub use inline::{Inline, checkbox_text, inlines_are_empty, inlines_to_plain_text};
pub use link::{AnchorId, ImageSource, LinkTarget};
pub use list::{List, ListItem, MarkerKind};
pub use style::Style;
pub use table::{Cell, CellSlot, Table, TableKind};

/// Frontends build grids; consumers read them off [`Table::grid`].
pub(crate) use table::GridBuilder;

/// A parsed document: its body, its notes, and the bytes of everything it
/// embedded.
#[derive(Debug, Clone, Default)]
pub struct Document {
    /// Body content in reading order.
    pub blocks: Vec<Block>,
    /// Note bodies, in the order the document defines them. Text refers to
    /// them by id through [`Inline::NoteRef`].
    pub notes: Vec<Note>,
    /// Every embedded asset, indexed by [`AssetId`].
    pub assets: Vec<Asset>,
    // markitai: presentations carry their slide boundaries; upstream joined
    // every slide into one run of blocks with no page edges.
    /// For a presentation whose slide boundaries are known (ODP, PPT), the
    /// index in [`Document::blocks`] where each slide begins, one entry per
    /// slide in slide order. A slide with no blocks repeats the index of the
    /// next slide, or `blocks.len()` when it is last. Empty for every other
    /// format, and for a legacy PPT read in raw stream order, where slides
    /// cannot be told apart.
    pub slide_starts: Vec<usize>,
    // markitai: what a reader had to leave out and a reader of the output
    // should know about.
    /// Content the reader left out of `blocks` on purpose, one sentence
    /// each (a Word document's embedded part in a format it does not read).
    /// Empty when nothing was left out that way.
    pub warnings: Vec<String>,
}

/// Footnote or endnote body, referenced from text by [`Inline::NoteRef`].
#[derive(Debug, Clone)]
pub struct Note {
    /// Document-scoped id the referencing [`Inline::NoteRef`] carries.
    pub id: String,
    /// Whether the source placed this note on the page or at the end.
    pub kind: NoteKind,
    /// The note's own content.
    pub blocks: Vec<Block>,
}

/// Where the source document places a note.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoteKind {
    /// Placed at the foot of the page that references it.
    Footnote,
    /// Collected at the end of the document or section.
    Endnote,
}
