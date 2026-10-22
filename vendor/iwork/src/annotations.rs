//! Comments, annotation authors and change tracking — the review layer.
//!
//! Every iWork document carries an `Index/AnnotationAuthorStorage*.iwa` with
//! exactly one `TSK.AnnotationAuthorStorageArchive` in it, and that object is
//! the anchor of everything here: a comment has an author, a tracked change has
//! a session and a session has an author, and all of them are entries in that
//! one storage.
//!
//! ```text
//!  TSK.AnnotationAuthorStorageArchive (213)   Index/AnnotationAuthorStorage*
//!    └── annotation_author (1) → TSK.AnnotationAuthorArchive (212)
//!                                  name, colour, collaboration public id
//!
//!  TSD.CommentStorageArchive (3056)   the comment itself
//!    ├── text (1), creation_date (2), author (3) → 212
//!    └── replies (4) → more 3056, in order
//!
//!  reached from three places, one per thing a comment can be put on:
//!    text     TSWP.StorageArchive.table_highlight (23) or
//!             table_overlapping_highlight (25) → TSWP.HighlightArchive (2013)
//!             → comment storage; the bubble on the page is a
//!             TSWP.CommentInfoArchive (2014), a shape with the same storage
//!    object   TSD.DrawableArchive.comment (6) → comment storage
//!    cell     TST.CommentStorageWrapperArchive → comment storage
//!
//!  TSWP.ChangeArchive (2060)   one tracked insertion or deletion
//!    ├── kind (1): 1 = insertion, 2 = deletion — there is no zero
//!    ├── session (2) → TSWP.ChangeSessionArchive (2062) → author, date
//!    └── reached from table_insertion (21) / table_deletion (22)
//! ```
//!
//! ## What of this has been seen
//!
//! For a long time: the storage and nothing else — no scripting dictionary
//! has a comment command or a change-tracking property, and templates ship
//! without review state, so every payload was a zero-byte type-213 and the
//! decoders below were schema-only. Then the screen was unlocked, the Insert
//! and Edit menus became drivable, and `pages-comments.pages` and
//! `pages-tracked.pages` gave the decoders their first live examples
//! (`scripts/applescript/*-ui.applescript` holds the recipes).
//!
//! What the fixtures settled, against the schema guesses:
//!
//! - **The anchor tables are one level deeper than the schema reads.** A
//!   storage's field 21/22/23/25 is the *table* — a wrapper whose repeated
//!   field 1 holds the `{character_index, reference}` entries, exactly like
//!   every other attribute table. Decoding the field as though it were an
//!   entry finds nothing, and every comment reports unattached; that is the
//!   bug the first fixture exposed here.
//! - A comment (3056) is `{1 text, 2.1 f64 creation date, 3 → author,
//!   5 {1,2} a two-u64 UUID}`; its text anchor is a `TSWP.HighlightArchive`
//!   (2013) of `{1 → comment, 2 UUID string}`, pointed at by the run table.
//! - A tracked change (2060) is reached from `table_insertion` /
//!   `table_deletion` the same way, and the author storage finally holds a
//!   real `TSK.AnnotationAuthorArchive` — name and colour, as documented.
//!
//! Still never seen, and still schema-only: replies, a resolved state (no
//! descriptor anywhere spells `resolv`), cell comments, the overlapping
//! highlight table (25), deprecated change authors (2061), and what an
//! accepted or rejected change leaves behind. The reader keeps reporting
//! rather than pretending, and the remaining tripwires still guard those.
//!
//! ## What writes here
//!
//! [`add_comment`] attaches a comment to a range of text: the comment, its
//! `TSWP.HighlightArchive`, an author in the document's one author storage if
//! that name is not there yet, and two entries in the run-anchored
//! `table_highlight` — the anchor at the start, and a *bare index* at the end,
//! which is where the run stops rather than a comment on nothing. The table
//! also has to start at 0, whatever the comment does. Nothing else here writes:
//! a reply, a resolved state and a cell comment are all still schema-only, and
//! a tracked change is refused for the reason below.
//!
//! ## Tracked deletions keep their characters
//!
//! The one claim here that changes what an *edit* may do. A deletion under
//! change tracking does not remove the characters; it adds an entry to
//! `table_deletion` covering them, and Pages draws them struck through. So the
//! text of a storage with tracked changes contains text nobody is going to see,
//! and a remap that treats field 22 as an ordinary run table will move a
//! deletion marker onto characters nobody deleted. Since no probe here can
//! watch Pages do it, [`crate::Document::replace_text`] refuses instead —
//! [`crate::Error::TrackedChanges`].

use std::collections::BTreeMap;

use crate::drawable::Color;
use crate::pb::{decode_nested, Message, Value};
use crate::style::reference_target;

/// `TSK.AnnotationAuthorArchive` — a comment or change author.
pub const TYPE_AUTHOR: u32 = 212;
/// `TSK.AnnotationAuthorStorageArchive` — the document's list of them.
pub const TYPE_AUTHOR_STORAGE: u32 = 213;
/// `TSD.CommentStorageArchive` — a comment, or one reply to one.
pub const TYPE_COMMENT_STORAGE: u32 = 3056;
/// `TSWP.CommentInfoArchive` — the floating bubble a text comment is drawn as.
pub const TYPE_COMMENT_INFO: u32 = 2014;
/// `TSWP.HighlightArchive` — a comment's anchor in text.
pub const TYPE_HIGHLIGHT: u32 = 2013;
/// `TSWP.ChangeArchive` — one tracked insertion or deletion.
pub const TYPE_CHANGE: u32 = 2060;
/// `TSK.DeprecatedChangeAuthorArchive` — a pre-collaboration change author.
pub const TYPE_DEPRECATED_CHANGE_AUTHOR: u32 = 2061;
/// `TSWP.ChangeSessionArchive` — an editing session, with its author and date.
pub const TYPE_CHANGE_SESSION: u32 = 2062;
/// `TST.TableDataList.ListType.COMMENT_STORAGE` — the list a cell's comment is
/// an entry of.
pub const CELL_COMMENT_LIST: u32 = 10;

/// Field numbers of `TSK.AnnotationAuthorArchive`.
pub mod author_field {
    pub const NAME: u32 = 1;
    pub const COLOR: u32 = 2;
    pub const PUBLIC_ID: u32 = 3;
    pub const IS_PUBLIC_AUTHOR: u32 = 4;
    pub const PUBLIC_IDS: u32 = 5;
}

/// Field numbers of `TSD.CommentStorageArchive`.
pub mod comment_field {
    pub const TEXT: u32 = 1;
    pub const CREATION_DATE: u32 = 2;
    pub const AUTHOR: u32 = 3;
    pub const REPLIES: u32 = 4;
    pub const STORAGE_UUID: u32 = 5;
}

/// Field numbers of `TSWP.ChangeArchive`.
pub mod change_field {
    pub const KIND: u32 = 1;
    pub const SESSION: u32 = 2;
    pub const DATE: u32 = 3;
    pub const TEXT_ATTRIBUTE_UUID: u32 = 4;
}

/// Field numbers of `TSWP.ChangeSessionArchive`.
pub mod session_field {
    pub const SESSION_UID: u32 = 1;
    pub const AUTHOR: u32 = 2;
    pub const DATE: u32 = 3;
}

/// One entry of the document's author storage.
#[derive(Debug, Clone)]
pub struct Author {
    pub identifier: u64,
    pub name: Option<String>,
    /// The colour the app tints this author's comments and changes with.
    pub color: Option<Color>,
    /// The collaboration identity, when the document has been shared.
    pub public_id: Option<String>,
    pub is_public_author: bool,
}

/// What a comment is attached to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Anchor {
    /// A range of text: an entry in `table_highlight` (23) or
    /// `table_overlapping_highlight` (25) of this storage.
    Text {
        storage: u64,
        /// Field number of the table the entry came from — 23 or 25.
        table: u32,
        /// First character covered, in UTF-16 code units.
        start: u64,
        /// How many characters, where the table says so. `table_highlight` is a
        /// run table and does not: its entry reaches to the next one.
        length: Option<u64>,
    },
    /// An image, a shape, a table or a chart: `TSD.DrawableArchive.comment`.
    Drawable(u64),
    /// A cell of a table, through `TST.CommentStorageWrapperArchive`.
    Cell(u64),
    /// A `TSWP.CommentInfoArchive` — the bubble, whose own anchor is the
    /// highlight in the text it points at.
    Bubble(u64),
    /// A reply: it hangs off another comment, not off the document.
    ReplyTo(u64),
    /// The comment is in the document and nothing this crate walks points at
    /// it. Reported rather than dropped.
    Unattached,
}

impl Anchor {
    pub fn as_str(&self) -> String {
        match self {
            Anchor::Text {
                storage,
                table,
                start,
                length,
            } => {
                let name = crate::text::table(*table).map(|t| t.name).unwrap_or("?");
                match length {
                    Some(n) => format!("storage {storage} {start}..{} ({name})", start + n),
                    None => format!("storage {storage} from {start} ({name})"),
                }
            }
            Anchor::Drawable(id) => format!("drawable {id}"),
            Anchor::Cell(id) => format!("cell storage {id}"),
            Anchor::Bubble(id) => format!("comment shape {id}"),
            Anchor::ReplyTo(id) => format!("reply to comment {id}"),
            Anchor::Unattached => "unattached".to_string(),
        }
    }
}

/// One comment or one reply.
#[derive(Debug, Clone)]
pub struct Comment {
    pub identifier: u64,
    pub stream: String,
    pub text: Option<String>,
    /// Seconds since 2001-01-01, as `TSP.Date` counts them.
    pub created: Option<f64>,
    /// The `TSK.AnnotationAuthorArchive` this belongs to.
    pub author: Option<u64>,
    /// Replies, in the order the archive lists them.
    pub replies: Vec<u64>,
    pub anchor: Anchor,
}

/// One tracked insertion or deletion.
#[derive(Debug, Clone)]
pub struct Change {
    pub identifier: u64,
    pub stream: String,
    pub kind: ChangeKind,
    pub session: Option<u64>,
    pub created: Option<f64>,
    /// The storage and character index the change covers, where a table points
    /// at it.
    pub anchor: Option<(u64, u32, u64)>,
}

/// `TSWP.ChangeArchive.ChangeKind`.
///
/// **There is no zero value**: insertion is 1 and deletion is 2. An archive
/// whose kind field is absent or zero is therefore not an insertion — it is an
/// archive this crate does not understand, which is what [`ChangeKind::Unknown`]
/// says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeKind {
    Insertion,
    Deletion,
    Unknown(u64),
}

impl ChangeKind {
    pub fn from_varint(value: Option<u64>) -> ChangeKind {
        match value {
            Some(1) => ChangeKind::Insertion,
            Some(2) => ChangeKind::Deletion,
            other => ChangeKind::Unknown(other.unwrap_or(0)),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ChangeKind::Insertion => "insertion",
            ChangeKind::Deletion => "deletion",
            ChangeKind::Unknown(_) => "unknown kind",
        }
    }
}

/// Everything the review layer of one document holds.
///
/// In every document this crate has been shown, `authors` is empty and so is
/// everything else; `author_storage` is the one object that is always there.
#[derive(Debug, Clone, Default)]
pub struct Annotations {
    /// The `TSK.AnnotationAuthorStorageArchive`, and the stream it lives in.
    pub author_storage: Option<(u64, String)>,
    pub authors: Vec<Author>,
    pub comments: Vec<Comment>,
    pub changes: Vec<Change>,
    /// `TSWP.ChangeSessionArchive` objects, by identifier.
    pub sessions: Vec<u64>,
    /// Storages carrying `table_insertion` (21) or `table_deletion` (22): the
    /// text an edit must not be allowed through.
    pub tracked_storages: Vec<u64>,
    /// Storages carrying `table_highlight` (23) or
    /// `table_overlapping_highlight` (25).
    pub commented_storages: Vec<u64>,
    /// Objects of a type this module knows about but did not reach from
    /// anywhere — reported so a document with an unexpected shape says so
    /// rather than looking empty.
    pub unreached: Vec<(u64, u32)>,
}

impl Annotations {
    /// Nothing to report: no comment, no tracked change, no author.
    pub fn is_empty(&self) -> bool {
        self.authors.is_empty()
            && self.comments.is_empty()
            && self.changes.is_empty()
            && self.tracked_storages.is_empty()
            && self.commented_storages.is_empty()
    }

    /// One line for a reader — what `iwork inspect` prints.
    pub fn summary(&self) -> String {
        if self.is_empty() {
            return "no comments, no tracked changes, no annotation authors".to_string();
        }
        let mut parts = Vec::new();
        if !self.authors.is_empty() {
            parts.push(format!("{} annotation author(s)", self.authors.len()));
        }
        if !self.comments.is_empty() {
            parts.push(format!("{} comment archive(s)", self.comments.len()));
        }
        if !self.changes.is_empty() {
            parts.push(format!("{} tracked change(s)", self.changes.len()));
        }
        if !self.tracked_storages.is_empty() {
            parts.push(format!(
                "{} storage(s) with tracked changes",
                self.tracked_storages.len()
            ));
        }
        if !self.commented_storages.is_empty() {
            parts.push(format!(
                "{} storage(s) with comment anchors",
                self.commented_storages.len()
            ));
        }
        parts.join(", ")
    }
}

/// Read the review layer of a document.
///
/// One pass over `objects()` into a map, then a walk — the shape
/// [`crate::Document::structure`] established. It costs a decode of the whole
/// document and is fine at this scale.
pub fn annotations(document: &crate::Document) -> Annotations {
    let mut archives: BTreeMap<u64, (u32, String, Message)> = BTreeMap::new();
    for (stream, object) in document.objects() {
        if let Ok(message) = Message::decode(object.payload()) {
            archives.insert(
                object.identifier,
                (object.message_type(), stream.to_string(), message),
            );
        }
    }

    let mut out = Annotations::default();

    // The author storage, and the authors it lists. Field 1 is repeated, so a
    // document with several authors has several entries and each is a
    // reference.
    if let Some((identifier, (_, stream, storage))) = archives
        .iter()
        .find(|(_, (message_type, _, _))| *message_type == TYPE_AUTHOR_STORAGE)
        .map(|(id, entry)| (*id, entry.clone()))
    {
        out.author_storage = Some((identifier, stream));
        for target in references(&storage, 1) {
            if let Some((message_type, _, author)) = archives.get(&target) {
                if *message_type == TYPE_AUTHOR {
                    out.authors.push(read_author(&target, author));
                }
            }
        }
    }
    // An author the storage does not list is still an author, and a reader that
    // only walks the storage would miss it.
    for (identifier, (message_type, _, author)) in &archives {
        if *message_type == TYPE_AUTHOR && !out.authors.iter().any(|a| a.identifier == *identifier)
        {
            out.authors.push(read_author(identifier, author));
        }
    }

    // Where a comment is anchored, gathered before the comments themselves so
    // each one can be told what points at it.
    let mut anchors: BTreeMap<u64, Anchor> = BTreeMap::new();
    // Highlights the anchor walk passes through: pointed at by a table entry,
    // pointing at a comment — reached, not orphaned.
    let mut reached_highlights: std::collections::BTreeSet<u64> = Default::default();
    for (identifier, (message_type, _, archive)) in &archives {
        if *message_type == crate::TYPE_STORAGE {
            for field in [
                crate::text::HIGHLIGHT_TABLE,
                crate::text::OVERLAPPING_HIGHLIGHT_TABLE,
            ] {
                let found = highlight_anchors(
                    archive,
                    field,
                    *identifier,
                    &archives,
                    &mut reached_highlights,
                );
                if !found.is_empty() {
                    out.commented_storages.push(*identifier);
                    for (comment, anchor) in found {
                        anchors.entry(comment).or_insert(anchor);
                    }
                }
            }
            if archive.get(crate::text::INSERTION_TABLE).is_some()
                || archive.get(crate::text::DELETION_TABLE).is_some()
            {
                out.tracked_storages.push(*identifier);
            }
        }
        // An object comment: `TSD.DrawableArchive.comment`, wherever the
        // drawable archive sits inside the payload.
        for target in comment_references(archive) {
            anchors
                .entry(target)
                .or_insert(Anchor::Drawable(*identifier));
        }
        if *message_type == TYPE_COMMENT_INFO {
            if let Some(target) = reference(archive, 2) {
                anchors.entry(target).or_insert(Anchor::Bubble(*identifier));
            }
        }
        // A comment on a *cell* is an entry in a `TST.TableDataList` of type 10
        // — the same string-table indirection every other cell payload uses.
        // The list is the anchor this crate can name; which cell holds the key
        // needs a tile walk, and there is no list of this type anywhere to walk
        // one for.
        if *message_type == crate::table::TYPE_DATA_LIST {
            let list = crate::table::DataList::decode(archive);
            if list.list_type == CELL_COMMENT_LIST {
                for entry in list.entries.values() {
                    if let Some(target) = entry.comment_storage {
                        anchors.entry(target).or_insert(Anchor::Cell(*identifier));
                    }
                }
            }
        }
    }
    out.commented_storages.sort_unstable();
    out.commented_storages.dedup();
    out.tracked_storages.sort_unstable();
    out.tracked_storages.dedup();

    // Replies win over any other anchor: a reply belongs to its parent.
    for (identifier, (message_type, _, archive)) in &archives {
        if *message_type != TYPE_COMMENT_STORAGE {
            continue;
        }
        for reply in references(archive, comment_field::REPLIES) {
            anchors.insert(reply, Anchor::ReplyTo(*identifier));
        }
    }

    for (identifier, (message_type, stream, archive)) in &archives {
        match *message_type {
            TYPE_COMMENT_STORAGE => out.comments.push(Comment {
                identifier: *identifier,
                stream: stream.clone(),
                text: crate::style::string_at(archive, &[comment_field::TEXT]),
                created: date(archive, comment_field::CREATION_DATE),
                author: reference(archive, comment_field::AUTHOR),
                replies: references(archive, comment_field::REPLIES),
                anchor: anchors
                    .get(identifier)
                    .cloned()
                    .unwrap_or(Anchor::Unattached),
            }),
            TYPE_CHANGE => out.changes.push(Change {
                identifier: *identifier,
                stream: stream.clone(),
                kind: ChangeKind::from_varint(archive.varint(change_field::KIND)),
                session: reference(archive, change_field::SESSION),
                created: date(archive, change_field::DATE),
                anchor: None,
            }),
            TYPE_CHANGE_SESSION => out.sessions.push(*identifier),
            TYPE_HIGHLIGHT if reached_highlights.contains(identifier) => {}
            TYPE_HIGHLIGHT | TYPE_COMMENT_INFO | TYPE_DEPRECATED_CHANGE_AUTHOR => {
                out.unreached.push((*identifier, *message_type))
            }
            _ => {}
        }
    }
    out.comments.sort_by_key(|c| c.identifier);
    out.changes.sort_by_key(|c| c.identifier);

    // Now that the changes are known, say where each one is anchored.
    let placements = change_anchors(&archives);
    for change in &mut out.changes {
        change.anchor = placements.get(&change.identifier).copied();
    }
    out
}

fn read_author(identifier: &u64, archive: &Message) -> Author {
    Author {
        identifier: *identifier,
        name: crate::style::string_at(archive, &[author_field::NAME]),
        color: archive
            .bytes(author_field::COLOR)
            .and_then(crate::pb::decode_nested)
            .and_then(|m| Color::decode(&m)),
        public_id: crate::style::string_at(archive, &[author_field::PUBLIC_ID]),
        is_public_author: archive.varint(author_field::IS_PUBLIC_AUTHOR) == Some(1),
    }
}

/// Comment storages a highlight table points at, with where the entry sits.
///
/// `table_highlight` (23) is a run table — `{character_index, reference}` — and
/// `table_overlapping_highlight` (25) carries an explicit `TSP.Range` so that
/// two comments may cover the same characters. Both hold their reference
/// through a `TSWP.HighlightArchive`, whose field 1 is the comment storage.
fn highlight_anchors(
    storage: &Message,
    field: u32,
    identifier: u64,
    archives: &BTreeMap<u64, (u32, String, Message)>,
    reached_highlights: &mut std::collections::BTreeSet<u64>,
) -> Vec<(u64, Anchor)> {
    let mut out = Vec::new();
    // The storage's field is the table itself — one wrapper message whose
    // repeated field 1 holds the `{character_index, reference}` entries, the
    // same shape as every other attribute table. `pages-comments.pages` is
    // what settled this: the schema alone read as though the entry sat
    // directly on the storage, one level too shallow, and every comment
    // reported as unattached.
    let entries = storage
        .fields
        .iter()
        .filter(|f| f.number == field)
        .filter_map(|f| match &f.value {
            Value::Bytes(raw) => crate::pb::decode_nested(raw),
            _ => None,
        })
        .flat_map(|table| {
            table
                .fields
                .iter()
                .filter(|f| f.number == 1)
                .filter_map(|f| match &f.value {
                    Value::Bytes(raw) => crate::pb::decode_nested(raw),
                    _ => None,
                })
                .collect::<Vec<_>>()
        });
    for entry in entries {
        let start = entry.varint(1).unwrap_or(0);
        // The overlapping table's entry carries `{location, length}` rather
        // than a bare index; the run table's does not.
        let length = entry
            .bytes(3)
            .and_then(crate::pb::decode_nested)
            .and_then(|range| range.varint(2));
        let Some(target) = reference(&entry, 2) else {
            continue;
        };
        // The entry points at a highlight, and the highlight at the comment.
        // A document that points straight at the comment is also accepted:
        // nothing here has ever been seen, so neither shape is assumed.
        let comment = match archives.get(&target) {
            Some((TYPE_HIGHLIGHT, _, highlight)) => {
                reached_highlights.insert(target);
                reference(highlight, 1).unwrap_or(target)
            }
            _ => target,
        };
        out.push((
            comment,
            Anchor::Text {
                storage: identifier,
                table: field,
                start,
                length,
            },
        ));
    }
    out
}

/// Storage, table field and character index of every tracked-change entry.
fn change_anchors(
    archives: &BTreeMap<u64, (u32, String, Message)>,
) -> BTreeMap<u64, (u64, u32, u64)> {
    let mut out = BTreeMap::new();
    for (identifier, (message_type, _, archive)) in archives {
        if *message_type != crate::TYPE_STORAGE {
            continue;
        }
        for field in [crate::text::INSERTION_TABLE, crate::text::DELETION_TABLE] {
            // The same wrapper as every attribute table: the storage's field
            // is the table, and repeated field 1 holds the entries. Settled
            // by pages-tracked.pages, alongside the highlight tables.
            let entries = archive
                .fields
                .iter()
                .filter(|f| f.number == field)
                .filter_map(|f| match &f.value {
                    Value::Bytes(raw) => crate::pb::decode_nested(raw),
                    _ => None,
                })
                .flat_map(|table| {
                    table
                        .fields
                        .iter()
                        .filter(|f| f.number == 1)
                        .filter_map(|f| match &f.value {
                            Value::Bytes(raw) => crate::pb::decode_nested(raw),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                });
            for entry in entries {
                if let Some(target) = reference(&entry, 2) {
                    out.insert(target, (*identifier, field, entry.varint(1).unwrap_or(0)));
                }
            }
        }
    }
    out
}

/// Every `TSD.DrawableArchive.comment` in a payload, at whatever depth the
/// drawable archive sits: field 6 of a message that also has a geometry at 1
/// and is not itself a comment storage.
///
/// Walking by shape rather than by type is what makes this work for the six
/// different archives that embed a drawable — an image, a shape, a group, a
/// table info, a chart, a Keynote placeholder — without a table of paths.
fn comment_references(archive: &Message) -> Vec<u64> {
    let mut out = Vec::new();
    walk_for_comments(archive, 0, &mut out);
    out
}

fn walk_for_comments(message: &Message, depth: usize, out: &mut Vec<u64>) {
    // A drawable archive is never more than three levels down — `[1]` for an
    // image, `[1, 1]` for a shape info, `[1, 1, 1]` for a placeholder — and a
    // bound keeps this off the deep object graphs a table builds.
    if depth > 3 {
        return;
    }
    let looks_like_drawable = message.get(crate::drawable::field::GEOMETRY).is_some()
        || message.get(crate::drawable::field::PARENT).is_some();
    if looks_like_drawable {
        if let Some(target) = reference(message, crate::drawable::field::COMMENT) {
            out.push(target);
        }
    }
    for field in &message.fields {
        if let Value::Bytes(raw) = &field.value {
            if let Some(nested) = crate::pb::decode_nested(raw) {
                walk_for_comments(&nested, depth + 1, out);
            }
        }
    }
}

fn reference(message: &Message, field: u32) -> Option<u64> {
    message
        .bytes(field)
        .and_then(crate::pb::decode_nested)
        .and_then(|m| reference_target(&m))
}

fn references(message: &Message, field: u32) -> Vec<u64> {
    message
        .fields
        .iter()
        .filter(|f| f.number == field)
        .filter_map(|f| match &f.value {
            Value::Bytes(raw) => crate::pb::decode_nested(raw).and_then(|m| reference_target(&m)),
            _ => None,
        })
        .collect()
}

/// A `TSP.Date` — one required double, seconds from 2001-01-01.
fn date(message: &Message, field: u32) -> Option<f64> {
    let nested = crate::pb::decode_nested(message.bytes(field)?)?;
    match nested.get(1)? {
        Value::Fixed64(bytes) => Some(f64::from_le_bytes(*bytes)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pb::Field;

    fn message(fields: Vec<(u32, Value)>) -> Message {
        Message {
            fields: fields
                .into_iter()
                .map(|(number, value)| Field { number, value })
                .collect(),
        }
    }

    fn reference_bytes(identifier: u64) -> Value {
        Value::Bytes(message(vec![(1, Value::Varint(identifier))]).encode())
    }

    #[test]
    fn a_change_kind_has_no_zero() {
        assert_eq!(
            ChangeKind::from_varint(Some(1)),
            ChangeKind::Insertion,
            "1 is an insertion"
        );
        assert_eq!(ChangeKind::from_varint(Some(2)), ChangeKind::Deletion);
        // The point of the enum: a missing kind is not the first variant.
        assert_eq!(ChangeKind::from_varint(None), ChangeKind::Unknown(0));
        assert_eq!(ChangeKind::from_varint(Some(0)), ChangeKind::Unknown(0));
        assert_eq!(ChangeKind::from_varint(Some(7)), ChangeKind::Unknown(7));
    }

    #[test]
    fn an_author_reads_its_name_and_colour() {
        let colour = message(vec![
            (1, Value::Varint(1)),
            (3, Value::Fixed32(1.0f32.to_le_bytes())),
            (4, Value::Fixed32(0.5f32.to_le_bytes())),
            (5, Value::Fixed32(0.0f32.to_le_bytes())),
            (6, Value::Fixed32(1.0f32.to_le_bytes())),
        ]);
        let author = read_author(
            &42,
            &message(vec![
                (author_field::NAME, Value::Bytes(b"Ada Lovelace".to_vec())),
                (author_field::COLOR, Value::Bytes(colour.encode())),
                (author_field::PUBLIC_ID, Value::Bytes(b"abc123".to_vec())),
                (author_field::IS_PUBLIC_AUTHOR, Value::Varint(1)),
            ]),
        );
        assert_eq!(author.name.as_deref(), Some("Ada Lovelace"));
        assert_eq!(author.public_id.as_deref(), Some("abc123"));
        assert!(author.is_public_author);
        assert_eq!(author.color.map(|c| c.to_string()), Some("#ff8000".into()));
    }

    /// `table_highlight` is a run table: `{index, reference}`, no length. The
    /// reference is to a `TSWP.HighlightArchive`, and *that* points at the
    /// comment — two hops, and a decoder that takes one lands on the wrong
    /// object.
    #[test]
    fn a_run_highlight_reaches_the_comment_through_the_highlight() {
        let mut archives: BTreeMap<u64, (u32, String, Message)> = BTreeMap::new();
        archives.insert(
            700,
            (
                TYPE_HIGHLIGHT,
                "Index/Document.iwa".into(),
                message(vec![(1, reference_bytes(900))]),
            ),
        );
        let entry = message(vec![(1, Value::Varint(12)), (2, reference_bytes(700))]);
        // The table wraps its entries in repeated field 1 — the shape
        // pages-comments.pages exhibits, not the flat one first guessed.
        let table = message(vec![
            (1, Value::Bytes(entry.encode())),
            (
                1,
                Value::Bytes(message(vec![(1, Value::Varint(20))]).encode()),
            ),
        ]);
        let storage = message(vec![(
            crate::text::HIGHLIGHT_TABLE,
            Value::Bytes(table.encode()),
        )]);

        let mut reached = Default::default();
        let found = highlight_anchors(
            &storage,
            crate::text::HIGHLIGHT_TABLE,
            5,
            &archives,
            &mut reached,
        );
        assert!(reached.contains(&700), "the highlight was walked through");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].0, 900, "the comment, not the highlight");
        assert_eq!(
            found[0].1,
            Anchor::Text {
                storage: 5,
                table: crate::text::HIGHLIGHT_TABLE,
                start: 12,
                length: None,
            }
        );
    }

    /// `table_overlapping_highlight` carries an explicit range, which is how
    /// two comments cover the same characters.
    #[test]
    fn an_overlapping_highlight_carries_its_own_length() {
        let archives = BTreeMap::new();
        let range = message(vec![(1, Value::Varint(12)), (2, Value::Varint(9))]);
        let entry = message(vec![
            (1, Value::Varint(12)),
            (2, reference_bytes(900)),
            (3, Value::Bytes(range.encode())),
        ]);
        let table = message(vec![(1, Value::Bytes(entry.encode()))]);
        let storage = message(vec![(
            crate::text::OVERLAPPING_HIGHLIGHT_TABLE,
            Value::Bytes(table.encode()),
        )]);

        let found = highlight_anchors(
            &storage,
            crate::text::OVERLAPPING_HIGHLIGHT_TABLE,
            5,
            &archives,
            &mut Default::default(),
        );
        assert_eq!(
            found[0].1,
            Anchor::Text {
                storage: 5,
                table: crate::text::OVERLAPPING_HIGHLIGHT_TABLE,
                start: 12,
                length: Some(9),
            }
        );
        assert_eq!(
            found[0].1.as_str(),
            "storage 5 12..21 (table_overlapping_highlight)"
        );
    }

    /// A comment on an image is field 6 of the drawable archive, and the
    /// drawable archive is one level inside a `TSD.ImageArchive`.
    #[test]
    fn an_object_comment_is_found_inside_the_archive_that_carries_it() {
        let drawable = message(vec![
            (crate::drawable::field::GEOMETRY, Value::Bytes(Vec::new())),
            (crate::drawable::field::COMMENT, reference_bytes(901)),
        ]);
        let image = message(vec![(1, Value::Bytes(drawable.encode()))]);
        assert_eq!(comment_references(&image), vec![901]);
    }

    #[test]
    fn a_date_is_a_double_in_a_submessage() {
        let inner = message(vec![(1, Value::Fixed64(768_000_000f64.to_le_bytes()))]);
        let outer = message(vec![(2, Value::Bytes(inner.encode()))]);
        assert_eq!(date(&outer, 2), Some(768_000_000.0));
        assert_eq!(date(&outer, 3), None);
    }
}

// -- writing a comment -------------------------------------------------------

/// Field numbers of `TSWP.HighlightArchive` and of the author storage — the two
/// archives with no module of their own above, because until now nothing wrote
/// them.
mod highlight_field {
    /// `TSWP.HighlightArchive.comment`.
    pub const COMMENT: u32 = 1;
    /// `TSWP.HighlightArchive.uuid`, a *string* UUID and not a `TSP.UUID`.
    pub const UUID: u32 = 2;
    /// `TSK.AnnotationAuthorStorageArchive.annotation_author`.
    pub const STORAGE_AUTHOR: u32 = 1;
}

/// Who is commenting, and what they said.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CommentEdit {
    /// The author's name as the app shows it. An author of that name already in
    /// the document is reused; otherwise one is added to the author storage.
    pub author: String,
    pub text: String,
}

/// Attach a comment to a range of text.
///
/// The range is in **UTF-16 code units**, the unit iWork indexes text in, and
/// half-open — the same convention as every other range in this crate.
///
/// Four objects are involved and the shape is the fixture's, not the schema's:
/// a `TSD.CommentStorageArchive` (3056) holding the words, the date and a
/// reference to a `TSK.AnnotationAuthorArchive` (212); a `TSWP.HighlightArchive`
/// (2013) pointing at the comment; and two entries in the storage's
/// `table_highlight` (23) — the anchor at `start`, and a **bare index at `end`**
/// that ends the run. A table entry with no reference is not a comment on
/// nothing; it is where the previous comment stops.
///
/// **What it refuses.** A storage carrying tracked changes, because every edit
/// to one is refused here; a range that runs past the end of the text; an empty
/// range, which would anchor a comment to nothing; and a range that overlaps a
/// comment already there — `table_overlapping_highlight` (25) is what the app
/// uses for two comments over one stretch of text, and no document in this
/// corpus has one to write from.
pub fn add_comment(
    document: &mut crate::Document,
    storage: u64,
    start: u64,
    end: u64,
    edit: &CommentEdit,
) -> Result<u64, crate::Error> {
    use crate::pb::Value;

    if end <= start {
        return Err(crate::Error::Format(format!(
            "a comment covers a range of text, and {start}..{end} is empty"
        )));
    }
    if edit.text.is_empty() {
        return Err(crate::Error::Format(
            "a comment with no words in it is not a comment".into(),
        ));
    }

    let text = document.storage_text(storage)?;
    let length = text.encode_utf16().count() as u64;
    if end > length {
        return Err(crate::Error::Format(format!(
            "storage {storage} holds {length} unit(s) of text, so a comment cannot cover \
             {start}..{end}"
        )));
    }

    let archive = document.archive(storage)?;
    for table in crate::text::CHANGE_TABLES {
        if archive.get(*table).is_some() {
            return Err(crate::Error::Format(format!(
                "storage {storage} carries tracked changes, and this crate does not edit one"
            )));
        }
    }
    if archive
        .get(crate::text::OVERLAPPING_HIGHLIGHT_TABLE)
        .is_some()
    {
        return Err(crate::Error::Format(format!(
            "storage {storage} uses table_overlapping_highlight (25), which no document here \
             has and nothing here writes"
        )));
    }

    // The entries already there, so an overlap is refused rather than layered.
    let mut entries = highlight_entries(&archive)?;
    if let Some((index, _)) = entries
        .iter()
        .find(|(index, target)| target.is_some() && *index < end && *index >= start)
    {
        return Err(crate::Error::Format(format!(
            "storage {storage} already has a comment anchored at {index}, inside {start}..{end}"
        )));
    }
    // …and the one in force *at* `start`, which would run through the new one.
    if let Some((index, _)) = entries
        .iter()
        .rfind(|(index, target)| *index <= start && target.is_some())
    {
        let ends_at = entries
            .iter()
            .find(|(after, _)| *after > *index)
            .map(|(after, _)| *after)
            .unwrap_or(length);
        if ends_at > start {
            return Err(crate::Error::Format(format!(
                "storage {storage} already has a comment covering {index}..{ends_at}, which \
                 {start}..{end} overlaps"
            )));
        }
    }

    let author = find_or_add_author(document, &edit.author)?;

    let mut grow = crate::create::Grow::new(document);
    let comment = grow.allocate();
    let highlight = grow.allocate();
    grow.beside(
        storage,
        comment,
        TYPE_COMMENT_STORAGE,
        &comment_archive(&edit.text, author, comment),
    )?;
    grow.beside(
        storage,
        highlight,
        TYPE_HIGHLIGHT,
        &highlight_archive(comment, highlight),
    )?;
    grow.finish()?;

    // The anchor and the end of its run, in index order — an attribute table
    // that is not sorted is one the app reads as a different range.
    entries.push((start, Some(highlight)));
    if !entries.iter().any(|(index, _)| *index == end) {
        entries.push((end, None));
    }
    // **A run-anchored table starts at 0**, whatever it says about the run: the
    // entry at 0 is what covers the text before the first comment, and a table
    // whose first entry is anywhere else leaves those characters with no
    // attribute at all — which `Document::problems` calls out by name.
    if !entries.iter().any(|(index, _)| *index == 0) {
        entries.push((0, None));
    }
    entries.sort_by_key(|(index, _)| *index);
    let mut archive = document.archive(storage)?;
    archive.clear(crate::text::HIGHLIGHT_TABLE);
    let mut table = Message::default();
    for (index, target) in &entries {
        let mut entry = Message::default();
        entry.set_in_order(1, Value::Varint(*index));
        if let Some(target) = target {
            entry.set_in_order(2, Value::Bytes(crate::create::reference_bytes(*target)));
        }
        table.append_in_order(1, Value::Bytes(entry.encode()));
    }
    archive.set_in_order(crate::text::HIGHLIGHT_TABLE, Value::Bytes(table.encode()));
    document.set_archive_of(storage, &archive)?;
    document.declare_external_references();
    Ok(comment)
}

/// The `table_highlight` entries of a storage, as `(index, target)`.
fn highlight_entries(archive: &Message) -> Result<Vec<(u64, Option<u64>)>, crate::Error> {
    let Some(table) = archive
        .bytes(crate::text::HIGHLIGHT_TABLE)
        .and_then(decode_nested)
    else {
        return Ok(Vec::new());
    };
    let mut out = Vec::new();
    for value in table.all(1) {
        let Value::Bytes(raw) = value else { continue };
        let entry = decode_nested(raw).ok_or_else(|| {
            crate::Error::Format("a table_highlight entry does not decode".into())
        })?;
        let index = entry.varint(1).unwrap_or(0);
        let target = entry
            .bytes(2)
            .and_then(decode_nested)
            .as_ref()
            .and_then(reference_target);
        out.push((index, target));
    }
    Ok(out)
}

/// `TSD.CommentStorageArchive` — the words, when, and by whom.
fn comment_archive(text: &str, author: u64, seed: u64) -> Message {
    use crate::create::{message, nested, reference, string, varint};
    message(vec![
        string(comment_field::TEXT, text),
        nested(
            comment_field::CREATION_DATE,
            vec![crate::create::double(1, now())],
        ),
        reference(comment_field::AUTHOR, author),
        // The comment's own identity. Nothing outside the document refers to
        // it, so it is derived from the object it belongs to rather than drawn
        // from anywhere — a document written twice is then the same document.
        nested(
            comment_field::STORAGE_UUID,
            vec![varint(1, mix(seed, 1)), varint(2, mix(seed, 2))],
        ),
    ])
}

/// `TSWP.HighlightArchive` — what the text points at, and what points at the
/// comment.
fn highlight_archive(comment: u64, seed: u64) -> Message {
    use crate::create::{message, reference, string};
    message(vec![
        reference(highlight_field::COMMENT, comment),
        string(highlight_field::UUID, &uuid_from(seed)),
    ])
}

/// An author of that name, or a new one in the document's author storage.
///
/// Every iWork document carries exactly one
/// `TSK.AnnotationAuthorStorageArchive`, empty until someone comments; this is
/// what fills it. The colour is the app's own first author colour — the yellow
/// `pages-comments` carries — because an author has to be tinted something and
/// nothing here can ask the app which.
fn find_or_add_author(document: &mut crate::Document, name: &str) -> Result<u64, crate::Error> {
    use crate::create::{float, message, nested, string, varint};
    use crate::pb::Value;

    let existing = annotations(document);
    if let Some(author) = existing
        .authors
        .iter()
        .find(|author| author.name.as_deref() == Some(name))
    {
        return Ok(author.identifier);
    }
    let (storage, _) = existing.author_storage.ok_or_else(|| {
        crate::Error::Format(
            "this document has no TSK.AnnotationAuthorStorageArchive, and a comment needs an \
             author to belong to"
                .into(),
        )
    })?;

    let mut grow = crate::create::Grow::new(document);
    let author = grow.allocate();
    grow.beside(
        storage,
        author,
        TYPE_AUTHOR,
        &message(vec![
            string(author_field::NAME, name),
            nested(
                author_field::COLOR,
                vec![
                    varint(1, 1),
                    float(3, 0.980_392_16),
                    float(4, 0.937_254_9),
                    float(5, 0.352_941_2),
                    float(6, 1.0),
                    varint(12, 1),
                    float(13, 1.0),
                ],
            ),
            // The collaboration identity: a UUID and a 64-hex digest, in the
            // shape the app writes one. It identifies this author within this
            // document and means nothing outside it.
            string(
                author_field::PUBLIC_ID,
                &format!("{}:{}", uuid_from(author), digest_from(author)),
            ),
            varint(author_field::IS_PUBLIC_AUTHOR, 0),
        ]),
    )?;
    grow.finish()?;

    let mut archive = document.archive(storage)?;
    archive.append_in_order(
        highlight_field::STORAGE_AUTHOR,
        Value::Bytes(crate::create::reference_bytes(author)),
    );
    document.set_archive_of(storage, &archive)?;
    document.declare_external_references();
    Ok(author)
}

/// Now, as `TSP.Date` counts it: seconds since 2001-01-01.
fn now() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() - crate::table::APPLE_EPOCH)
        .unwrap_or(0.0)
}

/// splitmix64, for identities that have to differ and mean nothing else.
fn mix(seed: u64, salt: u64) -> u64 {
    let mut z = seed
        .wrapping_mul(0x2545_F491_4F6C_DD1D)
        .wrapping_add(salt.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A UUID string in the shape Apple writes them, derived rather than drawn.
fn uuid_from(seed: u64) -> String {
    let hex: String = (0..4)
        .flat_map(|salt| mix(seed, salt + 10).to_be_bytes())
        .map(|b| format!("{b:02X}"))
        .collect();
    format!(
        "{}-{}-{}-{}-{}",
        &hex[0..8],
        &hex[8..12],
        &hex[12..16],
        &hex[16..20],
        &hex[20..32]
    )
}

/// The 64 hex digits after the colon in a collaboration public id.
fn digest_from(seed: u64) -> String {
    (0..4)
        .flat_map(|salt| mix(seed, salt + 20).to_be_bytes())
        .map(|b| format!("{b:02x}"))
        .collect()
}
