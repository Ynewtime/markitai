//! `TSWP.StorageArchive` — the text model shared by all three apps.
//!
//! Text and formatting are stored separately. The text is field 3; everything
//! else that says anything about it is an **attribute table**, a list of entries
//! anchored to character indices. There are twenty-two of them ([`TABLES`]), and
//! any of them can be the reason an edit damages a document — a style run, a
//! hyperlink, an anchored image, a footnote mark, a tracked change, a comment.
//!
//! ```text
//!  index   0    5    10   15   20
//!  text    Ü b e r s c h r i f t \r E i n   r o …
//!  ¶ table [0 → Title …………………………………] [12 → Red ……
//!  char    [0 nil ……………][8 → Bold ……………][17 nil …
//!  anchor                     ↑ 9 = U+FFFC, an image lives here
//! ```
//!
//! Delete `[5, 20)` from that and three different things have to happen. The
//! paragraph entry at 12 disappears, because its paragraph merged into the one
//! before it and 5 is no longer a paragraph start. The character entries at 8
//! and 17 both come back to 5, and the run that was wholly inside the range
//! loses to the one that outlived it. And the anchor at 9 is not remappable at
//! all — its character is gone, and with it the image.
//!
//! Everything in this module below [`length`] exists to get that right. What it
//! does was **measured**, by having Pages perform each edit and diffing the
//! archives; see `FORMAT.md` §Text for the probes and their results.
//!
//! ## Indices are UTF-16 code units
//!
//! Every index in this module — every argument, every table entry, every
//! paragraph range — counts UTF-16 code units, because the text model is
//! NSString-backed. An emoji is two. An edit may not land between the halves of
//! a surrogate pair, and [`utf16_offset`] is what refuses.

use crate::pb::{Field, Message, Value};

/// Field numbers holding attribute tables of the run-table shape —
/// `{character_index, reference}` — that a *style* can be reached through.
///
/// This is the narrower list [`crate::Document::delete_text_style`] and
/// [`crate::Document::text_style_usage`] walk, so that dropping a style cannot
/// leave a dangling reference behind in one of them. The complete inventory of
/// what a storage can carry, which is what an *edit* has to remap, is
/// [`TABLES`].
pub const ATTRIBUTE_TABLES: &[u32] = &[5, 7, 8, 9, 11, 12, 15, 16, 17];

/// How a table's entries are anchored into the text — which decides what an
/// edit does to them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchoring {
    /// One entry per paragraph, at the paragraph's first character. An entry
    /// may also sit at the very end of the text, which is where the style of a
    /// paragraph not yet typed comes from.
    Paragraph,
    /// A run of characters, reaching from this entry's index to the next
    /// entry's, or to the end of the text.
    Run,
    /// A single character — the `U+FFFC` an attachment stands in for. Delete
    /// that character and the thing it anchors has to go too, which is a
    /// document-wide operation, so this crate refuses instead.
    Character,
    /// An explicit `{location, length}` range, so two entries may cover the
    /// same characters. Only the two comment/annotation tables use it.
    Range,
}

/// One attribute table of `TSWP.StorageArchive`.
pub struct Table {
    /// Field number in the storage archive.
    pub field: u32,
    /// Apple's name for it, from the 15.3.1 schema.
    pub name: &'static str,
    pub anchoring: Anchoring,
    /// What its entries point at, for the reader.
    pub points_at: &'static str,
}

/// Every attribute table a `TSWP.StorageArchive` can carry.
///
/// Read off the 15.3.1 `TSWPArchives.proto` descriptor carved from the installed
/// binaries (`reference/protos-15.3/*/TSWPArchives.proto`), and cross-checked
/// against the corpus: fields 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 17, 24 and
/// 28 occur in documents these apps wrote, and the rest are named here so that
/// an edit knows what to do with one when it meets it.
///
/// **A field not in this list, and not one of the plain ones, is an error and
/// not something to skip** — see [`unknown_table`]. Every table nobody knows
/// about is a table a remapping breaks silently.
pub const TABLES: &[Table] = &[
    Table {
        field: 5,
        name: "table_para_style",
        anchoring: Anchoring::Paragraph,
        points_at: "TSWP.ParagraphStyleArchive (2022)",
    },
    Table {
        field: 6,
        name: "table_para_data",
        anchoring: Anchoring::Paragraph,
        points_at: "{first, second} — the list level and its flags",
    },
    Table {
        field: 7,
        name: "table_list_style",
        anchoring: Anchoring::Paragraph,
        points_at: "TSWP.ListStyleArchive (2023)",
    },
    Table {
        field: 8,
        name: "table_char_style",
        anchoring: Anchoring::Run,
        points_at: "TSWP.CharacterStyleArchive (2021)",
    },
    Table {
        field: 9,
        name: "table_attachment",
        anchoring: Anchoring::Character,
        points_at: "TSWP.DrawableAttachmentArchive (2003), TSWP.NumberAttachmentArchive (2043) …",
    },
    Table {
        field: 11,
        name: "table_smartfield",
        anchoring: Anchoring::Run,
        points_at: "TSWP.HyperlinkFieldArchive (2032) and the other smart fields, 2031–2042",
    },
    Table {
        field: 12,
        name: "table_layout_style",
        anchoring: Anchoring::Paragraph,
        points_at: "TSWP.ColumnStyleArchive (2024)",
    },
    Table {
        field: 14,
        name: "table_para_starts",
        anchoring: Anchoring::Paragraph,
        points_at: "{first, second} — paragraph-start bookkeeping",
    },
    Table {
        field: 15,
        name: "table_bookmark",
        anchoring: Anchoring::Run,
        points_at: "TSWP.BookmarkFieldArchive (2035)",
    },
    Table {
        field: 16,
        name: "table_footnote",
        anchoring: Anchoring::Character,
        points_at: "TSWP.FootnoteReferenceAttachmentArchive (2008)",
    },
    // Anchored like a paragraph, not like an attachment: a section entry sits
    // on the character *after* the `U+0004` that begins it, which is a
    // paragraph start — `pages-report` has one at 0 and one at 146, reading
    // `…123-4567\n\u{4}Company Name`, with the entry on the `C`. Deleting the
    // break itself is refused — by [`destroyed_sections`], because the
    // paragraph anchoring is exactly what puts it out of
    // [`destroyed_anchors`]' reach.
    Table {
        field: 17,
        name: "table_section",
        anchoring: Anchoring::Paragraph,
        points_at: "TP.SectionArchive / TSWP.SectionPlaceholderArchive (10011)",
    },
    Table {
        field: 18,
        name: "table_rubyfield",
        anchoring: Anchoring::Run,
        points_at: "TSWP.RubyFieldArchive (2042)",
    },
    Table {
        field: 19,
        name: "table_language",
        anchoring: Anchoring::Run,
        points_at: "a BCP-47 string",
    },
    Table {
        field: 20,
        name: "table_dictation",
        anchoring: Anchoring::Run,
        points_at: "a dictation metadata string",
    },
    Table {
        field: 21,
        name: "table_insertion",
        anchoring: Anchoring::Run,
        points_at: "TSWP.ChangeArchive (2060), kind = insertion",
    },
    Table {
        field: 22,
        name: "table_deletion",
        anchoring: Anchoring::Run,
        points_at: "TSWP.ChangeArchive (2060), kind = deletion",
    },
    Table {
        field: 23,
        name: "table_highlight",
        anchoring: Anchoring::Run,
        points_at: "TSWP.HighlightArchive (2013) — a comment anchor",
    },
    Table {
        field: 24,
        name: "table_para_bidi",
        anchoring: Anchoring::Paragraph,
        points_at: "{first, second} — writing direction",
    },
    Table {
        field: 25,
        name: "table_overlapping_highlight",
        anchoring: Anchoring::Range,
        points_at: "TSWP.HighlightArchive (2013), as explicit ranges",
    },
    Table {
        field: 26,
        name: "table_pencil_annotation",
        anchoring: Anchoring::Range,
        points_at: "TSWP.PencilAnnotationArchive (2016)",
    },
    Table {
        field: 27,
        name: "table_tatechuyoko",
        anchoring: Anchoring::Run,
        points_at: "TSWP.TateChuYokoFieldArchive (10023)",
    },
    Table {
        field: 28,
        name: "table_drop_cap_style",
        anchoring: Anchoring::Paragraph,
        points_at: "TSWP.DropCapStyleArchive (10024)",
    },
];

/// Storage fields that are not attribute tables: `kind` (1), the stylesheet
/// reference (2), the text (3), `has_itext` (4) and `in_document` (10).
///
/// Field 13 is a gap in every version of the schema and is deliberately absent:
/// a document that has one is a document this crate has not seen, and
/// [`unknown_table`] should say so rather than wave it through.
pub const PLAIN_FIELDS: &[u32] = &[1, 2, 3, 4, 10];

/// Field numbers of the tables other modules reach for by name.
///
/// The whole inventory is [`TABLES`]; these are the ones with a caller outside
/// this module, and a named constant reads better than a number at the call
/// site.
pub const PARAGRAPH_STYLE_TABLE: u32 = 5;
/// `table_layout_style` — column layouts, one entry per paragraph range.
pub const LAYOUT_TABLE: u32 = 12;
/// `table_bookmark`.
pub const BOOKMARK_TABLE: u32 = 15;
/// `table_footnote`.
pub const FOOTNOTE_TABLE: u32 = 16;
/// `table_section`.
pub const SECTION_TABLE: u32 = 17;
/// `table_insertion` — a tracked insertion.
pub const INSERTION_TABLE: u32 = 21;
/// `table_deletion` — a tracked deletion, whose characters are still in the
/// text.
pub const DELETION_TABLE: u32 = 22;
/// `table_highlight` — a comment anchor, run-anchored.
pub const HIGHLIGHT_TABLE: u32 = 23;
/// `table_overlapping_highlight` — a comment anchor with an explicit range, so
/// two comments may cover the same characters.
pub const OVERLAPPING_HIGHLIGHT_TABLE: u32 = 25;

/// The two tables change tracking uses.
///
/// A storage carrying either is a storage this crate will not edit — see
/// [`crate::Error::TrackedChanges`] and the module comment of
/// [`crate::annotations`].
pub const CHANGE_TABLES: &[u32] = &[INSERTION_TABLE, DELETION_TABLE];

/// The table at a storage field, if that field is one.
pub fn table(field: u32) -> Option<&'static Table> {
    TABLES.iter().find(|t| t.field == field)
}

/// The first field of `storage` that is neither a known table nor a known plain
/// field, and could therefore be a table this crate would remap wrongly.
///
/// Varint fields are safe whatever they are — they hold no character indices —
/// so only length-delimited ones count.
pub fn unknown_table(storage: &Message) -> Option<u32> {
    storage
        .fields
        .iter()
        .find(|field| {
            matches!(field.value, Value::Bytes(_))
                && !PLAIN_FIELDS.contains(&field.number)
                && table(field.number).is_none()
        })
        .map(|field| field.number)
}

/// The first field of `storage` that *is* a known attribute table and whose
/// bytes are not a message this crate can decode and re-encode unchanged.
///
/// [`crate::pb::decode_nested`] insists on that byte-identity round trip,
/// because bytes that merely happen to parse are how a string gets mistaken for
/// a message. A table that fails it is one [`apply`] would skip while remapping
/// every other table in the storage — leaving the skipped one holding indices
/// into text that has moved, which is worse than not editing at all. So it is a
/// refusal, not something to step over.
///
/// An **empty** table field is not that case: an empty message is a table with
/// no entries, there is nothing in it to remap, and `decode_nested` answers
/// `None` for zero bytes by design.
pub fn undecodable_table(storage: &Message) -> Option<u32> {
    storage
        .fields
        .iter()
        .find(|field| match &field.value {
            Value::Bytes(raw) => {
                table(field.number).is_some()
                    && !raw.is_empty()
                    && crate::pb::decode_nested(raw).is_none()
            }
            _ => false,
        })
        .map(|field| field.number)
}

/// Is every text run of `storage` valid UTF-8?
///
/// [`read`] decodes lossily, which is right for a reader — a storage whose bytes
/// are not UTF-8 should still be listed rather than blowing up an inspection —
/// and fatal for a writer: [`apply`] writes the text it was given back into
/// field 3, so an edit through a lossy read replaces every ill-formed sequence
/// with `U+FFFD` for good, and shifts every index after it. No storage in the
/// corpus is ill-formed, and an edit to one is refused rather than performed on
/// a string the document does not contain.
pub fn text_is_utf8(storage: &Message) -> bool {
    storage.all(3).all(|value| match value {
        Value::Bytes(raw) => std::str::from_utf8(raw).is_ok(),
        _ => true,
    })
}

/// Characters that end a paragraph.
///
/// `\n` is the obvious one. The others are not obvious and matter, because
/// the paragraph-style table puts a run immediately *after* each of them:
/// reading one as ordinary text splits the paragraphs one character wrong,
/// which is enough to make a paragraph style land in the wrong place.
///
/// **`\r` (`U+000D`) is the one this crate had wrong**, and it is the one the
/// apps write most often, because AppleScript's `return` is a carriage return
/// and every fixture built by script therefore has them. In `pages-styled`,
/// whose body reads `Überschrift\rEin roter Absatz…`, the paragraph table holds
/// `[0, 12, 74, 128]` — the characters after each `\r` — while
/// [`paragraph_ranges`] saw one paragraph of 171 characters. Four storages in
/// the corpus were affected, in Pages and in Keynote. It went unnoticed because
/// the test that would have caught it skips storages it believes have fewer
/// than two paragraphs, which is exactly what this bug made them look like.
///
/// `U+000C` is the **page or column break** a user inserts from the Insert
/// menu, and it ends a paragraph the same way. Found by checking this rule
/// against all 901 template bundles the three apps ship: 40 Pages templates put
/// a paragraph run immediately after one, always in the shape `…\n\u{c}Text`,
/// and every one of those runs was reported as landing off a paragraph
/// boundary until it was counted.
///
/// `U+0005` appears where a Pages document changes layout mid-storage.
/// Verified in a Pages article at `…\n\n\u{5}Features\n`, where the run sits on
/// the `F`.
///
/// `U+0004` marks a section boundary. Verified in a document Pages built from
/// its "Project Proposal" template, whose body storage reads
/// `…123-4567\n\u{4}Company Name\n` at both of its section breaks and whose
/// paragraph table has a run on the `C`. It also turns up alone as the whole of
/// a body storage in a document whose text lives in shapes, which is the same
/// character doing the same job with nothing either side of it.
pub const PARAGRAPH_BREAKS: &[u16] = &[0x000A, 0x000D, 0x000C, 0x0005, 0x0004];

/// Length of a storage's text in UTF-16 code units — the unit run indices are
/// counted in.
pub fn length(text: &str) -> u64 {
    text.encode_utf16().count() as u64
}

/// Character ranges of the paragraphs in `text`, in UTF-16 code units.
///
/// A paragraph ends at `\n`, or at one of the break characters in
/// [`PARAGRAPH_BREAKS`]. The terminator belongs to the paragraph it ends, so the
/// ranges tile the text with no gaps. A paragraph style applies to whole
/// paragraphs, and this is how to name them. Empty text has no paragraphs.
pub fn paragraph_ranges(text: &str) -> Vec<std::ops::Range<u64>> {
    let mut out = Vec::new();
    let mut start = 0u64;
    let mut index = 0u64;
    for unit in text.encode_utf16() {
        index += 1;
        if PARAGRAPH_BREAKS.contains(&unit) {
            out.push(start..index);
            start = index;
        }
    }
    if start < index {
        out.push(start..index);
    }
    out
}

/// Concatenate the text runs of a storage archive.
pub fn read(storage: &Message) -> String {
    storage
        .all(3)
        .filter_map(|value| match value {
            Value::Bytes(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
            _ => None,
        })
        .collect()
}

/// Does this storage hold anything a reader would call text?
///
/// Storages routinely exist only to anchor something else, and their contents
/// are placeholders rather than words:
///
/// - `U+FFFC OBJECT REPLACEMENT CHARACTER` stands in for an embedded drawable,
///   and is the entire contents of the storages attached to Numbers tables;
/// - `U+0004` appears alone in the Pages body storage of a document whose text
///   all lives in shapes.
pub fn has_content(text: &str) -> bool {
    text.chars()
        .any(|c| !c.is_whitespace() && !c.is_control() && c != '\u{FFFC}')
}

/// Byte offset in `text` of UTF-16 code unit `index`.
///
/// `None` when the index is past the end of the text, or lands **between the
/// halves of a surrogate pair** — an edit there would split an emoji into two
/// unpaired surrogates, which is not a string any more.
///
/// Run indices are character offsets, not byte offsets: in a storage reading
/// `"Von Benjamin Keller\nVeröffentlicht am 07.09.2017\nim Magazin …"` the
/// character table holds `[0, 20, 49]`, exactly the paragraph starts counted in
/// characters, where counting UTF-8 bytes would give `[0, 20, 50]`. UTF-16 is
/// the unit because the text model is NSString-backed.
pub fn utf16_offset(text: &str, index: u64) -> Option<usize> {
    let mut units = 0u64;
    for (offset, character) in text.char_indices() {
        if units == index {
            return Some(offset);
        }
        units += character.len_utf16() as u64;
        if units > index {
            return None; // the index was inside this character's pair
        }
    }
    (units == index).then_some(text.len())
}

/// Characters this crate will not insert, because the character alone is not
/// the thing.
///
/// `U+FFFC` is what an attachment table's entry points *at*: a lone one is an
/// attachment with nothing attached. `U+000E` is the same case one table along —
/// a **footnote mark**, whose entry in `table_footnote` carries the
/// `TSWP.FootnoteReferenceAttachmentArchive` and, through it, the note's own
/// storage; written bare it is a mark with no note behind it. `U+0004` and
/// `U+0005` are the section and layout breaks, and a section break with no
/// `TP.SectionArchive` behind it is a document that says it has a section it
/// does not have.
pub const UNWRITABLE: &[char] = &['\u{FFFC}', '\u{000E}', '\u{0004}', '\u{0005}'];

/// The characters a character-anchored entry has been seen sitting on:
/// `U+FFFC OBJECT REPLACEMENT CHARACTER` for a drawable or a number attachment,
/// `U+000E` for a footnote mark.
///
/// This is documentation, not a test: [`destroyed_anchors`] deliberately does
/// **not** consult it, because an anchor on a character this list has not met is
/// exactly the case where refusing matters most. It read `U+FFFC` and `U+0004`
/// once, and the footnote mark it had never seen walked straight through the
/// gap.
pub const ANCHOR_CHARACTERS: &[u16] = &[0xFFFC, 0x000E];

/// One edit to a storage's text: replace `removed` code units at `at` with
/// `inserted` of them.
///
/// An insertion has `removed == 0`, a deletion `inserted == 0`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Edit {
    pub at: u64,
    pub removed: u64,
    pub inserted: u64,
}

impl Edit {
    /// First index past the removed range.
    pub fn end(self) -> u64 {
        self.at + self.removed
    }

    /// Where an index lands afterwards, for a table anchored as `anchoring`.
    ///
    /// Everything at or past the end of the removed range simply shifts. What
    /// happens *inside* it is where the three shapes part company, and each
    /// answer is one Pages was observed to give:
    ///
    /// | Anchoring | an index inside the removed range | an index at its start |
    /// |---|---|---|
    /// | [`Anchoring::Run`], [`Anchoring::Range`] | goes to the far side of the inserted text — the characters it described are gone, and what is left of its run begins after what replaced them | goes with it, unless it is index 0, which a run table must always have |
    /// | [`Anchoring::Paragraph`] | collapses to `at`, and is dropped unless `at` is still a paragraph start | stays where it is |
    /// | [`Anchoring::Character`] | **destroyed** — the character it names is gone | destroyed |
    ///
    /// A run table therefore behaves exactly like an attributed string: a
    /// character that survives keeps its attributes, and text inserted at a
    /// boundary joins the run *before* it — which is what Pages did when it moved
    /// a character run's start from 19 to 22 as four units arrived there, and
    /// what it did when it deleted a bold run's whole extent and left the text
    /// after it unbold.
    ///
    /// A paragraph table does not, and that is the surprise this phase turned
    /// up. Told to delete a whole paragraph, break included, Pages kept the
    /// **deleted** paragraph's style on the boundary and dropped the style of the
    /// paragraph that moved up into it: `[0 Title, 12 Red, 74 Italic, 128 Body]`
    /// minus `[12, 74)` came back as `[0 Title, 12 Red, 66 Body]`. Paragraph
    /// style is anchored to the paragraph start, not carried by the characters.
    pub fn remap(self, index: u64, anchoring: Anchoring) -> Option<u64> {
        match anchoring {
            // A paragraph entry at the edit's own index marks a paragraph start
            // the edit did not move, and a run table's first entry is where the
            // first run begins — text inserted at the very start of a storage
            // takes its attributes, there being no earlier run to take them
            // from. Both come before the shift, or an insertion at 0 would
            // leave the table starting late and the first characters with no
            // attribute at all.
            Anchoring::Paragraph if index <= self.at => Some(index),
            Anchoring::Run | Anchoring::Range if index == 0 => Some(0),
            Anchoring::Character if index < self.at => Some(index),
            _ if index >= self.end() => Some(index - self.removed + self.inserted),
            Anchoring::Character => None,
            Anchoring::Paragraph => Some(self.at),
            _ if index < self.at => Some(index),
            _ => Some(self.at + self.inserted),
        }
    }
}

/// What one edit did to one storage.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EditReport {
    /// Entries whose index changed.
    pub moved: usize,
    /// Entries that went away: a paragraph that merged into the one before it,
    /// a run whose whole extent was deleted.
    pub dropped: usize,
    /// Entries added, one per paragraph an insertion created in a table that
    /// had one entry per paragraph before.
    pub added: usize,
    /// Fields of the storage that were rewritten.
    pub tables: Vec<u32>,
}

/// An entry of an attribute table, kept as the message it is so that whatever
/// the entry carries beyond its index survives the edit.
struct Entry {
    index: u64,
    message: Message,
}

/// Split a table into its entries and the fields that are not entries.
///
/// An entry is recognised by its first field: a character index in the three run
/// shapes, and a `TSP.Range` in the overlapping one. A field that is neither is
/// carried through verbatim; every table in the corpus is entries and nothing
/// else, so that is insurance rather than a case that has been seen.
fn split(table: &Message, anchoring: Anchoring) -> (Vec<Entry>, Vec<Field>, u32) {
    let mut entries = Vec::new();
    let mut rest = Vec::new();
    let mut entry_field = 1;
    for field in &table.fields {
        let decoded = match &field.value {
            Value::Bytes(raw) => crate::pb::decode_nested(raw),
            _ => None,
        };
        let indexed = decoded.and_then(|m| {
            let index = match anchoring {
                Anchoring::Range => range_of(&m).map(|(location, _)| location),
                _ => m.varint(1),
            }?;
            Some((index, m))
        });
        match indexed {
            Some((index, message)) => {
                entry_field = field.number;
                entries.push(Entry { index, message });
            }
            None => rest.push(field.clone()),
        }
    }
    (entries, rest, entry_field)
}

fn join(entries: Vec<Entry>, rest: Vec<Field>, entry_field: u32) -> Message {
    let mut out = Message { fields: rest };
    for entry in entries {
        out.fields.push(Field {
            number: entry_field,
            value: Value::Bytes(entry.message.encode()),
        });
    }
    out
}

/// Remap one table through `edit`.
///
/// `old_starts` and `new_starts` are the paragraph starts either side of the
/// edit; they are what keeps a paragraph table's bookkeeping exact.
fn remap_table(
    table: &Message,
    edit: Edit,
    anchoring: Anchoring,
    old_starts: &[u64],
    new_starts: &[u64],
    old_length: u64,
    new_length: u64,
) -> (Message, EditReport) {
    let (entries, rest, entry_field) = split(table, anchoring);
    let mut report = EditReport::default();

    if anchoring == Anchoring::Range {
        let mut out = Vec::new();
        for mut entry in entries {
            let Some((location, span)) = range_of(&entry.message) else {
                out.push(entry);
                continue;
            };
            let start = edit.remap(location, Anchoring::Run).unwrap_or(edit.at);
            let end = edit
                .remap(location + span, Anchoring::Run)
                .unwrap_or(edit.at);
            if end <= start {
                report.dropped += 1;
                continue;
            }
            if (start, end - start) != (location, span) {
                set_range(&mut entry.message, start, end - start);
                report.moved += 1;
            }
            out.push(entry);
        }
        return (join(out, rest, entry_field), report);
    }

    // A table that had an entry at every paragraph start keeps that shape: the
    // paragraphs an insertion created get one too. A sparse table — and the
    // list-style, para-data and bidi tables in this corpus routinely are one —
    // is left sparse, which is what Pages was observed to do.
    let was_dense = anchoring == Anchoring::Paragraph
        && !old_starts.is_empty()
        && old_starts
            .iter()
            .all(|start| entries.iter().any(|e| e.index == *start));

    let mut out: Vec<Entry> = Vec::new();
    for mut entry in entries {
        // The paragraph entry at the very end of the text is not a paragraph
        // start: it is the slot the style of the paragraph not yet typed comes
        // from, and 1,168 of the corpus's 1,622 storages have one. It is an
        // *end marker*, so it follows the end of the text wherever that goes —
        // which is what [`Edit::remap`] works out for every edit but one.
        // Typing at the very end has `at == old_length`, and the rule that
        // holds a paragraph start still (`index <= at`) then holds the marker
        // at the old end, where the next entry-must-be-a-paragraph-start check
        // drops it. `set-text` on `pages-columns`' text box followed by an
        // insert at the end lost it exactly that way.
        let end_marker = anchoring == Anchoring::Paragraph
            && old_length > 0
            && entry.index == old_length
            && !old_starts.contains(&entry.index);
        let remapped = if end_marker {
            Some(new_length)
        } else {
            edit.remap(entry.index, anchoring)
        };
        let Some(index) = remapped else {
            report.dropped += 1;
            continue;
        };
        // A paragraph entry may only sit at a paragraph start, or at the very
        // end of the text where the style of the next paragraph waits.
        if anchoring == Anchoring::Paragraph && index != new_length && !new_starts.contains(&index)
        {
            report.dropped += 1;
            continue;
        }
        if index != entry.index {
            entry.message.set(1, Value::Varint(index));
            report.moved += 1;
        }
        entry.index = index;
        // Two entries on the same character: the paragraph table keeps the
        // first, because the paragraph the edit began in keeps its style; a run
        // table keeps the last, because a run whose extent has gone should go
        // with it. Both were measured.
        match out.last() {
            Some(last) if last.index == index => match anchoring {
                Anchoring::Paragraph | Anchoring::Character => {
                    report.dropped += 1;
                    continue;
                }
                _ => {
                    out.pop();
                    report.dropped += 1;
                    out.push(entry);
                }
            },
            _ => out.push(entry),
        }
    }

    if was_dense {
        for start in new_starts {
            if out.iter().any(|e| e.index == *start) {
                continue;
            }
            let Some(position) = out.iter().rposition(|e| e.index < *start) else {
                continue;
            };
            let mut message = out[position].message.clone();
            message.set(1, Value::Varint(*start));
            // Pages writes the new paragraph's entry with **no object**: a nil
            // attribute, which is how the format says "whatever was in force
            // here". Observed by having it insert a paragraph break in the
            // middle of a styled paragraph — the entry it added was `{1: 22}`
            // and nothing else. An entry whose second field is a plain number
            // rather than a reference (the paragraph-data tables) keeps its
            // numbers, because those fields are required and a nil one would
            // not be a valid entry.
            if matches!(message.get(2), Some(Value::Bytes(_))) {
                message.clear(2);
            }
            out.insert(
                position + 1,
                Entry {
                    index: *start,
                    message,
                },
            );
            report.added += 1;
        }
    }

    // A run entry at or past the end of the text describes no characters at
    // all. The paragraph tables do carry one there — 1,168 of the corpus's
    // 1,622 storages have it, and it is where the style of a paragraph not yet
    // typed comes from — but a run table does not, and an edit that shortens
    // the text onto one has produced a run of nothing.
    if anchoring == Anchoring::Run {
        let before = out.len();
        out.retain(|entry| entry.index == 0 || entry.index < new_length);
        report.dropped += before - out.len();
    }

    // A run that now says exactly what the run before it says is not a run.
    // Pages goes further and drops a table that has stopped drawing any
    // boundary at all — see [`says_nothing`] — but coalescing is the part that
    // keeps the table from growing an entry per edit. Paragraph tables are left
    // alone: one entry per paragraph is the shape they are meant to have, and
    // two neighbouring paragraphs sharing a style is ordinary.
    if anchoring == Anchoring::Run {
        let mut coalesced: Vec<Entry> = Vec::new();
        for entry in out {
            let bare = |e: &Entry| {
                let mut m = e.message.clone();
                m.clear(1);
                m.encode()
            };
            if coalesced
                .last()
                .is_some_and(|last| bare(last) == bare(&entry))
            {
                report.dropped += 1;
                continue;
            }
            coalesced.push(entry);
        }
        out = coalesced;
    }

    (join(out, rest, entry_field), report)
}

/// Has a run table stopped saying anything — one entry, at the start of the
/// text, with no attribute on it?
///
/// Pages removes the field outright in that case: made to delete the whole of
/// the only styled run in a storage, it left the storage with no character
/// table at all rather than with a table asserting "nothing, from 0".
fn says_nothing(table: &Message, anchoring: Anchoring) -> bool {
    if anchoring != Anchoring::Run {
        return false;
    }
    let (entries, rest, _) = split(table, anchoring);
    rest.is_empty()
        && entries.len() == 1
        && entries[0].index == 0
        && entries[0].message.fields.len() == 1
}

/// The `(index, object)` of every entry of a table, in order.
///
/// The object is `None` on a **nil attribute** — an entry that carries a
/// character index and nothing else. Those are not corrupt and not empty: they
/// terminate the run before them (a hyperlink that stops short of the end of the
/// text has one), or assert that there is deliberately nothing here (the
/// drop-cap table's `{0}`), or say that a newly created paragraph takes whatever
/// was in force (which is what Pages writes when it splits one).
pub fn entry_indices(table: &Message, anchoring: Anchoring) -> Vec<(u64, Option<u64>)> {
    split(table, anchoring)
        .0
        .into_iter()
        .map(|entry| {
            let object = entry
                .message
                .bytes(2)
                .and_then(crate::pb::decode_nested)
                .and_then(|r| crate::style::reference_target(&r));
            (entry.index, object)
        })
        .collect()
}

/// The `(index, first)` of every entry of a paragraph-data table.
///
/// `first` is the **list level**, counted from 0. Confirmed against Apple's
/// `60_Academic_Modern_PM` Keynote theme, whose "Body Level One … Body Level
/// Five" storage carries `first` 0, 1, 2, 3, 4 on its five paragraph starts,
/// with `table_list_style` naming a style override at each of the same indices.
pub fn para_data(table: &Message) -> Vec<(u64, Option<u64>)> {
    split(table, Anchoring::Paragraph)
        .0
        .into_iter()
        .map(|entry| (entry.index, entry.message.varint(2)))
        .collect()
}

/// The `(location, length)` of every entry of an overlapping-field table.
pub fn ranges(table: &Message) -> Vec<(u64, u64)> {
    split(table, Anchoring::Range)
        .0
        .iter()
        .filter_map(|entry| range_of(&entry.message))
        .collect()
}

/// A `TSP.Range` — `{1: location, 2: length}` — inside an entry's field 1.
fn range_of(entry: &Message) -> Option<(u64, u64)> {
    let range = crate::pb::decode_nested(entry.bytes(1)?)?;
    Some((range.varint(1)?, range.varint(2).unwrap_or(0)))
}

fn set_range(entry: &mut Message, location: u64, length: u64) {
    let mut range = entry
        .bytes(1)
        .and_then(crate::pb::decode_nested)
        .unwrap_or_default();
    range.set(1, Value::Varint(location));
    range.set(2, Value::Varint(length));
    entry.set(1, Value::Bytes(range.encode()));
}

/// Every entry of `storage`'s character-anchored tables that the removed range
/// would destroy, as `(field, index, object)`.
///
/// These are the ones an edit cannot simply drop: the character *is* the
/// attachment, and deleting it is how Pages deletes an anchored image — which it
/// then also removes from the drawable list, the z-order and the media registry.
/// This crate does none of that, so it refuses instead.
///
/// **The character itself is not consulted, and that is the fix for a hole this
/// had.** The check used to pass unless the character was `U+FFFC` or `U+0004`;
/// a footnote mark is [`U+000E`](ANCHOR_CHARACTERS), so a delete across one
/// exited cleanly, orphaned the `TSWP.FootnoteReferenceAttachmentArchive` and
/// the kind-2 storage hanging off it, and left `iwork check` with nothing to
/// find. A character-anchored entry whose character goes is a destroyed entry
/// whatever the character was: if this crate does not recognise it, that is a
/// reason to refuse rather than a licence to proceed. `U+0004` was in the list
/// for no one: a section entry is paragraph-anchored, and
/// [`destroyed_sections`] is what sees it.
///
/// Every occurrence of the field is read, because [`apply`] rewrites every one.
pub fn destroyed_anchors(storage: &Message, edit: Edit) -> Vec<(u32, u64, Option<u64>)> {
    let mut out = Vec::new();
    for spec in TABLES
        .iter()
        .filter(|t| t.anchoring == Anchoring::Character)
    {
        for table in tables_at(storage, spec.field) {
            for entry in split(&table, Anchoring::Character).0 {
                if edit.remap(entry.index, Anchoring::Character).is_some() {
                    continue;
                }
                let object = entry
                    .message
                    .bytes(2)
                    .and_then(crate::pb::decode_nested)
                    .and_then(|r| crate::style::reference_target(&r));
                out.push((spec.field, entry.index, object));
            }
        }
    }
    out
}

/// Every occurrence of one table field, decoded.
///
/// [`Message::bytes`](crate::pb::Message::bytes) answers with the *first*
/// occurrence, which is the shape every storage in the corpus has; [`apply`]
/// rewrites all of them, so everything that decides whether an edit is allowed
/// has to read all of them too, or a second occurrence is remapped without ever
/// having been checked.
fn tables_at(storage: &Message, field: u32) -> Vec<Message> {
    storage
        .all(field)
        .filter_map(|value| match value {
            Value::Bytes(raw) => crate::pb::decode_nested(raw),
            _ => None,
        })
        .collect()
}

/// Every section the removed range would destroy, as `(index, section)`.
///
/// A section is not anchored like an attachment — its entry sits at a
/// paragraph start, on the character *after* the `U+0004` that begins it — so
/// [`destroyed_anchors`] does not see it. What destroys a section is deleting
/// **the break**, and the break is at `index - 1`.
///
/// The entry at index 0 is not at risk: the first section of a body begins at
/// character 0 with no break in front of it, and still begins wherever the
/// text now begins.
///
/// This is separate from [`destroyed_anchors`] because the answer is different
/// in kind. Deleting the `U+FFFC` an image hangs off has an observed
/// consequence — Pages deletes the image — and refusing is refusing to
/// reproduce it. Deleting a section break has **no observed consequence at
/// all**: Pages will not do it. `delete section 2` answers -10000, there is no
/// `make new section`, the menu that would do it needs a window, and setting a
/// section's body text to the empty string leaves the break exactly where it
/// was with a zero-length section behind it. So what a merge should do to the
/// two `TP.SectionArchive`s, their three `TP.SectionTemplateArchive`s each,
/// their eighteen header and footer storages apiece, their guide storages and
/// their background fills is not known, and this crate does not guess.
pub fn destroyed_sections(storage: &Message, text: &str, edit: Edit) -> Vec<(u64, Option<u64>)> {
    let units: Vec<u16> = text.encode_utf16().collect();
    let mut out = Vec::new();
    for table in tables_at(storage, SECTION_TABLE) {
        for (index, object) in entry_indices(&table, Anchoring::Paragraph) {
            if index == 0 {
                continue;
            }
            let break_at = index - 1;
            if units.get(break_at as usize).copied() != Some(0x0004) {
                continue;
            }
            if break_at >= edit.at && break_at < edit.end() {
                out.push((index, object));
            }
        }
    }
    out
}

/// Why an edit to a storage is refused — everything [`apply`] requires of its
/// caller, in one value.
///
/// [`crate::Document::replace_text`] turns each of these into the
/// [`crate::Error`] that names it; the storage identifier, which this module
/// does not have, is what it adds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// A length-delimited field that is not an attribute table this crate
    /// knows — see [`unknown_table`].
    UnknownTable(u32),
    /// A known attribute table whose bytes do not decode — see
    /// [`undecodable_table`].
    UndecodableTable(u32),
    /// Field 3 is not valid UTF-8 — see [`text_is_utf8`].
    InvalidText,
    /// The storage carries `table_insertion` or `table_deletion` — see
    /// [`CHANGE_TABLES`].
    TrackedChanges(u32),
    /// The removed range covers the character an object is anchored to — see
    /// [`destroyed_anchors`].
    AnchoredObject {
        field: u32,
        index: u64,
        object: Option<u64>,
    },
    /// The removed range covers the `U+0004` a section begins after — see
    /// [`destroyed_sections`].
    SectionBreak {
        /// Index of the section's first character; the break is one before it.
        index: u64,
        section: Option<u64>,
    },
}

impl Refusal {
    /// The error a `Document` reports for this refusal, given the storage it
    /// was found in.
    pub fn into_error(self, storage: u64) -> crate::Error {
        match self {
            Refusal::UnknownTable(field) => crate::Error::UnknownAttributeTable { storage, field },
            Refusal::UndecodableTable(field) => {
                crate::Error::UndecodableAttributeTable { storage, field }
            }
            Refusal::InvalidText => crate::Error::InvalidText { storage },
            Refusal::TrackedChanges(field) => crate::Error::TrackedChanges { storage, field },
            Refusal::AnchoredObject {
                field,
                index,
                object,
            } => crate::Error::AnchoredObject {
                storage,
                index,
                table: table(field).map(|t| t.name).unwrap_or("an anchor table"),
                object,
            },
            Refusal::SectionBreak { index, section } => crate::Error::SectionBreak {
                storage,
                index: index - 1,
                section,
            },
        }
    }
}

/// Everything that must be true of `storage` before [`apply`] may touch it.
///
/// One function rather than five call sites, so that the contract [`apply`]
/// documents and the checks a caller performs cannot drift apart: `apply`
/// asserts this in debug builds, and [`crate::Document::replace_text`] is the
/// caller that turns it into an error.
///
/// The order is the order the answers are worth having: what the crate cannot
/// read at all, then what it will not edit at all, then what this particular
/// edit would destroy.
pub fn refusal(storage: &Message, edit: Edit) -> Option<Refusal> {
    if let Some(field) = unknown_table(storage) {
        return Some(Refusal::UnknownTable(field));
    }
    if let Some(field) = undecodable_table(storage) {
        return Some(Refusal::UndecodableTable(field));
    }
    if !text_is_utf8(storage) {
        return Some(Refusal::InvalidText);
    }
    // Change tracking, before anything is measured. Both tables *look* like run
    // tables and would remap without complaint; `table_deletion` covers
    // characters that are still in the text and are not going to be shown,
    // which is a different thing from a style run and would need a probe
    // nothing here can perform. See `crate::annotations`.
    if let Some(field) = CHANGE_TABLES
        .iter()
        .find(|field| storage.get(**field).is_some())
    {
        return Some(Refusal::TrackedChanges(*field));
    }
    if let Some((field, index, object)) = destroyed_anchors(storage, edit).into_iter().next() {
        return Some(Refusal::AnchoredObject {
            field,
            index,
            object,
        });
    }
    let text = read(storage);
    if let Some((index, section)) = destroyed_sections(storage, &text, edit).into_iter().next() {
        return Some(Refusal::SectionBreak { index, section });
    }
    None
}

/// Apply `edit` to a storage archive, remapping every attribute table it
/// carries.
///
/// `new_text` must be the storage's text with the edit already performed;
/// [`crate::Document::replace_text`] is what computes it. This function is the
/// part that has to be right about the tables, and is separated so that it can
/// be tested on storages built by hand.
///
/// **The caller must have found [`refusal`] `None` first.** That is the whole
/// precondition — an unknown table, a table that does not decode, text that is
/// not UTF-8, change tracking, a destroyed anchor, a destroyed section break —
/// and it is asserted here in debug builds rather than only described, because
/// a contract listing three of the six checks is how the other three come to be
/// skipped.
pub fn apply(storage: &mut Message, edit: Edit, new_text: &str) -> EditReport {
    debug_assert!(
        refusal(storage, edit).is_none(),
        "apply() was called on a storage it must refuse: {:?}",
        refusal(storage, edit)
    );
    let old_text = read(storage);
    let old_starts: Vec<u64> = paragraph_ranges(&old_text)
        .iter()
        .map(|r| r.start)
        .collect();
    let new_starts: Vec<u64> = paragraph_ranges(new_text).iter().map(|r| r.start).collect();
    let old_length = length(&old_text);
    let new_length = length(new_text);

    let mut report = EditReport::default();
    let mut emptied: Vec<u32> = Vec::new();
    for field in &mut storage.fields {
        let Some(spec) = table(field.number) else {
            continue;
        };
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(decoded) = crate::pb::decode_nested(raw) else {
            continue;
        };
        let (rewritten, one) = remap_table(
            &decoded,
            edit,
            spec.anchoring,
            &old_starts,
            &new_starts,
            old_length,
            new_length,
        );
        if says_nothing(&rewritten, spec.anchoring) {
            emptied.push(field.number);
        }
        let encoded = rewritten.encode();
        if encoded != *raw {
            report.tables.push(field.number);
        }
        report.moved += one.moved;
        report.dropped += one.dropped;
        report.added += one.added;
        field.value = Value::Bytes(encoded);
    }
    storage.fields.retain(|f| !emptied.contains(&f.number));

    set_storage_text(storage, new_text);
    report
}

/// Put `text` into a storage as the single run iWork writes.
///
/// Field 3 is repeated in the schema and "in practice exactly one element";
/// several runs are collapsed into one. The field goes in **field order**,
/// because a storage that had no text at all would otherwise get its text
/// appended after every other field, which is not how iWork writes one.
fn set_storage_text(storage: &mut Message, text: &str) {
    let mut seen = false;
    storage.fields.retain(|field| {
        if field.number != 3 {
            return true;
        }
        let keep = !seen;
        seen = true;
        keep
    });
    storage.set_in_order(3, Value::Bytes(text.as_bytes().to_vec()));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An object-table entry: `{1: index, 2: {1: style}}`.
    fn entry(index: u64, style: Option<u64>) -> Message {
        let mut entry = Message::default();
        entry.set(1, Value::Varint(index));
        if let Some(style) = style {
            entry.set(2, Value::Bytes(crate::style::reference(style).encode()));
        }
        entry
    }

    /// A paragraph-data entry: `{1: index, 2: first, 3: second}`.
    fn para_entry(index: u64, first: u64, second: u64) -> Message {
        let mut entry = Message::default();
        entry.set(1, Value::Varint(index));
        entry.set(2, Value::Varint(first));
        entry.set(3, Value::Varint(second));
        entry
    }

    /// A range entry: `{1: {1: location, 2: length}, 2: {1: object}}`.
    fn range_entry(location: u64, length: u64, object: u64) -> Message {
        let mut entry = Message::default();
        set_range(&mut entry, location, length);
        entry.set(2, Value::Bytes(crate::style::reference(object).encode()));
        entry
    }

    fn table_of(entries: Vec<Message>) -> Value {
        let mut table = Message::default();
        for entry in entries {
            table.fields.push(Field {
                number: 1,
                value: Value::Bytes(entry.encode()),
            });
        }
        Value::Bytes(table.encode())
    }

    /// A storage with the given text and, for each `(field, entries)`, a table.
    fn storage(text: &str, tables: Vec<(u32, Vec<Message>)>) -> Message {
        let mut storage = Message::default();
        storage.set(3, Value::Bytes(text.as_bytes().to_vec()));
        for (field, entries) in tables {
            storage.set_in_order(field, table_of(entries));
        }
        storage
    }

    /// The `(index, object)` of every entry of one table.
    fn shape(storage: &Message, field: u32) -> Vec<(u64, Option<u64>)> {
        let Some(decoded) = storage.bytes(field).and_then(crate::pb::decode_nested) else {
            return Vec::new();
        };
        let anchoring = table(field).map(|t| t.anchoring).unwrap_or(Anchoring::Run);
        split(&decoded, anchoring)
            .0
            .into_iter()
            .map(|e| {
                let object = e
                    .message
                    .bytes(2)
                    .and_then(crate::pb::decode_nested)
                    .and_then(|r| crate::style::reference_target(&r));
                (e.index, object)
            })
            .collect()
    }

    /// Replace `[at, at+removed)` with `insert`, the way `Document` does.
    fn edit(storage: &mut Message, at: u64, removed: u64, insert: &str) -> EditReport {
        let old = read(storage);
        let from = utf16_offset(&old, at).unwrap();
        let to = utf16_offset(&old, at + removed).unwrap();
        let text = format!("{}{insert}{}", &old[..from], &old[to..]);
        apply(
            storage,
            Edit {
                at,
                removed,
                inserted: length(insert),
            },
            &text,
        )
    }

    #[test]
    fn placeholder_only_storages_are_not_content() {
        assert!(!has_content("\u{FFFC}"));
        assert!(!has_content("\u{4}"));
        assert!(!has_content("  \n\t "));
        assert!(has_content("Bossard"));
        assert!(has_content("\u{FFFC} Abbildung 1"));
    }

    #[test]
    fn reads_concatenated_runs() {
        let mut message = Message::default();
        message.fields.push(Field {
            number: 3,
            value: Value::Bytes(b"Hello, ".to_vec()),
        });
        message.fields.push(Field {
            number: 3,
            value: Value::Bytes(b"world".to_vec()),
        });
        assert_eq!(read(&message), "Hello, world");
    }

    #[test]
    fn every_table_the_schema_names_is_here_once() {
        let mut fields: Vec<u32> = TABLES.iter().map(|t| t.field).collect();
        let seen = fields.clone();
        fields.sort_unstable();
        fields.dedup();
        assert_eq!(fields, seen, "the table list is sorted and has no repeats");
        // Fields 5–28 are tables but for 10 (`in_document`, a bool) and 13,
        // which is a gap in every version of TSWPArchives.proto.
        for field in 5..=28u32 {
            let known = table(field).is_some();
            assert_eq!(known, field != 10 && field != 13, "field {field}");
        }
    }

    #[test]
    fn a_field_the_crate_does_not_know_is_found() {
        let mut s = storage("abc", vec![(5, vec![entry(0, Some(9))])]);
        assert_eq!(unknown_table(&s), None);
        s.set_in_order(4, Value::Varint(1)); // has_itext: plain, and harmless
        assert_eq!(unknown_table(&s), None);
        s.set_in_order(30, Value::Bytes(vec![8, 0]));
        assert_eq!(unknown_table(&s), Some(30));
    }

    // -- UTF-16 discipline ---------------------------------------------------

    #[test]
    fn offsets_count_utf16_code_units_and_refuse_to_split_a_pair() {
        let text = "a\u{1F600}b"; // a, two units of emoji, b — four units
        assert_eq!(length(text), 4);
        assert_eq!(utf16_offset(text, 0), Some(0));
        assert_eq!(utf16_offset(text, 1), Some(1));
        assert_eq!(utf16_offset(text, 2), None, "half an emoji");
        assert_eq!(utf16_offset(text, 3), Some(5));
        assert_eq!(utf16_offset(text, 4), Some(6));
        assert_eq!(utf16_offset(text, 5), None, "past the end");
    }

    #[test]
    fn paragraphs_count_utf16_code_units() {
        assert_eq!(paragraph_ranges("\u{1F600}\nx"), vec![0..3, 3..4]);
        assert_eq!(length("\u{1F600}\nx"), 4);
    }

    // -- paragraph breaks ----------------------------------------------------

    #[test]
    fn a_layout_break_ends_a_paragraph_too() {
        // "…ende\n\n\u{5}Features" — the next paragraph starts on the F.
        assert_eq!(
            paragraph_ranges("ab\n\n\u{5}Fe"),
            vec![0..3, 3..4, 4..5, 5..7]
        );
    }

    /// The break the apps actually write: AppleScript's `return` is a carriage
    /// return, and Pages and Keynote store it as the paragraph separator.
    #[test]
    fn a_carriage_return_ends_a_paragraph() {
        assert_eq!(paragraph_ranges("ab\rcd\r"), vec![0..3, 3..6]);
        assert_eq!(paragraph_ranges("Überschrift\rEin"), vec![0..12, 12..15]);
    }

    #[test]
    fn a_section_break_ends_a_paragraph_too() {
        // "…123-4567\n\u{4}Company Name\n" — the next paragraph starts on the C.
        assert_eq!(paragraph_ranges("67\n\u{4}Co\n"), vec![0..3, 3..4, 4..7]);
    }

    #[test]
    fn paragraphs_tile_the_text_and_keep_their_newline() {
        assert_eq!(paragraph_ranges("one\ntwo\nthree"), vec![0..4, 4..8, 8..13]);
        assert_eq!(paragraph_ranges("one\ntwo\n"), vec![0..4, 4..8]);
        assert_eq!(paragraph_ranges("no newline"), vec![0..10]);
        assert_eq!(paragraph_ranges("\n"), vec![0..1]);
        assert!(paragraph_ranges("").is_empty());
    }

    // -- what Pages does, reproduced ----------------------------------------
    //
    // The four fixtures below are the archives Pages wrote when it was made to
    // perform these very edits on `pages-styled`, whose body is
    // "Überschrift\rEin roter…\rEin kursiver…\rEin ganz…" — 171 units with
    // paragraph runs at 0, 12, 74 and 128.

    fn styled() -> Message {
        storage(
            "Überschrift\rEin roter Absatz, damit die Farbe irgendwo im Dokument \
             steht.\rEin kursiver Absatz, gesetzt über den Schriftschnitt.\rEin ganz \
             gewöhnlicher Absatz zum Vergleich.",
            vec![(
                5,
                vec![
                    entry(0, Some(1732648)),
                    entry(12, Some(1732651)),
                    entry(74, Some(1732654)),
                    entry(128, Some(1731511)),
                ],
            )],
        )
    }

    /// Pages, deleting `[5, 20)` — across the first paragraph break: the entry
    /// at 12 goes, because its paragraph merged into the one before it.
    #[test]
    fn a_delete_across_a_paragraph_break_drops_the_entry() {
        let mut s = styled();
        assert_eq!(length(&read(&s)), 171);
        let report = edit(&mut s, 5, 15, "");
        assert_eq!(
            shape(&s, 5),
            vec![
                (0, Some(1732648)),
                (59, Some(1732654)),
                (113, Some(1731511))
            ]
        );
        assert_eq!((report.moved, report.dropped, report.added), (2, 1, 0));
    }

    /// Pages, deleting `[12, 20)` — starting exactly at a paragraph start: the
    /// entry there stays where it is.
    #[test]
    fn a_delete_beginning_at_a_paragraph_start_keeps_its_entry() {
        let mut s = styled();
        edit(&mut s, 12, 8, "");
        assert_eq!(
            shape(&s, 5),
            vec![
                (0, Some(1732648)),
                (12, Some(1732651)),
                (66, Some(1732654)),
                (120, Some(1731511)),
            ]
        );
    }

    /// Pages, deleting `[12, 74)` — one whole paragraph, its break included.
    /// Two entries land on 12 and the **first** wins: the paragraph the edit
    /// began in keeps its style, and the one that followed loses its own.
    #[test]
    fn a_deleted_paragraph_leaves_its_style_behind() {
        let mut s = styled();
        edit(&mut s, 12, 62, "");
        assert_eq!(
            shape(&s, 5),
            vec![(0, Some(1732648)), (12, Some(1732651)), (66, Some(1731511))]
        );
    }

    /// Pages, deleting `[30, 90)` — from the middle of one paragraph to the
    /// middle of the next.
    #[test]
    fn a_delete_spanning_a_boundary_drops_the_entry_between() {
        let mut s = styled();
        edit(&mut s, 30, 60, "");
        assert_eq!(
            shape(&s, 5),
            vec![(0, Some(1732648)), (12, Some(1732651)), (68, Some(1731511))]
        );
    }

    /// Pages, inserting eight units at 21: everything past it moves, and
    /// nothing else changes.
    #[test]
    fn an_insert_moves_what_follows_it() {
        let mut s = styled();
        let report = edit(&mut s, 21, 0, "INSERTED");
        assert_eq!(
            shape(&s, 5),
            vec![
                (0, Some(1732648)),
                (12, Some(1732651)),
                (82, Some(1732654)),
                (136, Some(1731511)),
            ]
        );
        assert_eq!((report.moved, report.dropped, report.added), (2, 0, 0));
    }

    /// Pages, inserting a paragraph break: the new paragraph gets an entry of
    /// its own **with no style reference** — a nil attribute, which is the
    /// format's way of saying "whatever was in force here".
    #[test]
    fn an_inserted_paragraph_break_gets_a_nil_entry() {
        let mut s = styled();
        let report = edit(&mut s, 21, 0, "\rNEU");
        assert_eq!(
            shape(&s, 5),
            vec![
                (0, Some(1732648)),
                (12, Some(1732651)),
                (22, None),
                (78, Some(1732654)),
                (132, Some(1731511)),
            ]
        );
        assert_eq!(report.added, 1);
    }

    // -- run tables ----------------------------------------------------------

    /// A character run is a range, and a range that is wholly deleted goes
    /// away: Pages deleted `[19, 30)` from a storage whose character table read
    /// `[0 nil, 19 bold, 30 nil]` and was left with nothing at all.
    #[test]
    fn a_run_deleted_in_full_disappears() {
        let mut s = storage(
            &"x".repeat(60),
            vec![(
                8,
                vec![entry(0, None), entry(19, Some(700)), entry(30, None)],
            )],
        );
        edit(&mut s, 19, 11, "");
        assert!(
            s.get(8).is_none(),
            "Pages removes a table that says nothing"
        );
    }

    /// Pages, deleting `[15, 25)` across the start of a run: the run's start
    /// comes back to the edit and the run keeps what is left of it.
    #[test]
    fn a_run_starting_inside_the_delete_comes_back_to_it() {
        let mut s = storage(
            &"x".repeat(60),
            vec![(
                8,
                vec![entry(0, None), entry(19, Some(700)), entry(30, None)],
            )],
        );
        edit(&mut s, 15, 10, "");
        assert_eq!(shape(&s, 8), vec![(0, None), (15, Some(700)), (20, None)]);
    }

    /// Pages, deleting `[24, 40)` across the *end* of a run.
    #[test]
    fn a_run_ending_inside_the_delete_is_cut_short() {
        let mut s = storage(
            &"x".repeat(60),
            vec![(
                8,
                vec![entry(0, None), entry(19, Some(700)), entry(30, None)],
            )],
        );
        edit(&mut s, 24, 16, "");
        assert_eq!(shape(&s, 8), vec![(0, None), (19, Some(700)), (24, None)]);
    }

    /// Text typed at a run boundary joins the run **before** it: Pages moved a
    /// run's start from 19 to 22 when four units arrived there.
    #[test]
    fn text_inserted_at_a_run_boundary_joins_the_run_before_it() {
        let mut s = storage(
            &"x".repeat(60),
            vec![(
                8,
                vec![entry(0, None), entry(19, Some(700)), entry(30, None)],
            )],
        );
        edit(&mut s, 19, 0, "INS");
        assert_eq!(shape(&s, 8), vec![(0, None), (22, Some(700)), (33, None)]);
    }

    // -- the other shapes ----------------------------------------------------

    /// A paragraph-data entry's second and third fields are required, so a
    /// paragraph created inside one copies them rather than being made nil.
    #[test]
    fn paragraph_data_entries_keep_their_numbers() {
        let mut s = storage(
            "eins\rzwei\r",
            vec![(6, vec![para_entry(0, 1, 0), para_entry(5, 2, 0)])],
        );
        edit(&mut s, 7, 0, "\rdrei");
        let table = Message::decode(s.bytes(6).unwrap()).unwrap();
        let entries: Vec<(u64, u64, u64)> = split(&table, Anchoring::Paragraph)
            .0
            .iter()
            .map(|e| {
                (
                    e.index,
                    e.message.varint(2).unwrap(),
                    e.message.varint(3).unwrap(),
                )
            })
            .collect();
        assert_eq!(entries, vec![(0, 1, 0), (5, 2, 0), (8, 2, 0)]);
    }

    /// A sparse per-paragraph table stays sparse. Pages leaves the list-style,
    /// bidi and drop-cap tables of a multi-paragraph storage holding one entry
    /// each, and adding entries it never writes is not an improvement.
    #[test]
    fn a_sparse_paragraph_table_gains_nothing() {
        let mut s = storage("eins\rzwei\rdrei", vec![(7, vec![entry(0, Some(500))])]);
        edit(&mut s, 7, 0, "\rneu");
        assert_eq!(shape(&s, 7), vec![(0, Some(500))]);
    }

    /// An overlapping-highlight entry carries its own range, and one whose
    /// range is wholly deleted goes with it.
    #[test]
    fn range_entries_are_remapped_by_both_ends() {
        let mut s = storage(
            &"x".repeat(40),
            vec![(25, vec![range_entry(5, 10, 900), range_entry(20, 5, 901)])],
        );
        edit(&mut s, 20, 5, "");
        let table = Message::decode(s.bytes(25).unwrap()).unwrap();
        let ranges: Vec<(u64, u64)> = split(&table, Anchoring::Range)
            .0
            .iter()
            .filter_map(|e| range_of(&e.message))
            .collect();
        assert_eq!(ranges, vec![(5, 10)], "the second range was deleted whole");

        let mut s = storage(&"x".repeat(40), vec![(25, vec![range_entry(5, 10, 900)])]);
        edit(&mut s, 8, 4, "");
        let table = Message::decode(s.bytes(25).unwrap()).unwrap();
        let ranges: Vec<(u64, u64)> = split(&table, Anchoring::Range)
            .0
            .iter()
            .filter_map(|e| range_of(&e.message))
            .collect();
        assert_eq!(ranges, vec![(5, 6)]);
    }

    /// An attachment's character is the attachment. A delete that covers it is
    /// reported, so the caller can refuse rather than detach an image.
    #[test]
    fn a_deleted_attachment_character_is_reported() {
        let text = "Company Name\u{FFFC}\nProject";
        let s = storage(text, vec![(9, vec![entry(12, Some(57435))])]);
        let covered = Edit {
            at: 9,
            removed: 6,
            inserted: 0,
        };
        assert_eq!(destroyed_anchors(&s, covered), vec![(9, 12, Some(57435))]);
        let clear = Edit {
            at: 0,
            removed: 5,
            inserted: 0,
        };
        assert!(destroyed_anchors(&s, clear).is_empty());
    }

    /// A footnote mark is a `U+000E`, and the character is just as much the
    /// object as an attachment's `U+FFFC` is. The check that read the character
    /// and let anything else through is the hole this closes: it knew `U+FFFC`
    /// and `U+0004` — and `U+0004` was never reachable from a character-anchored
    /// table in the first place, the section table being paragraph-anchored.
    #[test]
    fn a_deleted_footnote_mark_is_reported() {
        let text = "Ein Absatz mit einer\u{E} Fussnote.";
        let s = storage(text, vec![(FOOTNOTE_TABLE, vec![entry(20, Some(1732790))])]);
        let covered = Edit {
            at: 18,
            removed: 6,
            inserted: 0,
        };
        assert_eq!(
            destroyed_anchors(&s, covered),
            vec![(FOOTNOTE_TABLE, 20, Some(1732790))]
        );
        // And an entry on a character this crate has never seen an anchor on is
        // refused too: not recognising it is the reason to stop, not a reason
        // to carry on.
        let odd = storage(
            "Ein ganz gewöhnlicher Satz.",
            vec![(9, vec![entry(4, Some(7))])],
        );
        let over_it = Edit {
            at: 3,
            removed: 3,
            inserted: 0,
        };
        assert_eq!(destroyed_anchors(&odd, over_it), vec![(9, 4, Some(7))]);
    }

    /// A table field may occur more than once — the storage archive's tables are
    /// singular in every document in the corpus, but [`apply`] rewrites every
    /// occurrence, so everything that decides whether an edit is allowed reads
    /// every occurrence too.
    #[test]
    fn every_occurrence_of_an_anchor_table_is_checked() {
        let text = "Company Name\u{FFFC}\nProject";
        let mut s = storage(text, vec![(9, vec![entry(3, Some(1))])]);
        s.fields.push(Field {
            number: 9,
            value: table_of(vec![entry(12, Some(57435))]),
        });
        let covered = Edit {
            at: 9,
            removed: 6,
            inserted: 0,
        };
        assert_eq!(
            destroyed_anchors(&s, covered),
            vec![(9, 12, Some(57435))],
            "the second occurrence of field 9 is checked like the first"
        );

        // The section table, the same way round.
        let text = "erste\u{4}zweite";
        let mut s = storage(text, vec![(SECTION_TABLE, vec![entry(0, Some(900))])]);
        s.fields.push(Field {
            number: SECTION_TABLE,
            value: table_of(vec![entry(6, Some(901))]),
        });
        let over_the_break = Edit {
            at: 3,
            removed: 4,
            inserted: 0,
        };
        assert_eq!(
            destroyed_sections(&s, text, over_the_break),
            vec![(6, Some(901))]
        );
    }

    /// A table this crate knows by number and cannot decode is a refusal, not a
    /// table to step over: [`apply`] would remap every other table in the
    /// storage and leave this one anchored to characters that have moved.
    #[test]
    fn a_table_that_does_not_decode_is_refused() {
        let mut s = storage("abcdef", vec![(5, vec![entry(0, Some(9))])]);
        let nothing = Edit {
            at: 0,
            removed: 0,
            inserted: 0,
        };
        assert_eq!(undecodable_table(&s), None);
        assert_eq!(refusal(&s, nothing), None);

        // A short string that parses as a message but does not re-encode to the
        // same bytes — the case `decode_nested`'s round trip exists for.
        s.set(8, Value::Bytes(b"Grosse Uberschrift".to_vec()));
        assert_eq!(undecodable_table(&s), Some(8));
        assert_eq!(refusal(&s, nothing), Some(Refusal::UndecodableTable(8)));

        // An empty table field is not that case: an empty message is a table
        // with no entries and there is nothing in it to remap.
        s.set(8, Value::Bytes(Vec::new()));
        assert_eq!(undecodable_table(&s), None);
        assert_eq!(refusal(&s, nothing), None);
    }

    /// Text that is not UTF-8 is read lossily, and writing a lossy reading back
    /// replaces every ill-formed sequence with `U+FFFD` for good.
    #[test]
    fn text_that_is_not_utf8_is_refused() {
        let mut s = storage("abcdef", vec![(5, vec![entry(0, Some(9))])]);
        assert!(text_is_utf8(&s));
        s.set(3, Value::Bytes(vec![b'a', 0xFF, 0xFE, b'b']));
        assert!(!text_is_utf8(&s));
        assert_eq!(
            refusal(
                &s,
                Edit {
                    at: 0,
                    removed: 1,
                    inserted: 0
                }
            ),
            Some(Refusal::InvalidText)
        );
        assert_eq!(read(&s), "a\u{FFFD}\u{FFFD}b", "the reader stays lossy");
    }

    /// The contract is enforced, not only described: `apply` on a storage
    /// [`refusal`] would refuse is a bug in the caller, and a debug build says
    /// so rather than remapping half the storage.
    #[test]
    #[should_panic(expected = "must refuse")]
    #[cfg(debug_assertions)]
    fn apply_asserts_its_preconditions() {
        let text = "Company Name\u{FFFC}\nProject";
        let mut s = storage(text, vec![(9, vec![entry(12, Some(57435))])]);
        apply(
            &mut s,
            Edit {
                at: 9,
                removed: 6,
                inserted: 0,
            },
            "Company N\nProject",
        );
    }

    /// The paragraph entry at the end of the text is an end marker, not a
    /// paragraph start, and it follows the end of the text — including when the
    /// insertion happens exactly at it, which is where it used to be dropped.
    /// `pages-columns`' text box is the document this came from.
    #[test]
    fn the_end_of_text_entry_moves_to_the_new_end() {
        let mut s = storage(
            "Hallo",
            vec![(5, vec![entry(0, Some(56820)), entry(5, None)])],
        );
        edit(&mut s, 5, 0, " Welt");
        assert_eq!(shape(&s, 5), vec![(0, Some(56820)), (10, None)]);

        // Typing into the middle leaves it at the end just the same.
        edit(&mut s, 2, 0, "xx");
        assert_eq!(shape(&s, 5), vec![(0, Some(56820)), (12, None)]);

        // And a storage that is empty has no end marker to move: the entry at 0
        // is the style of the paragraph being typed, and it stays at 0.
        let mut s = storage("", vec![(5, vec![entry(0, Some(56820))])]);
        edit(&mut s, 0, 0, "Neu");
        assert_eq!(shape(&s, 5), vec![(0, Some(56820))]);
    }

    /// Deleting the `U+0004` in front of a section is deleting the section,
    /// and it is not [`destroyed_anchors`] that sees it: the entry sits at a
    /// paragraph start, one character *past* the break.
    #[test]
    fn deleting_a_section_break_is_reported_and_deleting_around_it_is_not() {
        let text = "erste\u{4}zweite\u{4}dritte";
        let s = storage(
            text,
            vec![(
                17,
                vec![
                    entry(0, Some(900)),
                    entry(6, Some(901)),
                    entry(13, Some(902)),
                ],
            )],
        );
        let over_the_break = Edit {
            at: 3,
            removed: 4,
            inserted: 0,
        };
        assert_eq!(
            destroyed_sections(&s, text, over_the_break),
            vec![(6, Some(901))]
        );
        // Exactly the break, and nothing else.
        let just_the_break = Edit {
            at: 12,
            removed: 1,
            inserted: 0,
        };
        assert_eq!(
            destroyed_sections(&s, text, just_the_break),
            vec![(13, Some(902))]
        );
        // Up to the break but not over it.
        let short_of_it = Edit {
            at: 1,
            removed: 4,
            inserted: 0,
        };
        assert!(destroyed_sections(&s, text, short_of_it).is_empty());
        // And `destroyed_anchors` never sees any of this: the section table is
        // paragraph-anchored, and this is why the check is its own function.
        assert!(destroyed_anchors(&s, over_the_break).is_empty());
    }

    /// The first section of a Pages body starts at character 0 with no break
    /// character in front of it, so an edit at the start of the text does not
    /// destroy it — it still begins wherever the text now begins.
    #[test]
    fn a_section_that_begins_at_the_first_character_survives() {
        let text = "Company Name\nmore";
        let s = storage(text, vec![(17, vec![entry(0, Some(57296))])]);
        let edit = Edit {
            at: 0,
            removed: 5,
            inserted: 0,
        };
        assert!(destroyed_anchors(&s, edit).is_empty());
    }

    // -- housekeeping --------------------------------------------------------

    /// Text goes where iWork puts it even when the storage had none — an empty
    /// placeholder in a Keynote slide is exactly that case.
    #[test]
    fn text_added_to_an_empty_storage_lands_in_field_order() {
        let mut s = Message::default();
        s.set(2, Value::Varint(1));
        s.set(5, Value::Bytes(Vec::new()));
        s.set(24, Value::Varint(1));
        edit(&mut s, 0, 0, "Neu");
        let numbers: Vec<u32> = s.fields.iter().map(|f| f.number).collect();
        assert_eq!(numbers, vec![2, 3, 5, 24]);
        assert_eq!(read(&s), "Neu");
    }

    #[test]
    fn multiple_text_runs_collapse_to_one() {
        let mut s = storage("abc", vec![(5, vec![entry(0, Some(1))])]);
        s.fields.push(Field {
            number: 3,
            value: Value::Bytes(b"def".to_vec()),
        });
        edit(&mut s, 0, 6, "xyz");
        assert_eq!(s.all(3).count(), 1);
        assert_eq!(read(&s), "xyz");
    }

    /// Whatever else an entry carries has to survive the move.
    #[test]
    fn entries_keep_the_fields_this_crate_does_not_model() {
        let mut rich = entry(10, Some(700));
        rich.set(7, Value::Varint(42));
        let mut s = storage(&"x".repeat(40), vec![(8, vec![entry(0, None), rich])]);
        edit(&mut s, 2, 0, "ab");
        let table = Message::decode(s.bytes(8).unwrap()).unwrap();
        let kept: Vec<Option<u64>> = split(&table, Anchoring::Run)
            .0
            .iter()
            .map(|e| e.message.varint(7))
            .collect();
        assert_eq!(kept, vec![None, Some(42)]);
        assert_eq!(shape(&s, 8), vec![(0, None), (12, Some(700))]);
    }

    /// A run table's first entry stays at 0 however the text is edited: it is
    /// where the first run begins, and text inserted at the start of a storage
    /// has no earlier run to take its attributes from.
    #[test]
    fn a_run_table_keeps_its_first_entry_at_zero() {
        let mut s = storage(
            &"x".repeat(40),
            vec![(8, vec![entry(0, Some(100)), entry(10, Some(200))])],
        );
        edit(&mut s, 0, 0, "neu");
        assert_eq!(shape(&s, 8), vec![(0, Some(100)), (13, Some(200))]);

        // And a delete from the very start: the characters that survive keep
        // what they had, so the second run reaches 0.
        let mut s = storage(
            &"x".repeat(40),
            vec![(8, vec![entry(0, Some(100)), entry(10, Some(200))])],
        );
        edit(&mut s, 0, 10, "");
        assert_eq!(shape(&s, 8), vec![(0, Some(200))]);
    }

    /// Replacing the whole text is `set_text`, and it keeps the storage's first
    /// paragraph style over everything: the paragraphs the new text has get an
    /// entry each where the table had one per paragraph before.
    #[test]
    fn a_full_replace_keeps_the_first_style_and_repopulates() {
        let mut s = styled();
        edit(&mut s, 0, 171, "Eins\rZwei\rDrei");
        assert_eq!(
            shape(&s, 5),
            vec![(0, Some(1732648)), (5, None), (10, None)]
        );
        assert_eq!(read(&s), "Eins\rZwei\rDrei");
    }

    /// An astral character counts as two, and an edit that lands after one is
    /// still an edit at the index the table counts in.
    #[test]
    fn an_edit_past_an_emoji_counts_in_code_units() {
        let mut s = storage(
            "a\u{1F600}bcdef",
            vec![(8, vec![entry(0, Some(100)), entry(5, Some(200))])],
        );
        assert_eq!(length(&read(&s)), 8);
        edit(&mut s, 3, 0, "XY");
        assert_eq!(read(&s), "a\u{1F600}XYbcdef");
        assert_eq!(shape(&s, 8), vec![(0, Some(100)), (7, Some(200))]);
    }

    /// An edit that changes nothing leaves every table byte for byte.
    #[test]
    fn an_empty_edit_changes_nothing() {
        let mut s = styled();
        let before = s.encode();
        let report = edit(&mut s, 40, 0, "");
        assert_eq!(s.encode(), before);
        assert!(report.tables.is_empty());
    }
}
