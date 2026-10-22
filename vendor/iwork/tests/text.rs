//! Editing text: what moves with it, and what is refused.
//!
//! The unit tests in `src/text.rs` prove the remapper against storages built by
//! hand, and against the ten archives Pages wrote when it was made to perform
//! the same edits. These are the ones that need whole documents: that an edit
//! keeps the invariants `iwork check` enforces, that it touches one stream, that
//! everything anchored into the storage is still where it belongs, and — behind
//! `IWORK_APP_CHECK=1` — that the app opens the result and reads the new words
//! back.

use std::path::{Path, PathBuf};

use iwork::text::Anchoring;
use iwork::{style::StyleKind, Document, Error, Kind};

fn generated(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name);
    path.exists().then_some(path)
}

macro_rules! fixture {
    ($name:expr) => {
        match generated($name) {
            Some(path) => path,
            None => {
                eprintln!("no {} — skipping (run scripts/make-fixtures.sh)", $name);
                return;
            }
        }
    };
}

/// A password-protected package, which is not a fixture any test here can use:
/// its object streams are ciphertext and `Document::open` refuses it by design.
/// `tests/fixtures.rs` is where that refusal is asserted.
fn encrypted(path: &Path) -> bool {
    iwork::Package::read(path).is_ok_and(|package| package.contains(".iwpv2"))
}

/// The corpus is generated, never committed, so a sweep over it skips where
/// it is absent — the same rule `tests/fixtures.rs` states as "so a fresh
/// clone is green". The counting assertions in these sweeps keep their teeth
/// wherever the corpus *is* present, which is the only place they could ever
/// have said anything.
macro_rules! corpus {
    () => {
        if every_fixture().is_empty() {
            eprintln!("no corpus — skipping (run scripts/make-fixtures.sh)");
            return;
        }
    };
}

fn every_fixture() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("pages") | Some("numbers") | Some("key")
            )
        })
        .filter(|path| !encrypted(path))
        .collect();
    found.sort();
    found
}

/// Rewrite one object of a fixture and hand back the document that results.
///
/// Some of what an edit must refuse cannot be produced by this crate at all —
/// a storage whose text is not UTF-8, an attribute table whose bytes are not a
/// message, a footnote whose mark has been cut out from under it. They are
/// documents another tool could hand a caller, so they are built the way that
/// tool would build them: through `Package`, `iwa::parse` and `iwa::serialize`,
/// below every guard this crate has.
fn forge(path: &Path, identifier: u64, edit: impl FnOnce(&mut iwork::pb::Message)) -> Document {
    let mut package = iwork::Package::read(path).unwrap();
    let stream = "Index/Document.iwa";
    let mut objects = iwork::iwa::parse(package.get(stream).unwrap()).unwrap();
    let object = objects
        .iter_mut()
        .find(|object| object.identifier == identifier)
        .expect("the object is in the document stream");
    let mut archive = iwork::pb::Message::decode(object.payload()).unwrap();
    edit(&mut archive);
    object.messages[0].payload = archive.encode();
    package.set(stream, iwork::iwa::serialize(&objects));
    Document::from_package(package).unwrap()
}

/// Write to a scratch file beside the fixtures and reopen it.
fn reopen(doc: &Document, name: &str) -> Document {
    let out = std::env::temp_dir().join(format!("iwork-text-{name}"));
    doc.save(&out).unwrap();
    let reopened = Document::open(&out).unwrap();
    let _ = std::fs::remove_file(&out);
    reopened
}

// -- the corpus, edited ------------------------------------------------------

/// Every storage in the corpus, edited in the middle, still keeps every rule
/// `iwork check` knows.
///
/// This is the test the old clamping would fail: pull four paragraph runs down
/// onto one index and the paragraph table stops sitting on paragraph starts.
#[test]
fn an_edit_anywhere_leaves_a_document_that_checks_out() {
    corpus!();
    let mut edited = 0;
    for path in every_fixture() {
        let doc = Document::open(&path).unwrap();
        for storage in doc.storages() {
            if storage.length < 8 || storage.unknown_field.is_some() {
                continue;
            }
            let middle = storage.length / 2;
            for edit in [(middle, 0, "eingefügt"), (middle, 4, ""), (0, 3, "abc")] {
                let mut copy = Document::open(&path).unwrap();
                let outcome =
                    copy.replace_text(storage.identifier, edit.0..edit.0 + edit.1, edit.2);
                match outcome {
                    // A range covering an anchor is refused by name and the
                    // document is left exactly as it was.
                    Err(Error::AnchoredObject { .. })
                    | Err(Error::SplitSurrogate { .. })
                    | Err(Error::TrackedChanges { .. }) => {
                        assert!(
                            copy.changed_streams().is_empty(),
                            "{}: a refused edit changed a stream",
                            path.display()
                        );
                        continue;
                    }
                    Err(e) => panic!("{}: storage {}: {e}", path.display(), storage.identifier),
                    Ok(_) => {}
                }
                assert_eq!(
                    copy.problems(),
                    Vec::<String>::new(),
                    "{}: storage {} after {edit:?}",
                    path.display(),
                    storage.identifier
                );
                edited += 1;
            }
        }
    }
    assert!(edited > 100, "only {edited} edits were made");
}

/// One edit, one stream — and every other object byte for byte.
#[test]
fn an_edit_touches_only_the_stream_it_is_in() {
    let path = fixture!("pages-styled.pages");
    let doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap();

    let mut edited = Document::open(&path).unwrap();
    edited
        .insert_text(storage.identifier, 5, "eingeschoben")
        .unwrap();
    assert_eq!(edited.changed_streams(), vec![storage.stream.as_str()]);

    // And an edit that changes nothing rewrites nothing.
    let mut untouched = Document::open(&path).unwrap();
    untouched.insert_text(storage.identifier, 5, "").unwrap();
    assert!(untouched.changed_streams().is_empty());
}

/// The paragraph tables keep one entry per paragraph across a paragraph being
/// created and a paragraph being destroyed.
#[test]
fn paragraph_bookkeeping_survives_both_directions() {
    let path = fixture!("pages-styled.pages");
    let doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap().identifier;
    let before = doc.paragraph_ranges(storage).unwrap().len();

    let mut split = Document::open(&path).unwrap();
    let report = split.insert_text(storage, 20, "\rNeuer Absatz").unwrap();
    assert_eq!(split.paragraph_ranges(storage).unwrap().len(), before + 1);
    assert_eq!(report.report.added, 1, "the new paragraph got an entry");
    assert_eq!(split.problems(), Vec::<String>::new());

    // Deleting the break at 11 merges the first two paragraphs, and the entry
    // that began the second one goes with it.
    let mut merged = Document::open(&path).unwrap();
    let report = merged.delete_text(storage, 11..12).unwrap();
    assert_eq!(merged.paragraph_ranges(storage).unwrap().len(), before - 1);
    assert_eq!(report.report.dropped, 1);
    assert_eq!(merged.problems(), Vec::<String>::new());
}

/// Styles keep the words they were on. The storage carries four paragraph
/// styles; inserting into the middle of the second must leave the third and
/// fourth on the same words they started on.
#[test]
fn a_style_stays_on_its_own_words() {
    let path = fixture!("pages-styled.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap().identifier;

    let before: Vec<(String, Option<u64>)> = doc
        .paragraph_ranges(storage)
        .unwrap()
        .into_iter()
        .map(|range| {
            let text = doc.storage_text(storage).unwrap();
            let style = doc
                .style_of_run(storage, range.start, StyleKind::Paragraph)
                .unwrap()
                .map(|r| r.style);
            (text[..].to_string().len().to_string(), style)
        })
        .collect();

    doc.insert_text(storage, 30, "XXXXX").unwrap();
    let after: Vec<Option<u64>> = doc
        .paragraph_ranges(storage)
        .unwrap()
        .into_iter()
        .map(|range| {
            doc.style_of_run(storage, range.start, StyleKind::Paragraph)
                .unwrap()
                .map(|r| r.style)
        })
        .collect();
    assert_eq!(
        after,
        before.iter().map(|(_, s)| *s).collect::<Vec<_>>(),
        "the paragraphs must still be in the styles they were in"
    );
}

/// An anchored image is not silently detached: the edit is refused, by name,
/// and the drawable is still anchored where it was.
#[test]
fn text_that_anchors_an_object_is_not_replaced() {
    let path = fixture!("pages-report.pages");
    let mut doc = Document::open(&path).unwrap();
    let body = doc.text_storages().into_iter().next().unwrap().identifier;

    // The photo is anchored at character 12 of the body storage.
    let anchor = doc
        .drawables()
        .into_iter()
        .filter_map(|d| match d.placement {
            // The photo, at character 12. The report's table is anchored too,
            // at 1500, which is the last character of the storage.
            iwork::Placement::InText { storage, character }
                if storage == body && d.kind == iwork::drawable::Kind::Image =>
            {
                Some(character)
            }
            _ => None,
        })
        .next()
        .expect("the report has an image anchored in its text");

    match doc.delete_text(body, anchor..anchor + 2) {
        Err(Error::AnchoredObject { index, table, .. }) => {
            assert_eq!(index, anchor);
            assert_eq!(table, "table_attachment");
        }
        other => panic!("expected a named refusal, got {other:?}"),
    }
    assert!(doc.changed_streams().is_empty());

    // Deleting text that does not cover the anchor is fine, and moves it.
    let moved = doc.delete_text(body, anchor + 2..anchor + 6).unwrap();
    assert!(moved.report.moved > 0);
    assert_eq!(doc.problems(), Vec::<String>::new());
}

/// A footnote mark is an anchor like any other, and its character is a
/// **`U+000E`** — not the `U+FFFC` an attachment sits on.
///
/// This is the hole `pages-footnotes.pages` opened up and closed. The check
/// that decided whether an anchor was destroyed read the character first and
/// let anything but `U+FFFC` and `U+0004` through, so
/// `iwork delete-text pages-footnotes.pages 1732539 20 30` exited 0, orphaned a
/// `TSWP.FootnoteReferenceAttachmentArchive` and the kind-2 storage hanging off
/// it, and left `iwork check` with nothing to find.
#[test]
fn deleting_a_footnote_mark_is_refused() {
    let path = fixture!("pages-footnotes.pages");
    let structure = Document::open(&path).unwrap().structure().unwrap();
    assert_eq!(
        structure.footnotes.len(),
        2,
        "the fixture has two footnotes"
    );
    let note = structure.footnotes[0].clone();
    let body = note.storage;

    // The reviewer's range, and exactly the mark.
    for range in [20..30, note.index..note.index + 1] {
        let mut doc = Document::open(&path).unwrap();
        match doc.delete_text(body, range.clone()) {
            Err(Error::AnchoredObject {
                storage,
                index,
                table,
                object,
            }) => {
                assert_eq!(storage, body);
                assert_eq!(index, note.index);
                assert_eq!(table, "table_footnote");
                assert_eq!(object, note.attachment);
            }
            other => panic!("{range:?} was not refused: {other:?}"),
        }
        assert!(doc.changed_streams().is_empty(), "{range:?}");
    }

    // And an edit that leaves both marks alone is still allowed: they move, the
    // notes stay attached, and the document still checks out.
    let mut doc = Document::open(&path).unwrap();
    doc.insert_text(body, 0, "Neu ").unwrap();
    assert_eq!(doc.problems(), Vec::<String>::new());
    let after = doc.structure().unwrap();
    assert_eq!(
        after
            .footnotes
            .iter()
            .map(|n| n.index)
            .collect::<Vec<u64>>(),
        structure
            .footnotes
            .iter()
            .map(|n| n.index + 4)
            .collect::<Vec<u64>>(),
        "the marks moved with the text"
    );
    assert_eq!(
        after.footnotes.iter().map(|n| n.body).collect::<Vec<_>>(),
        structure
            .footnotes
            .iter()
            .map(|n| n.body)
            .collect::<Vec<_>>(),
        "and still contain their notes"
    );
}

/// Indices are UTF-16 code units, and an edit may not land inside a pair.
#[test]
fn an_edit_inside_a_surrogate_pair_is_refused() {
    let path = fixture!("pages-unicode.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap();
    let text = storage.text.clone();

    // The first astral character in the fixture, and the index of its low half.
    let mut units = 0u64;
    let mut split = None;
    for character in text.chars() {
        if character.len_utf16() == 2 {
            split = Some(units + 1);
            break;
        }
        units += character.len_utf16() as u64;
    }
    let split = split.expect("the unicode fixture has an emoji");

    match doc.insert_text(storage.identifier, split, "x") {
        Err(Error::SplitSurrogate { index, .. }) => assert_eq!(index, split),
        other => panic!("expected a named refusal, got {other:?}"),
    }
    // One code unit either side is fine, and the text comes back intact.
    doc.insert_text(storage.identifier, split - 1, "x").unwrap();
    let after = doc.storage_text(storage.identifier).unwrap();
    assert!(after.chars().count() == text.chars().count() + 1);
    assert_eq!(doc.problems(), Vec::<String>::new());
}

/// Characters that stand for objects are not text and are not written.
#[test]
fn a_placeholder_character_is_not_writable() {
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap().identifier;
    for character in iwork::text::UNWRITABLE {
        match doc.insert_text(storage, 0, &character.to_string()) {
            Err(Error::UnwritableCharacter { character: found }) => assert_eq!(found, *character),
            other => panic!("expected a named refusal for {character:?}, got {other:?}"),
        }
    }
}

/// And the damage that refusal prevents is damage `iwork check` can now see.
///
/// A footnote is a chain of three — the mark's `table_footnote` entry, the
/// `TSWP.FootnoteReferenceAttachmentArchive` it names, and the kind-2 storage
/// that archive contains — and the mark is the only route in. Cut it out and
/// nothing pointed at the other two, which is exactly the state
/// `delete-text … 20 30` used to leave behind and exactly what `check` had
/// nothing to say about.
#[test]
fn a_footnote_whose_mark_is_gone_is_a_problem() {
    let path = fixture!("pages-footnotes.pages");
    let doc = Document::open(&path).unwrap();
    assert_eq!(doc.problems(), Vec::<String>::new());
    let body = doc.structure().unwrap().footnotes[0].storage;

    let forged = forge(&path, body, |archive| {
        archive.clear(iwork::text::FOOTNOTE_TABLE);
    });
    let problems = forged.problems();
    assert_eq!(
        problems
            .iter()
            .filter(|p| p.contains("footnote attachment"))
            .count(),
        2,
        "{problems:?}"
    );
    assert_eq!(
        problems
            .iter()
            .filter(|p| p.contains("footnote body"))
            .count(),
        2,
        "{problems:?}"
    );
}

/// A storage this crate cannot read faithfully is not edited badly.
///
/// Two ways in, neither of them producible by this crate — hence [`forge`]:
///
/// * an attribute table whose bytes do not decode. Every *other* table would be
///   remapped and this one stepped over, leaving it anchored to characters that
///   have moved;
/// * text that is not valid UTF-8. The reader is lossy on purpose; the writer
///   would put the lossy reading back and turn every ill-formed sequence into a
///   `U+FFFD` for good, moving every index after it.
#[test]
fn a_storage_this_crate_cannot_read_faithfully_is_not_edited() {
    let path = fixture!("pages-plain.pages");
    let storage = Document::open(&path)
        .unwrap()
        .text_storages()
        .into_iter()
        .next()
        .unwrap()
        .identifier;

    // A short string in the character-style table: it parses as a message and
    // does not re-encode to the same bytes, which is what `decode_nested`'s
    // round trip is for.
    let mut doc = forge(&path, storage, |archive| {
        archive.set_in_order(8, iwork::pb::Value::Bytes(b"Grosse Uberschrift".to_vec()));
    });
    match doc.set_text(storage, "Ein anderer Absatz.") {
        Err(Error::UndecodableAttributeTable { storage: s, field }) => {
            assert_eq!(s, storage);
            assert_eq!(field, 8);
        }
        other => panic!("expected a named refusal, got {other:?}"),
    }
    assert!(doc.changed_streams().is_empty());
    assert!(
        doc.problems().iter().any(|p| p.contains("does not decode")),
        "check says nothing about it: {:?}",
        doc.problems()
    );

    // Text that is not UTF-8. `String::from_utf8_lossy` reads it; nothing
    // writes it back.
    let mut doc = forge(&path, storage, |archive| {
        archive.set_in_order(3, iwork::pb::Value::Bytes(vec![b'a', 0xFF, 0xFE, b'b']));
    });
    match doc.insert_text(storage, 1, "x") {
        Err(Error::InvalidText { storage: s }) => assert_eq!(s, storage),
        other => panic!("expected a named refusal, got {other:?}"),
    }
    assert!(doc.changed_streams().is_empty());
    assert!(
        doc.problems().iter().any(|p| p.contains("not valid UTF-8")),
        "check says nothing about it: {:?}",
        doc.problems()
    );
}

/// The end-of-text paragraph entry — the style of the paragraph not yet typed,
/// which 1,168 of the corpus's 1,622 storages carry — is an **end marker**, and
/// follows the end of the text.
///
/// `iwork set-text pages-columns.pages 57456 "Hallo"` moved it to 5 correctly;
/// a further insert *at* 5 then held it there, where it was neither a paragraph
/// start nor the end of the text, and it was dropped. Typing at the end of a
/// text box is not an exotic edit.
#[test]
fn the_paragraph_entry_at_the_end_of_the_text_survives_typing_there() {
    let path = fixture!("pages-columns.pages");
    let doc = Document::open(&path).unwrap();
    // A text box whose paragraph table carries the trailing entry.
    let (storage, entries) = doc
        .storages()
        .into_iter()
        .filter_map(|s| {
            let table = s.tables.iter().find(|t| t.field == 5)?;
            let ranges = doc.paragraph_ranges(s.identifier).ok()?;
            (s.length > 0 && table.entries > ranges.len()).then_some((s.identifier, table.entries))
        })
        .next()
        .expect("pages-columns has a storage with the end-of-text entry");

    let ending = |doc: &Document| -> Vec<u64> {
        let (_, object) = doc.object(storage).unwrap();
        let archive = iwork::pb::Message::decode(object.payload()).unwrap();
        let table = iwork::pb::decode_nested(archive.bytes(5).unwrap()).unwrap();
        iwork::text::entry_indices(&table, Anchoring::Paragraph)
            .into_iter()
            .map(|(index, _)| index)
            .collect()
    };

    let mut doc = Document::open(&path).unwrap();
    doc.set_text(storage, "Hallo").unwrap();
    assert_eq!(ending(&doc).last().copied(), Some(5));
    assert_eq!(ending(&doc).len(), entries);

    doc.insert_text(storage, 5, " Welt").unwrap();
    assert_eq!(
        ending(&doc).last().copied(),
        Some(10),
        "the end marker was dropped instead of following the end of the text"
    );
    assert_eq!(ending(&doc).len(), entries);
    assert_eq!(doc.problems(), Vec::<String>::new());

    // And it survives the save, which is what a caller sees.
    let saved = reopen(&doc, "end-marker.pages");
    assert_eq!(saved.storage_text(storage).unwrap(), "Hallo Welt");
    assert_eq!(saved.problems(), Vec::<String>::new());
}

/// Styling a range in the middle of a storage that has no table of that kind
/// leaves a table beginning at the range, and characters `0..start` with no
/// attribute at all — which is the invariant `iwork check` enforces, broken by
/// the writer that is supposed to keep it.
///
/// What Pages writes instead is a **head entry at 0 carrying nothing**: made to
/// set a bold 22pt font over characters 19–29, it wrote `[0 nil, 19 bold,
/// 30 nil]`. See `FORMAT.md` §Text.
#[test]
fn styling_from_the_middle_gives_the_table_a_head_entry_at_zero() {
    let path = fixture!("pages-plain.pages");
    let doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap().identifier;
    assert!(
        doc.storages()
            .iter()
            .find(|s| s.identifier == storage)
            .unwrap()
            .tables
            .iter()
            .all(|t| t.field != 8),
        "the fixture's body has no character table to begin with"
    );
    let mut character: Vec<u64> = doc
        .text_styles()
        .into_iter()
        .filter(|s| s.kind == StyleKind::Character)
        .map(|s| s.identifier)
        .collect();
    character.sort_unstable();
    let style = character[0];

    let mut doc = Document::open(&path).unwrap();
    doc.apply_text_style(storage, 4..9, style).unwrap();
    assert_eq!(doc.problems(), Vec::<String>::new());

    let doc = reopen(&doc, "head-entry.pages");
    let (_, object) = doc.object(storage).unwrap();
    let archive = iwork::pb::Message::decode(object.payload()).unwrap();
    let table = iwork::pb::decode_nested(archive.bytes(8).unwrap()).unwrap();
    assert_eq!(
        iwork::text::entry_indices(&table, Anchoring::Run),
        vec![(0, None), (4, Some(style))],
        "the head entry is at 0 and carries nothing"
    );
    // And the table went in in field order, where iWork puts its fields, rather
    // than being appended after field 28.
    let fields: Vec<u32> = archive.fields.iter().map(|f| f.number).collect();
    let mut sorted = fields.clone();
    sorted.sort_unstable();
    assert_eq!(
        fields, sorted,
        "the new table field was appended, not placed"
    );
}

/// Deleting a style with nothing to put in its place lets the run before it
/// extend over its text — except at index 0, where there is no run before.
///
/// `iwork delete-style pages-styled.pages 1732648` removed the entry that
/// covered the first paragraph and left `check` reporting `0..12` with no
/// attribute at all. What stays behind is the entry with its object taken off,
/// which is the format's "whatever was in force here" and what Pages writes for
/// an unstyled first run.
#[test]
fn deleting_the_style_of_the_first_run_leaves_a_nil_entry_behind() {
    let path = fixture!("pages-styled.pages");
    let doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap().identifier;
    let first = doc
        .style_of_run(storage, 0, StyleKind::Paragraph)
        .unwrap()
        .expect("the first paragraph has a style")
        .style;

    let mut doc = Document::open(&path).unwrap();
    let deletion = doc.delete_text_style(first, None).unwrap();
    assert_eq!(deletion.runs_dropped, 1);
    assert_eq!(doc.problems(), Vec::<String>::new());

    let doc = reopen(&doc, "delete-style.pages");
    assert!(doc.text_style(first).is_none());
    let (_, object) = doc.object(storage).unwrap();
    let archive = iwork::pb::Message::decode(object.payload()).unwrap();
    let table = iwork::pb::decode_nested(archive.bytes(5).unwrap()).unwrap();
    let entries = iwork::text::entry_indices(&table, Anchoring::Paragraph);
    assert_eq!(
        entries[0],
        (0, None),
        "the run at 0 is emptied, not removed: {entries:?}"
    );
    assert_eq!(doc.problems(), Vec::<String>::new());
}

/// A range past the end of the text is refused, not quietly pulled back to it:
/// `apply-style … 500 600` on a 54-unit body restyled the only paragraph it
/// had and reported success.
#[test]
fn styling_past_the_end_of_the_text_is_refused() {
    let path = fixture!("pages-plain.pages");
    let doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap();
    let length = iwork::text::length(&storage.text);
    let style = doc
        .text_styles()
        .into_iter()
        .find(|s| s.kind == StyleKind::Paragraph)
        .unwrap()
        .identifier;

    let mut doc = Document::open(&path).unwrap();
    match doc.apply_text_style(storage.identifier, 500..600, style) {
        Err(Error::TextRange {
            index, length: l, ..
        }) => {
            assert_eq!(index, 600);
            assert_eq!(l, length);
        }
        other => panic!("expected a named refusal, got {other:?}"),
    }
    assert!(doc.changed_streams().is_empty());
    // The whole text is still fair game.
    doc.apply_text_style(storage.identifier, 0..length, style)
        .unwrap();
    assert_eq!(doc.problems(), Vec::<String>::new());
}

// -- what a storage carries --------------------------------------------------

/// Every storage in the corpus is made of fields this crate can place. A field
/// it cannot place is a table an edit would remap by guesswork, and the whole
/// point of the inventory is that there is nowhere for one to hide.
#[test]
fn no_storage_in_the_corpus_carries_an_unknown_table() {
    corpus!();
    let mut seen: std::collections::BTreeSet<u32> = Default::default();
    for path in every_fixture() {
        let doc = Document::open(&path).unwrap();
        for storage in doc.storages() {
            assert_eq!(
                storage.unknown_field,
                None,
                "{}: storage {}",
                path.display(),
                storage.identifier
            );
            seen.extend(storage.tables.iter().map(|t| t.field));
        }
    }
    // What the corpus actually exercises, so a change to it is visible. The
    // UI-made fixtures added bookmarks (15), footnotes (16), tracked changes
    // (21, 22) and comment highlights (23).
    assert_eq!(
        seen.into_iter().collect::<Vec<u32>>(),
        vec![5, 6, 7, 8, 9, 11, 12, 14, 15, 16, 17, 19, 21, 22, 23, 24, 28]
    );
}

/// Hyperlinks: read, and their target changed.
#[test]
fn a_hyperlink_is_read_and_repointed() {
    let path = fixture!("numbers-links.numbers");
    let mut doc = Document::open(&path).unwrap();

    let links: Vec<iwork::document::SmartField> = doc
        .smart_fields()
        .into_iter()
        .filter(|f| f.message_type == iwork::document::TYPE_HYPERLINK_FIELD)
        .collect();
    assert_eq!(links.len(), 2, "the invoice template has two links");
    assert_eq!(
        links[0].payload.as_deref(),
        Some("mailto:no_reply@example.com")
    );
    assert_eq!(links[0].text, "no_reply@example.com");
    assert_eq!(links[1].payload.as_deref(), Some("http://example.com"));
    // The second link runs to the end of its storage with no terminating entry.
    assert_eq!(links[1].text, "example.com");

    doc.set_link_url(links[1].object, "https://zebrapig.com/iwork")
        .unwrap();
    let doc = reopen(&doc, "link.numbers");
    let after = doc
        .smart_fields()
        .into_iter()
        .find(|f| f.object == links[1].object)
        .unwrap();
    assert_eq!(after.payload.as_deref(), Some("https://zebrapig.com/iwork"));
    assert_eq!(after.range, links[1].range, "the run did not move");
    assert_eq!(doc.problems(), Vec::<String>::new());
}

/// A link's run moves with the text it covers.
#[test]
fn editing_around_a_link_moves_it() {
    let path = fixture!("numbers-links.numbers");
    let mut doc = Document::open(&path).unwrap();
    let link = doc
        .smart_fields()
        .into_iter()
        .find(|f| f.message_type == iwork::document::TYPE_HYPERLINK_FIELD)
        .unwrap();

    doc.insert_text(link.storage, 0, "Kontakt: ").unwrap();
    let moved = doc
        .smart_fields()
        .into_iter()
        .find(|f| f.object == link.object)
        .unwrap();
    assert_eq!(moved.range.start, link.range.start + 9);
    assert_eq!(moved.range.end, link.range.end + 9);
    assert_eq!(moved.text, link.text, "and it still covers the same words");
    assert_eq!(doc.problems(), Vec::<String>::new());
}

/// Lists: the level per paragraph and the style in force, read off a document
/// Pages wrote from its Real Estate Flyer template.
#[test]
fn list_levels_and_styles_are_read_per_paragraph() {
    let path = fixture!("pages-lists.pages");
    let doc = Document::open(&path).unwrap();
    let body = doc
        .storages()
        .into_iter()
        .find(|s| s.kind == 0)
        .expect("a body storage");
    let paragraphs = doc.list_paragraphs(body.identifier).unwrap();
    assert_eq!(paragraphs.len(), 14);

    let named = |style: Option<u64>| {
        style
            .and_then(|id| doc.text_style(id))
            .and_then(|s| s.name)
            .unwrap_or_default()
    };
    let shape: Vec<(u64, String)> = paragraphs
        .iter()
        .map(|p| (p.level, named(p.style)))
        .collect();
    assert_eq!(
        shape,
        vec![
            (0, "None".into()),
            (0, "None".into()),
            (0, "Bullet 2".into()),
            (0, "Bullet 2".into()),
            (0, "Bullet 2".into()),
            (0, "Bullet 2".into()),
            (0, "None".into()),
            (0, "None".into()),
            (0, "None".into()),
            (0, "Bullet".into()),
            (0, "Bullet".into()),
            // Two paragraphs one level in — "Use the Tab key to indent."
            (1, "Bullet".into()),
            (1, "Bullet".into()),
            (0, "Bullet".into()),
        ]
    );

    // The level survives an edit inside the indented paragraph.
    let mut edited = Document::open(&path).unwrap();
    let indented = paragraphs[11].range.clone();
    edited
        .insert_text(body.identifier, indented.start + 4, "sehr ")
        .unwrap();
    let after = edited.list_paragraphs(body.identifier).unwrap();
    assert_eq!(after[11].level, 1);
    assert_eq!(after[12].level, 1);
    assert_eq!(after[13].level, 0);
    assert_eq!(edited.problems(), Vec::<String>::new());
}

/// Named style, or named style plus local overrides — the distinction a run
/// carries and the reason editing "Title" can change nothing on screen.
#[test]
fn a_run_reports_its_named_style_and_its_overrides() {
    let path = fixture!("pages-styled.pages");
    let doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap().identifier;

    let mut variations = 0;
    let mut named = 0;
    for range in doc.paragraph_ranges(storage).unwrap() {
        let resolved = doc
            .style_of_run(storage, range.start, StyleKind::Paragraph)
            .unwrap()
            .expect("every paragraph has a style");
        assert!(
            resolved.name.is_some(),
            "a variation must still resolve to the named style it descends from"
        );
        if resolved.is_variation {
            variations += 1;
            assert_ne!(
                Some(resolved.style),
                resolved.named,
                "a variation is not the style it inherits from"
            );
            assert!(
                !resolved.overrides.is_empty(),
                "a variation that overrides nothing would be pointless"
            );
        } else {
            named += 1;
            assert_eq!(Some(resolved.style), resolved.named);
        }
    }
    assert!(
        variations > 0 && named > 0,
        "the styled fixture has both kinds: {variations} variation(s), {named} named"
    );

    // The red paragraph's variation overrides the font colour and nothing else
    // this crate can name.
    let red = doc
        .style_of_run(storage, 12, StyleKind::Paragraph)
        .unwrap()
        .unwrap();
    assert!(
        red.overrides.iter().any(|o| o == "font-color"),
        "expected a colour override, got {:?}",
        red.overrides
    );
}

/// The anchoring of a table is what decides what an edit does to it, so it is
/// worth asserting that the corpus's tables are anchored the way this crate
/// believes: every entry of a character-anchored table sits on a `U+FFFC`, or
/// on the `U+000E` of a footnote mark, which is what makes them unremappable.
///
/// The refusal itself does *not* consult this — an anchor on a character the
/// list has not met is the case where refusing matters most — so this is the
/// tripwire for the day a third placeholder turns up, not the check.
#[test]
fn a_character_anchored_entry_sits_on_a_placeholder() {
    corpus!();
    let mut checked = 0;
    for path in every_fixture() {
        let doc = Document::open(&path).unwrap();
        for storage in doc.storages() {
            let text: Vec<u16> = doc
                .storage_text(storage.identifier)
                .unwrap()
                .encode_utf16()
                .collect();
            for table in storage
                .tables
                .iter()
                .filter(|t| t.anchoring == Anchoring::Character)
            {
                let (_, object) = doc.object(storage.identifier).unwrap();
                let archive = iwork::pb::Message::decode(object.payload()).unwrap();
                let decoded =
                    iwork::pb::decode_nested(archive.bytes(table.field).unwrap()).unwrap();
                for (index, _) in iwork::text::entry_indices(&decoded, Anchoring::Character) {
                    let unit = text.get(index as usize).copied();
                    // An attachment sits on U+FFFC; a footnote MARK sits on
                    // U+000E — pages-footnotes.pages is what settled that a
                    // footnote's placeholder is its own character, not the
                    // object-replacement one.
                    let expected: &[u16] = if table.field == iwork::text::FOOTNOTE_TABLE {
                        &[0x000E]
                    } else {
                        &[0xFFFC]
                    };
                    assert!(
                        unit.is_none() || expected.contains(&unit.unwrap()),
                        "{}: storage {} {} entry at {index} sits on U+{:04X}",
                        path.display(),
                        storage.identifier,
                        table.name,
                        unit.unwrap_or(0)
                    );
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "nothing was checked");
}

// -- the app -----------------------------------------------------------------

fn app_check(path: &Path, expected: &str) {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
    let status = std::process::Command::new(&script)
        .arg(path)
        .arg(expected)
        .status()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        status.success(),
        "{}: the app did not accept it, or did not read back {expected:?} (exit {:?})",
        path.display(),
        status.code()
    );
}

/// The claim of the whole phase, put to the app.
///
/// Off unless `IWORK_APP_CHECK=1`. Three edits — an insertion into the middle
/// of a styled paragraph, a deletion across a paragraph boundary, and text
/// changed inside a Keynote shape — each written, opened by the app that owns
/// it, and read back. The document's drawables must still be where they were,
/// and `check` must still be clean.
#[test]
fn the_app_opens_an_edited_document_and_reads_the_new_words_back() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }

    // 1. Insert into the middle of a styled paragraph.
    let path = fixture!("pages-styled.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.text_storages().into_iter().next().unwrap().identifier;
    doc.insert_text(storage, 22, "ZWISCHENGESCHOBEN ").unwrap();
    let out = std::env::temp_dir().join("iwork-insert.pages");
    doc.save(&out).unwrap();
    app_check(&out, "ZWISCHENGESCHOBEN");

    // 2. Delete across a paragraph boundary — the edit that used to leave a
    //    paragraph run where no paragraph begins.
    let mut doc = Document::open(&path).unwrap();
    doc.delete_text(storage, 5..20).unwrap();
    let out = std::env::temp_dir().join("iwork-delete.pages");
    doc.save(&out).unwrap();
    app_check(&out, "Übersr Absatz");

    // 3. A drawable-bearing document: the photo must survive an edit to the
    //    text it is anchored in, and stay where it was.
    let path = fixture!("pages-report.pages");
    let before = Document::open(&path).unwrap();
    let body = before
        .text_storages()
        .into_iter()
        .next()
        .unwrap()
        .identifier;
    let places: Vec<(u64, f32, f32)> = before
        .drawables()
        .iter()
        .map(|d| (d.identifier, d.geometry.x, d.geometry.y))
        .collect();
    let mut doc = Document::open(&path).unwrap();
    doc.replace_text(body, 20..30, "Projektantrag").unwrap();
    let out = std::env::temp_dir().join("iwork-anchor.pages");
    doc.save(&out).unwrap();
    let after = Document::open(&out).unwrap();
    assert_eq!(
        after
            .drawables()
            .iter()
            .map(|d| (d.identifier, d.geometry.x, d.geometry.y))
            .collect::<Vec<_>>(),
        places,
        "every drawable must still be there and still be where it was"
    );
    assert_eq!(after.problems(), Vec::<String>::new());
    app_check(&out, "Projektantrag");

    // 4. Text inside a Keynote shape — a storage that is not the body.
    let path = fixture!("keynote-shapes.key");
    let mut doc = Document::open(&path).unwrap();
    // Not simply the first shape with a storage. A Keynote drawable's caption
    // is a storage whose whole contents are the `U+FFFC` of an attachment, and
    // replacing that is refused, correctly; and most of the deck's shapes are
    // on *layouts* (`TemplateSlide-*`), which the app does not enumerate, so an
    // edit to one would come back as "the app did not read it" when the app was
    // never going to.
    let shape = doc
        .drawables()
        .into_iter()
        .filter(|d| d.stream.contains("/Slide-"))
        .filter_map(|d| d.text)
        .find(|storage| {
            Document::open(&path)
                .unwrap()
                .set_text(*storage, "x")
                .is_ok()
        })
        .expect("a shape on a slide whose text can be replaced");
    let length = iwork::text::length(&doc.storage_text(shape).unwrap());
    doc.replace_text(shape, 0..length, "Neuer Formtext")
        .unwrap();
    let out = std::env::temp_dir().join("iwork-shape.key");
    doc.save(&out).unwrap();
    app_check(&out, "Neuer Formtext");

    // 5. A hyperlink repointed, and text inserted in front of it. Numbers has
    //    no way to report a link's target — no app's dictionary has one, which
    //    is why the fixture had to be a template — so what the app can say is
    //    that it opens the document and reads the words the link covers. The
    //    URL itself is checked by decoding the result, above.
    let path = fixture!("numbers-links.numbers");
    let mut doc = Document::open(&path).unwrap();
    let link = doc
        .smart_fields()
        .into_iter()
        .find(|f| f.message_type == iwork::document::TYPE_HYPERLINK_FIELD)
        .unwrap();
    doc.set_link_url(link.object, "mailto:leonce@zebrapig.com")
        .unwrap();
    doc.insert_text(link.storage, 0, "Kontakt\n").unwrap();
    let out = std::env::temp_dir().join("iwork-link.numbers");
    doc.save(&out).unwrap();
    let after = Document::open(&out).unwrap();
    let moved = after
        .smart_fields()
        .into_iter()
        .find(|f| f.object == link.object)
        .unwrap();
    assert_eq!(moved.payload.as_deref(), Some("mailto:leonce@zebrapig.com"));
    assert_eq!(moved.text, link.text);
    app_check(&out, &link.text);

    // 6. An edit beside a footnote — the one the crate now refuses to make
    //    *across*. Deleting the `U+000E` mark is refused; typing in front of it
    //    is ordinary, and the note must still be a note afterwards. Pages is
    //    the only thing that can say so: a document whose footnote chain is
    //    broken opens with the mark and no note, or does not open.
    let path = fixture!("pages-footnotes.pages");
    let mut doc = Document::open(&path).unwrap();
    let before = doc.structure().unwrap();
    let body = before.footnotes[0].storage;
    doc.insert_text(body, 0, "FUSSNOTENNAH ").unwrap();
    let out = std::env::temp_dir().join("iwork-footnote.pages");
    doc.save(&out).unwrap();
    app_check(&out, "FUSSNOTENNAH");
    let after = Document::open(&out).unwrap();
    let notes = after.structure().unwrap().footnotes;
    assert_eq!(notes.len(), before.footnotes.len());
    for (moved, was) in notes.iter().zip(&before.footnotes) {
        assert_eq!(moved.index, was.index + 13);
        assert_eq!(moved.attachment, was.attachment);
        assert_eq!(moved.body, was.body);
        assert_eq!(moved.text, was.text);
    }
    assert_eq!(after.problems(), Vec::<String>::new());

    // 7. And the app's own save of an edited document still decodes here — the
    //    strongest statement available without a screen: whatever Keynote made
    //    of what this crate wrote, it is something this crate reads back.
    let resaved = Document::open(&out).unwrap();
    assert_eq!(resaved.problems(), Vec::<String>::new());
}

// -- change tracking (phase 7) ------------------------------------------------

/// An edit through a storage with tracked changes is refused rather than
/// remapped.
///
/// Nothing in the corpus has one and nothing here can make Pages produce one —
/// its dictionary has no change-tracking property at all — so the storage is
/// built by hand, in the shape `iwork::text` says `table_deletion` has:
/// `{character_index, reference}`, a run table by its layout. The point of the
/// refusal is that the layout is the *only* thing it shares with a run table.
/// A tracked deletion covers characters that are still in the text and are not
/// going to be shown, and an edit that slides its index the way it slides a
/// style run moves a deletion marker onto characters nobody deleted.
#[test]
fn an_edit_through_a_tracked_change_is_refused_by_name() {
    const STORAGE: u64 = 4242;
    const CHANGE: u64 = 4243;

    let reference = |target: u64| {
        iwork::pb::Value::Bytes(
            iwork::pb::Message {
                fields: vec![iwork::pb::Field {
                    number: 1,
                    value: iwork::pb::Value::Varint(target),
                }],
            }
            .encode(),
        )
    };
    let entry = |index: u64, target: u64| iwork::pb::Message {
        fields: vec![
            iwork::pb::Field {
                number: 1,
                value: iwork::pb::Value::Varint(index),
            },
            iwork::pb::Field {
                number: 2,
                value: reference(target),
            },
        ],
    };

    let mut storage = iwork::pb::Message::default();
    storage.set(1, iwork::pb::Value::Varint(0));
    storage.set(3, iwork::pb::Value::Bytes(b"The quick brown fox".to_vec()));
    // The table wraps its entries in repeated field 1, as
    // pages-tracked.pages showed — an entry set directly on the storage is
    // the shape the schema misled everyone into.
    let table = iwork::pb::Message {
        fields: vec![iwork::pb::Field {
            number: 1,
            value: iwork::pb::Value::Bytes(entry(4, CHANGE).encode()),
        }],
    };
    storage.set(
        iwork::text::DELETION_TABLE,
        iwork::pb::Value::Bytes(table.encode()),
    );

    // `TSWP.ChangeArchive`: kind 2 is a deletion, and there is no kind 0.
    let mut change = iwork::pb::Message::default();
    change.set(1, iwork::pb::Value::Varint(2));

    let object = |identifier: u64, message_type: u32, payload: &iwork::pb::Message| {
        iwork::iwa::ArchiveObject {
            identifier,
            messages: vec![iwork::iwa::ArchiveMessage {
                message_type,
                version: vec![1, 0, 5],
                extra: Vec::new(),
                payload: payload.encode(),
            }],
            extra: Vec::new(),
        }
    };
    let objects = vec![
        object(STORAGE, iwork::TYPE_STORAGE, &storage),
        object(CHANGE, iwork::annotations::TYPE_CHANGE, &change),
    ];
    let package = iwork::Package {
        entries: vec![(
            "Index/Document.iwa".to_string(),
            iwork::iwa::serialize(&objects),
        )],
        ..Default::default()
    };
    let mut doc = Document::from_package(package).unwrap();

    // The reader sees it, and says so rather than pretending the document is
    // ordinary.
    let annotations = doc.annotations();
    assert_eq!(annotations.tracked_storages, vec![STORAGE]);
    assert_eq!(annotations.changes.len(), 1);
    assert_eq!(
        annotations.changes[0].kind,
        iwork::annotations::ChangeKind::Deletion
    );
    assert_eq!(
        annotations.changes[0].anchor,
        Some((STORAGE, iwork::text::DELETION_TABLE, 4))
    );

    match doc.set_text(STORAGE, "anything at all") {
        Err(Error::TrackedChanges { storage, field }) => {
            assert_eq!(storage, STORAGE);
            assert_eq!(field, iwork::text::DELETION_TABLE);
        }
        other => panic!("expected a TrackedChanges refusal, got {other:?}"),
    }
    // And an insertion table is refused just the same.
    assert!(matches!(
        doc.delete_text(STORAGE, 0..3),
        Err(Error::TrackedChanges { .. })
    ));
}

/// The app reads back text this crate pointed at a style it also made.
///
/// `create_text_style` had an app round-trip and `apply_text_style` did not:
/// the style existed in a document the app opened, and whether the *run*
/// pointing at it survived was never watched. Ground rule 1a. Off unless
/// `IWORK_APP_CHECK=1`.
#[test]
fn pages_resaves_text_pointed_at_a_style_this_crate_applied() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("pages-styled.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.body_storage().expect("a body");
    let template = doc
        .text_styles()
        .into_iter()
        .find(|style| style.kind == StyleKind::Paragraph)
        .expect("a paragraph style to copy")
        .identifier;
    let made = doc.create_text_style(template, "Kicker").unwrap();
    doc.apply_text_style(storage, 0..4, made.identifier)
        .unwrap();
    let before = doc.storage_text(storage).unwrap();

    let out = std::env::temp_dir().join("iwork-applied-style.pages");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    // The app loads the document into its own model and writes it back out;
    // a run pointing at a style it will not accept does not survive that.
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/resave.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Pages would not resave a document with an applied style:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let after = Document::open(&out).unwrap();
    assert_eq!(
        after.storage_text(storage).unwrap(),
        before,
        "the app rewrote the text the style was applied to"
    );
    assert!(
        after
            .text_styles()
            .iter()
            .any(|s| s.identifier == made.identifier),
        "the app dropped the style"
    );
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_file(&out);
}

/// **Where styling actually works, measured in the app.**
///
/// The crate can create a text style and point a run at it, and the archives
/// come out right either way — but whether the *app draws it* depends on
/// where the style came from, and the difference is stark enough to be worth
/// a test that says so.
///
/// In a document Pages wrote, pointing paragraphs at its own `Title`,
/// `Heading` and `Body` works: asked afterwards for the size of every
/// paragraph, Pages answered `30.0 18.0 11.0 18.0 11.0` — exactly the roles
/// applied, in order.
///
/// In a document from [`Document::new`], the same crate calls are accepted,
/// `iwork check` is clean, Pages opens it — and Pages draws every paragraph
/// at `12.0`, because a blank document's stylesheet holds one paragraph style
/// (`Body`), and a style copied from it is not one Pages honours.
///
/// This test asserts the shape of that: what a real document has to point at,
/// and what a blank one does not.
#[test]
fn a_blank_document_has_one_paragraph_style_and_a_real_one_has_a_hierarchy() {
    let blank = Document::new(Kind::Pages).unwrap();
    let named: Vec<String> = blank
        .text_styles()
        .into_iter()
        .filter(|s| s.kind == StyleKind::Paragraph)
        .filter_map(|s| s.name)
        .collect();
    // `Body`, plus the table styles a table added to such a document needs.
    // What is *not* there is any document structure to point a heading at.
    assert!(named.iter().any(|n| n == "Body"), "{named:?}");
    for role in ["Title", "Heading", "Heading 2", "Heading 3", "Caption"] {
        assert!(
            !named.iter().any(|n| n == role),
            "a blank Pages document has no {role}; if it now does, check whether \
             Pages draws a heading this crate wrote from nothing — and if it \
             does, this test has done its job: {named:?}"
        );
    }

    let real = match generated("pages-styled.pages") {
        Some(path) => Document::open(&path).unwrap(),
        None => {
            eprintln!("no pages-styled.pages — skipping (run scripts/make-fixtures.sh)");
            return;
        }
    };
    let named: Vec<String> = real
        .text_styles()
        .into_iter()
        .filter(|s| s.kind == StyleKind::Paragraph)
        .filter_map(|s| s.name)
        .collect();
    for role in ["Title", "Heading", "Heading 2", "Body"] {
        assert!(
            named.iter().any(|n| n == role),
            "a document Pages wrote carries {role}: {named:?}"
        );
    }
}
