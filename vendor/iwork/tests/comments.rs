//! Authoring a comment — four archives and an attribute table.
//!
//! A comment is not one object. The words, the date and the author live in a
//! `TSD.CommentStorageArchive` (3056); a `TSWP.HighlightArchive` (2013) points
//! at that; the storage's `table_highlight` (23) points at the highlight from
//! the character the comment starts on; and the author is a
//! `TSK.AnnotationAuthorArchive` (212) in the document's one author storage,
//! which is empty until somebody comments.
//!
//! Two things about the table decide whether the document is right. It is
//! **run-anchored**, so an entry says "from here on" and the comment ends where
//! the next entry begins — a *bare* entry, an index with no reference, is not a
//! comment on nothing but the place the previous one stops. And it **starts at
//! 0**, whatever the comment does: a table whose first entry is anywhere else
//! leaves the characters before it with no attribute at all.

use std::path::{Path, PathBuf};

use iwork::annotations::CommentEdit;
use iwork::Document;

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

fn edit(text: &str) -> CommentEdit {
    CommentEdit {
        author: "Prüferin".into(),
        text: text.into(),
    }
}

/// A comment on a document that had none: the author storage fills, the comment
/// reads back where it was put, and everything still hangs together.
#[test]
fn a_comment_can_be_put_on_a_document_that_had_none() {
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.body_storage().unwrap();
    assert!(
        doc.annotations().comments.is_empty(),
        "the fixture starts without one"
    );

    let comment = doc
        .add_comment(storage, 5, 12, &edit("Hier bitte prüfen."))
        .unwrap();

    let annotations = doc.annotations();
    assert_eq!(annotations.comments.len(), 1);
    let read = &annotations.comments[0];
    assert_eq!(read.identifier, comment);
    assert_eq!(read.text.as_deref(), Some("Hier bitte prüfen."));
    assert!(read.created.is_some(), "a comment carries its date");
    assert_eq!(
        annotations
            .authors
            .iter()
            .find(|a| Some(a.identifier) == read.author)
            .and_then(|a| a.name.clone()),
        Some("Prüferin".to_string()),
        "the author was added to the document's one author storage"
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());
}

/// The attribute table is run-anchored and **starts at 0**.
///
/// The entry at 0 covers the text before the comment; the entry at the comment's
/// start points at the highlight; the entry at its end is bare, and is where the
/// comment stops.
#[test]
fn the_anchor_table_starts_at_zero_and_ends_the_run() {
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.body_storage().unwrap();
    doc.add_comment(storage, 5, 12, &edit("x")).unwrap();

    let archive = doc.archive(storage).unwrap();
    let table = archive
        .bytes(iwork::text::HIGHLIGHT_TABLE)
        .and_then(iwork::pb::decode_nested)
        .expect("the storage has a table_highlight now");
    let entries: Vec<(u64, bool)> = table
        .fields
        .iter()
        .filter(|f| f.number == 1)
        .filter_map(|f| match &f.value {
            iwork::pb::Value::Bytes(raw) => iwork::pb::decode_nested(raw),
            _ => None,
        })
        .map(|entry| (entry.varint(1).unwrap_or(0), entry.get(2).is_some()))
        .collect();
    assert_eq!(
        entries,
        vec![(0, false), (5, true), (12, false)],
        "0 covers the text before it, 5 is the anchor, 12 ends the run"
    );
}

/// A second comment elsewhere in the same text joins the same table, and both
/// read back.
#[test]
fn two_comments_share_one_table() {
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.body_storage().unwrap();
    doc.add_comment(storage, 0, 4, &edit("erster")).unwrap();
    doc.add_comment(storage, 20, 30, &edit("zweiter")).unwrap();

    let comments = doc.annotations().comments;
    assert_eq!(comments.len(), 2);
    let mut texts: Vec<String> = comments.iter().filter_map(|c| c.text.clone()).collect();
    texts.sort();
    assert_eq!(texts, ["erster", "zweiter"]);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// One author, however many comments they leave.
#[test]
fn an_author_is_reused_rather_than_added_twice() {
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.body_storage().unwrap();
    doc.add_comment(storage, 0, 4, &edit("erster")).unwrap();
    doc.add_comment(storage, 20, 30, &edit("zweiter")).unwrap();
    assert_eq!(doc.annotations().authors.len(), 1);

    doc.add_comment(
        storage,
        40,
        44,
        &CommentEdit {
            author: "Jemand anders".into(),
            text: "dritter".into(),
        },
    )
    .unwrap();
    assert_eq!(
        doc.annotations().authors.len(),
        2,
        "a new name is a new author"
    );
}

/// Everything the writer will not do, refused by name.
#[test]
fn a_comment_that_cannot_be_honest_is_refused() {
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.body_storage().unwrap();
    let length = doc.storage_text(storage).unwrap().encode_utf16().count() as u64;

    // An empty range anchors a comment to nothing.
    let refusal = doc
        .add_comment(storage, 5, 5, &edit("x"))
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("empty"), "{refusal}");

    // Past the end of the text.
    let refusal = doc
        .add_comment(storage, 0, length + 1, &edit("x"))
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("unit(s) of text"), "{refusal}");

    // A comment with nothing in it.
    let refusal = doc
        .add_comment(storage, 0, 4, &edit(""))
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("no words"), "{refusal}");

    assert!(doc.annotations().comments.is_empty(), "nothing was written");

    // An overlap: the app uses table_overlapping_highlight (25) for that, and
    // no document here has one to write from.
    doc.add_comment(storage, 10, 20, &edit("erster")).unwrap();
    let refusal = doc
        .add_comment(storage, 15, 25, &edit("zweiter"))
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("overlap"), "{refusal}");
}

/// A storage carrying tracked changes is a storage this crate will not edit,
/// and a comment is an edit.
#[test]
fn a_tracked_storage_is_refused() {
    let path = fixture!("pages-tracked.pages");
    let mut doc = Document::open(&path).unwrap();
    let tracked = doc.annotations().tracked_storages.first().copied();
    let Some(storage) = tracked else {
        eprintln!("the fixture has no tracked storage — skipping");
        return;
    };
    let refusal = doc
        .add_comment(storage, 0, 4, &edit("x"))
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("tracked changes"), "{refusal}");
}

/// Pages opens a document with a comment this crate authored and writes it back
/// with the comment on it. Off unless `IWORK_APP_CHECK=1`.
///
/// This is the measure that counts: no scripting dictionary can read a comment
/// back, so the app's own model is the only witness there is.
#[test]
fn pages_resaves_a_comment_this_crate_authored() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let storage = doc.body_storage().unwrap();
    doc.add_comment(storage, 5, 12, &edit("Ein Kommentar aus iwork-rs."))
        .unwrap();

    let out = std::env::temp_dir().join("iwork-comment.pages");
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/resave.sh");
    let status = std::process::Command::new(&script)
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(status.success(), "Pages would not resave the document");

    let after = Document::open(&out).unwrap();
    let annotations = after.annotations();
    assert_eq!(annotations.comments.len(), 1, "Pages dropped the comment");
    assert_eq!(
        annotations.comments[0].text.as_deref(),
        Some("Ein Kommentar aus iwork-rs.")
    );
    assert_eq!(
        annotations.authors.first().and_then(|a| a.name.clone()),
        Some("Prüferin".to_string())
    );
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_file(&out);
}
