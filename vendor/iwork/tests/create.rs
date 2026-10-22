//! Documents made from nothing.
//!
//! Every other test in this repository starts from a document an app wrote.
//! These start from nothing at all: `Document::new` assembles the object graph,
//! the component index and the identity in memory, and what comes out has to be
//! a document by every measure the crate already has — it decodes, it round
//! trips, `check` finds no broken references, and the text goes in and comes
//! back out.
//!
//! The last measure is the app itself, behind `IWORK_APP_CHECK=1`, and it is
//! the one that matters: `Document::new(Kind::Pages)` was refused by Pages
//! until the component index declared a component's root the way iWork declares
//! one, and nothing on this side of the file could have told us.

use std::path::Path;

use iwork::{Document, Kind};

fn scratch(name: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&path);
    path
}

/// The shape of a new Pages document, asserted where it is fixed by the format
/// rather than by taste: object 1 is the document, 71 is its metadata, and the
/// component index is object 2 and says how high the identifiers have gone.
#[test]
fn a_new_pages_document_has_the_objects_the_format_fixes() {
    let doc = Document::new(Kind::Pages).unwrap();
    assert_eq!(doc.kind(), Kind::Pages);

    let (_, root) = doc.object(1).expect("no object 1");
    assert_eq!(root.message_type(), 10000, "object 1 is TP.DocumentArchive");
    let (stream, metadata) = doc.object(71).expect("no object 71");
    assert_eq!(metadata.message_type(), 11011, "object 71 is the metadata");
    assert_eq!(stream, "Index/DocumentMetadata.iwa");
    let (stream, index) = doc.object(2).expect("no object 2");
    assert_eq!(index.message_type(), iwork::TYPE_PACKAGE_METADATA);
    assert_eq!(stream, "Index/Metadata.iwa");

    // The high-water mark covers every identifier in the package. Understating
    // it is how a document hands out an identifier something already has.
    let highest = doc.objects().map(|(_, o)| o.identifier).max().unwrap();
    assert!(
        doc.last_object_identifier().unwrap() >= highest,
        "the component index understates the highest identifier"
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Every reference that leaves the component it is written in is declared.
///
/// The rule that took two hours: a *root* is declared as `{component}` and
/// anything else as `{component, object}`, and Pages refuses a package that
/// gets it the wrong way round.
#[test]
fn a_new_document_declares_every_reference_that_leaves_a_component() {
    let doc = Document::new(Kind::Pages).unwrap();
    assert!(
        doc.undeclared_references().is_empty(),
        "{:?}",
        doc.undeclared_references()
    );
}

/// It survives being written and read, and reading it back changes nothing.
#[test]
fn a_new_pages_document_round_trips_through_a_file() {
    let path = scratch("iwork-create-roundtrip.pages");
    let made = Document::new(Kind::Pages).unwrap();
    made.save(&path).unwrap();

    let reopened = Document::open(&path).unwrap();
    assert_eq!(reopened.kind(), Kind::Pages);
    assert_eq!(reopened.objects().count(), made.objects().count());
    assert!(reopened.problems().is_empty(), "{:?}", reopened.problems());

    // A save of what was just opened reproduces the file byte for byte, which
    // is the invariant the whole crate is built on.
    let again = scratch("iwork-create-roundtrip-2.pages");
    reopened.save(&again).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&again).unwrap(),
        "a new document does not survive a no-op save"
    );
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_file(&again);
}

/// Two documents are two documents: a new identity every time, or two files
/// made this way would collide in iCloud.
#[test]
fn every_new_document_has_an_identity_of_its_own() {
    let one = Document::new(Kind::Pages).unwrap();
    let two = Document::new(Kind::Pages).unwrap();
    let uuid = |doc: &Document| {
        doc.metadata()
            .unwrap()
            .properties
            .unwrap()
            .document_uuid
            .unwrap()
    };
    assert_ne!(uuid(&one), uuid(&two));

    // …and the three places that say which document this is agree, which is
    // what `problems` checks and what a bad identity looks like.
    let properties = one.metadata().unwrap().properties.unwrap();
    assert_eq!(properties.document_uuid, properties.share_uuid);
    assert_eq!(properties.document_uuid, properties.stable_document_uuid);
    assert!(properties
        .revision
        .unwrap()
        .ends_with(&properties.version_uuid.unwrap()));
}

/// The body is empty, and paragraphs go into it in order.
#[test]
fn paragraphs_go_into_the_body_of_a_new_document() {
    let mut doc = Document::new(Kind::Pages).unwrap();
    let body = doc.body_storage().expect("a new document has a body");
    assert_eq!(doc.storage_text(body).unwrap(), "");

    doc.append_paragraph("First").unwrap();
    doc.append_paragraph("Second").unwrap();
    // The first paragraph brings no newline with it; the second does.
    assert_eq!(doc.storage_text(body).unwrap(), "First\nSecond");
    assert_eq!(doc.paragraph_ranges(body).unwrap().len(), 2);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Paper is a choice, and the only thing it changes.
#[test]
fn paper_sets_the_page_and_nothing_else() {
    let a4 = Document::new_on(Kind::Pages, iwork::create::Paper::A4).unwrap();
    let letter = Document::new_on(Kind::Pages, iwork::create::Paper::Letter).unwrap();
    let setup = |doc: &Document| doc.structure().expect("a Pages document has one").setup;
    assert_eq!(setup(&a4).width.round(), 595.0);
    assert_eq!(setup(&a4).height.round(), 842.0);
    assert_eq!(setup(&a4).paper_id, "iso-a4");
    assert_eq!(setup(&letter).width.round(), 612.0);
    assert_eq!(setup(&letter).height.round(), 792.0);
    assert_eq!(setup(&letter).paper_id, "na-letter");
    assert_eq!(a4.objects().count(), letter.objects().count());
}

/// A document has to be one of the three.
#[test]
fn a_document_of_no_kind_is_refused_by_name() {
    assert!(Document::new(Kind::Unknown).is_err());
}

/// A new deck: one slide, one master, and a size.
#[test]
fn a_new_deck_has_a_slide_and_the_master_it_is_drawn_from() {
    let doc = Document::new(Kind::Keynote).unwrap();
    assert_eq!(doc.kind(), Kind::Keynote);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());

    let show = doc.show().expect("a deck has a show");
    assert_eq!(show.width.round(), 1920.0);
    assert_eq!(show.height.round(), 1080.0);
    assert_eq!(doc.slides().len(), 1);
    // The master is what the slide is drawn from, and it is told from a slide
    // by having a name.
    assert_eq!(doc.slide_layouts().len(), 1);
}

/// A document built up rather than merely made: two sheets, three tables, and
/// a cell written into the last of them.
#[test]
fn a_spreadsheet_can_be_built_up_sheet_by_sheet() {
    use iwork::table::CellValue;

    let mut doc = Document::new_spreadsheet("First", "T1", 4, 3).unwrap();
    let sheet = doc.add_sheet("Second", "T2", 5, 2).unwrap();
    let table = doc.add_table("Second", "T3", 3, 2).unwrap();
    assert!(sheet > 0 && table > 0);

    let names: Vec<String> = doc.tables().iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, ["T1", "T2", "T3"]);
    let sheets: Vec<String> = doc
        .tables()
        .iter()
        .filter_map(|t| t.sheet.clone())
        .collect();
    assert_eq!(sheets, ["First", "Second", "Second"]);

    // A table added to a document is a table like any other.
    doc.set_cell("T3", 2, 1, CellValue::Text("corner".into()))
        .unwrap();
    assert_eq!(doc.table("T3").unwrap().value(2, 1).to_text(), "corner");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());
}

/// Two tables of one name is something the app's formulas cannot tell apart, so
/// it is refused; so is a sheet that is not there.
#[test]
fn adding_a_table_refuses_a_name_or_a_sheet_that_would_not_work() {
    let mut doc = Document::new_spreadsheet("First", "T1", 4, 3).unwrap();
    assert!(
        doc.add_table("First", "T1", 2, 2).is_err(),
        "duplicate name"
    );
    assert!(
        doc.add_table("Nowhere", "T2", 2, 2).is_err(),
        "no such sheet"
    );
    assert!(
        doc.add_sheet("First", "T2", 2, 2).is_err(),
        "duplicate sheet"
    );
    assert!(doc.add_table("First", "T2", 0, 2).is_err(), "no rows");
}

/// A deck grows the same way, and the new slide is drawn from the deck's own
/// layout rather than one invented for it.
#[test]
fn a_deck_can_be_built_up_slide_by_slide() {
    let mut doc = Document::new(Kind::Keynote).unwrap();
    assert_eq!(doc.slides().len(), 1);

    let added = doc.add_slide(None).unwrap();
    assert_eq!(doc.slides().len(), 2);
    assert_eq!(added.index, 1);
    // It hangs off a layout the deck already had.
    let layouts = doc.slide_layouts();
    assert_eq!(layouts.len(), 1);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());

    // …and again, so the second one does not collide with the first.
    doc.add_slide(None).unwrap();
    assert_eq!(doc.slides().len(), 3);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The apps read back what was added. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn the_apps_read_back_an_added_sheet_table_and_slide() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let check = |path: &std::path::Path, expected: &str| {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
        let output = std::process::Command::new(&script)
            .arg(path)
            .arg(expected)
            .output()
            .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
        assert!(
            output.status.success(),
            "the app would not open {}:\n{}\n{}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    };

    let mut doc = Document::new_spreadsheet("First", "T1", 4, 3).unwrap();
    doc.add_sheet("Second", "T2", 5, 2).unwrap();
    doc.set_cell(
        "T2",
        0,
        0,
        iwork::table::CellValue::Text("Auf dem zweiten Blatt".into()),
    )
    .unwrap();
    let out = scratch("iwork-grown.numbers");
    doc.save(&out).unwrap();
    check(&out, "Auf dem zweiten Blatt");
    let _ = std::fs::remove_file(&out);

    let mut deck = Document::new(Kind::Keynote).unwrap();
    deck.add_slide(None).unwrap();
    let out = scratch("iwork-grown.key");
    deck.save(&out).unwrap();
    check(&out, "");
    let _ = std::fs::remove_file(&out);
}

/// The measure that counts, for the deck. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn keynote_opens_a_deck_this_crate_made_from_nothing() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let doc = Document::new(Kind::Keynote).unwrap();
    let out = scratch("iwork-created.key");
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Keynote would not open a deck made from nothing:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_file(&out);
}

/// A new spreadsheet: one sheet, one table, and every cell of it writable.
#[test]
fn a_new_spreadsheet_has_a_table_that_can_be_written_into() {
    use iwork::table::CellValue;

    let mut doc = Document::new_spreadsheet("Sales", "Q1", 4, 3).unwrap();
    assert_eq!(doc.kind(), Kind::Numbers);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    let table = doc.table("Q1").expect("no table");
    assert_eq!((table.rows, table.columns), (4, 3));
    assert_eq!(table.sheet.as_deref(), Some("Sales"));
    assert_eq!(table.cells().len(), 0);

    doc.set_cell("Q1", 0, 0, CellValue::Text("Region".into()))
        .unwrap();
    doc.set_cell("Q1", 3, 2, CellValue::Text("last".into()))
        .unwrap();
    let table = doc.table("Q1").expect("no table");
    assert_eq!(table.value(0, 0).to_text(), "Region");
    assert_eq!(table.value(3, 2).to_text(), "last");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The sizes it will not make, by name.
#[test]
fn a_table_with_no_rows_or_too_many_columns_is_refused() {
    assert!(Document::new_spreadsheet("S", "T", 0, 3).is_err());
    assert!(Document::new_spreadsheet("S", "T", 3, 0).is_err());
    assert!(Document::new_spreadsheet("S", "T", 3, 256).is_err());
    assert!(Document::new_spreadsheet("S", "T", 3, 255).is_ok());
}

/// The measure that counts, for the spreadsheet. Off unless
/// `IWORK_APP_CHECK=1`.
#[test]
fn numbers_opens_a_spreadsheet_this_crate_made_from_nothing() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = Document::new_spreadsheet("Sales", "Q1", 6, 3).unwrap();
    doc.set_cell(
        "Q1",
        0,
        0,
        iwork::table::CellValue::Text("Aus dem Nichts".into()),
    )
    .unwrap();
    let out = scratch("iwork-created.numbers");
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .arg("Aus dem Nichts")
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a spreadsheet made from nothing:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_file(&out);
}

/// The measure that counts. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn pages_opens_a_document_this_crate_made_from_nothing() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = Document::new(Kind::Pages).unwrap();
    doc.append_paragraph("Aus dem Nichts, mit Umlaut: Größe")
        .unwrap();
    let out = scratch("iwork-created.pages");
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .arg("Aus dem Nichts, mit Umlaut: Größe")
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Pages would not open a document made from nothing:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_file(&out);
}

/// The same document saved twice is the same file, byte for byte.
///
/// It was not: the ZIP writer stamped every entry with the clock, so two saves
/// a second apart differed in every local header while every entry's contents
/// matched. Nothing in iWork reads an entry's time, so the stamp is now the ZIP
/// epoch and a save is reproducible.
#[test]
fn saving_the_same_document_twice_gives_the_same_bytes() {
    let one = scratch("iwork-deterministic-1.numbers");
    let two = scratch("iwork-deterministic-2.numbers");
    let doc = Document::new_spreadsheet("Blatt", "T", 3, 2).unwrap();
    doc.save(&one).unwrap();
    doc.save(&two).unwrap();
    assert_eq!(
        std::fs::read(&one).unwrap(),
        std::fs::read(&two).unwrap(),
        "two saves of one document produced different files"
    );
    let _ = std::fs::remove_file(&one);
    let _ = std::fs::remove_file(&two);
}

/// A document made from nothing can be given a **table** — which needs two
/// things it did not have.
///
/// A Pages document keeps its `TST.TableInfoArchive` and `TableModelArchive` in
/// the *calculation engine's* component, so without an engine there is nowhere
/// to put a table; and a table is drawn with seventeen cell styles and eight
/// paragraph styles that have to exist first. Every Pages document the app
/// makes carries both — `pages-plain.pages` holds 102 cell styles and an empty
/// engine with not a table in sight — so this crate writes both too.
#[test]
fn a_new_pages_document_can_be_given_a_table() {
    use iwork::table::CellValue;

    let mut doc = Document::new(Kind::Pages).unwrap();
    assert_eq!(
        doc.objects()
            .filter(|(_, object)| object.message_type() == 4000)
            .count(),
        1,
        "a new Pages document carries a calculation engine"
    );
    assert_eq!(
        doc.objects()
            .filter(|(_, object)| object.message_type() == 6004)
            .count(),
        17,
        "…and the cell styles a table is drawn with"
    );

    doc.append_paragraph("Ein Bericht mit einer Tabelle")
        .unwrap();
    let table = doc.add_table("page 1", "Preise", 3, 2).unwrap();
    assert!(table > 0);

    let mut prices = doc.table_mut("Preise").unwrap();
    prices.set_block("A1", &[vec!["Artikel", "Preis"]]).unwrap();
    prices.set("A2", "Schrauben").unwrap();
    prices.currency("B2", 12.5, "CHF").unwrap();
    assert_eq!(prices.value("A2").unwrap(), CellValue::from("Schrauben"));
    assert!(prices.read().audit().is_empty());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());
}

/// A slide made from nothing can be given **presenter notes**.
///
/// A note is two objects — a `KN.NoteArchive`, which is a reference and nothing
/// else, and a `TSWP.StorageArchive` of kind 4 — and a slide this crate makes
/// has neither, so writing notes used to be refused by name. Both go in the
/// slide's own stream, and the storage is the one a text box uses with its kind
/// changed: the field set was read off `keynote-deck.key`'s own note and is
/// identical.
#[test]
fn a_new_deck_can_be_given_presenter_notes() {
    let mut doc = Document::new(Kind::Keynote).unwrap();
    doc.add_slide(None).unwrap();
    for index in 0..2 {
        let mut slide = doc.slide_mut(index).unwrap();
        assert!(slide.read().note.is_none(), "a new slide has no note");
        slide
            .notes(&format!("Notizen zu Folie {}", index + 1))
            .unwrap();
        assert_eq!(
            slide.read().notes,
            format!("Notizen zu Folie {}", index + 1)
        );
        assert!(slide.read().note.is_some());
    }
    // Writing the notes again edits the storage rather than making a second.
    let notes = doc
        .objects()
        .filter(|(_, object)| object.message_type() == 15)
        .count();
    doc.slide_mut(0).unwrap().notes("Anders").unwrap();
    assert_eq!(
        doc.objects()
            .filter(|(_, object)| object.message_type() == 15)
            .count(),
        notes
    );
    assert_eq!(doc.slides()[0].notes, "Anders");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());
}

/// The apps open both. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn the_apps_open_a_table_and_notes_made_from_nothing() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let check = |path: &std::path::Path, expected: &str| {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
        let output = std::process::Command::new(&script)
            .arg(path)
            .arg(expected)
            .output()
            .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
        assert!(
            output.status.success(),
            "the app would not open {}:\n{}\n{}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    };

    let mut pages = Document::new(Kind::Pages).unwrap();
    pages.append_paragraph("Ein Bericht").unwrap();
    pages.add_table("page 1", "Preise", 3, 2).unwrap();
    pages
        .table_mut("Preise")
        .unwrap()
        .set("A2", "Schrauben")
        .unwrap();
    let out = scratch("iwork-new-table.pages");
    pages.save(&out).unwrap();
    check(&out, "Schrauben");
    let _ = std::fs::remove_file(&out);

    let mut deck = Document::new(Kind::Keynote).unwrap();
    deck.slide_mut(0)
        .unwrap()
        .notes("Notizen aus dem Nichts")
        .unwrap();
    let out = scratch("iwork-new-notes.key");
    deck.save(&out).unwrap();
    check(&out, "Notizen aus dem Nichts");
    let _ = std::fs::remove_file(&out);
}
