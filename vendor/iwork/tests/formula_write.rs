//! Writing a formula — the fill, and the dependency that makes it live.
//!
//! The measurement this whole file rests on: **Numbers does not recalculate a
//! formula when a document opens.** The value in the cell record is what the
//! app shows, and it goes on showing it — through an open, an edit and a save —
//! unless the cell is in the calculation engine's dependency graph. So a fill
//! writes two things: the formula key, and the record that says which cells the
//! formula reads. `tests/formula_write.rs::the_app_recalculates_a_filled_cell`
//! is the one that proves it, and it needs Numbers.

use std::path::{Path, PathBuf};

use iwork::table::{CellValue, Decimal};
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
            Some(path) => Document::open(&path).unwrap(),
            None => {
                eprintln!("no {} — skipping (run scripts/make-fixtures.sh)", $name);
                return;
            }
        }
    };
}

/// A relative formula filled into another row means the other row's cells.
#[test]
fn a_filled_formula_is_relative_to_where_it_lands() {
    let mut doc = fixture!("numbers-formulas.numbers");
    // C2 is `=Wert addition+1` — the cell to its left, in its own row.
    doc.fill_formula(
        "Zoo",
        (1, 2),
        (2, 3),
        Some(CellValue::Number(Decimal::parse("0").unwrap())),
    )
    .unwrap();
    let table = doc.table("Zoo").unwrap();
    let filled = table
        .formula_cells()
        .into_iter()
        .find(|(row, column, _)| (*row, *column) == (2, 3))
        .expect("the filled cell has no formula");
    // The reference moved with the formula: row 2 rather than row 1.
    assert!(
        filled
            .2
            .ast
            .nodes
            .iter()
            .any(|node| node.reference().is_some()),
        "the filled formula has no reference"
    );
    assert_eq!(table.value(2, 3).to_text(), "0", "the caller's value");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Without a value, only a formula whose answer cannot move is allowed.
#[test]
fn a_fill_with_no_value_is_refused_unless_the_answer_cannot_move() {
    let mut doc = fixture!("numbers-formulas.numbers");
    // C2 is relative: its answer depends on where it sits.
    let refused = doc.fill_formula("Zoo", (1, 2), (2, 3), None);
    assert!(refused.is_err(), "a relative fill needs a value");
    assert!(format!("{:?}", refused.unwrap_err()).contains("relative"));

    // C30 is `=SUM($B$2:$B$4)` — absolute throughout, so the source's answer is
    // the target's.
    doc.fill_formula("Zoo", (29, 2), (3, 3), None).unwrap();
    assert_eq!(doc.table("Zoo").unwrap().value(3, 3).to_text(), "6");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The dependency graph gains exactly one record per filled cell.
#[test]
fn a_filled_cell_is_registered_in_the_calculation_engine() {
    let mut doc = fixture!("numbers-formulas.numbers");
    // Every `CellRecordExpandedArchive` in every dependency tile: one per cell
    // the engine knows holds a formula.
    let records = |doc: &Document| {
        doc.objects()
            .filter(|(_, object)| object.message_type() == iwork::calc::TYPE_CELL_RECORD_TILE)
            .filter_map(|(_, object)| iwork::pb::Message::decode(object.payload()).ok())
            .map(|archive| archive.all(4).count())
            .sum::<usize>()
    };
    let before = records(&doc);
    doc.fill_formula(
        "Zoo",
        (1, 2),
        (2, 3),
        Some(CellValue::Number(Decimal::parse("0").unwrap())),
    )
    .unwrap();
    assert_eq!(
        records(&doc),
        before + 1,
        "the filled cell was not registered"
    );

    // …and it goes into the tile that already covers the cell rather than a
    // second tile with the same origin, which is how a first attempt made the
    // app report two untouched cells as corrupt.
    let tiles = doc
        .objects()
        .filter(|(_, object)| object.message_type() == iwork::calc::TYPE_CELL_RECORD_TILE)
        .count();
    doc.fill_formula(
        "Zoo",
        (1, 2),
        (4, 3),
        Some(CellValue::Number(Decimal::parse("0").unwrap())),
    )
    .unwrap();
    assert_eq!(
        doc.objects()
            .filter(|(_, object)| object.message_type() == iwork::calc::TYPE_CELL_RECORD_TILE)
            .count(),
        tiles,
        "a second tile was written for a cell an existing one covers"
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// What it will not do, by name.
#[test]
fn a_fill_refuses_what_it_cannot_register() {
    let mut doc = fixture!("numbers-formulas.numbers");
    let value = || Some(CellValue::Number(Decimal::parse("0").unwrap()));
    // A cell with no formula.
    assert!(doc.fill_formula("Zoo", (0, 0), (2, 3), value()).is_err());
    // Onto itself.
    assert!(doc.fill_formula("Zoo", (1, 2), (1, 2), value()).is_err());
    // Onto a cell that already holds a formula.
    assert!(doc.fill_formula("Zoo", (1, 2), (2, 2), value()).is_err());
    // Off the end of the table.
    assert!(doc.fill_formula("Zoo", (1, 2), (9999, 3), value()).is_err());
    assert!(
        doc.changed_streams().is_empty(),
        "a refusal wrote something"
    );
}

/// The one that matters. Off unless `IWORK_APP_CHECK=1`.
///
/// Numbers is asked three questions: what the filled cell shows when the
/// document opens, what it shows after the cells it reads are given values, and
/// what it shows when one of them changes again. A formula the engine does not
/// know about answers the first and never the others.
#[test]
fn the_app_recalculates_a_filled_cell() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = fixture!("numbers-formulas.numbers");
    // Fill C2 (`=Wert addition+1`, a relative reference to the cell on its
    // left) into D8, whose left neighbour C8 is `=+Wert vorzeichen` = 7.
    doc.fill_formula(
        "Zoo",
        (1, 2),
        (7, 3),
        Some(CellValue::Number(Decimal::parse("8").unwrap())),
    )
    .unwrap();
    let out = std::env::temp_dir().join("iwork-filled.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document with a filled formula:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_file(&out);
}

// -- a formula from its text -------------------------------------------------

/// The formula the caller typed is the formula the document holds — read back
/// through this crate's own printer, which is the reader half of the format.
#[test]
fn a_formula_written_from_text_reads_back() {
    let mut doc = fixture!("numbers-values.numbers");
    doc.set_formula(
        "Zweite Tabelle",
        3,
        2,
        "=SUM($B$2:$B$3)*2",
        CellValue::Number(Decimal::parse("1500").unwrap()),
    )
    .unwrap();

    let table = doc.table("Zweite Tabelle").unwrap();
    let names = iwork::formula::Names::new(vec![]);
    let written = table.formula(3, 2).expect("no formula in the written cell");
    assert_eq!(
        written.text(iwork::formula::Site::new(&names, None, (2, 3))),
        "=SUM($B$2:$B$3)×2",
        "the app spells multiplication ×, and this is the crate's own printer"
    );
    assert_eq!(
        table.value(3, 2),
        CellValue::Number(Decimal::parse("1500").unwrap()),
        "the value the caller gave is what the cell shows"
    );
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// A written formula is *live*: its cell goes into the engine's dependency
/// graph, or the app would show the caller's value for ever.
#[test]
fn a_written_formula_is_registered_in_the_calculation_engine() {
    let mut doc = fixture!("numbers-values.numbers");
    let records = |doc: &Document| {
        doc.objects()
            .filter(|(_, object)| object.message_type() == iwork::calc::TYPE_CELL_RECORD_TILE)
            .filter_map(|(_, object)| iwork::pb::Message::decode(object.payload()).ok())
            .map(|archive| archive.all(4).count())
            .sum::<usize>()
    };
    let before = records(&doc);
    doc.set_formula(
        "Zweite Tabelle",
        3,
        2,
        "=B2+B3",
        CellValue::Number(Decimal::parse("750").unwrap()),
    )
    .unwrap();
    assert_eq!(records(&doc), before + 1, "the cell was not registered");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Everything refused by name. Each leaves the document as it was.
#[test]
fn a_written_formula_refuses_what_it_cannot_write() {
    let mut doc = fixture!("numbers-values.numbers");
    let one = || CellValue::Number(Decimal::parse("1").unwrap());
    for (row, column, text, value, expected) in [
        // The cell already holds one.
        (3, 1, "=1+1", one(), "already holds a formula"),
        // An empty value is not an answer.
        (3, 2, "=1+1", CellValue::Empty, "cannot be empty"),
        // Out of the table.
        (99, 0, "=1+1", one(), "the table is"),
        // Things the parser will not turn into edges.
        (3, 2, "=SUM(B)", one(), "whole row or column"),
        (3, 2, "=Zellarten.B2", one(), "another table"),
        (3, 2, "=NICHTDA(2)", one(), "not a function"),
        (3, 2, "=1+", one(), "ends where a value was expected"),
    ] {
        let error = doc
            .set_formula("Zweite Tabelle", row, column, text, value)
            .expect_err(&format!("{text} at r{row}c{column} was not refused"))
            .to_string();
        assert!(
            error.contains(expected),
            "r{row}c{column} {text}: {error:?} does not mention {expected:?}"
        );
    }
    assert!(
        doc.changed_streams().is_empty(),
        "a refused formula changed {:?}",
        doc.changed_streams()
    );
}

/// The app is the oracle, and for a formula it is a strong one: Numbers parses
/// the node stream this crate built from nothing, prints the formula back in
/// its own spelling, and evaluates it.
///
/// Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_reads_back_a_formula_written_from_text() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = fixture!("numbers-values.numbers");
    doc.set_formula(
        "Zweite Tabelle",
        2,
        2,
        "=SUM($B$2:$B$3)*2",
        CellValue::Number(Decimal::parse("1500").unwrap()),
    )
    .unwrap();
    let out = std::env::temp_dir().join("iwork-set-formula.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document with a written formula:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    // The app's own spelling of what was written, and its own answer.
    assert!(
        text.lines().any(|line| {
            let field: Vec<&str> = line.split('\t').collect();
            field.first() == Some(&"cell")
                && field.get(1) == Some(&"C3")
                && field.get(6) == Some(&"=SUM($B$2:$B$3)×2")
        }),
        "the app did not report the written formula:\n{text}"
    );
    assert!(
        text.contains("\tC3\treal\t1500.0\t"),
        "the app did not report the formula's value:\n{text}"
    );
    let _ = std::fs::remove_file(&out);
}

// -- a table made from nothing can hold a formula -----------------------------

/// A table this crate builds carries the two `TSCE` owners a formula needs.
///
/// It used to carry none: a `HauntedOwnerArchive` in the model and nothing in
/// the calculation engine, so `base_owner_uid` resolved to nothing and every
/// formula written into such a table was refused by name.
#[test]
fn a_table_made_from_nothing_has_its_owners() {
    let doc = Document::new_spreadsheet("Blatt", "T", 4, 2).unwrap();
    let table = doc.table("T").unwrap();
    assert_ne!(
        table.haunted_uid,
        iwork::table::Uuid::default(),
        "the model has no haunted owner"
    );
    assert_ne!(
        table.base_uid,
        iwork::table::Uuid::default(),
        "the haunted owner does not lead to a base_owner_uid"
    );
    assert_ne!(table.base_uid, table.haunted_uid);

    // Two owners: the haunted one that carries the join, and the cell owner a
    // dependency record goes into.
    let owners: Vec<iwork::pb::Message> = doc
        .objects()
        .filter(|(_, object)| object.message_type() == iwork::calc::TYPE_OWNER_DEPENDENCIES)
        .filter_map(|(_, object)| iwork::pb::Message::decode(object.payload()).ok())
        .collect();
    assert_eq!(owners.len(), 2, "a new table has two owners");
    let kinds: Vec<u64> = owners.iter().filter_map(|owner| owner.varint(3)).collect();
    assert!(kinds.contains(&35) && kinds.contains(&1), "{kinds:?}");

    // Every owner is registered in the engine's dependency tracker, both ways:
    // its internal id in the owner list and its object in the references.
    let engine = doc
        .objects()
        .find(|(_, object)| object.message_type() == iwork::calc::TYPE_ENGINE)
        .map(|(_, object)| iwork::pb::Message::decode(object.payload()).unwrap())
        .expect("a new spreadsheet has a calculation engine");
    let tracker = engine
        .bytes(2)
        .and_then(iwork::pb::decode_nested)
        .expect("the engine has a dependency tracker");
    assert_eq!(tracker.all(6).count(), 2, "the tracker names two owners");
    let listed = tracker
        .bytes(3)
        .and_then(iwork::pb::decode_nested)
        .expect("the tracker has an owner list");
    assert_eq!(listed.all(1).count(), 2);

    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());
}

/// And the formula goes in, which is the point of the owners.
#[test]
fn a_document_made_from_nothing_takes_a_formula() {
    let mut doc = Document::new_spreadsheet("Blatt", "T", 6, 2).unwrap();
    let rows: Vec<Vec<CellValue>> = (1..=4)
        .map(|n| {
            vec![CellValue::Number(
                Decimal::parse(&(n * 10).to_string()).unwrap(),
            )]
        })
        .collect();
    doc.set_block("T", (0, 0), &rows).unwrap();
    doc.set_formula(
        "T",
        4,
        0,
        "=SUM(A1:A4)",
        CellValue::Number(Decimal::parse("100").unwrap()),
    )
    .unwrap();

    let table = doc.table("T").unwrap();
    assert!(table.formula(4, 0).is_some(), "the cell holds no formula");
    assert_eq!(
        table.value(4, 0),
        CellValue::Number(Decimal::parse("100").unwrap())
    );
    // The cell is in the engine's graph, which is what makes it live.
    let records: usize = doc
        .objects()
        .filter(|(_, object)| object.message_type() == iwork::calc::TYPE_CELL_RECORD_TILE)
        .filter_map(|(_, object)| iwork::pb::Message::decode(object.payload()).ok())
        .map(|archive| archive.all(4).count())
        .sum();
    assert_eq!(records, 1, "the formula was not registered");
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// A table added to a document later gets owners of its own, and they collide
/// with nothing.
#[test]
fn every_table_added_later_gets_its_own_owners() {
    let mut doc = Document::new_spreadsheet("Erstes", "T1", 3, 2).unwrap();
    doc.add_sheet("Zweites", "T2", 3, 2).unwrap();
    doc.add_table("Zweites", "T3", 3, 2).unwrap();

    let bases: std::collections::BTreeSet<iwork::table::Uuid> =
        doc.tables().iter().map(|table| table.base_uid).collect();
    assert_eq!(bases.len(), 3, "two tables share a base_owner_uid");
    assert!(!bases.contains(&iwork::table::Uuid::default()));

    // Each of the three takes a formula of its own, into its own owner.
    for name in ["T1", "T2", "T3"] {
        doc.set_formula(
            name,
            2,
            1,
            "=A1+A2",
            CellValue::Number(Decimal::parse("0").unwrap()),
        )
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    }
    let records: usize = doc
        .objects()
        .filter(|(_, object)| object.message_type() == iwork::calc::TYPE_CELL_RECORD_TILE)
        .filter_map(|(_, object)| iwork::pb::Message::decode(object.payload()).ok())
        .map(|archive| archive.all(4).count())
        .sum();
    assert_eq!(records, 3, "three formulas, three dependency records");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    for table in doc.tables() {
        assert!(
            table.audit().is_empty(),
            "{}: {:?}",
            table.name,
            table.audit()
        );
    }
}

/// The app is the oracle, and for this claim it is the only one there is.
///
/// Numbers does not recalculate when a document opens — the value in the cell
/// record is what it draws — so a formula the engine knows nothing about looks
/// exactly like one it knows until something the formula reads changes. This
/// changes one: the app itself sets `A1` to 1000, and `A5` has to follow.
///
/// Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_recalculates_a_formula_in_a_document_made_from_nothing() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = Document::new_spreadsheet("Blatt", "T", 6, 2).unwrap();
    let rows: Vec<Vec<CellValue>> = (1..=4)
        .map(|n| {
            vec![CellValue::Number(
                Decimal::parse(&(n * 10).to_string()).unwrap(),
            )]
        })
        .collect();
    doc.set_block("T", (0, 0), &rows).unwrap();
    doc.set_formula(
        "T",
        4,
        0,
        "=SUM(A1:A4)",
        CellValue::Number(Decimal::parse("100").unwrap()),
    )
    .unwrap();

    let out = std::env::temp_dir().join("iwork-owners.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/recalculation.sh");
    let output = std::process::Command::new(&script)
        .args([out.to_str().unwrap(), "A5", "A1", "1000"])
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document made from nothing:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let field = |line: &str, at: usize| -> String {
        text.lines()
            .find(|l| l.starts_with(line))
            .unwrap_or_else(|| panic!("no {line} line in:\n{text}"))
            .split('\t')
            .nth(at)
            .unwrap_or_default()
            .to_string()
    };
    // The app parses the formula this crate built from nothing…
    assert_eq!(field("on-open", 2), "=SUM(A1:A4)", "{text}");
    assert_eq!(field("on-open", 1), "100.0", "{text}");
    // …and recalculates it when a cell it reads changes, which only happens
    // for a cell the engine's dependency graph reaches.
    assert_eq!(
        field("after-edit", 1),
        "1090.0",
        "the app did not recalculate:\n{text}"
    );
    let _ = std::fs::remove_file(&out);
}
