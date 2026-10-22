//! Inserting a row into a table — the byte level, then the app.
//!
//! The write is deliberately narrow: a plain rectangular table held in one
//! tile, no categories, filters, pivots, conditional highlighting, hidden or
//! collapsed rows, footer rows, merges at or below the insertion, or formulas
//! whose references would shift. Everything else is refused **by name**, and a
//! refused insert has to leave the document byte for byte as it was — which is
//! what most of this file asserts. The one supported case is then handed to
//! Numbers behind `IWORK_APP_CHECK=1`: one more row, the new row empty, and
//! every row below the insertion still holding its value and its format.

use std::path::{Path, PathBuf};

use iwork::table::{CellValue, Decimal, Uuid};
use iwork::Document;

fn generated(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name);
    path.exists().then_some(path)
}

fn open(name: &str) -> Option<Document> {
    let path = generated(name)?;
    Some(Document::open(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
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

/// The supported case: a plain 17×3 table with no formulas, merges, filters or
/// categories. A row goes in at index 8; the table grows by one, the new row is
/// empty, every row below keeps its value, and every table in the document still
/// adds up.
#[test]
fn inserting_a_row_into_a_plain_table() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();

    let before = doc.table("Formate").unwrap();
    assert_eq!(before.rows, 17);
    let below: Vec<CellValue> = (8..17).map(|r| before.value(r, 0)).collect();

    doc.insert_row("Formate", 8).unwrap();

    let after = doc.table("Formate").unwrap();
    assert_eq!(after.rows, 18, "the table did not grow");
    // The new row is empty across every column.
    for column in 0..after.columns {
        assert_eq!(
            after.value(8, column),
            CellValue::Empty,
            "the inserted row is not empty at column {column}"
        );
    }
    // Everything above is where it was.
    assert_eq!(after.value(0, 0), CellValue::Text("Format".into()));
    assert_eq!(after.value(7, 0), CellValue::Text("Text".into()));
    // Everything below moved down one and kept its value.
    for (offset, value) in below.into_iter().enumerate() {
        assert_eq!(
            after.value(9 + offset, 0),
            value,
            "the row that was at {} did not move to {}",
            8 + offset,
            9 + offset
        );
    }
    // The document is still internally consistent — every count adds up.
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

/// The write touches only the model, the tile and the row bucket streams (the
/// UUID map shares the model's stream), and leaves every other entry
/// byte-identical.
#[test]
fn inserting_a_row_rewrites_only_the_streams_it_must() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.insert_row("Formate", 8).unwrap();

    let changed = doc.changed_streams();
    assert_eq!(changed.len(), 3, "{changed:?}");
    assert!(changed.iter().any(|s| s.contains("CalculationEngine")));
    assert!(changed
        .iter()
        .any(|s| s.contains("HeaderStorageBucket-904914")));
    assert!(changed.iter().any(|s| s.ends_with("Tile.iwa")));

    // Save it out and confirm every untouched entry is the original's bytes.
    let out = std::env::temp_dir().join("iwork-insert-row-streams.numbers");
    let _ = std::fs::remove_file(&out);
    let changed: Vec<String> = changed.iter().map(|s| s.to_string()).collect();
    doc.save(&out).unwrap();
    let before = iwork::Package::read(generated("numbers-formats.numbers").unwrap()).unwrap();
    let after = iwork::Package::read(&out).unwrap();
    for entry in before.names() {
        if changed.iter().any(|c| c == entry) {
            continue;
        }
        assert_eq!(before.get(entry), after.get(entry), "{entry} was rewritten");
    }
    let _ = std::fs::remove_file(&out);
}

/// The row UUID map gains one entry, stays a clean permutation, and stays sorted
/// by UUID — the invariant a positional read would have missed.
#[test]
fn the_uuid_map_gains_a_sorted_unique_entry() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    let model = doc.table("Formate").unwrap().model;
    let uid_map = doc
        .archive(model)
        .unwrap()
        .bytes(46)
        .and_then(|raw| iwork::pb::decode_nested(raw).and_then(|m| m.varint(1)))
        .expect("the model names a ColumnRowUIDMap");

    doc.insert_row("Formate", 8).unwrap();

    // Every row 0..18 has exactly one UUID, and every UUID one index.
    let table = doc.table("Formate").unwrap();
    let mut uuids = std::collections::BTreeSet::new();
    for row in 0..table.rows {
        let uuid = table
            .uids
            .row_uid(row)
            .unwrap_or_else(|| panic!("row {row} has no UUID after the insert"));
        assert!(
            uuids.insert(uuid),
            "row {row}'s UUID collides with another's"
        );
        assert_eq!(
            table.uids.row(uuid),
            Some(row),
            "the map is not a bijection"
        );
    }
    assert_eq!(uuids.len(), 18);

    // Field 4 of the archive is sorted by the 128-bit value, high half first.
    let archive = doc.archive(uid_map).unwrap();
    let sorted: Vec<Uuid> = archive
        .all(4)
        .filter_map(|value| match value {
            iwork::pb::Value::Bytes(raw) => iwork::pb::decode_nested(raw).map(|m| Uuid::decode(&m)),
            _ => None,
        })
        .collect();
    assert_eq!(sorted.len(), 18);
    let mut in_order = sorted.clone();
    in_order.sort_by_key(|uuid| (uuid.upper, uuid.lower));
    assert_eq!(sorted, in_order, "the row UUID list is not sorted by UUID");
    // Fields 5 and 6 are inverse permutations of one another.
    let index_for_uid: Vec<u64> = archive
        .all(5)
        .filter_map(|v| match v {
            iwork::pb::Value::Varint(n) => Some(*n),
            _ => None,
        })
        .collect();
    let uid_for_index: Vec<u64> = archive
        .all(6)
        .filter_map(|v| match v {
            iwork::pb::Value::Varint(n) => Some(*n),
            _ => None,
        })
        .collect();
    assert_eq!(index_for_uid.len(), 18);
    assert_eq!(uid_for_index.len(), 18);
    for (position, &index) in index_for_uid.iter().enumerate() {
        assert_eq!(
            uid_for_index[index as usize] as usize, position,
            "field 5 and field 6 disagree"
        );
    }
}

/// Appending — `at == rows` — is allowed and grows the table at the end.
#[test]
fn a_row_can_be_appended() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.insert_row("Formate", 17).unwrap();
    let table = doc.table("Formate").unwrap();
    assert_eq!(table.rows, 18);
    assert_eq!(table.value(17, 0), CellValue::Empty);
    assert_eq!(table.value(16, 0), CellValue::Text("Datum".into()));
    assert!(table.audit().is_empty(), "{:?}", table.audit());
}

/// The inserted row is genuinely empty — it has no `TileRowInfo` at all, the
/// same shape the app gives a row with no cells. Filling it is what brings the
/// row into being: `set_cell` writes the `TileRowInfo` the row never had, in
/// the shape `Document::new` writes and all three apps open, and bumps the
/// tile's `numrows` to count it. Inserting a row and filling it now compose.
#[test]
fn the_inserted_row_can_then_be_filled() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.insert_row("Formate", 8).unwrap();
    doc.set_cell("Formate", 8, 0, CellValue::Text("Neu".into()))
        .unwrap();
    doc.set_cell(
        "Formate",
        8,
        1,
        CellValue::Number(Decimal::parse("42").unwrap()),
    )
    .unwrap();

    let table = doc.table("Formate").unwrap();
    assert_eq!(table.value(8, 0), CellValue::Text("Neu".into()));
    assert_eq!(
        table.value(8, 1),
        CellValue::Number(Decimal::parse("42").unwrap())
    );
    // The neighbours the new row was spliced between are where they were.
    assert_eq!(table.value(7, 0), CellValue::Text("Text".into()));
    assert_eq!(table.value(9, 0), CellValue::Text("Zahlensystem".into()));
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// A row that never had a cell is not only the inserted one: every row of a
/// table `Document::new` makes has a `TileRowInfo` already, but a document the
/// app wrote has rows that hold nothing at all. Writing into one of those goes
/// down the same path, and the row appears in the tile in `tile_row_index`
/// order — which is how every tile in the corpus keeps them.
#[test]
fn a_row_the_app_left_empty_can_be_given_its_first_cell() {
    fixture!("numbers-values.numbers");
    let mut doc = open("numbers-values.numbers").unwrap();
    // Find a table with a row holding no cells at all.
    let empty = doc
        .tables()
        .into_iter()
        .find_map(|table| {
            (0..table.rows)
                .find(|&row| (0..table.columns).all(|c| table.value(row, c) == CellValue::Empty))
                .map(|row| (table.name.clone(), row))
        })
        .expect("some table has an empty row");
    let (name, row) = empty;

    doc.set_cell(&name, row, 0, CellValue::Text("Erste".into()))
        .unwrap();
    let table = doc.table(&name).unwrap();
    assert_eq!(table.value(row, 0), CellValue::Text("Erste".into()));
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    // The tile still decodes, and re-encodes to what was written.
    let out = std::env::temp_dir().join("iwork-first-cell.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();
    let again = Document::open(&out).unwrap();
    assert_eq!(
        again.table(&name).unwrap().value(row, 0),
        CellValue::Text("Erste".into())
    );
    let _ = std::fs::remove_file(&out);
}

/// The app is the oracle for the row that was not there.
///
/// Off unless `IWORK_APP_CHECK=1`. A `TileRowInfo` built from nothing is
/// exactly the kind of object ground rule 3 warns about, so the measure is
/// Numbers opening the document and reporting the value in the row that had no
/// storage at all.
#[test]
fn numbers_reads_back_a_cell_in_a_row_that_had_none() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let _ = fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.insert_row("Formate", 8).unwrap();
    doc.set_cell("Formate", 8, 0, CellValue::Text("Neu".into()))
        .unwrap();

    let out = std::env::temp_dir().join("iwork-first-cell-app.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open the document:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        text.lines().any(|line| {
            let field: Vec<&str> = line.split('\t').collect();
            field.first() == Some(&"cell")
                && field.get(1) == Some(&"A9")
                && field.get(3) == Some(&"Neu")
        }),
        "the app did not report A9 as the written text:\n{text}"
    );
    let _ = std::fs::remove_file(&out);
}

/// Everything the writer will not do, refused by name, and each refusal leaves
/// the document byte for byte as it was.
#[test]
fn insert_row_refuses_what_it_cannot_verify() {
    // Out of bounds.
    if let Some(mut doc) = open("numbers-formats.numbers") {
        let err = doc
            .insert_row("Formate", 99)
            .expect_err("past the end")
            .to_string();
        assert!(err.contains("goes in at 0..=17"), "{err}");
        assert!(doc.changed_streams().is_empty());
    }

    // A merge at or below the insertion.
    if let Some(mut doc) = open("numbers-formats.numbers") {
        let err = doc
            .insert_row("Verbunden", 0)
            .expect_err("a merge would shift")
            .to_string();
        assert!(err.contains("merge"), "{err}");
        assert!(doc.changed_streams().is_empty());
    }

    // A bounded formula whose range the insertion would cross.
    if let Some(mut doc) = open("numbers-large.numbers") {
        let err = doc
            .insert_row("Zeilen", 5)
            .expect_err("a formula range crosses row 5")
            .to_string();
        assert!(err.contains("formula"), "{err}");
        assert!(doc.changed_streams().is_empty());
    }

    // A categorised table.
    if let Some(mut doc) = open("numbers-categories.numbers") {
        let table = doc.tables().into_iter().next().map(|t| t.identifier);
        if let Some(id) = table {
            let err = doc
                .insert_row(&id.to_string(), 1)
                .expect_err("categorised")
                .to_string();
            assert!(err.contains("categorised"), "{err}");
            assert!(doc.changed_streams().is_empty());
        }
    }

    // A footer row.
    if let Some(mut doc) = open("numbers-sorted.numbers") {
        let err = doc
            .insert_row("Reading Log", 2)
            .expect_err("footer row")
            .to_string();
        assert!(err.contains("footer"), "{err}");
        assert!(doc.changed_streams().is_empty());
    }
}

/// A safe formula stays computable: inserting a row into `Zweite Tabelle` above
/// its `SUM(B2:B3)` host, but with both range ends on the host's side of the
/// insertion, is allowed and keeps the table consistent.
#[test]
fn a_relative_reference_that_shifts_with_its_host_is_allowed() {
    fixture!("numbers-values.numbers");
    let mut doc = open("numbers-values.numbers").unwrap();
    // Row 1 is above B4 (row 3) and its range B2:B3 (rows 1,2). Inserting there
    // shifts host and range together, so the reference stays correct.
    doc.insert_row("Zweite Tabelle", 1).unwrap();
    let table = doc.table("Zweite Tabelle").unwrap();
    assert_eq!(table.rows, 5);
    assert_eq!(table.value(1, 0), CellValue::Empty);
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

// -- the oracle --------------------------------------------------------------

/// Ask Numbers to open the inserted-row document and read it back.
///
/// Off unless `IWORK_APP_CHECK=1`. The two halves matter equally: the table
/// reports one more row with the new one empty, and every row below the
/// insertion reads back the value it had before — which a writer that damaged
/// the tile's row indices or the row bucket would fail.
#[test]
fn numbers_reads_back_an_inserted_row() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let _ = fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    let before: Vec<String> = (0..17)
        .map(|r| doc.table("Formate").unwrap().value(r, 0).to_text())
        .collect();
    doc.insert_row("Formate", 8).unwrap();

    let out = std::env::temp_dir().join("iwork-insert-row.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open the document with an inserted row:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();

    // Read the Formate table's row count and its column-A values back.
    let mut rows = 0usize;
    let mut said: std::collections::BTreeMap<String, String> = std::collections::BTreeMap::new();
    let mut in_formate = false;
    for line in text.lines() {
        let field: Vec<&str> = line.split('\t').collect();
        match field.first() {
            Some(&"table") => {
                in_formate = field.get(1) == Some(&"Formate");
                if in_formate {
                    rows = field.get(2).and_then(|n| n.parse().ok()).unwrap_or(0);
                }
            }
            Some(&"cell") if in_formate && field.len() >= 4 => {
                said.insert(field[1].to_string(), field[3].to_string());
            }
            _ => {}
        }
    }

    assert_eq!(rows, 18, "the app did not see one more row");
    // The inserted row is empty.
    assert_eq!(said.get("A9").map(String::as_str), Some(""), "A9 not empty");
    // A1..A8 are unchanged; A9 is new; A10..A18 are the old A9..A17.
    for (index, value) in before.iter().enumerate() {
        let name = format!("A{}", if index < 8 { index + 1 } else { index + 2 });
        assert_eq!(
            said.get(&name).map(String::as_str),
            Some(value.as_str()),
            "{name} did not read back as {value:?}"
        );
    }
    let _ = std::fs::remove_file(&out);
}

/// A row inserted above a tile boundary crosses it.
///
/// A row's absolute index is `tileid * tile_size + tile_row_index`, so the last
/// row of tile 0 shifting down one becomes the *first row of tile 1* — the
/// `TileRowInfo` leaves one object and joins another. That is why the insert
/// does not edit tiles in place: it gathers every row of every tile by absolute
/// index, shifts, and lays them back out into whichever tile each now belongs
/// to.
///
/// The fixtures cannot show this — the only multi-tile one has a formula over
/// `C2:C301`, which any insert above row 300 rightly breaks — so the table is
/// one this crate makes: 300 rows, three columns, no formulas.
#[test]
fn a_row_inserted_below_a_tile_boundary_crosses_it() {
    let mut doc = Document::new_spreadsheet("Blatt", "Lang", 300, 3).unwrap();
    for row in 0..300 {
        doc.set_cell("Lang", row, 0, CellValue::Text(format!("Zeile {row}")))
            .unwrap();
    }

    doc.insert_row("Lang", 100).unwrap();

    let table = doc.table("Lang").unwrap();
    assert_eq!(table.rows, 301);
    assert_eq!(table.value(99, 0).to_text(), "Zeile 99");
    assert_eq!(table.value(100, 0), CellValue::Empty, "the new row");
    assert_eq!(table.value(101, 0).to_text(), "Zeile 100");
    // The boundary itself: 255 was the last row of tile 0 and is now the first
    // row of tile 1.
    assert_eq!(table.value(255, 0).to_text(), "Zeile 254");
    assert_eq!(table.value(256, 0).to_text(), "Zeile 255");
    assert_eq!(table.value(300, 0).to_text(), "Zeile 299");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(table.audit().is_empty(), "{:?}", table.audit());
}

/// A table that fills every tile it has is refused: the row would need a tile
/// of its own, which is a new object *and* a new component.
#[test]
fn a_table_that_fills_its_tiles_is_refused() {
    let mut doc = Document::new_spreadsheet("Blatt", "Voll", 256, 2).unwrap();
    let err = doc
        .insert_row("Voll", 0)
        .expect_err("256 rows is one full tile")
        .to_string();
    assert!(err.contains("open another one"), "{err}");
    assert!(doc.changed_streams().is_empty());
}

/// A table this crate makes has the identities an insert needs.
///
/// A `TST.ColumnRowUIDMapArchive` is where a new row's UUID goes, and a table
/// without one cannot be given a row at all. A new table gets a full map — one
/// UUID per row and per column, sorted by the 128-bit value the way the app
/// keeps them.
#[test]
fn a_new_table_carries_the_uid_map_an_insert_needs() {
    let doc = Document::new_spreadsheet("Blatt", "Neu", 8, 4).unwrap();
    let map = doc
        .objects()
        .find(|(_, object)| object.message_type() == 6267)
        .map(|(_, object)| object.identifier)
        .expect("a new table has a ColumnRowUIDMap");
    let archive = doc.archive(map).unwrap();
    let count = |number: u32| archive.fields.iter().filter(|f| f.number == number).count();
    assert_eq!(count(1), 4, "one UUID per column");
    assert_eq!(count(2), 4);
    assert_eq!(count(3), 4);
    assert_eq!(count(4), 8, "one UUID per row");
    assert_eq!(count(5), 8);
    assert_eq!(count(6), 8);

    let uuids: Vec<Uuid> = archive
        .fields
        .iter()
        .filter(|f| f.number == 4)
        .filter_map(|f| match &f.value {
            iwork::pb::Value::Bytes(raw) => iwork::pb::decode_nested(raw).map(|m| Uuid::decode(&m)),
            _ => None,
        })
        .collect();
    let mut sorted = uuids.clone();
    sorted.sort_by_key(|u| (u.upper, u.lower));
    assert_eq!(
        uuids, sorted,
        "the app keeps them sorted by the 128-bit value"
    );
    let unique: std::collections::BTreeSet<Uuid> = uuids.iter().copied().collect();
    assert_eq!(unique.len(), uuids.len(), "a row UUID identifies one row");
}

/// Numbers opens the cross-tile insert and writes it back. Off unless
/// `IWORK_APP_CHECK=1`.
///
/// The resave is the measure that matters here: the app loaded 301 rows spread
/// over two tiles into its own model — including the row that changed tiles —
/// and wrote them out again. A writer that left a `TileRowInfo` in the wrong
/// tile would produce a document that opens and shows the wrong rows.
#[test]
fn numbers_resaves_a_row_inserted_across_a_tile_boundary() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = Document::new_spreadsheet("Blatt", "Lang", 300, 3).unwrap();
    for row in 0..300 {
        doc.set_cell("Lang", row, 0, CellValue::Text(format!("Zeile {row}")))
            .unwrap();
    }
    doc.insert_row("Lang", 100).unwrap();

    let out = std::env::temp_dir().join("iwork-cross-tile.numbers");
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/resave.sh");
    let status = std::process::Command::new(&script)
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(status.success(), "Numbers would not resave the document");

    let after = Document::open(&out).unwrap();
    let table = after.table("Lang").unwrap();
    assert_eq!(table.rows, 301);
    assert_eq!(table.value(100, 0), CellValue::Empty);
    assert_eq!(table.value(255, 0).to_text(), "Zeile 254");
    assert_eq!(
        table.value(256, 0).to_text(),
        "Zeile 255",
        "across the boundary"
    );
    assert_eq!(table.value(300, 0).to_text(), "Zeile 299");
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_dir_all(&out);
    let _ = std::fs::remove_file(&out);
}

// -- deleting a row ----------------------------------------------------------

/// The supported case, and the mirror of the insert: the table shrinks by one,
/// every row below moves up keeping its value *and* its format, and every count
/// in the document still adds up — including the per-column cell counts, which
/// a deleted row takes one out of each of.
#[test]
fn deleting_a_row_from_a_plain_table() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    let before = doc.table("Formate").unwrap();
    let below: Vec<CellValue> = (9..17).map(|r| before.value(r, 0)).collect();
    let formats: Vec<String> = (9..17)
        .map(|r| {
            before
                .cell(r, 1)
                .map(|cell| cell.format.as_str().to_string())
                .unwrap_or_default()
        })
        .collect();

    doc.delete_row("Formate", 8).unwrap();

    let after = doc.table("Formate").unwrap();
    assert_eq!(after.rows, 16, "the table did not shrink");
    assert_eq!(after.value(7, 0), CellValue::Text("Text".into()));
    for (offset, value) in below.into_iter().enumerate() {
        assert_eq!(
            after.value(8 + offset, 0),
            value,
            "the row that was at {} did not move up to {}",
            9 + offset,
            8 + offset
        );
    }
    for (offset, format) in formats.into_iter().enumerate() {
        assert_eq!(
            after
                .cell(8 + offset, 1)
                .map(|cell| cell.format.as_str().to_string())
                .unwrap_or_default(),
            format,
            "the format at row {} did not move with its value",
            8 + offset
        );
    }
    assert!(after.audit().is_empty(), "{:?}", after.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The deleted row's cells give their references back, or the lists' counts and
/// the cells that point at them stop agreeing — which is exactly what `audit`
/// reports and what a document this crate wrote must never show.
#[test]
fn a_deleted_row_gives_its_references_back() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    // Row 8 holds "Zahlensystem" and a number in its own format.
    let before = doc.table("Formate").unwrap();
    let text = before.value(8, 0).to_text();
    assert!(!text.is_empty());
    let strings = |doc: &Document| {
        doc.table("Formate")
            .unwrap()
            .to_rows()
            .concat()
            .iter()
            .filter(|value| **value == text)
            .count()
    };
    assert_eq!(strings(&doc), 1);
    doc.delete_row("Formate", 8).unwrap();
    assert_eq!(strings(&doc), 0, "the text survived its row");
    assert!(doc.table("Formate").unwrap().audit().is_empty());
}

/// A row whose deletion cannot be maintained is refused, and the document is
/// left byte for byte as it was.
#[test]
fn delete_row_refuses_what_it_cannot_verify() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    for (table, at, expected) in [
        ("Formate", 99, "the table has 17 row(s)"),
        ("Formate", 0, "header"),
        ("Verbunden", 1, "merge"),
    ] {
        let error = doc
            .delete_row(table, at)
            .expect_err(&format!("{table} row {at} was not refused"))
            .to_string();
        assert!(error.contains(expected), "{table} {at}: {error:?}");
    }
    assert!(doc.changed_streams().is_empty());

    // A formula that names the row, from another table.
    if let Some(mut doc) = open("numbers-values.numbers") {
        let error = doc
            .delete_row("Zweite Tabelle", 1)
            .expect_err("a formula sums that row")
            .to_string();
        assert!(error.contains("formula"), "{error}");
        assert!(doc.changed_streams().is_empty());
    }
}

/// A delete across a tile boundary: the first row of tile 1 becomes the last
/// row of tile 0, which is the insert's crossing in reverse.
#[test]
fn a_deleted_row_moves_one_across_a_tile_boundary() {
    let mut doc = Document::new_spreadsheet("Blatt", "Lang", 300, 2).unwrap();
    let block: Vec<Vec<CellValue>> = (0..300)
        .map(|row| vec![CellValue::Text(format!("Zeile {row}"))])
        .collect();
    doc.set_block("Lang", (0, 0), &block).unwrap();

    doc.delete_row("Lang", 100).unwrap();

    let table = doc.table("Lang").unwrap();
    assert_eq!(table.rows, 299);
    assert_eq!(table.value(99, 0).to_text(), "Zeile 99");
    assert_eq!(table.value(100, 0).to_text(), "Zeile 101");
    // 256 was the first row of tile 1 and is now the last of tile 0.
    assert_eq!(table.value(255, 0).to_text(), "Zeile 256");
    assert_eq!(table.value(298, 0).to_text(), "Zeile 299");
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The app is the oracle. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_reads_back_a_table_with_a_row_deleted() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let _ = fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.delete_row("Formate", 8).unwrap();
    let out = std::env::temp_dir().join("iwork-delete-row.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open the document:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        text.contains("table\tFormate\t16\t3"),
        "the app did not see 16 rows:\n{text}"
    );
    assert!(
        !text.contains("Zahlensystem"),
        "the deleted row's text is still there"
    );
    // The row that moved up keeps its value and its control.
    assert!(text.contains("\tA9\ttext\tAnkreuzfeld\t"), "{text}");
    let _ = std::fs::remove_file(&out);
}
