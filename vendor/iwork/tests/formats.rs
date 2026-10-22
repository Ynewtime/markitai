//! Data formats and the sizes of rows and columns, written.
//!
//! Two writes that look alike and are not. A row or column's size is one float
//! in a header bucket, and the app reads it back exactly. A *format* is an
//! interned archive plus three things on the cell that have to agree, and the
//! rule that governs it was measured rather than assumed: **a format goes in
//! the slot the cell's value already uses, or it is never drawn.** A percent
//! format in a number cell's number slot is drawn as a percentage; a currency
//! format in the same cell's currency slot is ignored, and Numbers goes on
//! showing a plain number. That measurement is why `set_format` refuses a
//! format whose slot is not the cell's.

use std::path::{Path, PathBuf};

use iwork::table::Format;
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

/// The archives are the ones Numbers wrote for the same formats, field for
/// field — the automatic format is one field, and the decimal count that means
/// "as many as it takes" is 253.
#[test]
fn a_format_archive_is_the_shape_the_app_writes() {
    let bytes = |format: &Format| format.archive().unwrap().encode();
    // {1: 260}
    assert_eq!(bytes(&Format::Automatic), vec![8, 132, 2]);
    // {1: 256, 2: 253, 4: 0, 5: 0} — the number format of `numbers-formats`.
    assert_eq!(
        bytes(&Format::Number { decimals: None }),
        vec![8, 128, 2, 16, 253, 1, 32, 0, 40, 0]
    );
    assert_eq!(
        bytes(&Format::Percent { decimals: Some(1) }),
        vec![8, 130, 2, 16, 1, 32, 0, 40, 0]
    );
    // {1: 257, 2: 2, 3: "CHF", 4: 0, 5: 0, 6: 0}, byte for byte the fixture's.
    assert_eq!(
        bytes(&Format::Currency {
            code: "CHF".into(),
            decimals: Some(2)
        }),
        vec![8, 129, 2, 16, 2, 26, 3, 67, 72, 70, 32, 0, 40, 0, 48, 0]
    );
    // {1: 261, 14: "dd.MM.y"}
    assert_eq!(
        bytes(&Format::DateTime {
            pattern: "dd.MM.y".into()
        }),
        vec![8, 133, 2, 114, 7, 100, 100, 46, 77, 77, 46, 121]
    );

    // What is not a format: a currency that is not a code, an empty pattern.
    assert!(Format::Currency {
        code: "€".into(),
        decimals: None
    }
    .archive()
    .is_err());
    assert!(Format::DateTime {
        pattern: String::new()
    }
    .archive()
    .is_err());
}

/// Writing a format: the cell names the new key, byte 6 says the format was
/// chosen, and the reference counts still add up.
#[test]
fn a_written_format_is_the_one_the_cell_reports() {
    let mut doc = fixture!("numbers-formats.numbers");
    // B2 is a number nobody has formatted; B6 a scientific one.
    assert_eq!(
        doc.set_format("Formate", [(1, 1)], &Format::Percent { decimals: Some(1) })
            .unwrap(),
        1
    );
    assert_eq!(
        doc.set_format("Formate", [(5, 1)], &Format::Number { decimals: Some(3) })
            .unwrap(),
        1
    );
    let table = doc.table("Formate").unwrap();
    assert_eq!(table.cell(1, 1).unwrap().format.as_str(), "percent");
    assert_eq!(table.cell(5, 1).unwrap().format.as_str(), "number");
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// One entry however many cells take it, and a cell already carrying the format
/// is left alone — bytes and reference counts both.
#[test]
fn a_format_is_interned_once_for_the_whole_batch() {
    let mut doc = fixture!("numbers-formats.numbers");
    let entries = |doc: &Document| {
        doc.table("Formate")
            .unwrap()
            .cells()
            .iter()
            .filter_map(|cell| cell.record.format_id())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };
    let before = entries(&doc);
    let format = Format::Percent { decimals: Some(2) };
    doc.set_format("Formate", [(1, 1), (2, 1), (5, 1)], &format)
        .unwrap();
    assert_eq!(
        entries(&doc),
        before + 1 - 1,
        "three cells took more than one new entry, or none"
    );
    assert!(doc.table("Formate").unwrap().audit().is_empty());

    // Writing the same format again changes nothing at all.
    let changed = doc.changed_streams().len();
    assert_eq!(
        doc.set_format("Formate", [(1, 1)], &format).unwrap(),
        0,
        "a cell that already has the format was rewritten"
    );
    assert_eq!(doc.changed_streams().len(), changed);
}

/// A format in a slot the cell does not use would sit in the file and never be
/// drawn, which is worse than a refusal. Measured against the app: a currency
/// format written into a number cell's currency slot is ignored.
#[test]
fn a_format_for_another_slot_is_refused() {
    let mut doc = fixture!("numbers-formats.numbers");
    for (row, format, expected) in [
        (
            2usize,
            Format::Currency {
                code: "EUR".into(),
                decimals: None,
            },
            "never be drawn",
        ),
        (
            2,
            Format::DateTime {
                pattern: "y".into(),
            },
            "never be drawn",
        ),
        // An empty cell has no format to change: C2 holds nothing.
        (1, Format::Number { decimals: None }, "is empty"),
    ] {
        let column = if matches!(format, Format::Number { .. }) {
            2
        } else {
            1
        };
        let error = doc
            .set_format("Formate", [(row, column)], &format)
            .expect_err(&format!("r{row} was not refused"))
            .to_string();
        assert!(error.contains(expected), "r{row}: {error:?}");
    }
    assert!(
        doc.changed_streams().is_empty(),
        "a refused format changed {:?}",
        doc.changed_streams()
    );
}

/// Sizes: a float in the header bucket, `0` meaning the table's default. Every
/// other row and column stays exactly as it was.
#[test]
fn a_column_width_and_a_row_height_are_written() {
    let mut doc = fixture!("numbers-formats.numbers");
    doc.set_column_width("Formate", 0, Some(210.0)).unwrap();
    doc.set_row_height("Formate", 3, Some(33.0)).unwrap();

    let table = doc.table("Formate").unwrap();
    assert_eq!(table.column_extents[0].size, Some(210.0));
    assert_eq!(table.column_extents[1].size, None, "column B moved");
    assert_eq!(table.row_extents[3].size, Some(33.0));
    assert_eq!(table.row_extents[2].size, None, "row 3 moved");

    // `None` is the table's default, which is a literal zero and reads back as
    // "no size of its own".
    doc.set_column_width("Formate", 0, None).unwrap();
    assert_eq!(doc.table("Formate").unwrap().column_extents[0].size, None);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    // Out of the table, and a size that is not one.
    assert!(doc.set_column_width("Formate", 9, Some(10.0)).is_err());
    assert!(doc.set_row_height("Formate", 99, Some(10.0)).is_err());
    assert!(doc.set_column_width("Formate", 0, Some(-1.0)).is_err());
    assert!(doc.set_column_width("Formate", 0, Some(f32::NAN)).is_err());
}

/// The app is the oracle for both, and for the format it is the only one that
/// can be: whether a format is *drawn* is not visible in the file.
///
/// Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_reads_back_the_formats_and_the_sizes() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = fixture!("numbers-formats.numbers");
    doc.set_format("Formate", [(1, 1)], &Format::Percent { decimals: Some(1) })
        .unwrap();
    doc.set_format("Formate", [(5, 1)], &Format::Number { decimals: Some(3) })
        .unwrap();
    doc.set_format(
        "Formate",
        [(3, 1)],
        &Format::Currency {
            code: "EUR".into(),
            decimals: Some(2),
        },
    )
    .unwrap();
    doc.set_format(
        "Formate",
        [(14, 1)],
        &Format::DateTime {
            pattern: "y-MM-dd".into(),
        },
    )
    .unwrap();
    doc.set_column_width("Formate", 0, Some(210.0)).unwrap();
    doc.set_row_height("Formate", 1, Some(33.0)).unwrap();

    let out = std::env::temp_dir().join("iwork-formats.numbers");
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
    // What the app *draws*, which is the whole point of a format.
    for (cell, drawn, named) in [
        ("B2", "123450.0%", "percent"),
        // The app puts a *non-breaking* space between the symbol and the
        // amount, which is the kind of detail only the app can settle.
        ("B4", "€\u{a0}19.99", "currency"),
        ("B6", "12345.678", "number"),
        ("B15", "2024-03-01", "date and time"),
    ] {
        assert!(
            text.lines().any(|line| {
                let field: Vec<&str> = line.split('\t').collect();
                field.first() == Some(&"cell")
                    && field.get(1) == Some(&cell)
                    && field.get(4) == Some(&drawn)
                    && field.get(5) == Some(&named)
            }),
            "the app did not draw {cell} as {drawn} ({named}):\n{text}"
        );
    }
    // And the sizes, which the app reports as it stores them.
    assert!(text.contains("column\t1\t210.0"), "the column width");
    assert!(text.contains("row\t2\t33.0"), "the row height");
    let _ = std::fs::remove_file(&out);
}

// -- money -------------------------------------------------------------------

/// A currency cell is a **value type**, not a format, and this is the record
/// the app writes for one — measured by having Numbers convert a number cell
/// this crate wrote through its own inspector.
#[test]
fn money_is_written_the_way_the_app_writes_it() {
    use iwork::table::{CellValue, Decimal};

    let mut doc = Document::new_spreadsheet("Blatt", "T", 4, 2).unwrap();
    let mut t = doc.table_mut("T").unwrap();
    t.set("A1", 19.99).unwrap();
    t.currency("A2", 184_300.0, "CHF").unwrap();
    t.currency("A3", 1_234.5, "EUR").unwrap();
    t.currency("A4", 99.0, "CHF").unwrap();

    let table = t.read();
    // The value reads back as money, not as a number that looks like money.
    assert_eq!(
        table.value(1, 0),
        CellValue::Currency(Decimal::parse("184300").unwrap())
    );
    assert_eq!(table.cell(1, 0).unwrap().format.as_str(), "currency");
    // …and the record is the app's: type 10, and `extras` 0x0802 — byte 6's
    // currency bit and byte 7's 0x08, whose meaning is not established here.
    let record = &table.cell(1, 0).unwrap().record;
    assert_eq!(record.cell_type, 10);
    assert_eq!(record.extras, 0x0802);
    assert!(record.decimal.is_some(), "the amount is a decimal128");
    assert!(record.currency_format_id.is_some());
    assert_eq!(record.number_format_id, None, "the number slot is not used");

    // One entry per currency, and a second CHF cell reuses the first's.
    let chf = table.cell(1, 0).unwrap().record.currency_format_id;
    let eur = table.cell(2, 0).unwrap().record.currency_format_id;
    let again = table.cell(3, 0).unwrap().record.currency_format_id;
    assert_ne!(chf, eur, "two currencies, one entry");
    assert_eq!(chf, again, "CHF was interned twice");

    // The plain number beside them is untouched.
    assert_eq!(table.cell(0, 0).unwrap().record.cell_type, 2);
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Writing a number over money takes the money away — the currency slot's key,
/// byte 6's claim and byte 7 with it.
#[test]
fn a_number_written_over_money_stops_being_money() {
    use iwork::table::CellValue;

    let mut doc = Document::new_spreadsheet("Blatt", "T", 2, 2).unwrap();
    let mut t = doc.table_mut("T").unwrap();
    // A plain number beside it, because a number written into a table whose
    // only format is a currency has nothing to borrow — the donor rule, not
    // this write.
    t.set("A2", 1).unwrap();
    t.currency("A1", 19.99, "CHF").unwrap();
    assert_eq!(t.read().cell(0, 0).unwrap().record.extras, 0x0802);

    t.set("A1", 42).unwrap();
    let table = t.read();
    let record = &table.cell(0, 0).unwrap().record;
    assert_eq!(record.cell_type, 2, "still a currency cell");
    assert_eq!(record.extras & 0x0800, 0, "byte 7 kept its currency mark");
    assert_eq!(table.value(0, 0), CellValue::from(42));
    assert!(table.audit().is_empty(), "{:?}", table.audit());
}

/// What money will not be written from.
#[test]
fn money_refuses_what_is_not_an_amount() {
    let mut doc = Document::new_spreadsheet("Blatt", "T", 2, 2).unwrap();
    let mut t = doc.table_mut("T").unwrap();
    assert!(t.currency("A1", "zwanzig", "CHF").is_err(), "not an amount");
    assert!(t.currency("A1", 1, "€").is_err(), "not a currency code");
    assert!(t.currency("A1", 1, "").is_err(), "no currency code");
    assert!(t.currency("Z9", 1, "CHF").is_err(), "not in the table");
    assert!(doc.changed_streams().is_empty());
}

/// The app is the oracle: it draws what this crate called money as money, in
/// two currencies at once. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_draws_written_money_as_money() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = Document::new_spreadsheet("Blatt", "T", 3, 2).unwrap();
    let mut t = doc.table_mut("T").unwrap();
    t.set("A1", 19.99).unwrap();
    t.currency("A2", 184_300.0, "CHF").unwrap();
    t.currency("A3", 1_234.5, "EUR").unwrap();

    let out = std::env::temp_dir().join("iwork-money.numbers");
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
    for (cell, drawn, named) in [
        ("A1", "19.99", "automatic"),
        // The app's own separator between symbol and amount is a
        // *non-breaking* space, as `a_column_width_and_a_row_height` found.
        ("A2", "CHF\u{a0}184300.00", "currency"),
        ("A3", "€\u{a0}1234.50", "currency"),
    ] {
        assert!(
            text.lines().any(|line| {
                let field: Vec<&str> = line.split('\t').collect();
                field.first() == Some(&"cell")
                    && field.get(1) == Some(&cell)
                    && field.get(4) == Some(&drawn)
                    && field.get(5) == Some(&named)
            }),
            "the app did not draw {cell} as {drawn} ({named}):\n{text}"
        );
    }
    let _ = std::fs::remove_file(&out);
}
