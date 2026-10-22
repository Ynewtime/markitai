//! Tables, written — the byte level, and then the app.
//!
//! Phase 1 proved this crate can *read* a cell record: every one in the
//! corpus ends exactly on its last field. Writing needs a stronger claim, and
//! it is the one this file makes first: **the encoder is the decoder's exact
//! inverse**, on every record the three apps have written here, not merely on
//! the ones this crate has fields for. A record carrying a conditional-style
//! key, a comment key or a byte 7 nobody has decoded has to come back out as it
//! went in, because the alternative is a cell that quietly loses its
//! highlighting.
//!
//! Then the operation itself: which streams a `set-cell` rewrites (as few as
//! possible, and byte-identical everywhere else), what a written cell keeps,
//! and what the writer refuses to do rather than guess at. Finally, behind
//! `IWORK_APP_CHECK=1`, Numbers is asked to read the result back.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use iwork::pb::{decode_nested, Message, Value};
use iwork::table::{self, CellRecord, CellValue, Decimal};
use iwork::Document;

const FIXTURES: &[&str] = &[
    "numbers-values.numbers",
    "numbers-formats.numbers",
    "numbers-large.numbers",
    "numbers-categories.numbers",
    "numbers-pivot.numbers",
    "numbers-rules.numbers",
    "numbers-sorted.numbers",
    "pages-report.pages",
];

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

/// Every cell record in a document, exactly as it lies in its tile.
///
/// Walked here rather than taken from `Table::cells`, because the point is the
/// *bytes*: a decoded cell has already lost whatever the decoder did not keep,
/// and comparing it to itself would prove nothing.
fn raw_records(doc: &Document) -> Vec<(u64, usize, usize, Vec<u8>)> {
    let mut out = Vec::new();
    for (_, object) in doc.objects() {
        if object.message_type() != table::TYPE_TILE {
            continue;
        }
        let Ok(tile) = Message::decode(object.payload()) else {
            continue;
        };
        let tile_wide = tile.varint(8).unwrap_or(0) != 0;
        for value in tile.all(5) {
            let Value::Bytes(raw) = value else { continue };
            let Some(info) = decode_nested(raw) else {
                continue;
            };
            let row = info.varint(1).unwrap_or(0) as usize;
            let (Some(buffer), Some(offsets)) = (info.bytes(6), info.bytes(7)) else {
                continue;
            };
            let wide = tile_wide || info.varint(8).unwrap_or(0) != 0;
            for (column, bytes) in table::row_cells(buffer, offsets, wide)
                .into_iter()
                .flatten()
            {
                out.push((object.identifier, row, column, bytes.to_vec()));
            }
        }
    }
    out
}

// -- the encoder -------------------------------------------------------------

/// The claim the whole write path rests on: encode is decode's inverse, byte
/// for byte, on records this crate did not write and does not fully model.
#[test]
fn every_cell_record_re_encodes_to_the_bytes_it_came_from() {
    let mut records = 0usize;
    let mut conditional = 0usize;
    let mut controls = 0usize;
    for name in FIXTURES {
        let Some(doc) = open(name) else { continue };
        for (tile, row, column, bytes) in raw_records(&doc) {
            let record = table::decode_cell(&bytes)
                .unwrap_or_else(|e| panic!("{name} tile {tile} r{row}c{column}: {e}"));
            let again = record
                .encode()
                .unwrap_or_else(|e| panic!("{name} tile {tile} r{row}c{column}: {e}"));
            assert_eq!(
                again, bytes,
                "{name} tile {tile} r{row}c{column} did not re-encode to itself"
            );
            // The flag word is derived from the fields on the way out, so a
            // record that came in with one has to derive the same one.
            assert_eq!(record.derived_flags(), record.flags);
            if record.conditional_style_id.is_some() || record.conditional_rule_id.is_some() {
                conditional += 1;
            }
            if record.control_id.is_some() {
                controls += 1;
            }
            records += 1;
        }
    }
    if records == 0 {
        eprintln!("no fixtures — skipping (run scripts/make-fixtures.sh)");
        return;
    }
    assert!(records > 2000, "only {records} records seen");
    assert!(
        conditional > 0,
        "no conditionally highlighted cell in the corpus, so nothing proved \
         that a rewritten cell keeps its highlighting"
    );
    assert!(controls > 0, "no control cell in the corpus");
    eprintln!(
        "{records} records re-encoded byte for byte \
         ({conditional} carrying conditional keys, {controls} a control)"
    );
}

/// What a tile says about itself, which a rewritten tile has to keep saying.
///
/// Phase 1 recorded that 15.3.1 writes neither `storage_version` nor
/// `last_saved_in_BNC` on a tile. That is wrong, and looking at the tile rather
/// than at the `TileStorage` above it is what shows the difference: **every
/// tile in the corpus carries field 6 = 5 and field 7 = true**, and the
/// published "refuse a tile that is not `last_saved_in_BNC`" rule would accept
/// all of them. The dead field is `numCells`, which is 0 on a tile holding 2411
/// cells.
#[test]
fn every_tile_says_it_was_last_saved_by_the_current_storage_engine() {
    let mut tiles = 0usize;
    let mut wide = 0usize;
    for name in FIXTURES {
        let Some(doc) = open(name) else { continue };
        for (_, object) in doc.objects() {
            if object.message_type() != table::TYPE_TILE {
                continue;
            }
            let tile = Message::decode(object.payload()).unwrap();
            let at = format!("{name} tile {}", object.identifier);
            assert_eq!(tile.varint(6), Some(5), "{at}: storage_version");
            assert_eq!(tile.varint(7), Some(1), "{at}: last_saved_in_BNC");
            assert_eq!(tile.varint(3), Some(0), "{at}: numCells is a dead field");
            assert_eq!(
                tile.varint(4).unwrap_or(0) as usize,
                tile.all(5).count(),
                "{at}: numrows counts the TileRowInfos"
            );
            if tile.varint(8).unwrap_or(0) != 0 {
                wide += 1;
            }
            tiles += 1;
        }
    }
    if tiles == 0 {
        eprintln!("no fixtures — skipping (run scripts/make-fixtures.sh)");
        return;
    }
    assert!(
        wide > 0,
        "no tile in the corpus uses wide offsets, so the encoder's ×4 scaling \
         is untested against a real document"
    );
    eprintln!("{tiles} tiles, {wide} of them with wide offsets");
}

/// A record with every field set at once, including the ones this crate reads
/// and never writes, plus bytes it does not decode at all.
#[test]
fn a_record_this_crate_does_not_understand_still_round_trips() {
    let record = CellRecord {
        version: 5,
        cell_type: 2,
        // Zero in every real record; kept anyway, because "zero everywhere so
        // far" is not "zero always".
        reserved: [0xde, 0xad, 0xbe, 0xef],
        // Byte 6 says the user chose currency; byte 7 is undecoded and appears
        // as 0x08 on currency cells.
        extras: 0x0802,
        flags: 0,
        decimal: Some(Decimal {
            mantissa: -12_345_678_901_234_567,
            exponent: -7,
        }),
        double: Some(1.5),
        seconds: Some(729_772_200.0),
        string_id: Some(11),
        rich_id: Some(12),
        cell_style_id: Some(13),
        text_style_id: Some(14),
        conditional_style_id: Some(15),
        conditional_rule_id: Some(16),
        formula_id: Some(17),
        control_id: Some(18),
        formula_error_id: Some(19),
        format_kind: Some(2),
        number_format_id: Some(20),
        currency_format_id: Some(21),
        date_format_id: Some(22),
        duration_format_id: Some(23),
        text_format_id: Some(24),
        boolean_format_id: Some(25),
        comment_id: Some(26),
        import_warning_id: Some(27),
        tail: vec![0x01, 0x02, 0x03],
    };
    let bytes = record.encode().unwrap();
    // Twelve header bytes, one 16-byte decimal, two doubles, eighteen keys,
    // three bytes nobody claimed.
    assert_eq!(bytes.len(), 12 + 16 + 8 + 8 + 18 * 4 + 3);
    let back = table::decode_cell(&bytes).unwrap();
    assert_eq!(back.reserved, record.reserved);
    assert_eq!(back.extras, record.extras);
    assert_eq!(back.tail, record.tail);
    assert_eq!(back.conditional_style_id, Some(15));
    assert_eq!(back.decimal, record.decimal);
    assert_eq!(back.encode().unwrap(), bytes);
}

#[test]
fn the_encoder_refuses_a_flag_it_cannot_place_a_payload_for() {
    let record = CellRecord {
        version: 5,
        flags: 0x8000_0000,
        ..CellRecord::default()
    };
    assert!(record.encode().is_err());
}

/// A number goes in as digits and comes out as the same digits — through the
/// 16-byte decimal, not through an `f64`, which is the reason the format has
/// one.
#[test]
fn a_number_survives_the_decimal_it_is_stored_in() {
    for text in ["1.10", "-0.5", "3.14159", "42", "0", "-7", "1000000", "1e3"] {
        let decimal = Decimal::parse(text).unwrap_or_else(|| panic!("{text} did not parse"));
        let bytes = table::encode_decimal128(decimal).unwrap();
        assert_eq!(
            table::decode_decimal128(&bytes),
            decimal,
            "{text}: mantissa and exponent did not survive"
        );
        assert_eq!(decimal.to_f64(), text.parse::<f64>().unwrap(), "{text}");
    }
    // The stored form keeps the trailing zero; printing drops it, because
    // Numbers writes fifteen significant digits for everything it touches and
    // a reader that printed those would spell every number wrong.
    let ten = Decimal::parse("1.10").unwrap();
    assert_eq!((ten.mantissa, ten.exponent), (110, -2));
    assert_eq!(ten.to_string(), "1.1");

    assert!(Decimal::parse("").is_none());
    assert!(Decimal::parse("twelve").is_none());
    assert!(Decimal::parse("1.2.3").is_none());

    // Hostile `iwork set-cell n:…` input: an exponent at the bottom of `i32`
    // less a fraction length used to underflow. It must be refused, not wrapped
    // or panicked on.
    assert!(
        Decimal::parse("1.5e-2147483648").is_none(),
        "an exponent that underflows must be refused"
    );
}

/// A `double` immediate keeps its fraction, rather than being cast to its
/// integer part — the difference between a filter threshold of `2.5` reading
/// back as `2.5` and as `2`.
#[test]
fn a_double_becomes_a_decimal_that_keeps_its_fraction() {
    for (value, text) in [(2.5, "2.5"), (-0.9, "-0.9"), (0.0, "0"), (42.0, "42")] {
        assert_eq!(Decimal::from_f64(value), Decimal::parse(text).unwrap());
    }
    assert_eq!(Decimal::from_f64(2.5).to_f64(), 2.5);
}

#[test]
fn a_date_survives_the_text_form_it_is_written_in() {
    for text in [
        "2001-01-01T00:00:00Z",
        "2024-03-01T10:30:00Z",
        "1970-07-04T23:59:59Z",
        "2025-12-24T08:15:00Z",
    ] {
        let seconds = table::parse_date(text).unwrap_or_else(|| panic!("{text} did not parse"));
        assert_eq!(table::format_date(seconds), text);
    }
    assert_eq!(table::parse_date("2001-01-01"), Some(0.0));
    assert!(table::parse_date("2024-13-01").is_none());
    assert!(table::parse_date("tomorrow").is_none());

    // A year wide enough to overflow the day arithmetic is nonsense, not a date;
    // refused rather than multiplied into a panic or a wrap.
    assert!(table::parse_date("25000000000000000-01-01").is_none());
    assert!(table::parse_date("10000-01-01").is_none());
}

// -- the edit ----------------------------------------------------------------

fn edited(name: &str, table: &str, row: usize, column: usize, value: CellValue) -> Document {
    let mut doc = open(name).expect("caller checked the fixture is there");
    doc.set_cell(table, row, column, value)
        .unwrap_or_else(|e| panic!("{name}: {e}"));
    doc
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

/// Editing one number touches one stream out of ninety-seven.
///
/// The number matters as much as the name: a writer that re-encodes everything
/// it read produces a document where no diff can tell the edit from the noise.
#[test]
fn editing_a_cell_rewrites_only_the_streams_it_has_to() {
    fixture!("numbers-values.numbers");

    // A number over a number: the tile, and nothing else. No string is
    // interned, no format slot changes, no cell appears or disappears.
    let doc = edited(
        "numbers-values.numbers",
        "Zellarten",
        2,
        1,
        CellValue::Number(Decimal::parse("43").unwrap()),
    );
    assert_eq!(doc.changed_streams(), vec!["Index/Tables/Tile.iwa"]);

    // Text over text: the tile and the string list, because one entry is taken
    // and another given up.
    let doc = edited(
        "numbers-values.numbers",
        "Zellarten",
        2,
        0,
        CellValue::Text("Ganzzahl".into()),
    );
    assert_eq!(
        doc.changed_streams(),
        vec!["Index/Tables/DataList-904541.iwa", "Index/Tables/Tile.iwa"],
    );

    // Filling a cell that had no record: the two header buckets join in,
    // because a row and a column each hold a count of their stored cells.
    let doc = edited(
        "numbers-values.numbers",
        "Zellarten",
        1,
        3,
        CellValue::Text("neu".into()),
    );
    let changed = doc.changed_streams();
    assert!(
        changed
            .iter()
            .any(|s| s.contains("HeaderStorageBucket-904914")),
        "the row bucket was not touched: {changed:?}"
    );
    assert!(
        changed
            .iter()
            .any(|s| s.contains("HeaderStorageBucket-904958")),
        "the column bucket was not touched: {changed:?}"
    );
    assert_eq!(changed.len(), 5, "{changed:?}");
}

/// Writing what is already there writes nothing at all.
///
/// The sharpest test of the encoder there is: it reproduces the app's own bytes
/// for a value the app wrote, through the string table and the format keys and
/// the offsets, or the entry changes and this fails.
#[test]
fn writing_a_cell_the_value_it_already_holds_changes_no_byte() {
    fixture!("numbers-values.numbers");
    for (row, column, value) in [
        (2, 1, CellValue::Number(Decimal::parse("42").unwrap())),
        (2, 0, CellValue::Text("Zahl".into())),
        (
            4,
            1,
            CellValue::Date(table::parse_date("2024-03-01T10:30:00Z").unwrap()),
        ),
        (5, 1, CellValue::Duration(5400.0)),
    ] {
        let doc = edited("numbers-values.numbers", "Zellarten", row, column, value);
        assert!(
            doc.changed_streams().is_empty(),
            "r{row}c{column}: {:?}",
            doc.changed_streams()
        );
    }
}

/// A written cell keeps everything about it that was not the value.
#[test]
fn a_written_cell_keeps_what_the_writer_did_not_change() {
    fixture!("pages-report.pages");
    let before = open("pages-report.pages").unwrap();
    let old = before
        .table("Details")
        .unwrap()
        .cell(1, 0)
        .unwrap()
        .record
        .clone();
    // The Pages template gave this cell an explicit text format — byte 6 is
    // 0x80 — which is exactly the kind of thing a re-synthesised record loses.
    assert_eq!(old.extras & 0x80, 0x80);

    let doc = edited(
        "pages-report.pages",
        "Details",
        1,
        0,
        CellValue::Text("Konfekt".into()),
    );
    let new = doc
        .table("Details")
        .unwrap()
        .cell(1, 0)
        .unwrap()
        .record
        .clone();
    assert_eq!(new.extras, old.extras, "byte 6 and byte 7");
    assert_eq!(new.reserved, old.reserved);
    assert_eq!(new.cell_style_id, old.cell_style_id);
    assert_eq!(new.text_style_id, old.text_style_id);
    assert_eq!(new.text_format_id, old.text_format_id, "the format stayed");
    assert_eq!(new.format_kind, old.format_kind);
    assert_ne!(new.string_id, old.string_id, "the text did change");
    assert_eq!(
        doc.table("Details").unwrap().value(1, 0),
        CellValue::Text("Konfekt".into())
    );
}

/// Value and format travel together: a cell that changes type is given the
/// format key another cell of the new type already uses, and gives up its old
/// slot's key — which is what Numbers 15.3.1 does.
#[test]
fn a_cell_that_changes_type_changes_its_format_slot_with_it() {
    fixture!("numbers-values.numbers");
    let before = open("numbers-values.numbers").unwrap();
    let donor = before
        .table("Zellarten")
        .unwrap()
        .cell(2, 1)
        .unwrap()
        .record
        .number_format_id
        .expect("B3 is a number and carries a number format");

    let doc = edited(
        "numbers-values.numbers",
        "Zellarten",
        5,
        0,
        CellValue::Number(Decimal::parse("5").unwrap()),
    );
    let record = doc
        .table("Zellarten")
        .unwrap()
        .cell(5, 0)
        .unwrap()
        .record
        .clone();
    assert_eq!(record.cell_type, 2);
    assert_eq!(record.number_format_id, Some(donor));
    assert_eq!(record.text_format_id, None, "the text slot was given up");
    assert_eq!(record.format_kind, Some(1), "the number slot is current");
    assert!(record.string_id.is_none());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The interned string list is refcounted, and an entry nobody points at goes.
#[test]
fn the_string_table_is_kept_by_reference_count() {
    fixture!("numbers-values.numbers");
    let strings = |doc: &Document| -> BTreeMap<u32, (String, u32)> {
        doc.table("Zellarten")
            .unwrap()
            .side_tables()
            .strings
            .entries
            .values()
            .map(|e| (e.key, (e.string.clone().unwrap_or_default(), e.refcount)))
            .collect()
    };
    let before = strings(&open("numbers-values.numbers").unwrap());

    // "Zahl" was A3's and nobody else's, so replacing it takes the entry away
    // and puts the new text at the list's next key.
    let doc = edited(
        "numbers-values.numbers",
        "Zellarten",
        2,
        0,
        CellValue::Text("Ganzzahl".into()),
    );
    let after = strings(&doc);
    assert!(
        !after.values().any(|(text, _)| text == "Zahl"),
        "the released string is still there: {after:?}"
    );
    assert_eq!(
        after
            .values()
            .filter(|(text, _)| text == "Ganzzahl")
            .map(|(_, count)| *count)
            .collect::<Vec<_>>(),
        vec![1]
    );
    assert_eq!(after.len(), before.len());

    // Writing a string another cell already holds takes a second reference to
    // the entry it is in rather than adding one beside it.
    let doc = edited(
        "numbers-values.numbers",
        "Zellarten",
        2,
        0,
        CellValue::Text("Datum".into()),
    );
    let after = strings(&doc);
    assert_eq!(
        after.len(),
        before.len() - 1,
        "one entry fewer, not one more"
    );
    assert_eq!(
        after
            .values()
            .filter(|(text, _)| text == "Datum")
            .map(|(_, count)| *count)
            .collect::<Vec<_>>(),
        vec![2]
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Emptying a cell removes its record outright, which is what the app does.
#[test]
fn emptying_a_cell_takes_its_record_away() {
    fixture!("numbers-values.numbers");
    let doc = edited(
        "numbers-values.numbers",
        "Zellarten",
        7,
        0,
        CellValue::Empty,
    );
    let table = doc.table("Zellarten").unwrap();
    assert!(table.cell(7, 0).is_none(), "the record is still there");
    assert_eq!(table.row_extents[7].cell_count, 1, "the row's count fell");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Everything the writer will not do, refused by name.
#[test]
fn set_cell_refuses_what_it_cannot_do_honestly() {
    fixture!("numbers-values.numbers");
    let one = || CellValue::Number(Decimal::parse("1").unwrap());
    let mut doc = open("numbers-values.numbers").unwrap();
    for (row, column, value, expected) in [
        (2, 2, one(), "formula"),
        (8, 4, one(), "merge"),
        (2, 9, one(), "10×5"),
        (0, 0, CellValue::Error, "does not write"),
    ] {
        let error = doc
            .set_cell("Zellarten", row, column, value)
            .expect_err(&format!("r{row}c{column} was not refused"))
            .to_string();
        assert!(
            error.contains(expected),
            "r{row}c{column}: {error:?} does not mention {expected:?}"
        );
    }
    assert!(
        doc.changed_streams().is_empty(),
        "a refused edit changed something"
    );

    let mut pages = open("pages-report.pages").unwrap();
    let error = pages
        .set_cell("Details", 0, 0, CellValue::Text("hi".into()))
        .expect_err("a rich-text cell was not refused")
        .to_string();
    assert!(error.contains("rich text"), "{error}");
}

/// A refused write leaves the document byte for byte — and reference-count for
/// reference-count — as it was.
///
/// `Spaltenformat` is an all-currency column, so turning its text header into a
/// *number* can find no number format to copy and must fail. It used to fail
/// only *after* releasing the header's string entry, leaving a record that
/// still named a string the list no longer held: a dangling key `audit`
/// complains about, in a document this crate wrote itself. Planning the whole
/// write before applying any of it is what closes that.
#[test]
fn a_refused_write_changes_no_byte_and_no_reference_count() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    let before = doc.table("Spaltenformat").unwrap().value(0, 0);
    let err = doc
        .set_cell(
            "Spaltenformat",
            0,
            0,
            CellValue::Number(Decimal::parse("9").unwrap()),
        )
        .expect_err("a number with no donor format should be refused")
        .to_string();
    assert!(err.contains("format to copy"), "{err}");
    assert!(
        doc.changed_streams().is_empty(),
        "a refused write rewrote {:?}",
        doc.changed_streams()
    );
    let t = doc.table("Spaltenformat").unwrap();
    assert_eq!(t.value(0, 0), before);
    assert_eq!(t.value(0, 0), CellValue::Text("Betrag".into()));
    assert!(
        t.audit().is_empty(),
        "audit after a refusal: {:?}",
        t.audit()
    );
}

/// Emptying a cell gives up *every* reference the deleted record held — a key
/// in each format slot it carried, and its control spec — not just the one the
/// format kind names.
///
/// The header of a currency column is a text cell holding both a text and a
/// currency format key; dropping only the current one leaves the other's
/// refcount too high, which `audit` — now watching the control list too —
/// rejects.
#[test]
fn emptying_a_cell_gives_up_every_key_it_held() {
    fixture!("numbers-formats.numbers");

    // A text header carrying a second (currency) format slot.
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.set_cell("Spaltenformat", 0, 0, CellValue::Empty)
        .unwrap();
    let t = doc.table("Spaltenformat").unwrap();
    assert_eq!(t.value(0, 0), CellValue::Empty);
    assert!(
        t.audit().is_empty(),
        "audit after emptying the header: {:?}",
        t.audit()
    );

    // A cell carrying a control spec, whose reference the audit now checks.
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.set_cell("Formate", 10, 1, CellValue::Empty).unwrap();
    let t = doc.table("Formate").unwrap();
    assert!(
        t.audit().is_empty(),
        "audit after emptying a control cell: {:?}",
        t.audit()
    );
}

/// An ambiguous table name is refused on a *write*, where a read may settle for
/// the first match. Apple's pivot sample carries two tables called `Sales`, on
/// different sheets; silently editing the first is how a caller writes into the
/// wrong one.
#[test]
fn an_ambiguous_table_name_is_refused_on_write() {
    fixture!("numbers-pivot.numbers");
    let mut doc = open("numbers-pivot.numbers").unwrap();
    let err = doc
        .set_cell("Sales", 0, 0, CellValue::Text("x".into()))
        .expect_err("two tables are called Sales")
        .to_string();
    assert!(err.contains("names 2 tables"), "{err}");
    assert!(err.contains("904729") && err.contains("905059"), "{err}");
    assert!(doc.changed_streams().is_empty());

    // By identifier there is no ambiguity — whatever the write then does, it is
    // not the refusal above.
    if let Err(e) = doc.set_cell("904729", 0, 0, CellValue::Text("x".into())) {
        assert!(!e.to_string().contains("names"), "{e}");
    }
}

/// The patched-object refusal covers every object a write rewrites, not only
/// the tile. A text edit rewrites the string list too, and a stale version
/// patch left on *that* would describe it as it used to be.
#[test]
fn a_write_over_a_patched_side_table_is_refused() {
    let Some(mut doc) = patch_every("numbers-values.numbers", iwork::table::TYPE_DATA_LIST) else {
        return;
    };
    let err = doc
        .set_cell("Zellarten", 2, 0, CellValue::Text("Ganzzahl".into()))
        .expect_err("a write over a patched string list should be refused")
        .to_string();
    assert!(err.contains("version patches"), "{err}");
    assert!(doc.changed_streams().is_empty());
}

/// Splice a `type == 0` version patch onto every object of one message type, so
/// that [`Document::patched_objects`] reports them — the hostile shape a write
/// must refuse to rewrite. Returns `None` when the corpus is absent.
fn patch_every(name: &str, message_type: u32) -> Option<Document> {
    use iwork::iwa::{self, ArchiveMessage};
    let path = generated(name)?;
    let mut package = iwork::Package::read(&path).unwrap();
    let mut matched = false;
    for stream in package.iwa_names() {
        let mut objects = iwa::parse(package.get(&stream).unwrap()).unwrap();
        let mut touched = false;
        for object in objects.iter_mut() {
            if object.message_type() != message_type {
                continue;
            }
            object.messages.push(ArchiveMessage {
                message_type: 0,
                version: Vec::new(),
                extra: Vec::new(),
                payload: Vec::new(),
            });
            touched = true;
            matched = true;
        }
        if touched {
            package.set(&stream, iwa::serialize(&objects));
        }
    }
    assert!(matched, "{name}: no object of type {message_type} to patch");
    Some(Document::from_package(package).unwrap())
}

/// Every table in the corpus already keeps the invariants `iwork check` now
/// tests for, and an edited one still does.
#[test]
fn an_edited_table_still_adds_up() {
    for name in FIXTURES {
        let Some(doc) = open(name) else { continue };
        for table in doc.tables() {
            assert!(
                table.audit().is_empty(),
                "{name}/{}: {:?}",
                table.name,
                table.audit()
            );
        }
    }

    if generated("numbers-values.numbers").is_none() {
        return;
    }
    let mut doc = open("numbers-values.numbers").unwrap();
    for (row, column, value) in [
        (2, 1, CellValue::Number(Decimal::parse("43").unwrap())),
        (2, 0, CellValue::Text("Ganzzahl".into())),
        (1, 3, CellValue::Text("neu".into())),
        (1, 4, CellValue::Number(Decimal::parse("7").unwrap())),
        (5, 0, CellValue::Number(Decimal::parse("5").unwrap())),
        (7, 0, CellValue::Empty),
        (3, 3, CellValue::Bool(true)),
    ] {
        doc.set_cell("Zellarten", row, column, value)
            .unwrap_or_else(|e| panic!("r{row}c{column}: {e}"));
    }
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    // And every stream it did not touch is still the byte-identical original.
    let out = std::env::temp_dir().join("iwork-cells-battery.numbers");
    let changed: Vec<String> = doc
        .changed_streams()
        .iter()
        .map(|s| s.to_string())
        .collect();
    doc.save(&out).unwrap();
    let before = iwork::Package::read(generated("numbers-values.numbers").unwrap()).unwrap();
    let after = iwork::Package::read(&out).unwrap();
    for entry in before.names() {
        if changed.iter().any(|c| c == entry) {
            continue;
        }
        assert_eq!(before.get(entry), after.get(entry), "{entry} was rewritten");
    }
    let _ = std::fs::remove_file(&out);
}

// -- the type == 0 diff mechanism --------------------------------------------

/// What 15.3.1 actually writes, so the claim in `FORMAT.md` has a test.
///
/// One patched object per Numbers document — the view state — and none at all
/// in Pages or Keynote. No table archive carries one, which is why editing a
/// cell never has to merge or re-emit a patch, and why the writer's rule can be
/// the blunt one: refuse.
#[test]
fn only_the_view_state_carries_version_patches() {
    let mut seen = 0usize;
    for name in [
        "numbers-values.numbers",
        "numbers-formats.numbers",
        "numbers-large.numbers",
        "numbers-categories.numbers",
        "numbers-pivot.numbers",
        "numbers-rules.numbers",
        "numbers-sorted.numbers",
        "pages-plain.pages",
        "pages-report.pages",
        "pages-styled.pages",
        "pages-unicode.pages",
        "keynote-deck.key",
    ] {
        let Some(doc) = open(name) else { continue };
        seen += 1;
        let patched = doc.patched_objects();
        if !name.ends_with(".numbers") {
            assert!(patched.is_empty(), "{name}: {patched:?}");
            continue;
        }
        assert_eq!(patched.len(), 1, "{name}: {patched:?}");
        let (identifier, patches) = patched[0];
        assert_eq!(patches, 3, "{name}: three patches, for 11.0, 10.1 and 10.0");
        let (stream, object) = doc.object(identifier).unwrap();
        assert!(stream.starts_with("Index/ViewState"), "{name}: {stream}");
        assert_eq!(object.message_type(), 12026, "{name}: TN.UIStateArchive");
        // No tile, no data list, no model. Rewriting those is the whole of
        // Phase 2, and nothing it rewrites has a patch to go stale.
        for (_, object) in doc.objects() {
            if object.messages.iter().all(|m| m.message_type != 0) {
                continue;
            }
            assert_eq!(
                object.identifier, identifier,
                "{name}: a second patched object"
            );
        }
    }
    if seen == 0 {
        eprintln!("no fixtures — skipping (run scripts/make-fixtures.sh)");
    }
}

// -- the oracle --------------------------------------------------------------

/// Ask Numbers what it thinks of a document this crate edited.
///
/// Off unless `IWORK_APP_CHECK=1`. The two halves matter equally: the edited
/// cell reads back the new value, and every cell around it reads back the old
/// one. A writer that damaged the row's offsets would pass the first half and
/// fail the second — which is how a shifted cell looks from the outside.
///
/// The third thing it proves is not about this crate at all. `C3` holds
/// `=B3×2` and its cached value is still the old `84` in the file, because
/// nothing here evaluates a formula. **Numbers recalculates on open** and
/// answers `86`, which is why leaving the cache stale is a limitation and not a
/// corruption — and why a *reader* that trusts the cache is the one at risk.
#[test]
fn numbers_reads_back_an_edited_cell() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let source = fixture!("numbers-values.numbers");

    let mut doc = Document::open(&source).unwrap();
    for (row, column, value) in [
        (2, 1, CellValue::Number(Decimal::parse("43").unwrap())),
        (2, 0, CellValue::Text("Ganzzahl".into())),
        (
            4,
            1,
            CellValue::Date(table::parse_date("2025-12-24T08:15:00Z").unwrap()),
        ),
        (5, 1, CellValue::Duration(7200.0)),
        (1, 3, CellValue::Text("neu".into())),
        (1, 4, CellValue::Number(Decimal::parse("7").unwrap())),
        (5, 0, CellValue::Number(Decimal::parse("5").unwrap())),
        (7, 0, CellValue::Empty),
        (3, 3, CellValue::Bool(true)),
    ] {
        doc.set_cell("Zellarten", row, column, value)
            .unwrap_or_else(|e| panic!("r{row}c{column}: {e}"));
    }
    let out = std::env::temp_dir().join("iwork-set-cell.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open the edited document:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();

    // Only the first table's cells; the fixture has three and they repeat the
    // A1 names.
    let mut said: BTreeMap<String, (String, String)> = BTreeMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut table = 0usize;
    for line in text.lines() {
        let field: Vec<&str> = line.split('\t').collect();
        match field.first() {
            Some(&"table") => table += 1,
            Some(&"cell") if table == 1 && field.len() >= 5 => {
                names.push(field[1].to_string());
                said.insert(
                    field[1].to_string(),
                    (field[3].to_string(), field[4].to_string()),
                );
            }
            _ => {}
        }
    }
    assert!(!said.is_empty(), "the oracle reported no cells");

    let value = |name: &str| said.get(name).map(|(v, _)| v.clone()).unwrap_or_default();
    let shown = |name: &str| said.get(name).map(|(_, f)| f.clone()).unwrap_or_default();

    // What was written.
    assert_eq!(value("B3"), "43.0", "the edited number");
    assert_eq!(value("A3"), "Ganzzahl", "the edited text");
    assert_eq!(shown("B5"), "24.12.2025 08:15", "the edited date");
    assert_eq!(
        shown("B6"),
        "2h",
        "the edited duration, drawn as a duration"
    );
    assert_eq!(value("D2"), "neu", "a cell that had no record");
    assert_eq!(value("E2"), "7.0", "a column that had no header entry");
    assert_eq!(value("A6"), "5.0", "a text cell made a number");
    assert_eq!(value("A8"), "", "an emptied cell");
    assert_eq!(value("D4"), "true", "a boolean written from nothing");

    // What was not. The formula's cached value in the file is still 84; the
    // app recalculated it.
    assert_eq!(value("C3"), "86.0", "=B3×2 was recalculated on open");
    assert_eq!(value("A1"), "Art");
    assert_eq!(value("B2"), "Zeichenkette mit Umlaut: Größe");
    assert_eq!(value("B7"), "3.14159", "a decimal nobody touched");
    assert_eq!(value("C7"), "3.14");
    // The merge is still a merge, and the only way the scripting interface
    // admits one exists: a merged-away cell is reported under the *anchor's*
    // name, so `D9:E9` comes back as two cells both called D9.
    assert_eq!(value("D9"), "verbunden", "the merge kept its value");
    assert_eq!(
        names.iter().filter(|name| *name == "D9").count(),
        2,
        "D9:E9 is no longer merged"
    );
    let _ = std::fs::remove_file(&out);
}

// -- writing many cells at once ---------------------------------------------

/// The batch and the single write are the same write: `set_cell` is one cell
/// handed to `set_cells`, so what one does the other does.
#[test]
fn a_batch_writes_what_the_same_single_writes_would() {
    fixture!("numbers-formats.numbers");
    let mut one = open("numbers-formats.numbers").unwrap();
    let mut many = open("numbers-formats.numbers").unwrap();

    let cells = [
        (1, 2, CellValue::Text("Erste".into())),
        (2, 2, CellValue::Number(Decimal::parse("42").unwrap())),
        (3, 2, CellValue::Bool(true)),
    ];
    for (row, column, value) in cells.clone() {
        one.set_cell("Formate", row, column, value).unwrap();
    }
    assert_eq!(many.set_cells("Formate", cells).unwrap(), 3);

    for row in 0..one.table("Formate").unwrap().rows {
        for column in 0..one.table("Formate").unwrap().columns {
            assert_eq!(
                one.table("Formate").unwrap().value(row, column),
                many.table("Formate").unwrap().value(row, column),
                "r{row}c{column}"
            );
        }
    }
    // And byte for byte, which is the stronger claim: the same streams, with
    // the same contents.
    let out = |doc: &Document, name: &str| {
        let path = std::env::temp_dir().join(name);
        let _ = std::fs::remove_file(&path);
        doc.save(&path).unwrap();
        let package = iwork::Package::read(&path).unwrap();
        let bytes: Vec<(String, Vec<u8>)> = package
            .names()
            .map(|n| (n.to_string(), package.get(n).unwrap().to_vec()))
            .collect();
        let _ = std::fs::remove_file(&path);
        bytes
    };
    assert_eq!(
        out(&one, "iwork-batch-one.numbers"),
        out(&many, "iwork-batch-many.numbers"),
        "the batch and the loop wrote different documents"
    );
}

/// A refused cell leaves the whole batch unwritten — including the cells before
/// it, which a loop of single writes would have applied already.
#[test]
fn a_refused_cell_rolls_the_whole_batch_back() {
    fixture!("numbers-values.numbers");
    let mut doc = open("numbers-values.numbers").unwrap();
    // The first two are writable; the third holds a formula.
    let err = doc
        .set_cells(
            "Zellarten",
            [
                (1, 0, CellValue::Text("Eins".into())),
                (1, 1, CellValue::Text("Zwei".into())),
                (2, 2, CellValue::Number(Decimal::parse("1").unwrap())),
            ],
        )
        .expect_err("the formula cell is refused")
        .to_string();
    assert!(err.contains("formula"), "{err}");
    assert!(
        doc.changed_streams().is_empty(),
        "a refused batch left {:?} rewritten",
        doc.changed_streams()
    );
}

/// Naming one cell twice is refused: which of the two values was meant is not
/// something this crate decides for the caller.
#[test]
fn a_cell_named_twice_is_refused() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    let err = doc
        .set_cells(
            "Formate",
            [
                (1, 2, CellValue::Text("a".into())),
                (1, 2, CellValue::Text("b".into())),
            ],
        )
        .expect_err("the same cell twice")
        .to_string();
    assert!(err.contains("named twice"), "{err}");
    assert!(doc.changed_streams().is_empty());
}

/// A block goes in at a corner, and a short row leaves the rest of its row
/// alone rather than emptying it.
#[test]
fn a_block_lands_at_its_corner() {
    let mut doc = Document::new_spreadsheet("Blatt", "T", 4, 4).unwrap();
    doc.set_cell("T", 3, 3, CellValue::Text("Ecke".into()))
        .unwrap();
    let written = doc
        .set_block(
            "T",
            (1, 1),
            &[
                vec![CellValue::Text("a".into()), CellValue::Text("b".into())],
                vec![CellValue::Text("c".into())],
            ],
        )
        .unwrap();
    assert_eq!(written, 3);

    let table = doc.table("T").unwrap();
    assert_eq!(table.value(1, 1), CellValue::Text("a".into()));
    assert_eq!(table.value(1, 2), CellValue::Text("b".into()));
    assert_eq!(table.value(2, 1), CellValue::Text("c".into()));
    assert_eq!(table.value(2, 2), CellValue::Empty, "a short row spills");
    assert_eq!(
        table.value(3, 3),
        CellValue::Text("Ecke".into()),
        "the block reached past itself"
    );
    assert!(table.audit().is_empty(), "{:?}", table.audit());
}

/// A batch spanning tiles, filling rows that have no storage at all, and every
/// count still adding up. This is the shape a caller with a table of data has,
/// and the one the single-cell path is too slow to serve: 100 000 cells go in
/// in a quarter of a second, where a cell at a time takes minutes.
#[test]
fn a_batch_crosses_tiles_and_fills_empty_rows() {
    let rows = 300;
    let mut doc = Document::new_spreadsheet("Blatt", "Lang", rows, 3).unwrap();
    let block: Vec<Vec<CellValue>> = (0..rows)
        .map(|row| {
            (0..3)
                .map(|column| CellValue::Text(format!("r{row}c{column}")))
                .collect()
        })
        .collect();
    assert_eq!(doc.set_block("Lang", (0, 0), &block).unwrap(), rows * 3);

    let table = doc.table("Lang").unwrap();
    for row in [0, 1, 255, 256, 257, 299] {
        for column in 0..3 {
            assert_eq!(
                table.value(row, column),
                CellValue::Text(format!("r{row}c{column}")),
                "r{row}c{column}"
            );
        }
    }
    assert!(table.audit().is_empty(), "{:?}", table.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}
