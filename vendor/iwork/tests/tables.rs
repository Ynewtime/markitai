//! Tables, read — and checked against the app that wrote them.
//!
//! Two halves. The first reads the generated corpus and asserts what is in it;
//! it needs no app and runs everywhere. The second, behind `IWORK_APP_CHECK=1`,
//! asks Numbers itself what every cell of every table holds and compares the
//! two answers cell by cell. **The app is the oracle**: a value this crate
//! decodes and Numbers does not agree with is a bug here, and the whole point
//! of decoding a format nobody documents is to have something to be wrong
//! against.
//!
//! Without `tests/fixtures/generated`, both halves pass having asserted
//! nothing and say so — `scripts/make-fixtures.sh` builds it.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use iwork::pb::{decode_nested, Message, Value};
use iwork::table::{CellFormat, CellValue, Merge, Table};
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

fn table(doc: &Document, name: &str) -> Table {
    doc.table(name)
        .unwrap_or_else(|| panic!("no table called {name}"))
}

macro_rules! fixture {
    ($name:expr) => {
        match open($name) {
            Some(doc) => doc,
            None => {
                eprintln!("no {} — skipping (run scripts/make-fixtures.sh)", $name);
                return;
            }
        }
    };
}

// -- the object graph --------------------------------------------------------

#[test]
fn tables_are_found_with_their_sheets_and_sizes() {
    let doc = fixture!("numbers-values.numbers");
    let tables = doc.tables();
    let named: Vec<(&str, usize, usize, Option<&str>)> = tables
        .iter()
        .map(|t| (t.name.as_str(), t.rows, t.columns, t.sheet.as_deref()))
        .collect();
    assert_eq!(
        named,
        vec![
            ("Zellarten", 10, 5, Some("Werte")),
            ("Zweite Tabelle", 4, 3, Some("Werte")),
            ("Kennzahlen", 10, 5, Some("Zweites Blatt")),
        ]
    );
    for t in &tables {
        assert!(t.problems.is_empty(), "{}: {:?}", t.name, t.problems);
        assert!(!t.table_id.is_empty(), "{} has no table_id", t.name);
        assert!(t.parent.is_some(), "{} has no parent drawable", t.name);
    }
}

/// Header counts, freeze flags and the stored row/column geometry.
#[test]
fn structure_state_is_read_from_the_model() {
    let doc = fixture!("numbers-values.numbers");
    let t = table(&doc, "Zellarten");
    assert_eq!(t.header_rows, 1);
    assert_eq!(t.header_columns, 1);
    assert_eq!(t.footer_rows, 0);
    assert!(t.header_rows_frozen);
    assert!(t.header_columns_frozen);
    assert_eq!(t.hidden_rows, 0);
    assert_eq!(t.hidden_columns, 0);
    assert_eq!(t.row_extents.len(), t.rows);
    assert_eq!(t.column_extents.len(), t.columns);
    // Nothing was resized, so no row or column carries a size of its own and
    // every one of them falls back to the table's default.
    assert!(t.row_extents.iter().all(|e| e.size.is_none()));
    assert!(t.column_extents.iter().all(|e| e.size.is_none()));
    assert_eq!(t.column_width(0), t.default_column_width);
    assert_eq!(t.row_height(0), t.default_row_height);
    assert!(t.row_extents.iter().all(|e| !e.hidden()));
    // Header entries do carry the per-row and per-column cell counts, which is
    // how many of them are non-empty.
    assert_eq!(t.row_extents[0].cell_count, 3, "A1:C1 are filled");
    assert_eq!(t.column_extents[0].cell_count, 8);
}

/// A row height and a column width set by hand, against the many that are not.
#[test]
fn explicit_row_and_column_sizes_are_told_from_the_default() {
    let doc = fixture!("numbers-formats.numbers");
    let t = table(&doc, "Spaltenformat");
    assert_eq!(
        t.column_extents[1].size,
        Some(150.0),
        "column B was widened"
    );
    assert_eq!(t.column_width(1), 150.0);
    assert_eq!(t.row_extents[2].size, Some(40.0), "row 3 was made taller");
    assert_eq!(t.row_height(2), 40.0);
    // Everything else still says nothing and means "the table's default".
    assert_eq!(t.column_extents[0].size, None);
    assert_eq!(t.column_width(0), t.default_column_width);
}

/// A table past one tile's 256 rows, which is the whole reason the fixture is
/// 301 rows long.
#[test]
fn a_table_larger_than_one_tile_reads_every_row() {
    let doc = fixture!("numbers-large.numbers");
    let t = table(&doc, "Zeilen");
    assert_eq!((t.rows, t.columns), (301, 9));
    assert!(t.problems.is_empty(), "{:?}", t.problems);
    assert_eq!(t.value(0, 0), CellValue::Text("Region".into()));
    // Row 300 is in the second tile; the first holds rows 0..255.
    assert_eq!(t.value(300, 7), CellValue::Text("Zeile 300".into()));
    assert_eq!(t.value(300, 2).to_text(), "900");
    let filled = t.cells().iter().filter(|c| !c.value.is_empty()).count();
    assert!(filled > 2000, "only {filled} cells decoded");
}

/// Every record in the corpus is consumed exactly, with nothing left over.
///
/// This is the check that the flag word's twenty-one bits are the right widths
/// in the right order. Get one of them wrong and the cursor ends somewhere
/// else: too early on the last cell of a row, too late on all the others. It
/// says nothing about what the fields *mean* — the oracle does that — but a
/// decoder that disagreed about the layout could not add up.
#[test]
fn every_cell_record_is_consumed_to_the_byte() {
    let mut records = 0usize;
    for name in [
        "numbers-values.numbers",
        "numbers-formats.numbers",
        "numbers-large.numbers",
        "pages-report.pages",
    ] {
        let Some(path) = generated(name) else {
            continue;
        };
        let doc = Document::open(&path).unwrap();
        for t in doc.tables() {
            assert!(t.problems.is_empty(), "{name}/{}: {:?}", t.name, t.problems);
            for cell in t.cells() {
                assert_eq!(
                    cell.record.trailing(),
                    0,
                    "{name}/{}/r{}c{}: {} bytes past the last field",
                    t.name,
                    cell.row,
                    cell.column,
                    cell.record.trailing()
                );
                assert_eq!(cell.record.version, 5);
                assert_eq!(
                    cell.record.reserved, [0; 4],
                    "{name}/{}/r{}c{}: bytes 2-5 are not zero",
                    t.name, cell.row, cell.column
                );
                records += 1;
            }
        }
    }
    if records == 0 {
        eprintln!("no fixtures — skipping (run scripts/make-fixtures.sh)");
        return;
    }
    assert!(records > 2000, "only {records} records seen");
    eprintln!("{records} cell records decoded with nothing left over");
}

// -- cell values -------------------------------------------------------------

#[test]
fn every_cell_type_decodes() {
    let doc = fixture!("numbers-values.numbers");
    let t = table(&doc, "Zellarten");
    assert_eq!(t.value(0, 0), CellValue::Text("Art".into()));
    assert_eq!(
        t.value(1, 1),
        CellValue::Text("Zeichenkette mit Umlaut: Größe".into())
    );
    assert_eq!(t.value(2, 1).to_text(), "42");
    assert_eq!(t.value(2, 2).to_text(), "84", "the formula's cached value");
    assert_eq!(t.value(3, 2), CellValue::Bool(false));
    assert_eq!(t.value(4, 1).to_text(), "2024-03-01T10:30:00Z");
    assert_eq!(t.value(5, 1), CellValue::Duration(5400.0));
    assert_eq!(t.value(6, 1).to_text(), "3.14159");
    assert_eq!(t.value(6, 2).to_text(), "3.14");
    assert_eq!(t.value(7, 2).to_text(), "45.14159");
    assert!(t.cell(2, 2).unwrap().has_formula);
    assert!(!t.cell(2, 1).unwrap().has_formula);
}

/// Numbers keeps cell numbers as a decimal, not a `double`. `3.14159` must come
/// back as those digits and not as `3.1415899999999999`.
#[test]
fn numbers_keep_their_decimal_digits() {
    let doc = fixture!("numbers-values.numbers");
    let t = table(&doc, "Zellarten");
    let CellValue::Number(pi) = t.value(6, 1) else {
        panic!("B7 is not a number");
    };
    assert_eq!(pi.to_string(), "3.14159");
    // Spelled as a parse rather than a literal: the point is that the decimal
    // and the double agree, not that this happens to be a well-known constant.
    assert_eq!(pi.to_f64(), "3.14159".parse::<f64>().unwrap());
    // Numbers normalises to fifteen significant digits rather than storing the
    // shortest form, so the trailing zeroes are real and the printing has to
    // drop them.
    assert_eq!(pi.mantissa, 314_159_000_000_000);
    assert_eq!(pi.exponent, -14);
}

/// Pages tables put their text in `TSWP.StorageArchive`s reached through the
/// rich-text list, not in the table's own string table.
#[test]
fn a_pages_table_reads_its_rich_text_cells() {
    let doc = fixture!("pages-report.pages");
    let tables = doc.tables();
    assert_eq!(tables.len(), 1, "the template document has one table");
    let t = &tables[0];
    assert_eq!((t.rows, t.columns), (6, 4));
    assert!(t.sheet.is_none(), "Pages has no sheets");
    assert_eq!(t.value(0, 0), CellValue::RichText("Description".into()));
    assert_eq!(t.value(0, 3), CellValue::RichText("Cost".into()));
    assert_eq!(t.value(1, 0), CellValue::Text("Süßwaren".into()));
    assert!(matches!(t.value(1, 2), CellValue::Currency(_)));
}

// -- formats, controls and merges --------------------------------------------

#[test]
fn each_data_format_is_read_alongside_its_value() {
    let doc = fixture!("numbers-formats.numbers");
    let t = table(&doc, "Formate");
    let by_label: BTreeMap<String, CellFormat> = (1..t.rows)
        .filter_map(|row| match t.value(row, 0) {
            CellValue::Text(label) => Some((label, t.cell(row, 1)?.format)),
            _ => None,
        })
        .collect();
    for (label, expected) in [
        ("Automatisch", CellFormat::Automatic),
        ("Zahl", CellFormat::Number),
        ("Währung", CellFormat::Currency),
        ("Prozent", CellFormat::Percentage),
        ("Wissenschaftlich", CellFormat::Scientific),
        ("Bruch", CellFormat::Fraction),
        ("Text", CellFormat::Text),
        ("Zahlensystem", CellFormat::NumeralSystem),
        ("Ankreuzfeld", CellFormat::Checkbox),
        ("Bewertung", CellFormat::Rating),
        ("Schieberegler", CellFormat::Slider),
        ("Schrittwert", CellFormat::Stepper),
        ("Einblendmenü", CellFormat::PopUpMenu),
        ("Datum formatiert", CellFormat::DateTime),
        ("Dauer formatiert", CellFormat::Duration),
        // The value is a date; nobody asked for a format, so there is none.
        ("Datum", CellFormat::Automatic),
    ] {
        assert_eq!(by_label.get(label), Some(&expected), "row labelled {label}");
    }
}

/// A control is a second thing on top of the format: a slider cell carries the
/// plain number format and a `TST.CellSpecArchive` saying it is a slider.
#[test]
fn control_cells_are_identified() {
    use iwork::table::CellControl;
    let doc = fixture!("numbers-formats.numbers");
    let t = table(&doc, "Formate");
    let controls: Vec<Option<CellControl>> = (1..t.rows)
        .filter_map(|row| t.cell(row, 1).map(|c| c.control))
        .collect();
    for wanted in [
        CellControl::Checkbox,
        CellControl::Rating,
        CellControl::Slider,
        CellControl::Stepper,
        CellControl::PopUpMenu,
    ] {
        assert!(
            controls.contains(&Some(wanted)),
            "no {} cell found",
            wanted.as_str()
        );
    }
}

#[test]
fn merged_ranges_are_read_from_the_merge_owner() {
    let doc = fixture!("numbers-formats.numbers");
    let t = table(&doc, "Verbunden");
    assert_eq!(
        t.merges,
        vec![
            // B2:D2, B4:B6, D4:F5, B8:C8 — across, down, a rectangle, and one
            // whose top-left cell was never given a value.
            Merge {
                row: 1,
                column: 1,
                rows: 1,
                columns: 3
            },
            Merge {
                row: 3,
                column: 1,
                rows: 3,
                columns: 1
            },
            Merge {
                row: 3,
                column: 3,
                rows: 2,
                columns: 3
            },
            Merge {
                row: 7,
                column: 1,
                rows: 1,
                columns: 2
            },
        ]
    );
    // The covered cells hold nothing at all — not even a `spanCellType`
    // record. Their offsets are the plain −1 of an empty cell.
    assert!(t.value(1, 2).is_empty());
    assert!(t.value(1, 3).is_empty());
    assert_eq!(t.value(1, 1), CellValue::Text("quer".into()));
}

#[test]
fn a_simple_merge_is_found_in_the_values_fixture() {
    let doc = fixture!("numbers-values.numbers");
    let t = table(&doc, "Zellarten");
    assert_eq!(
        t.merges,
        vec![Merge {
            row: 8,
            column: 3,
            rows: 1,
            columns: 2
        }],
        "D9:E9"
    );
}

// -- how a table is organised ------------------------------------------------
//
// The four fixtures here are documents Apple wrote and Numbers 15.3.1 saved
// again — nothing in Numbers' scripting interface can sort, filter, categorise,
// highlight or pivot a table, so a script cannot build them. See
// `scripts/applescript/from-template.applescript`.
//
// The oracle for these is inside the documents. A pivot table's *cells* are the
// app's own rendering of the pivot rules beside them, so the decoded fields can
// be checked against the labels Numbers itself drew; a category's summary row
// holds the number the aggregate produced.

#[test]
fn a_category_names_its_column_its_groups_and_its_summary() {
    let doc = fixture!("numbers-categories.numbers");
    let tables = doc.tables();
    let t = tables
        .iter()
        .find(|t| !t.categories.is_empty())
        .expect("the categories fixture has a category");
    let category = &t.categories[0];

    assert!(category.enabled, "the category is switched on");
    assert_eq!(
        category
            .columns
            .iter()
            .map(|c| c.column)
            .collect::<Vec<_>>(),
        vec![Some(1)],
        "grouped by column B — the one holding the names"
    );

    let groups = category.groups();
    let values: Vec<String> = groups
        .iter()
        .map(|g| g.value.as_ref().map(|v| v.to_text()).unwrap_or_default())
        .collect();
    assert_eq!(values, vec!["Andy", "Chloe"]);
    // The values in column B, read back through the cells: the group's rows are
    // exactly the rows holding its value, which is what a category *is*.
    for group in &groups {
        let wanted = group.value.as_ref().unwrap().to_text();
        assert!(!group.rows.is_empty(), "{wanted}: no rows");
        for &row in &group.rows {
            assert_eq!(
                t.value(row, 1).to_text(),
                wanted,
                "row {row} is in group {wanted}"
            );
        }
    }

    assert_eq!(category.summaries.len(), 1);
    let summary = &category.summaries[0];
    assert_eq!(summary.column, Some(4), "the Amount column is summarised");
    assert_eq!(summary.function, iwork::table::Aggregate::Sum);
    assert_eq!(summary.level, 1);
}

/// A pivot's fields name columns of the table it summarises, not of itself.
///
/// The check is against the pivot table's own cells, which Numbers rendered
/// from these rules: the header reads `Power`/`Product` down the side,
/// `Date (Month)` across the top and `Units (Sum)` for the values, and each of
/// those is a column of `Sales` at the index the rules resolve to.
#[test]
fn a_pivot_resolves_its_fields_against_its_source_table() {
    let doc = fixture!("numbers-pivot.numbers");
    let tables = doc.tables();
    let source = tables
        .iter()
        .find(|t| t.name == "Sales" && t.pivot.is_none())
        .expect("a source table");
    let pivot_table = tables
        .iter()
        .find(|t| t.pivot.as_ref().is_some_and(|p| !p.empty))
        .expect("a pivot with rules");
    let pivot = pivot_table.pivot.as_ref().unwrap();

    assert_eq!(pivot.source_name, "Sales");
    let heading = |column: Option<usize>| source.value(0, column.unwrap()).to_text();

    assert_eq!(
        pivot
            .rows
            .iter()
            .map(|f| heading(f.column))
            .collect::<Vec<_>>(),
        vec!["Power", "Product"]
    );
    assert_eq!(
        pivot
            .columns
            .iter()
            .map(|f| heading(f.column))
            .collect::<Vec<_>>(),
        vec!["Date"]
    );
    assert!(
        pivot.columns[0].has_functor,
        "the date field is bucketed — the app renders it as Date (Month)"
    );
    assert_eq!(
        pivot
            .values
            .iter()
            .map(|v| (heading(v.column), v.function))
            .collect::<Vec<_>>(),
        vec![("Units".to_string(), iwork::table::Aggregate::Sum)]
    );
    assert!(!pivot.hide_grand_total_rows && !pivot.hide_grand_total_columns);

    // The same document carries a second pivot with nothing assigned. It is
    // still a pivot table, and saying so is the difference between "no rules
    // decoded" and "no rules there".
    let empty = tables
        .iter()
        .filter_map(|t| t.pivot.as_ref())
        .find(|p| p.empty)
        .expect("the practice sheet's empty pivot");
    assert!(empty.rows.is_empty() && empty.columns.is_empty() && empty.values.is_empty());
}

/// `hidingState` finally has both of its values in one document.
///
/// Phase 1 could only report the number, because nothing in the scripting
/// interface hides a row. Here three columns were hidden by hand and a filter
/// rule hides rows, in the same spreadsheet.
#[test]
fn hidden_rows_and_columns_say_who_hid_them() {
    use iwork::table::Hiding;
    let doc = fixture!("numbers-rules.numbers");
    let tables = doc.tables();

    let by_user: Vec<(usize, Hiding)> = tables
        .iter()
        .flat_map(|t| t.column_extents.iter().enumerate())
        .filter(|(_, e)| e.hidden())
        .map(|(i, e)| (i, e.hiding()))
        .collect();
    assert_eq!(
        by_user,
        vec![(6, Hiding::User), (11, Hiding::User), (12, Hiding::User)],
        "the portfolio table hides three columns by hand"
    );

    let filtered: Vec<Hiding> = tables
        .iter()
        .flat_map(|t| t.row_extents.iter())
        .filter(|e| e.hidden())
        .map(|e| e.hiding())
        .collect();
    assert!(!filtered.is_empty(), "the filter hides rows");
    assert!(
        filtered.iter().all(|&h| h == Hiding::Filter),
        "every hidden row is hidden by the filter, not by hand: {filtered:?}"
    );

    // The extent agrees about the columns exactly. It does *not* agree about
    // the rows — it marks every body row, not just the hidden ones — which is
    // recorded in FORMAT.md rather than asserted away.
    let columns = tables
        .iter()
        .find(|t| !t.column_states.user_hidden.is_empty())
        .expect("a table with hidden columns");
    assert_eq!(columns.column_states.user_hidden, vec![6, 11, 12]);
    assert!(columns.column_states.filtered.is_empty());
}

#[test]
fn a_filter_set_carries_its_rules_and_its_switch() {
    let doc = fixture!("numbers-rules.numbers");
    let tables = doc.tables();
    let filter = tables
        .iter()
        .find_map(|t| t.filter.as_ref())
        .expect("the filtered table");

    assert!(filter.enabled, "filters are on");
    assert!(!filter.match_any, "the set matches All, not Any");
    assert_eq!(filter.rules.len(), 1);
    let rule = &filter.rules[0];
    assert!(rule.enabled);
    assert_eq!(rule.column, Some(0), "filter_offsets puts the rule on A");
    assert!(
        rule.predicate.pre_pivot,
        "15.3.1 writes filter rules in the pre-pivot slot, whatever the \
         published references call current"
    );
    assert!(
        rule.predicate.has_formula,
        "the rule tests a formula, so its value is not an immediate"
    );

    // Every other table in the corpus has an empty filter set and must report
    // no filter rather than an empty one.
    for name in ["numbers-values.numbers", "numbers-formats.numbers"] {
        let Some(other) = open(name) else { continue };
        for t in other.tables() {
            assert!(t.filter.is_none(), "{name}/{}: unexpected filter", t.name);
        }
    }
}

#[test]
fn conditional_highlighting_rules_carry_the_value_they_compare_against() {
    let doc = fixture!("numbers-rules.numbers");
    let tables = doc.tables();
    let sets: Vec<&iwork::table::ConditionalStyles> = tables
        .iter()
        .flat_map(|t| t.conditional_styles.iter())
        .collect();
    assert_eq!(sets.len(), 2, "two rule sets");

    let mut seen: Vec<(i64, String)> = Vec::new();
    for set in &sets {
        assert!(set.key.is_some(), "a set is reached by a cell's key");
        for rule in &set.rules {
            assert!(
                rule.cell_style.is_some() && rule.text_style.is_some(),
                "a highlighting rule applies both styles"
            );
            assert!(
                !rule.predicate.pre_pivot,
                "the current slot wins: only it carries the compared value"
            );
            seen.push((
                rule.predicate.kind,
                rule.predicate
                    .values
                    .first()
                    .map(|v| v.to_text())
                    .unwrap_or_default(),
            ));
        }
    }
    seen.sort();
    assert_eq!(
        seen,
        vec![
            (7, "0".to_string()),
            (9, "0".to_string()),
            (36, "↑".to_string()),
            (36, "↓".to_string()),
        ],
        "two numeric rules against zero and two text rules against arrows"
    );
}

#[test]
fn a_custom_cell_format_is_named_and_document_scoped() {
    let doc = fixture!("numbers-rules.numbers");
    let formats = doc.custom_formats();
    assert_eq!(formats.len(), 1);
    assert_eq!(formats[0].name, "Millions");
    assert_eq!(formats[0].format_string, "#,###.##M");

    // Documents without one report none, rather than reporting an empty list
    // as a format.
    for name in ["numbers-values.numbers", "numbers-categories.numbers"] {
        let Some(other) = open(name) else { continue };
        assert!(other.custom_formats().is_empty(), "{name}");
    }
}

#[test]
fn a_sort_rule_names_a_column_and_a_direction() {
    let doc = fixture!("numbers-sorted.numbers");
    let tables = doc.tables();
    let sorted = tables
        .iter()
        .find(|t| !t.sort_rules.is_empty())
        .expect("the sorted fixture has a sort rule");
    assert_eq!(
        sorted.sort_rules,
        vec![iwork::table::SortRule {
            column: 2,
            descending: false
        }]
    );
    // An empty `TableSortOrderArchive` is written on every table in every
    // document; it must not read as a rule.
    for name in ["numbers-values.numbers", "numbers-formats.numbers"] {
        let Some(other) = open(name) else { continue };
        for t in other.tables() {
            assert!(t.sort_rules.is_empty(), "{name}/{}", t.name);
        }
    }
}

/// The UUID index is what everything above is addressed through, and it is
/// **sorted by UUID**, not by index. A map that read it positionally would
/// still map most columns correctly, so the check is that it maps *all* of
/// them and that a round trip through it is the identity.
#[test]
fn the_uid_map_covers_every_row_and_column() {
    for name in [
        "numbers-values.numbers",
        "numbers-categories.numbers",
        "numbers-pivot.numbers",
    ] {
        let Some(doc) = open(name) else { continue };
        for t in doc.tables() {
            if t.uids.is_empty() {
                continue;
            }
            let columns: Vec<usize> = (0..t.columns)
                .filter(|&c| t.column_uid(c).is_some())
                .collect();
            assert_eq!(
                columns,
                (0..t.columns).collect::<Vec<_>>(),
                "{name}/{}: every column has a UUID",
                t.name
            );
            for column in 0..t.columns {
                let uid = t.column_uid(column).unwrap();
                assert_eq!(
                    t.uids.column(uid),
                    Some(column),
                    "{name}/{}: column {column} round-trips",
                    t.name
                );
            }
        }
    }
}

/// Nothing in this phase writes to a table, and a save must prove it — on the
/// documents that carry the features, by name, so that the guarantee is
/// asserted about *them* and not merely about whatever happens to be in the
/// fixture directory.
#[test]
fn saving_a_document_organised_this_way_changes_no_byte_of_it() {
    for name in [
        "numbers-categories.numbers",
        "numbers-pivot.numbers",
        "numbers-rules.numbers",
        "numbers-sorted.numbers",
    ] {
        let Some(path) = generated(name) else {
            continue;
        };
        let doc = Document::open(&path).unwrap();
        // Reading everything this phase decodes, first: a decoder that mutates
        // what it reads would be caught here and nowhere else.
        let tables = doc.tables();
        let _ = doc.custom_formats();
        assert!(!tables.is_empty(), "{name}: no tables");

        let before = iwork::Package::read(&path).unwrap();
        assert!(
            doc.changed_streams().is_empty(),
            "{name}: reading changed a stream"
        );
        let out = std::env::temp_dir().join(format!("iwork-organise-{name}"));
        doc.save(&out).unwrap();
        let after = iwork::Package::read(&out).unwrap();
        assert_eq!(
            before.names().collect::<Vec<_>>(),
            after.names().collect::<Vec<_>>(),
            "{name}: entry names changed"
        );
        for entry in before.names() {
            assert_eq!(
                before.get(entry),
                after.get(entry),
                "{name}: {entry} was rewritten"
            );
        }
        let _ = std::fs::remove_file(&out);
    }
}

// -- hostile input -----------------------------------------------------------
//
// A document is untrusted bytes. These splice the shapes a real file could not
// hold — a tile id past what its base row can address, a category tree that
// references itself, a merge extent that reaches the top of `usize` — into a
// genuine fixture's streams, and assert the reader *returns*: a bounded error
// or a truncated read, never a panic or an abort.

/// Rewrite the first length-delimited field `number` whose decoded message the
/// closure accepts, re-encoding it in place. Returns whether one was changed;
/// the closure returning `false` moves on to the next field of that number, so
/// this walks past the rules that do not match to the one that does.
fn edit_first(msg: &mut Message, number: u32, mut f: impl FnMut(&mut Message) -> bool) -> bool {
    for field in msg.fields.iter_mut() {
        if field.number != number {
            continue;
        }
        let Value::Bytes(raw) = &field.value else {
            continue;
        };
        let Some(mut child) = decode_nested(raw) else {
            continue;
        };
        if f(&mut child) {
            field.value = Value::Bytes(child.encode());
            return true;
        }
    }
    false
}

/// Splice a fixture: for every object, hand `mutate` its id, message type and
/// decoded first message; re-encode any it changes and rebuild the `Document`.
/// `None` when the corpus is absent; panics if nothing matched, so a test can
/// never quietly assert against an unspliced document.
fn spliced(name: &str, mut mutate: impl FnMut(u64, u32, &mut Message) -> bool) -> Option<Document> {
    let path = generated(name)?;
    let mut package = iwork::Package::read(&path).unwrap();
    let mut matched = false;
    for stream in package.iwa_names() {
        let mut objects = iwork::iwa::parse(package.get(&stream).unwrap()).unwrap();
        let mut touched = false;
        for object in objects.iter_mut() {
            let mt = object.message_type();
            let Some(message) = object.messages.first_mut() else {
                continue;
            };
            let Ok(mut decoded) = Message::decode(&message.payload) else {
                continue;
            };
            if mutate(object.identifier, mt, &mut decoded) {
                message.payload = decoded.encode();
                touched = true;
                matched = true;
            }
        }
        if touched {
            package.set(&stream, iwork::iwa::serialize(&objects));
        }
    }
    assert!(matched, "{name}: the splice matched nothing");
    Some(Document::from_package(package).unwrap())
}

/// A `TSP.Reference` message, `{1: id}`, as a length-delimited value.
fn reference_to(id: u64) -> Value {
    let mut r = Message::default();
    r.set(1, Value::Varint(id));
    Value::Bytes(r.encode())
}

/// A tile id of 2⁶⁰ overflows `tile_id × tile_size`. The reader must skip that
/// tile, not wrap it to a plausible base row or panic multiplying it.
#[test]
fn a_tile_id_that_overflows_its_base_row_is_skipped_not_wrapped() {
    let Some(doc) = spliced("numbers-large.numbers", |_, mt, model| {
        if mt != iwork::table::TYPE_TABLE_MODEL {
            return false;
        }
        // model → field 4 DataStore → field 3 TileStorage → first tile → id.
        edit_first(model, 4, |store| {
            edit_first(store, 3, |tiles| {
                edit_first(tiles, 1, |tile| {
                    tile.set(1, Value::Varint(1u64 << 60));
                    true
                })
            })
        })
    }) else {
        return;
    };
    // The call returns; the overflowing tile is simply absent from the read.
    let tables = doc.tables();
    assert!(!tables.is_empty(), "the table survived the hostile tile id");
}

/// A category node whose child list points back at the root is a cycle. The
/// reader must break it, not recurse until the stack is gone.
#[test]
fn a_self_referential_category_tree_is_read_without_recursing_forever() {
    // The inline tree cannot cycle, so first find the by-reference root the
    // `GroupByArchive` names in field 18, then force the reader down that path
    // (drop the inline field 3) and give the root a child edge to itself.
    let path = match generated("numbers-categories.numbers") {
        Some(p) => p,
        None => return,
    };
    let root = {
        let pkg = iwork::Package::read(&path).unwrap();
        let mut found = 0u64;
        for stream in pkg.iwa_names() {
            for o in iwork::iwa::parse(pkg.get(&stream).unwrap()).unwrap() {
                if o.message_type() == 6373 {
                    if let Some(id) = Message::decode(o.payload()).ok().and_then(|m| {
                        m.bytes(18)
                            .and_then(decode_nested)
                            .and_then(|r| r.varint(1))
                    }) {
                        found = id;
                    }
                }
            }
        }
        found
    };
    assert!(
        root != 0,
        "no by-reference category root to point at itself"
    );

    let doc = spliced("numbers-categories.numbers", |id, mt, m| {
        if mt == 6373 {
            // Drop the inline root so the by-reference tree is what is read.
            let before = m.fields.len();
            m.fields.retain(|f| f.number != 3);
            return m.fields.len() != before;
        }
        if id == root {
            // The root gains a child edge to itself: root → root → …
            m.append_in_order(10, reference_to(root));
            return true;
        }
        false
    })
    .expect("the corpus is present");

    // If this returns at all, the cycle was broken; categories still read.
    let tables = doc.tables();
    assert!(!tables.is_empty());
}

/// A merge whose extents reach the top of `usize` — the shape a reversed or
/// wrapped colon tract used to yield — must not overflow the range arithmetic
/// that decides which cell a merge covers.
#[test]
fn a_hostile_merge_extent_does_not_overflow_merge_covering() {
    let doc = fixture!("numbers-values.numbers");
    let mut t = table(&doc, "Zellarten");
    t.merges = vec![Merge {
        row: 0,
        column: 0,
        rows: usize::MAX,
        columns: usize::MAX,
    }];
    // The lookup returns rather than panicking on `row + rows`.
    assert_eq!(t.merge_covering(5, 5), Some(t.merges[0]));
    assert_eq!(
        t.merge_covering(usize::MAX - 1, usize::MAX - 1),
        Some(t.merges[0])
    );
}

/// A predicate's immediate number keeps its fraction. Spliced 2.5 into a
/// highlighting rule that compared against 0, it reads back as 2.5, not 2.
#[test]
fn a_fractional_predicate_threshold_reads_back_whole() {
    let Some(doc) = spliced("numbers-rules.numbers", |_, mt, obj| {
        if mt != 6010 {
            return false;
        }
        // 6010 → field 3 wrapper → a rule (field 1) → predicate (field 1) →
        // one of its args (4/5/6) → data (field 2) → the immediate (field 1).
        edit_first(obj, 3, |wrapper| {
            edit_first(wrapper, 1, |rule| {
                edit_first(rule, 1, |pred| {
                    [4u32, 5, 6].into_iter().any(|arg| {
                        edit_first(pred, arg, |arg| {
                            edit_first(arg, 2, |data| {
                                if matches!(data.get(1), Some(Value::Fixed64(_))) {
                                    data.set(1, Value::Fixed64(2.5f64.to_le_bytes()));
                                    true
                                } else {
                                    false
                                }
                            })
                        })
                    })
                })
            })
        })
    }) else {
        return;
    };

    let threshold = doc
        .tables()
        .iter()
        .flat_map(|t| t.conditional_styles.clone())
        .flat_map(|s| s.rules)
        .find_map(|r| {
            r.predicate
                .values
                .into_iter()
                .find(|v| v.to_text() == "2.5")
        });
    assert_eq!(
        threshold,
        Some(CellValue::Number(
            iwork::table::Decimal::parse("2.5").unwrap()
        )),
        "the fractional threshold was truncated"
    );
}

// -- the oracle --------------------------------------------------------------

/// One cell as Numbers reports it.
#[derive(Debug)]
struct OracleCell {
    name: String,
    class: String,
    value: String,
    formatted: String,
    format: String,
    formula: String,
}

struct OracleTable {
    name: String,
    rows: usize,
    columns: usize,
    header_rows: u32,
    header_columns: u32,
    footer_rows: u32,
    cells: Vec<OracleCell>,
}

fn read_oracle(document: &Path) -> Vec<OracleTable> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(document)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "{}: Numbers would not report on {}:\n{}",
        script.display(),
        document.display(),
        String::from_utf8_lossy(&output.stderr)
    );

    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    let mut tables: Vec<OracleTable> = Vec::new();
    for line in text.lines() {
        let field: Vec<&str> = line.split('\t').collect();
        match field.first() {
            Some(&"table") if field.len() >= 7 => tables.push(OracleTable {
                name: field[1].to_string(),
                rows: field[2].parse().unwrap(),
                columns: field[3].parse().unwrap(),
                header_rows: field[4].parse().unwrap(),
                header_columns: field[5].parse().unwrap(),
                footer_rows: field[6].parse().unwrap(),
                cells: Vec::new(),
            }),
            Some(&"cell") if field.len() >= 7 => {
                let table = tables.last_mut().expect("a cell before any table");
                table.cells.push(OracleCell {
                    name: field[1].to_string(),
                    class: field[2].to_string(),
                    value: field[3].to_string(),
                    formatted: field[4].to_string(),
                    format: field[5].to_string(),
                    formula: field[6].to_string(),
                });
            }
            _ => {}
        }
    }
    tables
}

/// `A1` for a position, the way the app names cells.
fn reference_name(row: usize, column: usize) -> String {
    let mut letters = String::new();
    let mut n = column + 1;
    while n > 0 {
        letters.insert(0, (b'A' + ((n - 1) % 26) as u8) as char);
        n = (n - 1) / 26;
    }
    format!("{letters}{}", row + 1)
}

/// Does this crate's value agree with what the app said?
///
/// Some of the app's answers are locale- and timezone-dependent and cannot be
/// compared literally:
///
/// * a `real` comes back as AppleScript renders it — `255.0`, `1.2345678E+4` —
///   so numbers are compared as numbers, not as strings;
/// * a `date` comes back in the machine's timezone and its own long form,
///   while the stored seconds are timezone-naive. The *formatted* value is what
///   the app draws in the cell and what the stored number means, so dates are
///   compared against that;
/// * a duration is a `real` of seconds, which is exactly what is stored.
fn agrees(ours: &CellValue, oracle: &OracleCell) -> bool {
    match ours {
        CellValue::Empty | CellValue::Span => oracle.class == "empty",
        CellValue::Text(text) | CellValue::RichText(text) => text == &oracle.value,
        CellValue::Bool(b) => oracle.value.eq_ignore_ascii_case(&b.to_string()),
        CellValue::Number(d) | CellValue::Currency(d) => {
            near(d.to_f64(), oracle.value.parse::<f64>().ok())
        }
        CellValue::Duration(seconds) => near(*seconds, oracle.value.parse::<f64>().ok()),
        CellValue::Date(seconds) => {
            // "01.03.2024 10:30" or "01.03.2024" — compare the parts the
            // stored seconds actually determine.
            let iso = iwork::table::format_date(*seconds);
            let (date, time) = iso.split_at(10);
            let day = &date[8..10];
            let month = &date[5..7];
            let year = &date[..4];
            oracle.formatted.contains(&format!("{day}.{month}.{year}"))
                && (oracle.formatted.len() <= 10 || oracle.formatted.contains(&time[1..6]))
        }
        CellValue::Error => oracle.class == "empty" || !oracle.formula.is_empty(),
        CellValue::Unknown(_) => false,
    }
}

fn near(ours: f64, theirs: Option<f64>) -> bool {
    theirs.is_some_and(|t| (ours - t).abs() <= 1e-9 * ours.abs().max(1.0))
}

/// Every cell of every table of both Numbers fixtures, against Numbers.
///
/// Off unless `IWORK_APP_CHECK=1`: it drives the app. What it compares is the
/// value, the data format and whether the cell holds a formula — and, through
/// the names the app gives merged cells, the merged ranges too. A merged-away
/// cell reports the name of the cell it was merged into, which is the only way
/// the scripting interface admits a merge exists.
#[test]
fn every_cell_agrees_with_numbers() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the oracle comparison");
        return;
    }
    let mut compared = 0usize;
    for name in [
        "numbers-values.numbers",
        "numbers-formats.numbers",
        "numbers-large.numbers",
    ] {
        let Some(path) = generated(name) else {
            eprintln!("no {name} — skipping");
            continue;
        };
        let doc = Document::open(&path).unwrap();
        let ours = doc.tables();
        let theirs = read_oracle(&path);
        assert_eq!(
            ours.len(),
            theirs.len(),
            "{name}: we found {} tables, Numbers {}",
            ours.len(),
            theirs.len()
        );

        for (mine, app) in ours.iter().zip(&theirs) {
            assert_eq!(mine.name, app.name, "{name}: table names differ");
            assert_eq!(
                (mine.rows, mine.columns),
                (app.rows, app.columns),
                "{name}/{}",
                app.name
            );
            assert_eq!(mine.header_rows, app.header_rows, "{name}/{}", app.name);
            assert_eq!(
                mine.header_columns, app.header_columns,
                "{name}/{}",
                app.name
            );
            assert_eq!(mine.footer_rows, app.footer_rows, "{name}/{}", app.name);
            assert_eq!(
                app.cells.len(),
                app.rows * app.columns,
                "{name}/{}: the app reported a different number of cells",
                app.name
            );

            for (index, cell) in app.cells.iter().enumerate() {
                let row = index / app.columns;
                let column = index % app.columns;
                let where_ = format!("{name}/{}/{}", app.name, reference_name(row, column));

                // The app does not report a merged-away cell at all: it
                // repeats the cell the merge began in, name, value and
                // format. So the names it gives back *are* the merge map, and
                // the value to compare is the anchor's.
                let (row, column) = mine
                    .merges
                    .iter()
                    .find(|m| {
                        (m.row..m.row + m.rows).contains(&row)
                            && (m.column..m.column + m.columns).contains(&column)
                    })
                    .map(|m| (m.row, m.column))
                    .unwrap_or((row, column));
                assert_eq!(
                    cell.name,
                    reference_name(row, column),
                    "{where_}: the app calls this cell {}",
                    cell.name
                );
                let value = mine.value(row, column);

                assert!(
                    agrees(&value, cell),
                    "{where_}: we say {value:?}, Numbers says class {} value {:?} formatted {:?}",
                    cell.class,
                    cell.value,
                    cell.formatted
                );
                if let Some(decoded) = mine.cell(row, column) {
                    assert_eq!(
                        decoded.format.as_str(),
                        cell.format,
                        "{where_}: data format"
                    );
                    assert_eq!(
                        decoded.has_formula,
                        !cell.formula.is_empty(),
                        "{where_}: the app {} a formula",
                        if cell.formula.is_empty() {
                            "reports no"
                        } else {
                            "reports"
                        }
                    );
                } else {
                    assert_eq!(cell.format, "automatic", "{where_}: no record but a format");
                }
                compared += 1;
            }
        }
    }
    eprintln!("{compared} cells compared against Numbers");
    assert!(compared > 0, "nothing was compared");
}

// -- merging cells -----------------------------------------------------------

/// The measurement this write rests on: the node array this crate builds for a
/// range is **byte for byte the one the app wrote** for the same merge. All
/// four of `numbers-formats.numbers`'s merges are reproduced from nothing but
/// their row, column and size.
#[test]
fn a_merge_is_written_the_way_the_app_wrote_it() {
    fixture!("numbers-formats.numbers");
    let doc = open("numbers-formats.numbers").unwrap();
    let table = table(&doc, "Verbunden");
    assert_eq!(table.merges.len(), 4);

    // What the app has, as bytes: the node array of every merge formula.
    let model = doc.archive(table.model).unwrap();
    let owner = model
        .bytes(47)
        .and_then(decode_nested)
        .expect("merge owner");
    let store = owner
        .bytes(2)
        .and_then(decode_nested)
        .expect("formula store");
    let theirs: Vec<Vec<u8>> = store
        .all(3)
        .filter_map(|value| match value {
            Value::Bytes(raw) => decode_nested(raw),
            _ => None,
        })
        .filter_map(|pair| pair.bytes(2).and_then(decode_nested))
        .filter_map(|formula| formula.bytes(1).map(<[u8]>::to_vec))
        .collect();
    assert_eq!(theirs.len(), 4);

    for merge in &table.merges {
        let mine = iwork::document::merge_range_node_for_test(
            &table,
            merge.row,
            merge.column,
            merge.rows,
            merge.columns,
        )
        .unwrap();
        assert!(
            theirs.contains(&mine),
            "the range node for r{}c{} {}×{} is not one the app wrote",
            merge.row,
            merge.column,
            merge.rows,
            merge.columns
        );
    }
}

/// Merging: the range appears, the covered cells are emptied through the
/// ordinary writer so their references go back, and every count still adds up.
#[test]
fn cells_can_be_merged_and_unmerged() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.set_cell("Spaltenformat", 0, 1, CellValue::Text("weg".into()))
        .unwrap();
    assert!(table(&doc, "Spaltenformat").merges.is_empty());

    doc.merge_cells("Spaltenformat", 0, 0, 1, 2).unwrap();

    let merged = table(&doc, "Spaltenformat");
    assert_eq!(
        merged.merges,
        vec![Merge {
            row: 0,
            column: 0,
            rows: 1,
            columns: 2
        }]
    );
    assert_eq!(
        merged.value(0, 0),
        CellValue::Text("Betrag".into()),
        "the merge lost the cell it began in"
    );
    assert_eq!(
        merged.value(0, 1),
        CellValue::Empty,
        "the covered cell kept its value"
    );
    assert!(merged.audit().is_empty(), "{:?}", merged.audit());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    // …and apart again. The cells stay empty, which is what the merge left.
    doc.unmerge_cells("Spaltenformat", 0, 0).unwrap();
    let apart = table(&doc, "Spaltenformat");
    assert!(apart.merges.is_empty(), "{:?}", apart.merges);
    assert_eq!(apart.value(0, 0), CellValue::Text("Betrag".into()));
    assert!(apart.audit().is_empty(), "{:?}", apart.audit());
}

/// What a merge will not do.
#[test]
fn merging_refuses_what_it_cannot_do_honestly() {
    fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    for (name, row, column, rows, columns, expected) in [
        ("Spaltenformat", 0, 0, 1, 1, "one cell is not a merge"),
        ("Spaltenformat", 0, 0, 9, 1, "the table is 4×2"),
        ("Verbunden", 1, 1, 2, 2, "overlaps the merge"),
        ("Verbunden", 0, 0, 2, 2, "overlaps the merge"),
    ] {
        let error = doc
            .merge_cells(name, row, column, rows, columns)
            .expect_err(&format!("{name} r{row}c{column} was not refused"))
            .to_string();
        assert!(error.contains(expected), "{name}: {error:?}");
    }
    // A covered cell holding a formula: the cell writer refuses it by name.
    if let Some(mut doc) = open("numbers-values.numbers") {
        let error = doc
            .merge_cells("Zellarten", 2, 1, 1, 2)
            .expect_err("C3 holds a formula")
            .to_string();
        assert!(error.contains("formula"), "{error}");
        assert!(doc.changed_streams().is_empty());
    }
    assert!(
        doc.changed_streams().is_empty(),
        "a refused merge changed {:?}",
        doc.changed_streams()
    );
    // Unmerging where there is no merge.
    assert!(doc.unmerge_cells("Spaltenformat", 3, 1).is_err());
}

/// The app is the oracle, and it has no merge property to ask about: a
/// merged-away cell is reported under the *name and value of the cell the merge
/// began in*, so `A1 A1` across the first row is a 1×2 merge at A1.
///
/// Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_reports_a_written_merge() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let _ = fixture!("numbers-formats.numbers");
    let mut doc = open("numbers-formats.numbers").unwrap();
    doc.merge_cells("Spaltenformat", 0, 0, 1, 2).unwrap();
    let out = std::env::temp_dir().join("iwork-merge.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/table-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(&out)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document with a written merge:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let text = String::from_utf8_lossy(&output.stdout).into_owned();
    // The first row of Spaltenformat: two cells, both reported as A1.
    let mut in_table = false;
    let first_row: Vec<&str> = text
        .lines()
        .filter(|line| {
            let field: Vec<&str> = line.split('\t').collect();
            if field.first() == Some(&"table") {
                in_table = field.get(1) == Some(&"Spaltenformat");
            }
            in_table && field.first() == Some(&"cell")
        })
        .take(2)
        .collect();
    assert_eq!(first_row.len(), 2);
    for line in &first_row {
        let field: Vec<&str> = line.split('\t').collect();
        assert_eq!(
            field.get(1),
            Some(&"A1"),
            "the app does not see the two cells as one: {line}"
        );
    }
    let _ = std::fs::remove_file(&out);
}
