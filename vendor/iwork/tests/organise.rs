//! Writing the layer *above* the cells: how a table is organised.
//!
//! Sort rules, filters, categories and conditional highlighting were all read
//! in detail and written not at all, which is also why `insert_row` refuses
//! half the tables in the world — every one of those is an `Organised` refusal.
//! This file is what that layer's writes are measured by.
//!
//! The app cannot be *asked* about most of it: there is no scripting property
//! for a sort rule or a highlight. What the app can do is **open the document,
//! rewrite it, and save** — and what it then wrote is what it understood. That
//! is the bar used here, the same one authored comments are held to.

use std::path::{Path, PathBuf};

use iwork::table::SortRule;
use iwork::{Document, Refusal};

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

/// Sort rules go in, come back, and can be taken away again.
#[test]
fn a_table_can_be_told_what_to_sort_by() {
    let mut doc = fixture!("numbers-values.numbers");
    assert!(doc.table("Zellarten").unwrap().sort_rules.is_empty());

    doc.set_sort_rules(
        "Zellarten",
        &[
            SortRule {
                column: 1,
                descending: true,
            },
            SortRule {
                column: 0,
                descending: false,
            },
        ],
    )
    .unwrap();
    assert_eq!(
        doc.table("Zellarten").unwrap().sort_rules,
        vec![
            SortRule {
                column: 1,
                descending: true
            },
            SortRule {
                column: 0,
                descending: false
            }
        ]
    );
    // One stream — the model's — and nothing else in the document moved.
    assert_eq!(doc.changed_streams().len(), 1);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    // The rules are *what to sort by*, not the order of the rows: nothing
    // moved.
    let table = doc.table("Zellarten").unwrap();
    assert_eq!(table.value(1, 0).to_text(), "Text");

    // And a table that had one can be given none.
    let mut sorted = fixture!("numbers-sorted.numbers");
    assert_eq!(sorted.table("Reading Log").unwrap().sort_rules.len(), 1);
    sorted.set_sort_rules("Reading Log", &[]).unwrap();
    assert!(sorted.table("Reading Log").unwrap().sort_rules.is_empty());
    assert!(sorted.problems().is_empty(), "{:?}", sorted.problems());
}

/// What a sort rule will not be.
#[test]
fn sort_rules_refuse_what_they_cannot_mean() {
    let mut doc = fixture!("numbers-values.numbers");
    let error = doc
        .set_sort_rules(
            "Zellarten",
            &[SortRule {
                column: 99,
                descending: false,
            }],
        )
        .expect_err("past the table");
    assert_eq!(error.refusal(), Some(Refusal::OutOfBounds));

    let twice = &[
        SortRule {
            column: 1,
            descending: false,
        },
        SortRule {
            column: 1,
            descending: true,
        },
    ];
    let error = doc
        .set_sort_rules("Zellarten", twice)
        .expect_err("one column, two directions");
    assert_eq!(error.refusal(), Some(Refusal::NotACell));
    assert!(doc.changed_streams().is_empty());
}

/// The app's own copy is the measure: Numbers opens the document, edits a cell,
/// saves — and the rules this crate wrote are in what the app wrote back.
///
/// Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_keeps_the_sort_rules_it_was_given() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = fixture!("numbers-values.numbers");
    doc.set_sort_rules(
        "Zellarten",
        &[SortRule {
            column: 1,
            descending: true,
        }],
    )
    .unwrap();
    let out = std::env::temp_dir().join("iwork-sort-rules.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/edit-and-save.sh");
    let output = std::process::Command::new(&script)
        .args([out.to_str().unwrap(), "Zellarten", "A1", "1"])
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document with written sort rules:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let after = Document::open(&out).unwrap();
    assert_eq!(
        after.table("Zellarten").unwrap().sort_rules,
        vec![SortRule {
            column: 1,
            descending: true
        }],
        "the app did not keep the rules it was given"
    );
    let _ = std::fs::remove_file(&out);
}

// -- the filter switch -------------------------------------------------------

/// A filter can be turned off and on, and whether **all** or **any** of its
/// rules must match can be changed. The rules themselves cannot be written:
/// this corpus carries exactly one compiled filter condition, and a rule built
/// from one sample is an extrapolation.
#[test]
fn a_filter_can_be_switched_off_and_on() {
    let mut doc = fixture!("numbers-rules.numbers");
    let table = "30-Day History Table";
    assert_eq!(
        doc.table(table).unwrap().filter.as_ref().map(|f| f.enabled),
        Some(true)
    );

    doc.set_filter_enabled(table, false, None).unwrap();
    let filter = doc.table(table).unwrap().filter.unwrap();
    assert!(!filter.enabled);
    assert!(!filter.match_any, "the match mode was not asked to change");
    assert!(!filter.rules.is_empty(), "the rules are left alone");

    doc.set_filter_enabled(table, true, Some(true)).unwrap();
    let filter = doc.table(table).unwrap().filter.unwrap();
    assert!(filter.enabled && filter.match_any);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    // A table with no filter rules says so rather than inventing a filter.
    let mut plain = fixture!("numbers-values.numbers");
    let error = plain
        .set_filter_enabled("Zellarten", false, None)
        .expect_err("no filter to switch");
    assert_eq!(error.refusal(), Some(Refusal::Missing));
}

/// **The switch is not the hiding.** Which rows a filter hides is stored, not
/// worked out when the document opens — so turning the filter off leaves them
/// hidden until the app is asked to apply it again.
///
/// Measured: with the filter written off, Numbers opened the document, edited a
/// cell and saved. The switch came back off and the ten hidden rows were still
/// hidden.
#[test]
fn switching_a_filter_off_does_not_unhide_its_rows() {
    let mut doc = fixture!("numbers-rules.numbers");
    let table = "30-Day History Table";
    let hidden = |doc: &Document| {
        doc.table(table)
            .unwrap()
            .row_extents
            .iter()
            .filter(|extent| extent.hidden())
            .count()
    };
    let before = hidden(&doc);
    assert!(before > 0, "the fixture's filter hides rows");
    doc.set_filter_enabled(table, false, None).unwrap();
    assert_eq!(hidden(&doc), before, "the stored hiding is untouched");
}

// -- conditional highlighting ------------------------------------------------

/// The number a highlight compares against, changed — in both places each rule
/// keeps it, and in both slots the set keeps its rules.
#[test]
fn a_highlight_can_be_given_a_new_threshold() {
    use iwork::table::Decimal;

    let mut doc = fixture!("numbers-rules.numbers");
    let set = doc.table("Overview").unwrap().conditional_styles[0].identifier;
    let rules = |doc: &Document| -> Vec<(i64, String, String)> {
        doc.table("Overview").unwrap().conditional_styles[0]
            .rules
            .iter()
            .map(|rule| {
                (
                    rule.predicate.kind,
                    rule.predicate
                        .values
                        .first()
                        .map(|v| v.to_text())
                        .unwrap_or_default(),
                    rule.predicate
                        .formula
                        .as_ref()
                        .map(|f| {
                            f.text(iwork::formula::Site::new(
                                &iwork::formula::Names::new(vec![]),
                                None,
                                (0, 0),
                            ))
                        })
                        .unwrap_or_default(),
                )
            })
            .collect()
    };
    let before = rules(&doc);
    assert_eq!(before[0].1, "0");

    doc.set_conditional_threshold(set, 0, Decimal::parse("500").unwrap())
        .unwrap();

    let after = rules(&doc);
    assert_eq!(after[0].1, "500", "the immediate value");
    assert!(after[0].2.contains("500"), "the formula: {}", after[0].2);
    assert_eq!(after[1], before[1], "the other rule was touched");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// What a threshold will not be written into.
#[test]
fn a_threshold_refuses_a_rule_it_does_not_understand() {
    use iwork::table::Decimal;

    let mut doc = fixture!("numbers-rules.numbers");
    let table = doc.table("Overview").unwrap();
    // The second set tests *text* — "begins with ↑" — and what predicate 36
    // compares is not established here.
    let text_set = table.conditional_styles[1].identifier;
    let error = doc
        .set_conditional_threshold(text_set, 0, Decimal::parse("1").unwrap())
        .expect_err("predicate 36 is not a number comparison");
    assert_eq!(error.refusal(), Some(Refusal::UnwritableValue));

    let set = doc.table("Overview").unwrap().conditional_styles[0].identifier;
    let error = doc
        .set_conditional_threshold(set, 9, Decimal::parse("1").unwrap())
        .expect_err("no ninth rule");
    assert_eq!(error.refusal(), Some(Refusal::NotFound));
    assert!(doc.changed_streams().is_empty());
}

/// The app's own copy again: Numbers opens the document, edits, saves — and the
/// threshold this crate wrote is in what the app wrote back.
///
/// Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_keeps_a_rewritten_threshold() {
    use iwork::table::Decimal;

    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = fixture!("numbers-rules.numbers");
    let set = doc.table("Overview").unwrap().conditional_styles[0].identifier;
    doc.set_conditional_threshold(set, 0, Decimal::parse("500").unwrap())
        .unwrap();
    let out = std::env::temp_dir().join("iwork-threshold.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/edit-and-save.sh");
    let output = std::process::Command::new(&script)
        .args([out.to_str().unwrap(), "Overview", "B3", "42"])
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document with a rewritten highlight:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let after = Document::open(&out).unwrap();
    let rule = &after.table("Overview").unwrap().conditional_styles[0].rules[0];
    assert_eq!(
        rule.predicate
            .values
            .first()
            .map(|v| v.to_text())
            .unwrap_or_default(),
        "500",
        "the app did not keep the threshold it was given"
    );
    let _ = std::fs::remove_file(&out);
}

/// The filter switch, through the app.
///
/// Ground rule 1a's other half: a write with no app round-trip is a write
/// nobody has watched the app accept. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_keeps_a_filter_switched_off() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = fixture!("numbers-rules.numbers");
    let table = "30-Day History Table";
    doc.set_filter_enabled(table, false, Some(true)).unwrap();
    let out = std::env::temp_dir().join("iwork-filter-switch.numbers");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/edit-and-save.sh");
    let output = std::process::Command::new(&script)
        .args([out.to_str().unwrap(), table, "A1", "1"])
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document with the filter switched off:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let after = Document::open(&out).unwrap();
    let filter = after.table(table).unwrap().filter.expect("still a filter");
    assert!(!filter.enabled, "the app turned the filter back on");
    assert!(filter.match_any, "the app did not keep the match mode");
    assert!(
        !filter.rules.is_empty(),
        "the app dropped the rules the switch left alone"
    );
    let _ = std::fs::remove_file(&out);
}
