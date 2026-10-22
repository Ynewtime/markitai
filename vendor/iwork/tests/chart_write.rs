//! Writing a chart — the private grid, and the copy that carries it.
//!
//! A chart holds its data twice and the two copies are not the same thing. The
//! **private grid** is `ChartArchive.grid`, written inline, and it is what the
//! chart draws — all a Pages or Keynote chart has. A Numbers chart fed by a
//! table has a second layer, a `TN.ChartMediatorArchive` of `TSCE` formulas,
//! and its grid is a *cache* of what those last evaluated to. Writing numbers
//! into that cache would make the chart disagree with the table it claims to
//! follow, so a chart with a mediator is refused by name rather than quietly
//! given numbers of its own.
//!
//! Adding a chart is a **copy**, not an invention: a dozen theme styles stand
//! behind one — a preset, a chart style and non-style, a legend pair, two axis
//! pairs, six series styles, a list of paragraph styles — and none of them can
//! be made up honestly. The caller names a chart to copy; what the source keeps
//! in its own component is copied and renumbered, and what lives in the theme
//! is shared, exactly as two charts made from one preset share it in the app.

use std::path::{Path, PathBuf};

use iwork::chart::{ChartData, GridValue};
use iwork::Document;

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&path);
    path
}

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

/// New numbers in a chart that is already there, and a shape it did not have.
#[test]
fn a_charts_private_grid_can_be_rewritten() {
    let mut doc = fixture!("keynote-charts.key");
    let chart = doc.charts()[0].identifier;
    let was = doc.charts()[0].chart_type;

    let data = ChartData::numbers(
        &["Nord", "Süd", "Ost"],
        &["Jan", "Feb", "Mär", "Apr"],
        &[
            &[10.0, 20.0, 30.0, 40.0],
            &[15.0, 25.0, 35.0, 45.0],
            &[5.0, 7.0, 9.0, 11.0],
        ],
    );
    doc.set_chart_data(chart, &data).unwrap();

    let after = doc
        .charts()
        .into_iter()
        .find(|c| c.identifier == chart)
        .unwrap();
    assert_eq!(after.grid.row_names, ["Nord", "Süd", "Ost"]);
    assert_eq!(after.grid.column_names, ["Jan", "Feb", "Mär", "Apr"]);
    assert_eq!(after.grid.value(2, 3), GridValue::Number(11.0));
    assert_eq!(after.chart_type, was, "only the grid was written");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// A row that was already there keeps its identity; a new one gets a new one.
#[test]
fn rewriting_keeps_the_ids_of_the_rows_that_stay() {
    let mut doc = fixture!("keynote-charts.key");
    let chart = doc.charts()[0].identifier;
    let before = doc.charts()[0].grid.row_ids.clone();
    assert_eq!(before.len(), 2, "the fixture chart has two series");

    doc.set_chart_data(
        chart,
        &ChartData::numbers(&["A", "B", "C"], &["x"], &[&[1.0], &[2.0], &[3.0]]),
    )
    .unwrap();

    let after = doc
        .charts()
        .into_iter()
        .find(|c| c.identifier == chart)
        .unwrap();
    assert_eq!(after.grid.row_ids.len(), 3);
    assert_eq!(
        after.grid.row_ids[0].0, before[0].0,
        "the first series is the same series"
    );
    assert_eq!(after.grid.row_ids[1].0, before[1].0);
    assert!(
        !before.iter().any(|(id, _)| *id == after.grid.row_ids[2].0),
        "the third is new and says so"
    );
}

/// A blank cell is a present, zero-length message: leaving it out would shift
/// every value after it one column to the left.
#[test]
fn a_blank_cell_stays_where_it_is() {
    let mut doc = fixture!("keynote-charts.key");
    let chart = doc.charts()[0].identifier;
    doc.set_chart_data(
        chart,
        &ChartData {
            row_names: vec!["A".into()],
            column_names: vec!["x".into(), "y".into(), "z".into()],
            rows: vec![vec![
                GridValue::Number(1.0),
                GridValue::Empty,
                GridValue::Number(3.0),
            ]],
        },
    )
    .unwrap();

    let grid = doc
        .charts()
        .into_iter()
        .find(|c| c.identifier == chart)
        .unwrap()
        .grid;
    assert_eq!(grid.value(0, 0), GridValue::Number(1.0));
    assert_eq!(grid.value(0, 1), GridValue::Empty);
    assert_eq!(grid.value(0, 2), GridValue::Number(3.0));
}

/// A Numbers chart's grid is a cache of formulas this crate does not evaluate,
/// so it is refused by name.
#[test]
fn a_chart_fed_by_a_table_is_refused() {
    let mut doc = fixture!("numbers-charts.numbers");
    let chart = doc
        .charts()
        .into_iter()
        .find(|c| c.mediator.is_some())
        .expect("a Numbers chart follows a table")
        .identifier;
    let refusal = doc
        .set_chart_data(chart, &ChartData::numbers(&["A"], &["x"], &[&[1.0]]))
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("fed by a table"), "{refusal}");
    assert!(refusal.contains("cache"), "{refusal}");
}

/// Names and values that do not line up are refused rather than drawn short.
#[test]
fn data_that_does_not_line_up_is_refused() {
    let mut doc = fixture!("keynote-charts.key");
    let chart = doc.charts()[0].identifier;

    let refusal = doc
        .set_chart_data(
            chart,
            &ChartData {
                row_names: vec!["A".into(), "B".into()],
                column_names: vec!["x".into()],
                rows: vec![vec![GridValue::Number(1.0)]],
            },
        )
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("row name"), "{refusal}");

    let refusal = doc
        .set_chart_data(
            chart,
            &ChartData {
                row_names: vec!["A".into()],
                column_names: vec!["x".into()],
                rows: vec![vec![GridValue::Number(1.0), GridValue::Number(2.0)]],
            },
        )
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("column name"), "{refusal}");
}

/// A copy is a chart of its own: its own object, its own private non-styles,
/// the theme's styles shared.
#[test]
fn a_copied_chart_owns_what_was_private_and_shares_the_theme() {
    let mut doc = fixture!("pages-numbering.pages");
    let from = doc.charts()[0].identifier;
    let source = doc.charts()[0].clone();

    let chart = doc
        .add_chart(
            "page 1",
            from,
            &ChartData::numbers(
                &["Alpha", "Beta"],
                &["Q1", "Q2"],
                &[&[7.0, 8.0], &[3.0, 2.0]],
            ),
            (100.0, 100.0),
            (400.0, 300.0),
        )
        .unwrap();

    let copy = doc
        .charts()
        .into_iter()
        .find(|c| c.identifier == chart)
        .expect("the copy is a chart");
    assert_ne!(copy.identifier, source.identifier);
    assert_eq!(
        copy.chart_type, source.chart_type,
        "it looks like its source"
    );
    assert_eq!(copy.grid.row_names, ["Alpha", "Beta"]);
    assert_eq!(
        copy.series_theme_styles, source.series_theme_styles,
        "the theme's styles are shared, as they are between two charts in the app"
    );
    assert_ne!(
        copy.chart_non_style, source.chart_non_style,
        "what the source kept in its own component is the copy's own"
    );
    // It is on the page it was put on, at the rectangle it was given, and a
    // Pages page owns nothing — so no parent.
    let drawable = doc.drawable(chart).unwrap();
    assert_eq!(drawable.parent, None);
    assert_eq!(
        drawable.placement,
        iwork::drawable::Placement::Floating { page: 0 }
    );
    let frame = drawable.frame(None);
    assert_eq!((frame.x, frame.y), (100.0, 100.0));
    assert_eq!((frame.width, frame.height), (400.0, 300.0));

    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());
}

/// A slide owns the copy it is given, the way it owns every other drawable.
#[test]
fn a_slide_owns_a_chart_copied_onto_it() {
    let mut doc = fixture!("keynote-charts.key");
    let from = doc.charts()[0].identifier;
    let slide = doc.slides()[0].identifier;
    let chart = doc
        .add_chart(
            &slide.to_string(),
            from,
            &ChartData::numbers(&["A"], &["x"], &[&[1.0]]),
            (100.0, 100.0),
            (400.0, 300.0),
        )
        .unwrap();
    assert_eq!(doc.drawable(chart).unwrap().parent, Some(slide));
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Copying a chart that follows a table would claim to follow it too.
#[test]
fn a_chart_fed_by_a_table_is_not_copied_either() {
    let mut doc = fixture!("numbers-charts.numbers");
    let from = doc
        .charts()
        .into_iter()
        .find(|c| c.mediator.is_some())
        .unwrap()
        .identifier;
    let sheet = doc.tables()[0]
        .sheet
        .clone()
        .expect("a Numbers table is on a sheet");
    let refusal = doc
        .add_chart(
            &sheet,
            from,
            &ChartData::numbers(&["A"], &["x"], &[&[1.0]]),
            (0.0, 0.0),
            (100.0, 100.0),
        )
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("mediator"), "{refusal}");
}

/// The apps read a rewritten chart back and write it out again. Off unless
/// `IWORK_APP_CHECK=1`.
#[test]
fn the_apps_resave_a_chart_this_crate_rewrote() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the resave");
        return;
    }
    let resave = |path: &Path| {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/resave.sh");
        let status = std::process::Command::new(&script)
            .arg(path)
            .status()
            .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
        assert!(
            status.success(),
            "the app would not resave {}",
            path.display()
        );
    };
    let data = ChartData::numbers(
        &["Nord", "Süd", "Ost"],
        &["Jan", "Feb", "Mär", "Apr"],
        &[
            &[10.0, 20.0, 30.0, 40.0],
            &[15.0, 25.0, 35.0, 45.0],
            &[5.0, 7.0, 9.0, 11.0],
        ],
    );
    let kept = |path: &Path| {
        let after = Document::open(path).unwrap();
        let chart = after
            .charts()
            .into_iter()
            .find(|c| c.grid.row_names == ["Nord", "Süd", "Ost"])
            .unwrap_or_else(|| panic!("the app dropped the data out of {}", path.display()));
        assert_eq!(chart.grid.column_names, ["Jan", "Feb", "Mär", "Apr"]);
        assert_eq!(chart.grid.value(2, 3), GridValue::Number(11.0));
        assert!(after.problems().is_empty(), "{:?}", after.problems());
    };

    let mut deck = fixture!("keynote-charts.key");
    let chart = deck.charts()[0].identifier;
    deck.set_chart_data(chart, &data).unwrap();
    let out = scratch("iwork-chart.key");
    deck.save(&out).unwrap();
    resave(&out);
    kept(&out);
    let _ = std::fs::remove_dir_all(&out);

    let mut doc = fixture!("pages-numbering.pages");
    let from = doc.charts()[0].identifier;
    doc.add_chart("page 1", from, &data, (100.0, 100.0), (400.0, 300.0))
        .unwrap();
    let out = scratch("iwork-chart.pages");
    doc.save(&out).unwrap();
    resave(&out);
    kept(&out);
    let _ = std::fs::remove_dir_all(&out);
}

// -- the mediator: a chart that follows a table ------------------------------

/// Take the app's mediator out of a chart, the way a chart copied into a
/// Numbers document from elsewhere has none.
///
/// A chart with a mediator is refused by `bind_chart` — replacing one means
/// taking the first out of the calculation engine, which is unwritten — so a
/// test needs a chart without one, and every Numbers chart in the corpus has
/// one. This is the wire-level edit that makes one.
fn unbind(doc: &mut Document, chart: u64) {
    let mut archive = doc.archive(chart).unwrap();
    let mut model = iwork::pb::decode_nested(archive.bytes(10000).unwrap()).unwrap();
    model.clear(iwork::chart::field::MEDIATOR);
    archive.set_in_order(10000, iwork::pb::Value::Bytes(model.encode()));
    doc.set_archive_for(chart, &archive).unwrap();
}

/// A mediator this crate wrote, read back by this crate's own reader.
#[test]
fn a_chart_can_be_made_to_follow_a_table() {
    use iwork::chart::ChartBinding;

    let mut doc = fixture!("numbers-charts.numbers");
    let chart = 905027; // the two-axis chart, fed by 'Average Rainfall'
    unbind(&mut doc, chart);
    assert!(doc
        .charts()
        .into_iter()
        .find(|c| c.identifier == chart)
        .unwrap()
        .mediator
        .is_none());

    let mediator = doc
        .bind_chart(
            chart,
            "Average Rainfall",
            &ChartBinding {
                series: vec!["B2:B13".into(), "C2:C13".into()],
                row_labels: vec!["A2:A13".into()],
                column_labels: vec!["B1".into(), "C1".into()],
                series_by_row: false,
            },
        )
        .unwrap();

    let bound = doc
        .charts()
        .into_iter()
        .find(|c| c.identifier == chart)
        .unwrap();
    assert_eq!(bound.mediator, Some(mediator));
    let references = bound.references.as_ref().expect("it follows a table now");
    assert_eq!(
        references
            .data
            .iter()
            .map(|r| r.to_text())
            .collect::<Vec<_>>(),
        vec!["Average Rainfall!$B$2:$B$13", "Average Rainfall!$C$2:$C$13"]
    );
    assert_eq!(references.row_labels.len(), 1);
    assert_eq!(references.column_labels.len(), 2);
    // Every reference goes through function 175, as every one in the corpus
    // does.
    assert!(references.data.iter().all(|r| r.wrapped));

    // The mediator is keyed to the calculation engine by an entity id, and the
    // owner that makes it live is keyed by the same sixteen bytes.
    let archive = doc.archive(mediator).unwrap();
    let entity = String::from_utf8_lossy(archive.bytes(2).unwrap()).into_owned();
    let uid = iwork::chart::uuid_from_entity(&entity).expect("a UUID");
    let owned = doc
        .objects()
        .filter(|(_, o)| o.message_type() == iwork::calc::TYPE_OWNER_DEPENDENCIES)
        .filter_map(|(_, o)| iwork::pb::Message::decode(o.payload()).ok())
        .filter(|owner| owner.varint(3) == Some(2))
        .any(|owner| {
            owner
                .bytes(1)
                .and_then(iwork::pb::decode_nested)
                .is_some_and(|u| u.varint(1) == Some(uid.lower) && u.varint(2) == Some(uid.upper))
        });
    assert!(owned, "no kind-2 owner carries the mediator's identity");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// What binding will not do.
#[test]
fn binding_refuses_what_it_cannot_do_honestly() {
    use iwork::chart::ChartBinding;
    use iwork::Refusal;

    let mut doc = fixture!("numbers-charts.numbers");
    let binding = |series: &str| ChartBinding {
        series: vec![series.into()],
        ..Default::default()
    };

    // A chart that already follows a table.
    let error = doc
        .bind_chart(905027, "Average Rainfall", &binding("B2:B13"))
        .expect_err("it already has a mediator");
    assert!(error.to_string().contains("already follows a table"));

    unbind(&mut doc, 905027);
    // A range outside the table.
    let error = doc
        .bind_chart(905027, "Average Rainfall", &binding("B2:B999"))
        .expect_err("past the table");
    assert_eq!(error.refusal(), Some(Refusal::OutOfBounds));
    // No series at all.
    let error = doc
        .bind_chart(905027, "Average Rainfall", &ChartBinding::default())
        .expect_err("a chart follows something");
    assert_eq!(error.refusal(), Some(Refusal::NotACell));
    // A chart that is not there.
    assert_eq!(
        doc.bind_chart(1, "Average Rainfall", &binding("B2:B13"))
            .unwrap_err()
            .refusal(),
        Some(Refusal::NotFound)
    );
}

/// The one that matters, and the only measure there is: **the app
/// recalculates the chart from the formulas this crate wrote.**
///
/// A chart's grid is a cache of what its mediator last evaluated to, so a
/// mediator the app does not believe in leaves the cache exactly as it was.
/// This binds a chart, hands the document to Numbers, has the app itself change
/// a cell the binding names, saves, and reads the cache back.
///
/// Off unless `IWORK_APP_CHECK=1`.
#[test]
fn numbers_recalculates_a_chart_bound_by_this_crate() {
    use iwork::chart::ChartBinding;

    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut doc = fixture!("numbers-charts.numbers");
    let chart = 905027;
    unbind(&mut doc, chart);
    doc.bind_chart(
        chart,
        "Average Rainfall",
        &ChartBinding {
            series: vec!["B2:B13".into(), "C2:C13".into()],
            row_labels: vec!["A2:A13".into()],
            column_labels: vec!["B1".into(), "C1".into()],
            series_by_row: false,
        },
    )
    .unwrap();

    let out = scratch("iwork-bound-chart.numbers");
    doc.save(&out).unwrap();

    // The app edits a cell the chart follows, and saves.
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/edit-and-save.sh");
    let output = std::process::Command::new(&script)
        .args([out.to_str().unwrap(), "Average Rainfall", "B2", "999"])
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "Numbers would not open a document with a written mediator:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    // The cache the chart draws followed the table, which only happens for a
    // mediator the engine believes.
    let after = Document::open(&out).unwrap();
    let grid = &after
        .charts()
        .into_iter()
        .find(|c| c.identifier == chart)
        .expect("the chart survived")
        .grid;
    assert_eq!(
        grid.rows[0][0].to_text(),
        "999",
        "the app did not recalculate the chart from the written mediator"
    );
    let _ = std::fs::remove_file(&out);
}
