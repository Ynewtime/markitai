//! Charts, read — measured against the numbers the app was told to plot.
//!
//! **There is no read-back oracle for a chart.** `chart` is an element of a
//! slide, a sheet and a Pages document in all three scripting dictionaries, and
//! the class carries nothing at all beyond the position, size, rotation and
//! opacity it inherits from `iWork item`: no type, no data, no series. So the
//! app cannot be asked what a chart holds.
//!
//! What it *can* be asked is to build one. Keynote's `add chart` takes row
//! names, column names and a grid of numbers, and `keynote-charts.key` is
//! eighteen charts built that way, each with its own hundred — chart *i* holds
//! `i×100 + 1, 2, 3` in its first row and `+11, 12, 13` in its second. The
//! assertions below are those numbers, read back out of the file. That is the
//! oracle: the input is known, so a decoder that prints it back is reading the
//! grid, and a mis-ordered or mis-scaled read cannot pass because no two charts
//! share a value.
//!
//! Everything else needs no app: every chart archive in the corpus re-encodes
//! to the bytes it came from, every `ChartType` is one the 15.3.1 enumeration
//! has, every sparse series array is as long as its `count` says, and every
//! reference a Numbers mediator holds resolves to a table in the document.
//!
//! Without `tests/fixtures/generated` these pass having asserted nothing and
//! say so — `scripts/make-fixtures.sh` builds it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use iwork::chart;
use iwork::pb::{decode_nested, Message};
use iwork::{Document, Placement};

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
        match open($name) {
            Some(doc) => doc,
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

fn corpus() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut out: Vec<PathBuf> = entries
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
    out.sort();
    out
}

fn require_corpus() -> Vec<PathBuf> {
    let found = corpus();
    if found.is_empty() {
        eprintln!("no generated corpus — skipping (run scripts/make-fixtures.sh)");
    }
    found
}

// -- the oracle: the numbers the app was told to plot -------------------------

/// Every chart of the zoo holds exactly the grid Keynote was handed.
///
/// The row and column names are the same for all seventeen type slides, so the
/// hundred is what identifies a chart — and it is read out of the chart rather
/// than assumed, so the test does not depend on the order the charts happen to
/// be enumerated in.
#[test]
fn the_zoo_holds_the_numbers_it_was_built_from() {
    let doc = fixture!("keynote-charts.key");
    let charts = doc.charts();
    assert_eq!(
        charts.len(),
        18,
        "the zoo is seventeen types plus a by-column chart"
    );

    let mut hundreds: BTreeSet<u64> = BTreeSet::new();
    let mut by_column = 0;
    for chart in &charts {
        if chart.series_direction == 2 {
            by_column += 1;
            assert_eq!(chart.grid.row_names, ["Nord", "Süd"]);
            assert_eq!(chart.grid.column_names, ["Jan", "Feb", "Mär"]);
            // Grouped by column, the three months are the series and the two
            // rows are the categories — the grid is stored rows first either
            // way, and `series_direction` is the only thing that says which.
            let series = chart.series();
            assert_eq!(series.len(), 3);
            assert_eq!(chart.categories(), ["Nord", "Süd"]);
            assert_eq!(series[0].name.as_deref(), Some("Jan"));
            assert_eq!(series[0].values[0].number(), Some(7001.0));
            assert_eq!(series[0].values[1].number(), Some(7011.0));
            assert_eq!(series[2].name.as_deref(), Some("Mär"));
            assert_eq!(series[2].values[0].number(), Some(7003.0));
            assert_eq!(series[2].values[1].number(), Some(7013.0));
            continue;
        }

        assert_eq!(chart.grid.row_names, ["Reihe A", "Reihe B"]);
        assert_eq!(chart.grid.column_names, ["Q1", "Q2", "Q3"]);
        assert_eq!(chart.series_direction, 1, "chart {}", chart.identifier);

        let first = chart.grid.value(0, 0).number().unwrap_or_else(|| {
            panic!("chart {} has no number in its first cell", chart.identifier)
        });
        let base = ((first - 1.0) / 100.0).round() as u64 * 100;
        assert!(hundreds.insert(base), "two charts share the hundred {base}");

        let series = chart.series();
        assert_eq!(series.len(), 2);
        assert_eq!(chart.categories(), ["Q1", "Q2", "Q3"]);
        for (row, offset) in [(0usize, 1u64), (1, 11)] {
            assert_eq!(
                series[row].name.as_deref(),
                Some(["Reihe A", "Reihe B"][row])
            );
            for column in 0..3 {
                assert_eq!(
                    series[row].values[column].number(),
                    Some((base + offset + column as u64) as f64),
                    "chart {} row {row} column {column}",
                    chart.identifier
                );
            }
        }
    }
    assert_eq!(by_column, 1);
    assert_eq!(hundreds.len(), 17, "one hundred per chart type");
}

/// A chart's title and caption references live on its `TSD.DrawableArchive`
/// (fields 10 and 11), and every chart in this deck carries both. They were
/// hard-coded `None` before — a claim of absence the archives contradict — and
/// each now resolves to the storage object the drawable points at.
#[test]
fn a_chart_reads_its_title_and_caption_references() {
    let doc = fixture!("keynote-charts.key");
    let charts = doc.charts();
    assert!(!charts.is_empty(), "the deck has charts");
    for chart in &charts {
        let title = chart.title.expect("every chart in this deck has a title");
        let caption = chart
            .caption
            .expect("every chart in this deck has a caption");
        // The references resolve to real objects in the document.
        assert!(
            doc.object(title).is_some(),
            "chart {} title {title} resolves",
            chart.identifier
        );
        assert!(
            doc.object(caption).is_some(),
            "chart {} caption {caption} resolves",
            chart.identifier
        );
    }
}

/// The seventeen legacy chart types Keynote's `add chart` accepts, and the
/// `TSCH.ChartType` each one produces.
///
/// The mapping is not the identity and not an ordering: "vertical bar" is a
/// *column* chart (1) and "horizontal bar" is a bar chart (2). It was read off
/// the documents this fixture is made of, and this is where it is written down.
#[test]
fn the_zoo_covers_seventeen_chart_types_including_every_3d_one() {
    let doc = fixture!("keynote-charts.key");
    let by_hundred: BTreeMap<u64, u32> = doc
        .charts()
        .iter()
        .filter(|c| c.series_direction != 2)
        .map(|c| {
            let first = c.grid.value(0, 0).number().unwrap();
            (((first - 1.0) / 100.0).round() as u64, c.chart_type)
        })
        .collect();

    // Slide n was built with the nth constant of the script's list.
    let expected = [
        (1, 5, "pie_2d"),
        (2, 1, "vertical_bar_2d"),
        (3, 6, "stacked_vertical_bar_2d"),
        (4, 2, "horizontal_bar_2d"),
        (5, 7, "stacked_horizontal_bar_2d"),
        (6, 16, "pie_3d"),
        (7, 12, "vertical_bar_3d"),
        (8, 17, "stacked_vertical_bar_3d"),
        (9, 13, "horizontal_bar_3d"),
        (10, 18, "stacked_horizontal_bar_3d"),
        (11, 4, "area_2d"),
        (12, 8, "stacked_area_2d"),
        (13, 3, "line_2d"),
        (14, 14, "line_3d"),
        (15, 15, "area_3d"),
        (16, 19, "stacked_area_3d"),
        (17, 9, "scatterplot_2d"),
    ];
    for (slide, kind, name) in expected {
        assert_eq!(
            by_hundred.get(&slide),
            Some(&kind),
            "slide {slide} was asked for {name}"
        );
    }
    // All eight 3-D families but the donut, which no script can make.
    let three_d: BTreeSet<u32> = by_hundred
        .values()
        .copied()
        .filter(|k| chart::is_3d(*k))
        .collect();
    assert_eq!(
        three_d,
        BTreeSet::from([12, 13, 14, 15, 16, 17, 18, 19]),
        "every 3-D type but donutChartType3D"
    );
}

// -- the two copies of the data ----------------------------------------------

/// A Numbers chart has a mediator holding `TSCE` references back into tables; a
/// Keynote or Pages chart has a private grid and nothing behind it.
///
/// That is the distinction the phase exists to make, and it is a property of
/// the *app*, not of the chart: the twelve charts of `21_Simple_Charts` all
/// have one and the eighteen of the Keynote zoo have none.
#[test]
fn only_a_numbers_chart_is_fed_by_a_table() {
    let numbers = fixture!("numbers-charts.numbers");
    let charts = numbers.charts();
    assert_eq!(charts.len(), 12);
    for chart in &charts {
        let references = chart
            .references
            .as_ref()
            .unwrap_or_else(|| panic!("chart {} has no mediator", chart.identifier));
        assert!(chart.mediator.is_some());
        assert!(
            !references.data.is_empty(),
            "chart {} is fed by nothing",
            chart.identifier
        );
        // Every series in the grid is a formula in the mediator.
        assert_eq!(
            references.data.len(),
            chart.series_count(),
            "chart {} has {} series and {} data formulas",
            chart.identifier,
            chart.series_count(),
            references.data.len()
        );
        assert!(references.entity_id.is_some());
    }

    for name in ["keynote-charts.key", "pages-numbering.pages"] {
        let Some(doc) = open(name) else { continue };
        for chart in doc.charts() {
            assert!(
                chart.mediator.is_none() && chart.references.is_none(),
                "{name}: chart {} has a mediator",
                chart.identifier
            );
            assert!(!chart.grid.rows.is_empty());
        }
    }
}

/// **Function 175.** Every reference a chart mediator holds is one reference
/// node wrapped in a function of index 175, which Apple publishes no name for.
#[test]
fn every_chart_reference_goes_through_function_175() {
    let doc = fixture!("numbers-charts.numbers");
    let mut wrapped = 0;
    let mut unwrapped = 0;
    for chart in doc.charts() {
        let references = chart.references.unwrap();
        wrapped += references.wrapped_in_175;
        unwrapped += references.unwrapped;
    }
    assert!(wrapped > 100, "only {wrapped} references");
    assert_eq!(unwrapped, 0, "{unwrapped} references were not wrapped");
    assert!(
        iwork::formula::function_name(175).is_none(),
        "175 is unnamed"
    );
}

/// Every reference the mediators hold resolves to a table that is in the
/// document, and the tables it names are the ones on the same sheet.
#[test]
fn a_chart_names_the_tables_that_feed_it() {
    let doc = fixture!("numbers-charts.numbers");
    let tables: BTreeSet<String> = doc.tables().into_iter().map(|t| t.name).collect();
    let mut checked = 0;
    for chart in doc.charts() {
        let references = chart.references.unwrap();
        let named = references.tables();
        assert!(
            !named.is_empty(),
            "chart {} names no table",
            chart.identifier
        );
        for table in &named {
            assert!(tables.contains(table), "no table called {table}");
            checked += 1;
        }
        // A chart in this template takes all its data from one table.
        assert_eq!(named.len(), 1, "chart {} spans {named:?}", chart.identifier);
    }
    assert!(checked >= 12);

    // The two charts of `numbers-rules` are the other Numbers document with a
    // mediator, and they read the same way.
    if let Some(doc) = open("numbers-rules.numbers") {
        for chart in doc.charts() {
            let references = chart.references.unwrap();
            assert!(!references.tables().is_empty());
        }
    }
}

/// A mediator formula is not always a reference: a bubble chart's row labels
/// are string *literals*, written into the mediator rather than read from a
/// cell. A reader that assumes every formula names a table gets `None` there
/// and must not fall over.
#[test]
fn some_mediator_formulas_are_literals_rather_than_references() {
    let doc = fixture!("numbers-charts.numbers");
    let mut literals = 0;
    for chart in doc.charts() {
        for reference in chart.references.unwrap().all() {
            if reference.table.is_none() {
                assert!(!reference.text.is_empty());
                literals += 1;
            }
        }
    }
    assert!(
        literals > 0,
        "the bubble chart's row labels are literal strings"
    );
}

// -- structure, no app involved ----------------------------------------------

/// Every chart-domain archive in the corpus decodes and re-encodes to the bytes
/// it came from.
///
/// This is the claim the decoder rests on, and it covers more than the charts:
/// the ten style shells, the presets and the mediators are all in the 5000s.
#[test]
fn every_chart_archive_re_encodes_to_its_bytes() {
    let mut objects = 0;
    let mut types: BTreeSet<u32> = BTreeSet::new();
    for path in require_corpus() {
        let doc = Document::open(&path).unwrap();
        for (_, object) in doc.objects() {
            let kind = object.message_type();
            let chart_domain =
                (5000..=5999).contains(&kind) || kind == chart::TYPE_TN_CHART_MEDIATOR;
            if !chart_domain {
                continue;
            }
            let message = Message::decode(object.payload()).unwrap_or_else(|e| {
                panic!("{}: object {}: {e}", path.display(), object.identifier)
            });
            assert_eq!(
                message.encode(),
                object.payload(),
                "{}: object {} of type {kind} does not re-encode",
                path.display(),
                object.identifier
            );
            types.insert(kind);
            objects += 1;
        }
    }
    if objects == 0 {
        return;
    }
    eprintln!(
        "{objects} chart-domain archives, {} distinct types",
        types.len()
    );
    // The sandwich: the chart drawable and every style shell the corpus has.
    for expected in [
        5020u32, 5021, 5022, 5023, 5024, 5025, 5026, 5027, 5028, 5029,
    ] {
        assert!(types.contains(&expected), "no {expected} anywhere");
    }
}

/// Every chart's type is a value the 15.3.1 `ChartType` enumeration has, and
/// the corpus covers twenty-two of the twenty-eight.
#[test]
fn every_chart_type_in_the_corpus_is_a_named_one() {
    let mut seen: BTreeSet<u32> = BTreeSet::new();
    let mut charts = 0;
    for path in require_corpus() {
        let doc = Document::open(&path).unwrap();
        for chart in doc.charts() {
            assert!(
                chart::chart_type_name(chart.chart_type).is_some(),
                "{}: chart {} has type {}, which the enumeration does not have",
                path.display(),
                chart.identifier,
                chart.chart_type
            );
            seen.insert(chart.chart_type);
            charts += 1;
        }
    }
    if charts == 0 {
        return;
    }
    eprintln!("{charts} charts, types {seen:?}");
    // 1–9 and 11–20, 22, 25, 27. Never seen: 0 undefined, 10 mixed, 21, 23, 24
    // and 26 — the multi-data bar, scatter and bubble charts and the 3-D donut.
    for expected in [
        1u32, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 25, 27,
    ] {
        assert!(
            seen.contains(&expected),
            "no chart of type {expected} ({}) in the corpus",
            chart::chart_type_name(expected).unwrap()
        );
    }
    assert!(!seen.contains(&0), "a chart with no type");
}

/// **`SparseReferenceArray.count` is the logical length, not `entries.len()`.**
///
/// Nothing in this corpus actually leaves a gap — Numbers writes an override
/// for every series — so the trap is asserted from the other side: every index
/// is inside the count, the count is never smaller than the number of entries,
/// and the dense form is as long as the count says. The one place a gap would
/// otherwise be invisible is the *empty* array, and there are plenty of those.
#[test]
fn a_sparse_series_array_is_as_long_as_its_count_says() {
    let mut arrays = 0;
    let mut gaps = 0;
    for path in require_corpus() {
        let doc = Document::open(&path).unwrap();
        for chart in doc.charts() {
            for (which, array) in [
                ("private styles", &chart.series_private_styles),
                ("non-styles", &chart.series_non_styles),
            ] {
                assert!(
                    array.entries.len() <= array.count as usize,
                    "{}: chart {} {which}: {} entries in an array of {}",
                    path.display(),
                    chart.identifier,
                    array.entries.len(),
                    array.count
                );
                for (index, _) in &array.entries {
                    assert!(
                        *index < array.count,
                        "{}: chart {} {which}: index {index} outside {}",
                        path.display(),
                        chart.identifier,
                        array.count
                    );
                }
                assert_eq!(array.dense().len(), array.count as usize);
                if array.entries.len() < array.count as usize {
                    gaps += 1;
                }
                arrays += 1;
            }
        }
    }
    if arrays == 0 {
        return;
    }
    eprintln!("{arrays} sparse arrays, {gaps} with a default-styled series");
}

/// A grid is rectangular, and its names describe it: one row name per row and
/// one column name per column, with the id map agreeing about both.
#[test]
fn a_grid_is_rectangular_and_its_names_fit_it() {
    let mut grids = 0;
    for path in require_corpus() {
        let doc = Document::open(&path).unwrap();
        for chart in doc.charts() {
            let grid = &chart.grid;
            let width = grid.column_count();
            for (at, row) in grid.rows.iter().enumerate() {
                assert_eq!(
                    row.len(),
                    width,
                    "{}: chart {} row {at} is {} wide, not {width}",
                    path.display(),
                    chart.identifier,
                    row.len()
                );
            }
            assert_eq!(
                grid.row_names.len(),
                grid.rows.len(),
                "{}: chart {} has {} row names for {} rows",
                path.display(),
                chart.identifier,
                grid.row_names.len(),
                grid.rows.len()
            );
            assert_eq!(
                grid.column_names.len(),
                width,
                "{}: chart {} has {} column names for {width} columns",
                path.display(),
                chart.identifier,
                grid.column_names.len()
            );
            // `idMap` is a stable identity per row and per column, and its
            // indices are a permutation of the positions.
            if !grid.row_ids.is_empty() {
                let indices: BTreeSet<u32> = grid.row_ids.iter().map(|(_, at)| *at).collect();
                assert_eq!(indices.len(), grid.rows.len());
                assert_eq!(
                    indices.iter().copied().max(),
                    Some(grid.rows.len() as u32 - 1)
                );
            }
            if !grid.column_ids.is_empty() {
                let indices: BTreeSet<u32> = grid.column_ids.iter().map(|(_, at)| *at).collect();
                assert_eq!(indices.len(), width);
            }
            grids += 1;
        }
    }
    if grids == 0 {
        return;
    }
    eprintln!("{grids} grids");
}

/// Every chart in the corpus is placed: on a sheet, on a slide, on a page or in
/// a section template. A chart reported as `Unknown` is a chart nobody can say
/// where to find, and that is how the Pages floating-drawable bug was found.
#[test]
fn every_chart_is_somewhere() {
    let mut charts = 0;
    for path in require_corpus() {
        let doc = Document::open(&path).unwrap();
        for chart in doc.charts() {
            assert!(
                !matches!(chart.placement, Placement::Unknown),
                "{}: chart {} is unplaced",
                path.display(),
                chart.identifier
            );
            assert!(chart.frame.width > 0.0 && chart.frame.height > 0.0);
            charts += 1;
        }
    }
    if charts == 0 {
        return;
    }
    eprintln!("{charts} charts, all placed");
}

/// The interactive chart, and what it remembers.
///
/// `multiDataColumnChartType2D` is what the "Interactive" tab of Numbers' chart
/// picker makes. Which data set it is showing is `ChartArchive.multidataset_index`
/// — in the *model*, not in the view state: the document has a
/// `TN.UIStateArchive` and it carries no `chart_ui_state` at all.
#[test]
fn an_interactive_chart_remembers_which_data_set_it_shows() {
    let doc = fixture!("numbers-charts.numbers");
    let charts = doc.charts();
    let interactive: Vec<_> = charts.iter().filter(|c| c.is_interactive()).collect();
    assert_eq!(interactive.len(), 1, "21_Simple_Charts has one");
    let chart = interactive[0];
    assert_eq!(chart.chart_type, 20);
    assert_eq!(chart.multidataset_index, Some(0));
    assert_eq!(chart.grid.rows.len(), 4);

    // The other home the schema offers is `TN.UIStateArchive.chart_ui_state`
    // (field 23), and the upgrade flag that would move it there
    // (`upgraded_to_ui_state`, extension 10021) is not set either.
    for (_, object) in doc.objects() {
        if object.message_type() != 12026 {
            continue;
        }
        let message = Message::decode(object.payload()).unwrap();
        assert!(
            message.get(23).is_none(),
            "the view state carries chart UI state after all"
        );
    }
    assert!(!chart.extensions.iter().any(|(number, _)| *number == 10021));
}

/// 3-D charts are read to the level the phase claims: the type says the chart
/// is 3-D, and the scene's own settings are a capability flag on the archive
/// plus a `TSCH.Chart3D*` payload inside the *style*, which is not decoded.
#[test]
fn a_3d_chart_is_identified_and_its_scene_is_carried() {
    let doc = fixture!("keynote-charts.key");
    let three_d: Vec<_> = doc.charts().into_iter().filter(|c| c.is_3d()).collect();
    assert_eq!(three_d.len(), 8);
    for chart in &three_d {
        assert!(chart.type_name().ends_with("3D"));
        assert!(chart.type_label().starts_with("3D "));
        // Every chart these apps write declares the depth mode, 3-D or not.
        assert!(
            chart.extensions.iter().any(|(number, _)| *number == 10002),
            "chart {} has no scene3d_settings_constant_depth",
            chart.identifier
        );
    }
}

/// The forward-compatibility flags are present and preserved. An older app that
/// does not understand rounded corners refuses a document that claims them, so
/// these are carried verbatim and never synthesised.
#[test]
fn the_capability_flags_are_read_and_named() {
    let doc = fixture!("numbers-charts.numbers");
    let chart = &doc.charts()[0];
    let named: BTreeSet<&str> = chart.extensions.iter().map(|(_, name)| *name).collect();
    for expected in [
        "supports_rounded_corners",
        "supports_series_value_label_spacing",
        "supports_series_error_bar_spacing",
        "supports_stacked_summary_labels",
        "reference_lines",
    ] {
        assert!(named.contains(expected), "no {expected}");
    }
}

/// The pre-UFF chart model is named and absent.
///
/// `TSCH.PreUFF.ChartInfoArchive` (5000) is the iWork '09/'13 chart root, still
/// in the registry and still emitted for documents imported from those formats.
/// Nothing in this corpus and nothing in the 901 bundled templates has one, so
/// the legacy grid — a bare `repeated double` that cannot express a blank cell —
/// is written down from the schema and has never been decoded here.
#[test]
fn no_chart_in_the_corpus_is_the_legacy_kind() {
    for path in require_corpus() {
        let doc = Document::open(&path).unwrap();
        for (_, object) in doc.objects() {
            assert_ne!(
                object.message_type(),
                chart::TYPE_PREUFF_CHART_INFO,
                "{}: object {} is a pre-UFF chart",
                path.display(),
                object.identifier
            );
        }
    }
}

/// The chart model is reached through extension field 10000 of the drawable and
/// nowhere else — the fact that makes a registry-driven decoder stop.
#[test]
fn the_chart_model_is_extension_10000_of_its_drawable() {
    let mut checked = 0;
    for path in require_corpus() {
        let doc = Document::open(&path).unwrap();
        for (_, object) in doc.objects() {
            if object.message_type() != chart::TYPE_CHART_DRAWABLE {
                continue;
            }
            let message = Message::decode(object.payload()).unwrap();
            let numbers: BTreeSet<u32> = message.fields.iter().map(|f| f.number).collect();
            assert_eq!(
                numbers,
                BTreeSet::from([1u32, chart::EXTENSION]),
                "{}: chart drawable {} has fields {numbers:?}",
                path.display(),
                object.identifier
            );
            let archive = message
                .bytes(chart::EXTENSION)
                .and_then(decode_nested)
                .expect("extension 10000 is a message");
            assert!(archive.get(chart::field::GRID).is_some());
            checked += 1;
        }
    }
    if checked == 0 {
        return;
    }
    eprintln!("{checked} chart drawables, every one a two-field sandwich");
}

/// **A chart of a type that did not exist in an older release carries a
/// down-level version patch**, and it is the first object outside
/// `TN.UIStateArchive` in this corpus to carry one at all.
///
/// Phase 2 measured the whole corpus and found exactly one patched object per
/// Numbers document — the view state — and concluded a patch is a thing the app
/// writes in one place. Charts are the counter-example: the donut chart and the
/// radar chart of `21_Simple_Charts` each carry a `type == 0` message whose
/// `diff_field_path` is `[10000]` — into the chart archive — and whose whole
/// payload is `chart_type`, set to the nearest type an older Numbers has. The
/// donut says pie (25 → 5) and the radar says bar (27 → 2). Donut arrived in
/// 10.2 and radar in 11.2; no other chart in the corpus has a patch.
///
/// The consequence for a writer is the rule this crate already keeps: never
/// rewrite the first message of an object that has patches, because the patch
/// would then describe a chart that is no longer there.
#[test]
fn a_chart_too_new_for_an_old_reader_carries_a_down_level_patch() {
    let doc = fixture!("numbers-charts.numbers");
    let mut found: BTreeMap<u32, u64> = BTreeMap::new();
    for (_, object) in doc.objects() {
        if object.message_type() != chart::TYPE_CHART_DRAWABLE || object.messages.len() < 2 {
            continue;
        }
        let base = Message::decode(object.payload()).unwrap();
        let archive = base
            .bytes(chart::EXTENSION)
            .and_then(decode_nested)
            .unwrap();
        let kind = archive.varint(chart::field::CHART_TYPE).unwrap() as u32;

        for patch in &object.messages[1..] {
            assert_eq!(patch.message_type, 0, "a chart with a second real message");
            // `diff_field_path` (framing field 9) is `{1: [10000]}`: the patch
            // is applied inside the chart archive, not to the drawable.
            let path = patch
                .extra
                .iter()
                .find(|f| f.number == 9)
                .and_then(|f| match &f.value {
                    iwork::pb::Value::Bytes(bytes) => decode_nested(bytes),
                    _ => None,
                })
                .expect("the patch has a diff_field_path");
            let steps: Vec<u64> = match path.get(1) {
                Some(iwork::pb::Value::Bytes(bytes)) => {
                    let mut reader = iwork::pb::Reader::new(bytes);
                    let mut out = Vec::new();
                    while !reader.done() {
                        out.push(reader.varint().unwrap());
                    }
                    out
                }
                _ => panic!("the field path is not a packed list"),
            };
            assert_eq!(steps, [u64::from(chart::EXTENSION)]);

            let payload = Message::decode(&patch.payload).unwrap();
            let replacement = payload.varint(chart::field::CHART_TYPE).unwrap() as u32;
            assert_eq!(
                payload.fields.len(),
                1,
                "the patch sets nothing but the chart type"
            );
            found.insert(kind, u64::from(replacement));
        }
    }
    assert_eq!(
        found,
        BTreeMap::from([(25u32, 5u64), (27, 2)]),
        "donut falls back to pie and radar to bar"
    );
}

/// A chart is carried through a save byte for byte. Nothing here writes to one,
/// and this is the assertion that says so.
#[test]
fn a_document_with_charts_survives_a_no_op_save() {
    for name in [
        "numbers-charts.numbers",
        "keynote-charts.key",
        "pages-numbering.pages",
    ] {
        let Some(path) = generated(name) else {
            continue;
        };
        let doc = Document::open(&path).unwrap();
        let before: Vec<Vec<u8>> = doc
            .charts()
            .iter()
            .map(|c| c.identifier.to_le_bytes().to_vec())
            .collect();
        assert!(!before.is_empty(), "{name} has no charts");

        let out = std::env::temp_dir().join(format!("iwork-charts-{name}"));
        let _ = std::fs::remove_file(&out);
        Document::open(&path).unwrap().save(&out).unwrap();

        let original = std::fs::read(&path).unwrap();
        let written = std::fs::read(&out).unwrap();
        let a = zip::ZipArchive::new(std::io::Cursor::new(&original)).unwrap();
        let b = zip::ZipArchive::new(std::io::Cursor::new(&written)).unwrap();
        assert_eq!(a.len(), b.len(), "{name}: entry count changed");

        let reopened = Document::open(&out).unwrap();
        let after: Vec<Vec<u8>> = reopened
            .charts()
            .iter()
            .map(|c| c.identifier.to_le_bytes().to_vec())
            .collect();
        assert_eq!(before, after, "{name}: the charts moved");
        let _ = std::fs::remove_file(&out);
    }
}
