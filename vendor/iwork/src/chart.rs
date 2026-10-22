//! `TSCH` — charts, and the two different answers to "what data is this?".
//!
//! A chart on a page, a sheet or a slide is a `TSCH.ChartDrawableArchive`
//! (5021), and it is a drawable like any other: [`crate::drawable`] already
//! finds it, places it and reports its rectangle. What is unusual is where the
//! chart itself lives.
//!
//! **`TSCH.ChartArchive` has no type id.** It is not an object; it is proto2
//! extension field **10000** of the drawable archive, and the only way to know
//! that the bytes there are a chart is that the shell is a 5021. The ten style
//! shells 5022–5031 are the same sandwich — `{1: TSS.StyleArchive, 10000: the
//! properties}` — and the payload's schema is decided by the *shell's* type id,
//! not by anything in the bytes. A decoder driven purely by the registry sees
//! `{1: bytes, 10000: bytes}` and stops.
//!
//! **A chart carries its data twice, and the two copies are not the same
//! thing.**
//!
//! * The **private copy** is `ChartArchive.grid` (field 7), a
//!   `TSCH.ChartGridArchive` written *inline* — row names, column names, and a
//!   rectangular grid of `GridValue`s. Every chart in every app has one, and it
//!   is what the chart draws. A Keynote or Pages chart has nothing else.
//! * The **live references** are a Numbers-only second layer: a
//!   `TN.ChartMediatorArchive` (12006) holding `TSCE` formulas that point back
//!   into tables. The grid is then a cache of what those formulas last
//!   evaluated to, and the mediator is what makes the chart follow the table
//!   when a cell changes.
//!
//! [`Chart::grid`] is the first; [`Chart::references`] is the second, and
//! [`DataReferences`] says which tables and ranges feed the chart.
//!
//! **Every reference in a mediator goes through function 175.** Each formula
//! ends in a `FUNCTION_NODE` of index 175 over the reference nodes before it,
//! and 175 has no published name — it is one of the two holes in the function
//! table (the other is 337, the spill function, which Numbers itself prints as
//! `(null)`). It is not a one-argument wrapper: across the 65 chart mediators
//! in Apple's bundled templates its arity is nought, one or three, because a
//! series may be fed by several disjoint ranges — `175(B7, D7, F7)` — and a
//! label list may name nothing at all. Nothing here invents a name for it: the
//! node is dropped, its operands are printed, and the wrapper is reported as a
//! count.
//!
//! **What writes here, and what does not.** [`set_chart_data`] rewrites the
//! private grid and [`add_chart`] puts a copy of an existing chart somewhere
//! else with new numbers in it. Neither touches a chart that has a mediator:
//! that grid is a cache of a calculation this crate does not perform, and
//! numbers written into it disagree with the table the chart claims to follow
//! the moment Numbers recalculates. Everything else — the presets, the series
//! styles' three-level fallback, the `TSCH.Generated.*` properties — is read,
//! counted and passed through byte for byte, which is also why a new chart is
//! copied rather than invented.

use std::collections::BTreeMap;

use crate::drawable::{Frame, Placement};
use crate::formula::{Formula, Names, Site};
use crate::pb::{decode_nested, Message, Value};

/// `TSCH.ChartDrawableArchive` — the chart as it sits on the canvas.
pub const TYPE_CHART_DRAWABLE: u32 = 5021;
/// `TSCH.PreUFF.ChartInfoArchive` — the iWork '09/'13 chart root, still emitted
/// for imported documents. Absent from this corpus.
pub const TYPE_PREUFF_CHART_INFO: u32 = 5000;
/// `TSCH.PreUFF.ChartGridArchive` — the legacy standalone grid.
pub const TYPE_PREUFF_CHART_GRID: u32 = 5002;
/// `TSCH.ChartMediatorArchive` — the base mediator.
pub const TYPE_CHART_MEDIATOR: u32 = 5004;
/// `TSCH.ChartStylePreset`.
pub const TYPE_CHART_STYLE_PRESET: u32 = 5020;
/// `TSCH.ChartStyleArchive`.
pub const TYPE_CHART_STYLE: u32 = 5022;
/// `TSCH.ChartNonStyleArchive`.
pub const TYPE_CHART_NON_STYLE: u32 = 5023;
/// `TSCH.LegendStyleArchive`.
pub const TYPE_LEGEND_STYLE: u32 = 5024;
/// `TSCH.LegendNonStyleArchive`.
pub const TYPE_LEGEND_NON_STYLE: u32 = 5025;
/// `TSCH.ChartAxisStyleArchive`.
pub const TYPE_AXIS_STYLE: u32 = 5026;
/// `TSCH.ChartAxisNonStyleArchive`.
pub const TYPE_AXIS_NON_STYLE: u32 = 5027;
/// `TSCH.ChartSeriesStyleArchive`.
pub const TYPE_SERIES_STYLE: u32 = 5028;
/// `TSCH.ChartSeriesNonStyleArchive`.
pub const TYPE_SERIES_NON_STYLE: u32 = 5029;
/// `TSCH.ReferenceLineStyleArchive`.
pub const TYPE_REFERENCE_LINE_STYLE: u32 = 5030;
/// `TSCH.ReferenceLineNonStyleArchive`.
pub const TYPE_REFERENCE_LINE_NON_STYLE: u32 = 5031;
/// `TN.ChartMediatorArchive` — Numbers' subclass, the one that carries formulas.
pub const TYPE_TN_CHART_MEDIATOR: u32 = 12006;

/// The extension field the chart model hangs off its drawable, and the field
/// every style shell hangs its properties off.
pub const EXTENSION: u32 = 10000;

/// Field numbers of `TSCH.ChartArchive`, inside extension 10000.
pub mod field {
    pub const CHART_TYPE: u32 = 1;
    pub const SCATTER_FORMAT: u32 = 2;
    pub const LEGEND_FRAME: u32 = 3;
    pub const PRESET: u32 = 4;
    pub const SERIES_DIRECTION: u32 = 5;
    pub const CONTAINS_DEFAULT_DATA: u32 = 6;
    pub const GRID: u32 = 7;
    pub const MEDIATOR: u32 = 8;
    pub const CHART_STYLE: u32 = 9;
    pub const CHART_NON_STYLE: u32 = 10;
    pub const LEGEND_STYLE: u32 = 11;
    pub const LEGEND_NON_STYLE: u32 = 12;
    pub const VALUE_AXIS_STYLES: u32 = 13;
    pub const VALUE_AXIS_NON_STYLES: u32 = 14;
    pub const CATEGORY_AXIS_STYLES: u32 = 15;
    pub const CATEGORY_AXIS_NON_STYLES: u32 = 16;
    pub const SERIES_THEME_STYLES: u32 = 17;
    pub const SERIES_PRIVATE_STYLES: u32 = 18;
    pub const SERIES_NON_STYLES: u32 = 19;
    pub const PARAGRAPH_STYLES: u32 = 20;
    pub const MULTIDATASET_INDEX: u32 = 21;
    pub const DEFERRED_IMPORT_ACTION: u32 = 22;
    pub const OWNED_PRESET: u32 = 23;
    pub const IS_DIRTY: u32 = 24;
}

/// `TSCH.ChartType`, all twenty-eight values of the 15.3.1 enumeration.
///
/// Twenty-three of them are observed in this corpus: 1–19 and 22 come from the
/// Keynote zoo and Apple's Numbers templates, 20, 25 and 27 from
/// `21_Simple_Charts`, 11 from three templates. The five never seen are 0
/// (`undefinedChartType`, which is what an absent field means), 21, 23, 24 and
/// 26 — the multi-data bar, scatter and bubble charts and the 3-D donut.
pub fn chart_type_name(kind: u32) -> Option<&'static str> {
    Some(match kind {
        0 => "undefinedChartType",
        1 => "columnChartType2D",
        2 => "barChartType2D",
        3 => "lineChartType2D",
        4 => "areaChartType2D",
        5 => "pieChartType2D",
        6 => "stackedColumnChartType2D",
        7 => "stackedBarChartType2D",
        8 => "stackedAreaChartType2D",
        9 => "scatterChartType2D",
        10 => "mixedChartType2D",
        11 => "twoAxisChartType2D",
        12 => "columnChartType3D",
        13 => "barChartType3D",
        14 => "lineChartType3D",
        15 => "areaChartType3D",
        16 => "pieChartType3D",
        17 => "stackedColumnChartType3D",
        18 => "stackedBarChartType3D",
        19 => "stackedAreaChartType3D",
        20 => "multiDataColumnChartType2D",
        21 => "multiDataBarChartType2D",
        22 => "bubbleChartType2D",
        23 => "multiDataScatterChartType2D",
        24 => "multiDataBubbleChartType2D",
        25 => "donutChartType2D",
        26 => "donutChartType3D",
        27 => "radarChartType2D",
        _ => return None,
    })
}

/// A chart type in the words the app's own UI uses — "3D stacked column"
/// rather than `stackedColumnChartType3D`.
pub fn chart_type_label(kind: u32) -> String {
    let Some(name) = chart_type_name(kind) else {
        return format!("chart type {kind}");
    };
    let body = name
        .trim_end_matches("ChartType2D")
        .trim_end_matches("ChartType3D");
    let mut words = String::new();
    for (at, character) in body.char_indices() {
        if character.is_uppercase() && at != 0 {
            words.push(' ');
        }
        words.push(character.to_ascii_lowercase());
    }
    if name.ends_with("3D") {
        format!("3D {words}")
    } else if kind == 0 {
        "undefined".to_string()
    } else {
        words
    }
}

/// Whether the type is one of the eight 3-D families.
pub fn is_3d(kind: u32) -> bool {
    (12..=19).contains(&kind) || kind == 26
}

/// Whether the type is one of the four **interactive** families — the charts
/// Apple's UI creates from the "Interactive" tab, which step through data sets
/// with a slider or a pair of buttons.
pub fn is_interactive(kind: u32) -> bool {
    matches!(kind, 20 | 21 | 23 | 24)
}

/// `TSCH.SeriesDirection` — which axis of the grid is a series.
pub fn series_direction_name(direction: u32) -> &'static str {
    match direction {
        1 => "by row",
        2 => "by column",
        _ => "unknown",
    }
}

/// `TSCH.ScatterFormat`.
pub fn scatter_format_name(format: u32) -> &'static str {
    match format {
        1 => "separate X",
        2 => "shared X",
        _ => "unknown",
    }
}

/// One cell of a chart's private grid.
///
/// `TSCH.GridValue` is a union of four doubles decided by **which field is
/// present**, and a blank cell is a *present but zero-length* submessage. That
/// last case is the trap: reading "no fields set" as `0.0` fabricates a data
/// point, and the legacy `PreUFF` grid — a bare `repeated double` — cannot
/// express a blank at all, which is why old imported charts show spurious
/// zeroes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GridValue {
    /// A zero-length `GridValue`: the cell is empty.
    Empty,
    Number(f64),
    /// Seconds relative to the iWork epoch, 2001-01-01 UTC.
    Date(f64),
    /// The iWork-1.0 date slot (field 2). Modern writers use [`GridValue::Date`];
    /// a document written by Keynote 6 puts its dates here.
    LegacyDate(f64),
    /// Seconds.
    Duration(f64),
    /// Fields this crate does not know. Carried, never guessed at.
    Unknown,
}

impl GridValue {
    pub fn decode(message: &Message) -> GridValue {
        let double = |number: u32| match message.get(number) {
            Some(Value::Fixed64(bytes)) => Some(f64::from_le_bytes(*bytes)),
            _ => None,
        };
        if let Some(value) = double(1) {
            return GridValue::Number(value);
        }
        if let Some(value) = double(4) {
            return GridValue::Date(value);
        }
        if let Some(value) = double(3) {
            return GridValue::Duration(value);
        }
        if let Some(value) = double(2) {
            return GridValue::LegacyDate(value);
        }
        if message.fields.is_empty() {
            GridValue::Empty
        } else {
            GridValue::Unknown
        }
    }

    /// The number a numeric cell holds, and `None` for every other kind —
    /// including a blank, which is the distinction that matters.
    pub fn number(&self) -> Option<f64> {
        match self {
            GridValue::Number(value) => Some(*value),
            _ => None,
        }
    }

    pub fn to_text(&self) -> String {
        match self {
            GridValue::Empty => String::new(),
            GridValue::Number(value) => value.to_string(),
            GridValue::Date(value) => crate::table::format_date(*value),
            GridValue::LegacyDate(value) => {
                format!("{} (1.0 epoch)", crate::table::format_date(*value))
            }
            GridValue::Duration(value) => format!("{value}s"),
            GridValue::Unknown => "?".to_string(),
        }
    }
}

/// `TSCH.ChartGridArchive` — the chart's private copy of its data.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Grid {
    pub row_names: Vec<String>,
    pub column_names: Vec<String>,
    /// One entry per row, each as long as that row was written; a short row is
    /// left short rather than padded.
    pub rows: Vec<Vec<GridValue>>,
    /// `idMap.row_id_map` — a stable string id per row, and the index it is at.
    pub row_ids: Vec<(String, u32)>,
    pub column_ids: Vec<(String, u32)>,
}

impl Grid {
    pub fn decode(message: &Message) -> Grid {
        let strings = |number: u32| -> Vec<String> {
            message
                .all(number)
                .filter_map(|value| match value {
                    Value::Bytes(bytes) => Some(String::from_utf8_lossy(bytes).into_owned()),
                    _ => None,
                })
                .collect()
        };
        // **A blank cell is a present, zero-length submessage**, and
        // `decode_nested` refuses empty bytes — deliberately, because
        // everywhere else in this crate an empty length-delimited field is not
        // a message. Filtering those out here would not merely lose the blank:
        // it would shift every value after it one column to the left. So the
        // empty case is handled before the decode, on both levels.
        let cells = |row: &Message| -> Vec<GridValue> {
            row.all(1)
                .map(|value| match value {
                    Value::Bytes(bytes) if bytes.is_empty() => GridValue::Empty,
                    Value::Bytes(bytes) => match decode_nested(bytes) {
                        Some(cell) => GridValue::decode(&cell),
                        None => GridValue::Unknown,
                    },
                    _ => GridValue::Unknown,
                })
                .collect()
        };
        let rows = message
            .all(3)
            .map(|value| match value {
                Value::Bytes(bytes) if bytes.is_empty() => Vec::new(),
                Value::Bytes(bytes) => match decode_nested(bytes) {
                    Some(row) => cells(&row),
                    None => Vec::new(),
                },
                _ => Vec::new(),
            })
            .collect();
        let mut grid = Grid {
            row_names: strings(1),
            column_names: strings(2),
            rows,
            row_ids: Vec::new(),
            column_ids: Vec::new(),
        };
        if let Some(map) = message.bytes(4).and_then(decode_nested) {
            grid.row_ids = id_map(&map, 1);
            grid.column_ids = id_map(&map, 2);
        }
        grid
    }

    pub fn row_count(&self) -> usize {
        self.rows.len()
    }

    /// The widest row. A grid is rectangular in every document here, but a
    /// decoder that assumes it is will index out of a short one.
    pub fn column_count(&self) -> usize {
        self.rows.iter().map(Vec::len).max().unwrap_or(0)
    }

    pub fn value(&self, row: usize, column: usize) -> GridValue {
        self.rows
            .get(row)
            .and_then(|r| r.get(column))
            .copied()
            .unwrap_or(GridValue::Empty)
    }
}

fn id_map(map: &Message, number: u32) -> Vec<(String, u32)> {
    map.all(number)
        .filter_map(|value| match value {
            Value::Bytes(bytes) => decode_nested(bytes),
            _ => None,
        })
        .filter_map(|entry| {
            let id = match entry.get(1)? {
                Value::Bytes(bytes) => String::from_utf8_lossy(bytes).into_owned(),
                _ => return None,
            };
            Some((id, entry.varint(2)? as u32))
        })
        .collect()
}

/// One series, as the chart plots it: a name, and one value per category.
#[derive(Debug, Clone, PartialEq)]
pub struct Series {
    pub name: Option<String>,
    pub values: Vec<GridValue>,
}

/// `TSP.SparseReferenceArray` — a logical array in which only the entries that
/// differ from the default are written.
///
/// **`count` is the logical length, not `entries.len()`.** A series with no
/// private style simply has no entry, so sizing the series vector from the
/// entries drops every default-styled series and mis-aligns every index after
/// the first gap.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SparseReferences {
    pub count: u32,
    pub entries: Vec<(u32, u64)>,
}

impl SparseReferences {
    pub fn decode(message: &Message) -> SparseReferences {
        SparseReferences {
            count: message.varint(1).unwrap_or(0) as u32,
            entries: message
                .all(2)
                .filter_map(|value| match value {
                    Value::Bytes(bytes) => decode_nested(bytes),
                    _ => None,
                })
                .filter_map(|entry| {
                    let index = entry.varint(1).unwrap_or(0) as u32;
                    let target = entry
                        .bytes(2)
                        .and_then(decode_nested)
                        .and_then(|reference| reference.varint(1))?;
                    Some((index, target))
                })
                .collect(),
        }
    }

    /// The array laid out densely, `count` long, with `None` where the entry is
    /// absent and the default applies.
    pub fn dense(&self) -> Vec<Option<u64>> {
        let mut out = vec![None; self.count as usize];
        for (index, target) in &self.entries {
            if let Some(slot) = out.get_mut(*index as usize) {
                *slot = Some(*target);
            }
        }
        out
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0 && self.entries.is_empty()
    }
}

/// One formula out of a chart's mediator: where it points, and how it reads.
#[derive(Debug, Clone, PartialEq)]
pub struct ChartReference {
    /// The table the reference reaches, by name. An AST carries the table's
    /// `base_owner_uid`, so this is a lookup, not a guess — and it is `None`
    /// when the formula makes no reference at all, which happens: a bubble
    /// chart's row labels are string *literals*, not references.
    pub table: Option<String>,
    /// The reference as Numbers would spell it, printed against the table it
    /// reaches so that a reference inside that table reads `A2:A10` rather than
    /// naming the table on both sides of the colon. Where function 175 was
    /// given more than one operand they are joined with commas.
    pub text: String,
    /// One entry per operand of function 175 — usually one, sometimes three
    /// disjoint ranges, occasionally none at all.
    pub parts: Vec<String>,
    /// Whether the formula wrapped its operand in function 175.
    pub wrapped: bool,
}

impl ChartReference {
    /// `Table!reference`, the spelling this crate uses for "which cells".
    pub fn to_text(&self) -> String {
        match &self.table {
            Some(table) => format!("{table}!{}", self.text),
            None => self.text.clone(),
        }
    }
}

/// A chart's binding to the tables that feed it — the half of "what data is
/// this?" that the grid is only a cache of.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DataReferences {
    /// The `TN.ChartMediatorArchive` this came from.
    pub mediator: u64,
    /// `entity_id` — the mediator's identity to the calculation engine.
    pub entity_id: Option<String>,
    /// `columns_are_series` (field 4), when written.
    pub columns_are_series: Option<bool>,
    /// `TN.ChartMediatorFormulaStorage.direction` (field 5).
    pub direction: Option<i64>,
    /// One reference per series' data.
    pub data: Vec<ChartReference>,
    /// One per row label, in row order.
    pub row_labels: Vec<ChartReference>,
    /// One per column label, in column order.
    pub column_labels: Vec<ChartReference>,
    /// Error-bar formulas, counted rather than printed: four parallel lists,
    /// none of them exercised by this corpus.
    pub error_formulas: usize,
    /// `local_series_indexes` / `remote_series_indexes`, the parallel arrays of
    /// the base `TSCH.ChartMediatorArchive`.
    pub local_series_indexes: Vec<u64>,
    pub remote_series_indexes: Vec<u64>,
    /// How many of the formulas wrapped their operand in **function 175**, the
    /// unnamed function every chart reference in this corpus goes through.
    pub wrapped_in_175: usize,
    /// How many formulas did not have that shape.
    pub unwrapped: usize,
}

impl DataReferences {
    /// Every reference, in the order data, row labels, column labels.
    pub fn all(&self) -> impl Iterator<Item = &ChartReference> {
        self.data
            .iter()
            .chain(self.row_labels.iter())
            .chain(self.column_labels.iter())
    }

    /// The distinct tables that feed the chart, in the order they are first
    /// reached. A chart may take its series from more than one.
    pub fn tables(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for reference in self.all() {
            if let Some(table) = &reference.table {
                if !out.iter().any(|seen| seen == table) {
                    out.push(table.clone());
                }
            }
        }
        out
    }
}

/// A chart, as read.
#[derive(Debug, Clone)]
pub struct Chart {
    pub identifier: u64,
    pub stream: String,
    /// 5021 for a modern chart, 5000 for the pre-UFF one.
    pub message_type: u32,
    pub chart_type: u32,
    pub scatter_format: u32,
    pub series_direction: u32,
    pub contains_default_data: bool,
    pub is_dirty: bool,
    /// `multidataset_index` — which data set an interactive chart is showing.
    pub multidataset_index: Option<u32>,
    /// The legend's rectangle, in the chart's own coordinates.
    pub legend_frame: Option<Frame>,
    pub grid: Grid,
    pub mediator: Option<u64>,
    pub preset: Option<u64>,
    pub owned_preset: Option<u64>,
    pub chart_style: Option<u64>,
    pub chart_non_style: Option<u64>,
    pub legend_style: Option<u64>,
    pub legend_non_style: Option<u64>,
    pub value_axis_styles: Vec<u64>,
    pub value_axis_non_styles: Vec<u64>,
    pub category_axis_styles: Vec<u64>,
    pub category_axis_non_styles: Vec<u64>,
    pub series_theme_styles: Vec<u64>,
    /// Per-series overrides. Sparse: see [`SparseReferences`].
    pub series_private_styles: SparseReferences,
    pub series_non_styles: SparseReferences,
    pub paragraph_styles: Vec<u64>,
    /// Extension fields present on the chart archive, named where this crate
    /// knows the name. These are forward-compatibility tripwires — an older app
    /// that does not understand rounded corners refuses a document that claims
    /// them — so they are reported and never touched.
    pub extensions: Vec<(u32, &'static str)>,
    /// Where the chart sits, from the drawable chain.
    pub placement: Placement,
    /// The rectangle the app reports for the chart.
    pub frame: Frame,
    pub rotation: f32,
    pub parent: Option<u64>,
    /// The chart's title and caption storages, when it has them.
    pub title: Option<u64>,
    pub caption: Option<u64>,
    /// The live references, for a Numbers chart. `None` everywhere else: a
    /// Pages or Keynote chart has a private grid and nothing behind it.
    pub references: Option<DataReferences>,
}

impl Chart {
    /// How many series the chart plots, from the grid and the direction.
    pub fn series_count(&self) -> usize {
        match self.series_direction {
            2 => self.grid.column_count(),
            _ => self.grid.row_count(),
        }
    }

    /// The category labels — the names along the axis the series are *not*.
    pub fn categories(&self) -> &[String] {
        match self.series_direction {
            2 => &self.grid.row_names,
            _ => &self.grid.column_names,
        }
    }

    /// The chart's data as it is plotted: one [`Series`] per series, each with
    /// one value per category.
    ///
    /// `series_direction` is what decides which axis of the grid is which, and
    /// it is the only thing that does — the grid itself is always stored rows
    /// first.
    pub fn series(&self) -> Vec<Series> {
        let names = match self.series_direction {
            2 => &self.grid.column_names,
            _ => &self.grid.row_names,
        };
        (0..self.series_count())
            .map(|index| Series {
                name: names.get(index).cloned(),
                values: match self.series_direction {
                    2 => (0..self.grid.row_count())
                        .map(|row| self.grid.value(row, index))
                        .collect(),
                    _ => self.grid.rows.get(index).cloned().unwrap_or_default(),
                },
            })
            .collect()
    }

    pub fn type_name(&self) -> &'static str {
        chart_type_name(self.chart_type).unwrap_or("unknown chart type")
    }

    pub fn type_label(&self) -> String {
        chart_type_label(self.chart_type)
    }

    pub fn is_3d(&self) -> bool {
        is_3d(self.chart_type)
    }

    pub fn is_interactive(&self) -> bool {
        is_interactive(self.chart_type)
    }
}

/// Extension fields declared on `TSCH.ChartArchive`, with the message that
/// declares each. Everything here is a capability flag or an import carry-over;
/// none of it is decoded, and all of it is preserved.
const CHART_EXTENSIONS: &[(u32, &str)] = &[
    (10000, "preset_index_for_pasteboard"),
    (10001, "preset_uuid_for_pasteboard"),
    (10002, "scene3d_settings_constant_depth"),
    (10003, "custom_format_list_for_pasteboard"),
    (10004, "last_applied_fill_set_lookup_string"),
    (10005, "reference_lines"),
    (10010, "garlic_min_max_upgrade"),
    (10011, "garlic_label_format_upgrade"),
    (10021, "upgraded_to_ui_state"),
    (10023, "appearance_preserved_for_preset"),
    (10024, "supports_proportional_bended_callout_lines"),
    (10025, "deprecated_supports_rounded_corners"),
    (10026, "supports_rounded_corners"),
    (10027, "supports_series_value_label_spacing"),
    (10028, "supports_series_error_bar_spacing"),
    (10029, "supports_stacked_summary_labels"),
    (10030, "cached_data_formatter_persistable_style_objects"),
];

/// Every chart in the document.
///
/// The walk is over [`crate::Document::drawables`] rather than over the objects,
/// because a chart's placement, rectangle and rotation come from the drawable
/// chain and are worth having beside its data. A Numbers chart's references are
/// then resolved against the document's tables, which is what turns a reference
/// node into `Portfolio::Ticker`.
pub fn charts(document: &crate::Document) -> Vec<Chart> {
    let tables = document.tables();
    let names = crate::table::names(&tables);
    let mut objects: BTreeMap<u64, (u32, Message)> = BTreeMap::new();
    for (_, object) in document.objects() {
        if let Ok(message) = Message::decode(object.payload()) {
            objects.insert(object.identifier, (object.message_type(), message));
        }
    }

    let mut out = Vec::new();
    for drawable in document.drawables() {
        if drawable.message_type != TYPE_CHART_DRAWABLE {
            continue;
        }
        let Some((_, payload)) = objects.get(&drawable.identifier) else {
            continue;
        };
        let Some(archive) = payload.bytes(EXTENSION).and_then(decode_nested) else {
            continue;
        };
        out.push(decode(&drawable, &archive, &objects, &names));
    }
    out
}

fn decode(
    drawable: &crate::drawable::Drawable,
    archive: &Message,
    objects: &BTreeMap<u64, (u32, Message)>,
    names: &Names,
) -> Chart {
    let reference = |number: u32| -> Option<u64> {
        archive
            .bytes(number)
            .and_then(decode_nested)
            .and_then(|r| r.varint(1))
    };
    let references = |number: u32| -> Vec<u64> {
        archive
            .all(number)
            .filter_map(|value| match value {
                Value::Bytes(bytes) => decode_nested(bytes),
                _ => None,
            })
            .filter_map(|r| r.varint(1))
            .collect()
    };
    let sparse = |number: u32| -> SparseReferences {
        archive
            .bytes(number)
            .and_then(decode_nested)
            .map(|m| SparseReferences::decode(&m))
            .unwrap_or_default()
    };

    // The title and caption references live on the `TSD.DrawableArchive`
    // (fields 10 and 11), not on the `TSCH` archive read through `reference`
    // above. Reach the drawable archive through the path the drawable was found
    // by, and read them.
    let drawable_archive = objects.get(&drawable.identifier).and_then(|(_, payload)| {
        if drawable.path.is_empty() {
            Some(payload.clone())
        } else {
            crate::style::get_path(payload, &drawable.path).and_then(|value| match value {
                Value::Bytes(raw) => decode_nested(&raw),
                _ => None,
            })
        }
    });
    let drawable_ref = |number: u32| -> Option<u64> {
        drawable_archive
            .as_ref()
            .and_then(|d| d.bytes(number).and_then(decode_nested))
            .and_then(|r| r.varint(1))
    };

    let mediator = reference(field::MEDIATOR);
    Chart {
        identifier: drawable.identifier,
        stream: drawable.stream.clone(),
        message_type: drawable.message_type,
        chart_type: archive.varint(field::CHART_TYPE).unwrap_or(0) as u32,
        scatter_format: archive.varint(field::SCATTER_FORMAT).unwrap_or(0) as u32,
        series_direction: archive.varint(field::SERIES_DIRECTION).unwrap_or(0) as u32,
        contains_default_data: archive.varint(field::CONTAINS_DEFAULT_DATA).unwrap_or(0) != 0,
        is_dirty: archive.varint(field::IS_DIRTY).unwrap_or(0) != 0,
        multidataset_index: archive.varint(field::MULTIDATASET_INDEX).map(|v| v as u32),
        legend_frame: archive
            .bytes(field::LEGEND_FRAME)
            .and_then(decode_nested)
            .map(|rect| rect_frame(&rect)),
        grid: archive
            .bytes(field::GRID)
            .and_then(decode_nested)
            .map(|g| Grid::decode(&g))
            .unwrap_or_default(),
        mediator,
        preset: reference(field::PRESET),
        owned_preset: reference(field::OWNED_PRESET),
        chart_style: reference(field::CHART_STYLE),
        chart_non_style: reference(field::CHART_NON_STYLE),
        legend_style: reference(field::LEGEND_STYLE),
        legend_non_style: reference(field::LEGEND_NON_STYLE),
        value_axis_styles: references(field::VALUE_AXIS_STYLES),
        value_axis_non_styles: references(field::VALUE_AXIS_NON_STYLES),
        category_axis_styles: references(field::CATEGORY_AXIS_STYLES),
        category_axis_non_styles: references(field::CATEGORY_AXIS_NON_STYLES),
        series_theme_styles: references(field::SERIES_THEME_STYLES),
        series_private_styles: sparse(field::SERIES_PRIVATE_STYLES),
        series_non_styles: sparse(field::SERIES_NON_STYLES),
        paragraph_styles: references(field::PARAGRAPH_STYLES),
        extensions: CHART_EXTENSIONS
            .iter()
            .filter(|(number, _)| archive.get(*number).is_some())
            .map(|(number, name)| (*number, *name))
            .collect(),
        placement: drawable.placement.clone(),
        // The rectangle the app reports, with the rotated-bounding-box
        // correction `Drawable::frame` applies — the same the drawable reader
        // uses. It equals `base_rect` whenever the chart is unrotated, which is
        // every chart in the corpus, but a rotated one is now framed correctly.
        frame: drawable.frame(None),
        rotation: drawable.geometry.angle,
        parent: drawable.parent,
        title: drawable_ref(crate::drawable::field::TITLE),
        caption: drawable_ref(crate::drawable::field::CAPTION),
        references: mediator.and_then(|id| data_references(id, objects, names)),
    }
}

fn rect_frame(rect: &Message) -> Frame {
    let point = |number: u32| -> (f32, f32) {
        rect.bytes(number)
            .and_then(decode_nested)
            .map(|m| {
                let coordinate = |n: u32| match m.get(n) {
                    Some(Value::Fixed32(bytes)) => f32::from_le_bytes(*bytes),
                    _ => 0.0,
                };
                (coordinate(1), coordinate(2))
            })
            .unwrap_or((0.0, 0.0))
    };
    let (x, y) = point(1);
    let (width, height) = point(2);
    Frame {
        x,
        y,
        width,
        height,
    }
}

/// `TN.ChartMediatorArchive` (12006) — the chart's live binding to its tables.
///
/// The base `TSCH.ChartMediatorArchive` (field 1) carries only the two parallel
/// series-index arrays; everything that says *which cells* lives in Numbers'
/// subclass, in a `TN.ChartMediatorFormulaStorage` at field 3 with one
/// `TSCE.FormulaArchive` per series and per label.
fn data_references(
    mediator: u64,
    objects: &BTreeMap<u64, (u32, Message)>,
    names: &Names,
) -> Option<DataReferences> {
    let (message_type, archive) = objects.get(&mediator)?;
    if *message_type != TYPE_TN_CHART_MEDIATOR && *message_type != TYPE_CHART_MEDIATOR {
        return None;
    }
    let mut out = DataReferences {
        mediator,
        ..Default::default()
    };
    // The base class is field 1 of the Numbers subclass and the whole message
    // for the plain one.
    let base = archive.bytes(1).and_then(decode_nested);
    let base = match (*message_type, base) {
        (TYPE_TN_CHART_MEDIATOR, Some(base)) => base,
        _ => archive.clone(),
    };
    out.local_series_indexes = packed_varints(&base, 2);
    out.remote_series_indexes = packed_varints(&base, 3);
    if *message_type != TYPE_TN_CHART_MEDIATOR {
        return Some(out);
    }
    out.entity_id = match archive.get(2) {
        Some(Value::Bytes(bytes)) => Some(String::from_utf8_lossy(bytes).into_owned()),
        _ => None,
    };
    out.columns_are_series = archive.varint(4).map(|v| v != 0);
    let Some(storage) = archive.bytes(3).and_then(decode_nested) else {
        return Some(out);
    };
    out.direction = storage.varint(5).map(|v| v as i64);
    let printed = |number: u32, out: &mut DataReferences| -> Vec<ChartReference> {
        storage
            .all(number)
            .filter_map(|value| match value {
                Value::Bytes(bytes) => decode_nested(bytes),
                _ => None,
            })
            .map(|archive| {
                let Some(formula) = Formula::decode(&archive) else {
                    out.unwrapped += 1;
                    return ChartReference {
                        table: None,
                        text: "#FORMULA!".to_string(),
                        parts: Vec::new(),
                        wrapped: false,
                    };
                };
                let (ast, wrapped) = match unwrap_175(&formula) {
                    Some(inner) => {
                        out.wrapped_in_175 += 1;
                        (inner, true)
                    }
                    None => {
                        out.unwrapped += 1;
                        (formula.ast.clone(), false)
                    }
                };
                // The AST names its table by `base_owner_uid`, so printing
                // against *that* table is what makes `A2:A10` read as a range
                // rather than as two fully-qualified cells.
                let target = ast
                    .nodes
                    .iter()
                    .find_map(|node| node.reference().and_then(|r| r.table))
                    .and_then(|uid| names.by_uid(uid));
                // One operand per value left on the stack. A series fed by
                // three separate columns reads `B7, D7, F7`; one fed by nothing
                // leaves the stack empty.
                let parts = ast.texts(Site::new(names, target, (0, 0)));
                ChartReference {
                    table: target.map(|index| names.tables[index].name.clone()),
                    text: if parts.is_empty() {
                        "(nothing)".to_string()
                    } else {
                        parts.join(", ")
                    },
                    parts,
                    wrapped,
                }
            })
            .collect()
    };
    let data = printed(1, &mut out);
    let rows = printed(3, &mut out);
    let columns = printed(4, &mut out);
    out.data = data;
    out.row_labels = rows;
    out.column_labels = columns;
    out.error_formulas = (6..=9).map(|number| storage.all(number).count()).sum();
    Some(out)
}

/// **Function 175.** Every reference a chart mediator holds is a formula whose
/// last node is a `FUNCTION_NODE` of index 175, and 175 has no published name —
/// Apple's function table has a hole there, as it does at 337.
///
/// It is **not** always a wrapper around one operand. Across the 65 chart
/// mediators in Apple's bundled templates it appears with an arity of nought,
/// one and three: a series fed by three disjoint columns is `175(B7, D7, F7)`,
/// and a label list that names nothing is `175()`. So this drops the node and
/// hands back the fragment, which leaves one value on the stack per operand
/// rather than the single value a whole formula leaves.
///
/// Rather than print `(null)(Sales::A)` — which is what the formula printer
/// makes of an unnamed function, and what Numbers itself prints for 337 — the
/// operands are printed and the wrapper is reported as a count.
pub const CHART_REFERENCE_FUNCTION: u32 = 175;

fn unwrap_175(formula: &Formula) -> Option<crate::formula::Ast> {
    let last = formula.ast.nodes.last()?;
    let (index, _) = last.function()?;
    if index != CHART_REFERENCE_FUNCTION {
        return None;
    }
    Some(crate::formula::Ast {
        nodes: formula.ast.nodes[..formula.ast.nodes.len() - 1].to_vec(),
    })
}

/// A repeated scalar that may arrive packed or unpacked. proto2 does not mark
/// these `[packed=true]`, so Apple writes them one field per value — but a
/// reader has to accept both, and `local_series_indexes` is exactly the kind of
/// field a future writer would pack.
fn packed_varints(message: &Message, number: u32) -> Vec<u64> {
    let mut out = Vec::new();
    for value in message.all(number) {
        match value {
            Value::Varint(v) => out.push(*v),
            Value::Bytes(bytes) => {
                let mut reader = crate::pb::Reader::new(bytes);
                while !reader.done() {
                    match reader.varint() {
                        Ok(v) => out.push(v),
                        Err(_) => break,
                    }
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_blank_grid_cell_holds_its_place() {
        // One row of three cells, the middle one blank: `0x0A 0x00` is a
        // present, *zero-length* GridValue. Dropping it would not lose a blank,
        // it would move the third value into the second column.
        let cell = |value: f64| {
            let mut message = Message::default();
            message.set(1, Value::Fixed64(value.to_le_bytes()));
            message.encode()
        };
        let mut row = Message::default();
        row.append_in_order(1, Value::Bytes(cell(1.0)));
        row.append_in_order(1, Value::Bytes(Vec::new()));
        row.append_in_order(1, Value::Bytes(cell(3.0)));
        let mut grid = Message::default();
        grid.append_in_order(3, Value::Bytes(row.encode()));

        let grid = Grid::decode(&grid);
        assert_eq!(grid.column_count(), 3);
        assert_eq!(grid.value(0, 0), GridValue::Number(1.0));
        assert_eq!(grid.value(0, 1), GridValue::Empty);
        assert_eq!(grid.value(0, 1).number(), None);
        assert_eq!(grid.value(0, 2), GridValue::Number(3.0));
    }

    #[test]
    fn the_four_grid_slots_are_four_kinds() {
        let of = |number: u32, value: f64| {
            let mut message = Message::default();
            message.set(number, Value::Fixed64(value.to_le_bytes()));
            GridValue::decode(&message)
        };
        assert_eq!(of(1, 42.0), GridValue::Number(42.0));
        assert_eq!(of(2, 42.0), GridValue::LegacyDate(42.0));
        assert_eq!(of(3, 42.0), GridValue::Duration(42.0));
        assert_eq!(of(4, 42.0), GridValue::Date(42.0));
    }

    #[test]
    fn a_sparse_array_is_as_long_as_its_count_says() {
        // count 6, one entry at index 4: four defaults, an override, a default.
        let mut entry = Message::default();
        entry.set(1, Value::Varint(4));
        let mut target = Message::default();
        target.set(1, Value::Varint(99));
        entry.set(2, Value::Bytes(target.encode()));
        let mut array = Message::default();
        array.set(1, Value::Varint(6));
        array.set(2, Value::Bytes(entry.encode()));

        let sparse = SparseReferences::decode(&array);
        assert_eq!(sparse.count, 6);
        assert_eq!(sparse.entries.len(), 1);
        assert_eq!(sparse.dense().len(), 6);
        assert_eq!(sparse.dense()[4], Some(99));
        assert_eq!(sparse.dense()[5], None);
    }

    #[test]
    fn the_type_names_are_the_enum_and_the_labels_are_the_ui() {
        assert_eq!(chart_type_name(1), Some("columnChartType2D"));
        assert_eq!(chart_type_name(27), Some("radarChartType2D"));
        assert_eq!(chart_type_name(28), None);
        assert_eq!(chart_type_label(17), "3D stacked column");
        assert_eq!(chart_type_label(25), "donut");
        assert!(is_3d(16) && !is_3d(5));
        assert!(is_interactive(20) && !is_interactive(1));
    }

    #[test]
    fn a_grid_is_read_rows_first_and_direction_decides_the_series() {
        let grid = Grid {
            row_names: vec!["A".into(), "B".into()],
            column_names: vec!["x".into(), "y".into(), "z".into()],
            rows: vec![
                vec![
                    GridValue::Number(1.0),
                    GridValue::Number(2.0),
                    GridValue::Number(3.0),
                ],
                vec![
                    GridValue::Number(11.0),
                    GridValue::Number(12.0),
                    GridValue::Number(13.0),
                ],
            ],
            ..Grid::default()
        };
        let chart = |direction: u32| Chart {
            identifier: 0,
            stream: String::new(),
            message_type: TYPE_CHART_DRAWABLE,
            chart_type: 1,
            scatter_format: 0,
            series_direction: direction,
            contains_default_data: false,
            is_dirty: false,
            multidataset_index: None,
            legend_frame: None,
            grid: grid.clone(),
            mediator: None,
            preset: None,
            owned_preset: None,
            chart_style: None,
            chart_non_style: None,
            legend_style: None,
            legend_non_style: None,
            value_axis_styles: Vec::new(),
            value_axis_non_styles: Vec::new(),
            category_axis_styles: Vec::new(),
            category_axis_non_styles: Vec::new(),
            series_theme_styles: Vec::new(),
            series_private_styles: SparseReferences::default(),
            series_non_styles: SparseReferences::default(),
            paragraph_styles: Vec::new(),
            extensions: Vec::new(),
            placement: Placement::Unknown,
            frame: Frame {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
            },
            rotation: 0.0,
            parent: None,
            title: None,
            caption: None,
            references: None,
        };

        let by_row = chart(1);
        assert_eq!(by_row.series_count(), 2);
        assert_eq!(by_row.categories(), ["x", "y", "z"]);
        assert_eq!(by_row.series()[1].name.as_deref(), Some("B"));
        assert_eq!(by_row.series()[1].values[2].number(), Some(13.0));

        let by_column = chart(2);
        assert_eq!(by_column.series_count(), 3);
        assert_eq!(by_column.categories(), ["A", "B"]);
        assert_eq!(by_column.series()[2].name.as_deref(), Some("z"));
        assert_eq!(by_column.series()[2].values[0].number(), Some(3.0));
        assert_eq!(by_column.series()[2].values[1].number(), Some(13.0));
    }
}

// -- writing the private grid -------------------------------------------------

/// What a chart is to be made to draw: the same three lists [`Grid`] reads.
///
/// The row and column *ids* are not here. They are stable identities the app
/// keeps across an edit, so [`set_chart_data`] carries over the ones the chart
/// already has and mints a fresh UUID only for a row or column that did not
/// exist before — which is what makes a rewritten chart the same chart rather
/// than a new one that happens to sit in the same place.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ChartData {
    pub row_names: Vec<String>,
    pub column_names: Vec<String>,
    /// One row per entry, each as long as the row is. A short row is left
    /// short, the way the archive writes one.
    pub rows: Vec<Vec<GridValue>>,
}

impl ChartData {
    /// The common case: a rectangle of numbers, with names down the side and
    /// along the top.
    pub fn numbers(row_names: &[&str], column_names: &[&str], rows: &[&[f64]]) -> ChartData {
        ChartData {
            row_names: row_names.iter().map(|s| (*s).to_string()).collect(),
            column_names: column_names.iter().map(|s| (*s).to_string()).collect(),
            rows: rows
                .iter()
                .map(|row| row.iter().copied().map(GridValue::Number).collect())
                .collect(),
        }
    }
}

/// Rewrite the private grid a chart draws.
///
/// This is the copy every chart has and a Pages or Keynote chart's only one.
/// A Numbers chart fed by a table has a second layer — a
/// `TN.ChartMediatorArchive` of `TSCE` formulas — and its grid is a *cache* of
/// what those last evaluated to; writing numbers into that cache makes the
/// chart disagree with the table it claims to follow the moment Numbers
/// recalculates. So a chart with a mediator is refused by name.
///
/// What is written is the grid and nothing else: the chart keeps its type, its
/// styles, its legend and its rectangle. A row beyond the ones the chart had
/// takes its colour from the theme's series palette, which is a fixed six and
/// cycles — the same thing the app does when a chart is given a seventh series.
pub fn set_chart_data(
    document: &mut crate::Document,
    chart: u64,
    data: &ChartData,
) -> Result<(), crate::Error> {
    if data.rows.len() != data.row_names.len() {
        return Err(crate::Error::Format(format!(
            "the chart has {} row(s) of values and {} row name(s), and the app draws them side \
             by side",
            data.rows.len(),
            data.row_names.len()
        )));
    }
    let widest = data.rows.iter().map(Vec::len).max().unwrap_or(0);
    if widest > data.column_names.len() {
        return Err(crate::Error::Format(format!(
            "a row of the chart has {widest} value(s) and there are {} column name(s) to put \
             them under",
            data.column_names.len()
        )));
    }

    let existing = charts(document)
        .into_iter()
        .find(|c| c.identifier == chart)
        .ok_or_else(|| crate::Error::Format(format!("no chart {chart}")))?;
    if let Some(mediator) = existing.mediator {
        return Err(crate::Error::Format(format!(
            "chart {chart} is fed by a table through mediator {mediator}, and its grid is a \
             cache of what those formulas last evaluated to — writing numbers into it would \
             make the chart disagree with the table the moment Numbers recalculates"
        )));
    }

    // The ids the chart already has, by index, so a row that stays keeps its
    // identity and only a new one is minted.
    let carry = |had: &[(String, u32)], wanted: usize| -> Vec<(String, u32)> {
        (0..wanted)
            .map(|index| {
                let id = had
                    .iter()
                    .find(|(_, at)| *at as usize == index)
                    .map(|(id, _)| id.clone())
                    .unwrap_or_else(crate::metadata::uuid);
                (id, index as u32)
            })
            .collect()
    };
    let row_ids = carry(&existing.grid.row_ids, data.rows.len());
    let column_ids = carry(&existing.grid.column_ids, data.column_names.len());

    let mut archive = document.archive(chart)?;
    let mut model = archive
        .bytes(EXTENSION)
        .and_then(decode_nested)
        .ok_or_else(|| {
            crate::Error::Format(format!(
                "chart {chart} carries no chart archive at extension {EXTENSION}"
            ))
        })?;
    model.set_in_order(
        7,
        Value::Bytes(encode_grid(data, &row_ids, &column_ids).encode()),
    );
    // Whatever the chart had before, it is not the app's placeholder data now.
    if model.get(6).is_some() {
        model.set_in_order(6, Value::Varint(0));
    }
    archive.set_in_order(EXTENSION, Value::Bytes(model.encode()));
    document.set_archive_of(chart, &archive)?;
    Ok(())
}

/// `TSCH.ChartGridArchive`, in the shape the apps write one: names, rows of
/// `GridValue`, and the id map.
fn encode_grid(
    data: &ChartData,
    row_ids: &[(String, u32)],
    column_ids: &[(String, u32)],
) -> Message {
    let mut fields = Vec::new();
    for name in &data.row_names {
        fields.push(crate::pb::Field {
            number: 1,
            value: Value::Bytes(name.as_bytes().to_vec()),
        });
    }
    for name in &data.column_names {
        fields.push(crate::pb::Field {
            number: 2,
            value: Value::Bytes(name.as_bytes().to_vec()),
        });
    }
    for row in &data.rows {
        let mut cells = Message::default();
        for value in row {
            cells.fields.push(crate::pb::Field {
                number: 1,
                // A blank cell is a *present, zero-length* submessage: leaving
                // it out shifts every value after it one column left.
                value: Value::Bytes(encode_value(*value)),
            });
        }
        fields.push(crate::pb::Field {
            number: 3,
            value: Value::Bytes(cells.encode()),
        });
    }
    let entries = |number: u32, ids: &[(String, u32)]| -> Vec<crate::pb::Field> {
        ids.iter()
            .map(|(id, index)| {
                let mut entry = Message::default();
                entry.fields.push(crate::pb::Field {
                    number: 1,
                    value: Value::Bytes(id.as_bytes().to_vec()),
                });
                entry.fields.push(crate::pb::Field {
                    number: 2,
                    value: Value::Varint(u64::from(*index)),
                });
                crate::pb::Field {
                    number,
                    value: Value::Bytes(entry.encode()),
                }
            })
            .collect()
    };
    let mut map = Message::default();
    map.fields.extend(entries(1, row_ids));
    map.fields.extend(entries(2, column_ids));
    fields.push(crate::pb::Field {
        number: 4,
        value: Value::Bytes(map.encode()),
    });
    Message { fields }
}

/// One `TSCH.GridValue`: which field is set decides what it is, and an empty
/// message is an empty cell.
fn encode_value(value: GridValue) -> Vec<u8> {
    let number = match value {
        GridValue::Empty | GridValue::Unknown => return Vec::new(),
        GridValue::Number(_) => 1,
        GridValue::LegacyDate(_) => 2,
        GridValue::Duration(_) => 3,
        GridValue::Date(_) => 4,
    };
    let inner = match value {
        GridValue::Number(v)
        | GridValue::LegacyDate(v)
        | GridValue::Duration(v)
        | GridValue::Date(v) => v,
        GridValue::Empty | GridValue::Unknown => 0.0,
    };
    Message {
        fields: vec![crate::pb::Field {
            number,
            value: Value::Fixed64(inner.to_le_bytes()),
        }],
    }
    .encode()
}

/// The archives a chart owns rather than shares: one per style archive, and
/// the half of each pair that carries this chart's own state.
const NON_STYLES: [u32; 6] = [
    TYPE_CHART_NON_STYLE,
    TYPE_LEGEND_NON_STYLE,
    TYPE_AXIS_NON_STYLE,
    TYPE_SERIES_NON_STYLE,
    TYPE_REFERENCE_LINE_NON_STYLE,
    TYPE_CHART_STYLE_PRESET,
];

/// Put a copy of a chart the document already has on a slide, a sheet or a
/// page, drawing the data given.
///
/// **A chart is copied, not invented.** Its model points at a preset, a chart
/// style and non-style, a legend style and non-style, styles and non-styles for
/// both axes, six theme series styles and a list of paragraph styles — a dozen
/// objects whose properties are a theme's, and none of which this crate can
/// make up honestly. So the caller names a chart to copy, and gets one that
/// looks like it with different numbers in it.
///
/// What is *shared* and what is *copied* follows the archive's own division:
/// the theme's styles are shared, exactly as two charts made from one preset
/// share them in the app, and everything the source chart keeps in its **own
/// component** — its non-styles, which carry per-chart state like an axis
/// title — is copied and renumbered, so editing one chart cannot change the
/// other.
pub fn add_chart(
    document: &mut crate::Document,
    container: &str,
    from: u64,
    data: &ChartData,
    position: (f32, f32),
    size: (f32, f32),
) -> Result<u64, crate::Error> {
    use std::collections::{BTreeMap, BTreeSet};

    if size.0 <= 0.0 || size.1 <= 0.0 {
        return Err(crate::Error::Format(format!(
            "a chart needs a size, and {} × {} is not one",
            size.0, size.1
        )));
    }
    let source = charts(document)
        .into_iter()
        .find(|c| c.identifier == from)
        .ok_or_else(|| {
            crate::Error::Format(format!(
                "no chart {from} to copy — `iwork charts` lists the ones there are"
            ))
        })?;
    if source.mediator.is_some() {
        return Err(crate::Error::Format(format!(
            "chart {from} is fed by a table through its mediator, and a copy of it would claim \
             to follow that table while holding numbers of its own"
        )));
    }
    let container = crate::drawable::container_of(document, container)?;

    // What the chart keeps to itself, and what it shares with every other
    // chart made from the same preset. The archive draws the line itself: a
    // *style* is the theme's and a *non-style* is this chart's, which is what
    // the pairing of 5022/5023, 5024/5025, 5026/5027, 5028/5029 and 5030/5031
    // is for. Location will not do it — in a Keynote deck the non-styles sit
    // in the slide's own stream, in Pages they sit in `ObjectContainer`.
    let mut private: BTreeSet<u64> = source
        .chart_non_style
        .into_iter()
        .chain(source.legend_non_style)
        .chain(source.owned_preset)
        .chain(source.value_axis_non_styles.iter().copied())
        .chain(source.category_axis_non_styles.iter().copied())
        .chain(source.series_non_styles.entries.iter().map(|(_, id)| *id))
        .filter(|id| *id != 0)
        .collect();
    // …and whatever those keep to themselves in turn, which is again decided
    // by type rather than by where it happens to live.
    let mut frontier: Vec<u64> = private.iter().copied().collect();
    while let Some(identifier) = frontier.pop() {
        let Ok(archive) = document.archive(identifier) else {
            continue;
        };
        for target in crate::style::references(&archive) {
            if target == 0 || private.contains(&target) || target == from {
                continue;
            }
            let Some((_, object)) = document.object(target) else {
                continue;
            };
            if !NON_STYLES.contains(&object.message_type()) {
                continue;
            }
            private.insert(target);
            frontier.push(target);
        }
    }
    private.remove(&from);

    // Every payload read out before anything is allocated, so the copy is
    // assembled from a document nobody is halfway through changing.
    let mut sources: Vec<(u64, u32, Message)> = Vec::new();
    for identifier in std::iter::once(from).chain(private.iter().copied()) {
        let (_, object) = document
            .object(identifier)
            .ok_or(crate::Error::NoSuchObject(identifier))?;
        let archive = Message::decode(object.payload())
            .map_err(|e| crate::Error::Format(format!("object {identifier}: {e}")))?;
        sources.push((identifier, object.message_type(), archive));
    }

    let mut grow = crate::create::Grow::new(document);
    let chart = grow.allocate();
    let mut map: BTreeMap<u64, u64> = BTreeMap::new();
    map.insert(from, chart);
    for identifier in &private {
        map.insert(*identifier, grow.allocate());
    }

    let neighbour = container.neighbour();
    for (old, message_type, archive) in &mut sources {
        let (old, message_type) = (*old, *message_type);
        let new = map[&old];
        crate::keynote::remap_references(archive, &map, crate::keynote::MAX_DEPTH);
        if old == from {
            place(archive, container.parent(), position, size);
            let mut model = archive
                .bytes(EXTENSION)
                .and_then(decode_nested)
                .ok_or_else(|| {
                    crate::Error::Format(format!("chart {from} carries no chart archive"))
                })?;
            let ids = |count: usize| -> Vec<(String, u32)> {
                (0..count)
                    .map(|index| (crate::metadata::uuid(), index as u32))
                    .collect()
            };
            model.set_in_order(
                7,
                Value::Bytes(
                    encode_grid(data, &ids(data.rows.len()), &ids(data.column_names.len()))
                        .encode(),
                ),
            );
            if model.get(6).is_some() {
                model.set_in_order(6, Value::Varint(0));
            }
            archive.set_in_order(EXTENSION, Value::Bytes(model.encode()));
        }
        grow.beside(neighbour, new, message_type, archive)?;
    }
    grow.finish()?;

    crate::drawable::hold(document, &container, chart)?;
    document.declare_external_references();
    Ok(chart)
}

/// Put a copied drawable where it was asked for: its rectangle, and the parent
/// its new container does or does not give it.
///
/// The geometry is `super.super.geometry` for a chart drawable, which is one
/// level of nesting — but nothing here counts levels: it walks field 1 the way
/// [`crate::drawable::drawable_path`] does, because that is the only thing that
/// holds for every archive in `TSD`.
fn place(archive: &mut Message, parent: Option<u64>, position: (f32, f32), size: (f32, f32)) {
    let Some(path) = crate::drawable::drawable_path(archive) else {
        return;
    };
    let mut body = archive.clone();
    for step in &path {
        body = match body.bytes(*step).and_then(decode_nested) {
            Some(next) => next,
            None => return,
        };
    }
    let point = |x: f32, y: f32| {
        Value::Bytes(
            Message {
                fields: vec![
                    crate::pb::Field {
                        number: 1,
                        value: Value::Fixed32(x.to_le_bytes()),
                    },
                    crate::pb::Field {
                        number: 2,
                        value: Value::Fixed32(y.to_le_bytes()),
                    },
                ],
            }
            .encode(),
        )
    };
    let mut geometry = body
        .bytes(crate::drawable::field::GEOMETRY)
        .and_then(decode_nested)
        .unwrap_or_default();
    geometry.set_in_order(1, point(position.0, position.1));
    geometry.set_in_order(2, point(size.0, size.1));
    body.set_in_order(
        crate::drawable::field::GEOMETRY,
        Value::Bytes(geometry.encode()),
    );
    match parent {
        Some(parent) => body.set_in_order(
            crate::drawable::field::PARENT,
            Value::Bytes(crate::create::reference_bytes(parent)),
        ),
        None => body
            .fields
            .retain(|field| field.number != crate::drawable::field::PARENT),
    }
    // Back up the path, rebuilding each level around the one below it.
    let mut payload = body.encode();
    for step in path.iter().rev() {
        let mut level = archive.clone();
        for above in &path[..path.iter().position(|s| s == step).unwrap_or(0)] {
            level = match level.bytes(*above).and_then(decode_nested) {
                Some(next) => next,
                None => return,
            };
        }
        level.set_in_order(*step, Value::Bytes(payload));
        payload = level.encode();
    }
    if let Ok(rebuilt) = Message::decode(&payload) {
        *archive = rebuilt;
    }
}

// -- binding a chart to a table ----------------------------------------------

/// Which cells of which table feed a chart — see
/// [`crate::Document::bind_chart`].
///
/// One range per series, and the label ranges beside them. Named rather than
/// positional, because a chart binding is four lists and an order nobody
/// remembers.
#[derive(Debug, Clone, Default)]
pub struct ChartBinding {
    /// One range per series, in the order the chart draws them.
    pub series: Vec<crate::table::CellRange>,
    /// The cells the row labels are read from, one per row.
    pub row_labels: Vec<crate::table::CellRange>,
    /// …and the column labels.
    pub column_labels: Vec<crate::table::CellRange>,
    /// `direction`: series by row (1) or by column (2), as the mediator stores
    /// it. Every mediator in the corpus carries one or the other.
    pub series_by_row: bool,
}

/// `TN.ChartMediatorFormulaStorage` field numbers.
mod mediator_field {
    /// One `TSCE.FormulaArchive` per series.
    pub const DATA: u32 = 1;
    pub const ROW_LABELS: u32 = 3;
    pub const COLUMN_LABELS: u32 = 4;
    pub const DIRECTION: u32 = 5;
}

/// The function every mediator reference is wrapped in. Apple publishes no name
/// for it; see the module comment.
pub const MEDIATOR_FUNCTION: u64 = 175;

/// Make a chart follow a table.
///
/// This is the second half of a Numbers chart, and the half that makes it a
/// *chart of a table* rather than a picture of some numbers: the grid it draws
/// stays what it was — a cache — and the mediator says where those numbers came
/// from, so the app recalculates them when the table changes.
///
/// What is written, copied from the twelve mediators of
/// `numbers-charts.numbers`:
///
/// * a `TN.ChartMediatorArchive` (12006) in the calculation engine's stream,
///   holding `{2: -1, 3: series count}`, a fresh **entity id** as a UUID
///   string, and one `TSCE.FormulaArchive` per series and per label — each of
///   them a reference wrapped in **function 175**;
/// * a `FormulaOwnerDependenciesArchive` of **`owner_kind` 2**, whose
///   `formula_owner_uid` is that entity id read as a UUID and whose field 11
///   points at the chart — one per mediator, exactly as the corpus has twelve
///   of each;
/// * both registered in the engine's dependency tracker, the way every owner
///   is;
/// * and `ChartArchive.mediator` (field 8) pointing at the mediator.
///
/// Refused by name: a chart that already has a mediator, a range outside the
/// table, a binding with no series, and a table whose identity the engine does
/// not know.
pub fn bind_chart(
    document: &mut crate::Document,
    chart: u64,
    table: &crate::table::Table,
    binding: &ChartBinding,
) -> Result<u64, crate::Error> {
    use crate::{Error, Refusal};

    let existing = charts(document)
        .into_iter()
        .find(|c| c.identifier == chart)
        .ok_or_else(|| Error::refused(Refusal::NotFound, format!("no chart {chart}")))?;
    if let Some(mediator) = existing.mediator {
        return Err(Error::refused(
            Refusal::Missing,
            format!(
                "chart {chart} already follows a table through mediator {mediator}, and \
                 pointing it at another one means taking the first out of the calculation \
                 engine"
            ),
        ));
    }
    if binding.series.is_empty() {
        return Err(Error::refused(
            Refusal::NotACell,
            "a chart follows at least one series".to_string(),
        ));
    }
    if table.base_uid == crate::table::Uuid::default() {
        return Err(Error::refused(
            Refusal::Missing,
            format!(
                "table {} has no base_owner_uid, so a formula cannot name it — a table this \
                 crate made carries one",
                table.name
            ),
        ));
    }
    for range in binding
        .series
        .iter()
        .chain(&binding.row_labels)
        .chain(&binding.column_labels)
    {
        let rectangle = range.resolve()?;
        if rectangle.bottom_right.0 >= table.rows || rectangle.bottom_right.1 >= table.columns {
            return Err(Error::refused(
                Refusal::OutOfBounds,
                format!(
                    "{range} is outside {}, which is {}×{}",
                    table.name, table.rows, table.columns
                ),
            ));
        }
    }

    // The mediator's identity, in the two forms the format keeps it in: the
    // string the archive carries, and the same sixteen bytes as the UUID its
    // owner is keyed by. They are byte-reversed halves of one another —
    // `E357DB84-…-CFE8` is owner uid `…CFE8` / `E357DB84…` read the other way
    // round, which is how the corpus's twelve pairs line up.
    let entity = crate::metadata::uuid();
    let uid = uuid_from_entity(&entity)
        .ok_or_else(|| Error::Format(format!("{entity} is not a UUID this crate can read back")))?;

    let mut storage = crate::pb::Message::default();
    for range in &binding.series {
        storage.append_in_order(
            mediator_field::DATA,
            Value::Bytes(reference_formula(table, range)?),
        );
    }
    for range in &binding.row_labels {
        storage.append_in_order(
            mediator_field::ROW_LABELS,
            Value::Bytes(reference_formula(table, range)?),
        );
    }
    for range in &binding.column_labels {
        storage.append_in_order(
            mediator_field::COLUMN_LABELS,
            Value::Bytes(reference_formula(table, range)?),
        );
    }
    storage.set_in_order(
        mediator_field::DIRECTION,
        Value::Varint(if binding.series_by_row { 1 } else { 2 }),
    );

    let mut head = crate::pb::Message::default();
    // `local_series_indexes` is -1 on every mediator in the corpus, and
    // `remote_series_indexes` is the number of series.
    head.set_in_order(2, Value::Varint(u64::from(u32::MAX)));
    head.set_in_order(3, Value::Varint(binding.series.len() as u64));

    let mut archive = crate::pb::Message::default();
    archive.set_in_order(1, Value::Bytes(head.encode()));
    archive.set_in_order(2, Value::Bytes(entity.as_bytes().to_vec()));
    archive.set_in_order(3, Value::Bytes(storage.encode()));

    let mediator = crate::calc::add_chart_mediator(document, chart, &archive, uid)?;

    // The chart's half of the link.
    let mut chart_archive = document.archive(chart)?;
    let mut model = chart_archive
        .bytes(EXTENSION)
        .and_then(decode_nested)
        .ok_or_else(|| {
            Error::Format(format!(
                "chart {chart} carries no chart archive at extension {EXTENSION}"
            ))
        })?;
    model.set_in_order(
        field::MEDIATOR,
        Value::Bytes(crate::create::reference_bytes(mediator)),
    );
    chart_archive.set_in_order(EXTENSION, Value::Bytes(model.encode()));
    document.set_archive_of(chart, &chart_archive)?;
    Ok(mediator)
}

/// One mediator formula: a reference to a range of a table, wrapped in function
/// 175.
fn reference_formula(
    table: &crate::table::Table,
    range: &crate::table::CellRange,
) -> Result<Vec<u8>, crate::Error> {
    let rectangle = range.resolve()?;
    let cell = |row: usize, column: usize| {
        format!(
            "${}${}",
            crate::formula::column_letters(column as i64),
            row + 1
        )
    };
    let text = match rectangle.top_left == rectangle.bottom_right {
        true => format!("={}", cell(rectangle.top_left.0, rectangle.top_left.1)),
        false => format!(
            "={}:{}",
            cell(rectangle.top_left.0, rectangle.top_left.1),
            cell(rectangle.bottom_right.0, rectangle.bottom_right.1)
        ),
    };
    let ast = crate::formula_parse::parse(&text, (0, 0))
        .map_err(|e| crate::Error::Format(format!("{text}: {e}")))?;

    // The reference names the table it reaches, the way a cross-table reference
    // does — and a mediator's reference always does, because the chart is not
    // in the table.
    let mut nodes = ast.nodes;
    let mut reference = nodes
        .first()
        .ok_or_else(|| crate::Error::Format(format!("{text} parsed to nothing")))?
        .message()
        .clone();
    reference.set_in_order(28, Value::Bytes(cross_table_info(table.base_uid).encode()));
    nodes[0] = crate::formula::Node::decode(reference)
        .ok_or_else(|| crate::Error::Format("the range node is not a node".to_string()))?;

    // …and every one of them goes through function 175.
    let mut wrapper = crate::pb::Message::default();
    wrapper.set_in_order(1, Value::Varint(16));
    wrapper.set_in_order(2, Value::Varint(MEDIATOR_FUNCTION));
    wrapper.set_in_order(3, Value::Varint(1));
    nodes.push(
        crate::formula::Node::decode(wrapper)
            .ok_or_else(|| crate::Error::Format("the wrapper is not a node".to_string()))?,
    );

    let mut formula = crate::pb::Message::default();
    formula.set_in_order(1, Value::Bytes(crate::formula::Ast { nodes }.encode()));
    Ok(formula.encode())
}

/// `AST_cross_table_reference_extra_info` — field 28's `TSP.CFUUIDArchive`.
fn cross_table_info(uid: crate::table::Uuid) -> crate::pb::Message {
    let mut uuid = crate::pb::Message::default();
    uuid.set_in_order(2, Value::Varint(uid.lower & 0xffff_ffff));
    uuid.set_in_order(3, Value::Varint(uid.lower >> 32));
    uuid.set_in_order(4, Value::Varint(uid.upper & 0xffff_ffff));
    uuid.set_in_order(5, Value::Varint(uid.upper >> 32));
    let mut extra = crate::pb::Message::default();
    extra.set_in_order(1, Value::Bytes(uuid.encode()));
    extra
}

/// The owner uid for a mediator's entity id.
///
/// The two are the same sixteen bytes with the halves read the other way round:
/// entity `E357DB84-F3DD-4CBE-B2D8-8CD778D0CFE8` is owner uid `{lower:
/// 0x…CFE8D078…, upper: …}` — checked against all twelve pairs in
/// `numbers-charts.numbers`.
pub fn uuid_from_entity(entity: &str) -> Option<crate::table::Uuid> {
    let plain: String = entity.chars().filter(|c| *c != '-').collect();
    if plain.len() != 32 {
        return None;
    }
    let mut bytes = [0u8; 16];
    for (index, byte) in bytes.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&plain[index * 2..index * 2 + 2], 16).ok()?;
    }
    bytes.reverse();
    Some(crate::table::Uuid {
        lower: u64::from_be_bytes(bytes[8..].try_into().ok()?),
        upper: u64::from_be_bytes(bytes[..8].try_into().ok()?),
    })
}
