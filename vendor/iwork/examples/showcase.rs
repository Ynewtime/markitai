//! Everything 0.2.0 can put in a Numbers document, in one file — written
//! from Rust, with no Apple software anywhere in the picture.
//!
//! ```text
//! cargo run --example showcase -- Review.numbers
//! open Review.numbers
//! ```
//!
//! The shape worth noticing is the one Excel gets wrong: **a sheet is not a
//! grid.** It is a canvas, and it holds as many tables as you like, each with
//! its own name, its own size and its own position. Both sheets here carry
//! two, placed side by side — which is why every call names a table rather
//! than assuming the sheet *is* one.

use iwork::table::{CellValue, Decimal, Format, SortRule};
use iwork::Document;

struct Route {
    name: &'static str,
    passengers: u32,
    revenue: f64,
    /// Seats sold over seats run, as a fraction — Numbers draws the `%`.
    load: f64,
}

const ROUTES: &[Route] = &[
    Route {
        name: "Zürich – Chur",
        passengers: 184_320,
        revenue: 4_238_400.0,
        load: 0.871,
    },
    Route {
        name: "Genève – Zermatt",
        passengers: 121_880,
        revenue: 3_900_160.0,
        load: 0.794,
    },
    Route {
        name: "Luzern – Interlaken",
        passengers: 98_450,
        revenue: 2_165_900.0,
        load: 0.912,
    },
    Route {
        name: "Chur – St. Moritz",
        passengers: 76_210,
        revenue: 1_981_460.0,
        load: 0.688,
    },
    Route {
        name: "Lugano – Locarno",
        passengers: 41_005,
        revenue: 615_075.0,
        load: 0.543,
    },
];

struct Railcar {
    number: &'static str,
    entered: &'static str,
    kilometres: f64,
    /// Mean time between failures, in hours — big enough to want exponents.
    mtbf: f64,
}

const FLEET: &[Railcar] = &[
    Railcar {
        number: "Ge 4/4 III 651",
        entered: "1993-05-14",
        kilometres: 8_412_000.0,
        mtbf: 42_800.0,
    },
    Railcar {
        number: "Ge 4/4 III 652",
        entered: "1993-09-02",
        kilometres: 8_105_500.0,
        mtbf: 39_150.0,
    },
    Railcar {
        number: "ABe 8/12 3501",
        entered: "2010-03-28",
        kilometres: 3_288_400.0,
        mtbf: 61_200.0,
    },
    Railcar {
        number: "ABe 8/12 3502",
        entered: "2010-07-11",
        kilometres: 3_190_750.0,
        mtbf: 58_940.0,
    },
];

fn main() -> Result<(), iwork::Error> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Review.numbers".to_string());

    // Banner row, header row, one row per route, a total row.
    let rows = ROUTES.len() + 3;
    let mut doc = Document::new_spreadsheet("Revenue", "Routes", rows, 4)?;

    revenue(&mut doc)?;

    // A second table on the *same* sheet, placed to the right of the first.
    // This is the thing a spreadsheet library built for Excel cannot express.
    doc.add_table_at("Revenue", "Headline", 4, 2, (620.0, 0.0))?;
    headline(&mut doc)?;

    doc.add_sheet("Fleet", "Rolling Stock", FLEET.len() + 2, 4)?;
    fleet(&mut doc)?;

    doc.add_table_at("Fleet", "Maintenance", 3, 2, (620.0, 0.0))?;
    maintenance(&mut doc)?;

    doc.save(&out)?;
    println!("wrote {out}");
    println!(
        "  {} sheets, {} tables",
        doc.sheets().len(),
        doc.tables().len()
    );
    Ok(())
}

/// The main table: money, percentages, a formula row and a sort rule.
fn revenue(doc: &mut Document) -> Result<(), iwork::Error> {
    let mut t = doc.table_mut("Routes")?;

    // A banner across the top. The value goes in the anchor first — writing
    // into a cell a merge has swallowed is refused, and rightly.
    t.set("A1", "Alpine Rail — Q3 2026")?;
    t.merge("A1:D1")?;
    t.row_height(0, Some(38.0))?;

    t.set_block("A2", &[vec!["Route", "Passengers", "Revenue", "Load"]])?;

    for (index, route) in ROUTES.iter().enumerate() {
        let row = index + 3; // A1 numbering: banner is 1, header is 2.
        t.set(format!("A{row}"), route.name)?;
        t.set(format!("B{row}"), route.passengers)?;
        // Money is a *value type*, not a format — the cell becomes a currency
        // cell and Numbers draws the symbol without being told to.
        t.currency(format!("C{row}"), route.revenue, "CHF")?;
        t.set(format!("D{row}"), route.load)?;
    }

    let first = 3;
    let last = ROUTES.len() + 2;
    let total = last + 1;

    t.set(format!("A{total}"), "Total")?;
    // Real formulas, registered in the calculation engine: Numbers recalculates
    // these the moment you edit a figure above them. The value written is what
    // the cell shows until it does — this crate writes formulas and evaluates
    // none of them.
    t.formula(
        format!("B{total}"),
        &format!("=SUM(B{first}:B{last})"),
        ROUTES.iter().map(|r| r.passengers).sum::<u32>(),
    )?;
    t.formula(
        format!("C{total}"),
        &format!("=SUM(C{first}:C{last})"),
        CellValue::Currency(Decimal::from_f64(
            ROUTES.iter().map(|r| r.revenue).sum::<f64>(),
        )),
    )?;
    // A weighted mean, not an average of averages — and a formula the app
    // prints back in its own spelling.
    t.formula(
        format!("D{total}"),
        &format!("=SUMPRODUCT(B{first}:B{last},D{first}:D{last})/B{total}"),
        weighted_load(),
    )?;

    // Formats, over a range spelled the way the app spells it.
    t.format(
        format!("B{first}:B{total}"),
        &Format::Number { decimals: Some(0) },
    )?;
    t.format(
        format!("D{first}:D{total}"),
        &Format::Percent { decimals: Some(1) },
    )?;

    t.column_width(0, Some(190.0))?;
    t.column_width(2, Some(130.0))?;

    // What to sort *by*. Nothing here reorders a row — this is the rule the
    // app's own sort applies when you ask it to.
    t.sort_by(&[SortRule {
        column: 2,
        descending: true,
    }])?;
    Ok(())
}

/// A small second table on the same sheet, summarising the first.
///
/// These are values rather than formulas on purpose. `=Routes::C8` — a
/// reference into *another table* — is refused by 0.2.0's formula parser:
///
/// ```text
/// Refused { reason: NotACell, detail: "Headline B3: 'Routes' is neither a
/// cell reference nor a function this crate knows — a header name, a defined
/// name and a bare TRUE are all refused rather than guessed at" }
/// ```
///
/// Which is the crate behaving correctly. It will not emit a `TSCE` node it
/// has not watched the app write, and a cross-table reference is a shape
/// nobody has yet established. A refusal you can match on beats a formula the
/// app opens and quietly reads as something else.
fn headline(doc: &mut Document) -> Result<(), iwork::Error> {
    let mut t = doc.table_mut("Headline")?;

    t.set("A1", "Headline")?;
    t.merge("A1:B1")?;

    t.set("A2", "Routes")?;
    t.set("B2", ROUTES.len() as u32)?;

    t.set("A3", "Revenue")?;
    t.currency("B3", ROUTES.iter().map(|r| r.revenue).sum::<f64>(), "CHF")?;

    t.set("A4", "Mean load")?;
    t.set("B4", weighted_load())?;
    t.format("B4", &Format::Percent { decimals: Some(1) })?;

    t.column_width(0, Some(120.0))?;
    Ok(())
}

/// Dates and scientific notation, which are the two formats nobody demos.
fn fleet(doc: &mut Document) -> Result<(), iwork::Error> {
    let mut t = doc.table_mut("Rolling Stock")?;

    t.set_block(
        "A1",
        &[vec!["Railcar", "In service", "Kilometres", "MTBF (h)"]],
    )?;

    for (index, car) in FLEET.iter().enumerate() {
        let row = index + 2;
        t.set(format!("A{row}"), car.number)?;
        t.set(format!("B{row}"), car.entered)?;
        t.set(format!("C{row}"), car.kilometres)?;
        t.set(format!("D{row}"), car.mtbf)?;
    }

    let last = FLEET.len() + 1;
    t.format(format!("C2:C{last}"), &Format::Number { decimals: Some(0) })?;
    t.format(
        format!("D2:D{last}"),
        &Format::Scientific { decimals: Some(2) },
    )?;

    t.column_width(0, Some(150.0))?;
    t.column_width(2, Some(120.0))?;
    Ok(())
}

fn maintenance(doc: &mut Document) -> Result<(), iwork::Error> {
    let mut t = doc.table_mut("Maintenance")?;
    t.set("A1", "Fleet")?;
    t.merge("A1:B1")?;
    t.set("A2", "Railcars")?;
    t.set("B2", FLEET.len() as u32)?;
    t.set("A3", "Total km")?;
    t.set("B3", FLEET.iter().map(|c| c.kilometres).sum::<f64>())?;
    t.format("B3", &Format::Number { decimals: Some(0) })?;
    t.column_width(0, Some(120.0))?;
    Ok(())
}

fn weighted_load() -> f64 {
    let seats: f64 = ROUTES.iter().map(|r| r.passengers as f64).sum();
    ROUTES
        .iter()
        .map(|r| r.passengers as f64 * r.load)
        .sum::<f64>()
        / seats
}
