//! Write a Pages document from a slice of Rust data, with no Apple software
//! anywhere in the picture.
//!
//! ```text
//! cargo run --example report -- Quarter.pages
//! open Quarter.pages
//! ```

use iwork::{Document, Kind};

struct Region {
    name: &'static str,
    units: u32,
    revenue: f64,
}

const SALES: &[Region] = &[
    Region {
        name: "Zürich",
        units: 1_240,
        revenue: 184_300.0,
    },
    Region {
        name: "Genève",
        units: 980,
        revenue: 151_900.0,
    },
    Region {
        name: "Lugano",
        units: 415,
        revenue: 62_250.0,
    },
];

fn main() -> Result<(), iwork::Error> {
    let out = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "Quarter.pages".to_string());

    let mut doc = Document::new(Kind::Pages)?;
    doc.append_paragraph("Sales, first quarter")?;
    doc.append_paragraph("")?;

    let mut total = 0.0;
    for region in SALES {
        doc.append_paragraph(&format!(
            "{} — {} units, CHF {:.2}",
            region.name, region.units, region.revenue
        ))?;
        total += region.revenue;
    }
    doc.append_paragraph("")?;
    doc.append_paragraph(&format!("Total: CHF {total:.2}"))?;

    doc.save(&out)?;
    println!("wrote {out}");
    Ok(())
}
