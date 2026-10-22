//! Write a Keynote deck from a slice of Rust data, with no Apple software
//! anywhere in the picture.
//!
//! ```text
//! cargo run --example deck -- Quarter.key
//! open Quarter.key
//! ```
//!
//! **A slide is not a page with a title slot.** What a slide can hold is
//! decided by the *layout* it is built on: a title is a placeholder that layout
//! defines, and a deck made from nothing has a layout that defines none — so
//! `slide.title(…)` refuses by name here, and the words go into text boxes
//! placed on the slide. Open a deck built from one of Apple's themes and the
//! placeholders are there, and `title` and `body` write them.

use iwork::drawable::Frame;
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
        .unwrap_or_else(|| "Quarter.key".to_string());

    // A new deck comes with one slide; the rest are added to the end.
    let mut doc = Document::new(Kind::Keynote)?;
    for _ in 1..SALES.len() {
        doc.add_slide(None)?;
    }

    for (index, region) in SALES.iter().enumerate() {
        let mut slide = doc.slide_mut(index)?;
        slide.add_text_box(
            region.name,
            Frame {
                x: 100.0,
                y: 120.0,
                width: 1720.0,
                height: 160.0,
            },
        )?;
        slide.add_text_box(
            &format!("{} units\nCHF {:.2}", region.units, region.revenue),
            Frame {
                x: 100.0,
                y: 360.0,
                width: 1720.0,
                height: 320.0,
            },
        )?;
        // The app's own name for the effect, or the identifier on the wire.
        slide.transition("dissolve")?;
    }

    doc.save(&out)?;
    println!("wrote {out} — {} slides", doc.slides().len());
    Ok(())
}
