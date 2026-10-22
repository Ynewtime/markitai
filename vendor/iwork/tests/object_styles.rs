//! Painting a drawable: fill, outline and opacity.
//!
//! Two things had to be measured before any of this could be claimed, and
//! both were measured the same way — by making the app open the document and
//! **save it again**, then reading what it wrote.
//!
//! **A drawable needs a style of its own.** A document from nothing points
//! every shape at a theme preset — `line-style-preset-0` and its siblings —
//! and painting a preset does not survive: Keynote regenerated its presets on
//! save and the colour was gone. What the app does instead is give the object
//! its own archive naming the preset as parent and carrying only what differs.
//! That is what the first paint now makes.
//!
//! **`override_count` is not decoration.** A variation whose bag held a
//! colour while its count said `0` came back from Keynote's save with the
//! properties stripped out. The count is maintained from the bag, so the two
//! cannot disagree.
//!
//! With both in place the colours survive: a red rectangle and a blue ellipse
//! written here read back out of the file Keynote itself wrote, channels
//! unchanged, and the same in Pages.

use std::path::{Path, PathBuf};

use iwork::drawable::{Color, Fill, Outline};
use iwork::{Document, Kind};

fn scratch(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir_all(&path);
    path
}

const RED: Color = Color {
    red: 0.83,
    green: 0.18,
    blue: 0.18,
    alpha: 1.0,
};
const BLUE: Color = Color {
    red: 0.11,
    green: 0.35,
    blue: 0.62,
    alpha: 1.0,
};
const BLACK: Color = Color {
    red: 0.0,
    green: 0.0,
    blue: 0.0,
    alpha: 1.0,
};

fn fill_of(doc: &Document, drawable: u64) -> Option<Fill> {
    doc.drawables()
        .into_iter()
        .find(|d| d.identifier == drawable)
        .and_then(|d| d.style)
        .and_then(|style| doc.object_style(style))
        .and_then(|style| style.fill)
}

/// Two shapes sharing a preset end up with a style each, and the one that was
/// shared is left exactly as it was.
#[test]
fn painting_a_drawable_gives_it_a_style_of_its_own() {
    let mut deck = Document::new(Kind::Keynote).unwrap();
    let slide = deck.slides()[0].identifier.to_string();
    let a = deck
        .add_shape(
            &slide,
            Outline::Rectangle,
            "Rot",
            (120.0, 200.0),
            (500.0, 300.0),
        )
        .unwrap();
    let b = deck
        .add_shape(
            &slide,
            Outline::Ellipse,
            "Blau",
            (760.0, 200.0),
            (400.0, 300.0),
        )
        .unwrap();

    let shared = deck
        .drawables()
        .into_iter()
        .find(|d| d.identifier == a)
        .and_then(|d| d.style)
        .expect("a shape points at a style");
    assert_eq!(
        deck.drawables()
            .into_iter()
            .find(|d| d.identifier == b)
            .and_then(|d| d.style),
        Some(shared),
        "both shapes start out sharing the theme's preset"
    );

    deck.set_object_fill(a, Some(RED)).unwrap();
    deck.set_object_fill(b, Some(BLUE)).unwrap();

    let style_of = |doc: &Document, id: u64| {
        doc.drawables()
            .into_iter()
            .find(|d| d.identifier == id)
            .and_then(|d| d.style)
            .unwrap()
    };
    let (sa, sb) = (style_of(&deck, a), style_of(&deck, b));
    assert_ne!(sa, shared, "the first paint moved it off the preset");
    assert_ne!(sb, shared);
    assert_ne!(sa, sb, "and each shape got its own");

    // The preset it was sharing is untouched: it still paints nothing.
    assert!(
        matches!(
            deck.object_style(shared).and_then(|s| s.fill),
            Some(Fill::None) | None
        ),
        "the theme preset is left alone"
    );

    match fill_of(&deck, a) {
        Some(Fill::Color(c)) => assert!((c.red - 0.83).abs() < 1e-6, "{c:?}"),
        other => panic!("expected red, got {other:?}"),
    }
    match fill_of(&deck, b) {
        Some(Fill::Color(c)) => assert!((c.blue - 0.62).abs() < 1e-6, "{c:?}"),
        other => panic!("expected blue, got {other:?}"),
    }
    assert!(deck.problems().is_empty(), "{:?}", deck.problems());
}

/// Painting the same drawable twice does not make a second style for it.
#[test]
fn painting_twice_reuses_the_style_it_already_owns() {
    let mut deck = Document::new(Kind::Keynote).unwrap();
    let slide = deck.slides()[0].identifier.to_string();
    let shape = deck
        .add_shape(
            &slide,
            Outline::Rectangle,
            "",
            (100.0, 100.0),
            (300.0, 300.0),
        )
        .unwrap();

    deck.set_object_fill(shape, Some(RED)).unwrap();
    let first = deck
        .drawables()
        .into_iter()
        .find(|d| d.identifier == shape)
        .and_then(|d| d.style)
        .unwrap();

    deck.set_object_stroke(shape, BLACK, 3.0).unwrap();
    deck.set_object_opacity(shape, 0.5).unwrap();
    deck.set_object_fill(shape, Some(BLUE)).unwrap();

    let second = deck
        .drawables()
        .into_iter()
        .find(|d| d.identifier == shape)
        .and_then(|d| d.style)
        .unwrap();
    assert_eq!(first, second, "one style, painted four times");

    let style = deck.object_style(second).expect("a style");
    assert!(matches!(style.fill, Some(Fill::Color(c)) if (c.blue - 0.62).abs() < 1e-6));
    assert_eq!(style.opacity, Some(0.5));
    assert_eq!(style.stroke.as_ref().map(|s| s.width), Some(3.0));
    // Three properties in the bag, and the count says three.
    assert_eq!(
        style.override_count,
        Some(3),
        "override_count follows the bag"
    );
    assert!(deck.problems().is_empty(), "{:?}", deck.problems());
}

/// Values that are not colours or fractions are refused by name.
#[test]
fn an_impossible_paint_is_refused() {
    let mut deck = Document::new(Kind::Keynote).unwrap();
    let slide = deck.slides()[0].identifier.to_string();
    let shape = deck
        .add_shape(
            &slide,
            Outline::Rectangle,
            "",
            (100.0, 100.0),
            (300.0, 300.0),
        )
        .unwrap();

    let error = deck.set_object_opacity(shape, 1.5).unwrap_err();
    assert_eq!(
        error.refusal(),
        Some(iwork::Refusal::UnwritableValue),
        "{error:?}"
    );
    let error = deck.set_object_stroke(shape, BLACK, -1.0).unwrap_err();
    assert_eq!(
        error.refusal(),
        Some(iwork::Refusal::UnwritableValue),
        "{error:?}"
    );

    // A table has no object style, and says so rather than inventing one.
    let mut sheet = Document::new_spreadsheet("Blatt", "Tabelle", 2, 2).unwrap();
    let table = sheet.tables().first().map(|t| t.identifier).unwrap();
    let error = sheet.set_object_fill(table, Some(RED)).unwrap_err();
    assert_eq!(error.refusal(), Some(iwork::Refusal::NotDrawn), "{error:?}");
}

/// **The acceptance test.** The app opens the document, saves it again, and
/// the colours are still there. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn the_apps_keep_a_colour_this_crate_painted() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the resave");
        return;
    }
    let resave = |path: &Path| {
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/resave.sh");
        let output = std::process::Command::new(&script)
            .arg(path)
            .output()
            .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
        assert!(
            output.status.success(),
            "the app would not resave {}:\n{}\n{}",
            path.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    };

    // Keynote.
    let mut deck = Document::new(Kind::Keynote).unwrap();
    let slide = deck.slides()[0].identifier.to_string();
    let shape = deck
        .add_shape(
            &slide,
            Outline::Rectangle,
            "Rot",
            (120.0, 200.0),
            (500.0, 300.0),
        )
        .unwrap();
    deck.set_object_fill(shape, Some(RED)).unwrap();
    deck.set_object_stroke(shape, BLACK, 3.0).unwrap();
    deck.set_object_opacity(shape, 0.9).unwrap();
    let out = scratch("iwork-painted.key");
    deck.save(&out).unwrap();
    resave(&out);

    let after = Document::open(&out).unwrap();
    match fill_of(&after, shape) {
        Some(Fill::Color(c)) => {
            assert!((c.red - 0.83).abs() < 1e-4, "Keynote kept the red: {c:?}");
            assert!((c.green - 0.18).abs() < 1e-4, "{c:?}");
        }
        other => panic!("Keynote dropped the fill: {other:?}"),
    }
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_file(&out);

    // Pages.
    let mut doc = Document::new(Kind::Pages).unwrap();
    let circle = doc
        .add_shape(
            "page 1",
            Outline::Ellipse,
            "Kreis",
            (72.0, 300.0),
            (200.0, 200.0),
        )
        .unwrap();
    doc.set_object_fill(circle, Some(RED)).unwrap();
    doc.set_object_stroke(circle, BLACK, 2.0).unwrap();
    let out = scratch("iwork-painted.pages");
    doc.save(&out).unwrap();
    resave(&out);

    let after = Document::open(&out).unwrap();
    match fill_of(&after, circle) {
        Some(Fill::Color(c)) => assert!((c.red - 0.83).abs() < 1e-4, "Pages kept the red: {c:?}"),
        other => panic!("Pages dropped the fill: {other:?}"),
    }
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_file(&out);
}
