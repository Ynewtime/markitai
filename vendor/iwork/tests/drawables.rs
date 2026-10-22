//! Drawables, geometry and media — read, written, and then handed to the app.
//!
//! Three claims, in the order they have to hold.
//!
//! **The decoder agrees with the app.** Every rectangle `iwork drawables`
//! prints is one Keynote or Pages will report for the same object, including
//! the two cases where the archive says something else: a masked image, whose
//! frame is the mask's window offset by the picture's own position, and a
//! rotated one, whose reported position is the rotated bounding box's corner.
//! Behind `IWORK_APP_CHECK=1` that is checked object by object against
//! `scripts/drawable-oracle.sh`.
//!
//! **A write touches what it must and nothing else.** Moving a shape rewrites
//! one stream; a no-op move rewrites none; every other object comes back byte
//! for byte.
//!
//! **A replacement refuses to lie.** An image carrying a crop, a shaped mask,
//! an Instant Alpha path, adjustments or cached renderings of the old pixels is
//! not swapped, because the result would open, report the same geometry and
//! draw the wrong thing.

use std::path::{Path, PathBuf};

use iwork::drawable::{self, Kind};
use iwork::pb::{decode_nested, Message, Value};
use iwork::Document;

fn generated(name: &str) -> Option<PathBuf> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/generated")
        .join(name);
    path.exists().then_some(path)
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

/// A password-protected package, which is not a fixture any test here can use:
/// its object streams are ciphertext and `Document::open` refuses it by design.
/// `tests/fixtures.rs` is where that refusal is asserted.
fn encrypted(path: &Path) -> bool {
    iwork::Package::read(path).is_ok_and(|package| package.contains(".iwpv2"))
}

fn every_fixture() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
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
    found.sort();
    found
}

// -- reading -----------------------------------------------------------------

/// The rule every read here rests on: the geometry is found by walking `super`,
/// not by knowing how deep it is. Four levels is what a Keynote placeholder
/// needs and one is what an image needs, and both must come back.
#[test]
fn every_drawable_has_a_geometry_at_a_depth_nobody_assumed() {
    let mut seen = 0usize;
    let mut depths = std::collections::BTreeMap::new();
    for path in every_fixture() {
        let doc = Document::open(&path).unwrap();
        for drawable in doc.drawables() {
            seen += 1;
            *depths.entry(drawable.path.len()).or_insert(0usize) += 1;
            assert!(
                drawable.path.len() <= 4,
                "{}: drawable {} sits {} levels down",
                path.display(),
                drawable.identifier,
                drawable.path.len()
            );
            assert!(
                drawable.path.iter().all(|&field| field == 1),
                "{}: drawable {} was reached through something other than super",
                path.display(),
                drawable.identifier
            );
        }
    }
    if seen == 0 {
        eprintln!("no fixtures — skipping (run scripts/make-fixtures.sh)");
        return;
    }
    assert!(
        depths.len() > 1,
        "every drawable in the corpus sits at the same depth ({depths:?}), \
         so the walk is untested"
    );
}

/// Decoding a geometry and writing it straight back must not change a byte.
///
/// The same discipline the cell encoder is held to: a geometry carries flags
/// and an angle this crate does not interpret, and they have to survive an
/// edit that does not concern them.
#[test]
fn every_geometry_re_encodes_to_the_bytes_it_came_from() {
    let mut seen = 0usize;
    for path in every_fixture() {
        let doc = Document::open(&path).unwrap();
        for drawable in doc.drawables() {
            let (_, object) = doc.object(drawable.identifier).unwrap();
            let archive = Message::decode(object.payload()).unwrap();
            let mut at: Vec<u32> = drawable.path.clone();
            at.push(drawable::field::GEOMETRY);
            let Some(Value::Bytes(raw)) = iwork::style::get_path(&archive, &at) else {
                panic!(
                    "{}: drawable {} has no geometry",
                    path.display(),
                    drawable.identifier
                );
            };
            let mut message = decode_nested(&raw).unwrap();
            let before = message.encode();
            drawable::Geometry::decode(&message).write_into(&mut message);
            assert_eq!(
                message.encode(),
                before,
                "{}: geometry of drawable {} changed on a no-op write",
                path.display(),
                drawable.identifier
            );
            seen += 1;
        }
    }
    if seen == 0 {
        eprintln!("no fixtures — skipping (run scripts/make-fixtures.sh)");
    }
}

/// A masked image is reported as its mask's window, moved into the parent's
/// space. The numbers are the ones Pages itself reports for this fixture.
#[test]
fn a_masked_image_is_framed_by_its_mask() {
    let path = fixture!("pages-report.pages");
    let doc = Document::open(&path).unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image)
        .expect("the report has a photo");
    let mask = doc
        .drawable(image.mask().expect("the photo is cropped"))
        .unwrap();

    // What the archive says.
    assert!((image.geometry.x - 33.858_46).abs() < 0.01);
    assert!((image.geometry.width - 511.858_34).abs() < 0.01);
    assert!((mask.geometry.x - 25.891_663).abs() < 0.01);
    assert!((mask.geometry.width - 475.0).abs() < 0.01);

    // What Pages says: 60, 123, 475 × 383.
    let frame = image.frame(Some(&mask));
    assert!((frame.x - 59.75).abs() < 0.5, "frame x {}", frame.x);
    assert!((frame.y - 122.79).abs() < 0.5, "frame y {}", frame.y);
    assert!((frame.width - 475.0).abs() < 0.01);
    assert!((frame.height - 383.0).abs() < 0.01);
}

/// A rotated object's reported position is the corner of its rotated bounding
/// box, and its reported size is not rotated at all. Keynote says 470, 57 for
/// the 220 × 180 shape this fixture turns 30°.
#[test]
fn a_rotated_shape_is_framed_by_its_bounding_box() {
    let path = fixture!("keynote-shapes.key");
    let doc = Document::open(&path).unwrap();
    let turned = doc
        .drawables()
        .into_iter()
        .find(|d| d.geometry.angle != 0.0 && d.geometry.height == 180.0)
        .expect("the fixture has a shape turned 30 degrees");
    assert_eq!(turned.geometry.angle, 30.0);
    assert_eq!((turned.geometry.x, turned.geometry.y), (500.0, 100.0));

    let frame = turned.frame(None);
    assert!((frame.x - 469.75).abs() < 0.5, "frame x {}", frame.x);
    assert!((frame.y - 57.04).abs() < 0.5, "frame y {}", frame.y);
    assert_eq!((frame.width, frame.height), (220.0, 180.0));

    // A line is the same rule with a zero-height rectangle: stored at
    // 93.84 × 650, 412.31 wide at 346°, reported at 100 × 600.
    let line = doc
        .drawables()
        .into_iter()
        .find(|d| {
            d.path_source
                .as_ref()
                .is_some_and(|p| p.looks_like_a_line())
        })
        .expect("the fixture has a line");
    let frame = line.frame(None);
    assert!((frame.x - 100.0).abs() < 0.5, "line x {}", frame.x);
    assert!((frame.y - 600.0).abs() < 0.5, "line y {}", frame.y);
}

/// Opacity and reflection are style properties, and a style that does not carry
/// one inherits it. Keynote wrote 0.5 and 0.4 into a *variation* style whose
/// parent holds everything else.
#[test]
fn a_variation_style_inherits_what_it_does_not_override() {
    let path = fixture!("keynote-shapes.key");
    let doc = Document::open(&path).unwrap();
    let faded = doc
        .drawables()
        .into_iter()
        .filter(|d| d.kind == Kind::Shape)
        .find_map(|d| {
            let style = doc.object_style(d.style?)?;
            (style.opacity == Some(0.5)).then_some(style)
        })
        .expect("the fixture has a shape at 50% opacity");
    assert_eq!(faded.reflection, Some(0.4), "reflection value 40");
    assert!(
        !faded.inherited_from.is_empty(),
        "the variation carries only what changed, so the rest is inherited"
    );
    assert!(
        faded.fill.is_some(),
        "the fill came from the parent style, not from the variation"
    );
}

/// A shape owns its text, and the storage is two levels above the geometry —
/// on the `TSWP.ShapeInfoArchive`, not on the shape or the drawable. Phase 4
/// will want it: this is how the words in a slide's shapes are reached from the
/// drawable side.
#[test]
fn a_shape_owns_its_text_storage() {
    let path = fixture!("keynote-shapes.key");
    let doc = Document::open(&path).unwrap();
    let said: Vec<String> = doc
        .drawables()
        .into_iter()
        .filter(|d| d.kind == Kind::Shape)
        .filter_map(|d| d.text)
        .filter_map(|storage| doc.storage_text(storage).ok())
        .collect();
    assert!(
        said.iter().any(|text| text.contains("Ein Rechteck")),
        "the shape's own words: {said:?}"
    );
    assert!(
        said.iter()
            .any(|text| text.contains("Ein Textfeld mit Größe")),
        "the text box's words: {said:?}"
    );
}

/// The wider media model, read but never authored: the Keynote theme ships two
/// live-video sources, which ground rule 8 says this crate carries and reads
/// and must not synthesise.
#[test]
fn live_video_sources_are_read_and_named() {
    let path = fixture!("keynote-deck.key");
    let doc = Document::open(&path).unwrap();
    let movies: Vec<_> = doc
        .drawables()
        .into_iter()
        .filter(|d| d.kind == Kind::Movie)
        .collect();
    assert!(!movies.is_empty(), "the theme ships movie placeholders");
    for movie in &movies {
        let media = movie.media.as_ref().unwrap();
        assert!(
            media.live_video,
            "movie {} is a live video source",
            movie.identifier
        );
        assert!(
            movie.extensions.contains(&"live video"),
            "the KN.LiveVideoInfo extension is named"
        );
        assert!(media.poster.is_some(), "it has a poster frame");
        assert!(media.data.is_none(), "and no film");
    }
}

/// Alt text — `TSD.DrawableArchive.accessibility_description`, field 8 — is the
/// one accessibility field the apps expose to a script (`description` of an
/// `image`, read-write), and the corpus is full of it: Apple writes one on the
/// photographs in its own templates.
///
/// This is the assertion the README row rests on. It is a tripwire as much as a
/// measurement: a decoder that stopped reading the field would leave every one
/// of these empty and nothing else in the suite would notice.
#[test]
fn alt_text_is_read_wherever_the_corpus_has_it() {
    let mut documents = 0;
    let mut descriptions = 0;
    for path in every_fixture() {
        let doc = Document::open(&path).unwrap();
        let described: Vec<_> = doc
            .drawables()
            .into_iter()
            .filter(|d| d.description.is_some())
            .collect();
        if described.is_empty() {
            continue;
        }
        documents += 1;
        for drawable in &described {
            let text = drawable.description.as_deref().unwrap();
            assert!(
                !text.trim().is_empty(),
                "{}: drawable {} has an empty description, which is not a description",
                path.display(),
                drawable.identifier
            );
            descriptions += 1;
        }
    }
    if documents == 0 {
        eprintln!("no fixtures — skipping");
        return;
    }
    assert!(
        documents >= 12 && descriptions >= 59,
        "the corpus had 59 alt texts in 12 documents; this run found {descriptions} in \
         {documents}, which means either the corpus or the decoder changed"
    );
}

/// The media registry's digest is a raw SHA-1 of the file's bytes. Checked
/// against every stored file in the corpus, which is what makes it safe for
/// `replace_media` to compute one.
#[test]
fn every_digest_is_the_sha1_of_the_bytes() {
    let mut checked = 0usize;
    for path in every_fixture() {
        let doc = Document::open(&path).unwrap();
        for file in doc.data_files() {
            let Some(entry) = file.entry_name() else {
                continue;
            };
            let bytes = doc
                .package()
                .get(&entry)
                .unwrap_or_else(|| panic!("{}: {entry} is missing", path.display()));
            assert_eq!(
                file.digest,
                iwork::media::sha1(bytes).to_vec(),
                "{}: digest of {entry}",
                path.display()
            );
            checked += 1;
        }
    }
    if checked == 0 {
        eprintln!("no stored media in the corpus — skipping");
    }
}

// -- writing -----------------------------------------------------------------

/// Moving one shape rewrites one stream, and writing a drawable the geometry it
/// already has rewrites none.
#[test]
fn moving_a_drawable_rewrites_only_the_stream_it_lives_in() {
    let path = fixture!("keynote-shapes.key");

    let mut doc = Document::open(&path).unwrap();
    let shape = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Shape && d.geometry.width == 300.0)
        .expect("the fixture has a 300-point shape");
    doc.set_geometry(shape.identifier, Some((250.0, 300.0)), Some((400.0, 120.0)))
        .unwrap();
    assert_eq!(
        doc.changed_streams(),
        vec![shape.stream.as_str()],
        "one stream, and it is the one the shape is in"
    );

    let mut doc = Document::open(&path).unwrap();
    let frame = shape.frame(None);
    doc.set_geometry(
        shape.identifier,
        Some((frame.x, frame.y)),
        Some((frame.width, frame.height)),
    )
    .unwrap();
    assert!(
        doc.changed_streams().is_empty(),
        "writing a rectangle an object already has is not a change"
    );
}

/// Resizing a masked image scales the whole assembly — the picture, the mask's
/// offset, the mask's size and the mask path's natural size — by one factor, so
/// the frame lands where it was asked to and the crop keeps its proportions.
///
/// The numbers are Pages': asked to make the report's 475-point-wide photo 300
/// wide, it multiplied everything by 300/475.
#[test]
fn resizing_a_masked_image_scales_the_whole_assembly() {
    let path = fixture!("pages-report.pages");
    let mut doc = Document::open(&path).unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image)
        .unwrap();
    let mask = image.mask().unwrap();

    let change = doc
        .set_geometry(
            image.identifier,
            Some((60.0, 123.0)),
            Some((300.0, 241.894_73)),
        )
        .unwrap();
    assert_eq!(change.mask, Some(mask));

    let after = doc.drawable(image.identifier).unwrap();
    let after_mask = doc.drawable(mask).unwrap();
    let scale = 300.0 / 475.0;

    assert!(
        (after.geometry.width - 511.858_34 * scale).abs() < 0.01,
        "the picture itself was scaled: {}",
        after.geometry.width
    );
    assert!(
        (after_mask.geometry.x - 25.891_663 * scale).abs() < 0.01,
        "the mask's offset was scaled: {}",
        after_mask.geometry.x
    );
    assert!((after_mask.geometry.width - 300.0).abs() < 0.01);
    let natural = after_mask
        .path_source
        .as_ref()
        .and_then(|p| p.natural_size)
        .expect("the mask has a path source");
    assert!(
        (natural.0 - 300.0).abs() < 0.01,
        "the mask path's natural size follows the mask: {natural:?}"
    );

    // And the frame is where it was asked to be.
    let frame = after.frame(Some(&after_mask));
    assert!((frame.x - 60.0).abs() < 0.01, "frame x {}", frame.x);
    assert!((frame.y - 123.0).abs() < 0.01, "frame y {}", frame.y);

    // `originalSize` is left exactly as it was on a masked image: the app does
    // not fill it with the picture's own size there and the corpus does not
    // agree on what it does fill it with, so there is nothing to rewrite it to.
    let before = Document::open(&path).unwrap();
    let before_original = before
        .drawable(image.identifier)
        .unwrap()
        .media
        .unwrap()
        .original_size;
    let after_original = after.media.as_ref().unwrap().original_size;
    assert_eq!(
        after_original, before_original,
        "a masked image's originalSize must be untouched"
    );
}

/// Everything a geometry write does not concern comes back byte for byte.
#[test]
fn a_geometry_write_leaves_every_other_object_alone() {
    let path = fixture!("keynote-shapes.key");
    let before = Document::open(&path).unwrap();
    let mut after = Document::open(&path).unwrap();
    let shape = after
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Shape && d.geometry.width == 300.0)
        .unwrap();
    after
        .set_geometry(shape.identifier, Some((11.0, 22.0)), None)
        .unwrap();

    let mut compared = 0usize;
    for (_, object) in before.objects() {
        if object.identifier == shape.identifier {
            continue;
        }
        let (_, now) = after.object(object.identifier).unwrap();
        assert_eq!(
            now.payload(),
            object.payload(),
            "object {} changed and had no business changing",
            object.identifier
        );
        compared += 1;
    }
    assert!(compared > 100);
}

/// Replacing an image's bytes updates the registry entry, the package entry and
/// every drawable that shows the picture — and says so.
#[test]
fn replacing_media_brings_the_registry_and_the_drawables_into_step() {
    let path = fixture!("keynote-shapes.key");
    let mut doc = Document::open(&path).unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image && d.mask().is_none())
        .expect("the fixture has an uncropped image");
    let data = image.media.as_ref().unwrap().data.unwrap();

    // A 4 x 4 PNG, built here so the test needs no files.
    let png = tiny_png(4, 4);
    let replacement = doc
        .replace_media(image.identifier, &png, "tiny.png", None)
        .unwrap();

    assert_eq!(replacement.data, data);
    assert_eq!(
        replacement.now,
        "Data/tiny-9076.png".replace("9076", &data.to_string())
    );
    assert_eq!(replacement.new_pixel_size, (4.0, 4.0));
    assert!(replacement.drawables.contains(&image.identifier));
    assert!(
        replacement.aspect_changed,
        "32 x 24 became 4 x 4, which is a different shape"
    );
    assert!(replacement.aspect_note().is_some());

    // The package holds the new bytes under the new name and the digest agrees.
    assert_eq!(doc.package().get(&replacement.now), Some(png.as_slice()));
    assert!(!doc.package().contains(&replacement.was));
    let file = doc
        .data_files()
        .into_iter()
        .find(|f| f.identifier == data)
        .unwrap();
    assert_eq!(file.digest, iwork::media::sha1(&png).to_vec());
    assert_eq!(file.original_name, "tiny.png");

    // The drawable's natural size follows the picture, and it is marked as
    // replaced — which is the flag Keynote sets when it does this itself.
    let after = doc.drawable(image.identifier).unwrap();
    let media = after.media.as_ref().unwrap();
    assert_eq!(media.natural_size, Some((4.0, 4.0)));
    assert!(media.was_replaced());
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The refusal this method exists for. The fixture's second image is one
/// Keynote itself cropped, by being told to show a square picture in a 4:3
/// frame; swapping the bytes under that crop is exactly the change that opens
/// fine and renders wrong.
#[test]
fn replacing_media_refuses_an_image_that_is_cropped() {
    let path = fixture!("keynote-shapes.key");
    let mut doc = Document::open(&path).unwrap();
    let cropped = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image && d.edit_state.as_ref().is_some_and(|state| state.crops))
        .expect("the fixture has a cropped image");

    let before: Vec<Vec<u8>> = doc
        .objects()
        .map(|(_, object)| object.payload().to_vec())
        .collect();
    let error = doc
        .replace_media(cropped.identifier, &tiny_png(4, 4), "tiny.png", None)
        .unwrap_err();
    match &error {
        iwork::Error::NonDestructiveEdit { drawable, reasons } => {
            assert_eq!(*drawable, cropped.identifier);
            assert!(reasons.iter().any(|r| r.contains("cropped")), "{reasons:?}");
        }
        other => panic!("expected a refusal by name, got {other}"),
    }
    let after: Vec<Vec<u8>> = doc
        .objects()
        .map(|(_, object)| object.payload().to_vec())
        .collect();
    assert_eq!(before, after, "a refused replacement changes nothing");
}

/// The identity rule made exact. A mask crops unless its window is the whole
/// drawn picture at the origin — measured against the image's own **geometry**,
/// not `originalSize` (field 4), which for a masked image the app fills with the
/// mask window and so makes any size comparison a tautology. Slide a real crop's
/// window to (0, 0), leaving its size below the picture's, and it must still
/// classify as a crop and still be refused.
#[test]
fn a_crop_slid_to_the_origin_is_still_a_crop() {
    let path = fixture!("keynote-shapes.key");
    let mut doc = Document::open(&path).unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image && d.mask().is_some())
        .expect("the fixture has a masked image");
    let mask = image.mask().unwrap();

    // The window is 160 × 120 on a 160 × 160 picture. Move it to the origin,
    // keeping its size — it still shows less than the whole picture.
    doc.set_geometry(mask, Some((0.0, 0.0)), None).unwrap();
    let after = doc.drawable(image.identifier).unwrap();
    let state = after.edit_state.as_ref().unwrap();
    assert!(
        state.crops,
        "a 160 × 120 window on a 160 × 160 picture crops even at (0, 0)"
    );

    // And so replace_media refuses it, as it would any crop.
    let error = doc
        .replace_media(image.identifier, &tiny_png(4, 4), "tiny.png", None)
        .unwrap_err();
    assert!(
        matches!(error, iwork::Error::NonDestructiveEdit { .. }),
        "a crop at the origin must still be refused, got {error}"
    );
}

/// The other direction of the same rule: a window that really is the whole
/// picture at the origin is an identity mask and does not crop. Grow the crop's
/// window to the picture's own 160 × 160 at the origin and the classifier must
/// stop calling it a crop — which the old code could not, since it compared to
/// `originalSize` (160 × 120) rather than the image geometry.
#[test]
fn a_window_that_is_the_whole_picture_does_not_crop() {
    let path = fixture!("keynote-shapes.key");
    let mut doc = Document::open(&path).unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image && d.mask().is_some())
        .expect("the fixture has a masked image");
    let mask = image.mask().unwrap();

    doc.set_geometry(mask, Some((0.0, 0.0)), Some((160.0, 160.0)))
        .unwrap();
    let after = doc.drawable(image.identifier).unwrap();
    assert!(
        !after.edit_state.as_ref().unwrap().crops,
        "a 160 × 160 window at the origin on a 160 × 160 picture is an identity"
    );
}

/// A thumbnail (field 12, surfaced as `Media::poster`) is a separately stored
/// downscale of the *old* picture. It is one of the derived renderings the
/// method exists to refuse — 65 of the 69 corpus images carry it — and
/// replacing the bytes under it leaves it lying.
#[test]
fn replacing_media_refuses_an_image_with_a_thumbnail() {
    let path = fixture!("pages-layout.pages");
    let mut doc = Document::open(&path).unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| {
            d.kind == Kind::Image
                && d.media.as_ref().and_then(|m| m.poster).is_some()
                && d.mask().is_none()
        })
        .expect("pages-layout has an image with a thumbnail");
    assert!(
        image
            .edit_state
            .as_ref()
            .unwrap()
            .derived
            .contains(&"thumbnailImageData"),
        "the thumbnail is one of the derived renderings"
    );
    let error = doc
        .replace_media(image.identifier, &tiny_png(4, 4), "tiny.png", None)
        .unwrap_err();
    match &error {
        iwork::Error::NonDestructiveEdit { reasons, .. } => {
            assert!(
                reasons.iter().any(|r| r.contains("thumbnailImageData")),
                "{reasons:?}"
            );
        }
        other => panic!("expected a refusal by name, got {other}"),
    }
}

/// A table drawable has no object style: its field 2 is the table model, not a
/// style reference. Reading it as one reported a style that does not exist for
/// every table in the corpus.
#[test]
fn a_table_drawable_reports_no_object_style() {
    let path = fixture!("numbers-values.numbers");
    let doc = Document::open(&path).unwrap();
    let tables: Vec<_> = doc
        .drawables()
        .into_iter()
        .filter(|d| d.kind == Kind::Table)
        .collect();
    assert!(!tables.is_empty(), "the fixture has tables");
    for table in tables {
        assert_eq!(
            table.style, None,
            "table {} claimed a style it does not have",
            table.identifier
        );
    }
}

/// Media that lives in the app's theme bundle has no bytes in the document, so
/// there is nothing to replace and saying so is better than inventing an entry.
#[test]
fn replacing_a_theme_asset_says_where_it_lives() {
    let path = fixture!("pages-plain.pages");
    let mut doc = Document::open(&path).unwrap();
    let Some(asset) = doc
        .data_files()
        .into_iter()
        .find(|f| f.entry_name().is_none())
    else {
        eprintln!("no theme assets in this fixture — skipping");
        return;
    };
    let error = doc
        .replace_media(asset.identifier, &tiny_png(4, 4), "tiny.png", None)
        .unwrap_err()
        .to_string();
    assert!(error.contains("theme bundle"), "{error}");
}

/// A PNG of a given size, so the media tests carry no files.
fn tiny_png(width: u32, height: u32) -> Vec<u8> {
    fn crc32(data: &[u8]) -> u32 {
        let mut crc = 0xFFFF_FFFFu32;
        for byte in data {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }
    fn chunk(kind: &[u8], body: &[u8]) -> Vec<u8> {
        let mut out = (body.len() as u32).to_be_bytes().to_vec();
        let payload: Vec<u8> = kind.iter().chain(body).copied().collect();
        out.extend_from_slice(&payload);
        out.extend_from_slice(&crc32(&payload).to_be_bytes());
        out
    }
    // Uncompressed deflate blocks, so the file needs no compressor.
    let raw: Vec<u8> = (0..height)
        .flat_map(|_| std::iter::once(0u8).chain((0..width).flat_map(|_| [0x40u8, 0x80, 0xC0])))
        .collect();
    let mut deflate = vec![0x78, 0x01];
    for (i, block) in raw.chunks(65535).enumerate() {
        let last = u8::from((i + 1) * 65535 >= raw.len());
        deflate.push(last);
        deflate.extend_from_slice(&(block.len() as u16).to_le_bytes());
        deflate.extend_from_slice(&(!(block.len() as u16)).to_le_bytes());
        deflate.extend_from_slice(block);
    }
    let mut adler: (u32, u32) = (1, 0);
    for byte in &raw {
        adler.0 = (adler.0 + u32::from(*byte)) % 65521;
        adler.1 = (adler.1 + adler.0) % 65521;
    }
    deflate.extend_from_slice(&((adler.1 << 16) | adler.0).to_be_bytes());

    let mut header = width.to_be_bytes().to_vec();
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);

    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    png.extend_from_slice(&chunk(b"IHDR", &header));
    png.extend_from_slice(&chunk(b"IDAT", &deflate));
    png.extend_from_slice(&chunk(b"IEND", b""));
    png
}

#[test]
fn the_test_png_is_a_png() {
    let png = tiny_png(4, 6);
    assert_eq!(iwork::media::pixel_size(&png), Some((4.0, 6.0)));
}

// -- the oracle --------------------------------------------------------------

/// Ask the app for every rectangle and compare it, object by object.
///
/// Off unless `IWORK_APP_CHECK=1`. AppleScript exposes no object identifier, so
/// the comparison is made on the rectangles themselves: for every drawable this
/// crate reports, the app must report one with the same rectangle. That is the
/// claim being tested — that the composition rules for masks and rotation are
/// the app's rules and not this crate's inventions.
///
/// Four exclusions, each of them honest. A shape that **sizes itself to its
/// text** has no height in the archive at all: Keynote lays the box out and
/// reports 400 × 115 where the archive says 400 × 0 and puts the anchor 58
/// points lower, half the height it computed. A **mask** is not a separate
/// object to the app; it is how the image it belongs to is reported. A
/// **table** answers zeros to `position` and `width` — its rectangle lives in
/// `TST`. And anything on a **slide layout or a section template** is not on
/// the page the app enumerates.
#[test]
fn the_app_agrees_about_every_rectangle() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    for name in ["keynote-shapes.key", "pages-report.pages"] {
        let Some(path) = generated(name) else {
            eprintln!("no {name} — skipping");
            continue;
        };
        let said = ask_the_app(&path);
        assert!(!said.is_empty(), "{name}: the oracle reported nothing");

        let doc = Document::open(&path).unwrap();
        let all = doc.drawables();
        let by_id: std::collections::BTreeMap<u64, &iwork::Drawable> =
            all.iter().map(|d| (d.identifier, d)).collect();

        let mut compared = 0usize;
        for drawable in &all {
            // A table's rectangle comes from `TST` and the app reports zeros
            // for it through this interface, so there is nothing to compare.
            if matches!(drawable.kind, Kind::Mask | Kind::Table)
                || drawable.geometry.fits_its_text()
            {
                continue;
            }
            if !matches!(
                drawable.placement,
                iwork::Placement::Slide(_)
                    | iwork::Placement::Sheet(_)
                    | iwork::Placement::Floating { .. }
                    | iwork::Placement::InText { .. }
            ) {
                continue;
            }
            // A slide layout is a `KN.SlideArchive` too, and its drawables are
            // placed on it — but the app enumerates slides, not layouts.
            if drawable.stream.contains("TemplateSlide") {
                continue;
            }
            let mask = drawable.mask().and_then(|id| by_id.get(&id)).copied();
            let frame = drawable.frame(mask);
            let found = said.iter().any(|(_, x, y, width, height)| {
                // The app rounds every number to whole points.
                (frame.x - x).abs() < 1.0
                    && (frame.y - y).abs() < 1.0
                    && (frame.width - width).abs() < 1.0
                    && (frame.height - height).abs() < 1.0
            });
            assert!(
                found,
                "{name}: {} {} is at {:.1},{:.1} {:.1} × {:.1} here and the app reports \
                 nothing there: {said:?}",
                drawable.kind.as_str(),
                drawable.identifier,
                frame.x,
                frame.y,
                frame.width,
                frame.height
            );
            compared += 1;
        }
        // Pages contributes one — its photo, which is the whole point of the
        // masked-image rule; Keynote contributes nine.
        assert!(compared >= 1, "{name}: nothing was compared at all");
        eprintln!("{name}: {compared} rectangles agreed with the app");
    }
}

/// Move a shape and an image, save, and let the app read the rectangles back.
#[test]
fn keynote_reads_back_a_moved_drawable() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("keynote-shapes.key");
    let mut doc = Document::open(&path).unwrap();
    let shape = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Shape && d.geometry.width == 300.0)
        .unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image && d.mask().is_none())
        .unwrap();
    doc.set_geometry(shape.identifier, Some((211.0, 322.0)), Some((444.0, 128.0)))
        .unwrap();
    doc.set_geometry(image.identifier, Some((640.0, 240.0)), None)
        .unwrap();

    let out = std::env::temp_dir().join("iwork-set-geometry.key");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let said = ask_the_app(&out);
    assert!(
        said.iter().any(|(class, x, y, w, h)| class == "shape"
            && (*x - 211.0).abs() < 1.0
            && (*y - 322.0).abs() < 1.0
            && (*w - 444.0).abs() < 1.0
            && (*h - 128.0).abs() < 1.0),
        "Keynote did not report the moved shape: {said:?}"
    );
    assert!(
        said.iter().any(|(class, x, y, w, h)| class == "image"
            && (*x - 640.0).abs() < 1.0
            && (*y - 240.0).abs() < 1.0
            && (*w - 120.0).abs() < 1.0
            && (*h - 90.0).abs() < 1.0),
        "Keynote did not report the moved image: {said:?}"
    );
}

/// Swap an image's bytes and let Keynote open the result.
///
/// What this can and cannot prove is worth saying plainly: it proves the
/// document opens, that the picture is still 120 × 90 where it was, and that
/// the app is content with the registry entry. It cannot prove the *pixels* are
/// the new ones — nothing on a locked screen can see what is drawn.
#[test]
fn keynote_opens_a_document_whose_image_was_replaced() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("keynote-shapes.key");
    let mut doc = Document::open(&path).unwrap();
    let image = doc
        .drawables()
        .into_iter()
        .find(|d| d.kind == Kind::Image && d.mask().is_none())
        .unwrap();
    let frame = image.frame(None);
    doc.replace_media(image.identifier, &tiny_png(48, 36), "swapped.png", None)
        .unwrap();

    let out = std::env::temp_dir().join("iwork-replace-media.key");
    let _ = std::fs::remove_file(&out);
    doc.save(&out).unwrap();

    let said = ask_the_app(&out);
    assert!(
        said.iter().any(|(class, x, y, w, h)| class == "image"
            && (*x - frame.x).abs() < 1.0
            && (*y - frame.y).abs() < 1.0
            && (*w - frame.width).abs() < 1.0
            && (*h - frame.height).abs() < 1.0),
        "Keynote did not report the replaced image where it was: {said:?}"
    );
}

/// Run `scripts/drawable-oracle.sh` and parse its item lines.
fn ask_the_app(path: &Path) -> Vec<(String, f32, f32, f32, f32)> {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/drawable-oracle.sh");
    let output = std::process::Command::new(&script)
        .arg(path)
        .output()
        .unwrap_or_else(|e| panic!("{}: {e}", script.display()));
    assert!(
        output.status.success(),
        "the app would not open {}:\n{}",
        path.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let field: Vec<&str> = line.split('\t').collect();
            if field.first() != Some(&"item") || field.len() < 8 {
                return None;
            }
            let number = |at: usize| field[at].parse::<f32>().ok();
            Some((
                field[2].to_string(),
                number(3)?,
                number(4)?,
                number(5)?,
                number(6)?,
            ))
        })
        .collect()
}
