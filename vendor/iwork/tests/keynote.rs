//! The Keynote show: what a deck holds, and what the four writes do to it.
//!
//! Three claims, in the order they have to hold.
//!
//! **The deck this crate reads is the deck the app sees.** Behind
//! `IWORK_APP_CHECK=1`, `scripts/slide-oracle.sh` asks Keynote for the slide
//! count, the layout count, the slide size, every layout's name, and every
//! slide's number, base layout, skipped flag, title, body, presenter notes and
//! transition — and every one of those is compared. Keynote's dictionary is the
//! richest of the three and this is the phase that can afford to use all of it.
//!
//! **The shape is the shape in every deck.** A slide is its own component; a
//! placeholder's kind matches the field that names it; a note is a storage of
//! kind 4; the layouts a slide can be built on are the ones the theme lists.
//! Asserted over every `.key` in the corpus, not just the two the oracle runs
//! against.
//!
//! **A copied slide is a copied component, and the app opens it.** The
//! duplicate is the write that could break a document quietly, so it is checked
//! three ways: `iwork check` over the result, the two slides proved independent
//! by editing one, and — behind `IWORK_APP_CHECK=1` — Keynote asked to open the
//! deck and read the copy's title and notes back, then asked to *save* it, so
//! that what is on disk afterwards was written by Keynote from its own model.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use iwork::keynote::{PlaceholderKind, Role};
use iwork::{iwa, pb, Document, Package};

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

/// A password-protected package, whose object streams are ciphertext and which
/// `Document::open` refuses by design. Recognised by shape, not by name.
fn encrypted(path: &Path) -> bool {
    iwork::Package::read(path).is_ok_and(|package| package.contains(".iwpv2"))
}

/// The corpus is generated, never committed. A sweep over it is meaningless
/// where it is absent — on CI, and on a clean checkout before
/// `scripts/make-fixtures.sh` has run — so these tests skip rather than fail.
/// The "saw enough" assertions below keep their teeth wherever it *is*
/// present, which is where they were ever able to say anything.
macro_rules! corpus {
    () => {
        if keynote_fixtures().is_empty() {
            eprintln!("no .key corpus — skipping (run scripts/make-fixtures.sh)");
            return;
        }
    };
}

fn keynote_fixtures() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/generated");
    let Ok(entries) = std::fs::read_dir(&dir) else {
        return Vec::new();
    };
    let mut found: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("key"))
        .filter(|path| !encrypted(path))
        .collect();
    found.sort();
    found
}

fn script(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("scripts")
        .join(name)
}

fn temp(name: &str) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let _ = std::fs::remove_file(&path);
    path
}

// -- reading -----------------------------------------------------------------

/// Only Keynote has a show. A Pages or Numbers document must come back with
/// nothing rather than with something invented out of a colliding type number:
/// Numbers numbers its document archive 1 as well.
#[test]
fn only_a_keynote_document_has_a_show() {
    for name in ["numbers-values.numbers", "pages-report.pages"] {
        let Some(path) = generated(name) else {
            continue;
        };
        let doc = Document::open(&path).unwrap();
        assert!(doc.show().is_none(), "{name} reports a show");
        assert!(doc.slides().is_empty(), "{name} reports slides");
        assert!(doc.slide_layouts().is_empty(), "{name} reports layouts");
    }
}

/// Every deck: a theme with layouts, a slide size, and at least one slide.
#[test]
fn every_deck_has_a_theme_a_size_and_slides() {
    corpus!();
    let decks = keynote_fixtures();
    for path in decks {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap_or_else(|| panic!("{name}: no show"));
        assert!(!show.slides.is_empty(), "{name}: no slides");
        assert!(!show.layouts.is_empty(), "{name}: no slide layouts");
        assert!(
            show.width > 0.0 && show.height > 0.0,
            "{name}: slide size is {} × {}",
            show.width,
            show.height
        );
        assert!(!show.theme_name.is_empty(), "{name}: the theme has no name");
    }
}

/// The rules `iwork check` enforces, asserted directly so a failure names the
/// rule rather than a line of output: a layout that resolves, a placeholder
/// whose kind matches its field, a slide whose objects are its own.
#[test]
fn every_slide_keeps_the_shape_the_check_looks_for() {
    corpus!();
    let mut slides = 0usize;
    for path in keynote_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        let layouts: BTreeSet<u64> = show.layouts.iter().map(|l| l.identifier).collect();
        let mut streams: BTreeSet<String> = BTreeSet::new();
        for slide in &show.slides {
            slides += 1;
            assert!(
                layouts.contains(&slide.layout.unwrap_or(0)),
                "{name}: slide {} is built on {:?}, which the theme does not list",
                slide.identifier,
                slide.layout
            );
            assert!(
                streams.insert(slide.stream.clone()),
                "{name}: two slides in {}",
                slide.stream
            );
            for (place, wanted) in [
                (&slide.title, PlaceholderKind::Title),
                (&slide.body, PlaceholderKind::Body),
                (&slide.slide_number, PlaceholderKind::SlideNumber),
                (&slide.object, PlaceholderKind::Object),
            ] {
                if let Some(place) = place {
                    assert_eq!(
                        place.kind,
                        wanted,
                        "{name}: slide {} names {} as its {}",
                        slide.identifier,
                        place.identifier,
                        wanted.as_str()
                    );
                }
            }
            for text in &slide.texts {
                let (stream, _) = doc.object(text.storage).unwrap();
                assert_eq!(
                    stream, slide.stream,
                    "{name}: slide {} owns storage {} in another component",
                    slide.identifier, text.storage
                );
            }
        }
    }
    assert!(slides >= 15, "only {slides} slides were checked");
}

/// Presenter notes are a storage of kind 4, reached only through
/// `KN.NoteArchive`. Nothing else in a deck is kind 4.
#[test]
fn presenter_notes_are_the_only_storages_of_kind_four() {
    corpus!();
    let mut notes = 0usize;
    for path in keynote_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        let claimed: BTreeSet<u64> = show.slides.iter().filter_map(|s| s.note_storage).collect();
        let of_kind_four: BTreeSet<u64> = doc
            .storages()
            .into_iter()
            .filter(|s| s.kind == 4)
            .map(|s| s.identifier)
            .collect();
        assert_eq!(
            claimed, of_kind_four,
            "{name}: the note storages and the kind-4 storages are different sets"
        );
        notes += claimed.len();
    }
    assert!(notes >= 15, "only {notes} note storages were checked");
}

/// A skipped slide has no number, and the rest are numbered 1, 2, 3 … around
/// it. That is not a field: it is arithmetic over the deck, and the app agrees
/// by answering -1 for the skipped ones.
#[test]
fn numbering_counts_past_the_skipped_slides() {
    let path = fixture!("keynote-slides.key");
    let doc = Document::open(&path).unwrap();
    let show = doc.show().unwrap();
    let numbers: Vec<Option<usize>> = show.slides.iter().map(|s| s.number).collect();
    assert_eq!(
        numbers,
        vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(5),
            Some(6),
            None,
            None
        ],
        "the two skipped slides should have no number and not consume one"
    );
}

/// "Showing" is ownership. The app reports `title showing` false for a slide
/// whose field 5 still names a placeholder with text in it; what it means is
/// that the placeholder is not in `owned_drawables`.
#[test]
fn a_placeholder_is_shown_when_the_slide_owns_it() {
    let path = fixture!("keynote-deck.key");
    let doc = Document::open(&path).unwrap();
    let show = doc.show().unwrap();
    let statement = show
        .slides
        .iter()
        .find(|s| s.layout_name == "Statement")
        .expect("keynote-deck has a Statement slide");
    let title = statement.title.as_ref().expect("…which has a title field");
    assert!(
        !title.text.is_empty(),
        "the placeholder holds text the app does not draw"
    );
    assert!(!statement.title_showing(), "and is not shown");
    assert!(
        !statement.drawables.contains(&title.identifier),
        "because the slide does not own it"
    );
    assert!(statement.body_showing(), "while its body is shown");
}

/// The transition identifiers are the ones the app's dictionary lists, which is
/// what makes this an inventory rather than a guess.
#[test]
fn transitions_read_as_the_dictionary_names_them() {
    let path = fixture!("keynote-slides.key");
    let doc = Document::open(&path).unwrap();
    let effects: Vec<String> = doc
        .slides()
        .iter()
        .map(|s| s.transition.effect.clone())
        .collect();
    assert_eq!(
        effects,
        vec![
            "none",
            "apple:dissolve",
            "apple:push",
            "apple:magic-move-implied-motion-path",
            "apple:wipe",
            "com.apple.iWork.Keynote.KLNConfetti",
            "none",
            "none",
        ]
    );
    let push = &doc.slides()[2].transition;
    assert!(push.automatic, "slide 3 advances by itself");
    assert_eq!(push.delay, 3.0);
    assert_eq!(push.duration, 2.0);
    assert!(doc.slides()[0].transition.is_none());
}

/// The `custom_*` block belongs to the **effect**, and Keynote writes exactly
/// the parameters the chosen effect has.
///
/// `keynote-transitions` is 44 blank slides, one per effect in the app's own
/// enumeration, identical in every other respect — same layout, no text, same
/// duration, delay and automatic flag — so anything that differs between two of
/// them is the effect. Eleven of the 44 bring something; the other thirty-three
/// bring none, and that is the finding as much as the eleven are.
#[test]
fn every_transition_parameter_belongs_to_its_effect() {
    let path = fixture!("keynote-transitions.key");
    let doc = Document::open(&path).unwrap();
    let slides = doc.slides();
    let by_effect: BTreeMap<&str, &iwork::keynote::Transition> = slides
        .iter()
        .map(|s| (s.transition.effect.as_str(), &s.transition))
        .collect();
    assert_eq!(
        by_effect.len(),
        44,
        "all 44 of the app's effects, on 46 slides"
    );

    let p = |effect: &str| by_effect[effect].parameters;

    // Bounce is the object-effect checkbox, and it is written **whatever its
    // value** — false is a parameter the effect has, not a parameter it lacks.
    assert_eq!(p("apple:ca-dissolve-and-flip").bounce, Some(true));
    assert_eq!(p("apple:3D-cube").bounce, Some(true));
    assert_eq!(p("apple:revolve").bounce, Some(true));
    assert_eq!(
        p("com.apple.iWork.Keynote.BLTRevolvingDoor").bounce,
        Some(true)
    );
    assert_eq!(p("apple:ca-revolve").bounce, Some(false));
    assert_eq!(p("apple:scale").bounce, Some(false));
    assert_eq!(p("apple:dissolve").bounce, None, "dissolve has no bounce");

    assert_eq!(p("com.apple.iWork.Keynote.BUKTwist").twist, Some(3.3));
    assert_eq!(p("apple:fade-and-move").travel_distance, Some(1.0));
    assert_eq!(p("apple:radial wipe").angle, Some(90.0));
    assert_eq!(p("apple:radial wipe").blur_amount, Some(0.5));

    // Magic Move's three: fade unmatched, acceleration, text granularity.
    let magic = p("apple:magic-move-implied-motion-path");
    assert_eq!(magic.magic_move_fade_unmatched, Some(true));
    assert_eq!(
        magic.timing_curve,
        Some(iwork::keynote::TimingCurve::EaseInEaseOut)
    );
    assert_eq!(
        magic.text_delivery,
        Some(iwork::keynote::TextDelivery::ByObject)
    );

    // Fade Through Colour is the only effect of the 44 that writes a colour,
    // and it is black, opaque.
    let fade = by_effect["com.apple.iWork.Keynote.BLTFadeThruColor"];
    let colour = fade.color.expect("a colour to fade through");
    assert_eq!(
        (colour.red, colour.green, colour.blue, colour.alpha),
        (0.0, 0.0, 0.0, 1.0)
    );
    assert!(
        fade.parameters.is_empty(),
        "and no custom parameter besides"
    );

    let with_parameters = slides
        .iter()
        .filter(|s| !s.transition.parameters.is_empty() || s.transition.color.is_some())
        .count();
    assert_eq!(
        with_parameters, 12,
        "eleven of the 44 effects bring something, and magic move is on two slides"
    );
}

/// The parameters do not move when the timing does.
///
/// The last two slides of `keynote-transitions` repeat an effect at another
/// duration, delay and automatic flag. If the `custom_*` block were a function
/// of anything but the effect, this is where it would show.
#[test]
fn the_timing_does_not_disturb_the_parameters() {
    let path = fixture!("keynote-transitions.key");
    let doc = Document::open(&path).unwrap();
    let slides = doc.slides();
    let magic: Vec<&iwork::keynote::Transition> = slides
        .iter()
        .map(|s| &s.transition)
        .filter(|t| t.effect == "apple:magic-move-implied-motion-path")
        .collect();
    assert_eq!(magic.len(), 2);
    assert_ne!(magic[0].duration, magic[1].duration, "different timings");
    assert_ne!(magic[0].automatic, magic[1].automatic);
    assert_eq!(magic[0].parameters, magic[1].parameters, "same parameters");

    let wipe: Vec<&iwork::keynote::Transition> = slides
        .iter()
        .map(|s| &s.transition)
        .filter(|t| t.effect == "apple:wipe")
        .collect();
    assert_eq!(wipe.len(), 2);
    assert_ne!(wipe[0].duration, wipe[1].duration);
    assert!(wipe[0].parameters.is_empty() && wipe[1].parameters.is_empty());
}

/// **Keynote's own writer never emits a direction.** All 44 effects leave field
/// 4 absent, which reads as 0 — "whatever this effect does by default".
///
/// The values [`iwork::keynote::direction_name`] knows come from a PowerPoint
/// import instead (FORMAT.md §13); nothing in this corpus carries one, and this
/// test is what says so.
#[test]
fn the_app_writes_no_transition_direction() {
    for path in keynote_fixtures() {
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        for transition in show
            .slides
            .iter()
            .map(|s| &s.transition)
            .chain(show.layouts.iter().map(|l| &l.transition))
        {
            assert_eq!(
                transition.direction,
                0,
                "{}: {} has a direction",
                path.display(),
                transition.effect
            );
        }
    }
    assert_eq!(
        iwork::keynote::direction_name(0),
        "the effect's own default"
    );
}

/// The playback settings a script can move, read out of the deck that moves
/// them. `keynote-playback` differs from every other deck here in exactly four
/// fields — and in a fifth number that is not a field at all.
#[test]
fn playback_settings_are_four_fields_of_the_show() {
    let path = fixture!("keynote-playback.key");
    let control = fixture!("keynote-slides.key");
    let moved = Document::open(&path).unwrap().show().unwrap();
    let default = Document::open(&control).unwrap().show().unwrap();

    assert!(moved.loop_presentation && !default.loop_presentation);
    assert!(moved.plays_on_open && !default.plays_on_open);
    assert!(moved.idle_timer_active && !default.idle_timer_active);

    // **Seconds on the wire, minutes in the dictionary.** The script asked for
    // 137 and the field holds 8220.
    assert_eq!(moved.idle_timer_delay, 8220.0);
    assert_eq!(moved.idle_minutes(), 137.0);
    assert_eq!(
        default.idle_timer_delay, 900.0,
        "the schema default, 15 min"
    );

    // The three nothing can move stay where they are, in both decks — written
    // explicitly, at their defaults.
    for show in [&moved, &default] {
        assert_eq!(show.mode, 0);
        assert_eq!(show.mode_name(), "normal");
        assert_eq!(show.autoplay_transition_delay, 5.0);
        assert_eq!(show.autoplay_build_delay, 2.0);
    }
}

/// Every deck has a soundtrack and every soundtrack is empty — and nothing
/// available can fill one in. `make new audio clip` is *accepted* by Keynote
/// and then does nothing: no error, no clip, no `Data/` entry.
#[test]
fn the_soundtrack_is_present_and_empty_everywhere() {
    corpus!();
    let mut decks = 0;
    for path in keynote_fixtures() {
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        let track = show
            .soundtrack
            .as_ref()
            .unwrap_or_else(|| panic!("{}: no soundtrack", path.display()));
        assert_eq!(track.volume, 1.0, "{}", path.display());
        assert_eq!(track.mode, 0, "{}", path.display());
        assert_eq!(track.mode_name(), "play once");
        assert_eq!(track.tracks(), 0, "{}", path.display());
        assert!(track.media.is_empty());
        decks += 1;
    }
    assert!(decks >= 4, "only {decks} decks");
}

/// A recorded presentation and a live-video feed are the two never-author
/// cases. There is no recording anywhere; there is one live-video source in
/// every deck, and it is the theme's default camera.
#[test]
fn recordings_and_live_video_are_identified_not_authored() {
    corpus!();
    let mut decks = 0;
    for path in keynote_fixtures() {
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        assert!(
            show.recording.is_none(),
            "{}: a recording turned up — measure it before trusting the decode",
            path.display()
        );
        assert_eq!(show.live_video_sources.len(), 1, "{}", path.display());
        let source = &show.live_video_sources[0];
        assert_eq!(source.name, "Default Camera");
        assert!(source.is_default);
        assert!(
            !source.listed,
            "it is the collection's default_source and in no sources list"
        );
        decks += 1;
    }
    assert!(decks >= 4);
}

/// **The build tripwire.** There is no `KN.BuildArchive` in any document this
/// crate has been shown, so [`iwork::keynote::Build`] is decoded from the
/// 15.3.1 schema and has never been measured. The day a fixture grows one, this
/// fails — which is the signal to go and measure it rather than to trust it.
#[test]
fn builds_exist_exactly_where_they_were_added() {
    for path in keynote_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        if name != "keynote-builds.key" {
            assert_eq!(show.build_count(), 0, "{name} grew a build");
            continue;
        }
        // The tripwire fired: builds measured at last, from the Animate
        // inspector on an unlocked screen. Four Dissolve build-ins and four
        // Disappear build-outs — and Disappear's stored identifier is
        // `apple:bc-appear`, the direction living in `animation_type`.
        let builds: Vec<_> = show.slides.iter().flat_map(|s| &s.builds).collect();
        assert_eq!(builds.len(), 8, "{name}");
        let ins: Vec<_> = builds
            .iter()
            .filter(|b| b.animation.animation_type == "In")
            .collect();
        let outs: Vec<_> = builds
            .iter()
            .filter(|b| b.animation.animation_type == "Out")
            .collect();
        assert_eq!(ins.len(), 4, "{name}");
        assert_eq!(outs.len(), 4, "{name}");
        for b in &ins {
            assert_eq!(b.animation.effect, "apple:dissolve character", "{name}");
            assert!(
                b.drawable.is_some(),
                "{name}: a build with nothing to build"
            );
        }
        for b in &outs {
            assert_eq!(b.animation.effect, "apple:bc-appear", "{name}");
        }
        assert_eq!(
            doc.objects().filter(|(_, o)| o.message_type() == 8).count(),
            8,
            "{name}: every build archive is listed by a slide"
        );
    }
}

/// **The transition-parameter tripwire.** Every field of
/// `KN.TransitionAttributesArchive` the 15.3.1 schema names is decoded; a field
/// it does not name means the table has gone out of date. Field 14 is a hole in
/// the schema and would be caught here too.
#[test]
fn no_transition_carries_a_parameter_the_schema_does_not_name() {
    corpus!();
    let mut checked = 0usize;
    for path in keynote_fixtures() {
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        for transition in show
            .slides
            .iter()
            .map(|s| &s.transition)
            .chain(show.layouts.iter().map(|l| &l.transition))
        {
            assert!(
                transition.unknown_parameters.is_empty(),
                "{}: {} carries parameter field(s) {:?} the schema does not name",
                path.display(),
                transition.effect,
                transition.unknown_parameters
            );
            checked += 1;
        }
    }
    assert!(checked > 100, "only {checked} transitions");
}

/// The document-level "slide numbers showing" is not a document field: the app
/// writes `isSlideNumberVisible` on every node and leaves
/// `KN.ShowArchive.slideNumbersVisible` absent.
#[test]
fn slide_numbers_are_a_flag_on_every_node() {
    let on = fixture!("keynote-slides.key");
    let off = fixture!("keynote-deck.key");
    let with = Document::open(&on).unwrap().show().unwrap();
    let without = Document::open(&off).unwrap().show().unwrap();
    assert_eq!(with.numbers_shown_on(), with.slides.len());
    assert_eq!(without.numbers_shown_on(), 0);
    assert!(
        !with.slide_numbers_visible,
        "the show's own field stays absent even when the numbers are on"
    );
}

/// Every layout is named, and the theme lists them in the order the app numbers
/// them — which is what lets a slide's layout be reported by name.
#[test]
fn every_layout_has_a_name_and_a_place() {
    for path in keynote_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();
        for (index, layout) in show.layouts.iter().enumerate() {
            assert_eq!(layout.index, index, "{name}: layouts out of order");
            assert!(
                !layout.name.is_empty(),
                "{name}: layout {} has no name",
                layout.identifier
            );
            assert_eq!(
                layout.body_paragraph_styles.len(),
                5,
                "{name}: layout \"{}\" should have five outline levels",
                layout.name
            );
        }
    }
}

/// Roles are assigned once each. A storage that is a title is not also a text
/// box, and the free text box on a slide with placeholders is told apart from
/// them.
#[test]
fn a_storage_has_exactly_one_role() {
    let path = fixture!("keynote-slides.key");
    let doc = Document::open(&path).unwrap();
    let show = doc.show().unwrap();
    for slide in &show.slides {
        let mut seen: BTreeMap<u64, Role> = BTreeMap::new();
        for text in &slide.texts {
            assert!(
                seen.insert(text.storage, text.role).is_none(),
                "storage {} has two roles on slide {}",
                text.storage,
                slide.identifier
            );
        }
    }
    let free = show
        .slides
        .iter()
        .flat_map(|s| &s.texts)
        .find(|t| t.text == "Eine freie Textbox")
        .expect("keynote-slides has a text item that is not a placeholder");
    assert_eq!(free.role, Role::TextBox);
}

/// Every `KN` archive re-encodes to the bytes it was decoded from. The claim
/// this whole module stands on: a field this crate does not understand is a
/// field it does not lose.
#[test]
fn every_keynote_archive_re_encodes_to_itself() {
    corpus!();
    let mut checked = 0usize;
    for path in keynote_fixtures() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let doc = Document::open(&path).unwrap();
        for (_, object) in doc.objects() {
            // The app-scoped range is the KN one, and the only one where a
            // Keynote number means something Keynote-specific.
            if object.message_type() >= 200 {
                continue;
            }
            let decoded = iwork::pb::Message::decode(object.payload())
                .unwrap_or_else(|e| panic!("{name}: object {}: {e}", object.identifier));
            assert_eq!(
                decoded.encode(),
                object.payload(),
                "{name}: object {} (type {}) does not re-encode to itself",
                object.identifier,
                object.message_type()
            );
            checked += 1;
        }
    }
    assert!(checked >= 100, "only {checked} KN archives were re-encoded");
}

// -- writing -----------------------------------------------------------------

/// Skipping is one varint on the node, and it rewrites one stream.
#[test]
fn skipping_a_slide_touches_the_document_stream_and_nothing_else() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    assert!(!doc.show().unwrap().slides[0].skipped);

    assert!(doc.set_slide_skipped(slide, true).unwrap());
    assert!(
        !doc.set_slide_skipped(slide, true).unwrap(),
        "setting it again changes nothing"
    );
    assert_eq!(doc.changed_streams(), vec!["Index/Document.iwa"]);

    let show = doc.show().unwrap();
    assert!(show.slides[0].skipped);
    assert_eq!(show.slides[0].number, None, "and it loses its number");
    assert_eq!(show.slides[1].number, Some(1), "which the next one takes");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    assert!(doc.set_slide_skipped(slide, false).unwrap());
    assert!(
        doc.changed_streams().is_empty(),
        "and unskipping puts the bytes back exactly"
    );
}

/// Reordering is a permutation of the slide tree and touches nothing else — not
/// the nodes, not the slide components.
#[test]
fn moving_a_slide_permutes_the_tree_and_leaves_the_slides_alone() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let before: Vec<u64> = doc.slides().iter().map(|s| s.identifier).collect();
    let moved = before[0];

    assert_eq!(doc.move_slide(moved, 2).unwrap(), 2);
    assert_eq!(doc.changed_streams(), vec!["Index/Document.iwa"]);
    let after: Vec<u64> = doc.slides().iter().map(|s| s.identifier).collect();
    assert_eq!(
        after,
        vec![before[1], before[2], before[0], before[3], before[4], before[5]]
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    assert_eq!(doc.move_slide(moved, 0).unwrap(), 0);
    assert!(
        doc.changed_streams().is_empty(),
        "moving it back reproduces the original bytes"
    );
}

/// Presenter notes are a storage, and writing one goes through the Phase 4
/// remapper like any other text.
#[test]
fn presenter_notes_are_written_like_any_other_text() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    doc.set_presenter_notes(slide, "Ganz neue Notizen mit Größe 🎬")
        .unwrap();
    assert_eq!(
        doc.show().unwrap().slides[0].notes,
        "Ganz neue Notizen mit Größe 🎬"
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert_eq!(doc.changed_streams(), vec!["Index/Slide-2652176.iwa"]);
}

/// A slide with no notes is named rather than given some.
#[test]
fn a_slide_without_notes_refuses_rather_than_inventing_a_storage() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    assert!(doc.set_presenter_notes(999_999_999, "…").is_err());
}

/// `KN.SlideNodeArchive.hasNote` (8), read straight off the node.
fn has_note(doc: &Document, node: u64) -> bool {
    let (_, object) = doc.object(node).expect("the node exists");
    pb::Message::decode(object.payload())
        .unwrap()
        .varint(8)
        .is_some_and(|v| v != 0)
}

/// Writing presenter notes keeps the node's `hasNote` flag in step with the
/// text: setting text sets it, clearing to "" unsets it. Across the corpus the
/// flag is 1 exactly when the note storage is non-empty, and `set-notes` used
/// to leave it blind either way.
#[test]
fn setting_and_clearing_notes_moves_hasnote() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    let node = doc.show().unwrap().slide(slide).unwrap().node;
    assert!(
        has_note(&doc, node),
        "keynote-deck ships with notes on every slide"
    );

    // Clearing the note drops the flag — and rewrites the node's stream to say so.
    doc.set_presenter_notes(slide, "").unwrap();
    assert!(!has_note(&doc, node), "cleared notes drop hasNote");
    assert!(
        doc.changed_streams().contains(&"Index/Document.iwa"),
        "clearing rewrites the node in the document stream"
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    // Writing text back sets it again.
    doc.set_presenter_notes(slide, "Wieder Notizen").unwrap();
    assert!(has_note(&doc, node), "text sets hasNote");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// A slide layout has a node too, so `node_of` resolves its id — but skipping
/// it is refused, because the flag would land in a theme template and the CLI
/// would report a layout as a skipped slide. The theme node is left untouched.
#[test]
fn skipping_a_layout_is_refused() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let layout = doc.slide_layouts()[0].identifier;
    let err = doc.set_slide_skipped(layout, true).unwrap_err().to_string();
    assert!(err.contains("not a slide"), "{err}");
    assert!(
        doc.changed_streams().is_empty(),
        "the theme node is untouched"
    );
}

/// An out-of-range target is refused, not clamped: `move … 99` on a six-slide
/// deck used to report "now at position 5".
#[test]
fn moving_a_slide_past_the_end_is_refused() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    let err = doc.move_slide(slide, 99).unwrap_err().to_string();
    assert!(err.contains("past the end"), "{err}");
    assert!(doc.changed_streams().is_empty(), "nothing moved");
}

/// Rewrite the show's inline `KN.SlideTreeArchive` (field 3 of the show) with
/// `edit`, and hand back a package carrying the result — the forge every
/// slide-tree test shares.
fn forge_slide_tree(path: &Path, edit: impl FnOnce(&mut pb::Message)) -> Package {
    let mut package = Package::read(path).unwrap();
    let entry = package
        .entries
        .iter_mut()
        .find(|(name, _)| name == "Index/Document.iwa")
        .expect("a deck has Index/Document.iwa");
    let mut objects = iwa::parse(&entry.1).unwrap();
    let mut done = false;
    for object in &mut objects {
        if object.message_type() != 2 {
            continue;
        }
        let mut show = pb::Message::decode(object.payload()).unwrap();
        let raw = show.bytes(3).expect("the show has a slide tree").to_vec();
        let mut tree = pb::decode_nested(&raw).expect("the tree decodes");
        edit(&mut tree);
        show.set(3, pb::Value::Bytes(tree.encode()));
        object.messages[0].payload = show.encode();
        done = true;
        break;
    }
    assert!(done, "no KN.ShowArchive to forge");
    entry.1 = iwa::serialize(&objects);
    package
}

/// Drop the `TSP.ComponentInfo` that names `victim` from wherever the package
/// metadata lives, leaving the slide's stream declared by nobody.
fn forge_out_component(path: &Path, victim: u64) -> Package {
    let mut package = Package::read(path).unwrap();
    for (name, data) in package.entries.iter_mut() {
        if !name.ends_with(".iwa") {
            continue;
        }
        let Ok(mut objects) = iwa::parse(data) else {
            continue;
        };
        let mut touched = false;
        for object in &mut objects {
            if object.message_type() != iwork::TYPE_PACKAGE_METADATA {
                continue;
            }
            let mut meta = pb::Message::decode(object.payload()).unwrap();
            let before = meta.fields.len();
            meta.fields.retain(|f| {
                if f.number != 3 {
                    return true;
                }
                match &f.value {
                    pb::Value::Bytes(raw) => match pb::decode_nested(raw) {
                        Some(info) => info.varint(1) != Some(victim),
                        None => true,
                    },
                    _ => true,
                }
            });
            if meta.fields.len() != before {
                object.messages[0].payload = meta.encode();
                touched = true;
            }
        }
        if touched {
            *data = iwa::serialize(&objects);
        }
    }
    package
}

/// A slide-tree entry that is not a bare `{1: node}` reference is refused, not
/// silently dropped — a reorder used to rebuild the tree from the entries it
/// recognised and lose the odd one, and its slide with it.
#[test]
fn moving_refuses_a_tree_entry_it_cannot_reproduce() {
    let path = fixture!("keynote-deck.key");
    let package = forge_slide_tree(&path, |tree| {
        // Give the first slide entry an extra field, so it is no longer bare.
        let first = tree
            .fields
            .iter_mut()
            .find(|f| f.number == 2)
            .expect("the tree lists slides");
        let pb::Value::Bytes(raw) = &first.value else {
            panic!("a slide entry is bytes");
        };
        let mut entry = pb::decode_nested(raw).unwrap();
        entry.set(2, pb::Value::Varint(0));
        first.value = pb::Value::Bytes(entry.encode());
    });
    let mut doc = Document::from_package(package).unwrap();
    let slide = doc.slides()[0].identifier;
    let err = doc.move_slide(slide, 1).unwrap_err().to_string();
    assert!(err.contains("bare reference"), "{err}");
}

/// A slide the deck's tree does not list cannot be duplicated: membership of
/// the slide tree is what tells a slide from a template, not the stream name.
#[test]
fn duplicating_a_node_the_tree_does_not_list_is_refused() {
    let path = fixture!("keynote-deck.key");
    let doc = Document::open(&path).unwrap();
    let victim = doc.slides()[2].identifier;
    let victim_node = doc.show().unwrap().slide(victim).unwrap().node;
    drop(doc);

    // Drop the slide's tree entry, but keep its node, slide and component: it
    // is now a slide the deck does not list.
    let package = forge_slide_tree(&path, |tree| {
        tree.fields.retain(|f| {
            if f.number != 2 {
                return true;
            }
            match &f.value {
                pb::Value::Bytes(raw) => {
                    match pb::decode_nested(raw).and_then(|e| iwork::style::reference_target(&e)) {
                        Some(node) => node != victim_node,
                        None => true,
                    }
                }
                _ => true,
            }
        });
    });
    let mut doc = Document::from_package(package).unwrap();
    let err = doc.duplicate_slide(victim).unwrap_err().to_string();
    assert!(err.contains("not a slide in the show's deck"), "{err}");
}

/// A slide whose `TSP.ComponentInfo` is missing cannot be duplicated: the copy
/// would be declared by nobody and Keynote would answer `missing value` for it.
/// `problems()` reports the same gap on the source deck, which the old
/// component→stream check could not see.
#[test]
fn duplicating_a_slide_whose_component_is_missing_is_refused() {
    let path = fixture!("keynote-deck.key");
    let victim = Document::open(&path).unwrap().slides()[1].identifier;
    let package = forge_out_component(&path, victim);

    let doc = Document::from_package(package.clone()).unwrap();
    assert!(
        doc.problems()
            .iter()
            .any(|p| p.contains("no TSP.ComponentInfo")),
        "check flags the missing component: {:?}",
        doc.problems()
    );

    let mut doc = Document::from_package(package).unwrap();
    let err = doc.duplicate_slide(victim).unwrap_err().to_string();
    assert!(err.contains("ComponentInfo"), "{err}");
}

/// A cloned `TSP.ComponentInfo`'s fields stay in ascending order — the locator
/// (field 3) is written last but belongs between 2 and 4, so it is inserted in
/// order rather than appended past field 12.
#[test]
fn a_cloned_component_keeps_its_fields_in_order() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let copy = doc.duplicate_slide(doc.slides()[1].identifier).unwrap();

    // Find the new component and read its field numbers in the order stored.
    // Read from the edited streams (`objects`), not the original package.
    let meta = doc
        .objects()
        .find(|(_, o)| o.message_type() == iwork::TYPE_PACKAGE_METADATA)
        .map(|(_, o)| pb::Message::decode(o.payload()).unwrap())
        .expect("package metadata");
    let info = meta
        .all(3)
        .filter_map(|v| match v {
            pb::Value::Bytes(raw) => pb::Message::decode(raw).ok(),
            _ => None,
        })
        .find(|info| info.varint(1) == Some(copy.identifier))
        .expect("the copy's component");
    let numbers: Vec<u32> = info.fields.iter().map(|f| f.number).collect();
    let mut sorted = numbers.clone();
    sorted.sort_unstable();
    assert_eq!(
        numbers, sorted,
        "component fields are ascending: {numbers:?}"
    );
    assert!(numbers.contains(&3), "the locator was written");
}

/// A layout is not a slide, and copying one is refused by name — Keynote's own
/// dictionary will not do it either.
#[test]
fn a_slide_layout_cannot_be_duplicated() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let layout = doc.slide_layouts()[0].identifier;
    let error = doc.duplicate_slide(layout).unwrap_err().to_string();
    assert!(error.contains("layout"), "{error}");
}

/// The copy is a whole component: its own stream, its own objects, its own
/// entry in the package metadata, and nothing shared with the original except
/// what the original itself shares — the layout, the stylesheet, the media.
#[test]
fn a_duplicated_slide_is_a_duplicated_component() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let source = doc.slides()[1].clone();
    let before = doc.slides().len();

    let copy = doc.duplicate_slide(source.identifier).unwrap();
    assert_eq!(copy.index, 2, "the copy goes straight after its original");
    assert_eq!(doc.slides().len(), before + 1);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    let show = doc.show().unwrap();
    let made = &show.slides[2];
    assert_eq!(made.identifier, copy.identifier);
    assert_ne!(made.stream, source.stream, "its own component");
    assert_eq!(
        made.layout, source.layout,
        "the same layout as the original"
    );
    assert_eq!(made.title_text(), source.title_text());
    assert_eq!(made.body_text(), source.body_text());
    assert_eq!(made.notes, source.notes);
    assert_eq!(made.transition, source.transition);

    // Not one identifier in common: a copy that shared an object would edit
    // the original when the copy was edited.
    let mine: BTreeSet<u64> = made.texts.iter().map(|t| t.storage).collect();
    let theirs: BTreeSet<u64> = source.texts.iter().map(|t| t.storage).collect();
    assert!(
        mine.is_disjoint(&theirs),
        "the copy shares storages with the original: {:?}",
        mine.intersection(&theirs).collect::<Vec<_>>()
    );

    // Only three streams are rewritten: the new one, the document (node and
    // slide tree) and the metadata (the component entry).
    let mut changed = doc.changed_streams();
    changed.sort_unstable();
    assert_eq!(
        changed,
        vec![
            "Index/Document.iwa",
            "Index/Metadata.iwa",
            copy.stream.as_str()
        ]
    );

    // And the two are independent.
    doc.set_text(
        made.title.as_ref().unwrap().storage.unwrap(),
        "Nur die Kopie",
    )
    .unwrap();
    let show = doc.show().unwrap();
    assert_eq!(show.slides[2].title_text(), "Nur die Kopie");
    assert_eq!(show.slides[1].title_text(), source.title_text());
}

/// The identifiers a copy takes are above the package's high-water mark, and
/// the mark moves with them — otherwise iWork would hand the same number out
/// again and two objects would be one.
#[test]
fn a_copy_allocates_above_the_high_water_mark() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let mark = doc.last_object_identifier().unwrap();
    let copy = doc.duplicate_slide(doc.slides()[1].identifier).unwrap();
    assert!(copy.identifier > mark);
    assert!(doc.last_object_identifier().unwrap() >= copy.node);
    assert!(
        doc.next_object_identifier() > copy.node,
        "and the next allocation is above the copy"
    );
}

/// A copied image slide shares the original's media rather than growing the
/// package — which is what the app's own duplicate does.
#[test]
fn a_copied_image_slide_shares_its_media() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let with_media = doc
        .slides()
        .iter()
        .find(|s| {
            s.drawables
                .iter()
                .any(|d| doc.drawable(*d).is_some_and(|d| d.media.is_some()))
        })
        .expect("keynote-deck has an image slide")
        .identifier;
    let files = doc.data_files().len();
    let copy = doc.duplicate_slide(with_media).unwrap();
    assert!(copy.media > 0, "the copy declares the media it uses");
    assert_eq!(
        doc.data_files().len(),
        files,
        "and no Data/ entry was added for it"
    );
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// Two copies of the same slide are two slides, not one — the second must not
/// reuse the first's identifiers or its stream.
#[test]
fn a_slide_can_be_copied_twice() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let source = doc.slides()[1].identifier;
    let first = doc.duplicate_slide(source).unwrap();
    let second = doc.duplicate_slide(source).unwrap();
    assert_ne!(first.identifier, second.identifier);
    assert_ne!(first.stream, second.stream);
    assert_eq!(doc.slides().len(), 8);
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The duplicate is reproducible: the same document copied the same way twice
/// gives the same bytes. Object UUIDs are derived rather than drawn for exactly
/// this reason.
#[test]
fn duplicating_a_slide_is_reproducible() {
    let path = fixture!("keynote-deck.key");
    let out_one = temp("iwork-kn-dup-1.key");
    let out_two = temp("iwork-kn-dup-2.key");
    for out in [&out_one, &out_two] {
        let mut doc = Document::open(&path).unwrap();
        doc.duplicate_slide(doc.slides()[1].identifier).unwrap();
        doc.save(out).unwrap();
    }
    assert_eq!(
        std::fs::read(&out_one).unwrap(),
        std::fs::read(&out_two).unwrap(),
        "two identical duplicates produced different files"
    );
    let _ = std::fs::remove_file(&out_one);
    let _ = std::fs::remove_file(&out_two);
}

// -- the app -----------------------------------------------------------------

/// Ask Keynote for the deck: `show`, `layout` and `slide` records, tab
/// separated, as `scripts/applescript/slide-oracle.applescript` documents.
fn ask_the_app(path: &Path) -> Vec<Vec<String>> {
    let output = std::process::Command::new(script("slide-oracle.sh"))
        .arg(path)
        .output()
        .unwrap_or_else(|e| panic!("slide-oracle.sh: {e}"));
    assert!(
        output.status.success(),
        "slide-oracle.sh failed for {path:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .map(|line| line.split('\t').map(str::to_string).collect())
        .collect()
}

/// The app escapes tabs and line breaks inside a text so that one slide is one
/// line; this puts them back. Carriage return and linefeed stay apart, because
/// **Keynote separates a body's paragraphs with a carriage return** and folding
/// the two together would hide whether this crate reads that correctly.
fn unescape(text: &str) -> String {
    text.replace("\\n", "\n")
        .replace("\\r", "\r")
        .replace("\\t", "\t")
}

/// Everything Keynote will say about a deck, compared with everything this
/// crate says about it. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn the_app_agrees_about_every_slide() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let mut compared = 0usize;
    for name in [
        "keynote-deck.key",
        "keynote-slides.key",
        "keynote-playback.key",
        "keynote-transitions.key",
    ] {
        let Some(path) = generated(name) else {
            eprintln!("no {name} — skipping");
            continue;
        };
        let said = ask_the_app(&path);
        let doc = Document::open(&path).unwrap();
        let show = doc.show().unwrap();

        let header = said.iter().find(|r| r[0] == "show").expect("a show record");
        assert_eq!(
            header[1].parse::<usize>().unwrap(),
            show.slides.len(),
            "{name}: slide count"
        );
        assert_eq!(
            header[2].parse::<usize>().unwrap(),
            show.layouts.len(),
            "{name}: layout count"
        );
        assert_eq!(
            header[3].parse::<f32>().unwrap(),
            show.width,
            "{name}: slide width"
        );
        assert_eq!(
            header[4].parse::<f32>().unwrap(),
            show.height,
            "{name}: slide height"
        );
        assert_eq!(
            header[5] == "true",
            show.numbers_shown_on() == show.slides.len(),
            "{name}: slide numbers showing"
        );

        // The four playback settings the dictionary exposes. The fourth is the
        // one worth the round trip: `maximum idle duration` is in **minutes**
        // and the field is in seconds.
        let playback = said
            .iter()
            .find(|r| r[0] == "playback")
            .expect("a playback record");
        assert_eq!(
            playback[1] == "true",
            show.loop_presentation,
            "{name}: auto loop"
        );
        assert_eq!(
            playback[2] == "true",
            show.plays_on_open,
            "{name}: auto play"
        );
        assert_eq!(
            playback[3] == "true",
            show.idle_timer_active,
            "{name}: auto restart"
        );
        assert_eq!(
            playback[4].parse::<f64>().unwrap(),
            show.idle_minutes(),
            "{name}: maximum idle duration, in minutes"
        );
        compared += 4;

        let layouts: Vec<&Vec<String>> = said.iter().filter(|r| r[0] == "layout").collect();
        assert_eq!(layouts.len(), show.layouts.len());
        for (record, layout) in layouts.iter().zip(&show.layouts) {
            assert_eq!(
                record[1].parse::<usize>().unwrap(),
                layout.index + 1,
                "{name}: layout order"
            );
            assert_eq!(record[2], layout.name, "{name}: layout name");
            compared += 1;
        }

        let slides: Vec<&Vec<String>> = said.iter().filter(|r| r[0] == "slide").collect();
        assert_eq!(slides.len(), show.slides.len(), "{name}: slide count");
        for (record, slide) in slides.iter().zip(&show.slides) {
            let number: i64 = record[1].parse().unwrap();
            assert_eq!(
                number,
                slide.number.map(|n| n as i64).unwrap_or(-1),
                "{name}: slide {} number",
                slide.identifier
            );
            assert_eq!(
                record[2], slide.layout_name,
                "{name}: slide {} layout",
                slide.identifier
            );
            assert_eq!(
                record[3] == "true",
                slide.skipped,
                "{name}: slide {} skipped",
                slide.identifier
            );
            assert_eq!(
                record[4] == "true",
                slide.title_showing(),
                "{name}: slide {} title showing",
                slide.identifier
            );
            assert_eq!(
                record[5] == "true",
                slide.body_showing(),
                "{name}: slide {} body showing",
                slide.identifier
            );
            if record[6] != "-" {
                assert_eq!(
                    unescape(&record[6]),
                    slide.title_text(),
                    "{name}: slide {} title",
                    slide.identifier
                );
            }
            if record[7] != "-" {
                assert_eq!(
                    unescape(&record[7]),
                    slide.body_text(),
                    "{name}: slide {} body",
                    slide.identifier
                );
            }
            assert_eq!(
                unescape(&record[8]),
                slide.notes,
                "{name}: slide {} presenter notes",
                slide.identifier
            );
            // The app names an effect in English and the document names it by
            // identifier; `keynote::EFFECTS` is the app's own pairing, and this
            // is where it is *checked* — 46 slides of `keynote-transitions`
            // carry all 44 of them.
            assert_eq!(
                record[9] == "no transition effect",
                slide.transition.is_none(),
                "{name}: slide {} transition",
                slide.identifier
            );
            assert_eq!(
                iwork::keynote::effect_name(&slide.transition.effect),
                Some(record[9].as_str()),
                "{name}: slide {} effect {} is not what the app calls it",
                slide.identifier,
                slide.transition.effect
            );
            if !slide.transition.is_none() {
                assert_eq!(
                    record[12].parse::<f64>().unwrap(),
                    slide.transition.duration,
                    "{name}: slide {} transition duration",
                    slide.identifier
                );
                assert_eq!(
                    record[11].parse::<f64>().unwrap(),
                    slide.transition.delay,
                    "{name}: slide {} transition delay",
                    slide.identifier
                );
                assert_eq!(
                    record[10] == "true",
                    slide.transition.automatic,
                    "{name}: slide {} automatic transition",
                    slide.identifier
                );
            }
            compared += 1;
        }
    }
    assert!(compared >= 40, "only {compared} records were compared");
}

/// The four writes, put through Keynote and read back. Off unless
/// `IWORK_APP_CHECK=1`.
///
/// One document carries all four so that one pass of the app answers for all of
/// them, and so that they are shown not to interfere: the copy is made, then a
/// slide is skipped, another unskipped, one moved and the notes rewritten.
#[test]
fn keynote_reads_back_every_write() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("keynote-deck.key");
    let out = temp("iwork-keynote-writes.key");

    let mut doc = Document::open(&path).unwrap();
    let slides: Vec<u64> = doc.slides().iter().map(|s| s.identifier).collect();
    let skipped = doc.slides().iter().find(|s| s.skipped).unwrap().identifier;

    let copy = doc.duplicate_slide(slides[1]).unwrap();
    doc.set_presenter_notes(copy.identifier, "Notizen nur auf der Kopie.")
        .unwrap();
    doc.set_text(
        doc.show()
            .unwrap()
            .slide(copy.identifier)
            .unwrap()
            .title
            .as_ref()
            .unwrap()
            .storage
            .unwrap(),
        "Kopie mit eigenem Titel",
    )
    .unwrap();
    doc.set_slide_skipped(slides[0], true).unwrap();
    doc.set_slide_skipped(skipped, false).unwrap();
    doc.move_slide(slides[5], 0).unwrap();
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    doc.save(&out).unwrap();

    let said = ask_the_app(&out);
    let slides: Vec<&Vec<String>> = said.iter().filter(|r| r[0] == "slide").collect();
    assert_eq!(slides.len(), 7, "Keynote should see one more slide");

    let doc = Document::open(&out).unwrap();
    let show = doc.show().unwrap();
    for (record, slide) in slides.iter().zip(&show.slides) {
        assert_eq!(record[2], slide.layout_name);
        assert_eq!(record[3] == "true", slide.skipped);
        assert_eq!(unescape(&record[8]), slide.notes);
    }
    assert!(
        slides
            .iter()
            .any(|r| r[6] == "Kopie mit eigenem Titel" && r[8] == "Notizen nur auf der Kopie."),
        "the copy did not come back with its own title and notes"
    );
    assert!(
        slides
            .iter()
            .any(|r| r[6] == "Zahlen" && r[8] == "Hier langsam sprechen."),
        "and the original did not keep its own"
    );
    let _ = std::fs::remove_file(&out);
}

/// The deck with the copied slide, opened by Keynote and **saved** by it. What
/// is on disk afterwards was written by the app from its own model, so a
/// component it could not load would not come back. Off unless
/// `IWORK_APP_CHECK=1`.
#[test]
fn keynote_saves_the_deck_with_the_slide_this_crate_copied() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("keynote-slides.key");
    let out = temp("iwork-keynote-resave.key");

    let mut doc = Document::open(&path).unwrap();
    let source = doc.slides()[1].clone();
    let copy = doc.duplicate_slide(source.identifier).unwrap();
    doc.save(&out).unwrap();

    let status = std::process::Command::new(script("resave.sh"))
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("resave.sh: {e}"));
    assert!(status.success(), "Keynote would not open and save {out:?}");

    let after = Document::open(&out).unwrap();
    let show = after.show().unwrap();
    assert_eq!(show.slides.len(), 9, "Keynote dropped the copied slide");
    let made = show
        .slide(copy.identifier)
        .expect("and it kept the copy's identifier");
    assert_eq!(made.title_text(), source.title_text());
    assert_eq!(made.notes, source.notes);
    assert_eq!(made.layout_name, source.layout_name);
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_file(&out);
}

// -- writing a transition ----------------------------------------------------

/// The effect, its duration and its trigger, written and read back.
///
/// **A transition belongs to the slide it leaves**, so this is the slide the
/// audience is looking at when the effect runs. Two objects carry it: the
/// slide's `transition` (4) and the node's `has_transition` (7), and a deck
/// with one but not the other plays what its outline does not show.
#[test]
fn a_slide_can_be_given_a_transition() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    assert!(
        doc.slides()[0].transition.is_none(),
        "the deck starts plain"
    );

    let now = doc
        .set_transition(
            slide,
            &iwork::keynote::TransitionEdit {
                effect: "object cube".into(),
                duration: Some(1.5),
                automatic: Some(true),
                delay: Some(0.25),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(now.effect, "apple:ca-cube");
    assert_eq!(now.duration, 1.5);
    assert_eq!(now.delay, 0.25);
    assert!(now.automatic);

    let read = doc.slides()[0].transition.clone();
    assert_eq!(read, now, "what was written is what is read back");
    assert!(read.seed != 0, "a transition carries a seed");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());

    // The node's flag, which is the app's own navigator's answer.
    let node = doc.slides()[0].node;
    let archive = doc.archive(node).unwrap();
    assert_eq!(archive.varint(7), Some(1), "has_transition");
}

/// The identifier and the printed name are the same thing to the writer.
#[test]
fn an_effect_can_be_named_either_way() {
    let path = fixture!("keynote-deck.key");
    for spelling in ["dissolve", "apple:dissolve", "Dissolve"] {
        let mut doc = Document::open(&path).unwrap();
        let slide = doc.slides()[0].identifier;
        let now = doc
            .set_transition(
                slide,
                &iwork::keynote::TransitionEdit {
                    effect: spelling.into(),
                    ..Default::default()
                },
            )
            .unwrap();
        assert_eq!(now.effect, "apple:dissolve", "{spelling}");
        assert_eq!(now.duration, 1.0, "the app's own default for every effect");
    }
}

/// An effect Keynote does not have is refused by name rather than written as a
/// string the app will silently ignore.
#[test]
fn an_effect_that_does_not_exist_is_refused() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    let refusal = doc
        .set_transition(
            slide,
            &iwork::keynote::TransitionEdit {
                effect: "teleport".into(),
                ..Default::default()
            },
        )
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("44 effects"), "{refusal}");
    assert!(doc.changed_streams().is_empty(), "a refusal wrote nothing");
}

/// **`transition` is a required field of `KN.SlideArchive`.** "No transition"
/// is not the field's absence — Keynote refuses to parse a slide without it,
/// and the whole component then fails to load. What the app writes is the same
/// message with the effect `"none"`, and so does this.
#[test]
fn no_transition_is_written_as_none_rather_than_removed() {
    let path = fixture!("keynote-transitions.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[1].identifier;
    assert!(!doc.slides()[1].transition.is_none(), "it starts with one");

    let now = doc
        .set_transition(
            slide,
            &iwork::keynote::TransitionEdit {
                effect: "none".into(),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(now.is_none());

    let archive = doc.archive(slide).unwrap();
    let holder = archive
        .bytes(iwork::keynote::slide_field::TRANSITION)
        .and_then(pb::decode_nested)
        .expect("the field is still there, because the app requires it");
    let attributes = holder
        .bytes(iwork::keynote::transition_field::ATTRIBUTES)
        .and_then(pb::decode_nested)
        .expect("and it still holds attributes");
    let animation = attributes
        .bytes(iwork::keynote::transition_field::ANIMATION_ATTRIBUTES)
        .and_then(pb::decode_nested)
        .expect("and animation attributes");
    assert_eq!(
        animation
            .bytes(iwork::keynote::transition_field::EFFECT)
            .map(|raw| String::from_utf8_lossy(raw).into_owned()),
        Some("none".to_string())
    );
    let node = doc.slides()[1].node;
    assert_eq!(doc.archive(node).unwrap().varint(7), Some(0));
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
}

/// The `custom_*` parameters belong to the effect that wrote them.
///
/// Magic Move's "fade unmatched objects" means nothing to a dissolve, so
/// changing the effect drops the block; keeping the effect keeps it.
#[test]
fn the_parameters_belong_to_the_effect() {
    let path = fixture!("keynote-transitions.key");
    let with_parameters = Document::open(&path)
        .unwrap()
        .slides()
        .into_iter()
        .find(|s| !s.transition.parameters.is_empty())
        .expect("the fixture has one");

    // Same effect: the parameters stay.
    let mut doc = Document::open(&path).unwrap();
    let now = doc
        .set_transition(
            with_parameters.identifier,
            &iwork::keynote::TransitionEdit {
                effect: with_parameters.transition.effect.clone(),
                duration: Some(3.0),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(now.duration, 3.0);
    assert_eq!(now.parameters, with_parameters.transition.parameters);

    // A different effect: they go.
    let mut doc = Document::open(&path).unwrap();
    let now = doc
        .set_transition(
            with_parameters.identifier,
            &iwork::keynote::TransitionEdit {
                effect: "dissolve".into(),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        now.parameters.is_empty(),
        "a dissolve kept another effect's parameters: {:?}",
        now.parameters
    );
}

/// Keynote opens a deck this crate gave a transition, and writes it back with
/// the transition still on it. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn keynote_resaves_a_transition_this_crate_wrote() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("keynote-transitions.key");
    let mut doc = Document::open(&path).unwrap();
    let given = doc.slides()[0].identifier;
    let taken = doc.slides()[1].identifier;
    doc.set_transition(
        given,
        &iwork::keynote::TransitionEdit {
            effect: "dissolve".into(),
            duration: Some(2.5),
            automatic: Some(true),
            ..Default::default()
        },
    )
    .unwrap();
    doc.set_transition(
        taken,
        &iwork::keynote::TransitionEdit {
            effect: "none".into(),
            ..Default::default()
        },
    )
    .unwrap();

    let out = temp("iwork-transition.key");
    let _ = std::fs::remove_dir_all(&out);
    doc.save(&out).unwrap();
    let status = std::process::Command::new(script("resave.sh"))
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("resave.sh: {e}"));
    assert!(status.success(), "Keynote would not resave the deck");

    let after = Document::open(&out).unwrap();
    let slides = after.slides();
    let given = slides.iter().find(|s| s.identifier == given).unwrap();
    assert_eq!(given.transition.effect, "apple:dissolve");
    assert_eq!(given.transition.duration, 2.5);
    assert!(given.transition.automatic);
    let taken = slides.iter().find(|s| s.identifier == taken).unwrap();
    assert!(
        taken.transition.is_none(),
        "Keynote put a transition back: {:?}",
        taken.transition.effect
    );
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_dir_all(&out);
}

// -- writing a build ----------------------------------------------------------

/// A build is two objects and two lists: the `KN.BuildArchive` in the slide's
/// `builds`, and the `KN.BuildChunkArchive` in its `buildChunks`, which is the
/// order the app plays them in. The node's three counters are what the
/// navigator shows.
#[test]
fn a_drawable_can_be_animated_onto_a_slide() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    let shape = doc
        .add_text_box(
            &slide.to_string(),
            "Baut sich auf",
            (100.0, 100.0),
            (600.0, 120.0),
        )
        .unwrap();

    let build = doc
        .add_build(slide, shape, &iwork::keynote::BuildEdit::default())
        .unwrap();

    let read = doc.slides()[0].clone();
    assert_eq!(read.builds.len(), 1);
    assert_eq!(read.build_chunks.len(), 1, "one chunk per build event");
    let it = read
        .builds
        .iter()
        .find(|b| b.identifier == build)
        .expect("the build reads back");
    assert_eq!(it.drawable, Some(shape));
    assert_eq!(it.animation.animation_type, "In");
    assert_eq!(it.animation.effect, "apple:dissolve character");
    assert_eq!(it.animation.duration, 1.0);
    assert_eq!(it.delivery, "All at Once");
    assert!(
        it.unknown_attributes.is_empty(),
        "the writer wrote a field the reader cannot name: {:?}",
        it.unknown_attributes
    );

    let node = doc.archive(read.node).unwrap();
    assert_eq!(node.varint(6), Some(1), "has_builds");
    assert_eq!(node.varint(20), Some(1), "has_explicit_builds");
    assert_eq!(node.varint(15), Some(1), "build_event_count");
    assert!(doc.problems().is_empty(), "{:?}", doc.problems());
    assert!(doc.undeclared_references().is_empty());
}

/// A build animates something the slide owns. One naming a drawable from
/// somewhere else animates nothing and says nothing about it either.
#[test]
fn a_build_on_a_drawable_the_slide_does_not_own_is_refused() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    let elsewhere = doc.slides().iter().skip(1).find_map(|s| {
        doc.archive(s.identifier).ok().and_then(|a| {
            a.fields
                .iter()
                .find(|f| f.number == 7)
                .and_then(|f| match &f.value {
                    iwork::pb::Value::Bytes(raw) => {
                        iwork::pb::decode_nested(raw).and_then(|m| m.varint(1))
                    }
                    _ => None,
                })
        })
    });
    let Some(elsewhere) = elsewhere else {
        eprintln!("no drawable on another slide — skipping");
        return;
    };
    let refusal = doc
        .add_build(slide, elsewhere, &iwork::keynote::BuildEdit::default())
        .unwrap_err()
        .to_string();
    assert!(refusal.contains("is not on slide"), "{refusal}");
    assert!(doc.changed_streams().is_empty(), "a refusal wrote nothing");
}

/// A build-out says so, and takes the other default effect.
#[test]
fn a_build_out_says_out() {
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    let shape = doc
        .add_text_box(&slide.to_string(), "Geht ab", (0.0, 0.0), (100.0, 40.0))
        .unwrap();
    let build = doc
        .add_build(
            slide,
            shape,
            &iwork::keynote::BuildEdit {
                kind: iwork::keynote::BuildKind::Out,
                duration: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
    let slide = doc.slides().into_iter().next().unwrap();
    let it = slide.builds.iter().find(|b| b.identifier == build).unwrap();
    assert_eq!(it.animation.animation_type, "Out");
    assert_eq!(it.animation.effect, "apple:bc-appear");
    assert_eq!(it.animation.duration, 2.0);
}

/// Keynote opens a deck with a build this crate authored and writes it back
/// with the build on it. Off unless `IWORK_APP_CHECK=1`.
#[test]
fn keynote_resaves_a_build_this_crate_wrote() {
    if std::env::var("IWORK_APP_CHECK").as_deref() != Ok("1") {
        eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round trip");
        return;
    }
    let path = fixture!("keynote-deck.key");
    let mut doc = Document::open(&path).unwrap();
    let slide = doc.slides()[0].identifier;
    let shape = doc
        .add_text_box(
            &slide.to_string(),
            "Baut sich auf",
            (100.0, 100.0),
            (600.0, 120.0),
        )
        .unwrap();
    doc.add_build(slide, shape, &iwork::keynote::BuildEdit::default())
        .unwrap();

    let out = temp("iwork-build.key");
    let _ = std::fs::remove_dir_all(&out);
    doc.save(&out).unwrap();
    let status = std::process::Command::new(script("resave.sh"))
        .arg(&out)
        .status()
        .unwrap_or_else(|e| panic!("resave.sh: {e}"));
    assert!(status.success(), "Keynote would not resave the deck");

    let after = Document::open(&out).unwrap();
    let build = after
        .slides()
        .into_iter()
        .flat_map(|s| s.builds)
        .find(|b| b.drawable == Some(shape))
        .expect("Keynote dropped the build");
    assert_eq!(build.animation.effect, "apple:dissolve character");
    assert_eq!(build.animation.animation_type, "In");
    assert!(after.problems().is_empty(), "{:?}", after.problems());
    let _ = std::fs::remove_dir_all(&out);
}
