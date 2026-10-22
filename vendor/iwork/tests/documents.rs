//! How a document comes to be, and what shape it is on disk.
//!
//! Two claims, both of them about the container rather than the contents.
//!
//! **A package is a directory as readily as it is a file.** `File > Advanced >
//! Change File Type` writes the same entries into a folder with the document's
//! name, and macOS types the two differently — `mdls` says
//! `com.apple.iwork.pages.sffpages` for the file and
//! `com.apple.iwork.pages.pages`, conforming to `com.apple.package`, for the
//! folder. This crate reads both, keeps the shape it found on save, and puts
//! the same bytes in each entry either way. Pages was watched opening a package
//! this test's own code wrote.
//!
//! **A document is made by copying one that works.** `Document::from_template`
//! opens one of the 901 template bundles the three apps ship and gives it a new
//! identity; the app opens the result, and saves it, and leaves the identity
//! this crate wrote alone — which is the same signal `save_as_new` was accepted
//! on in phase 7.

use std::path::{Path, PathBuf};

use iwork::package::Form;
use iwork::{Document, Kind, Package};

fn corpus() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut found = Vec::new();
    collect(&dir, &mut found);
    found.retain(|path| !Package::read(path).is_ok_and(|p| p.contains(".iwpv2")));
    found.sort();
    if found.is_empty() {
        eprintln!("no fixtures in tests/fixtures — skipping (see tests/fixtures/README.md)");
    }
    found
}

fn collect(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(Result::ok) {
        let path = entry.path();
        let named = matches!(
            path.extension().and_then(|e| e.to_str()),
            Some("pages") | Some("numbers") | Some("key")
        );
        if named {
            if path.is_file() {
                found.push(path);
            }
        } else if path.is_dir() {
            collect(&path, found);
        }
    }
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("iwork-documents-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Off unless `IWORK_APP_CHECK=1`: it drives the app through AppleScript.
fn app_check(path: &Path) -> bool {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/app-check.sh");
    std::process::Command::new(&script)
        .arg(path)
        .status()
        .is_ok_and(|status| status.success())
}

fn app_checking() -> bool {
    if std::env::var("IWORK_APP_CHECK").as_deref() == Ok("1") {
        return true;
    }
    eprintln!("IWORK_APP_CHECK is not 1 — skipping the app round-trip");
    false
}

// -- the package form --------------------------------------------------------

/// The two shapes hold the same document, entry for entry and byte for byte,
/// and a document read as one shape is saved back as that shape.
#[test]
fn every_fixture_survives_the_trip_through_the_package_form() {
    let dir = scratch("form");
    for path in corpus() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let original = Package::read(&path).unwrap();
        assert_eq!(original.form, Form::SingleFile, "{name}");

        let as_package = dir.join(&name);
        original.write_as(&as_package, Form::Directory).unwrap();
        assert!(
            as_package.is_dir(),
            "{name}: the package is not a directory"
        );

        let reread = Package::read(&as_package).unwrap();
        assert_eq!(reread.form, Form::Directory, "{name}");
        let mut expected: Vec<_> = original.entries.clone();
        expected.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(
            reread.entries, expected,
            "{name}: the package form is not the same entries"
        );

        // And it is the same *document*: the same kind, the same objects, the
        // same identity, with nothing in the reader aware of the difference.
        let zipped = Document::open(&path).unwrap();
        let packaged = Document::open(&as_package).unwrap();
        assert_eq!(packaged.kind(), zipped.kind(), "{name}");
        assert_eq!(
            packaged.objects().count(),
            zipped.objects().count(),
            "{name}"
        );
        assert_eq!(
            packaged.metadata().unwrap().document_identifier,
            zipped.metadata().unwrap().document_identifier,
            "{name}"
        );
        assert!(
            packaged.changed_streams().is_empty(),
            "{name}: reading the package form changed a stream"
        );

        // A save keeps the shape, and every entry keeps its bytes: the same
        // byte-identity guarantee the ZIP form has had since phase 0.
        let saved = dir.join(format!("saved-{name}"));
        packaged.save(&saved).unwrap();
        assert!(saved.is_dir(), "{name}: a package was saved as one file");
        assert_eq!(
            Package::read(&saved).unwrap().entries,
            reread.entries,
            "{name}: a no-op save of the package form changed an entry"
        );
        let _ = std::fs::remove_dir_all(&saved);
        let _ = std::fs::remove_dir_all(&as_package);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The app opens a package this crate wrote — one per app, because the three
/// read their own container and nothing here would notice if one of them did
/// not.
#[test]
fn the_app_opens_a_package_this_crate_wrote() {
    if !app_checking() {
        return;
    }
    let dir = scratch("app-form");
    let mut seen = Vec::new();
    for path in corpus() {
        let kind = Kind::from_extension(&path);
        if seen.contains(&kind) {
            continue;
        }
        seen.push(kind);
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let as_package = dir.join(&name);
        Package::read(&path)
            .unwrap()
            .write_as(&as_package, Form::Directory)
            .unwrap();
        assert!(
            app_check(&as_package),
            "{name}: the app would not open the package form of it"
        );
        let _ = std::fs::remove_dir_all(&as_package);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// **What the app does with the shape, and it is not what a reader expects.**
/// Asked from a script to save a document it opened as a *package*, Pages
/// writes a single file back over it. `Change File Type` is a menu item and a
/// menu item needs a window, so nothing here can say what the app does when a
/// user has actually chosen the package form — only what it does with a
/// package it was handed and told to save. This crate does the opposite and
/// keeps the shape, because a save that silently changed the user's file type
/// would be a save that lost something.
#[test]
fn the_app_writes_one_file_back_over_a_package() {
    if !app_checking() {
        return;
    }
    let Some(path) = corpus()
        .into_iter()
        .find(|p| Kind::from_extension(p) == Kind::Pages)
    else {
        return;
    };
    let dir = scratch("app-resave");
    let as_package = dir.join(path.file_name().unwrap());
    Package::read(&path)
        .unwrap()
        .write_as(&as_package, Form::Directory)
        .unwrap();

    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/resave.sh");
    let status = std::process::Command::new(&script)
        .arg(&as_package)
        .status();
    assert!(
        status.is_ok_and(|s| s.success()),
        "Pages would not resave the package form"
    );
    assert!(
        as_package.is_file(),
        "Pages was measured converting a package to one file on save; it did not this time"
    );
    // Whatever shape it chose, it is still the document.
    assert!(Document::open(&as_package).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A locked document is locked in either shape. The detection is the presence
/// of `.iwpv2`, and in the package form that is a dot-file at the root — which
/// a directory walk that skipped dot-files would miss, and would then report a
/// password-protected document as a corrupt one.
#[test]
fn an_encrypted_package_is_refused_in_the_package_form_too() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures");
    let mut all = Vec::new();
    collect(&dir, &mut all);
    let Some(locked) = all
        .into_iter()
        .find(|p| Package::read(p).is_ok_and(|package| package.contains(".iwpv2")))
    else {
        eprintln!("no password-protected fixture — skipping");
        return;
    };

    let scratch = scratch("locked");
    let as_package = scratch.join(locked.file_name().unwrap());
    Package::read(&locked)
        .unwrap()
        .write_as(&as_package, Form::Directory)
        .unwrap();

    let package = Package::read(&as_package).unwrap();
    assert!(
        package.contains(".iwpv2"),
        "the key material was not an entry"
    );
    assert!(
        matches!(
            Document::open(&as_package),
            Err(iwork::Error::Encrypted { .. })
        ),
        "a locked package should be refused by name in either shape"
    );
    let _ = std::fs::remove_dir_all(&scratch);
}

// -- previews ----------------------------------------------------------------

/// The rule, stated as a test: an edit leaves the previews exactly where they
/// were, stale and all, and a document that had none still has none.
///
/// The argument is in `FORMAT.md` §1. The short of it: nothing in the document
/// refers to them, the app redraws them the next time *it* saves, and removing
/// them would cost the byte-identity a no-op save promises.
#[test]
fn an_edit_leaves_the_previews_alone() {
    let dir = scratch("previews");
    for path in corpus() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let mut doc = Document::open(&path).unwrap();
        let before: Vec<String> = doc.previews().iter().map(|n| n.to_string()).collect();

        // Any edit at all: the first text storage, rewritten with its own text
        // plus a word, which is enough to rewrite a stream.
        if let Some(storage) = doc.text_storages().into_iter().find(|s| !s.text.is_empty()) {
            let edited = format!("{} and more", storage.text);
            if doc.set_text(storage.identifier, &edited).is_err() {
                continue;
            }
        }
        let out = dir.join(&name);
        doc.save(&out).unwrap();

        let after = Document::open(&out).unwrap();
        let names: Vec<String> = after.previews().iter().map(|n| n.to_string()).collect();
        assert_eq!(names, before, "{name}: an edit changed the preview entries");
        for preview in &before {
            assert_eq!(
                after.package().get(preview),
                Package::read(&path).unwrap().get(preview),
                "{name}: {preview} was rewritten"
            );
        }
        let _ = std::fs::remove_file(&out);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// And the other half of the choice: a document with the previews taken out is
/// still a document. Every one of Apple's 901 template bundles is one.
#[test]
fn a_document_without_previews_is_still_a_document() {
    let dir = scratch("stripped");
    let checking = app_checking();
    let mut checked = Vec::new();
    for path in corpus() {
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let mut doc = Document::open(&path).unwrap();
        let had = doc.previews().len();
        assert_eq!(doc.strip_previews(), had, "{name}");
        assert!(doc.previews().is_empty(), "{name}");
        // Nothing refers to a preview, so nothing can be left dangling by
        // taking one away.
        assert!(doc.problems().is_empty(), "{name}: {:?}", doc.problems());

        let out = dir.join(&name);
        doc.save(&out).unwrap();
        let reread = Document::open(&out).unwrap();
        assert!(reread.previews().is_empty(), "{name}");
        assert_eq!(reread.objects().count(), doc.objects().count(), "{name}");

        let kind = Kind::from_extension(&path);
        if checking && !checked.contains(&kind) {
            checked.push(kind);
            assert!(
                app_check(&out),
                "{name}: the app would not open it without its previews"
            );
        }
        let _ = std::fs::remove_file(&out);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// -- documents from templates ------------------------------------------------

/// The template bundles the installed apps ship, newest-looking first. All 901
/// of them are ZIPs on this Mac — not one is in the package form, which is
/// worth knowing because a template is the most likely place to meet one.
fn templates() -> Vec<PathBuf> {
    let mut found = Vec::new();
    let Ok(apps) = std::fs::read_dir("/Applications") else {
        return found;
    };
    for app in apps.filter_map(Result::ok) {
        let dir = app.path().join("Contents/SharedSupport/Templates");
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        let mut per_app = Vec::new();
        for entry in entries.filter_map(Result::ok) {
            let Ok(variants) = std::fs::read_dir(entry.path()) else {
                continue;
            };
            for variant in variants.filter_map(Result::ok) {
                let path = variant.path();
                if matches!(
                    path.extension().and_then(|e| e.to_str()),
                    Some("template") | Some("nmbtemplate") | Some("kth")
                ) {
                    per_app.push(path);
                }
            }
        }
        per_app.sort();
        found.extend(per_app.into_iter().take(1));
    }
    found.sort();
    if found.is_empty() {
        eprintln!("no template bundles in /Applications — skipping");
    }
    found
}

/// A document made from a template is a document: it decodes, it checks out,
/// and it has an identity of its own — a *new* lineage, which is where this
/// differs from a copy.
#[test]
fn a_document_from_a_template_has_an_identity_of_its_own() {
    for template in templates() {
        let name = template.display().to_string();
        let first = Document::from_template(&template).unwrap();
        let second = Document::from_template(&template).unwrap();
        assert_ne!(first.kind(), Kind::Unknown, "{name}");
        assert!(first.objects().count() > 0, "{name}");
        assert!(
            first.problems().is_empty(),
            "{name}: {:?}",
            first.problems()
        );

        let one = first.metadata().unwrap().properties.unwrap();
        let two = second.metadata().unwrap().properties.unwrap();
        let template_properties =
            iwork::metadata::Properties::read(&Package::read(&template).unwrap())
                .unwrap()
                .unwrap();

        assert_ne!(one.document_uuid, two.document_uuid, "{name}");
        assert_ne!(one.private_uuid, two.private_uuid, "{name}");
        assert_ne!(one.version_uuid, two.version_uuid, "{name}");
        assert_ne!(
            one.document_uuid, template_properties.document_uuid,
            "{name}: the document kept the template's identity"
        );
        // The measured rule: a document made from a template begins its own
        // lineage rather than inheriting the template's.
        assert_eq!(
            one.stable_document_uuid, one.document_uuid,
            "{name}: stableDocumentUUID should be the document's own"
        );
        assert_ne!(
            one.stable_document_uuid, template_properties.stable_document_uuid,
            "{name}: the template's lineage was carried into the document"
        );
        assert_eq!(
            first.metadata().unwrap().document_identifier,
            one.document_uuid,
            "{name}: DocumentIdentifier does not match documentUUID"
        );
    }
}

/// And it says which template, the way the apps' own documents do.
#[test]
fn a_document_from_a_template_records_which_one() {
    for template in templates() {
        let doc = Document::from_template(&template).unwrap();
        let identifier = doc.metadata().unwrap().template_identifier.unwrap();
        assert!(
            identifier.starts_with("Application/"),
            "{}: {identifier}",
            template.display()
        );
        let stem = template.file_stem().unwrap().to_string_lossy();
        let folder = template
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy();
        assert_eq!(
            identifier,
            format!("Application/{folder}/{stem}"),
            "{}",
            template.display()
        );
    }
}

/// A template that is not one of the app's own is opened all the same, and
/// nothing is claimed about where it came from.
#[test]
fn a_template_from_anywhere_else_claims_no_identifier() {
    let Some(template) = templates().into_iter().next() else {
        return;
    };
    let dir = scratch("elsewhere");
    let copy = dir.join(template.file_name().unwrap());
    std::fs::copy(&template, &copy).unwrap();

    let doc = Document::from_template(&copy).unwrap();
    assert_ne!(doc.kind(), Kind::Unknown);
    assert_eq!(
        doc.metadata().unwrap().template_identifier,
        None,
        "a template outside an app bundle should claim no identifier"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A new document's on-disk shape is not the template's. A directory-form
/// template must still yield a single-file document — the directory form is a
/// choice a user makes for a document grown large (`Change File Type`), never
/// something inherited from whatever shape a template bundle sat in on disk.
#[test]
fn a_document_from_a_template_defaults_to_the_single_file_form() {
    // Any package will do; a template bundle is the natural source, a corpus
    // fixture the fallback so this runs on a machine without the apps.
    let Some(source) = templates()
        .into_iter()
        .next()
        .or_else(|| corpus().into_iter().next())
    else {
        eprintln!("no template or fixture — skipping");
        return;
    };

    let dir = scratch("template-form");
    let as_directory = dir.join(source.file_name().unwrap());
    Package::read(&source)
        .unwrap()
        .write_as(&as_directory, Form::Directory)
        .unwrap();
    assert_eq!(
        Package::read(&as_directory).unwrap().form,
        Form::Directory,
        "the template was not written in the directory form to begin with"
    );

    let document = Document::from_template(&as_directory).unwrap();
    assert_eq!(
        document.package().form,
        Form::SingleFile,
        "a directory-form template yielded a directory-form document"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The acceptance test: the app opens what `from_template` wrote, reads it,
/// and saves it back from its own model — and leaves the identity alone,
/// which is what says the document is properly a new one rather than a copy
/// the app feels it has to re-identify.
#[test]
fn the_app_opens_and_resaves_a_document_made_from_a_template() {
    if !app_checking() {
        return;
    }
    let dir = scratch("from-template");
    for template in templates() {
        let doc = Document::from_template(&template).unwrap();
        let Some(extension) = doc.kind().extension() else {
            continue;
        };
        let out = dir.join(format!(
            "{}.{extension}",
            template.file_stem().unwrap().to_string_lossy()
        ));
        doc.save(&out).unwrap();

        assert!(
            app_check(&out),
            "{}: the app would not open the document made from it",
            template.display()
        );

        let before = Document::open(&out).unwrap().metadata().unwrap();
        let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/resave.sh");
        let status = std::process::Command::new(&script).arg(&out).status();
        assert!(
            status.is_ok_and(|s| s.success()),
            "{}: the app would not save it back",
            template.display()
        );
        let after = Document::open(&out).unwrap().metadata().unwrap();
        assert_eq!(
            after.properties.as_ref().unwrap().document_uuid,
            before.properties.as_ref().unwrap().document_uuid,
            "{}: the app re-identified the document, which it does to a copy \
             it does not consider properly new",
            template.display()
        );
        let _ = std::fs::remove_file(&out);
    }
    let _ = std::fs::remove_dir_all(&dir);
}
