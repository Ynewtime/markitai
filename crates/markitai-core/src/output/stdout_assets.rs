//! Images of a document converted without an output directory.
//!
//! The CLI prints such a document to stdout. The images and page captures its
//! Markdown refers to are saved in one content-addressed store (by default
//! `MARKITAI_HOME/assets`) and the references become `file://` URIs, so the
//! printed links still open after the process exits. A file is named by the
//! SHA-256 of its bytes: repeated conversions reuse it, and an existing name
//! is only verified, never rewritten, so earlier output keeps its images.
//! Nothing is written for a document without such references. An image that
//! cannot be saved keeps its relative reference and the result says so.

use super::{check_path, rewrite_asset_references};
use crate::{Asset, ConversionOutput, Error, Result, config};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt::Write;
use std::path::{Path, PathBuf};

/// Files live one level below the store, as in the reference layout.
const BLOBS: &str = "blobs";

struct Entry<'a> {
    /// Destination as the converters write it, e.g. `.markitai/assets/x.jpg`.
    reference: String,
    /// Encoded destination inside a generated page-image comment
    /// (`<!-- ![Page 1](…) -->`), which ordinary reference rewriting leaves
    /// literal like any other comment.
    comment: Option<String>,
    name: &'a str,
    bytes: &'a [u8],
    screenshot: bool,
}

pub(crate) fn persist(
    store: &Path,
    result: &mut ConversionOutput,
    assets: &[Asset],
    screenshots: &[Asset],
    cfg: &Value,
) {
    let entries: Vec<Entry<'_>> = assets
        .iter()
        .map(|asset| Entry {
            reference: format!(".markitai/assets/{}", asset.name),
            comment: None,
            name: &asset.name,
            bytes: &asset.bytes,
            screenshot: false,
        })
        .chain(screenshots.iter().map(|shot| Entry {
            reference: format!(".markitai/screenshots/{}", shot.name),
            comment: comment_destination(&shot.name),
            name: &shot.name,
            bytes: &shot.bytes,
            screenshot: true,
        }))
        .collect();
    let wanted = referenced(result, &entries);
    if wanted.is_empty() {
        return;
    }
    let allow_symlinks = config::enabled(cfg, "/output/allow_symlinks");
    let blobs = match open(store, allow_symlinks) {
        Ok(blobs) => blobs,
        Err(error) => {
            result
                .warnings
                .push(unsaved(wanted.len(), store, &error.to_string()));
            return;
        }
    };
    let mut uris = HashMap::new();
    let mut comments = HashMap::new();
    let mut stored = HashMap::new();
    let mut failure = None;
    let mut failed = 0;
    for entry in wanted {
        let path = match save(&blobs, entry, allow_symlinks) {
            Ok(path) => path,
            Err(error) => {
                failed += 1;
                failure.get_or_insert(error.to_string());
                continue;
            }
        };
        // The store path was checked to be UTF-8 and the file name is ASCII.
        let uri = file_uri(&path).expect("stored paths are UTF-8");
        if let Some(destination) = &entry.comment {
            comments
                .entry(destination.as_str())
                .or_insert_with(|| encoded(&uri));
        }
        uris.insert(entry.reference.clone(), uri);
        let listed = if entry.screenshot {
            &mut result.screenshots
        } else {
            &mut result.assets
        };
        if !listed.contains(&path) {
            listed.push(path.clone());
        }
        stored.insert(entry.reference.clone(), path);
    }
    let link = |text: &str| relink_comments(&rewrite_asset_references(text, &uris), &comments);
    result.markdown = link(&result.markdown);
    if let Some(markdown) = &mut result.llm_markdown {
        *markdown = link(markdown);
    }
    for image in &mut result.images {
        let path = image
            .get("asset")
            .and_then(Value::as_str)
            .and_then(|asset| stored.get(asset));
        if let Some(path) = path {
            image["asset"] = path.to_string_lossy().as_ref().into();
        }
    }
    if let Some(error) = failure {
        result.warnings.push(unsaved(failed, store, &error));
    }
}

/// Entries the base or enhanced Markdown actually refers to, decided by a
/// trial rewrite: a reference in code or an unrelated comment stays literal
/// and stores nothing. When two entries share a destination, the first one
/// owns it, as in output publication.
fn referenced<'e, 'a>(result: &ConversionOutput, entries: &'e [Entry<'a>]) -> Vec<&'e Entry<'a>> {
    let marker = |index: usize| format!("markitai-stdout-asset-{index}-marker");
    let mut markers = HashMap::new();
    for (index, entry) in entries.iter().enumerate() {
        markers
            .entry(entry.reference.clone())
            .or_insert_with(|| marker(index));
    }
    let texts: Vec<&str> = std::iter::once(result.markdown.as_str())
        .chain(result.llm_markdown.as_deref())
        .collect();
    let marked: Vec<String> = texts
        .iter()
        .map(|text| rewrite_asset_references(text, &markers))
        .collect();
    let commented: std::collections::HashSet<&str> = texts
        .iter()
        .flat_map(|text| page_comments(text).map(|range| &text[range]))
        .collect();
    entries
        .iter()
        .enumerate()
        .filter(|(index, entry)| {
            let marker = marker(*index);
            marked.iter().any(|text| text.contains(&marker))
                || entry
                    .comment
                    .as_deref()
                    .is_some_and(|destination| commented.contains(destination))
        })
        .map(|(_, entry)| entry)
        .collect()
}

/// Destination ranges of generated page-image comments, `](…) -->` whose
/// destination is below `.markitai/screenshots/`. The converters encode the
/// capture name, so the first `)` ends the destination.
fn page_comments(text: &str) -> impl Iterator<Item = std::ops::Range<usize>> + '_ {
    let mut from = 0;
    std::iter::from_fn(move || {
        loop {
            let start = from + text[from..].find("](.markitai/screenshots/")? + 2;
            let end = start + text[start..].find(')')?;
            from = end;
            if text[end..].starts_with(") -->") {
                return Some(start..end);
            }
        }
    })
}

/// Replace known page-image comment destinations in one pass.
fn relink_comments(text: &str, uris: &HashMap<&str, String>) -> String {
    if uris.is_empty() {
        return text.to_owned();
    }
    let mut output = String::with_capacity(text.len());
    let mut copied = 0;
    for range in page_comments(text) {
        if let Some(uri) = uris.get(&text[range.clone()]) {
            output.push_str(&text[copied..range.start]);
            output.push_str(uri);
            copied = range.end;
        }
    }
    output.push_str(&text[copied..]);
    output
}

/// The blob directory, created if needed, as the canonical path the links
/// will name (no `..` from a relative `MARKITAI_HOME`).
fn open(store: &Path, allow_symlinks: bool) -> Result<PathBuf> {
    let blobs = std::path::absolute(store)?.join(BLOBS);
    check_path(&blobs, allow_symlinks)?;
    std::fs::create_dir_all(&blobs)?;
    check_path(&blobs, allow_symlinks)?;
    let blobs = std::fs::canonicalize(&blobs)?;
    if blobs.to_str().is_none() {
        return Err(Error::InvalidInput(
            "the image store path is not valid UTF-8".into(),
        ));
    }
    Ok(blobs)
}

fn save(blobs: &Path, entry: &Entry<'_>, allow_symlinks: bool) -> Result<PathBuf> {
    let digest = crate::hex(Sha256::digest(entry.bytes));
    let extension: String = Path::new(entry.name)
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or_default()
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .take(12)
        .collect();
    let extension = if extension.is_empty() {
        "bin"
    } else {
        &extension
    };
    // The short name matches output-directory assets. A file already using
    // it with other bytes (damaged, or edited by hand) is never replaced;
    // the full digest names these bytes instead.
    let mut first = None;
    for length in [24, 64] {
        let path = blobs.join(format!("{}.{extension}", &digest[..length]));
        let saved = check_path(&path, allow_symlinks)
            .and_then(|()| crate::asset_store::insert_or_verify(&path, entry.bytes));
        match saved {
            Ok(()) => return Ok(path),
            Err(error) => {
                first.get_or_insert(error);
            }
        }
    }
    Err(first.expect("at least one name was tried"))
}

/// The page-image comment destination exactly as the PDF and Office
/// converters write it for this capture name.
fn comment_destination(name: &str) -> Option<String> {
    let generated = crate::formats::pdf_screenshot_reference(1, name);
    let start = generated.find("](")? + 2;
    let end = generated.rfind(") -->")?;
    Some(generated.get(start..end)?.to_owned())
}

/// A `file://` URI for an absolute path, not yet escaped: reference
/// rewriting escapes it for the syntax of each destination it replaces.
fn file_uri(path: &Path) -> Option<String> {
    let text = path.to_str()?;
    if cfg!(windows) {
        let text = text.replace('\\', "/");
        let text = match text.strip_prefix("//?/UNC/") {
            Some(share) => format!("//{share}"),
            None => text.strip_prefix("//?/").unwrap_or(&text).to_owned(),
        };
        Some(match text.strip_prefix("//") {
            Some(share) => format!("file://{share}"),
            None => format!("file:///{text}"),
        })
    } else {
        Some(format!("file://{text}"))
    }
}

/// Percent-encode everything after the scheme except unreserved characters,
/// separators and a drive colon, so the URI is safe inside an HTML comment.
fn encoded(uri: &str) -> String {
    let (scheme, rest) = uri.split_at(uri.find("//").map_or(0, |index| index + 2));
    let mut output = String::with_capacity(uri.len());
    output.push_str(scheme);
    for byte in rest.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'/' | b':') {
            output.push(byte as char);
        } else {
            write!(output, "%{byte:02X}").expect("writing to a String cannot fail");
        }
    }
    output
}

fn unsaved(count: usize, store: &Path, error: &str) -> String {
    let (noun, references) = if count == 1 {
        ("image", "its reference still points")
    } else {
        ("images", "their references still point")
    };
    format!(
        "{count} {noun} could not be saved to {} and {references} into .markitai/, which stdout mode does not write: {error}. Use -o DIR to keep images.",
        store.display()
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn asset(name: &str, bytes: &[u8]) -> Asset {
        Asset {
            name: name.into(),
            bytes: bytes.to_vec(),
        }
    }

    fn stored(store: &Path) -> Vec<String> {
        let mut names: Vec<_> = match std::fs::read_dir(store.join(BLOBS)) {
            Ok(entries) => entries
                .map(|entry| entry.unwrap().file_name().into_string().unwrap())
                .collect(),
            Err(_) => Vec::new(),
        };
        names.sort();
        names
    }

    fn digest(bytes: &[u8]) -> String {
        crate::hex(Sha256::digest(bytes))
    }

    fn document(markdown: &str) -> ConversionOutput {
        ConversionOutput {
            markdown: markdown.into(),
            ..Default::default()
        }
    }

    #[test]
    fn uris_are_absolute_and_comment_safe() {
        if !cfg!(windows) {
            assert_eq!(
                file_uri(Path::new("/home/a b/x.jpg")).unwrap(),
                "file:///home/a b/x.jpg"
            );
        }
        assert_eq!(
            encoded("file:///home/a b/é-->#?.jpg"),
            "file:///home/a%20b/%C3%A9--%3E%23%3F.jpg"
        );
        assert_eq!(encoded("file:///C:/x/y.png"), "file:///C:/x/y.png");
        assert_eq!(
            comment_destination("doc.pdf.page0001.jpg").unwrap(),
            ".markitai/screenshots/doc.pdf.page0001.jpg"
        );
        assert_eq!(
            comment_destination("a b.jpg").unwrap(),
            ".markitai/screenshots/a%20b.jpg"
        );
    }

    #[test]
    fn referenced_images_and_captures_are_stored_once_and_linked() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        let shared = b"shared image bytes".as_slice();
        let mut result = document(
            "![a](.markitai/assets/a.png) ![b](.markitai/assets/b%20c.png)\n\
             `![literal](.markitai/assets/a.png)`\n\
             <img src=\".markitai/assets/a.png\" alt=\"html\">\n\
             <!-- ![Page 1](.markitai/screenshots/doc%20x.page0001.jpg) -->\n",
        );
        result.llm_markdown = Some("![enhanced](.markitai/assets/a.png)\n".into());
        result.images = vec![json!({"asset": ".markitai/assets/a.png"})];
        let assets = [
            asset("a.png", shared),
            asset("b c.png", shared),
            asset("unused.png", b"never referenced"),
        ];
        let screenshots = [asset("doc x.page0001.jpg", b"page capture")];
        persist(&store, &mut result, &assets, &screenshots, &json!({}));
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
        let blobs = std::fs::canonicalize(store.join(BLOBS)).unwrap();
        let image_name = format!("{}.png", &digest(shared)[..24]);
        let page_name = format!("{}.jpg", &digest(b"page capture")[..24]);
        let image = blobs.join(&image_name);
        let page = blobs.join(&page_name);
        // One file for the two names with equal bytes; none for the unused one.
        let mut expected = vec![image_name, page_name];
        expected.sort();
        assert_eq!(stored(&store), expected);
        assert_eq!(std::fs::read(&image).unwrap(), shared);
        assert_eq!(std::fs::read(&page).unwrap(), b"page capture");
        let uri = file_uri(&image).unwrap();
        let page_uri = encoded(&file_uri(&page).unwrap());
        let markdown = &result.markdown;
        assert!(markdown.contains(&format!("![a]({uri})")), "{markdown}");
        assert!(markdown.contains(&format!("![b]({uri})")), "{markdown}");
        assert!(
            markdown.contains(&format!("<img src=\"{uri}\"")),
            "{markdown}"
        );
        assert!(markdown.contains("`![literal](.markitai/assets/a.png)`"));
        assert!(
            markdown.contains(&format!("<!-- ![Page 1]({page_uri}) -->")),
            "{markdown}"
        );
        assert_eq!(
            result.llm_markdown.as_deref(),
            Some(format!("![enhanced]({uri})\n").as_str())
        );
        assert_eq!(result.images[0]["asset"], image.to_str().unwrap());
        assert_eq!(result.assets, vec![image.clone()]);
        assert_eq!(result.screenshots, vec![page.clone()]);

        // Converting again reuses the file without writing it.
        let before = std::fs::metadata(&image).unwrap().modified().unwrap();
        let mut again = document("![a](.markitai/assets/a.png)\n");
        persist(&store, &mut again, &assets[..1], &[], &json!({}));
        assert_eq!(again.markdown, format!("![a]({uri})\n"));
        assert_eq!(stored(&store).len(), 2);
        assert_eq!(
            std::fs::metadata(&image).unwrap().modified().unwrap(),
            before
        );
    }

    #[test]
    fn only_generated_page_comments_are_relinked() {
        let uris = HashMap::from([(".markitai/screenshots/a.jpg", "file:///s/a.jpg".to_owned())]);
        let text = "<!-- ![Page 1](.markitai/screenshots/a.jpg) -->\n\
                    <!-- ![Page 2](.markitai/screenshots/b.jpg) -->\n\
                    `](.markitai/screenshots/a.jpg)` and <!-- ![Slide 3](.markitai/screenshots/a.jpg) -->";
        assert_eq!(
            relink_comments(text, &uris),
            "<!-- ![Page 1](file:///s/a.jpg) -->\n\
             <!-- ![Page 2](.markitai/screenshots/b.jpg) -->\n\
             `](.markitai/screenshots/a.jpg)` and <!-- ![Slide 3](file:///s/a.jpg) -->"
        );
        let found: Vec<_> = page_comments(text).map(|range| &text[range]).collect();
        assert_eq!(
            found,
            [
                ".markitai/screenshots/a.jpg",
                ".markitai/screenshots/b.jpg",
                ".markitai/screenshots/a.jpg"
            ]
        );
        assert_eq!(relink_comments(text, &HashMap::new()), text);
    }

    #[test]
    fn links_name_the_canonical_store_directory() {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("side")).unwrap();
        let store = root.path().join("side/../store");
        let mut result = document("![x](.markitai/assets/x.png)\n");
        persist(
            &store,
            &mut result,
            &[asset("x.png", b"x")],
            &[],
            &json!({}),
        );
        let blobs = std::fs::canonicalize(root.path().join("store").join(BLOBS)).unwrap();
        let path = blobs.join(format!("{}.png", &digest(b"x")[..24]));
        assert_eq!(
            result.markdown,
            format!("![x]({})\n", file_uri(&path).unwrap())
        );
        assert!(!result.markdown.contains("/../"));
    }

    #[test]
    fn documents_without_image_references_leave_the_store_untouched() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        let markdown = "Text with `.markitai/assets/a.png` only.\n";
        let mut result = document(markdown);
        persist(
            &store,
            &mut result,
            &[asset("a.png", b"bytes")],
            &[],
            &json!({}),
        );
        assert_eq!(result.markdown, markdown);
        assert!(!store.exists());
        assert!(result.warnings.is_empty());
    }

    #[test]
    fn stored_names_come_from_content_never_from_the_source_name() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        let name = "../../escape.p-n_g";
        let mut result = document(&format!("![x](<.markitai/assets/{name}>)\n"));
        persist(
            &store,
            &mut result,
            &[asset(name, b"payload")],
            &[],
            &json!({}),
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
        let file = format!("{}.png", &digest(b"payload")[..24]);
        assert_eq!(stored(&store), vec![file.clone()]);
        let created: Vec<_> = std::fs::read_dir(root.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        assert_eq!(created, vec![std::ffi::OsString::from("store")]);
        assert!(result.markdown.contains(&file), "{}", result.markdown);
    }

    #[test]
    fn a_damaged_file_under_the_short_name_is_kept_and_the_full_digest_used() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        let bytes = b"real image".as_slice();
        let hash = digest(bytes);
        std::fs::create_dir_all(store.join(BLOBS)).unwrap();
        let short = store.join(BLOBS).join(format!("{}.png", &hash[..24]));
        std::fs::write(&short, b"damaged").unwrap();
        let mut result = document("![x](.markitai/assets/x.png)\n");
        persist(
            &store,
            &mut result,
            &[asset("x.png", bytes)],
            &[],
            &json!({}),
        );
        assert!(result.warnings.is_empty(), "{:?}", result.warnings);
        assert_eq!(std::fs::read(&short).unwrap(), b"damaged");
        let full = store.join(BLOBS).join(format!("{hash}.png"));
        assert_eq!(std::fs::read(&full).unwrap(), bytes);
        assert!(result.markdown.contains(&format!("/{hash}.png)")));
    }

    #[test]
    fn an_image_that_cannot_be_saved_keeps_its_reference_while_others_are_linked() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        let blocked = b"blocked".as_slice();
        let hash = digest(blocked);
        let blobs = store.join(BLOBS);
        std::fs::create_dir_all(&blobs).unwrap();
        // Both names these bytes could use already hold other content.
        for name in [format!("{}.png", &hash[..24]), format!("{hash}.png")] {
            std::fs::write(blobs.join(name), b"other").unwrap();
        }
        let mut result = document("![a](.markitai/assets/a.png) ![b](.markitai/assets/b.png)\n");
        let assets = [asset("a.png", blocked), asset("b.png", b"fine")];
        persist(&store, &mut result, &assets, &[], &json!({}));
        assert!(
            result
                .markdown
                .starts_with("![a](.markitai/assets/a.png) ![b](file://"),
            "{}",
            result.markdown
        );
        assert_eq!(result.warnings.len(), 1, "{:?}", result.warnings);
        let warning = &result.warnings[0];
        assert!(
            warning.starts_with("1 image could not be saved to ")
                && warning.contains("its reference still points into .markitai/")
                && warning.contains("Existing asset content does not match"),
            "{warning}"
        );
        assert_eq!(result.assets.len(), 1);
    }

    #[test]
    fn an_unusable_store_keeps_references_and_explains_why() {
        let root = tempfile::tempdir().unwrap();
        let store = root.path().join("store");
        std::fs::write(&store, b"a file, not a directory").unwrap();
        let markdown = "![x](.markitai/assets/x.png) ![y](.markitai/assets/y.png)\n";
        let mut result = document(markdown);
        let assets = [asset("x.png", b"x"), asset("y.png", b"y")];
        persist(&store, &mut result, &assets, &[], &json!({}));
        assert_eq!(result.markdown, markdown);
        assert_eq!(result.warnings.len(), 1);
        let warning = &result.warnings[0];
        assert!(
            warning.starts_with("2 images could not be saved to "),
            "{warning}"
        );
        assert!(warning.contains("their references still point into .markitai/"));
        assert!(warning.contains("Use -o DIR"));
        assert_eq!(std::fs::read(&store).unwrap(), b"a file, not a directory");
    }

    #[cfg(unix)]
    #[test]
    fn a_symlinked_store_is_refused_unless_symlinks_are_allowed() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("elsewhere");
        std::fs::create_dir(&target).unwrap();
        let store = root.path().join("store");
        std::os::unix::fs::symlink(&target, &store).unwrap();
        let markdown = "![x](.markitai/assets/x.png)\n";
        let convert = |cfg: Value| {
            let mut result = document(markdown);
            persist(&store, &mut result, &[asset("x.png", b"x")], &[], &cfg);
            result
        };
        let refused = convert(json!({}));
        assert_eq!(refused.markdown, markdown);
        assert_eq!(refused.warnings.len(), 1, "{:?}", refused.warnings);
        assert!(refused.warnings[0].starts_with("1 image could not be saved"));
        assert!(refused.warnings[0].contains("Symlink access is disabled"));
        assert_eq!(std::fs::read_dir(&target).unwrap().count(), 0);
        let allowed = convert(json!({"output": {"allow_symlinks": true}}));
        assert!(allowed.warnings.is_empty(), "{:?}", allowed.warnings);
        assert!(allowed.markdown.contains("file://"));
        assert_eq!(std::fs::read_dir(target.join(BLOBS)).unwrap().count(), 1);
    }
}
