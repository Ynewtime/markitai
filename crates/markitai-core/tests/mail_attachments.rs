//! Actual disk publication preserves downloadable mail payloads independently
//! of preview compression/filtering. All fixtures are synthetic and local.

use base64::Engine;
use markitai_core::{ConversionOutput, ConvertOptions, convert, formats};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::io::{Cursor, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy)]
enum Kind {
    Eml,
    Msg,
}

impl Kind {
    fn extension(self) -> &'static str {
        match self {
            Self::Eml => "eml",
            Self::Msg => "msg",
        }
    }
    fn asset_name(self, index: usize, filename: &str) -> String {
        let prefix = match self {
            Self::Eml => "email",
            Self::Msg => "msg",
        };
        format!("{prefix}-{index}-{filename}")
    }
}

struct Attachment<'a> {
    name: &'a str,
    bytes: &'a [u8],
    cid: Option<&'a str>,
    inline: bool,
    classification: Option<&'a [(u32, u64)]>,
}

fn png(width: u32, height: u32) -> Vec<u8> {
    let mut bytes = Cursor::new(Vec::new());
    image::RgbImage::from_fn(width, height, |x, y| {
        image::Rgb([(x % 256) as u8, (y % 256) as u8, ((x + y) % 256) as u8])
    })
    .write_to(&mut bytes, image::ImageFormat::Png)
    .unwrap();
    bytes.into_inner()
}

fn properties(header: usize, values: &[(u32, u64)]) -> Vec<u8> {
    let mut bytes = vec![0; header];
    for &(tag, value) in values {
        bytes.extend(tag.to_le_bytes());
        bytes.extend(0u32.to_le_bytes());
        bytes.extend(value.to_le_bytes());
    }
    bytes
}

fn utf16(value: &str) -> Vec<u8> {
    value.encode_utf16().flat_map(u16::to_le_bytes).collect()
}

fn message(kind: Kind, body: &str, html: bool, attachments: &[Attachment<'_>]) -> Vec<u8> {
    match kind {
        Kind::Eml => {
            let encode = |bytes| base64::engine::general_purpose::STANDARD.encode(bytes);
            let mime = if html { "text/html" } else { "text/plain" };
            let mut message = format!(
                "MIME-Version: 1.0\r\nSubject: Attachment fixture\r\nContent-Type: multipart/related; boundary=fixture\r\n\r\n--fixture\r\nContent-Type: {mime}; charset=utf-8\r\nContent-Transfer-Encoding: base64\r\n\r\n{}\r\n",
                encode(body.as_bytes()),
            );
            for attachment in attachments {
                let cid = attachment
                    .cid
                    .map_or_else(String::new, |cid| format!("Content-ID: <{cid}>\r\n"));
                let disposition = if attachment.inline {
                    "inline"
                } else {
                    "attachment"
                };
                message.push_str(&format!(
                    "--fixture\r\nContent-Type: image/png\r\nContent-Disposition: {disposition}; filename=\"{}\"\r\n{cid}Content-Transfer-Encoding: base64\r\n\r\n{}\r\n",
                    attachment.name, encode(attachment.bytes),
                ));
            }
            message.push_str("--fixture--\r\n");
            message.into_bytes()
        }
        Kind::Msg => {
            let mut compound = cfb::CompoundFile::create(Cursor::new(Vec::new())).unwrap();
            let mut put = |path: &str, bytes: &[u8]| {
                compound
                    .create_storage_all(Path::new(path).parent().unwrap())
                    .unwrap();
                compound
                    .create_stream(path)
                    .unwrap()
                    .write_all(bytes)
                    .unwrap();
            };
            put(
                "/__properties_version1.0",
                &properties(32, &[(0x3fde0003, 65001)]),
            );
            put("/__substg1.0_0037001F", &utf16("Attachment fixture"));
            if html {
                put("/__substg1.0_10130102", body.as_bytes());
            } else {
                put("/__substg1.0_1000001F", &utf16(body));
            }
            for (index, attachment) in attachments.iter().enumerate() {
                let prefix = format!("/__attach_version1.0_#{index:08X}");
                let default_classification = if attachment.inline {
                    vec![(0x7ffe000b, 1), (0x37140003, 4)]
                } else {
                    vec![(0x7ffe000b, 0)]
                };
                let mut values = vec![(0x37050003, 1)];
                values.extend_from_slice(
                    attachment.classification.unwrap_or(&default_classification),
                );
                put(
                    &format!("{prefix}/__properties_version1.0"),
                    &properties(8, &values),
                );
                put(
                    &format!("{prefix}/__substg1.0_3707001F"),
                    &utf16(attachment.name),
                );
                put(&format!("{prefix}/__substg1.0_37010102"), attachment.bytes);
                if let Some(cid) = attachment.cid {
                    put(&format!("{prefix}/__substg1.0_3712001F"), &utf16(cid));
                }
            }
            compound.into_inner().into_inner()
        }
    }
}

fn publish(root: &Path, kind: Kind, source: &[u8], config: Value) -> ConversionOutput {
    let input = root.join(format!("input.{}", kind.extension()));
    std::fs::write(&input, source).unwrap();
    let before = Sha256::digest(source);
    let result = convert(
        input.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(root.join("output")),
            config: Some(config),
            llm: Some(false),
            ocr: Some(false),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(std::fs::read(&input).unwrap(), source);
    assert_eq!(Sha256::digest(std::fs::read(input).unwrap()), before);
    assert!(result.output_path.as_ref().unwrap().is_file());
    result
}

fn target<'a>(markdown: &'a str, prefix: &str) -> &'a str {
    markdown
        .split_once(prefix)
        .unwrap_or_else(|| panic!("missing {prefix:?}: {markdown}"))
        .1
        .split_once(')')
        .unwrap()
        .0
}

fn bytes_at(result: &ConversionOutput, target: &str) -> Vec<u8> {
    let path: PathBuf = result
        .output_path
        .as_ref()
        .unwrap()
        .parent()
        .unwrap()
        .join(target.split(['?', '#']).next().unwrap());
    assert!(result.assets.contains(&path), "unrecorded asset: {path:?}");
    std::fs::read(path).unwrap()
}

#[test]
fn explicit_png_and_sniffed_image_downloads_keep_original_bytes_under_image_filters() {
    for kind in [Kind::Eml, Kind::Msg] {
        for name in ["diagram.png", "payload.bin"] {
            let root = tempfile::tempdir().unwrap();
            let image = png(2, 2);
            let source = message(
                kind,
                "Read the attached original.",
                false,
                &[Attachment {
                    name,
                    bytes: &image,
                    cid: None,
                    inline: false,
                    classification: None,
                }],
            );
            let result = publish(
                root.path(),
                kind,
                &source,
                json!({"image":{"compress":true,"filter":{"min_width":2000,"min_height":2000}}}),
            );
            let download = target(&result.markdown, &format!("[{name}]("));
            assert_eq!(bytes_at(&result, download), image);
            assert_eq!(result.assets.len(), 1);
            assert!(!result.markdown.contains(&format!("![{name}]")));
            assert_eq!(
                Sha256::digest(bytes_at(&result, download)),
                Sha256::digest(&image)
            );
            assert!(download.ends_with(name.rsplit('.').next().unwrap()));
        }
    }
}

#[test]
fn a_plain_text_image_reference_cannot_reclassify_a_byvalue_or_mime_attachment() {
    for kind in [Kind::Eml, Kind::Msg] {
        let root = tempfile::tempdir().unwrap();
        let image = png(120, 80);
        let name = kind.asset_name(1, "diagram.png");
        let source = message(
            kind,
            &format!("![Author preview](.markitai/assets/{name})"),
            false,
            &[Attachment {
                name: "diagram.png",
                bytes: &image,
                cid: None,
                inline: false,
                classification: None,
            }],
        );
        let result = publish(root.path(), kind, &source, json!({}));
        let download = target(&result.markdown, "[diagram.png](");
        let preview = target(&result.markdown, "![Author preview](");
        assert_ne!(download, preview);
        assert_eq!(bytes_at(&result, download), image);
        assert_eq!(
            image::guess_format(&bytes_at(&result, preview)).unwrap(),
            image::ImageFormat::Jpeg
        );
        assert_eq!(result.assets.len(), 2);
    }
}

#[test]
fn cid_body_images_are_optimized_without_replacing_a_same_payload_original_download() {
    for kind in [Kind::Eml, Kind::Msg] {
        for mixed in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let image = png(120, 80);
            let mut attachments = vec![Attachment {
                name: "inline.png",
                bytes: &image,
                cid: Some("logo"),
                inline: true,
                classification: None,
            }];
            if mixed {
                attachments.push(Attachment {
                    name: "diagram.png",
                    bytes: &image,
                    cid: None,
                    inline: false,
                    classification: None,
                });
            }
            let source = message(
                kind,
                "<p>Body.</p><img src='cid:logo' alt='Inline'>",
                true,
                &attachments,
            );
            let result = publish(root.path(), kind, &source, json!({}));
            let preview = target(&result.markdown, "![Inline](");
            assert_eq!(
                image::guess_format(&bytes_at(&result, preview)).unwrap(),
                image::ImageFormat::Jpeg
            );
            assert!(!result.markdown.contains("[inline.png]("));
            assert_eq!(result.assets.len(), if mixed { 2 } else { 1 });
            if mixed {
                let download = target(&result.markdown, "[diagram.png](");
                assert_ne!(download, preview);
                assert_eq!(bytes_at(&result, download), image);
            }
        }
    }
}

#[test]
fn an_unused_cid_or_a_cid_download_link_is_not_an_inline_image() {
    for kind in [Kind::Eml, Kind::Msg] {
        for body in [
            "<p>Body without an image.</p>",
            "<p><a href='cid:logo'>Download</a></p>",
        ] {
            let root = tempfile::tempdir().unwrap();
            let image = png(120, 80);
            let source = message(
                kind,
                body,
                true,
                &[Attachment {
                    name: "diagram.png",
                    bytes: &image,
                    cid: Some("logo"),
                    inline: true,
                    classification: None,
                }],
            );
            let result = publish(root.path(), kind, &source, json!({}));
            let download = target(&result.markdown, "[diagram.png](");
            assert_eq!(bytes_at(&result, download), image);
            assert_eq!(
                Sha256::digest(bytes_at(&result, download)),
                Sha256::digest(&image)
            );
            // A download/unused CID still cannot bind a body image. EML's
            // ordinary image attachment preview is independent of that CID.
            if matches!(kind, Kind::Eml) {
                let (body, attachments) = result.markdown.split_once("## Attachments").unwrap();
                assert!(!body.contains("!["));
                let preview = target(attachments, "![diagram.png](");
                assert_ne!(download, preview);
                assert_eq!(
                    image::guess_format(&bytes_at(&result, preview)).unwrap(),
                    image::ImageFormat::Jpeg
                );
                assert_eq!(
                    image::load_from_memory(&bytes_at(&result, preview))
                        .unwrap()
                        .width(),
                    120
                );
                assert_eq!(result.assets.len(), 2);
            } else {
                assert!(!result.markdown.contains("!["));
                assert_eq!(result.assets.len(), 1);
            }
        }
    }
}

#[test]
fn filtering_an_inline_cid_image_does_not_filter_a_matching_download() {
    for kind in [Kind::Eml, Kind::Msg] {
        let root = tempfile::tempdir().unwrap();
        let image = png(2, 2);
        let source = message(
            kind,
            "<p>Body.</p><img src='cid:logo' alt='Inline'>",
            true,
            &[
                Attachment {
                    name: "inline.png",
                    bytes: &image,
                    cid: Some("logo"),
                    inline: true,
                    classification: None,
                },
                Attachment {
                    name: "diagram.png",
                    bytes: &image,
                    cid: None,
                    inline: false,
                    classification: None,
                },
            ],
        );
        let result = publish(root.path(), kind, &source, json!({}));
        assert!(!result.markdown.contains("![Inline]"));
        assert!(!result.markdown.contains("[inline.png]("));
        let download = target(&result.markdown, "[diagram.png](");
        assert_eq!(bytes_at(&result, download), image);
        assert_eq!(result.assets.len(), 1);
    }
}

#[test]
fn explicit_cid_attachments_keep_original_downloads_even_when_the_preview_is_filtered() {
    for kind in [Kind::Eml, Kind::Msg] {
        for (width, height, filtered) in [(120, 80, false), (2, 2, true)] {
            let root = tempfile::tempdir().unwrap();
            let image = png(width, height);
            let source = message(
                kind,
                "<p>Both a body picture and a downloadable attachment.</p><img src='cid:logo' alt='Preview'>",
                true,
                &[Attachment {
                    name: "diagram.png",
                    bytes: &image,
                    cid: Some("logo"),
                    inline: false,
                    classification: None,
                }],
            );
            let result = publish(root.path(), kind, &source, json!({}));
            let download = target(&result.markdown, "[diagram.png](");
            assert_eq!(bytes_at(&result, download), image);
            assert_eq!(
                Sha256::digest(bytes_at(&result, download)),
                Sha256::digest(&image)
            );
            if filtered {
                assert!(!result.markdown.contains("![Preview]"));
                assert_eq!(result.assets.len(), 1);
            } else {
                let preview = target(&result.markdown, "![Preview](");
                assert_ne!(download, preview);
                assert_eq!(
                    image::guess_format(&bytes_at(&result, preview)).unwrap(),
                    image::ImageFormat::Jpeg
                );
                assert_eq!(result.assets.len(), 2);
            }
        }
    }
}

#[test]
fn eml_unknown_dispositions_keep_normal_and_tiny_cid_originals() {
    for disposition in [
        "x-project-download",
        "X-PROJECT-DOWNLOAD",
        "attachment",
        "INLINE",
        "",
    ] {
        for (width, height, filtered) in [(120, 80, false), (2, 2, true)] {
            let root = tempfile::tempdir().unwrap();
            let image = png(width, height);
            let encoded = message(
                Kind::Eml,
                "<p>Body.</p><img src='cid:logo' alt='Preview'>",
                true,
                &[Attachment {
                    name: "diagram.png",
                    bytes: &image,
                    cid: Some("logo"),
                    inline: false,
                    classification: None,
                }],
            );
            let mime = String::from_utf8(encoded).unwrap();
            let header = "Content-Disposition: attachment; filename=\"diagram.png\"\r\n";
            assert_eq!(mime.matches(header).count(), 1);
            let replacement = if disposition.is_empty() {
                String::new()
            } else {
                format!("Content-Disposition: {disposition}; filename=\"diagram.png\"\r\n")
            };
            let mime = mime.replace(header, &replacement).replace(
                "Content-Type: image/png\r\n",
                "Content-Type: image/png; name=\"diagram.png\"\r\n",
            );
            let result = publish(root.path(), Kind::Eml, mime.as_bytes(), json!({}));
            let download = !disposition.is_empty() && !disposition.eq_ignore_ascii_case("inline");
            assert_eq!(
                result.markdown.contains("[diagram.png]("),
                download,
                "{disposition:?}"
            );
            if download {
                let target = target(&result.markdown, "[diagram.png](");
                assert_eq!(bytes_at(&result, target), image, "{disposition:?}");
                assert_eq!(
                    Sha256::digest(bytes_at(&result, target)),
                    Sha256::digest(&image),
                    "{disposition:?}"
                );
            }
            if filtered {
                assert!(!result.markdown.contains("![Preview]"), "{disposition:?}");
                assert_eq!(
                    result.assets.len(),
                    usize::from(download),
                    "{disposition:?}"
                );
            } else {
                let preview = target(&result.markdown, "![Preview](");
                assert_eq!(
                    image::guess_format(&bytes_at(&result, preview)).unwrap(),
                    image::ImageFormat::Jpeg,
                    "{disposition:?}"
                );
                assert_eq!(
                    result.assets.len(),
                    if download { 2 } else { 1 },
                    "{disposition:?}"
                );
            }
        }
    }
}

#[test]
fn msg_inline_properties_are_typed_conservative_and_require_actual_html_image_use() {
    type Classification = (&'static str, Vec<(u32, u64)>, bool, bool);
    let classifications: Vec<Classification> = vec![
        (
            "hidden-rendered",
            vec![(0x7ffe000b, 1), (0x37140003, 4)],
            true,
            false,
        ),
        (
            "hidden-rendered-rtf-invisible",
            vec![(0x7ffe000b, 1), (0x37140003, 6)],
            true,
            false,
        ),
        (
            "visible-rendered",
            vec![(0x7ffe000b, 0), (0x37140003, 4)],
            false,
            false,
        ),
        ("missing-hidden", vec![(0x37140003, 4)], false, false),
        ("missing-flags", vec![(0x7ffe000b, 1)], false, false),
        ("missing-both", vec![], false, false),
        (
            "not-rendered",
            vec![(0x7ffe000b, 1), (0x37140003, 0)],
            false,
            false,
        ),
        (
            "html-invisible",
            vec![(0x7ffe000b, 1), (0x37140003, 5)],
            false,
            true,
        ),
        (
            "bad-hidden-value",
            vec![(0x7ffe000b, 2), (0x37140003, 4)],
            false,
            true,
        ),
        (
            "wrong-hidden-type",
            vec![(0x7ffe0003, 1), (0x37140003, 4)],
            false,
            true,
        ),
        (
            "wrong-flags-type",
            vec![(0x7ffe000b, 1), (0x3714000b, 4)],
            false,
            true,
        ),
        (
            "two-hidden-types",
            vec![(0x7ffe000b, 1), (0x7ffe0003, 1), (0x37140003, 4)],
            false,
            true,
        ),
        (
            "two-flags-types",
            vec![(0x7ffe000b, 1), (0x37140003, 4), (0x3714000b, 1)],
            false,
            true,
        ),
        (
            "conflicting-hidden",
            vec![(0x7ffe000b, 0), (0x7ffe000b, 1), (0x37140003, 4)],
            false,
            true,
        ),
        (
            "conflicting-flags",
            vec![(0x7ffe000b, 1), (0x37140003, 0), (0x37140003, 4)],
            false,
            true,
        ),
    ];
    for (name, classification, single_inline, warning) in classifications {
        let root = tempfile::tempdir().unwrap();
        let image = png(120, 80);
        let source = message(
            Kind::Msg,
            "<p>Body.</p><img src='cid:logo' alt='Inline'>",
            true,
            &[Attachment {
                name: "diagram.png",
                bytes: &image,
                cid: Some("logo"),
                inline: true,
                classification: Some(&classification),
            }],
        );
        let result = publish(root.path(), Kind::Msg, &source, json!({}));
        let preview = target(&result.markdown, "![Inline](");
        assert_eq!(
            image::guess_format(&bytes_at(&result, preview)).unwrap(),
            image::ImageFormat::Jpeg,
            "{name}"
        );
        assert_eq!(
            result.assets.len(),
            if single_inline { 1 } else { 2 },
            "{name}"
        );
        assert_eq!(
            result.markdown.contains("[diagram.png]("),
            !single_inline,
            "{name}"
        );
        if !single_inline {
            let download = target(&result.markdown, "[diagram.png](");
            assert_ne!(download, preview, "{name}");
            assert_eq!(bytes_at(&result, download), image, "{name}");
            assert_eq!(
                Sha256::digest(bytes_at(&result, download)),
                Sha256::digest(&image),
                "{name}"
            );
        }
        assert_eq!(
            result
                .warnings
                .iter()
                .any(|value| value.contains("inline classification")),
            warning,
            "{name}"
        );
    }
    // Valid flags still do not turn a plain-text image-looking path into a CID binding.
    let root = tempfile::tempdir().unwrap();
    let image = png(120, 80);
    let source = message(
        Kind::Msg,
        "![Preview](.markitai/assets/msg-1-diagram.png)",
        false,
        &[Attachment {
            name: "diagram.png",
            bytes: &image,
            cid: Some("logo"),
            inline: true,
            classification: None,
        }],
    );
    let result = publish(root.path(), Kind::Msg, &source, json!({}));
    assert_eq!(
        bytes_at(&result, target(&result.markdown, "[diagram.png](")),
        image
    );
    assert_eq!(result.assets.len(), 2);
}

#[test]
fn a_shared_reference_definition_downloads_the_original_and_previews_only_image_uses() {
    for kind in [Kind::Eml, Kind::Msg] {
        let root = tempfile::tempdir().unwrap();
        let image = png(120, 80);
        let name = kind.asset_name(1, "diagram.png");
        let encoded = format!("%{:02X}{}", name.as_bytes()[0], &name[1..]);
        let body = format!(
            "[Original][both]\n![Body preview][both]\n\n[both]: .markitai/assets/{encoded}?variant=%2F#detail \"Source title\"\n"
        );
        let source = message(
            kind,
            &body,
            false,
            &[Attachment {
                name: "diagram.png",
                bytes: &image,
                cid: None,
                inline: false,
                classification: None,
            }],
        );
        let result = publish(root.path(), kind, &source, json!({}));
        assert!(result.markdown.contains("[Original][both]"));
        let definition = result
            .markdown
            .split_once("[both]: ")
            .unwrap()
            .1
            .lines()
            .next()
            .unwrap();
        let original = definition.split(' ').next().unwrap();
        assert!(
            original.ends_with("?variant=%2F#detail"),
            "{}",
            result.markdown
        );
        assert_eq!(bytes_at(&result, original), image);
        let preview = target(&result.markdown, "![Body preview](")
            .split(' ')
            .next()
            .unwrap();
        assert!(
            preview.ends_with("?variant=%2F#detail"),
            "{}",
            result.markdown
        );
        assert_ne!(original, preview);
        assert!(result.markdown.contains("#detail \"Source title\")"));
        assert_eq!(
            image::guess_format(&bytes_at(&result, preview)).unwrap(),
            image::ImageFormat::Jpeg
        );
        assert_eq!(result.assets.len(), 2);
    }
}

#[test]
fn malformed_msg_methods_stay_strict_and_do_not_gain_attachment_data() {
    let image = png(120, 80);
    let classification = [(0x37050003, 2), (0x7ffe000b, 1), (0x37140003, 4)];
    let source = message(
        Kind::Msg,
        "<p>Readable body.</p>",
        true,
        &[Attachment {
            name: "diagram.png",
            bytes: &image,
            cid: Some("logo"),
            inline: true,
            classification: Some(&classification),
        }],
    );
    let root = tempfile::tempdir().unwrap();
    let result = publish(root.path(), Kind::Msg, &source, json!({}));
    assert!(result.markdown.contains("Readable body."));
    assert!(result.assets.is_empty());
    assert!(
        result
            .warnings
            .iter()
            .any(|value| value.contains("conflicting property 37050003"))
    );
}

#[test]
fn duplicate_downloads_keep_both_names_and_safe_source_filename_labels() {
    for kind in [Kind::Eml, Kind::Msg] {
        let root = tempfile::tempdir().unwrap();
        let image = png(2, 2);
        let source = message(
            kind,
            "Both attachments remain downloadable.",
            false,
            &[
                Attachment {
                    name: "../CON.png",
                    bytes: &image,
                    cid: None,
                    inline: false,
                    classification: None,
                },
                Attachment {
                    name: "../../报告.png",
                    bytes: &image,
                    cid: None,
                    inline: false,
                    classification: None,
                },
            ],
        );
        let input = root.path().join(format!("safe.{}", kind.extension()));
        std::fs::write(&input, &source).unwrap();
        let extracted = formats::extract(&input).unwrap();
        assert_eq!(extracted.assets.len(), 2);
        assert_eq!(extracted.assets[0].name, kind.asset_name(1, "_CON.png"));
        assert_eq!(extracted.assets[1].name, kind.asset_name(2, "报告.png"));
        assert!(
            extracted
                .assets
                .iter()
                .all(|asset| !asset.name.contains(['/', '\\']))
        );
        assert!(extracted.assets.iter().all(|asset| asset.bytes == image));
        let result = publish(root.path(), kind, &source, json!({}));
        // The shared content-addressed publication may reuse identical bytes;
        // both source filename labels and links still remain in the document.
        let first = target(&result.markdown, "CON.png](");
        let second = target(&result.markdown, "报告.png](");
        assert_eq!(first, second);
        assert_eq!(bytes_at(&result, first), image);
        assert_eq!(result.assets.len(), 1);
    }
}

#[test]
fn unbound_eml_images_keep_exact_downloads_and_independently_filtered_previews() {
    for disposition in ["attachment", "x-project-download", "inline", ""] {
        for cid in [None, Some("unused")] {
            for (width, height, filtered) in [(120, 80, false), (2, 2, true)] {
                let root = tempfile::tempdir().unwrap();
                let image = png(width, height);
                let encoded = message(
                    Kind::Eml,
                    "<p>Unbound attachment body survives.</p><p><a href='cid:unused'>CID download only</a></p>",
                    true,
                    &[Attachment {
                        name: "diagram.png",
                        bytes: &image,
                        cid,
                        inline: false,
                        classification: None,
                    }],
                );
                let mime = String::from_utf8(encoded).unwrap();
                let header = "Content-Disposition: attachment; filename=\"diagram.png\"\r\n";
                assert_eq!(mime.matches(header).count(), 1);
                let declaration = if disposition.is_empty() {
                    String::new()
                } else {
                    format!("Content-Disposition: {disposition}; filename=\"diagram.png\"\r\n")
                };
                let mime = mime.replace(header, &declaration).replace(
                    "Content-Type: image/png\r\n",
                    "Content-Type: image/png; name=diagram.png\r\n",
                );
                let result = publish(root.path(), Kind::Eml, mime.as_bytes(), json!({}));
                let (body, attachments) = result.markdown.split_once("## Attachments").unwrap();
                assert!(body.contains("Unbound attachment body survives."));
                assert!(body.contains("CID download only"));
                assert!(!body.contains("!["), "{disposition:?}, {cid:?}");
                let original = target(attachments, "[diagram.png](");
                assert_eq!(bytes_at(&result, original), image);
                assert_eq!(
                    Sha256::digest(bytes_at(&result, original)),
                    Sha256::digest(&image)
                );
                if filtered {
                    assert!(!attachments.contains("!["));
                    assert_eq!(result.assets.len(), 1);
                } else {
                    let preview = target(attachments, "![diagram.png](");
                    assert_ne!(original, preview);
                    let prepared = bytes_at(&result, preview);
                    assert_eq!(
                        image::guess_format(&prepared).unwrap(),
                        image::ImageFormat::Jpeg
                    );
                    let decoded = image::load_from_memory(&prepared).unwrap();
                    assert_eq!((decoded.width(), decoded.height()), (width, height));
                    assert_eq!(result.assets.len(), 2);
                }
            }
        }
    }
}

#[test]
fn eml_octet_stream_pngs_and_damaged_transfer_payloads_never_gain_attachment_previews() {
    let image = png(120, 80);
    for damaged in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let encoded = message(
            Kind::Eml,
            "<p>Opaque attachments stay downloadable.</p><img src='cid:logo' alt='Unresolved body image'>",
            true,
            &[Attachment {
                name: "diagram.png",
                bytes: &image,
                cid: Some("logo"),
                inline: false,
                classification: None,
            }],
        );
        let mime = String::from_utf8(encoded).unwrap();
        let mime = if damaged {
            let valid = base64::engine::general_purpose::STANDARD.encode(&image);
            assert_eq!(mime.matches(&valid).count(), 1);
            mime.replace(&valid, "!not base64!")
        } else {
            mime.replace(
                "Content-Type: image/png\r\n",
                "Content-Type: application/octet-stream\r\n",
            )
        };
        let input = root.path().join("parser.eml");
        std::fs::write(&input, mime.as_bytes()).unwrap();
        let extracted = formats::extract(&input).unwrap();
        assert_eq!(extracted.assets.len(), 1);
        // Damaged transfer data has parser-recovered bytes, not a promise of
        // exact transport payload. Compare publication with that recovered buffer.
        let original = extracted.assets[0].bytes.clone();
        if !damaged {
            assert_eq!(original, image);
        }
        let result = publish(root.path(), Kind::Eml, mime.as_bytes(), json!({}));
        let (body, attachments) = result.markdown.split_once("## Attachments").unwrap();
        assert!(body.contains("![Unresolved body image](cid:logo)"));
        assert!(!attachments.contains("!["));
        let download = target(attachments, "[diagram.png](");
        assert_eq!(bytes_at(&result, download), original);
        assert_eq!(
            Sha256::digest(bytes_at(&result, download)),
            Sha256::digest(&original)
        );
        assert_eq!(result.assets.len(), 1);
        assert!(
            result
                .warnings
                .iter()
                .any(|warning| warning.contains("Content-ID"))
        );
        if damaged {
            assert!(
                result
                    .warnings
                    .iter()
                    .any(|warning| warning.contains("malformed transfer encoding"))
            );
        }
    }
}
