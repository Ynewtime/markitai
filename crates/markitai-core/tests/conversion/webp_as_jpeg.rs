//! `image.format: webp` below quality 100 is written as JPEG (this build has
//! no lossy WebP encoder): names, links and bytes agree, and the document
//! carries one notice however many pictures or pages it has.
use super::*;
use std::path::Path;

const MIXED: &[u8] = include_bytes!("../../src/pdf_raster/fixtures/mixed-native-scanned-blank.pdf");

fn run(
    path: &Path,
    cfg: Value,
    output_dir: &Path,
) -> markitai_core::Result<markitai_core::ConversionOutput> {
    convert(
        path.to_str().unwrap(),
        ConvertOptions {
            config: Some(cfg),
            output_dir: Some(output_dir.to_owned()),
            ..Default::default()
        },
    )
}

#[test]
fn pdf_webp_screenshots_below_quality_100_are_jpg_with_one_notice() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.pdf");
    std::fs::write(&path, MIXED).unwrap();
    for (quality, extension, format, notices) in [
        (80, "jpg", image::ImageFormat::Jpeg, 1),
        (100, "webp", image::ImageFormat::WebP, 0),
    ] {
        let config = json!({"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false},"screenshot":{"enabled":true},"image":{"format":"webp","quality":quality}});
        let output = dir.path().join(format!("output-{quality}"));
        let result = run(&path, config, &output).unwrap();
        assert_eq!(result.screenshots.len(), 3);
        for capture in &result.screenshots {
            assert_eq!(capture.extension().unwrap(), extension);
            assert_eq!(
                image::guess_format(&std::fs::read(capture).unwrap()).unwrap(),
                format
            );
        }
        for page in 1..=3 {
            let link = format!(".pdf.page{page:04}.{extension})");
            assert!(result.markdown.contains(&link), "{}", result.markdown);
        }
        // One notice for the document, not one per page.
        assert_eq!(
            result
                .warnings
                .iter()
                .filter(|warning| warning.starts_with("image.format webp:"))
                .count(),
            notices,
            "{:?}",
            result.warnings
        );
    }
}
