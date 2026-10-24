//! `image.format: webp` page screenshots are lossy WebP at `image.quality`, as
//! Python markitai 1.2 wrote them: names, links and bytes agree, and there is
//! no notice.
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
fn pdf_webp_screenshots_are_lossy_webp_at_image_quality() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("mixed.pdf");
    std::fs::write(&path, MIXED).unwrap();
    let mut sizes = Vec::new();
    for quality in [30, 80, 100] {
        let config = json!({"cache":{"enabled":false},"history":{"record":false},"llm":{"enabled":false},"screenshot":{"enabled":true},"image":{"format":"webp","quality":quality}});
        let output = dir.path().join(format!("output-{quality}"));
        let result = run(&path, config, &output).unwrap();
        assert_eq!(result.screenshots.len(), 3);
        let mut total = 0;
        for capture in &result.screenshots {
            assert_eq!(capture.extension().unwrap(), "webp");
            let bytes = std::fs::read(capture).unwrap();
            assert_eq!(
                image::guess_format(&bytes).unwrap(),
                image::ImageFormat::WebP
            );
            // Lossy (`VP8 `) at every quality, as Pillow writes it.
            assert_eq!(&bytes[12..16], b"VP8 ");
            total += bytes.len();
        }
        sizes.push(total);
        for page in 1..=3 {
            let link = format!(".pdf.page{page:04}.webp)");
            assert!(result.markdown.contains(&link), "{}", result.markdown);
        }
        assert!(
            !result
                .warnings
                .iter()
                .any(|warning| warning.contains("webp")
                    || warning.contains("WebP")
                    || warning.contains("JPEG")),
            "{:?}",
            result.warnings
        );
    }
    assert!(sizes[0] < sizes[1] && sizes[1] < sizes[2], "{sizes:?}");
}
