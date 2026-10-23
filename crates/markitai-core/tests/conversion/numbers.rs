use markitai_core::{ConvertOptions, convert};
use serde_json::json;

#[test]
fn numbers_public_api_and_publication_keep_real_fixture_tables() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("工作簿.numbers");
    std::fs::write(
        &source,
        include_bytes!("../../src/formats/numbers/fixtures/test-1.numbers"),
    )
    .unwrap();
    let options = || ConvertOptions {
        config: Some(
            json!({"llm":{"enabled":false},"ocr":{"enabled":false},"cache":{"enabled":false},"history":{"enabled":false}}),
        ),
        ..Default::default()
    };
    let memory = convert(source.to_str().unwrap(), options()).unwrap();
    assert_eq!(
        markitai_core::formats::extract(&source).unwrap().metadata["format"],
        "NUMBERS"
    );
    assert_eq!(memory.frontmatter["source"], "工作簿.numbers");
    assert!(memory.markdown.contains("YYY\\_4\\_2"));
    assert!(memory.markdown.contains("XXX\\_3\\_5"));
    assert!(memory.assets.is_empty());
    let disk = convert(
        source.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(temp.path().join("out")),
            ..options()
        },
    )
    .unwrap();
    assert_eq!(memory.markdown, disk.markdown);
    let saved = std::fs::read_to_string(disk.output_path.unwrap()).unwrap();
    assert!(saved.ends_with(&memory.markdown));
}

fn unpack_package(bytes: &[u8], target: &std::path::Path) {
    std::fs::create_dir_all(target).unwrap();
    let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).unwrap();
    for index in (0..archive.len()).rev() {
        let mut entry = archive.by_index(index).unwrap();
        let path = target.join(entry.enclosed_name().unwrap());
        if entry.is_dir() {
            std::fs::create_dir_all(path).unwrap();
        } else {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::io::copy(&mut entry, &mut std::fs::File::create(path).unwrap()).unwrap();
        }
    }
}

fn package_options() -> ConvertOptions {
    ConvertOptions {
        config: Some(json!({
            "llm":{"enabled":false},"ocr":{"enabled":false},
            "screenshot":{"enabled":false},"cache":{"enabled":false},
            "history":{"record":false},"image":{"alt_enabled":false,"desc_enabled":false}
        })),
        ..Default::default()
    }
}

#[test]
fn modern_directory_packages_match_both_independent_zip_fixtures_through_public_api() {
    for bytes in [
        include_bytes!("../../src/formats/numbers/fixtures/test-1.numbers").as_slice(),
        include_bytes!("../../src/formats/numbers/fixtures/test-formats.numbers").as_slice(),
    ] {
        let temp = tempfile::tempdir().unwrap();
        let file = temp.path().join("single/预算.NUMBERS");
        let package = temp.path().join("package/预算.NUMBERS");
        std::fs::create_dir(file.parent().unwrap()).unwrap();
        std::fs::write(&file, bytes).unwrap();
        unpack_package(bytes, &package);
        assert!(!markitai_core::formats::is_numbers_package_path(&file));
        assert!(markitai_core::formats::is_numbers_package_path(&package));
        let mut file_document = markitai_core::formats::extract(&file).unwrap();
        let mut package_document = markitai_core::formats::extract(&package).unwrap();
        assert_eq!(package_document.metadata["format"], "NUMBERS");
        assert_eq!(
            package_document.metadata.remove("source").unwrap(),
            package.to_str().unwrap()
        );
        assert_eq!(
            file_document.metadata.remove("source").unwrap(),
            file.to_str().unwrap()
        );
        assert_eq!(file_document.metadata, package_document.metadata);
        assert_eq!(file_document.markdown, package_document.markdown);
        assert_eq!(file_document.warnings, package_document.warnings);
        assert!(package_document.assets.is_empty());
        let memory = convert(package.to_str().unwrap(), package_options()).unwrap();
        let zipped = convert(file.to_str().unwrap(), package_options()).unwrap();
        assert_eq!(memory.markdown, zipped.markdown);
        assert_eq!(memory.frontmatter["source"], "预算.NUMBERS");
        assert_eq!(memory.usage.requests, 0);
        let disk = convert(
            package.to_str().unwrap(),
            ConvertOptions {
                output_dir: Some(temp.path().join("out")),
                ..package_options()
            },
        )
        .unwrap();
        assert_eq!(memory.markdown, disk.markdown);
        let output = disk.output_path.unwrap();
        assert_eq!(output.file_name().unwrap(), "预算.NUMBERS.md");
        assert!(
            std::fs::read_to_string(output)
                .unwrap()
                .ends_with(&memory.markdown)
        );
    }
}

#[test]
fn package_admission_preserves_regular_directory_and_unsupported_visual_errors() {
    use markitai_core::Error;
    let temp = tempfile::tempdir().unwrap();
    let ordinary = temp.path().join("ordinary");
    std::fs::create_dir(&ordinary).unwrap();
    assert!(!markitai_core::formats::is_numbers_package_path(&ordinary));
    assert!(matches!(
        convert(ordinary.to_str().unwrap(), package_options()),
        Err(Error::IsDirectory(_))
    ));
    let package = temp.path().join("modern.numbers");
    unpack_package(
        include_bytes!("../../src/formats/numbers/fixtures/test-1.numbers"),
        &package,
    );
    let mut options = package_options();
    options.config.as_mut().unwrap()["screenshot"]["enabled"] = json!(true);
    let error = convert(package.to_str().unwrap(), options).unwrap_err();
    assert!(matches!(error, Error::Unsupported(_)), "{error}");
    // Local OCR renders no Office pages, so it reads the tables as without it.
    let mut options = package_options();
    options.config.as_mut().unwrap()["ocr"]["enabled"] = json!(true);
    assert_eq!(
        convert(package.to_str().unwrap(), options)
            .unwrap()
            .markdown,
        convert(package.to_str().unwrap(), package_options())
            .unwrap()
            .markdown
    );
    let legacy = temp.path().join("old.numbers");
    std::fs::create_dir(&legacy).unwrap();
    std::fs::write(legacy.join("index.xml"), "<document />").unwrap();
    assert!(markitai_core::formats::is_numbers_package_path(&legacy));
    assert!(matches!(
        convert(legacy.to_str().unwrap(), package_options()),
        Err(Error::Unsupported(_))
    ));
}
