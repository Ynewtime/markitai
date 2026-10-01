//! The native stdout image store: only a conversion without an output
//! directory that asks for it saves images there; library calls never do.
use super::*;
use markitai_core::{ConversionOutput, ConvertContext, convert_with_context};
use std::path::{Path, PathBuf};

fn png(color: [u8; 3], width: u32, height: u32) -> Vec<u8> {
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(width, height, image::Rgb(color))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    bytes.into_inner()
}

/// A Word document using one picture twice and another once.
fn docx(path: &Path) {
    let picture = |id: u32, rel: &str, descr: &str| {
        format!(
            r#"<w:p><w:r><w:drawing><wp:inline><wp:extent cx="1143000" cy="762000"/><wp:docPr id="{id}" name="P{id}" descr="{descr}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="{id}" name="P{id}" descr="{descr}"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rel}"/></pic:blipFill><pic:spPr/></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"#
        )
    };
    let document = format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?><w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:body><w:p><w:r><w:t>Figures</w:t></w:r></w:p>{}{}{}</w:body></w:document>"#,
        picture(1, "rRed", "Red chart"),
        picture(2, "rBlue", "Blue chart"),
        picture(3, "rRed", "Red again"),
    );
    let members = [
        ("[Content_Types].xml", r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#.as_bytes().to_vec()),
        ("_rels/.rels", r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#.as_bytes().to_vec()),
        ("word/_rels/document.xml.rels", r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rRed" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/><Relationship Id="rBlue" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image2.png"/></Relationships>"#.as_bytes().to_vec()),
        ("word/document.xml", document.into_bytes()),
        ("word/media/image1.png", png([200, 30, 30], 120, 80)),
        ("word/media/image2.png", png([30, 30, 200], 90, 90)),
    ];
    let mut archive = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
    for (name, bytes) in members {
        archive
            .start_file(name, zip::write::SimpleFileOptions::default())
            .unwrap();
        archive.write_all(&bytes).unwrap();
    }
    archive.finish().unwrap();
}

/// A one-page PDF with text and an uncompressed 80x80 RGB picture.
fn pdf(path: &Path) {
    use lopdf::{Object, Stream, dictionary};
    let mut document = lopdf::Document::with_version("1.7");
    let pages = document.new_object_id();
    let font = document.add_object(dictionary! {
        "Type" => "Font", "Subtype" => "Type1", "BaseFont" => "Helvetica"
    });
    let pixels: Vec<u8> = (0..80 * 80)
        .flat_map(|i| [(i % 80 * 3) as u8, 90, 160])
        .collect();
    let image = document.add_object(Stream::new(
        dictionary! {
            "Type" => "XObject", "Subtype" => "Image", "Width" => 80, "Height" => 80,
            "ColorSpace" => "DeviceRGB", "BitsPerComponent" => 8
        },
        pixels,
    ));
    let content = document.add_object(Stream::new(
        dictionary! {},
        b"BT /F1 12 Tf 72 720 Td (Report with an image) Tj ET q 128 0 0 128 72 500 cm /Im Do Q"
            .to_vec(),
    ));
    let page = document.add_object(dictionary! {
        "Type" => "Page", "Parent" => pages, "Contents" => content,
        "Resources" => dictionary! {
            "Font" => dictionary! { "F1" => font },
            "XObject" => dictionary! { "Im" => image }
        },
        "MediaBox" => vec![0.into(), 0.into(), 612.into(), Object::Integer(792)]
    });
    document.objects.insert(
        pages,
        dictionary! { "Type" => "Pages", "Kids" => vec![Object::Reference(page)], "Count" => 1 }
            .into(),
    );
    let catalog = document.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages });
    document.trailer.set("Root", catalog);
    document.save(path).unwrap();
}

fn inline(path: &Path) {
    use base64::Engine;
    let data = base64::engine::general_purpose::STANDARD.encode(png([30, 160, 30], 100, 60));
    std::fs::write(
        path,
        format!("# Inline\n\n![green](data:image/png;base64,{data})\n"),
    )
    .unwrap();
}

fn stdout(source: &Path, store: &Path) -> ConversionOutput {
    convert_with_context(
        source.to_str().unwrap(),
        options(),
        ConvertContext {
            stdout_assets: Some(store),
            ..Default::default()
        },
    )
    .unwrap()
}

/// Every Markdown image destination in order.
fn destinations(markdown: &str) -> Vec<String> {
    markdown
        .match_indices("](")
        .map(|(index, _)| {
            let rest = &markdown[index + 2..];
            rest[..rest.find(')').unwrap()].to_owned()
        })
        .collect()
}

fn stored(store: &Path) -> Vec<PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(store.join("blobs"))
        .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
        .unwrap_or_default();
    files.sort();
    files
}

#[cfg(unix)]
#[test]
fn stdout_store_links_docx_pdf_and_inline_images_and_reuses_their_files() {
    let dir = tempfile::tempdir().unwrap();
    let store = dir.path().join("home/assets");
    let blobs = store.join("blobs");
    for (name, build, images) in [
        ("figures.docx", docx as fn(&Path), 3),
        ("report.pdf", pdf, 1),
        ("inline.md", inline, 1),
    ] {
        let source = dir.path().join(name);
        build(&source);
        let first = stdout(&source, &store);
        let links = destinations(&first.markdown);
        assert_eq!(links.len(), images, "{name}: {}", first.markdown);
        for link in &links {
            let path = Path::new(link.strip_prefix("file://").unwrap());
            assert_eq!(
                std::fs::canonicalize(path.parent().unwrap()).unwrap(),
                std::fs::canonicalize(&blobs).unwrap(),
                "{name}: {link}"
            );
            assert_eq!(
                std::fs::canonicalize(path).unwrap(),
                path,
                "{name}: canonical"
            );
            let bytes = std::fs::read(path).unwrap();
            image::load_from_memory(&bytes).unwrap();
        }
        assert!(!first.markdown.contains(".markitai/"), "{}", first.markdown);
        assert!(!first.markdown.contains("data:image"), "{}", first.markdown);
        assert!(first.output_path.is_none());
        assert!(
            first
                .warnings
                .iter()
                .all(|w| !w.contains("could not be saved"))
        );
        let files = stored(&store);
        let stamps: Vec<_> = files
            .iter()
            .map(|file| std::fs::metadata(file).unwrap().modified().unwrap())
            .collect();
        let again = stdout(&source, &store);
        assert_eq!(destinations(&again.markdown), links, "{name}");
        assert_eq!(stored(&store), files, "{name}: no new files");
        let restamped: Vec<_> = files
            .iter()
            .map(|file| std::fs::metadata(file).unwrap().modified().unwrap())
            .collect();
        assert_eq!(
            restamped, stamps,
            "{name}: existing files are not rewritten"
        );
    }
    // The picture used twice in the Word document is one file.
    let docx = stdout(&dir.path().join("figures.docx"), &store);
    let links = destinations(&docx.markdown);
    assert_eq!(links[0], links[2]);
    assert_ne!(links[0], links[1]);
    // Nothing besides the store was written next to the sources.
    let mut created: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    created.sort();
    assert_eq!(created, ["figures.docx", "home", "inline.md", "report.pdf"]);
}

#[test]
fn library_and_output_directory_conversions_never_use_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("figures.docx");
    docx(&source);
    let store = dir.path().join("store");
    let memory = convert(source.to_str().unwrap(), options()).unwrap();
    assert!(
        memory.markdown.contains("](.markitai/assets/"),
        "{}",
        memory.markdown
    );
    let inline_source = dir.path().join("inline.md");
    inline(&inline_source);
    let memory_inline = convert(inline_source.to_str().unwrap(), options()).unwrap();
    assert!(memory_inline.markdown.contains("](data:image/png;base64,"));
    let out = dir.path().join("out");
    let published = convert_with_context(
        source.to_str().unwrap(),
        ConvertOptions {
            output_dir: Some(out.clone()),
            ..options()
        },
        ConvertContext {
            stdout_assets: Some(&store),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(published.markdown.contains("](.markitai/assets/"));
    assert!(!published.markdown.contains("file://"));
    assert!(out.join(".markitai/assets").is_dir());
    assert!(!store.exists());
}
