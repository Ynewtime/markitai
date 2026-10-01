//! A document printed to stdout links its images to files saved under
//! `MARKITAI_HOME/assets` (`image.stdout_persist`), never under the real
//! home or next to the input; turning that off keeps relative references.
use serde_json::json;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

struct Sandbox {
    _dir: tempfile::TempDir,
    root: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_owned();
        for name in ["work", "home", "user"] {
            std::fs::create_dir(root.join(name)).unwrap();
        }
        Self { _dir: dir, root }
    }
    fn work(&self) -> PathBuf {
        self.root.join("work")
    }
    fn blobs(&self) -> PathBuf {
        self.root.join("home/assets/blobs")
    }
    fn run(&self, args: &[&str]) -> Output {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for name in ["PATH", "SYSTEMROOT", "TMPDIR"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        // A private user home: the default store must follow MARKITAI_HOME.
        command
            .current_dir(self.work())
            .env("HOME", self.root.join("user"))
            .env("USERPROFILE", self.root.join("user"))
            .env("MARKITAI_HOME", self.root.join("home"))
            .args(args)
            .output()
            .expect("CLI starts")
    }
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= u32::from(byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// An RGB PNG with stored (uncompressed) deflate blocks.
fn png(width: u32, height: u32, seed: u8) -> Vec<u8> {
    let mut raw = Vec::new();
    for y in 0..height {
        raw.push(0);
        for x in 0..width {
            raw.extend_from_slice(&[seed, (x * 3) as u8, (y * 3) as u8]);
        }
    }
    let mut zlib = vec![0x78, 0x01];
    let blocks: Vec<_> = raw.chunks(65_535).collect();
    for (index, block) in blocks.iter().enumerate() {
        zlib.push(u8::from(index + 1 == blocks.len()));
        let length = block.len() as u16;
        zlib.extend_from_slice(&length.to_le_bytes());
        zlib.extend_from_slice(&(!length).to_le_bytes());
        zlib.extend_from_slice(block);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &byte in &raw {
        a = (a + u32::from(byte)) % 65_521;
        b = (b + a) % 65_521;
    }
    zlib.extend_from_slice(&((b << 16) | a).to_be_bytes());
    let mut header = width.to_be_bytes().to_vec();
    header.extend_from_slice(&height.to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    for (kind, data) in [
        (&b"IHDR"[..], header),
        (b"IDAT", zlib),
        (b"IEND", Vec::new()),
    ] {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut chunk = kind.to_vec();
        chunk.extend_from_slice(&data);
        out.extend_from_slice(&chunk);
        out.extend_from_slice(&crc32(&chunk).to_be_bytes());
    }
    out
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
        picture(1, "rFirst", "First chart"),
        picture(2, "rSecond", "Second chart"),
        picture(3, "rFirst", "First chart again"),
    );
    let members = [
        ("[Content_Types].xml", r#"<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Default Extension="png" ContentType="image/png"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#.as_bytes().to_vec()),
        ("_rels/.rels", r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>"#.as_bytes().to_vec()),
        ("word/_rels/document.xml.rels", r#"<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rFirst" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image1.png"/><Relationship Id="rSecond" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/image2.png"/></Relationships>"#.as_bytes().to_vec()),
        ("word/document.xml", document.into_bytes()),
        ("word/media/image1.png", png(120, 80, 200)),
        ("word/media/image2.png", png(90, 90, 20)),
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
    let stream = |dictionary: &str, data: &[u8]| {
        let mut object =
            format!("<< {dictionary} /Length {} >>\nstream\n", data.len()).into_bytes();
        object.extend_from_slice(data);
        object.extend_from_slice(b"\nendstream");
        object
    };
    let pixels: Vec<u8> = (0..80 * 80)
        .flat_map(|index| [(index % 80 * 3) as u8, 90, 160])
        .collect();
    let objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_vec(),
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> /XObject << /Im 5 0 R >> >> /Contents 6 0 R >>".to_vec(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
        stream(
            "/Type /XObject /Subtype /Image /Width 80 /Height 80 /ColorSpace /DeviceRGB /BitsPerComponent 8",
            &pixels,
        ),
        stream(
            "",
            b"BT /F1 12 Tf 72 720 Td (Report with an image) Tj ET q 160 0 0 160 72 500 cm /Im Do Q",
        ),
    ];
    let mut out = b"%PDF-1.7\n".to_vec();
    let mut offsets = Vec::new();
    for (index, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n", index + 1).as_bytes());
        out.extend_from_slice(body);
        out.extend_from_slice(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend_from_slice(
        format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).as_bytes(),
    );
    for offset in offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );
    std::fs::write(path, out).unwrap();
}

/// An HTML page with a picture beside it; image analysis localizes it.
fn html(path: &Path) {
    std::fs::write(path.with_file_name("chart.png"), png(100, 70, 120)).unwrap();
    std::fs::write(
        path,
        "<!doctype html><html><head><title>Gallery</title></head><body><h1>Gallery</h1>\
         <p>A chart:</p><p><img src=\"chart.png\" alt=\"Chart\"></p></body></html>",
    )
    .unwrap();
}

/// A model endpoint nothing listens on: enhancement fails fast and the base
/// Markdown, with its localized images, is kept.
fn unreachable_model() -> String {
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    json!({"llm":{"on_failure":"fallback","router_settings":{"num_retries":0,"timeout":2},
        "model_list":[{"model_name":"local","litellm_params":{"model":"openai/mock",
        "api_base":format!("http://127.0.0.1:{port}/v1"),"api_key":"isolated"},
        "model_info":{"supports_vision":true}}]},"cache":{"enabled":false}})
    .to_string()
}

/// Image destinations of Markdown images, in order.
fn images(markdown: &str) -> Vec<String> {
    markdown
        .match_indices("![")
        .filter_map(|(index, _)| {
            let rest = &markdown[index..];
            let start = rest.find("](")? + 2;
            let end = start + rest[start..].find(')')?;
            Some(rest[start..end].to_owned())
        })
        .collect()
}

fn decode(uri: &str) -> PathBuf {
    let rest = uri.strip_prefix("file://").expect("a file URI");
    let rest = if cfg!(windows) {
        rest.trim_start_matches('/')
    } else {
        rest
    };
    let bytes = rest.as_bytes();
    let mut decoded = Vec::new();
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = std::str::from_utf8(&bytes[index + 1..index + 3]).unwrap();
            decoded.push(u8::from_str_radix(hex, 16).unwrap());
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    PathBuf::from(String::from_utf8(decoded).unwrap())
}

fn stored(blobs: &Path) -> Vec<(PathBuf, std::time::SystemTime)> {
    let mut files: Vec<_> = std::fs::read_dir(blobs)
        .map(|entries| {
            entries
                .map(|entry| {
                    let path = entry.unwrap().path();
                    let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
                    (path, modified)
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

#[test]
fn docx_pdf_html_and_inline_images_on_stdout_open_from_the_isolated_home() {
    let sandbox = Sandbox::new();
    let work = sandbox.work();
    docx(&work.join("figures.docx"));
    pdf(&work.join("report.pdf"));
    html(&work.join("gallery.html"));
    let inline = format!(
        "# Inline\n\n![green](data:image/png;base64,{})\n",
        base64(&png(100, 60, 60))
    );
    std::fs::write(work.join("inline.md"), inline).unwrap();
    let model = unreachable_model();
    let blobs = std::fs::canonicalize(sandbox.root.join("home"))
        .unwrap()
        .join("assets/blobs");
    for (input, extra, count) in [
        ("figures.docx", vec![], 3),
        ("report.pdf", vec![], 1),
        (
            "gallery.html",
            vec!["--llm", "--alt", "--config-json", &model],
            1,
        ),
        ("inline.md", vec![], 1),
    ] {
        let mut args = vec![input];
        args.extend(extra);
        let output = sandbox.run(&args);
        let (stdout, stderr) = (text(&output.stdout), text(&output.stderr));
        assert!(output.status.success(), "{input}: {stderr}");
        let links = images(&stdout);
        assert_eq!(links.len(), count, "{input}: {stdout}");
        for link in &links {
            let path = decode(link);
            let path = std::fs::canonicalize(&path).unwrap_or(path);
            assert_eq!(path.parent(), Some(blobs.as_path()), "{input}: {link}");
            let bytes = std::fs::read(&path).unwrap();
            assert!(
                bytes.starts_with(b"\xff\xd8\xff") || bytes.starts_with(b"\x89PNG"),
                "{input}: {link} is not an image"
            );
        }
        assert!(!stdout.contains(".markitai/"), "{input}: {stdout}");
        assert!(!stdout.contains("data:image"), "{input}: {stdout}");
        assert!(!stderr.contains("stdout mode does not write"), "{stderr}");
        assert!(!stderr.contains("could not be saved"), "{stderr}");
    }
    // The run wrote nowhere else: not beside the inputs, not in the user home.
    assert!(!work.join(".markitai").exists());
    assert_eq!(
        std::fs::read_dir(sandbox.root.join("user"))
            .unwrap()
            .count(),
        0
    );
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let value = chunk
            .iter()
            .enumerate()
            .fold(0u32, |value, (index, &byte)| {
                value | u32::from(byte) << (16 - 8 * index)
            });
        for index in 0..4 {
            if index <= chunk.len() {
                out.push(ALPHABET[(value >> (18 - 6 * index) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

#[test]
fn repeated_runs_reuse_one_file_per_distinct_image() {
    let sandbox = Sandbox::new();
    docx(&sandbox.work().join("figures.docx"));
    let first = sandbox.run(&["figures.docx", "--pure"]);
    assert!(first.status.success(), "{}", text(&first.stderr));
    let links = images(&text(&first.stdout));
    assert_eq!(links.len(), 3);
    // The picture used twice is one file; the other picture another.
    assert_eq!(links[0], links[2]);
    assert_ne!(links[0], links[1]);
    let files = stored(&sandbox.blobs());
    assert_eq!(files.len(), 2);
    let second = sandbox.run(&["figures.docx", "--pure"]);
    assert!(second.status.success());
    assert_eq!(text(&second.stdout), text(&first.stdout));
    // Same names, and the existing files were not written again.
    assert_eq!(stored(&sandbox.blobs()), files);
}

#[test]
fn turning_persistence_off_keeps_relative_references_and_says_so() {
    let sandbox = Sandbox::new();
    docx(&sandbox.work().join("figures.docx"));
    let off = r#"{"image":{"stdout_persist":false}}"#;
    let output = sandbox.run(&["figures.docx", "--config-json", off]);
    assert!(output.status.success());
    let links = images(&text(&output.stdout));
    assert_eq!(links.len(), 3);
    assert!(
        links
            .iter()
            .all(|link| link.starts_with(".markitai/assets/"))
    );
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains("Warning: 2 image references point to files that stdout mode does not write because image.stdout_persist is false"),
        "{stderr}"
    );
    assert!(!sandbox.root.join("home/assets").exists());
    let quiet = sandbox.run(&["figures.docx", "--config-json", off, "-q"]);
    assert!(quiet.status.success());
    assert_eq!(text(&quiet.stderr), "");
}

#[test]
fn a_configured_store_is_used_and_output_directories_never_use_one() {
    let sandbox = Sandbox::new();
    docx(&sandbox.work().join("figures.docx"));
    let custom = sandbox.root.join("custom images");
    let config = json!({"image":{"stdout_persist_dir":custom}}).to_string();
    let output = sandbox.run(&["figures.docx", "--config-json", &config]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let links = images(&text(&output.stdout));
    // The space in the directory name is escaped in the link.
    assert!(links[0].contains("custom%20images/blobs/"), "{links:?}");
    assert_eq!(stored(&custom.join("blobs")).len(), 2);
    assert!(decode(&links[0]).is_file());
    assert!(!sandbox.root.join("home/assets").exists());

    let written = sandbox.run(&["figures.docx", "-o", "out"]);
    assert!(written.status.success(), "{}", text(&written.stderr));
    assert!(!sandbox.root.join("home/assets").exists());
    let document = std::fs::read_to_string(sandbox.work().join("out/figures.docx.md")).unwrap();
    assert!(document.contains("](.markitai/assets/"), "{document}");
    assert!(!document.contains("file://"));
}

#[cfg(unix)]
#[test]
fn a_symlinked_store_is_not_written_through() {
    let sandbox = Sandbox::new();
    docx(&sandbox.work().join("figures.docx"));
    let target = sandbox.root.join("elsewhere");
    std::fs::create_dir(&target).unwrap();
    std::os::unix::fs::symlink(&target, sandbox.root.join("home/assets")).unwrap();
    let output = sandbox.run(&["figures.docx"]);
    assert!(output.status.success());
    let links = images(&text(&output.stdout));
    assert!(
        links
            .iter()
            .all(|link| link.starts_with(".markitai/assets/"))
    );
    let stderr = text(&output.stderr);
    assert!(
        stderr.contains("Warning: 2 images could not be saved to "),
        "{stderr}"
    );
    assert!(stderr.contains("Symlink access is disabled"), "{stderr}");
    // Persistence is on; only the failure is reported.
    assert!(
        !stderr.contains("image.stdout_persist is false"),
        "{stderr}"
    );
    assert_eq!(std::fs::read_dir(&target).unwrap().count(), 0);
}

#[test]
fn profiles_keep_stored_images_as_file_links() {
    let sandbox = Sandbox::new();
    docx(&sandbox.work().join("figures.docx"));
    for args in [
        vec!["--profile", "rag"],
        vec!["--profile", "obsidian"],
        vec![
            "--profile",
            "obsidian",
            "--config-json",
            r#"{"output":{"wikilinks":true}}"#,
        ],
    ] {
        let mut command = vec!["figures.docx"];
        command.extend(args.iter().copied());
        let output = sandbox.run(&command);
        assert!(output.status.success(), "{}", text(&output.stderr));
        let stdout = text(&output.stdout);
        let links = images(&stdout);
        assert_eq!(links.len(), 3, "{args:?}: {stdout}");
        assert!(
            links.iter().all(|link| decode(link).is_file()),
            "{args:?}: {stdout}"
        );
        // Neither the visible `assets/` relocation nor a wikilink, which
        // cannot carry a file URI, applies to a stored image.
        assert!(!stdout.contains("](assets/"), "{args:?}: {stdout}");
        assert!(!stdout.contains("![["), "{args:?}: {stdout}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn pdf_page_screenshots_on_stdout_link_to_stored_captures() {
    let sandbox = Sandbox::new();
    pdf(&sandbox.work().join("report.pdf"));
    let output = sandbox.run(&["report.pdf", "--screenshot"]);
    assert!(output.status.success(), "{}", text(&output.stderr));
    let stdout = text(&output.stdout);
    let comment = stdout
        .lines()
        .find(|line| line.starts_with("<!-- ![Page 1](file://"))
        .unwrap_or_else(|| panic!("no linked page reference: {stdout}"));
    let uri = &comment["<!-- ![Page 1](".len()..comment.len() - ") -->".len()];
    let capture = std::fs::read(decode(uri)).unwrap();
    assert!(capture.starts_with(b"\xff\xd8\xff") || capture.starts_with(b"\x89PNG"));
    assert!(!stdout.contains(".markitai/"), "{stdout}");
    // The extracted picture and the page capture are two stored files.
    assert_eq!(stored(&sandbox.blobs()).len(), 2);
}
