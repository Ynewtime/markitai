use super::*;
use lopdf::{Object, Stream, dictionary};

const PRESENTATION: &[u8] = include_bytes!("fixtures/hidden-blank-three.pptx");
#[cfg_attr(not(unix), allow(dead_code))]
const WORD: &[u8] = include_bytes!("fixtures/blank-middle-three.docx");

pub(super) fn pdf(count: usize) -> Vec<u8> {
    let mut pdf = lopdf::Document::with_version("1.5");
    let pages = pdf.new_object_id();
    let content = pdf.add_object(Stream::new(dictionary! {}, Vec::new()));
    let children:Vec<Object>=(0..count).map(|_|pdf.add_object(dictionary!{"Type"=>"Page","Parent"=>pages,"MediaBox"=>vec![0.into(),0.into(),720.into(),540.into()],"Resources"=>dictionary!{},"Contents"=>content}).into()).collect();
    pdf.objects.insert(
        pages,
        dictionary! {"Type"=>"Pages","Count"=>count as i64,"Kids"=>children}.into(),
    );
    let catalog = pdf.add_object(dictionary! {"Type"=>"Catalog","Pages"=>pages});
    pdf.trailer.set("Root", catalog);
    let mut bytes = Vec::new();
    pdf.save_to(&mut bytes).unwrap();
    bytes
}

#[test]
fn all_existing_office_aliases_include_workbooks() {
    for extension in ["ppt", "pps", "pot", "PPTX", "pptm", "ppsx", "ppsm", "odp"] {
        assert_eq!(kind(extension), Some(OfficeKind::Presentation));
    }
    for extension in ["doc", "docx", "docm", "odt", "rtf"] {
        assert_eq!(kind(extension), Some(OfficeKind::WordProcessing));
    }
    for extension in ["xls", "xlsx", "xlsm", "xlsb", "ods", "numbers"] {
        assert_eq!(kind(extension), Some(OfficeKind::Spreadsheet));
    }
    assert_eq!(kind("pdf"), None);
    // Templates render as the documents they make.
    for extension in ["potx", "potm", "otp"] {
        assert_eq!(kind(extension), Some(OfficeKind::Presentation));
    }
    for extension in ["dot", "dotx", "DOTM", "ott"] {
        assert_eq!(kind(extension), Some(OfficeKind::WordProcessing));
    }
    for extension in ["xlt", "xltx", "xltm", "ots"] {
        assert_eq!(kind(extension), Some(OfficeKind::Spreadsheet));
    }
    assert!(matches!(
        export_pdf(Path::new("source.numbers"), OfficeKind::Spreadsheet),
        Err(Error::Unsupported(_))
    ));
}

#[test]
fn hidden_and_blank_slides_are_counted_and_partial_pdf_export_is_rejected() {
    assert_eq!(
        crate::formats::extract_presentation_count(PRESENTATION).unwrap(),
        3
    );
    assert_eq!(validate_pdf(&pdf(3), Some(3)).unwrap(), 3);
    assert!(validate_pdf(&pdf(2), Some(3)).is_err());
    assert!(validate_pdf(&pdf(0), None).is_err());
    assert!(validate_pdf(b"%PDF-not-a-document", None).is_err());
}

#[test]
fn odp_counts_only_actual_slides_not_master_pages_or_nested_shapes() {
    use std::io::{Cursor, Write};
    let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
    archive
        .start_file("content.xml", zip::write::SimpleFileOptions::default())
        .unwrap();
    archive.write_all(br#"<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"><office:styles><draw:page/></office:styles><office:body><office:presentation><draw:page/><draw:page draw:style-name="hidden"><draw:frame><draw:page/></draw:frame></draw:page><draw:page/></office:presentation></office:body></office:document-content>"#).unwrap();
    let bytes = archive.finish().unwrap().into_inner();
    assert_eq!(slides::odp(&bytes).unwrap(), 3);
}

#[cfg(unix)]
fn mock(root: &Path, script: &str) -> PathBuf {
    use std::os::unix::fs::PermissionsExt;
    let path = root.join("mock soffice");
    fs::write(&path,format!("#!/bin/sh\nset -eu\nout=''\nprevious=''\nfor value in \"$@\"; do\n if [ \"$previous\" = '--outdir' ]; then out=\"$value\"; fi\n previous=\"$value\"\ndone\n{script}\n")).unwrap();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o700)).unwrap();
    path
}
#[cfg(unix)]
fn quote(path: &Path) -> String {
    format!("'{}'", path.to_str().unwrap().replace('\'', "'\\''"))
}

#[cfg(unix)]
#[test]
fn export_requires_new_regular_correct_output_and_does_not_modify_input() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("报告 with spaces.pptx");
    fs::write(&input, PRESENTATION).unwrap();
    let fixture = dir.path().join("golden.pdf");
    fs::write(&fixture, pdf(3)).unwrap();
    let program = mock(
        dir.path(),
        &format!("cp {} \"$out/document.pdf\"", quote(&fixture)),
    );
    let result = export_with(
        &program,
        &input,
        OfficeKind::Presentation,
        Duration::from_secs(5),
        MAX_BYTES,
    )
    .unwrap();
    assert_eq!(result.pages, 3);
    assert_eq!(result.bytes, fs::read(&fixture).unwrap());
    assert_eq!(fs::read(&input).unwrap(), PRESENTATION);
    let private = result._workspace.path().to_owned();
    assert!(private.exists());
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&private).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }
    drop(result);
    assert!(!private.exists());
    for script in [
        "exit 0".to_owned(),
        format!("ln -s {} \"$out/document.pdf\"", quote(&fixture)),
        format!("cp {} \"$out/unexpected.pdf\"", quote(&fixture)),
        "printf nope > \"$out/document.pdf\"".into(),
        "exit 9".into(),
    ] {
        let program = mock(dir.path(), &script);
        assert!(
            export_with(
                &program,
                &input,
                OfficeKind::Presentation,
                Duration::from_secs(5),
                MAX_BYTES
            )
            .is_err(),
            "{script}"
        );
    }
}

#[cfg(unix)]
#[test]
fn timeout_kills_child_group_and_output_growth_stops_export() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("source.docx");
    fs::write(&input, WORD).unwrap();
    let pidfile = dir.path().join("child.pid");
    let program = mock(
        dir.path(),
        &format!("sleep 30 &\necho $! > {}\nwait", quote(&pidfile)),
    );
    // Exercise child cleanup directly; the unrelated global permit queue may
    // otherwise consume the entire deadline before the child has started.
    let profile = dir.path().join("profile");
    let output = dir.path().join("output");
    fs::create_dir(&profile).unwrap();
    fs::create_dir(&output).unwrap();
    let start = Instant::now();
    let error = process::convert(
        &program,
        &input,
        &output,
        &profile,
        "pdf",
        // Parallel framework tests can delay process startup; allow the mock
        // to publish its child PID before exercising the actual kill deadline.
        Instant::now() + Duration::from_secs(2),
        MAX_BYTES,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("timed out"));
    assert!(start.elapsed() < Duration::from_secs(5));
    let pid = fs::read_to_string(&pidfile)
        .unwrap()
        .trim()
        .parse::<i32>()
        .unwrap();
    // kill(pid,0) can briefly see a reparented zombie. It must be gone after reaping.
    let stop = Instant::now() + Duration::from_secs(2);
    while unsafe { libc::kill(pid, 0) } == 0 && Instant::now() < stop {
        std::thread::sleep(Duration::from_millis(10));
    }
    assert_ne!(
        unsafe { libc::kill(pid, 0) },
        0,
        "Office child survived its deadline"
    );
    let program = mock(
        dir.path(),
        "dd if=/dev/zero of=\"$out/document.pdf\" bs=1024 count=16 2>/dev/null\nsleep 30",
    );
    let start = Instant::now();
    let error = export_with(
        &program,
        &input,
        OfficeKind::WordProcessing,
        Duration::from_secs(5),
        8192,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("byte limit"));
    assert!(start.elapsed() < Duration::from_secs(3));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Requires installed LibreOffice; run explicitly for real renderer acceptance"]
fn installed_libreoffice_keeps_hidden_blank_slides_and_full_frame_pixels() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("original.pptx");
    fs::write(&path, PRESENTATION).unwrap();
    let result = export_pdf(&path, OfficeKind::Presentation).unwrap();
    assert_eq!(result.pages, 3);
    let session = crate::pdf_raster::PdfRasterSession::open(&result.bytes).unwrap();
    assert_eq!(session.pages(), 3);
    for (page, expected) in [
        (1, [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]]),
        (2, [[0, 255, 255], [255, 0, 255], [0, 255, 0], [255, 0, 0]]),
    ] {
        let pixels = session.render(page, 72.).unwrap();
        assert_eq!(pixels.dimensions(), (720, 540));
        for ((x, y), color) in [(36, 36), (684, 36), (36, 504), (684, 504)]
            .into_iter()
            .zip(expected)
        {
            for (actual, wanted) in pixels.get_pixel(x, y).0.into_iter().zip(color) {
                assert!(
                    (i16::from(actual) - wanted as i16).abs() < 8,
                    "page {page} corner {x},{y}"
                );
            }
        }
    }
    let blank = session.render(3, 72.).unwrap();
    assert!(blank.pixels().all(|p| p.0 == [255, 255, 255]));
    let pdf = lopdf::Document::load_mem(&result.bytes).unwrap();
    assert!(pdf.extract_text(&[1]).unwrap().contains("VISIBLE FIRST"));
    assert!(pdf.extract_text(&[2]).unwrap().contains("HIDDEN SECOND"));
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Requires installed LibreOffice; run explicitly for real renderer acceptance"]
fn installed_libreoffice_preserves_word_explicit_blank_middle_page() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("original.docx");
    fs::write(&path, WORD).unwrap();
    let result = export_pdf(&path, OfficeKind::WordProcessing).unwrap();
    assert_eq!(result.pages, 3);
    let pdf = lopdf::Document::load_mem(&result.bytes).unwrap();
    assert!(pdf.extract_text(&[1]).unwrap().contains("WORD FIRST PAGE"));
    assert!(pdf.extract_text(&[3]).unwrap().contains("WORD THIRD PAGE"));
    assert!(pdf.extract_text(&[2]).unwrap().trim().is_empty());
    let session = crate::pdf_raster::PdfRasterSession::open(&result.bytes).unwrap();
    let blank = session.render(2, 72.).unwrap();
    assert_eq!(blank.dimensions(), (612, 792));
    assert!(blank.pixels().all(|p| p.0 == [255, 255, 255]));
}

#[cfg(unix)]
#[test]
fn diagnostic_checks_startup_without_export_and_bounds_hung_programs() {
    let dir = tempfile::tempdir().unwrap();
    let profile = dir.path().join("private-profile");
    fs::create_dir(&profile).unwrap();
    let success = mock(
        dir.path(),
        "[ \"$out\" = '' ]\n[ \"$PWD\" -ef \"$TMPDIR\" ]\nlast=''\nfor arg in \"$@\"; do last=\"$arg\"; done\n[ \"$last\" = '--version' ]",
    );
    process::diagnose(&success, &profile, Instant::now() + Duration::from_secs(3)).unwrap();
    let rejected = mock(dir.path(), "printf 'private test stderr' >&2\nexit 9");
    let error = process::diagnose(&rejected, &profile, Instant::now() + Duration::from_secs(3))
        .unwrap_err();
    assert!(!error.to_string().contains("private test stderr"));
    let hung = mock(dir.path(), "sleep 30");
    let start = Instant::now();
    assert!(process::diagnose(&hung, &profile, start + Duration::from_millis(150)).is_err());
    assert!(start.elapsed() < Duration::from_secs(2));
}

#[cfg(unix)]
#[test]
fn workbook_export_selects_whole_sheet_mode_and_rejects_lost_sheets() {
    let directory = tempfile::tempdir().unwrap();
    let input = directory.path().join("source.xlsx");
    fs::write(&input, include_bytes!("fixtures/whole-workbook.xlsx")).unwrap();
    let fixture = directory.path().join("pages.pdf");
    fs::write(&fixture, pdf(4)).unwrap();
    let program = mock(
        directory.path(),
        &format!(
            "found=no\nfor value in \"$@\"; do\n case \"$value\" in *calc_pdf_Export*SinglePageSheets*true*) found=yes;; esac\ndone\n[ \"$found\" = yes ]\ncp {} \"$out/document.pdf\"",
            quote(&fixture)
        ),
    );
    let result = export_with(
        &program,
        &input,
        OfficeKind::Spreadsheet,
        Duration::from_secs(5),
        MAX_BYTES,
    )
    .unwrap();
    assert_eq!(result.pages, 4);
    assert!(
        result
            .warnings
            .iter()
            .any(|warning| warning.contains("print areas"))
    );
    fs::write(&fixture, pdf(3)).unwrap();
    let error = export_with(
        &program,
        &input,
        OfficeKind::Spreadsheet,
        Duration::from_secs(5),
        MAX_BYTES,
    )
    .err()
    .unwrap();
    assert!(error.to_string().contains("every workbook sheet"));
}

#[cfg(unix)]
#[test]
fn templates_export_as_the_documents_they_make_from_a_copy_under_their_own_name() {
    let directory = tempfile::tempdir().unwrap();
    for (name, bytes, kind, pages) in [
        ("deck.potx", PRESENTATION, OfficeKind::Presentation, 3),
        (
            "book.xltx",
            include_bytes!("fixtures/whole-workbook.xlsx").as_slice(),
            OfficeKind::Spreadsheet,
            4,
        ),
    ] {
        let input = directory.path().join(name);
        fs::write(&input, bytes).unwrap();
        let fixture = directory.path().join("pages.pdf");
        fs::write(&fixture, pdf(pages)).unwrap();
        let extension = name.rsplit('.').next().unwrap();
        // The private copy keeps the template's extension; only one export
        // runs, so the page count came from the package itself.
        let program = mock(
            directory.path(),
            &format!(
                "for value in \"$@\"; do last=\"$value\"; done\ncase \"$last\" in */document.{extension}) ;; *) exit 7;; esac\n[ ! -e \"$out/../ran\" ]\ntouch \"$out/../ran\"\ncp {} \"$out/document.pdf\"",
                quote(&fixture)
            ),
        );
        let result =
            export_with(&program, &input, kind, Duration::from_secs(5), MAX_BYTES).unwrap();
        assert_eq!(result.pages, pages, "{name}");
    }
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Requires installed LibreOffice and native macOS renderer; run explicitly"]
fn installed_libreoffice_whole_sheets_keep_wide_hidden_empty_and_last_content() {
    for (extension, bytes) in [
        (
            "xlsx",
            include_bytes!("fixtures/whole-workbook.xlsx").as_slice(),
        ),
        (
            "ods",
            include_bytes!("fixtures/whole-workbook.ods").as_slice(),
        ),
        (
            "xls",
            include_bytes!("fixtures/whole-workbook.xls").as_slice(),
        ),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let source = directory
            .path()
            .join(format!("工作簿 original.{extension}"));
        fs::write(&source, bytes).unwrap();
        let result = export_pdf(&source, OfficeKind::Spreadsheet).unwrap();
        assert_eq!(result.pages, 4, "{extension}");
        let document = lopdf::Document::load_mem(&result.bytes).unwrap();
        assert!(
            document
                .extract_text(&[1])
                .unwrap()
                .contains("VISIBLE FIRST")
        );
        assert!(
            document
                .extract_text(&[1])
                .unwrap()
                .contains("OUTSIDE PRINT AREA")
        );
        assert!(
            document
                .extract_text(&[2])
                .unwrap()
                .contains("HIDDEN MIDDLE")
        );
        assert!(document.extract_text(&[3]).unwrap().trim().is_empty());
        assert!(document.extract_text(&[4]).unwrap().contains("LAST SHEET"));
        let session = crate::pdf_raster::PdfRasterSession::open(&result.bytes).unwrap();
        let first = session.render(1, 72.).unwrap();
        assert!(
            first.width() > 700 && first.height() > 700,
            "full sheet was reduced to paper size"
        );
        for color in [[255, 0, 0], [0, 0, 255]] {
            assert!(
                first
                    .pixels()
                    .filter(|pixel| pixel
                        .0
                        .into_iter()
                        .zip(color)
                        .all(|(actual, expected)| (i16::from(actual) - expected as i16).abs() < 8))
                    .count()
                    > 100,
                "{extension}: missing cell background {color:?}"
            );
        }
        for (page, color) in [(2, [0, 255, 0]), (4, [255, 204, 0])] {
            let image = session.render(page, 72.).unwrap();
            assert!(
                image
                    .pixels()
                    .filter(|pixel| pixel
                        .0
                        .into_iter()
                        .zip(color)
                        .all(|(actual, expected)| (i16::from(actual) - expected as i16).abs() < 8))
                    .count()
                    > 100,
                "{extension}: page {page}"
            );
        }
        // Calc keeps the genuinely empty sheet as a very thin page; preserve
        // that geometry instead of inventing an A4/Letter canvas or dropping it.
        let blank = session.render(3, 72.).unwrap();
        assert!(blank.pixels().all(|pixel| pixel.0 == [255, 255, 255]));
        assert_eq!(fs::read(source).unwrap(), bytes);
    }
}

#[test]
fn windows_discovery_includes_machine_user_and_scoop_installs_without_relative_roots() {
    let root = tempfile::tempdir().unwrap();
    let env: std::collections::HashMap<&str, std::ffi::OsString> = [
        (
            "ProgramFiles",
            root.path().join("Program Files").into_os_string(),
        ),
        ("ProgramFiles(x86)", "relative-programs".into()),
        (
            "LOCALAPPDATA",
            root.path().join("用户 AppData").into_os_string(),
        ),
        ("USERPROFILE", root.path().join("用户").into_os_string()),
        (
            "SCOOP_GLOBAL",
            root.path().join("global scoop").into_os_string(),
        ),
    ]
    .into_iter()
    .collect();
    let candidates = windows_locations(|key| env.get(key).cloned());
    assert_eq!(candidates.len(), 12);
    let (pairs, remainder) = candidates.as_chunks::<2>();
    assert!(remainder.is_empty());
    for pair in pairs {
        assert_eq!(pair[0].file_name().unwrap(), "soffice.com");
        assert_eq!(pair[1].file_name().unwrap(), "soffice.exe");
        assert_eq!(pair[0].parent(), pair[1].parent());
    }
    assert!(
        candidates.contains(
            &root
                .path()
                .join("Program Files/LibreOffice/program/soffice.exe")
        )
    );
    assert!(
        candidates.contains(
            &root
                .path()
                .join("用户 AppData/Programs/LibreOffice/program/soffice.exe")
        )
    );
    assert!(
        candidates.contains(
            &root
                .path()
                .join("用户/scoop/apps/libreoffice/current/program/soffice.exe")
        )
    );
    assert!(
        candidates.contains(
            &root
                .path()
                .join("global scoop/apps/libreoffice/current/program/soffice.exe")
        )
    );
    assert!(candidates.iter().all(|path| path.is_absolute()));
    let explicit = windows_locations(|key| match key {
        "SCOOP" => Some(root.path().join("custom").into_os_string()),
        _ => None,
    });
    assert_eq!(explicit.len(), 4);
    assert!(explicit[0].starts_with(root.path().join("custom")));
}

#[cfg(windows)]
#[test]
fn windows_path_discovery_prefers_console_entry_and_keeps_exe_fallback() {
    let root = tempfile::tempdir().unwrap();
    let console = root.path().join("soffice.com");
    let gui = root.path().join("soffice.exe");
    fs::write(&console, b"fixture console entry").unwrap();
    fs::write(&gui, b"fixture GUI entry").unwrap();
    let path = std::env::join_paths([root.path()]).unwrap();
    let select = || {
        crate::process_groups::find_program(
            WINDOWS_PROGRAM_NAMES,
            Some(&path),
            Some(std::ffi::OsStr::new(".EXE;.COM")),
        )
    };
    assert_eq!(select(), Some(console.clone()));
    fs::remove_file(console).unwrap();
    assert_eq!(select(), Some(gui.clone()));
    fs::remove_file(gui).unwrap();
    assert_eq!(select(), None);
}

#[cfg(windows)]
#[test]
fn windows_fixed_discovery_prefers_console_entry_and_keeps_exe_fallback() {
    let root = tempfile::tempdir().unwrap();
    let programs = root.path().join("LibreOffice/program");
    fs::create_dir_all(&programs).unwrap();
    let console = programs.join("soffice.com");
    let gui = programs.join("soffice.exe");
    fs::write(&console, b"fixture console entry").unwrap();
    fs::write(&gui, b"fixture GUI entry").unwrap();
    let select = || {
        windows_locations(|name| {
            (name == "ProgramFiles").then(|| root.path().as_os_str().to_os_string())
        })
        .into_iter()
        .find(|candidate| executable(candidate))
    };
    assert_eq!(select(), Some(console.clone()));
    fs::remove_file(console).unwrap();
    assert_eq!(select(), Some(gui.clone()));
    fs::remove_file(gui).unwrap();
    assert_eq!(select(), None);
}
