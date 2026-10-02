//! The everyday command-line experience: `~` in output locations, what long
//! batches show on a terminal, how warnings read, what resuming says, how bad
//! inputs are described, previews and hidden files.
use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// A valid 1x1 RGBA PNG (every chunk checksum correct).
const TINY_PNG: &[u8] = &[
    137, 80, 78, 71, 13, 10, 26, 10, 0, 0, 0, 13, 73, 72, 68, 82, 0, 0, 0, 1, 0, 0, 0, 1, 8, 6, 0,
    0, 0, 31, 21, 196, 137, 0, 0, 0, 11, 73, 68, 65, 84, 120, 156, 99, 96, 0, 2, 0, 0, 5, 0, 1,
    122, 94, 171, 63, 0, 0, 0, 0, 73, 69, 78, 68, 174, 66, 96, 130,
];

/// A real Word document with three pages of text.
const DOCX: &[u8] =
    include_bytes!("../../markitai-core/src/office_render/fixtures/blank-middle-three.docx");

struct Setup {
    root: tempfile::TempDir,
}

impl Setup {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("home-dir")).unwrap();
        Self { root }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.root.path().join(name)
    }

    fn write(&self, name: &str, bytes: impl AsRef<[u8]>) {
        let path = self.path(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, bytes).unwrap();
    }

    fn command(&self, extra: &[(&str, &str)]) -> Command {
        let mut command = Command::new(env!("CARGO_BIN_EXE_markitai"));
        command.env_clear();
        for name in ["PATH", "SYSTEMROOT", "TMPDIR"] {
            if let Some(value) = std::env::var_os(name) {
                command.env(name, value);
            }
        }
        command
            .current_dir(self.root.path())
            .env("HOME", self.path("home-dir"))
            .env("MARKITAI_HOME", self.path("mh"))
            .envs(extra.iter().copied());
        command
    }

    fn run(&self, args: &[&str]) -> Output {
        self.command(&[]).args(args).output().expect("CLI starts")
    }

    fn run_env(&self, args: &[&str], extra: &[(&str, &str)]) -> Output {
        self.command(extra).args(args).output().expect("CLI starts")
    }
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

/// A PDF with one page per entry; an empty string is a blank page.
fn pdf(pages: &[&str]) -> Vec<u8> {
    let count = pages.len();
    let kids: Vec<_> = (0..count).map(|i| format!("{} 0 R", 4 + 2 * i)).collect();
    let mut objects: Vec<Vec<u8>> = vec![
        b"<< /Type /Catalog /Pages 2 0 R >>".to_vec(),
        format!(
            "<< /Type /Pages /Kids [{}] /Count {count} >>",
            kids.join(" ")
        )
        .into_bytes(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_vec(),
    ];
    for (index, text) in pages.iter().enumerate() {
        objects.push(
            format!(
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {} 0 R /Resources << /Font << /F1 3 0 R >> >> >>",
                5 + 2 * index
            )
            .into_bytes(),
        );
        let body = if text.is_empty() {
            String::new()
        } else {
            format!("BT /F1 24 Tf 72 700 Td ({text}) Tj ET")
        };
        objects
            .push(format!("<< /Length {} >>\nstream\n{body}\nendstream", body.len()).into_bytes());
    }
    let mut out = b"%PDF-1.4\n".to_vec();
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend(format!("{} 0 obj\n", index + 1).bytes());
        out.extend(object);
        out.extend(b"\nendobj\n");
    }
    let xref = out.len();
    out.extend(format!("xref\n0 {}\n0000000000 65535 f \n", objects.len() + 1).bytes());
    for offset in offsets {
        out.extend(format!("{offset:010} 00000 n \n").bytes());
    }
    out.extend(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
            objects.len() + 1
        )
        .bytes(),
    );
    out
}

// ---- ~ in the output location ------------------------------------------------

#[test]
fn a_leading_tilde_in_the_output_location_means_the_home_directory_everywhere() {
    let setup = Setup::new();
    setup.write("note.txt", "Text\n");
    setup.write("in/a.txt", "A\n");
    setup.write("in/sub/b.txt", "B\n");
    let home = setup.path("home-dir");
    let nothing_literal = |step: &str| {
        // No directory called `~` anywhere, and nothing was claimed in the
        // working directory on its behalf.
        assert!(!setup.path("~").exists(), "{step}: literal ~ directory");
        assert!(!setup.path(".markitai").exists(), "{step}");
    };

    // A single file, from the flag.
    let single = setup.run(&["note.txt", "-o", "~/single"]);
    assert!(single.status.success(), "{}", stderr(&single));
    assert!(home.join("single/note.txt.md").is_file());
    // The line names the real location, not the spelling that was typed.
    assert!(
        stderr(&single).contains(home.join("single/note.txt.md").to_str().unwrap()),
        "{}",
        stderr(&single)
    );
    nothing_literal("single -o");

    // A chosen file name under ~.
    let named = setup.run(&["note.txt", "-o", "~/named.md"]);
    assert!(named.status.success(), "{}", stderr(&named));
    assert!(home.join("named.md").is_file());
    nothing_literal("single -o x.md");

    // A batch, from the flag, with nested directories and recovery state.
    let batch = setup.run(&["in", "-o", "~/batch"]);
    assert!(batch.status.success(), "{}", stderr(&batch));
    assert!(home.join("batch/a.txt.md").is_file());
    assert!(home.join("batch/sub/b.txt.md").is_file());
    assert!(home.join("batch/.markitai/states").is_dir());
    nothing_literal("batch -o");
    // Resuming reads the same state: it finds the saved run through ~ too.
    let resumed = setup.run(&["in", "-o", "~/batch", "--resume"]);
    assert!(resumed.status.success(), "{}", stderr(&resumed));
    assert!(
        stderr(&resumed).contains("Resuming: 2 already done, 0 remaining"),
        "{}",
        stderr(&resumed)
    );
    nothing_literal("batch --resume");

    // A batch, from the configuration.
    let configured = setup.run(&[
        "in",
        "--config-json",
        &json!({"output": {"dir": "~/from-config"}}).to_string(),
    ]);
    assert!(configured.status.success(), "{}", stderr(&configured));
    assert!(home.join("from-config/a.txt.md").is_file());
    assert!(home.join("from-config/sub/b.txt.md").is_file());
    nothing_literal("batch output.dir");

    // The preview names the real place too and still writes nothing.
    let preview = setup.run(&["in", "-o", "~/preview", "--dry-run"]);
    assert!(preview.status.success(), "{}", stderr(&preview));
    assert!(
        stdout(&preview).contains(home.join("preview/a.txt.md").to_str().unwrap()),
        "{}",
        stdout(&preview)
    );
    assert!(!home.join("preview").exists());
    nothing_literal("dry run");

    // A directory that really is called ~ stays reachable by spelling it out.
    let literal = setup.run(&["note.txt", "-o", "./~/literal"]);
    assert!(literal.status.success(), "{}", stderr(&literal));
    assert!(setup.path("~/literal/note.txt.md").is_file());
}

#[test]
fn a_tilde_that_cannot_be_expanded_is_refused_instead_of_becoming_a_directory() {
    let setup = Setup::new();
    setup.write("note.txt", "Text\n");
    setup.write("in/a.txt", "A\n");
    // No HOME and no USERPROFILE: there is no home directory to name.
    for args in [vec!["note.txt", "-o", "~/x"], vec!["in", "-o", "~/x"]] {
        let mut command = setup.command(&[]);
        command.env_remove("HOME").env_remove("USERPROFILE");
        let output = command.args(&args).output().unwrap();
        assert_eq!(output.status.code(), Some(1), "{args:?}");
        let message = stderr(&output);
        assert!(
            message.contains("Cannot expand ~") && message.contains("HOME is not set"),
            "{message}"
        );
        assert!(!setup.path("~").exists());
    }
}

// ---- progress and warnings on the terminal ----------------------------------

#[test]
fn without_a_terminal_a_batch_writes_exactly_the_lines_it_always_did() {
    let setup = Setup::new();
    for index in 0..30 {
        setup.write(&format!("in/doc{index:02}.txt"), format!("Text {index}\n"));
    }
    let output = setup.run(&["in", "-o", "out", "-j", "1"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stderr(&output);
    assert!(
        !text.contains('\r') && !text.contains('\u{1b}') && !text.contains("[1/"),
        "{text:?}"
    );
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{lines:?}");
    assert!(lines[0].starts_with("Done: 30 files ("), "{lines:?}");
    assert!(lines[1].starts_with("Output: "), "{lines:?}");
}

#[cfg(unix)]
mod terminal {
    use super::*;
    use std::io::Read;
    use std::os::fd::{FromRawFd, OwnedFd};
    use std::process::Stdio;

    /// Run the CLI with its stderr on a pseudo-terminal; the text it wrote.
    fn on_a_terminal(setup: &Setup, args: &[&str], extra: &[(&str, &str)]) -> (String, i32) {
        let (mut master, slave) = unsafe {
            let (mut master, mut slave) = (0, 0);
            let mut size = libc::winsize {
                ws_row: 24,
                ws_col: 100,
                ws_xpixel: 0,
                ws_ypixel: 0,
            };
            assert_eq!(
                libc::openpty(
                    &mut master,
                    &mut slave,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    &mut size,
                ),
                0
            );
            (
                std::fs::File::from(OwnedFd::from_raw_fd(master)),
                OwnedFd::from_raw_fd(slave),
            )
        };
        let mut command = setup.command(extra);
        command
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(slave));
        let mut child = command.spawn().unwrap();
        // Dropping the command closes this process's copy of the slave, so the
        // master reads to the end once the child exits.
        drop(command);
        let reader = std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let mut buffer = [0_u8; 4096];
            loop {
                match master.read(&mut buffer) {
                    Ok(0) | Err(_) => break,
                    Ok(count) => bytes.extend(&buffer[..count]),
                }
            }
            bytes
        });
        let status = child.wait().unwrap();
        let bytes = reader.join().unwrap();
        (
            String::from_utf8_lossy(&bytes).into_owned(),
            status.code().unwrap_or(-1),
        )
    }

    /// Enough files that a one-at-a-time batch takes a while, and some that
    /// warn, so there is something to print before the summary.
    fn long_batch(setup: &Setup) {
        for index in 0..800 {
            setup.write(&format!("in/doc{index:04}.txt"), format!("Text {index}\n"));
        }
        setup.write("in/zz-pages.pdf", pdf(&["One", "", "", "Four"]));
    }

    #[test]
    fn a_batch_on_a_terminal_shows_one_overwritten_status_line() {
        let setup = Setup::new();
        long_batch(&setup);
        let (text, code) = on_a_terminal(
            &setup,
            &["in", "-o", "out", "-j", "1"],
            &[("TERM", "xterm")],
        );
        assert_eq!(code, 0, "{text:?}");
        // The line is redrawn in place: carriage return, text, erase to the end.
        assert!(text.contains("\r["), "{text:?}");
        let status: Vec<&str> = text
            .split('\r')
            .filter(|part| part.starts_with('['))
            .collect();
        assert!(!status.is_empty(), "{text:?}");
        for part in &status {
            assert!(part.starts_with('[') && part.contains("/801] "), "{part:?}");
            assert!(part.ends_with("\u{1b}[K"), "{part:?}");
            assert!(part.len() < 100, "{part:?}");
        }
        assert!(text.contains("doc"), "{text:?}");
        // Anything printed afterwards starts on a cleared line: the warning
        // and the summary are never glued to a status line.
        let tail = text.rsplit("\u{1b}[K").next().unwrap();
        assert!(
            tail.starts_with("Warning: zz-pages.pdf: PDF pages 2-3: "),
            "{tail:?}"
        );
        assert!(tail.contains("Done: 801 files ("), "{tail:?}");
        assert!(tail.trim_end().ends_with("Output: out"), "{tail:?}");
    }

    #[test]
    fn quiet_json_and_dumb_terminals_get_no_status_line() {
        let setup = Setup::new();
        long_batch(&setup);
        let (quiet, code) = on_a_terminal(
            &setup,
            &["in", "-o", "q", "-j", "1", "-q"],
            &[("TERM", "xterm")],
        );
        assert_eq!(code, 0);
        assert!(
            !quiet.contains('\r') && !quiet.contains('\u{1b}'),
            "{quiet:?}"
        );
        let (json, code) = on_a_terminal(
            &setup,
            &["in", "-o", "j", "-j", "1", "--json"],
            &[("TERM", "xterm")],
        );
        assert_eq!(code, 0);
        assert!(json.is_empty(), "{json:?}");
        let (dumb, code) =
            on_a_terminal(&setup, &["in", "-o", "d", "-j", "1"], &[("TERM", "dumb")]);
        assert_eq!(code, 0);
        // (A terminal turns each newline into \r\n; a status line is "\r[".)
        assert!(
            !dumb.contains("\r[") && !dumb.contains('\u{1b}'),
            "{dumb:?}"
        );
        assert!(dumb.contains("Done: 801 files ("), "{dumb:?}");
    }
}

#[test]
fn pages_that_need_ocr_become_one_warning_that_names_the_ranges_and_the_flag() {
    let setup = Setup::new();
    setup.write("scan.pdf", pdf(&["One", "", "", "Four", "", ""]));
    let output = setup.run(&["scan.pdf", "-o", "out"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stderr(&output);
    let warnings: Vec<_> = text
        .lines()
        .filter(|line| line.starts_with("Warning:"))
        .collect();
    assert_eq!(warnings.len(), 1, "{text}");
    assert!(
        warnings[0].starts_with("Warning: PDF pages 2-3, 5-6: native text was not recovered ("),
        "{text}"
    );
    assert!(warnings[0].contains("--ocr"), "{text}");
    // The report keeps every page; only the terminal is condensed.
    let json = setup.run(&["scan.pdf", "-o", "again", "--json"]);
    let body: Value = serde_json::from_slice(&json.stdout).unwrap();
    let per_page = body["items"][0]["warnings"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|warning| warning.as_str().unwrap().starts_with("PDF page "))
        .count();
    assert_eq!(per_page, 4, "{body}");
    // In a batch the same holds per document.
    setup.write("many/a.pdf", pdf(&["x", "", "y", "", ""]));
    setup.write("many/b.pdf", pdf(&["Text", "", "", ""]));
    let batch = setup.run(&["many", "-o", "many-out"]);
    assert!(batch.status.success(), "{}", stderr(&batch));
    let text = stderr(&batch);
    let warnings: Vec<_> = text
        .lines()
        .filter(|line| line.starts_with("Warning:"))
        .collect();
    assert_eq!(warnings.len(), 2, "{text}");
    assert!(
        warnings[0].starts_with("Warning: a.pdf: PDF pages 2, 4-5: "),
        "{text}"
    );
    assert!(
        warnings[1].starts_with("Warning: b.pdf: PDF pages 2-4: "),
        "{text}"
    );
    assert!(!text.contains("PDF page 2:"), "{text}");
}

// ---- resuming ---------------------------------------------------------------

#[test]
fn resume_says_where_it_picks_up_and_counts_every_item() {
    let setup = Setup::new();
    setup.write("in/a.txt", "A\n");
    setup.write("in/b.txt", "B\n");
    setup.write("in/c.docx", "not a package");
    let first = setup.run(&["in", "-o", "out"]);
    assert_eq!(first.status.code(), Some(10));
    assert!(
        stderr(&first).contains("Done: 2/3 files ("),
        "{}",
        stderr(&first)
    );
    // The broken file is replaced by a good one, then the run is resumed.
    setup.write("in/c.docx", DOCX);
    let resumed = setup.run(&["in", "-o", "out", "--resume"]);
    assert!(resumed.status.success(), "{}", stderr(&resumed));
    let text = stderr(&resumed);
    assert!(
        text.starts_with("Resuming: 2 already done, 1 remaining\n"),
        "{text}"
    );
    // All three items count, not just the one converted now.
    assert!(text.contains("Done: 3/3 files ("), "{text}");
    // Nothing left: everything is reported as already done.
    let again = setup.run(&["in", "-o", "out", "--resume"]);
    let text = stderr(&again);
    assert!(
        text.starts_with("Resuming: 3 already done, 0 remaining\n"),
        "{text}"
    );
    assert!(text.contains("Done: 3/3 items ("), "{text}");
    // Quiet and JSON runs say nothing extra.
    for flag in ["-q", "--json"] {
        let output = setup.run(&["in", "-o", "out", "--resume", flag]);
        assert!(
            !stderr(&output).contains("Resuming"),
            "{flag}: {}",
            stderr(&output)
        );
    }
}

// ---- bad inputs ---------------------------------------------------------------

#[test]
fn an_empty_file_a_wrong_extension_and_a_damaged_file_are_each_described() {
    let setup = Setup::new();
    setup.write("empty.docx", b"");
    setup.write("empty.pdf", b"");
    setup.write("empty.txt", b"");
    setup.write("hello.pdf", pdf(&["Hello there"]));
    setup.write("really-a-pdf.docx", pdf(&["Hello there"]));
    setup.write("really-a-docx.pdf", DOCX);
    setup.write("cut.docx", &DOCX[..DOCX.len() / 2]);
    setup.write("cut.pdf", &pdf(&["Hello there"])[..200]);

    for name in ["empty.docx", "empty.pdf"] {
        let output = setup.run(&[name, "-o", "out"]);
        assert_eq!(output.status.code(), Some(1), "{name}");
        assert_eq!(
            stderr(&output).trim(),
            "Error: File is empty (0 bytes)",
            "{name}"
        );
    }
    // An empty text file is still an (empty) document.
    assert!(setup.run(&["empty.txt", "-o", "out"]).status.success());

    // The content decides: a PDF named .docx and a Word file named .pdf are
    // converted as what they are, with one warning that says so.
    let pdf_as_docx = setup.run(&["really-a-pdf.docx", "-o", "out"]);
    assert!(pdf_as_docx.status.success(), "{}", stderr(&pdf_as_docx));
    let text = stderr(&pdf_as_docx);
    assert!(
        text.contains(
            "Warning: The content is a PDF document although the file name ends in .docx"
        ),
        "{text}"
    );
    let markdown = std::fs::read_to_string(setup.path("out/really-a-pdf.docx.md")).unwrap();
    assert!(markdown.contains("Hello there"), "{markdown}");
    let docx_as_pdf = setup.run(&["really-a-docx.pdf", "-o", "out"]);
    assert!(docx_as_pdf.status.success(), "{}", stderr(&docx_as_pdf));
    assert!(
        stderr(&docx_as_pdf).contains(
            "Warning: The content is a Word document although the file name ends in .pdf"
        ),
        "{}",
        stderr(&docx_as_pdf)
    );
    let markdown = std::fs::read_to_string(setup.path("out/really-a-docx.pdf.md")).unwrap();
    assert!(markdown.contains("WORD FIRST PAGE"), "{markdown}");
    // A genuine file draws no such warning.
    let genuine = setup.run(&["hello.pdf", "-o", "out"]);
    assert!(
        !stderr(&genuine).contains("content is"),
        "{}",
        stderr(&genuine)
    );

    // A damaged container reads as damaged, with the parser's words after it.
    for name in ["cut.docx", "cut.pdf"] {
        let output = setup.run(&[name, "-o", "out"]);
        assert_eq!(output.status.code(), Some(1), "{name}");
        assert!(
            stderr(&output).starts_with("Error: The file appears to be damaged or truncated ("),
            "{name}: {}",
            stderr(&output)
        );
    }
}

#[test]
fn an_image_that_is_not_one_is_an_error_and_a_real_one_is_skipped_by_name() {
    let setup = Setup::new();
    setup.write("pic.png", TINY_PNG);
    setup.write("empty.png", b"");
    setup.write("fake.png", b"this is text");
    setup.write("cut.png", &TINY_PNG[..TINY_PNG.len() - 20]);
    let skipped = setup.run(&[setup.path("pic.png").to_str().unwrap(), "-o", "out"]);
    assert!(skipped.status.success(), "genuine skip keeps exit status 0");
    assert!(
        stderr(&skipped).starts_with("Skipped pic.png: an image has no text to extract"),
        "{}",
        stderr(&skipped)
    );
    // A skip leaves nothing behind: no output directory, no ownership files.
    assert!(!setup.path("out").exists());
    for (name, message) in [
        ("empty.png", "Error: File is empty (0 bytes)"),
        ("fake.png", "Error: File is not a valid image: "),
        (
            "cut.png",
            "Error: File is not a valid image: the file appears to be damaged or truncated",
        ),
    ] {
        let output = setup.run(&[name, "-o", "out"]);
        assert_eq!(output.status.code(), Some(1), "{name}");
        assert!(
            stderr(&output).starts_with(message),
            "{name}: {}",
            stderr(&output)
        );
    }
    assert!(!setup.path("out").exists());
}

// ---- Office documents without LibreOffice -------------------------------------

#[test]
fn screenshots_without_libreoffice_still_publish_the_text_with_one_warning() {
    if markitai_core::office_render_available() {
        // The renderer is installed here; the degraded path cannot be reached.
        return;
    }
    let setup = Setup::new();
    setup.write("a.docx", DOCX);
    setup.write("in/a.docx", DOCX);
    setup.write("in/b.docx", DOCX);
    for args in [
        vec!["a.docx", "--screenshot", "-o", "out"],
        // The rich preset implies screenshots; --no-llm keeps the model out of it.
        vec!["a.docx", "-p", "rich", "--no-llm", "-o", "out"],
    ] {
        let output = setup.run(&args);
        assert!(output.status.success(), "{args:?}: {}", stderr(&output));
        let text = stderr(&output);
        let warnings: Vec<_> = text
            .lines()
            .filter(|line| line.starts_with("Warning:"))
            .collect();
        assert_eq!(warnings.len(), 1, "{args:?}: {text}");
        assert!(
            warnings[0].contains("brew install --cask libreoffice")
                && warnings[0].contains("--no-screenshot"),
            "{text}"
        );
        let markdown = std::fs::read_to_string(setup.path("out/a.docx.md")).unwrap();
        assert!(markdown.contains("WORD FIRST PAGE"), "{markdown}");
        std::fs::remove_dir_all(setup.path("out")).unwrap();
    }
    // A folder of Office files: the shared warning is written once.
    let batch = setup.run(&["in", "--screenshot", "-o", "batch-out"]);
    assert!(batch.status.success(), "{}", stderr(&batch));
    let text = stderr(&batch);
    assert_eq!(
        text.lines()
            .filter(|line| line.starts_with("Warning:"))
            .count(),
        1,
        "{text}"
    );
    assert!(
        text.contains("Warning: a.docx, b.docx: Page screenshots"),
        "{text}"
    );
    assert!(text.contains("Done: 2 files ("), "{text}");
    // Only screenshots as the whole output keep failing, and say what to do.
    let only = setup.run(&["a.docx", "--screenshot-only", "-o", "only"]);
    assert_eq!(only.status.code(), Some(1));
    assert!(
        stderr(&only).contains("brew install --cask libreoffice"),
        "{}",
        stderr(&only)
    );
    assert!(!setup.path("only/a.docx.md").exists());
}

// ---- previews -----------------------------------------------------------------

#[test]
fn a_preview_names_each_final_file_and_what_would_be_skipped() {
    let setup = Setup::new();
    setup.write("in/report.txt", "R\n");
    setup.write("in/new.txt", "N\n");
    setup.write("in/pic.png", TINY_PNG);
    setup.write("out/report.txt.md", "already here\n");
    let preview = setup.run(&["in", "-o", "out", "--dry-run"]);
    assert!(preview.status.success(), "{}", stderr(&preview));
    assert_eq!(
        stdout(&preview),
        "new.txt -> out/new.txt.md\n\
         pic.png -> skip (an image needs --ocr or --llm)\n\
         report.txt -> out/report.txt.v2.md (report.txt.md already exists)\n"
    );
    assert_eq!(
        stderr(&preview),
        "Dry run: 2 files would be converted and 1 skipped; nothing was written.\n"
    );
    // The policy decides what a name that is taken becomes.
    let skip = setup.run(&[
        "in/report.txt",
        "-o",
        "out",
        "--dry-run",
        "--config-json",
        r#"{"output":{"on_conflict":"skip"}}"#,
    ]);
    assert_eq!(
        stdout(&skip),
        "in/report.txt -> skip (out/report.txt.md already exists and output.on_conflict is skip)\n"
    );
    assert_eq!(
        stderr(&skip),
        "Dry run: nothing would be converted, 1 skipped; nothing was written.\n"
    );
    let overwrite = setup.run(&[
        "in/report.txt",
        "-o",
        "out",
        "--dry-run",
        "--config-json",
        r#"{"output":{"on_conflict":"overwrite"}}"#,
    ]);
    assert_eq!(
        stdout(&overwrite),
        "in/report.txt -> out/report.txt.md (replaces the existing file)\n"
    );
    // A single input gets the summary line too, and an explicit name is honored.
    let single = setup.run(&["in/new.txt", "-o", "out/chosen.md", "--dry-run"]);
    assert_eq!(stdout(&single), "in/new.txt -> out/chosen.md\n");
    assert_eq!(
        stderr(&single),
        "Dry run: 1 file would be converted; nothing was written.\n"
    );
    let to_stdout = setup.run(&["in/new.txt", "--dry-run"]);
    assert_eq!(stdout(&to_stdout), "in/new.txt -> stdout\n");
    let quiet = setup.run(&["in/new.txt", "--dry-run", "-q"]);
    assert!(quiet.stderr.is_empty());
    // Nothing was written by any of these.
    assert_eq!(
        std::fs::read_to_string(setup.path("out/report.txt.md")).unwrap(),
        "already here\n"
    );
    assert!(!setup.path("out/new.txt.md").exists());
}

// ---- hidden files, renames and typos ------------------------------------------

#[test]
fn directory_batches_leave_out_hidden_files_and_dependency_folders_unless_asked() {
    let setup = Setup::new();
    for path in [
        "docs/a.txt",
        "docs/sub/b.txt",
        "docs/.git/config.txt",
        "docs/.hidden/c.txt",
        "docs/.secret.txt",
        "docs/node_modules/pkg/readme.txt",
        "docs/sub/node_modules/deep.txt",
        "docs/~$lock.txt",
        "docs/.github/workflows/d.txt",
    ] {
        setup.write(path, "Text\n");
    }
    let names = |preview: &Output| -> Vec<String> {
        stdout(preview)
            .lines()
            .map(|line| line.split(" -> ").next().unwrap().to_owned())
            .collect()
    };
    let preview = setup.run(&["docs", "-o", "out", "--dry-run"]);
    assert!(preview.status.success(), "{}", stderr(&preview));
    assert_eq!(names(&preview), ["a.txt", "sub/b.txt"]);
    assert!(
        stderr(&preview).contains("2 files would be converted"),
        "{}",
        stderr(&preview)
    );
    // The run itself agrees.
    let run = setup.run(&["docs", "-o", "out"]);
    assert!(run.status.success(), "{}", stderr(&run));
    assert!(setup.path("out/a.txt.md").is_file());
    assert!(!setup.path("out/.git").exists() && !setup.path("out/.hidden").exists());
    assert!(!setup.path("out/node_modules").exists());
    // A glob that spells a hidden or dependency name out asks for it.
    let github = setup.run(&["docs", "-o", "out", "--dry-run", "-g", ".github/**/*.txt"]);
    assert_eq!(names(&github), [".github/workflows/d.txt"]);
    let deps = setup.run(&[
        "docs",
        "-o",
        "out",
        "--dry-run",
        "-g",
        "**/node_modules/**/*.txt",
    ]);
    assert_eq!(
        names(&deps),
        ["node_modules/pkg/readme.txt", "sub/node_modules/deep.txt"]
    );
    // An ordinary glob does not reach into them.
    let all = setup.run(&["docs", "-o", "out", "--dry-run", "-g", "**/*.txt"]);
    assert_eq!(names(&all), ["a.txt", "sub/b.txt"]);
    // A hidden directory given as the input itself is the user's choice.
    let inside = setup.run(&["docs/.hidden", "-o", "hidden-out", "--dry-run"]);
    assert_eq!(names(&inside), ["c.txt"]);
}

#[test]
fn a_rename_names_the_file_in_the_way_and_a_mistyped_command_is_suggested() {
    let setup = Setup::new();
    setup.write("note.txt", "Text\n");
    setup.write("in/a.txt", "A\n");
    assert!(setup.run(&["note.txt", "-o", "out"]).status.success());
    let second = setup.run(&["note.txt", "-o", "out"]);
    assert_eq!(
        stderr(&second).trim(),
        format!(
            "Wrote {} (note.txt.md already exists)",
            Path::new("out/note.txt.v2.md").display()
        )
    );
    // A batch rerun lists what was renamed.
    assert!(setup.run(&["in", "-o", "batch"]).status.success());
    let rerun = setup.run(&["in", "-o", "batch"]);
    let text = stderr(&rerun);
    assert!(
        text.contains("Renamed 1 item (output already exists): a.txt.v2.md. Set output.on_conflict to overwrite or skip to change this."),
        "{text}"
    );
    // The typo: the error stays, and a hint follows it.
    let typo = setup.run(&["docter"]);
    assert_eq!(typo.status.code(), Some(1));
    assert_eq!(
        stderr(&typo),
        "Error: Input file not found: docter\nHint: did you mean 'markitai doctor'?\n"
    );
    let zh = setup.run_env(&["docter"], &[("MARKITAI_LANG", "zh")]);
    assert!(
        stderr(&zh).contains("Hint: 你是想运行 'markitai doctor' 吗？"),
        "{}",
        stderr(&zh)
    );
    // Unrelated missing files get no guess.
    let other = setup.run(&["missing.txt"]);
    assert_eq!(
        stderr(&other).trim(),
        "Error: Input file not found: missing.txt"
    );
}

// ---- configuration and account commands -------------------------------------

#[test]
fn config_set_and_config_path_say_which_file_and_where_files_are_looked_for() {
    let setup = Setup::new();
    let set = setup.run(&["config", "set", "output.on_conflict", "skip"]);
    assert!(set.status.success(), "{}", stderr(&set));
    // stdout keeps the value alone; the file that changed is on stderr.
    assert_eq!(stdout(&set), "output.on_conflict = \"skip\"\n");
    assert_eq!(
        stderr(&set).trim(),
        format!("Saved to {}", setup.path("mh/config.json").display())
    );
    let zh = setup.run_env(
        &["config", "set", "output.on_conflict", "rename"],
        &[("MARKITAI_LANG", "zh")],
    );
    assert!(stderr(&zh).starts_with("已保存到 "), "{}", stderr(&zh));
    // With no file, the search order is listed; with one, only its path.
    let fresh = Setup::new();
    let none = fresh.run(&["config", "path"]);
    let text = stdout(&none);
    assert!(
        text.starts_with("No configuration file found; using built-in defaults."),
        "{text}"
    );
    assert!(
        text.contains(
            "Searched in this order: -c FILE, the MARKITAI_CONFIG variable, ./markitai.json, "
        ),
        "{text}"
    );
    let found = setup.run(&["config", "path"]);
    assert_eq!(
        stdout(&found).trim(),
        setup.path("mh/config.json").to_str().unwrap()
    );
}

#[test]
fn doctor_gives_the_install_command_for_a_missing_libreoffice() {
    if markitai_core::office_render_available() {
        return;
    }
    let setup = Setup::new();
    // An empty PATH holds no soffice, and the browser check is kept short.
    std::fs::create_dir(setup.path("bin")).unwrap();
    let nowhere = setup.path("absent-browser");
    let output = setup.run_env(
        &["doctor", "--json"],
        &[
            ("PATH", setup.path("bin").to_str().unwrap()),
            ("MARKITAI_BROWSER_EXECUTABLE", nowhere.to_str().unwrap()),
        ],
    );
    let body: Value = serde_json::from_slice(&output.stdout).unwrap();
    let hint = body["libreoffice"]["install_hint"].as_str().unwrap();
    assert!(hint.contains("brew install --cask libreoffice"), "{hint}");
}
