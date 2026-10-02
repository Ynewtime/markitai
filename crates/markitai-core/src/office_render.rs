//! Optional, isolated LibreOffice export. Native readers remain responsible for text.

use crate::{Error, Result};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    time::{Duration, Instant},
};
use tempfile::TempDir;

mod process;
mod slides;
#[cfg(test)]
mod tests;
mod workbooks;

const MAX_BYTES: u64 = 100 * 1024 * 1024;
const MAX_PAGES: usize = 1000;
const TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum OfficeKind {
    Presentation,
    WordProcessing,
    Spreadsheet,
}

pub(crate) struct OfficePdf {
    pub(crate) bytes: Vec<u8>,
    pub(crate) pages: usize,
    pub(crate) warnings: Vec<String>,
    _workspace: TempDir,
}

pub(crate) fn kind(extension: &str) -> Option<OfficeKind> {
    match extension
        .trim_start_matches('.')
        .to_ascii_lowercase()
        .as_str()
    {
        "ppt" | "pps" | "pot" | "pptx" | "pptm" | "ppsx" | "ppsm" | "odp" => {
            Some(OfficeKind::Presentation)
        }
        "doc" | "docx" | "docm" | "odt" | "rtf" => Some(OfficeKind::WordProcessing),
        "xls" | "xlsx" | "xlsm" | "xlsb" | "ods" | "numbers" => Some(OfficeKind::Spreadsheet),
        _ => None,
    }
}

fn failure(message: &str) -> Error {
    Error::Conversion(format!("Office page rendering: {message}"))
}

// Only the fixed install locations below need a direct check; PATH lookups go
// through the shared program search.
#[cfg(any(target_os = "macos", target_os = "windows"))]
fn executable(path: &Path) -> bool {
    crate::process_groups::launchable(path)
}

fn discover() -> Option<PathBuf> {
    // On Windows the GUI launcher is preferred to the `soffice.com` console
    // wrapper beside it; both wait for the export.
    let names: &[&str] = if cfg!(windows) {
        &["soffice.exe", "soffice", "libreoffice"]
    } else {
        &["soffice", "libreoffice"]
    };
    // Empty and relative PATH entries never select a document-local program.
    if let Some(path) = crate::process_groups::find_program(
        names,
        std::env::var_os("PATH").as_deref(),
        std::env::var_os("PATHEXT").as_deref(),
    ) {
        return Some(path);
    }
    #[cfg(target_os = "macos")]
    for path in [
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
        "/Applications/LibreOfficeDev.app/Contents/MacOS/soffice",
    ] {
        if executable(Path::new(path)) {
            return Some(path.into());
        }
    }
    #[cfg(target_os = "windows")]
    for variable in ["ProgramFiles", "ProgramFiles(x86)"] {
        if let Some(directory) = std::env::var_os(variable) {
            let candidate = PathBuf::from(directory).join("LibreOffice/program/soffice.exe");
            if executable(&candidate) {
                return Some(candidate);
            }
        }
    }
    None
}

pub(crate) fn available() -> bool {
    discover().is_some()
}

fn private_workspace(prefix: &str) -> Result<TempDir> {
    let mut builder = tempfile::Builder::new();
    builder.prefix(prefix);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(fs::Permissions::from_mode(0o700));
    }
    Ok(builder.tempdir()?)
}

/// Discover and start the optional program with private state and a short deadline.
pub(crate) fn diagnostic() -> Result<Option<PathBuf>> {
    let Some(program) = discover() else {
        return Ok(None);
    };
    let workspace = private_workspace("markitai-office-diagnostic-")?;
    let deadline = Instant::now() + Duration::from_secs(10);
    let _permit = process::acquire(deadline)?;
    process::diagnose(&program, workspace.path(), deadline).map_err(|_| {
        failure("LibreOffice diagnostic could not complete within its startup deadline")
    })?;
    Ok(Some(program))
}

pub(crate) fn export_pdf(input: &Path, requested_kind: OfficeKind) -> Result<OfficePdf> {
    if input
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("numbers"))
    {
        return Err(Error::Unsupported("Numbers complete-sheet screenshots are not supported by this LibreOffice adapter; native Numbers table reading remains available without screenshot or OCR options".into()));
    }
    let program = discover().ok_or_else(|| Error::Unsupported("Office screenshots require an installed LibreOffice (soffice on PATH) and the native PDF page renderer".into()))?;
    export_with(&program, input, requested_kind, TIMEOUT, MAX_BYTES)
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    let metadata = fs::symlink_metadata(path).map_err(|_| failure("expected file is missing"))?;
    if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() > limit {
        return Err(failure("file is not regular or exceeds the 100 MiB limit"));
    }
    let mut options = fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options
        .open(path)
        .map_err(|_| failure("file cannot be read"))?;
    if !file.metadata()?.is_file() {
        return Err(failure("file is not regular"));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1).read_to_end(&mut bytes)?;
    if bytes.len() as u64 > limit {
        return Err(failure("file exceeds its byte limit"));
    }
    Ok(bytes)
}

fn export_with(
    program: &Path,
    input: &Path,
    requested_kind: OfficeKind,
    timeout: Duration,
    byte_limit: u64,
) -> Result<OfficePdf> {
    let extension = input
        .extension()
        .and_then(|v| v.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if kind(&extension) != Some(requested_kind) {
        return Err(failure("unsupported or mismatched Office document type"));
    }
    let deadline = Instant::now() + timeout;
    let _permit = process::acquire(deadline)?;
    let bytes = read_bounded(input, byte_limit)?;
    let workspace = private_workspace("markitai-office-")?;
    let source_dir = workspace.path().join("input");
    let output_dir = workspace.path().join("output");
    let profile = workspace.path().join("profile");
    for directory in [&source_dir, &output_dir, &profile] {
        fs::create_dir(directory)?;
    }
    fs::create_dir(profile.join("user"))?;
    // Isolated configuration disables document macros and automatic linked-document updates.
    fs::write(profile.join("user/registrymodifications.xcu"), br#"<?xml version="1.0"?><oor:items xmlns:oor="http://openoffice.org/2001/registry"><item oor:path="/org.openoffice.Office.Common/Security/Scripting"><prop oor:name="MacroSecurityLevel" oor:op="fuse"><value>3</value></prop></item><item oor:path="/org.openoffice.Office.Common/Load"><prop oor:name="UpdateLink" oor:op="fuse"><value>2</value></prop></item></oor:items>"#)?;
    let source = source_dir.join(format!("document.{extension}"));
    fs::write(&source, &bytes)?;
    let mut expected = match requested_kind {
        OfficeKind::Presentation if extension == "odp" => Some(slides::odp(&bytes)?),
        OfficeKind::Presentation
            if matches!(extension.as_str(), "pptx" | "pptm" | "ppsx" | "ppsm") =>
        {
            Some(crate::formats::extract_presentation_count(&bytes)?)
        }
        OfficeKind::Spreadsheet if extension == "ods" => Some(workbooks::ods(&bytes)?),
        OfficeKind::Spreadsheet if matches!(extension.as_str(), "xlsx" | "xlsm") => {
            Some(workbooks::xlsx(&bytes)?)
        }
        _ => None,
    };
    drop(bytes);
    if expected.is_some_and(|count| count == 0 || count > MAX_PAGES) {
        return Err(failure(
            "source page or sheet count exceeds 1,000 or is empty",
        ));
    }
    let mut render_source = source;
    let mut remaining_output = byte_limit;
    let mut warnings = vec!["Office page layout is rendered by the installed LibreOffice; fonts and pagination can differ from Microsoft Office".into()];
    if requested_kind == OfficeKind::Presentation && expected.is_none() {
        let normalized = workspace.path().join("normalized");
        fs::create_dir(&normalized)?;
        process::convert(
            program,
            &render_source,
            &normalized,
            &profile,
            "pptx:Impress MS PowerPoint 2007 XML",
            deadline,
            byte_limit,
        )?;
        let converted = expected_file(&normalized, "document.pptx", byte_limit)?;
        remaining_output = byte_limit
            .checked_sub(converted.len() as u64)
            .ok_or_else(|| failure("normalized document exhausted export byte limit"))?;
        expected = Some(crate::formats::extract_presentation_count(&converted)?);
        if expected.is_some_and(|count| count == 0 || count > MAX_PAGES) {
            return Err(failure("slide count exceeds 1,000 or is empty"));
        }
        render_source = normalized.join("document.pptx");
        warnings.push("Legacy presentation slide count was verified against LibreOffice's imported PPTX model before PDF export".into());
    }
    if requested_kind == OfficeKind::Spreadsheet {
        if expected.is_none() {
            // Binary Excel and Numbers have different container models. Count the
            // complete private import rather than assuming one exported page.
            let normalized = workspace.path().join("normalized");
            fs::create_dir(&normalized)?;
            process::convert(
                program,
                &render_source,
                &normalized,
                &profile,
                "ods:calc8",
                deadline,
                byte_limit,
            )?;
            let converted = expected_file(&normalized, "document.ods", byte_limit)?;
            remaining_output = byte_limit
                .checked_sub(converted.len() as u64)
                .ok_or_else(|| failure("normalized workbook exhausted export byte limit"))?;
            expected = Some(workbooks::ods(&converted)?);
            render_source = normalized.join("document.ods");
            warnings.push("Workbook sheet count was verified against LibreOffice's imported ODS model; this does not establish complete source-format import fidelity".into());
        }
        warnings.push("Workbook screenshots use complete-sheet export: all sheets, including hidden and empty sheets, retain their full content bounds; paper sizes, print areas and manual print pagination are ignored. Oversized sheets fail the native page-pixel limits rather than being truncated.".into());
    }
    let filter = match requested_kind {
        OfficeKind::Presentation => {
            r#"pdf:impress_pdf_Export:{"ExportHiddenSlides":{"type":"boolean","value":"true"},"ExportNotesPages":{"type":"boolean","value":"false"}}"#
        }
        OfficeKind::WordProcessing => {
            r#"pdf:writer_pdf_Export:{"IsSkipEmptyPages":{"type":"boolean","value":"false"}}"#
        }
        OfficeKind::Spreadsheet => {
            r#"pdf:calc_pdf_Export:{"SinglePageSheets":{"type":"boolean","value":"true"}}"#
        }
    };
    process::convert(
        program,
        &render_source,
        &output_dir,
        &profile,
        filter,
        deadline,
        remaining_output,
    )?;
    let bytes = expected_file(&output_dir, "document.pdf", remaining_output)?;
    let pages = if requested_kind == OfficeKind::Spreadsheet {
        workbooks::validate_pdf(&bytes, expected.expect("workbook import was counted"))?
    } else {
        validate_pdf(&bytes, expected)?
    };
    Ok(OfficePdf {
        bytes,
        pages,
        warnings,
        _workspace: workspace,
    })
}

fn expected_file(directory: &Path, name: &str, limit: u64) -> Result<Vec<u8>> {
    let mut entries = fs::read_dir(directory)?;
    let entry = entries
        .next()
        .transpose()?
        .ok_or_else(|| failure("LibreOffice did not create the expected output"))?;
    if entry.file_name() != name || entries.next().is_some() {
        return Err(failure("LibreOffice created unexpected output files"));
    }
    read_bounded(&entry.path(), limit)
}

fn validate_pdf(bytes: &[u8], expected: Option<usize>) -> Result<usize> {
    if !bytes.starts_with(b"%PDF-") {
        return Err(failure("export is not a PDF document"));
    }
    let pdf = lopdf::Document::load_mem(bytes).map_err(|_| failure("exported PDF is invalid"))?;
    if pdf.is_encrypted() {
        return Err(failure("exported PDF is encrypted"));
    }
    let pages = pdf.get_pages();
    if pages.is_empty() || pages.len() > MAX_PAGES {
        return Err(failure("exported PDF has no pages or exceeds 1,000 pages"));
    }
    if expected.is_some_and(|expected| expected != pages.len()) {
        return Err(failure(
            "exported PDF page count does not match every source slide, including hidden and blank slides",
        ));
    }
    for id in pages.values() {
        let page = pdf
            .get_dictionary(*id)
            .map_err(|_| failure("exported PDF has an invalid page"))?;
        if page.get(b"Type").ok().and_then(|v| v.as_name().ok()) != Some(b"Page".as_slice()) {
            return Err(failure("exported PDF has an invalid page type"));
        }
    }
    Ok(pages.len())
}
