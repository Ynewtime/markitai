//! Explicit per-user installation of the official Chrome headless shell.
use crate::{Error, Result, browser, config, private_install};
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, Write};
use std::path::{Component, Path, PathBuf};
use std::time::Duration;

const MANIFEST: &str = "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json";
const MANIFEST_LIMIT: u64 = 1024 * 1024;
const ARCHIVE_LIMIT: u64 = 512 * 1024 * 1024;
const EXPANDED_LIMIT: u64 = 1536 * 1024 * 1024;
const FILE_LIMIT: usize = 16_384;

#[derive(Serialize, Deserialize)]
struct Receipt {
    schema: u8,
    version: String,
    platform: String,
    directory: String,
    archive_sha256: String,
    executable_sha256: String,
    source: String,
}

struct Download {
    version: String,
    url: String,
}

fn failure(message: &str) -> Error {
    Error::Conversion(format!("Browser installation: {message}"))
}

fn platform() -> Result<&'static str> {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("macos", "aarch64") => Ok("mac-arm64"),
        ("macos", "x86_64") => Ok("mac-x64"),
        ("linux", "aarch64") => Ok("linux-arm64"),
        ("linux", "x86_64") => Ok("linux64"),
        ("windows", "x86_64") => Ok("win64"),
        ("windows", "x86") => Ok("win32"),
        _ => Err(Error::Unsupported(
            "No official headless-shell installation is configured for this OS and architecture"
                .into(),
        )),
    }
}

fn version_valid(version: &str) -> bool {
    let parts: Vec<_> = version.split('.').collect();
    parts.len() == 4
        && parts.iter().all(|part| {
            !part.is_empty()
                && part.len() <= 10
                && part.bytes().all(|b| b.is_ascii_digit())
                && part.parse::<u32>().is_ok()
        })
}

fn select(bytes: &[u8], platform: &str) -> Result<Download> {
    if bytes.len() as u64 > MANIFEST_LIMIT {
        return Err(failure("manifest exceeds its byte limit"));
    }
    let value: Value =
        serde_json::from_slice(bytes).map_err(|_| failure("invalid official manifest"))?;
    let channel = &value["channels"]["Stable"];
    let version = channel["version"]
        .as_str()
        .filter(|v| version_valid(v))
        .ok_or_else(|| failure("manifest has no valid stable version"))?;
    let rows = channel
        .pointer("/downloads/chrome-headless-shell")
        .and_then(Value::as_array)
        .ok_or_else(|| failure("manifest has no headless shell"))?;
    let matches: Vec<_> = rows
        .iter()
        .filter(|row| row["platform"] == platform)
        .collect();
    if matches.len() != 1 {
        return Err(failure("manifest has no unique download for this platform"));
    }
    let expected = format!(
        "https://storage.googleapis.com/chrome-for-testing-public/{version}/{platform}/chrome-headless-shell-{platform}.zip"
    );
    if matches[0]["url"].as_str() != Some(&expected) {
        return Err(failure(
            "manifest download is not the expected official HTTPS asset",
        ));
    }
    Ok(Download {
        version: version.into(),
        url: expected,
    })
}

fn relative_executable(platform: &str) -> PathBuf {
    PathBuf::from(format!("chrome-headless-shell-{platform}")).join(
        if platform.starts_with("win") {
            "chrome-headless-shell.exe"
        } else {
            "chrome-headless-shell"
        },
    )
}

fn regular(path: &Path) -> bool {
    fs::symlink_metadata(path).is_ok_and(|meta| meta.is_file() && !meta.file_type().is_symlink())
}

fn installed_at(root: &Path, platform: &str) -> Option<PathBuf> {
    let meta = fs::symlink_metadata(root).ok()?;
    if !meta.is_dir() || meta.file_type().is_symlink() {
        return None;
    }
    let receipt = root.join("current.json");
    if !regular(&receipt) {
        return None;
    }
    let mut bytes = Vec::new();
    File::open(receipt)
        .ok()?
        .take(65_537)
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() > 65_536 {
        return None;
    }
    let receipt: Receipt = serde_json::from_slice(&bytes).ok()?;
    if receipt.schema != 1
        || receipt.platform != platform
        || !version_valid(&receipt.version)
        || receipt.directory.len() > 128
        || !receipt
            .directory
            .starts_with(&format!("headless-{}-{platform}-", receipt.version))
        || !receipt
            .directory
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.'))
    {
        return None;
    }
    let folder = root.join(&receipt.directory);
    for directory in [
        &folder,
        &folder.join(format!("chrome-headless-shell-{platform}")),
    ] {
        let metadata = fs::symlink_metadata(directory).ok()?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return None;
        }
    }
    let executable = folder.join(relative_executable(platform));
    if !regular(&executable) {
        return None;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if fs::metadata(&executable).ok()?.permissions().mode() & 0o111 == 0 {
            return None;
        }
    }
    Some(executable)
}

pub(crate) fn installed() -> Option<PathBuf> {
    installed_at(&config::home().join("browsers/native"), platform().ok()?)
}

fn private_directory(path: &Path) -> Result<()> {
    private_install::private_directory(path, failure)
}

fn lock(root: &Path) -> Result<private_install::InstallLock> {
    private_install::lock(
        root,
        private_install::Contention::Refuse(
            "another browser installation is active or locking is unavailable",
        ),
        failure,
    )
}

fn download(client: &Client, url: &str, limit: u64, output: &mut impl Write) -> Result<String> {
    private_install::download(client, url, limit, output, failure)
}

fn extract(reader: impl Read + Seek, output: &Path, platform: &str, limit: u64) -> Result<()> {
    let mut archive =
        zip::ZipArchive::new(reader).map_err(|_| failure("invalid browser ZIP archive"))?;
    if archive.len() > FILE_LIMIT {
        return Err(failure("browser ZIP contains too many entries"));
    }
    let required_root = format!("chrome-headless-shell-{platform}");
    let mut seen = HashSet::new();
    let mut total = 0u64;
    for index in 0..archive.len() {
        let mut entry = archive
            .by_index(index)
            .map_err(|_| failure("cannot read browser ZIP entry"))?;
        let name = entry.name();
        if name.contains(['\\', ':', '\0']) || name.len() > 1024 {
            return Err(failure("browser ZIP contains an invalid path"));
        }
        let relative = entry
            .enclosed_name()
            .ok_or_else(|| failure("browser ZIP path escapes installation"))?;
        if relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
            || relative
                .components()
                .next()
                .is_none_or(|c| c.as_os_str() != required_root.as_str())
            || !seen.insert(relative.to_string_lossy().to_ascii_lowercase())
        {
            return Err(failure(
                "browser ZIP contains an unexpected or duplicate path",
            ));
        }
        let mode = entry.unix_mode().unwrap_or(0);
        let kind = mode & 0o170000;
        if !matches!(kind, 0 | 0o040000 | 0o100000) {
            return Err(failure("browser ZIP contains a link or special file"));
        }
        let path = output.join(relative);
        if entry.is_dir() {
            if kind == 0o100000 {
                return Err(failure("inconsistent ZIP directory type"));
            }
            private_directory(&path)?;
            continue;
        }
        if kind == 0o040000 {
            return Err(failure("inconsistent ZIP file type"));
        }
        if entry.size() > limit.saturating_sub(total) {
            return Err(failure("expanded browser exceeds its byte limit"));
        }
        private_directory(
            path.parent()
                .ok_or_else(|| failure("invalid browser ZIP path"))?,
        )?;
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(if mode & 0o111 != 0 { 0o700 } else { 0o600 });
        }
        let mut file = options
            .open(path)
            .map_err(|_| failure("cannot create extracted browser file"))?;
        let copied = std::io::copy(
            &mut (&mut entry).take(limit.saturating_sub(total) + 1),
            &mut file,
        )
        .map_err(|_| failure("browser ZIP decompression failed"))?;
        total += copied;
        if total > limit || copied != entry.size() {
            return Err(failure("browser ZIP expanded size is inconsistent"));
        }
        file.sync_all()?;
    }
    if !regular(&output.join(relative_executable(platform))) {
        return Err(failure("browser ZIP has no expected executable"));
    }
    Ok(())
}

/// Network/install work happens only when this API is explicitly requested.
pub(crate) fn install() -> Result<PathBuf> {
    if std::env::var_os("MARKITAI_BROWSER_EXECUTABLE").is_some() {
        return Err(failure(
            "MARKITAI_BROWSER_EXECUTABLE is explicitly configured; repair that path or unset it; no installer was started",
        ));
    }
    let platform = platform()?;
    let root = config::home().join("browsers/native");
    private_directory(&root)?;
    let _lock = lock(&root)?;
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(15))
        .timeout(Duration::from_secs(600))
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(concat!("markitai/", env!("CARGO_PKG_VERSION")))
        .build()
        .map_err(|_| failure("cannot initialize download client"))?;
    let mut manifest = Vec::new();
    download(&client, MANIFEST, MANIFEST_LIMIT, &mut manifest)?;
    let chosen = select(&manifest, platform)?;
    let mut builder = tempfile::Builder::new();
    builder.prefix(".install-");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        builder.permissions(fs::Permissions::from_mode(0o700));
    }
    let stage = builder.tempdir_in(&root)?;
    let mut archive = tempfile::tempfile_in(stage.path())?;
    let archive_sha256 = download(&client, &chosen.url, ARCHIVE_LIMIT, &mut archive)?;
    archive.rewind()?;
    extract(&mut archive, stage.path(), platform, EXPANDED_LIMIT)?;
    drop(archive);
    publish(
        &root,
        stage,
        platform,
        chosen,
        archive_sha256,
        |executable| {
            browser::diagnostic_with(executable).map_err(|_| {
                failure("downloaded browser cannot start; the previous installation was retained")
            })
        },
    )
}

fn publish(
    root: &Path,
    stage: tempfile::TempDir,
    platform: &str,
    chosen: Download,
    archive_sha256: String,
    verify: impl FnOnce(&Path) -> Result<()>,
) -> Result<PathBuf> {
    let executable = stage.path().join(relative_executable(platform));
    verify(&executable)?;
    let executable_sha256 = private_install::hash_file(&executable)?;
    let suffix = stage
        .path()
        .file_name()
        .unwrap()
        .to_string_lossy()
        .replace('.', "");
    let directory = format!("headless-{}-{platform}-{suffix}", chosen.version);
    let receipt = Receipt {
        schema: 1,
        version: chosen.version,
        platform: platform.into(),
        directory,
        archive_sha256,
        executable_sha256,
        source: chosen.url,
    };
    let destination = root.join(&receipt.directory);
    crate::platform::rename(stage.path(), &destination)
        .map_err(|_| failure("cannot publish browser directory"))?;
    let mut current = tempfile::NamedTempFile::new_in(root)?;
    serde_json::to_writer_pretty(current.as_file_mut(), &receipt)?;
    current.as_file().sync_all()?;
    crate::platform::persist(current, &root.join("current.json"))
        .map_err(|_| failure("browser downloaded but current installation could not be updated"))?;
    #[cfg(unix)]
    File::open(root)?.sync_all()?;
    Ok(destination.join(relative_executable(platform)))
}

#[cfg(test)]
mod tests;
