//! Everything the app knows apart from GTK: the Convert list, History,
//! settings, the core's envelopes and how values read. No display needed, so
//! `cargo test` covers it.

pub mod errors;
pub mod format;
pub mod history;
pub mod outcome;
pub mod queue;
pub mod runs;
pub mod settings;
pub mod source;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const APP_ID: &str = "dev.markitai.native";

/// Write `<path>.tmp` in the same folder, flush it, then rename it over
/// `path`, creating the folder first.
pub fn write_atomically(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = PathBuf::from(tmp);
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, path)
}

/// The XDG base folder in `var`, else `$HOME/<fallback>`.
pub fn xdg_dir(var: &str, fallback: &str) -> PathBuf {
    std::env::var_os(var)
        .map(PathBuf::from)
        .filter(|p| p.is_absolute())
        .unwrap_or_else(|| home().join(fallback))
}

pub fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/"))
}

/// `$XDG_DATA_HOME/dev.markitai.native`: History and the working folder.
pub fn data_dir() -> PathBuf {
    xdg_dir("XDG_DATA_HOME", ".local/share").join(APP_ID)
}

/// `$XDG_CONFIG_HOME/dev.markitai.native/settings.json`.
pub fn settings_path() -> PathBuf {
    xdg_dir("XDG_CONFIG_HOME", ".config")
        .join(APP_ID)
        .join("settings.json")
}

/// UTC civil time from seconds since the epoch (Howard Hinnant's days-from-
/// civil inverse), so the model needs no date library.
fn utc_parts(time: SystemTime) -> (i64, u32, u32, u32, u32, u32) {
    let secs = time
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (
        year,
        month,
        day,
        (rem / 3600) as u32,
        (rem % 3600 / 60) as u32,
        (rem % 60) as u32,
    )
}

/// RFC 3339 in UTC with seconds: `2026-11-04T02:42:18Z`.
pub fn rfc3339_utc(time: SystemTime) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(time);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

/// `20261104T024218Z`, for the unreadable History file's name.
pub fn compact_utc(time: SystemTime) -> String {
    let (y, mo, d, h, mi, s) = utc_parts(time);
    format!("{y:04}{mo:02}{d:02}T{h:02}{mi:02}{s:02}Z")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn utc_stamps() {
        let at = UNIX_EPOCH + Duration::from_secs(1_793_760_138);
        assert_eq!(rfc3339_utc(at), "2026-11-04T02:42:18Z");
        assert_eq!(compact_utc(at), "20261104T024218Z");
        assert_eq!(rfc3339_utc(UNIX_EPOCH), "1970-01-01T00:00:00Z");
        let leap = UNIX_EPOCH + Duration::from_secs(951_782_400);
        assert_eq!(rfc3339_utc(leap), "2000-02-29T00:00:00Z");
    }

    #[test]
    fn the_app_and_the_core_share_one_version() {
        assert_eq!(env!("CARGO_PKG_VERSION"), markitai_app::version());
    }
}
