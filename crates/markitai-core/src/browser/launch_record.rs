//! A successful diagnostic launch is remembered for seven days, for the same
//! browser executable and the same markitai, as Python markitai remembers it:
//! a launch costs seconds on every doctor run where Chromium starts slowly
//! (Windows on Arm under x64 emulation, a new executable antivirus scans), and
//! a browser that launched does not stop launching by itself. A record that
//! cannot be read or written only means the next doctor launches again.

use serde_json::{Value, json};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// How long a remembered launch stands for a new one.
const VALID: Duration = Duration::from_secs(7 * 86_400);
/// A record is a short JSON object; anything larger is not one.
const MAX_RECORD: u64 = 4096;

pub(super) fn path() -> PathBuf {
    crate::config::home().join("doctor-chromium-launch.json")
}

/// What a remembered launch holds for: this executable (path, size and
/// modification time, so an update or a replacement launches again) and this
/// markitai (version and program), whose launch options it tested.
fn identity(executable: &Path) -> Option<String> {
    let metadata = std::fs::metadata(executable).ok()?;
    let modified = metadata
        .modified()
        .ok()?
        .duration_since(UNIX_EPOCH)
        .ok()?
        .as_nanos();
    let program = std::env::current_exe().ok()?;
    Some(format!(
        "{}|{}|{modified}|{}|{}",
        executable.display(),
        metadata.len(),
        crate::VERSION,
        program.display()
    ))
}

/// Whether `record` holds a launch of `executable` less than seven days
/// before `now`.
pub(super) fn remembered(record: &Path, executable: &Path, now: SystemTime) -> bool {
    let Some(identity) = identity(executable) else {
        return false;
    };
    let Some(saved) = crate::platform::open_read(record, false)
        .ok()
        .and_then(|file| {
            crate::platform::read_limited(file, MAX_RECORD)
                .ok()
                .flatten()
        })
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
    else {
        return false;
    };
    let Some(launched) = saved["launched_at"].as_u64() else {
        return false;
    };
    let Ok(now) = now.duration_since(UNIX_EPOCH) else {
        return false;
    };
    saved["identity"].as_str() == Some(identity.as_str())
        && now
            .checked_sub(Duration::from_secs(launched))
            .is_some_and(|age| age < VALID)
}

/// Record a successful launch of `executable` at `now` in `record`.
pub(super) fn remember(record: &Path, executable: &Path, now: SystemTime) {
    let (Some(identity), Ok(now)) = (identity(executable), now.duration_since(UNIX_EPOCH)) else {
        return;
    };
    let Some(directory) = record.parent() else {
        return;
    };
    let bytes = json!({"identity": identity, "launched_at": now.as_secs()}).to_string();
    let _ = (|| -> std::io::Result<()> {
        crate::platform::private_directory()
            .recursive(true)
            .create(directory)?;
        let mut temporary = tempfile::NamedTempFile::new_in(directory)?;
        std::io::Write::write_all(&mut temporary, bytes.as_bytes())?;
        crate::platform::persist(temporary, record).map_err(|error| error.error)?;
        Ok(())
    })();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_launch_is_remembered_for_seven_days_for_the_same_executable_only() {
        let directory = tempfile::tempdir().unwrap();
        let record = directory.path().join("home/doctor-chromium-launch.json");
        let browser = directory.path().join("chrome");
        let other = directory.path().join("chromium");
        std::fs::write(&browser, b"browser").unwrap();
        std::fs::write(&other, b"browser").unwrap();
        let launched = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
        assert!(!remembered(&record, &browser, launched));
        remember(&record, &browser, launched);
        let day = Duration::from_secs(86_400);
        assert!(remembered(&record, &browser, launched));
        assert!(remembered(
            &record,
            &browser,
            launched + 7 * day - Duration::from_secs(1)
        ));
        assert!(!remembered(&record, &browser, launched + 7 * day));
        // A record from the future, another executable or a changed one is no launch.
        assert!(!remembered(&record, &browser, launched - day));
        assert!(!remembered(&record, &other, launched));
        std::fs::write(&browser, b"updated browser").unwrap();
        assert!(!remembered(&record, &browser, launched));
        remember(&record, &browser, launched);
        assert!(remembered(&record, &browser, launched));
        for damaged in [&b"{"[..], b"[]", br#"{"identity":1,"launched_at":"x"}"#] {
            std::fs::write(&record, damaged).unwrap();
            assert!(!remembered(&record, &browser, launched));
        }
        std::fs::write(&record, vec![b' '; MAX_RECORD as usize + 1]).unwrap();
        assert!(!remembered(&record, &browser, launched));
        // A missing executable is never remembered, nor recorded.
        let missing = directory.path().join("missing");
        remember(&record, &missing, launched);
        assert!(!remembered(&record, &missing, launched));
    }

    #[cfg(unix)]
    #[test]
    fn a_linked_record_is_not_followed() {
        let directory = tempfile::tempdir().unwrap();
        let browser = directory.path().join("chrome");
        std::fs::write(&browser, b"browser").unwrap();
        let elsewhere = directory.path().join("elsewhere.json");
        let now = UNIX_EPOCH + Duration::from_secs(1_800_000_000);
        remember(&elsewhere, &browser, now);
        let record = directory.path().join("doctor-chromium-launch.json");
        std::os::unix::fs::symlink(&elsewhere, &record).unwrap();
        assert!(!remembered(&record, &browser, now));
    }
}
