//! New releases: the daily check and its notice, `markitai update`'s
//! install-method detection, and the in-place update of a setup-script
//! install (shared by the CLI and `markitai serve`).
//!
//! The check runs at most once a day (backing off from one hour to a day
//! after failures), on a background thread, with a two-second cap, and keeps
//! what it learnt in `MARKITAI_HOME/update-check.json`. It never runs and the
//! notice never shows for JSON, stdout output, `--quiet`, a stderr that is not
//! a terminal, CI, MCP, or a source build; `update.check: false` and
//! `MARKITAI_NO_UPDATE_CHECK=1` turn both off.
pub(crate) mod check;
pub(crate) mod control;
pub(crate) mod install;
pub(crate) mod method;
pub(crate) mod notes;
pub(crate) mod version;

use crate::app::i18n::{Lang, text};
use crate::term::{Mark, Style, Tone};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use version::Version;

/// The cache file in `MARKITAI_HOME`.
pub(crate) const CACHE: &str = "update-check.json";
const FORMAT: u32 = 1;
const HOUR: i64 = 3600;
const DAY: i64 = 24 * HOUR;
/// How long a run waits for the check at its end, counted from its start.
const CAP: Duration = Duration::from_secs(2);

/// This build's version and whether it counts as a source build (local
/// changes, or a debug build: releases and `cargo install` are release
/// builds). A debug build may pose as a release (`MARKITAI_TEST_UPDATE_AS=2.1.0`)
/// so the tests can exercise what a release does; a release build ignores
/// the variable.
pub(crate) fn current_identity() -> (String, bool) {
    if cfg!(debug_assertions) {
        return match std::env::var("MARKITAI_TEST_UPDATE_AS")
            .ok()
            .filter(|value| !value.is_empty())
        {
            Some(posed) => (posed, false),
            None => (markitai_core::VERSION.to_owned(), true),
        };
    }
    (
        markitai_core::VERSION.to_owned(),
        markitai_core::build_info::dirty(),
    )
}

/// This build's version as a release version; `None` for a source build
/// (a pre-release version or local changes), which is never checked.
pub(crate) fn current_release() -> Option<Version> {
    let (version, dirty) = current_identity();
    Version::parse(&version).filter(|version| !version.is_prerelease() && !dirty)
}

fn truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Whether checking is on: `update.check` in the configuration (default
/// on) and no `MARKITAI_NO_UPDATE_CHECK`.
pub(crate) fn enabled(cfg: Option<&Value>) -> bool {
    let env = markitai_core::config::environment();
    if env
        .get("MARKITAI_NO_UPDATE_CHECK")
        .is_some_and(|value| truthy(value))
    {
        return false;
    }
    cfg.is_none_or(|cfg| cfg["update"]["check"] != Value::Bool(false))
}

/// Whether this looks like a CI job, where nobody reads a notice.
pub(crate) fn ci() -> bool {
    let env = markitai_core::config::environment();
    let set = |name: &str| {
        env.get(name)
            .is_some_and(|value| !value.is_empty() && !matches!(value.as_str(), "0" | "false"))
    };
    [
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "BUILDKITE",
        "TF_BUILD",
        "JENKINS_URL",
        "TEAMCITY_VERSION",
    ]
    .iter()
    .any(|name| set(name))
}

fn now() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs() as i64)
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Notified {
    pub(crate) version: String,
    pub(crate) at: i64,
}

/// What the checks learnt, in `MARKITAI_HOME/update-check.json`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct Cache {
    pub(crate) format: u32,
    /// When the last check succeeded (Unix seconds).
    pub(crate) checked_at: Option<i64>,
    pub(crate) latest: Option<String>,
    /// The latest release's one-line description.
    pub(crate) lead: Option<String>,
    /// Failed checks in a row, and when the next may run.
    pub(crate) failures: u32,
    pub(crate) retry_at: Option<i64>,
    /// The last notice shown: once a day per version.
    pub(crate) notified: Option<Notified>,
}

impl Cache {
    pub(crate) fn path(home: &Path) -> PathBuf {
        home.join(CACHE)
    }

    /// The cache, or an empty one when there is none or it cannot be read.
    /// One a newer markitai wrote is left alone (`None`).
    pub(crate) fn load(home: &Path) -> Option<Self> {
        let path = Self::path(home);
        let Ok(file) = markitai_core::platform::open_read(&path, false) else {
            return Some(Self::default());
        };
        let Ok(Some(bytes)) = markitai_core::platform::read_limited(file, 64 * 1024) else {
            return Some(Self::default());
        };
        let value: Value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
        if value["format"]
            .as_u64()
            .is_some_and(|found| found > u64::from(FORMAT))
        {
            return None;
        }
        Some(serde_json::from_value(value).unwrap_or_default())
    }

    pub(crate) fn save(&self, home: &Path) -> std::io::Result<()> {
        markitai_core::output::create_private_dir(home)?;
        let mut saved = self.clone();
        saved.format = FORMAT;
        let mut temporary = tempfile::NamedTempFile::new_in(home)?;
        serde_json::to_writer_pretty(&mut temporary, &saved).map_err(std::io::Error::other)?;
        markitai_core::platform::persist(temporary, &Self::path(home))
            .map(drop)
            .map_err(|error| error.error)
    }

    /// Whether a check may run now: a day after the last success, and not
    /// before the backoff after a failure ends.
    pub(crate) fn due(&self, now: i64) -> bool {
        self.retry_at.is_none_or(|at| now >= at)
            && self.checked_at.is_none_or(|at| now - at >= DAY || now < at)
    }

    pub(crate) fn succeeded(&mut self, now: i64, latest: &Version, lead: Option<String>) {
        if self.latest.as_deref() != Some(&latest.to_string()) || lead.is_some() {
            self.lead = lead;
        }
        self.latest = Some(latest.to_string());
        self.checked_at = Some(now);
        self.failures = 0;
        self.retry_at = None;
    }

    /// One hour after the first failure, doubling up to a day.
    pub(crate) fn failed(&mut self, now: i64) {
        self.failures = self.failures.saturating_add(1);
        let wait = (HOUR << (self.failures - 1).min(5)).min(DAY);
        self.retry_at = Some(now + wait);
    }

    /// The newer release this cache knows of.
    pub(crate) fn newer(&self, current: &Version) -> Option<Version> {
        self.latest
            .as_deref()
            .and_then(Version::parse)
            .filter(|latest| latest > current && !latest.is_prerelease())
    }

    /// Whether the notice for `latest` may show now: once a day per version.
    pub(crate) fn notice_due(&self, latest: &Version, now: i64) -> bool {
        self.notified.as_ref().is_none_or(|notified| {
            notified.version != latest.to_string() || now - notified.at >= DAY || now < notified.at
        })
    }
}

/// The release base of this install: its receipt's, else the default
/// (`MARKITAI_DOWNLOAD_URL` overrides both).
pub(crate) fn base_for(method: &method::Method) -> Result<check::Base, String> {
    let recorded = match method {
        method::Method::Setup(install) => install.receipt.get("base").map(str::to_owned),
        _ => None,
    };
    check::Base::resolve(recorded.as_deref())
}

/// Ask for the latest release and record the answer (and on a new version
/// its description). Returns the cache as saved.
pub(crate) fn refresh(home: &Path, base: &check::Base, timeouts: check::Timeouts) -> Option<Cache> {
    let mut cache = Cache::load(home)?;
    let now = now();
    match check::latest(base, timeouts) {
        Ok(latest) => {
            let known =
                cache.latest.as_deref() == Some(&latest.to_string()) && cache.lead.is_some();
            let lead = (!known)
                .then(|| check::lead(base, &latest, timeouts))
                .flatten();
            cache.succeeded(now, &latest, lead);
        }
        Err(_) => cache.failed(now),
    }
    let _ = cache.save(home);
    Some(cache)
}

/// A check started for a run, finished (or abandoned) at its end.
pub(crate) struct Background {
    started: Instant,
    done: Option<mpsc::Receiver<()>>,
    current: Option<Version>,
}

impl Background {
    pub(crate) fn none() -> Self {
        Self {
            started: Instant::now(),
            done: None,
            current: None,
        }
    }

    /// Start the check when this run may show the notice and a check is due.
    pub(crate) fn start(shown: bool, cfg: Option<&Value>) -> Self {
        let current = current_release();
        if !shown || !enabled(cfg) || ci() || current.is_none() {
            return Self::none();
        }
        let home = markitai_core::config::home();
        let due = Cache::load(&home).is_some_and(|cache| cache.due(now()));
        let mut background = Self {
            current,
            ..Self::none()
        };
        if due {
            let detected = method::detect(&method::Facts::current());
            if let Ok(base) = base_for(&detected) {
                let (sender, receiver) = mpsc::channel();
                let spawned = std::thread::Builder::new()
                    .name("markitai-update-check".into())
                    .spawn(move || {
                        refresh(&home, &base, check::NOTICE);
                        let _ = sender.send(());
                    });
                if spawned.is_ok() {
                    background.done = Some(receiver);
                }
            }
        }
        background
    }

    /// At the run's end: wait for the check up to the cap, then the line to
    /// show (once a day per version), recorded as shown.
    pub(crate) fn finish(self, lang: Lang, style: Style) -> Option<String> {
        let current = self.current?;
        // A check still running after the cap is abandoned (the process
        // ends); what an earlier check learnt still counts.
        if let Some(done) = self.done {
            let _ = done.recv_timeout(CAP.saturating_sub(self.started.elapsed()));
        }
        take_notice(&current, lang, style)
    }
}

/// The L1 line when the cache knows a newer release and it was not shown in
/// the last day; marks it shown.
pub(crate) fn take_notice(current: &Version, lang: Lang, style: Style) -> Option<String> {
    let home = markitai_core::config::home();
    let mut cache = Cache::load(&home)?;
    let latest = cache.newer(current)?;
    let now = now();
    if !cache.notice_due(&latest, now) {
        return None;
    }
    cache.notified = Some(Notified {
        version: latest.to_string(),
        at: now,
    });
    let _ = cache.save(&home);
    Some(notice_line(
        lang,
        style,
        0,
        &latest,
        current,
        &command_for_notice(),
    ))
}

/// The update command the notice names: this install's, else `markitai update`
/// (which then explains what to run).
pub(crate) fn command_for_notice() -> String {
    method::detect(&method::Facts::current())
        .command()
        .unwrap_or_else(|| "markitai update".into())
}

/// The newer release the cache knows of, for `serve` and `doctor` (no
/// request, no record that it was shown).
pub(crate) fn cached_newer(cfg: Option<&Value>) -> Option<(Version, Version)> {
    let current = current_release()?;
    if !enabled(cfg) {
        return None;
    }
    let cache = Cache::load(&markitai_core::config::home())?;
    Some((cache.newer(&current)?, current))
}

/// The plain sentence of the notice, without the command.
pub(crate) fn notice_text(lang: Lang, latest: &Version, current: &Version) -> String {
    text!(lang =>
        "New version {latest} (current {current}):",
        "有新版本 {latest}（当前 {current}）：")
}

/// `• 有新版本 2.2.0（当前 2.1.0）：markitai update`, the command in the
/// accent colour; one line, never wrapped.
pub(crate) fn notice_line(
    lang: Lang,
    style: Style,
    indent: usize,
    latest: &Version,
    current: &Version,
    command: &str,
) -> String {
    let separator = if lang == Lang::En { " " } else { "" };
    format!(
        "{}{} {}{separator}{}",
        " ".repeat(indent),
        style.mark(Mark::Info),
        notice_text(lang, latest, current),
        style.paint(Tone::Accent, command)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(text: &str) -> Version {
        Version::parse(text).unwrap()
    }

    #[test]
    fn checks_run_daily_and_back_off_from_an_hour_to_a_day() {
        let mut cache = Cache::default();
        assert!(cache.due(1_000_000));
        cache.succeeded(1_000_000, &v("2.2.0"), Some("Lead.".into()));
        assert!(!cache.due(1_000_000 + DAY - 1));
        assert!(cache.due(1_000_000 + DAY));
        // A clock set back does not stop checks forever.
        assert!(cache.due(1_000_000 - 10));
        let mut waits = Vec::new();
        let start = 5_000_000;
        for _ in 0..8 {
            cache.failed(start);
            waits.push(cache.retry_at.unwrap() - start);
        }
        assert_eq!(
            waits,
            [HOUR, 2 * HOUR, 4 * HOUR, 8 * HOUR, 16 * HOUR, DAY, DAY, DAY]
        );
        assert!(!cache.due(start + DAY - 1));
        cache.succeeded(start, &v("2.2.0"), None);
        assert_eq!((cache.failures, cache.retry_at), (0, None));
        // The same release keeps its description when none was fetched again.
        assert_eq!(cache.lead.as_deref(), Some("Lead."));
        cache.succeeded(start, &v("2.3.0"), None);
        assert_eq!(cache.lead, None);
    }

    #[test]
    fn the_notice_shows_once_a_day_per_version_and_only_for_a_newer_release() {
        let mut cache = Cache {
            latest: Some("2.2.0".into()),
            ..Cache::default()
        };
        let current = v("2.1.0");
        assert_eq!(cache.newer(&current), Some(v("2.2.0")));
        assert_eq!(cache.newer(&v("2.2.0")), None);
        assert_eq!(cache.newer(&v("2.3.0")), None);
        let latest = v("2.2.0");
        assert!(cache.notice_due(&latest, 100));
        cache.notified = Some(Notified {
            version: "2.2.0".into(),
            at: 100,
        });
        assert!(!cache.notice_due(&latest, 100 + DAY - 1));
        assert!(cache.notice_due(&latest, 100 + DAY));
        // A newer version is news at once.
        assert!(cache.notice_due(&v("2.3.0"), 101));
        // A pre-release is never offered.
        cache.latest = Some("2.3.0-rc.1".into());
        assert_eq!(cache.newer(&current), None);
    }

    #[test]
    fn the_cache_round_trips_and_a_newer_format_is_left_alone() {
        let home = tempfile::tempdir().unwrap();
        let home = home.path().join("mh");
        assert_eq!(Cache::load(&home), Some(Cache::default()));
        let mut cache = Cache::default();
        cache.succeeded(42, &v("2.2.0"), Some("Lead.".into()));
        cache.save(&home).unwrap();
        let loaded = Cache::load(&home).unwrap();
        assert_eq!(loaded.latest.as_deref(), Some("2.2.0"));
        assert_eq!(loaded.format, FORMAT);
        std::fs::write(Cache::path(&home), r#"{"format":2,"latest":"9.0.0"}"#).unwrap();
        assert_eq!(Cache::load(&home), None);
        // A damaged cache reads as empty.
        std::fs::write(Cache::path(&home), "{not json").unwrap();
        assert_eq!(Cache::load(&home), Some(Cache::default()));
    }

    #[test]
    fn the_notice_line_is_l1_in_both_languages() {
        let plain = Style::PLAIN;
        assert_eq!(
            notice_line(
                Lang::Zh,
                plain,
                0,
                &v("2.2.0"),
                &v("2.1.0"),
                "markitai update"
            ),
            "• 有新版本 2.2.0（当前 2.1.0）：markitai update"
        );
        assert_eq!(
            notice_line(
                Lang::En,
                plain,
                2,
                &v("2.2.0"),
                &v("2.1.0"),
                "markitai update"
            ),
            "  • New version 2.2.0 (current 2.1.0): markitai update"
        );
        let coloured = Style {
            color: true,
            ..plain
        };
        assert!(
            notice_line(
                Lang::En,
                coloured,
                0,
                &v("2.2.0"),
                &v("2.1.0"),
                "markitai update"
            )
            .ends_with("\u{1b}[36mmarkitai update\u{1b}[0m")
        );
    }

    #[test]
    fn switches_and_ci_turn_the_check_off() {
        assert!(enabled(None));
        assert!(enabled(Some(
            &serde_json::json!({"update": {"check": true}})
        )));
        assert!(!enabled(Some(
            &serde_json::json!({"update": {"check": false}})
        )));
        for value in ["1", "true", "YES", "on"] {
            assert!(truthy(value));
        }
        for value in ["", "0", "false", "no"] {
            assert!(!truthy(value));
        }
    }
}
