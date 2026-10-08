//! Whether URLs may be sent to remote extraction services.
//!
//! A remote strategy that is selected (`-s jina`, or `fetch.strategy` set to
//! `defuddle`, `jina` or `cloudflare`) follows `fetch.remote_consent`
//! directly: `never` refuses it, `always` lets it run and `ask` lets it run
//! when it was chosen for this run, and asks otherwise.
//!
//! The `auto` chain is stricter. The contract's default for
//! `fetch.remote_consent` is `always`, the reference's value, and the
//! configuration a conversion receives cannot tell that default from a value
//! the user wrote. `auto` therefore stays local unless the host says the user
//! wrote `always` ([`RemoteFallback::explicitly_always`]), or the value is
//! `ask` and the person at the terminal answers yes. One answer serves the
//! whole process, and a disclosure for `always` is shown once per
//! `MARKITAI_HOME`.
//!
//! The same once-per-home store (empty marker files under
//! `MARKITAI_HOME/notices`) carries the other privacy notices: before a
//! selected remote strategy first sends a URL to its service, and before
//! images first go to a model that is not on this machine.

use crate::{Error, Result};
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

/// The question asked once before `auto` first sends a URL to a remote service.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsentRequest {
    /// The page local fetching could not read, without credentials or secret
    /// query values.
    pub url: String,
    /// The services the run may try, in order: `defuddle`, `jina`, `cloudflare`.
    pub services: Vec<&'static str>,
}

/// A one-line notice for the host to show.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteNotice {
    /// `fetch.remote_consent` is `always`: public URLs that local fetching
    /// cannot read go to these services. Shown once per `MARKITAI_HOME`.
    Disclosure { services: Vec<&'static str> },
    /// `fetch.remote_consent` is `ask` and nobody could be asked, so remote
    /// services were skipped. Shown once per process.
    NotAsked,
    /// A selected remote strategy (`-s` or `fetch.strategy`) is about to send
    /// a URL to this service. Shown once per `MARKITAI_HOME` and service.
    Strategy { service: &'static str },
    /// Images (page renders, screenshots or pictures) are about to be sent to
    /// a model that is not on this machine. `models` names, as configured,
    /// every model off this machine the request's group can route images to.
    /// Shown once per `MARKITAI_HOME`.
    Images { models: Vec<String> },
}

type Ask = Box<dyn Fn(&ConsentRequest) -> bool + Send + Sync>;
type Notify = Box<dyn Fn(&RemoteNotice) -> bool + Send + Sync>;

/// What a host knows about the user's own configuration that the effective
/// configuration it passes cannot say, because defaults are filled in.
/// Without one (the bindings, `serve`, `mcp`), `auto` never sends a URL to a
/// remote service and never starts with the browser for a
/// `fetch.fallback_patterns` domain.
pub struct RemoteFallback {
    /// The user's own configuration (file or overrides) sets
    /// `fetch.remote_consent` to `always`; the default of the same value is
    /// not an opt-in.
    pub explicitly_always: bool,
    /// The user's own configuration sets `fetch.fallback_patterns`; the
    /// contract's default list is not applied.
    pub explicit_fallback_patterns: bool,
    /// Asks the person at the terminal; `None` when nobody can be asked.
    pub ask: Option<Ask>,
    /// Shows a notice and says whether it was shown (a quiet run may not).
    pub notify: Notify,
}

/// Install the host's consent facts for this process. The first call wins;
/// it returns whether this call installed them.
pub fn set_remote_fallback(host: RemoteFallback) -> bool {
    INSTALLED
        .set(Gate::new(
            Some(host),
            Some(crate::config::home().join("notices")),
        ))
        .is_ok()
}

static INSTALLED: OnceLock<Gate> = OnceLock::new();
static ABSENT: Gate = Gate {
    host: None,
    decision: Mutex::new(None),
    shown: Mutex::new(Vec::new()),
    hinted: AtomicBool::new(false),
    notices: None,
};

/// The consent value a configuration states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Consent {
    Never,
    Ask,
    Always,
}

pub(crate) fn configured(cfg: &Value) -> Consent {
    match cfg.pointer("/fetch/remote_consent").and_then(Value::as_str) {
        Some("never") => Consent::Never,
        Some("ask") => Consent::Ask,
        _ => Consent::Always,
    }
}

/// `MARKITAI_NO_REMOTE_FETCH`, the hard opt-out that also refuses a selected
/// remote strategy.
pub(crate) fn hard_off(vars: &HashMap<String, String>) -> bool {
    vars.get("MARKITAI_NO_REMOTE_FETCH")
        .is_some_and(|value| crate::config::env_opt_out(value))
}

/// The refusal every consent failure starts with; the web interface
/// recognizes it.
pub(crate) const DISABLED: &str = "Remote fetching is disabled by policy";

/// The process-wide decision and the host that can make it.
pub(crate) struct Gate {
    host: Option<RemoteFallback>,
    decision: Mutex<Option<bool>>,
    /// Keys of the once-per-home notices this process has shown (or is
    /// showing).
    shown: Mutex<Vec<String>>,
    hinted: AtomicBool,
    /// Where the once-per-home notice markers live.
    notices: Option<PathBuf>,
}

impl Gate {
    pub(crate) fn new(host: Option<RemoteFallback>, notices: Option<PathBuf>) -> Self {
        Self {
            host,
            decision: Mutex::new(None),
            shown: Mutex::new(Vec::new()),
            hinted: AtomicBool::new(false),
            notices,
        }
    }

    /// The host's gate, or one that never opts in.
    pub(crate) fn installed() -> &'static Gate {
        INSTALLED.get().unwrap_or(&ABSENT)
    }

    fn explicitly_always(&self) -> bool {
        self.host
            .as_ref()
            .is_some_and(|host| host.explicitly_always)
    }

    /// Whether `fetch.fallback_patterns` was written by the user, so `auto`
    /// starts with the browser for those domains.
    pub(crate) fn fallback_patterns_configured(&self) -> bool {
        self.host
            .as_ref()
            .is_some_and(|host| host.explicit_fallback_patterns)
    }

    /// Whether `auto` may use remote services, when that is known without
    /// asking; `None` when it would have to ask.
    pub(crate) fn peek(&self, cfg: &Value, vars: &HashMap<String, String>) -> Option<bool> {
        if hard_off(vars) {
            return Some(false);
        }
        match configured(cfg) {
            Consent::Never => Some(false),
            Consent::Always => Some(self.explicitly_always()),
            Consent::Ask => *self.decision.lock().unwrap_or_else(PoisonError::into_inner),
        }
    }

    /// Whether `auto` may now send the URL to remote services. Under `ask`
    /// this asks once for the whole process (other conversions wait for the
    /// answer); under an explicit `always` it shows the disclosure once.
    pub(crate) fn fallback(
        &self,
        cfg: &Value,
        vars: &HashMap<String, String>,
        request: impl FnOnce() -> ConsentRequest,
    ) -> bool {
        if hard_off(vars) {
            return false;
        }
        match configured(cfg) {
            Consent::Never => false,
            Consent::Always if self.explicitly_always() => {
                self.disclose(request().services);
                true
            }
            Consent::Always => false,
            Consent::Ask => self.decide(request),
        }
    }

    /// Permission for a selected remote strategy. `chosen` is true when this
    /// run named the strategy itself (`-s`), which answers `ask`.
    pub(crate) fn selected(
        &self,
        cfg: &Value,
        vars: &HashMap<String, String>,
        chosen: bool,
        request: impl FnOnce() -> ConsentRequest,
    ) -> Result<()> {
        if hard_off(vars) {
            return Err(Error::Fetch(format!(
                "{DISABLED} (MARKITAI_NO_REMOTE_FETCH)"
            )));
        }
        match configured(cfg) {
            Consent::Never => Err(Error::Fetch(format!(
                "{DISABLED} (--no-remote-fetch or fetch.remote_consent=never)"
            ))),
            Consent::Always => Ok(()),
            Consent::Ask if chosen || self.decide(request) => Ok(()),
            Consent::Ask => Err(Error::Fetch(format!(
                "{DISABLED}: fetch.remote_consent=ask was not answered yes; select the strategy with -s for this run, or set fetch.remote_consent=always"
            ))),
        }
    }

    fn decide(&self, request: impl FnOnce() -> ConsentRequest) -> bool {
        let mut decision = self.decision.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(decision) = *decision {
            return decision;
        }
        let answer = match self.host.as_ref().and_then(|host| host.ask.as_ref()) {
            Some(ask) => ask(&request()),
            None => {
                self.hint();
                false
            }
        };
        *decision = Some(answer);
        answer
    }

    fn hint(&self) {
        if !self.hinted.swap(true, Ordering::SeqCst)
            && let Some(host) = &self.host
        {
            (host.notify)(&RemoteNotice::NotAsked);
        }
    }

    /// The disclosure for `always`.
    fn disclose(&self, services: Vec<&'static str>) {
        self.once("remote-fetch", || RemoteNotice::Disclosure { services });
    }

    /// The notice before a selected remote strategy first sends a URL to
    /// `service`.
    pub(crate) fn strategy(&self, service: &'static str) {
        self.once(&format!("remote-strategy-{service}"), || {
            RemoteNotice::Strategy { service }
        });
    }

    /// The notice before images first go to a model off this machine;
    /// `models` are all the models off this machine that may receive them.
    pub(crate) fn images(&self, models: &[&str]) {
        self.once("remote-images", || RemoteNotice::Images {
            models: models.iter().map(|model| (*model).to_owned()).collect(),
        });
    }

    /// Show a notice once per process and, through a marker file named
    /// `key` that holds nothing, once per `MARKITAI_HOME`. A notice the host
    /// did not show (a quiet run) stays due for a later run.
    fn once(&self, key: &str, notice: impl FnOnce() -> RemoteNotice) {
        let Some(host) = &self.host else {
            return;
        };
        {
            let mut shown = self.shown.lock().unwrap_or_else(PoisonError::into_inner);
            if shown.iter().any(|seen| seen == key) {
                return;
            }
            shown.push(key.to_owned());
        }
        let marker = self.notices.as_ref().map(|dir| dir.join(key));
        if marker.as_ref().is_some_and(|marker| marker.exists()) {
            return;
        }
        if !(host.notify)(&notice()) {
            self.shown
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .retain(|seen| seen != key);
            return;
        }
        if let Some(marker) = marker {
            // An unwritable home only means the notice comes again next run.
            let _ = record(&marker);
        }
    }
}

fn record(marker: &std::path::Path) -> std::io::Result<()> {
    if let Some(parent) = marker.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(marker) {
        Ok(_) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => Ok(()),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    fn cfg(consent: &str) -> Value {
        json!({"fetch": {"remote_consent": consent}})
    }

    fn request() -> ConsentRequest {
        ConsentRequest {
            url: "https://example.com/a".into(),
            services: vec!["defuddle", "jina"],
        }
    }

    struct Recorder {
        asked: Arc<AtomicUsize>,
        notices: Arc<Mutex<Vec<RemoteNotice>>>,
    }

    fn host(
        explicitly_always: bool,
        answer: Option<bool>,
        shown: bool,
    ) -> (RemoteFallback, Recorder) {
        let asked = Arc::new(AtomicUsize::new(0));
        let notices = Arc::new(Mutex::new(Vec::new()));
        let (count, seen) = (Arc::clone(&asked), Arc::clone(&notices));
        let ask: Option<Ask> = answer.map(|answer| {
            Box::new(move |request: &ConsentRequest| {
                assert_eq!(request.services, ["defuddle", "jina"]);
                count.fetch_add(1, Ordering::SeqCst);
                answer
            }) as Ask
        });
        let notify: Notify = Box::new(move |notice| {
            seen.lock().unwrap().push(notice.clone());
            shown
        });
        (
            RemoteFallback {
                explicitly_always,
                explicit_fallback_patterns: false,
                ask,
                notify,
            },
            Recorder { asked, notices },
        )
    }

    #[test]
    fn the_default_always_is_not_an_opt_in_for_auto_but_lets_a_selected_strategy_run() {
        let vars = HashMap::new();
        for gate in [
            Gate::new(None, None),
            Gate::new(Some(host(false, Some(true), true).0), None),
        ] {
            assert_eq!(gate.peek(&cfg("always"), &vars), Some(false));
            assert!(!gate.fallback(&cfg("always"), &vars, request));
            assert!(gate.selected(&cfg("always"), &vars, false, request).is_ok());
        }
    }

    #[test]
    fn an_explicit_always_discloses_once_per_home_and_only_when_shown() {
        let home = tempfile::tempdir().unwrap();
        let vars = HashMap::new();
        // A quiet run does not show it, so it stays due.
        let (quiet, quiet_seen) = host(true, None, false);
        let gate = Gate::new(Some(quiet), Some(home.path().into()));
        assert!(gate.fallback(&cfg("always"), &vars, request));
        assert!(!home.path().join("remote-fetch").exists());
        assert_eq!(quiet_seen.notices.lock().unwrap().len(), 1);

        let (loud, seen) = host(true, None, true);
        let gate = Gate::new(Some(loud), Some(home.path().into()));
        assert_eq!(gate.peek(&cfg("always"), &vars), Some(true));
        assert!(gate.fallback(&cfg("always"), &vars, request));
        assert!(gate.fallback(&cfg("always"), &vars, request));
        assert_eq!(
            *seen.notices.lock().unwrap(),
            [RemoteNotice::Disclosure {
                services: vec!["defuddle", "jina"]
            }]
        );
        assert!(home.path().join("remote-fetch").is_file());
        assert_eq!(
            std::fs::read(home.path().join("remote-fetch")).unwrap(),
            b""
        );
        // The next process with the same home says nothing.
        let (again, again_seen) = host(true, None, true);
        let gate = Gate::new(Some(again), Some(home.path().into()));
        assert!(gate.fallback(&cfg("always"), &vars, request));
        assert!(again_seen.notices.lock().unwrap().is_empty());
    }

    #[test]
    fn strategy_and_image_notices_use_the_same_store_once_per_key_and_only_when_shown() {
        let home = tempfile::tempdir().unwrap();
        // A quiet run shows nothing and records nothing.
        let (quiet, quiet_seen) = host(false, None, false);
        let gate = Gate::new(Some(quiet), Some(home.path().into()));
        gate.strategy("jina");
        gate.images(&["openai/gpt-test"]);
        assert_eq!(quiet_seen.notices.lock().unwrap().len(), 2);
        assert!(std::fs::read_dir(home.path()).unwrap().next().is_none());
        // Shown: once per key, whatever the number of calls.
        let (loud, seen) = host(false, None, true);
        let gate = Gate::new(Some(loud), Some(home.path().into()));
        for _ in 0..3 {
            gate.strategy("jina");
            gate.strategy("defuddle");
            gate.images(&["openai/gpt-test"]);
            gate.images(&["gemini/another"]);
        }
        assert_eq!(
            *seen.notices.lock().unwrap(),
            [
                RemoteNotice::Strategy { service: "jina" },
                RemoteNotice::Strategy {
                    service: "defuddle"
                },
                RemoteNotice::Images {
                    models: vec!["openai/gpt-test".into()]
                },
            ]
        );
        for key in [
            "remote-strategy-jina",
            "remote-strategy-defuddle",
            "remote-images",
        ] {
            assert_eq!(std::fs::read(home.path().join(key)).unwrap(), b"", "{key}");
        }
        // The next process with the same home only hears about a new service.
        let (again, again_seen) = host(false, None, true);
        let gate = Gate::new(Some(again), Some(home.path().into()));
        gate.strategy("jina");
        gate.images(&["openai/gpt-test"]);
        gate.strategy("cloudflare");
        assert_eq!(
            *again_seen.notices.lock().unwrap(),
            [RemoteNotice::Strategy {
                service: "cloudflare"
            }]
        );
        // Without a host (bindings, serve, mcp) nothing is shown or written.
        let other = tempfile::tempdir().unwrap();
        let gate = Gate::new(None, Some(other.path().into()));
        gate.strategy("jina");
        gate.images(&["openai/gpt-test"]);
        assert!(std::fs::read_dir(other.path()).unwrap().next().is_none());
    }

    #[test]
    fn ask_asks_once_per_process_even_from_many_threads() {
        let vars = HashMap::new();
        for answer in [true, false] {
            let (fallback, recorder) = host(false, Some(answer), true);
            let gate = Arc::new(Gate::new(Some(fallback), None));
            assert_eq!(gate.peek(&cfg("ask"), &vars), None);
            let threads: Vec<_> = (0..8)
                .map(|_| {
                    let gate = Arc::clone(&gate);
                    std::thread::spawn(move || gate.fallback(&cfg("ask"), &HashMap::new(), request))
                })
                .collect();
            for thread in threads {
                assert_eq!(thread.join().unwrap(), answer);
            }
            assert_eq!(recorder.asked.load(Ordering::SeqCst), 1);
            assert_eq!(gate.peek(&cfg("ask"), &vars), Some(answer));
            // A strategy the configuration selects follows the same answer.
            assert_eq!(
                gate.selected(&cfg("ask"), &vars, false, request).is_ok(),
                answer
            );
            assert_eq!(recorder.asked.load(Ordering::SeqCst), 1);
            assert!(recorder.notices.lock().unwrap().is_empty());
        }
    }

    #[test]
    fn ask_without_a_terminal_is_never_with_one_hint() {
        let vars = HashMap::new();
        let (fallback, recorder) = host(false, None, true);
        let gate = Gate::new(Some(fallback), None);
        assert!(!gate.fallback(&cfg("ask"), &vars, request));
        assert!(!gate.fallback(&cfg("ask"), &vars, request));
        assert_eq!(*recorder.notices.lock().unwrap(), [RemoteNotice::NotAsked]);
        // Choosing the strategy for this run is the answer.
        assert!(gate.selected(&cfg("ask"), &vars, true, request).is_ok());
        let Err(Error::Fetch(message)) = gate.selected(&cfg("ask"), &vars, false, request) else {
            panic!("a configured strategy under ask needs an answer");
        };
        assert!(message.starts_with(DISABLED), "{message}");
    }

    #[test]
    fn never_and_the_hard_opt_out_refuse_everything_without_asking() {
        let (fallback, recorder) = host(true, Some(true), true);
        let gate = Gate::new(Some(fallback), None);
        let off: HashMap<String, String> =
            [("MARKITAI_NO_REMOTE_FETCH".to_owned(), "Yes".to_owned())].into();
        for (consent, vars) in [
            ("never", HashMap::new()),
            ("always", off.clone()),
            ("ask", off),
        ] {
            assert_eq!(gate.peek(&cfg(consent), &vars), Some(false));
            assert!(!gate.fallback(&cfg(consent), &vars, request));
            let Err(Error::Fetch(message)) = gate.selected(&cfg(consent), &vars, true, request)
            else {
                panic!("{consent} must refuse");
            };
            assert!(message.starts_with(DISABLED), "{message}");
        }
        assert_eq!(recorder.asked.load(Ordering::SeqCst), 0);
        assert!(recorder.notices.lock().unwrap().is_empty());
    }
}
