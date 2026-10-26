//! Updating the server from the workbench (contract v1.24), and restarting
//! it for `markitai update`.
//!
//! Only a server that listens on loopback has the update routes and sends
//! `Capabilities.update`; an exposed one has neither (its terminal still
//! names a newer release). Only the admin may use them (the server token,
//! or a direct loopback client of a `--no-auth` server), never through a
//! forwarding proxy. The page may only ask for the release the server
//! itself found (`POST /api/update {"version"}`): it cannot pick another.
//!
//! The order keeps the pause short: download and verify while conversions
//! run; then hold new conversions (503 `updating`), let the running and
//! queued ones finish; then swap the files and restart (see `restart.rs`).
//! Until installing begins `DELETE /api/update` cancels and nothing changes.
//!
//! `markitai update` restarts a running server through
//! `/api/update/control` and `/api/update/restart`, authorised by the
//! per-launch secret in `MARKITAI_HOME/serve/control` (`update::control`).
use super::State;
use super::types::{ApiError, ApiResult};
use crate::update::check::{self, Base};
use crate::update::control::{self, Control, Pending};
use crate::update::install;
use crate::update::method::{self, Method};
use crate::update::version::Version;
use crate::update::{self as updates, Cache};
use axum::Json;
use axum::extract::{ConnectInfo, Request, State as ExtractState};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

api_routes! {
    get "/api/update" => status;
    post "/api/update" => start;
    delete "/api/update" => cancel;
}

/// The control routes, which every server has (see the module docs).
pub(in crate::server) mod control_routes {
    api_routes! {
        get "/api/update/control" => super::control_status;
        post "/api/update/restart" => super::control_restart;
    }
}

/// Paths the guard lets through without the access token: they carry the
/// control secret instead.
pub(super) fn control_path(path: &str) -> bool {
    matches!(path, "/api/update/control" | "/api/update/restart")
}

/// The web update's admin-only paths.
pub(super) fn update_path(path: &str) -> bool {
    path == "/api/update"
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Phase {
    Idle,
    Downloading,
    Verifying,
    Waiting,
    Installing,
    Restarting,
    Failed,
}

impl Phase {
    fn name(self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Downloading => "downloading",
            Self::Verifying => "verifying",
            Self::Waiting => "waiting",
            Self::Installing => "installing",
            Self::Restarting => "restarting",
            Self::Failed => "failed",
        }
    }

    fn busy(self) -> bool {
        !matches!(self, Self::Idle | Self::Failed)
    }

    /// New conversions are held from waiting on.
    fn holding(self) -> bool {
        matches!(self, Self::Waiting | Self::Installing | Self::Restarting)
    }

    fn cancellable(self) -> bool {
        matches!(self, Self::Downloading | Self::Verifying | Self::Waiting)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Failure {
    pub(super) code: &'static str,
    pub(super) reason: String,
    pub(super) rolled_back: bool,
}

struct Progress {
    phase: Phase,
    target: Option<Version>,
    done: u64,
    total: Option<u64>,
    error: Option<Failure>,
    /// Who asked: the page, or `markitai update` (a restart only).
    by_cli: bool,
}

/// Why and into what the server restarts; read by `restart::relaunch`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Restart {
    pub(crate) exe: PathBuf,
    /// The version restarting into, when known (`markitai update` says).
    pub(crate) to: Option<String>,
}

pub(crate) struct Update {
    /// The web update routes exist (a loopback server, checks on, a release build).
    pub(super) web: bool,
    pub(super) current: Option<Version>,
    pub(super) method: Method,
    pub(super) base: Option<Base>,
    pub(super) home: PathBuf,
    /// The program to start again: this install's path, recorded at startup
    /// (a replaced program's own path names the new one).
    pub(super) exe: PathBuf,
    pub(super) secret: String,
    /// This start came from a restart into a new version (`from`).
    pub(super) updated_from: Option<String>,
    /// An update the last start undid.
    pub(super) undone: Option<control::Failed>,
    progress: Mutex<Progress>,
    cancel: AtomicBool,
    /// Requests admitted that may still start conversions.
    admitting: AtomicUsize,
    pub(super) restart: tokio::sync::watch::Sender<Option<Restart>>,
    worker: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Default for Update {
    fn default() -> Self {
        Self {
            web: false,
            current: None,
            method: Method::Source,
            base: None,
            home: PathBuf::new(),
            exe: PathBuf::new(),
            secret: String::new(),
            updated_from: None,
            undone: None,
            progress: Mutex::new(Progress {
                phase: Phase::Idle,
                target: None,
                done: 0,
                total: None,
                error: None,
                by_cli: false,
            }),
            cancel: AtomicBool::new(false),
            admitting: AtomicUsize::new(0),
            restart: tokio::sync::watch::channel(None).0,
            worker: Mutex::new(None),
        }
    }
}

impl Update {
    /// The update state of a server starting with `cfg`; `web` when it
    /// listens on loopback only.
    pub(super) fn new(cfg: &Value, loopback: bool, secret: String) -> Self {
        let current = updates::current_release();
        let detected = method::detect(&method::Facts::current());
        // A folder swap a killed update left half done is undone first.
        if let Method::Setup(install) = &detected
            && let Ok(_exclusive) = install::exclusive(&markitai_core::config::home())
        {
            install::recover(&install.dir);
        }
        let exe = match &detected {
            Method::Setup(install) => install.exe.clone(),
            _ => std::env::current_exe().unwrap_or_default(),
        };
        let base = updates::base_for(&detected).ok();
        Self {
            web: loopback && current.is_some() && updates::enabled(Some(cfg)),
            current,
            method: detected,
            base,
            home: markitai_core::config::home(),
            exe,
            secret,
            ..Self::default()
        }
    }

    fn phase(&self) -> Phase {
        self.lock().phase
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Progress> {
        self.progress
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }

    /// The newer release the cache knows of, and its description (English,
    /// Chinese).
    fn latest(&self) -> Option<(Version, Option<String>, Option<String>)> {
        let current = self.current.as_ref()?;
        let cache = Cache::load(&self.home)?;
        let latest = cache.newer(current)?;
        Some((latest, cache.lead, cache.lead_zh))
    }

    fn can_apply(&self) -> bool {
        self.method.in_place() && self.latest().is_some() && !self.phase().busy()
    }

    /// `GET /api/update` and `Capabilities.update`.
    fn view(&self, waiting: u64) -> Value {
        let latest = self.latest();
        let progress = self.lock();
        let notes_url = latest
            .as_ref()
            .and_then(|(version, ..)| self.base.as_ref().map(|base| base.notes_url(version)));
        let error = progress.error.as_ref().map(|error| {
            json!({"code": error.code, "reason": error.reason, "rolled_back": error.rolled_back})
        });
        let undone = self.undone.as_ref().map(|failed| {
            json!({"code": "start_failed", "reason": failed.reason, "rolled_back": true, "version": failed.to})
        });
        json!({
            "state": progress.phase.name(),
            "current": self.current.as_ref().map(ToString::to_string),
            "latest": latest.as_ref().map(|(version, ..)| version.to_string()),
            "newer": latest.is_some(),
            "notes_url": notes_url,
            "lead": latest.as_ref().and_then(|(_, lead, _)| lead.clone()),
            "lead_zh": latest.as_ref().and_then(|(.., zh)| zh.clone()),
            "method": self.method.key(),
            "command": self.method.command(),
            "can_apply": self.method.in_place() && latest.is_some() && !progress.phase.busy(),
            "target": progress.target.as_ref().map(ToString::to_string),
            "progress": (progress.phase == Phase::Downloading).then(|| json!({"done": progress.done, "total": progress.total})),
            "waiting_jobs": waiting,
            "error": error.or(if progress.phase == Phase::Idle { undone } else { None }),
            "updated_from": self.updated_from,
            "by": progress.phase.busy().then_some(if progress.by_cli { "cli" } else { "web" }),
        })
    }

    fn set(&self, phase: Phase) {
        self.lock().phase = phase;
    }

    fn fail(&self, failure: Failure) {
        let mut progress = self.lock();
        progress.phase = Phase::Failed;
        progress.error = Some(failure);
    }
}

/// Conversions running or queued, item by item.
pub(super) fn active_conversions(state: &State) -> u64 {
    state
        .jobs
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .values()
        .map(|job| job.active.load(Ordering::SeqCst) as u64)
        .sum()
}

/// A request that may start conversions, counted while it runs so a drain
/// cannot miss one that was admitted just before holding began.
pub(super) struct Admission<'a>(&'a AtomicUsize);

impl Drop for Admission<'_> {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Admit a request that starts conversions, or refuse it (503 `updating`,
/// `Retry-After`) while an update holds them. Counted first, checked second;
/// the drain sets holding first and counts second, so neither misses the other.
pub(super) fn admit(state: &State) -> ApiResult<Admission<'_>> {
    let admitting = &state.update.admitting;
    admitting.fetch_add(1, Ordering::SeqCst);
    let admission = Admission(admitting);
    if state.update.phase().holding() {
        return Err(ApiError::new(
            503,
            "updating",
            "the server is updating; new conversions start after it restarts",
        ));
    }
    Ok(admission)
}

/// Whether nothing runs, waits or is being admitted.
fn drained(state: &State) -> u64 {
    active_conversions(state) + state.update.admitting.load(Ordering::SeqCst) as u64
}

/// What `Capabilities.update` says to the admin of a loopback server.
pub(super) fn capability(state: &State, admin: bool) -> Option<Value> {
    let update = &state.update;
    if !update.web || !admin {
        return None;
    }
    let view = update.view(active_conversions(state));
    Some(json!({
        "current": view["current"],
        "latest": view["latest"],
        "newer": view["newer"],
        "can_apply": view["can_apply"],
        "method": view["method"],
        "state": view["state"],
    }))
}

fn forwarded(request: &Request) -> bool {
    ["forwarded", "x-forwarded-for", "x-real-ip"]
        .iter()
        .any(|name| request.headers().contains_key(*name))
}

fn loopback_peer(request: &Request) -> bool {
    !forwarded(request)
        && request
            .extensions()
            .get::<ConnectInfo<std::net::SocketAddr>>()
            .is_some_and(|info| info.0.ip().to_canonical().is_loopback())
}

fn forbidden() -> ApiError {
    ApiError::new(
        403,
        "update_forbidden",
        "updating the server needs the admin token on a direct loopback connection",
    )
}

/// The request names this computer (`localhost` or a loopback address): one
/// that came through a proxy under another name does not.
fn loopback_host(request: &Request) -> bool {
    let Some(host) = request
        .headers()
        .get("host")
        .and_then(|value| value.to_str().ok())
    else {
        return false;
    };
    let name = match host.rsplit_once(':') {
        Some((name, port)) if port.bytes().all(|byte| byte.is_ascii_digit()) => name,
        _ => host,
    };
    let name = name.trim_start_matches('[').trim_end_matches(']');
    name.eq_ignore_ascii_case("localhost")
        || name
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.to_canonical().is_loopback())
}

/// The admin, directly: the guard has already refused user tokens.
fn admin(request: &Request) -> ApiResult<()> {
    let admin = request.extensions().get::<super::security::Principal>()
        == Some(&super::security::Principal::Admin);
    if admin && loopback_peer(request) && loopback_host(request) {
        Ok(())
    } else {
        Err(forbidden())
    }
}

fn no_store(value: Value) -> Response {
    let mut response = Json(value).into_response();
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        axum::http::HeaderValue::from_static("no-store"),
    );
    response
}

/// `GET /api/update`.
pub(super) async fn status(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<Response> {
    admin(&request)?;
    Ok(no_store(state.update.view(active_conversions(&state))))
}

/// `POST /api/update {"version": "<the latest the server knows>"}`.
pub(super) async fn start(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<Response> {
    admin(&request)?;
    let bytes = axum::body::to_bytes(request.into_body(), 1024)
        .await
        .map_err(|_| ApiError::new(413, "request_too_large", "request exceeds 1 KiB"))?;
    let body: Value = serde_json::from_slice(&bytes)
        .map_err(|_| ApiError::new(422, "invalid_json", "the body is {\"version\": \"X.Y.Z\"}"))?;
    let asked = body
        .as_object()
        .filter(|fields| fields.len() == 1)
        .and_then(|fields| fields.get("version"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ApiError::new(422, "invalid_json", "the body is {\"version\": \"X.Y.Z\"}")
        })?;
    let update = &state.update;
    if update.phase().busy() {
        return Err(ApiError::new(
            409,
            "update_running",
            "an update is already running",
        ));
    }
    let Some((latest, ..)) = update.latest() else {
        return Err(ApiError::new(
            409,
            "version_changed",
            "no newer release is known",
        ));
    };
    if Version::parse(asked).as_ref() != Some(&latest) {
        return Err(ApiError::new(
            409,
            "version_changed",
            format!("the latest release is {latest}; reload the page"),
        ));
    }
    if !update.can_apply() {
        return Err(ApiError::new(
            409,
            "update_unavailable",
            "this installation is updated with its own command",
        ));
    }
    {
        let mut progress = update.lock();
        if progress.phase.busy() {
            return Err(ApiError::new(
                409,
                "update_running",
                "an update is already running",
            ));
        }
        *progress = Progress {
            phase: Phase::Downloading,
            target: Some(latest.clone()),
            done: 0,
            total: None,
            error: None,
            by_cli: false,
        };
    }
    update.cancel.store(false, Ordering::SeqCst);
    let worker_state = state.clone();
    let handle = std::thread::Builder::new()
        .name("markitai-update".into())
        .spawn(move || run(&worker_state, latest))
        .map_err(ApiError::internal)?;
    *update
        .worker
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some(handle);
    Ok((
        axum::http::StatusCode::ACCEPTED,
        no_store(update.view(active_conversions(&state))),
    )
        .into_response())
}

/// `DELETE /api/update`: cancel until installing begins.
pub(super) async fn cancel(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<Response> {
    admin(&request)?;
    let phase = state.update.phase();
    // A restart `markitai update` asked for has its files swapped already.
    if !phase.cancellable() || state.update.lock().by_cli {
        return Err(ApiError::new(
            409,
            "not_cancellable",
            if phase.busy() {
                "the update is installing and can no longer be cancelled"
            } else {
                "no update is running"
            },
        ));
    }
    state.update.cancel.store(true, Ordering::SeqCst);
    Ok(no_store(state.update.view(active_conversions(&state))))
}

fn say(english: String, chinese: String, mark: crate::term::Mark) {
    let lang = crate::app::i18n::lang();
    crate::app::logging::shaped(
        crate::app::logging::Shape::item(mark),
        &english,
        (lang == crate::app::i18n::Lang::Zh).then_some(chinese.as_str()),
    );
}

/// The web update, on its own thread.
fn run(state: &Arc<State>, target: Version) {
    use crate::term::Mark;
    let update = &state.update;
    let cancelled = || update.cancel.load(Ordering::SeqCst) || state.closing.load(Ordering::SeqCst);
    say(
        format!("Updating to {target} (from the web workbench)"),
        format!("更新到 {target}（网页发起）"),
        Mark::Action,
    );
    let outcome = (|| -> Result<(), install::Error> {
        let Method::Setup(installed) = &update.method else {
            return Err(install::Error::Archive("not a setup install".into()));
        };
        // `markitai update` may have replaced the release since this server started.
        let installed = &installed.reread();
        let base = update
            .base
            .clone()
            .ok_or_else(|| install::Error::Archive("no release base".into()))?;
        let _exclusive = install::exclusive(&update.home)?;
        install::remove_leftovers(installed, &[]);
        let work = install::prepare(installed)?;
        let release = install::resolve(&base, &target, &installed.target(), check::COMMAND)?;
        let downloaded = install::download(
            &work,
            &release,
            check::COMMAND,
            &mut |done, total| {
                let mut progress = update.lock();
                progress.done = done;
                progress.total = total;
            },
            &cancelled,
        )?;
        update.set(Phase::Verifying);
        let unpacked = install::unpack(&work, &downloaded)?;
        let program = install::program(&unpacked, installed.script)?;
        // The new program loads this configuration and opens this history
        // read-only, without binding, before anything is replaced.
        let mut arguments: Vec<std::ffi::OsString> = std::env::args_os().skip(1).collect();
        arguments.push("--preflight".into());
        install::dry_start(&program, &arguments, None)?;
        if cancelled() {
            return Err(install::Error::Cancelled);
        }
        say(
            format!("Downloaded and verified {target}"),
            format!("已下载并校验 {target}"),
            Mark::Success,
        );
        update.set(Phase::Waiting);
        let mut told = None;
        loop {
            if cancelled() {
                return Err(install::Error::Cancelled);
            }
            let waiting = drained(state);
            if waiting == 0 {
                break;
            }
            if told != Some(waiting) {
                say(
                    format!(
                        "Waiting for {waiting} conversion{} to finish",
                        if waiting == 1 { "" } else { "s" }
                    ),
                    format!("等待 {waiting} 个转换完成"),
                    Mark::Info,
                );
                told = Some(waiting);
            }
            std::thread::sleep(Duration::from_millis(250));
        }
        update.set(Phase::Installing);
        let swapped = install::swap(installed, &release, &unpacked)?;
        drop(work);
        let from = update
            .current
            .as_ref()
            .map(ToString::to_string)
            .unwrap_or_default();
        let pending = Pending::new(&from, &target.to_string(), update.exe.clone(), swapped);
        if let Err(error) = pending.save(&update.home) {
            let _ = pending.swapped.rollback();
            return Err(install::Error::Io {
                what: "Can't record the update".into(),
                error,
                rolled_back: true,
            });
        }
        let shown = update.exe.display();
        say(
            format!("Installed: {shown}"),
            format!("已安装：{shown}"),
            Mark::Success,
        );
        Ok(())
    })();
    match outcome {
        Ok(()) => {
            update.set(Phase::Restarting);
            update.restart.send_replace(Some(Restart {
                exe: update.exe.clone(),
                to: Some(target.to_string()),
            }));
        }
        Err(error) => {
            let current = update
                .current
                .as_ref()
                .map(ToString::to_string)
                .unwrap_or_default();
            if matches!(error, install::Error::Cancelled) {
                say("Update cancelled".into(), "已取消更新".into(), Mark::Skip);
            } else {
                let reason = error.reason();
                say(
                    format!("Error: {reason}"),
                    crate::app::update::reason_zh(&error),
                    Mark::Error,
                );
                if error.rolled_back() {
                    say(
                        format!("Rolled back; still {current}"),
                        format!("已恢复，仍是 {current}"),
                        Mark::Info,
                    );
                }
            }
            update.fail(Failure {
                code: error.code(),
                reason: error.reason(),
                rolled_back: error.rolled_back(),
            });
        }
    }
}

/// Wait for the update thread (shutdown: it stops at its next step).
pub(super) fn join_worker(state: &State) {
    state.update.cancel.store(true, Ordering::SeqCst);
    let handle = state
        .update
        .worker
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .take();
    if let Some(handle) = handle {
        let _ = handle.join();
    }
}

fn control_secret(request: &Request) -> Option<&str> {
    request
        .headers()
        .get(control::HEADER)
        .and_then(|value| value.to_str().ok())
}

fn control_allowed(state: &State, request: &Request) -> ApiResult<()> {
    let secret = control_secret(request).unwrap_or_default();
    if !state.update.secret.is_empty()
        && loopback_peer(request)
        && super::security::equal_secret(secret.as_bytes(), state.update.secret.as_bytes())
    {
        Ok(())
    } else {
        Err(ApiError::new(
            403,
            "control_forbidden",
            "this route needs the control secret on a direct loopback connection",
        ))
    }
}

/// `GET /api/update/control`: the version and the work a restart waits for.
pub(super) async fn control_status(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<Response> {
    control_allowed(&state, &request)?;
    let (version, _) = updates::current_identity();
    Ok(no_store(json!({
        "version": version,
        "pid": std::process::id(),
        "waiting_jobs": active_conversions(&state),
        "state": state.update.phase().name(),
        "undone": state.update.undone,
    })))
}

/// `POST /api/update/restart`: drain, then start the installed program
/// again; asked by `markitai update` after it replaced the files.
pub(super) async fn control_restart(
    ExtractState(state): ExtractState<Arc<State>>,
    request: Request,
) -> ApiResult<Response> {
    control_allowed(&state, &request)?;
    let update = &state.update;
    {
        let mut progress = update.lock();
        if progress.phase.busy() {
            return Err(ApiError::new(
                409,
                "update_running",
                "an update is already running",
            ));
        }
        let to = Pending::load(&update.home).map(|pending| pending.to);
        *progress = Progress {
            phase: Phase::Waiting,
            target: to.as_deref().and_then(Version::parse),
            done: 0,
            total: None,
            error: None,
            by_cli: true,
        };
    }
    update.cancel.store(false, Ordering::SeqCst);
    let worker_state = state.clone();
    let handle = std::thread::Builder::new()
        .name("markitai-restart".into())
        .spawn(move || drain_and_restart(&worker_state))
        .map_err(ApiError::internal)?;
    *update
        .worker
        .lock()
        .unwrap_or_else(|error| error.into_inner()) = Some(handle);
    Ok((
        axum::http::StatusCode::ACCEPTED,
        no_store(json!({"waiting_jobs": active_conversions(&state)})),
    )
        .into_response())
}

fn drain_and_restart(state: &Arc<State>) {
    use crate::term::Mark;
    let update = &state.update;
    let to = Pending::load(&update.home).map(|pending| pending.to);
    match &to {
        Some(to) => say(
            format!("Restarting to use {to} (asked by markitai update)"),
            format!("重启以使用 {to}（markitai update 发起）"),
            Mark::Action,
        ),
        None => say(
            "Restarting (asked by markitai update)".into(),
            "重启（markitai update 发起）".into(),
            Mark::Action,
        ),
    }
    let mut told = None;
    loop {
        if state.closing.load(Ordering::SeqCst) {
            return;
        }
        let waiting = drained(state);
        if waiting == 0 {
            break;
        }
        if told != Some(waiting) {
            say(
                format!(
                    "Waiting for {waiting} conversion{} to finish",
                    if waiting == 1 { "" } else { "s" }
                ),
                format!("等待 {waiting} 个转换完成"),
                Mark::Info,
            );
            told = Some(waiting);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    update.set(Phase::Restarting);
    update.restart.send_replace(Some(Restart {
        exe: update.exe.clone(),
        to,
    }));
}

/// The daily check while the server runs: every hour it asks whether a
/// check is due (once a day, with the backoff after failures).
pub(super) fn start_checks(state: &Arc<State>, cfg: &Value) {
    if state.update.current.is_none() || !updates::enabled(Some(cfg)) || updates::ci() {
        return;
    }
    let Some(base) = state.update.base.clone() else {
        return;
    };
    let home = state.update.home.clone();
    let mut shutdown = state.shutdown.subscribe();
    let task = crate::task::spawn(async move {
        loop {
            let due = Cache::load(&home).is_some_and(|cache| {
                cache.due(
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map_or(0, |elapsed| elapsed.as_secs() as i64),
                )
            });
            if due {
                let (home, base) = (home.clone(), base.clone());
                let _ = crate::task::blocking(move || {
                    updates::refresh(&home, &base, check::COMMAND);
                })
                .await;
            }
            tokio::select! {
                changed = shutdown.changed() => if changed.is_err() || *shutdown.borrow() { break },
                _ = tokio::time::sleep(Duration::from_secs(3600)) => {}
            }
        }
    });
    state.tasks.lock().unwrap().push(task);
}

/// Write the control file once the server listens.
pub(super) fn publish_control(state: &State, address: std::net::SocketAddr) {
    let control =
        Control::new(address, state.update.secret.clone()).with_exe(state.update.exe.clone());
    if let Err(error) = control.write(&state.update.home) {
        eprintln!("Warning: Serve: cannot write the control file: {error}");
    }
}

pub(super) fn withdraw_control(state: &State) {
    if let Some(found) = Control::read(&state.update.home)
        && found.secret == state.update.secret
    {
        found.remove(&state.update.home);
    }
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;

    impl Update {
        /// A loopback server of `current` on a setup install under `home`,
        /// whose cache knows `latest`.
        pub(in crate::server) fn for_test(home: &std::path::Path, latest: Option<&str>) -> Self {
            let mut cache = Cache::default();
            if let Some(latest) = latest {
                cache.latest = Some(latest.into());
                cache.lead = Some("Faster tables.".into());
            }
            cache.save(home).unwrap();
            let install = method::SetupInstall {
                script: method::Script::Shell,
                exe: home.join("bin/markitai"),
                dir: home.join("bin"),
                receipt_dir: home.join("share/markitai/current"),
                receipt: method::Receipt::parse("version=2.1.0\n"),
            };
            Self {
                web: true,
                current: Version::parse("2.1.0"),
                method: Method::Setup(install),
                base: Base::parse("https://github.com/Ynewtime/markitai/releases").ok(),
                home: home.to_owned(),
                exe: home.join("bin/markitai"),
                secret: "c".repeat(64),
                ..Self::default()
            }
        }
    }

    use axum::body::{Body, to_bytes};
    use std::collections::{HashMap, HashSet};
    use tokio::sync::{Semaphore, watch};
    use tower::ServiceExt;

    fn state(root: &std::path::Path, token: Option<&str>, update: Update) -> Arc<State> {
        markitai_core::output::create_private_dir(root).unwrap();
        let cfg = markitai_core::config::normalize(&json!({"llm":{"enabled":false}})).unwrap();
        Arc::new(State {
            settings: super::super::settings::Store::new(
                cfg,
                super::super::SettingsSource {
                    path: root.join("config.json"),
                    origin: "default".into(),
                    overrides: None,
                },
            )
            .unwrap(),
            root: root.into(),
            jobs: Mutex::new(HashMap::new()),
            file_slots: Arc::new(Semaphore::new(1)),
            url_slots: Arc::new(Semaphore::new(1)),
            closing: AtomicBool::new(false),
            persistence_failed: AtomicBool::new(false),
            shutdown: watch::channel(false).0,
            tasks: Mutex::new(Vec::new()),
            token: std::sync::RwLock::new(token.map(str::to_owned)),
            token_generation: watch::channel(0).0,
            token_from_environment: false,
            address: Default::default(),
            allowed_hosts: HashSet::new(),
            tickets: Default::default(),
            previews: Default::default(),
            access: super::super::access::Access::load(root.join("access.json")).unwrap(),
            users: super::super::settings::Users::new(root.join("user-models")),
            update,
        })
    }

    fn request(
        method: &str,
        path: &str,
        peer: &str,
        headers: &[(&str, &str)],
        body: &str,
    ) -> axum::http::Request<Body> {
        let mut builder = axum::http::Request::builder()
            .method(method)
            .uri(path)
            .header("host", "127.0.0.1:3600");
        for (name, value) in headers {
            builder = builder.header(*name, *value);
        }
        let mut request = builder.body(Body::from(body.to_owned())).unwrap();
        request
            .extensions_mut()
            .insert(ConnectInfo(peer.parse::<std::net::SocketAddr>().unwrap()));
        request
    }

    async fn answer(
        state: &Arc<State>,
        request: axum::http::Request<Body>,
    ) -> (u16, Value, axum::http::HeaderMap) {
        let response = super::super::router(state).oneshot(request).await.unwrap();
        let status = response.status().as_u16();
        let headers = response.headers().clone();
        let bytes = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        (
            status,
            serde_json::from_slice(&bytes).unwrap_or(Value::Null),
            headers,
        )
    }

    const ADMIN: &str = "a-admin-token-0000000000000000000000000000000000000000000000000000";
    const LOCAL: &str = "127.0.0.1:50000";

    #[tokio::test]
    async fn an_exposed_server_has_no_update_route_and_no_update_capability() {
        let root = tempfile::tempdir().unwrap();
        let exposed = Update {
            web: false,
            ..Update::for_test(root.path(), Some("2.2.0"))
        };
        let state = state(root.path(), Some(ADMIN), exposed);
        let bearer = format!("Bearer {ADMIN}");
        let auth = [("authorization", bearer.as_str())];
        for (method, body) in [
            ("GET", ""),
            ("POST", r#"{"version":"2.2.0"}"#),
            ("DELETE", ""),
        ] {
            let (status, value, _) =
                answer(&state, request(method, "/api/update", LOCAL, &auth, body)).await;
            assert_eq!(status, 404, "{method}: {value}");
            assert_eq!(value["reason"], "route_not_found");
        }
        let (status, capabilities, _) = answer(
            &state,
            request("GET", "/api/capabilities", LOCAL, &auth, ""),
        )
        .await;
        assert_eq!(status, 200);
        assert!(capabilities.get("update").is_none(), "{capabilities}");
    }

    #[test]
    fn only_a_host_naming_this_computer_counts_as_loopback() {
        let named = |host: Option<&str>| {
            let mut builder = axum::http::Request::builder().uri("/api/update");
            if let Some(host) = host {
                builder = builder.header("host", host);
            }
            loopback_host(&builder.body(Body::empty()).unwrap())
        };
        for host in [
            "localhost",
            "LocalHost:3600",
            "127.0.0.1",
            "127.0.0.1:3600",
            "127.8.9.1:1",
            "[::1]",
            "[::1]:3600",
            "[::ffff:127.0.0.1]:3600",
        ] {
            assert!(named(Some(host)), "{host}");
        }
        // A proxy's public name, this computer's network address, a lookalike.
        for host in [
            "md.example.com",
            "md.example.com:443",
            "localhost.example.com",
            "127.0.0.1.example.com",
            "192.168.1.5:3600",
            "[2001:db8::1]:3600",
            "0.0.0.0:3600",
            "localhost:port",
            "",
        ] {
            assert!(!named(Some(host)), "{host}");
        }
        assert!(!named(None));
    }

    #[tokio::test]
    async fn the_admin_reaching_a_loopback_server_under_another_name_is_refused() {
        let root = tempfile::tempdir().unwrap();
        let mut state = state(
            root.path(),
            Some(ADMIN),
            Update::for_test(root.path(), Some("2.2.0")),
        );
        // Let through by the server's own Host check, as --allowed-host does
        // (which also turns the web update off; this is the check behind it).
        let allowed = super::super::security::allowed_host("md.example.com")
            .map_err(|error| error.detail)
            .unwrap();
        Arc::get_mut(&mut state)
            .unwrap()
            .allowed_hosts
            .insert(allowed);
        let bearer = format!("Bearer {ADMIN}");
        for (host, expected) in [("localhost:3600", 200), ("md.example.com", 403)] {
            let mut request = request(
                "GET",
                "/api/update",
                LOCAL,
                &[("authorization", &bearer)],
                "",
            );
            request
                .headers_mut()
                .insert("host", axum::http::HeaderValue::from_static(host));
            let (status, value, _) = answer(&state, request).await;
            assert_eq!(status, expected, "{host}: {value}");
            if expected == 403 {
                assert_eq!(value["reason"], "update_forbidden", "{value}");
            }
        }
    }

    #[tokio::test]
    async fn a_loopback_server_tells_its_admin_alone_and_only_directly() {
        let root = tempfile::tempdir().unwrap();
        let state = state(
            root.path(),
            Some(ADMIN),
            Update::for_test(root.path(), Some("2.2.0")),
        );
        let bearer = format!("Bearer {ADMIN}");
        let (status, capabilities, _) = answer(
            &state,
            request(
                "GET",
                "/api/capabilities",
                LOCAL,
                &[("authorization", &bearer)],
                "",
            ),
        )
        .await;
        assert_eq!(status, 200);
        assert_eq!(capabilities["update"]["latest"], "2.2.0");
        assert_eq!(capabilities["update"]["can_apply"], true);
        let (status, view, headers) = answer(
            &state,
            request(
                "GET",
                "/api/update",
                LOCAL,
                &[("authorization", &bearer)],
                "",
            ),
        )
        .await;
        assert_eq!((status, view["state"].as_str()), (200, Some("idle")));
        assert_eq!(headers.get("cache-control").unwrap(), "no-store");
        // Without the token, through a proxy, or from a user token: refused.
        let (status, _, _) = answer(&state, request("GET", "/api/update", LOCAL, &[], "")).await;
        assert_eq!(status, 401);
        let (status, value, _) = answer(
            &state,
            request(
                "GET",
                "/api/update",
                LOCAL,
                &[
                    ("authorization", &bearer),
                    ("x-forwarded-for", "203.0.113.9"),
                ],
                "",
            ),
        )
        .await;
        assert_eq!(
            (status, value["reason"].as_str()),
            (403, Some("update_forbidden"))
        );
        let created = state.access.create("phone".into()).unwrap();
        let user = format!("Bearer {}", created["token"].as_str().unwrap());
        let (status, _, _) = answer(
            &state,
            request("GET", "/api/update", LOCAL, &[("authorization", &user)], ""),
        )
        .await;
        assert_eq!(status, 403);
        let (_, capabilities, _) = answer(
            &state,
            request(
                "GET",
                "/api/capabilities",
                LOCAL,
                &[("authorization", &user)],
                "",
            ),
        )
        .await;
        assert!(capabilities.get("update").is_none());
    }

    #[tokio::test]
    async fn the_page_may_only_ask_for_the_release_the_server_found() {
        let root = tempfile::tempdir().unwrap();
        let state = state(
            root.path(),
            None,
            Update::for_test(root.path(), Some("2.2.0")),
        );
        // --no-auth: a direct loopback client is the admin.
        for (body, status, reason) in [
            (r#"{"version":"2.0.1"}"#, 409, "version_changed"),
            (r#"{"version":"2.2.0","extra":1}"#, 422, "invalid_json"),
            ("not json", 422, "invalid_json"),
        ] {
            let (got, value, _) =
                answer(&state, request("POST", "/api/update", LOCAL, &[], body)).await;
            assert_eq!(
                (got, value["reason"].as_str()),
                (status, Some(reason)),
                "{body}"
            );
        }
        // A remote --no-auth client is anonymous: no update.
        let (status, _, _) = answer(
            &state,
            request(
                "POST",
                "/api/update",
                "203.0.113.7:1",
                &[],
                r#"{"version":"2.2.0"}"#,
            ),
        )
        .await;
        assert_eq!(status, 403);
        // An install markitai does not replace itself: its command only.
        let root = tempfile::tempdir().unwrap();
        let uv = state_with(root.path(), Method::UvTool);
        let (status, value, _) = answer(
            &uv,
            request("POST", "/api/update", LOCAL, &[], r#"{"version":"2.2.0"}"#),
        )
        .await;
        assert_eq!(
            (status, value["reason"].as_str()),
            (409, Some("update_unavailable"))
        );
        // Nothing is running: nothing to cancel.
        let (status, value, _) =
            answer(&uv, request("DELETE", "/api/update", LOCAL, &[], "")).await;
        assert_eq!(
            (status, value["reason"].as_str()),
            (409, Some("not_cancellable"))
        );
    }

    fn state_with(root: &std::path::Path, method: Method) -> Arc<State> {
        state(
            root,
            None,
            Update {
                method,
                ..Update::for_test(root, Some("2.2.0"))
            },
        )
    }

    #[tokio::test]
    async fn new_conversions_are_held_with_retry_after_while_an_update_waits() {
        let root = tempfile::tempdir().unwrap();
        let state = state(
            root.path(),
            None,
            Update::for_test(root.path(), Some("2.2.0")),
        );
        state.update.set(Phase::Waiting);
        let (status, value, headers) = answer(
            &state,
            request(
                "POST",
                "/api/jobs",
                LOCAL,
                &[("content-type", "application/x-www-form-urlencoded")],
                "urls=https%3A%2F%2Fexample.com%2F",
            ),
        )
        .await;
        assert_eq!((status, value["reason"].as_str()), (503, Some("updating")));
        assert_eq!(headers.get("retry-after").unwrap(), "5");
        assert_eq!(state.update.admitting.load(Ordering::SeqCst), 0);
        let (status, view, _) = answer(&state, request("GET", "/api/update", LOCAL, &[], "")).await;
        assert_eq!(
            (status, view["state"].as_str(), view["can_apply"].as_bool()),
            (200, Some("waiting"), Some(false))
        );
        // A request admitted before holding began is waited for.
        state.update.set(Phase::Idle);
        let admission = admit(&state).unwrap();
        assert_eq!(drained(&state), 1);
        drop(admission);
        assert_eq!(drained(&state), 0);
    }

    #[tokio::test]
    async fn the_control_routes_take_the_secret_from_loopback_instead_of_the_token() {
        let root = tempfile::tempdir().unwrap();
        // Exposed or not, the control routes are there.
        let state = state(
            root.path(),
            Some(ADMIN),
            Update {
                web: false,
                ..Update::for_test(root.path(), None)
            },
        );
        let secret = "c".repeat(64);
        let (status, value, _) = answer(
            &state,
            request(
                "GET",
                "/api/update/control",
                LOCAL,
                &[(control::HEADER, &secret)],
                "",
            ),
        )
        .await;
        assert_eq!(status, 200, "{value}");
        assert_eq!(value["waiting_jobs"], 0);
        assert_eq!(value["pid"], std::process::id());
        for (peer, headers) in [
            (LOCAL, vec![(control::HEADER, "wrong")]),
            (LOCAL, vec![]),
            ("203.0.113.5:1", vec![(control::HEADER, secret.as_str())]),
            (
                LOCAL,
                vec![
                    (control::HEADER, secret.as_str()),
                    ("forwarded", "for=203.0.113.5"),
                ],
            ),
        ] {
            let (status, value, _) = answer(
                &state,
                request("GET", "/api/update/control", peer, &headers, ""),
            )
            .await;
            assert_eq!(
                (status, value["reason"].as_str()),
                (403, Some("control_forbidden")),
                "{peer} {headers:?}"
            );
            let (status, _, _) = answer(
                &state,
                request("POST", "/api/update/restart", peer, &headers, ""),
            )
            .await;
            assert_eq!(status, 403);
        }
        // The admin token is no control secret either.
        let bearer = format!("Bearer {ADMIN}");
        let (status, _, _) = answer(
            &state,
            request(
                "POST",
                "/api/update/restart",
                LOCAL,
                &[("authorization", &bearer)],
                "",
            ),
        )
        .await;
        assert_eq!(status, 403);
        // With the secret: a restart once nothing runs.
        let mut restarting = state.update.restart.subscribe();
        let (status, _, _) = answer(
            &state,
            request(
                "POST",
                "/api/update/restart",
                LOCAL,
                &[(control::HEADER, &secret)],
                "",
            ),
        )
        .await;
        assert_eq!(status, 202);
        tokio::time::timeout(Duration::from_secs(5), restarting.wait_for(Option::is_some))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(state.update.phase(), Phase::Restarting);
        join_worker(&state);
    }

    #[test]
    fn phases_say_when_work_is_held_and_what_can_be_cancelled() {
        assert!(!Phase::Downloading.holding() && Phase::Downloading.cancellable());
        assert!(Phase::Waiting.holding() && Phase::Waiting.cancellable());
        assert!(Phase::Installing.holding() && !Phase::Installing.cancellable());
        assert!(Phase::Restarting.holding() && !Phase::Restarting.cancellable());
        assert!(!Phase::Failed.busy() && !Phase::Idle.busy());
    }

    #[test]
    fn the_view_names_the_release_the_command_and_whether_it_can_apply() {
        let home = tempfile::tempdir().unwrap();
        let update = Update::for_test(home.path(), Some("2.2.0"));
        let view = update.view(2);
        assert_eq!(view["state"], "idle");
        assert_eq!(view["latest"], "2.2.0");
        assert_eq!(view["newer"], true);
        assert_eq!(view["can_apply"], true);
        assert_eq!(view["command"], "markitai update");
        assert_eq!(view["waiting_jobs"], 2);
        assert_eq!(
            view["notes_url"],
            "https://github.com/Ynewtime/markitai/releases/tag/v2.2.0"
        );
        assert_eq!(view["lead"], "Faster tables.");
        assert_eq!(view["lead_zh"], Value::Null);
        let other = Update {
            method: Method::UvTool,
            ..Update::for_test(home.path(), Some("2.2.0"))
        };
        let view = other.view(0);
        assert_eq!(
            (view["can_apply"].clone(), view["command"].clone()),
            (json!(false), json!("uv tool upgrade markitai"))
        );
        let latest = Update::for_test(home.path(), Some("2.1.0"));
        assert_eq!(latest.view(0)["newer"], false);
        assert_eq!(latest.view(0)["can_apply"], false);
    }
}
