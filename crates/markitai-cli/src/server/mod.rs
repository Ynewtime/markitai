//! Native job service. Blocking conversion work is drained before shutdown returns.

/// Declares a module's API routes once: `ROUTES`, the (method, path) list that
/// the OpenAPI document is tested against, and `api_routes()`, the router built
/// from the same list, so a route cannot be served without being listed.
macro_rules! api_routes {
    ($($method:ident $path:literal => $handler:expr;)+) => {
        #[cfg(test)]
        pub(in crate::server) const ROUTES: &[(&str, &str)] = &[$((stringify!($method), $path)),+];
        pub(in crate::server) fn api_routes() -> axum::Router<std::sync::Arc<crate::server::State>> {
            axum::Router::new()$(.route($path, axum::routing::$method($handler)))+
        }
    };
}

mod access;
mod convert_defaults;
mod documents;
mod files;
mod http;
mod jobs;
mod launch;
mod openapi;
mod original;
mod providers;
mod rerun;
mod restart;
mod retention;
mod security;
mod settings;
mod sidecar;
mod snapshots;
mod startup;
mod store;
mod tickets;
mod transaction;
mod types;
mod update;
mod web;

pub(super) use launch::open_config;
pub(crate) use launch::settings_source;
pub(crate) use settings::SettingsSource;

use axum::{Router, middleware};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    path::PathBuf,
    sync::{
        Arc, Mutex, OnceLock, RwLock,
        atomic::{AtomicBool, Ordering},
    },
};
use tokio::sync::{Semaphore, watch};

pub struct ServeOptions {
    pub host: String,
    pub port: u16,
    pub no_open: bool,
    pub no_auth: bool,
    pub allowed_host: Vec<String>,
    /// Load the configuration and the stored state, then exit without
    /// binding: the dry start of a downloaded update.
    pub preflight: bool,
}

struct State {
    settings: settings::Store,
    root: PathBuf,
    jobs: Mutex<HashMap<String, Arc<jobs::Job>>>,
    file_slots: Arc<Semaphore>,
    url_slots: Arc<Semaphore>,
    closing: AtomicBool,
    persistence_failed: AtomicBool,
    shutdown: watch::Sender<bool>,
    tasks: Mutex<Vec<tokio::task::JoinHandle<()>>>,
    /// The access token; `None` with `--no-auth`, which never changes. The
    /// web workbench may replace it (`POST /api/access-token/regenerate`).
    token: RwLock<Option<String>>,
    /// Counts token replacements; open event streams end when it changes.
    token_generation: watch::Sender<u64>,
    /// Whether the startup token came from `MARKITAI_SERVE_TOKEN`.
    token_from_environment: bool,
    /// The address the listener bound, set once after binding.
    address: OnceLock<std::net::SocketAddr>,
    allowed_hosts: HashSet<String>,
    tickets: tickets::Tickets,
    previews: original::Previews,
    /// The user tokens (`serve/access.json`).
    access: access::Access,
    /// Each user token's own models (`serve/user-models/`).
    users: settings::Users,
    /// Updating from the workbench and restarting (`update.rs`).
    update: update::Update,
}

api_routes! {
    get "/api/capabilities" => http::capabilities;
    get "/api/openapi.json" => openapi::document;
    post "/api/download-tickets" => tickets::issue;
    post "/api/access-token/regenerate" => security::regenerate;
    get "/api/access-tokens" => access::list;
    post "/api/access-tokens" => access::create;
    delete "/api/access-tokens/{id}" => access::revoke;
    post "/api/jobs" => http::create;
    get "/api/jobs/{job_id}" => http::snapshot;
    get "/api/jobs/{job_id}/events" => http::events;
    post "/api/jobs/{job_id}/cancel" => http::stop;
    post "/api/jobs/{job_id}/items/{item_id}/retry" => rerun::retry;
    delete "/api/jobs/{job_id}/items/{item_id}" => rerun::delete;
    get "/api/jobs/{job_id}/items/{item_id}/result" => files::result;
    get "/api/jobs/{job_id}/items/{item_id}/document" => documents::document;
    get "/api/jobs/{job_id}/items/{item_id}/assets/{*relpath}" => files::asset;
    get "/api/jobs/{job_id}/items/{item_id}/original" => original::descriptor;
    get "/api/jobs/{job_id}/items/{item_id}/original/file" => original::file;
    get "/api/jobs/{job_id}/items/{item_id}/original/view" => original::view;
    get "/api/jobs/{job_id}/items/{item_id}/original/blocks" => original::blocks;
    get "/api/jobs/{job_id}/items/{item_id}/archive" => files::item_archive;
    get "/api/jobs/{job_id}/items/{item_id}/original/pages/{n}" => original::page_image;
    get "/api/jobs/{job_id}/items/{item_id}/original/pages/{n}/text" => original::page_text;
    get "/api/jobs/{job_id}/items/{item_id}/original/outline" => original::outline;
    get "/api/jobs/{job_id}/files/{*relpath}" => files::download;
    get "/api/jobs/{job_id}/archive" => files::job_archive;
    get "/api/history" => http::history;
    get "/api/history/archive" => files::history_archive;
    delete "/api/history/{job_id}" => http::delete;
    post "/api/history/delete" => http::delete_many;
}

impl State {
    /// The current access token; `None` with `--no-auth`.
    fn token(&self) -> Option<String> {
        self.token
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }
    /// The configuration a conversion of `owner` runs with: the admin's for
    /// the admin's (and anonymous) jobs, else that user's pool of its own
    /// models and the admin's shared ones, never the server's environment.
    fn conversion_config(&self, owner: Option<&str>) -> Arc<Value> {
        match owner {
            None => self.settings.snapshot(),
            Some(owner) => Arc::new(settings::conversion_config(
                &self.settings,
                &self.users,
                owner,
            )),
        }
    }
    fn auth_enabled(&self) -> bool {
        self.token
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .is_some()
    }
}

/// Every API route the service registers, as (method, path).
#[cfg(test)]
fn api_table() -> Vec<(&'static str, &'static str)> {
    [
        ROUTES,
        settings::ROUTES,
        settings::USER_ROUTES,
        providers::ROUTES,
        convert_defaults::ROUTES,
        update::ROUTES,
        update::control_routes::ROUTES,
    ]
    .concat()
}

pub(crate) fn settings_config(source: &SettingsSource) -> Result<Value, String> {
    settings::load_base(source).map_err(|error| error.detail)
}

/// The service's configuration as written, before defaults (see
/// `remote_consent::install_service`).
pub(crate) fn settings_written(source: &SettingsSource) -> Result<Value, String> {
    settings::load_written(source).map_err(|error| error.detail)
}

/// How a server run ended.
enum Outcome {
    Stopped,
    /// Start the installed program again in this one's place.
    Restart {
        restart: update::Restart,
        token: Option<String>,
        kept: restart::Kept,
    },
}

/// A failure, and whether the server had become ready before it.
struct Failure {
    ready: bool,
    message: String,
}

impl From<String> for Failure {
    fn from(message: String) -> Self {
        Self {
            ready: false,
            message,
        }
    }
}

pub(crate) fn run(cfg: Value, source: SettingsSource, options: ServeOptions) -> Result<(), String> {
    if options.preflight {
        return preflight(cfg, source);
    }
    let home = markitai_core::config::home();
    let (version, _) = crate::update::current_identity();
    // An update whose new version never got ready is undone first.
    let handed = std::env::var(crate::update::control::UPDATE_ID).ok();
    let start = crate::update::control::on_start(&home, &version, handed.as_deref());
    if let crate::update::control::Start::Crashed(pending) = &start {
        std::process::exit(restart::roll_back_and_relaunch(
            pending,
            &home,
            "it stopped before it was ready",
            None,
            restart::Kept::default(),
        ));
    }
    let pending = match start {
        crate::update::control::Start::New(pending) => Some(pending),
        _ => None,
    };
    let mut handover = Some(restart::Handover::take());
    // Handed on again if this start fails and the old version takes it.
    let handed_token = handover
        .as_ref()
        .and_then(|handover| handover.token.clone());
    // And the handed socket (Unix), held only until the first start ends.
    #[cfg(unix)]
    let mut handed_kept = restart::Kept {
        listener: handover
            .as_ref()
            .and_then(|handover| handover.listener.as_ref())
            .and_then(|listener| std::os::fd::AsFd::as_fd(listener).try_clone_to_owned().ok()),
    };
    #[cfg(not(unix))]
    let mut handed_kept = restart::Kept::default();
    // The update this run had to undo, told in its startup block.
    let mut undone: Option<String> = None;
    loop {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| e.to_string())?;
        let outcome = runtime.block_on(serve(
            cfg.clone(),
            source.clone(),
            &options,
            handover.take().unwrap_or_default(),
            pending.as_ref().filter(|_| undone.is_none()),
            undone.take(),
        ));
        drop(runtime);
        let handed = std::mem::take(&mut handed_kept);
        match outcome {
            Ok(Outcome::Stopped) => return Ok(()),
            Ok(Outcome::Restart {
                restart,
                token,
                kept,
            }) => {
                let hands_over = restart::understands_handover(restart.to.as_deref());
                match restart::relaunch(
                    &restart.exe,
                    &version,
                    token.as_deref(),
                    &kept,
                    hands_over,
                    &home,
                ) {
                    restart::Relaunched::Exited(code) => std::process::exit(code),
                    restart::Relaunched::Failed(reason) => {
                        // The files go back; this process serves again.
                        let to = restart.to.clone().unwrap_or_default();
                        if let Some(pending) = crate::update::control::Pending::load(&home)
                            .filter(|pending| Some(&pending.to) == restart.to.as_ref())
                        {
                            let _ = pending.roll_back(&home, &reason);
                        }
                        say_restart_failed(&to, &version, &reason);
                        undone = Some(to);
                        handover = Some(restart::Handover::again(kept, token));
                    }
                }
            }
            Err(Failure { ready, message }) => {
                if !ready && let Some(pending) = &pending {
                    std::process::exit(restart::roll_back_and_relaunch(
                        pending,
                        &home,
                        &message,
                        handed_token.as_deref(),
                        handed,
                    ));
                }
                return Err(message);
            }
        }
    }
}

fn say_restart_failed(to: &str, current: &str, reason: &str) {
    use crate::app::logging::{Shape, shaped};
    let zh = crate::app::i18n::lang() == crate::app::i18n::Lang::Zh;
    shaped(
        Shape::item(crate::term::Mark::Error),
        &format!("Error: {to} didn't start: {reason}"),
        zh.then(|| format!("{to} 无法启动：{reason}")).as_deref(),
    );
    shaped(
        Shape::item(crate::term::Mark::Action),
        &format!("Back on {current}"),
        zh.then(|| format!("已回到 {current}")).as_deref(),
    );
}

/// `serve --preflight`: what a start reads, read-only and without binding.
fn preflight(cfg: Value, source: SettingsSource) -> Result<(), String> {
    let serve_home = markitai_core::config::home().join("serve");
    let root = serve_home.join("jobs");
    settings::Store::new(cfg, source).map_err(|error| error.detail)?;
    access::Access::load(serve_home.join("access.json"))?;
    settings::Users::new(serve_home.join("user-models")).check_formats()?;
    if root.exists() {
        std::fs::read_dir(&root).map_err(|error| format!("{}: {error}", root.display()))?;
    }
    Ok(())
}

async fn serve(
    cfg: Value,
    source: SettingsSource,
    options: &ServeOptions,
    handover: restart::Handover,
    pending: Option<&crate::update::control::Pending>,
    undone: Option<String>,
) -> Result<Outcome, Failure> {
    // A debug build can fail here on purpose: the tests of an update whose
    // new version does not start.
    if cfg!(debug_assertions)
        && (std::env::var_os("MARKITAI_TEST_UPDATE_FAIL_START").is_some()
            || crate::update::test_pose().is_some_and(|(_, fail)| *fail))
    {
        return Err(
            "cannot open the jobs folder (MARKITAI_TEST_UPDATE_FAIL_START)"
                .to_owned()
                .into(),
        );
    }
    let serve_home = markitai_core::config::home().join("serve");
    let root = serve_home.join("jobs");
    let service_lock = store::service_lock(&root).map_err(|e| e.to_string())?;
    store::clean_uploads(&root, &service_lock).map_err(|e| e.to_string())?;
    let (shutdown, _) = watch::channel(false);
    let configured = std::env::var("MARKITAI_SERVE_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let token_from_environment = !options.no_auth
        && configured.is_some()
        && (handover.token.is_none() || handover.token == configured);
    // A restarted server keeps the token the one before it had.
    let token = if options.no_auth {
        None
    } else {
        Some(
            handover
                .token
                .clone()
                .or(configured)
                .unwrap_or_else(security::new_token),
        )
    };
    let allowed_hosts = options
        .allowed_host
        .iter()
        .map(|s| security::allowed_host(s).map_err(|e| e.detail))
        .collect::<Result<HashSet<_>, _>>()?;
    // Normalized settings always carry both limits (defaults 10 and 5, minimum 1).
    let file_count = cfg["batch"]["concurrency"].as_u64().unwrap_or(10) as usize;
    let url_count = cfg["batch"]["url_concurrency"].as_u64().unwrap_or(5) as usize;
    let access = access::Access::load(serve_home.join("access.json"))?;
    let lang = crate::app::i18n::lang();
    let listener = match handover.listener {
        Some(listener) => listener
            .set_nonblocking(true)
            .and_then(|()| tokio::net::TcpListener::from_std(listener))
            .map_err(|e| e.to_string())?,
        None => restart::bind(
            &options.host,
            options.port,
            handover.from.is_some() || handover.again,
        )
        .await
        .map_err(|e| startup::bind_error(lang, &options.host, options.port, &e))?,
    };
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    let kept = restart::Kept::of(&listener);
    let check_cfg = cfg.clone();
    // A server also reachable under another name (--allowed-host, a proxy) is
    // treated as exposed: no web update.
    let mut update = update::Update::new(
        &cfg,
        !startup::exposed(address) && options.allowed_host.is_empty(),
        security::new_token(),
    );
    update.undone = crate::update::control::Pending::take_failure(&markitai_core::config::home())
        .filter(|failed| {
            update
                .current
                .as_ref()
                .is_some_and(|current| failed.from == current.to_string())
        });
    // A start after an undone update was not updated into.
    update.updated_from = handover.from.clone().filter(|from| {
        update.undone.is_none()
            && Some(from.as_str()) != update.current.as_ref().map(ToString::to_string).as_deref()
    });
    let state = Arc::new(State {
        settings: settings::Store::new(cfg, source).map_err(|error| error.detail)?,
        root,
        jobs: Mutex::new(HashMap::new()),
        file_slots: Arc::new(Semaphore::new(file_count)),
        url_slots: Arc::new(Semaphore::new(url_count)),
        closing: AtomicBool::new(false),
        persistence_failed: AtomicBool::new(false),
        shutdown,
        tasks: Mutex::new(Vec::new()),
        token: RwLock::new(token),
        token_generation: watch::channel(0).0,
        token_from_environment,
        address: OnceLock::new(),
        allowed_hosts,
        tickets: tickets::Tickets::default(),
        previews: original::Previews::default(),
        access,
        users: settings::Users::new(serve_home.join("user-models")),
        update,
    });
    state.users.check_formats()?;
    // The models of tokens revoked while the server was not running; with
    // unreadable tokens, none is known to be revoked.
    if state.access.tokens_read() {
        state.users.sweep(|id| state.access.exists(id));
    }
    store::rehydrate(&state.root, &state.jobs).map_err(|e| e.to_string())?;
    // Under the lifetime lock: no other server sweeps this history.
    let _ = &service_lock;
    if let Err(error) = retention::sweep(&state.root, &state.jobs, std::time::SystemTime::now()) {
        eprintln!("Warning: Serve: history retention sweep failed: {error}");
    }
    retention::start(&state);
    let router = router(&state);
    let _ = state.address.set(address);
    // Ready: the new version of an update stays, the old files go.
    if let Some(pending) = pending {
        pending.commit(&markitai_core::config::home());
    }
    update::publish_control(&state, address);
    update::start_checks(&state, &check_cfg);
    // The first OCR job would wait for the engine to load; with its models
    // installed (once OCR has been used here) it loads now, beside startup.
    let cfg = state.conversion_config(None);
    let _ = std::thread::Builder::new()
        .name("markitai-ocr-warm".into())
        .spawn(move || markitai_core::warm_local_ocr(&cfg));
    let token = state.token();
    let restarted = handover.from.is_some();
    if restarted && state.update.updated_from.is_some() {
        startup::print_restarted(
            lang,
            address,
            token.is_some(),
            state.update.updated_from.as_deref().unwrap_or_default(),
        );
    } else {
        print_startup(
            &state,
            lang,
            address,
            token.as_deref(),
            &check_cfg,
            undone.as_deref(),
        );
    }
    if !options.no_open && !restarted && undone.is_none() {
        let url = launch::browser_url(address, token.as_deref());
        if let Err(error) = launch::open_browser(&url) {
            crate::app::logging::shaped(
                crate::app::logging::Shape::item(crate::term::Mark::Warning),
                &format!(
                    "Warning: Could not open a browser: {error}; open the address above manually."
                ),
                Some(&crate::app::i18n::text!(lang =>
                    "Could not open a browser: {error}; open the address above manually.",
                    "无法打开浏览器：{error}；请手动打开上面的地址。")),
            );
        }
    }
    let shutdown_state = state.clone();
    let mut restart_requested = state.update.restart.subscribe();
    let outcome = axum::serve(
        listener,
        router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        let restarting = async {
            let _ = restart_requested.wait_for(Option::is_some).await;
        };
        #[cfg(unix)]
        let interrupted = async {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=terminate.recv()=>{} }
        };
        #[cfg(not(unix))]
        let interrupted = async {
            let _ = tokio::signal::ctrl_c().await;
        };
        let restart = tokio::select! { _ = interrupted => false, _ = restarting => true };
        shutdown_state.closing.store(true, Ordering::SeqCst);
        shutdown_state.shutdown.send_replace(true);
        if restart {
            crate::app::logging::shaped(
                crate::app::logging::Shape::item(crate::term::Mark::Action),
                "Restarting",
                (lang == crate::app::i18n::Lang::Zh).then_some("正在重启"),
            );
        } else {
            crate::app::logging::shaped(
                crate::app::logging::Shape::item(crate::term::Mark::Action),
                "Server stopping: draining active conversions and saving history.",
                (lang == crate::app::i18n::Lang::Zh)
                    .then_some("正在停止服务：等待进行中的转换完成并保存历史。"),
            );
        }
    })
    .await;
    state.closing.store(true, Ordering::SeqCst);
    state.shutdown.send_replace(true);
    let tasks = std::mem::take(&mut *state.tasks.lock().unwrap());
    for task in tasks {
        if let Err(error) = task.await {
            eprintln!("Error: Serve task failed: {error}");
            state.persistence_failed.store(true, Ordering::SeqCst);
        }
    }
    let restart = state.update.restart.borrow().clone();
    if restart.is_none() {
        // An update still downloading or waiting stops here; its files go.
        let worker = state.clone();
        let _ = crate::task::blocking(move || update::join_worker(&worker)).await;
    }
    update::withdraw_control(&state);
    let ready = |message: String| Failure {
        ready: true,
        message,
    };
    outcome.map_err(|e| ready(e.to_string()))?;
    if state.persistence_failed.load(Ordering::SeqCst) {
        return Err(ready(
            "one or more jobs could not be saved to history".into(),
        ));
    }
    Ok(match restart {
        Some(restart) => Outcome::Restart {
            restart,
            token: state.token(),
            kept,
        },
        None => Outcome::Stopped,
    })
}

/// Every route this server serves: the web update's only on a loopback server.
fn router(state: &Arc<State>) -> Router {
    let mut router = Router::new()
        .merge(web::routes())
        .merge(settings::routes())
        .merge(settings::user_routes())
        .merge(providers::routes())
        .merge(convert_defaults::routes())
        .merge(api_routes())
        .merge(update::control_routes::api_routes());
    if state.update.web {
        router = router.merge(update::api_routes());
    }
    router
        .fallback(http::missing)
        .method_not_allowed_fallback(http::method_not_allowed)
        .layer(axum::extract::DefaultBodyLimit::max(types::MAX_REQUEST))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            security::guard,
        ))
        .with_state(state.clone())
}

/// The startup block: the address, the token, what to open, where data lives,
/// the update line and how to stop.
fn print_startup(
    state: &State,
    lang: crate::app::i18n::Lang,
    address: std::net::SocketAddr,
    token: Option<&str>,
    cfg: &Value,
    undone: Option<&str>,
) {
    // Shown absolute: a relative MARKITAI_HOME would otherwise name nothing useful.
    let data = std::path::absolute(&state.root).unwrap_or_else(|_| state.root.clone());
    let failed = undone
        .map(str::to_owned)
        .or_else(|| state.update.undone.as_ref().map(|failed| failed.to.clone()));
    let newer = startup_notice(cfg);
    // After the lines that told why the update was undone.
    if failed.is_some() {
        crate::app::logging::blank();
    }
    for (kind, line) in startup::lines(
        lang,
        &startup::Startup {
            address,
            token,
            data: &data,
            network: address
                .ip()
                .is_unspecified()
                .then(startup::network_address)
                .flatten(),
            newer: newer.as_ref().map(|(latest, current, command)| {
                (latest.to_string(), current.to_string(), command.clone())
            }),
            failed_update: failed,
            current: crate::update::current_identity().0,
        },
    ) {
        // As written: redaction would drop the token from the addresses.
        use crate::app::logging::{Shape, blank, console};
        use crate::term::{Mark, Tone};
        let unwrapped = |mark, indent, tone| Shape {
            mark: Some(mark),
            indent,
            tone,
            wrap: false,
        };
        // The title's words are bold and its address, the last word scripts
        // read, is not: no colour code follows it even with FORCE_COLOR.
        let line = match (kind, line.rsplit_once(' ')) {
            (startup::Kind::Title, Some((words, address))) => format!(
                "{} {address}",
                crate::term::Style::stderr().paint(Tone::Structure, words)
            ),
            (startup::Kind::Notice, _) => match &newer {
                Some((latest, current, command)) => crate::update::notice_line(
                    lang,
                    crate::term::Style::stderr(),
                    2,
                    latest,
                    current,
                    command,
                ),
                None => line,
            },
            _ => line,
        };
        let shape = match kind {
            startup::Kind::Title => unwrapped(Mark::Title, 0, None),
            startup::Kind::Token => unwrapped(Mark::Info, 2, None),
            startup::Kind::Fact => Shape::item(Mark::Info),
            startup::Kind::Action => Shape::item(Mark::Action),
            startup::Kind::Warning => Shape::item(Mark::Warning),
            startup::Kind::More => Shape::under(4),
            startup::Kind::Address => Shape::COMMAND,
            startup::Kind::Notice => Shape {
                wrap: false,
                ..Shape::PLAIN
            },
        };
        console(shape, &line);
        if kind == startup::Kind::Title {
            blank();
        }
    }
}

/// The L1 line of the startup block, from the cache only: on a terminal,
/// outside CI, with checks on.
fn startup_notice(
    cfg: &Value,
) -> Option<(
    crate::update::version::Version,
    crate::update::version::Version,
    String,
)> {
    use std::io::IsTerminal;
    if !std::io::stderr().is_terminal() || crate::update::ci() {
        return None;
    }
    let (latest, current) = crate::update::cached_newer(Some(cfg))?;
    Some((latest, current, crate::update::command_for_notice()))
}
