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

pub(crate) fn run(cfg: Value, source: SettingsSource, options: ServeOptions) -> Result<(), String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?
        .block_on(serve(cfg, source, options))
}

async fn serve(cfg: Value, source: SettingsSource, options: ServeOptions) -> Result<(), String> {
    let serve_home = markitai_core::config::home().join("serve");
    let root = serve_home.join("jobs");
    let service_lock = store::service_lock(&root).map_err(|e| e.to_string())?;
    store::clean_uploads(&root, &service_lock).map_err(|e| e.to_string())?;
    let (shutdown, _) = watch::channel(false);
    let configured = std::env::var("MARKITAI_SERVE_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty());
    let token_from_environment = !options.no_auth && configured.is_some();
    let token = if options.no_auth {
        None
    } else {
        Some(configured.unwrap_or_else(security::new_token))
    };
    let allowed_hosts = options
        .allowed_host
        .iter()
        .map(|s| security::allowed_host(s).map_err(|e| e.detail))
        .collect::<Result<HashSet<_>, _>>()?;
    // Normalized settings always carry both limits (defaults 10 and 5, minimum 1).
    let file_count = cfg["batch"]["concurrency"].as_u64().unwrap_or(10) as usize;
    let url_count = cfg["batch"]["url_concurrency"].as_u64().unwrap_or(5) as usize;
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
        access: access::Access::load(serve_home.join("access.json")),
        users: settings::Users::new(serve_home.join("user-models")),
    });
    // The models of tokens revoked while the server was not running.
    state.users.sweep(|id| state.access.exists(id));
    store::rehydrate(&state.root, &state.jobs).map_err(|e| e.to_string())?;
    // Under the lifetime lock: no other server sweeps this history.
    let _ = &service_lock;
    if let Err(error) = retention::sweep(&state.root, &state.jobs, std::time::SystemTime::now()) {
        eprintln!("Warning: Serve: history retention sweep failed: {error}");
    }
    retention::start(&state);
    let router = Router::new()
        .merge(web::routes())
        .merge(settings::routes())
        .merge(settings::user_routes())
        .merge(providers::routes())
        .merge(convert_defaults::routes())
        .merge(api_routes())
        .fallback(http::missing)
        .method_not_allowed_fallback(http::method_not_allowed)
        .layer(axum::extract::DefaultBodyLimit::max(types::MAX_REQUEST))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            security::guard,
        ))
        .with_state(state.clone());
    let lang = crate::app::i18n::lang();
    let listener = tokio::net::TcpListener::bind((options.host.as_str(), options.port))
        .await
        .map_err(|e| startup::bind_error(lang, &options.host, options.port, &e))?;
    let address = listener.local_addr().map_err(|e| e.to_string())?;
    let _ = state.address.set(address);
    // The first OCR job would wait for the engine to load; with its models
    // installed (once OCR has been used here) it loads now, beside startup.
    let cfg = state.conversion_config(None);
    let _ = std::thread::Builder::new()
        .name("markitai-ocr-warm".into())
        .spawn(move || markitai_core::warm_local_ocr(&cfg));
    let token = state.token();
    // Shown absolute: a relative MARKITAI_HOME would otherwise name nothing useful.
    let data = std::path::absolute(&state.root).unwrap_or_else(|_| state.root.clone());
    for (kind, line) in startup::lines(
        lang,
        &startup::Startup {
            address,
            token: token.as_deref(),
            data: &data,
            network: address
                .ip()
                .is_unspecified()
                .then(startup::network_address)
                .flatten(),
        },
    ) {
        // As written: redaction would drop the token from the addresses.
        use crate::app::logging::{Shape, console};
        let shape = match kind {
            startup::Kind::Scripted => Shape {
                indent: 0,
                tone: None,
                ..Shape::COMMAND
            },
            startup::Kind::Plain => Shape::PLAIN,
            startup::Kind::Warning => Shape::verdict(crate::term::Mark::Warning),
            startup::Kind::More => Shape::under(2),
            startup::Kind::Address => Shape {
                indent: 4,
                ..Shape::COMMAND
            },
        };
        console(shape, &line);
    }
    if !options.no_open {
        let url = launch::browser_url(address, token.as_deref());
        if let Err(error) = launch::open_browser(&url) {
            eprintln!(
                "Warning: Could not open a browser: {error}; open the address above manually."
            );
        }
    }
    let shutdown_state = state.clone();
    let outcome = axum::serve(
        listener,
        router.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async move {
        #[cfg(unix)]
        {
            let mut terminate =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("install SIGTERM handler");
            tokio::select! { _=tokio::signal::ctrl_c()=>{}, _=terminate.recv()=>{} }
        }
        #[cfg(not(unix))]
        {
            let _ = tokio::signal::ctrl_c().await;
        }
        shutdown_state.closing.store(true, Ordering::SeqCst);
        shutdown_state.shutdown.send_replace(true);
        crate::app::logging::shaped(
            crate::app::logging::Shape::item(crate::term::Mark::Action),
            "Server stopping: draining active conversions and saving history.",
            None,
        );
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
    outcome.map_err(|e| e.to_string())?;
    if state.persistence_failed.load(Ordering::SeqCst) {
        return Err("one or more jobs could not be saved to history".into());
    }
    Ok(())
}
