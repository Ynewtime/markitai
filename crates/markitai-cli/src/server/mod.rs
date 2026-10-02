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

mod files;
mod http;
mod jobs;
mod launch;
mod openapi;
mod providers;
mod rerun;
mod security;
mod settings;
mod sidecar;
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
        Arc, Mutex,
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
    token: Option<String>,
    allowed_hosts: HashSet<String>,
    tickets: tickets::Tickets,
}

api_routes! {
    get "/api/capabilities" => http::capabilities;
    get "/api/openapi.json" => openapi::document;
    post "/api/download-tickets" => tickets::issue;
    post "/api/jobs" => http::create;
    get "/api/jobs/{job_id}" => http::snapshot;
    get "/api/jobs/{job_id}/events" => http::events;
    post "/api/jobs/{job_id}/cancel" => http::stop;
    post "/api/jobs/{job_id}/items/{item_id}/retry" => rerun::retry;
    delete "/api/jobs/{job_id}/items/{item_id}" => rerun::delete;
    get "/api/jobs/{job_id}/items/{item_id}/result" => files::result;
    get "/api/jobs/{job_id}/files/{*relpath}" => files::download;
    get "/api/jobs/{job_id}/archive" => files::job_archive;
    get "/api/history" => http::history;
    get "/api/history/archive" => files::history_archive;
    delete "/api/history/{job_id}" => http::delete;
}

/// Every API route the service registers, as (method, path).
#[cfg(test)]
fn api_table() -> Vec<(&'static str, &'static str)> {
    [ROUTES, settings::ROUTES, providers::ROUTES].concat()
}

pub(crate) fn settings_config(source: &SettingsSource) -> Result<Value, String> {
    settings::load_base(source).map_err(|error| error.detail)
}

pub(crate) fn run(cfg: Value, source: SettingsSource, options: ServeOptions) -> Result<(), String> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|e| e.to_string())?
        .block_on(serve(cfg, source, options))
}

async fn serve(cfg: Value, source: SettingsSource, options: ServeOptions) -> Result<(), String> {
    let root = markitai_core::config::home().join("serve/jobs");
    store::private_dir(&root).map_err(|e| e.to_string())?;
    let (shutdown, _) = watch::channel(false);
    let token = if options.no_auth {
        None
    } else {
        Some(
            std::env::var("MARKITAI_SERVE_TOKEN")
                .ok()
                .filter(|s| !s.trim().is_empty())
                .unwrap_or_else(|| {
                    format!(
                        "{}{}",
                        uuid::Uuid::new_v4().simple(),
                        uuid::Uuid::new_v4().simple()
                    )
                }),
        )
    };
    let allowed_hosts = options
        .allowed_host
        .iter()
        .map(|s| security::allowed_host(s).map_err(|e| e.detail))
        .collect::<Result<HashSet<_>, _>>()?;
    let file_count = cfg["batch"]["concurrency"].as_u64().unwrap_or(4).max(1) as usize;
    let url_count = cfg["batch"]["url_concurrency"].as_u64().unwrap_or(4).max(1) as usize;
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
        token,
        allowed_hosts,
        tickets: tickets::Tickets::default(),
    });
    store::rehydrate(&state.root, &state.jobs).map_err(|e| e.to_string())?;
    let router = Router::new()
        .merge(web::routes())
        .merge(settings::routes())
        .merge(providers::routes())
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
    // Shown absolute: a relative MARKITAI_HOME would otherwise name nothing useful.
    let data = std::path::absolute(&state.root).unwrap_or_else(|_| state.root.clone());
    for line in startup::lines(
        lang,
        &startup::Startup {
            address,
            token: state.token.as_deref(),
            data: &data,
            network: address
                .ip()
                .is_unspecified()
                .then(startup::network_address)
                .flatten(),
        },
    ) {
        eprintln!("{line}");
    }
    if !options.no_open {
        let url = launch::browser_url(address, state.token.as_deref());
        if let Err(error) = launch::open_browser(&url) {
            eprintln!("Could not open a browser: {error}; open the server address manually.");
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
        eprintln!("Server stopping: draining active conversions and saving history.");
    })
    .await;
    state.closing.store(true, Ordering::SeqCst);
    state.shutdown.send_replace(true);
    let tasks = std::mem::take(&mut *state.tasks.lock().unwrap());
    for task in tasks {
        if let Err(error) = task.await {
            eprintln!("Serve task failed: {error}");
            state.persistence_failed.store(true, Ordering::SeqCst);
        }
    }
    outcome.map_err(|e| e.to_string())?;
    if state.persistence_failed.load(Ordering::SeqCst) {
        return Err("one or more jobs could not be saved to history".into());
    }
    Ok(())
}
