//! One proxy decision for static fetches and native browser launches.
//!
//! As in the reference, the first nonempty of `HTTPS_PROXY`, `HTTP_PROXY`,
//! `ALL_PROXY` and their lowercase forms is the single proxy for every scheme.
//! An environment proxy prevents all operating-system reads. Otherwise manual
//! macOS, Windows or supported Linux desktop settings are read when a static
//! request is not already direct or a browser launches, and again at most every
//! 30 seconds, so a system proxy turned on or off is followed by a running
//! `markitai serve` (Python's aiohttp reads the settings on every request);
//! PAC/WPAD, credentials and network probes are never used. `NO_PROXY` (or
//! `no_proxy` when the former is unset or empty) always applies; the system
//! exception list is merged only when the system proxy is selected. Loopback
//! hosts are always direct. Model-provider clients (LLM requests, connection
//! checks, model discovery) and the browser download use the same decision, as
//! Python's aiohttp (LiteLLM) does through the environment variables and the
//! system settings, except that a SOCKS environment proxy is an explicit error
//! there (the client has no SOCKS support, and a silent direct connection would
//! leak the user's address). Provider Batch clients are not affected.
mod bypass;
mod system;
#[cfg(test)]
mod tests;

use crate::{Error, Result};
pub(crate) use bypass::Bypass;
use reqwest::blocking::ClientBuilder;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};
use url::Url;

const ENV_ORDER: [&str; 6] = [
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "https_proxy",
    "http_proxy",
    "all_proxy",
];
const MAX_SETTING: usize = 64 * 1024;

#[derive(Clone, Debug)]
struct Settings {
    environment: Option<Url>,
    /// The variable `environment` came from.
    source: Option<&'static str>,
    bypass: Bypass,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SystemProxy {
    endpoint: Url,
    bypass: String,
}

/// Selected operating-system proxy with its exception list.
type System = Option<(Url, Bypass)>;

fn invalid() -> Error {
    Error::Config("Proxy settings are malformed or exceed their supported bounds".into())
}

/// Validate a proxy endpoint. Environment values may name SOCKS5 or carry
/// credentials, as their owner wrote them; operating-system manual settings
/// are plain HTTP(S) host and port. A missing scheme means HTTP.
fn endpoint(value: &str, manual: bool) -> Result<Url> {
    let value = value.trim();
    if value.len() > MAX_SETTING
        || value
            .chars()
            .any(|ch| ch.is_whitespace() || ch.is_control())
    {
        return Err(invalid());
    }
    let spelled = if value.contains("://") {
        value.to_owned()
    } else {
        format!("http://{value}")
    };
    let parsed = Url::parse(&spelled).map_err(|_| invalid())?;
    let schemes: &[&str] = if manual {
        &["http", "https"]
    } else {
        &["http", "https", "socks5", "socks5h"]
    };
    if !schemes.contains(&parsed.scheme())
        || parsed.host_str().is_none_or(str::is_empty)
        || parsed.port() == Some(0)
        || !["", "/"].contains(&parsed.path())
        || parsed.query().is_some()
        || parsed.fragment().is_some()
        || (manual && (!parsed.username().is_empty() || parsed.password().is_some()))
    {
        return Err(invalid());
    }
    Ok(parsed)
}

impl Settings {
    fn resolve(env: &HashMap<String, String>) -> Result<Self> {
        let raw = ENV_ORDER.iter().find_map(|key| {
            env.get(*key)
                .map(String::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(|value| (*key, value))
        });
        let exceptions = env
            .get("NO_PROXY")
            .filter(|value| !value.is_empty())
            .or_else(|| env.get("no_proxy"));
        Ok(Self {
            environment: raw.map(|(_, raw)| endpoint(raw, false)).transpose()?,
            source: raw.map(|(key, _)| key),
            bypass: Bypass::parse(exceptions.map(String::as_str).unwrap_or(""))?,
        })
    }

    /// Loopback and `NO_PROXY` hosts are direct before any system read; an
    /// environment proxy prevents every system read.
    fn for_url(&self, url: &Url, system: impl FnOnce() -> System) -> Option<Url> {
        if self.bypass.matches(url) {
            return None;
        }
        if let Some(endpoint) = &self.environment {
            return Some(endpoint.clone());
        }
        system().and_then(|(endpoint, bypass)| (!bypass.matches(url)).then_some(endpoint))
    }

    /// A client that cannot speak SOCKS (model requests, the runtime download)
    /// refuses a SOCKS environment proxy instead of going direct behind the
    /// user's back.
    fn api_client(&self) -> Result<()> {
        if self
            .environment
            .as_ref()
            .is_some_and(|url| url.scheme().starts_with("socks"))
        {
            return Err(Error::Unsupported(format!(
                "SOCKS proxies are not used for model requests or downloads: {} is a SOCKS proxy, so set HTTPS_PROXY to an HTTP(S) proxy, or unset it to connect directly",
                self.source.unwrap_or("the proxy setting")
            )));
        }
        Ok(())
    }

    /// [`describe`] for these settings.
    fn describe(&self, url: &Url, system: impl FnOnce() -> System) -> Option<String> {
        let proxy = self.for_url(url, system)?;
        let place = match proxy.port() {
            Some(port) => format!("{}:{port}", proxy.host_str()?),
            None => proxy.host_str()?.to_owned(),
        };
        Some(format!(
            "{place} ({})",
            self.source
                .filter(|_| self.environment.is_some())
                .unwrap_or("the system proxy")
        ))
    }

    fn static_client(&self) -> Result<()> {
        if self
            .environment
            .as_ref()
            .is_some_and(|url| url.scheme().starts_with("socks"))
        {
            return Err(Error::Unsupported(
                "SOCKS proxies are available to the native browser only; static fetch requires an HTTP(S) proxy".into(),
            ));
        }
        Ok(())
    }

    fn browser(&self, system: impl FnOnce() -> System) -> Result<(Option<String>, String)> {
        let (endpoint, bypass) = match &self.environment {
            Some(endpoint) => (Some(endpoint.clone()), self.bypass.clone()),
            None => match system() {
                Some((endpoint, extra)) => {
                    let mut bypass = self.bypass.clone();
                    bypass.merge(extra)?;
                    (Some(endpoint), bypass)
                }
                None => (None, self.bypass.clone()),
            },
        };
        if let Some(url) = &endpoint {
            if !url.username().is_empty() || url.password().is_some() {
                return Err(Error::Unsupported(
                    "Native browser proxies cannot contain credentials".into(),
                ));
            }
            if url.scheme() == "socks5h" {
                return Err(Error::Unsupported(
                    "Native browser proxies support socks5, not socks5h".into(),
                ));
            }
        }
        Ok((
            endpoint.map(|url| url.as_str().trim_end_matches('/').to_owned()),
            bypass.chromium(),
        ))
    }
}

/// How long a read of the operating-system proxy settings is reused.
const SYSTEM_TTL: Duration = Duration::from_secs(30);

/// A system-proxy reading and when it was made.
struct Reading {
    at: Instant,
    value: System,
}

/// `cell`'s reading when it is younger than `ttl`, else `read()`, kept.
fn refreshed(
    cell: &Mutex<Option<Reading>>,
    ttl: Duration,
    now: Instant,
    read: impl FnOnce() -> System,
) -> System {
    let mut cell = cell.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(reading) = cell.as_ref()
        && now.saturating_duration_since(reading.at) < ttl
    {
        return reading.value.clone();
    }
    let value = read();
    *cell = Some(Reading {
        at: now,
        value: value.clone(),
    });
    value
}

/// The manual system proxy, read when first needed and again after
/// [`SYSTEM_TTL`]. Malformed or unavailable settings mean none, as in the
/// reference.
fn system(env: &HashMap<String, String>) -> System {
    static SYSTEM: Mutex<Option<Reading>> = Mutex::new(None);
    refreshed(&SYSTEM, SYSTEM_TTL, Instant::now(), || {
        let found = system::discover(env)?;
        Some((found.endpoint, Bypass::parse(&found.bypass).ok()?))
    })
}

/// Apply the shared decision to a static-fetch client. Redirects are routed
/// through the same custom selection; reqwest's own environment resolver is
/// disabled so a direct decision cannot be overridden.
pub(crate) fn http(builder: ClientBuilder) -> Result<ClientBuilder> {
    http_observed(builder, |_| {})
}

/// [`http`] that tells `observe` each proxy endpoint it chooses, before the
/// client connects to it.
pub(crate) fn http_observed(
    builder: ClientBuilder,
    observe: impl Fn(&Url) + Send + Sync + 'static,
) -> Result<ClientBuilder> {
    let env = crate::config::environment();
    let settings = Settings::resolve(&env)?;
    settings.static_client()?;
    Ok(builder.no_proxy().proxy(reqwest::Proxy::custom(move |url| {
        let endpoint = settings.for_url(url, || system(&env));
        if let Some(endpoint) = &endpoint {
            observe(endpoint);
        }
        endpoint
    })))
}

/// Apply the shared decision to a client that talks to a model provider or
/// downloads a runtime: HTTP(S) proxies from the environment, else the manual
/// system proxy. A malformed setting is a configuration error.
pub(crate) fn http_api(builder: ClientBuilder) -> Result<ClientBuilder> {
    let env = crate::config::environment();
    let settings = Settings::resolve(&env)?;
    settings.api_client()?;
    Ok(builder.no_proxy().proxy(reqwest::Proxy::custom(move |url| {
        settings.for_url(url, || system(&env))
    })))
}

/// The proxy a request to `url` goes through, for an error message: its host
/// and port and where it was set (never credentials), or `None` when the
/// request goes direct.
pub(crate) fn describe(url: &str) -> Option<String> {
    let env = crate::config::environment();
    let settings = Settings::resolve(&env).ok()?;
    settings.describe(&Url::parse(url).ok()?, || system(&env))
}

/// Chromium `--proxy-server` and `--proxy-bypass-list` values.
pub(crate) fn browser(env: &HashMap<String, String>) -> Result<(Option<String>, String)> {
    Settings::resolve(env)?.browser(|| system(env))
}
