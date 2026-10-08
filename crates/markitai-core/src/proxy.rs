//! One proxy decision for static fetches and native browser launches.
//!
//! As in the reference, the first nonempty of `HTTPS_PROXY`, `HTTP_PROXY`,
//! `ALL_PROXY` and their lowercase forms is the single proxy for every scheme.
//! An environment proxy prevents all operating-system reads. Otherwise manual
//! macOS, Windows or supported Linux desktop settings are read once per process,
//! when a static request is not already direct or a browser launches;
//! PAC/WPAD, credentials and network probes are never used. `NO_PROXY` (or
//! `no_proxy` when the former is unset or empty) always applies; the system
//! exception list is merged only when the system proxy is selected. Loopback
//! hosts are always direct. LLM and Provider Batch clients are not affected.
mod bypass;
mod system;
#[cfg(test)]
mod tests;

use crate::{Error, Result};
pub(crate) use bypass::Bypass;
use reqwest::blocking::ClientBuilder;
use std::collections::HashMap;
use std::sync::OnceLock;
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
        });
        let exceptions = env
            .get("NO_PROXY")
            .filter(|value| !value.is_empty())
            .or_else(|| env.get("no_proxy"));
        Ok(Self {
            environment: raw.map(|raw| endpoint(raw, false)).transpose()?,
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

/// The process-wide manual system proxy, read on first need. Malformed or
/// unavailable settings mean none, as in the reference.
fn system(env: &HashMap<String, String>) -> System {
    static SYSTEM: OnceLock<System> = OnceLock::new();
    SYSTEM
        .get_or_init(|| {
            let found = system::discover(env)?;
            Some((found.endpoint, Bypass::parse(&found.bypass).ok()?))
        })
        .clone()
}

/// Apply the shared decision to a static-fetch client. Redirects are routed
/// through the same custom selection; reqwest's own environment resolver is
/// disabled so a direct decision cannot be overridden.
pub(crate) fn http(builder: ClientBuilder) -> Result<ClientBuilder> {
    let env = crate::config::environment();
    let settings = Settings::resolve(&env)?;
    settings.static_client()?;
    Ok(builder.no_proxy().proxy(reqwest::Proxy::custom(move |url| {
        settings.for_url(url, || system(&env))
    })))
}

/// Chromium `--proxy-server` and `--proxy-bypass-list` values.
pub(crate) fn browser(env: &HashMap<String, String>) -> Result<(Option<String>, String)> {
    Settings::resolve(env)?.browser(|| system(env))
}
