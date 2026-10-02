use crate::{Error, Result, config};
use regex::Regex;
use serde_json::{Map, Value, json};
use std::collections::HashMap;
use url::Url;

pub(super) struct Options {
    pub persistent: bool,
    pub session_ttl: std::time::Duration,
    pub credentials: Option<super::auth::Credentials>,
    pub timeout: u64,
    pub wait_for: String,
    pub extra_wait: u64,
    pub selector: Option<String>,
    pub skip_scroll: bool,
    pub cookies: Vec<Value>,
    pub headers: Map<String, Value>,
    pub user_agent: Option<String>,
    /// The site turns away a browser that names itself `HeadlessChrome`: the
    /// browser's own user agent is used without that word (a configured
    /// `user_agent` always wins).
    pub regular_user_agent: bool,
    /// The site's own readers parse the dates and counters it renders in
    /// English (X), so its page is rendered with an `en-US` locale unless the
    /// caller sends an `Accept-Language` of its own. Other sites keep the
    /// browser's language, as the user's own browser would.
    pub english: bool,
    pub blocked: Vec<Regex>,
    pub proxy: Option<String>,
    pub bypass: String,
    pub width: u64,
    pub height: u64,
    pub quality: u64,
    pub tile_height: u64,
    pub max_height: u64,
}

fn unsigned(value: &Value, key: &str, default: u64, minimum: u64, maximum: u64) -> Result<u64> {
    let number = value.get(key).and_then(Value::as_u64).unwrap_or(default);
    if !(minimum..=maximum).contains(&number) {
        return Err(Error::Config(format!(
            "Native browser {key} must be between {minimum} and {maximum}"
        )));
    }
    Ok(number)
}

fn resource_pattern(pattern: &str) -> Result<Regex> {
    if pattern.len() > 4096 || pattern.contains(['{', '}', '[', ']', '\\']) {
        return Err(Error::Unsupported("Native browser resource patterns support literal URLs, * and **; braces, brackets and escapes are not supported".into()));
    }
    let mut regex = String::from("^");
    let mut chars = pattern.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '*' {
            if chars.peek() == Some(&'*') {
                chars.next();
                regex.push_str(".*");
            } else {
                regex.push_str("[^/]*");
            }
        } else {
            regex.push_str(&regex::escape(&ch.to_string()));
        }
    }
    regex.push('$');
    Regex::new(&regex).map_err(|_| Error::Config("Invalid browser resource pattern".into()))
}

fn cookies(value: &Value, url: &Url) -> Result<Vec<Value>> {
    let mut output = Vec::new();
    for cookie in value.as_array().into_iter().flatten() {
        let fields = cookie
            .as_object()
            .ok_or_else(|| Error::Config("Browser cookies must be objects".into()))?;
        if fields.keys().any(|key| {
            ![
                "name", "value", "url", "domain", "path", "expires", "httpOnly", "secure",
                "sameSite",
            ]
            .contains(&key.as_str())
        }) {
            return Err(Error::Unsupported(
                "Native browser cookie contains an unsupported field".into(),
            ));
        }
        let mut item = Map::new();
        for key in ["name", "value"] {
            let text = fields
                .get(key)
                .and_then(Value::as_str)
                .ok_or_else(|| Error::Config(format!("Browser cookie requires {key}")))?;
            item.insert(key.into(), json!(text));
        }
        for key in ["url", "domain", "path", "sameSite"] {
            if let Some(text) = fields.get(key).and_then(Value::as_str) {
                item.insert(key.into(), json!(text));
            }
        }
        if !item.contains_key("url") && !item.contains_key("domain") {
            item.insert("url".into(), json!(url.as_str()));
        }
        if item.contains_key("domain") && !item.contains_key("path") {
            item.insert("path".into(), json!("/"));
        }
        if let Some(same_site) = item.get("sameSite").and_then(Value::as_str)
            && !["Strict", "Lax", "None"].contains(&same_site)
        {
            return Err(Error::Config(
                "Cookie sameSite must be Strict, Lax or None".into(),
            ));
        }
        for key in ["secure", "httpOnly"] {
            if let Some(value) = fields.get(key) {
                let flag = match value {
                    Value::Bool(value) => *value,
                    Value::String(value) if ["true", "false"].contains(&value.as_str()) => {
                        value == "true"
                    }
                    _ => return Err(Error::Config(format!("Cookie {key} must be true or false"))),
                };
                item.insert(key.into(), json!(flag));
            }
        }
        if let Some(value) = fields.get("expires") {
            let number = value
                .as_f64()
                .or_else(|| value.as_str()?.parse::<f64>().ok())
                .filter(|number| number.is_finite())
                .ok_or_else(|| {
                    Error::Config("Cookie expires must be finite seconds since epoch".into())
                })?;
            item.insert("expires".into(), json!(number));
        }
        output.push(Value::Object(item));
    }
    if output.len() > 256 {
        return Err(Error::Config(
            "Native browser supports at most 256 initial cookies".into(),
        ));
    }
    Ok(output)
}

/// What a site gets without configuration: its waiting hints, and whether it
/// turns away a browser that names itself `HeadlessChrome` (such a site, X,
/// is also rendered in English for its readers).
fn builtin_profile(authority: &str) -> (Value, bool) {
    match authority {
        "github.com" => (
            json!({"wait_for_selector":".markdown-body","wait_for":"domcontentloaded","extra_wait_ms":300,"skip_auto_scroll":true}),
            false,
        ),
        // A post is an `article` whether or not the page tags it with test ids.
        "x.com" | "twitter.com" | "www.x.com" | "www.twitter.com" | "mobile.twitter.com" => (
            json!({"wait_for_selector":"article, [data-testid=\"tweet\"]","wait_for":"domcontentloaded","extra_wait_ms":500,"skip_auto_scroll":true,"reject_resource_patterns":["**/analytics/**","**/ads/**","**/tracking/**","**/*.mp4"]}),
            true,
        ),
        _ => (json!({}), false),
    }
}

/// The shared static-fetch proxy decision, projected into Chromium arguments.
fn proxy(env: &HashMap<String, String>) -> Result<(Option<String>, String)> {
    crate::proxy::browser(env)
}

impl Options {
    // Diagnostics do not load user configuration, cookies, credentials or proxies.
    pub(super) fn diagnostic() -> Self {
        Self {
            persistent: false,
            session_ttl: std::time::Duration::from_secs(600),
            credentials: None,
            timeout: 5000,
            wait_for: "domcontentloaded".into(),
            extra_wait: 0,
            selector: None,
            skip_scroll: true,
            cookies: Vec::new(),
            headers: Map::new(),
            user_agent: None,
            regular_user_agent: false,
            english: false,
            blocked: Vec::new(),
            proxy: None,
            bypass: String::new(),
            width: 1280,
            height: 720,
            quality: 85,
            tile_height: 2000,
            max_height: 10_000,
        }
    }

    pub fn from_config(cfg: &Value, url: &Url, capture: bool) -> Result<Self> {
        if !["http", "https"].contains(&url.scheme())
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
        {
            return Err(Error::InvalidInput(
                "Native browser requires an HTTP(S) URL without embedded credentials".into(),
            ));
        }
        let mut settings = cfg
            .pointer("/fetch/playwright")
            .cloned()
            .unwrap_or_else(|| json!({}));
        let persistent = match settings.get("session_mode") {
            None => false,
            Some(Value::String(mode)) if mode == "isolated" => false,
            Some(Value::String(mode)) if mode == "domain_persistent" => true,
            _ => {
                return Err(Error::Config(
                    "Browser session_mode must be isolated or domain_persistent".into(),
                ));
            }
        };
        let session_ttl = match settings.get("session_ttl_seconds") {
            None => 600,
            Some(value) => value
                .as_u64()
                .filter(|value| (60..=7200).contains(value))
                .ok_or_else(|| {
                    Error::Config("Browser session_ttl_seconds must be between 60 and 7200".into())
                })?,
        };
        let credentials = super::auth::parse(settings.get("http_credentials"), url)?;
        let authority = match url.port() {
            Some(port) => format!("{}:{port}", url.host_str().unwrap()),
            None => url.host_str().unwrap().to_owned(),
        };
        let (builtin, regular_user_agent) = builtin_profile(&authority);
        config::merge(&mut settings, builtin);
        if let Some(profile) = cfg
            .pointer("/fetch/domain_profiles")
            .and_then(|profiles| profiles.get(&authority))
            .and_then(Value::as_object)
        {
            for key in [
                "wait_for",
                "wait_for_selector",
                "extra_wait_ms",
                "skip_auto_scroll",
                "reject_resource_patterns",
            ] {
                if let Some(value) = profile.get(key).filter(|value| !value.is_null()) {
                    settings[key] = value.clone();
                }
            }
        }
        let timeout = unsigned(&settings, "timeout", 30_000, 1, 120_000)?;
        let wait_for = settings
            .get("wait_for")
            .and_then(Value::as_str)
            .unwrap_or("domcontentloaded")
            .to_owned();
        if !["load", "domcontentloaded", "networkidle"].contains(&wait_for.as_str()) {
            return Err(Error::Config("Unknown browser wait_for state".into()));
        }
        let mut headers = Map::new();
        if let Some(values) = settings
            .get("extra_http_headers")
            .and_then(Value::as_object)
        {
            for (key, value) in values {
                let value = value
                    .as_str()
                    .ok_or_else(|| Error::Config("Browser header values must be strings".into()))?;
                reqwest::header::HeaderName::from_bytes(key.as_bytes())
                    .map_err(|_| Error::Config("Invalid browser header name".into()))?;
                reqwest::header::HeaderValue::from_str(value)
                    .map_err(|_| Error::Config("Invalid browser header value".into()))?;
                headers.insert(key.clone(), json!(value));
            }
        }
        let blocked = settings
            .get("reject_resource_patterns")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .map(|value| resource_pattern(value.as_str().unwrap_or("")))
            .collect::<Result<Vec<_>>>()?;
        if blocked.len() > 256 {
            return Err(Error::Config(
                "Native browser supports at most 256 resource patterns".into(),
            ));
        }
        let screenshot = cfg.get("screenshot").cloned().unwrap_or_else(|| json!({}));
        let (proxy, bypass) = proxy(&config::environment())?;
        Ok(Self {
            persistent,
            session_ttl: std::time::Duration::from_secs(session_ttl),
            credentials,
            timeout,
            wait_for,
            extra_wait: unsigned(&settings, "extra_wait_ms", 3000, 0, 30_000)?,
            selector: settings
                .get("wait_for_selector")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .map(str::to_owned),
            skip_scroll: settings
                .get("skip_auto_scroll")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            cookies: cookies(settings.get("cookies").unwrap_or(&Value::Null), url)?,
            headers,
            user_agent: settings
                .get("user_agent")
                .and_then(Value::as_str)
                .map(str::to_owned),
            regular_user_agent,
            english: regular_user_agent,
            blocked,
            proxy,
            bypass,
            width: if capture {
                unsigned(&screenshot, "viewport_width", 1920, 1, 8192)?
            } else {
                1280
            },
            height: if capture {
                unsigned(&screenshot, "viewport_height", 1080, 1, 8192)?
            } else {
                720
            },
            quality: unsigned(&screenshot, "quality", 85, 1, 100)?,
            tile_height: unsigned(&screenshot, "tile_height", 2000, 0, 32_768)?,
            max_height: unsigned(&screenshot, "max_height", 10_000, 1, 32_768)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cookies_preserve_scope_and_convert_typed_protocol_fields() {
        let cookies = cookies(&json!([{"name":"session","value":"private","domain":"localhost","secure":"false","httpOnly":"true","expires":"1800000000"}]), &Url::parse("http://localhost/a").unwrap()).unwrap();
        assert_eq!(
            cookies[0],
            json!({"name":"session","value":"private","domain":"localhost","path":"/","secure":false,"httpOnly":true,"expires":1800000000.0})
        );
        assert!(
            super::cookies(
                &json!([{"name":"a","value":"b","partitionKey":"unsupported"}]),
                &Url::parse("http://localhost/").unwrap()
            )
            .is_err()
        );
    }
    #[test]
    fn resource_patterns_respect_path_boundaries_and_literal_query_marks() {
        assert!(
            resource_pattern("**/ads/**")
                .unwrap()
                .is_match("https://a.test/ads/x.js")
        );
        assert!(
            !resource_pattern("https://a.test/*.js")
                .unwrap()
                .is_match("https://a.test/deep/x.js")
        );
        assert!(
            resource_pattern("https://a.test/x?q=1")
                .unwrap()
                .is_match("https://a.test/x?q=1")
        );
        assert!(resource_pattern("**/*.{js,css}").is_err());
    }
    #[test]
    fn x_waits_for_a_post_and_presents_a_regular_browser_and_other_sites_do_not() {
        for authority in ["x.com", "twitter.com", "www.x.com", "mobile.twitter.com"] {
            let (profile, regular) = builtin_profile(authority);
            assert!(regular, "{authority}");
            // Both the tagged posts and the untagged 2026 page have an `article`.
            assert_eq!(
                profile["wait_for_selector"],
                "article, [data-testid=\"tweet\"]"
            );
        }
        for authority in [
            "github.com",
            "example.com",
            "x.com:8443",
            "notx.com",
            "x.com.example.org",
        ] {
            assert!(!builtin_profile(authority).1, "{authority}");
        }
    }

    #[test]
    fn proxy_credentials_are_not_put_in_process_arguments() {
        let env = HashMap::from([(
            "HTTPS_PROXY".into(),
            "http://user:secret@proxy.test:8080".into(),
        )]);
        assert!(proxy(&env).is_err());
        let env = HashMap::from([
            ("HTTPS_PROXY".into(), "http://proxy.test:8080".into()),
            ("NO_PROXY".into(), "example.test,::1".into()),
        ]);
        let (server, bypass) = proxy(&env).unwrap();
        assert_eq!(server.as_deref(), Some("http://proxy.test:8080"));
        assert!(bypass.contains("example.test") && bypass.contains("[::1]"));
    }
}
