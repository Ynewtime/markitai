//! Amazon Bedrock and Google Vertex AI without their SDKs: request signing
//! (AWS Signature Version 4, or a Bedrock API key), the AWS credential chain,
//! the Bedrock Converse wire format, and Google OAuth access tokens minted
//! from service-account or user credentials. Every network exchange here is
//! one bounded HTTPS request; credentials never reach logs or errors.
use super::*;
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

const AWS_SERVICE: &str = "bedrock";
const GOOGLE_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const GOOGLE_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const MAX_CREDENTIAL_FILE: u64 = 1024 * 1024;
const MAX_TOKEN_RESPONSE: u64 = 64 * 1024;

/// How a cloud deployment authenticates and shapes its requests.
pub(super) enum Cloud {
    /// Bedrock Converse: the deployment's Anthropic-shaped request is sent
    /// in the Converse format, and its answer read back in Anthropic's.
    Bedrock { auth: AwsAuth, region: String },
    /// Vertex AI: a Google access token as Bearer; Claude models take the
    /// Anthropic body with Vertex's version field instead of a model name.
    Vertex { tokens: GoogleTokens },
}
impl std::fmt::Debug for Cloud {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::Bedrock { .. } => "Bedrock",
            Self::Vertex { .. } => "Vertex",
        })
    }
}

pub(super) enum AwsAuth {
    /// A Bedrock API key (`AWS_BEARER_TOKEN_BEDROCK`).
    Bearer(String),
    Signed(AwsCredentials),
    /// A profile's credential program (`credential_process`, or the AWS CLI
    /// for SSO and role profiles), run when a request needs it.
    Program(Vec<String>),
}

#[derive(Clone)]
pub(super) struct AwsCredentials {
    pub access_key_id: String,
    pub secret_access_key: String,
    pub session_token: Option<String>,
}
impl std::fmt::Debug for AwsCredentials {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("AwsCredentials([REDACTED])")
    }
}

/// RFC 3986 unreserved characters stay; everything else is `%XX` (uppercase).
fn aws_encode(text: &str) -> String {
    let mut encoded = String::with_capacity(text.len());
    for byte in text.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~') {
            encoded.push(byte as char);
        } else {
            encoded.push_str(&format!("%{byte:02X}"));
        }
    }
    encoded
}

fn percent_decode(text: &str) -> String {
    let hex = |byte: u8| (byte as char).to_digit(16).map(|digit| digit as u8);
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%'
            && index + 2 < bytes.len()
            && let (Some(high), Some(low)) = (hex(bytes[index + 1]), hex(bytes[index + 2]))
        {
            out.push(high * 16 + low);
            index += 3;
        } else {
            out.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A host on this machine: `localhost` names and loopback addresses.
pub(super) fn local_host(host: &str) -> bool {
    let host = host.trim_start_matches('[').trim_end_matches(']');
    host.eq_ignore_ascii_case("localhost")
        || host.to_ascii_lowercase().ends_with(".localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|address| address.is_loopback())
}

fn sha256_hex(bytes: &[u8]) -> String {
    crate::hex(Sha256::digest(bytes))
}

fn hmac(key: &[u8], message: &[u8]) -> Vec<u8> {
    let key = ring::hmac::Key::new(ring::hmac::HMAC_SHA256, key);
    ring::hmac::sign(&key, message).as_ref().to_vec()
}

/// The SigV4 canonical request and its signed-header list. `path` and
/// `query` are as they appear on the wire; each path segment is encoded
/// again (non-S3 services), query pairs are decoded, re-encoded and sorted,
/// and header values are trimmed with inner runs of spaces collapsed.
pub(super) fn canonical_request(
    method: &str,
    path: &str,
    query: &str,
    headers: &[(String, String)],
    payload_hash: &str,
) -> (String, String) {
    let uri = if path.is_empty() {
        "/".to_owned()
    } else {
        path.split('/')
            .map(aws_encode)
            .collect::<Vec<_>>()
            .join("/")
    };
    let mut pairs: Vec<(String, String)> = query
        .split('&')
        .filter(|pair| !pair.is_empty())
        .map(|pair| {
            let (key, value) = pair.split_once('=').unwrap_or((pair, ""));
            (
                aws_encode(&percent_decode(key)),
                aws_encode(&percent_decode(value)),
            )
        })
        .collect();
    pairs.sort();
    let query = pairs
        .iter()
        .map(|(key, value)| format!("{key}={value}"))
        .collect::<Vec<_>>()
        .join("&");
    let mut canonical: Vec<(String, String)> = Vec::new();
    for (name, value) in headers {
        let name = name.to_ascii_lowercase();
        let value = value.split_whitespace().collect::<Vec<_>>().join(" ");
        match canonical.iter_mut().find(|(known, _)| *known == name) {
            Some((_, values)) => {
                values.push(',');
                values.push_str(&value);
            }
            None => canonical.push((name, value)),
        }
    }
    canonical.sort();
    let signed = canonical
        .iter()
        .map(|(name, _)| name.as_str())
        .collect::<Vec<_>>()
        .join(";");
    let lines = canonical
        .iter()
        .map(|(name, value)| format!("{name}:{value}\n"))
        .collect::<String>();
    (
        format!("{method}\n{uri}\n{query}\n{lines}\n{signed}\n{payload_hash}"),
        signed,
    )
}

/// `(YYYYMMDD'T'HHMMSS'Z', YYYYMMDD)` of a time.
fn amz_date(time: SystemTime) -> (String, String) {
    let stamp: chrono::DateTime<chrono::Utc> = time.into();
    (
        stamp.format("%Y%m%dT%H%M%SZ").to_string(),
        stamp.format("%Y%m%d").to_string(),
    )
}

/// The headers that sign one request: `x-amz-date`, the session token when
/// there is one, and `authorization`. `headers` are the other headers the
/// request sends and should be signed (at least `host`).
#[allow(clippy::too_many_arguments)]
pub(super) fn sign(
    method: &str,
    path: &str,
    query: &str,
    headers: &[(String, String)],
    body: &[u8],
    credentials: &AwsCredentials,
    region: &str,
    service: &str,
    time: SystemTime,
) -> Vec<(String, String)> {
    let (date_time, date) = amz_date(time);
    let mut added = vec![("x-amz-date".to_owned(), date_time.clone())];
    if let Some(token) = &credentials.session_token {
        added.push(("x-amz-security-token".to_owned(), token.clone()));
    }
    let mut all = headers.to_vec();
    all.extend(added.iter().cloned());
    let (canonical, signed) = canonical_request(method, path, query, &all, &sha256_hex(body));
    let scope = format!("{date}/{region}/{service}/aws4_request");
    let to_sign = format!(
        "AWS4-HMAC-SHA256\n{date_time}\n{scope}\n{}",
        sha256_hex(canonical.as_bytes())
    );
    let mut key = hmac(
        format!("AWS4{}", credentials.secret_access_key).as_bytes(),
        date.as_bytes(),
    );
    for part in [region, service, "aws4_request"] {
        key = hmac(&key, part.as_bytes());
    }
    let signature = crate::hex(hmac(&key, to_sign.as_bytes()));
    added.push((
        "authorization".to_owned(),
        format!(
            "AWS4-HMAC-SHA256 Credential={}/{scope}, SignedHeaders={signed}, Signature={signature}",
            credentials.access_key_id
        ),
    ));
    added
}

fn home(env: &HashMap<String, String>) -> Option<PathBuf> {
    env.get("HOME")
        .or_else(|| env.get("USERPROFILE"))
        .filter(|home| !home.is_empty())
        .map(PathBuf::from)
}

fn read_small(path: &Path) -> Option<String> {
    let file = crate::platform::open_read(path, true).ok()?;
    let metadata = file.metadata().ok()?;
    if !metadata.is_file() || metadata.len() > MAX_CREDENTIAL_FILE {
        return None;
    }
    let bytes = crate::platform::read_limited(file, MAX_CREDENTIAL_FILE).ok()??;
    String::from_utf8(bytes).ok()
}

/// One profile's keys from an AWS INI file (`[name]` in the credentials
/// file, `[profile name]` or `[default]` in the config file).
fn ini_profile(text: &str, section: &str) -> HashMap<String, String> {
    let mut found = HashMap::new();
    let mut inside = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('#') || line.starts_with(';') || line.is_empty() {
            continue;
        }
        if let Some(name) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            inside = name.trim() == section;
            continue;
        }
        if inside && let Some((key, value)) = line.split_once('=') {
            found.insert(key.trim().to_ascii_lowercase(), value.trim().to_owned());
        }
    }
    found
}

fn aws_files(env: &HashMap<String, String>) -> (Option<String>, Option<String>) {
    let file = |variable: &str, default: &str| {
        env.get(variable)
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .or_else(|| home(env).map(|home| home.join(".aws").join(default)))
            .and_then(|path| read_small(&path))
    };
    (
        file("AWS_SHARED_CREDENTIALS_FILE", "credentials"),
        file("AWS_CONFIG_FILE", "config"),
    )
}

/// Run a credential program (`credential_process`, or the AWS CLI's
/// `configure export-credentials`) and read its JSON (Version 1). Results
/// are reused until five minutes before their `Expiration`, or ten minutes.
/// Program answers by command line, with the instant they stop being reused.
type ProgramCache = Mutex<HashMap<Vec<String>, (AwsCredentials, Instant)>>;

fn program_credentials(command: &[String]) -> Result<AwsCredentials> {
    static CACHE: OnceLock<ProgramCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some((credentials, until)) = cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .get(command)
        && *until > Instant::now()
    {
        return Ok(credentials.clone());
    }
    let (credentials, lifetime) = run_credential_program(command)?;
    cache
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .insert(
            command.to_vec(),
            (credentials.clone(), Instant::now() + lifetime),
        );
    Ok(credentials)
}

fn run_credential_program(command: &[String]) -> Result<(AwsCredentials, Duration)> {
    let program = command.first().map(String::as_str).unwrap_or("");
    let failed = || {
        Error::Config(format!(
            "AWS credential program {program} did not return credentials"
        ))
    };
    let output = std::process::Command::new(program)
        .args(&command[1..])
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .output()
        .map_err(|_| failed())?;
    if !output.status.success() || output.stdout.len() > 64 * 1024 {
        return Err(failed());
    }
    let value: Value = serde_json::from_slice(&output.stdout).map_err(|_| failed())?;
    let field = |name: &str| {
        value
            .get(name)
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    };
    let lifetime = field("Expiration")
        .and_then(|expiration| chrono::DateTime::parse_from_rfc3339(&expiration).ok())
        .and_then(|expiration| {
            (expiration.with_timezone(&chrono::Utc) - chrono::Utc::now())
                .to_std()
                .ok()
        })
        .map_or(Duration::from_secs(600), |left| {
            left.saturating_sub(Duration::from_secs(300))
        });
    Ok((
        AwsCredentials {
            access_key_id: field("AccessKeyId").ok_or_else(failed)?,
            secret_access_key: field("SecretAccessKey").ok_or_else(failed)?,
            session_token: field("SessionToken"),
        },
        lifetime,
    ))
}

/// A shared-file profile: static keys, `credential_process`, or (SSO and
/// role profiles) the AWS CLI's exported credentials. Programs are not run
/// here, only named.
fn profile_credentials(
    name: &str,
    env: &HashMap<String, String>,
) -> Result<Option<(AwsAuth, Option<String>)>> {
    let (credentials, config) = aws_files(env);
    let mut keys = credentials
        .as_deref()
        .map(|text| ini_profile(text, name))
        .unwrap_or_default();
    let section = if name == "default" {
        "default".to_owned()
    } else {
        format!("profile {name}")
    };
    let settings = config
        .as_deref()
        .map(|text| ini_profile(text, &section))
        .unwrap_or_default();
    for (key, value) in &settings {
        keys.entry(key.clone()).or_insert_with(|| value.clone());
    }
    if keys.is_empty() {
        return Ok(None);
    }
    let region = keys.get("region").cloned();
    if let (Some(id), Some(secret)) = (
        keys.get("aws_access_key_id"),
        keys.get("aws_secret_access_key"),
    ) {
        return Ok(Some((
            AwsAuth::Signed(AwsCredentials {
                access_key_id: id.clone(),
                secret_access_key: secret.clone(),
                session_token: keys.get("aws_session_token").cloned(),
            }),
            region,
        )));
    }
    if let Some(process) = keys.get("credential_process") {
        let command: Vec<String> = process.split_whitespace().map(str::to_owned).collect();
        if command.is_empty() {
            return Err(Error::Config("AWS credential_process is empty".into()));
        }
        return Ok(Some((AwsAuth::Program(command), region)));
    }
    if keys.contains_key("sso_session")
        || keys.contains_key("sso_start_url")
        || keys.contains_key("role_arn")
    {
        let command = [
            "aws",
            "configure",
            "export-credentials",
            "--profile",
            name,
            "--format",
            "process",
        ]
        .map(str::to_owned)
        .to_vec();
        return Ok(Some((AwsAuth::Program(command), region)));
    }
    Ok(None)
}

/// A deployment's string parameter, with `env:` references resolved.
fn param(params: &Value, name: &str, env: &HashMap<String, String>) -> Result<Option<String>> {
    config::resolve_optional(
        params
            .get(name)
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty()),
        None,
        env,
        true,
    )
}

fn variable(env: &HashMap<String, String>, names: &[&str]) -> Option<String> {
    names.iter().find_map(|name| {
        env.get(*name)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .map(str::to_owned)
    })
}

/// Bedrock's authentication and region, in the order the AWS SDKs (and so
/// LiteLLM) apply: the deployment's `api_key` (a Bedrock API key),
/// `AWS_BEARER_TOKEN_BEDROCK`, the AWS key variables, then the profile
/// `AWS_PROFILE` or `default`. Like the reference's configuration, a
/// deployment names no other AWS setting; they come from the environment.
pub(super) fn bedrock_auth(
    params: &Value,
    env: &HashMap<String, String>,
    allow_environment: bool,
) -> Result<(AwsAuth, String)> {
    let empty = HashMap::new();
    let environment = if allow_environment { env } else { &empty };
    let mut region = None;
    let auth = if let Some(key) = param(params, "api_key", env)? {
        AwsAuth::Bearer(key)
    } else if let Some(token) = variable(environment, &["AWS_BEARER_TOKEN_BEDROCK"]) {
        AwsAuth::Bearer(token)
    } else if let (Some(id), Some(secret)) = (
        variable(environment, &["AWS_ACCESS_KEY_ID"]),
        variable(environment, &["AWS_SECRET_ACCESS_KEY"]),
    ) {
        AwsAuth::Signed(AwsCredentials {
            access_key_id: id,
            secret_access_key: secret,
            session_token: variable(environment, &["AWS_SESSION_TOKEN"]),
        })
    } else {
        let profile = variable(environment, &["AWS_PROFILE"]).unwrap_or_else(|| "default".into());
        let (auth, profile_region) = profile_credentials(&profile, environment)?
            .ok_or_else(|| {
                Error::Config(
                    "Bedrock needs AWS credentials: api_key or AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key), AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY, or an AWS profile".into(),
                )
            })?;
        region = region.or(profile_region);
        auth
    };
    let region = variable(
        environment,
        &["AWS_REGION_NAME", "AWS_REGION", "AWS_DEFAULT_REGION"],
    )
    .or(region)
    .unwrap_or_else(|| "us-east-1".into());
    if !region
        .bytes()
        .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
        || region.len() > 32
    {
        return Err(Error::Config("Bedrock needs a valid AWS region".into()));
    }
    Ok((auth, region))
}

/// `…/model/<id>/converse`, the ID encoded as one path segment (an ARN or a
/// versioned ID keeps its `:` and `/` inside it).
pub(super) fn converse_endpoint(base: &str, model: &str) -> Result<String> {
    let mut url = url::Url::parse(base)
        .map_err(|_| Error::Config("Bedrock runtime endpoint is not a valid URL".into()))?;
    if url.scheme() != "https" && !url.host_str().is_some_and(local_host) {
        return Err(Error::Config(
            "Bedrock runtime endpoint must use HTTPS".into(),
        ));
    }
    let path = format!(
        "{}/model/{}/converse",
        url.path().trim_end_matches('/'),
        aws_encode(model)
    );
    url.set_path(&path);
    url.set_query(None);
    Ok(url.into())
}

fn image_format(media_type: &str) -> Option<&'static str> {
    match media_type {
        "image/png" => Some("png"),
        "image/jpeg" | "image/jpg" => Some("jpeg"),
        "image/gif" => Some("gif"),
        "image/webp" => Some("webp"),
        _ => None,
    }
}

/// An Anthropic Messages request body in the Converse format.
pub(super) fn converse_request(body: &Value) -> Value {
    let text_blocks = |value: &Value| -> Vec<Value> {
        match value {
            Value::String(text) => vec![json!({"text":text})],
            Value::Array(blocks) => blocks
                .iter()
                .filter_map(|block| block.get("text").and_then(Value::as_str))
                .map(|text| json!({"text":text}))
                .collect(),
            _ => Vec::new(),
        }
    };
    let messages: Vec<Value> = body
        .get("messages")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|message| {
            let content: Vec<Value> = match message.get("content") {
                Some(Value::String(text)) => vec![json!({"text":text})],
                Some(Value::Array(blocks)) => blocks
                    .iter()
                    .filter_map(|block| match block.get("type").and_then(Value::as_str) {
                        Some("text") => Some(json!({"text":block["text"]})),
                        Some("image") => {
                            let media = block.pointer("/source/media_type")?.as_str()?;
                            Some(json!({"image":{"format":image_format(media)?,
                                "source":{"bytes":block.pointer("/source/data")?}}}))
                        }
                        _ => None,
                    })
                    .collect(),
                _ => Vec::new(),
            };
            json!({"role":message.get("role").cloned().unwrap_or_else(|| json!("user")),"content":content})
        })
        .collect();
    let mut request = json!({"messages":messages});
    if let Some(system) = body.get("system") {
        request["system"] = json!(text_blocks(system));
    }
    if let Some(tokens) = body.get("max_tokens") {
        request["inferenceConfig"] = json!({"maxTokens":tokens});
    }
    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        let specs: Vec<Value> = tools
            .iter()
            .map(|tool| {
                json!({"toolSpec":{"name":tool["name"],"description":tool["description"],
                    "inputSchema":{"json":tool["input_schema"]}}})
            })
            .collect();
        let mut config = json!({"tools":specs});
        if let Some(name) = body.pointer("/tool_choice/name") {
            config["toolChoice"] = json!({"tool":{"name":name}});
        }
        request["toolConfig"] = config;
    }
    request
}

/// A Converse answer in the Anthropic Messages shape the rest of the
/// pipeline reads: content blocks, stop reason and usage.
pub(super) fn converse_response(data: &Value, model: &str) -> Value {
    let content: Vec<Value> = data
        .pointer("/output/message/content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|block| {
            if let Some(text) = block.get("text") {
                Some(json!({"type":"text","text":text}))
            } else {
                block.get("toolUse").map(|call| {
                    json!({"type":"tool_use","id":call["toolUseId"],"name":call["name"],"input":call["input"]})
                })
            }
        })
        .collect();
    let stop = match data.get("stopReason").and_then(Value::as_str) {
        Some("max_tokens" | "model_context_window_exceeded") => "max_tokens",
        Some("content_filtered" | "guardrail_intervened") => "refusal",
        Some("tool_use") => "tool_use",
        Some("stop_sequence") => "stop_sequence",
        Some("end_turn") => "end_turn",
        _ => "malformed",
    };
    let mut usage = json!({});
    for (from, to) in [
        ("inputTokens", "input_tokens"),
        ("outputTokens", "output_tokens"),
        ("cacheReadInputTokens", "cache_read_input_tokens"),
        ("cacheWriteInputTokens", "cache_creation_input_tokens"),
    ] {
        if let Some(count) = data
            .pointer(&format!("/usage/{from}"))
            .and_then(Value::as_u64)
        {
            usage[to] = json!(count);
        }
    }
    let mut message = json!({"type":"message","role":"assistant","model":model,"content":content,"stop_reason":stop});
    if data.get("usage").is_some_and(Value::is_object) {
        message["usage"] = usage;
    }
    message
}

/// Google credentials for Vertex AI, in Application Default Credentials
/// order: `GOOGLE_APPLICATION_CREDENTIALS`, the gcloud ADC file, then the
/// gcloud CLI.
pub(super) enum GoogleSource {
    ServiceAccount {
        email: String,
        key: Vec<u8>,
        key_id: Option<String>,
        token_uri: String,
    },
    User {
        client_id: String,
        client_secret: String,
        refresh_token: String,
        token_uri: String,
    },
    Cli,
}

pub(super) struct GoogleTokens {
    source: GoogleSource,
    /// Identifies the credential in the process-wide token cache.
    identity: String,
    pub project: Option<String>,
}

fn token_uri(value: Option<&str>) -> Result<String> {
    let uri = value.unwrap_or(GOOGLE_TOKEN_URI);
    let url = url::Url::parse(uri)
        .map_err(|_| Error::Config("Google credentials name an invalid token_uri".into()))?;
    if url.scheme() == "https" || (url.scheme() == "http" && url.host_str().is_some_and(local_host))
    {
        Ok(uri.to_owned())
    } else {
        Err(Error::Config(
            "Google credentials must name an HTTPS token_uri".into(),
        ))
    }
}

fn pem_der(pem: &str) -> Option<Vec<u8>> {
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect();
    base64::engine::general_purpose::STANDARD.decode(body).ok()
}

impl GoogleTokens {
    fn from_json(text: &str, origin: &str) -> Result<Self> {
        let invalid = || Error::Config(format!("Google credentials in {origin} are not valid"));
        let value: Value = serde_json::from_str(text).map_err(|_| invalid())?;
        let field = |name: &str| value.get(name).and_then(Value::as_str).map(str::to_owned);
        let project = field("project_id").or_else(|| field("quota_project_id"));
        match field("type").as_deref() {
            Some("service_account") => {
                let email = field("client_email").ok_or_else(invalid)?;
                let key = field("private_key")
                    .as_deref()
                    .and_then(pem_der)
                    .ok_or_else(invalid)?;
                ring::signature::RsaKeyPair::from_pkcs8(&key)
                    .or_else(|_| ring::signature::RsaKeyPair::from_der(&key))
                    .map_err(|_| invalid())?;
                Ok(Self {
                    identity: format!("sa:{email}:{}", crate::hex(Sha256::digest(&key))),
                    source: GoogleSource::ServiceAccount {
                        email,
                        key,
                        key_id: field("private_key_id"),
                        token_uri: token_uri(field("token_uri").as_deref())?,
                    },
                    project,
                })
            }
            Some("authorized_user") => {
                let refresh_token = field("refresh_token").ok_or_else(invalid)?;
                Ok(Self {
                    identity: format!(
                        "user:{}",
                        crate::hex(Sha256::digest(refresh_token.as_bytes()))
                    ),
                    source: GoogleSource::User {
                        client_id: field("client_id").ok_or_else(invalid)?,
                        client_secret: field("client_secret").ok_or_else(invalid)?,
                        refresh_token,
                        token_uri: token_uri(field("token_uri").as_deref())?,
                    },
                    project,
                })
            }
            _ => Err(Error::Unsupported(format!(
                "Google credentials in {origin} are of a type this build cannot use (service_account and authorized_user are supported; use the LiteLLM proxy for workload identity federation)"
            ))),
        }
    }

    pub(super) fn resolve(env: &HashMap<String, String>, allow_environment: bool) -> Result<Self> {
        // Every Google credential this build reads is the server's own (its
        // variables, its gcloud files or its gcloud login); a connection that
        // may not use the environment's credentials has none to send.
        if !allow_environment {
            return Err(Error::Config(
                "Vertex AI uses the server's Google credentials, which this connection may not use; configure it in the server's configuration file".into(),
            ));
        }
        if let Some(path) = variable(env, &["GOOGLE_APPLICATION_CREDENTIALS"]) {
            let text = read_small(Path::new(&path)).ok_or_else(|| {
                Error::Config(
                    "GOOGLE_APPLICATION_CREDENTIALS names a file that cannot be read".into(),
                )
            })?;
            return Self::from_json(&text, "GOOGLE_APPLICATION_CREDENTIALS");
        }
        let adc = variable(env, &["CLOUDSDK_CONFIG"])
            .map(PathBuf::from)
            .or_else(|| {
                variable(env, &["APPDATA"])
                    .filter(|_| cfg!(windows))
                    .map(|data| PathBuf::from(data).join("gcloud"))
            })
            .or_else(|| home(env).map(|home| home.join(".config").join("gcloud")))
            .map(|directory| directory.join("application_default_credentials.json"));
        if let Some(text) = adc.as_deref().and_then(read_small) {
            return Self::from_json(&text, "the gcloud application default credentials");
        }
        Ok(Self {
            source: GoogleSource::Cli,
            identity: "gcloud-cli".into(),
            project: None,
        })
    }

    /// A valid access token, from the process cache when it has more than a
    /// minute left.
    pub(super) fn token(&self, client: &Client) -> Result<String> {
        static CACHE: OnceLock<Mutex<HashMap<String, (String, Instant)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some((token, expires)) = cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .get(&self.identity)
            && expires.saturating_duration_since(Instant::now()) > Duration::from_secs(60)
        {
            return Ok(token.clone());
        }
        let (token, lifetime) = self.fetch(client)?;
        cache
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .insert(
                self.identity.clone(),
                (
                    token.clone(),
                    Instant::now() + Duration::from_secs(lifetime),
                ),
            );
        Ok(token)
    }

    fn fetch(&self, client: &Client) -> Result<(String, u64)> {
        let failed =
            |what: &str| Error::Conversion(format!("Google access token request failed: {what}"));
        let (uri, form): (&str, Vec<(&str, String)>) = match &self.source {
            GoogleSource::Cli => {
                let output = std::process::Command::new("gcloud")
                    .args(["auth", "application-default", "print-access-token"])
                    .stdin(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .output()
                    .map_err(|_| {
                        Error::Config(
                            "Vertex AI needs Google credentials: vertex_credentials, GOOGLE_APPLICATION_CREDENTIALS, `gcloud auth application-default login`, or an installed gcloud CLI".into(),
                        )
                    })?;
                let token = String::from_utf8_lossy(&output.stdout).trim().to_owned();
                if !output.status.success() || token.is_empty() || token.len() > 8192 {
                    return Err(failed("gcloud returned no token"));
                }
                // gcloud's tokens last an hour; refresh well before that.
                return Ok((token, 30 * 60));
            }
            GoogleSource::ServiceAccount {
                email,
                key,
                key_id,
                token_uri,
            } => {
                let now = SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs();
                let encode = |value: &Value| {
                    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value.to_string())
                };
                let mut header = json!({"alg":"RS256","typ":"JWT"});
                if let Some(id) = key_id {
                    header["kid"] = json!(id);
                }
                let claims = json!({"iss":email,"scope":GOOGLE_SCOPE,"aud":token_uri,"iat":now,"exp":now + 3600});
                let unsigned = format!("{}.{}", encode(&header), encode(&claims));
                let pair = ring::signature::RsaKeyPair::from_pkcs8(key)
                    .or_else(|_| ring::signature::RsaKeyPair::from_der(key))
                    .map_err(|_| failed("the service-account key was rejected"))?;
                let mut signature = vec![0; pair.public().modulus_len()];
                pair.sign(
                    &ring::signature::RSA_PKCS1_SHA256,
                    &ring::rand::SystemRandom::new(),
                    unsigned.as_bytes(),
                    &mut signature,
                )
                .map_err(|_| failed("signing failed"))?;
                let assertion = format!(
                    "{unsigned}.{}",
                    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(&signature)
                );
                (
                    token_uri,
                    vec![
                        (
                            "grant_type",
                            "urn:ietf:params:oauth:grant-type:jwt-bearer".into(),
                        ),
                        ("assertion", assertion),
                    ],
                )
            }
            GoogleSource::User {
                client_id,
                client_secret,
                refresh_token,
                token_uri,
            } => (
                token_uri,
                vec![
                    ("grant_type", "refresh_token".into()),
                    ("client_id", client_id.clone()),
                    ("client_secret", client_secret.clone()),
                    ("refresh_token", refresh_token.clone()),
                ],
            ),
        };
        let response = client
            .post(uri)
            .form(&form)
            .send()
            .map_err(|_| failed("the token endpoint could not be reached"))?;
        let status = response.status().as_u16();
        let bytes = crate::platform::read_limited(response, MAX_TOKEN_RESPONSE)
            .map_err(|_| failed("the answer could not be read"))?
            .ok_or_else(|| failed("the answer is too large"))?;
        if !(200..300).contains(&status) {
            return Err(failed(&format!("HTTP {status}")));
        }
        let value: Value =
            serde_json::from_slice(&bytes).map_err(|_| failed("the answer is not JSON"))?;
        let token = value
            .get("access_token")
            .and_then(Value::as_str)
            .filter(|token| !token.is_empty() && token.len() <= 8192)
            .ok_or_else(|| failed("the answer has no access_token"))?;
        let lifetime = value
            .get("expires_in")
            .and_then(Value::as_u64)
            .unwrap_or(3600)
            .clamp(60, 24 * 3600);
        Ok((token.to_owned(), lifetime))
    }
}

fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'@')
        })
}

/// A `bedrock/…` or `vertex_ai/…` deployment of `llm.model_list`.
pub(super) fn deployment(
    entry: &Value,
    provider: &str,
    model: &str,
    env: &HashMap<String, String>,
    grouped: bool,
) -> Result<Deployment> {
    let params = &entry["litellm_params"];
    // The settings API's keyless placeholder never takes the environment's
    // AWS or Google credentials either.
    let allow_environment = params.get("api_key").and_then(Value::as_str)
        != Some(crate::provider_management::NO_KEY)
        && params
            .get("use_environment_credentials")
            .and_then(Value::as_bool)
            != Some(false);
    let empty = HashMap::new();
    let environment = if allow_environment { env } else { &empty };
    if params
        .get("max_tokens")
        .is_some_and(|value| !value.is_null() && value.as_u64().is_none_or(|value| value == 0))
    {
        return Err(Error::Config(
            "LLM max_tokens must be positive when configured".into(),
        ));
    }
    let max_tokens = params
        .get("max_tokens")
        .and_then(Value::as_u64)
        .or_else(|| {
            entry
                .pointer("/model_info/max_tokens")
                .and_then(Value::as_u64)
        });
    let base = param(params, "api_base", env)?;
    let (wire, endpoint, protocol, cloud, identity) = if provider == "bedrock" {
        // LiteLLM's route segments choose Converse or Invoke; Converse serves both.
        let id = model
            .strip_prefix("converse/")
            .or_else(|| model.strip_prefix("invoke/"))
            .unwrap_or(model);
        if id.is_empty() || id.len() > 2048 || id.chars().any(char::is_control) {
            return Err(Error::Config("Bedrock model ID is invalid".into()));
        }
        let (auth, region) = bedrock_auth(params, env, allow_environment)?;
        let base = match base {
            Some(base) => base,
            None => variable(environment, &["AWS_BEDROCK_RUNTIME_ENDPOINT"])
                .unwrap_or_else(|| format!("https://bedrock-runtime.{region}.amazonaws.com")),
        };
        let identity = match &auth {
            AwsAuth::Bearer(token) => format!(
                "bedrock-key:{}",
                crate::hex(Sha256::digest(token.as_bytes()))
            ),
            AwsAuth::Signed(credentials) => format!("aws:{}", credentials.access_key_id),
            AwsAuth::Program(command) => format!("aws-program:{}", command.join(" ")),
        };
        (
            id.to_owned(),
            converse_endpoint(&base, id)?,
            Protocol::Anthropic,
            Cloud::Bedrock { auth, region },
            identity,
        )
    } else {
        let tokens = GoogleTokens::resolve(env, allow_environment)?;
        let project = variable(
            environment,
            &["VERTEXAI_PROJECT", "VERTEX_PROJECT", "GOOGLE_CLOUD_PROJECT"],
        )
        .or_else(|| tokens.project.clone())
        .ok_or_else(|| {
            Error::Config(
                "Vertex AI needs a project: VERTEXAI_PROJECT, GOOGLE_CLOUD_PROJECT or the credentials' project".into(),
            )
        })?;
        let location = variable(environment, &["VERTEXAI_LOCATION", "VERTEX_LOCATION"])
            .unwrap_or_else(|| "us-central1".into());
        if !valid_name(&project)
            || !valid_name(&location)
            || !valid_name(model.split('/').next_back().unwrap_or(""))
        {
            return Err(Error::Config(
                "Vertex AI project, location or model name is invalid".into(),
            ));
        }
        let host = base.unwrap_or_else(|| {
            if location == "global" {
                "https://aiplatform.googleapis.com".into()
            } else {
                format!("https://{location}-aiplatform.googleapis.com")
            }
        });
        let url = url::Url::parse(&host)
            .map_err(|_| Error::Config("Vertex AI api_base is not a valid URL".into()))?;
        if url.scheme() != "https" && !url.host_str().is_some_and(local_host) {
            return Err(Error::Config("Vertex AI api_base must use HTTPS".into()));
        }
        let host = host.trim_end_matches('/');
        let scope = format!("{host}/v1/projects/{project}/locations/{location}");
        let identity = format!("vertex:{project}:{}", tokens.identity);
        // Claude on Vertex is the Anthropic Messages API behind rawPredict;
        // Gemini and the other Model Garden models are served by the
        // OpenAI-compatible endpoint, which names them <publisher>/<model>.
        if model.starts_with("claude") {
            (
                model.to_owned(),
                format!("{scope}/publishers/anthropic/models/{model}:rawPredict"),
                Protocol::Anthropic,
                Cloud::Vertex { tokens },
                identity,
            )
        } else {
            (
                if model.contains('/') {
                    model.to_owned()
                } else {
                    format!("google/{model}")
                },
                format!("{scope}/endpoints/openapi/chat/completions"),
                Protocol::Chat,
                Cloud::Vertex { tokens },
                identity,
            )
        }
    };
    let reasoning_effort = reasoning_effort(provider, protocol, params, model)?;
    Ok(Deployment {
        id: format!("{provider}/{model}"),
        explicit_id: nonempty(entry.pointer("/model_info/id")).map(str::to_owned),
        hidden_endpoint: super::hidden_endpoint(entry),
        public_only: super::public_only(entry),
        group: if grouped {
            entry
                .get("model_name")
                .and_then(Value::as_str)
                .unwrap_or("default")
                .into()
        } else {
            "default".into()
        },
        model: wire,
        provider: provider.into(),
        weight: params.get("weight").and_then(Value::as_u64).unwrap_or(1),
        key: Some(identity),
        key_variable: None,
        endpoint,
        protocol,
        max_tokens,
        reasoning_effort,
        supports_vision: entry
            .pointer("/model_info/supports_vision")
            .and_then(Value::as_bool),
        cloud: Some(Arc::new(cloud)),
    })
}

/// Shape and authenticate one request body for its cloud.
pub(super) fn prepare(
    cloud: &Cloud,
    client: &Client,
    request: reqwest::blocking::RequestBuilder,
    entry: &Deployment,
    mut body: Value,
) -> Result<reqwest::blocking::RequestBuilder> {
    match cloud {
        Cloud::Bedrock { auth, region } => {
            let bytes = serde_json::to_vec(&converse_request(&body))
                .map_err(|_| Error::Conversion("Bedrock request could not be encoded".into()))?;
            let request = request.header("content-type", "application/json");
            let program;
            let credentials = match auth {
                AwsAuth::Bearer(token) => return Ok(request.bearer_auth(token).body(bytes)),
                AwsAuth::Signed(credentials) => credentials,
                AwsAuth::Program(command) => {
                    program = program_credentials(command)?;
                    &program
                }
            };
            let url = url::Url::parse(&entry.endpoint)
                .map_err(|_| Error::Config("Bedrock endpoint is invalid".into()))?;
            let host = url
                .host_str()
                .ok_or_else(|| Error::Config("Bedrock endpoint is invalid".into()))?;
            let host = match url.port() {
                Some(port) => format!("{host}:{port}"),
                None => host.to_owned(),
            };
            let headers = [
                ("content-type".to_owned(), "application/json".to_owned()),
                ("host".to_owned(), host),
            ];
            let mut request = request;
            for (name, value) in sign(
                "POST",
                url.path(),
                url.query().unwrap_or(""),
                &headers,
                &bytes,
                credentials,
                region,
                AWS_SERVICE,
                SystemTime::now(),
            ) {
                request = request.header(name, value);
            }
            Ok(request.body(bytes))
        }
        Cloud::Vertex { tokens } => {
            if entry.protocol == Protocol::Anthropic
                && let Some(object) = body.as_object_mut()
            {
                object.remove("model");
                object.insert("anthropic_version".into(), json!(super::ANTHROPIC_VERSION));
            }
            let token = tokens.token(client)?;
            Ok(request.bearer_auth(token).json(&body))
        }
    }
}

/// Bedrock answers in the Converse format; the rest of the pipeline reads
/// Anthropic's. Vertex answers already have their protocol's shape.
pub(super) fn response(cloud: &Cloud, data: Value, entry: &Deployment) -> Value {
    match cloud {
        Cloud::Bedrock { .. } => converse_response(&data, &entry.model),
        Cloud::Vertex { .. } => data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn headers(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }
    fn vars(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect()
    }
    const EXAMPLE: fn() -> AwsCredentials = || AwsCredentials {
        access_key_id: "AKIDEXAMPLE".into(),
        secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
        session_token: None,
    };
    fn at_2015() -> SystemTime {
        std::time::UNIX_EPOCH + Duration::from_secs(1_440_938_160)
    }
    fn signature(added: &[(String, String)]) -> String {
        let authorization = &added
            .iter()
            .find(|(name, _)| name == "authorization")
            .unwrap()
            .1;
        authorization
            .rsplit("Signature=")
            .next()
            .unwrap()
            .to_owned()
    }

    // Vectors of the AWS Signature Version 4 test suite (as shipped in the
    // Apache-2.0 aws-sigv4 crate, aws-signing-test-suite/v4): request,
    // canonical request and signature for its example credentials.
    #[test]
    fn signatures_match_the_aws_test_suite() {
        type Case<'a> = (
            &'a str,
            &'a str,
            &'a str,
            &'a [(&'a str, &'a str)],
            &'a [u8],
            bool,
            &'a str,
        );
        let cases: [Case; 5] = [
            (
                "GET",
                "/",
                "Param2=value2&Param1=value1",
                &[("Host", "example.amazonaws.com")],
                b"",
                false,
                "b97d918cfa904a5beff61c982a1b6f458b799221646efd99d3219ec94cdf2500",
            ),
            (
                "GET",
                "/",
                "",
                &[
                    ("Host", "example.amazonaws.com"),
                    ("My-Header1", " value1"),
                    ("My-Header2", " \"a   b   c\""),
                ],
                b"",
                false,
                "acc3ed3afb60bb290fc8d2dd0098b9911fcaa05412b367055dee359757a9c736",
            ),
            (
                "GET",
                "/",
                "",
                &[("Host", "example.amazonaws.com")],
                b"",
                true,
                "07ec1639c89043aa0e3e2de82b96708f198cceab042d4a97044c66dd9f74e7f8",
            ),
            (
                "GET",
                "/\u{1234}",
                "",
                &[("Host", "example.amazonaws.com")],
                b"",
                false,
                "8318018e0b0f223aa2bbf98705b62bb787dc9c0e678f255a891fd03141be5d85",
            ),
            (
                "POST",
                "/",
                "",
                &[("Host", "example.amazonaws.com")],
                b"",
                false,
                "5da7c1a2acd57cee7505fc6676e4e544621c30862966e37dddb68e92efbe5d6b",
            ),
        ];
        for (method, path, query, pairs, body, token, expected) in cases {
            let mut credentials = EXAMPLE();
            if token {
                credentials.session_token =
                    Some("6e86291e8372ff2a2260956d9b8aae1d763fbf315fa00fa31553b73ebf194267".into());
            }
            // The suite does not sign the payload: its hash is of the empty body.
            let added = sign(
                method,
                path,
                query,
                &headers(pairs),
                body,
                &credentials,
                "us-east-1",
                "service",
                at_2015(),
            );
            assert_eq!(signature(&added), expected, "{method} {path}?{query}");
            assert_eq!(added[0], ("x-amz-date".into(), "20150830T123600Z".into()));
        }
        // A signed body adds x-amz-content-sha256 in the suite's form request.
        let body = b"Param1=value1";
        let digest = sha256_hex(body);
        let added = sign(
            "POST",
            "/",
            "",
            &headers(&[
                ("Content-Type", "application/x-www-form-urlencoded"),
                ("Host", "example.amazonaws.com"),
                ("Content-Length", "13"),
                ("x-amz-content-sha256", &digest),
            ]),
            body,
            &EXAMPLE(),
            "us-east-1",
            "service",
            at_2015(),
        );
        assert_eq!(
            signature(&added),
            "d3875051da38690788ef43de4db0d8f280229d82040bfac253562e56c3f20e0b"
        );
        // Wire paths are encoded once more: `%3D` becomes `%253D`, `@` `%40`.
        let (canonical, _) = canonical_request(
            "POST",
            "/test/@connections/JBDvjfGEIAMCERw%3D",
            "",
            &headers(&[
                ("Host", "tj9n5r0m12.execute-api.us-east-1.amazonaws.com"),
                ("X-amz-date", "20210511T154045Z"),
            ]),
            &sha256_hex(b""),
        );
        assert_eq!(
            canonical,
            "POST\n/test/%40connections/JBDvjfGEIAMCERw%253D\n\nhost:tj9n5r0m12.execute-api.us-east-1.amazonaws.com\nx-amz-date:20210511T154045Z\n\nhost;x-amz-date\ne3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn model_ids_are_one_encoded_path_segment() {
        assert_eq!(
            converse_endpoint(
                "https://bedrock-runtime.us-east-1.amazonaws.com",
                "anthropic.claude-haiku-4-5-20251001-v1:0"
            )
            .unwrap(),
            "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse"
        );
        let arn = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-haiku-4-5";
        assert!(
            converse_endpoint("https://vpce.example", arn)
                .unwrap()
                .ends_with("/model/arn%3Aaws%3Abedrock%3Aus-east-1%3A123456789012%3Ainference-profile%2Fus.anthropic.claude-haiku-4-5/converse")
        );
        assert!(converse_endpoint("http://bedrock.example", "m").is_err());
        assert!(converse_endpoint("http://127.0.0.1:9", "m").is_ok());
    }

    #[test]
    fn converse_requests_and_answers_map_to_the_anthropic_shape() {
        let body = json!({"model":"m","max_tokens":512,
            "system":[{"type":"text","text":"Clean it.","cache_control":{"type":"ephemeral"}}],
            "messages":[{"role":"user","content":[{"type":"text","text":"Doc"},
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0"}}]}],
            "tools":[{"name":"MarkitaiDocument","description":"d","input_schema":{"type":"object"},"strict":true}],
            "tool_choice":{"type":"tool","name":"MarkitaiDocument","disable_parallel_tool_use":true}});
        assert_eq!(
            converse_request(&body),
            json!({"messages":[{"role":"user","content":[{"text":"Doc"},
                {"image":{"format":"png","source":{"bytes":"iVBORw0"}}}]}],
                "system":[{"text":"Clean it."}],"inferenceConfig":{"maxTokens":512},
                "toolConfig":{"tools":[{"toolSpec":{"name":"MarkitaiDocument","description":"d",
                    "inputSchema":{"json":{"type":"object"}}}}],"toolChoice":{"tool":{"name":"MarkitaiDocument"}}}})
        );
        let plain = json!({"max_tokens":8,"system":"s","messages":[{"role":"user","content":"u"}]});
        assert_eq!(
            converse_request(&plain),
            json!({"messages":[{"role":"user","content":[{"text":"u"}]}],"system":[{"text":"s"}],"inferenceConfig":{"maxTokens":8}})
        );
        let answer = json!({"output":{"message":{"role":"assistant","content":[{"text":"Hello"},
            {"toolUse":{"toolUseId":"t1","name":"MarkitaiDocument","input":{"a":1}}}]}},
            "stopReason":"tool_use","usage":{"inputTokens":30,"outputTokens":5,"totalTokens":35,
            "cacheReadInputTokens":10,"cacheWriteInputTokens":2},"metrics":{"latencyMs":9}});
        assert_eq!(
            converse_response(&answer, "m"),
            json!({"type":"message","role":"assistant","model":"m","stop_reason":"tool_use",
                "content":[{"type":"text","text":"Hello"},{"type":"tool_use","id":"t1","name":"MarkitaiDocument","input":{"a":1}}],
                "usage":{"input_tokens":30,"output_tokens":5,"cache_read_input_tokens":10,"cache_creation_input_tokens":2}})
        );
        for (reason, mapped) in [
            ("max_tokens", "max_tokens"),
            ("model_context_window_exceeded", "max_tokens"),
            ("guardrail_intervened", "refusal"),
            ("content_filtered", "refusal"),
            ("end_turn", "end_turn"),
            ("malformed_model_output", "malformed"),
        ] {
            assert_eq!(
                converse_response(&json!({"stopReason":reason}), "m")["stop_reason"],
                mapped
            );
        }
    }

    #[test]
    fn bedrock_credentials_follow_the_sdk_order() {
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir(home.path().join(".aws")).unwrap();
        std::fs::write(
            home.path().join(".aws/credentials"),
            "[default]\naws_access_key_id = AKIDDEFAULT\naws_secret_access_key = secret-default\n\n[work]\naws_access_key_id=AKIDWORK\naws_secret_access_key=secret-work\naws_session_token=token-work\n",
        )
        .unwrap();
        std::fs::write(
            home.path().join(".aws/config"),
            "[default]\nregion = eu-west-1\n[profile work]\nregion=ap-south-1\n",
        )
        .unwrap();
        let home = home.path().to_str().unwrap();
        let id = |auth: &AwsAuth| match auth {
            AwsAuth::Bearer(token) => format!("bearer:{token}"),
            AwsAuth::Signed(credentials) => credentials.access_key_id.clone(),
            AwsAuth::Program(command) => command.join(" "),
        };
        let cases = [
            (
                json!({"api_key":"bedrock-key"}),
                vec![
                    ("AWS_ACCESS_KEY_ID", "AKIDENV"),
                    ("AWS_SECRET_ACCESS_KEY", "s"),
                    ("AWS_REGION", "us-west-2"),
                ],
                "bearer:bedrock-key",
                "us-west-2",
            ),
            (
                json!({"api_key":"env:BEDROCK_KEY"}),
                vec![("BEDROCK_KEY", "from-env")],
                "bearer:from-env",
                "us-east-1",
            ),
            (
                json!({}),
                vec![
                    ("AWS_BEARER_TOKEN_BEDROCK", "t"),
                    ("AWS_ACCESS_KEY_ID", "AKIDENV"),
                    ("AWS_SECRET_ACCESS_KEY", "s"),
                ],
                "bearer:t",
                "us-east-1",
            ),
            (
                json!({}),
                vec![
                    ("AWS_ACCESS_KEY_ID", "AKIDENV"),
                    ("AWS_SECRET_ACCESS_KEY", "s"),
                    ("AWS_REGION", "ca-central-1"),
                    ("HOME", home),
                ],
                "AKIDENV",
                "ca-central-1",
            ),
            (json!({}), vec![("HOME", home)], "AKIDDEFAULT", "eu-west-1"),
            (
                json!({}),
                vec![("HOME", home), ("AWS_PROFILE", "work")],
                "AKIDWORK",
                "ap-south-1",
            ),
            // The region variables win over the profile's region.
            (
                json!({}),
                vec![
                    ("HOME", home),
                    ("AWS_PROFILE", "work"),
                    ("AWS_REGION_NAME", "sa-east-1"),
                ],
                "AKIDWORK",
                "sa-east-1",
            ),
        ];
        for (params, env, expected, region) in cases {
            let (auth, found) = bedrock_auth(&params, &vars(&env), true).unwrap();
            assert_eq!(
                (id(&auth).as_str(), found.as_str()),
                (expected, region),
                "{params} {env:?}"
            );
        }
        // Credentials that may not come from the environment are refused.
        assert!(
            bedrock_auth(
                &json!({}),
                &vars(&[("AWS_BEARER_TOKEN_BEDROCK", "t")]),
                false
            )
            .is_err()
        );
        assert!(
            bedrock_auth(
                &json!({"api_key":"k"}),
                &vars(&[("AWS_REGION", "../x")]),
                true
            )
            .is_err()
        );
        // Debug output never shows a secret.
        assert!(!format!("{:?}", EXAMPLE()).contains("wJalr"));
    }

    #[cfg(unix)]
    #[test]
    fn credential_process_profiles_run_their_program() {
        let home = tempfile::tempdir().unwrap();
        std::fs::create_dir(home.path().join(".aws")).unwrap();
        let script = home.path().join("creds.sh");
        std::fs::write(
            &script,
            "#!/bin/sh\necho '{\"Version\":1,\"AccessKeyId\":\"AKIDPROCESS\",\"SecretAccessKey\":\"s\",\"SessionToken\":\"t\"}'\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o700)).unwrap();
        std::fs::write(
            home.path().join(".aws/config"),
            format!(
                "[profile tool]\ncredential_process = {}\n",
                script.display()
            ),
        )
        .unwrap();
        let env = vars(&[
            ("HOME", home.path().to_str().unwrap()),
            ("AWS_PROFILE", "tool"),
        ]);
        let (auth, _) = bedrock_auth(&json!({}), &env, true).unwrap();
        let AwsAuth::Program(command) = auth else {
            panic!("a credential process is named, not run, while deployments are built");
        };
        let credentials = program_credentials(&command).unwrap();
        assert_eq!(credentials.access_key_id, "AKIDPROCESS");
        assert_eq!(credentials.session_token.as_deref(), Some("t"));
        // The second request reuses the answer.
        std::fs::remove_file(&script).unwrap();
        assert_eq!(
            program_credentials(&command).unwrap().access_key_id,
            "AKIDPROCESS"
        );
    }
}

#[cfg(test)]
mod curl_tests {
    use super::*;
    use std::io::{Read, Write};

    /// curl's own SigV4 (`--aws-sigv4`) signs a Bedrock-shaped request; the
    /// same request signed here with curl's timestamp must carry the same
    /// signature. Needs curl 7.75 or later on PATH.
    #[test]
    #[ignore = "runs the curl executable"]
    fn signatures_agree_with_curl() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let url = format!(
            "http://127.0.0.1:{port}/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse"
        );
        let body = r#"{"messages":[{"role":"user","content":[{"text":"Hello Ω"}]}]}"#;
        let child = std::process::Command::new("curl")
            .args([
                "-s",
                "--max-time",
                "5",
                "--aws-sigv4",
                "aws:amz:us-west-2:bedrock",
                "--user",
                "AKIDEXAMPLE:wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
                "-H",
                "content-type: application/json",
                "--data-binary",
                body,
                &url,
            ])
            .spawn()
            .unwrap();
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .unwrap();
        let mut bytes = Vec::new();
        let mut chunk = [0; 4096];
        while !String::from_utf8_lossy(&bytes).contains(body) {
            let count = stream.read(&mut chunk).unwrap();
            assert!(count > 0);
            bytes.extend_from_slice(&chunk[..count]);
        }
        stream
            .write_all(b"HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n")
            .unwrap();
        drop(stream);
        let _ = child.wait_with_output();
        let text = String::from_utf8_lossy(&bytes).into_owned();
        let header = |name: &str| {
            text.lines()
                .find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.eq_ignore_ascii_case(name)
                        .then(|| value.trim().to_owned())
                })
                .unwrap()
        };
        let date = header("x-amz-date");
        let authorization = header("authorization");
        let time: SystemTime = chrono::NaiveDateTime::parse_from_str(&date, "%Y%m%dT%H%M%SZ")
            .unwrap()
            .and_utc()
            .into();
        let signed = authorization
            .split("SignedHeaders=")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .to_owned();
        let mut headers = Vec::new();
        for name in signed.split(';') {
            if name != "x-amz-date" {
                headers.push((name.to_owned(), header(name)));
            }
        }
        let path = "/model/anthropic.claude-haiku-4-5-20251001-v1%3A0/converse";
        let ours = sign(
            "POST",
            path,
            "",
            &headers,
            body.as_bytes(),
            &AwsCredentials {
                access_key_id: "AKIDEXAMPLE".into(),
                secret_access_key: "wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY".into(),
                session_token: None,
            },
            "us-west-2",
            "bedrock",
            time,
        );
        let ours = &ours
            .iter()
            .find(|(name, _)| name == "authorization")
            .unwrap()
            .1;
        assert_eq!(ours, &authorization, "{text}");
    }
}
