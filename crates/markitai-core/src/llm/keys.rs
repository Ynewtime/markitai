//! API keys checked before anything is sent, and named when refused.
//!
//! A deployment whose provider needs a key that nothing supplies would be
//! sent without one, refused, and sent again. It is left out of the pool
//! instead, with one warning per process that names the variable to set;
//! with no deployment left, the conversion fails with that message. A key
//! the provider refuses is reported with the variable that held it.
use super::Deployment;
use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// A deployment left out of the pool because a key it needs is not set.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct MissingKey {
    /// The model as configured (`deepseek/deepseek-flash`).
    pub model: String,
    /// The variable to set; `None` when the deployment reads no environment
    /// (its key must be written in `litellm_params.api_key`).
    pub variable: Option<String>,
}

/// Whether an `env:` reference names a variable, not a pasted key, so it can
/// be shown.
pub(super) fn variable_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

/// The command that sets a variable in the user's shell: PowerShell's on
/// Windows (its default terminal), `export` elsewhere.
pub fn set_command(variable: &str) -> String {
    if cfg!(windows) {
        format!("$env:{variable}=\"your_key\"")
    } else {
        format!("export {variable}=your_key")
    }
}

/// The `.env` file every front end also reads keys from.
pub fn dotenv_file() -> String {
    if crate::config::variable("MARKITAI_HOME").is_some() {
        crate::config::home().join(".env").display().to_string()
    } else {
        "~/.markitai/.env".into()
    }
}

fn not_set(item: &MissingKey) -> String {
    match &item.variable {
        Some(variable) => format!("{variable} is not set"),
        None => "it has no API key (set litellm_params.api_key)".into(),
    }
}

fn hint(variable: &str) -> String {
    format!(
        "set it with: {} (or add {variable}=your_key to {})",
        set_command(variable),
        dotenv_file()
    )
}

/// The warning for one deployment left out of a pool that still has others.
pub fn left_out_message(item: &MissingKey) -> String {
    match &item.variable {
        Some(variable) => format!(
            "{} is left out of the model pool: {}; {}",
            item.model,
            not_set(item),
            hint(variable)
        ),
        None => format!(
            "{} is left out of the model pool: {}",
            item.model,
            not_set(item)
        ),
    }
}

/// The error for a pool none of whose deployments has the key it needs.
pub fn no_usable_model_message(missing: &[MissingKey]) -> String {
    let mut variables: Vec<&Option<String>> = Vec::new();
    for item in missing {
        if !variables.contains(&&item.variable) {
            variables.push(&item.variable);
        }
    }
    let models = |variable: &Option<String>| {
        missing
            .iter()
            .filter(|item| &item.variable == variable)
            .map(|item| item.model.as_str())
            .collect::<Vec<_>>()
    };
    if let [variable] = variables.as_slice() {
        let models = models(variable);
        let needs = if models.len() == 1 { "needs" } else { "need" };
        return match variable {
            Some(name) => format!(
                "{name} is not set, and {} {needs} it; {}",
                models.join(", "),
                hint(name)
            ),
            None => format!(
                "No LLM model can be used: {} {} an API key; set litellm_params.api_key",
                models.join(", "),
                needs
            ),
        };
    }
    let parts: Vec<String> = variables
        .iter()
        .map(|variable| {
            let item = MissingKey {
                model: String::new(),
                variable: (*variable).clone(),
            };
            format!("{} ({})", not_set(&item), models(variable).join(", "))
        })
        .collect();
    let example = variables
        .iter()
        .find_map(|variable| variable.as_deref())
        .map(|first| {
            format!(
                " Set one of them, for example {} (or in {})",
                set_command(first),
                dotenv_file()
            )
        })
        .unwrap_or_default();
    format!("No LLM model can be used: {}.{example}", parts.join("; "))
}

fn reported() -> &'static Mutex<HashSet<MissingKey>> {
    static REPORTED: OnceLock<Mutex<HashSet<MissingKey>>> = OnceLock::new();
    REPORTED.get_or_init(Default::default)
}

/// Records deployments whose missing key the caller has already shown (the
/// command line says so once before converting).
pub fn mark_reported(items: &[MissingKey]) {
    let mut seen = reported().lock().unwrap_or_else(|e| e.into_inner());
    seen.extend(items.iter().cloned());
}

/// Warns once per process for each deployment left out of the pool.
pub(super) fn report_left_out(items: &[MissingKey]) {
    for item in items {
        let fresh = reported()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(item.clone());
        if fresh {
            super::note_document_warning(left_out_message(item));
        }
    }
}

/// Whether a refusal is the provider rejecting the API key: HTTP 401, or
/// Gemini's HTTP 400 `API_KEY_INVALID`.
pub(super) fn key_refused(status: u16, lowercase_body: &str) -> bool {
    status == 401
        || status == 400
            && (lowercase_body.contains("api_key_invalid")
                || lowercase_body.contains("api key not valid"))
}

/// What to do about a refused key, before the provider's own words: the
/// variable that held it, or the configured key. A keyless request to a
/// server that wants a key names the variable that would hold one. A key
/// refused at one region of a provider that issues keys per region (see
/// `providers::regions`) also names the other regions' endpoints.
pub(super) fn refused_hint(entry: &Deployment) -> String {
    let hint = key_hint(entry);
    match entry.key.as_ref().and(super::providers::region_hint(
        &entry.provider,
        &entry.endpoint,
    )) {
        Some(region) => format!("{hint}. {region}"),
        None => hint,
    }
}

fn key_hint(entry: &Deployment) -> String {
    let variable = entry
        .key_variable
        .as_deref()
        .filter(|name| variable_name(name));
    match (&entry.key, variable) {
        (Some(_), Some(variable)) => format!(
            "the API key in {variable} was refused; set a valid one with: {}",
            set_command(variable)
        ),
        (Some(_), None) => {
            "the api_key configured for this model was refused; check that key".into()
        }
        (None, Some(variable)) => format!(
            "the server asks for an API key and none was sent; set one with: {}",
            set_command(variable)
        ),
        (None, None) => {
            "the server asks for an API key and none was sent; set litellm_params.api_key".into()
        }
    }
}
