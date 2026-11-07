//! What the Failed detail says for each error code (DESIGN.md section 9).

/// The button after Retry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Second {
    Remove,
    OpenSettings,
    OpenConfig,
}

/// The sentence key and the second button for `code`.
pub fn error_copy(code: &str) -> (&'static str, Option<Second>) {
    use Second::*;
    match code {
        "not_found" => ("error.not_found", Some(Remove)),
        "is_directory" => ("error.is_directory", Some(Remove)),
        "no_model_configured" => ("error.no_model_configured", Some(OpenSettings)),
        "invalid_input" => ("error.invalid_input", None),
        "unsupported" => ("error.unsupported", Some(Remove)),
        "conversion_error" => ("error.conversion_error", None),
        "text_recognition_error" => ("error.text_recognition_error", None),
        "fetch_error" => ("error.fetch_error", None),
        "config_error" => ("error.config_error", Some(OpenConfig)),
        "io_error" => ("error.io_error", Some(OpenSettings)),
        "output_unavailable" => ("error.output_unavailable", Some(OpenSettings)),
        "image_only" => ("error.image_only", Some(OpenSettings)),
        "skipped" => ("error.skipped", None),
        "invalid_json" | "internal_error" => ("error.internal", None),
        _ => ("error.unknown", None),
    }
}

/// The copy key of each second button.
pub fn second_key(second: Second) -> &'static str {
    match second {
        Second::Remove => "convert.remove",
        Second::OpenSettings => "action.open_settings",
        Second::OpenConfig => "settings.config.open",
    }
}

/// Every code the table names.
#[cfg(test)]
pub const CODES: [&str; 16] = [
    "not_found",
    "is_directory",
    "no_model_configured",
    "invalid_input",
    "unsupported",
    "conversion_error",
    "text_recognition_error",
    "fetch_error",
    "config_error",
    "io_error",
    "output_unavailable",
    "image_only",
    "skipped",
    "invalid_json",
    "internal_error",
    "something_new",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strings::{Lang, text_in};

    #[test]
    fn every_code_has_copy_in_both_languages() {
        for code in CODES {
            let (key, second) = error_copy(code);
            for lang in [Lang::En, Lang::ZhHans] {
                assert_ne!(text_in(lang, key), key, "{code}: {key} missing");
                if let Some(second) = second {
                    let key = second_key(second);
                    assert_ne!(text_in(lang, key), key, "{code}: {key} missing");
                }
            }
        }
        assert_eq!(error_copy("something_new").0, "error.unknown");
        assert_eq!(
            error_copy("text_recognition_error").0,
            "error.text_recognition_error"
        );
        assert_eq!(error_copy("conversion_error").0, "error.conversion_error");
        assert_eq!(error_copy("invalid_json").0, "error.internal");
    }
}
