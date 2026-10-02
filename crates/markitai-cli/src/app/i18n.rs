//! English or Chinese wording for the terminal text of `doctor`, `cache`,
//! `config path/validate`, `init`, the conversion and batch lines on stderr and
//! `--help`.
//!
//! A nonempty `MARKITAI_LANG` decides on its own; otherwise the first of
//! `LC_ALL`, `LC_MESSAGES` and `LANG` that is set to something other than the
//! `C`/`POSIX` locale decides, in the order POSIX gives them (the reference
//! reads `LANG` before `LC_ALL`, so a user who exports `LC_ALL` got English).
//! A value that starts with `zh` in any case selects Chinese; every other
//! value, and no value, selects English. Variables come from the process first
//! and then from the same `.env` files that configuration selection reads.
//!
//! Only sentences meant for a person change. JSON, values shared with JSON
//! (check names, messages and hints), usage errors from the argument parser,
//! the file log and exit codes stay as they are in every language.
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lang {
    En,
    Zh,
}

/// The language for this process, resolved once on first use.
pub(crate) fn lang() -> Lang {
    static LANG: OnceLock<Lang> = OnceLock::new();
    *LANG.get_or_init(|| detect(&markitai_core::config::environment()))
}

fn detect(vars: &HashMap<String, String>) -> Lang {
    let chosen = ["MARKITAI_LANG", "LC_ALL", "LC_MESSAGES", "LANG"]
        .into_iter()
        .find_map(|name| {
            vars.get(name).filter(|value| {
                // The C locale names no language; it never hides a later choice.
                !value.is_empty() && (name == "MARKITAI_LANG" || !is_c_locale(value))
            })
        });
    match chosen.and_then(|value| value.get(..2)) {
        Some(prefix) if prefix.eq_ignore_ascii_case("zh") => Lang::Zh,
        _ => Lang::En,
    }
}

/// `C`, `POSIX` and their encoding variants such as `C.UTF-8`.
fn is_c_locale(value: &str) -> bool {
    value == "C" || value == "POSIX" || value.starts_with("C.") || value.starts_with("POSIX.")
}

/// Formats the English or the Chinese literal for the active language, or for
/// an explicit one with `text!(lang => …)`. Both literals capture the caller's
/// variables, e.g. `text!("{n} ok", "{n} 项正常")`.
macro_rules! text {
    ($lang:expr => $en:literal, $zh:literal $(,)?) => {
        match $lang {
            $crate::app::i18n::Lang::En => format!($en),
            $crate::app::i18n::Lang::Zh => format!($zh),
        }
    };
    ($en:literal, $zh:literal $(,)?) => {
        text!($crate::app::i18n::lang() => $en, $zh)
    };
}

// Also nameable by path (`crate::app::i18n::text`) for modules outside `app`.
pub(crate) use text;

#[cfg(test)]
mod tests {
    use super::*;

    fn detected(pairs: &[(&str, &str)]) -> Lang {
        detect(
            &pairs
                .iter()
                .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
                .collect(),
        )
    }

    #[test]
    fn markitai_lang_then_the_locale_variables_choose_chinese_by_prefix() {
        assert_eq!(detected(&[]), Lang::En);
        assert_eq!(detected(&[("MARKITAI_LANG", "zh")]), Lang::Zh);
        assert_eq!(detected(&[("MARKITAI_LANG", "ZH_cn")]), Lang::Zh);
        assert_eq!(detected(&[("LANG", "zh_CN.UTF-8")]), Lang::Zh);
        assert_eq!(detected(&[("LC_ALL", "zh_TW.UTF-8")]), Lang::Zh);
        assert_eq!(detected(&[("LANG", "en_US.UTF-8")]), Lang::En);
        // A nonempty MARKITAI_LANG wins even when it names another language.
        assert_eq!(
            detected(&[("MARKITAI_LANG", "en"), ("LANG", "zh_CN.UTF-8")]),
            Lang::En
        );
        assert_eq!(
            detected(&[("MARKITAI_LANG", "fr"), ("LANG", "zh_CN.UTF-8")]),
            Lang::En
        );
        // An empty value is skipped.
        assert_eq!(
            detected(&[("MARKITAI_LANG", ""), ("LANG", "zh_CN.UTF-8")]),
            Lang::Zh
        );
        // LC_ALL, then LC_MESSAGES, then LANG, as POSIX orders them.
        assert_eq!(
            detected(&[("LANG", "en_US.UTF-8"), ("LC_ALL", "zh_CN.UTF-8")]),
            Lang::Zh
        );
        assert_eq!(
            detected(&[("LC_ALL", "en_US.UTF-8"), ("LANG", "zh_CN.UTF-8")]),
            Lang::En
        );
        assert_eq!(
            detected(&[("LC_MESSAGES", "zh_CN.UTF-8"), ("LANG", "en_US.UTF-8")]),
            Lang::Zh
        );
        assert_eq!(
            detected(&[("LC_ALL", "zh_CN.UTF-8"), ("LC_MESSAGES", "en_US.UTF-8")]),
            Lang::Zh
        );
        assert_eq!(
            detected(&[("LANG", ""), ("LC_ALL", "zh_CN.UTF-8")]),
            Lang::Zh
        );
        // The C locale names no language and does not hide a later one.
        for c in ["C", "POSIX", "C.UTF-8"] {
            assert_eq!(
                detected(&[("LC_ALL", c), ("LANG", "zh_CN.UTF-8")]),
                Lang::Zh,
                "{c}"
            );
            assert_eq!(
                detected(&[("LC_ALL", "zh_CN.UTF-8"), ("LANG", c)]),
                Lang::Zh
            );
            assert_eq!(detected(&[("LANG", c)]), Lang::En);
        }
        // MARKITAI_LANG is taken as written, even when it is `C`.
        assert_eq!(
            detected(&[("MARKITAI_LANG", "C"), ("LANG", "zh_CN.UTF-8")]),
            Lang::En
        );
        // Only a leading zh counts; nothing is trimmed.
        assert_eq!(detected(&[("MARKITAI_LANG", " zh")]), Lang::En);
        assert_eq!(detected(&[("MARKITAI_LANG", "z")]), Lang::En);
        assert_eq!(detected(&[("MARKITAI_LANG", "中文")]), Lang::En);
    }
}
