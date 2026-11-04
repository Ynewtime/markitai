//! Every visible string, from `desktop/strings.json` (DESIGN.md sections 7.5
//! and 10). The language is chosen once at startup.

use std::collections::HashMap;
use std::sync::OnceLock;

const STRINGS_JSON: &str = include_str!("../../../strings.json");

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En,
    ZhHans,
}

impl Lang {
    /// zh-Hans when the first preferred UI language starts with `zh`.
    pub fn from_preferred<'a>(mut names: impl Iterator<Item = &'a str>) -> Lang {
        match names.next() {
            Some(first) if first.starts_with("zh") => Lang::ZhHans,
            _ => Lang::En,
        }
    }

    /// The `LANGUAGE` that puts GTK and libadwaita's own words (the back
    /// button, window buttons, text menus) in this language too, or None
    /// when the system's first choice `first` already does.
    pub fn toolkit_language(self, first: Option<&str>) -> Option<&'static str> {
        let first = first.unwrap_or_default();
        match self {
            Lang::En if first.starts_with("en") => None,
            Lang::En => Some("en"),
            Lang::ZhHans if first.starts_with("zh_CN") => None,
            Lang::ZhHans => Some("zh_CN"),
        }
    }
}

static LANG: OnceLock<Lang> = OnceLock::new();

pub fn set_lang(lang: Lang) {
    let _ = LANG.set(lang);
}

pub fn lang() -> Lang {
    *LANG.get().unwrap_or(&Lang::En)
}

fn table() -> &'static HashMap<String, HashMap<String, String>> {
    static TABLE: OnceLock<HashMap<String, HashMap<String, String>>> = OnceLock::new();
    TABLE.get_or_init(|| serde_json::from_str(STRINGS_JSON).expect("desktop/strings.json is valid"))
}

/// The text of `key` in `lang`. An unknown key gives the key itself (a unit
/// test checks every key the source names).
pub fn text_in(lang: Lang, key: &str) -> String {
    let column = match lang {
        Lang::En => "en",
        Lang::ZhHans => "zh-Hans",
    };
    table()
        .get(key)
        .and_then(|row| row.get(column))
        .cloned()
        .unwrap_or_else(|| key.to_owned())
}

/// `{name}` placeholders replaced by their values.
pub fn fill(text: &str, args: &[(&str, &str)]) -> String {
    args.iter().fold(text.to_owned(), |text, (name, value)| {
        text.replace(&format!("{{{name}}}"), value)
    })
}

pub fn tr(key: &str) -> String {
    text_in(lang(), key)
}

pub fn trf(key: &str, args: &[(&str, &str)]) -> String {
    fill(&tr(key), args)
}

/// `<base>.one` when `count` is 1, else `<base>.other`, with `{count}` filled.
pub fn plural_in(lang: Lang, base: &str, count: usize) -> String {
    let key = format!("{base}.{}", if count == 1 { "one" } else { "other" });
    fill(&text_in(lang, &key), &[("count", &count.to_string())])
}

pub fn plural(base: &str, count: usize) -> String {
    plural_in(lang(), base, count)
}

/// `{settings}` in copy: GNOME's name for the settings place.
pub fn settings_name() -> String {
    tr("nav.preferences")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn keys_named(source: &str, call: &str) -> Vec<String> {
        let mut keys = Vec::new();
        let needle = format!("{call}(\"");
        let mut rest = source;
        while let Some(at) = rest.find(&needle) {
            let before = rest[..at].chars().last();
            rest = &rest[at + needle.len()..];
            if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                continue;
            }
            if let Some(end) = rest.find('"') {
                keys.push(rest[..end].to_owned());
            }
        }
        keys
    }

    fn sources(dir: &Path, out: &mut Vec<String>) {
        for entry in std::fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                sources(&path, out);
            } else if path.extension().is_some_and(|e| e == "rs") {
                out.push(std::fs::read_to_string(path).unwrap());
            }
        }
    }

    #[test]
    fn every_key_the_source_names_is_in_strings_json() {
        let mut files = Vec::new();
        sources(&Path::new(env!("CARGO_MANIFEST_DIR")).join("src"), &mut files);
        let mut checked = 0;
        for file in &files {
            for key in ["tr", "trf"]
                .iter()
                .flat_map(|call| keys_named(file, call))
            {
                assert!(table().contains_key(&key), "{key} is not in desktop/strings.json");
                checked += 1;
            }
            for base in keys_named(file, "plural") {
                for form in ["one", "other"] {
                    let key = format!("{base}.{form}");
                    assert!(table().contains_key(&key), "{key} is not in desktop/strings.json");
                }
                checked += 1;
            }
        }
        assert!(checked > 80, "only {checked} keys found; the scan is broken");
    }

    #[test]
    fn both_languages_fill_placeholders_and_plurals() {
        assert_eq!(plural_in(Lang::En, "state.warnings", 1), "1 warning");
        assert_eq!(plural_in(Lang::En, "state.warnings", 2), "2 warnings");
        assert_eq!(plural_in(Lang::ZhHans, "state.warnings", 3), "3 条警告");
        let done = fill(&text_in(Lang::ZhHans, "state.done"), &[("duration", "18.6 秒")]);
        assert_eq!(done, "已完成，用时 18.6 秒");
        assert_eq!(text_in(Lang::En, "no.such.key"), "no.such.key");
    }

    #[test]
    fn the_first_preferred_language_decides() {
        assert_eq!(Lang::from_preferred(["zh_CN.UTF-8", "en"].into_iter()), Lang::ZhHans);
        assert_eq!(Lang::from_preferred(["zh_TW", "C"].into_iter()), Lang::ZhHans);
        assert_eq!(Lang::from_preferred(["en_US", "zh_CN"].into_iter()), Lang::En);
        assert_eq!(Lang::from_preferred(["C"].into_iter()), Lang::En);
    }

    #[test]
    fn the_toolkit_follows_the_app_language() {
        // Traditional Chinese and German systems would show 關閉 and
        // Schließen beside the app's own 转换 and Convert.
        assert_eq!(Lang::ZhHans.toolkit_language(Some("zh_TW")), Some("zh_CN"));
        assert_eq!(Lang::ZhHans.toolkit_language(Some("zh")), Some("zh_CN"));
        assert_eq!(Lang::En.toolkit_language(Some("de_DE.UTF-8")), Some("en"));
        assert_eq!(Lang::En.toolkit_language(Some("C")), Some("en"));
        assert_eq!(Lang::En.toolkit_language(None), Some("en"));
        assert_eq!(Lang::ZhHans.toolkit_language(Some("zh_CN.UTF-8")), None);
        assert_eq!(Lang::En.toolkit_language(Some("en_GB")), None);
    }
}
