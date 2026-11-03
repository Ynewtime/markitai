//! Durations, token counts, costs and paths as the UI shows them (DESIGN.md
//! 4.2 and 4.3).

use std::path::Path;

use crate::strings::{Lang, fill, text_in};

/// `18.6 s` below a minute, never less than `0.1 s`; `2 min 5 s` from there.
pub fn duration(lang: Lang, seconds: f64) -> String {
    let seconds = seconds.max(0.1);
    if seconds < 59.95 {
        let value = format!("{seconds:.1}");
        fill(&text_in(lang, "unit.seconds"), &[("value", &value)])
    } else {
        let total = seconds.round() as u64;
        fill(
            &text_in(lang, "unit.minutes"),
            &[
                ("minutes", &(total / 60).to_string()),
                ("seconds", &(total % 60).to_string()),
            ],
        )
    }
}

/// Digits grouped by three with commas, as both languages write them.
pub fn grouped(count: u64) -> String {
    let digits = count.to_string();
    let mut out = String::new();
    for (i, digit) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

/// USD with four decimals below $1, two from $1.
pub fn usd(cost: f64) -> String {
    if cost < 1.0 {
        format!("${cost:.4}")
    } else {
        format!("${cost:.2}")
    }
}

/// `m:ss` for the running conversion.
pub fn elapsed(seconds: u64) -> String {
    format!("{}:{:02}", seconds / 60, seconds % 60)
}

/// The path with the home folder written as `~`.
pub fn home_abbreviated(path: &Path, home: Option<&Path>) -> String {
    if let Some(home) = home.filter(|h| !h.as_os_str().is_empty() && *h != Path::new("/"))
        && let Ok(rest) = path.strip_prefix(home)
    {
        return if rest.as_os_str().is_empty() {
            "~".to_owned()
        } else {
            format!("~/{}", rest.to_string_lossy())
        };
    }
    path.to_string_lossy().into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durations_use_one_decimal_then_minutes() {
        assert_eq!(duration(Lang::En, 18.6), "18.6 s");
        assert_eq!(duration(Lang::ZhHans, 18.6), "18.6 秒");
        assert_eq!(duration(Lang::En, 0.04), "0.1 s");
        assert_eq!(duration(Lang::En, 0.0), "0.1 s");
        assert_eq!(duration(Lang::En, -1.0), "0.1 s");
        assert_eq!(duration(Lang::En, 0.16), "0.2 s");
        assert_eq!(duration(Lang::En, 59.96), "1 min 0 s");
        assert_eq!(duration(Lang::En, 125.4), "2 min 5 s");
        assert_eq!(duration(Lang::ZhHans, 125.4), "2 分 5 秒");
    }

    #[test]
    fn counts_costs_and_elapsed_time() {
        assert_eq!(grouped(24310), "24,310");
        assert_eq!(grouped(999), "999");
        assert_eq!(grouped(1_234_567), "1,234,567");
        assert_eq!(usd(0.0412), "$0.0412");
        assert_eq!(usd(1.5), "$1.50");
        assert_eq!(elapsed(65), "1:05");
        assert_eq!(elapsed(7), "0:07");
    }

    #[test]
    fn the_home_folder_is_a_tilde() {
        let home = Path::new("/home/example");
        assert_eq!(
            home_abbreviated(Path::new("/home/example/Documents/Markitai"), Some(home)),
            "~/Documents/Markitai"
        );
        assert_eq!(home_abbreviated(home, Some(home)), "~");
        assert_eq!(home_abbreviated(Path::new("/home/example-user/x"), Some(home)), "/home/example-user/x");
        assert_eq!(home_abbreviated(Path::new("/srv/x"), None), "/srv/x");
    }
}
