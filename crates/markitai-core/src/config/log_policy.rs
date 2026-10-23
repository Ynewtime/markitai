//! `log.rotation` and `log.retention` in the forms Python markitai's logger
//! (loguru) reads, so one configuration file means the same to both. The CLI
//! logger and configuration validation call the same parsers here, so a value
//! that validates is one the logger accepts.
use super::invalid;
use crate::Result;
use chrono::{
    Datelike, Days, Months, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, Timelike, Weekday,
};
use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;
use std::time::Duration;

/// When a session's log file is closed and its next one opened.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rotation {
    /// Before a record would take the file past this many bytes: `10 MB`.
    Size(u64),
    /// Once this long has passed since the file was opened: `1 week`.
    Interval(Duration),
    /// At the start of the next local hour, day, week (Monday), month or
    /// year: `daily`.
    Every(Period),
    /// At this local time every day, or every week on `weekday`: `00:00`,
    /// `monday at 12:00`, `sunday`.
    At {
        weekday: Option<Weekday>,
        time: NaiveTime,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Period {
    Hourly,
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

impl Rotation {
    /// The first local time after `opened` at which a file opened then is
    /// rotated, or `None` when time does not rotate it.
    pub fn next_after(&self, opened: NaiveDateTime) -> Option<NaiveDateTime> {
        let midnight = |date: NaiveDate| date.and_time(NaiveTime::MIN);
        let date = opened.date();
        match *self {
            Self::Size(_) => None,
            Self::Interval(interval) => {
                opened.checked_add_signed(TimeDelta::from_std(interval).ok()?)
            }
            Self::Every(Period::Hourly) => date
                .and_hms_opt(opened.hour(), 0, 0)?
                .checked_add_signed(TimeDelta::hours(1)),
            Self::Every(Period::Daily) => date.checked_add_days(Days::new(1)).map(midnight),
            Self::Every(Period::Weekly) => date
                .checked_add_days(Days::new(
                    7 - u64::from(date.weekday().num_days_from_monday()),
                ))
                .map(midnight),
            Self::Every(Period::Monthly) => date
                .with_day(1)?
                .checked_add_months(Months::new(1))
                .map(midnight),
            Self::Every(Period::Yearly) => {
                NaiveDate::from_ymd_opt(date.year() + 1, 1, 1).map(midnight)
            }
            Self::At { weekday, time } => (0..=7)
                .filter_map(|days| date.checked_add_days(Days::new(days)))
                .map(|day| day.and_time(time))
                .find(|at| *at > opened && weekday.is_none_or(|weekday| at.weekday() == weekday)),
        }
    }
}

/// Parse `log.rotation`: a size, an interval, a frequency, or a time of day
/// with an optional weekday.
pub fn rotation(value: &str) -> Result<Rotation> {
    let value = value.trim();
    let parsed = if let Some(bytes) = size(value) {
        Some(Rotation::Size(bytes))
    } else if let Some(seconds) = seconds(value) {
        (seconds > 0.0)
            .then(|| Duration::try_from_secs_f64(seconds).ok())
            .flatten()
            .map(Rotation::Interval)
    } else if let Some(period) = frequency(value) {
        Some(Rotation::Every(period))
    } else {
        daytime(value).map(|(weekday, time)| Rotation::At { weekday, time })
    };
    parsed.ok_or_else(|| {
        invalid(
            "log.rotation",
            &format!(
                "'{value}' is not a size ('10 MB'), an interval ('1 week'), a time of day \
                 ('00:00', 'monday at 12:00') or hourly/daily/weekly/monthly/yearly"
            ),
        )
    })
}

/// Parse `log.retention`: how long a finished session's logs are kept.
pub fn retention(value: &str) -> Result<Duration> {
    let value = value.trim();
    seconds(value)
        .filter(|seconds| *seconds >= 0.0)
        .and_then(|seconds| Duration::try_from_secs_f64(seconds).ok())
        .ok_or_else(|| {
            invalid(
                "log.retention",
                &format!("'{value}' is not a duration such as '7 days' or '1 week, 3 days'"),
            )
        })
}

/// Check a normalized `log` section's rotation and retention.
pub(super) fn validate(log: &Value) -> Result<()> {
    if let Some(value) = log.get("rotation").and_then(Value::as_str) {
        rotation(value)?;
    }
    if let Some(value) = log.get("retention").and_then(Value::as_str) {
        retention(value)?;
    }
    Ok(())
}

/// The digits, signs, points and exponents loguru reads as one number.
const NUMBER: &str = r"[eE+\-.0-9]+";

/// `10 MB`, `1.5 KiB`, `1e3 B`: positive bytes, in powers of 1000, or of 1024
/// with `i`. Loguru reads a lower-case `b` as bits; a byte size here is a byte
/// size in either case, as it always was in this logger.
fn size(value: &str) -> Option<u64> {
    static SIZE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(r"(?i)^({NUMBER})\s*([kmgtpezy])?(i)?b$")).expect("size pattern")
    });
    let captures = SIZE.captures(value)?;
    let number = captures[1].parse::<f64>().ok()?;
    let power = captures.get(2).map_or(0, |unit| {
        "kmgtpezy"
            .find(unit.as_str().to_ascii_lowercase().as_str())
            .map_or(0, |index| index as i32 + 1)
    });
    let base: f64 = if captures.get(3).is_some() {
        1024.0
    } else {
        1000.0
    };
    let bytes = number * base.powi(power);
    // A bigger size than a file can reach never rotates it.
    (bytes >= 1.0).then(|| bytes.min(u64::MAX as f64) as u64)
}

/// The longest duration loguru (Python's `timedelta`) accepts: 999999999 days.
const MAX_SECONDS: f64 = 999_999_999.0 * 86_400.0;

/// `7 days`, `1 week, 3 days`, `1h30min`: the sum in seconds of numbers and
/// units, which may be negative; `None` when it is not a finite duration.
fn seconds(value: &str) -> Option<f64> {
    static WHOLE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(r"(?i)^(?:{NUMBER}\s*[a-z]+[\s,]*)+$")).expect("duration pattern")
    });
    static PART: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(r"(?i)({NUMBER})\s*([a-z]+)[\s,]*")).expect("duration part")
    });
    if !WHOLE.is_match(value) {
        return None;
    }
    let mut total = 0.0;
    for part in PART.captures_iter(value) {
        let number = part[1].parse::<f64>().ok()?;
        let unit = match part[2].to_ascii_lowercase().as_str() {
            "y" | "year" | "years" => 31_536_000.0,
            "month" | "months" => 2_628_000.0,
            "w" | "week" | "weeks" => 604_800.0,
            "d" | "day" | "days" => 86_400.0,
            "h" | "hour" | "hours" => 3_600.0,
            "min" | "mins" | "minute" | "minutes" => 60.0,
            "s" | "sec" | "secs" | "second" | "seconds" => 1.0,
            "ms" | "millisecond" | "milliseconds" => 0.001,
            "us" | "microsecond" | "microseconds" => 0.000_001,
            _ => return None,
        };
        total += number * unit;
    }
    (total.is_finite() && total.abs() <= MAX_SECONDS).then_some(total)
}

fn frequency(value: &str) -> Option<Period> {
    match value.to_ascii_lowercase().as_str() {
        "hourly" => Some(Period::Hourly),
        "daily" => Some(Period::Daily),
        "weekly" => Some(Period::Weekly),
        "monthly" => Some(Period::Monthly),
        "yearly" => Some(Period::Yearly),
        _ => None,
    }
}

/// `monday at 12:00`, `w0 at 9 am`, `sunday` (at midnight) or `12:00` (every
/// day).
fn daytime(value: &str) -> Option<(Option<Weekday>, NaiveTime)> {
    static AT: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r"(?i)^(.*?)\s+at\s+(.*)$").expect("daytime pattern"));
    if let Some(parts) = AT.captures(value) {
        return Some((Some(weekday(&parts[1])?), clock(&parts[2])?));
    }
    if let Some(day) = weekday(value) {
        return Some((Some(day), NaiveTime::MIN));
    }
    clock(value).map(|time| (None, time))
}

/// `monday` … `sunday`, or `w0` (Monday) … `w6` (Sunday).
fn weekday(value: &str) -> Option<Weekday> {
    const DAYS: [Weekday; 7] = [
        Weekday::Mon,
        Weekday::Tue,
        Weekday::Wed,
        Weekday::Thu,
        Weekday::Fri,
        Weekday::Sat,
        Weekday::Sun,
    ];
    let value = value.trim().to_ascii_lowercase();
    let index = match value.as_str() {
        "monday" => 0,
        "tuesday" => 1,
        "wednesday" => 2,
        "thursday" => 3,
        "friday" => 4,
        "saturday" => 5,
        "sunday" => 6,
        _ => value
            .strip_prefix('w')
            .filter(|digits| !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))?
            .parse::<usize>()
            .ok()?,
    };
    DAYS.get(index).copied()
}

/// `0` … `23`, `12:00`, `12:00:30`, `12:00:30.5`, or a 12-hour time with
/// `am`/`pm`: `10 am`, `10:30 pm`.
fn clock(value: &str) -> Option<NaiveTime> {
    static CLOCK: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(
            r"(?i)^([0-9]{1,2})(?::([0-9]{1,2})(?::([0-9]{1,2})(?:\.([0-9]{1,6}))?)?)?\s*([ap]m)?$",
        )
        .expect("time pattern")
    });
    let parts = CLOCK.captures(value.trim())?;
    let number = |index: usize| {
        parts
            .get(index)
            .map_or(Some(0), |part| part.as_str().parse::<u32>().ok())
    };
    let mut hour = number(1)?;
    if let Some(half) = parts.get(5) {
        if !(1..=12).contains(&hour) {
            return None;
        }
        hour = hour % 12
            + if half.as_str().eq_ignore_ascii_case("pm") {
                12
            } else {
                0
            };
    }
    let micros = parts.get(4).map_or(Some(0), |fraction| {
        format!("{:0<6}", fraction.as_str()).parse::<u32>().ok()
    })?;
    NaiveTime::from_hms_micro_opt(hour, number(2)?, number(3)?, micros)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(text: &str) -> NaiveDateTime {
        NaiveDateTime::parse_from_str(text, "%Y-%m-%d %H:%M:%S").unwrap()
    }
    fn time(h: u32, m: u32, s: u32) -> NaiveTime {
        NaiveTime::from_hms_opt(h, m, s).unwrap()
    }

    #[test]
    fn rotation_reads_every_loguru_form() {
        for (value, expected) in [
            ("10 MB", Rotation::Size(10_000_000)),
            ("10MB", Rotation::Size(10_000_000)),
            ("10 mb", Rotation::Size(10_000_000)),
            ("1 KB", Rotation::Size(1_000)),
            ("1.5 KiB", Rotation::Size(1_536)),
            ("1e3 B", Rotation::Size(1_000)),
            ("2 GiB", Rotation::Size(2 << 30)),
            ("1 TB", Rotation::Size(1_000_000_000_000)),
            ("1 YB", Rotation::Size(u64::MAX)),
            ("1 week", Rotation::Interval(Duration::from_secs(604_800))),
            ("10 hours", Rotation::Interval(Duration::from_secs(36_000))),
            ("1 h 30 min", Rotation::Interval(Duration::from_secs(5_400))),
            (
                "1 week, 3 days",
                Rotation::Interval(Duration::from_secs(864_000)),
            ),
            ("1w3d", Rotation::Interval(Duration::from_secs(864_000))),
            (
                "1 month",
                Rotation::Interval(Duration::from_secs(2_628_000)),
            ),
            (
                "2 years",
                Rotation::Interval(Duration::from_secs(63_072_000)),
            ),
            ("500 ms", Rotation::Interval(Duration::from_millis(500))),
            ("1.5 h", Rotation::Interval(Duration::from_secs(5_400))),
            (" 1 s ", Rotation::Interval(Duration::from_secs(1))),
            ("hourly", Rotation::Every(Period::Hourly)),
            ("daily", Rotation::Every(Period::Daily)),
            ("Weekly", Rotation::Every(Period::Weekly)),
            ("monthly", Rotation::Every(Period::Monthly)),
            ("yearly", Rotation::Every(Period::Yearly)),
            (
                "00:00",
                Rotation::At {
                    weekday: None,
                    time: time(0, 0, 0),
                },
            ),
            (
                "12:00",
                Rotation::At {
                    weekday: None,
                    time: time(12, 0, 0),
                },
            ),
            (
                "12",
                Rotation::At {
                    weekday: None,
                    time: time(12, 0, 0),
                },
            ),
            (
                "10",
                Rotation::At {
                    weekday: None,
                    time: time(10, 0, 0),
                },
            ),
            (
                "1:5",
                Rotation::At {
                    weekday: None,
                    time: time(1, 5, 0),
                },
            ),
            (
                "12:00:30",
                Rotation::At {
                    weekday: None,
                    time: time(12, 0, 30),
                },
            ),
            (
                "12:00:30.5",
                Rotation::At {
                    weekday: None,
                    time: NaiveTime::from_hms_micro_opt(12, 0, 30, 500_000).unwrap(),
                },
            ),
            (
                "10 pm",
                Rotation::At {
                    weekday: None,
                    time: time(22, 0, 0),
                },
            ),
            (
                "12 AM",
                Rotation::At {
                    weekday: None,
                    time: time(0, 0, 0),
                },
            ),
            (
                "10:30 PM",
                Rotation::At {
                    weekday: None,
                    time: time(22, 30, 0),
                },
            ),
            (
                "monday at 12:00",
                Rotation::At {
                    weekday: Some(Weekday::Mon),
                    time: time(12, 0, 0),
                },
            ),
            (
                "Monday AT 12:00",
                Rotation::At {
                    weekday: Some(Weekday::Mon),
                    time: time(12, 0, 0),
                },
            ),
            (
                "sunday at 10 am",
                Rotation::At {
                    weekday: Some(Weekday::Sun),
                    time: time(10, 0, 0),
                },
            ),
            (
                "w2 at 13",
                Rotation::At {
                    weekday: Some(Weekday::Wed),
                    time: time(13, 0, 0),
                },
            ),
            (
                "sunday",
                Rotation::At {
                    weekday: Some(Weekday::Sun),
                    time: time(0, 0, 0),
                },
            ),
            (
                "w0",
                Rotation::At {
                    weekday: Some(Weekday::Mon),
                    time: time(0, 0, 0),
                },
            ),
            (
                "w6",
                Rotation::At {
                    weekday: Some(Weekday::Sun),
                    time: time(0, 0, 0),
                },
            ),
        ] {
            assert_eq!(rotation(value).unwrap(), expected, "{value}");
        }
    }

    #[test]
    fn rotation_refuses_what_loguru_refuses_and_what_would_never_end() {
        for value in [
            "midnight",
            "0 B",
            "0.5 B",
            "-1 KB",
            "NaN days",
            "25:00",
            "24",
            "13 pm",
            "0 am",
            "12:60",
            "w7",
            "monday at",
            "funday at 12:00",
            "monday at 25:00",
            "1 m",
            "1 e",
            "1e B",
            "0 s",
            "-1 days",
            "1 week, -7 days",
            "1e999 days",
            "10485760",
            "",
            "10 MB 1 s",
        ] {
            let error = rotation(value).unwrap_err().to_string();
            assert!(
                error.starts_with(&format!("log.rotation '{}' is not", value.trim())),
                "{value}: {error}"
            );
        }
    }

    #[test]
    fn retention_reads_loguru_durations() {
        for (value, seconds) in [
            ("7 days", 604_800),
            ("1 month", 2_628_000),
            ("1 week, 3 days", 864_000),
            ("1 week 3 days", 864_000),
            ("10 hours", 36_000),
            ("1 y", 31_536_000),
            ("90 min", 5_400),
            ("0 s", 0),
        ] {
            assert_eq!(
                retention(value).unwrap(),
                Duration::from_secs(seconds),
                "{value}"
            );
        }
        // A count of files, a size or a time of day is not a retention, in
        // loguru either.
        for value in [
            "10", "10 MB", "daily", "00:00", "NaN days", "-1 days", "1 m", "",
        ] {
            let error = retention(value).unwrap_err().to_string();
            assert!(error.starts_with("log.retention '"), "{value}: {error}");
        }
    }

    #[test]
    fn time_rotation_falls_on_the_next_local_boundary() {
        let opened = at("2026-10-21 10:15:30"); // a Wednesday
        let next = |value: &str| rotation(value).unwrap().next_after(opened).unwrap();
        assert_eq!(next("1 week"), at("2026-10-28 10:15:30"));
        assert_eq!(next("1 s"), at("2026-10-21 10:15:31"));
        assert_eq!(next("hourly"), at("2026-10-21 11:00:00"));
        assert_eq!(next("daily"), at("2026-10-22 00:00:00"));
        assert_eq!(next("weekly"), at("2026-10-26 00:00:00"));
        assert_eq!(next("monthly"), at("2026-11-01 00:00:00"));
        assert_eq!(next("yearly"), at("2027-01-01 00:00:00"));
        assert_eq!(next("12:00"), at("2026-10-21 12:00:00"));
        assert_eq!(next("10:15"), at("2026-10-22 10:15:00"));
        assert_eq!(next("10:15:30"), at("2026-10-22 10:15:30"));
        assert_eq!(next("wednesday at 23:00"), at("2026-10-21 23:00:00"));
        // Unlike loguru, a weekday rule never falls on another weekday.
        assert_eq!(next("sunday at 23:00"), at("2026-10-25 23:00:00"));
        assert_eq!(next("wednesday at 10:00"), at("2026-10-28 10:00:00"));
        assert_eq!(next("monday"), at("2026-10-26 00:00:00"));
        assert_eq!(rotation("10 MB").unwrap().next_after(opened), None);
        let december = at("2026-12-31 23:59:59");
        assert_eq!(
            Rotation::Every(Period::Monthly).next_after(december),
            Some(at("2027-01-01 00:00:00"))
        );
        assert_eq!(
            Rotation::Every(Period::Weekly).next_after(at("2026-10-26 00:00:00")),
            Some(at("2026-11-02 00:00:00"))
        );
    }

    #[test]
    fn configuration_validation_refuses_what_the_logger_refuses_with_its_message() {
        use serde_json::json;
        for (key, value) in [
            ("rotation", "midnight"),
            ("rotation", "25:00"),
            ("retention", "10"),
        ] {
            let message = if key == "rotation" {
                rotation(value).unwrap_err().to_string()
            } else {
                retention(value).unwrap_err().to_string()
            };
            let log = json!({ key: value });
            assert_eq!(
                super::super::normalize(&json!({ "log": log }))
                    .unwrap_err()
                    .to_string(),
                message
            );
            assert_eq!(
                super::super::normalize_model("LogConfig", &log)
                    .unwrap_err()
                    .to_string(),
                message
            );
        }
        let cfg = json!({"log":{"rotation":"1 week","retention":"1 month"}});
        assert_eq!(
            super::super::normalize(&cfg).unwrap()["log"]["rotation"],
            "1 week"
        );
        // The defaults are valid too.
        super::super::normalize(&json!({})).unwrap();
    }
}
