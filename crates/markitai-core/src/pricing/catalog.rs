//! Reviewed token tariffs; values are exact picodollars per token.
//!
//! Source rows, original notice and verification URLs are kept under
//! licenses/pricing. These public-list estimates are not invoice settlement.
use super::Provider;

pub(super) const SNAPSHOT: &str = "litellm-1.106.0.dev2-selected-2026-10-11";

#[derive(Clone, Copy)]
pub(super) struct Rates {
    pub input: u128,
    pub output: u128,
    pub cache_read: u128,
    /// The default prompt-cache write: Anthropic's 5-minute entry, OpenAI's
    /// 30-minute one.
    pub cache_write: Option<u128>,
    pub cache_write_1h: Option<u128>,
}
impl Rates {
    const fn new(
        input: u128,
        output: u128,
        cache_read: u128,
        cache_write: Option<u128>,
        cache_write_1h: Option<u128>,
    ) -> Self {
        Self {
            input,
            output,
            cache_read,
            cache_write,
            cache_write_1h,
        }
    }
    const fn plain(input: u128, output: u128, cache_read: u128) -> Self {
        Self::new(input, output, cache_read, None, None)
    }
    // The reviewed providers document this discount for these specific rows,
    // including prompt caching. It is never applied to a document aggregate.
    pub fn batch(self) -> Self {
        Self {
            input: self.input / 2,
            output: self.output / 2,
            cache_read: self.cache_read / 2,
            cache_write: self.cache_write.map(|value| value / 2),
            cache_write_1h: self.cache_write_1h.map(|value| value / 2),
        }
    }
}

/// Rates that replace the base ones for a whole request whose total input,
/// cache reads and writes included, is over `above` tokens.
#[derive(Clone, Copy)]
pub(super) struct LongContext {
    pub above: u64,
    pub rates: Rates,
}

#[derive(Clone, Copy)]
pub(super) struct Tariff {
    pub family: &'static str,
    pub rates: Rates,
    pub long: Option<LongContext>,
    /// The model's context window when it has no listed rates beyond it:
    /// larger input is not something the API accepts, so it stays unpriced.
    pub max_priced_input: Option<u64>,
    /// A dated list price: requests that end at or after this Unix second are
    /// unpriced.
    pub until: Option<u64>,
    /// DeepSeek's rates outside its peak hours; `rates` are the peak ones.
    pub off_peak: Option<Rates>,
    /// Whether the provider's Batch discount is reviewed for this row.
    pub batch: bool,
}
impl Tariff {
    const fn flat(family: &'static str, rates: Rates) -> Self {
        Self {
            family,
            rates,
            long: None,
            max_priced_input: None,
            until: None,
            off_peak: None,
            batch: true,
        }
    }
    /// OpenAI's current models: a prompt over "272K" tokens takes the long
    /// rates for the whole request. OpenAI's K is 1,000: its pricing page
    /// splits short (≤272K) from long (>272K) input, the GPT-5 page that the
    /// figure comes from lists "Maximum input tokens: 272,000", and the same
    /// model pages write a 1,050,000-token window as "1.05M".
    const fn openai_272k(family: &'static str, short: Rates, long: Rates) -> Self {
        Self {
            long: Some(LongContext {
                above: 272_000,
                rates: long,
            }),
            ..Self::flat(family, short)
        }
    }
    /// A Claude model with 5-minute and 1-hour cache writes.
    const fn claude(
        family: &'static str,
        input: u128,
        output: u128,
        cache_read: u128,
        write_5m: u128,
        write_1h: u128,
    ) -> Self {
        Self::flat(
            family,
            Rates::new(input, output, cache_read, Some(write_5m), Some(write_1h)),
        )
    }
    /// Gemini and DeepSeek rows: markitai submits no Batch to these providers.
    const fn standard_only(family: &'static str, rates: Rates) -> Self {
        Self {
            batch: false,
            ..Self::flat(family, rates)
        }
    }
}

/// 2027-01-01T00:00:00Z: Google lists Gemini 3.8 and 3.6 Flash prices
/// "through December 31, 2026" and doubles them from January 1, 2027.
const GEMINI_FLASH_INTRODUCTORY_UNTIL: u64 = 1_798_761_600;

/// DeepSeek bills Beijing time 09:00-12:00 and 14:00-18:00, Monday to Friday,
/// at its peak rates, and every other hour, weekends and Chinese public
/// holidays included, at the off-peak half. A weekday peak hour is only known
/// to be one on a day the State Council's holiday schedule is reviewed for:
/// these Beijing dates, as days since 1970-01-01, hold no public holiday
/// (2026-10-08 to 2026-12-31, from the 2026 schedule).
const DEEPSEEK_REVIEWED_DAYS: std::ops::RangeInclusive<u64> = 20_734..=20_818;

/// Whether DeepSeek bills a request ending at Unix second `at` at its peak
/// rates; `None` for a weekday peak hour on an unreviewed date.
pub(super) fn deepseek_peak(at: u64) -> Option<bool> {
    let beijing = at + 8 * 3600;
    let (day, second) = (beijing / 86_400, beijing % 86_400);
    // 1970-01-01 was a Thursday; 0 is Monday.
    let weekend = (day + 3) % 7 >= 5;
    let hours = (9 * 3600..12 * 3600).contains(&second) || (14 * 3600..18 * 3600).contains(&second);
    if weekend || !hours {
        Some(false)
    } else if DEEPSEEK_REVIEWED_DAYS.contains(&day) {
        Some(true)
    } else {
        None
    }
}

pub(super) fn lookup(provider: Provider, model: &str) -> Option<Tariff> {
    let tariff = match (provider, model) {
        // OpenAI's prompt caching guide: models before GPT-5.6 have "no
        // additional cache-write charge", and each input token takes the
        // uncached, cached or cache-write rate, so a write costs ordinary input.
        (Provider::OpenAi, "gpt-4.1" | "gpt-4.1-2025-04-14") => Tariff::flat(
            "openai/gpt-4.1",
            Rates::new(2_000_000, 8_000_000, 500_000, Some(2_000_000), None),
        ),
        (Provider::OpenAi, "gpt-4.1-mini" | "gpt-4.1-mini-2025-04-14") => Tariff::flat(
            "openai/gpt-4.1-mini",
            Rates::new(400_000, 1_600_000, 100_000, Some(400_000), None),
        ),
        (Provider::OpenAi, "gpt-4.1-nano" | "gpt-4.1-nano-2025-04-14") => Tariff::flat(
            "openai/gpt-4.1-nano",
            Rates::new(100_000, 400_000, 25_000, Some(100_000), None),
        ),
        (Provider::OpenAi, "gpt-6-astra") => Tariff::openai_272k(
            "openai/gpt-6-astra",
            Rates::new(10_000_000, 50_000_000, 1_000_000, Some(12_500_000), None),
            Rates::new(20_000_000, 75_000_000, 2_000_000, Some(25_000_000), None),
        ),
        (Provider::OpenAi, "gpt-6.1-sol") => Tariff::openai_272k(
            "openai/gpt-6.1-sol",
            Rates::new(2_000_000, 10_000_000, 100_000, Some(2_500_000), None),
            Rates::new(4_000_000, 15_000_000, 200_000, Some(5_000_000), None),
        ),
        (Provider::OpenAi, "gpt-6-sol") => Tariff::openai_272k(
            "openai/gpt-6-sol",
            Rates::new(2_000_000, 10_000_000, 200_000, Some(2_500_000), None),
            Rates::new(4_000_000, 15_000_000, 400_000, Some(5_000_000), None),
        ),
        (Provider::OpenAi, "gpt-6-luna") => Tariff::openai_272k(
            "openai/gpt-6-luna",
            Rates::new(100_000, 500_000, 10_000, Some(125_000), None),
            Rates::new(200_000, 750_000, 20_000, Some(250_000), None),
        ),
        // OpenAI's model page: "The gpt-5.6 alias routes requests to GPT-5.6 Sol."
        (Provider::OpenAi, "gpt-5.6-sol" | "gpt-5.6") => Tariff::openai_272k(
            "openai/gpt-5.6-sol",
            Rates::new(4_000_000, 20_000_000, 400_000, Some(5_000_000), None),
            Rates::new(8_000_000, 30_000_000, 800_000, Some(10_000_000), None),
        ),
        (Provider::OpenAi, "gpt-5.6-terra") => Tariff::openai_272k(
            "openai/gpt-5.6-terra",
            Rates::new(2_000_000, 12_000_000, 200_000, Some(2_500_000), None),
            Rates::new(4_000_000, 18_000_000, 400_000, Some(5_000_000), None),
        ),
        (Provider::OpenAi, "gpt-5.6-luna") => Tariff::openai_272k(
            "openai/gpt-5.6-luna",
            Rates::new(200_000, 1_200_000, 20_000, Some(250_000), None),
            Rates::new(400_000, 1_800_000, 40_000, Some(500_000), None),
        ),
        // Claude 4.6 and later models other than Haiku 5.5 keep these rates
        // across their whole 1M-token context window.
        (Provider::Anthropic, "claude-fable-5-1") => Tariff::claude(
            "anthropic/claude-fable-5-1",
            10_000_000,
            50_000_000,
            250_000,
            12_500_000,
            20_000_000,
        ),
        (Provider::Anthropic, "claude-fable-5") => Tariff::claude(
            "anthropic/claude-fable-5",
            10_000_000,
            50_000_000,
            1_000_000,
            12_500_000,
            20_000_000,
        ),
        (Provider::Anthropic, "claude-opus-5-5") => Tariff::claude(
            "anthropic/claude-opus-5-5",
            4_000_000,
            20_000_000,
            200_000,
            5_000_000,
            8_000_000,
        ),
        (Provider::Anthropic, "claude-opus-5") => Tariff::claude(
            "anthropic/claude-opus-5",
            5_000_000,
            25_000_000,
            500_000,
            6_250_000,
            10_000_000,
        ),
        (Provider::Anthropic, "claude-opus-4-8") => Tariff::claude(
            "anthropic/claude-opus-4-8",
            5_000_000,
            25_000_000,
            500_000,
            6_250_000,
            10_000_000,
        ),
        (Provider::Anthropic, "claude-opus-4-7") => Tariff::claude(
            "anthropic/claude-opus-4-7",
            5_000_000,
            25_000_000,
            500_000,
            6_250_000,
            10_000_000,
        ),
        (Provider::Anthropic, "claude-opus-4-6") => Tariff::claude(
            "anthropic/claude-opus-4-6",
            5_000_000,
            25_000_000,
            500_000,
            6_250_000,
            10_000_000,
        ),
        // The Claude 4.5 models have a 200K-token context window on the Claude
        // API (Sonnet 4.5's 1M beta and its long-context rates are withdrawn).
        (Provider::Anthropic, "claude-opus-4-5" | "claude-opus-4-5-20251101") => Tariff {
            max_priced_input: Some(200_000),
            ..Tariff::claude(
                "anthropic/claude-opus-4-5-20251101",
                5_000_000,
                25_000_000,
                500_000,
                6_250_000,
                10_000_000,
            )
        },
        (Provider::Anthropic, "claude-sonnet-5-5") => Tariff::claude(
            "anthropic/claude-sonnet-5-5",
            2_000_000,
            10_000_000,
            100_000,
            2_500_000,
            4_000_000,
        ),
        (Provider::Anthropic, "claude-sonnet-5") => Tariff::claude(
            "anthropic/claude-sonnet-5",
            2_000_000,
            10_000_000,
            200_000,
            2_500_000,
            4_000_000,
        ),
        (Provider::Anthropic, "claude-sonnet-4-6") => Tariff::claude(
            "anthropic/claude-sonnet-4-6",
            3_000_000,
            15_000_000,
            300_000,
            3_750_000,
            6_000_000,
        ),
        (Provider::Anthropic, "claude-sonnet-4-5" | "claude-sonnet-4-5-20250929") => Tariff {
            max_priced_input: Some(200_000),
            ..Tariff::claude(
                "anthropic/claude-sonnet-4-5-20250929",
                3_000_000,
                15_000_000,
                300_000,
                3_750_000,
                6_000_000,
            )
        },
        (Provider::Anthropic, "claude-haiku-4-5" | "claude-haiku-4-5-20251001") => Tariff {
            max_priced_input: Some(200_000),
            ..Tariff::claude(
                "anthropic/claude-haiku-4-5-20251001",
                1_000_000,
                5_000_000,
                100_000,
                1_250_000,
                2_000_000,
            )
        },
        (Provider::Anthropic, "claude-haiku-5-5") => Tariff {
            long: Some(LongContext {
                above: 100_000,
                rates: Rates::new(500_000, 2_500_000, 50_000, Some(625_000), Some(1_000_000)),
            }),
            ..Tariff::claude(
                "anthropic/claude-haiku-5-5",
                100_000,
                500_000,
                10_000,
                125_000,
                200_000,
            )
        },
        // Paid tier, text and image input. A `-latest` alias moves with
        // Google's releases; it is priced as its own family at the rates its
        // source row and Google's current Flash(-Lite) list give.
        (Provider::Gemini, "gemini-3.5-flash-lite") => Tariff::standard_only(
            "gemini/gemini-3.5-flash-lite",
            Rates::plain(300_000, 2_500_000, 30_000),
        ),
        (Provider::Gemini, "gemini-flash-lite-latest") => Tariff::standard_only(
            "gemini/gemini-flash-lite-latest",
            Rates::plain(300_000, 2_500_000, 30_000),
        ),
        (Provider::Gemini, "gemini-3.1-flash-lite") => Tariff::standard_only(
            "gemini/gemini-3.1-flash-lite",
            Rates::plain(250_000, 1_500_000, 25_000),
        ),
        (
            Provider::Gemini,
            model @ ("gemini-3.8-flash" | "gemini-3.6-flash" | "gemini-flash-latest"),
        ) => Tariff {
            until: Some(GEMINI_FLASH_INTRODUCTORY_UNTIL),
            ..Tariff::standard_only(
                match model {
                    "gemini-3.8-flash" => "gemini/gemini-3.8-flash",
                    "gemini-3.6-flash" => "gemini/gemini-3.6-flash",
                    _ => "gemini/gemini-flash-latest",
                },
                Rates::plain(750_000, 3_750_000, 75_000),
            )
        },
        // DeepSeek's pricing page: the legacy deepseek-v4-flash name is
        // served by the Flash model and billed at the Flash price.
        (Provider::DeepSeek, "deepseek-flash" | "deepseek-v4-flash") => Tariff {
            off_peak: Some(Rates::plain(150_000, 600_000, 3_000)),
            ..Tariff::standard_only(
                "deepseek/deepseek-flash",
                Rates::plain(300_000, 1_200_000, 6_000),
            )
        },
        (Provider::DeepSeek, "deepseek-v4-pro") => Tariff {
            off_peak: Some(Rates::plain(660_000, 1_980_000, 22_000)),
            ..Tariff::standard_only(
                "deepseek/deepseek-v4-pro",
                Rates::plain(1_320_000, 3_960_000, 44_000),
            )
        },
        _ => return None,
    };
    Some(tariff)
}

/// `(max_output_tokens, max_input_tokens)` of a reviewed row
/// (`licenses/pricing/source-rows.json`): the output cap a request asks for
/// when none is configured, as Python markitai does from the same catalog.
pub(super) fn limits(provider: Provider, model: &str) -> Option<(u64, u64)> {
    Some(match (provider, model) {
        (
            Provider::OpenAi,
            "gpt-4.1"
            | "gpt-4.1-2025-04-14"
            | "gpt-4.1-mini"
            | "gpt-4.1-mini-2025-04-14"
            | "gpt-4.1-nano"
            | "gpt-4.1-nano-2025-04-14",
        ) => (32_768, 1_047_576),
        (
            Provider::OpenAi,
            "gpt-6-luna" | "gpt-6-astra" | "gpt-6.1-sol" | "gpt-6-sol" | "gpt-5.6-sol" | "gpt-5.6"
            | "gpt-5.6-terra" | "gpt-5.6-luna",
        ) => (128_000, 922_000),
        (Provider::Anthropic, "claude-sonnet-4-5" | "claude-sonnet-4-5-20250929") => {
            (64_000, 1_000_000)
        }
        (
            Provider::Anthropic,
            "claude-haiku-4-5" | "claude-haiku-4-5-20251001" | "claude-opus-4-5-20251101",
        ) => (64_000, 200_000),
        (
            Provider::Anthropic,
            "claude-haiku-5-5" | "claude-fable-5-1" | "claude-fable-5" | "claude-opus-5-5"
            | "claude-opus-5" | "claude-opus-4-8" | "claude-opus-4-7" | "claude-opus-4-6"
            | "claude-sonnet-5-5" | "claude-sonnet-5" | "claude-sonnet-4-6",
        ) => (128_000, 1_000_000),
        (
            Provider::Gemini,
            "gemini-3.5-flash-lite"
            | "gemini-flash-lite-latest"
            | "gemini-3.1-flash-lite"
            | "gemini-3.8-flash"
            | "gemini-3.6-flash"
            | "gemini-flash-latest",
        ) => (65_536, 1_048_576),
        (Provider::DeepSeek, "deepseek-flash" | "deepseek-v4-flash" | "deepseek-v4-pro") => {
            (393_216, 1_000_000)
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{BillingClass, Identity, Provider, Quote, UnknownPrice, quote};
    use super::{SNAPSHOT, Tariff, deepseek_peak, limits, lookup};

    #[test]
    fn token_limits_are_the_reviewed_source_rows() {
        let rows: Value = serde_json::from_str(include_str!(
            "../../../../licenses/pricing/source-rows.json"
        ))
        .unwrap();
        let rows = rows.as_object().unwrap();
        assert_eq!(rows.len(), 37);
        for (key, row) in rows {
            let (provider, model) = if let Some(model) = key.strip_prefix("gemini/") {
                (Provider::Gemini, model)
            } else if let Some(model) = key.strip_prefix("deepseek/") {
                (Provider::DeepSeek, model)
            } else if key.starts_with("claude-") {
                (Provider::Anthropic, key.as_str())
            } else {
                (Provider::OpenAi, key.as_str())
            };
            assert_eq!(
                limits(provider, model),
                Some((
                    row["max_output_tokens"].as_u64().unwrap(),
                    row["max_input_tokens"].as_u64().unwrap()
                )),
                "{key}"
            );
        }
        assert_eq!(limits(Provider::OpenAi, "gpt-unknown"), None);
    }
    use serde_json::{Value, json};

    // The compiled rates must be exactly the recorded source rows, field by field.
    fn recorded_rate(tariff: &Tariff, field: &str) -> Option<u128> {
        if let Some(field) = field.strip_prefix("off_peak_pricing.") {
            let rates = tariff.off_peak?;
            return match field {
                "input_cost_per_token" => Some(rates.input),
                "output_cost_per_token" => Some(rates.output),
                "cache_read_input_token_cost" => Some(rates.cache_read),
                _ => None,
            };
        }
        let (batch, field) = match field.strip_suffix("_batches") {
            Some(standard) => (true, standard),
            None => (false, field),
        };
        // LiteLLM names a long-context price `<field>_above_<N>k_tokens`.
        let (rates, field) = match field.rsplit_once("_above_") {
            Some((base, band)) if band.ends_with("k_tokens") => {
                let long = tariff.long?;
                let thousands: u64 = band.strip_suffix("k_tokens")?.parse().ok()?;
                assert_eq!(long.above, thousands * 1000, "{field}");
                (long.rates, base)
            }
            _ => (tariff.rates, field),
        };
        let rates = if batch { rates.batch() } else { rates };
        match field {
            "input_cost_per_token" => Some(rates.input),
            // Thinking is billed as output.
            "output_cost_per_token" | "output_cost_per_reasoning_token" => Some(rates.output),
            "cache_read_input_token_cost" => Some(rates.cache_read),
            "cache_creation_input_token_cost" => rates.cache_write,
            "cache_creation_input_token_cost_above_1hr" => rates.cache_write_1h,
            _ => None,
        }
    }

    #[test]
    fn compiled_rates_are_the_recorded_provenance_rows() {
        let provenance: Value =
            serde_json::from_str(include_str!("../../../../licenses/pricing/provenance.json"))
                .unwrap();
        assert_eq!(provenance["snapshot"], SNAPSHOT);
        let rows = provenance["rows"].as_array().unwrap();
        assert!(!rows.is_empty());
        for row in rows {
            let key = row["model"].as_str().unwrap();
            let (provider, model) = if let Some(model) = key.strip_prefix("gemini/") {
                (Provider::Gemini, model)
            } else if let Some(model) = key.strip_prefix("deepseek/") {
                (Provider::DeepSeek, model)
            } else if key.starts_with("claude-") {
                (Provider::Anthropic, key)
            } else {
                (Provider::OpenAi, key)
            };
            let tariff = lookup(provider, model).unwrap_or_else(|| panic!("{model} not compiled"));
            assert_eq!(
                tariff.batch,
                matches!(provider, Provider::OpenAi | Provider::Anthropic),
                "{model}"
            );
            let sources = row["official_sources"].as_array().unwrap();
            assert!(!sources.is_empty(), "{model} cites no official source");
            for (field, value) in row["picodollars_per_token"].as_object().unwrap() {
                assert_eq!(
                    recorded_rate(&tariff, field),
                    Some(u128::from(value.as_u64().unwrap())),
                    "{model} {field}"
                );
            }
        }
    }
    #[test]
    fn deepseek_peak_hours_follow_beijing_time_and_the_reviewed_calendar() {
        // 2026-10-12 is a Monday; Beijing is UTC+8.
        let monday = 1_791_763_200; // 2026-10-12T00:00:00Z, 08:00 in Beijing
        let at = |hours: u64, minutes: u64| monday + hours * 3600 + minutes * 60;
        assert_eq!(deepseek_peak(at(0, 59)), Some(false));
        assert_eq!(deepseek_peak(at(1, 0)), Some(true));
        assert_eq!(deepseek_peak(at(3, 59)), Some(true));
        assert_eq!(deepseek_peak(at(4, 0)), Some(false));
        assert_eq!(deepseek_peak(at(6, 0)), Some(true));
        assert_eq!(deepseek_peak(at(9, 59)), Some(true));
        assert_eq!(deepseek_peak(at(10, 0)), Some(false));
        // Saturday and Sunday are off-peak all day.
        assert_eq!(deepseek_peak(at(5 * 24 + 2, 0)), Some(false));
        assert_eq!(deepseek_peak(at(6 * 24 + 2, 0)), Some(false));
        // The reviewed days end on 2026-12-31 (a Thursday); 2027's public
        // holidays are not reviewed, so a weekday peak hour is not known.
        let thursday = 1_798_675_200; // 2026-12-31T00:00:00Z
        assert_eq!(deepseek_peak(thursday + 2 * 3600), Some(true));
        assert_eq!(deepseek_peak(thursday + 86_400 + 2 * 3600), None);
        assert_eq!(deepseek_peak(thursday + 86_400 + 12 * 3600), Some(false));
        // National Day week (October 1-7) precedes the reviewed days.
        assert_eq!(deepseek_peak(1_791_331_200 + 2 * 3600), None);
    }

    #[test]
    fn reviewed_claude_aliases_share_only_their_exact_dated_tariff() {
        for (alias, dated) in [
            ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
            ("claude-opus-4-5", "claude-opus-4-5-20251101"),
            ("claude-sonnet-4-5", "claude-sonnet-4-5-20250929"),
        ] {
            let identity = Identity {
                provider: "anthropic",
                endpoint: "https://api.anthropic.com/v1/messages",
                model: alias,
            };
            let response = json!({"model":dated,"usage":{"input_tokens":100,"output_tokens":10}});
            let expected = quote(
                &Identity {
                    model: dated,
                    ..identity
                },
                &response,
                BillingClass::Standard,
            );
            assert!(matches!(expected, Quote::Known { .. }));
            assert_eq!(
                quote(&identity, &response, BillingClass::Standard),
                expected
            );
            let different = json!({"model":format!("{alias}-new"),"usage":{"input_tokens":100,"output_tokens":10}});
            assert_eq!(
                quote(&identity, &different, BillingClass::Standard),
                Quote::Unknown(UnknownPrice::ResponseModel)
            );
        }
    }
}
