//! Reviewed token tariffs; values are exact picodollars per token.
//!
//! Source rows, original notice and verification URLs are kept under
//! licenses/pricing. These public-list estimates are not invoice settlement.
use super::Provider;

pub(super) const SNAPSHOT: &str = "litellm-1.106.0.dev2-selected-2026-10-09";

#[derive(Clone, Copy)]
pub(super) struct Rates {
    pub input: u128,
    pub output: u128,
    pub cache_read: u128,
    pub cache_write_5m: Option<u128>,
    pub cache_write_1h: Option<u128>,
}
impl Rates {
    // The reviewed providers document this discount for these specific rows,
    // including prompt caching. It is never applied to a document aggregate.
    pub fn batch(self) -> Self {
        Self {
            input: self.input / 2,
            output: self.output / 2,
            cache_read: self.cache_read / 2,
            cache_write_5m: self.cache_write_5m.map(|value| value / 2),
            cache_write_1h: self.cache_write_1h.map(|value| value / 2),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Tariff {
    pub family: &'static str,
    pub rates: Rates,
    pub max_priced_input: Option<u64>,
}

pub(super) fn lookup(provider: Provider, model: &str) -> Option<Tariff> {
    let (family, input, output, cache_read, cache_write_5m, cache_write_1h, cap) =
        match (provider, model) {
            (Provider::OpenAi, "gpt-4.1" | "gpt-4.1-2025-04-14") => (
                "openai/gpt-4.1",
                2_000_000,
                8_000_000,
                500_000,
                None,
                None,
                None,
            ),
            (Provider::OpenAi, "gpt-4.1-mini" | "gpt-4.1-mini-2025-04-14") => (
                "openai/gpt-4.1-mini",
                400_000,
                1_600_000,
                100_000,
                None,
                None,
                None,
            ),
            (Provider::OpenAi, "gpt-4.1-nano" | "gpt-4.1-nano-2025-04-14") => (
                "openai/gpt-4.1-nano",
                100_000,
                400_000,
                25_000,
                None,
                None,
                None,
            ),
            (Provider::Anthropic, "claude-sonnet-4-5" | "claude-sonnet-4-5-20250929") => (
                "anthropic/claude-sonnet-4-5-20250929",
                3_000_000,
                15_000_000,
                300_000,
                Some(3_750_000),
                Some(6_000_000),
                Some(200_000),
            ),
            (Provider::Anthropic, "claude-haiku-4-5" | "claude-haiku-4-5-20251001") => (
                "anthropic/claude-haiku-4-5-20251001",
                1_000_000,
                5_000_000,
                100_000,
                Some(1_250_000),
                Some(2_000_000),
                Some(200_000),
            ),
            _ => return None,
        };
    Some(Tariff {
        family,
        rates: Rates {
            input,
            output,
            cache_read,
            cache_write_5m,
            cache_write_1h,
        },
        max_priced_input: cap,
    })
}

#[cfg(test)]
mod tests {
    use super::super::{BillingClass, Identity, Provider, Quote, UnknownPrice, quote};
    use super::{SNAPSHOT, Tariff, lookup};
    use serde_json::{Value, json};

    // The compiled rates must be exactly the recorded source rows, field by field.
    fn recorded_rate(tariff: &Tariff, field: &str) -> Option<u128> {
        let (rates, field) = match field.strip_suffix("_batches") {
            Some(standard) => (tariff.rates.batch(), standard),
            None => (tariff.rates, field),
        };
        match field {
            "input_cost_per_token" => Some(rates.input),
            "output_cost_per_token" => Some(rates.output),
            "cache_read_input_token_cost" => Some(rates.cache_read),
            "cache_creation_input_token_cost" => rates.cache_write_5m,
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
            let model = row["model"].as_str().unwrap();
            let provider = if model.starts_with("claude-") {
                Provider::Anthropic
            } else {
                Provider::OpenAi
            };
            let tariff = lookup(provider, model).unwrap_or_else(|| panic!("{model} not compiled"));
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
    fn reviewed_claude_aliases_share_only_their_exact_dated_tariff() {
        for (alias, dated) in [
            ("claude-haiku-4-5", "claude-haiku-4-5-20251001"),
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
