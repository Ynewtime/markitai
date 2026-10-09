//! Offline public-list pricing for a deliberately explicit provider/model set.
//! Missing prices and incomplete usage remain unknown, not known zero charges.
mod catalog;

use serde_json::{Map, Value};

const PICODOLLARS_PER_USD: u128 = 1_000_000_000_000;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct Usd(u128);
impl Usd {
    pub(crate) fn picodollars(self) -> u128 {
        self.0
    }
    pub(crate) fn to_f64(self) -> f64 {
        self.picodollars() as f64 / PICODOLLARS_PER_USD as f64
    }
    pub(crate) fn checked_add(self, other: Self) -> Option<Self> {
        self.0.checked_add(other.0).map(Self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BudgetError {
    Invalid,
    OutOfRange,
}

/// Floor the actual finite IEEE-754 value to picodollars, without a saturating
/// float-to-integer cast. Positive sub-picodollar budgets remain Some(0).
/// This can trip less than one picodollar early, never relax a user's limit.
pub(crate) fn budget_from_f64(value: f64) -> Result<Option<Usd>, BudgetError> {
    if !value.is_finite() || value < 0.0 {
        return Err(BudgetError::Invalid);
    }
    if value == 0.0 {
        return Ok(None);
    }
    let bits = value.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1u64 << 52) - 1);
    let (mantissa, power) = if exponent == 0 {
        (u128::from(fraction), -1074)
    } else {
        (u128::from(fraction | (1u64 << 52)), exponent - 1023 - 52)
    };
    let scaled = mantissa
        .checked_mul(PICODOLLARS_PER_USD)
        .ok_or(BudgetError::OutOfRange)?;
    let amount = if power >= 0 {
        let multiplier = 1u128
            .checked_shl(power as u32)
            .ok_or(BudgetError::OutOfRange)?;
        scaled
            .checked_mul(multiplier)
            .ok_or(BudgetError::OutOfRange)?
    } else {
        scaled.checked_shr((-power) as u32).unwrap_or(0)
    };
    Ok(Some(Usd(amount)))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum BillingClass {
    Standard,
    Batch,
}

// Intentionally no Debug: callers can hold a secret-bearing endpoint supplied
// by configuration; unknown identities never appear in error messages.
pub(crate) struct Identity<'a> {
    pub provider: &'a str,
    pub endpoint: &'a str,
    pub model: &'a str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum UnknownPrice {
    Provider,
    Endpoint,
    Model,
    ResponseModel,
    ServiceTier,
    Context,
    MissingUsage,
    InvalidUsage,
    CacheDetails,
    UnsupportedUsage,
    Overflow,
}
impl UnknownPrice {
    pub(crate) fn reason(self) -> &'static str {
        match self {
            Self::Provider => "provider has no reviewed tariff",
            Self::Endpoint => "endpoint has no verified first-party tariff",
            Self::Model => "model has no exact reviewed tariff",
            Self::ResponseModel => "response model differs from the reviewed requested tariff",
            Self::ServiceTier => "service tier has no reviewed tariff",
            Self::Context => "context size exceeds the reviewed price band",
            Self::MissingUsage => "provider did not report complete token usage",
            Self::InvalidUsage => "provider reported contradictory or invalid token usage",
            Self::CacheDetails => "cache creation duration or counters cannot be priced exactly",
            Self::UnsupportedUsage => "provider reported an unsupported billing category",
            Self::Overflow => "price arithmetic exceeds the supported range",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Quote {
    Known {
        amount: Usd,
        snapshot: &'static str,
        tariff: &'static str,
    },
    Unknown(UnknownPrice),
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum Provider {
    OpenAi,
    Anthropic,
}

fn resolve(identity: &Identity<'_>) -> Result<(Provider, catalog::Tariff), UnknownPrice> {
    let provider = match identity.provider {
        "openai" => Provider::OpenAi,
        "anthropic" => Provider::Anthropic,
        _ => return Err(UnknownPrice::Provider),
    };
    // Exact final endpoints deliberately exclude proxies, alternate regions,
    // custom ports, userinfo, queries and look-alike hosts. No network lookup.
    let valid = match provider {
        Provider::OpenAi => identity.endpoint == "https://api.openai.com/v1/chat/completions",
        Provider::Anthropic => identity.endpoint == "https://api.anthropic.com/v1/messages",
    };
    if !valid {
        return Err(UnknownPrice::Endpoint);
    }
    let tariff = catalog::lookup(provider, identity.model).ok_or(UnknownPrice::Model)?;
    Ok((provider, tariff))
}

/// Preflight checks tariff identity only; quote validates observed counters,
/// returned model, context size and service tier after the request completes.
pub(crate) fn preflight(identity: &Identity<'_>, _class: BillingClass) -> Result<(), UnknownPrice> {
    resolve(identity).map(|_| ())
}

pub(crate) fn quote(identity: &Identity<'_>, response: &Value, class: BillingClass) -> Quote {
    match calculate(identity, response, class) {
        Ok((amount, tariff)) => Quote::Known {
            amount,
            snapshot: catalog::SNAPSHOT,
            tariff,
        },
        Err(reason) => Quote::Unknown(reason),
    }
}

fn number(value: Option<&Value>) -> Result<Option<u64>, UnknownPrice> {
    value
        .map(|value| value.as_u64().ok_or(UnknownPrice::InvalidUsage))
        .transpose()
}
fn required(
    usage: &Map<String, Value>,
    primary: &str,
    alternate: Option<&str>,
) -> Result<u64, UnknownPrice> {
    let first = number(usage.get(primary))?;
    let second = number(alternate.and_then(|key| usage.get(key)))?;
    if first.zip(second).is_some_and(|(a, b)| a != b) {
        return Err(UnknownPrice::InvalidUsage);
    }
    first.or(second).ok_or(UnknownPrice::MissingUsage)
}
fn optional(usage: &Map<String, Value>, key: &str) -> Result<u64, UnknownPrice> {
    Ok(number(usage.get(key))?.unwrap_or(0))
}
fn object<'a>(
    usage: &'a Map<String, Value>,
    key: &str,
) -> Result<Option<&'a Map<String, Value>>, UnknownPrice> {
    usage
        .get(key)
        .map(|value| value.as_object().ok_or(UnknownPrice::InvalidUsage))
        .transpose()
}
// The same counter reported under both Chat Completions and Responses names
// must agree; an absent one is zero.
fn agree(values: [Option<u64>; 2]) -> Result<u64, UnknownPrice> {
    if values[0].zip(values[1]).is_some_and(|(a, b)| a != b) {
        return Err(UnknownPrice::InvalidUsage);
    }
    Ok(values[0].or(values[1]).unwrap_or(0))
}
fn add(a: u64, b: u64) -> Result<u64, UnknownPrice> {
    a.checked_add(b).ok_or(UnknownPrice::Overflow)
}
fn terms(values: &[(u64, u128)]) -> Result<Usd, UnknownPrice> {
    values
        .iter()
        .try_fold(Usd::default(), |sum, (count, rate)| {
            let amount = u128::from(*count)
                .checked_mul(*rate)
                .ok_or(UnknownPrice::Overflow)?;
            sum.checked_add(Usd(amount)).ok_or(UnknownPrice::Overflow)
        })
}
fn tier(value: Option<&Value>, class: BillingClass) -> Result<(), UnknownPrice> {
    let Some(value) = value else {
        return Ok(());
    };
    if value.is_null() {
        return Ok(());
    }
    match (value.as_str(), class) {
        (Some("default" | "standard"), _) | (Some("batch"), BillingClass::Batch) => Ok(()),
        _ => Err(UnknownPrice::ServiceTier),
    }
}
fn allowed_keys(usage: &Map<String, Value>, keys: &[&str]) -> Result<(), UnknownPrice> {
    if usage.keys().any(|key| !keys.contains(&key.as_str())) {
        return Err(UnknownPrice::UnsupportedUsage);
    }
    Ok(())
}

fn calculate(
    identity: &Identity<'_>,
    response: &Value,
    class: BillingClass,
) -> Result<(Usd, &'static str), UnknownPrice> {
    let (provider, tariff) = resolve(identity)?;
    if let Some(model) = response.get("model") {
        let model = model.as_str().ok_or(UnknownPrice::ResponseModel)?;
        if catalog::lookup(provider, model).is_none_or(|actual| actual.family != tariff.family) {
            return Err(UnknownPrice::ResponseModel);
        }
    }
    let usage = response
        .get("usage")
        .and_then(Value::as_object)
        .ok_or(UnknownPrice::MissingUsage)?;
    tier(response.get("service_tier"), class)?;
    tier(usage.get("service_tier"), class)?;
    let (base, cached, written, written_1h, output) = match provider {
        Provider::OpenAi => {
            allowed_keys(
                usage,
                &[
                    "prompt_tokens",
                    "input_tokens",
                    "completion_tokens",
                    "output_tokens",
                    "total_tokens",
                    "prompt_tokens_details",
                    "input_tokens_details",
                    "completion_tokens_details",
                    "output_tokens_details",
                    "service_tier",
                ],
            )?;
            let input = required(usage, "prompt_tokens", Some("input_tokens"))?;
            let output = required(usage, "completion_tokens", Some("output_tokens"))?;
            let (mut cached, mut written) = ([None, None], [None, None]);
            for (index, key) in ["prompt_tokens_details", "input_tokens_details"]
                .iter()
                .enumerate()
            {
                if let Some(details) = object(usage, key)? {
                    allowed_keys(
                        details,
                        &[
                            "cached_tokens",
                            "cache_write_tokens",
                            "audio_tokens",
                            "text_tokens",
                            "image_tokens",
                        ],
                    )?;
                    if optional(details, "audio_tokens")? != 0 {
                        return Err(UnknownPrice::UnsupportedUsage);
                    }
                    // Text and image counts split the prompt by modality; image
                    // input is metered as ordinary input tokens.
                    let modalities = add(
                        optional(details, "text_tokens")?,
                        optional(details, "image_tokens")?,
                    )?;
                    if modalities > input {
                        return Err(UnknownPrice::InvalidUsage);
                    }
                    cached[index] = number(details.get("cached_tokens"))?;
                    written[index] = number(details.get("cache_write_tokens"))?;
                }
            }
            // OpenAI's prompt caching guide prices the prompt as ordinary
            // input, cached reads and cache writes, each a disjoint part of it.
            let (cached, written) = (agree(cached)?, agree(written)?);
            if add(cached, written)? > input {
                return Err(UnknownPrice::InvalidUsage);
            }
            for key in ["completion_tokens_details", "output_tokens_details"] {
                if let Some(details) = object(usage, key)? {
                    allowed_keys(
                        details,
                        &[
                            "reasoning_tokens",
                            "audio_tokens",
                            "text_tokens",
                            "accepted_prediction_tokens",
                            "rejected_prediction_tokens",
                        ],
                    )?;
                    if optional(details, "audio_tokens")? != 0 {
                        return Err(UnknownPrice::UnsupportedUsage);
                    }
                    for counter in [
                        "reasoning_tokens",
                        "text_tokens",
                        "accepted_prediction_tokens",
                        "rejected_prediction_tokens",
                    ] {
                        if optional(details, counter)? > output {
                            return Err(UnknownPrice::InvalidUsage);
                        }
                    }
                }
            }
            (input - cached - written, cached, written, 0, output)
        }
        Provider::Anthropic => {
            allowed_keys(
                usage,
                &[
                    "input_tokens",
                    "output_tokens",
                    "cache_read_input_tokens",
                    "cache_creation_input_tokens",
                    "cache_creation",
                    "total_tokens",
                    "service_tier",
                    "inference_geo",
                ],
            )?;
            // The Messages API reports where it ran; US-only inference is
            // billed at a premium, so only an unrestricted placement keeps the
            // listed rates.
            match usage.get("inference_geo") {
                None | Some(Value::Null) => {}
                Some(value) if matches!(value.as_str(), Some("not_available" | "global")) => {}
                Some(_) => return Err(UnknownPrice::UnsupportedUsage),
            }
            let input = required(usage, "input_tokens", None)?;
            let output = required(usage, "output_tokens", None)?;
            let cached = optional(usage, "cache_read_input_tokens")?;
            let total_created = number(usage.get("cache_creation_input_tokens"))?;
            let (write5, write1) = if let Some(details) = object(usage, "cache_creation")? {
                allowed_keys(
                    details,
                    &["ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens"],
                )?;
                let five = optional(details, "ephemeral_5m_input_tokens")?;
                let hour = optional(details, "ephemeral_1h_input_tokens")?;
                let created = add(five, hour)?;
                if total_created.is_some_and(|total| created != total) {
                    return Err(UnknownPrice::InvalidUsage);
                }
                (five, hour)
            } else if total_created.unwrap_or(0) > 0 {
                return Err(UnknownPrice::CacheDetails);
            } else {
                (0, 0)
            };
            (input, cached, write5, write1, output)
        }
    };
    let all_input = add(add(add(base, cached)?, written)?, written_1h)?;
    if tariff
        .max_priced_input
        .is_some_and(|limit| all_input > limit)
    {
        return Err(UnknownPrice::Context);
    }
    let rates = match tariff.long {
        Some(long) if all_input >= long.long_from => long.rates,
        Some(long) if all_input > long.short_max => return Err(UnknownPrice::Context),
        _ => tariff.rates,
    };
    let rates = match class {
        BillingClass::Standard => rates,
        BillingClass::Batch => rates.batch(),
    };
    if let Some(total) = number(usage.get("total_tokens"))?
        && add(all_input, output)? != total
    {
        return Err(UnknownPrice::InvalidUsage);
    }
    let write_rate = |count, rate: Option<u128>| {
        if count == 0 {
            Ok(0)
        } else {
            rate.ok_or(UnknownPrice::CacheDetails)
        }
    };
    let amount = terms(&[
        (base, rates.input),
        (cached, rates.cache_read),
        (written, write_rate(written, rates.cache_write)?),
        (written_1h, write_rate(written_1h, rates.cache_write_1h)?),
        (output, rates.output),
    ])?;
    Ok((amount, tariff.family))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn openai() -> Identity<'static> {
        Identity {
            provider: "openai",
            endpoint: "https://api.openai.com/v1/chat/completions",
            model: "gpt-4.1",
        }
    }
    fn anthropic() -> Identity<'static> {
        Identity {
            provider: "anthropic",
            endpoint: "https://api.anthropic.com/v1/messages",
            model: "claude-sonnet-4-5-20250929",
        }
    }
    fn amount(id: &Identity<'_>, response: &Value, class: BillingClass) -> u128 {
        match quote(id, response, class) {
            Quote::Known {
                amount, snapshot, ..
            } => {
                assert_eq!(snapshot, catalog::SNAPSHOT);
                amount.picodollars()
            }
            other => panic!("expected exact known price, got {other:?}"),
        }
    }

    #[test]
    fn cached_input_and_reasoning_are_subsets_not_extra_tokens() {
        let response = json!({"model":"gpt-4.1-2025-04-14", "usage": {
            "prompt_tokens":1000, "completion_tokens":200, "total_tokens":1200,
            "prompt_tokens_details":{"cached_tokens":400,"audio_tokens":0},
            "completion_tokens_details":{"reasoning_tokens":150,"accepted_prediction_tokens":10,"rejected_prediction_tokens":20}
        }});
        assert_eq!(
            amount(&openai(), &response, BillingClass::Standard),
            3_000_000_000
        );
        assert_eq!(
            amount(&openai(), &response, BillingClass::Batch),
            1_500_000_000
        );
    }

    fn luna() -> Identity<'static> {
        Identity {
            model: "gpt-6-luna",
            ..openai()
        }
    }

    #[test]
    fn openai_cache_writes_are_a_separately_billed_part_of_the_prompt() {
        // The usage shape of OpenAI's Chat Completions reference.
        let response = json!({"model":"gpt-6-luna","service_tier":"default","usage":{
            "prompt_tokens":10000,"completion_tokens":500,"total_tokens":10500,
            "prompt_tokens_details":{"audio_tokens":0,"cache_write_tokens":3000,
                "cached_tokens":4000,"image_tokens":1200,"text_tokens":8800},
            "completion_tokens_details":{"accepted_prediction_tokens":0,"audio_tokens":0,
                "reasoning_tokens":300,"rejected_prediction_tokens":0,"text_tokens":200}}});
        let standard = 3000 * 100_000 + 4000 * 10_000 + 3000 * 125_000 + 500 * 500_000;
        assert_eq!(amount(&luna(), &response, BillingClass::Standard), standard);
        assert_eq!(
            amount(&luna(), &response, BillingClass::Batch),
            standard / 2
        );

        let mut both_names = response.clone();
        both_names["usage"]["input_tokens_details"] = json!({"cache_write_tokens":2999});
        let mut too_many = response.clone();
        too_many["usage"]["prompt_tokens_details"]["cache_write_tokens"] = json!(6001);
        let mut modalities = response.clone();
        modalities["usage"]["prompt_tokens_details"]["image_tokens"] = json!(1201);
        let mut text_output = response.clone();
        text_output["usage"]["completion_tokens_details"]["text_tokens"] = json!(501);
        for invalid in [both_names, too_many, modalities, text_output] {
            assert_eq!(
                quote(&luna(), &invalid, BillingClass::Standard),
                Quote::Unknown(UnknownPrice::InvalidUsage)
            );
        }

        // GPT-4.1 has no reviewed cache-write rate: a write is not guessed.
        let older = json!({"usage":{"prompt_tokens":10,"completion_tokens":1,
            "prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0}}});
        assert_eq!(
            amount(&openai(), &older, BillingClass::Standard),
            10 * 2_000_000 + 8_000_000
        );
        let mut written = older;
        written["usage"]["prompt_tokens_details"]["cache_write_tokens"] = json!(4);
        assert_eq!(
            quote(&openai(), &written, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::CacheDetails)
        );
    }

    #[test]
    fn openai_long_context_reprices_the_whole_request_once_certain() {
        let response = |prompt: u64| {
            json!({"usage":{"prompt_tokens":prompt,"completion_tokens":1000,
                "prompt_tokens_details":{"cached_tokens":100_000,"cache_write_tokens":50_000}}})
        };
        let short = 122_000 * 100_000 + 100_000 * 10_000 + 50_000 * 125_000 + 1000 * 500_000;
        assert_eq!(
            amount(&luna(), &response(272_000), BillingClass::Standard),
            short
        );
        // Between 272,000 and 272 x 1,024 the published "272K" boundary is unclear.
        for unclear in [272_001, 275_000, 278_528] {
            assert_eq!(
                quote(&luna(), &response(unclear), BillingClass::Standard),
                Quote::Unknown(UnknownPrice::Context)
            );
        }
        let long = 128_529 * 200_000 + 100_000 * 20_000 + 50_000 * 250_000 + 1000 * 750_000;
        assert_eq!(
            amount(&luna(), &response(278_529), BillingClass::Standard),
            long
        );
        assert_eq!(
            amount(&luna(), &response(278_529), BillingClass::Batch),
            long / 2
        );
    }

    #[test]
    fn an_unrestricted_inference_placement_keeps_the_listed_rates() {
        // The usage the Messages API returned for Claude Haiku 4.5 on 2026-10-02.
        let mut response = json!({"model":"claude-sonnet-4-5-20250929","usage":{"input_tokens":9,
            "cache_creation_input_tokens":0,"cache_read_input_tokens":0,
            "cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},
            "output_tokens":4,"service_tier":"standard","inference_geo":"not_available"}});
        let listed = amount(&anthropic(), &response, BillingClass::Standard);
        assert_eq!(listed, 9 * 3_000_000 + 4 * 15_000_000);
        response["usage"]["inference_geo"] = json!("global");
        assert_eq!(
            amount(&anthropic(), &response, BillingClass::Standard),
            listed
        );
        // US-only inference is billed at a premium these rates do not hold.
        for geo in [json!("us"), json!(1), json!("")] {
            response["usage"]["inference_geo"] = geo;
            assert_eq!(
                quote(&anthropic(), &response, BillingClass::Standard),
                Quote::Unknown(UnknownPrice::UnsupportedUsage)
            );
        }
    }

    #[test]
    fn anthropic_mixed_cache_ttls_are_each_charged_once_in_both_classes() {
        let response = json!({"usage": {"input_tokens":100,"output_tokens":50,
            "cache_read_input_tokens":200,"cache_creation_input_tokens":70,
            "cache_creation":{"ephemeral_5m_input_tokens":30,"ephemeral_1h_input_tokens":40}}});
        assert_eq!(
            amount(&anthropic(), &response, BillingClass::Standard),
            1_462_500_000
        );
        assert_eq!(
            amount(&anthropic(), &response, BillingClass::Batch),
            731_250_000
        );
        let mut ambiguous = response.clone();
        ambiguous["usage"]
            .as_object_mut()
            .unwrap()
            .remove("cache_creation");
        assert_eq!(
            quote(&anthropic(), &ambiguous, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::CacheDetails)
        );
        let mut contradictory = response;
        contradictory["usage"]["cache_creation_input_tokens"] = json!(71);
        assert_eq!(
            quote(&anthropic(), &contradictory, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::InvalidUsage)
        );
    }

    #[test]
    fn explicit_zero_is_known_but_missing_or_malformed_usage_is_not() {
        assert_eq!(
            amount(
                &openai(),
                &json!({"usage":{"prompt_tokens":0,"completion_tokens":0}}),
                BillingClass::Standard
            ),
            0
        );
        for usage in [json!({}), json!({"prompt_tokens":1})] {
            assert_eq!(
                quote(&openai(), &json!({"usage":usage}), BillingClass::Standard),
                Quote::Unknown(UnknownPrice::MissingUsage)
            );
        }
        for usage in [
            json!({"prompt_tokens":true,"completion_tokens":0}),
            json!({"prompt_tokens":-1,"completion_tokens":0}),
            json!({"prompt_tokens":1.5,"completion_tokens":0}),
        ] {
            assert_eq!(
                quote(&openai(), &json!({"usage":usage}), BillingClass::Standard),
                Quote::Unknown(UnknownPrice::InvalidUsage)
            );
        }
        assert_eq!(
            quote(
                &openai(),
                &json!({"error":{"message":"paid invalid response"}}),
                BillingClass::Standard
            ),
            Quote::Unknown(UnknownPrice::MissingUsage)
        );
    }

    #[test]
    fn unknown_endpoint_model_and_provider_are_never_guessed() {
        for endpoint in [
            "http://api.openai.com/v1/chat/completions",
            "https://api.openai.com.evil/v1/chat/completions",
            "https://api.openai.com/v1/chat/completions?token=secret",
            "https://proxy.example/v1/chat/completions",
            "https://api.openai.com:443/v1/chat/completions",
        ] {
            assert_eq!(
                preflight(
                    &Identity {
                        endpoint,
                        ..openai()
                    },
                    BillingClass::Standard
                ),
                Err(UnknownPrice::Endpoint)
            );
        }
        assert_eq!(
            preflight(
                &Identity {
                    model: "openai/gpt-4.1",
                    ..openai()
                },
                BillingClass::Standard
            ),
            Err(UnknownPrice::Model)
        );
        assert_eq!(
            preflight(
                &Identity {
                    model: "gpt-4.1-new",
                    ..openai()
                },
                BillingClass::Standard
            ),
            Err(UnknownPrice::Model)
        );
        assert_eq!(
            preflight(
                &Identity {
                    provider: "azure",
                    ..openai()
                },
                BillingClass::Standard
            ),
            Err(UnknownPrice::Provider)
        );
    }

    #[test]
    fn tier_and_response_model_cannot_override_identity_silently() {
        let mut response =
            json!({"usage":{"prompt_tokens":1,"completion_tokens":1},"model":"gpt-4.1-mini"});
        assert_eq!(
            quote(&openai(), &response, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::ResponseModel)
        );
        response.as_object_mut().unwrap().remove("model");
        for tier in ["priority", "flex", "auto", "batch"] {
            response["service_tier"] = json!(tier);
            assert_eq!(
                quote(&openai(), &response, BillingClass::Standard),
                Quote::Unknown(UnknownPrice::ServiceTier)
            );
        }
        response["service_tier"] = json!("batch");
        assert_eq!(amount(&openai(), &response, BillingClass::Batch), 5_000_000);
    }

    #[test]
    fn unknown_categories_and_inconsistent_counters_stay_unpriced() {
        for usage in [
            json!({"prompt_tokens":10,"completion_tokens":1,"total_tokens":12}),
            json!({"prompt_tokens":10,"input_tokens":11,"completion_tokens":1}),
            json!({"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"cached_tokens":11}}),
            json!({"prompt_tokens":10,"completion_tokens":1,"completion_tokens_details":{"reasoning_tokens":2}}),
        ] {
            assert_eq!(
                quote(&openai(), &json!({"usage":usage}), BillingClass::Standard),
                Quote::Unknown(UnknownPrice::InvalidUsage)
            );
        }
        for usage in [
            json!({"prompt_tokens":10,"completion_tokens":1,"server_tool_use":{"web_search_requests":1}}),
            json!({"prompt_tokens":10,"completion_tokens":1,"prompt_tokens_details":{"audio_tokens":1}}),
            json!({"input_tokens":10,"output_tokens":1,"future_billing_category":0}),
        ] {
            assert_eq!(
                quote(&openai(), &json!({"usage":usage}), BillingClass::Standard),
                Quote::Unknown(UnknownPrice::UnsupportedUsage)
            );
        }
    }

    #[test]
    fn context_band_counts_read_and_written_cache_tokens() {
        let mut response =
            json!({"usage":{"input_tokens":199999,"output_tokens":1,"cache_read_input_tokens":1}});
        assert!(matches!(
            quote(&anthropic(), &response, BillingClass::Standard),
            Quote::Known { .. }
        ));
        response["usage"]["cache_read_input_tokens"] = json!(2);
        assert_eq!(
            quote(&anthropic(), &response, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::Context)
        );
    }

    #[test]
    fn paid_error_body_is_priced_without_requiring_success_content() {
        let response = json!({"error":{"message":"semantic failure"},"usage":{"prompt_tokens":7,"completion_tokens":3}});
        assert_eq!(
            amount(&openai(), &response, BillingClass::Standard),
            38_000_000
        );
    }

    #[test]
    fn arithmetic_is_checked_instead_of_wrapping_or_saturating() {
        assert_eq!(terms(&[(u64::MAX, u128::MAX)]), Err(UnknownPrice::Overflow));
        assert_eq!(Usd(u128::MAX).checked_add(Usd(1)), None);
        let response = json!({"usage":{"input_tokens":u64::MAX,"output_tokens":0,"cache_read_input_tokens":1}});
        assert_eq!(
            quote(&anthropic(), &response, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::Overflow)
        );
    }

    #[test]
    fn finite_budget_uses_exact_binary_value_floored_to_picodollars() {
        assert_eq!(budget_from_f64(0.0), Ok(None));
        assert_eq!(budget_from_f64(-0.0), Ok(None));
        assert_eq!(
            budget_from_f64(1.0).unwrap().unwrap().picodollars(),
            1_000_000_000_000
        );
        assert_eq!(
            budget_from_f64(0.125).unwrap().unwrap().picodollars(),
            125_000_000_000
        );
        // 0.3's binary representation is slightly less than the decimal 0.3.
        assert_eq!(
            budget_from_f64(0.3).unwrap().unwrap().picodollars(),
            299_999_999_999
        );
        assert_eq!(budget_from_f64(f64::from_bits(1)), Ok(Some(Usd(0))));
        assert_eq!(budget_from_f64(f64::MAX), Err(BudgetError::OutOfRange));
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0] {
            assert_eq!(budget_from_f64(value), Err(BudgetError::Invalid));
        }
        let boundary = u128::MAX as f64 / PICODOLLARS_PER_USD as f64;
        assert!(budget_from_f64(f64::from_bits(boundary.to_bits() + 1)).is_err());
        assert!(budget_from_f64(f64::from_bits(boundary.to_bits() - 1)).is_ok());
        assert_eq!(Usd(125_000_000_000).to_f64(), 0.125);
    }
}
