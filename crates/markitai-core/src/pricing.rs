//! Offline public-list pricing for a deliberately explicit provider/model set,
//! and OpenRouter's own report of what it charged.
//! Missing prices and incomplete usage remain unknown, not known zero charges.
mod catalog;

use serde_json::{Map, Value};
use std::time::{SystemTime, UNIX_EPOCH};

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

/// When a request was sent and its response read, in Unix seconds. Dated and
/// peak-hour tariffs need both ends to fall in one price period.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Span {
    pub sent: u64,
    pub done: u64,
}
impl Span {
    pub(crate) fn since(sent: SystemTime) -> Option<Self> {
        let seconds = |at: SystemTime| at.duration_since(UNIX_EPOCH).ok().map(|d| d.as_secs());
        Some(Self {
            sent: seconds(sent)?,
            done: seconds(SystemTime::now())?,
        })
    }
}

/// OpenRouter's `usage.cost` is the amount it charged, not a catalog price.
const OPENROUTER_REPORTED: &str = "openrouter-reported-cost";

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
    Period,
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
            Self::Period => "request time falls outside a reviewed price period",
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
    Gemini,
    DeepSeek,
    OpenRouter,
}

/// The catalog tariff, or `None` for OpenRouter, which reports its charge.
fn resolve(identity: &Identity<'_>) -> Result<(Provider, Option<catalog::Tariff>), UnknownPrice> {
    let provider = match identity.provider {
        "openai" => Provider::OpenAi,
        "anthropic" => Provider::Anthropic,
        "gemini" => Provider::Gemini,
        "deepseek" => Provider::DeepSeek,
        "openrouter" => Provider::OpenRouter,
        _ => return Err(UnknownPrice::Provider),
    };
    // Exact final endpoints deliberately exclude proxies, alternate regions,
    // custom ports, userinfo, queries and look-alike hosts. No network lookup.
    let valid = match provider {
        Provider::OpenAi => identity.endpoint == "https://api.openai.com/v1/chat/completions",
        Provider::Anthropic => identity.endpoint == "https://api.anthropic.com/v1/messages",
        Provider::Gemini => {
            identity.endpoint
                == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        }
        // DeepSeek documents both base URLs for its OpenAI format.
        Provider::DeepSeek => matches!(
            identity.endpoint,
            "https://api.deepseek.com/chat/completions"
                | "https://api.deepseek.com/v1/chat/completions"
        ),
        Provider::OpenRouter => {
            identity.endpoint == "https://openrouter.ai/api/v1/chat/completions"
        }
    };
    if !valid {
        return Err(UnknownPrice::Endpoint);
    }
    if provider == Provider::OpenRouter {
        return Ok((provider, None));
    }
    let tariff = catalog::lookup(provider, identity.model).ok_or(UnknownPrice::Model)?;
    Ok((provider, Some(tariff)))
}

/// Preflight checks tariff identity only; quote validates observed counters,
/// returned model, context size and service tier after the request completes.
pub(crate) fn preflight(identity: &Identity<'_>, _class: BillingClass) -> Result<(), UnknownPrice> {
    resolve(identity).map(|_| ())
}

/// A quote without request times: dated and peak-hour tariffs stay unknown.
#[cfg(test)]
pub(crate) fn quote(identity: &Identity<'_>, response: &Value, class: BillingClass) -> Quote {
    quote_at(identity, response, class, None)
}

pub(crate) fn quote_at(
    identity: &Identity<'_>,
    response: &Value,
    class: BillingClass,
    span: Option<Span>,
) -> Quote {
    match calculate(identity, response, class, span) {
        Ok((amount, snapshot, tariff)) => Quote::Known {
            amount,
            snapshot,
            tariff,
        },
        Err(reason) => Quote::Unknown(reason),
    }
}

// The Messages API documents several counters as "number or null"; null
// reports nothing, like an absent field.
fn number(value: Option<&Value>) -> Result<Option<u64>, UnknownPrice> {
    value
        .filter(|value| !value.is_null())
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
        .filter(|value| !value.is_null())
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

/// OpenRouter's charge for the request, in USD credits, rounded to the
/// nearest picodollar. Bring-your-own-key requests are also billed by the
/// upstream provider, outside this figure, so they stay unpriced.
fn reported(response: &Value, class: BillingClass) -> Result<Usd, UnknownPrice> {
    if class != BillingClass::Standard {
        return Err(UnknownPrice::ServiceTier);
    }
    let usage = response
        .get("usage")
        .and_then(Value::as_object)
        .ok_or(UnknownPrice::MissingUsage)?;
    match usage.get("is_byok") {
        None | Some(Value::Null | Value::Bool(false)) => {}
        Some(Value::Bool(true)) => return Err(UnknownPrice::UnsupportedUsage),
        Some(_) => return Err(UnknownPrice::InvalidUsage),
    }
    let cost = match usage.get("cost") {
        None | Some(Value::Null) => return Err(UnknownPrice::MissingUsage),
        Some(value) => value.as_f64().ok_or(UnknownPrice::InvalidUsage)?,
    };
    if !cost.is_finite() || cost < 0.0 {
        return Err(UnknownPrice::InvalidUsage);
    }
    // Far beyond any single request; keeps the conversion exact to a picodollar.
    if cost > 1000.0 {
        return Err(UnknownPrice::Overflow);
    }
    Ok(Usd((cost * PICODOLLARS_PER_USD as f64).round() as u128))
}

fn calculate(
    identity: &Identity<'_>,
    response: &Value,
    class: BillingClass,
    span: Option<Span>,
) -> Result<(Usd, &'static str, &'static str), UnknownPrice> {
    let (provider, tariff) = resolve(identity)?;
    let Some(tariff) = tariff else {
        return Ok((
            reported(response, class)?,
            OPENROUTER_REPORTED,
            "openrouter/usage.cost",
        ));
    };
    if class == BillingClass::Batch && !tariff.batch {
        return Err(UnknownPrice::ServiceTier);
    }
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
                    "output_tokens_details",
                    "server_tool_use",
                    "total_tokens",
                    "service_tier",
                    "inference_geo",
                    "speed",
                ],
            )?;
            // The Messages API reports the speed it ran at; fast mode is
            // billed at a premium that the catalog does not list.
            match usage.get("speed") {
                None | Some(Value::Null) => {}
                Some(value) if value.as_str() == Some("standard") => {}
                Some(_) => return Err(UnknownPrice::UnsupportedUsage),
            }
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
            // Thinking is part of the billed output total, not an addition.
            if let Some(details) = object(usage, "output_tokens_details")? {
                allowed_keys(details, &["thinking_tokens"])?;
                if optional(details, "thinking_tokens")? > output {
                    return Err(UnknownPrice::InvalidUsage);
                }
            }
            // Server tools are billed per request, outside this catalog.
            if let Some(tools) = object(usage, "server_tool_use")? {
                allowed_keys(tools, &["web_fetch_requests", "web_search_requests"])?;
                if optional(tools, "web_fetch_requests")? != 0
                    || optional(tools, "web_search_requests")? != 0
                {
                    return Err(UnknownPrice::UnsupportedUsage);
                }
            }
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
        Provider::Gemini => {
            allowed_keys(
                usage,
                &[
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_tokens_details",
                    "completion_tokens_details",
                    "service_tier",
                ],
            )?;
            let input = required(usage, "prompt_tokens", None)?;
            let completion = required(usage, "completion_tokens", None)?;
            // The OpenAI-compatible endpoint leaves thinking out of
            // completion_tokens but counts it in total_tokens; Google bills
            // thinking as output, so the total is required.
            let total = required(usage, "total_tokens", None)?;
            let mut cached = 0;
            if let Some(details) = object(usage, "prompt_tokens_details")? {
                allowed_keys(
                    details,
                    &[
                        "cached_tokens",
                        "audio_tokens",
                        "text_tokens",
                        "image_tokens",
                    ],
                )?;
                // Audio input has its own rate on some models.
                if optional(details, "audio_tokens")? != 0 {
                    return Err(UnknownPrice::UnsupportedUsage);
                }
                let modalities = add(
                    optional(details, "text_tokens")?,
                    optional(details, "image_tokens")?,
                )?;
                if modalities > input {
                    return Err(UnknownPrice::InvalidUsage);
                }
                cached = optional(details, "cached_tokens")?;
            }
            if cached > input {
                return Err(UnknownPrice::InvalidUsage);
            }
            let thinking = total
                .checked_sub(add(input, completion)?)
                .ok_or(UnknownPrice::InvalidUsage)?;
            let mut reasoning = None;
            if let Some(details) = object(usage, "completion_tokens_details")? {
                allowed_keys(details, &["reasoning_tokens"])?;
                reasoning = number(details.get("reasoning_tokens"))?;
            }
            // A reasoning count is either part of completion_tokens (nothing
            // left over) or exactly the tokens total_tokens adds.
            match reasoning {
                Some(count) if thinking == 0 && count <= completion => {}
                Some(count) if count == thinking => {}
                Some(_) => return Err(UnknownPrice::InvalidUsage),
                None => {}
            }
            (input - cached, cached, 0, 0, add(completion, thinking)?)
        }
        Provider::DeepSeek => {
            allowed_keys(
                usage,
                &[
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                    "prompt_tokens_details",
                    "completion_tokens_details",
                    "prompt_cache_hit_tokens",
                    "prompt_cache_miss_tokens",
                ],
            )?;
            let input = required(usage, "prompt_tokens", None)?;
            let output = required(usage, "completion_tokens", None)?;
            let mut hits = [number(usage.get("prompt_cache_hit_tokens"))?, None];
            if let Some(details) = object(usage, "prompt_tokens_details")? {
                allowed_keys(details, &["cached_tokens"])?;
                hits[1] = number(details.get("cached_tokens"))?;
            }
            let hit = agree(hits)?;
            if hit > input {
                return Err(UnknownPrice::InvalidUsage);
            }
            if let Some(miss) = number(usage.get("prompt_cache_miss_tokens"))?
                && add(hit, miss)? != input
            {
                return Err(UnknownPrice::InvalidUsage);
            }
            // Reasoning is part of completion_tokens.
            if let Some(details) = object(usage, "completion_tokens_details")? {
                allowed_keys(details, &["reasoning_tokens"])?;
                if optional(details, "reasoning_tokens")? > output {
                    return Err(UnknownPrice::InvalidUsage);
                }
            }
            (input - hit, hit, 0, 0, output)
        }
        Provider::OpenRouter => unreachable!("OpenRouter reports its charge"),
    };
    let all_input = add(add(add(base, cached)?, written)?, written_1h)?;
    if tariff
        .max_priced_input
        .is_some_and(|limit| all_input > limit)
    {
        return Err(UnknownPrice::Context);
    }
    if let Some(until) = tariff.until
        && span.is_none_or(|span| span.done >= until)
    {
        return Err(UnknownPrice::Period);
    }
    let rates = match (tariff.long, tariff.off_peak) {
        (Some(long), _) if all_input > long.above => long.rates,
        // Both ends of the request must fall in one known period.
        (_, Some(off_peak)) => {
            let span = span
                .filter(|span| span.sent <= span.done)
                .ok_or(UnknownPrice::Period)?;
            match (
                catalog::deepseek_peak(span.sent),
                catalog::deepseek_peak(span.done),
            ) {
                (Some(false), Some(false)) => off_peak,
                (Some(true), Some(true)) => tariff.rates,
                _ => return Err(UnknownPrice::Period),
            }
        }
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
    Ok((amount, catalog::SNAPSHOT, tariff.family))
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

        // GPT-4.1 has no cache-write charge: a written token is ordinary input.
        let older = json!({"usage":{"prompt_tokens":10,"completion_tokens":1,
            "prompt_tokens_details":{"cached_tokens":2,"cache_write_tokens":0}}});
        let listed = 8 * 2_000_000 + 2 * 500_000 + 8_000_000;
        assert_eq!(amount(&openai(), &older, BillingClass::Standard), listed);
        let mut written = older;
        written["usage"]["prompt_tokens_details"]["cache_write_tokens"] = json!(4);
        assert_eq!(amount(&openai(), &written, BillingClass::Standard), listed);
        assert_eq!(amount(&openai(), &written, BillingClass::Batch), listed / 2);
    }

    #[test]
    fn openai_long_context_reprices_a_whole_prompt_over_272_000_tokens() {
        let response = |prompt: u64| {
            json!({"usage":{"prompt_tokens":prompt,"completion_tokens":1000,
                "prompt_tokens_details":{"cached_tokens":100_000,"cache_write_tokens":50_000}}})
        };
        let short = 122_000 * 100_000 + 100_000 * 10_000 + 50_000 * 125_000 + 1000 * 500_000;
        assert_eq!(
            amount(&luna(), &response(272_000), BillingClass::Standard),
            short
        );
        // OpenAI's "272K" is 272,000 tokens: one more takes the long rates.
        let long = 122_001 * 200_000 + 100_000 * 20_000 + 50_000 * 250_000 + 1000 * 750_000;
        assert_eq!(
            amount(&luna(), &response(272_001), BillingClass::Standard),
            long
        );
        assert_eq!(
            amount(&luna(), &response(272_001), BillingClass::Batch),
            long / 2
        );
    }

    fn haiku_5_5() -> Identity<'static> {
        Identity {
            model: "claude-haiku-5-5",
            ..anthropic()
        }
    }

    #[test]
    fn haiku_5_5_prompt_over_100k_tokens_reprices_the_whole_request() {
        let response = |input: u64| {
            json!({"model":"claude-haiku-5-5","usage":{"input_tokens":input,"output_tokens":1000,
                "cache_read_input_tokens":60_000,"cache_creation_input_tokens":30_000,
                "cache_creation":{"ephemeral_5m_input_tokens":20_000,"ephemeral_1h_input_tokens":10_000}}})
        };
        // The prompt counts cache reads and writes: 10,000 + 60,000 + 30,000.
        let short = 10_000 * 100_000 + 60_000 * 10_000 + 20_000 * 125_000 + 10_000 * 200_000;
        assert_eq!(
            amount(&haiku_5_5(), &response(10_000), BillingClass::Standard),
            short + 1000 * 500_000
        );
        let long = 10_001 * 500_000 + 60_000 * 50_000 + 20_000 * 625_000 + 10_000 * 1_000_000;
        assert_eq!(
            amount(&haiku_5_5(), &response(10_001), BillingClass::Standard),
            long + 1000 * 2_500_000
        );
        assert_eq!(
            amount(&haiku_5_5(), &response(10_001), BillingClass::Batch),
            (long + 1000 * 2_500_000) / 2
        );
        // Unlike Claude 4.5, the reviewed bands cover the whole context window.
        assert!(matches!(
            quote(&haiku_5_5(), &response(900_000), BillingClass::Standard),
            Quote::Known { .. }
        ));
        let mut other = response(10_000);
        other["model"] = json!("claude-haiku-4-5-20251001");
        assert_eq!(
            quote(&haiku_5_5(), &other, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::ResponseModel)
        );
    }

    #[test]
    fn a_real_haiku_5_5_response_is_priced() {
        // The `usage` of an actual claude-haiku-5-5 response (2026-10-17).
        let response = json!({"model":"claude-haiku-5-5","usage":{"input_tokens":17,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_5m_input_tokens":0,"ephemeral_1h_input_tokens":0},"output_tokens":4,"output_tokens_details":{"thinking_tokens":0},"service_tier":"standard","inference_geo":"global","speed":"standard"}});
        assert_eq!(
            amount(&haiku_5_5(), &response, BillingClass::Standard),
            17 * 100_000 + 4 * 500_000
        );
    }

    #[test]
    fn a_standard_speed_response_is_priced_and_a_fast_one_is_not() {
        let response = |speed: &str| {
            json!({"model":"claude-haiku-5-5","usage":{"input_tokens":1000,"output_tokens":100,
                "speed":speed,"service_tier":"standard","inference_geo":"global"}})
        };
        assert_eq!(
            amount(&haiku_5_5(), &response("standard"), BillingClass::Standard),
            1000 * 100_000 + 100 * 500_000
        );
        assert_eq!(
            quote(&haiku_5_5(), &response("fast"), BillingClass::Standard),
            Quote::Unknown(UnknownPrice::UnsupportedUsage)
        );
    }

    #[test]
    fn messages_usage_breakdowns_and_null_counters_follow_the_api_reference() {
        // The Messages API reference's usage shape, without server tool use.
        let mut response = json!({"model":"claude-haiku-5-5","usage":{
            "cache_creation":{"ephemeral_1h_input_tokens":0,"ephemeral_5m_input_tokens":0},
            "cache_creation_input_tokens":0,"cache_read_input_tokens":2051,
            "inference_geo":"global","input_tokens":2095,"output_tokens":503,
            "output_tokens_details":{"thinking_tokens":120},
            "server_tool_use":{"web_fetch_requests":0,"web_search_requests":0},
            "service_tier":"standard"}});
        let listed = 2095 * 100_000 + 2051 * 10_000 + 503 * 500_000;
        assert_eq!(
            amount(&haiku_5_5(), &response, BillingClass::Standard),
            listed
        );
        for key in [
            "cache_creation",
            "cache_creation_input_tokens",
            "server_tool_use",
        ] {
            response["usage"][key] = Value::Null;
        }
        assert_eq!(
            amount(&haiku_5_5(), &response, BillingClass::Standard),
            listed
        );
        let mut thinking = response.clone();
        thinking["usage"]["output_tokens_details"]["thinking_tokens"] = json!(504);
        assert_eq!(
            quote(&haiku_5_5(), &thinking, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::InvalidUsage)
        );
        for tools in [
            json!({"web_fetch_requests":2,"web_search_requests":0}),
            json!({"web_search_requests":0,"code_execution_requests":0}),
        ] {
            response["usage"]["server_tool_use"] = tools;
            assert_eq!(
                quote(&haiku_5_5(), &response, BillingClass::Standard),
                Quote::Unknown(UnknownPrice::UnsupportedUsage)
            );
        }
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
    fn current_claude_and_openai_models_are_priced_at_their_listed_rates() {
        let claude = |model: &'static str| Identity {
            model,
            ..anthropic()
        };
        let usage = |model: &str| {
            json!({"model":model,"usage":{"input_tokens":1000,"output_tokens":100,
                "cache_read_input_tokens":2000,"cache_creation_input_tokens":300,
                "cache_creation":{"ephemeral_5m_input_tokens":200,"ephemeral_1h_input_tokens":100}}})
        };
        // (model, input, output, cache read, 5m write, 1h write) per token.
        for (model, rates) in [
            (
                "claude-fable-5-1",
                [10_000_000, 50_000_000, 250_000, 12_500_000, 20_000_000],
            ),
            (
                "claude-opus-5-5",
                [4_000_000, 20_000_000, 200_000, 5_000_000, 8_000_000],
            ),
            (
                "claude-sonnet-5-5",
                [2_000_000, 10_000_000, 100_000, 2_500_000, 4_000_000],
            ),
            (
                "claude-sonnet-4-6",
                [3_000_000, 15_000_000, 300_000, 3_750_000, 6_000_000],
            ),
        ] {
            let [input, output, read, five, hour] = rates;
            let standard = 1000 * input + 100 * output + 2000 * read + 200 * five + 100 * hour;
            assert_eq!(
                amount(&claude(model), &usage(model), BillingClass::Standard),
                standard,
                "{model}"
            );
            assert_eq!(
                amount(&claude(model), &usage(model), BillingClass::Batch),
                standard / 2,
                "{model}"
            );
            // The whole 1M-token window is at the listed rates.
            let mut long = usage(model);
            long["usage"]["input_tokens"] = json!(900_000);
            assert!(matches!(
                quote(&claude(model), &long, BillingClass::Standard),
                Quote::Known { .. }
            ));
        }
        // GPT-5.6 routes to GPT-5.6 Sol and is answered as that model.
        let alias = Identity {
            model: "gpt-5.6",
            ..openai()
        };
        let answered =
            json!({"model":"gpt-5.6-sol","usage":{"prompt_tokens":1000,"completion_tokens":10}});
        assert_eq!(
            amount(&alias, &answered, BillingClass::Standard),
            1000 * 4_000_000 + 10 * 20_000_000
        );
    }

    fn gemini() -> Identity<'static> {
        Identity {
            provider: "gemini",
            endpoint: "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            model: "gemini-flash-lite-latest",
        }
    }
    fn at(seconds: u64) -> Option<Span> {
        Some(Span {
            sent: seconds,
            done: seconds + 30,
        })
    }

    #[test]
    fn gemini_thinking_missing_from_completion_tokens_is_billed_as_output() {
        let price =
            |response: &Value| match quote_at(&gemini(), response, BillingClass::Standard, None) {
                Quote::Known { amount, .. } => Ok(amount.picodollars()),
                Quote::Unknown(reason) => Err(reason),
            };
        // The usage Gemini's OpenAI-compatible endpoint returned on 2026-10-09
        // (an answer withheld by its recitation filter).
        let withheld = json!({"model":"gemini-flash-lite-latest","usage":{
            "completion_tokens":0,"prompt_tokens":3903,"total_tokens":3903}});
        assert_eq!(price(&withheld), Ok(3903 * 300_000));
        // total_tokens counts 500 thinking tokens completion_tokens leaves out.
        let thinking = json!({"usage":{"prompt_tokens":1000,"completion_tokens":100,
            "total_tokens":1600,"prompt_tokens_details":{"cached_tokens":400}}});
        let billed = 600 * 300_000 + 400 * 30_000 + 600 * 2_500_000;
        assert_eq!(price(&thinking), Ok(billed));
        let mut counted = thinking.clone();
        counted["usage"]["completion_tokens_details"] = json!({"reasoning_tokens":500});
        assert_eq!(price(&counted), Ok(billed));
        let mut included = json!({"usage":{"prompt_tokens":1000,"completion_tokens":600,
            "total_tokens":1600,"completion_tokens_details":{"reasoning_tokens":500}}});
        assert_eq!(price(&included), Ok(1000 * 300_000 + 600 * 2_500_000));
        included["usage"]["completion_tokens_details"]["reasoning_tokens"] = json!(601);
        assert_eq!(price(&included), Err(UnknownPrice::InvalidUsage));
        counted["usage"]["completion_tokens_details"]["reasoning_tokens"] = json!(499);
        assert_eq!(price(&counted), Err(UnknownPrice::InvalidUsage));
        for usage in [
            json!({"prompt_tokens":10,"completion_tokens":1}),
            json!({"prompt_tokens":10,"completion_tokens":1,"total_tokens":10}),
            json!({"prompt_tokens":10,"completion_tokens":1,"total_tokens":11,
                "prompt_tokens_details":{"audio_tokens":3}}),
        ] {
            assert!(price(&json!({"usage":usage})).is_err());
        }
        // markitai submits no Gemini Batch; its cache rate there is not half.
        assert_eq!(
            quote(&gemini(), &withheld, BillingClass::Batch),
            Quote::Unknown(UnknownPrice::ServiceTier)
        );
    }

    #[test]
    fn gemini_flash_introductory_prices_end_with_2026() {
        let flash = Identity {
            model: "gemini-3.8-flash",
            ..gemini()
        };
        let response =
            json!({"usage":{"prompt_tokens":1000,"completion_tokens":100,"total_tokens":1100}});
        let listed = 1000 * 750_000 + 100 * 3_750_000;
        let at_time = |span| match quote_at(&flash, &response, BillingClass::Standard, span) {
            Quote::Known { amount, .. } => Some(amount.picodollars()),
            Quote::Unknown(reason) => {
                assert_eq!(reason, UnknownPrice::Period);
                None
            }
        };
        assert_eq!(at_time(at(1_791_763_200)), Some(listed));
        // 2027-01-01T00:00:00Z, when the doubled price starts.
        assert_eq!(at_time(at(1_798_761_600 - 30)), None);
        assert_eq!(at_time(at(1_798_761_600 - 31)), Some(listed));
        assert_eq!(at_time(None), None);
    }

    fn deepseek() -> Identity<'static> {
        Identity {
            provider: "deepseek",
            endpoint: "https://api.deepseek.com/v1/chat/completions",
            model: "deepseek-flash",
        }
    }

    #[test]
    fn deepseek_is_priced_by_its_peak_hours_in_beijing_time() {
        // The usage DeepSeek returned on Saturday 2026-10-10.
        let response = json!({"model":"deepseek-flash","usage":{"prompt_tokens":1585,
            "completion_tokens":401,"total_tokens":1986,"prompt_tokens_details":{"cached_tokens":1408},
            "completion_tokens_details":{"reasoning_tokens":211},
            "prompt_cache_hit_tokens":1408,"prompt_cache_miss_tokens":177}});
        let price = |span| match quote_at(&deepseek(), &response, BillingClass::Standard, span) {
            Quote::Known { amount, .. } => Ok(amount.picodollars()),
            Quote::Unknown(reason) => Err(reason),
        };
        let off_peak = 177 * 150_000 + 1408 * 3_000 + 401 * 600_000;
        assert_eq!(price(at(1_791_616_141)), Ok(off_peak));
        // Monday 2026-10-12, 10:00 in Beijing.
        let peak = 177 * 300_000 + 1408 * 6_000 + 401 * 1_200_000;
        assert_eq!(price(at(1_791_763_200 + 2 * 3600)), Ok(peak));
        assert_eq!(peak, 2 * off_peak);
        // A request that runs past 12:00 in Beijing spans both prices.
        assert_eq!(
            price(at(1_791_763_200 + 4 * 3600 - 10)),
            Err(UnknownPrice::Period)
        );
        assert_eq!(price(None), Err(UnknownPrice::Period));
        // Both base URLs DeepSeek documents reach the same tariff.
        let unversioned = Identity {
            endpoint: "https://api.deepseek.com/chat/completions",
            ..deepseek()
        };
        assert!(matches!(
            quote_at(
                &unversioned,
                &response,
                BillingClass::Standard,
                at(1_791_616_141)
            ),
            Quote::Known { .. }
        ));
        let mut split = response.clone();
        split["usage"]["prompt_cache_miss_tokens"] = json!(178);
        let mut details = response;
        details["usage"]["prompt_tokens_details"]["cached_tokens"] = json!(1407);
        for invalid in [split, details] {
            assert_eq!(
                quote_at(
                    &deepseek(),
                    &invalid,
                    BillingClass::Standard,
                    at(1_791_616_141)
                ),
                Quote::Unknown(UnknownPrice::InvalidUsage)
            );
        }
    }

    fn openrouter() -> Identity<'static> {
        Identity {
            provider: "openrouter",
            endpoint: "https://openrouter.ai/api/v1/chat/completions",
            model: "google/gemini-3.5-flash-lite",
        }
    }

    #[test]
    fn openrouter_requests_cost_what_openrouter_reports_charging() {
        // The usage OpenRouter returned on 2026-10-10.
        let response = json!({"model":"google/gemini-3.5-flash-lite","usage":{
            "prompt_tokens":1233,"completion_tokens":152,"total_tokens":1385,"cost":0.0007499,
            "is_byok":false,"prompt_tokens_details":{"cached_tokens":0,"cache_write_tokens":0,
                "audio_tokens":0,"video_tokens":0},
            "cost_details":{"upstream_inference_cost":0.0007499,
                "upstream_inference_prompt_cost":0.0003699,"upstream_inference_completions_cost":0.00038},
            "completion_tokens_details":{"reasoning_tokens":0,"image_tokens":0,"audio_tokens":0}}});
        assert_eq!(
            quote(&openrouter(), &response, BillingClass::Standard),
            Quote::Known {
                amount: Usd(749_900_000),
                snapshot: OPENROUTER_REPORTED,
                tariff: "openrouter/usage.cost",
            }
        );
        // Any routed model: OpenRouter's figure is the charge.
        let alias = Identity {
            model: "~anthropic/claude-haiku-latest",
            ..openrouter()
        };
        assert!(matches!(
            quote(&alias, &response, BillingClass::Standard),
            Quote::Known { .. }
        ));
        let mut byok = response.clone();
        byok["usage"]["is_byok"] = json!(true);
        let mut missing = response.clone();
        missing["usage"].as_object_mut().unwrap().remove("cost");
        let mut negative = response.clone();
        negative["usage"]["cost"] = json!(-0.1);
        for (invalid, reason) in [
            (byok, UnknownPrice::UnsupportedUsage),
            (missing, UnknownPrice::MissingUsage),
            (negative, UnknownPrice::InvalidUsage),
        ] {
            assert_eq!(
                quote(&openrouter(), &invalid, BillingClass::Standard),
                Quote::Unknown(reason)
            );
        }
        assert_eq!(
            quote(&openrouter(), &response, BillingClass::Batch),
            Quote::Unknown(UnknownPrice::ServiceTier)
        );
        let proxied = Identity {
            endpoint: "https://proxy.example/api/v1/chat/completions",
            ..openrouter()
        };
        assert_eq!(
            quote(&proxied, &response, BillingClass::Standard),
            Quote::Unknown(UnknownPrice::Endpoint)
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
