//! Observed-response accounting; no token estimation or speculative reservation.
use crate::pricing::{self, BillingClass, BudgetError, Identity, Quote, Usd};
use crate::{ConversionUsage, Error, Result};
use serde_json::{Value, json};
use std::collections::BTreeSet;

pub(super) struct Dollars {
    limit: std::result::Result<Option<Usd>, BudgetError>,
    spent: Usd,
    incomplete: bool,
    overflow: bool,
}
impl Default for Dollars {
    fn default() -> Self {
        Self::new(&Value::Null)
    }
}
impl Dollars {
    pub(super) fn new(cfg: &Value) -> Self {
        let limit = match cfg.pointer("/llm/max_cost_per_document_usd") {
            None => Ok(None),
            Some(value) => value
                .as_f64()
                .ok_or(BudgetError::Invalid)
                .and_then(pricing::budget_from_f64),
        };
        Self {
            limit,
            spent: Usd::default(),
            incomplete: false,
            overflow: false,
        }
    }
    pub(super) fn admit(&self, identity: Option<&Identity<'_>>) -> Result<()> {
        let limit = self.limit.map_err(|_| {
            Error::Config(
                "LLM dollar budget is invalid or exceeds the supported fixed-point range".into(),
            )
        })?;
        let Some(limit) = limit else {
            return Ok(());
        };
        if self.incomplete || self.overflow {
            return Err(Error::Conversion("LLM dollar budget cannot continue because an observed response could not be priced completely".into()));
        }
        if self.spent > limit {
            return Err(Error::Conversion(
                "LLM per-document dollar budget exhausted".into(),
            ));
        }
        let identity = identity.ok_or_else(|| {
            Error::Config("LLM dollar budget requires a verified model and endpoint tariff".into())
        })?;
        pricing::preflight(identity, BillingClass::Standard).map_err(|reason| {
            Error::Unsupported(format!(
                "LLM dollar budget requires a verified tariff: {}",
                reason.reason()
            ))
        })
    }
    pub(super) fn observe(&mut self, quote: Quote) {
        match quote {
            Quote::Known { amount, .. } => match self.spent.checked_add(amount) {
                Some(total) => self.spent = total,
                None => self.overflow = true,
            },
            Quote::Unknown(_) => self.incomplete = true,
        }
    }
}

fn count(value: &Value, name: &str) -> u64 {
    value.get(name).and_then(Value::as_u64).unwrap_or(0)
}
fn cost(value: &Value) -> f64 {
    value
        .get("cost_usd")
        .and_then(Value::as_f64)
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(0.0)
}
fn price_counts(value: &Value) -> (u64, u64) {
    let requests = count(value, "requests");
    let priced = count(value, "priced_requests").min(requests);
    // Missing legacy counts are unknown even if an old numeric cost is present.
    (priced, requests.saturating_sub(priced))
}
fn snapshots(value: &Value, into: &mut BTreeSet<String>) {
    if price_counts(value).0 == 0 {
        return;
    }
    if let Some(snapshot) = value.get("pricing_snapshot").and_then(Value::as_str) {
        into.insert(snapshot.into());
    }
    if let Some(values) = value.get("pricing_snapshots").and_then(Value::as_array) {
        into.extend(values.iter().filter_map(Value::as_str).map(str::to_owned));
    }
}
fn annotate(value: &mut Value, priced: u64, unpriced: u64, snapshots: BTreeSet<String>) {
    value["priced_requests"] = json!(priced);
    value["unpriced_requests"] = json!(unpriced);
    value["cost_status"] = json!(match (
        priced > 0,
        unpriced > 0 || count(value, "incomplete_request_observations") > 0
    ) {
        (true, false) => "complete",
        (true, true) => "partial",
        _ => "unknown",
    });
    let object = value.as_object_mut().expect("usage rows are objects");
    object.remove("pricing_snapshot");
    object.remove("pricing_snapshots");
    if priced > 0 {
        if snapshots.len() == 1 {
            object.insert(
                "pricing_snapshot".into(),
                json!(snapshots.first().expect("one snapshot")),
            );
        } else if !snapshots.is_empty() {
            object.insert("pricing_snapshots".into(), json!(snapshots));
        }
    }
}

pub(super) fn response(model: &str, input: u64, output: u64, quote: Quote) -> ConversionUsage {
    let (amount, priced, snapshots) = match quote {
        Quote::Known {
            amount, snapshot, ..
        } => (amount.to_f64(), 1, BTreeSet::from([snapshot.into()])),
        Quote::Unknown(_) => (0.0, 0, BTreeSet::new()),
    };
    let mut detail =
        json!({"requests":1,"input_tokens":input,"output_tokens":output,"cost_usd":amount});
    annotate(&mut detail, priced, 1 - priced, snapshots);
    if let Quote::Unknown(reason) = quote {
        detail["unpriced_reason"] = json!(reason.reason());
    }
    let mut usage = ConversionUsage {
        cost_usd: amount,
        requests: 1,
        input_tokens: input,
        output_tokens: output,
        ..Default::default()
    };
    usage.by_model.insert(model.to_owned(), detail);
    usage
}

pub(super) fn merge(target: &mut ConversionUsage, source: &ConversionUsage) {
    target.requests = target.requests.saturating_add(source.requests);
    target.input_tokens = target.input_tokens.saturating_add(source.input_tokens);
    target.output_tokens = target.output_tokens.saturating_add(source.output_tokens);
    target.cost_usd += source.cost_usd;
    for (model, values) in &source.by_model {
        let entry = target.by_model.entry(model.clone()).or_insert_with(
            || json!({"requests":0,"input_tokens":0,"output_tokens":0,"cost_usd":0.0}),
        );
        let (old_priced, old_unpriced) = price_counts(entry);
        let (priced, unpriced) = price_counts(values);
        let mut sources = BTreeSet::new();
        snapshots(entry, &mut sources);
        snapshots(values, &mut sources);
        for name in [
            "requests",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "incomplete_request_observations",
            "timed_out_requests",
        ] {
            if matches!(
                name,
                "cache_creation_input_tokens"
                    | "incomplete_request_observations"
                    | "timed_out_requests"
            ) && entry.get(name).is_none()
                && values.get(name).is_none()
            {
                continue;
            }
            entry[name] = json!(count(entry, name).saturating_add(count(values, name)));
        }
        entry["cost_usd"] = json!(cost(entry) + cost(values));
        if entry.get("unpriced_reason").is_none()
            && let Some(reason) = values.get("unpriced_reason")
        {
            entry["unpriced_reason"] = reason.clone();
        }
        annotate(
            entry,
            old_priced.saturating_add(priced),
            old_unpriced.saturating_add(unpriced),
            sources,
        );
    }
}

pub(super) fn difference(after: &ConversionUsage, before: &ConversionUsage) -> ConversionUsage {
    let mut usage = after.clone();
    usage.requests = usage.requests.saturating_sub(before.requests);
    usage.input_tokens = usage.input_tokens.saturating_sub(before.input_tokens);
    usage.output_tokens = usage.output_tokens.saturating_sub(before.output_tokens);
    usage.cost_usd = (usage.cost_usd - before.cost_usd).max(0.0);
    for (model, entry) in &mut usage.by_model {
        let (mut priced, mut unpriced) = price_counts(entry);
        let mut sources = BTreeSet::new();
        snapshots(entry, &mut sources);
        if let Some(old) = before.by_model.get(model) {
            let (old_priced, old_unpriced) = price_counts(old);
            priced = priced.saturating_sub(old_priced);
            unpriced = unpriced.saturating_sub(old_unpriced);
            for name in [
                "requests",
                "input_tokens",
                "output_tokens",
                "cached_input_tokens",
                "cache_creation_input_tokens",
                "incomplete_request_observations",
                "timed_out_requests",
            ] {
                if matches!(
                    name,
                    "cache_creation_input_tokens"
                        | "incomplete_request_observations"
                        | "timed_out_requests"
                ) && entry.get(name).is_none()
                    && old.get(name).is_none()
                {
                    continue;
                }
                entry[name] = json!(count(entry, name).saturating_sub(count(old, name)));
            }
            entry["cost_usd"] = json!((cost(entry) - cost(old)).max(0.0));
        }
        annotate(entry, priced, unpriced, sources);
    }
    usage.by_model.retain(|_, entry| {
        count(entry, "requests") > 0
            || count(entry, "input_tokens") > 0
            || count(entry, "output_tokens") > 0
            || count(entry, "incomplete_request_observations") > 0
    });
    usage
}

#[cfg(test)]
mod tests;
