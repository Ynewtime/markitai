//! Cost values are known subtotals; coverage belongs to recorded requests.
use markitai_core::ConversionUsage;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CostStatus {
    Complete,
    Partial,
    Unknown,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Pricing {
    pub(crate) priced_requests: u64,
    pub(crate) unpriced_requests: u64,
    pub(crate) cost_status: CostStatus,
    pub(crate) pricing_snapshots: Vec<String>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub(crate) incomplete_request_observations: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

fn status(priced: u64, unpriced: u64, incomplete: u64) -> CostStatus {
    if priced == 0 {
        CostStatus::Unknown
    } else if unpriced > 0 || incomplete > 0 {
        CostStatus::Partial
    } else {
        CostStatus::Complete
    }
}

fn snapshots(value: &Value) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    for value in value.get("pricing_snapshot").into_iter().chain(
        value
            .get("pricing_snapshots")
            .and_then(Value::as_array)
            .into_iter()
            .flatten(),
    ) {
        if let Some(name) = value.as_str().filter(|name| {
            !name.is_empty() && name.len() <= 256 && !name.chars().any(char::is_control)
        }) {
            output.insert(name.to_owned());
        }
    }
    output
}

impl Pricing {
    fn model(record: &Value) -> Option<Self> {
        let requests = record.get("requests").and_then(Value::as_u64).unwrap_or(0);
        let incomplete = record
            .get("incomplete_request_observations")
            .and_then(Value::as_u64)
            .unwrap_or(0);
        if requests == 0 && incomplete == 0 {
            return None;
        }
        let counts = record
            .get("priced_requests")
            .and_then(Value::as_u64)
            .zip(record.get("unpriced_requests").and_then(Value::as_u64))
            .filter(|(priced, unpriced)| priced.checked_add(*unpriced) == Some(requests));
        let (priced, unpriced) = counts
            .filter(|(priced, unpriced)| {
                serde_json::to_value(status(*priced, *unpriced, incomplete))
                    .ok()
                    .as_ref()
                    == record.get("cost_status")
            })
            .unwrap_or((0, requests));
        Some(Self {
            priced_requests: priced,
            unpriced_requests: unpriced,
            cost_status: status(priced, unpriced, incomplete),
            incomplete_request_observations: incomplete,
            pricing_snapshots: if priced > 0 {
                snapshots(record).into_iter().collect()
            } else {
                Vec::new()
            },
        })
    }

    pub(crate) fn from_models(models: &Map<String, Value>) -> Option<Self> {
        let records = models.values().filter_map(Self::model).collect::<Vec<_>>();
        Self::aggregate(records.iter())
    }

    pub(crate) fn from_usage(usage: &ConversionUsage) -> Option<Self> {
        let models = Self::from_models(&usage.by_model);
        let recorded = models
            .as_ref()
            .and_then(|value| value.priced_requests.checked_add(value.unpriced_requests))
            .unwrap_or(0);
        let missing = usage.requests.saturating_sub(recorded);
        let Some(mut output) = models else {
            return (usage.requests > 0).then(|| Self {
                priced_requests: 0,
                unpriced_requests: usage.requests,
                cost_status: CostStatus::Unknown,
                incomplete_request_observations: 0,
                pricing_snapshots: Vec::new(),
            });
        };
        output.unpriced_requests = output.unpriced_requests.checked_add(missing)?;
        output.cost_status = status(
            output.priced_requests,
            output.unpriced_requests,
            output.incomplete_request_observations,
        );
        Some(output)
    }

    pub(crate) fn aggregate<'a>(values: impl IntoIterator<Item = &'a Self>) -> Option<Self> {
        let mut priced = 0u64;
        let mut unpriced = 0u64;
        let mut snapshots = BTreeSet::new();
        let mut incomplete = 0u64;
        for value in values {
            if !value.valid() {
                return None;
            }
            priced = priced.checked_add(value.priced_requests)?;
            unpriced = unpriced.checked_add(value.unpriced_requests)?;
            incomplete = incomplete.checked_add(value.incomplete_request_observations)?;
            snapshots.extend(value.pricing_snapshots.iter().cloned());
        }
        if snapshots.len() > 256 {
            return None;
        }
        let requests = priced.checked_add(unpriced)?;
        (requests > 0 || incomplete > 0).then(|| Self {
            priced_requests: priced,
            unpriced_requests: unpriced,
            cost_status: status(priced, unpriced, incomplete),
            incomplete_request_observations: incomplete,
            pricing_snapshots: snapshots.into_iter().collect(),
        })
    }

    pub(crate) fn valid(&self) -> bool {
        self.priced_requests
            .checked_add(self.unpriced_requests)
            .is_some_and(|total| total > 0 || self.incomplete_request_observations > 0)
            && self.cost_status
                == status(
                    self.priced_requests,
                    self.unpriced_requests,
                    self.incomplete_request_observations,
                )
            && (self.priced_requests > 0 || self.pricing_snapshots.is_empty())
            && self.pricing_snapshots.len() <= 256
            && self.pricing_snapshots.iter().all(|name| {
                !name.is_empty() && name.len() <= 256 && !name.chars().any(char::is_control)
            })
    }
}

/// A malformed optional historical field must not discard its output item.
pub(crate) fn deserialize_optional<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<Pricing>, D::Error> {
    let value = Value::deserialize(deserializer)?;
    Ok(serde_json::from_value::<Pricing>(value)
        .ok()
        .filter(Pricing::valid))
}

/// Preserve request coverage independently of cost rounding or missing old metadata.
pub(crate) fn merge_models(
    combined: &mut Map<String, Value>,
    records: &Map<String, Value>,
) -> Result<(), String> {
    for (name, incoming) in records {
        let merged = combined.entry(name.clone()).or_insert_with(|| {
            json!({
                "requests":0, "input_tokens":0, "output_tokens":0,
                "cost_usd":0.0, "cached_input_tokens":0,
            })
        });
        let prior = Pricing::model(merged);
        let next = Pricing::model(incoming);
        let coverage = Pricing::aggregate(prior.iter().chain(next.iter()));
        if coverage.is_none() && (prior.is_some() || next.is_some()) {
            return Err("Report pricing coverage exceeds its counter or snapshot limit".into());
        }
        for key in [
            "requests",
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "incomplete_request_observations",
            "timed_out_requests",
        ] {
            if matches!(
                key,
                "cache_creation_input_tokens"
                    | "incomplete_request_observations"
                    | "timed_out_requests"
            ) && merged.get(key).is_none()
                && incoming.get(key).is_none()
            {
                continue;
            }
            let total = merged[key]
                .as_u64()
                .unwrap_or(0)
                .checked_add(incoming[key].as_u64().unwrap_or(0))
                .ok_or("Report usage counter overflow")?;
            merged[key] = json!(total);
        }
        let cost = merged["cost_usd"].as_f64().unwrap_or(0.0)
            + incoming["cost_usd"].as_f64().unwrap_or(0.0);
        if !cost.is_finite() {
            return Err("Report model cost total is not finite".into());
        }
        merged["cost_usd"] = json!(cost);
        if let Some(coverage) = coverage {
            merged["priced_requests"] = json!(coverage.priced_requests);
            merged["unpriced_requests"] = json!(coverage.unpriced_requests);
            merged["cost_status"] = json!(coverage.cost_status);
            let object = merged
                .as_object_mut()
                .ok_or("Report model usage must be an object")?;
            object.remove("pricing_snapshot");
            object.remove("pricing_snapshots");
            match coverage.pricing_snapshots.as_slice() {
                [] => {}
                [snapshot] => {
                    object.insert("pricing_snapshot".into(), json!(snapshot));
                }
                snapshots => {
                    object.insert("pricing_snapshots".into(), json!(snapshots));
                }
            }
        }
    }
    Ok(())
}

pub(crate) fn schema() -> Value {
    json!({"type":"object","required":["priced_requests","unpriced_requests","cost_status","pricing_snapshots"],"properties":{
        "priced_requests":{"type":"integer","minimum":0},"unpriced_requests":{"type":"integer","minimum":0},
        "cost_status":{"type":"string","enum":["complete","partial","unknown"]},
        "pricing_snapshots":{"type":"array","items":{"type":"string"}},
        "incomplete_request_observations":{"type":"integer","minimum":0}
    }})
}

#[cfg(test)]
mod tests {
    use super::*;
    fn usage(value: Value) -> ConversionUsage {
        ConversionUsage {
            by_model: value.as_object().unwrap().clone(),
            ..Default::default()
        }
    }
    #[test]
    fn unknown_legacy_and_explicit_zero_have_different_coverage() {
        assert!(Pricing::from_usage(&ConversionUsage::default()).is_none());
        assert!(
            Pricing::from_usage(&ConversionUsage {
                cost_usd: 12.0,
                ..Default::default()
            })
            .is_none()
        );
        let legacy =
            Pricing::from_usage(&usage(json!({"old":{"requests":2,"cost_usd":1.2}}))).unwrap();
        assert_eq!(
            (
                legacy.priced_requests,
                legacy.unpriced_requests,
                legacy.cost_status
            ),
            (0, 2, CostStatus::Unknown)
        );
        let zero = Pricing::from_usage(&usage(json!({"priced":{"requests":1,"cost_usd":0.0,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"catalog-v1"}}))).unwrap();
        assert_eq!(zero.cost_status, CostStatus::Complete);
        assert_eq!(zero.priced_requests, 1);
        let missing_model = Pricing::from_usage(&ConversionUsage {
            requests: 3,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(missing_model.unpriced_requests, 3);
    }
    #[test]
    fn aggregation_preserves_known_subtotal_and_multiple_snapshot_provenance() {
        let mut merged = Map::new();
        for (cost, snapshot) in [(0.25, "catalog-v1"), (0.5, "catalog-v2")] {
            merge_models(&mut merged, json!({"same":{"requests":1,"input_tokens":7,"output_tokens":3,"cost_usd":cost,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":snapshot}}).as_object().unwrap()).unwrap();
        }
        merge_models(
            &mut merged,
            json!({"same":{"requests":1,"input_tokens":2,"output_tokens":1,"cost_usd":0.0}})
                .as_object()
                .unwrap(),
        )
        .unwrap();
        assert_eq!(merged["same"]["cost_usd"], 0.75);
        assert_eq!(merged["same"]["requests"], 3);
        assert_eq!(merged["same"]["priced_requests"], 2);
        assert_eq!(merged["same"]["unpriced_requests"], 1);
        assert_eq!(merged["same"]["cost_status"], "partial");
        assert!(merged["same"].get("pricing_snapshot").is_none());
        assert_eq!(
            merged["same"]["pricing_snapshots"],
            json!(["catalog-v1", "catalog-v2"])
        );
        assert_eq!(
            Pricing::from_models(&merged).unwrap().cost_status,
            CostStatus::Partial
        );
    }
    #[test]
    fn tokens_without_a_request_count_remain_unknown_through_reports_and_history() {
        let row = json!({"requests":0,"input_tokens":30,"output_tokens":7,"cost_usd":0.0,
            "priced_requests":0,"unpriced_requests":0,"cost_status":"unknown","incomplete_request_observations":1});
        let unknown = Pricing::model(&row).unwrap();
        assert_eq!(unknown.cost_status, CostStatus::Unknown);
        assert_eq!(unknown.priced_requests + unknown.unpriced_requests, 0);
        assert!(unknown.valid());
        let encoded = serde_json::to_value(&unknown).unwrap();
        let decoded: Pricing = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded, unknown);
        let mut merged = Map::new();
        let models = json!({"subscription":row}).as_object().unwrap().clone();
        merge_models(&mut merged, &models).unwrap();
        merge_models(&mut merged, &models).unwrap();
        assert_eq!(merged["subscription"]["requests"], 0);
        assert_eq!(merged["subscription"]["input_tokens"], 60);
        assert_eq!(merged["subscription"]["incomplete_request_observations"], 2);
        assert!(merged["subscription"].get("timed_out_requests").is_none());
        // A timed-out request's marker sums like the observation it explains.
        let timed_out = json!({"timeout":{"requests":0,"input_tokens":0,"output_tokens":0,"cost_usd":0.0,
            "priced_requests":0,"unpriced_requests":0,"cost_status":"unknown",
            "incomplete_request_observations":1,"timed_out_requests":1}});
        let mut timeouts = Map::new();
        merge_models(&mut timeouts, timed_out.as_object().unwrap()).unwrap();
        merge_models(&mut timeouts, timed_out.as_object().unwrap()).unwrap();
        assert_eq!(timeouts["timeout"]["timed_out_requests"], 2);
        assert_eq!(timeouts["timeout"]["cost_status"], "unknown");
        let mut total = usage(Value::Object(merged));
        total.input_tokens = 60;
        total.output_tokens = 14;
        let coverage = Pricing::from_usage(&total).unwrap();
        assert_eq!(coverage.incomplete_request_observations, 2);
        assert_eq!(coverage.cost_status, CostStatus::Unknown);
        let known=Pricing::model(&json!({"requests":1,"priced_requests":1,"unpriced_requests":0,"cost_status":"complete","pricing_snapshot":"fixture"})).unwrap();
        let combined = Pricing::aggregate([&known, &coverage]).unwrap();
        assert_eq!(combined.cost_status, CostStatus::Partial);
        assert_eq!(
            (combined.priced_requests, combined.unpriced_requests),
            (1, 0)
        );
        assert!(combined.valid());
    }

    #[test]
    fn inconsistent_or_malformed_coverage_never_becomes_complete() {
        for fields in [
            json!({"priced_requests":2,"unpriced_requests":0,"cost_status":"complete"}),
            json!({"priced_requests":1,"unpriced_requests":0,"cost_status":"unknown"}),
            json!({"priced_requests":-1,"unpriced_requests":0,"cost_status":"complete"}),
        ] {
            let mut record = json!({"requests":1,"cost_usd":0.0});
            record
                .as_object_mut()
                .unwrap()
                .extend(fields.as_object().unwrap().clone());
            let coverage = Pricing::model(&record).unwrap();
            assert_eq!(coverage.cost_status, CostStatus::Unknown);
            assert_eq!(coverage.unpriced_requests, 1);
        }
        let mut models = json!({"m":{"requests":u64::MAX}})
            .as_object()
            .unwrap()
            .clone();
        assert!(
            merge_models(
                &mut models,
                json!({"m":{"requests":1}}).as_object().unwrap()
            )
            .is_err()
        );
    }
}
