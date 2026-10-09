//! Reconcile overlapping runtime totals without inventing an API request count.
use super::*;
use crate::subscription::claude::{TokenCounts, UsageEvidence};
use std::collections::BTreeMap;

#[derive(Default)]
struct Row {
    requests: u64,
    // Uncached input, output, cache read, cache creation.
    tokens: [u64; 4],
    incomplete: u64,
}
fn counters(value: &TokenCounts) -> [Option<u64>; 4] {
    [
        value.input_tokens,
        value.output_tokens,
        value.cache_read_tokens,
        value.cache_write_tokens,
    ]
}
fn reconcile(row: &mut Row, totals: &TokenCounts, conflicting: &mut bool) {
    row.incomplete = 1;
    for (observed, reported) in row.tokens.iter_mut().zip(counters(totals)) {
        if let Some(reported) = reported {
            if reported < *observed {
                *conflicting = true;
            } else {
                *observed = reported;
            }
        }
    }
}

pub(super) fn observe_claude(
    usage: &mut ConversionUsage,
    entry: &Deployment,
    evidence: &UsageEvidence,
) -> Result<()> {
    let mut rows = BTreeMap::<String, Row>::new();
    for call in &evidence.calls {
        let row = rows.entry(call.model.clone()).or_default();
        row.requests += 1;
        for (total, value) in row.tokens.iter_mut().zip([
            call.input_tokens,
            call.output_tokens,
            call.cache_read_tokens,
            call.cache_write_tokens,
        ]) {
            *total = total.saturating_add(value.unwrap_or(0));
        }
    }
    let mut conflicting = false;
    if let Some(aggregate) = &evidence.aggregate {
        if aggregate.by_model.is_empty() {
            // A main-loop total cannot be allocated across multiple models.
            if rows.len() > 1 {
                conflicting = true;
                for row in rows.values_mut() {
                    row.incomplete = 1;
                }
            } else {
                let model = rows
                    .keys()
                    .next()
                    .cloned()
                    .unwrap_or_else(|| entry.id.clone());
                reconcile(
                    rows.entry(model).or_default(),
                    &aggregate.totals,
                    &mut conflicting,
                );
            }
        } else {
            // modelUsage includes auxiliary pipeline work and supersedes the
            // overlapping main-loop totals. Neither aggregate counts requests.
            for (model, totals) in &aggregate.by_model {
                let model = format!("claude-agent/{model}");
                reconcile(rows.entry(model).or_default(), totals, &mut conflicting);
            }
            for row in rows.values_mut() {
                row.incomplete = 1;
            }
        }
    }
    let mut delta = ConversionUsage::default();
    for (model, row) in rows {
        let input = row.tokens[0]
            .saturating_add(row.tokens[2])
            .saturating_add(row.tokens[3]);
        delta.requests = delta.requests.saturating_add(row.requests);
        delta.input_tokens = delta.input_tokens.saturating_add(input);
        delta.output_tokens = delta.output_tokens.saturating_add(row.tokens[1]);
        let mut value = json!({"requests":row.requests,"input_tokens":input,
            "output_tokens":row.tokens[1],"cached_input_tokens":row.tokens[2],
            "cache_creation_input_tokens":row.tokens[3],"cost_usd":0.0,
            "priced_requests":0,"unpriced_requests":row.requests,"cost_status":"unknown"});
        if row.incomplete > 0 {
            value["incomplete_request_observations"] = json!(row.incomplete);
        }
        delta.by_model.insert(model, value);
    }
    if !delta.by_model.is_empty() {
        DOCUMENT_ACCOUNTING.with(|slot| {
            if let Some(state) = slot.borrow().as_ref() {
                let mut state = state.lock().unwrap_or_else(|e| e.into_inner());
                state
                    .dollars
                    .observe(pricing::Quote::Unknown(pricing::UnknownPrice::Provider));
                merge_usage(&mut state.usage, &delta);
            }
        });
        merge_usage(usage, &delta);
    }
    if conflicting {
        Err(Error::Conversion(
            "Claude aggregate usage conflicts with observed API usage".into(),
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subscription::{ObservedCall, claude::AggregateUsage};
    fn entry() -> Deployment {
        Deployment {
            id: "claude-agent/sonnet".into(),
            explicit_id: None,
            group: "default".into(),
            model: "sonnet".into(),
            provider: "claude-agent".into(),
            weight: 1,
            key: None,
            endpoint: "claude-cli://fixture".into(),
            protocol: Protocol::Anthropic,
            max_tokens: None,
            reasoning_effort: None,
            supports_vision: None,
        }
    }
    fn counts(input: u64) -> TokenCounts {
        TokenCounts {
            input_tokens: Some(input),
            output_tokens: Some(7),
            cache_read_tokens: Some(4),
            cache_write_tokens: Some(6),
        }
    }
    fn evidence(calls: bool) -> UsageEvidence {
        UsageEvidence {
            calls: if calls {
                vec![ObservedCall {
                    model: "claude-agent/canonical".into(),
                    input_tokens: Some(11),
                    output_tokens: Some(5),
                    cache_read_tokens: Some(2),
                    cache_write_tokens: Some(3),
                }]
            } else {
                vec![]
            },
            aggregate: Some(AggregateUsage {
                totals: counts(999),
                by_model: BTreeMap::from([("canonical".into(), counts(20))]),
            }),
        }
    }
    #[test]
    fn overlapping_totals_replace_counts_without_adding_calls_or_main_loop() {
        let mut usage = ConversionUsage::default();
        observe_claude(&mut usage, &entry(), &evidence(true)).unwrap();
        assert_eq!(
            (usage.requests, usage.input_tokens, usage.output_tokens),
            (1, 30, 7)
        );
        assert_eq!(
            usage.by_model["claude-agent/canonical"]["incomplete_request_observations"],
            1
        );
        assert!(!usage.cost_complete());
        assert_eq!(usage.cost_usd, 0.0);
    }
    #[test]
    fn aggregate_only_tokens_survive_merge_delta_and_remain_unpriced() {
        let mut usage = ConversionUsage::default();
        observe_claude(&mut usage, &entry(), &evidence(false)).unwrap();
        assert_eq!(
            (usage.requests, usage.input_tokens, usage.output_tokens),
            (0, 30, 7)
        );
        assert!(!usage.cost_complete());
        let before = usage.clone();
        observe_claude(&mut usage, &entry(), &evidence(false)).unwrap();
        let delta = usage_difference(&usage, &before);
        assert_eq!(
            (delta.requests, delta.input_tokens, delta.output_tokens),
            (0, 30, 7)
        );
        assert_eq!(
            delta.by_model["claude-agent/canonical"]["incomplete_request_observations"],
            1
        );
        assert!(!delta.cost_complete());
        assert!(usage_difference(&usage, &usage).by_model.is_empty());
    }
    #[test]
    fn contradictory_aggregate_preserves_observed_lower_bound_and_returns_error() {
        let mut observed = evidence(true);
        observed
            .aggregate
            .as_mut()
            .unwrap()
            .by_model
            .insert("canonical".into(), counts(1));
        let mut usage = ConversionUsage::default();
        assert!(observe_claude(&mut usage, &entry(), &observed).is_err());
        assert_eq!(
            (usage.requests, usage.input_tokens, usage.output_tokens),
            (1, 21, 7)
        );
        assert!(!usage.cost_complete());
    }
    #[test]
    fn absent_evidence_never_invents_an_api_request_or_tokens() {
        let mut usage = ConversionUsage::default();
        observe_claude(&mut usage, &entry(), &UsageEvidence::default()).unwrap();
        assert_eq!(
            (usage.requests, usage.input_tokens, usage.output_tokens),
            (0, 0, 0)
        );
        assert!(usage.by_model.is_empty());
    }
}
