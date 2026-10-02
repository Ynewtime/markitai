//! Ordering belongs to the public conversion envelope, not the internal JSON
//! representation used by configuration, cache keys and run-state hashes.
use serde::ser::{Serialize, SerializeMap, SerializeSeq, Serializer};
use serde_json::Value;

#[derive(Clone, Copy)]
enum Scope {
    Envelope,
    Items,
    Item,
    Totals,
    Pricing,
    Plain,
}

struct Ordered<'a>(&'a Value, Scope);

impl Scope {
    fn fields(self) -> &'static [&'static str] {
        match self {
            Self::Envelope => &["version", "ok", "error", "batch", "items", "totals"],
            Self::Item => &[
                "kind",
                "source",
                "status",
                "output",
                "error",
                "warnings",
                "skip_reason",
                "images",
                "screenshots",
                "cost_usd",
                "duration_s",
                "cache_hit",
                "fetch_cache_hit",
                "llm_cache_hit",
                "fetch_strategy",
                "source_file",
                "llm_usage",
                "diagnostics",
                "pricing",
            ],
            Self::Totals => &[
                "total",
                "completed",
                "failed",
                "skipped",
                "pending",
                "cost_usd",
                "duration_s",
                "pricing",
            ],
            Self::Pricing => &[
                "priced_requests",
                "unpriced_requests",
                "cost_status",
                "pricing_snapshots",
                "incomplete_request_observations",
            ],
            Self::Items | Self::Plain => &[],
        }
    }

    fn child(self, key: &str) -> Self {
        match (self, key) {
            (Self::Envelope, "items") => Self::Items,
            (Self::Envelope, "totals") => Self::Totals,
            (Self::Item | Self::Totals, "pricing") => Self::Pricing,
            _ => Self::Plain,
        }
    }
}

impl Serialize for Ordered<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        if let (Value::Array(items), Scope::Items) = (self.0, self.1) {
            let mut array = serializer.serialize_seq(Some(items.len()))?;
            for item in items {
                array.serialize_element(&Ordered(item, Scope::Item))?;
            }
            return array.end();
        }
        let fields = self.1.fields();
        let Some(object) = self.0.as_object().filter(|_| !fields.is_empty()) else {
            return self.0.serialize(serializer);
        };
        let mut map = serializer.serialize_map(Some(object.len()))?;
        for key in fields {
            if let Some(value) = object.get(*key) {
                map.serialize_entry(key, &Ordered(value, self.1.child(key)))?;
            }
        }
        // Retain future extension fields without inventing or dropping values.
        for (key, value) in object {
            if !fields.contains(&key.as_str()) {
                map.serialize_entry(key, value)?;
            }
        }
        map.end()
    }
}

pub(super) fn render(envelope: &Value) -> String {
    serde_json::to_string_pretty(&Ordered(envelope, Scope::Envelope))
        .expect("JSON values serialize")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn envelope_orders_public_fields_without_changing_payload_or_internal_maps() {
        let original = json!({
            "totals":{"duration_s":0,"cost_usd":0,"pending":0,"skipped":0,"failed":0,"completed":1,"total":1},
            "items":[{"status":"completed","source":"note.txt","kind":"file","error":null,
                "llm_usage":{"z":{"output_tokens":1,"input_tokens":2},"a":{"requests":1}},
                "pricing":{"cost_status":"complete","unpriced_requests":0,"priced_requests":1,"pricing_snapshots":["fixture"]}}],
            "batch":null,"error":null,"ok":true,"version":"1.0","extension":{"z":1,"a":2}
        });
        let before = serde_json::to_vec(&original).unwrap();
        let text = render(&original);
        assert!(text.starts_with("{\n  \"version\": \"1.0\",\n  \"ok\": true,\n  \"error\": null,\n  \"batch\": null,\n  \"items\": ["));
        assert!(text.contains(
            "\"kind\": \"file\",\n      \"source\": \"note.txt\",\n      \"status\": \"completed\""
        ));
        assert!(text.contains("\"total\": 1,\n    \"completed\": 1,\n    \"failed\": 0"));
        assert!(text.contains("\"priced_requests\": 1,\n        \"unpriced_requests\": 0,\n        \"cost_status\": \"complete\""));
        assert_eq!(serde_json::from_str::<Value>(&text).unwrap(), original);
        assert_eq!(serde_json::to_vec(&original).unwrap(), before);
    }
}
