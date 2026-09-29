use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::HashMap;

pub(super) fn revision(models: &[Value], providers: &[Value]) -> String {
    let bytes = canonical(&json!({"models":models,"providers":providers}));
    markitai_core::hex(Sha256::digest(bytes.as_bytes()))
}

pub(super) fn id(entry: &Value, index: usize) -> String {
    if let Some(value) = entry
        .pointer("/model_info/id")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
    {
        return value.to_owned();
    }
    let params = &entry["litellm_params"];
    let data = json!({"routing_group":entry["model_name"],"model":params["model"],"api_base":params["api_base"],"weight":params.get("weight").unwrap_or(&json!(1)),"index":index});
    let raw = canonical(&data);
    let mut ascii = String::with_capacity(raw.len());
    use std::fmt::Write;
    for character in raw.chars() {
        if character <= '~' {
            ascii.push(character);
        } else {
            for unit in character.encode_utf16(&mut [0; 2]) {
                write!(ascii, "\\u{unit:04x}").unwrap();
            }
        }
    }
    format!(
        "legacy-{}",
        markitai_core::hex(Sha256::digest(ascii.as_bytes()))
    )[..27]
        .to_owned()
}

pub(super) fn backfill(models: &mut [Value]) -> HashMap<String, String> {
    let mut result = HashMap::new();
    for (index, entry) in models.iter_mut().enumerate() {
        if entry
            .pointer("/model_info/id")
            .and_then(Value::as_str)
            .is_some_and(|id| !id.is_empty())
        {
            continue;
        }
        let old = id(entry, index);
        let next = uuid::Uuid::new_v4().to_string();
        if !entry["model_info"].is_object() {
            entry["model_info"] = json!({});
        }
        entry["model_info"]["id"] = json!(next);
        result.insert(old, next);
    }
    result
}

// Python's JSON revision protocol uses compact UTF-8 and its float exponent spelling.
fn canonical(value: &Value) -> String {
    match value {
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(canonical).collect::<Vec<_>>().join(",")
        ),
        Value::Object(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(key, value)| format!(
                    "{}:{}",
                    serde_json::to_string(key).unwrap(),
                    canonical(value)
                ))
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Number(number) if number.is_f64() => python_float(&number.to_string()),
        _ => value.to_string(),
    }
}
fn python_float(raw: &str) -> String {
    let (sign, raw) = raw.strip_prefix('-').map_or(("", raw), |v| ("-", v));
    let (mantissa, exponent) = raw
        .split_once(['e', 'E'])
        .map_or((raw, 0), |(m, e)| (m, e.parse::<i32>().unwrap()));
    let mut position = mantissa.find('.').unwrap_or(mantissa.len()) as i32 + exponent;
    let mut digits = mantissa.replace('.', "");
    let leading = digits.bytes().take_while(|byte| *byte == b'0').count();
    if leading == digits.len() {
        return format!("{sign}0.0");
    }
    digits.drain(..leading);
    position -= leading as i32;
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let scientific = position - 1;
    if !(-4..16).contains(&scientific) {
        let fraction = if digits.len() > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        format!(
            "{sign}{}{fraction}e{}{abs:02}",
            &digits[..1],
            if scientific < 0 { "-" } else { "+" },
            abs = scientific.unsigned_abs()
        )
    } else if position <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-position) as usize))
    } else if position as usize >= digits.len() {
        format!(
            "{sign}{digits}{}.0",
            "0".repeat(position as usize - digits.len())
        )
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..position as usize],
            &digits[position as usize..]
        )
    }
}
