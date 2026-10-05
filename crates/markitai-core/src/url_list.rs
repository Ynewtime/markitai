//! Reading a `.urls` list: the file the CLI takes as an input and the workbench
//! accepts as an upload.
//!
//! Two forms are accepted, both written by hand. A JSON array holds URL strings
//! or objects with `url` and an optional `output_name`; anything else is one
//! entry per line, where a line starting with `#` is a comment and the first
//! word is the URL followed by an optional output name. A byte-order mark at the
//! start is ignored.
//!
//! The reader only splits the list; whether an entry is a usable HTTP(S) URL,
//! whether its output name is a plain filename, and how a repeated entry is
//! treated stay with the caller, which can also say which line or entry it
//! rejected.

/// Where an entry was found, so a caller can name it without echoing the text
/// (which may carry credentials).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    /// The n-th element of a JSON array, counting from 1.
    Entry(usize),
    /// The n-th line of a line-based list, counting from 1.
    Line(usize),
}

/// One entry of the list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub place: Place,
    pub url: String,
    /// The output name the entry asked for, without a `.md` suffix.
    pub output_name: Option<String>,
}

/// What a list holds: the entries and the JSON elements that were neither a
/// string nor an object with `url`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Parsed {
    pub entries: Vec<Entry>,
    pub skipped: Vec<Place>,
}

/// Splits a `.urls` list. The error is the parser's own message for a JSON list
/// that is not a JSON array; the caller adds the file it came from.
pub fn parse(text: &str) -> Result<Parsed, String> {
    let body = text.trim_start_matches('\u{feff}');
    let raw = body.trim();
    if raw.starts_with('[') {
        return parse_json(raw);
    }
    Ok(parse_lines(body))
}

fn parse_json(raw: &str) -> Result<Parsed, String> {
    let values: Vec<serde_json::Value> =
        serde_json::from_str(raw).map_err(|error| error.to_string())?;
    let mut parsed = Parsed::default();
    for (index, value) in values.into_iter().enumerate() {
        let place = Place::Entry(index + 1);
        let pair = if let Some(url) = value.as_str() {
            Some((url.to_owned(), None))
        } else {
            value["url"].as_str().map(|url| {
                (
                    url.to_owned(),
                    value["output_name"].as_str().map(str::to_owned),
                )
            })
        };
        match pair {
            Some((url, output_name)) => parsed.entries.push(Entry {
                place,
                url,
                output_name,
            }),
            None => parsed.skipped.push(place),
        }
    }
    Ok(parsed)
}

fn parse_lines(body: &str) -> Parsed {
    let mut parsed = Parsed::default();
    for (number, line) in body.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (url, output_name) = line
            .split_once(char::is_whitespace)
            .map(|(url, name)| (url, Some(name.trim().trim_matches(['\'', '"']).to_owned())))
            .unwrap_or((line, None));
        parsed.entries.push(Entry {
            place: Place::Line(number + 1),
            url: url.to_owned(),
            output_name,
        });
    }
    parsed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(number: usize) -> Place {
        Place::Line(number)
    }

    /// One entry per line, `#` comments and blank lines skipped, an optional
    /// name after the URL.
    #[test]
    fn a_line_based_list_keeps_its_line_numbers() {
        let parsed = parse(
            "\u{feff}# notes\n\nhttps://example.com/a\n  https://example.com/b  'Report'  \n",
        )
        .unwrap();
        assert_eq!(parsed.skipped, vec![]);
        assert_eq!(
            parsed.entries,
            vec![
                Entry {
                    place: line(3),
                    url: "https://example.com/a".into(),
                    output_name: None
                },
                Entry {
                    place: line(4),
                    url: "https://example.com/b".into(),
                    output_name: Some("Report".into())
                },
            ]
        );
    }

    /// A JSON array holds strings or objects with `url` and `output_name`; an
    /// element that is neither is reported by its entry number.
    #[test]
    fn a_json_list_names_its_entries_and_reports_the_others() {
        let parsed = parse(
            r#"[ "https://example.com/a", {"url": "https://example.com/b", "output_name": "B"}, 7 ]"#,
        )
        .unwrap();
        assert_eq!(
            parsed.entries,
            vec![
                Entry {
                    place: Place::Entry(1),
                    url: "https://example.com/a".into(),
                    output_name: None
                },
                Entry {
                    place: Place::Entry(2),
                    url: "https://example.com/b".into(),
                    output_name: Some("B".into())
                },
            ]
        );
        assert_eq!(parsed.skipped, vec![Place::Entry(3)]);
    }

    /// A list that opens a JSON array but is not one reports the parser's own
    /// message rather than reading it as lines.
    #[test]
    fn a_damaged_json_list_reports_the_parse_error() {
        let error = parse("[\"https://example.com\",").unwrap_err();
        assert!(error.contains("EOF") || error.contains("line"), "{error}");
    }

    /// An empty list is not an error here; the caller decides what to say.
    #[test]
    fn an_empty_list_holds_nothing() {
        assert_eq!(parse("   \n\n").unwrap(), Parsed::default());
    }
}
