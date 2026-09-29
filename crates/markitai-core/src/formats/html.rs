mod article;
mod code;
mod stream;

use crate::{Document, Error, Result};
use scraper::{ElementRef, Html, Selector};
use serde_json::{Map, Value};
use std::collections::{HashMap, HashSet};
use url::Url;

fn selector(query: &str) -> Selector {
    Selector::parse(query).expect("static selector")
}

fn plain(element: ElementRef<'_>) -> String {
    element
        .text()
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn meta(document: &Html, keys: &[&str]) -> Option<String> {
    for key in keys {
        for element in document.select(&selector("meta")) {
            let name = element
                .value()
                .attr("property")
                .or_else(|| element.value().attr("name"));
            if name.is_some_and(|name| name.eq_ignore_ascii_case(key))
                && let Some(value) = element
                    .value()
                    .attr("content")
                    .map(str::trim)
                    .filter(|v| !v.is_empty())
            {
                return Some(value.to_owned());
            }
        }
    }
    None
}

fn jsonld_documents(document: &Html) -> Vec<Value> {
    fn collect(value: Value, documents: &mut Vec<Value>, depth: usize) {
        if depth > 64 {
            return;
        }
        match value {
            Value::Object(mut object) => {
                let graph = object.remove("@graph");
                documents.push(Value::Object(object));
                if let Some(graph) = graph {
                    collect(graph, documents, depth + 1);
                }
            }
            Value::Array(values) => {
                for value in values {
                    collect(value, documents, depth + 1);
                }
            }
            _ => {}
        }
    }
    let mut documents = Vec::new();
    for script in document.select(&selector("script")) {
        if script
            .value()
            .attr("type")
            .is_some_and(|kind| kind.trim().eq_ignore_ascii_case("application/ld+json"))
            && let Ok(value) = serde_json::from_str(&script.text().collect::<String>())
        {
            collect(value, &mut documents, 0);
        }
    }
    documents
}

fn jsonld_text(documents: &[Value], key: &str) -> Option<String> {
    documents.iter().find_map(|document| {
        document
            .get(key)?
            .as_str()
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .map(str::to_owned)
    })
}

fn clean_title(value: &str, site: Option<&str>) -> Option<String> {
    let mut title = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if let Some(site) = site {
        let site = site.split_whitespace().collect::<Vec<_>>().join(" ");
        loop {
            let mut changed = false;
            for separator in [" | ", " - ", " -- ", " · ", " — ", " – "] {
                if let Some(trimmed) = title.strip_suffix(&format!("{separator}{site}")) {
                    title = trimmed.to_owned();
                    changed = true;
                    break;
                }
                if let Some(trimmed) = title.strip_prefix(&format!("{site}{separator}")) {
                    title = trimmed.to_owned();
                    changed = true;
                    break;
                }
            }
            if !changed {
                break;
            }
        }
    }
    if let Some((end, _)) = title.char_indices().nth(300) {
        title.truncate(end);
        if let Some(boundary) = title.rfind(' ').filter(|index| *index > 0) {
            title.truncate(boundary);
        }
        title.push('…');
    }
    (!title.is_empty()).then_some(title)
}

fn page_title(document: &Html, jsonld: &[Value], site: Option<&str>) -> Option<String> {
    let headline = jsonld_text(jsonld, "headline");
    let document_title = document.select(&selector("title")).next().map(plain);
    let social = meta(document, &["og:title", "twitter:title"]);
    let mut title = [
        headline.clone(),
        jsonld_text(jsonld, "name"),
        document_title.clone(),
        social.clone(),
    ]
    .into_iter()
    .flatten()
    .find_map(|value| clean_title(&value, site))?;
    if let Some(prefix) = title
        .strip_suffix('…')
        .or_else(|| title.strip_suffix("..."))
    {
        let prefix = prefix.trim_end();
        if let Some(candidate) = [document_title, social]
            .into_iter()
            .flatten()
            .filter_map(|value| clean_title(&value, site))
            .find(|value| {
                !value.ends_with('…')
                    && !value.ends_with("...")
                    && value.len() > title.len()
                    && value.starts_with(prefix)
            })
        {
            title = candidate;
        }
    }
    if title.contains(" | ") {
        let heading = document.select(&selector("h1")).next().map(plain);
        for candidate in [headline, heading].into_iter().flatten() {
            let candidate = candidate.split_whitespace().collect::<Vec<_>>().join(" ");
            if !candidate.is_empty() && title.starts_with(&format!("{candidate} | ")) {
                return Some(candidate);
            }
        }
    }
    Some(title)
}

fn escaped(value: &str, output: &mut String) {
    for ch in value.chars() {
        match ch {
            '&' => output.push_str("&amp;"),
            '<' => output.push_str("&lt;"),
            '>' => output.push_str("&gt;"),
            '"' => output.push_str("&quot;"),
            _ => output.push(ch),
        }
    }
}

fn safe_url(value: &str, base: Option<&Url>) -> Option<String> {
    fn destination(value: &str) -> String {
        value
            .replace('<', "%3C")
            .replace('>', "%3E")
            .replace('"', "%22")
    }
    let value = value.trim();
    if value.is_empty() || value.chars().any(char::is_control) {
        return None;
    }
    if value.starts_with('#') {
        return Some(destination(value));
    }
    if let Ok(url) = Url::parse(value) {
        return matches!(url.scheme(), "http" | "https" | "mailto" | "tel")
            .then(|| destination(value));
    }
    if let Some(base) = base {
        return base
            .join(value)
            .ok()
            .filter(|url| matches!(url.scheme(), "http" | "https"))
            .map(|url| url.to_string());
    }
    (!value.contains(':')).then(|| destination(value))
}

/// Inline image data is not an addressable resource. Like the reference, keep
/// a `data:<type>...` placeholder so alt text and position survive; active
/// document types are removed, and the payload itself is never retained.
fn image_source(value: &str, base: Option<&Url>) -> Option<String> {
    let value = value.trim();
    if !value
        .get(..5)
        .is_some_and(|scheme| scheme.eq_ignore_ascii_case("data:"))
    {
        return safe_url(value, base);
    }
    let lower = value.to_ascii_lowercase();
    if [
        "data:text/html",
        "data:image/svg+xml",
        "data:text/javascript",
    ]
    .iter()
    .any(|prefix| lower.starts_with(prefix))
    {
        return None;
    }
    let meta = value.split(',').next().unwrap_or(value);
    (meta.len() <= 256
        && !meta.chars().any(|ch| {
            ch.is_control()
                || ch.is_whitespace()
                || matches!(ch, '(' | ')' | '<' | '>' | '"' | '\\')
        }))
    .then(|| format!("{meta}..."))
}

fn block_tag(name: &str) -> bool {
    matches!(
        name,
        "html"
            | "body"
            | "article"
            | "main"
            | "section"
            | "div"
            | "p"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "pre"
            | "ul"
            | "ol"
            | "li"
            | "dl"
            | "dt"
            | "dd"
            | "table"
            | "thead"
            | "tbody"
            | "tfoot"
            | "tr"
            | "td"
            | "th"
            | "blockquote"
            | "figure"
            | "figcaption"
            | "details"
            | "summary"
            | "hr"
            | "header"
            | "footer"
    )
}

fn escaped_text(value: &str, output: &mut String) {
    // HTML source line boundaries remain soft Markdown line breaks. The marker
    // prevents the renderer from collapsing them with ordinary inline spaces.
    let mut start = 0;
    for (index, ch) in value.char_indices() {
        if ch == '\n' || ch == '\r' {
            let segment = &value[start..index];
            // The first segment can follow an inline element. Its leading
            // space separates words; only subsequent segments contain source
            // indentation following a line boundary.
            escaped(
                if start == 0 {
                    segment.trim_end_matches([' ', '\t'])
                } else {
                    segment.trim_matches([' ', '\t'])
                },
                output,
            );
            if !output.ends_with("</markitai-soft-break>") {
                output.push_str("<markitai-soft-break></markitai-soft-break>");
            }
            start = index + ch.len_utf8();
        }
    }
    escaped(
        if start > 0 {
            value[start..].trim_start_matches([' ', '\t'])
        } else {
            value
        },
        output,
    );
}

fn hidden_inline_style(style: &str) -> bool {
    // Parse declaration boundaries before inspecting property names. Semicolons
    // inside strings, comments or function/block values are not separators.
    fn record(declaration: &str, properties: &mut [Option<(bool, bool)>; 2]) {
        let Some((property, value)) = declaration.split_once(':') else {
            return;
        };
        let property = property.trim();
        let index = if property.eq_ignore_ascii_case("display") {
            0
        } else if property.eq_ignore_ascii_case("visibility") {
            1
        } else {
            return;
        };
        let value = value.trim();
        let (value, important) = value
            .rsplit_once('!')
            .filter(|(_, priority)| priority.trim().eq_ignore_ascii_case("important"))
            .map_or((value, false), |(value, _)| (value.trim(), true));
        // Strings and other invalid keyword values cannot create a hiding rule
        // or override an earlier valid declaration. CSS variable evaluation and
        // stylesheets belong to a browser's computed-style model.
        let keyword = value.to_ascii_lowercase();
        let hidden = match (index, keyword.as_str()) {
            (0, "none") | (1, "hidden" | "collapse") => true,
            (
                0,
                "block" | "inline" | "inline-block" | "flow-root" | "flex" | "inline-flex" | "grid"
                | "inline-grid" | "table" | "inline-table" | "table-row" | "table-cell"
                | "table-caption" | "table-column" | "table-column-group" | "table-row-group"
                | "table-header-group" | "table-footer-group" | "list-item" | "contents",
            )
            | (1, "visible")
            | (_, "initial" | "inherit" | "unset" | "revert" | "revert-layer") => false,
            _ => return,
        };
        if properties[index].is_none_or(|(_, previous_important)| important || !previous_important)
        {
            properties[index] = Some((hidden, important));
        }
    }

    let mut properties = [None; 2];
    let mut declaration = String::new();
    let mut quote = None;
    let mut nesting = 0usize;
    let mut characters = style.chars().peekable();
    while let Some(character) = characters.next() {
        if let Some(delimiter) = quote {
            declaration.push(character);
            if character == '\\' {
                if let Some(escaped) = characters.next() {
                    declaration.push(escaped);
                }
            } else if character == delimiter {
                quote = None;
            }
            continue;
        }
        if character == '/' && characters.peek() == Some(&'*') {
            characters.next();
            while let Some(comment) = characters.next() {
                if comment == '*' && characters.peek() == Some(&'/') {
                    characters.next();
                    break;
                }
            }
            declaration.push(' ');
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '(' | '[' | '{' => nesting += 1,
            ')' | ']' | '}' => nesting = nesting.saturating_sub(1),
            ';' if nesting == 0 => {
                record(&declaration, &mut properties);
                declaration.clear();
                continue;
            }
            '\\' => {
                declaration.push(character);
                if let Some(escaped) = characters.next() {
                    declaration.push(escaped);
                }
                continue;
            }
            _ => {}
        }
        declaration.push(character);
    }
    if quote.is_none() && nesting == 0 {
        record(&declaration, &mut properties);
    }
    properties.into_iter().flatten().any(|(hidden, _)| hidden)
}

fn is_hidden(element: ElementRef<'_>) -> bool {
    let value = element.value();
    value.attr("hidden").is_some() || value.attr("style").is_some_and(hidden_inline_style)
}

fn has_class(element: ElementRef<'_>, class: &str) -> bool {
    element
        .value()
        .attr("class")
        .is_some_and(|classes| classes.split_whitespace().any(|value| value == class))
}

fn tex_script(element: ElementRef<'_>) -> Option<bool> {
    if element.value().name() != "script" {
        return None;
    }
    let mut parts = element.value().attr("type")?.split(';');
    if !parts.next()?.trim().eq_ignore_ascii_case("math/tex") {
        return None;
    }
    Some(parts.any(|part| {
        part.split_once('=').is_some_and(|(key, value)| {
            key.trim().eq_ignore_ascii_case("mode") && value.trim().eq_ignore_ascii_case("display")
        })
    }))
}

fn math_symbol(value: &str) -> String {
    let symbol = match value {
        "α" => r"\alpha",
        "β" => r"\beta",
        "γ" => r"\gamma",
        "δ" => r"\delta",
        "ε" => r"\epsilon",
        "ζ" => r"\zeta",
        "η" => r"\eta",
        "θ" => r"\theta",
        "ι" => r"\iota",
        "κ" => r"\kappa",
        "λ" => r"\lambda",
        "μ" => r"\mu",
        "ν" => r"\nu",
        "ξ" => r"\xi",
        "π" => r"\pi",
        "ρ" => r"\rho",
        "σ" => r"\sigma",
        "τ" => r"\tau",
        "υ" => r"\upsilon",
        "ϕ" => r"\phi",
        "φ" => r"\varphi",
        "χ" => r"\chi",
        "ψ" => r"\psi",
        "ω" => r"\omega",
        "Γ" => r"\Gamma",
        "Δ" => r"\Delta",
        "Θ" => r"\Theta",
        "Λ" => r"\Lambda",
        "Ξ" => r"\Xi",
        "Π" => r"\Pi",
        "Σ" => r"\Sigma",
        "Φ" => r"\Phi",
        "Ψ" => r"\Psi",
        "Ω" => r"\Omega",
        "≠" => r"\neq",
        "≤" => r"\leq",
        "≥" => r"\geq",
        "±" => r"\pm",
        "∓" => r"\mp",
        "×" => r"\times",
        "÷" => r"\div",
        "∞" => r"\infty",
        "∑" => r"\sum",
        "∏" => r"\prod",
        "∫" => r"\int",
        "∮" => r"\oint",
        "∂" => r"\partial",
        "∇" => r"\nabla",
        "→" => r"\rightarrow",
        "←" => r"\leftarrow",
        "⇒" => r"\Rightarrow",
        "⇐" => r"\Leftarrow",
        "↦" => r"\mapsto",
        "∈" => r"\in",
        "∉" => r"\notin",
        "⊂" => r"\subset",
        "⊃" => r"\supset",
        "⊆" => r"\subseteq",
        "⊇" => r"\supseteq",
        "∪" => r"\cup",
        "∩" => r"\cap",
        "∧" => r"\wedge",
        "∨" => r"\vee",
        "¬" => r"\neg",
        "∅" => r"\emptyset",
        "∀" => r"\forall",
        "∃" => r"\exists",
        "⟹" => r"\implies",
        "⟸" => r"\impliedby",
        "⟺" => r"\iff",
        "⟨" => r"\langle",
        "⟩" => r"\rangle",
        "⌊" => r"\lfloor",
        "⌋" => r"\rfloor",
        "⌈" => r"\lceil",
        "⌉" => r"\rceil",
        "⋯" => r"\cdots",
        "…" => r"\ldots",
        "⋅" | "·" => r"\cdot",
        "−" => "-",
        "⁡" | "⁢" | "⁣" => "",
        _ => value,
    };
    symbol.to_owned()
}

fn inert_math_node(name: &str) -> bool {
    matches!(
        name,
        "script" | "style" | "annotation-xml" | "iframe" | "object" | "embed" | "template"
    )
}

fn math_text(element: ElementRef<'_>, depth: usize) -> Result<String> {
    if inert_math_node(element.value().name()) {
        return Ok(String::new());
    }
    if depth > 64 {
        return Err(Error::Conversion(
            "MathML nesting exceeds 64 elements".into(),
        ));
    }
    let mut result = String::new();
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            result.push_str(&math_text(child, depth + 1)?);
        } else if let scraper::Node::Text(text) = child.value() {
            result.push_str(text);
        }
    }
    Ok(result)
}

fn mathml(element: ElementRef<'_>, depth: usize) -> Result<String> {
    if inert_math_node(element.value().name()) {
        return Ok(String::new());
    }
    if depth > 64 {
        return Err(Error::Conversion(
            "MathML nesting exceeds 64 elements".into(),
        ));
    }
    let name = element.value().name();
    let children = element.child_elements().collect::<Vec<_>>();
    let mut parts = Vec::new();
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            if !matches!(child.value().name(), "annotation" | "annotation-xml") {
                parts.push(mathml(child, depth + 1)?);
            }
        } else if let scraper::Node::Text(text) = child.value()
            && !text.trim().is_empty()
        {
            parts.push(text.trim().to_owned());
        }
    }
    let joined = parts
        .iter()
        .filter(|part| !part.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ");
    let text = if matches!(name, "mi" | "mn" | "mo" | "mtext" | "ms") {
        math_text(element, depth)?
    } else {
        String::new()
    };
    let text = text.trim();
    let at = |index: usize| parts.get(index).map(String::as_str).unwrap_or("");
    Ok(match name {
        "semantics" => parts.into_iter().next().unwrap_or_default(),
        "annotation" | "annotation-xml" | "none" | "mprescripts" => String::new(),
        "mi" if text.chars().count() > 1 && text.chars().all(char::is_alphabetic) => {
            format!(r"\mathrm{{{text}}}")
        }
        "mi" | "mn" | "mo" => math_symbol(text),
        "mtext" | "ms" => {
            let mut escaped = String::new();
            for character in text.chars() {
                match character {
                    '\\' => escaped.push_str(r"\textbackslash{}"),
                    '{' | '}' | '$' | '&' | '%' | '#' | '_' => {
                        escaped.push('\\');
                        escaped.push(character);
                    }
                    '^' => escaped.push_str(r"\textasciicircum{}"),
                    '~' => escaped.push_str(r"\textasciitilde{}"),
                    _ => escaped.push(character),
                }
            }
            format!(r"\text{{{escaped}}}")
        }
        "mspace" => r"\;".into(),
        "msup" if parts.len() >= 2 => format!("{}^{{{}}}", at(0), at(1)),
        "msub" if parts.len() >= 2 => format!("{}_{{{}}}", at(0), at(1)),
        "msubsup" if parts.len() >= 3 => format!("{}_{{{}}}^{{{}}}", at(0), at(1), at(2)),
        "mfrac" if parts.len() >= 2 => format!(r"\frac{{{}}}{{{}}}", at(0), at(1)),
        "msqrt" => format!(r"\sqrt{{{joined}}}"),
        "mroot" if parts.len() >= 2 => format!(r"\sqrt[{}]{{{}}}", at(1), at(0)),
        "mover" if parts.len() >= 2 => {
            let raw = children
                .get(1)
                .map(|element| math_text(*element, depth + 1))
                .transpose()?
                .unwrap_or_default();
            let accent = match raw.trim() {
                "˙" | "̇" => Some("dot"),
                "¨" | "̈" => Some("ddot"),
                "¯" | "‾" => Some("overline"),
                "^" | "̂" => Some("hat"),
                "~" | "̃" | "˜" => Some("tilde"),
                "→" | "⃗" => Some("vec"),
                "⏞" => Some("overbrace"),
                _ => None,
            };
            accent.map_or_else(
                || format!(r"\overset{{{}}}{{{}}}", at(1), at(0)),
                |accent| format!("\\{accent}{{{}}}", at(0)),
            )
        }
        "munder" if parts.len() >= 2 => format!(r"\underset{{{}}}{{{}}}", at(1), at(0)),
        "munderover" if parts.len() >= 3 => format!("{}_{{{}}}^{{{}}}", at(0), at(1), at(2)),
        "mtable" => format!(r"\begin{{aligned}}{}\end{{aligned}}", parts.join(r" \\ ")),
        "mtr" | "mlabeledtr" => parts.join(" & "),
        "mfenced" => {
            let open = element.value().attr("open").unwrap_or("(");
            let close = element.value().attr("close").unwrap_or(")");
            let separators = element
                .value()
                .attr("separators")
                .unwrap_or(",")
                .chars()
                .filter(|ch| !ch.is_whitespace())
                .collect::<Vec<_>>();
            let mut content = String::new();
            for (index, part) in parts.iter().enumerate() {
                if index > 0 && !separators.is_empty() {
                    content.push(separators[(index - 1).min(separators.len() - 1)]);
                    content.push(' ');
                }
                content.push_str(part);
            }
            let delimiter = |value: &str| match value {
                "" => ".".into(),
                "{" => r"\{".into(),
                "}" => r"\}".into(),
                value => math_symbol(value),
            };
            format!(
                r"\left{}{content}\right{}",
                delimiter(open),
                delimiter(close)
            )
        }
        "menclose" if element.value().attr("notation") == Some("top") => {
            format!(r"\overline{{{joined}}}")
        }
        "mphantom" => format!(r"\phantom{{{joined}}}"),
        _ => joined,
    })
}

fn math_container(element: ElementRef<'_>) -> bool {
    element.value().name() == "math"
        || element.value().name() == "mjx-container"
        || [
            "katex",
            "katex-display",
            "math-inline",
            "math-block",
            "hurmet-tex",
            "mwe-math-element",
        ]
        .into_iter()
        .any(|class| has_class(element, class))
}

fn math_expression(element: ElementRef<'_>) -> Result<Option<(String, bool)>> {
    if let Some(block) = tex_script(element) {
        let text = element.text().collect::<String>();
        return Ok((!text.trim().is_empty()).then(|| (text.trim().to_owned(), block)));
    }
    if !math_container(element) {
        return Ok(None);
    }
    let inner = if element.value().name() == "math" {
        Some(element)
    } else {
        element.select(&selector("math")).next()
    };
    let block = has_class(element, "katex-display")
        || has_class(element, "math-block")
        || matches!(element.value().attr("display"), Some("block" | "true"))
        || inner.is_some_and(|math| math.value().attr("display") == Some("block"));
    let attribute = |element: ElementRef<'_>| {
        ["data-latex", "data-math", "data-entry", "alttext"]
            .into_iter()
            .find_map(|key| {
                element
                    .value()
                    .attr(key)
                    .map(str::trim)
                    .filter(|text| !text.is_empty())
                    .map(str::to_owned)
            })
    };
    let mut latex = attribute(element).or_else(|| inner.and_then(attribute));
    if latex.is_none() {
        latex = element
            .select(&selector("annotation"))
            .find(|node| {
                node.value().attr("encoding").is_some_and(|encoding| {
                    matches!(
                        encoding.to_ascii_lowercase().as_str(),
                        "application/x-tex" | "application/x-latex" | "text/tex"
                    )
                })
            })
            .map(|node| math_text(node, 0))
            .transpose()?
            .filter(|text| !text.trim().is_empty());
    }
    if latex.is_none()
        && let Some(math) = inner
    {
        latex = Some(mathml(math, 0)?);
    }
    if latex.is_none() && has_class(element, "mwe-math-element") {
        latex = element.select(&selector("img[alt]")).find_map(|image| {
            image
                .value()
                .attr("alt")
                .map(str::trim)
                .filter(|text| !text.is_empty())
                .map(str::to_owned)
        });
    }
    Ok(latex
        .filter(|text| !text.trim().is_empty())
        .map(|text| (text.trim().to_owned(), block)))
}

fn duplicate_math_preview(element: ElementRef<'_>) -> bool {
    [
        "MathJax_Preview",
        "MathJax",
        "MathJax_Display",
        "MathJax_SVG",
        "MathJax_MathML",
    ]
    .into_iter()
    .any(|class| has_class(element, class))
        && element
            .parent()
            .and_then(ElementRef::wrap)
            .is_some_and(|parent| {
                parent.child_elements().any(|sibling| {
                    tex_script(sibling).is_some()
                        && sibling.text().any(|text| !text.trim().is_empty())
                })
            })
}

fn emit_math(latex: &str, block: bool, output: &mut String) {
    output.push_str("<markitai-math data-latex=\"");
    escaped(latex, output);
    output.push_str(if block {
        "\" data-display=\"block\"></markitai-math>"
    } else {
        "\"></markitai-math>"
    });
}

// DOM element addresses serve only as stable identity keys while this immutable
// parse tree is borrowed. They are never dereferenced through these keys.
fn element_key(element: ElementRef<'_>) -> usize {
    element.value() as *const scraper::node::Element as usize
}

fn contains_element(parent: ElementRef<'_>, child: ElementRef<'_>) -> bool {
    parent == child || child.ancestors().any(|node| node.id() == parent.id())
}

fn note_heading(element: ElementRef<'_>) -> bool {
    matches!(
        element.value().name(),
        "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
    ) && matches!(
        plain(element).to_ascii_lowercase().as_str(),
        "footnote"
            | "footnotes"
            | "foot notes"
            | "endnotes"
            | "end notes"
            | "note"
            | "notes"
            | "reference"
            | "references"
            | "sidenotes"
    )
}

fn note_number(text: &str) -> Option<String> {
    let number = text.trim().trim_matches(['[', ']', '(', ')', '.']);
    (!number.is_empty()
        && number.len() <= 4
        && number.bytes().all(|byte| byte.is_ascii_digit())
        && !number.starts_with('0'))
    .then(|| number.to_owned())
}

fn first_content(element: ElementRef<'_>) -> Option<ElementRef<'_>> {
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            return Some(child);
        }
        if let scraper::Node::Text(text) = child.value()
            && !text.trim().is_empty()
        {
            return None;
        }
    }
    None
}

fn leading_note_marker(element: ElementRef<'_>) -> Option<(String, ElementRef<'_>)> {
    let mut first = first_content(element)?;
    if matches!(first.value().name(), "b" | "strong" | "span")
        && let Some(inner) = first_content(first)
        && inner.value().name() == "sup"
    {
        first = inner;
    }
    matches!(first.value().name(), "sup" | "strong")
        .then(|| note_number(&plain(first)).map(|number| (number, first)))?
}

fn note_context(element: ElementRef<'_>) -> bool {
    element.value().attr("role").is_some_and(|role| {
        matches!(
            role,
            "doc-footnote" | "doc-endnote" | "doc-endnotes" | "doc-footnotes"
        )
    }) || element
        .value()
        .attr("id")
        .is_some_and(|id| matches!(id, "footnotes" | "endnotes"))
        || element.value().attr("data-footnotes").is_some()
        || element.value().attr("data-type") == Some("footnote")
        || element.value().classes().any(|class| {
            matches!(
                class,
                "footnotes"
                    | "footnote"
                    | "footnotes-list"
                    | "references"
                    | "reflist"
                    | "footnote-definition"
                    | "footnote-definitions"
                    | "footdef"
                    | "footnotes-footer"
                    | "footnote-footer"
                    | "easy-footnotes-wrapper"
                    | "wp-block-footnotes"
                    | "footnotes-segment"
            )
        })
}

fn note_fragment(link: ElementRef<'_>, base: Option<&Url>) -> Option<String> {
    let href = link.value().attr("href")?;
    let (prefix, fragment) = href.rsplit_once('#')?;
    if fragment.is_empty() || safe_url(href, base).is_none() {
        return None;
    }
    // A remote page sharing an anchor name is not a local footnote. Explicit
    // HTMLBook markers and Word's generated export backlinks carry their own
    // structural evidence and are handled separately.
    if !prefix.is_empty()
        && link.value().attr("data-type") != Some("noteref")
        && !base.is_some_and(|base| {
            base.join(href).is_ok_and(|mut target| {
                let mut page = base.clone();
                page.set_fragment(None);
                target.set_fragment(None);
                target == page
            })
        })
    {
        return None;
    }
    Some(fragment.to_owned())
}

fn reference_candidate(element: ElementRef<'_>) -> bool {
    if element.value().name() != "a" {
        return true;
    }
    if element.value().attr("data-footnote-ref").is_some()
        || element.value().attr("data-type") == Some("noteref")
        || element
            .value()
            .attr("role")
            .is_some_and(|role| matches!(role, "doc-noteref" | "doc-biblioref"))
        || element
            .value()
            .classes()
            .any(|class| matches!(class, "footnote-ref" | "footnote-anchor" | "noteref"))
    {
        return true;
    }
    let text = plain(element);
    note_number(&text).is_some() || (!text.is_empty() && text.chars().all(|ch| "*†‡".contains(ch)))
}

fn literal_container(element: ElementRef<'_>) -> bool {
    matches!(element.value().name(), "pre" | "code" | "script" | "style")
        || code::is_block(element)
        || math_container(element)
        || duplicate_math_preview(element)
}

fn visible_reference(element: ElementRef<'_>, prune_chrome: bool) -> bool {
    std::iter::once(element)
        .chain(element.ancestors().filter_map(ElementRef::wrap))
        .all(|parent| {
            !is_hidden(parent)
                && !(prune_chrome && article::excluded(parent))
                && !matches!(
                    parent.value().name(),
                    "script"
                        | "style"
                        | "nav"
                        | "footer"
                        | "form"
                        | "button"
                        | "input"
                        | "select"
                        | "textarea"
                        | "iframe"
                        | "object"
                        | "embed"
                        | "head"
                        | "template"
                        | "noscript"
                )
        })
}

fn generic_note_section(element: ElementRef<'_>) -> bool {
    let label = |heading: ElementRef<'_>| {
        note_heading(heading)
            || (matches!(
                heading.value().name(),
                "h1" | "h2" | "h3" | "h4" | "h5" | "h6"
            ) && matches!(
                plain(heading).to_ascii_lowercase().as_str(),
                "references and notes" | "notes and references" | "bibliography"
            ))
    };
    note_context(element)
        || ["id", "class"].iter().any(|attr| {
            element.value().attr(attr).is_some_and(|value| {
                value
                    .split_ascii_whitespace()
                    .any(|token| matches!(token, "note" | "notes" | "endnotes" | "bibliography"))
            })
        })
        || element.child_elements().any(label)
        || element
            .prev_siblings()
            .filter_map(ElementRef::wrap)
            .next()
            .is_some_and(label)
}

fn generic_continuation(element: ElementRef<'_>) -> bool {
    if !matches!(
        element.value().name(),
        "p" | "ul" | "ol" | "blockquote" | "pre"
    ) || element.value().attr("id").is_some()
        || element.select(&selector("[id],a[name]")).next().is_some()
        || element
            .value()
            .classes()
            .any(|class| matches!(class, "seealso" | "see-also" | "related" | "notetitle"))
    {
        return false;
    }
    let text = plain(element).to_ascii_lowercase();
    if ["see also:", "related:", "update:"]
        .iter()
        .any(|label| text.starts_with(label))
    {
        return false;
    }
    !first_content(element).is_some_and(|first| {
        matches!(first.value().name(), "b" | "strong")
            && plain(first).to_ascii_lowercase().starts_with("update")
    })
}

fn numeric_text_marker(element: ElementRef<'_>, number: &str) -> Option<String> {
    for child in element.children() {
        match child.value() {
            scraper::Node::Comment(_) => continue,
            scraper::Node::Text(text) if text.trim().is_empty() => continue,
            scraper::Node::Text(text) => {
                let tail = text.trim_start().strip_prefix(number)?.strip_prefix('.')?;
                if !tail.is_empty() && !tail.chars().next().is_some_and(char::is_whitespace) {
                    return None;
                }
                let length = text.len() - tail.trim_start().len();
                return Some(text[..length].to_owned());
            }
            _ => return None,
        }
    }
    None
}

fn note_has_content(
    element: ElementRef<'_>,
    markers: &[ElementRef<'_>],
    mut text_marker: Option<&str>,
    depth: usize,
) -> bool {
    if depth > 256 {
        return true;
    } // The serializer reports the nesting error.
    if markers.contains(&element)
        || (depth > 0 && is_hidden(element))
        || element.value().attr("role") == Some("doc-backlink")
        || element.value().attr("data-footnote-backref").is_some()
        || element.value().classes().any(|class| {
            matches!(
                class,
                "footnote-backref"
                    | "data-footnote-backref"
                    | "mw-cite-backlink"
                    | "easy-footnote-to-top"
            )
        })
    {
        return false;
    }
    let name = element.value().name();
    if name == "script" && tex_script(element).is_some() {
        return !plain(element).is_empty();
    }
    if matches!(
        name,
        "script"
            | "style"
            | "nav"
            | "footer"
            | "form"
            | "button"
            | "input"
            | "select"
            | "textarea"
            | "iframe"
            | "object"
            | "embed"
            | "head"
            | "template"
            | "noscript"
    ) {
        return false;
    }
    if name == "img" {
        return element
            .value()
            .attr("data-src")
            .or_else(|| element.value().attr("src"))
            .is_some_and(|src| image_source(src, None).is_some());
    }
    if name == "a"
        && element
            .value()
            .attr("href")
            .is_some_and(|href| href.starts_with('#'))
        && !plain(element).is_empty()
        && plain(element)
            .chars()
            .all(|ch| "^↩↥↑↵⤴⤵⏎\u{fe0e}\u{fe0f}".contains(ch))
    {
        return false;
    }
    element.children().any(|child| {
        if let Some(child) = ElementRef::wrap(child) {
            note_has_content(child, markers, None, depth + 1)
        } else if let scraper::Node::Text(text) = child.value() {
            let text = if let Some(prefix) = text_marker
                && let Some(rest) = text.strip_prefix(prefix)
            {
                text_marker = None;
                rest
            } else {
                text.as_ref()
            };
            !text.trim().is_empty()
        } else {
            false
        }
    })
}

struct Footnote<'a> {
    nodes: Vec<ElementRef<'a>>,
    aliases: Vec<String>,
    markers: Vec<ElementRef<'a>>,
    references: Vec<ElementRef<'a>>,
    text_marker: Option<String>,
}

#[derive(Default)]
struct Footnotes<'a> {
    prune_chrome: bool,
    definitions: Vec<Footnote<'a>>,
    references: HashMap<usize, usize>,
    removed: HashSet<usize>,
    markers: HashSet<usize>,
    text_markers: HashMap<usize, String>,
}

impl<'a> Footnotes<'a> {
    fn add(
        &mut self,
        nodes: Vec<ElementRef<'a>>,
        aliases: Vec<String>,
        markers: Vec<ElementRef<'a>>,
        references: Vec<ElementRef<'a>>,
    ) {
        if nodes.is_empty()
            || !nodes
                .iter()
                .any(|node| note_has_content(*node, &markers, None, 0))
            || self.definitions.iter().any(|note| {
                note.nodes
                    .iter()
                    .any(|existing| nodes.iter().any(|node| contains_element(*existing, *node)))
            })
        {
            return;
        }
        self.definitions.push(Footnote {
            nodes,
            aliases,
            markers,
            references,
            text_marker: None,
        });
    }

    fn inside_definition(&self, element: ElementRef<'_>) -> bool {
        self.definitions.iter().any(|note| {
            note.nodes
                .iter()
                .any(|node| contains_element(*node, element))
        })
    }

    fn collect(
        root: ElementRef<'a>,
        document: ElementRef<'a>,
        base: Option<&Url>,
        prune_chrome: bool,
    ) -> Self {
        let mut notes = Self {
            prune_chrome,
            ..Self::default()
        };
        let references = root
            .select(&selector(
                "a[href], sup, span[data-definition], label.footref",
            ))
            .filter(|element| reference_candidate(*element))
            .collect::<Vec<_>>();
        let inline = root
            .select(&selector(
                "span.footnote-container, span.sidenote-container, span.inline-footnote",
            ))
            .collect::<Vec<_>>();
        // Definitions only move when a reference or inline popover resolves.
        // Ordinary documents need no document-wide ID, literal or note indexes.
        if references.is_empty() && inline.is_empty() {
            return notes;
        }
        let elements = document
            .descendants()
            .filter_map(ElementRef::wrap)
            .collect::<Vec<_>>();
        // DOM traversal is in parent-before-child order. Structural code
        // classification may inspect a subtree; never repeat it for every
        // descendant's ancestry during the separate footnote collection passes.
        let mut literal_elements = HashSet::new();
        for &element in &elements {
            if element
                .parent()
                .and_then(ElementRef::wrap)
                .is_some_and(|parent| literal_elements.contains(&element_key(parent)))
                || literal_container(element)
            {
                literal_elements.insert(element_key(element));
            }
        }
        let in_literal = |element| literal_elements.contains(&element_key(element));
        let mut ids = HashMap::new();
        for element in &elements {
            for attribute in ["id", "name"] {
                if let Some(id) = element.value().attr(attribute) {
                    ids.entry(id.to_owned()).or_insert(*element);
                }
            }
        }
        let references = references
            .into_iter()
            .filter(|element| !in_literal(*element) && visible_reference(*element, prune_chrome))
            .collect::<Vec<_>>();
        let external = elements
            .iter()
            .copied()
            .filter(|element| {
                !contains_element(root, *element)
                    && !contains_element(*element, root)
                    && matches!(
                        element.value().name(),
                        "div" | "section" | "aside" | "ol" | "ul" | "p" | "li"
                    )
                    && !in_literal(*element)
                    && !element
                        .ancestors()
                        .filter_map(ElementRef::wrap)
                        .any(|parent| {
                            matches!(
                                parent.value().name(),
                                "head" | "template" | "noscript" | "iframe" | "object" | "embed"
                            )
                        })
                    && (note_context(*element)
                        || (["id", "class"].iter().any(|attr| {
                            element.value().attr(attr).is_some_and(|value| {
                                value.to_ascii_lowercase().contains("footnote")
                            })
                        }) && element
                            .select(&selector("h1,h2,h3,h4,h5,h6"))
                            .any(note_heading)))
            })
            .collect::<Vec<_>>();
        let in_scope = |element| {
            contains_element(root, element)
                || external
                    .iter()
                    .any(|container| contains_element(*container, element))
        };

        // Inline popovers have a definition and a reference at the same DOM
        // position. Only the identified content root bypasses hidden styling.
        for container in inline {
            if in_literal(container) || !visible_reference(container, prune_chrome) {
                continue;
            }
            if let Some(content) = container
                .select(&selector(
                    "span.footnote,span.sidenote,span.footnoteContent",
                ))
                .next()
            {
                if content
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|ancestor| *ancestor != container)
                    .any(is_hidden)
                {
                    continue;
                }
                notes.add(vec![content], vec![], vec![], vec![container]);
            }
        }
        for reference in &references {
            if let Some(id) = reference.value().attr("data-definition")
                && let Some(target) = ids.get(id).copied()
                && target.value().name() == "aside"
                && in_scope(target)
            {
                notes.add(vec![target], vec![id.to_owned()], vec![], vec![*reference]);
            }
            if reference.value().name() == "label" && has_class(*reference, "footref") {
                let content = reference
                    .next_siblings()
                    .filter_map(ElementRef::wrap)
                    .find(|node| node.value().name() != "input");
                if let Some(content) = content.filter(|node| has_class(*node, "sidenote")) {
                    let markers = leading_note_marker(content)
                        .map(|(_, marker)| marker)
                        .into_iter()
                        .collect();
                    notes.add(vec![content], vec![], markers, vec![*reference]);
                }
            }
        }

        // Known definition structures are accepted even when there is one note.
        // Their bodies are retained unless a real reference can be resolved.
        for element in elements
            .iter()
            .copied()
            .filter(|element| in_scope(*element) && !in_literal(*element))
        {
            let name = element.value().name();
            let parent = element.parent().and_then(ElementRef::wrap);
            if name == "ol"
                && element.value().attr("start").is_some()
                && parent.is_some_and(|parent| parent.value().name() == "aside")
                && let Some(number) = element.value().attr("start").and_then(note_number)
            {
                let items = element
                    .child_elements()
                    .filter(|child| child.value().name() == "li")
                    .collect();
                notes.add(items, vec![format!("num:{number}")], vec![], vec![]);
                continue;
            }
            let context = note_context(element)
                || element
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|parent| in_scope(*parent))
                    .any(note_context);
            let id = element.value().attr("id").unwrap_or("");
            let known_id = ["fn:", "fn-", "fn.", "ftnt", "cite_note-", "footnote-"]
                .iter()
                .any(|prefix| id.starts_with(prefix))
                && !id.contains("ref")
                && !id.ends_with("-link");
            let explicit = matches!(name, "p" | "li" | "div" | "aside")
                && (context || known_id)
                && (name == "li"
                    || (name == "p" && (context || known_id))
                    || has_class(element, "footnote-definition")
                    || has_class(element, "footnote-footer")
                    || element.value().attr("role") == Some("doc-footnote")
                    || (known_id && !plain(element).is_empty()));
            if !explicit || notes.inside_definition(element) {
                continue;
            }
            let mut aliases = Vec::new();
            let mut markers = Vec::new();
            if !id.is_empty() {
                aliases.push(id.to_owned());
            }
            if let Some((number, marker)) = leading_note_marker(element) {
                aliases.push(format!("num:{number}"));
                markers.push(marker);
            }
            for anchor in element.select(&selector("[id],a[name]")) {
                if matches!(anchor.value().name(), "a" | "span" | "sup")
                    && (plain(anchor).is_empty() || note_number(&plain(anchor)).is_some())
                    && let Some(id) = anchor
                        .value()
                        .attr("id")
                        .or_else(|| anchor.value().attr("name"))
                {
                    aliases.push(id.to_owned());
                    markers.push(anchor);
                }
            }
            if name == "li"
                && let Some(list) = parent
            {
                let index = list
                    .child_elements()
                    .filter(|child| child.value().name() == "li")
                    .position(|child| child == element)
                    .unwrap_or(0);
                let start = list
                    .value()
                    .attr("start")
                    .and_then(|value| value.parse::<usize>().ok())
                    .unwrap_or(1);
                if let Some(number) = start.checked_add(index) {
                    aliases.push(format!("num:{number}"));
                }
            }
            if !aliases.is_empty() {
                notes.add(vec![element], aliases, markers, vec![]);
            }
        }

        // Word exports omit target IDs but identify each definition with a
        // generated _ftnref backlink. Keep surrounding prose and external links.
        for paragraph in elements
            .iter()
            .copied()
            .filter(|element| in_scope(*element) && element.value().name() == "p")
        {
            if notes.inside_definition(paragraph) {
                continue;
            }
            if let Some(link) = paragraph.select(&selector("a[href]")).next()
                && let Some(number) = link
                    .value()
                    .attr("href")
                    .and_then(|href| href.rsplit_once('#'))
                    .and_then(|(_, fragment)| fragment.strip_prefix("_ftnref"))
                    .and_then(note_number)
            {
                let marker = link
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|parent| *parent != paragraph)
                    .find(|parent| parent.value().name() == "sup")
                    .unwrap_or(link);
                notes.add(
                    vec![paragraph],
                    vec![format!("_ftn{number}")],
                    vec![marker],
                    vec![],
                );
            }
        }

        // Generic numeric links need a concentrated definition section, not
        // merely numeric equation/theorem targets scattered through an article.
        let mut generic = Vec::new();
        let mut numeric_references: HashMap<String, Vec<ElementRef<'_>>> = HashMap::new();
        for reference in &references {
            if reference.value().name() != "a" || note_number(&plain(*reference)).is_none() {
                continue;
            }
            let Some(fragment) = note_fragment(*reference, base) else {
                continue;
            };
            numeric_references
                .entry(fragment.clone())
                .or_default()
                .push(*reference);
            let Some(target) = ids
                .get(&fragment)
                .copied()
                .filter(|target| in_scope(*target))
            else {
                continue;
            };
            if notes.inside_definition(target) {
                continue;
            }
            let node = if matches!(target.value().name(), "p" | "li") {
                Some(target)
            } else {
                target
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|parent| *parent != root)
                    .find(|parent| {
                        matches!(parent.value().name(), "p" | "li")
                            || has_class(*parent, "reference")
                    })
            };
            if let Some(node) = node
                && !node
                    .select(&selector("a[href]"))
                    .any(|link| link == *reference)
                && !generic.iter().any(|(id, _, _)| *id == fragment)
            {
                generic.push((fragment, node, target));
            }
        }
        for container in elements.iter().copied().filter(|element| {
            in_scope(*element)
                && *element != root
                && matches!(
                    element.value().name(),
                    "ol" | "ul" | "div" | "section" | "aside"
                )
        }) {
            let matches = generic
                .iter()
                .filter(|(_, node, _)| contains_element(container, *node))
                .collect::<Vec<_>>();
            if matches.len() < 2 || !generic_note_section(container) {
                continue;
            }
            let external_count = numeric_references
                .values()
                .filter(|references| {
                    references
                        .iter()
                        .any(|reference| !contains_element(container, *reference))
                })
                .count();
            if matches.len() * 4 < external_count * 3 {
                continue;
            }
            // A common wrapper around all article content is not a notes list.
            if references.iter().any(|reference| {
                contains_element(container, *reference)
                    && !matches
                        .iter()
                        .any(|(_, node, _)| contains_element(*node, *reference))
            }) {
                continue;
            }
            for (id, node, target) in matches {
                let mut markers = Vec::new();
                if *node != *target
                    && (plain(*target).is_empty() || note_number(&plain(*target)).is_some())
                {
                    markers.push(*target);
                }
                let mut nodes = vec![*node];
                for sibling in node.next_siblings().filter_map(ElementRef::wrap) {
                    if generic.iter().any(|(_, target, _)| *target == sibling)
                        || !generic_continuation(sibling)
                    {
                        break;
                    }
                    if !is_hidden(sibling) {
                        nodes.push(sibling);
                    }
                }
                let before = notes.definitions.len();
                notes.add(nodes, vec![id.clone()], markers, vec![]);
                if notes.definitions.len() > before
                    && let Some(number) = numeric_references
                        .get(id)
                        .and_then(|links| links.first())
                        .and_then(|link| note_number(&plain(*link)))
                {
                    notes.definitions.last_mut().unwrap().text_marker =
                        numeric_text_marker(*node, &number);
                }
            }
        }

        // Loose numbered paragraphs require corroborating body references.
        // A section delimiter admits continuation blocks; otherwise only the
        // numbered paragraphs move, leaving updates and unrelated prose intact.
        let paragraphs = elements
            .iter()
            .copied()
            .filter(|element| {
                in_scope(*element)
                    && element.value().name() == "p"
                    && !notes.inside_definition(*element)
            })
            .filter_map(|element| {
                leading_note_marker(element).map(|(number, marker)| (element, number, marker))
            })
            .collect::<Vec<_>>();
        let matched = paragraphs
            .iter()
            .filter(|(_, number, _)| {
                references.iter().any(|reference| {
                    reference.value().name() == "sup"
                        && plain(*reference) == *number
                        && !paragraphs
                            .iter()
                            .any(|(paragraph, _, _)| contains_element(*paragraph, *reference))
                        && !notes.inside_definition(*reference)
                })
            })
            .count();
        if matched >= 2 {
            for (paragraph, number, marker) in &paragraphs {
                let siblings = paragraph
                    .parent()
                    .map(|parent| {
                        parent
                            .children()
                            .filter_map(ElementRef::wrap)
                            .collect::<Vec<_>>()
                    })
                    .unwrap_or_default();
                let position = siblings
                    .iter()
                    .position(|element| element == paragraph)
                    .unwrap_or(0);
                let boundary = siblings[..position]
                    .iter()
                    .rposition(|element| element.value().name() == "hr" || note_heading(*element));
                let mut nodes = vec![*paragraph];
                if boundary.is_some() {
                    for sibling in &siblings[position + 1..] {
                        if paragraphs.iter().any(|(node, _, _)| node == sibling)
                            || !matches!(sibling.value().name(), "p" | "ul" | "ol" | "blockquote")
                            || first_content(*sibling).is_some_and(|first| {
                                matches!(first.value().name(), "b" | "strong")
                                    && leading_note_marker(*sibling).is_none()
                            })
                        {
                            break;
                        }
                        if !is_hidden(*sibling) {
                            nodes.push(*sibling);
                        }
                    }
                }
                notes.add(nodes, vec![format!("num:{number}")], vec![*marker], vec![]);
            }
        }

        let order = elements
            .iter()
            .enumerate()
            .map(|(index, element)| (element_key(*element), index))
            .collect::<HashMap<_, _>>();
        notes.definitions.sort_by_key(|note| {
            order
                .get(&element_key(note.nodes[0]))
                .copied()
                .unwrap_or(usize::MAX)
        });
        let mut aliases = HashMap::new();
        for (index, note) in notes.definitions.iter().enumerate() {
            for alias in &note.aliases {
                aliases.entry(alias.clone()).or_insert(index);
            }
        }
        for reference in references {
            if notes.inside_definition(reference) || in_literal(reference) {
                continue;
            }
            let alias = if reference.value().name() == "a" {
                note_fragment(reference, base)
            } else if reference.value().name() == "sup"
                && reference.select(&selector("a")).next().is_none()
            {
                note_number(&plain(reference)).map(|number| format!("num:{number}"))
            } else {
                None
            };
            if let Some(index) = alias.as_ref().and_then(|alias| aliases.get(alias)).copied() {
                let mut replacement = reference;
                for parent in reference.ancestors().filter_map(ElementRef::wrap) {
                    if !matches!(parent.value().name(), "sup" | "span")
                        || plain(parent) != plain(reference)
                    {
                        break;
                    }
                    replacement = parent;
                }
                notes.definitions[index].references.push(replacement);
            }
        }
        notes.definitions.retain(|note| !note.references.is_empty());
        let reference_ids = notes
            .definitions
            .iter()
            .flat_map(|note| note.references.iter())
            .flat_map(|reference| {
                std::iter::once(*reference)
                    .chain(reference.descendants().filter_map(ElementRef::wrap))
            })
            .filter_map(|element| element.value().attr("id").map(str::to_owned))
            .collect::<HashSet<_>>();
        for note in &mut notes.definitions {
            for root in &note.nodes {
                for element in root.descendants().filter_map(ElementRef::wrap) {
                    let explicit = element.value().attr("role") == Some("doc-backlink")
                        || element.value().attr("data-footnote-backref").is_some()
                        || element.value().classes().any(|class| {
                            matches!(
                                class,
                                "footnote-backref"
                                    | "data-footnote-backref"
                                    | "mw-cite-backlink"
                                    | "easy-footnote-to-top"
                                    | "footnote-definition-label"
                            )
                        });
                    let link_back = element.value().name() == "a"
                        && element
                            .value()
                            .attr("href")
                            .and_then(|href| href.strip_prefix('#'))
                            .is_some_and(|fragment| {
                                reference_ids.contains(fragment)
                                    || (!plain(element).is_empty()
                                        && plain(element)
                                            .chars()
                                            .all(|ch| "^↩↥↑↵⤴⤵⏎\u{fe0e}\u{fe0f}".contains(ch)))
                            });
                    if explicit || link_back {
                        note.markers.push(element);
                    }
                }
            }
        }
        notes.definitions.retain(|note| {
            note.nodes.iter().enumerate().any(|(index, node)| {
                note_has_content(
                    *node,
                    &note.markers,
                    if index == 0 {
                        note.text_marker.as_deref()
                    } else {
                        None
                    },
                    0,
                )
            })
        });
        for (index, note) in notes.definitions.iter().enumerate() {
            if let Some(prefix) = &note.text_marker {
                notes
                    .text_markers
                    .insert(element_key(note.nodes[0]), prefix.clone());
            }
            for reference in &note.references {
                notes.references.insert(element_key(*reference), index + 1);
            }
            for node in &note.nodes {
                notes.removed.insert(element_key(*node));
            }
            for marker in &note.markers {
                notes.markers.insert(element_key(*marker));
            }
        }
        // Delete separators only when the following structure consists solely
        // of definitions. Notes headings that introduce additional prose remain.
        for element in &elements {
            if !(note_heading(*element) || element.value().name() == "hr") {
                continue;
            }
            let mut carrier = *element;
            while let Some(parent) = carrier.parent().and_then(ElementRef::wrap)
                && parent != root
                && parent.child_elements().count() == 1
                && plain(parent) == plain(carrier)
            {
                carrier = parent;
            }
            if let Some(next) = carrier.next_siblings().filter_map(ElementRef::wrap).next()
                && notes.definition_shell(next, 0)
            {
                notes.removed.insert(element_key(*element));
            }
        }
        // A duplicated sidenote is dropped only when its text matches the
        // adjacent resolved definition; unrelated marginal text stays visible.
        for sidenote in root.select(&selector("span.sidenote")) {
            if notes.inside_definition(sidenote) {
                continue;
            }
            if let Some(previous) = sidenote.prev_siblings().filter_map(ElementRef::wrap).next()
                && let Some(number) = notes.references.get(&element_key(previous))
            {
                let content = plain(sidenote);
                let content = content.trim_start_matches(|ch: char| {
                    ch.is_ascii_digit() || ch == '.' || ch.is_whitespace()
                });
                let definition = notes.definitions[*number - 1]
                    .nodes
                    .iter()
                    .map(|node| plain(*node))
                    .collect::<Vec<_>>()
                    .join(" ");
                let definition = definition.trim_start_matches(|ch: char| {
                    ch.is_ascii_digit() || ch == '.' || ch.is_whitespace()
                });
                if content == definition {
                    notes.removed.insert(element_key(sidenote));
                }
            }
        }
        notes
    }

    fn definition_shell(&self, element: ElementRef<'_>, depth: usize) -> bool {
        if depth > 256 {
            return false;
        }
        if self.removed.contains(&element_key(element)) {
            return true;
        }
        let mut has_definition = false;
        for node in element.children() {
            if let Some(child) = ElementRef::wrap(node) {
                if note_heading(child) || child.value().name() == "hr" {
                    continue;
                }
                if !self.definition_shell(child, depth + 1) {
                    return false;
                }
                has_definition = true;
            } else if let scraper::Node::Text(text) = node.value()
                && !text.trim().is_empty()
            {
                return false;
            }
        }
        has_definition
    }
}

fn serialize_clean(
    element: ElementRef<'_>,
    base: Option<&Url>,
    output: &mut String,
    depth: usize,
    notes: &Footnotes<'_>,
    definition: bool,
) -> Result<()> {
    if (is_hidden(element) || (notes.prune_chrome && article::excluded(element)))
        && !(definition && depth == 0)
    {
        return Ok(());
    }
    let key = element_key(element);
    if !definition && let Some(number) = notes.references.get(&key) {
        let visible_text = element
            .descendants()
            .filter_map(|node| {
                let scraper::Node::Text(text) = node.value() else {
                    return None;
                };
                (!node
                    .ancestors()
                    .filter_map(ElementRef::wrap)
                    .take_while(|parent| *parent != element)
                    .any(|parent| {
                        notes.removed.contains(&element_key(parent))
                            || !visible_reference(parent, notes.prune_chrome)
                    }))
                .then_some(&**text)
            })
            .collect::<String>();
        let start = visible_text.len() - visible_text.trim_start().len();
        let end = visible_text.trim_end().len();
        escaped_text(&visible_text[..start], output);
        output.push_str(&format!(
            "<markitai-footnote data-number=\"{number}\"></markitai-footnote>"
        ));
        escaped_text(&visible_text[end..], output);
        return Ok(());
    }
    if (notes.removed.contains(&key) && !(definition && depth == 0))
        || (definition && notes.markers.contains(&key))
    {
        return Ok(());
    }
    if depth > 256 {
        return Err(Error::Conversion(
            "HTML nesting exceeds 256 elements".into(),
        ));
    }
    if code::render(element, output, depth)? {
        return Ok(());
    }
    let value = element.value();
    let name = value.name();
    if duplicate_math_preview(element) {
        return Ok(());
    }
    if !element
        .ancestors()
        .filter_map(ElementRef::wrap)
        .any(|ancestor| matches!(ancestor.value().name(), "pre" | "code"))
        && let Some((latex, block)) = math_expression(element)?
    {
        emit_math(&latex, block, output);
        return Ok(());
    }
    if matches!(
        name,
        "script"
            | "style"
            | "nav"
            | "footer"
            | "form"
            | "button"
            | "input"
            | "select"
            | "textarea"
            | "iframe"
            | "object"
            | "embed"
            | "head"
            | "template"
            | "noscript"
    ) {
        return Ok(());
    }
    // Canonicalize lazy images without retaining arbitrary event or style attributes.
    let src = value
        .attr("data-src")
        .or_else(|| value.attr("data-original"))
        .or_else(|| value.attr("src"));
    let preformatted = matches!(name, "pre" | "code")
        || element
            .ancestors()
            .filter_map(ElementRef::wrap)
            .any(|parent| matches!(parent.value().name(), "pre" | "code"));
    let styled_code = name == "code"
        && value.attr("style").is_some_and(|style| {
            style
                .to_ascii_lowercase()
                .replace(char::is_whitespace, "")
                .contains("white-space:pre")
        })
        && !element
            .ancestors()
            .filter_map(ElementRef::wrap)
            .any(|parent| parent.value().name() == "pre");
    if styled_code {
        output.push_str("<pre>");
    }
    let serialized_name = if definition && depth == 0 && name == "li" {
        "div"
    } else {
        name
    };
    output.push('<');
    output.push_str(serialized_name);
    for attribute in [
        "href", "src", "alt", "title", "colspan", "rowspan", "start", "class", "id",
    ] {
        let raw = if attribute == "src" && name == "img" {
            src
        } else {
            value.attr(attribute)
        };
        if let Some(raw) = raw {
            let normalized = if attribute == "src" && name == "img" {
                image_source(raw, base)
            } else if matches!(attribute, "href" | "src") {
                safe_url(raw, base)
            } else {
                Some(raw.to_owned())
            };
            if let Some(normalized) = normalized {
                output.push(' ');
                output.push_str(attribute);
                output.push_str("=\"");
                escaped(&normalized, output);
                output.push('"');
            }
        }
    }
    output.push('>');
    let mut text_marker = if definition && depth == 0 {
        notes.text_markers.get(&key).map(String::as_str)
    } else {
        None
    };
    for child in element.children() {
        if let Some(child) = ElementRef::wrap(child) {
            serialize_clean(child, base, output, depth + 1, notes, definition)?;
        } else if let scraper::Node::Text(text) = child.value() {
            let text = if let Some(prefix) = text_marker
                && let Some(rest) = text.strip_prefix(prefix)
            {
                text_marker = None;
                rest
            } else {
                text.as_ref()
            };
            if preformatted {
                escaped(text, output);
            } else {
                let previous = child
                    .prev_siblings()
                    .find(|node| !matches!(node.value(), scraper::Node::Comment(_)));
                let next = child
                    .next_siblings()
                    .find(|node| !matches!(node.value(), scraper::Node::Comment(_)));
                let trim_start = previous.map_or(block_tag(name), |node| {
                    ElementRef::wrap(node).is_some_and(|element| block_tag(element.value().name()))
                });
                let trim_end = next.map_or(block_tag(name), |node| {
                    ElementRef::wrap(node).is_some_and(|element| block_tag(element.value().name()))
                });
                let text = if trim_start { text.trim_start() } else { text };
                let text = if trim_end { text.trim_end() } else { text };
                escaped_text(text, output);
            }
        }
    }
    if styled_code {
        output.push_str("</pre>");
    }
    if !matches!(
        name,
        "area" | "base" | "br" | "col" | "hr" | "img" | "link" | "meta" | "source" | "wbr"
    ) {
        output.push_str("</");
        output.push_str(serialized_name);
        output.push('>');
    }
    Ok(())
}

fn render_clean(root: ElementRef<'_>, base: Option<&Url>) -> Result<String> {
    render_with_footnotes(root, root, base, false)
}

fn render_with_footnotes<'a>(
    root: ElementRef<'a>,
    document: ElementRef<'a>,
    base: Option<&Url>,
    prune_chrome: bool,
) -> Result<String> {
    let notes = Footnotes::collect(root, document, base, prune_chrome);
    let mut cleaned = String::new();
    serialize_clean(root, base, &mut cleaned, 0, &notes, false)?;
    let mut markdown = render_sanitized(&cleaned)?;
    for (index, note) in notes.definitions.iter().enumerate() {
        let mut cleaned = String::new();
        for node in &note.nodes {
            serialize_clean(*node, base, &mut cleaned, 0, &notes, true)?;
        }
        let content = render_sanitized(&cleaned)?;
        if content.is_empty() {
            continue;
        }
        markdown.push_str(&format!("\n\n[^{}]: ", index + 1));
        let mut lines = content.lines();
        markdown.push_str(lines.next().unwrap_or_default());
        // Continuations remain inside the Markdown definition, including nested
        // lists, quotes and code fences; no source prose is flattened away.
        for line in lines {
            markdown.push('\n');
            if !line.is_empty() {
                markdown.push_str("    ");
                markdown.push_str(line);
            }
        }
    }
    Ok(markdown.trim().to_owned())
}

fn render_sanitized(cleaned: &str) -> Result<String> {
    htmd::HtmlToMarkdown::builder()
        .skip_tags(vec!["script", "style", "head"])
        .add_handler(
            vec!["markitai-footnote"],
            |_: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                let number = element
                    .attrs
                    .iter()
                    .find(|attr| attr.name.local.as_ref() == "data-number")?
                    .value
                    .parse::<usize>()
                    .ok()?;
                Some(format!("[^{number}]").into())
            },
        )
        .add_handler(
            vec!["markitai-math"],
            |_: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                let latex = element
                    .attrs
                    .iter()
                    .find(|attr| attr.name.local.as_ref() == "data-latex")?
                    .value
                    .as_ref();
                // Emit TeX syntax as data, never HTML supplied in an annotation.
                let latex = latex.replace('<', r"\lt ").replace('>', r"\gt ");
                let block = element.attrs.iter().any(|attr| {
                    attr.name.local.as_ref() == "data-display" && attr.value.as_ref() == "block"
                });
                Some(
                    if block {
                        format!("\n\n$${latex}$$\n\n")
                    } else {
                        format!("${latex}$")
                    }
                    .into(),
                )
            },
        )
        .add_handler(
            vec!["markitai-soft-break"],
            |_: &dyn htmd::element_handler::Handlers, _: htmd::Element| Some("\n".into()),
        )
        .add_handler(
            vec!["q"],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                Some(format!("\"{}\"", handlers.walk_children(element.node).content).into())
            },
        )
        .add_handler(
            vec!["sub", "sup"],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                Some(
                    format!(
                        "<{0}>{1}</{0}>",
                        element.tag,
                        handlers.walk_children(element.node).content.trim()
                    )
                    .into(),
                )
            },
        )
        .add_handler(
            vec![
                "div",
                "section",
                "article",
                "main",
                "figure",
                "figcaption",
                "dl",
                "dt",
                "dd",
                "details",
                "summary",
            ],
            |handlers: &dyn htmd::element_handler::Handlers, element: htmd::Element| {
                Some(
                    format!(
                        "\n\n{}\n\n",
                        handlers.walk_children(element.node).content.trim()
                    )
                    .into(),
                )
            },
        )
        .build()
        .convert(cleaned)
        .map(|markdown| markdown.trim().to_owned())
        .map_err(|error| Error::Conversion(format!("HTML rendering failed: {error}")))
}

#[derive(Debug)]
enum BbNode {
    Text(String),
    Tag {
        name: String,
        argument: Option<String>,
        children: Vec<BbNode>,
    },
}

struct BbFrame {
    name: String,
    argument: Option<String>,
    children: Vec<BbNode>,
}

fn bbcode_tree(source: &str) -> Result<Vec<BbNode>> {
    fn close(stack: &mut Vec<BbFrame>) {
        let frame = stack.pop().expect("a non-root BBCode frame");
        stack
            .last_mut()
            .expect("BBCode root frame")
            .children
            .push(BbNode::Tag {
                name: frame.name,
                argument: frame.argument,
                children: frame.children,
            });
    }
    let mut stack = vec![BbFrame {
        name: String::new(),
        argument: None,
        children: Vec::new(),
    }];
    let mut offset = 0;
    while offset < source.len() {
        let Some(relative_start) = source[offset..].find('[') else {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[offset..].to_owned()));
            break;
        };
        let start = offset + relative_start;
        if start > offset {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[offset..start].to_owned()));
        }
        let mut quote = None;
        let mut end = None;
        // Backslash-quoted delimiters occur in older event-store payloads even
        // after JSON decoding; the token normalizer below handles that form.
        for (index, character) in source[start + 1..].char_indices() {
            if character == '\'' || character == '"' {
                if quote == Some(character) {
                    quote = None;
                } else if quote.is_none() {
                    quote = Some(character);
                }
            } else if character == ']' && quote.is_none() {
                end = Some(start + 1 + index);
                break;
            }
        }
        let Some(end) = end else {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[start..].to_owned()));
            break;
        };
        offset = end + 1;
        let token = source[start + 1..end]
            .replace(r"\/", "/")
            .replace("\\\"", "\"");
        let token = token.trim();
        let closing = token.starts_with('/');
        let token = token.strip_prefix('/').unwrap_or(token);
        let (name, argument) = token
            .split_once('=')
            .map_or((token, None), |(name, value)| {
                (
                    name,
                    Some(value.trim().trim_matches(['\'', '"']).to_owned()),
                )
            });
        let name = name.trim().to_ascii_lowercase();
        let name = if name == "*" { "li" } else { &name };
        if !matches!(
            name,
            "p" | "b"
                | "i"
                | "u"
                | "s"
                | "url"
                | "img"
                | "h1"
                | "h2"
                | "h3"
                | "h4"
                | "h5"
                | "h6"
                | "list"
                | "li"
                | "code"
                | "quote"
                | "previewyoutube"
        ) {
            stack
                .last_mut()
                .unwrap()
                .children
                .push(BbNode::Text(source[start..offset].to_owned()));
            continue;
        }
        if closing {
            if let Some(position) = stack.iter().rposition(|frame| frame.name == name) {
                while stack.len() > position {
                    close(&mut stack);
                }
            } else {
                stack
                    .last_mut()
                    .unwrap()
                    .children
                    .push(BbNode::Text(source[start..offset].to_owned()));
            }
            continue;
        }
        if name == "code" {
            let tail = &source[offset..];
            let lower = tail.to_ascii_lowercase();
            let close = lower
                .find("[/code]")
                .map(|index| (index, 7))
                .or_else(|| lower.find(r"[\/code]").map(|index| (index, 8)));
            let (length, closing_length) = close.unwrap_or((tail.len(), 0));
            stack.last_mut().unwrap().children.push(BbNode::Tag {
                name: name.into(),
                argument,
                children: vec![BbNode::Text(tail[..length].to_owned())],
            });
            offset += length + closing_length;
            continue;
        }
        if name == "li"
            && let Some(list) = stack.iter().rposition(|frame| frame.name == "list")
            && let Some(item) = stack
                .iter()
                .enumerate()
                .skip(list + 1)
                .find_map(|(index, frame)| (frame.name == "li").then_some(index))
        {
            while stack.len() > item {
                close(&mut stack);
            }
        }
        if stack.len() > 64 {
            return Err(Error::Conversion("BBCode nesting exceeds 64 tags".into()));
        }
        stack.push(BbFrame {
            name: name.into(),
            argument,
            children: Vec::new(),
        });
    }
    while stack.len() > 1 {
        close(&mut stack);
    }
    Ok(stack.pop().unwrap().children)
}

fn bbcode_text(nodes: &[BbNode]) -> String {
    nodes
        .iter()
        .map(|node| match node {
            BbNode::Text(text) => text.clone(),
            BbNode::Tag { children, .. } => bbcode_text(children),
        })
        .collect()
}

fn escaped_bbcode_text(text: &str, output: &mut String) {
    // Keep angle brackets encoded after the HTML renderer's entity decoding,
    // so all displayed BBCode text stays literal in Markdown. Code nodes use
    // a separate fenced path; destinations use URL validation instead.
    escaped(&text.replace('<', "&lt;").replace('>', "&gt;"), output);
}

fn bbcode_html(nodes: &[BbNode], base: Option<&Url>, output: &mut String) {
    for node in nodes {
        match node {
            BbNode::Text(text) => {
                for (index, line) in text.split('\n').enumerate() {
                    if index > 0 {
                        output.push_str("<br>");
                    }
                    escaped_bbcode_text(line, output);
                }
            }
            BbNode::Tag {
                name,
                argument,
                children,
            } => {
                let argument = argument.as_deref().unwrap_or("");
                if name == "code" {
                    output.push_str("<pre><code");
                    if !argument.is_empty()
                        && argument
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
                    {
                        output.push_str(" class=\"language-");
                        escaped(argument, output);
                        output.push('"');
                    }
                    output.push('>');
                    escaped(&bbcode_text(children), output);
                    output.push_str("</code></pre>");
                    continue;
                }
                if matches!(name.as_str(), "url" | "img") {
                    let text = bbcode_text(children);
                    let destination = if argument.is_empty() {
                        text.as_str()
                    } else {
                        argument
                    };
                    let destination = destination.replace(r"\/", "/");
                    if let Some(url) = safe_url(&destination, base) {
                        output.push_str(if name == "img" {
                            "<img src=\""
                        } else {
                            "<a href=\""
                        });
                        escaped(&url, output);
                        output.push_str("\">");
                        if name == "url" {
                            bbcode_html(children, base, output);
                            output.push_str("</a>");
                        }
                    } else {
                        bbcode_html(children, base, output);
                    }
                    continue;
                }
                if name == "previewyoutube" {
                    let id = argument.split(';').next().unwrap_or("").trim();
                    if id.len() == 11
                        && id
                            .chars()
                            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_'))
                    {
                        output.push_str("<img src=\"https://www.youtube.com/watch?v=");
                        escaped(id, output);
                        output.push_str("\">");
                    } else {
                        bbcode_html(children, base, output);
                    }
                    continue;
                }
                let tag = match name.as_str() {
                    "b" => "strong",
                    "i" => "em",
                    "s" => "del",
                    "list" if !argument.is_empty() => "ol",
                    "list" => "ul",
                    "quote" => "blockquote",
                    name => name,
                };
                output.push('<');
                output.push_str(tag);
                output.push('>');
                if name == "quote" && !argument.is_empty() {
                    output.push_str("<p><strong>");
                    escaped_bbcode_text(argument, output);
                    output.push_str("</strong></p>");
                }
                bbcode_html(children, base, output);
                output.push_str("</");
                output.push_str(tag);
                output.push('>');
            }
        }
    }
}

struct Announcement {
    markdown: String,
    metadata: Map<String, Value>,
}

fn structured_announcement(document: &Html, base: Option<&Url>) -> Result<Option<Announcement>> {
    for carrier in document.select(&selector("[data-partnereventstore]")) {
        let raw = carrier.value().attr("data-partnereventstore").unwrap_or("");
        let Ok(data) = serde_json::from_str::<Value>(raw) else {
            continue;
        };
        let entries = match &data {
            Value::Array(entries) => entries.iter().collect::<Vec<_>>(),
            Value::Object(_) => vec![&data],
            _ => continue,
        };
        for entry in entries {
            let Some(body) = entry.get("announcement_body") else {
                continue;
            };
            let Some(text) = body
                .get("body")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
            else {
                continue;
            };
            let nodes = bbcode_tree(text)?;
            let mut html = String::new();
            bbcode_html(&nodes, base, &mut html);
            let tree = Html::parse_fragment(&html);
            let markdown = render_clean(tree.root_element(), base)?;
            if markdown.is_empty() {
                continue;
            }
            let mut metadata = Map::new();
            if let Some(title) = body
                .get("headline")
                .and_then(Value::as_str)
                .and_then(|value| clean_title(value, None))
            {
                metadata.insert("title".into(), title.into());
            }
            if let Some(time) = body
                .get("posttime")
                .and_then(Value::as_i64)
                .and_then(|seconds| chrono::DateTime::from_timestamp(seconds, 0))
            {
                metadata.insert("published".into(), time.to_rfc3339().into());
            }
            if let Some(group) = carrier
                .value()
                .attr("data-groupvanityinfo")
                .and_then(|raw| serde_json::from_str::<Value>(raw).ok())
                .and_then(|value| {
                    value
                        .as_array()
                        .and_then(|array| array.first())
                        .and_then(|group| group.get("group_name"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .filter(|group| !group.trim().is_empty())
            {
                metadata.insert("author".into(), group.trim().into());
            }
            return Ok(Some(Announcement { markdown, metadata }));
        }
    }
    Ok(None)
}

pub(super) fn fragment(source: &str) -> Result<String> {
    let document = Html::parse_fragment(source);
    render_clean(document.root_element(), None)
}

/// Extract an article candidate, metadata and Markdown without fetching links.
pub fn extract_html(source: &str, base_url: Option<&str>) -> Result<Document> {
    let mut document = Html::parse_document(source);
    stream::restore(&mut document)?;
    let base = base_url.and_then(|value| Url::parse(value).ok());
    let root = article::select(&document);
    let mut metadata = Map::new();
    let jsonld = jsonld_documents(&document);
    let site = meta(&document, &["og:site_name", "application-name"]);
    let title = page_title(&document, &jsonld, site.as_deref());
    if let Some(title) = title.filter(|value| !value.is_empty()) {
        metadata.insert("title".into(), title.into());
    }
    let author = meta(&document, &["author", "article:author"]).or_else(|| {
        jsonld.iter().find_map(|doc| {
            let value = doc.get("author")?;
            value
                .as_str()
                .or_else(|| value.get("name")?.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(str::to_owned)
        })
    });
    let published = meta(&document, &["article:published_time"])
        .or_else(|| jsonld_text(&jsonld, "datePublished"))
        .or_else(|| {
            document
                .select(&selector("time[datetime]"))
                .find_map(|time| {
                    let value = time.value().attr("datetime")?.trim();
                    let bytes = value.as_bytes();
                    (bytes.len() >= 7
                        && bytes[..4].iter().all(u8::is_ascii_digit)
                        && bytes[4] == b'-'
                        && bytes[5..7].iter().all(u8::is_ascii_digit))
                    .then(|| value.to_owned())
                })
        });
    for (key, value) in [
        ("author", author),
        ("site", site),
        ("published", published),
        (
            "description",
            meta(
                &document,
                &["description", "og:description", "twitter:description"],
            ),
        ),
    ] {
        if let Some(value) = value {
            metadata.insert(key.into(), value.into());
        }
    }
    if let Some(canonical) = document
        .select(&selector("link[rel]"))
        .find(|node| {
            node.value().attr("rel").is_some_and(|rel| {
                rel.split_whitespace()
                    .any(|token| token.eq_ignore_ascii_case("canonical"))
            })
        })
        .and_then(|node| node.value().attr("href"))
        .and_then(|value| safe_url(value, base.as_ref()))
        .filter(|value| {
            !base.as_ref().is_some_and(|source| {
                !source.path().trim_matches('/').is_empty()
                    && Url::parse(value)
                        .is_ok_and(|canonical| canonical.path().trim_matches('/').is_empty())
            })
        })
    {
        metadata.insert("canonical_url".into(), canonical.into());
    }
    if let Some(base) = &base {
        metadata.insert("source".into(), base.as_str().into());
        if let Some(host) = base.host_str() {
            metadata.insert("domain".into(), host.into());
        }
    }
    let markdown = if let Some(announcement) = structured_announcement(&document, base.as_ref())? {
        metadata.extend(announcement.metadata);
        announcement.markdown
    } else {
        render_with_footnotes(root, document.root_element(), base.as_ref(), true)?
    };
    if markdown.is_empty() {
        return Err(Error::Conversion(
            "HTML contains no extractable content".into(),
        ));
    }
    metadata.insert("converter".into(), "native-html".into());
    Ok(Document {
        markdown,
        metadata,
        ..Document::default()
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn streamed_article_keeps_metadata_links_and_resolved_notes() {
        let result = extract_html(
            r##"<title>Recovered article</title><main><!--$?--><template id="B:0"></template><p>Loading content</p><!--/$--></main>
            <div hidden id="S:0"><article><h1>Recovered article</h1><p>Readable evidence<a role="doc-noteref" href="#fn-1">1</a> and <a href="/source">its source</a>.</p><aside id="fn-1" role="doc-footnote">The complete supporting note.</aside><p hidden>Still hidden</p></article></div>
            <script>$RC("B:0","S:0");</script>"##,
            Some("https://example.test/article"),
        )
        .unwrap();
        assert_eq!(result.metadata["title"], "Recovered article");
        assert!(result.markdown.contains("Readable evidence[^1]"));
        assert!(
            result
                .markdown
                .contains("[its source](https://example.test/source)")
        );
        assert!(
            result
                .markdown
                .contains("[^1]: The complete supporting note.")
        );
        assert!(!result.markdown.contains("Loading content"));
        assert!(!result.markdown.contains("Still hidden"));
    }

    #[test]
    fn article_pruning_keeps_external_notes_and_drops_widget_only_references() {
        let source = r##"<body><article><p>Evidence from the article remains available together with its source and full explanatory paragraph. This main narrative must survive the removal of unrelated recommendation controls.<a role="doc-noteref" href="#fn-main">1</a></p>
            <div class="related-posts"><h2>Related posts</h2><a href="/other">Other story</a><a role="doc-noteref" href="#fn-widget">2</a></div></article>
            <aside role="doc-footnote" id="fn-main">The retained external source.</aside><aside role="doc-footnote" id="fn-widget">Widget-only definition.</aside></body>"##;
        let result = extract_html(source, None).unwrap();
        assert!(result.markdown.contains("main narrative must survive"));
        assert!(result.markdown.contains("[^1]"));
        assert!(
            result
                .markdown
                .ends_with("[^1]: The retained external source."),
            "{}",
            result.markdown
        );
        assert!(!result.markdown.contains("Other story"));
        assert!(!result.markdown.contains("Widget-only"));
        assert!(!result.markdown.contains("[^2]"));
    }

    #[test]
    fn document_fragments_retain_book_toc_and_related_reading() {
        let result = fragment(r##"<section class="toc"><h2>Contents</h2><ul><li><a href="#chapter">Chapter one</a></li></ul></section><section class="related-posts"><p>Related reading belongs to the book.</p></section><h2 id="chapter">Chapter one</h2><p>Full chapter text.</p>"##).unwrap();
        assert!(result.contains("[Chapter one](#chapter)"));
        assert!(result.contains("Related reading belongs to the book."));
        assert!(result.contains("Full chapter text."));
    }

    #[test]
    fn external_note_roles_do_not_recover_inert_or_literal_definitions() {
        for wrapper in ["pre", "template", "noscript"] {
            let source = format!(
                r##"<article><p>A substantial visible article retains its explanatory text and linked evidence. The referenced source is absent from rendered content and must not be recovered from an inert example.<a role="doc-noteref" href="#fn-example">1</a></p></article><{wrapper}><aside role="doc-footnote" id="fn-example">Inert definition.</aside></{wrapper}>"##
            );
            let result = extract_html(&source, None).unwrap();
            assert!(
                result.markdown.contains("[1](#fn-example)"),
                "{wrapper}: {}",
                result.markdown
            );
            assert!(
                !result.markdown.contains("[^1]"),
                "{wrapper}: {}",
                result.markdown
            );
            assert!(
                !result.markdown.contains("Inert definition"),
                "{wrapper}: {}",
                result.markdown
            );
        }
    }

    #[test]
    fn code_editor_numeric_links_are_literal_before_footnote_collection() {
        let result = extract_html(r##"<article><p>Example:</p><div class="cm-content"><div class="cm-line">goto <a role="doc-noteref" href="#fn-code">1</a></div><div class="cm-line">return 2</div></div><p>Explanation<a role="doc-noteref" href="#fn-real">2</a>.</p><ol class="footnotes"><li id="fn-code">Not a code footnote.</li><li id="fn-real">The prose source.</li></ol></article>"##, None).unwrap();
        assert!(
            result.markdown.contains("```\ngoto 1\nreturn 2\n```"),
            "{}",
            result.markdown
        );
        assert!(
            result.markdown.contains("Explanation[^1]."),
            "{}",
            result.markdown
        );
        assert!(result.markdown.ends_with("[^1]: The prose source."));
        assert!(!result.markdown.contains("[^2]"));
    }

    #[test]
    fn article_drops_navigation_and_resolves_links() {
        let doc = extract_html(r#"<title>Fallback</title><meta property="og:title" content="Article"><nav>menu</nav><article><h1>Article</h1><p>Some <strong>important</strong> text <a href="../next">next</a>.</p><img data-src="/image.png" alt="Photo"><p hidden>hidden</p><script>alert(1)</script></article><footer>footer</footer>"#, Some("https://example.test/posts/one")).unwrap();
        assert_eq!(doc.metadata["title"], "Fallback");
        assert!(doc.markdown.contains("**important**"));
        assert!(doc.markdown.contains("https://example.test/next"));
        assert!(doc.markdown.contains("https://example.test/image.png"));
        for excluded in ["menu", "hidden", "alert", "footer"] {
            assert!(!doc.markdown.contains(excluded));
        }
    }
    #[test]
    fn article_keeps_code_table_unicode_and_safe_text() {
        let doc = extract_html(r#"<main><h1>中文</h1><pre><code class="language-rust">let n = 1 &lt; 2;</code></pre><table><tr><th>A</th><th>B</th></tr><tr><td>x</td><td>y</td></tr></table><a href="javascript:alert(1)">safe label</a></main>"#, None).unwrap();
        assert!(doc.markdown.contains("let n = 1 < 2;"));
        assert!(doc.markdown.contains("|"));
        assert!(doc.markdown.contains("safe label"));
        assert!(!doc.markdown.contains("javascript:"));
    }

    #[test]
    fn structured_metadata_uses_graph_headline_and_author() {
        let doc = extract_html(r#"<title>Wrong title</title><meta property="og:title" content="Wrong social title"><meta name="og:site_name" content="Journal"><script type="application/ld+json">{"@graph":[{"name":"Website"},{"headline":"Journal - Structured headline · Journal","author":{"name":"Ada"},"datePublished":"2026-02-03"}]}</script><main><p>Article text.</p></main>"#, None).unwrap();
        assert_eq!(doc.metadata["title"], "Structured headline");
        assert_eq!(doc.metadata["author"], "Ada");
        assert_eq!(doc.metadata["published"], "2026-02-03");
        assert!(!doc.markdown.contains("Structured headline"));
    }

    #[test]
    fn title_layers_recover_truncation_and_keep_real_subtitles() {
        let doc = extract_html(r#"<title>A complete long headline</title><script type="application/ld+json">{"headline":"A complete…"}</script><p>Body</p>"#, None).unwrap();
        assert_eq!(doc.metadata["title"], "A complete long headline");
        for (title, expected) in [
            ("Article | Site", "Article"),
            ("Article - a retrospective", "Article - a retrospective"),
        ] {
            let doc = extract_html(
                &format!("<title>{title}</title><h1>Article</h1><p>Body</p>"),
                None,
            )
            .unwrap();
            assert_eq!(doc.metadata["title"], expected);
        }
        let long = "界".repeat(350);
        assert_eq!(clean_title(&long, None).unwrap().chars().count(), 301);
    }

    #[test]
    fn malformed_jsonld_and_meta_author_do_not_hide_valid_layers() {
        let doc = extract_html(r#"<meta name="author" content="Meta author"><meta property="article:published_time" content="2026-01-01"><title>Document title</title><meta property="og:title" content="Social title"><script type="application/ld+json">invalid</script><script type="application/ld+json">[{"author":"Schema author","datePublished":"2025-01-01"}]</script><p>Body</p>"#, None).unwrap();
        assert_eq!(doc.metadata["title"], "Document title");
        assert_eq!(doc.metadata["author"], "Meta author");
        assert_eq!(doc.metadata["published"], "2026-01-01");
        let doc = extract_html(
            "<time datetime=PT2H>Duration</time><time datetime=2026-03-04>Today</time>",
            None,
        )
        .unwrap();
        assert_eq!(doc.metadata["published"], "2026-03-04");
    }

    #[test]
    fn canonical_homepage_is_not_used_for_article() {
        let html = r#"<link rel="alternate canonical" href="https://example.test"><p>Body</p>"#;
        assert!(
            !extract_html(html, Some("https://example.test/post"))
                .unwrap()
                .metadata
                .contains_key("canonical_url")
        );
        assert_eq!(
            extract_html(html, Some("https://example.test/"))
                .unwrap()
                .metadata["canonical_url"],
            "https://example.test"
        );
    }

    #[test]
    fn inline_quotes_and_source_lines_preserve_reference_spelling() {
        let doc = extract_html("<p>WWF's goal is to:\n    <q>Build a <em>better</em> future.</q>\n</p>\n<a href=\"https://example.com\">Example Link</a>\n<img src=\"https://example.com/test.jpg\" alt=\"Test Image\">", None).unwrap();
        assert_eq!(
            doc.markdown,
            "WWF's goal is to:\n\"Build a *better* future.\"\n\n[Example Link](https://example.com)\n![Test Image](https://example.com/test.jpg)"
        );
        assert_eq!(
            fragment("<p>One <q>two <q>three</q></q> four.</p>").unwrap(),
            "One \"two \"three\"\" four."
        );
    }

    #[test]
    fn urls_are_validated_without_rewriting_absolute_spelling() {
        let base = Url::parse("https://example.test/posts/one").unwrap();
        for value in [
            "https://EXAMPLE.test",
            "https://example.test/%2f?q=a%2Bb#part",
            "mailto:name@example.test",
            "tel:+12025550123",
        ] {
            assert_eq!(safe_url(value, Some(&base)).as_deref(), Some(value));
        }
        assert_eq!(
            safe_url("../two", Some(&base)).as_deref(),
            Some("https://example.test/two")
        );
        assert_eq!(
            safe_url("https://example.test/<tag>\"", None).as_deref(),
            Some("https://example.test/%3Ctag%3E%22")
        );
        for value in [
            "javascript:alert(1)",
            "data:text/html,body",
            "java\nscript:alert(1)",
            "https://exam\tple.test/",
        ] {
            assert!(safe_url(value, Some(&base)).is_none());
        }
    }

    #[test]
    fn inline_image_data_keeps_reference_placeholder_without_payload() {
        let doc = extract_html(
            "<main><p>Before</p><p><img alt=\"blue box\" src=\"data:image/png;base64,iVBORw0KGgo=\"></p>\
             <p><img alt=\"vector\" src=\"DATA:image/svg+xml;base64,PHN2Zz4=\"></p>\
             <p><img alt=\"page\" src=\"data:text/html,<b>x</b>\"></p><p>After</p></main>",
            None,
        )
        .unwrap();
        assert!(
            doc.markdown
                .contains("![blue box](data:image/png;base64...)"),
            "{}",
            doc.markdown
        );
        assert!(!doc.markdown.contains("iVBOR"));
        assert!(!doc.markdown.contains("vector") && !doc.markdown.contains("svg"));
        assert!(!doc.markdown.contains("text/html"));
        assert_eq!(
            image_source("data:image/png;base64", None).as_deref(),
            Some("data:image/png;base64...")
        );
        assert!(image_source(&format!("data:{}", "a".repeat(300)), None).is_none());
        assert!(image_source("data:image/png;x=(1),AAAA", None).is_none());
        // Links never accept data URLs, only image sources do.
        assert!(safe_url("data:image/png;base64,AAAA", None).is_none());
    }

    #[test]
    fn block_boundaries_code_and_chemical_notation_survive() {
        let doc = extract_html("<main><div>First</div><div>Second</div><p>H<sub>2</sub>O and 10<sup>n</sup>.</p><pre><code>one\n  two\n\nthree</code></pre><code style=\"white-space: pre\">a\n  b</code></main>", None).unwrap();
        assert!(
            doc.markdown
                .starts_with("First\n\nSecond\n\nH<sub>2</sub>O and 10<sup>n</sup>.")
        );
        assert!(doc.markdown.contains("one\n  two\n\nthree"));
        assert!(doc.markdown.contains("```\na\n  b\n```"));
        assert!(!doc.markdown.contains("markitai-soft-break"));
    }

    #[test]
    fn candidate_inside_hidden_ancestor_cannot_become_article() {
        let doc = extract_html("<div hidden><article><p>Hidden text that outweighs the visible article by length.</p></article></div><main><p>Visible</p></main>", None).unwrap();
        assert_eq!(doc.markdown, "Visible");
    }

    #[test]
    fn css_custom_properties_preserve_visible_article_content() {
        let doc = extract_html(r#"<div style="--footer-display: none; --graph-controls-display: none;"><article><h1>开发者</h1><p>如果你熟悉 TypeScript 或 CSS，可以开发插件。</p></article></div>"#, None).unwrap();
        assert!(doc.markdown.contains("开发者"));
        assert!(doc.markdown.contains("可以开发插件"));
    }

    #[test]
    fn hidden_style_checks_property_tokens_not_embedded_strings_or_comments() {
        for style in [
            "--foo-display:none; --foo-visibility:hidden",
            "content: 'display:none; visibility:hidden'",
            r#"content: "escaped \"; display:none"; color: red"#,
            "/* display:none; visibility:hidden */ color:red",
            "background:url('x;display:none'); color:red",
            "--theme: { display:none; visibility:hidden }; color:red",
            "display:'none'; visibility:\"hidden\"",
            "dis/**/play:none; visibility:hiddenish",
            "content: 'unfinished; display:none",
            "color:red; /* unfinished comment; display:none",
        ] {
            assert!(!hidden_inline_style(style), "unexpectedly hidden: {style}");
        }
        for style in [
            "DISPLAY : NoNe",
            "display: /* explanation */ none ! IMPORTANT",
            "/* declaration */ visibility: hidden",
            "visibility:collapse",
            "content:'display:block;'; display:none",
            "--theme:{display:block;}; visibility:hidden",
        ] {
            assert!(hidden_inline_style(style), "expected hidden: {style}");
        }
    }

    #[test]
    fn inline_hidden_properties_follow_order_and_importance() {
        for style in [
            "display:none; display:block",
            "visibility:hidden!important; visibility:visible!important",
            "display:none; display:block!important; display:none",
        ] {
            assert!(!hidden_inline_style(style), "unexpectedly hidden: {style}");
        }
        for style in [
            "display:block; display:none",
            "display:none!important; display:block",
            "visibility:hidden; display:block",
            "display:none; display:'block'",
        ] {
            assert!(hidden_inline_style(style), "expected hidden: {style}");
        }
        let doc = extract_html(r#"<main><p style="content:'display:none'">Visible</p><p style="display:/* comment */none">Hidden</p><p hidden style="display:block">Also hidden</p></main>"#, None).unwrap();
        assert_eq!(doc.markdown, "Visible");
    }

    #[test]
    fn mathjax_tex_sources_replace_only_their_duplicate_previews() {
        let doc = extract_html(r#"<main><p>Energy <span class="MathJax_Preview">duplicate preview</span><span class="MathJax">duplicate glyphs</span><script type="math/tex">E_u</script> remains.</p><script type="math/tex; mode=display">\frac{a}{b}</script><script type="application/javascript">alert('bad')</script><script type="math/tex-evil">not math</script><p><span class="MathJax">Visible unannotated text</span></p></main>"#, None).unwrap();
        assert!(doc.markdown.contains("Energy $E_u$ remains."));
        assert!(doc.markdown.contains(r"$$\frac{a}{b}$$"));
        assert!(doc.markdown.contains("Visible unannotated text"));
        for text in ["duplicate", "alert", "not math"] {
            assert!(!doc.markdown.contains(text));
        }
    }

    #[test]
    fn katex_annotations_and_data_sources_are_emitted_once() {
        let doc = extract_html(r#"<main><p>A <span class="katex"><span class="katex-mathml"><math><semantics><mrow><mi>x</mi><mo>+</mo><mi>y</mi></mrow><annotation encoding="application/x-tex">x+y</annotation></semantics></math></span><span class="katex-html">duplicate</span></span> term.</p><span class="katex-display"><span class="katex"><math><semantics><mi>z</mi><annotation encoding="application/x-tex">z^2</annotation></semantics></math><span class="katex-html">other duplicate</span></span></span><p><span class="math-inline" data-math="\forall x"><span class="katex">visual x</span></span> and <span class="hurmet-tex" data-entry="\vec{F}"><math><mi>F</mi></math></span>.</p></main>"#, None).unwrap();
        assert!(doc.markdown.contains("A $x+y$ term."));
        assert_eq!(doc.markdown.matches("z^2").count(), 1);
        assert!(doc.markdown.contains("$$z^2$$"));
        assert!(doc.markdown.contains(r"$\forall x$ and $\vec{F}$."));
        assert!(!doc.markdown.contains("duplicate"));
        assert!(!doc.markdown.contains("visual"));
    }

    #[test]
    fn hidden_assistive_math_is_recovered_inside_known_visual_wrappers() {
        let doc = extract_html(r#"<p>Quadratic <span class="mwe-math-element"><span style="display:none"><math alttext="ax^2+bx+c=0"><mi>fallback</mi></math></span><img src="https://example.test/equation.svg" alt="ax^2+bx+c=0"></span>.</p><mjx-container display="true"><svg><path></path></svg><mjx-assistive-mml style="display:none"><math><msub><mi>r</mi><mtext>perf</mtext></msub></math></mjx-assistive-mml></mjx-container>"#, None).unwrap();
        assert!(doc.markdown.contains("Quadratic $ax^2+bx+c=0$."));
        assert!(doc.markdown.contains(r"$$r_{\text{perf}}$$"));
        assert!(!doc.markdown.contains("equation.svg"));
        assert!(!doc.markdown.contains("fallback"));
    }

    #[test]
    fn nested_presentation_mathml_keeps_fraction_roots_scripts_and_accents() {
        let doc = extract_html(r#"<math display="block"><mrow><mfrac><msubsup><mi>x</mi><mn>1</mn><mn>2</mn></msubsup><mroot><mi>y</mi><mn>3</mn></mroot></mfrac><mo>+</mo><msqrt><mrow><mi>α</mi><mo>×</mo><mi>β</mi></mrow></msqrt><mo>=</mo><mover><mi>F</mi><mo>→</mo></mover></mrow></math>"#, None).unwrap();
        assert_eq!(
            doc.markdown,
            r"$$\frac{x_{1}^{2}}{\sqrt[3]{y}} + \sqrt{\alpha \times \beta} = \vec{F}$$"
        );
        let doc = extract_html(r#"<math><mfenced open="[" close="]"><mi>a</mi><mfrac><mn>1</mn><mi>b</mi></mfrac></mfenced></math>"#, None).unwrap();
        assert_eq!(doc.markdown, r"$\left[a, \frac{1}{b}\right]$");
    }

    #[test]
    fn mathml_tables_preserve_rows_and_columns() {
        let doc = extract_html(r#"<math display="block"><mtable><mtr><mtd><mi>x</mi></mtd><mtd><mo>=</mo></mtd><mtd><mi>y</mi></mtd></mtr><mtr><mtd><mi>u</mi></mtd><mtd><mo>=</mo></mtd><mtd><mi>v</mi></mtd></mtr></mtable></math>"#, None).unwrap();
        assert_eq!(
            doc.markdown,
            r"$$\begin{aligned}x & = & y \\ u & = & v\end{aligned}$$"
        );
    }

    #[test]
    fn math_annotations_cannot_emit_raw_html_or_execute_active_scripts() {
        let doc = extract_html(r#"<p><span class="math-inline" data-math="x &lt;img src=x onerror=alert(1)&gt;">visual</span></p><script>alert(2)</script><pre>&lt;math&gt;$x_1$&lt;/math&gt;</pre>"#, None).unwrap();
        assert!(
            doc.markdown
                .contains(r"$x \lt img src=x onerror=alert(1)\gt $")
        );
        assert!(!doc.markdown.contains("alert(2)"));
        assert!(!doc.markdown.contains("<img"));
        assert!(doc.markdown.contains("<math>$x_1$</math>"));
    }

    #[test]
    fn deep_mathml_fails_explicitly_instead_of_overflowing() {
        let html = format!(
            "<math>{}<mi>x</mi>{}</math>",
            "<mrow>".repeat(66),
            "</mrow>".repeat(66)
        );
        assert!(
            matches!(extract_html(&html, None), Err(Error::Conversion(message)) if message.contains("MathML nesting"))
        );
    }

    #[test]
    fn mathml_skips_active_nodes_and_xml_annotations_inside_nested_text() {
        let doc = extract_html(r#"<math><mrow><mi>x</mi><script>secret_script()</script><style>secret_style</style><annotation-xml encoding="text/html"><mtext>secret_xml</mtext></annotation-xml><mo>+</mo><mtext>safe<script>secret_nested()</script> text</mtext></mrow></math>"#, None).unwrap();
        assert!(doc.markdown.contains("x +"));
        assert!(doc.markdown.contains(r"\text{safe text}"));
        assert!(!doc.markdown.contains("secret"));
    }

    fn announcement_html(body: &str) -> String {
        let data = serde_json::json!([{"announcement_body": {"body": body, "headline": "Patch notes", "posttime": 1736942400}}]);
        let mut html = String::from("<title>Generic title</title><div data-partnereventstore=\"");
        escaped(&data.to_string(), &mut html);
        html.push_str("\" data-groupvanityinfo='[{\"group_name\":\"Example Game\"}]'></div>");
        html
    }

    #[test]
    fn structured_announcement_recovers_metadata_links_and_video_from_json() {
        let html = announcement_html(
            r#"[p]Patch [b]released[/b]. Read [url=\"https:\/\/example.test\/notes\"]the notes[\/url].[\/p][previewyoutube=\"dQw4w9WgXcQ;full\"][\/previewyoutube]"#,
        );
        let doc = extract_html(&html, None).unwrap();
        assert_eq!(doc.metadata["title"], "Patch notes");
        assert_eq!(doc.metadata["author"], "Example Game");
        assert_eq!(doc.metadata["published"], "2025-01-15T12:00:00+00:00");
        assert!(doc.markdown.contains("Patch **released**."));
        assert!(
            doc.markdown
                .contains("[the notes](https://example.test/notes)")
        );
        assert!(
            doc.markdown
                .contains("![](https://www.youtube.com/watch?v=dQw4w9WgXcQ)")
        );
    }

    #[test]
    fn nested_bbcode_lists_quotes_and_literal_code_keep_structure() {
        let html = announcement_html(
            "[quote=Ada][list][*]First [b]bold [i]inner[/i][/b][list][*]Nested[/list][*]Second[/list][/quote][code=rust]let s = \"[b]literal[/b]\";\n  // <script> data\n[/code]",
        );
        let doc = extract_html(&html, None).unwrap();
        for text in [
            "Ada",
            "First",
            "**bold *inner***",
            "Nested",
            "Second",
            "```rust",
            "[b]literal[/b]",
            "  // <script> data",
        ] {
            assert!(
                doc.markdown.contains(text),
                "missing {text}: {}",
                doc.markdown
            );
        }
        assert!(doc.markdown.contains(">"));
    }

    #[test]
    fn bbcode_escapes_html_and_sanitizes_destinations_without_dropping_labels() {
        let html = announcement_html(
            r#"[p]<script>alert(1)</script> [url=javascript:alert(2)]safe label[/url] [img]data:text/html,bad[/img] [unknown]keep me[/unknown][/p]"#,
        );
        let doc = extract_html(&html, None).unwrap();
        assert!(doc.markdown.contains("safe label"));
        assert!(doc.markdown.contains("keep me"));
        assert!(
            doc.markdown
                .contains("&lt;script&gt;alert(1)&lt;/script&gt;")
        );
        assert!(!doc.markdown.contains("<script>"));
        assert!(!doc.markdown.contains("](javascript:"));
        assert!(!doc.markdown.contains("](data:"));
    }

    #[test]
    fn bbcode_protects_all_displayed_text_paths_and_keeps_image_alt_empty() {
        let html = announcement_html(
            r#"[h2]<img src=x onerror=heading>[/h2][quote="<img src=x onerror=author>"]Quoted body[/quote][url=https://example.test]<svg onload=label>Link label</svg>[/url][img]https://example.test/image.png[/img]"#,
        );
        let doc = extract_html(&html, None).unwrap();
        for expected in [
            "## &lt;img src=x onerror=heading&gt;",
            "&lt;img src=x onerror=author&gt;",
            "&lt;svg onload=label&gt;Link label&lt;/svg&gt;",
            "Quoted body",
            "![](https://example.test/image.png)",
        ] {
            assert!(
                doc.markdown.contains(expected),
                "missing {expected}: {}",
                doc.markdown
            );
        }
        assert!(!doc.markdown.contains("<img"));
        assert!(!doc.markdown.contains("<svg"));
    }

    #[test]
    fn invalid_event_data_falls_back_and_depth_limit_is_explicit() {
        let doc = extract_html(
            "<div data-partnereventstore='invalid'></div><p>Visible fallback</p>",
            None,
        )
        .unwrap();
        assert_eq!(doc.markdown, "Visible fallback");
        let body = format!("{}deep{}", "[b]".repeat(66), "[/b]".repeat(66));
        assert!(
            matches!(extract_html(&announcement_html(&body), None), Err(Error::Conversion(message)) if message.contains("BBCode nesting"))
        );
    }

    #[test]
    fn footnotes_repeated_references_preserve_inline_whitespace_and_labels() {
        let doc = extract_html(
            r##"<article><p>Before <a href="#fn-a">[1]</a> after;
            again<sup><a id="ref-a" href="#fn-a">1</a></sup>, done.
            <a href="#fn-a">Read the note</a>.</p>
            <section class="footnotes"><h2>Footnotes</h2><ol><li id="fn-a">
            <p>A <em>useful</em> note. <a href="#ref-a" role="doc-backlink">↩</a></p>
            </li></ol></section></article>"##,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.contains("Before [^1] after;"),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("again[^1], done."));
        assert!(doc.markdown.contains("[Read the note](#fn-a)"));
        assert_eq!(doc.markdown.matches("[^1]:").count(), 1);
        assert!(doc.markdown.ends_with("[^1]: A *useful* note."));
        assert!(!doc.markdown.contains("Footnotes"));
        assert!(!doc.markdown.contains('↩'));
    }

    #[test]
    fn footnotes_keep_nested_blocks_and_code_inside_the_definition() {
        let doc = extract_html(
            r##"<article><p>Claim<sup><a href="#note">1</a></sup>.</p>
            <ol class="footnotes"><li id="note"><p>First paragraph.</p>
            <ul><li>Nested item.</li></ul><blockquote>Quotation.</blockquote>
            <pre><code>literal [^9] &lt;script&gt;</code></pre><p>Last paragraph.</p>
            </li></ol></article>"##,
            None,
        )
        .unwrap();
        assert!(doc.markdown.contains("Claim[^1]."));
        assert!(
            doc.markdown
                .contains("[^1]: First paragraph.\n\n    *   Nested item."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("    > Quotation."));
        assert!(doc.markdown.contains("    literal [^9] <script>"));
        assert!(doc.markdown.ends_with("    Last paragraph."));
    }

    #[test]
    fn footnotes_restore_only_linked_hidden_definition_and_sanitize_its_body() {
        let doc = extract_html(r##"<article><p>Text<span data-definition="note"><a href="#">*</a></span> continues.</p>
            <aside id="note" style="display:none">Keep <a href="javascript:bad()">label</a>.
            <script>SECRET</script><span hidden>HIDDEN</span><a href="https://example.com/n">safe</a>.</aside>
            <aside hidden>UNRELATED</aside></article>"##, None).unwrap();
        assert!(
            doc.markdown.contains("Text[^1] continues."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("[^1]: Keep label."));
        assert!(doc.markdown.contains("[safe](https://example.com/n)"));
        assert!(!doc.markdown.contains("javascript:") && !doc.markdown.contains("SECRET"));
        assert!(!doc.markdown.contains("HIDDEN") && !doc.markdown.contains("UNRELATED"));
    }

    #[test]
    fn footnotes_inline_popovers_and_org_labels_move_each_body_once() {
        let doc = extract_html(r#"<article><p>A<span class="inline-footnote">1<span class="footnoteContent" hidden>First note.</span></span> B.</p>
            <p>C<label class="footref" for="fn.2">2</label><input id="fn.2" type="checkbox">
            <span class="sidenote"><sup>2</sup> Second note.</span> D.</p></article>"#, None).unwrap();
        assert!(doc.markdown.contains("A[^1] B."), "{}", doc.markdown);
        assert!(doc.markdown.contains("C[^2]"));
        assert!(doc.markdown.contains(" D."));
        assert!(
            doc.markdown
                .ends_with("[^1]: First note.\n\n[^2]: Second note.")
        );
        assert_eq!(doc.markdown.matches("Second note.").count(), 1);
    }

    #[test]
    fn footnotes_loose_definitions_preserve_bold_attribution_and_trailing_update() {
        let doc = extract_html(
            r#"<article><p>First<sup>1</sup> claim.</p><p>Second<sup>2</sup> claim.</p>
            <p><b><sup>1</sup> Note today:</b> First evidence.</p>
            <p><b><sup>2</sup> Note yesterday:</b> Second evidence.</p>
            <p><b>Update:</b> Keep this in the article.</p></article>"#,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.contains("First[^1] claim."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("Second[^2] claim."));
        assert!(
            doc.markdown
                .contains("**Update:** Keep this in the article.\n\n[^1]:")
        );
        assert!(
            doc.markdown
                .contains("[^1]: **Note today:** First evidence.")
        );
        assert!(
            doc.markdown
                .ends_with("[^2]: **Note yesterday:** Second evidence.")
        );
    }

    #[test]
    fn footnotes_labeled_external_section_adopts_continuations_without_unrelated_asides() {
        let doc = extract_html(
            r#"<article><p>A<sup>1</sup> and B<sup>2</sup>.</p></article>
            <aside>Unrelated advertising</aside><div class="page__footnotes"><h3>Footnotes</h3>
            <p><sup>1</sup> First evidence.</p><ul><li>Detail.</li></ul>
            <p><sup>2</sup> Second evidence.</p></div>"#,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.starts_with("A[^1] and B[^2]."),
            "{}",
            doc.markdown
        );
        assert!(
            doc.markdown
                .contains("[^1]: First evidence.\n\n    *   Detail.")
        );
        assert!(doc.markdown.ends_with("[^2]: Second evidence."));
        assert!(!doc.markdown.contains("Unrelated"));
    }

    #[test]
    fn footnotes_generic_named_anchors_require_a_shared_definition_section() {
        let doc = extract_html(
            r##"<article><p>Fact <a href="#one">[1]</a> and <a href="#two">[2]</a>.</p>
            <div class="notes"><p><a name="one"><sup>1</sup></a> First.</p>
            <p><a name="two"><sup>2</sup></a> Second.</p></div></article>"##,
            None,
        )
        .unwrap();
        assert!(
            doc.markdown.starts_with("Fact [^1] and [^2]."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: First.\n\n[^2]: Second."));
        let ordinary = extract_html(r##"<article><p>See <a href="#equation">1</a> and <a href="#theorem">2</a>.</p>
            <h2 id="equation">Equation</h2><p>x=2</p><h2 id="theorem">Theorem</h2><p>Proof.</p></article>"##, None).unwrap();
        assert!(!ordinary.markdown.contains("[^"));
        assert!(ordinary.markdown.contains("[1](#equation)"));
        assert!(ordinary.markdown.contains("## Theorem"));
    }

    #[test]
    fn footnotes_missing_targets_and_remote_numeric_links_remain_links() {
        let doc = extract_html(r##"<article><p><a href="#missing">1</a> and <a href="https://elsewhere.test/#fn-2">2</a>.</p>
            <ol class="footnotes"><li id="fn-2">Unreferenced but valid content.</li></ol>
            <pre><code>&lt;a href="#fn-2"&gt;2&lt;/a&gt;</code></pre></article>"##, Some("https://example.com/post")).unwrap();
        assert!(!doc.markdown.contains("[^"));
        assert!(doc.markdown.contains("[1](#missing)"));
        assert!(doc.markdown.contains("Unreferenced but valid content."));
        assert!(doc.markdown.contains("<a href=\"#fn-2\">2</a>"));
    }

    #[test]
    fn footnotes_do_not_delete_unmatched_sidenote_text_or_following_prose() {
        let doc = extract_html(r##"<article><p>Claim<sup><a href="#fn-1">1</a></sup><span class="sidenote">Different marginal remark.</span></p>
            <div class="footnotes"><h2>Footnotes</h2><ol><li id="fn-1">Evidence.</li></ol>
            <p>Independent concluding prose.</p></div></article>"##, None).unwrap();
        assert!(doc.markdown.contains("Different marginal remark."));
        assert!(doc.markdown.contains("Independent concluding prose."));
        assert!(doc.markdown.ends_with("[^1]: Evidence."));
    }

    #[test]
    fn footnotes_aside_start_and_word_backlinks_use_their_explicit_numbers() {
        let doc = extract_html(r##"<article><p>Claim<sup>7</sup> and Word<sup><a href="#_ftn1">[1]</a></sup>.</p>
            <aside><ol start="7"><li>Aside evidence.</li></ol></aside>
            <p><sup><a href="//document-guid#_ftnref1">[1]</a></sup> Word evidence.</p></article>"##, None).unwrap();
        assert!(
            doc.markdown.starts_with("Claim[^1] and Word[^2]."),
            "{}",
            doc.markdown
        );
        assert!(
            doc.markdown
                .ends_with("[^1]: Aside evidence.\n\n[^2]: Word evidence.")
        );
        assert!(!doc.markdown.contains("document-guid"));
    }

    #[test]
    fn footnotes_do_not_create_dangling_markers_for_empty_sanitized_bodies() {
        let doc = extract_html(
            r##"<article><p>Claim<sup><a href="#fn-1">1</a></sup>.</p>
            <div class="footnotes"><p id="fn-1"><script>unsafe()</script>
            <a role="doc-backlink" href="#ref-1">↩</a></p></div></article>"##,
            None,
        )
        .unwrap();
        assert!(!doc.markdown.contains("[^"));
        assert!(doc.markdown.contains("#fn-1"));
        assert!(!doc.markdown.contains("unsafe()"));
    }

    #[test]
    fn source_newlines_preserve_separating_spaces_after_inline_elements() {
        let markdown = fragment(
            "<p>First <a href=\"/x\">label</a> tail;\n    next <code>x</code> tail;\n  last.</p>",
        )
        .unwrap();
        assert_eq!(markdown, "First [label](/x) tail;\nnext `x` tail;\nlast.");
    }

    #[test]
    fn hidden_or_discarded_popover_references_do_not_restore_their_definitions() {
        for hidden in [
            r#"<span class="inline-footnote" hidden>1<span class="footnoteContent">Hidden secret</span></span>"#,
            r#"<div hidden><span class="inline-footnote">1<span class="footnoteContent">Hidden secret</span></span></div>"#,
            r#"<div style="display:none"><span class="footnote-container"><span class="footnote">Hidden secret</span></span></div>"#,
            r#"<nav><span class="inline-footnote">1<span class="footnoteContent">Hidden secret</span></span></nav>"#,
            r#"<span class="inline-footnote"><span hidden><span class="footnoteContent">Hidden secret</span></span></span>"#,
        ] {
            let doc =
                extract_html(&format!("<article><p>Visible.</p>{hidden}</article>"), None).unwrap();
            assert_eq!(doc.markdown, "Visible.", "{hidden}");
        }
        let visible = extract_html(r#"<article><p>Visible<span class="inline-footnote">1<span class="footnoteContent" hidden>Intended note.</span></span>.</p></article>"#, None).unwrap();
        assert_eq!(visible.markdown, "Visible[^1].\n\n[^1]: Intended note.");
    }

    #[test]
    fn ordinary_numbered_navigation_does_not_turn_instructions_into_notes() {
        for targets in [
            r#"<section><h2>Instructions</h2><p id="step1">Install app.</p><p id="step2">Launch app.</p></section>"#,
            r#"<ol><li id="step1">Install app.</li><li id="step2">Launch app.</li></ol>"#,
        ] {
            let source = format!(
                r##"<article><p>Steps <a href="#step1">1</a> and <a href="#step2">2</a>.</p>{targets}</article>"##
            );
            let doc = extract_html(&source, None).unwrap();
            assert!(!doc.markdown.contains("[^"), "{}", doc.markdown);
            assert!(doc.markdown.contains("[1](#step1) and [2](#step2)"));
            assert_eq!(doc.markdown.matches("Install app.").count(), 1);
            assert_eq!(doc.markdown.matches("Launch app.").count(), 1);
        }
    }

    #[test]
    fn footnote_detection_excludes_all_recognized_math_wrappers() {
        for wrapper in [
            "math-inline",
            "math-block",
            "katex",
            "katex-display",
            "hurmet-tex",
            "mwe-math-element",
        ] {
            for visual in ["<sup>2</sup>", "x<sup>2</sup>"] {
                let source = format!(
                    r#"<article><p>Formula <span class="{wrapper}" data-math="x^2">{visual}</span>.</p><ol class="footnotes"><li id="fn-1">Unused one.</li><li id="fn-2">Unused two.</li></ol></article>"#
                );
                let doc = extract_html(&source, None).unwrap();
                assert!(!doc.markdown.contains("[^"), "{wrapper}: {}", doc.markdown);
                assert_eq!(doc.markdown.matches("x^2").count(), 1);
                assert!(doc.markdown.contains("Unused two."));
            }
        }
        let doc = extract_html(r#"<article><p><mjx-container data-math="x^2"><sup>2</sup></mjx-container></p><ol class="footnotes"><li id="fn-1">Unused one.</li><li id="fn-2">Unused two.</li></ol></article>"#, None).unwrap();
        assert!(doc.markdown.contains("$x^2$"));
        assert!(!doc.markdown.contains("[^"));
    }

    #[test]
    fn backlink_only_definitions_do_not_leave_dangling_or_skipped_numbers() {
        let doc = extract_html(r##"<article><p>See <a id="ref1" href="#fn-1">1</a> and <a href="#fn-2">2</a>.</p>
            <ol class="footnotes"><li id="fn-1"><a href="#ref1">Back</a></li><li id="fn-2">Real note.</li></ol></article>"##, None).unwrap();
        assert!(
            doc.markdown.contains("[1](#fn-1) and [^1]"),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: Real note."));
        assert!(!doc.markdown.contains("[^2]"));
        assert!(doc.markdown.contains("[Back](#ref1)"));
    }

    #[test]
    fn oversized_footnote_list_start_does_not_overflow_or_drop_definitions() {
        let source = format!(
            r##"<article><p>A<a href="#a">1</a> and B<a href="#b">2</a>.</p><ol class="footnotes" start="{}"><li id="a">First.</li><li id="b">Second.</li></ol></article>"##,
            usize::MAX
        );
        let doc = extract_html(&source, None).unwrap();
        assert!(
            doc.markdown.contains("A[^1] and B[^2]."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: First.\n\n[^2]: Second."));
    }

    #[test]
    fn footnotes_ignore_duplicate_mathjax_previews_with_tex_source() {
        for wrapper in [
            "MathJax_Preview",
            "MathJax",
            "MathJax_Display",
            "MathJax_SVG",
            "MathJax_MathML",
        ] {
            let source = format!(
                r#"<article><p>Formula <span class="{wrapper}"><sup>2</sup></span><script type="math/tex">x^2</script>.</p><ol class="footnotes"><li id="fn-1">Unused one.</li><li id="fn-2">Unused two.</li></ol></article>"#
            );
            let doc = extract_html(&source, None).unwrap();
            assert!(!doc.markdown.contains("[^"), "{wrapper}: {}", doc.markdown);
            assert!(doc.markdown.starts_with("Formula $x^2$."));
            assert_eq!(doc.markdown.matches("Unused two.").count(), 1);
        }
    }

    #[test]
    fn generic_footnote_continuations_remain_with_their_definition() {
        let doc = extract_html(
            r##"<article><p>Claims <a href="#first">1</a> and <a href="#second">2</a>.</p>
            <div class="note"><p id="first">1. First evidence.</p><p>Continuation paragraph.</p>
            <ul><li>Supporting detail.</li></ul><blockquote>Quoted detail.</blockquote>
            <pre><code>literal x &lt; y</code></pre><p>Final continuation.</p>
            <p id="second">2. Second evidence.</p><p class="seealso">See also: another topic.</p>
            <p>Keep this with the related section.</p></div><p>Outside the notes.</p></article>"##,
            None,
        )
        .unwrap();
        assert!(doc.markdown.contains("Claims [^1] and [^2]."));
        let definition = doc.markdown.find("[^1]: First evidence.").unwrap();
        for text in [
            "Continuation paragraph.",
            "Supporting detail.",
            "Quoted detail.",
            "literal x < y",
            "Final continuation.",
        ] {
            assert_eq!(
                doc.markdown.matches(text).count(),
                1,
                "{text}: {}",
                doc.markdown
            );
            assert!(doc.markdown.find(text).unwrap() > definition);
        }
        assert!(
            doc.markdown
                .contains("\n\n    Continuation paragraph.\n\n    *   Supporting detail.")
        );
        assert!(doc.markdown.contains("\n    literal x < y\n"));
        assert!(doc.markdown.find("See also: another topic.").unwrap() < definition);
        assert!(doc.markdown.find("Outside the notes.").unwrap() < definition);
        assert!(doc.markdown.ends_with("[^2]: Second evidence."));
    }

    #[test]
    fn generic_footnotes_stop_at_section_id_and_update_boundaries() {
        for boundary in [
            "<h3>Next section</h3>",
            "<p id=\"new-section\">Next section</p>",
            "<p><a name=\"new-section\"></a>Next section</p>",
            "<p><b>Update today:</b> Next section</p>",
        ] {
            let source = format!(
                r##"<article><p><a href="#a">1</a> and <a href="#b">2</a>.</p><div class="notes"><p id="a">1. First.</p><p id="b">2. Second.</p>{boundary}<p>Independent content.</p></div></article>"##
            );
            let doc = extract_html(&source, None).unwrap();
            assert!(
                doc.markdown.find("Independent content.").unwrap()
                    < doc.markdown.find("[^1]:").unwrap(),
                "{}",
                doc.markdown
            );
            assert!(doc.markdown.ends_with("[^1]: First.\n\n[^2]: Second."));
        }
    }

    #[test]
    fn hidden_continuations_are_never_promoted_to_visible_definition_roots() {
        for hidden in [
            "hidden",
            "style=\"display:none\"",
            "style=\"visibility:hidden\"",
        ] {
            for notes in [
                format!(
                    r##"<p><a href="#a">1</a> and <a href="#b">2</a>.</p><div class="notes"><p id="a">1. First.</p><p {hidden}>SECRET</p><p>Visible continuation.</p><p id="b">2. Second.</p></div>"##
                ),
                format!(
                    r#"<p>A<sup>1</sup> B<sup>2</sup>.</p><hr><p><sup>1</sup>First.</p><p {hidden}>SECRET</p><p>Visible continuation.</p><p><sup>2</sup>Second.</p>"#
                ),
            ] {
                let doc = extract_html(&format!("<article>{notes}</article>"), None).unwrap();
                assert!(!doc.markdown.contains("SECRET"), "{}", doc.markdown);
                assert!(
                    doc.markdown
                        .contains("[^1]: First.\n\n    Visible continuation.")
                );
                assert!(doc.markdown.ends_with("[^2]: Second."));
            }
        }
    }

    #[test]
    fn generic_numeric_prefix_removal_requires_the_actual_reference_number() {
        let doc = extract_html(r##"<article><p><a href="#a">1</a>, <a href="#b">2</a>, <a href="#c">3</a>.</p>
            <div class="notes"><p id="a">1. <em>First evidence.</em></p><p id="b">2024. A historical date.</p><p id="c">3.5 is a decimal value.</p></div></article>"##, None).unwrap();
        assert!(
            doc.markdown.contains("[^1]: *First evidence.*"),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.contains("[^2]: 2024\\. A historical date."));
        assert!(doc.markdown.ends_with("[^3]: 3.5 is a decimal value."));
    }

    #[test]
    fn numeric_prefix_and_backlink_only_definition_is_not_emitted() {
        let doc = extract_html(r##"<article><p><a id="ref-a" href="#a">1</a> and <a href="#b">2</a>.</p>
            <div class="notes"><p id="a">1. <a href="#ref-a">Back</a></p><p id="b">2. Real content.</p></div></article>"##, None).unwrap();
        assert!(
            doc.markdown.starts_with("[1](#a) and [^1]."),
            "{}",
            doc.markdown
        );
        assert!(doc.markdown.ends_with("[^1]: Real content."));
        assert!(!doc.markdown.contains("[^2]"));
    }

    #[test]
    fn replacement_references_preserve_only_visible_wrapper_edge_whitespace() {
        let doc = extract_html(
            r##"<article><p><span>Word<span class="reference"> <sup>1</sup> </span></span>Next.</p>
            <ol class="footnotes"><li id="fn-1">Definition.</li></ol></article>"##,
            None,
        )
        .unwrap();
        assert_eq!(doc.markdown, "Word [^1] Next.\n\n[^1]: Definition.");
        let inline = extract_html(r#"<article><p>Before<span class="inline-footnote">1<span class="footnoteContent" hidden>  Hidden definition.  </span></span>After.</p></article>"#, None).unwrap();
        assert_eq!(
            inline.markdown,
            "Before[^1]After.\n\n[^1]: Hidden definition."
        );
    }
}
