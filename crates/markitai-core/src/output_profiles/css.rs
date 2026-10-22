//! CSS resource edits retain source spans and never revisit replacement text.

use super::{Replacements, html_space, unquote, visible_target};
use cssparser::{
    AtRuleParser, BasicParseErrorKind, CowRcStr, DeclarationParser, Delimiter, ParseError, Parser,
    ParserState, QualifiedRuleParser, RuleBodyItemParser, RuleBodyParser, StyleSheetParser, Token,
};
use std::ops::Range;

pub(super) struct Edit {
    pub range: Range<usize>,
    pub value: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Context {
    Sheet,
    Rules,
    Body,
    Inline,
}

struct Reader<'a> {
    source: &'a str,
    replacements: Replacements<'a>,
    visible: bool,
    context: Context,
    depth: usize,
    imports_allowed: bool,
}

enum Prelude {
    Import(Vec<Edit>),
    Block(Context),
    Layer,
}

pub(super) fn edits(
    source: &str,
    replacements: Replacements<'_>,
    visible: bool,
    stylesheet: bool,
) -> Vec<Edit> {
    let mut parser = Parser::new(source);
    Reader {
        source,
        replacements,
        visible,
        context: if stylesheet {
            Context::Sheet
        } else {
            Context::Inline
        },
        depth: 0,
        imports_allowed: stylesheet,
    }
    .read(&mut parser)
}

fn consume(parser: &mut Parser<'_>) {
    while parser.next_including_whitespace_and_comments().is_ok() {}
}

impl<'a> Reader<'a> {
    fn read(&mut self, parser: &mut Parser<'a>) -> Vec<Edit> {
        if self.depth >= 64 {
            consume(parser);
            return Vec::new();
        }
        if matches!(self.context, Context::Sheet | Context::Rules) {
            StyleSheetParser::new(parser, self)
                .filter_map(Result::ok)
                .flatten()
                .collect()
        } else {
            RuleBodyParser::new(parser, self)
                .filter_map(Result::ok)
                .flatten()
                .collect()
        }
    }

    fn child(&self, context: Context) -> Self {
        Self {
            source: self.source,
            replacements: self.replacements,
            visible: self.visible,
            context,
            depth: self.depth + 1,
            imports_allowed: false,
        }
    }

    fn resource(&self, payload: Range<usize>, decoded: &str) -> (Option<Edit>, bool) {
        // These are remote/embedded resources, not local asset filenames.
        if decoded.starts_with("//")
            || decoded.split_once(':').is_some_and(|(scheme, _)| {
                scheme
                    .as_bytes()
                    .first()
                    .is_some_and(u8::is_ascii_alphabetic)
                    && scheme
                        .bytes()
                        .all(|byte| byte.is_ascii_alphanumeric() || b"+-.".contains(&byte))
            })
        {
            return (None, false);
        }
        let split = decoded.find(['?', '#']).unwrap_or(decoded.len());
        let file = &decoded[..split];
        if file.is_empty() {
            return (None, false);
        }
        let replacement = self.replacements.get(&unquote(file));
        if replacement == Some("") {
            return (
                Some(Edit {
                    range: payload,
                    value: String::new(),
                }),
                true,
            );
        }
        let value = if let Some(path) = replacement {
            file_path(path)
        } else if self.visible {
            let Some(path) = visible_target(file) else {
                return (None, false);
            };
            // Visible profiles edit an existing URI, preserving its percent data.
            uri_path(&path, false)
        } else {
            return (None, false);
        };
        let raw = &self.source[payload.clone()];
        let end = raw_suffix(raw);
        (
            Some(Edit {
                range: payload.start..payload.start + end,
                value,
            }),
            false,
        )
    }

    fn string_resource(&self, start: usize, end: usize, decoded: &str) -> (Option<Edit>, bool) {
        let raw = &self.source[start..end];
        let quote = raw.as_bytes().first().copied();
        if !matches!(quote, Some(b'\'' | b'"'))
            || raw.len() < 2
            || raw.as_bytes().last().copied() != quote
            || escaped_last_character(raw)
        {
            return (None, false);
        }
        self.resource(start + 1..end - 1, decoded)
    }

    fn unquoted_resource(&self, start: usize, end: usize, decoded: &str) -> (Option<Edit>, bool) {
        let raw = &self.source[start..end];
        let Some(open) = raw.find('(') else {
            return (None, false);
        };
        // Leave a truncated token intact rather than repairing its syntax.
        let Some(body) = raw[open + 1..].strip_suffix(')') else {
            return (None, false);
        };
        if escaped_last_character(raw) {
            return (None, false);
        }
        let left = body.len() - body.trim_start_matches(html_space).len();
        let right = body.trim_end_matches(html_space).len();
        if left > right {
            return (None, false);
        }
        self.resource(start + open + 1 + left..start + open + 1 + right, decoded)
    }

    fn quoted_url(&self, parser: &mut Parser<'a>) -> Result<(Option<Edit>, bool), ParseError<()>> {
        parser.parse_nested_block(|inside| {
            inside.skip_whitespace();
            let start = inside.position().byte_index();
            let token = inside.next()?.clone();
            let Token::QuotedString(value) = token else {
                return Err(ParseError::custom(()));
            };
            let end = inside.position().byte_index();
            let resource = self.string_resource(start, end, &value);
            // CSS URL modifiers are identifiers or functions. Their strings are
            // not additional resource positions.
            while let Ok(token) = inside.next() {
                if !matches!(token, Token::Ident(_) | Token::Function(_)) {
                    return Err(ParseError::custom(()));
                }
            }
            Ok(resource)
        })
    }

    fn values(
        &self,
        parser: &mut Parser<'a>,
        depth: usize,
        custom_property: bool,
    ) -> Result<Vec<Edit>, ParseError<()>> {
        if depth >= 64 {
            consume(parser);
            return Ok(Vec::new());
        }
        let mut edits = Vec::new();
        loop {
            let start = parser.position().byte_index();
            let Ok(token) = parser.next_including_whitespace_and_comments().cloned() else {
                break;
            };
            match token {
                Token::UnquotedUrl(value) => {
                    let end = parser.position().byte_index();
                    let (edit, remove) = self.unquoted_resource(start, end, &value);
                    if remove {
                        edits.push(Edit {
                            range: start..end,
                            value: "url(\"\")".into(),
                        });
                    } else if let Some(edit) = edit {
                        edits.push(edit);
                    }
                }
                Token::Function(name)
                    if name.eq_ignore_ascii_case("url") || name.eq_ignore_ascii_case("src") =>
                {
                    if let Ok((Some(edit), _)) = self.quoted_url(parser) {
                        edits.push(edit);
                    }
                }
                Token::Function(name) if name.eq_ignore_ascii_case("image") => {
                    if let Ok(nested) = self.image_function(parser) {
                        edits.extend(nested);
                    }
                }
                Token::Function(name)
                    if name.eq_ignore_ascii_case("image-set")
                        || name.eq_ignore_ascii_case("-webkit-image-set") =>
                {
                    edits.extend(self.image_set(parser, start, depth + 1)?);
                }
                Token::Function(_) | Token::ParenthesisBlock | Token::SquareBracketBlock => {
                    let nested = parser.parse_nested_block(|inside| {
                        self.values(inside, depth + 1, custom_property)
                    });
                    if let Ok(nested) = nested {
                        edits.extend(nested);
                    }
                }
                Token::CurlyBracketBlock if custom_property => {
                    let nested =
                        parser.parse_nested_block(|inside| self.values(inside, depth + 1, true));
                    if let Ok(nested) = nested {
                        edits.extend(nested);
                    }
                }
                Token::CurlyBracketBlock => return Err(ParseError::custom(())),
                _ => {}
            }
        }
        Ok(edits)
    }

    fn image_function(&self, parser: &mut Parser<'a>) -> Result<Vec<Edit>, ParseError<()>> {
        parser.parse_nested_block(|inside| {
            inside.skip_whitespace();
            let mut start = inside.position().byte_index();
            let mut token = inside.next()?.clone();
            if matches!(&token, Token::Ident(name) if name.eq_ignore_ascii_case("ltr") || name.eq_ignore_ascii_case("rtl")) {
                inside.skip_whitespace();
                start = inside.position().byte_index();
                token = inside.next()?.clone();
            }
            let (edit, remove, quoted) = match token {
                Token::QuotedString(value) => {
                    let (edit, remove) = self.string_resource(start, inside.position().byte_index(), &value);
                    (edit, remove, true)
                }
                Token::UnquotedUrl(value) => {
                    let (edit, remove) = self.unquoted_resource(start, inside.position().byte_index(), &value);
                    (edit, remove, false)
                }
                Token::Function(name) if name.eq_ignore_ascii_case("url") || name.eq_ignore_ascii_case("src") => {
                    let (edit, remove) = self.quoted_url(inside)?;
                    (edit, remove, true)
                }
                _ => return Err(ParseError::custom(())),
            };
            let end = inside.position().byte_index();
            // The optional fallback is a color, never another resource string.
            if let Ok(token) = inside.next() {
                if !matches!(token, Token::Comma) { return Err(ParseError::custom(())); }
                consume(inside);
            }
            if remove && !quoted {
                Ok(vec![Edit { range: start..end, value: "url(\"\")".into() }])
            } else {
                Ok(edit.into_iter().collect())
            }
        })
    }

    fn image_set(
        &self,
        parser: &mut Parser<'a>,
        function_start: usize,
        depth: usize,
    ) -> Result<Vec<Edit>, ParseError<()>> {
        if depth >= 64 {
            parser.parse_nested_block(|inside| {
                consume(inside);
                Ok::<_, ParseError<()>>(())
            })?;
            return Ok(Vec::new());
        }
        let content_start = parser.position().byte_index();
        let mut candidates = Vec::new();
        parser.parse_nested_block(|inside| {
            loop {
                let start = inside.position().byte_index();
                if inside.is_exhausted() {
                    break;
                }
                let parsed = inside.parse_until_before(Delimiter::Comma, |candidate| {
                    let mut edits = Vec::new();
                    let mut remove = false;
                    candidate.skip_whitespace();
                    let token_start = candidate.position().byte_index();
                    let first = candidate.next()?.clone();
                    match first {
                        Token::QuotedString(value) => {
                            let result = self.string_resource(
                                token_start,
                                candidate.position().byte_index(),
                                &value,
                            );
                            if let Some(edit) = result.0 {
                                edits.push(edit);
                            }
                            remove = result.1;
                        }
                        Token::UnquotedUrl(value) => {
                            let result = self.unquoted_resource(
                                token_start,
                                candidate.position().byte_index(),
                                &value,
                            );
                            if let Some(edit) = result.0 {
                                edits.push(edit);
                            }
                            remove = result.1;
                        }
                        Token::Function(name)
                            if name.eq_ignore_ascii_case("url")
                                || name.eq_ignore_ascii_case("src") =>
                        {
                            let result = self.quoted_url(candidate)?;
                            if let Some(edit) = result.0 {
                                edits.push(edit);
                            }
                            remove = result.1;
                        }
                        Token::Function(name)
                            if name.eq_ignore_ascii_case("image-set")
                                || name.eq_ignore_ascii_case("-webkit-image-set") =>
                        {
                            edits.extend(self.image_set(candidate, token_start, depth + 1)?);
                        }
                        Token::Function(name) if name.eq_ignore_ascii_case("image") => {
                            edits.extend(self.image_function(candidate)?);
                        }
                        Token::Function(_) => {
                            edits.extend(candidate.parse_nested_block(|nested| {
                                self.values(nested, depth + 1, false)
                            })?);
                        }
                        _ => return Err(ParseError::custom(())),
                    }
                    let (mut resolution, mut media_type) = (false, false);
                    while let Ok(token) = candidate.next().cloned() {
                        match token {
                            Token::Dimension { value, unit, .. }
                                if !resolution
                                    && value.is_finite()
                                    && value > 0.0
                                    && ["x", "dppx", "dpi", "dpcm"]
                                        .iter()
                                        .any(|known| unit.eq_ignore_ascii_case(known)) =>
                            {
                                resolution = true;
                            }
                            Token::Function(name)
                                if !media_type && name.eq_ignore_ascii_case("type") =>
                            {
                                candidate.parse_nested_block(|nested| {
                                    nested.expect_string()?;
                                    nested.expect_exhausted()?;
                                    Ok::<_, ParseError<()>>(())
                                })?;
                                media_type = true;
                            }
                            _ => return Err(ParseError::custom(())),
                        }
                    }
                    Ok((edits, remove))
                });
                let end = inside.position().byte_index();
                let (edits, remove) = parsed.unwrap_or_default();
                candidates.push((start..end, edits, remove));
                if inside.next().is_err() {
                    break;
                }
            }
            Ok::<_, ParseError<()>>(())
        })?;
        let function_end = parser.position().byte_index();
        if !candidates.iter().any(|(_, _, remove)| *remove) {
            return Ok(candidates
                .into_iter()
                .flat_map(|(_, edits, _)| edits)
                .collect());
        }
        let retained: Vec<_> = candidates
            .into_iter()
            .filter(|(_, _, remove)| !remove)
            .collect();
        if retained.is_empty() {
            return Ok(vec![Edit {
                range: function_start..function_end,
                value: "url(\"\")".into(),
            }]);
        }
        let mut value = self.source[function_start..content_start].to_owned();
        for (index, (range, edits, _)) in retained.into_iter().enumerate() {
            if index > 0 {
                value.push(',');
            }
            let mut cursor = range.start;
            for edit in edits {
                value.push_str(&self.source[cursor..edit.range.start]);
                value.push_str(&edit.value);
                cursor = edit.range.end;
            }
            value.push_str(&self.source[cursor..range.end]);
        }
        value.push(')');
        Ok(vec![Edit {
            range: function_start..function_end,
            value,
        }])
    }
}

impl<'i> DeclarationParser<'i> for Reader<'i> {
    type Declaration = Vec<Edit>;
    type Error = ();
    fn parse_value(
        &mut self,
        name: CowRcStr<'i>,
        parser: &mut Parser<'i>,
        _: &ParserState,
    ) -> Result<Vec<Edit>, ParseError<()>> {
        self.values(parser, self.depth, name.starts_with("--"))
    }
}

impl<'i> QualifiedRuleParser<'i> for Reader<'i> {
    type Prelude = ();
    type QualifiedRule = Vec<Edit>;
    type Error = ();
    fn parse_prelude(&mut self, parser: &mut Parser<'i>) -> Result<(), ParseError<()>> {
        self.imports_allowed = false;
        consume(parser);
        Ok(())
    }
    fn parse_block(
        &mut self,
        _: (),
        _: &ParserState,
        parser: &mut Parser<'i>,
    ) -> Result<Vec<Edit>, ParseError<()>> {
        Ok(self.child(Context::Body).read(parser))
    }
}

impl<'i> AtRuleParser<'i> for Reader<'i> {
    type Prelude = Prelude;
    type AtRule = Vec<Edit>;
    type Error = ();
    fn parse_prelude(
        &mut self,
        name: CowRcStr<'i>,
        parser: &mut Parser<'i>,
    ) -> Result<Prelude, ParseError<()>> {
        if self.context == Context::Inline {
            return Err(ParseError::from_basic_kind(
                BasicParseErrorKind::AtRuleInvalid,
            ));
        }
        if name.eq_ignore_ascii_case("import") {
            if !self.imports_allowed {
                return Err(ParseError::custom(()));
            }
            parser.skip_whitespace();
            let start = parser.position().byte_index();
            let token = parser.next()?.clone();
            let (edit, _) = match token {
                Token::QuotedString(value) => {
                    self.string_resource(start, parser.position().byte_index(), &value)
                }
                Token::UnquotedUrl(value) => {
                    let end = parser.position().byte_index();
                    let (edit, remove) = self.unquoted_resource(start, end, &value);
                    (
                        if remove {
                            Some(Edit {
                                range: start..end,
                                value: "url(\"\")".into(),
                            })
                        } else {
                            edit
                        },
                        remove,
                    )
                }
                Token::Function(name)
                    if name.eq_ignore_ascii_case("url") || name.eq_ignore_ascii_case("src") =>
                {
                    self.quoted_url(parser)?
                }
                _ => return Err(ParseError::custom(())),
            };
            consume(parser);
            return Ok(Prelude::Import(edit.into_iter().collect()));
        }
        if name.eq_ignore_ascii_case("layer") {
            consume(parser);
            return Ok(Prelude::Layer);
        }
        self.imports_allowed = false;
        let context = if ["font-face", "page", "counter-style", "property"]
            .iter()
            .any(|known| name.eq_ignore_ascii_case(known))
        {
            Context::Body
        } else if [
            "media",
            "supports",
            "container",
            "scope",
            "starting-style",
            "keyframes",
            "-webkit-keyframes",
        ]
        .iter()
        .any(|known| name.eq_ignore_ascii_case(known))
        {
            if self.context == Context::Body {
                Context::Body
            } else {
                Context::Rules
            }
        } else {
            return Err(ParseError::from_basic_kind(
                BasicParseErrorKind::AtRuleInvalid,
            ));
        };
        consume(parser);
        Ok(Prelude::Block(context))
    }
    fn rule_without_block(&mut self, prelude: Prelude, _: &ParserState) -> Result<Vec<Edit>, ()> {
        match prelude {
            Prelude::Import(edits) => Ok(edits),
            Prelude::Layer => Ok(Vec::new()),
            Prelude::Block(_) => Err(()),
        }
    }
    fn parse_block(
        &mut self,
        prelude: Prelude,
        _: &ParserState,
        parser: &mut Parser<'i>,
    ) -> Result<Vec<Edit>, ParseError<()>> {
        self.imports_allowed = false;
        let context = match prelude {
            Prelude::Block(context) => context,
            Prelude::Layer => {
                if self.context == Context::Body {
                    Context::Body
                } else {
                    Context::Rules
                }
            }
            Prelude::Import(_) => return Err(ParseError::custom(())),
        };
        Ok(self.child(context).read(parser))
    }
}

impl<'i> RuleBodyItemParser<'i, Vec<Edit>, ()> for Reader<'i> {
    fn parse_declarations(&self) -> bool {
        true
    }
    fn parse_qualified(&self) -> bool {
        self.context != Context::Inline
    }
}

fn file_path(value: &str) -> String {
    uri_path(value, true)
}

fn escaped_last_character(value: &str) -> bool {
    value.as_bytes()[..value.len() - 1]
        .iter()
        .rev()
        .take_while(|byte| **byte == b'\\')
        .count()
        % 2
        == 1
}

fn uri_path(value: &str, escape_percent: bool) -> String {
    let mut output = String::with_capacity(value.len());
    for character in value.chars() {
        if character.is_ascii()
            && (character.is_ascii_control()
                || character.is_ascii_whitespace()
                || matches!(
                    character,
                    '\'' | '"' | '(' | ')' | '\\' | '<' | '>' | '#' | '?'
                )
                || escape_percent && character == '%')
        {
            use std::fmt::Write;
            write!(output, "%{:02X}", character as u32).unwrap();
        } else {
            output.push(character);
        }
    }
    output
}

/// CSS escape decoding is only needed to locate the original suffix span.
/// Token validity and the decoded URL itself come from cssparser.
fn raw_suffix(value: &str) -> usize {
    let mut index = 0;
    while index < value.len() {
        let character = value[index..].chars().next().unwrap();
        if matches!(character, '?' | '#') {
            return index;
        }
        if character != '\\' {
            index += character.len_utf8();
            continue;
        }
        let start = index;
        index += 1;
        let mut count = 0;
        let mut number = 0u32;
        while count < 6 {
            let Some(byte) = value.as_bytes().get(index) else {
                break;
            };
            let Some(digit) = (*byte as char).to_digit(16) else {
                break;
            };
            number = number * 16 + digit;
            index += 1;
            count += 1;
        }
        if count > 0 {
            if matches!(number, 0x23 | 0x3f) {
                return start;
            }
            if value[index..].starts_with("\r\n") {
                index += 2;
            } else if value[index..].starts_with(html_space) {
                index += 1;
            }
        } else if let Some(character) = value[index..].chars().next() {
            if matches!(character, '?' | '#') {
                return start;
            }
            index += character.len_utf8();
        }
    }
    value.len()
}

#[cfg(test)]
mod tests {
    use super::super::{has_image_references, rewrite_asset_references, rewrite_asset_target};
    use std::collections::HashMap;

    fn rewrite(source: &str) -> String {
        rewrite_asset_target(source, "assets/a.png", "assets/b.png")
    }

    #[test]
    fn attributes_and_stylesheets_share_resource_mapping_with_other_html_attributes() {
        let source = r#"<img src="assets/a.png" style="background:url(assets/a.png)">
<x-picture STYLE='mask:URL("assets/a.png")' data-style="url(assets/a.png)"></x-picture>
<style>@media screen { .x { background: url( assets/a.png ); } }</style>"#;
        let expected = r#"<img src="assets/b.png" style="background:url(assets/b.png)">
<x-picture STYLE='mask:URL("assets/b.png")' data-style="url(assets/a.png)"></x-picture>
<style>@media screen { .x { background: url( assets/b.png ); } }</style>"#;
        assert_eq!(rewrite(source), expected);
    }

    #[test]
    fn css_escapes_entities_and_uri_suffixes_each_decode_at_their_own_layer() {
        let source = r#"<div style="background:u\72l(&quot;assets/&#92;61.png?\76=1&amp;x=2#\70 review&quot;)"></div>
<style>.x{background:u\000072 l('assets/\61.png?\76=1&amp;x=2#\70 review')}</style>"#;
        let expected = r#"<div style="background:u\72l(&quot;assets/b.png?\76=1&amp;x=2#\70 review&quot;)"></div>
<style>.x{background:u\000072 l('assets/b.png?\76=1&amp;x=2#\70 review')}</style>"#;
        assert_eq!(rewrite(source), expected);
        let mapping = HashMap::from([
            ("assets/a#b?.png".into(), "assets/new#?.png".into()),
            ("assets/a&amp;b.png".into(), "assets/raw.png".into()),
            ("assets/a&num;b.png".into(), "assets/once.png".into()),
        ]);
        let source = r#"<style>.x{a:url('assets/a%23b%3F.png\23 frag');b:url('assets/a&amp;b.png')}</style>
<div style="a:url('assets/a&amp;num;b.png')"></div>"#;
        let expected = r#"<style>.x{a:url('assets/new%23%3F.png\23 frag');b:url('assets/raw.png')}</style>
<div style="a:url('assets/once.png')"></div>"#;
        assert_eq!(rewrite_asset_references(source, &mapping), expected);
    }

    #[test]
    fn identity_spellings_and_original_map_destinations_do_not_cascade() {
        let source = r#"<style>.x{a:url('assets/\61.png');b:image-set("assets/b.png" 1x)}</style>"#;
        let identity = HashMap::from([("assets/a.png".into(), "assets/a.png".into())]);
        assert_eq!(rewrite_asset_references(source, &identity), source);
        let mapping = HashMap::from([
            ("assets/a.png".into(), "assets/b.png".into()),
            ("assets/b.png".into(), "assets/c.png".into()),
        ]);
        assert_eq!(
            rewrite_asset_references(source, &mapping),
            r#"<style>.x{a:url('assets/b.png');b:image-set("assets/c.png" 1x)}</style>"#
        );
    }

    #[test]
    fn imports_and_image_set_strings_are_resource_positions_but_namespace_and_labels_are_not() {
        let source = r#"<style>
@import "assets/a.png" layer(theme) screen;
@namespace icon url(assets/a.png);
[data-src="assets/a.png"] { content:"assets/a.png"; --example:@import "assets/a.png";
background:image-set("assets/a.png" 1x type("assets/a.png"), url(assets/a.png) 2x);
mask:-webkit-image-set(url('assets/a.png') 1x, "assets/a.png" 2x);
src:local("assets/a.png"),url(assets/a.png) format("assets/a.png"); }
@import "assets/a.png";
</style>"#;
        let expected = r#"<style>
@import "assets/b.png" layer(theme) screen;
@namespace icon url(assets/a.png);
[data-src="assets/a.png"] { content:"assets/a.png"; --example:@import "assets/a.png";
background:image-set("assets/b.png" 1x type("assets/a.png"), url(assets/b.png) 2x);
mask:-webkit-image-set(url('assets/b.png') 1x, "assets/b.png" 2x);
src:local("assets/a.png"),url(assets/b.png) format("assets/a.png"); }
@import "assets/a.png";
</style>"#;
        assert_eq!(rewrite(source), expected);
    }

    #[test]
    fn filtering_preserves_url_grammar_fallbacks_and_import_layer_order() {
        let source = r#"<style>@import "assets/a.png" layer(theme) screen;
@font-face{src:url(assets/a.png),local("Fixture")}
.x{cursor:url('assets/a.png'),auto;background:cross-fade(url(assets/a.png) 50%,url(other.png) 50%)}</style>"#;
        let expected = r#"<style>@import "" layer(theme) screen;
@font-face{src:url(""),local("Fixture")}
.x{cursor:url(''),auto;background:cross-fade(url("") 50%,url(other.png) 50%)}</style>"#;
        assert_eq!(rewrite_asset_target(source, "assets/a.png", ""), expected);
        let source = r#"<style>.x{a:image-set("assets/a.png" 1x, "assets/keep.png" 2x);b:image-set(url(assets/a.png) 1x)}</style>"#;
        let output = rewrite_asset_target(source, "assets/a.png", "");
        assert_eq!(
            output,
            r#"<style>.x{a:image-set( "assets/keep.png" 2x);b:url("")}</style>"#
        );
    }

    #[test]
    fn static_src_and_image_strings_keep_fallback_colors_and_ignore_dynamic_values() {
        let source = r#"<style>.x{a:src("assets/a.png");b:image(ltr "assets/a.png",red);c:image(url(assets/a.png),blue);
d:src(var(--file));e:image(var(--file),red);f:content("assets/a.png")}</style>"#;
        let expected = r#"<style>.x{a:src("assets/b.png");b:image(ltr "assets/b.png",red);c:image(url(assets/b.png),blue);
d:src(var(--file));e:image(var(--file),red);f:content("assets/a.png")}</style>"#;
        assert_eq!(rewrite(source), expected);
        assert_eq!(
            rewrite_asset_target(
                r#"<style>.x{a:image("assets/a.png",red);b:image(url(assets/a.png),blue);c:src("assets/a.png")}</style>"#,
                "assets/a.png",
                ""
            ),
            r#"<style>.x{a:image("",red);b:image(url(""),blue);c:src("")}</style>"#
        );
    }

    #[test]
    fn style_type_duplicate_attributes_and_unquoted_html_boundaries_are_respected() {
        for source in [
            r#"<style type="text/plain">.x{background:url(assets/a.png)}</style>"#,
            r#"<style type="text/css; charset=utf-8">.x{background:url(assets/a.png)}</style>"#,
            r#"<style type=" text/css ">.x{background:url(assets/a.png)}</style>"#,
            r#"<style type=" ">.x{background:url(assets/a.png)}</style>"#,
            r#"<div style STYLE="background:url(assets/a.png)"></div>"#,
            r#"<div style="" style="background:url(assets/a.png)"></div>"#,
        ] {
            assert_eq!(rewrite(source), source);
        }
        let source = r#"<div style=background:url(assets/a.png)></div><style type TYPE="text/plain">.x{a:url(assets/a.png)}</style>"#;
        assert_eq!(
            rewrite_asset_target(source, "assets/a.png", "assets/new name.png"),
            r#"<div style=background:url(assets/new%20name.png)></div><style type TYPE="text/plain">.x{a:url(assets/new%20name.png)}</style>"#
        );
    }

    #[test]
    fn html_raw_text_closes_even_inside_css_strings_and_accepts_end_tag_whitespace() {
        let source = r#"<style/>.x{a:url(assets/a.png)}</StYlE ><img src="assets/a.png">
<style>.x{content:"</style/>";background:url(assets/a.png)}<img src="assets/a.png">"#;
        let expected = r#"<style/>.x{a:url(assets/b.png)}</StYlE ><img src="assets/b.png">
<style>.x{content:"</style/>";background:url(assets/a.png)}<img src="assets/b.png">"#;
        assert_eq!(rewrite(source), expected);
    }

    #[test]
    fn code_comments_plain_strings_and_unknown_css_contexts_remain_literal() {
        let source = r#"---
example: |
  <style>.x{a:url(assets/a.png)}</style>
---
`<div style="a:url(assets/a.png)">`
~~~css
@import "assets/a.png";
~~~
    <style>.x{a:url(assets/a.png)}</style>
<pre><div style="a:url(assets/a.png)"></div></pre>
<code><style>.x{a:url(assets/a.png)}</style></code>
<script>let example='<style>.x{a:url(assets/a.png)}</style>'</script>
<!-- <div style="a:url(assets/a.png)"> -->
<style>/* url(assets/a.png) */ .x{content:"url(assets/a.png)";a:url /**/(assets/a.png);b:url('bad
assets/a.png') } @unknown url(assets/a.png) {.x{a:url(assets/a.png)}}</style>"#;
        assert_eq!(rewrite(source), source);
    }

    #[test]
    fn filesystem_characters_cannot_escape_css_or_html_raw_text() {
        let path = "assets/</style>\"'()%#?\\\t\n.png ";
        let source = r##"<style>.x{a:url('assets/a.png?x="quoted"#f')}</style><div style="a:url('assets/a.png?x=&quot;quoted&quot;#f')"></div>"##;
        let output = rewrite_asset_target(source, "assets/a.png", path);
        let encoded = "assets/%3C/style%3E%22%27%28%29%25%23%3F%5C%09%0A.png%20";
        assert_eq!(
            output,
            format!(
                r##"<style>.x{{a:url('{encoded}?x="quoted"#f')}}</style><div style="a:url('{encoded}?x=&quot;quoted&quot;#f')"></div>"##
            )
        );
    }

    #[test]
    fn css_fonts_imports_and_backgrounds_do_not_trigger_image_enrichment() {
        for source in [
            r#"<div style="background:url(assets/a.png)"></div>"#,
            r#"<style>@import "assets/a.png";@font-face{src:url(assets/a.png)}.x{background:image-set("assets/a.png" 1x)}</style>"#,
        ] {
            assert!(!has_image_references(source));
        }
        assert!(has_image_references(
            r#"<img src="assets/a.png" style="background:url(assets/a.png)">"#
        ));
    }

    #[test]
    fn excessive_function_nesting_stays_bounded_without_rewriting_hidden_leaf() {
        let value = format!(
            "{}\"assets/a.png\" 1x{}",
            "image-set(".repeat(100),
            ") 1x".repeat(99) + ")"
        );
        let source = format!("<style>.x{{background:{value}}}</style>");
        assert_eq!(rewrite(&source), source);
    }

    #[test]
    fn literal_html_elements_rewrite_only_their_actual_opening_style_attribute() {
        for tag in ["pre", "code", "script"] {
            let source = format!(
                "<{tag} style='background:url(assets/a.png)'><div style='background:url(assets/a.png)'>literal</div></{tag}>"
            );
            let expected = format!(
                "<{tag} style='background:url(assets/b.png)'><div style='background:url(assets/a.png)'>literal</div></{tag}>"
            );
            assert_eq!(rewrite(&source), expected);
        }
        let source = r#"<style type="text/plain" style="background:url(assets/a.png)">url(assets/a.png)</style>"#;
        assert_eq!(
            rewrite(source),
            r#"<style type="text/plain" style="background:url(assets/b.png)">url(assets/a.png)</style>"#
        );
    }

    #[test]
    fn escaped_end_characters_do_not_repair_truncated_url_tokens() {
        let source = r#"<style>.x{a:url(assets/a\)"#;
        assert_eq!(
            rewrite_asset_target(source, "assets/a)", "assets/new.png"),
            source
        );
    }

    #[test]
    fn literal_closing_tags_inside_opening_attributes_cannot_end_the_body() {
        for tag in ["pre", "code", "script"] {
            let source = format!(
                "<{tag} title='</{tag}>' style='background:url(assets/a.png)'>literal url(assets/a.png)</{tag} ><span style='background:url(assets/a.png)'>after</span>"
            );
            let expected = format!(
                "<{tag} title='</{tag}>' style='background:url(assets/b.png)'>literal url(assets/a.png)</{tag} ><span style='background:url(assets/b.png)'>after</span>"
            );
            assert_eq!(rewrite(&source), expected);
        }
        let source = r#"<style title="</style>">.x{background:url(assets/a.png)}</STYLE/><img src="assets/a.png">"#;
        assert_eq!(
            rewrite(source),
            r#"<style title="</style>">.x{background:url(assets/b.png)}</STYLE/><img src="assets/b.png">"#
        );
    }
}
