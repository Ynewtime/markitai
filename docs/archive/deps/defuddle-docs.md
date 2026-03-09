---
title: Installation
source: https://defuddle.md/docs
description: Defuddle 文档概览：安装、浏览器/Node.js/CLI用法、选项、返回字段及HTML标准化规则。
tags:
- Defuddle
- Node.js
- CLI
- HTML-parsing
- Markdown
markitai_processed: '2026-03-06T00:31:10'
---

Defuddle extracts the main content from web pages, removing clutter like comments, sidebars, headers, and footers to return clean, readable HTML.

API: `curl defuddle.md/stephango.com`

-> Returns Markdown with YAML frontmatter. Append any URL path to convert it.

- [Installation](#installation)
- [Browser use](#browser-use)
- [Node.js use](#nodejs-use)
- [CLI use](#cli-use)
  - [CLI options](#cli-options)
- [Options](#options)
  - [Debug mode](#debug-mode)
- [Response](#response)
- [Bundles](#bundles)
- [HTML standardization](#html-standardization)
  - [Headings](#headings)
  - [Code blocks](#code-blocks)
  - [Footnotes](#footnotes)
  - [Math](#math)

## Installation

```
npm install defuddle
```

For Node.js use, you also need JSDOM:

```
npm install defuddle jsdom
```

## Browser use

In the browser, create a Defuddle instance with a `Document` object and call `parse()`.

```
import Defuddle from 'defuddle';

const result = new Defuddle(document).parse();

console.log(result.content); // cleaned HTML string
console.log(result.title); // page title
console.log(result.author); // author name
```

You can also parse HTML strings using `DOMParser`:

```
const parser = new DOMParser();
const doc = parser.parseFromString(htmlString, 'text/html');
const result = new Defuddle(doc).parse();
```

Pass options as the second argument:

```
const result = new Defuddle(document, {
 url: 'https://example.com/article',
 debug: true
}).parse();
```

## Node.js use

The Node.js API accepts an HTML string or a JSDOM instance and returns a promise.

```
import { Defuddle } from 'defuddle/node';

// From an HTML string
const result = await Defuddle(htmlString);

// From a JSDOM instance
import { JSDOM } from 'jsdom';
const dom = await JSDOM.fromURL('https://example.com/article');
const result = await Defuddle(dom);

// With URL and options
const result = await Defuddle(dom, 'https://example.com/article', {
 markdown: true,
 debug: true
});
```

**Note:** For `defuddle/node` to import properly, your `package.json` must have `"type": "module"`.

## CLI use

Defuddle includes a CLI for parsing web pages from the terminal.

```
# Parse a local HTML file
defuddle parse page.html

# Parse a URL
defuddle parse https://example.com/article

# Output as markdown
defuddle parse page.html --markdown

# Output as JSON with metadata
defuddle parse page.html --json

# Extract a specific property
defuddle parse page.html --property title

# Save output to a file
defuddle parse page.html --output result.html
```

### CLI options

| Option | Alias | Description |
| --- | --- | --- |
| `--output <file>` | `-o` | Write output to a file instead of stdout |
| `--markdown` | `-m` | Convert content to markdown |
| `--md` | | Alias for `--markdown` |
| `--json` | `-j` | Output as JSON with metadata and content |
| `--property <name>` | `-p` | Extract a specific property |
| `--debug` | | Enable debug mode |

## Options

Options can be passed when creating a Defuddle instance (browser) or as the third argument (Node.js).

| Option | Type | Default | Description |
| --- | --- | --- | --- |
| `url` | string | | URL of the page being parsed |
| `markdown` | boolean | false | Convert `content` to Markdown |
| `separateMarkdown` | boolean | false | Keep `content` as HTML and return Markdown in `contentMarkdown` |
| `removeExactSelectors` | boolean | true | Remove elements matching exact selectors (ads, social buttons, etc.) |
| `removePartialSelectors` | boolean | true | Remove elements matching partial selectors |
| `removeImages` | boolean | false | Remove images from the output |
| `debug` | boolean | false | Enable debug logging |

### Debug mode

When debug mode is enabled:

* More verbose console logging about the parsing process
* Preserves HTML class and id attributes that are normally stripped
* Retains all `data-*` attributes
* Skips div flattening to preserve document structure

## Response

The `parse()` method returns an object with the following properties:

| Property | Type | Description |
| --- | --- | --- |
| `content` | string | Cleaned HTML string of the extracted content |
| `contentMarkdown` | string | Markdown version (when `separateMarkdown` is true) |
| `title` | string | Title of the article |
| `description` | string | Description or summary |
| `author` | string | Author of the article |
| `site` | string | Name of the website |
| `domain` | string | Domain name of the website |
| `favicon` | string | URL of the website's favicon |
| `image` | string | URL of the article's main image |
| `published` | string | Publication date |
| `wordCount` | number | Number of words in the extracted content |
| `parseTime` | number | Time taken to parse in milliseconds |
| `metaTags` | object[] | Meta tags from the page |
| `schemaOrgData` | object | Schema.org data extracted from the page |
| `extractorType` | string | Type of site-specific extractor used, if any |

## Bundles

Defuddle is available in three bundles:

| Bundle | Import | Description |
| --- | --- | --- |
| Core | `defuddle` | Browser usage. No dependencies. Handles math content but without MathML/LaTeX conversion fallbacks. |
| Full | `defuddle/full` | Includes math equation parsing (MathML ↔ LaTeX) and Markdown conversion via Turndown. |
| Node.js | `defuddle/node` | For Node.js with JSDOM. Includes full capabilities for math and Markdown conversion. |

The core bundle is recommended for most use cases.

## HTML standardization

Defuddle standardizes HTML elements to provide a consistent input for downstream tools like Markdown converters.

### Headings

* The first H1 or H2 is removed if it matches the title.
* H1s are converted to H2s.
* Anchor links in headings are removed.

### Code blocks

Code blocks are standardized. Line numbers and syntax highlighting are removed, but the language is retained.

```
<pre>
 <code data-lang="js" class="language-js">
 // code
 </code>
</pre>
```

### Footnotes

Inline references and footnotes are converted to a standard format using `sup`, `a`, and an ordered list with `class="footnote"`.

### Math

Math elements, including MathJax and KaTeX, are converted to standard MathML with a `data-latex` attribute containing the original LaTeX source.
