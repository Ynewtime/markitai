// Formulas in Markdown: what is a formula and what stays text, against the vendored parser the page loads.
import assert from "node:assert/strict";
import test from "node:test";
// The vendored parser the page loads as /ui/marked.js.
import { marked } from "../../../../../../../vendor/web/marked.js";
import { mathExtensions, splitMath } from "./math-syntax.ts";

marked.use({ extensions: mathExtensions });
const html = (markdown: string) => marked.parse(markdown, { async: false, gfm: true }).trim();

test("inline and display formulas become .math elements holding their source", () => {
  assert.equal(html("When $a \\neq 0$ and $x_1 + y_1$ hold"), '<p>When <span class="math">$a \\neq 0$</span> and <span class="math">$x_1 + y_1$</span> hold</p>');
  assert.equal(html("$$x = \\frac{1}{2}$$"), '<div class="math math-display">$$x = \\frac{1}{2}$$</div>');
  assert.equal(html("text $$a_b$$ text"), '<p>text <span class="math math-display">$$a_b$$</span> text</p>');
});

test("a display may span lines, blank lines and an aligned environment", () => {
  const out = html("$$\n\\begin{aligned}a &= b \\\\\n\nc &= d\\end{aligned}\n$$\n\nafter");
  assert.match(out, /^<div class="math math-display">\$\$\\begin\{aligned\}a &amp;= b \\\\\n\nc &amp;= d\\end\{aligned\}\$\$<\/div>\n<p>after<\/p>$/);
});

test("underscores and stars inside a formula are not emphasis", () => {
  assert.ok(!/<em>|<strong>/.test(html("$a_b$ and $c_d$ and $2*3*4$")));
});

test("prices, code and escaped dollars stay text", () => {
  assert.equal(html("It costs $5 and $6 today"), "<p>It costs $5 and $6 today</p>");
  assert.equal(html("a `$x$` b"), "<p>a <code>$x$</code> b</p>");
  assert.equal(html("a \\$x\\$ b"), "<p>a $x$ b</p>");
  assert.equal(html("```\n$$ not math $$\n```"), "<pre><code>$$ not math $$\n</code></pre>");
  assert.equal(html("$ x $ and $$"), "<p>$ x $ and $$</p>");
});

test("markup in a formula is escaped, never passed on", () => {
  assert.equal(html("$a<b>c$"), '<p><span class="math">$a&lt;b&gt;c$</span></p>');
});

test("a heading's label keeps its formulas for the contents", () => {
  const heading = marked.lexer("## Getting $k$ heads in $n$ flips", { ...marked.defaults, gfm: true })[0] as { tokens: { type: string; raw: string; text?: string }[] };
  assert.deepEqual(heading.tokens.map((token) => token.raw), ["Getting ", "$k$", " heads in ", "$n$", " flips"]);
  assert.deepEqual(splitMath("Getting $k$ heads in $n$ flips").filter((piece) => piece.math).map((piece) => piece.text), ["$k$", "$n$"]);
  assert.deepEqual(splitMath("plain"), [{ text: "plain", math: false }]);
  assert.deepEqual(splitMath("costs $5 or $6"), [{ text: "costs $5 or $6", math: false }]);
});
