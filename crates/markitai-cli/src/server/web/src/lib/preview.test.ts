import assert from "node:assert/strict";
import test from "node:test";
import { entryAt, tocEntries } from "./toc.ts";
import { compareLines, foldRows } from "./diff.ts";
import { classifyTestText, testFailure } from "./model-test.ts";
import { countTableRows, emptyPlan, extendPlan, planTokens, splitMarkdown, type Token } from "./markdown.ts";
// The vendored parser the page loads as /ui/marked.js.
import { marked } from "../../../../../../../vendor/web/marked.js";

const lex = (markdown: string) => marked.lexer(markdown, { gfm: true }) as unknown as Token[] & { links?: unknown };
const html = (markdown: string) => marked.parse(markdown, { async: false, gfm: true }) as string;

/** A long document with everything a cut could break: fences and raw HTML with blank lines, loose lists, quotes, tables, indented code. */
function sample(sections = 60): string {
  const parts: string[] = [];
  for (let index = 1; index <= sections; index++) {
    parts.push(`## Section ${index}`, "", "长段落".repeat(40) + ` paragraph ${index}.`, "");
    if (index % 5 === 0) parts.push("```js", "const a = 1;", "", "", "const b = 2;", "```", "");
    if (index % 7 === 0) parts.push("1. first", "", "2. second", "", "   continued under the second item", "", "3. third", "");
    if (index % 9 === 0) parts.push("> quoted", "", "> still the same quote", "");
    if (index % 11 === 0) parts.push("| a | b |", "| - | -: |", "| 1 | 2 |", "", "    indented code", "", "    more of it", "");
    if (index % 13 === 0) parts.push("<!-- a comment", "", "that spans a blank line -->", "", "<pre>", "raw", "", "pre", "</pre>", "");
    if (index % 4 === 0) parts.push("Setext title", "------------", "");
  }
  return parts.join("\n");
}

test("a long document is cut only where every piece reads the same alone", () => {
  const markdown = sample();
  const pieces = splitMarkdown(markdown, { first: 600, size: 1200 });
  assert.ok(pieces.length > 8, `${pieces.length} pieces`);
  assert.equal(pieces.join(""), markdown);
  assert.equal(pieces.map(html).join(""), html(markdown));
  for (const piece of pieces.slice(1)) assert.match(piece, /^[^\s>*+\-\d|]/);
  // Reference definitions belong to the whole document: it is read at once.
  const referenced = `${markdown}\n[ref]: https://example.com\n`;
  assert.deepEqual(splitMarkdown(referenced, { first: 600, size: 1200 }), [referenced]);
  assert.deepEqual(splitMarkdown("short", { first: 600 }), ["short"]);
});

test("the plan cuts chunks near their size, before chapters, and lists the headings by chunk", () => {
  const markdown = sample(30);
  const plan = planTokens(lex(markdown), { first: 400, size: 1500 });
  assert.equal(plan.chunks.flat().map((token) => token.raw).join(""), markdown);
  assert.equal(plan.headings.filter((heading) => heading.depth === 2).length, 30 + 7);
  assert.ok(plan.chunks.length > 5);
  for (const [index, chunk] of plan.chunks.entries()) {
    assert.equal(plan.sizes[index], chunk.reduce((sum, token) => sum + token.raw.length, 0));
    // A chunk after the first starts at a chapter heading or once the previous one was full.
    if (index > 0) assert.ok(chunk[0]?.type === "heading" || (plan.sizes[index - 1] ?? 0) >= 1500);
  }
  for (const heading of plan.headings) assert.ok(plan.chunks[heading.chunk]?.some((token) => token.type === "heading" && token.raw.includes(heading.text)));
  assert.equal(plan.tables, countTableRows(markdown));
  // Read in pieces, the plan holds the same blocks and headings.
  const pieced = emptyPlan(countTableRows(markdown));
  for (const piece of splitMarkdown(markdown, { first: 400, size: 1500 })) extendPlan(pieced, lex(piece), { first: 400, size: 1500 });
  assert.deepEqual(pieced.headings.map((heading) => heading.text), plan.headings.map((heading) => heading.text));
  assert.equal(pieced.chunks.flat().map((token) => token.raw).join(""), markdown);
});

test("tables are counted from their delimiter rows outside code", () => {
  assert.equal(countTableRows("| a |\n| - |\n| 1 |\n\n```\n| x |\n|---|\n```\n\na | b\n:-- | --:\n"), 2);
  assert.equal(countTableRows("text\n---\n"), 0);
});

test("the diff stays exact on a long document with scattered edits and folds the rest", () => {
  const base = Array.from({ length: 3000 }, (_, index) => `line ${index}`);
  const llm = base.map((line, index) => (index % 97 === 5 ? `${line} (edited)` : line)).filter((_, index) => index % 401 !== 7);
  llm.splice(1500, 0, "inserted");
  const result = compareLines(base.join("\n"), llm.join("\n"));
  assert.equal(result.kind, "rows");
  if (result.kind !== "rows") return;
  // Both texts come back from the rows, with their own line numbers.
  assert.deepEqual(result.rows.filter((row) => row.type !== "add").map((row) => row.text), base);
  assert.deepEqual(result.rows.filter((row) => row.type !== "del").map((row) => row.text), llm);
  assert.deepEqual(result.rows.filter((row) => row.aNo !== null).map((row) => row.aNo), base.map((_, index) => index + 1));
  const edited = base.filter((_, index) => index % 97 === 5).length;
  const dropped = base.filter((_, index) => index % 401 === 7).length;
  assert.equal(result.removed, edited + dropped - base.filter((_, index) => index % 97 === 5 && index % 401 === 7).length);
  assert.equal(result.added, result.removed - dropped + 1);
  const segments = foldRows(result.rows);
  const shown = segments.filter((segment) => segment.kind === "rows").reduce((sum, segment) => sum + segment.rows.length, 0);
  assert.ok(shown < 600);
  assert.equal(segments.reduce((sum, segment) => sum + segment.rows.length, 0), result.rows.length);
  assert.deepEqual(foldRows(compareLines("a\nb", "a\nb").kind === "rows" ? [] : []), []);
  const same = compareLines(base.join("\n"), base.join("\n"));
  assert.equal(same.kind === "rows" && foldRows(same.rows).map((segment) => segment.kind).join(), "fold");
});

test("contents list the top three levels, skip a lone title, and follow the heading being read", () => {
  const headings = [
    { depth: 1, text: "Book", chunk: 0 },
    { depth: 2, text: "One", chunk: 0 },
    { depth: 3, text: "One.a", chunk: 0 },
    { depth: 2, text: "Two", chunk: 1 },
    { depth: 4, text: "Two.a.i", chunk: 1 },
    { depth: 5, text: "deep", chunk: 1 },
    { depth: 2, text: "", chunk: 2 },
    { depth: 2, text: "Three", chunk: 2 },
  ];
  const entries = tocEntries(headings);
  assert.deepEqual(entries.map((entry) => [entry.text, entry.level]), [["One", 0], ["One.a", 1], ["Two", 0], ["Two.a.i", 2], ["Three", 0]]);
  assert.equal(entryAt(entries, 5), 3);
  assert.equal(entryAt(entries, 0), -1);
  assert.deepEqual(tocEntries(headings.slice(0, 3)), []);
});

test("a failed model test is named by its category, from the server's reason or from its text", () => {
  // The server says it (Rust, contract v1.10).
  assert.deepEqual(testFailure({ detail: "DEEPSEEK_API_KEY is not set on the server", reason: "no_key", key_variable: "DEEPSEEK_API_KEY" }), {
    reason: "no_key",
    variable: "DEEPSEEK_API_KEY",
    detail: "DEEPSEEK_API_KEY is not set on the server",
  });
  assert.equal(testFailure({ detail: "Model connection returned HTTP 500 (server_error)", reason: "provider_error" }).reason, "provider_error");
  // An unknown reason falls back to the text.
  assert.equal(testFailure({ detail: "RateLimitError: x", reason: "something_new" }).reason, "rate_limited");
  // Python 1.2.1's LiteLLM texts, the owner's DeepSeek case first.
  const cases: [string, string][] = [
    ["AuthenticationError: litellm.AuthenticationError: AuthenticationError: DeepseekException - Authentication Fails (governor)", "key_rejected"],
    ['BadRequestError: litellm.BadRequestError: DeepseekException - {"error": {"message": "Authentication Fails (governor)"}}', "key_rejected"],
    ["NotFoundError: litellm.NotFoundError: OpenAIException - Model Not Exist", "model_not_found"],
    ["RateLimitError: litellm.RateLimitError: RateLimitError: OpenAIException - Rate limit reached for requests", "rate_limited"],
    ["InternalServerError: litellm.InternalServerError: InternalServerError: OpenAIException - Connection error.", "network"],
    ["Timeout: litellm.Timeout: Connection timed out after 30.0 seconds.", "timeout"],
    ["AuthenticationError: litellm.AuthenticationError: Missing API key. Set the GEMINI_API_KEY environment variable.", "no_key"],
    ["OPENAI_API_KEY is not set; set it with: export OPENAI_API_KEY=your_key", "no_key"],
    ["APIError: litellm.APIError: something else went wrong", "provider_error"],
  ];
  for (const [text, reason] of cases) assert.equal(classifyTestText(text), reason, text);
  assert.equal(testFailure({ detail: cases[7]?.[0] ?? "" }).variable, "OPENAI_API_KEY");
  assert.equal(testFailure({ detail: cases[0]?.[0] ?? "" }).variable, null);
});
