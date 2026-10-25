import assert from "node:assert/strict";
import test from "node:test";
import { docsPage } from "./docs.ts";

test("an index.md docs address maps to the page beside it (Rust before 2.1.0)", () => {
  const rust = "https://github.com/Ynewtime/markitai/blob/main/docs/index.md";
  assert.equal(docsPage(rust, "pricing"), "https://github.com/Ynewtime/markitai/blob/main/docs/pricing.md");
  assert.equal(docsPage(rust, "subscription-chatgpt"), "https://github.com/Ynewtime/markitai/blob/main/docs/subscription-chatgpt.md");
});

test("a site root, with or without its slash, maps to its guide pages (markitai.dev)", () => {
  assert.equal(docsPage("https://markitai.dev", "pricing"), "https://markitai.dev/guide/pricing");
  assert.equal(docsPage("https://markitai.dev/", "pricing"), "https://markitai.dev/guide/pricing");
  assert.equal(docsPage("https://markitai.dev/", "subscriptions"), "https://markitai.dev/guide/subscriptions");
  assert.equal(docsPage("https://example.test/docs/", "pricing"), "https://example.test/docs/guide/pricing");
});

test("any other docs address is kept as it is", () => {
  assert.equal(docsPage("https://example.test/manual", "pricing"), "https://example.test/manual");
  assert.equal(docsPage("https://example.test/?page=home", "pricing"), "https://example.test/?page=home");
});
