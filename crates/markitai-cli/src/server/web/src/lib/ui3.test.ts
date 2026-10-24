// 2.0.1 owner items (2026-10-24): the properties panel names every key markitai writes and
// leaves out what another row says; Docs and GitHub move to the phone header; a single choice
// of more than three options is a menu button.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { en } from "../i18n/en.ts";
import { ja } from "../i18n/ja.ts";
import { zh } from "../i18n/zh.ts";
import { docsHome } from "./docs.ts";
import { frontRows, frontValue } from "./front-rows.ts";
import { cardFacts, parseFrontMatter } from "./frontmatter.ts";

const read = (relative: string) => readFileSync(new URL(relative, import.meta.url), "utf8");

/** Every front-matter key markitai itself writes (output.rs `prepare`, the readers' metadata a URL
 * result keeps, the LLM, image analysis and the OKF profile). */
const WRITTEN = [
  "title", "source", "authors", "language", "publisher", "date", "description", "tags", "markitai_processed",
  "author", "author_name", "site", "published", "canonical_url", "domain", "word_count", "content_profile",
  "views", "upvotes", "account", "fetch_strategy", "renderer", "browser_ms_used", "source_url",
  "from", "to", "cc", "bcc", "ocr_layer_pages", "unread_pages", "pages", "native_pages",
  "ocr_pages_attempted", "ocr_pages_nonempty", "ocr_pages_blank", "ocr_images_attempted", "ocr_images_nonempty",
  "ocr_images_blank", "ocr_used", "ocr_path", "sheet_count", "table_count", "numbers_tables", "numbers_value_mode",
  "type", "resource", "generated",
];

test("every key markitai writes has a label in each language; other keys stay as written", () => {
  for (const dict of [en, zh, ja]) {
    for (const key of WRITTEN) assert.notEqual(dict.frontKey(key), key, `${key}`);
    assert.equal(dict.frontKey("my_own_field"), "my_own_field");
  }
  assert.equal(zh.frontKey("source_url"), "最终地址");
  assert.equal(en.frontKey("word_count"), "Words");
  assert.equal(ja.frontKey("domain"), "ドメイン");
});

test("markitai's own values are shown by name, the URL option's names for fetch methods", () => {
  assert.equal(frontValue(zh, "fetch_strategy", "static"), "静态");
  assert.equal(frontValue(zh, "renderer", "playwright"), "Playwright");
  assert.equal(frontValue(zh, "content_profile", "social_post"), "社交帖子");
  assert.equal(frontValue(en, "content_profile", "social_post"), "Social post");
  assert.equal(frontValue(ja, "type", "Document"), "ドキュメント");
  assert.equal(frontValue(en, "ocr_path", "paddle"), "PP-OCR");
  assert.equal(frontValue(zh, "ocr_used", "true"), "是");
  // the same word under a key that isn't markitai's enumeration stays as written
  assert.equal(frontValue(zh, "site", "social_post"), "social_post");
  assert.equal(frontValue(zh, "fetch_strategy", "something-new"), "something-new");
});

const X_POST = `---
title: Post by @lumen_dev on X
source: https://x.com/lumen_dev/status/1849000000000000001
author: '@lumen_dev'
site: X (Twitter)
published: '2026-10-23T09:14:00.000Z'
canonical_url: https://x.com/lumen_dev/status/1849000000000000001
description: A short note.
tags:
- release
markitai_processed: '2026-10-24T18:02:41.318+08:00'
fetch_strategy: playwright
author_name: Lumen Dev
content_profile: social_post
domain: x.com
renderer: playwright
source_url: https://x.com/lumen_dev/status/1849000000000000001
word_count: 58
---`;

test("an X post's properties: one address, one fetch method, the name beside the handle", () => {
  const front = cardFacts(parseFrontMatter(X_POST), "# Post by @lumen_dev on X\n")!;
  const rows = frontRows(front).map((row) => [row.key, row.kind === "tags" ? row.value.join(",") : row.value]);
  assert.deepEqual(rows, [
    ["description", "A short note."],
    ["tags", "release"],
    ["source", "https://x.com/lumen_dev/status/1849000000000000001"],
    ["author", "Lumen Dev (@lumen_dev)"],
    ["site", "X (Twitter)"],
    ["published", "2026-10-23T09:14:00.000Z"],
    ["markitai_processed", "2026-10-24T18:02:41.318+08:00"],
    ["fetch_strategy", "playwright"],
    ["content_profile", "social_post"],
    ["word_count", "58"],
  ]);
});

test("addresses that differ stay: a redirect's final URL, another canonical link, another domain", () => {
  const front = parseFrontMatter(`---
source: https://example.org/a
canonical_url: https://www.example.org/articles/a
source_url: https://www.example.org/articles/a
domain: www.example.org
fetch_strategy: cloudflare
renderer: cloudflare
---`)!;
  const keys = frontRows(front).map((row) => row.key);
  // the final address is the canonical link here: one row; the domain is the final address's host
  assert.deepEqual(keys, ["source", "canonical_url", "fetch_strategy"]);
  const moved = parseFrontMatter("---\nsource: https://example.org/a\nsource_url: https://example.net/b\ndomain: cdn.example.com\n---")!;
  assert.deepEqual(frontRows(moved).map((row) => row.key), ["source", "source_url", "domain"]);
  // a readable and a percent-encoded form of one address, and a trailing slash, are the same address
  const encoded = parseFrontMatter("---\nsource: https://example.org/文章/\nsource_url: https://example.org/%E6%96%87%E7%AB%A0\n---")!;
  assert.deepEqual(frontRows(encoded).map((row) => row.key), ["source"]);
  // an author without a handle keeps author_name as its own row
  const plain = parseFrontMatter("---\nauthor: Lumen Dev\nauthor_name: Lumen\n---")!;
  assert.deepEqual(frontRows(plain).map((row) => row.key), ["author", "author_name"]);
});

test("the phone header carries Docs and GitHub; the settings list ends with version · commit · build time", () => {
  const shell = read("../components/shell.tsx");
  const header = shell.slice(shell.indexOf('<header class="phone-top">'), shell.indexOf("</header>"));
  assert.match(header, /href=\{docsUrl\}[^>]*aria-label=\{t\.docsLabel\}/);
  assert.match(header, /href=\{sourceUrl\}[^>]*aria-label="GitHub"/);
  const settings = read("../components/settings-page.tsx");
  const index = settings.slice(settings.indexOf("export function SettingsIndex"));
  assert.ok(!index.includes("docsUrl") && !index.includes('"GitHub"'), "no Docs or GitHub rows in the list");
  assert.match(index, /class="build-commit" href=\{sourceUrl\}/);
  const css = read("../styles/app.css");
  assert.match(css, /\.phone-top \.phone-links \.icon-btn \{\n {4}width: 44px;\n {4}height: 44px;/);
  assert.match(css, /\.build-line\.is-index \{\n {2}font-family: var\(--font-mono\);/);
  // the sidebar keeps Docs and GitHub at its foot
  assert.match(shell, /<External href=\{docsUrl\} label=\{t\.docsLabel\}/);
});

test("the docs open in the page's language where the site has it", () => {
  assert.equal(docsHome("https://markitai.dev", "zh"), "https://markitai.dev/zh/");
  assert.equal(docsHome("https://markitai.dev/", "zh"), "https://markitai.dev/zh/");
  assert.equal(docsHome("https://markitai.dev", "ja"), "https://markitai.dev");
  assert.equal(docsHome("https://markitai.dev", "en"), "https://markitai.dev");
  const rust = "https://github.com/Ynewtime/markitai/blob/main/docs/index.md";
  assert.equal(docsHome(rust, "zh"), rust);
});

test("a single choice of more than three options is a menu button; the phone sizes are the owner's", () => {
  const panel = read("../components/options-panel.tsx");
  assert.match(panel, /if \(props\.choices\.length <= SEGMENTS_MAX\) return <Segments \{\.\.\.props\} \/>;/);
  for (const row of ["preset", "output", "strategy", "backend"]) assert.match(panel, new RegExp(`<Choice\\n\\s+name=\\{t\\.\\w+\\}\\n\\s+labelledBy=\\{\`\\$\\{id\\}-${row}\`\\}`));
  assert.ok(!/<Segments\n/.test(panel), "every option group goes through Choice");
  const picker = read("../components/provider-picker.tsx");
  assert.match(picker, /view\.filters\.length > SEGMENTS_MAX \? \(\n\s+<ChoiceMenu/);
  const menu = read("../components/pick-menu.tsx");
  assert.match(menu, /export const SEGMENTS_MAX = 3;/);
  assert.match(menu, /aria-haspopup="menu"[\s\S]*role="menuitemradio"[\s\S]*aria-disabled=\{off \|\| undefined\}/);
  const css = read("../styles/app.css");
  const phone = css.slice(css.indexOf("@media screen and (max-width: 760px)"));
  assert.match(
    phone,
    /\.opt-controls \.chip,\n {2}\.opt-controls \.seg button,\n {2}\.opt-controls \.opt-choice,\n {2}\.opt-controls \.opt-pick,\n {2}\.opt-controls \.opt-custom \{\n {4}position: relative;\n {4}height: 28px;\n {4}min-height: 28px;\n {4}padding: 0 8px;\n {4}font-size: var\(--text-xs\);/,
  );
  assert.match(css, /--text-xs: 11px;/);
});
