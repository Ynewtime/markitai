// Owner check 2: no second rule under the properties card, a notice's items listed exactly, the build tags beside Docs and GitHub.
import assert from "node:assert/strict";
import test from "node:test";
import type { ItemPayload } from "../api/types.ts";
import { commitLabel, versionLine } from "./build.ts";
import { submissions } from "./conversions.ts";
import { splitFrontmatter } from "./markdown.ts";
import { emptyOptions } from "./options.ts";
import { itemFromPayload } from "./session.ts";

const FRONT = "---\ntitle: A post\nsource: https://example.com/p\n---";

test("a rule that opens the body right after the front matter is not drawn twice", () => {
  for (const rule of ["---", "***", "___", " - - -", "-----"]) {
    assert.equal(splitFrontmatter(`${FRONT}\n\n${rule}\n\nBody\n`).body.trim(), "Body", rule);
  }
  assert.equal(splitFrontmatter(`${FRONT}\n---\nBody`).body.trim(), "Body");
});

test("real separators stay: later rules, a body without front matter, and a heading underline", () => {
  assert.equal(splitFrontmatter(`${FRONT}\n\nFirst\n\n---\n\nSecond\n`).body.trim(), "First\n\n---\n\nSecond");
  assert.equal(splitFrontmatter("---\n\nBody").body, "---\n\nBody");
  assert.equal(splitFrontmatter(`${FRONT}\n\nTitle\n---\n`).body.trim(), "Title\n---");
  assert.equal(splitFrontmatter(FRONT).frontmatter, FRONT);
});

const payload = (id: string, jobItems: Partial<ItemPayload> = {}): ItemPayload => ({
  item_id: id, name: `${id}.pdf`, kind: "file", status: "done", error: null, error_code: null, output: `${id}.md`, output_name: null, duration_ms: 5 as number | null,
  finished_at: "2026-10-04T10:00:00Z", cost_usd: null, llm_enhanced: false, operation: "convert", skipped: false, skip_reason: null, retryable: true,
  warnings: [], options: { ...emptyOptions(), backend: "native" }, ...jobItems }) as ItemPayload;

test("a list narrowed to some items shows exactly those, in their submissions", () => {
  const items = ["a", "b", "c"].map((id, index) => itemFromPayload(index < 2 ? "j1" : "j2", payload(id), index));
  const jobs = {};
  const keys = new Set([items[0]!.key, items[2]!.key]);
  const list = submissions(jobs, items, "all", "", keys);
  assert.deepEqual(list.flatMap((entry) => entry.shown.map((item) => item.key)).sort(), [...keys].sort());
  assert.equal(submissions(jobs, items, "all", "", new Set(["nothing"])).length, 0);
  assert.equal(submissions(jobs, items, "all", "", null).flatMap((entry) => entry.shown).length, 3);
});

test("the commit beside GitHub is short, -dirty for local changes, and absent when unknown", () => {
  const caps = (build: { version: string; commit: string | null; built_at: string | null; dirty: boolean } | undefined) => versionLine({ version: "1.3.0-dev", ...(build ? { build } : {}) } as never);
  assert.equal(commitLabel(caps({ version: "1.3.0-dev", commit: "6fd85a1c0ffee00", built_at: null, dirty: false })!), "6fd85a1c0ffe");
  assert.equal(commitLabel(caps({ version: "1.3.0-dev", commit: "6fd85a1", built_at: null, dirty: true })!), "6fd85a1-dirty");
  assert.equal(commitLabel(caps(undefined)!), null);
});
