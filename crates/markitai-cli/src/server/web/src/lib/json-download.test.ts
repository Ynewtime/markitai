import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { documentPath } from "../api/client.ts";

const inspector = readFileSync(new URL("../components/inspector.tsx", import.meta.url), "utf8");
const app = readFileSync(new URL("../app.tsx", import.meta.url), "utf8");

test("the item's document JSON has its own path", () => {
  assert.equal(documentPath("j1", "i 2"), "/api/jobs/j1/items/i%202/document");
});

test("a JSON entry is offered in the item's menu only for a finished item on a server that keeps documents", () => {
  const entry = /if \(([^)]*)\)\s*more\.push\(\{ key: "json"/.exec(inspector)?.[1] ?? "";
  assert.match(entry, /previewable/);
  assert.match(entry, /item\.status === "done"/);
  assert.match(entry, /ctx\.documentJson/);
  // the capability comes from the server's features, and the file is named like the output
  assert.match(app, /documentJson: features\?\.document_json === true/);
  assert.match(app, /downloadJson: \(item\) =>[\s\S]*documentPath\(item\.jobId, item\.itemId\)/);
});
