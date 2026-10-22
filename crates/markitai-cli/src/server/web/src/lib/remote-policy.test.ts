import test from "node:test";
import assert from "node:assert/strict";
import { offeredStrategies } from "./options.ts";

test("a user token is offered only the local strategies; the admin all six", () => {
  assert.deepEqual(offeredStrategies(true), ["auto", "static", "playwright"]);
  assert.deepEqual(offeredStrategies(false), ["auto", "static", "playwright", "defuddle", "jina", "cloudflare"]);
});
