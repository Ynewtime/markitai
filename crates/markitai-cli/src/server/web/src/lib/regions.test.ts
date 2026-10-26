import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { foundFromKey } from "./regions.ts";

test("a card with more than one region finds the address from the key", () => {
  assert.equal(foundFromKey({ regions: ["https://api.minimax.io/v1", "https://api.minimax.cn/v1"] }), true);
  assert.equal(foundFromKey({ regions: ["https://only.example/v1"] }), false);
  assert.equal(foundFromKey({}), false);
  assert.equal(foundFromKey(null), false);
  // The address field says so for every such provider, not only on the Kimi card.
  const page = readFileSync(new URL("../components/settings-models.tsx", import.meta.url), "utf8");
  assert.match(page, /kimi \|\| foundFromKey\(provider\)\s*\? t\.baseAuto/);
});
