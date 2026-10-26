import assert from "node:assert/strict";
import test from "node:test";
import { deliberate, PICK_SETTLE_MS } from "./files.ts";

test("only a deliberate click on Convert starts a job", () => {
  const picked = 50_000;
  // A key press on the focused button always counts.
  assert.equal(deliberate(0, picked + 10, picked), true);
  // The second click of a double click (on the system picker's button) never does.
  assert.equal(deliberate(2, picked + 5_000, picked), false);
  // A pointer click as the picker closes does not; a second later it does.
  assert.equal(deliberate(1, picked + 200, picked), false);
  assert.equal(deliberate(1, picked + PICK_SETTLE_MS, picked), true);
  // Nothing picked yet (URLs only).
  assert.equal(deliberate(1, 1_000, 0), true);
});
