import test from "node:test";
import assert from "node:assert/strict";
import { cliCommand } from "./cli.ts";
import { BUILTIN_DEFAULTS, changeAdvanced, changedOptions, initialComposer, publicOptions, readRemembered, resolveOptions } from "./options.ts";

test("Text only is sent and printed only when chosen, so a backend without it never sees the key", () => {
  const composer = initialComposer(readRemembered(null));
  const plain = resolveOptions(composer);
  assert.equal("text_only" in plain, false);
  assert.equal(cliCommand(["./scan.png"], plain).includes("--text-only"), false);
  const chosen = { ...composer, advanced: changeAdvanced(composer.advanced, "textOnly", true) };
  const options = resolveOptions(chosen);
  assert.equal(options.text_only, true);
  assert.match(cliCommand(["./scan.png"], options), / --text-only$/);
  // No default holds it: on, it differs from the defaults (the dot on Options, Other shown bright).
  assert.deepEqual([...changedOptions(chosen, BUILTIN_DEFAULTS)], ["text_only"]);
  assert.equal(changedOptions(composer, BUILTIN_DEFAULTS).size, 0);
  // A stored job's options keep it for a retry; false is the default and stays out.
  assert.equal(publicOptions({ ...options }).text_only, true);
  assert.equal("text_only" in publicOptions({ text_only: false }), false);
});
