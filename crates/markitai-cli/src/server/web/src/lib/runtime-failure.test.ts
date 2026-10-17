// A subscription runtime that lists no models: the cause the page names, and when adding a model by hand is shut.
import assert from "node:assert/strict";
import test from "node:test";
import type { DiscoveryResult } from "../api/types.ts";
import { addingIsBlocked, listFailure, runtimeFailure } from "./model-test.ts";

const result = (over: Partial<DiscoveryResult>): DiscoveryResult => ({ provider: "claude-agent", status: "unavailable", models: [], ...over });
const runtime = { name: "Claude Code CLI", required: "2.1.284 or a later 2.1 patch release" };

test("the runtime's cause, name and next step are read from the answer", () => {
  const missing = runtimeFailure(
    result({ reason: "runtime_missing", detail: "Claude Code CLI was not found on the server's PATH", runtime, next_step: { kind: "install", command: "npm install -g @anthropic-ai/claude-code@2.1.284" } }),
  );
  assert.equal(missing?.reason, "runtime_missing");
  assert.equal(missing?.name, "Claude Code CLI");
  assert.deepEqual(missing?.step, { kind: "install", command: "npm install -g @anthropic-ai/claude-code@2.1.284" });
  const old = runtimeFailure(result({ reason: "runtime_unsupported", runtime: { ...runtime, installed: "2.2.0" }, next_step: { kind: "install", command: "x" } }));
  assert.equal(old?.installed, "2.2.0");
  const signedOut = runtimeFailure(result({ reason: "not_signed_in", runtime, next_step: { kind: "sign_in", command: "markitai auth claude login" } }));
  assert.equal(signedOut?.step?.kind, "sign_in");
});

test("anything else keeps the page's other failure notes", () => {
  assert.equal(runtimeFailure(result({ reason: "provider_error", detail: "Official Claude runtime or subscription authentication is unavailable" })), null);
  assert.equal(runtimeFailure(result({ reason: "runtime_missing" })), null, "a backend that names a cause without its runtime");
  assert.equal(runtimeFailure({ status: "ok", reason: "runtime_missing", runtime }), null);
  // The runtime reasons are not the test's categories: the older note is not drawn for them.
  assert.equal(listFailure(result({ reason: "runtime_missing", detail: "x", runtime })), null);
});

test("adding a model by hand is shut for a missing, unsupported or signed-out runtime, and only cautioned when it merely failed", () => {
  for (const reason of ["runtime_missing", "runtime_unsupported", "not_signed_in"] as const) assert.equal(addingIsBlocked(runtimeFailure(result({ reason, runtime }))), true, reason);
  assert.equal(addingIsBlocked(runtimeFailure(result({ reason: "runtime_failed", runtime }))), false);
  assert.equal(addingIsBlocked(null), false);
});
