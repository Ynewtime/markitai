import test from "node:test";
import assert from "node:assert/strict";
import type { CloudflareCapability, JobOptions } from "../api/types.ts";
import { cloudflareBatchScope, confirmedCloudflareRequests, freezeCloudflareRequests } from "./cloudflare.ts";
import { emptyOptions } from "./options.ts";
import { prepareRetryBatch, submitRetryBatch } from "./retry-batch.ts";
import { seedItem, type SessionItem, type SessionJob } from "./session.ts";

const caps: CloudflareCapability = { configured: true, available: true, reason: null, browser_rendering: true, file_conversion: true, file_extensions: ["pdf"] };
const cloud: JobOptions = { ...emptyOptions(), strategy: "cloudflare", backend: "cloudflare", screenshot: false, ocr: false };
const native: JobOptions = { ...emptyOptions(), strategy: "auto", backend: "native" };
const item = (jobId: string, kind: "file" | "url", name: string): SessionItem => ({ ...seedItem(jobId, { itemId: "0", name, kind, sizeBytes: null }), status: "error" });
const job = (jobId: string, options: JobOptions): SessionJob => ({ jobId, status: "done", createdAt: null, options, persistenceError: null });
const toRequests = (batch: ReturnType<typeof prepareRetryBatch>) => freezeCloudflareRequests(batch.map(({ item, options }) => ({ options, sources: [{ name: item.name, kind: item.kind }] })));

test("one mixed-batch scope includes every selected URL/file and native fallback", () => {
  const batch = prepareRetryBatch([item("a", "url", "https://example.test"), item("b", "file", "report.PDF"), item("c", "file", "note.txt")],
    { a: job("a", cloud), b: job("b", cloud), c: job("c", native) }, native);
  const requests = toRequests(batch);
  assert.deepEqual(cloudflareBatchScope(requests, caps), { selected: true, urls: 1, files: 1, candidates: 1, native: 1 });
  assert.deepEqual(confirmedCloudflareRequests(requests, true, caps)?.map((options) => options.remote_processing), ["cloudflare", "cloudflare", undefined]);
});

test("cancelling a mixed batch sends zero requests, including its native items", async () => {
  const batch = prepareRetryBatch([item("a", "url", "https://example.test"), item("b", "file", "note.txt")], { a: job("a", cloud), b: job("b", native) }, native);
  let calls = 0;
  const result = await submitRetryBatch(batch, confirmedCloudflareRequests(toRequests(batch), false, caps), async () => { calls++; return null; });
  assert.equal(result, null);
  assert.equal(calls, 0);
});

test("added or changed items and options cannot join an already confirmed list", async () => {
  const original = item("a", "file", "report.pdf");
  const items = [original];
  const options = { ...cloud };
  const jobs = { a: job("a", options) };
  const batch = prepareRetryBatch(items, jobs, native);
  const requests = toRequests(batch);
  original.itemId = "replacement";
  original.name = "changed.pdf";
  options.backend = "native";
  items.push(item("b", "url", "https://late.example.test"));
  const calls: { id: string; name: string; backend: unknown; permission: unknown }[] = [];
  await submitRetryBatch(batch, confirmedCloudflareRequests(requests, true, caps), async (item, options) => {
    calls.push({ id: item.itemId, name: item.name, backend: options.backend, permission: options.remote_processing });
    items.push(item === original ? original : { ...original, itemId: "another" });
    return null;
  });
  assert.deepEqual(calls, [{ id: "0", name: "report.pdf", backend: "cloudflare", permission: "cloudflare" }]);
});

test("a native-only batch needs no capabilities or confirmation and strips stale consent", async () => {
  const batch = prepareRetryBatch([item("a", "file", "note.txt")], { a: job("a", { ...native, remote_processing: "cloudflare" }) }, native);
  const requests = toRequests(batch);
  assert.equal(cloudflareBatchScope(requests).selected, false);
  const calls: JobOptions[] = [];
  assert.deepEqual(await submitRetryBatch(batch, confirmedCloudflareRequests(requests, false), async (_, options) => { calls.push(options); return null; }), { attempted: 1, failed: 0 });
  assert.equal(calls[0].remote_processing, undefined);
});

test("policy revocation at confirmation rejects the entire frozen batch", async () => {
  const batch = prepareRetryBatch([item("a", "file", "report.pdf")], { a: job("a", cloud) }, native);
  let calls = 0;
  const revoked = { ...caps, available: false, reason: "disabled_by_policy" as const };
  await submitRetryBatch(batch, confirmedCloudflareRequests(toRequests(batch), true, revoked), async () => { calls++; return null; });
  assert.equal(calls, 0);
});

test("bulk retry keeps the existing failed-only scope and rejects mismatched approval before sending", async () => {
  const skipped = { ...item("a", "file", "scan.pdf"), status: "done" as const, skipped: true, skipReason: "image_only" };
  assert.equal(prepareRetryBatch([skipped], { a: job("a", cloud) }, native).length, 0);
  const batch = prepareRetryBatch([item("a", "file", "report.pdf")], { a: job("a", cloud) }, native);
  let calls = 0;
  await assert.rejects(submitRetryBatch(batch, [], async () => { calls++; return null; }), /does not match/);
  assert.equal(calls, 0);
});

test("items sharing a job retain their own routes after another item's retry changed job options", () => {
  const local = { ...item("same", "file", "local.pdf"), itemId: "local", options: { ...native, remote_processing: "cloudflare" as const } };
  const remote = { ...item("same", "file", "remote.pdf"), itemId: "remote", options: { ...cloud, remote_processing: "cloudflare" as const } };
  const oldServer = { ...item("legacy", "file", "legacy.pdf"), itemId: "legacy" };
  const batch = prepareRetryBatch([local, remote, oldServer], { same: job("same", cloud), legacy: job("legacy", native) }, cloud);
  assert.deepEqual(batch.map((entry) => entry.options.backend), ["native", "cloudflare", "native"]);
  assert.ok(batch.every((entry) => entry.options.remote_processing === undefined));
  assert.deepEqual(confirmedCloudflareRequests(toRequests(batch), true, caps)?.map((options) => options.remote_processing), [undefined, "cloudflare", undefined]);
  assert.equal(local.options.remote_processing, "cloudflare");
});

test("one incompatible strategy blocks the whole retry batch without changing either choice", async () => {
  for (const strategy of ["jina", "defuddle"] as const) {
    const original = { ...cloud, strategy };
    const batch = prepareRetryBatch([item("a", "file", "remote.pdf"), item("b", "file", "local.txt")], { a: job("a", original), b: job("b", native) }, native);
    const authorization = confirmedCloudflareRequests(toRequests(batch), true, caps);
    let calls = 0;
    const result = await submitRetryBatch(batch, authorization, async () => { calls++; return null; });
    assert.equal(result, null);
    assert.equal(calls, 0);
    assert.equal(original.strategy, strategy);
    assert.equal(original.backend, "cloudflare");
  }
});
