import test from "node:test";
import assert from "node:assert/strict";
import type { CloudflareCapability, JobOptions, RemoteProcessing } from "../api/types.ts";
import { cloudflareReason, cloudflareScope, hasCloudflareRequest, requestOptions } from "./cloudflare.ts";
import { emptyOptions, publicOptions, initialComposer, readRemembered, resolveOptions } from "./options.ts";
const caps: CloudflareCapability = { configured: true, available: true, reason: null, browser_rendering: true, file_conversion: true, file_extensions: ["pdf", "docx"] };
const cloud: JobOptions = { ...emptyOptions(), strategy: "cloudflare", backend: "cloudflare", ocr: false, screenshot: false };
const sources = [{ name: "report.PDF", kind: "file" as const }, { name: "table.csv", kind: "file" as const }, { name: "https://example.test/private", kind: "url" as const }];

test("native routes have no Cloudflare step or permission", () => {
  const native = resolveOptions(initialComposer(readRemembered(null)));
  assert.equal(cloudflareScope(native, sources, caps).selected, false);
  assert.equal(requestOptions(native, true, caps).remote_processing, undefined);
});
test("mixed input scope uses advertised formats and does not count URLs as files", () => {
  assert.deepEqual(cloudflareScope(cloud, sources, caps), { selected: true, urls: 1, files: 1, candidates: 2, native: 0 });
  assert.equal(cloudflareScope({ ...cloud, strategy: "auto" }, sources, caps).urls, 0);
  assert.equal(cloudflareScope(cloud, sources, { ...caps, file_extensions: [] }).files, 0);
});
test("OCR and both screenshot modes retain native file reading without disabling requested features", () => {
  for (const option of ["ocr", "screenshot", "screenshot_only"] as const) {
    const selected = { ...cloud, [option]: true };
    assert.deepEqual(cloudflareScope(selected, sources, caps), { selected: true, urls: 1, files: 0, candidates: 0, native: 2 });
    assert.equal(requestOptions(selected, true, caps)[option], true);
  }
});
test("permission is fresh for every create, retry and enhance; history carries none", () => {
  const approved = requestOptions(cloud, true, caps);
  assert.equal(approved.remote_processing, "cloudflare");
  assert.equal(requestOptions(approved, false, caps).remote_processing, undefined);
  assert.equal(publicOptions(approved).remote_processing, undefined);
  assert.equal(cloud.remote_processing, undefined);
});
test("each refusal remains distinct and cannot be overridden by prior approval", () => {
  for (const reason of ["not_configured", "invalid_configuration", "disabled_by_policy", "client_not_trusted"] as const) {
    const refused = { ...caps, available: false, reason };
    assert.equal(cloudflareReason(refused, cloud), reason);
    assert.equal(requestOptions({ ...cloud, remote_processing: "cloudflare" }, true, refused).remote_processing, undefined);
  }
  assert.equal(cloudflareReason(undefined), "unavailable");
  assert.equal(requestOptions(cloud, true).remote_processing, undefined);
  assert.equal(requestOptions(cloud, true, { ...caps, file_conversion: false }).remote_processing, undefined);
  assert.equal(requestOptions(cloud, true, { ...caps, browser_rendering: false }).remote_processing, undefined);
});
test("external cost note comes only from server request facts, never selection or numeric LLM cost", () => {
  const requested: RemoteProcessing = { provider: "cloudflare", requested: true, execution: "unknown", external_charges: "not_included" };
  assert.equal(hasCloudflareRequest(requested), true);
  assert.equal(hasCloudflareRequest(null), false);
  assert.equal(hasCloudflareRequest(undefined), false);
});

test("a misleading or extensionless name never hides a possible upload from consent", () => {
  const scope = cloudflareScope(cloud, [{ name: "unknown", kind: "file" }, { name: "disguised.txt", kind: "file" }], caps);
  assert.equal(scope.files, 0);
  assert.equal(scope.candidates, 2);
  assert.equal(scope.native, 0);
});

test("Cloudflare file conversion never grants consent to a conflicting remote URL service", () => {
  for (const strategy of ["defuddle", "jina"] as const) {
    const chosen = { ...cloud, strategy, remote_processing: "cloudflare" as const };
    assert.equal(cloudflareReason(caps, chosen), "incompatible_strategy");
    assert.equal(cloudflareReason(undefined, chosen), "incompatible_strategy");
    const refused = requestOptions(chosen, true, caps);
    assert.equal(refused.remote_processing, undefined);
    assert.equal(refused.strategy, strategy);
    assert.equal(refused.backend, "cloudflare");
    assert.equal(chosen.remote_processing, "cloudflare");
    assert.equal(cloudflareReason(caps, { ...chosen, backend: "native" }), null);
    assert.equal(cloudflareReason(caps, { ...chosen, strategy: "cloudflare" }), null);
  }
});
