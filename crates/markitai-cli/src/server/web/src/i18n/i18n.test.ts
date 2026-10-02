import assert from "node:assert/strict";
import { readdirSync, readFileSync } from "node:fs";
import test from "node:test";
import { en } from "./en.ts";
import { apiErrorText, isUnsupported, itemErrorText, MESSAGES, persistenceText, producibleKeys, serviceNote } from "./errors.ts";
import { detectLocale } from "./locale.ts";
import { zh } from "./zh.ts";

const placeholders = (text: string) => [...text.matchAll(/\{(\w+)\}/g)].map((match) => match[1]).sort();
const server = new URL("../../../", import.meta.url);
const read = (relative: string, base: URL = server) => readFileSync(new URL(relative, base), "utf8");

test("both interface languages define the same entries with the same shapes", () => {
  assert.deepEqual(Object.keys(zh).sort(), Object.keys(en).sort());
  for (const [key, value] of Object.entries(en)) {
    const other = (zh as Record<string, unknown>)[key];
    assert.equal(typeof other, typeof value, key);
    if (typeof value === "function") assert.equal((other as () => unknown).length, value.length, key);
    else assert.ok((other as string).length > 0, key);
  }
});

test("visible copy keeps the voice: no dashes as separators", () => {
  for (const table of [en, zh]) {
    for (const [key, value] of Object.entries(table)) {
      if (typeof value === "string") assert.ok(!/[–—]/.test(value), key);
    }
  }
  assert.equal(en.sessResults(1), "1 item in session · View results");
  assert.equal(en.addModelsCount(2), "Add 2 models");
  assert.equal(zh.retryAllFailed(3), "重试全部失败项（3）");
});

test("service messages exist in both languages with the same placeholders", () => {
  assert.deepEqual(Object.keys(MESSAGES.zh).sort(), Object.keys(MESSAGES.en).sort());
  for (const [key, text] of Object.entries(MESSAGES.en)) {
    assert.deepEqual(placeholders(MESSAGES.zh[key as keyof typeof MESSAGES.zh]), placeholders(text), key);
  }
  const keys = producibleKeys();
  assert.ok(keys.length > 70, `only ${keys.length}`);
  for (const key of keys) assert.ok(key in MESSAGES.en, key);
});

test("an explicit language wins, otherwise the browser language decides", () => {
  assert.equal(detectLocale("zh", "en-US"), "zh");
  assert.equal(detectLocale("en", "zh-CN"), "en");
  assert.equal(detectLocale(null, "zh-TW"), "zh");
  assert.equal(detectLocale("fr", "de-DE"), "en");
  assert.equal(detectLocale(null, undefined), "en");
});

test("every reason the Rust service sends and every core error code has localized text", () => {
  const rust = [
    "http.rs",
    "types.rs",
    "security.rs",
    "files.rs",
    "store.rs",
    "rerun.rs",
    "jobs.rs",
    "providers.rs",
    "tickets.rs",
    "settings.rs",
    ...readdirSync(new URL("settings/", server)).map((name) => `settings/${name}`),
  ]
    .map((name) => read(name))
    .join("\n");
  const reasons = new Set([...rust.matchAll(/(?:ApiError|Self)::(?:new|structured)\(\s*\d+,\s*"([a-z_]+)"/g)].map((match) => match[1]));
  for (const reason of ["request_too_large", "invalid_multipart"]) {
    assert.ok(rust.includes(`"${reason}"`), reason);
    reasons.add(reason);
  }
  assert.ok(reasons.size > 50, `only ${reasons.size} reasons found`);
  // Routing and startup errors keep the generic status wording by design.
  const generic = new Set(["route_not_found", "method_not_allowed", "invalid_allowed_host"]);
  for (const reason of reasons) {
    if (generic.has(reason ?? "")) continue;
    assert.notEqual(apiErrorText("en", { detail: "service words", reason })?.text, "service words", `unmapped reason ${reason}`);
  }
  const core = read("../../../markitai-core/src/types.rs");
  const codes = new Set(
    [...core.slice(core.indexOf("pub fn code"), core.indexOf("pub type Result")).matchAll(/=> "([a-z_]+)"/g)].map((match) => match[1]),
  );
  assert.ok(codes.size >= 9, [...codes].join());
  for (const code of ["cancelled", "shutdown", "internal_error", "enhancement_failed", "no_output", "output_conflict", "file_not_found", "output_identity_conflict"]) {
    assert.ok(rust.includes(`"${code}"`), code);
    codes.add(code);
  }
  for (const code of codes) {
    assert.notEqual(itemErrorText("en", { error: "opaque words", error_code: code }).text, "opaque words", `unmapped code ${code}`);
  }
});

test("the English phrases recognized here are still the ones the core and the service write", () => {
  const tree = (relative: string) =>
    readdirSync(new URL(relative, server), { recursive: true })
      .filter((name) => name.endsWith(".rs"))
      .map((name) => read(`${relative}${name}`))
      .join("\n");
  const rust = tree("../../../markitai-core/src/") + tree("./");
  for (const phrase of [
    "No model configured",
    "Unsupported file format: ",
    " Supported extensions: ",
    "Local OCR requires macOS",
    "{BACKEND_VARIABLE}=vision selects macOS Vision",
    "unset it or choose paddle",
    "{BACKEND_VARIABLE} must be vision or paddle",
    "Local OCR needs the model {}",
    "which is not installed: {}",
    "Local OCR download of {} failed ({cause})",
    "Local OCR models: {message}",
    "to replace this damaged managed model; ordinary OCR will not overwrite it",
    "managed home must be owned by this user and not writable by others",
    "managed installation directory must be owned by this user and private",
    "installation file is a link, junction or special file",
    "installation file must be private, owned by this user and have one hard link",
    "installation directory changed or is unsafe",
    "installation directory is a link, junction or special file",
    "installation file changed while it was held",
    "installation lock was replaced",
    "the download of {} does not match its published size and SHA-256; nothing was installed",
    "LLM returned HTTP {status}",
    "the model is not available in this region",
    "the account's quota or billing does not allow this request",
    "the model is unavailable",
    "LLM request timed out",
    "LLM request failed",
    "HTTP {}",
    "Remote fetching is disabled by policy",
    "URL returned no extractable content",
    "Model connection test timed out",
    "Model connection request failed",
    "Model connection returned HTTP {status}",
    "An API endpoint is required",
    "{} responded",
    "Model credentials or endpoint configuration are invalid or unavailable",
    "This model provider is not supported by the native runtime",
    "Model connection test failed",
    "Model discovery failed; check endpoint",
    "Refresh failed; showing previously discovered models",
    "Model discovery wait timed out",
    "Too many model discovery requests are active",
    "cancelled (stopped by request)",
    "cancelled (server shutdown)",
    "LLM enhancement did not produce an enhanced result",
    "history could not be persisted; completed artifacts remain available until shutdown",
    "output rollback failed; restart to recover the previous result",
    "item deletion rollback failed; restart to recover",
  ]) {
    assert.ok(rust.includes(phrase), `no Rust source writes "${phrase}" any more`);
  }
});

test("item errors are localized by code or shape, keeping the original wording", () => {
  const cases: [Parameters<typeof itemErrorText>[1], string][] = [
    [{ error: "cancelled (stopped by request)", error_code: "cancelled" }, "Stopped before conversion · retry to convert it"],
    [{ error: "No model configured; set MODEL", error_code: "no_model_configured" }, MESSAGES.en.errNoModel],
    [{ error: "HTTP 404 for https://example.com/a: gone", kind: "url", error_code: "fetch_error" }, "The page was not found (HTTP 404)."],
    [{ error: "HTTP 429", kind: "url" }, "The website is limiting requests (HTTP 429). Try again later."],
    [{ error: "operation timed out", kind: "url" }, "Fetching this page took too long."],
    [{ error: "Office export timed out", kind: "file" }, "The conversion took too long and was stopped."],
    [{ error: "LLM returned HTTP 403: the model is not available in this region" }, MESSAGES.en.errModelRegion.replace("{status}", "403")],
    [{ error: "error sending request: tcp connect error: Connection refused", kind: "url" }, MESSAGES.en.errUnreachable],
    [{ error: "Input exceeds the 500 MiB limit", error_code: "invalid_input" }, "The input is larger than a supported limit (500 MiB)."],
    [{ error: "Malformed XML", error_code: "conversion_error" }, "The document could not be converted."],
  ];
  for (const [item, text] of cases) {
    const result = itemErrorText("en", item);
    assert.equal(result.text, text, String(item.error));
    assert.equal(result.detail, item.error, String(item.error));
  }
  assert.equal(itemErrorText("en", { error: "cancelled (stopped by request)" }).hint, true);
  assert.deepEqual(itemErrorText("en", { error: "something entirely new" }), { text: "something entirely new", detail: "", hint: false, formats: "" });
  const unsupported = itemErrorText("en", { error: "Unsupported file format: '.xyz'. Supported extensions: .csv .docx.", error_code: "unsupported" });
  assert.equal(unsupported.text, "This file type is not supported: '.xyz'.");
  assert.equal(unsupported.formats, ".csv .docx");
  assert.equal(itemErrorText("zh", { error: "HTTP 404", kind: "url" }).text, "网页不存在（HTTP 404）。");
  assert.equal(itemErrorText("zh", { error: "x", error_code: "conversion_error" }).text, "无法转换这个文档。");
});

test("current portable OCR errors explain the right recovery and keep the original detail", () => {
  const model = "ppocrv6-det-small";
  const path = "C:\\Markitai state\\models\\ocr\\ppocrv6-det-small-12345678\\model.onnx";
  const remedy = `run \`markitai doctor --fix\` where the network is reachable, or download https://example.test/model.onnx (SHA-256 ${"a".repeat(64)}, 9929594 bytes) to ${path}`;
  const repair = "; run `markitai doctor --fix` to replace this damaged managed model; ordinary OCR will not overwrite it";
  const cases: [string, string, keyof typeof MESSAGES.en][] = [
    ["MARKITAI_OCR_BACKEND=vision selects macOS Vision, which needs macOS 11 or later; unset it or choose paddle", "unsupported", "errOcrVisionSelection"],
    ["MARKITAI_OCR_BACKEND must be vision or paddle", "config_error", "errOcrBackendSetting"],
    [`Local OCR needs the model ${model}, which is not installed: ${remedy}`, "unsupported", "errOcrModelMissing"],
    [`Local OCR download of ${model} failed (official download could not be reached); any existing damaged file was kept: ${remedy}`, "unsupported", "errOcrModelDownload"],
    [`Local OCR download of ${model} failed (official download was interrupted: timed out); any existing damaged file was kept: ${remedy}`, "unsupported", "errOcrModelDownload"],
    [`Local OCR models: ${path} has 10 bytes instead of its published 9929594 bytes${repair}`, "conversion_error", "errOcrModelCorrupt"],
    [`Local OCR models: ${path} does not match its published SHA-256${repair}`, "conversion_error", "errOcrModelCorrupt"],
    [`Local OCR models: the download of ${model} does not match its published size and SHA-256; nothing was installed`, "conversion_error", "errOcrModelPreparation"],
  ];
  for (const [error, error_code, key] of cases) {
    for (const locale of ["en", "zh"] as const) {
      const result = itemErrorText(locale, { error, error_code, kind: "file" });
      assert.equal(result.text, MESSAGES[locale][key], error);
      assert.equal(result.detail, error, error);
      assert.equal(result.hint, false);
      assert.notEqual(result.text, MESSAGES[locale].errOcrUnavailable, error);
    }
  }
});

test("only missing-model and download OCR errors can retry an unsupported item", () => {
  for (const error of [
    "Local OCR needs the model ppocrv6-det-small, which is not installed: run `markitai doctor --fix` where the network is reachable",
    "Local OCR download of ppocrv6-det-small failed (official download could not be reached); any existing damaged file was kept: run `markitai doctor --fix`",
  ]) {
    assert.equal(isUnsupported({ error, error_code: "unsupported" }), false, error);
  }
  for (const error of [
    "Unsupported file format: '.xyz'.",
    "MARKITAI_OCR_BACKEND=vision selects macOS Vision, which needs macOS 11 or later; unset it or choose paddle",
    "Local OCR requires macOS 11 or later; no local OCR backend is available on this platform",
    "Unknown unsupported capability",
    "Local OCR needs something else",
    "Local OCR download was not requested",
    "",
  ]) {
    assert.equal(isUnsupported({ error, error_code: "unsupported" }), true, error);
  }
  assert.equal(isUnsupported({ error: "Unsupported file format: '.xyz'." }), true);
  assert.equal(isUnsupported({ error: "ordinary conversion error", error_code: "conversion_error" }), false);
});

test("unsafe portable OCR paths require inspection, never blind automatic repair", () => {
  const reasons = [
    "installation file is a link, junction or special file",
    "installation file must be private, owned by this user and have one hard link",
    "managed home must be owned by this user and not writable by others",
    "managed installation directory must be owned by this user and private",
    "installation directory is a link, junction or special file",
    "installation directory changed or is unsafe",
    "installation file changed while it was held",
    "installation lock was replaced",
  ];
  for (const reason of reasons) {
    // Preflight reports once; an unsafe state returned by install can retain a nested prefix.
    for (const prefix of ["Local OCR models: ", "Local OCR models: Local OCR models: "]) {
      const error = `${prefix}C:\\Markitai state\\models\\ocr\\install.lock: ${reason}`;
      for (const locale of ["en", "zh"] as const) {
        const result = itemErrorText(locale, { error, error_code: "conversion_error" });
        assert.equal(result.text, MESSAGES[locale].errOcrModelPath, error);
        assert.equal(result.detail, error);
        assert.ok(result.text.includes("markitai doctor"), locale);
        assert.ok(!result.text.includes("doctor --fix"), locale);
      }
    }
  }
});

test("API refusals are read by reason, then structured code, then status code", () => {
  assert.deepEqual(apiErrorText("en", { detail: "file exceeds upload limit", code: "payload_too_large", reason: "file_too_large" }), {
    text: MESSAGES.en.errFileTooLarge,
    detail: "file exceeds upload limit",
  });
  assert.equal(apiErrorText("en", { detail: { code: "stale_revision", current_revision: "r2" }, code: "conflict" })?.text, MESSAGES.en.errStaleRevision);
  assert.equal(apiErrorText("zh", { detail: "x", code: "rate_limited" })?.text, MESSAGES.zh.errBusy);
  assert.deepEqual(apiErrorText("en", { detail: "plain words" }), { text: "plain words", detail: "" });
  assert.equal(apiErrorText("en", null), null);
});

test("probe and discovery phrases are translated; unknown text is kept", () => {
  assert.deepEqual(serviceNote("en", "openai/gpt-x responded"), { text: "openai/gpt-x responded.", detail: "openai/gpt-x responded" });
  assert.equal(serviceNote("en", "Model connection test timed out").text, MESSAGES.en.errModelTimeout);
  assert.equal(serviceNote("zh", "Model connection returned HTTP 401").text, "服务商拒绝了凭据（HTTP 401）。请检查 API key。");
  assert.deepEqual(serviceNote("en", "A brand new phrase"), { text: "A brand new phrase", detail: "" });
  assert.deepEqual(serviceNote("en", undefined), { text: "", detail: "" });
  assert.equal(
    persistenceText("en", "history could not be persisted; completed artifacts remain available until shutdown").text,
    MESSAGES.en.errHistoryNotSaved,
  );
  assert.deepEqual(persistenceText("en", "new failure"), { text: "new failure", detail: "" });
});
