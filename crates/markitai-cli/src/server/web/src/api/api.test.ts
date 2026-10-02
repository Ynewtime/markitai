import assert from "node:assert/strict";
import test from "node:test";
import { download } from "../lib/download.ts";
import { ApiError, downloadTicket, eventsPath, filePath, isRevisionConflict, NetworkError, throttle, upload } from "./client.ts";
import { bearer, hasToken, initToken, serviceURL, setToken, takeToken } from "./token.ts";

const ORIGIN = "http://127.0.0.1:3600";

function memoryStore(initial: Record<string, string> = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => void values.set(key, value),
    values,
  };
}

test("the fragment token wins, both copies leave the address and the rest survives", () => {
  assert.deepEqual(takeToken(`${ORIGIN}/jobs?token=q&x=1#token=f&tab=2`), { token: "f", cleaned: "/jobs?x=1#tab=2" });
  assert.deepEqual(takeToken(`${ORIGIN}/?token=q`), { token: "q", cleaned: "/" });
  assert.deepEqual(takeToken(`${ORIGIN}/jobs#section`), { token: null, cleaned: `${ORIGIN}/jobs#section` });
  const replaced: string[] = [];
  const store = memoryStore();
  initToken(`${ORIGIN}/#token=abc`, (path) => replaced.push(path), store);
  assert.deepEqual(replaced, ["/"]);
  assert.equal(store.values.get("markitai.service-token"), "abc");
  assert.deepEqual(bearer(), { Authorization: "Bearer abc" });
  // A later load without a token reads the session copy.
  initToken(`${ORIGIN}/jobs`, () => assert.fail("nothing to clean"), memoryStore({ "markitai.service-token": "kept" }));
  assert.equal(hasToken(), true);
  assert.deepEqual(bearer(), { Authorization: "Bearer kept" });
});

test("blocked storage keeps a memory-only token", () => {
  const blocked = {
    getItem: () => {
      throw new Error("blocked");
    },
    setItem: () => {
      throw new Error("blocked");
    },
  };
  initToken(`${ORIGIN}/#token=mem`, () => undefined, blocked);
  assert.deepEqual(bearer(), { Authorization: "Bearer mem" });
  initToken(`${ORIGIN}/`, () => undefined, blocked);
  assert.equal(hasToken(), false);
  setToken("  typed  ", blocked);
  assert.deepEqual(bearer(), { Authorization: "Bearer typed" });
  setToken("", blocked);
});

test("requests stay on this service and the token never enters a URL", async () => {
  assert.throws(() => serviceURL("//evil.example/api", ORIGIN), /External/);
  assert.throws(() => serviceURL("http://user:pw@127.0.0.1:3600/api", ORIGIN), /External/);
  assert.throws(() => serviceURL("https://evil.example/api/x", ORIGIN), /External/);
  assert.equal(serviceURL("/api/x?y=1", ORIGIN).href, `${ORIGIN}/api/x?y=1`);
  assert.equal(eventsPath("a b"), "/api/jobs/a%20b/events");
  assert.equal(filePath("j1", "assets/a b#.png"), "/api/jobs/j1/files/assets/a%20b%23.png");

  setToken("t0", memoryStore());
  const original = globalThis.fetch;
  const sent: { url: string; init: RequestInit }[] = [];
  let answer = { url: "/api/jobs/j1/archive?ticket=abc", ticket: "abc", expires_in: 60 };
  globalThis.fetch = (async (url: string | URL, init?: RequestInit) => {
    sent.push({ url: String(url), init: init ?? {} });
    return new Response(JSON.stringify(answer), { status: 201, headers: { "content-type": "application/json" } });
  }) as typeof fetch;
  try {
    assert.equal(await downloadTicket("/api/jobs/j1/archive"), "/api/jobs/j1/archive?ticket=abc");
    const [request] = sent;
    assert.ok(request && request.url.endsWith("/api/download-tickets") && !request.url.includes("t0"));
    assert.equal(request.init.method, "POST");
    assert.equal(new Headers(request.init.headers).get("authorization"), "Bearer t0");
    assert.deepEqual(JSON.parse(String(request.init.body)), { path: "/api/jobs/j1/archive" });
    // A ticket for another path, another origin or with other parameters is refused.
    for (const url of ["/api/history/archive?ticket=abc", "https://evil.example/api/jobs/j1/archive?ticket=abc", "/api/jobs/j1/archive?ticket=abc&token=t0"]) {
      answer = { url, ticket: "abc", expires_in: 60 };
      await assert.rejects(downloadTicket("/api/jobs/j1/archive"), /ticket|External/i, url);
    }
  } finally {
    globalThis.fetch = original;
    setToken("", memoryStore());
  }
});

test("archives are opened by the browser through a ticket, other files are read as Blobs", async () => {
  const saved: string[] = [];
  const deps = {
    ticket: async (path: string) => `${path}?ticket=x`,
    blob: async (path: string) => ({ blob: new Blob([path]), name: path.endsWith(".md") ? "server.md" : null }),
    saveURL: (url: string, name: string) => void saved.push(`url ${url} ${name}`),
    saveBlob: (_blob: Blob, name: string) => void saved.push(`blob ${name}`),
  };
  await download("/api/history/archive", "markitai-history.zip", true, { ...deps, tokenHeld: () => true });
  await download("/api/history/archive", "markitai-history.zip", true, { ...deps, tokenHeld: () => false });
  await download("/api/jobs/j/files/a.md", "a.md", false, { ...deps, tokenHeld: () => true });
  await download("/api/jobs/j/files/b.png", "b.png", false, { ...deps, tokenHeld: () => true });
  assert.deepEqual(saved, [
    "url /api/history/archive?ticket=x markitai-history.zip",
    "url /api/history/archive markitai-history.zip",
    "blob server.md",
    "blob b.png",
  ]);
});

test("a refusal keeps its reason, structured code, wording and current revision", () => {
  const plain = new ApiError(413, { detail: "file exceeds upload limit", code: "payload_too_large", reason: "file_too_large" });
  assert.equal(plain.reason, "file_too_large");
  assert.equal(plain.code, "payload_too_large");
  assert.equal(plain.detail, "file exceeds upload limit");
  const conflict = new ApiError(409, { detail: { code: "stale_revision", current_revision: "r9" }, code: "conflict" });
  assert.equal(conflict.reason, "stale_revision");
  assert.equal(conflict.currentRevision, "r9");
  assert.equal(isRevisionConflict(conflict), true);
  assert.equal(isRevisionConflict(new ApiError(409, { reason: "settings_read_only" })), false);
  assert.equal(isRevisionConflict(new ApiError(409, { reason: "config_changed" })), true);
  assert.equal(isRevisionConflict(new ApiError(400, { reason: "stale_revision" })), false);
  assert.equal(new ApiError(500, null).message, "HTTP 500");
});

class FakeRequest {
  static last: FakeRequest | null = null;
  headers: Record<string, string> = {};
  upload: { onprogress?: (event: { loaded: number; total: number; lengthComputable: boolean }) => void; onload?: (event: unknown) => void } = {};
  onabort?: () => void;
  onerror?: () => void;
  onload?: () => void;
  method = "";
  url = "";
  status = 0;
  responseText = "";
  responseURL = "";
  sent: unknown = null;
  constructor() {
    FakeRequest.last = this;
  }
  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }
  setRequestHeader(name: string, value: string) {
    this.headers[name] = value;
  }
  send(body: unknown) {
    this.sent = body;
  }
  abort() {
    this.onabort?.();
  }
}

const originalLocation = (globalThis as { location?: unknown }).location;
(globalThis as { location?: unknown }).location = { origin: ORIGIN, href: `${ORIGIN}/` };

test("uploads report byte progress, send the token and resolve with the job", async () => {
  setToken("up", memoryStore());
  const seen: [number, number][] = [];
  const pending = upload<{ job_id: string }>("/api/jobs", new FormData(), {
    Request: FakeRequest as unknown as { new (): XMLHttpRequest },
    onProgress: (loaded, total) => seen.push([loaded, total]),
  });
  const request = FakeRequest.last!;
  assert.equal(request.method, "POST");
  assert.equal(request.url, `${ORIGIN}/api/jobs`);
  assert.equal(request.headers.Authorization, "Bearer up");
  request.upload.onprogress?.({ loaded: 5, total: 10, lengthComputable: true });
  request.upload.onload?.({ lengthComputable: true, total: 10 });
  request.status = 201;
  request.responseURL = `${ORIGIN}/api/jobs`;
  request.responseText = '{"job_id":"abc"}';
  request.onload?.();
  assert.deepEqual(await pending, { job_id: "abc" });
  assert.deepEqual(seen, [
    [5, 10],
    [10, 10],
  ]);
  setToken("", memoryStore());
});

test("an upload can be cancelled; service, network and redirect failures keep their meaning", async () => {
  const hooks = { Request: FakeRequest as unknown as { new (): XMLHttpRequest } };
  const controller = new AbortController();
  const aborted = upload("/api/jobs", new FormData(), { ...hooks, signal: controller.signal });
  controller.abort();
  await assert.rejects(aborted, (error: Error) => error.name === "AbortError");

  const refused = upload("/api/jobs", new FormData(), hooks);
  Object.assign(FakeRequest.last!, { status: 422, responseURL: `${ORIGIN}/api/jobs`, responseText: '{"reason":"llm_unavailable","detail":"no model"}' });
  FakeRequest.last!.onload?.();
  await assert.rejects(refused, (error: ApiError) => error instanceof ApiError && error.reason === "llm_unavailable");

  const offline = upload("/api/jobs", new FormData(), hooks);
  FakeRequest.last!.onerror?.();
  await assert.rejects(offline, (error: Error) => error instanceof NetworkError);

  const redirected = upload("/api/jobs", new FormData(), hooks);
  Object.assign(FakeRequest.last!, { status: 200, responseURL: "http://elsewhere.example/", responseText: "{}" });
  FakeRequest.last!.onload?.();
  await assert.rejects(redirected, (error: Error) => error instanceof NetworkError);
  (globalThis as { location?: unknown }).location = originalLocation;
});

test("requests stay same-origin, refuse redirects and turn unreachable into NetworkError", async () => {
  (globalThis as { location?: unknown }).location = { origin: ORIGIN, href: `${ORIGIN}/` };
  const realFetch = globalThis.fetch;
  const seen: { url: string; init: RequestInit }[] = [];
  try {
    globalThis.fetch = (async (url: URL, init: RequestInit) => {
      seen.push({ url: String(url), init });
      return new Response('{"version":"1.3.0"}', { status: 200, headers: { "content-type": "application/json" } });
    }) as typeof fetch;
    const { fetchCapabilities, fetchSnapshot } = await import("./client.ts");
    assert.equal((await fetchCapabilities()).version, "1.3.0");
    assert.equal(seen[0]?.url, `${ORIGIN}/api/capabilities`);
    assert.equal(seen[0]?.init.redirect, "error");
    assert.equal(seen[0]?.init.credentials, "same-origin");
    globalThis.fetch = (async () => new Response('{"detail":"job not found","reason":"job_not_found"}', { status: 404 })) as typeof fetch;
    assert.equal(await fetchSnapshot("gone"), null);
    globalThis.fetch = (async () => {
      throw new TypeError("Failed to fetch");
    }) as typeof fetch;
    await assert.rejects(fetchCapabilities(), (error: Error) => error instanceof NetworkError);
  } finally {
    globalThis.fetch = realFetch;
    (globalThis as { location?: unknown }).location = originalLocation;
  }
});

test("progress redraws are throttled, keep the latest value and never drop completion", async () => {
  let clock = 0;
  const calls: [number, number][] = [];
  const report = throttle((loaded, total) => calls.push([loaded, total]), 100, () => clock);
  report(1, 10);
  clock = 10;
  report(2, 10);
  report(3, 10);
  clock = 20;
  report(10, 10);
  assert.deepEqual(calls, [
    [1, 10],
    [10, 10],
  ]);
  report.cancel();
});
