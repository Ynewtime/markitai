import assert from "node:assert/strict";
import test from "node:test";
import { EventParser, EventStream, type StreamEvent } from "./events.ts";

const ORIGIN = "http://127.0.0.1:3600";
const tick = () => new Promise((resolve) => setTimeout(resolve, 0));

test("the parser reads named events across arbitrary chunk boundaries and line endings", () => {
  const parser = new EventParser();
  const events: StreamEvent[] = [];
  const text = '﻿:ping\n\nevent: snapshot\ndata: {"a":\ndata: 1}\n\nevent:item\r\ndata:x\r\n\r\ndata: plain\r\rretry: 1500\nevent: job\ndata: {"status":"done"}\n\n';
  // Every split point must give the same events.
  for (let size = 1; size <= text.length; size++) {
    const each = new EventParser();
    const found: StreamEvent[] = [];
    for (let at = 0; at < text.length; at += size) found.push(...each.push(text.slice(at, at + size)));
    assert.deepEqual(
      found,
      [
        { type: "snapshot", data: '{"a":\n1}' },
        { type: "item", data: "x" },
        { type: "message", data: "plain" },
        { type: "job", data: '{"status":"done"}' },
      ],
      `chunk size ${size}`,
    );
    assert.equal(each.retry, 1500);
  }
  // An event without a final blank line is never dispatched.
  events.push(...parser.push("event: item\ndata: half"));
  assert.deepEqual(events, []);
});

function streamResponse(chunks: string[], { status = 200, type = "text/event-stream" } = {}) {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status, headers: { "content-type": type } });
}

test("a stream sends the header token, dispatches events and reconnects after the connection ends", async () => {
  const calls: { url: string; init: RequestInit }[] = [];
  const responses = [
    streamResponse(['event: snapshot\ndata: {"n":1}\n\n', 'event: item\ndata: {"n":2}\n\n']),
    streamResponse(['event: snapshot\ndata: {"n":3}\n\n']),
  ];
  const timers: (() => void)[] = [];
  const stream = new EventStream("/api/jobs/abc/events", {
    origin: ORIGIN,
    headers: () => ({ Authorization: "Bearer secret" }),
    fetch: async (url, init) => {
      calls.push({ url: String(url), init: init ?? {} });
      const next = responses.shift();
      if (!next) throw new TypeError("offline");
      return next;
    },
    setTimeout: (callback) => timers.push(callback),
  });
  const seen: string[] = [];
  const states: number[] = [];
  for (const type of ["snapshot", "item", "open", "error"]) {
    stream.addEventListener(type, (event) => {
      seen.push(`${event.type}:${event.data}`);
      states.push(stream.readyState);
    });
  }
  assert.equal(stream.readyState, EventStream.CONNECTING);
  for (let i = 0; i < 10; i++) await tick();
  assert.equal(calls.length, 1);
  assert.equal(calls[0]?.url, `${ORIGIN}/api/jobs/abc/events`);
  assert.ok(!calls[0]?.url.includes("token"));
  assert.deepEqual(calls[0]?.init.headers, { Accept: "text/event-stream", Authorization: "Bearer secret" });
  assert.equal(calls[0]?.init.redirect, "error");
  assert.deepEqual(seen, ["open:", 'snapshot:{"n":1}', 'item:{"n":2}', "error:"]);
  // The end of the body is a dropped connection: CONNECTING, retried after a delay.
  assert.equal(stream.readyState, EventStream.CONNECTING);
  assert.equal(timers.length, 1);
  timers.shift()?.();
  for (let i = 0; i < 10; i++) await tick();
  assert.equal(calls.length, 2);
  assert.deepEqual(seen.slice(4), ["open:", 'snapshot:{"n":3}', "error:"]);
  // A failing network keeps retrying, like EventSource.
  timers.shift()?.();
  for (let i = 0; i < 10; i++) await tick();
  assert.equal(stream.readyState, EventStream.CONNECTING);
  assert.equal(timers.length, 1);
  stream.close();
  assert.equal(stream.readyState, EventStream.CLOSED);
  timers.shift()?.();
  for (let i = 0; i < 10; i++) await tick();
  assert.equal(calls.length, 3, "a closed stream never reconnects");
  assert.ok(states.includes(EventStream.OPEN));
});

test("a refused stream is final, and only service API URLs are accepted", async () => {
  for (const response of [
    () => streamResponse(['{"reason":"job_not_found"}'], { status: 404, type: "application/json" }),
    () => streamResponse(["data: x\n\n"], { status: 200, type: "text/html" }),
  ]) {
    const timers: (() => void)[] = [];
    const stream = new EventStream("/api/jobs/gone/events", { origin: ORIGIN, headers: () => ({}), fetch: async () => response(), setTimeout: (callback) => timers.push(callback) });
    const errors: number[] = [];
    stream.addEventListener("error", () => errors.push(stream.readyState));
    stream.addEventListener("message", () => assert.fail("no events from a refused stream"));
    for (let i = 0; i < 10; i++) await tick();
    assert.deepEqual(errors, [EventStream.CLOSED]);
    assert.equal(timers.length, 0);
  }
  assert.throws(() => new EventStream("/ui/app.js", { origin: ORIGIN }), /restricted/);
  assert.throws(() => new EventStream("https://evil.example/api/x", { origin: ORIGIN }), /External/);
});

test("closing while connected stops dispatch and aborts the request", async () => {
  let aborted = false;
  let push: ((text: string) => void) | null = null;
  const stream = new EventStream("/api/jobs/abc/events", {
    origin: ORIGIN,
    headers: () => ({}),
    fetch: async (_url, init) => {
      init?.signal?.addEventListener("abort", () => {
        aborted = true;
      });
      const encoder = new TextEncoder();
      const body = new ReadableStream<Uint8Array>({
        start(controller) {
          push = (text) => controller.enqueue(encoder.encode(text));
          init?.signal?.addEventListener("abort", () => controller.error(new DOMException("aborted", "AbortError")));
        },
      });
      return new Response(body, { headers: { "content-type": "text/event-stream; charset=utf-8" } });
    },
    setTimeout: () => assert.fail("no reconnection after close"),
  });
  const seen: string[] = [];
  stream.addEventListener("item", (event) => {
    seen.push(event.data);
    stream.close();
  });
  for (let i = 0; i < 10; i++) await tick();
  (push as unknown as (text: string) => void)("event: item\ndata: 1\n\nevent: item\ndata: 2\n\n");
  for (let i = 0; i < 10; i++) await tick();
  assert.deepEqual(seen, ["1"]);
  assert.equal(aborted, true);
  assert.equal(stream.readyState, EventStream.CLOSED);
});
