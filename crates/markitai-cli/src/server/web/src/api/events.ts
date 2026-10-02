// Server-sent events read with fetch, so a job's stream carries the
// Authorization header like every other request. EventSource cannot send
// headers, and a token in its URL would stay in server logs, proxies and the
// browser's history. This keeps the part of the EventSource interface the
// workbench uses: named events, readyState, close(), and reconnecting after a
// dropped connection (the service replays a snapshot on every connection).
import { bearer, serviceURL } from "./token.ts";

export interface StreamEvent {
  type: string;
  data: string;
}

/** The text/event-stream line protocol, fed in arbitrary chunks. */
export class EventParser {
  private buffer = "";
  private started = false;
  private pendingCR = false;
  private type = "";
  private data: string[] = [];
  /** The `retry:` field, in milliseconds, when the stream sets one. */
  retry: number | null = null;

  push(chunk: string): StreamEvent[] {
    let text = chunk;
    if (!this.started && text) {
      this.started = true;
      if (text.startsWith("﻿")) text = text.slice(1);
    }
    // A CR at the end of the last chunk may be the first half of a CRLF.
    if (this.pendingCR && text.startsWith("\n")) text = text.slice(1);
    this.pendingCR = false;
    this.buffer += text;
    const events: StreamEvent[] = [];
    for (;;) {
      const match = /\r\n|\r|\n/.exec(this.buffer);
      if (!match) break;
      if (match[0] === "\r" && match.index === this.buffer.length - 1) {
        this.pendingCR = true;
      }
      const line = this.buffer.slice(0, match.index);
      this.buffer = this.buffer.slice(match.index + match[0].length);
      const event = this.line(line);
      if (event) events.push(event);
    }
    return events;
  }

  private line(line: string): StreamEvent | null {
    if (line === "") {
      const data = this.data;
      const type = this.type || "message";
      this.data = [];
      this.type = "";
      return data.length ? { type, data: data.join("\n") } : null;
    }
    if (line.startsWith(":")) return null;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.type = value;
    else if (field === "data") this.data.push(value);
    else if (field === "retry" && /^\d+$/.test(value)) this.retry = Number(value);
    return null;
  }
}

type Listener = (event: StreamEvent) => void;

export interface StreamOptions {
  fetch?: typeof fetch;
  headers?: () => Record<string, string>;
  setTimeout?: (callback: () => void, ms: number) => unknown;
  clearTimeout?: (handle: unknown) => void;
  origin?: string;
}

export class EventStream {
  static readonly CONNECTING = 0;
  static readonly OPEN = 1;
  static readonly CLOSED = 2;
  readyState: number = EventStream.CONNECTING;
  private readonly listeners = new Map<string, Set<Listener>>();
  private controller: AbortController | null = null;
  private timer: unknown = null;
  private retry = 3000;
  private readonly url: URL;
  private readonly options: Required<Omit<StreamOptions, "origin">>;

  constructor(path: string, options: StreamOptions = {}) {
    this.url = serviceURL(path, options.origin);
    if (!this.url.pathname.startsWith("/api/")) throw new Error("Event streams are restricted to service API URLs");
    this.options = {
      fetch: options.fetch ?? ((input, init) => globalThis.fetch(input, init)),
      headers: options.headers ?? bearer,
      setTimeout: options.setTimeout ?? ((callback, ms) => globalThis.setTimeout(callback, ms)),
      clearTimeout: options.clearTimeout ?? ((handle) => globalThis.clearTimeout(handle as ReturnType<typeof setTimeout>)),
    };
    queueMicrotask(() => void this.connect());
  }

  addEventListener(type: string, listener: Listener): void {
    let set = this.listeners.get(type);
    if (!set) this.listeners.set(type, (set = new Set()));
    set.add(listener);
  }

  removeEventListener(type: string, listener: Listener): void {
    this.listeners.get(type)?.delete(listener);
  }

  close(): void {
    this.readyState = EventStream.CLOSED;
    if (this.timer !== null) this.options.clearTimeout(this.timer);
    this.timer = null;
    this.controller?.abort();
    this.controller = null;
  }

  private emit(type: string, data = ""): void {
    if (this.readyState === EventStream.CLOSED && type !== "error") return;
    for (const listener of [...(this.listeners.get(type) ?? [])]) listener({ type, data });
  }

  /** A dropped connection is retried; a refused one (HTTP error, wrong type) is final. */
  private async connect(): Promise<void> {
    if (this.readyState === EventStream.CLOSED) return;
    const controller = new AbortController();
    this.controller = controller;
    let response: Response;
    try {
      response = await this.options.fetch(this.url.href, {
        headers: { Accept: "text/event-stream", ...this.options.headers() },
        cache: "no-store",
        credentials: "same-origin",
        redirect: "error",
        signal: controller.signal,
      });
    } catch {
      this.reconnect(controller);
      return;
    }
    if (controller !== this.controller) return;
    const type = response.headers.get("content-type") ?? "";
    if (!response.ok || !type.toLowerCase().startsWith("text/event-stream") || !response.body) {
      await response.body?.cancel().catch(() => undefined);
      this.readyState = EventStream.CLOSED;
      this.controller = null;
      this.emit("error");
      return;
    }
    this.readyState = EventStream.OPEN;
    this.emit("open");
    const parser = new EventParser();
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    try {
      for (;;) {
        const part = await reader.read();
        if (part.done) break;
        for (const event of parser.push(decoder.decode(part.value, { stream: true }))) {
          if (controller !== this.controller) return;
          this.emit(event.type, event.data);
        }
        if (parser.retry !== null) this.retry = parser.retry;
      }
    } catch {
      // A reset connection ends the read like a closed one.
    }
    this.reconnect(controller);
  }

  private reconnect(controller: AbortController): void {
    if (controller !== this.controller || this.readyState === EventStream.CLOSED) return;
    this.readyState = EventStream.CONNECTING;
    this.emit("error");
    if (this.readyState === EventStream.CLOSED) return;
    this.timer = this.options.setTimeout(() => {
      this.timer = null;
      void this.connect();
    }, this.retry);
  }
}
