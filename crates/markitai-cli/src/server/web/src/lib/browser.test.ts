// Browser-facing helpers against small stand-ins for the document: printing,
// copying, and the first-paint script.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import { runInNewContext } from "node:vm";
import { copyText } from "./clipboard.ts";
import { printDocument, waitForImages } from "./print.ts";

interface FakeNode {
  tag: string;
  className: string;
  attributes: Map<string, string>;
  children: FakeNode[];
  parent: FakeNode | null;
  setAttribute(name: string, value: string): void;
  removeAttribute(name: string): void;
  append(...nodes: FakeNode[]): void;
  remove(): void;
  querySelectorAll(selector: string): FakeNode[];
  cloneNode(): FakeNode;
  [key: string]: unknown;
}

function fakeDocument() {
  const make = (tag: string): FakeNode => {
    const node: FakeNode = {
      tag,
      className: "",
      attributes: new Map(),
      children: [],
      parent: null,
      setAttribute(name: string, value: string) {
        node.attributes.set(name, value);
      },
      removeAttribute(name: string) {
        node.attributes.delete(name);
      },
      append(...nodes: FakeNode[]) {
        for (const child of nodes) {
          child.parent = node;
          node.children.push(child);
        }
      },
      remove() {
        if (node.parent) node.parent.children = node.parent.children.filter((child) => child !== node);
        node.parent = null;
      },
      querySelectorAll(selector: string) {
        const found: FakeNode[] = [];
        const walk = (current: FakeNode) => {
          for (const child of current.children) {
            if (child.tag === selector) found.push(child);
            walk(child);
          }
        };
        walk(node);
        return found;
      },
      cloneNode() {
        const copy = make(node.tag);
        copy.attributes = new Map(node.attributes);
        for (const child of node.children) copy.append(child.cloneNode());
        Object.assign(copy, { complete: node.complete, naturalWidth: node.naturalWidth, decode: node.decode });
        return copy;
      },
    };
    return node;
  };
  const classes = new Set<string>();
  const body = make("body");
  Object.assign(body, { classList: { add: (...names: string[]) => names.forEach((name) => classes.add(name)), remove: (...names: string[]) => names.forEach((name) => classes.delete(name)) } });
  return { doc: { createElement: make, body, title: "Workbench" }, make, body, classes };
}

function fakeWindow() {
  const listeners = new Map<string, () => void>();
  let printed = 0;
  return {
    win: {
      addEventListener: (name: string, fn: () => void) => listeners.set(name, fn),
      removeEventListener: (name: string) => listeners.delete(name),
      print: () => {
        printed++;
      },
    },
    fire: (name: string) => listeners.get(name)?.(),
    printed: () => printed,
  };
}

test("printing clones the document without links, waits for images and restores the page", async () => {
  const { doc, make, body, classes } = fakeDocument();
  const source = make("div");
  const link = make("a");
  link.setAttribute("href", "/api/jobs/j/files/x?token=secret");
  link.setAttribute("download", "");
  const image = make("img");
  Object.assign(image, { complete: true, naturalWidth: 10, decode: async () => undefined });
  source.append(link, image);
  const { win, fire, printed } = fakeWindow();
  const job = printDocument({ source: source as never, title: "report\u0007", furniture: true, doc: doc as never, win: win as never });
  await new Promise((resolve) => setTimeout(resolve, 0));
  assert.equal(printed(), 1);
  assert.equal(doc.title, "report");
  assert.ok(classes.has("is-printing") && classes.has("print-furniture"));
  const host = body.children[0];
  assert.equal(host?.className, "print-host");
  const printedLink = host?.querySelectorAll("a")[0] as FakeNode;
  assert.equal(printedLink.attributes.has("href"), false);
  assert.equal(link.attributes.has("href"), true);
  fire("afterprint");
  await job.done;
  assert.equal(body.children.length, 0);
  assert.equal(doc.title, "Workbench");
  assert.equal(classes.size, 0);
});

test("a broken image stops printing with the explicit reason; a changed selection cancels", async () => {
  const { doc, make } = fakeDocument();
  const source = make("div");
  const image = make("img");
  Object.assign(image, { complete: false, naturalWidth: 0, decode: async () => Promise.reject(new Error("decoder words")) });
  source.append(image);
  const { win, printed } = fakeWindow();
  const broken = printDocument({ source: source as never, title: "x", furniture: false, doc: doc as never, win: win as never, messages: { loading: "loading", broken: "broken image" } });
  await assert.rejects(broken.done, /broken image/);
  assert.equal(printed(), 0);
  const empty = make("div");
  const stale = printDocument({ source: empty as never, title: "x", furniture: false, isCurrent: () => false, doc: doc as never, win: win as never });
  await assert.rejects(stale.done, (error: Error) => error.name === "AbortError");
  await assert.rejects(waitForImages([], { signal: AbortSignal.abort() }), (error: Error) => error.name === "AbortError");
});

test("copying falls back to the selection command when the clipboard is missing or refused", async () => {
  let executed = 0;
  const area = { value: "", setAttribute() {}, select() {}, remove() {}, className: "" };
  const doc = {
    activeElement: null,
    createElement: () => area,
    body: { append() {} },
    execCommand: () => {
      executed++;
      return true;
    },
  };
  assert.equal(await copyText("a", { clipboard: { writeText: async () => undefined }, doc: doc as never }), true);
  assert.equal(executed, 0);
  assert.equal(await copyText("b", { clipboard: { writeText: async () => Promise.reject(new Error("denied")) }, doc: doc as never }), true);
  assert.equal(await copyText("c", { clipboard: undefined, doc: doc as never }), true);
  assert.equal(executed, 2);
  assert.equal(await copyText("d", { clipboard: undefined, doc: { ...doc, execCommand: () => false } as never }), false);
});

test("boot.js applies a stored theme and the language before first paint", () => {
  const source = readFileSync(new URL("../../public/boot.js", import.meta.url), "utf8");
  const run = (stored: Record<string, string>, language: string, throws = false) => {
    const attributes = new Map<string, string>();
    const root = { lang: "", setAttribute: (name: string, value: string) => attributes.set(name, value) };
    const document = { documentElement: root, title: "Markitai" };
    const localStorage = {
      getItem: (key: string) => {
        if (throws) throw new Error("blocked");
        return stored[key] ?? null;
      },
    };
    runInNewContext(source, { document, localStorage, navigator: { language }, String });
    return { root, attributes, document };
  };
  const dark = run({ "markitai.theme": "dark", "markitai.lang": "zh" }, "en-US");
  assert.equal(dark.attributes.get("data-theme"), "dark");
  assert.equal(dark.root.lang, "zh-CN");
  assert.match(dark.document.title, /Markitai/);
  const auto = run({ "markitai.theme": "neon" }, "zh-TW");
  assert.equal(auto.attributes.has("data-theme"), false);
  assert.equal(auto.root.lang, "zh-CN");
  assert.equal(run({}, "fr", true).root.lang, "en");
  // Everything the page loads is a same-origin file: the CSP allows nothing inline.
  const html = readFileSync(new URL("../../public/index.html", import.meta.url), "utf8");
  for (const tag of html.match(/<script\b[^>]*>/g) ?? []) assert.match(tag, /\ssrc="\/ui\/[a-z]+\.js"/);
  assert.ok(!/\sstyle=/.test(html));
});
