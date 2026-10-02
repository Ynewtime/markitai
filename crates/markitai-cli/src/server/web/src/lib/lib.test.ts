import assert from "node:assert/strict";
import test from "node:test";
import { cliCommand, shellQuote } from "./cli.ts";
import { compareLines } from "./diff.ts";
import { isHiddenName, selectFolderFiles, walkEntries } from "./files.ts";
import { countWords, fmtBytes, fmtCost, fmtDateTime, fmtDur, splitName, timestampMs } from "./format.ts";
import { artifactPath, loadArtifactImages, markdownPair, rewrite, splitFrontmatter } from "./markdown.ts";
import { manualModelId } from "./models.ts";
import {
  ADVANCED_DEFAULTS,
  applyPreset,
  changeAdvanced,
  initialComposer,
  matchingPreset,
  publicOptions,
  readRemembered,
  remember,
  resolveOptions,
  withOcrFor,
  type Composer,
} from "./options.ts";
import { parseUrls } from "./urls.ts";

const base: Composer = { preset: "minimal", llm: false, ocr: false, profile: null, advanced: { ...ADVANCED_DEFAULTS } };

test("presets set their five features and leave output and source choices alone", () => {
  const rich = applyPreset({ ...base, profile: "rag", advanced: { ...ADVANCED_DEFAULTS, alt: false, noCache: true } }, "rich");
  assert.equal(rich.llm, true);
  assert.equal(rich.profile, "rag");
  assert.equal(rich.advanced.alt, null);
  assert.equal(rich.advanced.noCache, true);
  const options = resolveOptions(rich);
  assert.deepEqual([options.alt, options.desc, options.screenshot, options.no_cache], [true, true, true, true]);
  assert.equal(matchingPreset(rich), "rich");
  const adjusted = { ...rich, advanced: { ...rich.advanced, screenshot: false } };
  assert.equal(matchingPreset(adjusted), "standard");
  assert.equal(matchingPreset({ ...rich, ocr: true }), null);
  // Without the LLM, image analysis is off whatever the preset says.
  assert.equal(resolveOptions({ ...rich, llm: false }).alt, false);
  assert.equal(resolveOptions({ ...rich, advanced: { ...rich.advanced, pure: true } }).desc, false);
});

test("plain mode and screenshots as the source exclude each other", () => {
  const pure = changeAdvanced(ADVANCED_DEFAULTS, "pure", true);
  const shot = changeAdvanced(pure, "screenshotOnly", true);
  assert.deepEqual([shot.pure, shot.screenshotOnly], [false, true]);
  assert.equal(resolveOptions({ ...base, advanced: shot }).screenshot, true);
});

test("only public option keys with valid values survive", () => {
  assert.deepEqual(
    publicOptions({ llm: true, ocr: "yes", profile: "rag", strategy: "warp", backend: "native", origin: "cli", preset: "custom-one" }),
    { ...publicOptions(null), llm: true, profile: "rag", backend: "native", preset: "custom-one" },
  );
  assert.equal(publicOptions([1]).llm, null);
  assert.deepEqual(withOcrFor({ ...publicOptions(null), llm: false }).ocr, true);
  assert.deepEqual(withOcrFor({ ...publicOptions(null), llm: true }).ocr, null);
});

test("the last choice is remembered per browser; damaged, blocked and older stores are read safely", () => {
  const values = new Map<string, string>();
  const store = { getItem: (key: string) => values.get(key) ?? null, setItem: (key: string, value: string) => void values.set(key, value) };
  remember({ ...base, ocr: true, profile: "okf", advanced: { ...ADVANCED_DEFAULTS, alt: true, strategy: "jina" } }, store);
  const saved = readRemembered(store);
  assert.deepEqual(saved, { preset: "minimal", llm: false, ocr: true, profile: "okf", images: { alt: true, desc: null, screenshot: null } });
  // The remote fetch choice is never remembered.
  assert.equal(initialComposer(saved).advanced.strategy, "auto");
  values.set("markitai.options", "{broken");
  assert.equal(readRemembered(store).preset, null);
  values.set("markitai.options", JSON.stringify({ preset: "standard", llm: true, alt: false }));
  assert.deepEqual(readRemembered(store).images.alt, false);
  const blocked = {
    getItem: () => {
      throw new Error("blocked");
    },
    setItem: () => {
      throw new Error("blocked");
    },
  };
  assert.equal(readRemembered(blocked).llm, null);
  remember(base, blocked);
});

test("the CLI line spells out only deviations, quoted for a shell", () => {
  const options = resolveOptions({ ...base, ocr: true, profile: "rag", advanced: { ...ADVANCED_DEFAULTS, noCache: true, strategy: "static" } });
  assert.equal(cliCommand([], options), "markitai <your-files-or-url-or-url_files> -o out/ --preset minimal --ocr --profile rag --no-cache --strategy static");
  assert.equal(cliCommand(["https://example.com/a b"], resolveOptions(base)), "markitai 'https://example.com/a b' -o out/ --preset minimal");
  assert.equal(shellQuote("it's"), `'it'\\''s'`);
  assert.equal(shellQuote("~/x"), "'~/x'");
});

test("URL lines accept bare domains and report the first unusable line", () => {
  assert.deepEqual(parseUrls("example.com/page\n\n https://a.example/x \n"), { urls: ["https://example.com/page", "https://a.example/x"], invalid: null });
  assert.deepEqual(parseUrls("https://ok.example\nftp://files.example\nnext.example").invalid, { value: "ftp://files.example", reason: "badScheme" });
  assert.deepEqual(parseUrls("not a url").invalid, { value: "not a url", reason: "badUrl" });
});

test("formatting reads well in tabular columns", () => {
  assert.deepEqual([fmtDur(4200), fmtDur(59_960), fmtDur(83_000), fmtDur(3_723_000)], ["4.2s", "1:00", "1:23", "1:02:03"]);
  assert.deepEqual([fmtCost(0), fmtCost(0.0123), fmtCost(1.5)], ["$0", "$0.0123", "$1.5"]);
  assert.deepEqual([fmtBytes(512), fmtBytes(2048), fmtBytes(3 * 1024 * 1024)], ["512 B", "2.0 KB", "3.0 MB"]);
  assert.equal(timestampMs("2026-10-02T08:34:19.257123+00:00"), Date.UTC(2026, 9, 2, 8, 34, 19, 257));
  assert.equal(timestampMs("2026-10-02T10:00:00+02:00"), Date.UTC(2026, 9, 2, 8, 0, 0));
  assert.equal(timestampMs("yesterday"), null);
  assert.equal(fmtDateTime(null), "-");
  assert.equal(countWords("Hello world 你好"), 4);
  assert.deepEqual(splitName("a-very-long-document-name-here.pdf"), ["a-very-long-document-n", "ame-here.pdf"]);
  assert.equal(splitName("short.pdf"), null);
});

test("the rendered view leaves a leading YAML block to Source", () => {
  assert.deepEqual(splitFrontmatter("---\ntitle: x\n---\n\n# Body"), { frontmatter: "---\ntitle: x\n---", body: "\n\n# Body" });
  assert.deepEqual(splitFrontmatter("# No front matter\n---\n"), { frontmatter: null, body: "# No front matter\n---\n" });
});

test("a base/LLM pair needs exact, unambiguous artifact identity", () => {
  const artifacts = (paths: string[]) => paths.map((relpath) => ({ relpath, size: 1 }));
  assert.deepEqual(markdownPair(artifacts(["a.v1.md", "a.v1.llm.md", "img.png"])), { base: "a.v1.md", llm: "a.v1.llm.md" });
  assert.equal(markdownPair(artifacts(["a.llm.md"])), null);
  assert.equal(markdownPair(artifacts(["a.md", "a.llm.md", "b.md", "b.llm.md"])), null);
});

test("document references resolve only to listed artifacts", () => {
  const allowed = new Set(["assets/a b.png", "doc.md", "sub/c.png"]);
  assert.equal(artifactPath("assets/a%20b.png", "doc.md", allowed), "assets/a b.png");
  assert.equal(artifactPath("./c.png", "sub/x.md", allowed), "sub/c.png");
  assert.equal(artifactPath("../doc.md", "sub/x.md", allowed), "doc.md");
  for (const bad of ["https://evil.example/a.png", "//evil.example/a.png", "/etc/passwd", "assets\\a.png", "data:image/png;base64,AA", "missing.png", "%E0%A4%A"]) {
    assert.equal(artifactPath(bad, "doc.md", allowed), null, bad);
  }
});

test("line comparison numbers both sides and refuses oversized work", () => {
  const result = compareLines("a\nb\nc\n", "a\nB\nc\nd\n");
  assert.equal(result.kind, "rows");
  if (result.kind !== "rows") return;
  assert.deepEqual(
    result.rows.map((row) => [row.type, row.text, row.aNo, row.bNo]),
    [
      ["ctx", "a", 1, 1],
      ["del", "b", 2, null],
      ["add", "B", null, 2],
      ["ctx", "c", 3, 3],
      ["add", "d", null, 4],
      ["ctx", "", 4, 5],
    ],
  );
  assert.deepEqual([result.added, result.removed], [2, 1]);
  const crlf = compareLines("x\r\n<script>", "x\r\n<script>");
  assert.equal(crlf.kind === "rows" && crlf.added + crlf.removed, 0);
  assert.equal(compareLines("a\n".repeat(3000), "b\n".repeat(3000)).kind, "too-large");
  assert.equal(compareLines("x".repeat(10), "y", { lines: 5000, characters: 5, cells: 10 }).kind, "too-large");
  assert.equal(compareLines(Array.from({ length: 200 }, (_, i) => `a${i}`).join("\n"), Array.from({ length: 200 }, (_, i) => `b${i}`).join("\n"), { lines: 5000, characters: 1e6, cells: 100 }).kind, "too-large");
});

test("hidden and system files are skipped inside a chosen folder only", () => {
  assert.deepEqual(["a", ".git", "Thumbs.db", "DESKTOP.INI", "x.md"].map(isHiddenName), [false, true, true, true, false]);
  const file = (path: string) => Object.assign(new File(["x"], path.split("/").pop() ?? path), { markitaiPath: path });
  const chosen = selectFolderFiles([file("top/a.md"), file("top/.git/config"), file("top/sub/Thumbs.db"), file(".hidden-root/b.md")]);
  assert.deepEqual(chosen.files.map((item) => item.name), ["a.md", "b.md"]);
  assert.equal(chosen.hidden, 2);
});

test("dropped folders are walked in order, in batches, bounded and with unreadable items counted", async () => {
  const leaf = (name: string, fails = false) => ({
    name,
    isFile: true,
    isDirectory: false,
    file: (ok: (file: File) => void, no: (error: unknown) => void) => (fails ? no(new Error("denied")) : ok(new File(["x"], name))),
  });
  const dir = (name: string, children: unknown[]) => ({
    name,
    isFile: false,
    isDirectory: true,
    createReader: () => {
      const batches = [children.slice(0, 2), children.slice(2), []];
      return { readEntries: (ok: (batch: unknown[]) => void) => ok(batches.shift() ?? []) };
    },
  });
  const tree = dir("root", [leaf("b.md"), leaf(".DS_Store"), leaf("a.md"), dir("sub", [leaf("c.md"), leaf("locked.md", true)])]);
  const walked = await walkEntries([tree as never]);
  assert.deepEqual(
    walked.files.map((file) => (file as File & { markitaiPath: string }).markitaiPath),
    ["root/a.md", "root/b.md", "root/sub/c.md"],
  );
  assert.deepEqual([walked.hidden, walked.unreadable, walked.truncated], [1, 1, false]);
  const limited = await walkEntries([tree as never], { limit: 2 });
  assert.deepEqual([limited.files.length, limited.truncated], [2, true]);
});

test("a hand-typed model ID routes to the page's provider, slashes and all", () => {
  assert.equal(manualModelId("groq", " llama-3.3 "), "groq/llama-3.3");
  assert.equal(manualModelId("together_ai", "meta-llama/Llama-3-70b"), "together_ai/meta-llama/Llama-3-70b");
  assert.equal(manualModelId("openrouter", "google/gemini-x"), "openrouter/google/gemini-x");
  assert.equal(manualModelId("openrouter", "openrouter/google/gemini-x"), "openrouter/google/gemini-x");
  assert.equal(manualModelId("custom", "local-model"), "openai/local-model");
  assert.equal(manualModelId("custom", "openai/local-model"), "openai/local-model");
  assert.equal(manualModelId("azure", "my-deployment"), "azure/my-deployment");
  assert.equal(manualModelId("perplexity", "  "), "");
});

function fakeElement(tag: string, attributes: Record<string, string> = {}) {
  const values = new Map(Object.entries(attributes));
  const element = {
    tag,
    tabIndex: -1,
    className: "",
    textContent: "",
    replacedBy: null as unknown,
    getAttribute: (name: string) => values.get(name) ?? null,
    setAttribute: (name: string, value: string) => void values.set(name, value),
    removeAttribute: (name: string) => void values.delete(name),
    replaceWith(node: unknown) {
      element.replacedBy = node;
    },
    append: () => undefined,
  };
  return element;
}

test("rendered images and file links carry no token; with one, images wait to be fetched", () => {
  const render = (imageURL: (path: string) => string | null) => {
    const image = fakeElement("img", { src: "assets/a.png", alt: "" });
    const external = fakeElement("img", { src: "https://example.com/x.png" });
    const link = fakeElement("a", { href: "assets/a.png" });
    const fragment = {
      ownerDocument: { createElement: (tag: string) => fakeElement(tag) },
      querySelectorAll: (selector: string) => (selector === "img" ? [image, external] : selector === "a" ? [link] : []),
    };
    rewrite(fragment as never, {
      documentPath: "doc.md",
      artifacts: [{ relpath: "assets/a.png", size: 1 }],
      imageURL,
      fileURL: (path) => `/api/jobs/j/files/${path}`,
      placeholder: (alt) => `[${alt}]`,
      fallbackAlt: "figure",
      tableLabel: "table",
      codeLabel: "code",
    });
    return { image, external, link };
  };
  const plain = render((path) => `/api/jobs/j/files/${path}`);
  assert.equal(plain.image.getAttribute("src"), "/api/jobs/j/files/assets/a.png");
  assert.equal(plain.image.getAttribute("data-artifact-src"), null);
  assert.ok(plain.external.replacedBy, "external images stay blocked");
  assert.equal(plain.link.getAttribute("href"), "/api/jobs/j/files/assets/a.png");
  assert.equal(plain.link.getAttribute("data-artifact"), "assets/a.png");
  const held = render(() => null);
  assert.equal(held.image.getAttribute("src"), null);
  assert.equal(held.image.getAttribute("data-artifact-src"), "assets/a.png");
  assert.equal(held.image.getAttribute("alt"), "figure");
});

test("waiting images are fetched a few at a time, shown as object URLs and released", async () => {
  const images = ["a.png", "b.png", "bad.png", "c.png", "d.png", "e.png"].map((name) => fakeElement("img", { "data-artifact-src": name }));
  const root = { querySelectorAll: () => images };
  let active = 0;
  let peak = 0;
  const created: string[] = [];
  const revoked: string[] = [];
  let failed = 0;
  const loads = loadArtifactImages(
    root as never,
    async (path) => {
      active++;
      peak = Math.max(peak, active);
      await new Promise((resolve) => setTimeout(resolve, 2));
      active--;
      if (path === "bad.png") throw new Error("401");
      return new Blob([path]);
    },
    () => void failed++,
    {
      limit: 2,
      objectURL: {
        createObjectURL: () => {
          const url = `blob:${created.length}`;
          created.push(url);
          return url;
        },
        revokeObjectURL: (url: string) => void revoked.push(url),
      },
    },
  );
  await loads.done;
  assert.equal(peak, 2);
  assert.equal(created.length, 5);
  assert.equal(failed, 1);
  assert.ok(images.every((image) => image.getAttribute("data-artifact-src") === null));
  assert.equal(images[0]?.getAttribute("src"), "blob:0");
  assert.equal(images[2]?.getAttribute("src"), null);
  loads.dispose();
  assert.deepEqual(revoked, created);
});
