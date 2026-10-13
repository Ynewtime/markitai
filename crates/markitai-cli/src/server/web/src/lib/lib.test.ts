import assert from "node:assert/strict";
import test from "node:test";
import { cliCommand, shellQuote } from "./cli.ts";
import { compareLines } from "./diff.ts";
import { isHiddenName, selectFolderFiles, stagedBatch, walkEntries } from "./files.ts";
import { countWords, fmtBytes, fmtCost, fmtDateTime, fmtDur, timestampMs } from "./format.ts";
import { nameShape, shortenName, shortenShape, tailLength } from "./middle.ts";
import { artifactPath, drawingReference, drawingValue, imageByName, loadArtifactImages, markdownPair, presentationFromStyle, rewrite, splitFrontmatter } from "./markdown.ts";
import { endpointFor, manualModelId, sameEndpoint } from "./models.ts";
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
  assert.equal(cliCommand([], options), "markitai <files-or-urls> -o output/ --ocr --profile rag --no-cache --strategy static");
  assert.equal(cliCommand(["https://example.com/a b"], resolveOptions(base)), "markitai 'https://example.com/a b' -o output/ --preset minimal");
  assert.equal(shellQuote("it's"), `'it'\\''s'`);
  assert.equal(shellQuote("~/x"), "'~/x'");
});

test("a preset is named only while the request still equals it", () => {
  // Enabling enhancement leaves the minimal preset: the line names no preset it
  // would then override, and the reader's own choice is what it shows.
  assert.equal(cliCommand([], resolveOptions({ ...base, llm: true })), "markitai <files-or-urls> -o output/ --llm");
  assert.equal(cliCommand([], resolveOptions(applyPreset(base, "rich"))), "markitai <files-or-urls> -o output/ --preset rich");
  // Changing one feature of a larger preset lists what the request now means.
  const standard = resolveOptions({ ...applyPreset(base, "standard"), advanced: { ...ADVANCED_DEFAULTS, desc: false } });
  assert.equal(cliCommand([], standard), "markitai <files-or-urls> -o output/ --llm --alt");
});

test("URL lines accept bare domains and report the first unusable line", () => {
  assert.deepEqual(parseUrls("example.com/page\n\n https://a.example/x \n"), { urls: ["https://example.com/page", "https://a.example/x"], invalid: null });
  assert.deepEqual(parseUrls("https://ok.example\nftp://files.example\nnext.example").invalid, { value: "ftp://files.example", reason: "badScheme" });
  assert.deepEqual(parseUrls("not a url").invalid, { value: "not a url", reason: "badUrl" });
});

test("formatting reads well in tabular columns", () => {
  assert.deepEqual([fmtDur(240), fmtDur(4200), fmtDur(59_960), fmtDur(83_000), fmtDur(3_723_000)], ["240 ms", "4.2 s", "1:00", "1:23", "1:02:03"]);
  assert.deepEqual([fmtCost(0), fmtCost(0.0123), fmtCost(1.5)], ["$0", "$0.0123", "$1.5"]);
  assert.deepEqual([fmtBytes(512), fmtBytes(2048), fmtBytes(3 * 1024 * 1024)], ["512 B", "2.0 KB", "3.0 MB"]);
  assert.equal(timestampMs("2026-10-02T08:34:19.257123+00:00"), Date.UTC(2026, 9, 2, 8, 34, 19, 257));
  assert.equal(timestampMs("2026-10-02T10:00:00+02:00"), Date.UTC(2026, 9, 2, 8, 0, 0));
  assert.equal(timestampMs("yesterday"), null);
  assert.equal(fmtDateTime(null), "-");
  assert.equal(countWords("Hello world 你好"), 4);
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

test("a picture whose paragraph repeats its alt text is decorative", () => {
  const paragraph = { textContent: " 图2 散点图 " };
  const image = Object.assign(fakeElement("img", { src: "assets/a.png", alt: "图2 散点图" }), { closest: () => paragraph });
  const other = Object.assign(fakeElement("img", { src: "assets/a.png", alt: "封面" }), { closest: () => ({ textContent: "第一章 图表的语法" }) });
  const fragment = { ownerDocument: { createElement: (tag: string) => fakeElement(tag) }, querySelectorAll: (selector: string) => (selector === "img" ? [image, other] : []) };
  rewrite(fragment as never, {
    documentPath: "doc.md",
    artifacts: [{ relpath: "assets/a.png", size: 1 }],
    imageURL: (path) => `/x/${path}`,
    fileURL: (path) => `/x/${path}`,
    placeholder: (alt) => `[${alt}]`,
    fallbackAlt: "figure",
    tableLabel: "table",
    codeLabel: "code",
  });
  assert.equal(image.getAttribute("alt"), "");
  assert.equal(other.getAttribute("alt"), "封面");
});

test("a long file name keeps its extensions and four characters before them", () => {
  const name = "2025年度可持续发展报告（中英文对照版）——环境、社会及治理信息披露全文.pdf.llm.md";
  assert.equal(tailLength(name), "披露全文.pdf.llm.md".length);
  assert.equal(tailLength("report.final.v2.docx.md"), "inal.v2.docx.md".length);
  assert.equal(tailLength("abc"), 3);
  assert.equal(tailLength("README"), 4);
});

test("an image whose folder does not match resolves by its unique file name only", () => {
  const images = [".markitai/assets/cover.jpg", ".markitai/assets/图2-散点图.png", ".markitai/assets/a.png", "other/a.png"];
  assert.equal(imageByName("assets/cover.jpg", images), ".markitai/assets/cover.jpg");
  assert.equal(imageByName("../Images/%E5%9B%BE2-%E6%95%A3%E7%82%B9%E5%9B%BE.png", images), ".markitai/assets/图2-散点图.png");
  assert.equal(imageByName("cover.jpg?x=1", images), ".markitai/assets/cover.jpg");
  // Two listed images share the name: neither is guessed.
  assert.equal(imageByName("img/a.png", images), null);
  for (const bad of ["https://evil.example/cover.jpg", "//evil.example/cover.jpg", "/abs/cover.jpg", "..\\cover.jpg", "data:image/png;base64,AA", "missing.png", "notes.md", "%E0%A4%A"]) {
    assert.equal(imageByName(bad, images), null, bad);
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
  assert.equal(walked.folder, true);
  const limited = await walkEntries([tree as never], { limit: 2 });
  assert.deepEqual([limited.files.length, limited.truncated], [2, true]);
  // A single dropped file is not a folder, so it still converts at once.
  const single = await walkEntries([leaf("a.md") as never]);
  assert.equal(single.folder, false);
});

test("a staged batch's name says what it is; a single file needs none", () => {
  const file = (path: string) => Object.assign(new File(["x"], path.split("/").pop() ?? path), { markitaiPath: path });
  // One file, chosen with the Files entry: no batch name.
  assert.equal(stagedBatch([file("a.md")], false), null);
  // Several files, or any folder, are a batch.
  const several = stagedBatch([file("a.md"), file("b.md")], false);
  assert.equal(several?.kind, "files");
  assert.equal(several?.files.length, 2);
  const folder = stagedBatch([file("top/a.md"), file("top/sub/b.md")], true);
  assert.equal(folder?.kind, "folder");
  assert.equal(folder?.label, "top");
  // A `.urls` list is a batch of URLs, even on its own, and keeps its siblings.
  const list = stagedBatch([file("links.urls")], false);
  assert.equal(list?.kind, "urls");
  assert.equal(list?.label, "links.urls");
  const mixed = stagedBatch([file("a.md"), file("dir/links.URLS")], false);
  assert.equal(mixed?.kind, "urls");
  assert.equal(mixed?.label, "links.URLS");
  assert.deepEqual(mixed?.files.map((item) => item.name), ["a.md", "links.URLS"]);
});

test("a catalogue reads its deployments at the page's own endpoint", () => {
  // What the reader typed wins; a saved connection's address is the fallback.
  assert.equal(endpointFor(" http://127.0.0.1:8000/v1 ", "http://saved"), "http://127.0.0.1:8000/v1");
  assert.equal(endpointFor("", "http://saved"), "http://saved");
  assert.equal(endpointFor("   ", null), "");
  // A deployment the provider reaches at its default address stores none, and a
  // trailing slash is not a different address.
  assert.ok(sameEndpoint(null, "http://127.0.0.1:8000/v1"));
  assert.ok(sameEndpoint("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000/v1"));
  assert.ok(sameEndpoint("HTTP://127.0.0.1:8000/V1", "http://127.0.0.1:8000/v1"));
  assert.ok(!sameEndpoint("http://127.0.0.1:9000/v1", "http://127.0.0.1:8000/v1"));
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
    // A link the sanitizer emptied (it had a `javascript:` address).
    const bare = fakeElement("a", {});
    const blank = fakeElement("a", { href: " " });
    const fragment = {
      ownerDocument: { createElement: (tag: string) => fakeElement(tag) },
      querySelectorAll: (selector: string) => (selector === "img" ? [image, external] : selector === "a" ? [link, bare, blank] : []),
    };
    rewrite(fragment as never, {
      documentPath: "doc.md",
      artifacts: [{ relpath: "assets/a.png", size: 1 }, { relpath: "doc.md", size: 1 }],
      imageURL,
      fileURL: (path) => `/api/jobs/j/files/${path}`,
      placeholder: (alt) => `[${alt}]`,
      fallbackAlt: "figure",
      tableLabel: "table",
      codeLabel: "code",
    });
    return { image, external, link, bare, blank };
  };
  const plain = render((path) => `/api/jobs/j/files/${path}`);
  assert.equal(plain.image.getAttribute("src"), "/api/jobs/j/files/assets/a.png");
  assert.equal(plain.image.getAttribute("data-artifact-src"), null);
  assert.ok(plain.external.replacedBy, "external images stay blocked");
  assert.equal(plain.link.getAttribute("href"), "/api/jobs/j/files/assets/a.png");
  assert.equal(plain.link.getAttribute("data-artifact"), "assets/a.png");
  for (const empty of [plain.bare, plain.blank]) {
    assert.equal(empty.getAttribute("href"), null, "an empty link never points at the document");
    assert.equal(empty.getAttribute("download"), null);
  }
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
  await loads.all();
  assert.equal(peak, 2);
  assert.equal(created.length, 5);
  assert.equal(failed, 1);
  assert.ok(images.every((image) => image.getAttribute("data-artifact-src") === null));
  assert.equal(images[0]?.getAttribute("src"), "blob:0");
  assert.equal(images[2]?.getAttribute("src"), null);
  loads.dispose();
  assert.deepEqual(revoked, created);
});

test("drawing values resolve only to elements of the same drawing and never load or run anything", () => {
  const ids = new Map([["grad", "md-drawing-1-grad"], ["clip", "md-drawing-1-clip"]]);
  assert.equal(drawingValue("url(#grad)", ids), "url(#md-drawing-1-grad)");
  assert.equal(drawingValue("url( '#clip' )", ids), "url(#md-drawing-1-clip)");
  assert.equal(drawingValue("#94a3b8", ids), "#94a3b8");
  assert.equal(drawingValue("M 80 50 L 320 50", ids), "M 80 50 L 320 50");
  for (const hostile of [
    "url(#missing)", "url(https://evil.example/x.svg#grad)", "url(//evil.example/x)", "url(data:image/svg+xml,<svg/>)",
    "url(#grad) url(https://evil.example/)", "URL(\"https://evil.example/a\")", "url (https://evil.example/)",
    "image-set(\"https://evil.example/a.png\" 1x)", "-webkit-image-set(url(a.png) 1x)", "cross-fade(url(a), url(b))",
    "element(#x)", "expression(alert(1))", "javascript:alert(1)", " JavaScript:alert(1)", "data:text/html,<b>",
    "paint(worklet)", "attr(data-x url)", "env(safe-area-inset-top)",
  ]) {
    assert.equal(drawingValue(hostile, ids), null, hostile);
  }
  assert.equal(drawingReference("#grad", ids), "#md-drawing-1-grad");
  for (const hostile of ["#missing", "other.svg#grad", "https://evil.example/s.svg#grad", "javascript:alert(1)", "data:image/svg+xml,x", ""]) {
    assert.equal(drawingReference(hostile, ids), null, hostile);
  }
});

test("a drawing's style becomes drawing attributes, never references or other properties", () => {
  assert.deepEqual(presentationFromStyle("font-size:14px;font-weight:600"), [["font-size", "14px"], ["font-weight", "600"]]);
  assert.deepEqual(presentationFromStyle(" FILL : #f59e0b !important ; stroke:currentColor"), [["fill", "#f59e0b"], ["stroke", "currentColor"]]);
  assert.deepEqual(
    presentationFromStyle("fill:url(#g);background:url(https://evil.example/a.png);behavior:url(x.htc);position:fixed;stroke:image-set('a.png' 1x);color:red"),
    [["fill", "url(#g)"], ["color", "red"]],
  );
  assert.deepEqual(presentationFromStyle("fill:url(#g) url(https://evil.example/a);stroke:url(https://evil.example/s.svg#g)"), []);
  assert.deepEqual(presentationFromStyle("no declaration; :empty; fill:"), []);
});

test("an empty header cell becomes a plain cell; a named one stays a header", () => {
  const made: Array<{ tag: string; attributes: Record<string, string>; children: unknown[] }> = [];
  const cell = (text: string, image = false) => {
    const replaced: unknown[] = [];
    return {
      textContent: text,
      attributes: [{ name: "align", value: "right" }],
      childNodes: text ? [text] : [],
      querySelector: (selector: string) => (image && selector.includes("img") ? {} : null),
      replaceWith: (node: unknown) => replaced.push(node),
      replaced,
    };
  };
  const empty = cell("  ");
  const named = cell("Column B");
  const pictured = cell("", true);
  const fragment = {
    ownerDocument: {
      createElement: (tag: string) => {
        const element = { tag, attributes: {} as Record<string, string>, children: [] as unknown[], setAttribute(name: string, value: string) { element.attributes[name] = value; }, append(...nodes: unknown[]) { element.children.push(...nodes); } };
        made.push(element);
        return element;
      },
    },
    querySelectorAll: (selector: string) => (selector === "th" ? [empty, named, pictured] : []),
  };
  rewrite(fragment as never, {
    documentPath: "doc.md",
    artifacts: [],
    imageURL: () => null,
    fileURL: (path) => path,
    placeholder: (alt) => alt,
    fallbackAlt: "figure",
    tableLabel: "table",
    codeLabel: "code",
  });
  assert.equal(empty.replaced.length, 1);
  assert.deepEqual(made[0], { ...made[0], tag: "td", attributes: { align: "right" } });
  assert.equal(named.replaced.length, 0);
  assert.equal(pictured.replaced.length, 0);
});

test("a long name gives up whole folders first, keeps its host and extension, and its ellipsis touches what follows", () => {
  const chars = (limit: number) => (text: string) => Array.from(text).length <= limit;
  const cut = (name: string, kind: "file" | "url", limit: number) => {
    const short = shortenName(name, kind, chars(limit));
    return short.host + short.body;
  };
  const url = "127.0.0.1:9391/slow/annual-reports/2026/q3-statement/s5.html";
  assert.equal(cut(url, "url", 80), url);
  assert.equal(cut(url, "url", 50), "127.0.0.1:9391/slow/annual-reports/2026/…/s5.html");
  assert.equal(cut(url, "url", 40), "127.0.0.1:9391/slow/…/s5.html");
  assert.equal(cut(url, "url", 28), "127.0.0.1:9391/…/s5.html");
  assert.deepEqual(nameShape("http://127.0.0.1:41819/slow/s5.html?x=1#top", "url"), { host: "http://127.0.0.1:41819", path: "/slow/s5.html", query: "?x=1#top" });
  // A query gives way before the path.
  assert.equal(cut("http://127.0.0.1:9391/slow/s5.html?session=abcdef&x=1", "url", 40), "http://127.0.0.1:9391/slow/s5.html?…");
  // A file name loses its middle characters, never its extension; no gap or separator before the tail.
  const report = "2025年度可持续发展报告（中英文对照版）——环境、社会及治理信息披露全文.pdf.llm.md";
  assert.equal(cut(report, "file", 24), "2025年度可持…披露全文.pdf.llm.md");
  assert.equal(cut("a-very-long-document-name-here.pdf", "file", 20), "a-very-long…here.pdf");
  // A relative path (a folder upload) cuts at its folders, then its last segment.
  const path = "年度报告材料/2025/季度/第三季度/营收说明_final.docx";
  assert.equal(cut(path, "file", 30), "年度报告材料/2025/…/营收说明_final.docx");
  assert.equal(cut(path, "file", 20), "…/营收说明_final.docx");
  assert.equal(cut(path, "file", 14), "…/营收…inal.docx");
  assert.equal(cut(path, "file", 10), "…inal.docx");
  // A host the layout hides (phone rows) takes no room, and the query still goes first.
  const hidden = shortenShape({ ...nameShape("127.0.0.1:9899/archive/2025/年度报告材料/第三季度/营收说明与可持续发展信息披露全文.html?session=abcdef", "url"), host: "" }, chars(20));
  assert.equal(hidden.body, "/…/营收说明与可持…披露全文.html");
  // Only then does the host give up its middle; the result always fits when it can.
  assert.equal(cut(url, "url", 22), "127.0…1:9391/…/s5.html");
  for (const limit of [20, 30, 45, 70]) for (const name of [url, report, path]) assert.ok(Array.from(cut(name, name === url ? "url" : "file", limit)).length <= limit, `${name} at ${limit}`);
});
