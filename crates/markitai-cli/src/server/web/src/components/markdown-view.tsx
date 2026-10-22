// A result's document: Rendered, Source, Diff (only for an exact base/LLM
// pair) and Files, with the Base | LLM switch. Rendered starts with the
// front-matter card and the conversion's notices; long tables show their first
// 500 rows with a sticky header and "Show all rows"; code blocks are
// highlighted by a module loaded only when there is code. Every pane has the
// same fixed height, so switching tabs never moves the frame. Rendering is
// sanitized and pinned to the result's own files.
//
// A long document stays responsive: Rendered draws its first chunk at once and
// the rest a few chunks per frame, each chunk laid out and painted only near
// the viewport (`content-visibility`), pictures fetched only as they come
// near; a drawn document is kept for the next view of it (Details, Compare and
// back) instead of being drawn again. Source and Diff are built the first time
// they are opened, in blocks the browser also skips off screen, and Diff folds
// unchanged stretches. A document with enough headings gets its contents.
import type { ComponentChildren } from "preact";
import { createPortal } from "preact/compat";
import type { RefObject } from "preact";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { fetchBlob, fetchText, filePath, imagePath, ticketDownloads } from "../api/client.ts";
import { hasToken } from "../api/token.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { useOriginal } from "../hooks/use-original.ts";
import type { ResultDocument } from "../hooks/use-result-document.ts";
import { compareLines, foldRows, sideBySide, type Comparison, type DiffRow, type SideCell, type SideRow } from "../lib/diff.ts";
import { download, interceptDownload } from "../lib/download.ts";
import { basename, fmtBytes, localeTag, readable } from "../lib/format.ts";
import { cardFacts, parseFrontMatter } from "../lib/frontmatter.ts";
import {
  type DocumentPlan,
  type Heading,
  type ImageLoads,
  loadArtifactImages,
  countTableRows,
  emptyPlan,
  loadLibraries,
  readPiece,
  renderChunk,
  renderMarkdown,
  replacePicture,
  type RenderContext,
  splitFrontmatter,
  splitMarkdown,
} from "../lib/markdown.ts";
import { typeset } from "../lib/math.ts";
import { groupNotices } from "../lib/notices.ts";
import { cachedExcerpt, resultKey } from "../lib/results.ts";
import type { SessionItem } from "../lib/session.ts";
import { capTables, repairTables, freezeColumns, markNumericColumns, MARKER, ROW_CHUNK, ROW_LIMIT, tableSlice, type CappedTable } from "../lib/tables.ts";
import { entryAt, tocEntries } from "../lib/toc.ts";
import { FrontCard } from "./front-matter.tsx";
import { Icon, Logo } from "./icons.tsx";
import { Skeleton } from "./skeleton.tsx";
import { TocAside, TocSheet } from "./toc.tsx";
import { ViewMenu } from "./view-menu.tsx";

/** Larger sides are offered as downloads instead of being compared in the page. */
const DIFF_BYTES = 8 * 1024 * 1024;
/** Tables taller than this scroll inside their own frame, under a sticky header. */
const TALL_ROWS = 24;

export type Tab = "rendered" | "source" | "diff" | "files";

export interface RenderedState {
  printable: HTMLElement | null;
  images: ImageLoads | null;
  /** Settles once every chunk of the document is drawn. */
  complete: Promise<void> | null;
}

/** Work per slice while the rest of a document is drawn; then the page answers again. */
const SLICE_MS = 10;
/** The reading line, below the pane's top: the heading at or above it is the one being read (a
 * heading the contents scrolls to sits 12 px below the top). */
const SPY_LINE = 32;
/** What the reader does to scroll (any of these ends the contents' pin). */
const UNPIN_EVENTS = ["wheel", "touchstart", "keydown", "pointerdown"] as const;
/** Contents beside the document from this window width; below, a sheet. */
const TOC_WIDE = 900;
const TOC_KEY = "markitai.toc";
/** Drawn documents kept for their next view. */
const KEEP_DRAWN = 2;

/** A document drawn in full, kept while another view shows. */
interface Drawn {
  nodes: HTMLElement[];
  plan: DocumentPlan;
  /** The element of each of the plan's headings, once its chunk is drawn. */
  headings: (HTMLElement | null)[];
  capped: CappedTable[];
  capHosts: HTMLElement[];
  loads: ImageLoads;
  scroll: number;
}
const drawnDocuments = new Map<string, Drawn>();
function keepDrawn(key: string, drawn: Drawn): void {
  drawnDocuments.get(key)?.loads.dispose();
  drawnDocuments.delete(key);
  drawnDocuments.set(key, drawn);
  while (drawnDocuments.size > KEEP_DRAWN) {
    const [oldest, entry] = drawnDocuments.entries().next().value as [string, Drawn];
    entry.loads.dispose();
    drawnDocuments.delete(oldest);
  }
}
/** A kept document leaves the cache while a view shows it, so two views never share its nodes. */
function takeDrawn(key: string): Drawn | null {
  const drawn = drawnDocuments.get(key) ?? null;
  drawnDocuments.delete(key);
  return drawn;
}
const comparisons = new Map<string, Comparison>();

const yieldToMain = () => new Promise<void>((resolve) => setTimeout(resolve, 0));

/** A chunk's height before it is laid out, from its length and pictures, so the scroll bar is about right. */
function estimateHeight(characters: number, images: number, width: number): number {
  const perLine = Math.max(16, (width - 64) / 11);
  return Math.round((characters / perLine) * 29 + images * 300 + 24);
}

function readTocOpen(): boolean {
  try {
    return localStorage.getItem(TOC_KEY) !== "hidden";
  } catch {
    return true;
  }
}
function storeTocOpen(open: boolean): void {
  try {
    localStorage.setItem(TOC_KEY, open ? "shown" : "hidden");
  } catch {
    // A browser without storage keeps the choice for this page only.
  }
}

interface HighlightModule {
  highlightAll(root: ParentNode, done?: () => void): () => void;
}
let highlighter: Promise<HighlightModule> | null = null;
function loadHighlighter(): Promise<HighlightModule> {
  highlighter ??= import("/ui/highlight.js") as Promise<HighlightModule>;
  highlighter.catch(() => {
    highlighter = null;
  });
  return highlighter;
}

export function MarkdownView({
  t,
  locale,
  item,
  doc,
  createdAt,
  describe,
  tab,
  onTab,
  compact = false,
  rendered,
  onRendered,
  onError,
  barHost = null,
  compactBar = false,
  originals = false,
  onDownloadAll = null,
}: {
  t: Dict;
  locale: Locale;
  item: SessionItem;
  doc: ResultDocument;
  createdAt: string | null;
  describe: (error: unknown) => { text: string; detail: string };
  tab: Tab;
  onTab: (tab: Tab) => void;
  /** Compare: only Rendered and Source, no notices. */
  compact?: boolean;
  rendered?: RefObject<RenderedState>;
  /** The window's tab row lends this element to the view switch (and says when it must be compact). */
  barHost?: HTMLElement | null;
  compactBar?: boolean;
  /** Files lists the item's original too (the server keeps originals). */
  originals?: boolean;
  /** Files' Download all (.zip); null hides it. */
  onDownloadAll?: (() => void) | null;
  /** Called with the rendered body once its content is in place (and with null when it goes). */
  onRendered?: (root: HTMLElement | null) => void;
  onError?: (message: string) => void;
}) {
  const { result, pair, shown, path, markdown } = doc;
  const { descriptor: original } = useOriginal(item, originals && !compact);
  const kept = original?.download ? original : null;
  const fileCount = result ? result.artifacts.length + (kept ? 1 : 0) : undefined;
  const body = useRef<HTMLDivElement>(null);
  const printable = useRef<HTMLDivElement>(null);
  const tabs = useRef<Record<Tab, HTMLButtonElement | null>>({ rendered: null, source: null, diff: null, files: null });
  const [renderError, setRenderError] = useState<unknown>(null);
  /** The rendered body is in place; until then its skeleton holds the frame. */
  const [ready, setReady] = useState(false);
  const [diff, setDiff] = useState<Comparison | { kind: "error"; error: unknown } | null>(null);
  const [expanding, setExpanding] = useState<Record<number, number>>({});
  const capped = useRef<CappedTable[]>([]);
  /** The caption placeholder of each capped table, by index: never looked up in a large document. */
  const capHosts = useRef<HTMLElement[]>([]);
  const order: Tab[] = compact ? ["rendered", "source"] : pair ? ["rendered", "source", "diff", "files"] : ["rendered", "source", "files"];
  const active = order.includes(tab) ? tab : "rendered";
  const viewer = useRef<HTMLDivElement>(null);
  const pane = useRef<HTMLDivElement>(null);
  /** Draws every chunk up to the given one now (a contents jump past the drawn part). */
  const drawUpTo = useRef<(chunk: number) => void>(() => undefined);
  const headingEls = useRef<(HTMLElement | null)[]>([]);
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [reading, setReading] = useState(0);
  /** The heading chosen in the contents, kept marked until the reader scrolls. */
  const pinned = useRef<number | null>(null);
  const [wide, setWide] = useState(true);
  const [tocOpen, setTocOpen] = useState(readTocOpen);
  const [sheet, setSheet] = useState(false);
  const [sourceSeen, setSourceSeen] = useState(active === "source");
  const [opened, setOpened] = useState<ReadonlySet<number>>(new Set());

  const split = useMemo(() => (markdown === null ? null : splitFrontmatter(markdown)), [markdown]);
  // The card leaves out a title the document repeats as its first heading.
  const front = useMemo(() => cardFacts(parseFrontMatter(split?.frontmatter ?? null), split?.body ?? ""), [split]);
  const notices = useMemo(() => (compact ? [] : groupNotices(item.warnings.map((text) => ({ itemKey: item.key, itemName: item.name, text })))), [compact, item.warnings, item.key, item.name]);

  // ---- Rendered: the first chunk at once, the rest a few per frame
  useEffect(() => {
    const target = body.current;
    if (!target || split === null || !result) return;
    let stale = false;
    let complete = false;
    let drawn: Drawn | null = null;
    const stops: (() => void)[] = [];
    let finished: () => void = () => undefined;
    const done = new Promise<void>((resolve) => {
      finished = resolve;
    });
    if (rendered) rendered.current.complete = done;
    setRenderError(null);
    setExpanding({});
    setReading(0);
    pinned.current = null;
    const key = `${resultKey(item)}|${item.itemId}|${path}|${locale}|${hasToken() ? "header" : "url"}`;
    const placeholder = (alt: string) => t.imagePlaceholder(alt || t.imageUnavailable);
    const context: RenderContext = {
      documentPath: path,
      artifacts: result.artifacts,
      // With a token, images are fetched with the header instead of a tokened URL.
      imageURL: (relpath) => (hasToken() ? null : imagePath(item.jobId, item.itemId, relpath)),
      fileURL: (relpath) => filePath(item.jobId, relpath),
      placeholder,
      fallbackAlt: t.figureFrom(readable(item.name)),
      tableLabel: t.tableAria,
      codeLabel: t.codeAria,
    };
    // The section being read: the last heading at or above the top of the reading area. The
    // chunk at that line is found by its box (a chunk off screen is never laid out for this),
    // then only that chunk's headings are measured. After a jump from the contents the chosen
    // entry stays marked until the reader scrolls (pictures loading below it move the rest).
    let frame = 0;
    const locate = () => {
      frame = 0;
      if (pinned.current !== null) return setReading(pinned.current);
      const scroller = pane.current;
      const nodes = drawn?.nodes ?? [];
      if (!scroller || !drawn || !nodes.length) return;
      const line = scroller.getBoundingClientRect().top + SPY_LINE;
      let low = 0;
      let high = nodes.length - 1;
      let at = 0;
      while (low <= high) {
        const middle = (low + high) >> 1;
        if ((nodes[middle] as HTMLElement).getBoundingClientRect().top <= line) {
          at = middle;
          low = middle + 1;
        } else high = middle - 1;
      }
      let found = 0;
      for (const [index, heading] of drawn.plan.headings.entries()) {
        if (heading.chunk > at) break;
        const element = headingEls.current[index];
        if (heading.chunk < at || (element && element.getBoundingClientRect().top <= line)) found = index;
      }
      setReading(found);
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(locate);
    };
    // The reader's own scrolling (wheel, touch, keys, the scroll bar) ends a jump's pin.
    const unpin = () => {
      pinned.current = null;
    };
    const spying = !compact ? pane.current : null;
    spying?.addEventListener("scroll", onScroll, { passive: true });
    for (const kind of UNPIN_EVENTS) spying?.addEventListener(kind, unpin, { passive: true });
    const highlight = (root: HTMLElement) => {
      if (!root.querySelector("pre > code")) return;
      loadHighlighter().then(
        (module) => {
          if (!stale) stops.push(module.highlightAll(root));
        },
        () => undefined,
      );
    };
    const finish = () => {
      complete = true;
      if (rendered) rendered.current.images = drawn?.loads ?? null;
      setExpanding(Object.fromEntries((drawn?.capped ?? []).map((_, index) => [index, -1])));
      finished();
      onRendered?.(target);
    };

    const kept = takeDrawn(key);
    if (kept) {
      // Shown before: its nodes come back as they were, pictures included.
      drawn = kept;
      target.replaceChildren(...kept.nodes);
      capped.current = kept.capped;
      capHosts.current = kept.capHosts;
      headingEls.current = kept.headings;
      setHeadings(kept.plan.headings);
      setReady(true);
      const scroll = kept.scroll;
      requestAnimationFrame(() => {
        if (stale || !pane.current) return;
        if (scroll) pane.current.scrollTop = scroll;
        locate();
      });
      finish();
    } else {
      setReady(false);
      target.replaceChildren();
      headingEls.current = [];
      capHosts.current = [];
      const cut = capTables(repairTables(split.body));
      capped.current = cut.tables;
      loadLibraries()
        .then(async (libraries) => {
          if (stale) return;
          // Read a piece at a time as the chunks need them; the contents grow with it.
          const pieces = splitMarkdown(cut.markdown);
          const plan = emptyPlan(countTableRows(cut.markdown));
          let read = 0;
          const loads = loadArtifactImages(
            target,
            (relpath) => fetchBlob(imagePath(item.jobId, item.itemId, relpath)).then((file) => file.blob),
            (image) => void replacePicture(image, placeholder(image.getAttribute("alt") ?? "")),
            { lazy: true },
          );
          if (rendered) rendered.current.images = loads;
          const current: Drawn = { nodes: [], plan, headings: [], capped: cut.tables, capHosts: capHosts.current, loads, scroll: 0 };
          drawn = current;
          headingEls.current = current.headings;
          let listed = -1;
          const list = () => {
            if (plan.headings.length === listed) return;
            listed = plan.headings.length;
            setHeadings(plan.headings.slice());
          };
          const tables = { offset: 0 };
          const width = target.clientWidth || 760;
          let next = 0;
          const more = () => {
            while (next >= plan.chunks.length && read < pieces.length) readPiece(libraries, plan, pieces[read++] ?? "");
            return next < plan.chunks.length;
          };
          const drawOne = () => {
            const index = next++;
            const fragment = renderChunk(libraries, plan, index, context, tables);
            // A capped table's marker paragraph becomes its caption bar.
            for (const paragraph of [...fragment.querySelectorAll("p")]) {
              const text = paragraph.textContent ?? "";
              if (!text.startsWith(MARKER)) continue;
              const table = Number(text.slice(MARKER.length));
              if (!cut.tables[table] || !Number.isSafeInteger(table)) continue;
              const bar = document.createElement("div");
              bar.className = "table-cap";
              bar.setAttribute("data-cap", String(table));
              current.capHosts[table] = bar;
              paragraph.replaceWith(bar);
              bar.previousElementSibling?.classList.add("is-capped");
              setExpanding((previous) => ({ ...previous, [table]: -1 }));
            }
            for (const wrap of fragment.querySelectorAll<HTMLElement>(".table-scroll")) {
              const table = wrap.querySelector("table");
              if (!table) continue;
              markNumericColumns(table);
              if (table.rows.length > TALL_ROWS) wrap.classList.add("is-tall");
            }
            const chunk = document.createElement("div");
            chunk.className = "doc-chunk";
            chunk.style.setProperty("contain-intrinsic-size", `auto ${estimateHeight(plan.sizes[index] ?? 0, fragment.querySelectorAll("img").length, width)}px`);
            chunk.append(fragment);
            // The chunk's top-level headings, in order, are the plan's headings of this chunk.
            const wanted = plan.headings.map((heading, position) => ({ heading, position })).filter(({ heading }) => heading.chunk === index);
            const found = [...chunk.children].filter((element): element is HTMLElement => /^H[1-6]$/.test(element.tagName));
            let cursor = 0;
            for (const { heading, position } of wanted) {
              const exact = found.length === wanted.length ? found[cursor] : found.slice(cursor).find((element) => (element.textContent ?? "").trim() === heading.text);
              if (!exact) continue;
              cursor = found.indexOf(exact) + 1;
              current.headings[position] = exact;
            }
            target.append(chunk);
            void typeset(chunk);
            current.nodes.push(chunk);
            loads.add(chunk);
            highlight(chunk);
          };
          drawUpTo.current = (chunk) => {
            while (!stale && next <= chunk && more()) drawOne();
            list();
          };
          if (more()) drawOne();
          list();
          setReady(true);
          while (more()) {
            await yieldToMain();
            if (stale) return;
            const until = performance.now() + SLICE_MS;
            do drawOne();
            while (more() && performance.now() < until);
            list();
          }
          finish();
        })
        .catch((error: unknown) => !stale && setRenderError(error));
    }
    return () => {
      stale = true;
      drawUpTo.current = () => undefined;
      for (const stop of stops) stop();
      spying?.removeEventListener("scroll", onScroll);
      for (const kind of UNPIN_EVENTS) spying?.removeEventListener(kind, unpin);
      cancelAnimationFrame(frame);
      if (drawn && complete) {
        drawn.scroll = pane.current?.scrollTop ?? 0;
        keepDrawn(key, drawn);
      } else drawn?.loads.dispose();
      if (rendered && rendered.current.images === drawn?.loads) rendered.current.images = null;
      onRendered?.(null);
    };
  }, [split, path, result, item.jobId, item.name, t]);

  useEffect(() => {
    if (rendered) rendered.current.printable = printable.current;
  });

  // ---- Contents: beside the document on a wide view, a sheet otherwise
  useEffect(() => {
    const root = viewer.current;
    if (!root || compact || typeof ResizeObserver !== "function") return;
    const observer = new ResizeObserver(() => setWide(root.clientWidth >= TOC_WIDE));
    observer.observe(root);
    return () => observer.disconnect();
  }, [compact]);
  const entries = useMemo(() => (compact ? [] : tocEntries(headings)), [compact, headings]);
  const readingEntry = entryAt(entries, reading);
  const showAside = entries.length > 0 && wide && tocOpen && active === "rendered";
  useEffect(() => {
    if (!wide || active !== "rendered") setSheet(false);
  }, [wide, active]);
  /** Scroll the document to a heading, drawing the chunks before it first. */
  const goTo = (index: number) => {
    const heading = headings[index];
    const scroller = pane.current;
    if (!heading || !scroller) return;
    drawUpTo.current(heading.chunk);
    const element = headingEls.current[index];
    if (!element) return;
    const place = () => {
      scroller.scrollTop += element.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 12;
    };
    pinned.current = index;
    place();
    // Chunks above it are laid out once they come near; put it back at the top if they moved it.
    requestAnimationFrame(() => requestAnimationFrame(place));
    setReading(index);
    element.tabIndex = -1;
    element.focus({ preventScroll: true });
  };
  const toggleToc = () => {
    if (wide) {
      setTocOpen(!tocOpen);
      storeTocOpen(!tocOpen);
    } else setSheet(true);
  };

  useEffect(() => {
    if (active === "source") setSourceSeen(true);
  }, [active]);

  /** Draw a capped table's remaining rows in steps, so the page keeps answering. */
  const showAll = async (index: number) => {
    const table = capped.current[index];
    const bar = capHosts.current[index];
    const wrap = bar?.previousElementSibling;
    const tbody = wrap?.querySelector("table")?.tBodies[0];
    if (!table || !tbody || !result) return;
    const element = tbody.closest("table");
    if (element) freezeColumns(element as HTMLTableElement);
    let from = ROW_LIMIT;
    setExpanding((previous) => ({ ...previous, [index]: from }));
    while (from < table.total) {
      const to = Math.min(table.total, from + ROW_CHUNK);
      const fragment = await renderMarkdown(tableSlice(table, from, to), {
        documentPath: path,
        artifacts: result.artifacts,
        imageURL: (relpath) => (hasToken() ? null : imagePath(item.jobId, item.itemId, relpath)),
        fileURL: (relpath) => filePath(item.jobId, relpath),
        placeholder: (alt) => t.imagePlaceholder(alt || t.imageUnavailable),
        fallbackAlt: t.figureFrom(readable(item.name)),
        tableLabel: t.tableAria,
        codeLabel: t.codeAria,
      });
      if (!tbody.isConnected) return;
      const rows = fragment.querySelector("tbody");
      if (rows) {
        // The new rows take the numeric alignment of the columns they join,
        // in a row group of their own so the rows above are not laid out again.
        const numeric = [...(tbody.rows[0]?.cells ?? [])].map((cell) => cell.classList.contains("num"));
        for (const row of rows.querySelectorAll("tr")) for (let index = 0; index < row.cells.length; index++) if (numeric[index]) row.cells[index]?.classList.add("num");
        tbody.closest("table")?.append(rows);
      }
      from = to;
      setExpanding((previous) => ({ ...previous, [index]: from }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  };

  // ---- Diff
  useEffect(() => {
    if (active !== "diff" || !pair || !result) return;
    let stale = false;
    const key = `${resultKey(item)}|${pair.base}|${pair.llm}`;
    const known = comparisons.get(key);
    setOpened(new Set());
    if (known) {
      setDiff(known);
      return;
    }
    setDiff(null);
    const sizes = new Map(result.artifacts.map((artifact) => [artifact.relpath, artifact.size]));
    if ((sizes.get(pair.base) ?? 0) > DIFF_BYTES || (sizes.get(pair.llm) ?? 0) > DIFF_BYTES) {
      setDiff({ kind: "too-large" });
      return;
    }
    const read = (relpath: string) => {
      const known = doc.text(relpath);
      return known !== null ? Promise.resolve(known) : fetchText(filePath(item.jobId, relpath), DIFF_BYTES);
    };
    Promise.all([read(pair.base), read(pair.llm)]).then(
      ([base, llm]) => {
        if (stale) return;
        const comparison = compareLines(base, llm);
        comparisons.set(key, comparison);
        while (comparisons.size > KEEP_DRAWN) comparisons.delete(comparisons.keys().next().value as string);
        setDiff(comparison);
      },
      (error: unknown) => {
        if (stale) return;
        setDiff(error instanceof RangeError ? { kind: "too-large" } : { kind: "error", error });
      },
    );
    return () => {
      stale = true;
    };
  }, [active, pair, result, item.jobId]);

  const onTabKey = (event: KeyboardEvent) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight" && event.key !== "Home" && event.key !== "End") return;
    event.preventDefault();
    const index = order.indexOf(active);
    const next =
      event.key === "Home" ? order[0] : event.key === "End" ? order[order.length - 1] : order[(index + (event.key === "ArrowRight" ? 1 : order.length - 1)) % order.length];
    onTab(next ?? "rendered");
    tabs.current[next ?? "rendered"]?.focus();
  };
  const prefix = compact ? "cmp" : "pv";
  const tabButton = (value: Tab, label: string, count?: number) => (
    <button
      ref={(node) => {
        tabs.current[value] = node;
      }}
      type="button"
      role="tab"
      id={`${prefix}-tab-${value}`}
      aria-selected={active === value}
      aria-controls={`${prefix}-pane-${value}`}
      tabIndex={active === value ? 0 : -1}
      class={active === value ? "tab is-on" : "tab"}
      onClick={() => onTab(value)}
      onKeyDown={onTabKey}
    >
      {label}
      {count !== undefined && <span class="tab-count">{count}</span>}
    </button>
  );

  const failure = doc.error ?? renderError;
  // While the result loads: its size when the inspector already read its start.
  const known = cachedExcerpt(resultKey(item));
  const bytes = result?.markdown_bytes ?? known?.markdown_bytes ?? known?.artifacts.find((artifact) => artifact.relpath === item.output)?.size ?? null;
  const loadingLabel = result === null ? t.previewLoading(bytes !== null && bytes > 256 * 1024 ? fmtBytes(bytes) : null) : t.previewRendering;
  const note = (error: unknown, kind: "doc" | "source" | "list" = "doc") =>
    error ? <p class="pane-note line-error">{describe(error).text}</p> : <Skeleton label={loadingLabel} kind={kind} />;
  const docName = readable(basename(path || item.name)).replace(/\.llm\.md$/i, "").replace(/\.md$/i, "");
  // An upload keeps its name; a web page's snapshot is saved as the result's name in HTML.
  const originalName = item.kind === "url" ? `${docName}.html` : basename(kept?.name || item.name);
  const onDocumentLink = (event: MouseEvent) => {
    const link = (event.target as Element | null)?.closest?.("a[data-artifact]");
    const relpath = link?.getAttribute("data-artifact");
    if (!relpath) return;
    interceptDownload(event, filePath(item.jobId, relpath), basename(relpath), (error) => onError?.(describe(error).text));
  };
  const words = new Intl.NumberFormat(localeTag(locale));

  // Diff compares both versions: the switch stays, with both on; choosing one shows it rendered.
  const versioned = pair !== null && (active === "rendered" || active === "source" || active === "diff");
  const comparing = active === "diff";
  // Wide panes show the two versions side by side; narrow ones, one column of changes.
  const sideBySideDiff = wide;
  const contentsToggle = entries.length > 0 && wide && active === "rendered";
  const tabLabels: Record<Tab, string> = { rendered: t.rendered, source: t.source, diff: t.diffTab, files: t.filesTab };
  // Rendered | Source | Diff | Files with Base | LLM: in the window's tab row when it lends one (barHost).
  const bar = (
    <>
      <div class="tabs" role="tablist" aria-label={t.previewAria}>
        {tabButton("rendered", t.rendered)}
        {tabButton("source", t.source)}
        {!compact && pair && tabButton("diff", t.diffTab)}
        {!compact && tabButton("files", t.filesTab, fileCount)}
      </div>
      {contentsToggle || versioned ? (
        <span class="viewer-side">
          {contentsToggle && (
            <button
              type="button"
              class={showAside ? "icon-btn toc-toggle is-on" : "icon-btn toc-toggle"}
              aria-label={t.tocTitle}
              title={tocOpen ? t.tocHide : t.tocShow}
              aria-pressed={tocOpen}
              onClick={toggleToc}
            >
              <Icon name="ListBullets" size={16} />
            </button>
          )}
          {versioned && (
            <span class="seg" role="group" aria-label={t.versionAria}>
              {(["base", "llm"] as const).map((value) => (
                <button
                  key={value}
                  type="button"
                  class={comparing || shown === value ? "is-on" : undefined}
                  aria-pressed={comparing || shown === value}
                  onClick={() => {
                    doc.setVersion(value);
                    if (comparing) onTab("rendered");
                  }}
                >
                  {value === "base" ? t.baseTag : t.llmTag}
                </button>
              ))}
            </span>
          )}
        </span>
      ) : null}
    </>
  );
  // A row too narrow for all of them: one button naming the view, with the views and versions in its menu.
  const viewMenu = (
    <ViewMenu
      label={t.previewAria}
      views={order.map((value) => ({ value, label: tabLabels[value], count: value === "files" ? fileCount : undefined }))}
      view={active}
      onView={onTab}
      versions={pair ? [{ value: "base", label: t.baseTag }, { value: "llm", label: t.llmTag }] : []}
      versionLabel={t.versionAria}
      version={versioned && !comparing ? shown : null}
      onVersion={(value) => doc.setVersion(value as "base" | "llm")}
      contents={contentsToggle ? { label: t.tocTitle, on: tocOpen, toggle: toggleToc } : null}
    />
  );

  return (
    <div ref={viewer} class={compact ? "viewer is-compact" : "viewer"}>
      {barHost ? createPortal(compactBar ? viewMenu : bar, barHost) : <div class="viewer-bar">{bar}</div>}

      <div class="viewer-main">
      {showAside && <TocAside t={t} entries={entries} active={readingEntry} onPick={goTo} />}
      <div ref={pane} id={`${prefix}-pane-rendered`} role="tabpanel" aria-labelledby={`${prefix}-tab-rendered`} class="pane" tabIndex={0} hidden={active !== "rendered"}>
        {markdown === null || renderError ? (
          note(failure)
        ) : (
          <>
          {!ready && <Skeleton label={t.previewRendering} />}
          <div ref={printable} class={ready ? "pdf-doc" : "pdf-doc is-pending"}>
            <div class="print-head" aria-hidden="true">
              <span class="print-brand">
                <Logo size={20} />
                <strong>Markitai</strong>
              </span>
              <span class="print-title">{docName}</span>
            </div>
            <div class="doc">
              {notices.length > 0 && (
                <details class="doc-notices">
                  <summary>
                    <Icon name="Warning" size={14} />
                    <span>{t.noticesHeading(notices.reduce((sum, group) => sum + group.count, 0))}</span>
                    <span class="doc-notices-first">{t.noticeTitle(notices[0]?.kind ?? "other", notices[0]?.count ?? 0, notices[0]?.models ?? [], notices[0]?.sample ?? "", notices[0]?.pages ?? null)}</span>
                  </summary>
                  <ul>
                    {notices.map((group) => (
                      <li key={group.key}>
                        {t.noticeTitle(group.kind, group.count, group.models, group.sample, group.pages)}
                        {group.kind === "other" && group.count > 1 && group.pages === null && ` ×${group.count}`}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {front && <FrontCard t={t} front={front} />}
              <div ref={body} class="doc-body" onClick={onDocumentLink} onAuxClick={onDocumentLink} />
              {Object.entries(expanding).map(([key, from]) => {
                const index = Number(key);
                const table = capped.current[index];
                const bar = capHosts.current[index];
                if (!table || !bar?.isConnected) return null;
                return (
                  <CapBar key={key} t={t} words={words} host={bar} total={table.total} shown={from < 0 ? ROW_LIMIT : from} busy={from >= 0 && from < table.total} onShowAll={() => void showAll(index)} onSource={() => onTab("source")} />
                );
              })}
            </div>
            <div class="print-foot" aria-hidden="true">
              <span>{t.pdfPreparedBy}</span>
              <span class="print-source">
                {t.pdfSource}: {readable(item.name)}
                {createdAt !== null && ` · ${createdAt.slice(0, 10)}`}
              </span>
            </div>
          </div>
          </>
        )}
      </div>

      <div id={`${prefix}-pane-source`} role="tabpanel" aria-labelledby={`${prefix}-tab-source`} class="pane is-source" tabIndex={0} hidden={active !== "source"}>
        {markdown === null || split === null ? (
          note(failure, "source")
        ) : sourceSeen ? (
          <SourceText label={t.srcAria} frontmatter={split.frontmatter} body={split.body} />
        ) : null}
      </div>

      {!compact && pair && (
        <div id={`${prefix}-pane-diff`} role="tabpanel" aria-labelledby={`${prefix}-tab-diff`} class="pane" tabIndex={0} hidden={active !== "diff"}>
          {diff === null ? (
            <Skeleton label={t.loading} />
          ) : diff.kind === "error" ? (
            note(diff.error)
          ) : diff.kind === "too-large" ? (
            <p class="pane-note">{t.diffTooLarge}</p>
          ) : (
            <div class={sideBySideDiff ? "diff is-split" : "diff"} aria-label={t.diffAria}>
              <p class="doc-meta diff-meta">
                {basename(pair.base)} → {basename(pair.llm)} · +{diff.added} -{diff.removed}
                {diff.added === 0 && diff.removed === 0 && ` · ${t.diffSame}`}
              </p>
              {foldRows(diff.rows).map((segment, index) =>
                segment.kind === "rows" || opened.has(index) ? (
                  sideBySideDiff ? <DiffSplit key={index} rows={segment.rows} marks={{ add: t.diffAdded, del: t.diffRemoved }} /> : <DiffRows key={index} rows={segment.rows} />
                ) : (
                  <button key={index} type="button" class="diff-fold" onClick={() => setOpened((previous) => new Set(previous).add(index))}>
                    <Icon name="CaretDown" size={13} />
                    {t.diffFold(words.format(segment.rows.length))}
                  </button>
                ),
              )}
            </div>
          )}
        </div>
      )}

      {!compact && (
        <div id={`${prefix}-pane-files`} role="tabpanel" aria-labelledby={`${prefix}-tab-files`} class="pane" tabIndex={0} hidden={active !== "files"}>
          {result === null ? (
            note(doc.error, "list")
          ) : result.artifacts.length === 0 && !kept ? (
            <p class="pane-note">{t.filesNone}</p>
          ) : (
            <div class="files">
              <ul class="file-list" aria-label={t.filesAria}>
                {kept && <OriginalRow t={t} item={item} original={kept} name={originalName} onError={(error) => onError?.(describe(error).text)} />}
                {result.artifacts.map((artifact) => (
                  <li key={artifact.relpath}>
                    <Icon name={/\.(png|jpe?g|gif|webp|avif|svg)$/i.test(artifact.relpath) ? "Image" : "FileText"} size={14} />
                    <span class="file-path" title={readable(artifact.relpath)}>
                      {readable(artifact.relpath)}
                    </span>
                    <span class="file-size">{fmtBytes(artifact.size)}</span>
                    <a
                      class="icon-btn"
                      href={filePath(item.jobId, artifact.relpath)}
                      download={readable(basename(artifact.relpath))}
                      aria-label={t.downloadFile(readable(artifact.relpath))}
                      title={t.downloadFile(readable(artifact.relpath))}
                      onClick={(event) => interceptDownload(event, filePath(item.jobId, artifact.relpath), readable(basename(artifact.relpath)), (error) => onError?.(describe(error).text))}
                      onAuxClick={(event) => interceptDownload(event, filePath(item.jobId, artifact.relpath), readable(basename(artifact.relpath)), (error) => onError?.(describe(error).text))}
                    >
                      <Icon name="DownloadSimple" size={15} />
                    </a>
                  </li>
                ))}
              </ul>
              <div class="files-foot">
                <span class="files-total">{t.filesTotal(fileCount ?? 0, fmtBytes(result.artifacts.reduce((sum, artifact) => sum + artifact.size, kept?.size ?? 0)))}</span>
                {onDownloadAll && (
                  <button type="button" class="btn" onClick={onDownloadAll}>
                    <Icon name="DownloadSimple" size={14} />
                    {t.downloadAllZip}
                  </button>
                )}
              </div>
            </div>
          )}
        </div>
      )}
      </div>
      {/* A phone or a narrow window: the contents open from a button over the document's corner. */}
      {entries.length > 0 && !wide && active === "rendered" && (
        <button type="button" class="toc-fab" aria-label={t.tocTitle} title={t.tocTitle} aria-haspopup="dialog" aria-expanded={sheet} onClick={toggleToc}>
          <Icon name="ListBullets" size={18} />
        </button>
      )}
      {sheet && entries.length > 0 && (
        <TocSheet
          t={t}
          entries={entries}
          active={readingEntry}
          onPick={(index) => {
            setSheet(false);
            requestAnimationFrame(() => goTo(index));
          }}
          onClose={() => setSheet(false)}
        />
      )}
    </div>
  );
}

/** Files' first row: what was converted, as the server kept it. */
function OriginalRow({ t, item, original, name, onError }: { t: Dict; item: SessionItem; original: { download: string | null; size: number | null }; name: string; onError: (error: unknown) => void }) {
  const path = original.download ?? "";
  return (
    <li class="is-original">
      <Icon name={item.kind === "url" ? "Globe" : "FileText"} size={14} />
      <span class="file-path" title={readable(item.kind === "url" ? item.name : name)}>
        {readable(name)}
      </span>
      <span class="file-tag">{item.kind === "url" ? t.filesOriginalPage : t.original}</span>
      <span class="file-size">{original.size != null ? fmtBytes(original.size) : ""}</span>
      <button
        type="button"
        class="icon-btn"
        aria-label={t.downloadFile(readable(name))}
        title={t.downloadFile(readable(name))}
        // An upload can be large: the browser streams it through a ticket when the server issues them.
        onClick={() => void download(path, readable(name), ticketDownloads()).catch(onError)}
      >
        <Icon name="DownloadSimple" size={15} />
      </button>
    </li>
  );
}

/** Lines per block of Source and Diff: a block off screen is not laid out. */
const BLOCK_LINES = 200;

/** The Markdown as written, in blocks of lines the browser lays out only near the viewport. */
function SourceText({ label, frontmatter, body }: { label: string; frontmatter: string | null; body: string }) {
  const blocks = useMemo(() => {
    const lines = (frontmatter !== null ? body.replace(/^\n/, "") : body).split("\n");
    const out: string[] = [];
    for (let index = 0; index < lines.length; index += BLOCK_LINES) out.push(lines.slice(index, index + BLOCK_LINES).join("\n"));
    return out;
  }, [frontmatter, body]);
  return (
    <pre class="source" aria-label={label}>
      {frontmatter !== null && <span class="fm source-block">{frontmatter}</span>}
      {blocks.map((text, index) => (
        <span key={index} class="source-block" style={{ containIntrinsicSize: `auto ${Math.max(1, Math.ceil(text.length / 60)) * 23}px` }}>
          {text}
        </span>
      ))}
    </pre>
  );
}

/** The diff side by side: the base on the left, the LLM version on the right, a change's lines
 * beside each other with the words that changed marked; blocks of rows the browser skips off screen. */
function DiffSplit({ rows, marks }: { rows: DiffRow[]; marks: { add: string; del: string } }) {
  const sides = useMemo(() => sideBySide(rows), [rows]);
  const blocks: SideRow[][] = [];
  for (let index = 0; index < sides.length; index += BLOCK_LINES) blocks.push(sides.slice(index, index + BLOCK_LINES));
  const cell = (side: SideCell | null, which: "a" | "b") =>
    side === null ? (
      <span class={`dsplit-cell is-${which} is-empty`} aria-hidden="true">
        <span class="diff-no" />
        <span class="diff-text" />
      </span>
    ) : (
      <span class={side.type === "ctx" ? `dsplit-cell is-${which}` : `dsplit-cell is-${which} is-${side.type}`}>
        <span class="diff-no" aria-hidden="true">
          {side.no}
        </span>
        {side.type !== "ctx" && <span class="sr-only">{side.type === "add" ? marks.add : marks.del} </span>}
        <span class="diff-text">{side.parts ? side.parts.map((part, index) => (part.changed ? <mark key={index}>{part.text}</mark> : part.text)) : side.text}</span>
      </span>
    );
  return (
    <>
      {blocks.map((block, index) => (
        <div key={index} class="diff-block">
          {block.map((row, line) => (
            <div key={line} class="dsplit-row">
              {cell(row.left, "a")}
              {cell(row.right, "b")}
            </div>
          ))}
        </div>
      ))}
    </>
  );
}

/** Diff lines in blocks; `marks` names an added or removed line for screen readers. */
export function DiffRows({ rows, marks }: { rows: DiffRow[]; marks?: { add: string; del: string } }) {
  const blocks: DiffRow[][] = [];
  for (let index = 0; index < rows.length; index += BLOCK_LINES) blocks.push(rows.slice(index, index + BLOCK_LINES));
  return (
    <>
      {blocks.map((block, index) => (
        <div key={index} class="diff-block">
          {block.map((row, line) => (
            <div key={line} class={row.type === "add" ? "diff-line is-add" : row.type === "del" ? "diff-line is-del" : "diff-line"}>
              <span class="diff-no" aria-hidden="true">
                {row.aNo ?? ""}
              </span>
              <span class="diff-no" aria-hidden="true">
                {row.bNo ?? ""}
              </span>
              <span class="diff-mark" aria-hidden="true">
                {row.type === "add" ? "+" : row.type === "del" ? "-" : " "}
              </span>
              {marks && row.type !== "ctx" && <span class="sr-only">{row.type === "add" ? marks.add : marks.del} </span>}
              <span class="diff-text">{row.text}</span>
            </div>
          ))}
        </div>
      ))}
    </>
  );
}

/** The caption under a capped table, drawn into the placeholder the renderer left there. */
function CapBar({
  t,
  words,
  host,
  total,
  shown,
  busy,
  onShowAll,
  onSource,
}: {
  t: Dict;
  words: Intl.NumberFormat;
  host: HTMLElement;
  total: number;
  shown: number;
  busy: boolean;
  onShowAll: () => void;
  onSource: () => void;
}) {
  const [, setTick] = useState(0);
  useEffect(() => setTick((value) => value + 1), [host]);
  const complete = shown >= total;
  const content = (
    <>
      <span role="status">{complete ? t.tableAllShown(words.format(total)) : t.tableCapped(words.format(Math.min(shown, total)), words.format(total))}</span>
      {!complete && (
        <>
          <button type="button" class="btn btn-sm" disabled={busy} aria-busy={busy || undefined} onClick={onShowAll}>
            {busy ? t.tableShowing : t.tableShowAll}
          </button>
          <button type="button" class="text-link" onClick={onSource}>
            {t.tableOpenSource}
          </button>
        </>
      )}
    </>
  );
  return <Portal host={host}>{content}</Portal>;
}

function Portal({ host, children }: { host: HTMLElement; children: ComponentChildren }) {
  return createPortal(children, host);
}
