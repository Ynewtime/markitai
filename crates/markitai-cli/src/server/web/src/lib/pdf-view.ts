// The PDF viewer of Compare: zoom (fit width, fit page or a percentage), where
// each page sits at that zoom, which pages are in view, the page being read,
// a reading position that survives a zoom, the pixel width a page is drawn at,
// and the order pages are asked for (two at a time, nearest first, as the
// server draws them). Pure layout and scheduling; the component draws.

export type Zoom = "width" | "page" | number;
export interface Size {
  width: number;
  height: number;
}

/** The percentages offered (100 % is the page's printed size on a 96 dpi screen, as browsers' viewers). */
export const ZOOM_LEVELS = [50, 75, 100, 125, 150, 200] as const;
/** CSS pixels per PDF point at 100 %. */
export const PT_TO_PX = 96 / 72;
/** Space around the pages and between them (CSS pixels). */
export const PAGE_PAD = 16;
export const PAGE_GAP = 12;
/** The widest page image the server draws (`width` is clamped to 160–2400). */
export const MAX_RENDER_WIDTH = 2400;
const MIN_RENDER_WIDTH = 200;
/** The smallest area a fit leaves for a page, so a tiny pane still shows something. */
const MIN_FIT = 80;
const LETTER: Size = { width: 612, height: 792 };

/** Page boxes the layout can use: one the server could not measure (0 × 0) takes its neighbours' usual size. */
export function usableSizes(pages: readonly Size[]): Size[] {
  const good = pages.filter((page) => page.width > 0 && page.height > 0);
  const usual = good.length ? ([...good].sort((a, b) => a.width * a.height - b.width * b.height)[Math.floor(good.length / 2)] as Size) : LETTER;
  return pages.map((page) => (page.width > 0 && page.height > 0 ? page : usual));
}

/** CSS pixels per point of each page for a zoom in a view: a percentage scales every page alike; fit width
 * fits each page's width to the view, fit page each whole page (as the pages fitted the pane before the viewer). */
export function scalesFor(zoom: Zoom, view: Size, pages: readonly Size[]): number[] {
  const width = Math.max(MIN_FIT, view.width - 2 * PAGE_PAD);
  const height = Math.max(MIN_FIT, view.height - 2 * PAGE_PAD);
  return pages.map((page) => {
    const scale = typeof zoom === "number" ? (zoom / 100) * PT_TO_PX : zoom === "page" ? Math.min(width / page.width, height / page.height) : width / page.width;
    return Math.min(10, Math.max(0.05, scale));
  });
}

/** The zoom a scale shows, as a whole percentage (fit modes read as one in the zoom menu's tooltip and for steps). */
export const percentOf = (scale: number): number => Math.round((scale / PT_TO_PX) * 100);

export interface PageLayout {
  /** CSS pixels per point, per page. */
  scales: number[];
  tops: number[];
  lefts: number[];
  widths: number[];
  heights: number[];
  /** The scrolled area. */
  width: number;
  height: number;
}

/** Where each page sits: one column, centred when narrower than the view, from the left edge when wider. */
export function layoutPages(pages: readonly Size[], scale: number | readonly number[], viewWidth: number): PageLayout {
  const scales = pages.map((_, index) => (typeof scale === "number" ? scale : (scale[index] ?? 1)));
  const widths = pages.map((page, index) => Math.max(1, Math.round(page.width * (scales[index] as number))));
  const heights = pages.map((page, index) => Math.max(1, Math.round(page.height * (scales[index] as number))));
  const widest = widths.reduce((most, width) => Math.max(most, width), 0);
  const width = Math.max(viewWidth, widest + 2 * PAGE_PAD);
  const lefts = widths.map((page) => Math.max(PAGE_PAD, Math.floor((width - page) / 2)));
  const tops: number[] = [];
  let y = PAGE_PAD;
  for (const height of heights) {
    tops.push(y);
    y += height + PAGE_GAP;
  }
  return { scales, tops, lefts, widths, heights, width, height: pages.length ? y - PAGE_GAP + PAGE_PAD : 0 };
}

/** The page (0-based) whose slot (the page and the gap after it) holds `y`. */
export function pageAt(layout: PageLayout, y: number): number {
  const { tops } = layout;
  if (!tops.length) return 0;
  let low = 0;
  let high = tops.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if ((tops[middle] as number) <= y) low = middle;
    else high = middle - 1;
  }
  return low;
}

/** The pages (0-based, inclusive) within `margin` pixels of the view. */
export function pagesNear(layout: PageLayout, top: number, height: number, margin: number): [number, number] {
  if (!layout.tops.length) return [0, -1];
  return [pageAt(layout, Math.max(0, top - margin)), pageAt(layout, top + height + margin)];
}

/** The page being read (1-based): the one that fills most of the view; at the end of the document, the last one in view. */
export function pageInView(layout: PageLayout, top: number, height: number): number {
  const count = layout.tops.length;
  if (!count) return 1;
  const bottom = top + height;
  if (bottom >= layout.height - 2) return pageAt(layout, bottom - PAGE_PAD - 1) + 1;
  const [first, last] = pagesNear(layout, top, height, 0);
  let best = first;
  let shown = -1;
  for (let index = first; index <= last; index++) {
    const start = layout.tops[index] as number;
    const visible = Math.min(bottom, start + (layout.heights[index] as number)) - Math.max(top, start);
    if (visible > shown + 0.5) {
      best = index;
      shown = visible;
    }
  }
  return best + 1;
}

/** A reading position independent of the zoom: a point of a page (its fractions) and where the view shows it. */
export interface Anchor {
  page: number;
  fx: number;
  fy: number;
  /** The point's place in the view, CSS pixels from its top-left corner. */
  vx: number;
  vy: number;
}

/** The position at view point (vx, vy); without a point, the middle of the visible part of the page being read. */
export function anchorOf(layout: PageLayout, scroll: { top: number; left: number }, view: Size, point?: { vx: number; vy: number }): Anchor {
  let vx = point?.vx ?? view.width / 2;
  let vy = point?.vy;
  if (vy === undefined) {
    const index = pageInView(layout, scroll.top, view.height) - 1;
    const start = (layout.tops[index] ?? 0) - scroll.top;
    const end = start + (layout.heights[index] ?? 0);
    vy = (Math.max(0, start) + Math.min(view.height, end)) / 2;
  }
  const page = pageAt(layout, scroll.top + vy);
  const height = layout.heights[page] ?? 1;
  const width = layout.widths[page] ?? 1;
  vx = Math.max(0, vx);
  return {
    page,
    fy: (scroll.top + vy - (layout.tops[page] ?? 0)) / height,
    fx: (scroll.left + vx - (layout.lefts[page] ?? 0)) / width,
    vx,
    vy,
  };
}

/** The scroll position that puts an anchor's point back at its place in the view. */
export function scrollFor(layout: PageLayout, anchor: Anchor, view: Size): { top: number; left: number } {
  const page = Math.min(Math.max(0, anchor.page), layout.tops.length - 1);
  const top = (layout.tops[page] ?? 0) + anchor.fy * (layout.heights[page] ?? 0) - anchor.vy;
  const left = (layout.lefts[page] ?? 0) + anchor.fx * (layout.widths[page] ?? 0) - anchor.vx;
  return {
    top: Math.round(Math.min(Math.max(0, top), Math.max(0, layout.height - view.height))),
    left: Math.round(Math.min(Math.max(0, left), Math.max(0, layout.width - view.width))),
  };
}

/** The next percentage up or down from the zoom on screen (a fit mode steps from the percentage it shows). */
export function zoomStep(shown: number, direction: 1 | -1): number {
  if (direction > 0) return ZOOM_LEVELS.find((level) => level > shown + 1) ?? (ZOOM_LEVELS[ZOOM_LEVELS.length - 1] as number);
  return [...ZOOM_LEVELS].reverse().find((level) => level < shown - 1) ?? ZOOM_LEVELS[0];
}

/** The image width to ask for a page shown `css` pixels wide: device pixels, rounded up to 200 so answers cache, within the server's bounds. */
export function renderWidth(css: number, ratio: number): number {
  const wanted = Math.ceil((css * (ratio > 0 ? ratio : 1)) / 200) * 200;
  return Math.min(MAX_RENDER_WIDTH, Math.max(MIN_RENDER_WIDTH, wanted));
}

/** What Ctrl/⌘ with a key means to the viewer, or null. */
export function zoomKey(event: Pick<KeyboardEvent, "key" | "code" | "ctrlKey" | "metaKey" | "altKey">): "in" | "out" | "reset" | null {
  if (!(event.ctrlKey || event.metaKey) || event.altKey) return null;
  if (event.key === "+" || event.key === "=" || event.code === "NumpadAdd") return "in";
  if (event.key === "-" || event.key === "_" || event.code === "NumpadSubtract") return "out";
  if (event.key === "0" || event.code === "Numpad0") return "reset";
  return null;
}

/** A typed page number (full-width digits too), when it names a page from 1 to `count`. */
export function typedPage(text: string, count: number): number | null {
  const digits = text.normalize("NFKC").trim();
  if (!/^\d{1,7}$/.test(digits)) return null;
  const page = Number(digits);
  return page >= 1 && page <= count ? page : null;
}

/** A remembered zoom, when it is one. */
export function readZoom(value: unknown): Zoom | null {
  if (value === "width" || value === "page") return value;
  return typeof value === "number" && (ZOOM_LEVELS as readonly number[]).includes(value) ? value : null;
}

const ZOOM_STORE = "markitai.pdfZoom";
const ZOOMS_KEPT = 200;
const zooms = new Map<string, Zoom>();

/** Each item's zoom for this browser tab's session (kept across reloads of the tab). */
export function rememberedZoom(key: string): Zoom {
  if (zooms.has(key)) return zooms.get(key) as Zoom;
  try {
    const stored = JSON.parse(sessionStorage.getItem(ZOOM_STORE) ?? "{}") as Record<string, unknown>;
    return readZoom(stored[key]) ?? "width";
  } catch {
    return "width";
  }
}

export function rememberZoom(key: string, zoom: Zoom): void {
  zooms.delete(key);
  zooms.set(key, zoom);
  try {
    const stored = JSON.parse(sessionStorage.getItem(ZOOM_STORE) ?? "{}") as Record<string, unknown>;
    delete stored[key];
    stored[key] = zoom;
    const keys = Object.keys(stored);
    for (const old of keys.slice(0, Math.max(0, keys.length - ZOOMS_KEPT))) delete stored[old];
    sessionStorage.setItem(ZOOM_STORE, JSON.stringify(stored));
  } catch {
    // Storage refused (private mode, quota): the zoom is kept in memory for this page.
  }
}

/** One image to fetch: a page at a width; `rank` orders the queue (lower first). */
export interface PageTask {
  key: string;
  rank: number;
}

/**
 * Asks for page images at most `limit` at a time, the lowest rank first. `want` replaces the
 * wanted set (a task no longer wanted is not started; one running finishes). A busy answer is
 * asked again a little later, a few times; `run` says whether the server was busy.
 */
export class PageQueue<T extends PageTask> {
  private wanted = new Map<string, T>();
  private running = new Set<string>();
  private tries = new Map<string, number>();
  private waiting = new Set<string>();
  private closed = false;

  private readonly run: (task: T) => Promise<"done" | "busy" | "failed">;
  private readonly onGiveUp: (task: T) => void;
  private readonly limit: number;
  private readonly retries: number;
  private readonly delay: (attempt: number) => number;

  constructor(
    run: (task: T) => Promise<"done" | "busy" | "failed">,
    onGiveUp: (task: T) => void,
    limit = 2,
    retries = 4,
    delay: (attempt: number) => number = (attempt) => 400 * attempt,
  ) {
    this.run = run;
    this.onGiveUp = onGiveUp;
    this.limit = limit;
    this.retries = retries;
    this.delay = delay;
  }

  want(tasks: readonly T[]): void {
    this.wanted = new Map(tasks.map((task) => [task.key, task]));
    this.pump();
  }

  close(): void {
    this.closed = true;
    this.wanted.clear();
  }

  /** In flight now (for tests). */
  get active(): number {
    return this.running.size;
  }

  private pump(): void {
    if (this.closed) return;
    const next = [...this.wanted.values()].filter((task) => !this.running.has(task.key) && !this.waiting.has(task.key)).sort((a, b) => a.rank - b.rank);
    for (const task of next) {
      if (this.running.size >= this.limit) break;
      this.start(task);
    }
  }

  private start(task: T): void {
    this.running.add(task.key);
    this.wanted.delete(task.key);
    const finish = (outcome: "done" | "busy" | "failed") => {
      this.running.delete(task.key);
      if (this.closed) return;
      if (outcome === "busy") {
        const attempt = (this.tries.get(task.key) ?? 0) + 1;
        this.tries.set(task.key, attempt);
        if (attempt > this.retries) {
          this.onGiveUp(task);
        } else {
          this.waiting.add(task.key);
          setTimeout(() => {
            this.waiting.delete(task.key);
            if (this.closed) return;
            if (!this.wanted.has(task.key)) this.wanted.set(task.key, task);
            this.pump();
          }, this.delay(attempt));
        }
      } else {
        this.tries.delete(task.key);
        if (outcome === "failed") this.onGiveUp(task);
      }
      this.pump();
    };
    this.run(task).then(finish, () => finish("failed"));
  }
}
