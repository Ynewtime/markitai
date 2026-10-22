// Compare's PDF original, read like a browser's PDF viewer: a toolbar in the
// original's header (outline, previous / page box / next, zoom out / zoom
// menu / zoom in), pages drawn by the server at the width the zoom and the
// screen's pixel ratio need, only near the view, two at a time, and an outline
// panel (the PDF's bookmarks, else the Markdown's headings, else thumbnails)
// beside the pages, or over them in a narrow pane. Ctrl/⌘ + − 0 (and Ctrl +
// wheel) zoom while the pane has focus or the pointer; PageUp/PageDown turn
// pages. The outlines linking the Markdown's blocks to the page sit in page
// percentages, so they follow every zoom.
import type { ComponentChildren, VNode } from "preact";
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "preact/hooks";
import { ApiError, fetchBlob, fetchPageText, originalPagePath } from "../api/client.ts";
import type { OriginalDescriptor } from "../api/types.ts";
import type { ResultDocument } from "../hooks/use-result-document.ts";
import type { Dict } from "../i18n/index.ts";
import { locate, type Box, type Located, type PageLines, type Target } from "../lib/compare.ts";
import {
  ancestors,
  entryAt,
  firstFolded,
  itemOutline,
  markdownHeadings,
  outlineItems,
  outlineSource,
  placedHeadings,
  shownEntries,
  withChildren,
  type OutlineItem,
} from "../lib/pdf-outline.ts";
import {
  anchorOf,
  layoutPages,
  PAGE_PAD,
  PageQueue,
  PT_TO_PX,
  pageInView,
  pagesNear,
  percentOf,
  rememberedZoom,
  rememberZoom,
  renderWidth,
  scalesFor,
  scrollFor,
  typedPage,
  usableSizes,
  ZOOM_LEVELS,
  zoomKey,
  zoomStep,
  type Anchor,
  type PageLayout,
  type PageTask,
  type Size,
  type Zoom,
} from "../lib/pdf-view.ts";
import type { SessionItem } from "../lib/session.ts";
import { Icon } from "./icons.tsx";

/** The block linked across the two sides. */
export interface Linked {
  id: string;
  kind: Target["kind"];
  /** 1-based ordinal among targets of its kind, for the label ("Table 2"). */
  ordinal: number;
}

export const NO_IDS: ReadonlySet<string> = new Set();

/** Reports, while it is shown, which of the Markdown's blocks an original view can show. */
export function usePaired(onPaired: (ids: ReadonlySet<string>) => void, ids: Iterable<string>, key: unknown) {
  useEffect(() => {
    onPaired(new Set(ids));
  }, [key]);
  useEffect(() => () => onPaired(NO_IDS), []);
}

/** The located blocks of one page, as boxes in percentages of the page (so they hold at any zoom). */
export function PageMarks({
  t,
  page,
  size,
  targets,
  located,
  hover,
  onHover,
}: {
  t: Dict;
  page: number;
  size: Size | undefined;
  targets: { element: HTMLElement; target: Target }[];
  located: Map<string, Located>;
  hover: Linked | null;
  onHover: (id: string | null) => void;
}) {
  if (!size || size.width <= 0 || size.height <= 0) return null;
  const pct = (box: Box) => ({
    left: `${(box[0] / size.width) * 100}%`,
    top: `${(box[1] / size.height) * 100}%`,
    width: `${((box[2] - box[0]) / size.width) * 100}%`,
    height: `${((box[3] - box[1]) / size.height) * 100}%`,
  });
  const counters = { heading: 0, table: 0, block: 0, image: 0 };
  return (
    <span class="page-marks">
      {targets.map(({ target }) => {
        const ordinal = ++counters[target.kind];
        const spot = located.get(target.id);
        if (!spot || spot.page !== page) return null;
        const on = hover?.id === target.id;
        return (
          <span
            key={target.id}
            class={on ? `omark is-${target.kind} is-on` : `omark is-${target.kind}`}
            style={pct(spot.box)}
            onMouseEnter={() => onHover(target.id)}
            onMouseLeave={() => onHover(null)}
          >
            {on && (target.kind === "heading" || target.kind === "table") && <span class="omark-label">{target.kind === "table" ? t.tableN(ordinal) : t.headingMark}</span>}
          </span>
        );
      })}
    </span>
  );
}

/** Text layers read first (two at a time) before the first linking; the rest follow one at a time
 * behind the page images, linking again every RELINK_EVERY pages, so a long PDF links to its end. */
const TEXT_PAGES = 40;
const RELINK_EVERY = 40;
/** Times a busy (429) page is asked again before it reads as not drawn. */
const RENDER_RETRIES = 4;
/** Page images and thumbnails kept in memory at most; the farthest from the page being read go first. */
const PAGES_KEPT = 36;
const THUMBS_KEPT = 160;
/** The thumbnails' width in the panel (CSS pixels). */
const THUMB_WIDTH = 112;
/** Narrower than this, the outline panel lies over the pages instead of beside them. */
const SIDE_PANEL_MIN = 600;
/** Wheel distance (pixels) one zoom step takes with Ctrl or ⌘ held. */
const WHEEL_STEP = 60;
const PANEL_PREF = "markitai.pdfOutline";

interface Task extends PageTask {
  kind: "page" | "thumb";
  page: number;
  width: number;
}

const panelPref = (): boolean => {
  try {
    return sessionStorage.getItem(PANEL_PREF) === "1";
  } catch {
    return false;
  }
};
const savePanelPref = (open: boolean) => {
  try {
    sessionStorage.setItem(PANEL_PREF, open ? "1" : "0");
  } catch {
    // Not kept: the panel opens closed next time.
  }
};
const MAC = typeof navigator !== "undefined" && /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent);
const MOD = MAC ? "⌘" : "Ctrl";

export function PdfViewer({
  t,
  item,
  descriptor,
  doc,
  targets,
  hover,
  onHover,
  onPaired,
  header,
  phone,
}: {
  t: Dict;
  item: SessionItem;
  descriptor: OriginalDescriptor;
  doc: ResultDocument;
  /** A phone: the outline is a sheet from the bottom edge, as the document's contents are. */
  phone: boolean;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  onHover: (id: string | null) => void;
  onPaired: (ids: ReadonlySet<string>) => void;
  /** The original's header, holding the viewer's tools. */
  header: (tools: ComponentChildren) => VNode;
}) {
  const sizes = useMemo(() => usableSizes(descriptor.pages), [descriptor]);
  const count = sizes.length;
  const root = useRef<HTMLDivElement>(null);
  const body = useRef<HTMLDivElement>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<Size>({ width: 0, height: 0 });
  const [bodyWidth, setBodyWidth] = useState(0);
  const [zoom, setZoom] = useState<Zoom>(() => rememberedZoom(item.key));
  const [scroll, setScroll] = useState({ top: 0, left: 0 });
  const [announce, setAnnounce] = useState("");

  // The pane's size: the pages' view and the whole body (which decides where the panel goes).
  useEffect(() => {
    const node = scroller.current;
    const outer = body.current;
    if (!node || !outer) return;
    const measure = () => {
      setView((old) => (old.width === node.clientWidth && old.height === node.clientHeight ? old : { width: node.clientWidth, height: node.clientHeight }));
      setBodyWidth(outer.clientWidth);
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    observer.observe(outer);
    return () => observer.disconnect();
  }, []);

  const ready = view.width > 0;
  const layout = useMemo(() => layoutPages(sizes, ready ? scalesFor(zoom, view, sizes) : 1, view.width), [sizes, zoom, view.width, view.height]);
  const current = view.height > 0 ? pageInView(layout, scroll.top, view.height) : 1;
  /** The zoom on screen as a percentage: the page being read's (fit modes differ page to page). */
  const percent = percentOf(layout.scales[current - 1] ?? PT_TO_PX);

  // A zoom or a resize keeps the reading position: the same point of the same page stays where it was in the view.
  const pending = useRef<Anchor | null>(null);
  const shown = useRef<{ layout: PageLayout; view: Size } | null>(null);
  useLayoutEffect(() => {
    const node = scroller.current;
    const before = shown.current;
    shown.current = { layout, view };
    const anchor = pending.current ?? (before && before.layout.tops.length ? anchorOf(before.layout, { top: node?.scrollTop ?? 0, left: node?.scrollLeft ?? 0 }, before.view) : null);
    pending.current = null;
    if (!node || !ready || !anchor || !before || !before.view.width) return;
    // Nothing moved: the same geometry (a zoom that draws pages at the sizes they had).
    if (before.layout.height === layout.height && before.layout.width === layout.width && before.layout.widths[anchor.page] === layout.widths[anchor.page]) return;
    const next = scrollFor(layout, anchor, view);
    node.scrollTop = next.top;
    node.scrollLeft = next.left;
    setScroll(next);
  }, [layout, zoom]);
  useEffect(() => {
    setZoom(rememberedZoom(item.key));
    scroller.current?.scrollTo({ top: 0, left: 0 });
    setScroll({ top: 0, left: 0 });
  }, [item.key]);

  const zoomLabel = (value: Zoom) => (value === "width" ? t.pdfFitWidth : value === "page" ? t.pdfFitPage : t.pdfPercent(value));
  const applyZoom = (next: Zoom, point?: { vx: number; vy: number }, spoken = false) => {
    const node = scroller.current;
    const before = shown.current;
    pending.current = node && before ? anchorOf(before.layout, { top: node.scrollTop, left: node.scrollLeft }, view, point) : null;
    setZoom(next);
    rememberZoom(item.key, next);
    if (spoken) setAnnounce(t.pdfZoomShown(zoomLabel(next)));
  };
  const step = (action: "in" | "out" | "reset", point?: { vx: number; vy: number }, spoken = false) =>
    applyZoom(action === "reset" ? "width" : zoomStep(percent, action === "in" ? 1 : -1), point, spoken);
  const stepRef = useRef(step);
  stepRef.current = step;

  // Ctrl/⌘ + − 0 zoom the pages only while the pane has focus or the pointer is over it; elsewhere the browser zooms.
  const pointerIn = useRef(false);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const action = zoomKey(event);
      const node = root.current;
      if (!action || !node || event.defaultPrevented) return;
      if (!node.contains(document.activeElement) && !pointerIn.current) return;
      event.preventDefault();
      stepRef.current(action, undefined, true);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);
  // Ctrl/⌘ + wheel (and a trackpad pinch) zoom around the pointer.
  useEffect(() => {
    const node = scroller.current;
    if (!node) return;
    let travelled = 0;
    const onWheel = (event: WheelEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      event.preventDefault();
      travelled += event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? 400 : 1);
      if (Math.abs(travelled) < WHEEL_STEP) return;
      const frame = node.getBoundingClientRect();
      stepRef.current(travelled < 0 ? "in" : "out", { vx: event.clientX - frame.left, vy: event.clientY - frame.top });
      travelled = 0;
    };
    node.addEventListener("wheel", onWheel, { passive: false });
    return () => node.removeEventListener("wheel", onWheel);
  }, []);

  const frame = useRef(0);
  const onScroll = () => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      const node = scroller.current;
      if (node) setScroll((old) => (old.top === node.scrollTop && old.left === node.scrollLeft ? old : { top: node.scrollTop, left: node.scrollLeft }));
    });
  };
  useEffect(() => () => cancelAnimationFrame(frame.current), []);

  const goTo = (page: number) => {
    const node = scroller.current;
    if (!node || !count) return;
    const index = Math.min(Math.max(1, page), count) - 1;
    const top = Math.max(0, (layout.tops[index] ?? 0) - PAGE_PAD / 2);
    node.scrollTop = top;
    setScroll({ top: node.scrollTop, left: node.scrollLeft });
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if ((event.key !== "PageDown" && event.key !== "PageUp") || event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
    if ((event.target as Element | null)?.closest?.(".pdfv-panel")) return;
    event.preventDefault();
    if (event.key === "PageDown") goTo(current + 1);
    else goTo(scroll.top > (layout.tops[current - 1] ?? 0) - PAGE_PAD / 2 + 4 ? current : current - 1);
  };

  // Page images and thumbnails: a queue that asks the server two at a time, nearest first.
  const [images, setImages] = useState<Record<number, { url: string; width: number }>>({});
  const [thumbs, setThumbs] = useState<Record<number, string>>({});
  const [broken, setBroken] = useState<Record<number, true>>({});
  const [thumbsWanted, setThumbsWanted] = useState<number[]>([]);
  const queue = useRef<PageQueue<Task> | null>(null);
  const dropRef = useRef<(url: string | undefined) => void>(() => undefined);
  useEffect(() => {
    let stale = false;
    const urls = new Set<string>();
    const keep = (blob: Blob) => {
      const url = URL.createObjectURL(blob);
      urls.add(url);
      return url;
    };
    const drop = (url: string | undefined) => {
      if (url && urls.delete(url)) URL.revokeObjectURL(url);
    };
    const pages = new PageQueue<Task>(
      (task) =>
        fetchBlob(originalPagePath(item.jobId, item.itemId, task.page, task.width)).then(
          ({ blob }) => {
            if (stale) return "done";
            const url = keep(blob);
            if (task.kind === "thumb") {
              setThumbs((old) => {
                drop(old[task.page]);
                return { ...old, [task.page]: url };
              });
            } else {
              setImages((old) => {
                const had = old[task.page];
                if (had && had.width >= task.width) {
                  drop(url);
                  return old;
                }
                drop(had?.url);
                return { ...old, [task.page]: { url, width: task.width } };
              });
            }
            return "done";
          },
          (error: unknown) => (error instanceof ApiError && error.status === 429 ? "busy" : "failed"),
        ),
      (task) => {
        if (!stale && task.kind === "page") setBroken((old) => ({ ...old, [task.page]: true }));
      },
      2,
      RENDER_RETRIES,
    );
    queue.current = pages;
    dropRef.current = drop;
    return () => {
      stale = true;
      pages.close();
      queue.current = null;
      for (const url of urls) URL.revokeObjectURL(url);
      setImages({});
      setThumbs({});
      setBroken({});
    };
  }, [item.key, descriptor]);

  const [first, last] = view.height > 0 ? pagesNear(layout, scroll.top, view.height, view.height) : [0, -1];
  const [loadFirst, loadLast] = view.height > 0 ? pagesNear(layout, scroll.top, view.height, view.height / 2) : [0, -1];
  useEffect(() => {
    const pages = queue.current;
    if (!pages || !ready) return;
    const ratio = window.devicePixelRatio || 1;
    const tasks: Task[] = [];
    for (let index = loadFirst; index <= loadLast; index++) {
      const page = index + 1;
      if (broken[page]) continue;
      const width = renderWidth(layout.widths[index] ?? 0, ratio);
      if ((images[page]?.width ?? 0) >= width) continue;
      tasks.push({ key: `p${page}@${width}`, rank: Math.abs(page - current), kind: "page", page, width });
    }
    const thumbWidth = renderWidth(THUMB_WIDTH, ratio);
    for (const [position, page] of thumbsWanted.entries()) {
      if (!thumbs[page]) tasks.push({ key: `t${page}@${thumbWidth}`, rank: 10_000 + position, kind: "thumb", page, width: thumbWidth });
    }
    pages.want(tasks);
    // Memory: images far from the page being read go (and are drawn again when they come back).
    const held = Object.keys(images).map(Number);
    if (held.length > PAGES_KEPT) {
      const far = held.filter((page) => page - 1 < first || page - 1 > last).sort((a, b) => Math.abs(b - current) - Math.abs(a - current));
      const gone = new Set(far.slice(0, held.length - PAGES_KEPT));
      if (gone.size) {
        setImages((old) => {
          const kept: typeof old = {};
          for (const [page, image] of Object.entries(old)) {
            if (gone.has(Number(page))) dropRef.current(image.url);
            else kept[Number(page)] = image;
          }
          return kept;
        });
      }
    }
    const thumbsHeld = Object.keys(thumbs).map(Number);
    if (thumbsHeld.length > THUMBS_KEPT) {
      const wanted = new Set(thumbsWanted);
      const gone = new Set(thumbsHeld.filter((page) => !wanted.has(page)).slice(0, thumbsHeld.length - THUMBS_KEPT));
      if (gone.size) {
        setThumbs((old) => {
          const kept: typeof old = {};
          for (const [page, url] of Object.entries(old)) {
            if (gone.has(Number(page))) dropRef.current(url);
            else kept[Number(page)] = url;
          }
          return kept;
        });
      }
    }
  }, [loadFirst, loadLast, layout, images, thumbs, thumbsWanted, broken, current]);

  // Text layers for matching: the first pages, read once per item, two at a time.
  const [located, setLocated] = useState<Map<string, Located>>(new Map());
  useEffect(() => {
    if (!descriptor.text_layer || !targets.length) {
      setLocated(new Map());
      return;
    }
    const controller = new AbortController();
    const total = descriptor.page_count;
    const read = async (page: number): Promise<PageLines> => {
      for (let attempt = 0; ; attempt++) {
        try {
          const text = await fetchPageText(item.jobId, item.itemId, page, controller.signal);
          return { page: text.page, lines: text.lines, images: text.images?.map((image) => image.bbox) };
        } catch (error) {
          if (controller.signal.aborted || !(error instanceof ApiError && error.status === 429) || attempt >= RENDER_RETRIES) return { page, lines: [] };
          await new Promise((resolve) => setTimeout(resolve, 400 * (attempt + 1)));
        }
      }
    };
    const layers: PageLines[] = [];
    let next = 1;
    const worker = async (until: number) => {
      while (next <= until && !controller.signal.aborted) layers.push(await read(next++));
    };
    const relink = () => {
      if (!controller.signal.aborted) setLocated(locate([...layers].sort((a, b) => a.page - b.page), targets.map(({ target }) => target)));
    };
    void (async () => {
      const first = Math.min(total, TEXT_PAGES);
      await Promise.all([worker(first), worker(first)]);
      relink();
      while (next <= total && !controller.signal.aborted) {
        await worker(Math.min(total, next + RELINK_EVERY - 1));
        relink();
      }
    })();
    return () => controller.abort();
  }, [item.key, descriptor, targets]);
  usePaired(onPaired, located.keys(), located);

  // The block hovered in the Markdown comes into view on its page, at any zoom.
  useEffect(() => {
    const spot = hover ? located.get(hover.id) : undefined;
    const node = scroller.current;
    const size = spot ? sizes[spot.page - 1] : undefined;
    if (!spot || !node || !size) return;
    const index = spot.page - 1;
    const top = (layout.tops[index] ?? 0) + (spot.box[1] / size.height) * (layout.heights[index] ?? 0);
    const bottom = (layout.tops[index] ?? 0) + (spot.box[3] / size.height) * (layout.heights[index] ?? 0);
    const left = (layout.lefts[index] ?? 0) + (spot.box[0] / size.width) * (layout.widths[index] ?? 0);
    const right = (layout.lefts[index] ?? 0) + (spot.box[2] / size.width) * (layout.widths[index] ?? 0);
    const target: ScrollToOptions = { behavior: "smooth" };
    if (top < node.scrollTop + 40 || bottom > node.scrollTop + node.clientHeight - 40) target.top = Math.max(0, top - node.clientHeight / 3);
    if (left < node.scrollLeft || right > node.scrollLeft + node.clientWidth) target.left = Math.max(0, left - 24);
    if (target.top !== undefined || target.left !== undefined) node.scrollTo(target);
  }, [hover, located]);

  // The outline panel: beside the pages, over them in a narrow pane, a sheet on a phone (neither opens by itself).
  const overlay = phone || (bodyWidth > 0 && bodyWidth < SIDE_PANEL_MIN);
  const [sideOpen, setSideOpen] = useState(panelPref);
  const [overOpen, setOverOpen] = useState(false);
  const panelOpen = overlay ? overOpen : sideOpen;
  const toggleButton = useRef<HTMLButtonElement>(null);
  const setPanel = (open: boolean) => {
    if (overlay) setOverOpen(open);
    else {
      setSideOpen(open);
      savePanelPref(open);
    }
  };
  const closeOverlay = () => {
    setOverOpen(false);
    toggleButton.current?.focus();
  };
  const [bookmarks, setBookmarks] = useState<{ items: OutlineItem[]; truncated: boolean } | null>(null);
  const asked = useRef<string | null>(null);
  useEffect(() => {
    setBookmarks(null);
    asked.current = null;
  }, [item.key]);
  useEffect(() => {
    if (!panelOpen || asked.current === item.key) return;
    asked.current = item.key;
    let stale = false;
    itemOutline(item.jobId, item.itemId).then(
      (answer) => !stale && setBookmarks({ items: outlineItems(answer, descriptor.page_count), truncated: answer?.truncated === true }),
      () => !stale && setBookmarks({ items: [], truncated: false }),
    );
    return () => {
      stale = true;
      if (asked.current === item.key) asked.current = null;
    };
  }, [panelOpen, item.key]);
  const headings = useMemo(() => {
    const markdown = doc.markdown ? markdownHeadings(doc.markdown, descriptor.page_count) : { items: [], paged: false };
    if (markdown.paged) return markdown.items;
    return placedHeadings(
      targets
        .filter(({ target }) => target.kind === "heading")
        .map(({ element, target }) => ({ title: target.text.join(" "), depth: Number(element.tagName.slice(1)) || 1, page: located.get(target.id)?.page ?? null })),
    );
  }, [doc.markdown, targets, located, descriptor.page_count]);

  const pick = (page: number) => {
    goTo(page);
    if (overlay) {
      setOverOpen(false);
      scroller.current?.focus();
    }
  };

  const total = descriptor.page_count;
  const tools = (
    <PdfTools
      t={t}
      current={current}
      count={count}
      total={total}
      zoom={zoom}
      percent={percent}
      zoomLabel={zoomLabel}
      panelOpen={panelOpen}
      toggleRef={toggleButton}
      onPanel={() => setPanel(!panelOpen)}
      onPage={goTo}
      onZoom={(value) => applyZoom(value)}
      onStep={(direction) => step(direction)}
    />
  );
  return (
    <div
      ref={root}
      class="pdfv"
      onKeyDown={onKeyDown}
      onPointerEnter={() => (pointerIn.current = true)}
      onPointerLeave={() => (pointerIn.current = false)}
    >
      {header(tools)}
      <span class="sr-only" aria-live="polite">
        {announce}
      </span>
      <div ref={body} class={overlay ? "pdfv-body is-narrow" : "pdfv-body"}>
        {panelOpen && (
          <OutlinePanel
            t={t}
            overlay={overlay}
            sheet={phone}
            bookmarks={bookmarks}
            headings={headings}
            current={current}
            sizes={sizes}
            thumbs={thumbs}
            onPick={pick}
            onThumbs={setThumbsWanted}
            onClose={closeOverlay}
          />
        )}
        {panelOpen && overlay && !phone && <div class="pdfv-veil" onPointerDown={closeOverlay} />}
        <div ref={scroller} class="orig-scroll pdfv-scroll" tabIndex={0} aria-label={t.original} onScroll={onScroll}>
          <div class="pdfv-sheet" style={{ width: `${layout.width}px`, height: `${layout.height}px` }}>
            {ready &&
              Array.from({ length: Math.max(0, last - first + 1) }, (_, offset) => {
                const index = first + offset;
                const page = index + 1;
                const image = images[page];
                return (
                  <div
                    key={page}
                    class="pdfv-page"
                    data-page={page}
                    style={{ top: `${layout.tops[index]}px`, left: `${layout.lefts[index]}px`, width: `${layout.widths[index]}px`, height: `${layout.heights[index]}px` }}
                  >
                    {image ? (
                      <img src={image.url} alt={t.pageImage(page)} draggable={false} />
                    ) : (
                      <span class="page-note">{broken[page] ? t.pageUnavailable : t.loading}</span>
                    )}
                    <PageMarks t={t} page={page} size={sizes[index]} targets={targets} located={located} hover={hover} onHover={onHover} />
                  </div>
                );
              })}
          </div>
          {total > count && <p class="pane-note pdfv-more">{t.morePages(total - count)}</p>}
        </div>
      </div>
    </div>
  );
}

function PdfTools({
  t,
  current,
  count,
  total,
  zoom,
  percent,
  zoomLabel,
  panelOpen,
  toggleRef,
  onPanel,
  onPage,
  onZoom,
  onStep,
}: {
  t: Dict;
  current: number;
  count: number;
  total: number;
  zoom: Zoom;
  percent: number;
  zoomLabel: (zoom: Zoom) => string;
  panelOpen: boolean;
  toggleRef: { current: HTMLButtonElement | null };
  onPanel: () => void;
  onPage: (page: number) => void;
  onZoom: (zoom: Zoom) => void;
  onStep: (direction: "in" | "out") => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const valid = draft === null || typedPage(draft, count) !== null;
  const digits = String(Math.max(total, count)).length;
  return (
    <div class="pdfv-tools" role="group" aria-label={t.pdfTools}>
      <button
        ref={toggleRef}
        type="button"
        class={panelOpen ? "icon-btn pdfv-panel-btn is-on" : "icon-btn pdfv-panel-btn"}
        aria-pressed={panelOpen}
        aria-label={t.pdfOutline}
        title={panelOpen ? t.pdfOutlineHide : t.pdfOutlineShow}
        onClick={onPanel}
      >
        <Icon name="SidebarSimple" size={15} />
      </button>
      <span class="pdfv-sep" aria-hidden="true" />
      <span class="pdfv-nav">
        <button type="button" class="icon-btn pdfv-step" aria-label={t.pdfPrevPage} title={t.pdfPrevPage} disabled={current <= 1} onClick={() => onPage(current - 1)}>
          <Icon name="CaretUp" size={14} />
        </button>
        <span class="pdfv-page-box">
          <input
            class="pdfv-page-input"
            type="text"
            inputMode="numeric"
            enterKeyHint="go"
            autoComplete="off"
            spellcheck={false}
            aria-label={t.pdfPageInput(count)}
            aria-invalid={!valid}
            style={{ width: `${digits + 2}ch` }}
            value={draft ?? String(current)}
            onFocus={(event) => {
              setDraft(String(current));
              event.currentTarget.select();
            }}
            onInput={(event) => setDraft(event.currentTarget.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                const page = typedPage(draft ?? "", count);
                if (page !== null) {
                  onPage(page);
                  setDraft(String(page));
                }
                event.currentTarget.select();
              } else if (event.key === "Escape" && draft !== String(current)) {
                event.preventDefault();
                event.stopPropagation();
                setDraft(String(current));
                event.currentTarget.select();
              }
            }}
            onBlur={() => setDraft(null)}
          />
          <span class="pdfv-total" aria-hidden="true">
            {t.pdfPageTotal(total)}
          </span>
        </span>
        <button type="button" class="icon-btn pdfv-step" aria-label={t.pdfNextPage} title={t.pdfNextPage} disabled={current >= count} onClick={() => onPage(current + 1)}>
          <Icon name="CaretDown" size={14} />
        </button>
      </span>
      <span class="pdfv-sep pdfv-sep-zoom" aria-hidden="true" />
      <span class="pdfv-zoom">
        <button
          type="button"
          class="icon-btn pdfv-zoom-step"
          aria-label={t.pdfZoomOut}
          title={`${t.pdfZoomOut} (${MOD} −)`}
          disabled={percent <= ZOOM_LEVELS[0]}
          onClick={() => onStep("out")}
        >
          <Icon name="Minus" size={14} />
        </button>
        <ZoomMenu t={t} zoom={zoom} percent={percent} zoomLabel={zoomLabel} onZoom={onZoom} />
        <button
          type="button"
          class="icon-btn pdfv-zoom-step"
          aria-label={t.pdfZoomIn}
          title={`${t.pdfZoomIn} (${MOD} +)`}
          disabled={percent >= (ZOOM_LEVELS[ZOOM_LEVELS.length - 1] as number)}
          onClick={() => onStep("in")}
        >
          <Icon name="Plus" size={14} />
        </button>
      </span>
    </div>
  );
}

/** The zoom choices as a menu (the workbench's menu pattern): arrows walk it, Escape and a click elsewhere close it. */
function ZoomMenu({ t, zoom, percent, zoomLabel, onZoom }: { t: Dict; zoom: Zoom; percent: number; zoomLabel: (zoom: Zoom) => string; onZoom: (zoom: Zoom) => void }) {
  const [open, setOpen] = useState(false);
  const wrap = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const items = () => [...(wrap.current?.querySelectorAll<HTMLElement>('[role="menuitemradio"]') ?? [])];
  useEffect(() => {
    if (!open) return;
    requestAnimationFrame(() => (items().find((entry) => entry.getAttribute("aria-checked") === "true") ?? items()[0])?.focus());
    const away = (event: PointerEvent) => {
      if (!wrap.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [open]);
  const close = () => {
    setOpen(false);
    trigger.current?.focus();
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (!open) return;
    const list = items();
    const index = list.indexOf(document.activeElement as HTMLElement);
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      list[(index + (event.key === "ArrowDown" ? 1 : list.length - 1)) % list.length]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      list[event.key === "Home" ? 0 : list.length - 1]?.focus();
    } else if (event.key === "Tab") setOpen(false);
  };
  const choice = (value: Zoom) => (
    <button
      key={String(value)}
      type="button"
      role="menuitemradio"
      aria-checked={zoom === value}
      tabIndex={-1}
      onClick={() => {
        onZoom(value);
        close();
      }}
    >
      <span class="menu-check" aria-hidden="true">
        {zoom === value && <Icon name="Check" size={13} />}
      </span>
      {zoomLabel(value)}
    </button>
  );
  const fit = typeof zoom !== "number";
  return (
    <span ref={wrap} class="menu-wrap pdfv-zoom-menu" onKeyDown={onKeyDown}>
      <button
        ref={trigger}
        type="button"
        class="pdfv-zoom-btn"
        aria-haspopup="menu"
        aria-expanded={open}
        title={fit ? t.pdfZoomFitTitle(zoomLabel(zoom), percent) : t.pdfZoom}
        onClick={() => setOpen((value) => !value)}
      >
        <span class="pdfv-zoom-label">{zoomLabel(zoom)}</span>
        <span class="sr-only">, {t.pdfZoom}</span>
        <Icon name="CaretDown" size={12} />
      </button>
      {open && (
        <div class="menu pdfv-menu" role="menu" aria-label={t.pdfZoom}>
          <div role="group" aria-label={t.pdfZoom}>
            {choice("width")}
            {choice("page")}
          </div>
          <div role="group" aria-label={t.pdfZoom} class="menu-group">
            {ZOOM_LEVELS.map((level) => choice(level))}
          </div>
        </div>
      )}
    </span>
  );
}

function OutlinePanel({
  t,
  overlay,
  sheet,
  bookmarks,
  headings,
  current,
  sizes,
  thumbs,
  onPick,
  onThumbs,
  onClose,
}: {
  t: Dict;
  overlay: boolean;
  /** A phone's sheet over the whole window (the workbench's contents-sheet pattern, a modal layer). */
  sheet: boolean;
  bookmarks: { items: OutlineItem[]; truncated: boolean } | null;
  headings: OutlineItem[];
  current: number;
  sizes: Size[];
  thumbs: Record<number, string>;
  onPick: (page: number) => void;
  onThumbs: (pages: number[]) => void;
  onClose: () => void;
}) {
  const panel = useRef<HTMLElement | null>(null);
  const source = bookmarks ? outlineSource(bookmarks.items, headings) : null;
  const items = source === "pdf" ? (bookmarks as { items: OutlineItem[] }).items : source === "headings" ? headings : [];
  const [chosen, setChosen] = useState<"outline" | "pages" | null>(null);
  const tab = source === "pages" ? "pages" : (chosen ?? "outline");
  useEffect(() => {
    if (tab !== "pages") onThumbs([]);
  }, [tab]);
  useEffect(() => () => onThumbs([]), []);
  // Over the pages: focus moves in, and Escape closes the sheet (before the window it sits in sees the
  // key: the peek window closes on Escape from its capture listener) and gives focus back to the outline button.
  useEffect(() => {
    if (!overlay || source === null) return;
    requestAnimationFrame(() => (panel.current?.querySelector<HTMLElement>('[aria-current]') ?? panel.current?.querySelector<HTMLElement>(".pdfv-entry, .pdfv-thumb"))?.focus());
  }, [overlay, source]);
  const close = useRef(onClose);
  close.current = onClose;
  useEffect(() => {
    if (!overlay) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape" || document.querySelector(".menu, .confirm-card, .pdf-card")) return;
      event.preventDefault();
      event.stopPropagation();
      close.current();
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [overlay]);
  // Up and Down walk the entries (or thumbnails), Home and End jump: the window's own arrows (next item) stay out.
  const onKeyDown = (event: KeyboardEvent) => {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key) || event.altKey || event.ctrlKey || event.metaKey) return;
    const list = [...(panel.current?.querySelectorAll<HTMLElement>(".pdfv-entry, .pdfv-thumb") ?? [])];
    const index = list.indexOf(document.activeElement as HTMLElement);
    if (index < 0 && (event.key === "ArrowDown" || event.key === "ArrowUp")) return;
    event.preventDefault();
    const next = event.key === "Home" ? 0 : event.key === "End" ? list.length - 1 : Math.min(list.length - 1, Math.max(0, index + (event.key === "ArrowDown" ? 1 : -1)));
    list[next]?.focus();
  };
  const note =
    source === null
      ? t.loading
      : tab === "pages"
        ? source === "pages"
          ? t.pdfSourceNone
          : t.pdfSourceThumbs
        : source === "pdf"
          ? bookmarks?.truncated
            ? t.pdfSourcePdfCut(items.length)
            : t.pdfSourcePdf
          : t.pdfSourceHeadings;
  // A phone's sheet is a dialog (a div: an aside may not take that role); beside or over the pages, a complementary region.
  // Typed as the div: Preact's types allow role="dialog" on it, and the aside
  // never takes that role.
  const Panel = (sheet ? "div" : "aside") as "div";
  const content = (
    <Panel
      ref={(node: HTMLElement | null) => {
        panel.current = node;
      }}
      class={sheet ? "pdfv-panel is-sheet toc-sheet" : overlay ? "pdfv-panel is-over" : "pdfv-panel"}
      aria-label={t.pdfOutline}
      role={sheet ? "dialog" : undefined}
      aria-modal={sheet ? true : undefined}
      onKeyDown={onKeyDown}
    >
      <div class="pdfv-panel-head">
        {source !== null && source !== "pages" ? (
          <span class="seg pdfv-tabs" role="group" aria-label={t.pdfOutline}>
            <button type="button" class={tab === "outline" ? "is-on" : undefined} aria-pressed={tab === "outline"} onClick={() => setChosen("outline")}>
              {source === "pdf" ? t.pdfBookmarks : t.pdfHeadings}
            </button>
            <button type="button" class={tab === "pages" ? "is-on" : undefined} aria-pressed={tab === "pages"} onClick={() => setChosen("pages")}>
              {t.pdfThumbnails}
            </button>
          </span>
        ) : (
          <span class="pdfv-panel-title">{source === null ? t.pdfOutline : t.pdfThumbnails}</span>
        )}
        {overlay && (
          <button type="button" class="icon-btn pdfv-panel-close" aria-label={t.close} onClick={onClose}>
            <Icon name="X" size={15} />
          </button>
        )}
      </div>
      <p class="pdfv-source">{note}</p>
      {source === null ? null : tab === "pages" ? (
        <Thumbs t={t} sizes={sizes} thumbs={thumbs} current={current} onPick={onPick} onWant={onThumbs} />
      ) : (
        <OutlineList t={t} items={items} current={current} onPick={onPick} />
      )}
    </Panel>
  );
  return sheet ? (
    <div class="toc-veil" onMouseDown={(event) => event.target === event.currentTarget && onClose()}>
      {content}
    </div>
  ) : (
    content
  );
}

/** Keep the marked row in view inside the list's own scroll, never the page's. */
function follow(list: HTMLElement | null, row: HTMLElement | null | undefined) {
  if (!list || !row) return;
  const top = row.offsetTop;
  const bottom = top + row.offsetHeight;
  if (top < list.scrollTop + 8) list.scrollTop = Math.max(0, top - 24);
  else if (bottom > list.scrollTop + list.clientHeight - 8) list.scrollTop = bottom - list.clientHeight + 24;
}

function OutlineList({ t, items, current, onPick }: { t: Dict; items: OutlineItem[]; current: number; onPick: (page: number) => void }) {
  const list = useRef<HTMLUListElement>(null);
  const [folded, setFolded] = useState<Set<number>>(() => firstFolded(items));
  useEffect(() => setFolded(firstFolded(items)), [items]);
  const parents = useMemo(() => withChildren(items), [items]);
  const shownList = useMemo(() => shownEntries(items, folded), [items, folded]);
  const visible = useMemo(() => new Set(shownList), [shownList]);
  const active = entryAt(items, current);
  // The entry for the page being read; a folded one is marked on its nearest shown ancestor.
  const marked = active < 0 || visible.has(active) ? active : (ancestors(items, active).find((index) => visible.has(index)) ?? -1);
  useEffect(() => follow(list.current, list.current?.querySelector<HTMLElement>(`[data-entry="${marked}"]`)), [marked]);
  const toggle = (index: number) =>
    setFolded((old) => {
      const next = new Set(old);
      if (!next.delete(index)) next.add(index);
      return next;
    });
  return (
    <ul ref={list} class="pdfv-tree">
      {shownList.map((index) => {
        const entry = items[index] as OutlineItem;
        const title = entry.title || t.pdfUntitled;
        const holds = parents.has(index);
        const open = holds && !folded.has(index);
        return (
          <li key={index} class="pdfv-row" data-entry={index} style={{ paddingLeft: `${Math.min(entry.level - 1, 8) * 14}px` }}>
            {holds ? (
              <button type="button" class="pdfv-fold" aria-expanded={open} aria-label={open ? t.pdfCollapse(title) : t.pdfExpand(title)} onClick={() => toggle(index)}>
                <Icon name={open ? "CaretDown" : "CaretRight"} size={11} />
              </button>
            ) : (
              <span class="pdfv-fold-gap" aria-hidden="true" />
            )}
            <button type="button" class={index === marked ? "pdfv-entry is-on" : "pdfv-entry"} aria-current={index === marked ? "location" : undefined} title={title} onClick={() => onPick(entry.page)}>
              <span class="pdfv-entry-title">{title}</span>
              <span class="pdfv-entry-page">
                <span class="sr-only">, {t.pdfEntryPage(entry.page)}</span>
                <span aria-hidden="true">{entry.page}</span>
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** Small pages, drawn as they scroll into the panel's view. */
function Thumbs({
  t,
  sizes,
  thumbs,
  current,
  onPick,
  onWant,
}: {
  t: Dict;
  sizes: Size[];
  thumbs: Record<number, string>;
  current: number;
  onPick: (page: number) => void;
  onWant: (pages: number[]) => void;
}) {
  const list = useRef<HTMLOListElement>(null);
  useEffect(() => {
    const root = list.current;
    if (!root) return;
    const near = new Set<number>();
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          const page = Number((entry.target as HTMLElement).dataset.page);
          if (entry.isIntersecting) near.add(page);
          else near.delete(page);
        }
        onWant([...near].sort((a, b) => a - b));
      },
      { root, rootMargin: "240px 0px" },
    );
    for (const node of root.querySelectorAll<HTMLElement>(".pdfv-thumb")) observer.observe(node);
    return () => observer.disconnect();
  }, [sizes]);
  useEffect(() => follow(list.current, list.current?.querySelector<HTMLElement>(`[data-page="${current}"]`)?.parentElement), [current]);
  return (
    <ol ref={list} class="pdfv-thumbs">
      {sizes.map((size, index) => {
        const page = index + 1;
        const on = page === current;
        return (
          <li key={page}>
            <button type="button" class={on ? "pdfv-thumb is-on" : "pdfv-thumb"} data-page={page} aria-current={on ? "page" : undefined} aria-label={t.pdfThumb(page)} onClick={() => onPick(page)}>
              <span class="pdfv-thumb-page" style={{ aspectRatio: `${size.width} / ${size.height}` }}>
                {thumbs[page] && <img src={thumbs[page]} alt="" draggable={false} />}
              </span>
              <span class="pdfv-thumb-number" aria-hidden="true">
                {page}
              </span>
            </button>
          </li>
        );
      })}
    </ol>
  );
}
