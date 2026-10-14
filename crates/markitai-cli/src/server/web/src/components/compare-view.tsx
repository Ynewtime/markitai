// Compare (B2): the original beside the Markdown. The original is whatever the
// backend kept (a PDF's pages, an image, an HTML snapshot, plain text, or the
// conversion's page screenshots); page images load as they scroll into view,
// fetched with the header token. Hovering a heading or table on either side
// outlines its counterpart: PDF pages are matched through their text layer,
// HTML by element. A web page opens on its screenshot when the conversion took
// one; its other view is the kept snapshot in a sandboxed frame that runs no
// script and loads only the site's pictures, styles and fonts. The frame shares
// the workbench's origin (it still runs nothing: no `allow-scripts`, and its
// policy has no script source), so the workbench reads its blocks, pairs them
// with the Markdown's by their words and outlines them with a style of its own.
// On phones the original sits above the Markdown.
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { ApiError, downloadTicket, fetchBlob, fetchPageText, fetchShotBlocks, fetchText, originalPagePath } from "../api/client.ts";
import { hasToken } from "../api/token.ts";
import type { CaptureQuality, OriginalDescriptor } from "../api/types.ts";
import { useOriginal } from "../hooks/use-original.ts";
import type { ResultDocument } from "../hooks/use-result-document.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { locate, norm, onShots, pageBlocks, pairBlocks, pairElements, shotBlocks, TARGET_SELECTOR, targetsOf, webViews, type Box, type Located, type PageLines, type Target } from "../lib/compare.ts";
import { download } from "../lib/download.ts";
import { fmtBytes, readable } from "../lib/format.ts";
import { loadLibraries } from "../lib/markdown.ts";
import type { SessionItem } from "../lib/session.ts";
import { HelpTooltip } from "./help-tooltip.tsx";
import { Icon } from "./icons.tsx";
import { Skeleton } from "./skeleton.tsx";
import { MarkdownView, type Tab } from "./markdown-view.tsx";

/** Text layers read for matching, at most this many pages. */
const TEXT_PAGES = 40;
const ORIGINAL_TEXT_BYTES = 5 * 1024 * 1024;

interface Linked {
  id: string;
  kind: Target["kind"];
  /** 1-based ordinal among targets of its kind, for the label ("Table 2"). */
  ordinal: number;
}

/** Times a page refused as busy (429) is asked again before it reads as not drawn. */
const RENDER_RETRIES = 4;

export function CompareView({
  t,
  locale,
  item,
  doc,
  createdAt,
  describe,
  phone,
}: {
  t: Dict;
  locale: Locale;
  item: SessionItem;
  doc: ResultDocument;
  createdAt: string | null;
  describe: (error: unknown) => { text: string; detail: string };
  phone: boolean;
}) {
  const { descriptor, error } = useOriginal(item);
  const [tab, setTab] = useState<Tab>("rendered");
  const [targets, setTargets] = useState<{ element: HTMLElement; target: Target }[]>([]);
  const [hover, setHover] = useState<Linked | null>(null);
  // The blocks the original can show: only those are outlined when the Markdown is hovered.
  const [paired, setPaired] = useState<ReadonlySet<string>>(NONE);
  const markdownRoot = useRef<HTMLElement | null>(null);
  const url = item.kind === "url" && /^https?:\/\//i.test(item.name) ? item.name : null;
  // A web page: its kept page, its screenshots, or neither when what was kept is not the page
  // (a check page, a blank page built by script).
  const web = useMemo(() => (url !== null && descriptor ? webViews(descriptor) : null), [url, descriptor]);
  const webPage = web !== null && (web.page || web.shot || web.failed !== null);
  const [chosenView, setSnapView] = useState<"page" | "shot" | null>(null);
  useEffect(() => setSnapView(null), [item.key]);
  // The page itself first (it links block by block and shows what the site serves), else its screenshot.
  const snapView = web && chosenView && web[chosenView] ? chosenView : web?.page ? "page" : web?.shot ? "shot" : null;

  useEffect(() => setHover(null), [item.key, snapView]);

  const onRendered = useCallback((root: HTMLElement | null) => {
    markdownRoot.current = root;
    setTargets(root ? targetsOf(root) : []);
  }, []);

  const ordinals = useMemo(() => {
    const counters = { heading: 0, table: 0, block: 0, image: 0 };
    return new Map(targets.map(({ target }) => [target.id, ++counters[target.kind]]));
  }, [targets]);
  // A document has many blocks: look them up by id and by element.
  const byId = useMemo(() => new Map(targets.map((entry) => [entry.target.id, entry])), [targets]);
  const byElement = useMemo(() => new Map(targets.map((entry) => [entry.element as Element, entry])), [targets]);
  const linkOf = (id: string): Linked | null => {
    const entry = byId.get(id);
    return entry ? { id, kind: entry.target.kind, ordinal: ordinals.get(id) ?? 1 } : null;
  };

  // Markdown side: the hovered block (a table's scrolling frame, so the outline is not clipped).
  useEffect(() => {
    const entry = hover ? byId.get(hover.id) : undefined;
    if (!entry) return;
    const element = entry.element.closest(".table-scroll") ?? entry.element;
    element.classList.add("is-linked");
    return () => element.classList.remove("is-linked");
  }, [hover, byId]);
  const onMarkdownOver = (event: MouseEvent) => {
    // The innermost block under the pointer: a paragraph inside a list item is its own.
    const element = (event.target as Element | null)?.closest?.(TARGET_SELECTOR);
    const entry = element ? byElement.get(element) : undefined;
    // A block the original cannot show is not outlined: there is nothing to compare it with.
    setHover(entry && paired.has(entry.target.id) ? linkOf(entry.target.id) : null);
  };
  const reveal = (id: string) => {
    const entry = byId.get(id);
    const element = entry?.element;
    if (!element) return;
    const pane = element.closest(".pane");
    if (!pane) return;
    const box = element.getBoundingClientRect();
    const frame = pane.getBoundingClientRect();
    if (box.top < frame.top || box.bottom > frame.bottom) element.scrollIntoView({ block: "center", behavior: "smooth" });
  };
  const onOriginalHover = (id: string | null) => {
    setHover(id ? linkOf(id) : null);
    if (id) reveal(id);
  };

  const original = (
    <div class="orig">
      <OriginalHeader
        t={t}
        item={item}
        descriptor={descriptor}
        url={url}
        views={web?.page && web.shot && snapView ? { value: snapView, onChange: setSnapView } : null}
        framed={webPage}
        shot={snapView === "shot"}
      />
      {error ? (
        <p class="pane-note line-error">{describe(error).text}</p>
      ) : descriptor === null ? (
        <Skeleton label={t.loading} kind="page" />
      ) : webPage && url ? (
        snapView === "shot" ? (
          <Pages t={t} item={item} descriptor={descriptor} targets={targets} hover={hover} onHover={onOriginalHover} onPaired={setPaired} />
        ) : snapView === "page" ? (
          <PageFrame
            t={t}
            item={item}
            view={descriptor.view as string}
            describe={describe}
            onFailed={web?.shot ? () => setSnapView("shot") : null}
            targets={targets}
            hover={hover}
            onHover={onOriginalHover}
            onPaired={setPaired}
          />
        ) : (
          <PageNotKept t={t} url={url} failed={web?.failed ?? null} />
        )
      ) : (
        <OriginalBody t={t} item={item} descriptor={descriptor} targets={targets} hover={hover} describe={describe} onHover={onOriginalHover} onPaired={setPaired} />
      )}
    </div>
  );
  return (
    <div class={phone ? "compare is-stacked" : "compare"} onMouseLeave={() => setHover(null)}>
      {original}
      <div class="compare-doc" onMouseOver={onMarkdownOver}>
        <MarkdownView t={t} locale={locale} item={item} doc={doc} createdAt={createdAt} describe={describe} tab={tab} onTab={setTab} compact onRendered={onRendered} />
      </div>
    </div>
  );
}

const NONE: ReadonlySet<string> = new Set();

/** Reports, while it is shown, which of the Markdown's blocks an original view can show. */
function usePaired(onPaired: (ids: ReadonlySet<string>) => void, ids: Iterable<string>, key: unknown) {
  useEffect(() => {
    onPaired(new Set(ids));
  }, [key]);
  useEffect(() => () => onPaired(NONE), []);
}

/** A web page whose kept copy is not the page (a check page, a blank page built by script) and
 * that has no usable screenshot: say so, and open the page itself. */
function PageNotKept({ t, url, failed }: { t: Dict; url: string; failed: CaptureQuality | null }) {
  return (
    <div class="orig-none">
      <Icon name="Globe" size={28} />
      <p>{t.pageNotKept(failed?.state ?? null)}</p>
      <a class="btn btn-sm" href={url} target="_blank" rel="noopener noreferrer">
        <Icon name="ArrowSquareOut" size={14} />
        {t.openPage}
        <span class="sr-only"> {t.opensNewTab}</span>
      </a>
    </div>
  );
}

function OriginalHeader({
  t,
  item,
  descriptor,
  url,
  views,
  framed,
  shot,
}: {
  t: Dict;
  item: SessionItem;
  descriptor: OriginalDescriptor | null;
  url: string | null;
  views: { value: "page" | "shot"; onChange: (value: "page" | "shot") => void } | null;
  /** A web page: the page itself is the original, so no download here (Files keeps it). */
  framed: boolean;
  /** Its screenshot is shown. */
  shot: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const fetchable = framed ? null : descriptor?.download;
  return (
    <div class="orig-head">
      <span class="orig-label">{t.original}</span>
      {framed && <InfoTip label={t.originalPageInfo} text={shot ? t.originalShotNote : t.originalPageNote} />}
      {views && (
        <span class="seg orig-views" role="group" aria-label={t.originalViews}>
          {(["shot", "page"] as const).map((value) => (
            <button key={value} type="button" class={views.value === value ? "is-on" : undefined} aria-pressed={views.value === value} onClick={() => views.onChange(value)}>
              {value === "page" ? t.originalPage : t.originalScreenshot}
            </button>
          ))}
        </span>
      )}
      {descriptor?.size != null && <span class="orig-size">{fmtBytes(descriptor.size)}</span>}
      {fetchable && (
        <button
          type="button"
          class="icon-btn"
          aria-label={t.downloadOriginal}
          title={failed ? t.downloadFailed : t.downloadOriginal}
          onClick={() =>
            void download(fetchable, readable(descriptor?.name.split("/").pop() || item.name)).then(
              () => setFailed(false),
              () => setFailed(true),
            )
          }
        >
          <Icon name="DownloadSimple" size={15} />
        </button>
      )}
      {/* The one "Open the page" while Compare is open; a page not kept offers it in the body. */}
      {url && descriptor?.kind !== "none" && (
        <a class="icon-btn" href={url} target="_blank" rel="noopener noreferrer" aria-label={`${t.openPage} ${t.opensNewTab}`} title={t.openPage}>
          <Icon name="ArrowSquareOut" size={15} />
        </a>
      )}
    </div>
  );
}

/** A web page as the server fetched it, in a frame that runs no script and sends
 * no form; it loads only the site's pictures, style sheets and fonts (the
 * server's own CSP for this route). The frame is same-origin with the workbench
 * so the workbench can read it: hovering a block on either side outlines its
 * counterpart, drawn by a style the workbench puts in the frame. A frame cannot
 * send the token, so it opens with a single-use ticket. */
function PageFrame({
  t,
  item,
  view,
  describe,
  onFailed,
  targets,
  hover,
  onHover,
  onPaired,
}: {
  t: Dict;
  item: SessionItem;
  view: string;
  describe: (error: unknown) => { text: string; detail: string };
  onFailed: (() => void) | null;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  onHover: (id: string | null) => void;
  onPaired: (ids: ReadonlySet<string>) => void;
}) {
  const [src, setSrc] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  const frame = useRef<HTMLIFrameElement>(null);
  const [page, setPage] = useState<Document | null>(null);
  useEffect(() => {
    let stale = false;
    setSrc(null);
    setError(null);
    setPage(null);
    (hasToken() ? downloadTicket(view) : Promise.resolve(view)).then(
      (value) => !stale && setSrc(value),
      (reason: unknown) => {
        if (stale) return;
        if (onFailed) onFailed();
        else setError(reason);
      },
    );
    return () => {
      stale = true;
    };
  }, [item.key, view]);
  // The page's blocks paired with the Markdown's; an empty map when the frame cannot be read.
  const pairs = useMemo(() => {
    if (!page?.body) return new Map<string, Element>();
    const blocks = pageBlocks(page.body);
    const found = pairBlocks(
      targets.map((entry) => entry.target),
      blocks.map((entry) => entry.block),
    );
    return new Map([...found].map(([id, index]) => [id, blocks[index]?.element as Element]));
  }, [page, targets]);
  usePaired(onPaired, pairs.keys(), pairs);
  const reverse = useMemo(() => new Map([...pairs].map(([id, element]) => [element, id])), [pairs]);
  // Pointer over the page: the innermost paired block under it. Read from the page's own
  // :hover while the pointer is over the frame: WebKit (Safari) runs no listener in a page
  // that may run no script, not even one the workbench adds, so events cannot carry it.
  const reverseRef = useRef(reverse);
  reverseRef.current = reverse;
  const onHoverRef = useRef(onHover);
  onHoverRef.current = onHover;
  const watch = useRef<{ timer: number; id: string | null } | null>(null);
  const stopWatching = useCallback(() => {
    if (!watch.current) return;
    clearInterval(watch.current.timer);
    const had = watch.current.id;
    watch.current = null;
    if (had) onHoverRef.current(null);
  }, []);
  const startWatching = useCallback(() => {
    if (!page || watch.current) return;
    const state = { timer: 0, id: null as string | null };
    const read = () => {
      const hovered = page.querySelectorAll(":hover");
      let id: string | null = null;
      for (let index = hovered.length - 1; index >= 0 && id === null; index--) id = reverseRef.current.get(hovered[index] as Element) ?? null;
      if (id === state.id) return;
      state.id = id;
      onHoverRef.current(id);
    };
    state.timer = window.setInterval(read, HOVER_POLL_MS);
    watch.current = state;
    read();
  }, [page]);
  useEffect(() => {
    // The page arrived under a pointer that is already over the frame.
    if (page && frame.current?.matches(":hover")) startWatching();
    return stopWatching;
  }, [page, startWatching, stopWatching]);
  // The outline: an attribute on the paired block and the workbench's style for it, scrolled into the frame's view.
  useEffect(() => {
    const element = hover ? pairs.get(hover.id) : undefined;
    if (!page || !element) return;
    element.setAttribute(LINKED, "");
    const view = page.defaultView;
    const box = element.getBoundingClientRect();
    if (view && (box.top < 0 || box.bottom > view.innerHeight)) {
      view.scrollTo({ top: view.scrollY + box.top - Math.max(16, (view.innerHeight - box.height) / 2), behavior: "smooth" });
    }
    return () => element.removeAttribute(LINKED);
  }, [hover, pairs, page]);
  // The page is read as soon as it is parsed, not at its load event: a page that runs no script
  // loads every picture at once (lazy loading needs script), and on a slow or filtered network the
  // load event comes late or never, which left the page unlinked.
  const attach = useCallback((): boolean => {
    let doc: Document | null = null;
    try {
      doc = frame.current?.contentDocument ?? null;
    } catch {
      // A backend still on the opaque frame (v1.12) gives no document: the page shows, unlinked.
      return true;
    }
    if (!doc || doc.URL === "about:blank" || doc.readyState === "loading" || !doc.head || !doc.body) return false;
    if (!doc.getElementById(STYLE_ID)) {
      const style = doc.createElement("style");
      style.id = STYLE_ID;
      style.textContent = FRAME_STYLE;
      doc.head.append(style);
    }
    setPage((current) => (current === doc ? current : doc));
    return true;
  }, []);
  useEffect(() => {
    if (src === null) return;
    let timer = 0;
    const started = Date.now();
    const poll = () => {
      if (attach() || Date.now() - started > ATTACH_MS) return;
      timer = window.setTimeout(poll, 50);
    };
    poll();
    return () => clearTimeout(timer);
  }, [src, attach]);
  const onLoad = () => {
    if (!attach()) setPage(null);
  };
  if (error) return <p class="pane-note line-error">{describe(error).text}</p>;
  return (
    <div class="orig-frame-wrap">
      {src === null ? (
        <p class="pane-note">{t.loading}</p>
      ) : (
        <iframe
          ref={frame}
          class="orig-frame"
          src={src}
          title={t.originalPageTitle}
          sandbox="allow-same-origin allow-popups allow-popups-to-escape-sandbox"
          referrerpolicy="no-referrer"
          onLoad={onLoad}
          onMouseEnter={startWatching}
          onMouseLeave={stopWatching}
        />
      )}
    </div>
  );
}

/** A small (i) in a header whose bubble says what the view shows. */
function InfoTip({ label, text }: { label: string; text: string }) {
  return (
    <HelpTooltip text={text}>
      {(describedBy) => (
        <button type="button" class="icon-btn info-tip" aria-label={label} aria-describedby={describedBy}>
          <Icon name="Info" size={15} />
        </button>
      )}
    </HelpTooltip>
  );
}

/** How often the hovered block of a framed page is read while the pointer is over it. */
const HOVER_POLL_MS = 60;
/** How long a framed page is waited for before it is left unlinked until its load event. */
const ATTACH_MS = 60_000;

/** The outline the workbench draws in a framed page (the Markdown side's mark colour). */
const LINKED = "data-markitai-linked";
const STYLE_ID = "markitai-compare-style";
const FRAME_STYLE = `[${LINKED}]{outline:2px solid #7c3aed!important;outline-offset:3px!important;border-radius:3px;background-color:rgb(124 58 237 / 0.08)!important;transition:outline-color 120ms ease}@media (prefers-reduced-motion:reduce){[${LINKED}]{transition:none}}`;

function OriginalBody({
  t,
  item,
  descriptor,
  targets,
  hover,
  describe,
  onHover,
  onPaired,
}: {
  t: Dict;
  item: SessionItem;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  describe: (error: unknown) => { text: string; detail: string };
  onHover: (id: string | null) => void;
  onPaired: (ids: ReadonlySet<string>) => void;
}) {
  switch (descriptor.kind) {
    case "pdf":
    case "image":
    case "screenshots":
      return <Pages t={t} item={item} descriptor={descriptor} targets={targets} hover={hover} onHover={onHover} onPaired={onPaired} />;
    case "html":
      return <HtmlOriginal t={t} descriptor={descriptor} targets={targets} hover={hover} describe={describe} onHover={onHover} onPaired={onPaired} />;
    case "text":
      return <TextOriginal t={t} descriptor={descriptor} targets={targets} hover={hover} describe={describe} onHover={onHover} onPaired={onPaired} />;
    default: {
      const url = item.kind === "url" && /^https?:\/\//i.test(item.name) ? item.name : null;
      return (
        <div class="orig-none">
          <Icon name="FileText" size={28} />
          <p>{t.originalUnavailable(descriptor.reason ?? "", readable(item.name))}</p>
          {url && (
            <a class="btn btn-sm" href={url} target="_blank" rel="noopener noreferrer">
              <Icon name="ArrowSquareOut" size={14} />
              {t.openPage}
              <span class="sr-only"> {t.opensNewTab}</span>
            </a>
          )}
        </div>
      );
    }
  }
}

/** Page images, loaded as they come into view, with the located outlines on top. */
function Pages({
  t,
  item,
  descriptor,
  targets,
  hover,
  onHover,
  onPaired,
}: {
  t: Dict;
  item: SessionItem;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  onHover: (id: string | null) => void;
  onPaired: (ids: ReadonlySet<string>) => void;
}) {
  const scroller = useRef<HTMLDivElement>(null);
  const [urls, setUrls] = useState<Record<number, string>>({});
  const [broken, setBroken] = useState<Record<number, boolean>>({});
  const [current, setCurrent] = useState(1);
  const [located, setLocated] = useState<Map<string, Located>>(new Map());
  const pages = descriptor.pages;
  const owned = useRef<string[]>([]);

  // Page images: requested when a page nears the viewport, at the pane's pixel width.
  useEffect(() => {
    const root = scroller.current;
    if (!root) return;
    let stale = false;
    const requested = new Set<number>();
    const width = () => {
      const css = root.clientWidth - 24;
      return Math.min(2400, Math.max(400, Math.ceil((css * (window.devicePixelRatio || 1)) / 200) * 200));
    };
    // The server draws a few pages at a time and answers the rest 429 render_busy:
    // those are asked again, a little later each time, four times at most.
    const load = (page: number, attempt = 0) => {
      if (attempt === 0 && requested.has(page)) return;
      requested.add(page);
      fetchBlob(originalPagePath(item.jobId, item.itemId, page, width())).then(
        ({ blob }) => {
          if (stale) return;
          const url = URL.createObjectURL(blob);
          owned.current.push(url);
          setUrls((previous) => ({ ...previous, [page]: url }));
        },
        (error: unknown) => {
          if (stale) return;
          if (error instanceof ApiError && error.status === 429 && attempt < RENDER_RETRIES) {
            setTimeout(() => !stale && load(page, attempt + 1), 400 * (attempt + 1));
            return;
          }
          setBroken((previous) => ({ ...previous, [page]: true }));
        },
      );
    };
    const observer = new IntersectionObserver(
      (entries) => {
        let best: { page: number; ratio: number } | null = null;
        for (const entry of entries) {
          const page = Number((entry.target as HTMLElement).dataset.page);
          if (entry.isIntersecting) load(page);
          if (entry.intersectionRatio > (best?.ratio ?? 0)) best = { page, ratio: entry.intersectionRatio };
        }
        if (best) setCurrent(best.page);
      },
      { root, rootMargin: "600px 0px", threshold: [0, 0.25, 0.5, 0.75] },
    );
    for (const node of root.querySelectorAll<HTMLElement>(".opage")) observer.observe(node);
    return () => {
      stale = true;
      observer.disconnect();
      for (const url of owned.current.splice(0)) URL.revokeObjectURL(url);
      setUrls({});
    };
  }, [item.key, descriptor]);

  usePaired(onPaired, located.keys(), located);

  // A web page's screenshots: the boxes its blocks had when they were taken (v1.14), paired like the page's own.
  useEffect(() => {
    const path = descriptor.kind === "pdf" ? null : descriptor.blocks;
    if (!path || !targets.length) {
      if (descriptor.kind !== "pdf") setLocated(new Map());
      return;
    }
    let stale = false;
    fetchShotBlocks(path).then(
      (shot) => {
        if (stale || !Array.isArray(shot?.blocks)) return;
        const found = pairBlocks(
          targets.map((entry) => entry.target),
          shotBlocks(shot.blocks),
        );
        const spots = new Map<string, Located>();
        for (const [id, index] of found) {
          const box = shot.blocks[index]?.bbox;
          const spot = box ? onShots(box, shot.width, pages) : null;
          if (spot && spot.box[2] > spot.box[0] && spot.box[3] > spot.box[1]) spots.set(id, spot);
        }
        setLocated(spots);
      },
      // Without the boxes the screenshots show unlinked.
      () => !stale && setLocated(new Map()),
    );
    return () => {
      stale = true;
    };
  }, [item.key, descriptor, targets]);

  // Text layers for matching: the first pages, read once per item.
  useEffect(() => {
    if (descriptor.kind !== "pdf" || !descriptor.text_layer || !targets.length) {
      if (descriptor.kind === "pdf") setLocated(new Map());
      return;
    }
    const controller = new AbortController();
    const count = Math.min(descriptor.page_count, TEXT_PAGES);
    // Two at a time, as the server reads pages (it answers more 429 render_busy);
    // a busy answer is asked again a little later, a few times at most.
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
    const worker = async () => {
      while (next <= count && !controller.signal.aborted) layers.push(await read(next++));
    };
    void Promise.all([worker(), worker()]).then(() => {
      if (!controller.signal.aborted) setLocated(locate(layers.sort((a, b) => a.page - b.page), targets.map(({ target }) => target)));
    });
    return () => controller.abort();
  }, [item.key, descriptor, targets]);

  // The hovered element's page comes into view.
  useEffect(() => {
    if (!hover) return;
    const spot = located.get(hover.id);
    const root = scroller.current;
    if (!spot || !root) return;
    const node = root.querySelector<HTMLElement>(`.opage[data-page="${spot.page}"]`);
    if (!node) return;
    const size = pages[spot.page - 1];
    const top = node.offsetTop + (size ? (spot.box[1] / size.height) * node.clientHeight : 0);
    if (top < root.scrollTop + 40 || top > root.scrollTop + root.clientHeight - 80) root.scrollTo({ top: Math.max(0, top - root.clientHeight / 3), behavior: "smooth" });
  }, [hover, located]);

  const marks = (page: number) => {
    const size = pages[page - 1];
    if (!size) return null;
    const pct = (box: Box) => ({
      left: `${(box[0] / size.width) * 100}%`,
      top: `${(box[1] / size.height) * 100}%`,
      width: `${((box[2] - box[0]) / size.width) * 100}%`,
      height: `${((box[3] - box[1]) / size.height) * 100}%`,
    });
    const counters = { heading: 0, table: 0, block: 0, image: 0 };
    return targets.map(({ target }) => {
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
    });
  };

  return (
    <div class="orig-pages-wrap">
      {descriptor.page_count > 1 && (
        <p class="orig-position" aria-live="polite">
          {t.pageOf(current, descriptor.page_count)}
        </p>
      )}
      <div ref={scroller} class="orig-scroll" tabIndex={0} aria-label={t.original}>
        {pages.map((size, index) => {
          const page = index + 1;
          return (
            <div key={page} class="opage" data-page={page} style={{ aspectRatio: `${size.width} / ${size.height}` }}>
              {urls[page] ? (
                <img src={urls[page]} alt={t.pageImage(page)} draggable={false} />
              ) : broken[page] ? (
                <span class="page-note">{t.pageUnavailable}</span>
              ) : (
                <span class="page-note">{t.loading}</span>
              )}
              <span class="page-marks">{marks(page)}</span>
            </div>
          );
        })}
        {descriptor.page_count > pages.length && <p class="pane-note">{t.morePages(descriptor.page_count - pages.length)}</p>}
      </div>
    </div>
  );
}

const HTML_FORBID = ["style", "script", "iframe", "frame", "frameset", "object", "embed", "form", "input", "button", "textarea", "select", "video", "audio", "source", "picture", "link", "meta", "base", "svg", "math", "noscript", "template", "canvas", "dialog"];
const HTML_FORBID_ATTR = ["style", "class", "id", "name", "srcset", "poster", "ping", "target", "action", "formaction", "background", "width", "height"];

/** The kept page, sanitized: no scripts, styles, frames, forms or remote requests. */
function HtmlOriginal({
  t,
  descriptor,
  targets,
  hover,
  describe,
  onHover,
  onPaired,
}: {
  t: Dict;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  describe: (error: unknown) => { text: string; detail: string };
  onHover: (id: string | null) => void;
  onPaired: (ids: ReadonlySet<string>) => void;
}) {
  const root = useRef<HTMLDivElement>(null);
  const [state, setState] = useState<"loading" | "ready" | { error: unknown }>("loading");
  const [pairs, setPairs] = useState<Map<string, HTMLElement>>(new Map());
  useEffect(() => {
    const node = root.current;
    if (!node || !descriptor.download) return;
    let stale = false;
    setState("loading");
    Promise.all([fetchText(descriptor.download, ORIGINAL_TEXT_BYTES).catch((error: unknown) => (error instanceof RangeError ? null : Promise.reject(error))), loadLibraries()]).then(
      ([html, { purify }]) => {
        if (stale) return;
        if (html === null) {
          setState({ error: new RangeError("too-large") });
          return;
        }
        const fragment = purify.sanitize(html, { RETURN_DOM_FRAGMENT: true, WHOLE_DOCUMENT: false, FORBID_TAGS: HTML_FORBID, FORBID_ATTR: HTML_FORBID_ATTR, ALLOW_DATA_ATTR: false, ALLOW_ARIA_ATTR: false });
        // Nothing in the copy may load or navigate: images become their alt text, links plain text.
        for (const image of [...fragment.querySelectorAll("img")]) {
          const span = document.createElement("span");
          span.className = "img-blocked";
          span.textContent = t.imagePlaceholder(image.getAttribute("alt") || t.imageUnavailable);
          image.replaceWith(span);
        }
        for (const link of [...fragment.querySelectorAll("a")]) {
          const span = document.createElement("span");
          span.className = "orig-link";
          span.append(...link.childNodes);
          link.replaceWith(span);
        }
        for (const element of fragment.querySelectorAll("[src], [href]")) {
          element.removeAttribute("src");
          element.removeAttribute("href");
        }
        node.replaceChildren(fragment);
        setState("ready");
      },
      (error: unknown) => !stale && setState({ error }),
    );
    return () => {
      stale = true;
    };
  }, [descriptor]);
  useEffect(() => {
    if (state !== "ready" || !root.current) return;
    setPairs(pairElements(targets, root.current));
  }, [state, targets]);
  useEffect(() => {
    for (const [id, element] of pairs) element.classList.toggle("is-linked", hover?.id === id);
    const element = hover ? pairs.get(hover.id) : undefined;
    const pane = root.current?.parentElement;
    if (element && pane) {
      const box = element.getBoundingClientRect();
      const frame = pane.getBoundingClientRect();
      if (box.top < frame.top || box.bottom > frame.bottom) element.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }, [hover, pairs]);
  usePaired(onPaired, pairs.keys(), pairs);
  const reverse = useMemo(() => new Map([...pairs].map(([id, element]) => [element, id])), [pairs]);
  return (
    <div
      class="orig-scroll is-html"
      tabIndex={0}
      aria-label={t.original}
      onMouseOver={(event) => {
        const element = (event.target as Element | null)?.closest?.("h1, h2, h3, h4, h5, h6, table");
        onHover(element ? (reverse.get(element as HTMLElement) ?? null) : null);
      }}
    >
      {state === "loading" && <p class="pane-note">{t.loading}</p>}
      {typeof state === "object" && <p class="pane-note line-error">{state.error instanceof RangeError ? t.originalTooLarge : describe(state.error).text}</p>}
      <div ref={root} class="doc-body orig-html" />
    </div>
  );
}

/** A plain-text original, line by line; Markdown headings are matched by their text. */
function TextOriginal({
  t,
  descriptor,
  targets,
  hover,
  describe,
  onHover,
  onPaired,
}: {
  t: Dict;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  describe: (error: unknown) => { text: string; detail: string };
  onHover: (id: string | null) => void;
  onPaired: (ids: ReadonlySet<string>) => void;
}) {
  const [text, setText] = useState<string | null>(null);
  const [error, setError] = useState<unknown>(null);
  useEffect(() => {
    if (!descriptor.download) return;
    let stale = false;
    setText(null);
    fetchText(descriptor.download, ORIGINAL_TEXT_BYTES).then(
      (value) => !stale && setText(value),
      (reason: unknown) => !stale && setError(reason),
    );
    return () => {
      stale = true;
    };
  }, [descriptor]);
  const lines = useMemo(() => (text === null ? [] : text.split("\n").slice(0, 20_000)), [text]);
  const lineOf = useMemo(() => {
    const map = new Map<string, number>();
    let from = 0;
    for (const { target } of targets) {
      if (target.kind !== "heading") continue;
      const wanted = norm(target.text.join(" "));
      for (let index = from; index < lines.length; index++) {
        if (norm((lines[index] ?? "").replace(/^\s{0,3}#{1,6}\s+/, "")) === wanted && wanted) {
          map.set(target.id, index);
          from = index + 1;
          break;
        }
      }
    }
    return map;
  }, [lines, targets]);
  usePaired(onPaired, lineOf.keys(), lineOf);
  const byLine = useMemo(() => new Map([...lineOf].map(([id, line]) => [line, id])), [lineOf]);
  const hot = hover ? lineOf.get(hover.id) : undefined;
  useEffect(() => {
    if (hot === undefined) return;
    document.querySelector(`.orig-text [data-line="${hot}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [hot]);
  if (error) return <p class="pane-note line-error">{error instanceof RangeError ? t.originalTooLarge : describe(error).text}</p>;
  if (text === null) return <p class="pane-note">{t.loading}</p>;
  return (
    <div class="orig-scroll" tabIndex={0} aria-label={t.original}>
      <pre class="orig-text">
        {lines.map((line, index) => {
          const id = byLine.get(index);
          return (
            <span
              key={index}
              data-line={index}
              class={hot === index ? "is-linked" : undefined}
              onMouseEnter={id ? () => onHover(id) : undefined}
              onMouseLeave={id ? () => onHover(null) : undefined}
            >
              {line}
              {"\n"}
            </span>
          );
        })}
      </pre>
    </div>
  );
}
