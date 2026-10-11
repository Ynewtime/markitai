// Compare (B2): the original beside the Markdown. The original is whatever the
// backend kept (a PDF's pages, an image, an HTML snapshot, plain text, or the
// conversion's page screenshots); page images load as they scroll into view,
// fetched with the header token. Hovering a heading or table on either side
// outlines its counterpart: PDF pages are matched through their text layer,
// HTML by element. On phones the original sits above the Markdown.
import { useCallback, useEffect, useMemo, useRef, useState } from "preact/hooks";
import { fetchBlob, fetchOriginal, fetchPageText, fetchText, originalPagePath } from "../api/client.ts";
import type { OriginalDescriptor } from "../api/types.ts";
import type { ResultDocument } from "../hooks/use-result-document.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { locate, norm, pairElements, targetsOf, type Box, type Located, type PageLines, type Target } from "../lib/compare.ts";
import { download } from "../lib/download.ts";
import { fmtBytes } from "../lib/format.ts";
import { loadLibraries } from "../lib/markdown.ts";
import type { SessionItem } from "../lib/session.ts";
import { Icon } from "./icons.tsx";
import { Skeleton } from "./skeleton.tsx";
import { MarkdownView, type Tab } from "./markdown-view.tsx";

const descriptors = new Map<string, OriginalDescriptor>();
/** Text layers read for matching, at most this many pages. */
const TEXT_PAGES = 40;
const ORIGINAL_TEXT_BYTES = 5 * 1024 * 1024;

interface Linked {
  id: string;
  kind: Target["kind"];
  /** 1-based ordinal among targets of its kind, for the label ("Table 2"). */
  ordinal: number;
}

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
  const [descriptor, setDescriptor] = useState<OriginalDescriptor | null>(() => descriptors.get(item.key) ?? null);
  const [error, setError] = useState<unknown>(null);
  const [tab, setTab] = useState<Tab>("rendered");
  const [targets, setTargets] = useState<{ element: HTMLElement; target: Target }[]>([]);
  const [hover, setHover] = useState<Linked | null>(null);
  const markdownRoot = useRef<HTMLElement | null>(null);

  useEffect(() => {
    const known = descriptors.get(item.key);
    setDescriptor(known ?? null);
    setError(null);
    setHover(null);
    if (known) return;
    let stale = false;
    fetchOriginal(item.jobId, item.itemId).then(
      (value) => {
        descriptors.set(item.key, value);
        if (!stale) setDescriptor(value);
      },
      (reason: unknown) => !stale && setError(reason),
    );
    return () => {
      stale = true;
    };
  }, [item.key]);

  const onRendered = useCallback((root: HTMLElement | null) => {
    markdownRoot.current = root;
    setTargets(root ? targetsOf(root) : []);
  }, []);

  const ordinals = useMemo(() => {
    const counters = { heading: 0, table: 0 };
    return new Map(targets.map(({ target }) => [target.id, ++counters[target.kind]]));
  }, [targets]);
  const linkOf = (id: string): Linked | null => {
    const entry = targets.find(({ target }) => target.id === id);
    return entry ? { id, kind: entry.target.kind, ordinal: ordinals.get(id) ?? 1 } : null;
  };

  // Markdown side: the hovered heading or table (its scrolling frame, so the outline is not clipped).
  useEffect(() => {
    for (const { element, target } of targets) (element.closest(".table-scroll") ?? element).classList.toggle("is-linked", hover?.id === target.id);
  }, [hover, targets]);
  const onMarkdownOver = (event: MouseEvent) => {
    const element = (event.target as Element | null)?.closest?.("h1, h2, h3, h4, h5, h6, table");
    const entry = element ? targets.find((candidate) => candidate.element === element) : undefined;
    setHover(entry ? linkOf(entry.target.id) : null);
  };
  const reveal = (id: string) => {
    const entry = targets.find(({ target }) => target.id === id);
    const element = entry?.element;
    if (!element) return;
    const pane = element.closest(".pane");
    if (!pane) return;
    const box = element.getBoundingClientRect();
    const frame = pane.getBoundingClientRect();
    if (box.top < frame.top || box.bottom > frame.bottom) element.scrollIntoView({ block: "center", behavior: "smooth" });
  };

  const original = (
    <div class="orig">
      <OriginalHeader t={t} item={item} descriptor={descriptor} />
      {error ? (
        <p class="pane-note line-error">{describe(error).text}</p>
      ) : descriptor === null ? (
        <Skeleton label={t.loading} kind="page" />
      ) : (
        <OriginalBody
          t={t}
          item={item}
          descriptor={descriptor}
          targets={targets}
          hover={hover}
          describe={describe}
          onHover={(id) => {
            setHover(id ? linkOf(id) : null);
            if (id) reveal(id);
          }}
        />
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

function OriginalHeader({ t, item, descriptor }: { t: Dict; item: SessionItem; descriptor: OriginalDescriptor | null }) {
  const [failed, setFailed] = useState(false);
  const fetchable = descriptor?.download;
  return (
    <div class="orig-head">
      <span class="orig-label">{t.original}</span>
      {descriptor?.size != null && <span class="orig-size">{fmtBytes(descriptor.size)}</span>}
      {fetchable && (
        <button
          type="button"
          class="icon-btn"
          aria-label={t.downloadOriginal}
          title={failed ? t.downloadFailed : t.downloadOriginal}
          onClick={() =>
            void download(fetchable, descriptor?.name.split("/").pop() || item.name).then(
              () => setFailed(false),
              () => setFailed(true),
            )
          }
        >
          <Icon name="DownloadSimple" size={15} />
        </button>
      )}
    </div>
  );
}

function OriginalBody({
  t,
  item,
  descriptor,
  targets,
  hover,
  describe,
  onHover,
}: {
  t: Dict;
  item: SessionItem;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  describe: (error: unknown) => { text: string; detail: string };
  onHover: (id: string | null) => void;
}) {
  switch (descriptor.kind) {
    case "pdf":
    case "image":
    case "screenshots":
      return <Pages t={t} item={item} descriptor={descriptor} targets={targets} hover={hover} onHover={onHover} />;
    case "html":
      return <HtmlOriginal t={t} descriptor={descriptor} targets={targets} hover={hover} describe={describe} onHover={onHover} />;
    case "text":
      return <TextOriginal t={t} descriptor={descriptor} targets={targets} hover={hover} describe={describe} onHover={onHover} />;
    default: {
      const url = item.kind === "url" && /^https?:\/\//i.test(item.name) ? item.name : null;
      return (
        <div class="orig-none">
          <Icon name="FileText" size={28} />
          <p>{t.originalUnavailable(descriptor.reason ?? "", item.name)}</p>
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
}: {
  t: Dict;
  item: SessionItem;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  onHover: (id: string | null) => void;
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
    const load = (page: number) => {
      if (requested.has(page)) return;
      requested.add(page);
      fetchBlob(originalPagePath(item.jobId, item.itemId, page, width())).then(
        ({ blob }) => {
          if (stale) return;
          const url = URL.createObjectURL(blob);
          owned.current.push(url);
          setUrls((previous) => ({ ...previous, [page]: url }));
        },
        () => !stale && setBroken((previous) => ({ ...previous, [page]: true })),
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

  // Text layers for matching: the first pages, read once per item.
  useEffect(() => {
    if (descriptor.kind !== "pdf" || !descriptor.text_layer || !targets.length) {
      setLocated(new Map());
      return;
    }
    const controller = new AbortController();
    const count = Math.min(descriptor.page_count, TEXT_PAGES);
    Promise.all(
      Array.from({ length: count }, (_, index) =>
        fetchPageText(item.jobId, item.itemId, index + 1, controller.signal).then(
          (text): PageLines => ({ page: text.page, lines: text.lines }),
          (): PageLines => ({ page: index + 1, lines: [] }),
        ),
      ),
    ).then((layers) => {
      if (!controller.signal.aborted) setLocated(locate(layers, targets.map(({ target }) => target)));
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
    const counters = { heading: 0, table: 0 };
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
          {on && <span class="omark-label">{target.kind === "table" ? t.tableN(ordinal) : t.headingMark}</span>}
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
}: {
  t: Dict;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  describe: (error: unknown) => { text: string; detail: string };
  onHover: (id: string | null) => void;
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
}: {
  t: Dict;
  descriptor: OriginalDescriptor;
  targets: { element: HTMLElement; target: Target }[];
  hover: Linked | null;
  describe: (error: unknown) => { text: string; detail: string };
  onHover: (id: string | null) => void;
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
