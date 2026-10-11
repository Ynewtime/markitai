// A result's document: Rendered, Source, Diff (only for an exact base/LLM
// pair) and Files, with the Base | LLM switch. Rendered starts with the
// front-matter card and the conversion's notices; long tables show their first
// 500 rows with a sticky header and "Show all rows"; code blocks are
// highlighted by a module loaded only when there is code. Every pane has the
// same fixed height, so switching tabs never moves the frame. Rendering is
// sanitized and pinned to the result's own files.
import type { ComponentChildren } from "preact";
import { createPortal } from "preact/compat";
import type { MutableRef } from "preact/hooks";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { fetchBlob, fetchText, filePath, imagePath } from "../api/client.ts";
import { hasToken } from "../api/token.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import type { ResultDocument } from "../hooks/use-result-document.ts";
import { compareLines, type Comparison } from "../lib/diff.ts";
import { interceptDownload } from "../lib/download.ts";
import { basename, fmtBytes, localeTag } from "../lib/format.ts";
import { parseFrontMatter } from "../lib/frontmatter.ts";
import { type ImageLoads, loadArtifactImages, renderMarkdown, splitFrontmatter } from "../lib/markdown.ts";
import { groupNotices } from "../lib/notices.ts";
import { cachedExcerpt, resultKey } from "../lib/results.ts";
import type { SessionItem } from "../lib/session.ts";
import { capTables, freezeColumns, markNumericColumns, MARKER, ROW_CHUNK, ROW_LIMIT, tableSlice, type CappedTable } from "../lib/tables.ts";
import { FrontCard } from "./inspector.tsx";
import { Icon, Logo } from "./icons.tsx";
import { Skeleton } from "./skeleton.tsx";

/** Larger sides are offered as downloads instead of being compared in the page. */
const DIFF_BYTES = 8 * 1024 * 1024;
/** Tables taller than this scroll inside their own frame, under a sticky header. */
const TALL_ROWS = 24;

export type Tab = "rendered" | "source" | "diff" | "files";

export interface RenderedState {
  printable: HTMLElement | null;
  images: ImageLoads | null;
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
  rendered?: MutableRef<RenderedState>;
  /** Called with the rendered body once its content is in place (and with null when it goes). */
  onRendered?: (root: HTMLElement | null) => void;
  onError?: (message: string) => void;
}) {
  const { result, pair, shown, path, markdown } = doc;
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

  const split = useMemo(() => (markdown === null ? null : splitFrontmatter(markdown)), [markdown]);
  const front = useMemo(() => parseFrontMatter(split?.frontmatter ?? null), [split]);
  const notices = useMemo(() => (compact ? [] : groupNotices(item.warnings.map((text) => ({ itemKey: item.key, itemName: item.name, text })))), [compact, item.warnings, item.key, item.name]);

  // ---- Rendered
  useEffect(() => {
    const target = body.current;
    if (!target || split === null || !result) return;
    let stale = false;
    let loads: ImageLoads | null = null;
    let stopHighlight: (() => void) | null = null;
    setRenderError(null);
    setReady(false);
    setExpanding({});
    const placeholder = (alt: string) => t.imagePlaceholder(alt || t.imageUnavailable);
    const cut = capTables(split.body);
    capped.current = cut.tables;
    capHosts.current = [];
    renderMarkdown(cut.markdown, {
      documentPath: path,
      artifacts: result.artifacts,
      // With a token, images are fetched with the header instead of a tokened URL.
      imageURL: (relpath) => (hasToken() ? null : imagePath(item.jobId, item.itemId, relpath)),
      fileURL: (relpath) => filePath(item.jobId, relpath),
      placeholder,
      fallbackAlt: t.figureFrom(item.name),
      tableLabel: t.tableAria,
      codeLabel: t.codeAria,
    }).then(
      (fragment) => {
        if (stale) return;
        // A capped table's marker paragraph becomes its caption bar.
        for (const paragraph of [...fragment.querySelectorAll("p")]) {
          const text = paragraph.textContent ?? "";
          if (!text.startsWith(MARKER)) continue;
          const index = Number(text.slice(MARKER.length));
          const table = cut.tables[index];
          if (!table || !Number.isSafeInteger(index)) continue;
          const bar = document.createElement("div");
          bar.className = "table-cap";
          bar.setAttribute("data-cap", String(index));
          capHosts.current[index] = bar;
          paragraph.replaceWith(bar);
          bar.previousElementSibling?.classList.add("is-capped");
        }
        for (const wrap of fragment.querySelectorAll<HTMLElement>(".table-scroll")) {
          const table = wrap.querySelector("table");
          if (!table) continue;
          markNumericColumns(table);
          if (table.rows.length > TALL_ROWS) wrap.classList.add("is-tall");
        }
        target.replaceChildren(fragment);
        setReady(true);
        // Caption bars are drawn by Preact into the placeholders below.
        setExpanding(Object.fromEntries(cut.tables.map((_, index) => [index, -1])));
        loads = loadArtifactImages(
          target,
          (relpath) => fetchBlob(imagePath(item.jobId, item.itemId, relpath)).then((file) => file.blob),
          (image) => {
            const span = document.createElement("span");
            span.className = "img-blocked";
            span.textContent = placeholder(image.getAttribute("alt") ?? "");
            image.replaceWith(span);
          },
        );
        if (rendered) rendered.current.images = loads;
        if (target.querySelector("pre > code")) {
          loadHighlighter().then(
            (module) => {
              if (!stale) stopHighlight = module.highlightAll(target);
            },
            () => undefined,
          );
        }
        onRendered?.(target);
      },
      (error: unknown) => !stale && setRenderError(error),
    );
    return () => {
      stale = true;
      stopHighlight?.();
      loads?.dispose();
      if (rendered && rendered.current.images === loads) rendered.current.images = null;
      onRendered?.(null);
    };
  }, [split, path, result, item.jobId, item.name, t]);

  useEffect(() => {
    if (rendered) rendered.current.printable = printable.current;
  });

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
        fallbackAlt: t.figureFrom(item.name),
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
      ([base, llm]) => !stale && setDiff(compareLines(base, llm)),
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
  const docName = basename(path || item.name).replace(/\.llm\.md$/i, "").replace(/\.md$/i, "");
  const onDocumentLink = (event: MouseEvent) => {
    const link = (event.target as Element | null)?.closest?.("a[data-artifact]");
    const relpath = link?.getAttribute("data-artifact");
    if (!relpath) return;
    interceptDownload(event, filePath(item.jobId, relpath), basename(relpath), (error) => onError?.(describe(error).text));
  };
  const words = new Intl.NumberFormat(localeTag(locale));

  return (
    <div class={compact ? "viewer is-compact" : "viewer"}>
      <div class="viewer-bar">
        <div class="tabs" role="tablist" aria-label={t.previewAria}>
          {tabButton("rendered", t.rendered)}
          {tabButton("source", t.source)}
          {!compact && pair && tabButton("diff", t.diffTab)}
          {!compact && tabButton("files", t.filesTab, result?.artifacts.length)}
        </div>
        {pair && (active === "rendered" || active === "source") && (
          <span class="seg" role="group" aria-label={t.versionAria}>
            {(["base", "llm"] as const).map((value) => (
              <button key={value} type="button" class={shown === value ? "is-on" : undefined} aria-pressed={shown === value} onClick={() => doc.setVersion(value)}>
                {value === "base" ? t.baseTag : t.llmTag}
              </button>
            ))}
          </span>
        )}
      </div>

      <div id={`${prefix}-pane-rendered`} role="tabpanel" aria-labelledby={`${prefix}-tab-rendered`} class="pane" tabIndex={0} hidden={active !== "rendered"}>
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
                    <span class="doc-notices-first">{t.noticeTitle(notices[0]?.kind ?? "other", notices[0]?.count ?? 0, notices[0]?.models ?? [], notices[0]?.sample ?? "")}</span>
                  </summary>
                  <ul>
                    {notices.map((group) => (
                      <li key={group.key}>
                        {t.noticeTitle(group.kind, group.count, group.models, group.sample)}
                        {group.kind === "other" && group.count > 1 && ` ×${group.count}`}
                      </li>
                    ))}
                  </ul>
                </details>
              )}
              {front && (front.title || front.description || front.tags.length > 0) && <FrontCard front={front} />}
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
                {t.pdfSource}: {item.name}
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
        ) : (
          <pre class="source" aria-label={t.srcAria}>
            {split.frontmatter !== null && <span class="fm">{split.frontmatter}</span>}
            {split.body}
          </pre>
        )}
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
            <div class="diff" aria-label={t.diffAria}>
              <p class="doc-meta diff-meta">
                {basename(pair.base)} → {basename(pair.llm)} · +{diff.added} -{diff.removed}
                {diff.added === 0 && diff.removed === 0 && ` · ${t.diffSame}`}
              </p>
              {diff.rows.map((row, index) => (
                <div key={index} class={row.type === "add" ? "diff-line is-add" : row.type === "del" ? "diff-line is-del" : "diff-line"}>
                  <span class="diff-no" aria-hidden="true">
                    {row.aNo ?? ""}
                  </span>
                  <span class="diff-no" aria-hidden="true">
                    {row.bNo ?? ""}
                  </span>
                  <span class="diff-mark" aria-hidden="true">
                    {row.type === "add" ? "+" : row.type === "del" ? "-" : " "}
                  </span>
                  <span class="diff-text">{row.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {!compact && (
        <div id={`${prefix}-pane-files`} role="tabpanel" aria-labelledby={`${prefix}-tab-files`} class="pane" tabIndex={0} hidden={active !== "files"}>
          {result === null ? (
            note(doc.error, "list")
          ) : result.artifacts.length === 0 ? (
            <p class="pane-note">{t.filesNone}</p>
          ) : (
            <ul class="file-list" aria-label={t.filesAria}>
              {result.artifacts.map((artifact) => (
                <li key={artifact.relpath}>
                  <Icon name={/\.(png|jpe?g|gif|webp|avif|svg)$/i.test(artifact.relpath) ? "Image" : "FileText"} size={14} />
                  <span class="file-path" title={artifact.relpath}>
                    {artifact.relpath}
                  </span>
                  <span class="file-size">{fmtBytes(artifact.size)}</span>
                  <a
                    class="icon-btn"
                    href={filePath(item.jobId, artifact.relpath)}
                    download={basename(artifact.relpath)}
                    aria-label={t.downloadFile(artifact.relpath)}
                    title={t.downloadFile(artifact.relpath)}
                    onClick={(event) => interceptDownload(event, filePath(item.jobId, artifact.relpath), basename(artifact.relpath), (error) => onError?.(describe(error).text))}
                    onAuxClick={(event) => interceptDownload(event, filePath(item.jobId, artifact.relpath), basename(artifact.relpath), (error) => onError?.(describe(error).text))}
                  >
                    <Icon name="DownloadSimple" size={15} />
                  </a>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
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
