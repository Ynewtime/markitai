// The preview panel: Rendered, Source, Diff (when an exact base/LLM pair exists)
// and Files tabs; a Base | LLM switch for paired results; PDF export and the
// Markdown download. Rendering is sanitized and pinned to the result's files.
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { fetchBlob, fetchResult, fetchText, filePath } from "../api/client.ts";
import { hasToken } from "../api/token.ts";
import type { ItemResult } from "../api/types.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { copyText } from "../lib/clipboard.ts";
import { compareLines, type Comparison } from "../lib/diff.ts";
import { interceptDownload } from "../lib/download.ts";
import { basename, countWords, fmtBytes, fmtDate, utf8Bytes } from "../lib/format.ts";
import { type ImageLoads, loadArtifactImages, markdownPair, renderMarkdown, splitFrontmatter } from "../lib/markdown.ts";
import { printDocument, type PrintJob } from "../lib/print.ts";
import { Icon, Logo } from "./icons.tsx";
import { PdfSettings, readFurniture, storeFurniture } from "./pdf-settings.tsx";

/** Larger sides are offered as downloads instead of being compared in the page. */
const DIFF_BYTES = 8 * 1024 * 1024;

type Tab = "rendered" | "source" | "diff" | "files";

export interface PreviewTarget {
  key: string;
  jobId: string;
  itemId: string;
  name: string;
  output: string | null;
  finishedAt: string | null;
  llmEnhanced: boolean;
  operation: string;
}

const results = new Map<string, ItemResult>();

export function MarkdownView({
  t,
  locale,
  item,
  createdAt,
  announce,
  describe,
}: {
  t: Dict;
  locale: Locale;
  item: PreviewTarget;
  createdAt: string | null;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
}) {
  // A retry or enhancement keeps the item's identity but replaces its output.
  const cacheKey = `${item.key}|${item.finishedAt ?? ""}|${item.llmEnhanced}|${item.operation}`;
  const [result, setResult] = useState<ItemResult | null>(() => results.get(cacheKey) ?? null);
  const [loadError, setLoadError] = useState<unknown>(null);
  const [tab, setTab] = useState<Tab>("rendered");
  const [version, setVersion] = useState<"base" | "llm" | null>(null);
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [textError, setTextError] = useState<unknown>(null);
  const [diff, setDiff] = useState<Comparison | { kind: "error"; error: unknown } | null>(null);
  const [furniture, setFurniture] = useState(readFurniture);
  const [copied, setCopied] = useState<"idle" | "copied" | "failed">("idle");
  const [renderError, setRenderError] = useState<unknown>(null);
  const [printError, setPrintError] = useState<string | null>(null);
  const body = useRef<HTMLDivElement>(null);
  const images = useRef<ImageLoads | null>(null);
  const printable = useRef<HTMLDivElement>(null);
  const printing = useRef<PrintJob | null>(null);
  const tabs = useRef<Record<Tab, HTMLButtonElement | null>>({ rendered: null, source: null, diff: null, files: null });
  const currentPath = useRef("");

  useEffect(() => {
    const cached = results.get(cacheKey);
    if (cached) {
      setResult(cached);
      return;
    }
    let stale = false;
    setResult(null);
    setLoadError(null);
    fetchResult(item.jobId, item.itemId).then(
      (value) => {
        if (stale) return;
        results.set(cacheKey, value);
        setResult(value);
      },
      (error: unknown) => !stale && setLoadError(error),
    );
    return () => {
      stale = true;
    };
  }, [cacheKey, item.jobId, item.itemId]);

  const pair = useMemo(() => (result ? markdownPair(result.artifacts) : null), [result]);
  const shown = version ?? (result?.variant === "llm" ? "llm" : "base");
  const documentPath = useMemo(() => {
    if (!result) return item.output ?? "";
    if (pair) return pair[shown];
    const markdown = result.artifacts.find((artifact) => artifact.relpath === item.output) ?? result.artifacts.find((artifact) => artifact.relpath.endsWith(".md"));
    return markdown?.relpath ?? item.output ?? "";
  }, [result, pair, shown, item.output]);
  currentPath.current = documentPath;
  const resultPath = pair ? pair[result?.variant === "llm" ? "llm" : "base"] : documentPath;
  const markdown = result === null ? null : documentPath === resultPath ? result.markdown : (texts[documentPath] ?? null);

  // The other version of a pair is read on demand.
  useEffect(() => {
    if (!result || markdown !== null || !documentPath) return;
    let stale = false;
    setTextError(null);
    fetchText(filePath(item.jobId, documentPath), 64 * 1024 * 1024).then(
      (text) => !stale && setTexts((previous) => ({ ...previous, [documentPath]: text })),
      (error: unknown) => !stale && setTextError(error),
    );
    return () => {
      stale = true;
    };
  }, [result, markdown, documentPath, item.jobId]);

  useEffect(() => {
    if (tab === "diff" && !pair) setTab("rendered");
  }, [tab, pair]);

  const split = useMemo(() => (markdown === null ? null : splitFrontmatter(markdown)), [markdown]);
  const meta = useMemo(() => (markdown === null ? null : { words: countWords(markdown), bytes: utf8Bytes(markdown) }), [markdown]);

  useEffect(() => {
    const target = body.current;
    if (!target || split === null || !result) return;
    let stale = false;
    let loads: ImageLoads | null = null;
    setRenderError(null);
    const placeholder = (alt: string) => t.imagePlaceholder(alt || t.imageUnavailable);
    renderMarkdown(split.body, {
      documentPath,
      artifacts: result.artifacts,
      // With a token, images are fetched with the header instead of a tokened URL.
      imageURL: (path) => (hasToken() ? null : filePath(item.jobId, path)),
      fileURL: (path) => filePath(item.jobId, path),
      placeholder,
      fallbackAlt: t.figureFrom(item.name),
      tableLabel: t.tableAria,
      codeLabel: t.codeAria,
    }).then(
      (fragment) => {
        if (stale) return;
        target.replaceChildren(fragment);
        loads = loadArtifactImages(
          target,
          (path) => fetchBlob(filePath(item.jobId, path)).then((file) => file.blob),
          (image) => {
            const span = document.createElement("span");
            span.className = "img-blocked";
            span.textContent = placeholder(image.getAttribute("alt") ?? "");
            image.replaceWith(span);
          },
        );
        images.current = loads;
      },
      (error: unknown) => !stale && setRenderError(error),
    );
    return () => {
      stale = true;
      loads?.dispose();
      if (images.current === loads) images.current = null;
    };
  }, [split, documentPath, result, item.jobId, item.name, t]);

  useEffect(() => {
    if (tab !== "diff" || !pair || !result) return;
    let stale = false;
    setDiff(null);
    const sizes = new Map(result.artifacts.map((artifact) => [artifact.relpath, artifact.size]));
    if ((sizes.get(pair.base) ?? 0) > DIFF_BYTES || (sizes.get(pair.llm) ?? 0) > DIFF_BYTES) {
      setDiff({ kind: "too-large" });
      return;
    }
    const read = (path: string) =>
      path === resultPath ? Promise.resolve(result.markdown) : texts[path] !== undefined ? Promise.resolve(texts[path] as string) : fetchText(filePath(item.jobId, path), DIFF_BYTES);
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
    // A version text read meanwhile does not change the comparison, so `texts` is not a dependency.
  }, [tab, pair, result, resultPath, item.jobId]);

  useEffect(() => {
    if (copied === "idle") return;
    const timer = setTimeout(() => setCopied("idle"), 1500);
    return () => clearTimeout(timer);
  }, [copied]);

  useEffect(() => () => printing.current?.cancel(), []);

  const order: Tab[] = pair ? ["rendered", "source", "diff", "files"] : ["rendered", "source", "files"];
  const onTabKey = (event: KeyboardEvent) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    const index = order.indexOf(tab);
    const next = order[(index + (event.key === "ArrowRight" ? 1 : order.length - 1)) % order.length] ?? "rendered";
    setTab(next);
    tabs.current[next]?.focus();
  };
  const tabButton = (value: Tab, label: string) => (
    <button
      ref={(node) => {
        tabs.current[value] = node;
      }}
      type="button"
      role="tab"
      id={`tab-${value}`}
      aria-selected={tab === value}
      aria-controls={`pane-${value}`}
      tabIndex={tab === value ? 0 : -1}
      class={tab === value ? "tab is-on" : "tab"}
      onClick={() => setTab(value)}
      onKeyDown={onTabKey}
    >
      {label}
    </button>
  );

  const docName = basename(documentPath || item.name).replace(/\.llm\.md$/i, "").replace(/\.md$/i, "");
  const date = createdAt ? fmtDate(createdAt) : null;
  const versionLabel = result === null ? null : shown === "llm" && (pair !== null || result.variant === "llm") ? t.llmEnhancedLabel : t.baseTag;
  const failure = loadError ?? textError;
  const note = (error: unknown) => (
    <p class={error ? "pane-note line-error" : "pane-note"}>{error ? describe(error).text : t.loading}</p>
  );

  // A link to one of the result's files: with a token, downloaded with the header.
  const onDocumentLink = (event: MouseEvent) => {
    const link = (event.target as Element | null)?.closest?.("a[data-artifact]");
    const path = link?.getAttribute("data-artifact");
    if (!path) return;
    interceptDownload(event, filePath(item.jobId, path), basename(path), (error) => setPrintError(describe(error).text));
  };

  const exportPdf = async () => {
    if (!printable.current || markdown === null) return;
    printing.current?.cancel();
    setPrintError(null);
    const path = documentPath;
    // Images fetched with the header token must be in place before the clone.
    await images.current?.done;
    if (!printable.current || path !== currentPath.current) return;
    const job = printDocument({
      source: printable.current,
      title: docName,
      furniture,
      isCurrent: () => path === currentPath.current,
      messages: { loading: t.printImagesLoading, broken: t.printImageBroken },
    });
    printing.current = job;
    announce(t.exportPdf);
    job.done.then(
      () => undefined,
      (error: Error) => {
        if (error.name !== "AbortError") setPrintError(error.message);
      },
    );
  };

  return (
    <div class="viewer">
      <div class="viewer-bar">
        <div class="tabs" role="tablist" aria-label={t.previewAria}>
          {tabButton("rendered", t.rendered)}
          {tabButton("source", t.source)}
          {pair && tabButton("diff", t.diffTab)}
          {tabButton("files", t.filesTab)}
        </div>
        <div class="viewer-side">
          {meta && (
            <span class="viewer-meta">
              {meta.words.toLocaleString(locale === "zh" ? "zh-CN" : "en")} {t.words} · {fmtBytes(meta.bytes)}
            </span>
          )}
          <PdfSettings
            t={t}
            disabled={markdown === null}
            furniture={furniture}
            onToggle={() => {
              const next = !furniture;
              setFurniture(next);
              storeFurniture(next);
            }}
          />
          <button type="button" class="btn btn-ghost btn-sm" disabled={markdown === null} title={t.exportPdf} aria-label={t.exportPdf} onClick={() => void exportPdf()}>
            <Icon name="FilePdf" size={13} />
            <span class="btn-label">{t.exportPdf}</span>
          </button>
          {documentPath && (
            <a
              class="btn btn-ghost btn-sm"
              href={filePath(item.jobId, documentPath)}
              download={basename(documentPath)}
              aria-label={t.downloadMd}
              onClick={(event) => interceptDownload(event, filePath(item.jobId, documentPath), basename(documentPath), (error) => setPrintError(describe(error).text))}
              onAuxClick={(event) => interceptDownload(event, filePath(item.jobId, documentPath), basename(documentPath), (error) => setPrintError(describe(error).text))}
            >
              <Icon name="DownloadSimple" size={13} />
              <span class="btn-label">{t.downloadMd}</span>
            </a>
          )}
        </div>
      </div>
      {printError && (
        <p class="viewer-alert line-error" role="alert">
          {printError}
        </p>
      )}

      <div id="pane-rendered" role="tabpanel" aria-labelledby="tab-rendered" class="pane" tabIndex={0} hidden={tab !== "rendered"}>
        {markdown === null || renderError ? (
          note(failure ?? renderError)
        ) : (
          <div ref={printable} class="pdf-doc">
            <div class="print-head" aria-hidden="true">
              <span class="print-brand">
                <Logo size={20} />
                <strong>Markitai</strong>
              </span>
              <span class="print-title">{docName}</span>
            </div>
            <div class="doc">
              <p class="doc-meta">
                {item.name}
                {date !== null && ` · ${date}`}
                {versionLabel !== null && ` · ${versionLabel}`}
              </p>
              <div ref={body} class="doc-body" onClick={onDocumentLink} onAuxClick={onDocumentLink} />
            </div>
            <div class="print-foot" aria-hidden="true">
              <span>{t.pdfPreparedBy}</span>
              <span class="print-source">
                {t.pdfSource}: {item.name}
                {date !== null && ` · ${date}`}
              </span>
            </div>
          </div>
        )}
      </div>

      <div id="pane-source" role="tabpanel" aria-labelledby="tab-source" class="pane" hidden={tab !== "source"}>
        {markdown === null || split === null ? (
          note(failure)
        ) : (
          <div class="source-wrap">
            <div class="terminal">
              <div class="terminal-head">
                <span class="terminal-name">
                  {basename(documentPath || item.name)}
                  {meta && ` · ${fmtBytes(meta.bytes)} · utf-8`}
                </span>
                <button
                  type="button"
                  class="pill-btn"
                  onClick={() =>
                    void copyText(markdown).then((ok) => {
                      setCopied(ok ? "copied" : "failed");
                      announce(ok ? t.copied : t.copyFailed);
                    })
                  }
                >
                  {copied === "copied" ? t.copied : copied === "failed" ? t.copyFailed : t.copy}
                </button>
              </div>
              <pre tabIndex={0} aria-label={t.srcAria}>
                {split.frontmatter !== null && <span class="fm">{split.frontmatter}</span>}
                {split.body}
              </pre>
            </div>
          </div>
        )}
      </div>

      {pair && (
        <div id="pane-diff" role="tabpanel" aria-labelledby="tab-diff" class="pane" tabIndex={0} hidden={tab !== "diff"}>
          {diff === null ? (
            note(null)
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

      <div id="pane-files" role="tabpanel" aria-labelledby="tab-files" class="pane" tabIndex={0} hidden={tab !== "files"}>
        {result === null ? (
          note(loadError)
        ) : result.artifacts.length === 0 ? (
          <p class="pane-note">{t.filesNone}</p>
        ) : (
          <ul class="file-list" aria-label={t.filesAria}>
            {result.artifacts.map((artifact) => (
              <li key={artifact.relpath}>
                <Icon name="FileText" size={14} />
                <span class="file-path" title={artifact.relpath}>
                  {artifact.relpath}
                </span>
                <span class="file-size">{fmtBytes(artifact.size)}</span>
                <a
                  class="row-icon"
                  href={filePath(item.jobId, artifact.relpath)}
                  download={basename(artifact.relpath)}
                  aria-label={t.downloadFile(artifact.relpath)}
                  title={artifact.relpath}
                  onClick={(event) =>
                    interceptDownload(event, filePath(item.jobId, artifact.relpath), basename(artifact.relpath), (error) => setPrintError(describe(error).text))
                  }
                  onAuxClick={(event) =>
                    interceptDownload(event, filePath(item.jobId, artifact.relpath), basename(artifact.relpath), (error) => setPrintError(describe(error).text))
                  }
                >
                  <Icon name="DownloadSimple" size={15} />
                </a>
              </li>
            ))}
          </ul>
        )}
      </div>
      {pair && (
        <span class="seg viewer-version" role="group" aria-label={t.versionAria}>
          {(["base", "llm"] as const).map((value) => (
            <button key={value} type="button" class={shown === value ? "is-on" : undefined} aria-pressed={shown === value} onClick={() => setVersion(value)}>
              <Icon name={value === "base" ? "FileText" : "MagicWand"} size={12} />
              {value === "base" ? t.baseTag : t.llmTag}
            </button>
          ))}
        </span>
      )}
    </div>
  );
}
