import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  encodeRelPath,
  fetchItemResult,
  fetchJobFileText,
  jobFileUrl,
} from "../api/client";
import type { ItemResult } from "../api/types";
import type { SessionItem } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { diffLines, MAX_DIFF_LINES, type DiffLine } from "../lib/diff";
import { countWords, fmtBytes, fmtDate, utf8Bytes } from "../lib/format";
import { DownloadIcon, LogoMark, PdfIcon } from "./icons";

type Tab = "rendered" | "source" | "diff";
type RehypeHighlight = typeof import("rehype-highlight").default;

/** True for URLs with a scheme, absolute paths, fragments and queries. */
const ABSOLUTE_RE = /^([a-z][a-z0-9+.-]*:|\/|#|\?)/i;
const BADGE_IMAGE_RE =
  /(?:img\.shields\.io|shields\.io|badgen\.net|badge\.fury\.io|\/badge\.svg(?:[?#]|$)|\/badge\/|codecov\.io\/.+\/badge|coveralls\.io\/.+\/badge|readthedocs\.org\/.+\/badge)/i;
const BADGE_ALT_RE =
  /^(?:build|ci|tests?|coverage|codecov|version|release|license|stars?|downloads?|npm|pypi|twitter|follow|docs?|status)(?:\b|\s*:)/i;
const PDF_HEADER_KEY = "markitai.pdf.custom-header-footer";

function isBadgeImage(src: string | undefined, alt: string | undefined): boolean {
  return (
    (src !== undefined && BADGE_IMAGE_RE.test(src)) ||
    (alt !== undefined && BADGE_ALT_RE.test(alt))
  );
}

function isCurrentThemeDark(): boolean {
  const scheme = getComputedStyle(document.documentElement).colorScheme;
  if (scheme.split(/\s+/).includes("dark")) return true;
  if (document.documentElement.dataset.theme === "dark") return true;
  return (
    document.documentElement.dataset.theme !== "light" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
  );
}

function readCustomPdfHeader(): boolean {
  try {
    return localStorage.getItem(PDF_HEADER_KEY) === "true";
  } catch {
    return false;
  }
}

function splitFrontmatter(md: string): { fm: string | null; body: string } {
  const m = /^---\n[\s\S]*?\n---(?=\n|$)/.exec(md);
  if (m === null) return { fm: null, body: md };
  return { fm: m[0], body: md.slice(m[0].length) };
}

function basename(relpath: string): string {
  return relpath.split("/").pop() ?? relpath;
}

/** Repair historical generated image destinations that contain raw spaces.
 * CommonMark treats those as plain text before urlTransform gets a chance. */
function portableLocalImageDestinations(markdown: string): string {
  return markdown.replace(
    /(!\[[^\n]*?\]\()((?:\.\/)?\.markitai\/(?:assets|screenshots)\/[^)\n]+)(\))/g,
    (_match, open: string, path: string, close: string) => {
      const decoded = path
        .trim()
        .split("/")
        .map((segment) => {
          try {
            return decodeURIComponent(segment);
          } catch {
            return segment;
          }
        })
        .join("/");
      return `${open}${encodeRelPath(decoded)}${close}`;
    },
  );
}

/** Diff tab precondition: the item's artifacts hold BOTH the base `.md` and
 * the `.llm.md` (the server keeps the base via keep_base in web jobs). */
function findDiffPair(result: ItemResult | null): { base: string; llm: string } | null {
  if (result === null) return null;
  const rels = result.artifacts.map((a) => a.relpath);
  const llm = rels.find((r) => r.endsWith(".llm.md"));
  if (llm === undefined) return null;
  const base = `${llm.slice(0, -".llm.md".length)}.md`;
  return rels.includes(base) ? { base, llm } : null;
}

type DiffState =
  | { kind: "lines"; lines: DiffLine[]; adds: number; dels: number }
  | { kind: "toolarge" }
  | { kind: "error"; message: string };

/** Rendered/source/diff preview for one completed conversion. */
export function MarkdownPreview({
  t,
  item,
  createdAt,
  announce,
}: {
  t: Dict;
  item: SessionItem;
  createdAt: string | null;
  announce: (msg: string) => void;
}) {
  const [tab, setTab] = useState<Tab>("rendered");
  const [result, setResult] = useState<ItemResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [customPdfHeader, setCustomPdfHeader] = useState(readCustomPdfHeader);
  const [rehypeHighlight, setRehypeHighlight] = useState<RehypeHighlight | null>(null);
  const cacheRef = useRef<Map<string, ItemResult>>(new Map());
  const panelRef = useRef<HTMLDivElement>(null);
  const printCleanupRef = useRef<(() => void) | null>(null);
  const renderedTabRef = useRef<HTMLButtonElement>(null);
  const sourceTabRef = useRef<HTMLButtonElement>(null);
  const diffTabRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    let stale = false;
    void import("rehype-highlight").then(
      ({ default: plugin }) => {
        if (!stale) setRehypeHighlight(() => plugin);
      },
      () => undefined,
    );
    return () => {
      stale = true;
    };
  }, []);

  useEffect(() => {
    const cached = cacheRef.current.get(item.key);
    if (cached !== undefined) {
      setResult(cached);
      setError(null);
      return;
    }
    let stale = false;
    setResult(null);
    setError(null);
    fetchItemResult(item.jobId, item.itemId).then(
      (r) => {
        if (stale) return;
        cacheRef.current.set(item.key, r);
        setResult(r);
      },
      (e: unknown) => {
        if (!stale) setError(e instanceof Error ? e.message : String(e));
      },
    );
    return () => {
      stale = true;
    };
  }, [item.key, item.jobId, item.itemId]);

  // ---- diff tab (only when the artifacts hold both .md and .llm.md).
  const pair = useMemo(() => findDiffPair(result), [result]);
  const [diff, setDiff] = useState<DiffState | null>(null);
  const diffCacheRef = useRef<Map<string, DiffState>>(new Map());

  // Item switches land on rendered — the next item may not have a pair, and
  // `result` (so `pair`) lags one fetch behind the selection.
  useEffect(() => {
    setTab((v) => (v === "diff" ? "rendered" : v));
  }, [item.key]);
  useEffect(() => {
    if (tab === "diff" && pair === null) setTab("rendered");
  }, [tab, pair]);

  useEffect(() => {
    if (tab !== "diff" || pair === null) return;
    const cached = diffCacheRef.current.get(item.key);
    if (cached !== undefined) {
      setDiff(cached);
      return;
    }
    let stale = false;
    setDiff(null);
    Promise.all([
      fetchJobFileText(item.jobId, pair.base),
      fetchJobFileText(item.jobId, pair.llm),
    ]).then(
      ([baseText, llmText]) => {
        if (stale) return;
        let st: DiffState;
        if (baseText.split("\n").length + llmText.split("\n").length > MAX_DIFF_LINES) {
          st = { kind: "toolarge" };
        } else {
          const lines = diffLines(baseText, llmText);
          let adds = 0;
          let dels = 0;
          for (const l of lines) {
            if (l.type === "add") adds += 1;
            else if (l.type === "del") dels += 1;
          }
          st = { kind: "lines", lines, adds, dels };
        }
        diffCacheRef.current.set(item.key, st);
        setDiff(st);
      },
      (e: unknown) => {
        if (!stale) {
          setDiff({ kind: "error", message: e instanceof Error ? e.message : String(e) });
        }
      },
    );
    return () => {
      stale = true;
    };
  }, [tab, pair, item.key, item.jobId]);

  useEffect(() => {
    if (!copied) return;
    const id = window.setTimeout(() => setCopied(false), 1500);
    return () => window.clearTimeout(id);
  }, [copied]);

  // Relative refs (e.g. .markitai/assets/x.png) resolve against the job's
  // out dir — rewrite them to the files endpoint so images render.
  const urlTransform = useCallback(
    (url: string) => {
      if (ABSOLUTE_RE.test(url)) return defaultUrlTransform(url);
      const rel = url.replace(/^\.\//, "");
      const decoded = rel
        .split("/")
        .map((seg) => {
          try {
            return decodeURIComponent(seg);
          } catch {
            return seg;
          }
        })
        .join("/");
      return `/api/jobs/${encodeURIComponent(item.jobId)}/files/${encodeRelPath(decoded)}`;
    },
    [item.jobId],
  );

  const markdown = result?.markdown ?? null;
  const documentBase = useMemo(
    () =>
      basename(item.output ?? item.name)
        .replace(/\.llm\.md$/i, "")
        .replace(/\.md$/i, ""),
    [item.name, item.output],
  );
  const meta = useMemo(() => {
    if (markdown === null) return null;
    return {
      words: countWords(markdown),
      bytes: utf8Bytes(markdown),
      split: splitFrontmatter(markdown),
    };
  }, [markdown]);

  // Converted pages are full of links — they must never navigate the SPA
  // away (an accidental click would wipe the session). Fragments stay
  // in-pane; everything else opens a new tab. Extracted figures often have
  // empty alts (base pipeline) — fall back to a name derived from the item.
  const mdComponents = useMemo<Components>(
    () => ({
      table: ({ node: _node, ...props }) => (
        <div className="tablewrap" role="region" aria-label={t.tableAria} tabIndex={0}>
          <table {...props} />
        </div>
      ),
      a: ({ node: _node, href, ...props }) =>
        href !== undefined && href.startsWith("#") ? (
          <a href={href} {...props} />
        ) : (
          <a href={href} target="_blank" rel="noopener noreferrer" {...props} />
        ),
      img: ({ node: _node, alt, src, className, onLoad, ...props }) => (
        <img
          alt={alt !== undefined && alt !== "" ? alt : t.figureFrom(item.name)}
          src={src}
          className={[className, isBadgeImage(src, alt) ? "md-badge" : null]
            .filter(Boolean)
            .join(" ") || undefined}
          onLoad={(event) => {
            // Downloaded assets have numbered local names, so URL heuristics
            // are unavailable. Shields-style intrinsic geometry is a safe
            // fallback and keeps historical README badges inline as well.
            const image = event.currentTarget;
            if (
              image.naturalHeight > 0 &&
              image.naturalHeight <= 40 &&
              image.naturalWidth / image.naturalHeight >= 1.5
            ) {
              image.classList.add("md-badge");
            }
            onLoad?.(event);
          }}
          {...props}
        />
      ),
      pre: ({ node: _node, ...props }) => (
        <pre tabIndex={0} aria-label={t.codeAria} {...props} />
      ),
    }),
    [t, item.name],
  );

  const copySource = () => {
    if (markdown === null) return;
    navigator.clipboard.writeText(markdown).then(
      () => {
        setCopied(true);
        announce(t.copied);
      },
      () => undefined,
    );
  };

  useEffect(
    () => () => {
      printCleanupRef.current?.();
    },
    [],
  );

  const exportPdf = () => {
    if (markdown === null || panelRef.current === null) return;
    printCleanupRef.current?.();
    const previousTitle = document.title;
    // Browsers append .pdf to the suggested filename themselves. Keeping the
    // document title extension-free also makes Chrome's optional native print
    // header less awkward when the user enables it in the print dialog.
    document.title = documentBase;
    document.body.classList.add("printing-preview");
    if (isCurrentThemeDark()) {
      document.body.classList.add("printing-dark-theme");
      document.documentElement.classList.add("printing-dark-theme");
    }
    if (customPdfHeader) {
      document.body.classList.add("printing-custom-header-footer");
    }
    panelRef.current.classList.add("pdf-export-target");
    let cleaned = false;
    const cleanup = () => {
      if (cleaned) return;
      cleaned = true;
      document.title = previousTitle;
      document.body.classList.remove(
        "printing-preview",
        "printing-dark-theme",
        "printing-custom-header-footer",
      );
      document.documentElement.classList.remove("printing-dark-theme");
      panelRef.current?.classList.remove("pdf-export-target");
      window.removeEventListener("afterprint", cleanup);
      printCleanupRef.current = null;
    };
    printCleanupRef.current = cleanup;
    window.addEventListener("afterprint", cleanup, { once: true });
    window.print();
    window.setTimeout(cleanup, 60_000);
    announce(t.exportPdf);
  };

  const tabOrder: Tab[] = pair !== null ? ["rendered", "source", "diff"] : ["rendered", "source"];
  const tabRefs: Record<Tab, React.RefObject<HTMLButtonElement | null>> = {
    rendered: renderedTabRef,
    source: sourceTabRef,
    diff: diffTabRef,
  };
  const onTabKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const idx = tabOrder.indexOf(tab);
    const next =
      tabOrder[
        (idx + (e.key === "ArrowRight" ? 1 : tabOrder.length - 1)) % tabOrder.length
      ] ?? "rendered";
    setTab(next);
    tabRefs[next].current?.focus();
  };

  const docDate = createdAt !== null ? fmtDate(createdAt) : null;
  const variantLabel =
    result === null ? null : result.variant === "llm" ? "LLM enhanced" : "Base";
  const renderedBody = useMemo(
    () => portableLocalImageDestinations(meta?.split.body ?? markdown ?? ""),
    [markdown, meta?.split.body],
  );
  const renderedMarkdown = () => (
    <div className="md">
      <p className="docmeta">
        {item.name}
        {docDate !== null && ` · ${docDate}`}
        {variantLabel !== null && ` · ${variantLabel}`}
      </p>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={
          rehypeHighlight === null
            ? []
            : [[rehypeHighlight, { detect: false, ignoreMissing: true }]]
        }
        urlTransform={urlTransform}
        components={mdComponents}
        skipHtml // raw HTML (e.g. page-marker comments) stays in Source
      >
        {renderedBody}
      </ReactMarkdown>
    </div>
  );
  const printHeader = () => (
    <div className="pdf-print-header" aria-hidden="true">
      <span className="pdf-print-brand">
        <LogoMark size={20} />
        <strong>markitai</strong>
      </span>
      <span className="pdf-print-title">{documentBase}</span>
    </div>
  );
  const printFooter = () => (
    <div className="pdf-print-footer" aria-hidden="true">
      <span>{t.pdfPreparedBy}</span>
      <span className="pdf-print-source">
        {t.pdfSource}: {item.name}
        {docDate !== null && ` · ${docDate}`}
      </span>
    </div>
  );

  return (
    <div ref={panelRef} className="panel">
      <div className="panelbar">
        <div className="tabs" role="tablist" aria-label={t.previewAria}>
          <button
            ref={renderedTabRef}
            type="button"
            role="tab"
            id="tab-rendered"
            aria-selected={tab === "rendered"}
            aria-controls="pane-rendered"
            tabIndex={tab === "rendered" ? 0 : -1}
            className={tab === "rendered" ? "tab on" : "tab"}
            onClick={() => setTab("rendered")}
            onKeyDown={onTabKey}
          >
            {t.rendered}
          </button>
          <button
            ref={sourceTabRef}
            type="button"
            role="tab"
            id="tab-source"
            aria-selected={tab === "source"}
            aria-controls="pane-source"
            tabIndex={tab === "source" ? 0 : -1}
            className={tab === "source" ? "tab on" : "tab"}
            onClick={() => setTab("source")}
            onKeyDown={onTabKey}
          >
            {t.source}
          </button>
          {pair !== null && (
            <button
              ref={diffTabRef}
              type="button"
              role="tab"
              id="tab-diff"
              aria-selected={tab === "diff"}
              aria-controls="pane-diff"
              tabIndex={tab === "diff" ? 0 : -1}
              className={tab === "diff" ? "tab on" : "tab"}
              onClick={() => setTab("diff")}
              onKeyDown={onTabKey}
            >
              {t.diffTab}
            </button>
          )}
        </div>
        <div className="side">
          {meta !== null && (
            <span className="meta">
              {meta.words.toLocaleString("en-US")} {t.words} · {fmtBytes(meta.bytes)}
            </span>
          )}
          <span className="pdf-header-option" title={t.pdfPrintDialogHint}>
            <span>{t.pdfCustomHeaderFooter}</span>
            <button
              type="button"
              role="switch"
              aria-checked={customPdfHeader}
              aria-label={t.pdfCustomHeaderFooter}
              className={customPdfHeader ? "switch on" : "switch"}
              disabled={markdown === null}
              onClick={() => {
                const next = !customPdfHeader;
                setCustomPdfHeader(next);
                try {
                  localStorage.setItem(PDF_HEADER_KEY, String(next));
                } catch {
                  /* localStorage unavailable */
                }
              }}
            />
          </span>
          <button
            type="button"
            className="btn ghost sm"
            disabled={markdown === null}
            title={t.exportPdf}
            aria-label={t.exportPdf}
            onClick={exportPdf}
          >
            <PdfIcon size={13} />
            <span className="action-label">{t.exportPdf}</span>
          </button>
          {item.output !== null && (
            <a
              className="btn ghost sm"
              href={jobFileUrl(item.jobId, item.output)}
              download
              aria-label={t.downloadMd}
            >
              <DownloadIcon size={13} />
              <span className="action-label">{t.downloadMd}</span>
            </a>
          )}
        </div>
      </div>

      <div
        id="pane-rendered"
        role="tabpanel"
        aria-labelledby="tab-rendered"
        className="pane"
        tabIndex={0}
        hidden={tab !== "rendered"}
      >
        {markdown === null ? (
          <p className={error === null ? "pane-note" : "pane-note errline"}>
            {error ?? t.loading}
          </p>
        ) : (
          <div className="pdf-print-document">
            {customPdfHeader && printHeader()}
            {renderedMarkdown()}
            {customPdfHeader && printFooter()}
          </div>
        )}
      </div>

      <div
        id="pane-source"
        role="tabpanel"
        aria-labelledby="tab-source"
        className="pane"
        hidden={tab !== "source"}
      >
        {markdown === null ? (
          <p className={error === null ? "pane-note" : "pane-note errline"}>
            {error ?? t.loading}
          </p>
        ) : (
          <div className="srcwrap">
            <div className="term">
              <div className="term-head">
                <span className="tname">
                  {item.output !== null ? basename(item.output) : item.name}
                  {meta !== null && ` · ${fmtBytes(meta.bytes)} · utf-8`}
                </span>
                <button type="button" className="badge" onClick={copySource}>
                  {copied ? t.copied : t.copy}
                </button>
              </div>
              <pre tabIndex={0} aria-label={t.srcAria}>
                {meta !== null && meta.split.fm !== null && (
                  <span className="fm">{meta.split.fm}</span>
                )}
                {meta !== null ? meta.split.body : markdown}
              </pre>
            </div>
          </div>
        )}
      </div>

      {pair !== null && (
        <div
          id="pane-diff"
          role="tabpanel"
          aria-labelledby="tab-diff"
          className="pane"
          tabIndex={0}
          hidden={tab !== "diff"}
        >
          {diff === null ? (
            <p className="pane-note">{t.loading}</p>
          ) : diff.kind === "error" ? (
            <p className="pane-note errline">{diff.message}</p>
          ) : diff.kind === "toolarge" ? (
            <p className="pane-note">{t.diffTooLarge}</p>
          ) : (
            <div className="diffview" aria-label={t.diffAria}>
              <p className="docmeta diffmeta">
                {basename(pair.base)} → {basename(pair.llm)} · +{diff.adds} -{diff.dels}
              </p>
              {diff.lines.map((l, i) => (
                <div
                  key={i}
                  className={
                    l.type === "add" ? "dline add" : l.type === "del" ? "dline del" : "dline"
                  }
                >
                  <span className="dno" aria-hidden="true">
                    {l.aNo ?? ""}
                  </span>
                  <span className="dno" aria-hidden="true">
                    {l.bNo ?? ""}
                  </span>
                  <span className="dmark" aria-hidden="true">
                    {l.type === "add" ? "+" : l.type === "del" ? "-" : " "}
                  </span>
                  <span className="dtext">{l.text}</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
