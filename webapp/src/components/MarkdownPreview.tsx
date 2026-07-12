import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { encodeRelPath, fetchItemResult, jobFileUrl } from "../api/client";
import type { ItemResult } from "../api/types";
import type { SessionItem } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { countWords, fmtBytes, fmtDate, utf8Bytes } from "../lib/format";
import { DownloadIcon, EyeOffIcon } from "./icons";

type Tab = "rendered" | "source";

/** True for URLs with a scheme, absolute paths, fragments and queries. */
const ABSOLUTE_RE = /^([a-z][a-z0-9+.-]*:|\/|#|\?)/i;

function splitFrontmatter(md: string): { fm: string | null; body: string } {
  const m = /^---\n[\s\S]*?\n---(?=\n|$)/.exec(md);
  if (m === null) return { fm: null, body: md };
  return { fm: m[0], body: md.slice(m[0].length) };
}

function basename(relpath: string): string {
  return relpath.split("/").pop() ?? relpath;
}

/** Right panel: rendered/source tabs over the selected item's markdown.
 * Source lives in the brand's always-dark terminal card. Both panes scroll
 * internally under a viewport-height cap; `expanded` lifts the cap and hides
 * the list column for full-width reading. */
export function MarkdownPreview({
  t,
  item,
  createdAt,
  expanded,
  onToggleExpand,
  onHide,
  announce,
}: {
  t: Dict;
  item: SessionItem;
  createdAt: string | null;
  expanded: boolean;
  onToggleExpand: () => void;
  onHide: () => void;
  announce: (msg: string) => void;
}) {
  const [tab, setTab] = useState<Tab>("rendered");
  const [result, setResult] = useState<ItemResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const cacheRef = useRef<Map<string, ItemResult>>(new Map());
  const renderedTabRef = useRef<HTMLButtonElement>(null);
  const sourceTabRef = useRef<HTMLButtonElement>(null);

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
      img: ({ node: _node, alt, ...props }) => (
        <img alt={alt !== undefined && alt !== "" ? alt : t.figureFrom(item.name)} {...props} />
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

  const onTabKey = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const next: Tab = tab === "rendered" ? "source" : "rendered";
    setTab(next);
    (next === "rendered" ? renderedTabRef : sourceTabRef).current?.focus();
  };

  const docDate = createdAt !== null ? fmtDate(createdAt) : null;
  const variantLabel =
    result === null ? null : result.variant === "llm" ? "llm enhanced" : "base";

  return (
    <div className="panel">
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
        </div>
        <div className="side">
          {meta !== null && (
            <span className="meta">
              {meta.words.toLocaleString("en-US")} {t.words} · {fmtBytes(meta.bytes)}
            </span>
          )}
          <span className="pgroup">
            <button
              type="button"
              className="btn ghost sm"
              aria-pressed={expanded}
              onClick={onToggleExpand}
            >
              {expanded ? t.collapse : t.expand}
            </button>
            <button
              type="button"
              className="btn ghost sm icon"
              aria-label={t.hidePreview}
              title={t.hidePreview}
              onClick={onHide}
            >
              <EyeOffIcon size={14} />
            </button>
          </span>
          {item.output !== null && (
            <a
              className="btn ghost sm"
              href={jobFileUrl(item.jobId, item.output)}
              download
            >
              <DownloadIcon size={13} />
              {t.downloadMd}
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
          <div className="md">
            <p className="docmeta">
              {item.name}
              {docDate !== null && ` · ${docDate}`}
              {variantLabel !== null && ` · ${variantLabel}`}
            </p>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              urlTransform={urlTransform}
              components={mdComponents}
              skipHtml // raw HTML (e.g. page-marker comments) stays in Source
            >
              {meta?.split.body ?? markdown}
            </ReactMarkdown>
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
    </div>
  );
}
