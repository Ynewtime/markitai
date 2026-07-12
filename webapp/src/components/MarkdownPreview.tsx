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
import { DownloadIcon, EyeOffIcon } from "./icons";

type Tab = "rendered" | "source" | "diff";

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
  const diffTabRef = useRef<HTMLButtonElement>(null);

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
