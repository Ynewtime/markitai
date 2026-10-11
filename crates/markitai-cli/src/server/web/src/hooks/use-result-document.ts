// One item's result as the preview shows it: the whole result, the exact
// base/LLM pair when there is one, the version on screen, and that version's
// Markdown (the other version of a pair is read on demand).
import { useEffect, useMemo, useState } from "preact/hooks";
import { fetchText, filePath } from "../api/client.ts";
import type { ItemResult } from "../api/types.ts";
import { markdownPair } from "../lib/markdown.ts";
import { cachedResult, loadResult, resultKey } from "../lib/results.ts";
import type { SessionItem } from "../lib/session.ts";

export interface ResultDocument {
  result: ItemResult | null;
  error: unknown;
  pair: { base: string; llm: string } | null;
  shown: "base" | "llm";
  setVersion: (version: "base" | "llm") => void;
  /** The relpath of the Markdown on screen. */
  path: string;
  /** The Markdown on screen; null while it loads. */
  markdown: string | null;
  /** The text of a pair member already read, by relpath. */
  text: (path: string) => string | null;
}

export function useResultDocument(item: SessionItem): ResultDocument {
  const key = resultKey(item);
  const [result, setResult] = useState<ItemResult | null>(() => cachedResult(key));
  const [loadError, setLoadError] = useState<unknown>(null);
  const [version, setVersion] = useState<"base" | "llm" | null>(null);
  const [texts, setTexts] = useState<Record<string, string>>({});
  const [textError, setTextError] = useState<unknown>(null);

  useEffect(() => {
    setVersion(null);
    setTexts({});
    setTextError(null);
    const cached = cachedResult(key);
    setResult(cached);
    setLoadError(null);
    if (cached) return;
    let stale = false;
    loadResult(item).then(
      (value) => !stale && setResult(value),
      (error: unknown) => !stale && setLoadError(error),
    );
    return () => {
      stale = true;
    };
  }, [key]);

  const pair = useMemo(() => (result ? markdownPair(result.artifacts) : null), [result]);
  const shown = version ?? (result?.variant === "llm" ? "llm" : "base");
  const path = useMemo(() => {
    if (!result) return item.output ?? "";
    if (pair) return pair[shown];
    const markdown = result.artifacts.find((artifact) => artifact.relpath === item.output) ?? result.artifacts.find((artifact) => artifact.relpath.endsWith(".md"));
    return markdown?.relpath ?? item.output ?? "";
  }, [result, pair, shown, item.output]);
  const resultPath = pair ? pair[result?.variant === "llm" ? "llm" : "base"] : path;
  const markdown = result === null ? null : path === resultPath ? result.markdown : (texts[path] ?? null);

  useEffect(() => {
    if (!result || markdown !== null || !path) return;
    let stale = false;
    setTextError(null);
    fetchText(filePath(item.jobId, path), 64 * 1024 * 1024).then(
      (text) => !stale && setTexts((previous) => ({ ...previous, [path]: text })),
      (error: unknown) => !stale && setTextError(error),
    );
    return () => {
      stale = true;
    };
  }, [result, markdown, path, item.jobId]);

  return {
    result,
    error: loadError ?? textError,
    pair,
    shown,
    setVersion,
    path,
    markdown,
    text: (wanted) => (result && wanted === resultPath ? result.markdown : (texts[wanted] ?? null)),
  };
}
