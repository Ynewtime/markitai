// An item's original descriptor (`GET …/original`), read once per item and
// shared by Compare and Preview's Files.
import { useEffect, useState } from "preact/hooks";
import { fetchOriginal } from "../api/client.ts";
import type { OriginalDescriptor } from "../api/types.ts";
import type { SessionItem } from "../lib/session.ts";

const descriptors = new Map<string, OriginalDescriptor>();

export interface OriginalState {
  descriptor: OriginalDescriptor | null;
  error: unknown;
}

/** `enabled` false (the server keeps no originals) reads nothing. */
export function useOriginal(item: SessionItem, enabled = true): OriginalState {
  // A rerun gives the item a new result, and may replace its original.
  const key = `${item.key}|${item.finishedAt ?? ""}`;
  const [state, setState] = useState<OriginalState>(() => ({ descriptor: descriptors.get(key) ?? null, error: null }));
  useEffect(() => {
    const known = descriptors.get(key) ?? null;
    setState({ descriptor: known, error: null });
    if (known || !enabled) return;
    let stale = false;
    fetchOriginal(item.jobId, item.itemId).then(
      (value) => {
        descriptors.set(key, value);
        if (!stale) setState({ descriptor: value, error: null });
      },
      (error: unknown) => !stale && setState({ descriptor: null, error }),
    );
    return () => {
      stale = true;
    };
  }, [key, enabled]);
  return state;
}
