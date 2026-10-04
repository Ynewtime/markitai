import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import type { CloudflareCapability, JobOptions } from "../api/types.ts";
import { cloudflareBatchReason, cloudflareBatchScope, confirmedCloudflareRequests, freezeCloudflareRequests, type CloudflareRequest, type RemoteSource } from "../lib/cloudflare.ts";

interface Pending {
  requests: CloudflareRequest[];
  retry: boolean;
  resolve: (value: JobOptions[] | null) => void;
  opener: HTMLElement | null;
}

/** Consent lives only in this frozen request list; refresh, cancel and unmount discard it. */
export function useCloudflareConsent(capability?: CloudflareCapability) {
  const caps = useRef(capability);
  caps.current = capability;
  const pendingRef = useRef<Pending | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const finish = useCallback((confirmed: boolean) => {
    const current = pendingRef.current;
    if (!current) return;
    pendingRef.current = null;
    setPending(null);
    current.resolve(confirmed ? confirmedCloudflareRequests(current.requests, true, caps.current) : null);
    requestAnimationFrame(() => {
      const fallback = document.querySelector<HTMLElement>(".workspace .url-box, .landing .url-box");
      (current.opener?.isConnected ? current.opener : fallback)?.focus();
    });
  }, []);
  useEffect(() => () => {
    pendingRef.current?.resolve(null);
    pendingRef.current = null;
  }, []);
  const authorizeBatch = useCallback((requests: readonly CloudflareRequest[], retry = false): Promise<JobOptions[] | null> => {
    if (pendingRef.current) return Promise.resolve(null);
    const frozen = freezeCloudflareRequests(requests);
    if (!cloudflareBatchScope(frozen, caps.current).selected) return Promise.resolve(confirmedCloudflareRequests(frozen, false));
    return new Promise((resolve) => {
      const next = { requests: frozen, retry, resolve,
        opener: document.activeElement instanceof HTMLElement ? document.activeElement : null };
      pendingRef.current = next;
      setPending(next);
    });
  }, []);
  const authorize = useCallback(async (options: JobOptions, sources: RemoteSource[], retry = false): Promise<JobOptions | null> => {
    const accepted = await authorizeBatch([{ options, sources }], retry);
    return accepted?.[0] ?? null;
  }, [authorizeBatch]);
  return { authorize, authorizeBatch, pending, finish,
    scope: pending ? cloudflareBatchScope(pending.requests, capability) : null,
    reason: pending ? cloudflareBatchReason(pending.requests, capability) : null };
}
