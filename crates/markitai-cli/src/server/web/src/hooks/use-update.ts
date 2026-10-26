// The update of a loopback server (contract v1.24) as the page follows it:
// `GET /api/update` every 30 s at rest and every 1–3 s while the server works
// on one; `POST` to start, `DELETE` to cancel. Once the server restarts, the
// page asks every second until it answers again; a new version reloads the
// page (it says so after the reload), unless something unsaved would be lost,
// when a notice offers the reload instead. A server that came back on the
// same version shows why in the card.
import { useCallback, useEffect, useRef, useState } from "preact/hooks";
import { cancelUpdate, fetchCapabilities, fetchUpdate, NetworkError, startUpdate } from "../api/client.ts";
import type { Capabilities, UpdateStatus } from "../api/types.ts";
import { BUSY, entryLook, errorKey, HOLDING, pollDelay, shownError, type EntryLook } from "../lib/update.ts";

/** This tab reloaded into a new version: said once after the reload. */
export const DONE_KEY = "markitai.update.done";
/** The failure "OK" was pressed on, in this browser. */
const DISMISSED_KEY = "markitai.update.dismissed";
const RECONNECT_MS = 1000;

const read = (storage: () => Storage, key: string): string | null => {
  try {
    return storage().getItem(key);
  } catch {
    return null;
  }
};
const write = (storage: () => Storage, key: string, value: string | null): void => {
  try {
    if (value === null) storage().removeItem(key);
    else storage().setItem(key, value);
  } catch {
    /* Kept for this page only. */
  }
};

/** The build a page was loaded from: a different one after a restart means a new bundle. */
const buildOf = (caps: Pick<Capabilities, "version" | "build">): string => `${caps.build?.version ?? caps.version}/${caps.build?.commit ?? ""}`;

export interface UpdateControl {
  /** How the entry looks; null hides it (no `update` from the server, or nothing to say). */
  look: EntryLook | null;
  status: UpdateStatus | null;
  open: boolean;
  setOpen: (open: boolean) => void;
  confirming: boolean;
  setConfirming: (confirming: boolean) => void;
  /** Seconds spent waiting for conversions (the status row's clock). */
  waited: number;
  /** A request of the page's own is under way (start, cancel). */
  sending: boolean;
  start: () => void;
  cancel: () => void;
  dismiss: () => void;
  /** New conversions are held: the server waits for conversions, installs or restarts. */
  holding: boolean;
  /** The server restarted and the page waits for it to answer. */
  reconnecting: boolean;
  /** The version this page runs on: the one it loaded from until it reconnects to another. */
  running: string | null;
}

export function useUpdate({
  caps,
  hasUnsaved,
  onFailure,
  onReloadOffer,
  onBack,
}: {
  caps: Capabilities | null;
  /** Edits a reload would lose (the config editor, an open form, staged input). */
  hasUnsaved: () => boolean;
  onFailure: (error: unknown) => void;
  /** The new version answers but a reload would lose edits: offer it. */
  onReloadOffer: (version: string) => void;
  /** The server answers again on the same version (rolled back), or after a reload offer. */
  onBack: () => void;
}): UpdateControl {
  const summary = caps?.update;
  const enabled = summary !== undefined;
  const [status, setStatus] = useState<UpdateStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [sending, setSending] = useState(false);
  const [reconnecting, setReconnecting] = useState(false);
  const [dismissed, setDismissed] = useState(() => read(() => localStorage, DISMISSED_KEY));
  const [waitStart, setWaitStart] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  const loadedBuild = useRef<string | null>(null);
  const loadedVersion = useRef<string | null>(null);
  if (caps && loadedBuild.current === null) {
    loadedBuild.current = buildOf(caps);
    loadedVersion.current = caps.update?.current ?? null;
  }
  // The server reports the installed version as soon as it is installed; the page keeps saying
  // the one it runs on until it has reconnected.
  const [reconnectedTo, setReconnectedTo] = useState<string | null>(null);
  const running = reconnectedTo ?? (caps ? (loadedVersion.current ?? caps.build?.version ?? caps.version) : null);
  const statusRef = useRef(status);
  statusRef.current = status;
  const handlers = useRef({ hasUnsaved, onFailure, onReloadOffer, onBack });
  handlers.current = { hasUnsaved, onFailure, onReloadOffer, onBack };

  const refresh = useCallback(async () => {
    try {
      const next = await fetchUpdate();
      setStatus(next);
      // A restart can be quicker than a poll: another version answering is the same news.
      if (next.state === "restarting" || (loadedVersion.current !== null && next.current !== null && next.current !== loadedVersion.current)) setReconnecting(true);
      return next;
    } catch (error) {
      // The server went away while it worked on an update: it is restarting.
      const last = statusRef.current?.state;
      if (error instanceof NetworkError && last !== undefined && BUSY.has(last)) setReconnecting(true);
      return null;
    }
  }, []);

  // At rest every 30 s (another tab or `markitai update` may start one), quickly while it works.
  const state = status?.state ?? summary?.state ?? null;
  useEffect(() => {
    if (!enabled || reconnecting) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const next = await refresh();
      if (stop) return;
      timer = setTimeout(() => void tick(), pollDelay(next?.state ?? state, open));
    };
    void tick();
    return () => {
      stop = true;
      if (timer !== null) clearTimeout(timer);
    };
  }, [enabled, reconnecting, open, state === null ? null : BUSY.has(state), refresh]);

  // While waiting for conversions, the row counts the seconds.
  useEffect(() => {
    if (status?.state !== "waiting") {
      setWaitStart(null);
      return;
    }
    setWaitStart((started) => started ?? Date.now());
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [status?.state]);

  // After the restart: ask every second until the server answers.
  useEffect(() => {
    if (!reconnecting) return;
    let stop = false;
    const timer = setInterval(() => {
      void fetchCapabilities().then(
        (next) => {
          if (stop) return;
          stop = true;
          clearInterval(timer);
          const target = next.update?.current ?? next.build?.version ?? next.version;
          if (loadedBuild.current !== null && buildOf(next) !== loadedBuild.current) {
            write(() => sessionStorage, DONE_KEY, target);
            if (!handlers.current.hasUnsaved()) {
              location.reload();
              return;
            }
            handlers.current.onReloadOffer(target);
          }
          setReconnectedTo(target);
          setReconnecting(false);
          setOpen(false);
          void refresh();
          handlers.current.onBack();
        },
        () => undefined,
      );
    }, RECONNECT_MS);
    return () => {
      stop = true;
      clearInterval(timer);
    };
  }, [reconnecting, refresh]);

  const start = useCallback(() => {
    const latest = statusRef.current?.latest;
    if (!latest) return;
    setSending(true);
    void startUpdate(latest)
      .then(
        (next) => {
          setStatus(next);
          setConfirming(false);
        },
        (error: unknown) => {
          handlers.current.onFailure(error);
          void refresh();
        },
      )
      .finally(() => setSending(false));
  }, [refresh]);

  const cancel = useCallback(() => {
    setSending(true);
    void cancelUpdate()
      .then(
        (next) => setStatus(next),
        (error: unknown) => handlers.current.onFailure(error),
      )
      .finally(() => {
        setSending(false);
        void refresh();
      });
  }, [refresh]);

  const dismiss = useCallback(() => {
    const error = statusRef.current?.error;
    if (error) {
      const key = errorKey(error);
      write(() => localStorage, DISMISSED_KEY, key);
      setDismissed(key);
    }
    setOpen(false);
  }, []);

  // A failure the page has not seen yet opens no card by itself; the entry turns red.
  const look = enabled ? entryLook(summary, status, dismissed) : null;
  const shown = shownError(status, dismissed);
  useEffect(() => {
    if (look === null && !reconnecting) setOpen(false);
  }, [look, reconnecting]);
  const current = status?.state ?? summary?.state ?? "idle";
  return {
    look: reconnecting ? "busy" : look,
    status: status && shown === null && status.state === "failed" ? { ...status, state: "idle" } : status,
    open,
    setOpen,
    confirming,
    setConfirming,
    waited: waitStart === null ? 0 : (now - waitStart) / 1000,
    sending,
    start,
    cancel,
    dismiss,
    holding: reconnecting || HOLDING.has(current),
    reconnecting,
    running,
  };
}

/** The version this tab reloaded into after an update, read once. */
export function takeUpdated(): string | null {
  const version = read(() => sessionStorage, DONE_KEY);
  if (version !== null) write(() => sessionStorage, DONE_KEY, null);
  return version;
}
