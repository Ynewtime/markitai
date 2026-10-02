// Whether the service answers. A request that cannot reach it reports
// `markitai:offline`; from then on the page asks again every five seconds and
// calls `onBack` once it answers.
import { useEffect, useRef, useState } from "preact/hooks";
import { fetchCapabilities } from "../api/client.ts";

export const RECONNECT_MS = 5000;

export function useConnectivity(onBack: () => void): boolean {
  const [offline, setOffline] = useState(false);
  const back = useRef(onBack);
  back.current = onBack;
  const offlineRef = useRef(false);

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null;
    let checking = false;
    const probe = async () => {
      if (checking) return;
      checking = true;
      try {
        await fetchCapabilities();
      } catch {
        /* Still unreachable, or refused; either way the offline event decides. */
      } finally {
        checking = false;
      }
    };
    const goOffline = () => {
      if (offlineRef.current) return;
      offlineRef.current = true;
      setOffline(true);
      timer ??= setInterval(() => void probe(), RECONNECT_MS);
    };
    const goOnline = () => {
      if (!offlineRef.current) return;
      offlineRef.current = false;
      setOffline(false);
      if (timer !== null) clearInterval(timer);
      timer = null;
      back.current();
    };
    // A broken event stream asks for one check; requests themselves report the result.
    let lastCheck = 0;
    const check = () => {
      if (offlineRef.current || Date.now() - lastCheck < RECONNECT_MS) return;
      lastCheck = Date.now();
      void probe();
    };
    window.addEventListener("markitai:offline", goOffline);
    window.addEventListener("markitai:online", goOnline);
    window.addEventListener("markitai:check", check);
    return () => {
      window.removeEventListener("markitai:offline", goOffline);
      window.removeEventListener("markitai:online", goOnline);
      window.removeEventListener("markitai:check", check);
      if (timer !== null) clearInterval(timer);
    };
  }, []);
  return offline;
}
