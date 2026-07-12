/** Completion notifications. Permission is requested only on the first job
 * submit (never on page load); a past denial is remembered in localStorage
 * and never re-asked. Environments without Notification skip silently. */

const DENIED_KEY = "markitai.notifyDenied";

function supported(): boolean {
  return typeof window !== "undefined" && "Notification" in window;
}

function deniedStored(): boolean {
  try {
    return localStorage.getItem(DENIED_KEY) === "1";
  } catch {
    return false;
  }
}

let requesting = false;

export function requestNotifyPermission(): void {
  if (!supported()) return;
  if (Notification.permission !== "default") return;
  if (deniedStored() || requesting) return;
  requesting = true;
  void Promise.resolve(Notification.requestPermission())
    .then((p) => {
      if (p === "denied") {
        try {
          localStorage.setItem(DENIED_KEY, "1");
        } catch {
          /* localStorage unavailable */
        }
      }
    })
    .catch(() => undefined)
    .finally(() => {
      requesting = false;
    });
}

/** One system notification for a job that reached its terminal state while
 * the tab was hidden. Clicking it focuses the window. */
export function notifyJobDone(body: string): void {
  if (!supported()) return;
  if (!document.hidden) return;
  if (Notification.permission !== "granted") return;
  try {
    const n = new Notification("markitai", { body });
    n.onclick = () => {
      window.focus();
      n.close();
    };
  } catch {
    /* constructor may throw (e.g. platforms requiring a ServiceWorker) */
  }
}
