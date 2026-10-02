// A system notification when a job finishes while the tab is hidden. Permission
// is asked on the first submission only; a refusal is remembered, never re-asked.

const DENIED_KEY = "markitai.notify-denied";
let asking = false;

const supported = () => typeof window !== "undefined" && "Notification" in window;

function denied(): boolean {
  try {
    return localStorage.getItem(DENIED_KEY) === "1";
  } catch {
    return false;
  }
}

export function askNotifyPermission(): void {
  if (!supported() || Notification.permission !== "default" || denied() || asking) return;
  asking = true;
  Promise.resolve(Notification.requestPermission())
    .then((answer) => {
      if (answer === "denied") {
        try {
          localStorage.setItem(DENIED_KEY, "1");
        } catch {
          /* Asked again next session at most. */
        }
      }
    })
    .catch(() => undefined)
    .finally(() => {
      asking = false;
    });
}

export function notifyDone(body: string): void {
  if (!supported() || !document.hidden || Notification.permission !== "granted") return;
  try {
    const note = new Notification("Markitai", { body });
    note.onclick = () => {
      window.focus();
      note.close();
    };
  } catch {
    /* Some platforms require a service worker for notifications. */
  }
}
