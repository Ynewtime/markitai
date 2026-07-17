import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const DENIED_KEY = "markitai.notifyDenied";

class FakeNotification {
  static permission: NotificationPermission = "default";
  static requestPermission = vi.fn<() => Promise<NotificationPermission>>();
  static created: { title: string; body: string | undefined }[] = [];
  onclick: (() => void) | null = null;
  constructor(title: string, options?: NotificationOptions) {
    FakeNotification.created.push({ title, body: options?.body });
  }
  close(): void {}
}

/** notify.ts keeps an in-flight flag at module scope, so each test gets a
 * fresh copy of the module. */
async function loadNotify(): Promise<typeof import("./notify")> {
  vi.resetModules();
  return import("./notify");
}

/** Node 25 ships an experimental localStorage global that is undefined
 * without --localstorage-file and shadows jsdom's, so stub a working one. */
function fakeStorage(): Storage {
  const map = new Map<string, string>();
  return {
    get length() {
      return map.size;
    },
    clear: () => map.clear(),
    getItem: (k: string) => map.get(k) ?? null,
    key: (i: number) => [...map.keys()][i] ?? null,
    removeItem: (k: string) => {
      map.delete(k);
    },
    setItem: (k: string, v: string) => {
      map.set(k, v);
    },
  };
}

function setHidden(hidden: boolean): void {
  Object.defineProperty(document, "visibilityState", {
    configurable: true,
    get: () => (hidden ? "hidden" : "visible"),
  });
  Object.defineProperty(document, "hidden", {
    configurable: true,
    get: () => hidden,
  });
}

beforeEach(() => {
  vi.stubGlobal("localStorage", fakeStorage());
  FakeNotification.permission = "default";
  FakeNotification.requestPermission.mockReset().mockResolvedValue("granted");
  FakeNotification.created = [];
  vi.stubGlobal("Notification", FakeNotification);
  setHidden(false);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("requestNotifyPermission", () => {
  it("never re-requests after a stored denial", async () => {
    localStorage.setItem(DENIED_KEY, "1");
    const { requestNotifyPermission } = await loadNotify();
    requestNotifyPermission();
    expect(FakeNotification.requestPermission).not.toHaveBeenCalled();
  });

  it("asks only once while a request is in flight", async () => {
    const { requestNotifyPermission } = await loadNotify();
    requestNotifyPermission();
    requestNotifyPermission();
    expect(FakeNotification.requestPermission).toHaveBeenCalledTimes(1);
  });

  it("stores a denial so later sessions skip the prompt", async () => {
    FakeNotification.requestPermission.mockResolvedValue("denied");
    const { requestNotifyPermission } = await loadNotify();
    requestNotifyPermission();
    await vi.waitFor(() => expect(localStorage.getItem(DENIED_KEY)).toBe("1"));
  });
});

describe("notifyJobDone", () => {
  it("does nothing while the document is visible", async () => {
    const { notifyJobDone } = await loadNotify();
    FakeNotification.permission = "granted";
    setHidden(false);
    notifyJobDone("done");
    expect(FakeNotification.created).toHaveLength(0);
  });

  it("does nothing without granted permission", async () => {
    const { notifyJobDone } = await loadNotify();
    FakeNotification.permission = "default";
    setHidden(true);
    notifyJobDone("done");
    expect(FakeNotification.created).toHaveLength(0);
  });

  it("notifies when hidden and granted", async () => {
    const { notifyJobDone } = await loadNotify();
    FakeNotification.permission = "granted";
    setHidden(true);
    notifyJobDone("All files converted.");
    expect(FakeNotification.created).toEqual([
      { title: "markitai", body: "All files converted." },
    ]);
  });
});
