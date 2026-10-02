// Export PDF prints a temporary clone of the sanitized rendered document through
// the browser's print dialog. Links are stripped from the clone so that a
// generated PDF never embeds a URL carrying the service token, and printing
// waits for every image, refusing rather than producing a silently incomplete file.

export class PrintCancelled extends Error {
  constructor() {
    super("cancelled");
    this.name = "AbortError";
  }
}

export function waitForImages(
  images: HTMLImageElement[],
  { signal, timeoutMs = 10_000, loading = "loading", broken = "broken" }: { signal?: AbortSignal; timeoutMs?: number; loading?: string; broken?: string } = {},
): Promise<void> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const finish = (error?: Error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      signal?.removeEventListener("abort", cancel);
      if (error) reject(error);
      else resolve();
    };
    const cancel = () => finish(new PrintCancelled());
    const timer = setTimeout(() => finish(new Error(loading)), timeoutMs);
    signal?.addEventListener("abort", cancel, { once: true });
    if (signal?.aborted) {
      cancel();
      return;
    }
    Promise.all(
      images.map(async (image) => {
        image.loading = "eager";
        // decode() rejects with the browser's own wording; report one explicit reason.
        if (!image.complete) await image.decode().catch(() => undefined);
        if (!image.naturalWidth) throw new Error(broken);
      }),
    ).then(
      () => finish(),
      (error: Error) => finish(error),
    );
  });
}

export interface PrintJob {
  done: Promise<void>;
  cancel(): void;
}

export interface PrintOptions {
  source: HTMLElement;
  title: string;
  furniture: boolean;
  isCurrent?: () => boolean;
  doc?: Document;
  win?: Window;
  timeoutMs?: number;
  messages?: { loading: string; broken: string };
}

export function printDocument({
  source,
  title,
  furniture,
  isCurrent = () => true,
  doc = document,
  win = window,
  timeoutMs = 10_000,
  messages = { loading: "loading", broken: "broken" },
}: PrintOptions): PrintJob {
  const controller = new AbortController();
  const host = doc.createElement("section");
  host.className = "print-host";
  const clone = source.cloneNode(true) as HTMLElement;
  clone.removeAttribute("id");
  clone.removeAttribute("hidden");
  for (const link of [...clone.querySelectorAll("a")]) {
    link.removeAttribute("href");
    link.removeAttribute("target");
    link.removeAttribute("download");
  }
  host.append(clone);
  doc.body.append(host);
  const previousTitle = doc.title;
  let titled = false;
  let ended = false;
  let fallback: ReturnType<typeof setTimeout> | null = null;
  let finish: (error?: Error) => void = () => undefined;
  const afterPrint = () => finish();
  const done = new Promise<void>((resolve, reject) => {
    finish = (error?: Error) => {
      if (ended) return;
      ended = true;
      if (fallback !== null) clearTimeout(fallback);
      controller.abort();
      win.removeEventListener("afterprint", afterPrint);
      host.remove();
      doc.body.classList.remove("is-printing", "print-furniture");
      if (titled) doc.title = previousTitle;
      if (error) reject(error);
      else resolve();
    };
  });
  waitForImages([...clone.querySelectorAll("img")], {
    signal: controller.signal,
    timeoutMs,
    loading: messages.loading,
    broken: messages.broken,
  }).then(
    () => {
      if (ended) return;
      if (!isCurrent()) {
        finish(new PrintCancelled());
        return;
      }
      // Browsers suggest the document title as the PDF file name.
      doc.title = title.replace(/[\u0000-\u001f\u007f]/g, "").slice(0, 160) || "Markitai";
      titled = true;
      doc.body.classList.add("is-printing");
      if (furniture) doc.body.classList.add("print-furniture");
      win.addEventListener("afterprint", afterPrint);
      // Some browsers never fire afterprint when their dialog is dismissed.
      fallback = setTimeout(() => finish(), 120_000);
      try {
        win.print();
      } catch (error) {
        finish(error as Error);
      }
    },
    (error: Error) => finish(error),
  );
  return { done, cancel: () => finish(new PrintCancelled()) };
}
