import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fetchItemResult } from "../api/client";
import type { ItemResult } from "../api/types";
import type { SessionItem } from "../hooks/useJobs";
import { dicts } from "../i18n";
import { MarkdownPreview } from "./MarkdownPreview";

vi.mock("../api/client", () => ({
  encodeRelPath: (path: string) => path.split("/").map(encodeURIComponent).join("/"),
  fetchItemResult: vi.fn(),
  fetchJobFileText: vi.fn(),
  jobFileUrl: () => "/doc.md",
}));

const item: SessionItem = {
  key: "job-1/item-1",
  jobId: "job-1",
  itemId: "item-1",
  name: "doc.md",
  kind: "file",
  status: "done",
  error: null,
  output: "doc.md",
  durationMs: 120,
  finishedAt: "2026-07-13T10:00:01Z",
  costUsd: null,
  llmEnhanced: false,
  operation: "convert",
  skipped: false,
  skipReason: null,
  sizeBytes: 8,
  startedAt: null,
};

const baseResult: ItemResult = {
  name: "doc.md",
  variant: "base",
  markdown: "# Doc",
  artifacts: [{ relpath: "doc.md", size: 10 }],
};

const HEADER_KEY = "markitai.pdf.custom-header-footer";

/** This jsdom environment ships no window.localStorage (the app treats it as
 * optional); the persistence test installs an in-memory stand-in. */
function installLocalStorage(): Storage {
  const store = new Map<string, string>();
  const stub = {
    getItem: (key: string) => store.get(key) ?? null,
    setItem: (key: string, value: string) => void store.set(key, String(value)),
    removeItem: (key: string) => void store.delete(key),
    clear: () => store.clear(),
    key: (index: number) => Array.from(store.keys())[index] ?? null,
    get length() {
      return store.size;
    },
  } as Storage;
  Object.defineProperty(window, "localStorage", { configurable: true, value: stub });
  return stub;
}

async function openSettings(): Promise<HTMLElement> {
  const trigger = screen.getByRole("button", { name: dicts.en.pdfSettings });
  await waitFor(() => expect(trigger).toBeEnabled());
  fireEvent.click(trigger);
  return trigger;
}

describe("MarkdownPreview PDF settings", () => {
  beforeEach(() => {
    vi.mocked(fetchItemResult).mockReset();
    vi.mocked(fetchItemResult).mockResolvedValue(baseResult);
    window.localStorage?.removeItem(HEADER_KEY);
    document.body.classList.remove("printing-preview", "printing-custom-header-footer");
    document.documentElement.removeAttribute("data-theme");
  });

  afterEach(() => {
    // drop any installed localStorage stand-in
    delete (window as { localStorage?: unknown }).localStorage;
    vi.unstubAllGlobals();
  });

  it("defaults to custom header/footer on", async () => {
    render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );

    const trigger = screen.getByRole("button", { name: dicts.en.pdfSettings });
    expect(trigger).toHaveAttribute("aria-haspopup", "dialog");
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(trigger).toBeEnabled());
    // absent key: the furniture defaults into the DOM
    expect(document.querySelector(".pdf-print-header")).toBeInTheDocument();
    expect(document.querySelector(".pdf-print-footer")).toBeInTheDocument();

    fireEvent.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByRole("dialog", { name: dicts.en.pdfSettings })).toBeVisible();
    expect(
      screen.getByRole("switch", { name: dicts.en.pdfCustomHeaderFooter }),
    ).toHaveAttribute("aria-checked", "true");
    expect(screen.getByText(dicts.en.pdfPrintDialogHint)).toBeInTheDocument();
  });

  it("persists the switch and honors an explicitly stored opt-out", async () => {
    const storage = installLocalStorage();
    const first = render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );
    await openSettings();
    fireEvent.click(
      screen.getByRole("switch", { name: dicts.en.pdfCustomHeaderFooter }),
    );
    expect(storage.getItem(HEADER_KEY)).toBe("false");
    first.unmount();

    render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );
    await openSettings();
    expect(
      screen.getByRole("switch", { name: dicts.en.pdfCustomHeaderFooter }),
    ).toHaveAttribute("aria-checked", "false");
    expect(document.querySelector(".pdf-print-header")).not.toBeInTheDocument();
  });

  it("always exports the light layout, whatever the page theme", async () => {
    const print = vi.fn();
    vi.stubGlobal("print", print);
    document.documentElement.dataset.theme = "dark";
    render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );
    const exportBtn = screen.getByRole("button", { name: dicts.en.exportPdf });
    await waitFor(() => expect(exportBtn).toBeEnabled());

    fireEvent.click(exportBtn);
    expect(print).toHaveBeenCalledOnce();
    expect(document.body).toHaveClass("printing-preview");
    expect(document.body).not.toHaveClass("printing-dark-theme");
    window.dispatchEvent(new Event("afterprint"));
    expect(document.body).not.toHaveClass("printing-preview");
  });

  it("renders the card inside the component subtree (not portalled past aria-modal)", async () => {
    const view = render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );
    await openSettings();
    const card = screen.getByRole("dialog", { name: dicts.en.pdfSettings });
    // PreviewModal's dialog is aria-modal — AT treats nodes outside it as
    // inert, so the card must stay inside the rendered subtree
    expect(view.container.contains(card)).toBe(true);
  });

  it("closes via the in-card close button and returns focus to the trigger", async () => {
    render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );
    const trigger = await openSettings();
    const card = screen.getByRole("dialog", { name: dicts.en.pdfSettings });
    fireEvent.click(within(card).getByRole("button", { name: dicts.en.close }));
    expect(
      screen.queryByRole("dialog", { name: dicts.en.pdfSettings }),
    ).not.toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("moves focus into the popover on open and back to the trigger on Escape", async () => {
    render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );
    const trigger = await openSettings();
    expect(
      screen.getByRole("switch", { name: dicts.en.pdfCustomHeaderFooter }),
    ).toHaveFocus();

    fireEvent.keyDown(document, { key: "Escape" });
    expect(
      screen.queryByRole("dialog", { name: dicts.en.pdfSettings }),
    ).not.toBeInTheDocument();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await waitFor(() => expect(trigger).toHaveFocus());
  });

  it("closes on outside pointerdown without stealing focus", async () => {
    render(
      <MarkdownPreview t={dicts.en} item={item} createdAt={null} announce={vi.fn()} />,
    );
    await openSettings();

    fireEvent.pointerDown(document.body);
    expect(
      screen.queryByRole("dialog", { name: dicts.en.pdfSettings }),
    ).not.toBeInTheDocument();
  });
});
