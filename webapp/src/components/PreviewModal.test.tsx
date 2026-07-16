import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { SessionItem } from "../hooks/useJobs";
import { dicts } from "../i18n";
import { PreviewModal } from "./PreviewModal";

vi.mock("../api/client", () => ({
  encodeRelPath: (path: string) => path.split("/").map(encodeURIComponent).join("/"),
  fetchItemResult: vi.fn().mockResolvedValue({
    name: "result.md",
    variant: "base",
    markdown:
      "# Result\n\n![截图](.markitai/assets/截屏 下午.png)\n\n[![build](https://github.com/org/repo/actions/workflows/test.yml/badge.svg)](https://github.com/org/repo/actions)\n\n```javascript\nconst answer = 42;\n```",
    artifacts: [{ relpath: "result.md", size: 47 }],
  }),
  fetchJobFileText: vi.fn().mockResolvedValue("# Result"),
  jobFileUrl: () => "/result.md",
}));

const item: SessionItem = {
  key: "job-1/item-1",
  jobId: "job-1",
  itemId: "item-1",
  name: "result.md",
  kind: "file",
  status: "done",
  error: null,
  output: "result.md",
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

describe("PreviewModal", () => {
  beforeEach(() => {
    window.localStorage?.removeItem("markitai.pdf.custom-header-footer");
    document.body.classList.remove(
      "printing-preview",
      "printing-dark-theme",
      "printing-custom-header-footer",
    );
    document.documentElement.classList.remove("printing-dark-theme");
    document.documentElement.removeAttribute("data-theme");
  });

  it("shows the rendered tab and closes with Escape", () => {
    const onClose = vi.fn();
    render(
      <PreviewModal
        t={dicts.en}
        item={item}
        createdAt={null}
        onClose={onClose}
        announce={() => undefined}
      />,
    );

    expect(screen.getByRole("dialog", { name: item.name })).toBeVisible();
    expect(screen.getByRole("tab", { name: dicts.en.rendered })).toHaveAttribute(
      "aria-selected",
      "true",
    );

    fireEvent.keyDown(document, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it("keeps Markdown download and opens the PDF print export", async () => {
    const print = vi.fn();
    vi.stubGlobal("print", print);
    render(
      <PreviewModal
        t={dicts.en}
        item={item}
        createdAt={null}
        onClose={() => undefined}
        announce={() => undefined}
      />,
    );

    expect(await screen.findByRole("link", { name: /download \.md/i })).toHaveAttribute(
      "href",
      "/result.md",
    );
    expect(document.querySelector(".pdf-print-header")).not.toBeInTheDocument();
    expect(document.querySelector(".pdf-print-footer")).not.toBeInTheDocument();
    const customHeader = screen.getByRole("switch", {
      name: dicts.en.pdfCustomHeaderFooter,
    });
    expect(customHeader).toHaveAttribute("aria-checked", "false");
    fireEvent.click(screen.getByRole("button", { name: dicts.en.exportPdf }));
    expect(print).toHaveBeenCalledOnce();
    expect(document.body).toHaveClass("printing-preview");
    expect(document.body).not.toHaveClass("printing-custom-header-footer");
    expect(document.title).toBe("result");
    window.dispatchEvent(new Event("afterprint"));

    fireEvent.click(customHeader);
    expect(document.querySelector(".pdf-print-header")).toHaveTextContent(
      "markitairesult",
    );
    expect(document.querySelector(".pdf-print-footer")).toHaveTextContent(
      `${dicts.en.pdfPreparedBy}${dicts.en.pdfSource}: result.md`,
    );
    document.documentElement.dataset.theme = "dark";
    fireEvent.click(screen.getByRole("button", { name: dicts.en.exportPdf }));
    expect(print).toHaveBeenCalledTimes(2);
    expect(document.body).toHaveClass("printing-custom-header-footer");
    expect(document.body).toHaveClass("printing-dark-theme");
    expect(document.documentElement).toHaveClass("printing-dark-theme");
    window.dispatchEvent(new Event("afterprint"));
    expect(document.documentElement).not.toHaveClass("printing-dark-theme");
    vi.unstubAllGlobals();
  });

  it("renders URI-encoded Unicode asset names as images", async () => {
    render(
      <PreviewModal
        t={dicts.en}
        item={item}
        createdAt={null}
        onClose={() => undefined}
        announce={() => undefined}
      />,
    );

    const image = await screen.findByRole("img", { name: "截图" });
    const badge = await screen.findByRole("img", { name: "build" });
    expect(badge).toHaveClass("md-badge");
    expect(image).not.toHaveClass("md-badge");
    expect(image.getAttribute("src")).toContain(
      "/api/jobs/job-1/files/.markitai/assets/",
    );
    expect(image.getAttribute("src")).toContain(
      "%E6%88%AA%E5%B1%8F%20%E4%B8%8B%E5%8D%88.png",
    );
  });

  it("links URL preview titles to their source in a new tab", async () => {
    const url = "https://example.com/path?q=1";
    render(
      <PreviewModal
        t={dicts.en}
        item={{ ...item, name: url, kind: "url" }}
        createdAt={null}
        onClose={() => undefined}
        announce={() => undefined}
      />,
    );

    expect(await screen.findByRole("link", { name: url })).toHaveAttribute(
      "href",
      url,
    );
    expect(screen.getByRole("link", { name: url })).toHaveAttribute(
      "target",
      "_blank",
    );
  });

  it("highlights fenced code using its declared language", async () => {
    render(
      <PreviewModal
        t={dicts.en}
        item={item}
        createdAt={null}
        onClose={() => undefined}
        announce={() => undefined}
      />,
    );

    await waitFor(() => {
      expect(
        document.querySelector("code.hljs.language-javascript"),
      ).toBeInTheDocument();
    });
    const code = document.querySelector("code.hljs.language-javascript");
    expect(code?.querySelector(".hljs-keyword")).toHaveTextContent("const");
    expect(code?.querySelector(".hljs-number")).toHaveTextContent("42");
  });
});
