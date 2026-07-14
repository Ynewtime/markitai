import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import type { SessionItem } from "../hooks/useJobs";
import { dicts } from "../i18n";
import { PreviewModal } from "./PreviewModal";

vi.mock("../api/client", () => ({
  encodeRelPath: (path: string) => path,
  fetchItemResult: vi.fn().mockResolvedValue({
    name: "result.md",
    variant: "base",
    markdown: "# Result\n\n```javascript\nconst answer = 42;\n```",
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
  costUsd: null,
  skipped: false,
  skipReason: null,
  sizeBytes: 8,
  startedAt: null,
  archived: false,
  retried: false,
};

describe("PreviewModal", () => {
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
