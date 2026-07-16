import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { describe, expect, it, vi } from "vitest";
import type { SessionItem } from "../hooks/useJobs";
import { dicts } from "../i18n";
import { ItemList } from "./ItemList";

function item(itemId: string, skipped = false): SessionItem {
  return {
    key: `job-1/${itemId}`,
    jobId: "job-1",
    itemId,
    name: `${itemId}.png`,
    kind: "file",
    status: "done",
    error: null,
    output: skipped ? null : `${itemId}.md`,
    durationMs: skipped ? null : 10,
    finishedAt: "2026-07-16T03:52:00.000+08:00",
    costUsd: null,
    llmEnhanced: false,
    operation: "convert",
    skipped,
    skipReason: skipped ? "image_only" : null,
    sizeBytes: 10,
    startedAt: null,
  };
}

function Harness() {
  const items = [item("rendered"), item("skipped", true)];
  const [selected, setSelected] = useState(items[0]!.key);
  return (
    <ItemList
      t={dicts.en}
      items={items}
      showCost={false}
      now={Date.now()}
      stats={{
        done: 1,
        skipped: 1,
        failed: 0,
        total: 2,
        costTotal: 0,
        hasCost: false,
        doneDurationMs: 10,
      }}
      settled
      selectedKey={selected}
      onSelect={setSelected}
      onPreview={vi.fn()}
      focusKey={null}
      onFocusKeyHandled={vi.fn()}
      onRetry={vi.fn().mockResolvedValue(null)}
      onDelete={vi.fn().mockResolvedValue(null)}
      canDelete={() => true}
    />
  );
}

describe("ItemList roving selection", () => {
  it("lets a skipped row take focus and selection on click", async () => {
    const user = userEvent.setup();
    render(<Harness />);

    const skipped = screen.getByRole("option", { name: /skipped\.png/i });
    await user.click(skipped);

    expect(skipped).toHaveFocus();
    expect(skipped).toHaveAttribute("aria-selected", "true");
  });
});
