import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { dicts } from "../i18n";
import { JobStats } from "./JobStats";

describe("JobStats", () => {
  it("does not show current-session counters when only archived rows exist", () => {
    render(
      <JobStats
        t={dicts.zh}
        running={false}
        stats={{
          done: 0,
          skipped: 0,
          failed: 0,
          total: 0,
          costTotal: 0,
          hasCost: false,
          doneDurationMs: 0,
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "转换任务" })).toBeVisible();
    expect(screen.queryByText(/当前会话/)).not.toBeInTheDocument();
    expect(document.querySelector(".stats strong")).not.toBeInTheDocument();
  });
});
