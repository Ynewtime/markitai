import { render, screen, within } from "@testing-library/react";
import { createRef } from "react";
import { describe, expect, it, vi } from "vitest";
import { dicts } from "../i18n";
import { AppFooter, AppHeader } from "./AppHeader";

function renderHeader() {
  return render(
    <AppHeader
      t={dicts.en}
      version="1.0.0"
      locale="en"
      onLocale={vi.fn()}
      onHome={vi.fn()}
      onHistory={vi.fn()}
      historyActive={false}
      settingsOpen={false}
      onToggleSettings={vi.fn()}
      gearRef={createRef<HTMLButtonElement>()}
    />,
  );
}

describe("AppHeader control groups", () => {
  // The phone header grid dissolves .hdr-ctl (display: contents) and places
  // these two groups as units — losing a wrapper silently breaks that layout.
  it("groups the view icons and the locale/theme toggles for the mobile grid", () => {
    renderHeader();

    const icons = screen
      .getByRole("button", { name: dicts.en.historyAria })
      .closest(".hdr-icons");
    expect(icons).not.toBeNull();
    expect(
      screen.getByRole("button", { name: dicts.en.settingsAria }).closest(".hdr-icons"),
    ).toBe(icons);

    const toggles = document.querySelector(".hdr-ctl .hdr-toggles");
    expect(toggles?.querySelector(".langtoggle")).not.toBeNull();
    expect(toggles?.querySelector(".themetoggle")).not.toBeNull();
  });
});

describe("AppFooter", () => {
  // The phone layout hides .hdr-links and shows this footer instead — the
  // link set must stay identical so nothing is lost in the swap.
  it("mirrors the header's external links", () => {
    render(<AppFooter t={dicts.en} />);

    const footer = document.querySelector<HTMLElement>("footer.app-footer");
    expect(footer).not.toBeNull();
    const links = within(footer!).getAllByRole("link");
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "https://markitai.dev",
      "https://github.com/Ynewtime/markitai",
    ]);
    for (const link of links) expect(link).toHaveAttribute("target", "_blank");
  });
});
