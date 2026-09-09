import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createRef, useState } from "react";
import { describe, expect, it, vi } from "vitest";
import { dicts, type Locale } from "../i18n";
import { AppFooter, AppHeader } from "./AppHeader";
import { ClockCounterClockwise } from "@phosphor-icons/react";
import { renderToStaticMarkup } from "react-dom/server";

function renderHeader(locale: Locale = "en") {
  function Header() {
    const [language, setLanguage] = useState(locale);
    return <AppHeader t={dicts[language]} version="1.0.0" locale={language}
      onLocale={setLanguage} onHome={vi.fn()} onHistory={vi.fn()} historyActive={false}
      settingsOpen={false} onToggleSettings={vi.fn()} gearRef={createRef<HTMLButtonElement>()} />;
  }
  return render(<Header />);
}

describe("AppHeader", () => {
  it("uses actual Phosphor bold geometry only for the active history icon", () => {
    const props = { t: dicts.en, version: null, locale: "en" as const, onLocale: vi.fn(),
      onHome: vi.fn(), onHistory: vi.fn(), settingsOpen: false, onToggleSettings: vi.fn(),
      gearRef: createRef<HTMLButtonElement>() };
    const { rerender } = render(<AppHeader {...props} historyActive={false} />);
    const button = screen.getByRole("button", { name: dicts.en.historyAria });
    const geometry = (weight: "regular" | "bold") => {
      const element = document.createElement("div");
      element.innerHTML = renderToStaticMarkup(<ClockCounterClockwise weight={weight} />);
      return element.querySelector("svg")!.innerHTML;
    };
    expect(button.querySelector("svg")!.innerHTML).toBe(geometry("regular"));
    expect(button).not.toHaveAttribute("aria-current");
    rerender(<AppHeader {...props} historyActive />);
    expect(button).toHaveAttribute("aria-current", "page");
    expect(button.querySelector("svg")!.innerHTML).toBe(geometry("bold"));
    expect(geometry("bold")).not.toBe(geometry("regular"));
  });
  it.each(["en", "zh"] as const)("has one appearance trigger beside history and settings in %s", (locale) => {
    renderHeader(locale);
    const t = dicts[locale];
    const icons = screen.getByRole("button", { name: t.historyAria }).closest(".hdr-icons")!;
    expect(within(icons as HTMLElement).getAllByRole("button")).toHaveLength(3);
    expect(icons).toContainElement(screen.getByRole("button", { name: t.appearanceTitle }));
    expect(icons).toContainElement(screen.getByRole("button", { name: t.settingsAria }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
    expect(document.querySelector(".hdr-links")?.previousElementSibling).toHaveClass("brand");
  });

  it("opens with the keyboard, changes language and theme, and returns focus on Escape", async () => {
    const user = userEvent.setup();
    renderHeader();
    const trigger = screen.getByRole("button", { name: dicts.en.appearanceTitle });
    trigger.focus();
    await user.keyboard("{Enter}");
    const dialog = screen.getByRole("dialog", { name: dicts.en.appearanceTitle });
    expect(trigger).toHaveAttribute("aria-controls", dialog.id);
    expect(trigger).toHaveAttribute("aria-expanded", "true");
    expect(within(dialog).getByRole("button", { name: "EN" })).toHaveFocus();
    await user.click(within(dialog).getByRole("button", { name: "中" }));
    expect(dialog).toHaveAccessibleName(dicts.zh.appearanceTitle);
    expect(within(dialog).getByRole("button", { name: "中" })).toHaveAttribute("aria-pressed", "true");
    await user.click(within(dialog).getByRole("radio", { name: dicts.zh.themeLight }));
    await user.keyboard("{ArrowRight}");
    expect(within(dialog).getByRole("radio", { name: dicts.zh.themeDark })).toBeChecked();
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(trigger).toHaveFocus();
    await user.click(trigger);
    expect(screen.getByRole("radio", { name: dicts.zh.themeDark })).toBeChecked();
  });

  it("closes on touch outside, trigger toggle and focus leaving without trapping Tab", async () => {
    const user = userEvent.setup();
    renderHeader();
    const trigger = screen.getByRole("button", { name: dicts.en.appearanceTitle });
    await user.click(trigger);
    fireEvent.pointerDown(document.body, { pointerType: "touch" });
    // A tap elsewhere closes the menu but must not yank focus back to the trigger.
    expect(trigger).not.toHaveFocus();
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await user.click(trigger);
    await user.click(trigger);
    expect(trigger).toHaveAttribute("aria-expanded", "false");
    await user.click(trigger);
    screen.getByRole("radio", { checked: true }).focus();
    await user.tab();
    expect(screen.getByRole("button", { name: dicts.en.historyAria })).toHaveFocus();
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("applies the saved theme before the popover is opened", () => {
    vi.stubGlobal("localStorage", { getItem: () => "dark", setItem: vi.fn(), removeItem: vi.fn() });
    try {
      renderHeader();
      expect(document.documentElement).toHaveAttribute("data-theme", "dark");
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("AppFooter", () => {
  it("mirrors the header's external links", () => {
    render(<AppFooter t={dicts.en} />);
    const links = within(screen.getByRole("contentinfo")).getAllByRole("link");
    expect(links.map((link) => link.getAttribute("href"))).toEqual([
      "https://markitai.dev", "https://github.com/Ynewtime/markitai",
    ]);
    for (const link of links) expect(link).toHaveAttribute("target", "_blank");
  });
});
