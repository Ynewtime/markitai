import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { HelpTooltip } from "./HelpTooltip";

const box = (left: number, top: number, width: number, height: number) =>
  ({ left, top, width, height, right: left + width, bottom: top + height, x: left, y: top, toJSON: () => ({}) }) as DOMRect;
let anchorRect: DOMRect;
let bubbleHeight: number;
let resized: () => void;
const observe = vi.fn();
const disconnect = vi.fn();

beforeEach(() => {
  vi.clearAllMocks();
  anchorRect = box(400, 300, 40, 24);
  bubbleHeight = 64;
  vi.stubGlobal("innerWidth", 1000);
  vi.stubGlobal("innerHeight", 800);
  vi.stubGlobal("ResizeObserver", class {
    constructor(callback: () => void) { resized = callback; }
    observe = observe;
    disconnect = disconnect;
  });
  vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
    if (this.classList.contains("help-anchor")) return anchorRect;
    if (this.classList.contains("option-tooltip")) {
      return box(0, 0, 240, Math.min(bubbleHeight, parseFloat(this.style.maxHeight) || bubbleHeight));
    }
    return box(0, 0, 0, 0);
  });
});
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals(); vi.clearAllMocks(); });

function show() {
  const view = render(<div style={{ overflow: "hidden", transform: "translateX(30px)" }}>
    <HelpTooltip text="Brief option help"><button>Help</button></HelpTooltip>
  </div>);
  fireEvent.mouseEnter(screen.getByRole("button"));
  return { ...view, bubble: screen.getByRole("tooltip") };
}

function expectPlacement(bubble: HTMLElement, left: number, top: number) {
  expect(bubble).toHaveStyle({ left: `${left}px`, top: `${top}px`, visibility: "visible" });
  const height = bubble.getBoundingClientRect().height;
  expect(top + height <= anchorRect.top - 8 || top >= anchorRect.bottom + 8).toBe(true);
}

describe("HelpTooltip positioning", () => {
  it("portals directly to body and centers the measured bubble above its anchor", () => {
    const { bubble, container } = show();
    expect(bubble.parentElement).toBe(document.body);
    expect(container).not.toContainElement(bubble);
    expectPlacement(bubble, 300, 228);
  });

  it.each([
    [400, 12, 300, 44],
    [400, 750, 300, 678],
    [0, 300, 12, 228],
    [970, 300, 748, 228],
  ])("clamps viewport edges and flips below only when above cannot fit at (%i, %i)", (x, y, left, top) => {
    anchorRect = box(x, y, 40, 24);
    const { bubble } = show();
    expectPlacement(bubble, left, top);
  });

  it("repositions on captured nested scroll and viewport resize rather than dismissing", () => {
    const { bubble, container } = show();
    anchorRect = box(400, 700, 40, 24);
    fireEvent.scroll(container.firstElementChild!);
    expectPlacement(bubble, 300, 628);
    vi.stubGlobal("innerWidth", 600);
    vi.stubGlobal("innerHeight", 500);
    anchorRect = box(480, 450, 40, 24);
    fireEvent.resize(window);
    expectPlacement(bubble, 348, 378);
  });

  it("remeasures changing content and anchor size, and releases a cramped height cap", () => {
    vi.stubGlobal("innerHeight", 200);
    anchorRect = box(400, 90, 40, 24);
    bubbleHeight = 180;
    const { bubble, rerender } = show();
    expect(bubble).toHaveStyle({ maxHeight: "70px" });
    expectPlacement(bubble, 300, 12);
    vi.stubGlobal("innerHeight", 800);
    anchorRect = box(400, 300, 100, 24);
    act(() => resized());
    expectPlacement(bubble, 330, 112);
    expect(bubble).toHaveStyle({ maxHeight: "200px" });
    bubbleHeight = 100;
    anchorRect = box(400, 700, 100, 24);
    rerender(<div style={{ overflow: "hidden", transform: "translateX(30px)" }}>
      <HelpTooltip text="Changed help"><button>Help</button></HelpTooltip>
    </div>);
    expect(screen.getByRole("tooltip")).toBe(bubble);
    expectPlacement(bubble, 330, 592);
    expect(observe).toHaveBeenCalledWith(screen.getByRole("button").parentElement);
    expect(observe).toHaveBeenCalledWith(screen.getByRole("tooltip"));
  });

  it("keeps modal descriptions local and preserves existing description IDs", () => {
    render(<div role="dialog" aria-modal="true">
      <span id="existing">Existing description</span>
      <HelpTooltip text="Modal help"><button aria-describedby="existing">Help</button></HelpTooltip>
    </div>);
    const button = screen.getByRole("button");
    fireEvent.focus(button);
    expect(button).toHaveAccessibleDescription("Existing description Modal help");
    const ids = button.getAttribute("aria-describedby")!.split(" ");
    expect(screen.getByRole("dialog")).toContainElement(document.getElementById(ids[1]!));
    expect(screen.getByRole("tooltip").parentElement).toBe(document.body);
  });

  it("cleans up observers and listeners on dismissal and unmount", () => {
    const remove = vi.spyOn(window, "removeEventListener");
    const { unmount } = show();
    fireEvent.keyDown(window, { key: "Escape" });
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    expect(disconnect).toHaveBeenCalled();
    expect(remove).toHaveBeenCalledWith("scroll", expect.any(Function), true);
    expect(remove).toHaveBeenCalledWith("resize", expect.any(Function));
    fireEvent.mouseEnter(screen.getByRole("button"));
    unmount();
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    expect(disconnect).toHaveBeenCalledTimes(2);
  });

  it("auto-hides after a delay even while the pointer stays on the trigger", () => {
    vi.useFakeTimers();
    try {
      show();
      expect(screen.getByRole("tooltip")).toBeInTheDocument();
      act(() => { vi.advanceTimersByTime(4000); });
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
      // Re-hovering shows it again and restarts the clock.
      fireEvent.mouseEnter(screen.getByRole("button"));
      expect(screen.getByRole("tooltip")).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });

  it("hides as soon as the pointer leaves the trigger", () => {
    vi.useFakeTimers();
    try {
      show();
      fireEvent.mouseLeave(screen.getByRole("button"));
      act(() => { vi.advanceTimersByTime(150); });
      expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});
