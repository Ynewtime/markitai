import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { DropOverlay } from "./DropZone";

/** A drop event carrying one plain file (no directory entries). */
function fileDrop(): Event {
  const event = new Event("drop", { bubbles: true, cancelable: true });
  Object.defineProperty(event, "dataTransfer", {
    value: {
      types: ["Files"],
      files: [new File(["hello"], "hello.txt", { type: "text/plain" })],
      items: [],
    },
  });
  return event;
}

describe("DropOverlay", () => {
  it("delivers a drop to the composer", () => {
    const onFiles = vi.fn();
    render(<DropOverlay label="Drop to convert" onFiles={onFiles} />);

    window.dispatchEvent(fileDrop());

    expect(onFiles).toHaveBeenCalledTimes(1);
    const [files] = onFiles.mock.calls[0] as [File[]];
    expect(files).toHaveLength(1);
  });

  it("ignores a drop while a modal owns the screen", () => {
    const onFiles = vi.fn();
    render(<DropOverlay label="Drop to convert" onFiles={onFiles} suspended />);

    window.dispatchEvent(fileDrop());

    expect(onFiles).not.toHaveBeenCalled();
    // The veil never appears either.
    expect(screen.queryByText("Drop to convert")).toBeNull();
  });

  it("shows the veil while dragging files", () => {
    render(<DropOverlay label="Drop to convert" onFiles={vi.fn()} />);

    const enter = new Event("dragenter", { bubbles: true });
    Object.defineProperty(enter, "dataTransfer", { value: { types: ["Files"] } });
    fireEvent(window, enter);

    expect(screen.getByText("Drop to convert")).toBeVisible();
  });
});
