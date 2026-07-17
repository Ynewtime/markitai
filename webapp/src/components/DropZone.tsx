import { useEffect, useRef, useState, type ReactNode } from "react";
import { collectFromEntries } from "../lib/fileTree";

function dragHasFiles(e: DragEvent): boolean {
  return e.dataTransfer !== null && Array.from(e.dataTransfer.types).includes("Files");
}

/** Full-page drop target: window-level drag listeners + a hairline veil while
 * dragging. Dropping anywhere creates a new job with the current options.
 * Plain file drops keep the direct path; drops that include a directory go
 * through the webkitGetAsEntry() traversal (`fromFolder` tells App to apply
 * the 50-item truncation / empty-folder notices). */
export function DropOverlay({
  label,
  onFiles,
}: {
  label: string;
  onFiles: (files: File[], fromFolder?: boolean) => void;
}) {
  const [active, setActive] = useState(false);
  const depthRef = useRef(0);
  const onFilesRef = useRef(onFiles);
  useEffect(() => {
    onFilesRef.current = onFiles;
  }, [onFiles]);

  useEffect(() => {
    const onEnter = (e: DragEvent) => {
      if (!dragHasFiles(e)) return;
      e.preventDefault();
      depthRef.current += 1;
      setActive(true);
    };
    const onOver = (e: DragEvent) => {
      if (!dragHasFiles(e)) return;
      e.preventDefault();
    };
    const onLeave = (e: DragEvent) => {
      if (!dragHasFiles(e)) return;
      depthRef.current = Math.max(0, depthRef.current - 1);
      if (depthRef.current === 0) setActive(false);
    };
    const onDrop = (e: DragEvent) => {
      if (!dragHasFiles(e)) return;
      e.preventDefault();
      depthRef.current = 0;
      setActive(false);
      const dt = e.dataTransfer;
      if (dt === null) return;
      // webkitGetAsEntry() must be read synchronously — the item list is
      // neutered once the drop handler returns.
      const entries = Array.from(dt.items ?? []).map((it) =>
        typeof it.webkitGetAsEntry === "function" ? it.webkitGetAsEntry() : null,
      );
      if (entries.some((en) => en !== null && en.isDirectory)) {
        void collectFromEntries(entries).then((files) =>
          onFilesRef.current(files, true),
        );
        return;
      }
      const files = Array.from(dt.files);
      if (files.length > 0) onFilesRef.current(files);
    };
    window.addEventListener("dragenter", onEnter);
    window.addEventListener("dragover", onOver);
    window.addEventListener("dragleave", onLeave);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onEnter);
      window.removeEventListener("dragover", onOver);
      window.removeEventListener("dragleave", onLeave);
      window.removeEventListener("drop", onDrop);
    };
  }, []);

  if (!active) return null;
  return (
    <div className="dropveil" aria-hidden="true">
      <span className="veillbl">{label}</span>
    </div>
  );
}

/** Keyboard-usable file picker: visible label + visually-hidden input. */
export function FilePicker({
  label,
  onFiles,
  icon,
  className = "browse",
}: {
  label: string;
  onFiles: (files: File[]) => void;
  icon?: ReactNode;
  className?: string;
}) {
  return (
    // aria-label duplicates the visible text so the icon-only phone variant
    // (the span is display:none there) keeps its accessible name
    <label className={className} aria-label={label}>
      {icon}
      <span>{label}</span>
      <input
        type="file"
        multiple
        className="sr-only"
        onChange={(e) => {
          const files = e.currentTarget.files === null ? [] : Array.from(e.currentTarget.files);
          e.currentTarget.value = "";
          if (files.length > 0) onFiles(files);
        }}
      />
    </label>
  );
}
