import { useEffect, useRef, useState } from "react";

function dragHasFiles(e: DragEvent): boolean {
  return e.dataTransfer !== null && Array.from(e.dataTransfer.types).includes("Files");
}

/** Full-page drop target: window-level drag listeners + a hairline veil while
 * dragging. Dropping anywhere creates a new job with the current options. */
export function DropOverlay({
  label,
  onFiles,
}: {
  label: string;
  onFiles: (files: File[]) => void;
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
      const files = e.dataTransfer === null ? [] : Array.from(e.dataTransfer.files);
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
}: {
  label: string;
  onFiles: (files: File[]) => void;
}) {
  return (
    <label className="browse">
      {label}
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
