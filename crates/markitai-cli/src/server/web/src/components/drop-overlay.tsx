// Drop anywhere: a hairline veil over the page while files are dragged in.
// A drop that contains a folder is walked through the entry API, which only
// works during the drop event itself. While a dialog is open, drops are ignored
// (and never opened by the browser in place of the page).
//
// A desktop app whose web view keeps a drop's files from the page (WebKitGTK
// 2.52 and later) hands them over itself: it shows the veil with markitai:drag
// ("enter", "leave") and chooses the files in the drop input, a folder input
// that is disabled while drops are ignored. Browsers never use either.
import { useEffect, useRef, useState } from "preact/hooks";
import { handedDrop, walkEntries, type Walked } from "../lib/files.ts";

const carriesFiles = (event: DragEvent) => [...(event.dataTransfer?.types ?? [])].includes("Files");

export function DropOverlay({
  label,
  suspended,
  limit,
  onFiles,
  onFolder,
}: {
  label: string;
  suspended: boolean;
  limit: number;
  onFiles: (files: File[]) => void;
  onFolder: (walked: Walked) => void;
}) {
  const [active, setActive] = useState(false);
  const depth = useRef(0);
  const latest = useRef({ suspended, limit, onFiles, onFolder });
  latest.current = { suspended, limit, onFiles, onFolder };

  useEffect(() => {
    if (suspended) {
      depth.current = 0;
      setActive(false);
    }
  }, [suspended]);

  useEffect(() => {
    const enter = (event: DragEvent) => {
      if (!carriesFiles(event) || latest.current.suspended) return;
      event.preventDefault();
      depth.current++;
      setActive(true);
    };
    const over = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = latest.current.suspended ? "none" : "copy";
    };
    const leave = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      depth.current = Math.max(0, depth.current - 1);
      if (!depth.current) setActive(false);
    };
    const drop = (event: DragEvent) => {
      if (!carriesFiles(event)) return;
      event.preventDefault();
      depth.current = 0;
      setActive(false);
      if (latest.current.suspended || !event.dataTransfer) return;
      const entries = [...(event.dataTransfer.items ?? [])]
        .map((item) => (item.kind === "file" ? (item.webkitGetAsEntry?.() ?? null) : null))
        .filter((entry): entry is FileSystemEntry => entry !== null);
      if (entries.some((entry) => entry.isDirectory)) {
        void walkEntries(entries as never, { limit: latest.current.limit }).then((walked) => latest.current.onFolder(walked));
        return;
      }
      const files = [...event.dataTransfer.files];
      if (files.length) latest.current.onFiles(files);
    };
    const handed = (event: Event) => {
      depth.current = 0;
      setActive((event as CustomEvent<string>).detail === "enter" && !latest.current.suspended);
    };
    window.addEventListener("dragenter", enter);
    window.addEventListener("dragover", over);
    window.addEventListener("dragleave", leave);
    window.addEventListener("drop", drop);
    window.addEventListener("markitai:drag", handed);
    return () => {
      window.removeEventListener("dragenter", enter);
      window.removeEventListener("dragover", over);
      window.removeEventListener("dragleave", leave);
      window.removeEventListener("drop", drop);
      window.removeEventListener("markitai:drag", handed);
    };
  }, []);

  return (
    <>
      <input
        type="file"
        multiple
        hidden
        data-drop-input=""
        disabled={suspended}
        {...{ webkitdirectory: true }}
        onChange={(event) => {
          const input = event.currentTarget;
          const chosen = input.files ? [...input.files] : [];
          input.value = "";
          if (latest.current.suspended) return;
          const walked = handedDrop(chosen);
          if (walked.folder) latest.current.onFolder(walked);
          else latest.current.onFiles(walked.files);
        }}
      />
      {active && (
        <div class="drop-veil" aria-hidden="true">
          <span class="drop-label">{label}</span>
        </div>
      )}
    </>
  );
}
