// The Upload tool: a split button. Its main part picks files; the caret opens a
// small menu with Files and Folder. Each choice is a visible label around a
// visually hidden input, so the keyboard reaches the native picker. Chosen
// files start a job at once.
import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import type { Dict } from "../i18n/index.ts";
import { selectFolderFiles } from "../lib/files.ts";
import { Icon } from "./icons.tsx";

let pickers = 0;

function Picker({
  label,
  disabled,
  folder = false,
  className,
  children,
  onChosen,
}: {
  label: string;
  disabled: boolean;
  folder?: boolean;
  className: string;
  children: ComponentChildren;
  onChosen: (files: File[]) => void;
}) {
  return (
    <label class={`${className} tool-picker${disabled ? " is-disabled" : ""}`} aria-label={label} title={label}>
      {children}
      <input
        type="file"
        multiple
        class="sr-only"
        disabled={disabled}
        {...(folder ? { webkitdirectory: true } : {})}
        onChange={(event) => {
          const input = event.currentTarget;
          const files = input.files ? [...input.files] : [];
          input.value = "";
          // An empty folder still answers, so the page can say it held nothing.
          if (files.length || folder) onChosen(files);
        }}
      />
    </label>
  );
}

/** A folder's files arrive with relative paths; hidden and system files inside it are left out. */
export function UploadPicker({
  t,
  disabled,
  onFiles,
  onFolder,
}: {
  t: Dict;
  disabled: boolean;
  onFiles: (files: File[]) => void;
  onFolder: (files: File[], hidden: number) => void;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const [id] = useState(() => `upload-${++pickers}`);
  useEffect(() => {
    if (!open) return;
    const away = (event: PointerEvent) => {
      if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setOpen(false);
      trigger.current?.focus();
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  useEffect(() => {
    if (disabled) setOpen(false);
  }, [disabled]);
  const files = (chosen: File[]) => {
    setOpen(false);
    onFiles(chosen);
  };
  const folder = (chosen: File[]) => {
    setOpen(false);
    const kept = selectFolderFiles(chosen);
    onFolder(kept.files, kept.hidden);
  };
  return (
    <div
      class={open ? "upload-split is-open" : "upload-split"}
      ref={root}
      onFocusOut={(event) => {
        const next = (event as FocusEvent).relatedTarget;
        if (next instanceof Node && !root.current?.contains(next)) setOpen(false);
      }}
    >
      <Picker className="tool upload-main" label={t.browse} disabled={disabled} onChosen={files}>
        <Icon name="UploadSimple" size={14} />
        <span class="tool-text">{t.browse}</span>
      </Picker>
      <button
        ref={trigger}
        type="button"
        class="tool upload-more"
        aria-label={t.uploadMore}
        title={t.uploadMore}
        aria-haspopup="true"
        aria-expanded={open}
        aria-controls={id}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="CaretDown" size={12} />
      </button>
      <div class="upload-menu" id={id} role="group" aria-label={t.uploadMore} hidden={!open}>
        <Picker className="upload-item" label={t.uploadFiles} disabled={disabled} onChosen={files}>
          <Icon name="FileText" size={14} />
          <span>{t.uploadFiles}</span>
        </Picker>
        <Picker className="upload-item" label={t.folderAria} disabled={disabled} folder onChosen={folder}>
          <Icon name="FolderSimple" size={14} />
          <span>{t.folderAria}</span>
        </Picker>
      </div>
    </div>
  );
}
