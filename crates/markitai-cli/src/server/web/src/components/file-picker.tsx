// Upload and Folder tools: a visible label around a visually hidden input, so
// the keyboard reaches the native picker. Chosen files start a job at once.
import type { ComponentChildren } from "preact";
import type { Dict } from "../i18n/index.ts";
import { selectFolderFiles } from "../lib/files.ts";
import { Icon } from "./icons.tsx";

function Picker({
  label,
  disabled,
  folder = false,
  children,
  onChosen,
}: {
  label: string;
  disabled: boolean;
  folder?: boolean;
  children: ComponentChildren;
  onChosen: (files: File[]) => void;
}) {
  return (
    <label
      class={`tool tool-picker${folder ? " tool-folder" : ""}${disabled ? " is-disabled" : ""}`}
      aria-label={label}
      title={label}
    >
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

export function FilePicker({ t, disabled, onFiles }: { t: Dict; disabled: boolean; onFiles: (files: File[]) => void }) {
  return (
    <Picker label={t.browse} disabled={disabled} onChosen={onFiles}>
      <Icon name="UploadSimple" size={14} />
      <span class="tool-text">{t.browse}</span>
    </Picker>
  );
}

/** A folder's files arrive with relative paths; hidden and system files inside it are left out. */
export function FolderPicker({
  t,
  disabled,
  onFolder,
}: {
  t: Dict;
  disabled: boolean;
  onFolder: (files: File[], hidden: number) => void;
}) {
  return (
    <Picker
      label={t.folderAria}
      disabled={disabled}
      folder
      onChosen={(files) => {
        const chosen = selectFolderFiles(files);
        onFolder(chosen.files, chosen.hidden);
      }}
    >
      <Icon name="FolderSimple" size={14} />
      <span class="tool-text">{t.folder}</span>
    </Picker>
  );
}
