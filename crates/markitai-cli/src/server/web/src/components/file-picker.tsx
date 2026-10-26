// The Upload and Folder tools. Each is a visible label around a visually
// hidden input, so the keyboard reaches the native picker. Chosen files and
// folders wait in the composer until Convert.
import type { ComponentChildren } from "preact";
import type { Dict } from "../i18n/index.ts";
import { selectFolderFiles } from "../lib/files.ts";
import { Icon } from "./icons.tsx";

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

/** Two tools: Upload picks files, Folder picks a folder (whose hidden and system
 * files are left out; the rest arrive with their relative paths). */
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
  return (
    <>
      <Picker className="tool upload-main" label={t.uploadFiles} disabled={disabled} onChosen={onFiles}>
        <Icon name="UploadSimple" size={14} />
        <span class="tool-text">{t.browse}</span>
      </Picker>
      <Picker
        className="tool upload-folder"
        label={t.folderAria}
        disabled={disabled}
        folder
        onChosen={(chosen) => {
          const kept = selectFolderFiles(chosen);
          onFolder(kept.files, kept.hidden);
        }}
      >
        <Icon name="FolderSimple" size={14} />
        <span class="tool-text">{t.folderShort}</span>
      </Picker>
    </>
  );
}
