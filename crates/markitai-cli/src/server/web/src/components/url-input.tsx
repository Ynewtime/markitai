// The URL line: a textarea dressed as one input that grows a row per pasted URL
// (up to six), above a bar with the composer tools on the left and Convert on
// the right. Enter converts what was typed (with any chosen files), Shift+Enter
// breaks the line, and confirming an IME composition never submits. Chosen
// files wait above the line, each with its own remove button, until Convert
// sends them with the URLs: an Enter in an empty line sends nothing, so the
// key that confirms the system's file or folder picker cannot start a job.
import type { ComponentChildren } from "preact";
import { useRef, useState } from "preact/hooks";
import { NARROW, useMedia } from "../hooks/use-media.ts";
import type { Dict } from "../i18n/index.ts";
import { relativePath, type StagedSubmission } from "../lib/files.ts";
import { fmtBytes } from "../lib/format.ts";
import { MiddleName } from "./file-name.tsx";
import { Icon } from "./icons.tsx";

/** Rows drawn at most; the rest are counted. */
const STAGED_ROWS = 200;

/** The files waiting for Convert: the batch's name, count and size with Remove all,
 * then one row per file with its own remove button. */
function StagedList({ t, staged, onClear, onRemove }: { t: Dict; staged: StagedSubmission; onClear: () => void; onRemove: (file: File) => void }) {
  const total = staged.files.reduce((sum, file) => sum + file.size, 0);
  const title = staged.label || t.stagedCount(staged.files.length);
  const shown = staged.files.slice(0, STAGED_ROWS);
  return (
    <div class="staged" role="group" aria-label={t.stagedTitle}>
      {(staged.files.length > 1 || staged.label) && (
        <div class="staged-head">
          <Icon name={staged.kind === "urls" ? "Globe" : staged.kind === "folder" ? "FolderSimple" : "Files"} size={14} />
          <span class="staged-title" title={title}>
            {title}
          </span>
          <span class="staged-count">{staged.label ? `${t.stagedCount(staged.files.length)} · ${fmtBytes(total)}` : fmtBytes(total)}</span>
          <button type="button" class="staged-clear-all" onClick={onClear}>
            {t.stagedClearAll}
          </button>
        </div>
      )}
      <ul class="staged-list">
        {shown.map((file) => {
          const path = relativePath(file);
          // A folder's files read from inside the folder.
          const name = staged.kind === "folder" && staged.label && path.startsWith(`${staged.label}/`) ? path.slice(staged.label.length + 1) : path;
          return (
            <li key={`${path}\u0000${file.size}\u0000${file.lastModified}`} class="staged-file">
              <Icon name={/\.urls$/i.test(path) ? "Globe" : "FileText"} size={14} />
              <MiddleName text={name} kind="file" title={path} class="staged-name" />
              <span class="staged-size">{fmtBytes(file.size)}</span>
              <button type="button" class="staged-clear" aria-label={t.stagedClear(name)} title={t.stagedClear(name)} onClick={() => onRemove(file)}>
                <Icon name="X" size={12} />
              </button>
            </li>
          );
        })}
        {staged.files.length > shown.length && <li class="staged-more">{t.stagedMore(staged.files.length - shown.length)}</li>}
      </ul>
    </div>
  );
}

export function UrlInput({
  t,
  text,
  onText,
  onConvert,
  busy = false,
  staged = null,
  onClearStaged,
  onRemoveStaged,
  tools,
  held = false,
}: {
  t: Dict;
  text: string;
  onText: (text: string) => void;
  onConvert: (text: string) => Promise<boolean>;
  busy?: boolean;
  /** A batch that is waiting for Convert: a folder, several files or a URL list. */
  staged?: StagedSubmission | null;
  onClearStaged?: () => void;
  onRemoveStaged?: (file: File) => void;
  /** Controls placed at the start of the bottom bar. */
  tools?: ComponentChildren;
  /** v1.24: the server is updating and holds new conversions until it has restarted. */
  held?: boolean;
}) {
  const [sending, setSending] = useState(false);
  const field = useRef<HTMLTextAreaElement>(null);
  const narrow = useMedia(NARROW);
  // A staged batch is something to convert even with no URL typed.
  const empty = text.trim() === "" && staged === null;
  const rows = Math.min(6, Math.max(1, text.split("\n").length));
  const submit = async () => {
    if (empty || sending) return;
    setSending(true);
    try {
      if (await onConvert(text)) onText("");
    } finally {
      setSending(false);
      field.current?.focus({ preventScroll: true });
    }
  };
  return (
    <div class="url-row">
      {staged && <StagedList t={t} staged={staged} onClear={() => onClearStaged?.()} onRemove={(file) => onRemoveStaged?.(file)} />}
      <div class="url-field">
        <textarea
          ref={field}
          class="url-box"
          rows={rows}
          value={text}
          placeholder={narrow ? t.urlPlaceholderShort : t.urlPlaceholder}
          spellcheck={false}
          aria-label={t.urlPlaceholder}
          onInput={(event) => onText(event.currentTarget.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || event.shiftKey || event.isComposing) return;
            event.preventDefault();
            if (text.trim() !== "") void submit();
          }}
        />
      </div>
      {/* A press on the bar's empty space puts the caret back in the URL line. */}
      <div
        class="url-bar"
        onMouseDown={(event) => {
          if (event.target !== event.currentTarget) return;
          event.preventDefault();
          field.current?.focus();
        }}
      >
        {tools}
        <button
          type="button"
          class={held ? "tool tool-convert is-held" : "tool tool-convert"}
          aria-label={held ? t.updHeld : t.convert}
          title={held ? undefined : busy || sending ? t.convertBusy : empty ? t.convertNothing : undefined}
          disabled={busy || sending || empty || held}
          aria-busy={busy || sending || undefined}
          onClick={() => void submit()}
        >
          <Icon name="ArrowRight" size={14} />
          <span class="tool-text">{held ? t.updHeld : t.convert}</span>
        </button>
      </div>
    </div>
  );
}
