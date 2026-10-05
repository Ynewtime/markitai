// The 1120px preview dialog (a full-screen sheet on phones). A URL item's title
// links to the page it came from; conversion warnings live in the item's
// notification, not here.
import { useRef } from "preact/hooks";
import { useModal } from "../hooks/use-modal.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { Icon } from "./icons.tsx";
import { MarkdownView, type PreviewTarget } from "./markdown-view.tsx";

export function PreviewModal({
  t,
  locale,
  item,
  kind,
  createdAt,
  onClose,
  announce,
  describe,
}: {
  t: Dict;
  locale: Locale;
  item: PreviewTarget;
  kind: "file" | "url";
  createdAt: string | null;
  onClose: () => void;
  announce: (message: string) => void;
  describe: (error: unknown) => { text: string; detail: string };
}) {
  const dialog = useRef<HTMLDivElement>(null);
  useModal(dialog, onClose);
  const link = kind === "url" && /^https?:\/\//i.test(item.name) ? item.name : null;
  return (
    <div
      class="veil veil-sheet"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div ref={dialog} class="dialog dialog-preview" role="dialog" aria-modal="true" aria-label={item.name} tabIndex={-1}>
        <div class="dialog-head preview-head">
          <div class="preview-title">
            <span>{t.previewAria}</span>
            <h2 title={item.name}>
              {link === null ? (
                item.name
              ) : (
                <a href={link} target="_blank" rel="noopener noreferrer" title={`${item.name} ${t.opensNewTab}`}>
                  <span>{item.name}</span>
                  <Icon name="ArrowSquareOut" size={13} />
                </a>
              )}
            </h2>
          </div>
          <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={onClose}>
            <Icon name="X" size={16} />
          </button>
        </div>
        <MarkdownView t={t} locale={locale} item={item} createdAt={createdAt} announce={announce} describe={describe} />
      </div>
    </div>
  );
}
