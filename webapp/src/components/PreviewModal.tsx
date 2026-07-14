import { useEffect, useRef } from "react";
import type { SessionItem } from "../hooks/useJobs";
import type { Dict } from "../i18n";
import { MarkdownPreview } from "./MarkdownPreview";
import { XIcon } from "./icons";

export function PreviewModal({
  t,
  item,
  createdAt,
  onClose,
  announce,
}: {
  t: Dict;
  item: SessionItem;
  createdAt: string | null;
  onClose: () => void;
  announce: (msg: string) => void;
}) {
  const dialogRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, []);

  useEffect(() => {
    dialogRef.current?.focus();
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        event.stopPropagation();
        onClose();
        return;
      }
      if (event.key !== "Tab") return;

      const dialog = dialogRef.current;
      if (dialog === null) return;
      const focusable = Array.from(
        dialog.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])',
        ),
      ).filter((element) => !element.hasAttribute("disabled") && element.offsetParent !== null);

      if (focusable.length === 0) {
        event.preventDefault();
        dialog.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (!(active instanceof HTMLElement) || !dialog.contains(active)) {
        event.preventDefault();
        first?.focus();
      } else if (event.shiftKey && (active === first || active === dialog)) {
        event.preventDefault();
        last?.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first?.focus();
      }
    };

    document.addEventListener("keydown", onKeyDown, true);
    return () => document.removeEventListener("keydown", onKeyDown, true);
  }, [onClose]);

  return (
    <div
      className="mdl-veil"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className="mdl preview-modal"
        role="dialog"
        aria-modal="true"
        aria-label={item.name}
        tabIndex={-1}
      >
        <div className="mdl-head preview-modal-head">
          <div className="preview-modal-title">
            <span>{t.previewAria}</span>
            <h2 title={item.name}>{item.name}</h2>
          </div>
          <button
            type="button"
            className="gearbtn"
            aria-label={t.close}
            title={t.close}
            onClick={onClose}
          >
            <XIcon size={16} />
          </button>
        </div>
        <MarkdownPreview t={t} item={item} createdAt={createdAt} announce={announce} />
      </div>
    </div>
  );
}
