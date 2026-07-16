import { createPortal } from "react-dom";
import { CheckIcon, WarningIcon, XIcon } from "./icons";

export type NotificationTone = "warning" | "success" | "error";

/** Portalled in-app notification used for actionable outcomes and test errors. */
export function AppNotification({
  tone,
  title,
  message,
  actionLabel,
  closeLabel,
  onAction,
  onClose,
}: {
  tone: NotificationTone;
  title: string;
  message: string;
  actionLabel?: string;
  closeLabel: string;
  onAction?: () => void;
  onClose: () => void;
}) {
  const icon =
    tone === "success" ? (
      <CheckIcon size={18} />
    ) : tone === "error" ? (
      <WarningIcon size={18} />
    ) : (
      <WarningIcon size={18} />
    );
  return createPortal(
    <aside
      className={`warning-notification ${tone}`}
      role={tone === "error" ? "alert" : "status"}
      aria-live={tone === "error" ? "assertive" : "polite"}
    >
      <span className="warning-notification-icon" aria-hidden="true">
        {icon}
      </span>
      <span className="warning-notification-copy">
        <strong>{title}</strong>
        <span>{message}</span>
        {actionLabel !== undefined && onAction !== undefined && (
          <button type="button" className="notification-action" onClick={onAction}>
            {actionLabel}
          </button>
        )}
      </span>
      <button
        type="button"
        className="notification-close"
        aria-label={closeLabel}
        title={closeLabel}
        onClick={onClose}
      >
        <XIcon size={14} />
      </button>
    </aside>,
    document.body,
  );
}

/** Actionable warning for an image conversion skipped by configuration. */
export function WarningNotification(
  props: Omit<Parameters<typeof AppNotification>[0], "tone">,
) {
  return <AppNotification tone="warning" {...props} />;
}
