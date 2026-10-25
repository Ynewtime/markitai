// Updating a loopback server from the workbench (contract v1.24). The entry
// is an icon with a dot beside the sidebar toggle, or in the phone's top bar
// before Docs and GitHub; it shows only when the server reports `update`. It
// carries the update's state (a spinner while the server works, red after a
// failure); hover or a long press names the version. It opens one card: a
// flyout beside the sidebar on a wide screen, a bottom sheet on a phone.
//
// The card has one row system: rows of 28 px, 12 px apart, a 16 px icon column
// and the text 24 px in, every action at one right edge. A setup install
// confirms inline (the restart and the conversions that finish first), then a
// stepper (download, verify, install, restart) and one status row follow the
// update; a failure says what it left. Another install shows its command.
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { UpdateStatus } from "../api/types.ts";
import type { UpdateControl } from "../hooks/use-update.ts";
import { useModal } from "../hooks/use-modal.ts";
import type { Dict } from "../i18n/index.ts";
import { copyText } from "../lib/clipboard.ts";
import { CANCELLABLE, failedStep, failureText, fraction, shownError, statusRow, stepOf } from "../lib/update.ts";
import { Icon } from "./icons.tsx";

/** How long a press must last to show the version instead of opening the card. */
const LONG_PRESS_MS = 500;
const TIP_MS = 1800;
const SWIPE_CLOSE = 80;

/** The entry: an icon button, and the version as its tip. */
export function UpdateEntry({ t, update, place }: { t: Dict; update: UpdateControl; place: "sidebar" | "phone" }) {
  const [tip, setTip] = useState(false);
  const pressed = useRef<ReturnType<typeof setTimeout> | null>(null);
  const long = useRef(false);
  const look = update.look;
  useEffect(() => {
    if (!tip) return;
    const timer = setTimeout(() => setTip(false), TIP_MS);
    return () => clearTimeout(timer);
  }, [tip]);
  if (look === null) return null;
  const status = update.status;
  const version = status?.target ?? status?.latest ?? "";
  const label =
    look === "failed"
      ? t.updFailed(version)
      : look === "busy"
        ? update.reconnecting || status?.state === "restarting"
          ? t.updRestartNote
          : t.updBusyNote(version)
        : t.updAvailable(version);
  const size = place === "phone" ? 19 : 17;
  return (
    <button
      type="button"
      class={`icon-btn upd-icon is-${look}${update.open ? " is-on" : ""}${tip ? " is-tip" : ""}`}
      aria-label={label}
      aria-haspopup="dialog"
      aria-expanded={update.open}
      data-tip={label}
      data-upd-entry=""
      onPointerDown={(event) => {
        if (event.pointerType === "mouse") return;
        long.current = false;
        pressed.current = setTimeout(() => {
          long.current = true;
          setTip(true);
        }, LONG_PRESS_MS);
      }}
      onPointerUp={() => {
        if (pressed.current !== null) clearTimeout(pressed.current);
        pressed.current = null;
      }}
      onPointerCancel={() => {
        if (pressed.current !== null) clearTimeout(pressed.current);
        pressed.current = null;
      }}
      onContextMenu={(event) => {
        if (long.current) event.preventDefault();
      }}
      onClick={() => {
        if (long.current) {
          long.current = false;
          return;
        }
        update.setOpen(!update.open);
      }}
    >
      {look === "busy" ? <span class="spinner" aria-hidden="true" /> : <Icon name={look === "failed" ? "WarningCircle" : "ArrowCircleUp"} size={size} />}
      {look === "available" && <span class="upd-dot" aria-hidden="true" />}
    </button>
  );
}

function Row({ icon, children, end, tone }: { icon: ComponentChildren; children: ComponentChildren; end?: ComponentChildren; tone?: "muted" | "error" }) {
  return (
    <div class={tone ? `upd-r is-${tone}` : "upd-r"}>
      <span class="upd-ic">{icon}</span>
      <span class="upd-tx">{children}</span>
      {end && <span class="upd-end">{end}</span>}
    </div>
  );
}

function Steps({ t, now, done, failed }: { t: Dict; now: number; done: number | null; failed: number }) {
  return (
    <div class="upd-steps" aria-hidden="true">
      {t.updSteps.map((label, index) => {
        const cls = failed === index ? "is-failed" : failed >= 0 ? (index < failed ? "is-done" : "") : index < now ? "is-done" : index === now ? "is-now" : "";
        return (
          <span key={label} class={cls} style={index === now && done !== null ? { "--p": `${Math.round(done * 100)}%` } : undefined}>
            {label}
          </span>
        );
      })}
    </div>
  );
}

/** The card's content for the server's state. */
export function UpdateCard({
  t,
  update,
  phone,
  onClose,
  onCopied,
}: {
  t: Dict;
  update: UpdateControl;
  phone: boolean;
  onClose: () => void;
  onCopied: (ok: boolean) => void;
}) {
  const status = update.status;
  if (!status) {
    return (
      <div class="upd-card" aria-busy="true">
        <Row icon={<span class="spinner" aria-hidden="true" />}>{t.loading}</Row>
      </div>
    );
  }
  const version = status.target ?? status.latest ?? status.current ?? "";
  const current = status.current ?? "";
  const command = status.command ?? "markitai update";
  const copy = () => void copyText(command).then(onCopied);
  const terminal = (
    <button type="button" class="btn btn-icon" aria-label={t.updCopyCommand(command)} title={t.updCopyCommand(command)} onClick={copy}>
      <Icon name="Terminal" size={16} />
    </button>
  );
  const head = (
    <Row
      icon={<Icon name="ArrowCircleUp" size={16} />}
      end={
        <>
          {status.notes_url && (
            <a class="icon-btn" href={status.notes_url} target="_blank" rel="noopener noreferrer" aria-label={t.updNotes} title={t.updNotes}>
              <Icon name="ArrowSquareOut" size={16} />
            </a>
          )}
          {phone && (
            <button type="button" class="icon-btn" aria-label={t.close} title={t.close} onClick={onClose}>
              <Icon name="X" size={16} />
            </button>
          )}
        </>
      }
    >
      <span class="upd-title">
        <b>{version}</b>
        <span>{t.updCurrent(current)}</span>
      </span>
    </Row>
  );
  const busy = update.reconnecting ? ({ ...status, state: "restarting" } as UpdateStatus) : status;
  const error = shownError(status, null);
  let body: ComponentChildren;
  if (busy.state !== "idle" && busy.state !== "failed") {
    const row = statusRow(t, busy, update.waited);
    const cancellable = CANCELLABLE.has(busy.state) && !update.reconnecting;
    body = (
      <>
        <Steps t={t} now={stepOf(busy.state)} done={busy.state === "downloading" ? fraction(busy) : null} failed={-1} />
        <div class="upd-r is-status" role="status">
          <span class="upd-ic">
            <span class="spinner" aria-hidden="true" />
          </span>
          <span class="upd-tx upd-pair">
            <span>{row.text}</span>
            {row.value && <span class="upd-num">{row.value}</span>}
          </span>
          {cancellable && (
            <span class="upd-end">
              <button type="button" class="btn" disabled={update.sending} onClick={update.cancel}>
                {t.cancel}
              </button>
            </span>
          )}
        </div>
      </>
    );
  } else if (error) {
    const words = failureText(t, error, current, version);
    const step = failedStep(error);
    body = (
      <>
        {step >= 0 && <Steps t={t} now={-1} done={null} failed={step} />}
        <div role="alert">
          <Row icon={<Icon name="XCircle" size={16} />} tone="error">
            {words.title}
          </Row>
          <p class="upd-sub">{words.left}</p>
        </div>
        <div class="upd-acts">
          {terminal}
          {error.rolled_back && error.code === "start_failed" ? (
            <button type="button" class="btn" onClick={update.dismiss}>
              {t.updDismiss}
            </button>
          ) : status.can_apply ? (
            <button type="button" class="btn btn-primary" disabled={update.sending} onClick={update.start}>
              {t.updRetry}
            </button>
          ) : (
            <button type="button" class="btn" onClick={update.dismiss}>
              {t.updDismiss}
            </button>
          )}
        </div>
      </>
    );
  } else if (update.confirming) {
    body = (
      <>
        <Row icon={<Icon name="Info" size={16} />} tone="muted">
          {t.updConfirmNote(status.waiting_jobs)}
        </Row>
        <div class="upd-acts">
          <button type="button" class="btn" onClick={() => update.setConfirming(false)}>
            {t.cancel}
          </button>
          <button type="button" class="btn btn-primary" disabled={update.sending} onClick={update.start}>
            {t.updConfirm}
          </button>
        </div>
      </>
    );
  } else {
    body = (
      <>
        {status.lead && <p class="upd-lead">{status.lead}</p>}
        {status.can_apply ? (
          <div class="upd-acts">
            {terminal}
            <button type="button" class="btn btn-primary" onClick={() => update.setConfirming(true)}>
              {t.updApply}
            </button>
          </div>
        ) : (
          <div class="sheet-cli">
            <code class="cli-text" title={command}>
              <span class="cli-prompt" aria-hidden="true">
                ${" "}
              </span>
              {command}
            </code>
            <button type="button" class="icon-btn sheet-copy" aria-label={t.updCopyCommand(command)} title={t.updCopyCommand(command)} onClick={copy}>
              <Icon name="Copy" size={16} />
            </button>
          </div>
        )}
      </>
    );
  }
  return (
    <div class="upd-card">
      {head}
      {body}
    </div>
  );
}

/** The card where it opens: a flyout beside the sidebar (wide), a bottom sheet (phone). */
export function UpdatePanel({ t, update, phone, onCopied }: { t: Dict; update: UpdateControl; phone: boolean; onCopied: (ok: boolean) => void }) {
  const close = () => {
    update.setOpen(false);
    update.setConfirming(false);
    document.querySelector<HTMLElement>("[data-upd-entry]:not([hidden])")?.focus();
  };
  return phone ? <UpdateSheet t={t} update={update} onClose={close} onCopied={onCopied} /> : <UpdatePopover t={t} update={update} onClose={close} onCopied={onCopied} />;
}

function UpdatePopover({ t, update, onClose, onCopied }: { t: Dict; update: UpdateControl; onClose: () => void; onCopied: (ok: boolean) => void }) {
  const card = useRef<HTMLDivElement>(null);
  const [place, setPlace] = useState<{ left: number; top: number } | null>(null);
  useLayoutEffect(() => {
    const entry = [...document.querySelectorAll<HTMLElement>("[data-upd-entry]")].find((node) => node.getClientRects().length > 0);
    const sidebar = document.getElementById("sidebar")?.getBoundingClientRect();
    const rect = entry?.getBoundingClientRect();
    if (rect) setPlace({ left: (sidebar?.right ?? rect.right) + 8, top: rect.top });
  }, []);
  useEffect(() => {
    const away = (event: PointerEvent) => {
      const target = event.target as Element | null;
      if (card.current?.contains(target) || target?.closest?.("[data-upd-entry]")) return;
      onClose();
    };
    const key = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    document.addEventListener("pointerdown", away, true);
    document.addEventListener("keydown", key);
    return () => {
      document.removeEventListener("pointerdown", away, true);
      document.removeEventListener("keydown", key);
    };
  }, [onClose]);
  return (
    <div
      ref={card}
      class="menu upd-pop"
      role="dialog"
      aria-label={t.updAvailable(update.status?.latest ?? "")}
      style={place ? { left: `${place.left}px`, top: `${place.top}px` } : { visibility: "hidden" }}
    >
      <UpdateCard t={t} update={update} phone={false} onClose={onClose} onCopied={onCopied} />
    </div>
  );
}

function UpdateSheet({ t, update, onClose, onCopied }: { t: Dict; update: UpdateControl; onClose: () => void; onCopied: (ok: boolean) => void }) {
  const dialog = useRef<HTMLDivElement>(null);
  const drag = useRef<number | null>(null);
  const [pull, setPull] = useState(0);
  useModal(dialog, onClose);
  return (
    <div
      class="veil opts-veil upd-veil"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={dialog}
        class={pull > 0 ? "dialog opts-dialog upd-sheet is-pulled" : "dialog opts-dialog upd-sheet"}
        role="dialog"
        aria-modal="true"
        aria-label={t.updAvailable(update.status?.latest ?? "")}
        tabIndex={-1}
        style={pull > 0 ? { transform: `translateY(${pull}px)` } : undefined}
      >
        <div
          class="sheet-grab"
          onPointerDown={(event) => {
            if (event.button !== 0) return;
            drag.current = event.clientY;
            (event.currentTarget as Element).setPointerCapture(event.pointerId);
          }}
          onPointerMove={(event) => {
            if (drag.current !== null) setPull(Math.max(0, event.clientY - drag.current));
          }}
          onPointerUp={(event) => {
            if (drag.current === null) return;
            const distance = event.clientY - drag.current;
            drag.current = null;
            if (distance > SWIPE_CLOSE) onClose();
            else setPull(0);
          }}
          onPointerCancel={() => {
            drag.current = null;
            setPull(0);
          }}
        >
          <span class="sheet-handle" aria-hidden="true" />
        </div>
        <div class="sheet-body">
          <UpdateCard t={t} update={update} phone onClose={onClose} onCopied={onCopied} />
        </div>
      </div>
    </div>
  );
}
