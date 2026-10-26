// A "⋯" button with the actions that do not fit beside it: a menu that the
// arrow keys walk, Escape and a click elsewhere close, and that opens upward
// from a bar pinned to the bottom of the screen. Entries may come in groups
// split by hairlines; a destructive entry is last, in a group of its own, and
// asks first, in a confirmation anchored to the "⋯" button.
import { useEffect, useRef, useState } from "preact/hooks";
import type { IconName } from "./icon-paths.ts";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { Icon } from "./icons.tsx";

export interface MenuAction {
  key: string;
  label: string;
  icon?: IconName;
  run: () => void;
  disabled?: boolean;
  /** An external page: the entry is a link that opens it in a new tab. */
  href?: string;
  /** Words read after the label by assistive technology (e.g. "opens in a new tab"). */
  hint?: string;
  /** The saved file's extension (".md"), shown at the entry's end. */
  ext?: string;
  /** Entries of one group stay together, a hairline above each group after the first. */
  group?: string;
}

/** The menu's destructive entry: its words for the confirmation it opens. */
export interface MenuDanger {
  label: string;
  title: string;
  description: string;
  confirmLabel: string;
  cancelLabel: string;
  busyLabel: string;
  onConfirm: () => Promise<boolean>;
}

export function ActionMenu({
  label,
  actions,
  danger,
  up = false,
  disabled = false,
  plain = false,
  triggerRef,
  captions = {},
}: {
  label: string;
  actions: MenuAction[];
  danger?: MenuDanger;
  /** Open above the button (a bar at the bottom of the screen). */
  up?: boolean;
  disabled?: boolean;
  /** A bare icon among a window's other icon buttons, instead of a framed one. */
  plain?: boolean;
  /** Shares the "⋯" button, for a card that an entry opens beside it. */
  triggerRef?: { current: HTMLButtonElement | null };
  /** A group's caption, shown above its entries ("Download"). */
  captions?: Record<string, string>;
}) {
  const [open, setOpen] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const items = () => [...(root.current?.querySelectorAll<HTMLElement>('[role^="menuitem"]:not(:disabled)') ?? [])];
  useEffect(() => {
    if (!open) return;
    requestAnimationFrame(() => items()[0]?.focus());
    const away = (event: PointerEvent) => {
      const target = event.target as Element | null;
      // An entry's explanation bubble belongs to the menu.
      if (!root.current?.contains(target) && !target?.closest?.(".help-bubble")) setOpen(false);
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [open]);
  const onKeyDown = (event: KeyboardEvent) => {
    if (!open) return;
    const list = items();
    const index = list.indexOf(document.activeElement as HTMLElement);
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      setOpen(false);
      trigger.current?.focus();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      list[(index + (event.key === "ArrowDown" ? 1 : list.length - 1)) % list.length]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      list[event.key === "Home" ? 0 : list.length - 1]?.focus();
    } else if (event.key === "Tab") setOpen(false);
  };
  // Focus goes back to "⋯" before the action runs, so an action that moves it (a confirmation,
  // another window) still wins and one that does not leaves it on the button, not the page.
  const choose = (run: () => void) => {
    setOpen(false);
    trigger.current?.focus();
    run();
  };
  if (!actions.length && !danger) return null;
  const entry = (action: MenuAction) => {
    const mark = action.icon ? <Icon name={action.icon} size={15} /> : <span class="menu-gap" aria-hidden="true" />;
    const ext = action.ext && <span class="menu-ext">{action.ext}</span>;
    return action.href !== undefined ? (
      <a key={action.key} role="menuitem" tabIndex={-1} href={action.href} target="_blank" rel="noopener noreferrer" onClick={() => setOpen(false)}>
        {mark}
        {action.label}
        {action.hint && <span class="sr-only"> {action.hint}</span>}
        {ext}
      </a>
    ) : (
      <button key={action.key} type="button" role="menuitem" tabIndex={-1} disabled={action.disabled} onClick={() => choose(action.run)}>
        {mark}
        {action.label}
        {ext}
      </button>
    );
  };
  const dangerEntry = danger && (
    <button type="button" role="menuitem" tabIndex={-1} class="is-danger" onClick={() => choose(() => setConfirming(true))}>
      <Icon name="Trash" size={15} />
      {danger.label}
    </button>
  );
  const grouped = actions.some((action) => action.group !== undefined);
  const groups: [string, MenuAction[]][] = [];
  for (const action of actions) {
    const group = action.group ?? "";
    const last = groups[groups.length - 1];
    if (last && last[0] === group) last[1].push(action);
    else groups.push([group, [action]]);
  }
  return (
    <span ref={root} class="menu-wrap action-menu" onKeyDown={onKeyDown}>
      <button
        ref={(node) => {
          trigger.current = node;
          if (triggerRef) triggerRef.current = node;
        }}
        type="button"
        class={plain ? "icon-btn" : "btn btn-icon"}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-label={label}
        title={label}
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="DotsThree" size={18} />
      </button>
      {open && (
        <div class={up ? "menu is-up" : "menu"} role="menu" aria-label={label}>
          {grouped
            ? groups.map(([group, entries], at) => (
                <div key={`${group}-${at}`} role="group" aria-label={captions[group]} class={at > 0 ? "menu-group" : undefined}>
                  {captions[group] && (
                    <span class="menu-caption" aria-hidden="true">
                      {captions[group]}
                    </span>
                  )}
                  {entries.map(entry)}
                </div>
              ))
            : actions.map(entry)}
          {danger &&
            (grouped ? (
              <div role="group" class="menu-group">
                {dangerEntry}
              </div>
            ) : (
              dangerEntry
            ))}
        </div>
      )}
      {danger && (
        <ConfirmPopover
          triggerLabel={danger.label}
          title={danger.title}
          description={danger.description}
          confirmLabel={danger.confirmLabel}
          cancelLabel={danger.cancelLabel}
          busyLabel={danger.busyLabel}
          anchor={() => trigger.current}
          open={confirming}
          onClose={() => setConfirming(false)}
          onConfirm={danger.onConfirm}
        />
      )}
    </span>
  );
}
