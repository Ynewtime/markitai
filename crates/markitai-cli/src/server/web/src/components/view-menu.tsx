// Preview's view switch when the window's tab row has no room for its tabs (a
// phone): one button that names the view and version on screen ("Rendered ·
// LLM"), opening a menu with the views, the versions and the contents column.
// The arrow keys walk it, Home and End jump, Escape and a click elsewhere
// close it and give focus back to the button.
import { useEffect, useRef, useState } from "preact/hooks";
import { Icon } from "./icons.tsx";

interface Choice<T extends string> {
  value: T;
  label: string;
  count?: number | undefined;
}

export function ViewMenu<V extends string>({
  label,
  views,
  view,
  onView,
  versions,
  versionLabel,
  version,
  onVersion,
  contents,
}: {
  label: string;
  views: Choice<V>[];
  view: V;
  onView: (value: V) => void;
  versions: Choice<string>[];
  versionLabel: string;
  /** The version on screen, or null when the view has none (Diff, Files). */
  version: string | null;
  onVersion: (value: string) => void;
  contents: { label: string; on: boolean; toggle: () => void } | null;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const items = () => [...(root.current?.querySelectorAll<HTMLElement>('[role^="menuitem"]') ?? [])];
  useEffect(() => {
    if (!open) return;
    requestAnimationFrame(() => (items().find((item) => item.getAttribute("aria-checked") === "true") ?? items()[0])?.focus());
    const away = (event: PointerEvent) => {
      if (!root.current?.contains(event.target as Node)) setOpen(false);
    };
    document.addEventListener("pointerdown", away, true);
    return () => document.removeEventListener("pointerdown", away, true);
  }, [open]);
  const close = () => {
    setOpen(false);
    trigger.current?.focus();
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (!open) return;
    const list = items();
    const index = list.indexOf(document.activeElement as HTMLElement);
    if (event.key === "Escape") {
      event.preventDefault();
      event.stopPropagation();
      close();
    } else if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      list[(index + (event.key === "ArrowDown" ? 1 : list.length - 1)) % list.length]?.focus();
    } else if (event.key === "Home" || event.key === "End") {
      event.preventDefault();
      list[event.key === "Home" ? 0 : list.length - 1]?.focus();
    } else if (event.key === "Tab") setOpen(false);
  };
  const current = views.find((choice) => choice.value === view);
  const currentVersion = versions.find((choice) => choice.value === version);
  const radio = (checked: boolean, text: string, count: number | undefined, run: () => void, key: string) => (
    <button
      key={key}
      type="button"
      role="menuitemradio"
      aria-checked={checked}
      tabIndex={-1}
      onClick={() => {
        run();
        close();
      }}
    >
      <span class="menu-check" aria-hidden="true">
        {checked && <Icon name="Check" size={13} />}
      </span>
      {text}
      {count !== undefined && <span class="tab-count">{count}</span>}
    </button>
  );
  return (
    <span ref={root} class="menu-wrap view-menu" onKeyDown={onKeyDown}>
      {/* Its name starts with what it shows ("Rendered, LLM"); the switch's purpose follows, unseen. */}
      <button ref={trigger} type="button" class="view-menu-btn" aria-haspopup="menu" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        <span>{current?.label}</span>
        {currentVersion && (
          <span class="view-menu-version">
            <span class="sr-only">, </span>
            {currentVersion.label}
          </span>
        )}
        <span class="sr-only">, {label}</span>
        <Icon name="CaretDown" size={12} />
      </button>
      {open && (
        <div class="menu" role="menu" aria-label={label}>
          <div role="group" aria-label={label}>
            {views.map((choice) => radio(choice.value === view, choice.label, choice.count, () => onView(choice.value), `view-${choice.value}`))}
          </div>
          {versions.length > 0 && version !== null && (
            <div role="group" aria-label={versionLabel} class="menu-group">
              {versions.map((choice) => radio(choice.value === version, choice.label, undefined, () => onVersion(choice.value), `version-${choice.value}`))}
            </div>
          )}
          {contents && (
            <div role="group" class="menu-group">
              <button
                type="button"
                role="menuitemcheckbox"
                aria-checked={contents.on}
                tabIndex={-1}
                onClick={() => {
                  contents.toggle();
                  close();
                }}
              >
                <span class="menu-check" aria-hidden="true">
                  {contents.on && <Icon name="Check" size={13} />}
                </span>
                {contents.label}
              </button>
            </div>
          )}
        </div>
      )}
    </span>
  );
}
