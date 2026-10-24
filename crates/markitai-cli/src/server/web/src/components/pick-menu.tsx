// Small menus on one keyboard model (the arrow keys walk the entries, Home
// and End jump, Escape and a click elsewhere close and give focus back): a
// group of more than three choices, and a conversion's split Download
// button, whose caret lists the formats to save. Also the OCR languages' names.
import type { ComponentChildren } from "preact";
import { useEffect, useLayoutEffect, useRef, useState } from "preact/hooks";
import type { IconName } from "./icon-paths.ts";
import { Icon } from "./icons.tsx";

function useMenu() {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLSpanElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const items = () => [...(root.current?.querySelectorAll<HTMLElement>('[role^="menuitem"]:not(:disabled)') ?? [])];
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
  return { open, setOpen, root, trigger, close, onKeyDown };
}

/** The `ocr.lang` codes the picker names, in their own script; the default ("en") is the automatic policy. */
const LANGUAGES: Record<string, { name: string; lang: string }> = {
  zh: { name: "简体中文", lang: "zh-Hans" },
  zh_tw: { name: "繁體中文", lang: "zh-Hant" },
  ja: { name: "日本語", lang: "ja" },
  ko: { name: "한국어", lang: "ko" },
  ru: { name: "Русский", lang: "ru" },
  eslav: { name: "Русский", lang: "ru" },
  cyrillic: { name: "Кириллица", lang: "ru" },
  ar: { name: "العربية", lang: "ar" },
  th: { name: "ไทย", lang: "th" },
  hi: { name: "हिन्दी", lang: "hi" },
  devanagari: { name: "हिन्दी", lang: "hi" },
  el: { name: "Ελληνικά", lang: "el" },
  ta: { name: "தமிழ்", lang: "ta" },
  te: { name: "తెలుగు", lang: "te" },
  latin: { name: "Latin", lang: "la" },
};

/** A language's name: the automatic policy in the interface's words, a known code in its own script, others as written. */
export function languageName(code: string, automatic: string): string {
  return code === "en" ? automatic : (LANGUAGES[code]?.name ?? code);
}

/** The language tag a code's name is written in (none for the automatic policy). */
export function languageTag(code: string): string | undefined {
  return code === "en" ? undefined : (LANGUAGES[code]?.lang ?? undefined);
}

/** Groups of more than this many single choices are a menu button, smaller ones segments (owner rule, 2026-10-24). */
export const SEGMENTS_MAX = 3;

/** One choice of several as a menu button: the current value and a chevron; the menu checks the current
 * one. A disabled choice stays reachable (aria-disabled) so its description can say why. */
export function ChoiceMenu<T extends string | null>({
  name,
  value,
  choices,
  label,
  describedBy,
  disabled,
  class: extra,
  onPick,
}: {
  /** What the group sets ("URL"), read before the value. */
  name: string;
  value: T | null;
  choices: readonly T[];
  label: (choice: T) => string;
  describedBy?: (choice: T) => string;
  disabled?: (choice: T) => boolean;
  class?: string;
  onPick: (choice: T) => void;
}) {
  const menu = useMenu();
  const current = choices.find((choice) => choice === value);
  const list = useRef<HTMLDivElement>(null);
  const [up, setUp] = useState(false);
  // The menu opens downward while it fits above the phone's tab bar (or the window's edge), else
  // upward when there is more room there; either way it is capped to that room and scrolls.
  useLayoutEffect(() => {
    const box = list.current;
    const button = menu.trigger.current;
    if (!menu.open || !box || !button) return setUp(false);
    const margin = 8;
    const bar = document.querySelector<HTMLElement>(".tabbar");
    const floor = bar && bar.getClientRects().length > 0 ? bar.getBoundingClientRect().top : window.innerHeight;
    const rect = button.getBoundingClientRect();
    const below = floor - rect.bottom - 6 - margin;
    const above = rect.top - 6 - margin;
    box.style.maxHeight = "";
    const flip = box.scrollHeight > below && above > below;
    setUp(flip);
    box.style.maxHeight = `${Math.max(120, Math.floor(flip ? above : below))}px`;
  }, [menu.open]);
  return (
    <span ref={menu.root} class={extra ? `menu-wrap ${extra}` : "menu-wrap"} onKeyDown={menu.onKeyDown}>
      <button
        ref={menu.trigger}
        type="button"
        class="opt-choice"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        aria-describedby={current !== undefined ? describedBy?.(current) : undefined}
        onClick={() => menu.setOpen((open) => !open)}
      >
        <span class="sr-only">{name}: </span>
        <span class="opt-choice-value">{current !== undefined ? label(current) : "—"}</span>
        <Icon name="CaretDown" size={11} />
      </button>
      {menu.open && (
        <div ref={list} class={up ? "menu is-start is-up opt-choice-menu" : "menu is-start opt-choice-menu"} role="menu" aria-label={name}>
          {choices.map((choice) => {
            const off = disabled?.(choice) ?? false;
            return (
              <button
                key={choice ?? "default"}
                type="button"
                role="menuitemradio"
                aria-checked={choice === value}
                aria-disabled={off || undefined}
                aria-describedby={describedBy?.(choice)}
                tabIndex={-1}
                onClick={() => {
                  if (off) return;
                  onPick(choice);
                  menu.close();
                }}
              >
                <span class="menu-check" aria-hidden="true">
                  {choice === value && <Icon name="Check" size={13} />}
                </span>
                <span>{label(choice)}</span>
              </button>
            );
          })}
        </div>
      )}
    </span>
  );
}

export interface DownloadEntry {
  key: string;
  label: string;
  /** The saved file's extension (".md"), shown at the entry's end. */
  ext?: string;
  icon: IconName;
  run: () => void;
}

/** Download, and beside it a caret with the formats: the button saves the first entry. */
export function SplitDownload({
  label,
  menuLabel,
  title,
  entries,
  disabled = false,
  children,
}: {
  label: string;
  /** The caret's name ("Download as"). */
  menuLabel: string;
  title?: string;
  entries: DownloadEntry[];
  disabled?: boolean;
  children?: ComponentChildren;
}) {
  const menu = useMenu();
  const first = entries[0];
  return (
    <span ref={menu.root} class="menu-wrap split-btn" onKeyDown={menu.onKeyDown}>
      <button type="button" class="btn" title={title} disabled={disabled || !first} onClick={() => first?.run()}>
        <Icon name="DownloadSimple" size={14} />
        {label}
        {children}
      </button>
      <button
        ref={menu.trigger}
        type="button"
        class="btn split-caret"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        aria-label={menuLabel}
        title={menuLabel}
        disabled={disabled}
        onClick={() => menu.setOpen((open) => !open)}
      >
        <Icon name="CaretDown" size={11} />
      </button>
      {menu.open && (
        <div class="menu download-menu" role="menu" aria-label={menuLabel}>
          {entries.map((entry) => (
            <button
              key={entry.key}
              type="button"
              role="menuitem"
              tabIndex={-1}
              onClick={() => {
                // focus goes back to the caret (an entry that moves it, as Choose files does, wins)
                menu.close();
                entry.run();
              }}
            >
              <Icon name={entry.icon} size={15} />
              {entry.label}
              {entry.ext && <span class="menu-ext">{entry.ext}</span>}
            </button>
          ))}
        </div>
      )}
    </span>
  );
}
