// Two small menus on one keyboard model (the arrow keys walk the entries, Home
// and End jump, Escape and a click elsewhere close and give focus back):
// the OCR language picker after the OCR switch, and a conversion's split
// Download button, whose caret lists the formats to save.
import type { ComponentChildren } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
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

export function OcrLanguageMenu({
  label,
  engine,
  automatic,
  automaticShort,
  automaticHint,
  languages,
  value,
  describedBy,
  onPick,
}: {
  /** What the control sets ("OCR language"). */
  label: string;
  /** Where OCR runs ("Local"), before the language. */
  engine: string;
  automatic: string;
  automaticShort: string;
  /** The scripts the automatic policy reads ("中 · 英 · 日"). */
  automaticHint: string;
  languages: string[];
  value: string;
  describedBy?: string;
  onPick: (code: string) => void;
}) {
  const menu = useMenu();
  return (
    <span ref={menu.root} class="menu-wrap opt-pick-wrap" onKeyDown={menu.onKeyDown}>
      <button
        ref={menu.trigger}
        type="button"
        class="opt-pick"
        aria-haspopup="menu"
        aria-expanded={menu.open}
        aria-describedby={describedBy}
        onClick={() => menu.setOpen((open) => !open)}
      >
        <Icon name="Translate" size={14} />
        <span class="sr-only">{label}: </span>
        <span>
          {engine} · {languageName(value, automatic)}
        </span>
        <Icon name="CaretDown" size={11} />
      </button>
      {menu.open && (
        <div class="menu is-start opt-pick-menu" role="menu" aria-label={label}>
          {languages.map((code) => {
            const known = LANGUAGES[code];
            return (
              <button
                key={code}
                type="button"
                role="menuitemradio"
                aria-checked={code === value}
                tabIndex={-1}
                onClick={() => {
                  onPick(code);
                  menu.close();
                }}
              >
                <span class="menu-check" aria-hidden="true">
                  {code === value && <Icon name="Check" size={13} />}
                </span>
                {code === "en" ? (
                  <span>{automaticShort}</span>
                ) : (
                  <span lang={known?.lang} dir="auto">
                    {known?.name ?? code}
                  </span>
                )}
                {code === "en" && <span class="menu-hint">{automaticHint}</span>}
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
                menu.setOpen(false);
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
