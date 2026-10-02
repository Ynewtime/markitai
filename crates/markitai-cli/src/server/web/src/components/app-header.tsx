// The 56px top bar: brand and version, Docs and GitHub, and three icon buttons
// (appearance, conversions, settings). At phone width the external links move
// to a footer band so exactly one copy is ever visible.
import type { Ref } from "preact";
import { useEffect, useRef, useState } from "preact/hooks";
import type { Dict, Locale } from "../i18n/index.ts";
import { Icon, Logo } from "./icons.tsx";

const DOCS = "https://markitai.dev";
const GITHUB = "https://github.com/Ynewtime/markitai";

function External({ href, label, note }: { href: string; label: string; note: string }) {
  return (
    <a href={href} target="_blank" rel="noopener noreferrer">
      {label}
      <span class="ext-mark" aria-hidden="true">
        ↗
      </span>
      <span class="sr-only"> {note}</span>
    </a>
  );
}

export function AppFooter({ t }: { t: Dict }) {
  return (
    <footer class="foot-links">
      <External href={DOCS} label={t.docsLabel} note={t.opensNewTab} />
      <External href={GITHUB} label="GitHub" note={t.opensNewTab} />
    </footer>
  );
}

type Theme = "auto" | "light" | "dark";
const THEMES: Theme[] = ["auto", "light", "dark"];
export const THEME_KEY = "markitai.theme";

function storedTheme(): Theme {
  try {
    const value = localStorage.getItem(THEME_KEY);
    return value === "light" || value === "dark" ? value : "auto";
  } catch {
    return "auto";
  }
}

function applyTheme(theme: Theme): void {
  const root = document.documentElement;
  if (theme === "auto") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", theme);
  try {
    if (theme === "auto") localStorage.removeItem(THEME_KEY);
    else localStorage.setItem(THEME_KEY, theme);
  } catch {
    /* The choice lasts for this page only. */
  }
}

function ThemeChoice({ t }: { t: Dict }) {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  useEffect(() => applyTheme(theme), [theme]);
  const names: Record<Theme, string> = { auto: t.themeAuto, light: t.themeLight, dark: t.themeDark };
  const icons = { auto: "Desktop", light: "Sun", dark: "Moon" } as const;
  const onKeyDown = (event: KeyboardEvent) => {
    const index = THEMES.indexOf(theme);
    const moves: Record<string, number> = {
      ArrowRight: (index + 1) % THEMES.length,
      ArrowDown: (index + 1) % THEMES.length,
      ArrowLeft: (index + THEMES.length - 1) % THEMES.length,
      ArrowUp: (index + THEMES.length - 1) % THEMES.length,
      Home: 0,
      End: THEMES.length - 1,
    };
    const next = moves[event.key];
    if (next === undefined) return;
    event.preventDefault();
    setTheme(THEMES[next] ?? "auto");
    buttons.current[next]?.focus();
  };
  return (
    <div class="toggle-group" role="radiogroup" aria-label={t.themeAria} onKeyDown={onKeyDown}>
      {THEMES.map((value, index) => (
        <button
          key={value}
          ref={(node) => {
            buttons.current[index] = node;
          }}
          type="button"
          role="radio"
          class={value === theme ? "is-on" : undefined}
          aria-checked={value === theme}
          aria-label={names[value]}
          title={names[value]}
          tabIndex={value === theme ? 0 : -1}
          onClick={() => setTheme(value)}
        >
          <Icon name={icons[value]} size={14} />
        </button>
      ))}
    </div>
  );
}

function LangChoice({ t, locale, onLocale }: { t: Dict; locale: Locale; onLocale: (locale: Locale) => void }) {
  return (
    <div class="toggle-group" role="group" aria-label={t.langAria}>
      {(
        [
          ["en", "EN"],
          ["zh", "中"],
        ] as const
      ).map(([value, label]) => (
        <button
          key={value}
          type="button"
          lang={value === "zh" ? "zh-CN" : "en"}
          class={value === locale ? "is-on text" : "text"}
          aria-pressed={value === locale}
          onClick={() => onLocale(value)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function Appearance({ t, locale, onLocale }: { t: Dict; locale: Locale; onLocale: (locale: Locale) => void }) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);
  const card = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    card.current?.querySelector<HTMLButtonElement>('[aria-pressed="true"]')?.focus();
    const away = (event: PointerEvent) => {
      if (event.target instanceof Node && !root.current?.contains(event.target)) setOpen(false);
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setOpen(false);
      trigger.current?.focus();
    };
    document.addEventListener("pointerdown", away);
    document.addEventListener("keydown", escape);
    return () => {
      document.removeEventListener("pointerdown", away);
      document.removeEventListener("keydown", escape);
    };
  }, [open]);
  return (
    <div
      class="appearance"
      ref={root}
      onFocusOut={(event) => {
        const next = (event as FocusEvent).relatedTarget;
        if (!(next instanceof Node) || !root.current?.contains(next)) setOpen(false);
      }}
    >
      <button
        ref={trigger}
        type="button"
        class="icon-btn"
        aria-label={t.appearanceTitle}
        title={t.appearanceTitle}
        aria-haspopup="dialog"
        aria-expanded={open}
        aria-controls="appearance-card"
        onClick={() => setOpen((value) => !value)}
      >
        <Icon name="Palette" size={16} />
      </button>
      <div ref={card} id="appearance-card" class="appearance-card" role="dialog" aria-label={t.appearanceTitle} hidden={!open}>
        <div class="appearance-row">
          <span class="appearance-label">{t.langAria}</span>
          <LangChoice t={t} locale={locale} onLocale={onLocale} />
        </div>
        <div class="appearance-row">
          <span class="appearance-label">{t.themeAria}</span>
          <ThemeChoice t={t} />
        </div>
      </div>
    </div>
  );
}

export function AppHeader({
  t,
  version,
  locale,
  onLocale,
  onHome,
  onWorkspace,
  workspaceActive,
  settingsOpen,
  onSettings,
  gearRef,
}: {
  t: Dict;
  version: string | null;
  locale: Locale;
  onLocale: (locale: Locale) => void;
  onHome: () => void;
  onWorkspace: () => void;
  workspaceActive: boolean;
  settingsOpen: boolean;
  onSettings: () => void;
  gearRef: Ref<HTMLButtonElement>;
}) {
  return (
    <header class="topbar">
      <div class="page topbar-inner">
        <div class="brand">
          <a
            class="brand-home"
            href="/"
            aria-label={t.homeAria}
            onClick={(event) => {
              event.preventDefault();
              onHome();
            }}
          >
            <Logo size={24} />
            <span class="brand-word">Markitai</span>
          </a>
          {version !== null && <span class="brand-ver">v{version}</span>}
        </div>
        <nav class="topbar-links">
          <External href={DOCS} label={t.docsLabel} note={t.opensNewTab} />
          <External href={GITHUB} label="GitHub" note={t.opensNewTab} />
        </nav>
        <div class="topbar-tools">
          <Appearance t={t} locale={locale} onLocale={onLocale} />
          <button
            type="button"
            class={workspaceActive ? "icon-btn is-current" : "icon-btn"}
            aria-label={t.historyAria}
            aria-current={workspaceActive ? "page" : undefined}
            title={workspaceActive ? t.historyCurrent : t.historyAria}
            onClick={onWorkspace}
          >
            <Icon name={workspaceActive ? "ClockCounterClockwiseBold" : "ClockCounterClockwise"} size={16} />
          </button>
          <button
            ref={gearRef}
            type="button"
            class="icon-btn"
            aria-label={t.settingsAria}
            aria-expanded={settingsOpen}
            title={t.settingsAria}
            onClick={onSettings}
          >
            <Icon name="Gear" size={16} />
          </button>
        </div>
      </div>
    </header>
  );
}
