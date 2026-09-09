import { useEffect, useId, useRef, useState, type RefObject } from "react";
import type { Dict, Locale } from "../i18n";
import { HistoryIcon, LogoMark, PaletteIcon, SettingsIcon } from "./icons";
import { LangToggle } from "./LangToggle";
import { ThemeToggle } from "./ThemeToggle";

/** External nav link: proper-noun label + mono ↗ + sr-only new-tab notice. */
function ExtLink({ href, label, srNote }: { href: string; label: string; srNote: string }) {
  return (
    <a href={href} target="_blank" rel="noreferrer">
      {label}
      <span className="ext" aria-hidden="true">
        ↗
      </span>
      <span className="sr-only"> {srNote}</span>
    </a>
  );
}

/** Phone-only footer for the header's external links: at ≤780px the header
 * collapses to brand + view icons and .hdr-links hides, so Docs/GitHub move
 * here. CSS gates both ends of the swap on the same breakpoint — exactly one
 * copy of the links is ever visible (display: none also drops the hidden one
 * from the accessibility tree). */
export function AppFooter({ t }: { t: Dict }) {
  return (
    <footer className="app-footer">
      <ExtLink href="https://markitai.dev" label={t.docsLabel} srNote={t.opensNewTab} />
      <ExtLink
        href="https://github.com/Ynewtime/markitai"
        label="GitHub"
        srNote={t.opensNewTab}
      />
    </footer>
  );
}

export function AppHeader({
  t,
  version,
  locale,
  onLocale,
  onHome,
  onHistory,
  historyActive,
  settingsOpen,
  onToggleSettings,
  gearRef,
}: {
  t: Dict;
  version: string | null;
  locale: Locale;
  onLocale: (l: Locale) => void;
  onHome: () => void;
  onHistory: () => void;
  historyActive: boolean;
  settingsOpen: boolean;
  onToggleSettings: () => void;
  gearRef: RefObject<HTMLButtonElement | null>;
}) {
  const [appearanceOpen, setAppearanceOpen] = useState(false);
  const appearanceRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const panelRef = useRef<HTMLDivElement>(null);
  const appearanceId = useId();

  useEffect(() => {
    if (!appearanceOpen) return;
    panelRef.current?.querySelector<HTMLButtonElement>('button[aria-pressed="true"]')?.focus();
    const dismiss = () => {
      setAppearanceOpen(false);
      triggerRef.current?.focus();
    };
    const onPointerDown = (event: PointerEvent) => {
      // Clicking elsewhere closes the menu but leaves focus where the click landed.
      if (event.target instanceof Node && !appearanceRef.current?.contains(event.target)) setAppearanceOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        dismiss();
      }
    };
    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [appearanceOpen]);

  return (
    <header className="apphdr">
      <div className="shell">
        <div className="brand">
          <a
            className="homelink"
            href="/"
            aria-label={t.homeAria}
            onClick={(e) => {
              e.preventDefault();
              onHome();
            }}
          >
            <LogoMark size={24} />
            <span className="wordmark">Markitai</span>
          </a>
          {version !== null && <span className="ver mono">v{version}</span>}
        </div>
        <nav className="hdr-links">
          <ExtLink href="https://markitai.dev" label={t.docsLabel} srNote={t.opensNewTab} />
          <ExtLink
            href="https://github.com/Ynewtime/markitai"
            label="GitHub"
            srNote={t.opensNewTab}
          />
        </nav>
        <div className="hdr-ctl">
          <div className="hdr-icons">
            <div className="hdr-appearance" ref={appearanceRef}
              onBlur={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget)) setAppearanceOpen(false);
              }}>
              <button ref={triggerRef} type="button" className="gearbtn"
                aria-label={t.appearanceTitle} title={t.appearanceTitle}
                aria-haspopup="dialog" aria-expanded={appearanceOpen} aria-controls={appearanceId}
                onClick={() => setAppearanceOpen((open) => !open)}>
                <PaletteIcon size={16} />
              </button>
              <div ref={panelRef} id={appearanceId} className="appearance-popover"
                role="dialog" aria-label={t.appearanceTitle} hidden={!appearanceOpen}>
                <div className="appearance-opt">
                  <span className="lbl">{t.langAria}</span>
                  <LangToggle label={t.langAria} locale={locale} onLocale={onLocale} />
                </div>
                <div className="appearance-opt">
                  <span className="lbl">{t.themeAria}</span>
                  <ThemeToggle t={t} label={t.themeAria} />
                </div>
              </div>
            </div>
            <button
              type="button"
              className={historyActive ? "gearbtn tasknav on" : "gearbtn tasknav"}
              aria-label={t.historyAria}
              aria-current={historyActive ? "page" : undefined}
              title={historyActive ? t.historyCurrent : t.historyAria}
              onClick={onHistory}
            >
              <HistoryIcon size={16} selected={historyActive} />
            </button>
            <button
              ref={gearRef}
              type="button"
              className="gearbtn"
              aria-label={t.settingsAria}
              aria-expanded={settingsOpen}
              title={t.settingsAria}
              onClick={onToggleSettings}
            >
              <SettingsIcon size={16} />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
