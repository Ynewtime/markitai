import type { RefObject } from "react";
import type { Dict, Locale } from "../i18n";
import { HistoryIcon, LogoMark, SettingsIcon } from "./icons";
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
            <span className="wordmark">markitai</span>
          </a>
          {version !== null && <span className="ver mono">v{version}</span>}
        </div>
        <div className="hdr-ctl">
          <nav className="hdr-links">
            <ExtLink href="https://markitai.dev" label={t.docsLabel} srNote={t.opensNewTab} />
            <ExtLink
              href="https://github.com/Ynewtime/markitai"
              label="GitHub"
              srNote={t.opensNewTab}
            />
          </nav>
          {/* grouping spans: the phone header grid dissolves .hdr-ctl
              (display: contents) and needs the toggles and the two nav icons
              to travel as units — desktop spacing is unchanged */}
          <span className="hdr-toggles">
            <LangToggle label={t.langAria} locale={locale} onLocale={onLocale} />
            <ThemeToggle t={t} label={t.themeAria} />
          </span>
          <span className="hdr-icons">
            <button
              type="button"
              className={historyActive ? "gearbtn tasknav on" : "gearbtn tasknav"}
              aria-label={t.historyAria}
              aria-current={historyActive ? "page" : undefined}
              title={historyActive ? t.historyCurrent : t.historyAria}
              onClick={onHistory}
            >
              <HistoryIcon size={16} />
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
          </span>
        </div>
      </div>
    </header>
  );
}
