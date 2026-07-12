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

export function AppHeader({
  t,
  version,
  locale,
  onLocale,
  historyActive,
  onHome,
  onHistory,
  settingsOpen,
  onToggleSettings,
  gearRef,
}: {
  t: Dict;
  version: string | null;
  locale: Locale;
  onLocale: (l: Locale) => void;
  historyActive: boolean;
  onHome: () => void;
  onHistory: () => void;
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
            <ExtLink href="https://markitai.dev" label="Docs" srNote={t.opensNewTab} />
            <ExtLink
              href="https://github.com/Ynewtime/markitai"
              label="GitHub"
              srNote={t.opensNewTab}
            />
          </nav>
          <LangToggle label={t.langAria} locale={locale} onLocale={onLocale} />
          <ThemeToggle t={t} label={t.themeAria} />
          <button
            type="button"
            className={historyActive ? "gearbtn on" : "gearbtn"}
            aria-label={t.historyAria}
            aria-pressed={historyActive}
            title={t.historyAria}
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
        </div>
      </div>
    </header>
  );
}
