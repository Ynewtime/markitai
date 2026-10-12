// Settings as pages of the sidebar's Settings section: Models (see
// settings-models), Appearance and language, and Access token. On phones
// `/settings` lists the three sections and the external links in one list.
import { useEffect, useRef, useState } from "preact/hooks";
import { hasToken, setToken } from "../api/token.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { LOCALES } from "../i18n/locale.ts";
import { ownClick, type Route, type Section } from "../lib/route.ts";
import { applyTheme, storedTheme, THEMES, type Theme } from "../lib/theme.ts";
import { Icon } from "./icons.tsx";
import type { IconName } from "./icon-paths.ts";

/** A radio group of buttons: arrows move and choose, as a native radio group does. */
function Choice<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string; icon?: IconName; lang?: string }[]; onChange: (value: T) => void }) {
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);
  const index = options.findIndex((option) => option.value === value);
  const onKeyDown = (event: KeyboardEvent) => {
    const moves: Record<string, number> = {
      ArrowRight: (index + 1) % options.length,
      ArrowDown: (index + 1) % options.length,
      ArrowLeft: (index + options.length - 1) % options.length,
      ArrowUp: (index + options.length - 1) % options.length,
      Home: 0,
      End: options.length - 1,
    };
    const next = moves[event.key];
    if (next === undefined) return;
    event.preventDefault();
    const option = options[next];
    if (!option) return;
    onChange(option.value);
    buttons.current[next]?.focus();
  };
  return (
    <div class="choice" role="radiogroup" aria-label={label} onKeyDown={onKeyDown}>
      {options.map((option, position) => (
        <button
          key={option.value}
          ref={(node) => {
            buttons.current[position] = node;
          }}
          type="button"
          role="radio"
          lang={option.lang}
          aria-checked={option.value === value}
          tabIndex={option.value === value ? 0 : -1}
          class={option.value === value ? "is-on" : undefined}
          onClick={() => onChange(option.value)}
        >
          {option.icon && <Icon name={option.icon} size={15} />}
          {option.label}
        </button>
      ))}
    </div>
  );
}

const LANGUAGE_NAMES: Record<Locale, string> = { en: "English", zh: "中文", ja: "日本語" };
const LANGUAGE_TAGS: Record<Locale, string> = { en: "en", zh: "zh-CN", ja: "ja" };

export function AppearanceSettings({ t, locale, onLocale }: { t: Dict; locale: Locale; onLocale: (locale: Locale) => void }) {
  const [theme, setTheme] = useState<Theme>(storedTheme);
  useEffect(() => applyTheme(theme), [theme]);
  const names: Record<Theme, string> = { auto: t.themeAuto, light: t.themeLight, dark: t.themeDark };
  const icons: Record<Theme, IconName> = { auto: "Desktop", light: "Sun", dark: "Moon" };
  return (
    <section class="settings-page" aria-labelledby="appearance-title">
      <div class="page-head">
        <h1 id="appearance-title">{t.navAppearance}</h1>
      </div>
      <div class="settings-body">
        <div class="setting">
          <div class="setting-copy">
            <h2 id="language-label">{t.langAria}</h2>
            <p>{t.languageHint}</p>
          </div>
          <Choice label={t.langAria} value={locale} options={LOCALES.map((value) => ({ value, label: LANGUAGE_NAMES[value], lang: LANGUAGE_TAGS[value] }))} onChange={onLocale} />
        </div>
        <div class="setting">
          <div class="setting-copy">
            <h2>{t.themeAria}</h2>
            <p>{t.themeHint}</p>
          </div>
          <Choice label={t.themeAria} value={theme} options={THEMES.map((value) => ({ value, label: names[value], icon: icons[value] }))} onChange={setTheme} />
        </div>
      </div>
    </section>
  );
}

export function AccessSettings({ t, unauthorized, onToken }: { t: Dict; unauthorized: boolean; onToken: () => void }) {
  const [value, setValue] = useState("");
  const [held, setHeld] = useState(hasToken);
  const [saved, setSaved] = useState(false);
  // One status line: what this tab holds, with Forget beside it; one field with
  // Save attached; one hint under the field.
  const refused = unauthorized && !saved;
  const state = refused ? t.accessRefused : saved ? t.tokenSaved : held ? t.accessHeld : t.accessNone;
  const hint = refused ? t.accessRefusedHint : held ? t.replaceToken : t.accessExplain;
  return (
    <section class="settings-page" aria-labelledby="access-title">
      <div class="page-head">
        <h1 id="access-title">{t.navAccess}</h1>
      </div>
      <div class="settings-body access">
        <p class={refused ? "access-state is-error" : "access-state"} role="status">
          <Icon name={refused ? "Warning" : held ? "Check" : "Key"} size={15} />
          <span>{state}</span>
          {held && (
            <button
              type="button"
              class="text-link"
              onClick={() => {
                setToken("");
                setHeld(false);
                setSaved(false);
              }}
            >
              {t.forgetToken}
            </button>
          )}
        </p>
        <form
          class="access-form"
          onSubmit={(event) => {
            event.preventDefault();
            if (!value.trim()) return;
            setToken(value);
            setValue("");
            setHeld(true);
            setSaved(true);
            onToken();
          }}
        >
          <label class="field-label" for="access-token">
            {t.tokenField}
          </label>
          <div class="input-group">
            <input
              id="access-token"
              type="password"
              value={value}
              autoComplete="off"
              spellcheck={false}
              aria-describedby="access-hint"
              onInput={(event) => {
                setValue(event.currentTarget.value);
                setSaved(false);
              }}
            />
            <button type="submit" class="btn btn-primary" disabled={!value.trim()}>
              {t.save}
            </button>
          </div>
          <p id="access-hint" class="field-hint">
            {hint} {t.tokenNote}
          </p>
        </form>
      </div>
    </section>
  );
}

/** Phones: the section list behind the tab bar's Settings. */
export function SettingsIndex({ t, version, docsUrl, sourceUrl, onNavigate }: { t: Dict; version: string | null; docsUrl: string; sourceUrl: string; onNavigate: (route: Route) => void }) {
  const entry = (section: Section, icon: IconName, label: string, hint: string) => (
    <li>
      <a
        href={`/settings/${section}`}
        onClick={(event) => {
          if (!ownClick(event)) return;
          event.preventDefault();
          onNavigate({ view: "settings", section });
        }}
      >
        <Icon name={icon} size={18} />
        <span class="index-copy">
          <span class="index-label">{label}</span>
          <span class="index-hint">{hint}</span>
        </span>
        <Icon name="CaretRight" size={14} />
      </a>
    </li>
  );
  return (
    <section class="settings-page" aria-labelledby="settings-index-title">
      <div class="page-head">
        <h1 id="settings-index-title">{t.settingsTitle}</h1>
      </div>
      <ul class="settings-index">
        {entry("models", "Cpu", t.navModels, t.modelsHint)}
        {entry("appearance", "Palette", t.navAppearanceShort, t.appearanceHint)}
        {entry("access", "Key", t.navAccess, t.accessHint)}
        <li>
          <a href={docsUrl} target="_blank" rel="noopener noreferrer">
            <Icon name="BookOpen" size={18} />
            <span class="index-copy">
              <span class="index-label">{t.docsLabel}</span>
            </span>
            <Icon name="ArrowSquareOut" size={14} />
            <span class="sr-only"> {t.opensNewTab}</span>
          </a>
        </li>
        <li>
          <a href={sourceUrl} target="_blank" rel="noopener noreferrer">
            <Icon name="GithubLogo" size={18} />
            <span class="index-copy">
              <span class="index-label">GitHub</span>
            </span>
            <Icon name="ArrowSquareOut" size={14} />
            <span class="sr-only"> {t.opensNewTab}</span>
          </a>
        </li>
      </ul>
      {version && <p class="version">markitai {version}</p>}
    </section>
  );
}
