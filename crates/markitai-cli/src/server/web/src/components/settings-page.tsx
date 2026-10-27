// Settings as pages of the sidebar's Settings section: Appearance and
// language, Models (see settings-models), Access token and Edit config (see
// settings-config); Access token and Edit config only for the admin token, Models
// for a user token too (v1.15: its own models and the shared ones). On phones
// `/settings` lists the sections, then one line with the version, the commit and the build time.
import type { ComponentChildren } from "preact";
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import { createUserToken, fetchUserTokens, regenerateToken, revokeUserToken } from "../api/client.ts";
import { hasToken, setToken } from "../api/token.ts";
import type { RegeneratedToken, UserToken } from "../api/types.ts";
import { commitLabel, type VersionLine } from "../lib/build.ts";
import type { Dict, Locale } from "../i18n/index.ts";
import { LOCALES } from "../i18n/locale.ts";
import { ownClick, type Route, type Section } from "../lib/route.ts";
import { applyTheme, storedTheme, THEMES, type Theme } from "../lib/theme.ts";
import { fmtClock } from "../lib/format.ts";
import { copyText } from "../lib/clipboard.ts";
import { qrModules, qrPath } from "../lib/qr.ts";
import { ConfirmPopover } from "./confirm-popover.tsx";
import { InfoTip } from "./help-tooltip.tsx";
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

export function AppearanceSettings({
  t,
  locale,
  onLocale,
  userAccess,
}: {
  t: Dict;
  locale: Locale;
  onLocale: (locale: Locale) => void;
  /** Signed in with a user token: which one, and Sign out (another token signs in from the sign-in screen). */
  userAccess: { label: string | null } | null;
}) {
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
          </div>
          <Choice label={t.langAria} value={locale} options={LOCALES.map((value) => ({ value, label: LANGUAGE_NAMES[value], lang: LANGUAGE_TAGS[value] }))} onChange={onLocale} />
        </div>
        <div class="setting">
          <div class="setting-copy">
            <h2>{t.themeAria}</h2>
          </div>
          <Choice label={t.themeAria} value={theme} options={THEMES.map((value) => ({ value, label: names[value], icon: icons[value] }))} onChange={setTheme} />
        </div>
        {userAccess && (
          <div class="setting">
            <div class="setting-copy">
              <h2>{t.userAccessTitle}</h2>
              <p>{userAccess.label ? t.userAccessNamed(userAccess.label) : t.userAccessText}</p>
            </div>
            <div class="setting-actions">
              <button
                type="button"
                class="btn"
                title={t.signOutTabHint}
                onClick={() => {
                  setToken("");
                  location.reload();
                }}
              >
                {t.signOutTab}
              </button>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/** The admin's list of user tokens: create one with a label (its token shown once), revoke one. */
function UserTokens({ t, locale, describe }: { t: Dict; locale: Locale; describe: (error: unknown) => { text: string; detail: string } }) {
  const [tokens, setTokens] = useState<UserToken[] | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [label, setLabel] = useState("");
  const [creating, setCreating] = useState(false);
  const [fresh, setFresh] = useState<UserToken | null>(null);
  const load = () =>
    fetchUserTokens().then(
      (list) => setTokens(list),
      (error: unknown) => setFailure(describe(error).text),
    );
  useEffect(() => void load(), []);
  const listed = (tokens ?? []).filter((token) => token.id !== fresh?.id);
  const create = async (event: Event) => {
    event.preventDefault();
    const name = label.trim();
    if (!name || creating) return;
    setCreating(true);
    setFailure(null);
    try {
      const made = await createUserToken(name);
      setFresh(made);
      setLabel("");
      await load();
    } catch (error) {
      setFailure(describe(error).text);
    } finally {
      setCreating(false);
    }
  };
  return (
    <section class="access-section user-tokens" aria-labelledby="user-tokens-title">
      <div class="group-head-row">
        <h2 id="user-tokens-title" class="group-title">
          {t.userTokensTitle}
        </h2>
        <InfoTip label={t.userTokensInfo} text={t.userTokensText} />
      </div>
      {fresh?.token ? (
        <NewToken
          t={t}
          heading={t.userTokenCreated(fresh.label)}
          token={{ token: fresh.token, network_url: null, persistent: true, startup_variable: null }}
          onDone={() => {
            setFresh(null);
            void load();
          }}
        />
      ) : (
        <form class="token-create" onSubmit={create}>
          <label class="field-label" for="user-token-label">
            {t.userTokenLabel}
          </label>
          <div class="input-group">
            <input
              id="user-token-label"
              type="text"
              value={label}
              maxLength={60}
              autoComplete="off"
              placeholder={t.userTokenPlaceholder}
              onInput={(event) => setLabel(event.currentTarget.value)}
            />
            <button type="submit" class="btn btn-primary" disabled={!label.trim() || creating} aria-busy={creating || undefined}>
              <Icon name="Plus" size={14} />
              {t.userTokenCreate}
            </button>
          </div>
        </form>
      )}
      {failure && (
        <p class="line-error" role="alert">
          {failure}
        </p>
      )}
      {tokens === null ? (
        !failure && <p class="field-hint">{t.loading}</p>
      ) : listed.length === 0 ? (
        // While a new token is shown, the list waits for Done.
        fresh ? null : (
        <p class="token-empty">{t.userTokensNone}</p>
        )
      ) : (
        <ul class="token-list" aria-label={t.userTokensTitle}>
          {listed.map((token) => (
            <li key={token.id}>
              <span class="token-name">{token.label}</span>
              <code class="token-hint" title={t.userTokenHint}>
                …{token.hint}
              </code>
              <span class="token-when">
                {token.last_used_at ? t.userTokenUsed(fmtClock(token.last_used_at, locale)) : t.userTokenCreatedAt(fmtClock(token.created_at, locale))}
                {/* v1.15: how many models of its own it keeps (never which keys). */}
                {token.model_count !== undefined && token.model_count > 0 && (
                  <>
                    {" · "}
                    <span class="token-models" title={t.userTokenModelsHint}>
                      {t.userTokenModels(token.model_count)}
                    </span>
                  </>
                )}
              </span>
              <ConfirmPopover
                triggerLabel={t.userTokenRevokeAria(token.label)}
                triggerText={t.userTokenRevoke}
                title={t.userTokenRevokeTitle(token.label)}
                description={t.userTokenRevokeText(token.model_count ?? 0)}
                confirmLabel={t.userTokenRevoke}
                cancelLabel={t.cancel}
                busyLabel={t.userTokenRevoking}
                onConfirm={async () => {
                  setFailure(null);
                  try {
                    await revokeUserToken(token.id);
                  } catch (error) {
                    setFailure(describe(error).text);
                  }
                  await load();
                  return true;
                }}
              />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function AccessSettings({
  t,
  locale,
  unauthorized,
  canRegenerate,
  userTokens,
  describe,
  onToken,
}: {
  t: Dict;
  locale: Locale;
  unauthorized: boolean;
  /** The server can replace its token (`features.token_regenerate`). */
  canRegenerate: boolean;
  /** v1.13: the admin's user tokens (`features.user_tokens`). */
  userTokens: boolean;
  describe: (error: unknown) => { text: string; detail: string };
  onToken: () => void;
}) {
  const [value, setValue] = useState("");
  const [held, setHeld] = useState(hasToken);
  const [saved, setSaved] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [fresh, setFresh] = useState<RegeneratedToken | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const regenerate = useRef<HTMLButtonElement>(null);
  // One status line: what this tab holds, with Sign out beside it; then making a
  // new token for the server; then one field with Save for using another token.
  const refused = unauthorized && !saved;
  const state = refused ? t.accessRefused : saved ? t.tokenSaved : held ? t.accessHeld : t.accessNone;
  const hint = refused ? t.accessRefusedHint : t.accessExplain;
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
              title={t.signOutTabHint}
              onClick={() => {
                // Another token signs in from the sign-in screen the page opens on.
                setToken("");
                location.reload();
              }}
            >
              {t.signOutTab}
            </button>
          )}
        </p>
        {canRegenerate && (
          <section class="access-section" aria-labelledby="regenerate-title">
            <h2 id="regenerate-title" class="group-title">
              {t.regenerateTitle}
            </h2>
            {fresh ? (
              <NewToken t={t} token={fresh} onDone={() => setFresh(null)} />
            ) : (
              <>
                <p class="field-hint">{t.regenerateText}</p>
                <div>
                  <button ref={regenerate} type="button" class="btn" aria-haspopup="dialog" aria-expanded={confirming} onClick={() => setConfirming(true)}>
                    <Icon name="ArrowCounterClockwise" size={14} />
                    {t.regenerateButton}
                  </button>
                  <ConfirmPopover
                    triggerLabel={t.regenerateButton}
                    title={t.regenerateConfirmTitle}
                    description={t.regenerateConfirmText}
                    confirmLabel={t.regenerateConfirm}
                    cancelLabel={t.cancel}
                    busyLabel={t.regenerating}
                    anchor={() => regenerate.current}
                    open={confirming}
                    onClose={() => setConfirming(false)}
                    onConfirm={async () => {
                      setFailure(null);
                      try {
                        const next = await regenerateToken();
                        // This tab signs in with the new token at once; every other one must again.
                        setToken(next.token);
                        setHeld(true);
                        setSaved(false);
                        setFresh(next);
                        onToken();
                        return true;
                      } catch (error) {
                        setFailure(describe(error).text);
                        return true;
                      }
                    }}
                  />
                </div>
                {failure && (
                  <p class="line-error" role="alert">
                    {t.regenerateFailed} · {failure}
                  </p>
                )}
              </>
            )}
          </section>
        )}
        {userTokens && <UserTokens t={t} locale={locale} describe={describe} />}
        {/* Only a refused or missing token needs a field here; switching tokens is Sign out. */}
        {(refused || !held) && (
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
        )}
      </div>
    </section>
  );
}

/** The new token, shown this once: copy it, or sign in on another device by its link or QR code. */
function NewToken({ t, token, heading: title, onDone }: { t: Dict; token: RegeneratedToken; heading?: string; onDone: () => void }) {
  const [copied, setCopied] = useState<"token" | "link" | null>(null);
  const heading = useRef<HTMLParagraphElement>(null);
  useEffect(() => heading.current?.focus(), []);
  // Another device opens the server's network address; without one, this page's own.
  const link = token.network_url ?? `${location.origin}/#${new URLSearchParams({ token: token.token }).toString()}`;
  const modules = useMemo(() => qrModules(link), [link]);
  const copy = (what: "token" | "link", text: string) =>
    void copyText(text).then((ok) => {
      if (ok) setCopied(what);
    });
  return (
    <div class="new-token">
      <p ref={heading} class="access-state" tabIndex={-1}>
        <Icon name="Check" size={15} />
        <span>{title ?? t.regenerated}</span>
      </p>
      <div class="field">
        <span class="field-label" id="new-token-label">
          {t.newTokenLabel}
        </span>
        <div class="input-group">
          <code class="token-value" aria-labelledby="new-token-label">
            {token.token}
          </code>
          <button type="button" class="btn" onClick={() => copy("token", token.token)}>
            <Icon name={copied === "token" ? "Check" : "Copy"} size={14} />
            {copied === "token" ? t.copied : t.copy}
          </button>
        </div>
        <p class="field-hint">{t.newTokenOnce}</p>
      </div>
      <div class="token-device">
        {modules && (
          <svg class="token-qr" viewBox={`0 0 ${modules.length + 8} ${modules.length + 8}`} role="img" aria-label={t.qrLabel} shape-rendering="crispEdges">
            <rect width="100%" height="100%" fill="#fff" />
            <path d={qrPath(modules)} fill="#000" />
          </svg>
        )}
        <div class="token-device-copy">
          <strong>{t.signInDevice}</strong>
          <p class="field-hint">{token.network_url ? t.signInDeviceNetwork : t.signInDeviceHere}</p>
          <code class="token-link">{link}</code>
          <div>
            <button type="button" class="btn btn-sm" onClick={() => copy("link", link)}>
              <Icon name={copied === "link" ? "Check" : "Copy"} size={14} />
              {copied === "link" ? t.copied : t.copyLink}
            </button>
          </div>
        </div>
      </div>
      {!token.persistent && <p class="conn-note">{token.startup_variable ? t.tokenUntilRestartVariable(token.startup_variable) : t.tokenUntilRestart}</p>}
      <div class="new-token-done">
        <button type="button" class="btn btn-primary" onClick={onDone}>
          {t.done}
        </button>
      </div>
    </div>
  );
}

/** Phones: the section list behind the tab bar's Settings. */
export function SettingsIndex({ t, version, sourceUrl, sections, user = false, updateCheck, onNavigate }: { t: Dict; version: VersionLine | null; sourceUrl: string; sections: Section[]; user?: boolean; updateCheck?: ComponentChildren; onNavigate: (route: Route) => void }) {
  const entries: Record<Section, [IconName, string]> = {
    appearance: ["Palette", t.navAppearanceShort],
    models: ["Cpu", t.navModels],
    access: ["Key", t.navAccess],
    config: ["FileCode", user ? t.navConfigUser : t.navConfig],
  };
  const entry = (section: Section, icon: IconName, label: string) => (
    <li key={section}>
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
        </span>
        <Icon name="CaretRight" size={14} />
      </a>
    </li>
  );
  const commit = version && commitLabel(version);
  return (
    <section class="settings-page" aria-labelledby="settings-index-title">
      <div class="page-head">
        <h1 id="settings-index-title">{t.settingsTitle}</h1>
      </div>
      <ul class="settings-index">
        {sections.map((section) => entry(section, ...entries[section]))}
      </ul>
      {/* not a list item: one quiet line under the list, version · commit · build time (the commit opens GitHub) */}
      {version && (
        <p class="build-line is-index" title={t.buildTitle(version.parts)}>
          <span class="build-copy">
            {version.parts.version}
            {commit && (
              <>
                <span aria-hidden="true"> · </span>
                <a class="build-commit" href={sourceUrl} target="_blank" rel="noopener noreferrer">
                  {commit}
                  <span class="sr-only"> GitHub {t.opensNewTab}</span>
                </a>
              </>
            )}
            {version.parts.date && (
              <>
                <span aria-hidden="true"> · </span>
                {version.parts.date}
              </>
            )}
          </span>
          {updateCheck}
        </p>
      )}
    </section>
  );
}
