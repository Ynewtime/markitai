// The frame of every view: a left sidebar with the two work views, the
// settings sections (Appearance for everyone; Models for the admin and, v1.15,
// a user token's own; Access token and Edit config for the admin token only) and the external links. It collapses to a 60px icon rail
// (each entry keeps its name for assistive technology and shows it as a
// tooltip); the choice is remembered in this browser. On phones the sidebar
// gives way to a top bar and a bottom tab bar.
import type { ComponentChildren } from "preact";
import { useCallback, useEffect, useState } from "preact/hooks";
import { VersionText } from "./version-text.tsx";
import type { VersionLine } from "../lib/build.ts";
import type { Dict } from "../i18n/index.ts";
import { ownClick, routePath, type Route, type Section } from "../lib/route.ts";
import { Icon, Logo } from "./icons.tsx";
import type { IconName } from "./icon-paths.ts";

export const SIDEBAR_KEY = "markitai.sidebar";

function storedCollapsed(): boolean {
  try {
    return localStorage.getItem(SIDEBAR_KEY) === "collapsed";
  } catch {
    return false;
  }
}

export function useSidebar(): [boolean, () => void] {
  const [collapsed, setCollapsed] = useState(storedCollapsed);
  useEffect(() => {
    document.documentElement.toggleAttribute("data-rail", collapsed);
    try {
      if (collapsed) localStorage.setItem(SIDEBAR_KEY, "collapsed");
      else localStorage.removeItem(SIDEBAR_KEY);
    } catch {
      /* The choice lasts for this page only. */
    }
  }, [collapsed]);
  return [collapsed, useCallback(() => setCollapsed((value) => !value), [])];
}

function NavLink({
  to,
  icon,
  label,
  current,
  onNavigate,
  children,
}: {
  to: Route;
  icon: IconName;
  label: string;
  current: boolean;
  onNavigate: (route: Route) => void;
  children?: ComponentChildren;
}) {
  return (
    <a
      href={routePath(to)}
      class={current ? "nav-link is-on" : "nav-link"}
      aria-current={current ? "page" : undefined}
      data-tip={label}
      onClick={(event) => {
        if (!ownClick(event)) return;
        event.preventDefault();
        onNavigate(to);
      }}
    >
      <Icon name={icon} size={17} />
      <span class="nav-text">{label}</span>
      {children}
    </a>
  );
}

function External({ href, label, icon, note }: { href: string; label: string; icon: IconName; note: string }) {
  return (
    <a class="nav-link is-external" href={href} target="_blank" rel="noopener noreferrer" data-tip={label}>
      <Icon name={icon} size={17} />
      <span class="nav-text">{label}</span>
      <span class="sr-only"> {note}</span>
    </a>
  );
}

export interface ShellProps {
  t: Dict;
  route: Route;
  version: VersionLine | null;
  docsUrl: string;
  sourceUrl: string;
  /** Items converting or waiting, for the live mark. */
  active: number;
  total: number;
  collapsed: boolean;
  onToggle: () => void;
  onNavigate: (route: Route) => void;
  /** The settings pages this token may open, in sidebar order. */
  sections: Section[];
  children: ComponentChildren;
}

const SECTION_ICONS: Record<Section, IconName> = { appearance: "Palette", models: "Cpu", access: "Key", config: "FileCode" };

export function Shell({ t, route, version, docsUrl, sourceUrl, active, total, collapsed, onToggle, onNavigate, sections, children }: ShellProps) {
  const section = route.view === "settings" ? route.section ?? sections[0] ?? "appearance" : null;
  const sectionLabels: Record<Section, string> = { appearance: t.navAppearanceShort, models: t.navModels, access: t.navAccess, config: t.navConfig };
  const settingsLink = (to: Section, icon: IconName, label: string) => (
    <NavLink to={{ view: "settings", section: to }} icon={icon} label={label} current={section === to} onNavigate={onNavigate} />
  );
  const live = active > 0 && (
    <span class="live-dot" role="img" aria-label={t.navLive(active)}>
      <span />
    </span>
  );
  const count = total > 0 && <span class="nav-count">{total}</span>;
  return (
    <div class="shell">
      <a class="skip-link" href="#main">
        {t.skipToContent}
      </a>
      <aside id="sidebar" class="sidebar" aria-label={t.sidebarAria}>
        <div class="sidebar-head">
          <a
            class="brand"
            href="/"
            aria-label={t.homeAria}
            data-tip="Markitai"
            onClick={(event) => {
              if (!ownClick(event)) return;
              event.preventDefault();
              onNavigate({ view: "convert" });
            }}
          >
            <Logo size={26} />
            <span class="brand-name">Markitai</span>
          </a>
          <button
            type="button"
            class="rail-toggle"
            aria-controls="sidebar"
            aria-expanded={!collapsed}
            aria-label={collapsed ? t.expandSidebar : t.collapseSidebar}
            data-tip={collapsed ? t.expandSidebar : t.collapseSidebar}
            onClick={onToggle}
          >
            <Icon name="SidebarSimple" size={17} />
          </button>
        </div>
        <nav class="nav-group" aria-label={t.workNavAria}>
          <NavLink to={{ view: "convert" }} icon="Plus" label={t.navConvert} current={route.view === "convert"} onNavigate={onNavigate} />
          <NavLink to={{ view: "jobs" }} icon="Files" label={t.conversions} current={route.view === "jobs"} onNavigate={onNavigate}>
            {live}
            {count}
          </NavLink>
        </nav>
        <nav class="nav-group" aria-labelledby="nav-settings-title">
          <p id="nav-settings-title" class="nav-title">
            {t.settingsTitle}
          </p>
          {sections.map((entry) => settingsLink(entry, SECTION_ICONS[entry], sectionLabels[entry]))}
        </nav>
        <div class="sidebar-foot">
          <External href={docsUrl} label={t.docsLabel} icon="BookOpen" note={t.opensNewTab} />
          <External href={sourceUrl} label="GitHub" icon="GithubLogo" note={t.opensNewTab} />
          {version && <p class="version" title={t.buildTitle(version.parts)}><VersionText text={version.text} /></p>}
        </div>
      </aside>
      <div class="main-col">
        <header class="phone-top">
          <a
            class="brand"
            href="/"
            aria-label={t.homeAria}
            onClick={(event) => {
              if (!ownClick(event)) return;
              event.preventDefault();
              onNavigate({ view: "convert" });
            }}
          >
            <Logo size={26} />
            <span class="brand-name">Markitai</span>
          </a>
          {version && <span class="version" title={t.buildTitle(version.parts)}>{version.short}</span>}
        </header>
        <main id="main" tabIndex={-1}>
          {children}
        </main>
      </div>
      <nav class="tabbar" aria-label={t.workNavAria}>
        <NavLink to={{ view: "convert" }} icon="Plus" label={t.navConvert} current={route.view === "convert"} onNavigate={onNavigate} />
        <NavLink to={{ view: "jobs" }} icon="Files" label={t.conversions} current={route.view === "jobs"} onNavigate={onNavigate}>
          {live}
        </NavLink>
        <NavLink to={{ view: "settings", section: null }} icon="Gear" label={t.settingsTitle} current={route.view === "settings"} onNavigate={onNavigate} />
      </nav>
    </div>
  );
}
