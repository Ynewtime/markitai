import { useEffect, useRef, useState } from "react";
import type { Dict } from "../i18n";
import { MonitorIcon, MoonIcon, SunIcon } from "./icons";

type Mode = "auto" | "light" | "dark";

const MODES: Mode[] = ["auto", "light", "dark"];
const STORAGE_KEY = "markitai-theme";

function initialMode(): Mode {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    if (v === "light" || v === "dark") return v;
  } catch {
    /* localStorage unavailable */
  }
  return "auto";
}

/** Vercel-style icon pill: monitor / sun / moon, the checked one on a round
 * accent backing. Real radiogroup: one tab stop, arrows move and select,
 * "auto" removes data-theme so the prefers-color-scheme media query decides. */
export function ThemeToggle({ t, label }: { t: Dict; label: string }) {
  const [mode, setMode] = useState<Mode>(initialMode);
  const btnRefs = useRef<(HTMLButtonElement | null)[]>([]);

  useEffect(() => {
    const root = document.documentElement;
    if (mode === "auto") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", mode);
    try {
      if (mode === "auto") localStorage.removeItem(STORAGE_KEY);
      else localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      /* localStorage unavailable */
    }
  }, [mode]);

  const titles: Record<Mode, string> = {
    auto: t.themeAuto,
    light: t.themeLight,
    dark: t.themeDark,
  };
  const icons: Record<Mode, (props: { size?: number }) => React.JSX.Element> = {
    auto: MonitorIcon,
    light: SunIcon,
    dark: MoonIcon,
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    const idx = MODES.indexOf(mode);
    let next: number;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (idx + 1) % MODES.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp")
      next = (idx + MODES.length - 1) % MODES.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = MODES.length - 1;
    else return;
    e.preventDefault();
    const m = MODES[next];
    if (m === undefined) return;
    setMode(m);
    btnRefs.current[next]?.focus();
  };

  return (
    <div className="themetoggle" role="radiogroup" aria-label={label} onKeyDown={onKeyDown}>
      {MODES.map((m, i) => {
        const Icon = icons[m];
        return (
          <button
            key={m}
            ref={(el) => {
              btnRefs.current[i] = el;
            }}
            type="button"
            role="radio"
            aria-checked={m === mode}
            aria-label={titles[m]}
            title={titles[m]}
            tabIndex={m === mode ? 0 : -1}
            className={m === mode ? "on" : undefined}
            onClick={() => setMode(m)}
          >
            <Icon size={14} />
          </button>
        );
      })}
    </div>
  );
}
