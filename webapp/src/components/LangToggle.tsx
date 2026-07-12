import type { Locale } from "../i18n";

const LOCALES: { value: Locale; label: string }[] = [
  { value: "en", label: "EN" },
  { value: "zh", label: "中" },
];

/** Header EN/中 pill — same mono language as the theme toggle. */
export function LangToggle({
  label,
  locale,
  onLocale,
}: {
  label: string;
  locale: Locale;
  onLocale: (l: Locale) => void;
}) {
  return (
    <div className="langtoggle" role="group" aria-label={label}>
      {LOCALES.map(({ value, label: text }) => (
        <button
          key={value}
          type="button"
          lang={value === "zh" ? "zh-CN" : "en"}
          className={value === locale ? "on" : undefined}
          aria-pressed={value === locale}
          onClick={() => onLocale(value)}
        >
          {text}
        </button>
      ))}
    </div>
  );
}
