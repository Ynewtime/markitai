// One dictionary per language; components receive the active one as `t`.
// English is built in. Chinese and Japanese are separate files
// (/ui/i18n-zh.js, /ui/i18n-ja.js) loaded before they are first shown, so a
// visitor downloads only the language they read.
import { en, type Dict } from "./en.ts";
import { registerMessages } from "./errors.ts";
import type { Locale } from "./locale.ts";

export type { Dict } from "./en.ts";
export type { Locale } from "./locale.ts";
export { detectLocale, storeLocale } from "./locale.ts";

/** Until a pack has loaded, its language reads English. */
export const dicts: Record<Locale, Dict> = { en, zh: en, ja: en };

const packs: Partial<Record<Locale, Promise<void>>> = {};

/** Load a language's pack once; a failed load leaves English and may be tried again. */
export function loadLocale(locale: Locale): Promise<void> {
  if (locale === "en") return Promise.resolve();
  packs[locale] ??= (locale === "zh" ? import("/ui/i18n-zh.js") : import("/ui/i18n-ja.js")).then(
    (pack) => {
      dicts[locale] = pack.dict;
      registerMessages(locale, pack.messages);
    },
    (error: unknown) => {
      delete packs[locale];
      throw error;
    },
  );
  return packs[locale] as Promise<void>;
}
