// One dictionary per language; components receive the active one as `t`.
import { en, type Dict } from "./en.ts";
import type { Locale } from "./locale.ts";
import { zh } from "./zh.ts";

export type { Dict } from "./en.ts";
export type { Locale } from "./locale.ts";
export { detectLocale, storeLocale } from "./locale.ts";

export const dicts: Record<Locale, Dict> = { en, zh };
