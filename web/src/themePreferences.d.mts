export type ThemeMode = "light" | "dark" | "system";
export type EffectiveTheme = "light" | "dark";

export const THEME_STORAGE_KEY: string;
export function isThemeMode(value: unknown): value is ThemeMode;
export function readThemeMode(storage?: Pick<Storage, "getItem">): ThemeMode;
export function writeThemeMode(mode: ThemeMode, storage?: Pick<Storage, "setItem">): boolean;
export function systemPrefersDark(matchMedia?: typeof globalThis.matchMedia): boolean;
export function resolveTheme(mode: ThemeMode | string, prefersDark?: boolean): EffectiveTheme;
export function themeColorForTheme(theme: EffectiveTheme): string;
export function applyThemeToDocument(
  theme: EffectiveTheme,
  document?: Pick<Document, "documentElement" | "querySelector">,
): void;
export function subscribeSystemTheme(
  onChange: (prefersDark: boolean) => void,
  matchMedia?: typeof globalThis.matchMedia,
): () => void;
