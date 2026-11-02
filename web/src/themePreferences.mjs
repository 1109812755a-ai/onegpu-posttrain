export const THEME_STORAGE_KEY = "onegpu-posttrain-theme-v1";
const MODES = new Set(["light", "dark", "system"]);

function getStorage(storage) {
  if (storage) return storage;
  try {
    return globalThis.localStorage;
  } catch {
    return null;
  }
}

export function isThemeMode(value) {
  return MODES.has(value);
}

export function readThemeMode(storage) {
  try {
    const value = getStorage(storage)?.getItem(THEME_STORAGE_KEY);
    return isThemeMode(value) ? value : "light";
  } catch {
    return "light";
  }
}

export function writeThemeMode(mode, storage) {
  if (!isThemeMode(mode)) return false;
  try {
    getStorage(storage)?.setItem(THEME_STORAGE_KEY, mode);
    return true;
  } catch {
    return false;
  }
}

export function systemPrefersDark(matchMedia = globalThis.matchMedia) {
  try {
    return typeof matchMedia === "function"
      ? Boolean(matchMedia("(prefers-color-scheme: dark)").matches)
      : false;
  } catch {
    return false;
  }
}

export function resolveTheme(mode, prefersDark = false) {
  return mode === "dark" || (mode === "system" && prefersDark) ? "dark" : "light";
}

export function themeColorForTheme(theme) {
  return theme === "dark" ? "#111a18" : "#123d36";
}

export function applyThemeToDocument(theme, document = globalThis.document) {
  if (!document?.documentElement) return;
  document.documentElement.dataset.theme = theme;
  document.querySelector?.('meta[name="theme-color"]')?.setAttribute(
    "content",
    themeColorForTheme(theme),
  );
}

export function subscribeSystemTheme(onChange, matchMedia = globalThis.matchMedia) {
  if (typeof matchMedia !== "function") return () => {};
  let query;
  try {
    query = matchMedia("(prefers-color-scheme: dark)");
  } catch {
    return () => {};
  }
  const listener = (event) => onChange(Boolean(event.matches));
  if (typeof query.addEventListener === "function") {
    query.addEventListener("change", listener);
    return () => query.removeEventListener?.("change", listener);
  }
  if (typeof query.addListener === "function") {
    query.addListener(listener);
    return () => query.removeListener?.(listener);
  }
  return () => {};
}
