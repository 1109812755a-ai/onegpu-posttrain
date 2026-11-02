import assert from "node:assert/strict";
import test from "node:test";

import {
  isThemeMode,
  readThemeMode,
  resolveTheme,
  subscribeSystemTheme,
  systemPrefersDark,
  THEME_STORAGE_KEY,
  applyThemeToDocument,
  themeColorForTheme,
  writeThemeMode,
} from "../src/themePreferences.mjs";

function memoryStorage(initial = {}) {
  const values = new Map(Object.entries(initial));
  return {
    getItem(key) { return values.get(key) ?? null; },
    setItem(key, value) { values.set(key, String(value)); },
  };
}

test("first visit defaults to light and valid explicit modes persist and restore", () => {
  const storage = memoryStorage();
  assert.equal(readThemeMode(storage), "light");
  for (const mode of ["light", "dark", "system"]) {
    assert.equal(writeThemeMode(mode, storage), true);
    assert.equal(readThemeMode(storage), mode);
  }
  assert.equal(storage.getItem(THEME_STORAGE_KEY), "system");
});

test("invalid saved modes and unavailable storage safely default to light", () => {
  assert.equal(readThemeMode(memoryStorage({ [THEME_STORAGE_KEY]: "neon" })), "light");
  assert.equal(writeThemeMode("neon", memoryStorage()), false);
  assert.equal(readThemeMode({ getItem() { throw new Error("blocked"); } }), "light");
  assert.equal(writeThemeMode("dark", { setItem() { throw new Error("blocked"); } }), false);
});

test("effective theme resolves explicit modes and system preference", () => {
  assert.equal(resolveTheme("light", true), "light");
  assert.equal(resolveTheme("dark", false), "dark");
  assert.equal(resolveTheme("system", true), "dark");
  assert.equal(resolveTheme("system", false), "light");
  assert.equal(resolveTheme("invalid", true), "light");
  assert.equal(isThemeMode("system"), true);
  assert.equal(isThemeMode("invalid"), false);
});

test("effective theme updates only document presentation and browser theme-color metadata", () => {
  const document = {
    documentElement: { dataset: {} },
    meta: { content: "#123d36" },
    querySelector(selector) {
      assert.equal(selector, 'meta[name="theme-color"]');
      return { setAttribute: (name, value) => { this.meta[name] = value; } };
    },
  };
  assert.equal(themeColorForTheme("dark"), "#111a18");
  assert.equal(themeColorForTheme("light"), "#123d36");
  applyThemeToDocument("dark", document);
  assert.deepEqual(document.documentElement.dataset, { theme: "dark" });
  assert.equal(document.meta.content, "#111a18");
  applyThemeToDocument("light", document);
  assert.deepEqual(document.documentElement.dataset, { theme: "light" });
  assert.equal(document.meta.content, "#123d36");
});

test("system preference is guarded and live changes can be subscribed and unsubscribed", () => {
  let listener;
  let removed = false;
  const query = {
    matches: true,
    addEventListener(_name, callback) { listener = callback; },
    removeEventListener(_name, callback) { removed = callback === listener; },
  };
  const matchMedia = (queryText) => {
    assert.equal(queryText, "(prefers-color-scheme: dark)");
    return query;
  };
  assert.equal(systemPrefersDark(matchMedia), true);
  assert.equal(systemPrefersDark(undefined), false);
  const changes = [];
  const unsubscribe = subscribeSystemTheme((value) => changes.push(value), matchMedia);
  listener({ matches: false });
  unsubscribe();
  assert.deepEqual(changes, [false]);
  assert.equal(removed, true);
});

test("system preference subscription supports legacy media-query listeners", () => {
  let listener;
  let removed = false;
  const unsubscribe = subscribeSystemTheme(() => {}, () => ({
    addListener(callback) { listener = callback; },
    removeListener(callback) { removed = callback === listener; },
  }));
  assert.equal(typeof listener, "function");
  unsubscribe();
  assert.equal(removed, true);
});
