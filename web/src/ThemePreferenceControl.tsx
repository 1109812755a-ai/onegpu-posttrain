import type { ChangeEvent } from "react";
import type { ThemeMode } from "./themePreferences.mjs";

type ThemePreferenceControlProps = {
  mode: ThemeMode;
  onChange: (mode: ThemeMode) => void;
};

export function ThemePreferenceControl({ mode, onChange }: ThemePreferenceControlProps) {
  const handleChange = (event: ChangeEvent<HTMLSelectElement>) => {
    onChange(event.target.value as ThemeMode);
  };

  return (
    <label className="theme-control">
      <span>主题</span>
      <select aria-label="外观模式" value={mode} onChange={handleChange}>
        <option value="light">浅色</option>
        <option value="dark">深色</option>
        <option value="system">跟随系统</option>
      </select>
    </label>
  );
}
