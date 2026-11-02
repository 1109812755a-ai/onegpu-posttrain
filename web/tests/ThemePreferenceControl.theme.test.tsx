import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ThemePreferenceControl } from "../src/ThemePreferenceControl";

describe("ThemePreferenceControl", () => {
  afterEach(() => cleanup());

  it("exposes the three explicit choices and reports keyboard selection", async () => {
    const user = userEvent.setup();
    const onChange = vi.fn();
    render(<ThemePreferenceControl mode="light" onChange={onChange} />);
    const select = screen.getByRole("combobox", { name: "外观模式" });

    expect(screen.getByText("主题")).toBeTruthy();
    expect([...select.querySelectorAll("option")].map((option) => option.textContent)).toEqual([
      "浅色", "深色", "跟随系统",
    ]);
    expect(select.value).toBe("light");
    await user.tab();
    expect(document.activeElement).toBe(select);
    await user.selectOptions(select, "dark");
    expect(onChange).toHaveBeenCalledWith("dark");
  });

  it("reflects the shared mode when the parent page changes", () => {
    const { rerender } = render(<ThemePreferenceControl mode="light" onChange={vi.fn()} />);
    const select = screen.getByRole("combobox", { name: "外观模式" });
    rerender(<ThemePreferenceControl mode="system" onChange={vi.fn()} />);
    expect(select.value).toBe("system");
  });
});
