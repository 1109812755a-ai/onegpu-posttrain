import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { WorkspaceNavigation } from "../src/WorkspaceNavigation.tsx";

const renderIcon = () => <svg aria-hidden="true" />;

describe("WorkspaceNavigation", () => {
  afterEach(() => cleanup());

  it("nests data pages beneath Data and marks the current child", () => {
    render(<WorkspaceNavigation page="policies" onNavigate={vi.fn()} renderIcon={renderIcon} />);
    const navigation = screen.getByRole("navigation", { name: "工作区主导航" });
    const dataModule = within(navigation).getByRole("button", { name: "数据" });
    const dataViews = navigation.querySelector("#workspace-subnav-data");

    expect(dataModule.getAttribute("aria-current")).toBe("location");
    expect(dataModule.getAttribute("aria-expanded")).toBe("true");
    expect(dataModule.getAttribute("aria-controls")).toBe("workspace-subnav-data");
    expect(dataViews?.hidden).toBe(false);
    expect(dataModule.parentElement?.lastElementChild).toBe(dataViews);
    expect(within(dataViews as HTMLElement).getByRole("button", { name: "规则对照" }).getAttribute("aria-current")).toBe("page");
  });

  it("nests Analysis pages and routes its parent to the default page", async () => {
    const user = userEvent.setup();
    const onNavigate = vi.fn();
    render(<WorkspaceNavigation page="targeted" onNavigate={onNavigate} renderIcon={renderIcon} />);
    const navigation = screen.getByRole("navigation", { name: "工作区主导航" });
    const analysisModule = within(navigation).getByRole("button", { name: "分析与改进" });
    const analysisViews = navigation.querySelector("#workspace-subnav-analysis");

    expect(analysisModule.getAttribute("aria-expanded")).toBe("true");
    expect(within(analysisViews as HTMLElement).getByRole("button", { name: "薄弱场景" }).getAttribute("aria-current")).toBe("page");
    await user.click(analysisModule);
    expect(onNavigate).toHaveBeenCalledWith("comparison");
  });

  it("renders training workspace views under the training module", () => {
    render(<WorkspaceNavigation page="launcher" onNavigate={vi.fn()} renderIcon={renderIcon} />);
    const navigation = screen.getByRole("navigation", { name: "工作区主导航" });
    const trainingModule = within(navigation).getByRole("button", { name: "训练与评测" });

    expect(trainingModule.getAttribute("aria-expanded")).toBe("true");
    expect(trainingModule.getAttribute("aria-controls")).toBe("workspace-subnav-training");
    const trainingViews = navigation.querySelector("#workspace-subnav-training") as HTMLElement;
    expect(within(trainingViews).getByRole("button", { name: "阶段工作台" })).toBeTruthy();
    expect(within(trainingViews).getByRole("button", { name: "模型体验" })).toBeTruthy();
    expect(trainingModule.getAttribute("aria-current")).toBe("location");
  });

  it("keeps keyboard focus visible and the expanded sidebar scrollable", () => {
    const css = readFileSync("src/style.css", "utf8");
    expect(css).toMatch(/button:focus-visible[\s\S]*?outline:\s*3px/);
    expect(css).toMatch(/\.sidebar\s*\{[^}]*overflow-y:\s*auto/s);
    expect(css).toMatch(/\.workspace-secondary-nav\[hidden\]\s*\{\s*display:\s*none/s);
    expect(css).toMatch(/\.workspace-mobile-subnav\s*\{[^}]*display:\s*flex;[^}]*overflow-x:\s*auto/s);
    expect(css).toMatch(/@media\s*\(max-width:\s*850px\)[\s\S]*?\.workspace-secondary-nav\s*\{\s*display:\s*none/s);
  });

  it("keeps the active parent and child keyboard reachable with programmatic state", async () => {
    const user = userEvent.setup();
    render(<WorkspaceNavigation page="policies" onNavigate={vi.fn()} renderIcon={renderIcon} />);
    const parent = screen.getByRole("button", { name: "数据" });
    const selectedChild = screen.getByRole("button", { name: "规则对照" });

    expect(parent.getAttribute("aria-label")).toBe("数据");
    expect(parent.getAttribute("aria-expanded")).toBe("true");
    expect(selectedChild.getAttribute("aria-current")).toBe("page");
    await user.tab();
    expect(document.activeElement).toBe(parent);
    await user.tab();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "数据集" }));
  });
});
