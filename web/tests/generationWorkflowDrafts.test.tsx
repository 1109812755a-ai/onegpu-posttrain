import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { CPTGenerationPanel } from "../src/CPTGenerationPanel";
import { PreferenceGenerationPanel } from "../src/PreferenceGenerationPanel";

afterEach(() => {
  cleanup(); window.localStorage.clear(); window.sessionStorage.clear(); vi.unstubAllGlobals();
});

it("persists CPT ordinary/targeted drafts, restores them after remount, and mode switches never start a request", async () => {
  const calls: Array<{ url: string; method: string }> = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), method: init?.method || "GET" });
    return { ok: true, json: async () => ({ enabled: true, token: "local", profiles: [{ profile_id: "p", model: "fixture" }] }) };
  }));
  const user = userEvent.setup();
  const first = render(<CPTGenerationPanel />);
  await user.type(await screen.findByPlaceholderText("例如：细胞结构"), "代数概念");
  await user.selectOptions(screen.getByLabelText("生成模式"), "targeted");
  expect(JSON.parse(localStorage.getItem("onegpu.cpt-generation-draft.v1") || "{}")).toMatchObject({
    mode: "targeted", topic: "代数概念",
  });
  first.unmount();
  render(<CPTGenerationPanel />);
  expect((await screen.findByLabelText("生成模式") as HTMLSelectElement).value).toBe("targeted");
  expect((screen.getByPlaceholderText("例如：细胞结构") as HTMLTextAreaElement).value).toBe("代数概念");
  expect(calls.every((call) => call.method === "GET" && call.url.endsWith("/capabilities"))).toBe(true);
});

it("persists Preference inputs and switching modes remains provider-call free", async () => {
  const calls: Array<{ url: string; method: string }> = [];
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    calls.push({ url: String(input), method: init?.method || "GET" });
    return { ok: true, json: async () => ({ enabled: true, token: "local", profiles: [{ profile_id: "p", model: "fixture" }], targeted_plans: [] }) };
  }));
  const user = userEvent.setup();
  const first = render(<PreferenceGenerationPanel />);
  await user.type(await screen.findByPlaceholderText("例如：解释一元一次方程的解法"), "分数比较");
  await user.selectOptions(screen.getByLabelText("生成方式"), "targeted");
  expect(JSON.parse(localStorage.getItem("onegpu.preference-generation-draft.v1") || "{}")).toMatchObject({
    mode: "targeted", prompts: "分数比较",
  });
  first.unmount();
  render(<PreferenceGenerationPanel />);
  const modeSelect = await screen.findByLabelText("生成方式") as HTMLSelectElement;
  expect(modeSelect.value).toBe("targeted");
  await userEvent.setup().selectOptions(modeSelect, "ordinary");
  expect((screen.getByPlaceholderText("例如：解释一元一次方程的解法") as HTMLTextAreaElement).value).toBe("分数比较");
  expect(calls.every((call) => call.method === "GET" && call.url.endsWith("/capabilities"))).toBe(true);
});
