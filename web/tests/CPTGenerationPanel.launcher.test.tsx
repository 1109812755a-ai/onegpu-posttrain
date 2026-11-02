import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { CPTGenerationPanel } from "../src/CPTGenerationPanel";

afterEach(() => { cleanup(); window.sessionStorage.clear(); vi.unstubAllGlobals(); });

it("runs prepare → read-only preflight → explicit start → preview → explicit registration", async () => {
  const calls: Array<{ url: string; method: string; body?: any }> = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input); const method = init?.method || "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method, body });
    let value: any;
    if (url.endsWith("/capabilities")) value = { enabled: true, token: "local", profiles: [{ profile_id: "provider-local", model: "fixture-model" }] };
    else if (url.endsWith("/prepare")) value = { bundle_id: "cptgen-11111111111111111111" };
    else if (url.endsWith("/preflight")) value = { status: "preflight_ready", provider_calls: 0, contamination_scope: "example_fixture_only" };
    else if (url.endsWith("/start")) value = { execution_id: "cptgenrun-11111111111111111111", bundle_id: "cptgen-11111111111111111111", status: "queued" };
    else if (url.includes("/runs/") && url.endsWith("/preview")) value = { registered: false, candidate_count: 1, factuality: "not_verified", examples: [{ text: "candidate text" }] };
    else if (url.includes("/runs/") && url.endsWith("/register")) value = { registered_dataset_id: "ds-cpt-generated-test" };
    else if (url.includes("/runs/")) value = { execution_id: "cptgenrun-11111111111111111111", bundle_id: "cptgen-11111111111111111111", status: "completed", result: { candidate_count: 1 } };
    else throw new Error(`unexpected request: ${url}`);
    return { ok: true, json: async () => value };
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("confirm", vi.fn(() => true));
  render(<CPTGenerationPanel />);
  const user = userEvent.setup();
  await user.type(await screen.findByPlaceholderText("例如：细胞结构"), "细胞结构");
  await user.click(screen.getByRole("button", { name: "1. 准备方案" }));
  await user.click(await screen.findByRole("button", { name: "2. 只读预检" }));
  await user.click(await screen.findByRole("button", { name: "3. 确认并开始生成" }));
  await waitFor(() => expect(screen.getByText("CPT 生成任务：completed")).toBeTruthy());
  const replayButton = screen.getByRole("button", { name: "本方案已执行" }) as HTMLButtonElement;
  expect(replayButton.disabled).toBe(true);
  await user.click(replayButton);
  expect(calls.filter((call) => call.url.endsWith("/start")).length).toBe(1);
  await user.click(screen.getByRole("button", { name: "查看候选并审核" }));
  await user.click(await screen.findByRole("button", { name: "确认审核并登记" }));
  await waitFor(() => expect(screen.getByText("ds-cpt-generated-test")).toBeTruthy());
  expect(calls.find((call) => call.url.endsWith("/preflight"))?.body).toEqual({ bundle_id: "cptgen-11111111111111111111" });
  expect(calls.find((call) => call.url.endsWith("/start"))?.body.confirm_start).toBe(true);
  expect(calls.find((call) => call.url.endsWith("/register"))?.body.confirm_reviewed).toBe(true);
});
