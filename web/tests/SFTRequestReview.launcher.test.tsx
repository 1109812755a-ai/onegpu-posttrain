import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { SFTGenerationPanel } from "../src/SFTGenerationPanel";

afterEach(() => { cleanup(); window.localStorage.clear(); vi.unstubAllGlobals(); });

it("restores an unknown request and requires explicit risk acknowledgement without starting generation", async () => {
  window.localStorage.setItem("onegpu.sft-generation.execution", "sftgen-11111111111111111111");
  const attempt = "a".repeat(32);
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    let body: unknown;
    if (url.endsWith("/capabilities")) body = { enabled: true, token: "local" };
    else if (url.endsWith("/readiness")) body = { datasets: [], profiles: [], readiness: [] };
    else if (url.includes("/runs/")) body = { execution_id: "sftgen-11111111111111111111", bundle_id: "plan-sft-prompt-abc", status: "failed" };
    else if (url.includes("/request-review/")) body = { unresolved: init?.method === "POST" ? [] : [{ attempt_id: attempt, execution_id: "sftgen-11111111111111111111", state: "uncertain" }] };
    else throw new Error(`unexpected request: ${url}`);
    return { ok: true, json: async () => body };
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("confirm", vi.fn(() => true));
  render(<SFTGenerationPanel />);
  expect(await screen.findByText("存在结果未知的模型请求，暂不能重新生成")).toBeTruthy();
  await userEvent.click(screen.getByRole("button", { name: "已核查并确认风险" }));
  await waitFor(() => expect(screen.queryByText("存在结果未知的模型请求，暂不能重新生成")).toBeNull());
  expect(fetchMock.mock.calls.some(([url, init]) => String(url).includes("/request-review/") && init?.method === "POST")).toBe(true);
  expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/start"))).toBe(false);
});

it("shows an interrupted execution as stopped and never presents it as still generating", async () => {
  window.localStorage.setItem("onegpu.sft-generation.execution", "sftgen-22222222222222222222");
  vi.stubGlobal("fetch", vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    let body: unknown;
    if (url.endsWith("/capabilities")) body = { enabled: true, token: "local" };
    else if (url.endsWith("/readiness")) body = { datasets: [], profiles: [], readiness: [] };
    else if (url.includes("/runs/")) body = {
      execution_id: "sftgen-22222222222222222222", bundle_id: "plan-sft-prompt-def",
      status: "interrupted", detail: "服务重启时任务仍在运行；未自动重启 provider 请求",
    };
    else if (url.includes("/request-review/")) body = { unresolved: [] };
    else throw new Error(`unexpected request: ${url}`);
    return { ok: true, json: async () => body };
  }));
  render(<SFTGenerationPanel />);
  expect(await screen.findByText("服务中断，未自动重试")).toBeTruthy();
  expect(screen.queryByText("正在生成")).toBeNull();
});
