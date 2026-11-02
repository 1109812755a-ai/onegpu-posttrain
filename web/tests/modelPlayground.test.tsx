import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { readFileSync } from "node:fs";
import { ModelPlayground } from "../src/ModelPlayground";

const modelRows = ["m1", "m2", "m3", "m4"].map((model_id, index) => ({
  model_id, kind: "merged", stage: index ? "dpo" : "sft", digest: `${index}`.repeat(64),
  source_run_id: `run-${index}`, produced_by_run_id: `merge-${index}`,
  training_completed_at: index === 1 ? "2026-10-31T10:00:00Z" : index === 0 ? "2026-10-29T10:00:00Z" : null,
  display_name: null, available: true,
}));
const defaults = { max_new_tokens: 256, temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05, enable_thinking: false };

function mockApi(compareResult?: unknown, singleJobStatus: "completed" | "failed" = "completed", rows = modelRows) {
  let singleParameters = defaults;
  let comparisonParameters = defaults;
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/api/model-playground/models")) return Response.json({ status: "ready", detail: null, models: rows });
    if (url.includes("/api/model-playground/models/") && url.endsWith("/display-name") && init?.method === "PUT") {
      return Response.json({ model_id: url.split("/").at(-2), display_name: JSON.parse(String(init.body)).display_name });
    }
    if (url.includes("/api/model-playground/models/") && url.endsWith("/display-name") && init?.method === "DELETE") return new Response(null, { status: 204 });
    if (url.endsWith("/api/launcher/capabilities")) return Response.json({ enabled: true, token: "local-token" });
    if (url.endsWith("/api/model-playground/sessions") && init?.method === "POST") return Response.json({ session_id: "a".repeat(32), model_id: "m1", status: "loading" });
    if (url.endsWith(`/api/model-playground/sessions/${"a".repeat(32)}`)) return Response.json({ session_id: "a".repeat(32), model_id: "m1", status: "ready" });
    if (url.endsWith("/messages")) { singleParameters = JSON.parse(String(init?.body)).parameters; return Response.json({ job_id: "b".repeat(32), status: "queued" }); }
    if (url.endsWith("/api/model-playground/comparisons")) { comparisonParameters = JSON.parse(String(init?.body)).parameters; return Response.json({ job_id: "c".repeat(32), status: "queued" }); }
    if (url.endsWith(`/api/model-playground/jobs/${"b".repeat(32)}`)) return Response.json({ job_id: "b".repeat(32), status: singleJobStatus, detail: "本地模型生成失败", result: singleJobStatus === "completed" ? { text: "single answer", model_id: "m1", parameters: { ...singleParameters, do_sample: singleParameters.temperature > 0, top_p_active: singleParameters.temperature > 0, top_k_active: singleParameters.temperature > 0 && singleParameters.top_k > 0, thinking_mode_supported: true } } : null });
    if (url.endsWith(`/api/model-playground/jobs/${"c".repeat(32)}`)) return Response.json({ job_id: "c".repeat(32), status: "partial", result: compareResult || { status: "partial", results: [
      { status: "completed", model_id: "m1", text: "answer one", parameters: { ...comparisonParameters, do_sample: comparisonParameters.temperature > 0, top_p_active: comparisonParameters.temperature > 0, top_k_active: comparisonParameters.temperature > 0 && comparisonParameters.top_k > 0, thinking_mode_supported: true } }, { status: "failed", model_id: "m2", detail: "failed" },
    ] } });
    return Response.json({ detail: `unexpected ${url}` }, { status: 404 });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe("ModelPlayground", () => {
  it("keeps the form and comparison layout responsive on narrow and desktop screens", () => {
    const css = readFileSync("src/style.css", "utf8");
    expect(css).toMatch(/\.model-playground-columns\s*\{\s*display:\s*grid;\s*grid-template-columns:\s*repeat\(auto-fit/s);
    expect(css).toMatch(/@media\s*\(max-width:\s*600px\)[\s\S]*?\.model-playground-page/);
    expect(css).toMatch(/\.model-playground-page[^}]*button:focus-visible|\.model-playground-page button:focus-visible/);
  });

  it("supports keyboard-accessible single-model interaction with explicit qualitative label", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    render(<ModelPlayground onBack={() => undefined} />);
    const modelSelect = await screen.findByLabelText("体验模型");
    expect((screen.getByLabelText("最大新 Token") as HTMLInputElement).value).toBe("256");
    expect((screen.getByLabelText("Temperature") as HTMLInputElement).value).toBe("0.7");
    await user.click(screen.getByText(/高级生成参数/));
    expect((screen.getByLabelText("Top-p") as HTMLInputElement).value).toBe("0.8");
    expect((screen.getByLabelText("Top-k（0 为关闭）") as HTMLInputElement).value).toBe("20");
    expect((screen.getByLabelText("重复惩罚") as HTMLInputElement).value).toBe("1.05");
    await user.clear(screen.getByLabelText("Top-p"));
    await user.type(screen.getByLabelText("Top-p"), "0.6");
    expect((screen.getByLabelText("Top-p") as HTMLInputElement).value).toBe("0.6");
    await user.click(screen.getByRole("button", { name: "恢复推荐默认值" }));
    expect((screen.getByLabelText("Top-p") as HTMLInputElement).value).toBe("0.8");
    await user.selectOptions(modelSelect, "m1");
    await user.click(screen.getByRole("button", { name: "加载模型开始体验" }));
    await screen.findByText("模型已就绪，可以开始对话");
    const textbox = screen.getByLabelText("你的问题");
    await user.type(textbox, "hello");
    await user.click(screen.getByRole("button", { name: "发送消息" }));
    expect(await screen.findByText("single answer")).toBeTruthy();
    const request = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/messages"));
    expect(JSON.parse(String(request?.[1]?.body)).parameters).toEqual(defaults);
    expect(screen.getByText(/本次参数：max_new_tokens 256/)).toBeTruthy();
    expect(screen.getByText("定性输出 · 未经独立验证")).toBeTruthy();
    expect(screen.getByRole("button", { name: "结束会话并卸载模型" })).toBeTruthy();
  });

  it("supports bounded same-prompt comparison and clearly retains partial results", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    render(<ModelPlayground onBack={() => undefined} />);
    await user.click(await screen.findByRole("button", { name: "同题模型对照" }));
    await user.click(screen.getByLabelText(/SFT · 2026-10-29 · run-0/));
    await user.click(screen.getByLabelText(/DPO · 2026-10-31 · run-1/));
    await user.click(screen.getByText(/高级生成参数/));
    await user.clear(screen.getByLabelText("Temperature"));
    await user.type(screen.getByLabelText("Temperature"), "0");
    expect((screen.getByLabelText("Top-p") as HTMLInputElement).disabled).toBe(true);
    await user.type(screen.getByLabelText("你的问题"), "same question");
    await user.click(screen.getByRole("button", { name: "对照 2 个模型并生成" }));
    expect(await screen.findByText("answer one")).toBeTruthy();
    expect(screen.getByText("体验失败：failed")).toBeTruthy();
    expect(screen.getByText("对照未全部完成 · 已保留可用结果")).toBeTruthy();
    const request = fetchMock.mock.calls.find(([url]) => String(url).endsWith("/api/model-playground/comparisons"));
    expect(JSON.parse(String(request?.[1]?.body))).toMatchObject({ model_ids: ["m1", "m2"], message: "same question" });
    expect(JSON.parse(String(request?.[1]?.body)).parameters).toEqual({ ...defaults, temperature: 0 });
    expect(screen.getByText(/采样关闭；Top-p\/Top-k 未生效/)).toBeTruthy();
    expect(screen.getByLabelText("你的问题").getAttribute("maxLength")).toBe("16000");
  });

  it("clears a released session after generation failure so the user can reload it", async () => {
    mockApi(undefined, "failed");
    const user = userEvent.setup();
    render(<ModelPlayground onBack={() => undefined} />);
    await user.selectOptions(await screen.findByLabelText("体验模型"), "m1");
    await user.click(screen.getByRole("button", { name: "加载模型开始体验" }));
    await screen.findByText("模型已就绪，可以开始对话");
    await user.type(screen.getByLabelText("你的问题"), "hello");
    await user.click(screen.getByRole("button", { name: "发送消息" }));
    expect((await screen.findByRole("alert")).textContent).toContain("本地模型生成失败");
    expect(await screen.findByRole("button", { name: "加载模型开始体验" })).toBeTruthy();
  });

  it("defaults to the newest completed training model without loading it and preserves explicit Run handoff", async () => {
    const fetchMock = mockApi();
    const { unmount } = render(<ModelPlayground onBack={() => undefined} />);
    const select = await screen.findByLabelText("体验模型") as HTMLSelectElement;
    expect(select.value).toBe("m2");
    expect(screen.getByText("训练完成：2026-10-31T10:00:00Z")).toBeTruthy();
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith("/sessions") && init?.method === "POST")).toBe(false);

    unmount();
    render(<ModelPlayground onBack={() => undefined} initialModelId="m1" />);
    await screen.findByLabelText("体验模型");
    expect((screen.getByLabelText("体验模型") as HTMLSelectElement).value).toBe("m1");
  });

  it("keeps an unavailable Run handoff selected and asks for intentional model choice", async () => {
    const rows = modelRows.map((row) => row.model_id === "m1" ? { ...row, available: false, unavailable_reason: "权重不可用" } : row);
    const fetchMock = mockApi(undefined, "completed", rows);
    render(<ModelPlayground onBack={() => undefined} initialModelId="m1" />);
    const select = await screen.findByLabelText("体验模型") as HTMLSelectElement;
    expect(select.value).toBe("m1");
    expect((await screen.findByRole("status")).textContent).toContain("Run 指定的模型当前不可体验");
    expect((screen.getByRole("button", { name: "加载模型开始体验" }) as HTMLButtonElement).disabled).toBe(true);
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith("/sessions") && init?.method === "POST")).toBe(false);
  });

  it("lets users save and clear a display name with keyboard-operable controls", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    render(<ModelPlayground onBack={() => undefined} />);
    const select = await screen.findByLabelText("体验模型");
    await user.selectOptions(select, "m1");
    await user.click(screen.getByRole("button", { name: "设置用户显示名" }));
    const input = screen.getByLabelText("用户显示名") as HTMLInputElement;
    await user.type(input, "SFT 数学助手");
    await user.tab();
    expect(document.activeElement).toBe(screen.getByRole("button", { name: "保存名称" }));
    await user.keyboard("{Enter}");
    expect(await screen.findByText("SFT 数学助手")).toBeTruthy();
    expect(screen.getByText("显示名已保存")).toBeTruthy();
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith("/m1/display-name") && init?.method === "PUT")).toBe(true);
    await user.click(screen.getByRole("button", { name: "修改用户显示名" }));
    const renamedInput = screen.getByLabelText("用户显示名");
    await user.clear(renamedInput);
    await user.type(renamedInput, "SFT 数学模型");
    await user.click(screen.getByRole("button", { name: "保存名称" }));
    expect(await screen.findByText("SFT 数学模型")).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "清除" }));
    expect(await screen.findByText("已清除显示名")).toBeTruthy();
    expect(fetchMock.mock.calls.some(([url, init]) => String(url).endsWith("/m1/display-name") && init?.method === "DELETE")).toBe(true);
  });

  it("shows display-name validation errors before making a request", async () => {
    const fetchMock = mockApi();
    const user = userEvent.setup();
    render(<ModelPlayground onBack={() => undefined} />);
    await user.selectOptions(await screen.findByLabelText("体验模型"), "m1");
    await user.click(screen.getByRole("button", { name: "设置用户显示名" }));
    await user.click(screen.getByRole("button", { name: "保存名称" }));
    expect((await screen.findByRole("alert")).textContent).toContain("显示名不能为空");
    expect(fetchMock.mock.calls.some(([url]) => String(url).endsWith("/m1/display-name"))).toBe(false);
  });
});
