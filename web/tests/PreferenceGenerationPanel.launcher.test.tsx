import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { PreferenceGenerationPanel } from "../src/PreferenceGenerationPanel";

afterEach(() => { cleanup(); window.localStorage.clear(); vi.unstubAllGlobals(); });

it("connects ordinary Preference setup, zero-call preflight, explicit generation, review and registration", async () => {
  const calls: Array<{ url: string; method: string; body?: any }> = [];
  const candidates = [
    { candidate_id: "candidate-a", problem_id: "problem-a", raw_problem: "What is 1+1?", completion: "It is 2." },
    { candidate_id: "candidate-b", problem_id: "problem-a", raw_problem: "What is 1+1?", completion: "The answer is two." },
  ];
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input); const method = init?.method || "GET";
    const body = init?.body ? JSON.parse(String(init.body)) : undefined;
    calls.push({ url, method, body });
    let value: any;
    if (url.endsWith("/capabilities")) value = { enabled: true, token: "local", profiles: [{ profile_id: "provider-local", model: "fixture-model" }], targeted_plans: [] };
    else if (url.endsWith("/prepare")) value = { bundle_id: "plan-preference-ordinary-11111111111111111111" };
    else if (url.endsWith("/preflight")) value = { status: "preflight_ready", provider_calls: 0, request_budget: 2, candidate_budget: 2 };
    else if (url.endsWith("/start")) value = { execution_id: "prefgen-11111111111111111111", bundle_id: "plan-preference-ordinary-11111111111111111111", mode: "ordinary", status: "queued" };
    else if (url.endsWith("/preview")) value = { trainable: false, candidates };
    else if (url.endsWith("/review")) value = { execution_id: "prefgen-11111111111111111111", bundle_id: "plan-preference-ordinary-11111111111111111111", mode: "ordinary", status: "completed", review: { usable_pairs: 1 } };
    else if (url.endsWith("/register")) value = { execution_id: "prefgen-11111111111111111111", status: "completed", registration: { dataset_id: "ds-preference-human-reviewed", record_count: 1 } };
    else if (url.includes("/runs/")) value = { execution_id: "prefgen-11111111111111111111", bundle_id: "plan-preference-ordinary-11111111111111111111", mode: "ordinary", status: "completed", result: { candidate_count: 2 } };
    else throw new Error(`unexpected request: ${url}`);
    return { ok: true, json: async () => value };
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<PreferenceGenerationPanel />);
  const user = userEvent.setup();
  await user.type(await screen.findByPlaceholderText("例如：解释一元一次方程的解法"), "What is 1+1?");
  await user.click(screen.getByRole("button", { name: "1. 准备方案" }));
  await user.click(await screen.findByRole("button", { name: "2. 只读预检" }));
  await user.click(await screen.findByRole("button", { name: "3. 确认并开始生成" }));
  await waitFor(() => {
    const button = screen.getByRole("button", { name: /生成任务进行中|本方案已执行/ }) as HTMLButtonElement;
    expect(button.disabled).toBe(true);
  });
  await user.click(screen.getByRole("button", { name: /生成任务进行中|本方案已执行/ }));
  expect(calls.filter((call) => call.url.endsWith("/start")).length).toBe(1);
  await user.click(await screen.findByRole("button", { name: "4. 查看候选与排序证据" }));
  const chosenOptions = await screen.findAllByLabelText("Chosen");
  const rejectedOptions = await screen.findAllByLabelText("Rejected");
  await user.click(chosenOptions[0]); await user.click(rejectedOptions[1]);
  await user.type(screen.getByPlaceholderText("例如：本地用户"), "reviewer");
  await user.type(screen.getByLabelText("排序理由（至少 8 字）"), "答案更清晰且推导完整");
  await user.click(screen.getByRole("button", { name: "5. 保存人工排序（允许部分审核）" }));
  await user.click(await screen.findByRole("button", { name: "6. 确认审核并登记" }));
  await waitFor(() => expect(screen.getByText("ds-preference-human-reviewed")).toBeTruthy());
  expect(calls.find((call) => call.url.endsWith("/preflight"))?.body).toEqual({ bundle_id: "plan-preference-ordinary-11111111111111111111" });
  expect(calls.find((call) => call.url.endsWith("/start"))?.body.confirm_start).toBe(true);
  expect(calls.find((call) => call.url.endsWith("/review"))?.body.reviews[0]).toMatchObject({
    chosen_candidate_id: "candidate-a", rejected_candidate_id: "candidate-b",
  });
  expect(calls.find((call) => call.url.endsWith("/register"))?.body.confirm_reviewed).toBe(true);
});

it("keeps targeted Preference plan and train-seed identity through registration", async () => {
  const calls: Array<{ url: string; body?: any }> = [];
  const plan = { plan_id: "plan-targeted-12345678901234567890", plan_version: 2, scenario_id: "algebra", training_seed_dataset_id: "ds-seed-exact", request_budget: 2, candidate_budget: 2, max_output_tokens: 64 };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input); const body = init?.body ? JSON.parse(String(init.body)) : undefined; calls.push({ url, body });
    let value: any;
    if (url.endsWith("/capabilities")) value = { enabled: true, token: "local", profiles: [{ profile_id: "provider-local", model: "fixture-model" }], targeted_plans: [plan] };
    else if (url.endsWith("/prepare")) value = { bundle_id: "plan-preference-targeted-11111111111111111111" };
    else if (url.endsWith("/preflight")) value = { status: "preflight_ready", provider_calls: 0, request_budget: 2, candidate_budget: 2, parent_dataset_ref: { dataset_id: "ds-seed-exact", content_sha256: "a".repeat(64), role: "train_seed" } };
    else if (url.endsWith("/start")) value = { execution_id: "prefgen-22222222222222222222", bundle_id: "plan-preference-targeted-11111111111111111111", mode: "targeted", status: "queued" };
    else if (url.endsWith("/preview")) value = { trainable: false, candidates: [], pair_report: { usable_pairs: 1 } };
    else if (url.endsWith("/register")) value = { execution_id: "prefgen-22222222222222222222", status: "completed", registration: { dataset_id: "ds-preference-targeted", parent_dataset_refs: [{ dataset_id: "ds-seed-exact", content_sha256: "a".repeat(64), role: "train_seed" }] } };
    else if (url.includes("/runs/")) value = { execution_id: "prefgen-22222222222222222222", bundle_id: "plan-preference-targeted-11111111111111111111", mode: "targeted", status: "completed", result: { candidate_count: 2 } };
    else throw new Error(`unexpected request: ${url}`);
    return { ok: true, json: async () => value };
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<PreferenceGenerationPanel />);
  const user = userEvent.setup();
  await user.selectOptions(await screen.findByLabelText("生成方式"), "targeted");
  await user.click(screen.getByRole("button", { name: "1. 准备方案" }));
  await user.click(await screen.findByRole("button", { name: "2. 只读预检" }));
  await user.click(await screen.findByRole("button", { name: "3. 确认并开始生成" }));
  await user.click(await screen.findByRole("button", { name: "4. 查看候选与排序证据" }));
  await user.click(await screen.findByRole("button", { name: "6. 确认审核并登记" }));
  await waitFor(() => expect(screen.getByText("ds-preference-targeted")).toBeTruthy());
  expect(calls.find((call) => call.url.endsWith("/prepare"))?.body).toMatchObject({ mode: "targeted", stage_plan_id: plan.plan_id, plan_version: 2 });
  expect(calls.find((call) => call.url.endsWith("/register"))?.body.confirm_reviewed).toBe(true);
});
