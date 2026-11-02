import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, it, vi } from "vitest";
import { GRPOGenerationPanel } from "../src/GRPOGenerationPanel";

afterEach(() => { cleanup(); window.localStorage.clear(); vi.unstubAllGlobals(); });

it("ordinary GRPO generation is explicit, resumable, preview-only, and never registers missing references", async () => {
  const calls: Array<{ url: string; body?: any }> = [];
  const candidate = { candidate_id: "grpo-candidate-1", source_record_id: "ordinary-topic-1", raw_problem: "Generated algebra problem", reference_status: "missing" };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input); const body = init?.body ? JSON.parse(String(init.body)) : undefined; calls.push({ url, body });
    let value: any;
    if (url.endsWith("/capabilities")) value = { enabled: true, token: "local", profiles: [{ profile_id: "provider-local", model: "fixture" }], targeted_plans: [], warning: "Candidates are not trainable." };
    else if (url.endsWith("/prepare")) value = { bundle_id: "plan-grpo-ordinary-11111111111111111111" };
    else if (url.endsWith("/preflight")) value = { provider_calls: 0, trainable: false, blockers: ["reference missing"] };
    else if (url.endsWith("/start")) value = { execution_id: "grpo-gen-11111111111111111111", status: "queued", mode: "ordinary" };
    else if (url.endsWith("/preview")) value = { trainable: false, registration_allowed: false, reference_status: "missing_until_independently_bound", candidates: [candidate] };
    else if (url.includes("/runs/")) value = { execution_id: "grpo-gen-11111111111111111111", status: "completed", mode: "ordinary", result: { candidate_count: 1 } };
    else throw new Error(`unexpected request: ${url}`);
    return { ok: true, json: async () => value };
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("confirm", vi.fn(() => true));
  render(<GRPOGenerationPanel />);
  const user = userEvent.setup();
  await user.type(await screen.findByLabelText(/主题\/种子提示/), "linear equation");
  await user.click(screen.getByRole("button", { name: "准备方案并预检" }));
  await waitFor(() => expect(calls.some((call) => call.url.endsWith("/preflight"))).toBe(true));
  expect(calls.find((call) => call.url.endsWith("/preflight"))?.body).toMatchObject({ bundle_id: "plan-grpo-ordinary-11111111111111111111" });
  expect(calls.some((call) => call.url.endsWith("/start"))).toBe(false);
  await user.click(await screen.findByRole("button", { name: "确认并生成候选" }));
  await user.click(await screen.findByRole("button", { name: "查看候选" }));
  await screen.findByText("Generated algebra problem");
  expect(calls.find((call) => call.url.endsWith("/prepare"))?.body).toMatchObject({ mode: "ordinary", prompts: ["linear equation"] });
  expect(calls.find((call) => call.url.endsWith("/start"))?.body.confirm_start).toBe(true);
  expect(calls.some((call) => call.url.endsWith("/register"))).toBe(false);
});

it("targeted GRPO generation carries exact registered plan identity and source seed", async () => {
  const calls: Array<{ url: string; body?: any }> = [];
  const plan = { plan_id: "plan-grpo-targeted-11111111111111111111", plan_version: 3, scenario_id: "fractions", training_seed_dataset_id: "ds-exact-seed", max_requests: 2, max_candidates: 2, max_output_tokens: 128 };
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input); const body = init?.body ? JSON.parse(String(init.body)) : undefined; calls.push({ url, body });
    let value: any;
    if (url.endsWith("/capabilities")) value = { enabled: true, token: "local", profiles: [{ profile_id: "provider-local", model: "fixture" }], targeted_plans: [plan] };
    else if (url.endsWith("/prepare")) value = { bundle_id: "plan-grpo-targeted-11111111111111111111" };
    else if (url.endsWith("/preflight")) value = { provider_calls: 0, parent_dataset_ref: { dataset_id: "ds-exact-seed", content_sha256: "a".repeat(64), role: "train_seed" } };
    else if (url.endsWith("/start")) value = { execution_id: "grpo-gen-22222222222222222222", status: "queued", mode: "targeted" };
    else if (url.includes("/runs/")) value = { execution_id: "grpo-gen-22222222222222222222", status: "partial", mode: "targeted", result: { candidate_count: 1 } };
    else if (url.endsWith("/preview")) value = { trainable: false, registration_allowed: false, candidates: [] };
    else throw new Error(`unexpected request: ${url}`);
    return { ok: true, json: async () => value };
  });
  vi.stubGlobal("fetch", fetchMock);
  vi.stubGlobal("confirm", vi.fn(() => true));
  render(<GRPOGenerationPanel />);
  const user = userEvent.setup();
  await user.selectOptions(await screen.findByLabelText("生成方式"), "targeted");
  await user.selectOptions(screen.getByLabelText(/定向计划/), `${plan.plan_id}@${plan.plan_version}`);
  await user.click(screen.getByRole("button", { name: "准备方案并预检" }));
  await user.click(await screen.findByRole("button", { name: "确认并生成候选" }));
  await waitFor(() => expect(calls.some((call) => call.url.includes("/runs/"))).toBe(true));
  expect(calls.find((call) => call.url.endsWith("/prepare"))?.body).toMatchObject({
    mode: "targeted", stage_plan_id: plan.plan_id, plan_version: plan.plan_version,
  });
  expect(calls.some((call) => call.url.endsWith("/register"))).toBe(false);
});

it("restores a prior GRPO execution on refresh without replaying generation", async () => {
  window.localStorage.setItem("onegpu.grpo-generation-run.v1", JSON.stringify({
    execution_id: "grpo-gen-33333333333333333333", mode: "targeted",
  }));
  const calls: string[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input); calls.push(url);
    const value = url.endsWith("/capabilities")
      ? { enabled: true, token: "local", profiles: [], targeted_plans: [] }
      : { execution_id: "grpo-gen-33333333333333333333", status: "partial", mode: "targeted", result: { candidate_count: 1 } };
    return { ok: true, json: async () => value };
  });
  vi.stubGlobal("fetch", fetchMock);
  render(<GRPOGenerationPanel />);
  await screen.findByText(/GRPO 候选任务：partial/);
  expect(calls.filter((url) => url.includes("/runs/")).length).toBe(1);
  expect(calls.some((url) => url.endsWith("/start"))).toBe(false);
});
