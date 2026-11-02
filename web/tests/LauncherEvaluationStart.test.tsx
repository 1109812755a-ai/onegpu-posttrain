import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Launcher } from "../src/Launcher";

const trainingId = "run-sft-eval-fixture";
const evaluationId = "run-evaluate-choice-fixture";
const datasetId = "ds-eval-fixture";
const profile = {
  profile_id: "sft-mini-1024-v1", stage: "sft", algorithm: "sft", approved_recipe: "recipes/mini/23-sft.yaml",
  parent_model_stages: ["cpt"], parent_model_rule: "approved_mini_cpt", dataset_stage: "sft", dataset_roles: ["train"],
  evaluation_profiles: [{ profile_id: "choice-accuracy-v1", protocol: "choice-accuracy-v1", metrics: [] }],
  execution_supported: true, training_parameters: [],
};
const training = { run_id: trainingId, run_kind: "stage-training", status: "completed", stage: "sft",
  profile_id: profile.profile_id, evaluation_status: "pending", model_artifacts: [] };
const json = (value: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(value), {
  status, headers: { "Content-Type": "application/json" },
}));

describe("Launcher stage evaluation start handoff", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", `/?page=launcher&stage=sft&profile_id=${profile.profile_id}&launcher_run=${trainingId}&launcher_tab=evaluation`);
    window.sessionStorage.clear();
    vi.stubGlobal("confirm", vi.fn(() => true));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  async function ready(start: () => Promise<Response>, lookup: () => Promise<Response>) {
    let starts = 0;
    let trainingStatus = training;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/training/stage-profiles") return json({ profiles: [profile] });
      if (url === "/api/training/workbench-options") return json({ datasets: [
        { dataset_id: datasetId, stage: "eval", role: "evaluation_only", record_count: 81, eligible: true }], models: [] });
      if (url === "/api/launcher/capabilities") return json({ enabled: true, available: true, token: "token" });
      if (url === `/api/launcher/runs/${trainingId}`) return json(trainingStatus);
      if (url === `/api/launcher/runs/${evaluationId}`) return lookup();
      if (url === "/api/training/evaluation/preflight") return json({ ok: true, blockers: [], warnings: [],
        profile_id: "choice-accuracy-v1", protocol_version: "choice-accuracy-v1", confirmation_token: "once",
        evaluation_run_id: evaluationId, model: { model_id: "model-sft" }, dataset: { dataset_id: datasetId, record_count: 81 },
        output: `runs/evaluations/${evaluationId}` });
      if (url === "/api/training/evaluation/start") { starts += 1; trainingStatus = {
        ...training, evaluation_status: "running", evaluation_run_id: evaluationId,
      } as typeof training; return start(); }
      return json({});
    }));
    const onOpenRun = vi.fn();
    const user = userEvent.setup();
    const view = render(<Launcher onBack={vi.fn()} onOpenRun={onOpenRun} />);
    await user.selectOptions(await screen.findByLabelText("评测协议"), "choice-accuracy-v1");
    await user.selectOptions(await screen.findByLabelText("已登记客观题评测集"), datasetId);
    await user.click(screen.getByRole("button", { name: "检查评测配置" }));
    await screen.findByText("预检通过，等待确认");
    return { user, view, onOpenRun, starts: () => starts };
  }

  it("shows the exact evaluation Run after start, not a permanent submit spinner", async () => {
    const { user, view, onOpenRun, starts } = await ready(() => json({ run_id: evaluationId, status: "running" }),
      () => json({ run_id: evaluationId, status: "running" }));
    await user.click(screen.getByRole("button", { name: "启动阶段评测" }));
    const card = await waitFor(() => {
      const element = view.container.querySelector('.task-status-card[data-task-phase="run_running"]');
      expect(element).toBeTruthy();
      return element as HTMLElement;
    });
    await user.click(within(card).getByRole("button", { name: "查看 Run" }));
    expect(onOpenRun).toHaveBeenCalledWith(evaluationId);
    expect(starts()).toBe(1);
  });

  it("requires a new preflight after an explicit 409", async () => {
    const { user, starts } = await ready(() => json({ detail: "确认令牌已过期" }, 409),
      () => json({ detail: "Unknown run" }, 404));
    await user.click(screen.getByRole("button", { name: "启动阶段评测" }));
    expect(await screen.findByRole("button", { name: "重新预检" })).toBeTruthy();
    expect(starts()).toBe(1);
  });

  it("recovers a Run created before a network timeout by read-only lookup", async () => {
    const { user, view, starts } = await ready(() => Promise.reject(new TypeError("timeout")),
      () => json({ run_id: evaluationId, status: "running" }));
    await user.click(screen.getByRole("button", { name: "启动阶段评测" }));
    await waitFor(() => expect(view.container.querySelector('.task-status-card[data-task-phase="run_running"]')).toBeTruthy());
    expect(starts()).toBe(1);
  });

  it("keeps an unconfirmed timeout inspectable and never reposts automatically", async () => {
    const { user, starts } = await ready(() => Promise.reject(new TypeError("timeout")),
      () => json({ detail: "Unknown run" }, 404));
    await user.click(screen.getByRole("button", { name: "启动阶段评测" }));
    expect(await screen.findByRole("button", { name: "检查任务状态" })).toBeTruthy();
    expect(starts()).toBe(1);
    await user.click(screen.getByRole("button", { name: "检查任务状态" }));
    expect(await screen.findByRole("button", { name: "重新预检" })).toBeTruthy();
    expect(starts()).toBe(1);
  });
});
