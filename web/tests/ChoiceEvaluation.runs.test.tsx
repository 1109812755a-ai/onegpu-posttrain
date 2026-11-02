import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Runs } from "../src/Runs";

const runId = "run-sft-choice-fixture";
const datasetId = "ds-eval-upload-fixture";
const run = {
  run_id: runId, run_kind: "training", stage: "sft", status: "completed",
  started_at: "2026-11-01T00:00:00Z", completed_at: "2026-11-01T00:01:00Z",
  run_label: null, experiment_id: null, source_ref: "runs/sft/run.json",
  config: null, metrics: [
    { name: "choice_accuracy.overall", metric: "value", value: 0.4691358024691358, denominator: 81, denominator_unit: "questions" },
    { name: "choice_accuracy.single_choice", metric: "value", value: 0.4074074074074074, denominator: 27, denominator_unit: "questions" },
  ], logs: null, outputs: [], lineage: { lineage_status: "complete" },
  provenance: [], produced_artifacts: [], handoff_models: [], evaluations: [], parent_available: true,
};

function json(value: unknown) {
  return Promise.resolve(new Response(JSON.stringify(value), {
    status: 200, headers: { "Content-Type": "application/json" },
  }));
}

describe("objective choice evaluation in Runs", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", `/?page=runs&run=${runId}`);
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("shows the registered evaluation-only suite and sends its ID to preflight", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url.startsWith("/api/runs?")) return json({ runs: [run], total: 1, partial_scan: false });
      if (url === `/api/runs/${runId}`) return json(run);
      if (url === "/api/training/stage-profiles") return json({ profiles: [{
        profile_id: "sft-mini-1024-v1", stage: "sft", parent_model_stages: ["cpt"],
        parent_model_rule: "approved_mini_cpt", execution_supported: true,
        evaluation_profiles: [
          { profile_id: "sft-math-eval-v1", protocol: "math-eval-v1", metrics: ["gsm8k.accuracy"] },
          { profile_id: "choice-accuracy-v1", protocol: "choice-accuracy-v1", metrics: ["choice_accuracy.overall"] },
        ],
      }] });
      if (url === "/api/training/workbench-options") return json({ datasets: [
        { dataset_id: datasetId, stage: "eval", role: "evaluation_only", record_count: 81, eligible: true },
        { dataset_id: "sft-train-fixture", stage: "sft", role: "train", record_count: 10, eligible: true },
      ], models: [] });
      if (url === "/api/launcher/capabilities") return json({ enabled: true, available: true, token: "local-token" });
      if (url === "/api/training/evaluation/preflight") return json({
        ok: true, blockers: [], warnings: [], profile_id: "choice-accuracy-v1",
        protocol_version: "choice-accuracy-v1", confirmation_token: "preflight-token",
        model: { model_id: "model-sft", stage: "sft", sha256: "a".repeat(64) },
        evaluation_run_id: "run-evaluation-choice", output: "runs/evaluations/run-evaluation-choice",
        metrics: ["choice_accuracy.overall"],
      });
      return json({});
    });
    vi.stubGlobal("fetch", fetchMock);

    const user = userEvent.setup();
    render(<Runs onBack={vi.fn()} />);
    expect(await screen.findByText("总体准确率")).toBeTruthy();
    expect(screen.getByText("46.9%")).toBeTruthy();
    expect(screen.getByText("40.7%")).toBeTruthy();
    await user.selectOptions(await screen.findByLabelText("选择阶段评测方案"), "choice-accuracy-v1");
    const datasetSelect = await screen.findByLabelText("选择登记评测数据集");
    expect(Array.from((datasetSelect as HTMLSelectElement).options).map((option) => option.value))
      .toEqual(["", datasetId]);
    expect(screen.getByText("客观题评测集")).toBeTruthy();

    await user.selectOptions(datasetSelect, datasetId);
    await user.click(screen.getByRole("button", { name: "生成评测预检" }));

    const preflight = requests.find((request) => request.url === "/api/training/evaluation/preflight");
    await waitFor(() => expect(JSON.parse(String(preflight?.init?.body))).toEqual({
      training_run_id: runId, profile_id: "choice-accuracy-v1", dataset_id: datasetId,
      scoring_strategy: "exact_match",
    }));
    expect(await screen.findByText("预检通过，等待确认")).toBeTruthy();
    expect(screen.getByRole("button", { name: "确认并启动独立评测 Run" })).toBeTruthy();
  });

  async function readyToStart(startResponse: () => Promise<Response>, runLookup: () => Promise<Response>) {
    let starts = 0;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/runs?")) return json({ runs: [run], total: 1, partial_scan: false });
      if (url === `/api/runs/${runId}`) return json(run);
      if (url === "/api/training/stage-profiles") return json({ profiles: [{ profile_id: "sft-mini-1024-v1", stage: "sft",
        evaluation_profiles: [{ profile_id: "choice-accuracy-v1", protocol: "choice-accuracy-v1", metrics: [] }] }] });
      if (url === "/api/training/workbench-options") return json({ datasets: [
        { dataset_id: datasetId, stage: "eval", role: "evaluation_only", record_count: 81, eligible: true }], models: [] });
      if (url === "/api/launcher/capabilities") return json({ enabled: true, token: "token" });
      if (url === "/api/training/evaluation/preflight") return json({ ok: true, blockers: [], warnings: [], confirmation_token: "one-use",
        evaluation_run_id: "run-eval-choice", protocol_version: "choice-accuracy-v1", model: { model_id: "model-sft" }, output: "runs/evaluations/run-eval-choice" });
      if (url === "/api/training/evaluation/start") { starts += 1; return startResponse(); }
      if (url === "/api/launcher/runs/run-eval-choice") return runLookup();
      if (url === "/api/runs/run-eval-choice") return json({ ...run, run_id: "run-eval-choice", run_kind: "evaluation", status: "running" });
      return json({});
    }));
    const user = userEvent.setup();
    render(<Runs onBack={vi.fn()} />);
    await user.selectOptions(await screen.findByLabelText("选择阶段评测方案"), "choice-accuracy-v1");
    await user.selectOptions(await screen.findByLabelText("选择登记评测数据集"), datasetId);
    await user.click(screen.getByRole("button", { name: "生成评测预检" }));
    await screen.findByText("预检通过，等待确认");
    return { user, starts: () => starts };
  }

  it("uses the exact evaluation Run after a successful start and avoids a second click", async () => {
    const { user, starts } = await readyToStart(() => json({ run_id: "run-eval-choice" }), () => json({ run_id: "run-eval-choice" }));
    await user.click(screen.getByRole("button", { name: "确认并启动独立评测 Run" }));
    await waitFor(() => expect(window.location.search).toContain("run=run-eval-choice"));
    expect(starts()).toBe(1);
  });

  it("on 409 invalidates the used confirmation and requires a new preflight", async () => {
    const rejected = () => Promise.resolve(new Response(JSON.stringify({ detail: "阶段评测预检已过期或已使用" }), {
      status: 409, headers: { "Content-Type": "application/json" },
    }));
    const { user, starts } = await readyToStart(rejected, () => json({}));
    await user.click(screen.getByRole("button", { name: "确认并启动独立评测 Run" }));
    expect(await screen.findByRole("button", { name: "重新预检" })).toBeTruthy();
    expect(starts()).toBe(1);
  });

  it("on a network timeout recovers an already-created Run without a second POST", async () => {
    const { user, starts } = await readyToStart(() => Promise.reject(new TypeError("network timeout")),
      () => json({ run_id: "run-eval-choice" }));
    await user.click(screen.getByRole("button", { name: "确认并启动独立评测 Run" }));
    await waitFor(() => expect(window.location.search).toContain("run=run-eval-choice"));
    expect(starts()).toBe(1);
  });

  it("on an unconfirmed timeout asks for inspection before allowing another preflight", async () => {
    const notFound = () => Promise.resolve(new Response(JSON.stringify({ detail: "Unknown run" }), {
      status: 404, headers: { "Content-Type": "application/json" },
    }));
    const { user, starts } = await readyToStart(() => Promise.reject(new TypeError("network timeout")), notFound);
    await user.click(screen.getByRole("button", { name: "确认并启动独立评测 Run" }));
    expect(await screen.findByRole("button", { name: "检查任务状态" })).toBeTruthy();
    expect(starts()).toBe(1);
    await user.click(screen.getByRole("button", { name: "检查任务状态" }));
    expect(await screen.findByRole("button", { name: "重新预检" })).toBeTruthy();
    expect(starts()).toBe(1);
  });

  it("updates Bailian question counts from launcher polling without reloading the Run detail", async () => {
    const evaluationId = "run-eval-live-choice";
    const evaluation = { ...run, run_id: evaluationId, run_kind: "evaluation", stage: "base", status: "running",
      started_at: "2026-11-01T00:00:00Z", completed_at: null, evaluations: [], metrics: [] };
    window.history.replaceState({}, "", `/?page=runs&run=${evaluationId}`);
    let progress = { completed: 0, total: 3, phase: "loading_model", current_question: null, status: "running",
      counts: { correct: 0, incorrect: 0, undetermined: 0, provider_errors: 0 }, events: [] };
    const intervals: (() => void)[] = [];
    vi.spyOn(window, "setInterval").mockImplementation((handler) => {
      intervals.push(handler as () => void);
      return 1;
    });
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/runs?")) return json({ runs: [evaluation], total: 1, partial_scan: false });
      if (url === `/api/runs/${evaluationId}`) return json(evaluation);
      if (url === `/api/launcher/runs/${evaluationId}`) return json({ run_id: evaluationId, status: "running",
        scoring_strategy: "bailian_llm_judge", choice_judge_progress: progress });
      return json({});
    }));

    render(<Runs onBack={vi.fn()} />);
    expect(await screen.findByRole("heading", { name: "百炼评判实时进度" })).toBeTruthy();
    expect(screen.getByText("0 / 3 题")).toBeTruthy();

    progress = { completed: 2, total: 3, phase: "recorded", current_question: 2, status: "running",
      counts: { correct: 1, incorrect: 0, undetermined: 1, provider_errors: 0 },
      events: [{ completed: 2, total: 3, phase: "recorded", current_question: 2 }] };
    await act(async () => { intervals.forEach((callback) => callback()); });

    await waitFor(() => expect(screen.getByText("2 / 3 题")).toBeTruthy());
    expect(screen.getAllByText("1", { selector: "strong" })).toHaveLength(2);
    expect(screen.getByText("Launcher 每 1.5 秒自动更新 · 当前阶段：已保存本题结果")).toBeTruthy();
  });

  it("keeps a failed Run and missing evidence honest while hiding technical IDs one level down", async () => {
    const failed = { ...run, status: "failed", run_label: "十月客观题试跑", provenance: [], produced_artifacts: [],
      lineage: { lineage_status: "incomplete" }, metrics: [] };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/runs?")) return json({ runs: [failed], total: 1, partial_scan: false });
      if (url === `/api/runs/${runId}`) return json(failed);
      return json({});
    }));
    const user = userEvent.setup();
    render(<Runs onBack={vi.fn()} />);
    expect(await screen.findByRole("heading", { name: "十月客观题试跑" })).toBeTruthy();
    expect(screen.getByText("失败")).toBeTruthy();
    expect(screen.getByText(/0 项 · 运行未成功，不代表可交接/)).toBeTruthy();
    const technical = screen.getByText("查看 Run ID、实验 ID 与记录路径").closest("details") as HTMLDetailsElement;
    expect(technical.open).toBe(false);
    await user.click(screen.getByText("查看 Run ID、实验 ID 与记录路径"));
    expect(technical.open).toBe(true);
    expect(screen.queryByRole("button", { name: /试用此模型/ })).toBeNull();
  });

  it("shows readable result counts while keeping raw paths and configuration expandable", async () => {
    const finished = { ...run, run_label: "十一月 SFT 试跑", outputs: ["runs/sft/fixture/merged/model.json"],
      logs: { path: "runs/sft/fixture/logs/train.log", text: "step 20/20", truncated: false },
      config: { recipe: "smoke" }, produced_artifacts: [{ id: "model-1", kind: "model", role: "merged", path: "runs/sft/fixture/merged" }] };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/runs?")) return json({ runs: [finished], total: 1, partial_scan: false });
      if (url === `/api/runs/${runId}`) return json(finished);
      return json({});
    }));
    render(<Runs onBack={vi.fn()} />);
    expect(await screen.findByRole("heading", { name: "十一月 SFT 试跑" })).toBeTruthy();
    expect(screen.getByText("可读取产物 1 项。")).toBeTruthy();
    expect(screen.getByText("step 20/20")).toBeTruthy();
    expect((screen.getByText("查看产物文件与路径").closest("details") as HTMLDetailsElement).open).toBe(false);
    expect((screen.getByText("查看日志路径").closest("details") as HTMLDetailsElement).open).toBe(false);
    expect((screen.getByText("查看解析配置").closest("details") as HTMLDetailsElement).open).toBe(false);
  });
});
