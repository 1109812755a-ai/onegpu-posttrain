import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Launcher } from "../src/Launcher";

const runId = "run-sft-fixture";
const datasetId = "ds-objective-exam";
const profile = {
  profile_id: "sft-mini-1024-v1", stage: "sft", algorithm: "sft", approved_recipe: "recipes/mini/23-sft-1024-fallback.yaml",
  parent_model_stages: ["cpt"], parent_model_rule: "approved_mini_cpt", dataset_stage: "sft", dataset_roles: ["train"],
  evaluation_profiles: [{ profile_id: "choice-accuracy-v1", protocol: "choice-accuracy-v1", metrics: ["choice_accuracy.overall"] }],
  execution_supported: true, training_parameters: [],
};
const run = { run_id: runId, run_kind: "stage-training", status: "completed", stage: "sft", profile_id: profile.profile_id,
  evaluation_status: "completed", model_artifacts: [{ model_id: "model-sft", kind: "merged", stage: "sft" }],
  evaluation_metrics: { protocol_version: "choice-accuracy-v1",
    evaluation_suite: { dataset_id: datasetId, dataset_sha256: "a".repeat(64), sample_count: 82 },
    comparability: { key: "b".repeat(64), note: "仅相同评测集内容摘要、覆盖和协议的结果可直接比较。" },
    metrics: [{ name: "choice_accuracy.single_choice", metric: "value", value: 0.75, correct: 6,
      denominator: 8, denominator_unit: "questions", parse_failures: 1 }] } };

function json(value: unknown) {
  return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { "Content-Type": "application/json" } }));
}

describe("registered objective choice evaluation in Launcher", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", `/?page=launcher&stage=sft&profile_id=${profile.profile_id}&launcher_run=${runId}&launcher_tab=evaluation`);
    window.sessionStorage.clear();
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("selects an evaluation-only suite, checks coverage, and displays comparability identity", async () => {
    const requests: Array<{ url: string; init?: RequestInit }> = [];
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url === "/api/training/stage-profiles") return json({ profiles: [profile] });
      if (url === "/api/training/workbench-options") return json({ datasets: [
        { dataset_id: datasetId, display_name: "校内客观题集", stage: "eval", role: "evaluation_only", record_count: 82, eligible: true },
        { dataset_id: "not-eval", display_name: "SFT 训练集", stage: "sft", role: "train", record_count: 10, eligible: true },
      ], models: [] });
      if (url === "/api/launcher/capabilities") return json({ enabled: true, available: true, token: "local-token" });
      if (url === `/api/launcher/runs/${runId}`) return json(run);
      if (url === "/api/training/evaluation/preflight") return json({
        ok: true, profile_id: "choice-accuracy-v1", protocol_version: "choice-accuracy-v1",
        training_run_id: runId, evaluation_run_id: "run-evaluate-choice", output: "runs/evaluations/run-evaluate-choice",
        model: { model_id: "model-sft" }, dataset: { dataset_id: datasetId, record_count: 82,
          content_sha256: "a".repeat(64), protocol_version: "choice-accuracy-v1", comparability_key: "b".repeat(64) },
        warnings: [], blockers: [], confirmation_token: "preflight-token",
      });
      return json({});
    });
    vi.stubGlobal("fetch", fetchMock);
    const user = userEvent.setup();
    render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} />);
    await user.selectOptions(await screen.findByLabelText("评测协议"), "choice-accuracy-v1");
    const datasetSelect = await screen.findByLabelText("已登记客观题评测集");
    expect(Array.from((datasetSelect as HTMLSelectElement).options).map((option) => option.value)).toEqual(["", datasetId]);
    await user.selectOptions(datasetSelect, datasetId);
    await user.click(screen.getByRole("button", { name: "检查评测配置" }));
    expect(await screen.findByText(/仅当评测集摘要、题数和协议一致时结果可直接比较/)).toBeTruthy();
    expect(screen.getByText(/客观题评测结果 · ds-objective-exam/)).toBeTruthy();
    expect(screen.getByText(/正确 6 \/ 8 题 · 解析失败 1/)).toBeTruthy();
    expect(screen.getByText((_, element) => element?.tagName === "P"
      && element.textContent?.includes(datasetId) === true
      && element.textContent.includes("82 题"))).toBeTruthy();
    const preflight = requests.find((request) => request.url === "/api/training/evaluation/preflight");
    await waitFor(() => expect(JSON.parse(String(preflight?.init?.body))).toEqual({
      training_run_id: runId, profile_id: "choice-accuracy-v1", dataset_id: datasetId,
    }));
    expect(requests.some((request) => request.url === "/api/training/evaluation/start")).toBe(false);
  });
});
