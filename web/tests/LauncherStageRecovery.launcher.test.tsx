import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Launcher } from "../src/Launcher";

const profile = {
  profile_id: "grpo-lowmem-smoke-v1", stage: "grpo", algorithm: "grpo", approved_recipe: "recipes/smoke/41-grpo-lowmem.yaml",
  parent_model_stages: ["sft"], parent_model_rule: "registered_sft_merged", dataset_stage: "grpo", dataset_roles: ["train_prompts"],
  evaluation_profiles: [], execution_supported: true, training_parameters: [],
};
const model = { model_id: "model-sft-merged", stage: "sft", kind: "merged", path_available: true,
  digest: { status: "available", value: "a".repeat(64) }, source_run_id: "run-sft" };

function response(value: unknown) {
  return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { "Content-Type": "application/json" } }));
}

function setup(datasets: Array<Record<string, unknown>>) {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/training/stage-profiles") return response({ profiles: [profile] });
    if (url === "/api/training/workbench-options") return response({ datasets, models: [model] });
    if (url === "/api/launcher/capabilities") return response({ enabled: true, available: true, token: "local-token" });
    if (url === "/api/launcher/run") return response({ run: null });
    return response({});
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderLauncher(onPrepareGrpoData = vi.fn()) {
  return render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} onPrepareGrpoData={onPrepareGrpoData} />);
}

describe("GRPO launcher recovery entry", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/?page=launcher&stage=grpo&profile_id=grpo-lowmem-smoke-v1&parent_model_id=model-sft-merged");
    window.sessionStorage.clear();
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("offers a direct preparation action when no GRPO dataset is registered", async () => {
    setup([]);
    const onPrepare = vi.fn();
    renderLauncher(onPrepare);
    const action = await screen.findByRole("button", { name: "准备 GRPO 数据 →" });
    await waitFor(() => expect(action).toBeTruthy());
    action.click();
    expect(onPrepare).toHaveBeenCalledWith({ profileId: profile.profile_id, parentModelId: model.model_id });
  });

  it("keeps backend blocker details and disabled state while exposing recovery for all-blocked data", async () => {
    setup([{ dataset_id: "grpo-old", display_name: "旧 GRPO 数据", stage: "grpo", role: "train_prompts", record_count: 3,
      eligible: false, block_reason: "缺少可执行 reward/reference contract" }]);
    renderLauncher();
    const datasetGroup = await screen.findByRole("group", { name: /选择一个已登记训练数据集/ });
    const radio = within(datasetGroup).getByRole("radio");
    expect(radio.hasAttribute("disabled")).toBe(true);
    expect(within(datasetGroup).getByText(/缺少可执行 reward\/reference contract/)).toBeTruthy();
    expect(await screen.findByRole("button", { name: "准备 GRPO 数据 →" })).toBeTruthy();
  });

  it("does not hide a selectable dataset or show an empty-data recovery action for mixed availability", async () => {
    setup([
      { dataset_id: "grpo-old", display_name: "旧数据", stage: "grpo", role: "train_prompts", record_count: 3, eligible: false, block_reason: "缺少契约" },
      { dataset_id: "grpo-good", display_name: "可用数据", stage: "grpo", role: "train_prompts", record_count: 4, eligible: true },
    ]);
    renderLauncher();
    const datasetGroup = await screen.findByRole("group", { name: /选择一个已登记训练数据集/ });
    const radios = within(datasetGroup).getAllByRole("radio");
    expect(radios).toHaveLength(2);
    expect(radios[0].hasAttribute("disabled")).toBe(true);
    expect(radios[1].hasAttribute("disabled")).toBe(false);
    expect(screen.queryByRole("button", { name: "准备 GRPO 数据 →" })).toBeNull();
  });

  it("restores only the exact eligible dataset and compatible parent on return without starting or reusing preflight", async () => {
    window.history.replaceState({}, "", `/?page=launcher&stage=grpo&profile_id=${profile.profile_id}&parent_model_id=${model.model_id}&dataset_id=grpo-registered-exact&return_to=launcher`);
    const fetchMock = setup([{ dataset_id: "grpo-registered-exact", display_name: "新登记 GRPO 数据", stage: "grpo", role: "train_prompts", record_count: 3, eligible: true }]);
    renderLauncher();
    const datasetGroup = await screen.findByRole("group", { name: /选择一个已登记训练数据集/ });
    await waitFor(() => expect((within(datasetGroup).getByRole("radio") as HTMLInputElement).checked).toBe(true));
    expect((screen.getByLabelText("阶段父模型") as HTMLSelectElement).value).toBe(model.model_id);
    expect(screen.queryByText(/旧预检/)).toBeNull();
    const requestedUrls = fetchMock.mock.calls.map(([input]) => String(input));
    expect(requestedUrls).not.toContain("/api/training/stage/preflight");
    expect(requestedUrls).not.toContain("/api/training/stage/start");
    expect(window.sessionStorage.getItem("onegpu.stage-workbench.draft.v1:grpo") || "").not.toContain("token");

    window.history.replaceState({}, "", `/?page=launcher&stage=grpo&profile_id=removed-profile&parent_model_id=removed-model&dataset_id=grpo-registered-exact`);
    cleanup();
    renderLauncher();
    await screen.findByRole("group", { name: /选择一个已登记训练数据集/ });
    expect(await screen.findByText(/传入的模型 removed-model 不符合/)).toBeTruthy();
    expect((screen.getByLabelText("阶段父模型") as HTMLSelectElement).value).toBe("");
  });
});
