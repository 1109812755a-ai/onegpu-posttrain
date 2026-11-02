import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Launcher } from "../src/Launcher";

const profile = {
  profile_id: "sft-registered-v1", stage: "sft", algorithm: "sft",
  approved_recipe: "recipes/mini/24-sft-registered-parent.yaml", parent_model_stages: ["base", "cpt"],
  parent_model_rule: "registered_sft_parent", dataset_stage: "sft", dataset_roles: ["train"],
  evaluation_profiles: [], execution_supported: true, training_parameters: [], training_memory_modes: [],
};
const baseModel = { model_id: "model-base-qwen", stage: "base", kind: "base", path_available: true,
  digest: { status: "available", value: "a".repeat(64) }, registry_count: 1 };
const unavailableBase = { model_id: "model-base-missing", stage: "base", kind: "base", path_available: false,
  unavailable_reason: "缺少 Transformers 配置或权重文件", digest: { status: "available", value: "b".repeat(64) }, registry_count: 1 };

function response(value: unknown) {
  return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { "Content-Type": "application/json" } }));
}

describe("registered Base SFT selector", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", `/?page=launcher&stage=sft&profile_id=${profile.profile_id}`);
    window.sessionStorage.clear();
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("shows available registered Base models and sends the selected ID to stage preflight", async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url === "/api/training/stage-profiles") return response({ profiles: [profile] });
      if (url === "/api/training/workbench-options") return response({
        datasets: [{ dataset_id: "ds-sft", display_name: "已登记 SFT 数据", stage: "sft", role: "train", record_count: 8, eligible: true }],
        models: [baseModel, unavailableBase],
      });
      if (url === "/api/launcher/capabilities") return response({ enabled: true, available: true, token: "local-token" });
      if (url === "/api/launcher/run") return response({ run: null });
      if (url === "/api/training/stage/preflight") return response({ status: "ready_for_confirmation", blockers: [], warnings: [],
        datasets: [{ dataset_id: "ds-sft", license_status: "verified" }], parent_model: { model_id: baseModel.model_id } });
      if (url === "/api/training/stage/confirm") return response({ status: "ready_for_confirmation", confirmation_token: "confirm-token",
        run_id: "run-base-sft", output_location: "runs/sft/sft-registered-v1/run-base-sft" });
      return response({});
    });
    vi.stubGlobal("fetch", fetchMock);
    render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} />);

    const parentSelect = await screen.findByLabelText("阶段父模型") as HTMLSelectElement;
    expect(Array.from(parentSelect.options).some((option) => option.value === baseModel.model_id && !option.disabled)).toBe(true);
    expect(Array.from(parentSelect.options).some((option) => option.value === unavailableBase.model_id && option.disabled)).toBe(true);
    fireEvent.change(parentSelect, { target: { value: baseModel.model_id } });
    fireEvent.click(await screen.findByRole("radio", { name: /已登记 SFT 数据/ }));
    fireEvent.click(screen.getByRole("button", { name: "检查配置并预检" }));

    await waitFor(() => expect(fetchMock.mock.calls.some(([url]) => String(url) === "/api/training/stage/confirm")).toBe(true));
    const preflightCall = fetchMock.mock.calls.find(([url]) => String(url) === "/api/training/stage/preflight");
    expect(preflightCall).toBeTruthy();
    const payload = JSON.parse(String((preflightCall?.[1] as RequestInit).body));
    expect(payload.profile_id).toBe(profile.profile_id);
    expect(payload.parent_model_id).toBe(baseModel.model_id);
    expect(payload.dataset_ids).toEqual(["ds-sft"]);
  });
});
