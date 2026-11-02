import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Launcher } from "../src/Launcher";

const runId = "run-sft-status-fixture";
const profile = { profile_id: "sft-mini-1024-v1", stage: "sft", algorithm: "sft", approved_recipe: "recipes/mini/23-sft.yaml",
  parent_model_stages: ["cpt"], parent_model_rule: "approved_mini_cpt", dataset_stage: "sft", dataset_roles: ["train"],
  evaluation_profiles: [], execution_supported: true, training_parameters: [] };
const json = (value: unknown) => Promise.resolve(new Response(JSON.stringify(value), {
  status: 200, headers: { "Content-Type": "application/json" },
}));

describe("server-authoritative Launcher Run status", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", `/?page=launcher&stage=sft&profile_id=${profile.profile_id}&launcher_run=${runId}&launcher_tab=logs`);
    window.sessionStorage.clear();
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("shows truthful unknown progress and stops live state after a terminal poll and refresh", async () => {
    let current = { run_id: runId, run_kind: "stage-training", stage: "sft", profile_id: profile.profile_id,
      status: "running", evaluation_status: "pending", elapsed_seconds: 65, progress_status: "unknown",
      progress: null as { step: number; total: number } | null, updated_at: "2026-11-01T10:00:00+00:00",
      log_tail: "step started" };
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/training/stage-profiles") return json({ profiles: [profile] });
      if (url === "/api/training/workbench-options") return json({ datasets: [], models: [] });
      if (url === "/api/launcher/capabilities") return json({ enabled: true, available: true, token: "token" });
      if (url === `/api/launcher/runs/${runId}`) return json(current);
      return json({});
    }));
    const view = render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} />);
    expect(await screen.findByText("尚无可靠进度信息")).toBeTruthy();
    expect(screen.getByText("1 分 5 秒")).toBeTruthy();
    expect(screen.getByText("step started")).toBeTruthy();
    expect(screen.getByText(/2026/)).toBeTruthy();

    current = { ...current, status: "completed", elapsed_seconds: 90, progress_status: "measured",
      progress: { step: 20, total: 20 }, updated_at: "2026-11-01T10:01:00+00:00", log_tail: "training complete" };
    await waitFor(() => expect(screen.getByText("Step 20 / 20")).toBeTruthy(), { timeout: 3500 });
    expect(screen.getByText("1 分 30 秒")).toBeTruthy();
    expect(screen.getByText("training complete")).toBeTruthy();
    expect(screen.queryByText("● 实时更新")).toBeNull();

    view.unmount();
    render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} />);
    expect(await screen.findByText("Step 20 / 20")).toBeTruthy();
    expect(screen.queryByText("● 实时更新")).toBeNull();
  });
});
