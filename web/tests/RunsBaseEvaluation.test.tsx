import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Runs } from "../src/Runs";

const json = (value: unknown, status = 200) => Promise.resolve(new Response(JSON.stringify(value), {
  status, headers: { "Content-Type": "application/json" },
}));

describe("Runs Base evaluation task state", () => {
  beforeEach(() => window.history.replaceState({}, "", "/?page=runs"));
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  async function setup(start: () => Promise<Response>, lookup: () => Promise<Response> = () => json({ run_id: "eval-base-1" })) {
    let posts = 0;
    vi.stubGlobal("fetch", vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.startsWith("/api/runs?")) return json({ runs: [], total: 0, partial_scan: false });
      if (url === "/api/training/workbench-options") return json({ datasets: [{ dataset_id: "ds-choice", stage: "eval", role: "evaluation_only", record_count: 81 }], models: [] });
      if (url === "/api/model-playground/models") return json({ status: "ready", models: [{ model_id: "model-base", kind: "base", available: true }] });
      if (url === "/api/launcher/capabilities") return json({ enabled: true, token: "local-token" });
      if (url === "/api/training/evaluation/preflight") return json({ ok: true, blockers: [], warnings: [], confirmation_token: "once", evaluation_run_id: "eval-base-1", model: { model_id: "model-base" }, protocol_version: "choice-accuracy-v1", output: "runs/evaluations/eval-base-1" });
      if (url === "/api/training/evaluation/start" && init?.method === "POST") { posts += 1; return start(); }
      if (url === "/api/launcher/runs/eval-base-1") return lookup();
      if (url === "/api/runs/eval-base-1") return json({ run_id: "eval-base-1", run_kind: "evaluation", stage: "evaluate", status: "running", started_at: null, completed_at: null, run_label: "Base 客观题评测", experiment_id: null, source_ref: "fixture", config: null, metrics: [], logs: null, outputs: [], lineage: null, provenance: [], evaluations: [] });
      return json({});
    }));
    const user = userEvent.setup();
    render(<Runs onBack={vi.fn()} />);
    await user.selectOptions(await screen.findByLabelText("选择 Base 模型"), "model-base");
    await user.selectOptions(screen.getByLabelText("选择 Base 模型评测方案"), "choice-accuracy-v1");
    await user.selectOptions(await screen.findByLabelText("选择客观题评测集"), "ds-choice");
    await user.click(screen.getByRole("button", { name: "生成评测预检" }));
    await screen.findByText("预检通过，等待确认");
    return { user, posts: () => posts };
  }

  it("opens the exact evaluation Run after successful confirmation", async () => {
    const { user, posts } = await setup(() => json({ run_id: "eval-base-1" }));
    await user.click(screen.getByRole("button", { name: "确认并启动独立评测 Run" }));
    await waitFor(() => expect(window.location.search).toContain("run=eval-base-1"));
    expect(posts()).toBe(1);
  });

  it("requires a fresh preflight after 409 and does not duplicate the POST", async () => {
    const { user, posts } = await setup(() => json({ detail: "令牌已使用" }, 409));
    await user.click(screen.getByRole("button", { name: "确认并启动独立评测 Run" }));
    expect(await screen.findByRole("button", { name: "重新预检" })).toBeTruthy();
    expect(posts()).toBe(1);
  });

  it("recovers an accepted Run after a network timeout without restarting", async () => {
    const { user, posts } = await setup(() => Promise.reject(new TypeError("network timeout")));
    await user.click(screen.getByRole("button", { name: "确认并启动独立评测 Run" }));
    await waitFor(() => expect(window.location.search).toContain("run=eval-base-1"));
    expect(posts()).toBe(1);
  });

  it("invalidates old confirmation when the evaluation dataset changes", async () => {
    const { user, posts } = await setup(() => json({ run_id: "eval-base-1" }));
    await user.selectOptions(screen.getByLabelText("选择客观题评测集"), "");
    expect(screen.queryByRole("button", { name: "确认并启动独立评测 Run" })).toBeNull();
    expect(posts()).toBe(0);
  });
});
