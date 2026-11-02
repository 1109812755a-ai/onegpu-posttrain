import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Launcher } from "../src/Launcher";

const profile = {
  profile_id: "sft-mini-1024-v1", stage: "sft", algorithm: "sft", approved_recipe: "recipes/mini/23-sft-1024-fallback.yaml",
  parent_model_stages: ["cpt"], parent_model_rule: "approved_mini_cpt", dataset_stage: "sft", dataset_roles: ["train"],
  evaluation_profiles: [], execution_supported: true,
  training_parameters: [{ key: "max_length", label_zh: "最大序列长度", kind: "int", default: 1024, minimum: 512, maximum: 4047, choices: [] }],
  training_memory_modes: [
    { mode_id: "lora", label_zh: "标准 LoRA", description_zh: "默认模式", default: true, settings: {} },
    { mode_id: "qlora-4bit-nf4", label_zh: "低显存 QLoRA（4-bit NF4）", description_zh: "固定量化配置", default: false,
      settings: { quant_method: "bnb", quant_bits: 4, bnb_4bit_compute_dtype: "float16", bnb_4bit_quant_type: "nf4", bnb_4bit_use_double_quant: true } },
  ],
};

function mockApi(stagePreflight: Record<string, unknown>, launchCheck?: Record<string, unknown>, datasets?: Record<string, unknown>[]) {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    let value: unknown = {};
    if (url === "/api/training/stage-profiles") value = { profiles: [profile] };
    if (url === "/api/training/workbench-options") value = { datasets: datasets || [{ dataset_id: "sft-data", stage: "sft", role: "train", record_count: 4 }], models: [] };
    if (url === "/api/launcher/capabilities") value = { enabled: true, available: true, token: "local", parent_model_id: "model-cpt" };
    if (url === "/api/launcher/run") value = { run: null };
    if (url === "/api/training/preflight") value = stagePreflight;
    if (url === "/api/launcher/preflight") value = launchCheck || {};
    return Promise.resolve(new Response(JSON.stringify(value), { status: 200, headers: { "Content-Type": "application/json" } }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

describe("SFT QLoRA workbench controls", () => {
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); window.sessionStorage.clear(); });

  it("defaults to LoRA and shows an actionable dependency blocker for QLoRA preflight", async () => {
    const fetchMock = mockApi({ status: "blocked", blockers: ["QLoRA 需要当前项目 Python 环境安装 bitsandbytes；预检不会自动安装依赖。"],
      warnings: [], checks: [{ check_id: "training_memory:qlora_dependencies", status: "fail", detail: "QLoRA 依赖缺失" }] });
    window.history.replaceState({}, "", "/?page=launcher&stage=sft&profile_id=sft-mini-1024-v1");
    render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} />);

    const mode = await screen.findByLabelText("显存优化模式") as HTMLSelectElement;
    expect(mode.value).toBe("lora");
    fireEvent.change(mode, { target: { value: "qlora-4bit-nf4" } });
    expect(await screen.findByText(/预检只检查、不安装/)).toBeTruthy();
    fireEvent.click(screen.getByRole("radio"));
    fireEvent.click(screen.getByRole("button", { name: "检查配置并预检" }));
    expect((await screen.findAllByText(/QLoRA 需要当前项目 Python 环境安装 bitsandbytes/)).length).toBeGreaterThan(0);
    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([input]) => String(input) === "/api/training/preflight");
      expect(JSON.parse(String(request?.[1]?.body)).memory_mode_id).toBe("qlora-4bit-nf4");
    });
    expect(screen.queryByRole("button", { name: "启动 SFT 训练" })).toBeNull();
  });

  it("preflights an available QLoRA mode with the chosen max sequence length and clears stale results on change", async () => {
    const fetchMock = mockApi({ status: "ready_for_confirmation", blockers: [], warnings: [],
      parent_model: { model_id: "model-cpt", digest: "b".repeat(64) },
      datasets: [{ dataset_id: "sft-data", content_sha256: "a".repeat(64) }],
      training_parameters: { effective: { max_length: 2048 }, config_sha256: "c".repeat(64) } },
    { ok: true, blockers: [], warnings: [], confirmation_token: "preflight-token",
      model: { model_id: "model-cpt", sha256: "b".repeat(64) },
      dataset: { dataset_id: "sft-data", canonical_sha256: "a".repeat(64) },
      training_parameters: { training_config_sha256: "c".repeat(64) } });
    window.history.replaceState({}, "", "/?page=launcher&stage=sft&profile_id=sft-mini-1024-v1");
    render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} />);
    const mode = await screen.findByLabelText("显存优化模式") as HTMLSelectElement;
    fireEvent.change(mode, { target: { value: "qlora-4bit-nf4" } });
    fireEvent.click(screen.getByRole("radio"));
    fireEvent.change(screen.getByLabelText("最大序列长度"), { target: { value: "2048" } });
    fireEvent.click(screen.getByRole("button", { name: "检查配置并预检" }));
    expect(await screen.findByRole("heading", { name: "预检通过" })).toBeTruthy();
    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([input]) => String(input) === "/api/launcher/preflight");
      expect(JSON.parse(String(request?.[1]?.body)).memory_mode_id).toBe("qlora-4bit-nf4");
    });
    fireEvent.change(mode, { target: { value: "lora" } });
    expect(screen.queryByRole("heading", { name: "预检通过" })).toBeNull();
  });

  it("supports the default, reproducible split, and registered validation modes and invalidates stale preflight", async () => {
    const fetchMock = mockApi({ status: "blocked", blockers: [], warnings: [] },
      { ok: true, blockers: [], warnings: [], confirmation_token: "preflight-token",
        training_parameters: { config_sha256: "c".repeat(64) } }, [
        { dataset_id: "sft-data", stage: "sft", role: "train", record_count: 20, eligible: true },
        { dataset_id: "sft-validation", stage: "sft", role: "validation", record_count: 8, eligible: true },
        { dataset_id: "sft-final-holdout", stage: "sft", role: "final_holdout", record_count: 8, eligible: true },
      ]);
    window.history.replaceState({}, "", "/?page=launcher&stage=sft&profile_id=sft-mini-1024-v1");
    render(<Launcher onBack={vi.fn()} onOpenRun={vi.fn()} />);

    const source = await screen.findByLabelText("训练期间验证来源") as HTMLSelectElement;
    expect(source.value).toBe("none");
    expect(screen.queryByLabelText("独立验证集")).toBeNull();
    fireEvent.click(screen.getByRole("radio"));
    fireEvent.change(source, { target: { value: "random_split" } });
    fireEvent.change(screen.getByLabelText("验证比例"), { target: { value: "0.4" } });
    fireEvent.change(screen.getByLabelText("验证随机种子"), { target: { value: "123" } });
    fireEvent.click(screen.getByRole("button", { name: "检查配置并预检" }));
    await waitFor(() => {
      const request = fetchMock.mock.calls.find(([input]) => String(input) === "/api/launcher/preflight");
      expect(JSON.parse(String(request?.[1]?.body)).validation).toEqual({ mode: "random_split", ratio: 0.4, seed: 123 });
    });

    fireEvent.change(source, { target: { value: "registered" } });
    expect(screen.queryByRole("heading", { name: "预检通过" })).toBeNull();
    const validationDataset = await screen.findByLabelText("独立验证集") as HTMLSelectElement;
    expect(Array.from(validationDataset.options).map((option) => option.value)).not.toContain("sft-final-holdout");
    fireEvent.change(validationDataset, { target: { value: "sft-validation" } });
    fireEvent.click(screen.getByRole("button", { name: "检查配置并预检" }));
    await waitFor(() => {
      const requests = fetchMock.mock.calls.filter(([input]) => String(input) === "/api/launcher/preflight");
      expect(JSON.parse(String(requests.at(-1)?.[1]?.body)).validation).toEqual({ mode: "registered", dataset_id: "sft-validation" });
    });
  });
});
