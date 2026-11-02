import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { DatasetWorkbench } from "../src/DatasetWorkbench";

const datasetA = "ds-sft-local-a";
const datasetB = "ds-sft-generated-b";

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(payload), {
    status,
    headers: { "Content-Type": "application/json" },
  }));
}

function catalog(includeMixture = false, includeLegacy = false) {
  const builds = [{ id: "build-sft", name: "SFT fixtures", path: "artifacts/build-sft", source: "fixture",
      revision: null, mode: "real", is_demo: false, raw_count: 2, raw_unit: "rows", created_at: null,
      warnings: [], datasets: [
        { stage: "sft", dataset_id: datasetA, count: 2, target: null, unit: "examples", tokens: null,
          token_target: null, quality_status: "report_only", source: { kind: "local_upload", name: "本地导入" } },
        { stage: "sft", dataset_id: datasetB, count: 3, target: null, unit: "examples", tokens: null,
          token_target: null, quality_status: "report_only", source: { kind: "model_generated", name: "模型生成" } },
      ] }];
  if (includeMixture) builds.push({ id: "build-mixture", name: "组合结果", path: "artifacts/build-mixture", source: "fixture",
    revision: null, mode: "real", is_demo: false, raw_count: 5, raw_unit: "rows", created_at: null,
    warnings: [], datasets: [{ stage: "sft", dataset_id: "ds-sft-mixture-result", count: 5, target: null,
      unit: "examples", tokens: null, token_target: null, quality_status: "report_only",
      source: { kind: "dataset_mixture", name: "数据组合" } }] });
  if (includeLegacy) builds.push({ id: "build-legacy", name: "Legacy smoke", path: "data/processed/smoke",
    source: "open-r1/OpenR1-Math-220k", revision: "legacy", mode: "legacy", is_demo: false, raw_count: 9,
    raw_unit: "rows", created_at: null, warnings: [], datasets: [{ stage: "sft", dataset_id: "sft-smoke-legacy",
      count: 9, target: null, unit: "examples", tokens: null, token_target: null, quality_status: "NOT_EVALUATED",
      mixture_eligible: false, mixture_block_reason: "缺少登记 lineage、processing report 或 canonical_path",
      source: { kind: "open_source", name: "旧 smoke 数据" } }] });
  return { builds, issues: [], scanned_at: "2026-10-24T00:00:00Z", read_only: false, roots: [] };
}

function preflight() {
  return {
    stage: "sft", strategy: "concat", total_records: 5, config_sha256: "c".repeat(64), warnings: [],
    parents: [
      { dataset_id: datasetA, record_count: 2, selected_records: 2 },
      { dataset_id: datasetB, record_count: 3, selected_records: 3 },
    ],
  };
}

function buildResult() {
  return { dataset_id: "ds-sft-mixture-result", record_count: 5, quality_status: "report_only" };
}

function setupFetch({ preflightStatus = 200, includeLegacy = false } = {}) {
  const requests: { url: string; init?: RequestInit }[] = [];
  let built = false;
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    requests.push({ url, init });
    if (url === "/api/data") return jsonResponse(catalog(built, includeLegacy));
    if (url === "/api/data/capabilities") return jsonResponse({ capabilities: [] });
    if (url.startsWith("/api/data/import/preview?")) return jsonResponse({
      source_format: new URL(url, "http://localhost").searchParams.get("source_format"), valid: true,
      source_record_count: 1, output_record_count: 1, converter_version: "sft-source-format-v1",
      samples: [{ id: "sft-src-fixture", messages: [{ role: "user", content: "Q" }, { role: "assistant", content: "A" }] }],
      errors: [], mapping: { human: "user", gpt: "assistant" }, disclaimer: "结构预览，不是正确性验证",
    });
    if (url.startsWith("/api/data/import?")) return new Promise((resolve) => {
      window.setTimeout(() => resolve(new Response(JSON.stringify({ dataset_id: "ds-sft-upload-browser", record_count: 1, quality_status: "NOT_EVALUATED" }), {
        status: 201, headers: { "Content-Type": "application/json" },
      })), 30);
    });
    if (url === "/api/data/grpo-seeds") return jsonResponse({ candidates: [{ dataset_id: "ds-openr1-seed", display_name: "可信 OpenR1 SFT seed", eligible_count: 6, sample_count: 8, revision: "a".repeat(40), license_status: "not_provided", evidence_scope: "上游 OpenR1 标签；未在本系统独立重验" }] });
    if (url === "/api/data/grpo-seeds/preflight") return jsonResponse({ requested_count: 3, eligible_count: 6, available_count: 3, shortfall: 0, reference_evidence: "OpenR1 upstream evidence", limitations: ["上游标签非独立重验"] });
    if (url === "/api/data/grpo-seeds/build") return jsonResponse({ dataset_id: "ds-grpo-derived", record_count: 3, quality_status: "report_only" }, 201);
    if (url === "/api/synthetic/sft/capabilities") return jsonResponse({ enabled: false });
    if (url === "/api/data/mixtures/preflight") {
      return jsonResponse(preflightStatus === 200 ? preflight() : { detail: "SFT 数据集阶段冲突，请重新选择。" }, preflightStatus);
    }
    if (url === "/api/data/mixtures/build") { built = true; return jsonResponse(buildResult(), 201); }
    if (url === "/api/data/build-mixture") return jsonResponse({ steps: [], related_runs: [] });
    return jsonResponse({});
  });
  vi.stubGlobal("fetch", fetchMock);
  return { fetchMock, requests };
}

describe("DatasetWorkbench mixture flow", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/?page=workbench");
    window.localStorage.clear();
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => window.setTimeout(() => callback(0), 0));
  });

  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("previews, explicitly builds, and hands off the registered mixture without building early", async () => {
    const user = userEvent.setup();
    const { requests } = setupFetch();
    const onUseDataset = vi.fn();
    vi.stubGlobal("confirm", vi.fn(() => true));
    render(<DatasetWorkbench onViewEvidence={vi.fn()} onUseDataset={onUseDataset} />);

    const group = await screen.findByRole("group", { name: /选择数据集/ });
    const checkboxes = within(group).getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(2);
    await user.clear(screen.getByRole("textbox", { name: "数据集名称" }));
    await user.type(screen.getByRole("textbox", { name: "数据集名称" }), "数学推理 SFT 混合 v2");
    await user.click(checkboxes[0]);
    await user.click(checkboxes[1]);

    expect(requests.some((request) => request.url === "/api/data/mixtures/build")).toBe(false);
    await user.click(screen.getByRole("button", { name: "预检组合" }));
    const preflightRequest = requests.find((request) => request.url === "/api/data/mixtures/preflight");
    expect(JSON.parse(String(preflightRequest?.init?.body)).display_name).toBe("数学推理 SFT 混合 v2");
    expect(await screen.findByText(/预检通过 · 5 条输出/)).toBeTruthy();
    expect(requests.some((request) => request.url === "/api/data/mixtures/build")).toBe(false);

    await user.click(screen.getByRole("button", { name: "确认创建组合数据集" }));
    await waitFor(() => expect(screen.getByText(/组合数据集已创建并完成质检报告/)).toBeTruthy());
    expect(onUseDataset).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: /去训练工作台使用/ }));
    expect(onUseDataset).toHaveBeenCalledWith("ds-sft-mixture-result", "sft");

    const resultCardHeading = await screen.findByRole("heading", { name: "组合结果" });
    const resultCard = resultCardHeading.closest("article");
    expect(resultCard).not.toBeNull();
    await user.click(within(resultCard as HTMLElement).getByRole("button", { name: "查看质量与血缘详情" }));
    await waitFor(() => expect(requests.some((request) => request.url === "/api/data/build-mixture")).toBe(true));
    expect(await within(resultCard as HTMLElement).findByText("质量评估")).toBeTruthy();
    expect(within(resultCard as HTMLElement).getByText("处理血缘")).toBeTruthy();
    await user.click(within(resultCard as HTMLElement).getByRole("button", { name: /打开完整处理报告/ }));
    expect(onUseDataset).toHaveBeenCalledTimes(1);
  });

  it("opens the GRPO seed builder from a launcher deep link and hands back the exact registered dataset", async () => {
    const user = userEvent.setup();
    setupFetch();
    const onUseDataset = vi.fn();
    vi.stubGlobal("confirm", vi.fn(() => true));
    const scrollIntoView = vi.fn();
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", { configurable: true, value: scrollIntoView });
    window.history.replaceState({}, "", "/?page=workbench&stage=grpo&prepare_grpo=1&return_to=launcher&profile_id=grpo-lowmem-smoke-v1&parent_model_id=model-sft");
    render(<DatasetWorkbench onViewEvidence={vi.fn()} onUseDataset={onUseDataset} />);

    const heading = await screen.findByRole("heading", { name: "从可信训练种子派生 GRPO 数据" });
    expect(scrollIntoView).toHaveBeenCalledWith({ behavior: "smooth", block: "start" });
    expect(window.location.search).toContain("profile_id=grpo-lowmem-smoke-v1");
    expect(await screen.findByText(/许可提醒需按父数据来源自行确认/)).toBeTruthy();
    await user.click(await screen.findByRole("checkbox", { name: /可信 OpenR1 SFT seed/ }));
    await user.click(screen.getByRole("button", { name: "预检 GRPO 数据" }));
    await screen.findByRole("status", { name: "" }).catch(() => undefined);
    await waitFor(() => expect(screen.getByText(/可用 3 \/ 目标 3 条/)).toBeTruthy());
    await user.click(screen.getByRole("button", { name: "确认派生并登记" }));
    await screen.findByText(/GRPO 数据已派生、登记并生成质量\/血缘报告/);
    await user.click(screen.getByRole("button", { name: "去 GRPO 训练工作台使用 →" }));
    expect(onUseDataset).toHaveBeenCalledWith("ds-grpo-derived", "grpo");
    expect(heading).toBeTruthy();
  });

  it("restores the GRPO seed selection and requested count after a page refresh", async () => {
    setupFetch();
    window.localStorage.setItem("onegpu.grpo-seed-draft.v1", JSON.stringify({ ids: ["ds-openr1-seed"], target: 3, seed: 42 }));
    window.history.replaceState({}, "", "/?page=workbench&stage=grpo&prepare_grpo=1&return_to=launcher");
    render(<DatasetWorkbench onViewEvidence={vi.fn()} />);
    const candidate = await screen.findByRole("checkbox", { name: /可信 OpenR1 SFT seed/ });
    await waitFor(() => expect((candidate as HTMLInputElement).checked).toBe(true));
    expect((screen.getByLabelText("目标生成条数") as HTMLInputElement).value).toBe("3");
    expect((screen.getByLabelText("确定性随机种子") as HTMLInputElement).value).toBe("42");
    expect(await screen.findByText(/许可：not_provided/)).toBeTruthy();
  });

  it("explains preflight conflicts and never creates an artifact on a failed check", async () => {
    const user = userEvent.setup();
    const { requests } = setupFetch({ preflightStatus: 422 });
    render(<DatasetWorkbench onViewEvidence={vi.fn()} />);

    const group = await screen.findByRole("group", { name: /选择数据集/ });
    const checkboxes = within(group).getAllByRole("checkbox");
    await user.click(checkboxes[0]);
    await user.click(checkboxes[1]);
    await user.click(screen.getByRole("button", { name: "预检组合" }));

    expect((await screen.findByRole("alert")).textContent).toContain("SFT 数据集阶段冲突");
    expect(requests.some((request) => request.url === "/api/data/mixtures/build")).toBe(false);
  });

  it("disables legacy datasets without registration evidence before preflight", async () => {
    const user = userEvent.setup();
    const { fetchMock, requests } = setupFetch({ includeLegacy: true });
    render(<DatasetWorkbench onViewEvidence={vi.fn()} />);

    const group = await screen.findByRole("group", { name: /选择数据集/ });
    const checkboxes = within(group).getAllByRole("checkbox");
    expect(checkboxes).toHaveLength(3);
    expect(checkboxes[2].hasAttribute("disabled")).toBe(true);
    expect(within(group).getByText(/缺少登记 lineage、processing report 或 canonical_path/)).toBeTruthy();
    await user.click(checkboxes[0]);
    await user.click(checkboxes[2]);
    expect(requests.some((request) => request.url === "/api/data/mixtures/preflight")).toBe(false);
    expect(fetchMock).toHaveBeenCalled();
  });
});

describe("DatasetWorkbench source format import", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/?page=workbench");
    window.localStorage.clear();
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("requires an explicit format preview and invalidates preview when format changes", async () => {
    const user = userEvent.setup();
    const { requests } = setupFetch();
    render(<DatasetWorkbench onViewEvidence={vi.fn()} />);
    const format = await screen.findByLabelText(/来源格式/);
    expect((format as HTMLSelectElement).options).toHaveLength(4);
    expect(Array.from((format as HTMLSelectElement).options).map((option) => option.value)).toEqual([
      "canonical-v1", "alpaca-instruction-v1", "messages-role-content-v1", "sharegpt-conversations-v1",
    ]);
    await user.click(format);
    await user.keyboard("{ArrowDown}{Enter}");
    expect(document.activeElement).toBe(format);
    await user.selectOptions(format, "sharegpt-conversations-v1");
    const file = new File([JSON.stringify({ conversations: [{ from: "human", value: "Q" }, { from: "gpt", value: "A" }] }) + "\n"], "sharegpt.jsonl", { type: "application/x-ndjson" });
    await user.upload(screen.getByLabelText(/SFT JSONL 文件/), file);
    const nextButton = screen.getByRole("button", { name: /继续核对来源与许可/ });
    expect((nextButton as HTMLButtonElement).disabled).toBe(true);
    const previewButton = screen.getByRole("button", { name: "1. 预览并校验" });
    previewButton.focus();
    await user.keyboard("{Enter}");
    expect(await screen.findByText(/预览通过 · 尚未登记/)).toBeTruthy();
    expect((nextButton as HTMLButtonElement).disabled).toBe(false);
    const previewRequest = requests.find((request) => request.url.startsWith("/api/data/import/preview?"));
    expect(previewRequest?.url).toContain("source_format=sharegpt-conversations-v1");
    expect(requests.some((request) => request.url.startsWith("/api/data/import?"))).toBe(false);
    await user.selectOptions(format, "alpaca-instruction-v1");
    expect((nextButton as HTMLButtonElement).disabled).toBe(true);
    await user.click(screen.getByRole("button", { name: "1. 预览并校验" }));
    expect(await screen.findByText(/预览通过 · 尚未登记/)).toBeTruthy();
    await user.click(nextButton);
    const licenseSelect = screen.getByLabelText(/许可证状态/);
    expect(Array.from((licenseSelect as HTMLSelectElement).options).map((option) => option.value)).not.toContain("verified");
    expect(screen.getByText(/不会把许可证状态升级为已核验/)).toBeTruthy();
    const registerButton = screen.getByRole("button", { name: "确认来源声明并登记数据集" });
    await user.dblClick(registerButton);
    await screen.findByText("数据集已登记");
    const importRequests = requests.filter((request) => request.url.startsWith("/api/data/import?"));
    expect(importRequests).toHaveLength(1);
    expect(importRequests[0].url).toContain("source_format=alpaca-instruction-v1");
  });

  it("prevents an invalid conversion preview from reaching registration", async () => {
    const user = userEvent.setup();
    const { requests, fetchMock } = setupFetch();
    fetchMock.mockImplementation((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      requests.push({ url, init });
      if (url === "/api/data") return jsonResponse(catalog());
      if (url === "/api/data/capabilities") return jsonResponse({ capabilities: [] });
      if (url.startsWith("/api/data/import/preview?")) return jsonResponse({ valid: false, source_format: "messages-role-content-v1", source_record_count: 1, output_record_count: 0, errors: [{ line: 1, field: "messages[0].role", message: "unsupported role" }], samples: [], disclaimer: "no registration" });
      return jsonResponse({});
    });
    render(<DatasetWorkbench onViewEvidence={vi.fn()} />);
    await user.selectOptions(await screen.findByLabelText(/来源格式/), "messages-role-content-v1");
    const file = new File(["{}\n"], "invalid.jsonl", { type: "application/x-ndjson" });
    await user.upload(screen.getByLabelText(/SFT JSONL 文件/), file);
    await user.click(screen.getByRole("button", { name: "1. 预览并校验" }));
    expect(await screen.findByText(/预览发现问题 · 尚未登记/)).toBeTruthy();
    expect((screen.getByRole("button", { name: /继续核对来源与许可/ }) as HTMLButtonElement).disabled).toBe(true);
    expect(requests.some((request) => request.url.startsWith("/api/data/import?"))).toBe(false);
  });

  it("previews and explicitly registers an evaluation-only objective suite", async () => {
    const user = userEvent.setup();
    const { requests } = setupFetch();
    render(<DatasetWorkbench onViewEvidence={vi.fn()} />);
    await user.selectOptions(await screen.findByLabelText(/数据用途/), "eval");
    expect((screen.getByLabelText(/来源格式/) as HTMLSelectElement).value).toBe("objective-choice-jsonl-v1");
    const file = new File([JSON.stringify({ questionType: "single_choice", question: "1+1?", options: '["1","2"]', correctAnswer: "B" }) + "\n"], "exam.jsonl", { type: "application/x-ndjson" });
    await user.upload(screen.getByLabelText(/客观题评测 JSONL 文件/), file);
    await user.click(screen.getByRole("button", { name: "1. 预览并校验" }));
    expect(await screen.findByText(/预览通过 · 尚未登记/)).toBeTruthy();
    const previewRequest = requests.find((request) => request.url.startsWith("/api/data/import/preview?"));
    expect(previewRequest?.url).toContain("stage=eval");
    expect(previewRequest?.url).toContain("objective-choice-jsonl-v1");
    expect(requests.some((request) => request.url.startsWith("/api/data/import?"))).toBe(false);
    await user.click(screen.getByRole("button", { name: /继续核对来源与许可/ }));
    expect(screen.getByText(/不会把许可证状态升级为已核验/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: "确认来源声明并登记数据集" }));
    expect(await screen.findByText("评测数据集已登记")).toBeTruthy();
    const registration = requests.find((request) => request.url.startsWith("/api/data/import?"));
    expect(registration?.url).toContain("stage=eval");
    expect(registration?.url).toContain("source_format=objective-choice-jsonl-v1");
  });
});

describe("DatasetWorkbench ordinary GRPO seed derivation", () => {
  beforeEach(() => {
    window.history.replaceState({}, "", "/?page=workbench");
    window.localStorage.clear();
    vi.stubGlobal("requestAnimationFrame", (callback: FrameRequestCallback) => window.setTimeout(() => callback(0), 0));
  });
  afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

  it("preflights explicitly, registers only after confirmation, and hands off the exact dataset ID", async () => {
    const user = userEvent.setup();
    const { requests } = setupFetch();
    vi.stubGlobal("confirm", vi.fn(() => true));
    const onUseDataset = vi.fn();
    render(<DatasetWorkbench onViewEvidence={vi.fn()} onUseDataset={onUseDataset} />);

    const candidate = await screen.findByText("可信 OpenR1 SFT seed");
    const group = candidate.closest("fieldset");
    expect(group).not.toBeNull();
    await user.click(within(group as HTMLElement).getByRole("checkbox"));
    const target = screen.getByRole("spinbutton", { name: "目标生成条数" });
    await user.clear(target);
    await user.type(target, "3");
    expect(requests.some((request) => request.url === "/api/data/grpo-seeds/build")).toBe(false);
    await user.click(screen.getByRole("button", { name: "预检 GRPO 数据" }));
    expect(await screen.findByText(/可用 3 \/ 目标 3 条/)).toBeTruthy();
    expect(requests.some((request) => request.url === "/api/data/grpo-seeds/build")).toBe(false);
    await user.click(screen.getByRole("button", { name: "确认派生并登记" }));
    expect(await screen.findByText(/GRPO 数据已派生、登记并生成质量\/血缘报告/)).toBeTruthy();
    await user.click(screen.getByRole("button", { name: /去 GRPO 训练工作台使用/ }));
    expect(onUseDataset).toHaveBeenCalledWith("ds-grpo-derived", "grpo");
  });

  it("restores the selected seeds and requested count after a page remount", async () => {
    const user = userEvent.setup();
    setupFetch();
    const first = render(<DatasetWorkbench onViewEvidence={vi.fn()} />);
    const group = await screen.findByRole("group", { name: /可用 SFT 种子/ });
    await user.click(within(group).getByRole("checkbox"));
    const target = screen.getByRole("spinbutton", { name: "目标生成条数" });
    await user.clear(target);
    await user.type(target, "17");
    first.unmount();

    render(<DatasetWorkbench onViewEvidence={vi.fn()} />);
    const restoredGroup = await screen.findByRole("group", { name: /可用 SFT 种子/ });
    expect((within(restoredGroup).getByRole("checkbox") as HTMLInputElement).checked).toBe(true);
    expect((screen.getByRole("spinbutton", { name: "目标生成条数" }) as HTMLInputElement).value).toBe("17");
  });
});
