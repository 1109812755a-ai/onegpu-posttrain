import { useEffect, useMemo, useReducer, useRef, useState } from "react";
import type { BuildDetail, Catalog, Dataset } from "./types";
import { datasetImportUrl, MAX_DATASET_IMPORT_BYTES, postDatasetImport, previewDatasetImport } from "./datasetImport.mjs";
import { datasetPreviewReducer, fetchDatasetPreview } from "./datasetPreview.mjs";
import type { DatasetPreviewState } from "./datasetPreview.mjs";
import { SFTGenerationPanel } from "./SFTGenerationPanel";
import { CPTGenerationPanel } from "./CPTGenerationPanel";
import { PreferenceGenerationPanel } from "./PreferenceGenerationPanel";
import { GRPOGenerationPanel } from "./GRPOGenerationPanel";
import { TargetedExperiments } from "./TargetedExperiments";
import { loadDataWorkbenchResources } from "./dataWorkbenchResources.mjs";

const QUALITY_NAMES: Record<string, string> = {
  canonical_schema_valid: "标准格式有效率",
  schema_valid_rate: "标准格式有效率",
  duplicate_record_ids: "重复记录 ID",
  train_eval_overlap: "训练 / 评测重叠",
  near_duplicate_rate: "近重复率",
  semantic_correctness: "独立语义正确性",
  complete_rate: "完整推理比例",
  answer_parse_rate: "答案可解析率",
  nonempty_rate: "非空文本比例",
  chosen_correctness_rate: "Chosen 正确率",
  rejected_incorrectness_rate: "Rejected 错误率",
  pair_yield_rate: "偏好对产出率",
  truncation_rate: "生成截断率",
};
const QUALITY_STATUS: Record<string, string> = {
  measured: "已测量", estimated: "估算", not_measured: "未测量",
  not_applicable: "不适用", PASS: "通过", FAIL: "失败",
};
const GATE_LABEL: Record<string, string> = {
  PASS: "质量检查通过", FAIL: "质量检查未通过", WARN: "需注意", NOT_EVALUATED: "质量未评定",
  report_only: "仅报告", not_evaluated: "质量未评定", partial: "部分评估",
};

type Capability = {
  source_id: string;
  source_kind: string;
  supported_stages: string[];
  input_format: string;
  requires_network: boolean;
  requires_credentials: boolean;
  status: string;
  reason?: string | null;
};

const STAGES = ["cpt", "sft", "preference", "grpo"] as const;
const IMPORT_STAGES = [...STAGES, "eval"] as const;
type ImportStage = (typeof IMPORT_STAGES)[number];
const STAGE_NAMES: Record<string, string> = {
  cpt: "CPT 继续预训练",
  sft: "SFT 监督微调",
  preference: "Preference（DPO / SimPO）",
  grpo: "GRPO",
  eval: "客观题评测集（仅评测）",
};
const KIND_NAMES: Record<string, string> = {
  local_upload: "本地导入",
  open_source: "开源数据",
  model_generated: "模型生成",
  dataset_mixture: "派生组合",
  dataset_derived: "规则派生",
  unknown: "来源未记录",
};

function sourceOf(dataset: Dataset) {
  return dataset.source || { kind: "unknown", source_id: null, name: "未记录", license_status: "not_provided" };
}

function qualityOf(dataset: Dataset) {
  return dataset.quality?.gate?.summary_status || dataset.quality_status || "NOT_EVALUATED";
}

function readGrpoSeedDraft() {
  try {
    const value = JSON.parse(window.localStorage.getItem("onegpu.grpo-seed-draft.v1") || "null");
    if (!value || typeof value !== "object") return { ids: [] as string[], target: 100, seed: 42 };
    return {
      ids: Array.isArray(value.ids) ? value.ids.filter((item: unknown): item is string => typeof item === "string" && item.length <= 200) : [],
      target: Number.isInteger(value.target) && value.target > 0 && value.target <= 250000 ? value.target : 100,
      seed: Number.isInteger(value.seed) && value.seed >= 0 && value.seed <= 2147483647 ? value.seed : 42,
    };
  } catch { return { ids: [] as string[], target: 100, seed: 42 }; }
}

export function DatasetWorkbench({ onViewEvidence, onOpenTargeted, onOpenRun, onUseDataset }: { onViewEvidence: (buildId: string, stage: string) => void; onOpenTargeted?: () => void; onOpenRun?: (runId: string) => void; onUseDataset?: (datasetId: string, stage: string) => void }) {
  const targetedGeneration = new URLSearchParams(window.location.search).get("targeted_generation") === "1";
  const [generationStage, setGenerationStage] = useState<"cpt" | "sft" | "preference" | "grpo">("sft");
  const prepareGrpo = new URLSearchParams(window.location.search).get("prepare_grpo") === "1";
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [capabilities, setCapabilities] = useState<Capability[]>([]);
  const [error, setError] = useState("");
  const [capabilityError, setCapabilityError] = useState("");
  const [loading, setLoading] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [view, setView] = useState<"datasets" | "sources">("datasets");
  const [stageFilter, setStageFilter] = useState("all");
  const [kindFilter, setKindFilter] = useState("all");
  const [stage, setStage] = useState<ImportStage>("sft");
  const [file, setFile] = useState<File | null>(null);
  const [sourceFormat, setSourceFormat] = useState("canonical-v1");
  const [importPreview, setImportPreview] = useState<Record<string, unknown> | null>(null);
  const [previewBusy, setPreviewBusy] = useState(false);
  const [sourceName, setSourceName] = useState("");
  const [sourceUri, setSourceUri] = useState("");
  const [revision, setRevision] = useState("");
  const [revisionStatus, setRevisionStatus] = useState("not_provided");
  const [licenseId, setLicenseId] = useState("");
  const [licenseStatus, setLicenseStatus] = useState("not_provided");
  const [attribution, setAttribution] = useState("");
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [submitError, setSubmitError] = useState("");
  const uploadController = useRef<AbortController | null>(null);
  const importPreviewController = useRef<AbortController | null>(null);
  const [previews, dispatchPreview] = useReducer(datasetPreviewReducer, {} as DatasetPreviewState);
  const [expandedDatasetId, setExpandedDatasetId] = useState<string | null>(() => new URLSearchParams(window.location.search).get("dataset_id"));
  const [detailRefresh, setDetailRefresh] = useState(0);
  const [buildEvidence, setBuildEvidence] = useState<Record<string, { loading: boolean; error?: string; detail?: BuildDetail }>>({});
  const previewControllers = useRef(new Map<string, AbortController>());
  const [mixtureStage, setMixtureStage] = useState<(typeof STAGES)[number]>(() => {
    const requested = new URLSearchParams(window.location.search).get("compose_stage");
    return STAGES.includes(requested as (typeof STAGES)[number]) ? requested as (typeof STAGES)[number] : "sft";
  });
  const [mixtureIds, setMixtureIds] = useState<string[]>([]);
  const [mixtureName, setMixtureName] = useState("SFT 组合数据集");
  const [mixtureStrategy, setMixtureStrategy] = useState<"concat" | "weighted_sample">("concat");
  const [mixtureBudget, setMixtureBudget] = useState(100);
  const [mixtureSeed, setMixtureSeed] = useState(42);
  const [mixtureWeights, setMixtureWeights] = useState<Record<string, number>>({});
  const [mixturePreflight, setMixturePreflight] = useState<Record<string, unknown> | null>(null);
  const [mixtureResult, setMixtureResult] = useState<Record<string, unknown> | null>(null);
  const [mixtureError, setMixtureError] = useState("");
  const [mixtureBusy, setMixtureBusy] = useState(false);
  const [grpoSeedCandidates, setGrpoSeedCandidates] = useState<Array<Record<string, unknown>>>([]);
  const [grpoSeedLoading, setGrpoSeedLoading] = useState(false);
  const [grpoSeedError, setGrpoSeedError] = useState("");
  const [grpoSeedIds, setGrpoSeedIds] = useState<string[]>(() => readGrpoSeedDraft().ids);
  const [grpoSeedTarget, setGrpoSeedTarget] = useState(() => readGrpoSeedDraft().target);
  const [grpoSeedValue, setGrpoSeedValue] = useState(() => readGrpoSeedDraft().seed);
  const [grpoSeedPreflight, setGrpoSeedPreflight] = useState<Record<string, unknown> | null>(null);
  const [grpoSeedResult, setGrpoSeedResult] = useState<Record<string, unknown> | null>(null);
  const [grpoSeedBusy, setGrpoSeedBusy] = useState(false);

  useEffect(() => {
    if (!prepareGrpo) return;
    const heading = document.getElementById("grpo-seed-heading");
    heading?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    heading?.focus?.({ preventScroll: true });
  }, [prepareGrpo]);

  useEffect(() => () => {
    previewControllers.current.forEach((controller) => controller.abort());
    previewControllers.current.clear();
  }, []);

  useEffect(() => {
    if (new URLSearchParams(window.location.search).has("compose_stage")) {
      requestAnimationFrame(() => document.getElementById("dataset-mixture-card")?.scrollIntoView({ behavior: "smooth", block: "start" }));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    setCapabilityError("");
    loadDataWorkbenchResources(fetch, controller.signal).then(({ catalog: catalogResult, capabilities: capabilityResult }) => {
      if (controller.signal.aborted) return;
      if (catalogResult.status === "fulfilled") {
        setCatalog(catalogResult.value as Catalog);
      } else {
        setError(catalogResult.reason instanceof Error ? catalogResult.reason.message : "数据目录读取失败");
      }
      if (capabilityResult.status === "fulfilled") {
        setCapabilities((capabilityResult.value as { capabilities: Capability[] }).capabilities);
      } else {
        setCapabilityError(capabilityResult.reason instanceof Error ? capabilityResult.reason.message : "来源能力读取失败");
      }
    }).finally(() => {
      if (!controller.signal.aborted) setLoading(false);
    });
    return () => controller.abort();
  }, [refresh]);

  useEffect(() => {
    try { window.localStorage.setItem("onegpu.grpo-seed-draft.v1", JSON.stringify({ ids: grpoSeedIds, target: grpoSeedTarget, seed: grpoSeedValue })); }
    catch { /* Local persistence is a convenience; server validation remains authoritative. */ }
  }, [grpoSeedIds, grpoSeedTarget, grpoSeedValue]);

  useEffect(() => {
    const controller = new AbortController();
    setGrpoSeedLoading(true); setGrpoSeedError("");
    fetch("/api/data/grpo-seeds", { signal: controller.signal }).then(async (response) => {
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `GRPO 种子读取失败 (${response.status})`);
      return payload as { candidates?: Array<Record<string, unknown>> };
    }).then((payload) => { if (!controller.signal.aborted) setGrpoSeedCandidates(payload.candidates || []); })
      .catch((reason: unknown) => { if (!controller.signal.aborted) setGrpoSeedError(reason instanceof Error ? reason.message : "GRPO 种子读取失败"); })
      .finally(() => { if (!controller.signal.aborted) setGrpoSeedLoading(false); });
    return () => controller.abort();
  }, [refresh]);

  const datasets = useMemo(() => (catalog?.builds || []).flatMap((build) => build.datasets.map((dataset) => ({
    ...dataset, buildId: build.id, buildName: build.name, buildPath: build.path,
  }))), [catalog]);
  const filtered = datasets.filter((dataset) => {
    const source = sourceOf(dataset);
    return (stageFilter === "all" || dataset.stage === stageFilter)
      && (kindFilter === "all" || source.kind === kindFilter);
  });
  const expandedDataset = datasets.find((item) => item.dataset_id === expandedDatasetId);
  useEffect(() => {
    if (!expandedDataset) return;
    const buildId = expandedDataset.buildId;
    const controller = new AbortController();
    setBuildEvidence((current) => ({ ...current, [buildId]: { loading: true } }));
    fetch(`/api/data/${encodeURIComponent(buildId)}`, { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`处理与血缘证据读取失败 (${response.status})`);
        return response.json() as Promise<BuildDetail>;
      })
      .then((detail) => setBuildEvidence((current) => ({ ...current, [buildId]: { loading: false, detail } })))
      .catch((reason: unknown) => {
        if (!controller.signal.aborted) setBuildEvidence((current) => ({ ...current, [buildId]: { loading: false, error: reason instanceof Error ? reason.message : "证据读取失败" } }));
      });
    return () => controller.abort();
  }, [expandedDataset?.buildId, expandedDatasetId, detailRefresh]);

  async function submitImport() {
    if (!file) return;
    const controller = new AbortController();
    uploadController.current = controller;
    setSubmitting(true);
    setSubmitError("");
    try {
      const payload = await postDatasetImport({
        url: datasetImportUrl({ stage, sourceFormat, sourceName, sourceUri, revision, revisionStatus, licenseId, licenseStatus, attribution }),
        file, signal: controller.signal,
      });
      setResult(payload);
      setStep(3);
      setRefresh((value) => value + 1);
    } catch (reason) {
      if (reason instanceof DOMException && reason.name === "AbortError") setSubmitError("已取消上传；文件和来源信息仍保留，可修改后重试。");
      else setSubmitError(reason instanceof Error ? reason.message : "登记失败，请检查文件后重试");
    } finally {
      uploadController.current = null;
      setSubmitting(false);
    }
  }

  async function previewImport() {
    if (!file) return;
    importPreviewController.current?.abort();
    const controller = new AbortController();
    importPreviewController.current = controller;
    setPreviewBusy(true); setSubmitError(""); setImportPreview(null);
    try {
      const payload = await previewDatasetImport({ stage, sourceFormat, file, signal: controller.signal });
      setImportPreview(payload);
      if (payload.valid !== true) setSubmitError("转换预览发现结构错误。修正源文件后重新预览；当前文件不能登记。");
    } catch (reason) {
      if (!controller.signal.aborted) setSubmitError(reason instanceof Error ? reason.message : "转换预览失败，请重试");
    } finally {
      if (importPreviewController.current === controller) importPreviewController.current = null;
      setPreviewBusy(false);
    }
  }

  const mixtureCandidates = datasets.filter((dataset) => dataset.stage === mixtureStage);
  const eligibleMixtureIds = new Set(mixtureCandidates.filter((dataset) => dataset.mixture_eligible !== false).map((dataset) => dataset.dataset_id));
  function mixturePayload() {
    const datasetIds = mixtureIds.filter((id) => eligibleMixtureIds.has(id));
    return {
      stage: mixtureStage, display_name: mixtureName, dataset_ids: datasetIds, strategy: mixtureStrategy,
      ...(mixtureStrategy === "weighted_sample" ? {
        budget: mixtureBudget, seed: mixtureSeed,
        weights: Object.fromEntries(datasetIds.map((id) => [id, mixtureWeights[id] || 1])),
      } : {}),
    };
  }
  async function runMixturePreflight() {
    setMixtureBusy(true); setMixtureError(""); setMixtureResult(null); setMixturePreflight(null);
    try {
      const response = await fetch("/api/data/mixtures/preflight", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(mixturePayload()) });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `组合预检失败 (${response.status})`);
      setMixturePreflight(payload);
    } catch (reason) { setMixtureError(reason instanceof Error ? reason.message : "组合预检失败"); }
    finally { setMixtureBusy(false); }
  }
  async function buildMixture() {
    if (!mixturePreflight || !window.confirm(`确认创建 ${mixtureStage.toUpperCase()} 组合数据集？将生成独立数据版本并重新质检，不会修改父数据集。`)) return;
    setMixtureBusy(true); setMixtureError("");
    try {
      const response = await fetch("/api/data/mixtures/build", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(mixturePayload()) });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `组合构建失败 (${response.status})`);
      setMixtureResult(payload); setMixturePreflight(null); setRefresh((value) => value + 1);
    } catch (reason) { setMixtureError(reason instanceof Error ? reason.message : "组合构建失败"); }
    finally { setMixtureBusy(false); }
  }

  function grpoSeedPayload() { return { dataset_ids: grpoSeedIds, target_count: grpoSeedTarget, seed: grpoSeedValue }; }
  async function preflightGrpoSeed() {
    setGrpoSeedBusy(true); setGrpoSeedError(""); setGrpoSeedResult(null); setGrpoSeedPreflight(null);
    try {
      const response = await fetch("/api/data/grpo-seeds/preflight", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(grpoSeedPayload()) });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `GRPO 种子预检失败 (${response.status})`);
      setGrpoSeedPreflight(payload);
    } catch (reason) { setGrpoSeedError(reason instanceof Error ? reason.message : "GRPO 种子预检失败"); }
    finally { setGrpoSeedBusy(false); }
  }
  async function buildGrpoSeed() {
    if (!grpoSeedPreflight || !window.confirm(`确认从 ${grpoSeedIds.length} 个已登记 SFT 数据集派生 ${grpoSeedTarget} 条 GRPO train_seed？将创建新版本并重新质检。`)) return;
    setGrpoSeedBusy(true); setGrpoSeedError("");
    try {
      const response = await fetch("/api/data/grpo-seeds/build", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(grpoSeedPayload()) });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `GRPO 数据派生失败 (${response.status})`);
      setGrpoSeedResult(payload); setGrpoSeedPreflight(null); setRefresh((value) => value + 1);
    } catch (reason) { setGrpoSeedError(reason instanceof Error ? reason.message : "GRPO 数据派生失败"); }
    finally { setGrpoSeedBusy(false); }
  }

  async function openPreview(datasetId: string) {
    previewControllers.current.get(datasetId)?.abort();
    const controller = new AbortController();
    previewControllers.current.set(datasetId, controller);
    dispatchPreview({ type: "loading", datasetId });
    try {
      const data = await fetchDatasetPreview(datasetId, { signal: controller.signal });
      dispatchPreview({ type: "success", datasetId, data });
    } catch (reason) {
      if (controller.signal.aborted) return;
      dispatchPreview({ type: "error", datasetId, message: reason instanceof Error ? reason.message : "样本预览失败，请重试" });
    } finally {
      if (previewControllers.current.get(datasetId) === controller) previewControllers.current.delete(datasetId);
    }
  }

  function closePreview(datasetId: string) {
    previewControllers.current.get(datasetId)?.abort();
    previewControllers.current.delete(datasetId);
    dispatchPreview({ type: "close", datasetId });
  }

  return <main className="dataset-workbench-page">
    <header className="dataset-workbench-hero">
      <div>
        <div className="eyebrow">DATA SOURCES & DATASETS</div>
        <h1>数据来源与数据集工作台</h1>
        <p>来源与训练阶段分开管理；每个已登记数据集都带有明确 ID、内容摘要、质量状态和可追溯血缘。</p>
      </div>
      <button className="secondary-button" onClick={() => setRefresh((value) => value + 1)} disabled={loading}>刷新目录</button>
    </header>

    <div className="dataset-workbench-tabs" role="tablist" aria-label="数据工作台视图">
      <button role="tab" aria-selected={view === "datasets"} className={view === "datasets" ? "active" : ""} onClick={() => setView("datasets")}>已登记数据集 <span>{datasets.length}</span></button>
      <button role="tab" aria-selected={view === "sources"} className={view === "sources" ? "active" : ""} onClick={() => setView("sources")}>数据来源与阶段能力</button>
    </div>

    {view === "datasets" && error && <div className="dataset-workbench-alert" role="alert">{error}<button onClick={() => setRefresh((value) => value + 1)}>重试</button></div>}
    {view === "sources" && capabilityError && <div className="dataset-workbench-alert" role="alert">{capabilityError}<button onClick={() => setRefresh((value) => value + 1)}>重试</button></div>}
    {view === "datasets" && (catalog?.issues.length || 0) > 0 && <div className="dataset-workbench-alert" role="status"><strong>目录部分可读：</strong>{catalog?.issues.length} 个数据来源或报告未能读取。<details><summary>查看读取问题</summary><ul>{catalog?.issues.map((issue, index) => <li key={`${issue.path}-${index}`}><code>{issue.path}</code> · {issue.message}</li>)}</ul></details></div>}
    {view === "sources" ? loading && capabilities.length === 0 && !capabilityError ? <div className="workbench-loading" role="status">正在读取数据来源能力…</div> : capabilityError ? <div className="workbench-empty" role="alert"><strong>来源能力暂不可用</strong><span>{capabilityError}</span><button className="secondary-button" onClick={() => setRefresh((value) => value + 1)}>重试</button></div> : capabilities.length === 0 ? <div className="workbench-empty"><strong>尚未登记可用的数据来源</strong><span>可以直接导入 canonical JSONL；没有来源配置不影响已登记数据集。</span></div> : <section className="source-capability-grid" aria-label="数据来源能力">
      {capabilities.map((item) => <article className="source-capability-card" key={item.source_id}>
        <div className="source-capability-head"><span className={`source-kind source-kind-${item.source_kind}`}>{KIND_NAMES[item.source_kind] || item.source_kind}</span><span className={`source-capability-status ${item.status}`}>{item.status === "available" ? "可用" : "暂不支持"}</span></div>
        <h2>{item.source_id}</h2>
        <p>{item.reason || `${item.input_format}${item.requires_network ? " · 需要网络" : ""}${item.requires_credentials ? " · 需要凭据" : ""}`}</p>
        <div className="capability-stage-list">{STAGES.map((candidate) => <span key={candidate} className={item.supported_stages.includes(candidate) ? "supported" : "unsupported"}>{STAGE_NAMES[candidate]}</span>)}</div>
      </article>)}
    </section> : <>
      <section className="dataset-import-card" aria-labelledby="dataset-import-heading">
        <div className="dataset-import-heading"><div><div className="eyebrow">LOCAL JSONL IMPORT</div><h2 id="dataset-import-heading">登记本地数据集</h2><p>SFT 支持来源格式转换；客观题支持单选、多选、判断并登记为评测专用数据，原始文件与转换证据会保留。</p></div><span className="dataset-limit-badge">源文件 ≤ 256 MiB · 转换总量 ≤ 512 MiB · ≤ 250,000 行</span></div>
        <div className="import-steps" aria-label="导入步骤">
          {["选择文件和阶段", "核对来源与许可", "校验并登记"].map((label, index) => <div key={label} className={step === index + 1 ? "current" : step > index + 1 ? "done" : ""}><span>{index + 1}</span>{label}</div>)}
        </div>
        {step === 1 && <div className="dataset-import-form">
          <label>数据用途<select value={stage} onChange={(event) => { const next = event.target.value as ImportStage; setStage(next); setSourceFormat(next === "eval" ? "objective-choice-jsonl-v1" : "canonical-v1"); setImportPreview(null); setSubmitError(""); }}>{IMPORT_STAGES.map((item) => <option key={item} value={item}>{STAGE_NAMES[item]}</option>)}</select><small>客观题评测集只能用于评测，不会出现在训练数据选项中；Preference 可供 DPO / SimPO 使用。</small></label>
          {(stage === "sft" || stage === "eval") && <label>来源格式<select value={sourceFormat} onChange={(event) => { setSourceFormat(event.target.value); setImportPreview(null); setSubmitError(""); }}>
            {stage === "eval" ? <option value="objective-choice-jsonl-v1">客观题 JSONL（questionType / options / correctAnswer）</option> : <>
              <option value="canonical-v1">项目标准格式（Canonical JSONL）</option>
              <option value="alpaca-instruction-v1">Alpaca（instruction / input / output）</option>
              <option value="messages-role-content-v1">Messages（role / content 多轮）</option>
              <option value="sharegpt-conversations-v1">ShareGPT（conversations: from / value）</option>
            </>}
          </select><small>{stage === "eval" ? "支持 single_choice、multiple_choice、true_false；选项数量可变。格式固定，不会自动猜测。" : "必须与文件实际格式一致；不会自动猜测。ShareGPT 的 human→user、gpt→assistant。"}</small></label>}
          <label>{stage === "eval" ? "客观题评测 JSONL 文件" : stage === "sft" && sourceFormat !== "canonical-v1" ? "SFT JSONL 文件" : "Canonical JSONL 文件"}<input type="file" accept=".jsonl,.ndjson,application/x-ndjson" onChange={(event) => { setFile(event.target.files?.[0] || null); setImportPreview(null); setSubmitError(""); }} /><small>{file ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MiB` : "选择本地 UTF-8 JSONL；选择文件不会上传，点击预览时才提交到本地服务。"}</small></label>
          {file && file.size > MAX_DATASET_IMPORT_BYTES && <p className="field-error">文件超过 256 MiB 上限。</p>}
          <div className="import-actions"><button className="secondary-button" disabled={!file || file.size > MAX_DATASET_IMPORT_BYTES || previewBusy} onClick={() => void previewImport()}>{previewBusy ? "正在转换预览…" : importPreview?.valid === true ? "重新预览" : "1. 预览并校验"}</button><button className="primary-button" disabled={!file || file.size > MAX_DATASET_IMPORT_BYTES || importPreview?.valid !== true || previewBusy} onClick={() => { setSubmitError(""); setStep(2); }}>继续核对来源与许可 <span aria-hidden="true">→</span></button></div>
          {importPreview && <section className={`import-conversion-preview${importPreview.valid === true ? " is-valid" : " has-errors"}`} aria-live="polite" aria-label="来源格式转换预览">
            <h3>{importPreview.valid === true ? "预览通过 · 尚未登记" : "预览发现问题 · 尚未登记"}</h3>
            <p>输入行 {String(importPreview.source_record_count ?? 0)} · 可转换 {String(importPreview.output_record_count ?? 0)} · 来源格式 {String(importPreview.source_format)} · 转换器 {String(importPreview.converter_version)}</p>
            {Boolean(importPreview.mapping && Object.keys(importPreview.mapping as Record<string, unknown>).length > 0) && <p>角色映射：human → user；gpt → assistant。多轮顺序保持不变。</p>}
            {Number(importPreview.rejected_record_count || 0) > 0 && <p role="status">将排除 {String(importPreview.rejected_record_count)} 条无效记录；只有可校验记录会登记为评测集。</p>}
            {Array.isArray(importPreview.errors) && importPreview.errors.length > 0 && <ul>{(importPreview.errors as Array<Record<string, unknown>>).map((issue, index) => <li key={`${String(issue.line)}-${index}`}>第 {String(issue.line)} 行{issue.field ? ` · ${String(issue.field)}` : ""}：{String(issue.message)}</li>)}</ul>}
            {Array.isArray(importPreview.samples) && (importPreview.samples as unknown[]).length > 0 && <details><summary>查看最多 3 条转换后的 canonical 样例</summary><pre>{JSON.stringify(importPreview.samples, null, 2)}</pre></details>}
            <small>{String(importPreview.disclaimer || "预览仅校验结构与转换结果，不验证语义正确性或许可。")}</small>
          </section>}
        </div>}
        {step === 2 && <div className="dataset-import-form">
          <div className="import-file-summary"><strong>{STAGE_NAMES[stage]} · {sourceFormat}</strong><span>{file?.name} · {file ? `${(file.size / 1024 / 1024).toFixed(2)} MiB` : ""}</span></div>
          <div className="import-license-note">已预览 {String(importPreview?.output_record_count ?? 0)} 条 canonical 样例/记录。登记时服务端会重新转换并校验；来源原件也会保留用于审计。</div>
          <div className="import-metadata-grid">
            <label>来源名称<input value={sourceName} onChange={(event) => setSourceName(event.target.value)} placeholder="例如：团队整理的代数训练集" /></label>
            <label>来源 URI（可选）<input value={sourceUri} onChange={(event) => setSourceUri(event.target.value)} placeholder="https://… 或本地来源说明" /></label>
            <label>Revision（可选）<input value={revision} onChange={(event) => { setRevision(event.target.value); if (event.target.value) setRevisionStatus("known"); }} placeholder="数据源版本 / commit" /></label>
            <label>Revision 状态<select value={revisionStatus} onChange={(event) => setRevisionStatus(event.target.value)}><option value="known">已提供</option><option value="unknown">未知</option><option value="not_provided">未提供</option></select></label>
            <label>许可证标识（可选）<input value={licenseId} onChange={(event) => { setLicenseId(event.target.value); if (event.target.value) setLicenseStatus("user_declared_unverified"); }} placeholder="例如：CC-BY-4.0" /></label>
            <label>许可证状态<select value={licenseStatus} onChange={(event) => setLicenseStatus(event.target.value)}><option value="user_declared_unverified">用户声明，未独立核验</option><option value="unknown">未知</option><option value="not_provided">未提供</option></select></label>
            <label className="import-wide">Attribution（可选）<input value={attribution} onChange={(event) => setAttribution(event.target.value)} placeholder="署名或使用说明" /></label>
          </div>
          <p className="import-license-note">未知/未提供的来源或许可信息不阻止登记；使用客观题评测集前请确认有权使用。评测集不会用作训练数据；用户声明不会把许可证状态升级为已核验，答案也未经事实核验。</p>
          {submitError && <div className="dataset-workbench-alert" role="alert">{submitError}</div>}
          <div className="import-actions"><button className="secondary-button" disabled={submitting} onClick={() => setStep(1)}>返回修改</button>{submitting && <button className="secondary-button" onClick={() => uploadController.current?.abort()}>取消上传</button>}<button className="primary-button" disabled={submitting || !file || importPreview?.valid !== true} onClick={submitImport}>{submitting ? "正在重新转换并登记…" : "确认来源声明并登记数据集"}</button></div>
        </div>}
        {step === 3 && result && <div className="dataset-import-success" role="status"><div className="success-mark">✓</div><div><h3>{stage === "eval" ? "评测数据集已登记" : "数据集已登记"}</h3><p><code>{String(result.dataset_id)}</code> · {String(result.record_count)} 条 · 质量状态 {String(result.quality_status)}{stage === "eval" ? " · 评测专用" : ""}</p><small>来源原件与 canonical 版本均已登记；格式校验通过不等于语义质量通过或许可证已核验。</small></div><button className="secondary-button" onClick={() => { setStep(1); setFile(null); setImportPreview(null); setSourceFormat("canonical-v1"); setResult(null); setSourceName(""); setSourceUri(""); setRevision(""); setLicenseId(""); setAttribution(""); }}>继续导入</button></div>}
      </section>

      {targetedGeneration ? <TargetedExperiments generationOnly onRegistered={() => setRefresh((value) => value + 1)} /> : <>
        <div className="cpt-generation-tabs" role="tablist" aria-label="模型生成的数据阶段">
          {([['sft', 'SFT 指令数据'], ['cpt', 'CPT 语料'], ['preference', 'Preference 偏好对'], ['grpo', 'GRPO 题目候选']] as const).map(([id, label]) => <button key={id} type="button" role="tab" aria-selected={generationStage === id} className={generationStage === id ? "selected" : ""} onClick={() => setGenerationStage(id)}>{label}</button>)}
        </div>
        {generationStage === "cpt" ? <CPTGenerationPanel onRegistered={() => setRefresh((value) => value + 1)} /> : generationStage === "preference" ? <PreferenceGenerationPanel onRegistered={() => setRefresh((value) => value + 1)} /> : generationStage === "grpo" ? <GRPOGenerationPanel /> : <SFTGenerationPanel onOpenTargeted={onOpenTargeted} onRegistered={() => setRefresh((value) => value + 1)} />}
      </>}

      <section className="dataset-mixture-card" aria-labelledby="grpo-seed-heading">
        <div className="dataset-import-heading"><div><div className="eyebrow">GRPO DATA PREPARATION</div><h2 id="grpo-seed-heading" tabIndex={-1}>从可信训练种子派生 GRPO 数据</h2><p>当前仅接受固定 OpenR1 来源/revision 和逐条上游正确性证据的已登记 SFT train_seed。不会把用户上传字段当成可信参考答案。</p></div><span className="dataset-limit-badge">不需要定向方案</span></div>
        <div className="mixture-controls"><label>目标生成条数<input type="number" min={1} max={250000} value={grpoSeedTarget} onChange={(event) => { setGrpoSeedTarget(Number(event.target.value)); setGrpoSeedPreflight(null); setGrpoSeedResult(null); }} /></label><label>确定性随机种子<input type="number" min={0} value={grpoSeedValue} onChange={(event) => { setGrpoSeedValue(Number(event.target.value)); setGrpoSeedPreflight(null); setGrpoSeedResult(null); }} /></label></div>
        <fieldset className="mixture-dataset-list"><legend>可用 SFT 种子（{grpoSeedCandidates.length}）</legend>
          {grpoSeedLoading && <p role="status">正在核验已登记来源与逐条参考答案证据…</p>}
          {!grpoSeedLoading && grpoSeedCandidates.length === 0 && <p>没有找到符合当前严格来源契约的已登记训练种子。请登记具有固定 OpenR1 build protocol/revision 与逐条正确性证据的数据；普通上传数据不会自动获得该信任级别。</p>}
          {grpoSeedCandidates.map((candidate) => { const id = String(candidate.dataset_id); return <label className="mixture-dataset-option" key={id}><input type="checkbox" checked={grpoSeedIds.includes(id)} onChange={(event) => { setGrpoSeedIds((current) => event.target.checked ? [...current, id] : current.filter((value) => value !== id)); setGrpoSeedPreflight(null); setGrpoSeedResult(null); }} /><span><strong>{String(candidate.display_name || id)}</strong><small>{String(candidate.eligible_count)} 条可用 / {String(candidate.sample_count)} 条登记 · {String(candidate.rejected_count)} 条证据不符合 · revision {String(candidate.revision || "缺失")}</small><small>来源证据：{String(candidate.evidence_scope)} · 许可：{String(candidate.license_status)}{candidate.license_id ? ` (${String(candidate.license_id)})` : ""}</small><code>{id}</code></span></label>; })}
        </fieldset>
        <p className="mixture-lineage-note">参考答案来自已登记 SFT 记录的 parsed_answer，并绑定父 dataset ID/hash 和 source record ID。correctness_math_verify=true 是上游证据，本系统不声称独立重验；许可提醒需按父数据来源自行确认。结果质量状态为仅报告，不代表 GRPO 训练已运行。</p>
        {grpoSeedError && <div className="dataset-workbench-alert" role="alert">{grpoSeedError} <button className="secondary-button" onClick={() => setRefresh((value) => value + 1)}>刷新候选</button></div>}
        <div className="import-actions"><button className="secondary-button" disabled={grpoSeedBusy || grpoSeedIds.length === 0 || grpoSeedTarget < 1} onClick={() => void preflightGrpoSeed()}>{grpoSeedBusy ? "正在核验…" : "预检 GRPO 数据"}</button>{grpoSeedPreflight && <button className="primary-button" disabled={grpoSeedBusy || Number(grpoSeedPreflight.shortfall) > 0} onClick={() => void buildGrpoSeed()}>{grpoSeedBusy ? "正在登记并质检…" : "确认派生并登记"}</button>}</div>
        {grpoSeedPreflight && <div className="mixture-preflight" role="status"><strong>可用 {String(grpoSeedPreflight.available_count)} / 目标 {String(grpoSeedPreflight.requested_count)} 条 · 质量状态：仅报告</strong>{(() => { const contract = grpoSeedPreflight.reward_contract as Record<string, unknown> | undefined; return contract ? <p>服务端批准 reward contract：<code>{String(contract.contract_id)}</code> · v{String(contract.version)} · {(contract.rewards as string[]).join(" + ")}</p> : null; })()}<p>{String(grpoSeedPreflight.reference_evidence)}</p>{Array.isArray(grpoSeedPreflight.limitations) && <ul>{(grpoSeedPreflight.limitations as string[]).map((item) => <li key={item}>{item}</li>)}</ul>}</div>}
        {grpoSeedResult && <div className="dataset-import-success" role="status"><div className="success-mark">✓</div><div><h3>GRPO 数据已派生、登记并生成质量/血缘报告</h3><p><code>{String(grpoSeedResult.dataset_id)}</code> · {String(grpoSeedResult.record_count)} 条 · {String(grpoSeedResult.quality_status)}</p></div>{onUseDataset && <button className="primary-button" onClick={() => onUseDataset(String(grpoSeedResult.dataset_id), "grpo")}>去 GRPO 训练工作台使用 →</button>}</div>}
      </section>

      <section className="dataset-mixture-card" id="dataset-mixture-card" aria-labelledby="dataset-mixture-heading">
        <div className="dataset-import-heading"><div><div className="eyebrow">DATASET COMPOSITION</div><h2 id="dataset-mixture-heading">组合已登记数据集</h2><p>选择同一训练阶段的两个或更多数据集，生成一个新 dataset ID；训练仍只引用这个组合版本。</p></div><span className="dataset-limit-badge">支持 CPT · SFT · Preference · GRPO</span></div>
        <div className="mixture-controls">
          <label>数据集名称<input type="text" maxLength={120} value={mixtureName} onChange={(event) => { setMixtureName(event.target.value); setMixturePreflight(null); setMixtureResult(null); }} placeholder="例如：数学推理 SFT 混合数据 v1" /></label>
          <label>训练阶段<select value={mixtureStage} onChange={(event) => { const stage = event.target.value as (typeof STAGES)[number]; setMixtureStage(stage); setMixtureName(`${STAGE_NAMES[stage]} 组合数据集`); setMixtureIds([]); setMixturePreflight(null); setMixtureResult(null); }}>{STAGES.map((item) => <option key={item} value={item}>{STAGE_NAMES[item]}</option>)}</select></label>
          <label>组合方式<select value={mixtureStrategy} onChange={(event) => { setMixtureStrategy(event.target.value as "concat" | "weighted_sample"); setMixturePreflight(null); }}><option value="concat">全部拼接（保留所有样本）</option><option value="weighted_sample">按权重确定性采样（不放回）</option></select></label>
          {mixtureStrategy === "weighted_sample" && <><label>目标条数<input type="number" min={1} max={250000} value={mixtureBudget} onChange={(event) => { setMixtureBudget(Number(event.target.value)); setMixturePreflight(null); }} /></label><label>随机种子<input type="number" min={0} value={mixtureSeed} onChange={(event) => { setMixtureSeed(Number(event.target.value)); setMixturePreflight(null); }} /></label></>}
        </div>
        <fieldset className="mixture-dataset-list"><legend>选择数据集（{mixtureCandidates.length} 个 {STAGE_NAMES[mixtureStage]} 数据集）</legend>
          {mixtureCandidates.length === 0 ? <p>当前阶段尚无已登记数据集。</p> : mixtureCandidates.map((dataset) => {
            const blocked = dataset.mixture_eligible === false;
            return <label className={`mixture-dataset-option${blocked ? " is-disabled" : ""}`} key={dataset.dataset_id} title={blocked ? dataset.mixture_block_reason || "登记证据不完整" : undefined}><input type="checkbox" disabled={blocked} checked={mixtureIds.includes(dataset.dataset_id)} onChange={(event) => { setMixtureIds((current) => event.target.checked ? [...current, dataset.dataset_id] : current.filter((id) => id !== dataset.dataset_id)); setMixturePreflight(null); }} /><span><strong>{dataset.display_name || dataset.buildName}</strong><small>{dataset.count.toLocaleString()} 条 · {KIND_NAMES[sourceOf(dataset).kind] || sourceOf(dataset).kind} · 质量 {GATE_LABEL[qualityOf(dataset)] || qualityOf(dataset)}</small><code>{dataset.dataset_id}</code>{blocked && <small className="mixture-block-reason">不可组合：{dataset.mixture_block_reason || "登记证据不完整"}</small>}</span>{mixtureStrategy === "weighted_sample" && mixtureIds.includes(dataset.dataset_id) && <span className="mixture-weight"><small>权重</small><input aria-label={`${dataset.display_name || dataset.buildName} 采样权重`} type="number" min="0.001" step="0.1" value={mixtureWeights[dataset.dataset_id] ?? 1} onChange={(event) => { setMixtureWeights((current) => ({ ...current, [dataset.dataset_id]: Number(event.target.value) })); setMixturePreflight(null); }} /></span>}</label>;
          })}
        </fieldset>
        <p className="mixture-lineage-note">构建会重新校验输入内容摘要、阶段和训练角色；Preference pair 不拆分，GRPO 只允许 reward/reference 契约一致的数据集。组合数据会单独质检，许可提醒仍需逐来源确认。</p>
        {mixtureError && <div className="dataset-workbench-alert" role="alert">{mixtureError}</div>}
        <div className="import-actions"><button className="secondary-button" disabled={mixtureBusy || mixtureIds.filter((id) => eligibleMixtureIds.has(id)).length < 2} onClick={() => void runMixturePreflight()}>{mixtureBusy ? "正在校验…" : "预检组合"}</button>{mixturePreflight && <button className="primary-button" disabled={mixtureBusy} onClick={() => void buildMixture()}>{mixtureBusy ? "正在构建并质检…" : "确认创建组合数据集"}</button>}</div>
        {mixturePreflight && <div className="mixture-preflight" role="status"><strong>预检通过 · {Number(mixturePreflight.total_records).toLocaleString()} 条输出</strong><span>配置摘要：<code>{String(mixturePreflight.config_sha256)}</code></span><ul>{(mixturePreflight.parents as Array<Record<string, unknown>>).map((parent) => <li key={String(parent.dataset_id)}>{String(parent.dataset_id)} · 输入 {String(parent.record_count)} 条 → 选入 {String(parent.selected_records)} 条 · SHA256 已核对</li>)}</ul><p>{(mixturePreflight.warnings as string[]).join(" ")}</p></div>}
        {mixtureResult && <div className="dataset-import-success" role="status"><div className="success-mark">✓</div><div><h3>组合数据集已创建并完成质检报告</h3><p>{String(mixtureResult.display_name || mixtureName)} · <code>{String(mixtureResult.dataset_id)}</code> · {String(mixtureResult.record_count)} 条 · 质量状态 {String(mixtureResult.quality_status)}</p><small>父数据集、内容摘要和逐条来源记录均已写入血缘证据。</small></div>{onUseDataset && <button className="primary-button" onClick={() => onUseDataset(String(mixtureResult.dataset_id), mixtureStage)}>去训练工作台使用 →</button>}</div>}
      </section>

      <section className="registered-datasets-section">
        <div className="registered-datasets-head"><div><div className="eyebrow">REGISTERED DATASETS</div><h2>可追溯的数据版本</h2><p>记录数不作为硬性门槛；来源、质量和训练 readiness 分开显示。</p></div><div className="dataset-filters"><label>阶段<select value={stageFilter} onChange={(event) => setStageFilter(event.target.value)}><option value="all">全部阶段</option>{STAGES.map((item) => <option value={item} key={item}>{STAGE_NAMES[item]}</option>)}</select></label><label>来源<select value={kindFilter} onChange={(event) => setKindFilter(event.target.value)}><option value="all">全部来源</option><option value="local_upload">本地导入</option><option value="open_source">开源数据</option><option value="model_generated">模型生成</option><option value="dataset_mixture">派生组合</option><option value="dataset_derived">规则派生</option><option value="unknown">未记录</option></select></label></div></div>
      {loading && !catalog ? <div className="workbench-loading" role="status">正在读取本地数据目录…</div> : !catalog && error ? <div className="workbench-empty" role="alert"><strong>数据目录暂不可用</strong><span>可重试读取；本地来源能力和其他模块不受此错误影响。</span></div> : filtered.length === 0 ? <div className="workbench-empty"><strong>当前筛选下没有已登记数据集</strong><span>{datasets.length ? "调整筛选条件，或导入一份 canonical JSONL。" : "导入本地 JSONL，或运行已有数据构建流程后刷新目录。"}</span></div> : <div className="registered-dataset-grid">{filtered.map((dataset) => {
          const source = sourceOf(dataset);
          const quality = qualityOf(dataset);
          const preview = previews[dataset.dataset_id];
          const previewPanelId = `dataset-preview-${encodeURIComponent(dataset.dataset_id).replaceAll("%", "-")}`;
          return <article className="registered-dataset-card" key={`${dataset.dataset_id}-${dataset.stage}`}>
            <div className="registered-dataset-card-head"><span className="stage-chip">{STAGE_NAMES[dataset.stage] || dataset.stage}</span><span className={`quality-chip quality-${quality.toLowerCase().replaceAll("_", "-")}`}>{GATE_LABEL[quality] || quality}</span></div>
            <h3>{dataset.display_name || dataset.buildName}</h3><p className="dataset-source-name"><span className={`source-kind source-kind-${source.kind}`}>{KIND_NAMES[source.kind] || source.kind}</span>{source.name || source.source_id || "未记录"}</p>
            <div className="dataset-identity"><span>dataset ID</span><code title={dataset.dataset_id}>{dataset.dataset_id}</code></div>
            <div className="dataset-card-facts"><div><span>记录数</span><strong>{dataset.count.toLocaleString()} {dataset.unit}</strong></div><div><span>许可状态</span><strong>{source.license_status || "not_provided"}</strong></div><div><span>Revision</span><strong>{source.revision_status || "not_provided"}</strong></div></div>
            {onUseDataset && <button className="button secondary dataset-use-training" onClick={() => onUseDataset(dataset.dataset_id, dataset.stage)} aria-label={`在训练工作台选择数据集 ${dataset.dataset_id}`}>在训练工作台选择此数据 →</button>}
            {dataset.content_sha256 && <details className="dataset-digest"><summary>查看内容 SHA256</summary><code>{dataset.content_sha256}</code></details>}
            <button className="dataset-preview-toggle" aria-expanded={Boolean(preview)} aria-controls={previewPanelId}
              onClick={() => preview ? closePreview(dataset.dataset_id) : void openPreview(dataset.dataset_id)}>
              {preview?.status === "loading" ? "正在核验并读取样本…" : preview ? "收起样本预览" : "预览样本"}
            </button>
            {preview && <section className="dataset-preview-panel" id={previewPanelId} aria-label={`${dataset.dataset_id} 样本预览`}>
              <div className="dataset-preview-panel-head"><strong>样本预览</strong><button className="dataset-preview-close" aria-label="关闭样本预览" onClick={() => closePreview(dataset.dataset_id)}>关闭</button></div>
              {preview.status === "loading" && <p className="dataset-preview-message" role="status">正在验证 canonical 内容摘要；最多读取 3 条样本。</p>}
              {preview.status === "error" && <div className="dataset-preview-error" role="alert"><span>暂不可预览：{preview.message}</span><button className="secondary-button" onClick={() => void openPreview(dataset.dataset_id)}>重试</button></div>}
              {preview.status === "success" && <>
                <p className="dataset-preview-integrity" role="status"><span aria-hidden="true">✓</span> canonical SHA256 已核验 · {preview.data?.returned_records ?? 0}/{preview.data?.total_records ?? 0} 条</p>
                <p className="dataset-preview-disclaimer">完整性核验不代表语义质量通过、许可已核验或具备训练 readiness。卡片上的质量和许可状态保持独立。</p>
                {(preview.data?.samples.length ?? 0) === 0 ? <p className="dataset-preview-message">数据集为空，没有可显示的样本。</p> : <div className="dataset-preview-samples">{preview.data?.samples.map((sample, index) => <article className="dataset-preview-sample" key={`${sample.id}-${index}`}>
                  <div><span>样本 {index + 1}</span><code>{sample.id}</code></div>
                  <pre>{JSON.stringify(sample.fields, null, 2)}</pre>
                  {sample.truncated_fields.length > 0 && <small>部分字段为保护页面性能已截断：{sample.truncated_fields.join("、")}</small>}
                </article>)}</div>}
                {preview.data?.truncated && <small className="dataset-preview-more">仅显示前 {preview.data.returned_records} 条；数据集共有 {preview.data.total_records.toLocaleString()} 条。</small>}
              </>}
            </section>}
            <button className="dataset-preview-toggle dataset-detail-toggle" aria-expanded={expandedDatasetId === dataset.dataset_id}
              onClick={() => setExpandedDatasetId((current) => current === dataset.dataset_id ? null : dataset.dataset_id)}>
              {expandedDatasetId === dataset.dataset_id ? "收起数据集详情" : "查看质量与血缘详情"}
            </button>
            {expandedDatasetId === dataset.dataset_id && <section className="registered-dataset-detail" aria-label={`${dataset.dataset_id} 数据集质量与血缘详情`}>
              {buildEvidence[dataset.buildId]?.loading && <p role="status">正在读取该版本的处理报告和关联 Run…</p>}
              {buildEvidence[dataset.buildId]?.error && <div className="dataset-preview-error" role="alert"><span>{buildEvidence[dataset.buildId]?.error}</span><button className="secondary-button" onClick={() => setDetailRefresh((value) => value + 1)}>重试</button></div>}
              {buildEvidence[dataset.buildId]?.detail && <>
                <div className="registered-detail-section">
                  <h4>质量评估</h4>
                  {dataset.quality?.metrics && Object.keys(dataset.quality.metrics).length > 0 ? <div className="registered-quality-metrics">
                    {Object.entries(dataset.quality.metrics).map(([key, metric]) => <div key={key}><span title={key}>{QUALITY_NAMES[key] || key.replaceAll("_", " ")}</span><strong>{metric.value === null ? QUALITY_STATUS[metric.status] || "未记录" : `${metric.unit === "ratio" ? `${(metric.value * 100).toFixed(1)}%` : metric.value}${metric.unit && metric.unit !== "ratio" ? ` ${metric.unit}` : ""}`}</strong><small>{metric.denominator === null ? "分母未提供" : `实际分母 ${metric.denominator}`} · {QUALITY_STATUS[metric.status] || metric.status}</small></div>)}
                  </div> : <p>此数据版本没有可展示的质量指标明细；状态：{quality}。</p>}
                  {(dataset.quality?.limitations || []).length > 0 && <ul>{dataset.quality?.limitations.map((item, index) => <li key={`${index}-${item}`}>{item}</li>)}</ul>}
                </div>
                <div className="registered-detail-section">
                  <h4>处理血缘</h4>
                  <ol className="registered-processing-steps">{buildEvidence[dataset.buildId]?.detail?.steps.map((item) => <li key={item.step_id}>
                    <strong>{item.operation} · {item.operation_version}</strong><span>{item.input_count.value ?? "—"} {item.input_count.unit} → {item.output_count.value ?? "—"} {item.output_count.unit}</span>
                  </li>)}</ol>
                  <div className="registered-detail-actions"><button className="dataset-lineage-link" onClick={() => onViewEvidence(dataset.buildId, dataset.stage)}>打开完整处理报告 <span aria-hidden="true">→</span></button><details><summary>登记证据文件</summary><ul><li>{dataset.manifest?.label || "数据集清单"}: <code>{dataset.manifest?.path || "未记录"}</code></li>{dataset.quality_artifact && <li>{dataset.quality_artifact.label}: <code>{dataset.quality_artifact.path}</code></li>}</ul></details></div>
                </div>
                <div className="registered-detail-section">
                  <h4>关联训练运行</h4>
                  {(buildEvidence[dataset.buildId]?.detail?.related_runs || []).length > 0 ? <ul className="registered-related-runs">{buildEvidence[dataset.buildId]?.detail?.related_runs.map((run) => <li key={run.run_id}><span><strong>{run.stage} · {run.status}</strong><code>{run.run_id}</code></span>{onOpenRun && <button className="dataset-lineage-link" onClick={() => onOpenRun(run.run_id)}>查看 Run →</button>}</li>)}</ul> : <p>尚无明确引用此数据构建的训练 Run。</p>}
                </div>
              </>}
            </section>}
          </article>;
        })}</div>}
      </section>
    </>}
  </main>;
}
