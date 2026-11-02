import { useCallback, useEffect, useState } from "react";
import { TaskStatusCard } from "./TaskStatusCard";
import type { TaskRequestState } from "./taskPresentation";

type RunSummary = {
  run_id: string; run_kind: string; stage: string | null; status: string;
  started_at: string | null; completed_at: string | null; run_label: string | null;
  experiment_id: string | null; source_ref: string; source_refs?: string[];
  parent_run_id?: string | null; warnings?: string[];
};
type RunDetailData = RunSummary & {
  config: Record<string, unknown> | null;
  metrics: { name: string; metric: string; value: number | null; denominator: number | null;
    denominator_unit?: string | null; status?: string; reason?: string | null }[];
  logs: { path: string; text: string; truncated: boolean } | null;
  outputs: string[];
  lineage: Record<string, unknown> | null;
  provenance: { id?: string; kind?: string; role?: string; path?: string; availability?: string;
    digest?: { algorithm?: string; status?: string; value?: string | null; reason?: string } | null }[];
  produced_artifacts?: { id?: string; kind?: string; role?: string; path?: string; availability?: string;
    digest?: { algorithm?: string; status?: string; value?: string | null; reason?: string } | null }[];
  handoff_models?: RegisteredModel[];
  evaluations: RunSummary[];
  parent_available?: boolean;
};
type ChoiceJudgeProgress = {
  completed: number; total: number | null;
  counts: { correct: number; incorrect: number; undetermined: number; provider_errors: number };
  phase: string; current_question: number | null; status: string;
  events: { completed: number; total: number; phase: string; current_question: number | null;
    error_category?: string | null }[];
};
type LiveLauncherRun = { run_id: string; status: string; scoring_strategy?: string;
  choice_judge_progress?: ChoiceJudgeProgress; log_tail?: string };
type RunIndex = { runs: RunSummary[]; total: number; partial_scan: boolean };
type EvaluationProfile = { profile_id: string; protocol: string; metrics: string[] };
type StageProfile = { profile_id: string; stage: string; parent_model_stages: string[]; parent_model_rule: string;
  evaluation_profiles: EvaluationProfile[]; execution_supported: boolean; execution_blocker?: string | null };
type RegisteredModel = { model_id: string; stage: string; kind: string; display_name?: string | null; produced_by_run_id?: string | null;
  source_run_id?: string | null; digest?: { status: string; value?: string | null }; path_available?: boolean; registry_count?: number };
type EvaluationPreflight = { ok: boolean; blockers: string[]; warnings: string[]; profile_id?: string;
  protocol_version?: string; model?: { model_id: string; stage: string; sha256: string };
  dataset?: { dataset_id: string; record_count: number; content_sha256?: string; protocol_version?: string } | null;
  evaluation_run_id?: string; output?: string; metrics?: string[]; confirmation_token?: string; expires_in_seconds?: number;
  external_judge?: { provider?: string; provider_profile_id?: string; provider_model?: string;
    expected_request_count?: number; external_data_notice?: string; billing_notice?: string; prompt_version?: string } | null };
type WorkbenchOptions = { datasets: { dataset_id: string; stage: string; role: string | null; record_count: number;
  eligible?: boolean; block_reason?: string | null }[]; models: RegisteredModel[] };
type ChoiceReportRow = { id: string; question_type: string; question: string; options: string[]; gold: string;
  prediction: string | null; raw_output: string | null; parse_success: boolean | null; correct: boolean | null;
  judge_outcome?: "correct" | "incorrect" | "undetermined" | "provider_error" | null;
  effective_judge_outcome?: "correct" | "incorrect" | "undetermined" | "provider_error" | null;
  judge_rationale?: string | null; judge_error?: string | null; judge_request_uncertain?: boolean;
  retry_attempts?: { attempt_number: number; judge_outcome: "correct" | "incorrect" | "undetermined" | "provider_error";
    judge_rationale?: string | null; judge_error?: string | null; judge_request_uncertain?: boolean;
    provider_request_id?: string | null; created_at?: string | null }[];
  provider_request_id?: string | null;
  status: "correct" | "incorrect" | "parse_failed" | "not_evaluated" | "undetermined" | "provider_error" };
type ChoiceMetricSummary = { correct: number; incorrect: number; undetermined: number; provider_errors: number;
  not_evaluated: number; denominator: number; decided_denominator: number; accuracy: number | null;
  decision_coverage: number | null; status: string };
type ChoiceMetricSet = Record<string, ChoiceMetricSummary>;
type ChoiceReport = { run_id: string; dataset_id: string; status: "complete" | "partial";
  protocol_version?: string; provider_profile_id?: string | null; provider_model?: string | null;
  request_count?: number | null; prompt_version?: string | null; metrics?: Record<string, unknown>;
  original_metrics?: Record<string, ChoiceMetricSummary> | null; current_metrics?: ChoiceMetricSet | null;
  sample_count: number; evaluated_count: number; rows: ChoiceReportRow[] };

const EVALUATION_DATASET_REQUIREMENTS: Record<string, { stage: string; role: string; label: string }> = {
  "preference-heldout-v1": { stage: "preference", role: "preference_eval", label: "偏好留出评测集" },
  "cpt-math-v1": { stage: "cpt", role: "cpt_holdout", label: "CPT 留出评测集" },
  "choice-accuracy-v1": { stage: "eval", role: "evaluation_only", label: "客观题评测集" },
};

const statusLabel: Record<string, string> = {
  completed: "完成", failed: "失败", partial: "部分完成", interrupted: "中断",
  running: "运行中", starting: "启动中", queued: "排队中", oom: "显存不足", unknown: "未知",
};
const stageLabel: Record<string, string> = {
  cpt: "CPT", sft: "SFT", dpo: "DPO", simpo: "SimPO", grpo: "GRPO", sft_eval: "SFT 评测",
  preference: "偏好优化", evaluate: "评测",
};
const metricLabel: Record<string, string> = {
  "gsm8k.accuracy": "GSM8K 正确率", "math_500.accuracy": "MATH-500 正确率",
  "internal_acc.accuracy": "内部留出集正确率", "domain_ppl.perplexity": "领域困惑度（PPL）",
  "preference.accuracy": "偏好胜率", "preference.margin": "偏好平均差值",
  "choice_accuracy.single_choice.value": "单选准确率",
  "choice_accuracy.multiple_choice.value": "多选准确率",
  "choice_accuracy.true_false.value": "判断题准确率",
  "choice_accuracy.overall.value": "总体准确率",
  "choice_llm_judge.single_choice.accuracy.value": "单选语义判定准确率",
  "choice_llm_judge.multiple_choice.accuracy.value": "多选语义判定准确率",
  "choice_llm_judge.true_false.accuracy.value": "判断语义判定准确率",
  "choice_llm_judge.overall.accuracy.value": "总体语义判定准确率",
  "choice_llm_judge.single_choice.decision_coverage.value": "单选判定覆盖率",
  "choice_llm_judge.multiple_choice.decision_coverage.value": "多选判定覆盖率",
  "choice_llm_judge.true_false.decision_coverage.value": "判断判定覆盖率",
  "choice_llm_judge.overall.decision_coverage.value": "总体判定覆盖率",
};
const metricStatusLabel: Record<string, string> = {
  measured: "已测量", estimated: "估算值", not_measured: "未测量",
  not_applicable: "不适用", failed: "评测失败", partial: "部分完成", unknown: "状态未知",
};

function queryState() {
  const params = new URLSearchParams(window.location.search);
  const offset = Number(params.get("run_offset") || 0);
  return { run: params.get("run") || "", stage: params.get("run_stage") || "", status: params.get("run_status") || "", q: params.get("run_q") || "", offset: Number.isFinite(offset) ? Math.max(0, Math.min(offset, 5000)) : 0 };
}
function writeQuery(state: ReturnType<typeof queryState>, run = state.run, push = false) {
  const params = new URLSearchParams(window.location.search);
  params.set("page", "runs");
  if (run) params.set("run", run); else params.delete("run");
  if (state.stage) params.set("run_stage", state.stage); else params.delete("run_stage");
  if (state.status) params.set("run_status", state.status); else params.delete("run_status");
  if (state.q) params.set("run_q", state.q); else params.delete("run_q");
  if (state.offset) params.set("run_offset", String(state.offset)); else params.delete("run_offset");
  const url = `${window.location.pathname}?${params.toString()}`;
  if (push) window.history.pushState(null, "", url); else window.history.replaceState(null, "", url);
}
async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, { headers: { Accept: "application/json" } });
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail || `本地服务返回 ${response.status}`);
  return value as T;
}
class ApiRequestError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}
function formatDate(value: string | null | undefined) {
  if (!value) return "未记录";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString("zh-CN");
}
function formatMetricValue(name: string, value: number | null, status?: string) {
  if (value == null) return metricStatusLabel[status || "unknown"] || status || "状态未知";
  if (name.startsWith("choice_accuracy.") || name.endsWith(".accuracy") || name.endsWith(".decision_coverage")) return `${(value * 100).toFixed(1)}%`;
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 3 }).format(value);
}
function displayRunName(run: RunSummary) {
  return run.run_label?.trim() || `${stageLabel[run.stage || ""] || run.stage || "阶段未记录"} · ${run.run_kind === "evaluation" ? "评测" : "训练"}`;
}

export function Runs({ onBack, onTryModel }: { onBack: () => void; onTryModel?: (modelId: string, runId: string) => void }) {
  const [filters, setFilters] = useState(queryState);
  const [index, setIndex] = useState<RunIndex | null>(null);
  const [detail, setDetail] = useState<RunDetailData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [detailError, setDetailError] = useState("");
  const [evaluationProfiles, setEvaluationProfiles] = useState<EvaluationProfile[]>([]);
  const [evaluationDatasets, setEvaluationDatasets] = useState<WorkbenchOptions["datasets"]>([]);
  const [evaluationProfileId, setEvaluationProfileId] = useState("");
  const [evaluationDatasetId, setEvaluationDatasetId] = useState("");
  const [evaluationPreflight, setEvaluationPreflight] = useState<EvaluationPreflight | null>(null);
  const [evaluationError, setEvaluationError] = useState("");
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [evaluationRequest, setEvaluationRequest] = useState<TaskRequestState>("idle");
  const [evaluationScoringStrategy, setEvaluationScoringStrategy] = useState("exact_match");
  const [stageProfiles, setStageProfiles] = useState<StageProfile[]>([]);
  const [registeredModels, setRegisteredModels] = useState<RegisteredModel[]>([]);
  const [fixedMiniParentModelId, setFixedMiniParentModelId] = useState("");
  const [handoffCatalogLoaded, setHandoffCatalogLoaded] = useState(false);
  const [baseModels, setBaseModels] = useState<RegisteredModel[]>([]);
  const [baseCatalogLoading, setBaseCatalogLoading] = useState(true);
  const [baseModelId, setBaseModelId] = useState("");
  const [baseEvaluationProfile, setBaseEvaluationProfile] = useState("shared-math-eval-v1");
  const [baseScoringStrategy, setBaseScoringStrategy] = useState("exact_match");
  const [baseEvaluationDatasetId, setBaseEvaluationDatasetId] = useState("");
  const [baseEvaluationPreflight, setBaseEvaluationPreflight] = useState<EvaluationPreflight | null>(null);
  const [baseEvaluationBusy, setBaseEvaluationBusy] = useState(false);
  const [baseEvaluationRequest, setBaseEvaluationRequest] = useState<TaskRequestState>("idle");
  const [baseEvaluationError, setBaseEvaluationError] = useState("");
  const [choiceReport, setChoiceReport] = useState<ChoiceReport | null>(null);
  const [choiceMetricsView, setChoiceMetricsView] = useState<"current" | "original">("current");
  const [liveLauncherRun, setLiveLauncherRun] = useState<LiveLauncherRun | null>(null);
  const [choiceReportError, setChoiceReportError] = useState("");
  const [choiceReportLoading, setChoiceReportLoading] = useState(false);
  const [choiceReportFilter, setChoiceReportFilter] = useState("all");
  const [choiceReportType, setChoiceReportType] = useState("all");
  const [retryingChoiceRow, setRetryingChoiceRow] = useState<string | null>(null);
  const [choiceRetryErrors, setChoiceRetryErrors] = useState<Record<string, string>>({});

  const refresh = useCallback(() => {
    const params = new URLSearchParams({ limit: "100", offset: String(filters.offset) });
    if (filters.stage) params.set("stage", filters.stage);
    if (filters.status) params.set("status", filters.status);
    if (filters.q) params.set("q", filters.q);
    setLoading(true); setError("");
    getJson<RunIndex>(`/api/runs?${params}`)
      .then(setIndex).catch((e) => setError(e instanceof Error ? e.message : "读取 Runs 失败"))
      .finally(() => setLoading(false));
  }, [filters.stage, filters.status, filters.q, filters.offset]);

  useEffect(() => {
    const restore = () => setFilters(queryState());
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  useEffect(() => { writeQuery(filters); refresh(); }, [filters.stage, filters.status, filters.q, filters.offset, refresh]);
  useEffect(() => {
    if (filters.run) return;
    let alive = true;
    setBaseCatalogLoading(true);
    Promise.all([
      getJson<WorkbenchOptions>("/api/training/workbench-options"),
      (async () => {
        for (let attempt = 0; attempt < 30 && alive; attempt += 1) {
          const catalog = await getJson<{ status: string; models: (RegisteredModel & { available?: boolean })[] }>("/api/model-playground/models");
          if (catalog.status === "ready") return catalog;
          if (catalog.status === "failed") throw new Error("本地模型目录核验失败");
          await new Promise((resolve) => window.setTimeout(resolve, 500));
        }
        throw new Error("本地模型目录核验超时");
      })(),
    ]).then(([options, catalog]) => {
      if (!alive) return;
      setBaseModels(catalog.models.filter((model) => model.kind === "base" && model.available));
      setEvaluationDatasets(options.datasets.filter((dataset) => dataset.stage === "eval" && dataset.role === "evaluation_only"));
    }).catch((reason) => { if (alive) setBaseEvaluationError(reason instanceof Error ? reason.message : "无法读取已登记 Base 模型或评测集目录"); })
      .finally(() => { if (alive) setBaseCatalogLoading(false); });
    return () => { alive = false; };
  }, [filters.run]);
  useEffect(() => {
    setChoiceReport(null); setChoiceReportError(""); setChoiceReportFilter("all"); setChoiceReportType("all");
    if (!detail || detail.run_kind !== "evaluation" || !["completed", "partial", "failed", "interrupted"].includes(detail.status)) return;
    let alive = true;
    setChoiceReportLoading(true);
    getJson<{ protocol_version?: string }>(`/api/launcher/evaluations/${encodeURIComponent(detail.run_id)}/result`)
      .then((result) => {
        if (!(["choice-accuracy-v1", "choice-llm-judge-v1"].includes(result.protocol_version || ""))) return null;
        return getJson<ChoiceReport>(`/api/launcher/evaluations/${encodeURIComponent(detail.run_id)}/choice-report`);
      })
      .then((report) => { if (alive && report) setChoiceReport(report); })
      .catch((reason) => { if (alive) setChoiceReportError(reason instanceof Error ? reason.message : "逐题报告读取失败"); })
      .finally(() => { if (alive) setChoiceReportLoading(false); });
    return () => { alive = false; };
  }, [detail?.run_id, detail?.run_kind]);
  useEffect(() => {
    setLiveLauncherRun(null);
    if (!detail || detail.run_kind !== "evaluation" || !["starting", "running"].includes(detail.status)) return;
    let alive = true;
    const refreshLiveEvaluation = () => getJson<LiveLauncherRun>(
      `/api/launcher/runs/${encodeURIComponent(detail.run_id)}`,
    ).then((value) => { if (alive) setLiveLauncherRun(value); })
      .catch(() => { if (alive) setLiveLauncherRun(null); });
    void refreshLiveEvaluation();
    const timer = window.setInterval(() => { void refreshLiveEvaluation(); }, 1500);
    return () => { alive = false; window.clearInterval(timer); };
  }, [detail?.run_id, detail?.run_kind, detail?.status]);
  useEffect(() => {
    setDetail(null); setDetailError("");
    if (!filters.run) return;
    getJson<RunDetailData>(`/api/runs/${encodeURIComponent(filters.run)}`)
      .then(setDetail).catch((e) => setDetailError(e instanceof Error ? e.message : "读取运行详情失败"));
  }, [filters.run]);

  useEffect(() => {
    setEvaluationPreflight(null); setEvaluationError(""); setEvaluationProfileId(""); setEvaluationDatasetId("");
    setEvaluationRequest("idle");
    setHandoffCatalogLoaded(false);
    if (!detail || detail.run_kind !== "training" || detail.status !== "completed" || !detail.stage) {
      setEvaluationProfiles([]); setEvaluationDatasets([]); setHandoffCatalogLoaded(true); return;
    }
    let alive = true;
    Promise.all([
      getJson<{ profiles: StageProfile[] }>("/api/training/stage-profiles"),
      getJson<WorkbenchOptions>("/api/training/workbench-options"),
      getJson<{ parent_model_id?: string }>("/api/launcher/capabilities"),
    ]).then(([catalog, options, capability]) => {
      if (!alive) return;
      setStageProfiles(catalog.profiles); setRegisteredModels(options.models);
      setFixedMiniParentModelId(capability.parent_model_id || "");
      const profiles = [...new Map(catalog.profiles.filter((item) => item.stage === detail.stage)
        .flatMap((item) => item.evaluation_profiles).map((item) => [item.profile_id, item])).values()];
      setEvaluationProfiles(profiles); setEvaluationDatasets(options.datasets);
      setEvaluationProfileId(profiles[0]?.profile_id || "");
      setHandoffCatalogLoaded(true);
    }).catch((reason) => {
      if (alive) { setEvaluationError(reason instanceof Error ? reason.message : "评测能力读取失败"); setHandoffCatalogLoaded(true); }
    });
    return () => { alive = false; };
  }, [detail?.run_id, detail?.run_kind, detail?.stage, detail?.status]);

  const liveDetail = Boolean(detail && (
    ["queued", "starting", "running"].includes(detail.status)
    || detail.evaluations.some((item) => ["queued", "starting", "running"].includes(item.status))
  ));
  useEffect(() => {
    if (!filters.run || !liveDetail) return;
    const timer = window.setInterval(() => {
      getJson<RunDetailData>(`/api/runs/${encodeURIComponent(filters.run)}`)
        .then(setDetail).catch((e) => setDetailError(e instanceof Error ? e.message : "刷新运行详情失败"));
    }, 1500);
    return () => window.clearInterval(timer);
  }, [filters.run, liveDetail]);

  function changeFilter(key: "stage" | "status" | "q", value: string) {
    const next = { ...filters, [key]: value, offset: 0 };
    setFilters(next); writeQuery(next);
  }
  function openRun(run: string) {
    writeQuery(filters, run, true);
    setFilters({ ...filters, run });
  }
  function returnToList() {
    const next = { ...filters, run: "" };
    writeQuery(next, ""); setFilters(next);
  }

  const evaluationDatasetRequirement = EVALUATION_DATASET_REQUIREMENTS[evaluationProfileId];
  const needsEvaluationDataset = Boolean(evaluationDatasetRequirement);
  const compatibleEvaluationDatasets = evaluationDatasetRequirement
    ? evaluationDatasets.filter((item) => item.stage === evaluationDatasetRequirement.stage
      && item.role === evaluationDatasetRequirement.role)
    : [];
  const lineageComplete = detail?.lineage?.["lineage_status"] === "complete";
  const handoffModels = detail && lineageComplete && detail.status === "completed"
    ? (detail.handoff_models || []).flatMap((registered) => {
      if (!registered.model_id || registered.source_run_id !== detail.run_id || registered.registry_count !== 1
        || registered.digest?.status !== "available" || registered.path_available !== true) return [];
      const catalogEntry = registeredModels.find((model) => model.model_id === registered.model_id);
      if (!catalogEntry || catalogEntry.registry_count !== 1 || catalogEntry.produced_by_run_id !== registered.produced_by_run_id
        || catalogEntry.digest?.status !== "available" || catalogEntry.digest.value !== registered.digest.value) return [];
      const compatibleProfiles = stageProfiles.filter((profile) => {
        if (!profile.parent_model_stages.includes(registered.stage)) return false;
        if (profile.parent_model_rule === "approved_mini_cpt") return profile.stage === "sft" && registered.model_id === fixedMiniParentModelId;
        if (profile.parent_model_rule === "registered_sft_merged") return registered.stage === "sft" && registered.kind === "merged";
        return profile.parent_model_rule === "external_base" && registered.stage === "base";
      });
      return [{ model: registered, profiles: compatibleProfiles }];
    }) : [];
  const handoffBlocker = !handoffCatalogLoaded ? "正在核对模型登记与兼容的下游阶段…"
    : detail?.status !== "completed" ? "只有成功完成的训练 Run 才能交接模型。"
    : !lineageComplete ? "该 Run 缺少完整 lineage，不能证明模型来源；请先修复/核对记录。"
      : handoffModels.length === 0 ? "没有同时满足产物摘要、登记身份和 producing Run 核验的模型。" : "";
  function handoffToNextStage(model: RegisteredModel, profile: StageProfile) {
    if (!detail) return;
    const params = new URLSearchParams({ page: "launcher", stage: profile.stage, profile_id: profile.profile_id,
      parent_model_id: model.model_id, source_run_id: detail.run_id, return_to: "runs" });
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }
  async function launcherPost<T>(path: string, body: Record<string, string | boolean>): Promise<T> {
    const capability = await getJson<{ enabled: boolean; token?: string; reason?: string }>("/api/launcher/capabilities");
    if (!capability.enabled || !capability.token) throw new Error(capability.reason || "本地 Launcher 未启用");
    const response = await fetch(path, { method: "POST", headers: { "Content-Type": "application/json", "X-Local-Launcher-Token": capability.token }, body: JSON.stringify(body) });
    const value = await response.json();
    if (!response.ok) throw new ApiRequestError(response.status, value.detail || `本地服务返回 ${response.status}`);
    return value as T;
  }
  async function retryChoiceJudgment(row: ChoiceReportRow) {
    if (!detail || row.judge_outcome !== "provider_error" || retryingChoiceRow) return;
    const uncertainNotice = row.judge_request_uncertain
      ? "原请求结果不确定，百炼可能已经处理并计费；再次请求可能产生重复费用。"
      : "";
    const confirmed = window.confirm(
      `将重新发送以下内容到阿里百炼：题目、选项、标准答案、被测模型原始输出。\n\n${uncertainNotice}\n每次重试都会单独记录，原评测结果和指标不会修改。确认发送吗？`,
    );
    if (!confirmed) return;
    setRetryingChoiceRow(row.id);
    setChoiceRetryErrors((current) => ({ ...current, [row.id]: "" }));
    try {
      await launcherPost(`/api/launcher/evaluations/${encodeURIComponent(detail.run_id)}/choice-retry`, {
        question_id: row.id, confirm_external_submission: true,
      });
      const report = await getJson<ChoiceReport>(`/api/launcher/evaluations/${encodeURIComponent(detail.run_id)}/choice-report`);
      setChoiceReport(report);
    } catch (reason) {
      setChoiceRetryErrors((current) => ({ ...current, [row.id]: reason instanceof Error ? reason.message : "重试判别失败" }));
    } finally {
      setRetryingChoiceRow(null);
    }
  }
  async function preflightEvaluation() {
    if (!detail || !evaluationProfileId) return;
    setEvaluationBusy(true); setEvaluationRequest("preflight_pending"); setEvaluationError(""); setEvaluationPreflight(null);
    try {
      const payload: Record<string, string> = { training_run_id: detail.run_id, profile_id: evaluationProfileId };
      if (needsEvaluationDataset && evaluationDatasetId) payload.dataset_id = evaluationDatasetId;
      if (evaluationProfileId === "choice-accuracy-v1") payload.scoring_strategy = evaluationScoringStrategy;
      setEvaluationPreflight(await launcherPost<EvaluationPreflight>("/api/training/evaluation/preflight", payload));
    } catch (reason) { setEvaluationError(reason instanceof Error ? reason.message : "评测预检失败"); }
    finally { setEvaluationBusy(false); setEvaluationRequest("idle"); }
  }
  async function inspectEvaluationStart(initialCheck = false) {
    const runId = evaluationPreflight?.evaluation_run_id;
    if (!runId) { setEvaluationRequest("token_expired"); return; }
    try {
      const observed = await getJson<{ run_id: string }>(`/api/launcher/runs/${encodeURIComponent(runId)}`);
      if (observed.run_id === runId) { openRun(runId); return; }
    } catch { /* A missing Run does not prove the earlier POST was never accepted. */ }
    if (!initialCheck) {
      setEvaluationRequest("token_expired");
      setEvaluationPreflight(null);
      setEvaluationError("没有查到此次预检的评测 Run；请重新预检确认输入后再决定是否启动，不会自动重试。");
    }
  }
  async function startEvaluation() {
    if (!evaluationPreflight?.confirmation_token) return;
    setEvaluationBusy(true); setEvaluationRequest("submit_pending"); setEvaluationError("");
    try {
      const started = await launcherPost<{ run_id: string }>("/api/training/evaluation/start", { confirmation_token: evaluationPreflight.confirmation_token });
      openRun(started.run_id);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "启动评测失败";
      if (reason instanceof ApiRequestError) {
        setEvaluationPreflight(null);
        setEvaluationRequest("token_expired");
        setEvaluationError(`${message}。启动未确认；请重新预检，不会自动重试。`);
      } else {
        setEvaluationError(`${message}。正在核实预检对应的 Run，不会自动重试。`);
        setEvaluationRequest("submission_unknown");
        await inspectEvaluationStart(true);
      }
    }
    finally { setEvaluationBusy(false); }
  }
  async function preflightBaseEvaluation() {
    if (!baseModelId) return;
    setBaseEvaluationBusy(true); setBaseEvaluationRequest("preflight_pending"); setBaseEvaluationError(""); setBaseEvaluationPreflight(null);
    try {
      const payload: Record<string, string> = { model_id: baseModelId, profile_id: baseEvaluationProfile };
      if (baseEvaluationProfile === "choice-accuracy-v1") {
        payload.dataset_id = baseEvaluationDatasetId;
        payload.scoring_strategy = baseScoringStrategy;
      }
      setBaseEvaluationPreflight(await launcherPost<EvaluationPreflight>("/api/training/evaluation/preflight", payload));
    } catch (reason) { setBaseEvaluationError(reason instanceof Error ? reason.message : "Base 模型评测预检失败"); }
    finally { setBaseEvaluationBusy(false); setBaseEvaluationRequest("idle"); }
  }
  async function refreshBaseModelCatalog() {
    setBaseCatalogLoading(true);
    setBaseEvaluationError("");
    try {
      let catalog: { status: string; models: (RegisteredModel & { available?: boolean })[] } | null = null;
      let options: WorkbenchOptions | null = null;
      for (let attempt = 0; attempt < 30; attempt += 1) {
        const [current, workbenchOptions] = await Promise.all([
          getJson<{ status: string; models: (RegisteredModel & { available?: boolean })[] }>("/api/model-playground/models"),
          getJson<WorkbenchOptions>("/api/training/workbench-options"),
        ]);
        if (current.status === "ready") { catalog = current; options = workbenchOptions; break; }
        if (current.status === "failed") throw new Error("本地模型目录核验失败");
        await new Promise((resolve) => window.setTimeout(resolve, 500));
      }
      if (!catalog || catalog.status !== "ready" || !options) throw new Error("本地模型目录核验超时");
      const models = catalog.models.filter((model) => model.kind === "base" && model.available);
      const datasets = options.datasets.filter((dataset) => dataset.stage === "eval" && dataset.role === "evaluation_only");
      setBaseModels(models);
      setEvaluationDatasets(datasets);
      if (baseModelId && !models.some((model) => model.model_id === baseModelId)) {
        setBaseModelId("");
        setBaseEvaluationPreflight(null);
      }
      if (baseEvaluationDatasetId && !datasets.some((dataset) => dataset.dataset_id === baseEvaluationDatasetId)) {
        setBaseEvaluationDatasetId("");
        setBaseEvaluationPreflight(null);
      }
    } catch (reason) {
      setBaseEvaluationError(reason instanceof Error ? reason.message : "无法刷新已登记 Base 模型目录");
    } finally {
      setBaseCatalogLoading(false);
    }
  }
  async function inspectBaseEvaluationStart(initialCheck = false) {
    const runId = baseEvaluationPreflight?.evaluation_run_id;
    if (!runId) { setBaseEvaluationRequest("token_expired"); return; }
    try {
      const observed = await getJson<{ run_id: string }>(`/api/launcher/runs/${encodeURIComponent(runId)}`);
      if (observed.run_id === runId) { openRun(runId); return; }
    } catch { /* A missing Run does not prove that the POST was not accepted. */ }
    if (!initialCheck) {
      setBaseEvaluationRequest("token_expired"); setBaseEvaluationPreflight(null);
      setBaseEvaluationError("尚未查到对应评测 Run；请核对输入并重新预检，不会自动重复启动。");
    }
  }
  async function startBaseEvaluation() {
    if (!baseEvaluationPreflight?.confirmation_token) return;
    setBaseEvaluationBusy(true); setBaseEvaluationRequest("submit_pending"); setBaseEvaluationError("");
    try {
      const started = await launcherPost<{ run_id: string }>("/api/training/evaluation/start", { confirmation_token: baseEvaluationPreflight.confirmation_token });
      openRun(started.run_id);
    } catch (reason) {
      const message = reason instanceof Error ? reason.message : "Base 模型评测启动失败";
      if (reason instanceof ApiRequestError) {
        setBaseEvaluationPreflight(null); setBaseEvaluationRequest("token_expired");
        setBaseEvaluationError(`${message}。启动未确认；请重新预检，不会自动重试。`);
      } else {
        setBaseEvaluationError(`${message}。正在核实预检对应的 Run，不会自动重试。`);
        setBaseEvaluationRequest("submission_unknown");
        await inspectBaseEvaluationStart(true);
      }
    }
    finally { setBaseEvaluationBusy(false); }
  }
  function analyzeEvaluation() {
    if (!detail) return;
    const model = detail.provenance.find((item) => item.kind === "model")?.id;
    const evaluationDataset = detail.provenance.find((item) => item.kind === "dataset" && /eval|holdout/i.test(item.role || ""))?.id;
    const params = new URLSearchParams({ page: "targeted", eval_id: detail.run_id, return_to: "runs", run_id: detail.run_id });
    if (model) params.set("model_id", model);
    if (evaluationDataset) params.set("dataset_id", evaluationDataset);
    if (detail.stage) params.set("stage", detail.stage);
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    window.dispatchEvent(new PopStateEvent("popstate"));
  }
  const visibleChoiceRows = (choiceReport?.rows || []).filter((row) =>
    (choiceReportFilter === "all" || row.status === choiceReportFilter)
    && (choiceReportType === "all" || row.question_type === choiceReportType));
  const choiceTypeLabel: Record<string, string> = {
    single_choice: "单选", multiple_choice: "多选", true_false: "判断",
  };
  const judgePhaseLabel: Record<string, string> = {
    preparing: "准备评测", loading_model: "加载被测模型", ready: "模型已就绪",
    local_generation: "本地生成模型回答", bailian_judging: "等待百炼评判",
    recorded: "已保存本题结果", finalizing: "汇总结果", completed: "评测完成",
  };
  const choiceAnswerLabel = (answer: string | null, row: ChoiceReportRow) => {
    if (answer == null) return "未能解析";
    if (answer === "true") return "正确 / true";
    if (answer === "false") return "错误 / false";
    return answer.split(",").map((label) => {
      const index = label.charCodeAt(0) - 65;
      return index >= 0 ? `${label}${row.options[index] ? ` · ${row.options[index]}` : ""}` : label;
    }).join("、");
  };
  const choiceReportCounts = (choiceReport?.rows || []).reduce((counts, row) => {
    counts[row.status] = (counts[row.status] || 0) + 1;
    return counts;
  }, {} as Record<string, number>);
  const isChoiceJudgeMetrics = choiceReport?.protocol_version === "choice-llm-judge-v1"
    || detail?.metrics.some((metric) => metric.name.startsWith("choice_llm_judge.")) === true;
  const selectedChoiceMetrics = choiceMetricsView === "current"
    ? choiceReport?.current_metrics
    : choiceReport?.original_metrics;
  const choiceMetricGroupOrder = ["overall", "single_choice", "multiple_choice", "true_false"];
  const choiceMetricGroupLabel: Record<string, string> = {
    overall: "总体", single_choice: "单选题", multiple_choice: "多选题", true_false: "判断题",
  };
  const judgeOutcomeLabel: Record<string, string> = {
    correct: "答对", incorrect: "答错", undetermined: "无法判断", provider_error: "API 错误",
  };

  return <main className="runs-page">
    <header className="hero runs-hero">
      <div><div className="eyebrow">EXPERIMENT WORKSPACE ／ RUNS</div>
        <h1>{filters.run ? "运行详情与证据。" : "每次运行，都能追溯。"}</h1>
        <p>只读浏览训练、合并与评测记录；失败和缺失证据也会如实保留。</p></div>
      <div className="runs-hero-actions">
        {filters.run && <button className="button secondary" onClick={returnToList}>返回筛选结果</button>}
        <button className="button secondary" onClick={onBack}>数据工作区</button>
      </div>
    </header>
    {filters.run ? <section className="runs-panel" aria-labelledby="run-detail-title">
      {detailError && <div className="notice danger" role="alert">{detailError}</div>}
      {!detail && !detailError && <p role="status">正在读取运行详情…</p>}
      {detail && <>
        {liveDetail && <p className="runs-live-status" role="status">● 运行状态、日志和评测结果每 1.5 秒自动更新</p>}
        <div className="runs-detail-heading"><div><div className="eyebrow">{detail.run_kind === "evaluation" ? "EVALUATION RUN" : "TRAINING RUN"}</div><h2 id="run-detail-title">{displayRunName(detail)}</h2><p>{stageLabel[detail.stage || ""] || detail.stage || "阶段未记录"} · {detail.run_kind === "evaluation" ? "独立评测" : "训练运行"}</p></div><span className={`badge ${detail.status === "completed" ? "success" : ["failed", "oom"].includes(detail.status) ? "danger" : "warning"}`}>{statusLabel[detail.status] || detail.status}</span></div>
        <dl className="runs-facts">
          <div><dt>开始时间</dt><dd>{formatDate(detail.started_at)}</dd></div><div><dt>完成时间</dt><dd>{formatDate(detail.completed_at)}</dd></div>
          <div><dt>输入证据</dt><dd>{detail.provenance.length} 项{detail.provenance.length === 0 ? " · 未记录" : " · 详情见下方"}</dd></div>
          <div><dt>产物证据</dt><dd>{detail.produced_artifacts?.length || 0} 项{detail.status !== "completed" ? " · 运行未成功，不代表可交接" : ""}</dd></div>
        </dl>
        <details className="runs-technical-evidence"><summary>查看 Run ID、实验 ID 与记录路径</summary><dl className="runs-facts">
          <div><dt>Run ID</dt><dd><code>{detail.run_id}</code></dd></div><div><dt>experiment_id</dt><dd><code>{detail.experiment_id || "未记录"}</code></dd></div>
          <div><dt>记录来源</dt><dd><code>{detail.source_ref}</code></dd></div><div><dt>记录路径</dt><dd><code>{detail.source_refs?.join(" · ") || detail.source_ref}</code></dd></div>
        </dl></details>
        {detail.parent_run_id && <p className={detail.parent_available ? "runs-lineage-ok" : "runs-lineage-warning"}>父训练 Run：<button className="runs-link" onClick={() => openRun(detail.parent_run_id!)}>查看上游运行</button>{detail.parent_available ? " · 已找到" : " · 外部或当前工作区不可用"}</p>}
        {detail.run_kind === "evaluation" && <section className="runs-evaluation-workbench" aria-label="评测结果后续操作"><h3>下一步</h3><p>将当前评测 Run、模型和评测数据身份带入薄弱场景分析；这里只导航，不会生成数据或启动训练。</p><button className="button secondary" onClick={analyzeEvaluation}>分析 / 补充数据 →</button></section>}
        {detail.run_kind === "training" && <section className="runs-evaluation-workbench" aria-labelledby="run-handoff-title">
          <h3 id="run-handoff-title">将模型交接到下一阶段</h3>
          {handoffBlocker ? <p className="runs-lineage-warning" role="status">{handoffBlocker}</p> : <>
            <p>只展示已登记且 lineage/hash 与当前 Run 相符的模型。可以试用模型，或将其交接为兼容阶段的父模型；交接不会自动选数据、预检或启动训练。</p>
            {handoffModels.map(({ model, profiles }) => <article key={model.model_id} className="runs-handoff-model">
              <strong>{stageLabel[model.stage] || model.stage} · {model.kind} 模型</strong>
              <small>来源 Run 与产物摘要已匹配</small>
              <details><summary>查看模型身份</summary><code>{model.model_id}</code><small>来源 Run：{model.produced_by_run_id}</small></details>
              <div className="runs-handoff-actions">{onTryModel && <button className="button primary" onClick={() => onTryModel(model.model_id, detail.run_id)}>试用此模型 →</button>}{profiles.map((profile) => <button key={profile.profile_id} className="button secondary"
                onClick={() => handoffToNextStage(model, profile)} aria-label={`将 ${model.model_id} 用作 ${profile.stage} 父模型`}>
                用作 {stageLabel[profile.stage] || profile.stage} 父模型
              </button>)}{!profiles.length && <small>当前没有可直接交接的下游训练配置，但模型仍可用于本地体验。</small>}</div>
            </article>)}
          </>}
        </section>}
        {detail.run_kind === "training" && detail.status === "completed" && <section className="runs-evaluation-workbench" aria-labelledby="run-evaluation-title">
          <h3 id="run-evaluation-title">对这个模型进行阶段评测</h3>
          {evaluationError && <div className="notice danger" role="alert">{evaluationError}</div>}
          {evaluationProfiles.length === 0 && !evaluationError && <p role="status">正在读取阶段评测 profile…</p>}
          {evaluationProfiles.length > 0 && <div className="runs-evaluation-controls">
            <label>评测方案<select aria-label="选择阶段评测方案" value={evaluationProfileId} onChange={(event) => { setEvaluationProfileId(event.target.value); setEvaluationScoringStrategy("exact_match"); setEvaluationDatasetId(""); setEvaluationPreflight(null); setEvaluationRequest("idle"); setEvaluationError(""); }}>
              {evaluationProfiles.map((profile) => <option key={profile.profile_id} value={profile.profile_id}>{profile.profile_id} · {profile.protocol}</option>)}
            </select></label>
            {needsEvaluationDataset && <label>{evaluationDatasetRequirement.label}<select aria-label="选择登记评测数据集" value={evaluationDatasetId} onChange={(event) => { setEvaluationDatasetId(event.target.value); setEvaluationPreflight(null); setEvaluationRequest("idle"); setEvaluationError(""); }}>
              <option value="">{compatibleEvaluationDatasets.length ? "请选择评测数据集" : `没有符合条件的${evaluationDatasetRequirement.label}`}</option>{compatibleEvaluationDatasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id} disabled={dataset.eligible === false}>{dataset.dataset_id} · {dataset.record_count} 条{dataset.eligible === false ? ` · 不可用：${dataset.block_reason || "登记证据不完整"}` : ""}</option>)}
            </select></label>}
            {evaluationProfileId === "choice-accuracy-v1" && <label>评分策略<select aria-label="选择客观题评分策略" value={evaluationScoringStrategy} onChange={(event) => { setEvaluationScoringStrategy(event.target.value); setEvaluationPreflight(null); setEvaluationRequest("idle"); setEvaluationError(""); }}>
              <option value="exact_match">精确匹配（默认）</option><option value="bailian_llm_judge">阿里百炼大模型评判</option>
            </select></label>}
          </div>}
          {evaluationProfiles.length > 0 && <TaskStatusCard
            input={{ request: evaluationRequest, preflight: evaluationPreflight ? evaluationPreflight.ok ? "ready" : "blocked" : "none" }}
            actionLabel={evaluationRequest === "idle" && !evaluationPreflight ? "生成评测预检"
              : evaluationRequest === "idle" && evaluationPreflight?.ok ? "确认并启动独立评测 Run" : undefined}
            actionDisabled={evaluationBusy || (!evaluationProfileId || (needsEvaluationDataset && !evaluationDatasetId))}
            onAction={(action) => {
              if (action === "preflight") void preflightEvaluation();
              else if (action === "submit") void startEvaluation();
              else if (action === "inspect") void inspectEvaluationStart();
              else if (action === "edit") setEvaluationPreflight(null);
            }} />}
          {evaluationPreflight && <div className={`runs-evaluation-confirmation ${evaluationPreflight.ok ? "ready" : "blocked"}`} role={evaluationPreflight.ok ? "status" : "alert"}>
            {evaluationPreflight.blockers.map((blocker) => <p key={blocker}>{blocker}</p>)}
            {evaluationPreflight.ok && <><p>配置已核对：{evaluationProfileId}；实际指标与分母将在评测 Run 中显示。</p>
              {evaluationPreflight.dataset && <p>评测集：<code>{evaluationPreflight.dataset.dataset_id}</code> · 样本数：{evaluationPreflight.dataset.record_count}</p>}
              {evaluationPreflight.external_judge && <div className="notice warning" role="status">
                <p>{evaluationPreflight.external_judge.external_data_notice}</p>
                <p>百炼 Profile：<code>{evaluationPreflight.external_judge.provider_profile_id}</code> · 评判模型：{evaluationPreflight.external_judge.provider_model} · 预计 API 请求：{evaluationPreflight.external_judge.expected_request_count} 次。</p>
                <p>{evaluationPreflight.external_judge.billing_notice}</p>
                <p>点击“确认并启动”即确认将上述评测内容发送给阿里云百炼。</p>
              </div>}
              <details><summary>查看评测技术身份</summary><p>模型：<code>{evaluationPreflight.model?.model_id}</code> · 协议：{evaluationPreflight.protocol_version} · 输出：<code>{evaluationPreflight.output}</code></p><p>指标：{evaluationPreflight.metrics?.join("、") || "以评测报告实际产出为准"}</p></details>
              </>}
          </div>}
        </section>}
        {detail.warnings?.map((warning) => <div key={warning} className="notice warning" role="status">{warning}</div>)}
        {detail.run_kind === "evaluation" && detail.status === "running"
          && liveLauncherRun?.scoring_strategy === "bailian_llm_judge" && liveLauncherRun.choice_judge_progress && <section className="runs-choice-report" aria-label="百炼评判实时进度" role="status">
            {(() => {
              const progress = liveLauncherRun.choice_judge_progress!;
              const total = progress.total ?? 0;
              return <>
                <div className="runs-choice-report-heading"><div><h3>百炼评判实时进度</h3><p className="runs-muted">Launcher 每 1.5 秒自动更新 · 当前阶段：{judgePhaseLabel[progress.phase] || "评测运行中"}</p></div>
                  <span>{progress.completed} / {total || "?"} 题</span></div>
                {total > 0 && <progress className="runs-choice-progress-bar" aria-label="百炼逐题评判进度" max={total} value={Math.min(progress.completed, total)} />}
                <div className="runs-choice-report-summary" aria-label="已完成题目判定计数">
                  <span><strong>{progress.counts.correct}</strong> 答对</span>
                  <span><strong>{progress.counts.incorrect}</strong> 答错</span>
                  <span><strong>{progress.counts.undetermined}</strong> 无法判断</span>
                  <span><strong>{progress.counts.provider_errors}</strong> API 错误</span>
                </div>
                {progress.events.length > 0 && <details><summary>查看安全进度事件（最近 {Math.min(progress.events.length, 6)} 条）</summary>
                  <ol>{progress.events.slice(-6).map((event, index) => <li key={`${event.phase}-${event.completed}-${index}`}>
                    {event.phase === "recorded" ? `第 ${event.current_question ?? event.completed} 题已记录` :
                      event.current_question ? `第 ${event.current_question} 题：${judgePhaseLabel[event.phase] || "处理中"}` :
                        judgePhaseLabel[event.phase] || "评测运行中"}
                    {event.error_category ? ` · API 错误（${event.error_category}）` : ""}
                  </li>)}</ol>
                </details>}
              </>;
            })()}
          </section>}
        {detail.run_kind === "evaluation" && (choiceReportLoading || choiceReport || choiceReportError) && <section className="runs-choice-report" aria-labelledby="choice-report-title">
          <div className="runs-choice-report-heading"><div><h3 id="choice-report-title">逐题测评报告</h3><p className="runs-muted">{choiceReport?.protocol_version === "choice-llm-judge-v1" ? `百炼模型评判 · ${choiceReport.provider_model || "裁判模型未记录"} · Profile ${choiceReport.provider_profile_id || "未记录"} · ${choiceReport.request_count ?? "?"} 次请求 · 提示词 ${choiceReport.prompt_version || "未记录"}` : "按题目核对标准答案与模型作答，解析失败会单独列出。"}</p></div>
            {choiceReport && <span>{visibleChoiceRows.length} / {choiceReport.sample_count} 题 · 已完成 {choiceReport.evaluated_count}</span>}</div>
          {!["completed", "partial", "failed", "interrupted"].includes(detail.status) && <p className="runs-muted" role="status">评测完成后会在这里生成逐题报告。</p>}
          {choiceReportLoading && !choiceReport && <p role="status">正在核对评测题目与逐题预测…</p>}
          {choiceReportError && <div className="notice warning" role="status">逐题报告暂不可用：{choiceReportError}</div>}
          {choiceReport && <>
            {choiceReport.protocol_version === "choice-llm-judge-v1" && <div className="notice warning" role="status">这是百炼语义评判协议，准确率按已判定题目计算，判定覆盖率单独显示；不可与精确匹配分数直接比较。</div>}
            {choiceReport.status === "partial" && <div className="notice warning" role="status">{choiceReport.evaluated_count < choiceReport.sample_count
              ? `本次评测仅完成 ${choiceReport.evaluated_count} / ${choiceReport.sample_count} 题；未完成题目不会被计作错误。`
              : "全部题目均已尝试，但部分百炼请求失败；API 错误不会计作模型答错。"}</div>}
            <div className="runs-choice-report-summary" aria-label="逐题结果概览">
              <span><strong>{choiceReportCounts.correct || 0}</strong> 答对</span>
              <span><strong>{choiceReportCounts.incorrect || 0}</strong> 答错</span>
              {choiceReport.protocol_version === "choice-llm-judge-v1"
                ? <><span><strong>{choiceReportCounts.undetermined || 0}</strong> 无法判断</span><span><strong>{choiceReportCounts.provider_error || 0}</strong> API 错误</span></>
                : <span><strong>{choiceReportCounts.parse_failed || 0}</strong> 解析失败</span>}
              <span><strong>{choiceReportCounts.not_evaluated || 0}</strong> 未完成</span>
            </div>
            <details className="runs-choice-report-details">
              <summary>查看逐题明细（{choiceReport.sample_count} 题）</summary>
              <div className="runs-choice-report-content">
                <div className="runs-choice-report-controls">
                  <label>判定<select aria-label="按答题结果筛选" value={choiceReportFilter} onChange={(event) => setChoiceReportFilter(event.target.value)}>
                    <option value="all">全部题目</option><option value="incorrect">答错</option><option value="correct">答对</option>
                    {choiceReport.protocol_version === "choice-llm-judge-v1"
                      ? <><option value="undetermined">无法判断</option><option value="provider_error">API 错误</option></>
                      : <option value="parse_failed">解析失败</option>}
                    <option value="not_evaluated">未完成</option>
                  </select></label>
                  <label>题型<select aria-label="按题型筛选" value={choiceReportType} onChange={(event) => setChoiceReportType(event.target.value)}>
                    <option value="all">全部题型</option><option value="single_choice">单选</option><option value="multiple_choice">多选</option><option value="true_false">判断</option>
                  </select></label>
                  <span className="runs-choice-report-match-count" aria-live="polite">当前显示 {visibleChoiceRows.length} / {choiceReport.sample_count} 题</span>
                </div>
                {!visibleChoiceRows.length ? <div className="runs-empty"><strong>没有符合条件的题目</strong><p>换一个判定结果或题型筛选。</p></div> : <div className="runs-choice-question-list">
                  {visibleChoiceRows.map((row) => <article className="runs-choice-question" key={row.id}>
                    <header><strong>题目 {(choiceReport?.rows.findIndex((item) => item.id === row.id) ?? -1) + 1}</strong><code>{row.id}</code><span className="badge neutral">{choiceTypeLabel[row.question_type] || row.question_type}</span>
                      <span className={`badge ${row.status === "correct" ? "success" : row.status === "incorrect" ? "danger" : "neutral"}`}>
                        {row.status === "correct" ? "答对" : row.status === "incorrect" ? "答错" : row.status === "parse_failed" ? "解析失败" : row.status === "undetermined" ? "无法判断" : row.status === "provider_error" ? "API 错误" : "未完成"}
                      </span></header>
                    <p className="runs-choice-question-text">{row.question}</p>
                    {row.options.length > 0 && <ol className="runs-choice-options" type="A">{row.options.map((option, optionIndex) => <li key={`${row.id}-${optionIndex}`}>{option}</li>)}</ol>}
                    <div className={`runs-choice-answers ${choiceReport.protocol_version === "choice-llm-judge-v1" ? "judge" : ""}`}><div><small>标准答案</small><strong>{choiceAnswerLabel(row.gold, row)}</strong></div>
                      {choiceReport.protocol_version === "choice-llm-judge-v1" ? <><div><small>原始判定</small><strong>{row.judge_outcome ? judgeOutcomeLabel[row.judge_outcome] : "未完成"}</strong></div><div><small>当前有效判定</small><strong>{row.effective_judge_outcome ? judgeOutcomeLabel[row.effective_judge_outcome] : "未完成"}</strong></div></>
                        : <div><small>模型答案</small><strong>{choiceAnswerLabel(row.prediction, row)}</strong></div>}</div>
                    {row.judge_rationale && <p className="runs-muted"><strong>评判理由：</strong>{row.judge_rationale}</p>}
                    {row.judge_error && <p className="runs-muted"><strong>错误类别：</strong>{row.judge_error}</p>}
                    {row.judge_request_uncertain && <p className="runs-muted">请求结果不确定，百炼可能已处理该请求；系统未自动重试。</p>}
                    {row.provider_request_id && <small>百炼请求 ID：<code>{row.provider_request_id}</code></small>}
                    {row.retry_attempts?.length ? <div className="runs-choice-retry-history" aria-label="手动重试判定记录"><strong>手动重试记录（{row.retry_attempts.length} 次）</strong>{row.retry_attempts.map((attempt) => <div key={`${row.id}-retry-${attempt.attempt_number}`}><span>第 {attempt.attempt_number} 次：{attempt.judge_outcome === "provider_error" ? `API 错误${attempt.judge_error ? `（${attempt.judge_error}）` : ""}` : attempt.judge_outcome === "undetermined" ? "无法判断" : attempt.judge_outcome === "correct" ? "判为正确" : "判为错误"}</span>{attempt.judge_rationale && <p className="runs-muted">{attempt.judge_rationale}</p>}{attempt.judge_request_uncertain && <p className="runs-muted">请求结果不确定，可能已被百炼处理。</p>}{attempt.provider_request_id && <small>请求 ID：<code>{attempt.provider_request_id}</code></small>}</div>)}</div> : null}
                    {row.judge_outcome === "provider_error" && <div className="runs-choice-retry-action"><button type="button" className="secondary" disabled={retryingChoiceRow !== null} onClick={() => void retryChoiceJudgment(row)}>{retryingChoiceRow === row.id ? "正在重新判别…" : "重试判别"}</button>{choiceRetryErrors[row.id] && <p className="notice warning" role="alert">{choiceRetryErrors[row.id]}</p>}</div>}
                    {row.raw_output && <details className="runs-choice-raw"><summary>查看模型原始输出</summary><pre>{row.raw_output}</pre></details>}
                  </article>)}
                </div>}
              </div>
            </details>
          </>}
        </section>}
        <div className="runs-section-grid">
          <section><h3>评测指标</h3>{isChoiceJudgeMetrics ? choiceReport?.original_metrics && choiceReport.current_metrics ? <>
            <div className="runs-choice-metric-tabs" role="group" aria-label="选择评测指标视图">
              <button type="button" className={choiceMetricsView === "current" ? "button primary" : "button secondary"} aria-pressed={choiceMetricsView === "current"} onClick={() => setChoiceMetricsView("current")}>重试后当前指标</button>
              <button type="button" className={choiceMetricsView === "original" ? "button primary" : "button secondary"} aria-pressed={choiceMetricsView === "original"} onClick={() => setChoiceMetricsView("original")}>原始评测指标</button>
            </div>
            <p className="runs-muted">{choiceMetricsView === "current" ? "按每题最新一次成功的重试判定重新汇总；仍失败的重试不会覆盖已成功的判定。" : "来自原始评测记录，重试不会修改这组指标。"}</p>
            <div className="runs-choice-metric-groups">{choiceMetricGroupOrder.map((group) => {
              const metric = selectedChoiceMetrics?.[group];
              if (!metric) return null;
              return <article className={`runs-choice-metric-group ${group === "overall" ? "overall" : ""}`} key={`${choiceMetricsView}-${group}`}>
                <h4>{choiceMetricGroupLabel[group]}</h4>
                <div className="runs-choice-metric-cards">
                  <div><span>语义判定准确率</span><strong>{metric.accuracy == null ? "未测量" : formatMetricValue("choice_llm_judge.accuracy", metric.accuracy, metric.status)}</strong><small>{metric.correct} / {metric.decided_denominator} 道已判定题</small></div>
                  <div><span>判定覆盖率</span><strong>{metric.decision_coverage == null ? "未测量" : formatMetricValue("choice_llm_judge.decision_coverage", metric.decision_coverage, metric.status)}</strong><small>{metric.decided_denominator} / {metric.denominator} 道题</small></div>
                </div>
                <div className="runs-choice-metric-counts" aria-label={`${choiceMetricGroupLabel[group]}判定结果数量`}>
                  <span>答对 <strong>{metric.correct}</strong></span><span>答错 <strong>{metric.incorrect}</strong></span>
                  <span>无法判断 <strong>{metric.undetermined}</strong></span><span>未解决 API 错误 <strong>{metric.provider_errors}</strong></span>
                  <span>未完成 <strong>{metric.not_evaluated}</strong></span>
                </div>
              </article>;
            })}</div>
          </> : <p className="runs-muted" role="status">{choiceReportLoading ? "正在读取逐题评测结果…" : "逐题指标暂不可用。"}</p>
            : detail.metrics.length ? <div className="runs-metrics">{detail.metrics.map((metric) => <article key={`${metric.name}-${metric.metric}`}><span title={`${metric.name} · ${metric.metric}`}>{metricLabel[`${metric.name}.${metric.metric}`] || `${metric.name} · ${metric.metric}`}</span><strong>{formatMetricValue(metric.name, metric.value, metric.status)}</strong><small>实际分母：{metric.denominator ?? "未记录"}{metric.denominator_unit ? ` ${metric.denominator_unit}` : ""}</small>{metric.reason && <small>{metric.reason}</small>}</article>)}</div> : <p className="runs-muted">没有可显示的真实分数；不会用 0 代替缺失。</p>}</section>
          <section><h3>数据 / 模型血缘</h3>
            {detail.lineage && <p className={detail.lineage.lineage_status === "complete" ? "runs-lineage-ok" : "runs-lineage-warning"}>
              血缘状态：{String(detail.lineage.lineage_status || "历史记录未声明")}
              {Array.isArray(detail.lineage.lineage_warnings) && detail.lineage.lineage_warnings.length > 0 ? ` · ${detail.lineage.lineage_warnings.join("；")}` : ""}
            </p>}
            <details className="runs-technical-evidence"><summary>Run 输入（每个数据集/父模型保留独立身份）</summary>
            {detail.provenance.length ? <ul className="runs-provenance">{detail.provenance.map((item, index) => <li key={`${item.id}-${index}`}><strong>{item.kind || "artifact"} · {item.role || "input"}</strong><code>{item.id || "ID 未记录"}</code><small>{item.availability || "可用性未记录"}{item.path ? ` · ${item.path}` : ""}</small><small>SHA-256：{item.digest?.status === "available" ? item.digest.value : item.digest?.reason || item.digest?.status || "未记录"}</small></li>)}</ul> : <p className="runs-muted">没有可读取的血缘记录，或该历史运行未记录 lineage。</p>}
            <strong>Run 产物</strong>
            {detail.produced_artifacts?.length ? <ul className="runs-provenance">{detail.produced_artifacts.map((item, index) => <li key={`${item.id}-${index}`}><strong>{item.kind || "artifact"} · {item.role || "output"}</strong><code>{item.id || "ID 未记录"}</code><small>{item.availability || "可用性未记录"}{item.path ? ` · ${item.path}` : ""}</small><small>SHA-256：{item.digest?.status === "available" ? item.digest.value : item.digest?.reason || item.digest?.status || "未记录"}</small></li>)}</ul> : <p className="runs-muted">没有声明 Run 产物；失败或中断运行可能没有完整产物。</p>}
            </details>
          </section>
        </div>
        {detail.evaluations.length > 0 && <section><h3>关联评测</h3><ul className="runs-related">{detail.evaluations.map((evaluation) => <li key={evaluation.run_id}><button className="runs-link" onClick={() => openRun(evaluation.run_id)}>{evaluation.run_id} · {statusLabel[evaluation.status] || evaluation.status}</button></li>)}</ul></section>}
        <section><h3>配置与产物</h3>{detail.outputs.length ? <p className="runs-muted">可读取产物 {detail.outputs.length} 项。</p> : <p className="runs-muted">没有已发现的可读产物。</p>}
          {detail.outputs.length > 0 && <details className="runs-technical-evidence"><summary>查看产物文件与路径</summary><ul className="runs-output-list">{detail.outputs.map((path) => { const filename = path.split(/[\\/]/).pop() || ""; return <li key={path}><a href={`/api/runs/${encodeURIComponent(detail.run_id)}/artifacts/${encodeURIComponent(filename)}`} target="_blank" rel="noreferrer">{filename || "查看产物"}</a><code>{path}</code></li>; })}</ul></details>}
          {detail.config ? <details className="runs-technical-evidence"><summary>查看解析配置</summary><pre className="runs-code">{JSON.stringify(detail.config, null, 2)}</pre></details> : <p className="runs-muted">解析配置未记录。</p>}</section>
        <section><h3>日志尾部</h3>{detail.logs ? <><details className="runs-technical-evidence"><summary>查看日志路径</summary><code>{detail.logs.path}</code></details><p className="runs-muted">{detail.logs.truncated ? "仅显示末尾片段" : "已读取当前日志"}</p><pre className="launcher-log runs-log">{detail.logs.text || "日志为空"}</pre></> : <p className="runs-muted">日志不可用或历史运行未记录日志。</p>}</section>
      </>}
    </section> : <>
      <section className="runs-evaluation-workbench" aria-labelledby="base-evaluation-title">
        <h2 id="base-evaluation-title">评测已登记的 Base 模型</h2>
        <p>选择本机已登记且摘要可核验的 Base 模型，运行数学基准或客观题评测。评测会创建独立 Run。</p>
        {baseEvaluationError && <div className="notice danger" role="alert">{baseEvaluationError}</div>}
        <div className="runs-evaluation-controls">
          <label>Base 模型<select aria-label="选择 Base 模型" value={baseModelId} onChange={(event) => { setBaseModelId(event.target.value); setBaseEvaluationPreflight(null); setBaseEvaluationRequest("idle"); setBaseEvaluationError(""); }}>
            <option value="">请选择已登记 Base 模型</option>{baseModels.map((model) => <option key={model.model_id} value={model.model_id}>{model.display_name ? `${model.display_name} · ${model.model_id}` : model.model_id}</option>)}
          </select></label>
          <label>评测方案<select aria-label="选择 Base 模型评测方案" value={baseEvaluationProfile} onChange={(event) => { setBaseEvaluationProfile(event.target.value); setBaseScoringStrategy("exact_match"); setBaseEvaluationPreflight(null); setBaseEvaluationRequest("idle"); setBaseEvaluationError(""); }}>
            <option value="shared-math-eval-v1">数学基准 · math-eval-v1</option><option value="choice-accuracy-v1">客观题准确率 · choice-accuracy-v1</option>
          </select></label>
          {baseEvaluationProfile === "choice-accuracy-v1" && <label>客观题评测集<select aria-label="选择客观题评测集" value={baseEvaluationDatasetId} onChange={(event) => { setBaseEvaluationDatasetId(event.target.value); setBaseEvaluationPreflight(null); setBaseEvaluationRequest("idle"); setBaseEvaluationError(""); }}>
            <option value="">请选择 evaluation-only 数据集</option>{evaluationDatasets.map((dataset) => <option key={dataset.dataset_id} value={dataset.dataset_id}>{dataset.dataset_id} · {dataset.record_count} 题</option>)}
          </select></label>}
          {baseEvaluationProfile === "choice-accuracy-v1" && <label>评分策略<select aria-label="选择 Base 客观题评分策略" value={baseScoringStrategy} onChange={(event) => { setBaseScoringStrategy(event.target.value); setBaseEvaluationPreflight(null); setBaseEvaluationRequest("idle"); setBaseEvaluationError(""); }}>
            <option value="exact_match">精确匹配（默认）</option><option value="bailian_llm_judge">阿里百炼大模型评判</option>
          </select></label>}
          <button className="button secondary" type="button" disabled={baseCatalogLoading} onClick={() => void refreshBaseModelCatalog()}>{baseCatalogLoading ? "正在刷新…" : "刷新模型和评测集"}</button>
        </div>
        {baseCatalogLoading && <p className="runs-muted" role="status">正在核验本地 Base 模型目录…</p>}
        {!baseCatalogLoading && !baseModels.length && <p className="runs-muted">目录里没有可用的已登记 Base 模型。先将本地 Base 模型登记并核验摘要，再从这里评测。</p>}
        <TaskStatusCard
          input={{ request: baseEvaluationRequest, preflight: baseEvaluationPreflight ? baseEvaluationPreflight.ok ? "ready" : "blocked" : "none" }}
          actionLabel={baseEvaluationRequest === "idle" && !baseEvaluationPreflight ? "生成评测预检" : baseEvaluationRequest === "idle" && baseEvaluationPreflight?.ok ? "确认并启动独立评测 Run" : undefined}
          actionDisabled={baseEvaluationBusy || !baseModelId || (baseEvaluationProfile === "choice-accuracy-v1" && !baseEvaluationDatasetId)}
          onAction={(action) => {
            if (action === "preflight") void preflightBaseEvaluation();
            else if (action === "submit") void startBaseEvaluation();
            else if (action === "inspect") void inspectBaseEvaluationStart();
            else if (action === "edit") setBaseEvaluationPreflight(null);
          }} />
        {baseEvaluationPreflight && <div className={`runs-evaluation-confirmation ${baseEvaluationPreflight.ok ? "ready" : "blocked"}`} role={baseEvaluationPreflight.ok ? "status" : "alert"}>
          <strong>{baseEvaluationPreflight.ok ? "预检通过" : "当前不能启动评测"}</strong>{baseEvaluationPreflight.blockers.map((blocker) => <p key={blocker}>{blocker}</p>)}
          {baseEvaluationPreflight.ok && <>
            {baseEvaluationPreflight.dataset && <p>评测集：<code>{baseEvaluationPreflight.dataset.dataset_id}</code> · 样本数：{baseEvaluationPreflight.dataset.record_count}</p>}
            {baseEvaluationPreflight.external_judge && <div className="notice warning" role="status">
              <p>{baseEvaluationPreflight.external_judge.external_data_notice}</p>
              <p>百炼 Profile：<code>{baseEvaluationPreflight.external_judge.provider_profile_id}</code> · 评判模型：{baseEvaluationPreflight.external_judge.provider_model} · 预计 API 请求：{baseEvaluationPreflight.external_judge.expected_request_count} 次。</p>
              <p>{baseEvaluationPreflight.external_judge.billing_notice}</p>
              <p>点击“确认并启动”即确认将上述评测内容发送给阿里云百炼。</p>
            </div>}
            <details><summary>查看评测技术身份</summary><p>模型：<code>{baseEvaluationPreflight.model?.model_id}</code> · 协议：{baseEvaluationPreflight.protocol_version} · 输出：<code>{baseEvaluationPreflight.output}</code></p></details>
          </>}
        </div>}
      </section>
      <section className="runs-panel" aria-label="运行筛选">
        <div className="runs-filters"><label>阶段<select aria-label="按阶段筛选" value={filters.stage} onChange={(e) => changeFilter("stage", e.target.value)}><option value="">全部阶段</option>{[["cpt", "CPT"], ["sft", "SFT"], ["dpo", "DPO"], ["simpo", "SimPO"], ["grpo", "GRPO"], ["sft_eval", "SFT 评测"], ["evaluate", "评测"]].map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
          <label>状态<select aria-label="按状态筛选" value={filters.status} onChange={(e) => changeFilter("status", e.target.value)}><option value="">全部状态</option>{Object.entries(statusLabel).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select></label>
          <label className="runs-search">搜索<input aria-label="搜索 run_id 或实验标签" value={filters.q} onChange={(e) => changeFilter("q", e.target.value)} placeholder="run_id、experiment_id、标签" /></label>
          <button className="button secondary" onClick={refresh} disabled={loading}>{loading ? "读取中…" : "刷新"}</button>
        </div>
        {index?.partial_scan && <div className="notice warning" role="status">运行记录较多，扫描达到上限；当前结果可能不完整。</div>}
        {error && <div className="notice danger" role="alert">Runs 读取失败：{error}</div>}
        <div className="runs-list-heading"><h2>运行记录</h2><span>{index?.total ?? 0} 条</span></div>
        {loading && !index && <p role="status">正在扫描本地运行记录…</p>}
        {!loading && !error && index?.runs.length === 0 && <div className="runs-empty"><strong>没有匹配的运行记录</strong><p>尝试清除阶段/状态筛选，或确认项目里已有 run.json / evaluation_run.json。</p></div>}
        {index?.runs.length ? <div className="runs-table-wrap"><table className="runs-table"><thead><tr><th scope="col">运行 / 标签</th><th scope="col">类型</th><th scope="col">阶段</th><th scope="col">状态</th><th scope="col">开始时间</th><th scope="col">来源</th><th scope="col"><span className="sr-only">操作</span></th></tr></thead><tbody>{index.runs.map((run) => <tr key={run.run_id}><td><button className="runs-link" onClick={() => openRun(run.run_id)}>{displayRunName(run)}</button><small title={run.run_id}>Run ID：{run.run_id.length > 38 ? `${run.run_id.slice(0, 35)}…` : run.run_id}</small></td><td>{run.run_kind === "evaluation" ? "评测" : "训练"}</td><td>{stageLabel[run.stage || ""] || run.stage || "未记录"}</td><td><span className={`badge ${run.status === "completed" ? "success" : ["failed", "oom"].includes(run.status) ? "danger" : "neutral"}`}>{statusLabel[run.status] || run.status}</span></td><td>{formatDate(run.started_at)}</td><td><details><summary>查看路径</summary><code>{run.source_ref}</code></details></td><td><button className="button secondary runs-open" onClick={() => openRun(run.run_id)} aria-label={`查看运行 ${run.run_id}`}>详情</button></td></tr>)}</tbody></table></div> : null}
        {index && index.total > 100 && <div className="runs-pagination"><span>显示 {filters.offset + 1}–{Math.min(filters.offset + index.runs.length, index.total)} / {index.total}</span><div><button className="button secondary" disabled={filters.offset === 0 || loading} onClick={() => { const next = { ...filters, offset: Math.max(0, filters.offset - 100) }; setFilters(next); writeQuery(next); }}>上一页</button><button className="button secondary" disabled={filters.offset + 100 >= index.total || loading} onClick={() => { const next = { ...filters, offset: filters.offset + 100 }; setFilters(next); writeQuery(next); }}>下一页</button></div></div>}
      </section>
    </>}
  </main>;
}
