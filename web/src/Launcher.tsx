import { useCallback, useEffect, useRef, useState } from "react";
import { ModelDisplayNameEditor } from "./ModelDisplayNameEditor";
import { TaskStatusCard } from "./TaskStatusCard";
import type { TaskRequestState } from "./taskPresentation";

type Capability = { enabled: boolean; token?: string; available?: boolean; recipe_ref?: string; reason?: string;
  parent_model_id?: string; default_dataset_id?: string; datasets?: { dataset_id: string; count: number; quality_status: string; license_status: string; source_name: string }[] };
type Preflight = {
  ok: boolean; blockers: string[]; warnings: string[]; confirmation_token?: string; expires_in_seconds?: number;
  run_id?: string; recipe?: string; recipe_sha256?: string; resolved_config?: Record<string, unknown>;
  training_parameters?: { defaults: Record<string, number>; overrides: Record<string, number>;
    effective: Record<string, number>; config_sha256?: string | null };
  training_memory_mode?: { mode_id: string; label_zh: string; settings: Record<string, unknown>; compatibility?: Record<string, unknown> };
  model?: Record<string, unknown>; dataset?: Record<string, unknown>; output?: string;
  gpu?: Record<string, unknown>; fingerprint?: string;
};
type Run = {
  run_id: string; status: string; started_at?: string; completed_at?: string; updated_at?: string; elapsed_seconds?: number;
  run_kind?: string; stage?: string; profile_id?: string; dataset_id?: string; parent_model_id?: string; model_id?: string;
  parameter_overrides?: Record<string, number>; training_parameters?: { defaults?: Record<string, number>; overrides?: Record<string, number>; effective?: Record<string, number>; config_sha256?: string };
  training_memory_mode?: { mode_id: string; label_zh: string; settings?: Record<string, unknown>; compatibility?: Record<string, unknown> };
  effective_config_sha256?: string;
  progress?: { step: number; total: number | null } | null; progress_status?: string; detail?: string;
  log_tail?: string; log_truncated?: boolean; output?: string; outputs?: string[]; lineage_path?: string;
  model_ids?: string[];
  log_sources?: string[]; training_metrics?: Record<string, string | number>;
  validation?: { mode?: "none" | "registered" | "random_split"; status?: string; record_count?: number; validation_count?: number };
  validation_metrics?: Record<string, number>;
  checkpoint_selection?: { policy?: "final" | "best"; metric?: string | null; metric_value?: number | null;
    selected_step?: number; selected?: string; final?: string; latest?: string; best?: string | null; digests?: Record<string, string> };
  model_artifacts?: { model_id: string; kind?: string; stage?: string; produced_by_run_id?: string; display_name?: string | null }[];
  evaluation_status?: string;
  evaluation_run_id?: string; evaluation_detail?: string; evaluation_log_tail?: string;
  evaluation_outputs?: string[];
  evaluation_lineage_path?: string; evaluation_metrics?: { protocol_version?: string; status?: string; model_id?: string; run_id?: string;
    evaluation_suite?: { dataset_id?: string; dataset_sha256?: string; sample_count?: number; protocol_version?: string };
    comparability?: { key?: string; note?: string };
    metrics?: { name: string; metric: string; value: number | null; denominator: number | null;
      correct?: number; comparability_key?: string; status?: string; parse_failures?: number; denominator_unit?: string; reason?: string }[] } | null;
};
type EvaluationPreflight = {
  ok: boolean; blockers: string[]; warnings: string[]; confirmation_token?: string; expires_in_seconds?: number;
  training_run_id?: string; evaluation_run_id?: string; protocol_version?: string;
  benchmarks?: { name: string; samples: number; sampling?: string }[]; model?: Record<string, unknown>; output?: string;
};
type StageProfile = { profile_id: string; stage: string; algorithm: string; approved_recipe: string;
  parent_model_stages: string[]; parent_model_rule: string; dataset_stage: string; dataset_roles: string[];
  evaluation_profiles: { profile_id: string; protocol: string; metrics: string[] }[];
  execution_supported: boolean; execution_blocker?: string | null; training_validation_supported?: boolean;
  validation_dataset_roles?: string[]; validation_min_samples?: number;
  checkpoint_metric?: string; checkpoint_greater_is_better?: boolean;
  training_parameters: { key: string; label_zh: string; kind: "int" | "float"; default: number;
    minimum: number; maximum: number; choices: number[] }[];
  training_memory_modes?: { mode_id: string; label_zh: string; description_zh: string; default: boolean; settings: Record<string, unknown> }[] };
type WorkbenchOptions = { datasets: { dataset_id: string; display_name?: string | null; stage: string; role?: string | null; record_count: number;
  eligible?: boolean; block_reason?: string | null }[];
    models: { model_id: string; stage: string; kind: string; produced_by_run_id?: string | null; source_run_id?: string | null;
    digest?: { status: string; value?: string | null; reason?: string | null }; path_available?: boolean;
    unavailable_reason?: string | null; registry_count?: number }[] };
type RunTab = "overview" | "logs" | "evaluation" | "artifacts";
type StageDraft = { profileId?: string; datasetIds?: string[]; parentModelId?: string; parameterOverrides?: Record<string, number>; memoryModeId?: string;
  validationMode?: "none" | "registered" | "random_split"; validationDatasetId?: string; validationRatio?: number; validationSeed?: number;
  checkpointPolicy?: "final" | "best" };
type EvaluationDraft = { profileId?: string; datasetId?: string };
type ReadinessCheck = { check_id: string; status: "pass" | "warning" | "fail"; detail: string; repair_hint?: string };
const STAGE_DRAFT_PREFIX = "onegpu.stage-workbench.draft.v1:";
const EVALUATION_DRAFT_PREFIX = "onegpu.stage-workbench.evaluation-draft.v1:";
const RUN_TAB_VALUES: RunTab[] = ["overview", "logs", "evaluation", "artifacts"];
function readStageDraft(stage: string): StageDraft | null {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(`${STAGE_DRAFT_PREFIX}${stage}`) || "null");
    if (!value || typeof value !== "object") return null;
    return {
      profileId: typeof value.profileId === "string" ? value.profileId : undefined,
      datasetIds: Array.isArray(value.datasetIds) ? value.datasetIds.filter((item: unknown) => typeof item === "string") : undefined,
      parentModelId: typeof value.parentModelId === "string" ? value.parentModelId : undefined,
      parameterOverrides: value.parameterOverrides && typeof value.parameterOverrides === "object" ? value.parameterOverrides : undefined,
      memoryModeId: typeof value.memoryModeId === "string" ? value.memoryModeId : undefined,
      validationMode: ["none", "registered", "random_split"].includes(value.validationMode) ? value.validationMode : undefined,
      validationDatasetId: typeof value.validationDatasetId === "string" ? value.validationDatasetId : undefined,
      validationRatio: typeof value.validationRatio === "number" ? value.validationRatio : undefined,
      validationSeed: typeof value.validationSeed === "number" ? value.validationSeed : undefined,
      checkpointPolicy: value.checkpointPolicy === "best" ? "best" : "final",
    };
  } catch { return null; }
}
function readEvaluationDraft(runId: string): EvaluationDraft | null {
  try {
    const value = JSON.parse(window.sessionStorage.getItem(`${EVALUATION_DRAFT_PREFIX}${runId}`) || "null");
    if (!value || typeof value !== "object") return null;
    return { profileId: typeof value.profileId === "string" ? value.profileId : undefined,
      datasetId: typeof value.datasetId === "string" ? value.datasetId : undefined };
  } catch { return null; }
}
const messages: Record<string, string> = {
  starting: "正在启动", running: "训练中", completed: "训练完成", failed: "训练失败", oom: "显存不足（OOM）", interrupted: "已中断",
};
const pretty = (value: unknown) => JSON.stringify(value, null, 2);
const runDuration = (seconds: number | undefined) => seconds == null ? "未记录" : `${Math.floor(seconds / 60)} 分 ${Math.floor(seconds % 60)} 秒`;
const runUpdated = (value: string | undefined) => value && !Number.isNaN(Date.parse(value)) ? new Date(value).toLocaleString("zh-CN") : "历史运行未记录";
const readinessCheckLabels: Record<string, string> = {
  "dataset:registered_cpt": "CPT 数据集与登记证据",
  "parent_model:local_base": "固定 Base 模型与本地权重",
  "runtime:ms_swift": "训练运行环境 · ms-swift",
  "hardware:gpu": "本机 GPU",
  "launcher:execution_adapter": "Launcher 执行能力",
};
const readinessStatusLabels: Record<ReadinessCheck["status"], string> = { pass: "通过", warning: "提醒", fail: "阻断" };

function WorkflowRoadmap({ run }: { run: Run | null }) {
  const trainingActive = Boolean(run && ["starting", "running"].includes(run.status));
  const evaluationActive = Boolean(run && ["starting", "running"].includes(run.evaluation_status || ""));
  const trainingDone = run?.status === "completed";
  const evaluationDone = Boolean(run && ["completed", "partial"].includes(run.evaluation_status || ""));
  const failed = Boolean(run && ["failed", "oom", "interrupted"].includes(run.status));
  const steps = [
    { title: "配置阶段与数据", status: run ? "done" : "current", note: run ? "本次配置已绑定到 Run" : "选择阶段、配方、父模型和训练数据" },
    { title: "预检并启动训练", status: run ? "done" : "upcoming", note: run ? "已创建运行，不会因刷新重复启动" : "检查兼容性后显式确认启动" },
    { title: "查看训练过程", status: trainingActive ? "current" : trainingDone ? "done" : failed ? "blocked" : "upcoming", note: trainingActive ? "正在运行，可查看实时日志" : trainingDone ? "训练已结束" : failed ? "训练未成功，先查看失败原因" : "训练开始后可查看进度和日志" },
    { title: "阶段评测", status: evaluationActive ? "current" : evaluationDone ? "done" : trainingDone ? "available" : "upcoming", note: evaluationActive ? "评测进行中，刷新可继续查看" : evaluationDone ? "评测已结束，可查看指标" : trainingDone ? "训练完成后可选择协议并启动" : "训练成功后开放" },
    { title: "查看结果与血缘", status: evaluationDone ? "done" : trainingDone ? "available" : "upcoming", note: evaluationDone ? "查看评测结果、模型产物和血缘" : trainingDone ? "可先查看模型产物与训练血缘" : "训练完成后开放" },
  ];
  return <nav className="launcher-workflow-roadmap" aria-label="训练与评测流程">
    <h2>完整流程</h2>
    <ol>{steps.map((step, index) => <li key={step.title} className={`workflow-step workflow-${step.status}`}>
      <span className="workflow-step-number">{step.status === "done" ? "✓" : index + 1}</span>
      <span className="workflow-step-copy"><strong>{step.title}</strong><small>{step.note}</small></span>
    </li>)}</ol>
  </nav>;
}

function RunConfiguration({ run, options }: { run: Run; options: WorkbenchOptions }) {
  const dataset = options.datasets.find((item) => item.dataset_id === run.dataset_id);
  const parent = options.models.find((item) => item.model_id === run.parent_model_id);
  const effective = run.training_parameters?.effective;
  const overrides = run.training_parameters?.overrides || run.parameter_overrides || {};
  const configHash = run.training_parameters?.config_sha256 || run.effective_config_sha256;
  return <section className="launcher-run-configuration" aria-label="本次运行配置">
    <div><h2>本次运行配置</h2><span className="badge neutral">只读快照</span></div>
    <p>这是启动时绑定到此 Run 的配置；之后修改训练草稿不会改变这次运行。</p>
    <dl className="launcher-facts">
      <div><dt>阶段 / 配方</dt><dd>{(run.stage || "sft").toUpperCase()} · {run.profile_id || "sft-mini-1024-v1"}</dd></div>
      <div><dt>训练数据</dt><dd>{dataset?.display_name || run.dataset_id || "未记录"}{run.dataset_id && dataset?.display_name && <small><code>{run.dataset_id}</code></small>}</dd></div>
      <div><dt>父模型</dt><dd>{parent ? `${parent.stage} · ${parent.kind} · ${parent.model_id}` : run.parent_model_id || run.model_id || "由批准配方固定"}</dd></div>
      <div><dt>参数 / 显存模式</dt><dd>{Object.keys(overrides).length ? `自定义 ${Object.keys(overrides).length} 项` : "使用批准配方默认值"}{run.training_memory_mode?.label_zh ? ` · ${run.training_memory_mode.label_zh}` : ""}</dd></div>
      <div><dt>检查点策略</dt><dd>{run.checkpoint_selection?.policy === "best" ? `验证最佳${run.checkpoint_selection.metric ? ` · ${run.checkpoint_selection.metric}` : ""}${run.checkpoint_selection.metric_value !== undefined ? `=${run.checkpoint_selection.metric_value}` : ""}` : run.checkpoint_selection?.policy === "final" ? "最后保存" : "历史运行未记录选择策略"}{run.checkpoint_selection?.selected_step !== undefined ? ` · step ${run.checkpoint_selection.selected_step}` : ""}</dd></div>
      {run.training_memory_mode?.mode_id === "qlora-4bit-nf4" && <div><dt>QLoRA 依赖</dt><dd>{run.training_memory_mode.compatibility?.ok === true ? "预检通过（不代表训练必然成功）" : "未通过或未记录"}</dd></div>}
    </dl>
    {effective && <details><summary>查看本次生效参数</summary><pre>{pretty({ ...effective,
      ...(run.training_memory_mode?.settings || {}) })}</pre></details>}
    {configHash && <details><summary>查看配置摘要</summary><code>{configHash}</code></details>}
  </section>;
}

async function call<T>(url: string, token?: string, body?: unknown): Promise<T> {
  const response = await fetch(url, { method: body === undefined ? "GET" : "POST", headers: {
    ...(body === undefined ? {} : { "Content-Type": "application/json" }),
    ...(token ? { "X-Local-Launcher-Token": token } : {}),
  }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const value = await response.json();
  if (!response.ok) throw new ApiRequestError(response.status, value.detail || `本地服务返回 ${response.status}`);
  return value as T;
}
class ApiRequestError extends Error {
  constructor(readonly status: number, message: string) { super(message); }
}

export function Launcher({ onBack, onOpenRun, onComposeDatasets, onPrepareGrpoData }: { onBack: () => void; onOpenRun: (runId: string) => void; onComposeDatasets?: (stage: string) => void; onPrepareGrpoData?: (context: { profileId: string; parentModelId: string }) => void }) {
  const initialParams = new URLSearchParams(window.location.search);
  const requestedRunTab = initialParams.get("launcher_tab");
  const runTabWasExplicit = RUN_TAB_VALUES.includes(requestedRunTab as RunTab);
  const [cap, setCap] = useState<Capability | null>(null);
  const [preflight, setPreflight] = useState<Preflight | null>(null);
  const [acknowledgeUnknown, setAcknowledgeUnknown] = useState(false);
  const [evaluationPreflight, setEvaluationPreflight] = useState<EvaluationPreflight | null>(null);
  const [stageEvaluationPreflight, setStageEvaluationPreflight] = useState<Record<string, any> | null>(null);
  const [stageEvaluationRequest, setStageEvaluationRequest] = useState<TaskRequestState>("idle");
  const [stageEvaluationStartedRun, setStageEvaluationStartedRun] = useState<{ run_id: string; status: string } | null>(null);
  const [selectedEvaluationProfileId, setSelectedEvaluationProfileId] = useState("");
  const [selectedEvaluationDatasetId, setSelectedEvaluationDatasetId] = useState("");
  const [stageProfiles, setStageProfiles] = useState<StageProfile[]>([]);
  const [workbenchOptions, setWorkbenchOptions] = useState<WorkbenchOptions>({ datasets: [], models: [] });
  const [workbenchOptionsLoaded, setWorkbenchOptionsLoaded] = useState(false);
  const [selectedStage, setSelectedStage] = useState("sft");
  const [selectedProfileId, setSelectedProfileId] = useState("");
  const [selectedDatasetIds, setSelectedDatasetIds] = useState<string[]>([]);
  const datasetHandoffApplied = useRef(false);
  const modelHandoffApplied = useRef(false);
  const [selectedParentModelId, setSelectedParentModelId] = useState("");
  const [parameterOverrides, setParameterOverrides] = useState<Record<string, number>>({});
  const [memoryModeId, setMemoryModeId] = useState("lora");
  const [validationMode, setValidationMode] = useState<"none" | "registered" | "random_split">("none");
  const [validationDatasetId, setValidationDatasetId] = useState("");
  const [validationRatio, setValidationRatio] = useState(0.1);
  const [validationSeed, setValidationSeed] = useState(42);
  const [checkpointPolicy, setCheckpointPolicy] = useState<"final" | "best">("final");
  const [stagePreflight, setStagePreflight] = useState<Record<string, unknown> | null>(null);
  const [stageConfirmation, setStageConfirmation] = useState<Record<string, any> | null>(null);
  const [stageLicenseAcknowledged, setStageLicenseAcknowledged] = useState(false);
  const [runTab, setRunTabState] = useState<RunTab>(runTabWasExplicit ? requestedRunTab as RunTab : "overview");
  const [run, setRun] = useState<Run | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [handoffMessage, setHandoffMessage] = useState("");
  const [followLogs, setFollowLogs] = useState(true);
  const [newLogs, setNewLogs] = useState(false);
  const logElement = useRef<HTMLPreElement | null>(null);
  const lastLog = useRef("");
  const evaluationDraftHydratedRun = useRef("");
  const [draftHydrated, setDraftHydrated] = useState(false);
  const [focusedRunId, setFocusedRunId] = useState(initialParams.get("launcher_run"));
  const handoffSourceRunId = initialParams.get("source_run_id");

  function setRunTab(tab: RunTab) {
    setRunTabState(tab);
    const params = new URLSearchParams(window.location.search);
    params.set("launcher_tab", tab);
    if (run) { params.set("launcher_run", run.run_id); setFocusedRunId(run.run_id); }
    window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
  }

  function focusRun(runId: string, tab: RunTab, stage?: string) {
    const params = new URLSearchParams(window.location.search);
    params.set("page", "launcher"); params.set("launcher_run", runId); params.set("launcher_tab", tab);
    if (stage) params.set("stage", stage);
    window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
    setFocusedRunId(runId);
    setRunTabState(tab);
  }

  const refreshRun = useCallback(async (signal?: AbortSignal) => {
    if (focusedRunId) {
      const value = await call<Run>(`/api/launcher/runs/${encodeURIComponent(focusedRunId)}`, undefined, undefined);
      if (!signal?.aborted) setRun(value);
      return;
    }
    const value = await call<{ run: Run | null }>("/api/launcher/run", undefined, undefined);
    if (!signal?.aborted) setRun(value.run);
  }, [focusedRunId]);

  useEffect(() => {
    const controller = new AbortController();
    const requestedStage = new URLSearchParams(window.location.search).get("stage");
    if (["cpt", "sft", "dpo", "simpo", "grpo"].includes(requestedStage || "")) setSelectedStage(requestedStage!);
    call<{ profiles: StageProfile[] }>("/api/training/stage-profiles").then((value) => {
      if (!controller.signal.aborted) {
        setStageProfiles(value.profiles);
        const params = new URLSearchParams(window.location.search);
        const requested = params.get("stage") || "sft";
        const requestedProfile = params.get("profile_id");
        const profile = value.profiles.find((item) => item.stage === requested && item.profile_id === requestedProfile)
          || value.profiles.find((item) => item.stage === requested);
        setSelectedProfileId(profile?.profile_id || value.profiles.find((item) => item.stage === "sft")?.profile_id || "");
      }
    }).catch((e) => { if (!controller.signal.aborted) setError(e.message); });
    call<WorkbenchOptions>("/api/training/workbench-options").then((value) => {
      if (!controller.signal.aborted) { setWorkbenchOptions(value); setWorkbenchOptionsLoaded(true); }
    }).catch((e) => { if (!controller.signal.aborted) { setError(e.message); setWorkbenchOptionsLoaded(true); } });
    call<Capability>("/api/launcher/capabilities").then((value) => {
      setCap(value);
    }).catch((e) => setError(e.message));
    refreshRun(controller.signal).catch((e) => { if (!controller.signal.aborted) setError(e.message); });
    return () => controller.abort();
  }, [refreshRun]);

  useEffect(() => {
    if (!stageProfiles.length || !workbenchOptionsLoaded || draftHydrated) return;
    const params = new URLSearchParams(window.location.search);
    const draft = readStageDraft(selectedStage);
    if (!draft) {
      const profile = stageProfiles.find((item) => item.stage === selectedStage && item.profile_id === selectedProfileId);
      setMemoryModeId(profile?.training_memory_modes?.find((mode) => mode.default)?.mode_id || "lora");
      setDraftHydrated(true); return;
    }
    const requestedProfile = params.get("profile_id");
    const draftProfile = stageProfiles.find((item) => item.stage === selectedStage && item.profile_id === draft.profileId);
    if (!requestedProfile && draftProfile) setSelectedProfileId(draftProfile.profile_id);
    if (!params.has("dataset_id") && draft.datasetIds) setSelectedDatasetIds(draft.datasetIds);
    if (!params.has("parent_model_id") && draft.parentModelId) setSelectedParentModelId(draft.parentModelId);
    if (draft.parameterOverrides) setParameterOverrides(draft.parameterOverrides);
    const currentProfile = draftProfile || stageProfiles.find((item) => item.stage === selectedStage && item.profile_id === selectedProfileId);
    const draftMode = currentProfile?.training_memory_modes?.find((mode) => mode.mode_id === draft.memoryModeId);
    setMemoryModeId(draftMode?.mode_id || currentProfile?.training_memory_modes?.find((mode) => mode.default)?.mode_id || "lora");
    setValidationMode(draft.validationMode || "none");
    setValidationDatasetId(draft.validationDatasetId || "");
    setValidationRatio(draft.validationRatio ?? 0.1);
    setValidationSeed(draft.validationSeed ?? 42);
    setCheckpointPolicy(draft.checkpointPolicy || "final");
    setDraftHydrated(true);
  }, [stageProfiles, workbenchOptionsLoaded, selectedStage, draftHydrated]);

  useEffect(() => {
    if (!draftHydrated) return;
    try {
      window.sessionStorage.setItem(`${STAGE_DRAFT_PREFIX}${selectedStage}`, JSON.stringify({
        profileId: selectedProfileId, datasetIds: selectedDatasetIds,
        parentModelId: selectedParentModelId, parameterOverrides, memoryModeId,
        validationMode, validationDatasetId, validationRatio, validationSeed,
        checkpointPolicy,
      } satisfies StageDraft));
    } catch { /* Storage can be disabled; the in-memory workflow still works. */ }
  }, [draftHydrated, selectedStage, selectedProfileId, selectedDatasetIds, selectedParentModelId, parameterOverrides, memoryModeId,
    validationMode, validationDatasetId, validationRatio, validationSeed, checkpointPolicy]);

  const selectedProfile = stageProfiles.find((item) => item.profile_id === selectedProfileId && item.stage === selectedStage);
  const stageDatasets = workbenchOptions.datasets.filter((item) => item.stage === selectedProfile?.dataset_stage
    && (!selectedProfile.dataset_roles.length || !item.role || selectedProfile.dataset_roles.includes(item.role)));
  const eligibleStageDatasets = stageDatasets.filter((item) => item.eligible !== false);
  const eligibleSelectedDatasetIds = selectedDatasetIds.filter((id) => eligibleStageDatasets.some((item) => item.dataset_id === id));
  const validationStageDatasets = workbenchOptions.datasets.filter((item) => item.stage === selectedProfile?.dataset_stage
    && (selectedProfile?.validation_dataset_roles || ["validation", "diagnostic_dev", "cpt_holdout"]).includes(item.role || "")
    && item.eligible !== false);
  const evaluationDatasetRole = selectedEvaluationProfileId === "preference-heldout-v1" ? "preference_eval"
    : selectedEvaluationProfileId === "cpt-math-v1" ? "cpt_holdout"
      : selectedEvaluationProfileId === "choice-accuracy-v1" ? "evaluation_only" : null;
  const stageEvaluationDatasets = evaluationDatasetRole
    ? workbenchOptions.datasets.filter((item) => item.stage === (evaluationDatasetRole === "preference_eval" ? "preference"
      : evaluationDatasetRole === "cpt_holdout" ? "cpt" : "eval")
      && item.role === evaluationDatasetRole)
    : [];
  const stageParents = workbenchOptions.models.filter((item) => selectedProfile?.parent_model_stages.includes(item.stage)
    && (selectedProfile.parent_model_rule !== "registered_sft_merged" || item.kind === "merged"));

  useEffect(() => {
    if (selectedProfile?.parent_model_rule === "approved_mini_cpt" && cap?.parent_model_id) {
      setSelectedParentModelId(cap.parent_model_id);
    }
  }, [selectedProfile?.profile_id, selectedProfile?.parent_model_rule, cap?.parent_model_id]);

  useEffect(() => {
    if (datasetHandoffApplied.current || !selectedProfile || workbenchOptions.datasets.length === 0) return;
    const requested = new URLSearchParams(window.location.search).get("dataset_id");
    if (!requested) return;
    datasetHandoffApplied.current = true;
    if (stageDatasets.some((item) => item.dataset_id === requested && item.eligible !== false)) setSelectedDatasetIds([requested]);
    else if (stageDatasets.some((item) => item.dataset_id === requested)) setError(`传入的数据集 ${requested} 尚未完成登记，不能用于训练；请先在“数据”中完成处理与质量登记。`);
    else setError(`传入的数据集 ${requested} 不属于当前训练阶段；请检查阶段类型后手动选择。`);
  }, [selectedProfile, stageDatasets, workbenchOptions.datasets.length]);

  useEffect(() => {
    if (!workbenchOptionsLoaded || selectedDatasetIds.length === 0) return;
    if (selectedDatasetIds.some((id) => !eligibleStageDatasets.some((item) => item.dataset_id === id))) {
      setSelectedDatasetIds([]);
      setStagePreflight(null); setPreflight(null); setStageConfirmation(null);
    }
  }, [workbenchOptionsLoaded, selectedStage, selectedProfileId, eligibleStageDatasets, selectedDatasetIds]);

  useEffect(() => {
    if (modelHandoffApplied.current || !selectedProfile || !selectedProfileId || !workbenchOptionsLoaded) return;
    const params = new URLSearchParams(window.location.search);
    const requestedModelId = params.get("parent_model_id");
    if (!requestedModelId) return;
    if (selectedProfile.parent_model_rule === "approved_mini_cpt" && !cap?.parent_model_id) return;
    modelHandoffApplied.current = true;
    const sourceRunId = params.get("source_run_id") || "来源 Run 未记录";
    const compatible = selectedProfile.parent_model_rule === "approved_mini_cpt"
      ? requestedModelId === cap?.parent_model_id
      : stageParents.some((item) => item.model_id === requestedModelId && item.path_available !== false
        && item.digest?.status === "available");
    if (!compatible) {
      setHandoffMessage(`${sourceRunId} 传入的模型 ${requestedModelId} 不符合 ${selectedProfile.profile_id} 的父模型要求；未自动选用，请手动选择兼容模型。`);
      return;
    }
    setSelectedParentModelId(requestedModelId);
    setHandoffMessage(`已从 ${sourceRunId} 预选父模型 ${requestedModelId}。训练数据仍需单独选择，完成后请重新预检并显式启动。`);
  }, [selectedProfile, selectedProfileId, stageParents, cap?.parent_model_id, workbenchOptionsLoaded]);

  function chooseStage(stage: string) {
    const draft = readStageDraft(stage);
    const profile = stageProfiles.find((item) => item.stage === stage && item.profile_id === draft?.profileId)
      || stageProfiles.find((item) => item.stage === stage);
    setSelectedStage(stage);
    setSelectedProfileId(profile?.profile_id || "");
    setSelectedDatasetIds(draft?.datasetIds || []); setSelectedParentModelId(draft?.parentModelId || "");
    setParameterOverrides(draft?.parameterOverrides || {});
    setMemoryModeId(profile?.training_memory_modes?.find((mode) => mode.mode_id === draft?.memoryModeId)?.mode_id
      || profile?.training_memory_modes?.find((mode) => mode.default)?.mode_id || "lora");
    setValidationMode(draft?.validationMode || "none");
    setValidationDatasetId(draft?.validationDatasetId || "");
    setValidationRatio(draft?.validationRatio ?? 0.1);
    setValidationSeed(draft?.validationSeed ?? 42);
    setStagePreflight(null); setPreflight(null); setStageConfirmation(null);
    setAcknowledgeUnknown(false);
    const params = new URLSearchParams(window.location.search);
    params.set("page", "launcher"); params.set("stage", stage);
    params.delete("profile_id"); params.delete("parent_model_id"); params.delete("source_run_id");
    modelHandoffApplied.current = true; setHandoffMessage("");
    window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`);
  }

  async function checkStage() {
    const fixedBase = selectedProfile?.parent_model_rule === "external_base";
    if (!selectedProfile || !eligibleSelectedDatasetIds.length || eligibleSelectedDatasetIds.length !== selectedDatasetIds.length || (!fixedBase && !selectedParentModelId)) return;
    setBusy(true); setError(""); setStagePreflight(null); setPreflight(null); setAcknowledgeUnknown(false);
    setStageConfirmation(null); setStageLicenseAcknowledged(false);
    try {
      const validation = validationMode === "registered"
        ? { mode: "registered", dataset_id: validationDatasetId }
        : validationMode === "random_split"
          ? { mode: "random_split", ratio: validationRatio, seed: validationSeed }
          : { mode: "none" };
      if (selectedProfile.stage !== "sft" || selectedProfile.parent_model_rule === "registered_sft_parent") {
        if (eligibleSelectedDatasetIds.length !== 1) {
          setStagePreflight({ status: "blocked", blockers: ["请先将所选多个数据集组合并登记为一个 dataset ID，再执行预检。"], warnings: [] });
          return;
        }
        if (!cap?.token || !cap.enabled) throw new Error(cap?.reason || "本地训练 Launcher 未启用");
        const payload = {
          stage: selectedProfile.stage, profile_id: selectedProfile.profile_id,
          dataset_ids: eligibleSelectedDatasetIds,
          ...(selectedParentModelId ? { parent_model_id: selectedParentModelId } : {}),
          parameter_overrides: parameterOverrides,
          memory_mode_id: memoryModeId,
          validation,
          checkpoint_policy: checkpointPolicy,
        };
        const result = await call<Record<string, any>>("/api/training/stage/preflight", cap.token, payload);
        setStagePreflight(result);
        if (result.status === "ready_for_confirmation" && !(result.blockers || []).length) {
          const confirmation = await call<Record<string, any>>("/api/training/stage/confirm", cap.token, payload);
          setStageConfirmation(confirmation);
        }
        return;
      }
      if (validationMode !== "none") {
        if (!cap?.enabled || !cap.available || !cap.token) throw new Error(cap?.reason || "Mini SFT 启动器或固定配方当前不可用。");
        const launchCheck = await call<Preflight>("/api/launcher/preflight", cap.token, {
          dataset_id: eligibleSelectedDatasetIds[0], parameter_overrides: parameterOverrides,
          memory_mode_id: memoryModeId, validation, checkpoint_policy: checkpointPolicy,
        });
        setPreflight(launchCheck);
        setStagePreflight({ status: launchCheck.ok ? "ready_for_confirmation" : "blocked",
          blockers: launchCheck.blockers || [], warnings: launchCheck.warnings || [],
          training_parameters: launchCheck.training_parameters, validation: (launchCheck as any).validation,
          checks: launchCheck.ok ? [{ check_id: "training_validation", status: "pass", detail: "SFT 验证来源、成员身份与配置摘要已绑定预检令牌" }] : [] });
        return;
      }
      const result = await call<Record<string, any>>("/api/training/preflight", undefined, {
        stage: selectedProfile.stage, profile_id: selectedProfile.profile_id,
        dataset_ids: eligibleSelectedDatasetIds, parent_model_id: selectedParentModelId,
        parameter_overrides: parameterOverrides, memory_mode_id: memoryModeId, validation,
        checkpoint_policy: checkpointPolicy,
      });
      if (selectedProfile.execution_supported && selectedProfile.stage === "sft"
        && selectedProfile.parent_model_rule === "approved_mini_cpt") {
        if (eligibleSelectedDatasetIds.length !== 1) {
          result.blockers = [...(result.blockers || []), "Mini SFT 启动器消费一个已登记数据集 ID；请先将多个所选数据集组合成新的登记数据集。"];
          result.status = "blocked";
        } else if (!cap?.enabled || !cap.available || !cap.token) {
          result.blockers = [...(result.blockers || []), cap?.reason || "Mini SFT 启动器或固定配方当前不可用。"];
          result.status = "blocked";
        } else if (!(result.blockers || []).length) {
          const launchCheck = await call<Preflight>("/api/launcher/preflight", cap.token, {
            dataset_id: eligibleSelectedDatasetIds[0], parameter_overrides: parameterOverrides,
            memory_mode_id: memoryModeId, checkpoint_policy: checkpointPolicy,
          });
          const declaredParent = result.parent_model as Record<string, unknown> | null;
          const declaredDatasets = result.datasets as Record<string, unknown>[] | undefined;
          const mismatches: string[] = [];
          if (!launchCheck.ok) mismatches.push(...(launchCheck.blockers || []));
          if (declaredParent?.model_id !== launchCheck.model?.model_id) mismatches.push("阶段预检与启动配方的父模型 ID 不一致。");
          if (declaredParent?.digest !== launchCheck.model?.sha256) mismatches.push("阶段预检与启动配方的父模型 SHA-256 不一致。");
          if (declaredDatasets?.length !== 1 || declaredDatasets[0]?.dataset_id !== launchCheck.dataset?.dataset_id) mismatches.push("阶段预检与启动配方的数据集 ID 不一致。 ");
          if (declaredDatasets?.[0]?.content_sha256 !== launchCheck.dataset?.canonical_sha256) mismatches.push("阶段预检与启动配方的数据集 canonical SHA-256 不一致。 ");
          const stageConfigHash = (result.training_parameters as Record<string, any> | undefined)?.config_sha256;
          const launcherConfigHash = (launchCheck.training_parameters as Record<string, any> | undefined)?.training_config_sha256;
          if (!stageConfigHash || stageConfigHash !== launcherConfigHash) mismatches.push("阶段预检与实际启动配置摘要不一致。");
          if (mismatches.length) {
            result.blockers = [...(result.blockers || []), ...mismatches];
            result.status = "blocked";
          } else {
            setPreflight(launchCheck);
          }
        }
      }
      setStagePreflight(result);
    } catch (e) { setError(e instanceof Error ? e.message : "阶段预检失败"); }
    finally { setBusy(false); }
  }

  async function startRegisteredStage() {
    if (!cap?.token || !selectedProfile || !stageConfirmation?.confirmation_token || !eligibleSelectedDatasetIds[0]) return;
    const dataset = (stageConfirmation.datasets as Record<string, any>[] | undefined)?.[0];
    const licenseUnknown = dataset?.license_status !== "verified";
    if (licenseUnknown && !stageLicenseAcknowledged) return;
    const label = selectedProfile.stage.toUpperCase();
    if (!window.confirm(`即将启动 ${label} 真实本地训练。\n方案：${selectedProfile.profile_id}\n检查点策略：${checkpointPolicy === "best" ? "训练期验证 loss 最低" : "最后保存（默认）"}\n数据集：${eligibleSelectedDatasetIds[0]}\n父模型：${String(stageConfirmation.parent_model?.model_id || selectedParentModelId || "批准配方固定 Base")}\n输出：${stageConfirmation.output_location}\n训练会占用 GPU；关闭 UI 服务会中断任务。确认启动？`)) return;
    setBusy(true); setError("");
    try {
      const started = await call<{ run_id: string }>("/api/training/stage/start", cap.token, {
        stage: selectedProfile.stage, profile_id: selectedProfile.profile_id,
        dataset_ids: eligibleSelectedDatasetIds,
        ...(selectedParentModelId ? { parent_model_id: selectedParentModelId } : {}),
        parameter_overrides: parameterOverrides, memory_mode_id: memoryModeId,
        checkpoint_policy: checkpointPolicy,
        confirmation_token: stageConfirmation.confirmation_token,
        validation: validationMode === "registered" ? { mode: "registered", dataset_id: validationDatasetId }
          : validationMode === "random_split" ? { mode: "random_split", ratio: validationRatio, seed: validationSeed }
            : { mode: "none" },
        ...(licenseUnknown ? { acknowledged_dataset_id: eligibleSelectedDatasetIds[0] } : {}),
      });
      setStageConfirmation(null);
      setStagePreflight(null);
      const activeRun = await call<Run>(`/api/launcher/runs/${encodeURIComponent(started.run_id)}`);
      setRun(activeRun);
      focusRun(started.run_id, "logs", selectedProfile.stage);
    } catch (e) { setError(e instanceof Error ? e.message : "阶段训练启动失败"); }
    finally { setBusy(false); }
  }

  useEffect(() => {
    if (!run || (!["starting", "running"].includes(run.status) && !["starting", "running"].includes(run.evaluation_status || ""))) return;
    const timer = window.setInterval(() => { refreshRun().catch((e) => setError(e.message)); }, 1500);
    return () => window.clearInterval(timer);
  }, [run?.run_id, run?.status, run?.evaluation_status, refreshRun]);

  useEffect(() => {
    if (!run) return;
    const current = new URLSearchParams(window.location.search);
    let changed = false;
    const runIsActive = ["starting", "running"].includes(run.status) || ["starting", "running"].includes(run.evaluation_status || "");
    if (runIsActive && current.get("launcher_run") !== run.run_id) { current.set("launcher_run", run.run_id); setFocusedRunId(run.run_id); changed = true; }
    const currentTab = current.get("launcher_tab");
    if (!RUN_TAB_VALUES.includes(currentTab as RunTab)) {
      const defaultTab: RunTab = ["starting", "running"].includes(run.evaluation_status || "")
        ? "evaluation" : ["starting", "running"].includes(run.status) ? "logs" : "overview";
      current.set("launcher_tab", defaultTab);
      setRunTabState(defaultTab);
      changed = true;
    }
    if (changed) window.history.replaceState({}, "", `${window.location.pathname}?${current.toString()}`);
  }, [run?.run_id, run?.status, run?.evaluation_status]);

  useEffect(() => {
    const tail = run?.log_tail || "";
    if (tail !== lastLog.current) {
      if (!followLogs && lastLog.current) setNewLogs(true);
      lastLog.current = tail;
      if (followLogs && logElement.current) logElement.current.scrollTop = logElement.current.scrollHeight;
    }
    if (followLogs && runTab === "logs" && logElement.current) logElement.current.scrollTop = logElement.current.scrollHeight;
  }, [run?.log_tail, followLogs, runTab]);

  function handleLogScroll() {
    const element = logElement.current;
    if (!element) return;
    const atBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 24;
    setFollowLogs(atBottom);
    if (atBottom) setNewLogs(false);
  }

  function resumeLogFollow() {
    setFollowLogs(true); setNewLogs(false);
    if (logElement.current) logElement.current.scrollTop = logElement.current.scrollHeight;
  }

  async function start() {
    if (!cap?.token || !preflight?.ok || !preflight.confirmation_token) return;
    const unknownLicense = preflight.dataset?.license_status !== "verified";
    if (unknownLicense && !acknowledgeUnknown) return;
    const accepted = window.confirm(
      `即将按固定配置启动真实 SFT 训练。\n\nRun：${preflight.run_id}\n配置摘要：${preflight.fingerprint}\n输出：${preflight.output}\n\nRTX 4060 历史显存余量很小。关闭浏览器不会停止训练；关闭本地 UI 服务会中断训练。是否继续？`,
    );
    if (!accepted) return;
    setBusy(true); setError("");
    try {
      const started = await call<{ run_id?: string }>("/api/launcher/start", cap.token, { confirmation_token: preflight.confirmation_token,
        ...(unknownLicense ? { acknowledged_dataset_id: String(preflight.dataset?.dataset_id) } : {}) });
      setPreflight(null);
      if (started.run_id) {
        const activeRun = await call<Run>(`/api/launcher/runs/${encodeURIComponent(started.run_id)}`);
        setRun(activeRun);
        focusRun(started.run_id, "logs", selectedStage);
      } else await refreshRun();
    } catch (e) { setError(e instanceof Error ? e.message : "启动失败"); }
    finally { setBusy(false); }
  }

  const active = run && ["starting", "running"].includes(run.status);
  const evaluationActive = Boolean(run && ["starting", "running"].includes(run.evaluation_status || ""));
  const taskActive = Boolean(active || evaluationActive);
  const mergedReady = Boolean(run?.model_artifacts?.some((model) => model.kind === "merged" && model.stage === "sft"));
  const stageSelectionBlocker = !selectedProfile ? "正在读取阶段方案。"
    : !eligibleStageDatasets.length ? `当前阶段暂无可用训练数据，请先在“数据”模块导入、质检并登记 ${selectedProfile.dataset_stage} 数据。`
      : !selectedDatasetIds.length ? `请先选择一个已登记的 ${selectedProfile.dataset_stage} 训练数据集。`
        : eligibleSelectedDatasetIds.length !== selectedDatasetIds.length ? "当前选择含不可用于训练的数据集，请改选可用项。"
          : validationMode === "registered" && !validationDatasetId ? "请选择一个已登记的独立验证集，或切换为不使用验证/固定种子切分。"
            : validationMode === "random_split" && (!Number.isFinite(validationRatio) || validationRatio < 0.05 || validationRatio > 0.4
              || !Number.isInteger(validationSeed) || validationSeed < 0 || validationSeed > 2147483647)
              ? "验证比例须为 5%–40%，随机种子须为 0–2147483647 的整数。"
          : !["external_base", "approved_mini_cpt"].includes(selectedProfile.parent_model_rule) && !selectedParentModelId
            ? "请先选择上方已登记且符合阶段要求的父模型。"
            : !cap ? "正在检查本地训练启动能力。"
              : !cap.enabled ? cap.reason || "本地训练启动能力未启用。"
                : null;
  const evaluationSelectionBlocker = !selectedEvaluationProfileId ? "请先选择适用于当前阶段的评测协议。"
    : evaluationDatasetRole && !selectedEvaluationDatasetId ? `请先选择已登记的 ${evaluationDatasetRole} 评测数据集。`
      : null;

  useEffect(() => {
    if (!run || run.status !== "completed" || evaluationActive || evaluationDraftHydratedRun.current === run.run_id) return;
    evaluationDraftHydratedRun.current = run.run_id;
    const draft = readEvaluationDraft(run.run_id);
    if (draft?.profileId) setSelectedEvaluationProfileId(draft.profileId);
    if (draft?.datasetId) setSelectedEvaluationDatasetId(draft.datasetId);
  }, [run?.run_id, run?.status, run?.evaluation_status, evaluationActive]);

  function saveEvaluationDraft(profileId: string, datasetId: string) {
    if (!run?.run_id) return;
    try { window.sessionStorage.setItem(`${EVALUATION_DRAFT_PREFIX}${run.run_id}`, JSON.stringify({ profileId, datasetId } satisfies EvaluationDraft)); }
    catch { /* Storage can be disabled; the current form remains usable. */ }
  }

  async function checkEvaluation() {
    if (!cap?.token || !run?.run_id) return;
    setBusy(true); setError(""); setEvaluationPreflight(null);
    try {
      setEvaluationPreflight(await call<EvaluationPreflight>("/api/launcher/evaluation/preflight", cap.token, { training_run_id: run.run_id }));
      setRunTab("evaluation");
    } catch (e) { setError(e instanceof Error ? e.message : "评测预检失败"); }
    finally { setBusy(false); }
  }

  async function startEvaluation() {
    if (!cap?.token || !evaluationPreflight?.ok || !evaluationPreflight.confirmation_token) return;
    const accepted = window.confirm(`将启动固定探索性评测 ${evaluationPreflight.protocol_version}，使用 ${evaluationPreflight.benchmarks?.map((b) => `${b.name} ${b.samples} 条${b.sampling ? `（${b.sampling}）` : ""}`).join("、")}。\n模型：${String(evaluationPreflight.model?.model_id)}\n输出：${evaluationPreflight.output}\n评测会占用 GPU；缺失缓存时运行中可能下载数据。确认启动？`);
    if (!accepted) return;
    setBusy(true); setError("");
    try {
      await call("/api/launcher/evaluation/start", cap.token, { confirmation_token: evaluationPreflight.confirmation_token });
      setEvaluationPreflight(null); setRunTab("evaluation"); await refreshRun();
    } catch (e) { setError(e instanceof Error ? e.message : "评测启动失败"); }
    finally { setBusy(false); }
  }

  async function checkRegisteredStageEvaluation() {
    if (!cap?.token || !run?.run_id || !selectedEvaluationProfileId) return;
    setBusy(true); setStageEvaluationRequest("preflight_pending"); setError(""); setStageEvaluationPreflight(null);
    try {
      const payload = { training_run_id: run.run_id, profile_id: selectedEvaluationProfileId,
        ...(selectedEvaluationDatasetId ? { dataset_id: selectedEvaluationDatasetId } : {}) };
      const result = await call<Record<string, any>>("/api/training/evaluation/preflight", cap.token, payload);
      setStageEvaluationPreflight(result);
    } catch (e) { setError(e instanceof Error ? e.message : "阶段评测预检失败"); }
    finally { setBusy(false); setStageEvaluationRequest("idle"); }
  }

  async function inspectStageEvaluationStart(initialCheck = false) {
    const candidateId = stageEvaluationPreflight?.evaluation_run_id;
    if (!candidateId) { setStageEvaluationRequest("token_expired"); return; }
    try {
      const observed = await call<Run>(`/api/launcher/runs/${encodeURIComponent(candidateId)}`);
      if (observed.run_id === candidateId) {
        setStageEvaluationStartedRun({ run_id: candidateId, status: observed.status });
        setStageEvaluationPreflight(null);
        setStageEvaluationRequest("idle");
        setRunTab("evaluation");
        await refreshRun();
        return;
      }
    } catch { /* A missing Run does not establish that an in-flight POST was never accepted. */ }
    if (!initialCheck) {
      setStageEvaluationPreflight(null);
      setStageEvaluationRequest("token_expired");
      setError("没有查到本次预检对应的评测 Run；请重新预检后再决定是否启动，不会自动重试。");
    }
  }

  async function startRegisteredStageEvaluation() {
    if (!cap?.token || !stageEvaluationPreflight?.confirmation_token) return;
    const accepted = window.confirm(`将启动阶段评测 ${stageEvaluationPreflight.profile_id}，模型 ${stageEvaluationPreflight.model?.model_id}。\n输出：${stageEvaluationPreflight.output}\n评测会占用本地 GPU，确认启动？`);
    if (!accepted) return;
    setBusy(true); setStageEvaluationRequest("submit_pending"); setError("");
    try {
      const started = await call<{ run_id: string; status: string }>("/api/training/evaluation/start", cap.token, { confirmation_token: stageEvaluationPreflight.confirmation_token });
      setStageEvaluationStartedRun({ run_id: started.run_id, status: started.status || "starting" });
      setStageEvaluationRequest("idle");
      setStageEvaluationPreflight(null); setRunTab("evaluation");
      await refreshRun();
    } catch (e) {
      const message = e instanceof Error ? e.message : "阶段评测启动失败";
      if (e instanceof ApiRequestError) {
        setStageEvaluationPreflight(null); setStageEvaluationRequest("token_expired");
        setError(`${message}。请重新预检，不会自动重试。`);
      } else {
        setStageEvaluationRequest("submission_unknown");
        setError(`${message}。正在核实评测 Run，不会自动重试。`);
        await inspectStageEvaluationStart(true);
      }
    }
    finally { setBusy(false); }
  }
  return <main className="launcher-page">
    <header className="hero launcher-hero">
      <div>
        <div className="eyebrow">LOCAL EXPERIMENT LAUNCHER <span>／</span> {selectedStage.toUpperCase()}</div>
        <h1>从确认配置到观察真实训练<span>。</span></h1>
      <p>选择训练阶段、已登记数据与父模型；预检确认后启动本地训练，完成后进行阶段评测。</p>
      </div>
      <div>{focusedRunId && <button className="button secondary" onClick={() => { window.location.assign(`${window.location.pathname}?page=launcher&stage=${selectedStage}`); }}>返回当前 Launcher</button>}
        <button className="button secondary" onClick={onBack}>{handoffSourceRunId ? "返回来源 Run" : initialParams.get("return_to") === "launcher" ? "返回 GRPO 数据准备" : "返回数据页"}</button></div>
    </header>

    {!cap?.enabled && <div className="notice warning" role="status"><div><strong>训练启动未启用</strong><p>{cap?.reason || "请用 loopback 地址和 --enable-experiment-launcher 重启本地 UI。"}</p></div></div>}
    {error && <div className="notice danger" role="alert">{error}</div>}
    <WorkflowRoadmap run={focusedRunId ? run : null} />
    {focusedRunId && run && <RunConfiguration run={run} options={workbenchOptions} />}

    {!focusedRunId && <section className="launcher-card stage-workbench" aria-labelledby="stage-workbench-title">
      <div className="launcher-card-heading"><div><div className="eyebrow">阶段工作台</div><h2 id="stage-workbench-title">选择训练阶段</h2></div>
        <span className="badge neutral">切换阶段会清除当前预检摘要</span></div>
      <nav className="stage-selector" aria-label="训练阶段" role="tablist">
        {[ ["cpt", "CPT"], ["sft", "SFT"], ["dpo", "DPO"], ["simpo", "SimPO"], ["grpo", "GRPO"] ].map(([id, label]) =>
          <button key={id} type="button" role="tab" aria-selected={selectedStage === id}
            className={selectedStage === id ? "selected" : ""} onClick={() => chooseStage(id)}>{label}</button>)}
      </nav>
      {selectedProfile ? <>
        <div className="stage-profile-summary">
          <div><span>批准方案</span><code>{selectedProfile.profile_id}</code></div>
          <div><span>训练配方与默认参数</span><strong>{selectedProfile.approved_recipe}{selectedProfile.profile_id === "sft-mini-1024-v1" ? " · 1,024 tokens · seed 42 · LoRA" : ""}</strong></div>
          <div><span>训练数据类型</span><strong>{selectedProfile.dataset_stage} · {selectedProfile.dataset_roles.join(" / ")}</strong></div>
          <div><span>父模型要求</span><strong>{selectedProfile.parent_model_stages.join(" / ")} · {selectedProfile.parent_model_rule}</strong></div>
          <div><span>阶段评测</span><strong>{selectedProfile.evaluation_profiles.map((item) => item.profile_id).join(" / ") || "暂无"}</strong></div>
        </div>
        <div className="notice info" role="status"><strong>当前配置 · {selectedStage.toUpperCase()}</strong><p>方案 {selectedProfile.profile_id} · {selectedDatasetIds.length ? "已选训练数据" : "待选训练数据"} · {stageConfirmation || preflight?.ok ? "预检已通过，请确认是否启动" : stagePreflight ? "已完成检查，请核对结果" : "配置后执行只读预检"}</p></div>
        {selectedProfile.training_parameters?.length ? <details className="stage-custom-parameters">
          <summary>训练参数 <span>{Object.keys(parameterOverrides).length ? `自定义 ${Object.keys(parameterOverrides).length} 项` : "使用默认配置（推荐）"}</span></summary>
          <p className="launcher-muted">默认配置无需修改。可在批准范围内调整参数；修改后会生成仅供本次 Run 使用的 resolved recipe，并重新预检。</p>
          {Object.keys(parameterOverrides).length > 0 && <button type="button" className="button secondary"
            onClick={() => { setParameterOverrides({}); setStagePreflight(null); setPreflight(null); setAcknowledgeUnknown(false); }}>恢复方案默认值</button>}
          {selectedProfile.training_parameters?.length
            ? <div className="stage-parameter-grid">{selectedProfile.training_parameters.map((parameter) => {
              const value = parameterOverrides[parameter.key] ?? parameter.default;
              const onChange = (next: number) => {
                setParameterOverrides((current) => {
                  const updated = { ...current };
                  if (next === parameter.default) delete updated[parameter.key];
                  else updated[parameter.key] = next;
                  return updated;
                });
                setStagePreflight(null); setPreflight(null); setStageConfirmation(null); setAcknowledgeUnknown(false); setStageLicenseAcknowledged(false);
              };
              return <label key={parameter.key}>{parameter.label_zh}
                {parameter.choices.length
                  ? <select aria-label={parameter.label_zh} value={value} onChange={(event) => onChange(Number(event.target.value))}>
                    {Array.from(new Set([parameter.default, ...parameter.choices])).map((choice) => <option key={choice} value={choice}>{choice}{choice === parameter.default ? "（默认）" : ""}</option>)}
                  </select>
                  : <input aria-label={parameter.label_zh} type="number" min={parameter.minimum} max={parameter.maximum}
                    step={parameter.kind === "int" ? 1 : "any"} value={value}
                    onChange={(event) => { if (event.target.value !== "") onChange(Number(event.target.value)); }} />}
                <small>默认值：{parameter.default} · 允许范围：{parameter.minimum}–{parameter.maximum}{parameter.kind === "int" ? "（整数）" : ""}</small>
                {parameter.key === "max_length" && <small>序列越长越占显存，超过设备承载能力时训练仍可能 OOM；此上限不代表显存保证。</small>}
              </label>;
            })}</div>
            : <p className="launcher-muted">此 profile 尚未声明可自定义参数，使用批准配方默认值。</p>}
        </details> : <p className="launcher-muted">此方案未声明可修改参数，训练将严格使用批准配方默认值。</p>}
        {!!selectedProfile.training_memory_modes?.length && <div className="stage-memory-mode">
          <label>显存优化模式<select aria-label="显存优化模式" value={selectedProfile.training_memory_modes.some((mode) => mode.mode_id === memoryModeId)
            ? memoryModeId : selectedProfile.training_memory_modes.find((mode) => mode.default)?.mode_id || "lora"}
            onChange={(event) => { setMemoryModeId(event.target.value); setStagePreflight(null); setPreflight(null);
              setStageConfirmation(null); setAcknowledgeUnknown(false); setStageLicenseAcknowledged(false); }}>
            {selectedProfile.training_memory_modes.map((mode) => <option key={mode.mode_id} value={mode.mode_id}>{mode.label_zh}{mode.default ? "（默认）" : ""}</option>)}
          </select></label>
          <p className="launcher-muted">{selectedProfile.training_memory_modes.find((mode) => mode.mode_id === memoryModeId)?.description_zh}</p>
          {memoryModeId === "qlora-4bit-nf4" && <div className="notice warning" role="note">QLoRA 需本机已安装 bitsandbytes 且 CUDA 可用；预检只检查、不安装。量化可降低权重显存，但长序列和激活值仍可能导致 OOM。</div>}
        </div>}
        <div className="stage-input-grid">
          {stageProfiles.filter((item) => item.stage === selectedStage).length > 1 && <label>训练方案<select value={selectedProfileId}
            onChange={(event) => { const next = stageProfiles.find((item) => item.profile_id === event.target.value); setSelectedProfileId(event.target.value); setSelectedDatasetIds([]); setSelectedParentModelId(""); setParameterOverrides({}); setMemoryModeId(next?.training_memory_modes?.find((mode) => mode.default)?.mode_id || "lora"); setStagePreflight(null); setPreflight(null); setStageConfirmation(null); setAcknowledgeUnknown(false); setStageLicenseAcknowledged(false); setHandoffMessage("");
              const params = new URLSearchParams(window.location.search); params.set("profile_id", event.target.value); params.delete("parent_model_id");
              window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`); }}>
            {stageProfiles.filter((item) => item.stage === selectedStage).map((item) => <option key={item.profile_id} value={item.profile_id}>{item.profile_id}</option>)}
          </select></label>}
          {selectedProfile.parent_model_rule === "external_base"
            ? <div className="stage-parent-locked"><span>基座模型（由批准 CPT 配方与本地缓存固定）</span><strong>{selectedProfile.approved_recipe}</strong><small>预检会核验 pinned revision 与本机权重；不会联网下载。</small></div>
            : selectedProfile.parent_model_rule === "approved_mini_cpt" && cap?.parent_model_id
            ? <div className="stage-parent-locked"><span>固定父模型（由配方指定）</span><strong>Mini M1-CPT · 0.6B</strong><code>{cap.parent_model_id}</code></div>
            : <label>父模型<select id="stage-parent-model-select" aria-label="阶段父模型" value={selectedParentModelId} onChange={(event) => { setSelectedParentModelId(event.target.value); setStagePreflight(null); setPreflight(null); setStageConfirmation(null); setAcknowledgeUnknown(false); setStageLicenseAcknowledged(false); setHandoffMessage("");
              const params = new URLSearchParams(window.location.search); if (event.target.value) params.set("parent_model_id", event.target.value); else params.delete("parent_model_id");
              window.history.replaceState({}, "", `${window.location.pathname}?${params.toString()}`); }}>
              <option value="">选择已登记的父模型</option>{stageParents.map((item) => <option key={item.model_id} value={item.model_id}
                disabled={item.path_available === false || item.registry_count !== 1}>
                {item.stage} · {item.kind} · {item.model_id.slice(-12)}{item.path_available === false ? ` · 不可用：${item.unavailable_reason || "本机权重未通过校验"}` : ""}{item.registry_count !== 1 ? " · 登记不唯一" : ""}
              </option>)}
            </select></label>}
          {handoffMessage && <div className="notice info" role="status">{handoffMessage}</div>}
          {selectedParentModelId && (stageParents.find((item) => item.model_id === selectedParentModelId)?.source_run_id || stageParents.find((item) => item.model_id === selectedParentModelId)?.produced_by_run_id)
            && <p className="launcher-lineage-links">父模型来源 Run：<button className="button secondary" onClick={() => { const item = stageParents.find((candidate) => candidate.model_id === selectedParentModelId)!; onOpenRun(item.source_run_id || item.produced_by_run_id!); }}>{stageParents.find((item) => item.model_id === selectedParentModelId)?.source_run_id || stageParents.find((item) => item.model_id === selectedParentModelId)?.produced_by_run_id}</button></p>}
        </div>
        <fieldset id="stage-dataset-options" className="stage-dataset-options"><legend>选择一个已登记训练数据集（{eligibleStageDatasets.length} 个可用）</legend>
          {stageDatasets.length ? stageDatasets.map((item) => <label key={item.dataset_id}>
            <input type="radio" name="stage-training-dataset" checked={selectedDatasetIds[0] === item.dataset_id}
              disabled={item.eligible === false}
              onChange={() => { setSelectedDatasetIds([item.dataset_id]); setStagePreflight(null); setPreflight(null); setStageConfirmation(null); setAcknowledgeUnknown(false); setStageLicenseAcknowledged(false); }} />
            <span><strong>{item.display_name || item.dataset_id}</strong>{item.display_name && <small>{item.dataset_id}</small>}<small>{item.record_count} 条 · {item.role || "角色待预检"}</small>
              {item.eligible === false && <small className="dataset-option-blocked">不可用于训练：{item.block_reason || "登记证据不完整"}</small>}</span>
          </label>) : <p className="launcher-muted">本地没有发现符合此阶段的数据集。请先在“数据”中导入、质检并登记 canonical 数据。</p>}
        </fieldset>
        {selectedProfile && onComposeDatasets && <button className="button secondary launcher-compose-datasets" onClick={() => onComposeDatasets(selectedProfile.dataset_stage)}>
          需要组合多个来源？前往“数据”创建组合数据集 →
        </button>}
        <section className="stage-validation-options" aria-labelledby="stage-validation-title">
          <div><h3 id="stage-validation-title">训练期间验证</h3><p className="launcher-muted">默认不切分。验证集只用于训练期指标，不参与梯度更新；final_holdout 不可选。</p></div>
          {!selectedProfile.training_validation_supported && <div className="notice warning" role="status">当前方案的训练执行入口尚未接通训练期验证；暂时只能使用“不使用验证”。</div>}
          <label>验证来源<select aria-label="训练期间验证来源" value={validationMode}
            disabled={!selectedProfile.training_validation_supported}
            onChange={(event) => { setValidationMode(event.target.value as "none" | "registered" | "random_split");
              setStagePreflight(null); setStageConfirmation(null); setPreflight(null); }}>
            <option value="none">不使用验证（默认）</option>
            <option value="registered">选择已登记独立验证集</option>
            <option value="random_split">从训练集固定种子随机切分</option>
          </select></label>
          {validationMode === "registered" && <label>独立验证集<select aria-label="独立验证集" value={validationDatasetId}
            onChange={(event) => { setValidationDatasetId(event.target.value); setStagePreflight(null); setStageConfirmation(null); setPreflight(null); }}>
            <option value="">选择 validation / diagnostic_dev 数据集</option>
            {validationStageDatasets.map((item) => <option key={item.dataset_id} value={item.dataset_id}>{item.display_name || item.dataset_id} · {item.record_count} 条 · {item.role}</option>)}
          </select><small>必须与训练阶段 schema 一致，且不能与训练集有相同 canonical 内容或 record ID。</small></label>}
          {validationMode === "random_split" && <div className="stage-validation-random-fields">
            <label>验证比例<input aria-label="验证比例" type="number" min="0.05" max="0.4" step="0.05" value={validationRatio}
              onChange={(event) => { const value = Number(event.target.value); setValidationRatio(value); setStagePreflight(null); setStageConfirmation(null); setPreflight(null); }} />
              <small>允许 5%–40%；按稳定题目/来源分组切分，Preference pair 不拆分。</small></label>
            <label>随机种子<input aria-label="验证随机种子" type="number" min="0" max="2147483647" step="1" value={validationSeed}
              onChange={(event) => { const value = Number(event.target.value); setValidationSeed(value); setStagePreflight(null); setStageConfirmation(null); setPreflight(null); }} /></label>
          </div>}
          {Boolean(stagePreflight?.validation) && <div className="notice info" role="status"><strong>验证预检摘要：</strong> {String((stagePreflight!.validation as Record<string, unknown>).mode)} ·
            {String((stagePreflight!.validation as Record<string, unknown>).validation_count || (stagePreflight!.validation as Record<string, unknown>).record_count || 0)} 条验证样本
            {(stagePreflight!.validation as Record<string, unknown>).train_count !== undefined && <> · {(stagePreflight!.validation as Record<string, unknown>).train_count as number} 条训练样本</>}</div>}
        </section>
        <label className="stage-checkpoint-policy">训练后采用的检查点<select aria-label="训练后采用的检查点" value={checkpointPolicy}
          onChange={(event) => { setCheckpointPolicy(event.target.value as "final" | "best"); setStagePreflight(null); setStageConfirmation(null); setPreflight(null); }}>
          <option value="final">最后保存的检查点（默认，适合固定预算对比）</option>
          <option value="best">验证集最佳检查点（按验证 loss 最低选择）</option>
        </select><small>best 需要启用有效训练期验证；最终测试集不会用于选择。</small></label>
        {!(stageConfirmation || (selectedProfile.stage === "sft" && preflight?.ok)) && <div className="stage-workbench-actions launcher-action-bar">
          <div className="launcher-action-bar-copy"><strong>第 1 步 · 检查训练配置</strong><span>{taskActive ? "本机 GPU 正在执行任务，训练入口暂不可用。" : stageSelectionBlocker || "配置齐全，可以执行预检。"}</span>{taskActive && run && <button type="button" className="button secondary" onClick={() => onOpenRun(run.run_id)}>查看当前 Run · {run.run_id}</button>}{selectedStage === "grpo" && !eligibleStageDatasets.length && <button type="button" className="button secondary" onClick={() => onPrepareGrpoData?.({ profileId: selectedProfile.profile_id, parentModelId: selectedParentModelId || new URLSearchParams(window.location.search).get("parent_model_id") || "" })}>准备 GRPO 数据 →</button>}{stageSelectionBlocker?.includes("数据") && selectedStage !== "grpo" && !eligibleStageDatasets.length && <button type="button" className="button secondary" onClick={onBack}>前往数据模块登记数据</button>}{stageSelectionBlocker?.includes("选择一个") && <a className="button secondary" href="#stage-dataset-options">去选择训练数据集</a>}{stageSelectionBlocker?.includes("父模型") && <a className="button secondary" href="#stage-parent-model-select">去选择父模型</a>}</div>
          <button className={selectedProfile.execution_supported ? "button primary" : "button secondary"} disabled={busy || taskActive || Boolean(stageSelectionBlocker)} onClick={checkStage}>
            {busy ? "正在检查配置与数据…" : selectedProfile.execution_supported ? (stagePreflight ? "重新预检" : "检查配置并预检") : "检查阶段兼容性"}</button>
          {!selectedProfile.execution_supported && <span className="launcher-muted">当前方案仅支持查看要求，启动 adapter 尚未接通。</span>}
        </div>}
        {selectedProfile.execution_blocker && <div className="notice warning" role="status"><div><strong>当前执行能力</strong><p>{selectedProfile.execution_blocker}</p></div></div>}
        {stagePreflight && <section className="stage-preflight-summary" aria-label="阶段预检摘要">
          <h3>{String(Array.isArray(stagePreflight.blockers) && stagePreflight.blockers.length > 0
            ? "存在阻断"
            : Array.isArray(stagePreflight.warnings) && stagePreflight.warnings.length > 0
              ? "预检完成 · 有提醒"
              : stagePreflight.status === "ready_for_confirmation" || stagePreflight.ok
                ? "预检通过"
                : "预检未通过")}</h3>
          <p>{selectedProfile.profile_id} · {selectedDatasetIds.length} 个数据集 · 显存模式 {selectedProfile.training_memory_modes?.find((mode) => mode.mode_id === memoryModeId)?.label_zh || "标准 LoRA"} · 序列长度 {String((stagePreflight.training_parameters as Record<string, any> | undefined)?.effective?.max_length || parameterOverrides.max_length || selectedProfile.training_parameters.find((item) => item.key === "max_length")?.default || "按方案配置")} · 父模型 {String((stagePreflight.parent_model as Record<string, unknown> | undefined)?.model_id || selectedParentModelId || "批准配方固定 Base")}</p>
          {Array.isArray(stagePreflight.blockers) && stagePreflight.blockers.length > 0 && <div className="notice danger"><ul>{(stagePreflight.blockers as string[]).map((item) => <li key={item}>{item}</li>)}</ul></div>}
          {Array.isArray(stagePreflight.warnings) && stagePreflight.warnings.length > 0 && <div className="notice warning"><ul>{(stagePreflight.warnings as string[]).map((item) => <li key={item}>{item}</li>)}</ul></div>}
          {Array.isArray(stagePreflight.checks) && <div className="readiness-check-list" aria-label="逐项预检结果">
            {(stagePreflight.checks as ReadinessCheck[]).map((check) => <article key={check.check_id} className={`readiness-check readiness-${check.status}`}>
              <div><strong>{readinessCheckLabels[check.check_id] || check.check_id}</strong><span className={`readiness-status readiness-status-${check.status}`}>{readinessStatusLabels[check.status] || check.status}</span></div>
              <p>{check.detail}</p>
              {check.repair_hint && <small>处理建议：{check.repair_hint}</small>}
            </article>)}
          </div>}
          <p className="readiness-execution-state" role="status">执行状态：尚未启动。预检只读取本地数据、配置、依赖与硬件状态，不会安装依赖、下载权重或创建训练 Run。</p>
          <details><summary>查看逐项检查和摘要</summary><pre>{pretty(stagePreflight)}</pre></details>
          <p className="launcher-muted">“有提醒”不代表正式验收通过。可执行方案会在这份预检通过后显示确认启动；阶段检查与启动预检已核对相同的父模型 ID/hash 和数据集 ID/canonical hash。</p>
          {stageConfirmation && <div className="stage-launch-confirmation">
            <h3>启动前确认</h3>
            <p>方案 {selectedProfile.profile_id} · 数据集 {stageDatasets.find((item) => item.dataset_id === selectedDatasetIds[0])?.display_name || selectedDatasetIds[0]} · 父模型已绑定。确认后才会占用本机 GPU。</p>
            <details className="launcher-config"><summary>查看 Run、模型、数据及输出的技术身份</summary>
            <dl className="launcher-facts">
              <div><dt>Run ID</dt><dd><code>{stageConfirmation.run_id}</code></dd></div>
              <div><dt>配置指纹</dt><dd><code>{stageConfirmation.bound_config_sha256}</code></dd></div>
              <div><dt>父模型</dt><dd><code>{String(stageConfirmation.parent_model?.model_id || "固定 Base 模型")}</code></dd></div>
              <div><dt>数据集</dt><dd><code>{String((stageConfirmation.datasets as Record<string, any>[] | undefined)?.[0]?.dataset_id || selectedDatasetIds[0])}</code></dd></div>
              <div><dt>输出目录</dt><dd><code>{stageConfirmation.output_location}</code></dd></div>
            </dl></details>
            {(stageConfirmation.datasets as Record<string, any>[] | undefined)?.some((item) => item.license_status !== "verified") && <label className="launcher-acknowledgement"><input type="checkbox" checked={stageLicenseAcknowledged} onChange={(event) => setStageLicenseAcknowledged(event.target.checked)} />我确认按来源声明使用此数据，并理解许可尚未独立核验（dataset_id：<code>{selectedDatasetIds[0]}</code>）。</label>}
            <p className="launcher-muted">确认令牌有效期 {stageConfirmation.expires_in_seconds} 秒；启动时会再次校验数据、模型、配方和 Run 输出身份。</p>
            <div className="launcher-action-bar"><div className="launcher-action-bar-copy"><strong>第 2 步 · 启动 {selectedProfile.stage.toUpperCase()} 训练</strong><span>{taskActive ? "本机 GPU 正在执行任务，不能重复启动。" : (stageConfirmation.datasets as Record<string, any>[] | undefined)?.some((item) => item.license_status !== "verified") && !stageLicenseAcknowledged ? "请先勾选上方来源/许可确认框，再启动训练。" : "核对本次 Run、父模型和数据集后提交。"}</span>{taskActive && run && <button type="button" className="button secondary" onClick={() => onOpenRun(run.run_id)}>查看当前 Run · {run.run_id}</button>}</div><button className="button primary" disabled={busy || taskActive || ((stageConfirmation.datasets as Record<string, any>[] | undefined)?.some((item) => item.license_status !== "verified") && !stageLicenseAcknowledged)} onClick={startRegisteredStage}>{busy ? "正在启动…" : taskActive ? "本机 GPU 任务进行中" : `启动 ${selectedProfile.stage.toUpperCase()} 训练`}</button></div>
          </div>}
          {preflight && selectedProfile.execution_supported && <div className="stage-launch-confirmation">
            <h3>{preflight.ok ? "启动预检通过 · 请确认" : "启动预检未通过"}</h3>
            {preflight.blockers?.length > 0 && <div className="notice danger"><ul>{preflight.blockers.map((item) => <li key={item}>{item}</li>)}</ul></div>}
            {preflight.warnings?.length > 0 && <div className="notice warning"><ul>{preflight.warnings.map((item) => <li key={item}>{item}</li>)}</ul></div>}
            {preflight.ok && <>
              <p>方案 {selectedProfile.profile_id} · 数据集 {stageDatasets.find((item) => item.dataset_id === selectedDatasetIds[0])?.display_name || selectedDatasetIds[0]} · 数据质量 {String(preflight.dataset?.quality_status || "未测量")} · 许可 {String(preflight.dataset?.license_status || "未知")}。确认后才会占用本机 GPU。</p>
              <details className="launcher-config"><summary>查看 Run、模型、数据及输出的技术身份</summary>
              <dl className="launcher-facts">
                <div><dt>本次 run_id</dt><dd><code>{preflight.run_id}</code></dd></div>
                <div><dt>配方 / 配置指纹</dt><dd><code>{preflight.recipe} · {preflight.fingerprint}</code></dd></div>
                <div><dt>父模型 ID / SHA-256</dt><dd><code>{String(preflight.model?.model_id)} · {String(preflight.model?.sha256)}</code></dd></div>
                <div><dt>数据集 ID / canonical SHA-256</dt><dd><code>{String(preflight.dataset?.dataset_id)} · {String(preflight.dataset?.canonical_sha256)}</code></dd></div>
                <div><dt>数据质量 / 许可</dt><dd>{String(preflight.dataset?.quality_status || "未测量")} / {String(preflight.dataset?.license_status || "未知")}</dd></div>
                <div><dt>GPU 状态</dt><dd>{preflight.gpu?.available ? `${preflight.gpu.name} · ${preflight.gpu.free_mib} / ${preflight.gpu.total_mib} MiB 可用（仅供参考）` : "未知；未阻断预检"}</dd></div>
                <div><dt>输出目录</dt><dd><code>{preflight.output}</code></dd></div>
              </dl></details>
              <details className="launcher-config"><summary>查看精确解析配置</summary><pre>{pretty(preflight.resolved_config)}</pre></details>
              <p className="launcher-muted">启动令牌有效期 {preflight.expires_in_seconds} 秒；超时或输入变化后必须重新预检。</p>
              {preflight.dataset?.license_status !== "verified" && <label className="launcher-acknowledgement"><input type="checkbox" checked={acknowledgeUnknown} onChange={(event) => setAcknowledgeUnknown(event.target.checked)} />我确认仅依据来源/许可的当前声明使用数据，并理解该状态尚未独立核验（dataset_id：<code>{String(preflight.dataset?.dataset_id)}</code>）。</label>}
              <div className="launcher-action-bar"><div className="launcher-action-bar-copy"><strong>第 2 步 · 启动 SFT 训练</strong><span>{taskActive ? "本机 GPU 正在执行任务，不能重复启动。" : preflight.dataset?.license_status !== "verified" && !acknowledgeUnknown ? "请先勾选上方来源/许可确认框，再启动训练。" : "检查已绑定的配方、模型、数据和输出目录。"}</span>{taskActive && run && <button type="button" className="button secondary" onClick={() => onOpenRun(run.run_id)}>查看当前 Run · {run.run_id}</button>}</div><button className="button primary" disabled={busy || taskActive || (preflight.dataset?.license_status !== "verified" && !acknowledgeUnknown)} onClick={start}>{busy ? "正在启动…" : taskActive ? "本机 GPU 任务进行中" : "启动 SFT 训练"}</button></div>
              <p className="launcher-muted">启动前会再次核对确认令牌。历史 RTX 4060 显存余量较小；关闭 UI 服务会中断训练。</p>
            </>}
          </div>}
        </section>}
      </> : <p className="launcher-muted">正在读取批准的阶段方案…</p>}
    </section>}

    {run?.run_kind === "stage-training" && run.stage === selectedStage && <section className="launcher-card" aria-labelledby="stage-run-title" aria-live="polite">
      <div className="launcher-card-heading"><div><div className="eyebrow">{selectedStage.toUpperCase()} · 训练运行</div><h2 id="stage-run-title">{messages[run.status] || run.status}</h2></div>
        <span className={`badge ${run.status === "completed" ? "success" : active ? "warning" : run.status === "failed" || run.status === "oom" ? "danger" : "neutral"}`}>{messages[run.status] || run.status}</span></div>
      <dl className="launcher-facts">
        <div><dt>已运行</dt><dd>{runDuration(run.elapsed_seconds)}</dd></div>
        <div><dt>最近状态更新</dt><dd>{runUpdated(run.updated_at)}</dd></div>
        <div><dt>训练进度</dt><dd>{run.progress_status === "measured" && run.progress ? `Step ${run.progress.step}${run.progress.total ? ` / ${run.progress.total}` : ""}` : "尚无可靠进度信息"}</dd></div>
        <div><dt>评测状态</dt><dd>{run.evaluation_status === "running" ? "评测中" : run.evaluation_status === "starting" ? "评测启动中" : run.evaluation_status === "completed" ? "评测完成" : run.evaluation_status === "partial" ? "部分完成" : run.evaluation_status === "failed" ? "评测失败" : run.status === "completed" ? "可启动评测" : "训练完成后可启动"}</dd></div>
      </dl>
      <details className="launcher-config"><summary>查看 Run ID、数据集与父模型身份</summary><p>Run ID：<code>{run.run_id}</code></p><p>数据集：<code>{run.dataset_id || "未记录"}</code> · 父模型：<code>{run.parent_model_id || "由配方固定或未记录"}</code></p></details>
      {run.detail && <div className="notice warning">{run.detail}</div>}
      <nav className="launcher-tabs" role="tablist" aria-label="阶段运行详情">
        {([ ["overview", "概览"], ["logs", "日志"], ["evaluation", "阶段评测"], ["artifacts", "产物与血缘"] ] as const).map(([id, label]) =>
          <button key={id} role="tab" aria-selected={runTab === id} className={runTab === id ? "selected" : ""} onClick={() => setRunTab(id)}>{label}</button>)}
      </nav>
      {runTab === "overview" && <div className="launcher-run-overview"><p>本地阶段训练由批准 CLI adapter 执行；当前状态、日志与模型产物来自该 Run 实际记录。</p>
        <button className="button secondary" onClick={() => onOpenRun(run.run_id)}>在实验 Runs 查看规范 Run 详情</button></div>}
      {runTab === "logs" && <><div className="launcher-log-heading"><strong>训练日志 {active && <span className="launcher-live-indicator">● 实时更新</span>}</strong>
        <button className="button secondary" onClick={() => refreshRun().catch((e) => setError(e.message))}>立即刷新</button></div>
        <pre className="launcher-log" aria-label="阶段训练日志" aria-live="off">{run.log_tail || (active ? "训练已启动，等待训练器写出首条日志…" : "无可用日志")}</pre></>}
      {runTab === "evaluation" && <div className="launcher-evaluation-panel">
        <h3>阶段评测</h3>
        {evaluationActive ? <>
          <TaskStatusCard input={{ request: "idle", run: run.evaluation_run_id
            ? { run_id: run.evaluation_run_id, status: run.evaluation_status || "starting" } : stageEvaluationStartedRun }}
            onAction={(_, runId) => { if (runId) onOpenRun(runId); }}
            detail={<small>训练 Run：{run.run_id}；刷新页面会继续读取最新评测日志，不会重复启动。</small>} />
          {run.evaluation_detail && <div className="notice warning" role="status">{run.evaluation_detail}</div>}
          <section className="launcher-evaluation-log" aria-label="实时阶段评测日志" aria-live="polite">
            <div><strong>评测日志</strong><span className="launcher-live-indicator">● 实时更新</span></div>
            <pre>{run.evaluation_log_tail || "评测任务已启动，正在准备评测环境…"}</pre>
          </section>
        </> : <>
          {run.evaluation_detail && <div className="notice warning" role="status">{run.evaluation_detail}</div>}
          {run.status !== "completed" && <p className="launcher-muted">训练成功并登记模型后，此处会开放阶段评测。</p>}
          {run.status === "completed" && <>
          <label>评测协议<select id="stage-evaluation-profile" value={selectedEvaluationProfileId} onChange={(event) => { setSelectedEvaluationProfileId(event.target.value); setSelectedEvaluationDatasetId(""); setStageEvaluationPreflight(null); setStageEvaluationRequest("idle"); setError(""); saveEvaluationDraft(event.target.value, ""); }}>
            <option value="">选择适用协议</option>{(selectedProfile?.evaluation_profiles || []).map((item) => <option key={item.profile_id} value={item.profile_id}>{item.profile_id} · {item.protocol}</option>)}
          </select></label>
          {evaluationDatasetRole && <label>{evaluationDatasetRole === "evaluation_only" ? "已登记客观题评测集" : "已登记评测数据集"}<select id="stage-evaluation-dataset" value={selectedEvaluationDatasetId} onChange={(event) => { setSelectedEvaluationDatasetId(event.target.value); setStageEvaluationPreflight(null); setStageEvaluationRequest("idle"); setError(""); saveEvaluationDraft(selectedEvaluationProfileId, event.target.value); }}>
            <option value="">选择 {evaluationDatasetRole}</option>{stageEvaluationDatasets.map((item) => <option key={item.dataset_id} value={item.dataset_id} disabled={item.eligible === false}>{item.display_name || item.dataset_id} · {item.record_count} 条{item.eligible === false ? ` · 不可用：${item.block_reason || "登记证据不完整"}` : ""}</option>)}
          </select></label>}
          {taskActive && <div className="notice info" role="status"><strong>本机 GPU 任务进行中</strong><p>训练与评测共用 GPU 锁，当前不能重复启动。</p><button type="button" className="button secondary" onClick={() => onOpenRun(run.run_id)}>查看当前 Run · {run.run_id}</button></div>}
          <TaskStatusCard input={{ request: stageEvaluationRequest,
            preflight: stageEvaluationPreflight ? stageEvaluationPreflight.ok ? "ready" : "blocked" : "none" }}
            actionLabel={stageEvaluationRequest === "idle" && !stageEvaluationPreflight ? "检查评测配置"
              : stageEvaluationRequest === "idle" && stageEvaluationPreflight?.ok ? "启动阶段评测" : undefined}
            actionDisabled={busy || taskActive || Boolean(evaluationSelectionBlocker)}
            detail={<span>{evaluationSelectionBlocker || (taskActive ? "本机 GPU 正在执行任务。" : "选择协议和评测数据集后，先执行只读预检。")}</span>}
            onAction={(action) => {
              if (action === "preflight") void checkRegisteredStageEvaluation();
              else if (action === "submit") void startRegisteredStageEvaluation();
              else if (action === "inspect") void inspectStageEvaluationStart();
              else if (action === "edit") setStageEvaluationPreflight(null);
            }} />
          {stageEvaluationPreflight && <div className="launcher-eval-preflight">
            {stageEvaluationPreflight.blockers?.length > 0 && <div className="notice danger"><ul>{stageEvaluationPreflight.blockers.map((item: string) => <li key={item}>{item}</li>)}</ul></div>}
            {stageEvaluationPreflight.warnings?.length > 0 && <div className="notice warning"><ul>{stageEvaluationPreflight.warnings.map((item: string) => <li key={item}>{item}</li>)}</ul></div>}
            {stageEvaluationPreflight.ok && <><p>协议：{stageEvaluationPreflight.protocol_version} · 模型：<code>{stageEvaluationPreflight.model?.model_id}</code> · 输出：<code>{stageEvaluationPreflight.output}</code></p>
              {selectedEvaluationProfileId === "choice-accuracy-v1" && <p>评测集：<code>{stageEvaluationPreflight.dataset?.dataset_id}</code> · {stageEvaluationPreflight.dataset?.record_count ?? "?"} 题 · SHA-256 <code>{String(stageEvaluationPreflight.dataset?.content_sha256 || "").slice(0, 16)}…</code> · 仅当评测集摘要、题数和协议一致时结果可直接比较。</p>}
              </>}</div>}
          {run.evaluation_metrics?.evaluation_suite && <div className="notice info"><strong>客观题评测结果 · {run.evaluation_metrics.evaluation_suite.dataset_id}</strong><p>{run.evaluation_metrics.evaluation_suite.sample_count ?? "?"} 题 · {run.evaluation_metrics.protocol_version} · SHA-256 {String(run.evaluation_metrics.evaluation_suite.dataset_sha256 || "").slice(0, 16)}…</p><p>{run.evaluation_metrics.comparability?.note || "仅相同评测集内容摘要、覆盖和协议的结果可直接比较。"}</p></div>}
          {run.evaluation_metrics?.metrics?.length ? <div className="launcher-metric-grid">{run.evaluation_metrics.metrics.map((metric) => <article key={metric.name}><span>{metric.name} · {metric.metric}</span><strong>{typeof metric.value === "number" && Number.isFinite(metric.value) ? metric.value.toFixed(4) : "未测量"}</strong><small>{metric.correct !== undefined ? `正确 ${metric.correct} / ${metric.denominator ?? "未记录"} 题` : `实际分母：${metric.denominator ?? "未记录"}${metric.denominator_unit ? ` ${metric.denominator_unit}` : ""}`}{metric.parse_failures !== undefined ? ` · 解析失败 ${metric.parse_failures}` : ""}</small></article>)}</div> : null}
          </>}
        </>}
        {run.evaluation_run_id && <p className="launcher-lineage-links"><a href={`/api/launcher/evaluations/${encodeURIComponent(run.evaluation_run_id)}/result`} target="_blank" rel="noreferrer">查看评测结果</a> · <a href={`/api/launcher/evaluations/${encodeURIComponent(run.evaluation_run_id)}/lineage`} target="_blank" rel="noreferrer">查看评测 Lineage</a></p>}
      </div>}
      {runTab === "artifacts" && <div className="launcher-run-overview"><p><a href={`/api/launcher/runs/${encodeURIComponent(run.run_id)}/lineage`} target="_blank" rel="noreferrer">查看训练 Lineage</a></p>
        {run.outputs?.map((item) => <p key={item}><code>{item}</code></p>)}{run.model_artifacts?.map((item) => <article className="launcher-model-artifact" key={item.model_id}><strong>{item.display_name || `${item.stage || "模型"} · ${item.kind || "产物"}`}</strong><p>模型 ID：<code>{item.model_id}</code></p><p>来源训练 Run：<code>{run.run_id}</code></p>{item.produced_by_run_id && item.produced_by_run_id !== run.run_id && <p>合并 Run：<code>{item.produced_by_run_id}</code></p>}<ModelDisplayNameEditor modelId={item.model_id} displayName={item.display_name} token={cap?.token || ""}
          onSaved={(name) => setRun((current) => current ? { ...current, model_artifacts: current.model_artifacts?.map((model) => model.model_id === item.model_id ? { ...model, display_name: name } : model) } : current)} compact /></article>)}
      </div>}
    </section>}

    {selectedStage === "sft" && run && run.run_kind !== "stage-training" && <section className="launcher-card" aria-labelledby="run-title" aria-live="polite">
      <div className="launcher-card-heading"><div><div className="eyebrow">训练运行</div><h2 id="run-title">{evaluationActive ? "训练完成 · 评测中" : messages[run.status] || run.status}</h2></div>
        <span className={`badge ${run.status === "completed" && !evaluationActive ? "success" : active || evaluationActive ? "warning" : run.status === "failed" || run.status === "oom" ? "danger" : "neutral"}`}>{evaluationActive ? "评测中" : messages[run.status] || run.status}</span></div>
      <dl className="launcher-facts">
        <div><dt>run_id</dt><dd><code>{run.run_id}</code></dd></div>
        <div><dt>已运行</dt><dd>{runDuration(run.elapsed_seconds)}</dd></div>
        <div><dt>最近状态更新</dt><dd>{runUpdated(run.updated_at)}</dd></div>
        <div><dt>训练进度</dt><dd>{run.progress_status === "measured" && run.progress ? `Step ${run.progress.step}${run.progress.total ? ` / ${run.progress.total}` : ""}` : "尚无可靠进度信息"}</dd></div>
        <div><dt>统一评测</dt><dd>{run.evaluation_status === "running" ? "评测中" : run.evaluation_status === "completed" ? "评测完成" : run.evaluation_status === "partial" ? "部分完成" : run.evaluation_status === "failed" ? "评测失败" : run.status === "completed" ? "待启动" : "训练结束后可启动"}</dd></div>
      </dl>
      <p className="launcher-lineage-links"><button className="button secondary" onClick={() => onOpenRun(run.run_id)}>在实验 Runs 查看完整记录</button></p>
      {run.detail && <div className="notice warning">{run.detail}</div>}
      <nav className="launcher-tabs" role="tablist" aria-label="运行详情">
        {([ ["overview", "概览"], ["logs", "日志"], ["evaluation", "评测"], ["artifacts", "产物与血缘"] ] as const).map(([id, label]) =>
          <button key={id} role="tab" aria-selected={runTab === id} className={runTab === id ? "selected" : ""} onClick={() => setRunTab(id)}>{label}</button>)}
      </nav>
      {runTab === "overview" && <div className="launcher-run-overview">
        {run.training_metrics && <><h3>训练指标</h3><dl className="launcher-facts"><div><dt>{active ? "最新 loss" : "训练 loss"}</dt><dd>{String(run.training_metrics.loss ?? run.training_metrics.train_loss ?? "—")}</dd></div><div><dt>Token accuracy</dt><dd>{String(run.training_metrics.token_acc ?? "—")}</dd></div><div><dt>训练耗时</dt><dd>{String(run.training_metrics.elapsed_time ?? "—")}</dd></div><div><dt>剩余时间（训练中）</dt><dd>{String(run.training_metrics.remaining_time ?? "—")}</dd></div></dl></>}
        {run.validation?.mode && <section aria-label="训练期验证指标"><h3>训练期验证指标</h3>
          {run.validation.mode === "none" ? <p className="launcher-muted">未启用训练期间验证（not_measured）。</p>
            : run.validation_metrics && Object.keys(run.validation_metrics).length > 0
              ? <dl className="launcher-facts">{Object.entries(run.validation_metrics).map(([key, value]) => <div key={key}><dt>{key}</dt><dd>{String(value)}</dd></div>)}</dl>
              : <p className="launcher-muted">验证来源已配置{run.validation.validation_count !== undefined ? ` · ${run.validation.validation_count} 条验证样本` : ""}；尚未从训练日志测得 eval_* 指标。</p>}
        </section>}
        <p>训练日志每 1.5 秒自动更新。评测与训练串行执行，共用本地 GPU 任务锁。</p>
      </div>}
      {runTab === "logs" && <>
        <div className="launcher-log-heading"><strong>训练日志 {active && <span className="launcher-live-indicator">● 实时更新</span>}</strong><div className="launcher-log-actions">
          {!followLogs && <button className="button secondary" onClick={resumeLogFollow}>{newLogs ? "有新日志 · 回到最新" : "恢复自动跟随"}</button>}
          <button className="button secondary" onClick={() => refreshRun().catch((e) => setError(e.message))}>立即刷新</button></div></div>
        {run.log_sources?.length ? <p className="launcher-muted">日志来源：{run.log_sources.join("、")}</p> : null}
        {run.training_metrics && <p className="launcher-muted">最新训练指标：{Object.entries(run.training_metrics).map(([key, value]) => `${key} ${value}`).join(" · ")}</p>}
        {run.log_truncated && <p className="launcher-muted">只显示最近日志尾部。</p>}
        <pre ref={logElement} onScroll={handleLogScroll} className="launcher-log" aria-label="训练日志" aria-live="off">{run.log_tail || (active ? "训练已启动，等待训练器写出首条日志…" : "无可用日志")}</pre>
        {evaluationActive && <><h3>评测日志 <span className="launcher-live-indicator">● 实时更新</span></h3><pre className="launcher-log" aria-label="评测日志" aria-live="off">{run.evaluation_log_tail || "评测进程已启动，等待输出…"}</pre></>}
      </>}
      {runTab === "evaluation" && <div className="launcher-evaluation-panel">
        <div className="launcher-card-heading"><div><div className="eyebrow">MATH-EVAL-V1</div><h3>固定探索性评测</h3></div><span className={`badge ${run.evaluation_status === "completed" ? "success" : run.evaluation_status === "partial" ? "warning" : run.evaluation_status === "failed" || run.evaluation_status === "interrupted" ? "danger" : "neutral"}`}>{run.evaluation_status || "pending"}</span></div>
        {evaluationActive && <div className="notice info" role="status"><strong>本机 GPU 评测正在运行</strong><p>评测与训练共用 GPU 锁，请勿重复启动。</p><button type="button" className="button secondary" onClick={() => onOpenRun(run.run_id)}>查看当前 Run · {run.run_id}</button></div>}
        {run.status === "completed" && mergedReady && !evaluationActive && !evaluationPreflight?.ok && <div className="launcher-action-bar"><div className="launcher-action-bar-copy"><strong>第 1 步 · 检查评测配置</strong><span>下一步将核验适用协议和评测模型。</span></div><button className="button primary" disabled={busy || taskActive} onClick={checkEvaluation}>{busy ? "正在预检…" : evaluationPreflight ? "重新检查评测配置" : "检查评测配置"}</button></div>}
        {!mergedReady && <p className="launcher-muted">评测入口仅在训练成功并核实 merged M2-SFT 模型后开放。</p>}
        {run.evaluation_detail && <div className="notice warning">{run.evaluation_detail}</div>}
        {!evaluationActive && evaluationPreflight && <div className="launcher-eval-preflight">
          <h4>{evaluationPreflight.ok ? "评测预检通过，请确认" : "评测暂不可启动"}</h4>
          {evaluationPreflight.blockers?.length > 0 && <div className="notice danger"><ul>{evaluationPreflight.blockers.map((item) => <li key={item}>{item}</li>)}</ul></div>}
          {evaluationPreflight.warnings?.length > 0 && <div className="notice warning"><ul>{evaluationPreflight.warnings.map((item) => <li key={item}>{item}</li>)}</ul></div>}
          {evaluationPreflight.ok && <><dl className="launcher-facts"><div><dt>协议与样本</dt><dd>{evaluationPreflight.protocol_version} · {evaluationPreflight.benchmarks?.map((item) => `${item.name} ${item.samples}${item.sampling ? `（${item.sampling}）` : ""}`).join(" / ")}</dd></div><div><dt>模型 / 来源</dt><dd><code>{String(evaluationPreflight.model?.model_id)} · {String(evaluationPreflight.model?.produced_by_run_id)}</code></dd></div><div><dt>GPU</dt><dd>将占用本地 GPU；预检读数仅供参考</dd></div><div><dt>输出</dt><dd><code>{evaluationPreflight.output}</code></dd></div></dl><p>确认有效期 {evaluationPreflight.expires_in_seconds} 秒。仅固定协议与模型可用；不接受自定义参数或路径。</p><div className="launcher-action-bar"><div className="launcher-action-bar-copy"><strong>第 2 步 · 启动阶段评测</strong><span>{taskActive ? "本机 GPU 正在执行任务，评测无法启动。" : `协议：${evaluationPreflight.protocol_version} · 模型：${String(evaluationPreflight.model?.model_id)} · 输出：${evaluationPreflight.output}`}</span>{taskActive && run && <button type="button" className="button secondary" onClick={() => onOpenRun(run.run_id)}>查看当前 Run · {run.run_id}</button>}</div><button className="button primary" disabled={busy || taskActive} onClick={startEvaluation}>{taskActive ? "本机 GPU 任务进行中" : "启动阶段评测"}</button></div></>}
        </div>}
        {evaluationActive && <><h4>评测实时日志</h4><pre className="launcher-log" aria-label="评测日志" aria-live="off">{run.evaluation_log_tail || "等待评测日志…"}</pre></>}
        {run.evaluation_metrics?.metrics?.length ? <><h4>评测指标 · {run.evaluation_metrics.protocol_version}</h4><div className="launcher-metric-grid">{run.evaluation_metrics.metrics.map((metric) => <article key={metric.name}><span>{metric.name} · {metric.metric}</span><strong>{typeof metric.value === "number" && Number.isFinite(metric.value) ? (metric.value <= 1 ? `${(metric.value * 100).toFixed(1)}%` : metric.value.toFixed(2)) : "未测量"}</strong><small>实际样本数：{metric.denominator ?? "报告未记录"}{metric.denominator_unit ? ` ${metric.denominator_unit}` : ""}{metric.parse_failures !== undefined ? ` · 解析失败 ${metric.parse_failures}` : ""}{metric.reason ? ` · ${metric.reason}` : ""}</small></article>)}</div><p className="launcher-muted">此处只显示实际生成的分数；缺失指标显示为缺失，不会填 0。</p></> : run.evaluation_status === "completed" || run.evaluation_status === "partial" ? <p className="launcher-muted">暂无可用分数；评测报告可能未生成完整结果。</p> : null}
        {run.evaluation_run_id && <p className="launcher-lineage-links"><a href={`/api/launcher/evaluations/${encodeURIComponent(run.evaluation_run_id)}/result`} target="_blank" rel="noreferrer">查看 eval.json</a> · <a href={`/api/launcher/evaluations/${encodeURIComponent(run.evaluation_run_id)}/lineage`} target="_blank" rel="noreferrer">查看评测 Lineage</a></p>}
      </div>}
      {runTab === "artifacts" && <div><strong>训练 CLI 产物</strong>{run.outputs?.length ? <ul>{run.outputs.map((item) => <li key={item}><code>{item}</code></li>)}</ul> : <p className="launcher-muted">暂无已发现产物。</p>}
        {run.lineage_path && <p><a href={`/api/launcher/runs/${encodeURIComponent(run.run_id)}/lineage`} target="_blank" rel="noreferrer">查看训练 Lineage 记录</a></p>}
        {run.model_ids?.map((modelId) => <p key={modelId}><a href={`/api/launcher/runs/${encodeURIComponent(run.run_id)}/models/${encodeURIComponent(modelId)}`} target="_blank" rel="noreferrer">查看模型产物登记：{modelId}</a></p>)}
        {run.evaluation_outputs?.map((item) => <p key={item}><code>{item}</code></p>)}</div>}
    </section>}
  </main>;
}
