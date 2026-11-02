import { useEffect, useMemo, useRef, useState } from "react";
import {
  classifyOverviewLoadError,
  loadJsonResource,
} from "./targetedLoad.mjs";
import type { OverviewLoadError } from "./targetedLoad.mjs";
import { GenerationReviewPanel } from "./GenerationReviewPanel";

type Profile = {
  profile_id: string;
  adapter: string;
  endpoint?: string;
  model: string;
  api_key_env?: string;
  supported_parameters?: string[];
  supports_json_object?: boolean;
  revision?: string | null;
};
type Scenario = {
  scenario_id: string;
  report_id?: string;
  eval_id?: string;
  model_id?: string;
  dataset_id?: string;
  protocol_version?: string;
  sample_count?: number;
  denominator?: number | null;
  correct?: number;
  incorrect?: number;
  parse_failed?: number;
  unsupported?: number;
  missing_prediction?: number;
  accuracy?: number | null;
  status?: string;
};
type Overview = {
  available: boolean;
  scenarios: Scenario[];
  plans: Record<string, any>[];
  generations: Record<string, any>[];
  effects: Record<string, any>[];
  truncated?: boolean;
};
type Capabilities = {
  available: boolean;
  profiles: Profile[];
  prompt: {
    template_id: string;
    version: string;
    template: string;
    allowed_variables: string[];
  } | null;
  protocol: {
    protocol_id: string;
    version: string;
    response_format: string;
    required_fields: string[];
  } | null;
};
type StageCapability = {
  stage: "cpt" | "sft" | "preference" | "grpo";
  dataset_schema: string;
  parent_model_stage: string;
  generation_ui_available: boolean;
  training_ui_available: boolean;
  evaluation: { available: boolean; supported_metrics: string[] };
  ready_for_execution: boolean;
  generation_blockers?: string[];
  generation_modes?: Record<"ordinary" | "targeted", {
    status: "blocked" | "preflight_required";
    blockers: string[];
    pending_checks: string[];
  }>;
  training_blockers?: string[];
  blockers: string[];
};
type SftBundle = {
  bundle_id: string;
  source_dataset_id: string;
  scenario_id?: string;
  provider_profile_id?: string;
  model?: string;
  max_requests: number;
  max_candidates: number;
  shared_stage_plans?: { plan_id: string; version: number; scenario_id: string }[];
};
type SftGenerationCapabilities = {
  enabled: boolean;
  local_only?: boolean;
  token?: string;
  reason?: string;
  bundles?: SftBundle[];
};
type SftReadiness = {
  read_only: boolean;
  provider_calls: number;
  datasets: { dataset_id: string; sample_count: number; content_sha256?: string | null; eligible: boolean; missing_reasons: string[] }[];
  profiles: { profile_id: string; adapter: string; model: string; credential_env_configured: boolean; is_example: boolean }[];
  readiness: { dataset_id: string; eligible: boolean; missing_reasons: string[] }[];
  summary: { dataset_count: number; ready_count: number; scenario_report_count: number };
};
type SftPreflight = {
  status: string;
  bundle_id: string;
  source_dataset_id: string;
  model?: string;
  desired_usable_records: number;
  candidate_budget: number;
  max_requests: number;
  max_output_tokens_per_request: number;
  cost_status: string;
  estimated_output_cost_upper_bound_usd?: number | null;
  provider_calls: number;
  network_used: boolean;
  targeted_stage_plan?: { plan_id: string; plan_version: number };
};
type SftGenerationRun = {
  execution_id: string;
  status: string;
  phase?: string;
  detail?: string | null;
  progress?: Record<string, number> | null;
  result?: Record<string, any> | null;
  registered_dataset_id?: string;
  registration?: Record<string, any>;
};
type SftPreview = {
  count: number;
  canonical_sha256: string;
  semantic_correctness: string;
  records: { id: string; raw_problem: string; messages: { role: string; content: string }[] }[];
  truncated: boolean;
};

async function syntheticRequest<T>(path: string, token: string, body?: unknown): Promise<T> {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: {
      ...(body === undefined ? {} : { "Content-Type": "application/json" }),
      "X-Local-Synthetic-Token": token,
    },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const value = await response.json();
  if (!response.ok) {
    const detail = typeof value.detail === "string" ? value.detail : `本地 API 返回 ${response.status}`;
    throw new Error(detail);
  }
  return value as T;
}

function fmt(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toLocaleString("zh-CN")
    : "—";
}

function generationStatus(value: unknown): string {
  if (value === "not_generated_not_trained_not_reevaluated") return "待生成";
  if (value === "generated_not_trained_not_reevaluated")
    return "已生成 · 待训练与复测";
  if (value === "partial_not_trained_not_reevaluated")
    return "部分生成 · 待训练与复测";
  if (typeof value === "string" && value.includes("failed"))
    return `失败 · ${value}`;
  return typeof value === "string" ? value : "状态未知";
}

export function TargetedExperiments({ generationOnly = false, onRegistered }: { generationOnly?: boolean; onRegistered?: () => void } = {}) {
  const routeContext = useRef(new URLSearchParams(window.location.search)).current;
  const incomingScenarioId = routeContext.get("scenario_id") || "";
  const incomingBundleId = routeContext.get("bundle_id") || "";
  const incomingEvalId = routeContext.get("eval_id") || "";
  const incomingModelId = routeContext.get("model_id") || "";
  const incomingDatasetId = routeContext.get("dataset_id") || "";
  const incomingStage = routeContext.get("stage") || "";
  const [overview, setOverview] = useState<Overview | null>(null);
  const [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  const [stageCapabilities, setStageCapabilities] = useState<StageCapability[]>([]);
  const [stageCapabilitiesError, setStageCapabilitiesError] = useState("");
  const [overviewError, setOverviewError] =
    useState<OverviewLoadError | null>(null);
  const [capabilitiesError, setCapabilitiesError] = useState("");
  const [profileId, setProfileId] = useState("");
  const [endpoint, setEndpoint] = useState("");
  const [model, setModel] = useState("");
  const [apiKeyEnv, setApiKeyEnv] = useState("");
  const [scenarioId, setScenarioId] = useState(incomingScenarioId);
  const [prompt, setPrompt] = useState("");
  const [requiredFields, setRequiredFields] = useState(
    "problem_id, solution, final_answer",
  );
  const [maxRequests, setMaxRequests] = useState(10);
  const [maxOutputTokens, setMaxOutputTokens] = useState(1536);
  const [targetCandidates, setTargetCandidates] = useState(10);
  const [stage, setStage] = useState("sft");
  const [notice, setNotice] = useState("");
  const [refreshReports, setRefreshReports] = useState(0);
  const [draftSnapshot, setDraftSnapshot] = useState<{
    scenarioId: string;
    reportId: string;
    signature: string;
  } | null>(null);
  const [sftCaps, setSftCaps] = useState<SftGenerationCapabilities | null>(null);
  const [sftReadiness, setSftReadiness] = useState<SftReadiness | null>(null);
  const [selectedSftBundle, setSelectedSftBundle] = useState("");
  const [selectedStagePlan, setSelectedStagePlan] = useState("");
  const [usableTarget, setUsableTarget] = useState(10);
  const [sftMaxRequests, setSftMaxRequests] = useState(10);
  const [sftCandidateBudget, setSftCandidateBudget] = useState(20);
  const [sftPreflight, setSftPreflight] = useState<SftPreflight | null>(null);
  const [sftRunId, setSftRunId] = useState(() => sessionStorage.getItem("onegpu.sftgen.execution") || "");
  const [sftRun, setSftRun] = useState<SftGenerationRun | null>(null);
  const [sftPreview, setSftPreview] = useState<SftPreview | null>(null);
  const [sftReviewed, setSftReviewed] = useState(false);
  const [sftBusy, setSftBusy] = useState(false);
  const [sftError, setSftError] = useState("");
  const initialized = useRef(false);

  useEffect(() => {
    const controller = new AbortController();
    setOverviewError(null);
    setCapabilitiesError("");
    void loadJsonResource<Overview>(
      "/api/targeted/overview",
      controller.signal,
      setOverview,
      (error) => setOverviewError(classifyOverviewLoadError(error)),
      fetch,
    );
    void loadJsonResource<Capabilities>(
      "/api/targeted/capabilities",
      controller.signal,
      (caps) => {
        setCapabilities(caps);
        if (!initialized.current) {
          setProfileId(caps.profiles[0]?.profile_id ?? "");
          setPrompt(caps.prompt?.template ?? "");
          const remote = caps.profiles.find((item) => item.adapter !== "fake");
          setEndpoint(remote?.endpoint ?? "");
          setModel(remote?.model ?? "");
          setApiKeyEnv(remote?.api_key_env ?? "");
          setRequiredFields(
            caps.protocol?.required_fields.join(", ") ??
              "problem_id, solution, final_answer",
          );
          initialized.current = true;
        }
      },
      (error) => setCapabilitiesError(`示例配置读取失败 (${error.message})`),
      fetch,
    );
    void fetch("/api/targeted/stages", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<{ stages?: StageCapability[] }>;
      })
      .then((payload) => setStageCapabilities(payload.stages ?? []))
      .catch((error) => {
        if (!controller.signal.aborted) {
          setStageCapabilitiesError(error instanceof Error ? error.message : "阶段能力读取失败");
        }
      });
    void fetch("/api/synthetic/sft/capabilities", { signal: controller.signal })
      .then(async (response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<SftGenerationCapabilities>;
      })
      .then((caps) => {
        setSftCaps(caps);
        const matchingBundles = (caps.bundles || []).filter((bundle) => !incomingScenarioId || bundle.scenario_id === incomingScenarioId);
        const requestedBundle = matchingBundles.find((bundle) => bundle.bundle_id === incomingBundleId);
        setSelectedSftBundle((current) => current || requestedBundle?.bundle_id || matchingBundles[0]?.bundle_id || caps.bundles?.[0]?.bundle_id || "");
        const firstStagePlan = caps.bundles?.flatMap((bundle) => bundle.shared_stage_plans || [])[0];
        if (firstStagePlan) setSelectedStagePlan((current) => current || `${firstStagePlan.plan_id}@${firstStagePlan.version}`);
        if (caps.enabled && caps.token) {
          void syntheticRequest<SftReadiness>("/api/synthetic/sft/readiness", caps.token)
            .then(setSftReadiness)
            .catch((error) => {
              if (!controller.signal.aborted) setSftError(error instanceof Error ? error.message : "读取 SFT 准备状态失败");
            });
        } else {
          setSftReadiness(null);
        }
      })
      .catch((error) => {
        if (!controller.signal.aborted) setSftError(error instanceof Error ? error.message : "读取生成能力失败");
      });
    return () => controller.abort();
  }, [refreshReports, incomingBundleId, incomingScenarioId]);

  useEffect(() => {
    if (!sftRunId || !sftCaps?.token) return;
    let stopped = false;
    const refresh = async () => {
      try {
        const state = await syntheticRequest<SftGenerationRun>(`/api/synthetic/sft/runs/${encodeURIComponent(sftRunId)}`, sftCaps.token!);
        if (!stopped) setSftRun(state);
      } catch (error) {
        if (!stopped) setSftError(error instanceof Error ? error.message : "读取生成状态失败");
      }
    };
    void refresh();
    const active = !sftRun || ["queued", "running"].includes(sftRun.status);
    const timer = active ? window.setInterval(() => void refresh(), 1200) : undefined;
    return () => { stopped = true; if (timer) window.clearInterval(timer); };
  }, [sftRunId, sftCaps?.token, sftRun?.status]);

  useEffect(() => {
    const bundle = sftCaps?.bundles?.find((item) => item.bundle_id === selectedSftBundle);
    if (bundle) {
      setSftMaxRequests((value) => Math.min(value, bundle.max_requests));
      setSftCandidateBudget((value) => Math.min(value, bundle.max_candidates));
    }
  }, [selectedSftBundle, sftCaps]);

  const selectedProfile = capabilities?.profiles.find(
    (item) => item.profile_id === profileId,
  );
  const scenarios = overview?.scenarios ?? [];
  const selectedScenario = scenarios.find(
    (item) => item.scenario_id === scenarioId,
  );
  const currentDraftSignature = JSON.stringify({
    profileId,
    endpoint,
    model,
    apiKeyEnv,
    scenarioId,
    prompt,
    requiredFields,
    maxRequests,
    maxOutputTokens,
    targetCandidates,
    stage,
  });
  const draftIsStale = Boolean(
    draftSnapshot &&
    (draftSnapshot.signature !== currentDraftSignature ||
      draftSnapshot.scenarioId !== scenarioId ||
      draftSnapshot.reportId !== (selectedScenario?.report_id ?? "")),
  );
  const pending = useMemo(
    () => ({
      generation: !overview?.generations.length,
      training: !overview?.effects.some((item) => item.after_model_id),
      reevaluation: !overview?.effects.some((item) => item.after_eval_id),
    }),
    [overview],
  );

  const selectedSftPlan = sftCaps?.bundles?.find((item) => item.bundle_id === selectedSftBundle);
  const selectedStagePlanIdentity = selectedSftPlan?.shared_stage_plans?.find(
    (item) => `${item.plan_id}@${item.version}` === selectedStagePlan,
  );
  const sftActive = Boolean(sftRun && ["queued", "running"].includes(sftRun.status));
  const sftTerminal = Boolean(sftRun && ["completed", "partial", "cancelled", "failed", "interrupted"].includes(sftRun.status));

  async function preflightSftGeneration() {
    if (!sftCaps?.token || !selectedSftBundle) return;
    setSftBusy(true); setSftError(""); setSftPreflight(null);
    try {
      setSftPreflight(await syntheticRequest<SftPreflight>("/api/synthetic/sft/preflight", sftCaps.token, {
        bundle_id: selectedSftBundle, usable_target: usableTarget,
        max_requests: sftMaxRequests, target_candidates: sftCandidateBudget,
        ...(selectedStagePlanIdentity ? { stage_plan_id: selectedStagePlanIdentity.plan_id,
          plan_version: selectedStagePlanIdentity.version } : {}),
      }));
    } catch (error) {
      setSftError(error instanceof Error ? error.message : "生成预检失败");
    } finally { setSftBusy(false); }
  }

  async function startSftGeneration() {
    if (!sftCaps?.token || !sftPreflight || !selectedSftPlan) return;
    const outputCost = sftPreflight.estimated_output_cost_upper_bound_usd;
    const costText = outputCost == null
      ? "费用：未知（未配置输出费率；输入 token 和供应商重试费用也不在该估算内）"
      : `仅输出 token 的估算上界约 $${outputCost.toFixed(4)}；不含输入 token 与供应商重试费用`;
    const accepted = window.confirm(
      `即将调用已登记 Provider 生成 SFT 数据。\n\n方案：${sftPreflight.bundle_id}\n模型：${sftPreflight.model || selectedSftPlan.model || "未注明"}\n目标可用样本：${sftPreflight.desired_usable_records}\n候选上限：${sftPreflight.candidate_budget}\n请求上限：${sftPreflight.max_requests}\n单次输出上限：${sftPreflight.max_output_tokens_per_request} tokens\n${costText}\n\n只有明确确认后才会调用 Provider。继续？`,
    );
    if (!accepted) return;
    setSftBusy(true); setSftError("");
    try {
      const state = await syntheticRequest<SftGenerationRun>("/api/synthetic/sft/start", sftCaps.token, {
        bundle_id: selectedSftBundle, usable_target: usableTarget,
        max_requests: sftMaxRequests, target_candidates: sftCandidateBudget,
        ...(selectedStagePlanIdentity ? { stage_plan_id: selectedStagePlanIdentity.plan_id,
          plan_version: selectedStagePlanIdentity.version } : {}),
        confirm_start: true,
      });
      setSftRunId(state.execution_id); setSftRun(state); setSftPreflight(null);
      setSftPreview(null); setSftReviewed(false);
      sessionStorage.setItem("onegpu.sftgen.execution", state.execution_id);
      sessionStorage.setItem("onegpu.sftgen.context", JSON.stringify({
        scenario_id: scenarioId,
        bundle_id: selectedSftBundle,
        eval_id: selectedScenario?.eval_id || incomingEvalId,
        model_id: selectedScenario?.model_id || incomingModelId,
        dataset_id: selectedScenario?.dataset_id || incomingDatasetId,
      }));
    } catch (error) {
      setSftError(error instanceof Error ? error.message : "生成启动失败");
    } finally { setSftBusy(false); }
  }

  async function cancelSftGeneration() {
    if (!sftCaps?.token || !sftRunId) return;
    setSftBusy(true); setSftError("");
    try {
      const state = await syntheticRequest<SftGenerationRun>(`/api/synthetic/sft/runs/${encodeURIComponent(sftRunId)}/cancel`, sftCaps.token, {});
      setSftRun(state);
    } catch (error) { setSftError(error instanceof Error ? error.message : "取消请求失败"); }
    finally { setSftBusy(false); }
  }

  async function previewSftGeneration() {
    if (!sftCaps?.token || !sftRunId) return;
    setSftBusy(true); setSftError("");
    try {
      setSftPreview(await syntheticRequest<SftPreview>(`/api/synthetic/sft/runs/${encodeURIComponent(sftRunId)}/preview?limit=3`, sftCaps.token));
    } catch (error) { setSftError(error instanceof Error ? error.message : "生成样本预览失败"); }
    finally { setSftBusy(false); }
  }

  async function registerSftGeneration() {
    if (!sftCaps?.token || !sftRunId || !sftRun?.result || !sftReviewed) return;
    if (!window.confirm(`请确认你已查看生成结果、样本与质量报告，并接受“结构通过不代表语义正确”。\n\n将登记 ${fmt(sftRun.result.usable_records)} 条 SFT 数据，许可信息可能未知。确认登记？`)) return;
    setSftBusy(true); setSftError("");
    try {
      const state = await syntheticRequest<SftGenerationRun>(`/api/synthetic/sft/runs/${encodeURIComponent(sftRunId)}/register`, sftCaps.token, { confirm_reviewed: true });
      setSftRun(state);
      onRegistered?.();
      setNotice(`数据集已登记：${state.registered_dataset_id || state.registration?.dataset_id}。现可在 Launcher 中选择该 dataset ID。`);
    } catch (error) { setSftError(error instanceof Error ? error.message : "数据集登记失败"); }
    finally { setSftBusy(false); }
  }

  function openSftDatasetInLauncher() {
    const datasetId = sftRun?.registered_dataset_id || sftRun?.registration?.dataset_id;
    if (typeof datasetId === "string") window.location.href = `/?page=launcher&dataset_id=${encodeURIComponent(datasetId)}`;
  }

  function continueTargetedGeneration() {
    const params = new URLSearchParams({ page: "workbench", targeted_generation: "1" });
    if (scenarioId) params.set("scenario_id", scenarioId);
    if (selectedSftBundle) params.set("bundle_id", selectedSftBundle);
    if (selectedScenario?.eval_id || incomingEvalId) params.set("eval_id", selectedScenario?.eval_id || incomingEvalId);
    if (selectedScenario?.model_id || incomingModelId) params.set("model_id", selectedScenario?.model_id || incomingModelId);
    if (selectedScenario?.dataset_id || incomingDatasetId) params.set("dataset_id", selectedScenario?.dataset_id || incomingDatasetId);
    if (incomingStage) params.set("stage", incomingStage);
    params.set("return_to", "targeted");
    window.location.href = `/?${params.toString()}`;
  }

  function returnToDiagnostics() {
    const params = new URLSearchParams({ page: "targeted" });
    if (scenarioId) params.set("scenario_id", scenarioId);
    if (incomingEvalId) params.set("eval_id", incomingEvalId);
    if (incomingModelId) params.set("model_id", incomingModelId);
    if (incomingDatasetId) params.set("dataset_id", incomingDatasetId);
    if (incomingStage) params.set("stage", incomingStage);
    window.location.href = `/?${params.toString()}`;
  }

  function sftStatusLabel(status?: string) {
    return ({ queued: "排队中", running: "生成中", completed: "完成", partial: "部分完成",
      cancelled: "已取消", failed: "失败", interrupted: "服务重启后中断" } as Record<string, string>)[status || ""] || "未启动";
  }

  function exportDraft() {
    if (!selectedProfile || !prompt.trim()) {
      setNotice("先选择 provider profile 并填写提示词模板。");
      return;
    }
    if (selectedProfile.adapter !== "fake") {
      try {
        const parsed = new URL(endpoint);
        if (
          !/^https?:$/.test(parsed.protocol) ||
          parsed.username ||
          parsed.password ||
          [...parsed.searchParams.keys()].some((key) =>
            /key|token|secret|auth|password/i.test(key),
          )
        )
          throw new Error();
      } catch {
        setNotice(
          "Endpoint 必须是有效 HTTP(S) URL，且不能包含认证信息或敏感查询参数。",
        );
        return;
      }
      if (!model.trim() || !/^[A-Za-z_][A-Za-z0-9_]*$/.test(apiKeyEnv.trim())) {
        setNotice("请填写 model 和合法的凭据环境变量名；页面不接受密钥值。");
        return;
      }
    }
    const fields = requiredFields
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean);
    if (fields.length === 0 || new Set(fields).size !== fields.length) {
      setNotice("输出字段不能为空或重复。");
      return;
    }
    setDraftSnapshot({
      scenarioId,
      reportId: selectedScenario?.report_id ?? "",
      signature: currentDraftSignature,
    });
    const draft = {
      schema: "onegpu-targeted-synthetic-workbench-draft-v1",
      exported_at: new Date().toISOString(),
      status: "draft_not_registered",
      target: selectedScenario
        ? {
            scenario_id: selectedScenario.scenario_id,
            diagnosis_status: selectedScenario.status ?? "unknown",
            denominator: selectedScenario.denominator ?? null,
          }
        : null,
      stage,
      provider_profile:
        selectedProfile.adapter === "fake"
          ? selectedProfile
          : {
              ...selectedProfile,
              endpoint: endpoint.trim(),
              model: model.trim(),
              api_key_env: apiKeyEnv.trim(),
            },
      prompt: {
        template_id: capabilities?.prompt?.template_id ?? "custom-prompt",
        version: capabilities?.prompt?.version ?? "1",
        template: prompt,
        allowed_variables: capabilities?.prompt?.allowed_variables ?? [
          "question",
          "topic",
        ],
      },
      output_protocol: capabilities?.protocol
        ? { ...capabilities.protocol, required_fields: fields }
        : null,
      budget: {
        max_requests: maxRequests,
        max_output_tokens: maxOutputTokens,
        target_candidates: targetCandidates,
        cost_status: "unknown_until_provider_usage_or_user_declared_rates",
      },
      execution: "not_generated_not_trained_not_reevaluated",
      credential_policy:
        "Only api_key_env variable name is included; secret values are never requested or exported.",
    };
    const blob = new Blob([JSON.stringify(draft, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "targeted-synthetic-workbench-draft.json";
    anchor.click();
    URL.revokeObjectURL(url);
    setNotice(
      "已导出本地配置草稿；这不是已登记计划，也没有发起生成请求。请按页面下方 CLI 指引完成正式登记。",
    );
  }

  return (
    <div className={`targeted-page${generationOnly ? " targeted-generation-only" : ""}`}>
      <section className="targeted-hero">
        <div>
          <div className="eyebrow">
            TARGETED DATA <span>／</span> 诊断驱动补数
          </div>
          <h1>
            从薄弱场景到可复查的复测<span>。</span>
          </h1>
          <p>
            先看证据和分母，再准备数据计划；生成、训练与复测状态只按实际产物显示。
          </p>
        </div>
        <span className="targeted-readonly">
          <span /> 仅显式启动本地生成
        </span>
      </section>

      {overviewError?.kind === "error" && (
        <div className="targeted-alert" role="alert">
          报告读取失败（{overviewError.message}）。场景与运行报告暂不可用；配置区仍可使用。
          <button
            className="targeted-retry"
            onClick={() => setRefreshReports((n) => n + 1)}
          >
            重试
          </button>
        </div>
      )}
      {capabilitiesError && (
        <div className="targeted-alert" role="alert">
          {capabilitiesError}。配置模板未加载；仍可手动编辑配置。
        </div>
      )}
      {notice && (
        <div className="targeted-alert targeted-success" role="status">
          {notice}
          <button onClick={() => setNotice("")} aria-label="关闭提示">
            ×
          </button>
        </div>
      )}

      <section className="targeted-flow" aria-label="定向补数流程状态">
        {[
          [
            "01",
            "发现薄弱场景",
            "查看分母与证据",
            scenarios.length > 0 ? "有诊断" : "待诊断",
          ],
          [
            "02",
            "准备生成计划",
            "配置模板与预算",
            overview?.plans.length ? "已有计划" : "草稿",
          ],
          [
            "03",
            "生成与验证",
            "查看候选与成本",
            pending.generation
              ? "待生成"
              : overview?.generations.some((item) =>
                    String(item.execution_status).startsWith("partial"),
                  )
                ? "部分完成"
                : overview?.generations.some((item) =>
                      String(item.execution_status).includes("failed"),
                    )
                  ? "失败"
                  : "已生成",
          ],
          [
            "04",
            "训练与复测",
            "看目标及保护指标",
            pending.training
              ? "待训练"
              : pending.reevaluation
                ? "待复测"
                : "有复测",
          ],
        ].map(([number, title, subtitle, status]) => (
          <div className="targeted-flow-step" key={number}>
            <span className="targeted-step-number">{number}</span>
            <div>
              <strong>{title}</strong>
              <small>{subtitle}</small>
            </div>
            <span className="targeted-step-status">{status}</span>
          </div>
        ))}
      </section>

      <section className="targeted-stage-capabilities" aria-label="各阶段闭环能力">
        {stageCapabilitiesError ? (
          <p role="status">阶段能力暂不可用：{stageCapabilitiesError}。现有页面功能不受影响。</p>
        ) : stageCapabilities.length === 0 ? (
          <p role="status">正在读取 CPT、SFT、Preference 与 GRPO 的本地能力状态…</p>
        ) : stageCapabilities.map((item) => (
          <article className="targeted-stage-capability" key={item.stage}>
            <div className="targeted-stage-capability-head">
              <strong>{({ cpt: "CPT", sft: "SFT", preference: "Preference", grpo: "GRPO" } as const)[item.stage]}</strong>
              <span className={item.ready_for_execution ? "ready" : "blocked"}>
                {item.ready_for_execution ? "训练条件已就绪" : "训练条件有阻塞"}
              </span>
            </div>
            <small>数据格式 {item.dataset_schema} · 父模型阶段 {item.parent_model_stage}</small>
            <small>生成 UI {item.generation_ui_available ? "已接通" : "未接通"} · 训练 UI {item.training_ui_available ? "已接通" : "未接通"} · 评测 {item.evaluation.available ? "可用" : "未就绪"}</small>
            {(item.training_blockers || item.blockers).slice(0, 2).map((blocker) => <p key={blocker}>{blocker}</p>)}
            {item.generation_blockers?.length ? <p>数据准备提示：{item.generation_blockers[0]}</p> : null}
            {item.generation_modes && <details>
              <summary>查看普通生成与定向生成条件</summary>
              {(["ordinary", "targeted"] as const).map((mode) => {
                const capability = item.generation_modes![mode];
                return <div key={mode}>
                  <strong>{mode === "ordinary" ? "普通生成" : "定向生成"}：{capability.status === "blocked" ? "暂不可用" : "需完成具体预检"}</strong>
                  {capability.blockers.map((reason) => <p key={reason}>{reason}</p>)}
                  {capability.pending_checks.map((reason) => <p key={reason}>{reason}</p>)}
                </div>;
              })}
              <small>此处仅检查本地能力，不调用模型，也不代表已经获得启动许可。</small>
            </details>}
            {item.training_ui_available && <a className="button secondary targeted-stage-launch-link"
              href={`/?page=launcher&stage=${item.stage === "preference" ? "dpo" : item.stage}`}>
              前往{item.stage === "preference" ? "DPO / SimPO" : item.stage.toUpperCase()}训练与评测 →
            </a>}
          </article>
        ))}
      </section>

      <div className="targeted-grid">
        <section className="panel targeted-panel">
          <div className="targeted-section-head">
            <div>
              <span className="eyebrow">DIAGNOSIS</span>
              <h2>薄弱场景</h2>
            </div>
            <div className="targeted-header-actions">
              {draftSnapshot && (
                <span
                  className={`targeted-draft-state ${draftIsStale ? "stale" : "current"}`}
                >
                  {draftIsStale ? "草稿已过期" : "草稿与当前配置一致"}
                </span>
              )}
              <button
                className="targeted-refresh"
                onClick={() => setRefreshReports((n) => n + 1)}
              >
                刷新报告
              </button>
              <span className="targeted-count">{scenarios.length} 个切片</span>
            </div>
          </div>
          {!overview?.available || scenarios.length === 0 ? (
            <div className="targeted-empty">
              <strong>
                {overviewError?.kind === "missing"
                  ? "尚未生成场景诊断报告"
                  : "尚无可展示的场景诊断"}
              </strong>
              <p>
                {overviewError?.kind === "missing"
                  ? "本地还没有场景报告。请先用逐题 predictions 与场景标签生成报告；不会从总分推算场景准确率。"
                  : "不会从总分推算场景准确率。请先用逐题 predictions 与场景标签生成 scenario report。"}
              </p>
              <code>onegpu-posttrain build-scenario-report --help</code>
            </div>
          ) : (
            <div className="targeted-scenario-list">
              {scenarios.map((item, i) => (
                <button
                  key={`${item.scenario_id}-${i}`}
                  className={`targeted-scenario ${scenarioId === item.scenario_id ? "selected" : ""}`}
                  onClick={() => setScenarioId(item.scenario_id)}
                  aria-pressed={scenarioId === item.scenario_id}
                >
                  <span className="targeted-scenario-name">
                    {item.scenario_id}
                  </span>
                  <span className="targeted-scenario-metric">
                    {item.accuracy == null
                      ? "未评估"
                      : `${(item.accuracy * 100).toFixed(1)}%`}
                  </span>
                  <small>
                    正确 {fmt(item.correct)} / 可评分分母{" "}
                    {fmt(item.denominator)} · 样本 {fmt(item.sample_count)}
                  </small>
                  <span
                    className={`targeted-evidence ${item.status === "available" ? "evidence-ok" : ""}`}
                  >
                    {item.status === "available"
                      ? "证据可用"
                      : item.status === "insufficient_evidence"
                        ? "样本不足"
                        : "缺少有效证据"}
                  </span>
                </button>
              ))}
            </div>
          )}
          {selectedScenario && (
            <p className="targeted-footnote">
              解析失败 {fmt(selectedScenario.parse_failed)} · 不支持{" "}
              {fmt(selectedScenario.unsupported)} · 缺预测{" "}
              {fmt(selectedScenario.missing_prediction)}
              。分母只计入可评分的正确与错误样本。
              <br />
              来源：eval {selectedScenario.eval_id || "—"} · model{" "}
              {selectedScenario.model_id || "—"} · dataset{" "}
              {selectedScenario.dataset_id || "—"} · protocol{" "}
              {selectedScenario.protocol_version || "—"}
            </p>
          )}
        </section>

        <section className="panel targeted-panel targeted-config">
          <div className="targeted-section-head">
            <div>
              <span className="eyebrow">PLAN WORKBENCH</span>
              <h2>准备配置草稿</h2>
            </div>
            <span className="targeted-local-badge">仅本地编辑</span>
          </div>
          <div className="targeted-form-grid">
            <label>
              生成阶段
              <select value={stage} onChange={(e) => setStage(e.target.value)}>
                <option value="sft">SFT 候选</option>
                <option value="preference">Preference 候选</option>
              </select>
              <small>Preference 仍需同题且可验证的候选配对。</small>
            </label>
            <label>
              目标场景
              <select
                value={scenarioId}
                onChange={(e) => setScenarioId(e.target.value)}
              >
                <option value="">暂不选择</option>
                {scenarios.map((s, i) => (
                  <option key={`${s.scenario_id}-${i}`} value={s.scenario_id}>
                    {s.scenario_id} · {s.status ?? "unknown"}
                  </option>
                ))}
              </select>
              <small>最终 holdout 不能作为生成输入。</small>
            </label>
            <label className="targeted-wide">
              Provider profile
              <select
                value={profileId}
                onChange={(e) => setProfileId(e.target.value)}
              >
                {(capabilities?.profiles ?? []).map((p) => (
                  <option value={p.profile_id} key={p.profile_id}>
                    {p.profile_id} · {p.adapter}
                  </option>
                ))}
              </select>
              <small>
                {selectedProfile?.adapter === "fake"
                  ? "本地 fixture，不代表真实模型输出。"
                  : "执行时只从指定环境变量读取密钥值；页面绝不读取、接收或保存密钥。"}
              </small>
            </label>
            {selectedProfile?.adapter !== "fake" && (
              <>
                <label className="targeted-wide">
                  Endpoint
                  <input
                    value={endpoint}
                    onChange={(e) => setEndpoint(e.target.value)}
                    autoComplete="off"
                  />
                  <small>
                    仅接受 HTTP(S) URL；不要内嵌用户名、密码或 token 查询参数。
                  </small>
                </label>
                <label>
                  模型名称
                  <input
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    autoComplete="off"
                  />
                </label>
                <label>
                  凭据环境变量名
                  <input
                    value={apiKeyEnv}
                    onChange={(e) => setApiKeyEnv(e.target.value)}
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <small>这里只是变量名，不是密钥输入框。</small>
                </label>
              </>
            )}
            <label>
              最大请求数
              <input
                type="number"
                min={1}
                max={1000}
                value={maxRequests}
                onChange={(e) =>
                  setMaxRequests(
                    Math.min(1000, Math.max(1, Number(e.target.value) || 1)),
                  )
                }
              />
            </label>
            <label>
              每次输出上限（tokens）
              <input
                type="number"
                min={1}
                max={8192}
                value={maxOutputTokens}
                onChange={(e) =>
                  setMaxOutputTokens(
                    Math.min(8192, Math.max(1, Number(e.target.value) || 1)),
                  )
                }
              />
            </label>
            <label>
              目标候选数
              <input
                type="number"
                min={1}
                max={1000}
                value={targetCandidates}
                onChange={(e) =>
                  setTargetCandidates(
                    Math.min(1000, Math.max(1, Number(e.target.value) || 1)),
                  )
                }
              />
            </label>
            <label className="targeted-wide">
              提示词模板
              <textarea
                rows={7}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                aria-describedby="targeted-prompt-help"
              />
              <small id="targeted-prompt-help">
                可用变量：
                {capabilities?.prompt?.allowed_variables?.join(", ") ||
                  "question, topic"}
                。避免把诊断题、最终评测题或答案放进模板。
              </small>
            </label>
            <label className="targeted-wide">
              输出必需字段
              <input
                value={requiredFields}
                onChange={(e) => setRequiredFields(e.target.value)}
                aria-label="输出协议必需字段，逗号分隔"
              />
              <small>
                逗号分隔；导出配置会保留 JSON object 格式和这些必需字段。
              </small>
            </label>
          </div>
          <div className="targeted-budget-note">
            <strong>费用：未知</strong>
            <span>
              没有供应商 usage 或声明费率时显示 unknown，不按 0
              处理；请求数是硬边界，金额不是硬上限。
            </span>
          </div>
          <div className="targeted-actions">
            <button className="button primary" onClick={exportDraft}>
              导出配置草稿
            </button>
            <span>不会保存到服务端、检查环境变量或发起网络请求。</span>
          </div>
        </section>
      </div>

      <section className="panel targeted-panel targeted-sft-generation" aria-labelledby="sft-generation-title">
        <div className="targeted-section-head">
          <div>
            <span className="eyebrow">SFT GENERATION WORKFLOW</span>
          <h2 id="sft-generation-title">{generationOnly ? "定向生成、审核并登记" : "定向数据生成入口"}</h2>
          </div>
        {generationOnly ? <button className="secondary-button" onClick={returnToDiagnostics}>返回薄弱场景</button> : <span className="targeted-local-badge">数据模块 · 仅本机执行</span>}
        </div>
      {generationOnly ? <>
        {(incomingScenarioId || incomingEvalId || incomingModelId) && <div className="targeted-source-context" role="note">
          <strong>定向来源上下文</strong>
          <span>阶段 {incomingStage || "未记录"} · 场景 {incomingScenarioId || "—"} · 来源评测 {incomingEvalId || "—"} · 模型 {incomingModelId || "未记录"} · 数据 {incomingDatasetId || "未记录"}</span>
          <small>这些身份用于追溯；生成器只读取登记方案允许的训练 seed，不会读取诊断题或最终 holdout。</small>
        </div>}
        {!sftCaps?.enabled ? (
          <div className="targeted-empty">
            <strong>模型生成尚未启用</strong>
            <p>{sftCaps?.reason || "需在本地 UI 启动命令中单独加上 --enable-synthetic-generation。浏览和配置不会调用 Provider。"}</p>
          </div>
        ) : (sftCaps.bundles || []).length === 0 ? (
          <div className="targeted-empty">
            <strong>先检查本地 SFT 数据与证据</strong>
            <p>准备状态只读取 manifest、模板/协议、Provider profile 和逐题诊断证据；不会调用模型或读取密钥值。</p>
            {sftReadiness && <>
              <p>发现 {fmt(sftReadiness.summary.dataset_count)} 个 SFT 数据集，{fmt(sftReadiness.summary.ready_count)} 个已具备生成准备条件；逐题诊断报告 {fmt(sftReadiness.summary.scenario_report_count)} 份。</p>
              <div className="targeted-readiness-list">
                {sftReadiness.readiness.slice(0, 10).map((item) => <div className="targeted-report-row" key={item.dataset_id}>
                  <strong>{item.dataset_id} · {fmt(sftReadiness.datasets.find((dataset) => dataset.dataset_id === item.dataset_id)?.sample_count)} 条 · {item.eligible ? "可用于方案准备" : "尚缺准备证据"}</strong>
                  {!item.eligible && <>
                    <small>{item.missing_reasons.join("；")}</small>
                    {item.missing_reasons.some((reason) => reason.includes("诊断报告")) && <small>下一步：准备匹配的逐题评测预测与场景标签，再运行 <code>onegpu-posttrain build-scenario-report --help</code>；系统不会用总体分数猜场景。</small>}
                    {item.missing_reasons.some((reason) => reason.includes("角色登记")) && <small>下一步：在独立数据上登记 train_seed、diagnostic_dev、final_holdout 角色；holdout 不得兼作训练数据。</small>}
                  </>}
                </div>)}
              </div>
              <div className="targeted-actions">
                <a className="button secondary" href="/?page=data">查看已登记数据</a>
                <span>完整操作说明：docs/guides/SFT_GENERATION_WORKFLOW_zh.md</span>
              </div>
              {sftReadiness.profiles.some((profile) => profile.is_example) && <small>当前发现的 Provider 中包含示例配置；示例 profile 仅用于离线演示，不可用于真实生成。</small>}
            </>}
            {!sftReadiness && <p>正在读取本机数据与证据清单…</p>}
          </div>
        ) : (
          <>
            <div className="targeted-form-grid">
              <label className="targeted-wide">
                已登记的 SFT 生成方案
                <select value={selectedSftBundle} disabled={sftBusy || sftActive} onChange={(event) => { setSelectedSftBundle(event.target.value); setSftPreflight(null); setSelectedStagePlan(""); }}>
                  {(sftCaps.bundles || []).map((bundle) => <option key={bundle.bundle_id} value={bundle.bundle_id}>
                    {bundle.scenario_id || "SFT"} · {bundle.model || bundle.provider_profile_id} · {bundle.bundle_id}
                  </option>)}
                </select>
                <small>种子数据：{selectedSftPlan?.source_dataset_id || "—"}；Provider 和提示模板来自不可变登记方案。</small>
              </label>
              {Boolean(selectedSftPlan?.shared_stage_plans?.length) && <label className="targeted-wide">
                统一定向计划（可选）
                <select value={selectedStagePlan} disabled={sftBusy || sftActive} onChange={(event) => { setSelectedStagePlan(event.target.value); setSftPreflight(null); }}>
                  <option value="">仅使用 SFT 执行方案</option>
                  {selectedSftPlan?.shared_stage_plans?.map((plan) => <option key={`${plan.plan_id}@${plan.version}`} value={`${plan.plan_id}@${plan.version}`}>
                    {plan.plan_id} · v{plan.version} · {plan.scenario_id}
                  </option>)}
                </select>
                <small>选择后，预检会核对 scenario、train seed、诊断集、holdout、provider 和预算；登记数据将保留该统一 plan ID。</small>
              </label>}
              <label>
                目标可用样本数
                <input type="number" min={1} max={1000} value={usableTarget} disabled={sftActive}
                  onChange={(event) => { setUsableTarget(Math.min(1000, Math.max(1, Number(event.target.value) || 1))); setSftPreflight(null); }} />
                <small>这是产出目标，不是质量 gate；未达目标仍可审核部分结果。</small>
              </label>
              <label>
                候选生成上限
                <input type="number" min={1} max={selectedSftPlan?.max_candidates || 1000} value={sftCandidateBudget}
                  disabled={sftActive} onChange={(event) => { setSftCandidateBudget(Math.min(selectedSftPlan?.max_candidates || 1000, Math.max(1, Number(event.target.value) || 1))); setSftPreflight(null); }} />
                <small>登记方案上限：{fmt(selectedSftPlan?.max_candidates)}。</small>
              </label>
              <label>
                Provider 请求上限
                <input type="number" min={1} max={selectedSftPlan?.max_requests || 1000} value={sftMaxRequests}
                  disabled={sftActive} onChange={(event) => { setSftMaxRequests(Math.min(selectedSftPlan?.max_requests || 1000, Math.max(1, Number(event.target.value) || 1))); setSftPreflight(null); }} />
                <small>登记方案上限：{fmt(selectedSftPlan?.max_requests)}。</small>
              </label>
            </div>
            <div className="targeted-actions">
              <button className="button secondary" disabled={sftBusy || sftActive || !selectedSftPlan} onClick={preflightSftGeneration}>
                {sftBusy ? "处理中…" : "预检（不调用模型）"}
              </button>
              {sftPreflight && <button className="button primary" disabled={sftBusy || sftActive} onClick={startSftGeneration}>确认费用与预算并生成</button>}
              {sftActive && <button className="button secondary" disabled={sftBusy} onClick={cancelSftGeneration}>请求停止</button>}
              <span>编辑配置、预检与查看页面均为 0 Provider calls；只有确认启动后才调用模型。</span>
            </div>
            {sftError && <div className="targeted-alert" role="alert">{sftError}</div>}
            {sftPreflight && <div className="targeted-report-row">
              <strong>预检通过 · {sftPreflight.bundle_id}</strong>
              <span>模型 {sftPreflight.model || selectedSftPlan?.model || "未注明"} · seed dataset {sftPreflight.source_dataset_id}</span>
              <span>目标可用 {fmt(sftPreflight.desired_usable_records)} · 候选上限 {fmt(sftPreflight.candidate_budget)} · 请求上限 {fmt(sftPreflight.max_requests)} · 单次输出 ≤ {fmt(sftPreflight.max_output_tokens_per_request)} tokens</span>
              <small>{sftPreflight.cost_status === "output_only_upper_bound" ? `仅输出 token 估算上界 $${Number(sftPreflight.estimated_output_cost_upper_bound_usd).toFixed(4)}；不含输入 token 和重试。` : "费用未知；开始前会明确提示预算边界。"} 预检 Provider calls = {sftPreflight.provider_calls}，network = {String(sftPreflight.network_used)}。</small>
            </div>}
            {sftRun && <div className="targeted-run-status" aria-live="polite">
              <div className="targeted-section-head">
                <div><span className="eyebrow">PERSISTED EXECUTION</span><h3>{sftStatusLabel(sftRun.status)}</h3></div>
                <code>{sftRun.execution_id}</code>
              </div>
              {sftRun.detail && <p role="status">{sftRun.detail}</p>}
              {sftRun.progress && <p>实时进度：已完成候选 {fmt(sftRun.progress.completed_candidates)} / {fmt(sftRun.progress.candidate_target)} · 请求 {fmt(sftRun.progress.attempted_requests)} / {fmt(sftRun.progress.request_budget)}</p>}
              {sftRun.result && <div className="targeted-report-row">
                <strong>实际结果：可用 {fmt(sftRun.result.usable_records)} / 目标 {fmt(sftRun.result.desired_usable_records)}（短缺 {fmt(sftRun.result.shortfall)}）</strong>
                <span>候选 {fmt(sftRun.result.candidate_count)} · 请求 {fmt(sftRun.result.request_count)} / {fmt(sftRun.result.max_requests)} · Provider calls {fmt(sftRun.result.provider_calls)}</span>
                <span>格式有效 {fmt(sftRun.result.valid_format_count)} · 格式无效 {fmt(sftRun.result.invalid_format_count)} · 截断 {fmt(sftRun.result.truncated_count)} · 超长 {fmt(sftRun.result.over_length_count)}</span>
                <small>停止原因：{String(sftRun.result.stop_reason || "未记录")}。质量仅报告，不代表语义正确或正式验收。</small>
              </div>}
              <GenerationReviewPanel
                label="定向 SFT"
                canReview={Boolean(sftTerminal && sftRun.result && Number(sftRun.result.usable_records || 0) > 0)}
                busy={sftBusy}
                preview={sftPreview ? { records: sftPreview.records, count: sftPreview.count, canonical_sha256: sftPreview.canonical_sha256, truncated: sftPreview.truncated } : null}
                reviewed={sftReviewed}
                registeredDatasetId={sftRun.registered_dataset_id || sftRun.registration?.dataset_id}
                registrationSummary={sftRun.registration ? `${fmt(sftRun.registration.count)} 条 · 摘要 ${String(sftRun.registration.canonical_sha256 || "—")}` : undefined}
                onPreview={previewSftGeneration}
                onReviewedChange={setSftReviewed}
                onRegister={registerSftGeneration}
              />
              {(sftRun.registered_dataset_id || sftRun.registration?.dataset_id) && <div className="targeted-actions">
                <button className="button primary" onClick={openSftDatasetInLauncher}>在 Mini SFT Launcher 中选择</button>
                {generationOnly && <button className="secondary-button" onClick={returnToDiagnostics}>返回原薄弱场景诊断</button>}
              </div>}
            </div>}
          </>
        )}
      </> : <div className="targeted-generation-handoff">
        <p>诊断、场景选择和不可变计划仍保留在本页；模型调用、候选审核和数据集登记在“数据”模块完成，来源评测和场景身份会随链接带入。</p>
        {selectedScenario && <small>目标场景：{selectedScenario.scenario_id} · 来源评测 {selectedScenario.eval_id || "未记录"} · 模型 {selectedScenario.model_id || "未记录"}</small>}
        <button className="button primary" disabled={!selectedSftPlan} onClick={continueTargetedGeneration}>在数据模块继续生成与审核 →</button>
        {!selectedSftPlan && <small>当前没有匹配的已登记定向 SFT 方案；先在诊断与计划区完成准备。</small>}
      </div>}
      </section>

      {overview?.plans.length ? (
        <section className="panel targeted-plan-list">
          <div className="targeted-section-head">
            <div>
              <span className="eyebrow">REGISTERED PLANS</span>
              <h2>已登记计划</h2>
            </div>
            <span className="targeted-count">不可变历史</span>
          </div>
          {overview.plans.map((plan, i) => (
            <div className="targeted-report-row" key={i}>
              <strong>
                {plan.plan_id || "plan identity unavailable"} · v
                {fmt(plan.version)}
              </strong>
              <span>
                场景 {plan.target_scenario_id || "—"} · 阶段 {plan.stage || "—"}{" "}
                · profile {plan.provider_profile_id || "—"}
              </span>
              <span>
                设计 {plan.comparison_design || "—"} · 候选目标{" "}
                {fmt(plan.budget?.target_candidates)}
              </span>
              <small>
                起点模型 {plan.starting_model?.model_id || "—"} · 训练种子{" "}
                {plan.training_seed?.dataset_id || "—"} · 诊断 eval{" "}
                {plan.diagnostic?.eval_id || "—"} · 最终留出{" "}
                {plan.final_evaluation?.dataset_id || "—"}
                。编辑计划应创建新版本，不覆盖此登记。
              </small>
            </div>
          ))}
        </section>
      ) : null}

      <div className="targeted-grid targeted-evidence-grid">
        <section className="panel targeted-panel">
          <div className="targeted-section-head">
            <div>
              <span className="eyebrow">GENERATION & VALIDATION</span>
              <h2>生成与验证产出</h2>
            </div>
            <span className="targeted-step-status">
              {pending.generation ? "待生成" : "已有产物"}
            </span>
          </div>
          {!overview?.generations.length ? (
            <div className="targeted-empty">
              <p>当前没有公开的生成验证报告。草稿导出不会被计作已生成样本。</p>
              <div className="targeted-stat-row">
                <span>候选：待生成</span>
                <span>验证：待生成</span>
                <span>费用：未知</span>
              </div>
            </div>
          ) : (
            overview.generations.map((g, i) => (
              <div className="targeted-report-row" key={i}>
                <strong>
                  {g.provider_adapter === "fake"
                    ? "本地模拟 · "
                    : "远程来源 · "}
                  {generationStatus(g.execution_status)}
                </strong>
                <span>
                  请求 {fmt(g.request_count)} · 候选{" "}
                  {fmt(g.successful_candidates)} / 目标{" "}
                  {fmt(g.target_candidate_count)}
                </span>
                <span>
                  费用{" "}
                  {g.cost_status === "estimate"
                    ? `估算 $${fmt(g.estimated_cost_usd)}`
                    : g.cost_status === "known"
                      ? `$${fmt(g.estimated_cost_usd)}`
                      : "未知"}
                </span>
                <small>
                  {g.stop_reason || "停止原因未记录"} · provider{" "}
                  {g.provider_profile_id || "—"} · model {g.model || "—"}
                </small>
              </div>
            ))
          )}
        </section>
        <section className="panel targeted-panel">
          <div className="targeted-section-head">
            <div>
              <span className="eyebrow">MODEL EFFECT</span>
              <h2>训练后复测</h2>
            </div>
            <span className="targeted-step-status">
              {pending.reevaluation ? "待复测" : "有结果"}
            </span>
          </div>
          {!overview?.effects.length ? (
            <div className="targeted-empty">
              <p>
                尚无复测报告；不会把生成数量、训练 loss 或 reward 当作能力提升。
              </p>
              <div className="targeted-pending-list">
                <span>追加训练：待训练</span>
                <span>目标场景：待复测</span>
                <span>保护指标：待复测</span>
              </div>
            </div>
          ) : (
            overview.effects.map((effect, i) => (
              <div className="targeted-report-row" key={i}>
                <strong>{effect.status || "状态未知"}</strong>
                {effect.target_state && <span>目标判定：{effect.target_state}</span>}
                <span>
                  {effect.before_model_id || "起点模型未知"} →{" "}
                  {effect.after_model_id || "待训练"}
                </span>
                {effect.stage === "grpo" ? <>
                  <span>GRPO 训练 Run：{effect.training_run_id || "未记录"} · 评测：{effect.before_eval_id || "无匹配起点"} → {effect.after_eval_id || "未评测"}</span>
                  {effect.target_metric ? <span>
                    {effect.target_metric.name}：{effect.target_metric.before?.status === "measured" ? fmt(effect.target_metric.before.value) : "起点未测"}
                    {" → "}{effect.target_metric.after?.status === "measured" ? fmt(effect.target_metric.after.value) : "训练后未测"}
                    {effect.target_metric.delta != null ? ` · 变化 ${fmt(effect.target_metric.delta)}` : ""}
                    {effect.target_metric.threshold_status ? ` · 阈值 ${effect.target_metric.threshold_status}` : ""}
                  </span> : <span>目标指标：未配置或当前评测证据不可比较，不推断为 0 或达标。</span>}
                  {Object.values(effect.guardrail_metrics || {}).map((guardrail: any) => <span key={guardrail.name}>
                    保护指标 {guardrail.name}：{guardrail.after?.status === "measured" ? fmt(guardrail.after.value) : "未测"} · {guardrail.threshold_status || "未判定"}
                  </span>)}
                </> : <span>
                  场景 {effect.scenario_id || "—"} · 新增数据{" "}
                  {fmt(effect.actual_added_records)}
                </span>}
                <small>
                  {Array.isArray(effect.comparison_limitations)
                    ? effect.comparison_limitations.join("；")
                    : "需同时核对保护指标和评测身份"}
                </small>
              </div>
            ))
          )}
        </section>
      </div>

      <section className="targeted-cli panel">
        <div>
          <span className="eyebrow">NEXT STEP</span>
          <h2>从草稿到真实实验</h2>
          <p>
            导出文件只是 UI
            草稿，不替代带数据/模型身份的不可变登记计划。请根据当前配置补齐隔离的
            train_seed、角色注册、污染索引、训练预算和 final_holdout，再显式执行
            CLI。
          </p>
        </div>
        <ol>
          <li>生成逐题场景报告并核对分母、阈值来源和样本量。</li>
          <li>
            用 <code>onegpu-posttrain register-targeted-plan --help</code>{" "}
            登记完整计划；<code>dry-run-targeted-generation</code> 应显示 0
            provider calls。
          </li>
          <li>
            确认预算与凭据后，再由终端显式执行生成；之后单独训练并用固定协议复测。
          </li>
        </ol>
      </section>
      {overview?.truncated && (
        <p className="targeted-footnote">
          报告扫描达到上限，当前列表可能不完整。
        </p>
      )}
    </div>
  );
}
