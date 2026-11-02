import { useEffect, useMemo, useRef, useState } from "react";
import { canApplyPreviewResponse, createPreviewFingerprint, isPreviewStale, previewPolicyPair } from "./policyPreview.mjs";

type Policy = {
  schema_version: string;
  version: string;
  title: string;
  stage: "sft" | "preference";
  usage: "exploratory" | "baseline";
  prompt_template_id: string;
  tokenizer?: { repo: string; revision: string; test_only: boolean };
  rules: {
    correctness: "filter" | "report_only";
    completeness: "filter" | "report_only";
    answer_parse: "filter" | "report_only";
    min_length_ratio: number | null;
    max_length_ratio: number | null;
  };
  resources: { max_length: number; target_count: number; max_scan_rows: number | null };
};
type Source = { id: string; title: string; stage: string | null; input_scope: string; available: boolean; config_ref: string; tokenizer_repo: string; tokenizer_revision: string; test_only: boolean };
type Capabilities = { enabled: boolean; token: string | null; templates: { config: Policy }[]; sources: Source[] };
type Preview = { policy_id: string; source_id: string; source_pool_id: string; split_snapshot_id: string; report: Record<string, any> };
type ComparisonReport = { report_id: string; created_at: string; experiment_id: string; comparison_status: string; experiment_type: string; evidence_kind: string; compatibility_issues?: string[] | null; policy_changes?: { field: string; a: unknown; b: unknown }[] | null; data?: Record<string, any> | null; model_metrics?: { metric: string; unit: string; direction: string; a: number | null; b: number | null; delta: number | null; per_seed: { seed: number; a: number | null; b: number | null; denominators: (number | null)[] }[] }[] | null; limitations?: string[] | null };

const freshPolicy = (stage: "sft" | "preference"): Policy => ({
  schema_version: "1.0", version: "1", title: "探索策略", stage, usage: "exploratory",
  prompt_template_id: "qwen3-math-v1",
  rules: { correctness: "filter", completeness: "filter", answer_parse: "filter", min_length_ratio: stage === "preference" ? 0.5 : null, max_length_ratio: stage === "preference" ? 2 : null },
  resources: { max_length: 1024, target_count: 200, max_scan_rows: null },
});
const clone = <T,>(value: T): T => JSON.parse(JSON.stringify(value)) as T;
const pretty = (value: unknown) => JSON.stringify(value, null, 2);
const safeError = (e: unknown) => e instanceof Error ? e.message : "请求失败，请检查本地服务";

async function api<T>(url: string, token?: string, body?: unknown): Promise<T> {
  const response = await fetch(url, { method: body ? "POST" : "GET", headers: {
    ...(body ? { "Content-Type": "application/json" } : {}), ...(token ? { "X-Local-Editor-Token": token } : {}),
  }, ...(body ? { body: JSON.stringify(body) } : {}) });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `本地服务返回 ${response.status}`);
  return data as T;
}

export function PolicyEditor({ visible, reportsOnly = false, onBack, onViewEffects }: { visible: boolean; reportsOnly?: boolean; onBack: () => void; onViewEffects: () => void }) {
  const [initialDraft] = useState(() => {
    try { return JSON.parse(sessionStorage.getItem("onegpu-policy-editor-draft-v1") || "null"); }
    catch { return null; }
  });
  const [cap, setCap] = useState<Capabilities | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [sourceId, setSourceId] = useState(initialDraft?.sourceId || "");
  const [stage, setStage] = useState<"sft" | "preference">(initialDraft?.stage || "sft");
  const [experimentType, setExperimentType] = useState<"pipeline_effect" | "controlled_quality_ablation">(initialDraft?.experimentType || "controlled_quality_ablation");
  const [a, setA] = useState<Policy>(initialDraft?.a || freshPolicy("sft"));
  const [b, setB] = useState<Policy>(initialDraft?.b || { ...freshPolicy("sft"), title: "放宽质量过滤" });
  const [rows, setRows] = useState(initialDraft?.rows || 200);
  const [experimentId] = useState(() => initialDraft?.experimentId || new URLSearchParams(window.location.search).get("experiment_id") || `exp-ui-${crypto.randomUUID().replaceAll("-", "").slice(0, 16)}`);
  const [planFields, setPlanFields] = useState(initialDraft?.planFields || {
    question: "筛选规则变化对训练数据组成有什么影响？", parent_model_id: "", parent_model_sha256: "",
    recipe_sha256: "", training_json: "{\n  \"max_length\": 1024\n}", seeds: "42",
    budget_unit: stage === "sft" ? "examples" : "preference_pairs", budget_value: "2",
    eval_dataset_id: "", eval_dataset_sha256: "", eval_role: "diagnostic_dev",
    eval_protocol: "", eval_config_sha256: "", metric_name: "internal_math.accuracy", metric_unit: "ratio",
  });
  const [preview, setPreview] = useState<{ a?: Preview; b?: Preview; key?: string }>({});
  const [previewBusy, setPreviewBusy] = useState(false);
  const [saved, setSaved] = useState<{ draft_id: string; path: string; policy_ids: string[] } | null>(null);
  const [savedPlan, setSavedPlan] = useState<{ plan_path: string; experiment_id: string; config_sha256: string } | null>(null);
  const [planError, setPlanError] = useState("");
  const [reports, setReports] = useState<ComparisonReport[]>([]);
  const [reportError, setReportError] = useState("");
  const requestId = useRef(0);
  const draftFingerprint = useMemo(() => pretty({ sourceId, stage, experimentType, a, b, rows, experimentId, planFields }), [sourceId, stage, experimentType, a, b, rows, experimentId, planFields]);
  const currentPreviewFingerprint = useMemo(() => createPreviewFingerprint(sourceId, stage, a, b, rows), [sourceId, stage, a, b, rows]);
  const previewFingerprintRef = useRef(currentPreviewFingerprint);
  previewFingerprintRef.current = currentPreviewFingerprint;
  const source = cap?.sources.find((item) => item.id === sourceId);
  const sourceCompatible = !source || source.input_scope !== "existing_dataset" || source.stage === stage;
  const withSourceTokenizer = (policy: Policy): Policy => source ? ({ ...policy, tokenizer: {
    repo: source.tokenizer_repo, revision: source.tokenizer_revision, test_only: source.test_only,
  } }) : policy;
  const draftKey = draftFingerprint;
  const changes = policyDiff(a, b);
  const previewStale = isPreviewStale(Boolean(preview.a && preview.b), preview.key, currentPreviewFingerprint);
  const requestedReportId = new URLSearchParams(window.location.search).get("report_id") || "";
  const visibleReports = requestedReportId ? reports.filter((report) => report.report_id === requestedReportId) : reports;

  useEffect(() => {
    api<Capabilities>("/api/policies/capabilities").then((value) => {
      setCap(value);
      const first = value.sources[0];
      if (first && !initialDraft?.sourceId) setSourceId(first.id);
    }).catch((e) => setError(safeError(e))).finally(() => setLoading(false));
  }, [initialDraft]);
  useEffect(() => { sessionStorage.setItem("onegpu-policy-editor-draft-v1", draftKey); }, [draftKey]);
  useEffect(() => { setSavedPlan(null); }, [draftKey]);
  useEffect(() => {
    api<{ reports: ComparisonReport[] }>("/api/policies/reports")
      .then((result) => setReports(result.reports))
      .catch((e) => setReportError(safeError(e)));
  }, []);

  function changeStage(next: "sft" | "preference") {
    setStage(next);
    setA(freshPolicy(next));
    setB({ ...freshPolicy(next), title: "放宽质量过滤" });
    setSaved(null);
    setSavedPlan(null);
    setPlanFields((current: Record<string, string>) => ({ ...current, budget_unit: next === "sft" ? "examples" : "preference_pairs" }));
  }
  function edit(which: "a" | "b", update: (value: Policy) => Policy) {
    (which === "a" ? setA : setB)((current) => update(clone(current)));
    setSaved(null);
    setSavedPlan(null);
  }
  function loadTemplate(which: "a" | "b", relaxed: boolean) {
    const template = cap?.templates.find((item) => item.config.stage === stage &&
      (item.config.rules.completeness === "report_only") === relaxed)?.config;
    if (!template) return;
    edit(which, () => ({ ...template, title: which === "a" ? "A · 基线筛选" : "B · 放宽质量过滤", usage: "exploratory" }));
  }
  async function runPreview() {
    if (!cap?.enabled || !cap.token || !source) return;
    const current = ++requestId.current;
    const requestFingerprint = previewFingerprintRef.current;
    setPreviewBusy(true); setError("");
    try {
      const result = await previewPolicyPair(
        (policy) => api<Preview>("/api/policies/preview", cap.token!, { source_id: source.id, policy, rows }),
        [a, b],
      );
      if (canApplyPreviewResponse(current, requestId.current, requestFingerprint, previewFingerprintRef.current)) setPreview({ ...result, key: requestFingerprint });
    } catch (e) { if (canApplyPreviewResponse(current, requestId.current, requestFingerprint, previewFingerprintRef.current)) setError(safeError(e)); }
    finally { if (current === requestId.current) setPreviewBusy(false); }
  }
  async function saveDraft() {
    if (!cap?.enabled || !cap.token) return;
    try {
      setError("");
      const result = await api<typeof saved>("/api/policies/save", cap.token, { source_id: sourceId, policies: [withSourceTokenizer(a), withSourceTokenizer(b)] });
      setSaved(result);
      setSavedPlan(null);
    } catch (e) { setError(safeError(e)); }
  }
  function downloadPlan() {
    const plan = makeExperimentPlan();
    const blob = new Blob([pretty(plan)], { type: "application/yaml" });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = "experiment.yaml"; link.click(); URL.revokeObjectURL(link.href);
  }
  function makeExperimentPlan() {
    if (!source || !preview.a || !preview.b || previewStale || preview.a.source_pool_id !== preview.b.source_pool_id || preview.a.split_snapshot_id !== preview.b.split_snapshot_id) throw new Error("请先对同一来源完成一组有效且未过期的 A/B 预览");
    if (experimentType === "controlled_quality_ablation" && changes.length !== 1) throw new Error("受控质量消融要求恰好一个策略差异");
    const training = JSON.parse(planFields.training_json);
    const seeds = planFields.seeds.split(",").map((item: string) => Number(item.trim()));
    if (!training || typeof training !== "object" || Array.isArray(training) || !Object.keys(training).length) throw new Error("训练配置需要是非空 JSON 对象；seed 单独填写");
    if (!seeds.length || seeds.some((seed: number) => !Number.isInteger(seed)) || new Set(seeds).size !== seeds.length) throw new Error("seed 需为不重复的整数，使用逗号分隔");
    const sha = /^[0-9a-f]{64}$/;
    for (const key of ["parent_model_sha256", "recipe_sha256", "eval_dataset_sha256", "eval_config_sha256"]) if (!sha.test(planFields[key])) throw new Error(`${key} 必须是 64 位小写 SHA256`);
    if (!planFields.parent_model_id.trim() || !planFields.eval_dataset_id.trim() || !planFields.eval_protocol.trim() || !planFields.metric_name.trim()) throw new Error("模型、评测数据、评测协议和主指标都必须填写");
    const base = `artifacts/data-experiments/${saved?.draft_id || "DRAFT_ID_REQUIRED"}`;
    return { schema_version: "1.0", experiment_id: experimentId, question: planFields.question,
      evidence_kind: source.test_only ? "fixture" : "real", experiment_type: experimentType,
      independent_variable: experimentType === "controlled_quality_ablation" ? changes[0] : null,
      source_pool_id: preview.a.source_pool_id, split_snapshot_id: preview.a.split_snapshot_id,
      a: { policy: withSourceTokenizer(a), manifest: `${base}/data-a/manifests/${stage}-policy-v1.yaml`, budget: { unit: planFields.budget_unit, value: Number(planFields.budget_value) } },
      b: { policy: withSourceTokenizer(b), manifest: `${base}/data-b/manifests/${stage}-policy-v1.yaml`, budget: { unit: planFields.budget_unit, value: Number(planFields.budget_value) } },
      controls: { parent_model_id: planFields.parent_model_id, parent_model_sha256: planFields.parent_model_sha256,
        recipe_sha256: planFields.recipe_sha256, training, seeds },
      evaluation: { dataset_ids: [planFields.eval_dataset_id], dataset_sha256: { [planFields.eval_dataset_id]: planFields.eval_dataset_sha256 },
        roles: { [planFields.eval_dataset_id]: planFields.eval_role }, protocol_version: planFields.eval_protocol,
        config_sha256: planFields.eval_config_sha256 },
      primary_metrics: [{ name: planFields.metric_name, unit: planFields.metric_unit, direction: "higher" }], guardrail_metrics: [] };
  }
  async function saveExperimentPlan() {
    if (!saved || !cap?.token) { setPlanError("请先保存 A/B 策略草稿。 "); return; }
    try {
      setPlanError("");
      const plan = makeExperimentPlan() as Record<string, any>;
      const budget = plan.a.budget;
      delete plan.a; delete plan.b;
      const result = await api<typeof savedPlan>("/api/policies/save-experiment", cap.token, {
        draft_id: saved.draft_id, source_id: sourceId, budget,
        experiment: { ...plan, budget },
      });
      setSavedPlan(result);
    } catch (e) { setPlanError(safeError(e)); }
  }
  let planReady = false;
  try { makeExperimentPlan(); planReady = Boolean(saved && !previewStale); } catch { /* Form explains which required values are still missing. */ }

  return <section className={`policy-page${reportsOnly ? " policy-reports-page" : ""}`} hidden={!visible} aria-labelledby="policy-title">
    <div className="policy-title-row">
      <div><div className="eyebrow">{reportsOnly ? "ANALYSIS ／ 已执行策略效果" : "DATA POLICY LAB ／ 数据策略实验"}</div><h1 id="policy-title">{reportsOnly ? "查看已执行的策略效果。" : "规则变化会怎样影响数据？"}</h1>
        <p>{reportsOnly ? "仅展示已写出的报告；保存草稿或预览筛选不会被误认为模型效果。" : "先在同一来源上对照数据筛选与质量，再决定是否保存为实验草稿。"}</p></div>
      <button className="button secondary" onClick={onBack}>{reportsOnly ? "返回规则编辑" : "返回数据页面"}</button>
    </div>
    {!reportsOnly && requestedReportId && <p className="policy-report-context" role="status">从策略效果报告返回：experiment <code>{experimentId}</code> · report <code>{requestedReportId}</code>。此操作只恢复对象身份，不会重新登记或执行。</p>}
    {!cap?.enabled && <div className="policy-alert"><strong>当前为只读模式。</strong> 如需本地预览和保存，在终端运行 <code>onegpu-posttrain serve-ui --enable-policy-editor</code> 后刷新本页。</div>}
    {error && <div className="policy-alert" role="alert"><strong>操作未完成。</strong> {error} 草稿仍保留在当前页面。</div>}
    {loading ? <p className="policy-loading" aria-live="polite">正在读取本地来源与已执行报告…</p> : <>
      <section className="panel policy-setup">
        <h2>1. 选择共同条件</h2>
        <div className="policy-controls">
          <label>训练阶段<select value={stage} onChange={(e) => changeStage(e.target.value as "sft" | "preference")}>
            <option value="sft">SFT</option><option value="preference">Preference（DPO / SimPO 数据）</option>
          </select><small>CPT、GRPO 的规则编辑尚未开放。</small></label>
          <label>本地数据来源<select value={sourceId} onChange={(e) => { setSourceId(e.target.value); setSaved(null); setSavedPlan(null); }}>
            {cap?.sources.map((item) => <option value={item.id} key={item.id}>{item.title}{item.available ? "" : "（资源缺失）"}</option>)}
          </select><small>{source ? `${source.input_scope === "raw_source" ? "原始来源" : "已有筛选数据"} · ${source.test_only ? "测试 tokenizer" : "本地 tokenizer"}` : "没有发现来源配置；请在 configs/data/policy-sources/ 登记本地来源。"}</small></label>
          <label>预览 raw rows<input type="number" min={1} max={1000} value={rows} onChange={(e) => setRows(Math.max(1, Math.min(1000, Number(e.target.value) || 1)))} /><small>最多 1000 行，最多运行 30 秒；只读取本地文件。</small></label>
          <label>实验解释<select value={experimentType} onChange={(e) => setExperimentType(e.target.value as typeof experimentType)}><option value="controlled_quality_ablation">受控质量消融（只改一个规则）</option><option value="pipeline_effect">流水线效果（允许多项变化）</option></select><small>多项差异只能报告整条规则流水线的效果，不归因到单条规则。</small></label>
        </div>
      </section>
      <div className="policy-arms">
        {[{ key: "a" as const, value: a, label: "A · 基线规则" }, { key: "b" as const, value: b, label: "B · 对照规则" }].map(({ key, value, label }) => <section className="panel policy-arm" key={key}>
          <div className="policy-arm-head"><div><span className="policy-index">{key.toUpperCase()}</span><h2>{label}</h2></div><div className="policy-actions">
            <button className="text-button" onClick={() => loadTemplate(key, false)}>加载基线模板</button><button className="text-button" onClick={() => loadTemplate(key, true)}>放宽质量过滤模板</button>
          </div></div>
          <p>“仅报告”会保留样本并标记标签/完整性状态；未知标签不会被当作错误答案。</p>
          <label>正确性规则<select value={value.rules.correctness} disabled={stage === "preference"} onChange={(e) => edit(key, (p) => ({ ...p, rules: { ...p.rules, correctness: e.target.value as Policy["rules"]["correctness"] } }))}>
            <option value="filter">筛除不符合项</option><option value="report_only">仅报告，不因该项筛除</option>
          </select><small>Preference 必须筛选已知的正确/错误排序依据。</small></label>
          <label>推理完整性<select value={value.rules.completeness} onChange={(e) => edit(key, (p) => ({ ...p, rules: { ...p.rules, completeness: e.target.value as Policy["rules"]["completeness"] } }))}>
            <option value="filter">筛除不完整推理</option><option value="report_only">保留并单独报告</option>
          </select></label>
          <label>答案解析<select value={value.rules.answer_parse} onChange={(e) => edit(key, (p) => ({ ...p, rules: { ...p.rules, answer_parse: e.target.value as Policy["rules"]["answer_parse"] } }))}>
            <option value="filter">筛除无法解析答案</option><option value="report_only">保留并单独报告</option>
          </select></label>
          {stage === "preference" && <div className="policy-ratio"><label>chosen/rejected 最小长度比<input type="number" min="0.01" step="0.1" value={value.rules.min_length_ratio ?? 0.5} onChange={(e) => edit(key, (p) => ({ ...p, rules: { ...p.rules, min_length_ratio: Number(e.target.value) } }))} /></label><label>最大长度比<input type="number" min="0.01" step="0.1" value={value.rules.max_length_ratio ?? 2} onChange={(e) => edit(key, (p) => ({ ...p, rules: { ...p.rules, max_length_ratio: Number(e.target.value) } }))} /></label></div>}
          <label>单样本 token 上限<input type="number" min="1" max="131072" value={value.resources.max_length} onChange={(e) => edit(key, (p) => ({ ...p, resources: { ...p.resources, max_length: Number(e.target.value) } }))} /><small>使用来源配置中指定的 tokenizer；上限不是字符数。</small></label>
          <label>目标样本数（SFT） / 偏好对数<input type="number" min="1" value={value.resources.target_count} onChange={(e) => edit(key, (p) => ({ ...p, resources: { ...p.resources, target_count: Number(e.target.value) } }))} /></label>
          <button className="text-button" onClick={() => edit(key, () => ({ ...freshPolicy(stage), title: key === "a" ? "A · 基线规则" : "B · 对照规则" }))}>恢复阶段默认值</button>
        </section>)}
      </div>
      <section className="panel policy-diff"><h2>当前差异</h2><p>{experimentType === "controlled_quality_ablation" ? "受控质量消融应只改变一项；样本数量/覆盖变化仍需记录，不能仅由数据质量分数推断模型提升。" : "流水线效果保留多项规则和数据组成差异，不将结果归因到单一规则。"}</p>
        <ul>{changes.map((item) => <li key={item}><code>{item}</code></li>)}</ul>{changes.length === 0 && <p>两组策略相同，还没有可比较的规则差异。</p>}
        {experimentType === "controlled_quality_ablation" && changes.length > 1 && <div className="policy-alert">当前有 {changes.length} 项差异。请改为“流水线效果”或把受控消融缩减到一个变量。</div>}
        {source?.input_scope === "existing_dataset" && <div className="policy-alert">当前来源已经筛选过，无法恢复上游被删除的候选；只能比较现存数据上的后续规则。</div>}
      </section>
      <div className="policy-primary-actions"><button className="button primary" disabled={!cap?.enabled || !source?.available || !sourceCompatible || previewBusy || policyDiff(a,b).length === 0} onClick={runPreview}>{previewBusy ? "正在本地预览…" : "预览 A / B 数据"}</button>
        <button className="button secondary" disabled={!cap?.enabled} onClick={saveDraft}>保存策略草稿</button><button className="button secondary" onClick={downloadPlan}>导出草稿 JSON</button></div>
      {!source?.available && <div className="policy-alert">来源文件、污染索引或缓存暂不可用。不会自动下载，也不会用示例数据代替真实来源。</div>}
      {!sourceCompatible && <div className="policy-alert">该来源是已筛选的 {source?.stage?.toUpperCase()} 数据，无法用于当前 {stage.toUpperCase()} 阶段。</div>}
      {previewBusy && <p aria-live="polite">预览只扫描所选前缀；处理中可以继续编辑，旧响应不会覆盖较新的请求。</p>}
      {(preview.a || preview.b) && <section className="panel policy-results"><h2>预览结果 {previewStale && <span className="policy-stale">已过期，请重新预览</span>}</h2><p>以下是数据构建统计，不是训练结果；模型效果：<strong>未评估</strong>。</p>
        <div className="policy-result-grid">{(["a", "b"] as const).map((key) => <PreviewCard key={key} label={key.toUpperCase()} value={preview[key]} />)}</div>
        {!previewStale && preview.a && preview.b && <p>来源池 {preview.a.source_pool_id.slice(0, 24)}… · split {preview.a.split_snapshot_id.slice(0, 24)}… · 两组使用相同的题目切分。</p>}
      </section>}
      <section className="panel policy-setup"><h2>2. 补全可复现的实验合同</h2><p>填写同一 parent、recipe、训练配置、预算、seed 和评测协议。仅保存配置不会启动训练，也不会生成模型结果。</p>
        <div className="policy-controls">
          <label>实验问题<input value={planFields.question} onChange={(e) => setPlanFields({ ...planFields, question: e.target.value })} /></label>
          <label>共同 parent model ID<input value={planFields.parent_model_id} onChange={(e) => setPlanFields({ ...planFields, parent_model_id: e.target.value })} /></label>
          <label>Parent SHA256<input value={planFields.parent_model_sha256} onChange={(e) => setPlanFields({ ...planFields, parent_model_sha256: e.target.value })} /><small>64 位小写 SHA256。</small></label>
          <label>Recipe SHA256<input value={planFields.recipe_sha256} onChange={(e) => setPlanFields({ ...planFields, recipe_sha256: e.target.value })} /><small>recipe 文件内容的 SHA256。</small></label>
          <label>共同训练预算单位<select value={planFields.budget_unit} onChange={(e) => setPlanFields({ ...planFields, budget_unit: e.target.value })}><option value="examples">examples</option><option value="preference_pairs">preference_pairs</option><option value="tokens">tokens</option><option value="optimizer_steps">optimizer_steps</option></select><small>A/B 必须用相同单位与数值。</small></label>
          <label>共同预算值<input type="number" min="1" value={planFields.budget_value} onChange={(e) => setPlanFields({ ...planFields, budget_value: e.target.value })} /></label>
          <label>Training 参数 JSON（不含 seed）<textarea rows={5} value={planFields.training_json} onChange={(e) => setPlanFields({ ...planFields, training_json: e.target.value })} /><small>计划训练应对应的解析参数；使用同一个 seed 列表。</small></label>
          <label>Seeds（逗号分隔）<input value={planFields.seeds} onChange={(e) => setPlanFields({ ...planFields, seeds: e.target.value })} /></label>
          <label>固定评测 dataset ID<input value={planFields.eval_dataset_id} onChange={(e) => setPlanFields({ ...planFields, eval_dataset_id: e.target.value })} /></label>
          <label>评测数据 SHA256<input value={planFields.eval_dataset_sha256} onChange={(e) => setPlanFields({ ...planFields, eval_dataset_sha256: e.target.value })} /></label>
          <label>评测用途<select value={planFields.eval_role} onChange={(e) => setPlanFields({ ...planFields, eval_role: e.target.value })}><option value="diagnostic_dev">diagnostic_dev</option><option value="final_holdout">final_holdout</option></select></label>
          <label>评测协议版本<input value={planFields.eval_protocol} onChange={(e) => setPlanFields({ ...planFields, eval_protocol: e.target.value })} /></label>
          <label>评测配置 SHA256<input value={planFields.eval_config_sha256} onChange={(e) => setPlanFields({ ...planFields, eval_config_sha256: e.target.value })} /></label>
          <label>主指标名称<input value={planFields.metric_name} onChange={(e) => setPlanFields({ ...planFields, metric_name: e.target.value })} /></label>
          <label>指标类型<select value={planFields.metric_unit} onChange={(e) => setPlanFields({ ...planFields, metric_unit: e.target.value })}><option value="ratio">比例（ratio）</option><option value="number">数值（number）</option></select></label>
        </div>
        {planError && <div className="policy-alert" role="alert">实验计划未保存：{planError}</div>}
        <div className="policy-primary-actions"><button className="button secondary" disabled={!cap?.enabled || !saved || previewStale || !planReady || Boolean(savedPlan)} onClick={saveExperimentPlan}>校验并保存实验计划</button><button className="button secondary" disabled={!saved || !planReady} onClick={() => { try { downloadPlan(); setPlanError(""); } catch (e) { setPlanError(safeError(e)); } }}>导出 experiment.yaml</button></div>
        {savedPlan && <p>计划已按共享 `DataEffectExperiment` schema 校验并保存：<code>{savedPlan.plan_path}</code>。待完成数据构建后才可预登记。</p>}
        {!planReady && <p className="policy-help">填写 parent/recipe/eval 的 SHA256、训练 JSON、预算和 seed，并先完成有效 A/B 预览与策略草稿保存。</p>}
      </section>
      {saved && <section className="panel policy-saved"><h2>策略草稿已保存</h2><p><code>{saved.path}</code> · 尚未构建正式数据、训练或评测。</p>
        <CommandBlock source={source} path={saved.path} stage={stage} planPath={savedPlan?.plan_path} />
      </section>}
      <section className={`panel policy-results${reportsOnly ? " policy-executed-reports" : ""}`}><div className="policy-arm-head"><div><h2>{reportsOnly ? "已执行的策略效果报告" : "策略效果已集中到分析模块"}</h2><p>{reportsOnly ? "读取本地 artifacts/data-experiments 的实际报告；按报告身份关联数据变化和模型评测。" : "规则编辑与数据预览留在数据模块；完成执行后的效果与指标集中在分析模块。"}</p></div>{reportsOnly && <button className="text-button" onClick={() => api<{ reports: ComparisonReport[] }>("/api/policies/reports").then((result) => { setReports(result.reports); setReportError(""); }).catch((e) => setReportError(safeError(e)))}>重新读取报告</button>}</div>
        {!reportsOnly && <div className="policy-primary-actions"><button className="button secondary" onClick={onViewEffects}>前往分析查看策略效果 →</button><span>规则预览、草稿和实际训练/评测报告是不同状态。</span></div>}
        {reportsOnly && reportError && <p role="alert">报告读取失败：{reportError}</p>}
        {reportsOnly && requestedReportId && !visibleReports.length && !reportError && <p>未找到报告 {requestedReportId}；请刷新报告目录或返回策略效果列表。</p>}
        {reportsOnly && !requestedReportId && !reports.length ? <p>尚无已保存的比较报告。保存规则草稿不会自动生成训练结果。</p> : null}
        {reportsOnly && visibleReports.map((report) => <ComparisonView key={report.report_id} report={report} />)}
      </section>
      <section className="panel policy-boundary"><h2>评测驱动的数据迭代</h2><p>完整链路是“评测定位薄弱场景 → 生成或选择训练数据 → 质量检查 → 训练 → 同协议复测”。当前这页支持规则对照与离线数据预览；大模型生成、训练调度及场景回归后续按独立 OpenSpec 变更实现。</p></section>
    </>}
  </section>;
}

function policyDiff(a: Policy, b: Policy) {
  return [
    ...(a.rules.correctness !== b.rules.correctness ? ["rules.correctness"] : []),
    ...(a.rules.completeness !== b.rules.completeness ? ["rules.completeness"] : []),
    ...(a.rules.answer_parse !== b.rules.answer_parse ? ["rules.answer_parse"] : []),
    ...(a.rules.min_length_ratio !== b.rules.min_length_ratio || a.rules.max_length_ratio !== b.rules.max_length_ratio ? ["rules.length_ratio"] : []),
    ...(a.resources.max_length !== b.resources.max_length ? ["resources.max_length"] : []),
    ...(a.resources.target_count !== b.resources.target_count ? ["resources.target_count"] : []),
  ];
}
function PreviewCard({ label, value }: { label: string; value?: Preview }) {
  const counters = value?.report.counters || {};
  const fmt = (v: unknown) => typeof v === "number" ? new Intl.NumberFormat("zh-CN").format(v) : "—";
  return <article className="policy-result-card"><h3>{label} 组</h3>{!value ? <p>尚无有效预览。</p> : <>
    <dl>{[["扫描行数", counters.raw_rows, "raw rows"], ["阶段问题数", counters.stage_attempted, "problems"], ["符合策略", counters.eligible, "problems"], ["最终选中", value.report.selected?.[value.report.policy.config.stage], value.report.count_unit], ["未知正确性标签", counters.unknown_correctness_traces, "traces"], ["不完整推理", counters.incomplete_traces, "traces"], ["超 token 上限", counters.over_length_traces, "traces"]].map(([name, amount, unit]) => <div key={String(name)}><dt>{name}<small>{unit}</small></dt><dd>{fmt(amount)}</dd></div>)}</dl>
    <p>停止原因：{String(value.report.stop_reason)} · {value.report.partial ? "部分前缀预览" : "完整目标扫描"}</p></>}</article>;
}
function ComparisonView({ report }: { report: ComparisonReport }) {
  const value = (number: number | null, unit: string) => number == null ? "未评估" : unit === "ratio" ? `${(number * 100).toFixed(2)}%` : new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 3 }).format(number);
  const status = report.comparison_status === "COMPARABLE" ? "可比" : report.comparison_status === "NOT_EVALUATED" ? "未评估" : "不可直接比较";
  const compatibilityIssues = report.compatibility_issues ?? [];
  const policyChanges = report.policy_changes ?? [];
  const modelMetrics = report.model_metrics ?? [];
  const limitations = report.limitations ?? [];
  const data = report.data ?? {};
  return <article className="policy-report"><div className="policy-report-head"><h3>{report.experiment_id}</h3><span className={`policy-report-status ${report.comparison_status}`}>{status}</span></div>
    <p>{report.experiment_type} · {report.evidence_kind} · {new Date(report.created_at).toLocaleString("zh-CN")}</p>
    <div className="policy-report-identity"><code>report_id: {report.report_id}</code><a className="text-button" href={`/?page=policies&report_id=${encodeURIComponent(report.report_id)}&experiment_id=${encodeURIComponent(report.experiment_id)}`}>返回关联规则草稿 →</a></div>
    {!!compatibilityIssues.length && <details><summary>查看 {compatibilityIssues.length} 项兼容性/证据问题</summary><ul>{compatibilityIssues.map((issue) => <li key={issue}>{issue}</li>)}</ul></details>}
    <h4>数据变化</h4>{report.data ? <div className="policy-report-data">{(["a", "b"] as const).map((key) => { const arm = data[key] ?? {}; return <div key={key}><strong>{key.toUpperCase()} 组</strong><span>记录数：{arm.records ?? "—"} {arm.unit || ""}</span><span>质量门禁：{arm.quality?.gate?.summary_status ?? "未评估"}</span><span>来源策略：{arm.policy_id ?? "—"}</span><details><summary>查看逐项质量与覆盖（含分母）</summary><QualitySummary quality={arm.quality} coverage={arm.coverage} /></details></div>; })}</div> : <p>该历史报告未记录 A/B 数据摘要。</p>}
    <h4>模型评测</h4>{!modelMetrics.length || modelMetrics.every((metric) => metric.a == null && metric.b == null) ? <p>模型效果未评估。完成符合登记的训练与评测后，再用 CLI 生成新的报告。</p> : <div className="table-wrap"><table><thead><tr><th>指标</th><th>A</th><th>B</th><th>Δ (B−A)</th><th>seed 分母</th></tr></thead><tbody>{modelMetrics.map((metric) => <tr key={metric.metric}><th>{metric.metric}</th><td>{value(metric.a, metric.unit)}</td><td>{value(metric.b, metric.unit)}</td><td>{report.comparison_status === "COMPARABLE" ? value(metric.delta, metric.unit) : "不可比较"}</td><td>{(metric.per_seed ?? []).map((seed) => `${seed.seed}: ${(seed.denominators ?? []).map((d) => d ?? "—").join("/")}`).join(" · ") || "—"}</td></tr>)}</tbody></table></div>}
    <details><summary>查看策略改动和解释限制</summary><ul>{policyChanges.map((change) => <li key={change.field}><code>{change.field}</code>: {String(change.a)} → {String(change.b)}</li>)}{limitations.map((item) => <li key={item}>{item}</li>)}</ul></details>
  </article>;
}
function QualitySummary({ quality, coverage }: { quality?: Record<string, any>; coverage?: Record<string, any> }) {
  const metrics = quality?.metrics || {};
  const fmt = (metric: Record<string, any>) => metric.status !== "measured" || metric.value == null
    ? "未测量" : metric.unit === "ratio" ? `${(metric.value * 100).toFixed(2)}%` : String(metric.value);
  return <div className="quality-summary"><p>门禁汇总只反映该协议的检查状态，不是综合质量分数。</p>
    {Object.keys(metrics).length ? <div className="table-wrap"><table><thead><tr><th>指标</th><th>结果</th><th>分子 / 分母</th><th>方法与局限</th></tr></thead><tbody>{Object.entries(metrics).map(([name, metric]: [string, any]) => <tr key={name}><th>{name}</th><td>{fmt(metric)}<br />{metric.status}</td><td>{metric.numerator ?? "—"} / {metric.denominator ?? "—"} {metric.unit}</td><td>{metric.method}{metric.limitations?.length ? `；${metric.limitations.join("；")}` : ""}</td></tr>)}</tbody></table></div> : <p>没有质量指标。</p>}
    <details><summary>查看构建前后覆盖</summary><pre>{JSON.stringify(coverage || { status: "not_measured" }, null, 2)}</pre></details>
    {!!quality?.limitations?.length && <details><summary>查看评估限制</summary><ul>{quality.limitations.map((item: string) => <li key={item}>{item}</li>)}</ul></details>}
  </div>;
}
function CommandBlock({ source, path, stage, planPath }: { source?: Source; path: string; stage: string; planPath?: string }) {
  const quote = (v: string) => `"${v.replaceAll('"', '\\"')}"`;
  const commands = source ? [
    `.venv/Scripts/python.exe -m onegpu_posttrain build-policy-data --policy ${quote(`${path}/policy-a.yaml`)} --source ${quote(source.config_ref)} --output ${quote(`${path}/data-a`)}`,
    `.venv/Scripts/python.exe -m onegpu_posttrain build-policy-data --policy ${quote(`${path}/policy-b.yaml`)} --source ${quote(source.config_ref)} --output ${quote(`${path}/data-b`)}`,
    ...(planPath ? [`.venv/Scripts/python.exe -m onegpu_posttrain register-data-experiment --config ${quote(planPath)} --root . --output ${quote(`${path}/registration.json`)}`] : []),
  ] : [];
  return <div><p>以下命令在本地构建 A/B 数据并预登记实验；不会启动训练。复制前请核对资源、预算和评测信息：</p>{commands.map((command) => <div className="policy-command" key={command}><pre><code>{command}</code></pre><button className="text-button" onClick={() => navigator.clipboard?.writeText(command)}>复制命令</button></div>)}<p>当前阶段：{stage.toUpperCase()}。计划注册后，再按现有 CLI 流程执行训练和评测；该页面不启动训练进程。</p></div>;
}
