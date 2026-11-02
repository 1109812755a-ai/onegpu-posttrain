import { useEffect, useMemo, useState } from "react";
import { GenerationReviewPanel } from "./GenerationReviewPanel";

type Ready = { dataset_id: string; ordinary_sft: { eligible: boolean; missing_reasons: string[]; sample_count: number } };
type Setup = { datasets: { dataset_id: string; sample_count: number; content_sha256: string }[]; profiles: { profile_id: string; model: string; credential_env_configured: boolean; is_example: boolean }[]; readiness: Ready[] };
type Job = { execution_id: string; bundle_id: string; status: string; progress?: Record<string, number> | null; result?: Record<string, any> | null; detail?: string | null; registered_dataset_id?: string; };
type UnknownRequest = { attempt_id: string; execution_id: string | null; state: string };
const STORAGE_KEY = "onegpu.sft-generation.execution";
const DRAFT_KEY = "onegpu.sft-generation.draft.v1";

function jobStatusLabel(status: string): string {
  switch (status) {
    case "queued": return "排队中";
    case "running": return "正在生成";
    case "completed": return "生成完成，待审核";
    case "partial": return "部分完成，待审核";
    case "cancelled": return "已取消，可审核已生成候选";
    case "interrupted": return "服务中断，未自动重试";
    case "failed": return "生成失败";
    default: return "状态未知";
  }
}

function readDraft(): Partial<{ mode: "seeded" | "prompt_only"; topic: string; skill: string; difficulty: string; constraints: string; target: number; profileId: string; datasetId: string }> {
  try {
    const value = JSON.parse(window.localStorage.getItem(DRAFT_KEY) || "null");
    return value && typeof value === "object" ? value : {};
  } catch { return {}; }
}

async function request<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, { ...init, headers: { "Content-Type": "application/json", "X-Local-Synthetic-Token": token, ...(init?.headers || {}) } });
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new Error(typeof body?.detail === "string" ? body.detail : `请求失败 (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export function SFTGenerationPanel({ onOpenTargeted, onRegistered }: { onOpenTargeted?: () => void; onRegistered?: () => void }) {
  const [initialDraft] = useState(readDraft);
  const [token, setToken] = useState("");
  const [enabled, setEnabled] = useState(false);
  const [setup, setSetup] = useState<Setup | null>(null);
  const [datasetId, setDatasetId] = useState(initialDraft.datasetId || "");
  const [mode, setMode] = useState<"seeded" | "prompt_only">(initialDraft.mode === "prompt_only" ? "prompt_only" : "seeded");
  const [topic, setTopic] = useState(initialDraft.topic || "基础代数");
  const [skill, setSkill] = useState(initialDraft.skill || "一元一次方程求解");
  const [difficulty, setDifficulty] = useState(initialDraft.difficulty || "introductory");
  const [constraints, setConstraints] = useState(initialDraft.constraints || "");
  const [target, setTarget] = useState(initialDraft.target || 20);
  const [profileId, setProfileId] = useState(initialDraft.profileId || "");
  const [bundleId, setBundleId] = useState("");
  const [preflight, setPreflight] = useState<Record<string, any> | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [preview, setPreview] = useState<Record<string, any> | null>(null);
  const [reviewed, setReviewed] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [unknownRequests, setUnknownRequests] = useState<UnknownRequest[]>([]);
  const profiles = useMemo(() => (setup?.profiles || []).filter((item) => item.credential_env_configured && !item.is_example), [setup]);
  const requestLimit = Math.min(target * 2, 30000);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const caps = await fetch("/api/synthetic/sft/capabilities").then((res) => res.json());
        if (cancelled) return;
        setEnabled(Boolean(caps.enabled));
        if (!caps.enabled || typeof caps.token !== "string") return;
        setToken(caps.token);
        const data = await request<Setup>("/api/synthetic/sft/readiness", caps.token);
        if (cancelled) return;
        setSetup(data);
        const eligible = data.readiness.find((item) => item.ordinary_sft.eligible);
        const defaultProfile = data.profiles.filter((item) => item.credential_env_configured && !item.is_example);
        if (eligible) setDatasetId(eligible.dataset_id);
        if (defaultProfile.length === 1) setProfileId(defaultProfile[0].profile_id);
        const saved = window.localStorage.getItem(STORAGE_KEY);
        if (saved) {
          const state = await request<Job>(`/api/synthetic/sft/runs/${encodeURIComponent(saved)}`, caps.token);
          if (!cancelled) setJob(state);
        }
      } catch (reason) { if (!cancelled) setError(reason instanceof Error ? reason.message : "生成准备状态读取失败"); }
    })();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(DRAFT_KEY, JSON.stringify({ mode, topic, skill, difficulty, constraints, target, profileId, datasetId }));
  }, [mode, topic, skill, difficulty, constraints, target, profileId, datasetId]);

  useEffect(() => {
    if (!job || !token || !["queued", "running"].includes(job.status)) return;
    const timer = window.setInterval(async () => {
      try { setJob(await request<Job>(`/api/synthetic/sft/runs/${encodeURIComponent(job.execution_id)}`, token)); }
      catch (reason) { setError(reason instanceof Error ? reason.message : "任务状态读取失败"); }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [job?.execution_id, job?.status, token]);

  const eligible = setup?.readiness.find((item) => item.dataset_id === datasetId)?.ordinary_sft.eligible ?? false;
  const configuredProfile = profiles.find((item) => item.profile_id === profileId);
  const jobActive = Boolean(job && ["queued", "running"].includes(job.status));
  const reviewBundleId = bundleId || job?.bundle_id || "";
  useEffect(() => {
    if (!reviewBundleId || !token) { setUnknownRequests([]); return; }
    let cancelled = false;
    request<{ unresolved: UnknownRequest[] }>(`/api/synthetic/sft/request-review/${encodeURIComponent(reviewBundleId)}`, token)
      .then((value) => { if (!cancelled) setUnknownRequests(value.unresolved); })
      .catch((reason) => { if (!cancelled) setError(reason instanceof Error ? reason.message : "请求审查状态读取失败"); });
    return () => { cancelled = true; };
  }, [reviewBundleId, token, job?.status]);

  async function acknowledgeRequest(attemptId: string) {
    if (!window.confirm("请先在 Provider 控制台核查该请求。确认已了解其结果仍未知，之后重新生成可能重复计费？此操作不会自动重新生成。")) return;
    setBusy(true); setError(""); setPreflight(null);
    try {
      const result = await request<{ unresolved: UnknownRequest[] }>(`/api/synthetic/sft/request-review/${encodeURIComponent(reviewBundleId)}`, token,
        { method: "POST", body: JSON.stringify({ attempt_id: attemptId, confirm_billing_risk: true }) });
      setUnknownRequests(result.unresolved);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "请求审查失败"); }
    finally { setBusy(false); }
  }
  async function prepare() {
    setBusy(true); setError(""); setPreflight(null); setPreview(null); setReviewed(false);
    try {
      const input = mode === "prompt_only"
        ? { mode, topic, skill, difficulty, constraints, target_count: target, ...(profileId ? { profile_id: profileId } : {}) }
        : { dataset_id: datasetId, target_count: target, ...(profileId ? { profile_id: profileId } : {}) };
      const prepared = await request<{ bundle_id: string }>("/api/synthetic/sft/prepare", token, { method: "POST", body: JSON.stringify(input) });
      setBundleId(prepared.bundle_id);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "方案准备失败"); }
    finally { setBusy(false); }
  }
  async function checkPlan() {
    setBusy(true); setError(""); setPreflight(null);
    try { setPreflight(await request("/api/synthetic/sft/preflight", token, { method: "POST", body: JSON.stringify({ bundle_id: bundleId, usable_target: target, max_requests: requestLimit, target_candidates: target }) })); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "预检失败"); }
    finally { setBusy(false); }
  }
  async function start() {
    if (!window.confirm(`将调用 ${configuredProfile?.model || "已配置的模型"}，最多请求 ${requestLimit} 次。继续吗？`)) return;
    setBusy(true); setError("");
    try {
      const state = await request<Job>("/api/synthetic/sft/start", token, { method: "POST", body: JSON.stringify({ bundle_id: bundleId, usable_target: target, max_requests: requestLimit, target_candidates: target, confirm_start: true }) });
      setJob(state); window.localStorage.setItem(STORAGE_KEY, state.execution_id);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "启动失败"); }
    finally { setBusy(false); }
  }
  async function loadPreview() {
    if (!job) return;
    setBusy(true); setError("");
    try { setPreview(await request(`/api/synthetic/sft/runs/${encodeURIComponent(job.execution_id)}/preview?limit=5`, token)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "预览失败"); }
    finally { setBusy(false); }
  }
  async function register() {
    if (!job || !reviewed || !window.confirm("确认已审核候选内容并登记为新的不可变 SFT dataset 吗？")) return;
    setBusy(true); setError("");
    try { const state = await request<Job>(`/api/synthetic/sft/runs/${encodeURIComponent(job.execution_id)}/register`, token, { method: "POST", body: JSON.stringify({ confirm_reviewed: true }) }); setJob(state); onRegistered?.(); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "登记失败"); }
    finally { setBusy(false); }
  }

  return <section className="sft-generation-card" aria-labelledby="sft-generation-title">
    <div className="dataset-import-heading"><div><div className="eyebrow">SFT DATA GENERATION</div><h2 id="sft-generation-title">SFT 数据生成</h2><p>可参考已有 SFT 样本扩写，也可直接按主题生成新题；生成候选须审核后登记。</p></div><button className="secondary-button" onClick={onOpenTargeted}>转到定向补数 →</button></div>
    {!enabled && <div className="dataset-workbench-alert" role="status">尚未启用本地生成能力。启动 UI 时添加 <code>--enable-synthetic-generation</code>。</div>}
    {error && <div className="dataset-workbench-alert" role="alert">{error}<button onClick={() => setError("")}>关闭</button></div>}
    {enabled && !setup && !error && <p className="workbench-loading" role="status">正在检查已登记 SFT 数据、默认模板和本地 Provider 配置…</p>}
    {enabled && <>
      {job && ["queued", "running"].includes(job.status) && <div className="active-task-inline" role="status"><span>已有生成任务仍在运行，切换下方草稿不会取消它。</span><code>{job.execution_id}</code></div>}
      <div className="sft-generation-form">
        <label>生成方式<select value={mode} onChange={(event) => { setMode(event.target.value as "seeded" | "prompt_only"); setBundleId(""); setPreflight(null); }}><option value="seeded">参考已有数据集</option><option value="prompt_only">按主题生成（无需种子数据）</option></select><small>两种方式都生成新的候选，不会修改原数据集；切换只更改草稿，不会丢弃已启动任务。</small></label>
        {mode === "seeded" ? <label>参考 SFT 数据集<select value={datasetId} onChange={(event) => { setDatasetId(event.target.value); setBundleId(""); setPreflight(null); }}><option value="">选择已登记 SFT 数据集</option>{(setup?.datasets || []).map((item) => <option key={item.dataset_id} value={item.dataset_id}>{item.dataset_id} · {item.sample_count} 条</option>)}</select><small>从已登记样本提取题目，让模型生成解答。</small></label> : <>
          <label>主题<input maxLength={120} value={topic} onChange={(event) => { setTopic(event.target.value); setBundleId(""); setPreflight(null); }} placeholder="例如：基础代数" /></label>
          <label>能力点<input maxLength={120} value={skill} onChange={(event) => { setSkill(event.target.value); setBundleId(""); setPreflight(null); }} placeholder="例如：一元一次方程求解" /></label>
          <label>难度<select value={difficulty} onChange={(event) => { setDifficulty(event.target.value); setBundleId(""); setPreflight(null); }}><option value="introductory">入门</option><option value="intermediate">中等</option><option value="advanced">进阶</option></select></label>
          <label className="import-wide">附加要求（可选）<textarea maxLength={500} value={constraints} onChange={(event) => { setConstraints(event.target.value); setBundleId(""); setPreflight(null); }} placeholder="例如：使用生活化情境，答案为整数" /></label>
          <p className="dataset-preview-integrity import-wide">此模式不依赖种子数据；生成的问题和解答由模型创作，正确性未经独立验证，登记前请检查。</p>
        </>}
        <label>目标生成数量<input type="number" min={1} max={mode === "prompt_only" ? 1000 : 10000} value={target} onChange={(event) => { setTarget(Math.max(1, Math.min(mode === "prompt_only" ? 1000 : 10000, Number(event.target.value) || 1))); setBundleId(""); setPreflight(null); }} /><small>目标数是生成数量，不是数据质量门槛。</small></label>
        <label>Provider 配置<select value={profileId} onChange={(event) => { setProfileId(event.target.value); setBundleId(""); setPreflight(null); }}><option value="">自动使用唯一可用配置</option>{profiles.map((item) => <option key={item.profile_id} value={item.profile_id}>{item.profile_id} · {item.model}</option>)}</select><small>密钥只从本地环境读取，不会发送到浏览器。</small></label>
      </div>
      {mode === "seeded" && setup?.datasets.length === 0 && <p className="workbench-empty">没有可用的已登记 SFT 数据集。可以切换到“按主题生成”，无需先导入种子数据。</p>}
      {mode === "seeded" && datasetId && !eligible && <p className="field-error">所选数据集未通过格式/摘要检查，不能作为生成来源。</p>}
      {profiles.length === 0 && setup && <p className="workbench-empty">没有发现凭据已配置的 Provider。先在项目 `.env.local` 配置本地凭据并重启 UI；页面不会索取或保存密钥。</p>}
      {unknownRequests.length > 0 && <div className="dataset-workbench-alert" role="alert">
        <strong>存在结果未知的模型请求，暂不能重新生成</strong>
        <p>所属方案：<code>{reviewBundleId}</code></p>
        <p>请先核查 Provider 控制台及费用。确认后请求结果仍标记为未知，不会自动重新调用模型；再次生成可能重复计费。</p>
        {unknownRequests.map((item) => <p key={item.attempt_id}>请求 <code>{item.attempt_id}</code> · 执行 <code>{item.execution_id || "未知"}</code> <button className="secondary-button" disabled={busy || jobActive} onClick={() => acknowledgeRequest(item.attempt_id)}>已核查并确认风险</button></p>)}
      </div>}
      <div className="import-actions">
        <button className="primary-button" disabled={!token || busy || jobActive || (mode === "seeded" && !eligible) || (mode === "prompt_only" && (!topic.trim() || !skill.trim())) || !profiles.length || (profiles.length > 1 && !profileId)} onClick={prepare}>{busy ? "处理中…" : jobActive ? "当前任务完成后再准备" : "准备生成方案"}</button>
        {bundleId && <button className="secondary-button" disabled={busy || jobActive || unknownRequests.length > 0} onClick={checkPlan}>预检方案</button>}
        {preflight && <button className="primary-button" disabled={busy || jobActive || unknownRequests.length > 0 || preflight.status !== "preflight_ready"} onClick={start}>确认并生成</button>}
      </div>
      {bundleId && <p className="dataset-preview-integrity" role="status">方案已保存：<code>{bundleId}</code> · 准备/预检不调用模型。</p>}
      {preflight && <div className="dataset-preview-integrity" role="status">预检通过 · 目标 {preflight.desired_usable_records ?? target} 条 · 最多 {preflight.request_limit ?? requestLimit} 次请求 · 每次最多 {preflight.max_output_tokens_per_request ?? "—"} tokens。费用不可预估时请以 Provider 控制台为准。</div>}
      {job && <div className="sft-generation-job" aria-live="polite"><strong>{jobStatusLabel(job.status)}</strong><span>execution: <code>{job.execution_id}</code></span><span>进度：{job.progress?.completed_candidates ?? 0}/{job.progress?.candidate_target ?? target} 候选 · 请求 {job.progress?.attempted_requests ?? 0}/{job.progress?.request_budget ?? target}</span>{job.detail && <span>{job.detail}</span>}{job.result && <span>可用 {job.result.usable_records ?? 0} 条 · 拒绝 {Math.max(0, (job.result.candidate_count ?? 0) - (job.result.usable_records ?? 0))} 条 · 正确性未独立验证 · 质量状态 {job.result.quality_status || "未评定"}</span>}
        {job.result?.candidate_rejections && Object.keys(job.result.candidate_rejections).length > 0 && <details><summary>查看拒绝原因</summary><pre>{JSON.stringify(job.result.candidate_rejections, null, 2)}</pre></details>}
        <GenerationReviewPanel
          label="SFT"
          canReview={Number(job.result?.usable_records || 0) > 0}
          busy={busy}
          preview={preview ? { records: preview.records || [], count: preview.count, canonical_sha256: preview.canonical_sha256, truncated: preview.truncated } : null}
          reviewed={reviewed}
          registeredDatasetId={job.registered_dataset_id}
          registrationSummary={job.result?.usable_records ? `${job.result.usable_records} 条` : undefined}
          onPreview={loadPreview}
          onReviewedChange={setReviewed}
          onRegister={register}
        />
      </div>}
    </>}
  </section>;
}
