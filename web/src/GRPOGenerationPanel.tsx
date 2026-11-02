import { useEffect, useState } from "react";

type Profile = { profile_id: string; model?: string; revision?: string | null };
type Plan = { plan_id: string; plan_version: number; scenario_id: string; training_seed_dataset_id: string; max_requests: number; max_candidates: number; max_output_tokens: number; timeout_seconds: number };
type Snapshot = { execution_id: string; status: string; mode: string; result?: Record<string, unknown>; progress?: Record<string, number>; detail?: string };

const DRAFT_KEY = "onegpu.grpo-generation-draft.v1";
const RUN_KEY = "onegpu.grpo-generation-run.v1";

function draft() {
  try {
    const value = JSON.parse(window.localStorage.getItem(DRAFT_KEY) || "null");
    if (value && typeof value === "object") return value;
  } catch { /* Storage is a convenience, the server remains authoritative. */ }
  return { mode: "ordinary", profileId: "", planKey: "", prompts: "", requests: 3, candidates: 3, tokens: 256, timeoutSeconds: 120 };
}

export function GRPOGenerationPanel() {
  const saved = draft();
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [plans, setPlans] = useState<Plan[]>([]);
  const [warning, setWarning] = useState("");
  const [token, setToken] = useState("");
  const [mode, setMode] = useState<"ordinary" | "targeted">(saved.mode === "targeted" ? "targeted" : "ordinary");
  const [profileId, setProfileId] = useState(typeof saved.profileId === "string" ? saved.profileId : "");
  const [planKey, setPlanKey] = useState(typeof saved.planKey === "string" ? saved.planKey : "");
  const [prompts, setPrompts] = useState(typeof saved.prompts === "string" ? saved.prompts : "");
  const [requests, setRequests] = useState(Number.isInteger(saved.requests) ? saved.requests : 3);
  const [candidates, setCandidates] = useState(Number.isInteger(saved.candidates) ? saved.candidates : 3);
  const [tokens, setTokens] = useState(Number.isInteger(saved.tokens) ? saved.tokens : 256);
  const [timeoutSeconds, setTimeoutSeconds] = useState(Number.isInteger(saved.timeoutSeconds) ? saved.timeoutSeconds : 120);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [prepared, setPrepared] = useState<Record<string, unknown> | null>(null);
  const [preflight, setPreflight] = useState<Record<string, unknown> | null>(null);
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null);
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const selectedPlan = plans.find((item) => `${item.plan_id}@${item.plan_version}` === planKey);

  useEffect(() => {
    try { window.localStorage.setItem(DRAFT_KEY, JSON.stringify({ mode, profileId, planKey, prompts, requests, candidates, tokens, timeoutSeconds })); }
    catch { /* Do not fail the workflow when browser storage is disabled. */ }
  }, [mode, profileId, planKey, prompts, requests, candidates, tokens, timeoutSeconds]);

  useEffect(() => {
    let alive = true;
    fetch("/api/synthetic/grpo/capabilities").then(async (response) => {
      const value = await response.json();
      if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : `GRPO 能力读取失败 (${response.status})`);
      return value;
    }).then((value) => {
      if (!alive) return;
      setProfiles(value.profiles || []); setPlans(value.targeted_plans || []);
      setToken(value.token || ""); setWarning(value.warning || "");
      if (!profileId && value.profiles?.length) setProfileId(value.profiles[0].profile_id);
    }).catch((reason: unknown) => { if (alive) setError(reason instanceof Error ? reason.message : "GRPO 能力读取失败"); });
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    try {
      const value = JSON.parse(window.localStorage.getItem(RUN_KEY) || "null");
      if (value && typeof value.execution_id === "string") {
        setSnapshot({ execution_id: value.execution_id, status: "restoring", mode: value.mode });
      }
    } catch { /* Ignore malformed local run pointer. */ }
  }, []);

  useEffect(() => {
    if (!snapshot?.execution_id || !token || ["completed", "partial", "failed", "interrupted", "cancelled"].includes(snapshot.status)) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const response = await fetch(`/api/synthetic/grpo/runs/${encodeURIComponent(snapshot.execution_id)}`, {
          headers: { "X-Local-Synthetic-Token": token },
        });
        const value = await response.json();
        if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : "任务状态读取失败");
        if (!cancelled) setSnapshot(value);
      } catch (reason) { if (!cancelled) setError(reason instanceof Error ? reason.message : "任务状态读取失败"); }
    };
    void poll();
    const interval = window.setInterval(() => void poll(), 2000);
    return () => { cancelled = true; window.clearInterval(interval); };
  }, [snapshot?.execution_id, snapshot?.status, token]);

  async function api(path: string, payload?: Record<string, unknown>) {
    const response = await fetch(path, {
      method: payload ? "POST" : "GET",
      headers: { "X-Local-Synthetic-Token": token, ...(payload ? { "Content-Type": "application/json" } : {}) },
      ...(payload ? { body: JSON.stringify(payload) } : {}),
    });
    const value = await response.json();
    if (!response.ok) throw new Error(typeof value.detail === "string" ? value.detail : `请求失败 (${response.status})`);
    return value;
  }

  async function prepare() {
    setBusy(true); setError(""); setPrepared(null); setPreflight(null); setPreview(null);
    try {
      const base = { mode, profile_id: profileId, request_budget: requests, candidate_budget: candidates, max_output_tokens: tokens, timeout_seconds: timeoutSeconds };
      const payload = mode === "ordinary"
        ? { ...base, prompts: prompts.split(/\r?\n/).map((item: string) => item.trim()).filter(Boolean) }
        : { ...base, stage_plan_id: selectedPlan?.plan_id, plan_version: selectedPlan?.plan_version,
            timeout_seconds: Math.min(timeoutSeconds, selectedPlan?.timeout_seconds ?? timeoutSeconds) };
      const value = await api("/api/synthetic/grpo/prepare", payload);
      setPrepared(value);
      const checked = await api("/api/synthetic/grpo/preflight", { bundle_id: value.bundle_id });
      setPreflight(checked);
    } catch (reason) { setError(reason instanceof Error ? reason.message : "GRPO 方案准备/预检失败"); }
    finally { setBusy(false); }
  }

  async function start() {
    if (!prepared || !preflight || !window.confirm("确认调用 Provider 生成 GRPO prompt 候选？候选没有可信 reference，不会登记成可训练数据。")) return;
    setBusy(true); setError("");
    try {
      const value = await api("/api/synthetic/grpo/start", { bundle_id: prepared.bundle_id, confirm_start: true });
      setSnapshot(value);
      window.localStorage.setItem(RUN_KEY, JSON.stringify({ execution_id: value.execution_id, mode }));
    } catch (reason) { setError(reason instanceof Error ? reason.message : "GRPO 生成启动失败"); }
    finally { setBusy(false); }
  }

  async function loadPreview() {
    if (!snapshot) return;
    setBusy(true); setError("");
    try { setPreview(await api(`/api/synthetic/grpo/runs/${encodeURIComponent(snapshot.execution_id)}/preview`)); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "候选预览读取失败"); }
    finally { setBusy(false); }
  }

  async function cancel() {
    if (!snapshot || !window.confirm("确认请求停止 GRPO 候选生成？当前正在进行的单次 Provider 请求可能仍会完成。")) return;
    setBusy(true); setError("");
    try { setSnapshot(await api(`/api/synthetic/grpo/runs/${encodeURIComponent(snapshot.execution_id)}/cancel`, {})); }
    catch (reason) { setError(reason instanceof Error ? reason.message : "取消请求失败"); }
    finally { setBusy(false); }
  }

  const candidatesList = Array.isArray(preview?.candidates) ? preview.candidates as Array<Record<string, unknown>> : [];
  return <section className="grpo-generation-panel" aria-labelledby="grpo-generation-title">
    <div className="dataset-import-heading"><div><div className="eyebrow">GRPO PROMPT CANDIDATES</div><h2 id="grpo-generation-title">GRPO 模型生成候选</h2><p>普通模式从主题生成；定向模式绑定计划和精确 train_seed。候选仅是题目，不含可信参考答案或 reward。</p></div><div className="grpo-generation-heading-actions"><span className="dataset-limit-badge">不自动登记 / 不可直接训练</span><button className="secondary-button" type="button" onClick={() => document.getElementById("grpo-seed-heading")?.scrollIntoView({ behavior: "smooth", block: "start" })}>改用可信种子派生 ↓</button></div></div>
    {warning && <p className="mixture-lineage-note">{warning}</p>}
    <div className="import-steps" aria-label="GRPO 生成步骤">{["选择模式和来源", "准备并只读预检", "确认生成", "查看候选与缺失证据"].map((label, index) => <div key={label} className={snapshot ? index < 3 ? "done" : "current" : preflight ? index < 2 ? "done" : "current" : index === 0 ? "current" : ""}><span>{index + 1}</span>{label}</div>)}</div>
    {error && <div className="dataset-workbench-alert" role="alert">{error}</div>}
    <div className="dataset-import-form">
      <div className="import-metadata-grid">
        <label>生成方式<select value={mode} onChange={(event) => { setMode(event.target.value as "ordinary" | "targeted"); setPrepared(null); setPreflight(null); }}><option value="ordinary">普通生成（主题提示）</option><option value="targeted">定向补数（已登记 GRPO 计划）</option></select></label>
        <label>Provider<select value={profileId} onChange={(event) => { setProfileId(event.target.value); setPrepared(null); setPreflight(null); }}><option value="">选择可用 Provider</option>{profiles.map((profile) => <option key={profile.profile_id} value={profile.profile_id}>{profile.profile_id} · {profile.model}</option>)}</select></label>
        {mode === "ordinary" ? <label className="import-wide">主题/种子提示（每行一个，最多 50 行）<textarea rows={4} value={prompts} onChange={(event) => setPrompts(event.target.value)} placeholder={"例如：\n整数方程的应用题\n分数与比例的多步推理"} /></label> : <label className="import-wide">定向计划<select value={planKey} onChange={(event) => { setPlanKey(event.target.value); setPrepared(null); setPreflight(null); const plan = plans.find((item) => `${item.plan_id}@${item.plan_version}` === event.target.value); if (plan) setTimeoutSeconds(Math.min(timeoutSeconds, plan.timeout_seconds)); }}><option value="">选择 GRPO 定向计划</option>{plans.map((plan) => <option key={`${plan.plan_id}@${plan.plan_version}`} value={`${plan.plan_id}@${plan.plan_version}`}>{plan.scenario_id} · {plan.plan_id} v{plan.plan_version} · {plan.training_seed_dataset_id}</option>)}</select><small>只读取计划绑定数据集中的 train_seed；诊断和 holdout 内容不会成为生成输入。</small></label>}
        <label>最大请求数<input type="number" min={1} max={100} value={requests} onChange={(event) => setRequests(Number(event.target.value))} /></label>
        <label>候选数<input type="number" min={1} max={100} value={candidates} onChange={(event) => setCandidates(Number(event.target.value))} /></label>
        <label>单次输出 token<input type="number" min={1} max={2048} value={tokens} onChange={(event) => setTokens(Number(event.target.value))} /></label>
        <label>单次 Provider 超时（秒）<input type="number" min={1} max={600} value={timeoutSeconds} onChange={(event) => setTimeoutSeconds(Math.max(1, Math.min(600, Number(event.target.value) || 1)))} /><small>默认 120 秒；超时后结果可能未知，不会自动重试。</small></label>
      </div>
      <p className="mixture-lineage-note">预检不会调用 Provider。生成结果的 reference 状态仍为 missing；需要可信答案来源和 reward 契约验证后才能建立可训练数据集。</p>
      <div className="import-actions"><button className="secondary-button" disabled={busy || !token || !profileId || (mode === "ordinary" ? !prompts.trim() : !selectedPlan)} onClick={() => void prepare()}>{busy ? "处理中…" : "准备方案并预检"}</button>{prepared && preflight && <button className="primary-button" disabled={busy} onClick={() => void start()}>确认并生成候选</button>}</div>
    </div>
    {snapshot && <div className="dataset-import-success" role="status"><div className="success-mark">•</div><div><h3>GRPO 候选任务：{snapshot.status}</h3><p><code>{snapshot.execution_id}</code> · {String(snapshot.result?.candidate_count ?? snapshot.progress?.candidates ?? 0)} 条候选 · 不是可训练数据</p>{snapshot.detail && <small>{snapshot.detail}</small>}{Array.isArray(snapshot.result?.provider_diagnostics) && <details><summary>Provider 诊断信息（不含凭据）</summary><pre>{JSON.stringify(snapshot.result.provider_diagnostics, null, 2)}</pre></details>}</div>{["queued", "running"].includes(snapshot.status) && <button className="secondary-button" disabled={busy} onClick={() => void cancel()}>请求取消</button>}{["completed", "partial", "cancelled"].includes(snapshot.status) && <button className="secondary-button" disabled={busy} onClick={() => void loadPreview()}>查看候选</button>}</div>}
    {candidatesList.length > 0 && <div className="grpo-candidate-list"><h3>候选预览 · reference 缺失</h3>{candidatesList.map((candidate) => <article key={String(candidate.candidate_id)}><strong>{String(candidate.candidate_id)}</strong><p>{String(candidate.raw_problem)}</p><small>来源记录：{String(candidate.source_record_id)} · {String(candidate.reference_status)} · 未登记</small></article>)}<p className="mixture-lineage-note">当前没有把候选伪装成 GRPO 训练集。需要先通过可信种子派生路径，或补齐独立可信 reference/验证证据。</p></div>}
  </section>;
}
