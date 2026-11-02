import { useEffect, useRef, useState } from "react";

type Profile = { profile_id: string; model: string };
type Job = { execution_id: string; bundle_id: string; status: string; progress?: Record<string, number>; result?: Record<string, unknown>; provider_error?: Record<string, unknown>; detail?: string };
const DRAFT_KEY = "onegpu.cpt-generation-draft.v1";

function savedDraft() {
  try {
    const value = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    if (value && typeof value === "object") return value;
  } catch { /* Draft persistence is optional; server state remains authoritative. */ }
  return {};
}

async function api<T>(url: string, token: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", "X-Local-Synthetic-Token": token },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.detail || `请求失败 (${response.status})`);
  return value as T;
}

export function CPTGenerationPanel({ onRegistered }: { onRegistered?: () => void }) {
  const saved = savedDraft();
  const [token, setToken] = useState("");
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [profileId, setProfileId] = useState<string>(typeof saved.profileId === "string" ? saved.profileId : "");
  const [mode, setMode] = useState<"ordinary" | "targeted">(saved.mode === "targeted" ? "targeted" : "ordinary");
  const [domain, setDomain] = useState(typeof saved.domain === "string" ? saved.domain : "数学");
  const [topic, setTopic] = useState(typeof saved.topic === "string" ? saved.topic : "");
  const [tokenBudget, setTokenBudget] = useState(Number.isInteger(saved.tokenBudget) ? saved.tokenBudget : 2048);
  const [requestBudget, setRequestBudget] = useState(Number.isInteger(saved.requestBudget) ? saved.requestBudget : 4);
  const [candidateBudget, setCandidateBudget] = useState(Number.isInteger(saved.candidateBudget) ? saved.candidateBudget : 4);
  const [maxOutputTokens, setMaxOutputTokens] = useState(Number.isInteger(saved.maxOutputTokens) ? saved.maxOutputTokens : 1024);
  const [timeoutSeconds, setTimeoutSeconds] = useState(Number.isInteger(saved.timeoutSeconds) ? saved.timeoutSeconds : 120);
  const [planId, setPlanId] = useState(typeof saved.planId === "string" ? saved.planId : "");
  const [planVersion, setPlanVersion] = useState(Number.isInteger(saved.planVersion) ? saved.planVersion : 1);
  const [bundleId, setBundleId] = useState("");
  const [preflight, setPreflight] = useState<Record<string, unknown> | null>(null);
  const [job, setJob] = useState<Job | null>(() => {
    try { return JSON.parse(sessionStorage.getItem("onegpu.cpt-generation-job.v1") || "null"); } catch { return null; }
  });
  const [preview, setPreview] = useState<Record<string, unknown> | null>(null);
  const [registeredId, setRegisteredId] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const pollRef = useRef<number | null>(null);

  useEffect(() => {
    let alive = true;
    fetch("/api/synthetic/cpt/capabilities").then((response) => response.json()).then((value) => {
      if (!alive) return;
      if (!value.enabled) { setError(value.reason || "CPT 生成未启用"); return; }
      setToken(value.token || ""); setProfiles(value.profiles || []);
      setProfileId((current) => current || value.profiles?.[0]?.profile_id || "");
    }).catch(() => alive && setError("无法读取 CPT 生成能力；请确认本地服务已启用模型生成"));
    return () => { alive = false; if (pollRef.current) window.clearInterval(pollRef.current); };
  }, []);

  useEffect(() => {
    if (job) sessionStorage.setItem("onegpu.cpt-generation-job.v1", JSON.stringify(job));
  }, [job]);

  useEffect(() => {
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ profileId, mode, domain, topic, tokenBudget, requestBudget, candidateBudget, maxOutputTokens, timeoutSeconds, planId, planVersion })); }
    catch { /* A blocked browser storage must not block generation. */ }
  }, [profileId, mode, domain, topic, tokenBudget, requestBudget, candidateBudget, maxOutputTokens, timeoutSeconds, planId, planVersion]);

  useEffect(() => {
    if (!token || !job?.execution_id || !["queued", "running"].includes(job.status)) return;
    let alive = true;
    const poll = async () => {
      try {
        const fresh = await api<Job>(`/api/synthetic/cpt/runs/${encodeURIComponent(job.execution_id)}`, token);
        if (alive) { setJob(fresh); if (["partial", "completed", "failed", "interrupted", "cancelled"].includes(fresh.status)) setError(""); }
      } catch (cause) { if (alive) setError(cause instanceof Error ? cause.message : "读取生成状态失败"); }
    };
    void poll();
    pollRef.current = window.setInterval(poll, 1200);
    return () => { alive = false; if (pollRef.current) window.clearInterval(pollRef.current); };
  }, [token, job?.execution_id, job?.status]);

  const prepare = async () => {
    setBusy(true); setError(""); setPreflight(null); setBundleId(""); setRegisteredId(""); setPreview(null);
    try {
      const value = await api<{ bundle_id: string }>("/api/synthetic/cpt/prepare", token, {
        mode, domain, topic, target_token_budget: tokenBudget, request_budget: requestBudget,
        candidate_budget: candidateBudget, max_output_tokens: maxOutputTokens,
        timeout_seconds: timeoutSeconds, profile_id: profileId,
        ...(mode === "targeted" ? { plan_id: planId, plan_version: planVersion } : {}),
      });
      setBundleId(value.bundle_id);
    } catch (cause) { setError(cause instanceof Error ? cause.message : "方案准备失败"); }
    finally { setBusy(false); }
  };

  const check = async () => {
    setBusy(true); setError("");
    try { setPreflight(await api<Record<string, unknown>>("/api/synthetic/cpt/preflight", token, { bundle_id: bundleId })); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "预检失败"); }
    finally { setBusy(false); }
  };

  const start = async () => {
    setBusy(true); setError("");
    try {
      const value = await api<Job>("/api/synthetic/cpt/start", token, { bundle_id: bundleId, confirm_start: true });
      setJob(value);
    } catch (cause) { setPreflight(null); setError(cause instanceof Error ? cause.message : "启动失败"); }
    finally { setBusy(false); }
  };

  const cancel = async () => {
    if (!job) return;
    setBusy(true); setError("");
    try { setJob(await api<Job>(`/api/synthetic/cpt/runs/${encodeURIComponent(job.execution_id)}/cancel`, token, {})); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "取消请求失败"); }
    finally { setBusy(false); }
  };

  const loadPreview = async () => {
    if (!job) return;
    setBusy(true); setError("");
    try { setPreview(await api<Record<string, unknown>>(`/api/synthetic/cpt/runs/${encodeURIComponent(job.execution_id)}/preview`, token)); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "候选预览失败"); }
    finally { setBusy(false); }
  };

  const register = async () => {
    if (!job || !window.confirm("确认已审阅 CPT 候选？登记会创建不可变的正式数据集；文本事实仍标记为未验证。")) return;
    setBusy(true); setError("");
    try {
      const value = await api<Job>(`/api/synthetic/cpt/runs/${encodeURIComponent(job.execution_id)}/register`, token, { confirm_reviewed: true });
      setJob(value); setRegisteredId(String((value as any).registered_dataset_id || "")); onRegistered?.();
    } catch (cause) { setError(cause instanceof Error ? cause.message : "登记失败"); }
    finally { setBusy(false); }
  };

  const sameBundleAlreadyStarted = Boolean(job && job.bundle_id === bundleId);
  const generationBusy = job?.status === "queued" || job?.status === "running";
  const startDisabled = busy || preflight?.status !== "preflight_ready" || generationBusy || sameBundleAlreadyStarted;
  const startLabel = generationBusy ? "生成任务进行中" : sameBundleAlreadyStarted ? "本方案已执行" : "3. 确认并开始生成";

  return <section className="cpt-generation-panel" aria-labelledby="cpt-generation-title">
    <div className="section-heading"><div><div className="eyebrow">CPT DATA GENERATION</div><h2 id="cpt-generation-title">生成 CPT 继续预训练语料</h2><p>普通生成按主题创建候选；定向生成绑定已登记计划。生成不会自动登记，事实正确性不会因模型生成而被视为已验证。</p></div></div>
    {error && <div className="notice error" role="alert">{error}</div>}
    {!profiles.length && <div className="notice warning">没有找到凭据已配置的本地 Provider。先在本机环境配置凭据，再刷新本页。</div>}
    <div className="mixture-controls">
      <label>生成模式<select value={mode} onChange={(event) => { setMode(event.target.value as "ordinary" | "targeted"); setBundleId(""); setPreflight(null); }}><option value="ordinary">普通生成</option><option value="targeted">定向补数</option></select></label>
      <label>Provider<select value={profileId} onChange={(event) => { setProfileId(event.target.value); setBundleId(""); setPreflight(null); }}>{profiles.map((profile) => <option key={profile.profile_id} value={profile.profile_id}>{profile.profile_id} · {profile.model}</option>)}</select></label>
    </div>
    {mode === "targeted" && <div className="mixture-controls"><label>已登记 CPT 计划 ID<input value={planId} onChange={(event) => { setPlanId(event.target.value); setBundleId(""); setPreflight(null); }} placeholder="plan-targeted-…" /></label><label>计划版本<input type="number" min={1} value={planVersion} onChange={(event) => { setPlanVersion(Math.max(1, Number(event.target.value) || 1)); setBundleId(""); setPreflight(null); }} /></label></div>}
    <div className="mixture-controls"><label>领域<input value={domain} onChange={(event) => { setDomain(event.target.value); setBundleId(""); setPreflight(null); }} /></label><label>子主题<input value={topic} onChange={(event) => { setTopic(event.target.value); setBundleId(""); setPreflight(null); }} placeholder="例如：细胞结构" /></label></div>
    <div className="mixture-controls"><label>目标 token 数<input type="number" min={1} value={tokenBudget} onChange={(event) => setTokenBudget(Math.max(1, Number(event.target.value) || 1))} /></label><label>Provider 请求上限<input type="number" min={1} max={100} value={requestBudget} onChange={(event) => setRequestBudget(Math.max(1, Math.min(100, Number(event.target.value) || 1)))} /></label><label>候选数量上限<input type="number" min={1} max={100} value={candidateBudget} onChange={(event) => setCandidateBudget(Math.max(1, Math.min(100, Number(event.target.value) || 1)))} /></label><label>单次输出 token 上限<input type="number" min={1} max={8192} value={maxOutputTokens} onChange={(event) => setMaxOutputTokens(Math.max(1, Math.min(8192, Number(event.target.value) || 1)))} /></label><label>单次 Provider 超时（秒）<input type="number" min={1} max={600} value={timeoutSeconds} onChange={(event) => setTimeoutSeconds(Math.max(1, Math.min(600, Number(event.target.value) || 1)))} /><small>默认 120 秒，最多 600 秒；超时后本次结果仍可能未知，不会自动重试。</small></label></div>
    <div className="cpt-generation-actions"><button className="secondary-button" disabled={busy || !token || !profileId || !topic.trim() || (mode === "targeted" && !planId.trim())} onClick={prepare}>1. 准备方案</button>{bundleId && <button className="secondary-button" disabled={busy} onClick={check}>2. 只读预检</button>}{bundleId && preflight && <button className="primary-button" disabled={startDisabled} onClick={start}>{startLabel}</button>}</div>
    {bundleId && <p>方案已保存：<code>{bundleId}</code> · 准备与预检均不调用 Provider。</p>}
    {preflight && <div className="notice warning"><strong>预检提示：污染筛查仅覆盖本地 1 条示例，不构成正式污染排除证明。</strong><pre>{JSON.stringify(preflight, null, 2)}</pre></div>}
    {job && <div className="notice info" aria-live="polite"><strong>CPT 生成任务：{job.status}</strong><p>执行 ID：<code>{job.execution_id}</code></p><p>请求 {job.progress?.attempted_requests || 0} · 候选 {job.progress?.accepted_candidates || 0} · token {job.progress?.usable_tokens || 0}</p>{job.detail && <p>{job.detail}</p>}{job.provider_error && <details><summary>Provider 诊断信息（不含凭据）</summary><pre>{JSON.stringify(job.provider_error, null, 2)}</pre></details>}
      {["queued", "running"].includes(job.status) && <button className="secondary-button" disabled={busy} onClick={cancel}>请求停止（当前 Provider 请求结束后生效）</button>}
      {!(["queued", "running"].includes(job.status)) && job.result && Number(job.result.candidate_count || 0) > 0 && <div className="cpt-generation-actions"><button className="secondary-button" disabled={busy} onClick={loadPreview}>查看候选并审核</button><button className="primary-button" disabled={busy || !preview || Boolean(registeredId)} onClick={register}>确认审核并登记</button></div>}
      {!(["queued", "running"].includes(job.status)) && job.result && Number(job.result.candidate_count || 0) < 1 && <p>没有可审核的有效候选；此任务不能登记，请检查主题和请求拒绝原因后准备新方案。</p>}
      {registeredId && <p>已登记数据集：<code>{registeredId}</code>。可在训练与评测工作台选择 CPT 阶段接续训练。</p>}
    </div>}
    {preview && <div className="notice info"><strong>候选预览 · 事实正确性未验证</strong><pre>{JSON.stringify(preview, null, 2)}</pre></div>}
  </section>;
}
