import { useEffect, useMemo, useState } from "react";

type Profile = { profile_id: string; model: string };
type TargetPlan = { plan_id: string; plan_version: number; scenario_id: string; training_seed_dataset_id: string; request_budget: number; candidate_budget: number; max_output_tokens: number; timeout_seconds: number };
type Job = { execution_id: string; bundle_id: string; status: string; mode: string; result?: Record<string, unknown>; registration?: Record<string, unknown>; review?: Record<string, unknown>; detail?: string };
type Candidate = { candidate_id: string; problem_id: string; raw_problem?: string; problem?: string; completion: string };
const STORAGE_KEY = "onegpu.preference-generation.execution.v1";
const DRAFT_KEY = "onegpu.preference-generation-draft.v1";

function savedDraft() {
  try {
    const value = JSON.parse(localStorage.getItem(DRAFT_KEY) || "null");
    if (value && typeof value === "object") return value;
  } catch { /* Draft persistence is optional; server state remains authoritative. */ }
  return {};
}

async function api<T>(path: string, token: string, body?: unknown): Promise<T> {
  const response = await fetch(path, { method: body === undefined ? "GET" : "POST", headers: { "Content-Type": "application/json", "X-Local-Synthetic-Token": token }, ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.detail || `请求失败 (${response.status})`);
  return value as T;
}

export function PreferenceGenerationPanel({ onRegistered }: { onRegistered?: () => void }) {
  const saved = savedDraft();
  const [token, setToken] = useState("");
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [targetPlans, setTargetPlans] = useState<TargetPlan[]>([]);
  const [mode, setMode] = useState<"ordinary" | "targeted">(saved.mode === "targeted" ? "targeted" : "ordinary");
  const [prompts, setPrompts] = useState<string>(typeof saved.prompts === "string" ? saved.prompts : "");
  const [profileId, setProfileId] = useState<string>(typeof saved.profileId === "string" ? saved.profileId : "");
  const [targetPlanId, setTargetPlanId] = useState<string>(typeof saved.targetPlanId === "string" ? saved.targetPlanId : "");
  const [requestBudget, setRequestBudget] = useState(Number.isInteger(saved.requestBudget) ? saved.requestBudget : 4);
  const [candidateBudget, setCandidateBudget] = useState(Number.isInteger(saved.candidateBudget) ? saved.candidateBudget : 4);
  const [maxOutputTokens, setMaxOutputTokens] = useState(Number.isInteger(saved.maxOutputTokens) ? saved.maxOutputTokens : 512);
  const [timeoutSeconds, setTimeoutSeconds] = useState(Number.isInteger(saved.timeoutSeconds) ? saved.timeoutSeconds : 120);
  const [bundleId, setBundleId] = useState("");
  const [preflight, setPreflight] = useState<Record<string, unknown> | null>(null);
  const [job, setJob] = useState<Job | null>(() => { try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "null"); } catch { return null; } });
  const [preview, setPreview] = useState<Record<string, any> | null>(null);
  const [reviewerId, setReviewerId] = useState("");
  const [rationale, setRationale] = useState("");
  const [chosen, setChosen] = useState<Record<string, string>>({});
  const [rejected, setRejected] = useState<Record<string, string>>({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [registered, setRegistered] = useState("");
  const selectedPlan = useMemo(() => targetPlans.find((row) => `${row.plan_id}:v${row.plan_version}` === targetPlanId), [targetPlans, targetPlanId]);

  useEffect(() => {
    let alive = true;
    fetch("/api/synthetic/preference/capabilities").then((response) => response.json()).then((value) => {
      if (!alive) return;
      if (!value.enabled) { setError(value.reason || "Preference 生成未启用"); return; }
      setToken(value.token || ""); setProfiles(value.profiles || []); setTargetPlans(value.targeted_plans || []);
      setProfileId((current) => current || value.profiles?.[0]?.profile_id || "");
      setTargetPlanId((current: string) => current || (value.targeted_plans?.[0] ? `${value.targeted_plans[0].plan_id}:v${value.targeted_plans[0].plan_version}` : ""));
    }).catch(() => alive && setError("无法读取 Preference 生成能力；请确认本地服务已启用模型生成"));
    return () => { alive = false; };
  }, []);

  useEffect(() => { if (job) localStorage.setItem(STORAGE_KEY, JSON.stringify(job)); }, [job]);
  useEffect(() => {
    try { localStorage.setItem(DRAFT_KEY, JSON.stringify({ mode, prompts, profileId, targetPlanId, requestBudget, candidateBudget, maxOutputTokens, timeoutSeconds })); }
    catch { /* A blocked browser storage must not block generation. */ }
  }, [mode, prompts, profileId, targetPlanId, requestBudget, candidateBudget, maxOutputTokens, timeoutSeconds]);
  useEffect(() => {
    if (!token || !job?.execution_id || !["queued", "running"].includes(job.status)) return;
    let alive = true;
    const poll = async () => { try { const value = await api<Job>(`/api/synthetic/preference/runs/${encodeURIComponent(job.execution_id)}`, token); if (alive) { setJob(value); if (["partial", "completed", "failed", "interrupted", "cancelled"].includes(value.status)) setError(""); } } catch (reason) { if (alive) setError(reason instanceof Error ? reason.message : "读取生成状态失败"); } };
    void poll(); const timer = window.setInterval(poll, 1200);
    return () => { alive = false; window.clearInterval(timer); };
  }, [token, job?.execution_id, job?.status]);

  const run = async (action: () => Promise<void>) => { setBusy(true); setError(""); try { await action(); } catch (reason) { setError(reason instanceof Error ? reason.message : "Preference 操作失败"); } finally { setBusy(false); } };
  const prepare = () => run(async () => {
    setPreflight(null); setPreview(null); setRegistered("");
    const body = mode === "ordinary"
      ? { mode, profile_id: profileId, prompts: prompts.split(/\r?\n/).map((item) => item.trim()).filter(Boolean), request_budget: requestBudget, candidate_budget: candidateBudget, max_output_tokens: maxOutputTokens, timeout_seconds: timeoutSeconds }
      : { mode, profile_id: profileId, stage_plan_id: selectedPlan?.plan_id, plan_version: selectedPlan?.plan_version, request_budget: requestBudget, candidate_budget: candidateBudget, max_output_tokens: maxOutputTokens, timeout_seconds: Math.min(timeoutSeconds, selectedPlan?.timeout_seconds ?? timeoutSeconds) };
    const value = await api<{ bundle_id: string }>("/api/synthetic/preference/prepare", token, body);
    setBundleId(value.bundle_id);
  });
  const check = () => run(async () => setPreflight(await api("/api/synthetic/preference/preflight", token, { bundle_id: bundleId })));
  const start = () => run(async () => {
    try {
      const value = await api<Job>("/api/synthetic/preference/start", token, { bundle_id: bundleId, confirm_start: true });
      setJob(value); localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
    } catch (reason) {
      setPreflight(null);
      throw reason;
    }
  });
  const loadPreview = () => run(async () => setPreview(await api(`/api/synthetic/preference/runs/${encodeURIComponent(job!.execution_id)}/preview`, token)));
  const review = () => run(async () => {
    if (!preview) return;
    const grouped = new Map<string, Candidate[]>();
    for (const item of preview.candidates as Candidate[]) grouped.set(item.problem_id, [...(grouped.get(item.problem_id) || []), item]);
    const reviews = [...grouped.entries()].filter(([id]) => chosen[id] && rejected[id]).map(([problem_id]) => ({ problem_id, chosen_candidate_id: chosen[problem_id], rejected_candidate_id: rejected[problem_id], rationale }));
    setJob(await api<Job>(`/api/synthetic/preference/runs/${encodeURIComponent(job!.execution_id)}/review`, token, { reviewer_id: reviewerId, reviews }));
  });
  const register = () => run(async () => {
    const value = await api<Job>(`/api/synthetic/preference/runs/${encodeURIComponent(job!.execution_id)}/register`, token, { confirm_reviewed: true });
    setJob(value); setRegistered(String(value.registration?.dataset_id || "")); onRegistered?.();
  });

  const statusName: Record<string, string> = { queued: "排队中", running: "正在生成", completed: "生成完成", partial: "部分完成", failed: "生成失败", interrupted: "服务中断，未自动重试" };
  const sameBundleAlreadyStarted = Boolean(job && job.bundle_id === bundleId);
  const generationBusy = job?.status === "queued" || job?.status === "running";
  const startDisabled = busy || preflight?.status !== "preflight_ready" || generationBusy || sameBundleAlreadyStarted;
  const startLabel = generationBusy ? "生成任务进行中" : sameBundleAlreadyStarted ? "本方案已执行" : "3. 确认并开始生成";
  return <section className="cpt-generation-panel preference-generation-panel" aria-labelledby="preference-generation-title">
    <div className="section-heading"><div><div className="eyebrow">PREFERENCE DATA GENERATION</div><h2 id="preference-generation-title">生成 Preference 偏好数据</h2><p>普通模式由用户审阅同题回答排序；定向模式只从已登记计划的 train_seed 派生，并以独立答案校验排序。生成不自动登记，也不会把 Provider 自报标签当证据。</p></div></div>
    <ol className="preference-steps"><li>配置来源与预算</li><li>准备只读方案</li><li>预检（零 Provider 调用）</li><li>确认后生成</li><li>预览并审核排序</li><li>显式登记并用于 DPO / SimPO</li></ol>
    <div className="mixture-controls"><label>生成方式<select value={mode} onChange={(event) => { setMode(event.target.value as typeof mode); setBundleId(""); setPreflight(null); }}><option value="ordinary">普通生成：用户提供题目，生成未排序候选</option><option value="targeted">定向补数：按已登记目标计划生成</option></select></label>
      <label>Provider 配置<select value={profileId} onChange={(event) => setProfileId(event.target.value)}><option value="">选择 Provider</option>{profiles.map((item) => <option key={item.profile_id} value={item.profile_id}>{item.profile_id} · {item.model}</option>)}</select></label>
      {mode === "ordinary" ? <label className="import-wide">题目（每行一道，最多 100 道）<textarea rows={4} value={prompts} onChange={(event) => setPrompts(event.target.value)} placeholder="例如：解释一元一次方程的解法" /></label> : <label className="import-wide">定向计划<select value={targetPlanId} onChange={(event) => { setTargetPlanId(event.target.value); if (event.target.value) { const found = targetPlans.find((row) => `${row.plan_id}:v${row.plan_version}` === event.target.value); if (found) { setRequestBudget(found.request_budget); setCandidateBudget(found.candidate_budget); setMaxOutputTokens(found.max_output_tokens); setTimeoutSeconds(Math.min(timeoutSeconds, found.timeout_seconds)); } } }}><option value="">选择已登记 Preference 定向计划</option>{targetPlans.map((item) => <option key={`${item.plan_id}:v${item.plan_version}`} value={`${item.plan_id}:v${item.plan_version}`}>{item.scenario_id} · {item.training_seed_dataset_id} · v{item.plan_version}</option>)}</select></label>}
      <label>Provider 请求上限<input type="number" min={1} max={1000} value={requestBudget} onChange={(event) => setRequestBudget(Number(event.target.value))} /></label><label>候选条数上限<input type="number" min={1} max={500} value={candidateBudget} onChange={(event) => setCandidateBudget(Number(event.target.value))} /></label><label>每次输出 token 上限<input type="number" min={1} max={4096} value={maxOutputTokens} onChange={(event) => setMaxOutputTokens(Number(event.target.value))} /></label><label>单次 Provider 超时（秒）<input type="number" min={1} max={600} value={timeoutSeconds} onChange={(event) => setTimeoutSeconds(Math.max(1, Math.min(600, Number(event.target.value) || 1)))} /><small>默认 120 秒；定向模式不能超过已登记计划上限。</small></label>
    </div>
    <p className="notice">普通模式最终排序由你确认，审阅者身份为用户声明且不会被独立认证；定向模式只检查最终答案与可信参考答案，不证明推理过程质量。登记数据仍会标记探索性质量状态及未核验许可。</p>
    <div className="cpt-generation-actions"><button className="secondary-button" disabled={busy || !token || !profileId || (mode === "ordinary" ? !prompts.trim() : !selectedPlan)} onClick={() => void prepare()}>1. 准备方案</button>{bundleId && <button className="secondary-button" disabled={busy} onClick={() => void check()}>2. 只读预检</button>}{bundleId && preflight && <button className="primary-button" disabled={startDisabled} onClick={() => void start()}>{startLabel}</button>}</div>
    {preflight && <div className="mixture-preflight" role="status"><strong>预检通过 · Provider 调用 0 次</strong><p>方案 <code>{bundleId}</code> · 上限 {String(preflight.request_budget)} 请求 / {String(preflight.candidate_budget)} 候选{preflight.parent_dataset_ref ? ` · 父数据集 ${String((preflight.parent_dataset_ref as Record<string, unknown>).dataset_id)}` : ""}</p></div>}
    {job && <div className="mixture-preflight" role="status"><strong>{statusName[job.status] || job.status}</strong><p>执行 ID：<code>{job.execution_id}</code>{job.detail ? ` · ${job.detail}` : ""}</p>{Array.isArray(job.result?.provider_diagnostics) && <details><summary>Provider 诊断信息（不含凭据）</summary><pre>{JSON.stringify(job.result.provider_diagnostics, null, 2)}</pre></details>}</div>}
    {job && !["queued", "running", "failed", "interrupted"].includes(job.status) && <div className="cpt-generation-actions"><button className="secondary-button" disabled={busy} onClick={() => void loadPreview()}>4. 查看候选与排序证据</button>{preview && mode === "ordinary" && <>
      <label>审阅者标识<input value={reviewerId} onChange={(event) => setReviewerId(event.target.value)} placeholder="例如：本地用户" /></label><label>排序理由（至少 8 字）<input value={rationale} onChange={(event) => setRationale(event.target.value)} /></label>
      <div className="preference-review-list">{[...new Map((preview.candidates as Candidate[]).map((item) => [item.problem_id, item])).values()].map((problem) => { const options = (preview.candidates as Candidate[]).filter((item) => item.problem_id === problem.problem_id); return <fieldset key={problem.problem_id}><legend>{problem.raw_problem}</legend>{options.map((item) => <div key={item.candidate_id}><p>{item.completion}</p><label><input type="radio" name={`chosen-${problem.problem_id}`} checked={chosen[problem.problem_id] === item.candidate_id} onChange={() => setChosen({ ...chosen, [problem.problem_id]: item.candidate_id })} /> Chosen</label><label><input type="radio" name={`rejected-${problem.problem_id}`} checked={rejected[problem.problem_id] === item.candidate_id} onChange={() => setRejected({ ...rejected, [problem.problem_id]: item.candidate_id })} /> Rejected</label></div>)}</fieldset>; })}</div>
      <button className="secondary-button" disabled={busy || !reviewerId.trim() || rationale.trim().length < 8} onClick={() => void review()}>5. 保存人工排序（允许部分审核）</button>
    </>}{preview && mode === "targeted" && <p>独立校验后可用偏好对：{String((preview.pair_report as Record<string, unknown> | undefined)?.usable_pairs ?? 0)}。未达到目标时保留 partial，不补造标签。</p>}{preview && <button className="primary-button" disabled={busy || (mode === "ordinary" ? !job.review?.usable_pairs : !Number((preview.pair_report as Record<string, unknown> | undefined)?.usable_pairs)) || Boolean(registered)} onClick={() => void register()}>6. 确认审核并登记</button>}</div>}
    {registered && <div className="dataset-import-success" role="status"><div className="success-mark">✓</div><div><h3>Preference 数据集已登记，可供 DPO / SimPO 训练选择</h3><p><code>{registered}</code></p></div></div>}
    {error && <div className="dataset-workbench-alert" role="alert">{error}</div>}
  </section>;
}
