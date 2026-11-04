import { useEffect, useMemo, useState } from "react";

type Profile = { profile_id: string; name: string; role_instruction: string; built_in: boolean };
type Message = { message_id: string; role: "user" | "assistant"; content: string; evidence_ids: string[]; created_at: string; message_kind?: string };
type Session = { session_id: string; title: string; assistant_profile_id: string; updated_at: string; data_egress_consent_at?: string | null; messages?: Message[] };
type Evidence = { evidence_id: string; kind: string; entity_id: string; title: string; status: string; summary: string; limitations: string[]; source_href: string; facts?: { key: string; value: unknown }[] };
type Answer = { conclusion: string; facts: { text: string; evidence_ids: string[] }[]; interpretations: { text: string; evidence_ids: string[] }[]; limitations: string[]; clarification_options: string[] };
type Turn = { task: { session_id: string; status: string; error_code?: string | null }; answer: Answer | null; evidence: Evidence[]; consent_required: boolean };
type Status = { provider: string; model_profile_id: string; provider_configured: boolean; data_egress_consent: boolean; read_only: boolean; runtime?: string; observability?: { configured: boolean; state: string; detail: string; base_url: string } };
type Task = { task_id: string; status: string; created_at: string; updated_at: string; error_message?: string | null; error_code?: string | null; duration_ms?: number | null; evidence: Evidence[]; memory_ids?: string[]; context_summary_id?: string | null; langfuse_trace_id?: string | null; langfuse_trace_url?: string | null };
type EvaluationReport = { evaluation_id: string; harness_id: string; suite_version: string; configuration_id: string; sample_count: number; passed: number; failed: number; created_at: string; failed_cases: { case_id: string; findings: { check: string; passed: boolean; detail: string }[] }[] };
type Memory = { memory_id: string; content: string; status: "active" | "disabled"; scope: "workspace"; updated_at: string };
type MemoryDraft = { draft: Memory; saved: false; confirmation_required: true };
type Summary = { summary_id: string; revision: number; summary: string; source_message_ids: string[]; updated_at: string };

async function api<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, { ...options, headers: { ...(options.body ? { "Content-Type": "application/json" } : {}), ...options.headers } });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `请求失败（HTTP ${response.status}）`);
  return body as T;
}

function parseAnswer(message: Message): Answer | null {
  if (message.role !== "assistant") return null;
  try { return JSON.parse(message.content) as Answer; } catch { return null; }
}

export function WorkspaceAssistant({ onOpenRun, onOpenData }: { onOpenRun: (id: string) => void; onOpenData: () => void }) {
  const [profiles, setProfiles] = useState<Profile[]>([]);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [session, setSession] = useState<Session | null>(null);
  const [profileId, setProfileId] = useState("workspace-analyst");
  const [status, setStatus] = useState<Status | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [evaluationReports, setEvaluationReports] = useState<EvaluationReport[]>([]);
  const [memories, setMemories] = useState<Memory[]>([]);
  const [memoryInput, setMemoryInput] = useState("");
  const [memoryDraft, setMemoryDraft] = useState<Memory | null>(null);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [summaryEdit, setSummaryEdit] = useState("");
  const [summarySaving, setSummarySaving] = useState(false);
  const [evaluationBusy, setEvaluationBusy] = useState(false);
  const [question, setQuestion] = useState("");
  const [pendingQuestion, setPendingQuestion] = useState("");
  const [turn, setTurn] = useState<Turn | null>(null);
  const [name, setName] = useState("");
  const [role, setRole] = useState("");
  const [profileDescription, setProfileDescription] = useState("");
  const [editingProfile, setEditingProfile] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const reload = async () => {
    const [nextProfiles, nextSessions] = await Promise.all([
      api<Profile[]>("/api/assistant/profiles"), api<Session[]>("/api/assistant/sessions?limit=50"),
    ]);
    setProfiles(nextProfiles); setSessions(nextSessions);
    setMemories(await api<Memory[]>("/api/assistant/memories"));
  };
  useEffect(() => { reload().catch((e) => setError(e.message)); }, []);
  useEffect(() => { api<EvaluationReport[]>("/api/assistant/evaluations?limit=5").then(setEvaluationReports).catch((e) => setError(e.message)); }, []);
  useEffect(() => {
    api<Status>(`/api/assistant/status${session ? `?session_id=${encodeURIComponent(session.session_id)}` : ""}`)
      .then(setStatus).catch((e) => setError(e.message));
  }, [session?.session_id, session?.data_egress_consent_at]);
  useEffect(() => {
    if (!session) { setTasks([]); return; }
    api<Task[]>(`/api/assistant/tasks?session_id=${encodeURIComponent(session.session_id)}&limit=20`)
      .then(setTasks).catch((e) => setError(e.message));
  }, [session?.session_id, turn?.task.status]);

  const startSession = () => { setSession(null); setSummary(null); setSummaryEdit(""); setTurn(null); setError(""); };
  const openSession = async (id: string) => {
    setError(""); setTurn(null);
    try {
      const saved = await api<Session>(`/api/assistant/sessions/${encodeURIComponent(id)}`);
      setSession(saved); setProfileId(saved.assistant_profile_id);
      const savedSummary = await api<Summary | null>(`/api/assistant/sessions/${encodeURIComponent(id)}/summary`);
      setSummary(savedSummary); setSummaryEdit(savedSummary?.summary || "");
    } catch (e) { setError((e as Error).message); }
  };
  const submit = async (text = question, existingSession = session, allowBusy = false) => {
    if (!text.trim() || (busy && !allowBusy)) return;
    setBusy(true); setError(""); setPendingQuestion(text.trim());
    try {
      const result = await api<Turn>("/api/assistant/ask", { method: "POST", body: JSON.stringify({ question: text.trim(), session_id: existingSession?.session_id || null, assistant_profile_id: profileId }) });
      setTurn(result);
      const saved = await api<Session>(`/api/assistant/sessions/${encodeURIComponent(result.task.session_id)}`);
      setSession(saved); setQuestion(""); await reload();
      const nextSummary = await api<Summary | null>(`/api/assistant/sessions/${encodeURIComponent(result.task.session_id)}/summary`);
      setSummary(nextSummary); setSummaryEdit(nextSummary?.summary || "");
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const confirmAndContinue = async () => {
    if (!session) return;
    setBusy(true); setError("");
    try {
      await api(`/api/assistant/sessions/${encodeURIComponent(session.session_id)}/consent`, { method: "POST" });
      const saved = await api<Session>(`/api/assistant/sessions/${encodeURIComponent(session.session_id)}`);
      setSession(saved);
      setBusy(false);
      await submit(pendingQuestion, saved, true);
    } catch (e) { setError((e as Error).message); setBusy(false); }
  };
  const saveProfile = async () => {
    setBusy(true); setError("");
    try {
      const path = editingProfile ? `/api/assistant/profiles/${encodeURIComponent(editingProfile)}` : "/api/assistant/profiles";
      const profile = await api<Profile>(path, { method: editingProfile ? "PUT" : "POST", body: JSON.stringify({ name, role_instruction: role }) });
      await reload(); setProfileId(profile.profile_id); setName(""); setRole(""); setEditingProfile(null);
    } catch (e) { setError((e as Error).message); }
    finally { setBusy(false); }
  };
  const draftProfile = async () => {
    try {
      const result = await api<{ draft: { name: string; role_instruction: string }; saved: false }>("/api/assistant/profiles/draft", { method: "POST", body: JSON.stringify({ description: profileDescription }) });
      setName(result.draft.name); setRole(result.draft.role_instruction); setEditingProfile(null);
    } catch (e) { setError((e as Error).message); }
  };
  const editProfile = (profile: Profile) => { setEditingProfile(profile.profile_id); setName(profile.name); setRole(profile.role_instruction); };
  const deleteProfile = async (profile: Profile) => {
    if (!window.confirm(`删除专用助手“${profile.name}”？`)) return;
    try { await api(`/api/assistant/profiles/${encodeURIComponent(profile.profile_id)}`, { method: "DELETE" }); if (profileId === profile.profile_id) setProfileId("workspace-analyst"); await reload(); }
    catch (e) { setError((e as Error).message); }
  };
  const runEvaluation = async () => {
    setEvaluationBusy(true); setError("");
    try {
      const report = await api<EvaluationReport>("/api/assistant/evaluations/reference", { method: "POST" });
      setEvaluationReports((current) => [report, ...current.filter((item) => item.evaluation_id !== report.evaluation_id)].slice(0, 5));
    } catch (e) { setError((e as Error).message); }
    finally { setEvaluationBusy(false); }
  };
  const prepareMemory = async (content = memoryInput) => {
    try {
      const draft = await api<MemoryDraft>("/api/assistant/memories/draft", { method: "POST", body: JSON.stringify({ content: content.slice(0, 1000) }) });
      setMemoryDraft(draft.draft); setMemoryInput(draft.draft.content);
    } catch (e) { setError((e as Error).message); }
  };
  const confirmMemory = async () => {
    if (!memoryDraft) return;
    try {
      await api<Memory>("/api/assistant/memories", { method: "POST", body: JSON.stringify({ content: memoryDraft.content }) });
      setMemoryDraft(null); setMemoryInput(""); setMemories(await api<Memory[]>("/api/assistant/memories"));
    } catch (e) { setError((e as Error).message); }
  };
  const updateMemory = async (memory: Memory, patch: { content?: string; enabled?: boolean; remove?: boolean }) => {
    try {
      if (patch.remove) await api(`/api/assistant/memories/${encodeURIComponent(memory.memory_id)}`, { method: "DELETE" });
      else if (patch.enabled !== undefined) await api(`/api/assistant/memories/${encodeURIComponent(memory.memory_id)}/status?enabled=${patch.enabled}`, { method: "PATCH" });
      else await api(`/api/assistant/memories/${encodeURIComponent(memory.memory_id)}`, { method: "PUT", body: JSON.stringify({ content: patch.content }) });
      setMemories(await api<Memory[]>("/api/assistant/memories"));
    } catch (e) { setError((e as Error).message); }
  };
  const saveSummary = async () => {
    if (!session || !summary) return;
    setSummarySaving(true);
    try { const next = await api<Summary>(`/api/assistant/sessions/${encodeURIComponent(session.session_id)}/summary`, { method: "PUT", body: JSON.stringify({ summary: summaryEdit }) }); setSummary(next); setSummaryEdit(next.summary); }
    catch (e) { setError((e as Error).message); }
    finally { setSummarySaving(false); }
  };
  const rebuildSummary = async () => {
    if (!session) return;
    setSummarySaving(true);
    try { const next = await api<Summary>(`/api/assistant/sessions/${encodeURIComponent(session.session_id)}/summary/rebuild`, { method: "POST" }); setSummary(next); setSummaryEdit(next.summary); }
    catch (e) { setError((e as Error).message); }
    finally { setSummarySaving(false); }
  };
  const checkObservability = async () => {
    try { const next = await api<NonNullable<Status["observability"]>>("/api/assistant/observability/check", { method: "POST" }); setStatus((current) => current ? { ...current, observability: next } : current); }
    catch (e) { setError((e as Error).message); }
  };

  const renderedMessages = useMemo(() => session?.messages || [], [session?.messages]);
  return <main className="assistant-page">
    <header className="assistant-hero">
      <div><div className="eyebrow">WORKSPACE ASSISTANT ／ 只读实验分析</div><h1>把工作区证据带进对话。</h1><p>询问运行、评测或模型与数据来源；回答附带本地证据和可比性限制。</p></div>
      <button className="button secondary" onClick={startSession}>新建对话</button>
    </header>
    <div className="assistant-layout">
      <aside className="assistant-rail" aria-label="助手与会话">
        <div className="assistant-default-agent"><span className="eyebrow">默认助手</span><strong>工作区实验分析</strong><p>直接提问即可，无需先创建助手。它会根据工作区记录分析运行、评测和数据血缘。</p></div>
        <label>当前对话使用<select value={profileId} disabled={Boolean(session)} onChange={(e) => setProfileId(e.target.value)} aria-label="选择助手">{profiles.map((item) => <option key={item.profile_id} value={item.profile_id}>{item.name}{item.built_in ? "（默认）" : ""}</option>)}</select></label>
        <details className="assistant-advanced" aria-label="专用助手与偏好设置">
          <summary>高级设置：专用助手与长期偏好</summary>
          <section className="assistant-profile-editor" aria-label="专用助手配置">
          <strong>{editingProfile ? "编辑专用助手" : "创建专用助手"}</strong>
          <textarea aria-label="自然语言描述助手需求" placeholder="例如：创建一个分析 SFT 训练失败并追踪相关数据来源的助手" value={profileDescription} onChange={(e) => setProfileDescription(e.target.value)} maxLength={1000} rows={2} />
          <button className="button secondary" disabled={!profileDescription.trim()} onClick={draftProfile}>生成配置草稿（本地）</button>
          <input aria-label="助手名称" placeholder="例如：SFT 评测分析" value={name} onChange={(e) => setName(e.target.value)} maxLength={80} />
          <textarea aria-label="角色指引" placeholder="说明希望助手重点关注什么" value={role} onChange={(e) => setRole(e.target.value)} maxLength={3000} rows={3} />
          <div className="assistant-inline-actions"><button className="button primary" disabled={!name.trim() || busy} onClick={saveProfile}>{editingProfile ? "保存修改" : "创建助手"}</button>{editingProfile && <button className="button secondary" onClick={() => { setEditingProfile(null); setName(""); setRole(""); }}>取消</button>}</div>
          {profiles.filter((item) => !item.built_in).map((item) => <div className="assistant-profile-row" key={item.profile_id}><span>{item.name}</span><button className="text-button" onClick={() => editProfile(item)}>编辑</button><button className="text-button" onClick={() => deleteProfile(item)}>删除</button></div>)}
          <small>角色指引只影响回答重点，不会增加工具或操作权限。</small>
          </section>
          <section className="assistant-memory-manager" aria-label="工作区长期记忆">
          <strong>工作区长期偏好</strong><p>仅保存你确认的偏好；不会把实验事实当作记忆。</p>
          <textarea aria-label="长期偏好草稿" value={memoryInput} onChange={(e) => { setMemoryInput(e.target.value); setMemoryDraft(null); }} maxLength={1000} rows={2} placeholder="例如：解释评测变化时优先列出样本分母" />
          {memoryDraft ? <div role="alert"><p>待确认保存到当前工作区：{memoryDraft.content}</p><button className="button primary" onClick={confirmMemory}>确认保存</button><button className="text-button" onClick={() => setMemoryDraft(null)}>取消</button></div> : <button className="button secondary" disabled={!memoryInput.trim()} onClick={() => prepareMemory()}>生成记忆草稿</button>}
          {memories.map((memory) => <div className="assistant-memory-row" key={memory.memory_id}><span>{memory.content}<small>{memory.status === "active" ? "使用中" : "已停用"}</small></span><button className="text-button" onClick={() => updateMemory(memory, { enabled: memory.status !== "active" })}>{memory.status === "active" ? "停用" : "启用"}</button><button className="text-button" onClick={() => { const next = window.prompt("编辑工作区偏好", memory.content); if (next?.trim()) updateMemory(memory, { content: next.trim() }); }}>编辑</button><button className="text-button" onClick={() => updateMemory(memory, { remove: true })}>删除</button></div>)}
          </section>
        </details>
        <section className="assistant-sessions"><strong>最近对话</strong>{sessions.length ? sessions.map((item) => <button key={item.session_id} className={`assistant-session-link${session?.session_id === item.session_id ? " active" : ""}`} onClick={() => openSession(item.session_id)}>{item.title}<small>{new Date(item.updated_at).toLocaleString()}</small></button>) : <p>还没有已保存的对话。</p>}</section>
      </aside>
      <section className="assistant-chat" aria-label="实验问答">
        <div className="assistant-provider-state" role="status"><span className={`connection-dot ${status?.provider_configured ? "" : "offline"}`} />{status?.provider || "阿里云百炼"} · {status?.provider_configured ? "已配置" : "尚未配置"} · {status?.runtime === "agentscope" ? "AgentScope 助手" : "基础助手"} · {session?.data_egress_consent_at ? "本会话已确认数据外发" : "等待本会话授权"} · 只读</div>
        <details className="assistant-runtime-details"><summary>服务状态与诊断</summary><div className="assistant-observability-state">Langfuse：{status?.observability?.detail || "未配置"}{status?.observability?.configured && <button className="text-button" onClick={checkObservability}>检查连接</button>}</div></details>
        {status && !status.provider_configured && <p className="assistant-error">请在工作区服务端的本地 `.env.local` 配置 `ONEGPU_SYNTHETIC_API_KEY`，再刷新页面。密钥不会由浏览器发送。</p>}
        {busy && <p role="status" className="assistant-analysis-progress">正在查找工作区证据并分析问题，请稍候；复杂比较可能需要较长时间。</p>}
        {!renderedMessages.length && <div className="assistant-empty"><h2>直接问你的实验问题</h2><p>选择一个问题开始。若结果不明确，再补充 Run ID、模型 ID 或数据集 ID。</p><div className="assistant-suggestions">{["最近有哪些失败运行？", "最近一次 SFT 评测和上一次相比有什么变化？", "这个模型使用了哪些数据？"].map((suggestion) => <button className="button secondary" key={suggestion} onClick={() => setQuestion(suggestion)}>{suggestion}</button>)}</div></div>}
        <div className="assistant-messages" aria-live="polite">{renderedMessages.map((message) => {
          const answer = parseAnswer(message);
          return <article id={message.message_id} className={`assistant-message ${message.role}`} key={message.message_id}>
            <strong>{message.role === "user" ? "你" : "实验助手"}</strong>
            {message.message_kind === "consent_notice" ? <div className="assistant-consent-history"><strong>外发授权提示（不是分析结论）</strong><p>{answer?.conclusion || message.content}</p></div> : null}
            {answer && message.message_kind !== "consent_notice" ? <div className={`assistant-answer-outcome${answer.facts.length || answer.interpretations.length ? " supported" : " clarification"}`}><strong>{answer.facts.length || answer.interpretations.length ? "分析结果" : "需要进一步处理"}</strong><p>{answer.conclusion}</p></div> : null}
            {answer && message.message_kind !== "consent_notice" && [...answer.facts, ...answer.interpretations].map((claim, index) => <p className="assistant-claim" key={`${message.message_id}-${index}`}>{claim.text}<small>证据：{claim.evidence_ids.join("、")}</small></p>)}
            {answer && message.message_kind !== "consent_notice" && answer.limitations.map((item) => <p className="assistant-limitation" key={item}>{item}</p>)}
            {answer && message.message_kind !== "consent_notice" && answer.clarification_options.map((item) => <p className="assistant-option" key={item}>{item}</p>)}
            {message.role === "user" && /记住|以后|偏好|习惯|请总是|请优先/.test(message.content) && <button className="text-button" onClick={() => prepareMemory(message.content)}>将这条偏好作为草稿</button>}
          </article>;
        })}</div>
        {summary && <details className="assistant-summary-panel" open><summary>较早对话已压缩 · v{summary.revision}</summary><p>摘要仅保留对话意图和对象引用；原始消息仍完整保存。涉及实验事实时会重新读取工作区。</p><textarea aria-label="可编辑的上下文摘要" value={summaryEdit || summary.summary} onChange={(e) => setSummaryEdit(e.target.value)} maxLength={4000} rows={4} /><small>{summary.source_message_ids.length} 条来源消息 · {new Date(summary.updated_at).toLocaleString()}</small><details><summary>查看摘要来源消息</summary><ul>{summary.source_message_ids.map((id) => <li key={id}><a href={`#${id}`}>{id}</a></li>)}</ul></details><div><button className="button secondary" disabled={summarySaving} onClick={saveSummary}>保存摘要修改</button><button className="text-button" disabled={summarySaving} onClick={rebuildSummary}>重新生成摘要</button></div></details>}
        {turn?.consent_required && <div className="assistant-consent" role="alert"><strong>发送前请确认数据外发</strong><p>本轮问题、当前助手角色指引、必要的结构化实验摘要和有限的近期对话上下文会发送到阿里云百炼；不会发送凭据、原始数据样本或完整日志。</p><button className="button primary" disabled={busy || !status?.provider_configured} onClick={confirmAndContinue}>同意并继续</button><button className="button secondary" onClick={() => setTurn(null)}>暂不发送</button></div>}
        {turn?.evidence.length ? <section className="assistant-evidence"><h2>本轮证据</h2>{turn.evidence.map((item) => <article key={item.evidence_id}><div><strong>{item.title}</strong><span className={`badge ${item.status === "available" ? "success" : "neutral"}`}>{item.status}</span></div><p>{item.summary}</p>{item.limitations.map((limitation) => <small key={limitation}>{limitation}</small>)}<button className="text-button" onClick={() => {
          const producer = item.facts?.find((fact) => fact.key === "produced_by_run_id")?.value;
          const runId = typeof producer === "string" ? producer : item.entity_id.startsWith("run-") ? item.entity_id : null;
          if (runId) onOpenRun(runId); else onOpenData();
        }}>打开工作台 →</button></article>)}</section> : null}
        <details className="assistant-technical-details"><summary>分析记录与质量检查</summary>{session && <section className="assistant-task-history" aria-label="本地任务记录"><h2>本地分析记录</h2>{tasks.slice(0, 5).map((task) => <article key={task.task_id}><strong>{task.status}</strong><small>{new Date(task.created_at).toLocaleString()} · {task.evidence.length} 条证据{task.duration_ms != null ? ` · ${task.duration_ms} ms` : ""}</small>{Boolean(task.memory_ids?.length) && <small>使用偏好 ID：{task.memory_ids?.join("、")}</small>}{task.context_summary_id && <small>上下文摘要：{task.context_summary_id}</small>}{task.langfuse_trace_url && <a href={task.langfuse_trace_url} target="_blank" rel="noreferrer">打开 Langfuse trace →</a>}{task.error_message && <p>{task.error_message}</p>}</article>)}</section>}<section className="assistant-evaluation-panel"><p>固定样例只检查引用、可比性、证据限制和权限边界，不代表模型回答质量。</p><button className="button secondary" disabled={evaluationBusy} onClick={runEvaluation}>{evaluationBusy ? "评估中…" : "运行助手评估"}</button>{evaluationReports[0] && <div className="assistant-evaluation-result"><strong>{evaluationReports[0].suite_version} · {evaluationReports[0].passed}/{evaluationReports[0].sample_count} 通过</strong><small>配置：{evaluationReports[0].configuration_id} · {new Date(evaluationReports[0].created_at).toLocaleString()}</small>{evaluationReports[0].failed_cases.map((item) => <p key={item.case_id}>{item.case_id}：{item.findings.filter((finding) => !finding.passed).map((finding) => finding.check).join("、")}</p>)}</div>}</section></details>
        {error && <p className="assistant-error" role="alert">{error}</p>}
        <form className="assistant-composer" onSubmit={(e) => { e.preventDefault(); submit(); }}>
          <label htmlFor="assistant-question">你的实验问题</label><textarea id="assistant-question" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={2000} rows={3} placeholder="例如：最近一次 SFT 评测和上一次相比有什么变化？" />
          <div><small>内容只读分析；发送到百炼前会显示并记录本会话授权。</small><button className="button primary" disabled={!question.trim() || busy}>{busy ? "正在分析…" : "发送问题"}</button></div>
        </form>
      </section>
    </div>
  </main>;
}
