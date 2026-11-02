import { useEffect, useMemo, useState } from "react";
import { ModelDisplayNameEditor } from "./ModelDisplayNameEditor";

type Model = { model_id: string; stage: string; kind: string; digest: string; source_run_id?: string | null;
  produced_by_run_id?: string | null; training_completed_at?: string | null; display_name?: string | null;
  available: boolean; unavailable_reason?: string | null };
type Session = { session_id: string; model_id: string; status: string; detail?: string | null };
type Job = { job_id: string; status: string; result?: any; detail?: string | null };
type Catalog = { status: "idle" | "loading" | "ready" | "failed"; models: Model[]; detail?: string | null };
type GenerationParameters = { max_new_tokens: number; temperature: number; top_p: number; top_k: number;
  repetition_penalty: number; enable_thinking: boolean; do_sample?: boolean; sampling_parameters_active?: boolean;
  top_p_active?: boolean; top_k_active?: boolean; thinking_mode_supported?: boolean };
type Turn = { role: "user" | "assistant"; text: string; model?: string; parameters?: GenerationParameters };
type ParameterInputs = { max_new_tokens: string; temperature: string; top_p: string; top_k: string;
  repetition_penalty: string; enable_thinking: boolean };

const recommendedParameters: GenerationParameters = {
  max_new_tokens: 256, temperature: 0.7, top_p: 0.8, top_k: 20, repetition_penalty: 1.05, enable_thinking: false,
};
const recommendedInputs: ParameterInputs = {
  max_new_tokens: "256", temperature: "0.7", top_p: "0.8", top_k: "20", repetition_penalty: "1.05", enable_thinking: false,
};

function parameterInputError(values: ParameterInputs) {
  const maxTokens = Number(values.max_new_tokens), temperature = Number(values.temperature);
  const topP = Number(values.top_p), topK = Number(values.top_k), repetitionPenalty = Number(values.repetition_penalty);
  if (!Number.isInteger(maxTokens) || maxTokens < 1 || maxTokens > 1024) return "最大新 Token 须为 1–1024 的整数";
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) return "Temperature 须为 0–2";
  if (!Number.isFinite(topP) || topP < 0.01 || topP > 1) return "Top-p 须为 0.01–1";
  if (!Number.isInteger(topK) || topK < 0 || topK > 100) return "Top-k 须为 0–100 的整数";
  if (!Number.isFinite(repetitionPenalty) || repetitionPenalty < 0.5 || repetitionPenalty > 2) return "重复惩罚须为 0.5–2";
  return "";
}

function parameterSummary(parameters: GenerationParameters) {
  const sampling = parameters.do_sample === false ? "采样关闭；Top-p/Top-k 未生效" :
    `Top-p ${parameters.top_p}${parameters.top_p_active === false ? "（未生效）" : ""} · Top-k ${parameters.top_k}${parameters.top_k_active === false ? "（未生效）" : ""}`;
  const thinking = parameters.thinking_mode_supported === false ? "思考模式控制不受此模板支持（当前关闭）" :
    `思考模式${parameters.enable_thinking ? "开启" : "关闭"}`;
  return `max_new_tokens ${parameters.max_new_tokens} · temperature ${parameters.temperature} · ${sampling} · 重复惩罚 ${parameters.repetition_penalty} · ${thinking}`;
}

const stageNames: Record<string, string> = { base: "Base", cpt: "CPT", sft: "SFT", dpo: "DPO", simpo: "SimPO", grpo: "GRPO" };
const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));
function modelTitle(model: Model) {
  if (model.display_name) return model.display_name;
  const stage = stageNames[model.stage] || model.stage;
  const shortSource = (model.source_run_id || model.model_id).slice(-8);
  return `${stage} · ${model.training_completed_at ? model.training_completed_at.slice(0, 10) : "时间未知"} · ${shortSource}`;
}
function sortModels(rows: Model[]) {
  return [...rows].sort((left, right) => {
    const a = left.training_completed_at ? Date.parse(left.training_completed_at) : Number.NaN;
    const b = right.training_completed_at ? Date.parse(right.training_completed_at) : Number.NaN;
    const aValid = Number.isFinite(a), bValid = Number.isFinite(b);
    if (aValid !== bValid) return aValid ? -1 : 1;
    if (aValid && bValid && a !== b) return b - a;
    return left.model_id.localeCompare(right.model_id);
  });
}

async function responseJson<T>(response: Response): Promise<T> {
  const value = await response.json();
  if (!response.ok) throw new Error(value.detail || `本地服务返回 ${response.status}`);
  return value as T;
}

export function ModelPlayground({ onBack, initialModelId = "", returnRunId = "", onOpenRun }: {
  onBack: () => void; initialModelId?: string; returnRunId?: string; onOpenRun?: (runId: string) => void;
}) {
  const [models, setModels] = useState<Model[]>([]);
  const [catalogStatus, setCatalogStatus] = useState<Catalog["status"]>("loading");
  const [token, setToken] = useState("");
  const [loadError, setLoadError] = useState("");
  const [mode, setMode] = useState<"chat" | "compare">("chat");
  const [modelId, setModelId] = useState(initialModelId);
  const [selected, setSelected] = useState<string[]>(initialModelId ? [initialModelId] : []);
  const [session, setSession] = useState<Session | null>(null);
  const [message, setMessage] = useState("");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [parameterInputs, setParameterInputs] = useState<ParameterInputs>({ ...recommendedInputs });
  const parameters = useMemo<GenerationParameters>(() => ({
    max_new_tokens: Number(parameterInputs.max_new_tokens), temperature: Number(parameterInputs.temperature),
    top_p: Number(parameterInputs.top_p), top_k: Number(parameterInputs.top_k),
    repetition_penalty: Number(parameterInputs.repetition_penalty), enable_thinking: parameterInputs.enable_thinking,
  }), [parameterInputs]);
  const parameterError = parameterInputError(parameterInputs);
  const [busy, setBusy] = useState(false);
  const [activity, setActivity] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    const loadCatalog = async () => {
      for (let attempt = 0; attempt < 600; attempt += 1) {
        const catalog = await responseJson<Catalog>(await fetch("/api/model-playground/models"));
        if (!alive) return catalog;
        setCatalogStatus(catalog.status);
        if (catalog.status === "ready") return catalog;
        if (catalog.status === "failed") throw new Error(catalog.detail || "模型身份核验失败");
        await wait(1000);
      }
      throw new Error("本地模型目录核验耗时较长；请稍后刷新重试");
    };
    Promise.all([
      loadCatalog(),
      fetch("/api/launcher/capabilities").then((r) => responseJson<{ enabled: boolean; token?: string; reason?: string }>(r)),
    ]).then(([catalog, capability]) => {
      if (!alive) return;
      setModels(catalog.models);
      if (!capability.enabled || !capability.token) throw new Error(capability.reason || "请先在本机启用 Launcher");
      setToken(capability.token);
      const available = sortModels(catalog.models.filter((row) => row.available));
      setModelId(initialModelId || available[0]?.model_id || "");
    }).catch((reason) => alive && setLoadError(reason instanceof Error ? reason.message : "读取模型目录失败"));
    return () => { alive = false; };
  }, [initialModelId]);

  useEffect(() => {
    const activeSessionId = session?.session_id;
    if (!activeSessionId || !token) return;
    return () => { void fetch(`/api/model-playground/sessions/${activeSessionId}`, {
      method: "DELETE", headers: { "X-Local-Launcher-Token": token }, keepalive: true,
    }); };
  }, [session?.session_id, token]);

  const sortedModels = useMemo(() => sortModels(models), [models]);
  const availableModels = useMemo(() => sortedModels.filter((row) => row.available), [sortedModels]);
  const modelById = (id: string) => models.find((row) => row.model_id === id);
  const updateDisplayName = (id: string, displayName: string | null) => setModels((old) => old.map((row) =>
    row.model_id === id ? { ...row, display_name: displayName } : row));
  const headers = { "Content-Type": "application/json", "X-Local-Launcher-Token": token };
  async function waitSession(sessionId: string) {
    for (let i = 0; i < 240; i += 1) {
      const state = await responseJson<Session>(await fetch(`/api/model-playground/sessions/${sessionId}`, { headers }));
      if (state.status !== "loading") return state;
      await wait(500);
    }
    throw new Error("等待模型加载超时；请结束会话并检查本机资源");
  }
  async function waitJob(jobId: string) {
    for (let i = 0; i < 2400; i += 1) {
      const job = await responseJson<Job>(await fetch(`/api/model-playground/jobs/${jobId}`, { headers }));
      if (["completed", "partial", "failed", "cancelled"].includes(job.status)) return job;
      await wait(500);
    }
    throw new Error("等待生成结果超时；可结束模型会话后重试");
  }
  async function startChatSession() {
    if (!modelId || !token) return;
    setBusy(true); setError(""); setActivity("正在读取本地模型…"); setTurns([]);
    try {
      const started = await responseJson<Session>(await fetch("/api/model-playground/sessions", { method: "POST", headers, body: JSON.stringify({ model_id: modelId }) }));
      setSession(started);
      const ready = await waitSession(started.session_id);
      if (ready.status !== "ready") throw new Error(ready.detail || "模型加载失败");
      setSession(ready); setActivity("模型已就绪，可以开始对话");
    } catch (reason) { setSession(null); setActivity(""); setError(reason instanceof Error ? reason.message : "启动模型会话失败"); }
    finally { setBusy(false); }
  }
  async function sendMessage() {
    const prompt = message.trim();
    if (!prompt || !session || busy) return;
    setBusy(true); setError(""); setActivity("正在生成回答…");
    setTurns((old) => [...old, { role: "user", text: prompt }]); setMessage("");
    try {
      const queued = await responseJson<Job>(await fetch(`/api/model-playground/sessions/${session.session_id}/messages`, {
        method: "POST", headers, body: JSON.stringify({ message: prompt, parameters }),
      }));
      const result = await waitJob(queued.job_id);
      if (result.status !== "completed") throw new Error(result.detail || "生成失败");
      setTurns((old) => [...old, { role: "assistant", text: result.result.text, model: result.result.model_id,
        parameters: result.result.parameters || { ...parameters } }]);
      setActivity("回答完成 · 仅定性体验，未经独立评测");
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "生成失败"); setActivity("");
      // A failed generation releases the server-side model and GPU lease. Drop
      // the stale client session too, so the next action is an explicit reload.
      setSession(null);
    }
    finally { setBusy(false); }
  }
  async function compare() {
    const prompt = message.trim();
    if (!prompt || selected.length < 2 || selected.length > 3 || busy) return;
    setBusy(true); setError(""); setActivity("按顺序加载模型并生成中；GPU 由本任务独占…"); setTurns([{ role: "user", text: prompt }]); setMessage("");
    try {
      const queued = await responseJson<Job>(await fetch("/api/model-playground/comparisons", {
        method: "POST", headers, body: JSON.stringify({ model_ids: selected, message: prompt,
          parameters }),
      }));
      const job = await waitJob(queued.job_id);
      const rows = job.result?.results || [];
      setTurns([{ role: "user", text: prompt }, ...rows.map((row: any) => ({
        role: "assistant" as const, text: row.status === "completed" ? row.text : `体验失败：${row.detail || job.detail || "本地模型错误"}`,
        model: row.model_id, parameters: row.status === "completed" ? row.parameters || { ...parameters } : undefined,
      }))]);
      setActivity(job.status === "completed" ? "对照完成 · 输出未经独立评测" : "对照未全部完成 · 已保留可用结果");
    } catch (reason) { setError(reason instanceof Error ? reason.message : "模型对照失败"); setActivity(""); }
    finally { setBusy(false); }
  }
  async function closeSession() {
    if (!session) return;
    setBusy(true);
    try { await fetch(`/api/model-playground/sessions/${session.session_id}`, { method: "DELETE", headers }); }
    finally { setSession(null); setTurns([]); setActivity("模型已卸载，GPU 已释放"); setBusy(false); }
  }

  return <main className="model-playground-page">
    <header className="hero"><div><div className="eyebrow">TRAINING WORKSPACE ／ LOCAL MODEL PLAYGROUND</div>
      <h1>直接体验阶段模型<span>。</span></h1><p>使用自己的问题感受模型输出；这里是定性体验，不替代正式阶段评测。</p></div>
      <div className="data-hero-actions">{returnRunId && onOpenRun && <button className="button secondary" onClick={() => onOpenRun(returnRunId)}>返回来源 Run</button>}
        <button className="button secondary" onClick={onBack}>返回训练与评测</button></div></header>
    {loadError && <div className="notice warning" role="alert">模型体验不可用：{loadError}</div>}
    {!loadError && catalogStatus === "loading" && <div className="notice info" role="status">正在核验本地模型登记与 Run 血缘。这里只建立可体验模型清单；选中模型时还会单独核验权重摘要，再加载模型。</div>}
    <section className="launcher-card model-playground-card" aria-labelledby="playground-title">
      <div className="runs-detail-heading"><div><div className="eyebrow">QUALITATIVE · LOCAL · EPHEMERAL</div><h2 id="playground-title">选择体验方式</h2></div>
        <span className="badge warning">不产生 Run / 评测 / 训练数据</span></div>
      <div className="model-playground-mode" role="group" aria-label="体验模式">
        <button className={`button ${mode === "chat" ? "primary" : "secondary"}`} aria-pressed={mode === "chat"} disabled={Boolean(session) || busy} onClick={() => { setMode("chat"); setTurns([]); }}>单模型多轮对话</button>
        <button className={`button ${mode === "compare" ? "primary" : "secondary"}`} aria-pressed={mode === "compare"} disabled={Boolean(session) || busy} onClick={() => { setMode("compare"); setTurns([]); }}>同题模型对照</button>
      </div>
      <div className="model-playground-controls">
        {mode === "chat" ? <><label>体验模型<select value={modelId} onChange={(event) => setModelId(event.target.value)} disabled={Boolean(session) || busy}>
          <option value="">选择可体验模型</option>{initialModelId && !availableModels.some((model) => model.model_id === initialModelId) && <option value={initialModelId}>Run 指定模型当前不可用 · {initialModelId}</option>}{availableModels.map((model) => <option key={model.model_id} value={model.model_id}>{modelTitle(model)} · {model.kind}</option>)}</select></label>
          {modelById(session?.model_id || modelId) && <div className="model-playground-identity"><strong>{modelTitle(modelById(session?.model_id || modelId)!)}</strong><span>{stageNames[modelById(session?.model_id || modelId)!.stage] || modelById(session?.model_id || modelId)!.stage} · {modelById(session?.model_id || modelId)!.kind}</span>
            <small>训练完成：{modelById(session?.model_id || modelId)!.training_completed_at || "时间未知"}</small>
            <small>训练来源 Run：{modelById(session?.model_id || modelId)!.source_run_id || "Base / 未记录"}</small>
            {modelById(session?.model_id || modelId)!.produced_by_run_id && <small>合并 Run：{modelById(session?.model_id || modelId)!.produced_by_run_id}</small>}
            <details><summary>模型身份与摘要</summary><code>{session?.model_id || modelId}</code><small>SHA-256 {modelById(session?.model_id || modelId)!.digest}</small></details>
            {!session && token && <ModelDisplayNameEditor modelId={modelId} displayName={modelById(modelId)!.display_name} token={token}
              onSaved={(name) => updateDisplayName(modelId, name)} />}
          </div>}
          {initialModelId && modelById(initialModelId) && !modelById(initialModelId)!.available && modelId === initialModelId && <p className="notice warning" role="status">Run 指定的模型当前不可体验；请确认身份/权重，或手动选择其他可用模型。</p>}
        </>
          : <fieldset className="model-playground-model-list"><legend>选择 2–3 个模型（顺序执行，不会同时占用显存）</legend>
            {sortedModels.map((model) => <label key={model.model_id} className={!model.available ? "disabled" : ""}><input type="checkbox" disabled={!model.available || busy || (!selected.includes(model.model_id) && selected.length >= 3)} checked={selected.includes(model.model_id)} onChange={(event) => setSelected((old) => event.target.checked ? [...old, model.model_id] : old.filter((id) => id !== model.model_id))} />
              <span><strong>{modelTitle(model)} · {stageNames[model.stage] || model.stage} · {model.kind}</strong><code>{model.model_id}</code><small>{model.available ? `训练完成：${model.training_completed_at || "时间未知"} · 训练 Run：${model.source_run_id || "Base / 未记录"}${model.produced_by_run_id ? ` · 合并 Run：${model.produced_by_run_id}` : ""} · SHA-256 ${(model.digest || "").slice(0, 12)}…` : model.unavailable_reason || "当前不可体验"}</small></span></label>)}
          </fieldset>}
        <section className="model-playground-generation-settings" aria-label="生成参数">
          <div className="model-playground-parameters">
            <label>最大新 Token<input type="text" inputMode="numeric" value={parameterInputs.max_new_tokens} onChange={(event) => setParameterInputs((old) => ({ ...old, max_new_tokens: event.target.value }))} disabled={busy} aria-invalid={Boolean(parameterError && !Number.isInteger(Number(parameterInputs.max_new_tokens)))} /></label>
            <label>Temperature<input type="text" inputMode="decimal" value={parameterInputs.temperature} onChange={(event) => setParameterInputs((old) => ({ ...old, temperature: event.target.value }))} disabled={busy} /></label>
            <button className="button secondary" type="button" onClick={() => setParameterInputs({ ...recommendedInputs })} disabled={busy}>恢复推荐默认值</button>
          </div>
          <details className="model-playground-advanced-settings">
            <summary>高级生成参数 <span>Top-p、Top-k、重复惩罚与思考模式</span></summary>
            <div className="model-playground-parameters">
              <label>Top-p<input type="text" inputMode="decimal" value={parameterInputs.top_p} onChange={(event) => setParameterInputs((old) => ({ ...old, top_p: event.target.value }))} disabled={busy || parameters.temperature === 0} /></label>
              <label>Top-k（0 为关闭）<input type="text" inputMode="numeric" value={parameterInputs.top_k} onChange={(event) => setParameterInputs((old) => ({ ...old, top_k: event.target.value }))} disabled={busy || parameters.temperature === 0} /></label>
              <label>重复惩罚<input type="text" inputMode="decimal" value={parameterInputs.repetition_penalty} onChange={(event) => setParameterInputs((old) => ({ ...old, repetition_penalty: event.target.value }))} disabled={busy} /></label>
              <label className="model-playground-thinking-toggle"><input type="checkbox" checked={parameters.enable_thinking} onChange={(event) => setParameterInputs((old) => ({ ...old, enable_thinking: event.target.checked }))} disabled={busy} />启用 Qwen3 思考模式</label>
            </div>
            <p>推荐值用于本地定性体验；重复惩罚是温和起点，并非 Qwen 官方推荐。Temperature 为 0 时 Top-p/Top-k 不参与生成。</p>
          </details>
          {parameterError && <p className="notice warning" role="alert">{parameterError}</p>}
        </section>
      </div>
      <div className="model-playground-composer"><label htmlFor="playground-message">你的问题</label><textarea id="playground-message" rows={4} maxLength={16000} value={message} onChange={(event) => setMessage(event.target.value)} placeholder="输入想比较或体验的问题；内容只在本机内存中处理，不会自动保存。" disabled={busy} />
        <div className="model-playground-actions">{mode === "chat" ? session ? <><button className="button primary" onClick={sendMessage} disabled={busy || Boolean(parameterError) || !message.trim()}>{busy ? "生成中…" : "发送消息"}</button><button className="button secondary" onClick={closeSession} disabled={busy}>结束会话并卸载模型</button></> : <button className="button primary" onClick={startChatSession} disabled={busy || Boolean(parameterError) || !modelId || !availableModels.some((model) => model.model_id === modelId) || !token}>{busy ? "加载模型中…" : "加载模型开始体验"}</button>
          : <button className="button primary" onClick={compare} disabled={busy || Boolean(parameterError) || !message.trim() || selected.length < 2 || !token}>{busy ? "对照运行中…" : `对照 ${selected.length} 个模型并生成`}</button>}
          {activity && <span role="status">{activity}</span>}</div>
      </div>
      {error && <div className="notice danger" role="alert">{error}</div>}
      {turns.length > 0 && <section className="model-playground-results" aria-label="对话与模型输出"><h3>体验结果 <span className="badge warning">定性输出 · 未经独立验证</span></h3>
        {mode === "compare" ? <div className="model-playground-columns">{turns.filter((turn) => turn.role === "assistant").map((turn, index) => { const model = modelById(turn.model || ""); return <article key={`${turn.model}-${index}`}><h4>{model ? modelTitle(model) : "模型"}</h4>{model && <strong>{stageNames[model.stage] || model.stage} · {model.kind}</strong>}<code>{turn.model}</code>{model && <details><summary>来源与摘要</summary><small>训练完成：{model.training_completed_at || "时间未知"} · 训练 Run：{model.source_run_id || "Base / 未记录"} · 合并 Run：{model.produced_by_run_id || "未记录"}</small><small>SHA-256 {model.digest}</small></details>}{turn.parameters && <small className="model-playground-response-parameters">本次参数：{parameterSummary(turn.parameters)}</small>}<p>{turn.text}</p></article>; })}</div>
          : <div className="model-playground-transcript" aria-live="polite">{turns.map((turn, index) => <article key={index} className={`playground-turn ${turn.role}`}><strong>{turn.role === "user" ? "你" : modelTitle(modelById(turn.model || modelId) || { model_id: turn.model || modelId, stage: "unknown", kind: "unknown", digest: "", available: false })}</strong>{turn.role === "assistant" && <code>{turn.model || modelId}</code>}{turn.parameters && <small className="model-playground-response-parameters">本次参数：{parameterSummary(turn.parameters)}</small>}<p>{turn.text}</p></article>)}</div>}
      </section>}
    </section>
  </main>;
}
