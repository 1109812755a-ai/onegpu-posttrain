import { useCallback, useEffect, useState } from "react";

type Metric = { correct: number; denominator: number; accuracy: number; parse_failures: number; status: string };
type Report = { role: string; metrics: Record<string, Metric>; dataset_id: string; model_sha256: string };
type Candidate = { candidate_id: string; model_id?: string; candidate_sha256?: string; tensor_name?: string;
  changed_elements?: number; state: string; practice?: Report | null; final?: Report | null };
type Job = { id: string; kind: string; status: string; error_category?: string;
  result?: { status?: string; candidate_id?: string; error_category?: string; comparison?: { delta_accuracy: number } } };
type Status = { ready: boolean; reason?: string; experiment?: { target: { repository: string; revision: string };
  practice: { record_count: number }; final_holdout: { record_count: number } }; provider?: { model: string;
  credential_ready: boolean; pricing: string; max_provider_calls: number; max_tool_calls: number;
  max_output_tokens_per_call: number }; baseline?: { practice: Report; final: Report };
  candidates: Candidate[]; job?: Job | null; practice_predictions_available?: boolean };
type Preflight = { confirmation_token: string; egress: string; pricing: string; max_provider_calls: number;
  max_tool_calls: number; max_output_tokens_per_call: number };

async function api<T>(path: string, body?: object): Promise<T> {
  const response = await fetch(`/api/weight-space/${path}`, body ? {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  } : undefined);
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data as T;
}

function score(report?: Report | null) {
  if (!report || report.metrics?.overall?.status !== "measured") return "未完成";
  const item = report.metrics.overall;
  return `${item.correct}/${item.denominator} · ${(item.accuracy * 100).toFixed(1)}% · 解析失败 ${item.parse_failures}`;
}

export function WeightSpaceWorkbench() {
  const [status, setStatus] = useState<Status | null>(null);
  const [preflight, setPreflight] = useState<Preflight | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    try { setStatus(await api<Status>("status")); } catch (exc) { setError(String(exc)); }
  }, []);
  useEffect(() => { void refresh(); const id = window.setInterval(() => void refresh(), 2500); return () => window.clearInterval(id); }, [refresh]);

  async function action(run: () => Promise<unknown>) {
    setBusy(true); setError("");
    try { await run(); await refresh(); } catch (exc) { setError(exc instanceof Error ? exc.message : String(exc)); }
    finally { setBusy(false); }
  }

  return <main className="weight-space-page" aria-labelledby="weight-space-title">
    <header className="weight-space-header">
      <p className="eyebrow">Qwen3.5 · 易经客观题</p>
      <h1 id="weight-space-title">权重空间实验工作台</h1>
      <p>使用百炼 Qwen3.8-Max 提议有界权重编辑，在独立的 300 题练习集上筛选候选，再用 81 题最终集比较。</p>
    </header>
    {error && <p className="weight-space-alert" role="alert">{error}</p>}
    {!status && <section className="weight-space-card" role="status">正在读取本地实验状态…</section>}
    {status && !status.ready && <section className="weight-space-card" role="status"><h2>实验尚未就绪</h2><p>{status.reason}</p></section>}
    {status?.ready && <>
      <section className="weight-space-grid" aria-label="实验设置">
        <article className="weight-space-card"><h2>固定实验身份</h2>
          <p><strong>{status.experiment?.target.repository}</strong></p>
          <p>版本 {status.experiment?.target.revision.slice(0, 12)} · 练习 {status.experiment?.practice.record_count} 题 · 最终 {status.experiment?.final_holdout.record_count} 题</p>
          <p>基线练习：{score(status.baseline?.practice)}</p><p>基线最终：{score(status.baseline?.final)}</p>
        </article>
        <article className="weight-space-card"><h2>编辑器与调用预算</h2>
          <p>{status.provider?.model} · 凭据{status.provider?.credential_ready ? "可用" : "未配置"}</p>
          <p>每轮最多 {status.provider?.max_provider_calls} 次模型调用、{status.provider?.max_tool_calls} 次工具调用；每次最多 {status.provider?.max_output_tokens_per_call} 输出 token。</p>
          <p>价格：未知。实际费用由百炼账户计费，页面不估算金额。</p>
        </article>
      </section>
      <section className="weight-space-card" aria-labelledby="weight-space-iteration-title">
        <h2 id="weight-space-iteration-title">新一轮编辑</h2>
        <p>发送给百炼的数据包含最多 20 道练习题的问题、正确答案、基线输出及有界张量数值。最终 81 题的题目和答案不进入编辑请求。</p>
        {!preflight && <button className="button" disabled={busy || !status.provider?.credential_ready || status.job?.status === "running"}
          onClick={() => void action(async () => { setPreflight(await api<Preflight>("iteration/preflight", {})); setConfirmed(false); })}>检查本轮调用</button>}
        {preflight && <div className="weight-space-confirm">
          <p>{preflight.egress}</p><p>价格：未知；最多 {preflight.max_provider_calls} 次模型调用和 {preflight.max_tool_calls} 次工具调用。</p>
          <label><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> 我确认本轮向百炼发送练习数据并承担实际调用费用</label>
          <div className="weight-space-actions"><button className="button" disabled={!confirmed || busy} onClick={() => void action(async () => {
            await api("iteration/start", { confirmation_token: preflight.confirmation_token }); setPreflight(null); setConfirmed(false);
          })}>确认并开始本轮</button><button className="button secondary" onClick={() => { setPreflight(null); setConfirmed(false); }}>取消</button></div>
        </div>}
        {status.job && <p role="status">{status.job.kind === "iteration" ? "编辑与练习评测" : "最终评测"}：{status.job.status === "running" ? "运行中，结果尚不可用" : status.job.status === "failed" ? `失败（${status.job.error_category}）` : status.job.result?.status === "no_edit" ? "本轮未生成有效编辑" : status.job.result?.status === "failed" ? "提供方请求失败" : "已完成"}</p>}
        {status.job?.result?.error_category && <p role="alert">提供方请求失败：{status.job.result.error_category}</p>}
      </section>
      <section className="weight-space-card" aria-labelledby="weight-space-candidates-title">
        <h2 id="weight-space-candidates-title">候选模型</h2>
        {!status.candidates.length && <p>暂无候选。完成一轮有效编辑后会在此显示练习集结果。</p>}
        <div className="weight-space-candidates">{status.candidates.map((candidate) => <article key={candidate.candidate_id} className="weight-space-candidate">
          <h3>{candidate.candidate_id}</h3><p>状态：{candidate.state === "retained" ? "保留" : candidate.state === "discarded" ? "已丢弃" : "记录无效"}</p>
          {candidate.tensor_name && <p>编辑 {candidate.tensor_name} · {candidate.changed_elements} 个元素</p>}
          <p>练习集：{score(candidate.practice)}</p><p>最终集：{score(candidate.final)}</p>
          {candidate.final && status.baseline?.final && <p>相对基线：{((candidate.final.metrics.overall.accuracy - status.baseline.final.metrics.overall.accuracy) * 100).toFixed(1)} 个百分点</p>}
          <div className="weight-space-actions">
            {candidate.state === "retained" && <button className="button secondary" disabled={busy || status.job?.status === "running"}
              onClick={() => void action(() => api("candidates/transition", { candidate_id: candidate.candidate_id, action: "discard" }))}>丢弃</button>}
            {candidate.state === "discarded" && <button className="button secondary" disabled={busy || status.job?.status === "running"}
              onClick={() => void action(() => api("candidates/transition", { candidate_id: candidate.candidate_id, action: "restore" }))}>恢复</button>}
            {candidate.state === "retained" && candidate.practice && !candidate.final && <button className="button" disabled={busy || status.job?.status === "running"}
              onClick={() => void action(() => api("candidates/final", { candidate_id: candidate.candidate_id }))}>运行 81 题最终评测</button>}
          </div>
        </article>)}</div>
      </section>
    </>}
  </main>;
}
