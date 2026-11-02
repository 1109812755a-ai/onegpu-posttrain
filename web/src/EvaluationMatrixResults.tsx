import type { EvaluationMatrix } from "./types";

const METRIC_INFO: Record<string, { label: string; explanation: string }> = {
  gsm8k: { label: "GSM8K 正确率", explanation: "GSM8K 数学文字题中，按固定答案解析协议判定正确的比例。" },
  math_500: { label: "MATH-500 正确率", explanation: "MATH-500 子集上按固定评测协议判定的正确比例。" },
  internal_acc: { label: "内部留出集正确率", explanation: "项目内部未用于训练的评测样本正确比例；需结合实际分母解读。" },
  domain_ppl: { label: "领域困惑度（PPL）", explanation: "模型对领域文本的平均预测不确定性指标；越低不等于任务能力必然越强。" },
};

function numberText(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 4 }).format(value);
}

function statusText(status: string): string {
  const labels: Record<string, string> = {
    completed: "完成", partial: "部分结果", failed: "失败", available: "有值", unavailable: "无数据",
    verified: "来源已核验", incomplete: "来源不完整", mismatch: "来源不匹配", unknown: "来源待核验",
  };
  return labels[status] || status || "未知";
}

export function EvaluationMatrixResults({ matrix }: { matrix: EvaluationMatrix }) {
  return <>
    <section className={`comparison-summary ${matrix.comparability.status === "comparable" ? "" : "warning"}`} aria-label="比较条件">
      <div><span>矩阵</span><strong>{matrix.name}</strong><small>{matrix.protocol || "协议未记录"}</small></div>
      <div><span>基线</span><strong>{matrix.baseline_model_id || "未声明"}</strong><small>以矩阵声明为准</small></div>
      <div><span>可比性</span><strong>{matrix.comparability.status === "comparable" ? "协议一致" : `不可直接比较 · ${matrix.comparability.status}`}</strong>
        <small>protocol {matrix.comparability.protocol_version_consistent === true ? "一致" : matrix.comparability.protocol_version_consistent === false ? "不一致" : "未确认"} · 数据集 {matrix.comparability.dataset_ids_consistent === true ? "一致" : matrix.comparability.dataset_ids_consistent === false ? "不一致" : "未确认"}</small></div>
    </section>
    {(matrix.comparability.limitations.length > 0 || matrix.comparability.status !== "comparable" || matrix.truncated_models) &&
      <div className="comparison-alert" role="note">
        <strong>解读限制：</strong>{matrix.comparability.limitations.join("；") || "关键协议证据不一致或缺失，差值仅作原始数值展示。"}
        {matrix.truncated_models && " 已达到服务端显示上限。"}
      </div>}
    <section className="panel comparison-matrix-panel">
      <div className="comparison-table-wrap" tabIndex={0} aria-label="阶段比较表，可在窄屏内横向滚动">
        <table className="comparison-table">
          <caption>指标值保留原始量纲；n 为实际 / 请求样本数，Δ 为相对声明基线的数值差，不代表优劣结论。</caption>
          <thead><tr><th scope="col">模型 / 状态</th>{matrix.expected_metrics.map((key) => <th scope="col" key={key}><abbr title={METRIC_INFO[key]?.explanation || "评测指标定义随登记协议提供；当前缺少可读说明。"}>{METRIC_INFO[key]?.label || key}</abbr></th>)}</tr></thead>
          <tbody>{matrix.models.map((model) => <tr key={`${model.model_id}-${model.lineage.eval_id}`} className={model.model_id === matrix.baseline_model_id ? "baseline-row" : ""}>
            <th scope="row"><strong>{model.model_id || "模型 ID 缺失"}</strong><span className="comparison-status">{statusText(model.status)}</span>
              <span className="comparison-status">{statusText(model.provenance?.status || "unknown")}</span>
              <small>{model.lineage.model_artifact_id ? `模型产物 ${model.lineage.model_artifact_id}` : "模型产物血缘未解析"}</small>
            </th>
            {matrix.expected_metrics.map((key) => {
              const metric = model.metrics[key];
              return <td key={key}>
                <strong>{numberText(metric?.value)} <small>{metric?.unit || ""}</small></strong>
                <span>n {numberText(metric?.observed)} / {numberText(metric?.requested)}</span>
                <span>Δ {numberText(metric?.delta)}</span>
                <small>{statusText(metric?.status || "unavailable")}</small>
              </td>;
            })}
          </tr>)}</tbody>
        </table>
      </div>
    </section>
    <section className="panel comparison-lineage">
      <h2>评测与数据身份</h2><p>血缘按 artifact ID / run ID 精确关联；无法解析时保留结果并明确标记，不按模型名称猜测。</p>
      <div className="comparison-lineage-grid">{matrix.models.map((model) => <article key={model.model_id}>
        <strong>{model.model_id || "模型 ID 缺失"}</strong>
        <p className={`comparison-provenance ${model.provenance?.status === "verified" ? "" : "warning"}`}>来源：{statusText(model.provenance?.status || "unknown")}</p>
        {model.provenance?.issues?.length > 0 && <ul className="comparison-provenance-issues">{model.provenance.issues.map((issue) =>
          <li key={`${issue.code}-${issue.message}`}>{issue.message}</li>)}</ul>}
        <dl><dt>评测 ID</dt><dd>{model.lineage.evaluation_resolution?.resolved ? model.lineage.eval_id : "未解析"}</dd>
          <dt>评测 Run</dt><dd>{model.lineage.evaluation_resolution?.resolved ? `${model.lineage.evaluation_resolution.run_id} · ${model.lineage.evaluation_resolution.run_status || "状态未知"}` : "未解析"}</dd>
          <dt>评测模型 ID</dt><dd>{model.lineage.evaluation_resolution?.model_id || model.lineage.model_artifact_id || "未解析"}</dd>
          <dt>训练 Run</dt><dd>{model.lineage.training_resolution?.resolved ? `${model.lineage.training_resolution.run_id} · ${model.lineage.training_resolution.status || "状态未知"}` : "未解析"}</dd>
          <dt>父模型 ID</dt><dd>{model.lineage.training_resolution?.parent_model_ids?.join("、") || "未解析"}</dd>
          <dt>训练数据集 ID</dt><dd>{model.lineage.training_resolution?.dataset_ids?.join("、") || "未解析"}</dd>
          <dt>评测数据集 ID</dt><dd>{model.lineage.evaluation_resolution?.dataset_ids?.join("、") || model.lineage.dataset_ids.join("、") || "未解析"}</dd>
          <dt>协议 / 配置</dt><dd>{model.lineage.protocol_version || "未记录"} · {(model.lineage.config_sha256 || "配置摘要未记录").slice(0, 12)}</dd></dl>
      </article>)}</div>
    </section>
  </>;
}
