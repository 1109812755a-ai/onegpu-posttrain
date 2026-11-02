import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import { createRoot } from "react-dom/client";
import { PolicyEditor } from "./PolicyEditor";
import { TargetedExperiments } from "./TargetedExperiments";
import { EvaluationMatrixPage } from "./EvaluationMatrixPage";
import { Launcher } from "./Launcher";
import { Runs } from "./Runs";
import { ModelPlayground } from "./ModelPlayground";
import { DatasetWorkbench } from "./DatasetWorkbench";
import { ActivityTasks } from "./ActivityTasks";
import { ThemePreferenceControl } from "./ThemePreferenceControl";
import { MODULE_VIEWS, PRIMARY_MODULES, WorkspaceNavigation } from "./WorkspaceNavigation.tsx";
import { grpoPreparationSearch, isNavigationCurrent, launcherReturnSearch, navigationTarget, registeredDatasetTrainingSearch, workspaceModuleFromPage, workspaceRouteStateFromSearch, workspaceSearchForPage } from "./workspaceNavigation.mjs";
import { applyThemeToDocument, readThemeMode, resolveTheme, subscribeSystemTheme, systemPrefersDark, writeThemeMode } from "./themePreferences.mjs";
import type { ThemeMode } from "./themePreferences.mjs";
import type {
  Artifact,
  BuildDetail,
  BuildSummary,
  Catalog,
  Dataset,
  Metric,
  Step,
} from "./types";
import "./style.css";

const STAGES: Record<string, string> = {
  cpt: "CPT Train",
  cpt_holdout: "CPT Holdout",
  sft: "SFT",
  preference: "DPO / SimPO",
  preference_eval: "Preference Eval",
  grpo: "GRPO",
  eval: "Internal Eval",
};
const UNITS: Record<string, string> = {
  raw_rows: "原始行",
  documents: "文档",
  unique_problems: "唯一问题",
  generations: "推理候选",
  examples: "样本",
  preference_pairs: "偏好对",
  prompts: "prompts",
  tokens: "tokens",
  chunks: "chunks",
  records: "记录",
};
const METRICS: Record<string, string> = {
  complete_rate: "完整推理比例",
  answer_parse_rate: "答案可解析率",
  nonempty_rate: "非空文本比例",
  schema_valid_rate: "Schema 有效率",
  duplicate_record_ids: "重复记录 ID",
  train_eval_overlap: "训练 / 评测重叠",
  semantic_correctness: "独立语义正确性",
  near_duplicate_rate: "近重复率",
  upstream_label_eligibility_rate: "上游标签准入一致率",
};
const OPERATIONS: Record<string, string> = {
  normalize: "问题规范化",
  "exact-dedup": "精确去重",
  decontaminate: "基准去污染",
  "trace-eligibility": "推理完整性与答案解析",
  "verifiable-core": "可验证数学核心",
  "problem-split": "问题级切分",
  "build-sft": "构建 SFT 样本",
  "build-preference": "构建偏好对",
  "build-grpo": "构建 GRPO prompts",
  "build-holdout": "构建 Internal Holdout",
  "build-preference_eval": "构建偏好评测对",
  "validate-text": "非空文本检查",
  "document-split": "文档级切分",
  "tokenizer-chunking": "Tokenizer 分块",
  "token-budget-selection": "Token 预算选择",
  "normalize / exact dedup": "问题规范化与精确去重",
  "benchmark screening": "基准污染检查",
  "verifiable math core": "可验证数学核心",
  "source text validation": "读取与检查文本",
  "document split / screening": "文档切分与污染检查",
  "chunk / token budget": "分块与 Token 预算",
};
const REASONS: Record<string, string> = {
  empty_problem: "空问题",
  exact_duplicate_problem: "精确重复问题",
  benchmark_contamination: "基准污染命中",
  benchmark_contamination_removed: "基准污染命中",
  trace_empty: "空推理",
  trace_incomplete: "推理不完整 / 已截断",
  trace_unparseable: "无法提取最终答案",
  no_verifier_correct_trace: "没有满足标签准入的正确推理",
  sft_no_eligible_correct_trace: "正确推理不满足长度预算",
  preference_missing_valid_pair_side: "缺少有效 chosen / rejected",
  preference_length_ratio_gate: "偏好对长度比不合格",
  grpo_unverifiable_ground_truth: "标准答案不支持验证",
  grpo_prompt_length_gate: "Prompt 超出长度预算",
  empty_text: "空文本",
  exact_contamination: "精确污染命中",
  no_chunks: "未生成有效分块",
  zero_token_chunks: "零 token 分块",
  token_budget: "达到 token 配额",
};

const PAGE_LABELS: Record<string, string> = {
  data: "处理与质量",
  workbench: "数据集",
  policies: "规则对照",
  launcher: "训练与评测",
  "model-playground": "模型体验",
  runs: "运行记录",
  comparison: "阶段比较",
  "policy-effects": "策略效果",
  targeted: "薄弱场景",
};
const TABS = [
  { id: "flow", label: "处理流程" },
  { id: "quality", label: "质量与分布" },
  { id: "effects", label: "训练与评测" },
];
const nf = new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 2 });
const number = (n: number | null | undefined) =>
  n == null ? "—" : nf.format(n);
const unit = (u: string) => UNITS[u] || u;
const metricValue = (m?: Metric) =>
  !m || m.value == null || !["measured", "estimated"].includes(m.status)
    ? "—"
    : m.unit === "ratio"
      ? `${number(m.value * 100)}%`
      : number(m.value);
const STATUS_NAMES: Record<string, string> = {
  measured: "已测量",
  estimated: "估算值",
  not_measured: "未测量",
  not_applicable: "不适用",
  completed: "已记录",
  skipped: "未检查",
  derived: "旧版推导",
  FAIL: "检查失败",
  NOT_EVALUATED: "未完成评估",
  PASS: "通过",
};
const CHECK_NAMES: Record<string, string> = {
  canonical_schema: "Schema",
  unique_record_ids: "记录 ID 唯一",
  train_eval_separation: "训练 / 评测隔离",
};
const statusName = (s: string) => STATUS_NAMES[s] || s;
const operation = (s: Step) => OPERATIONS[s.operation] || s.operation;

function Icon({ name, size = 18 }: { name: string; size?: number }) {
  const paths: Record<string, string> = {
    grid: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
    data: "M20 6c0 2-4 3-8 3S4 8 4 6s4-3 8-3 8 1 8 3z M4 6v12c0 2 4 3 8 3s8-1 8-3V6 M4 12c0 2 4 3 8 3s8-1 8-3",
    runs: "M8 5h12M8 12h12M8 19h12M3 5h1M3 12h1M3 19h1",
    chart: "M4 3v17h17 M8 15l4-5 4 2 5-7",
    branch:
      "M6 3v12a3 3 0 003 3h8 M6 8h8a3 3 0 003-3V3 M3 18a3 3 0 106 0 3 3 0 00-6 0 M14 18a3 3 0 106 0 3 3 0 00-6 0",
    refresh: "M20 7v5h-5 M4 17v-5h5 M6 6a8 8 0 0113 3 M18 18A8 8 0 015 15",
    search: "M10 3a7 7 0 100 14 7 7 0 000-14z M15 15l6 6",
    arrow: "M4 12h15 M14 7l5 5-5 5",
    chevron: "M8 10l4 4 4-4",
    file: "M14 3H5v18h14V8z M14 3v5h5 M8 12h8 M8 16h6",
    lock: "M6 10h12v11H6z M8 10V6a4 4 0 018 0v4",
    close: "M6 6l12 12 M18 6L6 18",
    info: "M12 3a9 9 0 100 18 9 9 0 000-18z M12 10v6 M12 7v.2",
    check: "M5 12l4 4L19 6",
    beaker: "M8 3h8 M9 3v6L4 18a2 2 0 002 3h12a2 2 0 002-3l-5-9V3 M7 14h10",
    box: "M3 7l9-4 9 4v11l-9 4-9-4z M3 7l9 4 9-4 M12 11v11",
  };
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={paths[name] || paths.info} />
    </svg>
  );
}

async function request<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { signal });
  if (!response.ok)
    throw new Error(
      `读取失败（HTTP ${response.status}）。请确认本地 API 和报告是否可用。`,
    );
  return response.json();
}

function Badge({
  children,
  kind = "neutral",
}: {
  children: ReactNode;
  kind?: string;
}) {
  return <span className={`badge ${kind}`}>{children}</span>;
}
function Notice({
  children,
  kind = "info",
}: {
  children: ReactNode;
  kind?: string;
}) {
  return (
    <div className={`notice ${kind}`}>
      <Icon name="info" />
      <div>{children}</div>
    </div>
  );
}
function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="empty-icon">
        <Icon name="branch" size={26} />
      </div>
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  );
}
function EvidenceButton({
  artifact,
  onOpen,
  children,
}: {
  artifact?: Artifact | null;
  onOpen: (a: Artifact) => void;
  children?: ReactNode;
}) {
  return artifact ? (
    <button
      className="text-button"
      onClick={() => onOpen(artifact)}
      title={artifact.path}
    >
      <Icon name="file" size={15} />
      {children || "查看来源"}
    </button>
  ) : null;
}

function StageCard({
  dataset: d,
  active,
  onClick,
}: {
  dataset: Dataset;
  active: boolean;
  onClick: () => void;
}) {
  const value = d.tokens ?? d.count;
  const target = d.tokens != null ? d.token_target : d.target;
  const reached = target != null && value >= target;
  const shortfall = target != null && value < target;
  return (
    <button
      className={`stage-card ${active ? "selected" : ""}`}
      onClick={onClick}
      aria-pressed={active}
    >
      <span className="stage-title">
        {STAGES[d.stage]}
        <span
          className={`tiny-dot ${reached ? "green" : shortfall ? "amber" : ""}`}
        />
      </span>
      <span className="stage-number">
        {number(value)}
        <small>{target != null ? ` / ${number(target)}` : ""}</small>
      </span>
      <span className="stage-unit">
        {d.tokens != null ? "tokens" : unit(d.unit)}
        {shortfall ? " · 目标未达" : reached ? " · 目标已达" : ""}
      </span>
      <span className={`progress ${shortfall ? "amber" : ""}`}>
        <span
          style={{
            width:
              target && target > 0
                ? `${Math.min(100, (value / target) * 100)}%`
                : "0%",
          }}
        />
      </span>
    </button>
  );
}

function ReasonBars({
  reasons,
  label,
}: {
  reasons: Record<string, number>;
  label: string;
}) {
  const entries = Object.entries(reasons)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map(([, n]) => n));
  if (!entries.length)
    return (
      <div className="muted small">
        没有已记录的{label}。缺失计数不会作为零值补齐。
      </div>
    );
  return (
    <div className="reason-bars">
      {entries.map(([name, n]) => (
        <div key={name} className="reason-row">
          <div>
            <span title={name}>
              {REASONS[name] || name.replaceAll("_", " ")}
            </span>
            <strong>{number(n)}</strong>
          </div>
          <div className="bar-track">
            <span style={{ width: `${(n / max) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function StepMetrics({ step: s }: { step: Step }) {
  const keys = [
    ...new Set([
      ...Object.keys(s.metrics_before),
      ...Object.keys(s.metrics_after),
    ]),
  ];
  if (!keys.length)
    return (
      <div className="snapshot-empty">
        <Icon name="chart" />
        <div>
          <strong>这一步没有质量快照</strong>
          <p>当前报告只提供构建计数。未测量的质量变化保持缺失。</p>
        </div>
      </div>
    );
  return (
    <>
      <div className="table-scroll">
        <table className="metric-table">
          <thead>
            <tr>
              <th>指标</th>
              <th>处理前</th>
              <th>处理后</th>
              <th>变化</th>
            </tr>
          </thead>
          <tbody>
            {keys.map((key) => {
              const before = s.metrics_before[key],
                after = s.metrics_after[key];
              const compatible =
                before?.status === "measured" &&
                after?.status === "measured" &&
                before.value != null &&
                after.value != null &&
                before.method === after.method &&
                before.unit === after.unit;
              const delta = compatible ? after!.value! - before!.value! : null;
              return (
                <tr key={key}>
                  <td>
                    <strong>{METRICS[key] || key}</strong>
                    <small className="metric-method">
                      {before?.method || after?.method}
                    </small>
                  </td>
                  <td>
                    {metricValue(before)}
                    <small>
                      {before?.denominator != null
                        ? `${number(before.numerator)} / ${number(before.denominator)}`
                        : "未测量"}
                    </small>
                  </td>
                  <td>
                    {metricValue(after)}
                    <small>
                      {after?.denominator != null
                        ? `${number(after.numerator)} / ${number(after.denominator)}`
                        : "未测量"}
                    </small>
                  </td>
                  <td className="delta">
                    {delta == null
                      ? "不可比"
                      : `${delta >= 0 ? "+" : ""}${number(delta * (before.unit === "ratio" ? 100 : 1))}${before.unit === "ratio" ? " pp" : ""}`}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="micro-copy">
        前后分母对应不同筛选集合。条件比例上升不代表独立验证，更不代表模型能力提升。
      </p>
    </>
  );
}

function FlowPanel({
  build,
  onOpen,
}: {
  build: BuildDetail;
  onOpen: (a: Artifact) => void;
}) {
  const [selected, setSelected] = useState(build.steps[0]?.step_id || "");
  useEffect(() => setSelected(build.steps[0]?.step_id || ""), [build.id]);
  const current =
    build.steps.find((s) => s.step_id === selected) || build.steps[0];
  if (!current)
    return <Empty title="还没有处理轨迹">构建报告没有可读取的步骤。</Empty>;
  return (
    <div className="flow-layout">
      <section className="panel timeline-panel">
        <div className="panel-head">
          <div>
            <h3>从来源到训练数据</h3>
            <p>按实际 builder 顺序记录，分支独立构建。</p>
          </div>
          <Badge>{build.steps.length} 个步骤</Badge>
        </div>
        <div className="timeline">
          {build.steps.map((s) => (
            <div
              className={`timeline-row ${s.step_id === current.step_id ? "active" : ""}`}
              key={s.step_id}
            >
              <span className="step-node">
                {s.order.toString().padStart(2, "0")}
              </span>
              <button
                className="step-button"
                aria-expanded={s.step_id === current.step_id}
                onClick={() => setSelected(s.step_id)}
              >
                <div className="step-top">
                  <strong>{operation(s)}</strong>
                  <Badge
                    kind={
                      s.status === "skipped"
                        ? "warning"
                        : s.status === "derived"
                          ? "neutral"
                          : "success"
                    }
                  >
                    {statusName(s.status)}
                  </Badge>
                </div>
                <div className="step-counts">
                  <span>
                    <b>{number(s.input_count.value)}</b>
                    <small>{unit(s.input_count.unit)}</small>
                  </span>
                  <Icon name="arrow" size={17} />
                  <span>
                    <b>{number(s.output_count.value)}</b>
                    <small>{unit(s.output_count.unit)}</small>
                  </span>
                  <Icon name="chevron" size={15} />
                </div>
                <div className="step-foot">
                  {s.rejected_count != null && s.rejected_count > 0 ? (
                    <span className="rejected">
                      拒绝 {number(s.rejected_count)}
                    </span>
                  ) : (
                    <span>首次失败原因按规则顺序计数</span>
                  )}
                  {s.unselected_count != null && s.unselected_count > 0 && (
                    <span>配额未选 {number(s.unselected_count)}</span>
                  )}
                </div>
              </button>
            </div>
          ))}
        </div>
      </section>
      <section className="panel step-detail" aria-label="选中步骤详情">
        <div className="panel-head">
          <div>
            <div className="eyebrow">
              STEP {current.order.toString().padStart(2, "0")}
            </div>
            <h3>{operation(current)}</h3>
          </div>
          <EvidenceButton artifact={build.artifacts[0]} onOpen={onOpen} />
        </div>
        <div className="detail-block">
          <h4>为什么没有保留？</h4>
          <p className="muted small">按该步骤的输入单位计数。</p>
          <ReasonBars
            reasons={current.first_rejection_reasons}
            label="质量拒绝"
          />
          {current.unselected_count != null && current.unselected_count > 0 && (
            <>
              <h4 className="subhead">配额选择 · 不属于质量拒绝</h4>
              <ReasonBars
                reasons={current.unselected_reasons}
                label="配额排除"
              />
            </>
          )}
        </div>
        <div className="detail-block">
          <h4>清洗前 → 清洗后</h4>
          <StepMetrics step={current} />
        </div>
        <div className="detail-block rule-notes">
          <h4>方法与解释边界</h4>
          <code>
            {current.operation} · v{current.operation_version}
          </code>
          {current.limitations.map((note, i) => (
            <p key={i}>{note}</p>
          ))}
        </div>
      </section>
    </div>
  );
}

function QualityPanel({
  build,
  dataset,
  onOpen,
}: {
  build: BuildDetail;
  dataset?: Dataset;
  onOpen: (a: Artifact) => void;
}) {
  if (!dataset)
    return (
      <Empty title="缺少数据集 manifest">
        无法确认最终数据集身份，也不能展示已验证的质量报告。
      </Empty>
    );
  const metrics = Object.entries(dataset.quality?.metrics || {});
  const distributions = Object.entries(dataset.source_distribution || {}).sort(
    (a, b) => b[1] - a[1],
  );
  const total = distributions.reduce((n, [, count]) => n + count, 0);
  const lengths = Object.entries(dataset.lengths || {}).filter(
    ([, value]) => value.count && value.count > 0,
  );
  return (
    <div className="quality-stack">
      <section className="panel">
        <div className="panel-head">
          <div>
            <h3>{STAGES[dataset.stage]} · 质量证据</h3>
            <p>检查器的方法、分母和缺失项与数值同样重要。</p>
          </div>
          <EvidenceButton
            artifact={dataset.quality_artifact || dataset.manifest}
            onOpen={onOpen}
          />
        </div>
        <div className="quality-content">
          {dataset.quality ? (
            <>
              <div className="gate-row">
                <Badge
                  kind={
                    dataset.quality.gate.summary_status === "FAIL"
                      ? "danger"
                      : "neutral"
                  }
                >
                  {statusName(dataset.quality.gate.summary_status)}
                </Badge>
                <span>结构检查与完整语义 gate 分开报告。</span>
              </div>
              <div className="checks">
                {dataset.quality.gate.checks.map((check) => (
                  <span
                    key={check.check_id}
                    className={check.result === "FAIL" ? "danger-text" : ""}
                  >
                    <Icon
                      name={check.result === "PASS" ? "check" : "info"}
                      size={14}
                    />
                    {CHECK_NAMES[check.check_id] || check.check_id} ·{" "}
                    {statusName(check.result)}
                  </span>
                ))}
              </div>
              <div className="table-scroll">
                <table className="metric-table quality-table">
                  <thead>
                    <tr>
                      <th>指标 / 方法</th>
                      <th>值</th>
                      <th>分子 / 分母</th>
                      <th>证据状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.map(([key, metric]) => (
                      <tr key={key}>
                        <td>
                          <strong>{METRICS[key] || key}</strong>
                          <details>
                            <summary>方法与局限</summary>
                            <code>{metric.method}</code>
                            {metric.limitations.map((v, i) => (
                              <p key={i}>{v}</p>
                            ))}
                            <p>
                              类别：
                              {metric.class === "dataset_quality"
                                ? "数据质量"
                                : metric.class === "build_diagnostic"
                                  ? "构建诊断"
                                  : "模型 / recipe 适用性"}
                            </p>
                          </details>
                        </td>
                        <td className="value-cell">
                          {metricValue(metric)}
                          <small>
                            {metric.unit !== "ratio" ? unit(metric.unit) : ""}
                          </small>
                        </td>
                        <td>
                          {metric.denominator != null
                            ? `${number(metric.numerator)} / ${number(metric.denominator)}`
                            : "—"}
                        </td>
                        <td>
                          <Badge
                            kind={
                              metric.status === "measured"
                                ? "success"
                                : "neutral"
                            }
                          >
                            {statusName(metric.status)}
                          </Badge>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <Empty title="历史数据没有阶段质量快照">
              已有数量和分布不等同于质量已通过。当前版本没有回写历史 Smoke
              文件。
            </Empty>
          )}
        </div>
      </section>
      <div className="distribution-grid">
        <section className="panel">
          <div className="panel-head">
            <div>
              <h3>来源组成</h3>
              <p>观察选择偏差，不以来源均衡推断代表性。</p>
            </div>
          </div>
          <div className="distribution-body">
            {distributions.length ? (
              distributions.slice(0, 10).map(([name, count]) => (
                <div className="source-row" key={name}>
                  <div>
                    <span>{name}</span>
                    <small>
                      {number(count)} · {number((count / total) * 100)}%
                    </small>
                  </div>
                  <div className="bar-track">
                    <span style={{ width: `${(count / total) * 100}%` }} />
                  </div>
                </div>
              ))
            ) : (
              <div className="snapshot-empty">此数据集没有来源分布快照。</div>
            )}
          </div>
        </section>
        <section className="panel">
          <div className="panel-head">
            <div>
              <h3>Token 长度分位数</h3>
              <p>方法随 tokenizer / 最终模板而定。</p>
            </div>
          </div>
          <div className="distribution-body">
            {lengths.length ? (
              lengths.map(([name, values]) => (
                <div className="length-set" key={name}>
                  <code>{name}</code>
                  <div className="percentiles">
                    {["p50", "p90", "p95", "p99"].map((p) => (
                      <div key={p}>
                        <small>{p.toUpperCase()}</small>
                        <strong>{number(values[p])}</strong>
                      </div>
                    ))}
                  </div>
                  <small className="muted">
                    覆盖 {number(values.count)} 条记录
                  </small>
                </div>
              ))
            ) : (
              <div className="snapshot-empty">
                长度分位数未测量。缺失值不作为 0 展示。
              </div>
            )}
          </div>
        </section>
      </div>
      {build.is_demo && (
        <Notice kind="warning">
          这里是合成示例，长度来自测试用 whitespace
          tokenizer，不能作为正式训练预算。
        </Notice>
      )}
    </div>
  );
}

function EffectsPanel({ build, onOpenRun }: { build: BuildDetail; onOpenRun: (runId: string) => void }) {
  return (
    <section className="panel effects-panel">
      <div className="panel-head">
        <div>
          <h3>数据变化，是否改变了模型？</h3>
          <p>沿显式 dataset ID 血缘连接训练和评测。</p>
        </div>
        <Badge>模型效果未评估</Badge>
      </div>
      <div className="effect-chain">
        {[
          {
            icon: "data",
            label: "数据版本",
            note: `${build.datasets.length} 个 manifest`,
            done: build.datasets.length > 0,
          },
          {
            icon: "branch",
            label: "训练 run",
            note: build.related_runs.length
              ? `${build.related_runs.length} 个显式关联`
              : "血缘尚未接通",
            done: build.related_runs.length > 0,
          },
          {
            icon: "chart",
            label: "统一评测",
            note: build.related_runs.some((r) => r.metrics.length)
              ? "有已关联指标"
              : "没有关联证据",
            done: build.related_runs.some((r) => r.metrics.length),
          },
        ].map((item, i) => (
          <div
            className={`chain-node ${item.done ? "available" : ""}`}
            key={item.label}
          >
            {i > 0 && (
              <span className="chain-arrow">
                <Icon name="arrow" />
              </span>
            )}
            <span className="chain-icon">
              <Icon name={item.icon} size={24} />
            </span>
            <strong>{item.label}</strong>
            <small>{item.note}</small>
          </div>
        ))}
      </div>
      <Notice>
        {build.model_effect.reason}即使存在任务评测分数，也需要同
        parent、同预算、同评测的清洗策略对照，才能解释策略效果。
      </Notice>
      {build.related_runs.length ? (
        <div className="linked-runs">
          {build.related_runs.map((run) => (
            <article key={run.run_id}>
              <div className="run-head">
                <strong>{run.run_id}</strong>
                <Badge>{run.status}</Badge>
              </div>
              <code>{run.source}</code>
              <p>{run.note}</p>
              <button className="text-button" onClick={() => onOpenRun(run.run_id)}>打开关联 Run →</button>
              {run.metrics.length > 0 && (
                <table className="metric-table">
                  <thead>
                    <tr>
                      <th>评测项</th>
                      <th>结果</th>
                      <th>样本量</th>
                    </tr>
                  </thead>
                  <tbody>
                    {run.metrics.map((m) => (
                      <tr key={m.name}>
                        <td>{m.name}</td>
                        <td>
                          {m.value == null
                            ? "—"
                            : m.unit === "ratio"
                              ? `${number(m.value * 100)}%`
                              : number(m.value)}
                        </td>
                        <td>{number(m.samples)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </article>
          ))}
        </div>
      ) : (
        <Empty title="尚未找到可验证的数据 → 模型关联">
          已有训练目录不会自动匹配到数据版本。页面只接受 lineage.json
          中明确声明的 dataset ID，不按阶段或文件名猜测。
        </Empty>
      )}
    </section>
  );
}

function ArtifactDialog({
  artifact,
  onClose,
}: {
  artifact: Artifact | null;
  onClose: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [content, setContent] = useState<{
    path: string;
    sha256: string;
    data: unknown;
    notice: string;
  } | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    if (!artifact) return;
    const controller = new AbortController();
    setContent(null);
    setError("");
    dialog.current?.showModal();
    request<{ path: string; sha256: string; data: unknown; notice: string }>(
      `/api/artifacts/${artifact.id}`,
      controller.signal,
    )
      .then(setContent)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      });
    return () => {
      controller.abort();
      dialog.current?.close();
    };
  }, [artifact]);
  return (
    <dialog
      className="artifact-dialog"
      ref={dialog}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      aria-labelledby="artifact-title"
    >
      <div className="dialog-head">
        <div>
          <h3 id="artifact-title">来源证据</h3>
          <p>{artifact?.path}</p>
        </div>
        <button
          className="icon-button"
          autoFocus
          onClick={onClose}
          aria-label="关闭来源预览"
        >
          <Icon name="close" />
        </button>
      </div>
      {error ? (
        <Notice kind="warning">{error}</Notice>
      ) : content ? (
        <>
          <div className="preview-info">
            <Badge>只读 · 汇总预览</Badge>
            <p>{content.notice}</p>
            <code>SHA256 {content.sha256}</code>
          </div>
          <pre>{JSON.stringify(content.data, null, 2)}</pre>
        </>
      ) : (
        <div className="loading-inline">正在读取来源文件…</div>
      )}
    </dialog>
  );
}

function App() {
  const initial = useRef(workspaceRouteStateFromSearch(window.location.search)).current;
  const [themeMode, setThemeMode] = useState<ThemeMode>(readThemeMode);
  const [systemDark, setSystemDark] = useState(systemPrefersDark);
  const effectiveTheme = resolveTheme(themeMode, systemDark);
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [scope, setScope] = useState(initial.scope);
  const [query, setQuery] = useState(initial.query);
  const [filter, setFilter] = useState(initial.filter);
  const [buildId, setBuildId] = useState(initial.build);
  const [stage, setStage] = useState(initial.stage);
  const [tab, setTab] = useState(TABS.some((item) => item.id === initial.tab) ? initial.tab : "flow");
  const [refresh, setRefresh] = useState(0);
  const [matrixId, setMatrixId] = useState(initial.matrix);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [detail, setDetail] = useState<BuildDetail | null>(null);
  const [detailError, setDetailError] = useState("");
  const [artifact, setArtifact] = useState<Artifact | null>(null);
  const [page, setPage] = useState<"data" | "comparison" | "policy-effects" | "policies" | "targeted" | "launcher" | "model-playground" | "runs" | "workbench">(() => {
    return initial.page;
  });
  useEffect(() => {
    writeThemeMode(themeMode);
  }, [themeMode]);
  useEffect(() => {
    if (themeMode !== "system") return;
    setSystemDark(systemPrefersDark());
    return subscribeSystemTheme(setSystemDark);
  }, [themeMode]);
  useLayoutEffect(() => {
    applyThemeToDocument(effectiveTheme);
  }, [effectiveTheme]);
  const navigatePage = (target: typeof page) => {
    const params = workspaceSearchForPage(target, window.location.search, {
      scope,
      build: buildId,
      stage,
      tab,
      query,
      filter,
      matrix: matrixId,
    });
    const queryString = params.toString();
    window.history.pushState(null, "", `${window.location.pathname}${queryString ? `?${queryString}` : ""}`);
    setPage(target);
  };
  useEffect(() => {
    const restore = () => {
      const restored = workspaceRouteStateFromSearch(window.location.search);
      setPage(restored.page);
      setScope(restored.scope);
      setBuildId(restored.build);
      setStage(restored.stage);
      setTab(TABS.some((item) => item.id === restored.tab) ? restored.tab : "flow");
      setQuery(restored.query);
      setFilter(restored.filter);
      setMatrixId(restored.matrix);
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  const openRunDetail = (runId: string) => {
    const params = new URLSearchParams({ page: "runs", run: runId });
    const current = new URLSearchParams(window.location.search);
    for (const key of ["return_to", "dataset_id", "stage"]) {
      const value = current.get(key);
      if (value) params.set(key, value);
    }
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    setPage("runs");
  };
  const tryModel = (modelId: string, runId: string) => {
    const params = new URLSearchParams({ page: "model-playground", model_id: modelId, run_id: runId, return_to: "runs" });
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    setPage("model-playground");
  };
  const returnToWorkspaceOrigin = () => {
    const current = new URLSearchParams(window.location.search);
    if (current.get("return_to") === "runs") {
      const sourceRunId = current.get("source_run_id");
      const params = new URLSearchParams({ page: "runs" });
      if (sourceRunId) params.set("run", sourceRunId);
      window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
      setPage("runs");
      return;
    }
    if (current.get("return_to") === "workbench") {
      const params = new URLSearchParams({ page: "workbench" });
      const datasetId = current.get("dataset_id");
      if (datasetId) params.set("dataset_id", datasetId);
      window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
      setPage("workbench");
      return;
    }
    if (current.get("return_to") === "launcher") {
      const params = launcherReturnSearch(window.location.search);
      window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
      setPage("launcher");
      return;
    }
    navigatePage("data");
  };
  const openTrainingWithDataset = (datasetId: string, datasetStage: string) => {
    const params = registeredDatasetTrainingSearch(datasetId, datasetStage, window.location.search);
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    setPage("launcher");
  };
  const prepareGrpoData = (context: { profileId: string; parentModelId: string }) => {
    const params = grpoPreparationSearch(context.profileId, context.parentModelId);
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    setPage("workbench");
  };
  const openDatasetMixture = (stage: string) => {
    const params = new URLSearchParams({ page: "workbench", compose_stage: stage, return_to: "launcher" });
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    setPage("workbench");
  };
  const openLauncherRun = (runId: string) => {
    const params = new URLSearchParams({ page: "launcher", stage: "sft", launcher_run: runId });
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    setPage("launcher");
  };
  const openGenerationTask = (targeted: boolean, context: Record<string, string> = {}) => {
    const params = new URLSearchParams({ page: "workbench" });
    if (targeted) {
      params.set("targeted_generation", "1");
      for (const key of ["scenario_id", "bundle_id", "eval_id", "model_id", "dataset_id"]) {
        if (context[key]) params.set(key, context[key]);
      }
      params.set("return_to", "targeted");
    }
    window.history.pushState(null, "", `${window.location.pathname}?${params.toString()}`);
    setPage("workbench");
  };
  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError("");
    request<Catalog>("/api/data", controller.signal)
      .then(setCatalog)
      .catch((e) => {
        if (e.name !== "AbortError") setError(e.message);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [refresh]);
  const scoped =
    catalog?.builds.filter((b) => b.is_demo === (scope === "demo")) || [];
  const builds = scoped.filter(
    (b) =>
      `${b.name} ${b.source} ${b.path}`
        .toLowerCase()
        .includes(query.toLowerCase()) &&
      (filter === "all" || b.datasets.some((d) => d.stage === filter)),
  );
  const selected: BuildSummary | undefined =
    builds.find((b) => b.id === buildId) || builds[0];
  useEffect(() => {
    if (selected && selected.id !== buildId) setBuildId(selected.id);
  }, [selected?.id, buildId]);
  useEffect(() => {
    setDetail(null);
    setDetailError("");
    if (!selected) return;
    const controller = new AbortController();
    request<BuildDetail>(`/api/data/${selected.id}`, controller.signal)
      .then(setDetail)
      .catch((e) => {
        if (e.name !== "AbortError") setDetailError(e.message);
      });
    return () => controller.abort();
  }, [selected?.id, refresh, catalog?.scanned_at]);
  const dataset =
    detail?.datasets.find((d) => d.stage === stage) ||
    detail?.datasets.find((d) => d.stage === filter) ||
    detail?.datasets[0];
  useEffect(() => {
    if (page === "runs") return;
    const params = workspaceSearchForPage(page, window.location.search, {
      scope,
      build: selected?.id || buildId,
      stage: dataset?.stage || stage,
      tab,
      query,
      filter,
      matrix: matrixId,
    });
    const queryString = params.toString();
    window.history.replaceState(
      null,
      "",
      `${window.location.pathname}${queryString ? `?${queryString}` : ""}`,
    );
  }, [scope, selected?.id, buildId, dataset?.stage, stage, tab, query, filter, page, matrixId]);
  const currentModule = workspaceModuleFromPage(page);
  const currentPageDataError = page === "data" && Boolean(error);
  const currentModuleName = PRIMARY_MODULES.find((item) => item.module === currentModule)?.name || "数据";
  const currentModuleViews = MODULE_VIEWS[currentModule];
  return (
    <div className="app-shell" data-theme={effectiveTheme}>
      <aside className="sidebar">
        <a className="brand" href="/" aria-label="OneGPU PostTrain Data">
          <span className="brand-mark">
            O<span />
          </span>
          <span>
            OneGPU<small>POSTTRAIN LAB</small>
          </span>
        </a>
        <div className="sidebar-label">WORKSPACE</div>
        <WorkspaceNavigation
          page={page}
          onNavigate={(destination) => navigatePage(destination as typeof page)}
          renderIcon={(name) => <Icon name={name} />}
        />
        <div className="sidebar-note">
          <Icon name="beaker" />
          <strong>One GPU. Full evidence.</strong>
          <p>
            从数据处理到模型评测，
            <br />
            让每一步都有据可查。
          </p>
        </div>
        <div className="sidebar-bottom">
          <div className="hardware-dot" />
          <div>
            <strong>单卡后训练实验室</strong>
            <small>可复现 · 开源 · 本地优先</small>
          </div>
        </div>
      </aside>
      <div className="main-shell">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>实验室</span>
            <span>/</span>
            <strong>{currentModuleName}</strong>
            <span>/</span>
            <strong>{PAGE_LABELS[page]}</strong>
          </div>
          <div className="topbar-meta">
            <span className={`connection-dot ${currentPageDataError ? "offline" : ""}`} />
            <span>{currentPageDataError ? "数据报告暂不可用" : "本地 API"}</span>
            <span className="topbar-divider" />
            <Icon name="lock" size={14} />
            <span>{page === "policies" ? "本地策略实验" : page === "targeted" ? "配置草稿 · 无远程调用" : page === "launcher" ? "单机训练 · 显式启用" : page === "runs" ? "本地记录 · 只读浏览" : "只读模式"}</span>
          </div>
          <ThemePreferenceControl mode={themeMode} onChange={setThemeMode} />
        </header>
        {currentModuleViews.length > 1 && <nav className="workspace-mobile-subnav" aria-label={`${currentModuleName}视图`}>
          {currentModuleViews.map((item) => <button key={item.page} className={page === item.page ? "nav-active" : ""} aria-current={page === item.page ? "page" : undefined} onClick={() => navigatePage(item.page as typeof page)}>{item.name}</button>)}
        </nav>}
        <ActivityTasks onOpenRun={openRunDetail} onOpenGeneration={openGenerationTask} />
        <main hidden={page !== "data"}>
          <section className="hero">
            <div>
              <div className="eyebrow">
                DATA EXPLORER <span>／</span> 数据证据
              </div>
              <h1>
                看清数据如何变化<span>。</span>
              </h1>
              <p>追踪处理步骤、质量与覆盖变化，再沿血缘寻找模型效果的证据。</p>
            </div>
            <div className="data-hero-actions"><button className="button secondary" onClick={() => navigatePage("workbench")}>打开数据集工作台 <span aria-hidden="true">→</span></button><button
              className="button secondary"
              onClick={() => setRefresh((n) => n + 1)}
              disabled={loading}
            >
              <span className={loading ? "spinning" : ""}>
                <Icon name="refresh" size={16} />
              </span>
              {loading ? "读取中" : "重新读取"}
            </button></div>
          </section>
          <div className="workspace-toolbar">
            <div className="scope-switch" role="group" aria-label="数据范围">
              <button
                className={scope === "real" ? "active" : ""}
                onClick={() => setScope("real")}
              >
                真实产物{" "}
                <span>
                  {catalog?.builds.filter((b) => !b.is_demo).length ?? "—"}
                </span>
              </button>
              <button
                className={scope === "demo" ? "active" : ""}
                onClick={() => setScope("demo")}
              >
                <Icon name="beaker" size={14} />
                合成示例{" "}
                <span>
                  {catalog?.builds.filter((b) => b.is_demo).length ?? "—"}
                </span>
              </button>
            </div>
            <span className="scan-time">
              {catalog
                ? `读取于 ${new Date(catalog.scanned_at).toLocaleTimeString("zh-CN", { hour12: false })}`
                : "正在发现本地产物…"}
            </span>
          </div>
          {error && (
            <Notice kind="warning">
              {error}
              {catalog && " 当前显示上次读取的数据，可能已经过期。"}
            </Notice>
          )}
          {catalog?.issues.length ? (
            <details className="scan-issues">
              <summary>
                <Icon name="info" size={15} />
                {catalog.issues.length} 个产物读取问题
              </summary>
              {catalog.issues.map((issue, i) => (
                <p key={i}>
                  <code>{issue.path}</code> — {issue.message}
                </p>
              ))}
            </details>
          ) : null}
          {scope === "demo" && (
            <Notice kind="warning">
              <strong>合成示例，不是真实训练数据或模型结果。</strong>{" "}
              用于演示新版报告交互，源正确性标签由 fixture
              显式指定，未执行独立数学验证。
            </Notice>
          )}
          <div className="explorer-layout">
            <section className="build-list panel">
              <div className="build-list-head">
                <h2>
                  数据构建<span>{builds.length}</span>
                </h2>
                <label className="search-box">
                  <Icon name="search" size={16} />
                  <input
                    aria-label="搜索数据构建"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="搜索来源 / 路径"
                  />
                </label>
                <label className="filter-label">
                  <span>训练阶段</span>
                  <select
                    aria-label="按训练阶段筛选"
                    value={filter}
                    onChange={(e) => {
                      setFilter(e.target.value);
                      setStage(e.target.value === "all" ? "" : e.target.value);
                    }}
                  >
                    <option value="all">全部阶段</option>
                    {Object.entries(STAGES).map(([k, v]) => (
                      <option value={k} key={k}>
                        {v}
                      </option>
                    ))}
                  </select>
                </label>
              </div>
              <div className="build-options">
                {loading && !catalog ? (
                  <div className="skeleton list-skeleton" />
                ) : builds.length ? (
                  builds.map((b) => (
                    <button
                      className={`build-option ${selected?.id === b.id ? "active" : ""}`}
                      key={b.id}
                      onClick={() => {
                        setBuildId(b.id);
                        setStage("");
                      }}
                      aria-pressed={selected?.id === b.id}
                    >
                      <span className="build-icon">
                        <Icon name="data" size={20} />
                      </span>
                      <span className="build-option-body">
                        <span className="build-option-title">
                          {b.name}
                          <Icon name="arrow" size={14} />
                        </span>
                        <small title={b.source}>{b.source}</small>
                        <span className="build-option-tags">
                          <Badge>
                            {b.mode === "legacy" ? "旧版报告" : "版本化报告"}
                          </Badge>
                          <span>{b.datasets.length} 个数据版本</span>
                        </span>
                      </span>
                    </button>
                  ))
                ) : (
                  <div className="list-empty">
                    {query || filter !== "all"
                      ? "没有符合筛选条件的构建。"
                      : scope === "demo"
                        ? "尚未生成合成示例。详见本地 UI 使用文档。"
                        : "未发现真实数据报告。先使用 CLI 构建数据。"}
                    {(query || filter !== "all") && (
                      <button
                        className="text-button"
                        onClick={() => {
                          setQuery("");
                          setFilter("all");
                        }}
                      >
                        清除筛选
                      </button>
                    )}
                  </div>
                )}
              </div>
              <div className="build-list-footer">
                <Icon name="lock" size={13} />
                仅扫描已配置的数据根目录
              </div>
            </section>
            <div className="build-workspace">
              {detailError ? (
                <Notice kind="warning">{detailError}</Notice>
              ) : selected && !detail ? (
                <div className="detail-skeleton">
                  <div className="skeleton" />
                  <div className="skeleton" />
                  <div className="skeleton" />
                </div>
              ) : !selected ? (
                <section className="panel">
                  <Empty
                    title={
                      scope === "real"
                        ? "等待第一份数据报告"
                        : "演示数据尚未生成"
                    }
                  >
                    这里将展示真实的构建证据。不会填入虚构的质量数值。
                  </Empty>
                </section>
              ) : (
                detail && (
                  <>
                    <section className="build-overview">
                      <div className="build-heading">
                        <div>
                          <div className="eyebrow">
                            {detail.is_demo
                              ? "SYNTHETIC EXAMPLE"
                              : "LOCAL ARTIFACTS"}
                          </div>
                          <h2>
                            {detail.name}
                            <span>{detail.is_demo ? "Demo" : "Smoke"}</span>
                          </h2>
                          <div className="source-line">
                            <code title={detail.path}>{detail.path}</code>
                            <span title={detail.revision || "版本未记录"}>
                              rev{" "}
                              {detail.revision
                                ? detail.revision.slice(0, 9)
                                : "—"}
                            </span>
                          </div>
                        </div>
                        <EvidenceButton
                          artifact={detail.artifacts[0]}
                          onOpen={setArtifact}
                        >
                          构建报告
                        </EvidenceButton>
                      </div>
                      <div className="summary-cards">
                        <div className="summary-card">
                          <span>
                            <Icon name="data" size={15} />
                            处理输入
                          </span>
                          <strong>
                            {number(detail.raw_count)}
                            <small>{unit(detail.raw_unit)}</small>
                          </strong>
                          <p>仅覆盖实际扫描的来源前缀</p>
                        </div>
                        <div className="summary-card">
                          <span>
                            <Icon name="box" size={15} />
                            最终数据版本
                          </span>
                          <strong>
                            {detail.datasets.length}
                            <small>个 manifest</small>
                          </strong>
                          <p>样本、偏好对与 tokens 分别计数</p>
                        </div>
                        <div className="summary-card effect-summary">
                          <span>
                            <Icon name="chart" size={15} />
                            对模型的影响
                          </span>
                          <strong>
                            未评估
                            <span className="status-ring" />
                          </strong>
                          <p>需要登记的受控训练对照</p>
                        </div>
                      </div>
                      <div className="stage-cards">
                        {detail.datasets.map((d) => (
                          <StageCard
                            key={d.stage}
                            dataset={d}
                            active={dataset?.stage === d.stage}
                            onClick={() => {
                              setStage(d.stage);
                              setTab("quality");
                            }}
                          />
                        ))}
                      </div>
                      {detail.warnings.length > 0 && (
                        <div className="build-warnings">
                          <Icon name="info" size={16} />
                          <div>
                            {detail.warnings.map((w, i) => (
                              <p key={i}>{w}</p>
                            ))}
                          </div>
                        </div>
                      )}
                    </section>
                    <div
                      className="tabbar"
                      role="tablist"
                      aria-label="数据证据视图"
                    >
                      {TABS.map((t, i) => (
                        <button
                          key={t.id}
                          role="tab"
                          id={`tab-${t.id}`}
                          aria-controls={`panel-${t.id}`}
                          aria-selected={tab === t.id}
                          tabIndex={tab === t.id ? 0 : -1}
                          className={tab === t.id ? "active" : ""}
                          onClick={() => setTab(t.id)}
                          onKeyDown={(event) => {
                            if (
                              event.key === "ArrowRight" ||
                              event.key === "ArrowLeft"
                            ) {
                              event.preventDefault();
                              const next =
                                TABS[
                                  (i +
                                    (event.key === "ArrowRight"
                                      ? 1
                                      : TABS.length - 1)) %
                                    TABS.length
                                ];
                              setTab(next.id);
                              document
                                .getElementById(`tab-${next.id}`)
                                ?.focus();
                            }
                          }}
                        >
                          {t.label}
                          {t.id === "flow" && (
                            <span>{detail.steps.length}</span>
                          )}
                        </button>
                      ))}
                      <span className="tabbar-context">
                        {dataset && STAGES[dataset.stage]}
                      </span>
                    </div>
                    <div
                      role="tabpanel"
                      id={`panel-${tab}`}
                      aria-labelledby={`tab-${tab}`}
                      key={`${detail.id}-${tab}`}
                    >
                      {tab === "flow" ? (
                        <FlowPanel build={detail} onOpen={setArtifact} />
                      ) : tab === "quality" ? (
                        <QualityPanel
                          build={detail}
                          dataset={dataset}
                          onOpen={setArtifact}
                        />
                      ) : (
                        <EffectsPanel build={detail} onOpenRun={openRunDetail} />
                      )}
                    </div>
                    <footer className="workspace-footer">
                      <span>
                        <Icon name="lock" size={13} />
                        本页只读取报告，不修改数据或训练状态。
                      </span>
                      <EvidenceButton
                        artifact={dataset?.manifest}
                        onOpen={setArtifact}
                      >
                        Dataset manifest
                      </EvidenceButton>
                    </footer>
                  </>
                )
              )}
            </div>
          </div>
          <footer className="page-footer">
            <span>OneGPU-PostTrain</span>
            <span>One GPU. Full Post-Training. Reproducible Experiments.</span>
          </footer>
        </main>
        {page === "comparison" && <main className="comparison-page">
          <EvaluationMatrixPage matrixId={matrixId} onMatrixChange={setMatrixId} onBackToData={() => navigatePage("data")} />
        </main>}
        {page === "workbench" && <DatasetWorkbench onViewEvidence={(nextBuild, nextStage) => {
          const selectedBuild = catalog?.builds.find((item) => item.id === nextBuild);
          if (selectedBuild) setScope(selectedBuild.is_demo ? "demo" : "real");
          setBuildId(nextBuild); setStage(nextStage); setFilter("all"); setTab("effects"); navigatePage("data");
        }} onOpenTargeted={() => navigatePage("targeted")} onOpenRun={openRunDetail} onUseDataset={openTrainingWithDataset} />}
        {page === "policies" && <PolicyEditor visible onBack={() => navigatePage("data")} onViewEffects={() => navigatePage("policy-effects")} />}
        {page === "policy-effects" && <PolicyEditor visible reportsOnly onBack={() => navigatePage("policies")} onViewEffects={() => navigatePage("policy-effects")} />}
        {page === "targeted" && <TargetedExperiments />}
        {page === "launcher" && <Launcher onBack={returnToWorkspaceOrigin} onOpenRun={openRunDetail} onComposeDatasets={openDatasetMixture} onPrepareGrpoData={prepareGrpoData} />}
        {page === "model-playground" && <ModelPlayground onBack={() => navigatePage("launcher")} initialModelId={new URLSearchParams(window.location.search).get("model_id") || ""} returnRunId={new URLSearchParams(window.location.search).get("run_id") || ""} onOpenRun={openRunDetail} />}
        {page === "runs" && <Runs onBack={returnToWorkspaceOrigin} onTryModel={tryModel} />}
      </div>
      <ArtifactDialog artifact={artifact} onClose={() => setArtifact(null)} />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
