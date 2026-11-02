export type TaskRequestState = "idle" | "preflight_pending" | "submit_pending" | "submission_unknown" | "token_expired";
export type TaskPreflightState = "none" | "ready" | "blocked";
export type TaskAction = "preflight" | "submit" | "inspect" | "open_run" | "edit" | "none";
export type TaskTone = "neutral" | "working" | "success" | "warning" | "danger";

export type TaskDisplayInput = {
  request: TaskRequestState;
  preflight?: TaskPreflightState;
  run?: { run_id: string; status: string } | null;
};

export type TaskDisplayState = {
  phase: "editing" | "preflight_pending" | "preflight_ready" | "preflight_blocked" | "submit_pending" |
    "submission_unknown" | "token_expired" | "run_created" | "run_running" | "run_terminal" | "run_unknown";
  title: string;
  description: string;
  tone: TaskTone;
  action: TaskAction;
  actionLabel: string;
  actionDisabled: boolean;
  runId?: string;
};

/** The server Run is authoritative once an ID exists; a request spinner never stands in for a GPU job. */
export function taskDisplayState(input: TaskDisplayInput): TaskDisplayState {
  const run = input.run;
  if (run?.run_id) {
    const base = { runId: run.run_id, action: "open_run" as const, actionLabel: "查看 Run", actionDisabled: false };
    if (run.status === "queued" || run.status === "starting") return {
      ...base, phase: "run_created", title: "Run 已创建，等待执行", description: "提交已完成；请从 Run 查看后续状态。", tone: "working",
    };
    if (run.status === "running") return {
      ...base, phase: "run_running", title: "后台运行中", description: "页面可以离开；进度以服务端 Run 为准。", tone: "working",
    };
    if (["completed", "failed", "partial", "interrupted", "oom", "cancelled"].includes(run.status)) {
      const labels: Record<string, string> = {
        completed: "任务完成", failed: "任务失败", partial: "部分完成", interrupted: "任务已中断",
        oom: "显存不足", cancelled: "任务已取消",
      };
      return { ...base, phase: "run_terminal", title: labels[run.status],
        description: "任务已结束；请查看 Run 的结果、日志和证据。",
        tone: run.status === "completed" ? "success" : run.status === "partial" ? "warning" : "danger" };
    }
    return { ...base, phase: "run_unknown", title: "Run 状态待确认", description: "已收到 Run ID，但状态尚未核实；请查看 Run。", tone: "warning" };
  }
  if (input.request === "preflight_pending") return {
    phase: "preflight_pending", title: "正在检查配置", description: "只读预检请求尚未返回。", tone: "working",
    action: "none", actionLabel: "检查中…", actionDisabled: true,
  };
  if (input.request === "submit_pending") return {
    phase: "submit_pending", title: "正在提交启动请求", description: "请勿重复点击；此时尚不能认定 Run 已创建。", tone: "working",
    action: "none", actionLabel: "提交中…", actionDisabled: true,
  };
  if (input.request === "submission_unknown") return {
    phase: "submission_unknown", title: "启动结果待核实", description: "请求可能已到达服务端。先查询任务，再决定是否重新提交。", tone: "warning",
    action: "inspect", actionLabel: "检查任务状态", actionDisabled: false,
  };
  if (input.request === "token_expired") return {
    phase: "token_expired", title: "确认已失效", description: "重新预检以取得新确认；不会自动重试启动。", tone: "warning",
    action: "preflight", actionLabel: "重新预检", actionDisabled: false,
  };
  if (input.preflight === "blocked") return {
    phase: "preflight_blocked", title: "存在阻断", description: "请修正配置后重新预检。", tone: "danger",
    action: "edit", actionLabel: "返回修改", actionDisabled: false,
  };
  if (input.preflight === "ready") return {
    phase: "preflight_ready", title: "预检通过，等待确认", description: "启动将创建新的 Run，并可能占用本机 GPU。", tone: "success",
    action: "submit", actionLabel: "确认并启动", actionDisabled: false,
  };
  return { phase: "editing", title: "配置任务", description: "完成必填项后执行只读预检。", tone: "neutral",
    action: "preflight", actionLabel: "检查配置", actionDisabled: false };
}
