import { useEffect, useState } from "react";

type Activity = {
  kind: "generation" | "training" | "evaluation";
  id: string;
  status: string;
  detail: string;
  targeted?: boolean;
  context?: Record<string, string>;
};

const GENERATION_KEY = "onegpu.sft-generation.execution";
const TARGETED_GENERATION_KEY = "onegpu.sftgen.execution";

function readTargetedContext(): Record<string, string> {
  try {
    const value = JSON.parse(window.sessionStorage.getItem("onegpu.sftgen.context") || "{}");
    return value && typeof value === "object" ? value : {};
  } catch { return {}; }
}

function active(status: unknown) {
  return status === "queued" || status === "starting" || status === "running";
}

async function json(url: string, init?: RequestInit) {
  const response = await fetch(url, init);
  if (!response.ok) throw new Error(String(response.status));
  return response.json();
}

export function ActivityTasks({ onOpenRun, onOpenGeneration }: {
  onOpenRun: (runId: string) => void;
  onOpenGeneration: (targeted: boolean, context?: Record<string, string>) => void;
}) {
  const [activities, setActivities] = useState<Activity[]>([]);

  useEffect(() => {
    let disposed = false;
    let token = "";
    const refresh = async () => {
      const next: Activity[] = [];
      try {
        const current = await json("/api/launcher/run");
        const run = current?.run;
        if (run && (active(run.status) || active(run.evaluation_status))) {
          const evaluation = active(run.evaluation_status);
          next.push({ kind: evaluation ? "evaluation" : "training", id: run.run_id, status: evaluation ? run.evaluation_status : run.status,
            detail: evaluation ? "阶段评测正在运行" : "阶段训练正在运行" });
        }
      } catch { /* The launcher may be disabled; generation remains independent. */ }

      try {
        if (!token) {
          const capabilities = await json("/api/synthetic/sft/capabilities");
          token = capabilities?.enabled && typeof capabilities.token === "string" ? capabilities.token : "";
        }
        if (token) {
          const regularId = window.localStorage.getItem(GENERATION_KEY);
          const targetedId = window.sessionStorage.getItem(TARGETED_GENERATION_KEY);
          const refs = [
            ...(regularId ? [{ id: regularId, targeted: false, context: {} }] : []),
            ...(targetedId ? [{ id: targetedId, targeted: true, context: readTargetedContext() }] : []),
          ];
          const states = await Promise.all(refs.map(async (ref) => {
            try { return { ref, state: await json(`/api/synthetic/sft/runs/${encodeURIComponent(ref.id)}`, { headers: { "X-Local-Synthetic-Token": token } }) }; }
            catch { return null; }
          }));
          for (const item of states) {
            if (!item || !active(item.state.status)) continue;
            const progress = item.state.progress;
            const detail = progress
              ? `候选 ${progress.completed_candidates ?? 0}/${progress.candidate_target ?? "—"} · 请求 ${progress.attempted_requests ?? 0}/${progress.request_budget ?? "—"}`
              : "生成任务正在运行";
            next.push({ kind: "generation", id: item.ref.id, status: item.state.status, detail,
              targeted: item.ref.targeted, context: item.ref.context });
          }
        }
      } catch { /* Do not let an optional source hide other active tasks. */ }
      if (!disposed) setActivities(next);
    };
    void refresh();
    const timer = window.setInterval(() => void refresh(), 2500);
    return () => { disposed = true; window.clearInterval(timer); };
  }, []);

  if (!activities.length) return null;
  return <aside className="workspace-active-tasks" aria-label="活动任务" aria-live="polite">
    <strong>活动任务</strong>
    {activities.map((task) => <div className="workspace-active-task" key={`${task.kind}:${task.id}`}>
      <span><b>{task.kind === "generation" ? task.targeted ? "定向数据生成" : "SFT 数据生成" : task.kind === "evaluation" ? "阶段评测" : "阶段训练"}</b><small>{task.detail}</small></span>
      {task.kind === "generation"
        ? <button onClick={() => onOpenGeneration(Boolean(task.targeted), task.context)}>查看任务</button>
        : <button onClick={() => onOpenRun(task.id)}>查看 Run</button>}
    </div>)}
  </aside>;
}
