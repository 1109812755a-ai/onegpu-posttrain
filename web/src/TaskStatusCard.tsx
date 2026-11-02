import type { ReactNode } from "react";
import { taskDisplayState, type TaskAction, type TaskDisplayInput } from "./taskPresentation";

export function TaskStatusCard({ input, onAction, detail, actionLabel, actionDisabled }: {
  input: TaskDisplayInput;
  onAction?: (action: TaskAction, runId?: string) => void;
  detail?: ReactNode;
  actionLabel?: string;
  actionDisabled?: boolean;
}) {
  const state = taskDisplayState(input);
  return <section className={`task-status-card task-status-${state.tone}`} aria-live="polite" data-task-phase={state.phase}>
    <div className="task-status-copy">
      <span className={`task-status-badge task-status-badge-${state.tone}`}>{state.title}</span>
      <p>{state.description}</p>
      {detail}
    </div>
    {state.action !== "none" && <button type="button" className="button primary task-status-action"
      disabled={state.actionDisabled || actionDisabled || !onAction} onClick={() => onAction?.(state.action, state.runId)}>{actionLabel || state.actionLabel}</button>}
  </section>;
}
