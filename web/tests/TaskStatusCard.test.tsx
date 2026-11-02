import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { TaskStatusCard } from "../src/TaskStatusCard";
import { taskDisplayState } from "../src/taskPresentation";

afterEach(cleanup);

describe("shared task display state", () => {
  it("keeps repeated clicks disabled only while the start request is pending", () => {
    const action = vi.fn();
    render(<TaskStatusCard input={{ request: "submit_pending" }} onAction={action} />);
    expect(screen.getByText("正在提交启动请求")).toBeTruthy();
    expect(screen.queryByRole("button", { name: "确认并启动" })).toBeNull();
    expect(action).not.toHaveBeenCalled();
  });

  it("shows a created Run after a successful start instead of a permanent spinner", () => {
    const action = vi.fn();
    render(<TaskStatusCard input={{ request: "idle", run: { run_id: "run-1", status: "starting" } }} onAction={action} />);
    fireEvent.click(screen.getByRole("button", { name: "查看 Run" }));
    expect(action).toHaveBeenCalledWith("open_run", "run-1");
    expect(screen.getByText("Run 已创建，等待执行")).toBeTruthy();
  });

  it("does not silently retry after an uncertain network timeout", () => {
    const action = vi.fn();
    render(<TaskStatusCard input={{ request: "submission_unknown" }} onAction={action} />);
    fireEvent.click(screen.getByRole("button", { name: "检查任务状态" }));
    expect(action).toHaveBeenCalledWith("inspect", undefined);
    expect(screen.getByText(/先查询任务/)).toBeTruthy();
  });

  it("requires a fresh preflight when a confirmation token expires", () => {
    const action = vi.fn();
    render(<TaskStatusCard input={{ request: "token_expired", preflight: "ready" }} onAction={action} />);
    fireEvent.click(screen.getByRole("button", { name: "重新预检" }));
    expect(action).toHaveBeenCalledWith("preflight", undefined);
  });

  it("uses authoritative Run status for running and all terminal outcomes", () => {
    expect(taskDisplayState({ request: "submit_pending", run: { run_id: "r", status: "running" } }).phase).toBe("run_running");
    for (const status of ["completed", "failed", "partial", "interrupted", "oom"]) {
      const state = taskDisplayState({ request: "idle", run: { run_id: "r", status } });
      expect(state.phase).toBe("run_terminal");
      expect(state.action).toBe("open_run");
    }
    expect(taskDisplayState({ request: "idle", run: { run_id: "r", status: "mystery" } }).phase).toBe("run_unknown");
  });

  it("renders semantic error state and an explicitly disabled action", () => {
    const { container, rerender } = render(<TaskStatusCard input={{ request: "idle", preflight: "blocked" }} onAction={vi.fn()} />);
    expect(container.querySelector(".task-status-danger")?.getAttribute("data-task-phase")).toBe("preflight_blocked");
    rerender(<TaskStatusCard input={{ request: "idle", preflight: "ready" }} onAction={vi.fn()} actionDisabled />);
    expect(screen.getByRole("button", { name: "确认并启动" }).hasAttribute("disabled")).toBe(true);
    expect(container.querySelector(".task-status-success")).toBeTruthy();
  });
});
