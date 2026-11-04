import type { ReactNode } from "react";
import { isNavigationCurrent, moduleDefaultPage, workspaceModuleFromPage } from "./workspaceNavigation.mjs";
import type { WorkspacePage } from "./workspaceNavigation.mjs";

export const PRIMARY_MODULES = [
  { id: "data", name: "数据", icon: "data", module: "data" },
  { id: "training", name: "训练与评测", icon: "runs", module: "training" },
  { id: "runs", name: "实验 Runs", icon: "runs", module: "runs" },
  { id: "analysis", name: "分析与改进", icon: "chart", module: "analysis" },
  { id: "assistant", name: "实验助手", icon: "chat", module: "assistant" },
] as const;

export const MODULE_VIEWS = {
  data: [
    { page: "workbench", name: "数据集" },
    { page: "data", name: "处理与质量" },
    { page: "policies", name: "规则对照" },
  ],
  training: [
    { page: "launcher", name: "阶段工作台" },
    { page: "model-playground", name: "模型体验" },
    { page: "weight-space", name: "权重空间实验" },
  ],
  runs: [{ page: "runs", name: "运行记录" }],
  analysis: [
    { page: "comparison", name: "阶段比较" },
    { page: "policy-effects", name: "策略效果" },
    { page: "targeted", name: "薄弱场景" },
  ],
  assistant: [],
} as const;

type WorkspaceNavigationProps = {
  page: WorkspacePage;
  onNavigate: (page: WorkspacePage) => void;
  renderIcon: (name: string) => ReactNode;
};

export function WorkspaceNavigation({ page, onNavigate, renderIcon }: WorkspaceNavigationProps) {
  const currentModule = workspaceModuleFromPage(page);

  return (
    <nav className="workspace-navigation" aria-label="工作区主导航">
      <div className="workspace-primary-nav">
        {PRIMARY_MODULES.map((item) => {
          const active = isNavigationCurrent(item, page);
          const views = MODULE_VIEWS[item.module];
          const hasSubpages = views.length > 1;
          const subnavId = `workspace-subnav-${item.module}`;

          return (
            <div className="workspace-module-group" key={item.id}>
              <button
                className={`workspace-module-button${active ? " nav-active" : ""}`}
                aria-label={item.name}
                aria-current={active ? "location" : undefined}
                aria-expanded={hasSubpages ? active : undefined}
                aria-controls={hasSubpages ? subnavId : undefined}
                onClick={() => onNavigate(moduleDefaultPage(item.module))}
              >
                {renderIcon(item.icon)}
                <span>{item.name}</span>
              </button>
              {hasSubpages && (
                <div
                  id={subnavId}
                  className="workspace-secondary-nav"
                  aria-label={`${item.name}视图`}
                  hidden={!active || currentModule !== item.module}
                >
                  {views.map((view) => (
                    <button
                      key={view.page}
                      className={page === view.page ? "nav-active" : ""}
                      aria-current={page === view.page ? "page" : undefined}
                      onClick={() => onNavigate(view.page)}
                    >
                      <span>{view.name}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </nav>
  );
}
