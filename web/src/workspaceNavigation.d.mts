type WorkspaceNavItem = {
  module?: "data" | "training" | "runs" | "analysis";
  active?: boolean;
  comparison?: boolean;
  policy?: boolean;
  targeted?: boolean;
  launcher?: boolean;
  runs?: boolean;
  workbench?: boolean;
};

export type WorkspacePage = "data" | "comparison" | "policy-effects" | "policies" | "targeted" | "launcher" | "model-playground" | "runs" | "workbench";

export function grpoPreparationSearch(profileId?: string, parentModelId?: string): URLSearchParams;

export function registeredDatasetTrainingSearch(datasetId: string, datasetStage: string, currentSearch?: string): URLSearchParams;

export function launcherReturnSearch(currentSearch?: string): URLSearchParams;

export function workspaceModuleFromPage(page: WorkspacePage | string): "data" | "training" | "runs" | "analysis";

export function moduleDefaultPage(module: "data" | "training" | "runs" | "analysis"): WorkspacePage;

export function navigationTarget(
  item: WorkspaceNavItem,
): "data" | "comparison" | "policy-effects" | "policies" | "targeted" | "launcher" | "runs" | "workbench" | null;

export function isNavigationCurrent(
  item: WorkspaceNavItem,
  page: WorkspacePage,
): boolean;

export function workspacePageFromSearch(
  search: string,
): WorkspacePage;

export function workspaceRouteStateFromSearch(search: string): {
  page: WorkspacePage;
  scope: string;
  build: string;
  stage: string;
  tab: string;
  query: string;
  filter: string;
  matrix: string;
};

export function workspaceSearchForPage(
  page: WorkspacePage,
  currentSearch?: string,
  dataState?: { scope?: string; build?: string; stage?: string; tab?: string; query?: string; filter?: string; matrix?: string },
): URLSearchParams;
