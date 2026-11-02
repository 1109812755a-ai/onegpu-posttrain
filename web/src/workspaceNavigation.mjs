const PAGE_MODULE = {
  data: "data",
  workbench: "data",
  policies: "data",
  launcher: "training",
  "model-playground": "training",
  runs: "runs",
  comparison: "analysis",
  "policy-effects": "analysis",
  targeted: "analysis",
};

const HANDOFF_KEYS = ["dataset_id", "model_id", "run_id", "eval_id", "report_id", "experiment_id", "scenario_id", "plan_id", "bundle_id", "return_to", "stage"];

export function grpoPreparationSearch(profileId, parentModelId) {
  const params = new URLSearchParams({ page: "workbench", stage: "grpo", prepare_grpo: "1", return_to: "launcher" });
  if (profileId) params.set("profile_id", profileId);
  if (parentModelId) params.set("parent_model_id", parentModelId);
  return params;
}

export function registeredDatasetTrainingSearch(datasetId, datasetStage, currentSearch = "") {
  const stage = datasetStage === "preference" ? "dpo" : datasetStage;
  const current = new URLSearchParams(currentSearch);
  const returningToLauncher = stage === "grpo" && current.get("return_to") === "launcher";
  const params = new URLSearchParams({ page: "launcher", stage, dataset_id: datasetId,
    return_to: returningToLauncher ? "launcher" : "workbench" });
  if (returningToLauncher) {
    for (const key of ["profile_id", "parent_model_id"]) {
      const value = current.get(key);
      if (value) params.set(key, value);
    }
  }
  return params;
}

export function launcherReturnSearch(currentSearch = "") {
  const current = new URLSearchParams(currentSearch);
  const params = new URLSearchParams({ page: "launcher", stage: current.get("stage") || "grpo" });
  for (const key of ["profile_id", "parent_model_id"]) {
    const value = current.get(key);
    if (value) params.set(key, value);
  }
  return params;
}

export function workspaceModuleFromPage(page) {
  return PAGE_MODULE[page] || "data";
}

export function moduleDefaultPage(module) {
  return {
    data: "workbench",
    training: "launcher",
    runs: "runs",
    analysis: "comparison",
  }[module] || "workbench";
}

export function navigationTarget(item) {
  if (item.module) return moduleDefaultPage(item.module);
  if (item.active) return "workbench";
  if (item.comparison) return "comparison";
  if (item.policy) return "policies";
  if (item.targeted) return "targeted";
  if (item.launcher) return "launcher";
  if (item.runs) return "runs";
  if (item.workbench) return "workbench";
  return null;
}

export function isNavigationCurrent(item, page) {
  if (item.module) return item.module === workspaceModuleFromPage(page);
  return navigationTarget(item) === page;
}

export function workspacePageFromSearch(search) {
  const params = new URLSearchParams(search);
  const page = params.get("page");
  if (Object.hasOwn(PAGE_MODULE, page)) return page;

  // Old links without page= used the data report view when any report filter
  // was present. A plain root URL now opens the registered dataset catalog.
  const hasLegacyDataContext = ["scope", "build", "stage", "tab", "q", "filter"].some((key) => params.has(key));
  return hasLegacyDataContext ? "data" : "workbench";
}

export function workspaceRouteStateFromSearch(search) {
  const params = new URLSearchParams(search);
  const page = workspacePageFromSearch(search);
  const inData = workspaceModuleFromPage(page) === "data";
  return {
    page,
    scope: inData && params.get("scope") === "demo" ? "demo" : "real",
    build: inData ? params.get("build") || "" : "",
    stage: inData ? params.get("stage") || "" : "",
    tab: inData ? params.get("tab") || "flow" : "flow",
    query: inData ? params.get("q") || "" : "",
    filter: inData ? params.get("filter") || "all" : "all",
    matrix: page === "comparison" ? params.get("matrix") || "" : "",
  };
}

export function workspaceSearchForPage(page, currentSearch = "", dataState = {}) {
  const current = new URLSearchParams(currentSearch);
  const params = new URLSearchParams();
  const setIfPresent = (key, value) => {
    if (value !== undefined && value !== null && String(value) !== "") params.set(key, String(value));
  };

  if (page === "data") {
    setIfPresent("page", "data");
    setIfPresent("scope", dataState.scope);
    setIfPresent("build", dataState.build);
    setIfPresent("stage", dataState.stage);
    setIfPresent("tab", dataState.tab);
    setIfPresent("q", dataState.query);
    if (dataState.filter && dataState.filter !== "all") setIfPresent("filter", dataState.filter);
    return params;
  }

  setIfPresent("page", page);
  if (page === "workbench" && workspaceModuleFromPage(workspacePageFromSearch(currentSearch)) === "data") {
    for (const key of ["scope", "build", "stage", "tab", "q", "filter"]) {
      setIfPresent(key, current.get(key));
    }
  }
  if (page === "workbench" && current.get("targeted_generation") === "1") {
    setIfPresent("targeted_generation", "1");
    for (const key of ["scenario_id", "bundle_id", "eval_id", "model_id", "dataset_id", "return_to"]) {
      setIfPresent(key, current.get(key));
    }
  }
  if (page === "policies" && workspacePageFromSearch(currentSearch) === "policy-effects") {
    for (const key of ["report_id", "experiment_id"]) setIfPresent(key, current.get(key));
  }
  if (page === "launcher") {
    for (const key of ["dataset_id", "stage", "profile_id", "parent_model_id", "source_run_id", "return_to", "launcher_run"]) {
      setIfPresent(key, current.get(key));
    }
  } else if (page === "model-playground") {
    for (const key of ["model_id", "run_id", "return_to"]) setIfPresent(key, current.get(key));
  } else if (page === "comparison") {
    setIfPresent("matrix", dataState.matrix || current.get("matrix"));
  } else if (page === "policy-effects") {
    for (const key of ["report_id", "experiment_id"]) setIfPresent(key, current.get(key));
  } else if (page === "targeted") {
    for (const key of HANDOFF_KEYS) setIfPresent(key, current.get(key));
  } else if (page === "runs") {
    for (const key of ["run", "run_stage", "run_status", "run_q", "run_offset"]) {
      setIfPresent(key, current.get(key));
    }
  }
  return params;
}
