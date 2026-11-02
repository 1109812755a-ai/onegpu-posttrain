import assert from "node:assert/strict";
import test from "node:test";

import {
  isNavigationCurrent,
  grpoPreparationSearch,
  launcherReturnSearch,
  moduleDefaultPage,
  navigationTarget,
  registeredDatasetTrainingSearch,
  workspaceModuleFromPage,
  workspacePageFromSearch,
  workspaceRouteStateFromSearch,
  workspaceSearchForPage,
} from "../src/workspaceNavigation.mjs";

const modules = [
  [{ module: "data" }, "workbench", ["data", "workbench", "policies"]],
  [{ module: "training" }, "launcher", ["launcher", "model-playground"]],
  [{ module: "runs" }, "runs", ["runs"]],
  [{ module: "analysis" }, "comparison", ["comparison", "policy-effects", "targeted"]],
];

test("four primary modules navigate to a sensible landing view", () => {
  for (const [item, landing, pages] of modules) {
    assert.equal(navigationTarget(item), landing);
    assert.equal(moduleDefaultPage(item.module), landing);
    for (const page of ["data", "workbench", "policies", "launcher", "model-playground", "runs", "comparison", "policy-effects", "targeted"]) {
      assert.equal(isNavigationCurrent(item, page), pages.includes(page));
    }
  }
});

test("legacy deep links keep their original page while plain root opens the dataset catalog", () => {
  assert.equal(workspacePageFromSearch(""), "workbench");
  assert.equal(workspacePageFromSearch("?page=workbench&stage=sft"), "workbench");
  assert.equal(workspacePageFromSearch("?scope=real&build=b1&stage=sft&tab=flow"), "data");
  assert.equal(workspacePageFromSearch("?page=data&build=b1&tab=quality"), "data");
  assert.equal(workspacePageFromSearch("?page=policies"), "policies");
  assert.equal(workspacePageFromSearch("?page=targeted&eval_id=e1"), "targeted");
  assert.equal(workspacePageFromSearch("?page=launcher&dataset_id=d1"), "launcher");
  assert.equal(workspacePageFromSearch("?page=runs&run=r1"), "runs");
  assert.equal(workspacePageFromSearch("?page=comparison&matrix=m1"), "comparison");
  assert.equal(workspacePageFromSearch("?page=policy-effects&report_id=report-2"), "policy-effects");
  assert.equal(workspacePageFromSearch("?page=unknown"), "workbench");
});

test("module identity groups detail pages under the four task areas", () => {
  assert.equal(workspaceModuleFromPage("workbench"), "data");
  assert.equal(workspaceModuleFromPage("policies"), "data");
  assert.equal(workspaceModuleFromPage("launcher"), "training");
  assert.equal(workspaceModuleFromPage("model-playground"), "training");
  assert.equal(workspaceModuleFromPage("runs"), "runs");
  assert.equal(workspaceModuleFromPage("targeted"), "analysis");
  assert.equal(workspaceModuleFromPage("comparison"), "analysis");
  assert.equal(workspaceModuleFromPage("policy-effects"), "analysis");
});

test("route changes only carry parameters owned by the destination module", () => {
  const source = "?page=data&scope=real&build=b1&stage=sft&tab=quality&q=math&filter=sft&matrix=m1&dataset_id=d1";
  const launcher = workspaceSearchForPage("launcher", source);
  assert.deepEqual([...launcher.entries()], [["page", "launcher"], ["dataset_id", "d1"], ["stage", "sft"]]);

  const modelHandoff = workspaceSearchForPage("launcher", "?page=runs&stage=dpo&profile_id=dpo-smoke-v1&parent_model_id=model-merged&source_run_id=run-sft&return_to=runs");
  assert.deepEqual([...modelHandoff.entries()], [
    ["page", "launcher"], ["stage", "dpo"], ["profile_id", "dpo-smoke-v1"],
    ["parent_model_id", "model-merged"], ["source_run_id", "run-sft"], ["return_to", "runs"],
  ]);

  const playgroundHandoff = workspaceSearchForPage("model-playground", "?page=runs&model_id=model-sft&run_id=run-sft&return_to=runs&stage=sft");
  assert.deepEqual([...playgroundHandoff.entries()], [
    ["page", "model-playground"], ["model_id", "model-sft"], ["run_id", "run-sft"], ["return_to", "runs"],
  ]);

  const runs = workspaceSearchForPage("runs", `${source}&run=r1&run_status=completed`);
  assert.deepEqual([...runs.entries()], [["page", "runs"], ["run", "r1"], ["run_status", "completed"]]);

  const comparison = workspaceSearchForPage("comparison", source, { matrix: "m2" });
  assert.deepEqual([...comparison.entries()], [["page", "comparison"], ["matrix", "m2"]]);

  const report = workspaceSearchForPage("policy-effects", "?page=policies&report_id=rep-2&experiment_id=exp-2&build=b1");
  assert.deepEqual([...report.entries()], [["page", "policy-effects"], ["report_id", "rep-2"], ["experiment_id", "exp-2"]]);
  const rules = workspaceSearchForPage("policies", "?page=policy-effects&report_id=rep-2&experiment_id=exp-2");
  assert.deepEqual([...rules.entries()], [["page", "policies"], ["report_id", "rep-2"], ["experiment_id", "exp-2"]]);

  const data = workspaceSearchForPage("data", "?page=launcher", { scope: "real", build: "b2", stage: "sft", tab: "effects", query: "deep", filter: "all" });
  assert.deepEqual([...data.entries()], [["page", "data"], ["scope", "real"], ["build", "b2"], ["stage", "sft"], ["tab", "effects"], ["q", "deep"]]);
});

test("dataset catalog preserves old data context only when navigating within data", () => {
  const fromLegacyData = workspaceSearchForPage("workbench", "?scope=real&build=b1&stage=sft&tab=quality");
  assert.deepEqual([...fromLegacyData.entries()], [["page", "workbench"], ["scope", "real"], ["build", "b1"], ["stage", "sft"], ["tab", "quality"]]);

  const fromAnalysis = workspaceSearchForPage("workbench", "?page=targeted&scope=demo&build=old&eval_id=e1");
  assert.deepEqual([...fromAnalysis.entries()], [["page", "workbench"]]);
});

test("targeted generation task links retain scenario and evaluation identity while the data view normalizes its route", () => {
  const source = "?page=workbench&targeted_generation=1&scenario_id=weak-1&bundle_id=bundle-7&eval_id=eval-4&model_id=model-3&dataset_id=diag-2&return_to=targeted";
  assert.deepEqual([...workspaceSearchForPage("workbench", source).entries()], [
    ["page", "workbench"], ["targeted_generation", "1"], ["scenario_id", "weak-1"], ["bundle_id", "bundle-7"],
    ["eval_id", "eval-4"], ["model_id", "model-3"], ["dataset_id", "diag-2"], ["return_to", "targeted"],
  ]);
});

test("targeted generation handoff carries only explicit object context", () => {
  const result = workspaceSearchForPage("targeted", "?page=comparison&matrix=m1&scope=demo&eval_id=e1&model_id=model-1&run_id=run-1&scenario_id=weak-1&stage=sft");
  assert.deepEqual([...result.entries()], [
    ["page", "targeted"],
    ["model_id", "model-1"],
    ["run_id", "run-1"],
    ["eval_id", "e1"],
    ["scenario_id", "weak-1"],
    ["stage", "sft"],
  ]);
});

test("GRPO recovery round-trip preserves profile and parent and returns only the exact registered dataset", () => {
  const preparation = grpoPreparationSearch("grpo-lowmem-smoke-v1", "model-sft-merged");
  assert.deepEqual([...preparation.entries()], [
    ["page", "workbench"], ["stage", "grpo"], ["prepare_grpo", "1"], ["return_to", "launcher"],
    ["profile_id", "grpo-lowmem-smoke-v1"], ["parent_model_id", "model-sft-merged"],
  ]);

  const handoff = registeredDatasetTrainingSearch("ds-grpo-registered-exact", "grpo", `?${preparation}`);
  assert.deepEqual([...handoff.entries()], [
    ["page", "launcher"], ["stage", "grpo"], ["dataset_id", "ds-grpo-registered-exact"], ["return_to", "launcher"],
    ["profile_id", "grpo-lowmem-smoke-v1"], ["parent_model_id", "model-sft-merged"],
  ]);
  assert.deepEqual([...launcherReturnSearch(`?${preparation}`).entries()], [
    ["page", "launcher"], ["stage", "grpo"], ["profile_id", "grpo-lowmem-smoke-v1"], ["parent_model_id", "model-sft-merged"],
  ]);

  const ordinaryDataNavigation = registeredDatasetTrainingSearch("ds-sft-1", "sft", "?page=workbench");
  assert.equal(ordinaryDataNavigation.get("return_to"), "workbench");
  assert.equal(ordinaryDataNavigation.has("profile_id"), false);
});

test("back and forward route restoration recovers data filters without leaking them into other modules", () => {
  const dataUrl = "?page=data&scope=demo&build=b2&stage=sft&tab=quality&q=trace&filter=sft";
  const launcherUrl = "?page=launcher&dataset_id=d2&stage=sft";
  const runUrl = "?page=runs&run=run-7&run_stage=sft";
  const history = [dataUrl, launcherUrl, runUrl];

  assert.deepEqual(workspaceRouteStateFromSearch(history[0]), {
    page: "data", scope: "demo", build: "b2", stage: "sft", tab: "quality", query: "trace", filter: "sft", matrix: "",
  });
  assert.deepEqual(workspaceRouteStateFromSearch(history[1]), {
    page: "launcher", scope: "real", build: "", stage: "", tab: "flow", query: "", filter: "all", matrix: "",
  });
  assert.deepEqual(workspaceRouteStateFromSearch(history[2]), {
    page: "runs", scope: "real", build: "", stage: "", tab: "flow", query: "", filter: "all", matrix: "",
  });
  assert.deepEqual(workspaceRouteStateFromSearch(history[0]), workspaceRouteStateFromSearch(dataUrl));
});
