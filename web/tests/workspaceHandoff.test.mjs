import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const dataset = await readFile(new URL("../src/DatasetWorkbench.tsx", import.meta.url), "utf8");
const launcher = await readFile(new URL("../src/Launcher.tsx", import.meta.url), "utf8");
const runs = await readFile(new URL("../src/Runs.tsx", import.meta.url), "utf8");
const targeted = await readFile(new URL("../src/TargetedExperiments.tsx", import.meta.url), "utf8");
const main = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");
const navigation = await readFile(new URL("../src/workspaceNavigation.mjs", import.meta.url), "utf8");

test("registered dataset handoff carries its real ID and stage without launching training", () => {
  assert.match(dataset, /onUseDataset\(dataset\.dataset_id, dataset\.stage\)/);
  assert.match(navigation, /const stage = datasetStage === "preference" \? "dpo" : datasetStage/);
  assert.match(navigation, /new URLSearchParams\(\{ page: "launcher", stage, dataset_id: datasetId,\s*return_to: returningToLauncher \? "launcher" : "workbench" \}\)/);
  assert.match(launcher, /stageDatasets\.some\(\(item\) => item\.dataset_id === requested && item\.eligible !== false\)\) setSelectedDatasetIds\(\[requested\]\)/);
  assert.match(launcher, /else if \(stageDatasets\.some\(\(item\) => item\.dataset_id === requested\)\) setError/);
  assert.match(launcher, /datasetHandoffApplied\.current = true/);
  assert.doesNotMatch(main.slice(main.indexOf("const openTrainingWithDataset"), main.indexOf("const returnToWorkspaceOrigin")), /\/api\/training\/start|\/api\/launcher\/start/);
});

test("return navigation restores the original dataset details and training Run link", () => {
  assert.match(main, /returnToWorkspaceOrigin[\s\S]*current\.get\("return_to"\) === "workbench"[\s\S]*dataset_id/);
  assert.match(main, /const openRunDetail[\s\S]*\["return_to", "dataset_id", "stage"\]/);
  assert.match(dataset, /useState<string \| null>\(\(\) => new URLSearchParams\(window\.location\.search\)\.get\("dataset_id"\)\)/);
});

test("evaluation analysis carries recorded evidence IDs and navigation never starts generation", () => {
  assert.match(runs, /item\.kind === "model"/);
  assert.match(runs, /item\.kind === "dataset" && \/eval\|holdout\/i\.test/);
  assert.match(runs, /new URLSearchParams\(\{ page: "targeted", eval_id: detail\.run_id, return_to: "runs", run_id: detail\.run_id \}\)/);
  assert.match(runs, /if \(model\) params\.set\("model_id", model\)/);
  assert.match(runs, /分析 \/ 补充数据/);
  assert.match(targeted, /返回原薄弱场景诊断/);
  assert.match(targeted, /setNotice\(`数据集已登记：/);
  assert.match(targeted, /onClick=\{returnToDiagnostics\}/);
});
