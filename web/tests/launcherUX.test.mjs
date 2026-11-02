import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const component = await readFile(new URL("../src/Launcher.tsx", import.meta.url), "utf8");
const css = await readFile(new URL("../src/style.css", import.meta.url), "utf8");
const navigation = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");
const workspaceNavigation = await readFile(new URL("../src/WorkspaceNavigation.tsx", import.meta.url), "utf8");
const backendLauncher = await readFile(new URL("../../src/onegpu_posttrain/ui/launcher.py", import.meta.url), "utf8");

test("disabled launcher is informational and cannot expose a start action", () => {
  assert.match(component, /!cap\?\.enabled[\s\S]*训练启动未启用/);
  assert.match(component, /if \(!cap\?\.token \|\| !preflight\?\.ok/);
  assert.match(component, /!cap\?\.enabled \|\| !cap\.available \|\| !cap\.token/);
});

test("launch confirmation discloses the resolved identity and UI shutdown behavior", () => {
  assert.match(component, /window\.confirm/);
  assert.match(component, /配置摘要/);
  assert.match(component, /关闭本地 UI 服务会中断训练/);
  assert.match(component, /run\.evaluation_status === "running" \? "评测中/);
  assert.match(component, /启动阶段评测/);
});

test("launcher provides keyboard focus and a narrow-screen layout", () => {
  assert.match(component, /<button/);
  assert.match(component, /<summary>查看精确解析配置/);
  assert.match(css, /\.launcher-page :focus-visible/);
  assert.match(css, /@media \(max-width: 760px\)[\s\S]*\.launcher-facts \{ grid-template-columns: 1fr/);
});

test("active run polls logs and pauses follow while the reader scrolls up", () => {
  assert.match(component, /setInterval\([\s\S]*1500/);
  assert.match(component, /onScroll=\{handleLogScroll\}/);
  assert.match(component, /有新日志 · 回到最新/);
  assert.match(component, /恢复自动跟随/);
  assert.match(component, /refreshRun\(controller\.signal\)/);
});

test("refresh restores safe stage drafts and the exact active run tab without restoring confirmations", () => {
  assert.match(component, /STAGE_DRAFT_PREFIX = "onegpu\.stage-workbench\.draft\.v1:"/);
  assert.match(component, /window\.sessionStorage\.setItem\(`\$\{STAGE_DRAFT_PREFIX\}/);
  assert.match(component, /params\.set\("launcher_run", runId\)/);
  assert.match(component, /params\.set\("launcher_tab", tab\)/);
  assert.match(component, /requestedRunTab/);
  assert.match(component, /setRunTabState\(defaultTab\)/);
  assert.doesNotMatch(component, /sessionStorage\.setItem\([^\n]*confirmation_token/);
  assert.doesNotMatch(component, /sessionStorage\.setItem\([^\n]*licenseAcknowledged/);
});

test("training and evaluation pages show the whole workflow and retain a read-only run snapshot", () => {
  assert.match(component, /function WorkflowRoadmap/);
  assert.match(component, /配置阶段与数据/);
  assert.match(component, /预检并启动训练/);
  assert.match(component, /查看训练过程/);
  assert.match(component, /阶段评测/);
  assert.match(component, /查看结果与血缘/);
  assert.match(component, /function RunConfiguration/);
  assert.match(component, /本次运行配置/);
  assert.match(component, /只读快照/);
  assert.match(component, /run\.training_parameters\?\.effective/);
});

test("running stage evaluation replaces setup controls with one clear running state and live log", () => {
  assert.match(component, /evaluationActive \? <>[\s\S]*阶段评测正在运行[\s\S]*评测日志[\s\S]*: <>[\s\S]*检查评测配置/);
  assert.match(component, /评测准备中/);
  assert.match(component, /刷新页面不会中断或重新启动/);
});

test("evaluation is gated on completed training and registered merged SFT model", () => {
  assert.match(component, /run\?\.model_artifacts\?\.some\(\(model\) => model\.kind === "merged" && model\.stage === "sft"\)/);
  assert.match(component, /run\.status === "completed" && mergedReady && !evaluationActive/);
  assert.match(component, /evaluation_status === "partial"/);
  assert.match(component, /metric\.denominator \?\? "报告未记录"/);
  assert.doesNotMatch(component, /metrics\.map\([^\n]*\?\? 0/);
});

test("run tabs and metric cards collapse cleanly for mobile widths", () => {
  assert.match(component, /role="tablist"/);
  assert.match(css, /\.launcher-tabs \{[^}]*overflow-x: auto/);
  assert.match(css, /@media \(max-width: 760px\)[\s\S]*\.launcher-metric-grid \{ grid-template-columns: 1fr/);
});

test("launcher navigation is available without changing the existing read-only page route", () => {
  assert.match(workspaceNavigation, /id: "training", name: "训练与评测"[\s\S]*module: "training"/);
  assert.match(workspaceNavigation, /id: "runs", name: "实验 Runs", icon: "runs", module: "runs"/);
  assert.match(workspaceNavigation, /training: \[[\s\S]*page: "launcher", name: "阶段工作台"[\s\S]*page: "model-playground", name: "模型体验"/);
  assert.match(navigation, /<WorkspaceNavigation[\s\S]*page=\{page\}/);
});

test("Mini SFT can launch one registered dataset or a lineage-preserving registered mixture", () => {
  assert.match(component, /type="radio" name="stage-training-dataset"/);
  assert.match(component, /需要组合多个来源？前往“数据”创建组合数据集/);
  assert.match(component, /Mini SFT 启动器消费一个已登记数据集 ID；请先将多个所选数据集组合成新的登记数据集/);
  assert.match(component, /dataset_id: eligibleSelectedDatasetIds\[0\]/);
  assert.match(component, /parent_model_id\?: string/);
  assert.match(backendLauncher, /"parent_model_id": MODEL_ID/);
  assert.match(component, /quality_status/);
  assert.match(component, /acknowledged_dataset_id/);
  assert.match(component, /我确认仅依据来源\/许可的当前声明使用数据/);
  assert.match(component, /preflight\.dataset\?\.license_status !== "verified" && !acknowledgeUnknown/);
  assert.doesNotMatch(component, /launcher-recipe-title|launcher-preflight-title/);
});
