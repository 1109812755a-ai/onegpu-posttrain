import assert from "node:assert/strict";
import test from "node:test";
import { readFile } from "node:fs/promises";

const component = await readFile(new URL("../src/Launcher.tsx", import.meta.url), "utf8");
const styles = await readFile(new URL("../src/style.css", import.meta.url), "utf8");
const main = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");
const targeted = await readFile(new URL("../src/TargetedExperiments.tsx", import.meta.url), "utf8");

test("targeted stage capability cards hand off to the corresponding Launcher workflow", () => {
  assert.match(targeted, /item\.training_ui_available && <a className="button secondary targeted-stage-launch-link"/);
  assert.match(targeted, /item\.stage === "preference" \? "dpo" : item\.stage/);
  assert.match(targeted, /前往\{item\.stage === "preference" \? "DPO \/ SimPO" : item\.stage\.toUpperCase\(\)\}训练与评测/);
});

test("stage switching clears the stage-specific preflight and prior selections", () => {
  assert.match(component, /function chooseStage\(stage: string\)[\s\S]*setSelectedDatasetIds\(\[\]\)[\s\S]*setSelectedParentModelId\(""\)[\s\S]*setStagePreflight\(null\)/);
  assert.match(component, /params\.set\("stage", stage\)/);
  assert.match(component, /切换阶段会清除当前预检摘要/);
});

test("stage workbench selects one registered dataset and links to data composition", () => {
  assert.match(component, /\/api\/training\/workbench-options/);
  assert.match(component, /\/api\/training\/preflight/);
  assert.match(component, /dataset_ids: eligibleSelectedDatasetIds, parent_model_id: selectedParentModelId/);
  assert.match(component, /\/api\/launcher\/preflight/);
  assert.match(component, /\/api\/launcher\/start/);
  assert.match(component, /declaredDatasets\?\.\[0\]\?\.content_sha256 !== launchCheck\.dataset\?\.canonical_sha256/);
  assert.match(component, /type="radio" name="stage-training-dataset"/);
  assert.match(component, /需要组合多个来源？前往“数据”创建组合数据集/);
  assert.match(component, /item\.eligible === false/);
  assert.match(component, /不可用于训练：/);
  assert.match(component, /实时阶段评测日志/);
  assert.match(component, /run\.evaluation_log_tail/);
  assert.match(component, /本机 GPU 任务运行中…/);
  assert.doesNotMatch(component, /\/api\/data\/mixtures\/(preflight|build)/);
  assert.doesNotMatch(component, /固定配方卡片/);
  assert.match(component, /查看逐项检查和摘要/);
});

test("launcher primary actions have a scoped, high-contrast accessible visual style", () => {
  assert.match(styles, /\.launcher-page \.button\.primary\s*\{[^}]*min-height:\s*46px[^}]*background:\s*var\(--action-bg\)[^}]*font-size:\s*14px/s);
  assert.match(styles, /\.launcher-page \.button\.primary:focus-visible\s*\{[^}]*outline:/s);
  assert.match(styles, /\.launcher-page \.button\.primary:disabled\s*\{[^}]*cursor:\s*not-allowed/s);
  assert.doesNotMatch(styles, /(^|\n)\.button\.primary\s*\{/);
});

test("training and evaluation actions stay in a responsive sticky-in-flow action area", () => {
  assert.match(component, /stage-workbench-actions launcher-action-bar/);
  assert.ok((component.match(/className="launcher-action-bar"/g) || []).length >= 4);
  assert.match(styles, /\.launcher-action-bar\s*\{[^}]*position:\s*sticky[^}]*bottom:/s);
  assert.match(styles, /@media \(max-width: 760px\)\s*\{\s*\.launcher-action-bar\s*\{[^}]*flex-direction:\s*column/s);
  assert.match(styles, /safe-area-inset-bottom/);
  assert.match(styles, /\.launcher-action-bar :focus-visible\s*\{[^}]*scroll-margin-block:/s);
});

test("training preflight and launch are distinct steps with a single primary action", () => {
  assert.match(component, /第 1 步 · 检查训练配置/);
  assert.match(component, /检查配置并预检/);
  assert.match(component, /!\(stageConfirmation \|\| \(selectedProfile\.stage === "sft" && preflight\?\.ok\)\)/);
  assert.match(component, /第 2 步 · 启动 \{selectedProfile\.stage\.toUpperCase\(\)\} 训练/);
  assert.match(component, /`启动 \$\{selectedProfile\.stage\.toUpperCase\(\)\} 训练`/);
  assert.match(component, /启动 SFT 训练/);
});

test("stage evaluation exposes one primary action for each preflight step", () => {
  assert.match(component, /!stageEvaluationPreflight\?\.ok && <div className="launcher-action-bar"/);
  assert.match(component, /第 1 步 · 检查评测配置/);
  assert.match(component, /重新检查评测配置/);
  assert.match(component, /第 2 步 · 启动阶段评测/);
  assert.match(component, /模型：\{stageEvaluationPreflight\.model\?\.model_id\} · 输出：\{stageEvaluationPreflight\.output\}/);
  assert.match(component, /!evaluationPreflight\?\.ok && <div className="launcher-action-bar"/);
  assert.match(component, /disabled=\{busy \|\| taskActive\} onClick=\{startRegisteredStageEvaluation\}/);
});

test("disabled launch actions explain the blocker and link to the missing input or active Run", () => {
  assert.match(component, /const stageSelectionBlocker = !selectedProfile/);
  assert.match(component, /请先选择一个已登记的 \$\{selectedProfile\.dataset_stage\} 训练数据集/);
  assert.match(component, /请先选择上方已登记且符合阶段要求的父模型/);
  assert.match(component, /前往数据模块登记数据/);
  assert.match(component, /href="#stage-dataset-options"/);
  assert.match(component, /href="#stage-parent-model-select"/);
  assert.match(component, /请先勾选上方来源\/许可确认框，再启动训练/);
  assert.match(component, /const evaluationSelectionBlocker = !selectedEvaluationProfileId/);
  assert.match(component, /href="#stage-evaluation-profile"/);
  assert.match(component, /href="#stage-evaluation-dataset"/);
  assert.match(component, /查看当前 Run · \{run\.run_id\}/);
  assert.match(component, /本机 GPU 正在执行任务，训练入口暂不可用/);
  assert.match(component, /本机 GPU 正在执行任务，评测入口暂不可用/);
});

test("stage preflight is not mislabeled as blocked and displays one confirmation path", () => {
  assert.match(component, /stagePreflight\.warnings\.length > 0[\s\S]*预检完成 · 有提醒/);
  assert.match(component, /阶段检查与启动预检已核对相同的父模型 ID\/hash 和数据集 ID\/canonical hash/);
  assert.match(component, /启动 SFT 训练/);
  assert.match(component, /固定父模型（由配方指定）/);
});

test("stage preflight renders independent pass, blocker and warning cards with repair hints and no-start state", () => {
  assert.match(component, /type ReadinessCheck = \{ check_id: string; status: "pass" \| "warning" \| "fail"/);
  assert.match(component, /readinessCheckLabels\[check\.check_id\]/);
  assert.match(component, /readinessStatusLabels\[check\.status\]/);
  assert.match(component, /处理建议：\{check\.repair_hint\}/);
  assert.match(component, /执行状态：尚未启动/);
  assert.match(component, /不会安装依赖、下载权重或创建训练 Run/);
  assert.match(styles, /\.readiness-check\.readiness-fail/);
  assert.match(styles, /\.readiness-check\.readiness-warning/);
  assert.match(styles, /\.readiness-check\.readiness-pass/);
  assert.match(component, /result\.status === "ready_for_confirmation" && !\(result\.blockers \|\| \[\]\)\.length/);
  assert.match(component, /stageConfirmation\.datasets as Record<string, any>\[\] \| undefined\)\?\.some\(\(item\) => item\.license_status !== "verified"\)/);
  assert.match(component, /disabled=\{busy \|\| taskActive \|\| \(\(stageConfirmation\.datasets/);
});

test("stage navigation and selectors remain horizontally safe on narrow screens", () => {
  assert.match(component, /aria-label="训练阶段" role="tablist"/);
  assert.match(component, /aria-label="阶段父模型"/);
  assert.match(styles, /\.stage-selector \{[^}]*overflow-x: auto/);
  assert.match(styles, /@media \(max-width: 760px\)[\s\S]*\.stage-profile-summary,.stage-input-grid,.stage-parameter-grid,.stage-dataset-options \{ grid-template-columns: 1fr/);
});

test("profile parameters are bounded, custom changes invalidate preflight, and both checks bind the same config", () => {
  assert.match(component, /training_parameters/);
  assert.match(component, /setParameterOverrides/);
  assert.match(component, /setStagePreflight\(null\); setPreflight\(null\)/);
  assert.match(component, /parameter_overrides: parameterOverrides/);
  assert.match(component, /stageConfigHash !== launcherConfigHash/);
  assert.match(component, /允许范围/);
  assert.match(component, /type="number" min=\{parameter\.minimum\} max=\{parameter\.maximum\}/);
  assert.match(component, /默认值：\{parameter\.default\}/);
  assert.match(component, /序列越长越占显存[\s\S]*可能 OOM/);
  assert.match(component, /恢复方案默认值/);
  assert.match(component, /可在批准范围内调整参数/);
});

test("Run handoff restores exact stage, profile and parent model without selecting data or starting", () => {
  assert.match(component, /params\.get\("profile_id"\)/);
  assert.match(component, /params\.get\("parent_model_id"\)/);
  assert.match(component, /params\.get\("source_run_id"\)/);
  assert.match(component, /setSelectedParentModelId\(requestedModelId\)/);
  assert.match(component, /sourceRunId.*预选父模型[\s\S]*训练数据仍需单独选择/);
  assert.match(component, /父模型来源 Run/);
  assert.match(component, /modelHandoffApplied/);
  assert.match(component, /item\.stage === requested && item\.profile_id === requestedProfile/);
  assert.match(component, /parent_model_id: selectedParentModelId/);
  assert.match(main, /current\.get\("return_to"\) === "runs"[\s\S]*current\.get\("source_run_id"\)[\s\S]*setPage\("runs"\)/);
  assert.match(component, /handoffSourceRunId \? "返回来源 Run" : initialParams\.get\("return_to"\) === "launcher" \? "返回 GRPO 数据准备" : "返回数据页"/);
});
