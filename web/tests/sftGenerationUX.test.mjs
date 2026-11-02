import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";

const panelPath = new URL("../src/SFTGenerationPanel.tsx", import.meta.url);
const panel = await readFile(panelPath, "utf8");
const workbench = await readFile(new URL("../src/DatasetWorkbench.tsx", import.meta.url), "utf8");
const targeted = await readFile(new URL("../src/TargetedExperiments.tsx", import.meta.url), "utf8");
const review = await readFile(new URL("../src/GenerationReviewPanel.tsx", import.meta.url), "utf8");
const activity = await readFile(new URL("../src/ActivityTasks.tsx", import.meta.url), "utf8");
const policyEditor = await readFile(new URL("../src/PolicyEditor.tsx", import.meta.url), "utf8");

test("ordinary SFT page discovery and prepare/preflight are explicit provider-free steps", () => {
  assert.match(panel, /\/api\/synthetic\/sft\/readiness/);
  assert.match(panel, /\/api\/synthetic\/sft\/prepare/);
  assert.match(panel, /\/api\/synthetic\/sft\/preflight/);
  assert.match(panel, /provider_calls|不调用模型/);
  assert.match(panel, /credential_env_configured && !item\.is_example/);
  assert.match(panel, /页面不会索取或保存密钥/);
});

test("remote generation requires a distinct confirmation and records a recoverable execution ID", () => {
  assert.match(panel, /window\.confirm\(/);
  assert.match(panel, /confirm_start: true/);
  assert.match(panel, /window\.localStorage\.setItem\(STORAGE_KEY, state\.execution_id\)/);
  assert.match(panel, /window\.localStorage\.getItem\(STORAGE_KEY\)/);
  assert.match(panel, /window\.setInterval\(/);
});

test("candidate registration stays behind preview and a human review checkbox", () => {
  assert.match(panel, /\/preview\?limit=5/);
  assert.match(review, /我已审核预览内容/);
  assert.match(review, /disabled=\{busy \|\| !reviewed\}/);
  assert.match(panel, /confirm_reviewed: true/);
  assert.match(panel, /正确性未独立验证/);
});

test("ordinary and targeted SFT both use the shared human review and registration flow in the data workbench", () => {
  assert.match(workbench, /<SFTGenerationPanel/);
  assert.match(workbench, /<TargetedExperiments generationOnly/);
  assert.match(panel, /<GenerationReviewPanel/);
  assert.match(targeted, /<GenerationReviewPanel/);
  assert.match(targeted, /setSftReviewed\(false\)/);
  assert.match(panel, /转到定向补数/);
  assert.match(targeted, /定向来源上下文/);
  assert.match(targeted, /selectedScenario\?\.eval_id/);
  assert.match(targeted, /selectedScenario\?\.model_id/);
});

test("SFT generator offers seeded and source-free prompt-only modes", () => {
  assert.match(panel, /参考已有数据集/);
  assert.match(panel, /按主题生成（无需种子数据）/);
  assert.match(panel, /topic, skill, difficulty, constraints, target_count/);
  assert.match(panel, /此模式不依赖种子数据/);
  assert.match(panel, /mode === "seeded" \? /);
});

test("activity entry restores generation references and exposes active launcher runs without restarting them", () => {
  assert.match(panel, /DRAFT_KEY/);
  assert.match(panel, /不会丢弃已启动任务/);
  assert.doesNotMatch(panel, /setJob\(null\)/);
  assert.match(activity, /localStorage\.getItem\(GENERATION_KEY\)/);
  assert.match(activity, /sessionStorage\.getItem\(TARGETED_GENERATION_KEY\)/);
  assert.match(activity, /\/api\/launcher\/run/);
  assert.match(activity, /查看任务/);
  assert.match(activity, /查看 Run/);
});

test("analysis hosts executed policy reports while rule editing stays in data and retains report identity", () => {
  assert.match(policyEditor, /reportsOnly/);
  assert.match(policyEditor, /policy-executed-reports/);
  assert.match(policyEditor, /onViewEffects/);
  assert.match(policyEditor, /report_id === requestedReportId/);
  assert.match(policyEditor, /policy-report.*report_id|key=\{report\.report_id\}/s);
  assert.match(policyEditor, /返回关联规则草稿/);
  assert.match(policyEditor, /不会重新登记或执行/);
});
