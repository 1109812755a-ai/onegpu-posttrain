import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const runs = await readFile(new URL("../src/Runs.tsx", import.meta.url), "utf8");
const styles = await readFile(new URL("../src/style.css", import.meta.url), "utf8");
const main = await readFile(new URL("../src/main.tsx", import.meta.url), "utf8");
const workspaceNavigation = await readFile(new URL("../src/WorkspaceNavigation.tsx", import.meta.url), "utf8");

test("Runs page persists filters and selection in the URL and responds to browser history", () => {
  assert.match(runs, /params\.set\("run_stage"/);
  assert.match(runs, /params\.set\("run_status"/);
  assert.match(runs, /params\.set\("run_q"/);
  assert.match(runs, /params\.set\("run_offset"/);
  assert.match(runs, /window\.history\.pushState/);
  assert.match(runs, /window\.addEventListener\("popstate"/);
  assert.match(runs, /上一页/);
  assert.match(runs, /下一页/);
  assert.match(runs, /\/api\/runs\?/);
  assert.match(runs, /\/api\/runs\/\$\{encodeURIComponent\(filters\.run\)\}/);
});

test("a Run detail opened from any entry point refreshes the canonical record while active", () => {
  assert.match(runs, /const liveDetail = Boolean\(detail && \([\s\S]*detail\.evaluations\.some/);
  assert.match(runs, /setInterval\(\(\) => \{[\s\S]*\/api\/runs\/\$\{encodeURIComponent\(filters\.run\)\}[\s\S]*1500/);
  assert.match(runs, /运行状态、日志和评测结果每 1\.5 秒自动更新/);
  assert.match(main, /const openRunDetail = \(runId: string\)[\s\S]*new URLSearchParams\(\{ page: "runs", run: runId \}\)[\s\S]*setPage\("runs"\)/);
});

test("any completed stage Run can preflight and start an independent profile evaluation", () => {
  assert.match(runs, /detail\.run_kind === "training" && detail\.status === "completed"/);
  assert.match(runs, /\/api\/training\/stage-profiles/);
  assert.match(runs, /\/api\/training\/evaluation\/preflight/);
  assert.match(runs, /\/api\/training\/evaluation\/start/);
  assert.match(runs, /确认并启动独立评测 Run/);
  assert.match(runs, /setInterval\(\(\) => \{[\s\S]*\/api\/runs\/\$\{encodeURIComponent\(filters\.run\)\}/);
  assert.match(runs, /父训练 Run：[\s\S]*onClick=\{\(\) => openRun\(detail\.parent_run_id!/);
});

test("evaluation cards render unavailable status without fabricating a metric value", () => {
  assert.match(runs, /value: number \| null/);
  assert.match(runs, /metric\.value == null \? metricStatusLabel/);
  assert.match(runs, /metric\.denominator_unit/);
});

test("Run lineage detail exposes every input/output ID and its recorded digest", () => {
  assert.match(runs, /Run 输入（每个数据集\/父模型保留独立身份）/);
  assert.match(runs, /detail\.provenance\.map/);
  assert.match(runs, /item\.digest\?\.status === "available" \? item\.digest\.value/);
  assert.match(runs, /Run 产物/);
  assert.match(runs, /detail\.produced_artifacts\.map/);
  assert.match(runs, /lineage_status/);
});

test("a completed Run offers only lineage- and digest-matched downstream model handoffs", () => {
  assert.match(runs, /detail\.status === "completed"/);
  assert.match(runs, /detail\?\.lineage\?\.\["lineage_status"\] === "complete"/);
  assert.match(runs, /registered\.source_run_id !== detail\.run_id/);
  assert.match(runs, /catalogEntry\.digest\.value !== registered\.digest\.value/);
  assert.match(runs, /detail\.handoff_models \|\| \[\]/);
  assert.match(runs, /试用此模型/);
  assert.match(runs, /parent_model_rule === "registered_sft_merged"/);
  assert.match(runs, /profile\.stage, profile_id: profile\.profile_id,[\s\S]*parent_model_id: model\.model_id, source_run_id: detail\.run_id/);
  assert.match(runs, /用作 \{stageLabel\[profile\.stage\]/);
  assert.match(runs, /不会自动选数据/);
  assert.match(runs, /handoffBlocker/);
  assert.doesNotMatch(runs, /handoffToNextStage[\s\S]{0,600}\/api\/training\/preflight/);
});

test("Runs controls and records are keyboard accessible with explicit loading, empty and failure states", () => {
  assert.match(runs, /aria-label="按阶段筛选"/);
  assert.match(runs, /aria-label="按状态筛选"/);
  assert.match(runs, /aria-label="搜索 run_id 或实验标签"/);
  assert.match(runs, /aria-label=\{`查看运行 \$\{run\.run_id\}`\}/);
  assert.match(runs, /role="alert"/);
  assert.match(runs, /没有匹配的运行记录/);
  assert.match(runs, /正在扫描本地运行记录/);
  assert.match(runs, /aria-label="选择阶段评测方案"/);
  assert.match(runs, /aria-label="选择登记评测数据集"/);
  assert.match(runs, /role=\{evaluationPreflight\.ok \? "status" : "alert"\}/);
});

test("Runs stays within the viewport at narrow widths and keeps tables scroll-local", () => {
  assert.match(styles, /\.runs-table-wrap\s*\{[^}]*overflow-x:\s*auto/s);
  assert.match(styles, /@media\s*\(max-width:\s*390px\)/);
  assert.match(styles, /\.runs-page\s*:\s*focus-visible/);
  assert.match(styles, /@media\s*\(max-width:\s*390px\)[\s\S]*runs-evaluation-workbench/);
  assert.match(styles, /\.runs-evaluation-controls label\s*\{[^}]*flex-basis:\s*100%/s);
  assert.match(main, /page === "runs" && <Runs/);
  assert.match(workspaceNavigation, /id: "runs", name: "实验 Runs", icon: "runs", module: "runs"/);
});
