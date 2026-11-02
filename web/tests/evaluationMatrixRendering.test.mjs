import assert from "node:assert/strict";
import test from "node:test";
import { createElement } from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { createServer } from "vite";
import { readFileSync } from "node:fs";

const vite = await createServer({ server: { middlewareMode: true }, appType: "custom" });
const { EvaluationMatrixResults } = await vite.ssrLoadModule("/src/EvaluationMatrixResults.tsx");
const { MatrixLoadingState, MatrixErrorState, MatrixEmptyState } = await vite.ssrLoadModule("/src/EvaluationMatrixPage.tsx");
test.after(() => vite.close());

function matrix(overrides = {}) {
  return {
    matrix_id: "matrix-test", name: "mini-eval", protocol: "matrix-v1", baseline_model_id: "M2-SFT",
    expected_metrics: ["gsm8k", "domain_ppl"], requested_limits: {}, truncated_models: false,
    comparability: { status: "comparable", expected_protocol_version: "eval-v1", protocol_versions: ["eval-v1"],
      config_hashes: [], dataset_id_sets: [], protocol_version_consistent: true, config_hash_consistent: true,
      dataset_ids_consistent: true, limitations: ["仅 2 个样本"] },
    models: [{ model_id: "M2-SFT", status: "completed", metrics: {
      gsm8k: { value: 0, unit: "ratio", status: "available", requested: 2, observed: 2, delta: 0 },
      domain_ppl: { value: 5.2, unit: "ppl", status: "available", requested: 5, observed: 4, delta: 0 },
    }, samples: {}, provenance: { status: "verified", issues: [] }, lineage: { eval_id: "eval-1", evaluation_run_id: "run-1", model_artifact_id: "model-1",
      dataset_ids: ["dataset-1"], protocol_version: "eval-v1", config_sha256: "abcdef123456" }, errors_count: 0 }],
    ...overrides,
  };
}

test("complete matrix renders baseline, raw values, denominators, deltas and limitations", () => {
  const html = renderToStaticMarkup(createElement(EvaluationMatrixResults, { matrix: matrix() }));
  assert.match(html, /M2-SFT/);
  assert.match(html, /0 <small>ratio/);
  assert.match(html, /n 2 \/ 2/);
  assert.match(html, /仅 2 个样本/);
  assert.match(html, /评测 Run/);
  assert.match(html, /来源已核验/);
});

test("partial and incomparable matrices keep missing values explicit and do not claim a winner", () => {
  const value = matrix({
    comparability: { status: "not_comparable", expected_protocol_version: null, protocol_versions: [], config_hashes: [],
      dataset_id_sets: [], protocol_version_consistent: false, config_hash_consistent: null,
      dataset_ids_consistent: false, limitations: [] },
    models: [{ model_id: "M3-DPO", status: "partial", metrics: {
      gsm8k: { value: null, unit: "ratio", status: "unavailable", requested: 2, observed: 1, delta: null },
      domain_ppl: { value: 5.4, unit: "ppl", status: "partial", requested: 5, observed: 3, delta: 0.2 },
    }, samples: {}, provenance: { status: "mismatch", issues: [{ code: "run_id_mismatch", message: "评测来源不一致" }] },
      lineage: { eval_id: null, evaluation_run_id: null, model_artifact_id: null, dataset_ids: [], protocol_version: null, config_sha256: null }, errors_count: 1 }],
  });
  const html = renderToStaticMarkup(createElement(EvaluationMatrixResults, { matrix: value }));
  assert.match(html, /不可直接比较/);
  assert.match(html, /M3-DPO/);
  assert.match(html, /—/);
  assert.match(html, /n 1 \/ 2/);
  assert.match(html, /Δ —/);
  assert.match(html, /未解析/);
  assert.match(html, /来源不匹配/);
  assert.match(html, /评测来源不一致/);
  assert.doesNotMatch(html, /赢家|更优|提升/);
});

test("loading, empty, and failure states are distinct and failure exposes a keyboard-operable retry", () => {
  const loading = renderToStaticMarkup(createElement(MatrixLoadingState));
  const empty = renderToStaticMarkup(createElement(MatrixEmptyState, { onBackToData() {} }));
  const failed = renderToStaticMarkup(createElement(MatrixErrorState, { message: "本地 API 不可用", onRetry() {} }));
  assert.match(loading, /role="status"/);
  assert.match(empty, /尚无评测矩阵/);
  assert.match(empty, /返回数据页/);
  assert.match(failed, /role="alert"/);
  assert.match(failed, /重试/);
  const css = readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
  assert.match(css, /button:focus-visible/);
  assert.match(css, /\.comparison-page button:focus-visible/);
});

test("narrow comparison layout confines horizontal scrolling to the table region", () => {
  const css = readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
  const chineseSpec = readFileSync(new URL("../../docs/design/frontend/PAGE_SPEC_zh.md", import.meta.url), "utf8");
  assert.match(css, /\.comparison-page\s*\{\s*min-width:\s*0/);
  assert.match(css, /\.comparison-table-wrap\s*\{[^}]*max-width:\s*100%[^}]*overflow-x:\s*auto/s);
  assert.match(css, /@media\s*\(max-width:\s*760px\)\s*\{[^}]*\.comparison-page-content/s);
  assert.match(chineseSpec, /当前 Mini 只读切片/);
});
