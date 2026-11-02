import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";
import { datasetPreviewReducer, fetchDatasetPreview } from "../src/datasetPreview.mjs";

const sample = { id: "s1", fields: { prompt: "2+2" }, truncated_fields: [] };
const preview = {
  available: true, dataset_id: "ds/sft", stage: "sft", canonical_sha256: "abc", integrity: "verified",
  total_records: 1, returned_records: 1, truncated: false, samples: [sample],
};

test("preview request is lazy-callable and URL-encodes dataset IDs", async () => {
  let requested;
  const result = await fetchDatasetPreview("ds/sft ?", { fetchImpl: async (url, options) => {
    requested = { url, options };
    return { ok: true, json: async () => preview };
  } });
  assert.equal(requested.url, "/api/data/datasets/ds%2Fsft%20%3F/preview?limit=3");
  assert.equal(requested.options.headers.Accept, "application/json");
  assert.equal(result.samples.length, 1);
});

test("empty successful result can be represented without fabricating a sample", () => {
  const empty = { ...preview, total_records: 0, returned_records: 0, samples: [] };
  const state = datasetPreviewReducer({}, { type: "success", datasetId: "empty", data: empty });
  assert.equal(state.empty.status, "success");
  assert.deepEqual(state.empty.data.samples, []);
});

test("unpreviewable response preserves its readable reason for inline display and retry", async () => {
  await assert.rejects(fetchDatasetPreview("legacy", { fetchImpl: async () => ({
    ok: false, status: 409, json: async () => ({ detail: { code: "evidence_missing", message: "旧数据缺少 canonical 证据" } }),
  }) }), /旧数据缺少 canonical 证据/);
  const failed = datasetPreviewReducer({}, { type: "error", datasetId: "legacy", message: "旧数据缺少 canonical 证据" });
  assert.equal(failed.legacy.status, "error");
  const retrying = datasetPreviewReducer(failed, { type: "loading", datasetId: "legacy" });
  assert.equal(retrying.legacy.status, "loading");
  const recovered = datasetPreviewReducer(retrying, { type: "success", datasetId: "legacy", data: preview });
  assert.equal(recovered.legacy.data.integrity, "verified");
});

test("close removes only the chosen card preview and leaves other cards unchanged", () => {
  const state = { a: { status: "success", data: preview }, b: { status: "error", message: "unavailable" } };
  const next = datasetPreviewReducer(state, { type: "close", datasetId: "a" });
  assert.equal(next.a, undefined);
  assert.equal(next.b, state.b);
  assert.notEqual(next, state);
});

test("preview controls use native keyboard-focusable buttons and expose panel state", async () => {
  const source = await readFile(new URL("../src/DatasetWorkbench.tsx", import.meta.url), "utf8");
  assert.match(source, /<button className="dataset-preview-toggle" aria-expanded=\{Boolean\(preview\)\} aria-controls=/);
  assert.match(source, /<button className="dataset-preview-close" aria-label="关闭样本预览"/);
  assert.match(source, /className="dataset-preview-error" role="alert"/);
  assert.match(source, /className="dataset-preview-message" role="status"/);
});

test("preview integrity message is distinct from quality and license metadata", async () => {
  const source = await readFile(new URL("../src/DatasetWorkbench.tsx", import.meta.url), "utf8");
  assert.match(source, /canonical SHA256 已核验/);
  assert.match(source, /不代表语义质量通过、许可已核验/);
  assert.match(source, /quality-chip quality-/);
  assert.match(source, /许可状态/);
});
